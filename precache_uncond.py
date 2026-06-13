"""Precache DEGRADED-image latents for negative-model (uncond) LoRA training.

The unconditional transformer is image-only (no text frame), so this cache needs the
VAE encoder ONLY -- no Qwen text encoder, a few GB of VRAM. Each source image gets K
random *content-preserving* degradations (blur / JPEG / downscale-upscale / noise,
composed 1-3 at a time): the degraded copies share the natural images' content
distribution, so the only direction a LoRA trained on them can absorb is the quality
axis. Hook that LoRA into the uncond at inference and CFG steers away from it.

  CUDA_VISIBLE_DEVICES=0 python precache_uncond.py --config config/uncond_quality.yaml
  # shard across GPUs: --num-shards 4 --shard 0..3
"""
import argparse
import io
import os
import random

import torch
from PIL import Image, ImageFilter

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def degrade_image(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
  """Apply 1-3 random content-preserving degradations; returns (image, recipe)."""
  import numpy as np

  # Ranges start MILD: the knob mostly fights near-manifold softness, not wreckage.
  # Call this on the image already resized to training resolution -- the magnitudes
  # are in training-res pixels (applied at native size they'd be erased by downsample).
  ops = rng.sample(("blur", "jpeg", "downup", "noise"), k=rng.randint(1, 3))
  applied = []
  for op in ops:
    if op == "blur":
      r = rng.uniform(0.7, 4.0)
      img = img.filter(ImageFilter.GaussianBlur(radius=r))
      applied.append(f"blur{r:.1f}")
    elif op == "jpeg":
      q = rng.randint(5, 45)
      buf = io.BytesIO()
      img.save(buf, format="JPEG", quality=q)
      buf.seek(0)
      img = Image.open(buf).convert("RGB")
      applied.append(f"jpeg{q}")
    elif op == "downup":
      f = rng.uniform(0.25, 0.7)
      w, h = img.size
      interp = rng.choice((Image.BILINEAR, Image.NEAREST))
      img = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
      img = img.resize((w, h), interp)
      applied.append(f"downup{f:.2f}")
    elif op == "noise":
      sigma = rng.uniform(4.0, 25.0)
      arr = np.asarray(img).astype(np.float32)
      arr = arr + np.random.default_rng(rng.randrange(2**31)).normal(0.0, sigma, arr.shape)
      img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
      applied.append(f"noise{sigma:.0f}")
  return img, "+".join(applied)


def _load_vae(weights_repo: str, device, dtype):
  """VAE encoder + latent-norm constants, nothing else (no 8B text encoder)."""
  from ideogram4.pipeline_ideogram4 import (
    Ideogram4PipelineConfig, _load_autoencoder, hf_hub_download,
  )
  from ideogram4.latent_norm import get_latent_norm

  pcfg = Ideogram4PipelineConfig(weights_repo=weights_repo)
  ae = _load_autoencoder(
    hf_hub_download(repo_id=pcfg.weights_repo, filename=pcfg.autoencoder_filename),
    device, dtype)
  shift, scale = get_latent_norm()
  return ae, shift.to(device), scale.to(device), pcfg


@torch.no_grad()
def _encode(ae, shift, scale, ps, img: Image.Image, res: int, device, dtype) -> torch.Tensor:
  """RGB PIL -> normalized patch tokens (n, 128), mirroring train_edit.encode_image_tokens."""
  from ideogram4.train_edit import images_to_tensor, patchify_latent

  ten = images_to_tensor([img], res, res, device)
  moments = ae.encoder(ten.to(dtype))
  mean = moments[:, : moments.shape[1] // 2]
  tokens = patchify_latent(mean, ps).to(torch.float32)
  return ((tokens - shift) / scale)[0]


def main():
  ap = argparse.ArgumentParser(description="Precache degraded-image latents (uncond LoRA).")
  ap.add_argument("--config", default="config/uncond_quality.yaml")
  ap.add_argument("--variants", type=int, default=2, help="degraded variants per image")
  ap.add_argument("--num-shards", type=int, default=1)
  ap.add_argument("--shard", type=int, default=0)
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)
  res = int(cfg.data.resolution)
  cache_dir = cfg.paths.cache_dir
  os.makedirs(cache_dir, exist_ok=True)

  ae, shift, scale, pcfg = _load_vae(cfg.paths.weights, device, dtype)
  patch = pcfg.patch_size * pcfg.ae_scale_factor
  ps = pcfg.patch_size
  if res % patch != 0:
    raise ValueError(f"resolution {res} not divisible by patch {patch}")
  grid = res // patch
  print(f"[uncond-pre] VAE loaded | res {res} grid {grid}x{grid} "
        f"| variants {args.variants} | shard {args.shard}/{args.num_shards}", flush=True)

  paths = []
  for root, _, fnames in os.walk(cfg.paths.data_root):
    for fn in fnames:
      if fn.lower().endswith(_IMG_EXTS):
        paths.append(os.path.join(root, fn))
  paths.sort()
  if not paths:
    raise FileNotFoundError(f"no images under paths.data_root={cfg.paths.data_root!r}")
  print(f"[uncond-pre] {len(paths)} source images", flush=True)

  done = 0
  for idx, p in enumerate(paths):
    if idx % args.num_shards != args.shard:
      continue
    try:
      # Resize FIRST so degradations act in training-res pixels (and clean copy ==
      # exactly the image the variants derive from); the later encode resize is a no-op.
      img = Image.open(p).convert("RGB").resize((res, res))
    except Exception as exc:
      print(f"[uncond-pre] skip {p}: {exc}", flush=True)
      continue

    def _atomic_save(obj, out):
      torch.save(obj, out + ".tmp")  # never leave a truncated .pt behind a crash:
      os.replace(out + ".tmp", out)  # exists()-skip resume would treat it as done

    # Clean copy (suffix _c) for the no-op-on-clean anchor term, then K degraded variants.
    out_clean = os.path.join(cache_dir, f"{idx:06d}_c.pt")
    if not os.path.exists(out_clean):
      z = _encode(ae, shift, scale, ps, img, res, device, dtype)
      _atomic_save({"z": z.to(torch.float16).cpu(), "grid_h": grid, "grid_w": grid,
                    "deg": "clean", "src": os.path.relpath(p, cfg.paths.data_root)}, out_clean)
      done += 1
    for v in range(args.variants):
      out = os.path.join(cache_dir, f"{idx:06d}_{v}.pt")
      if os.path.exists(out):
        continue
      rng = random.Random(hash((idx, v)) & 0x7FFFFFFF)  # deterministic per (image, variant)
      deg, recipe = degrade_image(img, rng)
      z = _encode(ae, shift, scale, ps, deg, res, device, dtype)
      _atomic_save({"z": z.to(torch.float16).cpu(), "grid_h": grid, "grid_w": grid,
                    "deg": recipe, "src": os.path.relpath(p, cfg.paths.data_root)}, out)
      done += 1
      if done % 100 == 0:
        print(f"[uncond-pre] {done} cached (img {idx}/{len(paths)})", flush=True)
  print(f"[uncond-pre] DONE: {done} new files -> {cache_dir}", flush=True)


if __name__ == "__main__":
  main()
