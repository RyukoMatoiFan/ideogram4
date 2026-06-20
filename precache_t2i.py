"""Precompute VAE latents + text features for a plain text-to-image dataset (folder layout).

For each ``<name>.png`` + ``<name>.txt`` (caption) in a directory, writes a .pt with
{z_tgt, llm_text, grid_h, grid_w, idx, caption}. No reference frame (this is T2I, not
editing). Training then runs encoder-free (train_t2i_full_cached.py).

Ideogram-4 is native to STRUCTURED JSON captions (bbox schema); plain prose/tags are
OOD, so each caption is wrapped in the JSON schema. Video-prompt cruft (LTX-2 / camera /
sound sentences) is stripped so only the visual description conditions the image.

  python precache_t2i.py --config config/precache_t2i.yaml --img-dir /path/to/anime
"""
import argparse
import glob
import json
import os
import re
import time

import torch
from PIL import Image

from ideogram4 import train_edit
from ideogram4.constants import LLM_TOKEN_INDICATOR

_VIDEO_CRUFT = re.compile(
  r"(video|ltx|i2v|image-to-video|the sound|audio|camera (is|moves|pans|zooms|remains)|"
  r"as the (video|scene|animation) progresses|frame rate|motion|the scene is (calm|still)|"
  r"static and lacks movement|no additional background noise)", re.I)


def clean_caption(text: str) -> str:
  """Drop sentences that describe video/audio/camera, keep the visual description."""
  sents = re.split(r"(?<=[.!?])\s+", text.strip())
  kept = [s for s in sents if not _VIDEO_CRUFT.search(s)]
  out = " ".join(kept).strip()
  return out or text.strip()  # never return empty


def to_json_caption(text: str) -> str:
  """Wrap a danbooru tag string in the verifier-compliant ig4 JSON schema.

  medium signaled natively in HLD; tags live in the element ``desc`` (the schema's home
  for granular attributes); optional fields (bbox / style_description / aspect_ratio) are
  OMITTED rather than constant-placeholder-filled (a constant teaches the model to ignore
  the field). ``type`` must be ``obj`` (not ``subject``). ``background`` is required by the
  schema so it stays as an empty string. MUST match the inference-time wrapper exactly."""
  caption = {
    "high_level_description": "an anime illustration",
    "compositional_deconstruction": {
      "background": "",
      "elements": [{"type": "obj", "desc": text}],
    },
  }
  return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


@torch.no_grad()
def main():
  ap = argparse.ArgumentParser(description="Precache T2I VAE latents + text features.")
  ap.add_argument("--config", default="config/precache_t2i.yaml")
  ap.add_argument("--img-dir", required=True, help="folder of <name>.<ext> + <name>.txt (recursive)")
  ap.add_argument("--img-ext", default="png")
  ap.add_argument("--num-shards", type=int, default=1, help="split across N parallel processes (one GPU each)")
  ap.add_argument("--shard", type=int, default=0, help="this process's shard id in [0, num_shards)")
  ap.add_argument("--raw-caption", action="store_true", help="skip prose cleaning (for tag captions)")
  ap.add_argument("--te-ckpt", default="", help="re-precache with a fine-tuned text encoder "
                  "(decoupled: TE stage -> re-precache -> fast cached DiT full-FT)")
  args = ap.parse_args()
  num_shards = max(1, int(args.num_shards)); shard_id = int(args.shard) % num_shards

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  cache = cfg.paths.cache_dir
  res = int(cfg.data.resolution)
  os.makedirs(cache, exist_ok=True)

  t0 = time.time()
  pipe = train_edit.load_encoders_pipeline(cfg.paths.weights, cfg.runtime.device, dtype_of(cfg))
  print(f"[precache-t2i] encoders loaded in {time.time()-t0:.1f}s", flush=True)
  if args.te_ckpt:  # decoupled: encode features with the fine-tuned (dequantized) text encoder
    from safetensors.torch import load_file
    train_edit.dequantize_fp8_transformer(pipe.text_encoder, dtype=dtype_of(cfg))
    res_keys = pipe.text_encoder.load_state_dict(load_file(args.te_ckpt), strict=False)
    pipe.text_encoder.to(cfg.runtime.device).eval()
    print(f"[precache-t2i] loaded fine-tuned TE {args.te_ckpt} "
          f"({len(res_keys.missing_keys)} missing, {len(res_keys.unexpected_keys)} unexpected)", flush=True)
  patch = pipe.config.patch_size * pipe.config.ae_scale_factor
  ps = pipe.config.patch_size

  # Aspect-ratio bucketing: keep each image at its native AR (no square-squash distortion)
  # by resizing to the nearest of `num_buckets` fixed-area buckets. Per-image (gh, gw) grid.
  aspect_bucketing = bool(cfg.data.aspect_bucketing)
  buckets = None
  if aspect_bucketing:
    from ideogram4.training_utils import aspect_buckets, nearest_bucket
    buckets = aspect_buckets(int(cfg.data.bucket_pixels) or res * res, divisor=patch,
                             num=int(cfg.data.num_buckets))
    print(f"[precache-t2i] aspect bucketing ON: {len(buckets)} buckets {buckets}", flush=True)

  imgs = sorted(glob.glob(os.path.join(args.img_dir, f"**/*.{args.img_ext}"), recursive=True))
  print(f"[precache-t2i] {len(imgs)} images "
        f"{'(bucketed)' if aspect_bucketing else f'@ {res}px'} | shard {shard_id}/{num_shards}", flush=True)

  def _atomic_save(obj, out):
    torch.save(obj, out + ".tmp")  # never leave a truncated .pt behind a crash:
    os.replace(out + ".tmp", out)  # exists()-skip resume would treat it as done

  # Persist a path->idx manifest so re-runs (after deleting/inserting images) reuse the
  # SAME integer idx per image instead of re-deriving it from enumerate(sorted(glob)).
  # That position is BOTH the cache filename ({idx:06d}.pt) and the train/eval holdout key
  # (first n_eval = held out, see train_t2i_full_cached.py int(f[:-3]) < n_eval); without
  # this, removing one image would shift every later position -> overwrite unrelated caches
  # and silently move the eval split. Fresh dir (no manifest) -> sorted order -> idx 0..N,
  # byte-identical to the old enumerate() behaviour, so existing caches/splits are unchanged.
  manifest_path = os.path.join(cache, "manifest.json")
  path2idx = {}
  if os.path.exists(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
      path2idx = {k: int(v) for k, v in json.load(f).items()}
  next_idx = (max(path2idx.values()) + 1) if path2idx else 0
  rels = [os.path.relpath(ip, args.img_dir) for ip in imgs]
  for rel in rels:  # rels already in sorted(glob) order; new paths appended at the tail
    if rel not in path2idx:
      path2idx[rel] = next_idx
      next_idx += 1
  # Only the shard owner rewrites (shard 0); avoids concurrent clobber across processes.
  if shard_id == 0:
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump(path2idx, f, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, manifest_path)

  t0 = time.time()
  for pos, (ip, rel) in enumerate(zip(imgs, rels)):
    idx = path2idx[rel]
    if num_shards > 1 and (idx % num_shards) != shard_id:
      continue  # another shard owns this row (global idx -> no collision)
    out_path = f"{cache}/{idx:06d}.pt"
    if os.path.exists(out_path):
      continue
    cap_path = os.path.splitext(ip)[0] + ".txt"
    raw = open(cap_path, encoding="utf-8", errors="ignore").read() if os.path.exists(cap_path) else ""
    caption = to_json_caption(raw.strip() if args.raw_caption else clean_caption(raw))
    img = Image.open(ip).convert("RGB")
    if aspect_bucketing:
      H, W = nearest_bucket(img.width, img.height, buckets)
    else:
      H = W = res
    gh, gw = H // patch, W // patch
    x = train_edit.images_to_tensor([img], H, W, pipe.device)
    z_tgt = train_edit.encode_image_tokens(pipe, x, patch_size=ps)[0]
    inputs = train_edit.build_edit_inputs(pipe, [caption], gh, gw)
    llm = pipe._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
    mask = inputs["indicator"][0] == LLM_TOKEN_INDICATOR
    llm_text = llm[0][mask]
    _atomic_save({
      "z_tgt": z_tgt.to(torch.bfloat16).cpu(),
      "llm_text": llm_text.to(torch.bfloat16).cpu(),
      "grid_h": gh, "grid_w": gw, "idx": idx, "caption": caption,
    }, out_path)
    if (pos + 1) % 20 == 0:
      print(f"[precache-t2i] {pos+1}/{len(imgs)} ({(time.time()-t0)/(pos+1):.2f}s/img)", flush=True)
  print(f"[precache-t2i] DONE: {len(imgs)} images -> {cache}", flush=True)


if __name__ == "__main__":
  main()
