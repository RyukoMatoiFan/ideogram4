"""Low-VRAM held-out eval for the cached EDIT trainer (source|generated|target).

Counterpart to byg_eval_lowvram.py but for paired edit caches (precache_edit.py),
whose .pt files carry z_ref / z_tgt / llm_text. Loads ONLY the conditional
transformer (+ VAE decoder), applies a trained adapter, and for each held-out
sample (idx < data.n_eval_holdout) generates the edit from the cached reference
latent + instruction features -- no text encoder, peak VRAM ~14-19GB (<=24GB).

Saves a 3-panel grid: decoded source (z_ref) | generated edit | decoded target
(z_tgt), so instruction-following AND identity/background preservation are both
visible against the ground-truth target. Prints peak VRAM.

  CUDA_VISIBLE_DEVICES=0 python edit_eval_lowvram.py --config config/edit_lora_cached.yaml \
    --ckpt runs/pico-edit-r128/ckpts/<adapter>.safetensors --guidance 2.0 --steps 24

If --ckpt is omitted, picks the most recent *.safetensors under {paths.ckpt_dir}.
"""
import argparse
import glob
import os

import torch
from PIL import Image


def _load_lora_weights(wrapped, path, device, dtype):
  from safetensors.torch import load_file
  state = load_file(path)
  with torch.no_grad():
    for sub, m in wrapped.items():
      m.lora_A.copy_(state[f"diffusion_model.{sub}.lora_A.weight"].to(device, dtype))
      m.lora_B.copy_(state[f"diffusion_model.{sub}.lora_B.weight"].to(device, dtype))


def main():
  parser = argparse.ArgumentParser(description="Low-VRAM held-out edit eval (triplet grids).")
  parser.add_argument("--config", default="config/edit_lora_cached.yaml")
  parser.add_argument("--ckpt", default="", help="adapter path; default = latest in ckpt_dir")
  parser.add_argument("--guidance", type=float, default=2.0, help="instruction CFG scale")
  parser.add_argument("--steps", type=int, default=24, help="Euler steps")
  parser.add_argument("--base", action="store_true",
                      help="CONTROL: run the frozen base (LoRA left at zero-init = identity), "
                           "no adapter loaded. Shows whether the base alone can edit.")
  args = parser.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  from ideogram4.pipeline_ideogram4 import (
    Ideogram4PipelineConfig, _build_transformer, _load_indexed_or_single_state_dict,
  )
  from ideogram4.modeling_ideogram4 import Ideogram4Config
  from ideogram4 import lora as loramod
  from ideogram4.edit_sampler import sample_edit_cached, decode_latents, load_decoder
  from ideogram4.scheduler import get_schedule_for_resolution

  cfg = load_config(args.config)
  apply_runtime(cfg)

  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)
  rank = int(cfg.lora.rank)
  n_eval = int(cfg.data.n_eval_holdout)
  cache_dir = cfg.paths.cache_dir or os.path.join(cfg.paths.data_root, "cache", "train")
  tag = "eval_base" if args.base else "eval_lowvram"
  out_dir = os.path.join(cfg.paths.output_dir, tag)
  os.makedirs(out_dir, exist_ok=True)

  adapter = ""
  if not args.base:
    adapter = args.ckpt
    if not adapter:
      cands = sorted(glob.glob(os.path.join(cfg.paths.ckpt_dir, "*.safetensors")),
                     key=os.path.getmtime)
      if not cands:
        raise FileNotFoundError(f"no *.safetensors under {cfg.paths.ckpt_dir}; pass --ckpt.")
      adapter = cands[-1]
  print(f"[edit-eval] {'BASE (no adapter)' if args.base else 'adapter=' + adapter}", flush=True)

  cache_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
  holdout = [f for f in cache_files if int(f[:-3]) < n_eval]
  if not holdout:
    raise RuntimeError(f"no held-out caches (idx < {n_eval}) under {cache_dir}")

  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

  pcfg = Ideogram4PipelineConfig(weights_repo=cfg.paths.weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype)
  del sd
  wrapped = loramod.inject_lora(transformer, rank=rank)
  if not args.base:
    _load_lora_weights(wrapped, adapter, device, dtype)
  # else: LoRA-B stays zero-init -> adapter is identity -> pure frozen base.
  transformer.eval()

  ae, latent_shift, latent_scale, patch_size = load_decoder(cfg.paths.weights, device, dtype)
  dec_kw = dict(patch_size=patch_size, latent_shift=latent_shift,
                latent_scale=latent_scale, dtype=dtype)
  gen = torch.Generator(device=device).manual_seed(0)

  for f in holdout:
    item = torch.load(os.path.join(cache_dir, f), map_location="cpu")
    gh, gw = int(item["grid_h"]), int(item["grid_w"])
    z_ref = item["z_ref"].to(device, torch.float32)
    z_tgt = item["z_tgt"].to(device, torch.float32)
    llm_text = item["llm_text"].to(device, torch.float32)
    H, W = gh * patch_size * pcfg.ae_scale_factor, gw * patch_size * pcfg.ae_scale_factor
    schedule = get_schedule_for_resolution((H, W))

    z = sample_edit_cached(
      transformer, z_ref, llm_text, gh, gw,
      schedule=schedule, num_steps=args.steps,
      guidance_scale=args.guidance, generator=gen,
    )
    src = decode_latents(ae, z_ref.unsqueeze(0), gh, gw, **dec_kw)[0]
    out = decode_latents(ae, z, gh, gw, **dec_kw)[0]
    tgt = decode_latents(ae, z_tgt.unsqueeze(0), gh, gw, **dec_kw)[0]

    canvas = Image.new("RGB", (W * 3, H), (0, 0, 0))
    canvas.paste(src.resize((W, H)), (0, 0))
    canvas.paste(out.resize((W, H)), (W, 0))
    canvas.paste(tgt.resize((W, H)), (W * 2, 0))
    idx = int(f[:-3])
    p = os.path.join(out_dir, f"edit_{idx:06d}.png")
    canvas.save(p)
    print(f"[edit-eval] idx={idx} (src|gen|tgt) -> {p}", flush=True)

  if device.type == "cuda":
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"[edit-eval] PEAK torch VRAM: {peak:.2f} GB "
          f"({'OK <=24GB' if peak <= 24 else 'OVER 24GB'})", flush=True)
  print(f"[edit-eval] grids under {out_dir}", flush=True)


if __name__ == "__main__":
  main()
