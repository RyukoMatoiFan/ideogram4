"""Precompute VAE latents + text-encoder features for an edit dataset.

Loads the FULL pipeline (text encoder + VAE) once and writes, per pair, a .pt with
{z_ref, z_tgt, llm_text, grid_h, grid_w, idx, edit}. Training then runs encoder-free
(see train_edit_lora_cached.py): no Qwen3-VL / VAE in memory, no slow reload.

Config-driven so it serves both layouts:
  HQ-Edit : data_root/<split>/meta.jsonl, NNNNNN_src.png, instr_field "edit"
  pico    : data_root/meta.jsonl,         NNNNNN_src.jpg, instr_field "text"
Aspect-preserving pico images are squashed to RES^2 by images_to_tensor (plain
resize); src+tgt share the scene's aspect so the squash is identical for both,
preserving src<->tgt pixel correspondence.

  python precache_edit.py --config config/precache_edit.yaml
  # pico layout:
  IG4_DATA__IMG_EXT=jpg IG4_DATA__INSTR_FIELD=text IG4_DATA__META_AT_ROOT=1 \
    python precache_edit.py --config config/precache_edit.yaml
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4 import train_edit
from ideogram4.constants import LLM_TOKEN_INDICATOR


@torch.no_grad()
def main():
  parser = argparse.ArgumentParser(description="Precache edit VAE latents + text features.")
  parser.add_argument("--config", default="config/precache_edit.yaml",
                      help="Path to YAML config (default: config/precache_edit.yaml)")
  parser.add_argument("--num-shards", type=int, default=1,
                      help="Split the dataset across N parallel precache processes (one GPU each).")
  parser.add_argument("--shard", type=int, default=0,
                      help="This process's shard id in [0, num_shards); handles rows where i %% num_shards == shard.")
  args = parser.parse_args()
  num_shards = max(1, int(args.num_shards))
  shard_id = int(args.shard) % num_shards

  from ideogram4.training_config import load_config, apply_runtime, dtype_of

  cfg = load_config(args.config)
  apply_runtime(cfg)

  data = cfg.paths.data_root
  cache = cfg.paths.cache_dir
  res = int(cfg.data.resolution)
  img_ext = (cfg.data.img_ext or "png").lstrip(".")
  instr_field = cfg.data.instr_field
  meta_at_root = bool(cfg.data.meta_at_root)
  limit = int(cfg.data.limit)

  t0 = time.time()
  # Encoders-only pipeline (text encoder + VAE, NO transformers): precache only encodes,
  # so this keeps VRAM ~11GB instead of ~27GB (<=24GB portable).
  pipe = train_edit.load_encoders_pipeline(cfg.paths.weights, cfg.runtime.device, dtype_of(cfg))
  print(f"[precache] encoders loaded in {time.time()-t0:.1f}s (no transformers)", flush=True)

  patch = pipe.config.patch_size * pipe.config.ae_scale_factor
  grid = res // patch
  ps = pipe.config.patch_size

  aspect_bucketing = bool(cfg.data.aspect_bucketing)
  buckets = None
  if aspect_bucketing:
    from ideogram4.training_utils import aspect_buckets, nearest_bucket
    buckets = aspect_buckets(int(cfg.data.bucket_pixels) or res * res,
                             divisor=patch, num=int(cfg.data.num_buckets))
    print(f"[precache] aspect bucketing ON: {len(buckets)} buckets {buckets}", flush=True)

  for split in ("train",):  # eval held out from train downstream; eval/ meta is empty
    meta_path = f"{data}/meta.jsonl" if meta_at_root else f"{data}/{split}/meta.jsonl"
    if not os.path.exists(meta_path):
      continue
    meta = [json.loads(l) for l in open(meta_path)]
    if limit:
      meta = meta[:limit]
    os.makedirs(f"{cache}/{split}", exist_ok=True)
    t0 = time.time()
    for i, m in enumerate(meta):
      if num_shards > 1 and (i % num_shards) != shard_id:
        continue  # another shard process owns this row
      idx, edit = m["idx"], m[instr_field]
      out_path = f"{cache}/{split}/{idx:06d}.pt"
      if os.path.exists(out_path):
        continue
      sp = f"{data}/{split}/{idx:06d}_src.{img_ext}"
      tp = f"{data}/{split}/{idx:06d}_tgt.{img_ext}"
      if not (os.path.exists(sp) and os.path.exists(tp)):
        continue  # download may be mid-flight / pair failed
      src = Image.open(sp).convert("RGB")
      tgt = Image.open(tp).convert("RGB")
      # Bucket by the SOURCE aspect (src+tgt share the scene -> same bucket).
      if aspect_bucketing:
        H, W = nearest_bucket(src.width, src.height, buckets)
      else:
        H = W = res
      gh, gw = H // patch, W // patch
      s = train_edit.images_to_tensor([src], H, W, pipe.device)
      t = train_edit.images_to_tensor([tgt], H, W, pipe.device)
      z_ref = train_edit.encode_image_tokens(pipe, s, patch_size=ps)[0]
      z_tgt = train_edit.encode_image_tokens(pipe, t, patch_size=ps)[0]
      inputs = train_edit.build_edit_inputs(pipe, [edit], gh, gw)
      llm = pipe._encode_text(
        inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
      )
      mask = inputs["indicator"][0] == LLM_TOKEN_INDICATOR
      llm_text = llm[0][mask]  # (num_text, llm_dim)

      def _atomic_save(obj, out):
        torch.save(obj, out + ".tmp")  # never leave a truncated .pt behind a crash:
        os.replace(out + ".tmp", out)  # exists()-skip resume would treat it as done

      _atomic_save(
        {
          "z_ref": z_ref.to(torch.bfloat16).cpu(),
          "z_tgt": z_tgt.to(torch.bfloat16).cpu(),
          "llm_text": llm_text.to(torch.bfloat16).cpu(),
          "grid_h": gh, "grid_w": gw, "idx": idx, "edit": edit,
        },
        out_path,
      )
      if (i + 1) % 100 == 0:
        print(f"[precache] {split} {i+1}/{len(meta)} ({(time.time()-t0)/(i+1):.2f}s/pair)", flush=True)
    print(f"[precache] {split} done: {len(meta)} pairs", flush=True)
  print("[precache] DONE", flush=True)


if __name__ == "__main__":
  main()
