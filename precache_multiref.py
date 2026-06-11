"""Precompute VAE latents + text features for a multi-reference editing dataset.

Dataset-agnostic layout: ``<data_root>/train/meta.jsonl``, one row per sample::

    {"idx": N, "<instr_field>": "instruction",
     "refs": ["train/000000_ref0.jpg", "train/000000_ref1.jpg", ...],
     "target": "train/000000_tgt.jpg"}

(``refs``/``target`` paths are relative to ``data_root``; any N>=1 references allowed.)
Writes one ``.pt`` per sample with the encoded references (each on its own MRoPE frame
downstream), target, and instruction features, so ``train_multiref.py`` runs encoder-free.

  python precache_multiref.py --config config/multiref.yaml
  # parallel:
  python precache_multiref.py --config config/multiref.yaml --num-shards 4 --shard 0
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
  parser = argparse.ArgumentParser(description="Precache multi-reference latents + text features.")
  parser.add_argument("--config", default="config/multiref.yaml")
  parser.add_argument("--num-shards", type=int, default=1,
                      help="Split the dataset across N parallel processes (one GPU each).")
  parser.add_argument("--shard", type=int, default=0,
                      help="This process's shard id in [0, num_shards).")
  args = parser.parse_args()
  num_shards = max(1, int(args.num_shards))
  shard_id = int(args.shard) % num_shards

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config)
  apply_runtime(cfg)

  data = cfg.paths.data_root
  cache = cfg.paths.cache_dir
  res = int(cfg.data.resolution)
  instr_field = cfg.data.instr_field
  limit = int(cfg.data.limit)

  t0 = time.time()
  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=cfg.paths.weights),
    device=cfg.runtime.device, dtype=dtype_of(cfg),
  )
  print(f"[precache-mref] pipeline loaded in {time.time()-t0:.1f}s", flush=True)

  patch = pipe.config.patch_size * pipe.config.ae_scale_factor
  grid = res // patch
  ps = pipe.config.patch_size

  meta_path = f"{data}/train/meta.jsonl"
  if not os.path.exists(meta_path):
    raise FileNotFoundError(f"no meta at {meta_path}")
  meta = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
  if limit:
    meta = meta[:limit]
  os.makedirs(f"{cache}/train", exist_ok=True)

  def _encode(relpath):
    img = Image.open(f"{data}/{relpath}").convert("RGB")
    ten = train_edit.images_to_tensor([img], res, res, pipe.device)
    return train_edit.encode_image_tokens(pipe, ten, patch_size=ps)[0].to(torch.bfloat16).cpu()

  t0 = time.time()
  done = 0
  for i, m in enumerate(meta):
    if num_shards > 1 and (i % num_shards) != shard_id:
      continue
    idx = m["idx"]
    out_path = f"{cache}/train/{idx:06d}.pt"
    if os.path.exists(out_path):
      continue
    refs = m["refs"]
    target = m["target"]
    paths = [f"{data}/{p}" for p in refs] + [f"{data}/{target}"]
    if not all(os.path.exists(p) for p in paths):
      continue  # dataset still materializing / missing files
    z_refs = [_encode(p) for p in refs]
    z_tgt = _encode(target)
    inputs = train_edit.build_edit_inputs(pipe, [m[instr_field]], grid, grid)
    llm = pipe._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
    llm_text = llm[0][inputs["indicator"][0] == LLM_TOKEN_INDICATOR].to(torch.bfloat16).cpu()
    torch.save(
      {
        "z_refs": z_refs,
        "ref_grids": [(grid, grid)] * len(z_refs),
        "z_tgt": z_tgt,
        "llm_text": llm_text,
        "target_grid": (grid, grid),
        "idx": idx,
        "instruction": m[instr_field],
      },
      out_path,
    )
    done += 1
    if done % 100 == 0:
      print(f"[precache-mref] {done} written ({(time.time()-t0)/done:.2f}s/sample)", flush=True)
  print(f"[precache-mref] DONE shard {shard_id}/{num_shards}: {done} new caches", flush=True)


if __name__ == "__main__":
  main()
