"""Train a rank-128 in-context edit LoRA on a HQ-Edit subset (QLoRA over fp8).

Mini run: ~10k pairs, 3000 optimizer steps, effective batch 4 (batch 1 x grad-accum
4 to stay within 80GB), cosine LR with warmup, CFG instruction-dropout 0.1.
Checkpoints + held-out eval generations under the output dir.

  CUDA_VISIBLE_DEVICES=0 python train_edit_lora.py --config config/edit_lora.yaml
"""
import argparse
import json
import math
import os
import random
import time

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4 import train_edit, edit_sampler
from ideogram4 import lora as loramod


def main():
  parser = argparse.ArgumentParser(description="Train an in-context edit LoRA (full pipeline).")
  parser.add_argument("--config", default="config/edit_lora.yaml",
                      help="Path to YAML config (default: config/edit_lora.yaml)")
  args = parser.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of

  cfg = load_config(args.config)
  apply_runtime(cfg)

  data = cfg.paths.data_root
  out = cfg.paths.output_dir
  ckpt = cfg.paths.ckpt_dir
  res = int(cfg.data.resolution)
  rank = int(cfg.lora.rank)
  lr = float(cfg.optim.lr)
  steps = int(cfg.optim.steps)
  accum = int(cfg.optim.accum)
  warmup = int(cfg.optim.warmup)
  grad_clip = float(cfg.optim.grad_clip)
  cfg_drop = float(cfg.optim.cfg_dropout_prob)
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  n_eval = int(cfg.data.n_eval_holdout)

  os.makedirs(ckpt, exist_ok=True)

  def load_pair(sub, idx):
    src = Image.open(f"{data}/{sub}/{idx:06d}_src.png").convert("RGB")
    tgt = Image.open(f"{data}/{sub}/{idx:06d}_tgt.png").convert("RGB")
    return src, tgt

  t0 = time.time()
  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=cfg.paths.weights),
    device=cfg.runtime.device, dtype=dtype_of(cfg),
  )
  print(f"[train] pipeline loaded in {time.time()-t0:.1f}s", flush=True)

  transformer = pipe.conditional_transformer
  wrapped = loramod.inject_lora(transformer, rank=rank)
  params = loramod.lora_parameters(wrapped)
  print(f"[train] LoRA rank {rank}: {len(wrapped)} modules, "
        f"{sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)
  opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

  def lr_lambda(step):
    if step < warmup:
      return step / max(1, warmup)
    p = (step - warmup) / max(1, steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))

  sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

  all_meta = [json.loads(l) for l in open(f"{data}/train/meta.jsonl")]
  eval_meta = all_meta[:n_eval]     # held out from training, used for eval below
  meta = all_meta[n_eval:]
  print(f"[train] {len(meta)} training pairs, {len(eval_meta)} held-out for eval", flush=True)

  transformer.train()
  gen = torch.Generator(device="cuda").manual_seed(0)
  rng = random.Random(0)
  torch.cuda.reset_peak_memory_stats()
  run, t_last = 0.0, time.time()

  for step in range(steps):
    opt.zero_grad()
    acc = 0.0
    for _ in range(accum):
      m = rng.choice(meta)
      src, tgt = load_pair("train", m["idx"])
      s = train_edit.images_to_tensor([src], res, res, pipe.device)
      t = train_edit.images_to_tensor([tgt], res, res, pipe.device)
      loss = train_edit.edit_training_step(
        pipe, [m["edit"]], s, t, cfg_dropout_prob=cfg_drop, generator=gen
      )
      (loss / accum).backward()
      acc += loss.item() / accum
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    opt.step()
    sched.step()
    run += acc
    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      print(f"[train] step {step+1}/{steps} loss {run/log_every:.4f} "
            f"lr {sched.get_last_lr()[0]:.2e} | {dt:.2f}s/step "
            f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
      run, t_last = 0.0, time.time()
    if (step + 1) % ckpt_every == 0:
      loramod.save_lora(wrapped, f"{ckpt}/edit_lora_rank{rank}_step{step+1}.safetensors")
      print(f"[train] checkpoint @ step {step+1}", flush=True)

  final = f"{ckpt}/edit_lora_rank{rank}_final.safetensors"
  loramod.save_lora(wrapped, final)
  print(f"[train] saved final -> {final}", flush=True)

  # held-out eval
  transformer.eval()
  eval_dir = f"{out}/eval_out"
  os.makedirs(eval_dir, exist_ok=True)
  for e in eval_meta:
    src, tgt = load_pair("train", e["idx"])
    out_img = edit_sampler.edit_generate(
      pipe, e["edit"], src, height=res, width=res, num_steps=28, guidance_scale=3.0, seed=0
    )
    src.save(f"{eval_dir}/{e['idx']:06d}_src.png")
    tgt.save(f"{eval_dir}/{e['idx']:06d}_tgt.png")
    out_img.save(f"{eval_dir}/{e['idx']:06d}_out.png")
    with open(f"{eval_dir}/{e['idx']:06d}.txt", "w") as f:
      f.write(e["edit"])
    print(f"[eval] idx {e['idx']}: {e['edit'][:70]}", flush=True)
  print("[train] DONE", flush=True)


if __name__ == "__main__":
  main()
