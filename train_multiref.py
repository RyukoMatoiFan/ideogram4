"""Encoder-free multi-reference editing LoRA training from precomputed caches.

The reference-driven counterpart to train_edit_lora_cached.py: each cache (written by
precache_multiref.py) holds N reference latents on their own MRoPE frames plus a target
and instruction features. Loads ONLY the conditional transformer and trains the LoRA
with train_edit.edit_training_step_multiref (flow-matching MSE on the target tokens).
Saves an ai-toolkit-format adapter; ship it to BOTH transformers at inference.

  CUDA_VISIBLE_DEVICES=0 python train_multiref.py --config config/multiref.yaml
"""
import argparse
import json
import os
import random
import time

import torch

from ideogram4.pipeline_ideogram4 import (
  Ideogram4PipelineConfig,
  _build_transformer,
  _load_indexed_or_single_state_dict,
)
from ideogram4.modeling_ideogram4 import Ideogram4Config
from ideogram4.scheduler import get_schedule_for_resolution
from ideogram4 import train_edit
from ideogram4 import lora as loramod
from ideogram4.training_utils import (
  build_optimizer, is_finite_loss, build_lr_scheduler, is_schedule_free,
  save_training_state, load_training_state,
)
from ideogram4.byg import LoraEMA


def main():
  parser = argparse.ArgumentParser(description="Encoder-free multi-reference edit-LoRA training.")
  parser.add_argument("--config", default="config/multiref.yaml")
  args = parser.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config)
  apply_runtime(cfg)

  weights = cfg.paths.weights
  cache = cfg.paths.cache_dir
  ckpt = cfg.paths.ckpt_dir
  output_dir = cfg.paths.output_dir
  res = int(cfg.data.resolution)
  rank = int(cfg.lora.rank)
  lr = float(cfg.optim.lr)
  steps = int(cfg.optim.steps)
  accum = int(cfg.optim.accum)
  warmup = int(cfg.optim.warmup)
  grad_clip = float(cfg.optim.grad_clip)
  cfg_drop = float(cfg.optim.cfg_dropout_prob)
  grad_ckpt = bool(cfg.optim.grad_checkpointing)
  optim_name = cfg.optim.optimizer
  use_ema = bool(cfg.optim.use_ema)
  ema_decay = float(cfg.optim.ema_decay)
  nan_guard = bool(cfg.optim.nan_guard)
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  sample_every = int(cfg.logging.sample_every)
  n_eval = int(cfg.data.n_eval_holdout)
  dtype = dtype_of(cfg)
  device = torch.device(cfg.runtime.device)

  os.makedirs(ckpt, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  open(metrics_path, "w").close()

  # Load ONLY the conditional transformer (no text encoder / VAE / unconditional).
  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype)
  del sd
  print(f"[mref] transformer loaded in {time.time()-t0:.1f}s (no text encoder / VAE)", flush=True)

  wrapped = loramod.inject_lora(transformer, rank=rank)
  params = loramod.lora_parameters(wrapped)
  if grad_ckpt:
    transformer.gradient_checkpointing = True
  ema = LoraEMA(wrapped, decay=ema_decay) if use_ema else None
  print(f"[mref] LoRA rank {rank}: {len(wrapped)} modules, "
        f"{sum(p.numel() for p in params)/1e6:.1f}M params | accum={accum} "
        f"grad_ckpt={grad_ckpt} optim={optim_name} res={res}", flush=True)

  opt = build_optimizer(optim_name, params, lr)
  sf = is_schedule_free(optim_name)
  if sf:
    opt.train()
  sched_lr = build_lr_scheduler(
    opt, scheduler=cfg.optim.lr_scheduler, warmup=warmup, total_steps=steps,
    num_restarts=int(cfg.optim.num_restarts), min_lr_ratio=float(cfg.optim.min_lr_ratio))
  schedule = get_schedule_for_resolution((res, res), known_mean=1.0)

  # Preload caches to CPU RAM. idx < n_eval -> held-out (never trained).
  files = sorted(f for f in os.listdir(f"{cache}/train") if f.endswith(".pt"))
  cache_items, eval_raw = [], []
  for f in files:
    idx = int(f[:-3])
    (eval_raw if idx < n_eval else cache_items).append(
      torch.load(f"{cache}/train/{f}", map_location="cpu"))
  if not cache_items:
    raise RuntimeError(f"no training caches in {cache}/train (run precache_multiref.py first)")
  print(f"[mref] {len(cache_items)} training + {len(eval_raw)} held-out caches in RAM", flush=True)

  def _to_dev(c):
    return {
      "z_refs": [z.to(device) for z in c["z_refs"]],
      "ref_grids": [tuple(g) for g in c["ref_grids"]],
      "z_tgt": c["z_tgt"].to(device),
      "llm_text": c["llm_text"].to(device),
      "target_grid": tuple(c["target_grid"]),
    }

  eval_items = [_to_dev(c) for c in eval_raw]

  transformer.train()
  gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
  rng = random.Random(int(cfg.runtime.seed))
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  run, t_last = 0.0, time.time()
  n_skipped = 0

  def _meta(step_num):
    return {"base_model": weights, "type": "ideogram4-multiref-lora", "rank": rank,
            "step": step_num, "optimizer": optim_name, "ema": use_ema}

  def _save(tag, step_num):
    if sf:
      opt.eval()
    loramod.save_lora(wrapped, f"{ckpt}/multiref_rank{rank}_{tag}.safetensors", metadata=_meta(step_num))
    if ema is not None:
      with ema.swap_in():
        loramod.save_lora(wrapped, f"{ckpt}/multiref_rank{rank}_{tag}_ema.safetensors", metadata=_meta(step_num))
    if sf:
      opt.train()
    save_training_state(
      f"{ckpt}/resume.pt", step=step_num, optimizer=opt, scheduler=sched_lr,
      wrapped=wrapped, ema=ema, gen=gen, extra={"n_skipped": n_skipped, "rng": rng.getstate()})

  sample_dir = os.path.join(output_dir, "samples")
  decoder_state = {"d": None}

  def _sample(step_num):
    from PIL import Image
    from ideogram4 import edit_sampler
    if decoder_state["d"] is None:
      decoder_state["d"] = edit_sampler.load_decoder(weights, device, dtype)
    ae, shift, scale, patch = decoder_state["d"]
    os.makedirs(sample_dir, exist_ok=True)
    if sf:
      opt.eval()
    transformer.eval()
    for j, it in enumerate(eval_items):
      z_out = edit_sampler.sample_multiref(
        transformer, it["z_refs"], it["ref_grids"], it["llm_text"], it["target_grid"],
        schedule=schedule, num_steps=int(cfg.logging.sample_steps),
        guidance_scale=float(cfg.logging.sample_guidance),
        generator=torch.Generator(device=device).manual_seed(step_num))
      gh, gw = it["target_grid"]
      tgt = it["z_tgt"].unsqueeze(0)
      imgs = edit_sampler.decode_latents(ae, torch.cat([z_out, tgt], dim=0), gh, gw,
                                         patch_size=patch, latent_shift=shift,
                                         latent_scale=scale, dtype=dtype)
      w, h = imgs[0].size
      canvas = Image.new("RGB", (w * 2, h))
      for k, im in enumerate(imgs):
        canvas.paste(im, (k * w, 0))  # generated | target
      canvas.save(os.path.join(sample_dir, f"step{step_num:06d}_idx{j}.png"))
    transformer.train()
    if sf:
      opt.train()
    print(f"[mref] sampled {len(eval_items)} held-out @ step {step_num} -> {sample_dir}", flush=True)

  # Resume.
  start_step = 0
  resume_from = cfg.paths.resume_from
  if resume_from == "auto":
    resume_from = f"{ckpt}/resume.pt"
  if resume_from and os.path.exists(resume_from):
    start_step, ex = load_training_state(
      resume_from, optimizer=opt, scheduler=sched_lr, wrapped=wrapped,
      ema=ema, gen=gen, map_location=device)
    n_skipped = ex.get("n_skipped", 0)
    if "rng" in ex:
      rng.setstate(ex["rng"])
    print(f"[mref] RESUMED from {resume_from} at step {start_step}", flush=True)

  for step in range(start_step, steps):
    opt.zero_grad()
    acc = 0.0
    skip = False
    for _ in range(accum):
      it = _to_dev(rng.choice(cache_items))
      loss = train_edit.edit_training_step_multiref(
        transformer, it["z_refs"], it["ref_grids"], it["z_tgt"], it["llm_text"],
        it["target_grid"], schedule=schedule, cfg_dropout_prob=cfg_drop, generator=gen)
      if nan_guard and not is_finite_loss(loss):
        skip = True
        break
      (loss / accum).backward()
      acc += loss.item() / accum
    if skip:
      opt.zero_grad(set_to_none=True)
      n_skipped += 1
      continue
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    opt.step()
    if not sf:
      sched_lr.step()
    if ema is not None:
      ema.update()
    run += acc
    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step + 1, "loss": run / log_every, "lr": sched_lr.get_last_lr()[0],
             "s_per_step": dt, "peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                                           if device.type == "cuda" else 0.0),
             "skipped": n_skipped}
      print(f"[mref] step {step+1}/{steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
            f"| {dt:.2f}s/step peak {rec['peak_gb']:.1f}GB skipped={n_skipped}", flush=True)
      with open(metrics_path, "a") as mf:
        mf.write(json.dumps(rec) + "\n")
      run, t_last = 0.0, time.time()
    if (step + 1) % ckpt_every == 0:
      _save(f"step{step+1}", step + 1)
      print(f"[mref] checkpoint @ step {step+1}", flush=True)
    if sample_every and eval_items and (step + 1) % sample_every == 0:
      try:
        _sample(step + 1)
      except Exception as exc:
        print(f"[mref] sampling skipped @ step {step+1}: {exc}", flush=True)

  _save("final", steps)
  print(f"[mref] DONE -> {ckpt}/multiref_rank{rank}_final.safetensors "
        f"(skipped {n_skipped} non-finite steps)", flush=True)


if __name__ == "__main__":
  main()
