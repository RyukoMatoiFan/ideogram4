"""Joint text-encoder + DiT LoRA training for in-context editing (TE-LoRA).

The cached edit/T2I trainers are encoder-free: text features are precomputed and frozen.
This trainer instead runs the Qwen3-VL text encoder LIVE so a TE-LoRA adapter can learn
(image latents stay cached from precache_edit.py; only the instruction is encoded in-loop).
By default the DiT also carries a LoRA (joint adaptation); set lora.train_transformer:false
to freeze the DiT and train the text encoder alone (gradient still reaches the TE through
llm_features).

Memory: DiT (~fp8) + Qwen3-VL text encoder (bf16, ~16GB) co-resident -> 80GB-class GPU;
gradient checkpointing on both is recommended. Batch is forced to 1 (one live instruction
per micro-step); use optim.accum for a larger effective batch.

  CUDA_VISIBLE_DEVICES=3 python train_te_lora_cached.py --config config/te_lora.yaml

NOTE: implemented against the verified pipeline APIs (build_edit_inputs / _encode_text)
and unit-tested LoRA injection, but not yet smoke-tested end-to-end on the real weights.
Run a short --smoke pass on the server before a long run.

Full text-encoder fine-tune (not LoRA) is a follow-on: it needs the fp8->bf16 dequant +
stochastic-rounding fused-backward engine (see train_edit.dequantize_fp8_transformer /
fused_adamw) applied to the text encoder; scaffolded here via lora.train_transformer and
left as the next step.
"""
import argparse
import json
import os
import random
import time

import torch

from ideogram4.pipeline_ideogram4 import (
  Ideogram4PipelineConfig, _build_transformer, _load_indexed_or_single_state_dict,
)
from ideogram4.modeling_ideogram4 import Ideogram4Config
from ideogram4.scheduler import get_schedule_for_resolution
from ideogram4 import train_edit
from ideogram4 import lora as loramod
from ideogram4.training_utils import build_optimizer, is_finite_loss, build_lr_scheduler


def main():
  ap = argparse.ArgumentParser(description="Joint TE-LoRA + DiT-LoRA edit training from caches.")
  ap.add_argument("--config", default="config/te_lora.yaml")
  ap.add_argument("--smoke", action="store_true", help="run ~20 steps then exit (server smoke)")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)

  weights = cfg.paths.weights
  cache = cfg.paths.cache_dir
  ckpt = cfg.paths.ckpt_dir
  output_dir = cfg.paths.output_dir
  res = int(cfg.data.resolution)
  rank = int(cfg.lora.rank)
  te_rank = int(cfg.lora.te_rank) or rank
  train_dit = bool(cfg.lora.train_transformer)
  lr = float(cfg.optim.lr)
  steps = int(cfg.optim.steps)
  accum = int(cfg.optim.accum)
  warmup = int(cfg.optim.warmup)
  grad_clip = float(cfg.optim.grad_clip)
  cfg_drop = float(cfg.optim.cfg_dropout_prob)
  grad_ckpt = bool(cfg.optim.grad_checkpointing)
  optim_name = cfg.optim.optimizer
  ts_shift = float(cfg.flow.timestep_shift)
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  n_eval = int(cfg.data.n_eval_holdout)
  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)

  os.makedirs(ckpt, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  from ideogram4.trackers import Tracker
  tracker = Tracker(cfg.logging.tracker, project=cfg.logging.wandb_project,
                    run_name=cfg.logging.run_name or None, out_dir=output_dir)
  open(metrics_path, "w").close()

  # --- DiT (denoiser) ---
  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype); del sd
  if grad_ckpt:
    transformer.gradient_checkpointing = True
  print(f"[te-lora] DiT loaded in {time.time()-t0:.1f}s", flush=True)

  # --- text encoder (+ VAE/tokenizer) live, with TE-LoRA ---
  t1 = time.time()
  pipe = train_edit.load_encoders_pipeline(weights, device, dtype)
  pipe.text_encoder.train()
  te_wrapped = loramod.inject_lora_by_names(pipe.text_encoder, rank=te_rank, variant=cfg.lora.variant)
  params = loramod.lora_parameters(te_wrapped)
  print(f"[te-lora] text encoder loaded in {time.time()-t1:.1f}s | TE-LoRA rank {te_rank}: "
        f"{len(te_wrapped)} modules", flush=True)

  dit_wrapped = {}
  if train_dit:
    dit_wrapped = loramod.inject_lora(transformer, rank=rank, variant=cfg.lora.variant,
                                      target_adaln=cfg.lora.target_adaln)
    params = params + loramod.lora_parameters(dit_wrapped)
    print(f"[te-lora] DiT-LoRA rank {rank}: {len(dit_wrapped)} modules (joint)", flush=True)
  else:
    transformer.requires_grad_(False)
    print("[te-lora] DiT frozen (TE-only training)", flush=True)
  transformer.train()
  print(f"[te-lora] trainable params: {sum(p.numel() for p in params)/1e6:.1f}M | "
        f"accum={accum} optim={optim_name} res={res}", flush=True)

  opt = build_optimizer(optim_name, params, lr)
  sched_lr = build_lr_scheduler(opt, scheduler=cfg.optim.lr_scheduler, warmup=warmup,
                                total_steps=steps, num_restarts=int(cfg.optim.num_restarts),
                                min_lr_ratio=float(cfg.optim.min_lr_ratio))
  schedule = get_schedule_for_resolution(
    (res, res), known_mean=cfg.flow.schedule_mean, std=cfg.flow.schedule_std)

  # --- cache pool (reuse precache_edit caches: z_ref/z_tgt + raw instruction) ---
  files = sorted(f for f in os.listdir(cache) if f.endswith(".pt"))
  from collections import defaultdict
  bucket_files = defaultdict(list)
  for f in files:
    if int(f[:-3]) < n_eval:
      continue
    try:
      hdr = torch.load(f"{cache}/{f}", map_location="cpu", mmap=True)
      key = (int(hdr["grid_h"]), int(hdr["grid_w"])); del hdr
    except Exception:
      continue
    bucket_files[key].append(f)
  bucket_keys = list(bucket_files.keys())
  bucket_weights = [len(bucket_files[k]) for k in bucket_keys]
  print(f"[te-lora] {sum(bucket_weights)} training caches ({len(bucket_keys)} AR bucket(s))", flush=True)

  gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
  rng = random.Random(int(cfg.runtime.seed))
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

  def _load(f):
    c = torch.load(f"{cache}/{f}", map_location="cpu")
    return {"grid_h": int(c["grid_h"]), "grid_w": int(c["grid_w"]),
            "z_ref": c["z_ref"].to(device), "z_tgt": c["z_tgt"].to(device),
            "edit": str(c.get("edit", c.get("instruction", "")))}

  def sample_item():
    k = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
    return _load(rng.choice(bucket_files[k]))

  def _save(tag, step_num):
    meta = {"base_model": weights, "type": "ideogram4-te-lora", "step": step_num, "rank": te_rank}
    loramod.save_lora(te_wrapped, f"{ckpt}/te_lora_rank{te_rank}_{tag}.safetensors", metadata=meta)
    if dit_wrapped:
      loramod.save_lora(dit_wrapped, f"{ckpt}/dit_lora_rank{rank}_{tag}.safetensors",
                        metadata={**meta, "type": "ideogram4-edit-lora", "rank": rank})

  n_skipped = 0
  run, t_last = 0.0, time.time()
  for step in range(steps):
    opt.zero_grad(set_to_none=True)
    acc, skip = 0.0, False
    for _ in range(accum):
      it = sample_item()
      loss = train_edit.te_edit_training_step(
        transformer, pipe, it["edit"], it["z_ref"], it["z_tgt"],
        it["grid_h"], it["grid_w"], schedule=schedule, cfg_dropout_prob=cfg_drop,
        timestep_shift=ts_shift, generator=gen)
      if not is_finite_loss(loss):
        skip = True; break
      (loss / accum).backward(); acc += loss.item() / accum
    if skip:
      opt.zero_grad(set_to_none=True); n_skipped += 1; sched_lr.step(); continue
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    opt.step(); sched_lr.step(); run += acc

    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step + 1, "loss": run / log_every, "lr": sched_lr.get_last_lr()[0],
             "s_per_step": dt, "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
             "skipped": n_skipped}
      print(f"[te-lora] step {step+1}/{steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} | "
            f"{dt:.2f}s/step peak {rec['peak_gb']:.1f}GB skipped={n_skipped}", flush=True)
      with open(metrics_path, "a") as mf:
        mf.write(json.dumps(rec) + "\n")
      tracker.log(rec, rec["step"])
      run, t_last = 0.0, time.time()
      if args.smoke and step + 1 >= 20:
        print("[te-lora] SMOKE OK", flush=True); return
    if ckpt_every and (step + 1) % ckpt_every == 0:
      _save(f"step{step+1}", step + 1)

  _save("final", steps)
  print("[te-lora] DONE", flush=True)


if __name__ == "__main__":
  main()
