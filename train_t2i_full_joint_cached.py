"""JOINT full fine-tune of BOTH the T2I transformer (DiT) AND the text encoder (Qwen3-VL).

Both are published fp8-only, so both are dequantized fp8->bf16 into trainable nn.Linear
(lossy ~5e-3 each -- a slightly degraded start; that is the cost of the fp8-only release).
The DiT denoises cached VAE latents; the text encoder runs LIVE (caption re-encoded in-loop)
so its gradient flows. Memory is held down by:
  * fused per-parameter backward + stochastic-rounding AdamW with CPU-offloaded moments
    (so the ~138GB of fp32 moments live in host RAM, not VRAM),
  * gradient checkpointing on the DiT AND on the TE's manual layer loop.

Effective batch = optim.accum (the live-TE step is batch-1): `accum` step-losses are SUMMED
into ONE backward so the fused hooks free each grad immediately (grads never coexist).

WARNING: full-FT'ing an 8B chat-VLM text encoder on a narrow dataset risks catastrophic
forgetting of general language understanding. Defaults use a LOWER TE LR (optim.te_lr,
default lr/10) as a safeguard; monitor prompt generalization. DiT-full + TE-LoRA
(train_te_lora_cached.py) is the lower-risk alternative.

  CUDA_VISIBLE_DEVICES=0 python train_t2i_full_joint_cached.py --config config/t2i_full_joint.yaml
  Smoke: add --smoke
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
from ideogram4 import train_edit, train_t2i
from ideogram4.fused_adamw import build_fused_adamw, build_fused_adafactor, _sr_seed
from ideogram4.training_utils import is_finite_loss, build_lr_scheduler


def main():
  ap = argparse.ArgumentParser(description="Joint full-FT of DiT + text encoder (T2I).")
  ap.add_argument("--config", default="config/t2i_full_joint.yaml")
  ap.add_argument("--smoke", action="store_true")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)

  weights = cfg.paths.weights
  cache = cfg.paths.cache_dir
  ckpt = cfg.paths.ckpt_dir
  output_dir = cfg.paths.output_dir
  res = int(cfg.data.resolution)
  lr = float(cfg.optim.lr)
  train_dit = bool(cfg.optim.train_dit)           # False -> TE-only stage (DiT frozen fp8)
  te_lr = float(cfg.optim.te_lr) or (lr / 10.0 if train_dit else lr)  # TE-only -> TE uses main lr
  steps = int(cfg.optim.steps)
  accum = max(1, int(cfg.optim.accum))
  warmup = int(cfg.optim.warmup)
  grad_clip = float(cfg.optim.grad_clip)
  cfg_drop = float(cfg.optim.cfg_dropout_prob)
  ts_shift = float(cfg.flow.timestep_shift)
  n_eval = int(cfg.data.n_eval_holdout)
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)
  offload = bool(cfg.optim.offload_optimizer)

  os.makedirs(ckpt, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  from ideogram4.trackers import Tracker
  tracker = Tracker(cfg.logging.tracker, project=cfg.logging.wandb_project,
                    run_name=cfg.logging.run_name or None, out_dir=output_dir)
  open(metrics_path, "w").close()

  # --- DiT: fp8 -> bf16 trainable ---
  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype); del sd
  if train_dit:
    train_edit.dequantize_fp8_transformer(transformer, dtype=dtype)
    transformer.to(device); transformer.requires_grad_(True)
    print(f"[joint] DiT dequantized fp8->bf16 (trainable) in {time.time()-t0:.1f}s", flush=True)
  else:
    transformer.to(device); transformer.requires_grad_(False)
    print(f"[joint] DiT kept fp8 (FROZEN) in {time.time()-t0:.1f}s -- TE-only stage", flush=True)
  transformer.gradient_checkpointing = True  # .train() below; grad still flows through a frozen DiT
  transformer.train()

  # --- text encoder: fp8 -> bf16 trainable (same dequant; TE is also fp8-only) ---
  t1 = time.time()
  pipe = train_edit.load_encoders_pipeline(weights, device, dtype)
  train_edit.dequantize_fp8_transformer(pipe.text_encoder, dtype=dtype)
  pipe.text_encoder.to(device); pipe.text_encoder.requires_grad_(True); pipe.text_encoder.train()
  print(f"[joint] text encoder dequantized fp8->bf16 in {time.time()-t1:.1f}s", flush=True)

  dit_params = [p for p in transformer.parameters() if p.requires_grad]   # [] when DiT frozen
  te_params = [p for p in pipe.text_encoder.parameters() if p.requires_grad]
  n_dit = sum(p.numel() for p in dit_params)
  n_te = sum(p.numel() for p in te_params)
  print(f"[joint] trainable: DiT {n_dit/1e9:.2f}B (lr {lr:.1e}) + TE {n_te/1e9:.2f}B (lr {te_lr:.1e}) "
        f"| accum={accum} offload={offload}", flush=True)

  _sr_seed(cfg.runtime.seed, device)
  groups = []
  if dit_params:
    groups.append({"params": dit_params, "lr": lr})
  groups.append({"params": te_params, "lr": te_lr})
  if cfg.optim.optimizer_state.lower() == "adafactor":
    opt = build_fused_adafactor(groups, lr, stochastic_rounding=True)
    print("[joint] optimizer: per-parameter Adafactor (factored 2nd moment, on-GPU, no offload)", flush=True)
  else:
    opt = build_fused_adamw(groups, lr, stochastic_rounding=True, offload_states=offload)
    print(f"[joint] optimizer: fused AdamW (moments {'CPU-offload' if offload else 'on-GPU'})", flush=True)
  sched_lr = build_lr_scheduler(opt, scheduler=cfg.optim.lr_scheduler, warmup=warmup,
                                total_steps=steps, num_restarts=int(cfg.optim.num_restarts),
                                min_lr_ratio=float(cfg.optim.min_lr_ratio))

  # Fused per-parameter backward over BOTH models: each grad is consumed + freed the
  # instant it is ready, so DiT+TE grads (~35GB) never coexist. accum is done by SUMMING
  # `accum` step-losses into ONE backward (below), not by accumulating live grads.
  # accum==1: fused per-parameter backward (each grad freed instantly -> lowest VRAM;
  #   the only config that fits dual-FFT on one 80GB GPU). accum>1: standard accumulation
  #   + a manual SR step (grads coexist) -- viable for the lighter TE-only stage, but
  #   dual-FFT (DiT+TE) is effectively batch-1 on a single GPU (see header).
  fused = (accum == 1)
  handles = []
  if fused:
    for group in opt.param_groups:
      for i, p in enumerate(group["params"]):
        if not p.requires_grad:
          continue
        def _hook(param, g=group, idx=i):
          if grad_clip:
            torch.nn.utils.clip_grad_norm_(param, grad_clip)
          opt.step_parameter(param, g, idx)
          param.grad = None
        handles.append(p.register_post_accumulate_grad_hook(_hook))
    print(f"[joint] fused back pass: {len(handles)} per-parameter hooks", flush=True)
  else:
    print(f"[joint] grad-accum x{accum}: standard backward + manual SR step", flush=True)

  def _manual_opt_step():
    for group in opt.param_groups:
      for i, p in enumerate(group["params"]):
        if p.grad is None:
          continue
        if grad_clip:
          torch.nn.utils.clip_grad_norm_(p, grad_clip)
        opt.step_parameter(p, group, i)
        p.grad = None

  schedule = get_schedule_for_resolution(
    (res, res), known_mean=cfg.flow.schedule_mean, std=cfg.flow.schedule_std)

  # --- cache pool (precache_t2i: z_tgt + caption; llm_text ignored -> TE runs live) ---
  files = sorted(f for f in os.listdir(cache) if f.endswith(".pt") and f[:-3].isdigit())
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
  print(f"[joint] {sum(bucket_weights)} training caches ({len(bucket_keys)} AR bucket(s))", flush=True)

  gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
  rng = random.Random(int(cfg.runtime.seed))
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

  def sample_item():
    k = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
    c = torch.load(f"{cache}/{rng.choice(bucket_files[k])}", map_location="cpu")
    return {"grid_h": int(c["grid_h"]), "grid_w": int(c["grid_w"]),
            "z_tgt": c["z_tgt"].to(device), "caption": str(c.get("caption", ""))}

  def _save(tag, step_num):
    from safetensors.torch import save_file
    if train_dit:  # DiT unchanged when frozen -> no point re-saving it
      ds = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in transformer.state_dict().items()}
      save_file(ds, f"{ckpt}/joint_dit_{tag}.safetensors",
                metadata={"base_model": weights, "type": "ideogram4-t2i-full", "step": str(step_num)})
    ts = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in pipe.text_encoder.state_dict().items()}
    save_file(ts, f"{ckpt}/joint_te_{tag}.safetensors",
              metadata={"base_model": weights, "type": "ideogram4-text-encoder-full", "step": str(step_num)})
    print(f"[joint] saved {'DiT+TE' if train_dit else 'TE'} @ {step_num}", flush=True)

  n_skipped = 0
  run, t_last = 0.0, time.time()
  def _train_step():
    it = sample_item()
    return train_t2i.t2i_te_training_step(
      transformer, pipe, it["caption"], it["z_tgt"], it["grid_h"], it["grid_w"],
      schedule=schedule, cfg_dropout_prob=cfg_drop, timestep_shift=ts_shift, generator=gen)

  for step in range(steps):
    if fused:
      loss = _train_step()
      if not is_finite_loss(loss):
        n_skipped += 1; sched_lr.step(); continue
      loss.backward()           # fused hooks step + free per parameter
      run += float(loss.item())
    else:
      acc_loss, ok = 0.0, True
      for _ in range(accum):     # accumulate accum micro-batches, then one manual step
        l = _train_step()
        if not is_finite_loss(l):
          ok = False; break
        (l / accum).backward(); acc_loss += float(l.item()) / accum
      if not ok:
        for grp in opt.param_groups:
          for p in grp["params"]:
            p.grad = None
        n_skipped += 1; sched_lr.step(); continue
      _manual_opt_step(); run += acc_loss
    sched_lr.step()

    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step + 1, "loss": run / log_every, "lr": sched_lr.get_last_lr()[0],
             "te_lr": sched_lr.get_last_lr()[-1], "s_per_step": dt,
             "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
             "skipped": n_skipped}
      print(f"[joint] step {step+1}/{steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
            f"te_lr {rec['te_lr']:.2e} | {dt:.2f}s/step peak {rec['peak_gb']:.1f}GB skipped={n_skipped}",
            flush=True)
      with open(metrics_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
      tracker.log(rec, rec["step"])
      run, t_last = 0.0, time.time()
      if args.smoke and step + 1 >= 20:
        print("[joint] SMOKE OK", flush=True); return
    if ckpt_every and (step + 1) % ckpt_every == 0:
      _save(f"step{step+1:06d}", step + 1)

  _save("final", steps); print("[joint] DONE", flush=True)


if __name__ == "__main__":
  main()
