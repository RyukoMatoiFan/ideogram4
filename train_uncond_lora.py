"""Train a negative-model LoRA on the UNCONDITIONAL transformer (cached latents).

The idea: the uncond branch of CFG enters the update as ``g*v_cond + (1-g)*v_uncond``
with ``(1-g) < 0`` for any g>1 -- so whatever the uncond predicts *toward* gets pushed
AWAY from, amplified by ``(g-1)``. Fine-tune a LoRA on the uncond with a plain
flow-matching loss over DEGRADED images (see precache_uncond.py) and you get a
dial-able quality knob: hook the adapter into the uncond only, scale>0 steers
generations away from blur/JPEG/noise, scale<0 inverts (a causal sanity check).

Disentanglement: flow loss on degraded images ALONE also absorbs the dataset's
content/style (steering away then e.g. de-photographs the image). The clean-anchor
term (uncond_anchor_step_cached, weight optim.anchor_weight) pins the adapter to a
no-op on the CLEAN copies of the same images, cancelling the content component so
only the degradation axis survives.

No text encoder anywhere: the uncond is image-only, so training loads ONE 9.3B
transformer and (n,128) latent caches (all preloaded to RAM -- the set is ~2GB).
The train/val split is grouped by SOURCE image (hash-ordered, stable as the cache
grows): every degraded variant and the clean copy of one image land on the same
side, and val-side cleans are excluded from the anchor pool. Verify with
eval_uncond_lora.py.

  CUDA_VISIBLE_DEVICES=0 python train_uncond_lora.py --config config/uncond_quality.yaml
"""
import argparse
import glob
import hashlib
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
from ideogram4.train_t2i import uncond_training_step_cached, uncond_anchor_step_cached
from ideogram4 import lora as loramod
from ideogram4.training_utils import build_optimizer, build_lr_scheduler, is_finite_loss

VAL_TS = (0.25, 0.5, 0.75)  # fixed timesteps -> low-variance, comparable val curve
MAX_CONSECUTIVE_SKIPS = 50  # abort a zombie run instead of logging zeros to step N


def _src_idx(path: str) -> str:
  """Source-image index of a cache file ({idx}_{v}.pt / {idx}_c.pt)."""
  return os.path.basename(path).split("_")[0]


def main():
  ap = argparse.ArgumentParser(description="Train a negative-model (uncond) LoRA.")
  ap.add_argument("--config", default="config/uncond_quality.yaml")
  ap.add_argument("--smoke", action="store_true",
                  help="few steps, tiny cadence; outputs to <output_dir>/smoke (never "
                       "clobbers a real run's checkpoints or metrics)")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)
  rank = int(cfg.lora.rank)
  alpha = cfg.lora.alpha  # None -> inject_lora defaults to rank (scale 1.0)
  steps = int(cfg.optim.steps)
  batch = int(cfg.optim.batch)
  accum = int(cfg.optim.accum)
  output_dir = cfg.paths.output_dir
  ckpt = cfg.paths.ckpt_dir
  if args.smoke:
    steps = 20
    output_dir = os.path.join(output_dir, "smoke")
    ckpt = os.path.join(output_dir, "ckpts")
  os.makedirs(ckpt, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  state_path = os.path.join(ckpt, "train_state.pt")
  resume = (str(cfg.paths.resume_from or "") == "auto") and os.path.exists(state_path) \
           and not args.smoke

  # --- file lists: split grouped by SOURCE image, hash-ordered (stable as cache grows) ---
  all_files = sorted(glob.glob(os.path.join(cfg.paths.cache_dir, "*.pt")))
  files = [f for f in all_files if not f.endswith("_c.pt")]       # degraded variants
  clean_files = [f for f in all_files if f.endswith("_c.pt")]     # clean anchors
  if not files:
    raise FileNotFoundError(f"no caches under {cfg.paths.cache_dir}; run precache_uncond.py")
  anchor_w = float(cfg.optim.anchor_weight)
  if anchor_w > 0.0 and not clean_files:
    raise FileNotFoundError(
      f"anchor_weight={anchor_w} but no *_c.pt clean caches under {cfg.paths.cache_dir}; "
      "re-run precache_uncond.py (it now writes clean copies) or set anchor_weight: 0")
  indices = sorted({_src_idx(f) for f in files},
                   key=lambda s: hashlib.sha1(s.encode()).hexdigest())
  n_hold = min(int(cfg.data.n_eval_holdout), max(1, len(indices) // 10))
  val_idx = set(indices[:n_hold])
  train_files = [f for f in files if _src_idx(f) not in val_idx]
  val_files = [f for f in files if _src_idx(f) in val_idx]
  clean_train = [f for f in clean_files if _src_idx(f) not in val_idx]
  clean_val = [f for f in clean_files if _src_idx(f) in val_idx]
  if anchor_w > 0.0 and not clean_train:
    raise FileNotFoundError("no clean caches left on the train side; lower n_eval_holdout")
  probe = torch.load(files[0], map_location="cpu", weights_only=True)
  grid_h, grid_w = int(probe["grid_h"]), int(probe["grid_w"])
  print(f"[uncond] split by source image: {len(train_files)} train / {len(val_files)} val "
        f"degraded | {len(clean_train)} train / {len(clean_val)} val clean anchors "
        f"(w={anchor_w}) | grid {grid_h}x{grid_w}", flush=True)

  # --- preload every latent to RAM: kills per-step network-disk torch.load ---
  # Cast fp16->fp32 ONCE here (exact/widening, so values are identical and downstream
  # already upcast) instead of on every batch fetch; costs ~2x RAM for this buffer (~4GB).
  t0 = time.time()
  ram: dict[str, torch.Tensor] = {}
  for p in train_files + val_files + clean_train + clean_val:
    ram[p] = torch.load(p, map_location="cpu", weights_only=True)["z"].to(torch.float32)
  print(f"[uncond] preloaded {len(ram)} latents to RAM in {time.time()-t0:.1f}s", flush=True)

  def _load_z(paths):
    return torch.stack([ram[p] for p in paths]).to(device)

  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=cfg.paths.weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.unconditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype)
  del sd
  print(f"[uncond] UNCONDITIONAL transformer loaded in {time.time()-t0:.1f}s", flush=True)

  wrapped = loramod.inject_lora(transformer, rank=rank, alpha=alpha)
  params = loramod.lora_parameters(wrapped)
  if bool(cfg.optim.grad_checkpointing):
    transformer.gradient_checkpointing = True
  opt = build_optimizer(cfg.optim.optimizer, params, float(cfg.optim.lr))
  sched_lr = build_lr_scheduler(
    opt, scheduler=cfg.optim.lr_scheduler, warmup=int(cfg.optim.warmup), total_steps=steps,
    num_restarts=int(cfg.optim.num_restarts), min_lr_ratio=float(cfg.optim.min_lr_ratio))
  res = grid_h * pcfg.patch_size * pcfg.ae_scale_factor
  schedule = get_schedule_for_resolution((res, res), known_mean=1.0)
  ts_shift = float(cfg.flow.timestep_shift)
  print(f"[uncond] LoRA rank {rank}: {len(wrapped)} modules, "
        f"{sum(p.numel() for p in params)/1e6:.1f}M params | ts_shift {ts_shift}", flush=True)

  transformer.train()
  gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
  rng = random.Random(int(cfg.runtime.seed))
  order, cursor = list(range(len(train_files))), 0
  start_step = 0

  # --- resume: adapter + optimizer + LR + RNG + data order survive a crash ---
  if resume:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
      for sub, m in wrapped.items():
        m.lora_A.copy_(state["lora"][sub]["A"].to(device, m.lora_A.dtype))
        m.lora_B.copy_(state["lora"][sub]["B"].to(device, m.lora_B.dtype))
    opt.load_state_dict(state["opt"])
    sched_lr.load_state_dict(state["sched"])
    gen.set_state(state["gen"].to(gen.get_state().device)
                  if hasattr(state["gen"], "to") else state["gen"])
    rng.setstate(state["rng"])
    order, cursor = state["order"], state["cursor"]
    start_step = int(state["step"])
    print(f"[uncond] RESUMED from {state_path} @ step {start_step}", flush=True)
  open(metrics_path, "a" if resume else "w").close()
  with open(os.path.join(output_dir, "run_config.json"), "w") as cf:
    json.dump({"config": args.config, "rank": rank, "alpha": alpha, "steps": steps,
               "batch": batch, "accum": accum, "lr": float(cfg.optim.lr),
               "anchor_weight": anchor_w, "ts_shift": ts_shift,
               "n_train": len(train_files), "n_val": len(val_files)}, cf, indent=1)

  def _save_state(step_num):
    torch.save({
      "step": step_num, "opt": opt.state_dict(), "sched": sched_lr.state_dict(),
      "gen": gen.get_state(), "rng": rng.getstate(), "order": order, "cursor": cursor,
      "lora": {sub: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()}
               for sub, m in wrapped.items()},
    }, state_path + ".tmp")
    os.replace(state_path + ".tmp", state_path)

  @torch.no_grad()
  def _val_loss(base: bool = False):
    """Deterministic held-out losses at fixed timesteps: degraded-flow fit + clean-anchor
    drift. ``base=True`` measures the frozen base (run once -- the comparison floor)."""
    from ideogram4.lora import lora_disabled
    import contextlib
    ctx = lora_disabled(transformer) if base else contextlib.nullcontext()
    transformer.eval()
    try:
      with ctx:
        flow_tot, anc_tot = 0.0, 0.0
        for ti, tv in enumerate(VAL_TS):
          for k in range(0, len(val_files), batch):
            z = _load_z(val_files[k:k + batch])
            g = torch.Generator(device=device).manual_seed(
              int(cfg.runtime.seed) + 7919 * ti + k)  # varied but reproducible noise
            flow_tot += uncond_training_step_cached(
              transformer, z, grid_h, grid_w, schedule=schedule,
              t_override=torch.tensor(tv), generator=g).item() * z.shape[0]
        for k in range(0, len(clean_val), batch):
          zc = _load_z(clean_val[k:k + batch])
          g = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed) + k)
          anc_tot += uncond_anchor_step_cached(
            transformer, zc, grid_h, grid_w, schedule=schedule,
            generator=g).item() * zc.shape[0]
    finally:
      transformer.train()
    return (flow_tot / max(1, len(VAL_TS) * len(val_files)),
            anc_tot / max(1, len(clean_val)))

  sample_dir = os.path.join(output_dir, "samples")
  decoder_state = {"d": None}

  @torch.no_grad()
  def _sample(step_num):
    """Decode uncond-only draws at adapter scale [0 | 1]: the right column should
    drift toward the degradation manifold (and ONLY that -- same content/style) as
    the LoRA learns. train() is restored in a finally: a preview failure must not
    leave the model in eval mode (that would silently disable grad checkpointing)."""
    from PIL import Image as PILImage
    from ideogram4 import edit_sampler
    if decoder_state["d"] is None:
      decoder_state["d"] = edit_sampler.load_decoder(cfg.paths.weights, device, dtype)
    ae, shift, lscale, dpatch = decoder_state["d"]
    os.makedirs(sample_dir, exist_ok=True)
    transformer.eval()
    try:
      rows = []
      for k in range(int(cfg.logging.sample_count)):
        panels = []
        for factor in (0.0, 1.0):
          with loramod.lora_scaled(transformer, factor):
            z = edit_sampler.sample_uncond(
              transformer, grid_h, grid_w, schedule=schedule,
              num_steps=int(cfg.logging.sample_steps),
              generator=torch.Generator(device=device).manual_seed(7000 + k))
          panels.append(edit_sampler.decode_latents(
            ae, z, grid_h, grid_w, patch_size=dpatch, latent_shift=shift,
            latent_scale=lscale, dtype=dtype)[0])
        rows.append(panels)
    finally:
      transformer.train()
    w, h = rows[0][0].size
    canvas = PILImage.new("RGB", (w * 2, h * len(rows)), (15, 15, 15))
    for r, panels in enumerate(rows):
      for c, im in enumerate(panels):
        canvas.paste(im, (c * w, r * h))  # base uncond | lora uncond
    canvas.save(os.path.join(sample_dir, f"step{step_num:06d}.jpg"), quality=90)
    print(f"[uncond] sampled {len(rows)} [base|lora] @ step {step_num} -> {sample_dir}", flush=True)

  def _meta(step_num):
    return {"base_model": cfg.paths.weights, "type": "ideogram4-uncond-lora", "rank": rank,
            "alpha": alpha if alpha is not None else rank,
            "step": step_num, "target": "unconditional_transformer",
            "usage": "hook into the uncond ONLY; scale>0 steers away from the trained manifold"}

  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  val_every = int(cfg.logging.val_every)
  sample_every = int(cfg.logging.sample_every)
  if args.smoke:
    log_every, val_every, sample_every, ckpt_every = 5, 10, 20, 10**9
  if start_step == 0 and val_every:
    vb, ab = _val_loss(base=True)
    print(f"[uncond] BASE val flow {vb:.4f} anchor {ab:.4f} (frozen-base floor)", flush=True)
    with open(metrics_path, "a") as mf:
      mf.write(json.dumps({"step": 0, "val_loss": vb, "val_anchor": ab, "base": True}) + "\n")

  run_flow, run_anchor, t_last, t_step_prev = 0.0, 0.0, time.time(), time.time()
  consecutive_skips = 0

  for step in range(start_step, steps):
    opt.zero_grad()
    acc_flow, acc_anchor, contributed = 0.0, 0.0, 0
    for _ in range(accum):
      if cursor + batch > len(order):
        rng.shuffle(order); cursor = 0
      z = _load_z([train_files[i] for i in order[cursor:cursor + batch]])
      cursor += batch
      loss_flow = uncond_training_step_cached(
        transformer, z, grid_h, grid_w, schedule=schedule,
        timestep_shift=ts_shift, generator=gen)
      loss = loss_flow
      loss_anchor = None
      if anchor_w > 0.0:
        zc = _load_z([clean_train[rng.randrange(len(clean_train))] for _ in range(batch)])
        loss_anchor = uncond_anchor_step_cached(
          transformer, zc, grid_h, grid_w, schedule=schedule,
          timestep_shift=ts_shift, generator=gen)
        loss = loss + anchor_w * loss_anchor
      if cfg.optim.nan_guard and not is_finite_loss(loss):
        print(f"[uncond] WARN non-finite loss @ step {step+1}, micro-batch skipped", flush=True)
        continue
      (loss / accum).backward()
      acc_flow += loss_flow.item() / accum
      acc_anchor += (loss_anchor.item() if loss_anchor is not None else 0.0) / accum
      contributed += 1
    norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.optim.grad_clip))
    if contributed == 0 or not torch.isfinite(norm):
      # never let a NaN total-norm scale the grads into the weights (zombie-run guard)
      opt.zero_grad()
      consecutive_skips += 1
      print(f"[uncond] WARN step {step+1} skipped entirely "
            f"(contributed={contributed}, grad_norm={float(norm):.3g}); "
            f"{consecutive_skips} consecutive", flush=True)
      if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
        raise RuntimeError(
          f"{MAX_CONSECUTIVE_SKIPS} consecutive non-finite steps -- adapter or inputs are "
          f"corrupt; resume from the last good train_state.pt after investigating")
      sched_lr.step()
      continue
    consecutive_skips = 0
    if contributed < accum:  # un-bias the surviving gradient under accumulation
      for p in params:
        if p.grad is not None:
          p.grad.mul_(accum / contributed)
    opt.step()
    sched_lr.step()
    run_flow += acc_flow; run_anchor += acc_anchor
    now = time.time(); step_dt = now - t_step_prev; t_step_prev = now
    pgb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    rec = {"step": step + 1, "loss": acc_flow + anchor_w * acc_anchor, "flow": acc_flow,
           "anchor": acc_anchor, "lr": sched_lr.get_last_lr()[0],
           "s_per_step": step_dt, "peak_gb": pgb, "total": steps}
    if val_every and (step + 1) % val_every == 0:
      rec["val_loss"], rec["val_anchor"] = _val_loss()
      print(f"[uncond] val flow {rec['val_loss']:.4f} anchor {rec['val_anchor']:.4f} "
            f"@ step {step+1}", flush=True)
    with open(metrics_path, "a") as mf:
      mf.write(json.dumps(rec) + "\n")
    if (step + 1) % log_every == 0:
      print(f"[uncond] step {step+1}/{steps} flow {run_flow/log_every:.4f} "
            f"anchor {run_anchor/log_every:.4f} lr {sched_lr.get_last_lr()[0]:.2e} "
            f"| {(now-t_last)/log_every:.2f}s/step peak {pgb:.1f}GB", flush=True)
      run_flow, run_anchor, t_last = 0.0, 0.0, now
    if (step + 1) % ckpt_every == 0:
      loramod.save_lora(wrapped, f"{ckpt}/uncond_rank{rank}_step{step+1}.safetensors",
                        metadata=_meta(step + 1))
      _save_state(step + 1)
      print(f"[uncond] checkpoint + train state @ step {step+1}", flush=True)
    if sample_every and (step + 1) % sample_every == 0:
      try:
        _sample(step + 1)
      except Exception as exc:  # previews must never crash training
        print(f"[uncond] sampling skipped @ step {step+1}: {exc}", flush=True)

  loramod.save_lora(wrapped, f"{ckpt}/uncond_rank{rank}_final.safetensors", metadata=_meta(steps))
  print(f"[uncond] DONE -> {ckpt}/uncond_rank{rank}_final.safetensors", flush=True)


if __name__ == "__main__":
  main()
