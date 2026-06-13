"""Encoder-free rank-128 edit-LoRA training from precomputed caches.

Loads ONLY the conditional transformer (no text encoder, no VAE, no unconditional
transformer) and trains on the .pt caches written by precache_edit.py. Much lower
VRAM and no text-encoder rebuild. Saves the LoRA in the standard diffusion_model.*
key layout; evaluation is a separate step that loads the full pipeline (eval_edit_lora.py).

  CUDA_VISIBLE_DEVICES=0 python train_edit_lora_cached.py --config config/edit_lora_cached.yaml

The old RANK/BATCH/ACCUM/... env knobs are now served by the config loader's
IG4_* layer, e.g. RANK -> IG4_LORA__RANK, BATCH -> IG4_OPTIM__BATCH,
STEPS -> IG4_OPTIM__STEPS, GRAD_CKPT -> IG4_OPTIM__GRAD_CHECKPOINTING,
OPTIM -> IG4_OPTIM__OPTIMIZER, CACHE -> IG4_PATHS__CACHE_DIR.
"""
import argparse
import json
import math
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
from ideogram4 import edit_sampler
from ideogram4.training_utils import (
  build_optimizer, is_finite_loss, compute_val_loss, build_lr_scheduler, is_schedule_free,
  save_training_state, load_training_state,
)
from ideogram4.byg import LoraEMA


def main():
  parser = argparse.ArgumentParser(description="Encoder-free edit-LoRA training from caches.")
  parser.add_argument("--config", default="config/edit_lora_cached.yaml",
                      help="Path to YAML config (default: config/edit_lora_cached.yaml)")
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
  batch = int(cfg.optim.batch)        # true batch size (per micro-step)
  accum = int(cfg.optim.accum)        # gradient accumulation micro-steps
  warmup = int(cfg.optim.warmup)
  grad_clip = float(cfg.optim.grad_clip)
  cfg_drop = float(cfg.optim.cfg_dropout_prob)
  grad_ckpt = bool(cfg.optim.grad_checkpointing)
  optim_name = cfg.optim.optimizer    # adamw | adamw8bit | prodigy
  use_ema = bool(cfg.optim.use_ema)
  ema_decay = float(cfg.optim.ema_decay)
  nan_guard = bool(cfg.optim.nan_guard)
  ts_shift = float(cfg.flow.timestep_shift)
  ts_weighting = cfg.flow.timestep_weighting
  min_snr_gamma = float(cfg.flow.min_snr_gamma)
  noise_offset = float(cfg.flow.noise_offset)
  input_perturbation = float(cfg.flow.input_perturbation)
  prior_pres = float(cfg.optim.prior_preservation_weight)
  masked = bool(cfg.data.masked_loss)
  mask_q = float(cfg.data.mask_quantile)
  mask_bg = float(cfg.data.mask_bg_weight)
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  val_every = int(cfg.logging.val_every)
  sample_every = int(cfg.logging.sample_every)
  n_eval = int(cfg.data.n_eval_holdout)  # idx 0..n-1 reserved, not trained
  dtype = dtype_of(cfg)

  os.makedirs(ckpt, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  # Only truncate the loss log on a FRESH run. On resume (the step loop appends with
  # mode 'a'), keeping history avoids wiping the dashboard's prior metrics.
  _resume_target = f"{ckpt}/resume.pt" if cfg.paths.resume_from == "auto" else cfg.paths.resume_from
  _resuming = bool(cfg.paths.resume_from and _resume_target and os.path.exists(_resume_target))
  if not _resuming:
    open(metrics_path, "w").close()  # fresh loss log for the dashboard

  # Load ONLY the conditional transformer.
  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  device = torch.device(cfg.runtime.device)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype)
  del sd
  print(f"[cached] transformer loaded in {time.time()-t0:.1f}s "
        f"(no text encoder / VAE)", flush=True)

  wrapped = loramod.inject_lora(transformer, rank=rank)
  params = loramod.lora_parameters(wrapped)
  if grad_ckpt:
    transformer.gradient_checkpointing = True
  ema = LoraEMA(wrapped, decay=ema_decay) if use_ema else None
  print(f"[cached] LoRA rank {rank}: {len(wrapped)} modules, "
        f"{sum(p.numel() for p in params)/1e6:.1f}M params | batch={batch} accum={accum} "
        f"grad_ckpt={grad_ckpt} optim={optim_name} res={res} ema={use_ema} "
        f"masked={masked} ts_shift={ts_shift} ts_w={ts_weighting}", flush=True)
  opt = build_optimizer(optim_name, params, lr)
  sf = is_schedule_free(optim_name)   # schedule-free: own LR schedule + train/eval modes
  if sf:
    opt.train()

  sched_lr = build_lr_scheduler(
    opt, scheduler=cfg.optim.lr_scheduler, warmup=warmup, total_steps=steps,
    num_restarts=int(cfg.optim.num_restarts), min_lr_ratio=float(cfg.optim.min_lr_ratio),
  )
  schedule = get_schedule_for_resolution((res, res), known_mean=1.0)

  # Lazy cache index: hold only FILENAMES (not tensors) so CPU RAM stays flat
  # regardless of dataset size -- preloading 72.7K caches is ~575GB, the full 257K
  # ~2TB. Bucket by grid via an mmap scan (reads the small int fields, never the
  # 7.9MB tensors), then load each batch's .pt from NVMe on demand. idx >= n_eval
  # train; idx < n_eval are the held-out validation set (never trained).
  files = sorted(f for f in os.listdir(cache) if f.endswith(".pt"))
  # Optional curated training list (filenames; REPEATS oversample an edit type).
  # Held-out eval files always come from the cache dir itself (idx < n_eval).
  train_pool = files
  if cfg.data.train_list:
    train_pool = json.load(open(cfg.data.train_list))
    print(f"[cached] train_list: {len(train_pool)} entries "
          f"({len(set(train_pool))} unique) from {cfg.data.train_list}", flush=True)
  from collections import defaultdict
  # Persisted bucket index ({filename: [gh,gw,size,mtime_ns]}) so a restart re-probes only
  # NEW or CHANGED files instead of mmap-scanning all 257K every time (~10min -> instant on
  # warm restarts). The (size, mtime_ns) stamp guards against a stale grid surviving a
  # re-precache: if the .pt was rewritten, the cached grid is re-probed so a later .view()
  # cannot crash on a wrong grid. Unchanged caches return the identical grid, so bucketing
  # and sampling stay bit-identical.
  idx_path = os.path.join(cache, ".bucket_index.json")
  try:
    prev_idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
  except Exception:
    prev_idx = {}
  bucket_files = defaultdict(list)
  eval_files = [f for f in files if int(f[:-3]) < n_eval]
  n_unreadable, n_new = 0, 0
  for f in train_pool:
    if int(f[:-3]) < n_eval:
      continue
    fpath = f"{cache}/{f}"
    try:  # stat to validate the cached entry; skip files a concurrent precache deleted
      st = os.stat(fpath)
      size, mtime_ns = int(st.st_size), int(st.st_mtime_ns)
    except OSError:
      n_unreadable += 1
      continue
    ent = prev_idx.get(f)
    # Trust the cache only for a 4-field entry whose (size, mtime_ns) still matches the
    # file on disk. Legacy [gh,gw]-only entries (len 2) and mismatches force a re-probe.
    if isinstance(ent, list) and len(ent) == 4 and ent[2] == size and ent[3] == mtime_ns:
      gh_gw = [int(ent[0]), int(ent[1])]
    else:
      try:  # tensors mmap'd not read; skip a .pt a concurrent precache is mid-writing
        hdr = torch.load(fpath, map_location="cpu", mmap=True)
        gh_gw = [int(hdr["grid_h"]), int(hdr["grid_w"])]
        del hdr  # drop the mmap handle immediately (refcounted) so FDs don't accumulate
        prev_idx[f] = [gh_gw[0], gh_gw[1], size, mtime_ns]
        n_new += 1
      except Exception:
        n_unreadable += 1
        continue
    bucket_files[tuple(gh_gw)].append(f)
  if n_new:  # persist the (incrementally extended) index for instant future restarts
    try:
      json.dump(prev_idx, open(idx_path, "w"))
    except Exception:
      pass
  if n_unreadable:
    print(f"[cached] skipped {n_unreadable} unreadable/partial caches (concurrent precache?)", flush=True)
  bucket_keys = list(bucket_files.keys())
  bucket_weights = [len(bucket_files[k]) for k in bucket_keys]
  n_train = sum(bucket_weights)
  print(f"[cached] {n_train} training + {len(eval_files)} held-out caches "
        f"(lazy from disk, {len(bucket_keys)} AR bucket(s), val {'on' if val_every else 'off'})", flush=True)

  transformer.train()
  seed = int(cfg.runtime.seed)  # default 0 -> byte-identical to the old hardcoded seed
  gen = torch.Generator(device=device).manual_seed(seed)
  rng = random.Random(seed)
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  run, t_last = 0.0, time.time()

  def _to_dev(c):
    return {"grid_h": c["grid_h"], "grid_w": c["grid_w"],
            "z_ref": c["z_ref"].to(device), "z_tgt": c["z_tgt"].to(device),
            "llm_text": c["llm_text"].to(device)}

  def _load_dev(f):  # read one .pt from disk -> device (per-batch; keeps RAM flat)
    return _to_dev(torch.load(f"{cache}/{f}", map_location="cpu"))

  # Each batch is grid-homogeneous (the batched step requires a single grid; MRoPE
  # handles non-square grids fine) -- pick a bucket, then sample files within it.
  def sample_batch():
    k = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
    pool = bucket_files[k]
    return [_load_dev(rng.choice(pool)) for _ in range(batch)]

  # Held-out, small -> load once to device. Eval never feeds gradients, so a truncated
  # .pt must NOT abort startup: drop the bad item and keep going (log how many dropped).
  eval_items, eval_kept = [], []
  n_eval_dropped = 0
  for f in eval_files:
    try:
      eval_items.append(_load_dev(f))
      eval_kept.append(f)
    except Exception:
      n_eval_dropped += 1
  if n_eval_dropped:
    print(f"[cached] dropped {n_eval_dropped} unreadable held-out eval cache(s)", flush=True)
  eval_files = eval_kept  # keep the prompt sidecar aligned with the items we actually loaded

  # Prompt sidecar for the dashboard: {idx: instruction} for the held-out previews.
  try:
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    _pr = {}
    for f in eval_files:
      try:  # one bad preview cache must not lose the whole sidecar
        _pr[str(int(f[:-3]))] = str(torch.load(f"{cache}/{f}", map_location="cpu", mmap=True).get("edit", ""))
      except Exception:
        continue
    json.dump(_pr, open(os.path.join(output_dir, "samples", "prompts.json"), "w"), ensure_ascii=False)
  except Exception:
    pass

  n_skipped = 0

  def _meta(step_num):
    return {"base_model": weights, "type": "ideogram4-edit-lora", "rank": rank,
            "step": step_num, "optimizer": optim_name, "ema": use_ema}

  sample_dir = os.path.join(output_dir, "samples")
  decoder_state = {"d": None}  # lazily-loaded (autoencoder, shift, scale, patch_size)

  base_z = {"v": None}  # frozen-base (LoRA-disabled) z_out per held-out item -- computed ONCE

  def _label(img, text):  # black caption strip in the top-left of each panel
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, max(46, 7 * len(text) + 8), 18], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(245, 245, 245))
    return img

  def _gen(it, seed, lora_on):
    gh, gw = int(it["grid_h"]), int(it["grid_w"])
    fn = lambda: edit_sampler.sample_edit_cached(
      transformer, it["z_ref"], it["llm_text"], gh, gw, schedule=schedule,
      num_steps=int(cfg.logging.sample_steps), guidance_scale=float(cfg.logging.sample_guidance),
      generator=torch.Generator(device=device).manual_seed(seed))
    if lora_on:
      return fn()
    with loramod.lora_disabled(transformer):  # frozen base, no adapter
      return fn()

  def _sample(step_num):
    """Decode labeled held-out [source | base | LoRA | target]. Base = LoRA-off, once.

    try/finally is CRITICAL: if sampling raises while the transformer is in eval mode,
    grad-checkpointing stays disabled and the next training step stores full activations
    -> OOM. So train mode is always restored, the decoder freed (training VRAM stays flat).
    """
    from PIL import Image
    os.makedirs(sample_dir, exist_ok=True)
    if sf:
      opt.eval()
    transformer.eval()
    try:
      with torch.no_grad():
        if decoder_state["d"] is None:
          decoder_state["d"] = edit_sampler.load_decoder(weights, device, dtype)
        ae, shift, scale, patch = decoder_state["d"]
        if base_z["v"] is None:  # one-time frozen-base reference (computed with LoRA off)
          base_z["v"] = [_gen(it, 12345, lora_on=False) for it in eval_items]
        for j, it in enumerate(eval_items):
          gh, gw = int(it["grid_h"]), int(it["grid_w"])
          z_out = _gen(it, step_num, lora_on=True)
          quad = torch.cat([it["z_ref"].unsqueeze(0), base_z["v"][j], z_out, it["z_tgt"].unsqueeze(0)], dim=0)
          imgs = edit_sampler.decode_latents(ae, quad, gh, gw, patch_size=patch,
                                             latent_shift=shift, latent_scale=scale, dtype=dtype)
          for im, lab in zip(imgs, ("source", "base", "LoRA", "target")):
            _label(im, lab)
          w, h = imgs[0].size
          canvas = Image.new("RGB", (w * 4, h))
          for k, im in enumerate(imgs):
            canvas.paste(im, (k * w, 0))  # source | base | LoRA | target
          canvas.save(os.path.join(sample_dir, f"step{step_num:06d}_idx{j}.png"))
      print(f"[cached] sampled {len(eval_items)} held-out [src|base|LoRA|tgt] @ step {step_num}", flush=True)
    finally:
      decoder_state["d"] = None  # free the VAE so training VRAM stays flat (reloaded next sample)
      transformer.train()
      if sf:
        opt.train()
      if device.type == "cuda":
        torch.cuda.empty_cache()

  def _save(tag, step_num):
    if sf:
      opt.eval()  # swap to schedule-free averaged weights before reading them
    loramod.save_lora(wrapped, f"{ckpt}/edit_lora_rank{rank}_{tag}.safetensors", metadata=_meta(step_num))
    if ema is not None:
      with ema.swap_in():
        loramod.save_lora(wrapped, f"{ckpt}/edit_lora_rank{rank}_{tag}_ema.safetensors", metadata=_meta(step_num))
    if sf:
      opt.train()
    # Full training state for exact resume (latest only, overwritten each save).
    save_training_state(
      f"{ckpt}/resume.pt", step=step_num, optimizer=opt, scheduler=sched_lr,
      wrapped=wrapped, ema=ema, gen=gen, extra={"n_skipped": n_skipped, "rng": rng.getstate()},
    )

  # Resume from a prior full-state checkpoint, if requested.
  start_step = 0
  resume_from = cfg.paths.resume_from
  if resume_from == "auto":
    resume_from = f"{ckpt}/resume.pt"
  if resume_from and os.path.exists(resume_from):
    start_step, ex = load_training_state(
      resume_from, optimizer=opt, scheduler=sched_lr, wrapped=wrapped,
      ema=ema, gen=gen, map_location=device,
    )
    n_skipped = ex.get("n_skipped", 0)
    if "rng" in ex:
      rng.setstate(ex["rng"])
    print(f"[cached] RESUMED from {resume_from} at step {start_step}", flush=True)

  for step in range(start_step, steps):
    opt.zero_grad()
    acc = 0.0
    skip = False
    for _ in range(accum):
      loss = train_edit.edit_training_step_cached_batch(
        transformer, sample_batch(),
        schedule=schedule, cfg_dropout_prob=cfg_drop, generator=gen,
        timestep_shift=ts_shift, timestep_weighting=ts_weighting,
        min_snr_gamma=min_snr_gamma, masked_loss=masked, mask_quantile=mask_q,
        mask_bg_weight=mask_bg,
        noise_offset=noise_offset, input_perturbation=input_perturbation,
        prior_preservation_weight=prior_pres,
      )
      if nan_guard and not is_finite_loss(loss):
        skip = True
        break
      (loss / accum).backward()
      acc += loss.item() / accum
    if skip:                       # non-finite loss: drop grads, skip the step
      opt.zero_grad(set_to_none=True)
      n_skipped += 1
      continue
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    opt.step()
    if not sf:                       # schedule_free manages its own LR internally
      sched_lr.step()
    if ema is not None:
      ema.update()
    run += acc
    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step + 1, "loss": run / log_every,
             "lr": sched_lr.get_last_lr()[0], "s_per_step": dt,
             "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "skipped": n_skipped}
      # Held-out validation loss (deterministic, low-variance) on the val cadence.
      if val_every and eval_items and (step + 1) % val_every == 0:
        if sf:
          opt.eval()
        rec["val_loss"] = compute_val_loss(transformer, eval_items, schedule, device=device)
        if sf:
          opt.train()
      print(f"[cached] step {step+1}/{steps} loss {rec['loss']:.4f} "
            + (f"val {rec['val_loss']:.4f} " if "val_loss" in rec else "")
            + f"lr {rec['lr']:.2e} | {dt:.2f}s/step "
            f"peak {rec['peak_gb']:.1f}GB skipped={n_skipped}", flush=True)
      with open(metrics_path, "a") as mf:
        mf.write(json.dumps(rec) + "\n")
      run, t_last = 0.0, time.time()
    if (step + 1) % ckpt_every == 0:
      _save(f"step{step+1}", step + 1)
      print(f"[cached] checkpoint @ step {step+1}", flush=True)
    if sample_every and eval_items and (step + 1) % sample_every == 0:
      try:
        _sample(step + 1)
      except Exception as exc:  # sampling must never crash training
        print(f"[cached] sampling skipped @ step {step+1}: {exc}", flush=True)

  _save("final", steps)
  print(f"[cached] saved final -> {ckpt}/edit_lora_rank{rank}_final.safetensors "
        f"(ema={use_ema}, skipped {n_skipped} non-finite steps)", flush=True)
  print("[cached] DONE", flush=True)


if __name__ == "__main__":
  main()
