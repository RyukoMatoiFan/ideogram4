"""Encoder-free FULL fine-tune of the transformer for plain text-to-image (e.g. an anime
style/character retrain), on a single 80GB GPU (or 24GB with optim.offload_optimizer).

The T2I analogue of train_edit_full_cached.py: same fp8->bf16 dequant + fused-back-pass
AdamW + stochastic rounding engine, but the loss is the plain ``[text][target]`` flow-
matching step (no reference frame, no edit mask) and previews are pure text-to-image.

Full fine-tune (not LoRA) because an anime retrain is a whole-distribution domain shift:
the adapter capacity that suffices for a few concepts caps out moving the entire output
manifold toward an art style.

  CUDA_VISIBLE_DEVICES=0 python train_t2i_full_cached.py --config config/t2i_full.yaml
  Smoke: add --smoke. 24GB: set optim.offload_optimizer + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
from ideogram4 import train_edit, train_t2i, edit_sampler
from ideogram4.fused_adamw import build_fused_adamw, _sr_seed
from ideogram4.training_utils import is_finite_loss, build_lr_scheduler


def _marker(ckpt):
  return f"{ckpt}/resume_t2i_full.json"


def main():
  ap = argparse.ArgumentParser(description="Encoder-free FULL fine-tune for T2I from caches.")
  ap.add_argument("--config", default="config/t2i_full.yaml")
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
  steps = int(cfg.optim.steps)
  batch = int(cfg.optim.batch)
  warmup = int(cfg.optim.warmup)
  grad_clip = float(cfg.optim.grad_clip)
  cfg_drop = float(cfg.optim.cfg_dropout_prob)
  ts_shift = float(cfg.flow.timestep_shift)
  ts_weighting = cfg.flow.timestep_weighting
  min_snr_gamma = float(cfg.flow.min_snr_gamma)
  noise_offset = float(cfg.flow.noise_offset)
  n_eval = int(cfg.data.n_eval_holdout)
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  sample_every = int(cfg.logging.sample_every)
  val_every = int(cfg.logging.val_every)
  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)

  os.makedirs(ckpt, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
  sample_dir = os.path.join(output_dir, "samples")
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  from ideogram4.trackers import Tracker
  tracker = Tracker(cfg.logging.tracker, project=cfg.logging.wandb_project,
                    run_name=cfg.logging.run_name or None, out_dir=output_dir)
  if not (cfg.paths.resume_from and os.path.exists(_marker(ckpt))):
    open(metrics_path, "w").close()

  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype); del sd
  print(f"[t2i-full] fp8 transformer loaded in {time.time()-t0:.1f}s", flush=True)

  t1 = time.time()
  train_edit.dequantize_fp8_transformer(transformer, dtype=dtype)  # NO expand_reference_embedding (T2I)
  transformer.to(device); transformer.requires_grad_(True); transformer.train()
  transformer.gradient_checkpointing = True
  n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
  print(f"[t2i-full] dequantized fp8->bf16 in {time.time()-t1:.1f}s | {n_params/1e9:.2f}B params", flush=True)

  _sr_seed(cfg.runtime.seed, device)
  offload = bool(cfg.optim.offload_optimizer)
  # Opt-in (DEFAULT OFF -> behaviour bit-identical when absent): persist AdamW moments so a
  # restart is a WARM resume (no Adam re-warmup transient). Single file, overwritten each
  # ckpt -> only the LATEST optimizer state is ever kept (moments are ~4x the bf16 weights;
  # one matching snapshot is all a resume can use).
  save_opt = bool(getattr(cfg.optim, "save_optimizer", False))
  opt_path = f"{ckpt}/t2i_full_optimizer.pt"
  opt = build_fused_adamw([p for p in transformer.parameters() if p.requires_grad], lr,
                          stochastic_rounding=True, offload_states=offload)
  print(f"[t2i-full] fused AdamW + SR | states {'CPU(offload)' if offload else 'GPU'}", flush=True)
  sched_lr = build_lr_scheduler(opt, scheduler=cfg.optim.lr_scheduler, warmup=warmup,
                                total_steps=steps, num_restarts=int(cfg.optim.num_restarts),
                                min_lr_ratio=float(cfg.optim.min_lr_ratio))
  accum = int(cfg.optim.accum)
  fused_backward = (accum == 1)  # fused per-param backward only works with no coexisting grads
  handles = []
  if fused_backward:
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
    print(f"[t2i-full] fused back pass: {len(handles)} hooks", flush=True)
  else:
    print(f"[t2i-full] grad-accum x{accum}: standard backward + manual SR step "
          f"(grads coexist -> higher VRAM than accum=1)", flush=True)

  def _manual_opt_step():
    # Mirror the fused per-param hook (incl. stochastic rounding) AFTER accumulation.
    for group in opt.param_groups:
      for i, p in enumerate(group["params"]):
        if p.grad is None:
          continue
        if grad_clip:
          torch.nn.utils.clip_grad_norm_(p, grad_clip)
        opt.step_parameter(p, group, i)
        p.grad = None

  blocks_to_swap = int(getattr(cfg.optim, "blocks_to_swap", 0))
  if blocks_to_swap > 0:
    if fused_backward:
      print("[t2i-full] WARN: block-swap + accum==1 (fused backward) is unvalidated; "
            "prefer accum>1", flush=True)
    from ideogram4.block_swap import enable_block_swap
    ns = enable_block_swap(transformer, blocks_to_swap, device)
    print(f"[t2i-full] block-swap: {ns} deepest blocks offloaded to CPU (needs GPU smoke test)", flush=True)

  schedule = get_schedule_for_resolution(
    (res, res), known_mean=cfg.flow.schedule_mean, std=cfg.flow.schedule_std)

  files = sorted(f for f in os.listdir(cache) if f.endswith(".pt"))
  eval_files = [f for f in files if int(f[:-3]) < n_eval]
  # Grid-bucketed (aspect bucketing): each batch must be grid-homogeneous, so group train
  # files by (gh, gw) and sample a bucket per batch. Lazy index via .bucket_index.json
  # (mmap-probe only NEW files) so restarts are instant.
  from collections import defaultdict
  idx_path = os.path.join(cache, ".bucket_index.json")
  try:
    prev_idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
  except Exception:
    prev_idx = {}
  bucket_files = defaultdict(list)
  n_new = 0
  for f in files:
    if int(f[:-3]) < n_eval:
      continue
    try:
      st = os.stat(f"{cache}/{f}")
      sig = [int(st.st_size), int(st.st_mtime_ns)]
    except OSError:
      continue
    ent = prev_idx.get(f)
    # Validate cached entry against (size, mtime_ns); re-probe legacy ([gh,gw] only)
    # or stale entries. Identical caches -> identical bucketing/grid.
    gh_gw = None
    if isinstance(ent, list) and len(ent) == 4 and ent[2:] == sig:
      gh_gw = ent[:2]
    if gh_gw is None:
      try:
        hdr = torch.load(f"{cache}/{f}", map_location="cpu", mmap=True)
        gh_gw = [int(hdr["grid_h"]), int(hdr["grid_w"])]; del hdr
        prev_idx[f] = gh_gw + sig; n_new += 1
      except Exception:
        continue
    bucket_files[tuple(gh_gw)].append(f)
  if n_new:
    try:
      json.dump(prev_idx, open(idx_path, "w"))
    except Exception:
      pass
  bucket_keys = list(bucket_files.keys())
  bucket_weights = [len(bucket_files[k]) for k in bucket_keys]
  print(f"[t2i-full] {sum(bucket_weights)} train + {len(eval_files)} held-out caches "
        f"({len(bucket_keys)} AR bucket(s))", flush=True)

  gen = torch.Generator(device=device).manual_seed(cfg.runtime.seed)
  rng = random.Random(cfg.runtime.seed)
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

  # Opt-in (DEFAULT ON) RAM preload: read every TRAINING-pool cache file ONCE into CPU
  # tensors, so each step is a dict lookup instead of a disk read. The bucket-index loop
  # above only mmap-probes the (gh,gw) header (and skips entirely on a warm restart), so
  # do a dedicated deterministic pass over bucket_files to load the payloads. This does
  # NOT touch the rng: sample_batch() consumes rng exactly as before -> bit-identical
  # batch sequence and gradients.
  preload = bool(getattr(cfg.optim, "preload_caches", True))
  _cpu_cache = {}
  if preload:
    t_pre = time.time()
    for k in bucket_keys:                 # deterministic: fixed insertion order
      for f in bucket_files[k]:
        if f in _cpu_cache:
          continue
        try:
          c = torch.load(f"{cache}/{f}", map_location="cpu")
          _cpu_cache[f] = {"grid_h": c["grid_h"], "grid_w": c["grid_w"],
                           "z_tgt": c["z_tgt"], "llm_text": c["llm_text"]}
        except Exception:
          pass
    print(f"[t2i-full] preloaded {len(_cpu_cache)} train caches into RAM "
          f"in {time.time()-t_pre:.1f}s", flush=True)

  def _read_cpu(f):
    c = _cpu_cache.get(f)
    if c is None:
      c = torch.load(f"{cache}/{f}", map_location="cpu")
    return c

  def _load(f):
    # Per-batch H2D: pin only the (small, batch-sized) source tensors and copy async on
    # the default stream. Default-stream consumers serialize after the copy -> identical
    # bytes vs a plain .to(device). Pinning is per-tensor (bounded), not on the dataset.
    c = _read_cpu(f)
    z_tgt = c["z_tgt"]; llm = c["llm_text"]
    if device.type == "cuda":
      z_tgt = z_tgt.pin_memory().to(device, non_blocking=True)
      llm = llm.pin_memory().to(device, non_blocking=True)
    else:
      z_tgt = z_tgt.to(device); llm = llm.to(device)
    return {"grid_h": c["grid_h"], "grid_w": c["grid_w"],
            "z_tgt": z_tgt, "llm_text": llm}

  def sample_batch():
    k = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
    pool = bucket_files[k]
    return [_load(rng.choice(pool)) for _ in range(batch)]

  # Guard eval/preview cache reads: a single truncated .pt must not abort the run before
  # step 0. Drop+log the offending file (training-pool load above stays strict).
  eval_items = []
  for f in eval_files:
    try:
      eval_items.append(_load(f))
    except Exception as e:
      print(f"[t2i-full] WARN dropped held-out cache {f}: {e}", flush=True)

  # Curated preview dashboard: T2I-only generations from hand-written capability-probe
  # captions (llm_text precomputed offline into preview_dir/*.pt). Falls back to held-out
  # [T2I|target] pairs when no preview_dir is set.
  preview_items = []
  pdir = cfg.logging.preview_dir
  if pdir and os.path.isdir(pdir):
    for pf in sorted(f for f in os.listdir(pdir) if f.endswith(".pt")):
      try:
        c = torch.load(f"{pdir}/{pf}", map_location="cpu")
        preview_items.append({"grid_h": c["grid_h"], "grid_w": c["grid_w"],
                              "llm_text": c["llm_text"].to(device), "tags": c.get("tags", "")})
      except Exception as e:
        print(f"[t2i-full] WARN dropped preview cache {pf}: {e}", flush=True)
    print(f"[t2i-full] {len(preview_items)} curated preview prompts from {pdir}", flush=True)
    # Tags -> samples/prompts.json so the dashboard can show each prompt on hover.
    try:
      json.dump({str(i): it.get("tags", "") for i, it in enumerate(preview_items)},
                open(os.path.join(sample_dir, "prompts.json"), "w"), ensure_ascii=False)
    except Exception:
      pass
  decoder_state = {"d": None}
  # Deterministic held-out validation loss: mean flow-matching loss over the held-out
  # captions at FIXED timesteps (no sampling noise) -> a low-variance curve to read the
  # plateau / overfit onset from. Never-trained items (idx < n_eval excluded from train).
  VAL_TS = [0.25, 0.5, 0.75]

  @torch.no_grad()
  def _val_loss():
    was_training = transformer.training
    transformer.eval()
    try:
      vg = torch.Generator(device=device).manual_seed(1234)
      total, count = 0.0, 0
      for it in eval_items:
        for tv in VAL_TS:
          l = train_t2i.t2i_training_step_cached_batch(
            transformer, [it], schedule=schedule, cfg_dropout_prob=0.0, generator=vg,
            timestep_shift=ts_shift, t_override=torch.tensor(tv))
          total += float(l); count += 1
      return total / max(count, 1)
    finally:
      if was_training:
        transformer.train()

  def _label(img, text):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img); d.rectangle([0, 0, max(46, 7*len(text)+8), 18], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(245, 245, 245)); return img

  def _sample(step_num):
    from PIL import Image
    transformer.eval()
    try:
      with torch.no_grad():
        if decoder_state["d"] is None:
          decoder_state["d"] = edit_sampler.load_decoder(weights, device, dtype)
        ae, shift, scale, patch = decoder_state["d"]
        def _gen(it, i):
          gh, gw = int(it["grid_h"]), int(it["grid_w"])
          # FIXED per-prompt seed (cfg.runtime.seed + prompt index), NOT the step number, so
          # previews are comparable step-to-step (same noise across checkpoints -> the only
          # change you see is the model). Matches the seed-0 snapshot filmstrip convention.
          z = edit_sampler.sample_t2i(transformer, it["llm_text"], gh, gw, schedule=schedule,
                                      num_steps=int(cfg.logging.sample_steps),
                                      guidance_scale=float(cfg.logging.sample_guidance),
                                      generator=torch.Generator(device=device).manual_seed(cfg.runtime.seed + i))
          return gh, gw, z
        if preview_items:  # curated capability dashboard: T2I-only contact sheet
          thumbs = []
          for i, it in enumerate(preview_items):
            gh, gw, z = _gen(it, i)
            im = edit_sampler.decode_latents(ae, z, gh, gw, patch_size=patch,
                                             latent_shift=shift, latent_scale=scale, dtype=dtype)[0]
            thumbs.append(_label(im, str(it.get("tags", ""))[:34]))
          cols, tw = 4, 360
          scaled = [t.resize((tw, max(1, int(t.height * tw / t.width)))) for t in thumbs]
          ch = max(s.height for s in scaled)
          rows = (len(scaled) + cols - 1) // cols
          sheet = Image.new("RGB", (cols * tw, rows * ch), (15, 15, 15))
          for k, s in enumerate(scaled):
            sheet.paste(s, ((k % cols) * tw, (k // cols) * ch))
          sheet.save(os.path.join(sample_dir, f"step{step_num:06d}_dashboard.png"))
          print(f"[t2i-full] preview dashboard ({len(thumbs)}) @ {step_num}", flush=True)
        else:  # fallback: held-out [T2I|target]
          for j, it in enumerate(eval_items[:int(cfg.logging.sample_count)]):
            gh, gw, z = _gen(it, j)
            pair = torch.cat([z, it["z_tgt"].unsqueeze(0)], dim=0)
            imgs = edit_sampler.decode_latents(ae, pair, gh, gw, patch_size=patch,
                                               latent_shift=shift, latent_scale=scale, dtype=dtype)
            for im, lab in zip(imgs, ("T2I", "target")):
              _label(im, lab)
            w, h = imgs[0].size
            canvas = Image.new("RGB", (w * 2, h))
            for k, im in enumerate(imgs):
              canvas.paste(im, (k * w, 0))
            canvas.save(os.path.join(sample_dir, f"step{step_num:06d}_idx{j}.png"))
          print(f"[t2i-full] sampled held-out [T2I|target] @ {step_num}", flush=True)
    finally:
      decoder_state["d"] = None; transformer.train()
      if device.type == "cuda":
        torch.cuda.empty_cache()

  def _save(tag, step_num, keep_last=2):
    from safetensors.torch import save_file
    state = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in transformer.state_dict().items()}
    path = f"{ckpt}/t2i_full_{tag}.safetensors"
    save_file(state, path, metadata={"base_model": weights, "type": "ideogram4-t2i-full", "step": str(step_num)})
    json.dump({"step": step_num, "ckpt": os.path.basename(path)}, open(_marker(ckpt), "w"))
    print(f"[t2i-full] saved {path} @ {step_num}", flush=True)
    if save_opt:
      # Per-param moments aligned to opt.param_groups order (stable: optimizer is rebuilt
      # identically on resume). Moments -> CPU for a portable, device-agnostic blob.
      blobs = []
      for grp in opt.param_groups:
        for p in grp["params"]:
          st = opt.state.get(p)
          blobs.append(None if not st else {"step": int(st["step"]),
                       "exp_avg": st["exp_avg"].to("cpu"),
                       "exp_avg_sq": st["exp_avg_sq"].to("cpu")})
      tmp = opt_path + ".tmp"
      torch.save({"step": step_num, "blobs": blobs}, tmp)
      os.replace(tmp, opt_path)  # atomic; single file -> only the LATEST state survives
      print(f"[t2i-full] saved optimizer state @ {step_num}", flush=True)
    # Full-model bf16 ckpts are ~19GB: rotate step-tagged ones (never "final"/named tags)
    import re as _re
    stepped = sorted(f for f in os.listdir(ckpt)
                     if _re.fullmatch(r"t2i_full_step\d{6}\.safetensors", f))
    for old in stepped[:-keep_last]:
      os.remove(os.path.join(ckpt, old))
      print(f"[t2i-full] rotated out {old}", flush=True)

  start_step = 0
  if cfg.paths.resume_from and os.path.exists(_marker(ckpt)):
    mk = json.load(open(_marker(ckpt)))
    from safetensors.torch import load_file
    transformer.load_state_dict({k: v.to(device, dtype) for k, v in load_file(f"{ckpt}/{mk['ckpt']}").items()}, strict=True)
    start_step = int(mk["step"])
    for _ in range(start_step):
      sched_lr.step()
    print(f"[t2i-full] RESUMED at step {start_step}", flush=True)
    if save_opt and os.path.exists(opt_path):
      # Pre-populate opt.state BEFORE the first hook fires: step_adamw_parameter inits only
      # when len(state)==0, so restored moments are used verbatim (warm resume).
      payload = torch.load(opt_path, map_location="cpu")
      blobs = payload["blobs"]; bi = 0
      for grp in opt.param_groups:
        for p in grp["params"]:
          b = blobs[bi]; bi += 1
          if b is None:
            continue
          dev = "cpu" if offload else p.device
          st = opt.state[p]
          st["step"] = int(b["step"])
          st["exp_avg"] = b["exp_avg"].to(dev)
          st["exp_avg_sq"] = b["exp_avg_sq"].to(dev)
      print(f"[t2i-full] RESUMED optimizer state (warm) from step {payload.get('step')}", flush=True)
    elif save_opt:
      print("[t2i-full] WARN save_optimizer on but no optimizer file -> COLD optimizer", flush=True)

  os.makedirs(sample_dir, exist_ok=True)
  def _train_step():
    return train_t2i.t2i_training_step_cached_batch(
      transformer, sample_batch(), schedule=schedule, cfg_dropout_prob=cfg_drop, generator=gen,
      timestep_shift=ts_shift, timestep_weighting=ts_weighting, min_snr_gamma=min_snr_gamma,
      noise_offset=noise_offset)

  n_skipped = 0; run, t_last = 0.0, time.time()
  for step in range(start_step, steps):
    if fused_backward:
      loss = _train_step()
      if not is_finite_loss(loss):
        n_skipped += 1; sched_lr.step(); continue
      loss.backward(); sched_lr.step(); run += loss.item()
    else:
      acc_loss, ok = 0.0, True
      for _ in range(accum):  # accumulate accum micro-batches, then one SR step
        l = _train_step()
        if not is_finite_loss(l):
          ok = False; break
        (l / accum).backward(); acc_loss += l.item() / accum
      if not ok:
        for grp in opt.param_groups:
          for p in grp["params"]:
            p.grad = None
        n_skipped += 1; sched_lr.step(); continue
      _manual_opt_step(); sched_lr.step(); run += acc_loss

    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step+1, "loss": run/log_every, "lr": sched_lr.get_last_lr()[0],
             "s_per_step": dt, "peak_gb": torch.cuda.max_memory_allocated()/1e9, "skipped": n_skipped}
      if val_every and eval_items and (step + 1) % val_every == 0:
        rec["val_loss"] = _val_loss()
      vmsg = f" val {rec['val_loss']:.4f}" if "val_loss" in rec else ""
      print(f"[t2i-full] step {step+1}/{steps} loss {rec['loss']:.4f}{vmsg} lr {rec['lr']:.2e} | "
            f"{dt:.2f}s/step peak {rec['peak_gb']:.1f}GB", flush=True)
      with open(metrics_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
      tracker.log(rec, rec["step"])
      run, t_last = 0.0, time.time()
      if args.smoke and step + 1 >= 30:
        print("[t2i-full] SMOKE OK", flush=True); return
    if sample_every and (step + 1) % sample_every == 0:
      _sample(step + 1)
    if ckpt_every and (step + 1) % ckpt_every == 0:
      _save(f"step{step+1:06d}", step + 1)

  _save("final", steps); print("[t2i-full] DONE", flush=True)


if __name__ == "__main__":
  main()
