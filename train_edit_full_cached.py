"""Encoder-free FULL fine-tuning of the edit transformer on a single 80GB GPU.

Removes the LoRA capacity ceiling diagnosed for structural edits (object removal /
background replacement). Trains ALL transformer parameters, not a low-rank adapter.

Fits 9.3B on one 80GB GPU via two memory tricks (see ideogram4.fused_adamw):
  * bf16 weights + optimizer states with **stochastic rounding** -> no fp32 master.
  * **fused back pass**: each parameter's AdamW update runs in a post-accumulate-grad
    hook the instant its grad is ready, then the grad is freed -> full-model gradients
    never coexist. Requires accum == 1 (accumulation would keep grads alive).

Memory (9.3B): weights 18.6GB + states 37GB + grads ~0 + acts(grad-ckpt) ~few -> ~58GB.
With ``optim.offload_optimizer`` the Adam states move to CPU RAM -> measured ~21GiB peak
(batch 1), i.e. a single 24GB GPU full-fine-tunes the whole 9.3B model -- at ~10x slower
steps (CPU Adam + per-param PCIe). Needs ~74GB host RAM + expandable_segments alloc.

Base is published fp8 only -> we dequantize fp8->bf16 first (train_edit.dequantize_fp8_
transformer; exact for the stored fp8 values, lossy vs Ideogram's pre-quant weights).
Use a LOW lr (~1e-5): we adapt a pretrained model that already carries quant noise.

  CUDA_VISIBLE_DEVICES=0 python train_edit_full_cached.py --config config/edit_full.yaml
Set real paths in config/local.yaml or via IG4_* env overrides.
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
from ideogram4 import edit_sampler
from ideogram4.fused_adamw import build_fused_adamw, _sr_seed
from ideogram4.training_utils import is_finite_loss, build_lr_scheduler


def main():
  parser = argparse.ArgumentParser(description="Encoder-free FULL fine-tune from caches.")
  parser.add_argument("--config", default="config/edit_full.yaml")
  parser.add_argument("--smoke", action="store_true",
                      help="Verify dequant equivalence + a few steps, then exit.")
  parser.add_argument("--cap-gb", type=float, default=0.0,
                      help="Cap this process's CUDA memory (GB) to prove a VRAM target, e.g. 24.")
  args = parser.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)

  if args.cap_gb and torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(args.cap_gb / total, 1.0), 0)
    print(f"[full] CUDA memory capped at {args.cap_gb:.0f}GB of {total:.0f}GB "
          f"(OOM if the run exceeds it -> proves the target)", flush=True)

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
  input_perturbation = float(cfg.flow.input_perturbation)
  masked = bool(cfg.data.masked_loss)
  mask_q = float(cfg.data.mask_quantile)
  mask_bg = float(cfg.data.mask_bg_weight)
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  sample_every = int(cfg.logging.sample_every)
  n_eval = int(cfg.data.n_eval_holdout)
  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)

  os.makedirs(ckpt, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  sample_dir = os.path.join(output_dir, "samples")
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  if not (cfg.paths.resume_from and os.path.exists(_resume_marker(ckpt))):
    open(metrics_path, "w").close()

  # --- load + dequantize the conditional transformer ---
  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype)
  del sd
  print(f"[full] fp8 transformer loaded in {time.time()-t0:.1f}s", flush=True)

  if args.smoke:
    _verify_dequant_equivalence(transformer, device)

  t1 = time.time()
  train_edit.dequantize_fp8_transformer(transformer, dtype=dtype)
  train_edit.expand_reference_embedding(transformer)  # reference tokens get their own indicator slot
  transformer.to(device)
  transformer.requires_grad_(True)
  transformer.train()
  transformer.gradient_checkpointing = True  # mandatory for full-FT activations
  n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
  print(f"[full] dequantized fp8->bf16 + expanded ref embed in {time.time()-t1:.1f}s | "
        f"{n_params/1e9:.2f}B trainable params, grad_ckpt=on", flush=True)

  _sr_seed(cfg.runtime.seed, device)
  offload = bool(cfg.optim.offload_optimizer)
  opt = build_fused_adamw(
    [p for p in transformer.parameters() if p.requires_grad], lr,
    weight_decay=float(cfg.optim.weight_decay),
    stochastic_rounding=True, offload_states=offload,
  )
  print(f"[full] optimizer: fused AdamW + stochastic rounding | "
        f"states {'on CPU (offload, ~21GB VRAM)' if offload else 'on GPU (~58GB VRAM)'}", flush=True)
  sched_lr = build_lr_scheduler(
    opt, scheduler=cfg.optim.lr_scheduler, warmup=warmup, total_steps=steps,
    num_restarts=int(cfg.optim.num_restarts), min_lr_ratio=float(cfg.optim.min_lr_ratio),
  )

  # --- fused back pass: per-parameter step in a grad hook, then free the grad ---
  handles = []
  for group in opt.param_groups:
    for i, p in enumerate(group["params"]):
      if not p.requires_grad:
        continue
      def _hook(param, g=group, idx=i):
        if grad_clip:
          torch.nn.utils.clip_grad_norm_(param, grad_clip)  # per-parameter clip
        opt.step_parameter(param, g, idx)
        param.grad = None
      handles.append(p.register_post_accumulate_grad_hook(_hook))
  print(f"[full] fused back pass: {len(handles)} per-parameter grad hooks", flush=True)

  schedule = get_schedule_for_resolution((res, res), known_mean=1.0)

  # --- lazy cache index (filenames only; idx<n_eval held out) ---
  files = sorted(f for f in os.listdir(cache) if f.endswith(".pt"))
  train_pool = files
  if cfg.data.train_list:
    train_pool = json.load(open(cfg.data.train_list))
    print(f"[full] train_list: {len(train_pool)} entries ({len(set(train_pool))} unique)", flush=True)
  from collections import defaultdict
  idx_path = os.path.join(cache, ".bucket_index.json")
  try:
    prev_idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
  except Exception:
    prev_idx = {}
  bucket_files = defaultdict(list)
  eval_files = [f for f in files if int(f[:-3]) < n_eval]
  new_idx = {}
  n_new = 0
  for f in train_pool:
    if int(f[:-3]) < n_eval:
      continue
    try:
      st = os.stat(f"{cache}/{f}")
      size, mtime_ns = int(st.st_size), int(st.st_mtime_ns)
    except OSError:
      continue
    cached = prev_idx.get(f)
    # Cache entry is [gh, gw, size, mtime_ns]; legacy 2-element entries and any
    # (size, mtime_ns) mismatch are treated as a miss -> re-probe the header.
    if (isinstance(cached, (list, tuple)) and len(cached) == 4
        and int(cached[2]) == size and int(cached[3]) == mtime_ns):
      gh_gw = [int(cached[0]), int(cached[1])]
    else:
      try:
        hdr = torch.load(f"{cache}/{f}", map_location="cpu", mmap=True)
        gh_gw = [int(hdr["grid_h"]), int(hdr["grid_w"])]; del hdr
        n_new += 1
      except Exception:
        continue
    new_idx[f] = [gh_gw[0], gh_gw[1], size, mtime_ns]
    bucket_files[tuple(gh_gw)].append(f)
  if n_new or new_idx != prev_idx:
    try:
      json.dump(new_idx, open(idx_path, "w"))
    except Exception:
      pass
  bucket_keys = list(bucket_files.keys())
  bucket_weights = [len(bucket_files[k]) for k in bucket_keys]
  print(f"[full] {sum(bucket_weights)} training + {len(eval_files)} held-out caches", flush=True)

  gen = torch.Generator(device=device).manual_seed(cfg.runtime.seed)
  rng = random.Random(cfg.runtime.seed)
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

  def _h2d(t):
    # Pin a per-batch staging copy on the host, then async copy H2D. The pinned
    # buffer is bounded to one batch (it goes out of scope after the copy issues);
    # default-stream consumers serialize after this copy, so values are identical
    # to a plain blocking .to(device).
    if device.type == "cuda" and t.device.type == "cpu":
      t = t.pin_memory()
    return t.to(device, non_blocking=True)

  def _to_dev(c):
    return {"grid_h": c["grid_h"], "grid_w": c["grid_w"],
            "z_ref": _h2d(c["z_ref"]), "z_tgt": _h2d(c["z_tgt"]),
            "llm_text": _h2d(c["llm_text"])}

  # --- optional RAM preload of the training pool: kills per-step disk torch.load.
  # Lossless: the cache dict is identical whether served from RAM or re-read from
  # disk, and the preload pass touches no RNG -> the rng/gen draw order (and thus
  # every batch, noise, and optimizer update) is bit-for-bit unchanged. On a warm
  # restart this dedicated pass runs again before the loop. ---
  ram = {}
  if bool(cfg.optim.preload_caches):
    t_pre = time.time()
    pool_files = sorted({f for fs in bucket_files.values() for f in fs})
    for f in pool_files:
      try:
        ram[f] = torch.load(f"{cache}/{f}", map_location="cpu")
      except Exception:
        pass
    print(f"[full] preloaded {len(ram)} training caches to RAM in "
          f"{time.time()-t_pre:.1f}s", flush=True)

  def _load_cpu(f):
    c = ram.get(f)
    return c if c is not None else torch.load(f"{cache}/{f}", map_location="cpu")

  def _load_dev(f):
    return _to_dev(_load_cpu(f))

  def sample_batch():
    k = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
    pool = bucket_files[k]
    return [_load_dev(rng.choice(pool)) for _ in range(batch)]

  # Eval/preview loads are best-effort: a corrupt held-out cache should drop+log,
  # not abort a multi-hour run. The training-pool load above stays strict.
  eval_items = []
  for f in eval_files:
    try:
      eval_items.append(_load_dev(f))
    except Exception as e:
      print(f"[full] WARN: dropping unreadable eval cache {f}: {e}", flush=True)
  try:
    os.makedirs(sample_dir, exist_ok=True)
    _pr = {str(int(f[:-3])): str(torch.load(f"{cache}/{f}", map_location="cpu", mmap=True).get("edit", ""))
           for f in eval_files}
    json.dump(_pr, open(os.path.join(sample_dir, "prompts.json"), "w"), ensure_ascii=False)
  except Exception:
    pass

  decoder_state = {"d": None}

  def _label(img, text):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, max(46, 7 * len(text) + 8), 18], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(245, 245, 245))
    return img

  def _sample(step_num):
    from PIL import Image
    transformer.eval()
    try:
      with torch.no_grad():
        if decoder_state["d"] is None:
          decoder_state["d"] = edit_sampler.load_decoder(weights, device, dtype)
        ae, shift, scale, patch = decoder_state["d"]
        for j, it in enumerate(eval_items):
          gh, gw = int(it["grid_h"]), int(it["grid_w"])
          z_out = edit_sampler.sample_edit_cached(
            transformer, it["z_ref"], it["llm_text"], gh, gw, schedule=schedule,
            num_steps=int(cfg.logging.sample_steps), guidance_scale=float(cfg.logging.sample_guidance),
            generator=torch.Generator(device=device).manual_seed(step_num))
          triple = torch.cat([it["z_ref"].unsqueeze(0), z_out, it["z_tgt"].unsqueeze(0)], dim=0)
          imgs = edit_sampler.decode_latents(ae, triple, gh, gw, patch_size=patch,
                                             latent_shift=shift, latent_scale=scale, dtype=dtype)
          for im, lab in zip(imgs, ("source", "full-FT", "target")):
            _label(im, lab)
          w, h = imgs[0].size
          canvas = Image.new("RGB", (w * 3, h))
          for k, im in enumerate(imgs):
            canvas.paste(im, (k * w, 0))
          canvas.save(os.path.join(sample_dir, f"step{step_num:06d}_idx{j}.png"))
      print(f"[full] sampled {len(eval_items)} held-out [src|full-FT|tgt] @ step {step_num}", flush=True)
    finally:
      decoder_state["d"] = None
      transformer.train()
      if device.type == "cuda":
        torch.cuda.empty_cache()

  def _save(tag, step_num, keep_last=2):
    # bf16 transformer (18.6GB) -- the trained precision; final can be re-saved fp32.
    state = {k: v.detach().to("cpu", torch.bfloat16).contiguous()
             for k, v in transformer.state_dict().items()}
    from safetensors.torch import save_file
    path = f"{ckpt}/edit_full_{tag}.safetensors"
    save_file(state, path, metadata={"base_model": weights, "type": "ideogram4-edit-full",
                                     "step": str(step_num)})
    json.dump({"step": step_num, "ckpt": os.path.basename(path)}, open(_resume_marker(ckpt), "w"))
    print(f"[full] saved {path} ({len(state)} tensors) @ step {step_num}", flush=True)
    # Each full-model bf16 ckpt is ~18.6GB: rotate the step-tagged ones (the regex
    # excludes "final"/named tags) so the disk does not fill. sorted() puts the
    # newest last; always keep the newest keep_last so the resume marker -- which
    # points at the most recent step ckpt -- never dangles.
    import re as _re
    stepped = sorted(f for f in os.listdir(ckpt)
                     if _re.fullmatch(r"edit_full_step\d{6}\.safetensors", f))
    for old in stepped[:-keep_last]:
      os.remove(os.path.join(ckpt, old))
      print(f"[full] rotated out {old}", flush=True)

  # --- resume (weights + step only; optimizer moments restart) ---
  start_step = 0
  if cfg.paths.resume_from and os.path.exists(_resume_marker(ckpt)):
    mk = json.load(open(_resume_marker(ckpt)))
    from safetensors.torch import load_file
    sdr = load_file(f"{ckpt}/{mk['ckpt']}")
    transformer.load_state_dict({k: v.to(device, dtype) for k, v in sdr.items()}, strict=True)
    start_step = int(mk["step"])
    for _ in range(start_step):
      sched_lr.step()
    print(f"[full] RESUMED weights from {mk['ckpt']} at step {start_step} (optimizer moments restart)", flush=True)

  n_skipped = 0
  run, t_last = 0.0, time.time()
  for step in range(start_step, steps):
    b = sample_batch()
    loss = train_edit.edit_training_step_cached_batch(
      transformer, b, schedule=schedule, cfg_dropout_prob=cfg_drop, generator=gen,
      timestep_shift=ts_shift, timestep_weighting=ts_weighting, min_snr_gamma=min_snr_gamma,
      masked_loss=masked, mask_quantile=mask_q, mask_bg_weight=mask_bg,
      noise_offset=noise_offset, input_perturbation=input_perturbation,
    )
    if not is_finite_loss(loss):     # skip BEFORE backward -> no NaN update applied
      n_skipped += 1
      sched_lr.step()
      continue
    loss.backward()                  # fused hooks apply per-parameter AdamW here
    sched_lr.step()
    run += loss.item()

    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step + 1, "loss": run / log_every, "lr": sched_lr.get_last_lr()[0],
             "s_per_step": dt, "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
             "skipped": n_skipped}
      print(f"[full] step {step+1}/{steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} | "
            f"{dt:.2f}s/step peak {rec['peak_gb']:.1f}GB skipped={n_skipped}", flush=True)
      with open(metrics_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
      run, t_last = 0.0, time.time()
      if args.smoke and step + 1 >= 30:
        print("[full] SMOKE OK -- dequant verified + steps run + VRAM measured. Exiting.", flush=True)
        return

    if sample_every and (step + 1) % sample_every == 0:
      _sample(step + 1)
    if ckpt_every and (step + 1) % ckpt_every == 0:
      _save(f"step{step+1:06d}", step + 1)

  _save("final", steps)
  print("[full] DONE", flush=True)


def _resume_marker(ckpt):
  return f"{ckpt}/resume_full.json"


@torch.no_grad()
def _verify_dequant_equivalence(transformer, device):
  """Assert dequantized nn.Linear matches the fp8 forward to bf16 tolerance (proof the swap is faithful)."""
  from ideogram4.quantized_loading import Fp8Linear
  import copy
  fp8 = None
  for m in transformer.modules():
    if isinstance(m, Fp8Linear):
      fp8 = m; break
  if fp8 is None:
    print("[full][smoke] no Fp8Linear found (already dequantized?)", flush=True); return
  x = torch.randn(2, fp8.in_features, device=device, dtype=torch.bfloat16)
  y_fp8 = fp8(x)
  import torch.nn as nn
  lin = nn.Linear(fp8.in_features, fp8.out_features, bias=fp8.bias is not None).to(device, torch.bfloat16)
  w = fp8.weight.to(torch.float32) * fp8.weight_scale.to(torch.float32).unsqueeze(1)
  lin.weight.copy_(w.to(torch.bfloat16))
  if fp8.bias is not None:
    lin.bias.copy_(fp8.bias.to(torch.bfloat16))
  y_deq = lin(x)
  rel = (y_fp8 - y_deq).abs().max().item() / (y_fp8.abs().max().item() + 1e-8)
  print(f"[full][smoke] dequant equivalence: max rel err {rel:.2e} "
        f"({'PASS' if rel < 1e-2 else 'FAIL'})", flush=True)


if __name__ == "__main__":
  main()
