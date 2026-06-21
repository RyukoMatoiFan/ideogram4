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
import platform
import random
import signal
import time

import torch
from safetensors.torch import load_file, save_file

from ideogram4.pipeline_ideogram4 import (
  Ideogram4PipelineConfig, _build_transformer, _load_indexed_or_single_state_dict,
)
from ideogram4.modeling_ideogram4 import Ideogram4Config
from ideogram4.scheduler import get_schedule_for_resolution
from ideogram4 import train_edit, train_t2i, edit_sampler
from ideogram4.constants import LLM_TOKEN_INDICATOR
from ideogram4.fused_adamw import build_fused_adamw, build_fused_adafactor, _sr_seed
from ideogram4.training_utils import is_finite_loss, build_lr_scheduler


def _cap(hld, background, desc, art_style="digital illustration, clean linework, soft shading"):
  """Build a preview caption in the mixed tag+NL shape (NL frame + one full-frame element).
  MUST match the training/inference caption shape so previews are in-distribution."""
  return json.dumps({
    "high_level_description": hld,
    "style_description": {"aesthetics": "detailed, vibrant, clean", "lighting": "soft, even lighting",
                          "medium": "illustration", "art_style": art_style},
    "compositional_deconstruction": {"background": background,
      "elements": [{"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": desc}]},
  }, separators=(",", ":"), ensure_ascii=False)


# Capability probes: exercise the three desc modes (tags / NL / both) so the dashboard shows
# directly whether the fine-tune learned to read tag-style prompts. Override with logging.preview_dir
# for a domain-specific set.
DEFAULT_PREVIEWS = [
  ("tags: mountain lake", _cap("A landscape illustration.", "",
     "mountain, lake, sunset, pine trees, reflection, golden hour")),
  ("tags: city street", _cap("A landscape illustration.", "",
     "city street, neon signs, night, rain, puddles, reflections")),
  ("NL: lighthouse", _cap("A lighthouse on a rocky coast at dawn.",
     "A rocky shoreline with crashing waves, pink dawn sky.",
     "A tall white lighthouse with a red top, its beam glowing, gulls overhead.")),
  ("NL: cafe interior", _cap("A cozy cafe interior in the morning.",
     "A warm wooden cafe with hanging plants and soft lamplight.",
     "A wooden counter with an espresso machine and pastries in a glass case, steam rising.")),
  ("both: forest path", _cap("A forest path in autumn.",
     "A path covered in red and orange maple leaves.",
     "A winding dirt path through tall trees. Tags: forest, autumn, maple leaves, sunlight, fog")),
  ("both: still life", _cap("A still life of fruit on a table.",
     "A dark wooden table near a window with soft daylight.",
     "A bowl of apples and grapes beside a glass jug. Tags: still life, fruit, apples, grapes, window light")),
]


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
  sample_every = int(getattr(cfg.logging, "sample_every", 0))
  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)
  offload = bool(cfg.optim.offload_optimizer)
  fp8_qat = bool(getattr(cfg.optim, "fp8_qat", False))
  qat_stochastic = bool(getattr(cfg.optim, "qat_stochastic", False))
  val_every = int(getattr(cfg.logging, "val_every", 0))
  resume_from = cfg.paths.resume_from
  init_dit, init_te = cfg.paths.init_dit, cfg.paths.init_te

  # Refuse to write run output under the home/root volume (it may be small or full on the host).
  if platform.system() == "Linux":
    for _p in (output_dir, ckpt):
      _ap = os.path.abspath(_p)
      if _ap == "/" or _ap.startswith("/home") or _ap.startswith("/root"):
        raise SystemExit(f"[joint] refusing to write under the home/root volume: {_ap} -- point "
                         f"IG4_PATHS__OUTPUT_DIR (+ weights/cache) at a large data volume")

  os.makedirs(ckpt, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
  sample_dir = os.path.join(output_dir, "samples"); os.makedirs(sample_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  marker_path = os.path.join(ckpt, "resume_joint.json")
  resuming = bool(resume_from and os.path.exists(marker_path))
  from ideogram4.trackers import Tracker
  tracker = Tracker(cfg.logging.tracker, project=cfg.logging.wandb_project,
                    run_name=cfg.logging.run_name or None, out_dir=output_dir)
  if not resuming:
    open(metrics_path, "w").close()  # truncate only on a fresh run; a resume appends
  # QAT finishing pass must init from the trained bf16 master, else it would QAT the BASE and
  # discard the run while still stamping a 'qat' tag. Fail closed.
  if fp8_qat and not resuming and (not init_te or (train_dit and not init_dit)):
    raise SystemExit("[joint] fp8_qat finishing pass needs init weights (init_te"
                     + (", init_dit" if train_dit else "") + ") -- set IG4_PATHS__INIT_DIT/INIT_TE")

  # --- DiT: fp8 -> bf16 trainable ---
  t0 = time.time()
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype); del sd
  if train_dit:
    train_edit.dequantize_fp8_transformer(transformer, dtype=dtype, qat=fp8_qat, qat_stochastic=qat_stochastic)
    transformer.to(device); transformer.requires_grad_(True)
    print(f"[joint] DiT dequantized fp8->bf16 (trainable{', QAT' if fp8_qat else ''}) in "
          f"{time.time()-t0:.1f}s", flush=True)
  else:
    transformer.to(device); transformer.requires_grad_(False)
    print(f"[joint] DiT kept fp8 (FROZEN) in {time.time()-t0:.1f}s -- TE-only stage", flush=True)
  transformer.gradient_checkpointing = True  # .train() below; grad still flows through a frozen DiT
  transformer.train()

  # --- text encoder: fp8 -> bf16 trainable (same dequant; TE is also fp8-only) ---
  t1 = time.time()
  pipe = train_edit.load_encoders_pipeline(weights, device, dtype)
  train_edit.dequantize_fp8_transformer(pipe.text_encoder, dtype=dtype, qat=fp8_qat, qat_stochastic=qat_stochastic)
  pipe.text_encoder.to(device); pipe.text_encoder.requires_grad_(True); pipe.text_encoder.train()
  print(f"[joint] text encoder dequantized fp8->bf16 in {time.time()-t1:.1f}s", flush=True)

  # Optional init-from (QAT finishing pass): load the final bf16 weights at step 0.
  if init_dit:
    transformer.load_state_dict({k: v.to(device, dtype) for k, v in load_file(init_dit).items()}, strict=True)
    print(f"[joint] init DiT from {init_dit}", flush=True)
  if init_te:
    r = pipe.text_encoder.load_state_dict({k: v.to(device, dtype) for k, v in load_file(init_te).items()}, strict=False)
    print(f"[joint] init TE from {init_te} ({len(r.missing_keys)} miss, {len(r.unexpected_keys)} unexp)", flush=True)
  if fp8_qat:
    from ideogram4.quantized_loading import QATFp8Linear
    nq = sum(isinstance(m, QATFp8Linear) for m in transformer.modules()) + \
         sum(isinstance(m, QATFp8Linear) for m in pipe.text_encoder.modules())
    print(f"[joint] QAT: {nq} Linears fake-quantized (rounding={'SR' if qat_stochastic else 'RTN'}) "
          f"-- EXPORT with the same rounding", flush=True)

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
          if param.grad is None or not torch.isfinite(param.grad).all():
            param.grad = None; return  # never step inf/nan grads into weights (clip would nan them)
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
        if not torch.isfinite(p.grad).all():
          p.grad = None; continue  # never step inf/nan grads into weights
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

  qat_tag = ("rtn" if not qat_stochastic else "sr") if fp8_qat else "no"  # export must match this

  def _atomic_save(sd, path, meta):
    save_file(sd, path + ".tmp", metadata=meta)  # never leave a truncated ckpt behind an ENOSPC
    os.replace(path + ".tmp", path)

  def _save(tag, step_num, keep_last=2):
    dit_name, te_name = f"joint_dit_{tag}.safetensors", f"joint_te_{tag}.safetensors"
    if train_dit:  # DiT unchanged when frozen -> no point re-saving it
      ds = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in transformer.state_dict().items()}
      _atomic_save(ds, f"{ckpt}/{dit_name}", {"base_model": weights, "type": "ideogram4-t2i-full",
                   "step": str(step_num), "qat": qat_tag})
    ts = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in pipe.text_encoder.state_dict().items()}
    _atomic_save(ts, f"{ckpt}/{te_name}", {"base_model": weights, "type": "ideogram4-text-encoder-full",
                 "step": str(step_num), "qat": qat_tag})
    # resume marker (atomic): which step + which ckpt files to reload
    with open(marker_path + ".tmp", "w") as f:
      json.dump({"step": step_num, "dit": dit_name if train_dit else "", "te": te_name}, f)
    os.replace(marker_path + ".tmp", marker_path)
    print(f"[joint] saved {'DiT+TE' if train_dit else 'TE'} @ {step_num}", flush=True)
    # Each save is ~18.6GB (DiT) + ~16GB (TE); rotate step-tagged pairs to bound disk (keep last
    # keep_last). Named tags ("final"/"interrupt") are never rotated. DiT+TE rotate together by step.
    import re as _re
    for pre in (["joint_dit_", "joint_te_"] if train_dit else ["joint_te_"]):
      stepped = sorted(f for f in os.listdir(ckpt) if _re.fullmatch(pre + r"step\d{6}\.safetensors", f))
      for old in stepped[:-keep_last]:
        os.remove(os.path.join(ckpt, old)); print(f"[joint] rotated out {old}", flush=True)

  # --- preview sampling: re-encode capability-probe captions through the LIVE (fine-tuned)
  # text encoder each call, so previews reflect the joint model (not a stale cached encoding).
  # Saves a 4-col contact sheet stepNNNNNN_dashboard.png + samples/prompts.json; dashboard.py
  # slices it into one hoverable card per prompt.
  patch_px = pipe.config.patch_size * pipe.config.ae_scale_factor
  preview_caps = []
  pdir = getattr(cfg.logging, "preview_dir", "")
  if pdir and os.path.isdir(pdir):
    for pf in sorted(f for f in os.listdir(pdir) if f.endswith(".json")):
      try:
        preview_caps.append((pf[:-5], open(f"{pdir}/{pf}", encoding="utf-8").read().strip()))
      except Exception:
        pass
  if not preview_caps:
    preview_caps = DEFAULT_PREVIEWS
  try:
    json.dump({str(i): lbl for i, (lbl, _) in enumerate(preview_caps)},
              open(os.path.join(sample_dir, "prompts.json"), "w"), ensure_ascii=False)
  except Exception:
    pass
  decoder_state = {"d": None}

  def _label(img, text):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img); d.rectangle([0, 0, max(46, 7 * len(text) + 8), 18], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(245, 245, 245)); return img

  @torch.no_grad()
  def _encode_caption(caption, gh, gw):
    inputs = train_edit.build_edit_inputs(pipe, [caption], gh, gw)
    llm = pipe._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
    return llm[0][inputs["indicator"][0] == LLM_TOKEN_INDICATOR]

  def _sample(step_num):
    from PIL import Image
    transformer.eval(); pipe.text_encoder.eval()
    try:
      with torch.no_grad():
        if decoder_state["d"] is None:
          decoder_state["d"] = edit_sampler.load_decoder(weights, device, dtype)
        ae, shift, scale, patch = decoder_state["d"]
        gh = gw = res // patch_px
        thumbs = []
        for i, (lbl, cap) in enumerate(preview_caps):
          # FIXED per-prompt seed (cfg.seed + index), not the step -> previews are comparable
          # across checkpoints (same noise; the only change you see is the model).
          llm_text = _encode_caption(cap, gh, gw)
          z = edit_sampler.sample_t2i(
            transformer, llm_text, gh, gw, schedule=schedule,
            num_steps=int(cfg.logging.sample_steps), guidance_scale=float(cfg.logging.sample_guidance),
            generator=torch.Generator(device=device).manual_seed(int(cfg.runtime.seed) + i))
          im = edit_sampler.decode_latents(ae, z, gh, gw, patch_size=patch,
                                           latent_shift=shift, latent_scale=scale, dtype=dtype)[0]
          thumbs.append(_label(im, lbl[:34]))
        cols, tw = 4, 360
        scaled = [t.resize((tw, max(1, int(t.height * tw / t.width)))) for t in thumbs]
        ch = max(s.height for s in scaled); rows = (len(scaled) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * ch), (15, 15, 15))
        for k, s in enumerate(scaled):
          sheet.paste(s, ((k % cols) * tw, (k // cols) * ch))
        sheet.save(os.path.join(sample_dir, f"step{step_num:06d}_dashboard.png"))
        print(f"[joint] preview dashboard ({len(thumbs)}) @ {step_num}", flush=True)
    finally:
      decoder_state["d"] = None; transformer.train(); pipe.text_encoder.train()
      if device.type == "cuda":
        torch.cuda.empty_cache()

  # --- resume (crash recovery): reload weights + step from the marker; reseed the data/noise
  # stream by step so it isn't replayed (Adafactor state is on-GPU + unsaved -> a cold-optimizer
  # transient, tolerable with no first moment). ---
  start_step = 0
  if resuming:
    mk = json.load(open(marker_path))
    if train_dit and mk.get("dit"):
      transformer.load_state_dict({k: v.to(device, dtype) for k, v in load_file(f"{ckpt}/{mk['dit']}").items()}, strict=True)
    pipe.text_encoder.load_state_dict({k: v.to(device, dtype) for k, v in load_file(f"{ckpt}/{mk['te']}").items()}, strict=False)
    start_step = int(mk["step"])
    for _ in range(start_step):
      sched_lr.step()
    rng.seed(int(cfg.runtime.seed) + start_step)
    gen.manual_seed(int(cfg.runtime.seed) + start_step)
    print(f"[joint] RESUMED at step {start_step}", flush=True)

  # --- held-out deterministic val loss (fixed generator -> comparable across checkpoints) ---
  eval_items = []
  for f in sorted(ff for ff in os.listdir(cache) if ff.endswith(".pt") and ff[:-3].isdigit()):
    if int(f[:-3]) >= n_eval:
      continue
    try:
      c = torch.load(f"{cache}/{f}", map_location="cpu")
      eval_items.append({"grid_h": int(c["grid_h"]), "grid_w": int(c["grid_w"]),
                         "z_tgt": c["z_tgt"], "caption": str(c.get("caption", ""))})
    except Exception:
      pass
  print(f"[joint] {len(eval_items)} held-out eval caches", flush=True)

  @torch.no_grad()
  def _val_loss():
    was = transformer.training
    transformer.eval(); pipe.text_encoder.eval()
    try:
      vg = torch.Generator(device=device).manual_seed(1234)  # fixed -> same t/noise every eval
      tot, nv = 0.0, 0
      for it in eval_items:
        l = train_t2i.t2i_te_training_step(
          transformer, pipe, it["caption"], it["z_tgt"].to(device), it["grid_h"], it["grid_w"],
          schedule=schedule, cfg_dropout_prob=0.0, timestep_shift=ts_shift, generator=vg)
        tot += float(l); nv += 1
      return tot / max(nv, 1)
    finally:
      if was:
        transformer.train(); pipe.text_encoder.train()

  # --- graceful save on SIGTERM/SIGINT (fail2ban / SSH drop) so a long run loses <= one window ---
  _state = {"step": start_step}
  def _on_signal(signum, frame):
    print(f"[joint] signal {signum} -> interrupt save @ {_state['step']}", flush=True)
    try:
      _save(f"interrupt{_state['step']:06d}", _state["step"])
    finally:
      os._exit(0)
  for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
      signal.signal(_sig, _on_signal)
    except Exception:
      pass

  n_skipped = consec_skip = 0
  run, t_last = 0.0, time.time()
  def _train_step():
    it = sample_item()
    return train_t2i.t2i_te_training_step(
      transformer, pipe, it["caption"], it["z_tgt"], it["grid_h"], it["grid_w"],
      schedule=schedule, cfg_dropout_prob=cfg_drop, timestep_shift=ts_shift, generator=gen)

  for step in range(start_step, steps):
    _state["step"] = step
    try:
      if fused:
        loss = _train_step()
        if not is_finite_loss(loss):
          n_skipped += 1; consec_skip += 1; sched_lr.step()
          if consec_skip > 50:
            raise SystemExit("[joint] aborting: >50 consecutive non-finite/failed steps")
          continue
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
          n_skipped += 1; consec_skip += 1; sched_lr.step()
          if consec_skip > 50:
            raise SystemExit("[joint] aborting: >50 consecutive non-finite/failed steps")
          continue
        _manual_opt_step(); run += acc_loss
    except Exception as e:  # skip a pathological sample (caption > max tokens, transient OOM) not crash
      for grp in opt.param_groups:
        for p in grp["params"]:
          p.grad = None
      if device.type == "cuda":
        torch.cuda.empty_cache()
      n_skipped += 1; consec_skip += 1; sched_lr.step()
      print(f"[joint] WARN step {step} skipped: {str(e)[:160]}", flush=True)
      if consec_skip > 50:
        raise SystemExit("[joint] aborting: >50 consecutive non-finite/failed steps")
      continue
    consec_skip = 0
    sched_lr.step()

    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step + 1, "loss": run / log_every, "lr": sched_lr.get_last_lr()[0],
             "te_lr": sched_lr.get_last_lr()[-1], "s_per_step": dt,
             "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
             "skipped": n_skipped}
      if val_every and eval_items and (step + 1) % val_every == 0:
        rec["val_loss"] = _val_loss()
      vmsg = f" val {rec['val_loss']:.4f}" if "val_loss" in rec else ""
      print(f"[joint] step {step+1}/{steps} loss {rec['loss']:.4f}{vmsg} lr {rec['lr']:.2e} "
            f"te_lr {rec['te_lr']:.2e} | {dt:.2f}s/step peak {rec['peak_gb']:.1f}GB skipped={n_skipped}",
            flush=True)
      with open(metrics_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
      tracker.log(rec, rec["step"])
      run, t_last = 0.0, time.time()
      if args.smoke and step + 1 >= 20:
        _sample(step + 1)  # smoke also exercises the preview encode/sample/decode path
        print("[joint] SMOKE OK", flush=True); return
    if sample_every and (step + 1) % sample_every == 0:
      _sample(step + 1)
    if ckpt_every and (step + 1) % ckpt_every == 0:
      _save(f"step{step+1:06d}", step + 1)

  _save("final", steps); print("[joint] DONE", flush=True)


if __name__ == "__main__":
  main()
