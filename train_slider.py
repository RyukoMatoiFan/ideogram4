"""Train a Concept-Slider LoRA: a bidirectional attribute knob (e.g. a detailer).

Unlike the edit trainers there is no paired dataset. The slider regresses its +/-
adapter onto the FROZEN base's own prediction, nudged along (positive_prompt -
negative_prompt) in velocity space (see train_edit.slider_training_step). We load the
full pipeline once to (a) encode the c+/c-/anchor prompts to LLM feature rows and
(b) VAE-encode a folder of *context* images the slider should act on, then iterate the
slider step on the conditional transformer with an injected LoRA.

At inference, dial strength with lora.lora_scaled(transformer, factor) and sample with
OUR cached samplers (sample_edit_cached / sample_t2i, factor>0 enhances, <0 inverts).
Do NOT judge a slider through the stock dual-transformer pipeline: its negative branch
is a separate unconditional network, so the zero-text anchor trained here never occurs
there and a both-transformers adapter largely cancels under CFG. For a stock-pipeline
quality knob train the negative-model (uncond) LoRA instead (train_uncond_lora.py).

  CUDA_VISIBLE_DEVICES=0 python train_slider.py --config config/slider.yaml
"""
import argparse
import json
import os
import random
import time

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import (
  Ideogram4Pipeline, Ideogram4PipelineConfig,
  _build_transformer, _load_indexed_or_single_state_dict,
)
from ideogram4.modeling_ideogram4 import Ideogram4Config
from ideogram4.constants import LLM_TOKEN_INDICATOR
from ideogram4.scheduler import get_schedule_for_resolution
from ideogram4 import train_edit
from ideogram4 import lora as loramod
from ideogram4.training_utils import build_optimizer, build_lr_scheduler, is_finite_loss

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# Diverse prompts for --rollout context: the slider only needs realistic noised
# latents to probe the velocity field, so the base model's own generations work as
# self-contained, zero-external-data context.
_ROLLOUT_PROMPTS = (
  "a portrait photograph of a person, natural light",
  "a landscape with mountains and a lake at sunrise",
  "a city street at night with neon signs",
  "a still life of fruit on a wooden table",
  "a close-up of a flower with dew drops on the petals",
  "the interior of a cozy living room with warm lighting",
  "a forest path in autumn with fallen leaves",
  "a plate of gourmet food on a restaurant table",
)


def _encode_prompt_rows(pipe, prompt, grid):
  """Encode one prompt to its (num_text, llm_dim) LLM feature rows (float32)."""
  inputs = train_edit.build_edit_inputs(pipe, [prompt], grid, grid)
  llm = pipe._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
  return llm[0][inputs["indicator"][0] == LLM_TOKEN_INDICATOR].to(torch.float32)


def _pad_rows(rows, length, llm_dim, device):
  """Right-pad (or truncate) feature rows to a common text length with zero rows."""
  out = torch.zeros(length, llm_dim, device=device, dtype=torch.float32)
  k = min(rows.shape[0], length)
  out[:k] = rows[:k]
  return out


def main():
  ap = argparse.ArgumentParser(description="Train a Concept-Slider (attribute) LoRA.")
  ap.add_argument("--config", default="config/slider.yaml")
  ap.add_argument("--rollout", type=int, default=0,
                  help="If >0, generate this many context images from the base model "
                       "(zero external data) instead of reading paths.data_root.")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config)
  apply_runtime(cfg)

  device = torch.device(cfg.runtime.device)
  dtype = dtype_of(cfg)
  res = int(cfg.data.resolution)
  rank = int(cfg.lora.rank)
  steps = int(cfg.optim.steps)
  accum = int(cfg.optim.accum)
  output_dir = cfg.paths.output_dir
  ckpt = cfg.paths.ckpt_dir
  s = cfg.slider
  os.makedirs(ckpt, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  from ideogram4.trackers import Tracker
  tracker = Tracker(cfg.logging.tracker, project=cfg.logging.wandb_project,
                    run_name=cfg.logging.run_name or None, out_dir=output_dir)
  open(metrics_path, "w").close()

  import gc
  pcfg = Ideogram4PipelineConfig(weights_repo=cfg.paths.weights)
  patch = pcfg.patch_size * pcfg.ae_scale_factor
  ps = pcfg.patch_size
  grid = res // patch
  needs_gen = args.rollout > 0  # only rollout has to GENERATE

  # --- Phase 1: load the encoder(s) to encode prompts (+ context). Only --rollout needs
  # the full pipeline (it generates); folder mode loads encoders-only and stays <=24GB. ---
  t0 = time.time()
  if needs_gen:
    enc_pipe = Ideogram4Pipeline.from_pretrained(
      config=pcfg, device=cfg.runtime.device, dtype=dtype)
    print(f"[slider] full pipeline loaded in {time.time()-t0:.1f}s | grid {grid}x{grid}", flush=True)
  else:
    enc_pipe = train_edit.load_encoders_pipeline(cfg.paths.weights, device, dtype)
    print(f"[slider] encoders loaded in {time.time()-t0:.1f}s "
          f"(folder mode, <=24GB) | grid {grid}x{grid}", flush=True)

  # --- encode the +/- (and anchor) prompts to padded rows ---
  rows_pos = _encode_prompt_rows(enc_pipe, s.positive_prompt, grid)
  rows_neg = _encode_prompt_rows(enc_pipe, s.negative_prompt, grid)
  rows_anc = _encode_prompt_rows(enc_pipe, s.anchor_prompt, grid) if s.anchor_prompt else None
  llm_dim = rows_pos.shape[-1]
  num_text = max(rows_pos.shape[0], rows_neg.shape[0],
                 rows_anc.shape[0] if rows_anc is not None else 0)
  llm_pos = _pad_rows(rows_pos, num_text, llm_dim, device)
  llm_neg = _pad_rows(rows_neg, num_text, llm_dim, device)
  llm_anchor = _pad_rows(rows_anc, num_text, llm_dim, device) if rows_anc is not None else None
  ctx_name = "rollout" if needs_gen else "folder"
  print(f"[slider] axis: (+)'{s.positive_prompt[:40]}' (-)'{s.negative_prompt[:40]}' "
        f"| num_text={num_text} eta={s.eta} ctx={ctx_name}", flush=True)

  # Pre-encode a few generation prompts for in-training previews (text encoder still loaded).
  sample_every = int(cfg.logging.sample_every)
  sample_llm = []
  if sample_every:
    for k in range(min(int(cfg.logging.sample_count), len(_ROLLOUT_PROMPTS))):
      sample_llm.append(_encode_prompt_rows(enc_pipe, _ROLLOUT_PROMPTS[k], grid).to(torch.float32).cpu())
    print(f"[slider] pre-encoded {len(sample_llm)} preview prompts", flush=True)

  # --- gather context latents: real images only (a slider direction is only observable
  # on structured x_t along real sampling trajectories, never on random latents) ---
  z_ctxs = []
  if args.rollout > 0:
    with torch.no_grad():
      for k in range(args.rollout):
        img = enc_pipe([_ROLLOUT_PROMPTS[k % len(_ROLLOUT_PROMPTS)]], height=res, width=res,
                       num_steps=24, guidance_scale=5.0, seed=1000 + k, raise_on_caption_issues=False)[0]
        ten = train_edit.images_to_tensor([img], res, res, device)
        z_ctxs.append(train_edit.encode_image_tokens(enc_pipe, ten, patch_size=ps)[0].to(torch.float32).cpu())
        if (k + 1) % 8 == 0:
          print(f"[slider] rollout {k+1}/{args.rollout}", flush=True)
    print(f"[slider] generated {len(z_ctxs)} rollout context images", flush=True)
  else:
    paths = []
    for root, _, fnames in os.walk(cfg.paths.data_root):
      for fn in fnames:
        if fn.lower().endswith(_IMG_EXTS):
          paths.append(os.path.join(root, fn))
    paths.sort()
    if not paths:
      raise FileNotFoundError(
        f"no context images under paths.data_root={cfg.paths.data_root!r}; "
        "use --rollout N (self-generate)")
    with torch.no_grad():
      for p in paths:
        ten = train_edit.images_to_tensor([Image.open(p).convert("RGB")], res, res, device)
        z_ctxs.append(train_edit.encode_image_tokens(enc_pipe, ten, patch_size=ps)[0].to(torch.float32).cpu())
    print(f"[slider] encoded {len(z_ctxs)} context images from {cfg.paths.data_root}", flush=True)

  # --- obtain the conditional transformer, free the rest ---
  if needs_gen:
    transformer = enc_pipe.conditional_transformer
    enc_pipe.unconditional_transformer = None
    enc_pipe.text_encoder = None
    enc_pipe.text_tokenizer = None
    enc_pipe.autoencoder = None
  else:
    del enc_pipe
    gc.collect()
    if device.type == "cuda":
      torch.cuda.empty_cache()
    sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
    transformer = _build_transformer(Ideogram4Config(), sd, device, dtype)
    del sd
  gc.collect()
  if device.type == "cuda":
    torch.cuda.empty_cache()
  print("[slider] training the conditional transformer only", flush=True)

  # --- inject the slider adapter ---
  wrapped = loramod.inject_lora(transformer, rank=rank,
                                variant=cfg.lora.variant, target_adaln=cfg.lora.target_adaln)
  params = loramod.lora_parameters(wrapped)
  if bool(cfg.optim.grad_checkpointing):
    transformer.gradient_checkpointing = True
  opt = build_optimizer(cfg.optim.optimizer, params, float(cfg.optim.lr))
  sched_lr = build_lr_scheduler(
    opt, scheduler=cfg.optim.lr_scheduler, warmup=int(cfg.optim.warmup), total_steps=steps,
    num_restarts=int(cfg.optim.num_restarts), min_lr_ratio=float(cfg.optim.min_lr_ratio))
  schedule = get_schedule_for_resolution(
    (res, res), known_mean=cfg.flow.schedule_mean, std=cfg.flow.schedule_std)
  print(f"[slider] LoRA rank {rank}: {len(wrapped)} modules, "
        f"{sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)

  transformer.train()
  gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
  rng = random.Random(int(cfg.runtime.seed))
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  run, t_last = 0.0, time.time()
  t_step_prev = t_last
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)

  def _meta(step_num):
    return {"base_model": cfg.paths.weights, "type": "ideogram4-slider-lora", "rank": rank,
            "step": step_num, "positive": s.positive_prompt, "negative": s.negative_prompt,
            "eta": s.eta, "train_scale": s.train_scale, "infer_scale": s.infer_scale}

  sample_dir = os.path.join(output_dir, "samples")
  decoder_state = {"d": None}

  def _sample(step_num):
    """Decode in-training previews [-s | off | +s] per prompt to samples/ (the slider knob)."""
    from PIL import Image as PILImage
    from ideogram4 import edit_sampler
    from ideogram4.lora import lora_scaled
    if decoder_state["d"] is None:
      decoder_state["d"] = edit_sampler.load_decoder(cfg.paths.weights, device, dtype)
    ae, shift, lscale, dpatch = decoder_state["d"]
    os.makedirs(sample_dir, exist_ok=True)
    sc = float(s.infer_scale)
    transformer.eval()
    is_t2i = str(s.context) == "t2i"
    rows = []
    for llm in sample_llm:
      panels = []
      for factor in (-sc, 0.0, sc):
        # Per-step LOCAL generator (re-seeded each panel) so the training RNG stream is untouched.
        local_gen = torch.Generator(device=device).manual_seed(step_num)
        with lora_scaled(transformer, factor):
          if is_t2i:
            z = edit_sampler.sample_t2i(
              transformer, llm.to(device), grid, grid, schedule=schedule,
              num_steps=int(cfg.logging.sample_steps), guidance_scale=float(cfg.logging.sample_guidance),
              generator=local_gen)
          else:
            z = edit_sampler.sample_edit_cached(
              transformer, z_ctxs[0].to(device), llm.to(device), grid, grid, schedule=schedule,
              num_steps=int(cfg.logging.sample_steps), guidance_scale=float(cfg.logging.sample_guidance),
              generator=local_gen)
        panels.append(edit_sampler.decode_latents(
          ae, z, grid, grid, patch_size=dpatch, latent_shift=shift, latent_scale=lscale, dtype=dtype)[0])
      rows.append(panels)
    transformer.train()
    w, h = rows[0][0].size
    canvas = PILImage.new("RGB", (w * 3, h * len(rows)), (15, 15, 15))
    for r, panels in enumerate(rows):
      for c, im in enumerate(panels):
        canvas.paste(im, (c * w, r * h))  # -s | off | +s
    canvas.save(os.path.join(sample_dir, f"step{step_num:06d}.png"))
    print(f"[slider] sampled {len(rows)} [-{sc}|off|+{sc}] @ step {step_num} -> {sample_dir}", flush=True)

  for step in range(steps):
    opt.zero_grad()
    acc = 0.0
    for _ in range(accum):
      z_ctx = z_ctxs[rng.randrange(len(z_ctxs))].to(device)
      loss = train_edit.slider_training_step(
        transformer, z_ctx, llm_pos, llm_neg, grid, grid, schedule=schedule,
        llm_anchor=llm_anchor, eta=float(s.eta), slider_scale=float(s.train_scale),
        bidirectional=bool(s.bidirectional), context=s.context, generator=gen)
      if cfg.optim.nan_guard and not is_finite_loss(loss):
        continue
      (loss / accum).backward()
      acc += loss.item() / accum
    torch.nn.utils.clip_grad_norm_(params, float(cfg.optim.grad_clip))
    opt.step()
    sched_lr.step()
    run += acc
    now = time.time(); step_dt = now - t_step_prev; t_step_prev = now
    pgb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    rec = {"step": step + 1, "loss": acc, "lr": sched_lr.get_last_lr()[0],
           "s_per_step": step_dt, "peak_gb": pgb, "total": steps}
    with open(metrics_path, "a") as mf:  # per-step metrics (dense for the dashboard)
      mf.write(json.dumps(rec) + "\n")
    tracker.log(rec, rec["step"])
    if (step + 1) % log_every == 0:  # console print cadence (windowed average)
      print(f"[slider] step {step+1}/{steps} loss {run/log_every:.4f} lr {sched_lr.get_last_lr()[0]:.2e} "
            f"| {(now-t_last)/log_every:.2f}s/step peak {pgb:.1f}GB", flush=True)
      run, t_last = 0.0, now
    if (step + 1) % ckpt_every == 0:
      loramod.save_lora(wrapped, f"{ckpt}/slider_rank{rank}_step{step+1}.safetensors",
                        metadata=_meta(step + 1))
      print(f"[slider] checkpoint @ step {step+1}", flush=True)
    if sample_every and sample_llm and (step + 1) % sample_every == 0:
      try:
        _sample(step + 1)
      except Exception as exc:  # previews must never crash training
        print(f"[slider] sampling skipped @ step {step+1}: {exc}", flush=True)

  loramod.save_lora(wrapped, f"{ckpt}/slider_rank{rank}_final.safetensors", metadata=_meta(steps))
  print(f"[slider] DONE -> {ckpt}/slider_rank{rank}_final.safetensors", flush=True)


if __name__ == "__main__":
  main()
