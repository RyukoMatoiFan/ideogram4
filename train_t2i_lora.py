"""Train a regular text-to-image LoRA (subject / style) on the Ideogram 4 backbone.

The plain-LoRA path: no editing, no reference frame -- just a folder of images with
matching .txt captions (kohya/ai-toolkit layout: ``name.jpg`` + ``name.txt``). Loads
the full pipeline once to VAE-encode each image and text-encode each caption, frees
the encoders, then trains the LoRA on the conditional transformer with
``train_edit.t2i_training_step``. Ship the adapter to BOTH transformers at inference
(see eval_t2i.py).

  CUDA_VISIBLE_DEVICES=0 python train_t2i_lora.py --config config/t2i_lora.yaml
"""
import argparse
import json
import os
import random
import time

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import (
  Ideogram4PipelineConfig, _build_transformer, _load_indexed_or_single_state_dict,
)
from ideogram4.modeling_ideogram4 import Ideogram4Config
from ideogram4.constants import LLM_TOKEN_INDICATOR
from ideogram4.scheduler import get_schedule_for_resolution
from ideogram4 import train_edit  # shared latent/text-encoding helpers
from ideogram4 import train_t2i   # regular-LoRA training step
from ideogram4 import lora as loramod
from ideogram4.training_utils import build_optimizer, build_lr_scheduler, is_finite_loss

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main():
  ap = argparse.ArgumentParser(description="Train a regular text-to-image (subject/style) LoRA.")
  ap.add_argument("--config", default="config/t2i_lora.yaml")
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
  data_root = cfg.paths.data_root
  os.makedirs(ckpt, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, "metrics.jsonl")
  open(metrics_path, "w").close()

  import gc
  pcfg = Ideogram4PipelineConfig(weights_repo=cfg.paths.weights)
  patch = pcfg.patch_size * pcfg.ae_scale_factor
  ps = pcfg.patch_size
  grid = res // patch

  # --- collect (image, caption) pairs: name.<img> + name.json (preferred) or name.txt ---
  # Ideogram 4 is trained on structured JSON captions; if a .json sibling exists we feed
  # the canonical-serialized JSON string, else fall back to the plain .txt.
  pairs, n_json = [], 0
  for root, _, fnames in os.walk(data_root):
    for fn in fnames:
      stem, ext = os.path.splitext(fn)
      if ext.lower() not in _IMG_EXTS:
        continue
      cap_json = os.path.join(root, stem + ".json")
      cap_txt = os.path.join(root, stem + ".txt")
      if os.path.exists(cap_json):
        caption = json.dumps(json.load(open(cap_json, encoding="utf-8")),
                             separators=(",", ":"), ensure_ascii=False)
        n_json += 1
      elif os.path.exists(cap_txt):
        caption = open(cap_txt, encoding="utf-8").read().strip()
      else:
        caption = ""
      pairs.append((os.path.join(root, fn), caption))
  pairs.sort()
  if not pairs:
    raise FileNotFoundError(f"no image+caption pairs under {data_root}")
  print(f"[t2i] {len(pairs)} pairs ({n_json} JSON captions, {len(pairs)-n_json} plain-text)", flush=True)

  # --- PHASE 1 (<=24GB): encoders ONLY -> precompute z_tgt + llm_text -> free ---
  # Encoders-only pipeline (text encoder + VAE, no transformers); loading the full
  # pipeline here would peak ~27GB for nothing.
  t0 = time.time()
  enc = train_edit.load_encoders_pipeline(cfg.paths.weights, device, dtype)
  print(f"[t2i] phase1: encoders loaded in {time.time()-t0:.1f}s (no transformers)", flush=True)

  cache = []
  with torch.no_grad():
    for img_path, caption in pairs:
      img = Image.open(img_path).convert("RGB")
      ten = train_edit.images_to_tensor([img], res, res, device)
      z_tgt = train_edit.encode_image_tokens(enc, ten, patch_size=ps)[0].to(torch.float32).cpu()
      inputs = train_edit.build_edit_inputs(enc, [caption or " "], grid, grid)
      llm = enc._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
      llm_text = llm[0][inputs["indicator"][0] == LLM_TOKEN_INDICATOR].to(torch.float32).cpu()
      cache.append((z_tgt, llm_text))
  print(f"[t2i] precomputed {len(cache)} image+caption pairs", flush=True)

  del enc
  gc.collect()
  if device.type == "cuda":
    torch.cuda.empty_cache()

  # --- PHASE 2 (<=24GB): conditional transformer ONLY -> LoRA -> train ---
  t0 = time.time()
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  transformer = _build_transformer(Ideogram4Config(), sd, device, dtype)
  del sd
  gc.collect()
  if device.type == "cuda":
    torch.cuda.empty_cache()
  print(f"[t2i] phase2: conditional transformer loaded in {time.time()-t0:.1f}s", flush=True)

  wrapped = loramod.inject_lora(transformer, rank=rank)
  params = loramod.lora_parameters(wrapped)
  if bool(cfg.optim.grad_checkpointing):
    transformer.gradient_checkpointing = True
  opt = build_optimizer(cfg.optim.optimizer, params, float(cfg.optim.lr))
  sched_lr = build_lr_scheduler(
    opt, scheduler=cfg.optim.lr_scheduler, warmup=int(cfg.optim.warmup), total_steps=steps,
    num_restarts=int(cfg.optim.num_restarts), min_lr_ratio=float(cfg.optim.min_lr_ratio))
  schedule = get_schedule_for_resolution((res, res), known_mean=1.0)
  print(f"[t2i] LoRA rank {rank}: {len(wrapped)} modules, "
        f"{sum(p.numel() for p in params)/1e6:.1f}M params | {len(cache)} samples", flush=True)

  def _meta(step_num):
    return {"base_model": cfg.paths.weights, "type": "ideogram4-t2i-lora", "rank": rank,
            "step": step_num, "samples": len(cache)}

  transformer.train()
  gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
  rng = random.Random(int(cfg.runtime.seed))
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  run, t_last = 0.0, time.time()
  t_step_prev = t_last
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)
  sample_every = int(cfg.logging.sample_every)
  sample_items = cache[:int(cfg.logging.sample_count)] if sample_every else []
  sample_dir = os.path.join(output_dir, "samples")
  decoder_state = {"d": None}

  def _sample(step_num):
    """Decode in-training previews ([generated | target] per item) to samples/."""
    from PIL import Image as PILImage
    from ideogram4 import edit_sampler
    if decoder_state["d"] is None:
      decoder_state["d"] = edit_sampler.load_decoder(cfg.paths.weights, device, dtype)
    ae, shift, lscale, dpatch = decoder_state["d"]
    os.makedirs(sample_dir, exist_ok=True)
    transformer.eval()
    rows = []
    for z_tgt, llm_text in sample_items:
      z = edit_sampler.sample_t2i(
        transformer, llm_text.to(device), grid, grid, schedule=schedule,
        num_steps=int(cfg.logging.sample_steps), guidance_scale=float(cfg.logging.sample_guidance),
        generator=torch.Generator(device=device).manual_seed(step_num))
      pair = torch.cat([z, z_tgt.view(1, grid * grid, -1).to(device)], dim=0)
      rows.append(edit_sampler.decode_latents(
        ae, pair, grid, grid, patch_size=dpatch, latent_shift=shift, latent_scale=lscale, dtype=dtype))
    transformer.train()
    w, h = rows[0][0].size
    canvas = PILImage.new("RGB", (w * 2, h * len(rows)), (15, 15, 15))
    for r, imgs in enumerate(rows):
      canvas.paste(imgs[0], (0, r * h)); canvas.paste(imgs[1], (w, r * h))  # generated | target
    canvas.save(os.path.join(sample_dir, f"step{step_num:06d}.png"))
    print(f"[t2i] sampled {len(rows)} (generated|target) @ step {step_num} -> {sample_dir}", flush=True)

  for step in range(steps):
    opt.zero_grad()
    acc = 0.0
    for _ in range(accum):
      z_tgt, llm_text = rng.choice(cache)
      loss = train_t2i.t2i_training_step(
        transformer, z_tgt.to(device), llm_text.to(device), grid, grid,
        schedule=schedule, cfg_dropout_prob=float(cfg.optim.cfg_dropout_prob), generator=gen)
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
    with open(metrics_path, "a") as mf:  # per-step metrics (dense for the dashboard)
      mf.write(json.dumps({"step": step + 1, "loss": acc, "lr": sched_lr.get_last_lr()[0],
                           "s_per_step": step_dt, "peak_gb": pgb, "total": steps}) + "\n")
    if (step + 1) % log_every == 0:  # console print cadence (windowed average)
      print(f"[t2i] step {step+1}/{steps} loss {run/log_every:.4f} lr {sched_lr.get_last_lr()[0]:.2e} "
            f"| {(now-t_last)/log_every:.2f}s/step peak {pgb:.1f}GB", flush=True)
      run, t_last = 0.0, now
    if (step + 1) % ckpt_every == 0:
      loramod.save_lora(wrapped, f"{ckpt}/t2i_rank{rank}_step{step+1}.safetensors",
                        metadata=_meta(step + 1))
      print(f"[t2i] checkpoint @ step {step+1}", flush=True)
    if sample_every and sample_items and (step + 1) % sample_every == 0:
      try:
        _sample(step + 1)
      except Exception as exc:  # previews must never crash training
        print(f"[t2i] sampling skipped @ step {step+1}: {exc}", flush=True)

  loramod.save_lora(wrapped, f"{ckpt}/t2i_rank{rank}_final.safetensors", metadata=_meta(steps))
  print(f"[t2i] DONE -> {ckpt}/t2i_rank{rank}_final.safetensors", flush=True)


if __name__ == "__main__":
  main()
