"""Train a Concept-Slider LoRA: a bidirectional attribute knob (e.g. a detailer).

Unlike the edit trainers there is no paired dataset. The slider regresses its +/-
adapter onto the FROZEN base's own prediction, nudged along (positive_prompt -
negative_prompt) in velocity space (see train_edit.slider_training_step). We load the
full pipeline once to (a) encode the c+/c-/anchor prompts to LLM feature rows and
(b) VAE-encode a folder of *context* images the slider should act on, then iterate the
slider step on the conditional transformer with an injected LoRA.

At inference, dial strength with lora.lora_scaled(transformer, factor) and sample
normally (factor>0 enhances, <0 inverts), or apply the same adapter to BOTH
transformers as with any edit LoRA.

  CUDA_VISIBLE_DEVICES=0 python train_slider.py --config config/slider.yaml
"""
import argparse
import json
import os
import random
import time

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
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
  open(metrics_path, "w").close()

  t0 = time.time()
  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=cfg.paths.weights),
    device=cfg.runtime.device, dtype=dtype)
  patch = pipe.config.patch_size * pipe.config.ae_scale_factor
  ps = pipe.config.patch_size
  grid = res // patch
  print(f"[slider] pipeline loaded in {time.time()-t0:.1f}s | grid {grid}x{grid}", flush=True)

  # --- one-time conditioning: encode the +/- (and anchor) prompts to padded rows ---
  rows_pos = _encode_prompt_rows(pipe, s.positive_prompt, grid)
  rows_neg = _encode_prompt_rows(pipe, s.negative_prompt, grid)
  rows_anc = _encode_prompt_rows(pipe, s.anchor_prompt, grid) if s.anchor_prompt else None
  llm_dim = rows_pos.shape[-1]
  num_text = max(rows_pos.shape[0], rows_neg.shape[0],
                 rows_anc.shape[0] if rows_anc is not None else 0)
  llm_pos = _pad_rows(rows_pos, num_text, llm_dim, device)
  llm_neg = _pad_rows(rows_neg, num_text, llm_dim, device)
  llm_anchor = _pad_rows(rows_anc, num_text, llm_dim, device) if rows_anc is not None else None
  print(f"[slider] axis: (+)'{s.positive_prompt[:40]}' (-)'{s.negative_prompt[:40]}' "
        f"| num_text={num_text} eta={s.eta} train_scale={s.train_scale}", flush=True)

  # --- one-time: gather context images the slider acts on (rollouts or a folder) ---
  z_ctxs = []
  if args.rollout > 0:
    with torch.no_grad():
      for k in range(args.rollout):
        prompt = _ROLLOUT_PROMPTS[k % len(_ROLLOUT_PROMPTS)]
        img = pipe([prompt], height=res, width=res, num_steps=24, guidance_scale=5.0,
                   seed=1000 + k, raise_on_caption_issues=False)[0]
        ten = train_edit.images_to_tensor([img], res, res, pipe.device)
        z_ctxs.append(train_edit.encode_image_tokens(pipe, ten, patch_size=ps)[0]
                      .to(torch.float32).cpu())
        if (k + 1) % 8 == 0:
          print(f"[slider] rollout {k+1}/{args.rollout}", flush=True)
    print(f"[slider] generated {len(z_ctxs)} rollout context images (zero external data)", flush=True)
  else:
    data_root = cfg.paths.data_root
    paths = []
    for root, _, fnames in os.walk(data_root):
      for fn in fnames:
        if fn.lower().endswith(_IMG_EXTS):
          paths.append(os.path.join(root, fn))
    paths.sort()
    if not paths:
      raise FileNotFoundError(
        f"no context images ({_IMG_EXTS}) under paths.data_root={data_root!r}; "
        "point it at a folder of images, or pass --rollout N to self-generate context")
    with torch.no_grad():
      for p in paths:
        img = Image.open(p).convert("RGB")
        ten = train_edit.images_to_tensor([img], res, res, pipe.device)
        z_ctxs.append(train_edit.encode_image_tokens(pipe, ten, patch_size=ps)[0]
                      .to(torch.float32).cpu())
    print(f"[slider] encoded {len(z_ctxs)} context images from {data_root}", flush=True)

  # --- free everything the slider does NOT train (it uses only the conditional
  # transformer now that prompts + context are encoded). The full pipeline + two
  # training graphs would otherwise OOM at 512px. ---
  import gc
  transformer = pipe.conditional_transformer
  pipe.unconditional_transformer = None
  pipe.text_encoder = None
  pipe.text_tokenizer = None
  pipe.autoencoder = None
  gc.collect()
  if device.type == "cuda":
    torch.cuda.empty_cache()
  print("[slider] freed text encoder / VAE / unconditional transformer", flush=True)

  # --- inject the slider adapter ---
  wrapped = loramod.inject_lora(transformer, rank=rank)
  params = loramod.lora_parameters(wrapped)
  if bool(cfg.optim.grad_checkpointing):
    transformer.gradient_checkpointing = True
  opt = build_optimizer(cfg.optim.optimizer, params, float(cfg.optim.lr))
  sched_lr = build_lr_scheduler(
    opt, scheduler=cfg.optim.lr_scheduler, warmup=int(cfg.optim.warmup), total_steps=steps,
    num_restarts=int(cfg.optim.num_restarts), min_lr_ratio=float(cfg.optim.min_lr_ratio))
  schedule = get_schedule_for_resolution((res, res), known_mean=1.0)
  print(f"[slider] LoRA rank {rank}: {len(wrapped)} modules, "
        f"{sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)

  transformer.train()
  gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
  rng = random.Random(int(cfg.runtime.seed))
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  run, t_last = 0.0, time.time()
  log_every = int(cfg.logging.log_every)
  ckpt_every = int(cfg.logging.ckpt_every)

  def _meta(step_num):
    return {"base_model": cfg.paths.weights, "type": "ideogram4-slider-lora", "rank": rank,
            "step": step_num, "positive": s.positive_prompt, "negative": s.negative_prompt,
            "eta": s.eta, "train_scale": s.train_scale, "infer_scale": s.infer_scale}

  for step in range(steps):
    opt.zero_grad()
    acc = 0.0
    for _ in range(accum):
      z_ctx = z_ctxs[rng.randrange(len(z_ctxs))].to(device)
      loss = train_edit.slider_training_step(
        transformer, z_ctx, llm_pos, llm_neg, grid, grid, schedule=schedule,
        llm_anchor=llm_anchor, eta=float(s.eta), slider_scale=float(s.train_scale),
        bidirectional=bool(s.bidirectional), generator=gen)
      if cfg.optim.nan_guard and not is_finite_loss(loss):
        continue
      (loss / accum).backward()
      acc += loss.item() / accum
    torch.nn.utils.clip_grad_norm_(params, float(cfg.optim.grad_clip))
    opt.step()
    sched_lr.step()
    run += acc
    if (step + 1) % log_every == 0:
      dt = (time.time() - t_last) / log_every
      rec = {"step": step + 1, "loss": run / log_every, "lr": sched_lr.get_last_lr()[0],
             "s_per_step": dt, "peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                                           if device.type == "cuda" else 0.0)}
      print(f"[slider] step {step+1}/{steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
            f"| {dt:.2f}s/step peak {rec['peak_gb']:.1f}GB", flush=True)
      with open(metrics_path, "a") as mf:
        mf.write(json.dumps(rec) + "\n")
      run, t_last = 0.0, time.time()
    if (step + 1) % ckpt_every == 0:
      loramod.save_lora(wrapped, f"{ckpt}/slider_rank{rank}_step{step+1}.safetensors",
                        metadata=_meta(step + 1))
      print(f"[slider] checkpoint @ step {step+1}", flush=True)

  loramod.save_lora(wrapped, f"{ckpt}/slider_rank{rank}_final.safetensors", metadata=_meta(steps))
  print(f"[slider] DONE -> {ckpt}/slider_rank{rank}_final.safetensors", flush=True)


if __name__ == "__main__":
  main()
