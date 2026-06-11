"""Reusable training utilities: flow-matching timestep shift/weighting, masked loss,
optimizer factory (incl. Prodigy), and a NaN/inf guard.

These are model-agnostic helpers shared by the edit trainer and the ByG trainer.
Every function here is covered by a synthetic test in tests/test_training_utils.py
that demonstrates not just that it runs, but that it has the intended effect.

Conventions (matching this repo): flow-matching with t=0 noise, t=1 data,
x_t = t*x0 + (1-t)*eps, velocity target v = x0 - eps.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Timestep shift (flux/SD3-style resolution shift), applied to sampled t.
# --------------------------------------------------------------------------- #
def apply_flux_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
  """Monotonic reparametrization of t in [0,1] with fixed endpoints.

  t' = shift * t / (1 + (shift - 1) * t).
  shift == 1 -> identity. shift > 1 biases samples toward t=1 (the data side in
  this repo's convention); shift < 1 biases toward t=0 (more noise). Endpoints
  0 and 1 are fixed for any shift > 0.
  """
  if shift == 1.0:
    return t
  if shift <= 0.0:
    raise ValueError(f"shift must be > 0, got {shift}")
  return (shift * t) / (1.0 + (shift - 1.0) * t)


# --------------------------------------------------------------------------- #
# Per-sample timestep loss weighting.
# --------------------------------------------------------------------------- #
def timestep_weight(
  t: torch.Tensor, scheme: str = "uniform", *, gamma: float = 5.0, bell_sigma: float = 0.25
) -> torch.Tensor:
  """Per-sample loss weight as a function of timestep t (shape (B,)).

  schemes:
    "uniform"  -> 1 everywhere (no reweighting).
    "bell"     -> Gaussian centred at t=0.5 (emphasise mid-noise timesteps).
    "min_snr"  -> flow-matching min-SNR-gamma: min(SNR, gamma)/SNR, with
                  SNR(t) = (t / (1 - t))^2. Down-weights easy (high-SNR / low-noise)
                  timesteps so they do not dominate the gradient.
  Returns weights with the same shape as t (non-negative).
  """
  if scheme == "uniform":
    return torch.ones_like(t)
  if scheme == "bell":
    return torch.exp(-((t - 0.5) ** 2) / (2.0 * bell_sigma ** 2))
  if scheme == "min_snr":
    tc = t.clamp(1e-4, 1.0 - 1e-4)
    snr = (tc / (1.0 - tc)) ** 2
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr
  raise ValueError(f"unknown timestep weighting scheme: {scheme!r}")


# --------------------------------------------------------------------------- #
# Masked + weighted flow-matching loss.
# --------------------------------------------------------------------------- #
def flow_loss(
  pred: torch.Tensor,
  target: torch.Tensor,
  *,
  mask: Optional[torch.Tensor] = None,
  weight: Optional[torch.Tensor] = None,
  eps: float = 1e-8,
) -> torch.Tensor:
  """Masked, timestep-weighted MSE over (B, n, D) velocity tensors.

  mask:   per-token weights (B, n) or per-element (B, n, D), or None. Loss is the
          mean squared error over masked elements only (so a region/edit mask
          focuses the gradient there).
  weight: per-sample weights (B,) from ``timestep_weight``, or None. Combined as a
          weighted mean over the batch.
  """
  se = (pred - target) ** 2  # (B, n, D)
  if mask is not None:
    if mask.dim() == se.dim() - 1:
      mask = mask.unsqueeze(-1)  # (B, n, 1)
    mask = mask.to(se.dtype)
    se = se * mask
    norm = mask.expand_as(se).sum(dim=(1, 2)).clamp_min(eps)  # (B,)
    per_sample = se.sum(dim=(1, 2)) / norm
  else:
    per_sample = se.mean(dim=tuple(range(1, se.dim())))  # (B,)

  if weight is not None:
    weight = weight.to(per_sample.dtype)
    return (per_sample * weight).sum() / weight.sum().clamp_min(eps)
  return per_sample.mean()


# --------------------------------------------------------------------------- #
# Aspect-ratio bucketing (train at native-ish AR instead of square-squash).
# --------------------------------------------------------------------------- #
def aspect_buckets(target_pixels: int = 512 * 512, *, divisor: int = 16,
                   min_ar: float = 0.5, max_ar: float = 2.0, num: int = 9) -> list:
  """Generate `num` (H, W) pixel buckets of ~`target_pixels` area spanning the AR
  range [min_ar, max_ar] (AR = W/H), each side a multiple of `divisor` (the VAE
  patch * ae_scale, so the latent grid is integral). Deduplicated + sorted."""
  buckets = set()
  for i in range(num):
    ar = min_ar * (max_ar / min_ar) ** (i / max(1, num - 1))  # log-spaced W/H
    h = max(divisor, round(math.sqrt(target_pixels / ar) / divisor) * divisor)
    w = max(divisor, round(h * ar / divisor) * divisor)
    buckets.add((h, w))
  return sorted(buckets)


def nearest_bucket(w: int, h: int, buckets: list) -> tuple:
  """Pick the bucket whose aspect ratio (W/H) is closest (in log-space) to the image."""
  ar = math.log(max(1, w) / max(1, h))
  return min(buckets, key=lambda b: abs(math.log(b[1] / b[0]) - ar))


def resize_to_bucket(img, bucket_hw):
  """Resize a PIL image to a bucket (H, W). The bucket AR ~matches the image, so
  this is a near-isotropic resize (minimal distortion vs square-squash)."""
  h, w = bucket_hw
  return img.convert("RGB").resize((w, h))


def noise_with_offset(shape, offset: float, *, generator=None, device=None, dtype=torch.float32):
  """Sample (B, n, D) noise with an optional per-channel constant offset.

  Noise offset (Guttenberg) adds a per-channel constant shared across tokens, which
  lets the model learn global tone/brightness shifts. offset=0 -> plain N(0,1).
  """
  noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
  if offset:
    per_channel = torch.randn(
      (shape[0], 1, shape[-1]), generator=generator, device=device, dtype=dtype
    )
    noise = noise + offset * per_channel
  return noise


def derive_edit_mask(
  z_ref: torch.Tensor, z_tgt: torch.Tensor, *, quantile: float = 0.5
) -> torch.Tensor:
  """Per-token edit mask from |z_tgt - z_ref|: tokens that actually change.

  z_ref, z_tgt: (B, n, D) or (n, D). Returns a float mask (B, n) or (n,) with 1.0
  on tokens whose mean abs-difference is at or above the per-sample ``quantile``
  threshold, 0.0 elsewhere. Useful for edit training so the loss concentrates on
  the changed region instead of the (unchanged) background.
  """
  diff = (z_tgt - z_ref).abs().mean(dim=-1)  # (..., n)
  if diff.dim() == 1:
    thr = torch.quantile(diff, quantile)
    return (diff >= thr).to(z_ref.dtype)
  thr = torch.quantile(diff, quantile, dim=-1, keepdim=True)  # (B, 1)
  return (diff >= thr).to(z_ref.dtype)


# --------------------------------------------------------------------------- #
# LR schedulers (warmup + cosine / constant / linear / cosine-restarts).
# --------------------------------------------------------------------------- #
def build_lr_scheduler(
  optimizer,
  *,
  scheduler: str = "cosine",
  warmup: int = 0,
  total_steps: int = 1,
  num_restarts: int = 1,
  min_lr_ratio: float = 0.0,
):
  """LambdaLR with a linear warmup followed by the chosen decay.

  scheduler: cosine | constant | linear | cosine_restarts. `min_lr_ratio` is the
  floor (fraction of base LR) the decays approach.
  """
  def fn(step: int) -> float:
    if warmup and step < warmup:
      return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    if scheduler == "constant":
      return 1.0
    if scheduler == "linear":
      return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
    if scheduler == "cosine":
      return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler == "cosine_restarts":
      cyc = progress * max(1, num_restarts)
      frac = cyc - math.floor(cyc)
      if progress >= 1.0:
        frac = 1.0  # final point lands on the trough, not a fresh peak
      return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"unknown lr_scheduler: {scheduler!r}")

  return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------------------------------------------------------------------------- #
# Held-out validation loss (deterministic, low-variance).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_val_loss(
  transformer,
  items: list,
  schedule,
  *,
  quantiles: tuple = (0.1, 0.3, 0.5, 0.7, 0.9),
  seed: int = 0,
  device=None,
) -> float:
  """Plain flow-matching loss on held-out cache items at FIXED timestep quantiles.

  Evaluating at deterministic timesteps (the schedule's inverse-CDF quantiles) with
  a fixed-seed noise generator makes this a low-variance generalization signal
  (vs the noisy random-t training loss). Encoder-free: uses only the cached
  z_ref/z_tgt/llm_text, so no VAE/text-encoder needed. Returns the mean loss.
  """
  from collections import defaultdict
  from ideogram4.train_edit import edit_training_step_cached_batch

  if not items:
    return float("nan")
  device = device if device is not None else items[0]["z_tgt"].device
  # Group by AR bucket so each batched forward is grid-homogeneous.
  groups = defaultdict(list)
  for it in items:
    groups[(int(it["grid_h"]), int(it["grid_w"]))].append(it)
  was_training = transformer.training
  transformer.eval()
  total, count = 0.0, 0
  for q in quantiles:
    for grp in groups.values():
      t_q = schedule(torch.tensor([q], dtype=torch.float32)).to(device)
      gen = torch.Generator(device=device).manual_seed(seed)
      loss = edit_training_step_cached_batch(
        transformer, grp, schedule=schedule, cfg_dropout_prob=0.0,
        generator=gen, t_override=t_q.expand(len(grp)),  # plain MSE, no mask/weighting
      )
      total += float(loss.item()) * len(grp)
      count += len(grp)
  if was_training:
    transformer.train()
  return total / max(1, count)


# --------------------------------------------------------------------------- #
# Optimizer factory (adds Prodigy = auto-LR).
# --------------------------------------------------------------------------- #
def build_optimizer(name: str, params, lr: float, *, weight_decay: float = 0.0):
  """Build an optimizer by name: adamw | adamw8bit | prodigy | schedule_free | came.

  prodigy / schedule_free adapt their own learning rate (lr ~1.0 / a normal lr).
  schedule_free manages its own schedule and needs opt.train()/opt.eval() around
  the training/eval phases (see is_schedule_free + the trainer wiring). Non-core
  backends are imported lazily.
  """
  name = name.lower()
  if name == "adamw":
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
  if name == "adamw8bit":
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
  if name == "prodigy":
    from prodigyopt import Prodigy
    # Prodigy: lr acts as a scaling on the adapted step; 1.0 is the canonical value.
    return Prodigy(params, lr=lr, weight_decay=weight_decay, safeguard_warmup=True)
  if name == "schedule_free":
    from schedulefree import AdamWScheduleFree
    return AdamWScheduleFree(params, lr=lr, weight_decay=weight_decay)
  if name == "came":
    from pytorch_optimizer import CAME
    return CAME(params, lr=lr, weight_decay=weight_decay)
  raise ValueError(
    f"unknown optimizer: {name!r} (adamw | adamw8bit | prodigy | schedule_free | came)"
  )


def is_schedule_free(name: str) -> bool:
  """schedule_free optimizers need opt.train()/opt.eval() and no LR scheduler."""
  return name.lower() == "schedule_free"


# --------------------------------------------------------------------------- #
# Full training-state checkpoint (resume exactly: weights+opt+sched+RNG+step).
# --------------------------------------------------------------------------- #
def _lora_state(wrapped: dict) -> dict:
  return {sub: (m.lora_A.detach().cpu(), m.lora_B.detach().cpu()) for sub, m in wrapped.items()}


def save_training_state(path, *, step, optimizer, scheduler, wrapped, ema=None, gen=None, extra=None):
  """Atomically save everything needed to resume training byte-for-byte."""
  import random

  state = {
    "step": step,
    "lora": _lora_state(wrapped),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler is not None else None,
    "ema": ema.state_dict() if ema is not None else None,
    "rng_python": random.getstate(),
    "rng_torch": torch.get_rng_state(),
    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    "gen": gen.get_state() if gen is not None else None,
    "extra": extra or {},
  }
  tmp = f"{path}.tmp"
  torch.save(state, tmp)
  os.replace(tmp, path)


def load_training_state(path, *, optimizer, scheduler, wrapped, ema=None, gen=None, map_location="cpu"):
  """Restore a save_training_state checkpoint in place. Returns (step, extra)."""
  import random

  state = torch.load(path, map_location=map_location, weights_only=False)
  with torch.no_grad():
    for sub, m in wrapped.items():
      a, b = state["lora"][sub]
      m.lora_A.copy_(a.to(m.lora_A.device, m.lora_A.dtype))
      m.lora_B.copy_(b.to(m.lora_B.device, m.lora_B.dtype))
  optimizer.load_state_dict(state["optimizer"])
  if scheduler is not None and state.get("scheduler") is not None:
    scheduler.load_state_dict(state["scheduler"])
  if ema is not None and state.get("ema") is not None:
    ema.load_state_dict(state["ema"])
  random.setstate(state["rng_python"])
  torch.set_rng_state(state["rng_torch"])
  if state.get("rng_cuda") is not None and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(state["rng_cuda"])
  if gen is not None and state.get("gen") is not None:
    gen.set_state(state["gen"])
  return state["step"], state.get("extra", {})


# --------------------------------------------------------------------------- #
# NaN / inf guard.
# --------------------------------------------------------------------------- #
def is_finite_loss(loss: torch.Tensor) -> bool:
  """True iff the loss is finite (no NaN/inf) -- skip the optimizer step if False."""
  return bool(torch.isfinite(loss).all())
