"""EMA bootstrap rollout for the ByG objective.

The rollout produces the noisy intermediate ``y_t`` and the EMA clean estimate
``y0`` used as the straight-through base in the ByG cycle loss.

Convention (matches ``train_edit`` and ``edit_sampler``):
  t=0 is PURE NOISE, t=1 is DATA.
  x_t = t * x0 + (1 - t) * eps
  velocity target v = x0 - eps
  Euler update toward data: z <- z + v * (s - t)  for s > t.

The rollout integrates from near-zero (eps noise) upward to ``t_target``, which
is the forward-pass timestep sampled from the training schedule.  Using the EMA
weights (G_EMA) for this rollout prevents gradient leakage from the live LoRA
adapters and stabilizes training (matching the ByG paper's "bootstrap" stage).
"""

from __future__ import annotations

from typing import Optional

import torch

from ideogram4.byg.ema import LoraEMA
from ideogram4.byg.sequences import build_t2i_sequence_meta
from ideogram4.constants import (
  OUTPUT_IMAGE_INDICATOR,
  REFERENCE_IMAGE_INDICATOR,
)
from ideogram4.scheduler import LogitNormalSchedule, make_step_intervals
from ideogram4.train_edit import build_edit_sequence_meta


@torch.no_grad()
def ema_rollout(
  transformer: torch.nn.Module,
  ema: LoraEMA,
  z_src: torch.Tensor,
  llm_c: torch.Tensor,
  t_target: float,
  *,
  schedule: LogitNormalSchedule,
  steps: int,
  grid_h: int,
  grid_w: int,
  generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """EMA bootstrap rollout from noise to ``t_target``.

  Uses the EMA adapter weights (G_EMA) and the edit-sequence packing
  (reference = z_src, instruction = llm_c) to Euler-integrate from pure noise
  upward to the training timestep ``t_target``.

  The rollout mirrors the Euler loop in ``edit_sampler.edit_generate`` but runs
  only the sub-grid of step intervals below ``t_target``.

  Parameters
  ----------
  transformer:
    ``Ideogram4Transformer`` with injected LoRA adapters.
  ema:
    ``LoraEMA`` instance wrapping the same adapters.
  z_src:
    (n, 128) source latent tokens (clean).
  llm_c:
    (num_text_c, llm_dim) forward-instruction text features.
  t_target:
    The training timestep to roll up to (a scalar float in (0, 1)).
  schedule:
    ``LogitNormalSchedule`` used by the training step.
  steps:
    Number of Euler integration steps to use (``cfg.bootstrap_steps``).
  grid_h, grid_w:
    Spatial patch grid.
  generator:
    Optional seeded ``torch.Generator`` for reproducible noise.

  Returns
  -------
  (y_t, y0):
    ``y_t``: (1, n, 128) noisy latent at t_target.
    ``y0``:  (1, n, 128) EMA one-step clean estimate at t_target,
             computed as ``y_t + (1 - t_target) * v_last``.
  """
  device = z_src.device
  n = grid_h * grid_w
  latent_dim = z_src.shape[-1]
  num_text = llm_c.shape[0]

  # Build the full step grid [0, 1] with (steps + 1) points, then select the
  # sub-grid that falls at or below t_target (in the (0,1) logit-normal space).
  # Because the schedule maps uniform -> logit-normal values, we work in the
  # raw linear grid and only pick intervals where the schedule value <= t_target.
  full_intervals = make_step_intervals(steps).to(device)  # (steps+1,) in [0,1]

  # Map the full grid through the schedule to get the actual t values.
  with torch.no_grad():
    t_vals = schedule(full_intervals)  # (steps+1,) float32

  # Build the edit-sequence metadata (shared across all Euler steps).
  meta = build_edit_sequence_meta(num_text, grid_h, grid_w, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]
  ref_mask = indicator == REFERENCE_IMAGE_INDICATOR
  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR

  # Zero-padded LLM features for the full sequence (text positions only).
  llm = torch.zeros(1, seq_len, llm_c.shape[-1], device=device, dtype=torch.float32)
  llm[0, :num_text] = llm_c.to(torch.float32)

  # Start from pure noise at t=0 (noise side of the convention).
  z = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)

  # Euler loop inside the EMA-adapter context -- no grad is guaranteed by the
  # @torch.no_grad() decorator on this function.
  v_last = torch.zeros(1, n, latent_dim, device=device, dtype=torch.float32)

  with ema.swap_in():
    # `schedule` is DECREASING (schedule(0)≈1 data side, schedule(1)≈0 noise side),
    # so we iterate the grid exactly like edit_sampler.edit_generate: i from
    # (steps-1) down to 0, with t_val = schedule(intervals[i+1]) the CURRENT
    # (noise-ward) level and s_val = schedule(intervals[i]) the next level toward
    # data. We integrate noise -> data and STOP once we reach t_target.
    for i in range(steps - 1, -1, -1):
      t_val_i = float(t_vals[i + 1].item())  # current level (starts ≈0 = noise)
      s_val_i = float(t_vals[i].item())       # next level (toward data, larger)

      if t_val_i >= t_target:
        break  # already at/above the training timestep

      s_use = min(s_val_i, t_target)  # clamp so we land exactly on t_target

      t_tensor = torch.full((1,), t_val_i, dtype=torch.float32, device=device)

      # Assemble x: reference (clean z_src) + noised target (current z).
      x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
      x[ref_mask] = z_src.view(n, latent_dim).to(torch.float32)
      x[tgt_mask] = z.reshape(n, latent_dim)

      out = transformer(
        llm_features=llm,
        x=x,
        t=t_tensor,
        position_ids=meta["position_ids"],
        segment_ids=meta["segment_ids"],
        indicator=indicator,
      )  # (1, seq_len, 128)

      v_last = out[tgt_mask].reshape(1, n, latent_dim)  # velocity at target positions

      # Euler update: z <- z + v * (s - t)  [toward data, s > t]
      z = z + v_last * (s_use - t_val_i)

      if s_use >= t_target:
        break  # landed on t_target

  # y_t is the noisy latent at t_target.
  y_t = z

  # EMA one-step clean estimate: y0 = y_t + (1 - t_target) * v_last
  # This is the "bootstrap" clean estimate used for the straight-through hybrid.
  y0 = y_t + (1.0 - t_target) * v_last

  return y_t, y0
