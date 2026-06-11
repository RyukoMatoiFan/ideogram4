"""Plain text-to-image (regular subject/style) LoRA training.

The base case the in-context editor (``train_edit``) generalizes: the packed sequence
is just ``[text][target image]`` with no reference frame, and the flow-matching loss
regresses the velocity over the image tokens. Kept in its own module so ``train_edit``
stays about *editing*; the shared latent/text-encoding helpers (``encode_image_tokens``,
``images_to_tensor``, ``build_edit_inputs``) still live in ``train_edit`` and are reused
by the trainer (``train_t2i_lora.py``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ideogram4.constants import (
  IMAGE_POSITION_OFFSET,
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
)
from ideogram4.scheduler import LogitNormalSchedule


def build_t2i_sequence_meta(
  num_text: int, grid_h: int, grid_w: int, device: torch.device
) -> dict[str, torch.Tensor]:
  """Position/segment/indicator tensors for plain text-to-image: ``[text][target]``.

  The base case that ``train_edit.build_edit_sequence_meta`` generalizes -- drop the
  reference frame. Mirrors ``Ideogram4Pipeline._build_inputs`` packing (target image at
  t=0) but is tokenizer-free (counts only), for encoder-free regular-LoRA training.
  """
  n = grid_h * grid_w
  total = num_text + n
  h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
  w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
  zeros = torch.zeros_like(h_idx)
  target_pos = torch.stack([zeros, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

  position_ids = torch.zeros(1, total, 3, dtype=torch.long)
  tp = torch.arange(num_text)
  position_ids[0, :num_text] = torch.stack([tp, tp, tp], dim=1)
  position_ids[0, num_text:] = target_pos

  indicator = torch.empty(1, total, dtype=torch.long)
  indicator[0, :num_text] = LLM_TOKEN_INDICATOR
  indicator[0, num_text:] = OUTPUT_IMAGE_INDICATOR

  segment_ids = torch.ones(1, total, dtype=torch.long)
  return {
    "position_ids": position_ids.to(device),
    "segment_ids": segment_ids.to(device),
    "indicator": indicator.to(device),
  }


def t2i_training_step(
  transformer: nn.Module,
  z_tgt: torch.Tensor,
  llm_text: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  cfg_dropout_prob: float = 0.1,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """One plain text-to-image flow-matching step -- a regular (subject/style) LoRA.

  No reference frame: packs ``[text][target]``, noises the target, and regresses the
  velocity over the image tokens. ``z_tgt`` (n,128) and ``llm_text`` (num_text, llm_dim)
  are precomputed (VAE + text encoder), so training runs encoder-free. Ship the trained
  adapter to BOTH transformers at inference (as with any LoRA here).
  """
  device = z_tgt.device
  n = grid_h * grid_w
  num_text = llm_text.shape[0]
  latent_dim = z_tgt.shape[-1]
  meta = build_t2i_sequence_meta(num_text, grid_h, grid_w, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]

  llm = torch.zeros(1, seq_len, llm_text.shape[-1], device=device, dtype=torch.float32)
  drop = (
    cfg_dropout_prob > 0.0
    and torch.rand((), device=device, generator=generator).item() < cfg_dropout_prob
  )
  if not drop:
    llm[0, :num_text] = llm_text.to(torch.float32)

  z_tgt = z_tgt.view(1, n, latent_dim).to(torch.float32)
  noise = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
  t = schedule(torch.rand((1,), device=device, generator=generator)).to(torch.float32)
  t_b = t.view(1, 1, 1)
  x_t = t_b * z_tgt + (1.0 - t_b) * noise
  v_target = z_tgt - noise

  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR
  x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
  x[tgt_mask] = x_t.reshape(n, latent_dim)

  pred = transformer(
    llm_features=llm, x=x, t=t,
    position_ids=meta["position_ids"], segment_ids=meta["segment_ids"], indicator=indicator,
  )
  return F.mse_loss(pred[tgt_mask].reshape(1, n, latent_dim), v_target)
