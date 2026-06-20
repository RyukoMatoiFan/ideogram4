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


def t2i_te_training_step(
  transformer: nn.Module,
  pipeline,
  caption: str,
  z_tgt: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  cfg_dropout_prob: float = 0.1,
  timestep_shift: float = 1.0,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """One plain text-to-image flow step with a LIVE, trainable text encoder.

  The T2I analogue of ``train_edit.te_edit_training_step``: the caption is encoded
  in-loop through ``pipeline.text_encoder`` (gradient reaches a trained TE), while the
  target latent ``z_tgt`` comes from the VAE cache. Packs ``[text][target]`` (no
  reference frame) and regresses the velocity over the image tokens. Batch 1.
  """
  from ideogram4.constants import LLM_TOKEN_INDICATOR
  from ideogram4.train_edit import build_edit_inputs
  from ideogram4.training_utils import apply_flux_shift

  device = z_tgt.device
  n = grid_h * grid_w
  latent_dim = z_tgt.shape[-1]

  # Live text features (gradient flows to the text encoder). build_edit_inputs is reused
  # only to tokenize + pack the text; the image-block layout is irrelevant to the text
  # rows we extract (the TE attends over text positions only).
  inputs = build_edit_inputs(pipeline, [caption], grid_h, grid_w)
  llm_full = pipeline._encode_text(
    inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"], requires_grad=True
  )
  text_mask = inputs["indicator"][0] == LLM_TOKEN_INDICATOR
  llm_text = llm_full[0][text_mask]  # (num_text, llm_dim), differentiable
  num_text = llm_text.shape[0]
  llm_dim = llm_text.shape[-1]

  meta = build_t2i_sequence_meta(num_text, grid_h, grid_w, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]

  llm = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
  drop = (
    cfg_dropout_prob > 0.0
    and torch.rand((), device=device, generator=generator).item() < cfg_dropout_prob
  )
  if not drop:
    llm[0, :num_text] = llm_text.to(torch.float32)  # scatter keeps the grad path to the TE

  z0 = z_tgt.view(1, n, latent_dim).to(torch.float32)
  noise = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
  t = schedule(torch.rand((1,), device=device, generator=generator)).to(torch.float32)
  if timestep_shift != 1.0:
    t = apply_flux_shift(t, timestep_shift)
  t_b = t.view(1, 1, 1)
  x_t = t_b * z0 + (1.0 - t_b) * noise
  v_target = z0 - noise

  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR
  x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
  x[tgt_mask] = x_t.reshape(n, latent_dim)

  pred = transformer(
    llm_features=llm, x=x, t=t,
    position_ids=meta["position_ids"], segment_ids=meta["segment_ids"], indicator=indicator,
  )
  return F.mse_loss(pred[tgt_mask].reshape(1, n, latent_dim), v_target)


def uncond_training_step_cached(
  transformer: nn.Module,
  z_tgts: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  generator: Optional[torch.Generator] = None,
  timestep_shift: float = 1.0,
  t_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """One flow-matching step for the UNCONDITIONAL transformer (negative model).

  The uncond branch of ``Ideogram4Pipeline.__call__`` is image-only: its sequence is
  the conditional one with the text frame stripped (``position_ids[:, max_text:]``,
  zeroed ``llm_features``) -- exactly ``build_t2i_sequence_meta(num_text=0)``. So a
  LoRA trained here on *degraded* targets teaches the uncond to predict toward the
  degradation manifold; at inference CFG combines ``g*v_cond + (1-g)*v_uncond`` and the
  ``(1-g) < 0`` coefficient steers generations AWAY from it, amplified by ``(g-1)``.

  ``z_tgts`` is (B, n, 128) -- plain B-dim batching (each row its own sequence, as in
  the pipeline's neg pass); no text, no cfg-dropout (this branch is unconditional by
  construction). ``t_override`` fixes t for deterministic validation loss.
  """
  device = z_tgts.device
  bsz, n, latent_dim = z_tgts.shape
  if n != grid_h * grid_w:
    raise ValueError(f"z_tgts token count {n} != grid {grid_h}x{grid_w}")
  llm_dim = transformer.config.llm_features_dim

  meta = build_t2i_sequence_meta(0, grid_h, grid_w, device)
  position_ids = meta["position_ids"].expand(bsz, -1, -1)
  segment_ids = meta["segment_ids"].expand(bsz, -1)
  indicator = meta["indicator"].expand(bsz, -1)
  llm = torch.zeros(bsz, n, llm_dim, device=device, dtype=torch.float32)

  z0 = z_tgts.to(torch.float32)
  noise = torch.randn(bsz, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
  if t_override is not None:
    t = t_override.to(device=device, dtype=torch.float32).expand(bsz).clone()
  else:
    t = schedule(torch.rand((bsz,), device=device, generator=generator)).to(torch.float32)
    if timestep_shift != 1.0:
      from ideogram4.training_utils import apply_flux_shift
      t = apply_flux_shift(t, timestep_shift)
  t_b = t.view(bsz, 1, 1)
  x_t = t_b * z0 + (1.0 - t_b) * noise
  v_target = z0 - noise

  pred = transformer(
    llm_features=llm, x=x_t, t=t,
    position_ids=position_ids, segment_ids=segment_ids, indicator=indicator,
  )
  return F.mse_loss(pred, v_target)


def uncond_anchor_step_cached(
  transformer: nn.Module,
  z_tgts: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  generator: Optional[torch.Generator] = None,
  timestep_shift: float = 1.0,
) -> torch.Tensor:
  """Clean-anchor regularizer for the negative-model (uncond) LoRA.

  ``z_tgts`` here are CLEAN latents. Plain flow training on degraded images alone
  teaches the adapter ``(dataset content + degradations) - uncond prior`` -- steering
  away then also steers away from the dataset's *style* (e.g. photographic-ness), not
  just from low quality. This term pins the adapter to a no-op on clean data: on the
  same ``x_t`` it regresses the adapted prediction onto the frozen base's
  (``MSE(v_lora, v_base)``), so the content/style component cancels by construction
  and only the degradation-conditional direction survives the combined loss
  ``flow(degraded) + anchor_weight * anchor(clean)``.
  """
  from ideogram4.lora import lora_disabled

  device = z_tgts.device
  bsz, n, latent_dim = z_tgts.shape
  if n != grid_h * grid_w:
    raise ValueError(f"z_tgts token count {n} != grid {grid_h}x{grid_w}")
  llm_dim = transformer.config.llm_features_dim

  meta = build_t2i_sequence_meta(0, grid_h, grid_w, device)
  position_ids = meta["position_ids"].expand(bsz, -1, -1)
  segment_ids = meta["segment_ids"].expand(bsz, -1)
  indicator = meta["indicator"].expand(bsz, -1)
  llm = torch.zeros(bsz, n, llm_dim, device=device, dtype=torch.float32)

  z0 = z_tgts.to(torch.float32)
  noise = torch.randn(bsz, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
  t = schedule(torch.rand((bsz,), device=device, generator=generator)).to(torch.float32)
  if timestep_shift != 1.0:
    from ideogram4.training_utils import apply_flux_shift
    t = apply_flux_shift(t, timestep_shift)
  x_t = t.view(bsz, 1, 1) * z0 + (1.0 - t.view(bsz, 1, 1)) * noise

  def vel():
    return transformer(
      llm_features=llm, x=x_t, t=t,
      position_ids=position_ids, segment_ids=segment_ids, indicator=indicator,
    )

  with torch.no_grad(), lora_disabled(transformer):
    v_base = vel()
  return F.mse_loss(vel(), v_base)


def t2i_training_step_cached_batch(
  transformer: nn.Module,
  batch: list[dict],
  *,
  schedule: LogitNormalSchedule,
  cfg_dropout_prob: float = 0.1,
  generator: Optional[torch.Generator] = None,
  timestep_shift: float = 1.0,
  timestep_weighting: str = "uniform",
  min_snr_gamma: float = 5.0,
  noise_offset: float = 0.0,
  t_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Batched (B>1) cached plain text-to-image step -- the T2I analogue of
  ``train_edit.edit_training_step_cached_batch`` with NO reference frame.

  ``t_override`` (scalar tensor) fixes the timestep for deterministic validation loss
  (no sampling / cfg-dropout effect on t); leave None for training.

  Packs B samples ``[text][target]`` into one sequence with per-sample ``segment_ids``
  (block-diagonal attention, no cross-sample leakage) and left-padded text. Each item is
  a dict with ``z_tgt`` (n,128) and ``llm_text`` (num_text, llm_dim). All samples share
  the image grid (fixed-resolution training).
  """
  from ideogram4.training_utils import (
    apply_flux_shift, timestep_weight, flow_loss, noise_with_offset,
  )
  device = batch[0]["z_tgt"].device
  grid_h, grid_w = batch[0]["grid_h"], batch[0]["grid_w"]
  n = grid_h * grid_w
  latent_dim = batch[0]["z_tgt"].shape[-1]
  llm_dim = batch[0]["llm_text"].shape[-1]
  bsz = len(batch)
  num_texts = [b["llm_text"].shape[0] for b in batch]
  max_text = max(num_texts)
  seq_len = max_text + n

  h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
  w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
  zeros = torch.zeros_like(h_idx)
  target_pos = torch.stack([zeros, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

  position_ids = torch.zeros(bsz, seq_len, 3, dtype=torch.long)
  indicator = torch.zeros(bsz, seq_len, dtype=torch.long)
  from ideogram4.constants import SEQUENCE_PADDING_INDICATOR
  segment_ids = torch.full((bsz, seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
  llm = torch.zeros(bsz, seq_len, llm_dim, dtype=torch.float32)
  x = torch.zeros(bsz, seq_len, latent_dim, dtype=torch.float32)

  drop = torch.rand(bsz, device=device, generator=generator) < cfg_dropout_prob
  for b, item in enumerate(batch):
    nt = num_texts[b]
    ts, te = max_text - nt, max_text          # left-padded text
    tgts, tgte = max_text, seq_len
    tp = torch.arange(nt)
    position_ids[b, ts:te] = torch.stack([tp, tp, tp], dim=1)
    position_ids[b, tgts:tgte] = target_pos
    indicator[b, ts:te] = LLM_TOKEN_INDICATOR
    indicator[b, tgts:tgte] = OUTPUT_IMAGE_INDICATOR
    segment_ids[b, ts:tgte] = b + 1
    if not bool(drop[b]):
      llm[b, ts:te] = item["llm_text"].to(torch.float32)

  position_ids = position_ids.to(device)
  indicator = indicator.to(device)
  segment_ids = segment_ids.to(device)
  llm = llm.to(device)
  x = x.to(device)

  z_tgt = torch.stack([b["z_tgt"].view(n, latent_dim) for b in batch]).to(device, torch.float32)
  noise = noise_with_offset((bsz, n, latent_dim), noise_offset, generator=generator, device=device)
  if t_override is not None:
    t = t_override.to(device=device, dtype=torch.float32).expand(bsz).clone()
  else:
    t = schedule(torch.rand(bsz, device=device, generator=generator)).to(torch.float32)
    t = apply_flux_shift(t, timestep_shift)
  x_t = t.view(bsz, 1, 1) * z_tgt + (1.0 - t.view(bsz, 1, 1)) * noise
  v_target = z_tgt - noise

  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR
  x[tgt_mask] = x_t.reshape(-1, latent_dim)

  pred = transformer(
    llm_features=llm, x=x, t=t,
    position_ids=position_ids, segment_ids=segment_ids, indicator=indicator,
  )
  pred_tgt = pred[tgt_mask].reshape(bsz, n, latent_dim)

  weight = None
  if timestep_weighting != "uniform":
    weight = timestep_weight(t, timestep_weighting, gamma=min_snr_gamma)
  return flow_loss(pred_tgt, v_target, weight=weight)
