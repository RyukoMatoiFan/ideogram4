"""Image-editing fine-tuning on top of the Ideogram 4 text-to-image backbone.

Ideogram 4 ships as a text-to-image model: the packed sequence is
``[text tokens][target image tokens]`` and only the image tokens are denoised
(``pipeline_ideogram4.Ideogram4Pipeline._build_inputs``). It has no path for a
*source* image to enter the network.

This module adds instruction-based editing with **in-context reference tokens**
and essentially no new parameters. The source image is VAE-encoded into the same
128-d patch tokens as the target and spliced into the sequence as a separate MRoPE
"frame" (t=1), giving the layout::

    [pad][instruction text][reference image (clean)][target image (noised)]

The reference tokens carry clean (un-noised) latents and are excluded from the
flow-matching loss; the model attends to them when predicting the target velocity.
The only backbone change is in ``Ideogram4Transformer.forward``: tokens tagged
``REFERENCE_IMAGE_INDICATOR`` now also flow through ``input_proj`` and receive the
"image" indicator embedding (see modeling_ideogram4.py). At plain text-to-image
inference no reference tokens exist, so the backbone is unchanged.

This is a reference implementation of the training step and sequence packing; it is
verified against the inference code paths (latent normalization, patch layout, flow
convention) but has not been executed end-to-end here. Wrap ``edit_training_step``
in your own optimizer/dataloader/LoRA setup.
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
  REFERENCE_IMAGE_INDICATOR,
  SEQUENCE_PADDING_INDICATOR,
)
from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from ideogram4.scheduler import LogitNormalSchedule, get_schedule_for_resolution

# MRoPE frame index (the t axis of the (t, h, w) position) for reference-image
# tokens. The target image keeps t=0 (as in text-to-image inference); the reference
# sits at a distinct frame so the model can tell source from target even though they
# share the same (h, w) grid.
REFERENCE_POSITION_T = 1


def load_encoders_pipeline(weights_repo: str, device, dtype) -> Ideogram4Pipeline:
  """Load an encoders-ONLY pipeline (text encoder + VAE, transformers=None).

  Encoding images/captions for precache/precompute needs only the VAE + text encoder,
  NOT the two 9.3B transformers -- loading the full pipeline there peaks ~27GB for
  nothing. This keeps the encode phase well under 24GB (~11GB). The returned object
  supports ``encode_image_tokens``, ``build_edit_inputs``, and ``_encode_text``; do not
  call its transformer paths (they are None).
  """
  from ideogram4.pipeline_ideogram4 import (
    Ideogram4PipelineConfig, _load_qwen3_vl, _load_autoencoder,
  )
  from ideogram4 import pipeline_ideogram4 as _pm  # for the runtime-patched hf_hub_download

  pcfg = Ideogram4PipelineConfig(weights_repo=weights_repo)
  tok, text_encoder = _load_qwen3_vl(
    pcfg.weights_repo, device, dtype,
    tokenizer_subfolder=pcfg.tokenizer_subfolder, text_encoder_subfolder=pcfg.text_encoder_subfolder)
  ae = _load_autoencoder(
    _pm.hf_hub_download(repo_id=pcfg.weights_repo, filename=pcfg.autoencoder_filename), device, dtype)
  return Ideogram4Pipeline(
    conditional_transformer=None, unconditional_transformer=None,
    text_encoder=text_encoder, text_tokenizer=tok, autoencoder=ae,
    config=pcfg, device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Latent encoding (inverse of Ideogram4Pipeline._decode)
# --------------------------------------------------------------------------- #
def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
  """(B, C, H_lat, W_lat) VAE latent -> (B, num_tokens, C * patch_size**2) tokens.

  Exact inverse of the unpatchify in ``Ideogram4Pipeline._decode``: the per-token
  channel order is (patch_h, patch_w, ae_channels) with ae_channels fastest.
  """
  batch_size, channels, height, width = latent.shape
  p = patch_size
  if height % p != 0 or width % p != 0:
    raise ValueError(f"latent spatial dims must be divisible by patch_size={p}")
  grid_h, grid_w = height // p, width // p
  z = latent.view(batch_size, channels, grid_h, p, grid_w, p)
  z = z.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, gh, gw, ph, pw, C)
  return z.reshape(batch_size, grid_h * grid_w, p * p * channels)


@torch.no_grad()
def encode_image_tokens(
  pipeline: Ideogram4Pipeline,
  images: torch.Tensor,
  *,
  patch_size: int = 2,
) -> torch.Tensor:
  """Encode RGB images in [-1, 1] to normalized, patchified latent tokens.

  Args:
    images: (B, 3, H, W) float tensor in [-1, 1] on ``pipeline.device``.

  Returns:
    (B, num_tokens, 128) float32 latent tokens in the model's normalized space,
    matching what the transformer denoises for the target image.
  """
  moments = pipeline.autoencoder.encoder(images.to(pipeline.dtype))
  # encoder returns concatenated [mean(32) ; logvar(32)] (autoencoder.py conv_out);
  # the mean is the deterministic latent the decoder consumes.
  mean = moments[:, : moments.shape[1] // 2]
  tokens = patchify_latent(mean, patch_size).to(torch.float32)
  # Normalize into the transformer's latent space (inverse of _decode's
  # `z * latent_scale + latent_shift`).
  return (tokens - pipeline.latent_shift) / pipeline.latent_scale


def images_to_tensor(
  pil_images,
  height: int,
  width: int,
  device: torch.device,
) -> torch.Tensor:
  """List of PIL images -> (B, 3, H, W) float tensor in [-1, 1]."""
  import numpy as np

  out = []
  for img in pil_images:
    img = img.convert("RGB").resize((width, height))
    arr = torch.from_numpy(np.asarray(img)).float() / 127.5 - 1.0  # (H, W, 3)
    out.append(arr.permute(2, 0, 1))
  return torch.stack(out, dim=0).to(device)


# --------------------------------------------------------------------------- #
# Sequence packing: [pad][text][reference image][target image]
# --------------------------------------------------------------------------- #
def build_edit_inputs(
  pipeline: Ideogram4Pipeline,
  instructions: list[str],
  grid_h: int,
  grid_w: int,
) -> dict[str, torch.Tensor]:
  """Build the packed edit sequence for a batch.

  Source and target are assumed to share resolution (the common case for edit
  pairs), so both occupy ``grid_h * grid_w`` tokens. Text is left-padded to the
  batch max, mirroring ``Ideogram4Pipeline._build_inputs``.

  Returns a dict with ``token_ids``, ``text_position_ids``, ``position_ids``,
  ``segment_ids`` and ``indicator`` (all on ``pipeline.device``), where the
  ``indicator`` field marks reference vs target tokens for downstream scatter/gather.
  """
  tokenized = [pipeline._tokenize(p) for p in instructions]
  batch_size = len(instructions)
  num_image_tokens = grid_h * grid_w
  max_text_tokens = max(num_text for _, num_text in tokenized)
  total_seq_len = max_text_tokens + 2 * num_image_tokens

  # Shared (h, w) grid; the reference sits on a different MRoPE frame (t axis).
  h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
  w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
  zeros = torch.zeros_like(h_idx)
  target_pos = torch.stack([zeros, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
  ref_pos = (
    torch.stack([zeros + REFERENCE_POSITION_T, h_idx, w_idx], dim=1)
    + IMAGE_POSITION_OFFSET
  )

  token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
  text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
  position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
  segment_ids = torch.full(
    (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
  )
  indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

  ref_start = max_text_tokens
  ref_end = ref_start + num_image_tokens
  tgt_start = ref_end
  tgt_end = tgt_start + num_image_tokens

  for b, (toks, num_text) in enumerate(tokenized):
    text_start = max_text_tokens - num_text  # left pad
    text_end = max_text_tokens

    token_ids[b, text_start:text_end] = toks
    text_pos = torch.arange(num_text)
    text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
    text_position_ids[b, text_start:text_end] = text_pos_3d
    position_ids[b, text_start:text_end] = text_pos_3d
    position_ids[b, ref_start:ref_end] = ref_pos
    position_ids[b, tgt_start:tgt_end] = target_pos

    indicator[b, text_start:text_end] = LLM_TOKEN_INDICATOR
    indicator[b, ref_start:ref_end] = REFERENCE_IMAGE_INDICATOR
    indicator[b, tgt_start:tgt_end] = OUTPUT_IMAGE_INDICATOR

    # One segment spanning text + reference + target => full bidirectional
    # attention among them; the left-pad region stays at SEQUENCE_PADDING_INDICATOR.
    segment_ids[b, text_start:tgt_end] = 1

  return {
    "token_ids": token_ids.to(pipeline.device),
    "text_position_ids": text_position_ids.to(pipeline.device),
    "position_ids": position_ids.to(pipeline.device),
    "segment_ids": segment_ids.to(pipeline.device),
    "indicator": indicator.to(pipeline.device),
  }


# --------------------------------------------------------------------------- #
# Training step
# --------------------------------------------------------------------------- #
def edit_training_step(
  pipeline: Ideogram4Pipeline,
  instructions: list[str],
  source_images: torch.Tensor,
  target_images: torch.Tensor,
  *,
  schedule: Optional[LogitNormalSchedule] = None,
  cfg_dropout_prob: float = 0.1,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """One flow-matching training step for instruction-based editing.

  Args:
    instructions: B edit instructions (free text; goes through the chat template).
    source_images: (B, 3, H, W) in [-1, 1] on ``pipeline.device`` -- the image to edit.
    target_images: (B, 3, H, W) in [-1, 1] -- the desired edited result.
    cfg_dropout_prob: per-sample probability of zeroing the instruction features so
      the model also learns the reference-only conditional. This is what makes
      classifier-free guidance over the instruction usable at inference
      (``edit_sampler.edit_generate``); set 0 to disable.

  Returns:
    Scalar MSE loss between predicted and ground-truth velocity over target tokens.
    Backprop through this (train the transformer fully, or attach LoRA to its
    qkv/o/MLP linears and train only those).
  """
  device = pipeline.device
  batch_size, _, height, width = target_images.shape
  if source_images.shape != target_images.shape:
    raise ValueError("this reference implementation assumes matching source/target size")

  patch = pipeline.config.patch_size * pipeline.config.ae_scale_factor
  if height % patch != 0 or width % patch != 0:
    raise ValueError(f"height/width must be divisible by {patch}")
  grid_h, grid_w = height // patch, width // patch
  num_image_tokens = grid_h * grid_w
  latent_dim = pipeline.conditional_transformer.config.in_channels

  schedule = schedule or get_schedule_for_resolution((height, width), known_mean=1.0)

  # --- conditioning latents ---
  z_ref = encode_image_tokens(pipeline, source_images, patch_size=pipeline.config.patch_size)
  z_tgt = encode_image_tokens(pipeline, target_images, patch_size=pipeline.config.patch_size)

  inputs = build_edit_inputs(pipeline, instructions, grid_h, grid_w)
  llm_features = pipeline._encode_text(
    inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
  )

  # Classifier-free guidance dropout: zero the instruction for a random subset of
  # samples (the reference tokens are always kept). edit_generate's negative branch
  # mirrors this by zeroing llm_features over the same full sequence.
  if cfg_dropout_prob > 0.0:
    keep_text = (
      torch.rand(batch_size, device=device, generator=generator) >= cfg_dropout_prob
    )
    llm_features = llm_features * keep_text.view(batch_size, 1, 1).to(llm_features.dtype)

  ref_mask = inputs["indicator"] == REFERENCE_IMAGE_INDICATOR  # (B, L)
  tgt_mask = inputs["indicator"] == OUTPUT_IMAGE_INDICATOR

  # --- flow-matching noising of the TARGET only ---
  # Convention (from the sampler): t=0 -> noise, t=1 -> data;
  #   x_t = t * x0 + (1 - t) * noise,   velocity target v = x0 - noise.
  # Passing uniform samples through the logit-normal schedule yields the same
  # timestep density the model is sampled with at inference.
  u = torch.rand(batch_size, device=device, generator=generator)
  t = schedule(u).to(torch.float32)  # (B,) in (0, 1)
  noise = torch.randn(
    batch_size, num_image_tokens, latent_dim,
    device=device, dtype=torch.float32, generator=generator,
  )
  t_b = t.view(batch_size, 1, 1)
  x_t = t_b * z_tgt + (1.0 - t_b) * noise
  v_target = z_tgt - noise

  # --- assemble the full latent-token tensor x: clean reference + noised target ---
  seq_len = inputs["indicator"].shape[1]
  x = torch.zeros(batch_size, seq_len, latent_dim, device=device, dtype=torch.float32)
  x[ref_mask] = z_ref.reshape(-1, latent_dim)
  x[tgt_mask] = x_t.reshape(-1, latent_dim)

  pred = pipeline.conditional_transformer(
    llm_features=llm_features,
    x=x,
    t=t,
    position_ids=inputs["position_ids"],
    segment_ids=inputs["segment_ids"],
    indicator=inputs["indicator"],
  )  # (B, L, 128) float32; only target positions are meaningful

  pred_tgt = pred[tgt_mask].reshape(batch_size, num_image_tokens, latent_dim)
  return F.mse_loss(pred_tgt, v_target)


# --------------------------------------------------------------------------- #
# Cached training (precomputed VAE latents + text-encoder features)
#
# The instruction's text features are image-independent (Qwen3-VL is causal and the
# image tokens are attention-masked, so text positions never attend to them), and
# the VAE latents of source/target are fixed. Precomputing all three lets training
# run on the transformer alone -- no text encoder or VAE in memory, no 326s load.
# --------------------------------------------------------------------------- #
def build_edit_sequence_meta(
  num_text: int, grid_h: int, grid_w: int, device: torch.device
) -> dict[str, torch.Tensor]:
  """Position/segment/indicator tensors for one cached sample (batch 1, no padding).

  Mirrors the single-sample layout of ``build_edit_inputs``
  ([text][reference @ t=1][target @ t=0]) but needs no tokenizer -- only token
  counts -- so it runs in the encoder-free training process.
  """
  n = grid_h * grid_w
  total = num_text + 2 * n
  h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
  w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
  zeros = torch.zeros_like(h_idx)
  target_pos = torch.stack([zeros, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
  ref_pos = (
    torch.stack([zeros + REFERENCE_POSITION_T, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
  )

  position_ids = torch.zeros(1, total, 3, dtype=torch.long)
  tp = torch.arange(num_text)
  position_ids[0, :num_text] = torch.stack([tp, tp, tp], dim=1)
  position_ids[0, num_text:num_text + n] = ref_pos
  position_ids[0, num_text + n:] = target_pos

  indicator = torch.empty(1, total, dtype=torch.long)
  indicator[0, :num_text] = LLM_TOKEN_INDICATOR
  indicator[0, num_text:num_text + n] = REFERENCE_IMAGE_INDICATOR
  indicator[0, num_text + n:] = OUTPUT_IMAGE_INDICATOR

  segment_ids = torch.ones(1, total, dtype=torch.long)
  return {
    "position_ids": position_ids.to(device),
    "segment_ids": segment_ids.to(device),
    "indicator": indicator.to(device),
  }


def _grid_hw_index(grid_h: int, grid_w: int):
  """Row/col index vectors for a (grid_h, grid_w) image-token block."""
  h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
  w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
  return h_idx, w_idx


def build_multiref_sequence_meta(
  num_text: int,
  ref_grids: list[tuple[int, int]],
  target_grid: tuple[int, int],
  device: torch.device,
) -> dict[str, torch.Tensor]:
  """Position/segment/indicator tensors for an N-reference edit sample (batch 1).

  Generalizes ``build_edit_sequence_meta`` from one reference to many: the layout is
  ``[text][ref_1 @ t=1][ref_2 @ t=2]...[ref_N @ t=N][target @ t=0]``. Each reference
  image occupies its own MRoPE frame (the t axis), so the model tells the references
  apart (and from the target) while they share the (h, w) grid. References all carry
  ``REFERENCE_IMAGE_INDICATOR`` and flow through ``input_proj`` exactly like the single
  reference -- no modeling change is needed; only the sequence grows extra frames.

  ``ref_grids`` is one ``(grid_h, grid_w)`` per reference (they may differ in size);
  ``target_grid`` is the target's ``(grid_h, grid_w)``. With a single reference this
  reduces exactly to ``build_edit_sequence_meta``.
  """
  if not ref_grids:
    raise ValueError("build_multiref_sequence_meta needs >= 1 reference grid")
  ref_ns = [gh * gw for (gh, gw) in ref_grids]
  tgt_h, tgt_w = target_grid
  n_tgt = tgt_h * tgt_w
  total = num_text + sum(ref_ns) + n_tgt

  position_ids = torch.zeros(1, total, 3, dtype=torch.long)
  indicator = torch.empty(1, total, dtype=torch.long)

  # Text tokens: 1-D positions replicated across the three MRoPE axes.
  tp = torch.arange(num_text)
  position_ids[0, :num_text] = torch.stack([tp, tp, tp], dim=1)
  indicator[0, :num_text] = LLM_TOKEN_INDICATOR

  cursor = num_text
  for i, (gh, gw) in enumerate(ref_grids):
    n = ref_ns[i]
    h_idx, w_idx = _grid_hw_index(gh, gw)
    frame = torch.full_like(h_idx, REFERENCE_POSITION_T + i)  # t = 1, 2, ..., N
    ref_pos = torch.stack([frame, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    position_ids[0, cursor:cursor + n] = ref_pos
    indicator[0, cursor:cursor + n] = REFERENCE_IMAGE_INDICATOR
    cursor += n

  h_idx, w_idx = _grid_hw_index(tgt_h, tgt_w)
  zeros = torch.zeros_like(h_idx)
  target_pos = torch.stack([zeros, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
  position_ids[0, cursor:cursor + n_tgt] = target_pos
  indicator[0, cursor:cursor + n_tgt] = OUTPUT_IMAGE_INDICATOR

  segment_ids = torch.ones(1, total, dtype=torch.long)
  return {
    "position_ids": position_ids.to(device),
    "segment_ids": segment_ids.to(device),
    "indicator": indicator.to(device),
  }


def edit_training_step_multiref(
  transformer: nn.Module,
  z_refs: list[torch.Tensor],
  ref_grids: list[tuple[int, int]],
  z_tgt: torch.Tensor,
  llm_text: torch.Tensor,
  target_grid: tuple[int, int],
  *,
  schedule: LogitNormalSchedule,
  cfg_dropout_prob: float = 0.1,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """One flow-matching step for N-reference editing from precomputed tensors.

  Generalizes ``edit_training_step_cached`` to multiple references: each ``z_refs[i]``
  is the latent of reference image ``i`` (grid ``ref_grids[i]``) and is placed on its
  own MRoPE frame (t=i+1) via ``build_multiref_sequence_meta``; the target denoises at
  t=0. Loss is the flow-matching MSE on the target tokens only. ``z_refs`` order must
  match ``ref_grids`` order (the sequence packs them in that order).
  """
  if len(z_refs) != len(ref_grids):
    raise ValueError("z_refs and ref_grids must align")
  device = z_tgt.device
  tgt_h, tgt_w = target_grid
  n_tgt = tgt_h * tgt_w
  num_text = llm_text.shape[0]
  latent_dim = z_tgt.shape[-1]

  meta = build_multiref_sequence_meta(num_text, ref_grids, target_grid, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]

  llm = torch.zeros(1, seq_len, llm_text.shape[-1], device=device, dtype=torch.float32)
  drop = (
    cfg_dropout_prob > 0.0
    and torch.rand((), device=device, generator=generator).item() < cfg_dropout_prob
  )
  if not drop:
    llm[0, :num_text] = llm_text.to(torch.float32)

  z_tgt = z_tgt.view(1, n_tgt, latent_dim).to(torch.float32)
  noise = torch.randn(1, n_tgt, latent_dim, device=device, dtype=torch.float32, generator=generator)
  t = schedule(torch.rand((1,), device=device, generator=generator)).to(torch.float32)
  t_b = t.view(1, 1, 1)
  x_t = t_b * z_tgt + (1.0 - t_b) * noise
  v_target = z_tgt - noise

  ref_mask = indicator == REFERENCE_IMAGE_INDICATOR
  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR
  # Reference latents, concatenated in ref_grids order, fill the reference positions.
  ref_cat = torch.cat([zr.reshape(-1, latent_dim) for zr in z_refs], dim=0).to(torch.float32)
  x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
  x[ref_mask] = ref_cat
  x[tgt_mask] = x_t.reshape(n_tgt, latent_dim)

  pred = transformer(
    llm_features=llm, x=x, t=t,
    position_ids=meta["position_ids"], segment_ids=meta["segment_ids"], indicator=indicator,
  )
  return F.mse_loss(pred[tgt_mask].reshape(1, n_tgt, latent_dim), v_target)


def slider_training_step(
  transformer: nn.Module,
  z_ctx: torch.Tensor,
  llm_pos: torch.Tensor,
  llm_neg: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  llm_anchor: Optional[torch.Tensor] = None,
  eta: float = 2.0,
  slider_scale: float = 1.0,
  bidirectional: bool = True,
  context: str = "edit",
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """One disentangled Concept-Slider step in flow-matching *velocity* space.

  ``context`` picks the sequence layout, which MUST match how the slider is used at
  inference: ``"t2i"`` = plain ``[text][target]`` (a slider for text-to-image
  generation -- ``z_ctx`` only seeds the noised target, no reference frame); ``"edit"``
  = ``[text][reference][target]`` (a slider for the in-context editor). Training in the
  wrong context is why a t2i-applied slider trained with a reference frame barely
  transfers.

  Trains a slider LoRA so that scaling its adapter by ``+slider_scale`` shifts the
  model's velocity prediction toward an attribute (``llm_pos``) and ``-slider_scale``
  shifts away from it (``llm_neg``) -- a bidirectional, prompt-defined knob. Unlike
  the edit steps there is **no ground-truth target image**: the regression targets are
  the *frozen base's own* predictions, anchored on a neutral conditioning and nudged
  along the attribute direction. In velocity space the direction is unambiguous
  (velocity points toward data), so no noise-space sign juggling is needed::

      dir      = v_base(c+) - v_base(c-)                 # frozen, no grad
      target_+ = v_base(c_anchor) + eta * dir            # +scale should match this
      target_- = v_base(c_anchor) - eta * dir            # -scale should match this
      loss     = MSE(v_lora(+s), target_+) + MSE(v_lora(-s), target_-)

  ``z_ctx`` (n,128) is the context image the slider acts on (e.g. a source frame to be
  detailed); it fills the reference frame and seeds the noised target. ``llm_pos`` /
  ``llm_neg`` are the c+/c- instruction features and must share ``num_text``;
  ``llm_anchor`` defaults to zeros (the unconditional anchor), matching how the slider
  composes on top of arbitrary conditioning at inference.

  Requires ``transformer`` to carry injected ``LoRALinear`` adapters (the slider); the
  frozen-base passes run under ``lora_disabled`` and the +/- passes under
  ``lora_scaled``.
  """
  from ideogram4.lora import lora_disabled, lora_scaled

  if llm_pos.shape[0] != llm_neg.shape[0]:
    raise ValueError("llm_pos and llm_neg must share num_text (same sequence layout)")
  device = z_ctx.device
  n = grid_h * grid_w
  num_text = llm_pos.shape[0]
  llm_dim = llm_pos.shape[-1]
  latent_dim = z_ctx.shape[-1]

  if context == "t2i":
    from ideogram4.train_t2i import build_t2i_sequence_meta
    meta = build_t2i_sequence_meta(num_text, grid_h, grid_w, device)
  else:
    meta = build_edit_sequence_meta(num_text, grid_h, grid_w, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]
  ref_mask = indicator == REFERENCE_IMAGE_INDICATOR
  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR

  if llm_anchor is None:
    llm_anchor = torch.zeros(num_text, llm_dim, device=device, dtype=torch.float32)

  def _rows(rows: torch.Tensor) -> torch.Tensor:
    llm = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
    llm[0, :num_text] = rows.to(torch.float32)
    return llm

  # Noise the context image to a random flow time; reference frame stays clean.
  z0 = z_ctx.view(1, n, latent_dim).to(torch.float32)
  noise = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
  t = schedule(torch.rand((1,), device=device, generator=generator)).to(torch.float32)
  t_b = t.view(1, 1, 1)
  x_t = t_b * z0 + (1.0 - t_b) * noise
  x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
  if ref_mask.any():  # "edit" context: a clean reference frame; "t2i" has none
    x[ref_mask] = z_ctx.view(n, latent_dim).to(torch.float32)
  x[tgt_mask] = x_t.reshape(n, latent_dim)

  def vel(llm: torch.Tensor) -> torch.Tensor:
    out = transformer(
      llm_features=llm, x=x, t=t,
      position_ids=meta["position_ids"], segment_ids=meta["segment_ids"], indicator=indicator,
    )
    return out[tgt_mask].reshape(1, n, latent_dim)

  llm_p, llm_m, llm_a = _rows(llm_pos), _rows(llm_neg), _rows(llm_anchor)

  # Frozen-base anchor + attribute direction (detached targets).
  with torch.no_grad(), lora_disabled(transformer):
    v_anchor = vel(llm_a)
    direction = vel(llm_p) - vel(llm_m)
  target_plus = v_anchor + eta * direction

  # Slider adapter dialed +: regress its prediction onto the enhanced anchor.
  with lora_scaled(transformer, slider_scale):
    v_plus = vel(llm_a)
  loss = F.mse_loss(v_plus, target_plus)
  if bidirectional:
    # Also train the -scale branch toward the suppressed anchor -> a true +/- knob.
    target_minus = v_anchor - eta * direction
    with lora_scaled(transformer, -slider_scale):
      v_minus = vel(llm_a)
    loss = loss + F.mse_loss(v_minus, target_minus)
  return loss


def edit_training_step_cached_batch(
  transformer: nn.Module,
  batch: list[dict],
  *,
  schedule: LogitNormalSchedule,
  cfg_dropout_prob: float = 0.1,
  generator: Optional[torch.Generator] = None,
  timestep_shift: float = 1.0,
  timestep_weighting: str = "uniform",
  min_snr_gamma: float = 5.0,
  masked_loss: bool = False,
  mask_quantile: float = 0.5,
  mask_bg_weight: float = 0.0,
  noise_offset: float = 0.0,
  input_perturbation: float = 0.0,
  prior_preservation_weight: float = 0.0,
  t_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Batched (B>1) cached flow-matching step.

  Packs B samples into one sequence with per-sample ``segment_ids`` (block-diagonal
  attention, no cross-sample leakage) and left-padded text, mirroring
  ``build_edit_inputs``. All samples must share the image grid (fixed-resolution
  training). Each item is a dict with z_ref/z_tgt (n,128) and llm_text (num_text,dim).

  Flow-matching knobs (all no-ops at their defaults, preserving prior behaviour):
    timestep_shift     -- flux-style reparametrization of sampled t (1.0 = none).
    timestep_weighting -- per-sample loss weighting (uniform | bell | min_snr).
    masked_loss        -- concentrate the loss on edited tokens (|z_tgt-z_ref|).
  """
  from ideogram4.training_utils import (
    apply_flux_shift, timestep_weight, flow_loss, derive_edit_mask, noise_with_offset,
  )
  device = batch[0]["z_tgt"].device
  grid_h, grid_w = batch[0]["grid_h"], batch[0]["grid_w"]
  n = grid_h * grid_w
  latent_dim = batch[0]["z_tgt"].shape[-1]
  llm_dim = batch[0]["llm_text"].shape[-1]
  bsz = len(batch)
  num_texts = [b["llm_text"].shape[0] for b in batch]
  max_text = max(num_texts)
  seq_len = max_text + 2 * n

  h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
  w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
  zeros = torch.zeros_like(h_idx)
  target_pos = torch.stack([zeros, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
  ref_pos = (
    torch.stack([zeros + REFERENCE_POSITION_T, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
  )

  position_ids = torch.zeros(bsz, seq_len, 3, dtype=torch.long)
  indicator = torch.zeros(bsz, seq_len, dtype=torch.long)
  segment_ids = torch.full((bsz, seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
  llm = torch.zeros(bsz, seq_len, llm_dim, dtype=torch.float32)
  x = torch.zeros(bsz, seq_len, latent_dim, dtype=torch.float32)

  drop = torch.rand(bsz, device=device, generator=generator) < cfg_dropout_prob
  for b, item in enumerate(batch):
    nt = num_texts[b]
    ts, te = max_text - nt, max_text
    rs, re = max_text, max_text + n
    tgts, tgte = re, seq_len
    tp = torch.arange(nt)
    position_ids[b, ts:te] = torch.stack([tp, tp, tp], dim=1)
    position_ids[b, rs:re] = ref_pos
    position_ids[b, tgts:tgte] = target_pos
    indicator[b, ts:te] = LLM_TOKEN_INDICATOR
    indicator[b, rs:re] = REFERENCE_IMAGE_INDICATOR
    indicator[b, tgts:tgte] = OUTPUT_IMAGE_INDICATOR
    segment_ids[b, ts:tgte] = b + 1  # unique per sample => no cross-sample attention
    if not bool(drop[b]):
      llm[b, ts:te] = item["llm_text"].to(torch.float32)
    x[b, rs:re] = item["z_ref"].view(n, latent_dim).to(torch.float32)

  position_ids = position_ids.to(device)
  indicator = indicator.to(device)
  segment_ids = segment_ids.to(device)
  llm = llm.to(device)
  x = x.to(device)

  z_tgt = torch.stack([b["z_tgt"].view(n, latent_dim) for b in batch]).to(device, torch.float32)
  noise = noise_with_offset(
    (bsz, n, latent_dim), noise_offset, generator=generator, device=device,
  )
  if t_override is not None:
    # Deterministic evaluation at fixed timesteps (validation): no sampling/shift.
    t = t_override.to(device=device, dtype=torch.float32).expand(bsz).clone()
  else:
    t = schedule(torch.rand(bsz, device=device, generator=generator)).to(torch.float32)
    t = apply_flux_shift(t, timestep_shift)
  # Input perturbation: x_t is built from a slightly noisier eps, but the velocity
  # target stays clean (z_tgt - noise) -> robustness without biasing the target.
  eps_in = noise
  if input_perturbation:
    eps_in = noise + input_perturbation * torch.randn(
      bsz, n, latent_dim, device=device, dtype=torch.float32, generator=generator
    )
  x_t = t.view(bsz, 1, 1) * z_tgt + (1.0 - t.view(bsz, 1, 1)) * eps_in
  v_target = z_tgt - noise

  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR
  x[tgt_mask] = x_t.reshape(-1, latent_dim)

  pred = transformer(
    llm_features=llm, x=x, t=t,
    position_ids=position_ids, segment_ids=segment_ids, indicator=indicator,
  )
  pred_tgt = pred[tgt_mask].reshape(bsz, n, latent_dim)

  z_ref_stack = None
  if masked_loss or prior_preservation_weight:
    z_ref_stack = torch.stack(
      [b["z_ref"].view(n, latent_dim) for b in batch]
    ).to(device, torch.float32)

  loss_mask = None
  if masked_loss:
    loss_mask = derive_edit_mask(z_ref_stack, z_tgt, quantile=mask_quantile)
    if mask_bg_weight > 0.0:
      # Soft mask: edited tokens weigh 1.0, background mask_bg_weight -- the model
      # still learns copy/consistency, but sparse edits stop being gradient-drowned.
      loss_mask = mask_bg_weight + (1.0 - mask_bg_weight) * loss_mask
  weight = None
  if timestep_weighting != "uniform":
    weight = timestep_weight(t, timestep_weighting, gamma=min_snr_gamma)
  loss = flow_loss(pred_tgt, v_target, mask=loss_mask, weight=weight)

  # Prior preservation: on the BACKGROUND (unedited) tokens, keep the LoRA output
  # close to the frozen base (LoRA-off) output, so the adapter only changes the
  # edited region and does not drift the base behaviour. No-op at init (LoRA=0).
  if prior_preservation_weight:
    from ideogram4.lora import lora_disabled
    with torch.no_grad(), lora_disabled(transformer):
      base_tgt = transformer(
        llm_features=llm, x=x, t=t,
        position_ids=position_ids, segment_ids=segment_ids, indicator=indicator,
      )[tgt_mask].reshape(bsz, n, latent_dim)
    bg_mask = 1.0 - derive_edit_mask(z_ref_stack, z_tgt, quantile=mask_quantile)
    loss = loss + prior_preservation_weight * flow_loss(pred_tgt, base_tgt.detach(), mask=bg_mask)

  return loss


def te_edit_training_step(
  transformer: nn.Module,
  pipeline: Ideogram4Pipeline,
  instruction: str,
  z_ref: torch.Tensor,
  z_tgt: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  cfg_dropout_prob: float = 0.1,
  timestep_shift: float = 1.0,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """One edit flow-matching step with a LIVE, trainable text encoder (TE-LoRA).

  Unlike the cached steps, the instruction's features are computed in-loop through
  ``pipeline.text_encoder`` (so gradient reaches a TE adapter), while the image latents
  ``z_ref``/``z_tgt`` come from the VAE cache. The DiT may be frozen or carry its own
  adapter; either way gradient flows back to the TE through ``llm_features``.

  Batch 1 (one instruction); use gradient accumulation for a larger effective batch.
  """
  from ideogram4.training_utils import apply_flux_shift

  device = z_tgt.device
  n = grid_h * grid_w
  latent_dim = z_tgt.shape[-1]

  inputs = build_edit_inputs(pipeline, [instruction], grid_h, grid_w)
  llm_features = pipeline._encode_text(
    inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"],
    requires_grad=True,  # gradient must reach the TE-LoRA adapter
  )  # (1, L, llm_dim), differentiable w.r.t. the TE adapter
  if cfg_dropout_prob > 0.0 and torch.rand((), device=device, generator=generator).item() < cfg_dropout_prob:
    llm_features = llm_features * 0.0  # drop the instruction (keeps the reference-only conditional)

  ref_mask = inputs["indicator"] == REFERENCE_IMAGE_INDICATOR
  tgt_mask = inputs["indicator"] == OUTPUT_IMAGE_INDICATOR

  z_ref = z_ref.view(1, n, latent_dim).to(torch.float32)
  z_tgt = z_tgt.view(1, n, latent_dim).to(torch.float32)
  noise = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
  t = schedule(torch.rand((1,), device=device, generator=generator)).to(torch.float32)
  if timestep_shift != 1.0:
    t = apply_flux_shift(t, timestep_shift)
  t_b = t.view(1, 1, 1)
  x_t = t_b * z_tgt + (1.0 - t_b) * noise
  v_target = z_tgt - noise

  seq_len = inputs["indicator"].shape[1]
  x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
  x[ref_mask] = z_ref.reshape(-1, latent_dim)
  x[tgt_mask] = x_t.reshape(-1, latent_dim)

  pred = transformer(
    llm_features=llm_features, x=x, t=t,
    position_ids=inputs["position_ids"], segment_ids=inputs["segment_ids"],
    indicator=inputs["indicator"],
  )
  return F.mse_loss(pred[tgt_mask].reshape(1, n, latent_dim), v_target)


# --------------------------------------------------------------------------- #
# Full-rank finetuning helpers
# --------------------------------------------------------------------------- #
def expand_reference_embedding(transformer: nn.Module) -> nn.Module:
  """Grow ``embed_image_indicator`` from 2 to 3 rows for full-rank editing.

  Gives reference tokens a dedicated indicator embedding (slot 2) instead of
  sharing the target's "image" slot. The new row is initialized from the existing
  image row so the model starts unchanged and learns to specialize. The backbone
  forward auto-detects the 3-row layout. Pointless for LoRA (the row isn't an
  adapter); do this only when finetuning the full transformer. In-place + returned.
  """
  emb = transformer.embed_image_indicator
  if emb.num_embeddings >= 3:
    return transformer
  new = nn.Embedding(3, emb.embedding_dim)
  with torch.no_grad():
    new.weight[:2].copy_(emb.weight)
    new.weight[2].copy_(emb.weight[1])  # reference starts as a copy of the image slot
  transformer.embed_image_indicator = new.to(emb.weight.device, emb.weight.dtype)
  return transformer


def dequantize_fp8_transformer(
  transformer: nn.Module, *, dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
  """Replace ``Fp8Linear`` layers with trainable ``nn.Linear`` for full-rank FT.

  Ideogram published only quantized checkpoints (fp8 / nf4); fp8 weights are frozen
  buffers with no full-precision master copy, so the transformer cannot be
  full-rank finetuned as loaded. This reconstructs ``w = w_fp8 * scale`` into real
  ``nn.Linear`` weights. The reconstruction is exact for the fp8-representable
  values but **lossy relative to Ideogram's original pre-quantization weights** --
  you are finetuning from a slightly degraded starting point. For most finetuning
  that is acceptable; if you need the full quality ceiling, prefer LoRA/DoRA on the
  frozen fp8 base (no dequantization) or wait for official bf16 weights. In-place.
  """
  from ideogram4.quantized_loading import Fp8Linear

  for name, child in list(transformer.named_children()):
    if isinstance(child, Fp8Linear):
      lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
      with torch.no_grad():
        w = child.weight.to(torch.float32) * child.weight_scale.to(torch.float32).unsqueeze(1)
        lin.weight.copy_(w)
        if child.bias is not None:
          lin.bias.copy_(child.bias.to(torch.float32))
      setattr(transformer, name, lin.to(child.weight.device, dtype))
    else:
      dequantize_fp8_transformer(child, dtype=dtype)
  return transformer


# --------------------------------------------------------------------------- #
# Minimal usage sketch (not executed here)
# --------------------------------------------------------------------------- #
def example_training_loop() -> None:
  """Skeleton wiring; replace the dummy dataset with real edit pairs."""
  pipeline = Ideogram4Pipeline.from_pretrained(device="cuda", dtype=torch.bfloat16)

  # Train only the backbone (freeze text encoder + VAE). For LoRA, wrap the
  # qkv/o/feed_forward linears of conditional_transformer.layers instead and put
  # only the adapter params in the optimizer.
  transformer = pipeline.conditional_transformer
  transformer.train()
  pipeline.text_encoder.eval()
  pipeline.autoencoder.eval()
  optim = torch.optim.AdamW(transformer.parameters(), lr=1e-5)

  # Dataset yields: (instructions: list[str], source PIL list, target PIL list).
  # e.g. ScaleEdit-12M / GPT-Image-Edit-1.5M triplets {instruction, src, edited}.
  dataset = []  # plug in your dataloader
  height = width = 1024

  for instructions, src_pils, tgt_pils in dataset:
    src = images_to_tensor(src_pils, height, width, pipeline.device)
    tgt = images_to_tensor(tgt_pils, height, width, pipeline.device)
    loss = edit_training_step(pipeline, instructions, src, tgt)
    loss.backward()
    optim.step()
    optim.zero_grad()
