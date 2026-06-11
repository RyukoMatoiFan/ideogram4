"""Inference sampler for the in-context image editor trained by ``train_edit``.

This is the generation counterpart to ``train_edit.edit_training_step``: it packs
the same ``[pad][instruction][reference @ MRoPE frame t=1][target @ t=0]`` sequence,
denoises only the target tokens with a flow-matching Euler loop, and decodes them.

Unlike the stock text-to-image ``Ideogram4Pipeline.__call__`` (which runs asymmetric
CFG through a *separate* ``unconditional_transformer``), the negative branch here is
the **same conditional transformer** with the instruction features zeroed and the
reference kept -- i.e. classifier-free guidance over the instruction only. That is
what ``train_edit``'s ``cfg_dropout_prob`` trains for, and it means a LoRA attached
to ``conditional_transformer`` (e.g. via ``ideogram_region_lora/server/lora_apply``)
is active on both branches.

Verified for shape/finiteness against modeling_ideogram4.py; not run with real
weights here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from PIL import Image

from ideogram4.constants import OUTPUT_IMAGE_INDICATOR, REFERENCE_IMAGE_INDICATOR
from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from ideogram4.scheduler import (
  LogitNormalSchedule,
  get_schedule_for_resolution,
  make_step_intervals,
)
from ideogram4.train_edit import build_edit_inputs, encode_image_tokens, images_to_tensor


@torch.no_grad()
def sample_edit_cached(
  transformer: torch.nn.Module,
  z_ref: torch.Tensor,
  llm_text: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  num_steps: int = 20,
  guidance_scale: float = 1.0,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """Encoder-free edit generation: produce the target latent from CACHED features.

  Mirrors ``edit_generate`` but takes the already-encoded reference latent
  ``z_ref`` (n,128) and instruction features ``llm_text`` (num_text, llm_dim)
  directly -- no text encoder / VAE-encode needed. Returns the target latent
  ``(1, n, 128)`` (decode with ``decode_latents``). Used for in-training sampling
  in the cached trainer. Instruction-CFG: negative branch zeroes the text, keeps
  the reference (same as ``edit_generate``).
  """
  from ideogram4.train_edit import build_edit_sequence_meta

  device = z_ref.device
  n = grid_h * grid_w
  latent_dim = z_ref.shape[-1]
  num_text = llm_text.shape[0]
  llm_dim = llm_text.shape[-1]

  meta = build_edit_sequence_meta(num_text, grid_h, grid_w, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]
  ref_mask = indicator == REFERENCE_IMAGE_INDICATOR
  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR

  llm_pos = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
  llm_pos[0, :num_text] = llm_text.to(torch.float32)
  llm_neg = torch.zeros_like(llm_pos)
  do_cfg = guidance_scale != 1.0

  step_intervals = make_step_intervals(num_steps).to(device)
  z = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)

  for i in range(num_steps - 1, -1, -1):
    t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
    s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
    t = torch.full((1,), t_val, dtype=torch.float32, device=device)
    x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
    x[ref_mask] = z_ref.view(n, latent_dim).to(torch.float32)
    x[tgt_mask] = z.reshape(n, latent_dim)

    def vel(llm):
      out = transformer(llm_features=llm, x=x, t=t, position_ids=meta["position_ids"],
                        segment_ids=meta["segment_ids"], indicator=indicator)
      return out[tgt_mask].reshape(1, n, latent_dim)

    v = vel(llm_pos)
    if do_cfg:
      vn = vel(llm_neg)
      v = vn + guidance_scale * (v - vn)
    z = z + v * (s_val - t_val)

  return z


@torch.no_grad()
def sample_multiref(
  transformer: torch.nn.Module,
  z_refs: list[torch.Tensor],
  ref_grids: list[tuple[int, int]],
  llm_text: torch.Tensor,
  target_grid: tuple[int, int],
  *,
  schedule: LogitNormalSchedule,
  num_steps: int = 24,
  guidance_scale: float = 2.0,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """Encoder-free N-reference generation: compose ``z_refs`` into the target latent.

  Inference counterpart to ``edit_training_step_multiref``: packs
  ``[text][ref_1 @ t=1]...[ref_N @ t=N][target @ t=0]`` and denoises the target with
  instruction-CFG (negative branch zeroes the text, keeps the references). Returns the
  target latent ``(1, n_tgt, latent_dim)`` -- decode with ``decode_latents``.
  """
  from ideogram4.train_edit import build_multiref_sequence_meta

  device = z_refs[0].device
  tgt_h, tgt_w = target_grid
  n_tgt = tgt_h * tgt_w
  latent_dim = z_refs[0].shape[-1]
  num_text = llm_text.shape[0]
  llm_dim = llm_text.shape[-1]

  meta = build_multiref_sequence_meta(num_text, ref_grids, target_grid, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]
  ref_mask = indicator == REFERENCE_IMAGE_INDICATOR
  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR

  llm_pos = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
  llm_pos[0, :num_text] = llm_text.to(torch.float32)
  llm_neg = torch.zeros_like(llm_pos)
  do_cfg = guidance_scale != 1.0

  ref_cat = torch.cat([zr.reshape(-1, latent_dim) for zr in z_refs], dim=0).to(torch.float32)
  step_intervals = make_step_intervals(num_steps).to(device)
  z = torch.randn(1, n_tgt, latent_dim, device=device, dtype=torch.float32, generator=generator)

  for i in range(num_steps - 1, -1, -1):
    t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
    s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
    t = torch.full((1,), t_val, dtype=torch.float32, device=device)
    x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
    x[ref_mask] = ref_cat
    x[tgt_mask] = z.reshape(n_tgt, latent_dim)

    def vel(llm):
      out = transformer(llm_features=llm, x=x, t=t, position_ids=meta["position_ids"],
                        segment_ids=meta["segment_ids"], indicator=indicator)
      return out[tgt_mask].reshape(1, n_tgt, latent_dim)

    v = vel(llm_pos)
    if do_cfg:
      vn = vel(llm_neg)
      v = vn + guidance_scale * (v - vn)
    z = z + v * (s_val - t_val)

  return z


@torch.no_grad()
def sample_edit_sliders(
  transformer: torch.nn.Module,
  z_ref: torch.Tensor,
  llm_text: torch.Tensor,
  grid_h: int,
  grid_w: int,
  *,
  schedule: LogitNormalSchedule,
  num_steps: int = 24,
  guidance_scale: float = 2.0,
  slider_branches: Optional[Sequence[tuple[torch.Tensor, float]]] = None,
  late_step_frac: float = 0.0,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """Encoder-free edit generation with extra *slider* guidance branches.

  Generalizes :func:`sample_edit_cached`. On top of the usual instruction CFG it adds,
  per step, one steering term for each ``(llm_branch, weight)`` in ``slider_branches``::

      v = v_neg + guidance * (v_pos - v_neg)            # standard instruction CFG
          + sum_k weight_k * (v_pos - v_branch_k)       # push away from each branch concept

  Each branch is a velocity computed under a *different* instruction (e.g. a "low
  detail" or "low quality" prompt's features); subtracting it pushes the sample away
  from that concept -> toward its opposite. Positive ``weight`` enhances, negative
  suppresses. This is the prompt-defined / negative-model slider path; a *trained*
  slider adapter instead just needs its ``LoRALinear`` scale set (``lora.lora_scaled``)
  before calling :func:`sample_edit_cached`.

  ``late_step_frac`` in (0,1] restricts every slider branch to the final fraction of
  steps (detail/quality live in the low-noise tail); ``0`` applies them throughout.
  """
  from ideogram4.train_edit import build_edit_sequence_meta

  device = z_ref.device
  n = grid_h * grid_w
  latent_dim = z_ref.shape[-1]
  num_text = llm_text.shape[0]
  llm_dim = llm_text.shape[-1]
  branches = list(slider_branches or [])

  meta = build_edit_sequence_meta(num_text, grid_h, grid_w, device)
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]
  ref_mask = indicator == REFERENCE_IMAGE_INDICATOR
  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR

  def _rows(rows: torch.Tensor) -> torch.Tensor:
    llm = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
    llm[0, :num_text] = rows.to(torch.float32)
    return llm

  llm_pos = _rows(llm_text)
  llm_neg = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
  branch_llm = [(_rows(b), float(w)) for b, w in branches]
  do_cfg = guidance_scale != 1.0

  # Steps run high-noise -> low-noise as i decreases; "late" = small i.
  late_cut = int(round(late_step_frac * num_steps)) if late_step_frac > 0 else num_steps

  step_intervals = make_step_intervals(num_steps).to(device)
  z = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)

  for i in range(num_steps - 1, -1, -1):
    t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
    s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
    t = torch.full((1,), t_val, dtype=torch.float32, device=device)
    x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
    x[ref_mask] = z_ref.view(n, latent_dim).to(torch.float32)
    x[tgt_mask] = z.reshape(n, latent_dim)

    def vel(llm):
      out = transformer(llm_features=llm, x=x, t=t, position_ids=meta["position_ids"],
                        segment_ids=meta["segment_ids"], indicator=indicator)
      return out[tgt_mask].reshape(1, n, latent_dim)

    v_pos = vel(llm_pos)
    v = v_pos
    if do_cfg:
      v_neg = vel(llm_neg)
      v = v_neg + guidance_scale * (v_pos - v_neg)
    if branch_llm and i < late_cut:
      for llm_b, w in branch_llm:
        v = v + w * (v_pos - vel(llm_b))
    z = z + v * (s_val - t_val)

  return z


def decode_latents(autoencoder, z, grid_h, grid_w, *, patch_size, latent_shift, latent_scale, dtype):
  """Un-normalize + unpatch + VAE-decode latent tokens to PIL images (mirrors _decode)."""
  batch_size = z.shape[0]
  z = z * latent_scale + latent_shift
  ae_channels = z.shape[-1] // (patch_size * patch_size)
  z = z.view(batch_size, grid_h, grid_w, patch_size, patch_size, ae_channels)
  z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
  z = z.view(batch_size, ae_channels, grid_h * patch_size, grid_w * patch_size)
  decoded = autoencoder.decoder(z.to(dtype)).float().clamp(-1.0, 1.0)
  decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
  return [Image.fromarray(arr) for arr in decoded]


def load_decoder(weights: str, device, dtype):
  """Lazily load just the VAE decoder + latent-norm constants for in-training sampling."""
  from ideogram4.pipeline_ideogram4 import (
    _load_autoencoder, Ideogram4PipelineConfig, hf_hub_download,
  )
  from ideogram4.latent_norm import get_latent_norm

  cfg = Ideogram4PipelineConfig(weights_repo=weights)
  ae_path = hf_hub_download(repo_id=cfg.weights_repo, filename=cfg.autoencoder_filename)
  ae = _load_autoencoder(ae_path, device, dtype)
  shift, scale = get_latent_norm()
  return ae, shift.to(device), scale.to(device), cfg.patch_size


@torch.no_grad()
def edit_generate(
  pipeline: Ideogram4Pipeline,
  instruction: str,
  source_image: Image.Image,
  *,
  height: int = 1024,
  width: int = 1024,
  num_steps: int = 28,
  guidance_scale: float = 3.0,
  guidance_schedule: Optional[Sequence[float] | torch.Tensor] = None,
  mu: float = 1.0,
  std: float = 1.0,
  seed: Optional[int] = None,
  schedule: Optional[LogitNormalSchedule] = None,
) -> Image.Image:
  """Apply ``instruction`` to ``source_image`` with a trained in-context editor.

  Args:
    instruction: free-text edit instruction (e.g. "make the hair blonde").
    source_image: the PIL image to edit; resized to (width, height).
    guidance_scale: instruction CFG weight. 1.0 disables CFG (single forward/step).
    guidance_schedule: optional per-step weights (loop-index order, len == num_steps),
      matching the stock pipeline; overrides ``guidance_scale`` when given.

  Returns:
    The edited PIL image.
  """
  device = pipeline.device
  patch = pipeline.config.patch_size * pipeline.config.ae_scale_factor
  if height % patch != 0 or width % patch != 0:
    raise ValueError(f"height/width must be divisible by {patch}")
  grid_h, grid_w = height // patch, width // patch
  num_image_tokens = grid_h * grid_w
  latent_dim = pipeline.conditional_transformer.config.in_channels

  schedule = schedule or get_schedule_for_resolution((height, width), known_mean=mu, std=std)
  step_intervals = make_step_intervals(num_steps).to(device)

  if guidance_schedule is not None:
    gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
    if gw_per_step.shape != (num_steps,):
      raise ValueError(f"guidance_schedule must have shape ({num_steps},)")
  else:
    gw_per_step = torch.full((num_steps,), float(guidance_scale), dtype=torch.float32, device=device)

  # --- conditioning ---
  source = images_to_tensor([source_image], height, width, device)
  z_ref = encode_image_tokens(pipeline, source, patch_size=pipeline.config.patch_size)

  inputs = build_edit_inputs(pipeline, [instruction], grid_h, grid_w)
  llm_pos = pipeline._encode_text(
    inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
  )
  llm_neg = torch.zeros_like(llm_pos)  # instruction dropped; reference kept

  ref_mask = inputs["indicator"] == REFERENCE_IMAGE_INDICATOR
  tgt_mask = inputs["indicator"] == OUTPUT_IMAGE_INDICATOR
  seq_len = inputs["indicator"].shape[1]

  generator = torch.Generator(device=device)
  if seed is not None:
    generator.manual_seed(seed)
  z = torch.randn(
    1, num_image_tokens, latent_dim, dtype=torch.float32, device=device, generator=generator
  )

  do_cfg = bool((gw_per_step != 1.0).any().item())

  def velocity(llm_features: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = torch.zeros(1, seq_len, latent_dim, dtype=torch.float32, device=device)
    x[ref_mask] = z_ref.reshape(-1, latent_dim)
    x[tgt_mask] = z.reshape(-1, latent_dim)
    out = pipeline.conditional_transformer(
      llm_features=llm_features,
      x=x,
      t=t,
      position_ids=inputs["position_ids"],
      segment_ids=inputs["segment_ids"],
      indicator=inputs["indicator"],
    )
    return out[tgt_mask].reshape(1, num_image_tokens, latent_dim)

  for i in range(num_steps - 1, -1, -1):
    t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
    s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
    t = torch.full((1,), t_val, dtype=torch.float32, device=device)

    pos_v = velocity(llm_pos, t)
    if do_cfg:
      neg_v = velocity(llm_neg, t)
      gw = gw_per_step[i]
      v = gw * pos_v + (1.0 - gw) * neg_v
    else:
      v = pos_v
    z = z + v * (s_val - t_val)

  return pipeline._decode(z, grid_h=grid_h, grid_w=grid_w)[0]
