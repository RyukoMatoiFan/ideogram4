"""ByG training step -- unpaired image-editing objective.

Bootstrap Your Generator (ByG) trains an instruction-based image editor without
paired data.  It uses the frozen T2I base as a semantic direction prior and a
cycle-consistency loss to ground the edits.

Math convention (same as ``train_edit`` and ``edit_sampler``):
  t=0 is NOISE, t=1 is DATA.
  x_t = t * x0 + (1 - t) * eps        (forward diffusion)
  velocity target  v = x0 - eps
  Euler update toward data: z <- z + v * (s - t)   for s > t.

Per-step forward count (default cfg, cfg.t2i_cfg_scale=1.0, p_identity=1.0):
  1  EMA rollout (bootstrap_steps Euler steps, each = 1 transformer call)
  1  G_edit forward  (v_fwd)
  2  G_t2i forwards  (v_src, v_tgt) -- no_grad, adapters off
  1  G_edit forward  (v_rev)         -- cycle
  2  G_t2i forwards  (v_src2, v_tgt2) -- no_grad, adapters off, reverse prior
  1  G_edit forward  (L_id)           -- identity (p_identity gate)
  ---
  Total: 3 with grad + (bootstrap_steps + 4) without grad.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ideogram4.byg.ema import LoraEMA
from ideogram4.byg.rollout import ema_rollout
from ideogram4.byg.sequences import build_t2i_sequence_meta
from ideogram4.constants import OUTPUT_IMAGE_INDICATOR, REFERENCE_IMAGE_INDICATOR
from ideogram4.scheduler import LogitNormalSchedule
from ideogram4.train_edit import build_edit_sequence_meta


# =========================================================================== #
# Internal helpers
# =========================================================================== #

def _t2i_forward(
  transformer: torch.nn.Module,
  z_noised: torch.Tensor,
  llm: torch.Tensor,
  t: torch.Tensor,
  meta: dict,
) -> torch.Tensor:
  """Single no_grad T2I forward; returns velocity over image tokens.

  Parameters
  ----------
  transformer:
    Transformer in base-only mode (LoRA scale=0 set externally by caller).
  z_noised:
    (1, n, 128) noised latent tokens.
  llm:
    (1, seq_len, llm_dim) full-sequence LLM features (text positions filled,
    image positions zero).
  t:
    (1,) timestep.
  meta:
    Metadata dict from ``build_t2i_sequence_meta``.

  Returns
  -------
  (1, n, 128) velocity prediction over image tokens.
  """
  num_text = meta["num_text"]
  n = meta["n"]
  latent_dim = z_noised.shape[-1]
  device = z_noised.device
  seq_len = meta["indicator"].shape[1]

  # Assemble x: text slots zero, image slots = noised latent.
  x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
  tgt_mask = meta["indicator"] == OUTPUT_IMAGE_INDICATOR
  x[tgt_mask] = z_noised.reshape(n, latent_dim)

  out = transformer(
    llm_features=llm,
    x=x,
    t=t,
    position_ids=meta["position_ids"],
    segment_ids=meta["segment_ids"],
    indicator=meta["indicator"],
  )
  return out[tgt_mask].reshape(1, n, latent_dim)


def _edit_forward(
  transformer: torch.nn.Module,
  z_noised: torch.Tensor,
  z_ref: torch.Tensor,
  llm: torch.Tensor,
  t: torch.Tensor,
  meta: dict,
) -> torch.Tensor:
  """Single edit forward; returns velocity over target image tokens.

  Parameters
  ----------
  z_noised:
    (1, n, 128) noised target latent tokens.
  z_ref:
    (1, n, 128) clean reference latent tokens.
  llm:
    (1, seq_len, llm_dim) full-sequence LLM features.
  meta:
    Metadata dict from ``build_edit_sequence_meta``.
  """
  n = z_noised.shape[1]
  latent_dim = z_noised.shape[-1]
  device = z_noised.device
  indicator = meta["indicator"]
  seq_len = indicator.shape[1]

  ref_mask = indicator == REFERENCE_IMAGE_INDICATOR
  tgt_mask = indicator == OUTPUT_IMAGE_INDICATOR

  x = torch.zeros(1, seq_len, latent_dim, device=device, dtype=torch.float32)
  x[ref_mask] = z_ref.reshape(n, latent_dim).to(torch.float32)
  x[tgt_mask] = z_noised.reshape(n, latent_dim).to(torch.float32)

  out = transformer(
    llm_features=llm,
    x=x,
    t=t,
    position_ids=meta["position_ids"],
    segment_ids=meta["segment_ids"],
    indicator=indicator,
  )
  return out[tgt_mask].reshape(1, n, latent_dim)


def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """1 - cosine_similarity over the flattened last two dims (n, 128) per sample.

  Input shape: (B, n, 128).  Returns scalar (mean over batch).

  L_dir = 1 - cos(a, b)  where cos is over the flattened n*128 vector.
  """
  # Flatten to (B, n*128).
  a_flat = a.reshape(a.shape[0], -1)
  b_flat = b.reshape(b.shape[0], -1)
  cos_sim = F.cosine_similarity(a_flat, b_flat, dim=-1)  # (B,)
  return (1.0 - cos_sim).mean()


def _prior_loss(
  v_fwd: torch.Tensor,
  v_src: torch.Tensor,
  v_tgt: torch.Tensor,
  alpha_mse: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute L_dir + alpha_mse * L_mse (the prior loss for one direction).

  L_dir = 1 - cos(v_fwd - v_src, v_tgt - v_src)
  L_mse = mean ||v_fwd - v_tgt||^2

  Returns (L_prior, cos_dir).
  """
  delta_fwd = v_fwd - v_src
  delta_tgt = v_tgt - v_src
  l_dir = _cosine_loss(delta_fwd, delta_tgt)
  l_mse = F.mse_loss(v_fwd, v_tgt)
  # cos_dir is reported as a health metric (positive = aligned with target direction).
  cos_dir = 1.0 - l_dir.detach()
  return l_dir + alpha_mse * l_mse, cos_dir


# =========================================================================== #
# Main ByG training step
# =========================================================================== #

def byg_training_step(
  transformer: torch.nn.Module,
  ema: LoraEMA,
  sample: dict[str, torch.Tensor],
  *,
  schedule: LogitNormalSchedule,
  cfg: object,
  generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
  """One ByG training step (unpaired editing).

  Parameters
  ----------
  transformer:
    ``Ideogram4Transformer`` with injected LoRA adapters.
  ema:
    ``LoraEMA`` wrapping the same adapters.
  sample:
    Dict with keys (all on the same device as the transformer):
      ``z_src``     : (n, 128) source latent tokens (clean).
      ``llm_p_src`` : (num_text_p_src, llm_dim) source-caption text features.
      ``llm_p_tgt`` : (num_text_p_tgt, llm_dim) target-caption text features.
      ``llm_c``     : (num_text_c, llm_dim) forward instruction features.
      ``llm_c_rev`` : (num_text_c_rev, llm_dim) reverse instruction features.
      ``grid_h``    : int tensor or scalar int.
      ``grid_w``    : int tensor or scalar int.
  schedule:
    ``LogitNormalSchedule`` for timestep sampling.
  cfg:
    Duck-typed config object exposing:
      ``lambda_prior``   (float) -- weight for both direction prior losses.
      ``lambda_id``      (float) -- weight for identity loss.
      ``alpha_mse``      (float) -- MSE coefficient inside the prior loss.
      ``p_identity``     (float) -- probability of applying the identity branch.
      ``bootstrap_steps`` (int)  -- Euler steps for the EMA rollout.
      ``detach_rollout`` (bool)  -- if True, straight-through hybrid is used.
      ``t2i_cfg_scale``  (float) -- T2I CFG scale (1.0 = no CFG, single forward).
                                    Values != 1.0 are a TODO hook; unconditional
                                    branch is not implemented here.
  generator:
    Optional seeded ``torch.Generator`` for reproducible noise.

  Returns
  -------
  dict with keys:
    ``loss``         : scalar grad tensor -- total training loss.
    ``l_cycle``      : cycle loss component.
    ``l_prior_fwd``  : forward direction prior loss.
    ``l_prior_rev``  : reverse direction prior loss.
    ``l_id``         : identity loss (0 if identity branch not sampled).
    ``cos_dir``      : health metric (cosine alignment of v_fwd with target dir).

  Per-step forward count (default cfg):
    - bootstrap_steps no-grad EMA rollout forwards (G_EMA, edit-seq)
    - 1 grad forward: G_edit -> v_fwd
    - 2 no-grad forwards: G_t2i -> v_src, v_tgt
    - 1 grad forward: G_edit -> v_rev (cycle)
    - 2 no-grad forwards: G_t2i -> v_src2, v_tgt2 (reverse prior)
    - 1 grad forward (conditional on p_identity): G_edit -> L_id
    Total with grad: 2 or 3; without grad: bootstrap_steps + 4.
  """
  z_src = sample["z_src"]  # (n, 128)
  llm_p_src = sample["llm_p_src"]  # (num_text_p_src, llm_dim)
  llm_p_tgt = sample["llm_p_tgt"]  # (num_text_p_tgt, llm_dim)
  llm_c = sample["llm_c"]          # (num_text_c, llm_dim)
  llm_c_rev = sample["llm_c_rev"]  # (num_text_c_rev, llm_dim)
  grid_h = int(sample["grid_h"])
  grid_w = int(sample["grid_w"])

  device = z_src.device
  n = grid_h * grid_w
  latent_dim = z_src.shape[-1]
  llm_dim = llm_c.shape[-1]

  # ------------------------------------------------------------------ #
  # Step 1 -- Sample timestep t from the training schedule.
  # ------------------------------------------------------------------ #
  u = torch.rand((1,), device=device, generator=generator)
  t = schedule(u).to(torch.float32)          # (1,) float32
  t_val = float(t.item())

  # ------------------------------------------------------------------ #
  # Step 1b -- EMA bootstrap rollout (no_grad, G_EMA, edit-seq).
  # Produces y_t (noisy at t) and y0 (EMA clean estimate).
  # ------------------------------------------------------------------ #
  y_t, y0 = ema_rollout(
    transformer,
    ema,
    z_src,
    llm_c,
    t_val,
    schedule=schedule,
    steps=cfg.bootstrap_steps,
    grid_h=grid_h,
    grid_w=grid_w,
    generator=generator,
  )
  # y_t: (1, n, 128), y0: (1, n, 128), both no-grad.

  # ------------------------------------------------------------------ #
  # Build sequence metadata (cached for the whole step).
  # ------------------------------------------------------------------ #
  num_text_c = llm_c.shape[0]
  num_text_c_rev = llm_c_rev.shape[0]
  num_text_p_src = llm_p_src.shape[0]
  num_text_p_tgt = llm_p_tgt.shape[0]

  edit_meta_c = build_edit_sequence_meta(num_text_c, grid_h, grid_w, device)
  edit_meta_c_rev = build_edit_sequence_meta(num_text_c_rev, grid_h, grid_w, device)
  t2i_meta_src = build_t2i_sequence_meta(num_text_p_src, grid_h, grid_w, device)
  t2i_meta_tgt = build_t2i_sequence_meta(num_text_p_tgt, grid_h, grid_w, device)

  # Assemble the LLM feature tensors for the full sequences.
  def _make_llm_edit(llm_text: torch.Tensor, meta: dict) -> torch.Tensor:
    seq_len = meta["indicator"].shape[1]
    nt = llm_text.shape[0]
    llm = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
    llm[0, :nt] = llm_text.to(torch.float32)
    return llm

  def _make_llm_t2i(llm_text: torch.Tensor, meta: dict) -> torch.Tensor:
    seq_len = meta["indicator"].shape[1]
    nt = llm_text.shape[0]
    llm = torch.zeros(1, seq_len, llm_dim, device=device, dtype=torch.float32)
    llm[0, :nt] = llm_text.to(torch.float32)
    return llm

  llm_c_full = _make_llm_edit(llm_c, edit_meta_c)
  llm_c_rev_full = _make_llm_edit(llm_c_rev, edit_meta_c_rev)
  llm_p_src_full = _make_llm_t2i(llm_p_src, t2i_meta_src)
  llm_p_tgt_full = _make_llm_t2i(llm_p_tgt, t2i_meta_tgt)

  # ------------------------------------------------------------------ #
  # Step 2 -- Forward with grad: G_edit(y_t, t | c, ref=z_src) -> v_fwd
  # ------------------------------------------------------------------ #
  z_src_1 = z_src.unsqueeze(0)   # (1, n, 128)
  v_fwd = _edit_forward(
    transformer,
    y_t,
    z_src_1,
    llm_c_full,
    t,
    edit_meta_c,
  )  # (1, n, 128), has grad

  # ------------------------------------------------------------------ #
  # Step 3 -- T2I prior: G(y_t, t | p_src) and G(y_t, t | p_tgt).
  # No grad, LoRA off (G_t2i = base only).
  # ------------------------------------------------------------------ #
  with torch.no_grad(), ema.adapters_off():
    v_src = _t2i_forward(transformer, y_t, llm_p_src_full, t, t2i_meta_src)
    v_tgt = _t2i_forward(transformer, y_t, llm_p_tgt_full, t, t2i_meta_tgt)
  # v_src, v_tgt: (1, n, 128), no grad.

  # ------------------------------------------------------------------ #
  # Step 4 -- Prior loss (forward direction).
  # L_dir = 1 - cos(v_fwd - v_src, v_tgt - v_src)
  # L_mse = ||v_fwd - v_tgt||^2
  # L_prior_fwd = L_dir + alpha_mse * L_mse
  # ------------------------------------------------------------------ #
  l_prior_fwd, cos_dir = _prior_loss(v_fwd, v_src.detach(), v_tgt.detach(), cfg.alpha_mse)

  # ------------------------------------------------------------------ #
  # Step 5 -- Straight-through hybrid.
  # yhat = y_t + (1 - t) * v_fwd          (one-step x0 estimate)
  # yhat_hyb = sg(y0) + (yhat - sg(yhat)) (value from EMA, grad from v_fwd)
  # ------------------------------------------------------------------ #
  yhat = y_t + (1.0 - t_val) * v_fwd   # (1, n, 128), has grad through v_fwd

  if cfg.detach_rollout:
    # Straight-through: value = EMA clean estimate y0, gradient = v_fwd path.
    yhat_hyb = y0.detach() + (yhat - yhat.detach())
  else:
    yhat_hyb = yhat

  # ------------------------------------------------------------------ #
  # Step 6 -- Cycle loss (paper Algorithm 1: reverse pass reconstructs SOURCE).
  # The noised latent being denoised is the SOURCE (x_t = t2*z_src + (1-t2)*eps2);
  # the edited estimate yhat_hyb is the reference/image-condition; the reverse
  # instruction c_rev drives the un-edit. Gradient routes into v_fwd ONLY through
  # the yhat_hyb reference (x2 is detached from v_fwd since z_src is data).
  # x2 = t2 * z_src + (1 - t2) * eps2 ;  v_rev = G_edit(x2, t2 | c_rev, ref=yhat_hyb)
  # L_cycle = ||v_rev - (z_src - eps2)||^2
  # ------------------------------------------------------------------ #
  eps2 = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
  u2 = torch.rand((1,), device=device, generator=generator)
  t2 = schedule(u2).to(torch.float32)
  t2_val = float(t2.item())

  x2 = t2_val * z_src_1 + (1.0 - t2_val) * eps2  # noised SOURCE (1, n, 128)
  v_cycle_target = (z_src_1 - eps2).detach()      # velocity that reconstructs the source

  v_rev = _edit_forward(
    transformer,
    x2,
    yhat_hyb,   # reference = yhat_hyb (the estimated edited image; carries v_fwd grad)
    llm_c_rev_full,
    t2,
    edit_meta_c_rev,
  )  # (1, n, 128), grad flows into v_fwd through the yhat_hyb reference

  l_cycle = F.mse_loss(v_rev, v_cycle_target)

  # ------------------------------------------------------------------ #
  # Step 7 -- Reverse prior loss.
  # Roles swapped: source prompt -> "target" direction, target prompt -> "source".
  # Evaluated at (x2, t2) with two more no-grad T2I forwards.
  # ------------------------------------------------------------------ #
  # Build new t2i sequence metadata for x2 (same grid, same text lengths).
  t2i_meta_src2 = build_t2i_sequence_meta(num_text_p_src, grid_h, grid_w, device)
  t2i_meta_tgt2 = build_t2i_sequence_meta(num_text_p_tgt, grid_h, grid_w, device)
  llm_p_src2 = _make_llm_t2i(llm_p_src, t2i_meta_src2)
  llm_p_tgt2 = _make_llm_t2i(llm_p_tgt, t2i_meta_tgt2)

  with torch.no_grad(), ema.adapters_off():
    # Evaluate the T2I base at (x2, t2) for the reverse prior.
    v_src2 = _t2i_forward(transformer, x2.detach(), llm_p_src2, t2, t2i_meta_src2)
    v_tgt2 = _t2i_forward(transformer, x2.detach(), llm_p_tgt2, t2, t2i_meta_tgt2)

  # Reverse direction: the "edited" direction is p_src (going back to source),
  # so the roles of src/tgt prompts are swapped compared to the forward prior.
  # v_rev should point from p_tgt direction toward p_src direction.
  l_prior_rev, _ = _prior_loss(v_rev, v_tgt2.detach(), v_src2.detach(), cfg.alpha_mse)

  # ------------------------------------------------------------------ #
  # Step 8 -- Identity loss (gated by p_identity).
  # eps3, t3; x3 = t3 * z_src + (1 - t3) * eps3
  # G_edit(x3, t3 | c_rev, ref=z_src) should predict z_src - eps3.
  # ------------------------------------------------------------------ #
  do_identity = (
    torch.rand((), device=device, generator=generator).item() < cfg.p_identity
  )
  if do_identity:
    eps3 = torch.randn(1, n, latent_dim, device=device, dtype=torch.float32, generator=generator)
    u3 = torch.rand((1,), device=device, generator=generator)
    t3 = schedule(u3).to(torch.float32)
    t3_val = float(t3.item())

    x3 = t3_val * z_src_1 + (1.0 - t3_val) * eps3
    v_id_target = (z_src_1 - eps3).detach()

    v_id = _edit_forward(
      transformer,
      x3,
      z_src_1,           # reference = z_src (same as input)
      llm_c_rev_full,
      t3,
      edit_meta_c_rev,
    )
    l_id = F.mse_loss(v_id, v_id_target)
  else:
    l_id = torch.zeros((), device=device, dtype=torch.float32)

  # ------------------------------------------------------------------ #
  # Step 9 -- Total loss.
  # L = L_cycle + lambda_prior * (L_prior_fwd + L_prior_rev) + lambda_id * L_id
  # ------------------------------------------------------------------ #
  loss = (
    l_cycle
    + cfg.lambda_prior * (l_prior_fwd + l_prior_rev)
    + cfg.lambda_id * l_id
  )

  return {
    "loss": loss,
    "l_cycle": l_cycle,
    "l_prior_fwd": l_prior_fwd,
    "l_prior_rev": l_prior_rev,
    "l_id": l_id,
    "cos_dir": cos_dir,
  }
