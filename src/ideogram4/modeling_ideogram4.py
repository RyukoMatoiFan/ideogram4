"""Ideogram4 transformer backbone.

The transformer consumes Qwen3-VL embeddings and flow-matching noise tokens to
produce velocity predictions on image latents.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ideogram4.constants import (
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
  REFERENCE_IMAGE_INDICATOR,
  QWEN3_VL_ACTIVATION_LAYERS,
)

# FlashAttention-2 / memory-efficient SDPA backend selection. The attention here
# runs through F.scaled_dot_product_attention; PyTorch picks FlashAttention-2 when
# no explicit mask is passed (we drop the block-diagonal segment mask whenever it
# is trivial -- see Ideogram4Transformer._build_attn_mask). attention_backend lets
# the caller force a preference order; "auto" keeps PyTorch's own choice.
try:
  from torch.nn.attention import sdpa_kernel, SDPBackend

  _SDPA_BACKENDS = {
    "flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    "efficient": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    "math": [SDPBackend.MATH],
  }
except Exception:  # pragma: no cover - older torch without torch.nn.attention
  sdpa_kernel = None
  _SDPA_BACKENDS = {}


def _attention_context(backend: str):
  """Context manager selecting the SDPA backend preference (no-op for 'auto')."""
  order = _SDPA_BACKENDS.get(backend)
  if sdpa_kernel is None or order is None:
    return contextlib.nullcontext()
  return sdpa_kernel(order)


@dataclass
class Ideogram4Config:
  emb_dim: int = 4608
  num_layers: int = 34
  num_heads: int = 18
  intermediate_size: int = 12288
  adanln_dim: int = 512

  # Latent dimension after patchification: ae_channels (32) * patch_size**2 (4) = 128.
  in_channels: int = 128

  # Hidden size of Qwen3-VL-8B-Instruct multiplied by the number of layers we extract
  # Qwen3-VL hidden size = 4096
  llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)

  rope_theta: int = 5_000_000
  mrope_section: tuple[int, ...] = (24, 20, 20)

  norm_eps: float = 1e-5

  # SDPA backend preference: "auto" lets PyTorch choose (it selects FlashAttention-2
  # when the segment mask is dropped, i.e. any single-segment batch -- always true at
  # batch size 1). "flash"/"efficient"/"math" force a preference order with fallback.
  attention_backend: str = "auto"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
  q: torch.Tensor,
  k: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  # q, k: (B, num_heads, L, head_dim); cos/sin: (B, L, head_dim).
  cos = cos.unsqueeze(1)
  sin = sin.unsqueeze(1)
  q_embed = (q * cos) + (_rotate_half(q) * sin)
  k_embed = (k * cos) + (_rotate_half(k) * sin)
  return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
  inv_freq: torch.Tensor

  def __init__(
    self,
    head_dim: int,
    base: int,
    mrope_section: tuple[int, ...],
  ) -> None:
    super().__init__()
    inv_freq = 1.0 / (
      base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.mrope_section = tuple(mrope_section)
    self.head_dim = head_dim

  @torch.no_grad()
  def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # position_ids: (B, L, 3) of int.
    assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
    batch_size, seq_len, _ = position_ids.shape

    # (3, B, inv_freq_size, L)
    pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)  # type: ignore[arg-type]
    inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(
      3, batch_size, -1, 1
    )  # type: ignore[index]
    freqs = inv_freq @ pos.unsqueeze(2)
    freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

    # interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
    freqs_t = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
      length = self.mrope_section[axis] * 3
      idx = torch.arange(offset, length, 3, device=freqs_t.device)
      freqs_t[..., idx] = freqs[axis][..., idx]

    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
  def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
    super().__init__()
    assert hidden_size % num_heads == 0
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads

    self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
    self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
    self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
    self.o = nn.Linear(hidden_size, hidden_size, bias=False)

  def forward(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape

    qkv = self.qkv(x)
    qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)

    q = self.norm_q(q)
    k = self.norm_k(k)

    # SDPA expects (B, num_heads, L, head_dim).
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    # attn_mask is the precomputed (B, 1, L, L) block-diagonal mask, or None when it
    # is trivially all-True (single segment) so SDPA can use FlashAttention-2.
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
    return self.o(out)


class Ideogram4MLP(nn.Module):
  def __init__(self, dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
  def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    norm_eps: float,
    adanln_dim: int,
  ) -> None:
    super().__init__()
    self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
    self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

    self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
    self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
    self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
    self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)

    self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

  def forward(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    adaln_input: torch.Tensor,
  ) -> torch.Tensor:
    mod = self.adaln_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
    gate_msa = torch.tanh(gate_msa)
    gate_mlp = torch.tanh(gate_mlp)
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp

    attn_out = self.attention(
      self.attention_norm1(x) * scale_msa,
      attn_mask=attn_mask,
      cos=cos,
      sin=sin,
    )
    x = x + gate_msa * self.attention_norm2(attn_out)
    x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    return x


def _sinusoidal_embedding(
  t: torch.Tensor, dim: int, scale: float = 1e4
) -> torch.Tensor:
  t = t.to(torch.float32)
  half = dim // 2
  freq = math.log(scale) / (half - 1)
  freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)  # type: ignore[assignment]
  emb = t.unsqueeze(-1) * freq
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
  if dim % 2 == 1:
    emb = F.pad(emb, (0, 1))
  return emb


class Ideogram4EmbedScalar(nn.Module):
  def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
    super().__init__()
    self.dim = dim
    self.range_min, self.range_max = input_range
    assert self.range_max > self.range_min
    self.mlp_in = nn.Linear(dim, dim, bias=True)
    self.mlp_out = nn.Linear(dim, dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x is shape (..., 1) or (...,) holding a scalar per token.
    x = x.to(torch.float32)
    scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
    emb = _sinusoidal_embedding(scaled, self.dim)
    emb = emb.to(
      getattr(self.mlp_in, "compute_dtype", None) or self.mlp_in.weight.dtype
    )
    emb = F.silu(self.mlp_in(emb))
    return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
  def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
    super().__init__()
    self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.linear = nn.Linear(hidden_size, out_channels, bias=True)
    self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    scale = 1.0 + self.adaln_modulation(F.silu(c))
    return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nn.Module):
  """Ideogram 4 flow-matching transformer."""

  def __init__(self, config: Ideogram4Config) -> None:
    super().__init__()
    self.config = config
    # Opt-in activation checkpointing: recompute each block in backward instead of
    # storing its activations. Cuts training activation memory ~linearly in depth
    # (the dominant term), at ~1.2-1.3x compute -- the enabler for higher batch /
    # resolution / LoRA rank. No effect at inference (only active under .training).
    self.gradient_checkpointing = False

    head_dim = config.emb_dim // config.num_heads

    self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
    self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
    self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
    self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
    self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)

    self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

    self.rotary_emb = Ideogram4MRoPE(
      head_dim=head_dim,
      base=config.rope_theta,
      mrope_section=config.mrope_section,
    )

    self.layers = nn.ModuleList(
      [
        Ideogram4TransformerBlock(
          hidden_size=config.emb_dim,
          intermediate_size=config.intermediate_size,
          num_heads=config.num_heads,
          norm_eps=config.norm_eps,
          adanln_dim=config.adanln_dim,
        )
        for _ in range(config.num_layers)
      ]
    )

    self.final_layer = Ideogram4FinalLayer(
      hidden_size=config.emb_dim,
      out_channels=config.in_channels,
      adanln_dim=config.adanln_dim,
    )

  @property
  def device(self) -> torch.device:
    return next(self.parameters()).device

  def _build_attn_mask(self, segment_ids: torch.Tensor) -> Optional[torch.Tensor]:
    """Block-diagonal attention mask from segment ids, or None when trivial.

    When every token shares a segment id (always true at batch size 1, and for any
    fully-visible sequence) the mask is all-True, so we return None and let SDPA
    dispatch to FlashAttention-2 (which rejects explicit masks). Computed once per
    forward here instead of redundantly inside all 34 attention layers.
    """
    if bool((segment_ids == segment_ids[:, :1]).all()):
      return None
    return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

  def forward(
    self,
    *,
    llm_features: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    position_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    indicator: torch.Tensor,
  ) -> torch.Tensor:
    """Velocity prediction.

    Args:
      llm_features: (B, L, llm_features_dim) Qwen3-VL conditioning features.
      x: (B, L, in_channels) noise tokens.
      t: (B,) or (B, L) flow-matching time in [0, 1].
      position_ids: (B, L, 3) (t, h, w) positions for MRoPE.
      segment_ids: (B, L) sample id within a packed batch.
      indicator: (B, L) per-token role: LLM_TOKEN_INDICATOR or OUTPUT_IMAGE_INDICATOR.

    Returns:
      (B, L, in_channels) velocity prediction in float32. Only the positions
      with ``indicator == OUTPUT_IMAGE_INDICATOR`` are meaningful.
    """
    batch_size, seq_len, in_channels = x.shape
    assert in_channels == self.config.in_channels

    param_dtype = (
      getattr(self.input_proj, "compute_dtype", None) or self.input_proj.weight.dtype
    )
    x = x.to(param_dtype)
    t = t.to(param_dtype)
    llm_features = llm_features.to(param_dtype)

    indicator = indicator.to(torch.long)
    llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
    # Both the noised target latents and any clean reference-image latents are real
    # image content and must pass through input_proj. At plain text-to-image inference
    # no REFERENCE_IMAGE_INDICATOR tokens exist, so this reduces to the original mask.
    image_latent_mask = (
      (indicator == OUTPUT_IMAGE_INDICATOR) | (indicator == REFERENCE_IMAGE_INDICATOR)
    ).to(x.dtype).unsqueeze(-1)

    llm_features = llm_features * llm_token_mask
    x = x * image_latent_mask

    x = self.input_proj(x) * image_latent_mask

    # Keep shape (B, 1, ...) when t is per-sample so downstream adaln_modulation
    # projections don't pay for L identical copies.
    t_cond = self.t_embedding(t)
    if t.dim() == 1:
      t_cond = t_cond.unsqueeze(1)
    adaln_input = F.silu(self.adaln_proj(t_cond))

    llm_features = self.llm_cond_norm(llm_features)
    llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

    h = x + llm_features

    # Mark image-latent tokens with the "image" embedding. With the pretrained
    # nn.Embedding(2), target and reference share the "image" slot (1) and are told
    # apart only by their MRoPE frame index. If the embedding has been expanded to 3
    # rows (see train_edit.expand_reference_embedding, for full-rank finetuning),
    # reference tokens get their own slot (2). Both layouts are supported so stock
    # checkpoints keep loading unchanged.
    if self.embed_image_indicator.num_embeddings >= 3:
      idx = torch.zeros_like(indicator)
      idx = torch.where(indicator == OUTPUT_IMAGE_INDICATOR, torch.ones_like(idx), idx)
      idx = torch.where(
        indicator == REFERENCE_IMAGE_INDICATOR, torch.full_like(idx, 2), idx
      )
      image_indicator_embedding = self.embed_image_indicator(idx)
    else:
      image_indicator_embedding = self.embed_image_indicator(
        (image_latent_mask.squeeze(-1) > 0).to(torch.long)
      )
    h = h + image_indicator_embedding

    cos, sin = self.rotary_emb(position_ids)
    cos = cos.to(h.dtype)
    sin = sin.to(h.dtype)

    # Compute the attention mask once (was rebuilt in every layer); None => flash.
    attn_mask = self._build_attn_mask(segment_ids)

    with _attention_context(self.config.attention_backend):
      for layer in self.layers:
        if self.gradient_checkpointing and self.training:
          h = torch.utils.checkpoint.checkpoint(
            layer, h, attn_mask, cos, sin, adaln_input, use_reentrant=False
          )
        else:
          h = layer(h, attn_mask=attn_mask, cos=cos, sin=sin, adaln_input=adaln_input)

    out = self.final_layer(h, c=adaln_input)
    return out.to(torch.float32)
