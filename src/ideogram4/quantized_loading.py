from __future__ import annotations

import warnings

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


_BNB_SIBLING_SUFFIXES = (
  ".absmax",
  ".quant_map",
  ".nested_absmax",
  ".nested_quant_map",
)

# Largest magnitude representable by the e4m3 float8 format. Per-row weight
# scales map each row's max abs value onto this so we use the full range.
FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# Marker written into the text encoder's config.json so the loader knows to take
# the custom weight-only FP8 path instead of transformers' from_pretrained.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"


def _e4m3_positive_grid(device: torch.device) -> torch.Tensor:
  """Sorted finite non-negative magnitudes representable in e4m3 (float8_e4m3fn).
  Used to round onto the EXACT grid for stochastic rounding."""
  codes = torch.arange(256, dtype=torch.uint8, device=device)
  vals = codes.view(FP8_WEIGHT_DTYPE).to(torch.float32)
  return torch.unique(vals[torch.isfinite(vals)].abs())  # ascending, includes 0.0


def _stochastic_round_to_e4m3(q: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
  """Round q (already scaled into the e4m3 range) onto the e4m3 grid with STOCHASTIC
  rounding: round up to the upper grid neighbour with probability == fractional distance,
  so E[round(q)] == q (unbiased). Returns a float8_e4m3fn tensor."""
  grid = _e4m3_positive_grid(q.device)
  sign = torch.sign(q)
  aq = q.abs().clamp(max=FP8_E4M3_MAX)
  idx = torch.searchsorted(grid, aq)                          # first grid value >= aq
  hi = grid[idx.clamp(max=grid.numel() - 1)]
  lo = grid[(idx - 1).clamp(min=0)]
  span = (hi - lo).clamp_min(torch.finfo(torch.float32).tiny)
  p_up = ((aq - lo) / span).clamp(0.0, 1.0)                   # P(round to hi); 1.0 when aq on grid
  u = torch.rand(aq.shape, device=q.device, generator=generator)
  return (sign * torch.where(u < p_up, hi, lo)).to(FP8_WEIGHT_DTYPE)


def quantize_to_fp8(
  weight: torch.Tensor, *, stochastic: bool = True, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Inverse of the ``Fp8Linear`` dequant: quantize a bf16/f32 Linear weight ``[out, in]``
  to per-output-row symmetric e4m3. Returns ``(weight_fp8 [out,in] float8_e4m3fn,
  weight_scale [out] float32)`` such that ``weight_fp8.to(f32) * weight_scale[:, None]``
  reconstructs ``weight`` (matches ``dequantize_fp8_transformer`` / ``Fp8Linear.forward``).

  ``stochastic=True`` rounds onto the e4m3 grid with stochastic rounding (unbiased -> removes
  the correlated round-to-nearest bias that does NOT average out across a GEMM); ``False`` = RTN.
  Scales are recomputed from THIS weight's per-row absmax (never reuse pre-training scales).
  """
  w = weight.detach().to(torch.float32)
  scale = (w.abs().amax(dim=1) / FP8_E4M3_MAX).clamp_min(1e-8)   # [out]
  q = w / scale.unsqueeze(1)
  if stochastic:
    w_fp8 = _stochastic_round_to_e4m3(q, generator)
  else:
    w_fp8 = q.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
  return w_fp8, scale


def fake_quantize_fp8(
  weight: torch.Tensor, *, stochastic: bool = False, generator: torch.Generator | None = None
) -> torch.Tensor:
  """Straight-through fp8 fake-quantization for QAT (quantization-aware training).

  Forward returns ``dequant(quantize(weight))`` so the model sees EXACTLY the deployed fp8
  rounding; backward is identity to ``weight`` (straight-through estimator), so the bf16 master
  weight still receives real gradients. The forward MUST stay bit-identical to the exporter --
  it reuses ``quantize_to_fp8`` -- otherwise the QAT/inference (sim-deploy) gap reopens. Default
  ``stochastic=False`` (RTN): the model adapts to the deterministic deployed rounding, so export
  with the same RTN. Apply only to layers that will be exported fp8 (same keep-list as export).
  """
  wq, scale = quantize_to_fp8(weight, stochastic=stochastic, generator=generator)
  deq = (wq.to(torch.float32) * scale.unsqueeze(1)).to(weight.dtype)
  return weight + (deq - weight).detach()  # STE: forward=quantized, grad flows to weight


class QATFp8Linear(nn.Module):
  """Fake-quantized Linear for QAT: keeps a trainable bf16 MASTER weight, but the forward uses
  ``fake_quantize_fp8(weight)`` (straight-through) so the model trains against the EXACT fp8
  rounding it will be exported with -- closing the post-training quantization gap. State-dict keys
  match ``nn.Linear`` (``weight``/``bias``) so a bf16 checkpoint loads into it and ``state_dict()``
  writes the bf16 master. ``stochastic`` MUST match the exporter (RTN by default)."""

  def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, *, stochastic: bool = False):
    super().__init__()
    self.weight = nn.Parameter(weight)
    self.bias = nn.Parameter(bias) if bias is not None else None
    self.stochastic = stochastic

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = fake_quantize_fp8(self.weight, stochastic=self.stochastic)
    return F.linear(x, w.to(x.dtype), self.bias)


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if any key looks like a bnb 4-bit quant_state sibling."""
  return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_bnb4bit(
  module: nn.Module,
  compute_dtype: torch.dtype,
  *,
  quant_type: str = "nf4",
  compress_statistics: bool = False,
) -> None:
  for name, child in list(module.named_children()):
    if isinstance(child, nn.Linear):
      new_linear = bnb.nn.Linear4bit(
        child.in_features,
        child.out_features,
        bias=child.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
      )
      setattr(module, name, new_linear)
    else:
      swap_linears_to_bnb4bit(
        child,
        compute_dtype,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
      )


def load_bnb4bit_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> None:
  consumed: set[str] = set()
  for full_name, tensor in state_dict.items():
    if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
      continue
    parent_path, _, param_name = full_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    current = parent._parameters.get(param_name)
    if not isinstance(current, bnb.nn.Params4bit):
      continue
    prefix = full_name + "."
    quantized_stats = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    # bnb's from_prequantized pops keys it consumes from the dict, so snapshot
    # the names first.
    consumed.add(full_name)
    consumed.update(quantized_stats.keys())
    parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
      data=tensor,
      quantized_stats=quantized_stats,
      requires_grad=False,
      device=device,
    )

  remaining = {k: v for k, v in state_dict.items() if k not in consumed}
  for k in list(remaining):
    if remaining[k].is_floating_point():
      remaining[k] = remaining[k].to(device=device, dtype=dtype)
    else:
      remaining[k] = remaining[k].to(device=device)

  missing, unexpected = model.load_state_dict(remaining, strict=False)
  # Quantized weights are loaded via from_prequantized above, so they appear in
  # `missing` from load_state_dict's perspective — filter those out.
  real_missing = [m for m in missing if m not in consumed]
  if real_missing:
    raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
  if unexpected:
    raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")

  for p in model.parameters():
    if isinstance(p, bnb.nn.Params4bit):
      continue
    if p.is_floating_point() and p.dtype != dtype:
      p.data = p.data.to(dtype=dtype)
    if p.device != device:
      p.data = p.data.to(device=device)
  for name, b in list(model.named_buffers()):
    if b.is_floating_point() and b.dtype != dtype:
      parent_path, _, leaf = name.rpartition(".")
      parent = model.get_submodule(parent_path) if parent_path else model
      parent.register_buffer(
        leaf,
        b.to(device=device, dtype=dtype),
        persistent=leaf not in parent._non_persistent_buffers_set,
      )
    elif b.device != device:
      parent_path, _, leaf = name.rpartition(".")
      parent = model.get_submodule(parent_path) if parent_path else model
      parent.register_buffer(
        leaf,
        b.to(device=device),
        persistent=leaf not in parent._non_persistent_buffers_set,
      )


# ---------------------------------------------------------------------------
# Weight-only FP8 (e4m3)
#
# Activations stay in the compute dtype (e.g. bfloat16); only Linear weights are
# stored as float8 with a per-output-channel (per-row) float32 scale. At forward
# time the weight is dequantized back to the compute dtype and a normal bf16
# matmul runs, so this needs no FP8 tensor-core hardware and works on any device
# that can store float8 (CPU included). The win is ~2x smaller Linear weights.
# ---------------------------------------------------------------------------


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint carries weight-only FP8 Linear weights."""
  return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
    v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
  )


class Fp8Linear(nn.Module):
  """Linear layer holding an e4m3 float8 weight + per-row float32 scale.

  The weight and scale are registered as buffers (not parameters) so they load
  via ``load_state_dict`` and are excluded from optimizer/grad machinery. The
  dequantized matmul runs in ``compute_dtype``.
  """

  weight: torch.Tensor
  weight_scale: torch.Tensor
  bias: torch.Tensor | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool,
    compute_dtype: torch.dtype,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    self.register_buffer(
      "weight",
      torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
    )
    self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
    if bias:
      self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
    else:
      self.bias = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


def swap_linears_to_fp8(
  module: nn.Module,
  state_dict: dict[str, torch.Tensor],
  compute_dtype: torch.dtype,
  *,
  prefix: str = "",
) -> None:
  """Replace each ``nn.Linear`` that has a saved FP8 scale with an ``Fp8Linear``.

  Gating on the presence of ``<name>.weight_scale`` means only layers that were
  actually quantized at save time are swapped; everything else loads normally in
  the compute dtype.
  """
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    if (
      isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
    ):
      setattr(
        module,
        name,
        Fp8Linear(
          child.in_features,
          child.out_features,
          bias=child.bias is not None,
          compute_dtype=compute_dtype,
        ),
      )
    else:
      swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_fp8_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = False,
  strict: bool = True,
) -> None:
  """Load a weight-only FP8 checkpoint into ``model``.

  ``model`` must already have its FP8 Linear layers swapped in (see
  ``swap_linears_to_fp8``). FP8 weights are kept as float8, scales stay float32,
  and every other floating tensor is cast to ``dtype``.

  ``assign=True`` replaces the module's tensors with the prepared ones rather than
  copying into them. Use it when the model was built with ``from_config`` so the
  non-quantized params take the loaded dtype directly and computed non-persistent
  buffers (e.g. rotary caches) are left untouched. With ``assign=False`` (default),
  the caller must have already put the unquantized params in ``dtype``.

  ``strict=False`` downgrades missing keys to a warning (e.g. tied weights that a
  ``transformers`` model resolves itself); unexpected keys always raise.
  """
  prepared: dict[str, torch.Tensor] = {}
  for k, v in state_dict.items():
    if v.dtype == FP8_WEIGHT_DTYPE:
      prepared[k] = v.to(device=device)
    elif k.endswith(FP8_SCALE_SUFFIX):
      prepared[k] = v.to(device=device, dtype=torch.float32)
    elif v.is_floating_point():
      prepared[k] = v.to(device=device, dtype=dtype)
    else:
      prepared[k] = v.to(device=device)

  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if unexpected:
    raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

  model.to(device)
