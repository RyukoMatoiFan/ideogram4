"""Minimal trainable LoRA for the Ideogram 4 transformer.

Wraps the attention/MLP Linears of ``Ideogram4Transformer`` with low-rank adapters
and freezes the base weights. The adapter math matches the inference-time hook in
``ideogram_region_lora/server/lora_apply.py``::

    y = base(x) + (alpha / rank) * (x @ A^T) @ B^T

and ``save_lora`` writes ai-toolkit-compatible keys
``diffusion_model.layers.{i}.{sub}.lora_{A,B}.weight``. With the default
``alpha == rank`` the geometric scale is 1.0, exactly what ``load_lora_map``
assumes -- so a checkpoint trained here loads straight into the existing inference
stack (``load_lora_map`` -> ``apply_lora``) with no conversion.

Train on a bf16 (not fp8/nf4) base: load the pipeline with ``dtype=torch.bfloat16``
so the wrapped Linears are ordinary ``nn.Linear``.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn

# Sub-paths (relative to each transformer block) of the Linears we adapt.
DEFAULT_TARGETS = (
  "attention.qkv",
  "attention.o",
  "feed_forward.w1",
  "feed_forward.w2",
  "feed_forward.w3",
)


class LoRALinear(nn.Module):
  """``nn.Linear`` wrapped with a frozen base and a trainable low-rank delta."""

  def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
    super().__init__()
    self.base = base
    self.base.requires_grad_(False)  # no-op for fp8/nf4 (weights are buffers/frozen)
    self.rank = rank
    self.scale = alpha / rank

    in_features = base.in_features
    out_features = base.out_features
    # Adapters live in the compute dtype, not the (possibly float8/4-bit) base
    # weight dtype: a quantized base dequantizes to compute_dtype in its forward,
    # and the low-rank delta is added there.
    param_dtype = getattr(base, "compute_dtype", None) or base.weight.dtype
    if param_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
      param_dtype = torch.bfloat16
    device = base.weight.device
    self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=param_dtype))
    self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=param_dtype))
    # A kaiming, B zero -> the adapter starts as a no-op (delta = 0).
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.base(x)
    delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
    return out + self.scale * delta.to(out.dtype)


def inject_lora(
  transformer: nn.Module,
  *,
  rank: int = 16,
  alpha: float | None = None,
  targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> dict[str, LoRALinear]:
  """Replace target Linears in ``transformer.layers`` with ``LoRALinear``.

  ``alpha`` defaults to ``rank`` (scale 1.0), matching the inference loader.
  Returns a dict ``{"layers.{i}.{sub}": module}`` for ``save_lora`` and for
  collecting trainable parameters.

  The whole transformer is frozen first, so after injection only the freshly
  created adapter parameters require grad.
  """
  alpha = rank if alpha is None else alpha
  transformer.requires_grad_(False)
  wrapped: dict[str, LoRALinear] = {}
  for i, layer in enumerate(transformer.layers):
    for sub in targets:
      parent = layer
      *parents, attr = sub.split(".")
      for p in parents:
        parent = getattr(parent, p)
      base = getattr(parent, attr)
      # Accept nn.Linear and the quantized Linears (Fp8Linear, bnb Linear4bit):
      # all expose in_features/out_features/weight, and their forward is
      # differentiable w.r.t. the input, so QLoRA-style training works on the
      # frozen quantized base.
      if not all(hasattr(base, a) for a in ("in_features", "out_features", "weight")):
        raise TypeError(f"layers.{i}.{sub} is {type(base).__name__}; not a Linear-like module")
      lora = LoRALinear(base, rank=rank, alpha=alpha)
      setattr(parent, attr, lora)
      wrapped[f"layers.{i}.{sub}"] = lora
  return wrapped


@contextlib.contextmanager
def lora_disabled(module: nn.Module):
  """Temporarily zero every LoRALinear scale in `module` (forward == frozen base)."""
  mods = [m for m in module.modules() if isinstance(m, LoRALinear)]
  saved = [m.scale for m in mods]
  for m in mods:
    m.scale = 0.0
  try:
    yield
  finally:
    for m, s in zip(mods, saved):
      m.scale = s


@contextlib.contextmanager
def lora_scaled(module: nn.Module, factor: float):
  """Temporarily multiply every LoRALinear scale in `module` by ``factor``.

  This is the runtime "slider knob": a trained slider adapter has its geometric
  scale (``alpha/rank``) multiplied by ``factor`` so the low-rank delta is dialed
  up (``factor>1``), down, off (``0``), or *inverted* (``factor<0`` steers the
  opposite way along the trained axis). Used at inference to set slider strength
  and during ``slider_training_step`` to evaluate the +/- branches.
  """
  mods = [m for m in module.modules() if isinstance(m, LoRALinear)]
  saved = [m.scale for m in mods]
  for m in mods:
    m.scale = m.scale * factor
  try:
    yield
  finally:
    for m, s in zip(mods, saved):
      m.scale = s


def lora_parameters(wrapped: dict[str, LoRALinear]) -> list[nn.Parameter]:
  """Trainable adapter parameters, for the optimizer."""
  params: list[nn.Parameter] = []
  for m in wrapped.values():
    params.extend([m.lora_A, m.lora_B])
  return params


def save_lora(
  wrapped: dict[str, LoRALinear], path: str, metadata: dict | None = None
) -> None:
  """Write adapters as ai-toolkit-format safetensors (load_lora_map-compatible).

  ``metadata`` (optional) is written to the safetensors header for provenance /
  resume / ComfyUI (base model, step, rank, trigger words, ...). All values are
  coerced to str as required by the safetensors format.
  """
  from safetensors.torch import save_file

  state: dict[str, torch.Tensor] = {}
  for sub, m in wrapped.items():
    state[f"diffusion_model.{sub}.lora_A.weight"] = m.lora_A.detach().to("cpu", torch.float32)
    state[f"diffusion_model.{sub}.lora_B.weight"] = m.lora_B.detach().to("cpu", torch.float32)
  meta = {str(k): str(v) for k, v in metadata.items()} if metadata else None
  save_file(state, path, metadata=meta)
