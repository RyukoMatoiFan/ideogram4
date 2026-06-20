"""Trainable adapters for the Ideogram 4 transformer (LoRA + DoRA).

Wraps the attention/MLP (and optionally adaLN) Linears of ``Ideogram4Transformer``
with low-rank adapters and freezes the base weights.

Plain LoRA matches the inference-time hook in
``ideogram_region_lora/server/lora_apply.py``::

    y = base(x) + (alpha / rank) * (x @ A^T) @ B^T

``save_lora`` writes ai-toolkit-compatible keys
``diffusion_model.layers.{i}.{sub}.lora_{A,B}.weight`` (+ ``.dora_scale`` for DoRA).
With the default ``alpha == rank`` the geometric scale is 1.0.

DoRA (weight-decomposed LoRA, Liu et al. 2024) additionally learns a per-output-channel
magnitude and renormalizes the adapted weight direction; it is a strict generalization
(no-op at init) that trains within this repo's injected-module stack (train + the repo's
own eval/sampler). An external *plain-LoRA* loader will not apply the DoRA magnitude --
ship DoRA only where DoRA-aware loading exists.

Train on a bf16 base, or QLoRA-style on a frozen fp8/nf4 base (adapters live in the
compute dtype; the quantized base dequantizes in its own forward).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sub-paths (relative to each transformer block) of the Linears we adapt.
DEFAULT_TARGETS = (
  "attention.qkv",
  "attention.o",
  "feed_forward.w1",
  "feed_forward.w2",
  "feed_forward.w3",
)
# The per-block adaLN modulation Linear (opt-in: changes the conditioning projection).
ADALN_TARGET = "adaln_modulation"

VARIANTS = ("lora", "dora", "loha", "lokr")


def _adapter_dtype(base: nn.Module) -> torch.dtype:
  """Compute dtype the adapter params live in (bf16 for a float8 base)."""
  dt = getattr(base, "compute_dtype", None) or base.weight.dtype
  if dt in (torch.float8_e4m3fn, torch.float8_e5m2):
    return torch.bfloat16
  return dt


def _dequant_base_weight(base: nn.Module) -> torch.Tensor:
  """The base weight as a real (out, in) float tensor, dequantizing if needed."""
  if hasattr(base, "weight_scale"):  # Fp8Linear: w = w_fp8 * scale
    return base.weight.to(torch.float32) * base.weight_scale.to(torch.float32).unsqueeze(1)
  w = base.weight
  if w.__class__.__name__ == "Params4bit":  # bnb nf4/int4
    import bitsandbytes.functional as bnbF
    return bnbF.dequantize_4bit(w.data, w.quant_state).to(torch.float32)
  return w.to(torch.float32)


class _Adapter(nn.Module):
  """Shared base: frozen wrapped Linear + a geometric ``scale`` (alpha/rank)."""

  PARAM_NAMES: tuple[str, ...] = ()

  def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
    super().__init__()
    self.base = base
    self.base.requires_grad_(False)  # no-op for fp8/nf4 (weights are buffers/frozen)
    self.rank = rank
    self.scale = alpha / rank

  def adapter_parameters(self) -> list[nn.Parameter]:
    return [getattr(self, n) for n in self.PARAM_NAMES]

  def adapter_state(self) -> dict[str, torch.Tensor]:
    """Detached CPU copy of every trainable tensor (for resume checkpoints)."""
    return {n: getattr(self, n).detach().cpu() for n in self.PARAM_NAMES}

  def load_adapter_state(self, d: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
      for n in self.PARAM_NAMES:
        p = getattr(self, n)
        p.copy_(d[n].to(p.device, p.dtype))

  # subclasses implement these
  def save_tensors(self, sub: str) -> dict[str, torch.Tensor]:
    raise NotImplementedError

  @property
  def variant(self) -> str:
    raise NotImplementedError


class LoRALinear(_Adapter):
  """``nn.Linear`` wrapped with a frozen base and a trainable low-rank delta."""

  PARAM_NAMES = ("lora_A", "lora_B")

  def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
    super().__init__(base, rank, alpha)
    dt = _adapter_dtype(base)
    dev = base.weight.device
    self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
    self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B zero -> delta = 0 at init

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.base(x)
    delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
    return out + self.scale * delta.to(out.dtype)

  def save_tensors(self, sub):
    return {
      f"diffusion_model.{sub}.lora_A.weight": self.lora_A.detach().to("cpu", torch.float32),
      f"diffusion_model.{sub}.lora_B.weight": self.lora_B.detach().to("cpu", torch.float32),
    }

  @property
  def variant(self):
    return "lora"


class DoRALinear(_Adapter):
  """Weight-decomposed LoRA: learns direction (B@A) AND a per-channel magnitude.

  ``W_eff = m * (W + s*B@A) / ||W + s*B@A||_col``. At init ``B=0`` and ``m=||W||_col``,
  so ``W_eff == W`` (exact no-op). The base weight is dequantized each forward when the
  base is quantized -- correct but heavier than plain LoRA on an fp8/nf4 base.
  """

  PARAM_NAMES = ("lora_A", "lora_B", "dora_m")

  def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
    super().__init__(base, rank, alpha)
    dt = _adapter_dtype(base)
    dev = base.weight.device
    self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
    self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    with torch.no_grad():
      col_norm = _dequant_base_weight(base).norm(dim=1)  # (out,)
    self.dora_m = nn.Parameter(col_norm.to(device=dev, dtype=dt))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    delta = (self.lora_B @ self.lora_A) * self.scale            # (out, in)
    w = _dequant_base_weight(self.base).to(delta.dtype)
    v = w + delta
    norm = v.norm(dim=1, keepdim=True).clamp_min(1e-6)          # (out, 1)
    w_eff = (self.dora_m.unsqueeze(1) / norm) * v
    bias = getattr(self.base, "bias", None)
    out = F.linear(x.to(w_eff.dtype), w_eff, bias.to(w_eff.dtype) if bias is not None else None)
    return out.to(x.dtype)

  def save_tensors(self, sub):
    return {
      f"diffusion_model.{sub}.lora_A.weight": self.lora_A.detach().to("cpu", torch.float32),
      f"diffusion_model.{sub}.lora_B.weight": self.lora_B.detach().to("cpu", torch.float32),
      f"diffusion_model.{sub}.dora_scale.weight": self.dora_m.detach().to("cpu", torch.float32),
    }

  @property
  def variant(self):
    return "dora"


class LoHaLinear(_Adapter):
  """LyCORIS LoHa: delta = (B1 @ A1) (hadamard) (B2 @ A2), scaled. Higher effective
  rank than plain LoRA for the same param budget. No-op at init (B1 = 0).
  Materializes the (out, in) delta each forward (param-efficient, not flop-efficient).
  """

  PARAM_NAMES = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")

  def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
    super().__init__(base, rank, alpha)
    dt = _adapter_dtype(base)
    dev = base.weight.device
    i, o = base.in_features, base.out_features
    self.hada_w1_a = nn.Parameter(torch.empty(rank, i, device=dev, dtype=dt))
    self.hada_w1_b = nn.Parameter(torch.zeros(o, rank, device=dev, dtype=dt))  # -> delta=0
    self.hada_w2_a = nn.Parameter(torch.empty(rank, i, device=dev, dtype=dt))
    self.hada_w2_b = nn.Parameter(torch.empty(o, rank, device=dev, dtype=dt))
    nn.init.kaiming_uniform_(self.hada_w1_a, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.hada_w2_a, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.hada_w2_b, a=math.sqrt(5))

  def _delta(self) -> torch.Tensor:
    return (self.hada_w1_b @ self.hada_w1_a) * (self.hada_w2_b @ self.hada_w2_a)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.base(x)
    d = self._delta()
    return out + self.scale * F.linear(x.to(d.dtype), d).to(out.dtype)

  def save_tensors(self, sub):
    return {f"diffusion_model.{sub}.{n}": getattr(self, n).detach().to("cpu", torch.float32)
            for n in self.PARAM_NAMES}

  @property
  def variant(self):
    return "loha"


def _factor(n: int) -> tuple[int, int]:
  """Balanced factorization (a, b) with a*b == n and a the largest divisor <= sqrt(n)."""
  a = 1
  for i in range(1, int(math.isqrt(n)) + 1):
    if n % i == 0:
      a = i
  return a, n // a


class LoKrLinear(_Adapter):
  """LyCORIS LoKr: delta = lokr_w1 (kron) (B @ A), scaled. Very param-efficient via a
  Kronecker factorization of the (out, in) weight. No-op at init (B = 0).
  Materializes the (out, in) delta each forward.
  """

  PARAM_NAMES = ("lokr_w1", "lokr_w2_a", "lokr_w2_b")

  def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
    super().__init__(base, rank, alpha)
    dt = _adapter_dtype(base)
    dev = base.weight.device
    o1, o2 = _factor(base.out_features)
    i1, i2 = _factor(base.in_features)
    self._shape = (o1, o2, i1, i2)
    r = min(rank, o2, i2)
    self.lokr_w1 = nn.Parameter(torch.empty(o1, i1, device=dev, dtype=dt))
    self.lokr_w2_a = nn.Parameter(torch.empty(r, i2, device=dev, dtype=dt))
    self.lokr_w2_b = nn.Parameter(torch.zeros(o2, r, device=dev, dtype=dt))  # -> delta=0
    nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))

  def _delta(self) -> torch.Tensor:
    w2 = self.lokr_w2_b @ self.lokr_w2_a            # (o2, i2)
    return torch.kron(self.lokr_w1, w2)             # (o1*o2, i1*i2) = (out, in)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.base(x)
    d = self._delta()
    return out + self.scale * F.linear(x.to(d.dtype), d).to(out.dtype)

  def save_tensors(self, sub):
    return {f"diffusion_model.{sub}.{n}": getattr(self, n).detach().to("cpu", torch.float32)
            for n in self.PARAM_NAMES}

  @property
  def variant(self):
    return "lokr"


_ADAPTER_CLASSES = {
  "lora": LoRALinear, "dora": DoRALinear, "loha": LoHaLinear, "lokr": LoKrLinear,
}


def resolve_targets(targets=DEFAULT_TARGETS, *, target_adaln: bool = False) -> tuple[str, ...]:
  """The tuple of block-relative sub-paths to adapt (optionally incl. adaLN)."""
  return tuple(targets) + ((ADALN_TARGET,) if target_adaln else ())


def inject_lora(
  transformer: nn.Module,
  *,
  rank: int = 16,
  alpha: float | None = None,
  targets: tuple[str, ...] = DEFAULT_TARGETS,
  variant: str = "lora",
  target_adaln: bool = False,
) -> dict[str, _Adapter]:
  """Replace target Linears in ``transformer.layers`` with adapter modules.

  ``variant``: ``"lora"`` (default) or ``"dora"``. ``target_adaln`` also adapts each
  block's ``adaln_modulation`` Linear. ``alpha`` defaults to ``rank`` (scale 1.0).
  Returns ``{"layers.{i}.{sub}": module}``. The whole transformer is frozen first.
  """
  variant = variant.lower()
  if variant not in _ADAPTER_CLASSES:
    raise ValueError(f"unknown LoRA variant {variant!r} (one of {sorted(_ADAPTER_CLASSES)})")
  cls = _ADAPTER_CLASSES[variant]
  alpha = rank if alpha is None else alpha
  all_targets = resolve_targets(targets, target_adaln=target_adaln)
  transformer.requires_grad_(False)
  wrapped: dict[str, _Adapter] = {}
  for i, layer in enumerate(transformer.layers):
    for sub in all_targets:
      parent = layer
      *parents, attr = sub.split(".")
      for p in parents:
        parent = getattr(parent, p)
      base = getattr(parent, attr)
      if not all(hasattr(base, a) for a in ("in_features", "out_features", "weight")):
        raise TypeError(f"layers.{i}.{sub} is {type(base).__name__}; not a Linear-like module")
      adapter = cls(base, rank=rank, alpha=alpha)
      setattr(parent, attr, adapter)
      wrapped[f"layers.{i}.{sub}"] = adapter
  return wrapped


# Default Qwen-style attention/MLP projection names (text-encoder LoRA targets).
TE_DEFAULT_TARGETS = (
  "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def inject_lora_by_names(
  module: nn.Module,
  *,
  rank: int = 16,
  alpha: float | None = None,
  variant: str = "lora",
  target_names: tuple[str, ...] = TE_DEFAULT_TARGETS,
  freeze: bool = True,
) -> dict[str, _Adapter]:
  """Wrap every ``nn.Linear`` whose qualified-name leaf is in ``target_names``.

  Layout-agnostic (no ``.layers`` assumption), so it adapts the Qwen3-VL text encoder
  for TE-LoRA. Returns ``{qualified_name: adapter}``. ``freeze`` zeroes grad on the whole
  module first (only the adapters train).
  """
  variant = variant.lower()
  if variant not in _ADAPTER_CLASSES:
    raise ValueError(f"unknown LoRA variant {variant!r} (one of {sorted(_ADAPTER_CLASSES)})")
  cls = _ADAPTER_CLASSES[variant]
  alpha = rank if alpha is None else alpha
  if freeze:
    module.requires_grad_(False)
  targets = set(target_names)
  # Collect first; don't mutate while iterating named_modules().
  to_wrap = [
    (name, child) for name, child in module.named_modules()
    if name.split(".")[-1] in targets
    and all(hasattr(child, a) for a in ("in_features", "out_features", "weight"))
  ]
  wrapped: dict[str, _Adapter] = {}
  for name, base in to_wrap:
    parent = module
    *parents, attr = name.split(".")
    for p in parents:
      parent = getattr(parent, p)
    adapter = cls(base, rank=rank, alpha=alpha)
    setattr(parent, attr, adapter)
    wrapped[name] = adapter
  return wrapped


@contextlib.contextmanager
def lora_disabled(module: nn.Module):
  """Temporarily zero every adapter scale in `module` (forward == frozen base)."""
  mods = [m for m in module.modules() if isinstance(m, _Adapter)]
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
  """Temporarily multiply every adapter scale in `module` by ``factor`` (slider knob)."""
  mods = [m for m in module.modules() if isinstance(m, _Adapter)]
  saved = [m.scale for m in mods]
  for m in mods:
    m.scale = m.scale * factor
  try:
    yield
  finally:
    for m, s in zip(mods, saved):
      m.scale = s


def lora_parameters(wrapped: dict[str, _Adapter]) -> list[nn.Parameter]:
  """Trainable adapter parameters, for the optimizer."""
  params: list[nn.Parameter] = []
  for m in wrapped.values():
    params.extend(m.adapter_parameters())
  return params


def save_lora(
  wrapped: dict[str, _Adapter], path: str, metadata: dict | None = None
) -> None:
  """Write adapters as safetensors. ``variant`` is recorded in the header metadata
  (``lora``/``dora``); DoRA also writes a ``.dora_scale.weight`` per module.
  """
  from safetensors.torch import save_file

  state: dict[str, torch.Tensor] = {}
  variants = set()
  for sub, m in wrapped.items():
    state.update(m.save_tensors(sub))
    variants.add(m.variant)
  meta = {str(k): str(v) for k, v in (metadata or {}).items()}
  meta.setdefault("variant", variants.pop() if len(variants) == 1 else "mixed")
  save_file(state, path, metadata=meta or None)
