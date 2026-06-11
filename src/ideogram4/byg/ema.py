"""Exponential Moving Average (EMA) shadow weights for LoRA adapters.

The EMA object maintains fp32 shadow copies of the live lora_A / lora_B
parameters.  It provides two mutually exclusive context managers:

  ``swap_in()``     -- temporarily replace the live adapter params with the EMA
                       shadow so the transformer behaves as G_EMA.
  ``adapters_off()`` -- temporarily set every LoRALinear.scale to 0.0 so the
                        transformer behaves as the frozen T2I base (G_t2i).

These must NOT be nested with each other.  Document the invariant: on entry to
either context the live weights are stashed; on exit they are restored bit-exactly.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
import torch.nn as nn

from ideogram4.lora import LoRALinear


class LoraEMA:
  """EMA shadow copies of LoRA adapter parameters.

  Parameters
  ----------
  wrapped:
    Dict returned by ``inject_lora`` (``{"layers.i.sub": LoRALinear, ...}``).
  decay:
    EMA decay factor, e.g. 0.999.  Shadow update:
    ``shadow = decay * shadow + (1 - decay) * live``.

  Notes
  -----
  * Never allocates a second base transformer -- only the small adapter tensors
    (lora_A and lora_B) have shadows.
  * ``swap_in`` and ``adapters_off`` are mutually exclusive context managers.
    Nesting them is undefined behavior and will raise ``RuntimeError``.
  """

  def __init__(self, wrapped: dict[str, LoRALinear], decay: float) -> None:
    if not 0.0 < decay < 1.0:
      raise ValueError(f"decay must be in (0, 1), got {decay}")
    self.wrapped = wrapped
    self.decay = decay

    # fp32 shadow tensors on the same device as the live parameters.
    # Dict layout: key -> {"lora_A": Tensor, "lora_B": Tensor}
    self._shadow: dict[str, dict[str, torch.Tensor]] = {}
    for key, mod in wrapped.items():
      self._shadow[key] = {
        "lora_A": mod.lora_A.data.detach().float().clone(),
        "lora_B": mod.lora_B.data.detach().float().clone(),
      }

    # Guard against nested context managers.
    self._active_ctx: str | None = None  # "swap_in" | "adapters_off" | None

  # ---------------------------------------------------------------------- #
  # EMA update
  # ---------------------------------------------------------------------- #
  @torch.no_grad()
  def update(self) -> None:
    """Update shadow weights: shadow = decay * shadow + (1 - decay) * live.

    Call once per training step AFTER the optimizer step.
    """
    d = self.decay
    for key, mod in self.wrapped.items():
      sh = self._shadow[key]
      sh["lora_A"].mul_(d).add_(mod.lora_A.data.float(), alpha=1.0 - d)
      sh["lora_B"].mul_(d).add_(mod.lora_B.data.float(), alpha=1.0 - d)

  # ---------------------------------------------------------------------- #
  # swap_in: EMA weights replace live adapters for the duration of the block
  # ---------------------------------------------------------------------- #
  @contextlib.contextmanager
  def swap_in(self) -> Iterator[None]:
    """Context manager: EMA shadow -> live adapter params.

    Inside this block the transformer behaves as G_EMA.  Live weights are
    stashed and restored bit-exactly on exit.

    Must NOT be nested inside ``adapters_off``.
    """
    if self._active_ctx is not None:
      raise RuntimeError(
        f"Cannot enter swap_in while {self._active_ctx!r} context is active."
      )
    self._active_ctx = "swap_in"

    # Stash live params and load EMA shadow.
    stash: dict[str, dict[str, torch.Tensor]] = {}
    for key, mod in self.wrapped.items():
      stash[key] = {
        "lora_A": mod.lora_A.data.clone(),
        "lora_B": mod.lora_B.data.clone(),
      }
      sh = self._shadow[key]
      mod.lora_A.data.copy_(sh["lora_A"].to(mod.lora_A.dtype))
      mod.lora_B.data.copy_(sh["lora_B"].to(mod.lora_B.dtype))
    try:
      yield
    finally:
      # Restore live params exactly.
      for key, mod in self.wrapped.items():
        mod.lora_A.data.copy_(stash[key]["lora_A"])
        mod.lora_B.data.copy_(stash[key]["lora_B"])
      self._active_ctx = None

  # ---------------------------------------------------------------------- #
  # adapters_off: set scale=0 -> base-only T2I forward
  # ---------------------------------------------------------------------- #
  @contextlib.contextmanager
  def adapters_off(self) -> Iterator[None]:
    """Context manager: set all LoRALinear.scale to 0.

    Inside this block the transformer behaves as the frozen T2I base (G_t2i).
    Prior scales are restored on exit.

    Must NOT be nested inside ``swap_in``.
    """
    if self._active_ctx is not None:
      raise RuntimeError(
        f"Cannot enter adapters_off while {self._active_ctx!r} context is active."
      )
    self._active_ctx = "adapters_off"

    saved_scales: dict[str, float] = {}
    for key, mod in self.wrapped.items():
      saved_scales[key] = mod.scale
      mod.scale = 0.0
    try:
      yield
    finally:
      for key, mod in self.wrapped.items():
        mod.scale = saved_scales[key]
      self._active_ctx = None

  # ---------------------------------------------------------------------- #
  # Checkpoint helpers
  # ---------------------------------------------------------------------- #
  def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
    """Return a JSON-serializable shadow state for checkpointing.

    The returned dict has the same ``key -> {"lora_A": Tensor, "lora_B": Tensor}``
    structure as the internal shadow, with tensors detached and on CPU.
    """
    return {
      key: {
        "lora_A": sh["lora_A"].detach().cpu(),
        "lora_B": sh["lora_B"].detach().cpu(),
      }
      for key, sh in self._shadow.items()
    }

  def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
    """Restore shadow from a previously saved ``state_dict``.

    Tensors are moved to the same device as the corresponding live parameters.
    """
    for key, tensors in state.items():
      if key not in self._shadow:
        raise KeyError(f"Unknown adapter key in EMA state_dict: {key!r}")
      device = self.wrapped[key].lora_A.device
      self._shadow[key]["lora_A"].copy_(tensors["lora_A"].to(device))
      self._shadow[key]["lora_B"].copy_(tensors["lora_B"].to(device))
