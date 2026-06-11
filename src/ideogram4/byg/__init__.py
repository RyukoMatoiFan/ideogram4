"""Bootstrap Your Generator (ByG) -- unpaired image-editing training subpackage.

ByG trains an instruction-based image editor WITHOUT paired (source, edited) data
by using the frozen text-to-image base as a semantic prior and a cycle-consistency
objective.  Only the LoRA adapters receive gradients; the base transformer weights
are never modified.

Public API
----------
LoraEMA          -- EMA shadow for LoRA adapter weights + context managers.
byg_training_step -- one ByG training step (returns dict of loss tensors).
ema_rollout       -- EMA bootstrap rollout producing (y_t, y0).
build_t2i_sequence_meta -- sequence metadata for the plain T2I layout (no reference).
"""

from ideogram4.byg.ema import LoraEMA
from ideogram4.byg.rollout import ema_rollout
from ideogram4.byg.sequences import build_t2i_sequence_meta
from ideogram4.byg.step import byg_training_step

__all__ = [
  "LoraEMA",
  "byg_training_step",
  "ema_rollout",
  "build_t2i_sequence_meta",
]
