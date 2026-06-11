"""Sequence metadata helpers for ByG training.

Provides ``build_t2i_sequence_meta`` -- the plain text-to-image layout
(no reference frame) used for the T2I prior passes in the ByG objective.

The edit-sequence metadata (with reference frame) is produced by
``ideogram4.train_edit.build_edit_sequence_meta`` and is imported directly
rather than duplicated here.
"""

from __future__ import annotations

import torch

from ideogram4.constants import (
  IMAGE_POSITION_OFFSET,
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
)


def build_t2i_sequence_meta(
  num_text: int,
  grid_h: int,
  grid_w: int,
  device: torch.device,
) -> dict[str, torch.Tensor]:
  """Position/segment/indicator tensors for a plain T2I sequence (batch=1).

  Layout: ``[text tokens (num_text)][target image tokens (grid_h * grid_w)]``

  This mirrors ``Ideogram4Pipeline._build_inputs`` for a single sample with no
  padding and without a tokenizer -- only the token counts are required --
  matching the shape assumptions of ``edit_training_step_cached``.

  The x tensor for a T2I forward is assembled by the caller as::

      x = torch.zeros(1, num_text + n, 128)
      x[0, num_text:] = noised_target_tokens  # image slots; text slots stay zero

  The velocity output is read from ``out[0, num_text:]``.

  Parameters
  ----------
  num_text:
    Number of text-conditioning token slots.
  grid_h, grid_w:
    Spatial patch grid dimensions.
  device:
    Target device for all returned tensors.

  Returns
  -------
  dict with keys:
    ``position_ids`` : (1, num_text + n, 3) int64
    ``segment_ids``  : (1, num_text + n, int64) -- 1 for every slot (no padding)
    ``indicator``    : (1, num_text + n, int64)
    ``num_text``     : int (convenience echo)
    ``n``            : int  (grid_h * grid_w)
  """
  n = grid_h * grid_w
  total = num_text + n

  # --- image grid positions (t=0, h, w) + offset to avoid collision with text ---
  h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
  w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
  t_idx = torch.zeros_like(h_idx)
  image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET  # (n, 3)

  # --- position_ids ---
  position_ids = torch.zeros(1, total, 3, dtype=torch.long)
  tp = torch.arange(num_text)
  position_ids[0, :num_text] = torch.stack([tp, tp, tp], dim=1)
  position_ids[0, num_text:] = image_pos

  # --- indicator ---
  indicator = torch.zeros(1, total, dtype=torch.long)
  indicator[0, :num_text] = LLM_TOKEN_INDICATOR
  indicator[0, num_text:] = OUTPUT_IMAGE_INDICATOR

  # --- segment_ids: full sequence is one segment (no padding in cached path) ---
  segment_ids = torch.ones(1, total, dtype=torch.long)

  return {
    "position_ids": position_ids.to(device),
    "segment_ids": segment_ids.to(device),
    "indicator": indicator.to(device),
    "num_text": num_text,
    "n": n,
  }
