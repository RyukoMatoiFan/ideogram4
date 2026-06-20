"""Block swapping: keep the deepest ``num_blocks`` transformer blocks on CPU and move
each to the compute device just-in-time for its forward/backward, cutting resident
weight VRAM for very large models (trades VRAM for PCIe traffic).

Opt-in via ``optim.blocks_to_swap`` on the full-FT trainers. The plumbing is CPU-tested
(swapping to the same device is a no-op, so the forward/backward result is provably
unchanged); the genuine cuda<->cpu path needs a GPU smoke test.

IMPORTANT INTERACTION: with the fused per-parameter backward (accum == 1) a swapped
block's params can be moved back to CPU by the backward hook before/after the fused
optimizer step, which races the per-param AdamW state device. Use block swapping with
gradient accumulation (accum > 1, the standard backward + manual step path) where the
whole backward runs with the block resident, then a single manual step follows. The
trainers warn if blocks_to_swap > 0 is combined with accum == 1.
"""
from __future__ import annotations

import torch


def enable_block_swap(transformer, num_blocks: int, compute_device, *, blocks_attr: str = "layers") -> int:
  """Offload the deepest ``num_blocks`` blocks to CPU and swap them in/out on use.

  Returns the number of blocks actually set up for swapping. Call AFTER the model is
  on ``compute_device`` (this moves the chosen blocks back to CPU).
  """
  blocks = getattr(transformer, blocks_attr)
  n = len(blocks)
  num_blocks = max(0, min(int(num_blocks), n))
  if num_blocks == 0:
    return 0
  compute_device = torch.device(compute_device)
  cpu = torch.device("cpu")
  swap_idx = range(n - num_blocks, n)  # the deepest blocks

  def _fwd_pre(mod, args):
    mod.to(compute_device, non_blocking=True)

  def _fwd_post(mod, args, output):
    mod.to(cpu, non_blocking=True)

  def _bwd_pre(mod, grad_output):
    mod.to(compute_device, non_blocking=True)

  def _bwd_post(mod, grad_input, grad_output):
    mod.to(cpu, non_blocking=True)

  for i in swap_idx:
    blk = blocks[i]
    blk.to(cpu)
    blk.register_forward_pre_hook(_fwd_pre)
    blk.register_forward_hook(_fwd_post)
    blk.register_full_backward_pre_hook(_bwd_pre)
    blk.register_full_backward_hook(_bwd_post)
  return num_blocks
