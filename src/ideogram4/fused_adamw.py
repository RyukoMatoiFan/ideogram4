"""Memory-efficient full fine-tuning: fused-back-pass AdamW with bf16 stochastic rounding.

Two standard techniques that together make 9.3B full fine-tuning fit on a single 80GB GPU:

  * **Stochastic rounding** (Zamirai et al. 2020, "Revisiting BFloat16 Training",
    arXiv:2010.06192): update bf16 weights/states directly with no fp32 master copy. A
    naive bf16 += tiny update rounds to zero; stochastic rounding keeps the update in
    expectation, so we drop the fp32 master (halves weight memory) without the usual
    bf16 staleness. ``copy_stochastic_`` is the standard mantissa-dither implementation.

  * **Fused back pass** (driven by the trainer via register_post_accumulate_grad_hook):
    ``step_parameter`` runs the AdamW update for ONE parameter as soon as its grad is
    ready, then the trainer frees that grad. Full-model gradients never coexist -> the
    18.6GB gradient buffer for a 9.3B model collapses to ~one layer's worth. The per-
    parameter step is just the PyTorch AdamW algorithm applied to a single tensor.

Memory for ig4 (9.3B): weights bf16 18.6GB + states bf16 37GB + grads ~0 + acts ~few GB
-> ~58GB, fits 80GB. ``patch_adamw`` monkeypatches a stock ``torch.optim.AdamW`` to gain
``.step_parameter(p, group, i)``; the trainer drives the update via per-parameter grad
hooks (the optimizer's normal ``.step()`` is unused on this path).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import AdamW


# --------------------------------------------------------------------------- #
# bf16 stochastic rounding
# --------------------------------------------------------------------------- #
_sr_generator = None


def _sr_seed(seed: int, device: torch.device) -> None:
  global _sr_generator
  if _sr_generator is None or _sr_generator.device != device:
    _sr_generator = torch.Generator(device=device)
  _sr_generator.manual_seed(seed)


def copy_stochastic_(target: Tensor, source: Tensor) -> None:
  """Copy fp32 ``source`` into bf16 ``target`` with stochastic rounding of the mantissa."""
  global _sr_generator
  if _sr_generator is None or _sr_generator.device != source.device:
    _sr_generator = torch.Generator(device=source.device)
  result = torch.randint(
    size=source.shape, device=source.device, dtype=torch.int32,
    low=0, high=(1 << 16), generator=_sr_generator,
  )
  result.add_(source.view(dtype=torch.int32))
  result.bitwise_and_(-65536)  # zero the low 16 mantissa bits (FFFF0000)
  target.copy_(result.view(dtype=torch.float32))
  del result


def addcdiv_stochastic_(inp: Tensor, t1: Tensor, t2: Tensor, value: float = 1.0,
                        *, premul: float | None = None) -> None:
  """``inp = premul*inp + value * t1 / t2`` with stochastic rounding when ``inp`` is bf16.

  ``premul`` (the decoupled weight-decay factor ``1 - lr*wd``) is applied to the fp32 copy
  BEFORE the update, so decay + update share a SINGLE round-to-bf16 event. Applying decay as
  a separate ``inp.mul_`` on a bf16 weight would round straight back (decay << bf16 ULP).
  """
  result = inp.clone() if inp.dtype == torch.float32 else inp.to(dtype=torch.float32)
  if premul is not None:
    result.mul_(premul)
  result.addcdiv_(t1, t2, value=value)
  copy_stochastic_(inp, result)


def add_stochastic_(inp: Tensor, other: Tensor, alpha: float = 1.0,
                    *, premul: float | None = None) -> None:
  """``inp = premul*inp + alpha * other`` with stochastic rounding when ``inp`` is bf16.

  Used by the CPU-offload step: the fp32 update ``delta`` is computed on CPU, moved to
  the GPU, then folded into the bf16 weight here without a fp32 master copy. ``premul`` is
  the decoupled weight-decay factor, folded in before the single rounding event (see
  ``addcdiv_stochastic_``).
  """
  result = inp.to(dtype=torch.float32)
  if premul is not None:
    result.mul_(premul)
  result.add_(other, alpha=alpha)
  copy_stochastic_(inp, result)


# --------------------------------------------------------------------------- #
# Per-parameter AdamW step (fused back pass entry point)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def step_adamw_parameter(self, p: Tensor, group: dict, i: int) -> None:
  """AdamW update for a SINGLE parameter ``p``. Called from the grad hook.

  With ``self.offload_states`` the moment buffers live in CPU RAM (fp32): the grad is
  pulled to CPU, the Adam math runs on CPU, and only the small per-parameter update is
  moved back to apply to the GPU weight -- so the 37GB of states never touch VRAM.
  """
  if p.grad is None:
    return
  if p.grad.is_sparse:
    raise RuntimeError("AdamW does not support sparse gradients")
  offload = getattr(self, "offload_states", False)
  state = self.state[p]

  if len(state) == 0:
    state["step"] = torch.tensor(0.0, dtype=torch.float32)
    if offload:  # fp32 moments pinned in CPU RAM (2 x params x 4 bytes)
      state["exp_avg"] = torch.zeros(p.shape, dtype=torch.float32, device="cpu",
                                     pin_memory=p.is_cuda)
      state["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.float32, device="cpu",
                                        pin_memory=p.is_cuda)
    else:
      state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
      state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

  exp_avg = state["exp_avg"]
  exp_avg_sq = state["exp_avg_sq"]
  beta1, beta2 = group["betas"]

  grad = p.grad
  if group["maximize"]:
    grad = -grad

  # int(...) is load-bearing: AdamW.__setstate__ re-tensorizes 'step' on resume, so a
  # bare state['step'] + 1 would yield a tensor step_size that breaks addcdiv_. Forcing
  # a python int keeps the math bit-identical for fresh AND resumed runs.
  step = int(state["step"]) + 1
  state["step"] = step
  bias_correction1 = 1 - beta1 ** step
  bias_correction2 = 1 - beta2 ** step
  step_size = group["lr"] / bias_correction1
  bias_correction2_sqrt = math.sqrt(bias_correction2)

  # Decoupled weight decay: factor folded into the SAME fp32 expression as the update (via
  # the SR helpers' premul) so decay + update share ONE round-to-bf16. A standalone
  # p.mul_(1-lr*wd) on a bf16 weight rounds straight back (decay << bf16 ULP) -> silent
  # no-op, exactly the bf16 staleness this module's stochastic rounding exists to prevent.
  wd = group["weight_decay"]
  decay = (1.0 - group["lr"] * wd) if wd else None
  sr = p.dtype == torch.bfloat16 and getattr(self, "stochastic_rounding", False)

  if offload:
    g = grad.to("cpu", torch.float32)              # one small grad transfer down
    exp_avg.lerp_(g, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
    delta = exp_avg.div(denom).mul_(-step_size)    # fp32 update on CPU
    delta_gpu = delta.to(p.device, non_blocking=True)
    if sr:
      add_stochastic_(p, delta_gpu, premul=decay)
    else:
      if decay is not None:
        p.mul_(decay)
      p.add_(delta_gpu)
  else:
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
    if sr:
      addcdiv_stochastic_(p, exp_avg, denom, value=-step_size, premul=decay)
    else:
      if decay is not None:
        p.mul_(decay)
      p.addcdiv_(exp_avg, denom, value=-step_size)


# --------------------------------------------------------------------------- #
# Per-parameter Adafactor step (fused back pass entry point) -- memory-efficient
# factored second moment kept ON GPU (no offload), so single-GPU full-FT of large
# models is fast. No first moment (classic Adafactor) -> tiny state (row+col vectors
# per 2D weight instead of the full matrix). Stochastic rounding for bf16 weights.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def step_adafactor_parameter(self, p: Tensor, group: dict, i: int) -> None:
  """Adafactor update for a SINGLE parameter (fused-backward hook). On-GPU state."""
  if p.grad is None:
    return
  if p.grad.is_sparse:
    raise RuntimeError("Adafactor does not support sparse gradients")
  g = p.grad.to(torch.float32)
  if group.get("maximize"):
    g = -g
  state = self.state[p]
  factored = g.dim() >= 2
  if len(state) == 0:
    state["step"] = 0
    if factored:
      state["exp_avg_sq_row"] = torch.zeros(g.shape[:-1], device=p.device, dtype=torch.float32)
      state["exp_avg_sq_col"] = torch.zeros(g.shape[:-2] + g.shape[-1:], device=p.device, dtype=torch.float32)
    else:
      state["exp_avg_sq"] = torch.zeros_like(g)

  state["step"] += 1
  step = state["step"]
  eps1, eps2, clip_threshold, decay_rate = 1e-30, 1e-3, 1.0, 0.8
  beta2t = 1.0 - step ** (-decay_rate)
  upd_sq = g * g + eps1

  if factored:
    row, col = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
    row.mul_(beta2t).add_(upd_sq.mean(dim=-1), alpha=1.0 - beta2t)
    col.mul_(beta2t).add_(upd_sq.mean(dim=-2), alpha=1.0 - beta2t)
    r_factor = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = col.unsqueeze(-2).rsqrt()
    upd = g * r_factor * c_factor
  else:
    eas = state["exp_avg_sq"]
    eas.mul_(beta2t).add_(upd_sq, alpha=1.0 - beta2t)
    upd = g * eas.rsqrt()

  # RMS clip (keeps the update scale bounded) + relative (parameter-scaled) LR.
  rms = float(upd.pow(2).mean().sqrt())
  upd = upd / max(1.0, rms / clip_threshold)
  p_rms = float(p.detach().to(torch.float32).pow(2).mean().sqrt())
  lr_t = group["lr"] * max(eps2, p_rms)
  delta = upd.mul_(-lr_t)

  if p.dtype == torch.bfloat16 and getattr(self, "stochastic_rounding", False):
    add_stochastic_(p, delta)
  else:
    p.add_(delta.to(p.dtype))


def build_fused_adafactor(params, lr: float, *, stochastic_rounding: bool = True) -> AdamW:
  """AdamW shell (only as a param-group/state holder) patched to take per-parameter
  Adafactor steps. State lives ON GPU (tiny, factored) -> no CPU offload, so single-GPU
  full-FT of very large models stays fast. Drive it via the per-parameter grad hooks
  exactly like build_fused_adamw (its .step() is unused)."""
  opt = AdamW(params, lr=lr, foreach=False, fused=False)
  opt.stochastic_rounding = stochastic_rounding
  opt.offload_states = False
  opt.step_parameter = step_adafactor_parameter.__get__(opt, AdamW)
  return opt


def patch_adamw(optimizer: AdamW, stochastic_rounding: bool = True,
                offload_states: bool = False) -> AdamW:
  """Add ``step_parameter`` to a stock ``torch.optim.AdamW`` for the fused back pass."""
  optimizer.stochastic_rounding = stochastic_rounding
  optimizer.offload_states = offload_states
  optimizer.step_parameter = step_adamw_parameter.__get__(optimizer, AdamW)
  return optimizer


def build_fused_adamw(params, lr: float, *, weight_decay: float = 0.01,
                      betas=(0.9, 0.999), eps: float = 1e-8,
                      stochastic_rounding: bool = True,
                      offload_states: bool = False) -> AdamW:
  """Construct a stock AdamW and patch it for fused back pass + stochastic rounding.

  ``foreach``/``fused`` are disabled because the per-parameter hook path bypasses the
  batched kernels (and ``fused=True`` would try to own the step). ``offload_states``
  keeps the Adam moments in CPU RAM (fits 9.3B full-FT in ~21GB VRAM, ~74GB host RAM).
  """
  opt = AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
              foreach=False, fused=False)
  return patch_adamw(opt, stochastic_rounding=stochastic_rounding, offload_states=offload_states)
