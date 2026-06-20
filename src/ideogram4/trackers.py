"""Optional experiment tracking (Weights & Biases / TensorBoard).

No-op by default (``backend="none"``) and degrades gracefully if the backend library
is missing -- training never depends on it. Trainers already write metrics.jsonl; this
mirrors the same scalar dict to a tracker when one is configured.

  tracker = Tracker(cfg.logging.tracker, project=cfg.logging.wandb_project,
                    run_name=cfg.logging.run_name or None, out_dir=output_dir, config={...})
  tracker.log({"loss": 0.1, "lr": 1e-4}, step=42)
  tracker.close()
"""
from __future__ import annotations

from typing import Optional


class Tracker:
  def __init__(self, backend: str = "none", *, project: str = "ideogram4",
               run_name: Optional[str] = None, config: Optional[dict] = None,
               out_dir: Optional[str] = None) -> None:
    self.backend = (backend or "none").lower()
    self._wandb = None
    self._tb = None
    if self.backend == "none":
      return
    if self.backend == "wandb":
      try:
        import wandb
        self._wandb = wandb
        wandb.init(project=project, name=run_name, config=config or {})
      except Exception as e:
        print(f"[tracker] wandb unavailable ({e}); tracking disabled", flush=True)
        self.backend = "none"
    elif self.backend == "tensorboard":
      try:
        from torch.utils.tensorboard import SummaryWriter
        self._tb = SummaryWriter(log_dir=out_dir)
      except Exception as e:
        print(f"[tracker] tensorboard unavailable ({e}); tracking disabled", flush=True)
        self.backend = "none"
    else:
      print(f"[tracker] unknown backend {self.backend!r}; tracking disabled", flush=True)
      self.backend = "none"

  def log(self, metrics: dict, step: Optional[int] = None) -> None:
    """Log the numeric entries of ``metrics`` (non-numbers like flags are skipped)."""
    if self.backend == "none":
      return
    scalars = {k: float(v) for k, v in metrics.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not scalars:
      return
    try:
      if self._wandb is not None:
        self._wandb.log(scalars, step=step)
      elif self._tb is not None:
        for k, v in scalars.items():
          self._tb.add_scalar(k, v, step)
    except Exception as e:
      print(f"[tracker] log failed ({e}); disabling", flush=True)
      self.backend = "none"

  def close(self) -> None:
    try:
      if self._tb is not None:
        self._tb.close()
      if self._wandb is not None:
        self._wandb.finish()
    except Exception:
      pass
