"""Typed YAML config for the in-context edit / ByG training scripts.

Keeps all machine-specific paths and run hyperparameters out of code. Scripts
do ``from ideogram4.training_config import load_config, apply_runtime`` and read
everything from the returned :class:`TrainConfig`.

Resolution order (last wins):
  1. dataclass defaults
  2. the YAML at the given path (or ``$IG4_CONFIG`` when path is None)
  3. ``config/local.yaml`` deep-merged if present (next to the loaded yaml, else
     the repo's ``config/local.yaml``) -- holds real, gitignored machine paths
  4. ``IG4_<SECTION>__<KEY>`` env overrides (e.g. ``IG4_PATHS__WEIGHTS``,
     ``IG4_LORA__RANK``, ``IG4_OPTIM__STEPS``), cast to the target field type.

Only stdlib + PyYAML are imported at module load; ``torch`` is imported lazily
inside :func:`dtype_of` so this module can be imported without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass

import yaml


# --------------------------------------------------------------------------- #
# Schema (FROZEN -- byg code imports these dataclasses; do not rename fields)
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
  weights: str = "ideogram-ai/ideogram-4-fp8"  # local dir OR HF repo id
  data_root: str = "data"
  cache_dir: str = ""        # "" -> f"{data_root}/cache"
  output_dir: str = "runs"
  ckpt_dir: str = ""         # "" -> f"{output_dir}/ckpts"
  results_dir: str = ""
  resume_from: str = ""      # path to resume.pt; "auto" = {ckpt_dir}/resume.pt if present


@dataclass
class RuntimeConfig:
  device: str = "cuda"
  dtype: str = "bfloat16"
  hf_offline: bool = False
  extra_sys_path: list = field(default_factory=list)
  seed: int = 0
  tf32: bool = True           # enable TF32 matmul/cudnn (free speedup on Ampere+)


@dataclass
class LoraConfig:
  rank: int = 64
  alpha: float | None = None
  variant: str = "lora"        # lora | dora | loha | lokr
  target_adaln: bool = False   # also adapt each block's adaln_modulation Linear
  te_rank: int = 0             # TE-LoRA rank (train_te_lora_cached.py); 0 -> use rank
  train_transformer: bool = True  # TE trainer: also LoRA the DiT (False = DiT frozen, TE-only)


@dataclass
class DataConfig:
  resolution: int = 512
  img_ext: str = "png"
  instr_field: str = "edit"
  meta_at_root: bool = False
  limit: int = 0
  n_eval_holdout: int = 4
  masked_loss: bool = False        # weight loss to the edited region (derive_edit_mask)
  mask_quantile: float = 0.5       # tokens with top (1-q) |z_tgt-z_ref| are in the mask
  mask_bg_weight: float = 0.0      # soft mask: background weight (0 = hard mask; e.g. 0.2)
  train_list: str = ""             # JSON list of cache filenames (repeats = oversampling)
  aspect_bucketing: bool = False   # precache at nearest-AR bucket instead of square-squash
  bucket_pixels: int = 0           # target area for buckets; 0 -> resolution^2
  num_buckets: int = 9


@dataclass
class OptimConfig:
  lr: float = 1e-4
  steps: int = 3000
  batch: int = 1
  accum: int = 4
  warmup: int = 100
  optimizer: str = "adamw"   # adamw | adamw8bit | prodigy | schedule_free | came
  grad_clip: float = 1.0
  grad_checkpointing: bool = False
  cfg_dropout_prob: float = 0.1
  use_ema: bool = False            # maintain an EMA of the LoRA; the EMA adapter ships
  ema_decay: float = 0.999
  nan_guard: bool = True           # skip the optimizer step on a non-finite loss
  lr_scheduler: str = "cosine"     # cosine | constant | linear | cosine_restarts
  min_lr_ratio: float = 0.0        # LR floor (fraction of base) the decay approaches
  num_restarts: int = 1            # cycles for cosine_restarts
  prior_preservation_weight: float = 0.0  # keep background near the frozen base (anti-forgetting)
  offload_optimizer: bool = False  # full-FT: keep Adam moments in CPU RAM (~21GB VRAM vs ~58GB)
  blocks_to_swap: int = 0          # full-FT: offload N deepest blocks to CPU, swap in/out
                                   # per forward/backward (lower VRAM; use with accum>1)
  anchor_weight: float = 1.0       # uncond LoRA: weight of the no-op-on-clean regularizer
                                   # (disentangles the degradation axis from dataset content)
  weight_decay: float = 0.01       # full-FT AdamW weight decay (train_edit_full_cached.py)
  te_lr: float = 0.0               # joint full-FT: separate (lower) LR for the text encoder;
                                   # 0 -> lr/10 when training the DiT too, else lr (TE-only stage)
  train_dit: bool = True           # joint trainer: also full-FT the DiT. False = DiT stays
                                   # frozen fp8, ONLY the text encoder trains (decoupled stage 1)
  optimizer_state: str = "adamw"   # joint full-FT moment backend: adamw (offload) | adafactor (on-GPU)
  preload_caches: bool = True      # full-FT trainers: preload latent caches into RAM


@dataclass
class FlowConfig:
  # Logit-normal density the training timestep t is sampled from (center/spread before
  # the automatic per-image-area resolution shift). Default 1.0/1.0 reproduces the
  # original hardcoded behaviour AND exactly matches the edit sampler (edit_generate
  # defaults mu=1.0, std=1.0) -- so edit trainers are calibrated to their inference by
  # default. For text-to-image, tune toward the sampling preset you generate with
  # (e.g. schedule_mean: 0.0, schedule_std: 1.5 to match V4_QUALITY_48; the T2I pipeline
  # default is mu=0.5, std=1.0). This is a soft calibration knob, not a correctness
  # requirement: the flow convention itself matches train<->inference at any value.
  schedule_mean: float = 1.0
  schedule_std: float = 1.0
  timestep_shift: float = 1.0          # extra monotonic shift on sampled t (1.0 = none);
                                       # redundant with schedule_mean -- prefer one or the other
  timestep_weighting: str = "uniform"  # uniform | bell | min_snr | sigma_sqrt | cosmap
  min_snr_gamma: float = 5.0
  noise_offset: float = 0.0            # per-channel constant added to the sampled noise
  input_perturbation: float = 0.0      # extra noise on x_t only (target stays clean)


@dataclass
class BygConfig:
  lambda_prior: float = 1.0
  lambda_id: float = 0.2
  alpha_mse: float = 0.1
  p_identity: float = 0.15
  bootstrap_steps: int = 10
  ema_decay: float = 0.999
  t2i_cfg_scale: float = 1.0
  detach_rollout: bool = True


@dataclass
class SliderConfig:
  # Concept-Slider (slider_training_step): a bidirectional attribute knob.
  positive_prompt: str = ""    # c+ : attribute to enhance at +scale (e.g. "highly detailed")
  negative_prompt: str = ""    # c- : attribute at -scale (e.g. "blurry, low detail")
  anchor_prompt: str = ""      # neutral anchor c_t; "" -> unconditional (zeros)
  eta: float = 2.0             # direction strength defining the train targets
  train_scale: float = 1.0     # +/- adapter scale the slider is trained at
  bidirectional: bool = True   # True: ± knob; False: enhance-only (+scale branch only)
  context: str = "t2i"         # sequence layout to train in; MUST match inference: t2i | edit
  infer_scale: float = 2.0     # default slider strength at inference (lora_scaled factor)
  late_step_frac: float = 0.0  # restrict slider effect to the final fraction of sampler steps


@dataclass
class VlmConfig:
  backend: str = "transformers"   # transformers | openai_compatible
  model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
  api_base: str = ""
  max_new_tokens: int = 512
  edit_taxonomy_hint: str = ""


@dataclass
class LoggingConfig:
  log_every: int = 25
  ckpt_every: int = 1000
  val_every: int = 0          # 0 = off; else held-out validation loss every N steps
  sample_every: int = 0       # 0 = off; else decode in-training samples every N steps
  sample_steps: int = 20      # sampler steps for in-training samples
  sample_guidance: float = 2.0
  sample_count: int = 4       # how many prompts/items to sample each time
  preview_dir: str = ""       # dir of curated preview_*.pt (llm_text only) -> T2I-only dashboard
  tracker: str = "none"       # none | wandb | tensorboard (mirrors metrics.jsonl scalars)
  wandb_project: str = "ideogram4"
  run_name: str = ""          # tracker run name ("" -> backend default)


@dataclass
class TrainConfig:
  paths: PathsConfig = field(default_factory=PathsConfig)
  runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
  lora: LoraConfig = field(default_factory=LoraConfig)
  data: DataConfig = field(default_factory=DataConfig)
  optim: OptimConfig = field(default_factory=OptimConfig)
  flow: FlowConfig = field(default_factory=FlowConfig)
  byg: BygConfig = field(default_factory=BygConfig)
  slider: SliderConfig = field(default_factory=SliderConfig)
  vlm: VlmConfig = field(default_factory=VlmConfig)
  logging: LoggingConfig = field(default_factory=LoggingConfig)


# Section name -> dataclass type. Drives merging and env-override casting.
_SECTIONS = {
  "paths": PathsConfig,
  "runtime": RuntimeConfig,
  "lora": LoraConfig,
  "data": DataConfig,
  "optim": OptimConfig,
  "flow": FlowConfig,
  "byg": BygConfig,
  "slider": SliderConfig,
  "vlm": VlmConfig,
  "logging": LoggingConfig,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
  """Recursively merge ``override`` into ``base``, returning a new dict."""
  out = dict(base)
  for key, val in override.items():
    if isinstance(val, dict) and isinstance(out.get(key), dict):
      out[key] = _deep_merge(out[key], val)
    else:
      out[key] = val
  return out


def _load_yaml(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if data is None:
    return {}
  if not isinstance(data, dict):
    raise ValueError(f"config {path} must be a mapping at the top level")
  return data


def _check_unknown_keys(data: dict, path: str) -> None:
  """Raise a clear error for typo'd / unknown section or field names."""
  for section, values in data.items():
    if section not in _SECTIONS:
      raise ValueError(
        f"unknown config section {section!r} in {path}; "
        f"valid sections: {sorted(_SECTIONS)}"
      )
    if values is None:
      continue
    if not isinstance(values, dict):
      raise ValueError(
        f"section {section!r} in {path} must be a mapping, got {type(values).__name__}"
      )
    valid = {f.name for f in fields(_SECTIONS[section])}
    for key in values:
      if key not in valid:
        raise ValueError(
          f"unknown key {section}.{key!r} in {path}; "
          f"valid keys: {sorted(valid)}"
        )


def _cast_to(field_type, raw: str):
  """Cast a string env value to a dataclass field's annotated type."""
  type_str = str(field_type)
  if field_type is bool or "bool" in type_str:
    return raw.strip().lower() in ("1", "true", "yes")
  if field_type is int or "int" in type_str:
    return int(raw)
  if field_type is float or "float" in type_str:
    return float(raw)
  return raw


def _apply_env_overrides(data: dict) -> dict:
  """Apply ``IG4_<SECTION>__<KEY>=value`` env vars onto the merged dict."""
  out = dict(data)
  for env_name, raw in os.environ.items():
    if not env_name.startswith("IG4_") or "__" not in env_name:
      continue
    body = env_name[len("IG4_"):]
    section_part, key_part = body.split("__", 1)
    section = section_part.lower()
    key = key_part.lower()
    if section not in _SECTIONS:
      continue
    field_map = {f.name: f.type for f in fields(_SECTIONS[section])}
    if key not in field_map:
      continue
    out.setdefault(section, {})
    out[section][key] = _cast_to(field_map[key], raw)
  return out


def _build_section(cls, values: dict):
  """Instantiate a section dataclass from its (already validated) dict."""
  return cls(**(values or {}))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_config(path: str | None = None) -> TrainConfig:
  """Build a :class:`TrainConfig` from defaults, YAML, local.yaml, and env.

  ``path`` (or ``$IG4_CONFIG`` when None) names the preset YAML. A non-None
  path that does not exist raises ``FileNotFoundError``. Unknown YAML keys
  raise ``ValueError`` (typo protection).
  """
  if path is None:
    path = os.environ.get("IG4_CONFIG")

  merged: dict = {}
  yaml_dir = None

  if path is not None:
    if not os.path.exists(path):
      raise FileNotFoundError(f"config file not found: {path}")
    preset = _load_yaml(path)
    _check_unknown_keys(preset, path)
    merged = _deep_merge(merged, preset)
    yaml_dir = os.path.dirname(os.path.abspath(path))

  # Deep-merge config/local.yaml if present: next to the loaded yaml, else
  # the repo-level config/local.yaml.
  local_candidates = []
  if yaml_dir is not None:
    local_candidates.append(os.path.join(yaml_dir, "local.yaml"))
  repo_local = os.path.join(os.getcwd(), "config", "local.yaml")
  if repo_local not in local_candidates:
    local_candidates.append(repo_local)
  for local_path in local_candidates:
    if os.path.exists(local_path):
      local = _load_yaml(local_path)
      _check_unknown_keys(local, local_path)
      merged = _deep_merge(merged, local)
      break

  # Env overrides last.
  merged = _apply_env_overrides(merged)

  cfg = TrainConfig(
    **{name: _build_section(cls, merged.get(name)) for name, cls in _SECTIONS.items()}
  )

  # Resolve empty-string path defaults after all merging.
  if not cfg.paths.cache_dir:
    cfg.paths.cache_dir = f"{cfg.paths.data_root}/cache"
  if not cfg.paths.ckpt_dir:
    cfg.paths.ckpt_dir = f"{cfg.paths.output_dir}/ckpts"

  return cfg


def patch_local_weights() -> None:
  """Make the Ideogram4 pipeline read weights from a local directory.

  Monkeypatches ``ideogram4.pipeline_ideogram4.hf_hub_download`` so that when a
  weights ``repo_id`` is an existing local directory, files are served from disk
  instead of the Hugging Face hub. Also sets offline env defaults.
  """
  import ideogram4.pipeline_ideogram4 as pip_mod
  from huggingface_hub.errors import EntryNotFoundError

  orig = pip_mod.hf_hub_download

  def local_aware(*args, **kw):
    repo_id = kw.get("repo_id") if "repo_id" in kw else (args[0] if args else None)
    filename = kw.get("filename")
    if repo_id and isinstance(repo_id, str) and os.path.isdir(repo_id) and filename:
      local = os.path.join(repo_id, filename)
      if os.path.exists(local):
        return local
      raise EntryNotFoundError(local)
    return orig(*args, **kw)

  pip_mod.hf_hub_download = local_aware
  os.environ.setdefault("HF_HUB_OFFLINE", "1")
  os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def set_tf32(enabled: bool) -> None:
  """Toggle TF32 for matmul + cuDNN (no-op cost if torch is absent)."""
  import torch

  torch.backends.cuda.matmul.allow_tf32 = enabled
  torch.backends.cudnn.allow_tf32 = enabled
  torch.set_float32_matmul_precision("high" if enabled else "highest")


def apply_runtime(cfg: TrainConfig) -> None:
  """Apply runtime side effects: sys.path, offline env, TF32, local-weights patch."""
  import sys

  for entry in cfg.runtime.extra_sys_path:
    if entry and entry not in sys.path:
      sys.path.insert(0, entry)

  if cfg.runtime.hf_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

  if cfg.runtime.tf32:
    set_tf32(True)

  if os.path.isdir(cfg.paths.weights):
    patch_local_weights()


def dtype_of(cfg: TrainConfig):
  """Resolve ``runtime.dtype`` to a ``torch.dtype`` (torch imported lazily)."""
  import torch

  mapping = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
  }
  name = (cfg.runtime.dtype or "bfloat16").lower()
  if name not in mapping:
    raise ValueError(f"unsupported runtime.dtype {cfg.runtime.dtype!r}")
  return mapping[name]
