# Training configs

All machine-specific paths and run hyperparameters for the edit / ByG training
scripts live here as YAML, sourced by the typed loader in
`src/ideogram4/training_config.py`. No private paths live in code.

## Layering

Values resolve in this order (last wins):

1. **dataclass defaults** — defined in `training_config.py`.
2. **preset YAML** — the file passed via `--config` (e.g. `config/edit_lora.yaml`),
   or `$IG4_CONFIG` when no path is given. Committed presets contain only
   placeholder / relative paths.
3. **`config/local.yaml`** — deep-merged if present (looked up next to the loaded
   preset, else this `config/` dir). Holds the **real machine paths** for your
   box. It is **gitignored** and never committed.
4. **`IG4_*` environment overrides** — highest priority, for one-off tweaks.

## Machine paths: local.yaml workflow

```bash
cp config/local.example.yaml config/local.yaml
# edit config/local.yaml: set paths.weights, paths.data_root, paths.output_dir
```

Because `local.yaml` is deep-merged over every preset, you set your real
`weights` / `data_root` / `output_dir` once and every run picks them up.

## IG4_* environment overrides

Override any field with `IG4_<SECTION>__<KEY>` (double underscore between
section and key). Values are cast to the field's type; booleans accept
`1`/`true`/`yes`.

```bash
# bump LoRA rank for a single run
IG4_LORA__RANK=256 python train_edit_lora_cached.py --config config/edit_lora_cached.yaml

# shorten a run and point at different weights
IG4_OPTIM__STEPS=500 IG4_PATHS__WEIGHTS=/data/ideogram-4-fp8 \
  python train_edit_lora.py --config config/edit_lora.yaml

# precache a different layout
IG4_DATA__IMG_EXT=jpg IG4_DATA__META_AT_ROOT=1 \
  python precache_edit.py --config config/precache_edit.yaml
```

## Presets

These are **generic** presets. They carry placeholder relative paths; set your real
`weights` / `data_root` / `output_dir` in `config/local.yaml` (or via `IG4_*`). Configs
specific to a particular dataset or experiment do not live here.

| Preset | Script | Purpose |
| --- | --- | --- |
| `edit_lora.yaml` | `train_edit_lora.py` | instruction-edit LoRA, full pipeline (no precache) |
| `edit_lora_cached.yaml` | `train_edit_lora_cached.py` | instruction-edit LoRA, encoder-free (cached) |
| `edit_full.yaml` | `train_edit_full_cached.py` | instruction-edit FULL fine-tune (fp8→bf16, single 80GB GPU) |
| `precache_edit.yaml` | `precache_edit.py` | precompute z_ref/z_tgt/llm caches for editing |
| `multiref.yaml` | `precache_multiref.py` → `train_multiref.py` | reference-driven / multi-reference editing |
| `slider.yaml` | `train_slider.py` | Concept-Slider (detailer / attribute knob) |
| `uncond_quality.yaml` | `precache_uncond.py` → `train_uncond_lora.py` | negative-model (uncond) quality LoRA for the stock pipeline |

### Sliders

A slider is a **direction**, not a task. Trained/prompt-defined sliders live on OUR
cfg-dropout-trained samplers (`sample_edit_cached` / `sample_t2i` on the conditional
transformer): the stock pipeline's negative branch is a separate unconditional
network, so the slider's zero-text anchor never occurs there, and shipping the
adapter to both transformers largely cancels under CFG. For a stock-pipeline knob use
the **negative-model** flavor instead.

| Flavor | How to train | When |
| --- | --- | --- |
| **Bidirectional** (preferred) | `train_slider.py` (`slider.bidirectional: true`) | a true ± knob, decoupled from the prompt |
| **Unidirectional** | `train_slider.py` (`slider.bidirectional: false`) | one-way enhance-only; cheaper |
| **Data-paired** (detailer) | `train_edit_lora_cached.py` with low→high pairs | when you have aligned weak/strong examples |
| **Prompt-defined** (no training) | `edit_sampler.sample_edit_sliders` | instant; pass `slider_branches=[(llm,weight)]` |
| **Negative-model** (uncond) | `train_uncond_lora.py` on degraded images | global quality knob on the STOCK pipeline |

- **Trained adapter**: after training, set strength with `lora.lora_scaled(transformer,
  factor)` — `factor>0` enhances, `<0` inverts, `0` off — then sample with the matching
  cached sampler (`slider.context` MUST equal the inference layout: `edit` or `t2i`).
  The `slider.*` keys define `positive_prompt`/`negative_prompt` (axis), `eta` (target
  strength), `bidirectional`, and `infer_scale`/`late_step_frac` defaults. Use
  `--rollout N` to self-generate context (zero external data); real context images are
  required — an attribute direction is only observable on structured latents.
- **Prompt-defined**: `sample_edit_sliders` steers `v += weight·(v_pos − v_branch)` away
  from each branch concept; `late_step_frac` confines it to the low-noise (detail) tail.
- **Negative-model**: the only flavor for the stock dual-transformer pipeline. Hook the
  adapter into the uncond ONLY; CFG's `(1-g)<0` coefficient steers away from the trained
  manifold, amplified by `(g-1)`. Verify with `eval_uncond_lora.py` (sign inversion is
  the causal check).
