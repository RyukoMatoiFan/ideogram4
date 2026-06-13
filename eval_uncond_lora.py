"""Visual verify for a negative-model (uncond) LoRA.

Loads the full pipeline and injects the adapter into the UNCONDITIONAL transformer
ONLY (never the conditional one -- that is the whole trick). CFG forms
``g*v_cond + (1-g)*v_uncond``, so with the adapter at scale s the learned low-quality
direction enters with coefficient ``(1-g)*s``: s>0 steers AWAY from the trained
degradation manifold (cleaner/sharper), s<0 steers TOWARD it (visibly worse). The
montage is ``[-s | off | +s]`` per prompt at the SAME seed: quality should rise
left->right, and the -s panel degrading is the causal proof the knob is real.

  CUDA_VISIBLE_DEVICES=0 python eval_uncond_lora.py \
    --ckpt runs/uncond-quality/ckpts/uncond_rank16_final.safetensors --rank 16 --scale 1.0
"""
import argparse
import json
import os

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4 import lora as loramod
from ideogram4 import PRESETS


def to_json_caption(text: str) -> str:
  """Ideogram-4 JSON schema wrap -- plain text is OOD for pure T2I."""
  caption = {
    "high_level_description": text,
    "compositional_deconstruction": {
      "background": "",
      "elements": [{"type": "obj", "bbox": [120, 250, 960, 760], "desc": text}],
    },
  }
  return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def _load_lora(wrapped, path, device, dtype):
  from safetensors.torch import load_file
  state = load_file(path)
  with torch.no_grad():
    for sub, m in wrapped.items():
      m.lora_A.copy_(state[f"diffusion_model.{sub}.lora_A.weight"].to(device, dtype))
      m.lora_B.copy_(state[f"diffusion_model.{sub}.lora_B.weight"].to(device, dtype))


def main():
  ap = argparse.ArgumentParser(description="Visual verify of an uncond (negative-model) LoRA.")
  ap.add_argument("--config", default="config/uncond_quality.yaml")
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--rank", type=int, default=0,
                  help="0 = read rank from the checkpoint's safetensors metadata")
  ap.add_argument("--scale", type=float, default=1.0)
  ap.add_argument("--res", type=int, default=1024)
  ap.add_argument("--preset", default="V4_DEFAULT_20", choices=list(PRESETS))
  ap.add_argument("--out", default="runs/uncond-quality/verify")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  device = torch.device(cfg.runtime.device); dtype = dtype_of(cfg)

  rank, alpha = args.rank, None
  if not rank:  # rank/alpha travel in the checkpoint metadata (train_uncond_lora._meta)
    from safetensors import safe_open
    with safe_open(args.ckpt, framework="pt") as f:
      md = f.metadata() or {}
    rank = int(md.get("rank", 16))
    alpha = float(md["alpha"]) if "alpha" in md else None
    print(f"[uncond-eval] rank {rank} alpha {alpha} (from checkpoint metadata)", flush=True)

  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=cfg.paths.weights),
    device=cfg.runtime.device, dtype=dtype)

  # Adapter into the UNCONDITIONAL transformer ONLY.
  wu = loramod.inject_lora(pipe.unconditional_transformer, rank=rank, alpha=alpha)
  _load_lora(wu, args.ckpt, device, dtype)
  pipe.unconditional_transformer.eval()
  print(f"[uncond-eval] adapter into the uncond only ({len(wu)} modules)", flush=True)

  prompts = [
    "a portrait photograph of an elderly fisherman, soft window light",
    "a macro photograph of a butterfly on a leaf",
    "a wooden cabin in a snowy forest at dusk",
  ]
  os.makedirs(args.out, exist_ok=True)
  s = args.scale
  preset = PRESETS[args.preset]
  print(f"[uncond-eval] preset {args.preset} ({preset.num_steps} steps) @ {args.res}px, "
        f"scale +/-{s}", flush=True)

  def gen(prompt):
    return pipe([to_json_caption(prompt)], height=args.res, width=args.res,
                num_steps=preset.num_steps, guidance_schedule=preset.guidance_schedule,
                mu=preset.mu, std=preset.std, seed=0, raise_on_caption_issues=False)[0]

  for i, p in enumerate(prompts):
    panels = []
    for factor in (-s, 0.0, s):
      with loramod.lora_scaled(pipe.unconditional_transformer, factor):
        panels.append(gen(p))
    H = args.res
    canvas = Image.new("RGB", (H * 3, H), (15, 15, 15))
    for j, im in enumerate(panels):
      canvas.paste(im.resize((H, H)), (j * H, 0))
    pth = os.path.join(args.out, f"uncond_{i}.jpg")
    canvas.save(pth, quality=92)
    print(f"[uncond-eval] {i}: [-{s} | off | +{s}] -> {pth}  ('{p[:38]}')", flush=True)
  print("[uncond-eval] DONE -- quality should RISE left->right; the -s panel degrading "
        "is the causal check.", flush=True)


if __name__ == "__main__":
  main()
