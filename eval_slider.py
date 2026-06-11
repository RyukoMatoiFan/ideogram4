"""Visual verify for a trained Concept-Slider LoRA.

Loads the full pipeline, injects the slider adapter into BOTH transformers, and for
each prompt generates the same seed at slider strength -s / 0 / +s (via lora.lora_scaled),
saving a montage ``[-s | off | +s]``. If the attribute (e.g. detail) clearly increases
left->right, the slider works.

  CUDA_VISIBLE_DEVICES=7 python eval_slider.py --ckpt runs/slider-detail/ckpts/slider_rank32_final.safetensors \
    --rank 32 --scale 3.0
"""
import argparse
import os

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4 import lora as loramod


def _load_lora(wrapped, path, device, dtype):
  from safetensors.torch import load_file
  state = load_file(path)
  with torch.no_grad():
    for sub, m in wrapped.items():
      m.lora_A.copy_(state[f"diffusion_model.{sub}.lora_A.weight"].to(device, dtype))
      m.lora_B.copy_(state[f"diffusion_model.{sub}.lora_B.weight"].to(device, dtype))


def main():
  ap = argparse.ArgumentParser(description="Visual verify of a slider LoRA.")
  ap.add_argument("--config", default="config/slider.yaml")
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--rank", type=int, default=32)
  ap.add_argument("--scale", type=float, default=3.0)
  ap.add_argument("--res", type=int, default=512)
  ap.add_argument("--steps", type=int, default=24)
  ap.add_argument("--guidance", type=float, default=5.0)
  ap.add_argument("--out", default="runs/slider-detail/verify")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  device = torch.device(cfg.runtime.device); dtype = dtype_of(cfg)

  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=cfg.paths.weights),
    device=cfg.runtime.device, dtype=dtype)

  # Slider adapter into BOTH transformers (dual-transformer rule).
  wc = loramod.inject_lora(pipe.conditional_transformer, rank=args.rank)
  _load_lora(wc, args.ckpt, device, dtype)
  wu = loramod.inject_lora(pipe.unconditional_transformer, rank=args.rank)
  _load_lora(wu, args.ckpt, device, dtype)
  pipe.conditional_transformer.eval(); pipe.unconditional_transformer.eval()
  print(f"[slider-eval] adapter into both transformers ({len(wc)}+{len(wu)} modules)", flush=True)

  prompts = [
    "a portrait photograph of an elderly fisherman, soft window light",
    "a macro photograph of a butterfly on a leaf",
    "a wooden cabin in a snowy forest at dusk",
  ]
  os.makedirs(args.out, exist_ok=True)
  s = args.scale

  def gen(prompt):
    return pipe([prompt], height=args.res, width=args.res, num_steps=args.steps,
                guidance_scale=args.guidance, seed=0, raise_on_caption_issues=False)[0]

  for i, p in enumerate(prompts):
    panels = []
    for factor in (-s, 0.0, s):
      with loramod.lora_scaled(pipe.conditional_transformer, factor), \
           loramod.lora_scaled(pipe.unconditional_transformer, factor):
        panels.append(gen(p))
    H = args.res
    canvas = Image.new("RGB", (H * 3, H), (15, 15, 15))
    for j, im in enumerate(panels):
      canvas.paste(im.resize((H, H)), (j * H, 0))
    pth = os.path.join(args.out, f"slider_{i}.jpg")
    canvas.save(pth, quality=92)
    print(f"[slider-eval] {i}: [-{s} | off | +{s}] -> {pth}  ('{p[:38]}')", flush=True)
  print("[slider-eval] DONE -- if detail rises left->right, the slider works.", flush=True)


if __name__ == "__main__":
  main()
