"""Visual verify for a regular text-to-image LoRA: [BASE | LoRA] montages.

Loads the full pipeline, injects the trained adapter into BOTH transformers, and for
each prompt generates the same seed with the adapter off vs on. If the subject/style
appears with the adapter on, the regular-LoRA path works.

  CUDA_VISIBLE_DEVICES=0 python eval_t2i.py \
    --ckpt runs/t2i-lora/ckpts/t2i_rank32_final.safetensors --rank 32
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
  """Wrap a plain prompt in the Ideogram-4 JSON schema (train/infer consistency).

  Uses a large central element bbox ([y,x,y,x] 0-1000); the subject text drives both
  the high-level description and the element desc.
  """
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
  ap = argparse.ArgumentParser(description="Visual verify of a regular T2I LoRA.")
  ap.add_argument("--config", default="config/t2i_lora.yaml")
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--rank", type=int, default=32)
  ap.add_argument("--res", type=int, default=1024)
  ap.add_argument("--preset", default="V4_DEFAULT_20", choices=list(PRESETS),
                  help="Sampler preset: V4_TURBO_12 (fast) | V4_DEFAULT_20 (medium, default) "
                       "| V4_QUALITY_48 (best). Mirrors the production inference quality presets.")
  ap.add_argument("--prompts", nargs="*", default=[
    "a portrait photograph of a person smiling, soft studio light",
    "a person giving a speech at a podium, formal blazer",
    "a person walking outdoors in a park, casual clothes",
  ])
  ap.add_argument("--out", default="private/t2i_verify")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  device = torch.device(cfg.runtime.device); dtype = dtype_of(cfg)

  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=cfg.paths.weights),
    device=cfg.runtime.device, dtype=dtype)

  wc = loramod.inject_lora(pipe.conditional_transformer, rank=args.rank)
  _load_lora(wc, args.ckpt, device, dtype)
  wu = loramod.inject_lora(pipe.unconditional_transformer, rank=args.rank)
  _load_lora(wu, args.ckpt, device, dtype)
  pipe.conditional_transformer.eval(); pipe.unconditional_transformer.eval()
  print(f"[t2i-eval] adapter into both transformers ({len(wc)}+{len(wu)} modules)", flush=True)

  os.makedirs(args.out, exist_ok=True)
  preset = PRESETS[args.preset]
  print(f"[t2i-eval] preset {args.preset}: {preset.num_steps} steps, mu={preset.mu} std={preset.std} "
        f"@ {args.res}px (guidance_schedule)", flush=True)

  def gen(prompt):
    return pipe([to_json_caption(prompt)], height=args.res, width=args.res,
                num_steps=preset.num_steps, guidance_schedule=preset.guidance_schedule,
                mu=preset.mu, std=preset.std, seed=0, raise_on_caption_issues=False)[0]

  for i, p in enumerate(args.prompts):
    with loramod.lora_disabled(pipe.conditional_transformer), \
         loramod.lora_disabled(pipe.unconditional_transformer):
      img_base = gen(p)
    img_lora = gen(p)
    H = args.res
    canvas = Image.new("RGB", (H * 2, H), (15, 15, 15))
    canvas.paste(img_base.resize((H, H)), (0, 0))
    canvas.paste(img_lora.resize((H, H)), (H, 0))
    pth = os.path.join(args.out, f"t2i_{i}.jpg")
    canvas.save(pth, quality=92)
    print(f"[t2i-eval] {i}: [BASE | LoRA] -> {pth}  ('{p[:42]}')", flush=True)
  print("[t2i-eval] DONE -- if the subject appears with LoRA on, regular LoRA works.", flush=True)


if __name__ == "__main__":
  main()
