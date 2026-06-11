"""Multi-reference inference: generate composites with a trained multi-ref adapter.

Visual proof for the multi-ref smoke. Loads the full pipeline + the trained LoRA,
encodes a few Echo-4o samples' references (human/object/scene) + instruction, runs
``edit_sampler.sample_multiref`` to generate the composite, and saves a montage
``[ref_1 | ref_2 | ... | GENERATED | TARGET]`` per sample for inspection.

  CUDA_VISIBLE_DEVICES=0 python eval_multiref.py --config config/edit_lora_cached.yaml \
    --category hum_obj_sce --ckpt runs/mref-smoke/ckpts/mref_smoke_rank64.safetensors \
    --rank 64 --n 4 --guidance 2.0
"""
import argparse
import json
import os
import tarfile

import torch
from PIL import Image

from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4 import train_edit, lora as loramod, edit_sampler
from ideogram4.constants import LLM_TOKEN_INDICATOR
from ideogram4.scheduler import get_schedule_for_resolution

_CATEGORY_REFS = {
  "hum_sce": ["human", "sence"], "hum_obj": ["human", "object"],
  "hum_obj_sce": ["human", "object", "sence"],
}


def _load_lora(wrapped, path, device, dtype):
  from safetensors.torch import load_file
  state = load_file(path)
  with torch.no_grad():
    for sub, m in wrapped.items():
      m.lora_A.copy_(state[f"diffusion_model.{sub}.lora_A.weight"].to(device, dtype))
      m.lora_B.copy_(state[f"diffusion_model.{sub}.lora_B.weight"].to(device, dtype))


def main():
  ap = argparse.ArgumentParser(description="Multi-reference inference montages.")
  ap.add_argument("--config", default="config/edit_lora_cached.yaml")
  ap.add_argument("--category", default="hum_obj_sce", choices=list(_CATEGORY_REFS))
  ap.add_argument("--data-root", default="data/multiref-dataset",
                  help="Echo-4o-style multi-reference dataset root (multi-ic/*.jsonl + tar archives).")
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--rank", type=int, default=64)
  ap.add_argument("--n", type=int, default=4)
  ap.add_argument("--guidance", type=float, default=2.0)
  ap.add_argument("--steps", type=int, default=24)
  ap.add_argument("--res", type=int, default=512)
  ap.add_argument("--out", default="runs/mref-smoke/infer")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  device = torch.device(cfg.runtime.device); dtype = dtype_of(cfg)

  B = args.data_root
  rows = [json.loads(l) for l in open(f"{B}/multi-ic/{args.category}.jsonl", encoding="utf-8")][:args.n]
  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=cfg.paths.weights), device=cfg.runtime.device, dtype=dtype)
  patch = pipe.config.patch_size * pipe.config.ae_scale_factor
  ps = pipe.config.patch_size
  grid = args.res // patch
  res = args.res

  wrapped = loramod.inject_lora(pipe.conditional_transformer, rank=args.rank)
  _load_lora(wrapped, args.ckpt, device, dtype)
  pipe.conditional_transformer.eval()
  print(f"[infer-mref] loaded adapter {args.ckpt}", flush=True)

  tarc = {}
  def extract(archive, member):
    if archive not in tarc:
      tarc[archive] = tarfile.open(archive, "r:gz")
    return Image.open(tarc[archive].extractfile(member)).convert("RGB")

  os.makedirs(args.out, exist_ok=True)
  sched = get_schedule_for_resolution((res, res))
  gen = torch.Generator(device=device).manual_seed(0)

  for i, r in enumerate(rows):
    ref_imgs, z_refs = [], []
    for p in r["input_images"]:
      cat, fn = p.split("/")[-2], p.split("/")[-1]
      img = extract(f"{B}/multi_refer_images_input/{cat}/{cat}.tar.gz", fn)
      ref_imgs.append(img)
      ten = train_edit.images_to_tensor([img], res, res, pipe.device)
      z_refs.append(train_edit.encode_image_tokens(pipe, ten, patch_size=ps)[0].to(torch.float32))
    op = r["output_image"]; ocat, ofn = op.split("/")[-2], op.split("/")[-1]
    tgt_img = extract(f"{B}/multi_refer_images_output/{ocat}/{ocat}.tar.gz", ofn)

    inputs = train_edit.build_edit_inputs(pipe, [r["instruction"]], grid, grid)
    llm = pipe._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
    llm_text = llm[0][inputs["indicator"][0] == LLM_TOKEN_INDICATOR].to(torch.float32)

    z = edit_sampler.sample_multiref(
      pipe.conditional_transformer, z_refs, [(grid, grid)] * len(z_refs), llm_text, (grid, grid),
      schedule=sched, num_steps=args.steps, guidance_scale=args.guidance, generator=gen)
    gen_img = pipe._decode(z, grid_h=grid, grid_w=grid)[0]

    panels = ref_imgs + [gen_img, tgt_img]
    H = 320
    rs = [im.resize((max(1, int(im.width * H / im.height)), H)) for im in panels]
    canvas = Image.new("RGB", (sum(im.width for im in rs), H), (15, 15, 15))
    x = 0
    for im in rs:
      canvas.paste(im, (x, 0)); x += im.width
    pth = os.path.join(args.out, f"infer_{i:03d}.jpg")
    canvas.save(pth, quality=90)
    labels = "|".join(_CATEGORY_REFS[args.category]) + "|GENERATED|TARGET"
    print(f"[infer-mref] {i}: [{labels}] -> {pth}", flush=True)
  print("[infer-mref] DONE", flush=True)


if __name__ == "__main__":
  main()
