"""Matched comparison: FULL fine-tune vs rank-512 LoRA on held-out structural edits.

Loads each model in turn, generates the same held-out cached items at a fixed guidance
and seed, decodes, and scores two pixel metrics against the ground-truth pair:

  * preservation MAE -- mean |source - gen| over UNEDITED pixels (where |source-target|
    is small). Lower = the model left the background alone.
  * edit-fidelity MAE -- mean |target - gen| over EDITED pixels (where |source-target|
    is large). Lower = the model's edit matches the ground-truth edit.

Writes one montage ``[source | full-FT | LoRA | target]`` per item (with per-model MAEs)
and a metrics summary, so "which preserves better while still editing" is decided on
the same inputs. The eval items must be the ones held out of BOTH trainings.

  CUDA_VISIBLE_DEVICES=0 python eval_edit_compare.py \
    --fft runs/cmp-full/ckpts/edit_full_final.safetensors \
    --lora runs/cmp-r512/ckpts/edit_lora_rank512_final.safetensors --rank 512 \
    --eval-list config/cmp_eval_list.json --config config/edit_full.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from ideogram4.pipeline_ideogram4 import (
  Ideogram4PipelineConfig, _build_transformer, _load_indexed_or_single_state_dict,
)
from ideogram4.modeling_ideogram4 import Ideogram4Config
from ideogram4.scheduler import get_schedule_for_resolution
from ideogram4 import train_edit, edit_sampler
from ideogram4 import lora as loramod


def _build_fp8(weights, device, dtype):
  pcfg = Ideogram4PipelineConfig(weights_repo=weights)
  sd = _load_indexed_or_single_state_dict(pcfg.weights_repo, pcfg.conditional_index_filename)
  t = _build_transformer(Ideogram4Config(), sd, device, dtype)
  del sd
  return t


def _gen(transformer, it, device, schedule, steps, guidance, seed):
  gh, gw = int(it["grid_h"]), int(it["grid_w"])
  return edit_sampler.sample_edit_cached(
    transformer, it["z_ref"].to(device), it["llm_text"].to(device), gh, gw,
    schedule=schedule, num_steps=steps, guidance_scale=guidance,
    generator=torch.Generator(device=device).manual_seed(seed))


def _label(img, lines):
  d = ImageDraw.Draw(img)
  d.rectangle([0, 0, 150, 12 + 12 * len(lines)], fill=(0, 0, 0))
  for i, t in enumerate(lines):
    d.text((4, 3 + 12 * i), t, fill=(245, 245, 245))
  return img


def main():
  ap = argparse.ArgumentParser(description="FFT vs LoRA structural-edit comparison.")
  ap.add_argument("--config", default="config/edit_full.yaml")
  ap.add_argument("--fft", required=True, help="full-FT safetensors")
  ap.add_argument("--lora", required=True, help="rank-512 LoRA safetensors")
  ap.add_argument("--rank", type=int, default=512)
  ap.add_argument("--eval-list", default="config/cmp_eval_list.json")
  ap.add_argument("--steps", type=int, default=24)
  ap.add_argument("--guidance", type=float, default=3.5)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--edit-thresh", type=float, default=18.0,
                  help="pixel |src-tgt| above this = edited region (0-255).")
  ap.add_argument("--out", default="runs/cmp-eval")
  args = ap.parse_args()

  from ideogram4.training_config import load_config, apply_runtime, dtype_of
  cfg = load_config(args.config); apply_runtime(cfg)
  device = torch.device(cfg.runtime.device); dtype = dtype_of(cfg)
  cache = cfg.paths.cache_dir
  res = int(cfg.data.resolution)
  schedule = get_schedule_for_resolution((res, res), known_mean=1.0)
  os.makedirs(args.out, exist_ok=True)

  files = json.load(open(args.eval_list))
  items = [torch.load(f"{cache}/{f}", map_location="cpu") for f in files]
  print(f"[cmp] {len(items)} held-out structural items", flush=True)

  from safetensors.torch import load_file

  # --- arm A: full fine-tune (dequantize + expand ref embed + strict load) ---
  print("[cmp] loading full-FT ...", flush=True)
  tA = _build_fp8(cfg.paths.weights, device, dtype)
  train_edit.dequantize_fp8_transformer(tA, dtype=dtype)
  train_edit.expand_reference_embedding(tA)
  tA.to(device).eval()
  tA.load_state_dict({k: v.to(device, dtype) for k, v in load_file(args.fft).items()}, strict=True)
  gen_fft = []
  with torch.no_grad():
    for it in items:
      gen_fft.append(_gen(tA, it, device, schedule, args.steps, args.guidance, args.seed).cpu())
  del tA
  torch.cuda.empty_cache()
  print("[cmp] full-FT generations done", flush=True)

  # --- arm B: rank-512 LoRA on the frozen fp8 base ---
  print("[cmp] loading rank-512 LoRA ...", flush=True)
  tB = _build_fp8(cfg.paths.weights, device, dtype)
  wrapped = loramod.inject_lora(tB, rank=args.rank)
  st = load_file(args.lora)
  with torch.no_grad():
    for sub, m in wrapped.items():
      m.lora_A.copy_(st[f"diffusion_model.{sub}.lora_A.weight"].to(device, dtype))
      m.lora_B.copy_(st[f"diffusion_model.{sub}.lora_B.weight"].to(device, dtype))
  tB.eval()
  gen_lora = []
  with torch.no_grad():
    for it in items:
      gen_lora.append(_gen(tB, it, device, schedule, args.steps, args.guidance, args.seed).cpu())
  del tB
  torch.cuda.empty_cache()
  print("[cmp] LoRA generations done", flush=True)

  # --- decode + score ---
  ae, shift, scale, patch = edit_sampler.load_decoder(cfg.paths.weights, device, dtype)

  def decode(z, gh, gw):
    return edit_sampler.decode_latents(ae, z.to(device), gh, gw, patch_size=patch,
                                       latent_shift=shift, latent_scale=scale, dtype=dtype)[0]

  rows, agg = [], {"fft": {"pres": [], "edit": []}, "lora": {"pres": [], "edit": []}}
  for n, (f, it) in enumerate(zip(files, items)):
    gh, gw = int(it["grid_h"]), int(it["grid_w"])
    src = decode(it["z_ref"].unsqueeze(0), gh, gw)
    tgt = decode(it["z_tgt"].unsqueeze(0), gh, gw)
    gf = decode(gen_fft[n], gh, gw)
    gl = decode(gen_lora[n], gh, gw)
    s = np.asarray(src, np.float32); t = np.asarray(tgt, np.float32)
    edited = np.abs(s - t).mean(-1) >= args.edit_thresh     # GT edit mask
    unedited = ~edited
    def metrics(g):
      g = np.asarray(g, np.float32)
      pres = float(np.abs(s - g).mean(-1)[unedited].mean()) if unedited.any() else float("nan")
      edit = float(np.abs(t - g).mean(-1)[edited].mean()) if edited.any() else float("nan")
      return pres, edit
    pf, ef = metrics(gf); pl, el = metrics(gl)
    agg["fft"]["pres"].append(pf); agg["fft"]["edit"].append(ef)
    agg["lora"]["pres"].append(pl); agg["lora"]["edit"].append(el)
    idx = int(f[:-3])
    rows.append({"idx": idx, "edit_pct": round(float(edited.mean()) * 100, 1),
                 "fft_pres": round(pf, 2), "fft_edit": round(ef, 2),
                 "lora_pres": round(pl, 2), "lora_edit": round(el, 2),
                 "instruction": str(it.get("edit", ""))[:80]})
    # montage [src | FFT | LoRA | tgt]
    _label(gf, [f"full-FT", f"pres {pf:.1f}", f"edit {ef:.1f}"])
    _label(gl, [f"LoRA r{args.rank}", f"pres {pl:.1f}", f"edit {el:.1f}"])
    _label(src, ["source"]); _label(tgt, ["target"])
    w, h = src.size
    canvas = Image.new("RGB", (w * 4, h), (15, 15, 15))
    for k, im in enumerate((src, gf, gl, tgt)):
      canvas.paste(im, (k * w, 0))
    canvas.save(os.path.join(args.out, f"cmp_idx{idx:06d}.jpg"), quality=92)
    print(f"[cmp] idx{idx}: FFT(pres {pf:.1f}/edit {ef:.1f}) LoRA(pres {pl:.1f}/edit {el:.1f})", flush=True)

  def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")
  summary = {
    "n": len(items),
    "fft":  {"preservation_mae": round(mean(agg["fft"]["pres"]), 2),
             "edit_fidelity_mae": round(mean(agg["fft"]["edit"]), 2)},
    "lora": {"preservation_mae": round(mean(agg["lora"]["pres"]), 2),
             "edit_fidelity_mae": round(mean(agg["lora"]["edit"]), 2)},
    "rows": rows,
  }
  json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, ensure_ascii=False)
  print("\n=== AGGREGATE (lower is better) ===", flush=True)
  print(f"  full-FT : preservation {summary['fft']['preservation_mae']:5.2f} | "
        f"edit-fidelity {summary['fft']['edit_fidelity_mae']:5.2f}", flush=True)
  print(f"  LoRA512 : preservation {summary['lora']['preservation_mae']:5.2f} | "
        f"edit-fidelity {summary['lora']['edit_fidelity_mae']:5.2f}", flush=True)
  print(f"[cmp] montages + summary.json -> {args.out}", flush=True)


if __name__ == "__main__":
  main()
