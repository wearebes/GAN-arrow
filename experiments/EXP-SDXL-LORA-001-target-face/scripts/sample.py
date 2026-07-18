"""Generate sample grids from SDXL base, optionally with a trained LoRA.

Runs the full fp16 pipeline (VAE auto-upcasts for decode), one image at a
time, and writes both individual PNGs and a row-per-prompt contact sheet.
Seeds use a CPU generator so results are reproducible across runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def contact_sheet(rows: list[tuple[str, list[Image.Image]]], cell: int) -> Image.Image:
    cols = max(len(imgs) for _, imgs in rows)
    label_h = 28
    sheet = Image.new(
        "RGB", (cols * cell, len(rows) * (cell + label_h)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for r, (label, imgs) in enumerate(rows):
        y0 = r * (cell + label_h)
        draw.text((6, y0 + 6), label[:160], fill="black")
        for c, im in enumerate(imgs):
            sheet.paste(im.resize((cell, cell), Image.LANCZOS), (c * cell, y0 + label_h))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None, help="checkpoint dir with pytorch_lora_weights.safetensors")
    ap.add_argument("--prompts-file", required=True, help="text file, one prompt per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="filename prefix, e.g. lora_step400 or base")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--cell", type=int, default=384)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from diffusers import StableDiffusionXLPipeline

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = [
        line.strip()
        for line in Path(args.prompts_file).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16"
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.load_lora_weights(args.lora)
        print(f"loaded LoRA from {args.lora}", flush=True)

    rows = []
    for pi, prompt in enumerate(prompts):
        imgs = []
        for seed in seeds:
            g = torch.Generator("cpu").manual_seed(seed)
            img = pipe(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.res,
                width=args.res,
                generator=g,
            ).images[0]
            img.save(out_dir / f"{args.tag}_p{pi}_s{seed}.png")
            imgs.append(img)
            print(f"prompt {pi} seed {seed} done", flush=True)
        rows.append((f"[{pi}] {prompt}", imgs))

    sheet = contact_sheet(rows, args.cell)
    sheet_path = out_dir / f"{args.tag}_sheet.png"
    sheet.save(sheet_path)
    (out_dir / f"{args.tag}_meta.json").write_text(
        json.dumps(
            {"lora": args.lora, "prompts": prompts, "seeds": seeds,
             "steps": args.steps, "guidance": args.guidance, "res": args.res},
            indent=2,
        )
    )
    print(f"sheet saved: {sheet_path}", flush=True)


if __name__ == "__main__":
    main()
