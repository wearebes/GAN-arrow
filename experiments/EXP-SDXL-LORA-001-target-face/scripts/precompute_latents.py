"""Precompute SDXL VAE latent distributions and text embeddings.

Reads the prepared train/val copies under data/ (never the original dataset
directory), encodes every image at the requested resolutions, and caches
tensors under outputs/cache/.  Train images are additionally cached as a
horizontally flipped variant so the trainer can sample either without needing
the VAE at train time.

The VAE and text encoders are loaded one at a time and freed afterwards so the
whole step stays well under memory limits on a 24 GB machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def load_image(path: Path, res: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
    return arr.permute(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolutions", default="768,1024")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    vae = AutoencoderKL.from_pretrained(
        MODEL, subfolder="vae", variant="fp16", torch_dtype=torch.float32
    ).to(device).eval()

    for res_str in args.resolutions.split(","):
        res = int(res_str)
        for split, use_flip in (("train", True), ("val", False)):
            files = sorted((Path(args.data_root) / split).glob("*.png"))
            if not files:
                raise SystemExit(f"no PNGs found in {Path(args.data_root) / split}")
            params, names, flipped = [], [], []
            for f in files:
                img = load_image(f, res)
                variants = [(img, False)]
                if use_flip:
                    variants.append((torch.flip(img, dims=[2]), True))
                for tensor, is_flip in variants:
                    with torch.no_grad():
                        dist = vae.encode(tensor.unsqueeze(0).to(device)).latent_dist
                    params.append(dist.parameters.squeeze(0).cpu())
                    names.append(f.name)
                    flipped.append(is_flip)
            payload = {
                "latent_params": torch.stack(params),
                "files": names,
                "flipped": flipped,
                "resolution": res,
                "scaling_factor": vae.config.scaling_factor,
            }
            torch.save(payload, out / f"{split}_{res}.pt")
            print(f"cached {split}@{res}: {len(names)} entries", flush=True)

    del vae
    if device.type == "mps":
        torch.mps.empty_cache()

    tok1 = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(
        MODEL, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32
    ).to(device).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(
        MODEL, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float32
    ).to(device).eval()

    embeds, pooled = [], None
    with torch.no_grad():
        for tok, te, is_te2 in ((tok1, te1, False), (tok2, te2, True)):
            ids = tok(
                args.prompt,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            enc = te(ids, output_hidden_states=True)
            if is_te2:
                pooled = enc[0]
            embeds.append(enc.hidden_states[-2])
    prompt_embeds = torch.cat(embeds, dim=-1)

    torch.save(
        {
            "prompt": args.prompt,
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled_embeds": pooled.cpu(),
        },
        out / "text_embeds.pt",
    )
    (out / "precompute_meta.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "prompt": args.prompt,
                "resolutions": args.resolutions,
                "prompt_embeds_shape": list(prompt_embeds.shape),
            },
            indent=2,
        )
    )
    print("cached text embeddings:", tuple(prompt_embeds.shape), flush=True)


if __name__ == "__main__":
    main()
