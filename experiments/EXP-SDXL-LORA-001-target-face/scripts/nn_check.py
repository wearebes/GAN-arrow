"""Memorization check: nearest training image for each generated sample.

Compares generated PNGs against the train copies in pixel space at 128x128
(L2). Writes a side-by-side sheet (generated | nearest train | distance) so
copying can be judged visually, plus a JSON with distances. Also reports the
train-train nearest-neighbor distance distribution as a reference scale: a
generated image markedly closer to a train image than typical train-train
neighbors indicates memorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def embed(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((128, 128), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32).ravel() / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-glob", required=True, help="e.g. outputs/samples/pilot/lora_step0400_p0_*.png")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    gen_files = sorted(Path().glob(args.generated_glob)) or sorted(
        Path(args.generated_glob).parent.glob(Path(args.generated_glob).name)
    )
    if not gen_files:
        raise SystemExit(f"no files match {args.generated_glob}")
    train_files = sorted(Path(args.train_dir).glob("*.png"))

    train_vecs = np.stack([embed(f) for f in train_files])

    # Reference scale: train-train nearest neighbor distances.
    tt = []
    for i in range(len(train_files)):
        d = np.linalg.norm(train_vecs - train_vecs[i], axis=1)
        d[i] = np.inf
        tt.append(float(d.min()))
    tt_med = float(np.median(tt))

    results = []
    cell = 384
    sheet = Image.new("RGB", (2 * cell, len(gen_files) * (cell + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, gf in enumerate(gen_files):
        v = embed(gf)
        d = np.linalg.norm(train_vecs - v, axis=1)
        j = int(d.argmin())
        results.append(
            {"generated": gf.name, "nearest_train": train_files[j].name,
             "distance": float(d[j]), "ratio_to_train_train_median": float(d[j] / tt_med)}
        )
        y0 = r * (cell + 24)
        draw.text((6, y0 + 4),
                  f"{gf.name}  vs  {train_files[j].name}   L2={d[j]:.2f}  (train-train med {tt_med:.2f})",
                  fill="black")
        sheet.paste(Image.open(gf).convert("RGB").resize((cell, cell)), (0, y0 + 24))
        sheet.paste(Image.open(train_files[j]).convert("RGB").resize((cell, cell)), (cell, y0 + 24))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_prefix.with_suffix(".png"))
    out_prefix.with_suffix(".json").write_text(
        json.dumps({"train_train_nn_median": tt_med, "results": results}, indent=2)
    )
    print(f"saved {out_prefix.with_suffix('.png')}")
    for r in results:
        print(f"  {r['generated']}: nn={r['nearest_train']} ratio={r['ratio_to_train_train_median']:.2f}")


if __name__ == "__main__":
    main()
