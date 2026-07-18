"""Plot train/val loss curves from a run directory into outputs/eval/."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> dict[str, list[float]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[float]] = {}
    for key in rows[0].keys():
        out[key] = [float(r[key]) for r in rows]
    return out


def rolling(xs: list[float], k: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        out.append(sum(xs[lo : i + 1]) / (i - lo + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    run = Path(args.run_dir)
    train = read_csv(run / "train_log.csv")
    val = read_csv(run / "val_log.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.plot(train["step"], train["loss"], alpha=0.25, lw=0.8, label="train loss (raw)")
    ax.plot(train["step"], rolling(train["loss"], 25), lw=1.8, label="train loss (rolling 25)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("noise-pred MSE")
    ax.set_title("train loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(val["step"], val["val_loss"], "o-", lw=1.8, label="val (mean)")
    for key in val:
        if key.startswith("t"):
            ax.plot(val["step"], val[key], alpha=0.45, lw=1.0, label=f"val {key}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("noise-pred MSE (fixed grid)")
    ax.set_title("val loss (8 held-out images)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
