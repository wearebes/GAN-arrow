import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image


PREPARED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def prepared_image_paths(image_dir: Path):
    preferred_by_stem = {}
    extension_rank = {extension: rank for rank, extension in enumerate(PREPARED_IMAGE_EXTENSIONS)}
    for path in Path(image_dir).iterdir():
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in extension_rank:
            continue
        current = preferred_by_stem.get(path.stem)
        if current is None or extension_rank[suffix] < extension_rank[current.suffix.lower()]:
            preferred_by_stem[path.stem] = path
    return [preferred_by_stem[stem] for stem in sorted(preferred_by_stem)]


def build_augmentation(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02)],
                p=0.8,
            ),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.025, 0.025),
                scale=(0.94, 1.06),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=128,
            ),
            transforms.RandomPerspective(distortion_scale=0.06, p=0.35, fill=128),
        ]
    )


def generate_augmented(processed_dir: Path, output_dir: Path, count: int = 32, image_size: int = 256, seed: int = 42):
    paths = prepared_image_paths(processed_dir)
    if not paths:
        raise ValueError(f"No prepared images found in {processed_dir}")

    random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    augment = build_augmentation(image_size)
    tensors = []
    written = []
    to_tensor = transforms.ToTensor()
    for index in range(1, count + 1):
        source_path = random.choice(paths)
        with Image.open(source_path) as image:
            generated = augment(image.convert("RGB"))
        output_path = samples_dir / f"augmented_{index:03d}.jpg"
        generated.save(output_path, format="JPEG", quality=95)
        written.append(output_path)
        tensors.append(to_tensor(generated))

    grid_path = output_dir / "augmented_grid.png"
    vutils.save_image(torch.stack(tensors), grid_path, nrow=4)

    metrics = {
        "source_count": len(paths),
        "generated_count": len(written),
        "image_size": image_size,
        "seed": seed,
        "samples_dir": str(samples_dir),
        "grid": str(grid_path),
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Generate usable 256px arrow-target augmentations from cleaned ROI images.")
    parser.add_argument("--processed-dir", type=Path, default=Path("dataset/generate_data/processed_256_front"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/effective_augmented_256"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = generate_augmented(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        count=args.count,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
