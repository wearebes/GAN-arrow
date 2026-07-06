import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFilter


RING_COLORS = {
    "paper": (226, 232, 232),
    "outer": (36, 37, 32),
    "blue": (28, 157, 210),
    "red": (211, 53, 57),
    "yellow": (239, 210, 54),
}


def _jitter_color(color, amount=10):
    return tuple(max(0, min(255, channel + random.randint(-amount, amount))) for channel in color)


def _draw_arrow(draw: ImageDraw.ImageDraw, center, radius):
    if random.random() > 0.75:
        return
    angle = random.uniform(0, math.tau)
    end_distance = random.uniform(radius * 0.05, radius * 0.42)
    start_distance = random.uniform(radius * 0.75, radius * 1.22)
    end = (
        center[0] + math.cos(angle) * end_distance,
        center[1] + math.sin(angle) * end_distance,
    )
    start = (
        center[0] + math.cos(angle) * start_distance,
        center[1] + math.sin(angle) * start_distance,
    )
    draw.line([start, end], fill=(35, 35, 32), width=random.randint(2, 4))
    draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=(225, 225, 215))


def _perspective_coefficients(source, destination):
    matrix = []
    for (x_src, y_src), (x_dst, y_dst) in zip(source, destination):
        matrix.append([x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src])
        matrix.append([0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src])
    a = torch.tensor(matrix, dtype=torch.float64)
    b = torch.tensor([coord for point in destination for coord in point], dtype=torch.float64)
    return torch.linalg.solve(a, b).tolist()


def make_target_image(image_size=256):
    canvas_size = int(image_size * 1.35)
    image = Image.new("RGB", (canvas_size, canvas_size), _jitter_color(RING_COLORS["paper"], 8))
    draw = ImageDraw.Draw(image)
    center = (
        canvas_size / 2 + random.uniform(-canvas_size * 0.035, canvas_size * 0.035),
        canvas_size / 2 + random.uniform(-canvas_size * 0.035, canvas_size * 0.035),
    )
    radius = canvas_size * random.uniform(0.38, 0.43)
    rings = [
        (1.00, "outer"),
        (0.78, "paper"),
        (0.68, "blue"),
        (0.47, "red"),
        (0.26, "yellow"),
    ]
    for scale, color_name in rings:
        ring_radius = radius * scale
        box = (
            center[0] - ring_radius,
            center[1] - ring_radius,
            center[0] + ring_radius,
            center[1] + ring_radius,
        )
        draw.ellipse(box, fill=_jitter_color(RING_COLORS[color_name], 9))

    for scale in (0.16, 0.08):
        ring_radius = radius * scale
        box = (
            center[0] - ring_radius,
            center[1] - ring_radius,
            center[0] + ring_radius,
            center[1] + ring_radius,
        )
        draw.ellipse(box, outline=(180, 145, 35), width=2)

    for _ in range(random.randint(0, 3)):
        _draw_arrow(draw, center, radius)

    pad = int(canvas_size * 0.08)
    source = [(0, 0), (canvas_size, 0), (canvas_size, canvas_size), (0, canvas_size)]
    destination = [
        (random.randint(0, pad), random.randint(0, pad)),
        (canvas_size - random.randint(0, pad), random.randint(0, pad)),
        (canvas_size - random.randint(0, pad), canvas_size - random.randint(0, pad)),
        (random.randint(0, pad), canvas_size - random.randint(0, pad)),
    ]
    coeffs = _perspective_coefficients(source, destination)
    image = image.transform((canvas_size, canvas_size), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    image = image.rotate(random.uniform(-4, 4), resample=Image.Resampling.BICUBIC, fillcolor=RING_COLORS["paper"])
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.45)))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return image


def generate_targets(output_dir: Path, count: int = 32, image_size: int = 256, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()
    tensors = []
    for index in range(1, count + 1):
        image = make_target_image(image_size=image_size)
        output_path = samples_dir / f"target_{index:03d}.jpg"
        image.save(output_path, format="JPEG", quality=95)
        tensors.append(to_tensor(image))
    grid_path = output_dir / "target_grid.png"
    vutils.save_image(torch.stack(tensors), grid_path, nrow=4)
    metrics = {
        "generated_count": count,
        "image_size": image_size,
        "seed": seed,
        "samples_dir": str(samples_dir),
        "grid": str(grid_path),
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Generate usable synthetic 256px arrow target images.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/effective_procedural_targets_256"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = generate_targets(args.output_dir, count=args.count, image_size=args.image_size, seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
