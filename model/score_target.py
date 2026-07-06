import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from model.train_gan import _target_color_component_masks, prepared_image_paths


def _load_rgb_array(path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def score_array(array):
    red_mask, yellow_mask, blue_mask = _target_color_component_masks(array)
    inner_mask = red_mask | yellow_mask
    color_mask = inner_mask | blue_mask
    area = array.shape[0] * array.shape[1]

    blue_fraction = float(blue_mask.sum() / area)
    inner_fraction = float(inner_mask.sum() / area)
    color_fraction = float(color_mask.sum() / area)
    if color_mask.sum() == 0:
        return {
            "passed": False,
            "blue_fraction": blue_fraction,
            "inner_fraction": inner_fraction,
            "color_fraction": color_fraction,
            "inner_mean_radius": None,
            "blue_mean_radius": None,
            "radius_cv": None,
            "reason": "no_target_colored_pixels",
        }

    y_indices, x_indices = np.where(color_mask)
    center_y = float(y_indices.mean())
    center_x = float(x_indices.mean())
    yy, xx = np.indices(color_mask.shape)
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    inner_mean_radius = float(radius[inner_mask].mean()) if inner_mask.any() else float("inf")
    blue_mean_radius = float(radius[blue_mask].mean()) if blue_mask.any() else 0.0
    color_radii = radius[color_mask]
    radius_cv = float(color_radii.std() / max(color_radii.mean(), 1e-8))

    passed = (
        blue_fraction >= 0.003
        and inner_fraction >= 0.01
        and color_fraction >= 0.025
        and color_fraction <= 0.22
        and inner_mean_radius < blue_mean_radius * 2.5
        and radius_cv <= 0.85
    )
    reason = "passed" if passed else "failed_thresholds"
    return {
        "passed": bool(passed),
        "blue_fraction": blue_fraction,
        "inner_fraction": inner_fraction,
        "color_fraction": color_fraction,
        "inner_mean_radius": inner_mean_radius,
        "blue_mean_radius": blue_mean_radius,
        "radius_cv": radius_cv,
        "reason": reason,
    }


def score_image(path: Path):
    result = score_array(_load_rgb_array(path))
    result["path"] = str(path)
    return result


def score_directory(image_dir: Path):
    paths = prepared_image_paths(image_dir)
    results = [score_image(path) for path in paths]
    pass_count = sum(1 for result in results if result["passed"])
    return {
        "image_count": len(results),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(results) if results else 0.0,
        "results": results,
    }


def _image_vector(path: Path, size=128):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size))).astype(np.float32) / 255.0
    return array.reshape(-1)


def mean_pairwise_pixel_difference(paths, size=128):
    paths = list(paths)
    if len(paths) < 2:
        return 0.0
    vectors = [_image_vector(path, size=size) for path in paths]
    differences = [float(np.abs(vectors[i] - vectors[j]).mean()) for i, j in combinations(range(len(vectors)), 2)]
    return float(np.mean(differences))


def diversity_ratio(generated_dir: Path, real_dir: Path):
    generated_paths = prepared_image_paths(generated_dir)
    real_paths = prepared_image_paths(real_dir)
    generated_difference = mean_pairwise_pixel_difference(generated_paths)
    real_difference = mean_pairwise_pixel_difference(real_paths)
    ratio = generated_difference / max(real_difference, 1e-8)
    return {
        "generated_mean_pairwise_pixel_difference": generated_difference,
        "real_mean_pairwise_pixel_difference": real_difference,
        "diversity_ratio": ratio,
        "passes_diversity_gate": ratio >= 0.30,
    }


def write_score_csv(summary, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "passed",
        "blue_fraction",
        "inner_fraction",
        "color_fraction",
        "inner_mean_radius",
        "blue_mean_radius",
        "radius_cv",
        "reason",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["results"])


def main():
    parser = argparse.ArgumentParser(description="Score generated target images with a frozen HSV geometry heuristic.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = score_directory(args.image_dir)
    if args.real_dir is not None:
        summary["diversity"] = diversity_ratio(args.image_dir, args.real_dir)
    if args.out is not None:
        write_score_csv(summary, args.out)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
