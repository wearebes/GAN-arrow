import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from model.train_gan import prepared_image_paths


VARIANTS = ("original", "hflip", "affine", "photo", "affine_photo")


def _sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _border_color(image: Image.Image):
    pixels = np.asarray(image.convert("RGB"))
    border = np.concatenate(
        [pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]],
        axis=0,
    )
    return tuple(np.median(border, axis=0).astype(np.uint8).tolist())


def _mild_affine(image: Image.Image, rng: random.Random, *, combined=False):
    width, height = image.size
    angle_limit = 3.0 if combined else 4.0
    shift_limit = 0.0125 if combined else 0.02
    scale_limit = 0.02 if combined else 0.035
    angle = rng.uniform(-angle_limit, angle_limit)
    scale = rng.uniform(1.0 - scale_limit, 1.0 + scale_limit)
    translate_x = rng.uniform(-shift_limit, shift_limit) * width
    translate_y = rng.uniform(-shift_limit, shift_limit) * height
    fill = _border_color(image)
    transformed = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )
    inverse_scale = 1.0 / scale
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    offset_x = center_x * (1.0 - inverse_scale) - translate_x * inverse_scale
    offset_y = center_y * (1.0 - inverse_scale) - translate_y * inverse_scale
    transformed = transformed.transform(
        image.size,
        Image.Transform.AFFINE,
        (inverse_scale, 0.0, offset_x, 0.0, inverse_scale, offset_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
    return transformed, {
        "angle_degrees": round(angle, 6),
        "scale": round(scale, 6),
        "translate_x_pixels": round(translate_x, 6),
        "translate_y_pixels": round(translate_y, 6),
    }


def _mild_photo(image: Image.Image, rng: random.Random, *, combined=False):
    limit = 0.06 if combined else 0.10
    brightness = rng.uniform(1.0 - limit, 1.0 + limit)
    contrast = rng.uniform(1.0 - limit, 1.0 + limit)
    saturation = rng.uniform(1.0 - limit, 1.0 + limit)
    transformed = ImageEnhance.Brightness(image).enhance(brightness)
    transformed = ImageEnhance.Contrast(transformed).enhance(contrast)
    transformed = ImageEnhance.Color(transformed).enhance(saturation)
    return transformed, {
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "saturation": round(saturation, 6),
    }


def _save_original(source: Path, output_path: Path):
    if source.suffix.lower() == ".png":
        os.link(source, output_path)
    else:
        with Image.open(source) as image:
            image.convert("RGB").save(output_path)


def _make_preview(rows, output_path: Path, sources_to_show=4, cell_size=224):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_sources = []
    for row in rows:
        if row["source_path"] not in selected_sources:
            selected_sources.append(row["source_path"])
        if len(selected_sources) >= sources_to_show:
            break
    selected = [row for row in rows if row["source_path"] in selected_sources]
    label_height = 24
    sheet = Image.new(
        "RGB",
        (len(VARIANTS) * cell_size, len(selected_sources) * (cell_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, source_path in enumerate(selected_sources):
        source_rows = [row for row in selected if row["source_path"] == source_path]
        by_variant = {row["variant"]: row for row in source_rows}
        for column, variant in enumerate(VARIANTS):
            row = by_variant[variant]
            with Image.open(row["output_path"]) as image:
                thumbnail = ImageOps.fit(image.convert("RGB"), (cell_size, cell_size))
            x = column * cell_size
            y = row_index * (cell_size + label_height)
            sheet.paste(thumbnail, (x, y))
            draw.text((x + 4, y + cell_size + 4), variant, fill="black")
    sheet.save(output_path)


def augment_dataset(input_dir: Path, output_dir: Path, seed=42):
    input_paths = prepared_image_paths(input_dir)
    if not input_paths:
        raise ValueError(f"No prepared images found in {input_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.png")):
        raise FileExistsError(f"Augmented dataset already contains PNG images: {output_dir}")

    rng = random.Random(seed)
    rows = []
    for source in input_paths:
        source_hash = _sha256(source)
        with Image.open(source) as opened:
            image = opened.convert("RGB")

        outputs = []
        original_path = output_dir / f"orig_{source.stem}.png"
        _save_original(source, original_path)
        outputs.append(("original", original_path, {}))

        flip_path = output_dir / f"aug_hflip_{source.stem}.png"
        ImageOps.mirror(image).save(flip_path)
        outputs.append(("hflip", flip_path, {"horizontal_flip": True}))

        affine_image, affine_params = _mild_affine(image, rng)
        affine_path = output_dir / f"aug_affine_{source.stem}.png"
        affine_image.save(affine_path)
        outputs.append(("affine", affine_path, affine_params))

        photo_image, photo_params = _mild_photo(image, rng)
        photo_path = output_dir / f"aug_photo_{source.stem}.png"
        photo_image.save(photo_path)
        outputs.append(("photo", photo_path, photo_params))

        combined_image, combined_affine_params = _mild_affine(image, rng, combined=True)
        combined_image, combined_photo_params = _mild_photo(combined_image, rng, combined=True)
        combined_path = output_dir / f"aug_affine_photo_{source.stem}.png"
        combined_image.save(combined_path)
        outputs.append(
            (
                "affine_photo",
                combined_path,
                {**combined_affine_params, **combined_photo_params},
            )
        )

        for variant, output_path, parameters in outputs:
            rows.append(
                {
                    "output_path": str(output_path),
                    "source_path": str(source),
                    "variant": variant,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "source_sha256": source_hash,
                    "output_sha256": _sha256(output_path),
                }
            )

    manifest_path = output_dir / "augmentation_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "version": "whole_image_mild_v1",
        "seed": seed,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_count": len(input_paths),
        "variants_per_source": len(VARIANTS),
        "output_count": len(rows),
        "variants": list(VARIANTS),
        "excluded_transforms": [
            "vertical_flip",
            "mosaic",
            "mixup",
            "large_crop",
            "heavy_blur",
            "cutout",
        ],
        "manifest": str(manifest_path),
        "preview": str(output_dir / "_review" / "preview.png"),
    }
    (output_dir / "preprocessing_report.json").write_text(json.dumps(summary, indent=2))
    _make_preview(rows, output_dir / "_review" / "preview.png")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create a separate mild whole-image augmentation dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(augment_dataset(args.input, args.out, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
