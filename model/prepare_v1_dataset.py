"""Build the immutable real-image ``dataset/v1_1024`` split once.

Only deterministic decode/crop/resize output is written.  Random GAN
augmentations stay in the training loop and are never materialized here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from model.train_gan import (
    IMAGE_EXTENSIONS,
    PREPROCESS_VERSION,
    image_has_signal,
    image_has_target_anchor,
    image_meets_minimum_resolution,
    prepare_images,
    prepared_image_paths,
)


SPLITS = ("train", "val", "test")


def _source_paths(source_dir: Path):
    return sorted(
        path for path in Path(source_dir).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _capture_groups(paths, group_size: int):
    """Keep short consecutive filename bursts in the same split."""
    groups = []
    current = []
    previous_number = None
    for path in paths:
        digits = "".join(character for character in path.stem if character.isdigit())
        number = int(digits) if digits else None
        is_consecutive = number is not None and previous_number is not None and number == previous_number + 1
        if current and (len(current) >= group_size or not is_consecutive):
            groups.append(current)
            current = []
        current.append(path)
        previous_number = number
    if current:
        groups.append(current)
    return groups


def _choose_groups(groups, target_count: int, rng: random.Random):
    if target_count <= 0:
        return set()
    order = list(range(len(groups)))
    rng.shuffle(order)
    choices = {0: ()}
    for index in order:
        size = len(groups[index])
        for total, selected in list(choices.items())[::-1]:
            choices.setdefault(total + size, selected + (index,))
    best_total = min(choices, key=lambda total: (abs(total - target_count), total > target_count, -total))
    return set(choices[best_total])


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear_generated_outputs(output_dir: Path):
    for split in SPLITS:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in split_dir.glob("*.png"):
            path.unlink()
    for name in ("manifest.csv", "preprocessing_report.json", "README.md"):
        (output_dir / name).unlink(missing_ok=True)


def validate_v1_dataset(output_dir: Path, image_size: int = 1024):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    seen_sources = set()
    seen_prepared = set()
    group_splits = {}
    split_counts = {split: 0 for split in SPLITS}
    for row in rows:
        split = row["split"]
        path = Path(row["prepared_path"])
        if split not in split_counts:
            errors.append(f"unsupported split for {path}: {split}")
            continue
        split_counts[split] += 1
        if row["source_path"] in seen_sources:
            errors.append(f"duplicate source: {row['source_path']}")
        seen_sources.add(row["source_path"])
        if str(path) in seen_prepared:
            errors.append(f"duplicate prepared path: {path}")
        seen_prepared.add(str(path))
        group_splits.setdefault(row["capture_group"], set()).add(split)
        if path.parent != output_dir / split:
            errors.append(f"path outside declared split: {path}")
        if not path.exists():
            errors.append(f"missing file: {path}")
            continue
        with Image.open(path) as image:
            if image.size != (image_size, image_size):
                errors.append(f"wrong resolution for {path}: {image.size}")
            if image.mode != "RGB":
                errors.append(f"wrong mode for {path}: {image.mode}")
        if _sha256(path) != row["sha256"]:
            errors.append(f"sha256 mismatch: {path}")
    for group, splits in group_splits.items():
        if len(splits) != 1:
            errors.append(f"capture group crosses splits: {group} -> {sorted(splits)}")
    disk_paths = {str(path) for split in SPLITS for path in (output_dir / split).glob("*.png")}
    if disk_paths != seen_prepared:
        errors.append("manifest PNG paths do not exactly match files on disk")
    result = {
        "passed": not errors,
        "manifest_rows": len(rows),
        "split_counts": split_counts,
        "unique_sources": len(seen_sources),
        "unique_prepared_files": len(seen_prepared),
        "capture_groups": len(group_splits),
        "errors": errors,
    }
    if errors:
        raise ValueError("v1_1024 validation failed: " + "; ".join(errors))
    return result


def prepare_v1_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    image_size: int = 1024,
    test_count: int = 20,
    val_count: int = 0,
    capture_group_size: int = 5,
    seed: int = 42,
    min_target_anchor_fraction: float = 0.01,
    target_crop_expansion: float = 2.9,
    prepared_staging: Path | None = None,
    force: bool = False,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_paths = _source_paths(source_dir)
    if not source_paths:
        raise ValueError(f"No source images found in {source_dir}")
    if test_count < 0 or val_count < 0:
        raise ValueError("Split counts cannot be negative")
    if capture_group_size < 1:
        raise ValueError("capture_group_size must be at least 1")

    existing_images = [path for split in SPLITS for path in (output_dir / split).glob("*.png")]
    if existing_images and not force:
        raise FileExistsError(f"Dataset already contains {len(existing_images)} PNG files; use --force to rebuild")
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_outputs(output_dir)

    temporary = None
    if prepared_staging is None:
        temporary = TemporaryDirectory(prefix="gan-arrow-v1-")
        staging_dir = Path(temporary.name)
        prepare_images(
            source_dir,
            staging_dir,
            image_size,
            min_target_anchor_fraction=min_target_anchor_fraction,
            target_crop_expansion=target_crop_expansion,
            raise_on_empty=True,
        )
    else:
        staging_dir = Path(prepared_staging)
    try:
        prepared_by_stem = {
            path.stem: path
            for path in prepared_image_paths(staging_dir)
            if image_has_signal(path)
            and image_meets_minimum_resolution(path, image_size)
            and image_has_target_anchor(path, min_fraction=min_target_anchor_fraction)
        }
        accepted_sources = [path for path in source_paths if path.stem in prepared_by_stem]
        rejected_sources = [path for path in source_paths if path.stem not in prepared_by_stem]
        if test_count + val_count >= len(accepted_sources):
            raise ValueError(
                f"Requested val+test={val_count + test_count}, but only {len(accepted_sources)} images were accepted"
            )

        groups = _capture_groups(accepted_sources, capture_group_size)
        rng = random.Random(seed)
        test_group_indices = _choose_groups(groups, test_count, rng)
        remaining_groups = [group for index, group in enumerate(groups) if index not in test_group_indices]
        val_indices_in_remaining = _choose_groups(remaining_groups, val_count, rng)

        assignment = {}
        group_ids = {}
        remaining_index = 0
        for group_index, group in enumerate(groups):
            if group_index in test_group_indices:
                split = "test"
            else:
                split = "val" if remaining_index in val_indices_in_remaining else "train"
                remaining_index += 1
            for source_path in group:
                assignment[source_path.stem] = split
                group_ids[source_path.stem] = f"capture_{group_index:03d}"

        rows = []
        split_counts = {split: 0 for split in SPLITS}
        for source_path in accepted_sources:
            split = assignment[source_path.stem]
            prepared_path = prepared_by_stem[source_path.stem]
            destination = output_dir / split / prepared_path.name
            shutil.copy2(prepared_path, destination)
            with Image.open(destination) as image:
                width, height = image.size
            rows.append(
                {
                    "source_path": str(source_path),
                    "prepared_path": str(destination),
                    "split": split,
                    "capture_group": group_ids[source_path.stem],
                    "width": width,
                    "height": height,
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
            split_counts[split] += 1

        with (output_dir / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_path",
                    "prepared_path",
                    "split",
                    "capture_group",
                    "width",
                    "height",
                    "bytes",
                    "sha256",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        staging_report_path = staging_dir / "preprocessing_report.json"
        staging_report = json.loads(staging_report_path.read_text()) if staging_report_path.exists() else {}
        candidate_staging = {
            "preprocess_version": staging_report.get("preprocess_version"),
            "source_count": staging_report.get("source_count"),
            "candidate_accepted_count": staging_report.get("accepted_count"),
            "candidate_rejected_count": staging_report.get("rejected_count"),
            "note": (
                "Candidate staging is provenance only, not the final acceptance result. "
                "Every staged image was revalidated with the current preprocess_version before splitting."
            ),
        }
        report = {
            "dataset_version": "v1_1024",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "source_count": len(source_paths),
            "accepted_count": len(accepted_sources),
            "rejected_count": len(rejected_sources),
            "rejected_sources": [str(path) for path in rejected_sources],
            "split_counts": split_counts,
            "split_unit": "contiguous filename capture groups",
            "capture_group_size": capture_group_size,
            "seed": seed,
            "image_size": image_size,
            "preprocess_version": PREPROCESS_VERSION,
            "random_offline_augmentation": False,
            "candidate_staging_preprocessing": candidate_staging,
        }
        (output_dir / "README.md").write_text(
            "# v1_1024\n\n"
            "Immutable real-image dataset for GAN training. Each accepted source contributes exactly one "
            "deterministically decoded, target-cropped, Lanczos-resized 1024 x 1024 PNG.\n\n"
            f"- train: {split_counts['train']}\n"
            f"- val: {split_counts['val']} (intentionally unused when zero; ADA does not require validation data)\n"
            f"- test: {split_counts['test']}\n"
            "- split unit: short contiguous filename capture groups; a group never crosses splits\n"
            "- random offline augmentation: disabled\n"
            "- test images: never used by the GAN training loader or ADA\n\n"
            "Evidence boundary: the current sources are continuous photographs of the same physical "
            "target and background. The test directory is an internal same-scene holdout, not an "
            "independent real-world test set. A future external test must use a new session/background.\n\n"
            "Training-time augmentation belongs in the discriminator path and is generated in memory. "
            "Do not add flipped, recolored, cutout, or Copy-Paste variants to this directory.\n"
        )
        report["validation"] = validate_v1_dataset(output_dir, image_size=image_size)
        (output_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        if temporary is not None:
            temporary.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare immutable real-image v1_1024 dataset")
    parser.add_argument("--source-dir", type=Path, default=Path("dataset/origin_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/v1_1024"))
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--capture-group-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-target-anchor-fraction", type=float, default=0.01)
    parser.add_argument("--target-crop-expansion", type=float, default=2.9)
    parser.add_argument("--prepared-staging", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = prepare_v1_dataset(
        args.source_dir,
        args.output_dir,
        image_size=args.image_size,
        test_count=args.test_count,
        val_count=args.val_count,
        capture_group_size=args.capture_group_size,
        seed=args.seed,
        min_target_anchor_fraction=args.min_target_anchor_fraction,
        target_crop_expansion=args.target_crop_expansion,
        prepared_staging=args.prepared_staging,
        force=args.force,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
