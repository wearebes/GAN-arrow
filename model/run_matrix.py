import argparse
import csv
from pathlib import Path

import yaml

from model.train_gan import TrainingConfig, count_images, train


def run_matrix(matrix_path: Path, dataset_dir: Path, processed_dir: Path, output_dir: Path):
    spec = yaml.safe_load(matrix_path.read_text())
    default_epochs = spec.get("epochs", 8)
    image_size = spec.get("image_size", 256)
    spec_processed_dir = Path(spec.get("processed_dir", processed_dir))
    default_overrides = dict(spec.get("defaults", {}))
    dataset_size = count_images(dataset_dir)

    rows = []
    for experiment in spec["experiments"]:
        experiment = {**default_overrides, **dict(experiment)}
        name = experiment.pop("name")
        epochs = experiment.pop("epochs", default_epochs)
        config = TrainingConfig.from_dataset_size(
            dataset_size,
            dataset_dir=dataset_dir,
            processed_dir=spec_processed_dir,
            output_dir=output_dir / name,
            epochs=epochs,
            image_size=image_size,
            **experiment,
        )
        metrics = train(config)
        rows.append({"name": name, **metrics["final"], **metrics.get("diagnostics", {})})
        print(f"[{name}] done -> {metrics['artifacts']['run_dir']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary written to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run a matrix of GAN training configs from a YAML file.")
    parser.add_argument("--matrix", type=Path, default=Path("configs/matrix.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/origin_data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("dataset/generate_data/processed_256"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/matrix"))
    args = parser.parse_args()

    run_matrix(
        matrix_path=args.matrix,
        dataset_dir=args.dataset_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
