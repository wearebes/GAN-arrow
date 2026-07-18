import argparse
import json
from dataclasses import fields
from pathlib import Path

from model.experiment import (
    EXPERIMENT_FILENAME,
    SPEC_FIELDS,
    atomic_write_yaml,
    compute_spec_sha256,
    load_experiment,
    new_document,
)
from model.train_gan import TrainingConfig

EXPERIMENTS_ROOT = Path("experiments")

# kind, hypothesis, judgment. Judgments are the conclusions recorded in each REPORT.md, which this
# migration folds into the single YAML so the directory can drop the separate report file.
CATALOG = {
    "EXP-GAN-1024-000-local-smoke": (
        "smoke",
        "One real CPU forward/backward step through the exact 1024 px G48/D16 pipeline.",
        "Pipeline smoke only; ran 25.17 s on CPU. Not a model-quality result.",
    ),
    "EXP-GAN-1024-001-base": (
        "formal",
        "A conservative baseline should establish whether the compact model can learn target "
        "structure without immediate instability.",
        "Early stop at epoch 20. Numerically finite but the discriminator leads the generator; "
        "no recognizable target or arrow shape. Final discriminator balanced accuracy 98.02%.",
    ),
    "EXP-GAN-1024-002-midlr": (
        "formal",
        "Stronger generator updates may form fine arrow structure sooner without accelerating "
        "the discriminator.",
        None,
    ),
    "EXP-GAN-1024-003-highlr": (
        "formal",
        "Five-times-higher optimizer rates may accelerate learning, but carry a high risk of "
        "oscillation or collapse.",
        None,
    ),
    "EXP-GAN-1024-004-base-long100": (
        "formal",
        "Shapes may emerge with substantially longer training; resumes 001 from epoch 20 to "
        "total epoch 100.",
        "Longer training replaced mosaics with a coarse target/stand silhouette, but images stay "
        "blurry with no scoring rings or arrow shafts. Discriminator balanced accuracy 99.50%; "
        "generator struggling. Does not pass the downstream-data quality requirement.",
    ),
    "EXP-GAN-1024-005-g5e4-long500": (
        "formal",
        "Increasing only the generator learning rate to 5e-4 may improve target and arrow detail.",
        "Improves coarse target recognition over the 100-epoch baseline but does not meet the "
        "arrow/detail requirement. D separation widens and G loss rises near the end; further "
        "continuation with this configuration is not justified. Do not use this generator to "
        "create downstream ring-scoring training data.",
    ),
    "EXP-GAN-1024-006-g5e4-aug-v1-100": (
        "formal",
        "A separate offline augmentation dataset (500 images x 100 epochs) may improve detail "
        "at equal image exposures versus the 005 control.",
        None,
    ),
    "EXP-GAN-1024-010-lrprobe-base": (
        "probe",
        "Four-step local learning-rate probe at G 2e-4 / D 1e-4 with EMA disabled.",
        "Probe only; ran 32.28 s on CPU.",
    ),
    "EXP-GAN-1024-011-lrprobe-mid": (
        "probe",
        "Four-step local learning-rate probe at the mid learning rate with EMA disabled.",
        "Probe only.",
    ),
    "EXP-GAN-1024-012-lrprobe-high": (
        "probe",
        "Four-step local learning-rate probe at the high learning rate with EMA disabled.",
        "Probe only.",
    ),
    "EXP-GAN-1024-021-mps-throughput": (
        "probe",
        "Measure sustained 1024 px Apple MPS throughput over eight batches.",
        "Device throughput only.",
    ),
    "EXP-GAN-1024-030-ada-v1": (
        "formal",
        "Paper-default StyleGAN2-ADA augmentation may delay discriminator overfitting on the "
        "small dataset while keeping the G48/D16 architecture and 005 learning rates unchanged.",
        None,
    ),
}

# Which checkpoint becomes inference.pt, and what it actually is. Verified by tensor comparison:
# 001's best is an early-stop EMA snapshot that exists nowhere else, while 004/005 best == final == EMA.
INFERENCE_SOURCE = {
    "EXP-GAN-1024-001-base": ("best_generator.pt", "best_ema_snapshot (early stop, unique; not reproducible from resume.pt)"),
    "EXP-GAN-1024-004-base-long100": ("final_generator.pt", "final_ema_epoch_100 (best/final/EMA are identical; not an independently validated best)"),
    "EXP-GAN-1024-005-g5e4-long500": ("final_generator.pt", "final_ema_epoch_500 (best/final/EMA are identical; not an independently validated best)"),
}


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def build_spec(recorded_config: dict) -> dict:
    defaults = {field.name: field.default for field in fields(TrainingConfig)}
    spec = {}
    for name in SPEC_FIELDS:
        if name in recorded_config:
            value = recorded_config[name]
        else:
            value = defaults[name]
        spec[name] = str(value) if isinstance(value, Path) else value
    # Pre-ADA runs recorded augmentation only through the legacy diffaugment flag. Pin the mode the
    # trainer actually resolves so the YAML cannot be misread as "no augmentation".
    if spec["augmentation_mode"] == "none" and spec["diffaugment"]:
        spec["augmentation_mode"] = "diffaugment"
    return spec


def migrate_experiment(directory: Path, force: bool = False):
    experiment_id = directory.name
    if experiment_id not in CATALOG:
        return None, f"skipped (not in catalog)"
    target = directory / EXPERIMENT_FILENAME
    if target.exists() and not force:
        return None, "exists (use --force to rewrite)"

    kind, hypothesis, judgment = CATALOG[experiment_id]
    config = _read_json(directory / "config.json")
    metrics = _read_json(directory / "metrics.json")
    environment = _read_json(directory / "environment.json")
    if config is None and metrics is not None:
        config = metrics.get("config")
    if config is None:
        return None, "no config.json or metrics.json to migrate"

    spec = build_spec(config)
    document = new_document(experiment_id, kind, hypothesis, spec)

    group = "gan-1024"
    document["tracking"] = {
        "mode": "offline",
        "project": "gan-arrow",
        "group": group,
        "swanlab_id": None,
        "swanlab_url": None,
        "note": "migrated run; predates tracking integration",
    }

    if metrics is not None:
        final = metrics.get("final") or {}
        document["run"] = {
            "status": "completed",
            "started_at": None,
            "finished_at": (environment or {}).get("created_at"),
            "completed_epoch": int(metrics.get("completed_epochs") or 0),
            "completed_steps": int(metrics.get("completed_steps") or 0),
            "device": metrics.get("device") or (environment or {}).get("device"),
            "git_commit": (environment or {}).get("git_commit"),
            "spec_sha256": compute_spec_sha256(spec),
            "error": None,
        }
        source_name, source_label = INFERENCE_SOURCE.get(experiment_id, (None, None))
        document["result"] = {
            "loss_d": final.get("loss_d"),
            "loss_g": final.get("loss_g"),
            "d_real": final.get("d_real"),
            "d_fake": final.get("d_fake"),
            "inference_source": source_label,
            "judgment": judgment,
        }
    else:
        document["run"] = {
            "status": "planned",
            "started_at": None,
            "finished_at": None,
            "completed_epoch": 0,
            "completed_steps": 0,
            "device": None,
            "git_commit": None,
            "spec_sha256": None,
            "error": None,
        }
        document["result"] = {
            "loss_d": None,
            "loss_g": None,
            "d_real": None,
            "d_fake": None,
            "inference_source": None,
            "judgment": judgment,
        }

    atomic_write_yaml(target, document)
    load_experiment(target)
    return target, f"{kind}/{document['run']['status']}"


def main():
    parser = argparse.ArgumentParser(description="Generate one experiment.yaml per experiment directory")
    parser.add_argument("--root", type=Path, default=EXPERIMENTS_ROOT)
    parser.add_argument("--force", action="store_true", help="rewrite an existing experiment.yaml")
    args = parser.parse_args()

    directories = sorted(item for item in args.root.iterdir() if item.is_dir())
    written = 0
    for directory in directories:
        target, note = migrate_experiment(directory, force=args.force)
        status = "wrote" if target else "----"
        print(f"{status}  {directory.name:38s}  {note}")
        written += 1 if target else 0
    print(f"\n{written} experiment.yaml written, {len(directories) - written} skipped")


if __name__ == "__main__":
    main()
