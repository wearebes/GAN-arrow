import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from model.train_gan import PATH_CONFIG_FIELDS, TrainingConfig

SCHEMA_VERSION = 1
EXPERIMENT_FILENAME = "experiment.yaml"
EXPERIMENT_KINDS = {"smoke", "probe", "formal"}
RUN_STATUSES = {"planned", "running", "completed", "failed"}
TOP_LEVEL_SECTIONS = ("tracking", "run", "result", "artifacts")

RUNTIME_SPEC_FIELDS = {
    "output_dir",
    "resume_generator",
    "resume_discriminator",
    "resume_ema_generator",
    "resume_training_state",
}
SPEC_FIELDS = tuple(field.name for field in fields(TrainingConfig) if field.name not in RUNTIME_SPEC_FIELDS)
SPEC_PATH_FIELDS = tuple(field for field in SPEC_FIELDS if field in PATH_CONFIG_FIELDS)

DATASET_ROOT = Path("dataset")
DATASET_CACHE_ROOT = Path("dataset/generate_data")
HOLDOUT_SPLIT_NAMES = {"test", "val"}


class ExperimentError(Exception):
    pass


@dataclass(frozen=True)
class Experiment:
    path: Path
    id: str
    kind: str
    hypothesis: str
    spec: dict
    tracking: dict
    run: dict
    result: dict
    artifacts: dict

    @property
    def directory(self) -> Path:
        return self.path.parent


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def canonical_spec(spec: dict) -> dict:
    return {key: (str(spec[key]) if isinstance(spec[key], Path) else spec[key]) for key in sorted(spec)}


def compute_spec_sha256(spec: dict) -> str:
    payload = json.dumps(canonical_spec(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_spec(spec: dict):
    if not isinstance(spec, dict):
        raise ExperimentError("spec must be a mapping")
    unknown = sorted(set(spec) - set(SPEC_FIELDS))
    if unknown:
        raise ExperimentError(f"spec has unknown fields: {', '.join(unknown)}")
    missing = sorted(set(SPEC_FIELDS) - set(spec))
    if missing:
        raise ExperimentError(
            f"spec is missing fields: {', '.join(missing)}. "
            "Every TrainingConfig field must be pinned so the run cannot inherit a silent default."
        )
    validate_dataset_paths(spec)


def validate_dataset_paths(spec: dict):
    processed_dir = Path(spec["processed_dir"])
    if not spec.get("skip_prepare", False):
        if _is_relative_to(processed_dir, DATASET_ROOT) and not _is_relative_to(processed_dir, DATASET_CACHE_ROOT):
            raise ExperimentError(
                f"processed_dir {processed_dir} is inside {DATASET_ROOT} but skip_prepare is false; "
                "preprocessing would rewrite and unlink files in the dataset. "
                "Set skip_prepare: true for an already prepared dataset."
            )
    for field in ("dataset_dir", "processed_dir"):
        parts = set(Path(spec[field]).parts)
        holdout = parts & HOLDOUT_SPLIT_NAMES
        if holdout:
            raise ExperimentError(f"{field} points at the {sorted(holdout)[0]} split; training must only read train")


def build_training_config(spec: dict, output_dir: Path) -> TrainingConfig:
    validate_spec(spec)
    values = dict(spec)
    for field in SPEC_PATH_FIELDS:
        values[field] = Path(values[field])
    return TrainingConfig(output_dir=Path(output_dir), **values)


def _validate_document(data: dict, path: Path):
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ExperimentError(f"{path}: unsupported schema_version {data.get('schema_version')!r}")
    if data.get("kind") not in EXPERIMENT_KINDS:
        raise ExperimentError(f"{path}: kind must be one of {sorted(EXPERIMENT_KINDS)}")
    if not data.get("id"):
        raise ExperimentError(f"{path}: id is required")
    status = (data.get("run") or {}).get("status")
    if status not in RUN_STATUSES:
        raise ExperimentError(f"{path}: run.status must be one of {sorted(RUN_STATUSES)}")


def load_experiment(path: Path) -> Experiment:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ExperimentError(f"{path}: experiment file must contain a mapping")
    _validate_document(data, path)
    spec = data.get("spec") or {}
    validate_spec(spec)
    recorded = (data.get("run") or {}).get("spec_sha256")
    if recorded is not None and recorded != compute_spec_sha256(spec):
        raise ExperimentError(
            f"{path}: spec has been modified after the run started "
            f"(recorded {recorded[:12]}, actual {compute_spec_sha256(spec)[:12]})"
        )
    return Experiment(
        path=path,
        id=data["id"],
        kind=data["kind"],
        hypothesis=data.get("hypothesis", ""),
        spec=spec,
        tracking=data.get("tracking") or {},
        run=data.get("run") or {},
        result=data.get("result") or {},
        artifacts=data.get("artifacts") or {},
    )


def atomic_write_yaml(path: Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, encoding="utf-8"
    )
    try:
        with handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True, default_flow_style=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def update_experiment(path: Path, **sections):
    unknown = sorted(set(sections) - set(TOP_LEVEL_SECTIONS))
    if unknown:
        raise ExperimentError(f"update_experiment cannot write sections: {', '.join(unknown)}")
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    spec_before = compute_spec_sha256(data.get("spec") or {})
    for section, values in sections.items():
        current = data.get(section) or {}
        current.update(values)
        data[section] = current
    if compute_spec_sha256(data.get("spec") or {}) != spec_before:
        raise ExperimentError("spec is immutable once written")
    atomic_write_yaml(path, data)
    return data


def artifact_entry(path: Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def register_artifacts(path: Path, directory: Path, names):
    directory = Path(directory)
    artifacts = {}
    for name in names:
        candidate = directory / name
        if candidate.exists():
            artifacts[name] = artifact_entry(candidate)
    update_experiment(path, artifacts=artifacts)
    return artifacts


def new_document(experiment_id: str, kind: str, hypothesis: str, spec: dict, tracking: dict | None = None) -> dict:
    validate_spec(spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": experiment_id,
        "kind": kind,
        "hypothesis": hypothesis,
        "spec": canonical_spec(spec),
        "tracking": tracking
        or {"mode": "online", "project": "gan-arrow", "group": None, "swanlab_id": None, "swanlab_url": None},
        "run": {
            "status": "planned",
            "started_at": None,
            "finished_at": None,
            "completed_epoch": 0,
            "device": None,
            "git_commit": None,
            "spec_sha256": None,
            "error": None,
        },
        "result": {
            "loss_d": None,
            "loss_g": None,
            "d_real": None,
            "d_fake": None,
            "inference_source": None,
            "judgment": None,
        },
        "artifacts": {},
    }
