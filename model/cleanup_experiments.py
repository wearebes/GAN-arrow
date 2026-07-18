import argparse
import hashlib
import json
from pathlib import Path

import torch

from model.experiment import EXPERIMENT_FILENAME, load_experiment

EXPERIMENTS_ROOT = Path("experiments")

# The retained inference checkpoint per formal experiment, and its paired discriminator.
# 001 early-stopped, so its best_* pair is a snapshot that exists nowhere else and must be kept.
# 004/005 disabled early stopping, so best == final == EMA and the final state carries everything.
RETENTION = {
    "EXP-GAN-1024-001-base": {"inference": "best_generator.pt", "discriminator": "best_discriminator.pt"},
    "EXP-GAN-1024-004-base-long100": {"inference": "final_generator.pt", "discriminator": "discriminator.pt"},
    "EXP-GAN-1024-005-g5e4-long500": {"inference": "final_generator.pt", "discriminator": "discriminator.pt"},
}
RESUME_SOURCE = "training_state.pt"
KEEP_ALWAYS = {EXPERIMENT_FILENAME, "history.csv"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def load_tensors(path: Path):
    return _state_dict(torch.load(path, map_location="cpu", weights_only=False))


def tensors_equal(left, right) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    keys = [key for key in left if hasattr(left[key], "shape")]
    if set(left.keys()) != set(right.keys()) or not keys:
        return False
    return all(torch.equal(left[key], right[key]) for key in keys)


def is_training_state(obj) -> bool:
    return isinstance(obj, dict) and "generator" in obj and "optimizer_g" in obj


def _values_equal(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    return left == right


def _optimizers_equal(left, right) -> bool:
    if left.get("param_groups") != right.get("param_groups"):
        return False
    left_state, right_state = left.get("state", {}), right.get("state", {})
    if set(left_state.keys()) != set(right_state.keys()):
        return False
    for key in left_state:
        entry_left, entry_right = left_state[key], right_state[key]
        if set(entry_left.keys()) != set(entry_right.keys()):
            return False
        if not all(_values_equal(entry_left[name], entry_right[name]) for name in entry_left):
            return False
    return True


def training_states_equivalent(left, right) -> bool:
    """A full training state is a nested dict, so compare each part rather than top-level tensors."""
    if set(left.keys()) != set(right.keys()):
        return False
    for key in ("generator", "discriminator", "ema_generator"):
        if key in left and not tensors_equal(left[key], right[key]):
            return False
    for key in ("completed_steps", "completed_epoch"):
        if left.get(key) != right.get(key):
            return False
    for key in ("optimizer_g", "optimizer_d"):
        if key in left and not _optimizers_equal(left[key], right[key]):
            return False
    return left.get("config") == right.get("config")


def recoverability_of(path: Path, retained: dict, resume_state=None):
    """Return a human-readable proof that deleting `path` loses nothing, or None if it is unique."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # a checkpoint we cannot read is never safe to delete
        return None, f"unreadable ({error})"
    if is_training_state(raw):
        if resume_state is not None and training_states_equivalent(raw, resume_state):
            return RESUME_SOURCE, f"equivalent to {RESUME_SOURCE} (weights, optimizers, counters, config)"
        return None, "UNIQUE - full training state differing from the retained resume"
    candidate = _state_dict(raw)
    for label, tensors in retained.items():
        if tensors_equal(candidate, tensors):
            return label, f"tensor-identical to {label}"
    return None, "UNIQUE - not reproducible from any retained artifact"


def retained_tensor_sources(directory: Path, rules: dict):
    sources = {}
    resume = directory / RESUME_SOURCE
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        for key in ("generator", "ema_generator", "discriminator"):
            if key in state:
                sources[f"resume.pt['{key}']"] = state[key]
    for role in ("inference", "discriminator"):
        name = rules.get(role)
        if name and (directory / name).exists():
            sources[f"{role} ({name})"] = load_tensors(directory / name)
    return sources


def plan_experiment(directory: Path):
    experiment = load_experiment(directory / EXPERIMENT_FILENAME)
    rules = RETENTION.get(experiment.id, {})
    actions = []

    renames = {}
    if rules:
        renames[rules["inference"]] = "inference.pt"
        renames[rules["discriminator"]] = "discriminator.pt"
        renames[RESUME_SOURCE] = "resume.pt"

    retained = retained_tensor_sources(directory, rules) if rules else {}
    resume_path = directory / RESUME_SOURCE
    resume_state = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path.exists() else None

    for path in sorted(directory.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        relative = path.relative_to(directory)
        name = str(relative)

        if name in KEEP_ALWAYS:
            actions.append(("KEEP", relative, path.stat().st_size, "retained by policy", None))
            continue
        if name in renames:
            actions.append(("RENAME", relative, path.stat().st_size, f"-> {renames[name]}", None))
            continue
        if path.suffix == ".pt":
            source, proof = recoverability_of(path, retained, resume_state)
            verb = "DELETE" if source else "REFUSE"
            actions.append((verb, relative, path.stat().st_size, proof, sha256_of(path)))
            continue
        if path.name == "contact_sheet.png":
            actions.append(("COMPRESS", relative, path.stat().st_size, "-> preview.jpg (2048 px)", None))
            continue
        if path.name.startswith("samples_epoch_"):
            # These are the milestone previews each REPORT cites as visual evidence. Migrated runs
            # predate tracking, so no remote copy exists and deletion would be unrecoverable.
            actions.append(("COMPRESS", relative, path.stat().st_size, "milestone evidence -> progress.jpg", None))
            continue
        actions.append(("DELETE", relative, path.stat().st_size, "superseded by experiment.yaml", sha256_of(path)))
    return experiment, actions


def main():
    parser = argparse.ArgumentParser(description="Plan the experiment artifact cleanup (dry-run by default)")
    parser.add_argument("--root", type=Path, default=EXPERIMENTS_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("cleanup_manifest.json"))
    args = parser.parse_args()

    manifest = {}
    totals = {}
    refused = 0
    for directory in sorted(item for item in args.root.iterdir() if item.is_dir()):
        if not (directory / EXPERIMENT_FILENAME).exists():
            continue
        experiment, actions = plan_experiment(directory)
        print(f"\n=== {experiment.id}  [{experiment.kind}/{experiment.run['status']}] ===")
        for verb, relative, size, note, digest in actions:
            totals[verb] = totals.get(verb, 0) + size
            refused += 1 if verb == "REFUSE" else 0
            print(f"  {verb:9s} {size/1e6:8.2f} MB  {str(relative):46s} {note}")
        manifest[experiment.id] = [
            {"action": verb, "path": str(relative), "bytes": size, "note": note, "sha256": digest}
            for verb, relative, size, note, digest in actions
        ]

    print("\n" + "=" * 78)
    for verb in sorted(totals):
        print(f"{verb:9s} {totals[verb]/1e6:9.2f} MB")
    print(f"\nREFUSED (unique, would lose data): {refused}")
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"manifest written to {args.manifest}")
    print("\nDry-run only. No file was modified.")


if __name__ == "__main__":
    main()
