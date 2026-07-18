import argparse
import dataclasses
import sys
import time
import traceback
from pathlib import Path

from model.experiment import (
    EXPERIMENT_FILENAME,
    ExperimentError,
    build_training_config,
    compute_spec_sha256,
    load_experiment,
    update_experiment,
)
from model.tracking import Tracker, resolve_mode
from model.train_gan import (
    _git_output,
    count_images,
    prepared_image_paths,
    select_device,
    train,
)

RESUME_FILENAME = "resume.pt"
RESUMABLE_STATUSES = {"failed", "running"}


class RunError(Exception):
    pass


def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def check(experiment, config, resume: bool):
    """Validate everything cheap before spending hours on training."""
    problems = []
    directory = experiment.directory

    if not config.processed_dir.exists():
        problems.append(f"processed_dir does not exist: {config.processed_dir}")
    elif not prepared_image_paths(config.processed_dir):
        problems.append(f"no prepared images in {config.processed_dir}")
    if not config.skip_prepare and not config.dataset_dir.exists():
        problems.append(f"dataset_dir does not exist: {config.dataset_dir}")

    status = experiment.run.get("status")
    if resume:
        if not (directory / RESUME_FILENAME).exists():
            problems.append(f"--resume needs {directory / RESUME_FILENAME}, which does not exist")
        if status == "completed":
            problems.append("run.status is completed; refusing to resume a finished experiment")
    else:
        if status == "completed":
            problems.append("run.status is completed; use a new experiment id rather than overwriting it")
        if status == "running":
            problems.append("run.status is running; another process may own this directory")
        if (directory / RESUME_FILENAME).exists():
            problems.append(f"{RESUME_FILENAME} already exists; pass --resume or start a new experiment")

    recorded = experiment.run.get("spec_sha256")
    if recorded and recorded != compute_spec_sha256(experiment.spec):
        problems.append("spec no longer matches the recorded spec_sha256")
    return problems


def make_epoch_callback(tracker, experiment_path: Path, sample_interval: int):
    def on_epoch_end(event):
        tracker.log(event)
        sample_path = event.get("sample_path")
        if sample_path is not None:
            uploaded = tracker.log_preview(sample_path, epoch=event["epoch"])
            # Only drop the local copy once the remote actually has it, otherwise a degraded
            # tracker would silently destroy the only preview of this epoch.
            if uploaded:
                Path(sample_path).unlink(missing_ok=True)
        update_experiment(
            experiment_path,
            run={"completed_epoch": event["epoch"], "completed_steps": event.get("completed_steps")},
        )

    return on_epoch_end


def run(path: Path, resume: bool = False, mode: str | None = None, dry_check: bool = False):
    experiment = load_experiment(path)
    config = build_training_config(experiment.spec, experiment.directory)

    problems = check(experiment, config, resume)
    if dry_check:
        for problem in problems:
            print(f"  FAIL  {problem}")
        if not problems:
            print(f"  OK    {experiment.id} [{experiment.kind}] is ready to run")
            print(f"        device={select_device()} images={len(prepared_image_paths(config.processed_dir))}")
            print(f"        {config.image_size}px G{config.generator_features}/D{config.discriminator_features} "
                  f"epochs={config.epochs} aug={config.augmentation_mode}")
        return 0 if not problems else 1
    if problems:
        raise RunError("; ".join(problems))

    if resume:
        config = dataclasses.replace(config, resume_training_state=experiment.directory / RESUME_FILENAME)

    resolved_mode = resolve_mode(mode or experiment.tracking.get("mode"))
    tracker = Tracker.start(
        experiment_id=experiment.id,
        project=experiment.tracking.get("project") or "gan-arrow",
        group=experiment.tracking.get("group"),
        config=dict(experiment.spec),
        mode=resolved_mode,
        run_id=experiment.tracking.get("swanlab_id") if resume else None,
        description=experiment.hypothesis,
    )
    update_experiment(
        path,
        tracking=tracker.state.to_yaml(
            project=experiment.tracking.get("project") or "gan-arrow",
            group=experiment.tracking.get("group"),
        ),
        run={
            "status": "running",
            "started_at": _timestamp(),
            "device": str(select_device()),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "spec_sha256": compute_spec_sha256(experiment.spec),
            "error": None,
        },
    )
    if tracker.state.degraded_reason:
        print(f"tracking degraded to {tracker.state.mode}: {tracker.state.degraded_reason}", file=sys.stderr)

    try:
        metrics = train(config, on_epoch_end=make_epoch_callback(tracker, path, config.sample_interval))
    except BaseException as error:
        update_experiment(
            path,
            run={"status": "failed", "finished_at": _timestamp(), "error": f"{type(error).__name__}: {error}"},
            tracking=tracker.state.to_yaml(
                project=experiment.tracking.get("project") or "gan-arrow",
                group=experiment.tracking.get("group"),
            ),
        )
        tracker.finish(state="crashed", error=f"{type(error).__name__}: {error}")
        raise

    final = (metrics or {}).get("final", {})
    update_experiment(
        path,
        run={
            "status": "completed",
            "finished_at": _timestamp(),
            "completed_epoch": int((metrics or {}).get("completed_epochs") or 0),
            "completed_steps": int((metrics or {}).get("completed_steps") or 0),
            "error": None,
        },
        result={
            "loss_d": final.get("loss_d"),
            "loss_g": final.get("loss_g"),
            "d_real": final.get("d_real"),
            "d_fake": final.get("d_fake"),
        },
        tracking=tracker.state.to_yaml(
            project=experiment.tracking.get("project") or "gan-arrow",
            group=experiment.tracking.get("group"),
        ),
    )
    tracker.finish(state="success")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run one experiment from its experiment.yaml")
    parser.add_argument("experiment", type=Path, help=f"path to an {EXPERIMENT_FILENAME}")
    parser.add_argument("--resume", action="store_true", help=f"continue from {RESUME_FILENAME} and the recorded SwanLab run")
    parser.add_argument("--check", action="store_true", help="validate only; do not train")
    parser.add_argument("--mode", choices=("online", "local", "offline", "disabled"), help="override tracking mode")
    args = parser.parse_args()

    path = args.experiment
    if path.is_dir():
        path = path / EXPERIMENT_FILENAME
    try:
        return run(path, resume=args.resume, mode=args.mode, dry_check=args.check)
    except (ExperimentError, RunError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
