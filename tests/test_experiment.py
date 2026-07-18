import hashlib
import unittest
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from model.experiment import (
    SPEC_FIELDS,
    Experiment,
    ExperimentError,
    artifact_entry,
    atomic_write_yaml,
    build_training_config,
    compute_spec_sha256,
    load_experiment,
    new_document,
    register_artifacts,
    update_experiment,
    validate_spec,
)
from model.train_gan import TrainingConfig

FORMAL_SPEC = {
    "dataset_dir": "dataset/v1_1024/train",
    "processed_dir": "dataset/v1_1024/train",
    "image_size": 1024,
    "latent_dim": 100,
    "channels": 3,
    "batch_size": 1,
    "epochs": 3000,
    "d_lr": 0.0001,
    "g_lr": 0.0005,
    "beta1": 0.5,
    "real_label": 0.9,
    "generator_features": 48,
    "discriminator_features": 16,
    "augmentation_mode": "ada",
    "diffaugment": False,
    "diffaugment_policy": "color,translation,cutout",
    "ada_augpipe": "bgc",
    "ada_target": 0.6,
    "ada_interval": 4,
    "ada_kimg": 500.0,
    "ada_p_initial": 0.0,
    "ema_decay": 0.995,
    "amp": True,
    "grad_accum_steps": 16,
    "sample_interval": 50,
    "checkpoint_interval": 100,
    "early_stop_patience_evals": 0,
    "early_stop_min_epochs": 0,
    "early_stop_min_delta": 0.0,
    "skip_prepare": True,
    "max_steps": None,
    "min_target_anchor_fraction": 0.01,
    "target_crop_expansion": 2.9,
    "seed": 42,
    "workers": 0,
}


def write_experiment(root: Path, spec=None, **overrides):
    document = new_document("EXP-GAN-1024-031-ada", "formal", "ADA delays discriminator overfitting", spec or dict(FORMAL_SPEC))
    document.update(overrides)
    path = root / "experiment.yaml"
    atomic_write_yaml(path, document)
    return path


class ExperimentSpecTests(unittest.TestCase):
    def test_spec_fields_cover_every_training_config_field_except_runtime_paths(self):
        expected = {field.name for field in fields(TrainingConfig)} - {
            "output_dir",
            "resume_generator",
            "resume_discriminator",
            "resume_ema_generator",
            "resume_training_state",
        }
        self.assertEqual(set(SPEC_FIELDS), expected)
        self.assertEqual(set(FORMAL_SPEC), expected)

    def test_spec_missing_field_is_rejected_instead_of_inheriting_a_default(self):
        spec = dict(FORMAL_SPEC)
        del spec["generator_features"]
        with self.assertRaises(ExperimentError) as error:
            validate_spec(spec)
        self.assertIn("generator_features", str(error.exception))

    def test_spec_unknown_field_is_rejected(self):
        spec = dict(FORMAL_SPEC, mystery_knob=1)
        with self.assertRaises(ExperimentError) as error:
            validate_spec(spec)
        self.assertIn("mystery_knob", str(error.exception))

    def test_build_training_config_preserves_science_fields_without_silent_defaults(self):
        config = build_training_config(FORMAL_SPEC, Path("experiments/EXP-GAN-1024-031-ada"))
        self.assertEqual(config.generator_features, 48)
        self.assertEqual(config.discriminator_features, 16)
        self.assertEqual(config.grad_accum_steps, 16)
        self.assertEqual(config.ema_decay, 0.995)
        self.assertEqual(config.image_size, 1024)
        self.assertTrue(config.amp)
        self.assertEqual(config.augmentation_mode, "ada")
        self.assertEqual(config.ada_target, 0.6)
        self.assertEqual(config.ada_kimg, 500.0)
        self.assertEqual(config.processed_dir, Path("dataset/v1_1024/train"))
        self.assertEqual(config.output_dir, Path("experiments/EXP-GAN-1024-031-ada"))

    def test_build_training_config_differs_from_bare_defaults(self):
        defaults = TrainingConfig()
        config = build_training_config(FORMAL_SPEC, Path("experiments/EXP"))
        self.assertNotEqual(config.generator_features, defaults.generator_features)
        self.assertNotEqual(config.discriminator_features, defaults.discriminator_features)
        self.assertNotEqual(config.grad_accum_steps, defaults.grad_accum_steps)


class ExperimentDatasetGuardTests(unittest.TestCase):
    def test_preprocessing_into_the_prepared_dataset_is_refused(self):
        spec = dict(FORMAL_SPEC, skip_prepare=False)
        with self.assertRaises(ExperimentError) as error:
            validate_spec(spec)
        self.assertIn("skip_prepare", str(error.exception))

    def test_preprocessing_into_the_generate_data_cache_is_allowed(self):
        spec = dict(
            FORMAL_SPEC,
            skip_prepare=False,
            dataset_dir="dataset/origin_data",
            processed_dir="dataset/generate_data/processed_1024_front",
        )
        validate_spec(spec)

    def test_training_on_the_test_split_is_refused(self):
        spec = dict(FORMAL_SPEC, processed_dir="dataset/v1_1024/test")
        with self.assertRaises(ExperimentError) as error:
            validate_spec(spec)
        self.assertIn("test", str(error.exception))

    def test_training_on_the_val_split_is_refused(self):
        spec = dict(FORMAL_SPEC, dataset_dir="dataset/v1_1024/val")
        with self.assertRaises(ExperimentError):
            validate_spec(spec)


class ExperimentDocumentTests(unittest.TestCase):
    def test_round_trip_parses_every_section(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir))
            experiment = load_experiment(path)
            self.assertIsInstance(experiment, Experiment)
            self.assertEqual(experiment.id, "EXP-GAN-1024-031-ada")
            self.assertEqual(experiment.kind, "formal")
            self.assertEqual(experiment.run["status"], "planned")
            self.assertEqual(experiment.spec["generator_features"], 48)
            self.assertEqual(experiment.directory, Path(temp_dir))

    def test_unsupported_schema_version_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir), schema_version=99)
            with self.assertRaises(ExperimentError):
                load_experiment(path)

    def test_unknown_kind_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir), kind="exploratory")
            with self.assertRaises(ExperimentError):
                load_experiment(path)

    def test_unknown_run_status_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir))
            data = yaml.safe_load(path.read_text())
            data["run"]["status"] = "halfway"
            atomic_write_yaml(path, data)
            with self.assertRaises(ExperimentError):
                load_experiment(path)


class ExperimentImmutableSpecTests(unittest.TestCase):
    def test_spec_sha256_is_order_independent(self):
        reordered = {key: FORMAL_SPEC[key] for key in reversed(list(FORMAL_SPEC))}
        self.assertEqual(compute_spec_sha256(FORMAL_SPEC), compute_spec_sha256(reordered))

    def test_spec_sha256_changes_when_science_changes(self):
        mutated = dict(FORMAL_SPEC, generator_features=32)
        self.assertNotEqual(compute_spec_sha256(FORMAL_SPEC), compute_spec_sha256(mutated))

    def test_editing_spec_after_the_run_started_is_detected_on_load(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir))
            update_experiment(path, run={"status": "running", "spec_sha256": compute_spec_sha256(FORMAL_SPEC)})
            data = yaml.safe_load(path.read_text())
            data["spec"]["generator_features"] = 32
            atomic_write_yaml(path, data)
            with self.assertRaises(ExperimentError) as error:
                load_experiment(path)
            self.assertIn("modified", str(error.exception))

    def test_update_experiment_cannot_write_the_spec_section(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir))
            with self.assertRaises(ExperimentError):
                update_experiment(path, spec=dict(FORMAL_SPEC, epochs=1))

    def test_update_experiment_leaves_spec_untouched(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir))
            before = compute_spec_sha256(load_experiment(path).spec)
            update_experiment(path, run={"status": "running"}, result={"loss_g": 1.5})
            self.assertEqual(compute_spec_sha256(load_experiment(path).spec), before)


class ExperimentAtomicUpdateTests(unittest.TestCase):
    def test_update_merges_into_sections_without_dropping_other_keys(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir))
            update_experiment(path, run={"status": "running", "device": "mps"})
            update_experiment(path, run={"completed_epoch": 7})
            run = load_experiment(path).run
            self.assertEqual(run["status"], "running")
            self.assertEqual(run["device"], "mps")
            self.assertEqual(run["completed_epoch"], 7)

    def test_atomic_write_leaves_no_temporary_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_experiment(root)
            update_experiment(path, run={"status": "running"})
            self.assertEqual([item.name for item in root.iterdir()], ["experiment.yaml"])

    def test_failed_write_keeps_the_previous_document_intact(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_experiment(root)
            original = path.read_text()
            with self.assertRaises(Exception):
                atomic_write_yaml(path, {"run": {"status": object()}})
            self.assertEqual(path.read_text(), original)
            self.assertEqual([item.name for item in root.iterdir()], ["experiment.yaml"])

    def test_failure_status_records_the_error_for_recovery(self):
        with TemporaryDirectory() as temp_dir:
            path = write_experiment(Path(temp_dir))
            update_experiment(path, run={"status": "failed", "error": "MPS backward unsupported", "completed_epoch": 12})
            run = load_experiment(path).run
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["completed_epoch"], 12)
            self.assertIn("MPS", run["error"])


class ExperimentArtifactTests(unittest.TestCase):
    def test_artifact_entry_records_size_and_hash(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "inference.pt"
            target.write_bytes(b"arrow")
            entry = artifact_entry(target)
            self.assertEqual(entry["bytes"], 5)
            self.assertEqual(entry["sha256"], hashlib.sha256(b"arrow").hexdigest())

    def test_artifact_hash_matches_content_and_changes_with_content(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "inference.pt"
            target.write_bytes(b"arrow")
            first = artifact_entry(target)["sha256"]
            target.write_bytes(b"arrow2")
            self.assertNotEqual(artifact_entry(target)["sha256"], first)

    def test_register_artifacts_only_records_files_that_exist(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_experiment(root)
            (root / "inference.pt").write_bytes(b"weights")
            (root / "history.csv").write_text("epoch\n1\n")
            register_artifacts(path, root, ["inference.pt", "history.csv", "resume.pt"])
            artifacts = load_experiment(path).artifacts
            self.assertEqual(sorted(artifacts), ["history.csv", "inference.pt"])
            self.assertNotIn("resume.pt", artifacts)
            self.assertEqual(artifacts["inference.pt"]["bytes"], 7)


if __name__ == "__main__":
    unittest.main()
