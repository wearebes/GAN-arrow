import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from model.cleanup_experiments import (
    is_training_state,
    recoverability_of,
    sha256_of,
    tensors_equal,
    training_states_equivalent,
)


def make_state_dict(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return {
        "blocks.0.weight": torch.randn(4, 4, generator=generator),
        "blocks.0.bias": torch.randn(4, generator=generator),
    }


def make_training_state(seed: int = 1, completed_epoch: int = 100):
    return {
        "generator": make_state_dict(seed),
        "discriminator": make_state_dict(seed + 1),
        "ema_generator": make_state_dict(seed + 2),
        "optimizer_g": {"param_groups": [{"lr": 0.0005}], "state": {0: {"step": 10, "exp_avg": torch.zeros(4)}}},
        "optimizer_d": {"param_groups": [{"lr": 0.0001}], "state": {0: {"step": 10, "exp_avg": torch.zeros(4)}}},
        "config": {"image_size": 1024, "generator_features": 48},
        "completed_steps": completed_epoch * 101,
        "completed_epoch": completed_epoch,
    }


class TensorComparisonTests(unittest.TestCase):
    def test_identical_state_dicts_compare_equal(self):
        self.assertTrue(tensors_equal(make_state_dict(1), make_state_dict(1)))

    def test_different_weights_do_not_compare_equal(self):
        self.assertFalse(tensors_equal(make_state_dict(1), make_state_dict(2)))

    def test_mismatched_keys_do_not_compare_equal(self):
        left = make_state_dict(1)
        right = make_state_dict(1)
        del right["blocks.0.bias"]
        self.assertFalse(tensors_equal(left, right))

    def test_a_nested_training_state_is_never_treated_as_a_bare_state_dict(self):
        state = make_training_state()
        self.assertTrue(is_training_state(state))
        self.assertFalse(tensors_equal(state, copy.deepcopy(state)))


class TrainingStateEquivalenceTests(unittest.TestCase):
    def test_equivalent_states_are_detected(self):
        state = make_training_state()
        self.assertTrue(training_states_equivalent(state, copy.deepcopy(state)))

    def test_differing_weights_break_equivalence(self):
        left = make_training_state()
        right = copy.deepcopy(left)
        right["generator"]["blocks.0.weight"] += 1.0
        self.assertFalse(training_states_equivalent(left, right))

    def test_differing_epoch_breaks_equivalence(self):
        self.assertFalse(training_states_equivalent(make_training_state(completed_epoch=100), make_training_state(completed_epoch=90)))

    def test_differing_optimizer_moment_breaks_equivalence(self):
        left = make_training_state()
        right = copy.deepcopy(left)
        right["optimizer_g"]["state"][0]["exp_avg"] += 1.0
        self.assertFalse(training_states_equivalent(left, right))

    def test_differing_optimizer_lr_breaks_equivalence(self):
        left = make_training_state()
        right = copy.deepcopy(left)
        right["optimizer_d"]["param_groups"][0]["lr"] = 0.5
        self.assertFalse(training_states_equivalent(left, right))

    def test_differing_config_breaks_equivalence(self):
        left = make_training_state()
        right = copy.deepcopy(left)
        right["config"]["generator_features"] = 32
        self.assertFalse(training_states_equivalent(left, right))


class RecoverabilityTests(unittest.TestCase):
    def test_duplicate_checkpoint_is_reported_recoverable(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "generator_ema.pt"
            torch.save(make_state_dict(1), target)
            source, proof = recoverability_of(target, {"resume.pt['ema_generator']": make_state_dict(1)})
            self.assertEqual(source, "resume.pt['ema_generator']")
            self.assertIn("tensor-identical", proof)

    def test_unique_checkpoint_is_refused(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "best_generator.pt"
            torch.save(make_state_dict(99), target)
            source, proof = recoverability_of(target, {"resume.pt['ema_generator']": make_state_dict(1)})
            self.assertIsNone(source)
            self.assertIn("UNIQUE", proof)

    def test_wrapped_checkpoint_is_unwrapped_before_comparison(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "best_generator.pt"
            torch.save({"state_dict": make_state_dict(1), "config": {}, "uses_ema": True}, target)
            source, _ = recoverability_of(target, {"resume.pt['ema_generator']": make_state_dict(1)})
            self.assertEqual(source, "resume.pt['ema_generator']")

    def test_equivalent_training_state_is_recoverable_from_the_retained_resume(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "training_state_latest.pt"
            state = make_training_state()
            torch.save(state, target)
            source, proof = recoverability_of(target, {}, resume_state=copy.deepcopy(state))
            self.assertEqual(source, "training_state.pt")
            self.assertIn("optimizers", proof)

    def test_training_state_at_a_different_epoch_is_refused(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "training_state_epoch_050.pt"
            torch.save(make_training_state(completed_epoch=50), target)
            source, proof = recoverability_of(target, {}, resume_state=make_training_state(completed_epoch=100))
            self.assertIsNone(source)
            self.assertIn("UNIQUE", proof)

    def test_unreadable_checkpoint_is_never_deleted(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "corrupt.pt"
            target.write_bytes(b"not a checkpoint")
            source, proof = recoverability_of(target, {"resume.pt['generator']": make_state_dict(1)})
            self.assertIsNone(source)
            self.assertIn("unreadable", proof)


class ManifestHashTests(unittest.TestCase):
    def test_sha256_matches_hashlib_and_tracks_content(self):
        import hashlib

        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "artifact.bin"
            target.write_bytes(b"arrow")
            self.assertEqual(sha256_of(target), hashlib.sha256(b"arrow").hexdigest())
            target.write_bytes(b"arrow2")
            self.assertEqual(sha256_of(target), hashlib.sha256(b"arrow2").hexdigest())


if __name__ == "__main__":
    unittest.main()
