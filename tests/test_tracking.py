import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from PIL import Image

from model.tracking import PREVIEW_MAX_PIXELS, Tracker, TrackingState, resolve_mode


class FakeRun:
    def __init__(self, run_id="run-abc", url="https://swanlab.cn/@jcy/gan-arrow/run-abc", mode="online"):
        self.id = run_id
        if url is not None:
            self.url = url
        self.mode = mode
        self.logged = []
        self.finished = None
        self.log_error = None

    def log(self, data, step=None):
        if self.log_error:
            raise self.log_error
        self.logged.append((data, step))

    def finish(self, state="success", error=None):
        self.finished = (state, error)


class FakeImage:
    def __init__(self, data_or_path, caption=None, file_type=None, size=None):
        self.path = data_or_path
        self.caption = caption
        self.file_type = file_type
        self.size = size


class FakeClient:
    """Stands in for the swanlab module so degradation paths are testable without a network."""

    def __init__(self, fail_modes=(), run=None):
        self.fail_modes = set(fail_modes)
        self.calls = []
        self.run = run or FakeRun()
        self.Image = FakeImage

    def init(self, **kwargs):
        self.calls.append(kwargs)
        mode = kwargs.get("mode")
        if mode in self.fail_modes:
            raise ConnectionError(f"network unreachable for mode={mode}")
        self.run.mode = mode
        return self.run


def start(client, **overrides):
    options = dict(experiment_id="EXP-GAN-1024-031-ada", project="gan-arrow", group="gan-1024", client=client)
    options.update(overrides)
    return Tracker.start(**options)


class TrackerStartTests(unittest.TestCase):
    def test_online_run_is_private_and_records_id_and_url(self):
        client = FakeClient()
        tracker = start(client)
        self.assertTrue(tracker.active)
        self.assertEqual(tracker.state.mode, "online")
        self.assertEqual(tracker.state.swanlab_id, "run-abc")
        self.assertEqual(tracker.state.swanlab_url, "https://swanlab.cn/@jcy/gan-arrow/run-abc")
        self.assertIsNone(tracker.state.degraded_reason)
        self.assertIs(client.calls[0]["public"], False)

    def test_run_config_and_group_are_sent(self):
        client = FakeClient()
        start(client, config={"generator_features": 48})
        self.assertEqual(client.calls[0]["config"], {"generator_features": 48})
        self.assertEqual(client.calls[0]["group"], "gan-1024")
        self.assertEqual(client.calls[0]["name"], "EXP-GAN-1024-031-ada")

    def test_disabled_mode_never_calls_init(self):
        client = FakeClient()
        tracker = start(client, mode="disabled")
        self.assertFalse(tracker.active)
        self.assertEqual(tracker.state.mode, "disabled")
        self.assertEqual(client.calls, [])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            start(FakeClient(), mode="cloudy")


class TrackerDegradationTests(unittest.TestCase):
    def test_offline_machine_degrades_from_online_to_local_and_keeps_training(self):
        client = FakeClient(fail_modes={"online"})
        tracker = start(client)
        self.assertTrue(tracker.active)
        self.assertEqual(tracker.state.mode, "local")
        self.assertEqual(tracker.state.requested_mode, "online")
        self.assertIn("network unreachable", tracker.state.degraded_reason)
        self.assertEqual([call["mode"] for call in client.calls], ["online", "local"])

    def test_total_swanlab_failure_disables_tracking_without_raising(self):
        client = FakeClient(fail_modes={"online", "local"})
        tracker = start(client)
        self.assertFalse(tracker.active)
        self.assertEqual(tracker.state.mode, "disabled")
        self.assertIn("local init failed", tracker.state.degraded_reason)

    def test_a_logging_failure_mid_run_degrades_instead_of_killing_the_run(self):
        client = FakeClient()
        tracker = start(client)
        client.run.log_error = ConnectionResetError("connection dropped")
        self.assertFalse(tracker.log({"epoch": 5, "loss_d": 0.3, "loss_g": 2.0}))
        self.assertFalse(tracker.active)
        self.assertEqual(tracker.state.mode, "disabled")
        self.assertIn("log failed", tracker.state.degraded_reason)
        # A later epoch must stay silent rather than raise into the training loop.
        self.assertFalse(tracker.log({"epoch": 6, "loss_d": 0.2, "loss_g": 2.1}))

    def test_local_run_whose_url_property_raises_still_tracks(self):
        # swanlab's Run.url raises ValueError outside online mode rather than returning None.
        class NoUrlRun(FakeRun):
            def __init__(self):
                super().__init__(url=None)

            @property
            def url(self):
                raise ValueError("Run url is unavailable because the current run is not using online mode")

        client = FakeClient(run=NoUrlRun())
        tracker = start(client, mode="local")
        self.assertTrue(tracker.active)
        self.assertEqual(tracker.state.mode, "local")
        self.assertIsNone(tracker.state.swanlab_url)
        self.assertEqual(tracker.state.swanlab_id, "run-abc")

    def test_missing_swanlab_package_disables_tracking(self):
        with mock.patch("model.tracking._import_swanlab", side_effect=ImportError("no swanlab")):
            tracker = Tracker.start(experiment_id="EXP", client=None)
        self.assertFalse(tracker.active)
        self.assertEqual(tracker.state.mode, "disabled")
        self.assertIn("import failed", tracker.state.degraded_reason)


class TrackerResumeTests(unittest.TestCase):
    def test_resuming_reuses_the_same_run_id_and_refuses_to_fork(self):
        client = FakeClient()
        start(client, run_id="run-abc")
        self.assertEqual(client.calls[0]["id"], "run-abc")
        self.assertEqual(client.calls[0]["resume"], "must")

    def test_a_fresh_run_does_not_request_resume(self):
        client = FakeClient()
        start(client)
        self.assertIsNone(client.calls[0]["id"])
        self.assertIsNone(client.calls[0]["resume"])

    def test_resumed_run_keeps_its_id_in_state(self):
        client = FakeClient(run=FakeRun(run_id="run-xyz"))
        tracker = start(client, run_id="run-xyz")
        self.assertEqual(tracker.state.swanlab_id, "run-xyz")


class TrackerLogTests(unittest.TestCase):
    def test_epoch_metrics_are_logged_at_the_epoch_step(self):
        client = FakeClient()
        tracker = start(client)
        self.assertTrue(tracker.log({"epoch": 7, "loss_d": 0.3, "loss_g": 2.5, "d_real": 0.84, "d_fake": 0.12}))
        data, step = client.run.logged[0]
        self.assertEqual(step, 7)
        self.assertEqual(data, {"loss_d": 0.3, "loss_g": 2.5, "d_real": 0.84, "d_fake": 0.12})

    def test_ada_metrics_are_logged_when_present(self):
        client = FakeClient()
        tracker = start(client)
        tracker.log({"epoch": 1, "loss_d": 0.5, "ada_p": 0.02, "ada_rt": 0.61})
        data, _ = client.run.logged[0]
        self.assertEqual(data["ada_p"], 0.02)
        self.assertEqual(data["ada_rt"], 0.61)

    def test_none_valued_metrics_are_dropped(self):
        client = FakeClient()
        tracker = start(client)
        tracker.log({"epoch": 1, "loss_d": 0.5, "ada_p": 0.0, "ada_rt": None})
        data, _ = client.run.logged[0]
        self.assertNotIn("ada_rt", data)
        self.assertIn("ada_p", data)

    def test_non_metric_fields_are_not_logged(self):
        client = FakeClient()
        tracker = start(client)
        tracker.log({"epoch": 1, "loss_d": 0.5, "sample_path": Path("/tmp/x.png"), "completed_steps": 101})
        data, _ = client.run.logged[0]
        self.assertEqual(set(data), {"loss_d"})

    def test_event_without_metrics_is_not_logged(self):
        client = FakeClient()
        tracker = start(client)
        self.assertFalse(tracker.log({"epoch": 1}))
        self.assertEqual(client.run.logged, [])


class TrackerPreviewTests(unittest.TestCase):
    def _preview(self, root: Path):
        path = root / "samples_epoch_010.png"
        Image.new("RGB", (256, 256), (10, 20, 30)).save(path)
        return path

    def test_preview_is_uploaded_compressed_as_jpg(self):
        with TemporaryDirectory() as temp_dir:
            client = FakeClient()
            tracker = start(client)
            path = self._preview(Path(temp_dir))
            self.assertTrue(tracker.log_preview(path, epoch=10))
            data, step = client.run.logged[0]
            self.assertEqual(step, 10)
            image = data["preview"]
            self.assertEqual(image.file_type, "jpg")
            self.assertEqual(image.size, PREVIEW_MAX_PIXELS)
            self.assertEqual(image.caption, "epoch 10")

    def test_missing_preview_is_skipped_quietly(self):
        with TemporaryDirectory() as temp_dir:
            tracker = start(FakeClient())
            self.assertFalse(tracker.log_preview(Path(temp_dir) / "absent.png", epoch=1))

    def test_preview_upload_failure_degrades_instead_of_raising(self):
        with TemporaryDirectory() as temp_dir:
            client = FakeClient()
            tracker = start(client)
            client.run.log_error = ConnectionError("upload failed")
            self.assertFalse(tracker.log_preview(self._preview(Path(temp_dir)), epoch=10))
            self.assertEqual(tracker.state.mode, "disabled")

    def test_disabled_tracker_ignores_previews_and_logs(self):
        with TemporaryDirectory() as temp_dir:
            tracker = start(FakeClient(), mode="disabled")
            self.assertFalse(tracker.log_preview(self._preview(Path(temp_dir)), epoch=1))
            self.assertFalse(tracker.log({"epoch": 1, "loss_d": 0.5}))


class TrackerFinishTests(unittest.TestCase):
    def test_success_is_reported(self):
        client = FakeClient()
        tracker = start(client)
        self.assertTrue(tracker.finish())
        self.assertEqual(client.run.finished, ("success", None))

    def test_crash_is_reported_with_the_error(self):
        client = FakeClient()
        tracker = start(client)
        tracker.finish(state="crashed", error="MPS backward unsupported")
        self.assertEqual(client.run.finished, ("crashed", "MPS backward unsupported"))

    def test_finishing_a_disabled_tracker_is_a_no_op(self):
        tracker = start(FakeClient(), mode="disabled")
        self.assertFalse(tracker.finish())


class TrackingStateTests(unittest.TestCase):
    def test_state_serialises_into_the_yaml_tracking_section(self):
        state = TrackingState(mode="local", swanlab_id="run-abc", requested_mode="online", degraded_reason="offline")
        section = state.to_yaml(project="gan-arrow", group="gan-1024")
        self.assertEqual(section["mode"], "local")
        self.assertEqual(section["project"], "gan-arrow")
        self.assertEqual(section["group"], "gan-1024")
        self.assertEqual(section["swanlab_id"], "run-abc")
        self.assertEqual(section["degraded_reason"], "offline")


class ResolveModeTests(unittest.TestCase):
    def test_environment_override_wins(self):
        with mock.patch.dict("os.environ", {"SWANLAB_MODE": "disabled"}):
            self.assertEqual(resolve_mode("online"), "disabled")

    def test_requested_mode_is_used_without_an_override(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_mode("local"), "local")

    def test_default_is_online(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_mode(None), "online")

    def test_invalid_override_is_ignored(self):
        with mock.patch.dict("os.environ", {"SWANLAB_MODE": "banana"}):
            self.assertEqual(resolve_mode("online"), "online")


if __name__ == "__main__":
    unittest.main()
