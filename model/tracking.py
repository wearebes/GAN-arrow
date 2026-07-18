"""SwanLab tracking for GAN runs.

Training is the expensive, irreplaceable part of a run; tracking is not. Every entry point here is
therefore failure-tolerant: if SwanLab cannot start, cannot log, or cannot upload a preview, the run
degrades to a cheaper mode and keeps training, and the reason is recorded so the YAML can say why a
run has no remote history.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

TRACKING_MODES = ("online", "local", "offline", "disabled")
# Ordered fallbacks: a private online run is preferred, a local run still keeps curves, and a
# disabled run still trains.
FALLBACK_CHAIN = {"online": ("local", "disabled"), "local": ("disabled",), "offline": ("disabled",)}
PREVIEW_MAX_PIXELS = 2048
SCALAR_KEYS = (
    "loss_d",
    "loss_d_total",
    "loss_d_real",
    "loss_d_fake",
    "loss_g",
    "d_real",
    "d_fake",
    "ada_p",
    "ada_rt",
)


@dataclass
class TrackingState:
    mode: str
    swanlab_id: str | None = None
    swanlab_url: str | None = None
    requested_mode: str | None = None
    degraded_reason: str | None = None

    def to_yaml(self, project: str, group: str | None) -> dict:
        return {
            "mode": self.mode,
            "project": project,
            "group": group,
            "swanlab_id": self.swanlab_id,
            "swanlab_url": self.swanlab_url,
            "degraded_reason": self.degraded_reason,
        }


def _import_swanlab():
    import swanlab

    return swanlab


def _safe_attr(obj, name):
    """Run.url is a property that raises outside online mode, so getattr's default is not enough."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


class Tracker:
    def __init__(self, run, state: TrackingState, client=None):
        self._run = run
        self._client = client
        self.state = state

    @property
    def active(self) -> bool:
        return self._run is not None

    @classmethod
    def start(
        cls,
        *,
        experiment_id: str,
        project: str = "gan-arrow",
        group: str | None = None,
        config: dict | None = None,
        mode: str = "online",
        run_id: str | None = None,
        description: str | None = None,
        client=None,
    ):
        if mode not in TRACKING_MODES:
            raise ValueError(f"mode must be one of {TRACKING_MODES}")
        if client is None:
            try:
                client = _import_swanlab()
            except Exception as error:
                return cls(None, TrackingState(mode="disabled", requested_mode=mode, degraded_reason=f"swanlab import failed: {error}"))

        if mode == "disabled":
            return cls(None, TrackingState(mode="disabled", requested_mode=mode))

        attempts = (mode,) + FALLBACK_CHAIN.get(mode, ())
        reason = None
        for attempt in attempts:
            if attempt == "disabled":
                return cls(None, TrackingState(mode="disabled", requested_mode=mode, degraded_reason=reason))
            try:
                run = client.init(
                    project=project,
                    group=group,
                    name=experiment_id,
                    description=description,
                    config=config,
                    mode=attempt,
                    public=False,
                    id=run_id,
                    # An explicit id means we are continuing a known run and must not silently fork it.
                    resume="must" if run_id else None,
                )
            except Exception as error:
                reason = f"{attempt} init failed: {type(error).__name__}: {error}"
                continue
            state = TrackingState(
                mode=attempt,
                swanlab_id=_safe_attr(run, "id"),
                swanlab_url=_safe_attr(run, "url"),
                requested_mode=mode,
                degraded_reason=reason if attempt != mode else None,
            )
            return cls(run, state, client=client)
        return cls(None, TrackingState(mode="disabled", requested_mode=mode, degraded_reason=reason))

    def _degrade(self, error: Exception, action: str):
        self._run = None
        self.state.mode = "disabled"
        self.state.degraded_reason = f"{action} failed: {type(error).__name__}: {error}"

    def log(self, event: dict):
        if not self.active:
            return False
        payload = {key: event[key] for key in SCALAR_KEYS if event.get(key) is not None}
        if not payload:
            return False
        try:
            self._run.log(payload, step=event.get("epoch"))
        except Exception as error:
            self._degrade(error, "log")
            return False
        return True

    def log_preview(self, path: Path, epoch: int | None = None, caption: str | None = None):
        """Upload a downscaled JPEG so the remote history keeps previews the local run does not."""
        if not self.active:
            return False
        path = Path(path)
        if not path.exists():
            return False
        try:
            image = self._client.Image(
                str(path),
                caption=caption or (f"epoch {epoch}" if epoch is not None else None),
                file_type="jpg",
                size=PREVIEW_MAX_PIXELS,
            )
            self._run.log({"preview": image}, step=epoch)
        except Exception as error:
            self._degrade(error, "preview upload")
            return False
        return True

    def finish(self, state: str = "success", error: str | None = None):
        if not self.active:
            return False
        try:
            self._run.finish(state=state, error=error)
        except Exception as failure:
            self._degrade(failure, "finish")
            return False
        return True


def resolve_mode(requested: str | None) -> str:
    """SWANLAB_MODE lets CI and offline machines opt out without editing any experiment.yaml."""
    override = os.environ.get("SWANLAB_MODE")
    if override in TRACKING_MODES:
        return override
    if requested in TRACKING_MODES:
        return requested
    return "online"
