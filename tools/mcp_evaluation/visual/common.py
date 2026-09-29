# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared task base, cameras, rendering, and candidate logging for visual calibration.

Every condition imports this same module. Rendering goes through the public
:class:`newton.mcp.SimulationSession` observation path, so reference photos,
restart-script renders, and live MCP observations use one renderer.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton

# Kernel-load messages would add noise to every agent-visible output.
wp.config.quiet = True


@dataclass(frozen=True)
class Camera:
    """Pinhole camera in world coordinates [m]; ``fov_y`` in degrees."""

    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_y: float = 45.0
    width: int = 480
    height: int = 360

    def observe_arguments(self, *, width: int | None = None, height: int | None = None) -> dict:
        """Return keyword arguments for ``session.dispatch('observe', ...)`` / ``newton_observe``."""
        return {
            "eye": list(self.eye),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y": self.fov_y,
            "width": width or self.width,
            "height": height or self.height,
        }

    def to_json(self) -> dict:
        return asdict(self)


def decode_png(data: bytes) -> np.ndarray:
    """Decode PNG bytes to an RGB uint8 array."""
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def load_png(path: str | Path) -> np.ndarray:
    """Load an RGB uint8 image."""
    return decode_png(Path(path).read_bytes())


def save_png(path: str | Path, rgb: np.ndarray) -> Path:
    """Save an RGB uint8 image."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
    return path


def contact_sheet(tiles: list[list[np.ndarray]], labels: list[list[str]] | None = None, gap: int = 4) -> np.ndarray:
    """Tile equally sized RGB images row-major with optional text labels."""
    from PIL import Image, ImageDraw

    rows, cols = len(tiles), max(len(row) for row in tiles)
    height, width = tiles[0][0].shape[:2]
    sheet = Image.new("RGB", (cols * width + (cols - 1) * gap, rows * height + (rows - 1) * gap), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            x, y = c * (width + gap), r * (height + gap)
            sheet.paste(Image.fromarray(tile), (x, y))
            if labels is not None and labels[r][c]:
                text = labels[r][c]
                box = draw.textbbox((x + 4, y + 4), text)
                draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(0, 0, 0))
                draw.text((x + 4, y + 4), text, fill=(255, 255, 255))
    return np.asarray(sheet)


def mismatch(simulated: np.ndarray, reference: np.ndarray, threshold: int = 24) -> tuple[np.ndarray, dict]:
    """Mismatch panel (magenta = max channel difference above ``threshold``) and pixel statistics.

    Identical to the statistics returned by the live MCP reference comparison.
    """
    if simulated.shape != reference.shape:
        raise ValueError(f"Image sizes differ: simulated {simulated.shape}, reference {reference.shape}")
    difference = np.abs(simulated.astype(np.int16) - reference.astype(np.int16))
    mask = difference.max(axis=-1) > threshold
    gray = (0.6 * reference.mean(axis=-1)).astype(np.uint8)
    panel = np.repeat(gray[..., None], 3, axis=-1)
    panel[mask] = (255, 0, 200)
    return panel, {
        "mean_abs_difference": round(float(difference.mean()), 3),
        "mismatch_fraction": round(float(mask.mean()), 5),
        "mismatch_threshold": threshold,
    }


class CandidateLog:
    """Append-only JSONL log of parameter sets and simulated time, shared by all processes."""

    def __init__(self, path: str | Path | None = None):
        path = path or os.environ.get("NEWTON_VISUAL_LOG") or Path.cwd() / "candidates.jsonl"
        self.path = Path(path)
        self.pid = os.getpid()

    def write(self, **record) -> None:
        record = {"wall_time_unix": time.time(), "pid": self.pid, **record}
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def look_at_quat(eye, target, up) -> list[float]:
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(right, forward), -forward))
    return list(wp.quat_from_matrix(wp.mat33(*rotation.flatten())))


class VisualTask:
    """Base class for one deterministic calibration scene.

    Subclasses define parameters, episodes, cameras, and :meth:`build`. The
    public workflow is ``set_params`` -> ``reset`` -> ``step``/``simulate_to`` ->
    ``render``. ``rollout`` bundles these for the reference times. Parameter
    changes rebuild the scene; ``on_rebuild`` lets a live application rebind.
    """

    name: ClassVar[str] = ""
    PARAMS: ClassVar[dict[str, dict]] = {}
    TRAIN_EPISODES: ClassVar[tuple[str, ...]] = ()
    HELDOUT_EPISODES: ClassVar[tuple[str, ...]] = ()
    CAMERAS: ClassVar[tuple[Camera, ...]] = ()
    REFERENCE_TIMES: ClassVar[tuple[float, ...]] = ()
    FRAME_DT: ClassVar[float] = 1.0 / 60.0
    SUBSTEPS: ClassVar[int] = 1
    DURATION: ClassVar[float] = 1.0

    def __init__(self, params: dict | None = None, *, episode: str | None = None, device=None):
        self.device = wp.get_device(device)
        self.params = self.default_params()
        self.episode = episode or self.TRAIN_EPISODES[0]
        self._check_episode(self.episode)
        # Every simulation is logged; the evaluation harness chooses the log file.
        self.log = CandidateLog()
        self.on_rebuild = None
        self.on_state_change = None
        self.live_session = None
        self._session = None
        self._graph = None
        self._sim_seconds_unlogged = 0.0
        self.sim_seconds_total = 0.0
        atexit.register(self._flush_sim_time)
        if params:
            self._validate(params)
            self.params.update({k: float(v) for k, v in params.items()})
        self._build_all()
        if self.log:
            self.log.write(task=self.name, event="params", params=self.params, episode=self.episode)

    # ------------------------------------------------------------------ parameters
    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {name: float(spec["initial"]) for name, spec in cls.PARAMS.items()}

    @classmethod
    def _validate(cls, params: dict) -> None:
        for name, value in params.items():
            if name not in cls.PARAMS:
                raise KeyError(f"Unknown parameter {name!r}; valid: {sorted(cls.PARAMS)}")
            lower, upper = cls.PARAMS[name]["bounds"]
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not lower <= value <= upper:
                raise ValueError(f"{name}={value} outside bounds [{lower}, {upper}]")

    def set_params(self, params: dict) -> dict[str, float]:
        """Validate and apply parameter values, rebuild the scene, and reset to t=0.

        Returns:
            The complete current parameter dictionary.
        """
        self._validate(params)
        self.params.update({k: float(v) for k, v in params.items()})
        self._build_all()
        if self.log:
            self.log.write(task=self.name, event="params", params=self.params, episode=self.episode)
        return dict(self.params)

    def set_episode(self, episode: str) -> None:
        """Select a training episode (initial conditions) and reset to t=0."""
        self._check_episode(episode)
        self._flush_sim_time()
        self.episode = episode
        self._build_all()

    def _check_episode(self, episode: str) -> None:
        allowed = self.TRAIN_EPISODES + self.HELDOUT_EPISODES
        if episode not in allowed:
            raise ValueError(f"Unknown episode {episode!r}; training episodes: {list(self.TRAIN_EPISODES)}")

    # ------------------------------------------------------------------ lifecycle
    def _build_all(self) -> None:
        self._flush_sim_time()
        with wp.ScopedDevice(self.device):
            self.build()
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.initialize_state(self.state_0)
            self._initial_state = self._copy_state(self.state_0)
        self._graph = None
        self.time = 0.0
        self.frame = 0
        self._session = None
        if self.on_rebuild is not None:
            self.on_rebuild(self)

    @staticmethod
    def _copy_state(state) -> dict:
        return {
            name: getattr(state, name).numpy().copy()
            for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
            if getattr(state, name, None) is not None
        }

    def reset(self) -> None:
        """Return to the episode's initial state at t=0 without rebuilding."""
        self._flush_sim_time()
        for name, value in self._initial_state.items():
            getattr(self.state_0, name).assign(value)
            getattr(self.state_1, name).assign(value)
        # Clear solver history without overwriting the restored public state.
        self.solver.reset(self.state_0, flags=newton.StateFlags.NONE)
        self.state_1.assign(self.state_0)
        self.after_reset()
        self.time = 0.0
        self.frame = 0
        if self.on_state_change is not None:
            self.on_state_change(self)

    def step(self, frames: int = 1) -> None:
        """Advance whole frames of ``FRAME_DT`` seconds."""
        for _ in range(int(frames)):
            with wp.ScopedDevice(self.device):
                self.before_frame(self.time)
                if self.device.is_cuda and self.use_graph():
                    if self._graph is None:
                        with wp.ScopedCapture() as capture:
                            self.simulate_frame()
                        self._graph = capture.graph
                    wp.capture_launch(self._graph)
                else:
                    self.simulate_frame()
            self.time = round(self.time + self.FRAME_DT, 9)
            self.frame += 1
            self._sim_seconds_unlogged += self.FRAME_DT
            self.sim_seconds_total += self.FRAME_DT
        if self.on_state_change is not None:
            self.on_state_change(self)

    def simulate_to(self, t: float) -> None:
        """Advance until simulation time ``t`` [s] (rounded to whole frames)."""
        frames = round((t - self.time) / self.FRAME_DT)
        if frames < 0:
            raise ValueError(f"Cannot simulate backwards from {self.time} to {t}; call reset() first")
        self.step(frames)

    def _flush_sim_time(self) -> None:
        if self.log and self._sim_seconds_unlogged > 0:
            self.log.write(
                task=self.name,
                event="simulated",
                seconds=round(self._sim_seconds_unlogged, 6),
                params=self.params,
                episode=self.episode,
            )
        self._sim_seconds_unlogged = 0.0

    # ------------------------------------------------------------------ rendering
    def _render_session(self):
        if self.live_session is not None:
            return self.live_session
        if self._session is None or self._session.model is not self.model:
            from newton.mcp import SimulationSession  # noqa: PLC0415

            self._session = SimulationSession(
                self.model, self.solver, state=self.state_0, state_next=self.state_1, control=self.control
            )
        self._session.state = self.state_0
        return self._session

    def render(self, cameras=None, *, width: int | None = None, height: int | None = None) -> dict[str, np.ndarray]:
        """Render the current state from reference cameras (names or :class:`Camera`)."""
        cameras = self.CAMERAS if cameras is None else cameras
        cameras = [self.camera(c) if isinstance(c, str) else c for c in cameras]
        session = self._render_session()
        images = {}
        for camera in cameras:
            data = session.dispatch("observe", camera.observe_arguments(width=width, height=height))
            images[camera.name] = decode_png(base64.b64decode(data["image_base64"]))
        return images

    @classmethod
    def camera(cls, name: str) -> Camera:
        for camera in cls.CAMERAS:
            if camera.name == name:
                return camera
        raise KeyError(f"Unknown camera {name!r}; cameras: {[c.name for c in cls.CAMERAS]}")

    def rollout(self, times=None, cameras=None) -> dict[float, dict[str, np.ndarray]]:
        """Reset, simulate, and render at each time [s]; returns ``{time: {camera: rgb}}``."""
        times = sorted(self.REFERENCE_TIMES if times is None else times)
        self.reset()
        frames = {}
        for t in times:
            self.simulate_to(t)
            frames[t] = self.render(cameras)
        return frames

    # ------------------------------------------------------------------ subclass hooks
    def build(self) -> None:
        raise NotImplementedError

    def initialize_state(self, state) -> None:
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)

    def after_reset(self) -> None:
        pass

    def before_frame(self, t: float) -> None:
        pass

    def simulate_frame(self) -> None:
        raise NotImplementedError

    def use_graph(self) -> bool:
        return True

    def measurements(self) -> dict[str, np.ndarray]:
        """Hidden-truth comparison quantities for the verifier (not a scoring oracle)."""
        raise NotImplementedError

    @classmethod
    def public_spec(cls) -> dict:
        return {
            "task": cls.name,
            "parameters": {name: dict(spec) for name, spec in cls.PARAMS.items()},
            "training_episodes": list(cls.TRAIN_EPISODES),
            "reference_times_s": list(cls.REFERENCE_TIMES),
            "frame_dt_s": cls.FRAME_DT,
            "cameras": [c.to_json() for c in cls.CAMERAS],
        }
