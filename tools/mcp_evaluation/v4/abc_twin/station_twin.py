# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Digital twin of a real ABC YAM bimanual station, seen through its top camera.

The recording comes from ABC-130k (https://abc.bot/#data), episode "place the
plates into the plastic bin on the countertop": two 6-DoF YAM arms with parallel
grippers in a white enclosure, filmed by a RealSense D405 above the table.
``joint_log.npz`` holds the measured joint positions for the first 43 camera
frames (30 Hz), in which the arms move and the objects on the table do not.
``frames/`` holds recorded top-camera frames (640x480) for some of them, and
``camera.json`` the camera's calibrated intrinsics and distortion.

``build_model`` loads the station MJCF (station/). The robot is posed
kinematically from the log; ``twin_render.TopCamera`` renders the model from
``CAMERA_POSITION``/``CAMERA_ROTATION`` with the real intrinsics and ``LOOK``,
and ``evaluate`` scores the renders against the recorded frames.

Run: ``python station_twin.py --viewer null``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import twin_render

import newton
import newton.examples

HERE = Path(__file__).resolve().parent
STATION = HERE / "station" / "yam_bimanual_empty.xml"
FRAME_DT = 1.0 / 30.0  # camera frame period [s]

# ---------------- Twin parameters (tunable) ----------------
# Top-camera pose in the world frame: position [m] and orientation (x, y, z, w).
# The camera looks along its -Z axis; +Y is image up. Nominal CAD mount.
CAMERA_POSITION = (0.086, 0.0, 1.704)
CAMERA_ROTATION = (-0.183012, 0.183012, 0.683013, -0.683013)

# Light direction (world, the direction the light travels), shadows, and exposure (linear gain).
LOOK = {"light_direction": (-0.57735, 0.57735, -0.57735), "shadows": True, "exposure": 1.0}


def build_scene(builder: newton.ModelBuilder) -> None:
    """Add the objects in the station (static, not simulated). Nothing yet."""


def build_model() -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(STATION))
    build_scene(builder)
    return builder.finalize()


# -----------------------------------------------------------


def recorded_frames() -> dict[int, np.ndarray]:
    """Recorded top-camera frames by log frame index, each shape [480, 640, 3], dtype uint8."""
    from PIL import Image

    return {
        int(path.stem.split("_f")[1]): np.asarray(Image.open(path).convert("RGB"))
        for path in sorted((HERE / "frames").glob("top_f*.png"))
    }


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = FRAME_DT
        self.sim_time = 0.0
        self.log = np.load(HERE / "joint_log.npz")
        self.model = build_model()
        self.state = self.model.state()
        self.frame = 0
        twin_render.pose_robot(self.model, self.state, self.log["q"][0])
        self.camera = twin_render.TopCamera(self.model, CAMERA_POSITION, CAMERA_ROTATION, LOOK)
        self.viewer.set_model(self.model)

    def step(self):
        """Advance the joint-log playback by one camera frame (looping)."""
        self.frame = (self.frame + 1) % len(self.log["q"])
        twin_render.pose_robot(self.model, self.state, self.log["q"][self.frame])
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_top(self, frame: int) -> np.ndarray:
        """Top-camera image of log frame ``frame``, shape [480, 640, 3], dtype uint8."""
        state = self.model.state()
        twin_render.pose_robot(self.model, state, self.log["q"][frame])
        return self.camera.render(state)

    def evaluate(self, frames: dict[int, np.ndarray] | None = None) -> dict[str, float]:
        """Mean image scores (see twin_render.score) over recorded frames."""
        frames = recorded_frames() if frames is None else frames
        scores = [twin_render.score(self.render_top(i), image) for i, image in frames.items()]
        return {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}

    def test_final(self):
        scores = self.evaluate()
        assert all(np.isfinite(value) for value in scores.values())

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    print(example.evaluate())
