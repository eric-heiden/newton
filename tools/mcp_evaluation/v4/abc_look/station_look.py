# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Station twin from ABC-130k for matching the real camera's look in Blender.

The geometry is fixed: ``scene.json`` holds a fitted top-camera pose, the resized
enclosure and table, and the objects on the table of the ABC plates episode
(https://abc.bot/#data). The robot is posed from ``joint_log.npz``; ``frames/``
holds recorded top-camera frames and ``camera.json`` the calibrated intrinsics.
Renders come from Blender EEVEE (``look_common.LookRenderer``), with the look
applied by the ``bpy`` code in ``look.py``.

Run: ``python station_look.py --viewer null``
"""

from __future__ import annotations

from pathlib import Path

import look_common
import numpy as np
import twin_render

import newton
import newton.examples

HERE = Path(__file__).resolve().parent
FRAME_DT = 1.0 / 30.0  # camera frame period [s]


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
        self.model = look_common.build_model(HERE)
        self.state = self.model.state()
        # Top-camera pose [x, y, z, qx, qy, qz, qw] and intrinsics, e.g. for
        # observe(backend='blender', pose=example.pose, intrinsics=example.intrinsics, width=640, height=480).
        self.pose, self.intrinsics = look_common.camera(HERE)
        self.frame = 0
        self.show_frame(0)
        self.viewer.set_model(self.model)

    def show_frame(self, frame: int):
        """Pose the robot in ``self.state`` from log frame ``frame``."""
        self.frame = frame
        twin_render.pose_robot(self.model, self.state, self.log["q"][frame])

    def step(self):
        """Advance the joint-log playback by one camera frame (looping)."""
        self.show_frame((self.frame + 1) % len(self.log["q"]))
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        assert np.isfinite(self.state.body_q.numpy()).all()

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
