# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Panda joint-encoder offset calibration from multi-view photographs (kinematic)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.solvers

from .common import Camera, VisualTask

MENAGERIE = Path(os.environ.get("NEWTON_EVAL_MENAGERIE", "/home/horde/artifacts/newton-live-mcp/menagerie"))


def _panda_mjcf() -> str:
    path = MENAGERIE / "franka_emika_panda" / "panda.xml"
    root = ET.parse(path).getroot()
    for child in list(root):
        if child.tag in ("actuator", "keyframe"):
            root.remove(child)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attr in ("meshdir", "texturedir"):
        compiler.set(attr, str((path.parent / compiler.get(attr, ".")).resolve()))
    return ET.tostring(root, encoding="unicode")


class ArmOffsets(VisualTask):
    """A Panda arm is photographed at commanded joint angles.

    The physical arm's joint zero positions differ from the model's: the
    actual angle of joint ``i`` is ``commanded[i] + offset_i``. Parameters are
    the seven offsets. This is a kinematic calibration; time does not advance
    the pose.
    """

    name = "arm_offsets"
    PARAMS: ClassVar[dict[str, dict]] = {
        f"offset_{i}": {
            "bounds": [-0.4, 0.4],
            "initial": 0.0,
            "unit": "rad",
            "description": f"Encoder zero offset of Panda joint{i}: actual = commanded + offset.",
        }
        for i in range(1, 8)
    }
    COMMANDS: ClassVar[dict[str, tuple]] = {
        "pose_0": (0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.8),
        "pose_1": (0.6, 0.2, -0.3, -1.8, 0.4, 2.2, -0.5),
        "pose_2": (-0.8, -0.1, 0.5, -2.5, -0.6, 1.4, 1.5),
        "pose_3": (0.3, 0.6, 0.2, -1.2, 1.2, 1.6, 0.0),
        "heldout_0": (-0.3, 0.4, -0.6, -1.5, -0.9, 2.5, 0.3),
        "heldout_1": (1.0, -0.2, 0.3, -2.0, 0.8, 1.2, -1.0),
        "heldout_2": (-1.2, 0.3, 0.1, -1.0, 0.2, 1.8, 2.0),
        "heldout_3": (0.1, -0.7, -0.4, -2.7, 0.5, 2.8, -1.8),
    }
    TRAIN_EPISODES = ("pose_0", "pose_1", "pose_2", "pose_3")
    HELDOUT_EPISODES = ("heldout_0", "heldout_1", "heldout_2", "heldout_3")
    CAMERAS = (
        Camera("front", eye=(1.5, -1.25, 1.0), target=(0.25, 0.0, 0.45), fov_y=45.0),
        Camera("side", eye=(0.3, -1.9, 0.55), target=(0.25, 0.0, 0.45), fov_y=45.0),
        Camera("top", eye=(0.3, 0.0, 2.1), target=(0.3, 0.0, 0.3), up=(1.0, 0.0, 0.0), fov_y=45.0),
    )
    REFERENCE_TIMES = (0.0,)
    FRAME_DT = 1.0 / 60.0
    DURATION = 0.0
    FINGER_OPENING = 0.03

    def build(self) -> None:
        builder = newton.ModelBuilder()
        builder.add_ground_plane(color=(0.75, 0.75, 0.75))
        builder.add_mjcf(_panda_mjcf(), enable_self_collisions=False)
        self.model = builder.finalize(device=self.device)
        self.solver = newton.solvers.SolverXPBD(self.model)
        self.pipeline = None
        self.contacts = None
        labels = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        starts = self.model.joint_q_start.numpy()
        self._arm_q = [int(starts[labels.index(f"joint{i}")]) for i in range(1, 8)]
        self._finger_q = [int(starts[i]) for i, label in enumerate(labels) if label.startswith("finger_joint")]

    def _pose(self) -> np.ndarray:
        q = self.model.joint_q.numpy().copy()
        command = self.COMMANDS[self.episode]
        for i, index in enumerate(self._arm_q):
            q[index] = command[i] + self.params[f"offset_{i + 1}"]
        for index in self._finger_q:
            q[index] = self.FINGER_OPENING
        return q

    def initialize_state(self, state) -> None:
        q = wp.array(self._pose(), dtype=float, device=self.device)
        state.joint_q.assign(q)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _apply_pose(self) -> None:
        self.initialize_state(self.state_0)
        self.state_1.assign(self.state_0)
        self._initial_state = self._copy_state(self.state_0)

    def set_params(self, params: dict) -> dict[str, float]:
        """Validate and apply offsets; the pose updates immediately (no rebuild)."""
        self._validate(params)
        self.params.update({k: float(v) for k, v in params.items()})
        self._apply_pose()
        if self.log:
            self.log.write(task=self.name, event="params", params=self.params, episode=self.episode)
        return dict(self.params)

    def set_episode(self, episode: str) -> None:
        """Select a commanded pose (no rebuild)."""
        self._check_episode(episode)
        self._flush_sim_time()
        self.episode = episode
        self._apply_pose()

    def simulate_frame(self) -> None:
        pass

    def use_graph(self) -> bool:
        return False

    def measurements(self) -> dict[str, np.ndarray]:
        return {"body_q": self.state_0.body_q.numpy().copy()}
