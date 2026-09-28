# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Planar pushing calibration from top/side video frames (SolverMuJoCo, CUDA)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.solvers

from .common import Camera, VisualTask


@wp.kernel
def _drive_pusher(
    clock: wp.array[float],
    start: wp.vec3,
    velocity: wp.vec3,
    push_time: float,
    ramp_up: float,
    ramp_down: float,
    dt: float,
    q_start: int,
    qd_start: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    # Trapezoidal speed profile: ramp up, cruise, ramp down, then hold still.
    t = wp.clamp(clock[0], 0.0, push_time)
    cruise = push_time - ramp_up - ramp_down
    distance = float(0.0)
    speed = float(0.0)
    if t < ramp_up:
        speed = t / ramp_up
        distance = 0.5 * t * t / ramp_up
    elif t < ramp_up + cruise:
        speed = 1.0
        distance = 0.5 * ramp_up + (t - ramp_up)
    else:
        remaining = push_time - t
        speed = remaining / ramp_down
        distance = 0.5 * ramp_up + cruise + 0.5 * (ramp_down * ramp_down - remaining * remaining) / ramp_down
    if clock[0] >= push_time:
        speed = 0.0
    p = start + velocity * distance
    joint_q[q_start + 0] = p[0]
    joint_q[q_start + 1] = p[1]
    joint_q[q_start + 2] = p[2]
    joint_q[q_start + 3] = 0.0
    joint_q[q_start + 4] = 0.0
    joint_q[q_start + 5] = 0.0
    joint_q[q_start + 6] = 1.0
    for i in range(6):
        joint_qd[qd_start + i] = 0.0
    joint_qd[qd_start + 0] = velocity[0] * speed
    joint_qd[qd_start + 1] = velocity[1] * speed
    joint_qd[qd_start + 2] = velocity[2] * speed
    clock[0] = clock[0] + dt


class PlanarPush(VisualTask):
    """A kinematic cylinder pushes a loaded box across a table, then stops.

    The box has a hidden internal load (center-of-mass offset). Parameters are
    table friction, pusher friction, and the in-plane center-of-mass offset.
    Box geometry, mass, pusher paths, and solver settings are fixed.
    """

    name = "push"
    PARAMS: ClassVar[dict[str, dict]] = {
        "table_friction": {
            "bounds": [0.05, 1.5],
            "initial": 1.0,
            "unit": "1",
            "description": "Coulomb friction between box and table.",
        },
        "pusher_friction": {
            "bounds": [0.05, 1.5],
            "initial": 1.0,
            "unit": "1",
            "description": "Coulomb friction between pusher and box.",
        },
        "com_x": {
            "bounds": [-0.06, 0.06],
            "initial": 0.0,
            "unit": "m",
            "description": "Center-of-mass offset along the box's long (+x, yellow-marker) axis, box frame.",
        },
        "com_y": {
            "bounds": [-0.04, 0.04],
            "initial": 0.0,
            "unit": "m",
            "description": "Center-of-mass offset along the box's short (+y) axis, box frame.",
        },
    }
    TRAIN_EPISODES = ("push_a", "push_b")
    HELDOUT_EPISODES = ("push_angled", "push_side")
    CAMERAS = (
        Camera("top", eye=(0.12, 0.0, 1.1), target=(0.12, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov_y=40.0),
        Camera("side", eye=(0.25, -0.75, 0.35), target=(0.12, 0.0, 0.02), fov_y=40.0),
    )
    REFERENCE_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
    FRAME_DT = 0.01
    SUBSTEPS = 5
    DURATION = 1.5
    BOX_HALF = (0.08, 0.05, 0.02)
    BOX_MASS = 0.5
    PUSHER_RADIUS = 0.015
    RAMP_UP = 0.15

    # start xy [m], direction [deg], cruise speed [m/s], push duration including ramps [s], stop ramp [s]
    _EPISODES: ClassVar[dict[str, dict]] = {
        "push_a": {"start": (-0.14, 0.025), "direction_deg": 0.0, "speed": 0.35, "duration": 0.8, "ramp_down": 0.15},
        "push_b": {"start": (-0.14, -0.03), "direction_deg": 0.0, "speed": 0.5, "duration": 0.5, "ramp_down": 0.02},
        "push_angled": {
            "start": (-0.13, 0.09),
            "direction_deg": -25.0,
            "speed": 0.4,
            "duration": 0.7,
            "ramp_down": 0.02,
        },
        "push_side": {"start": (0.02, -0.14), "direction_deg": 90.0, "speed": 0.45, "duration": 0.5, "ramp_down": 0.15},
    }

    def build(self) -> None:
        p, episode = self.params, self._EPISODES[self.episode]
        builder = newton.ModelBuilder()
        table = builder.default_shape_cfg.copy()
        table.mu = p["table_friction"]
        builder.add_ground_plane(cfg=table, color=(0.78, 0.74, 0.66))
        hx, hy, hz = self.BOX_HALF
        m = self.BOX_MASS
        inertia = wp.mat33(
            m * (hy * hy + hz * hz) / 3.0,
            0.0,
            0.0,
            0.0,
            m * (hx * hx + hz * hz) / 3.0,
            0.0,
            0.0,
            0.0,
            m * (hx * hx + hy * hy) / 3.0,
        )
        self.box = builder.add_body(
            xform=wp.transform((0.0, 0.0, hz), wp.quat_identity()),
            com=wp.vec3(p["com_x"], p["com_y"], 0.0),
            inertia=inertia,
            mass=m,
            lock_inertia=True,
            label="box",
        )
        box_cfg = builder.default_shape_cfg.copy()
        # MuJoCo takes the larger geom friction of a pair, so the table/pusher values govern.
        box_cfg.mu = 0.01
        box_cfg.density = 0.0
        builder.add_shape_box(self.box, hx=hx, hy=hy, hz=hz, cfg=box_cfg, color=(0.25, 0.35, 0.75))
        marker = builder.default_shape_cfg.copy()
        marker.density = 0.0
        marker.has_shape_collision = False
        marker.has_particle_collision = False
        builder.add_shape_box(
            self.box,
            xform=wp.transform((0.6 * hx, 0.0, hz + 0.002), wp.quat_identity()),
            hx=0.25 * hx,
            hy=0.8 * hy,
            hz=0.002,
            cfg=marker,
            color=(0.95, 0.80, 0.10),
        )
        direction = math.radians(episode["direction_deg"])
        self._velocity = wp.vec3(episode["speed"] * math.cos(direction), episode["speed"] * math.sin(direction), 0.0)
        # The capsule clears the table so it cannot wedge under the box.
        self._start = wp.vec3(*episode["start"], self.PUSHER_RADIUS + 0.03 + 0.004)
        self._push_time = float(episode["duration"])
        self._ramp_down = float(episode["ramp_down"])
        self.pusher = builder.add_body(
            xform=wp.transform(self._start, wp.quat_identity()),
            is_kinematic=True,
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
            label="pusher",
        )
        pusher_cfg = builder.default_shape_cfg.copy()
        pusher_cfg.mu = p["pusher_friction"]
        pusher_cfg.density = 0.0
        builder.add_shape_capsule(
            self.pusher, radius=self.PUSHER_RADIUS, half_height=0.03, cfg=pusher_cfg, color=(0.85, 0.15, 0.15)
        )
        self.model = builder.finalize(device=self.device)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=True, njmax=200, nconmax=100, iterations=100, ls_iterations=50
        )
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = None
        self._clock = wp.zeros(1, dtype=float, device=self.device)
        starts = self.model.joint_q_start.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        pusher_joint = int(np.flatnonzero(self.model.joint_child.numpy() == self.pusher)[0])
        self._pusher_q, self._pusher_qd = int(starts[pusher_joint]), int(dof_starts[pusher_joint])

    def after_reset(self) -> None:
        self._clock.zero_()

    def simulate_frame(self) -> None:
        dt = self.FRAME_DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            wp.launch(
                _drive_pusher,
                dim=1,
                inputs=[
                    self._clock,
                    self._start,
                    self._velocity,
                    self._push_time,
                    self.RAMP_UP,
                    self._ramp_down,
                    dt,
                    self._pusher_q,
                    self._pusher_qd,
                ],
                outputs=[self.state_0.joint_q, self.state_0.joint_qd],
            )
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def measurements(self) -> dict[str, np.ndarray]:
        return {"box_q": self.state_0.body_q.numpy()[self.box].copy()}
