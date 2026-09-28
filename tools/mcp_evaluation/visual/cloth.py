# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Cloth drape material calibration from multi-view photographs (VBD, CUDA)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.solvers

from .common import Camera, VisualTask


class ClothDrape(VisualTask):
    """A 1 m square cloth is dropped flat onto an obstacle and drapes under gravity.

    Episodes differ only in obstacle and initial cloth placement. Parameters are
    the cloth's material and friction; geometry, resolution, and solver settings
    are fixed.
    """

    name = "cloth_drape"
    PARAMS: ClassVar[dict[str, dict]] = {
        "stretch_stiffness": {
            "bounds": [100.0, 100000.0],
            "initial": 100.0,
            "unit": "N/m",
            "description": "In-plane membrane stiffness (VBD tri_ke = tri_ka).",
        },
        "bend_stiffness": {
            "bounds": [0.0001, 100.0],
            "initial": 100.0,
            "unit": "N*m/rad",
            "description": "Dihedral bending stiffness per edge (VBD edge_ke).",
        },
        "areal_density": {
            "bounds": [0.02, 2.0],
            "initial": 1.0,
            "unit": "kg/m^2",
            "description": "Cloth mass per unit area.",
        },
        "friction": {
            "bounds": [0.0, 1.5],
            "initial": 0.5,
            "unit": "1",
            "description": "Coulomb friction between cloth and obstacle/ground.",
        },
    }
    TRAIN_EPISODES = ("center", "corner")
    HELDOUT_EPISODES = ("offset_rotated", "sphere")
    CAMERAS = (
        Camera("front", eye=(1.7, -2.0, 1.35), target=(0.0, 0.0, 0.35), fov_y=40.0),
        Camera("top", eye=(0.0, -0.8, 2.6), target=(0.0, 0.0, 0.3), fov_y=45.0),
    )
    REFERENCE_TIMES = (0.5, 1.0, 2.0)
    FRAME_DT = 1.0 / 60.0
    SUBSTEPS = 10
    DURATION = 2.0
    RESOLUTION = 40
    SIZE = 1.0

    _EPISODES: ClassVar[dict[str, dict]] = {
        "center": {"obstacle": "box", "offset": (0.0, 0.0), "yaw_deg": 0.0},
        "corner": {"obstacle": "box", "offset": (0.3, 0.25), "yaw_deg": 0.0},
        "offset_rotated": {"obstacle": "box", "offset": (-0.28, 0.12), "yaw_deg": 30.0},
        "sphere": {"obstacle": "sphere", "offset": (0.05, -0.05), "yaw_deg": 15.0},
    }

    def build(self) -> None:
        p, episode = self.params, self._EPISODES[self.episode]
        builder = newton.ModelBuilder()
        ground = builder.default_shape_cfg.copy()
        ground.mu = p["friction"]
        builder.add_ground_plane(cfg=ground, color=(0.72, 0.72, 0.70))
        obstacle = builder.default_shape_cfg.copy()
        obstacle.mu = p["friction"]
        if episode["obstacle"] == "box":
            builder.add_shape_box(
                -1,
                xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()),
                hx=0.25,
                hy=0.25,
                hz=0.25,
                cfg=obstacle,
                color=(0.20, 0.45, 0.80),
            )
        else:
            builder.add_shape_sphere(
                -1,
                xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()),
                radius=0.3,
                cfg=obstacle,
                color=(0.20, 0.45, 0.80),
            )
        n, size = self.RESOLUTION, self.SIZE
        cell = size / n
        yaw = math.radians(episode["yaw_deg"])
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        corner = np.array([-0.5 * size, -0.5 * size, 0.0])
        c, s = math.cos(yaw), math.sin(yaw)
        corner = np.array([c * corner[0] - s * corner[1], s * corner[0] + c * corner[1], 0.0])
        position = corner + np.array([*episode["offset"], 0.8])
        # Stiffness damping is fixed relative to stiffness so it does not become a free parameter.
        builder.add_cloth_grid(
            pos=wp.vec3(*position),
            rot=rotation,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=n,
            dim_y=n,
            cell_x=cell,
            cell_y=cell,
            mass=p["areal_density"] * cell * cell,
            tri_ke=p["stretch_stiffness"],
            tri_ka=p["stretch_stiffness"],
            tri_kd=1.0e-3 * p["stretch_stiffness"],
            edge_ke=p["bend_stiffness"],
            edge_kd=1.0e-3 * p["bend_stiffness"],
            particle_radius=0.008,
            color=(0.95, 0.55, 0.15),
        )
        builder.color(include_bending=True)
        self.model = builder.finalize(device=self.device)
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e-2
        self.model.soft_contact_mu = p["friction"]
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=10,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.01,
            particle_self_contact_gap=0.005,
        )
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()

    def initialize_state(self, state) -> None:
        pass

    def simulate_frame(self) -> None:
        dt = self.FRAME_DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def use_graph(self) -> bool:
        # An even substep count returns the state buffers to their original roles after each frame.
        return self.SUBSTEPS % 2 == 0

    def measurements(self) -> dict[str, np.ndarray]:
        return {"particle_q": self.state_0.particle_q.numpy().copy()}
