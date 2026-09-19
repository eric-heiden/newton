# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Release a folded flap against a fixed lower panel using IPC or VBD.

Command: python -m newton.examples cloth_folding --solver ipc
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import style3d


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_ipc = args.solver == "ipc"
        nx, ny = args.resolution, args.resolution // 2
        if nx < 16 or nx % 16:
            raise ValueError("resolution must be a positive multiple of 16 to align the fixed panel")
        builder = newton.ModelBuilder()
        newton.solvers.SolverIPC.register_custom_attributes(builder)
        builder.default_tri_ke = 250.0
        builder.default_tri_ka = 0.0
        builder.add_ground_plane()
        style3d.add_cloth_grid(
            builder,
            pos=(0.0, 0.0, 0.12),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=nx,
            dim_y=ny,
            cell_x=0.8 / nx,
            cell_y=0.4 / ny,
            mass=0.3 / ((nx + 1) * (ny + 1)),
            particle_radius=0.004,
            tri_aniso_ke=wp.vec3(500.0, 500.0, 250.0),
            tri_ka=0.0,
            tri_kd=0.0,
            edge_aniso_ke=wp.vec3(2.0e-5),
            edge_kd=0.0,
        )
        rest = np.asarray(builder.particle_q, dtype=np.float32)
        self.pinned = rest[:, 0] <= 0.35 + 1.0e-6
        for index in np.flatnonzero(self.pinned):
            builder.particle_mass[index] = 0.0
        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_mu = 0.0
        self.pipeline, self.contacts = None, None
        if self.use_ipc:
            self.solver = newton.solvers.SolverIPC(
                self.model,
                config=newton.solvers.SolverIPC.Config(
                    minimum_separation=0.004,
                    contact_distance=0.05,
                    barrier_stiffness=0.005,
                    self_contact_thickness=0.008,
                    self_contact_distance=0.01,
                    max_newton_iterations=128,
                    max_pcg_iterations=32,
                    max_line_search_iterations=24,
                    absolute_tolerance=0.01,
                    relative_tolerance=0.002,
                    energy_tolerance=1.0e-8,
                    velocity_damping=1.0,
                ),
            )
        else:
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=20,
                particle_enable_self_contact=True,
                particle_self_contact_margin=0.008,
                particle_self_contact_gap=0.008,
                particle_vertex_contact_buffer_size=128,
                particle_edge_contact_buffer_size=256,
            )
            self.pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        distance = np.maximum(rest[:, 0] - 0.35, 0.0)
        angle = np.minimum(distance / 0.04, np.pi)
        beyond = np.maximum(distance - np.pi * 0.04, 0.0)
        folded = rest.copy()
        folded[~self.pinned, 0] = (0.35 + 0.04 * np.sin(angle) - beyond)[~self.pinned]
        folded[~self.pinned, 2] += (0.04 * (1.0 - np.cos(angle)))[~self.pinned]
        for state in (self.state_0, self.state_1):
            state.particle_q.assign(folded)
        self.fixed_positions = folded[self.pinned].copy()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.4, -0.9, 0.6), -24.0, -270.0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        if self.use_ipc:
            assert int(self.solver.diagnostics.failed_steps.numpy()[0]) == 0
        q = self.state_0.particle_q.numpy()
        assert np.isfinite(q).all()
        np.testing.assert_array_equal(q[self.pinned], self.fixed_positions)

    def test_final(self):
        self.test_post_step()
        assert np.isfinite(self.state_0.particle_qd.numpy()).all()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--resolution", type=int, default=16, help="Cells along the sheet; a multiple of 16.")
        parser.add_argument("--solver", choices=("ipc", "vbd"), default="ipc")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
