# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth IPC
#
# A tilted cloth patch falls onto a fixed plane. The SolverIPC barrier and
# conservative point-plane CCD keep every committed particle above the plane.
#
# Command: python -m newton.examples cloth_ipc
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import style3d


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        newton.solvers.SolverIPC.register_custom_attributes(builder)
        builder.add_ground_plane()

        resolution = args.resolution
        style3d.add_cloth_grid(
            builder,
            pos=wp.vec3(-0.5 * resolution * 0.04, -0.5 * resolution * 0.04, 1.2),
            rot=wp.quat_rpy(0.12, -0.18, 0.2),
            vel=wp.vec3(0.0),
            dim_x=resolution,
            dim_y=resolution,
            cell_x=0.04,
            cell_y=0.04,
            mass=0.005,
            particle_radius=0.01,
            tri_aniso_ke=wp.vec3(5.0e2, 5.0e2, 5.0e1),
            edge_aniso_ke=wp.vec3(2.0e-5, 2.0e-5, 2.0e-5),
        )
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverIPC(
            self.model,
            config=newton.solvers.SolverIPC.Config(
                minimum_separation=0.01,
                contact_distance=0.05,
                barrier_stiffness=0.005,
                max_newton_iterations=64,
                max_pcg_iterations=24,
                max_line_search_iterations=10,
                absolute_tolerance=1.0e-2,
                relative_tolerance=2.0e-3,
                energy_tolerance=1.0e-5,
                initial_step_size=1.0,
            ),
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -2.2, 1.6), -24.0, -270.0)
        self.capture()

    def capture(self):
        if wp.get_device().is_cpu:
            self.graph = None
            return
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
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

    def test_final(self):
        status = int(self.solver.diagnostics.status.numpy()[0])
        assert status == int(self.solver.Status.CONVERGED), self.solver.Status(status).name
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        assert np.isfinite(positions).all()
        assert np.isfinite(velocities).all()
        assert float(positions[:, 2].min()) > 0.0

    def test_post_step(self):
        assert int(self.solver.diagnostics.failed_steps.numpy()[0]) == 0

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--resolution", type=int, default=32, help="Cloth cells along each axis.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
