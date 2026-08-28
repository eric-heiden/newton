# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD Cloth Hanging
#
# A small cloth patch hangs from its top edge. This is the minimal example
# for XPBD membrane and bending constraints without auxiliary springs.
#
# Command: python -m newton.examples cloth_xpbd_hanging
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.dim_x = 20
        self.dim_y = 20
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, 0.0, 1.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi),
            vel=wp.vec3(0.0, 0.5, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=0.1,
            cell_y=0.1,
            mass=2.0e-3,
            fix_top=False,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.0,
            edge_ke=2.0,
            edge_kd=0.1,
            particle_radius=0.025,
        )
        top_start = self.dim_y * (self.dim_x + 1)
        builder.particle_mass[top_start] = 0.0
        builder.particle_mass[top_start + self.dim_x] = 0.0
        builder.particle_qd[top_start] = wp.vec3(0.0)
        builder.particle_qd[top_start + self.dim_x] = wp.vec3(0.0)
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.mu = 0.2
        ground_cfg.ka = 0.0
        builder.add_ground_plane(cfg=ground_cfg)

        self.model = builder.finalize()
        self.model.particle_mu = 0.2
        self.model.soft_contact_mu = 0.2
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            particle_enable_self_contact=True,
            particle_enable_triangle_intersection_recovery=True,
            particle_self_contact_relaxation=0.5,
            particle_self_contact_radius=0.05,
            particle_self_contact_margin=0.08,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.04,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.pinned_particles = np.array([top_start, top_start + self.dim_x])
        self.pinned_positions = self.state_0.particle_q.numpy()[self.pinned_particles].copy()

        self.viewer.set_model(self.model)
        self.viewer.configure_picking(
            particle_pick_radius=0.25,
            particle_pick_stiffness=3600.0,
            particle_pick_damping=120.0,
            particle_pick_max_acceleration=100.0,
        )
        self.viewer.set_camera(wp.vec3(3.0, -7.0, 2.0), 0.0, 113.0)
        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Keep the pinned edge fixed while the free cloth remains bounded."""
        positions = self.state_0.particle_q.numpy()
        np.testing.assert_allclose(positions[self.pinned_particles], self.pinned_positions, atol=1.0e-6)
        assert np.all(np.isfinite(positions)), "cloth positions must remain finite"
        assert float(np.min(positions[:, 2])) > -0.02, "cloth must remain above the ground"
        assert float(np.max(np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1))) < 20.0, (
            "cloth velocity must remain bounded"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
