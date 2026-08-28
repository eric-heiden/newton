# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD Cloth and Rigid Picking
#
# Right-click and drag either the hanging cloth or a rigid body. Cloth
# picking moves a smoothly weighted geodesic patch instead of one particle.
# The patch follows the cloth topology, so overlapping layers are not
# accidentally selected together.
#
# Command: python -m newton.examples cloth_xpbd_picking
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.dim_x = 24
        self.dim_y = 18
        builder = newton.ModelBuilder()

        contact_cfg = builder.default_shape_cfg.copy()
        contact_cfg.density = 200.0
        contact_cfg.mu = 0.7
        contact_cfg.ka = 0.0
        builder.add_ground_plane(cfg=contact_cfg)

        sphere = builder.add_body(
            xform=wp.transform(wp.vec3(-1.0, -0.9, 0.5), wp.quat_identity()),
            label="pickable sphere",
        )
        builder.add_shape_sphere(
            sphere,
            radius=0.38,
            cfg=contact_cfg,
            color=wp.vec3(0.92, 0.35, 0.18),
        )

        box = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, -0.9, 0.45),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.25),
            ),
            label="pickable box",
        )
        builder.add_shape_box(
            box,
            hx=0.38,
            hy=0.3,
            hz=0.3,
            cfg=contact_cfg,
            color=wp.vec3(0.25, 0.55, 0.9),
        )

        capsule = builder.add_body(
            xform=wp.transform(
                wp.vec3(1.0, -0.9, 0.65),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi),
            ),
            label="pickable capsule",
        )
        builder.add_shape_capsule(
            capsule,
            radius=0.24,
            half_height=0.38,
            cfg=contact_cfg,
            color=wp.vec3(0.35, 0.8, 0.42),
        )

        builder.add_cloth_grid(
            pos=wp.vec3(-1.2, 0.25, 1.15),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi),
            vel=wp.vec3(0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=0.1,
            cell_y=0.1,
            mass=2.0e-3,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.0,
            edge_ke=8.0,
            edge_kd=0.25,
            particle_radius=0.025,
            label="pickable cloth",
        )
        top_start = self.dim_y * (self.dim_x + 1)
        for particle in (top_start, top_start + self.dim_x):
            builder.particle_mass[particle] = 0.0
            builder.particle_qd[particle] = wp.vec3(0.0)

        self.model = builder.finalize()
        self.model.particle_mu = 0.6
        self.model.soft_contact_mu = 0.6
        self.model.shape_material_ka.zero_()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            particle_enable_self_contact=True,
            particle_enable_triangle_intersection_recovery=True,
            particle_self_contact_relaxation=0.4,
            particle_self_contact_radius=0.05,
            particle_self_contact_margin=0.08,
            particle_triangle_contact_buffer_size=64,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.08,
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
            pick_stiffness=70.0,
            pick_damping=12.0,
            pick_max_acceleration=5.0,
            particle_pick_radius=0.3,
            particle_pick_stiffness=3600.0,
            particle_pick_damping=120.0,
            particle_pick_max_acceleration=100.0,
        )
        self.viewer.set_camera(wp.vec3(4.2, -7.0, 3.2), -10.0, 122.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 55.0
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
        """Keep the passive scene finite and the cloth support fixed."""
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        np.testing.assert_allclose(particle_q[self.pinned_particles], self.pinned_positions, atol=1.0e-6)
        assert np.all(np.isfinite(particle_q)), "cloth positions must remain finite"
        assert np.all(np.isfinite(body_q)), "rigid transforms must remain finite"
        assert float(np.min(particle_q[:, 2])) > -0.02, "cloth must remain above the ground"
        assert float(np.min(body_q[:, 2])) > 0.15, "rigid bodies must remain above the ground"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
