# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD Cloth Rigid Contact
#
# A cloth patch drops onto a rigid sphere and the ground. Shape adhesion is
# explicitly disabled so contact friction does not masquerade as sticking.
#
# Command: python -m newton.examples cloth_xpbd_rigid_contact
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
        self._test_resting_speed_p99_peak = 0.0

        builder = newton.ModelBuilder()
        contact_cfg = builder.default_shape_cfg.copy()
        contact_cfg.mu = 0.6
        contact_cfg.ka = 0.0
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.65), wp.quat_identity()),
            radius=0.6,
            cfg=contact_cfg,
        )
        builder.add_ground_plane(cfg=contact_cfg)
        builder.add_cloth_grid(
            pos=wp.vec3(-1.2, -1.2, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=24,
            dim_y=24,
            cell_x=0.1,
            cell_y=0.1,
            mass=1.5e-3,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=2.0,
            edge_ke=10.0,
            edge_kd=0.5,
            particle_radius=0.025,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.6
        self.model.soft_contact_mu = 0.6
        self.model.shape_material_ka.zero_()
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            particle_enable_self_contact=True,
            particle_enable_triangle_intersection_recovery=True,
            particle_self_contact_relaxation=0.5,
            particle_self_contact_radius=0.05,
            particle_self_contact_margin=0.08,
            particle_damping=2.0,
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

        self.viewer.set_model(self.model)
        self.viewer.configure_picking(
            particle_pick_radius=0.25,
            particle_pick_stiffness=3600.0,
            particle_pick_damping=120.0,
            particle_pick_max_acceleration=100.0,
        )
        self.viewer.set_camera(wp.vec3(3.5, -4.5, 3.0), -22.0, 128.0)
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Track residual motion after the drape should be at rest."""
        if self.sim_time >= 6.0:
            speeds = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
            self._test_resting_speed_p99_peak = max(
                self._test_resting_speed_p99_peak,
                float(np.percentile(speeds, 99.0)),
            )

    def test_final(self):
        """Drape over rigid geometry without penetration or adhesive sticking."""
        positions = self.state_0.particle_q.numpy()
        speeds = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
        center = positions[len(positions) // 2]
        assert np.all(np.isfinite(positions)), "cloth positions must remain finite"
        assert float(np.min(positions[:, 2])) > -0.02, "cloth must remain above the ground"
        assert float(center[2]) > 1.15, "cloth center must remain supported by the sphere"
        assert self._test_resting_speed_p99_peak < 0.15, (
            f"draped cloth must stay at rest; peak late p99 speed was {self._test_resting_speed_p99_peak} m/s"
        )
        assert float(np.percentile(speeds, 99.0)) < 0.03, "draped cloth must settle"
        assert not np.any(self.model.shape_material_ka.numpy()), "passive rigid shapes must have no adhesion"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # The skirt slides down the sphere under Coulomb friction before it reaches
    # the ground, so the drape needs roughly seven seconds to come to rest.
    parser.set_defaults(num_frames=420)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
