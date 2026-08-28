# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD Cloth Gripper
#
# Four parallel grippers close on free cloth, lift it, and release it. The
# stations exercise near-zero and zero jaw gaps, flat and folded cloth, and a
# frictionless control that distinguishes a contact-supported pinch from
# adhesion or geometric attachment.
#
# Command: python -m newton.examples cloth_xpbd_gripper
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.func
def gripper_motion(time: float, target_half_gap: float, base_height: float):
    half_gap = 0.11
    height = base_height
    if time < 0.5:
        half_gap = wp.lerp(0.11, target_half_gap, 2.0 * time)
    elif time < 1.0:
        half_gap = target_half_gap
    elif time < 2.0:
        half_gap = target_half_gap
        height = wp.lerp(base_height, base_height + 0.35, time - 1.0)
    elif time < 3.0:
        half_gap = target_half_gap
        height = base_height + 0.35
    elif time < 3.5:
        half_gap = wp.lerp(target_half_gap, 0.2, 2.0 * (time - 3.0))
        height = base_height + 0.35
    else:
        half_gap = 0.2
        height = base_height + 0.35
    return wp.vec2(half_gap, height)


@wp.kernel
def prescribe_grippers(
    time: wp.array[float],
    dt: float,
    bodies: wp.array[int],
    station_x: wp.array[float],
    target_half_gap: wp.array[float],
    base_height: float,
    body_q_0: wp.array[wp.transform],
    body_qd_0: wp.array[wp.spatial_vector],
    body_q_1: wp.array[wp.transform],
    body_qd_1: wp.array[wp.spatial_vector],
):
    paddle = wp.tid()
    station = paddle // 2
    side = float((paddle % 2) * 2 - 1)
    body = bodies[paddle]
    motion_0 = gripper_motion(time[0], target_half_gap[station], base_height)
    motion_1 = gripper_motion(time[0] + dt, target_half_gap[station], base_height)
    p_0 = wp.vec3(station_x[station], side * motion_0[0], motion_0[1])
    p_1 = wp.vec3(station_x[station], side * motion_1[0], motion_1[1])
    velocity = (p_1 - p_0) / dt
    body_q_0[body] = wp.transform(p_0, wp.quat_identity())
    body_q_1[body] = wp.transform(p_1, wp.quat_identity())
    body_qd_0[body] = wp.spatial_vector(velocity[0], velocity[1], velocity[2], 0.0, 0.0, 0.0)
    body_qd_1[body] = body_qd_0[body]


@wp.kernel
def prescribe_gravity(time: wp.array[float], gravity: wp.array[wp.vec3]):
    world = wp.tid()
    if time[0] < 0.55:
        gravity[world] = wp.vec3(0.0)
    else:
        gravity[world] = wp.vec3(0.0, 0.0, -9.81)


@wp.kernel
def advance_time(time: wp.array[float], dt: float):
    time[0] += dt


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.dim_x = 12
        self.dim_y = 16
        self.cell_size = 0.04
        self.cloth_base_height = 0.45
        self.gripper_base_height = 0.87
        self.station_x = np.array((-1.5, -0.5, 0.5, 1.5), dtype=np.float32)
        self.case_names = (
            "near-closed flat",
            "fully closed wrinkled",
            "fully closed folded",
            "frictionless control",
        )
        surface_gaps = np.array((0.006, 0.0, 0.0, 0.0), dtype=np.float32)
        self.target_half_gaps = 0.5 * surface_gaps
        self.valid_grasp_cases = np.arange(3, dtype=np.int32)
        self.control_case = 3

        builder = newton.ModelBuilder()
        self.cloth_particles = []
        for station, center_x in enumerate(self.station_x):
            first_particle = len(builder.particle_q)
            builder.add_cloth_grid(
                pos=wp.vec3(center_x - 0.5 * self.dim_x * self.cell_size, 0.0, self.cloth_base_height),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi),
                vel=wp.vec3(0.0),
                dim_x=self.dim_x,
                dim_y=self.dim_y,
                cell_x=self.cell_size,
                cell_y=self.cell_size,
                mass=5.0e-4,
                tri_ke=8.0e3,
                tri_ka=8.0e3,
                tri_kd=0.0,
                edge_ke=0.02,
                edge_kd=0.0,
                particle_radius=0.01,
                label=self.case_names[station],
            )
            self.cloth_particles.append(np.arange(first_particle, len(builder.particle_q), dtype=np.int32))

        loaded_paddle_cfg = builder.default_shape_cfg.copy()
        loaded_paddle_cfg.ke = 1.0e4
        loaded_paddle_cfg.kd = 100.0
        loaded_paddle_cfg.mu = 3.0
        loaded_paddle_cfg.ka = 0.0
        loaded_paddle_cfg.density = 0.0
        control_paddle_cfg = loaded_paddle_cfg.copy()
        control_paddle_cfg.mu = 0.0

        self.gripper_bodies = []
        self.gripper_shapes = []
        for station, center_x in enumerate(self.station_x):
            station_bodies = []
            station_shapes = []
            paddle_cfg = control_paddle_cfg if station == self.control_case else loaded_paddle_cfg
            for side in (-1.0, 1.0):
                body = builder.add_link(
                    xform=wp.transform(wp.vec3(center_x, side * 0.11, self.gripper_base_height), wp.quat_identity()),
                    is_kinematic=True,
                    label=f"{self.case_names[station]} paddle",
                )
                shape = builder.add_shape_plane(
                    body=body,
                    xform=wp.transform(
                        wp.vec3(0.0),
                        wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), side * 0.5 * wp.pi),
                    ),
                    width=0.44,
                    length=0.36,
                    cfg=paddle_cfg,
                )
                station_bodies.append(body)
                station_shapes.append(shape)
            self.gripper_bodies.append(station_bodies)
            self.gripper_shapes.append(station_shapes)

        self.model = builder.finalize()
        # Put all Coulomb friction on the paddle material so the last station is
        # an exact zero-friction control while sharing the same cloth model.
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0
        self.model.particle_adhesion = 0.0
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            integrate_with_external_rigid_solver=True,
            soft_contact_max_depenetration_velocity=0.3,
            particle_enable_self_contact=True,
            particle_enable_triangle_intersection_recovery=True,
            particle_self_contact_relaxation=0.3,
            particle_triangle_intersection_relaxation=0.3,
            particle_self_contact_radius=0.012,
            particle_self_contact_margin=0.02,
            particle_max_depenetration_velocity=0.2,
            particle_vertex_contact_buffer_size=48,
            particle_edge_contact_buffer_size=96,
            particle_triangle_contact_buffer_size=48,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.025,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_1.body_qd.assign(self.state_0.body_qd)

        initial_positions = self.state_0.particle_q.numpy()
        stride = self.dim_x + 1
        wrinkled = self.cloth_particles[1].reshape(self.dim_y + 1, stride)
        for row in range(self.dim_y + 1):
            for column in range(stride):
                initial_positions[wrinkled[row, column], 1] = 0.008 * np.sin(
                    2.0 * np.pi * column / self.dim_x + 0.35 * row
                )

        folded = self.cloth_particles[2].reshape(self.dim_y + 1, stride)
        fold_start = 10
        fold_segments = 3
        for row in range(fold_start + 1, self.dim_y + 1):
            folded_height = 0.0
            folded_z = fold_start * self.cell_size
            transition_segments = min(row - fold_start, fold_segments)
            for segment in range(1, transition_segments + 1):
                angle = np.pi * segment / fold_segments
                folded_z += self.cell_size * np.cos(angle)
                folded_height += self.cell_size * np.sin(angle)
            folded_z -= max(row - fold_start - fold_segments, 0) * self.cell_size
            initial_positions[folded[row], 1] = -folded_height
            initial_positions[folded[row], 2] = self.cloth_base_height + folded_z

        self.state_0.particle_q.assign(initial_positions)
        self.state_1.particle_q.assign(initial_positions)
        self.initial_heights = np.array(
            [np.median(initial_positions[particles, 2]) for particles in self.cloth_particles], dtype=np.float32
        )
        self.grasp_particles = []
        for station, particles in enumerate(self.cloth_particles):
            grid = particles.reshape(self.dim_y + 1, stride)
            first_row = 7 if station == 2 else 8
            self.grasp_particles.append(grid[first_row:15].ravel())

        flat_bodies = np.asarray(self.gripper_bodies, dtype=np.int32).ravel()
        self.gripper_body_array = wp.array(flat_bodies, dtype=wp.int32, device=self.model.device)
        self.station_x_array = wp.array(self.station_x, dtype=float, device=self.model.device)
        self.target_half_gap_array = wp.array(self.target_half_gaps, dtype=float, device=self.model.device)
        self.sim_time_array = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self._test_contact_loads = None
        self._test_pre_lift_heights = None
        self._test_lift_heights = None
        self._test_hold_heights = None
        self._test_release_contacts = None
        self._test_max_closed_speeds = np.zeros(len(self.case_names), dtype=np.float32)

        self.viewer.set_model(self.model)
        self.viewer.configure_picking(
            particle_pick_radius=0.2,
            particle_pick_stiffness=3600.0,
            particle_pick_damping=120.0,
            particle_pick_max_acceleration=100.0,
        )
        self.viewer.set_camera(wp.vec3(3.0, -4.2, 1.9), -8.0, 138.0)
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
            wp.launch(
                prescribe_gravity, dim=self.model.gravity.shape[0], inputs=[self.sim_time_array, self.model.gravity]
            )
            wp.launch(
                prescribe_grippers,
                dim=2 * len(self.case_names),
                inputs=[
                    self.sim_time_array,
                    self.sim_dt,
                    self.gripper_body_array,
                    self.station_x_array,
                    self.target_half_gap_array,
                    self.gripper_base_height,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.sim_time_array, self.sim_dt])

    def step(self):
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def _cloth_heights(self):
        positions = self.state_0.particle_q.numpy()
        return np.array([np.median(positions[particles, 2]) for particles in self.grasp_particles], dtype=np.float32)

    def test_post_step(self):
        """Measure contact loading, lift retention, stability, and release."""
        if 0.4 <= self.sim_time <= 3.0:
            speeds = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
            for station, particles in enumerate(self.cloth_particles):
                self._test_max_closed_speeds[station] = max(
                    self._test_max_closed_speeds[station], float(np.max(speeds[particles]))
                )

        if self._test_contact_loads is None and self.sim_time >= 0.8:
            contact_count = min(
                int(self.contacts.soft_contact_count.numpy()[0]), self.contacts.soft_contact_shape.shape[0]
            )
            shapes = self.contacts.soft_contact_shape.numpy()[:contact_count]
            lambdas = np.abs(self.solver.soft_contact_lambdas.numpy()[:contact_count])
            loads = []
            for left_shape, right_shape in self.gripper_shapes:
                left_load = float(np.sum(lambdas[shapes == left_shape]) / self.sim_dt**2)
                right_load = float(np.sum(lambdas[shapes == right_shape]) / self.sim_dt**2)
                loads.append((left_load, right_load))
            self._test_contact_loads = np.asarray(loads, dtype=np.float32)

        if self._test_pre_lift_heights is None and self.sim_time >= 0.9:
            self._test_pre_lift_heights = self._cloth_heights()
        if self._test_lift_heights is None and self.sim_time >= 2.5:
            self._test_lift_heights = self._cloth_heights()
        if self._test_hold_heights is None and self.sim_time >= 2.9:
            self._test_hold_heights = self._cloth_heights()
        if self._test_release_contacts is None and self.sim_time >= 4.0:
            contact_count = min(
                int(self.contacts.soft_contact_count.numpy()[0]), self.contacts.soft_contact_shape.shape[0]
            )
            shapes = self.contacts.soft_contact_shape.numpy()[:contact_count]
            gripper_shapes = np.asarray(self.gripper_shapes, dtype=np.int32).ravel()
            self._test_release_contacts = int(np.count_nonzero(np.isin(shapes, gripper_shapes)))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Support free cloth with loaded contacts through extreme jaw closure."""
        assert self._test_contact_loads is not None
        for station in self.valid_grasp_cases:
            assert np.min(self._test_contact_loads[station]) > 0.1, (
                f"{self.case_names[station]} must load both jaws; loads were {self._test_contact_loads[station]} N"
            )

        assert self._test_pre_lift_heights is not None and self._test_lift_heights is not None
        lift = self._test_lift_heights - self._test_pre_lift_heights
        for station in self.valid_grasp_cases:
            assert lift[station] > 0.22, (
                f"{self.case_names[station]} must lift the free cloth; lift was {lift[station]:.3f} m"
            )
        assert lift[self.control_case] < 0.08, (
            f"the zero-friction control must not follow the lift; lift was {lift[self.control_case]:.3f} m"
        )
        assert self._test_lift_heights[self.control_case] < self.initial_heights[self.control_case] - 0.2, (
            "the zero-friction cloth must fall under gravity instead of remaining geometrically attached"
        )

        assert self._test_hold_heights is not None
        hold_slip = np.abs(self._test_hold_heights - self._test_lift_heights)
        for station in self.valid_grasp_cases:
            assert hold_slip[station] < 0.08, (
                f"{self.case_names[station]} slipped {hold_slip[station]:.3f} m during the loaded hold"
            )
            assert self._test_max_closed_speeds[station] < 5.0, (
                f"{self.case_names[station]} became unstable at {self._test_max_closed_speeds[station]:.3f} m/s"
            )

        assert self._test_release_contacts == 0, (
            f"cloth must release cleanly; {self._test_release_contacts} gripper contacts remained"
        )
        assert not np.any(self.model.shape_material_ka.numpy()), "mechanical grasp must not use shape adhesion"
        assert self.model.particle_adhesion == 0.0, "mechanical grasp must not use particle adhesion"
        assert np.all(np.isfinite(self.state_0.particle_q.numpy())), "cloth positions must remain finite"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
