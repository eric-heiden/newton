# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A Franka arm endlessly carries a transparent cup of SPH water between two
spots on the floor (A -> B -> A ...). The arm is driven kinematically by IK
waypoints; the cup is attached to the gripper while carried, and the water is
a live SPH fluid colliding with the cup walls. A GUI slider scales the robot
speed: crank it up and the water's inertia makes it slosh over the rim."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverSPH


@wp.kernel
def set_cup_state(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_index: int,
    xform: wp.transform,
    linear_velocity: wp.vec3,
):
    body_q[body_index] = xform
    body_qd[body_index] = wp.spatial_vector(linear_velocity, wp.vec3(0.0))


class Example:
    # phase schedule: (name, duration [s] at speed 1)
    PHASES = (
        ("move_above_pick", 1.6),
        ("descend_pick", 1.0),
        ("grasp", 0.6),
        ("lift", 1.0),
        ("traverse", 2.0),
        ("descend_place", 1.0),
        ("release", 0.6),
        ("ascend", 1.0),
    )

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.speed = args.speed

        self.spot_a = np.array(args.spot_a, dtype=np.float32)
        self.spot_b = np.array(args.spot_b, dtype=np.float32)
        self.lift_height = args.lift_height
        # the gripper pinches the cup wall at the rim from above; the hand
        # frame sits ~0.105 m above the fingertips
        self.fingertip_offset = 0.105
        self.rim_depth = 0.03
        self.grasp_radius = args.cup_inner_radius + 0.004

        # cup geometry: octagonal wall of thin boxes plus a bottom plate
        self.cup_inner_radius = args.cup_inner_radius
        self.cup_height = args.cup_height
        wall_thickness = 0.010

        # IK runs on a robot-only model
        urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        ik_builder = newton.ModelBuilder()
        ik_builder.add_urdf(urdf, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), enable_self_collisions=False)
        self.ik_model = ik_builder.finalize()

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.default_particle_radius = args.radius
        builder.add_urdf(urdf, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), enable_self_collisions=False)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.4), color=(0.55, 0.56, 0.54))

        self.arm_dofs = 7
        self.robot_coords = self.ik_model.joint_coord_count

        self.cup_body = builder.add_body(
            xform=wp.transform(wp.vec3(self.spot_a[0], self.spot_a[1], 0.0), wp.quat_identity()),
            label="cup",
        )
        cup_mesh = self._build_cup_mesh(self.cup_inner_radius, wall_thickness, self.cup_height)
        builder.add_shape_mesh(
            self.cup_body,
            mesh=cup_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.3),
            color=(0.75, 0.85, 0.95),
            opacity=args.cup_opacity,
        )

        # water column inside the cup (wall boxes catch the grid corners)
        fill = self.cup_inner_radius * 1.35
        dim_xy = max(int(fill / args.spacing), 2)
        builder.add_particle_grid(
            pos=wp.vec3(
                float(self.spot_a[0] - 0.5 * (dim_xy - 1) * args.spacing),
                float(self.spot_a[1] - 0.5 * (dim_xy - 1) * args.spacing),
                3.0 * wall_thickness,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dim_xy,
            dim_y=dim_xy,
            dim_z=args.fill_layers,
            cell_x=args.spacing,
            cell_y=args.spacing,
            cell_z=args.spacing,
            mass=args.rest_density * args.spacing**3,
            jitter=0.0005,
            radius_mean=args.radius,
            radius_std=0.0,
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.sph_solver = SolverSPH(
            self.model,
            smoothing_length=args.smoothing_length,
            rest_density=args.rest_density,
            viscosity=0.005,
            cohesion=0.02,
            xsph_strength=0.04,
            dissipation=0.1,
            velocity_damping=0.01,
            bounds_lower=wp.vec3(-1.2, -1.2, 0.0),
            bounds_upper=wp.vec3(1.2, 1.2, 1.6),
            shape_collision_distance=args.radius,
            shape_friction=0.2,
            shape_adhesion=0.0,
            max_velocity=6.0,
            max_acceleration=120.0,
            pbf_iterations=3,
            pbf_artificial_pressure=0.002,
            shape_collision_body_feedback=False,
            max_diffuse_particles=args.foam_max_particles,
            diffuse_threshold=1.0,
            diffuse_lifetime=1.5,
            diffuse_buoyancy=1.0,
            render_update_interval=1,
        )

        self._setup_ik()

        # phase machine
        self.phase_index = 0
        self.phase_time = 0.0
        self.pick_spot = self.spot_a.copy()
        self.place_spot = self.spot_b.copy()
        self.cup_attached = False
        self.cup_pos = np.array([self.spot_a[0], self.spot_a[1], 0.0], dtype=np.float32)
        self.prev_cup_pos = self.cup_pos.copy()
        self.ee_pos = self.home_pos.copy()
        self._update_cup_state(self.cup_pos, np.zeros(3, dtype=np.float32))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_fluid = True
        self.viewer.show_fluid_diffuse = True
        self.fluid_color = tuple(args.fluid_color)
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            sun = np.array((0.5, -0.35, 0.75), dtype=np.float32)
            renderer._sun_direction = sun / np.linalg.norm(sun)
            renderer.sky_upper = (0.55, 0.75, 0.95)
            renderer.sky_lower = (0.78, 0.82, 0.80)
            renderer.ambient_sky = (0.92, 0.96, 1.0)
            renderer.ambient_ground = (0.52, 0.54, 0.56)
            if hasattr(renderer, "exposure"):
                renderer.exposure = 1.1
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.gui, position="side")

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._update_cup_state(self.cup_pos, np.zeros(3, dtype=np.float32))

    @staticmethod
    def _build_cup_mesh(inner_radius, wall_thickness, height, segments=40):
        """Closed solid of revolution: cylindrical cup with an open cavity."""
        ri = inner_radius
        ro = inner_radius + wall_thickness
        t = wall_thickness
        # profile from the bottom-outer edge up and over the rim into the cavity
        profile = [(ro, 0.0), (ro, height), (ri, height), (ri, t)]
        vertices = []
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            c, sn = np.cos(angle), np.sin(angle)
            for r, z in profile:
                vertices.append((r * c, r * sn, z))
        bottom_center = len(vertices)
        vertices.append((0.0, 0.0, 0.0))
        cavity_center = len(vertices)
        vertices.append((0.0, 0.0, t))

        rows = len(profile)
        indices = []
        for i in range(segments):
            j = (i + 1) % segments
            for k in range(rows - 1):
                a = i * rows + k
                b = i * rows + k + 1
                c0 = j * rows + k
                d = j * rows + k + 1
                # outward-facing winding along the revolved strips
                indices += [a, c0, b, b, c0, d]
            # bottom disk (faces down) and cavity floor (faces up)
            indices += [i * rows + 0, bottom_center, j * rows + 0]
            indices += [i * rows + rows - 1, j * rows + rows - 1, cavity_center]

        return newton.Mesh(
            np.asarray(vertices, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
        )

    def _setup_ik(self):
        self.ee_index = 11  # fr3 hand link
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, self.ik_model.state())
        ik_state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, ik_state)
        ee_tf = wp.transform(*ik_state.body_q.numpy()[self.ee_index])
        self.ee_rotation = np.array(wp.transform_get_rotation(ee_tf), dtype=np.float32)
        self.home_pos = np.array(wp.transform_get_translation(ee_tf), dtype=np.float32)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*self.home_pos)], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*self.ee_rotation)], dtype=wp.vec4),
        )
        limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
        )
        self.joint_q_ik = wp.clone(self.ik_model.joint_q.reshape((1, -1)))
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    # ------------------------------------------------------------------
    def _phase_targets(self, name):
        """Returns (ee_start, ee_end, gripper_start, gripper_end) for a phase."""
        grasp_z = self.cup_height - self.rim_depth + self.fingertip_offset
        high_z = self.cup_height + self.lift_height
        pick_high = np.array([self.pick_spot[0] + self.grasp_radius, self.pick_spot[1], high_z], np.float32)
        pick_low = np.array([self.pick_spot[0] + self.grasp_radius, self.pick_spot[1], grasp_z], np.float32)
        place_high = np.array([self.place_spot[0] + self.grasp_radius, self.place_spot[1], high_z], np.float32)
        place_low = np.array([self.place_spot[0] + self.grasp_radius, self.place_spot[1], grasp_z], np.float32)
        g_open, g_closed = 0.04, 0.005
        if name == "move_above_pick":
            return self.ee_pos.copy(), pick_high, g_open, g_open
        if name == "descend_pick":
            return pick_high, pick_low, g_open, g_open
        if name == "grasp":
            return pick_low, pick_low, g_open, g_closed
        if name == "lift":
            return pick_low, pick_high, g_closed, g_closed
        if name == "traverse":
            return pick_high, place_high, g_closed, g_closed
        if name == "descend_place":
            return place_high, place_low, g_closed, g_closed
        if name == "release":
            return place_low, place_low, g_closed, g_open
        return place_low, place_high, g_open, g_open  # ascend

    def _update_cup_state(self, position, velocity):
        # eval_fk reposes the cup from its (stale) free-joint coordinates, so
        # both simulation states must be overwritten or the fluid collides
        # against a ghost cup left at the previous location.
        for state in (self.state_0, self.state_1):
            wp.launch(
                set_cup_state,
                dim=1,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    self.cup_body,
                    wp.transform(wp.vec3(*[float(v) for v in position]), wp.quat_identity()),
                    wp.vec3(*[float(v) for v in velocity]),
                ],
                device=self.model.device,
            )

    def step(self):
        name, duration = self.PHASES[self.phase_index]
        self.phase_time += self.frame_dt * self.speed
        t = min(self.phase_time / duration, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)

        ee_start, ee_end, g_start, g_end = self._phase_targets(name)
        ee_target = ee_start * (1.0 - smooth_t) + ee_end * smooth_t
        gripper = g_start * (1.0 - smooth_t) + g_end * smooth_t

        # IK toward the interpolated waypoint
        self.pos_obj.set_target_positions(wp.array([wp.vec3(*[float(v) for v in ee_target])], dtype=wp.vec3))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=12)

        # apply the IK solution kinematically: arm joints + cosmetic fingers
        joint_q = self.model.joint_q.numpy()
        ik_q = self.joint_q_ik.numpy()[0]
        joint_q[: self.arm_dofs] = ik_q[: self.arm_dofs]
        joint_q[self.arm_dofs : self.robot_coords] = gripper
        self.model.joint_q.assign(joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        # carry or rest the cup
        if name == "grasp" and t >= 1.0:
            self.cup_attached = True
        if name == "release" and t >= 0.5:
            self.cup_attached = False
        if self.cup_attached:
            new_pos = np.array(
                [
                    ee_target[0] - self.grasp_radius,
                    ee_target[1],
                    ee_target[2] - self.fingertip_offset - self.cup_height + self.rim_depth,
                ],
                np.float32,
            )
        else:
            new_pos = self.cup_pos.copy()
            new_pos[2] = 0.0
        cup_vel = (new_pos - self.prev_cup_pos) / self.frame_dt
        self.prev_cup_pos = self.cup_pos.copy()
        self.cup_pos = new_pos
        self._update_cup_state(new_pos, cup_vel)
        self.ee_pos = ee_target

        # fluid substeps against the kinematically posed shapes
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.sph_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_q, self.state_1.particle_q = self.state_1.particle_q, self.state_0.particle_q
            self.state_0.particle_qd, self.state_1.particle_qd = self.state_1.particle_qd, self.state_0.particle_qd

        if t >= 1.0:
            self.phase_index = (self.phase_index + 1) % len(self.PHASES)
            self.phase_time = 0.0
            if self.phase_index == 0:
                self.pick_spot, self.place_spot = self.place_spot, self.pick_spot

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        show_fluid = self.viewer.show_fluid
        if show_fluid:
            self.viewer.show_fluid = False
        try:
            self.viewer.log_state(self.state_0)
        finally:
            self.viewer.show_fluid = show_fluid
        self._log_fluid()
        self.viewer.end_frame()

    def _log_fluid(self):
        solver = self.sph_solver
        if (
            not self.viewer.show_fluid
            or getattr(self.viewer, "fluids", None) is None
            or not solver.render_buffers_valid
            or solver.render_positions is None
        ):
            return
        self.viewer.log_fluid(
            "/model/fluid",
            solver.render_positions,
            radii=self.model.particle_radius,
            radius_scale=1.8,
            color=self.fluid_color,
            blur_radius_world=0.02,
            anisotropy=solver.render_anisotropy,
            anisotropy_secondary=solver.render_anisotropy_secondary,
            anisotropy_tertiary=solver.render_anisotropy_tertiary,
            hidden=False,
        )
        if self.viewer.show_fluid_diffuse and solver.diffuse_positions is not None:
            self.viewer.log_fluid_diffuse(
                "/model/fluid/diffuse",
                solver.diffuse_positions,
                solver.diffuse_velocities,
                radius=0.006,
                lifetime=1.5,
                surface_bias=0.02,
                hidden=False,
            )

    def gui(self, ui):
        _, self.speed = ui.slider_float("Robot Speed", self.speed, 0.25, 5.0, "%.2f")
        ui.text("Crank up the speed to spill the water.")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("SPH particles contain non-finite positions")
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("Bodies contain non-finite transforms")
        # at default speed the water should still be carried with the cup
        active = (self.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        heights = q[active][:, 2]
        if heights.max() < 0.005:
            raise ValueError("All water ended on the floor; the cup carry failed")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--speed", type=float, default=1.0)
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--spot-a", type=float, nargs=2, default=(0.5, -0.3))
        parser.add_argument("--spot-b", type=float, nargs=2, default=(0.5, 0.3))
        parser.add_argument("--lift-height", type=float, default=0.32)
        parser.add_argument("--cup-inner-radius", type=float, default=0.05)
        parser.add_argument("--cup-height", type=float, default=0.13)
        parser.add_argument("--cup-opacity", type=float, default=0.35)

        parser.add_argument("--spacing", type=float, default=0.011)
        parser.add_argument("--radius", type=float, default=0.0083)
        parser.add_argument("--smoothing-length", type=float, default=0.0237)
        parser.add_argument("--fill-layers", type=int, default=8)
        parser.add_argument("--rest-density", type=float, default=460.0)
        parser.add_argument("--foam-max-particles", type=int, default=2000)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.45, -0.85, 0.75))
        parser.add_argument("--camera-pitch", type=float, default=-20.0)
        parser.add_argument("--camera-yaw", type=float, default=150.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
