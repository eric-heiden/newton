# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Tricycle Origin Offset
#
# Drives the same floating-base tricycle twice with SolverFeatherstone: once
# near the world origin and once translated far away. The final test compares
# the two trajectories after subtracting the initial offset, probing whether
# world-origin-referenced Featherstone intermediates still affect numerics.
#
# Command: python -m newton.examples robot_tricycle_origin_offset --far-offset 100
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


CHASSIS_HALF_EXTENTS = (0.52, 0.28, 0.08)
CHASSIS_MASS = 9.0
CHASSIS_Z = 0.30
WHEEL_RADIUS = 0.18
WHEEL_HALF_WIDTH = 0.035
WHEEL_MASS = 0.7
WHEEL_AXLE_Z = WHEEL_RADIUS + 0.015
WHEEL_LOCAL_POSITIONS = (
    ("front_wheel", 0.48, 0.0, WHEEL_AXLE_Z - CHASSIS_Z),
    ("rear_left_wheel", -0.42, 0.25, WHEEL_AXLE_Z - CHASSIS_Z),
    ("rear_right_wheel", -0.42, -0.25, WHEEL_AXLE_Z - CHASSIS_Z),
)
CAMERA_TARGET_OFFSET = np.array((0.2, 0.0, 0.2), dtype=np.float32)
CAMERA_CHASE_OFFSET = np.array((-2.2, -3.0, 1.2), dtype=np.float32)


def box_inertia(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    """Return a box inertia tensor about its center of mass."""
    x = 2.0 * hx
    y = 2.0 * hy
    z = 2.0 * hz
    ixx = mass * (y * y + z * z) / 12.0
    iyy = mass * (x * x + z * z) / 12.0
    izz = mass * (x * x + y * y) / 12.0
    return wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)


def wheel_inertia(mass: float, radius: float, half_width: float) -> wp.mat33:
    """Return a cylinder inertia tensor with its axle along the body Y axis."""
    width = 2.0 * half_width
    i_axle = 0.5 * mass * radius * radius
    i_transverse = mass * (3.0 * radius * radius + width * width) / 12.0
    return wp.mat33(i_transverse, 0.0, 0.0, 0.0, i_axle, 0.0, 0.0, 0.0, i_transverse)


def camera_angles_z_up(pos: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return pitch and yaw for a Z-up camera looking from pos to target."""
    direction = target - pos
    distance = float(np.linalg.norm(direction.astype(np.float64)))
    if distance <= 1.0e-6:
        return 0.0, 0.0

    direction = direction / distance
    pitch = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    return pitch, yaw


def make_shape_cfg(*, mu: float, color_density: float = 0.0) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig(
        density=color_density,
        ke=2.0e4,
        kd=2.0e2,
        kf=2.0e3,
        mu=mu,
        restitution=0.0,
    )
    return cfg


def add_tricycle(
    builder: newton.ModelBuilder,
    origin: wp.vec3,
    label: str,
) -> tuple[list[int], list[int], list[int]]:
    """Add one floating-base tricycle and return bodies, joints, and driven joints."""
    chassis_cfg = make_shape_cfg(mu=0.8)
    wheel_cfg = make_shape_cfg(mu=1.25)
    chassis_color = (0.22, 0.45, 0.84) if label == "near" else (0.9, 0.45, 0.18)
    wheel_color = (0.07, 0.07, 0.08)

    hx, hy, hz = CHASSIS_HALF_EXTENTS
    chassis = builder.add_link(
        xform=wp.transform(p=origin + wp.vec3(0.0, 0.0, CHASSIS_Z), q=wp.quat_identity()),
        mass=CHASSIS_MASS,
        inertia=box_inertia(CHASSIS_MASS, hx, hy, hz),
        lock_inertia=True,
        label=f"{label}_chassis",
    )
    builder.add_shape_box(
        chassis,
        hx=hx,
        hy=hy,
        hz=hz,
        cfg=chassis_cfg,
        color=chassis_color,
        label=f"{label}_chassis_shape",
    )

    root_joint = builder.add_joint_free(chassis, label=f"{label}_root")
    bodies = [chassis]
    joints = [root_joint]
    driven_joints = []

    wheel_shape_xform = wp.transform(
        q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
    )

    for wheel_label, x, y, z in WHEEL_LOCAL_POSITIONS:
        wheel_body = builder.add_link(
            xform=wp.transform(
                p=origin + wp.vec3(x, y, CHASSIS_Z + z),
                q=wp.quat_identity(),
            ),
            mass=WHEEL_MASS,
            inertia=wheel_inertia(WHEEL_MASS, WHEEL_RADIUS, WHEEL_HALF_WIDTH),
            lock_inertia=True,
            label=f"{label}_{wheel_label}",
        )
        builder.add_shape_cylinder(
            wheel_body,
            xform=wheel_shape_xform,
            radius=WHEEL_RADIUS,
            half_height=WHEEL_HALF_WIDTH,
            cfg=wheel_cfg,
            color=wheel_color,
            label=f"{label}_{wheel_label}_shape",
        )
        wheel_joint = builder.add_joint_revolute(
            parent=chassis,
            child=wheel_body,
            parent_xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
            axis=newton.Axis.Y,
            armature=0.01,
            label=f"{label}_{wheel_label}_axle",
        )
        bodies.append(wheel_body)
        joints.append(wheel_joint)
        if "rear" in wheel_label:
            driven_joints.append(wheel_joint)

    builder.add_articulation(joints, label=f"{label}_tricycle")
    return bodies, joints, driven_joints


def create_tricycle_model(
    origin: np.ndarray,
    label: str,
) -> tuple[newton.Model, list[int], list[int], list[int]]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg.armature = 0.02
    builder.default_shape_cfg.ke = 2.0e4
    builder.default_shape_cfg.kd = 2.0e2
    builder.default_shape_cfg.kf = 2.0e3
    builder.default_shape_cfg.mu = 1.0
    builder.add_ground_plane()

    bodies, joints, driven_joints = add_tricycle(builder, wp.vec3(*origin), label)
    return builder.finalize(), bodies, joints, driven_joints


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.far_offset = float(args.far_offset)
        self.invariance_tolerance = float(args.invariance_tolerance)
        self.print_metrics = bool(args.print_metrics)

        self.near_origin = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        self.far_origin = np.array((self.far_offset, 0.0, 0.0), dtype=np.float32)
        self.origin_delta = self.far_origin - self.near_origin

        self.model, self.near_bodies, self.near_joints, near_driven = create_tricycle_model(
            self.near_origin, "near"
        )
        self.far_model, self.far_bodies, self.far_joints, far_driven = create_tricycle_model(
            self.far_origin, "far"
        )

        self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=0.02)
        self.far_solver = newton.solvers.SolverFeatherstone(self.far_model, angular_damping=0.02)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.far_state_0 = self.far_model.state()
        self.far_state_1 = self.far_model.state()
        self.far_control = self.far_model.control()
        self.far_contacts = self.far_model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.far_model, self.far_model.joint_q, self.far_model.joint_qd, self.far_state_0)

        qd_starts = self.model.joint_qd_start.numpy()
        joint_f = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for joint_idx in near_driven:
            joint_f[int(qd_starts[joint_idx])] = float(args.drive_torque)
        self.control.joint_f.assign(joint_f)
        self.far_control.joint_f.assign(joint_f)

        self.initial_body_q = self.state_0.body_q.numpy().copy()
        self.far_initial_body_q = self.far_state_0.body_q.numpy().copy()

        self.viewer.set_model(self.model)
        self.update_camera()

        self.capture()

    def update_camera(self):
        root_pos = self.state_0.body_q.numpy()[self.near_bodies[0], :3]
        target = root_pos + CAMERA_TARGET_OFFSET
        pos = root_pos + CAMERA_CHASE_OFFSET
        pitch, yaw = camera_angles_z_up(pos, target)
        self.viewer.set_camera(pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), pitch=pitch, yaw=yaw)

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.far_state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.far_model.collide(self.far_state_0, self.far_contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.far_solver.step(
                self.far_state_0,
                self.far_state_1,
                self.far_control,
                self.far_contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.far_state_0, self.far_state_1 = self.far_state_1, self.far_state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
            self.sim_time += self.frame_dt

    def render(self):
        self.update_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        far_body_q = self.far_state_0.body_q.numpy()
        far_body_qd = self.far_state_0.body_qd.numpy()
        far_joint_q = self.far_state_0.joint_q.numpy()
        far_joint_qd = self.far_state_0.joint_qd.numpy()

        assert np.isfinite(body_q).all(), "body_q contains non-finite values"
        assert np.isfinite(body_qd).all(), "body_qd contains non-finite values"
        assert np.isfinite(joint_q).all(), "joint_q contains non-finite values"
        assert np.isfinite(joint_qd).all(), "joint_qd contains non-finite values"
        assert np.isfinite(far_body_q).all(), "far body_q contains non-finite values"
        assert np.isfinite(far_body_qd).all(), "far body_qd contains non-finite values"
        assert np.isfinite(far_joint_q).all(), "far joint_q contains non-finite values"
        assert np.isfinite(far_joint_qd).all(), "far joint_qd contains non-finite values"

        near_root = self.near_bodies[0]
        far_root = self.far_bodies[0]
        near_delta = body_q[near_root, :3] - self.initial_body_q[near_root, :3]
        far_delta = far_body_q[far_root, :3] - self.far_initial_body_q[far_root, :3]
        root_delta_error = float(np.linalg.norm((far_delta - near_delta).astype(np.float64)))

        near_local = body_q[self.near_bodies, :3] - body_q[near_root, :3]
        far_local = far_body_q[self.far_bodies, :3] - far_body_q[far_root, :3]
        local_pose_error = float(np.max(np.linalg.norm((far_local - near_local).astype(np.float64), axis=1)))

        velocity_error = float(
            np.max(
                np.linalg.norm(
                    (far_body_qd[self.far_bodies] - body_qd[self.near_bodies]).astype(np.float64),
                    axis=1,
                )
            )
        )
        if self.print_metrics:
            print(
                "Origin-offset probe: "
                f"offset={self.far_offset:.6g} m, "
                f"near_dx={near_delta[0]:.6g} m, "
                f"far_dx={far_delta[0]:.6g} m, "
                f"root_delta_error={root_delta_error:.6g} m, "
                f"local_pose_error={local_pose_error:.6g} m, "
                f"velocity_error={velocity_error:.6g}"
            )

        assert near_delta[0] > 0.01, f"near tricycle did not roll forward enough: dx={near_delta[0]:.6f}"
        assert root_delta_error < self.invariance_tolerance, (
            f"far root displacement diverged from near root by {root_delta_error:.6g} m "
            f"at offset {self.far_offset:.6g} m"
        )
        assert local_pose_error < self.invariance_tolerance, (
            f"far local body layout diverged from near by {local_pose_error:.6g} m "
            f"at offset {self.far_offset:.6g} m"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--far-offset",
            type=float,
            default=100.0,
            help="World-space X offset [m] for the translated tricycle.",
        )
        parser.add_argument(
            "--drive-torque",
            type=float,
            default=0.7,
            help="Rear wheel drive torque [N*m].",
        )
        parser.add_argument(
            "--invariance-tolerance",
            type=float,
            default=0.25,
            help="Allowed near-vs-far trajectory difference after subtracting the initial offset.",
        )
        parser.add_argument(
            "--print-metrics",
            action="store_true",
            help="Print final near-vs-far trajectory error metrics.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
