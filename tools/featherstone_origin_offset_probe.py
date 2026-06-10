# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe SolverFeatherstone origin-offset stability with a floating tricycle."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

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


def make_shape_cfg(mu: float, color_density: float = 0.0) -> newton.ModelBuilder.ShapeConfig:
    """Create a contact material matching the tricycle example."""
    return newton.ModelBuilder.ShapeConfig(
        density=color_density,
        ke=2.0e4,
        kd=2.0e2,
        kf=2.0e3,
        mu=mu,
        restitution=0.0,
    )


def add_tricycle(
    builder: newton.ModelBuilder,
    origin: wp.vec3,
    label: str,
) -> tuple[list[int], list[int], list[int]]:
    """Add one floating-base tricycle and return bodies, joints, and driven joints."""
    chassis_cfg = make_shape_cfg(mu=0.8)
    wheel_cfg = make_shape_cfg(mu=1.25)
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
    """Create a floating tricycle model at the requested world origin."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg.armature = 0.02
    builder.default_shape_cfg.ke = 2.0e4
    builder.default_shape_cfg.kd = 2.0e2
    builder.default_shape_cfg.kf = 2.0e3
    builder.default_shape_cfg.mu = 1.0
    builder.add_ground_plane()

    bodies, joints, driven_joints = add_tricycle(builder, wp.vec3(*origin), label)
    return builder.finalize(), bodies, joints, driven_joints


def finite_all(*arrays: np.ndarray) -> bool:
    """Return whether all arrays are finite."""
    return all(np.isfinite(array).all() for array in arrays)


def max_abs_or_nan(array: np.ndarray | None) -> float:
    """Return max absolute value, preserving NaN for non-finite arrays."""
    if array is None or array.size == 0:
        return 0.0
    if not np.isfinite(array).all():
        return float("nan")
    return float(np.max(np.abs(array)))


class Probe:
    """Headless tricycle origin-offset probe."""

    def __init__(self, args: argparse.Namespace):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.near_origin = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        self.far_origin = np.array((args.far_offset, 0.0, 0.0), dtype=np.float32)

        self.model, self.near_bodies, self.near_joints, near_driven = create_tricycle_model(
            self.near_origin, "near"
        )
        self.far_model, self.far_bodies, self.far_joints, _ = create_tricycle_model(self.far_origin, "far")

        self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=args.angular_damping)
        self.far_solver = newton.solvers.SolverFeatherstone(
            self.far_model, angular_damping=args.angular_damping
        )

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

    def step(self):
        """Advance one rendered frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.far_state_0.clear_forces()
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

    def sample(self, frame: int, elapsed: float) -> dict[str, float | int | bool]:
        """Collect scalar diagnostics."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        far_body_q = self.far_state_0.body_q.numpy()
        far_body_qd = self.far_state_0.body_qd.numpy()
        far_joint_q = self.far_state_0.joint_q.numpy()
        far_joint_qd = self.far_state_0.joint_qd.numpy()

        finite = finite_all(body_q, body_qd, joint_q, joint_qd, far_body_q, far_body_qd, far_joint_q, far_joint_qd)
        near_root = self.near_bodies[0]
        far_root = self.far_bodies[0]

        if finite:
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
            near_dx = float(near_delta[0])
            far_dx = float(far_delta[0])
            far_root_x = float(far_body_q[far_root, 0])
        else:
            root_delta_error = float("nan")
            local_pose_error = float("nan")
            velocity_error = float("nan")
            near_dx = float("nan")
            far_dx = float("nan")
            far_root_x = float("nan")

        near_h = self.solver.H.numpy() if hasattr(self.solver, "H") else None
        far_h = self.far_solver.H.numpy() if hasattr(self.far_solver, "H") else None
        near_j = self.solver.J.numpy() if hasattr(self.solver, "J") else None
        far_j = self.far_solver.J.numpy() if hasattr(self.far_solver, "J") else None
        near_m = self.solver.M.numpy() if hasattr(self.solver, "M") else None
        far_m = self.far_solver.M.numpy() if hasattr(self.far_solver, "M") else None

        return {
            "frame": frame,
            "elapsed_s": elapsed,
            "near_dx_m": near_dx,
            "far_dx_m": far_dx,
            "far_root_x_m": far_root_x,
            "root_delta_error_m": root_delta_error,
            "local_pose_error_m": local_pose_error,
            "velocity_error": velocity_error,
            "finite": finite,
            "near_J_max_abs": max_abs_or_nan(near_j),
            "far_J_max_abs": max_abs_or_nan(far_j),
            "near_M_max_abs": max_abs_or_nan(near_m),
            "far_M_max_abs": max_abs_or_nan(far_m),
            "near_H_max_abs": max_abs_or_nan(near_h),
            "far_H_max_abs": max_abs_or_nan(far_h),
            "near_J_finite": near_j is None or bool(np.isfinite(near_j).all()),
            "far_J_finite": far_j is None or bool(np.isfinite(far_j).all()),
            "near_M_finite": near_m is None or bool(np.isfinite(near_m).all()),
            "far_M_finite": far_m is None or bool(np.isfinite(far_m).all()),
            "near_H_finite": near_h is None or bool(np.isfinite(near_h).all()),
            "far_H_finite": far_h is None or bool(np.isfinite(far_h).all()),
        }


def run_probe(args: argparse.Namespace) -> dict:
    """Run the probe and return a JSON-serializable result."""
    if args.device:
        device = wp.get_device(args.device)
    else:
        device = wp.get_device()

    with wp.ScopedDevice(device):
        probe = Probe(args)
        samples = []
        first_nonfinite_frame = None

        for _ in range(args.warmup_frames):
            probe.step()

        start = time.perf_counter()
        for frame in range(1, args.frames + 1):
            probe.step()
            should_sample = args.sample_interval > 0 and frame % args.sample_interval == 0
            should_sample = should_sample or frame == 1 or frame == args.frames
            if should_sample:
                sample = probe.sample(frame + args.warmup_frames, time.perf_counter() - start)
                samples.append(sample)
                if not sample["finite"] and first_nonfinite_frame is None:
                    first_nonfinite_frame = frame + args.warmup_frames
                if args.stop_on_nonfinite and first_nonfinite_frame is not None:
                    break

        elapsed = time.perf_counter() - start
        simulated_frames = samples[-1]["frame"] - args.warmup_frames if samples else 0
        result = {
            "label": args.label,
            "device": str(device),
            "frames_requested": args.frames,
            "warmup_frames": args.warmup_frames,
            "frames_completed": simulated_frames,
            "substeps": args.substeps,
            "far_offset_m": args.far_offset,
            "drive_torque_Nm": args.drive_torque,
            "elapsed_s": elapsed,
            "fps": float(simulated_frames / elapsed) if elapsed > 0.0 else float("nan"),
            "first_nonfinite_frame": first_nonfinite_frame,
            "samples": samples,
        }
        return result


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="probe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--far-offset", type=float, default=100.0)
    parser.add_argument("--drive-torque", type=float, default=0.7)
    parser.add_argument("--angular-damping", type=float, default=0.02)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--stop-on-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    """Run the command-line probe."""
    args = create_parser().parse_args()
    result = run_probe(args)
    text = json.dumps(result, indent=2, allow_nan=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
