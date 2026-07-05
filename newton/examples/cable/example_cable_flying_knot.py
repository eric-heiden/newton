# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Flying Knot
#
# Replicates the "flying knot" from "Learning Dynamic Rope Manipulation
# Using Task-Level Iterative Learning Control" (Suresh & Atkeson, CMU,
# arXiv:2602.21302, https://flying-knots.github.io/): an overhand knot is
# tied mid-air by a single fast throw of a weighted rope.
#
# The throw motion is the paper's executed end-effector trajectory,
# digitized from the follow-through figure of the paper and refit with the
# paper's own 8-control-point Bezier command parametrization (Appendix D).
# An xArm7 (the paper's robot) bolted to a fixed pedestal tracks the
# handle trajectory with its 7 joints via Newton's IK module and is
# animated kinematically. (The paper's command also translates the robot
# base; here the mount is chosen so the fixed-base arm can reach the
# whole trajectory.) The rope is a chain of capsules coupled by cable joints,
# simulated with SolverVBD; its root capsule is driven kinematically along
# the recorded trajectory, mirroring the paper's position-driven particle
# rope model.
#
# Phases: settle -> throw (0.7 s) -> free flight -> slow lift to exhibit
# (and tighten) the knot. test_final() verifies knot topology via the
# polyline writhe and an end-to-end length deficit.
#
# Command: python -m newton.examples cable_flying_knot
###########################################################################

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# ---------------------------------------------------------------------------
# Recorded throw command
#
# 8-knot Bezier control points (paper Appendix D) fit to the end-effector
# position trajectory digitized from the follow-through figure at
# https://flying-knots.github.io/ (see scripts/flying_knot/digitize_fttraj.py).
# Columns are x, y, z in meters; the curve spans t in [0, T_THROW] seconds.
# ---------------------------------------------------------------------------
BEZIER_CTRL = np.array(
    [
        [0.7997, -0.0955, 0.1920],
        [0.4014, -0.1201, 0.2392],
        [2.1393, 0.0938, -0.3338],
        [-1.3780, -0.8675, 2.6763],
        [1.7898, 1.7733, -0.3805],
        [0.8518, -1.3020, 0.4817],
        [0.7160, 0.0488, -0.0500],
        [0.7729, -0.2453, 0.0154],
    ]
)
T_THROW = 0.7  # [s]

# Handle held by the robot: attachment flange to rope tip (paper handle CAD).
HANDLE_LENGTH = 0.15  # [m]


def bezier_eval(ctrl: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate a Bezier curve with Bernstein basis at s in [0, 1]."""
    n = len(ctrl) - 1
    s = np.asarray(s)[..., None]
    binom = np.array([math.comb(n, i) for i in range(n + 1)], dtype=np.float64)
    basis = binom * s ** np.arange(n + 1) * (1.0 - s) ** (n - np.arange(n + 1))
    return basis @ ctrl


# ---------------------------------------------------------------------------
# Knot metrics
# ---------------------------------------------------------------------------


def polyline_writhe(points: np.ndarray) -> float:
    """Discrete writhe of a polyline via the Gauss integral.

    Uses the segment-pair solid-angle formula (Klenin & Langowski 2000,
    method 1a). For a closed trefoil |Wr| is approximately 3.4; an open
    overhand (trefoil) knot in a hanging rope gives |Wr| well above 2,
    while unknotted configurations stay near 0.
    """
    p = np.asarray(points, dtype=np.float64)
    n = len(p) - 1
    wr = 0.0
    for i in range(n - 1):
        p1, p2 = p[i], p[i + 1]
        for j in range(i + 2, n):
            p3, p4 = p[j], p[j + 1]
            r13 = p3 - p1
            r14 = p4 - p1
            r23 = p3 - p2
            r24 = p4 - p2
            n1 = np.cross(r13, r14)
            n2 = np.cross(r14, r24)
            n3 = np.cross(r24, r23)
            n4 = np.cross(r23, r13)
            norms = [np.linalg.norm(v) for v in (n1, n2, n3, n4)]
            if min(norms) < 1e-12:
                continue
            n1, n2, n3, n4 = n1 / norms[0], n2 / norms[1], n3 / norms[2], n4 / norms[3]
            omega = (
                math.asin(np.clip(n1 @ n2, -1.0, 1.0))
                + math.asin(np.clip(n2 @ n3, -1.0, 1.0))
                + math.asin(np.clip(n3 @ n4, -1.0, 1.0))
                + math.asin(np.clip(n4 @ n1, -1.0, 1.0))
            )
            sign = np.sign(np.cross(p4 - p3, p2 - p1) @ r13)
            wr += omega * sign
    return wr / (2.0 * math.pi)


def count_crossings(points: np.ndarray, axis: int = 2) -> int:
    """Count crossings of the polyline projected along the given axis."""
    p = np.asarray(points, dtype=np.float64)
    dims = [d for d in range(3) if d != axis]
    q = p[:, dims]
    n = len(q) - 1
    crossings = 0
    for i in range(n - 1):
        a, b = q[i], q[i + 1]
        d1 = b - a
        for j in range(i + 2, n):
            c, d = q[j], q[j + 1]
            d2 = d - c
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) < 1e-14:
                continue
            t = ((c[0] - a[0]) * d2[1] - (c[1] - a[1]) * d2[0]) / denom
            u = ((c[0] - a[0]) * d1[1] - (c[1] - a[1]) * d1[0]) / denom
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                crossings += 1
    return crossings


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw layout, numpy)
# ---------------------------------------------------------------------------


def quat_from_z_to(direction: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) rotating local +Z onto the given unit direction."""
    z = np.array([0.0, 0.0, 1.0])
    d = direction / np.linalg.norm(direction)
    c = np.cross(z, d)
    w = 1.0 + z @ d
    if w < 1e-8:
        # 180 degree flip: rotate about x.
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([c[0], c[1], c[2], w])
    return q / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation (xyzw) taking unit vector a to unit vector b."""
    c = np.cross(a, b)
    w = 1.0 + a @ b
    if w < 1e-8:
        # Antipodal: pick any orthogonal axis.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    q = np.array([c[0], c[1], c[2], w])
    return q / np.linalg.norm(q)


def nlerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    if q0 @ q1 < 0.0:
        q1 = -q1
    q = (1.0 - t) * q0 + t * q1
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# xArm7 URDF preprocessing
# ---------------------------------------------------------------------------

XARM_GRIPPER_LINKS = {
    "xarm_gripper_base_link",
    "left_outer_knuckle",
    "left_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_finger",
    "right_inner_knuckle",
    "link_tcp",
    "world",
}


def preprocess_xarm_urdf(urdf_path: Path) -> str:
    """Strip ROS/gazebo/gripper cruft from the flying-knots xArm7 URDF.

    Returns URDF XML with absolute mesh paths, keeping link_base..link7 and
    link_eef only. The gripper is replaced in the scene by the rope handle.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    package_root = urdf_path.parent.parent  # directory containing xarm_description/

    for tag in ("ros2_control", "gazebo", "transmission"):
        for el in root.findall(tag):
            root.remove(el)

    def wanted(el):
        name = el.get("name", "")
        if "${" in name:
            return False
        if el.tag == "link":
            return name not in XARM_GRIPPER_LINKS
        if el.tag == "joint":
            parent = el.find("parent")
            child = el.find("child")
            links = {parent.get("link") if parent is not None else "", child.get("link") if child is not None else ""}
            return not (links & XARM_GRIPPER_LINKS)
        return True

    for el in list(root):
        if el.tag in ("link", "joint") and not wanted(el):
            root.remove(el)

    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            mesh.set("filename", str(package_root / fn.removeprefix("package://")))

    return ET.tostring(root, encoding="unicode")


def default_xarm_dir() -> Path | None:
    candidates = [
        os.environ.get("FLYING_KNOTS_XARM_DIR"),
        "~/repos/flying_knots_public/models/xarm_description",
    ]
    for c in candidates:
        if c and (Path(c).expanduser() / "xarm7.urdf").exists():
            return Path(c).expanduser()
    return None


# ---------------------------------------------------------------------------
# Kinematic driving kernels
# ---------------------------------------------------------------------------


@wp.kernel
def drive_kinematic_bodies(
    step_idx: wp.array[wp.int32],
    num_steps: int,
    body_indices: wp.array[wp.int32],
    schedule: wp.array2d[wp.transform],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = wp.min(step_idx[0], num_steps - 1)
    body = body_indices[tid]
    T = schedule[idx, tid]
    body_q0[body] = T
    body_q1[body] = T


@wp.kernel
def advance_step(step_idx: wp.array[wp.int32]):
    step_idx[0] = step_idx[0] + 1


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = getattr(args, "substeps", 32)
        self.sim_iterations = getattr(args, "iterations", 10)
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Throw command scaling
        self.time_scale = getattr(args, "time_scale", 0.8)
        self.throw_scale = getattr(args, "throw_scale", 1.0)
        self.z_offset = getattr(args, "z_offset", 1.1)

        # Rope parameters (paper: 1.1 m rope, weighted tip)
        self.rope_length = getattr(args, "rope_length", 1.1)
        self.rope_segments = getattr(args, "rope_segments", 36)
        self.rope_radius = getattr(args, "rope_radius", 0.005)
        self.rope_linear_density = getattr(args, "rope_density", 0.05)  # [kg/m]
        self.tip_mass = getattr(args, "tip_mass", 0.05)  # [kg]
        self.stretch_stiffness = getattr(args, "stretch_stiffness", 2.0e5)
        self.bend_stiffness = getattr(args, "bend_stiffness", 2.0e-3)
        self.bend_damping = getattr(args, "bend_damping", 1.0e-4)
        self.stretch_damping = getattr(args, "stretch_damping", 0.5)
        self.friction = getattr(args, "friction", 1.0)
        self.use_arm = not getattr(args, "no_arm", False)
        self.save_traj = getattr(args, "save_traj", None)
        self.expect_knot = getattr(args, "expect_knot", False)

        # Phase timing [s]
        self.t_settle = getattr(args, "t_settle", 1.5)
        self.t_throw = T_THROW * self.time_scale
        self.t_flight = getattr(args, "t_flight", 2.0)
        self.t_lift = getattr(args, "t_lift", 2.5)
        self.t_hold = getattr(args, "t_hold", 1.5)
        self.duration = self.t_settle + self.t_throw + self.t_flight + self.t_lift + self.t_hold
        self.num_frames = int(round(self.duration * self.fps))
        self.lift_height = getattr(args, "lift_height", 0.55)
        # Drift the lift away from the pedestal so the loose knot cannot snag
        # on the column while it tightens.
        self.lift_drift = getattr(args, "lift_drift", 0.12)

        # --- Handle-tip trajectory (position + rope-root axis) ------------
        self.n_sub_total = self.num_frames * self.sim_substeps
        tip_pos, tip_axis = self._build_handle_schedule(self.n_sub_total)
        self.tip_pos = tip_pos
        self.tip_axis = tip_axis
        self.root_quats = self._build_root_quats()

        # --- Build the scene ----------------------------------------------
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = self.friction

        kin_bodies: list[int] = []
        self.arm_body_count = 0
        if self.use_arm:
            self.arm_body_count = self._add_arm(builder)
            kin_bodies.extend(range(self.arm_body_count))

        # Handle: kinematic capsule from flange to rope attachment tip.
        handle_body = builder.add_link(
            xform=wp.transform(wp.vec3(*self._handle_com(0)), wp.quat(*self.root_quats[0])),
            label="handle",
        )
        builder.add_shape_capsule(
            body=handle_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            radius=0.012,
            half_height=HANDLE_LENGTH / 2,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
            color=(0.15, 0.15, 0.17),
            label="handle_shape",
        )
        kin_bodies.append(handle_body)

        # Rope: capsule chain hanging straight down from the handle tip.
        seg_len = self.rope_length / self.rope_segments
        tip0 = self.tip_pos[0]
        a0 = self.tip_axis[0]
        rope_points = [wp.vec3(*(tip0 + a0 * (i * seg_len))) for i in range(self.rope_segments + 1)]

        # Match the paper's rope linear density via capsule volume.
        capsule_volume = math.pi * self.rope_radius**2 * seg_len + 4.0 / 3.0 * math.pi * self.rope_radius**3
        rope_density = self.rope_linear_density * seg_len / capsule_volume
        rope_cfg = newton.ModelBuilder.ShapeConfig(
            density=rope_density,
            ke=1.0e5,
            kd=0.0,
            mu=self.friction,
        )

        rope_bodies, _rope_joints = builder.add_rod(
            positions=rope_points,
            radius=self.rope_radius,
            cfg=rope_cfg,
            stretch_stiffness=self.stretch_stiffness,
            stretch_damping=self.stretch_damping,
            bend_stiffness=self.bend_stiffness,
            bend_damping=self.bend_damping,
            label="rope",
            color=(0.85, 0.45, 0.1),
            body_frame_origin="com",
        )
        self.rope_bodies = rope_bodies
        self.seg_len = seg_len

        # Weighted tip (paper: "each rope has a mass affixed to aid the
        # formation of the knot").
        tip_radius = 0.014
        tip_volume = 4.0 / 3.0 * math.pi * tip_radius**3
        builder.add_shape_sphere(
            body=rope_bodies[-1],
            xform=wp.transform(wp.vec3(0.0, 0.0, seg_len / 2), wp.quat_identity()),
            radius=tip_radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=self.tip_mass / tip_volume, ke=1.0e5, kd=0.0, mu=self.friction),
            color=(0.2, 0.2, 0.55),
            label="tip_weight",
        )

        # Rope root capsule is kinematic: driven along the recorded command
        # (paper's particle model drives the first rope node directly).
        root = rope_bodies[0]
        kin_bodies.append(root)
        for b in kin_bodies:
            builder.body_mass[b] = 0.0
            builder.body_inv_mass[b] = 0.0
            builder.body_inertia[b] = wp.mat33(0.0)
            builder.body_inv_inertia[b] = wp.mat33(0.0)

        builder.add_ground_plane(color=(0.42, 0.44, 0.47))
        builder.color()
        self.model = builder.finalize()

        # --- Kinematic schedule -------------------------------------------
        self.kin_bodies = wp.array(kin_bodies, dtype=wp.int32)
        schedule = self._build_body_schedule(kin_bodies, handle_body, root)
        self.schedule = wp.array2d(schedule, dtype=wp.transform)
        self.step_idx = wp.zeros(1, dtype=wp.int32)

        # --- Solver ---------------------------------------------------------
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_body_contact_buffer_size=1024,
            rigid_contact_history=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.model.contacts(collision_pipeline=pipeline)

        # Apply initial kinematic poses to both states.
        wp.launch(
            drive_kinematic_bodies,
            dim=len(kin_bodies),
            inputs=[self.step_idx, self.n_sub_total, self.kin_bodies, self.schedule],
            outputs=[self.state_0.body_q, self.state_1.body_q],
        )

        self.viewer.set_model(self.model)
        self._set_camera()

        # Rope centerline recording for knot metrics; arm-base recording for
        # the fixed-pedestal check.
        self.frame_index = 0
        self.rope_traj: list[np.ndarray] = []
        self.base_traj: list[np.ndarray] = []

        self.capture()

    # ------------------------------------------------------------------
    # Trajectory construction
    # ------------------------------------------------------------------

    def _phase_times(self):
        t0 = self.t_settle
        t1 = t0 + self.t_throw
        t2 = t1 + self.t_flight
        t3 = t2 + self.t_lift
        return t0, t1, t2, t3

    def _build_handle_schedule(self, n: int):
        """Handle tip position and rope-root axis for every substep."""
        t = (np.arange(n) + 1) * self.sim_dt
        t0, t1, t2, t3 = self._phase_times()

        start = bezier_eval(BEZIER_CTRL, np.array([0.0]))[0]
        end = bezier_eval(BEZIER_CTRL, np.array([1.0]))[0]
        centroid = 0.5 * (BEZIER_CTRL.max(axis=0) + BEZIER_CTRL.min(axis=0))

        pos = np.zeros((n, 3))
        # Settle: hold start.
        pos[t <= t0] = start
        # Throw: Bezier (optionally amplified about its centroid).
        mask = (t > t0) & (t <= t1)
        s = (t[mask] - t0) / self.t_throw
        p = bezier_eval(BEZIER_CTRL, s)
        pos[mask] = centroid + self.throw_scale * (p - centroid)
        end_scaled = centroid + self.throw_scale * (end - centroid)
        # Flight: hold end.
        pos[(t > t1) & (t <= t2)] = end_scaled
        # Lift: raise smoothly to exhibit/tighten the knot.
        mask = (t > t2) & (t <= t3)
        u = np.clip((t[mask] - t2) / self.t_lift, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        lift_vec = np.array([self.lift_drift, 0.0, self.lift_height])
        pos[mask] = end_scaled + np.outer(smooth, lift_vec)
        # Hold: final pose.
        pos[t > t3] = end_scaled + lift_vec

        pos[:, 2] += self.z_offset

        # Rope-root axis: hangs down at rest, trails opposite the hand
        # velocity during fast motion.
        vel = np.gradient(pos, self.sim_dt, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        down = np.array([0.0, 0.0, -1.0])
        w = np.clip(speed / 2.5, 0.0, 0.85)
        axis = (1.0 - w[:, None]) * down - w[:, None] * (vel / np.maximum(speed, 1e-9)[:, None])
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        # Light smoothing to avoid axis jitter.
        k = max(1, int(0.02 / self.sim_dt))
        kernel = np.ones(k) / k
        for d in range(3):
            axis[:, d] = np.convolve(axis[:, d], kernel, mode="same")
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        # The rope is at rest (straight down) before the throw begins.
        axis[t <= t0] = down
        return pos, axis

    def _handle_com(self, i: int) -> np.ndarray:
        return self.tip_pos[i] - self.tip_axis[i] * (HANDLE_LENGTH / 2)

    def _build_root_quats(self):
        """Twist-free rope-root orientations via incremental minimal rotations."""
        n = len(self.tip_axis)
        quats = np.zeros((n, 4))
        q = quat_from_z_to(self.tip_axis[0])
        quats[0] = q
        for i in range(1, n):
            dq = quat_between(self.tip_axis[i - 1], self.tip_axis[i])
            q = quat_mul(dq, q)
            q /= np.linalg.norm(q)
            quats[i] = q
        return quats

    def _build_body_schedule(self, kin_bodies, handle_body, root_body):
        """Per-substep transforms for all kinematically driven bodies."""
        n = self.n_sub_total
        schedule = np.zeros((n, len(kin_bodies), 7), dtype=np.float32)
        kin_index = {b: i for i, b in enumerate(kin_bodies)}

        # Rope root: com at tip + axis * seg_len/2, +Z aligned to axis.
        root_com = self.tip_pos + self.tip_axis * (self.seg_len / 2)
        col = kin_index[root_body]
        schedule[:, col, :3] = root_com
        schedule[:, col, 3:] = self.root_quats

        # Handle: same orientation, com behind the tip.
        handle_com = self.tip_pos - self.tip_axis * (HANDLE_LENGTH / 2)
        col = kin_index[handle_body]
        schedule[:, col, :3] = handle_com
        schedule[:, col, 3:] = self.root_quats

        # Arm links: FK of the IK joint trajectory, interpolated per substep.
        if self.use_arm:
            frame_q = self.arm_body_q_frames  # [num_frames+1, arm_bodies, 7]
            sub = np.arange(n) / self.sim_substeps
            f0 = np.clip(sub.astype(int), 0, len(frame_q) - 2)
            u = (sub - f0)[:, None]
            for b in range(self.arm_body_count):
                col = kin_index[b]
                q0 = frame_q[f0, b]
                q1 = frame_q[f0 + 1, b]
                schedule[:, col, :3] = (1 - u) * q0[:, :3] + u * q1[:, :3]
                dot = np.sum(q0[:, 3:] * q1[:, 3:], axis=1, keepdims=True)
                q1r = np.where(dot < 0, -q1[:, 3:], q1[:, 3:])
                quat = (1 - u) * q0[:, 3:] + u * q1r
                quat /= np.linalg.norm(quat, axis=1, keepdims=True)
                schedule[:, col, 3:] = quat

        return schedule

    # ------------------------------------------------------------------
    # xArm7
    # ------------------------------------------------------------------

    def _add_arm(self, builder: newton.ModelBuilder) -> int:
        xarm_dir = default_xarm_dir()
        if xarm_dir is None:
            print("xArm7 assets not found (set FLYING_KNOTS_XARM_DIR); running without the arm.")
            self.use_arm = False
            return 0

        urdf_xml = preprocess_xarm_urdf(xarm_dir / "xarm7.urdf")

        # Fixed mount: the paper's command includes 3 base translations, but a
        # translating base reads as an unrealistic sliding pedestal. Instead the
        # arm is bolted to a static pedestal and the 7 arm joints alone track
        # the command. The mount is chosen by a workspace search
        # (scripts/flying_knot/fixed_base_search.py) so the entire trajectory
        # stays reachable; peak flange IK error is ~14 mm at the whip peak.
        self.base_pos = np.array([0.30, -0.20, self.z_offset + 0.20])

        def build_arm(b: newton.ModelBuilder):
            b.add_urdf(
                urdf_xml,
                xform=wp.transform(wp.vec3(*self.base_pos), wp.quat_identity()),
                floating=False,
                enable_self_collisions=False,
                collapse_fixed_joints=False,
            )

        n_bodies_before = len(builder.body_q)
        n_shapes_before = len(builder.shape_collision_group)
        build_arm(builder)
        arm_body_count = len(builder.body_q) - n_bodies_before
        # The arm is animated, not simulated: disable all its collisions.
        for s in range(n_shapes_before, len(builder.shape_collision_group)):
            builder.shape_collision_group[s] = 0

        # Pedestal column: static world geometry from the floor to the mount.
        # Collision stays on so the rope rests against the column instead of
        # tunneling through it during the lift phase.
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(self.base_pos[0], self.base_pos[1], self.base_pos[2] / 2), wp.quat_identity()),
            radius=0.075,
            half_height=self.base_pos[2] / 2,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, ke=1.0e5, kd=0.0, mu=self.friction, has_particle_collision=False
            ),
            color=(0.35, 0.36, 0.4),
            label="pedestal",
        )
        self.arm_base_body = n_bodies_before  # link_base, must remain fixed

        # Solve IK for the arm to track the handle over all frames.
        self.arm_body_q_frames = self._solve_arm_ik(build_arm, arm_body_count)
        return arm_body_count

    def _solve_arm_ik(self, build_arm, arm_body_count: int) -> np.ndarray:
        """Track flange + handle-tip targets over all frames; returns FK body poses."""
        ik_builder = newton.ModelBuilder()
        build_arm(ik_builder)
        body_keys = list(ik_builder.body_label)
        ik_model = ik_builder.finalize()

        ee_index = next(i for i, k in enumerate(body_keys) if k.endswith("link_eef"))

        # Frame-rate samples of the handle schedule (substep 0 of each frame).
        idx = np.arange(self.num_frames + 1) * self.sim_substeps
        idx = np.clip(idx, 0, self.n_sub_total - 1)
        tips = self.tip_pos[idx]
        axes = self.tip_axis[idx]
        flanges = tips - axes * HANDLE_LENGTH

        flange_obj = ik.IKObjectivePosition(
            link_index=ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*flanges[0])], dtype=wp.vec3),
            weight=1.0,
        )
        tip_obj = ik.IKObjectivePosition(
            link_index=ee_index,
            link_offset=wp.vec3(0.0, 0.0, HANDLE_LENGTH),
            target_positions=wp.array([wp.vec3(*tips[0])], dtype=wp.vec3),
            weight=1.0,
        )
        limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=ik_model.joint_limit_lower,
            joint_limit_upper=ik_model.joint_limit_upper,
            weight=10.0,
        )
        solver = ik.IKSolver(
            model=ik_model,
            n_problems=1,
            objectives=[flange_obj, tip_obj, limit_obj],
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
        )

        n_coords = ik_model.joint_coord_count
        joint_q = wp.zeros((1, n_coords), dtype=wp.float32)
        # Ready-pose initial guess: elbow bent toward +x.
        init = np.zeros(n_coords, dtype=np.float32)
        off = n_coords - 7
        init[off + 1] = 0.6  # joint2
        init[off + 3] = 0.8  # joint4
        joint_q.assign(init.reshape(1, -1))

        ik_state = ik_model.state()
        arm_q = np.zeros((len(tips), arm_body_count, 7), dtype=np.float32)
        errs = np.zeros(len(tips))
        q_solved = np.zeros((len(tips), n_coords), dtype=np.float32)
        for i in range(len(tips)):
            flange_obj.set_target_position(0, wp.vec3(*flanges[i]))
            tip_obj.set_target_position(0, wp.vec3(*tips[i]))
            solver.step(joint_q, joint_q, iterations=24 if i else 96)
            q_np = joint_q.numpy()[0]
            q_solved[i] = q_np
            newton.eval_fk(ik_model, wp.array(q_np, dtype=wp.float32), ik_model.joint_qd, ik_state)
            body_q = ik_state.body_q.numpy()
            arm_q[i] = body_q[:arm_body_count]
            # Flange tracking error.
            ee_t = body_q[ee_index]
            errs[i] = np.linalg.norm(ee_t[:3] - flanges[i])

        self.ik_errors = errs
        self.arm_joint_q = q_solved
        print(f"IK tracking error [m]: mean {errs.mean():.4f}, max {errs.max():.4f}")
        return arm_q

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def _set_camera(self):
        target = np.array([0.7, -0.1, self.z_offset + 0.1])
        pos = target + np.array([2.6, -2.2, 0.7])
        d = target - pos
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = math.degrees(math.atan2(d[2], np.linalg.norm(d[:2])))
        try:
            self.viewer.set_camera(wp.vec3(*pos), pitch, yaw)
        except (AttributeError, TypeError, NotImplementedError):
            pass

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                drive_kinematic_bodies,
                dim=len(self.kin_bodies),
                inputs=[self.step_idx, self.n_sub_total, self.kin_bodies, self.schedule],
                outputs=[self.state_0.body_q, self.state_1.body_q],
            )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_step, dim=1, inputs=[self.step_idx])

    def rope_centerline(self, body_q: np.ndarray) -> np.ndarray:
        """Rope node positions from body transforms (com origin capsules)."""
        n = len(self.rope_bodies)
        nodes = np.zeros((n + 1, 3))
        for i, b in enumerate(self.rope_bodies):
            t = body_q[b]
            pos, quat = t[:3], t[3:]
            # Rotate local +Z*half_len by quat.
            x, y, z, w = quat
            hz = self.seg_len / 2
            axis = np.array(
                [2.0 * (x * z + w * y) * hz, 2.0 * (y * z - w * x) * hz, (1.0 - 2.0 * (x * x + y * y)) * hz]
            )
            nodes[i] = pos - axis
            if i == n - 1:
                nodes[n] = pos + axis
        return nodes

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        body_q = self.state_0.body_q.numpy()
        self.rope_traj.append(self.rope_centerline(body_q))
        if self.use_arm:
            self.base_traj.append(body_q[self.arm_base_body].copy())

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def knot_metrics(self, nodes: np.ndarray) -> dict:
        writhe = polyline_writhe(nodes)
        crossings = max(count_crossings(nodes, axis=0), count_crossings(nodes, axis=1))
        e2e = np.linalg.norm(nodes[-1] - nodes[0])
        arc = np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()
        return {
            "writhe": float(writhe),
            "crossings": int(crossings),
            "end_to_end": float(e2e),
            "arc_length": float(arc),
            "length_ratio": float(e2e / arc) if arc > 0 else 1.0,
        }

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.isfinite(body_q).all(), "Non-finite body transforms"
        assert np.isfinite(body_qd).all(), "Non-finite body velocities"

        # The robot is bolted to a static pedestal: its base link must not move.
        if self.use_arm and self.base_traj:
            base = np.array(self.base_traj)
            base_drift = np.abs(base - base[0]).max()
            print(f"arm base transform drift over {len(base)} frames: {base_drift:.2e}")
            assert base_drift < 1.0e-6, f"arm base moved: max transform drift {base_drift:.2e}"

        nodes = self.rope_traj[-1] if self.rope_traj else self.rope_centerline(body_q)
        metrics = self.knot_metrics(nodes)
        print(
            f"final rope metrics: writhe {metrics['writhe']:+.2f}, "
            f"crossings {metrics['crossings']}, "
            f"end-to-end/arc {metrics['length_ratio']:.3f}"
        )

        if self.save_traj:
            np.savez_compressed(
                self.save_traj,
                rope_traj=np.array(self.rope_traj, dtype=np.float32),
                tip_pos=self.tip_pos[:: self.sim_substeps].astype(np.float32),
                metrics_writhe=metrics["writhe"],
                metrics_crossings=metrics["crossings"],
                metrics_length_ratio=metrics["length_ratio"],
                ik_errors=getattr(self, "ik_errors", np.zeros(1)),
                arm_joint_q=getattr(self, "arm_joint_q", np.zeros((0, 7))).astype(np.float32),
                base_traj=np.array(self.base_traj, dtype=np.float32) if self.base_traj else np.zeros((0, 7)),
            )
            print(f"saved rope trajectory to {self.save_traj}")

        if self.expect_knot:
            assert abs(metrics["writhe"]) > 2.0, f"no knot: writhe {metrics['writhe']:.2f}"
            assert metrics["length_ratio"] < 0.95, f"no knot: rope taut ratio {metrics['length_ratio']:.3f}"


def add_arguments(parser):
    parser.add_argument("--time-scale", type=float, default=0.8, dest="time_scale")
    parser.add_argument("--throw-scale", type=float, default=1.0, dest="throw_scale")
    parser.add_argument("--z-offset", type=float, default=1.1, dest="z_offset")
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rope-length", type=float, default=1.1, dest="rope_length")
    parser.add_argument("--rope-segments", type=int, default=36, dest="rope_segments")
    parser.add_argument("--rope-radius", type=float, default=0.005, dest="rope_radius")
    parser.add_argument("--rope-density", type=float, default=0.05, dest="rope_density")
    parser.add_argument("--tip-mass", type=float, default=0.05, dest="tip_mass")
    parser.add_argument("--stretch-stiffness", type=float, default=2.0e5, dest="stretch_stiffness")
    parser.add_argument("--stretch-damping", type=float, default=0.5, dest="stretch_damping")
    parser.add_argument("--bend-stiffness", type=float, default=2.0e-3, dest="bend_stiffness")
    parser.add_argument("--bend-damping", type=float, default=1.0e-4, dest="bend_damping")
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--no-arm", action="store_true", dest="no_arm")
    parser.add_argument("--save-traj", type=str, default=None, dest="save_traj")
    parser.add_argument("--expect-knot", action="store_true", dest="expect_knot")
    parser.add_argument("--t-settle", type=float, default=1.5, dest="t_settle")
    parser.add_argument("--t-flight", type=float, default=2.0, dest="t_flight")
    parser.add_argument("--t-lift", type=float, default=2.5, dest="t_lift")
    parser.add_argument("--t-hold", type=float, default=1.5, dest="t_hold")
    parser.add_argument("--lift-height", type=float, default=0.55, dest="lift_height")
    parser.add_argument("--lift-drift", type=float, default=0.12, dest="lift_drift")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    add_arguments(parser)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
