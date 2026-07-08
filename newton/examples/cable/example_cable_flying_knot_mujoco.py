# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Flying Knot (MuJoCo-coupled)
#
# Dynamically driven variant of example_cable_flying_knot: instead of
# animating the xArm7 kinematically, the arm is simulated by SolverMuJoCo
# with PD position targets and coupled to the SolverVBD cable through the
# experimental SolverCoupledProxy framework. The end-effector body is
# exposed to the VBD entry as a proxy body, and the rope's root capsule is
# attached to the handle tip by a cable joint owned by the VBD entry, so
# rope reaction forces feed back onto the arm.
#
# The joint-space reference comes from the same digitized throw command
# (IK-tracked at 60 Hz), zero-phase low-pass filtered and interpolated with
# a natural cubic spline, so the commanded position is C2-continuous
# (smooth velocity and continuous acceleration). The MuJoCo dynamics then
# low-pass the residual, removing the frame-rate velocity steps that make
# the kinematic variant look choppy.
#
# Requires the xArm7 assets from flying_knots_public (see
# FLYING_KNOTS_XARM_DIR in example_cable_flying_knot).
#
# Command: python -m newton.examples cable_flying_knot_mujoco
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
from newton.examples.cable.example_cable_flying_knot import (
    BEZIER_CTRL,
    HANDLE_LENGTH,
    T_THROW,
    bezier_eval,
    count_crossings,
    default_xarm_dir,
    polyline_writhe,
    preprocess_xarm_urdf,
    quat_between,
    quat_from_z_to,
    quat_mul,
)
from newton.solvers import SolverMuJoCo, SolverVBD

# xArm7 effort limits from the paper (Table V) [N*m].
XARM_EFFORT_LIMIT = [130.0, 130.0, 40.0, 40.0, 40.0, 20.0, 20.0]
# PD position gains for the MuJoCo drive, proximal to distal.
XARM_TARGET_KE = [3000.0, 3000.0, 2000.0, 2000.0, 1200.0, 800.0, 800.0]
XARM_TARGET_KD = [120.0, 120.0, 80.0, 80.0, 40.0, 25.0, 25.0]


def natural_cubic_spline_coeffs(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Natural cubic spline coefficients (C2-continuous) for samples y(t).

    Returns (a, b, c, d) so that on segment i:
    ``y(s) = a[i] + b[i] h + c[i] h^2 + d[i] h^3`` with ``h = s - t[i]``.
    Multi-column y is supported (one spline per column).
    """
    n = len(t) - 1
    h = np.diff(t)
    y = np.atleast_2d(y.T).T  # ensure 2d [n+1, m]
    m = y.shape[1]
    # Solve the tridiagonal system for second derivatives (natural BC).
    rhs = np.zeros((n + 1, m))
    rhs[1:n] = 3.0 * ((y[2:] - y[1:-1]) / h[1:, None] - (y[1:-1] - y[:-2]) / h[:-1, None])
    lower = np.zeros(n + 1)
    diag = np.ones(n + 1)
    upper = np.zeros(n + 1)
    lower[1:n] = h[:-1]
    diag[1:n] = 2.0 * (h[:-1] + h[1:])
    upper[1:n] = h[1:]
    # Thomas algorithm.
    c = np.zeros((n + 1, m))
    cp = np.zeros(n + 1)
    dp = np.zeros((n + 1, m))
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n + 1):
        denom = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / denom
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / denom
    c[n] = dp[n]
    for i in range(n - 1, -1, -1):
        c[i] = dp[i] - cp[i] * c[i + 1]
    a = y[:-1]
    b = (y[1:] - y[:-1]) / h[:, None] - h[:, None] * (2.0 * c[:-1] + c[1:]) / 3.0
    d = (c[1:] - c[:-1]) / (3.0 * h[:, None])
    return a, b, c[:-1], d


def eval_spline(t_knots: np.ndarray, coeffs, t_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate spline position and velocity at t_eval."""
    a, b, c, d = coeffs
    idx = np.clip(np.searchsorted(t_knots, t_eval, side="right") - 1, 0, len(t_knots) - 2)
    h = (t_eval - t_knots[idx])[:, None]
    pos = a[idx] + b[idx] * h + c[idx] * h**2 + d[idx] * h**3
    vel = b[idx] + 2.0 * c[idx] * h + 3.0 * d[idx] * h**2
    return pos, vel


def zero_phase_smooth(y: np.ndarray, width: int) -> np.ndarray:
    """Edge-padded zero-phase moving average along axis 0."""
    if width <= 1:
        return y
    kernel = np.ones(width) / width
    pad = width // 2
    out = np.empty_like(y)
    for col in range(y.shape[1]):
        padded = np.concatenate([np.full(pad, y[0, col]), y[:, col], np.full(pad, y[-1, col])])
        smoothed = np.convolve(padded, kernel, mode="same")
        out[:, col] = smoothed[pad : pad + len(y)]
    return out


@wp.kernel
def _posture_residuals(
    body_q: wp.array2d[wp.transform],
    joint_q: wp.array2d[wp.float32],
    anchor: wp.array[wp.float32],
    n_coords: int,
    start_idx: int,
    weight: float,
    problem_idx: wp.array[wp.int32],
    residuals: wp.array2d[wp.float32],
):
    row_idx = wp.tid()
    for c in range(n_coords):
        residuals[row_idx, start_idx + c] = weight * (joint_q[row_idx, c] - anchor[c])


@wp.kernel
def _posture_jacobian(
    n_coords: int,
    start_idx: int,
    weight: float,
    jacobian: wp.array3d[wp.float32],
):
    row_idx, c = wp.tid()
    jacobian[row_idx, start_idx + c, c] = weight


class IKObjectivePosture(ik.IKObjective):
    """Pull the arm coordinates toward an anchor configuration.

    Regularizes the 2-dimensional nullspace of the two-point position task so
    sequential warm-started solves stay on one solution branch instead of
    drifting or flipping elbow/wrist configurations between samples.
    """

    def __init__(self, n_coords: int, anchor: wp.array, weight: float):
        super().__init__()
        self.n_coords = n_coords
        self.anchor = anchor
        self.weight = weight

    def residual_dim(self):
        return self.n_coords

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _posture_residuals,
            dim=body_q.shape[0],
            inputs=[body_q, joint_q, self.anchor, self.n_coords, start_idx, self.weight, problem_idx],
            outputs=[residuals],
            device=joint_q.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        # The posture Jacobian is constant and diagonal: d r_c / d q_c = weight.
        wp.launch(
            _posture_jacobian,
            dim=(jacobian.shape[0], self.n_coords),
            inputs=[self.n_coords, start_idx, self.weight],
            outputs=[jacobian],
            device=jacobian.device,
        )


@wp.kernel
def apply_joint_targets(
    step_idx: wp.array[wp.int32],
    num_steps: int,
    q_ref: wp.array2d[float],
    qd_ref: wp.array2d[float],
    n_coords: int,
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
):
    tid = wp.tid()
    idx = wp.min(step_idx[0], num_steps - 1)
    if tid < n_coords:
        joint_target_q[tid] = q_ref[idx, tid]
        joint_target_qd[tid] = qd_ref[idx, tid]


@wp.kernel
def drive_root_from_ee(
    step_idx: wp.array[wp.int32],
    num_steps: int,
    ee_body: int,
    root_body: int,
    root_local: int,
    tip_offset: wp.vec3,
    seg_half: float,
    axis_sched: wp.array[wp.vec3],
    quat_sched: wp.array[wp.quat],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
    entry_q0: wp.array[wp.transform],
    entry_q1: wp.array[wp.transform],
):
    """Kinematic root drive: place the rope root at the live handle tip.

    The tip position comes from the dynamically simulated EE body (previous
    substep, a lag of one sim_dt), while the root orientation follows the
    precomputed twist-free trailing-axis schedule as in the kinematic
    example.
    """
    idx = wp.min(step_idx[0], num_steps - 1)
    tip = wp.transform_point(body_q0[ee_body], tip_offset)
    a = axis_sched[idx]
    T = wp.transform(tip + a * seg_half, quat_sched[idx])
    body_q0[root_body] = T
    body_q1[root_body] = T
    # The coupled solver reconciles entry-local output states over the parent
    # state, and VBD leaves kinematic bodies untouched in its output, so the
    # prescribed pose must also be written into the entry buffers.
    entry_q0[root_local] = T
    entry_q1[root_local] = T


@wp.kernel
def drive_root_from_schedule(
    step_idx: wp.array[wp.int32],
    num_steps: int,
    root_body: int,
    root_local: int,
    pos_sched: wp.array[wp.vec3],
    quat_sched: wp.array[wp.quat],
    vel_sched: wp.array[wp.spatial_vector],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
    body_qd0: wp.array[wp.spatial_vector],
    body_qd1: wp.array[wp.spatial_vector],
    entry_q0: wp.array[wp.transform],
    entry_q1: wp.array[wp.transform],
    entry_qd0: wp.array[wp.spatial_vector],
    entry_qd1: wp.array[wp.spatial_vector],
):
    idx = wp.min(step_idx[0], num_steps - 1)
    T = wp.transform(pos_sched[idx], quat_sched[idx])
    v = vel_sched[idx]
    body_q0[root_body] = T
    body_q1[root_body] = T
    body_qd0[root_body] = v
    body_qd1[root_body] = v
    # See drive_root_from_ee: entry output buffers must be written directly.
    entry_q0[root_local] = T
    entry_q1[root_local] = T
    entry_qd0[root_local] = v
    entry_qd1[root_local] = v


@wp.kernel
def apply_rope_air_drag(
    body_indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    k_perp: float,
    k_par: float,
    body_f: wp.array[wp.spatial_vector],
):
    """Quadratic aerodynamic drag on rope capsules (per-segment cylinder drag).

    Real cotton cord at whip speeds (5-7 m/s) sees drag comparable to half its
    weight; it is what keeps a thrown loop open in air, and the VBD rope has
    no aerodynamic model of its own.
    """
    tid = wp.tid()
    b = body_indices[tid]
    v = wp.spatial_top(body_qd[b])  # linear velocity at COM, world frame
    axis = wp.quat_rotate(wp.transform_get_rotation(body_q[b]), wp.vec3(0.0, 0.0, 1.0))
    v_par = wp.dot(v, axis) * axis
    v_perp = v - v_par
    f = -k_perp * wp.length(v_perp) * v_perp - k_par * wp.length(v_par) * v_par
    t = wp.spatial_bottom(body_f[b])
    fw = wp.spatial_top(body_f[b]) + f
    body_f[b] = wp.spatial_vector(fw, t)


@wp.kernel
def advance_step(step_idx: wp.array[wp.int32]):
    step_idx[0] = step_idx[0] + 1


def solve_branch_continuous_ik(
    urdf_xml: str,
    base_pos: np.ndarray,
    tips: np.ndarray,
    flanges: np.ndarray,
    *,
    posture_weight: float = 0.1,
    flange_weight: float = 0.0,
    axis_weight: float = 0.0,
    axis_quats: np.ndarray | None = None,
    max_step: float = 0.02,
    iters: int = 12,
    n_arm_coords: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential warm-started, posture-regularized IK over a tip trajectory.

    Fast task-space motion is subdivided so consecutive targets move at most
    ``max_step``, and a posture objective anchors each solve at its warm
    start. Together these keep the solution on one branch instead of flipping
    elbow/wrist configurations between samples (the source of the kinematic
    example's choppy joint motion).

    Returns (q_frames [n, n_arm_coords], flange position errors [n]).
    """
    ik_builder = newton.ModelBuilder()
    ik_builder.add_urdf(
        urdf_xml,
        xform=wp.transform(wp.vec3(*base_pos), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
    )
    body_labels = list(ik_builder.body_label)
    ik_model = ik_builder.finalize()
    ee_index = next(i for i, k in enumerate(body_labels) if k.endswith("link_eef"))

    # The rope pivots freely at the handle tip, so only the tip position must
    # track the command; constraining the flange as well forces the wrist to
    # whip the handle axis around at extreme joint rates.
    flange_obj = ik.IKObjectivePosition(
        link_index=ee_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(*flanges[0])], dtype=wp.vec3),
        weight=flange_weight,
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
    n_coords = ik_model.joint_coord_count
    posture_anchor = wp.zeros(n_coords, dtype=wp.float32)
    posture_obj = IKObjectivePosture(n_coords=n_coords, anchor=posture_anchor, weight=posture_weight)
    objectives = [flange_obj, tip_obj, limit_obj, posture_obj]
    rot_obj = None
    if axis_weight > 0.0 and axis_quats is not None:
        # Steer the handle axis like the demonstrator's wrist: the twist-free
        # orientation series aligns EE +z with the trailing-velocity axis.
        rot_obj = ik.IKObjectiveRotation(
            link_index=ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*axis_quats[0])], dtype=wp.vec4),
            weight=axis_weight,
        )
        objectives.append(rot_obj)
    solver = ik.IKSolver(
        model=ik_model,
        n_problems=1,
        objectives=objectives,
        optimizer=ik.IKOptimizer.LM,
        jacobian_mode=ik.IKJacobianType.AUTODIFF,
    )
    joint_q = wp.zeros((1, n_coords), dtype=wp.float32)
    init = np.zeros(n_coords, dtype=np.float32)
    init[1] = 0.6
    init[3] = 0.8
    joint_q.assign(init.reshape(1, -1))

    ik_state = ik_model.state()

    def solve_target(flange, tip, quat, n_iters):
        # Anchor the posture at the current (warm-start) configuration.
        wp.copy(posture_anchor, joint_q.reshape((-1,)))
        flange_obj.set_target_position(0, wp.vec3(*flange))
        tip_obj.set_target_position(0, wp.vec3(*tip))
        if rot_obj is not None and quat is not None:
            rot_obj.set_target_rotation(0, wp.vec4(*quat))
        solver.step(joint_q, joint_q, iterations=n_iters)

    def nlerp_quat(qa, qb, alpha):
        if qa @ qb < 0.0:
            qb = -qb
        q = (1.0 - alpha) * qa + alpha * qb
        return q / np.linalg.norm(q)

    posture_obj.weight = 0.0
    solve_target(flanges[0], tips[0], axis_quats[0] if axis_quats is not None else None, 96)
    posture_obj.weight = posture_weight
    q_frames = np.zeros((len(tips), n_arm_coords), dtype=np.float64)
    q_frames[0] = joint_q.numpy()[0][:n_arm_coords]
    errs = np.zeros(len(tips))
    for i in range(1, len(tips)):
        dist = float(np.linalg.norm(tips[i] - tips[i - 1]))
        n_sub = max(1, int(np.ceil(dist / max_step)))
        for k in range(1, n_sub + 1):
            alpha = k / n_sub
            quat = None
            if axis_quats is not None:
                quat = nlerp_quat(axis_quats[i - 1], axis_quats[i], alpha)
            solve_target(
                flanges[i - 1] + alpha * (flanges[i] - flanges[i - 1]),
                tips[i - 1] + alpha * (tips[i] - tips[i - 1]),
                quat,
                iters,
            )
        q_np = joint_q.numpy()[0]
        q_frames[i] = q_np[:n_arm_coords]
        newton.eval_fk(ik_model, wp.array(q_np, dtype=wp.float32), ik_model.joint_qd, ik_state)
        body_q = ik_state.body_q.numpy()[ee_index]
        x, y, z, w = body_q[3:]
        tip_axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        errs[i] = np.linalg.norm(body_q[:3] + tip_axis * HANDLE_LENGTH - tips[i])
    return q_frames, errs


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = getattr(args, "substeps", 32)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.vbd_iterations = getattr(args, "vbd_iterations", 10)

        # Throw command scaling (defaults re-tuned for the dynamic arm).
        self.time_scale = getattr(args, "time_scale", 0.8)
        self.throw_scale = getattr(args, "throw_scale", 1.0)
        self.z_offset = getattr(args, "z_offset", 1.1)

        # Rope parameters (as in the kinematic example).
        self.rope_length = getattr(args, "rope_length", 1.1)
        self.rope_segments = getattr(args, "rope_segments", 36)
        self.rope_radius = getattr(args, "rope_radius", 0.005)
        self.rope_linear_density = getattr(args, "rope_density", 0.05)
        self.tip_mass = getattr(args, "tip_mass", 0.05)
        # Cable joint stiffness/damping are per-joint quantities: for the same
        # physical rope (fixed EA and EI), the per-joint values scale inversely
        # with segment length, i.e. linearly with segment count. The CLI values
        # are defined at the 36-segment reference resolution and rescaled here,
        # so --rope-segments changes discretization fidelity, not the rope.
        res_scale = self.rope_segments / 36.0
        self.stretch_stiffness = getattr(args, "stretch_stiffness", 2.0e5) * res_scale
        self.bend_stiffness = getattr(args, "bend_stiffness", 2.0e-3) * res_scale
        self.bend_damping = getattr(args, "bend_damping", 1.0e-4) * res_scale
        self.stretch_damping = getattr(args, "stretch_damping", 0.5) * res_scale
        self.friction = getattr(args, "friction", 1.0)
        self.save_traj = getattr(args, "save_traj", None)
        self.expect_knot = getattr(args, "expect_knot", False)
        self.ref_filter_width = getattr(args, "ref_filter_width", 5)
        self.posture_weight = getattr(args, "posture_weight", 0.1)
        self.flange_weight = getattr(args, "flange_weight", 0.0)
        self.attach_bend_scale = getattr(args, "attach_bend_scale", 1.0)
        self.follow_scale = getattr(args, "follow_scale", 1.0)
        self.axis_weight = getattr(args, "axis_weight", 0.0)
        self.bezier_delta_file = getattr(args, "bezier_delta_file", None)
        self.root_drive = getattr(args, "root_drive", "attached")
        self.ik_max_step = getattr(args, "ik_max_step", 0.02)
        self.ik_iters = getattr(args, "ik_iters", 12)
        self.command_kind = getattr(args, "command", "digitized")
        self.air_drag = getattr(args, "air_drag", 0.0)
        self.replay_file = getattr(args, "replay_file", None)
        self.replay_node = getattr(args, "replay_node", 3)
        self.yank_vec = np.array(
            [getattr(args, "yank_dx", 0.0), getattr(args, "yank_dy", 0.0), getattr(args, "yank_dz", 0.0)]
        )
        self.yank_delay = getattr(args, "yank_delay", 0.2)
        self.yank_t = getattr(args, "yank_t", 0.2)
        if self.command_kind == "replay":
            # The replayed node is `replay_node` segments from the old root:
            # shorten the rope so the tail matches the recorded system.
            self.rope_segments = self.rope_segments - self.replay_node
            self.rope_length = self.rope_length * self.rope_segments / (self.rope_segments + self.replay_node)
        # Lasso parameters: rx, rz, yaw, y_amp, span_deg, loop_frac, rdx, rdy, rdz.
        self.lasso_params = np.array([0.28, 0.30, 0.25, 0.08, 380.0, 0.75, -0.25, -0.05, -0.30])
        lasso_file = getattr(args, "lasso_file", None)
        if lasso_file:
            self.lasso_params = np.load(lasso_file)["params"]
        self.gain_scale = getattr(args, "gain_scale", 1.0)
        self.effort_scale = getattr(args, "effort_scale", 1.0)

        # Phase timing [s].
        self.t_settle = getattr(args, "t_settle", 1.5)
        self.t_throw = T_THROW * self.time_scale
        self.t_flight = getattr(args, "t_flight", 2.0)
        self.t_lift = getattr(args, "t_lift", 2.5)
        self.t_hold = getattr(args, "t_hold", 1.5)
        self.duration = self.t_settle + self.t_throw + self.t_flight + self.t_lift + self.t_hold
        self.num_frames = int(round(self.duration * self.fps))
        self.lift_height = getattr(args, "lift_height", 0.7)
        self.n_sub_total = self.num_frames * self.sim_substeps
        if getattr(args, "command", "digitized") == "searched":
            # Best command from scripts/flying_knot/command_search.py (cross-
            # entropy search over Bezier control-point deltas + timing + tip
            # mass, with a threading-persistence objective and a dynamic-
            # feasibility penalty): ties and holds the overhand knot under the
            # fully dynamic coupled simulation. The IK discretization is part
            # of the optimized reference and is pinned alongside the command.
            self.bezier_delta_file = str(Path(newton.examples.get_asset("flying_knot_searched_command.npz")))
            self.time_scale = 0.7861
            self.t_throw = T_THROW * self.time_scale
            self.tip_mass = 0.0399
            self.ik_max_step = 0.035
            self.ik_iters = 8
            self.duration = self.t_settle + self.t_throw + self.t_flight + self.t_lift + self.t_hold
            self.num_frames = int(round(self.duration * self.fps))
            self.n_sub_total = self.num_frames * self.sim_substeps
        elif getattr(args, "command", "digitized") == "searched-hires":
            # High-resolution variant: 72 cable segments (15 mm at 5 mm radius)
            # for a well-resolved knot bundle. The command was re-derived with
            # the basin-aware search (the 36-segment command sits on a
            # razor-thin manifold that does not survive the discretization
            # change), and the knot cinch at this resolution needs a finer
            # solve: 48 substeps and 16 VBD iterations. Solver settings, IK
            # discretization, and the cinch yank are pinned with the command.
            preset = np.load(Path(newton.examples.get_asset("flying_knot_searched_command_hires.npz")))
            self._preset_delta = preset["delta"]
            self.time_scale = float(preset["time_scale"])
            self.t_throw = T_THROW * self.time_scale
            self.tip_mass = float(preset["tip_mass"])
            yank = preset["yank"]
            self.yank_vec = yank[:3].copy()
            self.yank_delay = float(yank[3])
            self.yank_t = float(yank[4])
            self.rope_segments = 72
            self.sim_substeps = 48
            self.sim_dt = self.frame_dt / self.sim_substeps
            self.vbd_iterations = 16
            self.ik_max_step = 0.035
            self.ik_iters = 8
            res_scale = self.rope_segments / 36.0
            self.stretch_stiffness = getattr(args, "stretch_stiffness", 2.0e5) * res_scale
            self.bend_stiffness = getattr(args, "bend_stiffness", 2.0e-3) * res_scale
            self.bend_damping = getattr(args, "bend_damping", 1.0e-4) * res_scale
            self.stretch_damping = getattr(args, "stretch_damping", 0.5) * res_scale
            self.duration = self.t_settle + self.t_throw + self.t_flight + self.t_lift + self.t_hold
            self.num_frames = int(round(self.duration * self.fps))
            self.n_sub_total = self.num_frames * self.sim_substeps

        xarm_dir = default_xarm_dir()
        if xarm_dir is None:
            raise RuntimeError(
                "example_cable_flying_knot_mujoco requires the xArm7 assets from "
                "flying_knots_public; set FLYING_KNOTS_XARM_DIR."
            )
        self.urdf_xml = preprocess_xarm_urdf(xarm_dir / "xarm7.urdf")
        # Fixed mount from scripts/flying_knot/mount_velocity_search.py
        # (minimizes peak joint velocity of the branch-continuous IK).
        self.base_pos = np.array([0.35, -0.25, self.z_offset + 0.13])
        self.n_arm_coords = 7

        # Handle-tip task trajectory and joint-space reference.
        tip_pos = self._build_tip_schedule()
        q_frames = self._solve_arm_ik(tip_pos)
        self._build_joint_reference(q_frames)

        # --- Scene ----------------------------------------------------------
        builder = newton.ModelBuilder(gravity=-9.81)
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = self.friction

        # Arm (MuJoCo entry).
        arm_body_start, arm_joint_start, arm_shape_start = builder.body_count, builder.joint_count, builder.shape_count
        builder.add_urdf(
            self.urdf_xml,
            xform=wp.transform(wp.vec3(*self.base_pos), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        self.arm_bodies = list(range(arm_body_start, builder.body_count))
        self.arm_joints = list(range(arm_joint_start, builder.joint_count))
        builder.joint_q[: self.n_arm_coords] = self.q_ref_frames[0].tolist()
        builder.joint_target_q[: self.n_arm_coords] = self.q_ref_frames[0].tolist()
        builder.joint_target_ke[: self.n_arm_coords] = [self.gain_scale * k for k in XARM_TARGET_KE]
        builder.joint_target_kd[: self.n_arm_coords] = [self.gain_scale * k for k in XARM_TARGET_KD]
        builder.joint_effort_limit[: self.n_arm_coords] = [self.effort_scale * e for e in XARM_EFFORT_LIMIT]
        builder.joint_armature[: self.n_arm_coords] = [0.05] * self.n_arm_coords
        # POSITION_VELOCITY drive: the C2 velocity reference feeds forward, so
        # tracking does not lag by (kd/ke) * qd during the fast throw.
        builder.joint_target_mode[: self.n_arm_coords] = [
            int(newton.JointTargetMode.POSITION_VELOCITY)
        ] * self.n_arm_coords

        # Gravity compensation for clean PD tracking (as in the Franka examples).
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in self.arm_bodies:
            gravcomp.values[body] = 1.0

        body_labels = list(builder.body_label)
        self.ee_body = next(i for i, k in enumerate(body_labels) if k.endswith("link_eef"))

        # Handle: rigid extension of the EE (visual only).
        builder.add_shape_capsule(
            body=self.ee_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, HANDLE_LENGTH / 2), wp.quat_identity()),
            radius=0.012,
            half_height=HANDLE_LENGTH / 2,
            cfg=newton.ModelBuilder.ShapeConfig(density=100.0, has_shape_collision=False, has_particle_collision=False),
            color=(0.15, 0.15, 0.17),
            label="handle_shape",
        )
        # Pedestal column (static, visual only).
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(self.base_pos[0], self.base_pos[1], self.base_pos[2] / 2), wp.quat_identity()),
            radius=0.075,
            half_height=self.base_pos[2] / 2,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
            color=(0.35, 0.36, 0.4),
            label="pedestal",
        )
        self.arm_shapes = list(range(arm_shape_start, builder.shape_count))
        # The arm is coupled through the attachment joint, not contact.
        for s in self.arm_shapes:
            builder.shape_collision_group[s] = 0

        # Rope (VBD entry), hanging from the handle tip at t=0.
        rope_shape_start = builder.shape_count
        seg_len = self.rope_length / self.rope_segments
        tip0 = self.tip_pos_frames[0]
        rope_points = [
            wp.vec3(*(tip0 + np.array([0.0, 0.0, -1.0]) * (i * seg_len))) for i in range(self.rope_segments + 1)
        ]
        capsule_volume = math.pi * self.rope_radius**2 * seg_len + 4.0 / 3.0 * math.pi * self.rope_radius**3
        rope_density = self.rope_linear_density * seg_len / capsule_volume
        rope_cfg = newton.ModelBuilder.ShapeConfig(density=rope_density, ke=1.0e5, kd=0.0, mu=self.friction)
        rope_bodies, rope_joints = builder.add_rod(
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

        if self.root_drive == "attached":
            # Attachment: handle tip -> rope root, owned by the VBD entry with
            # the EE body arriving as a proxy. Bend stiffness ~rope so the rope
            # pivots at the handle tip like a tied-on cord.
            attach_joint = builder.add_joint_cable(
                parent=self.ee_body,
                child=rope_bodies[0],
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, HANDLE_LENGTH), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, -seg_len / 2), wp.quat_identity()),
                stretch_stiffness=self.stretch_stiffness,
                stretch_damping=self.stretch_damping,
                bend_stiffness=self.bend_stiffness * self.attach_bend_scale,
                bend_damping=self.bend_damping * self.attach_bend_scale,
                label="rope_attach",
            )
            self.rope_joints = [*rope_joints, attach_joint]
        else:
            # Kinematic root ("kinematic": prescribed from the live EE pose;
            # "command": prescribed from the analytic command schedule exactly
            # as in the kinematic example, while the MuJoCo arm tracks the
            # same command dynamically).
            root = rope_bodies[0]
            builder.body_mass[root] = 0.0
            builder.body_inv_mass[root] = 0.0
            builder.body_inertia[root] = wp.mat33(0.0)
            builder.body_inv_inertia[root] = wp.mat33(0.0)
            self.rope_joints = list(rope_joints)
        rope_shapes = list(range(rope_shape_start, builder.shape_count))
        ground_shape = builder.add_ground_plane(color=(0.42, 0.44, 0.47))
        self.rope_shapes = [*rope_shapes, ground_shape]

        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device

        # Make model body_q consistent with the initial joint configuration so
        # solver rest poses (attachment joint, VBD structural rest) are valid.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # --- Coupled solver ---------------------------------------------------
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        solver="newton",
                        integrator="implicitfast",
                        iterations=getattr(args, "mujoco_iterations", 20),
                        ls_iterations=10,
                        use_mujoco_contacts=False,
                        disable_contacts=True,
                    ),
                    bodies=self.arm_bodies,
                    joints=self.arm_joints,
                    shapes=self.arm_shapes,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v,
                        iterations=self.vbd_iterations,
                        rigid_body_contact_buffer_size=1024,
                        rigid_contact_history=True,
                    ),
                    bodies=self.rope_bodies,
                    joints=self.rope_joints,
                    shapes=self.rope_shapes,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=(
                    [
                        SolverCoupledProxy.Proxy(
                            source="mjc",
                            destination="vbd",
                            bodies=[self.ee_body],
                            mass_scale=getattr(args, "mass_scale", 1.0),
                            mode=getattr(args, "coupling_mode", "lagged"),
                        ),
                    ]
                    if True  # keep proxy in both modes; empty proxy list breaks coupled stepping
                    else []
                ),
                iterations=getattr(args, "proxy_iterations", 1),
            ),
        )

        vbd_entry = self.solver._entries["vbd"]
        self.vbd_entry_state_0 = vbd_entry.state_0
        self.vbd_entry_state_1 = vbd_entry.state_1
        self.root_local = int(vbd_entry.body_global_to_local.numpy()[self.rope_bodies[0]])

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        # Aerodynamic drag on the rope (0.5 * rho * Cd * d * L per segment).
        self.rope_body_wp = wp.array(self.rope_bodies, dtype=wp.int32)
        seg_area = 2.0 * self.rope_radius * self.seg_len
        self.drag_k_perp = self.air_drag * 0.5 * 1.2 * 1.1 * seg_area
        self.drag_k_par = 0.02 * self.drag_k_perp

        # Device-side reference buffers.
        self.q_ref_wp = wp.array2d(self.q_ref_sub.astype(np.float32), dtype=float)
        self.qd_ref_wp = wp.array2d(self.qd_ref_sub.astype(np.float32), dtype=float)
        self.step_idx = wp.zeros(1, dtype=wp.int32)
        if self.root_drive in ("kinematic", "command"):
            axes_sub, quats_sub, tips_sub = self._root_axis_schedule()
            self.root_axis_wp = wp.array(axes_sub.astype(np.float32), dtype=wp.vec3)
            self.root_quat_wp = wp.array(quats_sub.astype(np.float32), dtype=wp.quat)
            root_pos = tips_sub + axes_sub * (self.seg_len / 2)
            self.root_pos_wp = wp.array(root_pos.astype(np.float32), dtype=wp.vec3)
            # Spatial velocity of the prescribed root (angular from the axis
            # rotation, linear at the body origin), so the VBD entry sees the
            # correct kinematic velocity instead of a zero-velocity teleport.
            lin = np.gradient(root_pos, self.sim_dt, axis=0)
            ang = np.cross(axes_sub[:-1], np.diff(axes_sub, axis=0)) / self.sim_dt
            ang = np.vstack([ang, ang[-1:]])
            vel = np.concatenate([ang, lin], axis=1)
            self.root_vel_wp = wp.array(vel.astype(np.float32), dtype=wp.spatial_vector)

        self.viewer.set_model(self.model)
        self._set_camera()

        self.frame_index = 0
        self.rope_traj: list[np.ndarray] = []
        self.ee_traj: list[np.ndarray] = []
        self.joint_q_traj: list[np.ndarray] = []

        self.capture()

    # ------------------------------------------------------------------
    # Reference trajectory
    # ------------------------------------------------------------------

    def _phase_times(self):
        t0 = self.t_settle
        t1 = t0 + self.t_throw
        t2 = t1 + self.t_flight
        t3 = t2 + self.t_lift
        return t0, t1, t2, t3

    def _lasso_throw(self, u: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Parametric lasso throw: a tilted circular sweep plus a retreat.

        Unlike the digitized human command (which relied on wrist steering the
        demonstrator applied), this family is designed for a free-pivot rope:
        the hand itself traces the loop that the rope should inherit.
        """
        (rx, rz, yaw, y_amp, span_deg, loop_frac, rdx, rdy, rdz) = self.lasso_params
        span = np.radians(span_deg)
        theta0 = 0.0  # start at the loop's forward edge; sweep goes up and back
        # Split the throw into loop and retreat.
        ul = np.clip(u / loop_frac, 0.0, 1.0)
        ur = np.clip((u - loop_frac) / (1.0 - loop_frac), 0.0, 1.0)
        theta = theta0 + span * ul
        cy, sy = np.cos(yaw), np.sin(yaw)
        # Loop center chosen so the sweep starts at the hand's rest position.
        cx0 = -rx * np.cos(theta0)
        cz0 = -rz * np.sin(theta0)
        x = (cx0 + rx * np.cos(theta)) * cy
        y = (cx0 + rx * np.cos(theta)) * sy + y_amp * np.sin(theta - theta0)
        z = cz0 + rz * np.sin(theta)
        pos = start + np.stack([x, y, z], axis=1)
        # Retreat: pull away smoothly to strip the loop over the tip.
        smooth = ur * ur * (3.0 - 2.0 * ur)
        pos += np.outer(smooth, [rdx, rdy, rdz])
        return pos

    def _build_tip_schedule(self) -> np.ndarray:
        """Handle-tip positions at frame rate (task-space keyframes for IK)."""
        n = self.num_frames + 1
        t = np.arange(n) * self.frame_dt
        t0, t1, t2, t3 = self._phase_times()

        start = bezier_eval(BEZIER_CTRL, np.array([0.0]))[0]
        end = bezier_eval(BEZIER_CTRL, np.array([1.0]))[0]
        centroid = 0.5 * (BEZIER_CTRL.max(axis=0) + BEZIER_CTRL.min(axis=0))

        # Optionally amplify the follow-through (last three Bezier control
        # points) about the centroid: with a free-pivot rope attachment the
        # loop must be driven by the hand path itself, not by prescribing the
        # rope-root orientation as in the kinematic example.
        ctrl = BEZIER_CTRL.copy()
        if getattr(self, "_preset_delta", None) is not None:
            ctrl = ctrl + self._preset_delta
        elif self.bezier_delta_file:
            ctrl = ctrl + np.load(self.bezier_delta_file)["delta"]
        ctrl[5:] = centroid + self.follow_scale * (ctrl[5:] - centroid)
        end = bezier_eval(ctrl, np.array([1.0]))[0]

        if self.command_kind == "replay":
            # Replay a recorded rope-node trajectory from a successful
            # kinematic-example run: the motion of a node a few segments from
            # the steered root already contains the orientation-steering
            # effect a free-pivot attachment cannot transmit, so using it as
            # the hand-tip target reproduces the tail rope's knotting motion.
            d = np.load(self.replay_file)
            node_traj = d["rope_traj"][:, self.replay_node, :].astype(np.float64)
            pos = np.empty((n, 3))
            m = min(n, len(node_traj))
            pos[:m] = node_traj[:m]
            pos[m:] = node_traj[-1]
            # Keep the settle phase static at the initial node position.
            pos[t <= t0] = node_traj[0]
            self.tip_pos_frames = pos
            return pos

        pos = np.zeros((n, 3))
        pos[t <= t0] = start
        mask = (t > t0) & (t <= t1)
        s = (t[mask] - t0) / self.t_throw
        if self.command_kind == "lasso":
            throw = self._lasso_throw(s, start)
            pos[mask] = throw
            end_scaled = self._lasso_throw(np.array([1.0]), start)[0]
        else:
            pos[mask] = centroid + self.throw_scale * (bezier_eval(ctrl, s) - centroid)
            end_scaled = centroid + self.throw_scale * (end - centroid)
        pos[(t > t1) & (t <= t2)] = end_scaled
        # Optional cinch yank: a sharp pull shortly after the throw, timed to
        # capture the tip while it is threaded through the flying loop.
        if np.linalg.norm(self.yank_vec) > 0.0:
            y0 = t1 + self.yank_delay
            mask = (t > y0) & (t <= t2)
            u = np.clip((t[mask] - y0) / self.yank_t, 0.0, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            pos[mask] = pos[mask] + np.outer(smooth, self.yank_vec)
        base_end = end_scaled + self.yank_vec if np.linalg.norm(self.yank_vec) > 0.0 else end_scaled
        mask = (t > t2) & (t <= t3)
        u = np.clip((t[mask] - t2) / self.t_lift, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        pos[mask] = base_end + np.outer(smooth, [0.0, 0.0, self.lift_height])
        pos[t > t3] = base_end + np.array([0.0, 0.0, self.lift_height])
        pos[:, 2] += self.z_offset
        self.tip_pos_frames = pos
        return pos

    def _tip_axis(self, tips: np.ndarray) -> np.ndarray:
        """Handle axis (flange->tip) heuristic: down at rest, trails velocity."""
        vel = np.gradient(tips, self.frame_dt, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        down = np.array([0.0, 0.0, -1.0])
        w = np.clip(speed / 2.5, 0.0, 0.85)
        axis = (1.0 - w[:, None]) * down - w[:, None] * (vel / np.maximum(speed, 1e-9)[:, None])
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        t = np.arange(len(tips)) * self.frame_dt
        axis[t <= self.t_settle] = down
        return axis

    def _root_axis_schedule(self) -> tuple[np.ndarray, np.ndarray]:
        """Trailing-axis direction + twist-free quats at substep resolution."""
        t_sub = (np.arange(self.n_sub_total) + 1) * self.sim_dt
        t_frames = np.arange(len(self.tip_pos_frames)) * self.frame_dt
        tips_sub = np.stack([np.interp(t_sub, t_frames, self.tip_pos_frames[:, k]) for k in range(3)], axis=1)
        vel = np.gradient(tips_sub, self.sim_dt, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        down = np.array([0.0, 0.0, -1.0])
        w = np.clip(speed / 2.5, 0.0, 0.85)
        axes = (1.0 - w[:, None]) * down - w[:, None] * (vel / np.maximum(speed, 1e-9)[:, None])
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        k = max(1, int(0.02 / self.sim_dt))
        kernel = np.ones(k) / k
        for dim in range(3):
            axes[:, dim] = np.convolve(axes[:, dim], kernel, mode="same")
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        axes[t_sub <= self.t_settle] = down
        quats = self._axis_quats(axes)
        return axes, quats, tips_sub

    def _axis_quats(self, axes: np.ndarray) -> np.ndarray:
        """Twist-free EE orientation series aligning +z with the handle axis."""
        quats = np.zeros((len(axes), 4))
        q = quat_from_z_to(axes[0])
        quats[0] = q
        for i in range(1, len(axes)):
            q = quat_mul(quat_between(axes[i - 1], axes[i]), q)
            q /= np.linalg.norm(q)
            quats[i] = q
        return quats

    def _solve_arm_ik(self, tips: np.ndarray) -> np.ndarray:
        """Branch-continuous IK over frame-rate tip targets (7-DOF arm)."""
        axes = self._tip_axis(tips)
        flanges = tips - axes * HANDLE_LENGTH
        q_frames, errs = solve_branch_continuous_ik(
            self.urdf_xml,
            self.base_pos,
            tips,
            flanges,
            posture_weight=self.posture_weight,
            flange_weight=self.flange_weight,
            axis_weight=self.axis_weight,
            axis_quats=self._axis_quats(axes) if self.axis_weight > 0.0 else None,
            max_step=self.ik_max_step,
            iters=self.ik_iters,
        )
        self.ik_errors = errs
        qd = np.abs(np.diff(q_frames, axis=0)).max() * self.fps
        print(f"IK tracking error [m]: mean {errs.mean():.4f}, max {errs.max():.4f}; peak |qd| {qd:.1f} rad/s")
        return q_frames

    def _build_joint_reference(self, q_frames: np.ndarray):
        """Filter + C2 spline the IK samples; evaluate at substep rate."""
        q_smooth = zero_phase_smooth(q_frames, self.ref_filter_width)
        t_knots = np.arange(len(q_smooth)) * self.frame_dt
        coeffs = natural_cubic_spline_coeffs(t_knots, q_smooth)
        # Target for substep k is the reference at the end of that substep.
        t_eval = (np.arange(self.n_sub_total) + 1) * self.sim_dt
        q_ref, qd_ref = eval_spline(t_knots, coeffs, t_eval)
        self.q_ref_frames = q_smooth
        self.q_ref_sub = q_ref
        self.qd_ref_sub = qd_ref

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _set_camera(self):
        target = np.array([0.72, -0.10, self.z_offset + 0.22])
        pos = target + np.array([1.55, -1.30, 0.18])
        d = target - pos
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = math.degrees(math.atan2(d[2], np.linalg.norm(d[:2])))
        try:
            self.viewer.set_camera(wp.vec3(*pos), pitch, yaw)
        except (AttributeError, TypeError, NotImplementedError):
            pass

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def _drive_root(self, post_step: bool):
        """Prescribe the kinematic rope-root pose in parent and entry states."""
        if self.root_drive == "kinematic":
            wp.launch(
                drive_root_from_ee,
                dim=1,
                inputs=[
                    self.step_idx,
                    self.n_sub_total,
                    self.ee_body,
                    self.rope_bodies[0],
                    self.root_local,
                    wp.vec3(0.0, 0.0, HANDLE_LENGTH),
                    self.seg_len / 2,
                    self.root_axis_wp,
                    self.root_quat_wp,
                ],
                outputs=[
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.vbd_entry_state_0.body_q,
                    self.vbd_entry_state_1.body_q,
                ],
            )
        elif self.root_drive == "command":
            wp.launch(
                drive_root_from_schedule,
                dim=1,
                inputs=[
                    self.step_idx,
                    self.n_sub_total,
                    self.rope_bodies[0],
                    self.root_local,
                    self.root_pos_wp,
                    self.root_quat_wp,
                    self.root_vel_wp,
                ],
                outputs=[
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_qd,
                    self.vbd_entry_state_0.body_q,
                    self.vbd_entry_state_1.body_q,
                    self.vbd_entry_state_0.body_qd,
                    self.vbd_entry_state_1.body_qd,
                ],
            )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.substep()

    def substep(self):
        """Advance the coupled simulation by one substep."""
        self.state_0.clear_forces()
        wp.launch(
            apply_joint_targets,
            dim=self.n_arm_coords,
            inputs=[
                self.step_idx,
                self.n_sub_total,
                self.q_ref_wp,
                self.qd_ref_wp,
                self.n_arm_coords,
            ],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
        )
        if self.air_drag > 0.0:
            wp.launch(
                apply_rope_air_drag,
                dim=len(self.rope_bodies),
                inputs=[
                    self.rope_body_wp,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.drag_k_perp,
                    self.drag_k_par,
                ],
                outputs=[self.state_0.body_f],
            )
        self._drive_root(post_step=False)
        self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        # The coupled solver reconciles entry output states over the parent
        # output, and prescribed kinematic poses do not survive the entry
        # solve, so re-apply the root prescription to the output state.
        self._drive_root(post_step=True)
        newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(advance_step, dim=1, inputs=[self.step_idx])

    def rope_centerline(self, body_q: np.ndarray) -> np.ndarray:
        n = len(self.rope_bodies)
        nodes = np.zeros((n + 1, 3))
        for i, b in enumerate(self.rope_bodies):
            t = body_q[b]
            pos, quat = t[:3], t[3:]
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
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        body_q = self.state_0.body_q.numpy()
        self.rope_traj.append(self.rope_centerline(body_q))
        self.ee_traj.append(body_q[self.ee_body].copy())
        self.joint_q_traj.append(self.state_0.joint_q.numpy()[: self.n_arm_coords].copy())

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

        nodes = self.rope_traj[-1] if self.rope_traj else self.rope_centerline(body_q)
        metrics = self.knot_metrics(nodes)

        # Arm tracking quality: actual joint positions vs the smooth reference.
        joint_q = np.array(self.joint_q_traj)
        n = min(len(joint_q), len(self.q_ref_frames) - 1)
        track_err = np.abs(joint_q[:n] - self.q_ref_frames[1 : n + 1]).max() if n else 0.0
        print(
            f"final rope metrics: writhe {metrics['writhe']:+.2f}, "
            f"crossings {metrics['crossings']}, "
            f"end-to-end/arc {metrics['length_ratio']:.3f}; "
            f"max joint tracking error {track_err:.4f} rad"
        )

        if self.save_traj:
            np.savez_compressed(
                self.save_traj,
                rope_traj=np.array(self.rope_traj, dtype=np.float32),
                ee_traj=np.array(self.ee_traj, dtype=np.float32),
                joint_q_traj=joint_q.astype(np.float32),
                q_ref_frames=self.q_ref_frames.astype(np.float32),
                tip_pos=self.tip_pos_frames.astype(np.float32),
                metrics_writhe=metrics["writhe"],
                metrics_crossings=metrics["crossings"],
                metrics_length_ratio=metrics["length_ratio"],
                ik_errors=self.ik_errors,
            )
            print(f"saved trajectory to {self.save_traj}")

        if self.expect_knot:
            assert abs(metrics["writhe"]) > 2.0, f"no knot: writhe {metrics['writhe']:.2f}"
            assert metrics["length_ratio"] < 0.95, f"no knot: rope taut ratio {metrics['length_ratio']:.3f}"


def add_arguments(parser):
    parser.add_argument("--time-scale", type=float, default=0.8, dest="time_scale")
    parser.add_argument("--throw-scale", type=float, default=1.0, dest="throw_scale")
    parser.add_argument("--z-offset", type=float, default=1.1, dest="z_offset")
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--vbd-iterations", type=int, default=10, dest="vbd_iterations")
    parser.add_argument("--mujoco-iterations", type=int, default=20, dest="mujoco_iterations")
    parser.add_argument("--proxy-iterations", type=int, default=1, dest="proxy_iterations")
    parser.add_argument("--coupling-mode", type=str, default="lagged", dest="coupling_mode")
    parser.add_argument("--mass-scale", type=float, default=1.0, dest="mass_scale")
    parser.add_argument("--ref-filter-width", type=int, default=5, dest="ref_filter_width")
    parser.add_argument("--posture-weight", type=float, default=0.1, dest="posture_weight")
    parser.add_argument("--flange-weight", type=float, default=0.0, dest="flange_weight")
    parser.add_argument("--attach-bend-scale", type=float, default=1.0, dest="attach_bend_scale")
    parser.add_argument("--follow-scale", type=float, default=1.0, dest="follow_scale")
    parser.add_argument("--axis-weight", type=float, default=0.0, dest="axis_weight")
    parser.add_argument("--bezier-delta-file", type=str, default=None, dest="bezier_delta_file")
    parser.add_argument(
        "--command",
        type=str,
        default="digitized",
        choices=["digitized", "searched", "searched-hires", "lasso", "replay"],
    )
    parser.add_argument("--replay-file", type=str, default=None, dest="replay_file")
    parser.add_argument("--replay-node", type=int, default=3, dest="replay_node")
    parser.add_argument("--yank-dx", type=float, default=0.0, dest="yank_dx")
    parser.add_argument("--yank-dy", type=float, default=0.0, dest="yank_dy")
    parser.add_argument("--yank-dz", type=float, default=0.0, dest="yank_dz")
    parser.add_argument("--yank-delay", type=float, default=0.2, dest="yank_delay")
    parser.add_argument("--yank-t", type=float, default=0.2, dest="yank_t")
    parser.add_argument("--lasso-file", type=str, default=None, dest="lasso_file")
    parser.add_argument(
        "--air-drag",
        type=float,
        default=0.0,
        dest="air_drag",
        help="Aerodynamic drag multiplier for the rope (1.0 = physical cylinder drag).",
    )
    parser.add_argument("--ik-max-step", type=float, default=0.02, dest="ik_max_step")
    parser.add_argument("--ik-iters", type=int, default=12, dest="ik_iters")
    parser.add_argument(
        "--root-drive", type=str, default="attached", choices=["attached", "kinematic", "command"], dest="root_drive"
    )
    parser.add_argument("--gain-scale", type=float, default=1.0, dest="gain_scale")
    parser.add_argument("--effort-scale", type=float, default=1.0, dest="effort_scale")
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
    parser.add_argument("--save-traj", type=str, default=None, dest="save_traj")
    parser.add_argument("--expect-knot", action="store_true", dest="expect_knot")
    parser.add_argument("--t-settle", type=float, default=1.5, dest="t_settle")
    parser.add_argument("--t-flight", type=float, default=2.0, dest="t_flight")
    parser.add_argument("--t-lift", type=float, default=2.5, dest="t_lift")
    parser.add_argument("--t-hold", type=float, default=1.5, dest="t_hold")
    parser.add_argument("--lift-height", type=float, default=0.7, dest="lift_height")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    add_arguments(parser)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
