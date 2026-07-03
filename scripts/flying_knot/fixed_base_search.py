# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose base-joint usage and search for a feasible fixed mount pose.

Stage 1: solve the current 10-DOF (3 base + 7 joint) IK and report how much
the base translates -- the source of the "moving pedestal" artifact.
Stage 2: geometric prune of candidate fixed mounts (flange targets must stay
inside an annulus around the shoulder), then full 7-DOF IK scoring of the
best candidates.

Usage: uv run python scripts/flying_knot/fixed_base_search.py [--stage 1|2|all]
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))

from example_cable_flying_knot import (  # noqa: E402
    HANDLE_LENGTH,
    Example,
    default_xarm_dir,
    preprocess_xarm_urdf,
)

import newton  # noqa: E402
import newton.ik as ik  # noqa: E402

SHOULDER_HEIGHT = 0.267  # xArm7 link_base top to joint2 axis [m]


def build_schedule(lift_height=0.7):
    """Handle-tip schedule at frame rate via a rope-only Example instance."""
    viewer = newton.viewer.ViewerNull()
    args = SimpleNamespace(no_arm=True, lift_height=lift_height)
    ex = Example(viewer, args)
    idx = np.clip(np.arange(ex.num_frames + 1) * ex.sim_substeps, 0, ex.n_sub_total - 1)
    tips = ex.tip_pos[idx]
    axes = ex.tip_axis[idx]
    flanges = tips - axes * HANDLE_LENGTH
    t0, t1, _, _ = ex._phase_times()
    throw_mask = (idx * ex.sim_dt >= t0) & (idx * ex.sim_dt <= t1 + 0.1)
    return ex, tips, flanges, throw_mask


def solve_ik_over_schedule(urdf_xml, base_pos, base_joint, tips, flanges, iters=24, first_iters=96):
    builder = newton.ModelBuilder()
    builder.add_urdf(
        urdf_xml,
        xform=wp.transform(wp.vec3(*base_pos), wp.quat_identity()),
        base_joint=base_joint,
        floating=False if base_joint is None else None,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
    )
    body_keys = list(builder.body_label)
    model = builder.finalize()
    ee_index = next(i for i, k in enumerate(body_keys) if k.endswith("link_eef"))

    flange_obj = ik.IKObjectivePosition(
        link_index=ee_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(*flanges[0])], dtype=wp.vec3),
    )
    tip_obj = ik.IKObjectivePosition(
        link_index=ee_index,
        link_offset=wp.vec3(0.0, 0.0, HANDLE_LENGTH),
        target_positions=wp.array([wp.vec3(*tips[0])], dtype=wp.vec3),
    )
    limit_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=10.0,
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[flange_obj, tip_obj, limit_obj],
        optimizer=ik.IKOptimizer.LM,
        jacobian_mode=ik.IKJacobianType.AUTODIFF,
    )
    n_coords = model.joint_coord_count
    joint_q = wp.zeros((1, n_coords), dtype=wp.float32)
    init = np.zeros(n_coords, dtype=np.float32)
    # Elbow-ready initial guess (skip base coords if present).
    off = n_coords - 7
    init[off + 1] = 0.6
    init[off + 3] = 0.8
    joint_q.assign(init.reshape(1, -1))

    state = model.state()
    errs = np.zeros(len(tips))
    qs = np.zeros((len(tips), n_coords), dtype=np.float32)
    for i in range(len(tips)):
        flange_obj.set_target_position(0, wp.vec3(*flanges[i]))
        tip_obj.set_target_position(0, wp.vec3(*tips[i]))
        solver.step(joint_q, joint_q, iterations=iters if i else first_iters)
        q_np = joint_q.numpy()[0]
        qs[i] = q_np
        newton.eval_fk(model, wp.array(q_np, dtype=wp.float32), model.joint_qd, state)
        errs[i] = np.linalg.norm(state.body_q.numpy()[ee_index][:3] - flanges[i])
    return qs, errs


def stage1(urdf_xml, tips, flanges, throw_mask, z_offset):
    base_pos = np.array([0.0, 0.0, z_offset + 0.3])
    base_lim = 0.2
    base_joint = {
        "joint_type": newton.JointType.D6,
        "linear_axes": [
            newton.ModelBuilder.JointDofConfig(axis=ax, limit_lower=-base_lim, limit_upper=base_lim)
            for ax in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        ],
    }
    qs, errs = solve_ik_over_schedule(urdf_xml, base_pos, base_joint, tips, flanges)
    base = qs[:, :3]
    print("== stage 1: current 10-DOF solution, base joint usage ==")
    print(f"base x range [{base[:, 0].min():+.3f}, {base[:, 0].max():+.3f}] m")
    print(f"base y range [{base[:, 1].min():+.3f}, {base[:, 1].max():+.3f}] m")
    print(f"base z range [{base[:, 2].min():+.3f}, {base[:, 2].max():+.3f}] m")
    print(f"base displacement span: {np.ptp(base, axis=0)} m")
    print(f"flange err mean {errs.mean() * 1000:.2f} mm, max {errs.max() * 1000:.2f} mm")


def geometric_score(mount, flanges):
    """Annulus violation of flange targets around the shoulder."""
    shoulder = mount + np.array([0.0, 0.0, SHOULDER_HEIGHT])
    d = np.linalg.norm(flanges - shoulder, axis=1)
    r_max, r_min = 0.66, 0.20
    return np.maximum(0, d - r_max).max() + np.maximum(0, r_min - d).max(), d.min(), d.max()


def stage2(urdf_xml, tips, flanges, throw_mask, z_offset):
    print("== stage 2: fixed-base mount search ==")
    best = []
    for x in np.arange(-0.05, 0.45, 0.05):
        for y in np.arange(-0.5, 0.35, 0.05):
            for dz in np.arange(0.1, 0.55, 0.05):
                mount = np.array([x, y, z_offset + dz])
                viol, dmin, dmax = geometric_score(mount, flanges)
                best.append((viol, mount, dmin, dmax))
    best.sort(key=lambda r: r[0])
    print("top geometric candidates (violation, mount, d_min, d_max):")
    for viol, mount, dmin, dmax in best[:8]:
        print(
            f"  viol={viol * 1000:6.1f} mm  mount=({mount[0]:+.2f},{mount[1]:+.2f},{mount[2]:.2f})  "
            f"d=[{dmin:.3f},{dmax:.3f}]"
        )

    for _viol, mount, _, _ in best[:4]:
        _qs, errs = solve_ik_over_schedule(urdf_xml, mount, None, tips, flanges)
        print(
            f"IK @ mount ({mount[0]:+.2f},{mount[1]:+.2f},{mount[2]:.2f}): "
            f"err mean {errs.mean() * 1000:.1f} mm, max {errs.max() * 1000:.1f} mm, "
            f"throw-window max {errs[throw_mask].max() * 1000:.1f} mm"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--lift-height", type=float, default=0.7)
    args = ap.parse_args()

    urdf_xml = preprocess_xarm_urdf(default_xarm_dir() / "xarm7.urdf")
    ex, tips, flanges, throw_mask = build_schedule(lift_height=args.lift_height)
    print(f"schedule: {len(tips)} frames, lift height {args.lift_height} m")

    if args.stage in ("1", "all"):
        stage1(urdf_xml, tips, flanges, throw_mask, ex.z_offset)
    if args.stage in ("2", "all"):
        stage2(urdf_xml, tips, flanges, throw_mask, ex.z_offset)


if __name__ == "__main__":
    main()
