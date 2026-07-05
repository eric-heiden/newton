# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Search fixed arm mounts that minimize peak joint velocity over the throw.

The reachability-only mount (fixed_base_search.py) puts the whip near the
workspace boundary, where even branch-continuous IK needs huge joint rates.
This script scores geometrically feasible mounts by the peak and 95th
percentile joint velocity of the branch-continuous IK over the throw window.

Usage: uv run python scripts/flying_knot/mount_velocity_search.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))

from example_cable_flying_knot import HANDLE_LENGTH, default_xarm_dir, preprocess_xarm_urdf  # noqa: E402
from example_cable_flying_knot_mujoco import Example, solve_branch_continuous_ik  # noqa: E402

SHOULDER_HEIGHT = 0.267
REACH_MIN, REACH_MAX = 0.18, 0.68
FPS = 60


def build_targets():
    class _Args(SimpleNamespace):
        def __getattr__(self, name):
            raise AttributeError(name)

    args = SimpleNamespace()
    ex = object.__new__(Example)
    ex.args = args
    ex.fps = FPS
    ex.frame_dt = 1.0 / FPS
    ex.sim_substeps = 32
    ex.sim_dt = ex.frame_dt / 32
    ex.time_scale = 0.8
    ex.throw_scale = 1.0
    ex.z_offset = 1.1
    ex.t_settle = 1.5
    ex.t_throw = 0.7 * ex.time_scale
    ex.t_flight = 2.0
    ex.t_lift = 2.5
    ex.t_hold = 1.5
    ex.duration = ex.t_settle + ex.t_throw + ex.t_flight + ex.t_lift + ex.t_hold
    ex.num_frames = int(round(ex.duration * FPS))
    ex.lift_height = 0.7
    ex.n_sub_total = ex.num_frames * 32
    tips = ex._build_tip_schedule()
    axes = ex._tip_axis(tips)
    flanges = tips - axes * HANDLE_LENGTH
    t = np.arange(len(tips)) / FPS
    # Wider window: include lead-in and follow-through around the throw.
    window = (t >= ex.t_settle - 0.3) & (t <= ex.t_settle + ex.t_throw + 0.4)
    return tips, flanges, window


def geometric_ok(mount, flanges):
    shoulder = np.array(mount) + np.array([0.0, 0.0, SHOULDER_HEIGHT])
    d = np.linalg.norm(flanges - shoulder, axis=1)
    return d.min() > REACH_MIN and d.max() < REACH_MAX


def main():
    xarm_dir = default_xarm_dir()
    urdf_xml = preprocess_xarm_urdf(xarm_dir / "xarm7.urdf")
    tips, flanges, window = build_targets()
    tips_w = tips[window]
    flanges_w = flanges[window]

    z0 = 1.1
    xs = np.arange(0.05, 0.56, 0.1)
    ys = np.arange(-0.55, 0.16, 0.1)
    zs = z0 + np.arange(0.0, 0.51, 0.125)
    candidates = [
        (x, y, z) for x in xs for y in ys for z in zs if geometric_ok((x, y, z), flanges)
    ]
    print(f"{len(candidates)} geometrically feasible mounts")

    results = []
    for i, mount in enumerate(candidates):
        q, errs = solve_branch_continuous_ik(
            urdf_xml,
            np.array(mount),
            tips_w,
            flanges_w,
            posture_weight=0.1,
            flange_weight=0.0,
            max_step=0.04,
            iters=8,
        )
        qd = np.abs(np.diff(q, axis=0)) * FPS
        peak = float(qd.max())
        p95 = float(np.percentile(qd, 99.5))
        err = float(errs.max())
        results.append((peak, p95, err, mount))
        print(
            f"[{i + 1}/{len(candidates)}] mount=({mount[0]:.2f},{mount[1]:.2f},{mount[2]:.2f}) "
            f"peak|qd|={peak:7.1f} p99.5={p95:6.1f} maxerr={err:.3f}",
            flush=True,
        )

    results.sort(key=lambda r: (r[0] if r[2] < 0.05 else r[0] + 1e3))
    print("\nbest mounts (err < 5 cm):")
    for peak, p95, err, mount in results[:8]:
        print(f"  ({mount[0]:.2f}, {mount[1]:.2f}, {mount[2]:.2f})  peak {peak:.1f} rad/s  p99.5 {p95:.1f}  err {err:.3f}")


if __name__ == "__main__":
    main()
