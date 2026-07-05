# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoothness comparison figures: kinematic vs MuJoCo-coupled flying knot.

Usage: uv run python scripts/flying_knot/make_smoothness_figures.py KIN.npz MJC.npz OUTDIR
"""

import sys
from pathlib import Path

import matplotlib  # noqa: TID253

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np

FPS = 60


def main():
    kin = np.load(sys.argv[1])
    mjc = np.load(sys.argv[2])
    outdir = Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)

    qk = kin["arm_joint_q"][:, :7]
    qm = mjc["joint_q_traj"]
    qm_ref = mjc["q_ref_frames"]

    def derivs(q):
        qd = np.diff(q, axis=0) * FPS
        qdd = np.diff(qd, axis=0) * FPS
        return qd, qdd

    qd_k, qdd_k = derivs(qk)
    qd_m, qdd_m = derivs(qm)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    tk = np.arange(len(qd_k)) / FPS
    tm = np.arange(len(qd_m)) / FPS
    for col, (qd, qdd, t, title) in enumerate(
        [
            (qd_k, qdd_k, tk, "kinematic example (raw per-frame IK)"),
            (qd_m, qdd_m, tm, "MuJoCo-coupled example (C2 reference + dynamics)"),
        ]
    ):
        ax = axes[0, col]
        ax.plot(t, np.abs(qd).max(axis=1), lw=1.0, color="tab:blue")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("max |joint velocity| [rad/s]" if col == 0 else "")
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        ax.plot(t[:-1], np.abs(qdd).max(axis=1), lw=1.0, color="tab:red")
        ax.set_ylabel("max |joint accel| [rad/s$^2$]" if col == 0 else "")
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
    # Shared y-limits per row to make the contrast honest but readable (log scale).
    for row in range(2):
        for col in range(2):
            axes[row, col].set_yscale("log")
        lims = [axes[row, c].get_ylim() for c in range(2)]
        lo = min(l[0] for l in lims)
        hi = max(l[1] for l in lims)
        for col in range(2):
            axes[row, col].set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(outdir / "smoothness_comparison.png", dpi=150)

    # Summary metrics.
    jerk_k = np.diff(qdd_k, axis=0) * FPS
    jerk_m = np.diff(qdd_m, axis=0) * FPS
    n = min(len(qm), len(qm_ref) - 1)
    track = np.abs(qm[:n] - qm_ref[1 : n + 1]).max()
    print(
        f"kinematic: peak |qd| {np.abs(qd_k).max():.1f}  |qdd| {np.abs(qdd_k).max():.0f}  |jerk| {np.abs(jerk_k).max():.2e}"
    )
    print(
        f"mujoco:    peak |qd| {np.abs(qd_m).max():.1f}  |qdd| {np.abs(qdd_m).max():.0f}  |jerk| {np.abs(jerk_m).max():.2e}"
    )
    print(f"mujoco max joint tracking error vs C2 reference: {track:.4f} rad")


if __name__ == "__main__":
    main()
