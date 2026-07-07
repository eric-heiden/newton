# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Knot-bundle discretization comparison: 36 vs 72 cable segments.

Usage: uv run python scripts/flying_knot/make_resolution_figure.py TRAJ36.npz TRAJ72.npz OUT.png
"""

import sys

import matplotlib  # noqa: TID253

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np


def knot_window(nodes: np.ndarray) -> tuple[int, int]:
    """Node index range spanning the knot bundle (densest self-proximity)."""
    n = len(nodes)
    best_i, best_c = 0, -1
    for i in range(n):
        d = np.linalg.norm(nodes - nodes[i], axis=1)
        # Count non-neighbor nodes within 3 cm.
        idx = np.where(d < 0.03)[0]
        c = int(np.sum(np.abs(idx - i) > 3))
        if c > best_c:
            best_i, best_c = i, c
    lo = max(0, best_i - n // 6)
    hi = min(n, best_i + n // 6)
    return lo, hi


def main():
    t36 = np.load(sys.argv[1])["rope_traj"][-1]
    t72 = np.load(sys.argv[2])["rope_traj"][-1]
    out = sys.argv[3]

    fig = plt.figure(figsize=(11, 5.2))
    for k, (nodes, title) in enumerate([(t36, "36 segments (30.6 mm)"), (t72, "72 segments (15.3 mm)")]):
        lo, hi = knot_window(nodes)
        sel = nodes[lo:hi]
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot(sel[:, 0], sel[:, 1], sel[:, 2], "-", color="chocolate", lw=2.2)
        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], color="saddlebrown", s=14, depthshade=False)
        c = sel.mean(axis=0)
        r = 0.055
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{title}\n{hi - lo} nodes through the knot bundle", fontsize=11)
        ax.view_init(elev=12, azim=-70)
        ax.tick_params(labelsize=6)
    fig.suptitle("Final hanging knot centerline, node-resolved", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
