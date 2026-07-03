# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot rope centerline snapshots from a --save-traj npz for tuning inspection.

Usage: uv run python scripts/flying_knot/plot_rope_traj.py TRAJ.npz [OUT.png]
"""

import sys
from pathlib import Path

import matplotlib  # noqa: TID253

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np


def main():
    path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_suffix(".png")
    d = np.load(path)
    traj = d["rope_traj"]  # [frames, nodes, 3]
    n_frames = len(traj)

    # Sample frames around the throw (settle=1.5s -> frame 90; throw 0.7s -> ~42 frames).
    fps = 60
    times = [1.4, 1.7, 1.9, 2.05, 2.2, 2.35, 2.5, 2.7, 3.0, 3.5, 5.0, n_frames / fps - 0.05]
    frames = [min(int(t * fps), n_frames - 1) for t in times]

    fig = plt.figure(figsize=(16, 12))
    for k, f in enumerate(frames):
        ax = fig.add_subplot(3, 4, k + 1, projection="3d")
        pts = traj[f]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color="chocolate", lw=1.5)
        ax.scatter(*pts[0], color="black", s=25)  # handle end
        ax.scatter(*pts[-1], color="navy", s=25)  # weighted tip
        ax.set_title(f"t={f / fps:.2f}s", fontsize=9)
        ax.set_xlim(0.2, 1.3)
        ax.set_ylim(-0.8, 0.4)
        ax.set_zlim(0, 2.0)
        ax.set_box_aspect((1.1, 1.2, 2.0))
        ax.tick_params(labelsize=5)
    fig.suptitle(
        f"writhe={float(d['metrics_writhe']):+.2f} crossings={int(d['metrics_crossings'])} ratio={float(d['metrics_length_ratio']):.3f}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
