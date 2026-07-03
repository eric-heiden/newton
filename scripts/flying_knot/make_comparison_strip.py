# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side phase strip: human demonstration (paper video) vs Newton sim.

Usage: uv run python scripts/flying_knot/make_comparison_strip.py REF.mp4 SIM.mp4 OUT.png
  REF: human_slomo.mp4 from flying-knots.github.io
  SIM: flying_knot_throw_slowmo.mp4 (starts at sim t=1.3 s, 4x slow motion)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib  # noqa: TID253

matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np
from PIL import Image  # noqa: TID253

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# (label, ref video time [s], sim video time [s])
# Sim video: recorded from sim t=1.3 at 4x slowmo -> video_t = (sim_t - 1.3) * 4.
PHASES = [
    ("rest", 0.3, 0.3),
    ("upswing", 1.75, 1.7),
    ("loop opens", 3.0, 2.1),
    ("tip threads loop", 3.6, 2.5),
    ("knot in flight", 4.4, 3.0),
    ("knot hangs", 6.8, 7.5),
]


def grab(video: Path, t: float, out: Path):
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-vframes", "1", str(out)],
        check=True,
    )
    return np.asarray(Image.open(out))


def main():
    ref, sim, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    n = len(PHASES)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6.2))
    with tempfile.TemporaryDirectory() as td:
        for k, (label, t_ref, t_sim) in enumerate(PHASES):
            img_r = grab(ref, t_ref, Path(td) / f"r{k}.png")
            img_s = grab(sim, t_sim, Path(td) / f"s{k}.png")
            # Crop reference to the right-hand thrower region and sim to center.
            _h, w = img_r.shape[:2]
            img_r = img_r[:, int(0.45 * w) :]
            _h, w = img_s.shape[:2]
            img_s = img_s[:, int(0.18 * w) : int(0.82 * w)]
            axes[0, k].imshow(img_r)
            axes[1, k].imshow(img_s)
            axes[0, k].set_title(label, fontsize=11)
            for ax in (axes[0, k], axes[1, k]):
                ax.axis("off")
    axes[0, 0].text(
        -0.08,
        0.5,
        "human demo\n(paper video)",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=11,
    )
    axes[1, 0].text(
        -0.08,
        0.5,
        "Newton\nSolverVBD",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
