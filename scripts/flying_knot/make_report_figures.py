# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate figures for the flying-knots report.

Usage: uv run python scripts/flying_knot/make_report_figures.py OUTDIR
"""

import json
import sys
from pathlib import Path

import matplotlib  # noqa: TID253

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))

from example_cable_flying_knot import polyline_writhe  # noqa: E402

FPS = 60


def writhe_series(npz_path, stride=2):
    traj = np.load(npz_path)["rope_traj"]
    idx = np.arange(0, len(traj), stride)
    return idx / FPS, np.array([polyline_writhe(traj[i]) for i in idx])


def fig_writhe(outdir: Path):
    t1, w1 = writhe_series("/tmp/fk/champion.npz")
    t2, w2 = writhe_series("/tmp/fk/paper_faithful.npz")
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(t1, w1, color="tab:green", lw=2, label="tuned command (knot: writhe locks in near +3)")
    ax.plot(t2, w2, color="tab:red", lw=2, label="verbatim digitized command (knot never locks)")
    for x, lbl, ha in [
        (1.5, "throw starts ", "right"),
        (1.5 + 0.56, " throw ends", "left"),
        (4.06, " lift starts", "left"),
    ]:
        ax.axvline(x, color="gray", ls=":", lw=1)
        ax.text(x, 3.75, lbl, fontsize=8, color="gray", va="top", ha=ha)
    ax.set_ylim(-1.2, 3.9)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Rope centerline writhe")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(outdir / "writhe_over_time.png", dpi=150)
    print("writhe fig done")


def fig_ik(outdir: Path):
    d = np.load("/tmp/fk/champ_arm/champion_000_r0.npz")
    errs = d["ik_errors"] * 1000.0
    t = np.arange(len(errs)) / FPS
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(t, errs, color="tab:blue", lw=1.5)
    ax.axvspan(1.5, 1.5 + 0.56, color="orange", alpha=0.15, label="throw window")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("IK flange tracking error [mm]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "ik_error.png", dpi=150)
    print(f"ik fig done (mean {errs.mean():.2f} mm, max {errs.max():.2f} mm)")


def success_table(outdir: Path):
    rows = []
    for name in ["/tmp/fk/sweep_focus/results_focus.json", "/tmp/fk/basin/results_basin.json"]:
        p = Path(name)
        if not p.exists():
            continue
        results = json.load(open(p))
        by_cfg = {}
        for rec in results:
            key = json.dumps(rec["params"], sort_keys=True)
            by_cfg.setdefault(key, []).append(rec.get("knotted", False))
        for key, oks in by_cfg.items():
            params = json.loads(key)
            rows.append(
                {
                    "grid": p.stem.replace("results_", ""),
                    "time_scale": params.get("time-scale"),
                    "tip_mass": params.get("tip-mass"),
                    "bend_stiffness": params.get("bend-stiffness"),
                    "success": sum(oks),
                    "trials": len(oks),
                }
            )
    with open(outdir / "success_rates.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"success table done ({len(rows)} configs)")


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    fig_writhe(outdir)
    fig_ik(outdir)
    success_table(outdir)


if __name__ == "__main__":
    main()
