# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Black-box throw-command search for the MuJoCo-coupled flying knot.

Plays the role of the paper's task-level ILC: the digitized human command
does not knot under the coupled (free-pivot) rope dynamics, so we search
perturbations of the middle Bezier control points plus timing and tip mass.
A cross-entropy-style loop keeps the best quantile each generation and
samples around it. Objective favors transient coiling (max |writhe|) and
rewards a locked final knot.

Usage: uv run python scripts/flying_knot/command_search.py [--generations 6] [--pop 12]
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))

from example_cable_flying_knot import count_crossings, polyline_writhe  # noqa: E402

# Perturbation applies to control points 2..4 (the throw's core) in x/y/z.
N_CTRL = 3
PARAM_DIM = N_CTRL * 3 + 2  # + time-scale + tip-mass


def unpack(theta):
    delta = np.zeros((8, 3))
    delta[2:5] = theta[: N_CTRL * 3].reshape(N_CTRL, 3)
    time_scale = float(np.clip(theta[-2], 0.6, 1.0))
    tip_mass = float(np.clip(theta[-1], 0.03, 0.09))
    return delta, time_scale, tip_mass


def evaluate(theta, tag, outdir):
    delta, time_scale, tip_mass = unpack(theta)
    delta_file = outdir / f"{tag}_delta.npz"
    npz = outdir / f"{tag}.npz"
    np.savez(delta_file, delta=delta)
    cmd = [
        "uv",
        "run",
        "-m",
        "newton.examples",
        "cable_flying_knot_mujoco",
        "--viewer",
        "null",
        "--test",
        "--num-frames",
        "484",
        "--time-scale",
        str(time_scale),
        "--tip-mass",
        str(tip_mass),
        "--bezier-delta-file",
        str(delta_file),
        "--save-traj",
        str(npz),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200, check=False)
    if proc.returncode != 0 or not npz.exists():
        return {"tag": tag, "score": -10.0, "error": proc.stderr[-300:]}
    d = np.load(npz)
    traj = d["rope_traj"]
    if not np.isfinite(traj).all():
        return {"tag": tag, "score": -10.0, "error": "non-finite"}
    sample = range(60, len(traj), 4)
    writhes = np.array([polyline_writhe(traj[f]) for f in sample])
    final_writhe = polyline_writhe(traj[-1])
    e2e = np.linalg.norm(traj[-1][-1] - traj[-1][0])
    arc = np.linalg.norm(np.diff(traj[-1], axis=0), axis=1).sum()
    ratio = e2e / arc
    knotted = abs(final_writhe) > 2.0 and ratio < 0.95
    crossings = max(count_crossings(traj[-1], axis=0), count_crossings(traj[-1], axis=1))
    score = float(np.abs(writhes).max()) + (5.0 + abs(final_writhe)) * float(knotted)
    return {
        "tag": tag,
        "score": round(score, 3),
        "max_abs_writhe": round(float(np.abs(writhes).max()), 3),
        "final_writhe": round(float(final_writhe), 3),
        "ratio": round(float(ratio), 3),
        "crossings": int(crossings),
        "knotted": bool(knotted),
        "theta": [round(float(v), 4) for v in theta],
        "elapsed": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="/tmp/fk/cmd_search")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mean = np.zeros(PARAM_DIM)
    mean[-2] = 0.78  # time-scale
    mean[-1] = 0.055  # tip-mass
    std = np.concatenate([np.full(N_CTRL * 3, 0.10), [0.06, 0.012]])

    history = []
    best = None
    for gen in range(args.generations):
        thetas = [mean + std * rng.standard_normal(PARAM_DIM) for _ in range(args.pop)]
        if best is not None:
            thetas[0] = np.array(best["theta"])  # keep the incumbent
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            recs = list(pool.map(lambda p: evaluate(p[1], f"g{gen}_i{p[0]}", outdir), enumerate(thetas)))
        recs.sort(key=lambda r: -r["score"])
        history.extend(recs)
        if best is None or recs[0]["score"] > best["score"]:
            best = recs[0]
        elites = [np.array(r["theta"]) for r in recs[: args.elite] if "theta" in r]
        if elites:
            mean = np.mean(elites, axis=0)
            std = 0.75 * std + 0.25 * np.std(elites, axis=0)
            std = np.maximum(std, np.concatenate([np.full(N_CTRL * 3, 0.02), [0.01, 0.003]]))
        print(
            f"gen {gen}: best {recs[0]['score']} (all-time {best['score']}) "
            f"wr_max={recs[0].get('max_abs_writhe')} knotted={recs[0].get('knotted')}",
            flush=True,
        )
        with open(outdir / "history.json", "w") as f:
            json.dump({"history": history, "best": best}, f, indent=1)
    print("all-time best:", json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
