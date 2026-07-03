# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parameter sweep driver for the flying-knot example.

Runs the example headless over a grid of parameters, computes knot metrics
over the whole rope trajectory (not just the final frame), and writes a
JSON results table.

Usage: uv run python scripts/flying_knot/sweep.py [--grid NAME] [--out results.json]
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))

from example_cable_flying_knot import count_crossings, polyline_writhe  # noqa: E402


def knot_cluster(nodes, sep=4, thresh=0.05):
    """Indices of rope nodes in self-contact proximity (the knot)."""
    diff = nodes[:, None, :] - nodes[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    n = len(nodes)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = (jj - ii >= sep) & (dist < thresh)
    return np.unique(np.concatenate([ii[mask], jj[mask]]))


def knot_pos_fraction(nodes):
    """Mean arc-length fraction (0 = handle, 1 = tip) of the knot cluster."""
    idx = knot_cluster(nodes)
    if len(idx) == 0:
        return None
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    return float(arc[idx].mean() / arc[-1])


GRIDS = {
    "broad": {
        "time-scale": [0.8, 1.0],
        "throw-scale": [1.0, 1.3, 1.6],
        "tip-mass": [0.015, 0.03, 0.06],
        "bend-stiffness": [1e-3, 5e-3, 2e-2],
        "stretch-damping": [0.5],
    },
    "focus": {
        "time-scale": [0.75, 0.8, 0.85],
        "throw-scale": [1.0],
        "tip-mass": [0.05, 0.06, 0.07],
        "bend-stiffness": [5e-4, 1e-3, 2e-3],
        "stretch-damping": [0.5],
    },
    "retention": {
        "time-scale": [0.8],
        "tip-mass": [0.06],
        "bend-stiffness": [1e-3],
        "stretch-damping": [0.5],
        "friction": [1.0, 1.5, 2.0],
        "t-flight": [0.8, 2.0],
        "rope-segments": [36, 48],
    },
    "champion": {
        "time-scale": [0.8],
        "tip-mass": [0.05],
        "bend-stiffness": [0.002],
        "stretch-damping": [0.5],
    },
    "earlylift": {
        "time-scale": [0.8],
        "tip-mass": [0.05],
        "bend-stiffness": [0.002],
        "stretch-damping": [0.5],
        "t-flight": [0.6, 0.9, 1.3],
    },
    "highknot": {
        "time-scale": [0.75],
        "tip-mass": [0.06, 0.07],
        "bend-stiffness": [0.0005],
        "stretch-damping": [0.5],
    },
    "highknot2": {
        "time-scale": [0.75],
        "tip-mass": [0.04, 0.05],
        "bend-stiffness": [0.0005],
        "stretch-damping": [0.5],
    },
    "knotpos": {
        "time-scale": [0.8],
        "tip-mass": [0.05],
        "bend-stiffness": [0.002, 0.003],
        "stretch-damping": [0.5],
        "friction": [1.0, 1.5, 2.0],
    },
    "basin": {
        "time-scale": [0.78, 0.8, 0.82],
        "tip-mass": [0.045, 0.05, 0.055],
        "bend-stiffness": [0.0015, 0.002, 0.003],
        "stretch-damping": [0.5],
    },
}


def analyze(npz_path: Path) -> dict:
    d = np.load(npz_path)
    traj = d["rope_traj"]
    n = len(traj)
    stable = bool(np.isfinite(traj).all())
    # Per-frame writhe (sampled) to catch transient knots that slip off.
    sample = range(0, n, 4)
    writhes = np.array([polyline_writhe(traj[f]) for f in sample]) if stable else np.zeros(1)
    final_writhe = polyline_writhe(traj[-1]) if stable else float("nan")
    final_cross = max(count_crossings(traj[-1], axis=0), count_crossings(traj[-1], axis=1)) if stable else -1
    e2e = np.linalg.norm(traj[-1][-1] - traj[-1][0])
    arc = np.linalg.norm(np.diff(traj[-1], axis=0), axis=1).sum()
    idx_max = int(np.argmax(np.abs(writhes)))
    knot_pos = knot_pos_fraction(traj[-1]) if stable else None
    return {
        "final_knot_pos": knot_pos,
        "stable": stable,
        "final_writhe": float(final_writhe),
        "final_crossings": int(final_cross),
        "final_length_ratio": float(e2e / arc),
        "max_abs_writhe": float(np.abs(writhes).max()),
        "max_writhe_time": float(list(sample)[idx_max] / 60.0),
        "knotted": bool(abs(final_writhe) > 2.0 and e2e / arc < 0.95),
    }


def run_one(params: dict, tag: str, outdir: Path, extra_args: list[str], use_arm: bool = False) -> dict:
    npz = outdir / f"{tag}.npz"
    cmd = [
        "uv",
        "run",
        "-m",
        "newton.examples",
        "cable_flying_knot",
        "--viewer",
        "null",
        "--test",
        "--num-frames",
        "600",
        "--save-traj",
        str(npz),
        *([] if use_arm else ["--no-arm"]),
        *extra_args,
    ]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900, check=False)
    elapsed = time.time() - t0
    rec = {"params": params, "tag": tag, "elapsed_s": round(elapsed, 1)}
    if proc.returncode != 0 or not npz.exists():
        rec["error"] = proc.stderr[-500:]
        return rec
    rec.update(analyze(npz))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="broad")
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--outdir", default="/tmp/fk/sweep")
    ap.add_argument("extra", nargs="*", help="extra args passed to the example")
    args = ap.parse_args()

    grid = GRIDS[args.grid]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else outdir / f"results_{args.grid}.json"

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    jobs = []
    for i, combo in enumerate(combos):
        for r in range(args.repeats):
            jobs.append((dict(zip(keys, combo, strict=True)), f"{args.grid}_{i:03d}_r{r}"))
    print(f"grid '{args.grid}': {len(combos)} configs x {args.repeats} repeats = {len(jobs)} runs")

    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, params, tag, outdir, args.extra, args.arm) for params, tag in jobs]
        for i, fut in enumerate(futures):
            rec = fut.result()
            results.append(rec)
            status = "ERR" if "error" in rec else ("KNOT" if rec.get("knotted") else "----")
            print(
                f"[{i + 1}/{len(jobs)}] {status} {rec['tag']} {rec['params']} "
                f"wr_final={rec.get('final_writhe', float('nan')):+.2f} "
                f"wr_max={rec.get('max_abs_writhe', float('nan')):.2f}@{rec.get('max_writhe_time', 0):.1f}s "
                f"ratio={rec.get('final_length_ratio', float('nan')):.3f} "
                f"knotpos={rec.get('final_knot_pos') if rec.get('final_knot_pos') is None else format(rec['final_knot_pos'], '.2f')} "
                f"({rec['elapsed_s']}s)",
                flush=True,
            )
            with open(out_json, "w") as f:
                json.dump(results, f, indent=1)

    # Success-rate summary per config.
    by_cfg = {}
    for rec in results:
        key = json.dumps(rec["params"], sort_keys=True)
        by_cfg.setdefault(key, []).append(rec.get("knotted", False))
    print("\nsuccess rates:")
    for key, oks in sorted(by_cfg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {sum(oks)}/{len(oks)}  {key}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
