# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Black-box throw-command search for the MuJoCo-coupled flying knot.

Plays the role of the paper's task-level ILC: the digitized human command
does not knot under the coupled (free-pivot) rope dynamics, so we search
perturbations of the Bezier control points plus timing and tip mass with a
cross-entropy loop. The objective is shaped for *threading*, not just
coiling: it rewards writhe that persists after the throw (a coil that
collapses scores low), a locked final knot, and penalizes commands the
dynamic arm cannot track.

Usage:
  uv run python scripts/flying_knot/command_search.py [--generations 12] [--pop 16]
      [--resume PATH/history.json] [--seed 0]
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

# Perturbation applies to control points 1..7 (P0 anchors the start pose).
CTRL_SLICE = slice(1, 8)
N_CTRL = 7
# Layout: 21 ctrl deltas + time-scale + tip-mass + yank (dx,dy,dz,delay,T).
PARAM_DIM = N_CTRL * 3 + 2 + 5

T_SETTLE = 1.5
T_THROW_BASE = 0.7
FPS = 60

# Shortened tail phases for search rollouts (retention is still exercised).
PHASE_ARGS = ["--t-flight", "1.6", "--t-lift", "2.0", "--t-hold", "0.8"]
# Additional rollout arguments (set from --rollout-args).
EXTRA_ROLLOUT_ARGS: list[str] = []


def unpack(theta):
    delta = np.zeros((8, 3))
    delta[CTRL_SLICE] = theta[: N_CTRL * 3].reshape(N_CTRL, 3)
    time_scale = float(np.clip(theta[N_CTRL * 3], 0.62, 1.15))
    tip_mass = float(np.clip(theta[N_CTRL * 3 + 1], 0.03, 0.10))
    yank = theta[N_CTRL * 3 + 2 :].copy()
    yank[:3] = np.clip(yank[:3], -0.45, 0.45)
    yank[3] = float(np.clip(yank[3], 0.02, 0.5))  # delay after throw end
    yank[4] = float(np.clip(yank[4], 0.08, 0.35))  # yank duration
    return delta, time_scale, tip_mass, yank


N_ROBUST_EVALS = 1
BASIN_PROBES = 0
BASIN_SIGMA = 0.008
_BASIN_RNG = np.random.default_rng(1234)


def evaluate(theta, tag, outdir):
    """Evaluate a candidate; optionally score its basin membership.

    With BASIN_PROBES > 0, knotting candidates are additionally evaluated at
    small parameter perturbations. Candidates that only knot on a razor-thin
    manifold (whose neighbors miss) receive no basin bonus, steering the
    search toward basin interiors that reproduce across environments.
    """
    recs = [_evaluate_once(theta, f"{tag}_e{k}" if N_ROBUST_EVALS > 1 else tag, outdir) for k in range(N_ROBUST_EVALS)]
    rec = min(recs, key=lambda r: r["score"])
    rec = dict(rec)
    rec["tag"] = tag
    rec["knotted"] = all(r.get("knotted", False) for r in recs)
    if rec["knotted"] and BASIN_PROBES > 0:
        hits = 0
        for k in range(BASIN_PROBES):
            pert = np.array(theta) + BASIN_SIGMA * _BASIN_RNG.standard_normal(len(theta))
            probe = _evaluate_once(pert, f"{tag}_b{k}", outdir)
            hits += int(probe.get("knotted", False))
        rec["basin"] = hits / BASIN_PROBES
        rec["score"] = round(rec["score"] + 6.0 * rec["basin"], 3)
    return rec


def _evaluate_once(theta, tag, outdir):
    delta, time_scale, tip_mass, yank = unpack(theta)
    delta_file = outdir / f"{tag}_delta.npz"
    npz = outdir / f"{tag}.npz"
    np.savez(delta_file, delta=delta)
    duration = T_SETTLE + T_THROW_BASE * time_scale + 1.6 + 2.0 + 0.8
    num_frames = int(round(duration * FPS))
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
        str(num_frames),
        "--time-scale",
        str(time_scale),
        "--tip-mass",
        str(tip_mass),
        "--bezier-delta-file",
        str(delta_file),
        "--yank-dx",
        str(yank[0]),
        "--yank-dy",
        str(yank[1]),
        "--yank-dz",
        str(yank[2]),
        "--yank-delay",
        str(yank[3]),
        "--yank-t",
        str(yank[4]),
        "--save-traj",
        str(npz),
        "--ik-max-step",
        "0.035",
        "--ik-iters",
        "8",
        *PHASE_ARGS,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200, check=False)
    rec = {"tag": tag, "theta": [float(v) for v in theta], "score": -10.0}
    if proc.returncode != 0 or not npz.exists():
        rec["error"] = proc.stderr[-300:]
        return rec
    d = np.load(npz)
    traj = d["rope_traj"]
    if not np.isfinite(traj).all():
        rec["error"] = "non-finite"
        return rec

    throw_end = T_SETTLE + T_THROW_BASE * time_scale
    f0 = int(T_SETTLE * FPS)
    f1 = min(int((throw_end + 2.0) * FPS), len(traj))
    sample = range(f0, f1, 3)
    writhes = np.array([polyline_writhe(traj[f]) for f in sample])
    wr_max = float(np.abs(writhes).max()) if len(writhes) else 0.0

    # Threading persistence: |writhe| held in the second after the throw.
    p0 = int((throw_end + 0.2) * FPS)
    p1 = min(int((throw_end + 1.2) * FPS), len(traj))
    persist_samples = [polyline_writhe(traj[f]) for f in range(p0, p1, 3)]
    persistence = float(np.mean(np.abs(persist_samples))) if persist_samples else 0.0

    final_writhe = polyline_writhe(traj[-1])
    e2e = np.linalg.norm(traj[-1][-1] - traj[-1][0])
    arc = np.linalg.norm(np.diff(traj[-1], axis=0), axis=1).sum()
    ratio = float(e2e / arc)
    knotted = abs(final_writhe) > 2.0 and ratio < 0.95
    crossings = max(count_crossings(traj[-1], axis=0), count_crossings(traj[-1], axis=1))

    # Dynamic-feasibility penalty: commands the arm cannot track are invalid.
    jq = d["joint_q_traj"]
    qref = d["q_ref_frames"]
    n = min(len(jq), len(qref) - 1)
    track_err = float(np.abs(jq[:n] - qref[1 : n + 1]).max()) if n else 0.0

    score = wr_max + 2.5 * persistence + (6.0 + abs(final_writhe)) * float(knotted)
    score -= 4.0 * max(0.0, track_err - 0.15)
    rec.update(
        {
            "score": round(float(score), 3),
            "wr_max": round(wr_max, 3),
            "persistence": round(persistence, 3),
            "final_writhe": round(float(final_writhe), 3),
            "ratio": round(ratio, 3),
            "crossings": int(crossings),
            "knotted": bool(knotted),
            "track_err": round(track_err, 3),
            "time_scale": round(time_scale, 4),
            "tip_mass": round(tip_mass, 4),
            "elapsed": round(time.time() - t0, 1),
        }
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="/tmp/fk/cmd_search2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None, help="history.json to seed the incumbent from")
    ap.add_argument("--seed-theta", type=str, default=None, help="JSON list of theta values for the incumbent")
    ap.add_argument(
        "--rollout-args", type=str, default="", help="Extra args appended to every rollout, e.g. '--rope-segments 72'"
    )
    ap.add_argument("--robust-evals", type=int, default=1, help="Rollouts per candidate; score is the minimum.")
    ap.add_argument(
        "--basin-probes",
        type=int,
        default=0,
        help="Perturbed rollouts scoring basin membership of knotting candidates.",
    )
    args = ap.parse_args()
    if args.rollout_args:
        EXTRA_ROLLOUT_ARGS.extend(args.rollout_args.split())
    global N_ROBUST_EVALS, BASIN_PROBES  # noqa: PLW0603
    N_ROBUST_EVALS = max(1, args.robust_evals)
    BASIN_PROBES = max(0, args.basin_probes)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mean = np.zeros(PARAM_DIM)
    mean[N_CTRL * 3] = 0.8  # time-scale
    mean[N_CTRL * 3 + 1] = 0.055  # tip-mass
    mean[N_CTRL * 3 + 2 :] = [0.0, 0.0, 0.15, 0.2, 0.15]  # yank
    std = np.concatenate([np.full(N_CTRL * 3, 0.12), [0.10, 0.015], [0.15, 0.10, 0.15, 0.12, 0.08]])
    std_floor = np.concatenate([np.full(N_CTRL * 3, 0.03), [0.02, 0.004], [0.03, 0.03, 0.03, 0.03, 0.02]])

    best = None
    if args.seed_theta:
        seeded = np.array(json.loads(args.seed_theta), dtype=float)
        if len(seeded) == 23:
            seeded = np.concatenate([seeded, [0.0, 0.0, 0.0, 0.2, 0.15]])
        best = {"theta": [float(v) for v in seeded], "score": -1e9}
        mean = seeded.copy()
        print("seeded incumbent from --seed-theta")
    if args.resume:
        prev = json.load(open(args.resume))
        prev_best = prev.get("best")
        if prev_best and "theta" in prev_best:
            old = np.array(prev_best["theta"])
            seeded = np.zeros(PARAM_DIM)
            # Old layout: ctrl pts 2..4 (9 values) + ts + tip.
            if len(old) == 11:
                seeded[3 : 3 + 9] = old[:9]  # P2..P4 occupy slots 1..3 of CTRL_SLICE
                seeded[N_CTRL * 3 : N_CTRL * 3 + 2] = old[-2:]
                seeded[N_CTRL * 3 + 2 :] = [0.0, 0.0, 0.0, 0.2, 0.15]
            elif len(old) == 23:
                seeded[:23] = old
                seeded[N_CTRL * 3 + 2 :] = [0.0, 0.0, 0.0, 0.2, 0.15]
            elif len(old) == PARAM_DIM:
                seeded = old
            best = {"theta": [float(v) for v in seeded], "score": -1e9}
            mean = seeded.copy()
            print(f"resumed incumbent from {args.resume}")

    history = []
    for gen in range(args.generations):
        thetas = [mean + std * rng.standard_normal(PARAM_DIM) for _ in range(args.pop)]
        if best is not None:
            thetas[0] = np.array(best["theta"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            recs = list(pool.map(lambda p, g=gen: evaluate(p[1], f"g{g}_i{p[0]}", outdir), enumerate(thetas)))
        for r in recs:
            print(
                f"  {r['tag']}: score={r['score']} wr_max={r.get('wr_max')} "
                f"persist={r.get('persistence')} final={r.get('final_writhe')} "
                f"ratio={r.get('ratio')} knot={r.get('knotted')} basin={r.get('basin')} track={r.get('track_err')} "
                f"ts={r.get('time_scale')} tip={r.get('tip_mass')}",
                flush=True,
            )
        recs.sort(key=lambda r: -r["score"])
        history.extend(recs)
        if best is None or recs[0]["score"] > best.get("score", -1e9):
            cand = recs[0]
            if cand.get("knotted"):
                # Winners must reproduce in a solo re-run: concurrent rollouts
                # are subject to contact nondeterminism, and near-manifold
                # flukes must not become the stored incumbent.
                verify = evaluate(np.array(cand["theta"]), f"{cand['tag']}_verify", outdir)
                print(f"  verify {cand['tag']}: score={verify['score']} knot={verify.get('knotted')}", flush=True)
                if verify.get("knotted"):
                    best = verify
                elif best is None:
                    best = cand
            else:
                best = cand
        elites = [np.array(r["theta"]) for r in recs[: args.elite]]
        mean = np.mean(elites, axis=0)
        std = np.maximum(0.8 * std + 0.2 * np.std(elites, axis=0), std_floor)
        print(
            f"gen {gen}: best {recs[0]['score']} (all-time {best['score']}) knotted={recs[0].get('knotted')}",
            flush=True,
        )
        with open(outdir / "history.json", "w") as f:
            json.dump({"history": history, "best": best}, f, indent=1)
    print("all-time best:", json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
