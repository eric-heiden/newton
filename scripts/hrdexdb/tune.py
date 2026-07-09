# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CMA-ES tuning of per-hand simulation parameters.

One shared parameter set per hand, optimized in log-space over a small set of
training episodes and then FROZEN — evaluation on held-out episodes happens in
``evaluate.py``. The objective is the mean object ADD RMSE across training
episodes; the object stays fully passive throughout (no cheating).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from batch import BatchReplayer
from dataset import load_episode
from scene import SimParams

RESULTS = Path(__file__).parent / "results"

# (name, lower, upper) — log10-space search
PARAM_SPACE = [
    ("arm_ke", 500.0, 20000.0),
    ("arm_kd", 10.0, 500.0),
    ("hand_ke", 5.0, 500.0),
    ("hand_kd", 0.1, 20.0),
    ("friction", 0.2, 2.5),
    ("object_mass", 0.03, 1.0),
    ("joint_armature", 1e-3, 0.1),
]

# Training episodes per hand (objects chosen for diversity of size/shape;
# everything else is held out).
# blue_plastic_box excluded from training: its large flat contact patches
# make episodes ~10x more expensive to simulate, dominating tuning wall time
# (it remains in the held-out evaluation).
TRAIN_EPISODES = {
    "allegro_v5": [("banana", None), ("apple", None), ("baseball", None), ("book", None), ("beige_brush", None)],
    # banana/book episodes are incomplete for inspire_f1 on the remote;
    # donut and coffee_tin substitute as diverse small/cylindrical objects.
    "inspire_f1": [("apple", None), ("baseball", None), ("beige_brush", None), ("donut", None), ("coffee_tin", None)],
}


def params_from_log(x: np.ndarray) -> SimParams:
    kwargs = {}
    for (name, lo, hi), xi in zip(PARAM_SPACE, x):
        val = 10.0 ** float(np.clip(xi, np.log10(lo), np.log10(hi)))
        kwargs[name] = val
    return SimParams(**kwargs)


def log_from_params(p: SimParams) -> np.ndarray:
    return np.array([np.log10(np.clip(getattr(p, name), lo, hi)) for name, lo, hi in PARAM_SPACE])


def objective_terms(metrics: dict) -> float:
    return metrics["add_rmse"]


def resolve_train_episodes(hand: str, manifest: dict) -> list[tuple[str, str]]:
    out = []
    for obj, scene in TRAIN_EPISODES[hand]:
        scenes = manifest.get(hand, {}).get(obj, [])
        if not scenes:
            continue
        out.append((obj, scene or scenes[0]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, choices=["allegro_v5", "inspire_f1"])
    parser.add_argument("--target-source", default="cmd", choices=["cmd", "meas"])
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=0.25)
    parser.add_argument("--budget-hours", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=4)
    args = parser.parse_args()

    import cma

    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
    episodes = resolve_train_episodes(args.hand, manifest)
    print(f"training episodes: {episodes}")

    replayers = []
    for obj, scene in episodes:
        ep = load_episode(args.hand, obj, scene)
        replayers.append(
            BatchReplayer(ep, num_worlds=args.popsize, target_source=args.target_source, substeps=args.substeps)
        )

    x0 = log_from_params(SimParams())
    bounds = [[np.log10(lo) for _, lo, _ in PARAM_SPACE], [np.log10(hi) for _, _, hi in PARAM_SPACE]]
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, {"popsize": args.popsize, "bounds": bounds, "seed": 42})

    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / f"tune_{args.hand}_{args.target_source}.csv"
    best_path = RESULTS / f"tuned_params_{args.hand}_{args.target_source}.json"
    log_f = open(log_path, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(
        ["gen", "cand", *[n for n, _, _ in PARAM_SPACE], *[f"add_{o}_{s}" for o, s in episodes], "mean_add"]
    )

    t_start = time.time()
    gen = 0
    best = (np.inf, None)
    while not es.stop() and gen < args.maxiter and (time.time() - t_start) < args.budget_hours * 3600:
        xs = es.ask()
        plist = [params_from_log(np.asarray(x)) for x in xs]
        per_ep = []
        for (obj, scene), rep in zip(episodes, replayers):
            t_ep = time.time()
            res = rep.run(plist)
            per_ep.append([objective_terms(r) for r in res])
            print(f"  gen {gen} {obj}/{scene}: {time.time() - t_ep:.0f}s", flush=True)
        per_ep = np.array(per_ep)  # (episodes, popsize)
        # Rare contact-solver blowups produce NaN states; penalize rather
        # than poisoning the CMA-ES ranking.
        per_ep = np.where(np.isnan(per_ep), 1.0, per_ep)
        costs = per_ep.mean(axis=0)
        es.tell(xs, costs.tolist())
        for c, (x, cost) in enumerate(zip(xs, costs)):
            p = params_from_log(np.asarray(x))
            writer.writerow(
                [
                    gen,
                    c,
                    *[f"{getattr(p, n):.5g}" for n, _, _ in PARAM_SPACE],
                    *[f"{v:.5f}" for v in per_ep[:, c]],
                    f"{cost:.5f}",
                ]
            )
        log_f.flush()
        i_best = int(np.argmin(costs))
        if costs[i_best] < best[0]:
            best = (float(costs[i_best]), params_from_log(np.asarray(xs[i_best])))
            best_path.write_text(json.dumps({**best[1].__dict__, "_mean_add_rmse": best[0], "_gen": gen}, indent=1))
        elapsed = time.time() - t_start
        print(
            f"gen {gen}: best {costs[i_best]:.4f} (all-time {best[0]:.4f}) "
            f"mean {costs.mean():.4f} sigma {es.sigma:.3f} [{elapsed / 60:.1f} min]"
        )
        gen += 1

    log_f.close()
    print(f"done: best mean add_rmse {best[0]:.4f}")
    print(json.dumps(best[1].__dict__, indent=1))


if __name__ == "__main__":
    main()
