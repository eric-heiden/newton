# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Repeatability study: run identical episodes N times, measure metric spread.

Grasping is contact-chaotic; atomics in the contact solver make rollouts
nondeterministic. This quantifies the run-to-run spread of ADD RMSE, which
bounds how much of any parameter-tuning improvement is signal vs noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import load_episode  # noqa: E402
from replay import Replayer  # noqa: E402
from scene import SimParams  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=["allegro_v5/banana/2", "allegro_v5/baseball/0", "inspire_f1/apple/1", "inspire_f1/beige_brush/2"],
    )
    args = parser.parse_args()

    out = {}
    for spec in args.episodes:
        hand, obj, scene = spec.split("/")
        params_file = RESULTS / f"tuned_params_{hand}_cmd.json"
        if params_file.exists():
            raw = {k: v for k, v in json.loads(params_file.read_text()).items() if not k.startswith("_")}
            params = SimParams(**raw)
        else:
            params = SimParams()
        ep = load_episode(hand, obj, scene)
        adds, lifts = [], []
        for r in range(args.runs):
            rep = Replayer(ep, params)
            m = rep.run().metrics
            adds.append(m["add_rmse"])
            lifts.append(m["sim_lifted"])
            print(f"{spec} run {r}: add {m['add_rmse']:.4f} lifted {m['sim_lifted']}", flush=True)
        out[spec] = {"add_rmse": adds, "sim_lifted": lifts}
        (RESULTS / "repeatability.json").write_text(json.dumps(out, indent=1))
    for spec, v in out.items():
        a = np.array(v["add_rmse"])
        a = a[~np.isnan(a)]
        print(f"{spec}: add {a.mean():.4f} ± {a.std():.4f} (range {a.min():.4f}–{a.max():.4f}), lift rate {np.mean(v['sim_lifted']):.2f}")


if __name__ == "__main__":
    main()
