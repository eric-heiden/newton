# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Revalidate tuned parameters against the training episodes.

CMA-ES on a noisy objective suffers from winner's curse: the reported best
score is optimistically biased. This re-evaluates the frozen tuned parameters
(and the defaults) on the training episodes in fresh single-world rollouts and
stores both numbers for the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dataset import load_episode
from replay import Replayer
from scene import SimParams
from tune import TRAIN_EPISODES, resolve_train_episodes

RESULTS = Path(__file__).parent / "results"


def eval_params(hand: str, params: SimParams, episodes) -> list[float]:
    adds = []
    for obj, scene in episodes:
        ep = load_episode(hand, obj, scene)
        m = Replayer(ep, params).run().metrics
        adds.append(m["add_rmse"])
        print(f"  {obj}/{scene}: {m['add_rmse']:.4f}", flush=True)
    return adds


def main():
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
    out_path = RESULTS / "revalidation.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for hand in TRAIN_EPISODES:
        tuned_file = RESULTS / f"tuned_params_{hand}_cmd.json"
        if not tuned_file.exists() or hand in out:
            continue
        episodes = resolve_train_episodes(hand, manifest)
        raw = {k: v for k, v in json.loads(tuned_file.read_text()).items() if not k.startswith("_")}
        print(f"{hand} tuned:")
        tuned_adds = eval_params(hand, SimParams(**raw), episodes)
        print(f"{hand} default:")
        default_adds = eval_params(hand, SimParams(), episodes)
        out[hand] = {
            "episodes": [f"{o}/{s}" for o, s in episodes],
            "tuned_train_adds": tuned_adds,
            "default_train_adds": default_adds,
            "tuned_train_mean": float(np.nanmean(tuned_adds)),
            "default_train_mean": float(np.nanmean(default_adds)),
            "cma_reported_best": json.loads(tuned_file.read_text()).get("_mean_add_rmse"),
        }
        out_path.write_text(json.dumps(out, indent=1))
        print(json.dumps(out[hand], indent=1))


if __name__ == "__main__":
    main()
