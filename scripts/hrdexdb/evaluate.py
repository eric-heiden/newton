# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Broad evaluation of HRDexDB dynamic replay across many episodes.

Runs one frozen parameter set over all locally available episodes of a hand
and stores per-episode trajectories + metrics (resumable). Also used for the
pilot study comparing PD target sources (recorded commands vs measured
joint positions).
"""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import numpy as np

from dataset import load_episode  # noqa: E402
from replay import Replayer  # noqa: E402
from scene import SimParams  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def run_episode(hand, obj, scene, params, target_source, substeps, out_path: Path):
    ep = load_episode(hand, obj, scene)
    rep = Replayer(ep, params, target_source=target_source, substeps=substeps)
    res = rep.run()
    res.metrics.update(hand=hand, object=obj, scene=scene, target_source=target_source)
    res.save(out_path)
    return res.metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, choices=["allegro_v5", "inspire_f1"])
    parser.add_argument("--target-source", default="cmd", choices=["cmd", "meas"])
    parser.add_argument("--params", type=str, default=None, help="JSON file with SimParams (default: untuned)")
    parser.add_argument("--tag", default="default", help="results subdirectory tag")
    parser.add_argument("--objects", nargs="*", default=None, help="restrict to these objects")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=4)
    args = parser.parse_args()

    if args.params:
        raw = {k: v for k, v in json.loads(Path(args.params).read_text()).items() if not k.startswith("_")}
        params = SimParams(**raw)
    else:
        params = SimParams()

    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
    episodes = []
    for obj, scenes in sorted(manifest.get(args.hand, {}).items()):
        if args.objects and obj not in args.objects:
            continue
        episodes.extend((args.hand, obj, s) for s in scenes)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    out_dir = RESULTS / args.tag / f"{args.hand}_{args.target_source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    print(f"{len(episodes)} episodes -> {out_dir}")
    for i, (hand, obj, scene) in enumerate(episodes):
        key = f"{obj}/{scene}"
        npz = out_dir / f"{obj}_{scene}.npz"
        if key in summary and npz.exists():
            continue
        try:
            metrics = run_episode(hand, obj, scene, params, args.target_source, args.substeps, npz)
            summary[key] = metrics
            print(
                f"[{i + 1}/{len(episodes)}] {key}: add {metrics['add_rmse']:.4f} pos {metrics['pos_rmse']:.4f} "
                f"lift {metrics['sim_lifted']}/{metrics['gt_lifted']} ({metrics['wall_time']:.0f}s)"
            )
        except Exception as e:
            summary[key] = {"error": str(e)}
            print(f"[{i + 1}/{len(episodes)}] {key}: FAILED {e}")
            traceback.print_exc()
        summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
        gc.collect()

    ok = [m for m in summary.values() if "error" not in m]
    if ok:
        add = np.array([m["add_rmse"] for m in ok])
        pos = np.array([m["pos_rmse"] for m in ok])
        lift = np.mean([m["lift_match"] for m in ok])
        print(f"\n=== {args.hand}/{args.target_source}/{args.tag}: {len(ok)} episodes ===")
        print(f"add_rmse mean {add.mean():.4f} median {np.median(add):.4f}")
        print(f"pos_rmse mean {pos.mean():.4f} median {np.median(pos):.4f}")
        print(f"lift match rate {lift:.2%}")


if __name__ == "__main__":
    main()
