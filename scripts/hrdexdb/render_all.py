# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the representative overlay videos for the report.

Picks best/median/worst episodes per hand (by ADD RMSE from the tuned
evaluation), renders each with the tuned parameters, and writes
``videos/videos.json`` metadata for ``make_report.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import load_episode  # noqa: E402
from render import render_episode  # noqa: E402
from scene import SimParams  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
VIDEOS = HERE / "videos"


def tuned_params(hand: str) -> SimParams:
    p = RESULTS / f"tuned_params_{hand}_cmd.json"
    if p.exists():
        raw = {k: v for k, v in json.loads(p.read_text()).items() if not k.startswith("_")}
        return SimParams(**raw)
    return SimParams()


def pick_episodes(hand: str, tag: str) -> dict[str, tuple[str, str, dict]]:
    summary_path = RESULTS / tag / f"{hand}_cmd" / "summary.json"
    if not summary_path.exists():
        return {}
    s = {
        k: v
        for k, v in json.loads(summary_path.read_text()).items()
        if "error" not in v and not v.get("calib_outlier")
    }
    ranked = sorted(s.items(), key=lambda kv: kv[1]["add_rmse"])
    if not ranked:
        return {}
    return {
        "best": ranked[0],
        "median": ranked[len(ranked) // 2],
        "worst": ranked[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="tuned")
    parser.add_argument("--camera", default="side")
    args = parser.parse_args()

    VIDEOS.mkdir(exist_ok=True)
    meta_path = VIDEOS / "videos.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
    have = {m["file"] for m in meta}

    for hand, hand_label in [("allegro_v5", "Allegro V5"), ("inspire_f1", "Inspire F1")]:
        params = tuned_params(hand)
        for label, (key, metrics) in pick_episodes(hand, args.tag).items():
            obj, scene = key.split("/")
            fname = f"{hand}_{obj}_{scene}_{args.tag}.mp4"
            if fname not in have:
                ep = load_episode(hand, obj, scene)
                render_episode(ep, params, target_source="cmd", output=VIDEOS / fname, camera=args.camera)
                have.add(fname)
            title = f"{hand_label}: {obj.replace('_', ' ')} ({label})"
            caption = (
                f"ADD RMSE {metrics['add_rmse'] * 100:.1f} cm, "
                f"lift {'matched' if metrics['lift_match'] else 'mismatched'} "
                f"(sim {'lifted' if metrics['sim_lifted'] else 'missed'}, "
                f"real {'lifted' if metrics['gt_lifted'] else 'no lift'}). "
                "Green ghost = ground-truth object pose."
            )
            meta = [m for m in meta if m["file"] != fname]
            meta.append({"file": fname, "title": title, "caption": caption, "hand": hand, "kind": label})
            meta_path.write_text(json.dumps(meta, indent=1))
    print(f"{len(meta)} videos in {VIDEOS}")


if __name__ == "__main__":
    main()
