# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recompute palm_obj_min_dist / calib_outlier in existing summaries.

The Inspire URDF spells "palm" as "plam", so the original runs measured the
wrist instead of the palm and over-flagged calibration outliers. This re-runs
only the FK-based sanity check (no simulation) and patches summary.json files.
Episodes with NaN metrics are dropped so the resumable evaluator re-runs them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from dataset import load_episode
from replay import Replayer

RESULTS = Path(__file__).parent / "results"


def main():
    cache: dict[str, float] = {}
    for summary_path in sorted(RESULTS.glob("*/*/summary.json")):
        hand = summary_path.parent.name.rsplit("_", 1)[0]
        summary = json.loads(summary_path.read_text())
        changed = False
        for key in list(summary):
            v = summary[key]
            if "error" in v:
                del summary[key]
                changed = True
                continue
            if math.isnan(v.get("add_rmse", float("nan"))):
                del summary[key]
                (summary_path.parent / f"{key.replace('/', '_')}.npz").unlink(missing_ok=True)
                changed = True
                continue
            ck = f"{hand}/{key}"
            if ck not in cache:
                obj, scene = key.split("/")
                try:
                    ep = load_episode(hand, obj, scene)
                    rep = Replayer.__new__(Replayer)  # skip solver construction
                    from scene import build_scene

                    rep.ep = ep
                    rep.info = build_scene(ep)
                    rep.model = rep.info.model
                    cache[ck] = rep.palm_object_min_dist()
                except Exception as e:
                    print(f"{ck}: sanity recompute failed ({e})")
                    cache[ck] = v.get("palm_obj_min_dist", 0.0)
            dist = cache[ck]
            outlier = dist > 0.25
            if v.get("palm_obj_min_dist") != dist or v.get("calib_outlier") != outlier:
                v["palm_obj_min_dist"] = dist
                v["calib_outlier"] = outlier
                changed = True
        if changed:
            summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
            n_out = sum(1 for v in summary.values() if v.get("calib_outlier"))
            print(f"{summary_path.parent.name}: {len(summary)} entries, {n_out} outliers")


if __name__ == "__main__":
    main()
