# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Score every parameter set an agent simulated against the held-out truth, after the trial.

This measures search progress independently of when the agent decided to stop:
the elapsed time at which the agent first simulated a candidate that would pass
held-out verification. Candidates are identified from the shared task log, so
the same accounting covers live-MCP and restart trials.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .tasks import THRESHOLDS, TRUTH_TIMES, compare, measure, task_class


def candidates(workspace: Path) -> list[dict]:
    """Distinct parameter sets in first-use order, with seconds since the trial started."""
    spec = json.loads((workspace / "summary.json").read_text())
    start = spec["started_unix"] + spec["application_startup_seconds"]
    seen, result = set(), []
    path = workspace / "candidates.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        record = json.loads(line)
        if record.get("event") not in ("params", "simulated"):
            continue
        key = json.dumps(record["params"], sort_keys=True)
        # Only parameter sets that were actually simulated or rendered count as candidates.
        if key in seen:
            continue
        seen.add(key)
        result.append({"seconds": max(0.0, record["wall_time_unix"] - start), "params": record["params"]})
    return result


def score(workspace: Path, private: Path, limit: int = 40) -> dict:
    spec = json.loads((workspace / "summary.json").read_text())
    name = spec["task"]
    cls = task_class(name)
    with np.load(private / f"{name}_truth.npz") as data:
        truth = {key: data[key] for key in data.files}
    rows = candidates(workspace)
    if len(rows) > limit:
        # Keep the first, the last, and evenly spaced candidates in between.
        index = np.unique(np.linspace(0, len(rows) - 1, limit).round().astype(int))
        rows = [rows[i] for i in index]
    task = None
    for row in rows:
        if task is None:
            task = cls(row["params"])
        else:
            task.set_params(row["params"])
        measured = measure(task, cls.HELDOUT_EPISODES, TRUTH_TIMES[name])
        worst = compare(name, measured, truth, cls.HELDOUT_EPISODES)["worst"]
        row["normalized"] = {key: worst[key] / limit_value for key, limit_value in THRESHOLDS[name].items()}
        row["passes"] = all(value <= 1.0 for value in row["normalized"].values())
    first = next((row["seconds"] for row in rows if row["passes"]), None)
    result = {"task": name, "candidates": rows, "first_passing_seconds": first, "scored": len(rows)}
    (workspace / "anytime.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspaces", type=Path, nargs="+")
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    # Post-hoc scoring must not add entries to the trial's own candidate log.
    os.environ["NEWTON_VISUAL_LOG"] = os.devnull
    for workspace in args.workspaces:
        result = score(workspace, args.private, args.limit)
        print(json.dumps({"workspace": str(workspace), "first_passing_seconds": result["first_passing_seconds"], "scored": result["scored"]}))


if __name__ == "__main__":
    main()
