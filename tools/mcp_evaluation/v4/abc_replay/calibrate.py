# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the ``abc_replay`` verifier gates from fresh-process runs of the reference and the starter.

Runs ``verify.py --single`` (one fresh process each) on the reference solution and on the starter, and sets
every gate with the task rule (``fruits/task_spec_draft.json``):

- an upper bound on a continuous metric is ``clip(1.25 * reference median + 3 * sigma_rerun, floor,
  0.7 * starter median)`` (the cap applies only where it lies above the floor, i.e. where the starter fails
  the metric at all);
- a lower bound on a fraction (``lifted_fraction``) applies the same rule to its complement;
- an event count is the reference's lowest count over the runs minus one ensemble member: one copy for
  the per-fruit main-episode counts, one copy of an episode's three fruits for the pooled held-out rate.

Usage::

    python -m tools.mcp_evaluation.v4.abc_replay.calibrate --reference REF.py --starter STARTER.py \\
        --out DIR [--runs 8] [--starter-runs 3] [--write]

``--write`` stores the thresholds and the calibration table in ``~/.newton-visual-private/abc_replay/
thresholds.json``, which :mod:`verify` reads. Existing run files in ``--out`` are reused.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "abc_replay"

# Upper bounds: threshold key -> (gate metric, floor, resolution). Floors are the measurement resolution: track
# and final position 2.5 cm (ground-truth accuracy plus identical-run spread), one 10 fps frame (3 rows) and
# 0.1 s for timing, 5 mm for the finger gap (nominal fruit sizes vs the real ones), 0.02 rad for arm tracking.
# Thresholds are rounded up to the resolution.
UPPER = {
    "carry_track_err_m_max": ("main_carry_track_err_m_max", 0.025, 1e-4),
    "liftoff_err_rows_max": ("main_liftoff_err_rows_max", 3.0, 1.0),
    "release_err_s_max": ("main_release_err_s_max", 0.10, 0.01),
    "moved_before_grasp_m_max": ("main_moved_before_grasp_m_max", 0.02, 1e-4),
    "arm_rmse_rad_max": ("main_arm_rmse_rad_max", 0.02, 1e-4),
    "grip_gap_err_mm_max": ("main_grip_gap_err_mm_max", 5.0, 0.1),
    "final_xy_err_m_max": ("main_final_xy_err_m_max", 0.025, 1e-3),
    "heldout_arm_rmse_rad_max": ("heldout_arm_rmse_rad_max", 0.02, 1e-4),
}
# Upper bounds without the starter cap. Held-out arm error runs 0.9 to 1.2 times the main episode's for honest
# arm calibrations, so the capped value (within 3 % of the reference) would fail calibrations that pass every
# main-episode gate; the uncapped rule value still fails the starter.
UNCAPPED = {"heldout_arm_rmse_rad_max"}
# Lower bounds on fractions, calibrated on 1 - value: threshold key -> (gate metric, floor of 1 - value).
FRACTION = {"lifted_fraction_min": ("main_lifted_fraction_min", 0.10)}
# Event counts: threshold key -> (gate metric, member size).
COUNTS = {"main_held_min": ("main_held_min", 1), "main_placed_min": ("main_placed_min", 1)}
# Not calibrated: the release lower bound (the reference releases late, never early), the negative controls
# (cheat detectors), and the report-only settings.
FIXED = {
    "release_err_s_min": -0.10,
    "control_rise_m_max": 0.01,
    "heldout_copies_min": 3,
    "heldout_exclude_verdicts": ["unreliable"],
}


def run(script: Path, output: Path) -> dict:
    """One ``verify.py --single`` run of ``script`` in a fresh process (reused if ``output`` exists)."""
    if not output.exists():
        command = [sys.executable, str(HERE / "verify.py"), str(script), "--single", "--output", str(output)]
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        tic = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
        output.with_suffix(".log").write_text(process.stdout + process.stderr)
        print(f"{output.name}: {time.perf_counter() - tic:.0f} s", flush=True)
    result = json.loads(output.read_text())
    if result.get("integrity") and not all(result["integrity"].values()):
        raise RuntimeError(f"{output}: integrity failed {result['failed_checks']}")
    return result


def _number(value) -> float:
    return float(value) if isinstance(value, (int, float)) else math.inf


def calibrate(reference: list[dict], starter: list[dict]) -> tuple[dict, dict]:
    """Thresholds and the calibration table from the reference and starter runs."""
    thresholds, table = dict(FIXED), {}

    def values(runs: list[dict], metric: str) -> np.ndarray:
        return np.array([_number(r["metrics"].get(metric)) for r in runs])

    for key, (metric, floor, resolution) in UPPER.items():
        ref, start = values(reference, metric), values(starter, metric)
        median, sigma, starter_median = float(np.median(ref)), float(np.std(ref, ddof=1)), float(np.median(start))
        raw = 1.25 * median + 3.0 * sigma
        cap = 0.7 * starter_median
        value = max(raw, floor)
        if cap > floor and key not in UNCAPPED:
            value = min(value, cap)
        value = round(math.ceil(value / resolution - 1e-9) * resolution, 6)
        thresholds[key] = value
        table[key] = {
            "metric": metric,
            "reference_median": median,
            "reference_max": float(ref.max()),
            "sigma_rerun": sigma,
            "starter_median": starter_median,
            "raw": raw,
            "floor": floor,
            "cap": cap if cap > floor and key not in UNCAPPED else None,
            "threshold": value,
        }
    for key, (metric, floor) in FRACTION.items():
        ref, start = 1.0 - values(reference, metric), 1.0 - values(starter, metric)
        median, sigma, starter_median = float(np.median(ref)), float(np.std(ref, ddof=1)), float(np.median(start))
        raw = 1.25 * median + 3.0 * sigma
        cap = 0.7 * starter_median
        miss = max(raw, floor)
        if cap > floor:
            miss = min(miss, cap)
        thresholds[key] = 1.0 - miss
        table[key] = {
            "metric": f"1 - {metric}",
            "reference_median": median,
            "sigma_rerun": sigma,
            "starter_median": starter_median,
            "raw": raw,
            "floor": floor,
            "cap": cap if cap > floor else None,
            "threshold": 1.0 - miss,
        }
    for key, (metric, member) in COUNTS.items():
        ref, start = values(reference, metric), values(starter, metric)
        thresholds[key] = int(ref.min()) - member
        table[key] = {
            "metric": metric,
            "reference_min": int(ref.min()),
            "reference_median": float(np.median(ref)),
            "starter_max": float(start.max()),
            "member": member,
            "threshold": thresholds[key],
        }
    # Held-out: one member is one copy of an episode, i.e. its three fruits.
    counts = values(reference, "heldout_fruit_count")
    total = {int(r["metrics"]["heldout_fruit_total"]) for r in reference}
    if len(total) != 1:
        raise RuntimeError(f"held-out totals differ between runs: {total}")
    total = total.pop()
    rate = (int(counts.min()) - 3) / total
    thresholds["heldout_fruit_rate_min"] = math.floor(rate * 1000) / 1000
    table["heldout_fruit_rate_min"] = {
        "metric": "heldout_fruit_count / heldout_fruit_total",
        "total": total,
        "reference_counts": sorted(int(c) for c in counts),
        "reference_min_rate": float(counts.min() / total),
        "starter_rates": sorted(float(r["metrics"]["heldout_fruit_rate"]) for r in starter),
        "member": 3,
        "threshold": thresholds["heldout_fruit_rate_min"],
    }
    return thresholds, table


def failures(metrics: dict, t: dict) -> list[str]:
    """Gates of :func:`verify.evaluate` that a run's metrics fail under thresholds ``t``."""
    m = {key: _number(value) for key, value in metrics.items()}
    tests = {
        "main_held_min": m["main_held_min"] >= t["main_held_min"],
        "main_placed_min": m["main_placed_min"] >= t["main_placed_min"],
        "main_lifted_fraction_min": m["main_lifted_fraction_min"] >= t["lifted_fraction_min"],
        "main_carry_track_err_m_max": m["main_carry_track_err_m_max"] <= t["carry_track_err_m_max"],
        "main_liftoff_err_rows_max": m["main_liftoff_err_rows_max"] <= t["liftoff_err_rows_max"],
        "main_release_err_s": t["release_err_s_min"] <= m["main_release_err_s_min"]
        and m["main_release_err_s_max"] <= t["release_err_s_max"],
        "main_moved_before_grasp_m_max": m["main_moved_before_grasp_m_max"] <= t["moved_before_grasp_m_max"],
        "main_arm_rmse_rad_max": m["main_arm_rmse_rad_max"] <= t["arm_rmse_rad_max"],
        "main_grip_gap_err_mm_max": m["main_grip_gap_err_mm_max"] <= t["grip_gap_err_mm_max"],
        "main_final_xy_err_m_max": m["main_final_xy_err_m_max"] <= t["final_xy_err_m_max"],
        "heldout_fruit_rate": m["heldout_fruit_rate"] >= t["heldout_fruit_rate_min"],
        "heldout_arm_rmse_rad_max": m["heldout_arm_rmse_rad_max"] <= t["heldout_arm_rmse_rad_max"],
        "control_grip_open_rise_m": m["control_grip_open_rise_m"] <= t["control_rise_m_max"],
        "control_low_mu_rise_m": m["control_low_mu_rise_m"] <= t["control_rise_m_max"],
    }
    return [name for name, ok in tests.items() if not ok]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, default=PRIVATE / "reference" / "fruit_replay.py")
    parser.add_argument("--starter", type=Path, default=HERE / "fruit_replay.py")
    parser.add_argument("--out", type=Path, required=True, help="directory for the run results")
    parser.add_argument("--runs", type=int, default=8, help="fresh-process runs of the reference")
    parser.add_argument("--starter-runs", type=int, default=3)
    parser.add_argument("--write", action="store_true", help="write PRIVATE/thresholds.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = [run(args.reference.resolve(), args.out / f"reference_{k}.json") for k in range(args.runs)]
    starter = [run(args.starter.resolve(), args.out / f"starter_{k}.json") for k in range(args.starter_runs)]
    thresholds, table = calibrate(reference, starter)
    print(json.dumps(table, indent=1))
    print(json.dumps(thresholds, indent=1))
    checks = {
        "reference_failures": [failures(r["metrics"], thresholds) for r in reference],
        "starter_failures": [failures(r["metrics"], thresholds) for r in starter],
    }
    table["_check"] = checks
    print(json.dumps(checks))
    if any(checks["reference_failures"]) or not all(checks["starter_failures"]):
        print("REJECTED: the reference must pass and the starter fail every run")
    (args.out / "calibration.json").write_text(json.dumps({"thresholds": thresholds, "table": table}, indent=1) + "\n")
    if args.write:
        record = {
            "_note": "abc_replay verifier gates, written by tools/mcp_evaluation/v4/abc_replay/calibrate.py "
            "(rule in its docstring) from fresh-process verify.py --single runs; verify.py reads these keys.",
            "_calibration": {
                "date": time.strftime("%Y-%m-%d", time.gmtime()),
                "reference": str(args.reference),
                "starter": str(args.starter),
                "runs": {"reference": args.runs, "starter": args.starter_runs},
                "results": str(args.out),
                "table": table,
            },
            **thresholds,
        }
        (PRIVATE / "thresholds.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {PRIVATE / 'thresholds.json'}")


if __name__ == "__main__":
    main()
