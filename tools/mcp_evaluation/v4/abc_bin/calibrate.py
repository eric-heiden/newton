# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the ``abc_bin`` verifier gates from fresh-process runs of the reference and the starter.

Runs ``verify.py --single`` (one fresh process each) on the reference solution and on the starter, and sets
every gate with abc_replay's rule (build plan section 8):

- an upper bound on a continuous metric is ``clip(1.25 * reference median + 3 * sigma_rerun, floor,
  0.7 * starter median)`` (the cap applies only where it lies above the floor, i.e. where the starter fails
  the metric at all; the held-out arm bound is not capped, see :data:`UNCAPPED`);
- a lower bound on a fraction applies the same rule to its complement (no such gate at present);
- an event count is the reference's lowest count over the runs minus one ensemble member: one copy for the
  main-episode counts, one copy of one episode for the held-out rate.

Rival configurations (``--rival NAME=PATH``) are run the same way and reported against the new thresholds.

Usage::

    python -m tools.mcp_evaluation.v4.abc_bin.calibrate --reference REF.py --starter WORKSPACE/screwdriver_replay.py \\
        --out DIR [--runs 9] [--starter-runs 3] [--rival NAME=PATH ...] [--write]

Every script runs from its own directory, as the verifier sees a submission: a workspace with the agent dataset
(the verifier scans every Python file beside the script, so not this directory).

``--write`` stores the thresholds and the calibration table in ``~/.newton-visual-private/abc_bin/
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
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "abc_bin"

# Upper bounds: threshold key -> (gate metric, floor, resolution). Floors are the measurement resolution: in-hand
# rotation 6 deg (the build plan's value; over the main carry window the real handle collar moves 6.1 px in the
# right wrist view, and the simulated in-hand pivot about the pinch axis moves it 1.3 px/deg, so the real grasp
# turns at most about 4.7 deg), slip 5 mm, finger gap 5 mm, one 10 fps frame (3 rows) for lift-off, track and
# final tip 2.5 cm (ground-truth accuracy plus identical-run spread), movement before the grasp 2 cm, arm tracking
# 0.02 rad. Thresholds are rounded up to the resolution.
UPPER = {
    "inhand_rot_deg_max": ("main_inhand_rot_deg_max", 6.0, 0.1),
    "slip_m_max": ("main_slip_m_max", 0.005, 1e-4),
    "grip_gap_err_mm_max": ("main_grip_gap_err_mm_max", 5.0, 0.1),
    "liftoff_err_rows_max": ("main_liftoff_err_rows_max", 3.0, 1.0),
    "carry_track_err_m_max": ("main_carry_track_err_m_max", 0.025, 1e-4),
    "final_tip_xy_err_m_max": ("main_final_tip_xy_err_m_max", 0.025, 1e-3),
    "moved_before_grasp_m_max": ("main_moved_before_grasp_m_max", 0.02, 1e-4),
    "arm_rmse_rad_max": ("main_arm_rmse_rad_max", 0.02, 1e-4),
    "heldout_arm_rmse_rad_max": ("heldout_arm_rmse_rad_max", 0.02, 1e-4),
}
# Upper bounds without the starter cap. Held-out arm error runs 0.9 to 1.2 times the main episode's for honest
# arm calibrations, so the capped value would fail calibrations that pass every main-episode gate; the uncapped
# rule value still fails the starter.
UNCAPPED = {"heldout_arm_rmse_rad_max"}
# Lower bounds on fractions, calibrated on 1 - value: threshold key -> (gate metric, floor of 1 - value).
FRACTION = {}
# Event counts: threshold key -> (gate metric, member size).
COUNTS = {"main_held_min": ("main_held_min", 1), "main_placed_min": ("main_placed_min", 1)}
# Held-out rate: one member is one copy of one episode.
HELDOUT_MEMBER = 1
HELDOUT_RATE_MAX = 0.9
# Not calibrated: the negative controls (cheat detectors), the held-out copy rule, and the report-only settings.
FIXED = {
    "control_rise_m_max": 0.01,
    "control_low_mu_held_max": 0,
    "heldout_rot_max_deg": 12.0,  # replay_common.HELDOUT_ROT_MAX_DEG
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
    counts = values(reference, "heldout_count")
    total = {int(r["metrics"]["heldout_total"]) for r in reference}
    if len(total) != 1:
        raise RuntimeError(f"held-out totals differ between runs: {total}")
    total = total.pop()
    rate = (int(counts.min()) - HELDOUT_MEMBER) / total
    # The reference holds and places every held-out copy, so "minus one copy" leaves honest variants (zero command
    # delay, a 20 N grip) at the edge of run-to-run noise; the rate is capped at HELDOUT_RATE_MAX, which still
    # rejects the rival grips (fruit-style fix 0.667, 10 N soft 0.75).
    thresholds["heldout_rate_min"] = min(math.floor(rate * 1000) / 1000, HELDOUT_RATE_MAX)
    table["heldout_rate_min"] = {
        "metric": "heldout_count / heldout_total",
        "total": total,
        "reference_counts": sorted(int(c) for c in counts),
        "reference_min_rate": float(counts.min() / total),
        "starter_rates": sorted(float(r["metrics"]["heldout_rate"]) for r in starter),
        "member": HELDOUT_MEMBER,
        "threshold": thresholds["heldout_rate_min"],
    }
    return thresholds, table


def failures(metrics: dict, t: dict) -> list[str]:
    """Gates of :func:`verify.evaluate` that a run's metrics fail under thresholds ``t``."""
    m = {key: _number(value) for key, value in metrics.items()}

    def get(key: str) -> float:
        return m.get(key, math.inf)

    tests = {
        "main_held_min": m.get("main_held_min", -math.inf) >= t["main_held_min"],
        "main_placed_min": m.get("main_placed_min", -math.inf) >= t["main_placed_min"],
        "main_inhand_rot_deg_max": get("main_inhand_rot_deg_max") <= t["inhand_rot_deg_max"],
        "main_slip_m_max": get("main_slip_m_max") <= t["slip_m_max"],
        "main_grip_gap_err_mm_max": get("main_grip_gap_err_mm_max") <= t["grip_gap_err_mm_max"],
        "main_liftoff_err_rows_max": get("main_liftoff_err_rows_max") <= t["liftoff_err_rows_max"],
        "main_carry_track_err_m_max": get("main_carry_track_err_m_max") <= t["carry_track_err_m_max"],
        "main_final_tip_xy_err_m_max": get("main_final_tip_xy_err_m_max") <= t["final_tip_xy_err_m_max"],
        "main_moved_before_grasp_m_max": get("main_moved_before_grasp_m_max") <= t["moved_before_grasp_m_max"],
        "main_arm_rmse_rad_max": get("main_arm_rmse_rad_max") <= t["arm_rmse_rad_max"],
        "heldout_rate": m.get("heldout_rate", -math.inf) >= t["heldout_rate_min"],
        "heldout_arm_rmse_rad_max": get("heldout_arm_rmse_rad_max") <= t["heldout_arm_rmse_rad_max"],
        "control_grip_open_rise_m": get("control_grip_open_rise_m") <= t["control_rise_m_max"],
        "control_low_mu_held": get("control_low_mu_held") <= t["control_low_mu_held_max"],
    }
    return [name for name, ok in tests.items() if not ok]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, default=PRIVATE / "reference" / "screwdriver_replay.py")
    # The verifier scans every Python file beside the script, so the starter runs from a workspace copy (the
    # agent dataset with screwdriver_replay.py), not from this directory.
    parser.add_argument("--starter", type=Path, required=True, help="screwdriver_replay.py in a workspace copy")
    parser.add_argument("--out", type=Path, required=True, help="directory for the run results")
    parser.add_argument("--runs", type=int, default=9, help="fresh-process runs of the reference")
    parser.add_argument("--starter-runs", type=int, default=3)
    parser.add_argument("--rival", action="append", default=[], help="NAME=PATH of a rival config (reported)")
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
    for item in args.rival:
        name, path = item.split("=", 1)
        result = run(Path(path).resolve(), args.out / f"rival_{name}.json")
        checks[f"rival_{name}"] = {
            "failures": failures(result["metrics"], thresholds),
            "metrics": {k: v for k, v in result["metrics"].items() if k.startswith(("main_", "heldout_", "control_"))},
        }
    table["_check"] = checks
    print(json.dumps(checks))
    if any(checks["reference_failures"]) or not all(checks["starter_failures"]):
        print("REJECTED: the reference must pass and the starter fail every run")
    (args.out / "calibration.json").write_text(json.dumps({"thresholds": thresholds, "table": table}, indent=1) + "\n")
    if args.write:
        record = {
            "_note": "abc_bin verifier gates, written by tools/mcp_evaluation/v4/abc_bin/calibrate.py "
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
