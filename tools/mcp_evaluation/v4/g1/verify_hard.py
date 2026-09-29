# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a G1 controller on the hard reference clip (high5) as well as the wave clips.

Same protocol and integrity checks as ``verify.py``; the high5 clip (turning and
shifting the root) is gated instead of reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import verify as base

base.CASES = {
    "wave": ("wave.csv", 1.0, True),
    "high5": ("high5.csv", 1.0, True),
    "high5_slow": ("high5.csv", 0.9, True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = base.load(args.script.resolve())
    results = {name: base.run_case(module, base.HERE / clip, speed) for name, (clip, speed, _) in base.CASES.items()}
    checks = {f"{name}.{k}": v for name, r in results.items() for k, v in r["checks"].items()}
    integrity = all(checks.values())
    worst = {key: max(r[key] if r["upright"] else float("inf") for r in results.values()) for key in base.THRESHOLDS}
    success = integrity and all(worst[k] <= v for k, v in base.THRESHOLDS.items())
    result = {
        "task": "g1_hard",
        "success": bool(success),
        "integrity": integrity,
        "failed_checks": [k for k, v in checks.items() if not v],
        "metrics": {**worst, **{f"{n}_upright": r["upright"] for n, r in results.items()}},
        "normalized_worst": {k: worst[k] / v for k, v in base.THRESHOLDS.items()},
        "details": results,
    }
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({"success": success, "metrics": result["metrics"], "failed_checks": result["failed_checks"]}))


if __name__ == "__main__":
    main()
