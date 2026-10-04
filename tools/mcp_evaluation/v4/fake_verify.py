# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Stand-in task verifier for the harness tests (test_run_v4): passes when the submission contains ``PASS = True``.

It is called like the task verifiers (``verify.py SCRIPT --output JSON``) and writes the same result keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    passed = "PASS = True" in args.script.read_text()
    result = {
        "success": passed,
        "integrity": True,
        "failed_checks": [] if passed else ["pass"],
        "metrics": {"files": sorted(path.name for path in args.script.parent.iterdir())},
        "normalized_worst": 0.5 if passed else 2.0,
    }
    args.output.write_text(json.dumps(result) + "\n")
    print("verified", passed)


if __name__ == "__main__":
    main()
