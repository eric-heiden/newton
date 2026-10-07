# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Re-run the current verifier on finished trials' final workspaces.

Verifier fixes made during an iteration should apply to every trial of a task alike. This restores each run's
final workspace where its trial ran, runs the task's current verifier in the same sandbox as the trial's own
verification, on the Newton the trial ran on (:func:`run_v4.newton_source`), and writes the result to
``RUN_DIR/reverify/result.json`` and the verifier's records next to it (the trial's original
``verification.json`` is kept). ``run_v4 --verify-snapshots`` takes this result as the final snapshot's verdict.
Never run it while trials are running: verifiers time submissions.

Usage::

    python -m tools.mcp_evaluation.v4.reverify RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from tools.mcp_evaluation.v4 import run_v4


def reverify(run_dir: Path) -> dict:
    spec = json.loads((run_dir / "spec.json").read_text())
    task = run_v4._task(spec["task"])
    sandbox_root = run_v4.TRIALS / spec["trial_id"]
    if sandbox_root.exists():
        raise RuntimeError(f"{sandbox_root} exists: the trial is still running or was not cleaned up")
    records = run_dir / "reverify"
    records.mkdir(exist_ok=True)
    try:
        shutil.copytree((run_dir / "workspace").resolve(), sandbox_root / "work", symlinks=True)
        caches = run_v4._trial_caches(run_dir, spec)
        if caches.is_dir():
            shutil.copytree(caches, sandbox_root / "caches", symlinks=True)
        env = run_v4.rerun_env(spec, sandbox_root, "reverify")
        start = time.time()
        result = run_v4.verify(sandbox_root / "work", records, task, env, run_v4._container(sandbox_root, run_dir))
        result["seconds"] = time.time() - start
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=run_v4.ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        result["newton_commit"] = spec.get("commit")
        (records / "result.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
        return result
    finally:
        shutil.rmtree(sandbox_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        original = (
            json.loads((run_dir / "verification.json").read_text()) if (run_dir / "verification.json").exists() else {}
        )
        result = reverify(run_dir.resolve())
        print(
            f"{run_dir.name}: original {original.get('success')} {original.get('failed_checks')} -> "
            f"{result.get('success')} {result.get('failed_checks')} ({result['seconds']:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
