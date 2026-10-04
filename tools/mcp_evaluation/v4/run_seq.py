# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run an iteration of paired trials one at a time, with R replicates per task x model x condition.

Every trial runs alone: no concurrent trials, so no shared CPU or GPU load and no visible partner processes.
Replicates run in rounds (each round runs every pair once), so an iteration cut short keeps equal counts per
cell. In round ``r`` the pair at position ``i`` runs ``first`` first when ``i + r`` is even and the other
condition first otherwise, so each task x model alternates its order from round to round.

The plan is a JSON object read from a file or from stdin (``-``, the default), which keeps it off this
process's command line; other processes on the machine, trial agents included, can read command lines::

    {
        "directory": "/path/to/loop/i16",
        "pairs": ["abc_scratch:opus", "g1_mpc:opus", "abc_scratch:astra", "g1_mpc:astra"],
        "replicates": 3,
        "first": "mcp",
        "start": 0,
        "tag": "p",
        "harness_version": "h16",
        "verify_snapshots": false,
    }

``start`` is the first replicate index, ``tag`` the prefix of the run-directory suffix
(``DIRECTORY/TASK-MODEL-CONDITION-pR``), and ``verify_snapshots`` runs ``run_v4 --verify-snapshots`` on every
run directory after the last trial (``"all"`` verifies every snapshot version). Runs that already have a
``summary.json`` are skipped, and a run directory left by an interrupted run is renamed ``*.aborted-N`` and run
again, so a stopped iteration resumes where it stopped. Progress lines go to ``DIRECTORY/launch.log``.

Usage::

    python -m tools.mcp_evaluation.v4.run_seq < plan.json
    python -m tools.mcp_evaluation.v4.run_seq --dry-run plan.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv/bin/python"
CONDITIONS = ("mcp", "restart")


def schedule(pairs: list[str], replicates: int, first: str = "mcp", start: int = 0, tag: str = "p") -> list[dict]:
    """Trials in run order: one dict per trial with ``task``, ``model``, ``condition``, ``replicate``, ``name``."""
    if first not in CONDITIONS:
        raise ValueError(f"first must be one of {CONDITIONS}")
    other = CONDITIONS[1 - CONDITIONS.index(first)]
    trials = []
    for replicate in range(start, start + replicates):
        for index, pair in enumerate(pairs):
            task, _, model = pair.partition(":")
            if not task or not model:
                raise ValueError(f"pair {pair!r} is not TASK:MODEL")
            order = (first, other) if (index + replicate) % 2 == 0 else (other, first)
            for condition in order:
                name = f"{task}-{model}-{condition}-{tag}{replicate}"
                trials.append(
                    {"task": task, "model": model, "condition": condition, "replicate": replicate, "name": name}
                )
    return trials


def _log(directory: Path, line: str) -> None:
    with (directory / "launch.log").open("a") as log:
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")


def run(plan: dict, dry_run: bool = False) -> list[Path]:
    """Run the plan's trials one after another; returns their run directories."""
    directory = Path(plan["directory"]).resolve()
    trials = schedule(
        plan["pairs"],
        int(plan.get("replicates", 1)),
        plan.get("first", "mcp"),
        int(plan.get("start", 0)),
        plan.get("tag", "p"),
    )
    if dry_run:
        for trial in trials:
            print(trial["name"])
        return [directory / trial["name"] for trial in trials]
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        NEWTON_HARNESS_VERSION=str(plan.get("harness_version") or os.environ.get("NEWTON_HARNESS_VERSION", "dev")),
    )
    run_dirs = []
    for trial in trials:
        run_dir = directory / trial["name"]
        run_dirs.append(run_dir)
        if (run_dir / "summary.json").exists():
            _log(directory, f"SKIP {trial['name']} (finished)")
            continue
        if run_dir.exists():
            attempt = 1
            while run_dir.with_name(f"{run_dir.name}.aborted-{attempt}").exists():
                attempt += 1
            run_dir.rename(run_dir.with_name(f"{run_dir.name}.aborted-{attempt}"))
        # The runner reads its options from a file in the (hidden) iteration directory, not its command line.
        spec = directory / f".spec-{uuid.uuid4().hex}.json"
        spec.write_text(
            json.dumps(
                {
                    "task": trial["task"],
                    "condition": trial["condition"],
                    "model": trial["model"],
                    "workspace": str(run_dir),
                    "run": True,
                }
            )
            + "\n"
        )
        _log(directory, f"START {trial['name']}")
        try:
            with (directory / f"{trial['name']}.log").open("w") as out:
                code = subprocess.run(
                    [str(PYTHON), "-m", "tools.mcp_evaluation.v4.run_v4", "--spec", str(spec)],
                    cwd=ROOT,
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
        finally:
            spec.unlink(missing_ok=True)
        _log(directory, f"END {trial['name']} rc={code}")
    verify = plan.get("verify_snapshots")
    if verify:
        # After the last trial only: snapshot verification uses the GPU and must not overlap a trial.
        _log(directory, "VERIFY_SNAPSHOTS")
        command = [str(PYTHON), "-m", "tools.mcp_evaluation.v4.run_v4", "--verify-snapshots", *map(str, run_dirs)]
        with (directory / "verify_snapshots.log").open("w") as out:
            code = subprocess.run(
                [*command, *(["--all"] if verify == "all" else [])],
                cwd=ROOT,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        _log(directory, f"VERIFY_SNAPSHOTS_DONE rc={code}")
    _log(directory, "DONE")
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plan", nargs="?", default="-", help="plan JSON file, or - for stdin (default)")
    parser.add_argument("--dry-run", action="store_true", help="print the run order and exit")
    args = parser.parse_args()
    plan = json.loads(sys.stdin.read() if args.plan == "-" else Path(args.plan).read_text())
    run(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
