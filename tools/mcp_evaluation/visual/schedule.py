# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Register and run the confirmation schedule: paired MCP/restart trials started together.

``register`` writes the seeded order, source commit, and hashes of the task
sources and reference photos before any scored agent starts. ``run`` executes
the registered pairs, ``parallel`` pairs at a time, appending to a ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASKS = ("cloth_drape", "push", "arm_offsets")
MODELS = ("opus", "astra")
CONDITIONS = ("mcp", "restart")


def _hash_tree(paths) -> dict:
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def register(output: Path, private: Path, replicates: int, seed: int, seconds: int) -> dict:
    blocks = [(task, model, r) for task in TASKS for model in MODELS for r in range(replicates)]
    random.Random(seed).shuffle(blocks)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "newton", "tools/mcp_evaluation/visual"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        raise RuntimeError("Commit Newton and task sources before registration:\n" + dirty)
    registration = {
        "created_unix": time.time(),
        "commit": commit,
        "seed": seed,
        "seconds": seconds,
        "conditions": list(CONDITIONS),
        "pairs": [{"index": i, "task": t, "model": m, "replicate": r} for i, (t, m, r) in enumerate(blocks)],
        "source_sha256": _hash_tree([*(ROOT / "newton/_src/mcp").glob("*.py"), *(ROOT / "tools/mcp_evaluation/visual").glob("*.py")]),
        "reference_sha256": {
            f"{task}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
            for task in TASKS
            for p in sorted((private / "reference" / task).iterdir())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registration, indent=2) + "\n")
    return registration


def _run_pair(pair: dict, directory: Path, seconds: int, ledger: Path) -> None:
    processes = []
    for condition in CONDITIONS:
        name = f"{pair['task']}-{pair['model']}-{condition}-{pair['replicate']}"
        workspace = directory / name
        if workspace.exists():
            continue
        log = (directory / f"{name}.log").open("w")
        processes.append(
            (
                name,
                subprocess.Popen(
                    [
                        str(ROOT / ".venv/bin/python"),
                        "-m",
                        "tools.mcp_evaluation.visual.run_visual",
                        "--task",
                        pair["task"],
                        "--condition",
                        condition,
                        "--model",
                        pair["model"],
                        "--workspace",
                        str(workspace),
                        "--seconds",
                        str(seconds),
                        "--phase",
                        "confirmation",
                        "--run",
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
            )
        )
    for name, process in processes:
        process.wait()
        summary = directory / name / "summary.json"
        digest = hashlib.sha256(summary.read_bytes()).hexdigest() if summary.exists() else None
        with ledger.open("a") as stream:
            stream.write(json.dumps({"pair": pair["index"], "trial": name, "finished_unix": time.time(), "exit": process.returncode, "summary_sha256": digest}) + "\n")


def run(registration_path: Path, directory: Path, parallel: int) -> None:
    registration = json.loads(registration_path.read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if commit != registration["commit"]:
        raise RuntimeError(f"Source commit {commit} differs from registered {registration['commit']}")
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "ledger.jsonl"
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for pair in registration["pairs"]:
            pool.submit(_run_pair, pair, directory, registration["seconds"], ledger)
            # Stagger starts so concurrent pairs do not start their agents at the same instant.
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--output", type=Path, required=True)
    reg.add_argument("--private", type=Path, required=True)
    reg.add_argument("--replicates", type=int, default=3)
    reg.add_argument("--seed", type=int, default=20260929)
    reg.add_argument("--seconds", type=int, default=1200)
    go = sub.add_parser("run")
    go.add_argument("--registration", type=Path, required=True)
    go.add_argument("--directory", type=Path, required=True)
    go.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(register(args.output, args.private, args.replicates, args.seed, args.seconds)["pairs"][:3]))
    else:
        run(args.registration, args.directory, args.parallel)


if __name__ == "__main__":
    sys.exit(main())
