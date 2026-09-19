# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure repeated live edits versus process restarts, separately from LLM trials."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from newton.mcp import SimulationClient

from .scenarios import ROOT, SPECS, initial_config


def _quality_differences(first: dict, second: dict, thresholds: dict) -> list[str]:
    """Compare every scored metric and the candidate identity and completion state."""
    differences = [
        key
        for key in ("scenario", "variant", "config", "frames", "sample_count", "expected_frames", "finite", "success")
        if key not in first or key not in second or first[key] != second[key]
    ]
    for key in (*thresholds, "simulation_time_s"):
        a, b = first.get(key), second.get(key)
        if (
            not isinstance(a, int | float)
            or not isinstance(b, int | float)
            or not np.isfinite(a)
            or not np.isfinite(b)
            or not np.isclose(a, b, atol=1e-6, rtol=1e-5)
        ):
            differences.append(key)
    return differences


def main() -> None:
    """Time matched scripted experiments with shared on-disk compilation caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SPECS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cfg = initial_config(args.scenario)
    config_path = args.output / "config.py"
    config_path.write_text("CONFIG = " + repr(cfg))
    command = [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(ROOT),
        "python",
        "-m",
        "tools.mcp_evaluation.rollout",
        "--scenario",
        args.scenario,
        "--config",
        str(config_path),
        "--output",
        str(args.output / "metrics.json"),
    ]
    warmup_start = time.perf_counter()
    warmup = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    warmup_seconds = time.perf_counter() - warmup_start
    (args.output / "warmup.log").write_text(warmup.stdout + warmup.stderr)
    connection = args.output / "connection.json"
    records = []
    with (args.output / "server.log").open("w") as log:
        startup_start = time.perf_counter()
        server = subprocess.Popen(
            [*command, "--live", "--connection-file", str(connection)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while not (args.output / "server_ready.json").exists():
                if server.poll() is not None or time.perf_counter() - startup_start > 120:
                    raise RuntimeError("Live server failed to start")
                time.sleep(0.05)
            startup_seconds = time.perf_counter() - startup_start
            client = SimulationClient(connection)
            for index in range(args.repetitions):
                config = dict(cfg)
                key = "wrist_kp" if args.scenario == "hug" else "kp"
                config[key] *= 1 + index * 0.1
                config_path.write_text("CONFIG = " + repr(config))
                before = time.perf_counter()
                baseline = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
                restart_seconds = time.perf_counter() - before
                (args.output / f"restart-{index}.log").write_text(baseline.stdout + baseline.stderr)
                baseline_result = json.loads((args.output / "metrics.json").read_text())
                code = f"session.scenario.apply_config({config!r})\nsession.dispatch('reset')\nsession.dispatch('step', {{'count': 1500}})\nresult = session.scenario.metrics()"
                before = time.perf_counter()
                live_response = client.request("execute", code=code)
                live_seconds = time.perf_counter() - before
                live_result = live_response["result"]
                differences = _quality_differences(baseline_result, live_result, SPECS[args.scenario]["thresholds"])
                records.append(
                    {
                        "index": index,
                        "config": config,
                        "restart_seconds": restart_seconds,
                        "live_seconds": live_seconds,
                        "quality_equal": not differences,
                        "quality_differences": differences,
                        "restart": baseline_result,
                        "live": live_result,
                    }
                )
                (args.output / "measurements.json").write_text(json.dumps(records, indent=2))
        finally:
            server.terminate()
            server.wait(timeout=15)
    result = {
        "type": "scripted systems microbenchmark, not agent performance",
        "scenario": args.scenario,
        "warmup_seconds": warmup_seconds,
        "live_startup_seconds": startup_seconds,
        "records": records,
        "all_quality_equal": all(x["quality_equal"] for x in records),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_quality_equal": result["all_quality_equal"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
