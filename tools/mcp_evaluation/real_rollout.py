# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run one measured-robot candidate or host it in the Newton live application."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

from .real_robot import RealRobotScenario
from .rollout import camera, make_session


def main() -> None:
    """Use numeric JSON parameters for both agent candidates and final verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--connection-file", type=Path, default=Path("connection.json"))
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / "process_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "simulation_process_start",
                    "pid": os.getpid(),
                    "wall_time_unix": time.time(),
                    "live": args.live,
                }
            )
            + "\n"
        )
    scenario = RealRobotScenario(
        json.loads(args.config.read_text()), reference_file=args.reference, variant=args.variant
    )
    (args.output.parent / "provenance.json").write_text(json.dumps(scenario.provenance, indent=2) + "\n")
    if args.live:
        from newton.mcp import SimulationServer  # noqa: PLC0415

        session = make_session(scenario, args.output.parent / "observations")
        scenario.rollout_log = args.output.parent / "live_rollouts.jsonl"
        server = SimulationServer(session, connection_file=args.connection_file)
        server.start()
        (args.output.parent / "server_ready.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "build_seconds": scenario.build_seconds,
                    "startup_seconds": time.perf_counter() - started,
                }
            )
            + "\n"
        )
        try:
            session.run()
        finally:
            server.close()
        return
    result = scenario.rollout()
    scenario.save_trace(args.output.with_suffix(".npz"))
    result.update(
        {
            "trace_path": scenario.last_trace_path,
            "pid": os.getpid(),
            "process_seconds_after_import": time.perf_counter() - started,
        }
    )
    if args.observe:
        session = make_session(scenario, args.output.parent / "observations")
        observation = session.dispatch("observe", {**camera("panda_real"), "width": 640, "height": 480})
        image_path = args.output.with_suffix(".png")
        image_path.write_bytes(base64.b64decode(observation.pop("image_base64")))
        result["observation"] = observation | {"image_path": str(image_path)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with (args.output.parent / "rollouts.jsonl").open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
