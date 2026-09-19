# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run a fresh one-shot baseline or host the identical scenario over live MCP."""

from __future__ import annotations

import argparse
import base64
import json
import os
import runpy
import time
from pathlib import Path

import numpy as np

from .scenarios import SPECS, Scenario


def make_scenario(name: str, config: dict, *, variant: int = 0, reference_file: Path | None = None):
    """Construct an original task or the separately specified calibration task."""
    if name == "panda_calibration":
        from .calibration import CalibrationScenario  # noqa: PLC0415

        if reference_file is None:
            raise ValueError("panda_calibration requires a supplied reference NPZ")
        return CalibrationScenario(config, reference_file=reference_file, variant=variant)
    if reference_file is not None:
        raise ValueError("A reference NPZ is only used for panda_calibration")
    return Scenario(name, config, variant=variant)


def make_session(scenario: Scenario, directory: Path):
    """Bind the same application-owned simulation to the public live API."""
    from newton.mcp import SimulationSession  # noqa: PLC0415

    def bindings(current):
        return {
            "model": current.model,
            "solver": current.solver,
            "state": current.state,
            "state_next": current.state_next,
            "control": current.control,
            "collision_pipeline": current.pipeline,
            "contacts": current.contacts,
        }

    def rebuild(session, *, config=None):
        previous = session.scenario
        values = dict(previous.config)
        if config is not None:
            values.update(config)
        replacement = make_scenario(
            scenario.name, values, variant=scenario.variant, reference_file=getattr(previous, "reference_file", None)
        )
        if hasattr(previous, "rollout_log"):
            replacement.rollout_log = previous.rollout_log
        session.scenario = replacement
        session.step_callback = replacement.session_step
        session.reset_callback = replacement.session_reset
        return bindings(replacement)

    session = SimulationSession(
        **bindings(scenario),
        dt=scenario.dt,
        step_callback=scenario.session_step,
        reset_callback=scenario.session_reset,
        rebuild_callback=rebuild,
        allow_execute=True,
        artifact_directory=directory,
    )
    session.scenario = scenario
    session.frame = scenario.frame
    session.time = scenario.frame * scenario.dt
    return session


def camera(name: str) -> dict:
    """Return a prespecified camera shared by both conditions."""
    if name in ("panda", "panda_calibration"):
        return {"eye": [1.2, -1.2, 0.85], "target": [0, 0, 0.45], "up": [0, 0, 1]}
    if name == "allegro":
        return {"eye": [0.4, -0.45, 0.53], "target": [0, 0, 0.32], "up": [0, 0, 1]}
    return {"eye": [0.45, -0.4, 0.38], "target": [0, 0, 0.07], "up": [0, 0, 1]}


def main() -> None:
    """Execute one reproducible experiment and record its process lifecycle."""
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SPECS, required=True)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--connection-file", type=Path, default=Path("connection.json"))
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.scenario == "panda_calibration":
        if args.variant < 0:
            parser.error("calibration variant must be nonnegative")
    elif args.variant not in (0, 1):
        parser.error("variant must be 0 or 1 for this scenario")
    config = runpy.run_path(str(args.config))["CONFIG"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / "process_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "event": "simulation_process_start",
                    "wall_time_unix": time.time(),
                    "live": args.live,
                }
            )
            + "\n"
        )
    scenario = make_scenario(args.scenario, config, variant=args.variant, reference_file=args.reference)
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
                    "startup_seconds": time.perf_counter() - process_start,
                }
            )
        )
        print(f"READY: {args.connection_file}", flush=True)
        try:
            session.run()
        finally:
            server.close()
        return
    result = scenario.rollout()
    result["pid"] = os.getpid()
    result["process_seconds_after_import"] = time.perf_counter() - process_start
    if args.scenario == "panda_calibration":
        scenario.save_trace(args.output.with_suffix(".npz"))
        result["trace_path"] = scenario.last_trace_path
    if args.observe:
        session = make_session(scenario, args.output.parent / "observations")
        observation = session.dispatch("observe", {**camera(args.scenario), "width": 640, "height": 480})
        image_path = args.output.with_suffix(".png")
        image_path.write_bytes(base64.b64decode(observation.pop("image_base64")))
        observation["image_path"] = str(image_path)
        result["observation"] = observation
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if args.scenario != "panda_calibration":
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            body_q=np.asarray(scenario.pose_trace),
            target_q=np.asarray(scenario.target_trace),
            errors=np.asarray(scenario.errors),
            dt=scenario.dt,
            trace_step_interval=25,
        )
    with (args.output.parent / "rollouts.jsonl").open("a") as stream:
        stream.write(json.dumps(result, default=str) + "\n")
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
