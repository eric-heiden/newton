# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Host a visual calibration task as a live Newton MCP application."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .tasks import task_class


def worker_guide(count: int) -> str:
    """Usage notes for sibling worker applications."""
    return f"""
Parallel workers: `workers` is a pool of {count} more copies of this application, each with its own `task` scene and persistent Python state. Use them to evaluate candidates concurrently:
workers.broadcast("def evaluate(p):\\n    task.set_params(p)\\n    ...\\n    return loss")  # define helpers on every worker once
losses = workers.map("evaluate(args)", [params_1, params_2, ...])  # runs in parallel, results in order
workers.submit(code, args) returns a future. Worker results must be JSON data; failed jobs return {{"error": ...}}."""


def guide(cls, workspace: Path) -> str:
    """Application-specific notes delivered through the MCP server instructions."""
    cameras = ", ".join(f"'{c.name}'" for c in cls.CAMERAS)
    first = cls.TRAIN_EPISODES[0]
    times = list(cls.REFERENCE_TIMES)
    camera = cls.CAMERAS[0]
    references = [str(workspace / "reference" / f"{first}_{camera.name}_t{t:.2f}.png") for t in times]
    return f"""Calibration task '{cls.name}'. Global `task` controls the scene (same API as the restart workflow):
- task.params; task.set_params({{...}}) rebuilds with new parameters and resets to t=0 (bounds: task.PARAMS).
- task.set_episode(name); training episodes {list(cls.TRAIN_EPISODES)}.
- task.reset(); task.step(frames); task.simulate_to(t_seconds); frame dt {cls.FRAME_DT:g} s.
- task.render([{cameras}]) -> {{camera: RGB uint8 array}} from the reference cameras; task.rollout() renders all reference times.
- task.camera('{camera.name}').observe_arguments() gives eye/target/up/fov_y/width/height for newton_observe/newton_filmstrip.
Reference photos: {workspace / "reference"}/<episode>_<camera>_t<time>.png at times {times}; sheet_<episode>.png shows each episode.
Example comparison of the current parameters with the reference, one image:
newton_filmstrip(reset=true, times={times}, views=[<task.camera('{camera.name}').observe_arguments()>], references=[{json.dumps(references)}])
(call task.set_episode(...) first for another episode). Write final parameters to {workspace / "params.json"}."""


def make_session(task, workspace: Path, workers: list[Path] | None = None):
    """Bind the task to the public live API, adapting to the embedded MCP version."""
    from newton.mcp import SimulationSession  # noqa: PLC0415

    def bindings(current):
        return {
            "model": current.model,
            "solver": current.solver,
            "state": current.state_0,
            "state_next": current.state_1,
            "control": current.control,
            "collision_pipeline": getattr(current, "pipeline", None),
            "contacts": getattr(current, "contacts", None),
        }

    stepping = {"active": False}

    def step(session, dt):
        stepping["active"] = True
        try:
            task.step(1)
        finally:
            stepping["active"] = False
        session.state, session.state_next = task.state_0, task.state_1

    def reset(session):
        task.reset()
        session.state, session.state_next = task.state_0, task.state_1

    def rebuild(session, config=None, **_):
        if config:
            task.set_params(config)
        else:
            task._build_all()
        return bindings(task)

    options = {}
    parameters = inspect.signature(SimulationSession).parameters
    if "namespace" in parameters:
        options["namespace"] = {"task": task}
    if "guide" in parameters:
        options["guide"] = guide(type(task), workspace) + (worker_guide(len(workers)) if workers else "")
    if workers:
        options["workers"] = workers
    session = SimulationSession(
        **bindings(task),
        dt=task.FRAME_DT,
        step_callback=step,
        reset_callback=reset,
        rebuild_callback=rebuild,
        allow_execute=True,
        artifact_directory=workspace / "observations",
        **options,
    )
    session.task = task
    task.live_session = session
    keep = "keep_workspace" in inspect.signature(session.replace).parameters

    def on_rebuild(current):
        if keep:
            session.replace(**bindings(current), keep_workspace=True)
        else:
            # Older sessions clear the workspace on replace; rebind attributes instead so a running cell survives.
            session.model, session.solver = current.model, current.solver
            session.state, session.state_next, session.control = current.state_0, current.state_1, current.control
            if getattr(session, "_renderer", None) is not None:
                session._renderer.close()
                session._renderer = None
            session._initial = session._snapshot()
            session.revision += 1
        session.time, session.frame = current.time, current.frame

    def on_state_change(current):
        if not stepping["active"]:
            session.state, session.state_next = current.state_0, current.state_1
            session.time, session.frame = current.time, current.frame

    task.on_rebuild = on_rebuild
    task.on_state_change = on_state_change
    return session


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--connection-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0, help="Start this many sibling worker applications")
    parser.add_argument("--ready-file", type=Path, help="Readiness marker (default: server_ready.json)")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    params_path = workspace / "params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else None
    with (workspace / "process_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"pid": os.getpid(), "event": "live_application_start", "wall_time_unix": time.time()}) + "\n"
        )
    worker_paths, children = [], []
    for index in range(args.workers):
        # Workers share the task and candidate log; each has its own scene and connection.
        worker_paths.append(workspace / f".worker-{index}.json")
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tools.mcp_evaluation.visual.app",
                    "--task",
                    args.task,
                    "--workspace",
                    str(workspace),
                    "--connection-file",
                    str(worker_paths[-1]),
                    "--ready-file",
                    str(workspace / f".worker-{index}-ready.json"),
                ],
                cwd=workspace,
                stdout=(workspace / f"worker-{index}.log").open("w"),
                stderr=subprocess.STDOUT,
            )
        )
    task = task_class(args.task)(params)
    for index, child in enumerate(children):
        while not (workspace / f".worker-{index}-ready.json").exists():
            if child.poll() is not None:
                raise RuntimeError(f"Worker {index} exited; see worker-{index}.log")
            time.sleep(0.05)
    session = make_session(task, workspace, worker_paths or None)
    from newton.mcp import SimulationServer  # noqa: PLC0415

    server = SimulationServer(session, connection_file=args.connection_file)
    server.start()
    (args.ready_file or workspace / "server_ready.json").write_text(
        json.dumps({"pid": os.getpid(), "startup_seconds": time.perf_counter() - started})
    )
    print(f"READY: {args.connection_file}", flush=True)
    try:
        session.run()
    finally:
        server.close()
        for child in children:
            child.terminate()


if __name__ == "__main__":
    main()
