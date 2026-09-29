# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run one harness-improvement-loop trial: a Newton script task with or without the live MCP.

In the ``mcp`` condition the workspace script runs live under
``python -m newton.mcp host`` and the agent also keeps its shell; in ``restart``
the agent edits and reruns scripts. The submitted script is verified in a fresh
process after the agent exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from tools.mcp_evaluation.visual.run_visual import (
    MODELS,
    UNAVAILABLE,
    _agent_command,
    _stop,
    digest,
    mcp_available,
    parse_events,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"
# MCP tool profile: "lean" advertises only execute and rebuild to keep per-turn context small.
PROFILE = os.environ.get("NEWTON_MCP_PROFILE", "lean")
# Sibling live copies the MCP condition may use for parallel sweeps.
WORKERS = int(os.environ.get("NEWTON_MCP_WORKERS", "2"))
DP_DATA = Path(os.environ.get("NEWTON_DP_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/dp_real_task"))


def _task(name: str) -> dict:
    tasks = {
        "grasp_drift": {
            "files": {"franka_cube_shake.py": HERE / "grasp/franka_cube_shake.py"},
            "script": "franka_cube_shake.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/grasp/verify.py",
            "goal": """franka_cube_shake.py is a Newton simulation (SolverMuJoCo, Newton collision pipeline) of a Franka FR3 that grasps a 4 cm cube, lifts it, and shakes it along a Lissajous path. Problem: the grasp is not reliable. With a 10 cm / 1 Hz shake (`--shake-amplitude 0.1`) the cube slips and is dropped, and smaller shakes also let it creep.

Goal: change the scene so the grasp holds. Verification runs the script's Example in a fresh process and measures the cube position in the hand (TCP) frame from the start of the shake: over 11.5 s of 3 cm / 1 Hz shaking it must move less than 1 mm, and over 6 s of 10 cm / 1 Hz shaking it must stay in the hand and move less than 5 mm.
Constraints (checked): do not change the scripted motion (phases, durations, IK, shake path), joint gains or effort limits, gripper commands, the cube's size, density, or mass, gravity, or the timestep (60 Hz frames, 16 substeps), and do not add bodies, joints, or constraints between gripper and cube. Contact, material, collision, and solver settings may change. Keep the file runnable as a Newton example (`python franka_cube_shake.py --viewer null`). Report the root cause you found.""",
        },
        "g1_track": {
            "files": {name: HERE / "g1" / name for name in ("g1_track.py", "wave.csv", "high5.csv")},
            "script": "g1_track.py",
            "host_args": ["--motion", "wave.csv"],
            "verifier": "tools/mcp_evaluation/v4/g1/verify.py",
            "goal": """g1_track.py simulates a floating-base Unitree G1 humanoid (SolverMuJoCo, 50 Hz frames, 2 ms substeps) that should follow a Kimodo reference motion (wave.csv: root pose and 29 joint angles at 30 fps) using position servos with torque limits. Problem: with the current controller the robot falls.

Goal: set up the controller so the robot follows the reference motion without falling. Verification runs the script in a fresh process on wave.csv and on an unseen faster playback of the same clip: the root must stay upright (height above 0.6 m), root position RMSE at most 4 cm, and joint-angle RMSE at most 0.05 rad. It also reports a harder stretch clip (high5.csv) that is not required.
Constraints (checked): do not change the robot model (bodies, masses, armature, torque limits), the timestep, or the motion input and playback, and apply no forces or torques to the floating base. The controller (gains, targets, feedforward, feedback, anything in the step logic) may change. Keep the file runnable as a Newton example (`python g1_track.py --motion wave.csv --viewer null`).""",
        },
        "dp_real": {
            "files": {
                "double_pendulum.py": HERE / "dp_real/double_pendulum.py",
                **{f"train_{i:02d}.csv": DP_DATA / f"train_{i:02d}.csv" for i in range(10)},
            },
            "script": "double_pendulum.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/dp_real/verify.py",
            "goal": """double_pendulum.py models a real robot: the DFKI torque-controlled double pendulum (two quasi-direct-drive AK80-6 motors with 6:1 planetary gearing, one at the shoulder and one at the elbow; shoulder-to-elbow distance 0.2 m, second link 0.3 m). train_00.csv ... train_09.csv are recordings from the real hardware: measured joint angles and velocities and the measured motor torques (about 500 Hz). Problem: the model uses nominal CAD guesses, and its predictions drift quickly away from the recordings.

Goal: calibrate the model so it predicts the real robot. Verification imports the script's build_model(num_worlds) and make_solver(model) and runs its own multiple-shooting evaluation (the same protocol as the script's rollout/window_errors, 2 ms steps) on two held-out recordings of the same robot: a 20 s excitation run and a 75 s run with full swings. Each 0.5 s window starts from the measured state and is driven open loop by the measured torques. The mean joint-angle RMSE over the windows must be at most {heldout_10_rad} rad on the 20 s run and at most {heldout_11_rad} rad on the 75 s run. The starter scores about 0.14 and 0.34 rad.
Constraints (checked): keep the two revolute joints, the 0.2 m shoulder-to-elbow offset, gravity, positive masses and valid inertias, and the build_model/make_solver interface. Any physical parameter (masses, centers of mass, inertias, armature, damping, friction, ...) and the solver settings may change, and other Newton modeling features may be used inside build_model/make_solver. Keep the file runnable (`python double_pendulum.py --viewer null`).""",
        },
    }
    if name == "dp_real":
        from tools.mcp_evaluation.v4.dp_real.verify import THRESHOLDS  # noqa: PLC0415

        tasks[name]["goal"] = tasks[name]["goal"].format(**THRESHOLDS)
    return tasks[name]


def prompt_for(name: str, condition: str, workspace: Path, seconds: int, guide: str | None) -> str:
    task = _task(name)
    common = f"""You are working on a Newton physics simulation task.

Workspace: {workspace}
Newton source tree (read-only reference, including docs and examples): {ROOT}

{task["goal"]}

Deliverable: the edited {task["script"]} in the workspace, then a brief report. You have {seconds // 60} minutes; working efficiently matters. Do not modify files outside the workspace, do not look for other trials or hidden verification data, and do not use subagents.
"""
    run = f"uv run --no-sync --project {ROOT} python {task['script']} --viewer null --num-frames <N> {' '.join(task['host_args'])}".rstrip()
    if condition == "restart":
        return (
            common
            + f"""
Workflow: this is a script-based setup. Run the simulation with
  {run}
or write your own scripts that import the Example class; each run starts a fresh simulator process. There is no display; to look at the scene, render images yourself (for example with newton.sensors.SensorTiledCamera) and open them with your image-viewing tool.
"""
        )
    return (
        common
        + f"""
Workflow: {task["script"]} is already running live, hosted by the Newton MCP (`newton` server tools, e.g. newton_execute and newton_rebuild). Images returned by MCP tools appear directly in your context, and Python state persists between calls. After editing the script on disk, newton_rebuild reloads it in the live process. You may also run scripts ({run}) when that is more efficient; each such run starts a fresh process.
If no newton tools are available to you, reply only with {UNAVAILABLE} and stop.
{guide or ""}
"""
    )


def prepare(workspace: Path, name: str, condition: str, model: str, seconds: int, phase: str) -> dict:
    if condition not in ("mcp", "restart"):
        raise ValueError("condition must be mcp or restart")
    task = _task(name)
    workspace.mkdir(parents=True, exist_ok=False)
    for target, source in task["files"].items():
        shutil.copyfile(source, workspace / target)
    guide = None
    if condition == "mcp":
        guide = _host_guide(workspace, task)
    prompt = prompt_for(name, condition, workspace, seconds, guide)
    (workspace / "TASK.md").write_text(prompt)
    spec = {
        "task": name,
        "condition": condition,
        "model": model,
        **MODELS[model],
        "phase": phase,
        "budget_seconds": seconds,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "starter_sha256": {target: digest(workspace / target) for target in task["files"]},
        "harness_version": os.environ.get("NEWTON_HARNESS_VERSION", "unversioned"),
    }
    (workspace / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return {"spec": spec, "prompt": prompt}


def _host_guide(workspace: Path, task: dict) -> str:
    """The same application guide the MCP server sends in its instructions."""
    code = (
        "import sys; from newton.mcp import ExampleHost; "
        f"host = ExampleHost({str(workspace / task['script'])!r}, {task['host_args']!r}); "
        f"host.example = type('E', (), {{'frame_dt': '?'}})(); print(host.guide({WORKERS}))"
    )
    result = subprocess.run([str(PYTHON), "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    return result.stdout.strip()


def run_trial(workspace: Path, prepared: dict) -> dict:
    spec = prepared["spec"]
    task = _task(spec["task"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MCP_TOOL_TIMEOUT"] = "300000"
    env["MAX_MCP_OUTPUT_TOKENS"] = "60000"
    host, mcp, startup = None, None, 0.0
    started = time.time()
    if spec["condition"] == "mcp":
        connection = workspace / ".connection.json"
        before = time.perf_counter()
        log = (workspace / "host.log").open("w")
        host = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "newton.mcp",
                "host",
                task["script"],
                "--connection-file",
                str(connection),
                "--artifacts",
                str(workspace / "observations"),
                "--workers",
                str(WORKERS),
                "--",
                *task["host_args"],
            ],
            cwd=workspace,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ready = connection.with_suffix(".ready")
        while not ready.exists():
            if host.poll() is not None or time.perf_counter() - before > 300:
                _stop(host)
                raise RuntimeError(f"Host failed to start; see {workspace / 'host.log'}")
            time.sleep(0.1)
        startup = time.perf_counter() - before
        mcp = {
            "command": str(PYTHON),
            "args": ["-m", "newton.mcp", "--connect", str(connection), "--profile", PROFILE, "--timeout", "300"],
            # Load the Newton tools up front instead of behind a tool-search round trip (Claude Code only).
            "alwaysLoad": True,
        }
    command = _agent_command(spec, workspace, mcp)
    (workspace / "command.json").write_text(json.dumps(command))
    agent_start = time.perf_counter()
    timed_out = False
    with (
        (workspace / "agent.jsonl").open("w") as out,
        (workspace / "agent.times.jsonl").open("w") as times,
        (workspace / "agent.stderr").open("w") as err,
    ):
        agent = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            cwd=workspace,
            env=env,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def pump():
            for number, line in enumerate(agent.stdout):
                out.write(line)
                out.flush()
                times.write(json.dumps({"line": number, "seconds": time.perf_counter() - agent_start}) + "\n")

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            agent.stdin.write(prepared["prompt"])
            agent.stdin.close()
            agent.wait(timeout=spec["budget_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(agent.pid, 2)
                agent.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        finally:
            _stop(agent)
            reader.join(timeout=10)
    elapsed = time.perf_counter() - agent_start
    if host is not None:
        _stop(host)
    activity = parse_events(workspace / "agent.jsonl", spec["cli"])
    activity["mcp_available"] = mcp_available(workspace / "agent.jsonl", spec) if mcp is not None else None
    verification = verify(workspace, task)
    summary = {
        **spec,
        "started_unix": started,
        "application_startup_seconds": startup,
        "agent_seconds": elapsed,
        "total_seconds": elapsed + startup,
        "timed_out": timed_out,
        "agent_exit_code": agent.returncode,
        "load_average_end": os.getloadavg(),
        **activity,
        "verification": verification,
        "success": bool(verification.get("success")) and not timed_out,
    }
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def verify(workspace: Path, task: dict) -> dict:
    output = workspace / "verification.json"
    try:
        result = subprocess.run(
            [str(PYTHON), str(ROOT / task["verifier"]), str(workspace / task["script"]), "--output", str(output)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "verification timed out"}
    (workspace / "verification.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0 or not output.exists():
        return {"success": False, "error": (result.stderr or result.stdout)[-2000:]}
    data = json.loads(output.read_text())
    return {k: data[k] for k in ("success", "integrity", "failed_checks", "metrics", "normalized_worst")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("grasp_drift", "g1_track", "dp_real"))
    parser.add_argument("--condition", required=True, choices=("mcp", "restart"))
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--phase", default="loop")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    prepared = prepare(workspace, args.task, args.condition, args.model, args.seconds, args.phase)
    if not args.run:
        print(f"Prepared {workspace}")
        return
    summary = run_trial(workspace, prepared)
    attempt = 1
    while summary.get("mcp_available") is False and attempt < 3:
        workspace.rename(workspace.with_name(f"{workspace.name}.infra-failure-{attempt}"))
        prepared = prepare(workspace, args.task, args.condition, args.model, args.seconds, args.phase)
        summary = run_trial(workspace, prepared)
        summary["infrastructure_retries"] = attempt
        (workspace / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        attempt += 1
    print(
        json.dumps(
            {k: summary[k] for k in ("success", "total_seconds", "tool_call_total")} | {"usage": summary["usage"]}
        )
    )


if __name__ == "__main__":
    sys.exit(main())
