# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepare isolated tasks and retain raw, independent Astra trial measurements.

This command launches agents only when --run is explicit. Confirmation workspaces
must have distinct output paths; existing paths are never silently reused.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .recording import Recording, digest
from .rollout import camera
from .scenarios import HUG_DATA, ROOT, SPECS, initial_config


def _command(name: str, variant: int, workspace: Path, *, output: Path | None = None) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(ROOT),
        "python",
        "-m",
        "tools.mcp_evaluation.rollout",
        "--scenario",
        name,
        "--variant",
        str(variant),
        "--config",
        str(workspace / "config.py"),
        "--output",
        str(output if output is not None else workspace / "metrics.json"),
    ]


def prepare(
    workspace: Path, name: str, condition: str, variant: int, seconds: int, *, phase: str = "development"
) -> dict:
    """Write identical task definitions with condition-specific tool directions."""
    workspace.mkdir(parents=True, exist_ok=False)
    config = initial_config(name, variant)
    (workspace / "config.py").write_text(
        "# Edit only these physical/setup parameters.\nCONFIG = " + repr(config) + "\n"
    )
    # A normal runnable baseline entrypoint; the scientific harness stays shared and immutable.
    (workspace / "rollout.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, "
        + repr(str(ROOT))
        + ")\nfrom tools.mcp_evaluation.rollout import main\nif __name__ == '__main__':\n    main()\n"
    )
    spec = {
        "scenario": name,
        "variant": variant,
        "condition": condition,
        "phase": phase,
        "budget_seconds": seconds,
        **SPECS[name],
        "camera": camera(name),
        "source_hashes": {
            str(path.relative_to(ROOT)): digest(path)
            for path in sorted(
                [
                    *Path(__file__).parent.glob("*.py"),
                    *(ROOT / "newton/_src/mcp").glob("*.py"),
                    ROOT / "newton/mcp.py",
                    ROOT / "newton/__init__.py",
                    ROOT / "newton/_src/solvers/mujoco/solver_mujoco.py",
                    ROOT / "newton/_src/solvers/mujoco/kernels.py",
                ]
            )
        },
    }
    (workspace / "task.json").write_text(json.dumps(spec, indent=2) + "\n")
    if name == "hug":
        provenance = Recording(HUG_DATA, variant=variant).provenance
        (workspace / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    common = f"""You are an independent GPT-6 Astra evaluation agent. Solve this Newton simulation task in this workspace.

Task: {SPECS[name]["description"]}
Variant: {variant}. Time budget: {seconds} seconds, maximum 12 full candidate rollouts.
Read task.json for parameter bounds and fixed quality thresholds. Success requires all thresholds and 1500 finite steps. Submit final parameters by editing this workspace's config.py CONFIG dict. Do not modify the reference trajectory, dynamics, scoring, task files, imported assets, or shared implementation. Only config.py is editable. You may inspect the common Newton and harness source files, official docs, and local source assets for this task. Do not read other trial directories, scenario feasibility results, other agents' conversations, or tuned answers. No subagents.

Both conditions have the same physical simulator, targets, measurements, images, parameter ranges, and final fresh-process verification. Use observations if useful; avoid unnecessary expensive rendering. Keep all runs and failed attempts. Do not claim success without completing a measured rollout. Finish with a brief report of your config and measured quality. Quality is independently verified after your process exits.

Common source directory: {ROOT / "tools/mcp_evaluation"}
Simulation backend: Newton SolverMuJoCo CPU with native MuJoCo contacts. Images use the same Newton sensor renderer. A generated collision-pipeline contact query is a diagnostic, distinct from native solver contacts. Duration 3 seconds, dt .002 seconds.
"""
    if condition == "restart":
        instructions = f"""
Edit config.py, then run a fresh process per candidate:
uv run --no-sync --project {ROOT} python rollout.py --scenario {name} --variant {variant} --config config.py --output metrics.json
Add --observe to save an image after a rollout. Read metrics.json and provenance.json. Each invocation must exit after its single rollout. You may batch independent candidates with one new process each. Do not keep a simulator process alive across candidates or use live MCP.
"""
    else:
        instructions = """
The Newton application is already running and the newton MCP server is configured with a compact code profile: describe, execute, observe and rebuild. You must use actual MCP tools for live queries and rollout control. Discovery is optional; use describe or session.dispatch('query', {...}) inside execute when useful. Use execute for the application's validated parameter interface:
result = session.scenario.apply_config({...})
For one complete candidate, batching in one execute call is allowed and efficient:
session.scenario.apply_config({...})
session.dispatch('reset')
session.dispatch('step', {'count': 1500})
result = session.scenario.metrics()
All structured operations remain available through session.dispatch inside execute. Multiple candidates may be batched within the same total 12-rollout budget.
The step operation advances the application's fixed targets and scoring. Use the observe MCP tool directly with the camera from task.json if useful; it returns an image content block. For HUG, execute result = session.scenario.provenance exposes the source/frame setup. Do not call scenario.rollout directly, modify scoring, or mutate state/targets. You may use dispatch query/edit to inspect or demonstrate model parameter handling, but candidate parameter changes should use apply_config so final settings are reproducible. If execute fails, use the rebuild tool to recover in the same process; it retains the last validated configuration by default and starts a fresh trial history. Optional rebuild arguments may contain a config dict. MCP startup is bounded at 30 seconds and each tool call at 300 seconds; a timed-out running mutation has an unknown outcome and must not be automatically retried. After success, write the exact final configuration to config.py for the independent fresh-process verification.
"""
    prompt = common + instructions
    (workspace / "TASK.md").write_text(prompt)
    return {"prompt": prompt, "spec": spec}


def _usage(events: list[dict]) -> dict:
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    result = dict.fromkeys(fields, 0)
    for event in events:
        if event.get("type") == "turn.completed":
            for field in fields:
                result[field] += int(event.get("usage", {}).get(field, 0))
    result["note"] = "Cached input and reasoning output are subsets, not additional tokens."
    return result


def _trial_measurements(workspace: Path) -> dict:
    """Count completed candidate and process logs throughout the trial workspace."""
    workspace = workspace.resolve()
    logs = {"candidate_rollout_logs": [], "simulation_process_logs": []}
    seen = set()
    for path in sorted(workspace.rglob("*.jsonl")):
        if not path.is_file():
            continue
        if path.name not in {"rollouts.jsonl", "live_rollouts.jsonl", "process_events.jsonl"}:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(workspace) or resolved in seen:
            continue
        relative = resolved.relative_to(workspace)
        if relative.parts[0] == "verification":
            continue
        seen.add(resolved)
        key = "simulation_process_logs" if path.name == "process_events.jsonl" else "candidate_rollout_logs"
        logs[key].append({"path": str(path.relative_to(workspace)), "records": len(path.read_text().splitlines())})
    return {
        "candidate_rollouts": sum(log["records"] for log in logs["candidate_rollout_logs"]),
        "simulation_process_starts_during_trial": sum(log["records"] for log in logs["simulation_process_logs"]),
        **logs,
    }


def _stop_process(process: subprocess.Popen) -> None:
    """Stop a trial and its POSIX process group before independent verification."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        # The agent may have exited before a background rollout in its group.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_trial(workspace: Path, prepared: dict) -> dict:
    """Launch one independent context, retain all events, then verify physics."""
    spec = prepared["spec"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    server = None
    startup = 0.0
    cmd = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--model",
        "gpt-6-astra",
        "-c",
        'model_reasoning_effort="xhigh"',
        "--sandbox",
        "danger-full-access",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "-C",
        str(workspace),
    ]
    if spec["condition"] == "live":
        connection = workspace / "connection.json"
        server_log = (workspace / "server.log").open("w")
        before = time.perf_counter()
        server = subprocess.Popen(
            [*_command(spec["scenario"], spec["variant"], workspace), "--live", "--connection-file", str(connection)],
            cwd=ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        while not (workspace / "server_ready.json").exists():
            if server.poll() is not None or time.perf_counter() - before > 120:
                _stop_process(server)
                server_log.close()
                raise RuntimeError(f"Live server did not become ready; inspect {workspace / 'server.log'}")
            time.sleep(0.05)
        startup = time.perf_counter() - before
        args = [
            "run",
            "--no-sync",
            "--project",
            str(ROOT),
            "python",
            "-m",
            "newton.mcp",
            "--connect",
            str(connection),
            "--profile",
            "code",
        ]
        cmd += [
            "-c",
            'mcp_servers.newton.command="uv"',
            "-c",
            "mcp_servers.newton.args=" + json.dumps(args),
            "-c",
            f'mcp_servers.newton.cwd="{ROOT}"',
            "-c",
            "mcp_servers.newton.tool_timeout_sec=300",
            "-c",
            "mcp_servers.newton.startup_timeout_sec=30",
        ]
    cmd.append("-")
    start = time.perf_counter()
    timed_out = False
    with (workspace / "agent.jsonl").open("w") as stdout, (workspace / "agent.stderr").open("w") as stderr:
        agent = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=workspace,
            env=env,
            text=True,
            start_new_session=os.name == "posix",
        )
        try:
            agent.communicate(prepared["prompt"], timeout=spec["budget_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _stop_process(agent)
    elapsed = time.perf_counter() - start
    if server is not None:
        _stop_process(server)
        server_log.close()
    events, malformed = [], []
    for line in (workspace / "agent.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            malformed.append(line)
    items = [e.get("item", {}) for e in events if e.get("type") == "item.completed"]
    tool_items = [i for i in items if i.get("type") in ("command_execution", "mcp_tool_call", "tool_call")]
    measurements = _trial_measurements(workspace)
    verification_output = workspace / "verification" / "metrics.json"
    verified = subprocess.run(
        _command(spec["scenario"], spec["variant"], workspace, output=verification_output),
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (workspace / "verification.log").write_text(verified.stdout + verified.stderr)
    quality = (
        json.loads(verification_output.read_text())
        if verified.returncode == 0
        else {"success": False, "verification_exit": verified.returncode}
    )
    summary = {
        "scenario": spec["scenario"],
        "variant": spec["variant"],
        "condition": spec["condition"],
        "phase": spec["phase"],
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "agent_elapsed_seconds": elapsed,
        "live_startup_seconds": startup,
        "startup_inclusive_seconds": elapsed + startup,
        "usage": _usage(events),
        "exit_code": agent.returncode,
        "timed_out": timed_out,
        "tool_items": len(tool_items),
        **measurements,
        "candidate_rollout_definition": "Logged completed rollouts; partial or interrupted attempts are retained in raw events but are not counted here.",
        "verification_process_starts": 1,
        "within_candidate_budget": measurements["candidate_rollouts"] <= 12,
        "shared_sources_unchanged": all(
            digest(ROOT / name) == checksum for name, checksum in spec["source_hashes"].items()
        ),
        "mcp_tool_items": sum(i.get("type") == "mcp_tool_call" for i in tool_items),
        "malformed_event_count": len(malformed),
        "quality": quality,
        "raw_events": "agent.jsonl",
        "verification_metrics": str(verification_output.relative_to(workspace)),
        "task_source_hashes": spec["source_hashes"],
    }
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Prepare or explicitly launch one evaluation trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SPECS, required=True)
    parser.add_argument("--condition", choices=("live", "restart"), required=True)
    parser.add_argument("--variant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    args = parser.parse_args()
    prepared = prepare(
        args.workspace.resolve(), args.scenario, args.condition, args.variant, args.seconds, phase=args.phase
    )
    if args.run:
        print(json.dumps(run_trial(args.workspace.resolve(), prepared), indent=2))
    else:
        print(f"Prepared {args.workspace}; no evaluation agent launched.")


if __name__ == "__main__":
    main()
