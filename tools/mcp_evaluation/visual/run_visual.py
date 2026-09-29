# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run one visual-calibration trial: live MCP or edit/restart, Claude Code or Codex.

Agents launch only with --run. Workspaces must not exist beforehand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private"))
MODELS = {
    "opus": {"cli": "claude", "model": "claude-opus-5-5", "effort": "xhigh"},
    "astra": {"cli": "codex", "model": "gpt-6-astra", "effort": "xhigh"},
}
DESCRIPTIONS = {
    "cloth_drape": "A 1 m square cloth is dropped flat onto a box (or other obstacle) and drapes under gravity. "
    "Calibrate the cloth material: stretch stiffness, bending stiffness, areal density, and friction.",
    "push": "A red capsule pusher pushes a blue box (yellow marker on its +x end) across a table and stops. The box "
    "carries a hidden internal load. Calibrate table friction, pusher friction, and the box's in-plane "
    "center-of-mass offset.",
    "arm_offsets": "A Franka Panda arm is photographed at several commanded joint configurations. The real arm's "
    "joint zero positions differ from the model's: actual angle = commanded + offset. Calibrate the seven joint "
    "offsets.",
}


UNAVAILABLE = "NEWTON_TOOLS_UNAVAILABLE"
WORKERS = 3
"""Sibling worker applications in the mcp_workers condition (four application processes in total)."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_for(task: str, condition: str, workspace: Path, seconds: int, guide: str | None) -> str:
    minutes = seconds // 60
    common = f"""You are calibrating a Newton physics simulation so that it reproduces reference photographs of a physical system.

Task: {DESCRIPTIONS[task]}
Workspace: {workspace}
- reference/reference.json: parameter names, bounds, units, training episodes, reference times, and the calibrated camera for every photo.
- reference/<episode>_<camera>_t<time>.png: reference photos; reference/sheet_<episode>.png: all photos of one episode.
- params.json: starting parameters. They are generic library defaults, not an informed guess.

Goal: find parameters for which the simulation reproduces the reference photos. Final quality is scored after you finish on held-out episodes with different initial conditions against hidden ground truth, so identify the physics rather than overfitting pixels. You have {minutes} minutes; working efficiently matters, and you should stop once the simulation matches the photos as well as you can make it.
Deliverable: write the final parameters as a JSON object containing every parameter name to {workspace / "params.json"}, then give a brief report.
Rules: work only inside this workspace plus the read-only Newton source tree {ROOT} (and its docs). Do not search for or read hidden ground truth, other trials, or files elsewhere. Do not use subagents.
"""
    if condition == "restart":
        return (
            common
            + f"""
Workflow: this is a script-based setup. Run
  uv run --no-sync --project {ROOT} python sim.py
to simulate the training episodes with params.json. It writes out/compare_<episode>.png (rows per camera: simulated, reference, mismatch in magenta) and prints pixel statistics as JSON. Edit params.json (or pass --params other.json) and rerun; each run starts a fresh simulator process. You may also write your own scripts using the same task API that sim.py uses (tools/mcp_evaluation/visual in the source tree). Look at images with your image-viewing tool.
"""
        )
    tools = (
        "newton_execute, newton_observe, newton_filmstrip, newton_describe, newton_rebuild"
        if condition == "mcp"
        else "newton_execute, newton_observe, newton_describe, newton_rebuild"
    )
    return (
        common
        + f"""
Workflow: the simulation is already running in a live Newton application, connected through the `newton` MCP server (tools: {tools}). Images returned by MCP tools appear directly in your context, and Python state persists in the application between calls. You may also write and run your own scripts with the same task API (tools/mcp_evaluation/visual in the source tree) when that is more efficient; each script run starts a fresh simulator process.
Start by calling newton_describe once to confirm the connection. If no newton tools are available to you, reply only with {UNAVAILABLE} and stop.
{guide}
"""
    )


def prepare(workspace: Path, task: str, condition: str, model: str, seconds: int, *, phase: str) -> dict:
    if condition not in ("mcp", "mcp_workers", "mcp_v2", "restart"):
        raise ValueError("condition must be mcp, mcp_workers, mcp_v2, or restart")
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    workspace.mkdir(parents=True, exist_ok=False)
    source = PRIVATE / "reference" / task
    shutil.copytree(source, workspace / "reference", ignore=shutil.ignore_patterns("*.npz"))
    sys.path.insert(0, str(ROOT))
    from tools.mcp_evaluation.visual.tasks import task_class  # noqa: PLC0415

    cls = task_class(task)
    (workspace / "params.json").write_text(json.dumps(cls.default_params(), indent=2) + "\n")
    guide = None
    if condition == "restart":
        template = (Path(__file__).parent / "sim_template.py").read_text()
        (workspace / "sim.py").write_text(template.replace("__ROOT__", repr(str(ROOT))).replace("__TASK__", repr(task)))
    else:
        from tools.mcp_evaluation.visual.app import guide as make_guide  # noqa: PLC0415

        guide = make_guide(cls, workspace)
        if condition == "mcp_workers":
            from tools.mcp_evaluation.visual.app import worker_guide  # noqa: PLC0415

            guide += worker_guide(WORKERS)
        if condition == "mcp_v2":
            guide = (
                "This MCP version has no application globals: first execute `task = session.task`.\n"
                + guide.split("Example comparison")[0]
                + (
                    "Use newton_observe with a reference camera's arguments to see the current state; "
                    f"Write final parameters to {workspace / 'params.json'}."
                )
            )
    prompt = prompt_for(task, condition, workspace, seconds, guide)
    (workspace / "TASK.md").write_text(prompt)
    spec = {
        "task": task,
        "condition": condition,
        "model": model,
        **MODELS[model],
        "phase": phase,
        "budget_seconds": seconds,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "reference_sha256": {p.name: digest(p) for p in sorted((workspace / "reference").iterdir())},
    }
    (workspace / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return {"spec": spec, "prompt": prompt}


def _stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _agent_command(spec: dict, workspace: Path, mcp: dict | None) -> list[str]:
    if spec["cli"] == "claude":
        command = [
            "claude",
            "-p",
            "--model",
            spec["model"],
            "--effort",
            spec["effort"],
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--setting-sources",
            "project",
            "--strict-mcp-config",
            "--disallowedTools",
            "Task,Agent,WebSearch,WebFetch",
        ]
        config = {"mcpServers": {}}
        if mcp is not None:
            config["mcpServers"]["newton"] = {"command": mcp["command"], "args": mcp["args"]}
            if mcp.get("alwaysLoad"):
                # Keep only this server's tools out of tool search; built-in tools stay deferred.
                config["mcpServers"]["newton"]["alwaysLoad"] = True
        (workspace / ".agent-mcp.json").write_text(json.dumps(config))
        return [*command, "--mcp-config", str(workspace / ".agent-mcp.json")]
    command = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--model",
        spec["model"],
        "-c",
        f'model_reasoning_effort="{spec["effort"]}"',
        "-c",
        'web_search="disabled"',
        "--sandbox",
        "danger-full-access",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "-C",
        str(workspace),
    ]
    if mcp is not None:
        command += [
            "-c",
            f'mcp_servers.newton.command="{mcp["command"]}"',
            "-c",
            "mcp_servers.newton.args=" + json.dumps(mcp["args"]),
            "-c",
            "mcp_servers.newton.tool_timeout_sec=300",
            "-c",
            "mcp_servers.newton.startup_timeout_sec=60",
            # Without this, Codex intermittently starts a session before the server's tools are listed.
            "-c",
            "mcp_servers.newton.required=true",
        ]
    return [*command, "-"]


def parse_events(path: Path, cli: str) -> dict:
    """Normalize usage and tool activity across Claude Code and Codex event streams."""
    events = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    tools: dict[str, int] = {}
    images_returned = 0
    errors = 0
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0}
    cost = None
    turns = 0
    final_text = ""
    if cli == "claude":
        pending = {}
        for event in events:
            if event.get("type") == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        tools[name] = tools.get(name, 0) + 1
                        pending[block.get("id")] = name
                    elif block.get("type") == "text":
                        final_text = block.get("text", "")
            elif event.get("type") == "user":
                content = event.get("message", {}).get("content", [])
                for block in content if isinstance(content, list) else []:
                    if block.get("type") == "tool_result":
                        if block.get("is_error"):
                            errors += 1
                        inner = block.get("content")
                        if isinstance(inner, list):
                            images_returned += sum(1 for item in inner if item.get("type") == "image")
            elif event.get("type") == "result":
                u = event.get("usage", {})
                usage = {
                    "input_tokens": int(u.get("input_tokens", 0))
                    + int(u.get("cache_read_input_tokens", 0))
                    + int(u.get("cache_creation_input_tokens", 0)),
                    "cached_input_tokens": int(u.get("cache_read_input_tokens", 0)),
                    "cache_write_tokens": int(u.get("cache_creation_input_tokens", 0)),
                    "output_tokens": int(u.get("output_tokens", 0)),
                }
                cost = event.get("total_cost_usd")
                turns = int(event.get("num_turns", 0))
                final_text = event.get("result", final_text) or final_text
    else:
        for event in events:
            if event.get("type") == "turn.completed":
                u = event.get("usage", {})
                usage["input_tokens"] += int(u.get("input_tokens", 0))
                usage["cached_input_tokens"] += int(u.get("cached_input_tokens", 0))
                usage["cache_write_tokens"] += int(u.get("cache_write_input_tokens", 0))
                usage["output_tokens"] += int(u.get("output_tokens", 0))
            if event.get("type") != "item.completed":
                continue
            item = event.get("item", {})
            kind = item.get("type")
            if kind == "mcp_tool_call":
                name = f"mcp__{item.get('server')}__{item.get('tool')}"
                tools[name] = tools.get(name, 0) + 1
                result = item.get("result") or {}
                images_returned += sum(1 for part in result.get("content", []) or [] if part.get("type") == "image")
                if item.get("error") or item.get("status") == "failed" or result.get("isError"):
                    errors += 1
            elif kind == "command_execution":
                tools["shell"] = tools.get("shell", 0) + 1
                if item.get("exit_code") not in (None, 0):
                    errors += 1
            elif kind == "agent_message":
                final_text = item.get("text", "")
                turns += 1
            elif kind not in ("reasoning",):
                tools[kind] = tools.get(kind, 0) + 1
    usage["uncached_input_plus_output"] = usage["input_tokens"] - usage["cached_input_tokens"] + usage["output_tokens"]
    return {
        "usage": usage,
        "cost_usd": cost,
        "tool_calls": tools,
        "tool_call_total": sum(tools.values()),
        "tool_errors": errors,
        "mcp_images_returned": images_returned,
        "agent_turns": turns,
        "final_message": final_text[-4000:],
        "event_count": len(events),
    }


def run_trial(workspace: Path, prepared: dict, *, mcp_root: Path | None = None) -> dict:
    spec = prepared["spec"]
    env = dict(os.environ)
    env["NEWTON_VISUAL_LOG"] = str(workspace / "candidates.jsonl")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MCP_TOOL_TIMEOUT"] = "300000"
    env["MAX_MCP_OUTPUT_TOKENS"] = "60000"
    app = None
    startup = 0.0
    mcp = None
    started = time.time()
    if spec["condition"] in ("mcp", "mcp_workers", "mcp_v2"):
        app_env = dict(env)
        overlay = None
        if spec["condition"] == "mcp_v2":
            # Run the unchanged older Newton package with the same task code: shadow only `newton`.
            overlay = workspace / ".newton-v2"
            overlay.mkdir()
            (overlay / "newton").symlink_to((mcp_root or ROOT) / "newton")
            app_env["PYTHONPATH"] = os.pathsep.join([str(overlay), env["PYTHONPATH"]])
        connection = workspace / ".connection.json"
        log = (workspace / "app.log").open("w")
        before = time.perf_counter()
        app = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "tools.mcp_evaluation.visual.app",
                "--task",
                spec["task"],
                "--workspace",
                str(workspace),
                "--connection-file",
                str(connection),
                *(["--workers", str(WORKERS)] if spec["condition"] == "mcp_workers" else []),
            ],
            cwd=workspace,
            env=app_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while not (workspace / "server_ready.json").exists():
            if app.poll() is not None or time.perf_counter() - before > 180:
                _stop(app)
                raise RuntimeError(f"Live application failed; see {workspace / 'app.log'}")
            time.sleep(0.05)
        startup = time.perf_counter() - before
        mcp = {
            "command": str(ROOT / ".venv/bin/python"),
            "args": ["-m", "newton.mcp", "--connect", str(connection), "--profile", "code", "--timeout", "300"],
        }
        if overlay is not None:
            mcp["command"] = "env"
            mcp["args"] = [f"PYTHONPATH={overlay}", str(ROOT / ".venv/bin/python"), *mcp["args"]]
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
            # Arrival times per event line separate model latency from tool execution time.
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
            # An interrupt lets the CLI flush its final usage record before termination.
            try:
                os.killpg(agent.pid, signal.SIGINT)
                agent.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        finally:
            _stop(agent)
            reader.join(timeout=10)
    elapsed = time.perf_counter() - agent_start
    if app is not None:
        _stop(app)
    activity = parse_events(workspace / "agent.jsonl", spec["cli"])
    activity["mcp_available"] = mcp_available(workspace / "agent.jsonl", spec) if mcp is not None else None
    load = os.getloadavg()
    verification = verify(workspace, spec["task"])
    candidates = candidate_summary(workspace / "candidates.jsonl", started)
    summary = {
        **spec,
        "started_unix": started,
        "application_startup_seconds": startup,
        "agent_seconds": elapsed,
        "total_seconds": elapsed + startup,
        "timed_out": timed_out,
        "agent_exit_code": agent.returncode,
        "load_average_end": load,
        **activity,
        **candidates,
        "verification": verification,
        "success": bool(verification.get("success")) and not timed_out,
        "references_unchanged": all(
            digest(workspace / "reference" / name) == value for name, value in spec["reference_sha256"].items()
        ),
    }
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def mcp_available(path: Path, spec: dict) -> bool:
    """Whether the agent session exposed the newton MCP tools (Codex occasionally omits them)."""
    text = path.read_text(errors="replace")
    if spec["cli"] == "claude":
        for line in text.splitlines()[:5]:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("subtype") == "init":
                return any(
                    s.get("name") == "newton" and s.get("status") == "connected" for s in event.get("mcp_servers", [])
                )
    return '"mcp_tool_call"' in text and UNAVAILABLE not in text


def candidate_summary(path: Path, started: float) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    params = [r for r in records if r.get("event") == "params"]
    unique = {json.dumps(r["params"], sort_keys=True) for r in params}
    return {
        "parameter_sets_applied": len(params),
        "unique_parameter_sets": len(unique),
        "simulated_seconds": round(sum(r.get("seconds", 0.0) for r in records if r.get("event") == "simulated"), 3),
        "simulator_processes": len({r["pid"] for r in records}),
    }


def verify(workspace: Path, task: str) -> dict:
    params_path = workspace / "params.json"
    output = workspace / "verification.json"
    try:
        result = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "tools.mcp_evaluation.visual.tasks",
                "verify",
                "--task",
                task,
                "--private",
                str(PRIVATE),
                "--params",
                str(params_path),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env={**os.environ, "NEWTON_VISUAL_LOG": str(workspace / "verification-candidates.jsonl")},
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "verification timed out"}
    (workspace / "verification.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0 or not output.exists():
        return {"success": False, "error": (result.stderr or result.stdout)[-2000:]}
    data = json.loads(output.read_text())
    return {
        "success": data["success"],
        "heldout_worst": data["heldout"]["worst"],
        "training_worst": data["training"]["worst"],
        "normalized_worst": data["normalized_worst"],
        "params": data["params"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(DESCRIPTIONS))
    parser.add_argument("--condition", required=True, choices=("mcp", "mcp_workers", "mcp_v2", "restart"))
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--mcp-root", type=Path, help="Newton source root for the mcp_v2 condition")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepared = prepare(args.workspace.resolve(), args.task, args.condition, args.model, args.seconds, phase=args.phase)
    if args.run:
        workspace = args.workspace.resolve()
        summary = run_trial(workspace, prepared, mcp_root=args.mcp_root)
        attempt = 1
        # A session without the MCP tools is an infrastructure failure, not a result; retain it and retry.
        while summary.get("mcp_available") is False and attempt < 3:
            failed = workspace.with_name(f"{workspace.name}.infra-failure-{attempt}")
            workspace.rename(failed)
            prepared = prepare(workspace, args.task, args.condition, args.model, args.seconds, phase=args.phase)
            summary = run_trial(workspace, prepared, mcp_root=args.mcp_root)
            summary["infrastructure_retries"] = attempt
            (workspace / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            attempt += 1
        print(json.dumps(summary, indent=2))
    else:
        print(f"Prepared {args.workspace}")


if __name__ == "__main__":
    sys.exit(main())
