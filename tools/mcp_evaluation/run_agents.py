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
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from .recording import Recording, digest
from .rollout import camera
from .scenarios import HUG_DATA, ROOT, SPECS, initial_config


def source_hashes() -> dict:
    """Cover the full numerical implementation, including inverse dynamics and importers."""
    paths = sorted({*(ROOT / "newton").rglob("*.py"), *Path(__file__).parent.glob("*.py")})
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def write_task(workspace: Path, spec: dict) -> None:
    """Keep scientific task instructions compact while retaining full source commitments."""
    manifest = {key: spec[key] for key in ("source_hashes", "geometry_hashes") if key in spec}
    (workspace / "integrity-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    spec["integrity_manifest_sha256"] = digest(workspace / "integrity-manifest.json")
    public = {key: value for key, value in spec.items() if key not in manifest}
    public["integrity_manifest"] = "integrity-manifest.json"
    (workspace / "task.json").write_text(json.dumps(public, indent=2) + "\n")


def _command(
    name: str, variant: int, workspace: Path, *, output: Path | None = None, reference_file: Path | None = None
) -> list[str]:
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
        name,
        "--variant",
        str(variant),
        "--config",
        str(workspace / "config.py"),
        "--output",
        str(output if output is not None else workspace / "metrics.json"),
    ]
    if name == "panda_calibration":
        command += ["--reference", str(workspace / "reference.npz" if reference_file is None else reference_file)]
    return command


def prepare(
    workspace: Path,
    name: str,
    condition: str,
    variant: int,
    seconds: int,
    *,
    phase: str = "development",
    reference_file: Path | None = None,
    verification_reference_file: Path | None = None,
) -> dict:
    """Write identical task definitions with condition-specific tool directions."""
    if condition not in {"live", "restart", "ipython", "ipython_fixed"}:
        raise ValueError("condition must be live, restart, ipython, or ipython_fixed")
    if name == "panda_calibration":
        if reference_file is None or verification_reference_file is None:
            raise ValueError("Calibration requires separate training and held-out reference NPZ files")
        for path, episodes in ((reference_file, (0, 1)), (verification_reference_file, (2,))):
            with np.load(path, allow_pickle=False) as reference:
                if tuple(reference["episodes"]) != episodes or reference["q"].shape != (len(episodes), 1500, 7):
                    raise ValueError("Calibration reference has unexpected episodes or observation shape")
                if not np.isfinite(reference["q"]).all():
                    raise ValueError("Calibration reference observations must be finite")
    elif reference_file is not None or verification_reference_file is not None:
        raise ValueError("Reference files are only used for panda_calibration")
    workspace.mkdir(parents=True, exist_ok=False)
    if reference_file is not None:
        shutil.copyfile(reference_file, workspace / "reference.npz")
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
        "source_hashes": source_hashes(),
    }
    if reference_file is not None:
        spec["reference_sha256"] = digest(reference_file)
        spec["verification_reference_sha256"] = digest(verification_reference_file)
    write_task(workspace, spec)
    if name == "hug":
        provenance = Recording(HUG_DATA, variant=variant).provenance
        (workspace / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    frames = round(SPECS[name]["duration_s"] / SPECS[name]["dt_s"])
    common = f"""You are an independent GPT-6 Astra evaluation agent. Solve this Newton simulation task in this workspace.

Task: {SPECS[name]["description"]}
Variant: {variant}. Time budget: {seconds} seconds, maximum 12 full candidate rollouts.
Read task.json for parameter bounds and fixed quality thresholds. Success requires all thresholds and {frames} finite steps. Submit final parameters by editing this workspace's config.py CONFIG dict. Do not modify the reference trajectory, dynamics, scoring, task files, imported assets, or shared implementation. Only config.py is editable. You may inspect the common Newton and harness source files, official docs, and local source assets for this task. Do not read other trial directories, scenario feasibility results, other agents' conversations, or tuned answers. No subagents.

All conditions have the same physical simulator, targets, measurements, images, parameter ranges, and final fresh-process verification. Use observations if useful; avoid unnecessary expensive rendering. Keep all runs and failed attempts. Do not claim success without completing a measured rollout. Finish with a brief report of your config and measured quality. Quality is independently verified after your process exits.

Common source directory: {ROOT / "tools/mcp_evaluation"}
Simulation backend: Newton SolverMuJoCo CPU with native MuJoCo contacts. Images use the same Newton sensor renderer. A generated collision-pipeline contact query is a diagnostic, distinct from native solver contacts. Duration {SPECS[name]["duration_s"]:g} seconds, dt .002 seconds.
"""
    if name == "panda_calibration":
        common += """
This is a separately specified synthetic identification task on a real Panda asset, not physical robot calibration. Gain/controller settings and command motions are fixed. Infer payload_mass [kg], damping_multiplier [dimensionless], and joint_friction [N m]. reference.npz contains only episodes [0,1] and noisy response q [2,1500,7], with independent prescribed Gaussian position noise SD .0002 rad. The public command formula is CalibrationScenario.target(time_s, episode). One candidate consists of both training episodes (3000 steps total); the application automatically resets physics between them. All per-episode and pooled thresholds must pass. You may use numerical fitting or analytical estimates, and may batch candidates; there is no minimum candidate count. Do not create extra simulations outside the allowed candidate workflow.

Each completed candidate exports q/qd/target_q/errors at every step and body_q every 25 steps to the NPZ identified by metrics.trace_path. Both conditions receive these observations. You may load the NPZ or inspect the same current in-memory traces through execute. Final verification tests a third withheld response with the same physical parameters. The withheld response, generating parameters, seeds, private generator files, and feasibility artifacts are outside your allowed inputs: do not search for or read them. The common forward-model code does not encode the reference parameter tuple. Submit a configuration with measured passing training quality; final success additionally requires the independent held-out verification.
"""
    if condition == "restart":
        instructions = f"""
Edit config.py, then run a fresh process per candidate:
uv run --no-sync --project {ROOT} python rollout.py --scenario {name} --variant {variant} --config config.py --output metrics.json{" --reference reference.npz" if name == "panda_calibration" else ""}
Add --observe to save an image after a rollout. Read metrics.json and provenance.json. Each invocation must exit after its single rollout. You may batch independent candidates with one new process each. Do not keep a simulator process alive across candidates or use live MCP.
"""
    elif condition in {"ipython", "ipython_fixed"}:
        instructions = f"""
The same Newton application is running inside a fresh IPython kernel. The upstream ipython-mcp server is configured. Call connect_to_kernel(connection_file={str(workspace / "kernel.json")!r}) once, then execute_code(code=...) for actual Python access. Do not start a second kernel or simulator. The kernel contains session, scenario, model, solver, state, state_next, control, contacts, np and wp. User variables, imports and functions persist across calls; live application aliases refresh between cells after state swaps and rebuilds.
For one complete candidate, batching in one code call is allowed and efficient:
session.scenario.apply_config({{...}})
session.dispatch('reset')
session.dispatch('step', {{'count': {frames}}})
session.scenario.metrics()
The final expression is displayed; unlike Newton MCP, assigning result alone does not display it. You may loop over candidates and return or print a compact list, within the same total 12-rollout budget. All shared structured helpers remain available through session.dispatch; direct Python application inspection is allowed.
Use session.dispatch('observe', camera_settings) to generate the same sensor observations, save the returned image_base64 as a PNG and inspect that file with your image tool. Avoid printing image_base64. For HUG, session.scenario.provenance exposes the source/frame setup. Do not call scenario.rollout directly, modify scoring, or mutate state/targets. Candidate parameter changes use apply_config so final settings are reproducible.
The upstream server waits 30 seconds for a code reply internally. Keep each execution within that duration; splitting a batch is allowed. A timeout does not prove execution stopped: do not automatically repeat mutations. MCP tool timeout is 300 seconds. Ordinary Python errors retain the IPython namespace and may have partially mutated the scene. Use session.dispatch('rebuild') if recovery is necessary; it retains the latest validated configuration by default. Do not read or print the kernel connection file; pass its path to connect_to_kernel.
After success, write the exact final configuration to config.py for independent fresh-process verification. Use actual IPython MCP tools for running simulation access; do not connect to the kernel using a separate shell client.
"""
        if condition == "ipython_fixed":
            instructions = instructions.replace(
                "The upstream server waits 30 seconds for a code reply internally. Keep each execution within that duration; splitting a batch is allowed.",
                "This sensitivity control uses the upstream server with a disclosed reply-correlation fix and a 300-second execution wait. Keep each execution within that duration; splitting a batch is allowed.",
            )
            spec["ipython_server_variant"] = "reply_correlation_fix"
        else:
            spec["ipython_server_variant"] = "unmodified_upstream"
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
The step operation advances the application's fixed targets and scoring. Use the observe MCP tool directly with the camera from task.json if useful; it returns an image content block. For HUG, execute result = session.scenario.provenance exposes the source/frame setup. Do not call scenario.rollout directly, modify scoring, or mutate state/targets. You may use dispatch query/edit to inspect or demonstrate model parameter handling, but candidate parameter changes should use apply_config so final settings are reproducible. Python imports, variables and functions persist across execute calls; the final expression is returned, or assign result explicitly. Execution errors preserve variables but pause and invalidate the simulation. Use execute with recovery="inspect" to diagnose, or recovery="acknowledge" only after verifying or repairing coherent state. Alternatively use rebuild in the same process; it retains the last validated configuration by default and clears the Python workspace. Optional rebuild arguments may contain a config dict. MCP startup is bounded at 30 seconds and each tool call at 300 seconds; a timed-out running mutation has an unknown outcome and must not be automatically retried. After success, write the exact final configuration to config.py for the independent fresh-process verification.
"""
        instructions = instructions.replace("{'count': 1500}", "{'count': " + str(frames) + "}")
    prompt = common + instructions
    write_task(workspace, spec)
    (workspace / "TASK.md").write_text(prompt)
    return {
        "prompt": prompt,
        "spec": spec,
        "task_sha256": digest(workspace / "task.json"),
        "verification_reference_file": None
        if verification_reference_file is None
        else str(verification_reference_file.resolve()),
    }


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


def _action_failures(items: list[dict]) -> dict:
    """Distinguish failed commands and tool error indications from successful calls."""
    commands = []
    mcp = []
    for item in items:
        if item.get("type") == "command_execution" and item.get("exit_code") not in (None, 0):
            commands.append(item.get("id"))
        if item.get("type") != "mcp_tool_call":
            continue
        result = item.get("result") or {}
        texts = [part.get("text", "") for part in result.get("content", []) if part.get("type") == "text"]
        if (
            item.get("error")
            or item.get("status") == "failed"
            or result.get("isError")
            or any(text.lstrip().startswith("❌") for text in texts)
        ):
            mcp.append(item.get("id"))
    return {"failed_command_ids": commands, "mcp_error_indication_ids": mcp}


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


def _calibration_training_quality(workspace: Path, config: dict) -> dict:
    """Retain the last complete training measurement of the submitted candidate."""
    matches = []
    for path in workspace.rglob("*.jsonl"):
        if path.name not in {"rollouts.jsonl", "live_rollouts.jsonl"}:
            continue
        if path.relative_to(workspace).parts[0] == "verification":
            continue
        for index, line in enumerate(path.read_text().splitlines()):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(result, dict):
                continue
            if (
                result.get("scenario") == "panda_calibration"
                and result.get("config") == config
                and result.get("episodes") == [0, 1]
                and result.get("frames") == 3000
                and result.get("sample_count") == 3000
            ):
                matches.append((path.stat().st_mtime_ns, index, result))
    return (
        max(matches, key=lambda match: match[:2])[2]
        if matches
        else {"success": False, "reason": "No complete training measurement of the submitted configuration"}
    )


def _calibration_verification(workspace: Path, spec: dict, quality: dict, reference_file: Path) -> tuple[dict, dict]:
    """Require measured training quality and an unchanged held-out verification."""
    quality = dict(quality)
    training = _calibration_training_quality(workspace, quality.get("config", {}))
    reference_unchanged = _matches_digest(workspace / "reference.npz", spec["reference_sha256"]) and _matches_digest(
        reference_file, spec["verification_reference_sha256"]
    )
    quality["held_out_success"] = bool(quality.get("success", False))
    quality["training_success"] = bool(training.get("success", False))
    quality["success"] = quality["held_out_success"] and quality["training_success"] and reference_unchanged
    return quality, {"training_quality": training, "references_unchanged": reference_unchanged}


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


def _run_agent(cmd: list[str], prompt: str, workspace: Path, env: dict, seconds: int) -> tuple[float, bool, int]:
    """Retain raw events and stop all agent child processes at the trial boundary."""
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
            agent.communicate(prompt, timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _stop_process(agent)
    return time.perf_counter() - start, timed_out, agent.returncode


def run_context(
    workspace: Path, spec: dict, prompt: str, application_command: list[str], *, reference_file: Path | None = None
) -> dict:
    """Run the same isolated Astra context and application lifecycle for every task."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    server = None
    kernel = None
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
            [*application_command, "--live", "--connection-file", str(connection)],
            cwd=workspace,
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
    elif spec["condition"] in {"ipython", "ipython_fixed"}:
        from .ipython_session import IpythonSession  # noqa: PLC0415

        kernel = IpythonSession(
            workspace,
            spec["scenario"],
            spec["variant"],
            reference_file=reference_file,
        )
        args = ["run", "--no-sync", "--project", str(ROOT), "python"]
        corrected_server = None
        if spec["condition"] == "ipython_fixed":
            configured_server = os.environ.get("NEWTON_EVAL_IPYTHON_FIXED_SERVER")
            if not configured_server or not Path(configured_server).is_file():
                raise ValueError(
                    "ipython_fixed requires NEWTON_EVAL_IPYTHON_FIXED_SERVER pointing to the patched server.py"
                )
            corrected_server = Path(configured_server).resolve()
            args += [str(corrected_server)]
        else:
            args += ["-m", "ipython_mcp.server"]
        kernel.start()
        startup = kernel.startup_seconds
        if corrected_server is not None:
            info = json.loads((workspace / "ipython_environment.json").read_text())
            info["corrected_entry_path"] = str(corrected_server)
            info["corrected_entry_sha256"] = digest(corrected_server)
            info["variant"] = "reply_correlation_fix"
            (workspace / "ipython_environment.json").write_text(json.dumps(info, indent=2) + "\n")
        cmd += [
            "-c",
            'mcp_servers.ipython.command="uv"',
            "-c",
            "mcp_servers.ipython.args=" + json.dumps(args),
            "-c",
            f'mcp_servers.ipython.cwd="{ROOT}"',
            "-c",
            "mcp_servers.ipython.tool_timeout_sec=300",
            "-c",
            "mcp_servers.ipython.startup_timeout_sec=30",
        ]
    cmd.append("-")
    try:
        elapsed, timed_out, agent_exit = _run_agent(cmd, prompt, workspace, env, spec["budget_seconds"])
    finally:
        if server is not None:
            _stop_process(server)
            server_log.close()
        if kernel is not None:
            kernel.close()
    events, malformed = [], []
    for line in (workspace / "agent.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            malformed.append(line)
    items = [e.get("item", {}) for e in events if e.get("type") == "item.completed"]
    tool_items = [i for i in items if i.get("type") in ("command_execution", "mcp_tool_call", "tool_call")]
    external_sources_unchanged = True
    if kernel is not None:
        external = json.loads((workspace / "ipython_environment.json").read_text())
        external_sources_unchanged = digest(Path(external["server_path"])) == external["server_sha256"]
        if "corrected_entry_path" in external:
            external_sources_unchanged &= (
                digest(Path(external["corrected_entry_path"])) == external["corrected_entry_sha256"]
            )
    return {
        "agent_elapsed_seconds": elapsed,
        "live_startup_seconds": startup,
        "application_startup_seconds": startup,
        "startup_inclusive_seconds": elapsed + startup,
        "usage": _usage(events),
        "exit_code": agent_exit,
        "timed_out": timed_out,
        "tool_items": len(tool_items),
        "mcp_tool_items": sum(i.get("type") == "mcp_tool_call" for i in tool_items),
        "external_sources_unchanged": external_sources_unchanged,
        **_action_failures(items),
        "malformed_event_count": len(malformed),
    }


def _matches_digest(path: Path, expected: str | None) -> bool:
    """Treat missing or unreadable immutable inputs as failed commitments."""
    try:
        return path.is_file() and digest(path) == expected
    except OSError:
        return False


def _trial_integrity(workspace: Path, prepared: dict) -> dict:
    """Check frozen numerical sources and trial inputs before independent verification."""
    spec = prepared["spec"]
    try:
        sources_unchanged = source_hashes() == spec["source_hashes"]
    except OSError:
        sources_unchanged = False
    references_unchanged = True
    if spec["scenario"] == "panda_calibration":
        reference = prepared.get("verification_reference_file")
        references_unchanged = bool(
            reference
            and _matches_digest(workspace / "reference.npz", spec["reference_sha256"])
            and _matches_digest(Path(reference), spec["verification_reference_sha256"])
        )
    return {
        "shared_sources_unchanged": sources_unchanged,
        "task_unchanged": _matches_digest(workspace / "task.json", prepared.get("task_sha256")),
        "integrity_manifest_unchanged": _matches_digest(
            workspace / "integrity-manifest.json", spec.get("integrity_manifest_sha256")
        ),
        "references_unchanged": references_unchanged,
    }


def _verify_trial(workspace: Path, spec: dict, reference: str | None, env: dict) -> tuple[dict, int]:
    """Retain failed or timed-out physical verification and its available process output."""
    output = workspace / "verification" / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            _command(
                spec["scenario"],
                spec["variant"],
                workspace,
                output=output,
                reference_file=None if reference is None else Path(reference),
            ),
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        parts = [
            part.decode(errors="replace") if isinstance(part, bytes) else part or ""
            for part in (error.stdout, error.stderr)
        ]
        (workspace / "verification.log").write_text("".join(parts))
        return {"success": False, "verification_timed_out": True}, 1
    except OSError as error:
        (workspace / "verification.log").write_text(str(error))
        return {"success": False, "verification_error": str(error)}, 0
    (workspace / "verification.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        return {"success": False, "verification_exit": result.returncode}, 1
    try:
        quality = json.loads(output.read_text())
        if not isinstance(quality, dict) or not isinstance(quality.get("success"), bool):
            raise ValueError("Verifier metrics must be an object with a boolean success field")
        return quality, 1
    except (OSError, ValueError) as error:
        return {"success": False, "verification_error": str(error)}, 1


def run_trial(workspace: Path, prepared: dict) -> dict:
    """Launch one independent context, retain all events, then verify physics."""
    spec = prepared["spec"]
    context = run_context(
        workspace,
        spec,
        prepared["prompt"],
        _command(spec["scenario"], spec["variant"], workspace),
        reference_file=workspace / "reference.npz" if spec["scenario"] == "panda_calibration" else None,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    measurements = _trial_measurements(workspace)
    verification_reference = prepared.get("verification_reference_file")
    integrity = _trial_integrity(workspace, prepared)
    if all(integrity.values()) and context["external_sources_unchanged"]:
        quality, verification_starts = _verify_trial(workspace, spec, verification_reference, env)
        after = _trial_integrity(workspace, prepared)
        integrity = {key: unchanged and after[key] for key, unchanged in integrity.items()}
    else:
        quality = {"success": False, "reason": "Integrity mismatch; fresh verification not executed"}
        verification_starts = 0
        (workspace / "verification.log").write_text(quality["reason"] + "\n")
    calibration_details = {}
    if spec["scenario"] == "panda_calibration":
        quality, calibration_details = _calibration_verification(workspace, spec, quality, Path(verification_reference))
    physics_success = bool(quality.get("success"))
    within_candidate_budget = measurements["candidate_rollouts"] <= 12
    study_success = bool(
        physics_success
        and all(integrity.values())
        and context["external_sources_unchanged"]
        and not context["timed_out"]
        and context["exit_code"] == 0
        and within_candidate_budget
    )
    quality["physics_success"] = physics_success
    quality["success"] = study_success
    summary = {
        "scenario": spec["scenario"],
        "variant": spec["variant"],
        "condition": spec["condition"],
        "phase": spec["phase"],
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        **context,
        **measurements,
        "candidate_rollout_definition": "Logged completed rollouts; partial or interrupted attempts are retained in raw events but are not counted here.",
        "verification_process_starts": verification_starts,
        "within_candidate_budget": within_candidate_budget,
        "physics_success": physics_success,
        "study_success": study_success,
        "quality": quality,
        **calibration_details,
        **integrity,
        "raw_events": "agent.jsonl",
        "verification_metrics": "verification/metrics.json",
        "task_source_hashes": spec["source_hashes"],
    }
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Prepare or explicitly launch one evaluation trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SPECS, required=True)
    parser.add_argument("--condition", choices=("live", "restart", "ipython", "ipython_fixed"), required=True)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--verification-reference", type=Path)
    args = parser.parse_args()
    prepared = prepare(
        args.workspace.resolve(),
        args.scenario,
        args.condition,
        args.variant,
        args.seconds,
        phase=args.phase,
        reference_file=args.reference,
        verification_reference_file=args.verification_reference,
    )
    if args.run:
        print(json.dumps(run_trial(args.workspace.resolve(), prepared), indent=2))
    else:
        print(f"Prepared {args.workspace}; no evaluation agent launched.")


if __name__ == "__main__":
    main()
