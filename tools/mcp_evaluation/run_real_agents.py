# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare isolated Astra agents on clean-slate identification from measured robot data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from .real_robot import REAL_SPEC, initial_config
from .recording import digest
from .rollout import camera
from .run_agents import _trial_measurements, run_context, source_hashes, write_task
from .scenarios import ROOT


def command(workspace: Path, reference: Path, output: Path, *, variant: int = 0) -> list[str]:
    """Build the identical fresh candidate or verifier command."""
    return [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(ROOT),
        "python",
        "-m",
        "tools.mcp_evaluation.real_rollout",
        "--variant",
        str(variant),
        "--config",
        str(workspace / "config.json"),
        "--reference",
        str(reference),
        "--output",
        str(output),
    ]


def prepare(
    workspace: Path,
    condition: str,
    variant: int,
    public: Path,
    heldout: Path,
    *,
    seconds: int = 1200,
    phase: str = "development",
) -> dict:
    """Give every condition identical sanitized inputs and unrestricted fitting choices."""
    if condition not in {"live", "restart", "ipython", "ipython_fixed"}:
        raise ValueError("Unknown evaluation condition")
    workspace.mkdir(parents=True, exist_ok=False)
    for name in ("training.npz", "training-regressor.npz", "training-regressor.manifest.json"):
        shutil.copyfile(public / name, workspace / name)
    (workspace / "config.json").write_text(json.dumps(initial_config(variant), indent=2) + "\n")
    geometry = (public / "geometry/panda_geometry.xml").resolve()
    spec = dict(REAL_SPEC) | {
        "scenario": "panda_real",
        "condition": condition,
        "variant": variant,
        "phase": phase,
        "budget_seconds": seconds,
        "candidate_budget": 60,
        "camera": camera("panda_real"),
        "geometry_file": str(geometry),
        "source_hashes": source_hashes(),
        "input_hashes": {
            name: digest(workspace / name)
            for name in ("training.npz", "training-regressor.npz", "training-regressor.manifest.json")
        },
        "geometry_hashes": {
            str(path.relative_to(public)): digest(path)
            for path in sorted((public / "geometry").rglob("*"))
            if path.is_file()
        },
    }
    if condition.startswith("ipython"):
        spec["ipython_server_variant"] = "unmodified_upstream" if condition == "ipython" else "reply_correlation_fix"
    common = f"""You are an independent GPT-6 Astra evaluation agent. Identify a Newton Panda model from REAL measured physical-robot recordings, starting from a clean slate. You have {seconds} seconds and at most 60 complete physical candidate evaluations. No subagents.

Read task.json, config.json and training-regressor.manifest.json. Geometry and kinematics are supplied; all seven moving links begin with identical placeholders: mass 1 kg, COM zero, COM inertia .01*identity kg*m^2, and zero joint losses, bias and armature. These are not authored Panda dynamics. Identify the full physical parameters in config.json, not correction factors around a nominal asset. Geometry: {geometry}.

training.npz contains three physical Panda recordings from the published LIP4RobotInverseDynamics data (DOI 10.5281/zenodo.12516500): measured q, qd, qdd and publisher-filtered/interpolated joint torque tau, timestamps, episode_ids/episode_offsets. This is effective link/joint dynamics identification; individual link parameters need not be uniquely identifiable. Measurements are filtered, and measured joint torque is not a recovered raw motor command. The seven-link model has 98 independent coefficients, including full symmetric link inertias, masses, centers of mass, viscous and Coulomb friction, torque bias and armature. All submitted values must satisfy the broad physical bounds in task.json.

training-regressor.npz provides the same immutable numerical design matrix to every condition: A[3150,98], b[3150], sample_indices and sample_episode_ids. A is constructed from known geometry and measured q/qd/qdd with Newton inverse dynamics, without fitted or nominal inertial parameters. Column and row mappings are in its manifest and real_robot_model.physical_coefficients. Computing these immutable features is common preprocessing outside every agent timer. You must write your own estimation code and choose fitting, physical constraints and regularization. NumPy, SciPy and CVXPY are available equally. Offline algebra and fitting do not count as physical candidates. You may write helper Python files and run them through uv run --no-sync --project {ROOT} python helper.py. No particular fitting algorithm is prescribed.

Success requires BOTH measured torque agreement AND measured short-horizon forward motion agreement, all pooled and per recording. Every physical candidate covers 36 independent 100ms windows, initialized at measured q/qd, driven by interpolated measured joint torque, 1800 Newton steps at dt .002. Read all thresholds in task.json. The final independently published test fold contains 16 further physical recordings with different motions; final verification runs 9600 steps in a fresh process after your context exits. You cannot access that fold, fitted answers, feasibility results or other trials. Do not search for nominal Panda dynamic parameters or alternative data. The only allowed data are this workspace's three supplied input files and sanitized geometry. You may inspect Newton public source and these shared physics files: real_robot.py, real_robot_model.py, real_robot_regressor.py, real_robot_data.py, real_rollout.py and rollout.py in {ROOT / "tools/mcp_evaluation"}. Do not inspect other evaluation runners, tests, reports or artifact directories. Shared sources, geometry, reference files, manifests and task.json are immutable.

Use candidate configuration changes only through scenario.apply_config and the prescribed candidate workflow; do not modify scoring, targets, measured windows, callbacks, or solver algorithm. Do not create additional simulators outside that workflow. The same physical observations and optional images are available in every condition. Complete traces of q/qd, measured reference and torque predictions are written to the NPZ at metrics.trace_path. Preserve every failed attempt and log; do not overwrite candidate traces. Submit the exact final numeric configuration in config.json with a passing completed training measurement. An independent process verifies training and held-out quality and source/data integrity. Finish with a brief account of fitted quality and remaining limitations.
"""
    if condition == "restart":
        interface = f"""
No MCP or persistent simulator is available in this condition. Edit config.json and run a new process for EACH physical candidate:
uv run --no-sync --project {ROOT} python -m tools.mcp_evaluation.real_rollout --config config.json --reference training.npz --output candidate-001/metrics.json
Choose a new output directory per candidate to retain its trace and logs. Each invocation must exit after its single candidate. You may loop over candidates with one fresh process each, and keep ordinary numerical fitting helpers alive, but never retain a simulation model/solver between candidate evaluations. Add --observe for an optional image. Read resulting metrics and traces.
"""
    else:
        interface = """
The identical simulation is already running. Shared application objects include session, model, solver, state, state_next, control, contacts, np and wp. User imports, variables and functions persist. Apply one complete candidate through this workflow (batching loops is allowed within 60 total candidates):
session.scenario.apply_config(candidate_config)
session.dispatch('reset')
session.dispatch('step', {'count': 1800})
session.scenario.metrics()
The reset and step helpers keep measurement callbacks and native state coherent. Do not call scenario.rollout directly. Access any ordinary Python diagnostics and fitting libraries; you are not restricted to predefined MCP entrypoints. You may execute a helper file in the persistent namespace. After a candidate completes, metrics.trace_path identifies its saved full observations. Rebuild uses the same validated configuration by default if recovery is necessary. Submit the final configuration in config.json as well as applying and measuring it.
"""
        if condition == "live":
            interface += """
Use the configured Newton MCP tools execute, describe, observe and rebuild for running simulator access. execute(code=...) accepts ordinary Python and returns the final expression or an explicitly assigned result. Prefer compact diagnostics; large intermediate objects remain in the namespace. Full helper execution can use exec(compile(Path('helper.py').read_text(), 'helper.py', 'exec')) after importing Path. Live aliases refresh between cells. An execution error preserves Python variables but pauses and invalidates simulation access: recovery='inspect' allows diagnosis; recovery='acknowledge' is an explicit assertion that you verified/repaired coherent state. Rebuild clears the workspace and replaces the simulation in the same process. Use observe directly for image content. Tool timeout 300s; an interrupted/timeout mutation has an unknown outcome, so verify status before retrying. Use actual MCP tools, not a shell socket client.
"""
        else:
            interface += f"""
Use the configured actual ipython-mcp server: connect_to_kernel(connection_file={str(workspace / "kernel.json")!r}) once, then execute_code(code=...) for ordinary Python. Do not read/print kernel.json or start another kernel. The final expression is displayed; assigning result alone does not display it. Namespace and imports persist; live aliases refresh before/after each cell. IPython errors preserve the namespace and may have partially changed the scene; verify consistency or rebuild when necessary. session.dispatch('observe', camera_settings) produces the same sensor image; save image_base64 to a PNG and inspect it with the image tool instead of printing the large payload. Use actual IPython MCP tools, not a separate shell kernel client.
"""
            interface += (
                "The unmodified upstream server has an internal 30s execution wait. Keep each call below 30s or split work across cells. A timeout does not stop execution; inspect the outcome before retrying a mutation. The outer tool timeout is 300s.\n"
                if condition == "ipython"
                else "This disclosed sensitivity control uses upstream ipython-mcp with matching execution-reply IDs and a 300s internal wait. The outer tool timeout is 300s. A timeout does not stop execution; inspect outcome before retrying a mutation.\n"
            )
    prompt = common + interface
    write_task(workspace, spec)
    (workspace / "TASK.md").write_text(prompt)
    private_inputs = {
        str(path.resolve()): digest(path)
        for path in (
            heldout,
            heldout.with_name(heldout.stem + "-regressor.npz"),
            heldout.with_name(heldout.stem + "-regressor.manifest.json"),
        )
    }
    return {
        "spec": spec,
        "prompt": prompt,
        "heldout": str(heldout.resolve()),
        "private_input_hashes": private_inputs,
        "public": str(public.resolve()),
        "task_sha256": digest(workspace / "task.json"),
    }


def training_quality(workspace: Path, config: dict) -> dict:
    """Require a completed training measurement of the exact submitted configuration."""
    matches = []
    for path in workspace.rglob("*.jsonl"):
        if (
            path.name not in {"rollouts.jsonl", "live_rollouts.jsonl"}
            or path.relative_to(workspace).parts[0] == "verification"
        ):
            continue
        for index, line in enumerate(path.read_text().splitlines()):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                result.get("scenario") == "panda_real"
                and result.get("config") == config
                and result.get("frames") == 1800
                and result.get("sample_count") == 1800
                and result.get("episodes") == [2, 3, 4]
            ):
                matches.append((path.stat().st_mtime_ns, index, result))
    return (
        max(matches, key=lambda item: item[:2])[2]
        if matches
        else {"success": False, "reason": "No complete training evaluation of submitted configuration"}
    )


def verify(workspace: Path, reference: Path, output: Path, variant: int) -> dict:
    """Retain verifier errors and timeouts as failed outcomes, never lost trials."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command(workspace, reference, output, variant=variant),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output.with_suffix(".log").write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            return {"success": False, "verification_exit": result.returncode}
        try:
            metrics = json.loads(output.read_text())
            if not isinstance(metrics, dict) or not isinstance(metrics.get("success"), bool):
                return {"success": False, "verification_error": "Expected a metrics object with boolean success"}
            return metrics
        except (OSError, json.JSONDecodeError) as error:
            return {"success": False, "verification_error": str(error)}
    except subprocess.TimeoutExpired as error:
        parts = [
            (part.decode(errors="replace") if isinstance(part, bytes) else part or "")
            for part in (error.stdout, error.stderr)
        ]
        output.with_suffix(".log").write_text("".join(parts))
        return {"success": False, "verification_timed_out": True}
    except OSError as error:
        output.with_suffix(".log").write_text(str(error) + "\n")
        return {"success": False, "verification_error": str(error)}


def integrity(workspace: Path, prepared: dict) -> tuple[bool, bool, bool, bool]:
    """Check immutable inputs and executable sources before opening held-out data."""
    spec = prepared["spec"]
    references_unchanged = all(
        (workspace / name).is_file() and digest(workspace / name) == checksum
        for name, checksum in spec["input_hashes"].items()
    ) and all(
        Path(name).is_file() and digest(Path(name)) == checksum
        for name, checksum in prepared["private_input_hashes"].items()
    )
    geometry_unchanged = all(
        (Path(prepared["public"]) / name).is_file() and digest(Path(prepared["public"]) / name) == checksum
        for name, checksum in spec["geometry_hashes"].items()
    )
    sources_unchanged = source_hashes() == spec["source_hashes"]
    task_unchanged = (
        (workspace / "task.json").is_file()
        and digest(workspace / "task.json") == prepared["task_sha256"]
        and (workspace / "integrity-manifest.json").is_file()
        and digest(workspace / "integrity-manifest.json") == spec["integrity_manifest_sha256"]
    )
    return references_unchanged, geometry_unchanged, sources_unchanged, task_unchanged


def run_trial(workspace: Path, prepared: dict) -> dict:
    """Time one fresh agent context and independently verify its submitted physics."""
    spec = prepared["spec"]
    previous_geometry = os.environ.get("NEWTON_EVAL_REAL_GEOMETRY")
    os.environ["NEWTON_EVAL_REAL_GEOMETRY"] = spec["geometry_file"]
    verification_process_starts = 0
    try:
        context = run_context(
            workspace,
            spec,
            prepared["prompt"],
            command(workspace, workspace / "training.npz", workspace / "metrics.json", variant=spec["variant"]),
            reference_file=workspace / "training.npz",
        )
        measurements = _trial_measurements(workspace)
        references_unchanged, geometry_unchanged, sources_unchanged, task_unchanged = integrity(workspace, prepared)
        if all(
            (
                references_unchanged,
                geometry_unchanged,
                sources_unchanged,
                task_unchanged,
                context["external_sources_unchanged"],
            )
        ):
            verification_process_starts += 1
            fresh_training = verify(
                workspace, workspace / "training.npz", workspace / "verification/training/metrics.json", spec["variant"]
            )
            verification_process_starts += 1
            quality = verify(
                workspace, Path(prepared["heldout"]), workspace / "verification/metrics.json", spec["variant"]
            )
        else:
            fresh_training = {"success": False, "reason": "Integrity mismatch; fresh verification not executed"}
            quality = dict(fresh_training)
    finally:
        if previous_geometry is None:
            os.environ.pop("NEWTON_EVAL_REAL_GEOMETRY", None)
        else:
            os.environ["NEWTON_EVAL_REAL_GEOMETRY"] = previous_geometry
    try:
        submitted = json.loads((workspace / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        submitted = {}
    training = training_quality(workspace, submitted)
    before_verification_integrity = (references_unchanged, geometry_unchanged, sources_unchanged, task_unchanged)
    after_verification_integrity = integrity(workspace, prepared)
    references_unchanged, geometry_unchanged, sources_unchanged, task_unchanged = tuple(
        before and after
        for before, after in zip(before_verification_integrity, after_verification_integrity, strict=True)
    )
    quality["held_out_success"] = bool(quality.get("success"))
    quality["training_success"] = bool(training.get("success") and fresh_training.get("success"))
    quality["physics_success"] = quality["held_out_success"] and quality["training_success"]
    quality["success"] = bool(
        quality["held_out_success"]
        and quality["training_success"]
        and references_unchanged
        and geometry_unchanged
        and sources_unchanged
        and task_unchanged
        and context["external_sources_unchanged"]
        and not context["timed_out"]
        and context["exit_code"] == 0
        and measurements["candidate_rollouts"] <= spec["candidate_budget"]
    )
    summary = {
        "scenario": "panda_real",
        "variant": spec["variant"],
        "condition": spec["condition"],
        "phase": spec["phase"],
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        **context,
        **measurements,
        "within_candidate_budget": measurements["candidate_rollouts"] <= spec["candidate_budget"],
        "verification_process_starts": verification_process_starts,
        "quality": quality,
        "study_success": quality["success"],
        "training_quality": training,
        "fresh_training_quality": fresh_training,
        "references_unchanged": references_unchanged,
        "geometry_unchanged": geometry_unchanged,
        "shared_sources_unchanged": sources_unchanged,
        "task_unchanged": task_unchanged,
        "task_source_hashes": spec["source_hashes"],
        "raw_events": "agent.jsonl",
        "verification_metrics": "verification/metrics.json",
    }
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Prepare only by default; launch a measured independent agent with --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--condition", choices=("live", "restart", "ipython", "ipython_fixed"), required=True)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    prepared = prepare(
        workspace,
        args.condition,
        args.variant,
        args.public.resolve(),
        args.heldout.resolve(),
        seconds=args.seconds,
        phase=args.phase,
    )
    if args.run:
        print(json.dumps(run_trial(workspace, prepared), indent=2))
    else:
        print(f"Prepared {workspace}; no evaluation agent launched.")


if __name__ == "__main__":
    main()
