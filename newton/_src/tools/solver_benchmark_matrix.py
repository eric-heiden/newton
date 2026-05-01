# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run lightweight solver benchmark scenarios and export dashboard artifacts."""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import socket
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils.benchmark.configs import make_benchmark_configs
from newton._src.solvers.kamino._src.utils.benchmark.problems import make_benchmark_problems
from newton._src.solvers.kamino._src.utils.benchmark.runner import make_benchmark_simulator
from newton._src.solvers.kamino._src.utils.sim import Simulator


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_solver_benchmark_results_dir() -> Path:
    """Return the default directory used for solver benchmark artifacts."""
    configured = os.environ.get("NEWTON_SOLVER_BENCHMARK_RESULTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (_repo_root() / "benchmarks" / "results").resolve()


def default_solver_benchmark_runs_dir() -> Path:
    """Return the default directory used for individual benchmark runs."""
    return default_solver_benchmark_results_dir() / "runs"


def default_solver_benchmark_index_path() -> Path:
    """Return the aggregated solver benchmark index path."""
    return default_solver_benchmark_results_dir() / "index.json"


def _solver_display_name(solver_id: str) -> str:
    mapping = {
        "kamino": "SolverKamino",
        "xpbd": "SolverXPBD",
        "vbd": "SolverVBD",
        "semi_implicit": "SolverSemiImplicit",
        "featherstone": "SolverFeatherstone",
        "mujoco": "SolverMuJoCo",
        "style3d": "SolverStyle3D",
    }
    return mapping.get(solver_id, solver_id)


def _make_args_with_defaults(
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    parser = newton.examples.create_parser()
    if configure_parser is not None:
        configure_parser(parser)
    args = newton.examples.default_args(parser)
    args.viewer = "null"
    args.num_frames = 1
    return args


@dataclass(frozen=True)
class SolverCoverage:
    solver_id: str
    status: str
    reason: str


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    capability: str
    description: str
    module_name: str
    solver_ids: tuple[str, ...]
    coverage: tuple[SolverCoverage, ...]
    build_example: Callable[[str], Any]


class _DirectSolverBenchmarkExample:
    def __init__(
        self,
        model: newton.Model,
        solver: Any,
        contacts: newton.Contacts | None,
        step_dt: float,
        substeps: int = 1,
    ):
        self.model = model
        self.solver = solver
        self.contacts = contacts
        self.step_dt = step_dt
        self.substeps = substeps
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.control = model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def step(self) -> None:
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.step_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0


class _ManagedBenchmarkExample:
    def __init__(self, simulator: Any):
        self._simulator = simulator
        self.viewer = getattr(simulator, "viewer", None)
        self.model = simulator.model if hasattr(simulator, "model") else simulator.sim.model

    def step(self) -> None:
        self._simulator.step_once()

    def close(self) -> None:
        if self.viewer is not None and hasattr(self.viewer, "close"):
            self.viewer.close()


def _build_problem_benchmark_example(problem_name: str, num_worlds: int) -> Any:
    problem, control, camera = make_benchmark_problems([problem_name], num_worlds=num_worlds)[problem_name]
    benchmark_config = make_benchmark_configs()["Default"]
    simulator = make_benchmark_simulator(
        problem=problem,
        configs=Simulator.Config(dt=0.001, solver=benchmark_config),
        control=control,
        camera=camera,
        device=wp.get_device().alias,
        use_cuda_graph=False,
        max_steps=1,
        viewer=False,
    )
    return _ManagedBenchmarkExample(simulator)


def _build_gravity_capsule_example(solver_id: str) -> Any:
    builder = newton.ModelBuilder(gravity=-9.81)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_capsule(body, radius=0.1, half_height=0.2)
    builder.add_ground_plane()
    model = builder.finalize()

    contacts = model.contacts()
    if solver_id == "xpbd":
        solver = newton.solvers.SolverXPBD(model)
    elif solver_id == "semi_implicit":
        solver = newton.solvers.SolverSemiImplicit(model)
    elif solver_id == "featherstone":
        solver = newton.solvers.SolverFeatherstone(model)
    elif solver_id == "mujoco":
        solver = newton.solvers.SolverMuJoCo(model)
        contacts = None
    else:
        raise ValueError(f"Unsupported solver for gravity_capsule: {solver_id}")

    return _DirectSolverBenchmarkExample(model=model, solver=solver, contacts=contacts, step_dt=1.0 / 60.0)


def _build_pendulum_example(solver_id: str) -> Any:
    builder = newton.ModelBuilder()
    link_0 = builder.add_link()
    builder.add_shape_box(link_0, hx=0.5, hy=0.1, hz=0.1)
    link_1 = builder.add_link()
    builder.add_shape_box(link_1, hx=0.5, hy=0.1, hz=0.1)

    joint_0 = builder.add_joint_revolute(
        parent=-1,
        child=link_0,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    joint_1 = builder.add_joint_revolute(
        parent=link_0,
        child=link_1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint_0, joint_1], label="benchmark_pendulum")
    model = builder.finalize()

    if solver_id == "xpbd":
        solver = newton.solvers.SolverXPBD(model)
        contacts = model.contacts()
    elif solver_id == "semi_implicit":
        solver = newton.solvers.SolverSemiImplicit(model)
        contacts = model.contacts()
    elif solver_id == "featherstone":
        solver = newton.solvers.SolverFeatherstone(model)
        contacts = None
    elif solver_id == "mujoco":
        solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
        contacts = None
    else:
        raise ValueError(f"Unsupported solver for pendulum: {solver_id}")

    return _DirectSolverBenchmarkExample(model=model, solver=solver, contacts=contacts, step_dt=1.0 / 120.0)


def _build_basic_shapes_example(solver_id: str) -> Any:
    module = importlib.import_module("newton.examples.basic.example_basic_shapes")
    args = _make_args_with_defaults(
        lambda parser: parser.add_argument("--solver", choices=["xpbd", "vbd"], default="xpbd")
    )
    args.solver = solver_id
    return module.Example(newton.viewer.ViewerNull(num_frames=1), args)


def _build_basic_joints_example(solver_id: str) -> Any:
    module = importlib.import_module("newton.examples.basic.example_basic_joints")
    args = _make_args_with_defaults(
        lambda parser: parser.add_argument("--solver", choices=["xpbd", "vbd"], default="xpbd")
    )
    args.solver = solver_id
    return module.Example(newton.viewer.ViewerNull(num_frames=1), args)


def _build_cloth_hanging_example(solver_id: str) -> Any:
    module = importlib.import_module("newton.examples.cloth.example_cloth_hanging")
    return module.Example(
        newton.viewer.ViewerNull(num_frames=1),
        solver_type=solver_id,
        width=12,
        height=8,
    )


def _build_fourbar_loop_example(solver_id: str) -> Any:
    if solver_id != "kamino":
        raise ValueError(f"Unsupported solver for fourbar_loop: {solver_id}")
    return _build_problem_benchmark_example("fourbar", num_worlds=64)


def _build_dr_legs_loop_example(solver_id: str) -> Any:
    if solver_id != "kamino":
        raise ValueError(f"Unsupported solver for dr_legs_loop: {solver_id}")
    return _build_problem_benchmark_example("dr_legs", num_worlds=32)


def _build_humanoid_loop_heavy_example(solver_id: str) -> Any:
    if solver_id != "mujoco":
        raise ValueError(f"Unsupported solver for humanoid_loop_heavy: {solver_id}")
    return _build_problem_benchmark_example("humanoid", num_worlds=16)


def _build_allegro_hand_example(solver_id: str) -> Any:
    if solver_id != "mujoco":
        raise ValueError(f"Unsupported solver for allegro_hand: {solver_id}")
    return _build_problem_benchmark_example("allegro_hand", num_worlds=8)


def _build_heterogeneous_worlds_example(solver_id: str) -> Any:
    if solver_id != "kamino":
        raise ValueError(f"Unsupported solver for heterogeneous_worlds: {solver_id}")
    return _build_problem_benchmark_example("heterogeneous", num_worlds=1)


def _build_basic_heightfield_example(solver_id: str) -> Any:
    module = importlib.import_module("newton.examples.basic.example_basic_heightfield")
    args = _make_args_with_defaults(
        lambda parser: parser.add_argument("--solver", choices=["xpbd", "mujoco"], default="xpbd")
    )
    args.solver = solver_id
    return module.Example(newton.viewer.ViewerNull(num_frames=1), args)


SCENARIOS: dict[str, ScenarioDefinition] = {
    "gravity_capsule": ScenarioDefinition(
        name="gravity_capsule",
        capability="rigid_body_dynamics",
        description="Single rigid capsule dropped onto a plane across core rigid-body solvers.",
        module_name="newton._src.tools.solver_benchmark_matrix",
        solver_ids=("xpbd", "semi_implicit", "featherstone", "mujoco"),
        coverage=(
            SolverCoverage("xpbd", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage("semi_implicit", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage("featherstone", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage("mujoco", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage(
                "vbd",
                "harness_gap",
                "The solver supports rigid bodies, but this parity slice is not wired into the matrix yet.",
            ),
            SolverCoverage(
                "style3d", "unsupported", "Style3D is a cloth solver and does not support rigid-body dynamics."
            ),
        ),
        build_example=_build_gravity_capsule_example,
    ),
    "pendulum": ScenarioDefinition(
        name="pendulum",
        capability="joint_chains",
        description="Double-pendulum articulation parity across maximal-coordinate and generalized-coordinate solvers.",
        module_name="newton._src.tools.solver_benchmark_matrix",
        solver_ids=("xpbd", "semi_implicit", "featherstone", "mujoco"),
        coverage=(
            SolverCoverage("xpbd", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage("semi_implicit", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage("featherstone", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage("mujoco", "benchmarked", "Direct builder-backed benchmark implemented."),
            SolverCoverage(
                "vbd",
                "harness_gap",
                "VBD joint support exists, but this articulation parity slice is not wired into the matrix yet.",
            ),
            SolverCoverage("style3d", "unsupported", "Style3D does not support articulated rigid-body joint chains."),
        ),
        build_example=_build_pendulum_example,
    ),
    "basic_shapes": ScenarioDefinition(
        name="basic_shapes",
        capability="rigid_contacts",
        description="Rigid collision shapes resting on the ground.",
        module_name="newton.examples.basic.example_basic_shapes",
        solver_ids=("xpbd", "vbd"),
        coverage=(
            SolverCoverage("xpbd", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("vbd", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("semi_implicit", "harness_gap", "The rigid-contact example only exposes XPBD/VBD today."),
            SolverCoverage("featherstone", "harness_gap", "The rigid-contact example only exposes XPBD/VBD today."),
            SolverCoverage("mujoco", "harness_gap", "The rigid-contact example only exposes XPBD/VBD today."),
            SolverCoverage("style3d", "unsupported", "Style3D does not support rigid-body contact benchmarks."),
        ),
        build_example=_build_basic_shapes_example,
    ),
    "basic_joints": ScenarioDefinition(
        name="basic_joints",
        capability="articulations",
        description="Programmatic jointed rigid-body system with articulated motion.",
        module_name="newton.examples.basic.example_basic_joints",
        solver_ids=("xpbd", "vbd"),
        coverage=(
            SolverCoverage("xpbd", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("vbd", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("semi_implicit", "harness_gap", "The example only exposes XPBD/VBD today."),
            SolverCoverage(
                "featherstone",
                "harness_gap",
                "This maximal-coordinate example is not wired into the generic articulation benchmark path.",
            ),
            SolverCoverage(
                "mujoco",
                "harness_gap",
                "This maximal-coordinate example is not wired into the generic articulation benchmark path.",
            ),
            SolverCoverage("style3d", "unsupported", "Style3D does not support articulated rigid-body joints."),
        ),
        build_example=_build_basic_joints_example,
    ),
    "basic_heightfield": ScenarioDefinition(
        name="basic_heightfield",
        capability="terrain_contacts",
        description="Heightfield terrain drop test across the currently wired terrain solvers.",
        module_name="newton.examples.basic.example_basic_heightfield",
        solver_ids=("xpbd", "mujoco"),
        coverage=(
            SolverCoverage("xpbd", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("mujoco", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("semi_implicit", "harness_gap", "The terrain example only exposes XPBD/MuJoCo today."),
            SolverCoverage("featherstone", "harness_gap", "The terrain example only exposes XPBD/MuJoCo today."),
            SolverCoverage("vbd", "harness_gap", "The terrain example only exposes XPBD/MuJoCo today."),
            SolverCoverage("style3d", "unsupported", "Style3D does not support terrain-backed rigid-body contacts."),
        ),
        build_example=_build_basic_heightfield_example,
    ),
    "cloth_hanging": ScenarioDefinition(
        name="cloth_hanging",
        capability="cloth",
        description="Hanging cloth benchmark across supported cloth solvers.",
        module_name="newton.examples.cloth.example_cloth_hanging",
        solver_ids=("semi_implicit", "style3d", "xpbd", "vbd"),
        coverage=(
            SolverCoverage("semi_implicit", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("style3d", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("xpbd", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage("vbd", "benchmarked", "Example-backed benchmark implemented."),
            SolverCoverage(
                "featherstone",
                "harness_gap",
                "Newton uses Featherstone as a helper in mixed examples, not as a standalone cloth benchmark here.",
            ),
            SolverCoverage("mujoco", "unsupported", "MuJoCo does not support Newton cloth simulation."),
        ),
        build_example=_build_cloth_hanging_example,
    ),
    "fourbar_loop": ScenarioDefinition(
        name="fourbar_loop",
        capability="closed_loop_linkage",
        description="Closed-loop four-bar linkage workload using the Kamino benchmark harness.",
        module_name="newton._src.solvers.kamino._src.utils.benchmark.problems",
        solver_ids=("kamino",),
        coverage=(
            SolverCoverage("kamino", "benchmarked", "Kamino benchmark problem wired into the solver dashboard matrix."),
            SolverCoverage(
                "xpbd",
                "harness_gap",
                "The closed-loop linkage parity slice is not yet exposed through Newton's public XPBD benchmark examples.",
            ),
            SolverCoverage(
                "mujoco",
                "harness_gap",
                "The closed-loop linkage parity slice is not yet exposed through Newton's public MuJoCo benchmark examples.",
            ),
            SolverCoverage(
                "style3d", "unsupported", "Style3D does not support rigid closed-loop articulation benchmarks."
            ),
        ),
        build_example=_build_fourbar_loop_example,
    ),
    "dr_legs_loop": ScenarioDefinition(
        name="dr_legs_loop",
        capability="loop_heavy_legged_robot",
        description="DR Legs closed-chain workload using repo-backed Disney Research assets in the Kamino harness.",
        module_name="newton._src.solvers.kamino._src.utils.benchmark.problems",
        solver_ids=("kamino",),
        coverage=(
            SolverCoverage("kamino", "benchmarked", "Repo-backed DR Legs benchmark wired into the Kamino matrix."),
            SolverCoverage(
                "xpbd",
                "harness_gap",
                "Newton does not yet expose a public XPBD benchmark path for the DR Legs asset pack.",
            ),
            SolverCoverage(
                "mujoco",
                "harness_gap",
                "Newton does not yet expose a public MuJoCo benchmark path for the DR Legs asset pack.",
            ),
            SolverCoverage(
                "style3d", "unsupported", "Style3D does not support articulated rigid-body robot benchmarks."
            ),
        ),
        build_example=_build_dr_legs_loop_example,
    ),
    "humanoid_loop_heavy": ScenarioDefinition(
        name="humanoid_loop_heavy",
        capability="loop_heavy_humanoid",
        description="Loop-heavy humanoid workload using the public humanoid plotting example as a benchmark harness.",
        module_name="newton.examples.basic.example_basic_plotting",
        solver_ids=("mujoco",),
        coverage=(
            SolverCoverage(
                "mujoco", "benchmarked", "Public humanoid example benchmarked through the dashboard matrix."
            ),
            SolverCoverage(
                "kamino",
                "harness_gap",
                "A Kamino-native humanoid benchmark path is not yet productized in the shared benchmark harness.",
            ),
            SolverCoverage(
                "xpbd",
                "harness_gap",
                "The public humanoid example does not expose an XPBD-backed benchmark path.",
            ),
            SolverCoverage(
                "style3d", "unsupported", "Style3D does not support articulated humanoid rigid-body benchmarks."
            ),
        ),
        build_example=_build_humanoid_loop_heavy_example,
    ),
    "allegro_hand": ScenarioDefinition(
        name="allegro_hand",
        capability="tendon_hand",
        description="Public Allegro hand workload for tendon-style dexterous manipulation benchmarking.",
        module_name="newton.examples.robot.example_robot_allegro_hand",
        solver_ids=("mujoco",),
        coverage=(
            SolverCoverage("mujoco", "benchmarked", "Public Allegro hand benchmark wired into the dashboard matrix."),
            SolverCoverage(
                "kamino",
                "harness_gap",
                "A Kamino-native tendon-hand benchmark path is not yet benchmark-ready in the shared harness.",
            ),
            SolverCoverage(
                "xpbd",
                "harness_gap",
                "Newton does not yet expose an XPBD-backed Allegro hand benchmark path.",
            ),
            SolverCoverage(
                "style3d", "unsupported", "Style3D does not support rigid-body hand articulation benchmarks."
            ),
        ),
        build_example=_build_allegro_hand_example,
    ),
    "heterogeneous_worlds": ScenarioDefinition(
        name="heterogeneous_worlds",
        capability="heterogeneous_batching",
        description="Heterogeneous-world batching workload that mixes structurally different Kamino worlds in one run.",
        module_name="newton._src.solvers.kamino._src.utils.benchmark.problems",
        solver_ids=("kamino",),
        coverage=(
            SolverCoverage(
                "kamino",
                "benchmarked",
                "Kamino heterogeneous-world batching benchmark wired into the dashboard matrix.",
            ),
            SolverCoverage(
                "xpbd",
                "harness_gap",
                "Newton's public benchmark flow does not yet expose heterogeneous multi-world batching for XPBD.",
            ),
            SolverCoverage(
                "mujoco",
                "harness_gap",
                "Newton's public benchmark flow does not yet expose heterogeneous multi-world batching for MuJoCo.",
            ),
            SolverCoverage(
                "style3d", "unsupported", "Style3D does not support heterogeneous rigid-world batching benchmarks."
            ),
        ),
        build_example=_build_heterogeneous_worlds_example,
    ),
}


def make_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for solver benchmark matrix generation."""
    parser = argparse.ArgumentParser(description="Run solver-comparison benchmark scenarios for dashboard artifacts.")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="Benchmark a specific scenario. Repeat to benchmark multiple scenarios.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Timed simulation steps per solver/scenario pair.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Warmup steps excluded from recorded timings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Warp device to run on, e.g. 'cuda:0' or 'cpu'. Defaults to Warp's preferred device.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_solver_benchmark_results_dir(),
        help="Directory used for run artifacts and the aggregated index.",
    )
    return parser


def _measure_example(example: Any, steps: int, warmup_steps: int) -> list[float]:
    for _ in range(warmup_steps):
        example.step()
        wp.synchronize()

    step_times_ms: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        example.step()
        wp.synchronize()
        step_times_ms.append((time.perf_counter() - start) * 1000.0)
    return step_times_ms


def _model_summary(example: Any) -> dict[str, int]:
    model = example.model
    return {
        "body_count": int(getattr(model, "body_count", 0)),
        "joint_count": int(getattr(model, "joint_count", 0)),
        "joint_dof_count": int(getattr(model, "joint_dof_count", 0)),
        "shape_count": int(getattr(model, "shape_count", 0)),
        "particle_count": int(getattr(model, "particle_count", 0)),
    }


def _result_stats(step_times_ms: Sequence[float]) -> dict[str, float]:
    total_time_ms = float(sum(step_times_ms))
    median_time_ms = float(statistics.median(step_times_ms))
    return {
        "total_time_ms": total_time_ms,
        "mean_step_time_ms": float(statistics.fmean(step_times_ms)),
        "median_step_time_ms": median_time_ms,
        "min_step_time_ms": float(min(step_times_ms)),
        "max_step_time_ms": float(max(step_times_ms)),
        "steps_per_second": float(len(step_times_ms) * 1000.0 / total_time_ms) if total_time_ms > 0.0 else 0.0,
    }


def _coverage_summary(scenario_defs: Sequence[ScenarioDefinition]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for scenario in scenario_defs:
        for coverage in scenario.coverage:
            summary[coverage.status] = summary.get(coverage.status, 0) + 1
    return summary


def run_solver_benchmark_matrix(
    scenario_names: Sequence[str],
    steps: int,
    warmup_steps: int,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the selected solver benchmark matrix and return a JSON-ready artifact."""
    if steps <= 0:
        raise ValueError("--steps must be greater than zero.")
    if warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")

    scenario_defs = [SCENARIOS[name] for name in (scenario_names or tuple(SCENARIOS))]
    run_started_at = _utc_now()

    if device is not None:
        wp.set_device(device)
    selected_device = wp.get_device()

    results: list[dict[str, Any]] = []
    for scenario in scenario_defs:
        for solver_id in scenario.solver_ids:
            example = scenario.build_example(solver_id)
            try:
                step_times_ms = _measure_example(example, steps=steps, warmup_steps=warmup_steps)
                result = {
                    "scenario": scenario.name,
                    "capability": scenario.capability,
                    "description": scenario.description,
                    "example_module": scenario.module_name,
                    "solver_id": solver_id,
                    "solver_name": _solver_display_name(solver_id),
                    "step_times_ms": [round(value, 6) for value in step_times_ms],
                    "model": _model_summary(example),
                }
                result.update(_result_stats(step_times_ms))
                results.append(result)
            finally:
                if hasattr(example, "close"):
                    example.close()
                elif hasattr(example, "viewer") and hasattr(example.viewer, "close"):
                    example.viewer.close()

    return {
        "schema_version": 2,
        "run_id": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": _utc_now(),
        "started_at": run_started_at,
        "device": selected_device.alias,
        "host": socket.gethostname(),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "coverage_summary": _coverage_summary(scenario_defs),
        "scenarios": [
            {
                "name": scenario.name,
                "capability": scenario.capability,
                "description": scenario.description,
                "solvers": list(scenario.solver_ids),
                "example_module": scenario.module_name,
                "coverage": [
                    {
                        "solver_id": coverage.solver_id,
                        "solver_name": _solver_display_name(coverage.solver_id),
                        "status": coverage.status,
                        "reason": coverage.reason,
                    }
                    for coverage in scenario.coverage
                ],
            }
            for scenario in scenario_defs
        ],
        "results": results,
        "notes": [
            "This matrix combines in-tree examples with direct builder-backed scenarios to provide honest solver-vs-capability coverage.",
            "Coverage metadata distinguishes benchmarked paths from harness gaps and unsupported solver/scenario combinations.",
        ],
    }


def _matrix_rows_for_run(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in run_payload.get("results", []):
        grouped.setdefault(str(result["scenario"]), []).append(result)

    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_results in grouped.items():
        ranked = sorted(scenario_results, key=lambda item: (-float(item["steps_per_second"]), str(item["solver_name"])))
        best_rate = float(ranked[0]["steps_per_second"]) if ranked else 0.0
        for rank, result in enumerate(ranked, start=1):
            row = {
                "scenario": scenario_name,
                "capability": result["capability"],
                "solver_name": result["solver_name"],
                "solver_id": result["solver_id"],
                "rank": rank,
                "steps_per_second": result["steps_per_second"],
                "median_step_time_ms": result["median_step_time_ms"],
                "total_time_ms": result["total_time_ms"],
                "relative_to_best": (float(result["steps_per_second"]) / best_rate if best_rate > 0.0 else 0.0),
            }
            rows.append(row)
    return rows


def _coverage_rows_for_run(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_payload in run_payload.get("scenarios", []):
        for coverage in scenario_payload.get("coverage", []):
            rows.append(
                {
                    "scenario": scenario_payload["name"],
                    "capability": scenario_payload["capability"],
                    "solver_id": coverage["solver_id"],
                    "solver_name": coverage["solver_name"],
                    "status": coverage["status"],
                    "reason": coverage["reason"],
                }
            )
    return rows


def build_solver_benchmark_index(run_payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build an aggregated, dashboard-ready index from individual benchmark runs."""
    runs = sorted(run_payloads, key=lambda item: (str(item.get("generated_at")), str(item.get("run_id"))), reverse=True)
    latest_run = runs[0] if runs else None

    scenarios: list[dict[str, Any]] = []
    latest_matrix = _matrix_rows_for_run(latest_run) if latest_run is not None else []
    latest_coverage = _coverage_rows_for_run(latest_run) if latest_run is not None else []
    if latest_run is not None:
        for scenario in latest_run.get("scenarios", []):
            scenario_name = str(scenario["name"])
            scenario_results = [result for result in latest_run["results"] if result["scenario"] == scenario_name]
            scenarios.append(
                {
                    "name": scenario_name,
                    "capability": scenario["capability"],
                    "description": scenario["description"],
                    "coverage": scenario.get("coverage", []),
                    "throughput_plot": [
                        {
                            "solver_name": result["solver_name"],
                            "solver_id": result["solver_id"],
                            "steps_per_second": result["steps_per_second"],
                        }
                        for result in sorted(scenario_results, key=lambda item: item["solver_name"])
                    ],
                    "step_time_plot": [
                        {
                            "solver_name": result["solver_name"],
                            "solver_id": result["solver_id"],
                            "values_ms": result["step_times_ms"],
                        }
                        for result in sorted(scenario_results, key=lambda item: item["solver_name"])
                    ],
                }
            )

    return {
        "schema_version": 2,
        "generated_at": _utc_now(),
        "run_count": len(runs),
        "runs": [
            {
                "run_id": run["run_id"],
                "generated_at": run["generated_at"],
                "device": run["device"],
                "steps": run["steps"],
                "warmup_steps": run["warmup_steps"],
                "result_count": len(run.get("results", [])),
                "coverage_summary": run.get("coverage_summary", {}),
            }
            for run in runs
        ],
        "latest_run": latest_run,
        "latest_matrix": latest_matrix,
        "latest_coverage": latest_coverage,
        "scenarios": scenarios,
    }


def write_solver_benchmark_artifacts(results_dir: Path, run_payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write the individual run artifact and the aggregated index."""
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_path = runs_dir / f"{run_payload['run_id']}.json"
    run_path.write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")

    run_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(runs_dir.glob("*.json"))]
    index_payload = build_solver_benchmark_index(run_payloads)
    index_path = results_dir / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    return run_path, index_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for solver benchmark matrix generation."""
    args = make_argument_parser().parse_args(argv)
    run_payload = run_solver_benchmark_matrix(
        scenario_names=args.scenario or tuple(SCENARIOS),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        device=args.device,
    )
    run_path, index_path = write_solver_benchmark_artifacts(args.results_dir, run_payload)
    print(f"Wrote benchmark run: {run_path}")
    print(f"Wrote benchmark index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
