#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import html
import json
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "artifacts" / "solver_mujoco_determinism"
DEFAULT_REPORT_PATH = DEFAULT_RESULTS_DIR / "solver_mujoco_determinism_report.html"

WARP_MODE_NOT_GUARANTEED = "not_guaranteed"
WARP_MODE_GPU_TO_GPU = "gpu_to_gpu"


@dataclass(frozen=True)
class ModeConfig:
    name: str
    warp_deterministic: str
    pipeline_deterministic: bool
    description: str


@dataclass(frozen=True)
class SolverConfig:
    solver: str = "newton"
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    iterations: int = 40
    ls_iterations: int = 20
    njmax: int = 4000
    nconmax: int = 4000
    ls_parallel: bool = False


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    family: str
    description: str
    broad_phase: str
    frames: int
    substeps: int
    seed: int
    rigid_contact_max: int
    solver: SolverConfig
    random_actuation: bool = False
    world_count: int = 1


MODES: dict[str, ModeConfig] = {
    "baseline": ModeConfig(
        name="baseline",
        warp_deterministic=WARP_MODE_NOT_GUARANTEED,
        pipeline_deterministic=False,
        description="Warp global determinism off, Newton collision sorting off.",
    ),
    "pipeline": ModeConfig(
        name="pipeline",
        warp_deterministic=WARP_MODE_NOT_GUARANTEED,
        pipeline_deterministic=True,
        description="Warp global determinism off, Newton collision sorting on.",
    ),
    "global": ModeConfig(
        name="global",
        warp_deterministic=WARP_MODE_GPU_TO_GPU,
        pipeline_deterministic=False,
        description="Warp global deterministic lowering on, Newton collision sorting off.",
    ),
    "combined": ModeConfig(
        name="combined",
        warp_deterministic=WARP_MODE_GPU_TO_GPU,
        pipeline_deterministic=True,
        description="Warp global deterministic lowering on, Newton collision sorting on.",
    ),
}


SCENARIOS: dict[str, ScenarioConfig] = {
    "mixed_shapes_nxn": ScenarioConfig(
        name="mixed_shapes_nxn",
        family="mixed-primitive rigid stack",
        description="4x4x4 primitive-only stack (sphere/box/capsule/cylinder/ellipsoid) over a plane using NXN broad phase.",
        broad_phase="nxn",
        frames=60,
        substeps=4,
        seed=42,
        rigid_contact_max=50000,
        solver=SolverConfig(njmax=3000, nconmax=3000, iterations=50, ls_iterations=25),
    ),
    "mixed_shapes_sap": ScenarioConfig(
        name="mixed_shapes_sap",
        family="mixed-primitive rigid stack",
        description="4x4x4 primitive-only stack (sphere/box/capsule/cylinder/ellipsoid) over a plane using SAP broad phase.",
        broad_phase="sap",
        frames=60,
        substeps=4,
        seed=42,
        rigid_contact_max=50000,
        solver=SolverConfig(njmax=3000, nconmax=3000, iterations=50, ls_iterations=25),
    ),
    "box_crowd_sap": ScenarioConfig(
        name="box_crowd_sap",
        family="crowd stress",
        description="100 free boxes with jittered orientations dropping onto a plane using SAP broad phase.",
        broad_phase="sap",
        frames=80,
        substeps=4,
        seed=123,
        rigid_contact_max=30000,
        solver=SolverConfig(njmax=5000, nconmax=5000, iterations=60, ls_iterations=30),
    ),
    "box_crowd_nxn": ScenarioConfig(
        name="box_crowd_nxn",
        family="crowd stress",
        description="100 free boxes with jittered orientations dropping onto a plane using NXN broad phase.",
        broad_phase="nxn",
        frames=80,
        substeps=4,
        seed=123,
        rigid_contact_max=30000,
        solver=SolverConfig(njmax=5000, nconmax=5000, iterations=60, ls_iterations=30),
    ),
}


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _hash_numpy_arrays(pairs: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.blake2b(digest_size=32)
    for name, array in pairs:
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _hash_state(state: Any) -> str:
    arrays: list[tuple[str, np.ndarray]] = []
    for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
        value = getattr(state, name, None)
        if value is not None:
            arrays.append((name, value.numpy()))
    return _hash_numpy_arrays(arrays)


def _hash_contacts(contacts: Any) -> str:
    count = int(contacts.rigid_contact_count.numpy()[0])
    arrays: list[tuple[str, np.ndarray]] = [("rigid_contact_count", np.asarray([count], dtype=np.int64))]
    if count > 0:
        for name in (
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_point0",
            "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_offset0",
            "rigid_contact_offset1",
            "rigid_contact_margin0",
            "rigid_contact_margin1",
        ):
            value = getattr(contacts, name, None)
            if value is not None:
                arrays.append((name, value.numpy()[:count]))
    return _hash_numpy_arrays(arrays)


def _maxrss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _to_megabytes(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / (1024.0 * 1024.0)


def _apply_control(wp: Any, control: Any, model: Any, step_index: int, seed: int) -> None:
    if getattr(control, "joint_target_pos", None) is None or model.joint_dof_count == 0:
        return
    idx = np.arange(model.joint_dof_count, dtype=np.float32)
    values = np.sin(0.07 * step_index + 0.13 * idx + 0.01 * seed).astype(np.float32)
    wp.copy(control.joint_target_pos, wp.array(values, dtype=wp.float32, device=model.device))


def _build_convex_icosahedron_mesh(newton: Any) -> Any:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    radius = 0.35
    base_vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    base_vertices = base_vertices / np.linalg.norm(base_vertices, axis=1, keepdims=True) * radius
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )
    vertices = []
    normals = []
    indices = []
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = base_vertices[face[0]], base_vertices[face[1]], base_vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)
        vertices.extend([v0, v1, v2])
        normals.extend([normal, normal, normal])
        base = face_idx * 3
        indices.extend([base, base + 1, base + 2])
    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float32),
    )


def _build_cube_mesh(newton: Any) -> Any:
    hs = 0.3
    verts = np.array(
        [
            [-hs, -hs, -hs],
            [hs, -hs, -hs],
            [hs, hs, -hs],
            [-hs, hs, -hs],
            [-hs, -hs, hs],
            [hs, -hs, hs],
            [hs, hs, hs],
            [-hs, hs, hs],
        ],
        dtype=np.float32,
    )
    tris = np.array(
        [0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
        dtype=np.int32,
    )
    mesh = newton.Mesh(verts, tris)
    mesh.build_sdf(max_resolution=64)
    return mesh


def _build_mixed_shapes_model(newton: Any, wp: Any, device: str, scenario: ScenarioConfig):
    rng = np.random.default_rng(scenario.seed)
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 2.0e4
    builder.default_shape_cfg.kd = 4.0e2
    builder.default_shape_cfg.kf = 1.0e2
    builder.default_shape_cfg.mu = 0.8
    builder.add_ground_plane()

    shape_types = ["sphere", "box", "capsule", "cylinder", "ellipsoid"]
    shape_index = 0
    grid_offset = np.array([-3.0, -3.0, 0.7], dtype=np.float32)

    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                pos = grid_offset + np.array(
                    [
                        ix * 1.15 + (rng.random() - 0.5) * 0.18,
                        iy * 1.15 + (rng.random() - 0.5) * 0.18,
                        iz * 1.05 + (rng.random() - 0.5) * 0.18,
                    ],
                    dtype=np.float32,
                )
                body = builder.add_body(xform=wp.transform(p=wp.vec3(*pos.tolist()), q=wp.quat_identity()))
                shape_type = shape_types[shape_index % len(shape_types)]
                shape_index += 1
                if shape_type == "sphere":
                    builder.add_shape_sphere(body, radius=0.28)
                elif shape_type == "box":
                    builder.add_shape_box(body, hx=0.28, hy=0.28, hz=0.28)
                elif shape_type == "capsule":
                    builder.add_shape_capsule(body, radius=0.18, half_height=0.35)
                elif shape_type == "cylinder":
                    builder.add_shape_cylinder(body, radius=0.22, half_height=0.32)
                elif shape_type == "ellipsoid":
                    builder.add_shape_ellipsoid(body, rx=0.33, ry=0.24, rz=0.19)

    return builder.finalize(device=device)


def _build_box_crowd_model(newton: Any, wp: Any, device: str, scenario: ScenarioConfig):
    rng = np.random.default_rng(scenario.seed)
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 2.5e4
    builder.default_shape_cfg.kd = 6.0e2
    builder.default_shape_cfg.kf = 1.0e2
    builder.default_shape_cfg.mu = 0.85
    builder.add_ground_plane()

    count_x = 10
    count_y = 10
    base_height = 0.25
    for ix in range(count_x):
        for iy in range(count_y):
            height_layer = (ix + iy) % 3
            z = base_height + height_layer * 0.52
            pos = np.array(
                [
                    -4.2 + ix * 0.92 + (rng.random() - 0.5) * 0.04,
                    -4.2 + iy * 0.92 + (rng.random() - 0.5) * 0.04,
                    z,
                ],
                dtype=np.float32,
            )
            axis = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
            axis /= np.linalg.norm(axis)
            angle = float(rng.uniform(-0.35, 0.35))
            quat = wp.quat_from_axis_angle(wp.vec3(*axis.tolist()), angle)
            body = builder.add_body(xform=wp.transform(p=wp.vec3(*pos.tolist()), q=quat))
            hx = float(rng.uniform(0.16, 0.22))
            hy = float(rng.uniform(0.16, 0.22))
            hz = float(rng.uniform(0.16, 0.22))
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)

    return builder.finalize(device=device)


def _build_ant_model(device: str, scenario: ScenarioConfig):
    sys.path.insert(0, str(REPO_ROOT))
    from asv.benchmarks.benchmark_mujoco import Example  # noqa: PLC0415

    builder = Example.create_model_builder(
        robot="ant",
        world_count=scenario.world_count,
        environment="None",
        randomize=True,
        seed=scenario.seed,
    )
    return builder.finalize(device=device)


def _build_model(newton: Any, wp: Any, device: str, scenario: ScenarioConfig):
    if scenario.name.startswith("mixed_shapes"):
        return _build_mixed_shapes_model(newton, wp, device, scenario)
    if scenario.name.startswith("box_crowd_"):
        return _build_box_crowd_model(newton, wp, device, scenario)
    raise ValueError(f"Unsupported scenario: {scenario.name}")


def _make_solver(newton: Any, model: Any, scenario: ScenarioConfig):
    cfg = scenario.solver
    return newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        use_mujoco_contacts=False,
        solver=cfg.solver,
        integrator=cfg.integrator,
        cone=cfg.cone,
        iterations=cfg.iterations,
        ls_iterations=cfg.ls_iterations,
        njmax=cfg.njmax,
        nconmax=cfg.nconmax,
        ls_parallel=cfg.ls_parallel,
    )


def _warm_up(wp: Any, model: Any, solver: Any, pipeline: Any, control: Any, state_0: Any, state_1: Any, contacts: Any, scenario: ScenarioConfig):
    for warmup_step in range(3):
        if scenario.random_actuation:
            _apply_control(wp, control, model, warmup_step, scenario.seed)
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0 / scenario.substeps)
        state_0, state_1 = state_1, state_0
    wp.synchronize_device(model.device)
    return state_0, state_1


def run_case(case_args: argparse.Namespace) -> dict[str, Any]:
    scenario = SCENARIOS[case_args.scenario]
    mode = MODES[case_args.mode]

    import warp as wp  # noqa: PLC0415

    wp.config.enable_backward = False
    wp.config.quiet = True
    wp.config.deterministic = mode.warp_deterministic

    import newton  # noqa: PLC0415
    import mujoco_warp  # noqa: PLC0415,F401

    device = case_args.device
    start_setup = time.perf_counter()
    with wp.ScopedDevice(device):
        model = _build_model(newton, wp, device, scenario)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase=scenario.broad_phase,
            deterministic=mode.pipeline_deterministic,
            reduce_contacts=True,
            rigid_contact_max=scenario.rigid_contact_max,
        )
        contacts = pipeline.contacts()
        solver = _make_solver(newton, model, scenario)
        control = model.control()
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        state_0, state_1 = _warm_up(wp, model, solver, pipeline, control, state_0, state_1, contacts, scenario)
    wp.synchronize_device(device)
    setup_seconds = time.perf_counter() - start_setup

    state_hashes: list[str] = []
    contact_hashes: list[str] = []
    contact_counts: list[int] = []
    step_times_ms: list[float] = []

    total_steps = scenario.frames * scenario.substeps
    step_index = 0
    with wp.ScopedDevice(device):
        for _frame in range(scenario.frames):
            for _substep in range(scenario.substeps):
                if scenario.random_actuation:
                    _apply_control(wp, control, model, step_index, scenario.seed)
                state_0.clear_forces()
                t0 = time.perf_counter()
                pipeline.collide(state_0, contacts)
                contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
                contact_hashes.append(_hash_contacts(contacts))
                solver.step(state_0, state_1, control, contacts, 1.0 / 240.0 / scenario.substeps)
                state_0, state_1 = state_1, state_0
                wp.synchronize_device(device)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)
                state_hashes.append(_hash_state(state_0))
                step_index += 1

    wp.synchronize_device(device)
    peak_rss_mb = _maxrss_mb()
    mempool_high_bytes = wp.get_mempool_used_mem_high(device) if wp.get_device(device).is_cuda else None
    mempool_current_bytes = wp.get_mempool_used_mem_current(device) if wp.get_device(device).is_cuda else None

    return {
        "scenario": scenario.name,
        "scenario_family": scenario.family,
        "scenario_description": scenario.description,
        "mode": mode.name,
        "mode_description": mode.description,
        "repeat_index": case_args.repeat_index,
        "device": str(device),
        "warp_version": wp.__version__,
        "warp_path": wp.__file__,
        "newton_version": newton.__version__,
        "newton_path": newton.__file__,
        "warp_deterministic": mode.warp_deterministic,
        "pipeline_deterministic": mode.pipeline_deterministic,
        "broad_phase": scenario.broad_phase,
        "frames": scenario.frames,
        "substeps": scenario.substeps,
        "total_steps": total_steps,
        "setup_seconds": setup_seconds,
        "mean_step_ms": statistics.fmean(step_times_ms) if step_times_ms else 0.0,
        "stdev_step_ms": statistics.pstdev(step_times_ms) if len(step_times_ms) > 1 else 0.0,
        "peak_rss_mb": peak_rss_mb,
        "warp_mempool_high_mb": _to_megabytes(mempool_high_bytes),
        "warp_mempool_current_mb": _to_megabytes(mempool_current_bytes),
        "final_state_hash": state_hashes[-1] if state_hashes else None,
        "final_contact_hash": contact_hashes[-1] if contact_hashes else None,
        "state_hashes": state_hashes,
        "contact_hashes": contact_hashes,
        "contact_counts": contact_counts,
        "max_contact_count": max(contact_counts) if contact_counts else 0,
        "mean_contact_count": statistics.fmean(contact_counts) if contact_counts else 0.0,
    }


def _first_divergent_index(reference: list[Any], candidate: list[Any]) -> int | None:
    for idx, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        if left != right:
            return idx
    return None


def aggregate_results(result_files: list[Path]) -> dict[str, Any]:
    raw_results = [json.loads(path.read_text()) for path in sorted(result_files)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in raw_results:
        grouped.setdefault((result["scenario"], result["mode"]), []).append(result)

    rows: list[dict[str, Any]] = []
    for (scenario_name, mode_name), runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda row: row["repeat_index"])
        ref = runs[0]
        state_stepwise_identical = True
        contact_stepwise_identical = True
        contact_count_trace_identical = True
        first_divergent_step: int | None = None
        final_state_identical = True
        final_contact_identical = True

        for run in runs[1:]:
            state_div = _first_divergent_index(ref["state_hashes"], run["state_hashes"])
            contact_div = _first_divergent_index(ref["contact_hashes"], run["contact_hashes"])
            count_div = _first_divergent_index(ref["contact_counts"], run["contact_counts"])
            if state_div is not None:
                state_stepwise_identical = False
            if contact_div is not None:
                contact_stepwise_identical = False
            if count_div is not None:
                contact_count_trace_identical = False
            candidate_divs = [idx for idx in (state_div, contact_div, count_div) if idx is not None]
            if candidate_divs:
                first = min(candidate_divs)
                first_divergent_step = first if first_divergent_step is None else min(first_divergent_step, first)
            final_state_identical = final_state_identical and (ref["final_state_hash"] == run["final_state_hash"])
            final_contact_identical = final_contact_identical and (ref["final_contact_hash"] == run["final_contact_hash"])

        rows.append(
            {
                "scenario": scenario_name,
                "mode": mode_name,
                "scenario_family": ref["scenario_family"],
                "scenario_description": ref["scenario_description"],
                "mode_description": ref["mode_description"],
                "warp_deterministic": ref["warp_deterministic"],
                "pipeline_deterministic": ref["pipeline_deterministic"],
                "broad_phase": ref["broad_phase"],
                "frames": ref["frames"],
                "substeps": ref["substeps"],
                "repeats": len(runs),
                "state_stepwise_identical": state_stepwise_identical,
                "contact_stepwise_identical": contact_stepwise_identical,
                "contact_count_trace_identical": contact_count_trace_identical,
                "first_divergent_step": first_divergent_step,
                "final_state_identical": final_state_identical,
                "final_contact_identical": final_contact_identical,
                "mean_step_ms": statistics.fmean(run["mean_step_ms"] for run in runs),
                "stdev_step_ms": statistics.pstdev(run["mean_step_ms"] for run in runs) if len(runs) > 1 else 0.0,
                "mean_peak_rss_mb": statistics.fmean(run["peak_rss_mb"] for run in runs),
                "max_peak_rss_mb": max(run["peak_rss_mb"] for run in runs),
                "mean_warp_mempool_high_mb": statistics.fmean((run["warp_mempool_high_mb"] or 0.0) for run in runs),
                "max_warp_mempool_high_mb": max((run["warp_mempool_high_mb"] or 0.0) for run in runs),
                "mean_contact_count": statistics.fmean(run["mean_contact_count"] for run in runs),
                "max_contact_count": max(run["max_contact_count"] for run in runs),
                "contact_count_trace_example": ref["contact_counts"],
            }
        )

    return {"raw_results": raw_results, "rows": rows}


def _plotly_div(div_id: str, figure_json: str) -> str:
    escaped = html.escape(figure_json)
    return f'<div id="{div_id}" class="plot"></div><script>Plotly.newPlot("{div_id}", JSON.parse({json.dumps(figure_json)}).data, JSON.parse({json.dumps(figure_json)}).layout, {{responsive:true}});</script>'


def render_report(aggregate: dict[str, Any], output_path: Path) -> Path:
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    import plotly.io as pio  # noqa: PLC0415

    rows = sorted(aggregate["rows"], key=lambda row: (row["scenario"], row["mode"]))
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    modes = list(MODES.keys())
    mode_labels = {
        "baseline": "baseline",
        "pipeline": "pipeline-only",
        "global": "global-only",
        "combined": "combined",
    }
    colors = {
        "baseline": "#9aa4b2",
        "pipeline": "#f59e0b",
        "global": "#2563eb",
        "combined": "#10b981",
    }
    row_map = {(row["scenario"], row["mode"]): row for row in rows}

    runtime_fig = go.Figure()
    for mode in modes:
        runtime_fig.add_trace(
            go.Bar(
                name=mode_labels[mode],
                x=scenarios,
                y=[row_map[(scenario, mode)]["mean_step_ms"] for scenario in scenarios],
                error_y={"type": "data", "array": [row_map[(scenario, mode)]["stdev_step_ms"] for scenario in scenarios]},
                marker_color=colors[mode],
            )
        )
    runtime_fig.update_layout(
        title="Runtime per simulation substep",
        template="plotly_white",
        barmode="group",
        height=460,
        xaxis_title="Scenario",
        yaxis_title="ms / substep",
    )

    memory_fig = make_subplots(rows=1, cols=2, subplot_titles=("Peak host RSS", "Warp GPU mempool high watermark"))
    for mode in modes:
        memory_fig.add_trace(
            go.Bar(
                name=mode_labels[mode],
                x=scenarios,
                y=[row_map[(scenario, mode)]["mean_peak_rss_mb"] for scenario in scenarios],
                marker_color=colors[mode],
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        memory_fig.add_trace(
            go.Bar(
                name=mode_labels[mode],
                x=scenarios,
                y=[row_map[(scenario, mode)]["mean_warp_mempool_high_mb"] for scenario in scenarios],
                marker_color=colors[mode],
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    memory_fig.update_layout(template="plotly_white", height=460, title="Memory footprint")
    memory_fig.update_yaxes(title_text="MiB", row=1, col=1)
    memory_fig.update_yaxes(title_text="MiB", row=1, col=2)

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=[
                [
                    1.0
                    if row_map[(scenario, mode)]["state_stepwise_identical"] and row_map[(scenario, mode)]["contact_stepwise_identical"]
                    else 0.0
                    for mode in modes
                ]
                for scenario in scenarios
            ],
            x=[mode_labels[mode] for mode in modes],
            y=scenarios,
            text=[
                [
                    "state+contacts stable"
                    if row_map[(scenario, mode)]["state_stepwise_identical"] and row_map[(scenario, mode)]["contact_stepwise_identical"]
                    else f"diverges @ {row_map[(scenario, mode)]['first_divergent_step']}"
                    for mode in modes
                ]
                for scenario in scenarios
            ],
            texttemplate="%{text}",
            colorscale=[[0.0, "#ef4444"], [1.0, "#10b981"]],
            showscale=False,
        )
    )
    heatmap_fig.update_layout(title="Reproducibility pass/fail across repeated runs", template="plotly_white", height=360)

    trace_fig = go.Figure()
    for scenario in scenarios:
        combined_row = row_map[(scenario, "combined")]
        trace = combined_row["contact_count_trace_example"]
        stride = max(1, len(trace) // 120)
        trace_fig.add_trace(
            go.Scatter(
                name=scenario,
                x=list(range(0, len(trace), stride)),
                y=trace[::stride],
                mode="lines",
            )
        )
    trace_fig.update_layout(
        title="Combined-mode contact count traces (example repeat)",
        template="plotly_white",
        height=420,
        xaxis_title="substep",
        yaxis_title="rigid contact count",
    )

    table_header = [
        "scenario",
        "mode",
        "stepwise state",
        "stepwise contacts",
        "first divergent step",
        "mean ms/substep",
        "peak RSS MiB",
        "Warp mempool MiB",
    ]
    table_cells = [
        [row["scenario"] for row in rows],
        [mode_labels[row["mode"]] for row in rows],
        ["yes" if row["state_stepwise_identical"] else "no" for row in rows],
        ["yes" if row["contact_stepwise_identical"] else "no" for row in rows],
        [row["first_divergent_step"] if row["first_divergent_step"] is not None else "—" for row in rows],
        [f"{row['mean_step_ms']:.3f}" for row in rows],
        [f"{row['mean_peak_rss_mb']:.1f}" for row in rows],
        [f"{row['mean_warp_mempool_high_mb']:.1f}" for row in rows],
    ]
    table_fig = go.Figure(
        data=[
            go.Table(
                header={"values": table_header, "fill_color": "#111827", "font": {"color": "white", "size": 13}},
                cells={"values": table_cells, "fill_color": "#f9fafb", "align": "left", "font": {"size": 12}},
            )
        ]
    )
    table_fig.update_layout(height=760, margin={"l": 10, "r": 10, "t": 20, "b": 10})

    total_rows = len(rows)
    stable_rows = sum(
        1 for row in rows if row["state_stepwise_identical"] and row["contact_stepwise_identical"] and row["final_state_identical"]
    )
    combined_stable = sum(
        1 for row in rows if row["mode"] == "combined" and row["state_stepwise_identical"] and row["contact_stepwise_identical"]
    )
    fastest = min(rows, key=lambda row: row["mean_step_ms"])
    summary_cards = [
        ("Scenario/mode cells", str(total_rows)),
        ("Fully reproducible cells", str(stable_rows)),
        ("Combined-mode passes", f"{combined_stable} / {len(scenarios)}"),
        ("Fastest cell", f"{fastest['scenario']} / {mode_labels[fastest['mode']]}") ,
    ]
    summary_html = "".join(
        f'<div class="card"><div class="label">{html.escape(title)}</div><div class="value">{html.escape(value)}</div></div>'
        for title, value in summary_cards
    )

    intro = "".join(
        [
            "<ul>",
            "<li>SolverMuJoCo always runs with use_mujoco_contacts=False, so contacts come from Newton's CollisionPipeline.</li>",
            "<li>The Warp branch under test is the local deterministic helper-atomics branch, with mujoco_warp coming from the local global-determinism workaround branch.</li>",
            "<li>Mode meanings: baseline = neither Warp global determinism nor Newton collision sorting; pipeline-only = Newton sorting only; global-only = Warp deterministic lowering only; combined = both enabled.</li>",
            "<li>Runtime is measured per simulation substep after a short warmup. Memory reports peak process RSS plus Warp's GPU mempool high watermark from a fresh subprocess per repeat.</li>",
            "<li>Benchmark scenes are restricted to primitive-only rigid bodies because current SolverMuJoCo conversion/runtime breaks on mesh-backed geoms in this environment, and articulated Ant conversion also fails before stepping.</li>",
            "</ul>",
        ]
    )

    # full_html=False already returns the inline Plotly loader snippet; older/newer
    # Plotly versions do not wrap it in <body> tags consistently, so embed it as-is.
    plotly_bundle = pio.to_html(go.Figure(), include_plotlyjs="inline", full_html=False)
    runtime_div = _plotly_div("runtime_plot", runtime_fig.to_json())
    memory_div = _plotly_div("memory_plot", memory_fig.to_json())
    heatmap_div = _plotly_div("heatmap_plot", heatmap_fig.to_json())
    trace_div = _plotly_div("trace_plot", trace_fig.to_json())
    table_div = _plotly_div("table_plot", table_fig.to_json())

    html_text = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Newton SolverMuJoCo determinism report</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: #f3f4f6; color: #111827; }}
    .container {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 30px; margin-bottom: 10px; }}
    .muted {{ color: #4b5563; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 18px 0 26px; }}
    .card {{ background: white; border-radius: 14px; padding: 16px; box-shadow: 0 10px 24px rgba(0,0,0,0.06); }}
    .label {{ font-size: 13px; color: #6b7280; margin-bottom: 8px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .section {{ background: white; border-radius: 16px; padding: 18px; margin: 16px 0; box-shadow: 0 10px 24px rgba(0,0,0,0.06); }}
    .footer {{ margin-top: 30px; color: #6b7280; font-size: 13px; }}
    .plot {{ width: 100%; min-height: 320px; }}
    ul {{ line-height: 1.55; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Newton SolverMuJoCo determinism report</h1>
    <div class=\"muted\">Standalone report for Newton-side determinism experiments using SolverMuJoCo with Newton CollisionPipeline contacts (use_mujoco_contacts=False).</div>
    <div class=\"cards\">{summary_html}</div>
    <div class=\"section\">
      <h2>Experiment notes</h2>
      {intro}
    </div>
    <div class=\"section\"><h2>Reproducibility summary table</h2>{table_div}</div>
    <div class=\"section\"><h2>Runtime</h2>{runtime_div}</div>
    <div class=\"section\"><h2>Memory</h2>{memory_div}</div>
    <div class=\"section\"><h2>Reproducibility heatmap</h2>{heatmap_div}</div>
    <div class=\"section\"><h2>Contact-count traces</h2>{trace_div}</div>
    <div class=\"footer\">Generated from {len(aggregate['raw_results'])} subprocess runs.</div>
  </div>
  {plotly_bundle}
</body>
</html>
"""
    _ensure_parent(output_path)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    selected_scenarios = args.scenarios or list(SCENARIOS.keys())
    selected_modes = args.modes or list(MODES.keys())

    for scenario_name in selected_scenarios:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    for mode_name in selected_modes:
        if mode_name not in MODES:
            raise ValueError(f"Unknown mode: {mode_name}")

    result_files: list[Path] = []
    for scenario_name in selected_scenarios:
        for mode_name in selected_modes:
            for repeat_index in range(args.repeats):
                output_path = results_dir / f"{scenario_name}__{mode_name}__repeat{repeat_index}.json"
                cmd = [
                    sys.executable,
                    str(script_path),
                    "run-case",
                    "--scenario",
                    scenario_name,
                    "--mode",
                    mode_name,
                    "--repeat-index",
                    str(repeat_index),
                    "--device",
                    args.device,
                    "--output",
                    str(output_path),
                ]
                print("$", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
                result_files.append(output_path)

    aggregate = aggregate_results(result_files)
    report_path = render_report(aggregate, Path(args.report))
    aggregate_path = results_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, default=_json_default), encoding="utf-8")
    return {
        "results_dir": str(results_dir),
        "report": str(report_path),
        "aggregate": str(aggregate_path),
        "runs": len(result_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Newton SolverMuJoCo determinism and performance harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix = subparsers.add_parser("run-matrix", help="Run the full experiment matrix and render the HTML report.")
    matrix.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    matrix.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    matrix.add_argument("--device", default="cuda:0")
    matrix.add_argument("--repeats", type=int, default=3)
    matrix.add_argument("--scenarios", nargs="*")
    matrix.add_argument("--modes", nargs="*")

    case = subparsers.add_parser("run-case", help="Run one scenario/mode/repeat in a fresh process.")
    case.add_argument("--scenario", required=True, choices=sorted(SCENARIOS.keys()))
    case.add_argument("--mode", required=True, choices=sorted(MODES.keys()))
    case.add_argument("--repeat-index", required=True, type=int)
    case.add_argument("--device", default="cuda:0")
    case.add_argument("--output", type=str, default="")

    report = subparsers.add_parser("render-report", help="Render a report from existing JSON result files.")
    report.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    report.add_argument("--report", default=str(DEFAULT_REPORT_PATH))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-case":
        result = run_case(args)
        output = json.dumps(result, indent=2, default=_json_default)
        if args.output:
            output_path = Path(args.output)
            _ensure_parent(output_path)
            output_path.write_text(output, encoding="utf-8")
        print(output)
        return
    if args.command == "run-matrix":
        summary = run_matrix(args)
        print(json.dumps(summary, indent=2, default=_json_default))
        return
    if args.command == "render-report":
        results_dir = Path(args.results_dir)
        result_files = [path for path in sorted(results_dir.glob("*.json")) if path.name != "aggregate.json"]
        aggregate = aggregate_results(result_files)
        report_path = render_report(aggregate, Path(args.report))
        print(json.dumps({"report": str(report_path)}, indent=2))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
