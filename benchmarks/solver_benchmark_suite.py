# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenario-based solver benchmark suite with tuning, screenshots, and dashboard output."""

from __future__ import annotations

import argparse
import copy
import datetime as datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import warp as wp
from PIL import Image
from pxr import Usd

import newton
import newton.ik as ik
import newton.usd
from newton import JointTargetMode


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"
RESULTS_ROOT = BENCHMARK_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
INDEX_PATH = RESULTS_ROOT / "index.json"
DASHBOARD_PATH = BENCHMARK_ROOT / "dashboard" / "index.html"
SCREENSHOT_ROOT = BENCHMARK_ROOT / "dashboard" / "images"
SCHEMA_VERSION = 2
DEFAULT_SOLVERS = ("SolverXPBD", "SolverMuJoCo", "SolverMABD", "SolverFeatherstone")
DEFAULT_WORLD_SWEEP = (4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192)


def _register_solver_custom_attributes(solver_cls: type[Any], builder: newton.ModelBuilder) -> None:
    register = getattr(solver_cls, "register_custom_attributes", None)
    if callable(register):
        register(builder)
        return

    config_cls = getattr(solver_cls, "Config", None)
    if config_cls is None:
        return

    register = getattr(config_cls, "register_custom_attributes", None)
    if callable(register):
        register(builder)


def _resolve_solver_class(solver_name: str) -> type[Any]:
    solver_cls = getattr(newton.solvers, solver_name, None)
    if solver_cls is None:
        raise AttributeError(f"newton.solvers.{solver_name} is not available in this checkout.")
    return solver_cls


def _format_timestamp_utc() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git_command(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def _collect_run_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _run_git_command(repo_root, "rev-parse", "HEAD")
    branch = _run_git_command(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = _run_git_command(repo_root, "status", "--porcelain", "--untracked-files=no")
    current_device = wp.get_device()

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": _format_timestamp_utc(),
        "git": {
            "commit": commit,
            "commit_short": commit[:7] if commit else None,
            "branch": branch,
            "dirty": bool(dirty_output),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node() or os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
            "warp_version": getattr(wp, "__version__", None),
            "newton_version": getattr(newton, "__version__", None),
            "device": str(current_device),
            "cuda": bool(getattr(current_device, "is_cuda", False)),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _load_run_logs(runs_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not runs_root.exists():
        return runs

    for path in sorted(runs_root.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue

    runs.sort(key=lambda run: run.get("timestamp_utc", ""))
    return runs


def _gpu_memory_used_mib() -> float | None:
    try:
        pid = str(os.getpid())
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    total_mib = 0.0
    found = False
    for line in completed.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2 or parts[0] != pid:
            continue
        try:
            total_mib += float(parts[1])
            found = True
        except ValueError:
            continue
    return total_mib if found else None


@wp.kernel(enable_backward=False)
def _broadcast_ik_solution_kernel(
    ik_solution: wp.array2d(dtype=wp.float32),
    joint_targets: wp.array2d(dtype=wp.float32),
    active_dofs: int,
    gripper_value: float,
    gripper_start: int,
):
    world_idx = wp.tid()
    for j in range(active_dofs):
        joint_targets[world_idx, j] = ik_solution[0, j]
    if gripper_start >= 0:
        joint_targets[world_idx, gripper_start] = gripper_value
        joint_targets[world_idx, gripper_start + 1] = gripper_value


def _quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


def _make_box_mesh(hx: float, hy: float, hz: float) -> newton.Mesh:
    mesh = newton.Mesh.create_box(hx, hy, hz, duplicate_vertices=True, compute_normals=False, compute_uvs=False)
    mesh.finalize()
    return mesh


def _load_mesh_asset(asset_name: str) -> newton.Mesh:
    stage = Usd.Stage.Open(str(newton.utils.download_asset(asset_name) / "model.usda"))
    prim = stage.GetPrimAtPath("/root/Model/Model")
    mesh = newton.usd.get_mesh(prim, load_normals=True, face_varying_normal_conversion="vertex_splitting")
    scale = np.asarray(newton.usd.get_scale(stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
    if not np.allclose(scale, 1.0):
        mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
    mesh.finalize()
    return mesh


def _configure_default_contact_material(builder: newton.ModelBuilder) -> None:
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1.0e-5)
    builder.default_shape_cfg.ke = 2.5e3
    builder.default_shape_cfg.kd = 2.5e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.8


def _sample_world_offsets(world_count: int, spacing_xy: tuple[float, float]) -> wp.vec3:
    grid = max(1, math.ceil(math.sqrt(world_count)))
    return wp.vec3(spacing_xy[0], spacing_xy[1], 0.0) if grid > 1 else wp.vec3(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str
    default_world_count: int
    tuning_world_count: int
    screenshot_world_count: int
    candidate_substeps: tuple[int, ...]
    screenshot_camera_pos: tuple[float, float, float]
    screenshot_camera_pitch: float
    screenshot_camera_yaw: float
    screenshot_world_spacing: tuple[float, float]


@dataclass
class ScenarioBuildResult:
    definition: ScenarioDefinition
    model: newton.Model
    model_single: newton.Model | None
    frame_dt: float
    screenshot_relative_path: str | None = None
    initial_control_joint_targets: np.ndarray | None = None
    control_update_fn: Callable[[int, newton.State, newton.Control], None] | None = None
    validate_fn: Callable[[dict[str, Any], newton.State, newton.Contacts | None], None] | None = None
    world_spacing: tuple[float, float] = (1.0, 1.0)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverTuningCandidate:
    sim_substeps: int
    solver_kwargs: dict[str, Any]


@dataclass
class SolverTuningResult:
    sim_substeps: int
    solver_kwargs: dict[str, Any]
    validation_seconds: float
    validation_frames: int
    notes: dict[str, Any]


def _build_g1_humanoid_scenario(
    definition: ScenarioDefinition,
    solver_classes: list[type[Any]],
    world_count: int,
) -> ScenarioBuildResult:
    g1 = newton.ModelBuilder()
    for solver_cls in solver_classes:
        _register_solver_custom_attributes(solver_cls, g1)
    _configure_default_contact_material(g1)

    asset_path = newton.utils.download_asset("unitree_g1")
    g1.add_usd(
        str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.2)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        skip_mesh_approximation=True,
    )
    g1.approximate_meshes("bounding_box")

    for dof_index in range(6, g1.joint_dof_count):
        g1.joint_target_ke[dof_index] = 500.0
        g1.joint_target_kd[dof_index] = 10.0
        g1.joint_target_mode[dof_index] = int(JointTargetMode.POSITION)

    builder = newton.ModelBuilder()
    for solver_cls in solver_classes:
        _register_solver_custom_attributes(solver_cls, builder)
    builder.replicate(g1, world_count)
    builder.default_shape_cfg.ke = 2.5e3
    builder.default_shape_cfg.kd = 2.5e2
    builder.add_ground_plane()
    model = builder.finalize()

    def validate(metrics: dict[str, Any], state: newton.State, contacts: newton.Contacts | None) -> None:
        body_q = state.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Non-finite G1 body transforms.")
        z_values = body_q[:, 2]
        if float(z_values.min()) < -0.3 or float(z_values.max()) > 3.0:
            raise ValueError("G1 bodies left the expected Z range.")
        if metrics["max_contacts"] <= 0:
            raise ValueError("G1 scenario did not generate any contacts.")

    return ScenarioBuildResult(
        definition=definition,
        model=model,
        model_single=None,
        frame_dt=1.0 / 60.0,
        validate_fn=validate,
        world_spacing=definition.screenshot_world_spacing,
    )


def _build_franka_manipulation_scenario(
    definition: ScenarioDefinition,
    solver_classes: list[type[Any]],
    world_count: int,
    *,
    include_mesh_clutter: bool,
) -> ScenarioBuildResult:
    builder = newton.ModelBuilder()
    for solver_cls in solver_classes:
        _register_solver_custom_attributes(solver_cls, builder)
    _configure_default_contact_material(builder)

    franka_asset = newton.utils.download_asset("franka_emika_panda")
    builder.add_urdf(
        str(franka_asset / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform(wp.vec3(-0.48, -0.5, 0.0), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        parse_visuals_as_colliders=False,
    )
    builder.joint_q[:9] = [0.0, -0.25, 0.0, -1.95, 0.0, 2.2, 0.8, 0.04, 0.04]
    builder.joint_target_ke[:9] = [250.0] * 9
    builder.joint_target_kd[:9] = [35.0] * 9
    builder.joint_effort_limit[:7] = [80.0] * 7
    builder.joint_effort_limit[7:9] = [20.0] * 2
    builder.joint_target_mode[:9] = [int(JointTargetMode.POSITION)] * 9
    builder.joint_armature[:7] = [0.1] * 7
    builder.joint_armature[7:9] = [0.5] * 2

    builder.approximate_meshes("convex_hull", keep_visual_shapes=True)

    table_mesh = _make_box_mesh(0.44, 0.36, 0.04)
    table_body = builder.add_body(xform=wp.transform(wp.vec3(0.02, -0.5, 0.04), wp.quat_identity()), label="table")
    builder.add_shape_mesh(body=table_body, mesh=table_mesh, cfg=copy.deepcopy(builder.default_shape_cfg))

    object_body_labels: list[str] = []
    if include_mesh_clutter:
        cup_mesh = _load_mesh_asset("manipulation_objects/cup")
        pad_mesh = _load_mesh_asset("manipulation_objects/pad")
        cup_body = builder.add_body(xform=wp.transform(wp.vec3(0.08, -0.56, 0.10), wp.quat_identity()), label="cup_a")
        builder.add_shape_mesh(body=cup_body, mesh=cup_mesh, cfg=copy.deepcopy(builder.default_shape_cfg))
        object_body_labels.append("cup_a")

        cup_body_b = builder.add_body(xform=wp.transform(wp.vec3(-0.05, -0.46, 0.10), wp.quat_identity()), label="cup_b")
        builder.add_shape_mesh(body=cup_body_b, mesh=cup_mesh, cfg=copy.deepcopy(builder.default_shape_cfg))
        object_body_labels.append("cup_b")

        pad_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4)
        pad_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, -0.52, 0.085), pad_rot), label="pad")
        builder.add_shape_mesh(body=pad_body, mesh=pad_mesh, cfg=copy.deepcopy(builder.default_shape_cfg))
        object_body_labels.append("pad")
    else:
        cube_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, -0.5, 0.105), wp.quat_identity()), label="cube")
        builder.add_shape_box(body=cube_body, hx=0.025, hy=0.025, hz=0.025, cfg=copy.deepcopy(builder.default_shape_cfg))
        object_body_labels.append("cube")

    model_single = builder.finalize()
    local_label_to_index = {label: index for index, label in enumerate(model_single.body_label)}

    scene = newton.ModelBuilder()
    for solver_cls in solver_classes:
        _register_solver_custom_attributes(solver_cls, scene)
    scene.replicate(builder, world_count)
    scene.add_ground_plane()
    model = scene.finalize()

    control = model.control()
    joint_targets_2d = wp.zeros((world_count, control.joint_target_pos.shape[0] // world_count), dtype=wp.float32)
    ik_state = model_single.state()
    newton.eval_fk(model_single, model_single.joint_q, model_single.joint_qd, ik_state)
    ee_index = 10
    ik_dofs = 7
    ik_solution = wp.clone(model_single.joint_q.reshape((1, -1)))
    pos_obj = ik.IKObjectivePosition(
        link_index=ee_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([[0.0, -0.5, 0.28]], dtype=wp.vec3),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=ee_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([_quat_to_vec4(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi))], dtype=wp.vec4),
    )
    ik_solver = ik.IKSolver(
        model=model_single,
        n_problems=1,
        objectives=[pos_obj, rot_obj],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )

    if include_mesh_clutter:
        waypoints = [
            (wp.vec3(-0.16, -0.47, 0.33), 45, 0.0),
            (wp.vec3(0.12, -0.58, 0.23), 50, 0.2),
            (wp.vec3(0.12, -0.42, 0.23), 50, 0.45),
            (wp.vec3(-0.12, -0.52, 0.28), 45, 0.1),
        ]
    else:
        waypoints = [
            (wp.vec3(-0.10, -0.50, 0.31), 45, 0.0),
            (wp.vec3(0.02, -0.50, 0.19), 50, 0.65),
            (wp.vec3(0.11, -0.47, 0.32), 45, 0.65),
            (wp.vec3(0.11, -0.47, 0.20), 40, 0.1),
        ]

    def control_update(frame_index: int, state: newton.State, control_buffer: newton.Control) -> None:
        period = sum(duration for _, duration, _ in waypoints)
        frame_mod = frame_index % period
        cursor = 0
        target_pos = waypoints[0][0]
        grasp_value = waypoints[0][2]
        for idx, (waypoint_pos, duration, grasp) in enumerate(waypoints):
            next_pos, _, next_grasp = waypoints[(idx + 1) % len(waypoints)]
            if frame_mod < cursor + duration:
                t = (frame_mod - cursor) / max(duration, 1)
                target_pos = wp.vec3(
                    waypoint_pos[0] * (1.0 - t) + next_pos[0] * t,
                    waypoint_pos[1] * (1.0 - t) + next_pos[1] * t,
                    waypoint_pos[2] * (1.0 - t) + next_pos[2] * t,
                )
                grasp_value = grasp * (1.0 - t) + next_grasp * t
                break
            cursor += duration
        pos_obj.set_target_positions(wp.array([target_pos], dtype=wp.vec3))
        ik_solver.step(ik_solution, ik_solution, iterations=24)
        gripper_value = 0.06 * (1.0 - grasp_value)
        wp.launch(
            _broadcast_ik_solution_kernel,
            dim=world_count,
            inputs=[ik_solution, joint_targets_2d, ik_dofs, gripper_value, ik_dofs],
        )
        wp.copy(control_buffer.joint_target_pos, joint_targets_2d.flatten())

    def validate(metrics: dict[str, Any], state: newton.State, contacts: newton.Contacts | None) -> None:
        body_q = state.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Non-finite Franka body transforms.")
        z_values = body_q[:, 2]
        if float(z_values.min()) < -0.25 or float(z_values.max()) > 2.5:
            raise ValueError("Franka bodies left the expected Z range.")
        if metrics["max_contacts"] <= 0 or metrics["contact_frames"] <= 0:
            raise ValueError("Franka scenario did not sustain rigid contacts.")

        bodies_per_world = model.body_count // world_count
        for world_idx in range(world_count):
            base = world_idx * bodies_per_world
            for label in object_body_labels:
                object_body = base + local_label_to_index[label]
                if float(body_q[object_body][2]) < 0.05:
                    raise ValueError(f"Object {label} dropped through or under the table.")

    return ScenarioBuildResult(
        definition=definition,
        model=model,
        model_single=model_single,
        frame_dt=1.0 / 60.0,
        control_update_fn=control_update,
        validate_fn=validate,
        world_spacing=definition.screenshot_world_spacing,
        notes={"object_labels": object_body_labels, "include_mesh_clutter": include_mesh_clutter},
    )


def _build_h1_mesh_sweep_scenario(
    definition: ScenarioDefinition,
    solver_classes: list[type[Any]],
    world_count: int,
) -> ScenarioBuildResult:
    builder = newton.ModelBuilder()
    for solver_cls in solver_classes:
        _register_solver_custom_attributes(solver_cls, builder)
    _configure_default_contact_material(builder)

    builder.add_mjcf(newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml", floating=False)
    table_mesh = _make_box_mesh(0.32, 0.20, 0.04)
    table_body = builder.add_body(xform=wp.transform(wp.vec3(0.52, -0.10, 0.04), wp.quat_identity()), label="table")
    builder.add_shape_mesh(body=table_body, mesh=table_mesh, cfg=copy.deepcopy(builder.default_shape_cfg))

    cup_mesh = _load_mesh_asset("manipulation_objects/cup")
    pad_mesh = _load_mesh_asset("manipulation_objects/pad")
    cup_body = builder.add_body(xform=wp.transform(wp.vec3(0.48, -0.13, 0.10), wp.quat_identity()), label="cup")
    builder.add_shape_mesh(body=cup_body, mesh=cup_mesh, cfg=copy.deepcopy(builder.default_shape_cfg))
    pad_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.56, -0.04, 0.085), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6)),
        label="pad",
    )
    builder.add_shape_mesh(body=pad_body, mesh=pad_mesh, cfg=copy.deepcopy(builder.default_shape_cfg))

    model_single = builder.finalize()
    scene = newton.ModelBuilder()
    for solver_cls in solver_classes:
        _register_solver_custom_attributes(solver_cls, scene)
    scene.replicate(builder, world_count)
    scene.add_ground_plane()
    model = scene.finalize()

    control = model.control()
    joint_targets_2d = wp.zeros((world_count, control.joint_target_pos.shape[0] // world_count), dtype=wp.float32)
    ik_state = model_single.state()
    newton.eval_fk(model_single, model_single.joint_q, model_single.joint_qd, ik_state)
    ee_index = 33
    ik_solution = wp.clone(model_single.joint_q.reshape((1, -1)))
    pos_obj = ik.IKObjectivePosition(
        link_index=ee_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([[0.55, -0.15, 0.26]], dtype=wp.vec3),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=ee_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([_quat_to_vec4(wp.quat_identity())], dtype=wp.vec4),
    )
    obj_joint_limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=model_single.joint_limit_lower,
        joint_limit_upper=model_single.joint_limit_upper,
    )
    ik_solver = ik.IKSolver(
        model=model_single,
        n_problems=1,
        objectives=[pos_obj, rot_obj, obj_joint_limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    waypoints = [
        wp.vec3(0.40, -0.18, 0.29),
        wp.vec3(0.61, -0.17, 0.20),
        wp.vec3(0.62, -0.02, 0.20),
        wp.vec3(0.43, -0.01, 0.28),
    ]

    def control_update(frame_index: int, state: newton.State, control_buffer: newton.Control) -> None:
        phase = (frame_index % 240) / 240.0
        segment = int(phase * len(waypoints))
        next_segment = (segment + 1) % len(waypoints)
        local_t = phase * len(waypoints) - segment
        a = waypoints[segment]
        b = waypoints[next_segment]
        pos_obj.set_target_positions(
            wp.array(
                [[a[0] * (1.0 - local_t) + b[0] * local_t, a[1] * (1.0 - local_t) + b[1] * local_t, a[2] * (1.0 - local_t) + b[2] * local_t]],
                dtype=wp.vec3,
            )
        )
        ik_solver.step(ik_solution, ik_solution, iterations=32)
        wp.launch(
            _broadcast_ik_solution_kernel,
            dim=world_count,
            inputs=[ik_solution, joint_targets_2d, joint_targets_2d.shape[1], 0.04, -1],
        )
        wp.copy(control_buffer.joint_target_pos, joint_targets_2d.flatten())

    def validate(metrics: dict[str, Any], state: newton.State, contacts: newton.Contacts | None) -> None:
        body_q = state.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Non-finite H1 body transforms.")
        z_values = body_q[:, 2]
        if float(z_values.min()) < -0.5 or float(z_values.max()) > 3.5:
            raise ValueError("H1 bodies left the expected Z range.")
        if metrics["max_contacts"] <= 0 or metrics["contact_frames"] <= 0:
            raise ValueError("H1 scenario did not sustain rigid contacts.")

    return ScenarioBuildResult(
        definition=definition,
        model=model,
        model_single=model_single,
        frame_dt=1.0 / 60.0,
        control_update_fn=control_update,
        validate_fn=validate,
        world_spacing=definition.screenshot_world_spacing,
        notes={"object_labels": ["cup", "pad"]},
    )


SCENARIOS: dict[str, ScenarioDefinition] = {
    "g1_humanoid": ScenarioDefinition(
        name="g1_humanoid",
        description="Unitree G1 standing benchmark with sustained foot-ground contacts.",
        default_world_count=4,
        tuning_world_count=4,
        screenshot_world_count=9,
        candidate_substeps=(4, 6, 8),
        screenshot_camera_pos=(6.0, -6.0, 2.5),
        screenshot_camera_pitch=-12.0,
        screenshot_camera_yaw=-135.0,
        screenshot_world_spacing=(2.8, 2.8),
    ),
    "franka_cube_manipulation": ScenarioDefinition(
        name="franka_cube_manipulation",
        description="Fixed-base Franka arm picking and placing a cube on a table with scripted IK targets.",
        default_world_count=4,
        tuning_world_count=4,
        screenshot_world_count=9,
        candidate_substeps=(4, 6, 8, 10),
        screenshot_camera_pos=(1.2, 0.2, 0.65),
        screenshot_camera_pitch=-18.0,
        screenshot_camera_yaw=-145.0,
        screenshot_world_spacing=(1.1, 1.0),
    ),
    "franka_mesh_clutter": ScenarioDefinition(
        name="franka_mesh_clutter",
        description="Fixed-base Franka arm sweeping multiple mesh bodies on a table with scripted IK targets.",
        default_world_count=4,
        tuning_world_count=4,
        screenshot_world_count=9,
        candidate_substeps=(6, 8, 10),
        screenshot_camera_pos=(1.35, 0.3, 0.72),
        screenshot_camera_pitch=-16.0,
        screenshot_camera_yaw=-145.0,
        screenshot_world_spacing=(1.15, 1.1),
    ),
    "h1_mesh_sweep": ScenarioDefinition(
        name="h1_mesh_sweep",
        description="Fixed-base Unitree H1 sweeping mesh objects across a table with whole-body IK.",
        default_world_count=4,
        tuning_world_count=4,
        screenshot_world_count=9,
        candidate_substeps=(4, 6, 8),
        screenshot_camera_pos=(4.0, -2.0, 1.7),
        screenshot_camera_pitch=-12.0,
        screenshot_camera_yaw=-110.0,
        screenshot_world_spacing=(2.0, 2.0),
    ),
}


def _build_scenario(
    definition: ScenarioDefinition,
    solver_classes: list[type[Any]],
    world_count: int,
) -> ScenarioBuildResult:
    if definition.name == "g1_humanoid":
        return _build_g1_humanoid_scenario(definition, solver_classes, world_count)
    if definition.name == "franka_cube_manipulation":
        return _build_franka_manipulation_scenario(definition, solver_classes, world_count, include_mesh_clutter=False)
    if definition.name == "franka_mesh_clutter":
        return _build_franka_manipulation_scenario(definition, solver_classes, world_count, include_mesh_clutter=True)
    if definition.name == "h1_mesh_sweep":
        return _build_h1_mesh_sweep_scenario(definition, solver_classes, world_count)
    raise KeyError(f"Unsupported scenario {definition.name}.")


def _make_solver_candidates(solver_name: str, scenario: ScenarioDefinition) -> list[SolverTuningCandidate]:
    candidates: list[SolverTuningCandidate] = []
    for sim_substeps in scenario.candidate_substeps:
        if solver_name == "SolverXPBD":
            for iterations in (4, 8, 12):
                candidates.append(SolverTuningCandidate(sim_substeps=sim_substeps, solver_kwargs={"iterations": iterations}))
        elif solver_name == "SolverMuJoCo":
            for iterations in (20, 40, 80):
                candidates.append(
                    SolverTuningCandidate(
                        sim_substeps=sim_substeps,
                        solver_kwargs={
                            "use_mujoco_cpu": False,
                            "solver": "newton",
                            "integrator": "implicitfast",
                            "njmax": 3000,
                            "nconmax": 3000,
                            "cone": "elliptic",
                            "impratio": 100.0,
                            "iterations": iterations,
                            "ls_iterations": max(20, iterations),
                            "use_mujoco_contacts": False,
                        },
                    )
                )
        elif solver_name == "SolverFeatherstone":
            for impulse_iterations in (2, 4, 6):
                candidates.append(
                    SolverTuningCandidate(
                        sim_substeps=sim_substeps,
                        solver_kwargs={
                            "update_mass_matrix_interval": sim_substeps,
                            "use_tile_gemm": True,
                            "fuse_cholesky": True,
                            "rigid_contact_method": "force",
                            "impulse_contact_iterations": impulse_iterations,
                            "impulse_contact_warmstart_scale": 0.9,
                        },
                    )
                )
        elif solver_name == "SolverMABD":
            candidates.append(
                SolverTuningCandidate(
                    sim_substeps=sim_substeps,
                    solver_kwargs={"use_tile_gemm": True},
                )
            )
        else:
            candidates.append(SolverTuningCandidate(sim_substeps=sim_substeps, solver_kwargs={}))
    return candidates


def _instantiate_solver(
    solver_name: str,
    solver_cls: type[Any],
    model: newton.Model,
    solver_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    try:
        return solver_cls(model, **solver_kwargs), dict(solver_kwargs)
    except TypeError:
        config_cls = getattr(solver_cls, "Config", None)
        if config_cls is None:
            raise
        return solver_cls(model, config=config_cls(), **solver_kwargs), dict(solver_kwargs)


def _make_contacts(
    solver_name: str,
    solver: Any,
    model: newton.Model,
    *,
    use_mujoco_contacts: bool,
) -> newton.Contacts | None:
    if solver_name == "SolverMuJoCo" and use_mujoco_contacts:
        get_max_contact_count = getattr(solver, "get_max_contact_count", None)
        if not callable(get_max_contact_count):
            raise AttributeError("SolverMuJoCo does not expose get_max_contact_count().")
        return newton.Contacts(get_max_contact_count(), 0)
    return model.contacts()


def _create_simulation_state(model: newton.Model) -> tuple[newton.State, newton.State, newton.Control]:
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    return state_in, state_out, control


def _step_frame(
    *,
    solver_name: str,
    solver: Any,
    model: newton.Model,
    state_in: newton.State,
    state_out: newton.State,
    control: newton.Control,
    contacts: newton.Contacts | None,
    sim_substeps: int,
    sim_dt: float,
    use_mujoco_contacts: bool,
) -> tuple[newton.State, newton.State, int]:
    max_contacts = 0
    for _ in range(sim_substeps):
        if contacts is not None and not (solver_name == "SolverMuJoCo" and use_mujoco_contacts):
            model.collide(state_in, contacts)
            max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, sim_dt)
        state_in, state_out = state_out, state_in
    if solver_name == "SolverMuJoCo" and use_mujoco_contacts and contacts is not None:
        solver.update_contacts(contacts, state_in)
        max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
    return state_in, state_out, max_contacts


def _run_validation_pass(
    *,
    scenario_build: ScenarioBuildResult,
    solver_name: str,
    solver_cls: type[Any],
    candidate: SolverTuningCandidate,
    validation_frames: int,
    warmup_frames: int,
) -> tuple[SolverTuningResult | None, str | None]:
    solver = None
    try:
        solver, solver_kwargs = _instantiate_solver(solver_name, solver_cls, scenario_build.model, candidate.solver_kwargs)
        contacts = _make_contacts(solver_name, solver, scenario_build.model, use_mujoco_contacts=False)
        state_in, state_out, control = _create_simulation_state(scenario_build.model)
        if scenario_build.initial_control_joint_targets is not None:
            wp.copy(control.joint_target_pos, wp.array(scenario_build.initial_control_joint_targets, dtype=wp.float32))

        sim_dt = scenario_build.frame_dt / candidate.sim_substeps
        metrics = {"max_contacts": 0, "contact_frames": 0}

        for frame_idx in range(warmup_frames):
            if scenario_build.control_update_fn is not None:
                scenario_build.control_update_fn(frame_idx, state_in, control)
            state_in, state_out, frame_contacts = _step_frame(
                solver_name=solver_name,
                solver=solver,
                model=scenario_build.model,
                state_in=state_in,
                state_out=state_out,
                control=control,
                contacts=contacts,
                sim_substeps=candidate.sim_substeps,
                sim_dt=sim_dt,
                use_mujoco_contacts=False,
            )
            if frame_contacts > 0:
                metrics["contact_frames"] += 1
            metrics["max_contacts"] = max(metrics["max_contacts"], frame_contacts)

        wp.synchronize()
        validation_start = time.perf_counter()
        for local_frame in range(validation_frames):
            frame_idx = warmup_frames + local_frame
            if scenario_build.control_update_fn is not None:
                scenario_build.control_update_fn(frame_idx, state_in, control)
            state_in, state_out, frame_contacts = _step_frame(
                solver_name=solver_name,
                solver=solver,
                model=scenario_build.model,
                state_in=state_in,
                state_out=state_out,
                control=control,
                contacts=contacts,
                sim_substeps=candidate.sim_substeps,
                sim_dt=sim_dt,
                use_mujoco_contacts=False,
            )
            if frame_contacts > 0:
                metrics["contact_frames"] += 1
            metrics["max_contacts"] = max(metrics["max_contacts"], frame_contacts)
        wp.synchronize()

        if scenario_build.validate_fn is not None:
            scenario_build.validate_fn(metrics, state_in, contacts)

        return SolverTuningResult(
            sim_substeps=candidate.sim_substeps,
            solver_kwargs=solver_kwargs,
            validation_seconds=time.perf_counter() - validation_start,
            validation_frames=validation_frames,
            notes=metrics,
        ), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        close_viewer = getattr(solver, "close_mujoco_viewer", None)
        if callable(close_viewer):
            close_viewer()


def _tune_solver_for_scenario(
    *,
    scenario: ScenarioDefinition,
    solver_name: str,
    solver_cls: type[Any],
    scenario_build: ScenarioBuildResult,
) -> SolverTuningResult:
    candidates = _make_solver_candidates(solver_name, scenario)
    successes: list[SolverTuningResult] = []
    last_error = "unknown error"
    for candidate in candidates:
        result, error = _run_validation_pass(
            scenario_build=scenario_build,
            solver_name=solver_name,
            solver_cls=solver_cls,
            candidate=candidate,
            validation_frames=120,
            warmup_frames=60,
        )
        if result is not None:
            successes.append(result)
        elif error is not None:
            last_error = error
    if not successes:
        raise RuntimeError(f"No stable configuration found for {solver_name} on {scenario.name}. Last error: {last_error}")
    successes.sort(key=lambda item: (item.validation_seconds / item.validation_frames, item.sim_substeps))
    return successes[0]


def _run_solver_benchmark(
    *,
    scenario_build: ScenarioBuildResult,
    solver_name: str,
    solver_cls: type[Any] | None,
    tuning: SolverTuningResult,
    warmup_frames: int,
    measure_frames: int,
) -> dict[str, Any]:
    setup_start = time.perf_counter()
    solver = None

    try:
        if solver_cls is None:
            raise AttributeError(f"newton.solvers.{solver_name} is not available in this checkout.")
        solver, solver_options = _instantiate_solver(solver_name, solver_cls, scenario_build.model, tuning.solver_kwargs)
        contacts = _make_contacts(solver_name, solver, scenario_build.model, use_mujoco_contacts=False)
        state_in, state_out, control = _create_simulation_state(scenario_build.model)
        setup_seconds = time.perf_counter() - setup_start
        sim_dt = scenario_build.frame_dt / tuning.sim_substeps

        if scenario_build.initial_control_joint_targets is not None:
            wp.copy(control.joint_target_pos, wp.array(scenario_build.initial_control_joint_targets, dtype=wp.float32))

        metrics = {
            "max_contacts": 0,
            "contact_frames": 0,
            "initial_contact_count": 0,
        }

        if contacts is not None:
            scenario_build.model.collide(state_in, contacts)
            metrics["initial_contact_count"] = int(contacts.rigid_contact_count.numpy()[0])

        wp.synchronize()
        warmup_start = time.perf_counter()
        for frame_idx in range(warmup_frames):
            if scenario_build.control_update_fn is not None:
                scenario_build.control_update_fn(frame_idx, state_in, control)
            state_in, state_out, frame_contacts = _step_frame(
                solver_name=solver_name,
                solver=solver,
                model=scenario_build.model,
                state_in=state_in,
                state_out=state_out,
                control=control,
                contacts=contacts,
                sim_substeps=tuning.sim_substeps,
                sim_dt=sim_dt,
                use_mujoco_contacts=False,
            )
            if frame_contacts > 0:
                metrics["contact_frames"] += 1
            metrics["max_contacts"] = max(metrics["max_contacts"], frame_contacts)
        wp.synchronize()
        warmup_seconds = time.perf_counter() - warmup_start
        gpu_memory_after_warmup_mib = _gpu_memory_used_mib()

        wp.synchronize()
        measure_start = time.perf_counter()
        for local_frame in range(measure_frames):
            frame_idx = warmup_frames + local_frame
            if scenario_build.control_update_fn is not None:
                scenario_build.control_update_fn(frame_idx, state_in, control)
            state_in, state_out, frame_contacts = _step_frame(
                solver_name=solver_name,
                solver=solver,
                model=scenario_build.model,
                state_in=state_in,
                state_out=state_out,
                control=control,
                contacts=contacts,
                sim_substeps=tuning.sim_substeps,
                sim_dt=sim_dt,
                use_mujoco_contacts=False,
            )
            if frame_contacts > 0:
                metrics["contact_frames"] += 1
            metrics["max_contacts"] = max(metrics["max_contacts"], frame_contacts)
        wp.synchronize()
        measure_seconds = time.perf_counter() - measure_start
        gpu_memory_measure_mib = _gpu_memory_used_mib()

        if scenario_build.validate_fn is not None:
            scenario_build.validate_fn(metrics, state_in, contacts)

        frames_per_second = measure_frames / measure_seconds
        sim_seconds_per_second = (measure_frames * scenario_build.frame_dt) / measure_seconds

        return {
            "solver": solver_name,
            "status": "ok",
            "solver_options": solver_options,
            "tuned_substeps": tuning.sim_substeps,
            "tuning_validation_seconds": tuning.validation_seconds,
            "tuning_notes": tuning.notes,
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "measure_seconds": measure_seconds,
            "frames_per_second": frames_per_second,
            "sim_seconds_per_second": sim_seconds_per_second,
            "mean_frame_ms": 1000.0 * measure_seconds / measure_frames,
            "gpu_memory_after_warmup_mib": gpu_memory_after_warmup_mib,
            "gpu_memory_measure_mib": gpu_memory_measure_mib,
            "max_contacts": metrics["max_contacts"],
            "contact_frames": metrics["contact_frames"],
            "initial_contact_count": metrics["initial_contact_count"],
        }
    except Exception as exc:
        return {
            "solver": solver_name,
            "status": "error",
            "setup_seconds": time.perf_counter() - setup_start,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        close_viewer = getattr(solver, "close_mujoco_viewer", None)
        if callable(close_viewer):
            close_viewer()


def _take_scenario_screenshot(
    *,
    scenario_name: str,
    scenario_build: ScenarioBuildResult,
    solver_name: str,
    solver_cls: type[Any],
    tuning: SolverTuningResult,
) -> str:
    viewer = newton.viewer.ViewerGL(headless=True)
    solver = None
    try:
        viewer.set_model(scenario_build.model)
        viewer.set_world_offsets(_sample_world_offsets(scenario_build.model.world_count, scenario_build.world_spacing))
        camera_pos = wp.vec3(*scenario_build.definition.screenshot_camera_pos)
        viewer.set_camera(camera_pos, scenario_build.definition.screenshot_camera_pitch, scenario_build.definition.screenshot_camera_yaw)

        solver, _ = _instantiate_solver(solver_name, solver_cls, scenario_build.model, tuning.solver_kwargs)
        contacts = _make_contacts(solver_name, solver, scenario_build.model, use_mujoco_contacts=False)
        state_in, state_out, control = _create_simulation_state(scenario_build.model)
        sim_dt = scenario_build.frame_dt / tuning.sim_substeps

        for frame_idx in range(90):
            if scenario_build.control_update_fn is not None:
                scenario_build.control_update_fn(frame_idx, state_in, control)
            state_in, state_out, _ = _step_frame(
                solver_name=solver_name,
                solver=solver,
                model=scenario_build.model,
                state_in=state_in,
                state_out=state_out,
                control=control,
                contacts=contacts,
                sim_substeps=tuning.sim_substeps,
                sim_dt=sim_dt,
                use_mujoco_contacts=False,
            )

        viewer.begin_frame(90.0 * scenario_build.frame_dt)
        viewer.log_state(state_in)
        if contacts is not None:
            viewer.log_contacts(contacts, state_in)
        viewer.end_frame()
        wp.synchronize()
        frame = viewer.get_frame().numpy()
        relative_path = f"images/{scenario_name}.png"
        output_path = BENCHMARK_ROOT / "dashboard" / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(output_path)
        return relative_path.replace("\\", "/")
    finally:
        if solver is not None:
            close_viewer = getattr(solver, "close_mujoco_viewer", None)
            if callable(close_viewer):
                close_viewer()
        viewer.close()


def _build_run_payload(
    metadata: dict[str, Any],
    *,
    scenario_build: ScenarioBuildResult,
    world_count: int,
    warmup_frames: int,
    measure_frames: int,
    solvers: list[str],
    scenario_setup_seconds: float,
    tuning: dict[str, SolverTuningResult],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **metadata,
        "benchmark": {
            "warmup_frames": warmup_frames,
            "measure_frames": measure_frames,
            "solvers": solvers,
            "scenario_setup_seconds": scenario_setup_seconds,
        },
        "scenario": {
            "name": scenario_build.definition.name,
            "description": scenario_build.definition.description,
            "world_count": world_count,
            "frame_dt": scenario_build.frame_dt,
            "screenshot_relative_path": scenario_build.screenshot_relative_path,
            "notes": scenario_build.notes,
        },
        "tuning": {
            solver_name: {
                "sim_substeps": tuning_result.sim_substeps,
                "solver_kwargs": tuning_result.solver_kwargs,
                "validation_seconds": tuning_result.validation_seconds,
                "validation_frames": tuning_result.validation_frames,
                "notes": tuning_result.notes,
            }
            for solver_name, tuning_result in tuning.items()
        },
        "results": results,
    }


def _render_dashboard_html(index_payload: dict[str, Any]) -> str:
    embedded_payload = json.dumps(index_payload, indent=2).replace("</", "<\\/")
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Newton Solver Benchmarks</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #f3efe6; --panel: #fffaf2; --border: #d8cbb6; --ink: #181814; --muted: #665f55;
      --accent: #0b6e5c; --bad: #9a2f23; --bad-soft: #f6dfda; --good: #1a6038; --good-soft: #dff2e4;
      --shadow: 0 16px 42px rgba(24, 24, 20, 0.08); --radius: 16px;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      --sans: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); font-family: var(--sans); background: var(--bg); }
    main { max-width: 1440px; margin: 0 auto; padding: 28px 18px 48px; }
    h1 { margin: 0; font-size: clamp(2rem, 4vw, 3rem); letter-spacing: -0.04em; }
    .subtitle { color: var(--muted); max-width: 76ch; line-height: 1.5; margin: 10px 0 0; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); }
    .card { padding: 18px; }
    .tabs { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }
    .tab { border: 1px solid var(--border); background: rgba(255,255,255,0.55); color: var(--ink); border-radius: 999px; padding: 10px 16px; cursor: pointer; font-size: 0.95rem; }
    .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }
    .plot { min-height: 340px; }
    .summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .summary-item { padding: 12px; border-radius: 12px; background: rgba(255,255,255,0.62); border: 1px solid rgba(216,203,182,0.85); }
    .summary-label { color: var(--muted); font-size: 0.84rem; margin-bottom: 6px; }
    .summary-value { font-family: var(--mono); font-size: 0.93rem; word-break: break-word; }
    .hero { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; align-items: stretch; }
    .hero img { width: 100%; height: 100%; object-fit: cover; border-radius: 14px; border: 1px solid rgba(216,203,182,0.85); background: #efe5d5; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    th, td { padding: 11px 12px; text-align: left; border-top: 1px solid rgba(216,203,182,0.7); vertical-align: top; }
    th { color: var(--muted); font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; border-top: none; }
    .status { display: inline-flex; border-radius: 999px; padding: 4px 10px; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.03em; }
    .status-ok { color: var(--good); background: var(--good-soft); }
    .status-error { color: var(--bad); background: var(--bad-soft); }
    .mono { font-family: var(--mono); }
    .empty { padding: 28px; color: var(--muted); line-height: 1.6; }
    .footer { margin-top: 18px; color: var(--muted); font-size: 0.9rem; }
    @media (max-width: 960px) { .grid, .hero { grid-template-columns: 1fr; } .summary-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header><h1>Newton Solver Benchmarks</h1><p class="subtitle">Static dashboard for solver sweeps across world counts, with tuning metadata, sustained-contact sanity checks, and scenario screenshots.</p></header>
    <section class="panel card">
      <div id="tab-bar" class="tabs"></div>
      <div id="empty-state" class="empty" hidden>No benchmark runs were found.</div>
      <div id="scenario-view" hidden>
        <div class="hero">
          <section class="panel card"><h2 id="scenario-title"></h2><p id="scenario-description"></p><div id="scenario-summary" class="summary-grid"></div></section>
          <section class="panel card"><img id="scenario-image" alt="Scenario screenshot"></section>
        </div>
        <div class="grid" style="margin-top: 18px;">
          <section class="panel card"><h3>FPS vs Worlds</h3><div id="fps-plot" class="plot"></div></section>
          <section class="panel card"><h3>GPU Memory vs Worlds</h3><div id="memory-plot" class="plot"></div></section>
        </div>
        <div class="grid" style="margin-top: 18px;">
          <section class="panel card"><h3>Sim x Realtime</h3><div id="throughput-plot" class="plot"></div></section>
          <section class="panel card"><h3>Latest Sweep Notes</h3><div id="scenario-notes" class="summary-grid"></div></section>
        </div>
        <section class="panel card" style="margin-top: 18px;">
          <h3>Historical Runs</h3>
          <div class="table-wrap"><table><thead><tr><th>Timestamp</th><th>Commit</th><th>Worlds</th><th>Solver</th><th>Status</th><th>FPS</th><th>GPU MiB</th><th>Substeps</th><th>Contacts</th><th>Notes</th></tr></thead><tbody id="runs-table-body"></tbody></table></div>
        </section>
      </div>
      <div class="footer" id="footer-text"></div>
    </section>
  </main>
  <script>
    const DASHBOARD_DATA = __DASHBOARD_DATA__;
    function formatNumber(v, d = 3) { if (v === null || v === undefined || Number.isNaN(v)) return "n/a"; return Number(v).toFixed(d); }
    function formatTimestamp(v) { if (!v) return "n/a"; return v.replace("T", " ").replace("Z", " UTC"); }
    function scenarioNamesFromRuns(runs) { return [...new Set(runs.map((run) => run.scenario.name))].sort(); }
    function runsForScenario(name) { return DASHBOARD_DATA.runs.filter((run) => run.scenario.name === name).sort((a, b) => a.timestamp_utc.localeCompare(b.timestamp_utc)); }
    function latestSuccessfulByWorld(runs) {
      const best = new Map();
      for (const run of runs) for (const result of run.results) if (result.status === "ok") best.set(result.solver + "|" + run.scenario.world_count, { run, result });
      return [...best.values()].sort((a, b) => a.run.scenario.world_count - b.run.scenario.world_count);
    }
    function renderTabs(names, activeName) {
      const tabBar = document.getElementById("tab-bar"); tabBar.innerHTML = "";
      for (const name of names) { const button = document.createElement("button"); button.className = "tab" + (name === activeName ? " active" : ""); button.textContent = name; button.type = "button"; button.addEventListener("click", () => renderScenario(name)); tabBar.appendChild(button); }
    }
    function setSummaryGrid(elementId, items) {
      const container = document.getElementById(elementId); container.innerHTML = "";
      for (const item of items) { const wrapper = document.createElement("div"); wrapper.className = "summary-item"; wrapper.innerHTML = "<div class=\\"summary-label\\">" + item.label + "</div><div class=\\"summary-value\\">" + item.value + "</div>"; container.appendChild(wrapper); }
    }
    function renderSweepPlot(elementId, entries, valueKey, yTitle) {
      const tracesBySolver = new Map();
      for (const entry of entries) {
        const solver = entry.result.solver;
        if (!tracesBySolver.has(solver)) tracesBySolver.set(solver, { x: [], y: [], customdata: [] });
        const series = tracesBySolver.get(solver);
        series.x.push(entry.run.scenario.world_count);
        series.y.push(entry.result[valueKey]);
        series.customdata.push([entry.run.timestamp_utc, entry.result.mean_frame_ms, entry.result.tuned_substeps, entry.result.max_contacts]);
      }
      const traces = [...tracesBySolver.entries()].map(([solver, series]) => ({ type: "scatter", mode: "lines+markers", name: solver, x: series.x, y: series.y, customdata: series.customdata, hovertemplate: "<b>%{fullData.name}</b><br>Worlds: %{x}<br>Value: %{y:.3f}<br>Run: %{customdata[0]}<br>Frame ms: %{customdata[1]:.3f}<br>Substeps: %{customdata[2]}<br>Max contacts: %{customdata[3]}<extra></extra>" }));
      Plotly.newPlot(elementId, traces, { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(255,255,255,0.55)", margin: { l: 60, r: 20, t: 20, b: 60 }, xaxis: { title: "World count", type: "log" }, yaxis: { title: yTitle, rangemode: "tozero" }, legend: { orientation: "h" } }, { responsive: true, displayModeBar: false });
    }
    function renderRunsTable(runs) {
      const tbody = document.getElementById("runs-table-body"); tbody.innerHTML = "";
      for (const run of [...runs].reverse()) for (const result of run.results) { const row = document.createElement("tr"); const statusClass = result.status === "ok" ? "status-ok" : "status-error"; row.innerHTML = "<td class=\\"mono\\">" + formatTimestamp(run.timestamp_utc) + "</td><td class=\\"mono\\">" + (run.git.commit_short || "n/a") + "</td><td class=\\"mono\\">" + run.scenario.world_count + "</td><td class=\\"mono\\">" + result.solver + "</td><td><span class=\\"status " + statusClass + "\\">" + result.status + "</span></td><td class=\\"mono\\">" + formatNumber(result.frames_per_second, 2) + "</td><td class=\\"mono\\">" + formatNumber(result.gpu_memory_after_warmup_mib, 1) + "</td><td class=\\"mono\\">" + (result.tuned_substeps || "n/a") + "</td><td class=\\"mono\\">" + (result.max_contacts || "0") + "</td><td>" + (result.error || "") + "</td>"; tbody.appendChild(row); }
    }
    function renderScenario(name) {
      const runs = runsForScenario(name); const latestRun = runs[runs.length - 1]; const sweepEntries = latestSuccessfulByWorld(runs);
      renderTabs(scenarioNamesFromRuns(DASHBOARD_DATA.runs), name);
      document.getElementById("empty-state").hidden = true; document.getElementById("scenario-view").hidden = false;
      document.getElementById("scenario-title").textContent = name; document.getElementById("scenario-description").textContent = latestRun.scenario.description;
      const image = document.getElementById("scenario-image"); image.src = latestRun.scenario.screenshot_relative_path || ""; image.hidden = !latestRun.scenario.screenshot_relative_path;
      setSummaryGrid("scenario-summary", [{ label: "Latest run", value: formatTimestamp(latestRun.timestamp_utc) }, { label: "Commit", value: latestRun.git.commit || "n/a" }, { label: "Branch", value: latestRun.git.branch || "n/a" }, { label: "Host", value: latestRun.environment.hostname || "n/a" }, { label: "Device", value: latestRun.environment.device || "n/a" }, { label: "Sweep points", value: String(new Set(runs.map((run) => run.scenario.world_count)).size) }]);
      setSummaryGrid("scenario-notes", [{ label: "Measure frames", value: String(latestRun.benchmark.measure_frames) }, { label: "Warmup frames", value: String(latestRun.benchmark.warmup_frames) }, { label: "Frame dt", value: formatNumber(latestRun.scenario.frame_dt, 6) }, { label: "Scenario setup s", value: formatNumber(latestRun.benchmark.scenario_setup_seconds, 2) }, { label: "Screenshot", value: latestRun.scenario.screenshot_relative_path || "n/a" }, { label: "Generated", value: formatTimestamp(DASHBOARD_DATA.generated_at_utc) }]);
      renderSweepPlot("fps-plot", sweepEntries, "frames_per_second", "Frames / second");
      renderSweepPlot("memory-plot", sweepEntries, "gpu_memory_after_warmup_mib", "GPU memory [MiB]");
      renderSweepPlot("throughput-plot", sweepEntries, "sim_seconds_per_second", "Sim seconds / wall second");
      renderRunsTable(runs);
    }
    function initializeDashboard() {
      const names = scenarioNamesFromRuns(DASHBOARD_DATA.runs);
      document.getElementById("footer-text").textContent = "Dashboard generated " + formatTimestamp(DASHBOARD_DATA.generated_at_utc) + " from " + DASHBOARD_DATA.runs.length + " logged benchmark run(s).";
      if (names.length === 0) { document.getElementById("empty-state").hidden = false; document.getElementById("scenario-view").hidden = true; renderTabs([], null); return; }
      renderScenario(names[0]);
    }
    initializeDashboard();
  </script>
</body>
</html>
"""
    return template.replace("__DASHBOARD_DATA__", embedded_payload)


def _regenerate_index_and_dashboard() -> None:
    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _format_timestamp_utc(),
        "runs": _load_run_logs(RUNS_ROOT),
    }
    _write_json(INDEX_PATH, index_payload)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(_render_dashboard_html(index_payload), encoding="utf-8")


def _parse_world_sweep(args: argparse.Namespace) -> list[int]:
    if args.world_count is not None:
        return [args.world_count]
    if args.world_sweep:
        return [int(item.strip()) for item in args.world_sweep.split(",") if item.strip()]
    return list(DEFAULT_WORLD_SWEEP)


def _run_scenarios(args: argparse.Namespace) -> None:
    scenario_names = args.scenarios or list(SCENARIOS)
    solver_names = args.solvers or list(DEFAULT_SOLVERS)

    resolved_solver_classes: dict[str, type[Any] | None] = {}
    for solver_name in solver_names:
        try:
            resolved_solver_classes[solver_name] = _resolve_solver_class(solver_name)
        except AttributeError:
            resolved_solver_classes[solver_name] = None

    available_solver_classes = [solver_cls for solver_cls in resolved_solver_classes.values() if solver_cls is not None]
    if not available_solver_classes:
        raise RuntimeError("No requested solvers are available in this checkout.")

    for scenario_name in scenario_names:
        definition = SCENARIOS[scenario_name]
        print(f"[benchmark] tuning scenario={scenario_name}")
        tuning_build = _build_scenario(definition, available_solver_classes, definition.tuning_world_count)

        tuning_results: dict[str, SolverTuningResult] = {}
        for solver_name in solver_names:
            solver_cls = resolved_solver_classes[solver_name]
            if solver_cls is None:
                continue
            tuning_result = _tune_solver_for_scenario(
                scenario=definition,
                solver_name=solver_name,
                solver_cls=solver_cls,
                scenario_build=tuning_build,
            )
            tuning_results[solver_name] = tuning_result
            print(
                "[benchmark] tuned "
                f"{scenario_name} {solver_name} substeps={tuning_result.sim_substeps} "
                f"config={tuning_result.solver_kwargs}"
            )

        screenshot_solver_name = next(iter(tuning_results))
        screenshot_solver_cls = resolved_solver_classes[screenshot_solver_name]
        assert screenshot_solver_cls is not None
        screenshot_build = _build_scenario(definition, available_solver_classes, definition.screenshot_world_count)
        screenshot_path = _take_scenario_screenshot(
            scenario_name=scenario_name,
            scenario_build=screenshot_build,
            solver_name=screenshot_solver_name,
            solver_cls=screenshot_solver_cls,
            tuning=tuning_results[screenshot_solver_name],
        )
        print(f"[benchmark] screenshot saved to {BENCHMARK_ROOT / 'dashboard' / screenshot_path}")

        world_sweep = _parse_world_sweep(args)
        for world_count in world_sweep:
            print(f"[benchmark] scenario={scenario_name} worlds={world_count}")
            scenario_setup_start = time.perf_counter()
            scenario_build = _build_scenario(definition, available_solver_classes, world_count)
            scenario_build.screenshot_relative_path = screenshot_path
            scenario_setup_seconds = time.perf_counter() - scenario_setup_start

            results: list[dict[str, Any]] = []
            for solver_name in solver_names:
                solver_cls = resolved_solver_classes.get(solver_name)
                tuning = tuning_results.get(solver_name)
                if tuning is None:
                    results.append(
                        {
                            "solver": solver_name,
                            "status": "error",
                            "error": f"{solver_name} unavailable for tuning.",
                            "setup_seconds": 0.0,
                        }
                    )
                    continue
                print(f"[benchmark] running {solver_name} worlds={world_count}")
                result = _run_solver_benchmark(
                    scenario_build=scenario_build,
                    solver_name=solver_name,
                    solver_cls=solver_cls,
                    tuning=tuning,
                    warmup_frames=args.warmup_frames,
                    measure_frames=args.measure_frames,
                )
                results.append(result)

            metadata = _collect_run_metadata(REPO_ROOT)
            run_log = _build_run_payload(
                metadata,
                scenario_build=scenario_build,
                world_count=world_count,
                warmup_frames=args.warmup_frames,
                measure_frames=args.measure_frames,
                solvers=solver_names,
                scenario_setup_seconds=scenario_setup_seconds,
                tuning=tuning_results,
                results=results,
            )
            timestamp_label = run_log["timestamp_utc"].replace(":", "").replace("-", "")
            commit_short = run_log["git"]["commit_short"] or "unknown"
            output_name = f"{timestamp_label}_{scenario_name}_{world_count}_{commit_short}.json"
            _write_json(RUNS_ROOT / output_name, run_log)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Newton solvers across scenarios and world-count sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=[],
        choices=sorted(SCENARIOS),
        help="Scenario name to run. Can be specified multiple times.",
    )
    parser.add_argument(
        "--solver",
        action="append",
        dest="solvers",
        default=[],
        help="Solver class name from newton.solvers. Can be specified multiple times.",
    )
    parser.add_argument("--world-count", type=int, default=None, help="Run a single world count instead of a sweep.")
    parser.add_argument(
        "--world-sweep",
        type=str,
        default=",".join(str(item) for item in DEFAULT_WORLD_SWEEP),
        help="Comma-separated world counts for sweep runs.",
    )
    parser.add_argument("--warmup-frames", type=int, default=90, help="Warmup frames excluded from measurements.")
    parser.add_argument("--measure-frames", type=int, default=120, help="Measured frames per solver.")
    parser.add_argument("--device", type=str, default=None, help="Optional Warp device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print the available scenario names and exit.")
    return parser


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        for scenario in SCENARIOS.values():
            print(f"{scenario.name}: {scenario.description}")
        return

    if args.device:
        wp.set_device(args.device)

    _run_scenarios(args)
    _regenerate_index_and_dashboard()
    print(f"Wrote benchmark index to {INDEX_PATH}")
    print(f"Wrote dashboard to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
