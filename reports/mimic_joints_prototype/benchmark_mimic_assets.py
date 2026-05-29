# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverSemiImplicit, SolverVBD, SolverXPBD

if not hasattr(wp.config, "log_level"):
    wp.config.log_level = getattr(wp, "LOG_INFO", 20)
for _log_name, _log_value in (("LOG_DEBUG", 10), ("LOG_INFO", 20), ("LOG_WARNING", 30), ("LOG_ERROR", 40)):
    if not hasattr(wp, _log_name):
        setattr(wp, _log_name, _log_value)

SYNTHETIC_ROBOTIQ_URDF = """
<robot name="robotiq_mimic_synthetic">
  <link name="base"/>
  <link name="driver"/>
  <link name="right_inner"/>
  <link name="right_tip"/>
  <link name="left_driver"/>
  <link name="left_inner"/>
  <link name="left_tip"/>
  <joint name="right_driver_joint" type="revolute">
    <parent link="base"/><child link="driver"/>
    <origin xyz="0 -0.045 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-0.8" upper="0.8"/>
  </joint>
  <joint name="right_inner_joint" type="revolute">
    <parent link="driver"/><child link="right_inner"/>
    <origin xyz="0 0.030 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1.0" upper="1.0"/>
    <mimic joint="right_driver_joint" multiplier="0.46" offset="-0.52"/>
  </joint>
  <joint name="right_tip_joint" type="revolute">
    <parent link="right_inner"/><child link="right_tip"/>
    <origin xyz="0 0.026 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1.0" upper="1.0"/>
    <mimic joint="right_driver_joint" multiplier="0.18" offset="0.36"/>
  </joint>
  <joint name="left_driver_joint" type="revolute">
    <parent link="base"/><child link="left_driver"/>
    <origin xyz="0 0.045 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1.0" upper="1.0"/>
    <mimic joint="right_driver_joint" multiplier="-1.0" offset="0.0"/>
  </joint>
  <joint name="left_inner_joint" type="revolute">
    <parent link="left_driver"/><child link="left_inner"/>
    <origin xyz="0 -0.030 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1.0" upper="1.0"/>
    <mimic joint="right_driver_joint" multiplier="-0.46" offset="-0.52"/>
  </joint>
  <joint name="left_tip_joint" type="revolute">
    <parent link="left_inner"/><child link="left_tip"/>
    <origin xyz="0 -0.026 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1.0" upper="1.0"/>
    <mimic joint="right_driver_joint" multiplier="-0.18" offset="0.36"/>
  </joint>
</robot>
"""


ASSETS = (
    {
        "name": "synthetic_robotiq_urdf",
        "kind": "urdf",
        "source": SYNTHETIC_ROBOTIQ_URDF,
    },
    {
        "name": "robotiq_2f85_v4_usd",
        "kind": "usd",
        "source": "/home/horde/apps/newton-assets/robotiq_2f85_v4/usd_structured/Dual_wrist_camera.usda",
    },
    {
        "name": "robotiq_2f85_v4_mjcf",
        "kind": "mjcf",
        "source": "/home/horde/.cache/newton/mujoco_menagerie_robotiq_2f85_v4_f233abbd_feadf76d/robotiq_2f85_v4/2f85.xml",
    },
    {
        "name": "leap_hand_right_mjcf",
        "kind": "mjcf",
        "source": "/home/horde/.cache/newton/mujoco_menagerie_leap_hand_d94a1630_feadf76d/leap_hand/right_hand.xml",
    },
    {
        "name": "shadow_hand_right_mjcf",
        "kind": "mjcf",
        "source": "/home/horde/.cache/newton/mujoco_menagerie_shadow_hand_46b0dcac_feadf76d/shadow_hand/right_hand.xml",
    },
    {
        "name": "unitree_g1_with_hands_urdf",
        "kind": "urdf",
        "source": "/home/horde/apps/newton-assets/unitree_g1/urdf/g1_29dof_with_hand_rev_1_0.urdf",
    },
)

VIDEO_ASSET_NAMES = {
    "robotiq_2f85_v4_usd",
    "robotiq_2f85_v4_mjcf",
    "leap_hand_right_mjcf",
    "shadow_hand_right_mjcf",
    "unitree_g1_with_hands_urdf",
}


def _mimic_type_value() -> int | None:
    mimic_type = getattr(newton.JointType, "MIMIC", None)
    return int(mimic_type) if mimic_type is not None else None


def _model_stats(model: newton.Model) -> dict[str, int]:
    joint_types = model.joint_type.numpy()
    mimic_type = _mimic_type_value()
    mimic_joint_count = int(np.count_nonzero(joint_types == mimic_type)) if mimic_type is not None else 0
    return {
        "body_count": int(model.body_count),
        "shape_count": int(model.shape_count),
        "joint_count": int(model.joint_count),
        "joint_dof_count": int(model.joint_dof_count),
        "joint_coord_count": int(model.joint_coord_count),
        "mimic_joint_count": mimic_joint_count,
        "mimic_constraint_count": int(getattr(model, "constraint_mimic_count", 0)),
        "equality_constraint_count": int(getattr(model, "equality_constraint_count", 0)),
    }


def _build_model(asset: dict[str, str], *, rich_visuals: bool = False, color: bool = False) -> newton.Model:
    builder = newton.ModelBuilder()
    kind = asset["kind"]
    source = asset["source"]
    if kind != "urdf" and not Path(source).exists():
        raise FileNotFoundError(source)
    if kind == "urdf":
        builder.add_urdf(source, ignore_inertial_definitions=True)
    elif kind == "usd":
        builder.add_usd(source)
    elif kind == "mjcf":
        if rich_visuals:
            builder.add_mjcf(source, ignore_inertial_definitions=True, collapse_fixed_joints=False)
        else:
            builder.add_mjcf(
                source,
                parse_meshes=False,
                parse_visuals=False,
                ignore_inertial_definitions=True,
                collapse_fixed_joints=False,
            )
    else:
        raise ValueError(f"Unsupported asset kind {kind!r}")
    if color:
        builder.color()
    try:
        return builder.finalize()
    except TypeError:
        return builder.finalize(skip_validation_joints=True)
    except ValueError:
        return builder.finalize(skip_validation_joints=True)


def _source_label(asset: dict[str, str]) -> str:
    source = asset["source"]
    if asset["name"] == "synthetic_robotiq_urdf":
        return "inline synthetic URDF"
    return source


def _apply_mimic_constraints(model: newton.Model, q: np.ndarray) -> None:
    if getattr(model, "constraint_mimic_count", 0) == 0:
        return
    q_start = model.joint_q_start.numpy()
    for follower, leader, coef0, coef1 in zip(
        model.constraint_mimic_joint0.numpy(),
        model.constraint_mimic_joint1.numpy(),
        model.constraint_mimic_coef0.numpy(),
        model.constraint_mimic_coef1.numpy(),
        strict=True,
    ):
        q[q_start[int(follower)]] = float(coef0) + float(coef1) * q[q_start[int(leader)]]


def _set_sample_pose(model: newton.Model, state: newton.State, phase: float) -> None:
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q.fill(0.0)
    qd.fill(0.0)

    joint_types = model.joint_type.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    mimic_type = _mimic_type_value()
    scalar_types = {int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)}
    for joint_idx, joint_type in enumerate(joint_types):
        if mimic_type is not None and int(joint_type) == mimic_type:
            continue
        if int(joint_type) not in scalar_types:
            continue
        if int(joint_dof_dim[joint_idx, 0] + joint_dof_dim[joint_idx, 1]) != 1:
            continue
        amp = 0.25 if int(joint_type) == int(newton.JointType.REVOLUTE) else 0.02
        q[q_start[joint_idx]] = amp * math.sin(phase + 0.07 * joint_idx)
        qd[qd_start[joint_idx]] = amp * math.cos(phase + 0.07 * joint_idx)
    _apply_mimic_constraints(model, q)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)


def _time_fk(model: newton.Model, state: newton.State, *, samples: int, repeats: int) -> dict[str, float]:
    timings = []
    for repeat in range(repeats):
        wp.synchronize()
        start = time.perf_counter()
        for sample in range(samples):
            _set_sample_pose(model, state, 2.0 * math.pi * (sample + repeat * samples) / max(samples * repeats, 1))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.synchronize()
        timings.append((time.perf_counter() - start) / samples)
    return {
        "mean_us": statistics.fmean(timings) * 1.0e6,
        "median_us": statistics.median(timings) * 1.0e6,
        "min_us": min(timings) * 1.0e6,
        "max_us": max(timings) * 1.0e6,
    }


def _solver_factories() -> tuple[tuple[str, Any], ...]:
    return (
        ("semi_implicit", SolverSemiImplicit),
        ("xpbd", SolverXPBD),
        ("featherstone", SolverFeatherstone),
        ("vbd", SolverVBD),
        ("mujoco", lambda model: SolverMuJoCo(model, disable_contacts=True)),
    )


def _time_solver_steps(
    model: newton.Model,
    *,
    samples: int,
    repeats: int,
    dt: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for solver_name, solver_factory in _solver_factories():
        try:
            state_in = model.state()
            state_out = model.state()
            control = model.control()
            contacts = model.contacts()
            _set_sample_pose(model, state_in, 0.0)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            wp.synchronize()

            solver = solver_factory(model)
            solver.step(state_in, state_out, control, contacts, dt)
            wp.synchronize()
            state_in, state_out = state_out, state_in

            timings = []
            for _repeat in range(repeats):
                wp.synchronize()
                start = time.perf_counter()
                for _sample in range(samples):
                    solver.step(state_in, state_out, control, contacts, dt)
                    state_in, state_out = state_out, state_in
                wp.synchronize()
                timings.append((time.perf_counter() - start) / samples)

            results[solver_name] = {
                "status": "ok",
                "mean_us": statistics.fmean(timings) * 1.0e6,
                "median_us": statistics.median(timings) * 1.0e6,
                "min_us": min(timings) * 1.0e6,
                "max_us": max(timings) * 1.0e6,
            }
        except Exception as exc:
            results[solver_name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def _camera_position_for_state(state: newton.State, asset_name: str) -> tuple[wp.vec3, tuple[float, float, float]]:
    body_q = state.body_q.numpy()
    if len(body_q) == 0:
        center = np.zeros(3, dtype=float)
        radius = 1.0
    else:
        xyz = np.asarray(body_q[:, :3], dtype=float)
        bounds_min = np.min(xyz, axis=0)
        bounds_max = np.max(xyz, axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        radius = max(float(np.linalg.norm(bounds_max - bounds_min)), 0.2)

    if "robotiq" in asset_name:
        offset = np.array([1.25, -1.55, 0.85], dtype=float) * max(0.85 * radius, 0.14)
    elif "unitree" in asset_name:
        offset = np.array([1.8, -2.4, 1.0], dtype=float) * max(0.55 * radius, 0.75)
        center[2] += 0.3
    else:
        offset = np.array([1.6, -2.1, 1.1], dtype=float) * max(0.8 * radius, 0.14)

    pos = center + offset
    return wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), (
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )


def _write_viewergl_video(
    viewer: Any,
    model: newton.Model,
    state: newton.State,
    path: Path,
    *,
    asset_name: str,
    frames: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    import imageio.v2 as imageio  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    viewer.set_model(model)
    viewer.renderer.draw_fps = False
    viewer.renderer.draw_wireframe = False
    viewer.renderer.draw_shadows = True
    viewer.renderer.spotlight_enabled = True
    viewer.camera.fov = 32.0

    frame_buffer: wp.array | None = None
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame_idx in range(frames):
            phase = 2.0 * math.pi * frame_idx / max(frames - 1, 1)
            _set_sample_pose(model, state, phase)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            wp.synchronize()

            camera_pos, target = _camera_position_for_state(state, asset_name)
            viewer.set_camera(camera_pos, viewer.camera.pitch, viewer.camera.yaw)
            viewer.camera.look_at(target)

            viewer.begin_frame(frame_idx / fps)
            viewer.log_state(state)
            viewer.end_frame()

            frame = viewer.get_frame(target_image=frame_buffer)
            if frame_buffer is None:
                frame_buffer = frame
            writer.append_data(np.ascontiguousarray(frame.numpy()))


def _create_viewergl(width: int, height: int) -> Any:
    import pyglet

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    pyglet.options["headless"] = True

    from newton.viewer import ViewerGL  # noqa: PLC0415

    return ViewerGL(width=width, height=height, headless=True)


def run(
    *,
    samples: int,
    repeats: int,
    device: str | None,
    video_dir: Path | None,
    video_frames: int,
    video_fps: int,
    video_width: int,
    video_height: int,
    benchmark_solvers: bool,
    solver_samples: int,
    solver_repeats: int,
    solver_dt: float,
) -> dict[str, Any]:
    if device is not None:
        wp.set_device(device)

    results: dict[str, Any] = {
        "device": str(wp.get_device()),
        "newton_file": str(Path(newton.__file__).resolve()),
        "samples": samples,
        "repeats": repeats,
        "solver_samples": solver_samples if benchmark_solvers else 0,
        "solver_repeats": solver_repeats if benchmark_solvers else 0,
        "solver_dt": solver_dt,
        "assets": {},
    }
    viewer = _create_viewergl(video_width, video_height) if video_dir is not None else None
    for asset in ASSETS:
        name = asset["name"]
        try:
            model = _build_model(asset)
            state = model.state()
            _set_sample_pose(model, state, 0.0)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            wp.synchronize()
            asset_result: dict[str, Any] = {
                "kind": asset["kind"],
                "source": _source_label(asset),
                "stats": _model_stats(model),
                "eval_fk_timing": _time_fk(model, state, samples=samples, repeats=repeats),
            }
            if benchmark_solvers:
                solver_model = _build_model(asset, color=True)
                asset_result["solver_step_timing"] = _time_solver_steps(
                    solver_model,
                    samples=solver_samples,
                    repeats=solver_repeats,
                    dt=solver_dt,
                )
            if video_dir is not None and name in VIDEO_ASSET_NAMES:
                video_model = _build_model(asset, rich_visuals=True)
                video_state = video_model.state()
                video_path = video_dir / f"{name}.mp4"
                _write_viewergl_video(
                    viewer,
                    video_model,
                    video_state,
                    video_path,
                    asset_name=name,
                    frames=video_frames,
                    fps=video_fps,
                    width=video_width,
                    height=video_height,
                )
                asset_result["video"] = str(video_path)
            results["assets"][name] = asset_result
        except Exception as exc:
            results["assets"][name] = {
                "kind": asset["kind"],
                "source": _source_label(asset),
                "error": f"{type(exc).__name__}: {exc}",
            }
    if viewer is not None:
        viewer.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-frames", type=int, default=48)
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--skip-solvers", action="store_true")
    parser.add_argument("--solver-samples", type=int, default=10)
    parser.add_argument("--solver-repeats", type=int, default=3)
    parser.add_argument("--solver-dt", type=float, default=1.0 / 240.0)
    args = parser.parse_args()

    result = run(
        samples=args.samples,
        repeats=args.repeats,
        device=args.device,
        video_dir=args.video_dir,
        video_frames=args.video_frames,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        benchmark_solvers=not args.skip_solvers,
        solver_samples=args.solver_samples,
        solver_repeats=args.solver_repeats,
        solver_dt=args.solver_dt,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
