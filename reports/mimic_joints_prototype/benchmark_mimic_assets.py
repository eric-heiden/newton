# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton

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


def _build_model(asset: dict[str, str]) -> newton.Model:
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
        builder.add_mjcf(
            source,
            parse_meshes=False,
            parse_visuals=False,
            ignore_inertial_definitions=True,
            collapse_fixed_joints=False,
        )
    else:
        raise ValueError(f"Unsupported asset kind {kind!r}")
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


def _write_gif(model: newton.Model, state: newton.State, path: Path, *, frames: int = 32) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    scat = ax.scatter([], [], s=10, color="#1f77b4")

    def update(frame: int):
        _set_sample_pose(model, state, 2.0 * math.pi * frame / max(frames - 1, 1))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.synchronize()
        body_q = state.body_q.numpy()
        if len(body_q) == 0:
            pts = np.zeros((1, 2), dtype=float)
        else:
            xyz = np.asarray(body_q[:, :3], dtype=float)
            pts = xyz[:, [0, 2]]
            if pts.shape[0] > 250:
                pts = pts[np.linspace(0, pts.shape[0] - 1, 250).astype(int)]
        scat.set_offsets(pts)
        if pts.size:
            pad = 0.05 + 0.05 * max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
            ax.set_xlim(float(np.min(pts[:, 0])) - pad, float(np.max(pts[:, 0])) + pad)
            ax.set_ylim(float(np.min(pts[:, 1])) - pad, float(np.max(pts[:, 1])) + pad)
        return (scat,)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=70, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=14))
    plt.close(fig)


def run(*, samples: int, repeats: int, device: str | None, gif_dir: Path | None) -> dict[str, Any]:
    if device is not None:
        wp.set_device(device)

    results: dict[str, Any] = {
        "device": str(wp.get_device()),
        "newton_file": str(Path(newton.__file__).resolve()),
        "samples": samples,
        "repeats": repeats,
        "assets": {},
    }
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
            if gif_dir is not None and name in {
                "synthetic_robotiq_urdf",
                "robotiq_2f85_v4_usd",
                "leap_hand_right_mjcf",
                "shadow_hand_right_mjcf",
            }:
                gif_path = gif_dir / f"{name}.gif"
                _write_gif(model, state, gif_path)
                asset_result["gif"] = str(gif_path)
            results["assets"][name] = asset_result
        except Exception as exc:
            results["assets"][name] = {
                "kind": asset["kind"],
                "source": _source_label(asset),
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--gif-dir", type=Path, default=None)
    args = parser.parse_args()

    result = run(samples=args.samples, repeats=args.repeats, device=args.device, gif_dir=args.gif_dir)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
