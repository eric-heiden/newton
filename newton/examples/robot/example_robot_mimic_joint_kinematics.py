# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

MIMIC_SPECS = (
    ("right_inner", "right_driver", -0.52, 0.46),
    ("right_tip", "right_driver", 0.36, 0.18),
    ("left_driver", "right_driver", 0.0, -1.0),
    ("left_inner", "right_driver", -0.52, -0.46),
    ("left_tip", "right_driver", 0.36, -0.18),
)


def _add_link(builder: newton.ModelBuilder, label: str) -> int:
    return builder.add_link(mass=0.01, label=label)


def build_robotiq_style_kinematic_model() -> newton.Model:
    """Build a small Robotiq-2F85-like two-finger kinematic tree.

    The real menagerie Robotiq asset also has tendon and equality-constraint
    details. This minimal model isolates the mimic-joint part of the question:
    all follower knuckles are scalar joints driven by one gripper command.
    """

    builder = newton.ModelBuilder()

    body = {
        name: _add_link(builder, name)
        for name in (
            "right_driver",
            "right_inner",
            "right_tip",
            "left_driver",
            "left_inner",
            "left_tip",
        )
    }

    joints: dict[str, int] = {}
    joints["right_driver"] = builder.add_joint_revolute(
        parent=-1,
        child=body["right_driver"],
        axis=(0.0, 0.0, 1.0),
        parent_xform=((0.0, -0.045, 0.0), (0.0, 0.0, 0.0, 1.0)),
        label="right_driver_joint",
    )

    parent_by_joint = {
        "right_inner": body["right_driver"],
        "right_tip": body["right_inner"],
        "left_driver": -1,
        "left_inner": body["left_driver"],
        "left_tip": body["left_inner"],
    }
    parent_xform_by_joint = {
        "right_inner": ((0.0, 0.030, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "right_tip": ((0.0, 0.026, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "left_driver": ((0.0, 0.045, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "left_inner": ((0.0, -0.030, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "left_tip": ((0.0, -0.026, 0.0), (0.0, 0.0, 0.0, 1.0)),
    }

    for joint_name, leader_name, offset, multiplier in MIMIC_SPECS:
        joint_args = {
            "parent": parent_by_joint[joint_name],
            "child": body[joint_name],
            "axis": (0.0, 0.0, 1.0),
            "parent_xform": parent_xform_by_joint[joint_name],
            "label": f"{joint_name}_joint",
        }
        joints[joint_name] = builder.add_joint_mimic(
            leader_joint=joints[leader_name],
            coef0=offset,
            coef1=multiplier,
            mimic_type=newton.JointType.REVOLUTE,
            **joint_args,
        )

    builder.add_articulation(list(joints.values()), label="robotiq_style_mimic_kinematics")
    return builder.finalize()


def _set_joint_q(model: newton.Model, state: newton.State, values_by_label: dict[str, float]) -> None:
    q = state.joint_q.numpy()
    q_start = model.joint_q_start.numpy()
    for label, value in values_by_label.items():
        joint_idx = model.joint_label.index(label)
        q[q_start[joint_idx]] = value
    state.joint_q.assign(q)


def apply_mimic_joint_coordinate(model: newton.Model, state: newton.State, driver_q: float) -> None:
    _set_joint_q(model, state, {"right_driver_joint": driver_q})


def _time_call(fn, *, samples: int, repeats: int) -> dict[str, float]:
    timings = []
    for _ in range(repeats):
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(samples):
            fn()
        wp.synchronize()
        timings.append((time.perf_counter() - start) / samples)
    return {
        "mean_us": statistics.fmean(timings) * 1.0e6,
        "median_us": statistics.median(timings) * 1.0e6,
        "min_us": min(timings) * 1.0e6,
        "max_us": max(timings) * 1.0e6,
    }


def _model_stats(model: newton.Model) -> dict[str, int]:
    joint_types = model.joint_type.numpy()
    return {
        "body_count": model.body_count,
        "joint_count": model.joint_count,
        "joint_dof_count": model.joint_dof_count,
        "mimic_joint_count": int(np.count_nonzero(joint_types == int(newton.JointType.MIMIC))),
        "mimic_constraint_count": model.constraint_mimic_count,
    }


def run_benchmark(*, samples: int, repeats: int, driver_q: float, device: str | None = None) -> dict[str, Any]:
    if device is not None:
        wp.set_device(device)

    mimic_model = build_robotiq_style_kinematic_model()

    mimic_state = mimic_model.state()
    apply_mimic_joint_coordinate(mimic_model, mimic_state, driver_q)

    newton.eval_fk(mimic_model, mimic_state.joint_q, mimic_state.joint_qd, mimic_state)
    wp.synchronize()

    mimic_fk = _time_call(
        lambda: newton.eval_fk(mimic_model, mimic_state.joint_q, mimic_state.joint_qd, mimic_state),
        samples=samples,
        repeats=repeats,
    )

    return {
        "case": "robotiq_style_mimic_kinematics",
        "samples": samples,
        "repeats": repeats,
        "driver_q": driver_q,
        "device": str(wp.get_device()),
        "mimic_joints": {
            "stats": _model_stats(mimic_model),
            "eval_fk_timing": mimic_fk,
        },
    }


def write_gif(path: Path, *, frames: int = 36) -> None:
    import matplotlib.animation as animation  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    model = build_robotiq_style_kinematic_model()
    state = model.state()

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.02, 0.16)
    ax.axis("off")
    (line,) = ax.plot([], [], "-o", color="#2456a6", lw=3, ms=5)

    def update(frame: int):
        driver_q = 0.55 * (0.5 - 0.5 * math.cos(2.0 * math.pi * frame / max(frames - 1, 1)))
        apply_mimic_joint_coordinate(model, state, driver_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.synchronize()
        body_q = state.body_q.numpy()
        points = [(0.0, 0.0)]
        for idx in range(model.body_count):
            points.append((float(body_q[idx][0]), float(body_q[idx][1] + 0.08)))
        x, y = zip(*points, strict=True)
        line.set_data(x, y)
        return (line,)

    path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=60, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=16))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--driver-q", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--gif-out", type=Path, default=None)
    args = parser.parse_args()

    result = run_benchmark(samples=args.samples, repeats=args.repeats, driver_q=args.driver_q, device=args.device)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.gif_out is not None:
        write_gif(args.gif_out)


if __name__ == "__main__":
    main()
