# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a calibrated YAM arm model against held-out ABC-130k arm logs.

Builds the submitted model through ``build_model``/``make_solver`` and runs this
verifier's own multiple-shooting rollout: 1 s windows every 0.5 s, each started
from the measured state and driven by the logged joint commands through the
model's joint position targets, delayed by ``PARAMS["command_delay"]``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

HERE = Path(__file__).resolve().parent
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "abc_arm"
SIM_DT, HORIZON, STRIDE, GRIPPER_TRAVEL = 0.002, 1.0, 0.5, 0.0475
ARM_JOINTS = [f"left_joint{j}" for j in range(1, 7)]
# Held-out limit [rad]: mean over logs of the mean window RMSE of the six arm joint angles.
THRESHOLDS = {"heldout_rad": 0.023}
# Reference scores: starter 0.0385, gains from a torque regression 0.0256, coordinate-descent fit 0.0212.
STARTER_RAD = 0.038


def load(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_arm_replay", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _indices(model: newton.Model):
    names = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    starts = model.joint_q_start.numpy()
    per_world = model.joint_coord_count // model.world_count
    arm = np.array([starts[names.index(name)] for name in ARM_JOINTS])
    fingers = np.array([starts[names.index(name)] for name in ("left_left_finger", "left_right_finger")])
    return arm, fingers, per_world


def integrity(model: newton.Model, reference: newton.Model, worlds: int, delay: float) -> dict:
    """The arm's kinematics, mass properties' validity, gravity, and the command latency range."""
    per_world = reference.joint_count
    x_p = model.joint_X_p.numpy()[:per_world]
    axes = model.joint_axis.numpy()[: reference.joint_dof_count]
    return {
        "one_arm_per_world": model.world_count == worlds and model.joint_count == per_world * worlds,
        "kinematics": bool(
            np.allclose(x_p, reference.joint_X_p.numpy(), atol=1e-5)
            and np.allclose(axes, reference.joint_axis.numpy(), atol=1e-5)
            and np.array_equal(model.joint_type.numpy()[:per_world], reference.joint_type.numpy())
        ),
        "gravity": bool(np.allclose(model.gravity.numpy().reshape(-1)[:3], [0.0, 0.0, -9.81], atol=1e-4)),
        "positive_mass": bool(np.all(model.body_mass.numpy() >= 0.0)),
        "valid_inertia": bool(np.all(np.linalg.eigvalsh(model.body_inertia.numpy()) >= -1e-9)),
        "command_delay": 0.0 <= delay <= 0.2,
    }


def evaluate(module, logs: list[np.ndarray], reference: newton.Model) -> tuple[list[float], dict]:
    segments = [
        (data, start, index)
        for index, data in enumerate(logs)
        for start in np.arange(0.0, data[-1, 0] - HORIZON, STRIDE)
    ]
    worlds = len(segments)
    delay = float(module.PARAMS.get("command_delay", 0.0))
    model = module.build_model(worlds)
    solver = module.make_solver(model)
    checks = integrity(model, reference, worlds, delay)
    arm, fingers, coords = _indices(model)
    steps = round(HORIZON / SIM_DT)
    times = np.arange(steps + 1) * SIM_DT
    q0 = model.joint_q.numpy().reshape(worlds, coords)
    qd0 = np.zeros_like(q0)
    targets = np.repeat(q0[None], steps, axis=0)
    for w, (data, start, _) in enumerate(segments):
        row = np.searchsorted(data[:, 0], start)
        q0[w, arm], qd0[w, arm] = data[row, 1:7], data[row, 7:13]
        opening = np.clip(data[row, 25], 0.0, 1.0) * GRIPPER_TRAVEL
        q0[w, fingers] = (opening, -opening)
        rows = np.clip(
            np.searchsorted(data[:, 0], data[row, 0] + times[:-1] - delay, side="right") - 1, 0, len(data) - 1
        )
        targets[:, w, arm] = data[rows, 19:25]
        targets[:, w, fingers[0]] = np.clip(data[rows, 26], 0.0, 1.0) * GRIPPER_TRAVEL
    state_0, state_1, control = model.state(), model.state(), model.control()
    state_0.joint_q.assign(q0.reshape(-1).astype(np.float32))
    state_0.joint_qd.assign(qd0.reshape(-1).astype(np.float32))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    predicted = [q0[:, arm].copy()]
    for k in range(steps):
        control.joint_target_q.assign(targets[k].reshape(-1).astype(np.float32))
        solver.step(state_0, state_1, control, None, SIM_DT)
        state_0, state_1 = state_1, state_0
        predicted.append(state_0.joint_q.numpy().reshape(worlds, coords)[:, arm])
    predicted = np.asarray(predicted)
    measured = np.stack(
        [
            np.stack([np.interp(start + times, data[:, 0], data[:, 1 + j]) for j in range(6)], axis=-1)
            for data, start, _ in segments
        ],
        axis=1,
    )
    errors = np.sqrt(np.mean((predicted - measured) ** 2, axis=(0, 2)))
    owners = np.array([index for _, _, index in segments])
    per_log = [
        float(np.mean(errors[owners == i])) if np.isfinite(errors[owners == i]).all() else float("inf")
        for i in range(len(logs))
    ]
    return per_log, checks


def verify(path: Path) -> dict:
    module = load(path)
    started = time.perf_counter()
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(PRIVATE / "yam_arm.xml"))
    reference = builder.finalize()
    files = sorted(PRIVATE.glob("*.csv"))
    per_log, checks = evaluate(module, [np.loadtxt(f, delimiter=",", skiprows=1) for f in files], reference)
    metrics = {"heldout_rad": float(np.mean(per_log))}
    failed = [k for k, v in checks.items() if not v]
    success = not failed and all(metrics[k] <= v for k, v in THRESHOLDS.items())
    return {
        "task": "abc_arm",
        "success": bool(success),
        "integrity": not failed,
        "failed_checks": failed + [k for k, v in THRESHOLDS.items() if not metrics[k] <= v],
        "metrics": metrics,
        "normalized_worst": {k: metrics[k] / v for k, v in THRESHOLDS.items()},
        "details": {
            "per_log": dict(zip([f.stem for f in files], per_log, strict=True)),
            "seconds": time.perf_counter() - started,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING
    result = verify(args.script.resolve())
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {"success": result["success"], "metrics": result["metrics"], "failed_checks": result["failed_checks"]}
        )
    )


if __name__ == "__main__":
    main()
