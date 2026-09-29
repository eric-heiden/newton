# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a calibrated double-pendulum model against held-out real-robot recordings.

Builds the submitted model through ``build_model``/``make_solver`` and runs this
verifier's own multiple-shooting rollout: 0.5 s windows every 0.25 s, each
started from the measured state and driven by the measured motor torques.
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
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "dp_real"
SIM_DT, HORIZON, STRIDE, L1 = 0.002, 0.5, 0.25, 0.2
# Held-out error limits [rad]: mean window RMSE of the joint angles.
THRESHOLDS = {"heldout_10_rad": 0.060, "heldout_11_rad": 0.150}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_pendulum", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def evaluate(module, data: np.ndarray) -> tuple[float, dict]:
    starts = np.arange(data[0, 0], data[-1, 0] - HORIZON, STRIDE)
    worlds = len(starts)
    model = module.build_model(worlds)
    solver = module.make_solver(model)
    checks = {
        "bodies": model.body_count == 2 * worlds,
        "dofs": model.joint_dof_count == 2 * worlds,
        "revolute": bool(np.all(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))),
        "gravity": bool(np.allclose(model.gravity.numpy().reshape(-1)[:3], [0.0, 0.0, -9.81], atol=1e-4)),
        "elbow_offset": bool(
            np.isclose(np.linalg.norm(model.joint_X_p.numpy()[1, :3]), L1, atol=1e-4)
            and np.isclose(np.linalg.norm(model.joint_X_p.numpy()[0, :3]), 0.0, atol=1e-4)
        ),
        "positive_mass": bool(np.all(model.body_mass.numpy() > 0.0)),
        "valid_inertia": bool(np.all(np.linalg.eigvalsh(model.body_inertia.numpy()) >= -1e-9)),
    }
    state_0, state_1, control = model.state(), model.state(), model.control()
    rows = np.searchsorted(data[:, 0], starts)
    state_0.joint_q.assign(data[rows, 1:3].reshape(-1).astype(np.float32))
    state_0.joint_qd.assign(data[rows, 3:5].reshape(-1).astype(np.float32))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    steps = round(HORIZON / SIM_DT)
    times = np.arange(steps + 1) * SIM_DT
    held = np.searchsorted(data[:, 0], data[rows, 0][None, :] + times[:-1, None], side="right") - 1
    torques = data[held, 5:7].reshape(steps, -1).astype(np.float32)
    predicted = [data[rows, 1:3]]
    for k in range(steps):
        control.joint_f.assign(torques[k])
        solver.step(state_0, state_1, control, None, SIM_DT)
        state_0, state_1 = state_1, state_0
        predicted.append(state_0.joint_q.numpy().reshape(-1, 2))
    predicted = np.asarray(predicted)
    t0 = data[rows, 0]
    measured = np.stack(
        [np.stack([np.interp(t + times, data[:, 0], data[:, 1 + j]) for j in range(2)], axis=-1) for t in t0], axis=1
    )
    errors = np.sqrt(np.mean((predicted - measured) ** 2, axis=(0, 2)))
    error = float(np.mean(errors)) if np.isfinite(errors).all() else float("inf")
    return error, checks


def verify(path: Path) -> dict:
    module = load(path)
    started = time.perf_counter()
    metrics, checks = {}, {}
    for name, file in (("heldout_10_rad", PRIVATE / "heldout_10.csv"), ("heldout_11_rad", PRIVATE / "heldout_11.csv")):
        metrics[name], case_checks = evaluate(module, np.loadtxt(file, delimiter=",", skiprows=1))
        checks.update({f"{name}.{k}": v for k, v in case_checks.items()})
    train = {}
    for file in sorted(path.parent.glob("train_*.csv")):
        train[file.stem], _ = evaluate(module, np.loadtxt(file, delimiter=",", skiprows=1))
    integrity = all(checks.values())
    success = integrity and all(metrics[k] <= v for k, v in THRESHOLDS.items())
    return {
        "task": "dp_real",
        "success": bool(success),
        "integrity": integrity,
        "failed_checks": [k for k, v in checks.items() if not v],
        "metrics": {**metrics, "train_mean_rad": float(np.mean(list(train.values()))) if train else None},
        "normalized_worst": {k: metrics[k] / v for k, v in THRESHOLDS.items()},
        "details": {"train": train, "seconds": time.perf_counter() - started},
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
