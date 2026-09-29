# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a submitted grasp-shake scene: the cube must not slip in the gripper.

Runs the submitted ``franka_cube_shake.py`` in a fresh process through its public
``Example`` class, measures the cube pose in the hand (TCP) frame during two
shakes, and checks that the task itself (motion, grip, timestep, bodies) is
unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer

THRESHOLDS = {"drift_3cm_mm": 1.0, "drift_10cm_mm": 5.0}
SHAKES = {"3cm": (0.03, 1.0, 11.5), "10cm": (0.10, 1.0, 6.0)}
EXPECTED = {
    "sim_dt": 1.0 / 960.0,
    "frame_dt": 1.0 / 60.0,
    "cube_mass_kg": 0.04**3 * 500.0,
    "gripper_closed": -0.01,
    "finger_ke": 400.0,
    "finger_effort": 100.0,
    "phase_durations": [0.25, 1.0, 0.75, 1.5, 1.0],
}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_scene", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _relative(body_q: np.ndarray, hand: int, cube: int) -> np.ndarray:
    """Cube position expressed in the hand frame [m]."""
    hand_q, cube_q = body_q[hand], body_q[cube]
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*hand_q[3:7])), dtype=np.float64).reshape(3, 3)
    return rotation.T @ (cube_q[:3] - hand_q[:3])


def run_shake(module, amplitude: float, frequency: float, seconds: float) -> dict:
    parser = module.create_parser() if hasattr(module, "create_parser") else newton.examples.create_parser()
    args, _ = parser.parse_known_args([])
    args.viewer, args.shake_amplitude, args.shake_frequency = "null", amplitude, frequency
    example = module.Example(newton.viewer.ViewerNull(num_frames=1 << 30), args)
    checks = {
        "sim_dt": abs(example.sim_dt - EXPECTED["sim_dt"]) < 1e-9,
        "frame_dt": abs(example.frame_dt - EXPECTED["frame_dt"]) < 1e-9,
        "cube_mass": abs(float(example.model.body_mass.numpy()[example.cube_index]) - EXPECTED["cube_mass_kg"]) < 1e-6,
        "finger_gains": bool(np.allclose(example.model.joint_target_ke.numpy()[7:9], EXPECTED["finger_ke"])),
        "finger_effort": bool(np.allclose(example.model.joint_effort_limit.numpy()[7:9], EXPECTED["finger_effort"])),
        "phase_durations": bool(np.allclose(example.phase_durations.numpy(), EXPECTED["phase_durations"])),
        "gravity": bool(np.allclose(example.model.gravity.numpy().reshape(-1)[:3], [0.0, 0.0, -9.81], atol=1e-4)),
        "body_count": example.model.body_count == 15,
        "joint_count": example.model.joint_count == 15,
    }
    hand, cube = example.ee_index, example.cube_index
    frames, started = 0, time.perf_counter()
    while int(example.phase_index.numpy()[0]) < 4 and frames < 60 * 20:
        example.step()
        frames += 1
    if int(example.phase_index.numpy()[0]) < 4:
        return {"reached_shake": False, "checks": checks}
    reference = _relative(example.state_0.body_q.numpy(), hand, cube)
    start_hand = example.state_0.body_q.numpy()[hand, :3].copy()
    drift, excursion, grip = 0.0, 0.0, []
    for _ in range(round(seconds / example.frame_dt)):
        example.step()
        body_q = example.state_0.body_q.numpy()
        drift = max(drift, float(np.linalg.norm(_relative(body_q, hand, cube) - reference)))
        excursion = max(excursion, float(abs(body_q[hand, 0] - start_hand[0])))
        grip.append(example.control.joint_target_q.numpy()[7:9].copy())
    body_q = example.state_0.body_q.numpy()
    held = float(np.linalg.norm(body_q[cube, :3] - body_q[hand, :3])) < 0.08 and body_q[cube, 2] > 0.04
    checks["gripper_command"] = bool(np.allclose(np.asarray(grip), EXPECTED["gripper_closed"], atol=1e-6))
    # The hand's x excursion must follow the commanded shake (IK tracking error allowed).
    checks["shake_motion"] = abs(excursion - amplitude) < 0.25 * amplitude + 0.004
    return {
        "reached_shake": True,
        "held": bool(held and np.isfinite(body_q).all()),
        "drift_mm": drift * 1000.0 if np.isfinite(drift) else float("inf"),
        "hand_excursion_m": excursion,
        "seconds": time.perf_counter() - started,
        "checks": checks,
    }


def verify(path: Path) -> dict:
    module = load(path)
    results = {name: run_shake(module, *spec) for name, spec in SHAKES.items()}
    checks = {f"{name}.{key}": value for name, r in results.items() for key, value in r["checks"].items()}
    drift_3 = results["3cm"].get("drift_mm", float("inf")) if results["3cm"].get("held") else float("inf")
    drift_10 = results["10cm"].get("drift_mm", float("inf")) if results["10cm"].get("held") else float("inf")
    metrics = {"drift_3cm_mm": drift_3, "drift_10cm_mm": drift_10}
    integrity = all(checks.values())
    success = integrity and all(metrics[k] <= v for k, v in THRESHOLDS.items())
    return {
        "task": "grasp_drift",
        "success": bool(success),
        "integrity": integrity,
        "failed_checks": [k for k, v in checks.items() if not v],
        "metrics": metrics,
        "normalized_worst": {k: metrics[k] / v for k, v in THRESHOLDS.items()},
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.script.resolve())
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {"success": result["success"], "metrics": result["metrics"], "failed_checks": result["failed_checks"]}
        )
    )


if __name__ == "__main__":
    main()
