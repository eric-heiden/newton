# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a submitted G1 reference-following controller in a fresh process.

Runs the submitted ``g1_track.py`` on the training clip, on an unseen
time-scaled variant, and on a harder stretch clip, and measures root and joint
tracking. The robot model, timestep, and motion input must be unchanged, and
the root must not be actuated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import newton
import newton.viewer

HERE = Path(__file__).resolve().parent
THRESHOLDS = {"root_rmse_m": 0.04, "joint_rmse_rad": 0.05}
CASES = {
    # name: (clip, playback speed, gates success)
    "wave": ("wave.csv", 1.0, True),
    "wave_fast": ("wave.csv", 1.4, True),
    "high5": ("high5.csv", 1.0, False),
}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def run_case(module, clip: Path, speed: float) -> dict:
    parser = module.Example.create_parser()
    args, _ = parser.parse_known_args(["--motion", str(clip)])
    args.viewer = "null"
    example = module.Example(newton.viewer.ViewerNull(num_frames=1 << 30), args)
    reference_motion = np.loadtxt(clip, delimiter=",")
    # Playback speed is applied to the clip itself, so the controller sees an ordinary, faster motion.
    example.motion.fps = 30.0 * speed
    example.motion.duration = (len(example.motion.qpos) - 1) / example.motion.fps
    armature = example.model.joint_armature.numpy()[6:]
    checks = {
        "motion_input": bool(np.allclose(example.motion.qpos, reference_motion)),
        "sim_dt": abs(example.sim_dt - 0.002) < 1e-9 and abs(example.frame_dt - 0.02) < 1e-9,
        "armature": bool(np.allclose(armature, module.ARMATURE if hasattr(module, "ARMATURE") else armature)),
        "body_count": example.model.body_count == 30,
        "mass": abs(float(example.model.body_mass.numpy().sum()) - 33.34) < 0.5,
        "gravity": bool(np.allclose(example.model.gravity.numpy().reshape(-1)[:3], [0.0, 0.0, -9.81], atol=1e-4)),
        "root_unactuated": True,
    }
    effort = example.model.joint_effort_limit.numpy()[6:]
    root_err, joint_err, min_height = [], [], np.inf
    started = time.perf_counter()
    for _ in range(round(example.motion.duration / example.frame_dt)):
        example.step()
        joint_f = example.control.joint_f.numpy() if example.control.joint_f is not None else np.zeros(6)
        if np.any(np.abs(joint_f[:6]) > 1e-9):
            checks["root_unactuated"] = False
        q = example.state_0.joint_q.numpy()
        r = example._shifted(example.motion.sample(example.sim_time))
        root_err.append(np.linalg.norm(q[:3] - r[:3]))
        joint_err.append(np.sqrt(np.mean((q[7:] - r[7:]) ** 2)))
        min_height = min(min_height, float(q[2]))
    finite = bool(np.isfinite(root_err).all())
    return {
        "root_rmse_m": float(np.sqrt(np.mean(np.square(root_err)))) if finite else float("inf"),
        "joint_rmse_rad": float(np.sqrt(np.mean(np.square(joint_err)))) if finite else float("inf"),
        "min_root_height_m": min_height,
        "upright": bool(finite and min_height > 0.6),
        "effort_limits": effort.tolist()[:3],
        "seconds": time.perf_counter() - started,
        "checks": checks,
    }


def verify(path: Path, clips: Path = HERE) -> dict:
    module = load(path)
    results = {name: run_case(module, clips / clip, speed) for name, (clip, speed, _) in CASES.items()}
    checks = {f"{name}.{k}": v for name, r in results.items() for k, v in r["checks"].items()}
    integrity = all(checks.values())
    gated = [name for name, (_, _, gate) in CASES.items() if gate]
    worst = {
        key: max(results[name][key] if results[name]["upright"] else float("inf") for name in gated)
        for key in THRESHOLDS
    }
    success = integrity and all(worst[k] <= v for k, v in THRESHOLDS.items())
    return {
        "task": "g1_track",
        "success": bool(success),
        "integrity": integrity,
        "failed_checks": [k for k, v in checks.items() if not v],
        "metrics": {
            **worst,
            "stretch_high5_root_rmse_m": results["high5"]["root_rmse_m"],
            "stretch_high5_upright": results["high5"]["upright"],
        },
        "normalized_worst": {k: worst[k] / v for k, v in THRESHOLDS.items()},
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
