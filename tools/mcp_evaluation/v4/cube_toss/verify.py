# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a calibrated cube-toss contact model against held-out real tosses.

Builds the submitted model through ``build_model``/``make_solver`` (and
``make_pipeline`` if defined) and runs this verifier's own open-loop rollout of
every held-out toss from its first measured frame, at the script's substep count.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "cube_toss"
FRAME_DT, HALF_WIDTH, MASS, INERTIA = 1.0 / 148.0, 0.0524, 0.37, 0.00081
# Held-out limits: mean over tosses of the time-averaged error.
THRESHOLDS = {"position_m": 0.020, "rotation_rad": 0.45}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_cube_toss", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def load_tosses(path: Path) -> list[dict]:
    data = np.load(path)
    offsets = data["offsets"]
    return [
        {key: data[key][start:end] for key in ("pos", "quat", "vel", "ang_vel")}
        for start, end in itertools.pairwise(offsets)
    ]


def evaluate(module, tosses: list[dict]) -> tuple[dict, dict]:
    worlds = len(tosses)
    model = module.build_model(worlds)
    solver = module.make_solver(model)
    pipeline = module.make_pipeline(model) if hasattr(module, "make_pipeline") else newton.CollisionPipeline(model)
    substeps = int(module.SUBSTEPS)
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()
    shape_scale = model.shape_scale.numpy()[model.shape_body.numpy() >= 0]
    checks = {
        "one_cube_per_world": model.body_count == worlds and model.joint_dof_count == 6 * worlds,
        "mass": bool(np.allclose(mass, MASS, rtol=1e-4)),
        "inertia": bool(np.allclose(inertia, np.eye(3) * INERTIA, atol=1e-7)),
        "cube_size": bool(np.allclose(shape_scale, HALF_WIDTH, atol=1e-5)),
        "gravity": bool(np.allclose(model.gravity.numpy().reshape(-1)[:3], [0.0, 0.0, -9.81], atol=1e-4)),
        "substeps": 1 <= substeps <= 100,
    }
    state_0, state_1, control = model.state(), model.state(), model.control()
    state_0.joint_q.assign(
        np.concatenate([np.concatenate([t["pos"][0], t["quat"][0]]) for t in tosses]).astype(np.float32)
    )
    state_0.joint_qd.assign(
        np.concatenate([np.concatenate([t["vel"][0], t["ang_vel"][0]]) for t in tosses]).astype(np.float32)
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    contacts = pipeline.contacts()
    dt = FRAME_DT / substeps
    frames = max(len(t["pos"]) for t in tosses)
    first = state_0

    def frame():
        current, scratch = first, state_1
        for _ in range(substeps):
            current.clear_forces()
            pipeline.collide(current, contacts)
            solver.step(current, scratch, control, contacts, dt)
            current, scratch = scratch, current
        if current is not first:
            first.assign(current)

    graph = None
    if model.device.is_cuda:
        with wp.ScopedCapture() as capture:
            frame()
        graph = capture.graph
    poses = [first.body_q.numpy()]
    for _ in range(frames - 1):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            frame()
        poses.append(first.body_q.numpy())
    poses = np.asarray(poses)
    position, rotation = [], []
    for world, toss in enumerate(tosses):
        n = len(toss["pos"])
        predicted = poses[:n, world]
        position.append(np.mean(np.linalg.norm(predicted[:, :3] - toss["pos"], axis=1)))
        dot = np.abs(np.sum(predicted[:, 3:7] * toss["quat"], axis=1))
        rotation.append(np.mean(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))
    finite = bool(np.isfinite(poses).all())
    metrics = {
        "position_m": float(np.mean(position)) if finite else float("inf"),
        "rotation_rad": float(np.mean(rotation)) if finite else float("inf"),
    }
    return metrics, checks


def verify(path: Path) -> dict:
    module = load(path)
    started = time.perf_counter()
    metrics, checks = evaluate(module, load_tosses(PRIVATE / "heldout.npz"))
    train = path.parent / "tosses.npz"
    train_metrics = evaluate(module, load_tosses(train))[0] if train.exists() else {}
    integrity = all(checks.values())
    success = integrity and all(metrics[k] <= v for k, v in THRESHOLDS.items())
    return {
        "task": "cube_toss",
        "success": bool(success),
        "integrity": integrity,
        "failed_checks": [k for k, v in checks.items() if not v],
        "metrics": {**metrics, **{f"train_{k}": v for k, v in train_metrics.items()}},
        "normalized_worst": {k: metrics[k] / v for k, v in THRESHOLDS.items()},
        "details": {"seconds": time.perf_counter() - started},
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
