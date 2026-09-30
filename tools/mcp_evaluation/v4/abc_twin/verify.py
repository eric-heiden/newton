# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify an ABC station twin against held-out recorded top-camera frames.

Imports the submitted script's ``build_model``, ``CAMERA_POSITION``,
``CAMERA_ROTATION``, and ``LOOK`` in a copy of the workspace without the
recorded frames, poses the robot from the joint log at held-out frames,
renders them with this verifier's own copy of ``twin_render.TopCamera``
(textures off, shape colors only), and scores the renders against the
recordings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import newton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twin_render  # noqa: E402

PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "abc_twin"
# Held-out limits on the mean over frames.
THRESHOLDS = {"edge_ncc": 0.67, "ssim": 0.64, "color_psnr_db": 17.0}
ARM_LINKS = [f"{side}_link_{i}" for side in ("left", "right") for i in range(1, 7)]
MAX_ADDED_SHAPES = 60


def load(path: Path):
    # The submitted module imports twin_render; give it the verifier's copy, not the workspace's.
    sys.modules["twin_render"] = twin_render
    spec = importlib.util.spec_from_file_location("submitted_station_twin", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _link_poses(model: newton.Model, state: newton.State) -> dict[str, np.ndarray]:
    body_q = state.body_q.numpy()
    index = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(model.body_label)}
    return {name: body_q[index[name]] for name in ARM_LINKS if name in index}


def _relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose of b in the frame of a (7-vectors p, q xyzw) as a 4x4 matrix."""

    def matrix(t):
        x, y, z, w = t[3:7]
        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = t[:3]
        return m

    return np.linalg.inv(matrix(a)) @ matrix(b)


def kinematics_check(model: newton.Model, reference: newton.Model, q: np.ndarray) -> bool:
    """The arm chains (link 1 to link 6, per arm) must match the station MJCF within 1 mm and 0.01 rad."""
    poses = []
    for m in (model, reference):
        state = m.state()
        twin_render.pose_robot(m, state, q)
        poses.append(_link_poses(m, state))
    if len(poses[0]) != len(ARM_LINKS):
        return False
    for side in ("left", "right"):
        for link in range(2, 7):
            a = _relative(poses[0][f"{side}_link_1"], poses[0][f"{side}_link_{link}"])
            b = _relative(poses[1][f"{side}_link_1"], poses[1][f"{side}_link_{link}"])
            if np.abs(a[:3, 3] - b[:3, 3]).max() > 1e-3 or np.abs(a[:3, :3] - b[:3, :3]).max() > 1e-2:
                return False
    return True


def verify(path: Path) -> dict:
    started = time.perf_counter()
    # Run from a clean copy of the workspace without the recorded frames.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for item in path.parent.iterdir():
            if item.name in ("frames", "__pycache__", "twin_render.py") or item.suffix in (
                ".png",
                ".jpg",
                ".npy",
                ".npz",
            ):
                continue
            if item.is_dir():
                shutil.copytree(item, work / item.name)
            else:
                shutil.copy2(item, work / item.name)
        shutil.copy2(HERE / "twin_render.py", work / "twin_render.py")
        shutil.copy2(PRIVATE / "camera.json", work / "camera.json")
        shutil.copy2(PRIVATE / "joint_log.npz", work / "joint_log.npz")
        module = load(work / path.name)
        model = module.build_model()
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(PRIVATE / "station" / "yam_bimanual_empty.xml"))
        reference = builder.finalize()
        heldout = np.load(PRIVATE / "heldout.npz")
        checks = {
            "single_world": model.world_count == 1,
            "arm_kinematics": kinematics_check(model, reference, heldout["q"][0]),
            "shape_budget": model.shape_count <= reference.shape_count + MAX_ADDED_SHAPES,
        }
        camera = twin_render.TopCamera(
            model,
            module.CAMERA_POSITION,
            module.CAMERA_ROTATION,
            getattr(module, "LOOK", None),
            camera_file=work / "camera.json",
        )
        scores = []
        for q, recorded in zip(heldout["q"], heldout["image"], strict=True):
            state = model.state()
            twin_render.pose_robot(model, state, q)
            scores.append(twin_render.score(camera.render(state), recorded))
    metrics = {key: float(np.mean([s[key] for s in scores])) for key in THRESHOLDS}
    failed = [key for key, limit in THRESHOLDS.items() if not metrics[key] >= limit]
    failed += [name for name, ok in checks.items() if not ok]
    worst = {key: (limit / metrics[key] if metrics[key] > 0 else float("inf")) for key, limit in THRESHOLDS.items()}
    return {
        "success": not failed,
        "integrity": checks,
        "failed_checks": failed,
        "metrics": metrics,
        "per_frame": [{k: round(v, 4) for k, v in s.items()} for s in scores],
        "thresholds": THRESHOLDS,
        "normalized_worst": worst,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.script.resolve())
    text = json.dumps(result, indent=2, default=float)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
