# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Task registry, reference generation, and held-out verification.

Hidden generating parameters and held-out truth measurements are read from a
private directory supplied on the command line; they are not part of this
source tree or any agent workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .common import VisualTask, contact_sheet, save_png

# Held-out gates, frozen from private perturbation studies before any agent trial.
# Cloth folding is nondeterministic (a truth rerun differs by ~1 cm mean, ~6 cm p95),
# so its gate checks material identification at roughly factor-of-two precision.
THRESHOLDS = {
    "cloth_drape": {"mean_particle_error_m": 0.04, "p95_particle_error_m": 0.20},
    "push": {"position_error_m": 0.015, "yaw_error_deg": 10.0},
    "arm_offsets": {"link_position_error_m": 0.03, "hand_rotation_error_deg": 4.0},
}

TRUTH_TIMES = {
    "cloth_drape": (0.5, 1.0, 1.5, 2.0),
    "push": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    "arm_offsets": (0.0,),
}


def task_class(name: str) -> type[VisualTask]:
    if name == "cloth_drape":
        from .cloth import ClothDrape  # noqa: PLC0415

        return ClothDrape
    if name == "push":
        from .push import PlanarPush  # noqa: PLC0415

        return PlanarPush
    if name == "arm_offsets":
        from .arm import ArmOffsets  # noqa: PLC0415

        return ArmOffsets
    raise KeyError(f"Unknown task {name!r}")


def measure(task: VisualTask, episodes, times) -> dict[str, np.ndarray]:
    """Record truth-comparison measurements for each episode at each time."""
    result = {}
    for episode in episodes:
        task.set_episode(episode)
        task.reset()
        for t in times:
            task.simulate_to(t)
            for key, value in task.measurements().items():
                result[f"{episode}/{t:g}/{key}"] = value
    return result


def write_reference(name: str, params: dict, output: Path, private: Path, device=None) -> dict:
    """Render public reference photos and record private truth measurements."""
    cls = task_class(name)
    task = cls(params, device=device)
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for episode in cls.TRAIN_EPISODES:
        task.set_episode(episode)
        frames = task.rollout()
        tiles, labels = [], []
        for camera in cls.CAMERAS:
            row, row_labels = [], []
            for t in sorted(frames):
                filename = f"{episode}_{camera.name}_t{t:.2f}.png"
                save_png(output / filename, frames[t][camera.name])
                index.append({"episode": episode, "camera": camera.name, "time_s": t, "file": filename})
                row.append(frames[t][camera.name])
                row_labels.append(f"REFERENCE {episode} {camera.name} t={t:.2f}s")
            tiles.append(row)
            labels.append(row_labels)
        save_png(output / f"sheet_{episode}.png", contact_sheet(tiles, labels))
    spec = cls.public_spec()
    spec["images"] = index
    (output / "reference.json").write_text(json.dumps(spec, indent=2) + "\n")
    truth = measure(task, cls.TRAIN_EPISODES + cls.HELDOUT_EPISODES, TRUTH_TIMES[name])
    private.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private / f"{name}_truth.npz", **truth)
    return spec


def _yaw_deg(q: np.ndarray) -> float:
    x, y, z, w = q[3:7]
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def _rotation_angle_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    dot = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2 * np.arccos(min(1.0, dot))))


def compare(name: str, measured: dict, truth: dict, episodes) -> dict:
    """Continuous error metrics for the selected episodes."""
    rows = {}
    for key, reference in truth.items():
        episode, t, _ = key.split("/")
        if episode not in episodes:
            continue
        value = measured[key]
        if name == "cloth_drape":
            error = np.linalg.norm(value - reference, axis=1)
            finite = bool(np.isfinite(value).all())
            rows[f"{episode}@{t}"] = {
                "mean_particle_error_m": float(error.mean()) if finite else float("inf"),
                "p95_particle_error_m": float(np.percentile(error, 95)) if finite else float("inf"),
            }
        elif name == "push":
            finite = bool(np.isfinite(value).all())
            yaw_error = abs((_yaw_deg(value) - _yaw_deg(reference) + 180.0) % 360.0 - 180.0)
            rows[f"{episode}@{t}"] = {
                "position_error_m": float(np.linalg.norm(value[:3] - reference[:3])) if finite else float("inf"),
                "yaw_error_deg": float(yaw_error) if finite else float("inf"),
            }
        else:
            finite = bool(np.isfinite(value).all())
            positions = np.linalg.norm(value[:, :3] - reference[:, :3], axis=1)
            # The last arm body before the fingers carries the hand frame.
            rows[f"{episode}@{t}"] = {
                "link_position_error_m": float(positions.max()) if finite else float("inf"),
                "hand_rotation_error_deg": _rotation_angle_deg(value[-3, 3:7], reference[-3, 3:7])
                if finite
                else float("inf"),
            }
    keys = next(iter(rows.values())).keys()
    worst = {key: max(row[key] for row in rows.values()) for key in keys}
    return {"worst": worst, "rows": rows}


def verify(name: str, params: dict, private: Path, device=None) -> dict:
    """Fresh-process held-out verification of a submitted parameter set."""
    started = time.perf_counter()
    cls = task_class(name)
    cls._validate(params)
    missing = sorted(set(cls.PARAMS) - set(params))
    if missing:
        raise ValueError(f"Submission is missing parameters {missing}")
    with np.load(private / f"{name}_truth.npz") as data:
        truth = {key: data[key] for key in data.files}
    task = cls(params, device=device)
    measured = measure(task, cls.TRAIN_EPISODES + cls.HELDOUT_EPISODES, TRUTH_TIMES[name])
    heldout = compare(name, measured, truth, cls.HELDOUT_EPISODES)
    training = compare(name, measured, truth, cls.TRAIN_EPISODES)
    thresholds = THRESHOLDS[name]
    success = all(v is not None for v in thresholds.values()) and all(
        heldout["worst"][key] <= limit for key, limit in thresholds.items()
    )
    return {
        "task": name,
        "params": params,
        "thresholds": thresholds,
        "heldout": heldout,
        "training": training,
        "success": bool(success),
        "normalized_worst": {
            key: heldout["worst"][key] / limit for key, limit in thresholds.items() if limit is not None
        },
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--task", required=True)
    generate.add_argument("--private", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--task", required=True)
    check.add_argument("--private", type=Path, required=True)
    check.add_argument("--params", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reference generation and verification never count as agent candidates.
    os.environ.setdefault("NEWTON_VISUAL_LOG", os.devnull)
    if args.command == "generate":
        truth = json.loads((args.private / "truth.json").read_text())[args.task]
        spec = write_reference(args.task, truth, args.output, args.private)
        print(json.dumps({"task": args.task, "images": len(spec["images"])}))
    else:
        params = json.loads(args.params.read_text())
        result = verify(args.task, params, args.private)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"success": result["success"], "worst": result["heldout"]["worst"]}))


if __name__ == "__main__":
    main()
