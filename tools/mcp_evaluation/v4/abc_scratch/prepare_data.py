# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the ``abc_scratch`` task files from the ``abc_replay`` fruit-bowl data.

The episode is ABC-130k ``7067ee1f`` ("place and organize the fake fruits in the fruit bowl"), already
decoded for ``abc_replay`` (``abc_replay/prepare_data.py``). This script only selects and repackages:

- the agent dataset (``--task``): ``photos/`` (top-camera frames at the start, before each grasp, after each
  release, and at the end, and the wrist-camera frame of each grasp, with ``photos/index.json``: camera, frame
  time, the state row the photo shows and its time),
  ``episode.npz`` (measured joints, velocities, torques including the gripper effort, gripper openings, and
  logged joint and gripper commands; no camera clocks or logged end-effector poses), ``camera.json``, and
  ``station/`` (the ABC simulator's station MJCF and meshes, no objects). ``arm_logs/`` (the ``abc_arm``
  dataset's logs) and ``FORMAT.md`` (next to this script) come through the task registry (``run_v4.py``);
- the hidden verification data (``--private``, ``~/.newton-visual-private/abc_scratch``): ``truth.json`` (per
  object the grasping arm, video-tracker start, size range, centre height, event frames, final position, and
  the FK grasp start with the MJCF arm bases; the tray at the start and the end), ``gt.npz`` (the video
  tracker's per-frame tracks), and the verifier's copies of ``episode.npz`` and ``station/``.

Usage::

    python -m tools.mcp_evaluation.v4.abc_scratch.prepare_data [--source ~/.newton-visual-private/abc_replay] \\
        [--frames DATASETS/abc_replay_task/frames/main] [--task DATASETS/abc_scratch_task] \\
        [--private ~/.newton-visual-private/abc_scratch]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = Path("/home/horde/artifacts/newton-live-mcp-v4/datasets")
PRIVATE_ROOT = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private"))
OBJECTS = ("pear", "orange", "dark_fruit")
SIDES = ("left", "right")
# Episode keys handed to the agent (``<side>_`` prefixed); the camera clocks and logged poses stay out.
EPISODE_KEYS = ("t", "q", "qd", "tau", "cmd_t", "cmd", "grip_t", "grip", "grip_cmd_t", "grip_cmd")
# (camera, frame, what it shows); frames of the main episode's 30 fps streams (event frames in its scene).
PHOTOS = (
    ("top", 0, "start"),
    ("top", 40, "left gripper open around the first object, before it closes"),
    ("left_wrist", 47, "left gripper closing on the first object"),
    ("top", 107, "after the first release"),
    ("top", 115, "left gripper open around the second object, before it closes"),
    ("left_wrist", 122, "left gripper closing on the second object"),
    ("top", 181, "right gripper open around the third object, before it closes"),
    ("right_wrist", 189, "right gripper closing on the third object"),
    ("top", 199, "after the second release"),
    ("top", 259, "after the third release"),
    ("top", 280, "end"),
)
CAMERA_OF = {"top": "top", "left_wrist": "left", "right_wrist": "right"}


def _replay_common(source: Path):
    """``abc_replay``'s replay_common (StationFK) for the FK grasp starts."""
    spec = importlib.util.spec_from_file_location("abc_replay_common", HERE.parent / "abc_replay" / "replay_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sorted_ranges(fruit: dict) -> tuple[list[list[float]], list[float]]:
    """Size ranges and nominal sizes of the principal extents, longest first [m]."""
    ranges, size = fruit["size_range_m"], fruit["size_m"]
    if "diameter" in ranges:
        return [list(ranges["diameter"])] * 3, [size["diameter"]] * 3
    return [list(ranges["length"]), list(ranges["width"]), list(ranges["width"])], [
        size["length"],
        size["width"],
        size["width"],
    ]


def rest_positions(gt: dict, name: str) -> list[list[float]]:
    """Distinct xy centres [m] of the frames in which the tracker saw an object resting after its release."""
    frames = np.flatnonzero(gt[f"{name}_phase"] == 4)
    points = gt[f"{name}_pos"][frames, :2]
    points = points[np.all(np.isfinite(points), axis=1)]
    return np.unique(np.round(points, 4), axis=0).tolist()


def mjcf_bases(fk) -> dict[str, list[float]]:
    """Arm base positions of the station MJCF [m] (bodies ``<side>_arm`` welded to the world)."""
    model = fk.model
    leaves = [label.rsplit("/", 1)[-1] for label in model.body_label]
    child, x_p = model.joint_child.numpy().tolist(), model.joint_X_p.numpy()
    return {side: [round(float(v), 6) for v in x_p[child.index(leaves.index(f"{side}_arm"))][:3]] for side in SIDES}


def build(args: argparse.Namespace) -> None:
    source, task, private = args.source, args.task, args.private
    episode = dict(np.load(source / "episodes" / "main.npz"))
    scene = json.loads((source / "scenes" / "main.json").read_text())
    gt = dict(np.load(source / "gt" / "main.npz"))
    tracker = json.loads((source / "fruits" / "ground_truth.json").read_text())

    # ---- agent dataset
    task.mkdir(parents=True, exist_ok=True)
    photos = task / "photos"
    if photos.exists():
        shutil.rmtree(photos)
    photos.mkdir()
    index = []
    rows = len(episode["left_t"])
    for number, (folder, frame, what) in enumerate(PHOTOS, start=1):
        camera = CAMERA_OF[folder]
        name = f"{number:02d}_{folder}.jpg"
        shutil.copy2(args.frames / folder / f"{frame:03d}.jpg", photos / name)
        row = int(np.clip(frame - 1, 0, rows - 1))
        index.append(
            {
                "file": name,
                "camera": camera,
                "time_s": round(float(episode[f"t_{camera}"][frame]), 4),
                "state_row": row,
                "state_time_s": round(float(episode["left_t"][row]), 4),
                "shows": what,
            }
        )
    (photos / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    agent_episode = {f"{side}_{key}": episode[f"{side}_{key}"] for side in SIDES for key in EPISODE_KEYS}
    np.savez(task / "episode.npz", **agent_episode)
    camera = json.loads((source / "camera.json").read_text())
    camera.pop("top_fit", None)
    camera["time_base"] = "camera frame i shows arm-state row max(i - 1, 0); photos/index.json gives each photo's row"
    (task / "camera.json").write_text(json.dumps(camera, indent=1) + "\n")
    if (task / "station").exists():
        shutil.rmtree(task / "station")
    shutil.copytree(source / "station", task / "station")

    # ---- hidden verification data
    rc = _replay_common(source)
    fk = rc.StationFK(source / "station" / "yam_bimanual_empty.xml", None)  # the MJCF's arm bases
    fk_starts = fk.fruit_starts(episode, scene)
    objects = {}
    for name in OBJECTS:
        fruit, record = scene["fruits"][name], tracker["objects"][name]
        ranges, nominal = _sorted_ranges(fruit)
        objects[name] = {
            "arm": fruit["arm"],
            "shape": fruit["shape"],
            "start_image_xyz": [float(v) for v in gt[f"{name}_rest_xyz"]],
            "final_xyz": [float(v) for v in gt[f"{name}_final_xyz"]],
            # Centres where the tracker saw the object resting after its release (phase 4), before and after
            # other objects pushed it.
            "rest_xy": rest_positions(gt, name),
            "extent_ranges_m": ranges,
            "nominal_extents_m": nominal,
            "centre_height_m": fruit["centre_height_m"],
            "grip_gap_m": fruit["grip_gap_m"],
            "events": fruit["events"],
            "start_fk_mjcf_bases": fk_starts[name],
            "start_fk_scene_bases": fruit["start"],
            "tracker_appearance": record.get("appearance"),
        }
    tray = scene["tray"]
    truth = {
        "_note": "abc_scratch hidden truth (main episode 7067ee1f), built by tools/mcp_evaluation/v4/abc_scratch/"
        "prepare_data.py from ~/.newton-visual-private/abc_replay (video tracker and scene). Event values are top "
        "frames; top frame i shows arm-state row max(i - 1, 0).",
        "episode": scene["uuid"],
        "table_z": scene["table_z"],
        "mjcf_bases": mjcf_bases(fk),
        "video_fit_bases": scene["bases"],
        "objects": objects,
        "tray": {
            "start": {"apex_xy": tray["apex_xy"], "yaw_deg": tray["yaw_deg"]},
            "end": {"apex_xy": tray["end_apex_xy"], "yaw_deg": tray["end_yaw_deg"]},
            "radius_m": tray["radius_m"],
            "half_angle_deg": tray["half_angle_deg"],
            "rim_height_m": tray["rim_height_m"],
            "floor_height_m": tray["floor_height_m"],
            "iou": tray.get("iou"),
        },
    }
    private.mkdir(parents=True, exist_ok=True)
    (private / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    np.savez(private / "gt.npz", **gt)
    np.savez(private / "episode.npz", **agent_episode)
    shutil.copy2(task / "camera.json", private / "camera.json")
    if (private / "station").exists():
        shutil.rmtree(private / "station")
    shutil.copytree(source / "station", private / "station")
    print(f"wrote {task} ({len(index)} photos) and {private}")
    for name, record in objects.items():
        start = np.asarray(record["start_fk_mjcf_bases"]["pos"][:2]) - np.asarray(record["start_image_xyz"][:2])
        print(f"  {name}: FK start (MJCF bases) - video start = {np.round(1000 * start, 1).tolist()} mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=PRIVATE_ROOT / "abc_replay")
    parser.add_argument("--frames", type=Path, default=DATASETS / "abc_replay_task" / "frames" / "main")
    parser.add_argument("--task", type=Path, default=DATASETS / "abc_scratch_task")
    parser.add_argument("--private", type=Path, default=PRIVATE_ROOT / "abc_scratch")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
