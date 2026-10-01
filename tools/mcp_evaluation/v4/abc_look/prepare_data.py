# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the ABC look-matching task files from a station-twin solution.

Serializes the twin's geometry (camera pose, edited station shapes, added
objects) into ``scene.json`` with neutral colors on the added objects, and
copies the station-twin data files (frames, joint log, camera, station MJCF).

Usage: ``python prepare_data.py TWIN_WORKSPACE TASK_DIR``
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import newton

NEUTRAL = (0.6, 0.6, 0.6)


def main() -> None:
    twin, task = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    sys.path.insert(0, str(twin))
    spec = importlib.util.spec_from_file_location("twin", twin / "station_twin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    station = newton.ModelBuilder()
    station.add_mjcf(str(module.STATION))
    edited = newton.ModelBuilder()
    edited.add_mjcf(str(module.STATION))
    module.build_scene(edited)
    shapes = []
    for i in range(len(station.shape_label)):
        changed = {}
        if not np.allclose(np.asarray(edited.shape_transform[i]), np.asarray(station.shape_transform[i])):
            changed["xform"] = [float(v) for v in edited.shape_transform[i]]
        if not np.allclose(np.asarray(edited.shape_scale[i]), np.asarray(station.shape_scale[i])):
            changed["scale"] = [float(v) for v in edited.shape_scale[i]]
        if edited.shape_flags[i] != station.shape_flags[i]:
            changed["visible"] = bool(edited.shape_flags[i] & int(newton.ShapeFlags.VISIBLE))
        if changed:
            shapes.append({"edit": station.shape_label[i].rsplit("/", 1)[-1], **changed})
    for i in range(len(station.shape_label), len(edited.shape_label)):
        shapes.append(
            {
                "add": int(edited.shape_type[i]),
                "xform": [float(v) for v in edited.shape_transform[i]],
                "scale": [float(v) for v in edited.shape_scale[i]],
                "label": edited.shape_label[i] or f"object_{i}",
            }
        )
    scene = {
        "camera": {"position": list(module.CAMERA_POSITION), "rotation": list(module.CAMERA_ROTATION)},
        "shapes": shapes,
    }
    task.mkdir(parents=True, exist_ok=True)
    (task / "scene.json").write_text(json.dumps(scene, indent=1) + "\n")
    for name in ("camera.json", "joint_log.npz"):
        shutil.copyfile(twin / name, task / name)
    for name in ("frames", "station"):
        shutil.copytree(twin / name, task / name, dirs_exist_ok=True)
    print(f"{sum('add' in s for s in shapes)} added objects, {sum('edit' in s for s in shapes)} edited station shapes")


if __name__ == "__main__":
    main()
