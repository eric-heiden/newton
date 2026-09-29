# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Convert ContactNets cube tosses into task files.

Source: https://github.com/DAIRLab/contact-nets (BSD-3-Clause), ``data/tosses_processed``,
already converted to SI units in ``cube_tosses_si.npz`` (positions [m], quaternions wxyz,
world-frame velocity [m/s], body-frame angular velocity [rad/s], 148 Hz).
Tosses 0-399 become the agent's ``tosses.npz``; 400-569 are held out.

Usage: ``python prepare_data.py cube_tosses_si.npz TRAIN_DIR HELDOUT_DIR``
"""

import sys
from pathlib import Path

import numpy as np


def rotate(quat_xyzw: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors into the world frame."""
    xyz, w = quat_xyzw[:, :3], quat_xyzw[:, 3:4]
    t = 2.0 * np.cross(xyz, vectors)
    return vectors + w * t + np.cross(xyz, t)


def main() -> None:
    source, train, heldout = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    data = np.load(source)
    quat = data["quat_wxyz"][:, [1, 2, 3, 0]]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    arrays = {"pos": data["pos"], "quat": quat, "vel": data["vel"], "ang_vel": rotate(quat, data["ang_vel"])}
    offsets = data["offsets"]
    for directory, name, tosses in ((train, "tosses.npz", range(400)), (heldout, "heldout.npz", range(400, 570))):
        directory.mkdir(parents=True, exist_ok=True)
        rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in tosses])
        lengths = [offsets[i + 1] - offsets[i] for i in tosses]
        np.savez_compressed(
            directory / name,
            **{key: value[rows] for key, value in arrays.items()},
            offsets=np.concatenate([[0], np.cumsum(lengths)]),
            toss_id=np.asarray(list(tosses)),
        )


if __name__ == "__main__":
    main()
