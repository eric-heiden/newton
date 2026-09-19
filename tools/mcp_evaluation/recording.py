# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Ingest actual HUG recordings without distributing recordings or MANO assets.

Frame conventions were checked against the local manosim-newton loader and
the source data: hand fits are in the Aria device frame, object placement is
in the Z-up SLAM world, and Newton quaternions use xyzw ordering.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import warp as wp


def digest(path: Path) -> str:
    """Compute a source file's SHA-256 digest."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def rotvec_quat(v: np.ndarray) -> np.ndarray:
    """Convert rotation vectors [rad] to xyzw quaternions."""
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = 0.5 * np.sinc(angle / (2.0 * np.pi))
    return np.concatenate((v * scale, np.cos(angle * 0.5)), axis=-1)


def interpolate_quat(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate normalized quaternions along their shortest arc."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    denominator = np.sinc(angle / np.pi)
    first = (1 - alpha) * np.sinc((1 - alpha) * angle / np.pi) / denominator
    second = alpha * np.sinc(alpha * angle / np.pi) / denominator
    value = first * a + second * b
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read metric OBJ vertices and triangulate polygon faces."""
    vertices, indices = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields and fields[0] == "f":
            face = [int(x.split("/")[0]) for x in fields[1:]]
            face = [x - 1 if x > 0 else len(vertices) + x for x in face]
            for index in range(1, len(face) - 1):
                indices.extend((face[0], face[index], face[index + 1]))
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


class Recording:
    """A contiguous recorded right-hand approach to a scanned softball."""

    def __init__(self, data_root: Path, *, variant: int = 0):
        scene = "medium_2"
        self.source = data_root / "scenes" / scene / "aria_data.pkl"
        self.assets = data_root / "hug_bench" / "test" / scene / "softball" / "sim_assets"
        self.properties = json.loads((self.assets / "properties.json").read_text())
        # Only load the operator-provided local recording; pickle is not a safe interchange format.
        with self.source.open("rb") as stream:
            data = pickle.load(stream)
        self.fps = float(data["fps"])
        indices, positions, wrists, fingers, landmarks = [], [], [], [], []
        for frame in data["frame_data"]:
            hand = (frame.get("mano_hand_tracking") or {}).get("right")
            if hand is None:
                continue
            device = np.asarray(frame["T_world_device"], dtype=float)
            wrist = device @ np.asarray(hand["T_device_wrist"], dtype=float)
            indices.append(int(frame["index"]))
            positions.append(wrist[:3, 3])
            wrists.append(np.asarray(wp.quat_from_matrix(wp.mat33(wrist[:3, :3].flatten()))))
            fingers.append(rotvec_quat(np.asarray(hand["pose"])[0]))
            points = np.asarray(hand["landmarks"])
            landmarks.append(points @ device[:3, :3].T + device[:3, 3])
        indices = np.asarray(indices)
        positions = np.asarray(positions)
        landmarks = np.asarray(landmarks)
        center = np.asarray(self.properties["world_position"])
        distance = np.linalg.norm(landmarks - center[None, None, :], axis=-1).min(axis=1)
        # Select by recorded proximity alone, before any simulation or tuning result exists.
        candidates = np.argsort(distance)
        chosen = None
        for candidate in candidates:
            lo, hi = candidate - 10 - variant, candidate + 20 - variant
            if lo >= 0 and hi < len(indices) and np.all(np.diff(indices[lo : hi + 1]) == 1):
                chosen = (lo, hi)
                break
        if chosen is None:
            raise ValueError("No contiguous three-second tracked approach in the recording")
        lo, hi = chosen
        self.indices = indices[lo : hi + 1]
        self.time = (self.indices - self.indices[0]) / self.fps
        vertices, _ = load_obj(self.assets / "visual.obj")
        table_z = center[2] + vertices[:, 2].min()
        self.offset = np.array([-center[0], -center[1], -table_z])
        self.positions = positions[lo : hi + 1] + self.offset
        self.wrists = np.asarray(wrists)[lo : hi + 1]
        self.fingers = np.asarray(fingers)[lo : hi + 1]
        self.landmarks = landmarks[lo : hi + 1] + self.offset
        self.object_position = center + self.offset
        self.provenance = {
            "dataset": "Human Universal Grasping / HUG-Bench",
            "primary_sources": ["https://grasping.io/", "https://github.com/KevinyWu/aria2mesh"],
            "scene": scene,
            "object": "softball",
            "recording_path": str(self.source),
            "recording_sha256": digest(self.source),
            "frame_indices": self.indices.tolist(),
            "sample_rate_hz": self.fps,
            "window_selection": "minimum landmark/object distance with contiguous 3 s tracking",
            "world_transform": "T_world_wrist = T_world_device @ T_device_wrist; Z-up; meters; xyzw",
            "recenter_translation_m": self.offset.tolist(),
            "assets": {
                str(p.relative_to(self.assets)): digest(p) for p in sorted(self.assets.rglob("*")) if p.is_file()
            },
            "limitations": "Recorded MANO fits and authored object initial pose; no tracked object trajectory or success labels. Fixed supplied hand morphology, not subject-specific MANO shape fitting.",
        }

    def sample(self, time: float) -> np.ndarray:
        """Sample the recorded wrist and finger target in Newton coordinates."""
        high = int(np.clip(np.searchsorted(self.time, time, side="right"), 1, len(self.time) - 1))
        low = high - 1
        alpha = float(np.clip((time - self.time[low]) / (self.time[high] - self.time[low]), 0, 1))
        position = (1 - alpha) * self.positions[low] + alpha * self.positions[high]
        wrist = interpolate_quat(self.wrists[low], self.wrists[high], alpha)
        fingers = interpolate_quat(self.fingers[low], self.fingers[high], alpha)
        return np.concatenate((position, wrist, fingers.flatten()))
