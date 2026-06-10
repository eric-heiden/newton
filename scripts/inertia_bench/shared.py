# SPDX-FileCopyrightText: Copyright (c) 2026 Eric Heiden
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the inertia mesh bench harness.

This module is loaded by every variant runner. It owns:
  - The asset registry (G1, H1, plus a curated menagerie sample).
  - Cache I/O for the pre-extracted per-shape meshes (meshes.npz).
  - Deterministic hashing of the per-call outputs.
  - A high-resolution timer that synchronizes Warp before measuring.
  - A nvidia-smi based peak-memory probe.

The intent is: every variant computes from the same input bytes, and the
captured per-call output (volume, first moment, second moment, mass, com,
inertia, vol) is hashed and timed identically across variants.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Asset registry — keep in lock-step with newton.tests.test_menagerie_usd_mujoco.
# We do not import that module so we can run before newton is even installed.
USD_ASSETS: dict[str, dict[str, str]] = {
    "h1": {"asset_folder": "unitree_h1", "scene_file": "usd_structured/h1.usda"},
    "g1_with_hands": {"asset_folder": "unitree_g1", "scene_file": "usd_structured/g1_29dof_with_hand_rev_1_0.usda"},
    "shadow_hand": {"asset_folder": "shadow_hand", "scene_file": "usd_structured/left_shadow_hand.usda"},
    "apptronik_apollo": {"asset_folder": "apptronik_apollo", "scene_file": "usd_structured/apptronik_apollo.usda"},
    "booster_t1": {"asset_folder": "booster_t1", "scene_file": "usd_structured/T1.usda"},
    "wonik_allegro": {"asset_folder": "wonik_allegro", "scene_file": "usd_structured/allegro_left.usda"},
    "ur5e": {"asset_folder": "universal_robots_ur5e", "scene_file": "usd_structured/ur5e.usda"},
}

# The subset used for the report. We picked 7 to balance G1/H1 + size variety.
DEFAULT_ASSETS: list[str] = list(USD_ASSETS.keys())

BENCH_ROOT = Path(__file__).resolve().parent
MESH_CACHE_PATH = BENCH_ROOT / "meshes.npz"
RESULTS_DIR = BENCH_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class MeshRecord:
    """One mesh shape extracted from a robot's USD."""

    robot: str
    body_name: str
    shape_index: int
    vertices: np.ndarray  # (V, 3) float32
    indices: np.ndarray  # (T*3,) int32
    is_solid: bool
    density: float
    thickness: float
    num_tris: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_tris = self.indices.size // 3


def hash_bytes(obj: Any) -> str:
    """Hash f64-promoted bytes of a number / np array / nested list."""
    h = hashlib.sha256()
    if isinstance(obj, (int, float)):
        h.update(np.asarray([obj], dtype=np.float64).tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(np.ascontiguousarray(obj.astype(np.float64)).tobytes())
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            h.update(hash_bytes(x).encode())
    else:
        h.update(repr(obj).encode())
    return h.hexdigest()[:16]


def gpu_mem_used_mib() -> float:
    """Return current GPU0 used memory in MiB via nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
    ).strip()
    return float(out.splitlines()[0])


def save_meshes(meshes: list[MeshRecord], path: Path = MESH_CACHE_PATH) -> None:
    """Persist a list of MeshRecord to a single .npz."""
    payload: dict[str, np.ndarray] = {}
    payload["__n__"] = np.array([len(meshes)], dtype=np.int64)
    for i, m in enumerate(meshes):
        prefix = f"m{i:04d}"
        payload[f"{prefix}_robot"] = np.array([m.robot], dtype=object)
        payload[f"{prefix}_body"] = np.array([m.body_name], dtype=object)
        payload[f"{prefix}_shape_index"] = np.array([m.shape_index], dtype=np.int64)
        payload[f"{prefix}_vertices"] = m.vertices.astype(np.float32, copy=False)
        payload[f"{prefix}_indices"] = m.indices.astype(np.int32, copy=False)
        payload[f"{prefix}_is_solid"] = np.array([m.is_solid], dtype=bool)
        payload[f"{prefix}_density"] = np.array([m.density], dtype=np.float64)
        payload[f"{prefix}_thickness"] = np.array([m.thickness], dtype=np.float64)
    np.savez_compressed(path, **payload, allow_pickle=True)


def load_meshes(path: Path = MESH_CACHE_PATH) -> list[MeshRecord]:
    """Load MeshRecord list from .npz."""
    data = np.load(path, allow_pickle=True)
    n = int(data["__n__"][0])
    out: list[MeshRecord] = []
    for i in range(n):
        prefix = f"m{i:04d}"
        out.append(
            MeshRecord(
                robot=str(data[f"{prefix}_robot"][0]),
                body_name=str(data[f"{prefix}_body"][0]),
                shape_index=int(data[f"{prefix}_shape_index"][0]),
                vertices=data[f"{prefix}_vertices"].astype(np.float32, copy=True),
                indices=data[f"{prefix}_indices"].astype(np.int32, copy=True).flatten(),
                is_solid=bool(data[f"{prefix}_is_solid"][0]),
                density=float(data[f"{prefix}_density"][0]),
                thickness=float(data[f"{prefix}_thickness"][0]),
            )
        )
    return out


def write_results(variant: str, payload: dict[str, Any]) -> Path:
    """Write per-variant results json to results/."""
    path = RESULTS_DIR / f"results_{variant}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def read_results(variant: str) -> dict[str, Any] | None:
    """Load per-variant results json or None if missing."""
    path = RESULTS_DIR / f"results_{variant}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def env_snapshot() -> dict[str, str]:
    """Capture a small env snapshot for reproducibility."""
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader", "--id=0"],
            text=True,
        ).strip()
    except Exception as e:  # pragma: no cover
        gpu = f"unknown ({e})"
    return {
        "gpu": gpu,
        "cwd": str(Path.cwd()),
        "warp_deterministic_env": os.environ.get("WARP_DETERMINISTIC", ""),
    }


def sync_warp() -> None:
    """Synchronize the default Warp device (lazy import so V2's Warp loads cleanly)."""
    import warp as wp

    wp.synchronize_device()


def now_ms() -> float:
    return time.perf_counter() * 1000.0
