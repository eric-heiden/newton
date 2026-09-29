# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify SDF material removal in a submitted grinding example.

Runs the submitted ``sdf_grinding.py`` for the full pass in a fresh process,
then checks the workpiece geometry the model collides with: the remaining
volume against a Monte-Carlo estimate of the ellipsoid minus the swept wheel,
probe points inside and below the groove, and the hydroelastic normal load
when the wheel is placed back into the finished groove.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer

EXPECTED = {
    "WORKPIECE_RADII": (0.45, 0.25, 0.12),
    "GRINDER_RADIUS": 0.13,
    "GRINDER_HALF_WIDTH": 0.04,
    "GRIND_DEPTH": 0.035,
    "GRIND_FRAMES": 270,
    "HYDROELASTIC_STIFFNESS": 1.0e8,
}
# Relative volume error vs. the analytic removal, groove probe fractions, and the
# normal load in the finished groove relative to an unground workpiece.
THRESHOLDS = {"volume_error": 0.15, "groove_cleared": 0.9, "base_kept": 0.9, "groove_load_ratio": 0.25}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_grinding", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def reference_pose(frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Grinder center [m] and axis of the expected tool path."""
    a, _, c = EXPECTED["WORKPIECE_RADII"]
    x = -0.36 + min(frame / EXPECTED["GRIND_FRAMES"], 1.0) * 0.72
    surface = c * math.sqrt(1.0 - min((x / a) ** 2, 1.0))
    return np.array([x, 0.0, surface + EXPECTED["GRINDER_RADIUS"] - EXPECTED["GRIND_DEPTH"]]), np.array([0.0, 1.0, 0.0])


def expected_volume(frames: int, samples: int = 4_000_000, seed: int = 0) -> float:
    """Monte-Carlo volume [m^3] of the ellipsoid minus every wheel pose up to ``frames``."""
    radii = np.asarray(EXPECTED["WORKPIECE_RADII"])
    rng = np.random.default_rng(seed)
    points = rng.uniform(-radii, radii, size=(samples, 3))
    inside = np.sum((points / radii) ** 2, axis=1) <= 1.0
    points = points[inside]
    removed = np.zeros(len(points), dtype=bool)
    centers = np.stack([reference_pose(f)[0] for f in range(frames + 1)])
    near = np.abs(points[:, 1]) <= EXPECTED["GRINDER_HALF_WIDTH"]
    for center in centers:
        dx, dz = points[:, 0] - center[0], points[:, 2] - center[2]
        removed |= near & (dx * dx + dz * dz <= EXPECTED["GRINDER_RADIUS"] ** 2)
    box = float(np.prod(2.0 * radii))
    return box * float(np.count_nonzero(inside)) / samples * float(np.count_nonzero(~removed)) / len(points)


def mesh_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i", vertices[triangles[:, 0]], np.cross(vertices[triangles[:, 1]], vertices[triangles[:, 2]])
            )
        )
        / 6.0
    )


@wp.kernel
def _inside(mesh: wp.uint64, points: wp.array[wp.vec3], result: wp.array[wp.int32]):
    i = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh, points[i], 1.0e6)
    result[i] = wp.where(query.result and query.sign < 0.0, 1, 0)


def inside_fraction(vertices, triangles, points: np.ndarray) -> float:
    mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3), indices=wp.array(triangles.reshape(-1), dtype=wp.int32))
    flags = wp.zeros(len(points), dtype=wp.int32)
    wp.launch(_inside, len(points), inputs=[mesh.id, wp.array(points, dtype=wp.vec3), flags])
    return float(flags.numpy().mean())


def groove_load(example, module, frame: int) -> float:
    """Hydroelastic normal load [N] with the wheel placed at the pose of ``frame``."""
    example._set_grinder_pose(module.Example._grinder_pose(frame))
    example.collision_pipeline.collide(example.state_0, example.contacts)
    distance = wp.empty(example.contacts.rigid_contact_max, dtype=wp.float32, device=example.model.device)
    newton.eval_rigid_contact_kinematics(example.model, example.state_0, example.contacts, out_distance=distance)
    count = int(example.contacts.rigid_contact_count.numpy()[0])
    stiffness = example.contacts.rigid_contact_stiffness.numpy()[:count]
    return float(np.sum(np.maximum(-distance.numpy()[:count] * stiffness, 0.0)))


def verify(path: Path) -> dict:
    module = load(path)
    started = time.perf_counter()
    checks = {name: bool(np.allclose(getattr(module, name, None), value)) for name, value in EXPECTED.items()}
    checks["tool_path"] = all(
        np.allclose(np.asarray(module.Example._grinder_pose(f))[:3], reference_pose(f)[0], atol=1e-6)
        for f in (0, 90, 200, 300)
    )
    example = module.Example(newton.viewer.ViewerNull(num_frames=1 << 30), None)
    checks["hydroelastic"] = bool(
        np.allclose(example.model.shape_material_kh.numpy(), EXPECTED["HYDROELASTIC_STIFFNESS"])
    )
    fresh_load = groove_load(example, module, EXPECTED["GRIND_FRAMES"] // 2)
    example._set_grinder_pose(module.Example._grinder_pose(0))
    workpiece = next(i for i, label in enumerate(example.model.shape_label) if "workpiece" in label)
    # Removal is measured against the isomesh of an unground workpiece SDF at the submission's
    # resolution, which cancels most of the extraction's discretization error.
    blank = newton.Mesh.create_ellipsoid(
        *EXPECTED["WORKPIECE_RADII"], num_latitudes=32, num_longitudes=64, compute_normals=False, compute_uvs=False
    ).build_sdf(
        max_resolution=int(getattr(module, "WORKPIECE_RESOLUTION", 256)),
        narrow_band_range=(-0.08, 0.08),
        margin=0.04,
        texture_format="float32",
        paired_samples=False,
    )
    initial = blank.extract_isomesh(device=example.model.device)
    initial_volume = mesh_volume(
        np.asarray(initial.vertices, dtype=np.float64), np.asarray(initial.indices, dtype=np.int32).reshape(-1, 3)
    )
    frames = EXPECTED["GRIND_FRAMES"] + 10
    for _ in range(frames):
        example.step()
    # The geometry the model collides with: the SDF attached to the workpiece mesh shape.
    surface = example.model.shape_source[workpiece].sdf.extract_isomesh(device=example.model.device)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    triangles = np.asarray(surface.indices, dtype=np.int32).reshape(-1, 3)
    volume = mesh_volume(vertices, triangles)
    expected = expected_volume(frames)
    radii = np.asarray(EXPECTED["WORKPIECE_RADII"])
    full = 4.0 / 3.0 * math.pi * float(np.prod(radii))
    xs = np.linspace(-0.3, 0.3, 61)
    groove, base = [], []
    for x in xs:
        center = reference_pose(int(round((x + 0.36) / 0.72 * EXPECTED["GRIND_FRAMES"])))[0]
        bottom = center[2] - EXPECTED["GRINDER_RADIUS"]
        for y in (-0.02, 0.0, 0.02):
            groove.append([x, y, bottom + 0.4 * EXPECTED["GRIND_DEPTH"]])
            base.append([x, y, bottom - 0.02])
    groove_cleared = 1.0 - inside_fraction(vertices, triangles, np.asarray(groove))
    base_kept = inside_fraction(vertices, triangles, np.asarray(base))
    after_load = groove_load(example, module, EXPECTED["GRIND_FRAMES"] // 2)
    metrics = {
        "removed_volume_m3": initial_volume - volume,
        "expected_removed_m3": full - expected,
        "volume_error": abs((initial_volume - volume) - (full - expected)) / (full - expected),
        "groove_cleared": groove_cleared,
        "base_kept": base_kept,
        "groove_load_ratio": after_load / max(fresh_load, 1e-9),
        "fresh_groove_load_N": fresh_load,
        "final_groove_load_N": after_load,
    }
    integrity = all(checks.values())
    success = (
        integrity
        and metrics["volume_error"] <= THRESHOLDS["volume_error"]
        and metrics["groove_cleared"] >= THRESHOLDS["groove_cleared"]
        and metrics["base_kept"] >= THRESHOLDS["base_kept"]
        and metrics["groove_load_ratio"] <= THRESHOLDS["groove_load_ratio"]
    )
    worst = {
        "volume_error": metrics["volume_error"] / THRESHOLDS["volume_error"],
        "groove_cleared": (1.0 - metrics["groove_cleared"]) / (1.0 - THRESHOLDS["groove_cleared"]),
        "base_kept": (1.0 - metrics["base_kept"]) / (1.0 - THRESHOLDS["base_kept"]),
        "groove_load_ratio": metrics["groove_load_ratio"] / THRESHOLDS["groove_load_ratio"],
    }
    return {
        "task": "sdf_grind",
        "success": bool(success),
        "integrity": integrity,
        "failed_checks": [k for k, v in checks.items() if not v],
        "metrics": metrics,
        "normalized_worst": worst,
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
