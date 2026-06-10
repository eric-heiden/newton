# SPDX-FileCopyrightText: Copyright (c) 2026 Eric Heiden
# SPDX-License-Identifier: Apache-2.0

"""Walk every USD asset, monkey-patch ``compute_inertia_mesh`` to capture every
(vertices, indices, is_solid, density, thickness) tuple it sees, and persist
them to ``meshes.npz``.

This guarantees that the four bench variants compute on byte-identical inputs.

Run once with the V0 environment (current main); the captured meshes are
treated as the canonical input set for every later variant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make sibling shared.py importable when run via "uv run python collect_meshes.py".
sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    DEFAULT_ASSETS,
    USD_ASSETS,
    MeshRecord,
    save_meshes,
)


def _build_recorder() -> tuple[list[MeshRecord], Any]:
    """Build a closure that captures every call into compute_inertia_mesh."""
    captured: list[MeshRecord] = []
    current_robot: dict[str, str] = {"name": "?"}

    import newton._src.geometry.inertia as inertia_module

    original = inertia_module.compute_inertia_mesh

    def wrapper(
        density: float,
        vertices: Any,
        indices: Any,
        is_solid: bool = True,
        thickness: float | list[float] = 0.001,
    ) -> Any:
        verts_np = np.asarray(vertices, dtype=np.float32)
        if verts_np.ndim == 1:
            verts_np = verts_np.reshape(-1, 3)
        inds_np = np.asarray(indices, dtype=np.int32).flatten()
        # Normalize thickness to a scalar; in practice newton always passes a scalar.
        th = float(thickness) if isinstance(thickness, (int, float)) else float(np.asarray(thickness).mean())
        captured.append(
            MeshRecord(
                robot=current_robot["name"],
                body_name=f"shape_{len(captured)}",
                shape_index=len(captured),
                vertices=verts_np,
                indices=inds_np,
                is_solid=bool(is_solid),
                density=float(density),
                thickness=th,
            )
        )
        return original(density, vertices, indices, is_solid, thickness)

    # Patch both the source module and the builder import.
    inertia_module.compute_inertia_mesh = wrapper
    import newton._src.sim.builder as builder_module

    if hasattr(builder_module, "compute_inertia_mesh"):
        builder_module.compute_inertia_mesh = wrapper
    if hasattr(builder_module, "compute_inertia_shape"):
        # compute_inertia_shape forwards to compute_inertia_mesh internally;
        # the module-level reference inside compute_inertia_shape was bound at
        # import time, so we also patch the symbol it sees.
        import newton._src.geometry.inertia as inertia_mod2

        inertia_mod2.compute_inertia_mesh = wrapper

    return captured, current_robot


def load_robot(robot_key: str, current_robot: dict[str, str]) -> None:
    """Load a single robot via newton.ModelBuilder.add_usd to trigger inertia calls."""
    import newton
    import newton.utils
    from newton.solvers import SolverMuJoCo
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton

    current_robot["name"] = robot_key
    cfg = USD_ASSETS[robot_key]
    asset_root = newton.utils.download_asset(cfg["asset_folder"])
    usd_path = asset_root / cfg["scene_file"]
    if not usd_path.exists():
        raise FileNotFoundError(f"USD not found: {usd_path}")

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(
        str(usd_path),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
    )
    # We don't need to finalize — add_usd already triggered the inertia path
    # for every mesh shape with non-zero density.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--limit-per-robot", type=int, default=0, help="Cap meshes per robot (0=no cap)")
    args = parser.parse_args()

    captured, current = _build_recorder()

    for robot in args.assets:
        before = len(captured)
        try:
            load_robot(robot, current)
        except Exception as e:
            print(f"[WARN] Failed to load {robot}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        after = len(captured)
        added = after - before
        if args.limit_per_robot > 0 and added > args.limit_per_robot:
            # Drop excess
            captured[:] = captured[:before] + captured[before : before + args.limit_per_robot]
        print(f"[{robot}] captured {min(added, args.limit_per_robot or added)} meshes "
              f"(total triangles: {sum(m.num_tris for m in captured[before:])})")

    if not captured:
        print("No meshes captured. Aborting.", file=sys.stderr)
        sys.exit(2)

    save_meshes(captured)
    print()
    print(f"Saved {len(captured)} mesh records to scripts/inertia_bench/meshes.npz")
    print(f"Total triangles: {sum(m.num_tris for m in captured):,}")
    tris_by_robot: dict[str, int] = {}
    for m in captured:
        tris_by_robot[m.robot] = tris_by_robot.get(m.robot, 0) + m.num_tris
    print("Per-robot triangle totals:")
    for k, v in sorted(tris_by_robot.items()):
        print(f"  {k:24s} {v:>10,d}")


if __name__ == "__main__":
    main()
