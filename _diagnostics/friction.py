# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coulomb friction validation for XPBD cloth.

A cloth patch is released on a plane inclined by ``theta``. Coulomb's law says
it must hold when ``mu > tan(theta)`` and slide when ``mu < tan(theta)``, so
sweeping mu across tan(theta) exposes whether friction is physical or merely
damped sliding.

Two modes:
  rigid -- cloth directly on an inclined rigid plane (cloth/rigid friction)
  self  -- cloth on top of a second, pinned cloth layer (cloth/cloth friction)

Usage:
    uv run --extra examples _diagnostics/friction.py --mode rigid --theta 30
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import warp as wp

import newton


def build_rigid(theta_rad: float, mu: float, args):
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = mu
    cfg.ka = 0.0
    # Incline about +Y so gravity drives motion along +X downhill.
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta_rad)
    builder.add_shape_plane(body=-1, xform=wp.transform(wp.vec3(0.0), rot), width=60.0, length=60.0, cfg=cfg)

    size = args.dim * args.cell
    # Seat the patch just above the incline surface at its own centre.
    builder.add_cloth_grid(
        pos=wp.vec3(
            -0.5 * size * math.cos(theta_rad) + 0.02 * math.sin(theta_rad),
            -0.5 * size,
            0.5 * size * math.sin(theta_rad) + 0.02 * math.cos(theta_rad),
        ),
        rot=rot,
        vel=wp.vec3(0.0),
        dim_x=args.dim,
        dim_y=args.dim,
        cell_x=args.cell,
        cell_y=args.cell,
        mass=1.0e-3,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=2.0,
        edge_ke=1.0,
        edge_kd=0.1,
        particle_radius=0.01,
    )
    model = builder.finalize()
    model.soft_contact_mu = mu
    model.particle_mu = mu
    model.shape_material_ka.zero_()
    return model, None


def build_self(theta_rad: float, mu: float, args):
    """Sliding layer resting on a pinned layer; the plane is frictionless."""
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = 0.0
    cfg.ka = 0.0
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta_rad)
    builder.add_shape_plane(body=-1, xform=wp.transform(wp.vec3(0.0), rot), width=60.0, length=60.0, cfg=cfg)

    size = args.dim * args.cell
    normal = wp.vec3(math.sin(theta_rad), 0.0, math.cos(theta_rad))

    def origin(lift, pad):
        return wp.vec3(
            -0.5 * (size + pad) * math.cos(theta_rad) + lift * normal[0],
            -0.5 * (size + pad),
            0.5 * (size + pad) * math.sin(theta_rad) + lift * normal[2],
        )

    common = {
        "rot": rot,
        "vel": wp.vec3(0.0),
        "cell_x": args.cell,
        "cell_y": args.cell,
        "mass": 1.0e-3,
        "tri_ke": 1.0e4,
        "tri_ka": 1.0e4,
        "tri_kd": 2.0,
        "edge_ke": 1.0,
        "edge_kd": 0.1,
        "particle_radius": 0.01,
    }
    # Lower layer is wider so the upper layer never runs off its edge.
    lower_first = len(builder.particle_q)
    builder.add_cloth_grid(pos=origin(0.012, 4 * args.cell), dim_x=args.dim + 4, dim_y=args.dim + 4, **common)
    lower_ids = np.arange(lower_first, len(builder.particle_q), dtype=np.int32)

    upper_first = len(builder.particle_q)
    builder.add_cloth_grid(pos=origin(0.045, 0.0), dim_x=args.dim, dim_y=args.dim, **common)
    upper_ids = np.arange(upper_first, len(builder.particle_q), dtype=np.int32)

    model = builder.finalize()
    model.soft_contact_mu = 0.0
    model.particle_mu = mu
    model.shape_material_ka.zero_()
    # Pin the lower layer so only cloth/cloth friction can hold the upper one.
    inv_mass = model.particle_inv_mass.numpy()
    inv_mass[lower_ids] = 0.0
    model.particle_inv_mass.assign(inv_mass)
    return model, upper_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rigid", "self"), default="rigid")
    parser.add_argument("--theta", type=float, default=30.0)
    parser.add_argument("--mu", type=float, nargs="+", default=[0.2, 0.4, 0.5, 0.7, 0.9])
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    theta = math.radians(args.theta)
    results = []
    for mu in args.mu:
        if args.mode == "rigid":
            model, tracked = build_rigid(theta, mu, args)
        else:
            model, tracked = build_self(theta, mu, args)

        solver = newton.solvers.SolverXPBD(
            model,
            iterations=args.iterations,
            particle_enable_self_contact=(args.mode == "self"),
            particle_self_contact_radius=0.012 if args.mode == "self" else None,
            particle_self_contact_margin=0.03 if args.mode == "self" else None,
        )
        pipeline = newton.CollisionPipeline(
            model, soft_contact_margin=0.03, enable_rigid_soft_full_surface_contact=True
        )
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        sim_dt = 1.0 / 60.0 / args.substeps

        if tracked is None:
            tracked = np.arange(model.particle_count, dtype=np.int32)
        start = state_0.particle_q.numpy()[tracked].mean(axis=0)

        for _ in range(args.frames):
            if solver.trimesh_collision_detector is not None:
                solver.rebuild_bvh(state_0)
            for _ in range(args.substeps):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

        end = state_0.particle_q.numpy()[tracked].mean(axis=0)
        # Downhill unit direction along the incline surface.
        downhill = np.array([math.cos(theta), 0.0, -math.sin(theta)])
        slide = float(np.dot(end - start, downhill))
        speeds = np.linalg.norm(state_0.particle_qd.numpy()[tracked], axis=1)
        duration = args.frames / 60.0
        accel = 9.81 * (math.sin(theta) - mu * math.cos(theta))
        analytic = 0.5 * max(accel, 0.0) * duration * duration
        results.append(
            {
                "mu": mu,
                "analytic_m": round(analytic, 4),
                "slide_m": round(slide, 4),
                "final_speed": round(float(np.mean(speeds)), 4),
                "finite": bool(np.all(np.isfinite(end))),
            }
        )

    print(
        "JSONOUT"
        + json.dumps(
            {"mode": args.mode, "theta_deg": args.theta, "tan_theta": round(math.tan(theta), 4), "results": results}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
