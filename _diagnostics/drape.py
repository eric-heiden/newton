# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast, parameterizable reproduction of the cloth-over-sphere drape failure.

Mirrors example_cloth_xpbd_rigid_contact but exposes solver knobs so the
persistent-intersection and jitter failure modes can be bisected.

Usage:
    uv run --extra examples _diagnostics/drape.py --iterations 8 --recovery 1
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector


def build(args):
    builder = newton.ModelBuilder()
    contact_cfg = builder.default_shape_cfg.copy()
    contact_cfg.mu = 0.6
    contact_cfg.ka = 0.0
    builder.add_shape_sphere(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.65), wp.quat_identity()),
        radius=0.6,
        cfg=contact_cfg,
    )
    builder.add_ground_plane(cfg=contact_cfg)
    builder.add_cloth_grid(
        pos=wp.vec3(-1.2, -1.2, 2.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=args.dim,
        dim_y=args.dim,
        cell_x=2.4 / args.dim,
        cell_y=2.4 / args.dim,
        mass=1.5e-3,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=2.0,
        edge_ke=10.0,
        edge_kd=0.5,
        particle_radius=0.025,
    )
    model = builder.finalize()
    model.particle_mu = 0.6
    model.soft_contact_mu = 0.6
    model.shape_material_ka.zero_()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--recovery", type=int, default=1)
    parser.add_argument("--self-contact", type=int, default=1)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--label", type=str, default="run")
    args = parser.parse_args()

    model = build(args)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=args.iterations,
        particle_enable_self_contact=bool(args.self_contact),
        particle_enable_triangle_intersection_recovery=bool(args.recovery),
        particle_self_contact_relaxation=args.relaxation,
        particle_self_contact_radius=args.radius,
        particle_self_contact_margin=args.margin,
        particle_damping=args.damping,
    )
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.04, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / args.substeps

    probe = TriMeshCollisionDetector(model)

    def simulate():
        nonlocal state_0, state_1
        solver.rebuild_bvh(state_0)
        for _ in range(args.substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    with wp.ScopedCapture() as capture:
        simulate()
    graph = capture.graph

    rows = []
    for frame in range(args.frames):
        wp.capture_launch(graph)
        if frame % args.sample == 0 or frame == args.frames - 1:
            probe.refit(state_0.particle_q)
            probe.triangle_triangle_intersection_detection()
            counts = probe.triangle_intersecting_triangles_count.numpy()
            speeds = np.linalg.norm(state_0.particle_qd.numpy(), axis=1)
            rows.append(
                {
                    "t": round((frame + 1) * frame_dt, 3),
                    "ix": int(counts.sum() // 2),
                    "p50": round(float(np.percentile(speeds, 50)), 5),
                    "p99": round(float(np.percentile(speeds, 99)), 5),
                    "max": round(float(np.max(speeds)), 5),
                }
            )

    tail = rows[-4:]
    print(
        json.dumps(
            {
                "label": args.label,
                "ix_tail_mean": round(float(np.mean([r["ix"] for r in tail])), 2),
                "p99_tail_mean": round(float(np.mean([r["p99"] for r in tail])), 5),
                "max_tail_mean": round(float(np.mean([r["max"] for r in tail])), 5),
                "rows": rows,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
