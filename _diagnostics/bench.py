# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock comparison of SolverXPBD and SolverVBD on an identical cloth drape.

Usage:
    uv run --extra examples _diagnostics/bench.py --dim 64
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

import newton


def build(dim: int, color: bool = False):
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = 0.6
    cfg.ka = 0.0
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.65), wp.quat_identity()), radius=0.6, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    builder.add_cloth_grid(
        pos=wp.vec3(-1.2, -1.2, 1.6),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        cell_x=2.4 / dim,
        cell_y=2.4 / dim,
        mass=1.5e-3,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=2.0,
        edge_ke=10.0,
        edge_kd=0.5,
        particle_radius=0.025,
    )
    if color:
        # VBD requires a graph coloring of the particle mesh.
        builder.color()
    model = builder.finalize()
    model.particle_mu = 0.6
    model.soft_contact_mu = 0.6
    model.shape_material_ka.zero_()
    return model


def bench(name: str, model, solver, substeps: int, frames: int):
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.04, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    sim_dt = 1.0 / 60.0 / substeps

    def simulate():
        nonlocal state_0, state_1
        if getattr(solver, "trimesh_collision_detector", None) is not None:
            solver.rebuild_bvh(state_0)
        for _ in range(substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    with wp.ScopedCapture() as capture:
        simulate()
    graph = capture.graph

    for _ in range(10):
        wp.capture_launch(graph)
    wp.synchronize_device()

    times = []
    for _ in range(frames):
        start = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize_device()
        times.append((time.perf_counter() - start) * 1e3)

    speeds = np.linalg.norm(state_0.particle_qd.numpy(), axis=1)
    finite = bool(np.all(np.isfinite(state_0.particle_q.numpy())))
    print(
        f"{name:28s} {np.median(times):8.3f} ms/frame  p95={np.percentile(times, 95):8.3f}  "
        f"final_p99_speed={np.percentile(speeds, 99):7.4f}  finite={finite}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    model = build(args.dim)
    print(f"cloth {args.dim}x{args.dim} = {model.particle_count} particles, {model.tri_count} triangles")

    bench(
        "XPBD self-contact",
        model,
        newton.solvers.SolverXPBD(
            model,
            iterations=8,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.05,
            particle_self_contact_margin=0.08,
        ),
        substeps=4,
        frames=args.frames,
    )

    model_recovery = build(args.dim)
    bench(
        "XPBD + exact recovery",
        model_recovery,
        newton.solvers.SolverXPBD(
            model_recovery,
            iterations=8,
            particle_enable_self_contact=True,
            particle_enable_triangle_intersection_recovery=True,
            particle_self_contact_radius=0.05,
            particle_self_contact_margin=0.08,
        ),
        substeps=4,
        frames=args.frames,
    )

    model_vbd = build(args.dim, color=True)
    bench(
        "VBD self-contact",
        model_vbd,
        newton.solvers.SolverVBD(
            model_vbd,
            iterations=8,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.05,
            particle_self_contact_margin=0.08,
        ),
        substeps=4,
        frames=args.frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
