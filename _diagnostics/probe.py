# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quantitative probe for XPBD cloth examples.

Runs an example headless and reports, per sampled frame: triangle-triangle
intersection count, particle speed percentiles (jitter), and wall-clock cost.

Usage:
    uv run --extra examples _diagnostics/probe.py cloth_xpbd_self_contact --frames 300
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector

EXAMPLE_MODULES = {
    "cloth_xpbd_self_contact": "newton.examples.cloth.example_cloth_xpbd_self_contact",
    "cloth_xpbd_hanging": "newton.examples.cloth.example_cloth_xpbd_hanging",
    "cloth_xpbd_rigid_contact": "newton.examples.cloth.example_cloth_xpbd_rigid_contact",
    "cloth_xpbd_picking": "newton.examples.cloth.example_cloth_xpbd_picking",
    "cloth_xpbd_gripper": "newton.examples.cloth.example_cloth_xpbd_gripper",
}


class IntersectionProbe:
    """Counts triangle intersections with a private detector.

    A separate detector is essential: reusing the solver's would overwrite the
    intersection buffers it consumes across iterations, silently improving the
    very behavior being measured.
    """

    def __init__(self, model):
        self.detector = TriMeshCollisionDetector(model) if model.tri_count else None

    def count(self, state) -> int | None:
        if self.detector is None:
            return None
        self.detector.refit(state.particle_q)
        self.detector.triangle_triangle_intersection_detection()
        return int(self.detector.triangle_intersecting_triangles_count.numpy().sum() // 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("example")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sample", type=int, default=20, help="sample every N frames")
    parser.add_argument("--json", type=str, default=None)
    probe_args = parser.parse_args()

    module = importlib.import_module(EXAMPLE_MODULES[probe_args.example])

    example_parser = newton.examples.create_parser()
    example_parser.set_defaults(num_frames=probe_args.frames)
    sys.argv = [sys.argv[0], "--viewer", "null", "--num-frames", str(probe_args.frames)]
    viewer, args = newton.examples.init(example_parser)
    example = module.Example(viewer, args)

    intersections = IntersectionProbe(example.model)

    samples = []
    step_times = []
    for frame in range(probe_args.frames):
        start = time.perf_counter()
        example.step()
        wp.synchronize_device()
        step_times.append((time.perf_counter() - start) * 1e3)

        if frame % probe_args.sample == 0 or frame == probe_args.frames - 1:
            speeds = np.linalg.norm(example.state_0.particle_qd.numpy(), axis=1)
            positions = example.state_0.particle_q.numpy()
            samples.append(
                {
                    "frame": frame,
                    "time": round(example.sim_time, 4),
                    "intersections": intersections.count(example.state_0),
                    "speed_p50": round(float(np.percentile(speeds, 50)), 5),
                    "speed_p99": round(float(np.percentile(speeds, 99)), 5),
                    "speed_max": round(float(np.max(speeds)), 5),
                    "z_min": round(float(np.min(positions[:, 2])), 5),
                    "finite": bool(np.all(np.isfinite(positions))),
                }
            )

    step_times_np = np.array(step_times)
    summary = {
        "example": probe_args.example,
        "particles": int(example.model.particle_count),
        "triangles": int(example.model.tri_count),
        "substeps": int(example.sim_substeps),
        "ms_per_frame_median": round(float(np.median(step_times_np)), 3),
        "ms_per_frame_p95": round(float(np.percentile(step_times_np, 95)), 3),
        "samples": samples,
    }

    print(json.dumps(summary, indent=1))
    if probe_args.json:
        with open(probe_args.json, "w") as handle:
            json.dump(summary, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
