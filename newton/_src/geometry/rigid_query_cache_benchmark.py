# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal ROB-138 benchmark for cached rigid-query logging on primitive contact."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton

BENCHMARK_ARTIFACT = "rob-138-rigid-query-cache-benchmark"
BENCHMARK_ENTRYPOINT = "python -m newton._src.geometry.rigid_query_cache_benchmark"
BENCHMARK_TOUCH_POINTS = (
    "newton/_src/sim/collide.py",
    "newton/_src/sim/contacts.py",
    "newton/_src/geometry/rigid_query_cache_benchmark.py",
)


@dataclass(frozen=True)
class RigidQueryCacheBenchmarkSpec:
    """Configuration for a primitive cached-query sweep.

    Args:
        name: Stable benchmark name.
        step_count: Number of sampled positions.
        sphere_radius: Sphere radius [m].
        box_half_extents: Box half extents [m].
        sphere_start_x: Initial sphere center position along x [m].
        sphere_end_x: Final sphere center position along x [m].
    """

    name: str = "sphere_box_cache_sweep"
    step_count: int = 33
    sphere_radius: float = 0.25
    box_half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5)
    sphere_start_x: float = 0.74
    sphere_end_x: float = 0.58


def default_benchmark_spec() -> RigidQueryCacheBenchmarkSpec:
    """Return the default cached-query benchmark configuration."""

    return RigidQueryCacheBenchmarkSpec()


def _build_model(
    spec: RigidQueryCacheBenchmarkSpec,
    device: wp.context.Devicelike | None = None,
) -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.request_contact_attributes("rigid_query")
    builder.add_shape_box(
        body=-1,
        hx=spec.box_half_extents[0],
        hy=spec.box_half_extents[1],
        hz=spec.box_half_extents[2],
        label="cache_box",
    )
    sphere_body = builder.add_body(
        xform=wp.transform(wp.vec3(spec.sphere_start_x, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        label="cache_sphere",
    )
    builder.add_shape_sphere(body=sphere_body, radius=spec.sphere_radius, label="cache_sphere_shape")
    return builder.finalize(device=device), sphere_body


def _extract_query_record(contacts: newton.Contacts) -> dict[str, Any]:
    query_count = int(contacts.rigid_query_count.numpy()[0])
    rigid_contact_count = int(contacts.rigid_contact_count.numpy()[0])
    if query_count == 0:
        raise RuntimeError("Benchmark expected at least one rigid query record per step.")

    shape0 = contacts.rigid_query_shape0.numpy()[:query_count]
    shape1 = contacts.rigid_query_shape1.numpy()[:query_count]
    point0 = contacts.rigid_query_point0.numpy()[:query_count]
    point1 = contacts.rigid_query_point1.numpy()[:query_count]
    normal = contacts.rigid_query_normal.numpy()[:query_count]
    distance = contacts.rigid_query_distance.numpy()[:query_count]
    active = contacts.rigid_query_active.numpy()[:query_count]

    return {
        "rigid_contact_count": rigid_contact_count,
        "rigid_query_count": query_count,
        "shape0": int(shape0[0]),
        "shape1": int(shape1[0]),
        "point0": [float(value) for value in point0[0]],
        "point1": [float(value) for value in point1[0]],
        "normal": [float(value) for value in normal[0]],
        "distance": float(distance[0]),
        "active": int(active[0]),
    }


def run_benchmark(
    spec: RigidQueryCacheBenchmarkSpec | None = None,
    *,
    device: wp.context.Devicelike | None = None,
) -> dict[str, Any]:
    """Run the primitive cached-query sweep and return a JSON-ready artifact."""

    spec = spec or default_benchmark_spec()
    model, sphere_body = _build_model(spec, device=device)
    state = model.state()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    body_q = state.body_q.numpy()
    offsets = np.linspace(spec.sphere_start_x, spec.sphere_end_x, spec.step_count, dtype=np.float32)

    samples: list[dict[str, Any]] = []
    previous_distance: float | None = None
    for step_index, sphere_x in enumerate(offsets):
        body_q[sphere_body] = np.array([sphere_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        state.body_q.assign(body_q)
        model.collide(state, contacts)

        query_record = _extract_query_record(contacts)
        sample = {
            "step": step_index,
            "sphere_center_x": float(sphere_x),
            "distance_delta": None if previous_distance is None else float(query_record["distance"] - previous_distance),
            **query_record,
        }
        samples.append(sample)
        previous_distance = query_record["distance"]

    distances = [sample["distance"] for sample in samples]
    active_steps = [sample["step"] for sample in samples if sample["active"] == 1]
    normals = [sample["normal"] for sample in samples]

    return {
        "artifact": BENCHMARK_ARTIFACT,
        "entrypoint": BENCHMARK_ENTRYPOINT,
        "touch_points": list(BENCHMARK_TOUCH_POINTS),
        "benchmark": {
            "name": spec.name,
            "step_count": spec.step_count,
            "sphere_radius": spec.sphere_radius,
            "box_half_extents": list(spec.box_half_extents),
            "sphere_start_x": spec.sphere_start_x,
            "sphere_end_x": spec.sphere_end_x,
        },
        "summary": {
            "query_pair": [samples[0]["shape0"], samples[0]["shape1"]],
            "min_distance": float(min(distances)),
            "max_distance": float(max(distances)),
            "first_active_step": active_steps[0] if active_steps else None,
            "last_active_step": active_steps[-1] if active_steps else None,
            "active_step_count": len(active_steps),
            "inactive_step_count": len(samples) - len(active_steps),
            "normal_x_range": [float(min(normal[0] for normal in normals)), float(max(normal[0] for normal in normals))],
        },
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and optionally write the artifact to disk."""

    parser = argparse.ArgumentParser(description="Benchmark cached rigid-query logging for primitive contact.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    artifact = run_benchmark()
    payload = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
