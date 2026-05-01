# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental helpers for the ROB-108 differentiable collision-query spike."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import warp as wp

from newton import GeoType

from .simplex_solver import create_solve_closest_distance
from .support_function import GenericShapeData, SupportMapDataProvider, support_map

MAX_GJK_ITERATIONS = 30
SPIKE_TARGET_MODULES = (
    "newton/_src/geometry/simplex_solver.py",
    "newton/_src/geometry/support_function.py",
    "newton/_src/geometry/collision_convex.py",
)
SPIKE_DECISION_GATE = (
    "If translation-sweep finite differences stay aligned with the returned GJK normal for "
    "sphere-box and capsule-box, expand the spike to witness-point exposure and rotated box-box cases. "
    "If they do not, add a smoothing or branch-tracking layer before solver integration."
)
SPIKE_INTERFACE_READOUT = {
    "recommended_api_location": "Keep the experiment in newton._src.geometry until witness-point semantics stabilize.",
    "required_internal_boundary": (
        "Introduce an internal closest-feature query helper beside simplex_solver.py that returns "
        "distance, normal, midpoint, and witness points without exposing solver-facing policy."
    ),
    "must_stay_internal": (
        "Witness-point ordering, overlap semantics, and non-convex coverage must remain internal until "
        "rotated convex primitives and near-contact continuity are proven stable."
    ),
    "solver_integration_recommendation": (
        "Go for internal-only solver prototyping via a geometry helper that returns witness pairs; "
        "no-go for promoting this to the public Newton API or general mesh support yet."
    ),
}
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class QuerySweepSpec:
    """Host-side description of one primitive-pair translation experiment.

    Args:
        name: Stable experiment name for reports.
        geom_a_type: Geometry type for shape A.
        geom_a_size: Shape A size parameters in Newton support-map convention.
        geom_a_pos: World-space center for shape A [m].
        geom_b_type: Geometry type for shape B.
        geom_b_size: Shape B size parameters in Newton support-map convention.
        geom_b_base_pos: Base world-space center for shape B [m].
        translation_axis: Coordinate axis for the translation sweep.
        offsets: Translation offsets along `translation_axis` [m].
        geom_a_orientation: World-space quaternion for shape A `(x, y, z, w)`.
        geom_b_orientation: World-space quaternion for shape B `(x, y, z, w)`.
    """

    name: str
    geom_a_type: int
    geom_a_size: tuple[float, float, float]
    geom_a_pos: tuple[float, float, float]
    geom_b_type: int
    geom_b_size: tuple[float, float, float]
    geom_b_base_pos: tuple[float, float, float]
    translation_axis: int
    offsets: tuple[float, ...]
    geom_a_orientation: tuple[float, float, float, float] = _IDENTITY_QUAT
    geom_b_orientation: tuple[float, float, float, float] = _IDENTITY_QUAT


def _as_vec3(values: tuple[float, float, float]) -> wp.vec3:
    return wp.vec3(values[0], values[1], values[2])


def _as_quat(values: tuple[float, float, float, float]) -> wp.quat:
    return wp.quat(values[0], values[1], values[2], values[3])


def _as_list(values: wp.vec3) -> list[float]:
    return [float(values[0]), float(values[1]), float(values[2])]


def _quat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
    half_angle = 0.5 * angle
    sin_half = math.sin(half_angle)
    return (
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        math.cos(half_angle),
    )


@wp.kernel
def _closest_features_kernel(
    type_a: int,
    size_a: wp.vec3,
    orientation_a: wp.quat,
    pos_a: wp.vec3,
    type_b: int,
    size_b: wp.vec3,
    orientation_b: wp.quat,
    pos_b: wp.vec3,
    collision_out: wp.array(dtype=int),
    distance_out: wp.array(dtype=float),
    point_a_out: wp.array(dtype=wp.vec3),
    point_b_out: wp.array(dtype=wp.vec3),
    point_out: wp.array(dtype=wp.vec3),
    normal_out: wp.array(dtype=wp.vec3),
):
    """Compute closest-feature data through Newton's internal GJK solver."""

    shape_a = GenericShapeData()
    shape_a.shape_type = type_a
    shape_a.scale = size_a
    shape_a.auxiliary = wp.vec3(0.0)

    shape_b = GenericShapeData()
    shape_b.shape_type = type_b
    shape_b.scale = size_b
    shape_b.auxiliary = wp.vec3(0.0)

    data_provider = SupportMapDataProvider()

    relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
    relative_position_b = wp.quat_rotate_inv(orientation_a, pos_b - pos_a)

    separated, point_a_local, point_b_local, normal_local, distance = wp.static(
        create_solve_closest_distance(support_map).core
    )(
        shape_a,
        shape_b,
        relative_orientation_b,
        relative_position_b,
        0.0,
        data_provider,
        MAX_GJK_ITERATIONS,
        1.0e-6,
    )

    point_a = wp.quat_rotate(orientation_a, point_a_local) + pos_a
    point_b = wp.quat_rotate(orientation_a, point_b_local) + pos_a
    point = 0.5 * (point_a + point_b)
    normal = wp.quat_rotate(orientation_a, normal_local)

    collision_out[0] = int(not separated)
    distance_out[0] = distance
    point_a_out[0] = point_a
    point_b_out[0] = point_b
    point_out[0] = point
    normal_out[0] = normal


def query_closest_distance(
    geom_a_type: int,
    geom_a_size: tuple[float, float, float],
    geom_a_pos: tuple[float, float, float],
    geom_b_type: int,
    geom_b_size: tuple[float, float, float],
    geom_b_pos: tuple[float, float, float],
    geom_a_orientation: tuple[float, float, float, float] = _IDENTITY_QUAT,
    geom_b_orientation: tuple[float, float, float, float] = _IDENTITY_QUAT,
) -> dict[str, Any]:
    """Run one closest-distance query for a convex primitive pair.

    Args:
        geom_a_type: Geometry type for shape A.
        geom_a_size: Shape A size parameters in Newton support-map convention.
        geom_a_pos: World-space center for shape A [m].
        geom_b_type: Geometry type for shape B.
        geom_b_size: Shape B size parameters in Newton support-map convention.
        geom_b_pos: World-space center for shape B [m].
        geom_a_orientation: World-space quaternion for shape A `(x, y, z, w)`.
        geom_b_orientation: World-space quaternion for shape B `(x, y, z, w)`.

    Returns:
        JSON-serializable query result containing collision, distance, midpoint,
        witness points, and normal.
    """

    collision_out = wp.zeros(1, dtype=int)
    distance_out = wp.zeros(1, dtype=float)
    point_a_out = wp.zeros(1, dtype=wp.vec3)
    point_b_out = wp.zeros(1, dtype=wp.vec3)
    point_out = wp.zeros(1, dtype=wp.vec3)
    normal_out = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
        _closest_features_kernel,
        dim=1,
        inputs=[
            int(geom_a_type),
            _as_vec3(geom_a_size),
            _as_quat(geom_a_orientation),
            _as_vec3(geom_a_pos),
            int(geom_b_type),
            _as_vec3(geom_b_size),
            _as_quat(geom_b_orientation),
            _as_vec3(geom_b_pos),
        ],
        outputs=[collision_out, distance_out, point_a_out, point_b_out, point_out, normal_out],
    )

    return {
        "collision": int(collision_out.numpy()[0]),
        "distance": float(distance_out.numpy()[0]),
        "point_a": _as_list(point_a_out.numpy()[0]),
        "point_b": _as_list(point_b_out.numpy()[0]),
        "point": _as_list(point_out.numpy()[0]),
        "normal": _as_list(normal_out.numpy()[0]),
    }


def estimate_translation_gradient(spec: QuerySweepSpec, offset: float, epsilon: float = 1.0e-3) -> float:
    """Estimate distance gradient for shape-B translation with a central difference.

    Args:
        spec: Query sweep specification.
        offset: Translation offset along the configured axis [m].
        epsilon: Finite-difference half-step [m].

    Returns:
        Central-difference gradient `d(distance) / d(offset)`.
    """

    pos_minus = list(spec.geom_b_base_pos)
    pos_plus = list(spec.geom_b_base_pos)
    pos_minus[spec.translation_axis] += offset - epsilon
    pos_plus[spec.translation_axis] += offset + epsilon

    dist_minus = query_closest_distance(
        spec.geom_a_type,
        spec.geom_a_size,
        spec.geom_a_pos,
        spec.geom_b_type,
        spec.geom_b_size,
        tuple(pos_minus),
        spec.geom_a_orientation,
        spec.geom_b_orientation,
    )["distance"]
    dist_plus = query_closest_distance(
        spec.geom_a_type,
        spec.geom_a_size,
        spec.geom_a_pos,
        spec.geom_b_type,
        spec.geom_b_size,
        tuple(pos_plus),
        spec.geom_a_orientation,
        spec.geom_b_orientation,
    )["distance"]
    return (dist_plus - dist_minus) / (2.0 * epsilon)


def sample_translation_sweep(spec: QuerySweepSpec) -> list[dict[str, Any]]:
    """Collect distance, normal, and finite-difference gradient over a translation sweep.

    Args:
        spec: Query sweep specification.

    Returns:
        Sweep samples ordered by offset.
    """

    samples: list[dict[str, Any]] = []
    for offset in spec.offsets:
        geom_b_pos = list(spec.geom_b_base_pos)
        geom_b_pos[spec.translation_axis] += offset
        result = query_closest_distance(
            spec.geom_a_type,
            spec.geom_a_size,
            spec.geom_a_pos,
            spec.geom_b_type,
            spec.geom_b_size,
            tuple(geom_b_pos),
            spec.geom_a_orientation,
            spec.geom_b_orientation,
        )
        gradient = estimate_translation_gradient(spec, offset)
        samples.append(
            {
                "offset": float(offset),
                "distance": result["distance"],
                "collision": result["collision"],
                "point_a": result["point_a"],
                "point_b": result["point_b"],
                "point": result["point"],
                "normal": result["normal"],
                "finite_difference_gradient": float(gradient),
            }
        )
    return samples


def default_spike_specs() -> tuple[QuerySweepSpec, ...]:
    """Return the first primitive-pair experiments for the differentiable query spike."""

    return (
        QuerySweepSpec(
            name="sphere_box_translation_x",
            geom_a_type=int(GeoType.SPHERE),
            geom_a_size=(0.5, 0.0, 0.0),
            geom_a_pos=(0.0, 0.0, 0.0),
            geom_b_type=int(GeoType.BOX),
            geom_b_size=(0.75, 0.5, 0.5),
            geom_b_base_pos=(1.5, 0.0, 0.0),
            translation_axis=0,
            offsets=(0.0, 0.2, 0.4),
        ),
        QuerySweepSpec(
            name="capsule_box_translation_x",
            geom_a_type=int(GeoType.CAPSULE),
            geom_a_size=(0.25, 0.6, 0.0),
            geom_a_pos=(0.0, 0.0, 0.0),
            geom_b_type=int(GeoType.BOX),
            geom_b_size=(0.45, 0.45, 0.45),
            geom_b_base_pos=(1.1, 0.0, 0.0),
            translation_axis=0,
            offsets=(0.0, 0.15, 0.3),
        ),
        QuerySweepSpec(
            name="box_box_rotated_translation_x",
            geom_a_type=int(GeoType.BOX),
            geom_a_size=(0.5, 0.35, 0.25),
            geom_a_pos=(0.0, 0.0, 0.0),
            geom_b_type=int(GeoType.BOX),
            geom_b_size=(0.45, 0.3, 0.25),
            geom_b_base_pos=(1.15, 0.1, 0.0),
            translation_axis=0,
            offsets=(0.0, 0.15, 0.3),
            geom_b_orientation=_quat_from_axis_angle((0.0, 0.0, 1.0), math.pi / 4.0),
        ),
    )


def build_differentiable_query_spike_report() -> dict[str, Any]:
    """Build a JSON-serializable report for the current spike scope."""

    specs = default_spike_specs()
    return {
        "artifact": "rob-108-differentiable-query-spike",
        "target_modules": list(SPIKE_TARGET_MODULES),
        "first_shape_pairs": [spec.name for spec in specs],
        "decision_gate": SPIKE_DECISION_GATE,
        "interface_readout": dict(SPIKE_INTERFACE_READOUT),
        "sweeps": [{"name": spec.name, "samples": sample_translation_sweep(spec)} for spec in specs],
    }


def make_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the spike report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the spike report to stdout or a JSON file."""

    args = make_argument_parser().parse_args(argv)
    report = build_differentiable_query_spike_report()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
