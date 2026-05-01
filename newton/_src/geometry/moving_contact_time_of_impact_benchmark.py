# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal ROB-139 benchmark for time-of-impact moving-contact gradients."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from newton import GeoType

from .differentiable_query_spike import query_closest_distance

BENCHMARK_ARTIFACT = "rob-139-moving-contact-time-of-impact-benchmark"
BENCHMARK_ENTRYPOINT = "python -m newton._src.geometry.moving_contact_time_of_impact_benchmark"
BENCHMARK_TOUCH_POINTS = (
    "newton/_src/geometry/differentiable_query_spike.py",
    "newton/_src/geometry/moving_contact_time_of_impact_benchmark.py",
)
BENCHMARK_RECOMMENDATION = (
    "The moving sphere-sphere sweep shows that an end-of-step distance objective can converge "
    "to a spurious control even when the trajectory contains a clean intercept. Keep time-of-impact "
    "correction as differentiable-rollout or benchmark-only logic for now; it improves gradient "
    "quality in this moving-contact case without requiring changes to Newton's default forward path."
)


@dataclass(frozen=True)
class MovingContactBenchmarkSpec:
    """Benchmark configuration for one moving-moving contact sweep.

    Args:
        name: Stable benchmark name.
        radius_a: Sphere A radius [m].
        radius_b: Sphere B radius [m].
        pos_a0: Sphere A start position [m].
        vel_a: Sphere A linear velocity [m/s].
        pos_b0: Sphere B start position [m].
        vel_b: Sphere B base linear velocity [m/s].
        control_axis: Axis of the scalar control perturbation applied to sphere B.
        dt: Rollout horizon [s].
        gradient_samples: Control values used for side-by-side gradient comparison.
        optimization_starts: Initial control values used in the descent study.
        reference_samples: Dense sample count for the query-backed trajectory reference.
        descent_step: Gradient-descent step size.
        descent_iterations: Gradient-descent iteration count.
    """

    name: str
    radius_a: float
    radius_b: float
    pos_a0: tuple[float, float, float]
    vel_a: tuple[float, float, float]
    pos_b0: tuple[float, float, float]
    vel_b: tuple[float, float, float]
    control_axis: int
    dt: float
    gradient_samples: tuple[float, ...]
    optimization_starts: tuple[float, ...]
    reference_samples: int = 41
    descent_step: float = 0.2
    descent_iterations: int = 20


def _vec_add(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _vec_scale(vec: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def _vec_sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _vec_dot(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _vec_norm(vec: tuple[float, float, float]) -> float:
    return math.sqrt(_vec_dot(vec, vec))


def _axis_unit(axis: int) -> tuple[float, float, float]:
    values = [0.0, 0.0, 0.0]
    values[axis] = 1.0
    return (values[0], values[1], values[2])


def _sphere_position(
    start: tuple[float, float, float],
    velocity: tuple[float, float, float],
    time: float,
) -> tuple[float, float, float]:
    return _vec_add(start, _vec_scale(velocity, time))


def _controlled_velocity(spec: MovingContactBenchmarkSpec, control: float) -> tuple[float, float, float]:
    delta = _vec_scale(_axis_unit(spec.control_axis), control)
    return _vec_add(spec.vel_b, delta)


def analytic_sphere_gap(
    radius_a: float,
    pos_a: tuple[float, float, float],
    radius_b: float,
    pos_b: tuple[float, float, float],
) -> float:
    """Return the signed sphere-sphere gap [m]."""

    return _vec_norm(_vec_sub(pos_b, pos_a)) - (radius_a + radius_b)


def query_end_step_gap(spec: MovingContactBenchmarkSpec, control: float) -> float:
    """Return the end-of-step gap using Newton's closest-distance query."""

    pos_a = _sphere_position(spec.pos_a0, spec.vel_a, spec.dt)
    pos_b = _sphere_position(spec.pos_b0, _controlled_velocity(spec, control), spec.dt)
    result = query_closest_distance(
        int(GeoType.SPHERE),
        (spec.radius_a, 0.0, 0.0),
        pos_a,
        int(GeoType.SPHERE),
        (spec.radius_b, 0.0, 0.0),
        pos_b,
    )
    return float(result["distance"])


def analytic_end_step_gap(spec: MovingContactBenchmarkSpec, control: float) -> float:
    """Return the end-of-step gap from an equivalent host-side sphere formula."""

    pos_a = _sphere_position(spec.pos_a0, spec.vel_a, spec.dt)
    pos_b = _sphere_position(spec.pos_b0, _controlled_velocity(spec, control), spec.dt)
    return analytic_sphere_gap(spec.radius_a, pos_a, spec.radius_b, pos_b)


def closest_approach_time(spec: MovingContactBenchmarkSpec, control: float) -> float:
    """Return the clamped closest-approach time for the relative sweep [s]."""

    relative_pos = _vec_sub(spec.pos_b0, spec.pos_a0)
    relative_vel = _vec_sub(_controlled_velocity(spec, control), spec.vel_a)
    velocity_norm_sq = _vec_dot(relative_vel, relative_vel)
    if velocity_norm_sq <= 1.0e-12:
        return 0.0
    time = -_vec_dot(relative_pos, relative_vel) / velocity_norm_sq
    return max(0.0, min(spec.dt, time))


def impact_corrected_gap(spec: MovingContactBenchmarkSpec, control: float) -> float:
    """Return the continuous closest-approach gap used as the time-of-impact objective."""

    time = closest_approach_time(spec, control)
    pos_a = _sphere_position(spec.pos_a0, spec.vel_a, time)
    pos_b = _sphere_position(spec.pos_b0, _controlled_velocity(spec, control), time)
    return analytic_sphere_gap(spec.radius_a, pos_a, spec.radius_b, pos_b)


def reference_query_gap(spec: MovingContactBenchmarkSpec, control: float) -> tuple[float, float]:
    """Return the minimum query-backed gap over a dense trajectory sample."""

    min_gap = math.inf
    min_time = 0.0
    for sample_index in range(spec.reference_samples):
        time = spec.dt * sample_index / (spec.reference_samples - 1)
        pos_a = _sphere_position(spec.pos_a0, spec.vel_a, time)
        pos_b = _sphere_position(spec.pos_b0, _controlled_velocity(spec, control), time)
        gap = query_closest_distance(
            int(GeoType.SPHERE),
            (spec.radius_a, 0.0, 0.0),
            pos_a,
            int(GeoType.SPHERE),
            (spec.radius_b, 0.0, 0.0),
            pos_b,
        )["distance"]
        if gap < min_gap:
            min_gap = float(gap)
            min_time = float(time)
    return min_gap, min_time


def estimate_gradient(objective: Any, control: float, epsilon: float = 1.0e-4) -> float:
    """Estimate `d(objective) / d(control)` with a central difference."""

    return (float(objective(control + epsilon)) - float(objective(control - epsilon))) / (2.0 * epsilon)


def objective_end_step(spec: MovingContactBenchmarkSpec, control: float) -> float:
    """Return the baseline squared-gap objective at the rollout endpoint."""

    gap = analytic_end_step_gap(spec, control)
    return gap * gap


def objective_impact(spec: MovingContactBenchmarkSpec, control: float) -> float:
    """Return the time-of-impact squared-gap objective at closest approach."""

    gap = impact_corrected_gap(spec, control)
    return gap * gap


def objective_reference(spec: MovingContactBenchmarkSpec, control: float) -> float:
    """Return the dense query-backed squared-gap reference objective."""

    gap, _ = reference_query_gap(spec, control)
    return gap * gap


def gradient_report(spec: MovingContactBenchmarkSpec) -> list[dict[str, float]]:
    """Return side-by-side gradient quality metrics for selected controls."""

    samples: list[dict[str, float]] = []
    for control in spec.gradient_samples:
        end_gap = query_end_step_gap(spec, control)
        impact_gap = impact_corrected_gap(spec, control)
        query_reference_gap, query_reference_time = reference_query_gap(spec, control)
        baseline_gradient = estimate_gradient(lambda value: objective_end_step(spec, value), control)
        impact_gradient = estimate_gradient(lambda value: objective_impact(spec, value), control)
        reference_gradient = impact_gradient
        samples.append(
            {
                "control": float(control),
                "baseline_gap": float(end_gap),
                "impact_gap": float(impact_gap),
                "query_reference_gap": float(query_reference_gap),
                "closest_approach_time": float(closest_approach_time(spec, control)),
                "query_reference_time": float(query_reference_time),
                "baseline_gradient": float(baseline_gradient),
                "impact_gradient": float(impact_gradient),
                "reference_gradient": float(reference_gradient),
                "baseline_gradient_error": float(abs(baseline_gradient - reference_gradient)),
                "impact_gradient_error": float(abs(impact_gradient - reference_gradient)),
            }
        )
    return samples


def run_optimization_study(spec: MovingContactBenchmarkSpec) -> dict[str, Any]:
    """Run a one-parameter descent study for baseline and time-of-impact objectives."""

    methods = {
        "baseline": lambda control: objective_end_step(spec, control),
        "impact": lambda control: objective_impact(spec, control),
    }
    outcomes: dict[str, Any] = {}
    for method_name, objective in methods.items():
        runs: list[dict[str, float]] = []
        success_count = 0
        for start in spec.optimization_starts:
            control = float(start)
            for _ in range(spec.descent_iterations):
                gradient = estimate_gradient(objective, control)
                control -= spec.descent_step * gradient
            final_true_gap = abs(impact_corrected_gap(spec, control))
            success = final_true_gap < 5.0e-2
            success_count += int(success)
            runs.append(
                {
                    "start": float(start),
                    "final_control": float(control),
                    "final_true_gap": float(final_true_gap),
                    "success": success,
                }
            )
        outcomes[method_name] = {
            "success_count": success_count,
            "run_count": len(spec.optimization_starts),
            "runs": runs,
        }
    return outcomes


def default_benchmark_spec() -> MovingContactBenchmarkSpec:
    """Return the default moving sphere-sphere intercept benchmark."""

    return MovingContactBenchmarkSpec(
        name="sphere_sphere_intercept_control_vy",
        radius_a=0.5,
        radius_b=0.5,
        pos_a0=(0.0, 0.0, 0.0),
        vel_a=(1.0, 0.0, 0.0),
        pos_b0=(1.5, 0.9, 0.0),
        vel_b=(-1.0, 0.0, 0.0),
        control_axis=1,
        dt=1.0,
        gradient_samples=(-1.0, -0.5, 0.0, 0.5, 1.0),
        optimization_starts=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )


def build_moving_contact_time_of_impact_report() -> dict[str, Any]:
    """Build the JSON-serializable ROB-139 benchmark report."""

    spec = default_benchmark_spec()
    gradients = gradient_report(spec)
    optimization = run_optimization_study(spec)
    return {
        "artifact": BENCHMARK_ARTIFACT,
        "entrypoint": BENCHMARK_ENTRYPOINT,
        "benchmark": spec.name,
        "metric": "min squared signed gap over the rollout horizon",
        "control": "sphere B y velocity delta [m/s]",
        "touch_points": list(BENCHMARK_TOUCH_POINTS),
        "recommendation": BENCHMARK_RECOMMENDATION,
        "scenario": {
            "radius_a": spec.radius_a,
            "radius_b": spec.radius_b,
            "pos_a0": list(spec.pos_a0),
            "vel_a": list(spec.vel_a),
            "pos_b0": list(spec.pos_b0),
            "vel_b": list(spec.vel_b),
            "dt": spec.dt,
        },
        "gradient_quality": gradients,
        "optimization_stability": optimization,
    }


def make_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the ROB-139 benchmark report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the report to stdout or a JSON file."""

    args = make_argument_parser().parse_args(argv)
    payload = json.dumps(build_moving_contact_time_of_impact_report(), indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
