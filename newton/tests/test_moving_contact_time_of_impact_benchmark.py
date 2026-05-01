# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from newton._src.geometry.moving_contact_time_of_impact_benchmark import (
    BENCHMARK_ARTIFACT,
    build_moving_contact_time_of_impact_report,
    closest_approach_time,
    default_benchmark_spec,
    gradient_report,
    impact_corrected_gap,
    main,
    query_end_step_gap,
    reference_query_gap,
    run_optimization_study,
)


class MovingContactTimeOfImpactBenchmarkTest(unittest.TestCase):
    def test_report_includes_entrypoint_and_touch_points(self):
        report = build_moving_contact_time_of_impact_report()

        self.assertEqual(report["artifact"], BENCHMARK_ARTIFACT)
        self.assertEqual(report["benchmark"], "sphere_sphere_intercept_control_vy")
        self.assertEqual(
            report["touch_points"],
            [
                "newton/_src/geometry/differentiable_query_spike.py",
                "newton/_src/geometry/moving_contact_time_of_impact_benchmark.py",
            ],
        )

    def test_query_end_step_gap_matches_dense_reference_endpoint_behavior(self):
        spec = default_benchmark_spec()

        query_gap = query_end_step_gap(spec, 0.0)
        reference_gap, _ = reference_query_gap(spec, 0.0)

        self.assertGreater(query_gap, 0.0)
        self.assertLess(reference_gap, query_gap)

    def test_time_of_impact_gradient_tracks_reference_better_than_baseline(self):
        spec = default_benchmark_spec()

        samples = gradient_report(spec)
        baseline_errors = [sample["baseline_gradient_error"] for sample in samples]
        impact_errors = [sample["impact_gradient_error"] for sample in samples]

        self.assertGreater(sum(baseline_errors), sum(impact_errors))
        self.assertEqual(sum(impact_errors), 0.0)
        self.assertGreater(baseline_errors[0], 0.0)

    def test_time_of_impact_optimization_converges_more_reliably(self):
        spec = default_benchmark_spec()

        results = run_optimization_study(spec)

        self.assertEqual(results["impact"]["success_count"], results["impact"]["run_count"])
        self.assertEqual(results["baseline"]["success_count"], 0)

    def test_closest_approach_occurs_mid_step(self):
        spec = default_benchmark_spec()

        time = closest_approach_time(spec, 0.0)
        self.assertGreater(time, 0.0)
        self.assertLess(time, spec.dt)
        self.assertLess(abs(impact_corrected_gap(spec, 0.0)), 0.15)

    def test_main_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "moving_contact_time_of_impact.json"

            exit_code = main(["--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact"], BENCHMARK_ARTIFACT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
