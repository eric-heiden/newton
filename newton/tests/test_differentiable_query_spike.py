# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from newton._src.geometry.differentiable_query_spike import (
    SPIKE_INTERFACE_READOUT,
    SPIKE_TARGET_MODULES,
    build_differentiable_query_spike_report,
    default_spike_specs,
    estimate_translation_gradient,
    main,
    query_closest_distance,
    sample_translation_sweep,
)


class DifferentiableQuerySpikeTest(unittest.TestCase):
    def test_report_lists_target_modules_and_shape_pairs(self):
        report = build_differentiable_query_spike_report()

        self.assertEqual(report["artifact"], "rob-108-differentiable-query-spike")
        self.assertEqual(report["target_modules"], list(SPIKE_TARGET_MODULES))
        self.assertEqual(
            report["first_shape_pairs"],
            [
                "sphere_box_translation_x",
                "capsule_box_translation_x",
                "box_box_rotated_translation_x",
            ],
        )
        self.assertEqual(len(report["sweeps"]), 3)
        self.assertEqual(report["interface_readout"], SPIKE_INTERFACE_READOUT)

    def test_sphere_box_query_returns_consistent_witness_points(self):
        spec = default_spike_specs()[0]

        result = query_closest_distance(
            spec.geom_a_type,
            spec.geom_a_size,
            spec.geom_a_pos,
            spec.geom_b_type,
            spec.geom_b_size,
            spec.geom_b_base_pos,
        )
        gradient = estimate_translation_gradient(spec, offset=0.0)

        self.assertEqual(result["collision"], 0)
        self.assertGreater(result["distance"], 0.0)
        self.assertAlmostEqual(gradient, result["normal"][spec.translation_axis], places=3)
        self.assertAlmostEqual(result["normal"][0], 1.0, places=4)

        witness_delta = [result["point_b"][i] - result["point_a"][i] for i in range(3)]
        projected_distance = sum(witness_delta[i] * result["normal"][i] for i in range(3))
        midpoint = [(result["point_a"][i] + result["point_b"][i]) * 0.5 for i in range(3)]

        self.assertAlmostEqual(projected_distance, result["distance"], places=4)
        for component, expected in zip(result["point"], midpoint, strict=True):
            self.assertAlmostEqual(component, expected, places=6)

    def test_capsule_box_sweep_distance_increases_with_positive_offset(self):
        spec = default_spike_specs()[1]

        samples = sample_translation_sweep(spec)

        self.assertEqual([sample["offset"] for sample in samples], list(spec.offsets))
        self.assertTrue(all(sample["collision"] == 0 for sample in samples))
        self.assertGreater(samples[1]["distance"], samples[0]["distance"])
        self.assertGreater(samples[2]["distance"], samples[1]["distance"])

    def test_rotated_box_box_sweep_stays_separated_and_gradient_matches_normal(self):
        spec = default_spike_specs()[2]

        samples = sample_translation_sweep(spec)

        self.assertEqual([sample["offset"] for sample in samples], list(spec.offsets))
        self.assertTrue(all(sample["collision"] == 0 for sample in samples))
        self.assertTrue(all(sample["distance"] > 0.0 for sample in samples))
        self.assertTrue(all(sample["point_b"][0] > sample["point_a"][0] for sample in samples))
        self.assertGreater(abs(spec.geom_b_orientation[2]), 0.0)
        self.assertAlmostEqual(spec.geom_b_orientation[2], math.sin(math.pi / 8.0), places=6)

        for sample in samples:
            self.assertAlmostEqual(sample["finite_difference_gradient"], sample["normal"][0], places=3)

    def test_main_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "spike.json"

            exit_code = main(["--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact"], "rob-108-differentiable-query-spike")


if __name__ == "__main__":
    unittest.main(verbosity=2)
