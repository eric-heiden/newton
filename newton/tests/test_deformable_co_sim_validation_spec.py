# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path


class DeformableCoSimValidationSpecTest(unittest.TestCase):
    def test_sample_artifact_covers_cloth_cable_and_observability_sections(self):
        sample_path = Path(__file__).resolve().parents[2] / "benchmarks" / "deformable_co_sim_validation.sample.json"

        payload = json.loads(sample_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["benchmark"]["name"], "deformable_co_sim_validation")
        self.assertIn("real2sim_cloth", payload)
        self.assertIn("imagined_rollout_cable", payload)
        self.assertIn("observability", payload)
        self.assertGreaterEqual(len(payload["observability"]["world_packets"]), 1)

    def test_sample_artifact_keeps_newton_policy_revalidation_gate(self):
        sample_path = Path(__file__).resolve().parents[2] / "benchmarks" / "deformable_co_sim_validation.sample.json"

        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        policy_revalidation = payload["imagined_rollout_cable"]["policy_revalidation"]

        self.assertGreater(policy_revalidation["learned_success_rate"], policy_revalidation["baseline_success_rate"])
        self.assertGreaterEqual(policy_revalidation["retained_gain_fraction"], 0.9)
        self.assertTrue(policy_revalidation["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
