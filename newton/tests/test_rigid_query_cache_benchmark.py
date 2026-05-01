# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from newton._src.geometry.rigid_query_cache_benchmark import (
    BENCHMARK_ARTIFACT,
    default_benchmark_spec,
    main,
    run_benchmark,
)


class TestRigidQueryCacheBenchmark(unittest.TestCase):
    def test_run_benchmark_reports_cached_query_sweep(self):
        payload = run_benchmark()

        self.assertEqual(payload["artifact"], BENCHMARK_ARTIFACT)
        self.assertEqual(payload["benchmark"]["step_count"], default_benchmark_spec().step_count)
        self.assertEqual(len(payload["samples"]), default_benchmark_spec().step_count)

        summary = payload["summary"]
        self.assertIsNotNone(summary["first_active_step"])
        self.assertGreater(summary["active_step_count"], 0)
        self.assertEqual(summary["inactive_step_count"], 0)
        self.assertEqual(summary["first_active_step"], 0)
        self.assertEqual(summary["last_active_step"], default_benchmark_spec().step_count - 1)
        self.assertLess(summary["first_active_step"], default_benchmark_spec().step_count)

        distances = [sample["distance"] for sample in payload["samples"]]
        self.assertLess(distances[0], 0.0)
        self.assertLess(distances[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(distances, distances[1:])))

        shape_pairs = {(sample["shape0"], sample["shape1"]) for sample in payload["samples"]}
        self.assertEqual(len(shape_pairs), 1)
        self.assertTrue(all(sample["rigid_query_count"] == 1 for sample in payload["samples"]))
        self.assertTrue(all(sample["rigid_contact_count"] == 1 for sample in payload["samples"]))
        self.assertTrue(all(sample["active"] == 1 for sample in payload["samples"]))
        self.assertTrue(all(abs(sample["normal"][0]) > 0.99999 for sample in payload["samples"]))

    def test_main_writes_json_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rigid-query-cache-benchmark.json"
            exit_code = main(["--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact"], BENCHMARK_ARTIFACT)
            self.assertEqual(len(payload["samples"]), default_benchmark_spec().step_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
