# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton._src.tools.tiled_sensing_benchmark import (
    BenchmarkConfig,
    build_tiled_sensing_index,
    run_tiled_sensing_benchmark,
    write_tiled_sensing_artifacts,
)


class TiledSensingBenchmarkTest(unittest.TestCase):
    def test_build_index_prefers_latest_run(self):
        older = {
            "run_id": "20260401T120000Z",
            "generated_at": "2026-04-01T12:00:00+00:00",
            "benchmark": {"device": "cpu"},
            "scenario": {"world_count": 4},
            "summary": {"mean_render_time_ms": 8.0, "steps_per_second": 125.0},
            "runtime": {"alignment": "internal_sensor_runtime_boundary"},
        }
        newer = {
            "run_id": "20260401T130000Z",
            "generated_at": "2026-04-01T13:00:00+00:00",
            "benchmark": {"device": "cuda:0"},
            "scenario": {"world_count": 8},
            "summary": {"mean_render_time_ms": 4.0, "steps_per_second": 250.0},
            "runtime": {"alignment": "internal_sensor_runtime_boundary"},
        }

        index = build_tiled_sensing_index([older, newer])

        self.assertEqual(index["run_count"], 2)
        self.assertEqual(index["latest_run"]["run_id"], "20260401T130000Z")
        self.assertEqual(index["latest_runtime"]["alignment"], "internal_sensor_runtime_boundary")

    def test_write_artifacts_persists_previews_and_index(self):
        run_payload = {
            "schema_version": 1,
            "run_id": "20260401T190000Z",
            "generated_at": "2026-04-01T19:00:00+00:00",
            "benchmark": {"device": "cpu"},
            "scenario": {"world_count": 2},
            "runtime": {"alignment": "internal_sensor_runtime_boundary"},
            "summary": {"mean_render_time_ms": 3.0, "steps_per_second": 333.0},
            "observability": {"world_packets": []},
            "artifacts": {},
        }
        color_preview = np.zeros((2, 1, 4, 4), dtype=np.uint32)
        depth_preview = np.ones((2, 1, 4, 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_path, index_path = write_tiled_sensing_artifacts(
                Path(tmpdir),
                run_payload,
                color_preview,
                depth_preview,
            )

            self.assertTrue(run_path.exists())
            self.assertTrue(index_path.exists())
            payload = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifacts"]["color_preview"], "previews/20260401T190000Z_color.npy")
            self.assertEqual(payload["artifacts"]["depth_preview"], "previews/20260401T190000Z_depth.npy")
            np.testing.assert_array_equal(np.load(Path(tmpdir) / payload["artifacts"]["color_preview"]), color_preview)

    def test_run_benchmark_returns_structured_observability(self):
        config = BenchmarkConfig(world_count=2, steps=2, warmup_steps=0, resolution_width=24, resolution_height=24)

        payload, color_preview, depth_preview = run_tiled_sensing_benchmark(config, device="cpu")

        self.assertEqual(payload["benchmark"]["name"], "visual_first_tiled_sensing")
        self.assertEqual(payload["scenario"]["world_count"], 2)
        self.assertEqual(len(payload["observability"]["world_packets"]), 4)
        self.assertEqual(color_preview.shape, (2, 1, 24, 24))
        self.assertEqual(depth_preview.shape, (2, 1, 24, 24))
        self.assertGreater(payload["summary"]["mean_render_time_ms"], 0.0)
        self.assertTrue(
            any(packet["depth_hit_pixel_count"] > 0 for packet in payload["observability"]["world_packets"])
        )


if __name__ == "__main__":
    unittest.main()
