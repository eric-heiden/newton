# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from newton._src.tools.solver_benchmark_matrix import (
    SCENARIOS,
    build_solver_benchmark_index,
    default_solver_benchmark_index_path,
    default_solver_benchmark_results_dir,
    make_argument_parser,
    write_solver_benchmark_artifacts,
)


def _make_run(run_id: str, generated_at: str, scenario: str, capability: str, solver_name: str, rate: float) -> dict:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "generated_at": generated_at,
        "started_at": generated_at,
        "device": "cuda:0",
        "host": "runner",
        "steps": 4,
        "warmup_steps": 1,
        "coverage_summary": {"benchmarked": 2, "harness_gap": 1, "unsupported": 1},
        "scenarios": [
            {
                "name": scenario,
                "capability": capability,
                "description": "Scenario description.",
                "solvers": ["xpbd", "vbd"],
                "example_module": "newton.examples.fake",
                "coverage": [
                    {
                        "solver_id": "xpbd",
                        "solver_name": "SolverXPBD",
                        "status": "benchmarked",
                        "reason": "Example-backed benchmark implemented.",
                    },
                    {
                        "solver_id": "vbd",
                        "solver_name": "SolverVBD",
                        "status": "benchmarked",
                        "reason": "Example-backed benchmark implemented.",
                    },
                    {
                        "solver_id": "mujoco",
                        "solver_name": "SolverMuJoCo",
                        "status": "harness_gap",
                        "reason": "Not wired into this scenario yet.",
                    },
                    {
                        "solver_id": "style3d",
                        "solver_name": "SolverStyle3D",
                        "status": "unsupported",
                        "reason": "Unsupported for this scenario.",
                    },
                ],
            }
        ],
        "results": [
            {
                "scenario": scenario,
                "capability": capability,
                "description": "Scenario description.",
                "example_module": "newton.examples.fake",
                "solver_id": solver_name.lower(),
                "solver_name": solver_name,
                "step_times_ms": [5.0, 4.0, 3.0, 4.0],
                "model": {
                    "body_count": 2,
                    "joint_count": 1,
                    "joint_dof_count": 1,
                    "shape_count": 2,
                    "particle_count": 0,
                },
                "total_time_ms": 16.0,
                "mean_step_time_ms": 4.0,
                "median_step_time_ms": 4.0,
                "min_step_time_ms": 3.0,
                "max_step_time_ms": 5.0,
                "steps_per_second": rate,
            }
        ],
        "notes": [],
    }


class SolverBenchmarkMatrixTest(unittest.TestCase):
    def test_parser_defaults_to_benchmark_results_dir(self):
        args = make_argument_parser().parse_args([])

        self.assertEqual(args.results_dir, default_solver_benchmark_results_dir())
        self.assertEqual(default_solver_benchmark_index_path(), default_solver_benchmark_results_dir() / "index.json")

    def test_registry_exposes_solver_capability_matrix_scenarios(self):
        self.assertIn("gravity_capsule", SCENARIOS)
        self.assertIn("pendulum", SCENARIOS)
        self.assertIn("basic_shapes", SCENARIOS)
        self.assertIn("basic_joints", SCENARIOS)
        self.assertIn("basic_heightfield", SCENARIOS)
        self.assertIn("cloth_hanging", SCENARIOS)
        self.assertIn("fourbar_loop", SCENARIOS)
        self.assertIn("dr_legs_loop", SCENARIOS)
        self.assertIn("humanoid_loop_heavy", SCENARIOS)
        self.assertIn("allegro_hand", SCENARIOS)
        self.assertIn("heterogeneous_worlds", SCENARIOS)
        self.assertIn("kamino", SCENARIOS["fourbar_loop"].solver_ids)
        self.assertIn("style3d", SCENARIOS["cloth_hanging"].solver_ids)

    def test_build_solver_benchmark_index_sorts_runs_and_ranks_latest_matrix(self):
        newer = _make_run(
            "20260330T120000Z", "2026-03-30T12:00:00+00:00", "basic_shapes", "rigid_contacts", "SolverXPBD", 240.0
        )
        newer["results"].append(
            {
                **newer["results"][0],
                "solver_id": "solverwvbd",
                "solver_name": "SolverVBD",
                "steps_per_second": 180.0,
            }
        )
        older = _make_run(
            "20260329T120000Z", "2026-03-29T12:00:00+00:00", "basic_shapes", "rigid_contacts", "SolverXPBD", 120.0
        )

        index = build_solver_benchmark_index([older, newer])

        self.assertEqual(index["run_count"], 2)
        self.assertEqual(index["latest_run"]["run_id"], "20260330T120000Z")
        self.assertEqual(index["latest_matrix"][0]["solver_name"], "SolverXPBD")
        self.assertEqual(index["latest_matrix"][0]["rank"], 1)
        self.assertEqual(index["latest_matrix"][1]["rank"], 2)
        self.assertEqual(index["scenarios"][0]["name"], "basic_shapes")
        self.assertEqual(len(index["scenarios"][0]["throughput_plot"]), 2)
        self.assertEqual(index["scenarios"][0]["step_time_plot"][0]["solver_name"], "SolverVBD")
        self.assertEqual(index["latest_coverage"][0]["status"], "benchmarked")
        self.assertEqual(index["latest_coverage"][-1]["status"], "unsupported")
        self.assertEqual(index["runs"][0]["coverage_summary"]["benchmarked"], 2)
        self.assertEqual(index["schema_version"], 2)

    def test_write_solver_benchmark_artifacts_updates_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            first_run = _make_run(
                "20260330T120000Z",
                "2026-03-30T12:00:00+00:00",
                "basic_joints",
                "articulations",
                "SolverXPBD",
                100.0,
            )
            second_run = _make_run(
                "20260330T130000Z",
                "2026-03-30T13:00:00+00:00",
                "cloth_hanging",
                "cloth",
                "SolverVBD",
                80.0,
            )

            run_path, index_path = write_solver_benchmark_artifacts(results_dir, first_run)
            self.assertTrue(run_path.exists())
            self.assertTrue(index_path.exists())

            write_solver_benchmark_artifacts(results_dir, second_run)
            payload = json.loads(index_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["run_count"], 2)
            self.assertEqual(payload["latest_run"]["run_id"], "20260330T130000Z")
            self.assertEqual(len(list((results_dir / "runs").glob("*.json"))), 2)
