# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
import datetime
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from newton._src.tools.benchmark_dashboard import (
    BenchmarkDashboardRequestHandler,
    _render_index_html,
    assess_benchmark_freshness,
    build_dashboard_summary,
    default_benchmark_index_path,
    default_benchmark_max_result_age_hours,
    make_argument_parser,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class BenchmarkDashboardSummaryTest(unittest.TestCase):
    def test_build_dashboard_summary_prefers_solver_matrix_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "asv" / "results"
            html_dir = root / "asv" / "html"
            benchmark_index = root / "benchmarks" / "results" / "index.json"
            run_path = benchmark_index.parent / "runs" / "20260330T120000Z.json"

            html_dir.mkdir(parents=True)
            _write_json(
                run_path,
                {
                    "schema_version": 1,
                    "run_id": "20260330T120000Z",
                    "generated_at": "2026-03-30T12:00:00+00:00",
                    "started_at": "2026-03-30T11:59:00+00:00",
                    "device": "cuda:0",
                    "host": "eheiden-claw2",
                    "results": [
                        {
                            "scenario": "basic_shapes",
                            "capability": "rigid_contacts",
                            "solver_id": "xpbd",
                            "solver_name": "SolverXPBD",
                            "total_time_ms": 10.0,
                            "steps_per_second": 400.0,
                        },
                        {
                            "scenario": "basic_shapes",
                            "capability": "rigid_contacts",
                            "solver_id": "vbd",
                            "solver_name": "SolverVBD",
                            "total_time_ms": 18.0,
                            "steps_per_second": 220.0,
                        },
                    ],
                    "notes": ["Parity gap remains for MuJoCo/MABD coverage."],
                },
            )
            _write_json(
                benchmark_index,
                {
                    "schema_version": 1,
                    "generated_at": "2026-03-30T12:00:00+00:00",
                    "run_count": 1,
                    "runs": [
                        {
                            "run_id": "20260330T120000Z",
                            "generated_at": "2026-03-30T12:00:00+00:00",
                            "device": "cuda:0",
                            "steps": 4,
                            "warmup_steps": 1,
                            "result_count": 2,
                        }
                    ],
                    "latest_run": {
                        "run_id": "20260330T120000Z",
                        "generated_at": "2026-03-30T12:00:00+00:00",
                        "notes": ["Parity gap remains for MuJoCo/MABD coverage."],
                    },
                },
            )

            summary = build_dashboard_summary(results_dir, html_dir, benchmark_index)

            self.assertEqual(summary["status"], "ready")
            self.assertTrue(summary["benchmark_index_available"])
            self.assertEqual(summary["stats"]["machine_count"], 1)
            self.assertEqual(summary["stats"]["latest_run_count"], 1)
            self.assertEqual(summary["stats"]["benchmark_case_count"], 2)
            self.assertEqual(summary["stats"]["comparison_group_count"], 1)
            self.assertEqual(summary["benchmark_cases"][0]["machine"], "eheiden-claw2")
            self.assertEqual(summary["benchmark_cases"][0]["env_name"], "cuda:0")
            self.assertEqual(summary["benchmark_cases"][0]["unit"], "steps/s")
            self.assertEqual(summary["comparison_groups"][0]["series_param"], "solver")
            self.assertEqual(
                [item["label"] for item in summary["comparison_groups"][0]["series"]],
                ["SolverVBD", "SolverXPBD"],
            )
            self.assertEqual(summary["notes"], ["Parity gap remains for MuJoCo/MABD coverage."])

    def test_build_dashboard_summary_tracks_previous_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "asv" / "results"
            html_dir = root / "asv" / "html"
            machine_dir = results_dir / "linux-runner"

            _write_json(
                machine_dir / "machine.json",
                {
                    "version": 1,
                    "machine": "linux-runner",
                    "cpu": "AMD EPYC",
                    "gpu": "RTX 6000",
                },
            )
            _write_json(
                results_dir / "benchmarks.json",
                {
                    "bench.time_step": {"unit": "seconds"},
                },
            )
            _write_json(
                machine_dir / "12345678-env.json",
                {
                    "version": 2,
                    "commit_hash": "12345678abcdef00",
                    "date": 1000,
                    "env_name": "py312",
                    "params": {"machine": "linux-runner"},
                    "python": "3.12",
                    "requirements": {},
                    "env_vars": {},
                    "result_columns": ["result", "params", "version", "started_at", "duration"],
                    "results": {
                        "bench.time_step": [[2.0], [], "1", 1000, 1.2],
                    },
                },
            )
            _write_json(
                machine_dir / "abcdef01-env.json",
                {
                    "version": 2,
                    "commit_hash": "abcdef0199999999",
                    "date": 2000,
                    "env_name": "py312",
                    "params": {"machine": "linux-runner"},
                    "python": "3.12",
                    "requirements": {},
                    "env_vars": {},
                    "result_columns": ["result", "params", "version", "started_at", "duration"],
                    "results": {
                        "bench.time_step": [[1.5], [], "1", 2000, 1.0],
                    },
                },
            )
            html_dir.mkdir(parents=True)

            summary = build_dashboard_summary(results_dir, html_dir, root / "benchmarks" / "results" / "missing.json")

            self.assertTrue(summary["results_available"])
            self.assertTrue(summary["html_available"])
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["stats"]["machine_count"], 1)
            self.assertEqual(summary["stats"]["benchmark_case_count"], 1)
            self.assertEqual(summary["stats"]["comparison_group_count"], 1)
            self.assertEqual(summary["stats"]["latest_run_count"], 2)
            self.assertEqual(summary["stats"]["environment_count"], 1)
            self.assertEqual(summary["stats"]["improvement_count"], 1)
            self.assertEqual(summary["stats"]["new_case_count"], 0)
            self.assertEqual(len(summary["benchmark_cases"]), 1)
            self.assertEqual(len(summary["highlights"]["improvements"]), 1)
            self.assertEqual(len(summary["highlights"]["recent_runs"]), 2)
            self.assertEqual(summary["machines"][0]["run_count"], 2)
            self.assertEqual(summary["machines"][0]["benchmark_case_count"], 2)

            case = summary["benchmark_cases"][0]
            self.assertEqual(case["machine"], "linux-runner")
            self.assertEqual(case["env_name"], "py312")
            self.assertEqual(case["commit_short"], "abcdef01")
            self.assertAlmostEqual(case["value"], 1.5)
            self.assertAlmostEqual(case["previous_value"], 2.0)
            self.assertAlmostEqual(case["delta_pct"], -25.0)
            self.assertEqual(case["status"], "improvement")
            self.assertEqual(case["history_length"], 2)
            self.assertEqual(len(case["history"]), 2)
            self.assertAlmostEqual(case["history_min"], 1.5)
            self.assertAlmostEqual(case["history_max"], 2.0)

            comparison_group = summary["comparison_groups"][0]
            self.assertEqual(comparison_group["title"], "bench.time_step")
            self.assertEqual(comparison_group["series_param"], None)
            self.assertEqual(comparison_group["series_count"], 1)
            self.assertEqual(comparison_group["status_counts"]["improvement"], 1)
            self.assertEqual(comparison_group["series"][0]["label"], "bench.time_step")
            self.assertEqual(comparison_group["series"][0]["status"], "improvement")

    def test_build_dashboard_summary_marks_first_result_as_new_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "asv" / "results"
            html_dir = root / "asv" / "html"
            machine_dir = results_dir / "linux-runner"

            _write_json(
                machine_dir / "machine.json",
                {
                    "version": 1,
                    "machine": "linux-runner",
                },
            )
            _write_json(
                machine_dir / "12345678-env.json",
                {
                    "version": 2,
                    "commit_hash": "12345678abcdef00",
                    "date": 1000,
                    "env_name": "py312",
                    "params": {"machine": "linux-runner"},
                    "python": "3.12",
                    "requirements": {},
                    "env_vars": {},
                    "result_columns": ["result", "params", "version"],
                    "results": {
                        "bench.time_step": [[2.0], [], "1"],
                    },
                },
            )
            html_dir.mkdir(parents=True)

            summary = build_dashboard_summary(results_dir, html_dir, root / "benchmarks" / "results" / "missing.json")

            self.assertEqual(summary["stats"]["new_case_count"], 1)
            self.assertEqual(summary["benchmark_cases"][0]["status"], "new")
            self.assertIsNone(summary["benchmark_cases"][0]["delta_pct"])
            self.assertEqual(summary["filters"]["environments"][0]["value"], "py312")

    def test_build_dashboard_summary_groups_solver_comparisons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "asv" / "results"
            html_dir = root / "asv" / "html"
            machine_dir = results_dir / "linux-runner"

            _write_json(
                machine_dir / "machine.json",
                {
                    "version": 1,
                    "machine": "linux-runner",
                },
            )
            _write_json(
                results_dir / "benchmarks.json",
                {
                    "bench.solve": {
                        "unit": "seconds",
                        "param_names": ["solver", "scenario"],
                    },
                },
            )
            _write_json(
                machine_dir / "abcdef01-env.json",
                {
                    "version": 2,
                    "commit_hash": "abcdef0199999999",
                    "date": 2000,
                    "env_name": "py312",
                    "result_columns": ["result", "params", "version"],
                    "results": {
                        "bench.solve": [[1.2, 0.8, 1.8, 1.4], [["xpbd", "sap"], ["cloth", "rigid"]], "1"],
                    },
                },
            )
            html_dir.mkdir(parents=True)

            summary = build_dashboard_summary(results_dir, html_dir, root / "benchmarks" / "results" / "missing.json")

            self.assertEqual(summary["stats"]["comparison_group_count"], 2)
            comparison_titles = [group["title"] for group in summary["comparison_groups"]]
            self.assertIn("bench.solve [scenario=cloth]", comparison_titles)
            self.assertIn("bench.solve [scenario=rigid]", comparison_titles)

            cloth_group = next(
                group for group in summary["comparison_groups"] if group["title"] == "bench.solve [scenario=cloth]"
            )
            self.assertEqual(cloth_group["series_param"], "solver")
            self.assertEqual([item["label"] for item in cloth_group["series"]], ["sap", "xpbd"])
            self.assertEqual(cloth_group["scenario_params"], [{"name": "scenario", "value": "cloth"}])
            self.assertEqual(cloth_group["series"][0]["status"], "new")

    def test_assess_benchmark_freshness_uses_solver_run_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "asv" / "results"
            html_dir = root / "asv" / "html"
            benchmark_index = root / "benchmarks" / "results" / "index.json"
            run_path = benchmark_index.parent / "runs" / "20260330T120000Z.json"
            html_dir.mkdir(parents=True)
            _write_json(
                run_path,
                {
                    "run_id": "20260330T120000Z",
                    "generated_at": "2026-03-30T12:00:00+00:00",
                    "started_at": "2026-03-30T11:55:00+00:00",
                    "results": [],
                },
            )
            _write_json(
                benchmark_index,
                {
                    "generated_at": "2026-03-30T12:00:00+00:00",
                    "latest_run": {
                        "run_id": "20260330T120000Z",
                        "generated_at": "2026-03-30T12:00:00+00:00",
                    },
                },
            )

            freshness = assess_benchmark_freshness(
                results_dir,
                benchmark_index,
                24.0,
                now=datetime.datetime(2026, 4, 1, 12, 0, tzinfo=datetime.timezone.utc),
            )

            self.assertTrue(freshness["is_stale"])
            self.assertEqual(freshness["source"], "artifact")
            self.assertEqual(freshness["artifact_timestamp"], "2026-03-30T12:00:00+00:00")
            self.assertGreater(freshness["age_hours"], 24.0)


class BenchmarkDashboardHandlerTest(unittest.TestCase):
    def test_default_benchmark_index_path_honors_environment_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "solver-benchmarks" / "index.json"
            with mock.patch.dict(os.environ, {"NEWTON_BENCHMARK_INDEX_PATH": str(artifact_path)}):
                self.assertEqual(default_benchmark_index_path(), artifact_path.resolve())

    def test_argument_parser_uses_dashboard_service_defaults(self):
        args = make_argument_parser().parse_args([])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 7000)
        self.assertEqual(args.results_dir, "asv/results")
        self.assertEqual(args.html_dir, "asv/html")
        self.assertEqual(Path(args.benchmark_index), default_benchmark_index_path())

    def test_argument_parser_uses_env_default_freshness_threshold(self):
        with mock.patch.dict(os.environ, {"NEWTON_BENCHMARK_MAX_RESULT_AGE_HOURS": "10"}):
            self.assertEqual(default_benchmark_max_result_age_hours(), 10.0)
            args = make_argument_parser().parse_args([])

        self.assertEqual(args.max_result_age_hours, 10.0)

    def test_request_handler_serves_summary_and_asv_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "asv" / "results"
            html_dir = root / "asv" / "html"
            machine_dir = results_dir / "linux-runner"

            _write_json(
                machine_dir / "machine.json",
                {
                    "version": 1,
                    "machine": "linux-runner",
                },
            )
            _write_json(
                machine_dir / "12345678-env.json",
                {
                    "version": 2,
                    "commit_hash": "12345678abcdef00",
                    "date": 1000,
                    "env_name": "py312",
                    "params": {"machine": "linux-runner"},
                    "python": "3.12",
                    "requirements": {},
                    "env_vars": {},
                    "result_columns": ["result", "params", "version"],
                    "results": {
                        "bench.time_step": [[2.0], [], "1"],
                    },
                },
            )
            (html_dir / "index.html").parent.mkdir(parents=True, exist_ok=True)
            (html_dir / "index.html").write_text("<html>asv</html>", encoding="utf-8")

            handler = partial(
                BenchmarkDashboardRequestHandler,
                results_dir=results_dir,
                html_dir=html_dir,
                benchmark_index_path=root / "benchmarks" / "results" / "index.json",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(f"{base_url}/api/summary") as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["stats"]["benchmark_case_count"], 1)
                self.assertEqual(payload["status"], "ready")
                self.assertEqual(payload["benchmark_cases"][0]["status"], "new")

                with urllib.request.urlopen(f"{base_url}/asv/") as response:
                    html = response.read().decode("utf-8")
                self.assertIn("asv", html)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_request_handler_serves_percent_encoded_asv_graph_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "asv" / "results"
            html_dir = root / "asv" / "html"
            graph_path = (
                html_dir
                / "graphs"
                / "branch-main"
                / "cpu-Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz (128 CPUs)"
                / "machine-paperclip-newton"
                / "os-Linux"
                / "python-_home_horde_newton_.venv_bin_python3"
                / "ram-1.0TiB"
                / "summary.json"
            )
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text('{"series": []}', encoding="utf-8")

            handler = partial(
                BenchmarkDashboardRequestHandler,
                results_dir=results_dir,
                html_dir=html_dir,
                benchmark_index_path=root / "benchmarks" / "results" / "index.json",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                encoded_path = urllib.parse.quote(str(graph_path.relative_to(html_dir)).replace("\\\\", "/"))
                with urllib.request.urlopen(f"{base_url}/asv/{encoded_path}") as response:
                    payload = response.read().decode("utf-8")
                self.assertEqual(payload, '{"series": []}')
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


class BenchmarkDashboardHtmlTest(unittest.TestCase):
    def test_rendered_html_contains_overhauled_sections(self):
        html = _render_index_html()

        self.assertIn("Comparative plots", html)
        self.assertIn("Latest benchmark cases", html)
        self.assertIn('id="comparison-groups"', html)
        self.assertIn("Recent runs", html)

    def test_rendered_html_keeps_quote_escape_mapping_valid_for_browser_js(self):
        html = _render_index_html()

        self.assertIn('.replaceAll(\'"\', "&quot;")', html)
        self.assertIn("payload.comparison_groups", html)
