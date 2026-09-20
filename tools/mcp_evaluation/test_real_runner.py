# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check independent training evidence and equal clean-slate task inputs."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .real_robot import initial_config
from .run_real_agents import command, integrity, prepare, training_quality, verify


class TestRealRunner(unittest.TestCase):
    """Do not accept a different candidate, partial rollout or held-out log as training."""

    def test_partial_log_retains_failed_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "live_rollouts.jsonl").write_text('{"scenario":')
            self.assertFalse(training_quality(workspace, initial_config())["success"])

    def test_replicate_reaches_application(self):
        args = command(Path("trial"), Path("training.npz"), Path("metrics.json"), variant=5)
        self.assertEqual(args[args.index("--variant") + 1], "5")

    def test_verifier_timeout_is_a_failed_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("test", 120, output=b"partial output")):
                result = verify(workspace, Path("training.npz"), workspace / "verification/metrics.json", 0)
            self.assertFalse(result["success"])
            self.assertTrue(result["verification_timed_out"])

    def test_verifier_launch_error_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("subprocess.run", side_effect=OSError("could not launch")):
                result = verify(workspace, Path("training.npz"), workspace / "verification/metrics.json", 0)
            self.assertFalse(result["success"])

    def test_verifier_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            output = workspace / "metrics.json"
            output.write_text("[]")
            with patch("subprocess.run", return_value=subprocess.CompletedProcess("test", 0, "", "")):
                result = verify(workspace, Path("training.npz"), output, 0)
            self.assertFalse(result["success"])

    def test_exact_complete_training_required(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = initial_config()
            result = {
                "scenario": "panda_real",
                "config": config,
                "episodes": [2, 3, 4],
                "frames": 1800,
                "sample_count": 1800,
                "success": True,
            }
            (workspace / "verification").mkdir()
            (workspace / "verification/rollouts.jsonl").write_text(json.dumps(result) + "\n")
            self.assertFalse(training_quality(workspace, config)["success"])
            for changed in (
                {"frames": 1799},
                {"sample_count": 1799},
                {"episodes": [21, 22, 23]},
                {"config": config | {"mass": [2.0] * 7}},
            ):
                (workspace / "live_rollouts.jsonl").write_text(json.dumps(result | changed) + "\n")
                self.assertFalse(training_quality(workspace, config)["success"])
            (workspace / "live_rollouts.jsonl").write_text(json.dumps(result) + "\n")
            self.assertTrue(training_quality(workspace, config)["success"])

    def test_identical_numeric_inputs_and_private_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            (public / "geometry").mkdir(parents=True)
            (public / "geometry/panda_geometry.xml").write_text("<mujoco/>")
            for name in ("training.npz", "training-regressor.npz", "training-regressor.manifest.json"):
                (public / name).write_text(name)
            heldout = root / "private/heldout.npz"
            heldout.parent.mkdir()
            for name in ("heldout.npz", "heldout-regressor.npz", "heldout-regressor.manifest.json"):
                (heldout.parent / name).write_text("private fixture")
            specs = []
            for condition in ("live", "restart", "ipython", "ipython_fixed"):
                workspace = root / condition
                prepared = prepare(workspace, condition, 0, public, heldout)
                specs.append(prepared["spec"])
                self.assertEqual(json.loads((workspace / "config.json").read_text()), initial_config())
                self.assertFalse((workspace / "config.py").exists())
                self.assertNotIn(str(heldout), (workspace / "task.json").read_text())
                self.assertNotIn("private fixture", (workspace / "TASK.md").read_text())
                self.assertEqual(prepared["spec"]["candidate_budget"], 60)
                with patch(
                    "tools.mcp_evaluation.run_real_agents.source_hashes",
                    return_value=prepared["spec"]["source_hashes"] | {"new.py": "unexpected"},
                ):
                    self.assertFalse(integrity(workspace, prepared)[2])
            for spec in specs[1:]:
                for key in (
                    "initial",
                    "bounds",
                    "thresholds",
                    "input_hashes",
                    "geometry_hashes",
                    "source_hashes",
                    "budget_seconds",
                ):
                    self.assertEqual(spec[key], specs[0][key])


if __name__ == "__main__":
    unittest.main()
