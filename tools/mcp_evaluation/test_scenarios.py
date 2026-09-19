# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check replay provenance, reset reproducibility, and condition equivalence."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from .microbenchmark import _quality_differences
from .recording import Recording
from .rollout import make_session
from .run_agents import _stop_process, _trial_measurements
from .scenarios import HUG_DATA, MENAGERIE, Scenario


class TestHarnessAccounting(unittest.TestCase):
    """Check verification isolation and complete quality comparisons."""

    def test_nested_candidate_logs_exclude_verification(self):
        """Count nested trial outputs without including independent verification."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for name in ("candidate_01", "batches/candidate_02", "verification", "verification/nested"):
                output = workspace / name
                output.mkdir(parents=True)
                (output / "rollouts.jsonl").write_text('{"frames": 1500}\n')
                (output / "process_events.jsonl").write_text('{"event": "simulation_process_start"}\n')
            result = _trial_measurements(workspace)
            self.assertEqual(result["candidate_rollouts"], 2)
            self.assertEqual(result["simulation_process_starts_during_trial"], 2)
            self.assertEqual(
                {log["path"] for log in result["candidate_rollout_logs"]},
                {"candidate_01/rollouts.jsonl", "batches/candidate_02/rollouts.jsonl"},
            )
            self.assertTrue(all(log["records"] == 1 for log in result["simulation_process_logs"]))
            (workspace / "live_rollouts.jsonl").write_text('{"frames": 1500}\n')
            self.assertEqual(_trial_measurements(workspace)["candidate_rollouts"], 3)

    def test_equal_rmse_does_not_hide_other_quality_differences(self):
        """Reject equal-RMSE candidates with different tails, controls, or completion."""
        first = {
            "scenario": "panda",
            "variant": 0,
            "config": {"kp": 50},
            "frames": 1500,
            "sample_count": 1500,
            "expected_frames": 1500,
            "finite": True,
            "success": True,
            "simulation_time_s": 3.0,
            "rmse": 0.01,
            "p95": 0.03,
        }
        thresholds = {"rmse": 0.045, "p95": 0.09}
        self.assertEqual(_quality_differences(first, dict(first), thresholds), [])
        for key, value in (("p95", 0.1), ("config", {"kp": 60}), ("sample_count", 1490), ("finite", False)):
            with self.subTest(field=key):
                self.assertIn(key, _quality_differences(first, first | {key: value}, thresholds))

    @unittest.skipUnless(os.name == "posix", "Process-group cleanup is supported on POSIX")
    def test_stop_exited_parent_also_stops_background_writer(self):
        """Stop an agent's background rollout before it can overwrite verification."""
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "late-output.txt"
            child = f"import time; from pathlib import Path; time.sleep(0.5); Path({str(marker)!r}).touch()"
            parent = (
                "import subprocess, sys; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
            )
            process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
            process.wait(timeout=5)
            _stop_process(process)
            time.sleep(0.6)
            self.assertFalse(marker.exists())


@unittest.skipUnless((MENAGERIE / "franka_emika_panda/panda_nohand.xml").exists(), "local Menagerie assets required")
class TestConditionEquivalence(unittest.TestCase):
    """Exercise the same scenario through both application control paths."""

    def setUp(self):
        """Allocate a platform-independent test artifact directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_live_gain_change_matches_fresh_build(self):
        """Require live parameter notification to match fresh-process physics."""
        live = Scenario("panda")
        live.apply_config({"kp": 70.0, "kd": 0.4})
        session = make_session(live, self.directory.name)
        session.dispatch("reset")
        session.dispatch("step", {"count": live.steps})
        fresh = Scenario("panda", {"kp": 70.0, "kd": 0.4})
        fresh.rollout()
        np.testing.assert_allclose(live.state.joint_q.numpy(), fresh.state.joint_q.numpy(), atol=1e-5)
        self.assertAlmostEqual(live.metrics()["tracking_rmse_rad"], fresh.metrics()["tracking_rmse_rad"], places=6)

    def test_reset_repeats_same_experiment(self):
        """Require a repeated reset and rollout to reproduce measurements."""
        scenario = Scenario("panda")
        first = scenario.rollout()
        second = scenario.rollout()
        self.assertAlmostEqual(first["tracking_rmse_rad"], second["tracking_rmse_rad"], places=8)
        self.assertEqual(first["frames"], second["frames"])

    def test_checkpoint_does_not_count_missing_samples(self):
        """Reject partial histories while preserving checkpoint application time."""
        scenario = Scenario("panda")
        session = make_session(scenario, self.directory.name)
        session.dispatch("step", {"count": 10})
        session.dispatch("checkpoint", {"name": "partial"})
        session.dispatch("step", {"count": 10})
        session.dispatch("restore", {"name": "partial"})
        self.assertEqual(scenario.frame, 10)
        self.assertEqual(len(scenario.errors), 0)
        session.dispatch("step", {"count": scenario.steps - 10})
        self.assertFalse(scenario.metrics()["success"])
        self.assertEqual(scenario.metrics()["sample_count"], scenario.steps - 10)

    def test_failed_execute_rebuilds_same_trial(self):
        """Recover failed live code while retaining configuration and rollout logging."""
        scenario = Scenario("panda", {"kp": 70.0, "kd": 0.4}, variant=1)
        scenario.rollout_log = Path(self.directory.name) / "live_rollouts.jsonl"
        session = make_session(scenario, self.directory.name)
        self.addCleanup(session.close)
        with self.assertRaises(RuntimeError):
            session.dispatch("execute", {"code": "session.scenario.apply_config({'kp': -1})"})
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            session.dispatch("reset")
        session.dispatch("rebuild")
        rebuilt = session.scenario
        self.assertIsNot(rebuilt, scenario)
        self.assertEqual(rebuilt.variant, 1)
        self.assertEqual(rebuilt.config, scenario.config)
        self.assertEqual(session.frame, 0)
        self.assertEqual(session.artifact_directory, Path(self.directory.name))
        session.dispatch("step", {"count": rebuilt.steps})
        logged = [json.loads(line) for line in scenario.rollout_log.read_text().splitlines()]
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["sample_count"], rebuilt.steps)
        fresh = Scenario("panda", rebuilt.config, variant=1)
        fresh.rollout()
        np.testing.assert_allclose(rebuilt.state.joint_q.numpy(), fresh.state.joint_q.numpy(), atol=1e-5)
        self.assertAlmostEqual(rebuilt.metrics()["tracking_rmse_rad"], fresh.metrics()["tracking_rmse_rad"], places=6)


@unittest.skipUnless((HUG_DATA / "scenes/medium_2/aria_data.pkl").exists(), "local HUG recording required")
class TestRecording(unittest.TestCase):
    """Verify the reconstructed target uses actual contiguous source samples."""

    def test_recording_targets_preserve_reset_defaults(self):
        """Keep model defaults unchanged and reproduce recorded rollouts after reset."""
        scenario = Scenario("hug")
        defaults = scenario.model.joint_q.numpy().copy()
        scenario.target(scenario.duration)
        np.testing.assert_array_equal(scenario.model.joint_q.numpy(), defaults)
        first = scenario.rollout()
        second = scenario.rollout()
        np.testing.assert_array_equal(scenario.model.joint_q.numpy(), defaults)
        for metric in scenario.spec["thresholds"]:
            self.assertAlmostEqual(first[metric], second[metric], places=7, msg=metric)

    def test_source_time_and_frame_conversion(self):
        """Preserve recorded sample intervals and shared world recentering."""
        recording = Recording(HUG_DATA)
        np.testing.assert_array_equal(np.diff(recording.indices), np.ones(30))
        self.assertAlmostEqual(recording.time[-1], 3.0)
        for index, sample_time in enumerate(recording.time):
            target = recording.sample(sample_time)
            np.testing.assert_allclose(target[:3], recording.positions[index], atol=1e-7)
            self.assertAlmostEqual(float(np.linalg.norm(target[3:7])), 1.0, places=6)
        np.testing.assert_allclose(
            recording.object_position, np.asarray(recording.properties["world_position"]) + recording.offset
        )


if __name__ == "__main__":
    unittest.main()
