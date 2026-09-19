# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check synthetic calibration physics without embedding study answers."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .calibration import CalibrationScenario
from .recording import digest
from .rollout import make_session
from .run_agents import _calibration_verification, prepare
from .scenarios import MENAGERIE, ROOT


class TestCalibrationInputs(unittest.TestCase):
    """Check that trajectories and parameters define a bounded forward model."""

    def test_invalid_parameters_rejected(self):
        """Reject incomplete, unknown, nonfinite and out-of-bounds candidates."""
        initial = CalibrationScenario.initial
        for config in (
            {},
            initial | {"gain": 100},
            initial | {"payload_mass": np.nan},
            initial | {"damping_multiplier": 3.0},
            initial | {"joint_friction": -0.1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                CalibrationScenario.validate_config(config)

    def test_trajectories_start_and_finish_at_home(self):
        """Preserve the same reset pose and terminal hold for every episode."""
        for episode in range(3):
            for time_s in (0.0, 2.5, 3.0):
                np.testing.assert_allclose(
                    CalibrationScenario.target(time_s, episode), CalibrationScenario.home, atol=1e-14
                )
        with self.assertRaises(ValueError):
            CalibrationScenario.target(0.0, 3)

    def test_preparation_separates_reference_inputs(self):
        """Give agents training data while retaining only a digest of held-out data."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, heldout = root / "training.npz", root / "heldout.npz"
            np.savez(training, episodes=[0, 1], q=np.zeros((2, 1500, 7)))
            np.savez(heldout, episodes=[2], q=np.zeros((1, 1500, 7)))
            for condition in ("live", "restart"):
                workspace = root / condition
                prepared = prepare(
                    workspace,
                    "panda_calibration",
                    condition,
                    2,
                    600,
                    reference_file=training,
                    verification_reference_file=heldout,
                )
                self.assertEqual(digest(workspace / "reference.npz"), digest(training))
                self.assertNotIn(str(heldout), (workspace / "TASK.md").read_text())
                self.assertNotIn(str(heldout), (workspace / "task.json").read_text())
                self.assertIn("3000 finite steps", prepared["prompt"])
                if condition == "live":
                    self.assertIn("{'count': 3000}", prepared["prompt"])
                else:
                    self.assertIn("--reference reference.npz", prepared["prompt"])

    def test_final_success_requires_matching_training_and_heldout_quality(self):
        """Reject missing training, changed candidates, failed holdout and changed data."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, heldout = root / "reference.npz", root / "heldout.npz"
            training.write_bytes(b"training")
            heldout.write_bytes(b"heldout")
            spec = {"reference_sha256": digest(training), "verification_reference_sha256": digest(heldout)}
            final = {"success": True, "config": dict(CalibrationScenario.initial)}
            self.assertFalse(_calibration_verification(root, spec, final, heldout)[0]["success"])
            measured = {
                "scenario": "panda_calibration",
                "config": final["config"],
                "episodes": [0, 1],
                "frames": 3000,
                "sample_count": 3000,
                "success": True,
            }
            (root / "live_rollouts.jsonl").write_text(json.dumps(measured) + "\n")
            self.assertTrue(_calibration_verification(root, spec, final, heldout)[0]["success"])
            failed_heldout = _calibration_verification(root, spec, final | {"success": False}, heldout)[0]
            self.assertTrue(failed_heldout["training_success"])
            self.assertFalse(failed_heldout["success"])
            changed_config = final | {"config": final["config"] | {"payload_mass": 0.9}}
            self.assertFalse(_calibration_verification(root, spec, changed_config, heldout)[0]["success"])
            training.write_bytes(b"changed")
            self.assertFalse(_calibration_verification(root, spec, final, heldout)[0]["success"])


@unittest.skipUnless((MENAGERIE / "franka_emika_panda/panda_nohand.xml").exists(), "local Menagerie asset required")
class TestCalibrationPhysics(unittest.TestCase):
    """Check exact payload construction and live-versus-fresh trajectories."""

    def test_cli_accepts_calibration_variant_two_only(self):
        """Run calibration variant 2 while rejecting invalid original-task variants."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = output / "config.py"
            config.write_text("CONFIG = " + repr(CalibrationScenario.initial) + "\n")
            reference = output / "reference.npz"
            np.savez(reference, episodes=[0, 1], q=np.zeros((2, 1500, 7)))
            command = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "tools.mcp_evaluation.rollout",
                "--config",
                str(config),
                "--output",
                str(output / "metrics.json"),
            ]
            completed = subprocess.run(
                [*command, "--scenario", "panda_calibration", "--variant", "2", "--reference", str(reference)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["variant"], 2)
            self.assertEqual(metrics["frames"], 3000)
            self.assertTrue(metrics["finite"])
            for scenario, variant in (("panda", "2"), ("panda_calibration", "-1")):
                with self.subTest(scenario=scenario, variant=variant):
                    invalid = subprocess.run(
                        [*command, "--scenario", scenario, "--variant", variant],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(invalid.returncode, 2, invalid.stdout + invalid.stderr)
                    self.assertIn("variant", invalid.stderr)

    def test_live_physical_edit_matches_fresh(self):
        """Require mass, inertia and joint-loss notifications to preserve physics."""
        config = {"payload_mass": 2.0, "damping_multiplier": 2.0, "joint_friction": 0.3}
        live = CalibrationScenario()
        live.apply_config(config)
        fresh = CalibrationScenario(config)
        payload = fresh.payload_body
        self.assertAlmostEqual(float(fresh.model.body_mass.numpy()[payload]), config["payload_mass"], places=6)
        expected_inertia = np.eye(3) * (0.4 * config["payload_mass"] * fresh.payload_radius**2)
        np.testing.assert_allclose(fresh.model.body_inertia.numpy()[payload], expected_inertia, atol=1e-9)
        reference_traces = []
        for episode in (0, 1, 2):
            observed, expected = live.rollout_episode(episode), fresh.rollout_episode(episode)
            reference_traces.append(expected["q"])
            self.assertTrue(observed["finite"])
            self.assertEqual(observed["frames"], CalibrationScenario.steps)
            np.testing.assert_allclose(observed["q"], expected["q"], atol=2e-6, rtol=0)
            np.testing.assert_allclose(observed["qd"], expected["qd"], atol=2e-5, rtol=0)
        repeated = live.rollout_episode(2)
        np.testing.assert_array_equal(repeated["q"], observed["q"])
        np.testing.assert_array_equal(repeated["qd"], observed["qd"])

        with tempfile.TemporaryDirectory() as directory:
            reference_file = Path(directory) / "reference.npz"
            reference_q = np.stack(reference_traces[:2])
            np.savez(reference_file, episodes=[0, 1], q=reference_q)
            candidate = CalibrationScenario(reference_file=reference_file)
            candidate.apply_config(config)
            session = make_session(candidate, Path(directory) / "observations")
            self.addCleanup(session.close)
            session.dispatch("reset")
            session.dispatch("step", {"count": 3000})
            self.assertTrue(candidate.metrics()["success"])
            np.testing.assert_allclose(np.asarray(candidate.q_trace), reference_q.reshape(-1, 7), atol=2e-6, rtol=0)
            first = np.asarray(candidate.q_trace).copy()
            session.dispatch("reset")
            session.dispatch("step", {"count": 3000})
            np.testing.assert_array_equal(candidate.q_trace, first)
            candidate.reference_q[0] += 0.0004
            candidate.rollout()
            self.assertLess(candidate.metrics()["trajectory_rmse_rad"], 0.00035)
            self.assertFalse(candidate.metrics()["success"])


if __name__ == "__main__":
    unittest.main()
