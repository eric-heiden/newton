# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Preserve failed old-task trials and verify immutable inputs before executing them."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import run_agents
from .recording import digest


class TestRunnerIntegrity(unittest.TestCase):
    def setUp(self):
        """Create a temporary numerical source commitment without touching repository files."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.physics = self.checkout / "physics.py"
        self.physics.write_text("PHYSICS = 'fixed'\n")
        self.counter = 0
        self.context = {"exit_code": 0, "timed_out": False, "external_sources_unchanged": True}
        for replacement in (
            patch.object(run_agents, "ROOT", self.checkout),
            patch.object(run_agents, "source_hashes", side_effect=lambda: {"physics.py": digest(self.physics)}),
            patch.object(run_agents, "run_context", side_effect=lambda *args, **kwargs: dict(self.context)),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

    def prepare(self):
        """Prepare one ordinary old-task trial with frozen source and task hashes."""
        self.counter += 1
        workspace = self.root / f"trial-{self.counter}"
        prepared = run_agents.prepare(workspace, "panda", "restart", 0, 30)
        return workspace, prepared

    @staticmethod
    def passed_verifier(args, **kwargs):
        """Write a valid fresh physical result in place of starting a simulator."""
        output = Path(args[args.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"success": True, "finite": True, "config": {}}))
        return subprocess.CompletedProcess(args, 0, stdout="verified\n", stderr="")

    def test_clean_trial_keeps_physics_and_study_success(self):
        """Accept fresh passing physics only when the study eligibility gates also pass."""
        workspace, prepared = self.prepare()
        with patch.object(run_agents.subprocess, "run", side_effect=self.passed_verifier) as verifier:
            summary = run_agents.run_trial(workspace, prepared)
        verifier.assert_called_once()
        self.assertTrue(summary["physics_success"])
        self.assertTrue(summary["study_success"])
        self.assertTrue(summary["quality"]["success"])
        self.assertEqual(json.loads((workspace / "summary.json").read_text()), summary)

    def test_agent_failure_and_candidate_budget_disqualify_passing_physics(self):
        """Retain physical quality while disqualifying timed-out, failed or over-budget contexts."""
        for context, count in (({"timed_out": True}, 1), ({"exit_code": 1}, 1), ({}, 13)):
            with self.subTest(context=context, count=count):
                workspace, prepared = self.prepare()
                self.context = {"exit_code": 0, "timed_out": False, "external_sources_unchanged": True} | context
                with (
                    patch.object(run_agents.subprocess, "run", side_effect=self.passed_verifier),
                    patch.object(run_agents, "_trial_measurements", return_value={"candidate_rollouts": count}),
                ):
                    summary = run_agents.run_trial(workspace, prepared)
                self.assertTrue(summary["physics_success"])
                self.assertFalse(summary["study_success"])
                self.assertFalse(summary["quality"]["success"])

    def test_changed_numerical_source_blocks_verification(self):
        """Reject changed physics before launching any process with verification inputs."""
        workspace, prepared = self.prepare()
        self.physics.write_text("PHYSICS = 'changed'\n")
        with patch.object(run_agents.subprocess, "run", side_effect=self.passed_verifier) as verifier:
            summary = run_agents.run_trial(workspace, prepared)
        verifier.assert_not_called()
        self.assertFalse(summary["shared_sources_unchanged"])
        self.assertFalse(summary["study_success"])
        self.assertEqual(summary["verification_process_starts"], 0)

    def test_changed_task_or_manifest_blocks_verification(self):
        """Check both task instructions and the source commitment manifest before verification."""
        for name in ("task.json", "integrity-manifest.json"):
            with self.subTest(name=name):
                workspace, prepared = self.prepare()
                (workspace / name).write_text("{}\n")
                with patch.object(run_agents.subprocess, "run", side_effect=self.passed_verifier) as verifier:
                    summary = run_agents.run_trial(workspace, prepared)
                verifier.assert_not_called()
                self.assertFalse(summary["study_success"])

    def test_changed_calibration_reference_blocks_verification(self):
        """Verify both public and held-out reference commitments before opening either in a child."""
        for changed in ("reference.npz", "heldout.npz"):
            with self.subTest(changed=changed):
                workspace, prepared = self.prepare()
                for name in ("reference.npz", "heldout.npz"):
                    (workspace / name).write_bytes(b"fixed numeric fixture")
                prepared["spec"].update(
                    scenario="panda_calibration",
                    reference_sha256=digest(workspace / "reference.npz"),
                    verification_reference_sha256=digest(workspace / "heldout.npz"),
                )
                prepared["verification_reference_file"] = str(workspace / "heldout.npz")
                run_agents.write_task(workspace, prepared["spec"])
                prepared["task_sha256"] = digest(workspace / "task.json")
                (workspace / changed).write_bytes(b"changed numeric fixture")
                with patch.object(run_agents.subprocess, "run", side_effect=self.passed_verifier) as verifier:
                    summary = run_agents.run_trial(workspace, prepared)
                verifier.assert_not_called()
                self.assertFalse(summary["references_unchanged"])
                self.assertFalse(summary["study_success"])

    def test_changed_external_server_blocks_verification(self):
        """Treat an altered third-party execution server as an integrity failure."""
        workspace, prepared = self.prepare()
        self.context["external_sources_unchanged"] = False
        with patch.object(run_agents.subprocess, "run", side_effect=self.passed_verifier) as verifier:
            summary = run_agents.run_trial(workspace, prepared)
        verifier.assert_not_called()
        self.assertFalse(summary["study_success"])

    def test_verifier_timeout_retains_failed_summary_and_partial_output(self):
        """Keep timed-out verification as a failed recorded trial instead of raising away its evidence."""
        workspace, prepared = self.prepare()
        error = subprocess.TimeoutExpired("verifier", 120, output=b"partial stdout", stderr=b"partial stderr")
        with patch.object(run_agents.subprocess, "run", side_effect=error):
            summary = run_agents.run_trial(workspace, prepared)
        self.assertFalse(summary["study_success"])
        self.assertTrue(summary["quality"]["verification_timed_out"])
        self.assertEqual((workspace / "verification.log").read_text(), "partial stdoutpartial stderr")
        self.assertFalse(json.loads((workspace / "summary.json").read_text())["study_success"])

    def test_verifier_missing_or_malformed_result_remains_a_failed_measurement(self):
        """Record verifier output failures without losing the enclosing trial summary."""
        for output in (None, "not JSON", "[]"):
            with self.subTest(output=output):
                workspace, prepared = self.prepare()

                def invalid_verifier(args, _output=output, **kwargs):
                    path = Path(args[args.index("--output") + 1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if _output is not None:
                        path.write_text(_output)
                    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

                with patch.object(run_agents.subprocess, "run", side_effect=invalid_verifier):
                    summary = run_agents.run_trial(workspace, prepared)
                self.assertFalse(summary["study_success"])
                self.assertIn("verification_error", summary["quality"])

    def test_verifier_side_effect_cannot_change_frozen_sources(self):
        """Recheck integrity after a verifier completes while preserving its physical result."""
        workspace, prepared = self.prepare()

        def changed_verifier(args, **kwargs):
            result = self.passed_verifier(args, **kwargs)
            self.physics.write_text("PHYSICS = 'changed by verifier'\n")
            return result

        with patch.object(run_agents.subprocess, "run", side_effect=changed_verifier):
            summary = run_agents.run_trial(workspace, prepared)
        self.assertTrue(summary["physics_success"])
        self.assertFalse(summary["shared_sources_unchanged"])
        self.assertFalse(summary["study_success"])


if __name__ == "__main__":
    unittest.main()
