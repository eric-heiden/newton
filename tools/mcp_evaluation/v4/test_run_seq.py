# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests of the sequential replicate runner. CPU only; no trials run.

``python -m unittest tools.mcp_evaluation.v4.test_run_seq``
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.mcp_evaluation.v4 import run_seq


class TestSchedule(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_rounds_alternate_the_order_per_pair(self):
        trials = run_seq.schedule(["a:opus", "b:astra"], 3, first="mcp")
        names = [t["name"] for t in trials]
        self.assertEqual(names[:4], ["a-opus-mcp-p0", "a-opus-restart-p0", "b-astra-restart-p0", "b-astra-mcp-p0"])
        self.assertEqual(names[4:6], ["a-opus-restart-p1", "a-opus-mcp-p1"])
        self.assertEqual(len(names), 12)
        for pair in ("a-opus", "b-astra"):
            firsts = [names[i] for i in range(0, 12, 2) if names[i].startswith(pair)]
            self.assertEqual(
                sorted("mcp" in name for name in firsts),
                [False, True, True] if pair == "a-opus" else [False, False, True],
            )
        self.assertEqual(run_seq.schedule(["a:opus"], 1, first="restart", start=4)[0]["name"], "a-opus-restart-p4")
        with self.assertRaises(ValueError):
            run_seq.schedule(["a"], 1)

    def test_run_skips_finished_trials_and_verifies_after_the_last(self):
        directory = self.root / "i16"
        (directory / "a-opus-mcp-p0").mkdir(parents=True)
        (directory / "a-opus-mcp-p0" / "summary.json").write_text("{}")
        (directory / "a-opus-restart-p0").mkdir()  # left by an interrupted run
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if "--spec" in command:
                spec = json.loads(Path(command[command.index("--spec") + 1]).read_text())
                Path(spec["workspace"]).mkdir()
                (Path(spec["workspace"]) / "summary.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0)

        plan = {"directory": str(directory), "pairs": ["a:opus"], "replicates": 2, "verify_snapshots": "all"}
        with mock.patch.object(run_seq.subprocess, "run", side_effect=fake_run):
            run_dirs = run_seq.run(plan)
        self.assertEqual(len(commands), 4)  # p0 restart, p1 restart, p1 mcp, verification
        self.assertTrue((directory / "a-opus-restart-p0.aborted-1").is_dir())
        self.assertEqual(commands[-1][-1], "--all")
        self.assertEqual(commands[-1][3:5], ["--verify-snapshots", str(run_dirs[0])])
        self.assertEqual(list(directory.glob(".spec-*")), [])
        log = (directory / "launch.log").read_text()
        self.assertIn("SKIP a-opus-mcp-p0", log)
        self.assertTrue(log.rstrip().endswith("DONE"))


if __name__ == "__main__":
    unittest.main()
