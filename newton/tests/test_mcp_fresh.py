# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a hosted script in clean subprocesses with the MCP ``fresh`` helper."""

import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from newton.mcp import ExampleHost

_SCRIPT = textwrap.dedent(
    """
    import os
    import time

    import newton
    import newton.examples

    GAIN = 1.0


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder()
            builder.add_shape_sphere(builder.add_body(), radius=0.1)
            self.model = builder.finalize(device="cpu")
            self.state_0 = self.model.state()
            self.frame_dt = 0.1
            self.moves = 0
            self.gain = GAIN
            self.hang = args.hang
            self.started = time.time()
            if args.pid_file:
                with open(args.pid_file, "w") as file:
                    file.write(str(os.getpid()))
            time.sleep(args.sleep)

        def step(self):
            self.moves += 1
            while self.hang:
                time.sleep(0.05)

        def summary(self):
            return {"moves": self.moves, "gain": self.gain, "started": self.started, "ended": time.time()}

        @staticmethod
        def create_parser():
            parser = newton.examples.create_parser()
            parser.set_defaults(num_frames=3)
            parser.add_argument("--sleep", type=float, default=0.0)
            parser.add_argument("--hang", action="store_true")
            parser.add_argument("--pid-file", default=None)
            return parser
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z")


class TestMcpFresh(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.script = Path(self.directory.name) / "hosted.py"
        self.script.write_text(_SCRIPT)
        self.host = ExampleHost(self.script, ["--sleep", "0"])
        self.session = self.host.session(artifact_directory=self.directory.name)
        self.addCleanup(self.session.close)

    def execute(self, code: str):
        return self.session.dispatch("execute", {"code": code})["result"]

    def test_runs_the_script_on_disk_not_live_edits(self):
        """Run the saved file with the session's arguments, ignoring live edits and flagging disk edits."""
        self.execute("example.gain = 5.0\nmodule.GAIN = 7.0")
        (report,) = self.execute("fresh(call='example.summary()')")
        self.assertEqual(report["status"], "ok", report)
        self.assertEqual(report["argv"], ["--sleep", "0"])
        self.assertEqual(report["frames"], 3)
        self.assertEqual((report["value"]["moves"], report["value"]["gain"]), (3, 1.0))
        self.assertIn("live edits in this session are not applied", report["source"])
        self.assertFalse(report["script_changed_since_build"])
        self.script.write_text(_SCRIPT.replace("GAIN = 1.0", "GAIN = 2.0"))
        (report,) = self.execute("fresh(['--sleep', '0'], call='example.summary()', frames=5)")
        self.assertEqual((report["value"]["moves"], report["value"]["gain"]), (5, 2.0))
        self.assertTrue(report["script_changed_since_build"])
        # The live example is untouched by fresh runs.
        self.assertEqual(self.execute("(example.gain, example.moves)"), [5.0, 0])

    def test_queues_runs_beyond_the_parallel_limit(self):
        """Never run more than `parallel` processes at once, in argv order."""
        reports = self.execute("fresh([['--sleep', '1.5']] * 3, call='example.summary()', parallel=2)")
        self.assertEqual([report["status"] for report in reports], ["ok"] * 3, reports)
        intervals = [(report["value"]["started"], report["value"]["ended"]) for report in reports]
        overlap = [sum(start <= t < end for start, end in intervals) for t, _ in intervals]
        self.assertLessEqual(max(overlap), 2)
        # The first two launched at once; the third waited for one of them to finish.
        self.assertLess(max(reports[0]["queued_seconds"], reports[1]["queued_seconds"]), 1.0)
        self.assertGreater(reports[2]["queued_seconds"], 1.5)
        self.assertGreaterEqual(intervals[2][0], min(intervals[0][1], intervals[1][1]))

    def test_background_handle_cancel_and_timeout(self):
        """Return a handle without waiting, cancel it, and kill a run that exceeds its timeout."""
        self.execute("handle = fresh([['--hang']], wait=False)")
        self.assertFalse(self.execute("handle.done()"))
        self.execute("handle.cancel()")
        (report,) = self.execute("handle.result(timeout=30)")
        self.assertEqual(report["status"], "cancelled")
        self.assertTrue(self.execute("handle.done()"))
        # Phase and stack reporting are covered by test_examples_headless; under a loaded test
        # runner the process may still be importing when this short timeout expires.
        (report,) = self.execute("fresh([['--hang']], timeout=4)")
        self.assertEqual((report["status"], report["frames"]), ("timeout", 0), report)
        self.assertGreaterEqual(report["wall_seconds"], 4.0)
        if sys.platform.startswith("linux"):
            self.assertEqual(report["signal"], "SIGKILL")

    def test_session_close_kills_running_processes(self):
        """Closing the session kills fresh processes that are still running."""
        pid_file = Path(self.directory.name) / "fresh.pid"
        self.execute(f"handle = fresh([['--hang', '--pid-file', {str(pid_file)!r}]], wait=False)")
        handle = self.session._workspace["handle"]
        deadline = time.monotonic() + 60
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        pid = int(pid_file.read_text())
        self.assertTrue(_alive(pid))
        self.session.close()
        (report,) = handle.result(timeout=30)
        self.assertEqual(report["status"], "cancelled")
        self.assertFalse(_alive(pid))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.host.fresh([])
        # A new session on the same host gets a working runner.
        second = self.host.session(artifact_directory=self.directory.name)
        self.addCleanup(second.close)
        self.assertEqual(second.dispatch("execute", {"code": "fresh([])"})["result"], [])

    def test_rejects_invalid_arguments(self):
        """Validate options in the calling cell instead of failing every run."""
        for code in ("fresh(parallel=0)", "fresh(frames=-1)", "fresh(timeout=0)", "fresh([[{'a': 1}]])"):
            with self.assertRaisesRegex(RuntimeError, "ValueError"):
                self.session.dispatch("execute", {"code": code})


if __name__ == "__main__":
    unittest.main(verbosity=2)
