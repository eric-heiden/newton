# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run example scripts in clean processes with ``python -m newton.examples.headless``."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from newton.examples.headless import _parse_override, _resolve, _split, run_headless

_SCRIPT = textwrap.dedent(
    """
    import subprocess
    import sys
    import time

    import numpy as np

    import newton.examples

    GAIN = 1.0
    SEED = -1
    PARAMS = {"horizon": 10, "mode": "slow"}
    WEIGHTS = np.array([1.0, 2.0], dtype=np.float32)


    class Example:
        label = "default"

        def __init__(self, viewer, args):
            self.args = args
            self.count = 0
            self.sim_time = 0.0
            self.gain = GAIN
            if args.spawn:
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
                with open(args.spawn, "w") as file:
                    file.write(str(child.pid))
            print("constructed with scale", args.scale)

        def step(self):
            self.count += 1
            self.sim_time += 0.25
            time.sleep(self.args.sleep)
            if self.count == self.args.fail_at:
                raise RuntimeError(f"failure in frame {self.count}")
            if self.count == self.args.hang_at:
                self.hang()

        def hang(self):
            while True:
                time.sleep(0.05)

        def test_final(self):
            if self.count < 3:
                raise AssertionError(f"only {self.count} frames")

        @staticmethod
        def create_parser():
            parser = newton.examples.create_parser()
            parser.set_defaults(num_frames=4)
            parser.add_argument("--scale", type=float, default=1.0)
            parser.add_argument("--fail-at", type=int, default=-1)
            parser.add_argument("--hang-at", type=int, default=-1)
            parser.add_argument("--spawn", default=None)
            parser.add_argument("--sleep", type=float, default=0.0)
            return parser


    if __name__ == "__main__":
        raise SystemExit("the runner must not execute the main block")
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed process may linger as a zombie until its new parent reaps it.
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z")


class TestHeadlessRunner(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="newton-headless-test-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True))
        self.script = self.directory / "counter.py"
        self.script.write_text(_SCRIPT)

    def cli(self, *arguments: str) -> tuple[subprocess.CompletedProcess, dict]:
        output = self.directory / "report.json"
        output.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "newton.examples.headless",
            str(self.script),
            *arguments,
            "--json",
            str(output),
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        return process, json.loads(output.read_text())

    def test_success_writes_report_with_converted_value(self):
        """Step the script's own frame count, evaluate the call, and convert its value to JSON."""
        call = (
            "{'count': example.count * args.scale, 'array': np.array([0.5, np.nan, -np.inf]), 'time': example.sim_time}"
        )
        process, report = self.cli("--scale", "2", "--call", call)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["phase"], "done")
        self.assertEqual((report["frames"], report["frames_requested"]), (4, 4))
        self.assertEqual(report["value"], {"count": 8.0, "array": [0.5, "nan", "-inf"], "time": 1.0})
        self.assertEqual(report["sim_time"], 1.0)
        self.assertEqual(report["argv"], ["--scale", "2"])
        self.assertEqual((report["exit_code"], report["exception"], report["stack"]), (0, None, None))
        self.assertIn("constructed with scale 2.0", report["stdout_tail"])
        # The CLI forwards the script's output and ends with a summary line.
        self.assertIn("constructed with scale 2.0", process.stdout)
        self.assertIn("headless: ok", process.stderr)

    def test_exception_reports_phase_frames_and_traceback(self):
        """Report where the script failed with a traceback that starts in the script."""
        process, report = self.cli("--fail-at", "3", "--frames", "10")
        self.assertEqual(process.returncode, 1)
        self.assertEqual(report["status"], "error")
        self.assertEqual((report["phase"], report["frames"], report["frames_requested"]), ("step", 2, 10))
        self.assertEqual(report["exception"]["type"], "RuntimeError")
        self.assertIn("failure in frame 3", report["exception"]["message"])
        traceback = report["exception"]["traceback"]
        self.assertIn(str(self.script), traceback)
        self.assertIn("in step", traceback)
        self.assertNotIn("headless.py", traceback)

    @unittest.skipUnless(sys.platform.startswith("linux"), "process-group kill and stack dumps are tested on Linux")
    def test_timeout_kills_process_tree_and_reports_stack(self):
        """Kill a hung run and the processes it started, and report the stack it hung in."""
        pid_file = self.directory / "grandchild.pid"
        # Generous enough for a loaded test runner to import Newton and reach the hang.
        process, report = self.cli("--spawn", str(pid_file), "--hang-at", "2", "--frames", "100", "--timeout", "15")
        self.assertEqual(process.returncode, 124)
        self.assertEqual(report["status"], "timeout")
        self.assertEqual((report["phase"], report["frames"]), ("step", 1), report)
        self.assertEqual(report["signal"], "SIGKILL")
        self.assertLess(report["wall_seconds"], 25.0)
        self.assertIn("in hang", report["stack"])
        self.assertIn(str(self.script), report["stack"])
        self.assertNotIn("headless.py", report["stack"])
        grandchild = int(pid_file.read_text())
        self.assertFalse(_alive(grandchild), "the process started by the script survived the timeout")

    def test_python_api_test_mode_statements_and_argument_errors(self):
        """Run --test checks after the call, return `result` from statements, and report parser errors."""
        failed = run_headless(self.script, ["--test"], frames=2, call="result = example.count")
        self.assertEqual((failed["status"], failed["phase"], failed["value"]), ("error", "test", 2))
        self.assertEqual(failed["exception"]["type"], "AssertionError")
        passed = run_headless(str(self.script), ["--test"], tail=0)
        self.assertEqual((passed["status"], passed["frames"], passed["stdout_tail"]), ("ok", 4, ""))
        unknown = run_headless(self.script, ["--no-such-option"], frames=1)
        self.assertEqual((unknown["status"], unknown["phase"]), ("error", "build"))
        self.assertEqual(unknown["exception"]["type"], "SystemExit")
        self.assertIn("unrecognized arguments: --no-such-option", unknown["stderr_tail"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "signal names are tested on Linux")
    def test_crash_is_reported_with_signal(self):
        """Report a process that dies without finishing as crashed, with its signal and phase."""
        report = run_headless(self.script, frames=0, call="import os; os.abort()")
        self.assertEqual((report["status"], report["phase"], report["signal"]), ("crashed", "call", "SIGABRT"))
        self.assertIn("Fatal Python error", report["stderr_tail"])

    def test_set_assigns_globals_attributes_and_keys(self):
        """--set assigns existing module globals, class attributes, and dictionary keys before the example is built."""
        call = (
            "{'gain': example.gain, 'params': module.PARAMS, 'label': example.label, "
            "'weights': module.WEIGHTS, 'dtype': str(module.WEIGHTS.dtype)}"
        )
        overrides = ["GAIN=2.5", "PARAMS.horizon=20", "PARAMS.mode=quick", "Example.label='fast'", "WEIGHTS=[3, 4]"]
        process, report = self.cli(*[item for text in overrides for item in ("--set", text)], "--call", call)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            report["value"],
            {
                "gain": 2.5,
                "params": {"horizon": 20, "mode": "quick"},
                "label": "fast",
                "weights": [3.0, 4.0],
                "dtype": "float32",
            },
        )
        self.assertEqual(
            report["overrides"],
            {"GAIN": 2.5, "PARAMS.horizon": 20, "PARAMS.mode": "quick", "Example.label": "fast", "WEIGHTS": [3, 4]},
        )
        self.assertEqual(report["realtime_factor"], round(1.0 / report["seconds"]["step"], 4))
        # A name the script does not define stops the run before the example is built.
        process, report = self.cli("--set", "GAINN=2")
        self.assertEqual(process.returncode, 1)
        self.assertEqual((report["status"], report["phase"]), ("error", "load"))
        self.assertEqual(report["exception"]["type"], "AttributeError")
        self.assertIn("the script has no name 'GAINN'; did you mean 'GAIN'?", report["exception"]["message"])
        _, report = self.cli("--set", "PARAMS.horizn=3")
        self.assertIn("'PARAMS' has no key 'horizn'; did you mean 'horizon'?", report["exception"]["message"])

    def test_repeat_runs_fresh_processes_and_reports_progress(self):
        """--repeat runs one process per run with {run} in --set values; --progress prints lines while it runs."""
        process, report = self.cli(
            "--repeat", "3", "--set", "SEED={run}", "--call", "(module.SEED, example.count)", "--progress", "0"
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual((report["status"], report["repeat"], report["status_counts"]), ("ok", 3, {"ok": 3}))
        self.assertEqual(report["values"], [[0, 4], [1, 4], [2, 4]])
        self.assertEqual([run["run"] for run in report["runs"]], [0, 1, 2])
        walls = [run["wall_seconds"] for run in report["runs"]]
        self.assertEqual(report["wall_seconds"]["total"], round(sum(walls), 3))
        self.assertEqual((report["wall_seconds"]["min"], report["wall_seconds"]["max"]), (min(walls), max(walls)))
        self.assertIn("headless: run 2/3: ok", process.stderr)
        self.assertIn("headless: 3 of 3 runs: 3 ok", process.stderr)
        self.assertNotIn(", phase step, frame", process.stderr)
        # A failing run sets the overall status and exit code.
        process, report = self.cli("--repeat", "2", "--set", "Example.label={run}", "--fail-at", "2", "--progress", "0")
        self.assertEqual(process.returncode, 1)
        self.assertEqual((report["status"], report["status_counts"]), ("error", {"error": 2}))
        # Progress lines name the phase and frame of a run in flight.
        process, report = self.cli("--sleep", "0.25", "--frames", "12", "--progress", "0.5")
        self.assertEqual(report["status"], "ok", process.stderr)
        self.assertRegex(process.stderr, r"headless: \d+ s, phase step, frame \d+/12")

    def test_python_api_overrides_must_be_literals(self):
        """run_headless() takes overrides as Python literals and rejects other values and names before starting."""
        report = run_headless(self.script, frames=1, overrides={"GAIN": 3, "PARAMS.mode": "x"}, call="example.gain")
        self.assertEqual((report["status"], report["value"]), ("ok", 3))
        self.assertEqual(report["overrides"], {"GAIN": 3, "PARAMS.mode": "x"})
        with self.assertRaisesRegex(ValueError, "Python literal"):
            run_headless(self.script, overrides={"GAIN": object()})
        with self.assertRaisesRegex(ValueError, "global name"):
            run_headless(self.script, overrides={"GAIN[0]": 1})
        with self.assertRaisesRegex(ValueError, "progress"):
            run_headless(self.script, progress=0)
        self.assertEqual(_parse_override("A={run}", 3), ("A", 3))
        self.assertEqual(_parse_override("B = 'x=1'"), ("B", "x=1"))
        self.assertEqual(_parse_override("C=fast mode"), ("C", "fast mode"))
        with self.assertRaisesRegex(ValueError, "NAME=VALUE"):
            _parse_override("C")

    def test_command_line_splitting_and_script_resolution(self):
        """Take runner options anywhere after SCRIPT, pass everything after -- to the script."""
        options = ["--frames=7", "--render", "--set", "A=1", "--repeat", "2", "--progress=0"]
        runner, script, script_args = _split(["--timeout", "5", "s.py", "--seed", "3", *options, "--", "--frames", "2"])
        self.assertEqual(
            runner, ["--timeout", "5", "--frames=7", "--render", "--set", "A=1", "--repeat", "2", "--progress=0"]
        )
        self.assertEqual(script, "s.py")
        self.assertEqual(script_args, ["--seed", "3", "--frames", "2"])
        self.assertEqual(_resolve("basic_pendulum"), {"module": "newton.examples.basic.example_basic_pendulum"})
        self.assertEqual(_resolve(self.script), {"script": str(self.script.resolve())})
        with self.assertRaises(FileNotFoundError):
            _resolve(self.directory / "missing.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
