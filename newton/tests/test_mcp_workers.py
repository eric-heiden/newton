# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import warp as wp

import newton
from newton._src.mcp import shipping
from newton.mcp import ExampleHost, SimulationClient, SimulationSession, WorkerPool


def _scene():
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.1)
    return builder.finalize(device="cpu")


_SCRIPT = textwrap.dedent(
    """
    import newton


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_shape_sphere(builder.add_body(), radius=0.1)
            self.model = builder.finalize(device="cpu")
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.speed = 1.0
            self.frame_dt = 0.1

        def step(self):
            self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A terminated child of this process stays a zombie until reaped; it no longer runs.
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class TestMcpWorkers(unittest.TestCase):
    """A session with two worker sessions attached through their connection files."""

    @classmethod
    def setUpClass(cls):
        # Separate processes: Warp is not used from several threads of one process at once.
        cls.directory = tempfile.TemporaryDirectory()
        cls.script = Path(cls.directory.name) / "tiny.py"
        cls.script.write_text(_SCRIPT)
        cls.paths = [Path(cls.directory.name) / f"worker-{i}.json" for i in range(2)]
        cls.processes = [
            subprocess.Popen(
                [sys.executable, "-m", "newton.mcp", "host", str(cls.script), "--connection-file", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for path in cls.paths
        ]
        deadline = time.monotonic() + 120
        for path, process in zip(cls.paths, cls.processes, strict=True):
            while not path.with_suffix(".ready").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        for process in cls.processes:
            process.terminate()
        for process in cls.processes:
            process.wait(timeout=30)
        cls.directory.cleanup()

    def setUp(self):
        for process in self.processes:
            self.assertIsNone(process.poll(), "worker session exited")
        model = _scene()
        self.session = SimulationSession(
            model, newton.solvers.SolverXPBD(model), allow_execute=True, workers=self.paths
        )
        self.addCleanup(self.session.close)

    def execute(self, code):
        return self.session.dispatch("execute", {"code": code})

    def test_broadcast_map_and_errors(self):
        """Define helpers on every worker, fan out jobs concurrently, and isolate failures."""
        self.assertEqual(self.session.dispatch("describe")["capabilities"]["workers"], 2)
        result = self.execute(
            "workers.broadcast('import time\\ndef slow(x):\\n    time.sleep(0.4)\\n    return x * x\\nstate.body_q.shape[0]')"
        )
        self.assertEqual(result["result"], [1, 1])
        started = time.perf_counter()
        result = self.execute("workers.map('result = slow(args)', [1, 2, 3, 4])")
        elapsed = time.perf_counter() - started
        self.assertEqual(result["result"], [1, 4, 9, 16])
        # Four 0.4 s jobs on two workers take about 0.8 s, not 1.6 s.
        self.assertLess(elapsed, 1.4)
        result = self.execute("workers.map('result = 1 / args', [1, 0, 2])")
        self.assertEqual(result["result"][0], 1)
        self.assertIn("ZeroDivisionError", result["result"][1]["error"])
        self.assertEqual(result["result"][2], 0.5)
        # The failed job did not block its worker.
        self.assertEqual(self.execute("workers.map('result = args + 1', [1, 2, 3])")["result"], [2, 3, 4])

    def test_functions_ship_by_source_with_helpers_and_globals(self):
        """Send cell functions, lambdas, and closures with the helpers and globals they read."""
        self.execute("def helper(x):\n    return x * SCALE\nSCALE = 3\nTABLE = np.ones(4)")
        self.execute(
            "OFFSET = 0.5\ndef evaluate(x, offset=OFFSET):\n    return helper(x) + float(TABLE.sum()) + offset"
        )
        self.assertEqual(self.execute("workers.map(evaluate, [1, 2, 3])")["result"], [7.5, 10.5, 13.5])
        # Functions see the session's current globals; defaults keep their definition-time value.
        self.assertEqual(self.execute("SCALE = 10\nOFFSET = 9.0\nworkers.map(evaluate, [1])")["result"], [14.5])
        self.assertEqual(self.execute("workers.map(lambda a, b: a * b + SCALE, [1, 2], [3, 4])")["result"], [13, 18])
        result = self.execute(
            "def make(k):\n    def inner(x):\n        return x * k + SCALE\n    return inner\nworkers.map(make(5), [1, 2])"
        )
        self.assertEqual(result["result"], [15, 20])
        # Live objects keep their names on the worker: `state` is the worker's own state.
        self.assertEqual(self.execute("workers.submit(lambda: state.body_q.shape[0]).result()")["result"], 1)
        # Cell classes travel by source; their instances in arguments and results are pickled.
        self.execute(
            "from dataclasses import dataclass\n@dataclass\nclass Params:\n    a: float\n    b: float\n"
            "def scaled(p, factor=2.0):\n    return Params(p.a * factor, p.b * factor)"
        )
        result = self.execute(
            "[(type(p).__name__, p.a, p.b) for p in workers.map(scaled, [Params(1, 2), Params(3, 4)])]"
        )
        self.assertEqual(result["result"], [["Params", 2.0, 4.0], ["Params", 6.0, 8.0]])
        result = self.execute("workers.submit(scaled, Params(1, 1), factor=5.0).result().a")
        self.assertEqual(result["result"], 5.0)
        # Warp arrays arrive as Warp arrays on the same device; Warp kernels from cells travel too.
        self.execute(
            "@wp.kernel\ndef twice(a: wp.array[wp.vec3]):\n    i = wp.tid()\n    a[i] = a[i] * 2.0\n"
            "def double(a):\n    wp.launch(twice, a.shape[0], inputs=[a], device=a.device)\n    return a"
        )
        result = self.execute(
            "out = workers.submit(double, wp.array(np.ones((2, 3), np.float32), dtype=wp.vec3, device='cpu')).result()\n"
            "(out.dtype is wp.vec3, out.numpy().tolist())"
        )
        self.assertEqual(result["result"], [True, [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]])
        # Results larger than one message travel through a file.
        self.assertEqual(self.execute("workers.submit(np.ones, 200_000).result().sum()")["result"], 200000.0)

    def test_worker_errors_name_the_cell_line_and_unsent_globals(self):
        """Report the cell and line of an exception raised on a worker, and globals that could not be sent."""
        self.execute("def ratio(x):\n    y = 1.0\n    return y / x")
        result = self.execute("workers.map(ratio, [1, 0])")["result"]
        self.assertEqual(result[0], 1.0)
        self.assertIn("ZeroDivisionError", result[1]["error"])
        self.assertRegex(result[1]["error"], r"cell \d+ line 3 in ratio: return y / x")
        result = self.execute("solver_copy = solver\nworkers.map(lambda x: solver_copy, [1])")["result"]
        self.assertIn("'solver_copy' was not sent", result[0]["error"])
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
            self.execute("workers.submit(ratio, 0).result()")

    def test_sync_copies_values_once(self):
        """Copy session values into every worker; functions do not resend a synced object."""
        result = self.execute("big = np.arange(1_000_000, dtype=np.float64)\nworkers.sync(big=big)")["result"]
        self.assertEqual(result["names"], ["big"])
        self.assertEqual(result["workers"], 2)
        self.assertGreater(result["bytes"], 8_000_000)
        self.assertEqual(self.execute("workers.map('float(big[args])', [5, 6])")["result"], [5.0, 6.0])
        self.execute("def pick(i):\n    return float(big[i])\nsmall = 3")
        self.assertEqual(self.execute("workers.sync('small')")["result"]["names"], ["small"])
        self.assertEqual(self.execute("workers.map(lambda i: pick(i) + small, [1])")["result"], [4.0])
        pool = self.session.workers
        sent = shipping.prepare(self.session._workspace["pick"], pool._skip)
        self.assertLess(len(sent.data), 4096)
        self.execute("big = np.zeros(10)")
        resent = shipping.prepare(self.session._workspace["pick"], pool._skip)
        self.assertNotEqual(resent.key, sent.key)
        self.assertEqual(self.execute("workers.map(pick, [5])")["result"], [0.0])

    def test_background_jobs_report_in_the_next_response(self):
        """Start jobs that return at once, list finished ones in the next response, and collect results."""
        self.execute("import time\ndef slow(x):\n    time.sleep(0.3)\n    return x * 2")
        started = time.perf_counter()
        result = self.execute("[jobs.start(slow, 1), jobs.start(slow, 2), jobs.start('result = 1 / args', 0)]")
        self.assertLess(time.perf_counter() - started, 0.25)
        self.assertEqual(result["result"], [1, 2, 3])
        unfinished = result["jobs"].get("running", []) + result["jobs"].get("queued", [])
        self.assertEqual(sorted(unfinished), [1, 2, 3])
        deadline = time.monotonic() + 10
        report = {}
        while time.monotonic() < deadline and len(report.get("finished", [])) < 3:
            time.sleep(0.2)
            response = self.execute("None")
            report.setdefault("finished", []).extend(response.get("jobs", {}).get("finished", []))
        finished = {entry["id"]: entry for entry in report["finished"]}
        self.assertEqual(finished[1]["result"], "2")
        self.assertEqual(finished[2]["status"], "done")
        self.assertEqual(finished[3]["status"], "failed")
        self.assertIn("ZeroDivisionError", finished[3]["error"])
        # Reported jobs are not reported again, but wait() still returns their full results once.
        self.assertNotIn("jobs", self.execute("None"))
        waited = self.execute("jobs.wait(timeout=5, any=False)")["result"]
        self.assertEqual([entry["result"] for entry in waited["finished"][:2]], [2, 4])
        self.assertEqual(self.execute("jobs.wait(timeout=0)")["result"], {"finished": [], "running": []})
        # A queued job can be cancelled; a running one cannot.
        result = self.execute(
            "ids = [jobs.start(slow, i) for i in range(4)]\ntime.sleep(0.1)\n[jobs.cancel(i) for i in ids]"
        )
        self.assertEqual(result["result"][:2], [False, False])
        self.assertEqual(result["result"][2:], [True, True])
        self.assertEqual(self.execute("jobs.result(ids[1])")["result"], 2)
        with self.assertRaisesRegex(RuntimeError, "not available"):
            self.execute("jobs.start(slow, 1, where='fresh')")


class TestMcpLaunchedWorkers(unittest.TestCase):
    """Worker processes owned by the pool: rebuilds, restarts, resizing, and progress of jobs."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.script = Path(self.directory.name) / "tiny.py"
        self.script.write_text(_SCRIPT)
        self.pool = WorkerPool.launch(
            self.script, [], count=1, max_count=2, directory=self.directory.name, name="session"
        )
        self.addCleanup(self._close)
        host = ExampleHost(self.script)
        host.build()
        self.assertEqual([w["state"] for w in self.pool.wait_ready(120)], ["ready"])
        self.session = host.session(artifact_directory=self.directory.name, workers=self.pool)
        self.addCleanup(self.session.close)
        self.pids = set()

    def _close(self):
        self.pids.update(w["pid"] for w in self.pool.status() if w["pid"])
        self.pool.close()
        for pid in self.pids:
            self.assertFalse(_alive(pid), f"worker process {pid} still runs")

    def execute(self, code):
        result = self.session.dispatch("execute", {"code": code})
        self.pids.update(w["pid"] for w in self.pool.status() if w["pid"])
        return result

    def test_rebuild_crash_restart_and_resize(self):
        """Follow a rebuild, restart a crashed worker with its setup replayed, and change the worker count."""
        self.execute("def speed(_):\n    print('speed', example.speed)\n    return example.speed")
        result = self.execute("workers.map(speed, [0])")
        self.assertEqual(result["result"], [1.0])
        self.assertEqual(result["stdout"], "[worker 0] speed 1.0\n")
        self.execute("workers.broadcast('helper_value = 41')\nworkers.sync(TARGET=np.arange(3))")
        self.script.write_text(_SCRIPT.replace("self.speed = 1.0", "self.speed = 3.0"))
        rebuilt = self.session.dispatch("rebuild", {})
        self.assertEqual(rebuilt["workers_rebuilt"]["rebuilt"], 1)
        self.assertEqual(self.execute("workers.map(speed, [0])")["result"], [3.0])
        first_pid = self.pool.status()[0]["pid"]
        with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
            self.execute("import os\nworkers.submit(lambda: os._exit(3)).result()")
        result = self.execute("workers.map(lambda _: (helper_value, TARGET.tolist(), example.speed), [0])")
        self.assertEqual(result["result"], [[41, [0, 1, 2], 3.0]])
        self.assertEqual(len(result["workers"]), 1)
        self.assertRegex(result["workers"][0], r"worker 0 exited with code 3 .*restarted in .* replayed 2")
        self.assertFalse(_alive(first_pid))
        self.assertNotEqual(self.pool.status()[0]["pid"], first_pid)
        # A new worker replays the setup too; a removed one stops.
        self.assertEqual(self.execute("workers.resize(2)['count']")["result"], 2)
        result = self.execute(
            "import time\nsorted(set(workers.map(lambda _: (time.sleep(0.1), helper_value, os.getpid())[1:], range(6))))"
        )
        self.assertEqual(len(result["result"]), 2)
        self.assertEqual({row[0] for row in result["result"]}, {41})
        with self.assertRaisesRegex(RuntimeError, r"in \[0, 2\]"):
            self.execute("workers.resize(3)")
        self.assertEqual(self.execute("workers.resize(1)['count']")["result"], 1)
        self.assertEqual(self.session.dispatch("describe")["capabilities"]["workers"], 1)

    def test_failed_cuda_context_restarts_the_worker(self):
        """Restart a worker whose device check fails after a call and report it in the next response."""
        pid = self.pool.status()[0]["pid"]
        # Stand in for a failed CUDA context (see device_error) on this worker process only.
        with self.assertRaisesRegex(RuntimeError, r"CUDA context failed during this call \(cuda:0: simulated\)"):
            self.execute(
                "workers.broadcast('import os\\nimport newton._src.mcp.shipping as s\\n"
                f'if os.getpid() == {pid}:\\n    s.device_error = lambda: "cuda:0: simulated"\')'
            )
        self.pool.wait_ready(60)
        result = self.execute("workers.map(lambda x: x + 1, [1])")
        self.assertEqual(result["result"], [2])
        self.assertRegex(result["workers"][0], r"worker 0 CUDA context failed during broadcast code .*restarted")
        self.assertFalse(_alive(pid))

    def test_job_progress_lines(self):
        """Return the lines a running job printed so far, then its result."""
        self.execute(
            "def count(n):\n    for i in range(n):\n        print('step', i, flush=True)\n        time.sleep(0.25)\n"
            "    return n\nimport time"
        )
        result = self.execute("job = jobs.start(count, 6)\ntime.sleep(0.6)\njobs.wait(timeout=0.01)")["result"]
        self.assertEqual(result["finished"], [])
        early = result["running"][0]["lines"]
        self.assertGreaterEqual(len(early), 1)
        result = self.execute("jobs.wait(timeout=10)")["result"]
        self.assertEqual(result["finished"][0]["result"], 6)
        self.assertEqual(early + result["finished"][0]["lines"], [f"step {i}" for i in range(6)])


class TestMcpHostWorkers(unittest.TestCase):
    def test_host_command_line_launches_and_stops_workers(self):
        """Serve a script with --workers and stop its worker processes when the host is terminated."""
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "tiny.py"
            script.write_text(_SCRIPT)
            connection = Path(directory) / "session.json"
            host = subprocess.Popen(
                [
                    *(sys.executable, "-m", "newton.mcp", "host", str(script), "--connection-file", str(connection)),
                    *("--workers", "1", "--max-workers", "2"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                ready = connection.with_suffix(".ready")
                deadline = time.monotonic() + 120
                while not ready.exists() and host.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                self.assertTrue(ready.exists())
                client = SimulationClient(connection, timeout=60)
                result = client.request("execute", code="(workers.map(lambda x: x + 1, [1, 2]), workers.status())")
                values, status = result["result"]
                self.assertEqual(values, [2, 3])
                self.assertIn("workers.resize(n) sets the count (0 to 2)", client.request("guide")["guide"])
                pid = status[0]["pid"]
                self.assertTrue(_alive(pid))
                host.send_signal(signal.SIGTERM)
                host.wait(timeout=60)
                deadline = time.monotonic() + 15
                while _alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                self.assertFalse(_alive(pid))
                self.assertFalse((Path(directory) / "session.worker-0.json").exists())
                self.assertEqual(json.loads(ready.read_text())["pid"], host.pid)
            finally:
                if host.poll() is None:
                    host.kill()
                    host.wait()


if __name__ == "__main__":
    unittest.main()
