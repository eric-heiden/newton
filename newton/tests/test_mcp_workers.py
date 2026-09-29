# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import threading
import time
import unittest
from pathlib import Path

import warp as wp

import newton
from newton.mcp import SimulationServer, SimulationSession


def _scene():
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.1)
    return builder.finalize(device="cpu")


class TestMcpWorkers(unittest.TestCase):
    def setUp(self):
        """Start two worker sessions, each pumped by its own owner thread."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = [Path(self.directory.name) / f"worker-{i}.json" for i in range(2)]
        self.stops = []
        ready = threading.Barrier(3)

        def worker(path):
            model = _scene()
            session = SimulationSession(model, newton.solvers.SolverXPBD(model), allow_execute=True)
            stop = threading.Event()
            self.stops.append(stop)
            with SimulationServer(session, connection_file=path):
                ready.wait()
                while not stop.is_set():
                    session.pump()
                    time.sleep(0.001)
            session.close()

        self.threads = [threading.Thread(target=worker, args=(path,), daemon=True) for path in self.paths]
        for thread in self.threads:
            thread.start()
        ready.wait(timeout=60)
        model = _scene()
        self.session = SimulationSession(
            model, newton.solvers.SolverXPBD(model), allow_execute=True, workers=self.paths
        )
        self.addCleanup(self.session.close)
        self.addCleanup(self._stop)

    def _stop(self):
        for stop in self.stops:
            stop.set()
        for thread in self.threads:
            thread.join(timeout=10)

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


if __name__ == "__main__":
    unittest.main()
