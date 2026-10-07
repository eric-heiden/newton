# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tool calls within client limits: early replies for long cells, bounded waits, and the response budget."""

import base64
import concurrent.futures
import json
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

import numpy as np

import newton
from newton._src.mcp.imaging import encode_png, fit_images, to_rgb
from newton._src.mcp.jobs import JobQueue
from newton._src.mcp.protocol import _Protocol
from newton.mcp import ExampleHost, SimulationClient, SimulationServer, SimulationSession


class TestMcpLongCalls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        self.session = SimulationSession(
            model,
            newton.solvers.SolverSemiImplicit(model),
            allow_execute=True,
            artifact_directory=self.directory.name,
        )
        self.addCleanup(self.session.close)

    def serve(self, worker) -> None:
        """Pump the session on this thread while ``worker(connection_file)`` makes requests on another."""
        path = Path(self.directory.name) / "session.json"
        errors, done = [], threading.Event()

        def run():
            try:
                worker(path)
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        with SimulationServer(self.session, connection_file=path):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 60.0
            while not done.is_set() and time.monotonic() < deadline:
                self.session.pump()
                time.sleep(0.005)
            thread.join(timeout=5.0)
        if errors:
            raise errors[0]

    def test_long_cell_replies_early_and_its_result_arrives_later(self):
        """Reply at the limit with the output so far; the next call waits for the cell and carries its result."""
        replies = {}

        def worker(path):
            client = SimulationClient(path, reply_within=1.0)
            started = time.monotonic()
            replies["first"] = client.request(
                "execute", code="import time\nprint('started')\ntime.sleep(1.6)\nprint('finished')\n41 + 1"
            )
            replies["seconds"] = time.monotonic() - started
            replies["second"] = client.request("execute", code="'next'")

        self.serve(worker)
        running = replies["first"]["running"]
        self.assertLess(replies["seconds"], 1.5)
        self.assertEqual(running["operation"], "execute")
        self.assertEqual(running["stdout"], "started\n")
        self.assertIn("still running", replies["first"]["note"])
        second = replies["second"]
        self.assertEqual(second["result"], "next")
        [finished] = second["finished_calls"]
        self.assertEqual(finished["call"], running["call"])
        self.assertEqual(finished["result"], 42)
        # Only the output printed after the early reply.
        self.assertEqual(finished["stdout"], "finished\n")

    def test_call_queued_behind_a_running_cell_is_not_run(self):
        """A call that cannot start before its reply limit is refused, naming the running call and its output."""
        replies, errors = {}, {}

        def worker(path):
            client = SimulationClient(path, reply_within=0.5)
            replies["first"] = client.request(
                "execute", code="import time\nprint('working')\ntime.sleep(2.0)\nvalue = 7\nvalue"
            )
            try:
                client.request("execute", code="value = 0")
            except TimeoutError as error:
                errors["second"] = str(error)
            time.sleep(1.5)
            replies["third"] = client.request("execute", code="value")

        self.serve(worker)
        self.assertIn("running", replies["first"])
        self.assertRegex(errors["second"], r"Not run: call \d+ \(execute\) is still running")
        third = replies["third"]
        # The refused call never ran.
        self.assertEqual(third["result"], 7)
        self.assertEqual([entry["result"] for entry in third["finished_calls"]], [7])

    def test_failed_background_cell_reports_its_error(self):
        """A detached cell that raises reports the error and its rollback in a later response."""
        replies = {}

        def worker(path):
            client = SimulationClient(path, reply_within=0.5)
            replies["first"] = client.request("execute", code="import time\ntime.sleep(0.9)\nraise ValueError('late')")
            replies["second"] = client.request("execute", code="1")

        self.serve(worker)
        [finished] = replies["second"]["finished_calls"]
        self.assertIn("ValueError", finished["error"])
        self.assertIn("late", finished["error"])

    def test_job_waits_end_before_the_reply_limit(self):
        """jobs.result() and jobs.wait() without a timeout return shortly before the running call must reply."""
        never = concurrent.futures.Future()
        jobs = JobQueue(seconds_left=lambda: JobQueue.reply_margin + 0.2)
        jobs.register_backend("test", lambda function, args, kwargs, progress: never)
        job = jobs.start(lambda: None, where="test")
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, rf"Job {job} is still running"):
            jobs.result(job)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(jobs.wait()["running"][0]["id"], job)
        never.set_result(5)
        self.assertEqual(jobs.result(job), 5)
        # Without a reply limit, a timeout still applies when given.
        unlimited = JobQueue()
        unlimited.register_backend("test", lambda function, args, kwargs, progress: concurrent.futures.Future())
        with self.assertRaises(TimeoutError):
            unlimited.result(unlimited.start(lambda: None, where="test"), timeout=0.05)


class _Client:
    reply_within = None

    def __init__(self, response):
        self.response = response

    def request(self, operation, **arguments):
        return dict(self.response)


def _call(protocol, name="newton_execute", arguments=None):
    protocol.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    message = {"name": name, "arguments": arguments or {"code": "1"}}
    return protocol.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": message})["result"]


class TestMcpResponseBudget(unittest.TestCase):
    def test_images_shrink_to_the_response_budget(self):
        """Re-encode, downscale and finally drop images so a result stays within its character budget."""
        rng = np.random.default_rng(0)
        noise = [rng.integers(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(3)]
        images = [
            {"image_base64": base64.b64encode(encode_png(rgb)).decode(), "mime_type": "image/png"} for rgb in noise
        ]
        protocol = _Protocol(_Client({"result": 1, "images": images}), profile="lean", response_budget=400_000)
        result = _call(protocol)
        size = len(json.dumps(result["content"], separators=(",", ":")))
        self.assertLessEqual(size, 400_000)
        text = json.loads(result["content"][0]["text"])
        self.assertIn("response budget", text["images_note"])
        self.assertGreaterEqual(len(result["content"]), 2)
        shown = to_rgb(base64.b64decode(result["content"][1]["data"]))
        self.assertEqual(shown.ndim, 3)
        # Small results are untouched.
        small = [{"image_base64": base64.b64encode(encode_png(noise[0][:32, :32])).decode(), "mime_type": "image/png"}]
        result = _call(_Protocol(_Client({"images": small}), profile="lean"))
        self.assertEqual(result["content"][1]["data"], small[0]["image_base64"])
        self.assertNotIn("images_note", json.loads(result["content"][0]["text"]))

    def test_fit_images_drops_what_cannot_shrink(self):
        """Images that stay over the budget at the smallest size are dropped and counted."""
        rgb = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        data = base64.b64encode(encode_png(rgb)).decode()
        images, note = fit_images([(data, "image/png"), (data, "image/png")], 10)
        self.assertEqual(images, [])
        self.assertIn("dropped", note)


_SCRIPT = textwrap.dedent(
    """
    import newton


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder()
            builder.add_shape_sphere(builder.add_body(), radius=0.1)
            self.model = builder.finalize()
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.frame_dt = 0.01
            self.marker = MARKER

        def step(self):
            self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    """
)


class TestMcpRebuildCode(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.script = Path(directory.name) / "marker.py"
        self.script.write_text("MARKER = 1\n" + _SCRIPT)
        self.session = ExampleHost(self.script).session(artifact_directory=directory.name)
        self.addCleanup(self.session.close)

    def test_rebuild_runs_a_cell_afterwards(self):
        """newton_rebuild(code=...) returns the rebuild counts and the cell's value and output."""
        self.script.write_text("MARKER = 2\n" + _SCRIPT)
        result = self.session.dispatch("rebuild", {"code": "print('built')\nexample.marker"})
        self.assertEqual(result["result"], 2)
        self.assertEqual(result["stdout"], "built\n")
        self.assertEqual(result["counts"]["body_count"], 1)
        with self.assertRaisesRegex(RuntimeError, "The rebuild succeeded .*then the cell failed.*NameError"):
            self.session.dispatch("rebuild", {"code": "undefined_name"})
        self.script.write_text("MARKER = undefined_name\n" + _SCRIPT)
        with self.assertRaisesRegex(RuntimeError, "Rebuild failed"):
            self.session.dispatch("rebuild", {"code": "ran = True"})
        self.assertNotIn("ran", self.session._workspace)
        with self.assertRaisesRegex(ValueError, "cannot follow a restart"):
            self.session.dispatch("rebuild", {"code": "1", "restart": True})

    def test_stale_hosted_script_is_flagged(self):
        """Say once that the script changed on disk after the build, and in errors until the next rebuild."""
        self.script.write_text("MARKER = 3\n" + _SCRIPT + "\n\nclass Planner:\n    pass\n")
        result = self.session.dispatch("execute", {"code": "module.MARKER"})
        self.assertEqual(result["result"], 1)
        self.assertIn("marker.py changed on disk after the last build", result["note"])
        self.assertNotIn("note", self.session.dispatch("execute", {"code": "module.MARKER"}))
        with self.assertRaisesRegex(RuntimeError, "AttributeError.*marker.py changed on disk"):
            self.session.dispatch("execute", {"code": "module.Planner"})
        self.session.dispatch("rebuild", {})
        self.assertNotIn("note", self.session.dispatch("execute", {"code": "module.Planner"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
