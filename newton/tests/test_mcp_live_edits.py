# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Failed-cell rollback, rebuild overrides, reset of step-advanced scalars, and automatic CUDA-graph recapture."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.mcp.host import _outermost, _restart_command
from newton._src.mcp.protocol import _Protocol
from newton.mcp import ExampleHost, SimulationClient, SimulationServer

_SCRIPT = textwrap.dedent(
    """
    import warp as wp

    import newton

    SUBSTEPS = 2
    PARAMS = {"speed": 1.0, "drive": {"kp": 10.0, "kd": 1.0}}
    START = (0.0, 0.0, 0.0)
    FAIL_AT = -1
    DEVICE = None


    @wp.kernel
    def push(body_qd: wp.array[wp.spatial_vector], speed: float, counter: wp.array[int]):
        body_qd[0] = wp.spatial_vector(wp.vec3(speed, 0.0, 0.0), wp.vec3(0.0))
        counter[0] = counter[0] + 1


    class Drive:
        def __init__(self):
            self.gain = 1.0


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body(xform=wp.transform(wp.vec3(*START), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1)
            self.model = builder.finalize(device=DEVICE)
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.speed = PARAMS["speed"]
            self.drive = Drive()
            self.counter = wp.zeros(1, dtype=int, device=self.model.device)
            self.ticks = 0
            self.frame_dt = 0.1
            self.graph = None
            if self.model.device.is_cuda:
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.simulate()
                self.graph = capture.graph

        def simulate(self):
            # SUBSTEPS is read while recording, so a new value needs a new CUDA graph.
            for _ in range(SUBSTEPS):
                wp.launch(push, 1, inputs=[self.state_0.body_qd, self.speed, self.counter], device=self.model.device)
                self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt / SUBSTEPS)
                self.state_0, self.state_1 = self.state_1, self.state_0

        def step(self):
            self.ticks += 1
            if self.ticks == FAIL_AT:
                raise RuntimeError("controller diverged")
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
    """
)


def _line(script: str, text: str) -> int:
    return next(number for number, line in enumerate(script.splitlines(), 1) if text in line)


class _HostedTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.script = Path(self.directory.name) / "live_edits.py"
        self.script.write_text(_SCRIPT)

    def host(self, **kwargs):
        host = ExampleHost(self.script, **kwargs)
        session = host.session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        return host, session

    @staticmethod
    def execute(session, code):
        return session.dispatch("execute", {"code": code})

    @staticmethod
    def advance(session):
        """Step one frame; return the body's x displacement [m] and the substeps it ran."""
        x0, count0 = float(session.state.body_q.numpy()[0, 0]), int(session.host.example.counter.numpy()[0])
        session.dispatch("step", {"count": 1})
        x1, count1 = float(session.state.body_q.numpy()[0, 0]), int(session.host.example.counter.numpy()[0])
        return x1 - x0, count1 - count0


class TestMcpRollback(_HostedTest):
    def test_failed_cell_restores_example_module_model_and_state(self):
        """Undo stepping, example and module edits, and model edits of a cell that raises."""
        host, session = self.host()
        self.execute(session, "rollout(2)")
        body_q = session.state.body_q.numpy().copy()
        mass = session.model.body_mass.numpy().copy()
        with self.assertRaises(RuntimeError) as raised:
            self.execute(
                session,
                "example.speed = 4.0\n"
                "module.SUBSTEPS = 4\n"
                "module.PARAMS['drive']['kp'] = 99.0\n"
                "rollout(3)\n"
                "model.body_mass.fill_(2.0)\n"
                "raise KeyError('late')",
            )
        message = str(raised.exception)
        self.assertIn("rolled back from t=0.5 s (frame 5) to t=0.2 s (frame 2)", message)
        for name in ("speed", "ticks", "SUBSTEPS", "PARAMS", "body_mass", "body_q"):
            self.assertIn(name, message)
        self.assertEqual((session.frame, host.example.ticks, host.example.speed), (2, 2, 1.0))
        self.assertEqual(host.module.SUBSTEPS, 2)
        self.assertEqual(host.module.PARAMS["drive"]["kp"], 10.0)
        self.assertEqual(int(host.example.counter.numpy()[0]), 4)
        np.testing.assert_array_equal(session.state.body_q.numpy(), body_q)
        np.testing.assert_array_equal(session.model.body_mass.numpy(), mass)
        self.assertTrue(session.valid)
        # The restored example (and its CUDA graph) advances with the original speed and substeps.
        displacement, substeps = self.advance(session)
        self.assertAlmostEqual(displacement, 0.1, places=4)
        self.assertEqual(substeps, 2)

    def test_error_in_script_step_rolls_back_and_names_script_line(self):
        """Report the hosted script's frame when its step() raises inside a rollout, and roll back."""
        host, session = self.host()
        with self.assertRaises(RuntimeError) as raised:
            self.execute(session, "module.FAIL_AT = 3\nrollout(5)")
        message = str(raised.exception)
        self.assertIn("controller diverged", message)
        self.assertIn(json.dumps(str(self.script.resolve()))[1:-1], message)
        self.assertIn(f'"line": {_line(_SCRIPT, "raise RuntimeError")}', message)
        self.assertIn("to t=0 s (frame 0)", message)
        self.assertEqual((session.frame, host.example.ticks, host.module.FAIL_AT), (0, 0, -1))
        self.assertEqual(self.execute(session, "rollout(4)['frames']")["result"], 4)

    def test_failed_cell_restores_rebound_class_attributes(self):
        """Undo methods a failed cell rebound on a script class, so rerunning the cell wraps the original.

        i15 (g1_mpc): the rollback restored ``example.controller`` but kept ``module.Controller.compute``
        rebound, so the retried cell wrapped its own patch and recursed.
        """
        self.script.write_text(
            _SCRIPT.replace(
                "        self.gain = 1.0\n",
                "        self.gain = 1.0\n\n    def output(self):\n        return self.gain\n",
            )
        )
        _, session = self.host()
        code = (
            "Drive = module.Drive\n"
            "_original = Drive.output\n"
            "def output(self):\n"
            "    return 2.0 * _original(self)\n"
            "Drive.output = output\n"
            "Drive.limit = 3.0\n"
            "patched = example.drive.output()\n"
            "raise KeyError('after the patch')"
        )
        with self.assertRaises(RuntimeError) as raised:
            self.execute(session, code)
        message = str(raised.exception)
        self.assertIn("Drive.output", message)
        self.assertIn("Drive.limit", message)
        result = self.execute(session, "patched, example.drive.output(), hasattr(module.Drive, 'limit')")
        self.assertEqual(result["result"], [2.0, 1.0, False])
        with self.assertRaises(RuntimeError) as raised:
            self.execute(session, code)
        self.assertIn("KeyError", str(raised.exception))
        self.assertNotIn("RecursionError", str(raised.exception))

    def test_cell_without_simulation_changes_reports_nothing_restored(self):
        """Keep the hidden solver state and say so when a failing cell changed nothing."""
        _, session = self.host()
        self.execute(session, "rollout(2)")
        with self.assertRaisesRegex(RuntimeError, "nothing was restored"):
            self.execute(session, "values = [state.body_q.numpy()[0, 0]]\nvalues[3]")
        self.assertEqual(self.execute(session, "len(values), session.frame")["result"], [1, 2])


class TestMcpRebuild(_HostedTest):
    def test_failed_rebuild_keeps_previous_scene_and_reports_script_line(self):
        """Keep serving the previous scene when the edited script fails, with file:line in the error."""
        host, session = self.host()
        self.execute(session, "rollout(2)")
        example = host.example
        broken = _SCRIPT.replace("        self.solver = ", "        1 / 0\n        self.solver = ")
        self.script.write_text(broken)
        with self.assertRaises(RuntimeError) as raised:
            session.dispatch("rebuild", {})
        message = str(raised.exception)
        self.assertIn("previous scene keeps running", message)
        self.assertIn("ZeroDivisionError", message)
        self.assertIn(f"{self.script.resolve()}:{_line(broken, '1 / 0')} in __init__", message)
        self.assertIs(host.example, example)
        self.assertTrue(session.valid)
        self.assertEqual(session.dispatch("step")["frame"], 3)
        self.script.write_text(_SCRIPT.replace("FAIL_AT = -1", "FAIL_AT = (-1"))
        with self.assertRaisesRegex(
            RuntimeError, f"(?s)SyntaxError.*{self.script.resolve()}:{_line(_SCRIPT, 'FAIL_AT')}"
        ):
            session.dispatch("rebuild", {})
        self.assertEqual(session.dispatch("step")["frame"], 4)
        self.script.write_text(_SCRIPT)
        self.assertEqual(session.dispatch("rebuild", {})["frame"], 0)

    def test_overrides_set_module_globals_and_are_echoed_until_cleared(self):
        """Merge dict overrides, replace scalars, keep script types, and echo the active set."""
        host, session = self.host()
        overrides = {"SUBSTEPS": 4, "PARAMS": {"speed": 2, "drive": {"kp": 50}}, "START": [1, 0, 0]}
        result = session.dispatch("rebuild", {"overrides": overrides})
        self.assertEqual(result["overrides"], overrides)
        module = host.module
        self.assertEqual(module.SUBSTEPS, 4)
        self.assertEqual(module.PARAMS, {"speed": 2.0, "drive": {"kp": 50.0, "kd": 1.0}})
        self.assertIsInstance(module.PARAMS["drive"]["kp"], float)
        self.assertEqual(module.START, (1, 0, 0))
        self.assertAlmostEqual(float(session.state.body_q.numpy()[0, 0]), 1.0)
        self.assertEqual(self.execute(session, "example.speed")["overrides"], overrides)
        displacement, substeps = self.advance(session)
        self.assertAlmostEqual(displacement, 0.2, places=4)
        self.assertEqual(substeps, 4)
        # A plain rebuild keeps the active overrides.
        self.assertEqual(session.dispatch("rebuild", {})["overrides"], overrides)
        self.assertEqual(host.module.SUBSTEPS, 4)
        # An unknown name fails before Example() and keeps the scene and the active overrides.
        example = host.example
        with self.assertRaisesRegex(RuntimeError, "no module global SUBSTEP; its data globals are .*SUBSTEPS"):
            session.dispatch("rebuild", {"overrides": {"SUBSTEP": 8}})
        self.assertIs(host.example, example)
        self.assertEqual(session.dispatch("describe")["overrides"], overrides)
        cleared = session.dispatch("rebuild", {"overrides": {}})
        self.assertNotIn("overrides", cleared)
        self.assertEqual(host.module.SUBSTEPS, 2)
        self.assertEqual(self.advance(session)[1], 2)

    def test_errors_sent_to_clients_carry_active_overrides(self):
        """Append the active overrides to error responses that cross the transport."""
        _, session = self.host(overrides={"SUBSTEPS": 4})
        path = Path(self.directory.name) / "session.json"
        errors = []

        def call():
            try:
                SimulationClient(path, timeout=10).request("execute", code="1 / 0")
            except Exception as error:
                errors.append(str(error))

        with SimulationServer(session, connection_file=path):
            thread = threading.Thread(target=call)
            thread.start()
            deadline = time.monotonic() + 30
            while thread.is_alive() and time.monotonic() < deadline:
                session.pump()
                time.sleep(0.002)
            thread.join(timeout=1)
        self.assertEqual(len(errors), 1)
        self.assertIn('{"overrides":{"SUBSTEPS":4}}', errors[0])

    def test_protocol_forwards_rebuild_overrides(self):
        """Accept overrides as a top-level newton_rebuild argument and pass them to the rebuild callback."""
        calls = []

        class Client:
            def request(self, operation, **arguments):
                calls.append((operation, arguments))
                return {"frame": 0}

        protocol = _Protocol(Client(), profile="lean", app_guide=False)
        protocol.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        tools = protocol.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        rebuild = next(tool for tool in tools if tool["name"] == "newton_rebuild")
        self.assertIn("overrides", rebuild["inputSchema"]["properties"])
        execute = next(tool for tool in tools if tool["name"] == "newton_execute")
        self.assertNotIn("recovery", execute["inputSchema"]["properties"])
        protocol.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "newton_rebuild", "arguments": {"overrides": {"SUBSTEPS": 4}}},
            }
        )
        self.assertEqual(calls[-1], ("rebuild", {"reset_namespace": False, "overrides": {"SUBSTEPS": 4}}))

    def test_restart_keeps_overrides_on_the_command_line(self):
        """Carry the active overrides into the re-executed host command and its workers."""
        argv = ["push.py", "--connection-file", "c.json", "--overrides", '{"A": 1}', "--", "--seed", "3"]
        self.assertEqual(
            _restart_command(argv, {"B": 2}, ["--seed", "3"]),
            ["push.py", "--connection-file", "c.json", "--overrides", '{"B": 2}', "--", "--seed", "3"],
        )
        self.assertEqual(
            _restart_command(argv, {}, ["--seed", "4"]),
            ["push.py", "--connection-file", "c.json", "--", "--seed", "4"],
        )
        fallback = {"argv": ["--seed", "3"], "overrides": {"A": 1}}
        argv_with_fallback = [*argv[:5], "--restart-fallback", "{}", "--", "--seed", "3"]
        self.assertEqual(
            _restart_command(argv_with_fallback, {"B": 2}, ["--seed", "4"], fallback=fallback),
            [
                *("push.py", "--connection-file", "c.json", "--overrides", '{"B": 2}'),
                *("--restart-fallback", json.dumps(fallback), "--", "--seed", "4"),
            ],
        )
        host, session = self.host()
        session.dispatch("rebuild", {"restart": True, "overrides": {"SUBSTEPS": 4}, "argv": ["--seed", "5"]})
        self.assertTrue(host.restart_requested)
        self.assertEqual(host.overrides, {"SUBSTEPS": 4})
        self.assertEqual(host.argv, ["--seed", "5"])
        self.assertEqual(host.restart_fallback, {"argv": [], "overrides": {}})
        with self.assertRaisesRegex(RuntimeError, "unexpected keyword argument 'override_set'"):
            session.dispatch("rebuild", {"override_set": {"SUBSTEPS": 4}})

    def test_restart_refuses_inputs_the_new_process_cannot_load(self):
        """Check overrides, arguments and the script in a new process before the host re-executes itself."""
        host, session = self.host()
        with self.assertRaisesRegex(RuntimeError, "(?s)previous scene keeps running.*no module global NOPE"):
            session.dispatch("rebuild", {"restart": True, "overrides": {"NOPE": 1}})
        with self.assertRaisesRegex(RuntimeError, "invalid int value: 'abc'"):
            session.dispatch("rebuild", {"restart": True, "argv": ["--num-frames", "abc"]})
        self.script.write_text(_SCRIPT.replace("FAIL_AT = -1", "FAIL_AT = (-1"))
        with self.assertRaisesRegex(RuntimeError, "SyntaxError"):
            session.dispatch("rebuild", {"restart": True})
        self.assertFalse(host.restart_requested)
        self.assertEqual((host.argv, host.overrides), ([], {}))
        self.assertTrue(session.valid)
        self.assertEqual(session.dispatch("step")["frame"], 1)

    def test_restarted_host_builds_the_previous_inputs_when_the_new_ones_fail(self):
        """Build with the fallback arguments when Example() fails with the requested ones, and say so."""
        connection = Path(self.directory.name) / "restart.json"
        overrides = {"FAIL_AT": 1, "PARAMS": {"speed": 0}, "DEVICE": "cpu"}
        script = self.script.with_name("live_edits_restart.py")
        # The requested overrides make Example() raise; the fallback ones build.
        script.write_text(_SCRIPT.replace("self.ticks = 0", "self.ticks = 1 / PARAMS['speed']"))
        command = [
            *(sys.executable, "-m", "newton.mcp", "host", str(script), "--connection-file", str(connection)),
            *("--overrides", json.dumps(overrides)),
            *("--restart-fallback", json.dumps({"argv": [], "overrides": {"DEVICE": "cpu", "SUBSTEPS": 3}})),
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(process.wait, 30)
        self.addCleanup(process.terminate)
        deadline = time.monotonic() + 120
        while not connection.with_suffix(".ready").exists():
            self.assertIsNone(process.poll(), "the host exited instead of building the fallback")
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.1)
        result = SimulationClient(connection, timeout=60).request("execute", code="module.SUBSTEPS")
        self.assertEqual(result["result"], 3)
        self.assertEqual(result["overrides"], {"DEVICE": "cpu", "SUBSTEPS": 3})
        self.assertIn("The restart could not build", result["note"])
        self.assertIn("ZeroDivisionError", result["note"])

    def test_argument_errors_and_sys_exit_keep_the_host_running(self):
        """Report argparse errors and SystemExit from cells and stepping instead of ending the process."""
        host, session = self.host()
        example = host.example
        with self.assertRaisesRegex(RuntimeError, "(?s)previous scene keeps running.*rejected \\['--num-frames'"):
            session.dispatch("rebuild", {"argv": ["--num-frames", "abc"]})
        self.assertIs(host.example, example)
        with self.assertRaisesRegex(RuntimeError, r"SystemExit\(3\) raised at line 2; the session keeps running"):
            self.execute(session, "rollout(2)\nraise SystemExit(3)")
        self.assertEqual(session.frame, 0)
        host.example.step = lambda: sys.exit(4)
        with self.assertRaisesRegex(RuntimeError, r"SystemExit\(4\)"):
            session.dispatch("step")
        del host.example.step
        self.assertTrue(session.valid)
        self.assertEqual(session.dispatch("step")["frame"], 1)

    def test_rebuild_imports_edited_helper_modules_again(self):
        """Reload modules imported from the script's directory, and restore them when the rebuild fails."""
        helper = f"live_edits_helper_{os.getpid()}_{id(self)}"
        self.addCleanup(sys.modules.pop, helper, None)
        directory = Path(self.directory.name)
        (directory / f"{helper}.py").write_text("GAIN = 1.0\n")
        script = directory / "uses_helper.py"
        script.write_text(
            f"from {helper} import GAIN\n" + _SCRIPT.replace('self.speed = PARAMS["speed"]', "self.speed = GAIN")
        )
        host = ExampleHost(script)
        session = host.session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        self.assertEqual(host.example.speed, 1.0)
        (directory / f"{helper}.py").write_text("GAIN = 2.5\n")
        session.dispatch("rebuild", {})
        self.assertEqual(host.example.speed, 2.5)
        loaded = sys.modules[helper]
        (directory / f"{helper}.py").write_text("GAIN = 4.0\n1 / 0\n")
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
            session.dispatch("rebuild", {})
        self.assertIs(sys.modules[helper], loaded)
        self.assertEqual(host.example.speed, 2.5)

    def test_workers_follow_overrides(self):
        """Rebuild hosted workers with the main session's overrides and arguments."""
        paths, stops, threads = [], [], []
        ready = threading.Barrier(2)
        # One worker thread: concurrent model construction in threads races on Warp's default device.
        device = wp.get_device()
        self.addCleanup(wp.set_device, device)

        def worker(index, path):
            # Separate script files keep the workers' Warp modules apart within this test process, and CPU
            # devices keep their threads from recording CUDA graphs concurrently.
            script = Path(self.directory.name) / f"live_edits_worker_{index}.py"
            script.write_text(_SCRIPT.replace("DEVICE = None", 'DEVICE = "cpu"'))
            session = ExampleHost(script).session(artifact_directory=self.directory.name)
            stop = threading.Event()
            stops.append(stop)
            with SimulationServer(session, connection_file=path):
                ready.wait()
                while not stop.is_set():
                    session.pump()
                    time.sleep(0.001)
            session.close()

        for index in range(1):
            paths.append(Path(self.directory.name) / f"worker-{index}.json")
            threads.append(threading.Thread(target=worker, args=(index, paths[-1]), daemon=True))
            threads[-1].start()
        ready.wait(timeout=120)

        def stop_workers():
            for stop in stops:
                stop.set()
            for thread in threads:
                thread.join(timeout=10)

        self.addCleanup(stop_workers)
        host = ExampleHost(self.script)
        session = host.session(artifact_directory=self.directory.name, workers=paths)
        self.addCleanup(session.close)
        result = session.dispatch("rebuild", {"overrides": {"SUBSTEPS": 4}})
        self.assertEqual(result["workers_rebuild"]["pending"], [0])
        # The worker rebuilds in the background, before this call.
        self.assertEqual(self.execute(session, "workers.broadcast('module.SUBSTEPS')")["result"], [4])
        session.dispatch("rebuild", {"overrides": {}})
        self.assertEqual(self.execute(session, "workers.broadcast('module.SUBSTEPS')")["result"], [2])


class TestMcpReset(_HostedTest):
    def setUp(self):
        super().setUp()
        timed = _SCRIPT.replace("        self.ticks = 0\n", "        self.ticks = 0\n        self.sim_time = 0.0\n")
        timed = timed.replace(
            "        self.ticks += 1\n", "        self.ticks += 1\n        self.sim_time += self.frame_dt\n"
        )
        self.script.write_text(timed)

    def test_reset_rewinds_scalars_advanced_by_direct_example_steps(self):
        """Rewind timers that cells advanced through example.step() (i15 g1_mpc: records started at 1.01 s)."""
        host, session = self.host()
        session.dispatch("rebuild", {})
        self.execute(session, "for _ in range(10):\n    example.step()")
        self.assertAlmostEqual(host.example.sim_time, 1.0)
        code = (
            "example.speed = 2.0\n"
            "session.dispatch('reset', {})\n"
            "records = []\n"
            "for _ in range(3):\n"
            "    example.step()\n"
            "    records.append(round(example.sim_time, 2))\n"
            "records, example.ticks, example.speed"
        )
        self.assertEqual(self.execute(session, code)["result"], [[0.1, 0.2, 0.3], 3, 2.0])
        code = (
            "session.dispatch('checkpoint', {'name': 'three'})\n"
            "for _ in range(4):\n"
            "    example.step()\n"
            "session.dispatch('restore', {'name': 'three'})\n"
            "round(example.sim_time, 2), example.ticks"
        )
        self.assertEqual(self.execute(session, code)["result"], [0.3, 3])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graphs need a CUDA device")
    def test_direct_steps_do_not_recapture_graphs(self):
        """Treat timers that example.step() advances as state, not as settings baked into the CUDA graph."""
        host, session = self.host(overrides={"DEVICE": "cuda:0"})
        self.assertIsNotNone(host.example.graph)
        self.execute(session, "for _ in range(3):\n    example.step()")
        result = self.execute(session, "rollout(2)\nsession.dispatch('reset', {})\nexample.step()")
        self.assertNotIn("note", result)
        self.assertEqual(host.recaptures, 0)


class TestMcpRecapture(_HostedTest):
    def test_notes_name_only_the_outermost_changed_settings(self):
        """Name a replaced object once, not every setting below it (i15: 'example.model, ..., 502 more')."""
        keys = ["example.model", "example.model.actuators", "example.model.mujoco.solref", "example.solver.iterations"]
        self.assertEqual(_outermost(keys), ["example.model", "example.solver.iterations"])

    def test_fingerprint_covers_module_globals_and_solver_settings(self):
        """Track module globals, nested plain data, solver settings, and the script's own objects."""
        host, _ = self.host()
        before = host.fingerprint()
        host.module.SUBSTEPS = 4
        host.module.PARAMS["drive"]["kp"] = 5.0
        host.example.solver.angular_damping = 0.5
        host.example.drive.gain = 2.0
        after = host.fingerprint()
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        expected = {"module.SUBSTEPS", "module.PARAMS", "example.solver.angular_damping", "example.drive.gain"}
        self.assertEqual(changed, expected)
        self.assertNotIn("module.SUBSTEPS", host.fingerprint(deep=False))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graphs need a CUDA device")
    def test_graphs_follow_module_globals_and_solver_settings(self):
        """Re-record graphs for module-global, solver-setting, and solver edits but not for rewinds."""
        # The override pins the scene to CUDA; graphs are re-recorded on their own device even while
        # Warp's default device is the CPU.
        host, session = self.host(overrides={"DEVICE": "cuda:0"})
        self.assertIsNotNone(host.example.graph)
        self.assertEqual(self.advance(session)[1], 2)
        with wp.ScopedDevice("cpu"):
            note = self.execute(session, "module.SUBSTEPS = 4")["note"]
        self.assertEqual(note, "CUDA graphs recaptured after changes to module.SUBSTEPS")
        self.assertEqual(self.advance(session)[1], 4)
        note = self.execute(session, "example.solver.angular_damping = 0.5")["note"]
        self.assertIn("example.solver.angular_damping", note)
        # Rewinding timers and stepping again re-uses the graph.
        result = self.execute(session, "rollout(3, start=True)\nrollout(3, start=True)")
        self.assertNotIn("note", result)
        recaptures = host.recaptures
        self.execute(session, "session.dispatch('checkpoint', {'name': 'a'})\nrollout(2)\nrollout(2, start='a')")
        self.assertEqual(host.recaptures, recaptures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
