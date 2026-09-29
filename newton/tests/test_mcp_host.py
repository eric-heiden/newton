# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise hosted example scripts and the trusted-execution helpers."""

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.mcp import ExampleHost, SimulationSession

_SCRIPT = textwrap.dedent(
    """
    import warp as wp

    import newton


    @wp.kernel
    def push(body_qd: wp.array[wp.spatial_vector], speed: float):
        body_qd[0] = wp.spatial_vector(wp.vec3(speed, 0.0, 0.0), wp.vec3(0.0))


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body()
            builder.add_shape_sphere(body, radius=0.1)
            self.model = builder.finalize()
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.speed = 1.0
            self.ticks = 0
            self.frame_dt = 0.1
            self.graph = None
            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

        def simulate(self):
            for _ in range(2):
                wp.launch(push, 1, inputs=[self.state_0.body_qd, self.speed])
                self.solver.step(self.state_0, self.state_1, self.control, None, 0.5 * self.frame_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

        def step(self):
            self.ticks += 1
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
    """
)


class TestMcpHost(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.script = Path(self.directory.name) / "push.py"
        self.script.write_text(_SCRIPT)

    def test_hosted_example_graph_edits_are_reported_or_recaptured(self):
        """Warn when a captured value changes; recapture once the example gets a new solver."""
        host = ExampleHost(self.script)
        session = host.session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        result = session.dispatch("execute", {"code": "rollout(1)['t'].tolist()"})
        self.assertEqual(result["result"], [0.0, 0.1])
        captured = host.example.graph is not None

        def advance():
            x0 = float(session.state.body_q.numpy()[0, 0])
            session.dispatch("step", {"count": 1})
            return float(session.state.body_q.numpy()[0, 0]) - x0

        result = session.dispatch("execute", {"code": "example.speed = 2.0"})
        if captured:
            # The speed is a kernel argument inside the captured graph, so it cannot change yet.
            self.assertIn("newton_rebuild", result["note"])
            self.assertAlmostEqual(advance(), 0.1, places=4)
            with self.assertRaisesRegex(RuntimeError, "same solver"):
                session.dispatch("execute", {"code": "recapture()"})
            result = session.dispatch(
                "execute", {"code": "example.solver = newton.solvers.SolverSemiImplicit(example.model)"}
            )
            self.assertIn("recaptured", result["note"])
        self.assertAlmostEqual(advance(), 0.2, places=4)

    def test_reset_rewinds_step_state_but_keeps_assigned_settings(self):
        """Rewind scalars that step() advances while keeping settings the agent assigned."""
        host = ExampleHost(self.script)
        session = host.session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        session.dispatch("step", {"count": 3})
        session.dispatch("execute", {"code": "example.speed = 2.0"})
        session.dispatch("reset", {})
        self.assertEqual(session.dispatch("execute", {"code": "(example.ticks, example.speed)"})["result"], [0, 2.0])

    def test_hosted_errors_keep_scene_valid_and_rebuild_reloads_script(self):
        """Report Python errors without invalidating, and reload the edited script in place."""
        host = ExampleHost(self.script)
        session = host.session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        with self.assertRaisesRegex(RuntimeError, "stays valid"):
            session.dispatch("execute", {"code": "kept = 3\nmissing_name"})
        self.assertTrue(session.valid)
        self.assertEqual(session.dispatch("execute", {"code": "kept"})["result"], 3)
        self.script.write_text(_SCRIPT.replace("self.speed = 1.0", "self.speed = 3.0"))
        session.dispatch("rebuild", {})
        self.assertEqual(session.dispatch("execute", {"code": "(example.speed, kept)"})["result"], [3.0, 3])


class TestMcpHelpers(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        self.session = SimulationSession(
            model,
            newton.solvers.SolverXPBD(model),
            dt=0.01,
            allow_execute=True,
            artifact_directory=self.directory.name,
        )
        self.addCleanup(self.session.close)

    def test_rollout_records_series_and_stops_early(self):
        """Sample expressions and callables, reset first, and stop on a condition."""
        result = self.session.rollout(
            20,
            record={
                "z": "state.body_q.numpy()[0, 2]",
                "vz": lambda s: s.state.body_qd.numpy()[0, 2],
                "frame": lambda: self.session.frame,
            },
            every=5,
        )
        self.assertEqual(result["frames"], 20)
        np.testing.assert_allclose(result["t"], [0.0, 0.05, 0.1, 0.15, 0.2], atol=1e-9)
        self.assertLess(result["z"][-1], result["z"][0])
        self.assertLess(result["vz"][-1], 0.0)
        self.assertEqual(result["frame"].tolist(), [0, 5, 10, 15, 20])
        result = self.session.rollout(
            seconds=1.0, start=True, record={"z": "state.body_q.numpy()[0, 2]"}, until="session.frame >= 10"
        )
        self.assertEqual(result["frames"], 10)
        self.assertIsNotNone(result["stopped"])
        self.assertAlmostEqual(result["z"][0], 0.5, places=5)

    def test_health_and_contact_report(self):
        """Report resting contacts per shape pair and flag non-finite state."""
        self.session.rollout(seconds=1.5)
        self.session._collide()
        report = self.session.solver_contacts()
        self.assertEqual(report["source"], "newton")
        self.assertGreater(report["count"], 0)
        self.assertIn("ke", report["pairs"][0]["shapes"][0])
        self.assertTrue(self.session.health()["ok"])
        body_q = self.session.state.body_q.numpy()
        body_q[0, 2] = np.nan
        self.session.state.body_q.assign(body_q)
        self.assertFalse(self.session.health()["ok"])

    def test_workspace_preloads_newton_and_helpers(self):
        """Provide newton and the helper functions without imports."""
        result = self.session.dispatch("execute", {"code": "(newton.__name__, callable(rollout), health()['ok'])"})
        self.assertEqual(result["result"], ["newton", True, True])


class TestMcpLeanProfile(unittest.TestCase):
    def test_lean_profile_lists_execute_and_rebuild(self):
        """Advertise two tools and short instructions that point to Python-side observation."""
        from newton._src.mcp.protocol import _Protocol  # noqa: PLC0415

        class Client:
            def request(self, operation, **_):
                return {"guide": "guide-marker"} if operation == "guide" else {}

        protocol = _Protocol(Client(), profile="lean")
        initialized = protocol.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        instructions = initialized["result"]["instructions"]
        self.assertIn("session.dispatch('observe'", instructions)
        self.assertTrue(instructions.endswith("guide-marker"))
        bare = _Protocol(Client(), profile="lean", app_guide=False)
        instructions = bare.handle({"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}})["result"]
        self.assertNotIn("guide-marker", instructions["instructions"])
        listed = protocol.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual([tool["name"] for tool in listed["result"]["tools"]], ["newton_execute", "newton_rebuild"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
