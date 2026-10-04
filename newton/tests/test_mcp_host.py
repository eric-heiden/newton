# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise hosted example scripts and the trusted-execution helpers."""

import base64
import tempfile
import textwrap
import threading
import time
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

    def test_hosted_example_recaptures_graphs_after_attribute_edits(self):
        """Re-record the example's CUDA graph when a cell changes a kernel argument baked into it."""
        host = ExampleHost(self.script)
        session = host.session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        result = session.dispatch("execute", {"code": "rollout(1)['t'].tolist()"})
        self.assertEqual(result["result"], [0.0, 0.1])

        def advance():
            x0 = float(session.state.body_q.numpy()[0, 0])
            session.dispatch("step", {"count": 1})
            return float(session.state.body_q.numpy()[0, 0]) - x0

        self.assertAlmostEqual(advance(), 0.1, places=4)
        # The speed is a kernel argument inside the captured graph.
        result = session.dispatch("execute", {"code": "example.speed = 2.0"})
        if host.example.graph is not None:
            self.assertIn("recaptured", result["note"])
        self.assertAlmostEqual(advance(), 0.2, places=4)
        session.dispatch("execute", {"code": "example.solver = newton.solvers.SolverSemiImplicit(example.model)"})
        self.assertAlmostEqual(advance(), 0.2, places=4)

    def test_hosted_errors_keep_scene_valid_and_rebuild_reloads_script(self):
        """Report Python errors without invalidating, and reload the edited script in place."""
        host = ExampleHost(self.script)
        session = host.session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        with self.assertRaisesRegex(RuntimeError, "nothing was restored"):
            session.dispatch("execute", {"code": "kept = 3\nmissing_name"})
        self.assertTrue(session.valid)
        self.assertEqual(session.dispatch("execute", {"code": "kept"})["result"], 3)
        self.script.write_text(_SCRIPT.replace("self.speed = 1.0", "self.speed = 3.0"))
        session.dispatch("rebuild", {})
        self.assertEqual(session.dispatch("execute", {"code": "(example.speed, kept)"})["result"], [3.0, 3])

    def test_hosts_examples_without_solver_or_second_state(self):
        """Serve kinematic examples that own a single state and no dynamics solver."""
        script = Path(self.directory.name) / "kinematic.py"
        script.write_text(
            textwrap.dedent(
                """
                import newton


                class Example:
                    def __init__(self, viewer, args):
                        builder = newton.ModelBuilder()
                        builder.add_shape_sphere(builder.add_body(), radius=0.1)
                        self.model = builder.finalize()
                        self.state_0 = self.model.state()
                        self.frame_dt = 0.5
                        self.moves = 0

                    def step(self):
                        self.moves += 1
                """
            )
        )
        session = ExampleHost(script).session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        session.dispatch("step", {"count": 2})
        session.dispatch("checkpoint", {"name": "two"})
        session.dispatch("step", {"count": 1})
        session.dispatch("restore", {"name": "two"})
        self.assertEqual(session.dispatch("execute", {"code": "example.moves"})["result"], 2)

    def test_observations_include_meshes_the_example_renders(self):
        """Composite meshes logged in render() over the model's shapes and frame them."""
        script = Path(self.directory.name) / "logged.py"
        script.write_text(
            textwrap.dedent(
                """
                import numpy as np
                import warp as wp

                import newton


                class Example:
                    def __init__(self, viewer, args):
                        self.viewer = viewer
                        builder = newton.ModelBuilder()
                        builder.add_shape_sphere(-1, radius=0.02)
                        self.model = builder.finalize(device="cpu")
                        self.state_0 = self.model.state()
                        self.frame_dt = 0.1
                        self.draw = True
                        points = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
                        self.points = wp.array(points, dtype=wp.vec3, device="cpu")
                        self.indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device="cpu")

                    def step(self):
                        pass

                    def render(self):
                        self.viewer.begin_frame(0.0)
                        if self.draw:
                            self.viewer.log_mesh("/quad", self.points, self.indices, backface_culling=False)
                        self.viewer.end_frame()
                """
            )
        )
        session = ExampleHost(script).session(artifact_directory=self.directory.name)
        self.addCleanup(session.close)

        def lit_pixels():
            camera = {"eye": [0.0, 0.0, 3.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]}
            options = {"width": 64, "height": 64, "shadows": False, "environment": False}
            result = session.dispatch("observe", {**camera, **options})
            from newton._src.mcp.imaging import decode_png  # noqa: PLC0415

            return int((decode_png(base64.b64decode(result["image_base64"])).max(axis=-1) > 0).sum())

        with_quad = lit_pixels()
        session.dispatch("execute", {"code": "example.draw = False"})
        self.assertGreater(with_quad, 4 * lit_pixels())


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

    def test_render_returns_arrays(self):
        """Render arrays directly; removed helpers and options are not part of the workspace."""
        camera = {"eye": [1.5, -1.5, 1.0], "target": [0.0, 0.0, 0.3], "width": 64, "height": 48}
        image = self.session.render(**camera)
        self.assertEqual(image.shape, (48, 64, 3))
        self.assertEqual(image.dtype, np.uint8)
        result = self.session.dispatch(
            "execute", {"code": f"a = render(**{camera!r})\n(a.shape, 'compare_images' in globals())"}
        )
        self.assertEqual(result["result"], [[48, 64, 3], False])
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'plot'"):
            self.session.rollout(1, plot=True)

    def test_workspace_preloads_newton_and_helpers(self):
        """Provide newton and the helper functions without imports."""
        result = self.session.dispatch("execute", {"code": "(newton.__name__, callable(rollout), health()['ok'])"})
        self.assertEqual(result["result"], ["newton", True, True])

    def test_client_follows_a_restarted_server(self):
        """Re-read the connection file when the server behind it restarts with a new port and token."""
        from newton.mcp import SimulationClient, SimulationServer  # noqa: PLC0415

        connection = Path(self.directory.name) / "restart.json"

        def call(client):
            # Requests execute on the session's owner thread, so pump here while a thread waits.
            result = {}
            thread = threading.Thread(target=lambda: result.update(client.request("describe")))
            thread.start()
            while thread.is_alive():
                self.session.pump()
                time.sleep(0.005)
            return result

        first = SimulationServer(self.session, connection_file=connection)
        first.start()
        client = SimulationClient(connection, timeout=5)
        self.assertEqual(call(client)["frame"], 0)
        first.close()
        second = SimulationServer(self.session, connection_file=connection)
        second.start()
        self.addCleanup(second.close)
        self.assertEqual(call(client)["frame"], 0)


@unittest.skipUnless(wp.is_cuda_available(), "SolverMuJoCo contact forces need CUDA")
class TestMcpContactsBetween(unittest.TestCase):
    def _session(self, use_mujoco_contacts: bool):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), label="box")
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))
        model = builder.finalize(device="cuda:0")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = SimulationSession(
            model,
            newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=use_mujoco_contacts),
            dt=0.005,
            allow_execute=True,
            artifact_directory=directory.name,
        )
        self.addCleanup(session.close)
        return session, float(model.body_mass.numpy()[0])

    def test_resting_box_carries_its_weight_and_sliding_box_slips(self):
        """Report the supporting force of a resting box and the slip of a sliding one, as rollout series."""
        for use_mujoco_contacts in (True, False):
            session, mass = self._session(use_mujoco_contacts)
            resting = session.rollout(seconds=0.5, record={"box": lambda s=session: s.contacts_between("box")})
            self.assertAlmostEqual(resting["box.normal_force"][-1], 9.81 * mass, delta=0.02 * 9.81 * mass)
            # The same series is reachable through the probe's name.
            self.assertIs(resting["box"]["normal_force"], resting["box.normal_force"])
            self.assertGreater(resting["box.touching"][-1], 0)
            self.assertLess(resting["box.slip_max"][-1], 1.0e-3)
            joint_qd = session.state.joint_qd.numpy()
            joint_qd[0] = 1.0
            session.state.joint_qd.assign(joint_qd)
            session.rollout(frames=1)
            sliding = session.contacts_between({"body": "box"}, "ground", detail=True)
            self.assertGreater(sliding["slip_max"], 0.3)
            self.assertIn("box | world", sliding["by_body"])
            with self.assertRaises(ValueError):
                session.contacts_between("no_such_shape")


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

    def test_rtx_backend_is_advertised_only_when_installed(self):
        """Instructions mention backend='rtx' only if the optional ovrtx renderer can be imported."""
        from unittest import mock  # noqa: PLC0415

        from newton._src.mcp import protocol as protocol_module  # noqa: PLC0415

        class Client:
            def request(self, operation, **_):
                return {}

        for installed in (False, True):
            with mock.patch.object(protocol_module, "rtx_available", return_value=installed):
                server = protocol_module._Protocol(Client(), profile="lean", app_guide=False)
                result = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})["result"]
            self.assertEqual("backend='rtx'" in result["instructions"], installed)
            self.assertNotIn("<<RTX>>", result["instructions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
