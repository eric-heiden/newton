# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise live operations, transport ownership, and parameter propagation."""

import asyncio
import base64
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.mcp.protocol import _Protocol
from newton.mcp import SimulationClient, SimulationServer, SimulationSession
from newton.solvers import SolverMuJoCo, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled


class TestMcp(unittest.TestCase):
    def setUp(self):
        """Build a small CPU scene with real rigid-body dynamics."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        self.session = SimulationSession(model, SolverXPBD(model), artifact_directory=self.directory.name)
        self.addCleanup(self.session.close)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_cuda_edit_step_and_observe(self):
        """Run live array edits, physics, queries and sensor rendering on CUDA."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cuda:0")
        session = SimulationSession(model, SolverXPBD(model), artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        session.dispatch("edit", {"patches": [{"field": "gravity", "values": [[0, 0, 0]]}]})
        session.dispatch("step", {"count": 2})
        result = session.dispatch("query", {"field": "body_q"})
        self.assertAlmostEqual(result["values"][0][2], 1)
        result = session.dispatch("observe", {"eye": [0, -3, 1], "target": [0, 0, 1], "width": 32, "height": 32})
        self.assertTrue(base64.b64decode(result["image_base64"]).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_live_edit_affects_dynamics_and_reset(self):
        """Change live gravity in place and restore state/control without losing tuning."""
        session = self.session
        initial = session.state.body_q.numpy().copy()
        gravity_pointer = session.model.gravity.ptr
        session.dispatch("step", {"count": 6})
        self.assertLess(session.state.body_q.numpy()[0, 2], initial[0, 2])
        session.dispatch("edit", {"patches": [{"field": "gravity", "values": [[0, 0, 0]]}]})
        self.assertEqual(session.model.gravity.ptr, gravity_pointer)
        session.control.joint_f.fill_(8)
        session.dispatch("reset")
        self.assertEqual(session.time, 0)
        self.assertEqual(session.frame, 0)
        np.testing.assert_array_equal(session.control.joint_f.numpy(), 0)
        session.dispatch("step", {"count": 6})
        np.testing.assert_allclose(session.state.body_q.numpy(), initial)

    def test_edit_validation_and_mass_inverses(self):
        """Validate entire patches before writing and keep mass/inertia inverses coherent."""
        session = self.session
        gravity = session.model.gravity.numpy().copy()
        with self.assertRaises(ValueError):
            session.dispatch(
                "edit",
                {
                    "patches": [
                        {"field": "gravity", "values": [[0, 0, 0]]},
                        {"field": "body_mass", "indices": [0], "values": [-1]},
                    ]
                },
            )
        np.testing.assert_array_equal(session.model.gravity.numpy(), gravity)
        for values in ([[float("nan"), 0, 0]], [[1e100, 0, 0]]):
            with self.assertRaises(ValueError):
                session.dispatch("edit", {"patches": [{"field": "gravity", "values": values}]})
        mass = session.model.body_mass.numpy()[0]
        inertia = session.model.body_inertia.numpy().copy()
        session.dispatch("edit", {"patches": [{"field": "body_mass", "values": [float(2 * mass)]}]})
        self.assertAlmostEqual(session.model.body_inv_mass.numpy()[0], 1 / (2 * mass))
        np.testing.assert_allclose(session.model.body_inertia.numpy(), inertia * 2)
        with self.assertRaises(ValueError):
            session.dispatch("edit", {"patches": [{"field": "shape_body", "values": [0, 0]}]})
        with self.assertRaises(ValueError):
            session.dispatch(
                "edit", {"expected_revision": -1, "patches": [{"field": "gravity", "values": [[0, 0, 0]]}]}
            )

    def test_mass_edit_preserves_unselected_kinematic_inverses(self):
        """Preserve authored kinematic inverse values when another body's mass changes."""
        builder = newton.ModelBuilder()
        for kinematic in (True, False):
            body = builder.add_body(is_kinematic=kinematic)
            builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        inverse = model.body_inv_mass.numpy()
        inverse[0] = 0
        model.body_inv_mass.assign(inverse)
        session = SimulationSession(model, SolverXPBD(model))
        self.addCleanup(session.close)
        mass = float(model.body_mass.numpy()[1])
        session.dispatch("edit", {"patches": [{"field": "body_mass", "indices": [1], "values": [2 * mass]}]})
        self.assertEqual(int(model.body_flags.numpy()[0]), int(newton.BodyFlags.KINEMATIC))
        self.assertEqual(model.body_inv_mass.numpy()[0], 0)
        self.assertAlmostEqual(model.body_inv_mass.numpy()[1], 1 / (2 * mass))

    def test_playback_failure_keeps_session_available_for_recovery(self):
        """Keep serving requests after playback fails so a client can reset and step."""
        path = Path(self.directory.name) / "playback-session.json"
        completed = threading.Event()
        errors, results, attempts = [], [], []
        original_step, original_pump = self.session.solver.step, self.session.pump
        deadline = time.monotonic() + 10

        def flaky_step(*args):
            if not attempts:
                attempts.append(True)
                raise RuntimeError("playback failure " * 400)
            return original_step(*args)

        def pump():
            count = original_pump()
            if completed.is_set():
                self.session.close()
            if time.monotonic() > deadline:
                raise TimeoutError("Playback recovery client did not finish")
            return count

        def worker():
            try:
                client = SimulationClient(path)
                client.request("play")
                status = client.request("describe")
                while status["valid"]:
                    status = client.request("describe")
                self.assertTrue(status["paused"])
                self.assertIn("playback failure", status["last_error"])
                self.assertLessEqual(len(status["last_error"]), 4096)
                client.request("reset")
                results.append(client.request("step"))
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        with SimulationServer(self.session, connection_file=path):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            try:
                with (
                    patch.object(self.session.solver, "step", side_effect=flaky_step),
                    patch.object(self.session, "pump", side_effect=pump),
                ):
                    self.session.run()
            finally:
                thread.join(timeout=2)
        if errors:
            raise errors[0]
        self.assertEqual(results[0]["frame"], 1)
        self.assertTrue(results[0]["valid"])

    def test_failed_notification_requires_rebuild(self):
        """Reject state-only recovery when a failed notification leaves model coherence unknown."""
        with patch.object(self.session.solver, "notify_model_changed", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                self.session.dispatch("edit", {"patches": [{"field": "gravity", "values": [[0, 0, 0]]}]})
        self.assertFalse(self.session.valid)
        self.assertTrue(self.session.paused)
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            self.session.dispatch("reset")

    def test_world_frequency_filters_and_global_gravity(self):
        """Select explicit local/global rows using metadata rather than equal array lengths."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for world in range(2):
            builder.begin_world(gravity=(0, 0, -world - 1))
            body = builder.add_body(xform=wp.transform(wp.vec3(world, 0, 1), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1)
            builder.end_world()
        model = builder.finalize(device="cpu")
        session = SimulationSession(model, SolverXPBD(model))
        self.addCleanup(session.close)
        result = session.dispatch("query", {"root": "state", "field": "body_q", "world": 1})
        self.assertEqual(result["indices"], [1])
        result = session.dispatch("query", {"root": "model", "field": "gravity", "world": -1})
        self.assertEqual(result["indices"], [2])
        result = session.dispatch("query", {"root": "model", "field": "shape_label", "world": -1})
        self.assertEqual(result["indices"], [0])
        result = self.session.dispatch("query", {"root": "model", "field": "gravity", "world": -1})
        self.assertEqual(result["indices"], [0])
        result = session.dispatch("query", {"root": "control", "field": "joint_f", "joint": 1})
        self.assertEqual(result["indices"], list(range(6, 12)))
        with self.assertRaises(ValueError):
            session.dispatch("query", {"root": "solver", "field": "iterations", "world": 0})

    def test_query_gathers_only_requested_rows(self):
        """Read a bounded page without transferring an entire large Warp field."""
        self.session.state.large = wp.zeros(2_000_010, dtype=float, device="cpu")
        result = self.session.dispatch("query", {"field": "large", "offset": 2_000_005, "limit": 2})
        self.assertEqual(result["indices"], [2_000_005, 2_000_006])
        self.assertEqual(result["values"], [0, 0])
        self.assertEqual(result["total_matched"], 2_000_010)
        del self.session.state.large

    def test_contact_world_coordinates_and_freshness(self):
        """Return reconstructed world support points and identify contact generation time."""
        state = self.session.state.body_q.numpy()
        state[0, 2] = 0.09
        self.session.state.body_q.assign(state)
        result = self.session.dispatch("contacts", {"refresh": True, "body": 0})
        self.assertGreater(result["count"], 0)
        row = result["rows"][0]
        self.assertLess(abs(row["point0"][2]), 0.11)
        self.assertLess(abs(row["point1"][2]), 0.11)
        self.assertLess(row["distance"], 0)
        self.assertEqual(result["contact_revision"], self.session.revision)
        self.session.dispatch("step")
        result = self.session.dispatch("contacts")
        self.assertEqual(result["contact_frame"], self.session.frame - 1)

    def test_soft_contacts_include_particle_world(self):
        """Match particles in a local world when their contact shape is global."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.begin_world()
        builder.add_particle(pos=wp.vec3(0, 0, 0.005), vel=wp.vec3(), mass=1, radius=0.01)
        builder.end_world()
        model = builder.finalize(device="cpu")
        session = SimulationSession(model, SolverXPBD(model))
        self.addCleanup(session.close)
        result = session.dispatch("contacts", {"refresh": True, "kind": "soft", "world": 0})
        self.assertGreater(result["total_matched"], 0)
        self.assertEqual(result["rows"][0]["particle"], 0)

    def test_coupled_public_queries_and_single_notification(self):
        """Inspect actual coupled entries and notify their children exactly once."""
        model = self.session.model
        solver = SolverCoupled(model, [SolverCoupled.Entry("rigid", SolverXPBD, bodies=[0], joints=[0], shapes=[0, 1])])
        self.session.replace(model, solver)
        description = self.session.dispatch("describe")
        self.assertIn("rigid", description["solver"]["entries"])
        self.assertEqual(self.session.dispatch("query", {"entry": "rigid", "field": "body_q"})["shape"], [1, 7])
        self.session.dispatch("contacts", {"entry": "rigid"})
        child = solver.solver("rigid")
        with patch.object(child, "notify_model_changed", wraps=child.notify_model_changed) as notify:
            self.session.dispatch("edit", {"patches": [{"field": "gravity", "values": [[0, 0, 0]]}]})
        notify.assert_called_once_with(int(newton.ModelFlags.MODEL_PROPERTIES))

    def test_checkpoint_and_step_failure_recovery(self):
        """Restore checkpoint time and recover an interrupted state step through reset."""
        self.session.dispatch("step", {"count": 3})
        saved = self.session.state.body_q.numpy().copy()
        self.session.dispatch("checkpoint", {"name": "trial"})
        self.session.dispatch("step", {"count": 2})
        self.session.dispatch("restore", {"name": "trial"})
        self.assertEqual(self.session.frame, 3)
        np.testing.assert_array_equal(self.session.state.body_q.numpy(), saved)
        with patch.object(self.session.solver, "step", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                self.session.dispatch("step")
        self.assertFalse(self.session.valid)
        self.session.dispatch("reset")
        self.assertTrue(self.session.valid)

    def test_execute_is_opt_in_bounded_and_invalidates_failure(self):
        """Bound trusted output and require rebuilding after arbitrary execution failure."""
        with self.assertRaises(PermissionError):
            self.session.dispatch("execute", {"code": "result = 1"})
        self.session.allow_execute = True
        result = self.session.dispatch("execute", {"code": "print('x' * 50000); result = {'frame': session.frame}"})
        self.assertEqual(len(result["stdout"]), 16384)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["result"], {"frame": 0})
        with self.assertRaises(RuntimeError):
            self.session.dispatch("execute", {"code": "model.gravity.zero_(); raise ValueError('broken')"})
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            self.session.dispatch("reset")

    def test_result_encoding_failure_preserves_completed_execution(self):
        """Keep completed Python mutations valid when their result exceeds the output budget."""
        self.session.allow_execute = True
        revision = self.session.revision
        with self.assertRaisesRegex(RuntimeError, "Python completed"):
            self.session.dispatch("execute", {"code": "model.gravity.zero_(); result = list(range(30000))"})
        self.assertTrue(self.session.valid)
        self.assertEqual(self.session.revision, revision + 1)
        np.testing.assert_array_equal(self.session.model.gravity.numpy(), 0)
        self.assertEqual(self.session.dispatch("step")["frame"], 1)

    def test_timeout_cancels_pending_mutation(self):
        """Ensure a timed-out queued edit never executes on a later pump."""
        original = self.session.model.gravity.numpy().copy()
        request = self.session.enqueue("edit", {"patches": [{"field": "gravity", "values": [[0, 0, 0]]}]}, timeout=0.01)
        with self.assertRaises(TimeoutError):
            request.wait(0.02)
        self.session.pump()
        np.testing.assert_array_equal(self.session.model.gravity.numpy(), original)

    def test_shutdown_rejects_pending_requests(self):
        """Wake queued clients when their session closes."""
        request = self.session.enqueue("step", {}, timeout=1)
        self.session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            request.wait(0.01)

    def test_server_shutdown_cancels_queued_mutation(self):
        """Cancel this server's queued work without closing the live simulation."""
        path = Path(self.directory.name) / "closing-session.json"
        server = SimulationServer(self.session, connection_file=path).start()
        self.addCleanup(server.close)
        errors = []

        def worker():
            try:
                SimulationClient(path).request("step")
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while self.session._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.002)
        self.assertFalse(self.session._queue.empty())
        server.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.session.pump()
        self.assertEqual(self.session.frame, 0)
        self.assertTrue(self.session.valid)

    def test_invalid_control_quaternion_does_not_write(self):
        """Reject zero-norm coordinate-layout rotation targets before updating controls."""
        original = self.session.control.joint_target_q.numpy().copy()
        with self.assertRaisesRegex(ValueError, "quaternion"):
            self.session.dispatch(
                "edit", {"patches": [{"root": "control", "field": "joint_target_q", "values": [0] * 7}]}
            )
        np.testing.assert_array_equal(self.session.control.joint_target_q.numpy(), original)

    def test_owner_thread_and_authenticated_client(self):
        """Run loopback operations on the simulation owner thread and reject bad tokens."""
        path = Path(self.directory.name) / "session.json"
        with SimulationServer(self.session, connection_file=path):
            results = []

            def worker():
                try:
                    self.session.dispatch("step")
                except RuntimeError as error:
                    results.append(str(error))
                results.append(SimulationClient(path).request("step", count=2))

            thread = threading.Thread(target=worker)
            thread.start()
            deadline = time.monotonic() + 10
            while thread.is_alive() and time.monotonic() < deadline:
                self.session.pump()
                time.sleep(0.002)
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertIn("owning thread", results[0])
            self.assertEqual(results[1]["frame"], 2)
            descriptor = json.loads(path.read_text())
            with socket.create_connection((descriptor["host"], descriptor["port"])) as connection:
                connection.sendall(b'{"token":"bad", "operation":"step"}\n')
                error = json.loads(connection.makefile("rb").readline())
            self.assertEqual(error["error"]["type"], "PermissionError")
        self.assertFalse(path.exists())

    def test_malformed_tool_name_preserves_protocol(self):
        """Reject malformed tool names and continue servicing JSON-RPC requests."""
        protocol = _Protocol(object())
        protocol.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        for name in ([], {}, None, 12):
            response = protocol.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name}})
            self.assertEqual(response["error"]["code"], -32602)
        response = protocol.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        self.assertEqual(response["result"], {})

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "Requires optional MCP SDK for interoperability validation")
    def test_official_sdk_code_profile(self):
        """Expose four code-profile tools and retain structured execution and rebuild recovery."""
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415
        from mcp.shared.exceptions import McpError  # noqa: PLC0415

        path = Path(self.directory.name) / "code-session.json"
        self.session.allow_execute = True
        self.session.rebuild_callback = lambda session: {"model": session.model, "solver": session.solver}
        errors = []
        completed = threading.Event()

        async def conversation():
            parameters = StdioServerParameters(
                command=sys.executable, args=["-m", "newton.mcp", "--connect", str(path), "--profile", "code"]
            )
            async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
                await client.initialize()
                listing = await client.list_tools()
                self.assertEqual(
                    {tool.name for tool in listing.tools},
                    {"newton_describe", "newton_execute", "newton_observe", "newton_rebuild"},
                )
                description = await client.call_tool("newton_describe", {})
                self.assertIn("step", description.structuredContent["operations"])
                result = await client.call_tool(
                    "newton_execute", {"code": "result = session.dispatch('step', {'count': 2})"}
                )
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent["result"]["frame"], 2)
                with self.assertRaises(McpError):
                    await client.call_tool("newton_step", {"count": 1})
                result = await client.call_tool("newton_execute", {"code": "raise ValueError('trial failure')"})
                self.assertTrue(result.isError)
                result = await client.call_tool("newton_rebuild", {})
                self.assertFalse(result.isError)
                self.assertTrue(result.structuredContent["valid"])

        def worker():
            try:
                asyncio.run(conversation())
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        with SimulationServer(self.session, connection_file=path):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            deadline = time.monotonic() + 45
            while not completed.is_set() and time.monotonic() < deadline:
                self.session.pump()
                time.sleep(0.002)
            self.assertTrue(completed.is_set(), "Official MCP code-profile client did not finish")
            thread.join(timeout=1)
        if errors:
            raise errors[0]

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "Requires optional MCP SDK for interoperability validation")
    def test_official_sdk_stdio_interoperability(self):
        """Use the official MCP client to edit, step, query and observe a live session."""
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415

        path = Path(self.directory.name) / "sdk-session.json"
        errors = []
        completed = threading.Event()

        async def conversation():
            parameters = StdioServerParameters(
                command=sys.executable, args=["-m", "newton.mcp", "--connect", str(path)]
            )
            async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
                initialized = await client.initialize()
                self.assertEqual(initialized.serverInfo.name, "newton-live")
                listing = await client.list_tools()
                self.assertIn("newton_observe", {tool.name for tool in listing.tools})
                observation = next(tool for tool in listing.tools if tool.name == "newton_observe")
                self.assertEqual(observation.inputSchema["properties"]["contact_depth"]["enum"], ["visible", "always"])
                self.assertEqual(observation.inputSchema["properties"]["width"]["minimum"], 1)
                self.assertEqual(observation.inputSchema["properties"]["fov_y"]["maximum"], 175)
                result = await client.call_tool(
                    "newton_edit", {"patches": [{"field": "gravity", "values": [[0, 0, 0]]}]}
                )
                self.assertFalse(result.isError)
                result = await client.call_tool("newton_step", {"count": 2})
                self.assertFalse(result.isError)
                result = await client.call_tool("newton_query", {"field": "body_q", "limit": 1})
                self.assertEqual(result.structuredContent["frame"], 2)
                self.assertEqual(result.structuredContent["values"][0][2], 1)
                result = await client.call_tool(
                    "newton_observe",
                    {
                        "world_id": 0,
                        "eye": [0, -3, 1],
                        "target": [0, 0, 1],
                        "width": 32,
                        "height": 32,
                        "pick": [[16, 16]],
                        "contacts": True,
                        "contact_depth": "always",
                    },
                )
                self.assertFalse(result.isError, str(result.content))
                images = [item for item in result.content if item.type == "image"]
                self.assertEqual(len(images), 1)
                self.assertTrue(base64.b64decode(images[0].data).startswith(b"\x89PNG\r\n\x1a\n"))
                result = await client.call_tool("newton_edit", {"patches": [{"field": "shape_body", "values": [0]}]})
                self.assertTrue(result.isError)

        def worker():
            try:
                asyncio.run(conversation())
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        with SimulationServer(self.session, connection_file=path):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            deadline = time.monotonic() + 45
            while not completed.is_set() and time.monotonic() < deadline:
                self.session.pump()
                time.sleep(0.002)
            self.assertTrue(completed.is_set(), "Official MCP client did not finish")
            thread.join(timeout=1)
        if errors:
            raise errors[0]

    def test_observe_record_and_scene_replacement(self):
        """Render real MCP images and discard checkpoints when replacing topology."""
        options = {"eye": [0, -3, 1], "target": [0, 0, 1], "width": 32, "height": 32}
        image = self.session.dispatch("observe", options)
        self.assertTrue(base64.b64decode(image["image_base64"]).startswith(b"\x89PNG\r\n\x1a\n"))
        self.session.dispatch("record", {"action": "start", "max_frames": 2, **options})
        self.session.dispatch("step")
        recording = self.session.dispatch("record", {"action": "status"})
        self.assertFalse(recording["active"])
        self.session.dispatch("checkpoint")
        self.session.replace(self.session.model, self.session.solver)
        with self.assertRaises(KeyError):
            self.session.dispatch("restore")


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"), "Requires sim extra"
)
class TestMcpMuJoCoNotification(unittest.TestCase):
    @staticmethod
    def scene():
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_link()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, target_ke=5, target_kd=1)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        return model, solver

    def test_cpu_live_drive_update_matches_fresh_solver(self):
        """Make live CPU drive gain edits match a fresh solver's force response."""
        model, solver = self.scene()
        model.joint_target_ke.fill_(50)
        model.joint_target_kd.fill_(4)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        fresh = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        np.testing.assert_allclose(solver.mj_model.actuator_gainprm, fresh.mj_model.actuator_gainprm)
        np.testing.assert_allclose(solver.mj_model.actuator_biasprm, fresh.mj_model.actuator_biasprm)
        control = model.control()
        control.joint_target_q.fill_(0.5)
        state, output = model.state(), model.state()
        fresh_state, fresh_output = model.state(), model.state()
        solver.step(state, output, control, None, 0.01)
        fresh.step(fresh_state, fresh_output, control, None, 0.01)
        np.testing.assert_allclose(output.joint_qd.numpy(), fresh_output.joint_qd.numpy(), atol=1e-6)
        self.assertGreater(abs(output.joint_qd.numpy()[0]), 0)

    def test_ball_effort_edit_matches_fresh_solver(self):
        """Update ball-joint axis force limits and preserve authored actuator ranges."""
        for use_mujoco_cpu in (True, False):
            for authored_limit in (False, True):
                with self.subTest(use_mujoco_cpu=use_mujoco_cpu, authored_limit=authored_limit):
                    builder = newton.ModelBuilder(gravity=(0, 0, 0))
                    actuator = (
                        '<actuator><position joint="ball" kp="8" forcerange="-0.3 0.4"/></actuator>'
                        if authored_limit
                        else ""
                    )
                    builder.add_mjcf(
                        '<mujoco><option gravity="0 0 0"/><worldbody><body>'
                        '<joint name="ball" type="ball"/><geom type="sphere" size="0.1" mass="1"/>'
                        f"</body></worldbody>{actuator}</mujoco>"
                    )
                    builder.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)] * 3
                    builder.joint_target_ke[:] = [8.0] * 3
                    builder.joint_effort_limit[:] = [0.8] * 3
                    device = "cuda:0" if not use_mujoco_cpu and wp.is_cuda_available() else "cpu"
                    model = builder.finalize(device=device)
                    solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                    model.joint_effort_limit.assign(np.asarray([0.2, 0.4, 0.6], dtype=np.float32))
                    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                    fresh = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                    np.testing.assert_allclose(
                        solver.mjw_model.actuator_forcerange.numpy(), fresh.mjw_model.actuator_forcerange.numpy()
                    )
                    if use_mujoco_cpu:
                        np.testing.assert_allclose(
                            solver.mj_model.actuator_forcerange, fresh.mj_model.actuator_forcerange
                        )
                        control = model.control()
                        control.joint_target_q.assign(np.asarray([0.3, 0.4, 0.1, np.sqrt(0.74)], dtype=np.float32))
                        state, output = model.state(), model.state()
                        fresh_state, fresh_output = model.state(), model.state()
                        solver.step(state, output, control, None, 0.01)
                        fresh.step(fresh_state, fresh_output, control, None, 0.01)
                        np.testing.assert_allclose(output.joint_qd.numpy(), fresh_output.joint_qd.numpy(), atol=1e-6)

    def test_ball_velocity_effort_edit_matches_fresh_solver(self):
        """Update effort limits on both ball position and velocity sub-actuators."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_link()
        builder.add_shape_sphere(body, radius=0.1)
        joint = builder.add_joint_ball(-1, body)
        builder.add_articulation([joint])
        builder.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION_VELOCITY)] * 3
        builder.joint_target_ke[:] = [8.0] * 3
        builder.joint_target_kd[:] = [1.0] * 3
        builder.joint_effort_limit[:] = [0.8] * 3
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        model.joint_effort_limit.assign(np.asarray([0.2, 0.4, 0.6], dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        fresh = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        self.assertEqual(solver.mj_model.nu, 6)
        np.testing.assert_allclose(solver.mj_model.actuator_forcerange, fresh.mj_model.actuator_forcerange)

    def test_cpu_direct_actuator_parameters_match_fresh_solver(self):
        """Propagate direct actuator gains, biases, gear and force limits into CPU buffers."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            """<mujoco><option gravity="0 0 0"/><worldbody><body>
            <joint name="hinge" type="hinge"/><geom type="sphere" size="0.1" mass="1"/>
            </body></worldbody><actuator><general joint="hinge" gainprm="2" biasprm="0 -2 -1"
            biastype="affine" forcerange="-10 10" ctrlrange="-1 1"/></actuator></mujoco>""",
            ctrl_direct=True,
        )
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        for field, values in {
            "actuator_gainprm": [12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "actuator_biasprm": [0, -12, -3, 0, 0, 0, 0, 0, 0, 0],
            "actuator_gear": [2, 0, 0, 0, 0, 0],
            "actuator_forcerange": [-6, 6],
            "actuator_ctrlrange": [-0.5, 0.5],
        }.items():
            getattr(model.mujoco, field).assign(np.asarray([values], dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES)
        fresh = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        for field in (
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_actrange",
            "actuator_gear",
            "actuator_cranklength",
        ):
            np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(fresh.mj_model, field), err_msg=field)

    def test_cpu_shape_friction_update_matches_fresh_solver(self):
        """Refresh CPU shape friction from live Newton material parameters."""
        model, solver = self.scene()
        model.shape_material_mu.fill_(0.87)
        model.shape_material_mu_torsional.fill_(0.023)
        model.shape_material_mu_rolling.fill_(0.004)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        fresh = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        np.testing.assert_allclose(solver.mj_model.geom_friction, fresh.mj_model.geom_friction)


class TestMcpCheckpoint(unittest.TestCase):
    """Restore actual noninitial states across supported reset implementations."""

    def _check_checkpoint(self, solver_factory):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            session = SimulationSession(model, solver_factory(model), dt=0.01, artifact_directory=directory)
            self.addCleanup(session.close)
            session.dispatch("step", {"count": 5})
            self.assertFalse(np.array_equal(session.state.body_q.numpy(), model.body_q.numpy()))
            session.control.joint_f.fill_(0.1)
            saved = {
                field: getattr(session.state, field).numpy().copy()
                for field in ("body_q", "body_qd", "joint_q", "joint_qd")
            }
            session.dispatch("checkpoint", {"name": "moving"})
            session.dispatch("step")
            expected = {
                field: getattr(session.state, field).numpy().copy()
                for field in ("body_q", "body_qd", "joint_q", "joint_qd")
            }
            session.dispatch("step", {"count": 3})
            session.control.joint_f.zero_()
            session.dispatch("restore", {"name": "moving"})
            self.assertEqual(session.frame, 5)
            self.assertAlmostEqual(session.time, 0.05)
            for field, data in saved.items():
                np.testing.assert_array_equal(getattr(session.state, field).numpy(), data, err_msg=field)
                np.testing.assert_array_equal(getattr(session.state_next, field).numpy(), data, err_msg=field)
            np.testing.assert_allclose(session.control.joint_f.numpy(), 0.1)
            session.dispatch("step")
            for field, data in expected.items():
                np.testing.assert_allclose(getattr(session.state, field).numpy(), data, atol=1e-6, err_msg=field)

    def test_xpbd_checkpoint_preserves_public_state(self):
        """Restore a moving XPBD body and continue from its saved state and control."""
        self._check_checkpoint(SolverXPBD)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"), "Requires sim extra"
    )
    def test_mujoco_checkpoint_preserves_public_state(self):
        """Restore reduced coordinates with ordinary and disabled per-step synchronization."""
        for interval in (1, 0):
            with self.subTest(update_data_interval=interval):
                self._check_checkpoint(
                    lambda model, interval=interval: SolverMuJoCo(
                        model, use_mujoco_cpu=True, disable_contacts=True, update_data_interval=interval
                    )
                )

    def test_reset_clears_contacts_until_explicit_refresh(self):
        """Clear contact counts and freshness without eagerly regenerating collisions."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.09), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            session = SimulationSession(model, SolverXPBD(model), artifact_directory=directory)
            self.addCleanup(session.close)
            before = session.dispatch("contacts", {"refresh": True})
            self.assertGreater(before["count"], 0)
            pointer = session.contacts.rigid_contact_point0.ptr
            generation = int(session.contacts.contact_generation.numpy()[0])
            with patch.object(
                session.collision_pipeline, "collide", wraps=session.collision_pipeline.collide
            ) as collide:
                session.dispatch("reset")
                collide.assert_not_called()
            cleared = session.dispatch("contacts")
            self.assertEqual(cleared["count"], 0)
            self.assertIsNone(cleared["contact_frame"])
            self.assertIsNone(cleared["contact_revision"])
            self.assertGreater(int(session.contacts.contact_generation.numpy()[0]), generation)
            self.assertEqual(session.contacts.rigid_contact_point0.ptr, pointer)
            refreshed = session.dispatch("contacts", {"refresh": True})
            self.assertGreater(refreshed["count"], 0)
            self.assertEqual(refreshed["contact_revision"], session.revision)


if __name__ == "__main__":
    unittest.main()
