# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise persistent trusted Python cells and failed-cell rollback."""

import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import linecache
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.mcp.cells import register_cell_source, retain_cell_sources
from newton.mcp import SimulationServer, SimulationSession


def _decode(data: str) -> np.ndarray:
    from newton._src.mcp.imaging import decode_png  # noqa: PLC0415

    return decode_png(base64.b64decode(data))


def _payload(result) -> dict:
    """Decode the compact JSON text that accompanies every MCP tool result."""
    return json.loads(result.content[0].text)


class TestMcpExecutor(unittest.TestCase):
    def setUp(self):
        """Construct a real CPU simulation with trusted execution enabled."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        self.session = SimulationSession(
            model, newton.solvers.SolverXPBD(model), allow_execute=True, artifact_directory=self.directory.name
        )
        self.addCleanup(self.session.close)

    def execute(self, code, **kwargs):
        return self.session.dispatch("execute", {"code": code, **kwargs})

    def test_imports_functions_and_live_state_persist(self):
        """Keep imported functions while refreshing their global state after odd buffer swaps."""
        self.execute(
            "import math\nvalues = [1, 2, 3]\ndef sample():\n    return [math.fsum(values), state is session.state, float(state.body_q.numpy()[0, 2])]"
        )
        first_state = self.session.state
        self.session.dispatch("step", {"count": 1})
        self.assertIsNot(self.session.state, first_state)
        result = self.execute("sample()")
        self.assertEqual(result["result"][:2], [6, True])
        self.assertLess(result["result"][2], 1)
        result = self.execute("session.dispatch('step', {'count': 1})\nsample()")
        self.assertTrue(result["result"][1])
        self.assertIn("sample", result["workspace"]["variables"])

    def test_last_expression_is_returned_and_result_is_an_ordinary_variable(self):
        """Return only the last expression; a variable named result is neither returned nor deleted."""
        self.assertEqual(self.execute("2 + 3")["result"], 5)
        self.assertEqual(self.execute("_ + 1")["result"], 6)
        self.assertEqual(self.execute("result = 9\n42")["result"], 42)
        self.assertIsNone(self.execute("value = 12")["result"])
        self.assertEqual(self.execute("value")["result"], 12)
        self.assertEqual(self.execute("result")["result"], 9)
        self.assertIsNone(self.execute("result = 13")["result"])
        self.assertIn("result", self.execute("None")["workspace"]["variables"])

    def test_reset_preserves_workspace_and_refreshes_bindings(self):
        """Retain analysis variables through physical resets and update live function globals."""
        self.execute(
            "observations = []\ndef current():\n    return state is session.state and control is session.control"
        )
        self.session.dispatch("step", {"count": 3})
        generation = self.execute("observations.append(session.frame)")["workspace"]["generation"]
        self.session.dispatch("reset")
        result = self.execute("[observations, current(), session.frame]")
        self.assertEqual(result["result"], [[3], True, 0])
        self.assertEqual(result["workspace"]["generation"], generation)

    def test_replacement_keeps_workspace_and_refreshes_bindings(self):
        """Keep user definitions across scene replacement unless explicitly cleared."""
        result = self.execute("old_state = state\nhelper = lambda captured=state: captured\nvalue = 7")
        generation = result["workspace"]["generation"]
        self.session.replace(self.session.model, newton.solvers.SolverXPBD(self.session.model))
        result = self.execute("[value, old_state is state, state is session.state, helper() is old_state]")
        self.assertEqual(result["result"], [7, False, True, True])
        self.assertEqual(result["workspace"]["generation"], generation)
        self.session.replace(self.session.model, self.session.solver, keep_workspace=False)
        result = self.execute("'value' in globals()")
        self.assertFalse(result["result"])
        self.assertGreater(result["workspace"]["generation"], generation)
        self.execute("value = 8")
        result = self.execute("['value' in globals(), state is session.state]", reset_namespace=True)
        self.assertEqual(result["result"], [False, True])

    def test_compile_error_preserves_state_and_namespace(self):
        """Keep a valid scene and existing variables when compilation fails before execution."""
        self.execute("value = 8")
        revision = self.session.revision
        with self.assertRaises(SyntaxError):
            self.execute("value = 99\nif :\n    pass", reset_namespace=True)
        self.assertTrue(self.session.valid)
        self.assertEqual(self.session.revision, revision)
        status = self.session.dispatch("describe")
        self.assertFalse(status["requires_rebuild"])
        self.assertEqual(status["workspace"]["last_error"]["line"], 2)
        self.assertEqual(self.execute("value")["result"], 8)

    def test_failed_cell_keeps_workspace_and_session_usable(self):
        """Report an analysis error, keep earlier variables, and leave the scene valid without any recovery step."""
        with self.assertRaisesRegex(RuntimeError, "line 3.*nothing was restored"):
            self.execute("samples = [1, 2]\nprint('before failure')\nmissing_name")
        status = self.session.dispatch("describe")
        self.assertTrue(status["valid"])
        self.assertFalse(status["requires_rebuild"])
        self.assertEqual(status["workspace"]["last_error"]["type"], "NameError")
        self.assertIn("samples", status["workspace"]["variables"])
        self.assertEqual(self.execute("samples")["result"], [1, 2])
        self.assertEqual(self.session.dispatch("step")["frame"], 1)

    def test_failed_cell_rolls_back_the_simulation(self):
        """Undo state, time, control, and model edits made before an error, and notify the solver."""
        self.session.dispatch("step", {"count": 2})
        body_q = self.session.state.body_q.numpy().copy()
        gravity = self.session.model.gravity.numpy().copy()
        with self.assertRaises(RuntimeError) as raised:
            self.execute(
                "session.dispatch('step', {'count': 3})\n"
                "model.gravity.zero_()\n"
                "control.joint_f.fill_(5.0)\n"
                "after = session.frame\n"
                "raise ValueError('after mutation')"
            )
        message = str(raised.exception)
        self.assertIn("rolled back from t=", message)
        self.assertIn("model (gravity)", message)
        self.assertIn("MODEL_PROPERTIES", message)
        self.assertEqual(self.session.frame, 2)
        np.testing.assert_array_equal(self.session.state.body_q.numpy(), body_q)
        np.testing.assert_array_equal(self.session.model.gravity.numpy(), gravity)
        np.testing.assert_array_equal(self.session.control.joint_f.numpy(), 0)
        self.assertTrue(self.session.valid)
        # Python variables are not rolled back.
        self.assertEqual(self.execute("after")["result"], 5)
        # Dynamics continue from the restored state with the restored gravity.
        reference = SimulationSession(
            self.session.model, newton.solvers.SolverXPBD(self.session.model), allow_execute=True
        )
        self.addCleanup(reference.close)
        reference.dispatch("step", {"count": 3})
        self.session.dispatch("step", {"count": 1})
        np.testing.assert_allclose(self.session.state.body_q.numpy(), reference.state.body_q.numpy(), atol=1e-6)

    def test_rollout_error_rolls_back_and_session_stays_valid(self):
        """Roll back a rollout whose probe raises halfway, then keep stepping normally."""
        with self.assertRaisesRegex(RuntimeError, "rolled back from t=.*to t=0 s"):
            self.execute(
                "def probe():\n"
                "    if session.frame == 4:\n"
                "        raise np.linalg.LinAlgError('SVD did not converge')\n"
                "    return session.frame\n"
                "rollout(10, record={'f': probe})"
            )
        self.assertEqual(self.session.frame, 0)
        self.assertTrue(self.session.valid)
        self.assertEqual(self.execute("rollout(3)['frames']")["result"], 3)

    def test_invalid_scene_keeps_python_and_points_to_rebuild(self):
        """Allow Python while invalid, refuse stepping with a rebuild instruction, and recover by rebuilding."""
        model = self.session.model
        self.session.rebuild_callback = lambda session: {"model": model, "solver": newton.solvers.SolverXPBD(model)}
        self.execute("kept = 7")
        self.session._invalidate(requires_rebuild=True)
        self.session.last_error = "test failure"
        self.assertEqual(self.execute("kept")["result"], 7)
        with self.assertRaisesRegex(RuntimeError, "invalid \\(test failure\\).*newton_rebuild"):
            self.execute("rollout(1)")
        with self.assertRaisesRegex(RuntimeError, "newton_rebuild"):
            self.session.dispatch("step")
        self.session.dispatch("rebuild", {})
        self.assertTrue(self.session.valid)
        self.assertEqual(self.execute("rollout(2)['frames'], kept")["result"], [2, 7])

    def test_large_variable_named_result_keeps_the_cell_and_its_value(self):
        """Keep a large ``result`` variable and the cell's output (i15: the cell failed and the variable vanished)."""
        result = self.execute("samples = np.arange(30000)\nresult = samples\nprint('size', result.size)")
        self.assertTrue(result["valid"])
        self.assertIsNone(result["result"])
        self.assertEqual(result["stdout"], "size 30000\n")
        self.assertEqual(self.execute("result[:3].tolist()")["result"], [0, 1, 2])
        # A large last expression is summarized, and _ keeps it.
        summary = self.execute("samples")
        self.assertIn("30000", summary["result_repr"])
        self.assertEqual(self.execute("_[:3].tolist()")["result"], [0, 1, 2])

    def test_function_source_and_error_stack_lines_remain_available(self):
        """Retain source for persistent functions and identify errors across cell boundaries."""
        self.execute("import inspect\ndef sample():\n    return missing_value")
        source = self.execute("inspect.getsource(sample)")["result"]
        self.assertIn("def sample():", source)
        with self.assertRaises(RuntimeError):
            self.execute("sample()")
        error = self.session.dispatch("describe")["workspace"]["last_error"]
        self.assertEqual([frame["line"] for frame in error["frames"]], [1, 3])
        self.assertEqual(error["frames"][-1]["source"], "return missing_value")

    def test_automatic_expression_repr(self):
        """Display opaque expression values as a bounded summary."""
        result = self.execute("object()")
        self.assertIsNone(result["result"])
        self.assertIn("object object", result["result_repr"])
        self.assertLessEqual(len(result["result_repr"]), 16384)
        self.assertTrue(result["valid"])

    def test_automatic_display_never_calls_user_representation(self):
        """Summarize user objects without running noisy or mutating representation callbacks."""
        gravity = self.session.model.gravity.numpy().copy()
        escaped = io.StringIO()
        with contextlib.redirect_stdout(escaped), contextlib.redirect_stderr(escaped):
            result = self.execute("""class Noisy:
    def __repr__(self):
        print('escaped output' * 10000)
        model.gravity.zero_()
        raise ValueError('representation ran')
Noisy()
""")
        self.assertEqual(escaped.getvalue(), "")
        self.assertTrue(result["valid"])
        self.assertIn("Noisy", result["result_repr"])
        np.testing.assert_array_equal(self.session.model.gravity.numpy(), gravity)
        self.assertEqual(self.execute("type(_).__name__")["result"], "Noisy")
        dataclass = self.execute(
            "from dataclasses import dataclass\n@dataclass\nclass Sample:\n    count: int = 3\nSample()"
        )
        self.assertIn("Sample", dataclass["result_repr"])
        self.assertEqual(self.execute("_.count")["result"], 3)

    def test_large_array_display_is_bounded(self):
        """Summarize large arrays and strings without dense conversion."""
        result = self.execute("samples = np.zeros(1_000_000, dtype=np.complex128)\nsamples")
        self.assertTrue(result["valid"])
        self.assertIsNone(result["result"])
        self.assertIn("1000000", result["result_repr"])
        self.assertIn("complex128", result["result_repr"])
        self.assertEqual(self.execute("_.shape")["result"], [1_000_000])
        displayed = self.execute("'x' * 100000")
        self.assertIsNone(displayed["result"])
        self.assertLessEqual(len(displayed["result_repr"]), 16384)

    def test_result_serialization_never_calls_custom_dictionary_keys(self):
        """Reject custom JSON keys without calling arbitrary user string conversion."""
        self.execute("""class Key:
    def __str__(self):
        model.gravity.zero_()
        print('escaped key conversion')
        return 'key'
saved = {Key(): 1}
""")
        gravity = self.session.model.gravity.numpy().copy()
        escaped = io.StringIO()
        with contextlib.redirect_stdout(escaped):
            result = self.execute("saved")
        self.assertIsNone(result["result"])
        self.assertIn("dict length=1", result["result_repr"])
        self.assertEqual(escaped.getvalue(), "")
        self.assertTrue(self.session.valid)
        np.testing.assert_array_equal(self.session.model.gravity.numpy(), gravity)

    def test_dataclass_and_warp_kernel_definitions_span_cells(self):
        """Define normal dataclasses, Warp functions and kernels once and launch them in later cells."""
        self.execute("""from dataclasses import dataclass
@dataclass
class Settings:
    scale: float = 2.0
@wp.func
def scaled(value: float, scale: float):
    return value * scale
@wp.kernel
def apply(values: wp.array[float], scale: float):
    i = wp.tid()
    values[i] = scaled(values[i], scale)
settings = Settings()
values = wp.array([1.0, 2.0, 3.0], dtype=float, device='cpu')
""")
        self.assertTrue(self.execute("Settings.__module__")["result"].startswith("_newton_mcp_"))
        self.assertEqual(self.execute("import pickle\npickle.loads(pickle.dumps(settings)).scale")["result"], 2)
        result = self.execute(
            "wp.launch(apply, dim=3, inputs=[values, settings.scale], device='cpu')\nvalues.numpy().tolist()"
        )
        self.assertEqual(result["result"], [2, 4, 6])
        self.session.dispatch("step", {"count": 1})
        result = self.execute(
            "wp.launch(apply, dim=3, inputs=[values, settings.scale], device='cpu')\nvalues.numpy().tolist()"
        )
        self.assertEqual(result["result"], [4, 8, 12])

    def test_cell_source_cache_is_bounded_and_cleared(self):
        """Bound cached cell source, keep cells of live definitions, and clear it with the workspace."""
        self.session._MAX_CELL_SOURCES = 66
        first = self.execute("def first():\n    return 7\nfirst.__code__.co_filename")["result"]
        dropped = self.execute("def dropped():\n    return 1\ndropped.__code__.co_filename")["result"]
        self.execute("del dropped")
        self.assertTrue(linecache.getlines(first))
        for index in range(70):
            self.execute(f"counter = {index}")
        # The limit is held; the cell that still defines `first` is kept and older unused cells are dropped.
        self.assertEqual(self.session.dispatch("describe")["workspace"]["source_cells"], 66)
        self.assertTrue(linecache.getlines(first))
        self.assertFalse(linecache.getlines(dropped))
        self.assertEqual(self.execute("first()")["result"], 7)
        self.execute("first = None")
        self.execute("counter = 0")
        self.assertEqual(self.session.dispatch("describe")["workspace"]["source_cells"], 66)
        self.assertFalse(linecache.getlines(first))
        self.execute("counter = 0", reset_namespace=True)
        self.assertEqual(self.session.dispatch("describe")["workspace"]["source_cells"], 1)
        latest = self.execute("def latest():\n    return 1\nlatest.__code__.co_filename")["result"]
        module_name = self.execute("__name__")["result"]
        self.assertIn(module_name, sys.modules)
        self.session.close()
        self.assertFalse(linecache.getlines(latest))
        self.assertNotIn(module_name, sys.modules)

    def test_sources_of_definitions_held_in_containers_are_kept(self):
        """Keep the cells of functions and classes reachable only through containers, partials and instances."""
        from newton._src.mcp import shipping  # noqa: PLC0415

        self.session._MAX_CELL_SOURCES = 66
        self.execute(
            "import functools\n"
            "def controller(x):\n    return 2 * x\n"
            "def scaled(x, k):\n    return k * x\n"
            "class Policy:\n    def act(self, x):\n        return x + 1\n"
            "CONTROLLERS = {'p': controller, 'scaled': [functools.partial(scaled, k=3)]}\n"
            "POLICY = Policy()\n"
            "del controller, scaled, Policy"
        )
        unused = self.execute("def unused():\n    pass\nunused.__code__.co_filename")["result"]
        self.execute("del unused")
        for index in range(70):
            self.execute(f"counter = {index}")
        self.assertFalse(linecache.getlines(unused))
        result = self.execute(
            "import inspect\n"
            "(inspect.getsource(CONTROLLERS['p']).splitlines()[0], "
            "inspect.getsource(CONTROLLERS['scaled'][0].func).splitlines()[0], "
            "inspect.getsource(type(POLICY)).splitlines()[0])"
        )["result"]
        self.assertEqual(result, ["def controller(x):", "def scaled(x, k):", "class Policy:"])
        shipment = shipping.prepare(self.session._workspace["CONTROLLERS"]["p"], lambda name, value: False)
        self.assertEqual(shipment.label, "controller")

    def test_sys_exit_in_a_cell_rolls_back_and_keeps_the_session(self):
        """Treat SystemExit raised by a cell like any error: roll back and keep serving."""
        with self.assertRaisesRegex(
            RuntimeError, r"SystemExit\(0\) raised at line 3; the session keeps running.*rolled back"
        ):
            self.execute("import sys\nsession.dispatch('step', {'count': 2})\nsys.exit(0)")
        self.assertEqual(self.session.frame, 0)
        self.assertTrue(self.session.valid)
        self.assertEqual(self.execute("1 + 1")["result"], 2)

    def test_namespace_clear_releases_owned_warp_definitions(self):
        """Unload executor-owned Warp definitions when explicitly clearing the namespace."""
        self.execute("@wp.func\ndef increment(value: float):\n    return value + 1.0")
        module_name = self.execute("__name__")["result"]
        module = wp.get_module(module_name)
        self.assertEqual(len(module.functions), 1)
        self.execute("", reset_namespace=True)
        self.assertEqual(len(module.functions), 0)
        self.assertEqual(len(module.kernels), 0)
        self.assertEqual(len(module.structs), 0)
        self.assertEqual(len(module.execs), 0)
        self.execute("@wp.func\ndef increment(value: float):\n    return value + 2.0")
        self.assertEqual(len(module.functions), 1)
        self.session.close()
        self.assertEqual(len(module.functions), 0)

    def test_python_workspaces_are_isolated_between_sessions(self):
        """Keep independent simulation sessions from sharing Python user variables."""
        self.execute("samples = [1, 2]")
        other = SimulationSession(self.session.model, newton.solvers.SolverXPBD(self.session.model), allow_execute=True)
        self.addCleanup(other.close)
        result = other.dispatch("execute", {"code": "'samples' in globals()"})
        self.assertFalse(result["result"])
        self.assertNotEqual(
            self.execute("__name__")["result"], other.dispatch("execute", {"code": "__name__"})["result"]
        )

    def test_execution_arguments_and_permission_are_validated(self):
        """Reject invalid arguments without running code and preserve the trusted-execution opt-in."""
        revision = self.session.revision
        with self.assertRaises(ValueError):
            self.execute("value = 1", reset_namespace=1)
        with self.assertRaises(TypeError):
            self.execute("value = 1", recovery="acknowledge")
        self.assertEqual(self.session.revision, revision)
        self.session.allow_execute = False
        with self.assertRaises(PermissionError):
            self.execute("value = 1")

    def test_show_returns_inline_images(self):
        """Attach arrays, observations, and figures to the execute response as PNG images."""
        result = self.execute(
            "show(np.zeros((8, 12, 3), dtype=np.uint8), 'black')\n"
            "show(np.ones((4, 4)))\n"
            "show(session.dispatch('observe', {'width': 16, 'height': 12}))"
        )
        images = result["images"]
        self.assertEqual(len(images), 3)
        shapes = [_decode(image["image_base64"]).shape for image in images]
        self.assertEqual(shapes, [(8, 12, 3), (4, 4, 3), (12, 16, 3)])
        self.assertEqual(images[0]["label"], "black")
        self.assertTrue(np.all(_decode(images[1]["image_base64"]) == 255))
        self.assertNotIn("images", self.execute("1 + 1"))
        with self.assertRaisesRegex(RuntimeError, "At most 8 images"):
            self.execute("for _ in range(9):\n    show(np.zeros((2, 2, 3)))")
        with self.assertRaisesRegex(RuntimeError, "only available"):
            self.session.show(np.zeros((2, 2, 3)))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Requires matplotlib")
    def test_show_matplotlib_figure(self):
        """Render matplotlib figures without a display."""
        result = self.execute(
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
            "figure, axis = plt.subplots(figsize=(2, 1), dpi=50)\naxis.plot([0, 1], [1, 0])\nshow(figure)"
        )
        self.assertEqual(_decode(result["images"][0]["image_base64"]).shape, (50, 100, 3))

    def test_application_namespace_and_guide(self):
        """Expose application objects as refreshed globals and return the application guide."""
        marker = object()
        self.session.namespace["task"] = marker
        self.session.guide = "Call task.run()."
        self.assertTrue(self.execute("task is session.namespace['task']")["result"])
        self.assertNotIn("task", self.session.dispatch("describe")["workspace"]["variables"])
        self.assertEqual(self.session.dispatch("guide"), {"guide": "Call task.run()."})
        self.execute("task = None")
        self.assertTrue(self.execute("task is session.namespace['task']")["result"])

    def test_protocol_emits_shown_images_and_compact_text(self):
        """Return shown images as MCP image content with a compact JSON status."""
        from newton._src.mcp.protocol import _Protocol  # noqa: PLC0415

        session = self.session

        class Client:
            def request(self, operation, **arguments):
                return session.dispatch(operation, arguments)

        protocol = _Protocol(Client(), profile="code")
        protocol.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        response = protocol.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "newton_execute", "arguments": {"code": "show(np.zeros((2, 3, 3)))\n7"}},
            }
        )["result"]
        self.assertEqual([item["type"] for item in response["content"]], ["text", "image"])
        self.assertNotIn("structuredContent", response)
        payload = json.loads(response["content"][0]["text"])
        self.assertEqual(payload["result"], 7)
        self.assertNotIn("workspace", payload)
        self.assertNotIn("images", payload)

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "Requires optional MCP SDK for interoperability validation")
    def test_official_sdk_workspace_and_rollback(self):
        """Exercise persistent Python cells and failed-cell rollback through the real stdio MCP bridge."""
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415

        path = Path(self.directory.name) / "executor-session.json"
        errors = []
        completed = threading.Event()

        async def conversation():
            parameters = StdioServerParameters(
                command=sys.executable, args=["-m", "newton.mcp", "--connect", str(path), "--profile", "code"]
            )
            async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
                await client.initialize()
                listing = await client.list_tools()
                tool = next(tool for tool in listing.tools if tool.name == "newton_execute")
                self.assertEqual(set(tool.inputSchema["properties"]), {"code", "reset_namespace"})
                first = await client.call_tool(
                    "newton_execute",
                    {
                        "code": "import math\nvalues = [1, 2]\ndef sample():\n    return [math.fsum(values), state is session.state]"
                    },
                )
                self.assertFalse(first.isError)
                second = await client.call_tool(
                    "newton_execute", {"code": "session.dispatch('step', {'count': 1})\nsample()"}
                )
                self.assertFalse(second.isError)
                self.assertEqual(_payload(second)["result"], [3, True])
                defined = await client.call_tool(
                    "newton_execute",
                    {
                        "code": "@wp.kernel\ndef increment(data: wp.array[float]):\n    i = wp.tid()\n    data[i] = data[i] + 1.0\ndata = wp.zeros(2, dtype=float, device='cpu')"
                    },
                )
                self.assertFalse(defined.isError)
                launched = await client.call_tool(
                    "newton_execute",
                    {"code": "wp.launch(increment, dim=2, inputs=[data], device='cpu')\ndata.numpy().tolist()"},
                )
                self.assertFalse(launched.isError)
                self.assertEqual(_payload(launched)["result"], [1, 1])
                failed = await client.call_tool(
                    "newton_execute", {"code": "saved = 19\nsession.dispatch('step', {'count': 2})\nmissing_name"}
                )
                self.assertTrue(failed.isError)
                self.assertIn("line 3", failed.content[0].text)
                self.assertIn("rolled back from t=", failed.content[0].text)
                inspected = await client.call_tool("newton_execute", {"code": "saved, session.frame"})
                self.assertEqual(_payload(inspected)["result"], [19, 1])
                # Compact responses omit the flag for valid scenes.
                self.assertNotIn("valid", _payload(inspected))
                cleared = await client.call_tool(
                    "newton_execute", {"code": "'saved' in globals()", "reset_namespace": True}
                )
                self.assertFalse(_payload(cleared)["result"])

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
            self.assertTrue(completed.is_set(), "Official MCP workspace client did not finish")
            thread.join(timeout=1)
        if errors:
            raise errors[0]


class TestMcpCellSources(unittest.TestCase):
    def test_registration_is_idempotent_and_keeps_live_definitions(self):
        """Register cell source once under a stable name and keep only cells that live definitions need."""
        names = [f"<_newton_mcp_test:cell-{i}>" for i in range(5)]
        for index, name in enumerate(names):
            register_cell_source(name, f"def f{index}():\n    return {index}")
        entry = linecache.cache[names[0]]
        register_cell_source(names[0], "def f0():\n    return 0")
        self.assertIs(linecache.cache[names[0]], entry)
        namespace = {}
        exec(compile("def f1():\n    return 1", names[1], "exec"), namespace)
        self.assertEqual(retain_cell_sources(names, namespace, recent=2, maximum=5), names)
        kept = retain_cell_sources(names, namespace, recent=2, maximum=3)
        self.assertEqual(kept, [names[1], names[3], names[4]])
        self.assertFalse(linecache.getlines(names[0]))
        self.assertTrue(linecache.getlines(names[1]))
        for name in kept:
            linecache.cache.pop(name, None)


if __name__ == "__main__":
    unittest.main()
