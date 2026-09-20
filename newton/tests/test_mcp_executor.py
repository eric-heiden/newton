# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise persistent trusted Python cells and explicit failure recovery."""

import asyncio
import contextlib
import importlib.util
import io
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
from newton.mcp import SimulationServer, SimulationSession


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

    def test_last_expression_and_explicit_result_do_not_leak(self):
        """Return the last expression and preserve explicit result precedence without stale output."""
        self.assertEqual(self.execute("2 + 3")["result"], 5)
        self.assertEqual(self.execute("_ + 1")["result"], 6)
        self.assertEqual(self.execute("result = 9\n42")["result"], 9)
        self.assertIsNone(self.execute("value = 12")["result"])
        self.assertEqual(self.execute("value")["result"], 12)
        self.assertIsNone(self.execute("result = None\n13")["result"])

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

    def test_rebuild_and_explicit_namespace_reset_clear_user_variables(self):
        """Discard scene aliases and closures on replacement and explicit workspace reset."""
        result = self.execute("old_state = state\nhelper = lambda captured=state: captured\nvalue = 7")
        generation = result["workspace"]["generation"]
        self.session.replace(self.session.model, newton.solvers.SolverXPBD(self.session.model))
        result = self.execute("[name in globals() for name in ['old_state', 'helper', 'value']]")
        self.assertEqual(result["result"], [False, False, False])
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

    def test_failure_preserves_workspace_and_requires_explicit_recovery(self):
        """Preserve partial Python work without treating exploration errors as proof of safety."""
        with self.assertRaisesRegex(RuntimeError, "line 3"):
            self.execute("samples = [1, 2]\nprint('before failure')\nmissing_name")
        status = self.session.dispatch("describe")
        self.assertFalse(status["valid"])
        self.assertTrue(status["paused"])
        self.assertTrue(status["requires_rebuild"])
        self.assertEqual(status["workspace"]["last_error"]["type"], "NameError")
        self.assertIn("samples", status["workspace"]["variables"])
        with self.assertRaises(RuntimeError):
            self.execute("samples")
        inspected = self.execute("samples", recovery="inspect")
        self.assertEqual(inspected["result"], [1, 2])
        self.assertFalse(inspected["valid"])
        self.assertTrue(inspected["requires_rebuild"])
        with self.assertRaises(RuntimeError):
            self.session.dispatch("step")
        recovered = self.execute("samples.append(3)", recovery="acknowledge")
        self.assertTrue(recovered["valid"])
        self.assertTrue(recovered["paused"])
        self.assertFalse(recovered["requires_rebuild"])
        self.assertEqual(self.execute("samples")["result"], [1, 2, 3])
        self.assertEqual(self.session.dispatch("step")["frame"], 1)

    def test_recovery_never_rolls_back_and_failed_acknowledgement_stays_invalid(self):
        """Keep partial mutations visible and require successful explicit acknowledgement."""
        with self.assertRaises(RuntimeError):
            self.execute("model.gravity.zero_()\nraise ValueError('after mutation')")
        np.testing.assert_array_equal(self.session.model.gravity.numpy(), 0)
        with self.assertRaises(RuntimeError):
            self.execute("raise ValueError('repair failed')", recovery="acknowledge")
        self.assertFalse(self.session.valid)
        self.assertTrue(self.session.dispatch("describe")["requires_rebuild"])
        result = self.execute("model.gravity.numpy().tolist()", recovery="inspect")
        self.assertEqual(result["result"], [[0, 0, 0]])
        self.assertFalse(result["valid"])

    def test_result_budget_does_not_destroy_saved_work(self):
        """Inspect a smaller slice of an already computed result without repeating the computation."""
        with self.assertRaisesRegex(RuntimeError, "Python completed"):
            self.execute("samples = np.arange(30000)\nresult = samples")
        self.assertTrue(self.session.valid)
        self.assertEqual(self.execute("_[:3].tolist()")["result"], [0, 1, 2])
        self.assertEqual(self.execute("samples.size")["result"], 30000)

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

    def test_automatic_expression_repr_and_strict_explicit_result(self):
        """Display opaque expression values while retaining strict explicit JSON result validation."""
        result = self.execute("object()")
        self.assertIsNone(result["result"])
        self.assertIn("object object", result["result_repr"])
        self.assertLessEqual(len(result["result_repr"]), 16384)
        self.assertTrue(result["valid"])
        with self.assertRaisesRegex(RuntimeError, "Python completed"):
            self.execute("result = object()")
        self.assertTrue(self.session.valid)

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

    def test_large_array_display_is_bounded_and_explicit_result_fails_early(self):
        """Summarize large arrays and reject oversized explicit results before dense conversion."""
        result = self.execute("samples = np.zeros(1_000_000, dtype=np.complex128)\nsamples")
        self.assertTrue(result["valid"])
        self.assertIsNone(result["result"])
        self.assertIn("1000000", result["result_repr"])
        self.assertIn("complex128", result["result_repr"])
        self.assertEqual(self.execute("_.size")["result"], 1_000_000)
        with self.assertRaisesRegex(RuntimeError, "component budget"):
            self.execute("result = samples")
        self.assertTrue(self.session.valid)
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
        with contextlib.redirect_stdout(escaped), self.assertRaisesRegex(RuntimeError, "Python completed"):
            self.execute("result = saved")
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
        """Bound cached cell source and remove it when clearing or closing the workspace."""
        first = self.execute("def first():\n    return 7\nfirst.__code__.co_filename")["result"]
        self.assertTrue(linecache.getlines(first))
        for index in range(70):
            self.execute(f"counter = {index}")
        self.assertEqual(self.session.dispatch("describe")["workspace"]["source_cells"], 64)
        self.assertFalse(linecache.getlines(first))
        self.assertEqual(self.execute("first()")["result"], 7)
        self.execute("counter = 0", reset_namespace=True)
        self.assertEqual(self.session.dispatch("describe")["workspace"]["source_cells"], 1)
        latest = self.execute("def latest():\n    return 1\nlatest.__code__.co_filename")["result"]
        module_name = self.execute("__name__")["result"]
        self.assertIn(module_name, sys.modules)
        self.session.close()
        self.assertFalse(linecache.getlines(latest))
        self.assertNotIn(module_name, sys.modules)

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

    def test_recovery_arguments_and_execution_permission_are_validated(self):
        """Reject invalid recovery choices and preserve the trusted-execution opt-in."""
        revision = self.session.revision
        for options in ({"recovery": "guess"}, {"recovery": []}, {"reset_namespace": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.execute("value = 1", **options)
        self.assertEqual(self.session.revision, revision)
        self.session.allow_execute = False
        with self.assertRaises(PermissionError):
            self.execute("value = 1", recovery="acknowledge")

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "Requires optional MCP SDK for interoperability validation")
    def test_official_sdk_workspace_and_recovery(self):
        """Exercise persistent Python cells and explicit recovery through the real stdio MCP bridge."""
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
                self.assertEqual(tool.inputSchema["properties"]["recovery"]["enum"], ["none", "inspect", "acknowledge"])
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
                self.assertEqual(second.structuredContent["result"], [3, True])
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
                self.assertEqual(launched.structuredContent["result"], [1, 1])
                failed = await client.call_tool("newton_execute", {"code": "saved = 19\nmissing_name"})
                self.assertTrue(failed.isError)
                self.assertIn("line 2", failed.content[0].text)
                inspected = await client.call_tool("newton_execute", {"code": "saved", "recovery": "inspect"})
                self.assertEqual(inspected.structuredContent["result"], 19)
                self.assertFalse(inspected.structuredContent["valid"])
                acknowledged = await client.call_tool("newton_execute", {"code": "", "recovery": "acknowledge"})
                self.assertTrue(acknowledged.structuredContent["valid"])
                self.assertTrue(acknowledged.structuredContent["paused"])
                cleared = await client.call_tool(
                    "newton_execute", {"code": "'saved' in globals()", "reset_namespace": True}
                )
                self.assertFalse(cleared.structuredContent["result"])

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


if __name__ == "__main__":
    unittest.main()
