# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate the third-party IPython condition and shared simulation behavior."""

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .ipython_session import IpythonSession
from .run_agents import _action_failures, prepare
from .scenarios import MENAGERIE, ROOT, Scenario


class TestIpythonPreparation(unittest.TestCase):
    """Check that the IPython agent receives its actual interface instructions."""

    def test_ipython_prompt_uses_kernel_tools(self):
        """Configure the existing kernel without directing the agent to Newton MCP."""
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare(Path(directory) / "trial", "panda", "ipython", 0, 600)
            self.assertIn("connect_to_kernel", prepared["prompt"])
            self.assertIn("execute_code", prepared["prompt"])
            self.assertNotIn("newton MCP server is configured", prepared["prompt"])
            self.assertIn("30 seconds", prepared["prompt"])

    def test_unknown_condition_rejected(self):
        """Reject misspelled condition names before creating an experiment."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                prepare(Path(directory) / "trial", "panda", "ipythno", 0, 600)

    def test_corrected_control_is_labeled(self):
        """Disclose the optional IPython correction without changing the task physics."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = prepare(root / "upstream", "panda", "ipython", 0, 600)
            corrected = prepare(root / "corrected", "panda", "ipython_fixed", 0, 600)
            self.assertEqual(upstream["spec"]["thresholds"], corrected["spec"]["thresholds"])
            self.assertEqual(upstream["spec"]["initial"], corrected["spec"]["initial"])
            self.assertIn("300-second execution wait", corrected["prompt"])
            self.assertEqual(corrected["spec"]["ipython_server_variant"], "reply_correlation_fix")

    def test_textual_ipython_error_is_counted(self):
        """Retain upstream textual errors even when MCP reports a completed tool call."""
        result = _action_failures(
            [
                {"id": "command", "type": "command_execution", "exit_code": 1},
                {
                    "id": "upstream",
                    "type": "mcp_tool_call",
                    "status": "completed",
                    "result": {"content": [{"type": "text", "text": "❌ Execution failed: "}]},
                },
                {"id": "ok", "type": "mcp_tool_call", "result": {"content": [{"type": "text", "text": "42"}]}},
            ]
        )
        self.assertEqual(result["failed_command_ids"], ["command"])
        self.assertEqual(result["mcp_error_indication_ids"], ["upstream"])


@unittest.skipUnless(importlib.util.find_spec("ipython_mcp"), "Requires separately installed ipython-mcp")
class TestIpythonKernel(unittest.TestCase):
    """Exercise upstream MCP through a real kernel with the shared Newton app."""

    def test_real_mcp_namespace_and_rollout_parity(self):
        """Preserve Python functions and match fresh Newton physics through upstream MCP."""
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415

        if not (MENAGERIE / "franka_emika_panda/panda_nohand.xml").exists():
            self.skipTest("Requires Menagerie Panda")
        config = {"kp": 1500.0, "kd": 50.0}
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "config.py").write_text("CONFIG = " + repr(config) + "\n")
            host = IpythonSession(workspace, "panda", 0)
            host.start()

            async def run():
                parameters = StdioServerParameters(
                    command=sys.executable, args=["-m", "ipython_mcp.server"], cwd=str(ROOT)
                )
                async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
                    await client.initialize()
                    listing = await client.list_tools()
                    self.assertIn("execute_code", {tool.name for tool in listing.tools})
                    reply = await client.call_tool("connect_to_kernel", {"connection_file": str(host.connection_file)})
                    self.assertIn("Connected", reply.content[0].text)
                    await client.call_tool(
                        "execute_code",
                        {
                            "code": "import threading\nowner_id = threading.get_ident()\ndef mass_sum():\n    return float(model.body_mass.numpy().sum())"
                        },
                    )
                    reply = await client.call_tool(
                        "execute_code",
                        {
                            "code": "@wp.kernel\ndef scale(values: wp.array[float]):\n    i = wp.tid()\n    values[i] *= 2.0\nvalues = wp.array([1.0, 2.0, 3.0], dtype=float, device='cpu')"
                        },
                    )
                    self.assertFalse(any(c.type == "text" and c.text.startswith("❌") for c in reply.content))
                    reply = await client.call_tool(
                        "execute_code",
                        {
                            "code": "wp.launch(scale, dim=3, inputs=[values], device='cpu')\nassert values.numpy().tolist() == [2.0, 4.0, 6.0]\nprint('kernel persisted')"
                        },
                    )
                    self.assertIn("kernel persisted", "\n".join(c.text for c in reply.content if c.type == "text"))
                    reply = await client.call_tool(
                        "execute_code",
                        {
                            "code": "assert threading.get_ident() == owner_id\nassert mass_sum() > 0\nassert model is session.model\nsession.dispatch('step', {'count': 1500})\nimport json\nprint(json.dumps(session.scenario.metrics()))"
                        },
                    )
                    lines = "\n".join(c.text for c in reply.content if c.type == "text").splitlines()
                    metrics = next(json.loads(line) for line in reversed(lines) if line.startswith('{"scenario"'))
                    reply = await client.call_tool(
                        "execute_code", {"code": "assert state is session.state\nprint('bindings refreshed')"}
                    )
                    self.assertIn("bindings refreshed", reply.content[0].text)
                    return metrics

            try:
                actual = asyncio.run(run())
            finally:
                host.close()
            expected = Scenario("panda", config).rollout()
            for key in (*expected["thresholds"], "frames", "sample_count", "success"):
                np.testing.assert_equal(actual[key], expected[key], err_msg=key)
            self.assertEqual(len((workspace / "live_rollouts.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((workspace / "process_events.jsonl").read_text().splitlines()), 1)
