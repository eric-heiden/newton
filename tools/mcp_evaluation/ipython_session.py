# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Host the common application in an IPython kernel for the upstream MCP server.

This module bootstraps an ordinary kernel; agent execution uses the separately
installed, unmodified ``ipython-mcp`` package. Newton's MCP transport and Python
executor are not involved in this condition.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from .recording import digest
from .scenarios import ROOT


def initialize_application(workspace: str, name: str, variant: int, reference_file: str | None = None) -> dict:
    """Construct the shared simulation on the kernel's execution thread."""
    import runpy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415
    from IPython import get_ipython

    from .rollout import make_scenario, make_session  # noqa: PLC0415

    root = Path(workspace)
    with (root / "process_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "simulation_process_start",
                    "pid": os.getpid(),
                    "condition": "ipython",
                    "wall_time_unix": time.time(),
                }
            )
            + "\n"
        )
    config = (
        json.loads((root / "config.json").read_text())
        if (root / "config.json").exists()
        else runpy.run_path(str(root / "config.py"))["CONFIG"]
    )
    scenario = make_scenario(
        name, config, variant=variant, reference_file=None if reference_file is None else Path(reference_file)
    )
    scenario.rollout_log = root / "live_rollouts.jsonl"
    session = make_session(scenario, root / "observations")
    shell = get_ipython()

    def refresh_bindings(_result=None):
        shell.user_ns.update(
            {
                "session": session,
                "scenario": session.scenario,
                "model": session.model,
                "solver": session.solver,
                "state": session.state,
                "state_next": session.state_next,
                "control": session.control,
                "contacts": session.contacts,
                "np": np,
                "wp": wp,
            }
        )

    # Application-owned bindings follow state swaps and explicit rebuilds.
    shell.events.register("pre_run_cell", refresh_bindings)
    shell.events.register("post_run_cell", refresh_bindings)
    refresh_bindings()
    (root / "provenance.json").write_text(json.dumps(scenario.provenance, indent=2) + "\n")
    ready = {"pid": os.getpid(), "build_seconds": scenario.build_seconds, "condition": "ipython"}
    (root / "server_ready.json").write_text(json.dumps(ready, indent=2) + "\n")
    return ready


class IpythonSession:
    """Own one separately installed IPython kernel and its startup log."""

    def __init__(self, workspace: Path, name: str, variant: int, *, reference_file: Path | None = None):
        self.workspace = workspace.resolve()
        self.name = name
        self.variant = variant
        self.reference_file = reference_file
        self.connection_file = self.workspace / "kernel.json"
        self.manager = None
        self.client = None
        self.log = None
        self.startup_seconds = 0.0

    def start(self) -> None:
        """Launch and initialize the kernel using this experiment's Python environment."""
        from jupyter_client import KernelManager  # noqa: PLC0415

        before = time.perf_counter()
        self.log = (self.workspace / "server.log").open("w")
        self.manager = KernelManager(connection_file=str(self.connection_file))
        self.manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            self.manager.start_kernel(cwd=str(self.workspace), env=env, stdout=self.log, stderr=self.log)
            self.client = self.manager.blocking_client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=60)
            arguments = repr(
                (
                    str(self.workspace),
                    self.name,
                    self.variant,
                    None if self.reference_file is None else str(self.reference_file),
                )
            )
            code = (
                "from tools.mcp_evaluation.ipython_session import initialize_application\ninitialize_application(*"
                + arguments
                + ")"
            )

            def capture(message):
                content = message.get("content", {})
                self.log.write(content.get("text", ""))
                if message.get("msg_type") == "error":
                    self.log.write("\n".join(content.get("traceback", [])) + "\n")
                self.log.flush()

            reply = self.client.execute_interactive(code, timeout=120, output_hook=capture)
            if reply["content"]["status"] != "ok" or not (self.workspace / "server_ready.json").exists():
                raise RuntimeError("IPython application initialization failed; inspect server.log")
            self.startup_seconds = time.perf_counter() - before
            self.client.stop_channels()
            self.client = None
            (self.workspace / "ipython_environment.json").write_text(json.dumps(self.environment(), indent=2) + "\n")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def environment() -> dict:
        """Record actual third-party server code and dependency versions."""
        import ipython_mcp  # noqa: PLC0415

        source = Path(ipython_mcp.__file__).parent / "server.py"
        return {
            "server_path": str(source),
            "server_sha256": digest(source),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("ipython-mcp", "ipython", "ipykernel", "jupyter-client", "mcp", "pyzmq")
            },
            "python": sys.version,
        }

    def close(self) -> None:
        """Stop the kernel before independent verification or another timed trial."""
        if self.client is not None:
            self.client.stop_channels()
            self.client = None
        if self.manager is not None:
            try:
                self.manager.shutdown_kernel(now=True)
            finally:
                self.manager.cleanup_resources()
                self.manager = None
        if self.log is not None:
            self.log.close()
            self.log = None
