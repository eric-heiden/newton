# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental live Newton simulation tools and stdio MCP attachment.

.. experimental::

    This entire module may change without a deprecation period. It requires
    explicit embedding in the simulation process and owner-thread request
    pumping. Trusted Python is opt-in and is not a security sandbox.

Run ``python -m newton.mcp --connect session.json`` as a stdio MCP server, and
``python -m newton.mcp host example.py --connection-file session.json`` to serve
an unmodified Newton example script live.
Add ``--profile code`` to advertise only describe, execute, observe, filmstrip, and rebuild,
or ``--profile lean`` for only execute and rebuild (observation stays available from Python);
this changes tool presentation, not permissions.
"""

from typing import TYPE_CHECKING

__all__ = ["ExampleHost", "JobQueue", "SimulationClient", "SimulationServer", "SimulationSession", "WorkerPool"]

if TYPE_CHECKING:
    from ._src.mcp.host import ExampleHost
    from ._src.mcp.jobs import JobQueue
    from ._src.mcp.session import SimulationSession
    from ._src.mcp.transport import SimulationClient, SimulationServer
    from ._src.mcp.workers import WorkerPool


def __getattr__(name: str):
    if name == "SimulationSession":
        from ._src.mcp.session import SimulationSession  # noqa: PLC0415

        return SimulationSession
    if name in {"SimulationClient", "SimulationServer"}:
        from ._src.mcp.transport import SimulationClient, SimulationServer  # noqa: PLC0415

        return {"SimulationClient": SimulationClient, "SimulationServer": SimulationServer}[name]
    if name == "ExampleHost":
        from ._src.mcp.host import ExampleHost  # noqa: PLC0415

        return ExampleHost
    if name == "JobQueue":
        from ._src.mcp.jobs import JobQueue  # noqa: PLC0415

        return JobQueue
    if name == "WorkerPool":
        from ._src.mcp.workers import WorkerPool  # noqa: PLC0415

        return WorkerPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["host"]:
        from ._src.mcp.host import main

        main(sys.argv[2:])
    else:
        from ._src.mcp.protocol import main

        main()
