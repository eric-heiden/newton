# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental live Newton simulation tools and stdio MCP attachment.

.. experimental::

    This entire module may change without a deprecation period. It requires
    explicit embedding in the simulation process and owner-thread request
    pumping. Trusted Python is opt-in and is not a security sandbox.

Run ``python -m newton.mcp --connect session.json`` as a stdio MCP server.
Add ``--profile code`` to advertise only describe, execute, observe, and rebuild;
this changes tool presentation, not permissions.
"""

from typing import TYPE_CHECKING

__all__ = ["SimulationClient", "SimulationServer", "SimulationSession"]

if TYPE_CHECKING:
    from ._src.mcp.session import SimulationSession
    from ._src.mcp.transport import SimulationClient, SimulationServer


def __getattr__(name: str):
    if name == "SimulationSession":
        from ._src.mcp.session import SimulationSession  # noqa: PLC0415

        return SimulationSession
    if name in {"SimulationClient", "SimulationServer"}:
        from ._src.mcp.transport import SimulationClient, SimulationServer  # noqa: PLC0415

        return {"SimulationClient": SimulationClient, "SimulationServer": SimulationServer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from ._src.mcp.protocol import main

    main()
