.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.mcp
==========

Experimental live Newton simulation tools and stdio MCP attachment.

.. experimental::

    This entire module may change without a deprecation period. It requires
    explicit embedding in the simulation process and owner-thread request
    pumping. Trusted Python is opt-in and is not a security sandbox.

Run ``python -m newton.mcp --connect session.json`` as a stdio MCP server, and
``python -m newton.mcp host example.py --connection-file session.json`` to serve
an unmodified Newton example script live.
Add ``--profile code`` to advertise only describe, execute, observe, filmstrip,
and rebuild, or ``--profile lean`` for only execute and rebuild; this changes
tool presentation, not permissions.

.. py:module:: newton.mcp
.. currentmodule:: newton.mcp

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ExampleHost
   SimulationClient
   SimulationServer
   SimulationSession
   WorkerPool
