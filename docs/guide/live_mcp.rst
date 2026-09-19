.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Live simulation tools
=====================

.. experimental::

   The entire :mod:`newton.mcp` module and its tools may change without a
   deprecation period. A simulation must explicitly embed the session; the
   bridge cannot attach to an arbitrary uninstrumented Python process.

A :class:`newton.mcp.SimulationSession` holds a live model, solver, current
state, output state, controls, collision pipeline, and optional viewer.
Requests execute on the thread that constructed the session. Network threads
only authenticate and queue requests, so Warp and OpenGL stay on their owner
thread. The implementation adds no dependency beyond Newton's dependencies
and the Python standard library.

Start an embedded session
-------------------------

.. code-block:: python

   import newton
   import warp as wp
   from newton.mcp import SimulationServer, SimulationSession

   builder = newton.ModelBuilder()
   body = builder.add_body(
       xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
   )
   builder.add_shape_sphere(body, radius=0.1)
   builder.add_ground_plane()
   model = builder.finalize()
   solver = newton.solvers.SolverXPBD(model)
   session = SimulationSession(model, solver, dt=1.0 / 240.0)
   with SimulationServer(session, connection_file="session.json"):
       session.run()

The session starts paused. ``run()`` services requests while paused and steps
while playing. Its playback is not paced to wall-clock time. A GUI application
can call ``session.pump()`` each iteration, inspect ``session.paused``, and use
``session.dispatch("step")`` to advance. Keep pumping while paused.

An existing application may pass its state buffers and a
``step_callback(session, dt)``. That callback performs collision detection,
control updates, the solver step, and buffer swaps by assigning
``session.state`` and ``session.state_next``. The session increments its time
and frame counters afterwards. A ``reset_callback(session)`` restores
application-owned controller histories or metrics after session reset.
Applications that capture CUDA graphs remain responsible for graph lifecycle.
Parameter edits retain array storage; replacing scene topology requires
rebuilding graphs and every dependent binding.

Connect an MCP client
---------------------

Configure a stdio MCP server with this command, using an absolute connection
file path when the MCP client's working directory differs:

.. code-block:: bash

   uv run -m newton.mcp --connect /absolute/path/session.json

The default ``--profile full`` advertises every structured tool. Add
``--profile code`` to advertise only ``newton_describe``, ``newton_execute``,
``newton_observe``, and ``newton_rebuild``. This reduces tool-schema context for
clients that prefer Python. The profile changes presentation, not permissions:
trusted execution still requires ``allow_execute=True``. Python can call
``session.dispatch(operation, arguments)`` for every structured operation listed
by ``describe``. Rebuild remains a separate tool because an invalid session
rejects Python execution until the scene has been rebuilt.

This adapter implements newline-delimited JSON-RPC initialization and tools,
following the `MCP stdio transport specification
<https://modelcontextprotocol.io/specification/2025-11-25/basic/transports>`_.
Its authenticated TCP connection to the embedded session is an internal
loopback attachment protocol, not an HTTP MCP endpoint. The official Python
MCP SDK is used only for optional interoperability tests.

Keep ``session.json`` private: its authentication token grants access to the
session's enabled operations. The server creates a new file and refuses to
overwrite an existing one. POSIX permissions are owner-only; on Windows use a
private parent directory with appropriate ACLs. Close the server to remove its
connection file, and close the session on its owner thread to reject queued
requests. A crashed server may leave a stale connection file.

Python applications can use :class:`newton.mcp.SimulationClient` from a
separate thread or process:

.. code-block:: python

   from newton.mcp import SimulationClient

   client = SimulationClient("session.json")
   client.request("describe")
   client.request("query", root="state", field="body_q", offset=0, limit=4)
   client.request("edit", patches=[
       {"root": "model", "field": "shape_material_mu", "indices": [0], "values": [0.8]}
   ])
   client.request("step", count=120)

Call ``describe`` to discover solver entries, supported parameter fields, and
budgets. ``query`` uses model frequency metadata for world, body, shape, and
joint selection, including joint coordinate and DOF rows. ``world=-1`` selects
global entities. Coupled entry paths use public solver/view/state accessors;
indices follow that entry's view. Unfiltered Warp queries gather only the
requested page. Filter maps and edit host transfers are limited to two million
rows/components, query pages to 256 rows and 4096 components, and state/control
snapshots to 256 MiB each. Numeric field units follow the corresponding Newton
model, state, and control documentation.

Mutation and observation semantics
----------------------------------

All edit patches validate before writes begin. Values must be finite and match
the selected row shape exactly; scalar broadcasting is unsupported. The model
edit list excludes topology and index arrays. Positive mass edits scale inertia
and update inverse mass/inertia. Notifications reach the top-level solver once;
coupled solvers forward them to children. ``expected_revision`` can reject a
stale edit before applying it.

``reset`` restores initial state, controls, session time, and solver/contact
caches while retaining tuned model parameters. Named checkpoints store public
state/control arrays and time. They do not capture every hidden solver state,
so restoring a checkpoint resets solver caches and does not promise bitwise
replay. Reset and restore clear contact buffers and generation metadata without
running collision detection. Use ``contacts(refresh=True)`` or ``collide`` to
regenerate diagnostic contacts, including before an observation with contact
overlays. The default step path regenerates its contacts before physics;
application callbacks remain responsible for their own collision workflow.
A failed physics step pauses and invalidates the session until reset. Playback
keeps servicing requests and exposes a bounded ``last_error`` in status.
An arbitrary Python failure or model-notification failure may leave model and
solver data inconsistent and requires a rebuilt scene. ``replace()`` accepts a
new model and solver; a registered rebuild callback can expose this through
MCP without restarting the process.

``contacts`` distinguishes generated collision-pipeline records from coupled
entry records. It reports counts, capacities, possible overflow, generation
frame/revision when known, world support and surface points, normals, and signed
surface gaps in meters. Set ``refresh=True`` to regenerate top-level contacts
at the current state. Solver-native contacts that are not exposed as Newton
``Contacts`` are not converted into these rows. ``include_global=True`` includes
global-global contacts when filtering a local world.

``observe`` defaults to ``SensorTiledCamera`` and needs no GL context. Camera
positions and targets use world coordinates in meters. ``pose`` contains
``[x, y, z, qx, qy, qz, qw]`` with local -Z forward and +Y up; ``fov_y`` is in
degrees. Select a scene with ``world_id`` and request color, albedo, depth,
forward depth, normals, or shape IDs. The MCP response includes a PNG image and
metadata; ``raw=True`` writes numeric artifacts. Pixel picking accepts a list
of ``[x, y]`` image coordinates. ``backend="viewer"`` captures an attached
``ViewerGL`` on its owner thread and supports its wireframe mode. Unsupported
backend settings fail explicitly. Recording writes bounded PNG sequences with
simulation timestamps and a manifest, without requiring ffmpeg.

``allow_execute=True`` enables unrestricted trusted Python with ``session``,
``model``, ``solver``, ``state``, ``control``, ``wp``, and ``np`` globals. Assign a
JSON-compatible ``result`` to return it. Captured text and serialized result
sizes are bounded; this is not a sandbox or an execution-time limit. If Python
completes but its result is unsupported or too large, the scene remains valid
and the error explicitly reports that execution completed. Query a smaller
result instead of repeating the mutation. Queue
waiting has a deadline and expired pending mutations are cancelled. Once an
operation starts, the client waits for completion because running Python,
Warp, and GL cannot safely be preempted. A connection failure during execution
has an unknown outcome; do not automatically retry mutations.

The transport uses TCP and ordinary Python threads for portability. Runtime,
CPU/GPU rendering, and official SDK interoperability have been tested on Linux;
Windows execution has not been verified in this prototype.
