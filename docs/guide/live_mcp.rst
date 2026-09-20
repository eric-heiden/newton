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
rejects ordinary Python execution while invalid. Explicit trusted recovery
modes remain available through ``newton_execute`` as described below.

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
solver data inconsistent. Rebuilding restores known bindings; an explicitly
acknowledged Python recovery is also available to callers that can verify or
repair coherence, as described below. ``replace()`` accepts a
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

Persistent Python workspace
---------------------------

``allow_execute=True`` enables unrestricted synchronous Python cells in one
persistent workspace per simulation session. Imports, variables, functions,
and classes survive subsequent ``execute`` calls and physical ``reset`` or
checkpoint ``restore`` operations. A registered Python module and bounded cell
source cache support ordinary dataclasses, ``inspect.getsource()`` for recent
functions, and ``@wp.func`` / ``@wp.kernel`` definitions that can be launched
in later calls. IPython magics and top-level ``await`` are not implemented.

The reserved names ``session``, ``model``, ``solver``, ``state``, ``state_next``,
``control``, ``contacts``, ``viewer``, ``wp``, and ``np`` refresh before each call
and after managed state changes. Functions that read these globals see the
current state after an odd number of buffer swaps, including steps initiated
inside the same cell. User-created aliases and captured default arguments are
ordinary Python references and do not automatically follow buffer swaps.

.. code-block:: python

   client.request("execute", code="""import math
   samples = []
   def current_height():
       return float(state.body_q.numpy()[0, 2])
   """)
   client.request("step", count=1)
   result = client.request("execute", code="""samples.append(current_height())
   math.fsum(samples) / len(samples)
   """)
   print(result["result"])

The last expression produces the returned value. Explicit ``result = ...``
retains its previous behavior and takes precedence; it is cleared before the
next call so an old result cannot leak into a new response. ``_`` holds the
last non-``None`` value, including a value too large to return. JSON-compatible
values use the ``result`` field. An opaque or oversized last expression uses a
bounded ``result_repr`` summary instead, without calling user ``__repr__`` or
converting large arrays to lists. Inspect selected attributes or slices of ``_``
for details. Explicit ``result`` assignments remain strict about JSON
serialization: use built-in scalar keys and containers, or NumPy values.
Captured stdout/stderr is limited to 16384 characters, serialized JSON results
to 65536 characters, and result conversion to 16384 components and 64 nesting
levels. Arrays above the component limit or 1 MiB are rejected before conversion.
If Python completes but its result cannot be returned, validity is unchanged
and the error says execution completed.
Inspect ``_[:10]`` or another saved variable rather than repeating a mutation.

``execute(reset_namespace=True, code=...)`` clears Python variables, imports,
functions, classes, previous results, and source history before running the
new cell. It does not reset or repair simulation state. Compilation happens
before this clear, so a syntax error preserves the existing workspace.
Scene ``replace``/``rebuild`` always clears the workspace to drop old scene
aliases and closures. Closing the session removes its Python module registration
and cached cell sources. Clearing or closing unloads the workspace's Warp module
and drops its registered definitions. Escaped references, separately named Warp
modules, and captured CUDA graphs remain application-owned; Warp's ordinary
module metadata and disk kernel cache are not globally erased.

The source cache retains at most 64 cells of at most 65536 characters each.
Older Python functions still execute, but source inspection may be unavailable
after their cell is evicted. ``describe`` and successful execution responses
include workspace generation, cell count, a bounded list of variable names,
and the most recent execution diagnostic. Diagnostics identify the exception,
cell, source line, and up to eight user-code stack frames.

Explicit recovery after an execution error
------------------------------------------

A syntax or compilation error runs no Python and leaves scene validity
unchanged. A runtime exception may occur after partial mutations. It preserves
Python variables and imports that were created before the error, pauses
playback, and sets ``valid=False`` and ``requires_rebuild=True``. There is no
automatic rollback, and the exception type alone is not evidence that model
and solver data are coherent. A ``NameError`` can occur after a successful
mutation just as a solver error can.

The ``recovery`` argument has three explicit choices:

* ``"none"`` is the default and rejects execution while the session is invalid.
* ``"inspect"`` permits trusted Python diagnosis or repair while invalid. It
  does not automatically restore validity or resume playback. Structured
  simulation stepping remains blocked until recovery is acknowledged or the
  scene is rebuilt.
* ``"acknowledge"`` explicitly accepts the caller's responsibility for checking
  or repairing model/solver coherence. If the cell completes successfully,
  it clears invalidation and leaves playback paused. A failed cell remains
  invalid. This does not reset arrays, notify solvers, clear solver caches,
  or prove that arbitrary mutations were repaired.

For a known analysis-only error, inspect the retained variables and acknowledge
only after verifying that no simulation mutation needs repair:

.. code-block:: python

   client.request("execute", code="len(samples)", recovery="inspect")
   client.request("execute", code="", recovery="acknowledge")

After an actual model mutation, diagnosis and repair may require restoring
saved arrays and notifying or rebuilding the solver. Use the application's
``rebuild`` callback when coherence is uncertain. Trusted recovery remains
unrestricted Python and requires the same explicit ``allow_execute=True``
opt-in as ordinary execution; it is not a read-only sandbox.

Queue waiting has a deadline and expired pending mutations are cancelled.
Once an operation starts, the client waits for completion because running
Python, Warp, and GL cannot safely be preempted. A connection failure during
execution has an unknown outcome; do not automatically retry mutations.

The transport uses TCP and ordinary Python threads for portability. Runtime,
CPU/CUDA sensor rendering, attached CPU ViewerGL capture, and official SDK
interoperability have been tested on Linux. Windows CPU runtime, sensor
rendering, and official SDK interoperability passed the
`Windows CI tests <https://github.com/eric-heiden/newton/actions/runs/35434465266>`_.
Windows CUDA execution and attached ViewerGL capture remain unverified.
