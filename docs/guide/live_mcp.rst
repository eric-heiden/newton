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

Host an example script
----------------------

Any script that follows the Newton example convention (an ``Example(viewer,
args)`` class with ``model``, ``solver``, ``state_0``/``state_1``, ``control``,
and a per-frame ``step()``) can be served without changes:

.. code-block:: bash

   python -m newton.mcp host my_scene.py --connection-file session.json -- --my-arg 3

Arguments after ``--`` go to the script's own parser. The host
(:class:`newton.mcp.ExampleHost`) enables trusted execution, exposes the live
``example`` and its ``module``, and includes the example's own Warp arrays and
scalar attributes in checkpoints so controller phases rewind with the physics.

Before the first step after a cell, the host compares the example's attributes,
the script's module globals, and the attributes of the objects the example
holds (solver, model, collision pipeline, the script's own objects such as
controllers, and their option objects, three levels deep) with their values
when the example's CUDA graphs were recorded. Scalars and small plain-data containers compare by value, other
objects by identity. If any of them changed (a gain, ``SUBSTEPS``, a solver
option, a replacement solver), the host re-records the graphs and reports this
as ``note`` in the execution result; settings that stepping itself advances,
such as timers, do not count. ``recapture()`` re-records them explicitly.

``newton_rebuild`` reloads the edited script from disk in the same process. If
loading or constructing the example raises, the previous scene keeps running and
the error shows the traceback of the script with ``file:line``. Rebuild
``overrides`` set module globals after the script is loaded and before
``Example()`` is constructed:

.. code-block:: python

   client.request("rebuild", overrides={"SUBSTEPS": 32, "PARAMS": {"dt": 0.001}})

A dictionary merges into a dictionary global (recursively); other values replace
the global, keeping a float global a float and a tuple a tuple. Module code that
already ran while loading, such as a constant computed from the original value
or a default argument, keeps the original value. The given mapping becomes the
active set for later rebuilds and restarts; omit ``overrides`` to keep it and
pass ``{}`` to clear it. Every response, including errors, reports the active
set as ``overrides``. ``python -m newton.mcp host ... --overrides JSON`` starts
with an active set.

``--workers N`` also hosts ``N`` sibling copies of the script in separate
processes and exposes them as ``workers`` (see :class:`newton.mcp.WorkerPool`)
for parallel parameter sweeps, plus ``jobs`` (:class:`newton.mcp.JobQueue`) for
background calls; ``--max-workers M`` (default ``max(N, 4)``) bounds
``workers.resize(n)``. Workers follow every successful ``newton_rebuild``,
including its ``overrides`` and example arguments.

Connect an MCP client
---------------------

Configure a stdio MCP server with this command, using an absolute connection
file path when the MCP client's working directory differs:

.. code-block:: bash

   uv run -m newton.mcp --connect /absolute/path/session.json

The default ``--profile full`` advertises every structured tool. Add
``--profile code`` to advertise only ``newton_describe``, ``newton_execute``,
``newton_observe``, ``newton_filmstrip``, and ``newton_rebuild``, or ``--profile lean`` to
advertise only ``newton_execute`` and ``newton_rebuild`` with shorter server
instructions. Tool schemas and instructions are resent on every model turn, so
the smaller profiles reduce per-turn context for clients that prefer Python;
images remain available through ``show(session.dispatch("observe", ...))``.
The profile changes presentation, not permissions:
trusted execution still requires ``allow_execute=True``. Python can call
``session.dispatch(operation, arguments)`` for every structured operation listed
by ``describe``. Rebuild remains a separate tool because it is the way out of
an invalid scene, which refuses stepping and observation.

This adapter implements newline-delimited JSON-RPC initialization and tools,
following the `MCP stdio transport specification
<https://modelcontextprotocol.io/specification/2025-11-25/basic/transports>`_.
Tool results carry a compact JSON text block (default-valued status fields are
omitted) followed by any images. The server instructions include a short
workflow guide, followed by application notes passed as
``SimulationSession(..., guide=...)``.

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
A failed ``step`` (or ``filmstrip``) rolls back to the state before the call,
as described in :ref:`live-mcp-rollback`. Playback pauses after a failed step
and exposes a bounded ``last_error`` in status. A failed model notification
during ``edit`` leaves model and solver coherence unknown and invalidates the
scene until it is rebuilt. ``replace()`` accepts a new model and solver; a
registered rebuild callback can expose this through MCP without restarting the
process.

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

Omitting ``eye``, ``target``, and ``pose`` frames the current scene
automatically: the camera looks at the bounding sphere of non-plane shapes and
particles in the selected world from a ``view`` preset (``iso``, ``front``,
``back``, ``left``, ``right``, or ``top``). ``views`` renders several presets
or camera dictionaries into one labeled grid image. ``reference`` names an
image file taken with the same camera; the render uses the reference size and
the response shows simulated, reference, and mismatch panels, with magenta
marking pixels whose largest channel difference exceeds 24/255, plus the mean
absolute difference and mismatch fraction.

``filmstrip`` advances the simulation and returns one labeled grid whose
columns are capture times and whose rows are views. Give absolute ``times``
(optionally after ``reset=True`` or ``restore=<checkpoint>``) or ``count``
frames ``every_steps`` apart. ``references`` adds reference and mismatch rows
per view, one reference image per time. Stepping uses the normal step path,
so application callbacks and recordings behave as in ``step``. Times wrap into
bands and pages sized for how MCP clients display images (about 1568 px on the
long edge); frames shrink only as far as needed to fit ``max_pages`` pages, and
pages after the first are returned in ``images``.

``camera_body`` mounts the camera on a body, given by label or index; labels
that replicated worlds share resolve within ``world_id``. ``camera_offset``
is the camera pose in the body frame (``[x, y, z, qx, qy, qz, qw]``, default
identity), and ``observe``, ``filmstrip``, and ``record`` read the body pose at
every capture, so a wrist camera with calibrated ``intrinsics`` can be
compared frame by frame with recorded wrist video. ``overlay`` maps names to
simulated world points in meters: ``{"body": ..., "point": [x, y, z]}`` (a
point in the body frame), fixed coordinates, or, with trusted execution, a
Python expression or callable returning a point or an ``(N, 3)`` array.
``observe`` and ``filmstrip`` project the points through the same camera,
including distortion, draw labeled rings on the simulated and reference images
after the comparison metrics are computed, and return their pixel coordinates.

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
``control``, ``contacts``, ``viewer``, ``wp``, ``np``, and ``show`` refresh before
each call and after managed state changes. Objects passed as
``SimulationSession(..., namespace={...})`` are also refreshed before each
call, so applications can expose their own controllers or task objects. Functions that read these globals see the
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

``show(image, label=None)`` attaches up to eight images to the response of the
current call. It accepts arrays (``HxW``, ``HxWx3``, or ``HxWx4``; floats in
``[0, 1]``), Pillow images, matplotlib figures, PNG bytes, image paths, and
``observe``/``filmstrip`` results. MCP clients receive them as image content,
so custom plots and composites need no file round trip.

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
Scene ``replace``/``rebuild`` keeps Python variables and refreshes the reserved
bindings; user references to old scene objects stay stale until reassigned.
Pass ``replace(..., keep_workspace=False)`` or ``rebuild(reset_namespace=True)``
to clear the workspace as well. Closing the session removes its Python module registration
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

Analysis helpers
----------------

Trusted cells can use these helpers without imports (``newton``, ``np``, and
``wp`` are preloaded as well):

- :meth:`~newton.mcp.SimulationSession.rollout` steps the scene and samples
  named series (callables or workspace expressions such as
  ``"state.body_q.numpy()[3, 2]"``) in one call, optionally resetting or
  restoring a checkpoint first, stopping on a condition, and plotting the result.
- :meth:`~newton.mcp.SimulationSession.solver_contacts` groups the solver's
  active contacts by shape pair and lists the parameters the solver actually
  integrates, such as MuJoCo ``solref``, ``solimp``, and friction after geom
  priority and material mixing, next to the authored shape materials.
- :meth:`~newton.mcp.SimulationSession.health` flags non-finite state, runaway
  velocities, deep penetration, and full solver contact or constraint buffers.
- :meth:`~newton.mcp.SimulationSession.swap_solver` replaces the solver with
  ``factory(model)``: it installs the new solver (a hosted example re-records
  its CUDA graphs), steps a copy of the current state for two frames, runs
  ``health()``, and restores the state. If any of this raises, or ``health()``
  reports a warning the current state does not already show, the previous
  solver, graphs, and state are reinstated and the error is raised.

Parallel worker sessions
------------------------

Every request to one session runs on its owner thread, so candidate
evaluations issued through one live application run one after another.
Script-based workflows can instead run several simulator processes at once.
To recover that parallelism without giving up persistent state, a session can
own a :class:`newton.mcp.WorkerPool` of sibling sessions (typically more
instances of the same application). ``python -m newton.mcp host ... --workers N``
launches one with :meth:`~newton.mcp.WorkerPool.launch`; an embedding
application can also attach existing sessions by passing their connection files
as ``SimulationSession(..., workers=[...])``. Trusted execution receives the
pool as ``workers``:

.. code-block:: python

   def evaluate(stiffness):
       model.joint_target_ke.fill_(stiffness)
       solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
       series = rollout(seconds=2.0, start=True, record={"x": "state.body_q.numpy()[0, 0]"})
       return float(abs(series["x"][-1] - TARGET))

   TARGET = 0.25
   losses = workers.map(evaluate, np.linspace(100.0, 1000.0, 16))

``map`` calls a function once per item on the free workers and returns the
results in input order, ``submit`` returns a future, and ``broadcast`` runs a
function or code string once on every worker. Each worker keeps its own scene
and Python workspace:

* Functions defined in trusted-execution cells, including lambdas and
  closures, are sent by source. Their cells stay registered in
  :mod:`linecache`, so :func:`inspect.getsource` and tracebacks also work for
  them. The cell functions and classes a function uses and the session globals
  it reads (``TARGET`` above) are sent with it; arguments, those globals, and
  results are pickled, and Warp arrays travel as NumPy data.
* Names bound to a session's live objects (``example``, ``model``, ``state``,
  ``solver``, ``rollout``, ...) are not sent: on a worker they refer to the
  worker's own scene, which does not have the main session's live edits.
* ``workers.sync(name=value)`` copies values into every worker's globals once;
  functions do not resend a synced global while the session still binds the
  same object.
* Code strings run as cells on a worker with the item bound to ``args``.
* A failed call rolls back its worker's simulation like any failed cell. It
  returns ``{"error": ...}`` from ``map`` (with the worker's cell and line) and
  raises from ``submit`` and ``broadcast``.

A launched pool follows ``newton_rebuild`` with the same arguments. A worker
whose process exits or whose CUDA context fails is restarted, and earlier
``broadcast`` and ``sync`` calls are replayed on it in order;
``workers.resize(n)`` adds or removes workers the same way. The session lists
such events under ``workers`` in its next execution response.

``jobs.start(fn_or_code, *args, where="worker")`` queues a background call and
returns a job id at once; nothing runs on the main session's simulation.
``jobs.wait(timeout=..., any=True)`` returns finished results and the lines
running jobs printed since the previous wait, and every execution response
lists jobs that finished since the previous response. Other places to run jobs
are added with :meth:`~newton.mcp.JobQueue.register_backend`.

.. _live-mcp-rollback:

Rollback after a failed call
----------------------------

A syntax or compilation error runs no Python. Before a cell runs, the session
copies the simulation's mutable arrays on their device: state, control, model
arrays, and arrays the application registers (a hosted example's own Warp
arrays). A hosted example also remembers its attributes and its script's module
globals, including the contents of small plain-data lists and dictionaries.
If the cell raises, also inside ``rollout`` or a step, the session

* rebinds replaced objects (the solver, state buffers, example attributes, and
  module globals) and their CUDA graphs,
* writes back the arrays whose contents changed and notifies the solver about
  restored model fields with the matching :class:`~newton.ModelFlags`,
* restores time and frame and, if the state moved, resets solver caches and
  contacts as ``restore`` does,

and reports what it restored in the error. If nothing changed, it says so and
leaves hidden solver state alone. Python variables assigned before the error
are kept. Objects changed in place outside these copies (solver internals,
meshes, large containers) are not restored. Array groups larger than 256 MiB
keep their identities but not their contents, and the error lists them. A
failed ``step`` or ``filmstrip`` request rolls back the same way.

Only a failure of the rollback itself, a failed model notification, or a failed
``replace`` invalidates the scene. While invalid, Python cells still run (for
example to save results), and stepping or observing raises an error that points
to ``rebuild``; ``rebuild`` with ``restart`` starts a fresh process after a CUDA
error.

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
