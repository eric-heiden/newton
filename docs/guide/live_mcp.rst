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
``reset`` and ``restore`` rewind the state and only what stepping changes: the
scalar attributes ``step()`` advances (timers, phase counters) and the
example's and the control's Warp arrays a step has written (a command cursor,
controller memory, targets the controller sets), whether the session stepped
or a cell called ``example.step()``. Writes are learned from device checksums
taken around each step of the arrays not yet known to be written; the
comparison stays on the device until a reset or restore reads it. Everything
else a cell changed stays: assigned gains, an edited command schedule, a
perturbation, control inputs no step overwrites. The response lists the
restored arrays as ``rewound`` and the changed but kept ones as ``kept``.
Reset and restore do not call the example's own ``reset()``; attributes that
``reset()`` reads and that changed since the snapshot, such as an initial pose
``example.q0``, are listed as ``not_applied``. Inside a cell, kept and
not-applied names also appear once in ``note``.

Before the first step after a cell, the host compares the example's attributes,
the script's module globals, and the attributes of the objects the example
holds (solver, model, collision pipeline, the script's own objects such as
controllers, and their option objects, three levels deep) with their values
when the example's CUDA graphs were recorded. Scalars and small plain-data containers compare by value, other
objects by identity. If any of them changed (a gain, ``SUBSTEPS``, a solver
option, a replacement solver), the host re-records the graphs and reports this
as ``note`` in the execution result, unless the cell itself assigned every
changed attribute (``example.gain = 2.0``); settings that stepping itself
advances, such as timers, do not count. ``recapture()`` re-records them
explicitly. When the script or one of its helper modules changed on disk after
the last build, the next execution result says so once, and errors of cells
repeat it until a rebuild: ``module`` and ``example`` are the built version.

``newton_rebuild`` reloads the edited script from disk in the same process,
together with the modules it imports from its own directory. If loading or
constructing the example raises, the previous scene (and the previously loaded
helper modules) keep running and the error shows the traceback of the script
with ``file:line``. ``newton_rebuild(arguments={"restart": true})`` first loads
the script with the requested overrides and parses the requested arguments in a
new process; if that fails, the restart is refused and the current process keeps
running. If ``Example()`` then fails in the restarted process, it builds with the
previous arguments and overrides instead and says so in the ``note`` of the next
execution result. A ``SystemExit`` raised by a cell, a step, or the example's
argument parser is reported as an error instead of ending the host. Rebuild
``overrides`` set module globals after the script is loaded and before
``Example()`` is constructed:

.. code-block:: python

   client.request("rebuild", overrides={"SUBSTEPS": 32, "PARAMS": {"dt": 0.001}})

``newton_rebuild(code=...)`` runs a cell after a successful rebuild and returns
its value and output with the rebuild result; if the rebuild fails, the cell
does not run. A dictionary merges into a dictionary global (recursively); other
values replace the global, keeping a float global a float and a tuple a tuple. Module code that
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
``workers.resize(n)``. The worker processes start on first use, so an unused
pool costs no processes or device memory. Workers follow every successful
``newton_rebuild``, including its ``overrides`` and example arguments, without
delaying it.

``python -m newton.examples.headless`` (see :doc:`development`) runs the
script as saved on disk in a new process, without the session's live edits.

Connect an MCP client
---------------------

Configure a stdio MCP server with this command, using an absolute connection
file path when the MCP client's working directory differs:

.. code-block:: bash

   uv run -m newton.mcp --connect /absolute/path/session.json

Tool calls reply within ``--reply-within`` seconds (default 240, below the
300 s tool timeout of common clients; ``0`` waits for completion). A call
still running then continues in the session, and the reply holds ``running``
with its call number and the output printed so far. The session runs one call
at a time: a later call waits for the running one (up to its own reply limit,
otherwise it is not run and says so), and its response carries the earlier
call's value, remaining output, images, or error under ``finished_calls``.
``jobs.result()`` and ``jobs.wait()`` without a ``timeout`` return shortly
before the running call's reply limit. Each tool result stays within
``--response-budget`` characters (default 1,000,000, below the 1 MiB some
clients accept): larger images are re-encoded as JPEG (with Pillow), halved in
size, or dropped, and ``images_note`` says which.

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
omitted) followed by any images. The server instructions describe the tools
and helpers, followed by application notes passed as
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

``reset`` restores the initial state, session time, and solver/contact caches
while retaining tuned model parameters; control arrays return to their initial
values only if a step has written them (``rewound``), so inputs set between
steps stay (``kept``). Named checkpoints store public state/control arrays and
time and restore them the same way, and with ``include`` the Python objects
named by workspace paths (see :ref:`live-mcp-batch`). They do not capture every
hidden solver state, so restoring a checkpoint resets solver caches and does
not promise bitwise replay. Reset and restore clear contact buffers and generation metadata without
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

``observe`` defaults to :class:`~newton.sensors.SensorCamera` and needs no GL context. Camera
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
Calibrated ``intrinsics``, ``pick``, and overlay pixels use the image
coordinates of :class:`~newton.sensors.SensorCamera.Intrinsics` (x right, y
down, integer values at pixel centers), which also projects and unprojects
points outside the MCP. In Python cells, ``intrinsics`` may also be a
:class:`~newton.sensors.SensorCamera.Intrinsics`. A calibration dictionary,
such as one camera's entry of a ``camera.json`` file with ``K`` (3x3, row
major), ``D`` (OpenCV coefficient order), ``width``, ``height`` and
``distortion_model``, is read like ``SensorCamera.Intrinsics.from_dict()``;
its ``position`` and ``rotation_xyzw`` place the camera when no ``eye``,
``pose``, ``view`` or ``camera_body`` is given. ``pose`` also accepts
``{"position": [...], "rotation_xyzw": [...]}``. A calibrated camera renders at
its calibration size unless ``width``/``height`` are given.

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
``control``, ``contacts``, ``viewer``, ``wp``, ``np``, ``newton``, ``show``, and the
analysis helpers below refresh before each call and after managed state changes. A helper name
that a cell bound to its own value (for example its own ``evaluate`` function) keeps that value;
the helper stays available as a session method (``session.evaluate``). Objects passed as
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

The value of the cell's last expression is returned; a cell that ends with a
statement returns ``None``. A variable named ``result`` is an ordinary
variable. ``_`` holds the last non-``None`` value, including a value too large
to return. JSON-compatible values use the ``result`` field of the response. An
opaque or oversized value uses a bounded ``result_repr`` summary instead,
without calling user ``__repr__`` or converting large arrays to lists; selected
attributes or slices of ``_`` can then be inspected. Captured stdout/stderr is
limited to 16384 characters, serialized JSON results to 65536 characters, and
result conversion to 16384 components and 64 nesting levels. Arrays above the
component limit or 1 MiB are summarized without conversion.

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

The source cache retains the 64 most recent cells of at most 65536 characters
each, plus older cells that still define a function or class bound in the
workspace (at most 512 cells in total), so :func:`inspect.getsource` keeps
working for live definitions. Classes defined at the top level of a cell are
moved into a per-cell module whose ``__file__`` names the cached cell source,
which is where :func:`inspect.getsource` looks for a class; they still pickle
by reference. Older Python functions still execute, but source inspection may
be unavailable after their cell is evicted. ``describe`` and successful execution responses
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
  restoring a checkpoint first and stopping on a condition.
- :meth:`~newton.mcp.SimulationSession.evaluate`,
  :meth:`~newton.mcp.SimulationSession.branch`, and
  :meth:`~newton.mcp.SimulationSession.checkpoint` run candidates, scenarios,
  and variants as worlds of N copies of the live scene and save Python objects
  with a checkpoint (see :ref:`live-mcp-batch`).
- :meth:`~newton.mcp.SimulationSession.solver_contacts` groups the solver's
  active contacts by shape pair (short labels, counts, smallest distance),
  optionally only pairs whose shape or body label matches ``select``; with
  ``detail=True`` it lists the parameters the solver actually integrates, such
  as MuJoCo ``solref``, ``solimp``, and friction after geom priority and
  material mixing, next to the authored shape materials.
- :meth:`~newton.mcp.SimulationSession.contacts_between` reports the contact
  count, normal and friction force, slip speed, and penetration between two
  shape sets; it also works as a ``rollout`` probe.
- :meth:`~newton.mcp.SimulationSession.solver_params` lists, per actuator,
  joint, geom, body, equality constraint, or solver option, the value
  :class:`~newton.solvers.SolverMuJoCo` integrates, the Newton model array and
  index it comes from, the :class:`~newton.ModelFlags` category that refreshes
  it, whether it can differ per world, and model values that differ from the
  compiled ones (``pending``). Other solvers report the Newton model values.
  It is :func:`newton.utils.report_solver_params` applied to the session's
  solver.
- :meth:`~newton.mcp.SimulationSession.health` checks any solver and state
  (default: the session's) for non-finite values, runaway speeds, full contact
  or constraint buffers, and penetrating shape pairs, naming the worlds
  involved; ``twins=True`` also reports worlds whose joint state deviates from
  the others. It is :func:`newton.utils.report_health` applied to the session's
  model, state, solver, and contacts.

After each cell, and before each ``rollout``, step, or ``filmstrip`` inside a
cell, the session compares device checksums of the model arrays solvers read
with their previous values. Changed arrays whose :class:`~newton.ModelFlags`
category no ``notify_model_changed`` call covered are notified with the inferred
flags, and the execution result's ``note`` has one line naming those fields and
flags, plus one line per edited field the current solver configuration does not
read (for example ``mujoco.actuator_gainprm`` of actuators driven by
``joint_target_ke``), each such line once per scene. Edits a ``notify_model_changed`` call covered are not
reported, nor are edits whose values :class:`~newton.solvers.SolverMuJoCo`
already integrates in every world because the cell also wrote its compiled
arrays (as :meth:`~newton.mcp.SimulationSession.solver_params` shows without
``pending``). A call covers a changed array only if it was made on the session's
solver with the array's category after the array's last change; calls on other
solver objects, or before the edit, are named in the note.
``session.watch.mode = "report"`` reports without notifying and ``"off"``
disables the checks. Edits made by the application's own ``step()``, also when a
cell calls ``example.step()``, are not reported. Reading the checksums before a
step waits for the device work queued before it.

.. _live-mcp-batch:

Batched evaluation of the live scene
------------------------------------

:meth:`~newton.mcp.SimulationSession.evaluate` and
:meth:`~newton.mcp.SimulationSession.branch` run
:class:`newton.utils.BatchRollout` (see :doc:`/concepts/batched_evaluation`)
on a model with N copies of the hosted scene. While ``Example()`` is
constructed, :class:`~newton.mcp.ExampleHost` records the one-world
:class:`~newton.ModelBuilder` the example finalized into ``example.model`` and
the constructor arguments of ``example.solver`` and
``example.collision_pipeline``; the copies are built from these, so the worlds
use the same solver settings. Collision-pipeline capacities given as arguments
(``rigid_contact_max``, ``soft_contact_max``, ``shape_pairs_max``) are
multiplied by the world count. ``solver=``, ``pipeline=``, ``dt=``, and
``substeps=`` replace the recorded ones; a frame defaults to one session frame
(the example's ``sim_dt`` steps per ``frame_dt``).

- Every world starts from ``start``: the live state (default), ``"initial"``
  (the state after the scene was built), or a checkpoint, with the live (or
  saved) control values. ``example.step()`` does not run in the worlds;
  ``control`` (schedules and Warp control functions) and per-world setups
  drive them. The live session does not change.
- The N-world model, its solver, and its CUDA graphs are kept for later
  calls (two models per session), also across ``newton_rebuild`` when the
  rebuilt scene has the same structure and integer attributes, solver
  arguments, and steps. Before each call the live model's values that differ
  from the copies are written into every world and notified; a differing value
  the copies' solver reads only when it is constructed builds a new model.
- Results are returned with facts: the start, whether the model was reused or
  built (and why), the live values copied, and the wall time with the CUDA
  graph capture and kernel loading of new graphs. A cell whose value is an
  evaluation, a branch result, an evaluation diff, or a
  :class:`~newton.utils.TrajectoryComparison` returns its text table.

``branch(n, setup, ...)`` returns ``records`` of shape ``[T, n, ...]`` (row 0
at the start), their times ``t``, and the ``metrics`` of an optional
``score(records)``. With ``sequential=True`` the variants run one after
another through the session's own step (``example.step()`` with its
controllers): before each variant the start is restored, together with the
Python objects of the start checkpoint and the example's other objects as they
were at the call, ``setup(world, i)`` edits the live scene (``world`` has the
methods of :class:`newton.utils.BatchRollout.WorldSetup` except schedules) or
Python objects, and its model edits and those of stepping are undone after the
variant. Records then also accept functions ``fn()`` returning values.

``checkpoint(name, include=["example.controller", ...])`` saves Python
objects with the state, for example a controller with its warm start. The
paths name workspace variables or their attributes. The snapshot copies their
Warp and NumPy arrays and remembers the bindings of their attributes, list
items, and dictionary entries, four levels deep into objects of the hosted
script, its helper modules, and cells (and Newton states and controls).
Other objects, such as solvers and models, and the session's own state,
control, model, and solver stay bound as they are. Restoring the checkpoint
(``session.dispatch("restore", ...)``, ``rollout(start=name)``,
``filmstrip(restore=name)``, or ``branch(start=name)``) writes the copies back
into the same arrays, so captured CUDA graphs stay valid, and the response
lists ``objects_restored``.

Writing live results back to the script
---------------------------------------

Values and definitions developed in trusted cells can be written into the
hosted script (:attr:`~newton.mcp.SimulationSession.source_path`, which
:class:`~newton.mcp.ExampleHost` sets) instead of being retyped:

- :meth:`~newton.mcp.SimulationSession.persist` replaces the value of a
  module-level ``NAME = <literal>`` assignment. NumPy and Warp values become
  plain literals. When the old and new values have the same dictionary keys or
  sequence lengths, only the differing entries are rewritten, so comments inside
  the literal remain.
- :meth:`~newton.mcp.SimulationSession.persist_source` replaces a top-level
  ``def`` or ``class`` (or a method, with ``target="Class.method"``) with the
  source of the object defined in a cell, re-indented to fit.

Both refuse a target that is missing, bound more than once at that level, or
not a literal assignment or definition, and they change nothing else in the
file. They print a unified diff, save the previous file under
``<artifact_directory>/persist/``, and then rebuild the scene and any worker
sessions unless ``rebuild=False``. A ``check`` expression is evaluated before
writing and again after the rebuild; the result reports both values and
whether they agree within ``tolerance``.

.. code-block:: python

   module.PARAMS["kp"] = 80.0  # live edit while experimenting
   persist("PARAMS", check="rollout(seconds=1.0, start=True, record={'x': 'state.body_q.numpy()[0, 0]'})['x'][-1]")
   persist_source(step, target="Example.step")

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
       solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
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
* Code strings run as cells on a worker with the item bound to ``args``; the
  value of their last expression is returned.
* A failed call rolls back its worker's simulation like any failed cell, and
  the error says what was restored; the worker's Python variables are kept. It
  returns ``{"error": ...}`` from ``map`` (with the worker's cell and line) and
  raises from ``submit`` and ``broadcast``.
* Warp kernels sent with a function are defined in a module named after their
  source, so all workers (and restarted ones) share their kernel-cache entry.

A launched pool follows ``newton_rebuild`` with the same arguments without
delaying it. A started worker rebuilds in the background, after its running
call and before any queued call; the rebuild response lists these workers under
``workers_rebuild``, and a failed worker rebuild is listed under ``workers`` in
a later response. Workers that have not started yet start with the new
arguments. Each rebuild increments the pool's ``build`` count, which
``workers.status()`` and every job record report. A worker
whose process exits or whose CUDA context fails is restarted, and earlier
``broadcast`` and ``sync`` calls are replayed on it in order;
``workers.resize(n)`` adds or removes workers the same way. The session lists
such events under ``workers`` in its next execution response.

``jobs.start(fn_or_code, *args, where="worker")`` queues a background call and
returns a job id at once; nothing runs on the main session's simulation.
``jobs.wait(timeout=..., any=True)`` returns finished results and the lines
running jobs printed since the previous wait, with the worker and the ``build``
its scene had when the job started, and every execution response lists jobs
that finished since the previous response. Other places to run jobs
are added with :meth:`~newton.mcp.JobQueue.register_backend`.

.. _live-mcp-rollback:

Rollback after a failed call
----------------------------

A syntax or compilation error runs no Python. Before a cell runs, the session
copies the simulation's mutable arrays on their device: state, control, model
arrays, and arrays the application registers (a hosted example's own Warp
arrays). A hosted example also remembers its attributes, its script's module
globals, and the attributes of the classes defined at the top level of the
script and of the modules it imports from its own directory (so a method a cell
rebinds, such as ``module.Controller.compute``, is restored), including the
contents of small plain-data lists and dictionaries.
If the cell raises, also inside ``rollout`` or a step, the session

* rebinds replaced objects (the solver, state buffers, example attributes,
  module globals, and class attributes) and their CUDA graphs,
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
