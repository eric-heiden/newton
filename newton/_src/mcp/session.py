# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Owner-thread operations for an explicitly instrumented simulation."""

from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import functools
import inspect
import io
import json
import math
import queue
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import warp as wp

from ..sim.collide import CollisionPipeline
from ..sim.contact_kinematics import eval_rigid_contact_kinematics
from ..sim.enums import JointType, ModelFlags, StateFlags
from ..sim.model import Model
from .cells import cell_filename, forget_cell_source, register_cell_source, retain_cell_sources

if TYPE_CHECKING:
    from .workers import WorkerPool

_OMITTED = object()


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _field(obj: Any, name: str) -> Any:
    parts = name.split(".") if isinstance(name, str) else []
    if not 1 <= len(parts) <= 3 or any(not p.isidentifier() or p.startswith("_") for p in parts):
        raise ValueError("Use a public field path with at most three components")
    for part in parts:
        obj = getattr(obj, part)
    if callable(obj):
        raise ValueError("Methods cannot be queried as data")
    return obj


def _array(value: Any, *, maximum: int = 2_000_000) -> np.ndarray:
    if isinstance(value, wp.array):
        width = getattr(value.dtype, "_length_", 1)
        if value.size * width > maximum:
            raise ValueError(f"Array exceeds the {maximum} component host-transfer budget")
        return value.numpy()
    result = np.asarray(value)
    if result.size > maximum:
        raise ValueError(f"Array exceeds the {maximum} component budget")
    return result


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, list | tuple):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    raise ValueError(f"Unsupported result type {type(value).__name__}; return JSON-compatible data")


def _result_json(value: Any) -> Any:
    """Bound trusted Python output without invoking user-defined conversion callbacks."""
    remaining = [16384, 65536]

    def convert(item: Any, depth: int = 0) -> Any:
        remaining[0] -= 1
        remaining[1] -= 2
        if remaining[0] < 0 or depth > 64:
            raise ValueError("result exceeds the 16384 component budget or 64-level nesting budget")
        if remaining[1] < 0:
            raise ValueError("result exceeds 65536 characters")
        kind = type(item)
        if kind is str:
            remaining[1] -= len(item)
            if remaining[1] < 0:
                raise ValueError("result exceeds 65536 characters")
            return item
        if item is None or kind is bool:
            return item
        if kind is int:
            remaining[1] -= item.bit_length() * 30103 // 100000 + 2
            if remaining[1] < 0:
                raise ValueError("result exceeds the integer display budget")
            return item
        if kind is float:
            remaining[1] -= 24
            return item if math.isfinite(item) else None
        if issubclass(kind, np.generic):
            return convert(np.generic.item(item), depth + 1)
        if kind is np.ndarray:
            if item.size > remaining[0] or item.nbytes > 1048576:
                raise ValueError(
                    "result exceeds the 16384 component budget or 1 MiB array conversion budget; select a smaller slice"
                )
            return convert(item.tolist(), depth + 1)
        if kind is list or kind is tuple:
            if len(item) > remaining[0]:
                raise ValueError("result exceeds the 16384 component budget")
            return [convert(element, depth + 1) for element in item]
        if kind is dict:
            if 2 * len(item) > remaining[0]:
                raise ValueError("result exceeds the 16384 component budget")
            result = {}
            for key, element in item.items():
                if not any(type(key) is allowed for allowed in (str, bool, int, float, type(None))):
                    raise ValueError("Dictionary result keys must be built-in JSON scalar types")
                convert(key, depth + 1)
                result[str(key)] = convert(element, depth + 1)
            return result
        raise ValueError("Unsupported result type; return built-in JSON data or NumPy values")

    return convert(value)


def _session_callable(function: Callable) -> Callable:
    """Accept both ``fn()`` and ``fn(session)`` for rollout probes."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return function
    positional = [
        p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    variadic = any(p.kind == p.VAR_POSITIONAL for p in parameters)
    if not positional and not variadic:
        return lambda _session: function()
    return function


def _result_summary(value: Any) -> str:
    """Describe an opaque value without calling its repr or iterating its contents."""
    kind = type(value)
    if kind is np.ndarray:
        return f"<numpy.ndarray shape={value.shape} dtype={value.dtype.name}; inspect a slice of _>"
    if any(kind is allowed for allowed in (str, list, tuple, dict)):
        return f"<{kind.__name__} length={len(value)}; inspect a slice or selected entries of _>"
    name = type.__getattribute__(kind, "__name__") if type(kind) is type else "Python"
    name = name[:128] if type(name) is str else "Python"
    return f"<{name} object; inspect selected attributes of _>"


class SimulationSession:
    """Own the live bindings and serialize simulation operations on one thread.

    .. experimental::

        This entire class may change without a deprecation period. Applications
        must explicitly embed a session and pump it on their simulation thread.
        Hidden solver state is reset, not checkpointed, so restore does not
        promise bitwise replay. Trusted execution uses a persistent Python
        workspace, not a sandbox. It survives physical resets and is cleared
        by scene replacement. A failed cell or step rolls the simulation back
        to its state before the call (Python variables are kept); see
        :ref:`live-mcp-rollback` for what the rollback covers.

    Args:
        model: Finalized model.
        solver: Solver bound to the model.
        state: Current simulation state, or a newly allocated model state.
        state_next: Second state buffer, or a newly allocated model state.
        control: Control inputs, or newly allocated model control.
        collision_pipeline: Collision pipeline, or a newly constructed pipeline.
        contacts: Contact buffers, or buffers allocated by the pipeline.
        viewer: Optional viewer owned by the calling thread.
        dt: Default physics timestep [s].
        step_callback: Optional ``callback(session, dt)`` that performs one
            physics step and updates ``session.state`` and ``session.state_next``.
            The session advances its own time and frame counters afterwards.
        reset_callback: Optional ``callback(session)`` to reset application state
            after restoring arrays, solver caches, and session time.
        rebuild_callback: Optional ``callback(session, **arguments)`` returning
            keyword bindings for :meth:`replace`.
        allow_execute: Enable trusted, unrestricted Python execution in a
            persistent workspace. A cell that raises is rolled back.
        artifact_directory: Directory for observation and recording artifacts.
        namespace: Extra application objects exposed as globals in trusted
            execution, refreshed before every cell.
        guide: Application-specific usage notes; MCP clients receive them in
            the server instructions.
        snapshot_callback: Optional ``callback(session) -> object`` capturing
            application-owned state (controller phases, timers, targets) with
            every checkpoint and the initial reset snapshot.
        restore_callback: Optional ``callback(session, data)`` restoring what
            ``snapshot_callback`` captured, before ``reset_callback`` runs.
        workers: Connection files of sibling sessions (usually more instances
            of the same application), or a :class:`WorkerPool`. Trusted
            execution receives it as ``workers``, whose ``map``/``submit``/
            ``broadcast`` run functions and cells on them concurrently, and a
            :class:`JobQueue` as ``jobs`` for background calls. Worker sessions
            follow :meth:`dispatch` ``rebuild``; finished jobs and worker events
            are added to the next execution response.
        execute_callback: Optional ``callback(session)`` run after each
            successful cell; a returned string is added as ``note``.
        overlay_callback: Optional ``callback(session)`` returning meshes the
            application draws itself.
        undo_callback: Optional ``callback(session, copies)`` called before a
            cell or step. It may register application arrays with
            ``copies.capture(label, owner)`` and returns a function that
            restores the application's Python attributes when the call fails
            and returns the names it restored.
        batch_callback: Optional ``callback(session)`` run after the session
            itself changed the scene: a batch of consecutive steps (one ``step``
            call or one :meth:`rollout`) or a rebuild.
        close_callback: Optional ``callback(session)`` called once on the
            owning thread when the session closes.
    """

    class _Request:
        def __init__(self, operation: str, arguments: dict, timeout: float):
            self.operation = operation
            self.arguments = arguments
            self.deadline = time.monotonic() + timeout
            self.lock = threading.Lock()
            self.done = threading.Event()
            self.started = False
            self.cancelled = False
            self.result = None
            self.error = None

        def wait(self, timeout: float) -> Any:
            if not self.done.wait(timeout):
                with self.lock:
                    if not self.started:
                        self.cancelled = True
                        raise TimeoutError("Request expired before execution; it will not be applied")
                # Running Warp/GL/Python cannot be safely interrupted. Only queue
                # waiting has a deadline, and completion remains observable.
                self.done.wait()
            if self.error is not None:
                raise self.error
            return self.result

    class _Output(io.TextIOBase):
        def __init__(self, limit: int):
            self.limit = limit
            self.parts = []
            self.length = 0
            self.truncated = False

        def write(self, value: str) -> int:
            remaining = max(0, self.limit - self.length)
            self.parts.append(value[:remaining]) if remaining else None
            self.length += min(len(value), remaining)
            self.truncated |= len(value) > remaining
            return len(value)

    def __init__(
        self,
        model: Model,
        solver: Any,
        *,
        state: Any = None,
        state_next: Any = None,
        control: Any = None,
        collision_pipeline: Any = None,
        contacts: Any = None,
        viewer: Any = None,
        dt: float = 1.0 / 60.0,
        step_callback: Callable | None = None,
        reset_callback: Callable | None = None,
        rebuild_callback: Callable | None = None,
        allow_execute: bool = False,
        artifact_directory: str | Path | None = None,
        namespace: dict[str, Any] | None = None,
        guide: str | None = None,
        workers: list[str | Path] | WorkerPool | None = None,
        snapshot_callback: Callable | None = None,
        restore_callback: Callable | None = None,
        execute_callback: Callable | None = None,
        overlay_callback: Callable | None = None,
        undo_callback: Callable | None = None,
        batch_callback: Callable | None = None,
        close_callback: Callable | None = None,
        sync_callback: Callable | None = None,
    ):
        self._owner = threading.get_ident()
        self._queue = queue.Queue(maxsize=64)
        self._queue_lock = threading.Lock()
        self._closed = False
        self._renderer = None
        self._checkpoints = {}
        self._workspace = {}
        self._workspace_name = f"_newton_mcp_{id(self):x}"
        self._workspace_module = None
        self._workspace_warp_module = None
        self._workspace_sources = []
        self._cell_modules = {}
        self._workspace_generation = 0
        self._cell_count = 0
        self._execution_error = None
        self._shown_images = None
        self._transaction_depth = 0
        self._cell_undo = None
        self._batch_step = 0
        self._scene_generation = 0
        self.snapshot_callback = snapshot_callback
        self.restore_callback = restore_callback
        self.execute_callback = execute_callback
        """Called as ``execute_callback(session)`` after each successful trusted execution; a returned
        string is added to the execution result as ``note``."""
        self.overlay_callback = overlay_callback
        """Returns ``[(name, points, indices, color), ...]`` meshes drawn by the application itself,
        which color observations composite over the model's shapes."""
        self.undo_callback = undo_callback
        self.batch_callback = batch_callback
        self.status_fields: dict[str, Any] = {}
        """Extra fields included in every status and appended to every error sent to clients
        (e.g. a hosted script's active build overrides)."""
        self.close_callback = close_callback
        """Called once as ``close_callback(session)`` when the session closes, e.g. to stop application
        subprocesses."""
        self.sync_callback = sync_callback
        """Called as ``sync_callback(session)`` before model-edit checks so ``solver``/``state`` bindings are
        current when application code replaced them."""
        from .solverview import ModelWatch  # noqa: PLC0415

        self.watch = ModelWatch(self)
        """Model-edit detection around trusted execution; ``watch.mode`` is ``"notify"``, ``"report"`` or
        ``"off"``."""
        self.namespace = dict(namespace or {})
        """Extra names available in trusted execution, refreshed before each cell."""
        self.guide = guide
        """Application usage notes appended to the MCP server instructions."""
        self.workers = None
        """Optional :class:`WorkerPool` of sibling sessions, exposed as ``workers`` in trusted execution."""
        self._owns_workers = False
        if workers is not None and not isinstance(workers, list | tuple):
            self.workers = workers
        elif workers:
            from .workers import WorkerPool  # noqa: PLC0415

            self.workers = WorkerPool(list(workers))
            self._owns_workers = True
        from .jobs import JobQueue  # noqa: PLC0415

        self.jobs = JobQueue(self.workers)
        """Background jobs, exposed as ``jobs`` in trusted execution."""
        if self.workers is not None:
            self.workers.attach(lambda: self._workspace, self._session_names)
            self.namespace.setdefault("workers", self.workers)
        self.namespace.setdefault("jobs", self.jobs)
        self.artifact_directory = Path(artifact_directory or tempfile.mkdtemp(prefix="newton-mcp-"))
        self.dt = self._timestep(dt)
        self.step_callback = step_callback
        self.reset_callback = reset_callback
        self.rebuild_callback = rebuild_callback
        self.allow_execute = allow_execute
        self.source_path: Path | None = None
        """Script that :meth:`persist` and :meth:`persist_source` edit; :class:`ExampleHost` sets it."""
        self.revision = 0
        self.replace(
            model,
            solver,
            state=state,
            state_next=state_next,
            control=control,
            collision_pipeline=collision_pipeline,
            contacts=contacts,
            viewer=viewer,
        )

    @staticmethod
    def _timestep(dt: float) -> float:
        if isinstance(dt, bool) or not isinstance(dt, int | float) or not math.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("dt must be finite and in (0, 1] seconds")
        return float(dt)

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("Simulation operations must run on the session's owning thread; use a client or pump")

    def replace(
        self,
        model: Model,
        solver: Any,
        *,
        state: Any = None,
        state_next: Any = None,
        control: Any = None,
        collision_pipeline: Any = None,
        contacts: Any = None,
        viewer: Any = None,
        keep_workspace: bool = True,
    ) -> None:
        """Replace all scene bindings and discard old snapshots in this process.

        Topology changes require a newly built model and matching solver. By
        default, Python variables and functions survive and the live bindings
        (``model``, ``state``, ...) refresh; user references to old scene
        objects remain stale until reassigned. Any escaped references or
        application CUDA graphs must be rebuilt by the application too.

        Args:
            model: New finalized model.
            solver: New solver bound to the model.
            state: Initial state, or a new model state.
            state_next: Output buffer, or a new model state.
            control: Control inputs, or a new model control.
            collision_pipeline: New collision pipeline, or one constructed here.
            contacts: Contact buffers, or buffers from the pipeline.
            viewer: Viewer to bind to the replacement model.
            keep_workspace: Keep Python variables, functions, and source
                history. ``False`` clears them and the workspace Warp module.
        """
        self._assert_owner()
        self.paused = True
        self.valid = False
        self._requires_rebuild = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._scene_generation += 1
        self.model, self.solver = model, solver
        self.state = state if state is not None else model.state()
        self.state_next = state_next if state_next is not None else model.state()
        self.control = control if control is not None else model.control()
        self.collision_pipeline = collision_pipeline if collision_pipeline is not None else CollisionPipeline(model)
        self.contacts = contacts if contacts is not None else self.collision_pipeline.contacts()
        self.viewer = viewer
        if viewer is not None:
            viewer.set_model(model)
        self.time = 0.0
        """Elapsed simulation time [s]."""
        self.frame = 0
        self.revision += 1
        self._checkpoints.clear()
        self._initial = self._snapshot()
        self._contact_frame = None
        self._contact_revision = None
        self.last_error: str | None = None
        self._requires_rebuild = False
        self.valid = True
        self.watch.reset()
        if keep_workspace:
            self._refresh_workspace()
        else:
            self._clear_workspace()

    def _snapshot(self) -> dict:
        total_bytes = 0

        def copy_array(array):
            nonlocal total_bytes
            total_bytes += array.capacity
            if total_bytes > 256 * 1024 * 1024:
                raise ValueError("State/control snapshot exceeds 256 MiB")
            return array.numpy().copy()

        def arrays(obj):
            result = {}
            for name, value in vars(obj).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, wp.array):
                    result[name] = copy_array(value)
                elif isinstance(value, Model.AttributeNamespace):
                    for child, array in vars(value).items():
                        if not child.startswith("_") and isinstance(array, wp.array):
                            result[f"{name}.{child}"] = copy_array(array)
            return result

        snapshot = {
            "state": arrays(self.state),
            "control": arrays(self.control),
            "time": self.time,
            "frame": self.frame,
        }
        if getattr(self, "snapshot_callback", None) is not None:
            snapshot["application"] = self.snapshot_callback(self)
        return snapshot

    def _resync(self, time: float, frame: int) -> None:
        """Reset solver history and contacts after the public state arrays were rewritten."""
        # Reset history without overwriting the restored public state with model defaults.
        self.solver.reset(self.state, flags=StateFlags.NONE)
        self.state_next.assign(self.state)
        self.collision_pipeline.reset_contact_matching()
        self.contacts.clear(bump_generation=True)
        self._contact_frame = self._contact_revision = None
        self.time, self.frame = time, frame

    def _restore(self, snapshot: dict) -> dict:
        self.paused = True
        for root in ("state", "control"):
            for field, data in snapshot[root].items():
                _field(getattr(self, root), field).assign(data)
        self._resync(snapshot["time"], snapshot["frame"])
        if self.restore_callback is not None and "application" in snapshot:
            self.restore_callback(self, snapshot["application"])
        if self.reset_callback is not None:
            self.reset_callback(self)
        self.valid = True
        self.revision += 1
        self.last_error = None
        self._refresh_workspace()
        return self._status()

    def enqueue(self, operation: str, arguments: dict, *, timeout: float = 30.0) -> _Request:
        """Queue an operation from a transport thread without touching device data.

        Args:
            operation: Operation name accepted by :meth:`dispatch`.
            arguments: JSON-compatible keyword arguments.
            timeout: Maximum waiting time before execution starts [s].

        Returns:
            Internal completion handle used by :class:`SimulationServer`.
        """
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("Queue timeout must be in (0, 300] seconds")
        request = self._Request(operation, arguments, timeout)
        with self._queue_lock:
            if self._closed:
                raise RuntimeError("Session is closed")
            self._queue.put_nowait(request)
        return request

    def pump(self, *, max_requests: int = 16) -> int:
        """Run queued requests on the owning thread, including while paused.

        Args:
            max_requests: Maximum requests to process in this call.

        Returns:
            Number of requests consumed, including expired requests.
        """
        self._assert_owner()
        _integer(max_requests, "max_requests", 1, 64)
        count = 0
        while count < max_requests:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            count += 1
            with request.lock:
                if request.cancelled or time.monotonic() >= request.deadline:
                    request.error = request.error or TimeoutError(
                        "Request expired before execution; it was not applied"
                    )
                    request.done.set()
                    continue
                request.started = True
            try:
                request.result = self.dispatch(request.operation, request.arguments)
            except SystemExit as error:
                # Application code calling sys.exit() must not end the serving process.
                request.error = RuntimeError(f"SystemExit({error.code!r}) raised during {request.operation}")
            except Exception as error:
                if self.status_fields:
                    # Clients see status fields such as active overrides on errors too (see transport).
                    with contextlib.suppress(Exception):
                        error.newton_status = _json(self.status_fields)
                request.error = error
            finally:
                request.done.set()
        return count

    def run(self, until: Callable[[], bool] | None = None) -> None:
        """Pump requests and advance playback until closed, interrupted, or ``until()`` is true.

        Playback uses the configured timestep [s] without wall-clock pacing.
        Embed :meth:`pump` in an application loop for custom rendering/pacing.

        Args:
            until: Optional predicate checked between requests; returning ``True``
                stops the loop and closes the session.
        """
        self._assert_owner()
        try:
            while not self._closed and not (until is not None and until()):
                self.pump()
                if not self.paused:
                    try:
                        self.dispatch("step", {"count": 1})
                    except Exception as error:
                        # The failed step was rolled back (or the scene is now invalid); stop playback.
                        self.paused = True
                        self.last_error = str(error)[:4096]
                else:
                    time.sleep(0.005)
        finally:
            self.close()

    def close(self) -> None:
        """Stop playback and reject pending requests on the owning thread."""
        self._assert_owner()
        self.paused = True
        with self._queue_lock:
            self._closed = True
            while True:
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    break
                request.error = RuntimeError("Session closed before request execution")
                request.done.set()
        if self.close_callback is not None:
            callback, self.close_callback = self.close_callback, None
            callback(self)
        if self._renderer is not None:
            self._renderer.close()
        if self._owns_workers:
            self.workers.close()
        self._clear_workspace()
        if self._workspace_module is not None:
            # Also the aliases under which definitions shipped from other sessions live (see shipping).
            for name in [name for name, module in sys.modules.items() if module is self._workspace_module]:
                del sys.modules[name]

    def _status(self) -> dict:
        status = {
            "time": self.time,
            "frame": self.frame,
            "revision": self.revision,
            "paused": self.paused,
            "valid": self.valid,
            "closed": self._closed,
            "last_error": self.last_error,
            "requires_rebuild": self._requires_rebuild,
            **self.status_fields,
        }
        return status

    def _bindings(self, entry: str | list[str] | None = None) -> tuple:
        model, solver, state, contacts = self.model, self.solver, self.state, self.contacts
        path = entry.split("/") if isinstance(entry, str) else (entry or [])
        if not isinstance(path, list) or len(path) > 8:
            raise ValueError("entry must be a slash-separated path or list of at most eight entry names")
        for name in path:
            if name not in solver.entry_names():
                raise ValueError(f"Unknown coupled entry {name!r}")
            model, state, contacts = solver.view(name), solver.entry_state(name), solver.entry_contacts(name, contacts)
            solver = solver.solver(name)
        return model, solver, state, contacts

    def _describe_solver(self, solver: Any, depth: int = 0) -> dict:
        result = {"type": type(solver).__name__}
        if hasattr(solver, "entry_names") and depth < 8:
            result["entries"] = {
                name: self._describe_solver(solver.solver(name), depth + 1) for name in solver.entry_names()[:64]
            }
        return result

    def _describe(self) -> dict:
        return {
            **self._status(),
            "experimental": True,
            "dt": self.dt,
            "device": str(self.model.device),
            "counts": {
                name: int(getattr(self.model, name))
                for name in (
                    "world_count",
                    "body_count",
                    "shape_count",
                    "joint_count",
                    "joint_dof_count",
                    "particle_count",
                )
            },
            "solver": self._describe_solver(self.solver),
            "roots": ["model", "state", "control", "solver", "collision"],
            "operations": [
                "describe",
                "query",
                "edit",
                "contacts",
                "collide",
                "step",
                "play",
                "pause",
                "reset",
                "checkpoint",
                "restore",
                "observe",
                "filmstrip",
                "record",
                "execute",
                "rebuild",
            ],
            "guide": self.guide,
            "model_flags": {flag.name: int(flag) for flag in ModelFlags},
            "editable_model_fields": sorted(self._EDIT_FLAGS),
            "workspace": self._workspace_info(),
            "capabilities": {
                "execute": self.allow_execute,
                "workers": 0 if self.workers is None else self.workers.count,
                "workers_max": 0 if self.workers is None else self.workers.max_count,
                "rebuild": self.rebuild_callback is not None,
                "observation": {"sensor": True, "viewer": self.viewer is not None},
                "record": True,
                "checkpoint_replay": "public arrays plus solver reset; not bitwise",
            },
            "limits": {
                "query_rows": 256,
                "query_components": 4096,
                "host_components": 2_000_000,
                "step_count": 10000,
                "pending_requests": 64,
                "checkpoints": 8,
            },
        }

    def dispatch(self, operation: str, arguments: dict | None = None) -> dict:
        """Execute one structured operation on the simulation thread.

        Args:
            operation: Tool operation, such as ``describe``, ``query``, ``edit``,
                ``step``, ``reset``, ``observe``, or ``execute``.
            arguments: Keyword arguments. The MCP tool schemas describe each
                operation; ``expected_revision`` provides a stale-write guard.

        Returns:
            JSON-compatible result with simulation metadata.
        """
        self._assert_owner()
        if self._closed:
            raise RuntimeError("Session is closed")
        if not isinstance(arguments, dict | type(None)):
            raise ValueError("arguments must be an object")
        args = dict(arguments or {})
        expected = args.pop("expected_revision", None)
        if expected is not None and expected != self.revision:
            raise ValueError(f"Stale revision {expected}; current revision is {self.revision}")
        operations = {
            "describe": self._describe,
            "query": self._query,
            "edit": self._edit,
            "contacts": self.contact_data,
            "collide": self._collide,
            "step": self._step,
            "pause": self._pause,
            "play": self._play,
            "reset": self._reset,
            "checkpoint": self._checkpoint,
            "restore": self._restore_named,
            "observe": self._observe,
            "filmstrip": self._filmstrip,
            "record": self._record,
            "guide": self._guide,
            "execute": self._execute,
            "rebuild": self._rebuild,
        }
        if operation not in operations:
            raise ValueError(f"Unknown operation {operation!r}")
        # Python stays available while invalid, e.g. to save results before a restart.
        if not self.valid and operation not in {
            "describe",
            "guide",
            "query",
            "pause",
            "reset",
            "restore",
            "rebuild",
            "execute",
        }:
            raise RuntimeError(self._invalid_message())
        if self._requires_rebuild and operation in {"reset", "restore"}:
            raise RuntimeError(self._invalid_message())
        return operations[operation](**args)

    def _invalidate(self, *, requires_rebuild: bool = False) -> None:
        self.paused = True
        self.valid = False
        self._requires_rebuild |= requires_rebuild
        self.revision += 1

    def _invalid_message(self) -> str:
        reason = f" ({self.last_error[:1024]})" if self.last_error else ""
        return (
            f"The scene is invalid{reason}. newton_rebuild reloads it in this process and keeps Python variables; "
            "newton_rebuild(arguments={'restart': true}) restarts the process, e.g. after a CUDA error."
        )

    def _undo_point(self):
        """Snapshot for rolling back a top-level operation, or ``None`` if nested or invalid."""
        if self._transaction_depth or not self.valid:
            return None
        from .rollback import UndoPoint  # noqa: PLC0415

        try:
            return UndoPoint(self)
        except Exception as error:
            # E.g. out of device memory for the copies; the operation still runs, without rollback.
            self.last_error = f"no rollback snapshot: {type(error).__name__}: {error}"[:4096]
            return None

    def _roll_back(self, undo, *, quiet: bool = False, nested: bool = False) -> str | None:
        """Undo a failed operation and describe the outcome.

        Args:
            undo: Undo point taken before the operation, or ``None``.
            quiet: Return ``None`` instead of a description when nothing changed.
            nested: The operation ran inside another rolled-back operation.
        """
        from .rollback import summarize  # noqa: PLC0415

        if undo is None:
            if not self.valid:
                return "The scene was already invalid, so nothing was rolled back."
            if nested:
                return "Nothing was rolled back here; the enclosing call rolls back if it fails."
            return f"Nothing was rolled back ({self.last_error}); statements before the error kept their effects."
        try:
            report = undo.rollback()
        except Exception as error:
            self._invalidate(requires_rebuild=True)
            self.last_error = f"rollback failed: {type(error).__name__}: {error}"[:4096]
            return f"Rolling the simulation back failed ({type(error).__name__}: {str(error)[:1024]}). " + (
                self._invalid_message()
            )
        unchanged = not report.get("replaced") and not (report["restored"] or report["moved"] or report["bindings"])
        if quiet and unchanged:
            return None
        self.paused = True
        if not report.get("replaced"):
            self.valid = True
            self._requires_rebuild = False
            self.last_error = None
        self.revision += 1
        if self._renderer is not None:
            self._renderer.invalidate()
        self._refresh_workspace()
        return summarize(report, undo.uncovered)

    def _roll_back_cell(self) -> str | None:
        """Roll the simulation back to the start of the running top-level cell, e.g. after a failed worker call.

        Returns:
            A description of what was restored, or ``None`` if nothing changed or no cell is running.
        """
        if self._cell_undo is None:
            return None
        outcome = self._roll_back(self._cell_undo, quiet=True)
        # Later edits in the cell are checked against the restored arrays.
        self.watch.reset()
        return outcome

    def _undoable(self, label: str, run: Callable[[], Any]) -> Any:
        """Run a top-level operation that rolls back when it raises."""
        undo = self._undo_point()
        self._transaction_depth += 1
        try:
            return run()
        except (Exception, SystemExit) as error:
            outcome = self._roll_back(undo, quiet=True) if undo is not None else None
            if isinstance(error, SystemExit):
                # sys.exit() in application code must not end the host process.
                raise RuntimeError(
                    f"SystemExit({error.code!r}) raised during {label}. {outcome or ''}".strip()
                ) from error
            if outcome is None:
                raise
            if error.args and isinstance(error.args[0], str):
                # Keep the exception type (e.g. ValueError for bad arguments) and append the outcome.
                error.args = (f"{error.args[0][:4096].rstrip('. ')}. {outcome}", *error.args[1:])
                raise
            raise RuntimeError(f"{type(error).__name__} during {label}: {str(error)[:4096]}. {outcome}") from error
        finally:
            self._transaction_depth -= 1

    def _step(self, *, count: int = 1, dt: float | None = None) -> dict:
        _integer(count, "count", 1, 10000)
        dt = self.dt if dt is None else self._timestep(dt)
        # Edits made earlier in the same cell are checked (and notified) before stepping.
        with self.watch.stepping("step"):
            self._undoable("step", lambda: self._advance(count, dt))
        return self._status()

    def _advance(self, count: int, dt: float, *, first: bool = True, last: bool = True) -> None:
        """Step ``count`` times; ``first``/``last`` mark the ends of a batch of consecutive steps."""
        if not self.valid:
            raise RuntimeError(self._invalid_message())
        for index in range(count):
            self._batch_step = 0 if first and index == 0 else self._batch_step + 1
            if self.step_callback is None:
                self.state.clear_forces()
                self.collision_pipeline.collide(self.state, self.contacts, dt=dt)
                self._contact_frame, self._contact_revision = self.frame, self.revision
                self.solver.step(self.state, self.state_next, self.control, self.contacts, dt)
                self.state, self.state_next = self.state_next, self.state
            else:
                self._contact_frame = self._contact_revision = None
                self.step_callback(self, dt)
            self.time += dt
            self.frame += 1
            self.revision += 1
            self._refresh_workspace()
            if self._renderer is not None:
                self._renderer.after_step()
        if last and self.batch_callback is not None:
            self.batch_callback(self)

    def _pause(self) -> dict:
        self.paused = True
        return self._status()

    def _play(self) -> dict:
        self.paused = False
        return self._status()

    def _reset(self) -> dict:
        try:
            return self._restore(self._initial)
        except Exception:
            # Inside a cell, the cell's rollback repairs the state.
            if not self._transaction_depth:
                self._invalidate()
            raise

    def _checkpoint(self, *, name: str = "default") -> dict:
        if not isinstance(name, str) or not 1 <= len(name) <= 64:
            raise ValueError("Checkpoint name must contain 1 to 64 characters")
        if name not in self._checkpoints and len(self._checkpoints) >= 8:
            raise ValueError("At most eight checkpoints are supported; overwrite an existing name")
        self._checkpoints[name] = self._snapshot()
        return {**self._status(), "name": name, "replay": "public arrays plus solver reset; not bitwise"}

    def _restore_named(self, *, name: str = "default") -> dict:
        snapshot = self._checkpoints[name]
        try:
            return self._restore(snapshot)
        except Exception:
            if not self._transaction_depth:
                self._invalidate()
            raise

    def _renderer_get(self):
        if self._renderer is None:
            from .observation import ObservationRenderer  # noqa: PLC0415

            self._renderer = ObservationRenderer(self)
        return self._renderer

    def _observe(self, **kwargs) -> dict:
        return self._renderer_get().observe(**kwargs)

    def _record(self, **kwargs) -> dict:
        return self._renderer_get().record(**kwargs)

    def _filmstrip(self, **kwargs) -> dict:
        with self.watch.stepping("filmstrip"):
            return self._undoable("filmstrip", lambda: self._renderer_get().filmstrip(**kwargs))

    def _guide(self) -> dict:
        return {"guide": self.guide}

    _MAX_SHOWN_IMAGES: ClassVar[int] = 8
    _MAX_CELL_SOURCES: ClassVar[int] = 512

    def show(self, image: Any, label: str | None = None) -> None:
        """Attach an image to the current trusted-execution response.

        Accepts HxW/HxWx3/HxWx4 arrays (uint8, or floats in [0, 1]), Pillow
        images, matplotlib figures, PNG bytes, image file paths, or an
        ``observe``/``filmstrip`` result (every page of a paged filmstrip). At most eight images of up to four
        megapixels each are returned per call; MCP clients see them inline.

        Args:
            image: Image-like object to display.
            label: Optional caption drawn on the image and returned as text.
        """
        from .imaging import draw_label, encode_png, to_rgb  # noqa: PLC0415

        if self._shown_images is None:
            raise RuntimeError("show() is only available during trusted execution")
        if len(self._shown_images) >= self._MAX_SHOWN_IMAGES:
            raise ValueError(f"At most {self._MAX_SHOWN_IMAGES} images can be shown per execute call")
        if isinstance(image, dict) and image.get("images"):
            # A paged filmstrip: every page, the first one labeled.
            self.show({"image_base64": image["image_base64"]}, label)
            for page in image["images"]:
                self.show(page)
            return
        if isinstance(image, dict) and "image_base64" in image:
            rgb = to_rgb(base64.b64decode(image["image_base64"]))
        else:
            rgb = to_rgb(image)
        if rgb.shape[0] * rgb.shape[1] > 4_194_304:
            raise ValueError("Shown images are limited to four megapixels; downsample first")
        if label:
            rgb = rgb.copy()
            draw_label(rgb, str(label))
        self._shown_images.append(
            {
                "image_base64": base64.b64encode(encode_png(rgb)).decode("ascii"),
                "mime_type": "image/png",
                "label": label,
            }
        )

    _SCENE_BINDINGS: ClassVar[tuple[str, ...]] = (
        "model",
        "solver",
        "state",
        "state_next",
        "control",
        "contacts",
        "viewer",
    )
    _HELPERS: ClassVar[tuple[str, ...]] = (
        "show",
        "rollout",
        "health",
        "solver_contacts",
        "solver_params",
        "render",
        "contacts_between",
        "persist",
        "persist_source",
    )
    """Session methods bound under the same name in trusted execution."""
    _WORKSPACE_BINDINGS: ClassVar[tuple[str, ...]] = ("session", *_SCENE_BINDINGS, "np", "wp", "newton", *_HELPERS)
    _EXPRESSION_RESULT = "__newton_expression_result__"
    _CLASS_HOOK = "__newton_cell_class__"

    def _refresh_workspace(self) -> None:
        if self._workspace_module is None or self._closed:
            return
        self._workspace.update({name: getattr(self, name) for name in self._SCENE_BINDINGS})
        self._workspace.update(self.namespace)
        import newton  # noqa: PLC0415

        self._workspace.update({name: getattr(self, name) for name in self._HELPERS})
        self._workspace.update(
            session=self,
            np=np,
            wp=wp,
            newton=newton,
            __name__=self._workspace_name,
            __package__=None,
            __loader__=None,
            __spec__=None,
            __builtins__=builtins.__dict__,
        )

    def _session_names(self) -> set[str]:
        """Names bound to this session's own objects, which worker calls resolve in their own session."""
        return {*self._WORKSPACE_BINDINGS, *self.namespace, "collision_pipeline"}

    def _background_report(self) -> dict:
        report = {}
        jobs = self.jobs.report()
        if jobs:
            report["jobs"] = jobs
        events = self.workers.drain_events() if self.workers is not None else []
        if events:
            report["workers"] = events
        return report

    def _clear_workspace(self) -> None:
        for filename in self._workspace_sources:
            self._forget_cell(filename)
        self._workspace_sources.clear()
        self._workspace.clear()
        self._workspace_generation += 1
        self._execution_error = None
        if self._workspace_warp_module is not None:
            # This module contains only executor-owned definitions. Escaped
            # references and captured graphs remain the application's responsibility.
            self._workspace_warp_module.unload()
            self._workspace_warp_module.kernels.clear()
            self._workspace_warp_module.functions.clear()
            self._workspace_warp_module.structs.clear()
        self._refresh_workspace()

    def _workspace_info(self) -> dict:
        reserved = {*self._WORKSPACE_BINDINGS, *self.namespace, "_"}
        variables = sorted(
            name
            for name in self._workspace
            if isinstance(name, str) and name not in reserved and not name.startswith("__")
        )
        return {
            "generation": self._workspace_generation,
            "cell_count": self._cell_count,
            "variables": [name[:128] for name in variables[:100]],
            "variable_count": len(variables),
            "variables_truncated": len(variables) > 100 or any(len(name) > 128 for name in variables[:100]),
            "source_cells": len(self._workspace_sources),
            "last_error": self._execution_error,
        }

    def _cache_cell_source(self, filename: str, code: str) -> None:
        register_cell_source(filename, code)
        if filename not in self._workspace_sources:
            self._workspace_sources.append(filename)
        # Cells stay cached so inspect.getsource(), tracebacks, persist_source(), and workers.map(fn) keep
        # working; beyond the limit, the oldest cells no reachable workspace definition comes from go first.
        self._workspace_sources = retain_cell_sources(
            self._workspace_sources, self._workspace, maximum=self._MAX_CELL_SOURCES, forget=self._forget_cell
        )

    def _forget_cell(self, filename: str) -> None:
        forget_cell_source(filename)
        module_name = self._cell_modules.pop(filename, None)
        if module_name is not None:
            sys.modules.pop(module_name, None)

    def _bind_cell_class(self, cls: Any, *, filename: str, module_name: str) -> None:
        from .persist import bind_cell_class  # noqa: PLC0415

        # Source lookup is a convenience; it must never fail the cell that defines the class.
        with contextlib.suppress(Exception):
            if bind_cell_class(cls, self._workspace, self._workspace_name, filename, module_name):
                self._cell_modules[filename] = module_name

    def _execution_diagnostic(self, error: BaseException, filename: str) -> dict:
        from .rollback import user_frame  # noqa: PLC0415

        cell_prefix = f"<{self._workspace_name}:"
        # Cell frames plus frames in user files such as a hosted script, skipping Newton/Warp internals.
        frames = [
            {
                "cell" if frame.filename.startswith(cell_prefix) else "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
                "source": (frame.line or "")[:200],
            }
            for frame in traceback.extract_tb(error.__traceback__)
            if frame.filename.startswith(cell_prefix)
            or (user_frame(frame.filename) and not frame.filename.startswith("<"))
        ][-8:]
        cell_frames = [frame for frame in frames if "cell" in frame]
        line = getattr(error, "lineno", None) or (cell_frames[-1]["line"] if cell_frames else None)
        return {
            "type": type(error).__name__,
            "message": str(error)[:4096],
            "cell": filename,
            "line": line,
            "frames": frames,
        }

    def _execute(self, *, code: str, reset_namespace: bool = False) -> dict:
        if not self.allow_execute:
            raise PermissionError("Trusted Python execution was not enabled by the embedding application")
        if not isinstance(code, str) or len(code) > 65536:
            raise ValueError("code must be a string of at most 65536 characters")
        if not isinstance(reset_namespace, bool):
            raise ValueError("reset_namespace must be a boolean")
        self._cell_count += 1
        filename = cell_filename(self._workspace_name, self._cell_count)
        try:
            tree = ast.parse(code, filename=filename, mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                expression = tree.body[-1]
                tree.body[-1] = ast.copy_location(
                    ast.Assign(targets=[ast.Name(id=self._EXPRESSION_RESULT, ctx=ast.Store())], value=expression.value),
                    expression,
                )
                ast.fix_missing_locations(tree)
            defines_classes = any(isinstance(node, ast.ClassDef) for node in tree.body)
            if defines_classes:
                from .persist import instrument_cell_classes  # noqa: PLC0415

                instrument_cell_classes(tree, self._CLASS_HOOK)
            compiled = compile(tree, filename, "exec")
        except SyntaxError as error:
            self._execution_error = self._execution_diagnostic(error, filename)
            self.last_error = f"Python did not execute: {error}"[:4096]
            raise
        if reset_namespace:
            self._clear_workspace()
        if self._workspace_module is None:
            self._workspace_module = ModuleType(self._workspace_name)
            self._workspace_module.__dict__.update(self._workspace)
            self._workspace = self._workspace_module.__dict__
            sys.modules[self._workspace_name] = self._workspace_module
            self._workspace_warp_module = wp.get_module(self._workspace_name)
        self._refresh_workspace()
        self._cache_cell_source(filename, code)
        scope = self._workspace
        scope.pop(self._EXPRESSION_RESULT, None)
        if defines_classes:
            # Gives cell classes a source file for inspect.getsource() (see persist.bind_cell_class).
            module_name = f"{self._workspace_name}.cell_{self._cell_count}"
            scope[self._CLASS_HOOK] = functools.partial(
                self._bind_cell_class, filename=filename, module_name=module_name
            )
        output = self._Output(16384)
        nested = self._transaction_depth > 0
        undo = self._undo_point()
        self._shown_images = []
        self._transaction_depth += 1
        completed = False
        if not nested:
            self._cell_undo = undo
            self.watch.begin()
        try:
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(compiled, scope)
                completed = True
            finally:
                scope.pop(self._CLASS_HOOK, None)
                self._refresh_workspace()
            # Re-recording CUDA graphs after the cell can fail too; it then rolls back like the cell.
            note = self.execute_callback(self) if self.execute_callback is not None and self.valid else None
            if not nested:
                edits = self.watch.end() if self.valid else self.watch.abort()
                note = "\n".join(part for part in (note, edits) if part) or None
        except (Exception, SystemExit) as error:
            self._shown_images = None
            self._execution_error = self._execution_diagnostic(error, filename)
            diagnostic = self._execution_error
            if completed:
                message = f"The cell completed, then updating the application (e.g. re-recording CUDA graphs) raised {diagnostic['type']}: {diagnostic['message']}"
            elif isinstance(error, SystemExit):
                message = f"SystemExit({error.code!r}) raised at line {diagnostic['line']}; the session keeps running"
            else:
                message = f"Python {diagnostic['type']} at line {diagnostic['line']}: {diagnostic['message']}"
            message = message[:4096].rstrip(". ")
            if not nested:
                # The rollback restores model arrays itself; the next cell starts from a fresh baseline.
                self.watch.abort()
            outcome = self._roll_back(undo, nested=nested)
            raise RuntimeError(
                f"{message}. {outcome} Python variables assigned before the error are kept. "
                f"frames={json.dumps(diagnostic['frames'])}; stdout={''.join(output.parts)!r}"
                f"{self._background_suffix()}"
            ) from error
        finally:
            self._transaction_depth -= 1
            if not nested:
                self._cell_undo = None
        self.revision += 1
        if not nested:
            # Nothing touched the model since the check that ended the cell.
            self.watch.settle()
        if self.valid:
            self.last_error = None
            self._execution_error = None
        if self._renderer is not None:
            self._renderer.invalidate()
        images, self._shown_images = self._shown_images, None
        # Only the last expression is returned; a variable named ``result`` is an ordinary variable.
        value = scope.pop(self._EXPRESSION_RESULT, None)
        if value is not None:
            scope["_"] = value
        representation = None
        try:
            result = _result_json(value)
            if len(json.dumps(result)) > 65536:
                raise ValueError("result exceeds 65536 characters")
        except Exception:
            # The cell completed; a value that cannot be converted is summarized instead of failing it.
            result = None
            representation = _result_summary(value)
        return {
            **self._status(),
            "result": result,
            "result_repr": representation,
            "stdout": "".join(output.parts),
            "truncated": output.truncated,
            "workspace": self._workspace_info(),
            **({"note": str(note)[:4096]} if note else {}),
            **({"images": images} if images else {}),
            **self._background_report(),
        }

    def _background_suffix(self) -> str:
        report = self._background_report()
        return f"; background={json.dumps(_result_json(report))}" if report else ""

    def _rebuild(self, *, reset_namespace: bool = False, **kwargs) -> dict:
        if self.rebuild_callback is None:
            raise ValueError("No rebuild callback was registered")
        from .rollback import describe_exception  # noqa: PLC0415

        try:
            bindings = self.rebuild_callback(self, **kwargs)
            if not isinstance(bindings, dict):
                raise ValueError("Rebuild callback must return replacement keyword bindings")
        except (Exception, SystemExit) as error:
            # Nothing was replaced yet: the previous scene, its checkpoints, and its graphs stay in place.
            previous = "keeps running unchanged" if self.valid else "is unchanged and still invalid"
            raise RuntimeError(
                f"Rebuild failed; the previous scene {previous}.\n{describe_exception(error)}"
            ) from error
        try:
            self.replace(**bindings, keep_workspace=not reset_namespace)
            if self.batch_callback is not None:
                self.batch_callback(self)
        except Exception:
            self._invalidate(requires_rebuild=True)
            raise
        workers = None
        if self.workers is not None and not kwargs.get("restart"):
            # Started workers rebuild in the background after this session succeeded, from the kernels it just
            # compiled; workers not started yet only record the arguments.
            workers = self.workers.rebuild({**kwargs, "reset_namespace": reset_namespace})
            workers = workers if workers["pending"] else None
        # Rebuilds repeat often while iterating on a script; the full describe payload (guide,
        # operation list, limits) would be re-sent into the agent's context every time.
        return {
            **self._status(),
            "dt": self.dt,
            "counts": {
                name: int(getattr(self.model, name))
                for name in ("world_count", "body_count", "shape_count", "joint_count", "joint_dof_count")
            },
            "solver": self._describe_solver(self.solver),
            **({"workers_rebuild": workers} if workers is not None else {}),
            **self._background_report(),
        }

    def _query(
        self,
        *,
        root: str = "state",
        field: str | None = None,
        entry: str | list[str] | None = None,
        offset: int = 0,
        limit: int = 100,
        world: int | list[int] | None = None,
        body: int | list[int] | None = None,
        shape: int | list[int] | None = None,
        joint: int | list[int] | None = None,
    ) -> dict:
        _integer(offset, "offset", 0, 2**31 - 1)
        _integer(limit, "limit", 1, 256)
        model, solver, state, _ = self._bindings(entry)
        roots = {
            "model": model,
            "solver": solver,
            "state": state,
            "control": self.control,
            "collision": self.collision_pipeline,
        }
        if root not in roots:
            raise ValueError(f"Unknown root {root!r}")
        if entry and root in {"control", "collision"}:
            raise ValueError("Entry-local control/collision queries are unavailable; query the top-level binding")
        obj = roots[root]
        if field is None:
            if any(value is not None for value in (world, body, shape, joint)):
                raise ValueError("Entity filters require a field")
            names = sorted(name for name in vars(obj) if not name.startswith("_"))
            return {
                **self._status(),
                "fields": names[offset : offset + limit],
                "total_matched": len(names),
                "offset": offset,
            }
        value = _field(obj, field)
        if value is None or isinstance(value, str | int | float | bool):
            if any(value is not None for value in (world, body, shape, joint)):
                raise ValueError("Scalar fields do not support entity filters")
            return {**self._status(), "field": field, "value": _json(value)}
        if not isinstance(value, wp.array | np.ndarray | list | tuple):
            raise ValueError("Select a numeric, string, or array field inside this object")
        is_warp = isinstance(value, wp.array)
        data = None if is_warp else _array(value)
        if not is_warp and (data.ndim == 0 or data.dtype.kind not in "biufUS"):
            raise ValueError("Only numeric and string arrays can be queried")
        count = value.shape[0] if is_warp else len(data)
        try:
            frequency = model.get_attribute_frequency(field) if root in {"model", "state", "control"} else None
        except KeyError:
            frequency = None
        if root == "model" and field == "gravity":
            # Gravity's legacy ONCE metadata predates its local/global world rows.
            frequency = Model.AttributeFrequency.WORLD
        if all(v is None for v in (world, body, shape, joint)):
            total = count
            page_ids = np.arange(min(offset, count), min(offset + limit, count), dtype=np.int32)
        else:
            if count > 2_000_000:
                raise ValueError("Filtered row map exceeds two million rows; use unfiltered pagination")
            indices = self._selection(model, frequency, count, world=world, body=body, shape=shape, joint=joint)
            total = len(indices)
            page_ids = indices[offset : offset + limit].astype(np.int32)
        if is_warp:
            shape_full = tuple(value.shape) + tuple(getattr(value.dtype, "_shape_", ()))
            row_width = math.prod(shape_full[1:])
            if len(page_ids) * row_width > 4096:
                raise ValueError("Query page exceeds 4096 components; reduce limit")
            if len(page_ids):
                gather_ids = wp.array(page_ids, dtype=wp.int32, device=value.device)
                page = wp.indexedarray(value, [gather_ids] + [None] * (value.ndim - 1)).contiguous().numpy()
            else:
                scalar_dtype = getattr(value.dtype, "_wp_scalar_type_", value.dtype)
                page = np.empty((0, *shape_full[1:]), dtype=wp.dtype_to_numpy(scalar_dtype))
        else:
            shape_full = data.shape
            page = data[page_ids]
            if page.size > 4096:
                raise ValueError("Query page exceeds 4096 components; reduce limit")
        result = {
            **self._status(),
            "root": root,
            "field": field,
            "frequency": getattr(frequency, "name", frequency),
            "shape": list(shape_full),
            "dtype": str(page.dtype),
            "indices": page_ids.tolist(),
            "values": _json(page),
            "offset": offset,
            "total_matched": total,
            "next_offset": offset + len(page_ids) if offset + len(page_ids) < total else None,
        }
        if page.dtype.kind in "biuf" and page.size:
            finite = np.isfinite(page)
            result["page_statistics"] = {
                "finite": int(finite.sum()),
                "nonfinite": int((~finite).sum()),
                "minimum": float(page[finite].min()) if finite.any() else None,
                "maximum": float(page[finite].max()) if finite.any() else None,
            }
        return result

    @staticmethod
    def _ids(value: int | list[int], name: str, maximum: int, *, minimum: int = 0) -> np.ndarray:
        values = value if isinstance(value, list) else [value]
        if not values or len(values) > 256:
            raise ValueError(f"{name} must select between 1 and 256 indices")
        return np.asarray([_integer(item, name, minimum, maximum - 1) for item in values], dtype=np.int64)

    def _selection(self, model: Any, frequency: Any, count: int, *, world=None, body=None, shape=None, joint=None):
        indices = np.arange(count)
        if all(value is None for value in (world, body, shape, joint)):
            return indices
        name = getattr(frequency, "name", frequency)
        if name is None or name == "ONCE":
            raise ValueError("This field has no entity frequency; filters would be ambiguous")
        mask = np.ones(count, dtype=bool)
        domain = {
            "BODY": "body",
            "SHAPE": "shape",
            "JOINT": "joint",
            "PARTICLE": "particle",
            "ARTICULATION": "articulation",
            "WORLD": "world",
        }.get(name)
        joint_ids = None
        if name in {"JOINT_DOF", "JOINT_COORD"}:
            starts = _array(model.joint_qd_start if name == "JOINT_DOF" else model.joint_q_start)
            joint_ids = np.searchsorted(starts[1:], indices, side="right")
            domain = "joint"
        if world is not None:
            selected = self._ids(world, "world", model.world_count, minimum=-1)
            if domain == "world":
                worlds = indices.copy()
                if count == model.world_count + 1:
                    worlds[-1] = -1
                elif count == 1 and model.world_count == 1 and -1 in selected:
                    selected = np.unique(np.append(selected, 0))
            elif domain is not None:
                worlds = _array(getattr(model, f"{domain}_world"))
                worlds = worlds[joint_ids] if joint_ids is not None else worlds
            elif isinstance(frequency, str) and frequency in model.custom_frequency_articulation:
                owners = _array(model.custom_frequency_articulation[frequency])
                worlds = np.full(len(owners), -1)
                valid = owners >= 0
                worlds[valid] = _array(model.articulation_world)[owners[valid]]
            else:
                raise ValueError(f"World filtering is unavailable for frequency {name!r}")
            if len(worlds) != count:
                raise ValueError("Structured/sentinel arrays cannot be filtered as plain entity rows")
            mask &= np.isin(worlds, selected)
        for filter_name, selection in (("body", body), ("shape", shape), ("joint", joint)):
            if selection is None:
                continue
            selected = self._ids(selection, filter_name, getattr(model, f"{filter_name}_count"))
            if filter_name == domain:
                row_ids = joint_ids if joint_ids is not None else indices
            elif filter_name == "body" and domain == "shape":
                row_ids = _array(model.shape_body)
            elif filter_name == "body" and domain == "joint":
                row_ids = _array(model.joint_child)
                row_ids = row_ids[joint_ids] if joint_ids is not None else row_ids
            elif filter_name == "shape" and domain == "body":
                selected = _array(model.shape_body)[selected]
                row_ids = indices
            elif filter_name == "joint" and domain == "body":
                selected = _array(model.joint_child)[selected]
                row_ids = indices
            else:
                raise ValueError(f"{filter_name} filtering is unavailable for frequency {name!r}")
            if len(row_ids) != count:
                raise ValueError("Structured/sentinel arrays cannot be filtered as plain entity rows")
            mask &= np.isin(row_ids, selected)
        return indices[mask]

    _EDIT_FLAGS: ClassVar[dict[str, ModelFlags]] = {
        **dict.fromkeys(
            (
                "joint_target_ke",
                "joint_target_kd",
                "joint_damping",
                "joint_armature",
                "joint_friction",
                "joint_effort_limit",
                "joint_velocity_limit",
                "joint_limit_ke",
                "joint_limit_kd",
                "joint_limit_lower",
                "joint_limit_upper",
            ),
            ModelFlags.JOINT_DOF_PROPERTIES,
        ),
        **dict.fromkeys(
            (
                "shape_material_mu",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_kf",
                "shape_material_restitution",
                "shape_material_mu_torsional",
                "shape_material_mu_rolling",
            ),
            ModelFlags.SHAPE_PROPERTIES,
        ),
        "body_mass": ModelFlags.BODY_INERTIAL_PROPERTIES,
        "gravity": ModelFlags.MODEL_PROPERTIES,
    }

    def _edit(self, *, patches: list[dict], flags: int | list[str] | None = None) -> dict:
        if not isinstance(patches, list) or not 1 <= len(patches) <= 32:
            raise ValueError("Provide between 1 and 32 patches")
        prepared = {}
        inferred = 0
        for patch in patches:
            if not isinstance(patch, dict) or set(patch) - {"root", "field", "indices", "values"}:
                raise ValueError("Patch keys are root, field, indices, values")
            root, field = patch.get("root", "model"), patch.get("field")
            if root not in {"model", "control"}:
                raise ValueError(
                    "Only model/control edits are supported; use reset or trusted execute for state changes"
                )
            if root == "model" and field not in self._EDIT_FLAGS:
                raise ValueError(f"Model field {field!r} is not editable; rebuild topology or use trusted execute")
            target = _field(getattr(self, root), field)
            if not isinstance(target, wp.array):
                raise ValueError("Edits require an existing Warp array")
            key = (root, field)
            if key not in prepared:
                data = _array(target).copy()
                if data.dtype.kind != "f":
                    raise ValueError("Only floating-point parameter and control arrays are editable")
                prepared[key] = (target, data)
            target, data = prepared[key]
            ids = self._ids(patch["indices"], "indices", len(data)) if "indices" in patch else np.arange(len(data))
            if len(ids) != len(np.unique(ids)):
                raise ValueError("Duplicate patch indices are not supported")
            values = np.asarray(patch["values"])
            if values.dtype.kind not in "iuf" or values.size > 4096 or not np.all(np.isfinite(values)):
                raise ValueError("Patch values must contain at most 4096 finite numeric components")
            if values.shape != data[ids].shape:
                raise ValueError(
                    f"Expected values shape {data[ids].shape}, got {values.shape}; broadcasting is not supported"
                )
            with np.errstate(over="ignore"):
                cast = values.astype(data.dtype)
            if not np.all(np.isfinite(cast)):
                raise ValueError("Patch values overflow the target dtype")
            if root == "model":
                inferred |= int(self._EDIT_FLAGS[field])
                if field not in {"gravity", "joint_limit_lower", "joint_limit_upper"} and np.any(cast < 0):
                    raise ValueError(f"{field} must be nonnegative")
                if field == "shape_material_restitution" and np.any(cast > 1):
                    raise ValueError("Restitution must be in [0, 1]")
                if field == "body_mass" and (np.any(cast <= 0) or np.any(data[ids] <= 0)):
                    raise ValueError(
                        "Mass edits require positive old/new masses; rebuild to change static/dynamic bodies"
                    )
            data[ids] = cast
        if ("control", "joint_target_q") in prepared and self.model.use_coord_layout_targets:
            targets = prepared[("control", "joint_target_q")][1]
            starts = _array(self.model.joint_q_start)
            for joint, kind in enumerate(_array(self.model.joint_type)):
                if kind in (JointType.FREE, JointType.DISTANCE, JointType.BALL):
                    start = int(starts[joint]) + (0 if kind == JointType.BALL else 3)
                    norm = float(np.linalg.norm(targets[start : start + 4]))
                    if not math.isclose(norm, 1.0, abs_tol=1e-4):
                        raise ValueError(f"Joint {joint} target quaternion must have unit length")
        lower_key, upper_key = ("model", "joint_limit_lower"), ("model", "joint_limit_upper")
        if lower_key in prepared or upper_key in prepared:
            lower = prepared[lower_key][1] if lower_key in prepared else _array(self.model.joint_limit_lower)
            upper = prepared[upper_key][1] if upper_key in prepared else _array(self.model.joint_limit_upper)
            if np.any(lower > upper):
                raise ValueError("Joint lower limits must not exceed upper limits")
        if ("model", "body_mass") in prepared:
            mass = prepared[("model", "body_mass")][1]
            original = _array(self.model.body_mass)
            ratio = np.ones_like(mass)
            changed = mass != original
            ratio[changed] = mass[changed] / original[changed]
            inverse = _array(self.model.body_inv_mass).copy()
            dynamic = changed & (inverse > 0)
            inverse[dynamic] = 1.0 / mass[dynamic]
            for field, data in (
                ("body_inv_mass", inverse),
                ("body_inertia", _array(self.model.body_inertia) * ratio[:, None, None]),
                ("body_inv_inertia", _array(self.model.body_inv_inertia) / ratio[:, None, None]),
            ):
                if not np.all(np.isfinite(data)):
                    raise ValueError("Mass edit produces nonfinite dependent inertia")
                prepared[("model", field)] = (_field(self.model, field), data)
        requested = 0
        if isinstance(flags, list):
            for flag in flags:
                requested |= int(ModelFlags[flag])
        elif flags is not None:
            requested = _integer(flags, "flags", 0, int(ModelFlags.ALL))
        if flags is not None and inferred & requested != inferred:
            raise ValueError("Explicit flags must include every inferred notification category")
        notification = inferred | requested
        try:
            for target, data in prepared.values():
                target.assign(data)
            if notification:
                # Coupled solvers forward this once to their children.
                self.solver.notify_model_changed(notification)
            self.revision += 1
            if self._renderer is not None:
                self._renderer.invalidate()
        except Exception:
            self._invalidate(requires_rebuild=True)
            raise
        return {**self._status(), "fields": [f"{root}.{field}" for root, field in prepared], "flags": notification}

    def _collide(self) -> dict:
        self.collision_pipeline.collide(self.state, self.contacts)
        self._contact_frame = self.frame
        self._contact_revision = self.revision
        return {**self._status(), "source": "collision_pipeline"}

    def rollout(
        self,
        frames: int | None = None,
        *,
        seconds: float | None = None,
        record: dict[str, Callable | str] | None = None,
        every: int = 1,
        start: bool | str = False,
        until: Callable | str | None = None,
    ) -> dict:
        """Step the scene and record time series in one call (trusted execution helper).

        Args:
            frames: Number of steps; alternatively give ``seconds``.
            seconds: Simulated duration [s], rounded to whole steps.
            record: Series to sample, as ``name: callable`` (taking no arguments or
                the session) or a Python expression evaluated in the workspace
                (``"state.body_q.numpy()[3, 2]"``). A probe that returns a dictionary
                records one series per numeric key, named ``name.key`` and also
                available as ``result[name][key]``.
            every: Sample every ``every`` steps (the final step is always sampled).
            start: ``True`` resets to the initial state, a string restores that
                checkpoint, ``False`` continues from the current state.
            until: Stop early once this callable/expression is truthy; the
                reason is reported in ``stopped``.

        Returns:
            Dictionary with ``t`` [s] and one NumPy array per recorded name
            (stacked over samples), plus ``frames`` and ``stopped``.
        """
        self._assert_owner()
        if frames is None:
            if seconds is None:
                raise ValueError("Give frames or seconds")
            frames = max(1, round(float(seconds) / self.dt))
        _integer(frames, "frames", 1, 1_000_000)
        _integer(every, "every", 1, 1_000_000)
        probes = {}
        for name, probe in (record or {}).items():
            if isinstance(probe, str):
                code = compile(probe, f"<rollout:{name}>", "eval")
                probes[name] = lambda _session, code=code: eval(code, self._eval_scope())
            else:
                probes[name] = _session_callable(probe)
        stop = _session_callable(until) if callable(until) else until
        if isinstance(until, str):
            stop_code = compile(until, "<rollout:until>", "eval")

            def stop(_session):
                return eval(stop_code, self._eval_scope())

        series = {name: [] for name in probes}
        times, stopped = [], None

        def sample():
            times.append(self.time)
            for name, probe in probes.items():
                value = probe(self)
                if isinstance(value, dict):
                    series[name].append(value)
                else:
                    series[name].append(value.numpy() if isinstance(value, wp.array) else np.asarray(value))

        def run() -> int:
            nonlocal stopped
            if start is True:
                self._reset()
            elif isinstance(start, str):
                self._restore_named(name=start)
            sample()
            for index in range(frames):
                # One batch for the whole rollout: application settings are checked before its first step.
                self._advance(1, self.dt, first=index == 0, last=False)
                last = index == frames - 1
                done = bool(stop(self)) if stop is not None else False
                if done or last or (index + 1) % every == 0:
                    sample()
                if done:
                    stopped = f"until at t={self.time:.4g} s"
                    break
            if self.batch_callback is not None:
                self.batch_callback(self)
            return index + 1

        with self.watch.stepping("rollout()"):
            count = self._undoable("rollout", run)
        result = {"t": np.asarray(times)}
        for name, values in series.items():
            if values and isinstance(values[0], dict):
                # Dictionary probes become one series per numeric key, e.g. "grip.normal_force", also reachable as
                # result["grip"]["normal_force"].
                nested = {}
                for key in values[0]:
                    column = [v.get(key) for v in values]
                    if all(isinstance(x, (int, float, np.number)) or x is None for x in column):
                        nested[key] = np.asarray([np.nan if x is None else x for x in column], dtype=float)
                        result[f"{name}.{key}"] = nested[key]
                result[name] = nested
            else:
                result[name] = np.stack(values)
        result.update(frames=count, stopped=stopped)
        return result

    def _eval_scope(self) -> dict:
        if self._workspace_module is not None:
            return self._workspace
        # Outside trusted execution, expressions still see the live bindings.
        return {name: getattr(self, name) for name in ("model", "solver", "state", "control", "contacts")} | {
            "session": self,
            "np": np,
            "wp": wp,
        }

    def health(
        self,
        solver: Any = None,
        state: Any = None,
        *,
        per_world: bool = True,
        twins: bool = False,
        penetration: float = 0.01,
        twins_tolerance: float = 1e-3,
    ) -> dict:
        """Check state and solver for non-finite values, runaway speeds, buffer overflow, and deep penetration.

        Works on any solver object, including ones built in trusted execution; MuJoCo solvers add
        per-world checks of their own data, ``njmax``/``nconmax`` buffers, and penetrating shape pairs.

        Args:
            solver: Solver to inspect (default: the session's).
            state: State to inspect (default: the session's).
            per_world: Name the worlds behind each finding.
            twins: Compare the worlds' joint states with each other, for worlds built identical.
            penetration: Overlap [m] above which contacts are reported by shape pair.
            twins_tolerance: Deviation [m or rad, and per second for velocities] that counts as disagreement.

        Returns:
            ``{"ok", "warnings", "stats", "checked", ...}`` with ``worlds``, ``penetration`` and
            ``unsupported`` entries when they apply.
        """
        from .diagnostics import health  # noqa: PLC0415

        self._assert_owner()
        return health(
            self,
            solver,
            state,
            per_world=per_world,
            twins=twins,
            penetration=penetration,
            twins_tolerance=twins_tolerance,
        )

    def solver_params(
        self, kind: str, select: str | list[str] | None = None, world: int = 0, *, solver: Any = None, limit: int = 64
    ) -> dict:
        """What the solver integrates for one kind of entity, where each value comes from, and what refreshes it.

        For :class:`~newton.solvers.SolverMuJoCo` each row holds the compiled MuJoCo value, ``from``
        (the Newton model array and index it is computed from), and ``pending`` (values whose model
        array differs from the compiled value, i.e. edits no ``notify_model_changed`` has applied).
        The result also lists the :class:`~newton.ModelFlags` that refresh each source, whether the
        MuJoCo field can differ between worlds, and fields that are read only at construction or
        not at all. Other solvers report the Newton model values and say that compiled values are
        unavailable.

        Args:
            kind: ``"actuator"``, ``"joint"``, ``"geom"``, ``"body"``, ``"equality"``, or ``"option"``.
            select: Label glob(s) matched against full labels or their last path component; a
                pattern without wildcards also matches as a substring of the last component.
            world: World whose rows are reported.
            solver: Solver to inspect (default: the session's); its own ``model`` is used.
            limit: Maximum number of rows.

        Returns:
            Dictionary with ``rows`` (``options`` for ``kind="option"``) and the facts above.
        """
        from .solverview import solver_params  # noqa: PLC0415

        self._assert_owner()
        solver = self.solver if solver is None else solver
        model = getattr(solver, "model", None) or self.model
        return solver_params(model, solver, kind, select, world=world, limit=limit)

    def render(self, *, metadata: bool = False, **options):
        """Render the current state to an RGB array without PNG encoding (trusted execution helper).

        Accepts the camera and rendering options of ``observe`` (``view``, ``eye``/``target``,
        ``pose`` or ``camera_body``/``camera_offset``, ``fov_y`` or ``intrinsics``, ``width``/``height``,
        ``world_id``, ``backend``, ``channel``, ``environment``, ...), which makes it the fast path for
        fitting loops. ``intrinsics`` may also be a :class:`~newton.sensors.SensorCamera.Intrinsics`.

        Args:
            metadata: Also return the observation metadata (camera pose, settings).

        Returns:
            ``uint8`` array of shape ``(height, width, 3)``, or ``(image, metadata)``.
        """
        self._assert_owner()
        image, info = self._renderer_get()._single(**options)
        return (image, info) if metadata else image

    def contacts_between(self, a, b=None, *, detail: bool = False) -> dict:
        """Contact count, solver normal and friction force, slip speed, and penetration between two shape sets.

        ``a`` and ``b`` select shapes by label substring (last path component of the shape's or its
        body's label), shape index, ``{"shape": ...}``, ``{"body": ...}``, or a list of these; ``b=None``
        means everything else. Returns flat scalars, so it can be recorded over time with
        ``rollout(record={"grip": lambda: contacts_between("finger", "apple")})`` (series ``grip.count``,
        ``grip.normal_force``, ...). ``detail=True`` adds per-body-pair counts and forces (``by_body``).
        """
        from .diagnostics import contacts_between  # noqa: PLC0415

        self._assert_owner()
        return contacts_between(self, a, b, detail=detail)

    def persist(
        self,
        name: str,
        value: Any = _OMITTED,
        *,
        rebuild: bool = True,
        check: str | Callable | None = None,
        tolerance: float = 1e-6,
        path: str | Path | None = None,
    ) -> dict:
        """Write a value into the script's module-level ``NAME = <literal>`` assignment (trusted execution helper).

        Only the assignment's value changes; the rest of the file stays byte for byte. When the old and
        new values are dictionaries with the same keys or sequences of the same length, only the
        differing entries are rewritten, so comments inside the literal remain. Refuses if ``name`` is
        not assigned exactly once at module level, or if its current value is not a literal. Prints a
        unified diff and saves the previous file under ``<artifact_directory>/persist/``.

        Args:
            name: Module-level variable, such as ``"PARAMS"``.
            value: New value: dicts, lists, tuples, non-empty sets, numbers, strings, bytes, bools,
                ``None``, NumPy scalars/arrays and Warp vectors/arrays (converted to plain literals).
                Omitted: the current value of ``module.<name>`` in the hosted script module.
            rebuild: Rebuild the scene (and worker sessions) from the edited file.
            check: Expression (evaluated in the workspace) or callable evaluated before writing and
                again after the rebuild; both values are reported with ``reproduced``.
            tolerance: Relative and absolute tolerance of ``reproduced`` for numeric check values.
            path: File to edit instead of the hosted script, relative to the script's directory.

        Returns:
            ``path``, ``changed``, ``backup``, the edited ``line``, rebuild timing, and ``check``.
        """
        from .persist import MISSING, persist  # noqa: PLC0415

        self._assert_owner()
        value = MISSING if value is _OMITTED else value
        return persist(self, name, value, rebuild=rebuild, check=check, tolerance=tolerance, path=path)

    def persist_source(
        self,
        obj: Any,
        *,
        target: str | None = None,
        rebuild: bool = True,
        check: str | Callable | None = None,
        tolerance: float = 1e-6,
        path: str | Path | None = None,
    ) -> dict:
        """Replace a def or class in the script with the source of one defined in a cell (trusted execution helper).

        The script's single top-level definition with the same name (or ``target``) is replaced, from
        its first decorator to its last line, by the cell's source re-indented to match; everything else
        in the file stays byte for byte. Refuses if the target is missing, bound more than once, or not
        a def/class. Prints a unified diff and saves the previous file under
        ``<artifact_directory>/persist/``. Global names the new definition reads that the script never
        binds are listed in ``names_not_defined_in_script``; methods assigned to a class from other cells
        (not part of its class statement) in ``methods_from_other_cells``.

        Args:
            obj: Function or class (or its workspace name) defined in a trusted-execution cell.
            target: Definition to replace: ``"Name"`` or ``"Class.method"``; defaults to the object's
                qualified name. A different leaf name renames the definition to it.
            rebuild: Rebuild the scene (and worker sessions) from the edited file.
            check: Expression or callable evaluated before writing and after the rebuild; both values
                are reported with ``reproduced``.
            tolerance: Relative and absolute tolerance of ``reproduced`` for numeric check values.
            path: File to edit instead of the hosted script, relative to the script's directory.

        Returns:
            ``path``, ``changed``, ``backup``, the replaced ``lines``, rebuild timing, and ``check``.
        """
        from .persist import persist_source  # noqa: PLC0415

        self._assert_owner()
        return persist_source(self, obj, target=target, rebuild=rebuild, check=check, tolerance=tolerance, path=path)

    def solver_contacts(self, limit: int = 20) -> dict:
        """Active solver contacts grouped by shape pair, with the effective solver parameters."""
        from .diagnostics import solver_contacts  # noqa: PLC0415

        return solver_contacts(self, limit=limit)

    def contact_data(
        self,
        *,
        refresh: bool = False,
        kind: str = "rigid",
        offset: int = 0,
        limit: int = 100,
        world: int | list[int] | None = None,
        body: int | list[int] | None = None,
        shape: int | list[int] | None = None,
        entry: str | list[str] | None = None,
        include_global: bool = False,
    ) -> dict:
        """Read bounded contact rows using world-space geometry.

        Args:
            refresh: Recompute contacts for the current state before reading.
            kind: ``rigid`` or ``soft`` contact rows.
            offset: First matching row.
            limit: Maximum returned rows, at most 256.
            world: Selected world indices; ``-1`` selects global entities.
            body: Selected body indices, matching either side.
            shape: Selected shape indices, matching either side.
            entry: Optional coupled solver entry path; indices are entry-local.
            include_global: Also include contacts entirely in global world ``-1``
                when selecting a local world, matching shared scene rendering.

        Returns:
            Contact rows with positions [m], unit world normals, and rigid
            signed surface distances [m]. Stored counts and capacities identify
            possible overflow. Collision-pipeline rows are distinct from native
            solver contacts. Refresh is unavailable for coupled entry contacts.
        """
        self._assert_owner()
        _integer(offset, "offset", 0, 2_000_000)
        _integer(limit, "limit", 1, 256)
        if kind not in {"rigid", "soft"}:
            raise ValueError("kind must be rigid or soft")
        if refresh:
            if entry:
                raise ValueError("Refresh entry contacts by stepping the coupled solver")
            self._collide()
        model, _, state, contacts = self._bindings(entry)
        if contacts is None:
            raise ValueError("This solver entry does not expose Newton Contacts")
        capacity = int(getattr(contacts, f"{kind}_contact_max"))
        count = int(_array(getattr(contacts, f"{kind}_contact_count"))[0])
        available = min(max(count, 0), capacity)
        shape_body = _array(model.shape_body)
        shape_world = _array(model.shape_world)
        if kind == "rigid":
            shape0 = _array(contacts.rigid_contact_shape0)[:available]
            shape1 = _array(contacts.rigid_contact_shape1)[:available]
        else:
            shape0 = _array(contacts.soft_contact_shape)[:available]
            shape1 = np.full(available, -1)
        valid0 = (shape0 >= 0) & (shape0 < len(shape_body))
        valid1 = (shape1 >= 0) & (shape1 < len(shape_body))
        body0, body1 = np.full(available, -1), np.full(available, -1)
        world0, world1 = np.full(available, -2), np.full(available, -2)
        body0[valid0], body1[valid1] = shape_body[shape0[valid0]], shape_body[shape1[valid1]]
        world0[valid0], world1[valid1] = shape_world[shape0[valid0]], shape_world[shape1[valid1]]
        if kind == "soft":
            particles = _array(contacts.soft_contact_indices)[:available]
            particle_worlds = _array(model.particle_world)
            # A contact's soft feature belongs to one world, even for edge/face records.
            first_particle = particles[:, 0]
            valid_particle = (first_particle >= 0) & (first_particle < len(particle_worlds))
            world1[valid_particle] = particle_worlds[first_particle[valid_particle]]
        mask = valid0 & (valid1 if kind == "rigid" else True)
        for name, selected, array0, array1, total in (
            ("world", world, world0, world1, model.world_count),
            ("body", body, body0, body1, model.body_count),
            ("shape", shape, shape0, shape1, model.shape_count),
        ):
            if selected is not None:
                ids = self._ids(selected, name, total, minimum=-1 if name == "world" else 0)
                selected_mask = np.isin(array0, ids) | np.isin(array1, ids)
                if name == "world" and include_global:
                    selected_mask |= (array0 == -1) & (array1 == -1)
                mask &= selected_mask
        matched = np.flatnonzero(mask)
        selected = matched[offset : offset + limit]
        rows = []
        if kind == "rigid" and len(selected):
            if capacity * 3 > 2_000_000:
                raise ValueError("Contact capacity exceeds the host-transfer budget")
            distance = wp.empty(capacity, dtype=float, device=model.device)
            point0 = wp.empty(capacity, dtype=wp.vec3, device=model.device)
            point1 = wp.empty_like(point0)
            eval_rigid_contact_kinematics(
                model, state, contacts, out_distance=distance, out_point0_world=point0, out_point1_world=point1
            )
            distance, point0, point1 = distance.numpy(), point0.numpy(), point1.numpy()
            normal = _array(contacts.rigid_contact_normal)
            margin0, margin1 = _array(contacts.rigid_contact_margin0), _array(contacts.rigid_contact_margin1)
            rows = [
                {
                    "index": int(i),
                    "shape0": int(shape0[i]),
                    "shape1": int(shape1[i]),
                    "body0": int(body0[i]),
                    "body1": int(body1[i]),
                    "point0": point0[i].tolist(),
                    "point1": point1[i].tolist(),
                    "normal": normal[i].tolist(),
                    "distance": float(distance[i]),
                    "margin0": float(margin0[i]),
                    "margin1": float(margin1[i]),
                    "surface0": (point0[i] + normal[i] * margin0[i]).tolist(),
                    "surface1": (point1[i] - normal[i] * margin1[i]).tolist(),
                }
                for i in selected
            ]
        elif kind == "soft" and len(selected):
            particle = _array(contacts.soft_contact_particle)
            points = _array(contacts.soft_contact_body_pos)
            normals = _array(contacts.soft_contact_normal)
            poses = _array(state.body_q)
            for i in selected:
                point = points[i]
                if body0[i] >= 0:
                    point = np.asarray(wp.transform_point(wp.transform(*poses[body0[i]]), wp.vec3(*point)))
                rows.append(
                    {
                        "index": int(i),
                        "particle": int(particle[i]),
                        "shape": int(shape0[i]),
                        "body": int(body0[i]),
                        "point": point.tolist(),
                        "normal": normals[i].tolist(),
                    }
                )
        return {
            **self._status(),
            "rows": _json(rows),
            "kind": kind,
            "count": count,
            "capacity": capacity,
            "possible_overflow": count >= capacity if capacity else count > 0,
            "total_matched": len(matched),
            "offset": offset,
            "source": "coupled_entry_contacts" if entry else "collision_pipeline",
            "contact_frame": getattr(self, "_contact_frame", None) if not entry else None,
            "contact_revision": getattr(self, "_contact_revision", None) if not entry else None,
            "geometry": "world support points; signed surface gap in meters",
        }
