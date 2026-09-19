# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Owner-thread operations for an explicitly instrumented simulation."""

from __future__ import annotations

import contextlib
import io
import json
import math
import queue
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

from ..sim.collide import CollisionPipeline
from ..sim.contact_kinematics import eval_rigid_contact_kinematics
from ..sim.enums import JointType, ModelFlags, StateFlags
from ..sim.model import Model


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


class SimulationSession:
    """Own the live bindings and serialize simulation operations on one thread.

    .. experimental::

        This entire class may change without a deprecation period. Applications
        must explicitly embed a session and pump it on their simulation thread.
        Hidden solver state is reset, not checkpointed, so restore does not
        promise bitwise replay. Trusted execution is full Python, not a sandbox.

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
        allow_execute: Enable trusted, unrestricted Python execution.
        artifact_directory: Directory for observation and recording artifacts.
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
    ):
        self._owner = threading.get_ident()
        self._queue = queue.Queue(maxsize=64)
        self._queue_lock = threading.Lock()
        self._closed = False
        self._renderer = None
        self._checkpoints = {}
        self.artifact_directory = Path(artifact_directory or tempfile.mkdtemp(prefix="newton-mcp-"))
        self.dt = self._timestep(dt)
        self.step_callback = step_callback
        self.reset_callback = reset_callback
        self.rebuild_callback = rebuild_callback
        self.allow_execute = allow_execute
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
    ) -> None:
        """Replace all scene bindings and discard old snapshots in this process.

        Topology changes require a newly built model and matching solver. Any
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
        """
        self._assert_owner()
        self.paused = True
        self.valid = False
        self._requires_rebuild = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
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

        return {"state": arrays(self.state), "control": arrays(self.control), "time": self.time, "frame": self.frame}

    def _restore(self, snapshot: dict) -> dict:
        self.paused = True
        for root in ("state", "control"):
            for field, data in snapshot[root].items():
                _field(getattr(self, root), field).assign(data)
        # Reset history without overwriting the restored public state with model defaults.
        self.solver.reset(self.state, flags=StateFlags.NONE)
        self.state_next.assign(self.state)
        self.collision_pipeline.reset_contact_matching()
        self.contacts.clear(bump_generation=True)
        self._contact_frame = self._contact_revision = None
        self.time, self.frame = snapshot["time"], snapshot["frame"]
        if self.reset_callback is not None:
            self.reset_callback(self)
        self.valid = True
        self.revision += 1
        self.last_error = None
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
            except Exception as error:
                request.error = error
            finally:
                request.done.set()
        return count

    def run(self) -> None:
        """Pump requests and advance playback until closed or interrupted.

        Playback uses the configured timestep [s] without wall-clock pacing.
        Embed :meth:`pump` in an application loop for custom rendering/pacing.
        """
        self._assert_owner()
        try:
            while not self._closed:
                self.pump()
                if not self.paused:
                    try:
                        self.dispatch("step", {"count": 1})
                    except Exception as error:
                        self.last_error = str(error)[:4096]
                        if self.valid:
                            self._invalidate()
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
        if self._renderer is not None:
            self._renderer.close()

    def _status(self) -> dict:
        return {
            "time": self.time,
            "frame": self.frame,
            "revision": self.revision,
            "paused": self.paused,
            "valid": self.valid,
            "closed": self._closed,
            "last_error": self.last_error,
        }

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
                "record",
                "execute",
                "rebuild",
            ],
            "model_flags": {flag.name: int(flag) for flag in ModelFlags},
            "editable_model_fields": sorted(self._EDIT_FLAGS),
            "capabilities": {
                "execute": self.allow_execute,
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
            "record": self._record,
            "execute": self._execute,
            "rebuild": self._rebuild,
        }
        if operation not in operations:
            raise ValueError(f"Unknown operation {operation!r}")
        if not self.valid and operation not in {"describe", "query", "pause", "reset", "restore", "rebuild"}:
            raise RuntimeError("Session is invalid after a failed mutation; reset or rebuild before continuing")
        if self._requires_rebuild and operation in {"reset", "restore"}:
            raise RuntimeError("Model/solver coherence is unknown after a failed mutation; rebuild the scene")
        return operations[operation](**args)

    def _invalidate(self, *, requires_rebuild: bool = False) -> None:
        self.paused = True
        self.valid = False
        self._requires_rebuild |= requires_rebuild
        self.revision += 1

    def _step(self, *, count: int = 1, dt: float | None = None) -> dict:
        _integer(count, "count", 1, 10000)
        dt = self.dt if dt is None else self._timestep(dt)
        try:
            for _ in range(count):
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
                if self._renderer is not None:
                    self._renderer.after_step()
        except Exception:
            self._invalidate()
            raise
        return self._status()

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

    def _execute(self, *, code: str) -> dict:
        if not self.allow_execute:
            raise PermissionError("Trusted Python execution was not enabled by the embedding application")
        if not isinstance(code, str) or len(code) > 65536:
            raise ValueError("code must be a string of at most 65536 characters")
        compiled = compile(code, "<newton-mcp>", "exec")
        output = self._Output(16384)
        scope = {
            "session": self,
            "model": self.model,
            "solver": self.solver,
            "state": self.state,
            "control": self.control,
            "np": np,
            "wp": wp,
        }
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compiled, scope)
        except Exception as error:
            self._invalidate(requires_rebuild=True)
            raise RuntimeError(
                f"Execution failed and may have mutated the scene: {str(error)[:4096]}; stdout={''.join(output.parts)!r}"
            ) from error
        self.revision += 1
        if self._renderer is not None:
            self._renderer.invalidate()
        try:
            result = _json(scope.get("result"))
            if len(json.dumps(result)) > 65536:
                raise ValueError("result exceeds 65536 characters")
        except Exception as error:
            raise RuntimeError(
                f"Python completed; result cannot be returned: {str(error)[:4096]}. "
                "The scene remains valid; do not retry the mutation. Query a smaller result."
            ) from error
        return {**self._status(), "result": result, "stdout": "".join(output.parts), "truncated": output.truncated}

    def _rebuild(self, **kwargs) -> dict:
        if self.rebuild_callback is None:
            raise ValueError("No rebuild callback was registered")
        try:
            bindings = self.rebuild_callback(self, **kwargs)
            if not isinstance(bindings, dict):
                raise ValueError("Rebuild callback must return replacement keyword bindings")
            self.replace(**bindings)
        except Exception:
            self._invalidate(requires_rebuild=True)
            raise
        return self._describe()

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
