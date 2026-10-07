# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched evaluation bound to a live session, and checkpoints of Python objects.

``evaluate`` and ``branch`` run :class:`newton.utils.BatchRollout` on a model
with N copies of the scene that a hosted script built, starting from the live
state. :class:`SceneCapture` records that scene while ``Example()`` is
constructed: the one-world :class:`~newton.ModelBuilder` the example finalized,
and the solver and collision pipeline it created, with their arguments. The
N-world models stay alive between calls and across rebuilds that keep the
scene's structure; live model values are copied into them before every call.

:class:`ObjectSnapshot` saves Python objects in place (their Warp and NumPy
arrays, attributes, and the contents of their lists and dictionaries), so a
restore returns a controller with its warm start without replacing the
buffers its CUDA graphs use.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import hashlib
import sys
import threading
import time
import types
import weakref
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

import numpy as np
import warp as wp

from ..sim import CollisionPipeline, Contacts, Control, Model, ModelBuilder, State, StateFlags, eval_fk
from ..sim.enums import ModelFlags
from ..solvers.solver import SolverBase
from ..utils.batch_rollout import (
    BatchRollout,
    _build_lock,
    _BuildSource,
    _is_value_attribute,
    _model_arrays,
    _split_key,
)
from ..utils.world_view import Frequency, WorldView, _attribute_owner, _row_words

MAX_WORLDS = 256
"""Worlds per batch when ``worlds`` is not given; more cases run in several batches."""

_CACHE_SIZE = 2
"""N-world models kept per session."""

_SCALE_PIPELINE_ARGUMENTS = ("rigid_contact_max", "soft_contact_max", "shape_pairs_max")
"""CollisionPipeline arguments that count the whole model, multiplied by the world count of a batch."""

_FLOAT_KINDS = (wp.float16, wp.float32, wp.float64)

_MISSING = object()


# ----------------------------------------------------------------------------------------------------------------------
# Recording the scene while an example is constructed


def _subclasses(base: type) -> list[type]:
    found, stack = [base], [base]
    while stack:
        for cls in stack.pop().__subclasses__():
            if cls not in found:
                found.append(cls)
                stack.append(cls)
    return found


def _replayable(value: Any, depth: int = 0) -> bool:
    """Whether a constructor argument can be passed again for another model (plain data, not scene objects)."""
    if depth > 8:
        return False
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, Enum, np.generic)):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_replayable(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(_replayable(key, depth + 1) and _replayable(item, depth + 1) for key, item in value.items())
    if isinstance(value, (wp.array, Model, State, Control, Contacts, ModelBuilder, SolverBase, CollisionPipeline)):
        return False
    if isinstance(value, np.ndarray):
        return value.size <= 64
    # Classes, functions, and option objects (e.g. a solver's config dataclass) are shared as they are.
    return isinstance(value, (type, types.FunctionType, types.BuiltinFunctionType, functools.partial)) or (
        hasattr(value, "__dict__")
        and all(_replayable(item, depth + 1) for item in vars(value).values() if not callable(item))
    )


class _Call:
    """A recorded constructor call ``cls(model, *args, **kwargs)`` that can be repeated for another model."""

    def __init__(self, cls: type, args: tuple, kwargs: dict):
        self.cls = cls
        self.keyword = not args
        self.args = tuple(args[1:])
        self.kwargs = {key: value for key, value in kwargs.items() if not (self.keyword and key == "model")}
        unplain = next(
            (
                f"{name}={type(value).__name__}"
                for name, value in [*enumerate(self.args, 1), *self.kwargs.items()]
                if not _replayable(value)
            ),
            None,
        )
        self.problem: str | None = (
            f"it was constructed with {unplain}, which belongs to the scene's own model" if unplain else None
        )
        """Why the call cannot be repeated for another model, or ``None``."""

    def _text(self) -> str:
        parts = [repr(value) for value in self.args] + [f"{key}={value!r}" for key, value in self.kwargs.items()]
        return f"{self.cls.__name__}({', '.join(['model', *parts])})"

    def label(self) -> str:
        text = self._text()
        return text if len(text) <= 160 else text[:157] + "..."

    def key(self) -> tuple:
        return ("call", self.cls.__module__, self.cls.__qualname__, id(self.cls), self._text())

    def factory(self, scale: tuple[str, ...] = ()) -> Callable[[Model], Any]:
        if self.problem is not None:
            raise ValueError(f"{self.cls.__name__} cannot be created again: {self.problem}")

        def make(model: Model) -> Any:
            kwargs = dict(self.kwargs)
            for name in scale:
                if isinstance(kwargs.get(name), int) and not isinstance(kwargs[name], bool):
                    kwargs[name] = kwargs[name] * model.world_count
            if self.keyword:
                return self.cls(model=model, **kwargs)
            return self.cls(model, *self.args, **kwargs)

        return make


class SceneSource:
    """The one-world scene of a live session, as :class:`newton.utils.BatchRollout` copies it.

    Args:
        builder: Builder that holds one world of the scene.
        model: The live model finalized from ``builder``.
        options: Keyword arguments ``builder.finalize()`` was called with.
        solver: Constructor call of the scene's solver, or a function that creates a solver for a model.
        pipeline: Constructor call of the scene's collision pipeline, a function that creates one for a
            model, or ``None`` for a scene without one.
        dt: Physics time step [s].
        substeps: Physics steps per frame of the session.
        problem: Why the scene cannot be copied, or ``None``.
    """

    def __init__(
        self,
        builder: ModelBuilder | None,
        model: Model | None,
        options: dict | None = None,
        *,
        solver: _Call | Callable | None = None,
        pipeline: _Call | Callable | None = None,
        dt: float | None = None,
        substeps: int = 1,
        problem: str | None = None,
    ):
        self.builder, self.model, self.options = builder, model, dict(options or {})
        self.solver, self.pipeline = solver, pipeline
        self.dt, self.substeps = dt, max(1, int(substeps))
        self.problem = problem
        self.solver_object: weakref.ref | None = None
        """The solver the scene was built with, to detect a replaced solver."""


class SceneCapture:
    """Context manager that records how a scene is built on the calling thread.

    Records the builders that :meth:`~newton.ModelBuilder.finalize` turned
    into models, and the constructor arguments of solvers and collision
    pipelines. :meth:`source` then picks the ones an example uses.
    """

    def __init__(self):
        self.finalized: list[tuple[ModelBuilder, Model, dict]] = []
        self.solvers: list[tuple[Any, _Call]] = []
        self.pipelines: list[tuple[Any, _Call]] = []
        self._patched: list[tuple[type, str, Any]] = []

    def __enter__(self) -> SceneCapture:
        _build_lock.acquire()
        thread = threading.get_ident()
        try:
            original = ModelBuilder.finalize
            finalized = self.finalized

            @functools.wraps(original)
            def finalize(builder, *args, **kwargs):
                model = original(builder, *args, **kwargs)
                if threading.get_ident() == thread:
                    options = dict(kwargs)
                    if args:
                        options["device"] = args[0]
                    finalized.append((builder, model, options))
                return model

            self._patched.append((ModelBuilder, "finalize", original))
            ModelBuilder.finalize = finalize
            for base, records in ((SolverBase, self.solvers), (CollisionPipeline, self.pipelines)):
                active: set[int] = set()
                for cls in _subclasses(base):
                    self._patch(cls, records, thread, active)
                # Classes defined while the scene is built (e.g. a solver module imported on first use).
                self._patched.append((base, "__init_subclass__", vars(base).get("__init_subclass__", _MISSING)))
                base.__init_subclass__ = classmethod(self._subclass_hook(base, records, thread, active))
        except BaseException:
            self._unpatch()
            raise
        return self

    def _subclass_hook(self, base: type, records: list, thread: int, active: set[int]) -> Callable:
        def hook(cls, **kwargs):
            super(base, cls).__init_subclass__(**kwargs)
            self._patch(cls, records, thread, active)

        return hook

    def _patch(self, cls: type, records: list, thread: int, active: set[int]) -> None:
        init = vars(cls).get("__init__")
        if init is None or getattr(init, "_newton_mcp_records", False):
            return
        self._patched.append((cls, "__init__", init))
        cls.__init__ = self._recording(cls, init, records, thread, active)

    @staticmethod
    def _recording(owner: type, init: Callable, records: list, thread: int, active: set[int]) -> Callable:
        @functools.wraps(init)
        def __init__(obj, *args, **kwargs):
            # Only the outermost __init__ of an object sees the caller's arguments.
            if threading.get_ident() != thread or id(obj) in active:
                return init(obj, *args, **kwargs)
            active.add(id(obj))
            try:
                call = _Call(type(obj), args, kwargs)
                defining = next((c for c in type(obj).__mro__ if "__init__" in vars(c)), owner)
                if defining is not owner:
                    # A subclass __init__ that was not recorded passed these arguments on (e.g. via super()).
                    call.problem = f"the arguments of {type(obj).__name__}.__init__ were not recorded"
                records.append((obj, call))
                return init(obj, *args, **kwargs)
            finally:
                active.discard(id(obj))

        __init__._newton_mcp_records = True
        return __init__

    def _unpatch(self) -> None:
        for cls, name, original in reversed(self._patched):
            if original is _MISSING:
                with contextlib.suppress(AttributeError):
                    delattr(cls, name)
            else:
                setattr(cls, name, original)
        self._patched.clear()
        _build_lock.release()

    def __exit__(self, *exc) -> None:
        self._unpatch()

    def source(self, example: Any, *, dt: float | None = None) -> SceneSource:
        """The scene of ``example`` (its ``model``, ``solver`` and ``collision_pipeline``); then forgets the records."""
        try:
            return self._source(example, dt)
        except Exception as error:
            # Recording is a convenience for evaluate() and branch(); it must not fail the example's build.
            return SceneSource(None, None, problem=f"recording the scene failed ({type(error).__name__}: {error})")
        finally:
            self.finalized, self.solvers, self.pipelines = [], [], []

    def _source(self, example: Any, dt: float | None) -> SceneSource:
        model = getattr(example, "model", None)
        frame_dt = getattr(example, "frame_dt", None) or dt
        sim_dt = getattr(example, "sim_dt", None)
        if isinstance(sim_dt, (int, float)) and isinstance(frame_dt, (int, float)) and 0 < sim_dt <= frame_dt:
            step, substeps = float(sim_dt), max(1, round(frame_dt / sim_dt))
        else:
            step, substeps = (float(frame_dt) if isinstance(frame_dt, (int, float)) else None), 1
        found = next(
            ((builder, options) for builder, built, options in reversed(self.finalized) if built is model), None
        )
        problem = None
        if not isinstance(model, Model):
            problem = "the example has no model attribute"
        elif found is None:
            problem = "Example() did not finalize example.model from a ModelBuilder while it was constructed"
        elif model.world_count != 1:
            problem = f"the hosted scene has {model.world_count} worlds; evaluate and branch copy a one-world scene"
        solver = self._match(self.solvers, getattr(example, "solver", None), model)
        pipeline = self._match(self.pipelines, getattr(example, "collision_pipeline", None), model)
        builder, options = found if found is not None else (None, {})
        source = SceneSource(
            builder, model, options, solver=solver, pipeline=pipeline, dt=step, substeps=substeps, problem=problem
        )
        with contextlib.suppress(TypeError):
            source.solver_object = weakref.ref(getattr(example, "solver", None))
        return source

    @staticmethod
    def _match(records: list[tuple[Any, _Call]], obj: Any, model: Any) -> _Call | None:
        for created, call in records:
            if obj is not None and created is obj:
                return call
        for created, call in records:
            if getattr(created, "model", None) is model:
                return call
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Checkpoints of Python objects


def _user_class(value: Any) -> bool:
    """Whether ``value``'s class comes from a hosted script, its helper modules, or cells (not a library)."""
    from .rollback import user_frame  # noqa: PLC0415

    name = getattr(type(value), "__module__", "") or ""
    if name.startswith(("_newton_hosted_", "_newton_mcp_")):
        return True
    path = getattr(sys.modules.get(name), "__file__", None)
    return bool(path) and user_frame(path)


def _snapshot_into(value: Any) -> bool:
    """Whether a snapshot copies ``value``'s contents (not only keeps it bound)."""
    if isinstance(value, (type, types.ModuleType, types.FunctionType, types.MethodType, wp.Graph)):
        return False
    if isinstance(value, (dict, list, State, Control, types.SimpleNamespace)):
        return True
    return hasattr(value, "__dict__") and _user_class(value)


class ObjectSnapshot:
    """In-place copies of Python objects, named by workspace paths such as ``"planner"`` or ``"example.controller"``.

    The snapshot keeps every object it saved, the bindings of their attributes
    (and of list items and dictionary entries), and copies of their Warp and
    NumPy arrays. :meth:`restore` puts the bindings back and writes the copies
    into the same arrays, so buffers that CUDA graphs captured stay valid.
    Objects of the hosted script, its helper modules, and cells are walked
    (four levels deep), as are lists, dictionaries, tuples, and Newton states
    and controls; other objects (models, solvers, graphs) stay bound by
    identity.

    Args:
        namespace: Workspace globals that the paths start from.
        paths: Workspace names or attribute paths of the objects to save.
        budget: Bytes of array copies; arrays beyond it keep their identity only.
        exclude: Objects that are neither copied nor rebound, such as the
            session's own state, control, model, and solver (attributes
            bound to them keep their current binding on restore).
    """

    _DEPTH = 4

    def __init__(
        self,
        namespace: Mapping[str, Any],
        paths: list[str],
        *,
        budget: int = 256 * 1024 * 1024,
        exclude: tuple = (),
    ):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = [str(path) for path in paths]
        self.budget = budget
        self.bytes = 0
        self.uncovered: list[str] = []
        self._roots: list[tuple[str, Any, str, Any]] = []
        self._owners: list[tuple[str, Any, Any, set]] = []
        self._arrays: list[tuple[str, Any, Any]] = []
        self._exclude = {id(value) for value in exclude if value is not None}
        self._seen: set[int] = set(self._exclude)
        for path in self.paths:
            parent, key = self._resolve(namespace, path)
            value = parent[key] if isinstance(parent, dict) else getattr(parent, key)
            self._roots.append((path, parent, key, value))
            self._capture(path, value, 0)

    @staticmethod
    def _resolve(namespace: Mapping[str, Any], path: str) -> tuple[Any, str]:
        parts = path.split(".")
        if not parts or not all(part.isidentifier() for part in parts):
            raise ValueError(
                f"include names workspace variables or attribute paths, e.g. 'example.controller'; got {path!r}"
            )
        if parts[0] not in namespace:
            raise KeyError(f"include: the workspace has no variable {parts[0]!r}")
        if len(parts) == 1:
            return namespace, parts[0]
        parent = namespace[parts[0]]
        for part in parts[1:-1]:
            parent = getattr(parent, part)
        if not hasattr(parent, parts[-1]):
            raise AttributeError(f"include: {'.'.join(parts[:-1])} has no attribute {parts[-1]!r}")
        return parent, parts[-1]

    def _copy(self, label: str, owner: Any, value: Any) -> None:
        size = value.capacity if isinstance(value, wp.array) else value.nbytes
        if self.bytes + size > self.budget:
            self.uncovered.append(f"{label} ({size / 2**20:.0f} MiB)")
            return
        self.bytes += size
        if isinstance(value, wp.array):
            copy = wp.clone(value, requires_grad=False) if value.ptr is not None and value.size else None
        else:
            copy = value.copy()
        if copy is not None:
            self._arrays.append((label, value, copy))

    def _capture(self, label: str, value: Any, depth: int) -> None:
        if isinstance(value, (wp.array, np.ndarray)):
            if id(value) not in self._seen:
                self._seen.add(id(value))
                self._copy(label, None, value)
            return
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                self._capture(f"{label}[{index}]", item, depth + 1)
            return
        if depth > self._DEPTH or id(value) in self._seen or not _snapshot_into(value):
            return
        self._seen.add(id(value))
        if isinstance(value, list):
            self._owners.append((label, value, list(value), set()))
            items = [(f"{label}[{index}]", item) for index, item in enumerate(value)]
        else:
            mapping = value if isinstance(value, dict) else vars(value)
            # Bindings to excluded objects (e.g. swapped state buffers) are left as they are.
            skip = {name for name, item in mapping.items() if id(item) in self._exclude}
            saved = {name: item for name, item in mapping.items() if name not in skip}
            self._owners.append((label, value, saved, skip))
            if isinstance(value, dict):
                items = [(f"{label}[{key!r}]", item) for key, item in saved.items()]
            else:
                items = [(f"{label}.{name}", item) for name, item in saved.items()]
        for child, item in items:
            self._capture(child, item, depth + 1)

    def restore(self) -> list[str]:
        """Return the saved objects to their saved bindings and array contents.

        Returns:
            Paths of the bindings and arrays that differed and were restored.
        """
        changed: list[str] = []
        for path, parent, key, value in self._roots:
            current = parent.get(key, _MISSING) if isinstance(parent, dict) else getattr(parent, key, _MISSING)
            if current is not value:
                if isinstance(parent, dict):
                    parent[key] = value
                else:
                    setattr(parent, key, value)
                changed.append(path)
        for label, owner, saved, skip in self._owners:
            if isinstance(owner, list):
                if len(owner) != len(saved) or any(a is not b for a, b in zip(owner, saved, strict=False)):
                    owner[:] = saved
                    changed.append(label)
                continue
            current = owner if isinstance(owner, dict) else vars(owner)
            for name in [name for name in current if name not in saved and name not in skip]:
                if isinstance(owner, dict):
                    del owner[name]
                else:
                    delattr(owner, name)
                changed.append(f"{label}.{name}")
            for name, item in saved.items():
                present = current.get(name, _MISSING)
                if present is item or (type(present) is type(item) and _scalar(item) and present == item):
                    continue
                if isinstance(owner, dict):
                    owner[name] = item
                else:
                    setattr(owner, name, item)
                changed.append(f"{label}.{name}" if not isinstance(owner, dict) else f"{label}[{name!r}]")
        for label, original, copy in self._arrays:
            if isinstance(original, wp.array):
                small = original.capacity <= 1 << 22
                if not small or not np.array_equal(original.numpy(), copy.numpy()):
                    original.assign(copy)
                    changed.append(label)
            elif original.shape == copy.shape and original.dtype == copy.dtype:
                if not np.array_equal(original, copy, equal_nan=original.dtype.kind in "fc"):
                    original[...] = copy
                    changed.append(label)
        return list(dict.fromkeys(changed))

    def summary(self) -> dict:
        """Saved paths, the number and size of copied arrays, and arrays beyond the budget."""
        result = {"saved": self.paths, "arrays": len(self._arrays), "MiB": round(self.bytes / 2**20, 3)}
        if self.uncovered:
            result["not_copied"] = self.uncovered[:8]
        return result


def session_objects(session) -> tuple:
    """The session's own scene objects, which its snapshots and restores manage (not object snapshots)."""
    names = ("model", "solver", "state", "state_next", "control", "contacts", "collision_pipeline", "viewer")
    return tuple(getattr(session, name, None) for name in names)


def _scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, complex, str, bytes, type(None), Enum, np.generic))


# ----------------------------------------------------------------------------------------------------------------------
# Warm N-world copies of the live scene


def _structure_key(model: Model) -> str:
    """Hash of what a value copy cannot change: counts, labels, non-float arrays, shape geometry, model scalars."""
    digest = hashlib.sha1()
    for name, array in sorted(_model_arrays(model).items()):
        if name.startswith("bvh_"):
            continue  # acceleration data; finalize() fills it in a nondeterministic order
        digest.update(f"{name}:{array.dtype.__name__}:{array.shape}".encode())
        if array.dtype == wp.uint64 or not array.size:
            continue
        if getattr(array.dtype, "_wp_scalar_type_", array.dtype) not in _FLOAT_KINDS:
            digest.update(np.ascontiguousarray(array.numpy()).tobytes())
    for name, value in sorted(vars(model).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, (bool, int, float, str)):
            digest.update(f"{name}={value!r}".encode())
        elif isinstance(value, list) and name.endswith("_label"):
            digest.update(f"{name}={value!r}".encode())
    for source in getattr(model, "shape_source", None) or ():
        if source is None:
            digest.update(b"-")
            continue
        digest.update(type(source).__name__.encode())
        for field in ("vertices", "indices", "data", "scale", "maxhullvert", "is_solid"):
            value = getattr(source, field, None)
            if value is not None and not callable(value):
                digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    return digest.hexdigest()


def _callable_key(fn: Any) -> tuple:
    """Identity of a solver or pipeline factory that survives redefining the same function in a later cell."""
    if isinstance(fn, _Call):
        return fn.key()
    if isinstance(fn, type):
        return ("class", fn.__module__, fn.__qualname__, id(fn))
    if isinstance(fn, functools.partial):
        return ("partial", _callable_key(fn.func), repr(fn.args), repr(sorted(fn.keywords.items())))
    code = getattr(fn, "__code__", None)
    if code is None:
        return ("object", id(fn))
    cells = tuple(repr(cell.cell_contents) for cell in (getattr(fn, "__closure__", None) or ()))
    return ("code", code.co_code, repr(code.co_consts), repr(code.co_names), cells, repr(fn.__defaults__))


def _label(fn: Any) -> str:
    if isinstance(fn, _Call):
        return fn.label()
    return getattr(fn, "__qualname__", None) or type(fn).__name__


def _control_arrays(control: Control) -> dict[str, wp.array]:
    return {name: array for name, array in WorldView._state_arrays(control).items() if array.size}


def _copy_control(rollout: BatchRollout, control: Control, model: Model) -> None:
    """Copy the one-world ``control`` of ``model`` into every world of the rollout's control."""
    view = rollout.view
    source_view = view._source_view(model)
    targets = _control_arrays(rollout.control)
    for name, array in _control_arrays(control).items():
        destination = targets.get(name)
        if destination is None:
            continue
        try:
            frequency = view._resolve(name)[1]
            dst_rows = view._rows(name, frequency, None, None)
            src_rows = source_view._rows(name, frequency, None, [0])
        except (KeyError, ValueError):
            continue
        if src_rows.shape[1] != dst_rows.shape[1]:
            continue
        view._copy_rows(
            array,
            view._device_index(np.tile(src_rows[0], dst_rows.shape[0])),
            destination,
            view._device_index(dst_rows),
        )


@wp.kernel(enable_backward=False)
def _differs_u32(
    a: wp.array2d[wp.uint32],
    a_rows: wp.array[wp.int32],
    b: wp.array2d[wp.uint32],
    b_rows: wp.array[wp.int32],
    slot: int,
    flags: wp.array[wp.int32],
):
    i, j = wp.tid()
    if a[a_rows[i], j] != b[b_rows[i], j]:
        flags[slot] = 1


@wp.kernel(enable_backward=False)
def _differs_u8(
    a: wp.array2d[wp.uint8],
    a_rows: wp.array[wp.int32],
    b: wp.array2d[wp.uint8],
    b_rows: wp.array[wp.int32],
    slot: int,
    flags: wp.array[wp.int32],
):
    i, j = wp.tid()
    if a[a_rows[i], j] != b[b_rows[i], j]:
        flags[slot] = 1


class _ValueSync:
    """Copies the live model's values into every world of a warm rollout.

    The comparison of the live values with world 0 of the rollout is planned
    once per pair of models and runs on the device (one flag per attribute,
    one host read), so a call whose values did not change costs a few kernel
    launches.
    """

    def __init__(self, rollout: BatchRollout, live: Model):
        self.rollout, self.live = rollout, live
        self.target_view, self.live_view = WorldView(rollout.model), WorldView(live)
        self.entries: list[tuple] = []
        self.problem: str | None = None
        target_arrays = _model_arrays(rollout.model)
        for name, array in _model_arrays(live).items():
            destination = target_arrays.get(name)
            if destination is None or not array.size or not destination.size or destination.dtype != array.dtype:
                continue
            try:
                frequency = Frequency.WORLD if name == "gravity" else live.get_attribute_frequency(name)
            except (KeyError, AttributeError):
                continue
            if frequency == Frequency.ONCE:
                if destination.shape == array.shape:
                    rows = np.arange(array.shape[0])
                    self._add(name, frequency, array, rows, destination, rows)
                continue
            if not _is_value_attribute(self.target_view, name, frequency, destination):
                continue
            try:
                live_rows = self.live_view._rows(name, frequency, None, [0])[0]
                target_rows = self.target_view._rows(name, frequency, None, [0])[0]
            except (KeyError, ValueError):
                continue
            if live_rows.shape != target_rows.shape:
                self.problem = f"{name} has {live_rows.size} rows in the live scene, {target_rows.size} per world"
                return
            self._add(name, frequency, array, live_rows, destination, target_rows)
        self.flags = wp.zeros(max(len(self.entries), 1), dtype=wp.int32, device=rollout.device)

    def _add(self, name, frequency, live, live_rows, target, target_rows) -> None:
        live_words, target_words = _row_words(live), _row_words(target)
        index = self.target_view._device_index
        self.entries.append(
            (name, frequency, live, live_words, index(live_rows), live_rows, target, target_words, index(target_rows))
        )

    def changed(self) -> list[tuple]:
        """Entries whose live values differ from world 0 of the rollout."""
        if not self.entries:
            return []
        self.flags.zero_()
        for slot, (_, _, _, live_words, live_index, _, _, target_words, target_index) in enumerate(self.entries):
            wp.launch(
                _differs_u32 if live_words.dtype == wp.uint32 else _differs_u8,
                dim=(live_index.shape[0], live_words.shape[1]),
                inputs=[live_words, live_index, target_words, target_index, slot],
                outputs=[self.flags],
                device=self.rollout.device,
            )
        flags = self.flags.numpy()
        return [entry for entry, flag in zip(self.entries, flags, strict=False) if flag]

    def apply(self) -> tuple[list[str], str | None]:
        """Copy changed values into every world and notify the solver.

        Returns:
            The copied attributes, or the reason the rollout must be built
            again (a changed value that its solver reads only when it is
            constructed or shares across worlds).
        """
        if self.problem is not None:
            return [], self.problem
        rollout, target = self.rollout, self.rollout.model
        unread = getattr(rollout.solver, "_UNREAD_ATTRIBUTES", {})
        writes = []
        for name, frequency, live, _, _, live_rows, _, _, _ in self.changed():
            if frequency == Frequency.ONCE:
                return [], f"{name} differs (one value for all worlds of a model)"
            notify = True
            try:
                rollout.solver.check_world_values(name, self.target_view.get_indices(name).ravel())
            except ValueError as error:
                if name not in unread:
                    return [], f"{name} differs ({error})"
                notify = False
            writes.append((name, live.numpy()[live_rows], notify))
        flags = 0
        for name, values, notify in writes:
            self.target_view.set_attribute(name, target, np.broadcast_to(values, (target.world_count, *values.shape)))
            if notify:
                flags |= int(ModelFlags.from_attributes(name))
        if flags:
            rollout.solver.notify_model_changed(flags)
        return [name for name, _, _ in writes], None


def _user(fn: Callable | None) -> Callable | None:
    if fn is None:
        return None

    @functools.wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException as error:
            with contextlib.suppress(Exception):
                error.__newton_user_error__ = True
            raise

    return call


def _frames(frames: int | None, seconds: float | None, frame_dt: float) -> int:
    if frames is None:
        if seconds is None:
            raise ValueError("Give frames or seconds")
        frames = max(1, round(float(seconds) / frame_dt))
    if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or not 1 <= frames <= 10_000_000:
        raise ValueError(f"frames must be a positive integer, got {frames!r}")
    return int(frames)


def _seconds(value: float) -> str:
    return f"{value:.3g} s"


class LiveEvaluation(BatchRollout.Evaluation):
    """A :class:`newton.utils.BatchRollout.Evaluation` of the live scene, with facts about the run.

    ``format()`` (and the response of the cell that returns it) starts with
    the facts: the start state, the frames, the N-world model (reused or
    built and why), copied live values, and the wall time with CUDA graph
    warm-up.
    """

    def __init__(self, evaluation: BatchRollout.Evaluation, facts: dict[str, Any]):
        super().__init__(
            evaluation.rows,
            evaluation.candidates,
            evaluation.scenarios,
            evaluation.batches,
            evaluation.metrics,
            evaluation.worst,
        )
        self.facts: dict[str, Any] = facts
        """Start, frames, model reuse, copied live values, and timing of the call."""

    def format(self, *, rows: bool | None = None, digits: int = 4) -> str:
        return _facts_text("evaluate", self.facts) + "\n" + super().format(rows=rows, digits=digits)


class Branches:
    """Records of the variants of :meth:`SimulationSession.branch <newton.mcp.SimulationSession.branch>`.

    ``format()`` (and the response of the cell that returns it) shows the
    facts of the run and one row per variant with its metrics, or the last
    recorded values of each probe.
    """

    def __init__(
        self,
        records: dict[str, np.ndarray],
        t: np.ndarray,
        metrics: dict[str, np.ndarray],
        facts: dict[str, Any],
        batches: list[dict] | None = None,
    ):
        self.records: dict[str, np.ndarray] = records
        """Recorded probes by name, shape ``[T, n, ...]`` (row 0 is the start)."""
        self.t: np.ndarray = t
        """Session time of each record row [s], shape ``[T]``."""
        self.metrics: dict[str, np.ndarray] = metrics
        """Values that ``score`` returned, one per variant."""
        self.facts: dict[str, Any] = facts
        """Start, frames, execution, and timing of the call."""
        self.batches: list[dict] = batches or []
        """Batches of worlds the variants ran in (see :attr:`newton.utils.BatchRollout.Evaluation.batches`)."""

    def __len__(self) -> int:
        return int(self.facts.get("cases", 0))

    def format(self, *, digits: int = 4, limit: int = 40) -> str:
        """The facts and one row per variant (up to ``limit`` rows) as text."""
        lines = [_facts_text("branch", self.facts)]
        for index, batch in enumerate(self.batches):
            if batch.get("reason"):
                lines.append(f"batch {index}: {batch['worlds']} variant(s) in a separate model: {batch['reason']}")
        count = len(self)
        columns = list(self.metrics) or list(self.records)
        header = ["variant", *(columns if self.metrics else [f"{name}[-1]" for name in columns])]
        table = []
        for i in range(min(count, limit)):
            row = [str(i)]
            for name in columns:
                value = self.metrics[name][i] if self.metrics else self.records[name][-1, i]
                row.append(_value_text(value, digits))
            table.append(row)
        from ..utils.batch_rollout import _table  # noqa: PLC0415

        lines.append(_table(header, table))
        if count > limit:
            lines.append(f"... {count - limit} more variant(s)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"Branches({len(self)} variants, records {list(self.records)})"


def _value_text(value: Any, digits: int) -> str:
    array = np.asarray(value)
    if array.dtype == object:
        return str(value)
    if array.size <= 4:
        flat = array.ravel()
        parts = [f"{float(v):.{digits}g}" if array.dtype.kind in "fc" else str(v) for v in flat]
        return parts[0] if array.ndim == 0 else "[" + ", ".join(parts) + "]"
    return f"<{'x'.join(map(str, array.shape))}>"


def _facts_text(kind: str, facts: dict[str, Any]) -> str:
    cases = facts["cases"]
    noun = "case(s)" if kind == "evaluate" else "variant(s)"
    frame = f"{facts['frame_dt']:.6g} s" + (
        f" ({facts['dt']:.6g} s x {facts['substeps']})" if facts.get("substeps", 1) > 1 else ""
    )
    lines = [f"{kind}: {cases} {noun} from {facts['start']}; {facts['frames']} frames of {frame} each"]
    if facts.get("sequential"):
        restored = [f"{', '.join(facts['objects'])} (from the checkpoint)"] if facts.get("objects") else []
        if facts.get("others"):
            restored.append(f"{', '.join(facts['others'])} (as at the call)")
        lines.append(
            "ran one after another through example.step(); Python objects restored before each variant: "
            + ("; ".join(restored) or "none")
            + "; the session and these objects were returned to their state before the call"
        )
    else:
        model = facts["model"]
        lines.append(
            f"model: {facts['world_count']} world(s), {model}; solver {facts['solver']}; collision {facts['collision']}"
        )
        if facts.get("copied"):
            names = facts["copied"]
            shown = ", ".join(names[:8]) + (f", +{len(names) - 8} more" if len(names) > 8 else "")
            lines.append(f"live model values copied into the worlds: {shown}")
        if facts.get("siblings"):
            lines.append(
                f"separate models built in this call: {facts['siblings']} ({_seconds(facts['sibling_seconds'])})"
            )
    timing = f"wall time {_seconds(facts['seconds'])}"
    if facts.get("captures"):
        timing += f", including {_seconds(facts['warmup_seconds'])} for {facts['captures']} new CUDA graph(s) (first frame, kernel loading, capture)"
    elif facts.get("cuda") and not facts.get("sequential"):
        timing += " (cached CUDA graphs)"
    lines.append(timing)
    return "\n".join(lines)


class _LiveWorld:
    """Edits of the live scene for one variant of a sequential branch (the interface of
    :class:`newton.utils.BatchRollout.WorldSetup`); the branch undoes them after the variant."""

    def __init__(self, session):
        self._session = session
        self._view = WorldView(session.model)

    @property
    def model(self) -> Model:
        return self._session.model

    def _read(self, name: str, source: Any, labels: Any) -> np.ndarray:
        return self._view.get_attribute(name, source, labels=labels)[0]

    def get_model(self, name: str, labels: Any = None) -> np.ndarray:
        return self._read(name, self._session.model, labels)

    def get_state(self, name: str, labels: Any = None) -> np.ndarray:
        return self._read(name, self._session.state, labels)

    def get_control(self, name: str, labels: Any = None) -> np.ndarray:
        return self._read(name, self._session.control, labels)

    def set_model(self, name: str, value: Any, labels: Any = None) -> None:
        session = self._session
        self._view.set_attribute(name, session.model, np.asarray(value)[None], labels=labels, solver=session.solver)

    def set_state(self, name: str, value: Any, labels: Any = None) -> None:
        session = self._session
        self._view.set_attribute(name, session.state, np.asarray(value)[None], labels=labels)
        if name.replace(".", ":", 1) in ("joint_q", "joint_qd") and session.model.joint_count:
            eval_fk(session.model, session.state.joint_q, session.state.joint_qd, session.state)
        session.solver.reset(session.state, flags=StateFlags.NONE)
        session.state_next.assign(session.state)

    def set_control(self, name: str, value: Any, labels: Any = None) -> None:
        session = self._session
        self._view.set_attribute(name, session.control, np.asarray(value)[None], labels=labels)

    def set_schedule(self, name: str, values: Any, labels: Any = None) -> None:
        raise ValueError(
            "In a sequential branch example.step() sets the controls; schedules apply to branches run as worlds "
            "(sequential=False)"
        )


class LiveBatches:
    """``evaluate`` and ``branch`` of one session, with its warm N-world models."""

    def __init__(self, session):
        self._session = weakref.ref(session)
        self._cache: collections.OrderedDict[tuple, BatchRollout] = collections.OrderedDict()
        self._syncs: dict[int, _ValueSync] = {}

    def close(self) -> None:
        self._cache.clear()
        self._syncs.clear()

    # -- models ----------------------------------------------------------------------------------------------------------

    def _source(self, solver: Any, pipeline: Any) -> tuple[SceneSource, Any, Any]:
        session = self._session()
        source = getattr(session, "scene_source", None)
        if source is None:
            raise RuntimeError(
                "evaluate and branch copy the scene that ExampleHost recorded while Example() was constructed; "
                "this session has no recorded scene (session.scene_source)"
            )
        if source.problem is not None:
            raise RuntimeError(f"The hosted scene cannot be copied into worlds: {source.problem}")
        if source.model is not session.model:
            raise RuntimeError(
                "session.model is not the model the hosted script built (a cell replaced it); rebuild the scene"
            )
        if solver is None:
            solver = source.solver
            if solver is None:
                raise RuntimeError("No solver was constructed for the hosted scene's model; pass solver=fn(model)")
            if isinstance(solver, _Call) and solver.problem is not None:
                raise RuntimeError(
                    f"The scene's solver cannot be created again for the worlds ({solver.problem}); pass "
                    "solver=fn(model)"
                )
            built = source.solver_object() if source.solver_object is not None else session.solver
            if session.solver is not built:
                raise RuntimeError(
                    "session.solver was replaced after the scene was built; pass solver=fn(model) for the worlds"
                )
        if pipeline is None:
            pipeline = source.pipeline
            if isinstance(pipeline, _Call) and pipeline.problem is not None:
                raise RuntimeError(
                    f"The scene's collision pipeline cannot be created again for the worlds ({pipeline.problem}); "
                    "pass pipeline=fn(model) (or pipeline=False for none)"
                )
        elif pipeline is False:
            pipeline = None
        return source, solver, pipeline

    def _rollout(self, needed: int, worlds: int | None, solver, pipeline, dt, substeps) -> tuple[BatchRollout, dict]:
        session = self._session()
        source, solver, pipeline = self._source(solver, pipeline)
        dt = float(dt) if dt is not None else source.dt
        if dt is None:
            raise RuntimeError("The hosted example has no frame_dt; pass dt=")
        substeps = int(substeps) if substeps is not None else (source.substeps if dt == source.dt else 1)
        if worlds is not None:
            if isinstance(worlds, bool) or not isinstance(worlds, (int, np.integer)) or worlds < 1:
                raise ValueError(f"worlds must be a positive integer, got {worlds!r}")
            world_count = int(worlds)
        else:
            world_count = max(1, min(needed, MAX_WORLDS))
        live = session.model
        key = (_structure_key(live), _callable_key(solver), _callable_key(pipeline) if pipeline else None, dt, substeps)
        key += (str(live.device),)
        facts = {
            "dt": dt,
            "substeps": substeps,
            "frame_dt": dt * substeps,
            "solver": _label(solver),
            "collision": _label(pipeline) if pipeline else "none",
            "cuda": live.device.is_cuda,
        }
        candidate = None
        for (cached_key, cached_worlds), rollout in reversed(self._cache.items()):
            if cached_key != key:
                continue
            # Spare worlds run unscored; on a GPU they cost little, on a CPU as much as used ones.
            spare = max(world_count, 8) if worlds is None and live.device.is_cuda else 0
            if world_count <= cached_worlds <= world_count + spare:
                candidate = ((cached_key, cached_worlds), rollout)
                break
        if candidate is not None:
            cache_key, rollout = candidate
            sync = self._syncs.get(id(rollout))
            if sync is None or sync.rollout is not rollout or sync.live is not live:
                sync = self._syncs[id(rollout)] = _ValueSync(rollout, live)
            with session.watch.muted():
                copied, reason = sync.apply()
            if reason is None:
                self._cache.move_to_end(cache_key)
                facts.update(model="reused", world_count=rollout.world_count, copied=copied)
                return rollout, facts
            self._evict(rollout)
        else:
            reason = self._why_new(key)
        started = time.perf_counter()
        solver_factory = solver.factory() if isinstance(solver, _Call) else solver
        pipeline_factory = pipeline.factory(_SCALE_PIPELINE_ARGUMENTS) if isinstance(pipeline, _Call) else pipeline
        with session.watch.muted():
            rollout = BatchRollout(
                _BuildSource(source.builder, live, source.options),
                world_count,
                solver=solver_factory,
                pipeline=pipeline_factory,
                dt=dt,
                substeps=substeps,
                device=live.device,
            )
        self._check_layout(rollout, live)
        seconds = time.perf_counter() - started
        self._cache[(key, world_count)] = rollout
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        kept = {id(value) for value in self._cache.values()}
        self._syncs = {key: sync for key, sync in self._syncs.items() if key in kept}
        facts.update(
            model=f"built in {_seconds(seconds)} ({reason})", world_count=world_count, copied=rollout.copied_attributes
        )
        return rollout, facts

    _KEY_PARTS = (
        "the scene's structure or integer values",
        "the solver",
        "the collision pipeline",
        "dt",
        "substeps",
        "the device",
    )

    def _why_new(self, key: tuple) -> str:
        """Why no kept model serves ``key``."""
        if not self._cache:
            return "first use"
        same = [worlds for cached, worlds in self._cache if cached == key]
        if same:
            return f"the kept model has {same[-1]} worlds"
        (latest, _), _ = next(reversed(self._cache.items()))
        changed = [name for name, a, b in zip(self._KEY_PARTS, latest, key, strict=True) if a != b]
        return f"{' and '.join(changed)} changed since the kept model was built"

    @staticmethod
    def _check_layout(rollout: BatchRollout, live: Model) -> None:
        for name in (
            "body_count",
            "shape_count",
            "joint_count",
            "joint_dof_count",
            "joint_coord_count",
            "particle_count",
        ):
            batch, one = int(getattr(rollout.model, name)), int(getattr(live, name))
            if batch != one * rollout.world_count:
                raise RuntimeError(
                    f"The recorded builder no longer matches the live model ({name}: {batch} in "
                    f"{rollout.world_count} worlds, {one} live); was the builder changed after finalize()?"
                )

    def _evict(self, rollout: BatchRollout) -> None:
        for key, value in list(self._cache.items()):
            if value is rollout:
                del self._cache[key]
        self._syncs.pop(id(rollout), None)

    def _start(self, start: Any) -> tuple[State, Control, str, float, dict | None]:
        """Start state and control (of the live model), a description, the start time, and the snapshot."""
        session = self._session()
        if start is None or start == "current":
            return session.state, session.control, f"the live state at t={session.time:.6g} s", session.time, None
        if start is True or start == "initial":
            snapshot, label = session._initial, "the initial state"
        elif isinstance(start, str):
            snapshot = session._checkpoints.get(start)
            if snapshot is None:
                names = ", ".join(map(repr, session._checkpoints)) or "none"
                raise KeyError(f"No checkpoint {start!r}; saved checkpoints: {names}")
            label = f"checkpoint {start!r} (t={snapshot['time']:.6g} s)"
        else:
            raise TypeError("start must be None (the live state), 'initial', or a checkpoint name")
        state, control = session.model.state(), session.model.control()
        for root, target in (("state", state), ("control", control)):
            for field, data in snapshot[root].items():
                owner = target
                parts = field.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                array = getattr(owner, parts[-1], None)
                # Snapshots hold NumPy copies, whose shape includes the dtype's (7 per transform).
                if isinstance(array, wp.array) and (*array.shape, *getattr(array.dtype, "_shape_", ())) == data.shape:
                    array.assign(data)
        return state, control, label, float(snapshot["time"]), snapshot

    @staticmethod
    def _stats(rollout: BatchRollout) -> dict[int, tuple[BatchRollout, int, float]]:
        rollouts = [rollout, *rollout._siblings.values()]
        return {id(r): (r, r._stats["captures"], r._stats["warmup_seconds"]) for r in rollouts}

    @staticmethod
    def _timing(rollout: BatchRollout, before: dict, facts: dict, started: float) -> None:
        captures, warmup, siblings, sibling_seconds = 0, 0.0, 0, 0.0
        for r in [rollout, *rollout._siblings.values()]:
            if id(r) in before and before[id(r)][0] is r:
                captures += r._stats["captures"] - before[id(r)][1]
                warmup += r._stats["warmup_seconds"] - before[id(r)][2]
            else:
                captures += r._stats["captures"]
                warmup += r._stats["warmup_seconds"]
                siblings += 1
                sibling_seconds += r._stats["build_seconds"]
        facts.update(captures=captures, warmup_seconds=warmup, seconds=time.perf_counter() - started)
        if siblings:
            facts.update(siblings=siblings, sibling_seconds=sibling_seconds)

    def _run(self, rollout: BatchRollout, call: Callable[[], Any]) -> Any:
        try:
            # Setups notify the batch's solver; those calls say nothing about the live solver's edits.
            with self._session().watch.muted():
                return call()
        except BaseException as error:
            if not getattr(error, "__newton_user_error__", False):
                # The batch may hold a failed capture or a partial edit; the next call builds a new one.
                self._evict(rollout)
            raise

    # -- evaluate and branch ---------------------------------------------------------------------------------------------

    def evaluate(
        self,
        candidates: Any,
        scenarios: Any = None,
        *,
        frames: int | None = None,
        seconds: float | None = None,
        score: Callable,
        setup: Callable | None = None,
        control: Any = None,
        record: Mapping[str, Any] | None = None,
        every: int = 1,
        passed: Callable | None = None,
        worst: Mapping[str, str] | None = None,
        build: Callable | None = None,
        start: Any = None,
        worlds: int | None = None,
        solver: Callable | None = None,
        pipeline: Any = None,
        dt: float | None = None,
        substeps: int | None = None,
    ) -> LiveEvaluation:
        started = time.perf_counter()
        count = len(candidates) * (1 if scenarios is None else len(scenarios))
        rollout, facts = self._rollout(count, worlds, solver, pipeline, dt, substeps)
        frames = _frames(frames, seconds, rollout.frame_dt)
        state, control_values, label, _, _ = self._start(start)
        session = self._session()
        _copy_control(rollout, control_values, session.model)
        before = self._stats(rollout)
        evaluation = self._run(
            rollout,
            lambda: rollout.evaluate(
                candidates,
                scenarios,
                frames=frames,
                score=_user(score),
                setup=_user(setup),
                control=control,
                record=record,
                every=every,
                passed=_user(passed),
                worst=worst,
                build=_user(build),
                initial_state=state,
                initial_model=session.model,
                initial_world=0,
            ),
        )
        facts.update(start=label, frames=frames, cases=count)
        self._timing(rollout, before, facts, started)
        return LiveEvaluation(evaluation, facts)

    def branch(
        self,
        n: int,
        setup: Callable | None = None,
        *,
        frames: int | None = None,
        seconds: float | None = None,
        start: Any = None,
        record: Mapping[str, Any] | None = None,
        control: Any = None,
        every: int = 1,
        score: Callable | None = None,
        sequential: bool = False,
        worlds: int | None = None,
        solver: Callable | None = None,
        pipeline: Any = None,
        dt: float | None = None,
        substeps: int | None = None,
    ) -> Branches:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        n = int(n)
        if sequential:
            if control is not None:
                raise ValueError(
                    "In a sequential branch example.step() sets the controls; control= applies to branches run as "
                    "worlds (sequential=False)"
                )
            if any(value is not None for value in (worlds, solver, pipeline, dt, substeps)):
                raise ValueError("worlds, solver, pipeline, dt and substeps apply to branches run as worlds")
            return self._sequential(n, setup, frames, seconds, start, record or {}, every, score)
        started = time.perf_counter()
        rollout, facts = self._rollout(n, worlds, solver, pipeline, dt, substeps)
        frames = _frames(frames, seconds, rollout.frame_dt)
        state, control_values, label, start_time, _ = self._start(start)
        session = self._session()
        _copy_control(rollout, control_values, session.model)
        captured: dict[int, dict[str, np.ndarray]] = {}

        def keep(records, cases):
            for index, (variant, _) in enumerate(cases):
                captured[variant] = {name: np.array(array[:, index]) for name, array in records.items()}
            return {}

        variant_setup = _user(setup)
        before = self._stats(rollout)
        evaluation = self._run(
            rollout,
            lambda: rollout.evaluate(
                list(range(n)),
                None,
                frames=frames,
                score=keep,
                setup=(lambda world, variant, _: variant_setup(world, variant)) if setup is not None else None,
                control=control,
                record=record,
                every=every,
                initial_state=state,
                initial_model=session.model,
                initial_world=0,
            ),
        )
        records = {name: np.stack([captured[i][name] for i in range(n)], axis=1) for name in (captured[0] if n else {})}
        times = start_time + rollout.record_time
        facts.update(start=label, frames=frames, cases=n)
        self._timing(rollout, before, facts, started)
        metrics = self._score(score, records, n)
        return Branches(records, times, metrics, facts, evaluation.batches)

    @staticmethod
    def _score(score: Callable | None, records: dict, n: int) -> dict[str, np.ndarray]:
        if score is None:
            return {}
        result = score(records)
        if not isinstance(result, Mapping):
            raise TypeError(f"score must return a mapping of metric names to {n} values, got {type(result)!r}")
        metrics = {}
        for name, values in result.items():
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != n:
                raise ValueError(f"score returned {array.shape} values of '{name}' for {n} variants")
            metrics[name] = array
        return metrics

    def _sequential(self, n, setup, frames, seconds, start, record, every, score) -> Branches:
        from .rollback import UndoPoint  # noqa: PLC0415

        session = self._session()
        started = time.perf_counter()
        frames = _frames(frames, seconds, session.dt)
        if isinstance(every, bool) or not isinstance(every, (int, np.integer)) or every < 1:
            raise ValueError(f"every must be a positive integer, got {every!r}")
        _, _, label, _, snapshot = self._start(start)
        pre = session._snapshot()
        start_snapshot = snapshot if snapshot is not None else pre
        objects = start_snapshot.get("objects")
        scope = session._eval_scope()
        included = list(objects.paths) if objects is not None else []
        # The example's other objects (controllers, reports) start every variant as they are now, and every
        # saved object is put back after the last variant.
        example = scope.get("example")
        others = [
            f"example.{name}"
            for name, value in (vars(example).items() if hasattr(example, "__dict__") else ())
            if not name.startswith("__") and f"example.{name}" not in included and _snapshot_into(value)
        ]
        exclude = session_objects(session)
        current = ObjectSnapshot(scope, others, exclude=exclude)
        included_now = ObjectSnapshot(scope, included, exclude=exclude) if included else None
        probes = self._live_probes(record)
        series: dict[str, list[np.ndarray]] = {name: [] for name in probes}
        times = None
        try:
            for variant in range(n):
                current.restore()
                session._restore(start_snapshot)
                undo = UndoPoint(session)
                try:
                    if setup is not None:
                        setup(_LiveWorld(session), variant)
                    rows, times = self._step_variant(probes, frames, every)
                finally:
                    # Model edits of the setup and of stepping (e.g. gains a controller writes) are undone.
                    undo.rollback()
                    session.watch.reset()
                for name in probes:
                    series[name].append(rows[name])
        finally:
            current.restore()
            if included_now is not None:
                included_now.restore()
            session._restore(pre)
        records = {name: np.stack(values, axis=1) for name, values in series.items()}
        facts = {
            "start": label,
            "frames": frames,
            "cases": n,
            "frame_dt": session.dt,
            "dt": session.dt,
            "substeps": 1,
            "sequential": True,
            "objects": included,
            "others": others,
            "cuda": session.model.device.is_cuda,
        }
        facts["seconds"] = time.perf_counter() - started
        metrics = self._score(score, records, n)
        return Branches(records, times if times is not None else np.zeros(0), metrics, facts)

    def _step_variant(self, probes: dict, frames: int, every: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Step the live session ``frames`` frames, sampling ``probes`` at the start and every ``every`` frames."""
        session = self._session()
        rows: dict[str, list] = {name: [] for name in probes}
        times = []

        def sample():
            times.append(session.time)
            for name, probe in probes.items():
                rows[name].append(probe())

        with session.watch.stepping("branch()"):
            sample()
            for index in range(frames):
                session._advance(1, session.dt, first=index == 0, last=index == frames - 1)
                if (index + 1) % every == 0:
                    sample()
        return {name: np.stack(values) for name, values in rows.items()}, np.asarray(times)

    def _live_probes(self, record: Mapping[str, Any]) -> dict[str, Callable[[], np.ndarray]]:
        from .session import _session_callable  # noqa: PLC0415

        session = self._session()
        view = WorldView(session.model)
        probes = {}
        for name, spec in record.items():
            if callable(spec):
                fn = _session_callable(spec)

                def read(fn=fn):
                    value = fn(session)
                    return value.numpy() if isinstance(value, wp.array) else np.asarray(value)

            else:
                attribute, labels = _split_key(spec)
                attribute = attribute.replace(".", ":", 1)
                owner, leaf = _attribute_owner(session.state, attribute)
                in_state = owner is not None and isinstance(getattr(owner, leaf, None), wp.array)

                def read(attribute=attribute, labels=labels, in_state=in_state):
                    source = session.state if in_state else session.control
                    return view.get_attribute(attribute, source, labels=labels)[0]

            probes[name] = read
        return probes
