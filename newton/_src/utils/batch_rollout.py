# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched rollouts and candidate x scenario evaluations over the worlds of one model."""

from __future__ import annotations

import collections
import functools
import hashlib
import math
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

from ..sim import CollisionPipeline, Control, Model, ModelBuilder, State, StateFlags, eval_fk
from .world_view import Frequency, WorldView, _attribute_owner, _pattern_key, _row_words, _value_shape

# ModelBuilder settings that finalize() reads, copied to the builder of the batch.
_BUILDER_SETTINGS = (
    "balance_inertia",
    "bound_mass",
    "bound_inertia",
    "validate_inertia_detailed",
    "particle_max_velocity",
    "rigid_gap",
    "num_rigid_contacts_per_world",
)

# Name parts of model arrays that hold indices, counts, or pointers rather than per-entity values.
_INDEX_NAME_PARTS = ("_start", "_end", "_world", "_count", "_index", "_indices", "_ptr", "_offset", "_parent", "_child")

_build_lock = threading.RLock()

LabelsLike = str | list[str] | Any | None


# ----------------------------------------------------------------------------------------------------------------------
# Kernels: rows are copied as raw 4-byte (or 1-byte) words, so any dtype works.


@wp.kernel(enable_backward=False)
def _apply_rows_u32(
    values: wp.array3d[wp.uint32],
    frame: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    dst: wp.array2d[wp.uint32],
):
    i, j = wp.tid()
    row = wp.min(frame[0], values.shape[0] - 1)
    dst[rows[i], j] = values[row, i, j]


@wp.kernel(enable_backward=False)
def _apply_rows_u8(
    values: wp.array3d[wp.uint8],
    frame: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    dst: wp.array2d[wp.uint8],
):
    i, j = wp.tid()
    row = wp.min(frame[0], values.shape[0] - 1)
    dst[rows[i], j] = values[row, i, j]


@wp.kernel(enable_backward=False)
def _record_rows_u32(
    src: wp.array2d[wp.uint32],
    rows: wp.array[wp.int32],
    counter: wp.array[wp.int32],
    every: int,
    dst: wp.array3d[wp.uint32],
):
    i, j = wp.tid()
    count = counter[0]
    if count % every != 0:
        return
    row = count / every
    if row >= dst.shape[0]:
        return
    dst[row, i, j] = src[rows[i], j]


@wp.kernel(enable_backward=False)
def _record_rows_u8(
    src: wp.array2d[wp.uint8],
    rows: wp.array[wp.int32],
    counter: wp.array[wp.int32],
    every: int,
    dst: wp.array3d[wp.uint8],
):
    i, j = wp.tid()
    count = counter[0]
    if count % every != 0:
        return
    row = count / every
    if row >= dst.shape[0]:
        return
    dst[row, i, j] = src[rows[i], j]


@wp.kernel(enable_backward=False)
def _advance(counter: wp.array[wp.int32]):
    counter[0] = counter[0] + 1


# ----------------------------------------------------------------------------------------------------------------------
# Building the batch from one world


class _BuildSource:
    """A built world: its builder, the Model the build returned (if any), finalize() options, and shared values."""

    def __init__(self, builder: ModelBuilder, one_world: Model | None, options: dict, presets: tuple = ()):
        self.builder, self.one_world, self.options, self.presets = builder, one_world, options, tuple(presets)


def _call_build(build: Any) -> _BuildSource:
    """The one-world builder of ``build``, the Model it returned (if any), and the finalize() keyword arguments."""
    if isinstance(build, _BuildSource):
        return build
    if isinstance(build, ModelBuilder):
        return _BuildSource(build, None, {})
    if not callable(build):
        raise TypeError(f"build must be a ModelBuilder or a function that returns one world, got {type(build)!r}")
    finalized: list[tuple[ModelBuilder, Model, dict]] = []
    thread = threading.get_ident()
    with _build_lock:
        original = ModelBuilder.finalize

        @functools.wraps(original)
        def finalize(self, *args, **kwargs):
            model = original(self, *args, **kwargs)
            if threading.get_ident() == thread:
                options = dict(kwargs)
                if args:
                    options["device"] = args[0]
                finalized.append((self, model, options))
            return model

        ModelBuilder.finalize = finalize
        try:
            result = build()
        finally:
            ModelBuilder.finalize = original
    if isinstance(result, ModelBuilder):
        return _BuildSource(result, None, {})
    if isinstance(result, Model):
        for builder, model, options in reversed(finalized):
            if model is result:
                if result.world_count != 1:
                    raise ValueError(
                        f"build() returned a Model with {result.world_count} worlds; it must return one world "
                        "(BatchRollout makes the copies)"
                    )
                return _BuildSource(builder, result, options)
        raise ValueError(
            "build() returned a Model that it did not finalize from a ModelBuilder during the call; return the "
            "ModelBuilder instead"
        )
    raise TypeError(f"build() must return a ModelBuilder or a Model, got {type(result)!r}")


def _replicate(builder: ModelBuilder, world_count: int, device: Any, options: dict) -> Model:
    if builder.world_count > 1:
        raise ValueError(f"The build has {builder.world_count} worlds; it must hold one world")
    scene = ModelBuilder(
        up_axis=builder.up_axis, gravity=builder._gravity, sdf_texture_paired_samples=builder.sdf_texture_paired_samples
    )
    for name in _BUILDER_SETTINGS:
        if hasattr(builder, name):
            setattr(scene, name, getattr(builder, name))
    scene._requested_contact_attributes |= set(builder._requested_contact_attributes)
    scene._requested_state_attributes |= set(builder._requested_state_attributes)
    scene.replicate(builder, world_count)
    options = {key: value for key, value in options.items() if key != "device"}
    return scene.finalize(device=device, **options)


def _model_arrays(model: Model) -> dict[str, wp.array]:
    arrays = {}
    for name, value in model.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, wp.array):
            arrays[name] = value
        elif isinstance(value, Model.AttributeNamespace):
            for leaf, array in value.__dict__.items():
                if isinstance(array, wp.array) and not leaf.startswith("_"):
                    arrays[f"{name}:{leaf}"] = array
    return arrays


def _is_value_attribute(view: WorldView, name: str, frequency: Any, array: wp.array) -> bool:
    """Whether ``name`` holds per-entity values (not indices, counts, pointers, or structure)."""
    leaf = name.rsplit(":", 1)[-1]
    if any(part in leaf for part in _INDEX_NAME_PARTS) or leaf.endswith("_label"):
        return False
    try:
        view._check_model_attribute(name, frequency)
    except ValueError:
        return False
    spec = view.model._attribute_spec(name)
    scalar = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
    if spec is None and scalar not in (wp.float16, wp.float32, wp.float64):
        return False
    return array.dtype != wp.uint64


def _copy_world_values(source: Model, target: Model) -> list[str]:
    """Copy the values of ``source``'s single world into every world of ``target`` where they differ.

    Picks up edits that a build function made to its Model after finalize(). Returns the copied attributes.
    """
    source_view, target_view = WorldView(source), WorldView(target)
    target_arrays = _model_arrays(target)
    copied = []
    for name, array in _model_arrays(source).items():
        destination = target_arrays.get(name)
        if destination is None or not array.size or not destination.size or destination.dtype != array.dtype:
            continue
        if name == "gravity":
            frequency = Frequency.WORLD
        else:
            try:
                frequency = source.get_attribute_frequency(name)
            except (KeyError, AttributeError):
                continue
        if frequency == Frequency.ONCE:
            if destination.shape == array.shape and not np.array_equal(destination.numpy(), array.numpy()):
                destination.assign(array)
                copied.append(name)
            continue
        if not _is_value_attribute(target_view, name, frequency, destination):
            continue
        try:
            source_rows = source_view._rows(name, frequency, None, [0])
            target_rows = target_view._rows(name, frequency, None, None)
        except (KeyError, ValueError):
            continue
        if source_rows.shape[1] != target_rows.shape[1]:
            continue
        values = array.numpy()[source_rows[0]]
        current = destination.numpy()[target_rows[0]]
        if np.array_equal(values, current, equal_nan=values.dtype.kind == "f"):
            continue
        target_view.set_attribute(name, target, np.broadcast_to(values, (target.world_count, *values.shape)))
        copied.append(name)
    return copied


def _fingerprint(model: Model) -> str:
    """Content hash of a one-world model: its arrays (without pointers) and the geometry of its shape sources."""
    digest = hashlib.sha1()
    for name, array in sorted(_model_arrays(model).items()):
        if array.dtype == wp.uint64:
            continue
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(array.numpy()).tobytes())
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


def _keyed(items: Any, what: str) -> tuple[list[Any], list[Any]]:
    """Keys and values of candidates or scenarios given as a mapping or a sequence."""
    if items is None:
        return [None], [None]
    if isinstance(items, Mapping):
        keys, values = list(items.keys()), list(items.values())
    elif isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError(f"{what} must be a sequence or a mapping of names to values")
    else:
        keys, values = list(range(len(items))), list(items)
    if not keys:
        raise ValueError(f"{what} is empty")
    return keys, values


def _selection_key(name: str, labels: Any) -> tuple[str, Any]:
    return name.replace(".", ":", 1), _pattern_key(_labels(labels))


def _labels(labels: Any) -> Any:
    """Labels with tuples (hashable, as in control keys) as lists."""
    return list(labels) if isinstance(labels, tuple) else labels


def _split_key(key: Any) -> tuple[str, Any]:
    """``(name, labels)`` of a schedule key or probe given as ``name`` or ``(name, labels)``."""
    if isinstance(key, str):
        return key, None
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str):
        return key[0], _labels(key[1])
    raise TypeError(f"expected an attribute name or (name, labels), got {key!r}")


def _describe(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 1:
        return f"{array.reshape(()).item():.6g}" if array.dtype.kind == "f" else str(array.reshape(()).item())
    return np.array2string(array.ravel()[:6], precision=4, separator=", ") + ("..." if array.size > 6 else "")


# ----------------------------------------------------------------------------------------------------------------------
# Run plan: schedules, controllers, and probes resolved to device arrays


class _Schedule:
    def __init__(self, key, rows: wp.array, values: wp.array, target: wp.array):
        self.key = key
        self.rows = rows
        self.values = values  # [F, n, words]
        self.target = _row_words(target)
        self.kernel = _apply_rows_u32 if self.target.dtype == wp.uint32 else _apply_rows_u8

    def signature(self):
        return (self.key, self.rows.ptr, self.values.ptr, self.values.shape, self.target.ptr)


class _Probe:
    def __init__(self, name: str, source: Callable[[], wp.array] | wp.array, rows: wp.array | None, shape: tuple):
        self.name = name
        self.source = source  # array, or callable returning the array (callable probes)
        self.rows = rows
        self.shape = shape  # per-record shape of the typed view, e.g. (W, k)
        self.buffer: wp.array | None = None  # [capacity, n, words]
        self.typed: wp.array | None = None

    def signature(self):
        source = self.source if callable(self.source) else self.source.ptr
        return (self.name, source, None if self.rows is None else self.rows.ptr, self.buffer.ptr, self.buffer.shape)


class BatchRollout:
    """Rolls out many worlds of one model together and records named probes.

    The rollout builds a model with ``world_count`` copies of one world, the
    solver, and the collision pipeline once, and owns the states and the
    control. :meth:`reset` sets the start state of every world, from the
    model or from a saved state (branching); :attr:`view` and
    :meth:`set_state` set per-world values by label; :meth:`run` steps all
    worlds and records probes into arrays of shape ``[T, world, ...]``
    without a host synchronization per step, replaying a CUDA graph per
    frame on CUDA devices. :meth:`evaluate` runs a table of candidates x
    scenarios on top of this, with agent-written setup, scores, and pass
    predicates. See :doc:`/concepts/batched_evaluation`.

    Example:

    .. code-block:: python

        rollout = newton.utils.BatchRollout(build, 64, solver=newton.solvers.SolverMuJoCo, dt=0.002)
        rollout.view.set_attribute("shape_material_mu", rollout.model, np.linspace(0.2, 1.0, 64), labels="*pad*")
        rollout.reset(saved_state, world=0)  # every world starts from world 0 of saved_state
        records = rollout.run(500, record={"ball": ("body_q", "ball")}, every=10)
        heights = records["ball"][:, :, 0, 2]  # [T, world]

    Args:
        build: One world: a :class:`~newton.ModelBuilder`, or a function
            without arguments that returns one or returns the
            :class:`~newton.Model` it finalized from one. For a returned
            Model, the rollout copies the builder that the function finalized
            and then the values of the Model's arrays, so edits the function
            made after :meth:`~newton.ModelBuilder.finalize` apply to every
            world.
        world_count: Number of worlds in the batch.
        solver: Function that creates the solver for a model, e.g. a solver
            class or a script's ``make_solver``.
        pipeline: Function that creates the
            :class:`~newton.CollisionPipeline` for a model (e.g.
            ``newton.CollisionPipeline``), or ``None`` to step without one
            (for solvers that find their own contacts). The pipeline collides
            before every physics step.
        dt: Physics time step [s].
        substeps: Physics steps per frame. Schedules advance one row per
            frame, and probes are recorded at frame boundaries.
        device: Device of the batch; ``None`` for the device of a returned
            Model or the current Warp device.
        capture: Whether to replay each frame as a CUDA graph on CUDA
            devices. Without a graph, frames launch their kernels one by one.
    """

    def __init__(
        self,
        build: ModelBuilder | Callable[[], ModelBuilder | Model],
        world_count: int,
        *,
        solver: Callable[[Model], Any],
        pipeline: Callable[[Model], CollisionPipeline | None] | None = None,
        dt: float,
        substeps: int = 1,
        device: wp.DeviceLike = None,
        capture: bool = True,
    ):
        if int(world_count) < 1:
            raise ValueError(f"world_count must be at least 1, got {world_count}")
        if int(substeps) < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}")
        if not dt > 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        source = _call_build(build)
        builder, one_world, options = source.builder, source.one_world, source.options
        if device is None:
            device = one_world.device if one_world is not None else options.get("device")
        self._source = source
        self._solver_factory, self._pipeline_factory = solver, pipeline
        self._capture_enabled = bool(capture)

        self.world_count: int = int(world_count)
        """Number of worlds in the batch."""
        self.dt: float = float(dt)
        """Physics time step [s]."""
        self.substeps: int = int(substeps)
        """Physics steps per frame."""
        self.frame_dt: float = self.dt * self.substeps
        """Duration of a frame [s]."""

        self.model: Model = _replicate(builder, self.world_count, device, options)
        """The model with :attr:`world_count` copies of the built world."""
        self.copied_attributes: list[str] = _copy_world_values(one_world, self.model) if one_world is not None else []
        """Model attributes whose values the rollout copied from the Model that ``build`` returned (edits made
        after finalize())."""
        self.device = self.model.device
        """Device of the batch."""
        for name, labels, value in source.presets:
            self._write_shared(name, labels, value)

        self.solver = solver(self.model)
        """The solver that steps all worlds."""
        self.pipeline: CollisionPipeline | None = pipeline(self.model) if pipeline is not None else None
        """The collision pipeline, or ``None``."""
        self.contacts = self.pipeline.contacts() if self.pipeline is not None else None
        """Contacts the pipeline writes, or ``None``."""
        self.view = WorldView(self.model, solver=self.solver)
        """:class:`~newton.selection.WorldView` of :attr:`model` that checks model edits against :attr:`solver`
        (refusing attributes it shares across worlds) and notifies it."""

        self.state_0: State = self.model.state()
        self.state_1: State = self.model.state()
        self.control: Control = self.model.control()
        """The control that every frame steps with."""
        self._initial = self.model.state()

        self.frame_index = wp.zeros(1, dtype=wp.int32, device=self.device)
        """Frames since the last :meth:`reset`, on the device (schedules read it)."""
        self._run_frames = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.frames_done: int = 0
        """Frames since the last :meth:`reset`."""
        self.records: dict[str, wp.array] = {}
        """Device arrays of the last :meth:`run`, shape ``[T, world, ...]`` (see :meth:`run`)."""
        self.record_time: np.ndarray = np.zeros(0)
        """Times [s] since :meth:`reset` of the rows of the last :meth:`run`'s records, shape [T]."""

        self._graphs: dict[Any, Any] = {}
        self._schedules: dict[Any, wp.array] = {}
        self._probes: dict[Any, _Probe] = {}
        self._siblings: collections.OrderedDict[Any, BatchRollout] = collections.OrderedDict()
        self.reset()

    # ------------------------------------------------------------------------------------------------------------------
    # State

    @property
    def state(self) -> State:
        """The current state of all worlds (the start state after :meth:`reset`)."""
        return self.state_0

    def reset(self, state: State | None = None, *, model: Model | None = None, world: int | None = None) -> None:
        """Set the start state of every world and rewind the frame counter.

        Without ``state``, every world starts from the model's initial state:
        :attr:`Model.joint_q <newton.Model.joint_q>` and
        :attr:`Model.joint_qd <newton.Model.joint_qd>` with body poses from
        :func:`~newton.eval_fk`, and the model's particle state. A ``state``
        is copied array by array as it is (see
        :meth:`newton.selection.WorldView.copy_state`), which branches the
        batch from a saved state:

        - a state of :attr:`model` is copied world by world, or, with
          ``world``, from that world into every world;
        - a state of another model whose worlds have the same layout (for
          example the single-world model of a running simulation, or a
          checkpoint of a model with another world count) is copied from its
          world ``world`` (default 0) into every world. Pass that model as
          ``model``; a state with exactly one world's rows needs none.

        The solver's internal buffers (warm starts, actuator activations) and
        the contacts are cleared.

        Args:
            state: State to start from, or ``None`` for the model's initial
                state.
            model: Model of ``state`` when it is not :attr:`model`.
            world: World of ``state`` to copy into every world.
        """
        if state is None:
            initial, source = self._initial, self.model
            for name in ("joint_q", "joint_qd", "body_q", "body_qd", "particle_q", "particle_qd"):
                array = getattr(initial, name, None)
                if isinstance(array, wp.array) and array.size:
                    array.assign(getattr(source, name))
            if self.model.joint_count:
                eval_fk(self.model, initial.joint_q, initial.joint_qd, initial)
            self.state_0.assign(initial)
        elif model is not None and model is not self.model:
            self.view.copy_state(self.state_0, state, src_world=0 if world is None else world, src_model=model)
        elif self._same_layout(state):
            if world is None:
                self.state_0.assign(state)
            else:
                self.view.copy_state(self.state_0, state, src_world=world)
        elif self._one_world_layout(state):
            if world not in (None, 0):
                raise ValueError(f"state holds one world; world must be 0, got {world}")
            self._copy_one_world(state)
        else:
            raise ValueError(
                "state does not have the layout of this rollout's model or of one of its worlds; pass its model as "
                "model="
            )
        self.state_1.assign(self.state_0)
        self.frame_index.zero_()
        self.frames_done = 0
        self._reset_solver(None)
        if self.pipeline is not None and hasattr(self.pipeline, "reset_contact_matching"):
            self.pipeline.reset_contact_matching()
        if self.contacts is not None:
            self.contacts.clear()

    def _reset_solver(self, world_mask: wp.array | None) -> None:
        if hasattr(self.solver, "reset"):
            self.solver.reset(self.state_0, world_mask=world_mask, flags=StateFlags.NONE)

    @staticmethod
    def _state_arrays(state: State) -> dict[str, wp.array]:
        return {name: array for name, array in WorldView._state_arrays(state).items() if array.size}

    def _same_layout(self, state: State) -> bool:
        mine = self._state_arrays(self.state_0)
        theirs = self._state_arrays(state)
        common = [name for name in mine if name in theirs]
        return bool(common) and all(mine[name].shape == theirs[name].shape for name in common)

    def _one_world_layout(self, state: State) -> bool:
        mine = self._state_arrays(self.state_0)
        theirs = self._state_arrays(state)
        common = [name for name in mine if name in theirs]
        return bool(common) and all(
            theirs[name].shape[0] * self.world_count == mine[name].shape[0]
            and theirs[name].shape[1:] == mine[name].shape[1:]
            for name in common
        )

    def _copy_one_world(self, state: State) -> None:
        view = self.view
        dst_arrays, src_arrays = self._state_arrays(self.state_0), self._state_arrays(state)
        for name, dst in dst_arrays.items():
            src = src_arrays.get(name)
            if src is None:
                continue
            frequency = view._resolve(name)[1]
            dst_rows = view._rows(name, frequency, None, None)
            src_rows = np.tile(np.arange(src.shape[0]), dst_rows.shape[0])
            if dst_rows.size != src_rows.size:
                raise ValueError(f"State.{name}: {src.shape[0]} rows per world, the rollout has {dst_rows.shape[1]}")
            view._copy_rows(src, view._device_index(src_rows), dst, view._device_index(dst_rows))

    def set_state(
        self,
        name: str,
        values: Any,
        *,
        labels: LabelsLike = None,
        worlds: Sequence[int] | slice | int | None = None,
    ) -> None:
        """Write per-world values to the current state, keeping it consistent.

        Writes like :meth:`newton.selection.WorldView.set_attribute` (values
        of shape ``[len(worlds), k, ...]`` for the ``k`` selected rows of each
        world, broadcasting). After a write to ``joint_q`` or ``joint_qd``,
        the body poses and velocities of the written worlds follow from
        :func:`~newton.eval_fk`. The solver's internal buffers of the written
        worlds are cleared.

        Args:
            name: State attribute, e.g. ``"joint_q"``, ``"joint_qd"``, or
                ``"particle_q"``.
            values: Per-world values.
            labels: Label pattern or model indices selecting the rows, or
                ``None`` for every row of a world.
            worlds: Worlds to write; ``None`` for all worlds.
        """
        self._write_states([(name, labels, values, worlds)])

    def _write_states(self, writes: list[tuple[str, Any, Any, Any]]) -> None:
        worlds_written: set[int] = set()
        kinematic = False
        for name, labels, values, worlds in writes:
            self.view.set_attribute(name, self.state_0, values, labels=labels, worlds=worlds)
            selected = self.view._world_indices(worlds)
            worlds_written.update(int(w) for w in selected)
            kinematic |= name.replace(".", ":", 1) in ("joint_q", "joint_qd")
        if not worlds_written:
            return
        model = self.model
        world_mask_host = np.zeros(self.world_count + 1, dtype=bool)
        world_mask_host[sorted(worlds_written)] = True
        if kinematic and model.articulation_count:
            articulation_world = model.articulation_world.numpy()
            mask = np.isin(articulation_world, sorted(worlds_written))
            if self.world_count == 1:
                mask |= articulation_world < 0
            eval_fk(
                model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                mask=wp.array(mask, dtype=wp.bool, device=self.device),
            )
        self.state_1.assign(self.state_0)
        self._reset_solver(wp.array(world_mask_host, dtype=wp.bool, device=self.device))

    def _write_shared(self, name: str, labels: Any, value: Any) -> None:
        """Write one value to every world (or the single shared array) before the solver is built."""
        name = name.replace(".", ":", 1)
        try:
            frequency = self.model.get_attribute_frequency(name) if name != "gravity" else Frequency.WORLD
        except KeyError:
            frequency = None
        if frequency == Frequency.ONCE:
            owner, leaf = _attribute_owner(self.model, name)
            array = getattr(owner, leaf)
            array.assign(np.broadcast_to(np.asarray(value), (array.shape[0], *_value_shape(array))).copy())
            return
        view = WorldView(self.model)
        values = np.broadcast_to(np.asarray(value), (self.world_count, *np.shape(value)))
        view.set_attribute(name, self.model, values, labels=labels)

    # ------------------------------------------------------------------------------------------------------------------
    # Running

    def run(
        self,
        frames: int,
        *,
        control: Any = None,
        record: Mapping[str, Any] | None = None,
        every: int = 1,
    ) -> dict[str, np.ndarray]:
        """Step every world ``frames`` frames from the current state and record probes.

        A frame is :attr:`substeps` physics steps. Before every physics step
        the rollout clears the state's forces, applies the schedules of
        ``control``, calls its control functions, collides (with a pipeline),
        and steps the solver. On CUDA devices the frame is captured once as a
        CUDA graph and replayed; the schedules and probes are resolved to
        device arrays, so no step synchronizes with the host. Runs continue
        from the current state; :meth:`reset` starts over.

        ``control`` holds schedules, control functions, or a list of both:

        - A schedule maps a control attribute (or ``(attribute, labels)``)
          to values of shape ``[F, world, k, ...]`` or, for the same values
          in every world, ``[F, k, ...]``, where ``k`` is the number of rows
          the labels select in a world. Frame ``f`` since :meth:`reset` applies
          row ``min(f, F - 1)``. Values may be NumPy arrays (copied into a
          device buffer that later runs reuse) or Warp arrays with the
          attribute's dtype and ``F * world * k`` rows (used in place).
        - A control function ``fn(rollout)`` runs before every physics step.
          On CUDA devices it is captured into the graph, so it must only
          launch Warp work; :attr:`frame_index` holds the frame on the
          device.

        ``record`` maps names to probes:

        - ``"body_q"`` or ``("body_q", labels)``: the rows of a state (or
          control) attribute that the labels select in each world, recorded
          with shape ``[T, world, k]`` of the attribute's dtype (NumPy:
          ``[T, world, k, *value_shape]``, e.g. ``7`` for transforms);
        - a function ``fn(rollout) -> wp.array`` that computes a device array
          with one row per world (the same array on every call), recorded
          with shape ``[T, *array.shape]``.

        Row 0 holds the probes at the start of the run, row ``i`` after
        ``i * every`` frames, so ``T = frames // every + 1``.

        Args:
            frames: Number of frames to step.
            control: Schedules and control functions; ``None`` steps with
                the current :attr:`control`.
            record: Probes to record by name.
            every: Record every ``every`` frames.

        Returns:
            The records as NumPy arrays by name. The device arrays stay in
            :attr:`records`, and :attr:`record_time` holds the time of each
            row [s].
        """
        frames, every = int(frames), int(every)
        if frames < 0:
            raise ValueError(f"frames must not be negative, got {frames}")
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        schedules, controllers = self._resolve_control(control)
        rows = frames // every + 1
        probes = self._resolve_probes(record or {}, rows)

        start = self.frames_done
        self._run_frames.zero_()
        for probe in probes:
            self._record(probe, every)  # row 0
        if frames:
            signature = (
                tuple(schedule.signature() for schedule in schedules),
                tuple(controllers),  # the functions themselves: ids of collected functions get reused
                tuple(probe.signature() for probe in probes),
                every,
            )
            graph = self._graphs.get(signature)
            done = 0
            if graph is None and self._capture_enabled and self.device.is_cuda:
                self._frame(schedules, controllers, probes, every)  # loads kernels and lets the solver allocate
                done = 1
                graph = self._capture(schedules, controllers, probes, every)
                if graph is not None:
                    self._graphs[signature] = graph
                    while len(self._graphs) > 16:
                        self._graphs.pop(next(iter(self._graphs)))
            for _ in range(frames - done):
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    self._frame(schedules, controllers, probes, every)
        self.frames_done = start + frames

        self.record_time = (start + np.arange(rows) * every) * self.frame_dt
        self.records = {probe.name: probe.typed[:rows] for probe in probes}
        return {name: array.numpy() for name, array in self.records.items()}

    def _capture(self, schedules, controllers, probes, every):
        try:
            with wp.ScopedCapture(device=self.device) as capture:
                self._frame(schedules, controllers, probes, every)
        except Exception as error:
            warnings.warn(f"BatchRollout: CUDA graph capture failed, stepping without a graph: {error}", stacklevel=3)
            self._capture_enabled = False
            return None
        return capture.graph

    def _frame(self, schedules, controllers, probes, every):
        for substep in range(self.substeps):
            self.state_0.clear_forces()
            for schedule in schedules:
                wp.launch(
                    schedule.kernel,
                    dim=(schedule.values.shape[1], schedule.values.shape[2]),
                    inputs=[schedule.values, self.frame_index, schedule.rows],
                    outputs=[schedule.target],
                    device=self.device,
                )
            for fn in controllers:
                fn(self)
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            if self.substeps % 2 == 1 and substep == self.substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(_advance, dim=1, inputs=[self._run_frames], device=self.device)
        for probe in probes:
            self._record(probe, every)
        wp.launch(_advance, dim=1, inputs=[self.frame_index], device=self.device)

    def _record(self, probe: _Probe, every: int) -> None:
        source = probe.source() if callable(probe.source) else probe.source
        words = _row_words(source)
        rows = probe.rows
        if rows is None:
            rows = self.view._device_index(np.arange(source.shape[0]))
        kernel = _record_rows_u32 if words.dtype == wp.uint32 else _record_rows_u8
        wp.launch(
            kernel,
            dim=(rows.shape[0], words.shape[1]),
            inputs=[words, rows, self._run_frames, every],
            outputs=[probe.buffer],
            device=self.device,
        )

    # Control and probe resolution

    def _attribute(self, name: str, *sources) -> tuple[Any, wp.array]:
        name = name.replace(".", ":", 1)
        for source in sources:
            owner, leaf = _attribute_owner(source, name)
            array = getattr(owner, leaf, None) if owner is not None else None
            if isinstance(array, wp.array):
                return source, array
        kinds = " or ".join(type(source).__name__ for source in sources)
        raise AttributeError(f"{kinds} has no array '{name}'")

    def _resolve_control(self, control: Any) -> tuple[list[_Schedule], list[Callable]]:
        if control is None:
            return [], []
        items = control if isinstance(control, (list, tuple)) else [control]
        schedules, controllers = [], []
        for item in items:
            if callable(item):
                controllers.append(item)
            elif isinstance(item, Mapping):
                for key, values in item.items():
                    name, labels = _split_key(key)
                    schedules.append(self._schedule(name, labels, values))
            else:
                raise TypeError(f"control items must be mappings of schedules or functions, got {type(item)!r}")
        return schedules, controllers

    def _schedule(self, name: str, labels: Any, values: Any) -> _Schedule:
        name = name.replace(".", ":", 1)
        _, target = self._attribute(name, self.control)
        frequency = self.view._resolve(name)[1]
        rows = self.view._rows(name, frequency, labels, None)
        world_count, row_count = rows.shape
        value_shape = _value_shape(target)
        if wp.types.is_array(values):
            frames = values.shape[0]
            if values.dtype != target.dtype or values.size != frames * world_count * row_count * int(
                np.prod(target.shape[1:], dtype=np.int64)
            ):
                raise ValueError(
                    f"schedule for '{name}' must be a {target.dtype.__name__} array with F x {world_count} worlds x "
                    f"{row_count} rows"
                )
            flat = wp.array(
                ptr=values.ptr,
                dtype=target.dtype,
                shape=(frames * world_count * row_count, *target.shape[1:]),
                device=values.device,
                copy=False,
            )
            words = _row_words(flat)
            device_values = wp.array(
                ptr=words.ptr,
                dtype=words.dtype,
                shape=(frames, world_count * row_count, words.shape[1]),
                device=values.device,
                copy=False,
            )
            device_values._ref = values  # keep the caller's array alive with the view
        else:
            host = np.asarray(values)
            per_world = (world_count, row_count, *value_shape)
            if host.ndim == 1 and row_count == 1 and not value_shape:
                host = host[:, None]
            if host.shape[1:] == per_world:
                pass
            elif host.shape[1:] == per_world[1:]:
                host = np.broadcast_to(host[:, None], (host.shape[0], *per_world))
            else:
                raise ValueError(
                    f"schedule for '{name}' has shape {host.shape}; expected [F, {world_count}, {row_count}"
                    f"{', ' if value_shape else ''}{', '.join(map(str, value_shape))}] or [F, {row_count}"
                    f"{', ' if value_shape else ''}{', '.join(map(str, value_shape))}]"
                )
            frames = host.shape[0]
            if frames < 1:
                raise ValueError(f"schedule for '{name}' has no rows")
            scalar = getattr(target.dtype, "_wp_scalar_type_", target.dtype)
            typed = np.ascontiguousarray(host, dtype=wp.dtype_to_numpy(scalar))
            buffer_key = ("schedule", name, _pattern_key(labels), typed.shape)
            buffer = self._schedules.get(buffer_key)
            if buffer is None:
                buffer = wp.empty(
                    (frames * world_count * row_count, *target.shape[1:]), dtype=target.dtype, device=self.device
                )
                self._schedules[buffer_key] = buffer
            buffer.assign(typed.reshape(frames * world_count * row_count, *value_shape))
            words = _row_words(buffer)
            device_values = wp.array(
                ptr=words.ptr,
                dtype=words.dtype,
                shape=(frames, world_count * row_count, words.shape[1]),
                device=self.device,
                copy=False,
            )
            device_values._ref = buffer
        return _Schedule((name, _pattern_key(labels)), self.view._device_index(rows), device_values, target)

    def _resolve_probes(self, record: Mapping[str, Any], rows: int) -> list[_Probe]:
        probes = []
        for name, spec in record.items():
            if callable(spec):
                array = spec(self)
                if not isinstance(array, wp.array):
                    raise TypeError(f"probe '{name}' must return a Warp array, got {type(array)!r}")
                key = ("probe", name, id(spec), array.ptr, array.shape, array.dtype)
                probe = self._probes.get(key)
                if probe is None:
                    probe = _Probe(name, functools.partial(spec, self), None, tuple(array.shape))
                    self._probes[key] = probe
                sample = array
            else:
                attribute, labels = _split_key(spec)
                attribute = attribute.replace(".", ":", 1)
                _, array = self._attribute(attribute, self.state_0, self.control)
                frequency = self.view._resolve(attribute)[1]
                selected = self.view._rows(attribute, frequency, labels, None)
                key = ("probe", name, attribute, _pattern_key(labels), array.ptr)
                probe = self._probes.get(key)
                if probe is None:
                    probe = _Probe(name, array, self.view._device_index(selected), (*selected.shape, *array.shape[1:]))
                    self._probes[key] = probe
                sample = array
            words = _row_words(sample)
            count = probe.rows.shape[0] if probe.rows is not None else sample.shape[0]
            if probe.buffer is None or probe.buffer.shape[0] < rows:
                probe.buffer = wp.zeros((rows, count, words.shape[1]), dtype=words.dtype, device=self.device)
                probe.typed = wp.array(
                    ptr=probe.buffer.ptr,
                    dtype=sample.dtype,
                    shape=(rows, *probe.shape),
                    device=self.device,
                    copy=False,
                )
                probe.typed._ref = probe.buffer
            probes.append(probe)
        return probes

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation

    class WorldSetup:
        """Edits of one case's world, collected by :meth:`BatchRollout.evaluate` from its ``setup`` function.

        Reads return the values every world starts from (shape ``[k, ...]``
        for the ``k`` rows the labels select in a world); writes apply to
        this case's world only. :meth:`~BatchRollout.evaluate` applies the
        writes of all worlds of a batch together, one write per attribute.
        """

        def __init__(self, rollout: BatchRollout, start: State):
            """Created by :meth:`BatchRollout.evaluate` for each case."""
            self._rollout = rollout
            self._start = start
            self._model_edits: list[tuple[str, Any, Any]] = []
            self._state_edits: list[tuple[str, Any, Any]] = []
            self._control_edits: list[tuple[str, Any, Any]] = []
            self._schedule_edits: list[tuple[str, Any, Any]] = []
            self._shared: list[tuple[str, Any, Any]] = []
            self._skip: set[tuple] = set()

        @property
        def model(self) -> Model:
            """The batch's model (for lookups; edit through :meth:`set_model`)."""
            return self._rollout.model

        def _read(self, name: str, source: Any, labels: Any) -> np.ndarray:
            return self._rollout.view.get_attribute(name, source, labels=_labels(labels), worlds=[0])[0]

        def get_model(self, name: str, labels: LabelsLike = None) -> np.ndarray:
            """Model values of the selected rows, shape ``[k, ...]``."""
            return self._read(name, self._rollout.model, labels)

        def get_state(self, name: str, labels: LabelsLike = None) -> np.ndarray:
            """Start-state values of the selected rows, shape ``[k, ...]``."""
            return self._read(name, self._start, labels)

        def get_control(self, name: str, labels: LabelsLike = None) -> np.ndarray:
            """Control values of the selected rows, shape ``[k, ...]``."""
            return self._read(name, self._rollout.control, labels)

        def set_model(self, name: str, value: Any, labels: LabelsLike = None) -> None:
            """Set a model attribute of the selected rows in this world.

            ``value`` broadcasts to ``[k, ...]``. Attributes that the solver
            takes per world (see
            :meth:`~newton.solvers.SolverBase.check_world_values`) are set in
            this case's world. Values of attributes the solver shares across
            worlds or reads only when it is constructed make this case run in
            a separate batch whose model has the value in every world; the
            evaluation's :attr:`~BatchRollout.Evaluation.batches` lists why.
            """
            self._model_edits.append((name, _labels(labels), np.asarray(value)))

        def set_state(self, name: str, value: Any, labels: LabelsLike = None) -> None:
            """Set start-state values of the selected rows in this world (see :meth:`BatchRollout.set_state`)."""
            self._state_edits.append((name, _labels(labels), np.asarray(value)))

        def set_control(self, name: str, value: Any, labels: LabelsLike = None) -> None:
            """Set constant control values of the selected rows in this world."""
            self._control_edits.append((name, _labels(labels), np.asarray(value)))

        def set_schedule(self, name: str, values: Any, labels: LabelsLike = None) -> None:
            """Set per-frame control values of the selected rows in this world, shape ``[F, k, ...]``.

            Frame ``f`` applies row ``min(f, F - 1)``. This world's schedule
            replaces a shared schedule of the same attribute and labels.
            """
            values = np.asarray(values)
            if values.ndim < 1 or values.shape[0] < 1:
                raise ValueError("a schedule needs at least one row")
            self._schedule_edits.append((name, _labels(labels), values))

    class Evaluation:
        """Result of :meth:`BatchRollout.evaluate`: one row per candidate x scenario and a summary per candidate."""

        def __init__(self, rows, candidates, scenarios, batches, metrics, worst):
            """Created by :meth:`BatchRollout.evaluate`."""
            self.rows: list[dict[str, Any]] = rows
            """One dict per case: ``candidate`` and ``scenario`` (keys), ``batch``, the metrics, and ``passed``
            (``None`` without a pass predicate)."""
            self.candidates: list[Any] = candidates
            """Candidate keys, in order."""
            self.scenarios: list[Any] = scenarios
            """Scenario keys, in order."""
            self.batches: list[dict[str, Any]] = batches
            """One dict per batch: ``worlds`` (cases in the batch), ``world_count`` (worlds of its model),
            ``cases`` (``(candidate, scenario)`` keys), ``build`` (``"rollout"`` or the candidate whose build it
            used), ``shared`` (values set in every world), and ``reason`` (why it is a separate batch, or
            ``None``)."""
            self.metrics: list[str] = metrics
            """Metric names, in the order ``score`` returned them."""
            self.worst: dict[str, str] = dict(worst or {})
            """Which end of each metric is worse (``"max"`` or ``"min"``), as passed to evaluate."""
            self.summary: list[dict[str, Any]] = self._summarize()
            """One dict per candidate: ``candidate``, ``cases``, ``passed`` (count), ``pass_fraction``,
            ``failed`` (scenario keys), ``min``/``max``/``mean`` (per metric), and ``worst`` (per metric named in
            :attr:`worst`: the worst value and its scenario)."""

        def _summarize(self) -> list[dict[str, Any]]:
            summary = []
            for candidate in self.candidates:
                rows = [row for row in self.rows if row["candidate"] == candidate]
                verdicts = [row["passed"] for row in rows]
                judged = [v for v in verdicts if v is not None]
                entry: dict[str, Any] = {
                    "candidate": candidate,
                    "cases": len(rows),
                    "passed": sum(bool(v) for v in judged) if judged else None,
                    "pass_fraction": (sum(bool(v) for v in judged) / len(judged)) if judged else None,
                    "failed": [row["scenario"] for row in rows if row["passed"] is False],
                    "min": {},
                    "max": {},
                    "mean": {},
                    "worst": {},
                }
                for metric in self.metrics:
                    values = np.array([_as_float(row.get(metric)) for row in rows], dtype=np.float64)
                    finite = values[~np.isnan(values)]
                    entry["min"][metric] = float(finite.min()) if finite.size else math.nan
                    entry["max"][metric] = float(finite.max()) if finite.size else math.nan
                    entry["mean"][metric] = float(finite.mean()) if finite.size else math.nan
                    direction = self.worst.get(metric)
                    if direction is not None and finite.size:
                        index = int(np.nanargmax(values) if direction == "max" else np.nanargmin(values))
                        entry["worst"][metric] = (float(values[index]), rows[index]["scenario"])
                summary.append(entry)
            return summary

        def best(self, metric: str, *, reduce: str = "mean", minimize: bool = True) -> Any:
            """Key of the candidate with the best ``metric`` over its scenarios.

            Args:
                metric: Metric name.
                reduce: Reduction over scenarios: ``"mean"``, ``"min"``, or
                    ``"max"``.
                minimize: Whether smaller values are better.
            """
            if reduce not in ("mean", "min", "max"):
                raise ValueError(f"reduce must be 'mean', 'min', or 'max', got {reduce!r}")
            values = np.array([entry[reduce].get(metric, math.nan) for entry in self.summary], dtype=np.float64)
            if np.all(np.isnan(values)):
                raise ValueError(f"no finite values of metric '{metric}'")
            index = int(np.nanargmin(values) if minimize else np.nanargmax(values))
            return self.summary[index]["candidate"]

        def compare(self, previous: BatchRollout.Evaluation) -> BatchRollout.EvaluationDiff:
            """Differences from an earlier evaluation, case by case (a regression check).

            Cases are matched by candidate and scenario keys. When both
            evaluations have a single candidate, cases are matched by scenario
            alone, so a new version of one candidate compares against the
            previous one.
            """
            return BatchRollout.EvaluationDiff(previous, self)

        def format(self, *, rows: bool | None = None, digits: int = 4) -> str:
            """The batches, the table, and the summary as text.

            Args:
                rows: Whether to include one row per case; ``None`` includes
                    them for up to 40 cases.
                digits: Significant digits of numbers.
            """
            lines = []
            first = 0
            while first < len(self.batches):
                batch = self.batches[first]
                last = first
                while last + 1 < len(self.batches) and all(
                    self.batches[last + 1][key] == batch[key] for key in ("world_count", "build", "reason")
                ):
                    last += 1
                reason = f": {batch['reason']}" if batch["reason"] else ""
                counts = "+".join(str(self.batches[i]["worlds"]) for i in range(first, last + 1))
                name = f"batch {first}" if first == last else f"batches {first}-{last}"
                lines.append(
                    f"{name}: {counts} case(s) in {batch['world_count']} world(s), build {batch['build']}{reason}"
                )
                first = last + 1
            if rows is None:
                rows = len(self.rows) <= 40
            if rows:
                header = ["candidate", "scenario", *self.metrics, "passed"]
                table = [
                    [_cell(row["candidate"], digits), _cell(row["scenario"], digits)]
                    + [_cell(row.get(metric), digits) for metric in self.metrics]
                    + [_cell(row["passed"], digits)]
                    for row in self.rows
                ]
                lines.append(_table(header, table))
            header = ["candidate", "passed", "failed scenarios"]
            columns = []
            for metric in self.metrics:
                direction = self.worst.get(metric)
                columns.append((metric, direction))
                header.append(f"{metric} ({direction})" if direction else f"{metric} (min..max)")
            table = []
            for entry in self.summary:
                passed = "-" if entry["passed"] is None else f"{entry['passed']}/{entry['cases']}"
                failed = ", ".join(str(key) for key in entry["failed"]) or "-"
                line = [_cell(entry["candidate"], digits), passed, failed]
                for metric, direction in columns:
                    if direction and metric in entry["worst"]:
                        value, scenario = entry["worst"][metric]
                        line.append(f"{_cell(value, digits)} @ {scenario}")
                    else:
                        line.append(f"{_cell(entry['min'][metric], digits)}..{_cell(entry['max'][metric], digits)}")
                table.append(line)
            lines.append(_table(header, table))
            return "\n".join(lines)

        def __str__(self) -> str:
            return self.format()

        def __repr__(self) -> str:
            return f"BatchRollout.Evaluation({len(self.candidates)} candidates x {len(self.scenarios)} scenarios)"

    class EvaluationDiff:
        """Case-by-case differences between two :class:`BatchRollout.Evaluation` results."""

        def __init__(self, before: BatchRollout.Evaluation, after: BatchRollout.Evaluation):
            """Created by :meth:`BatchRollout.Evaluation.compare`."""
            single = len(before.candidates) == 1 and len(after.candidates) == 1

            def key(row):
                return row["scenario"] if single else (row["candidate"], row["scenario"])

            old = {key(row): row for row in before.rows}
            new = {key(row): row for row in after.rows}
            self.regressions: list[Any] = [
                k for k in new if k in old and old[k]["passed"] and new[k]["passed"] is False
            ]
            """Cases that passed before and fail now."""
            self.fixes: list[Any] = [k for k in new if k in old and old[k]["passed"] is False and new[k]["passed"]]
            """Cases that failed before and pass now."""
            self.added: list[Any] = [k for k in new if k not in old]
            """Cases only in the newer evaluation."""
            self.removed: list[Any] = [k for k in old if k not in new]
            """Cases only in the earlier evaluation."""
            self.changes: dict[str, list[tuple[Any, float, float]]] = {}
            """Per metric of both evaluations: ``(case, before, after)`` of the matched cases, largest absolute
            change first."""
            for metric in after.metrics:
                if metric not in before.metrics:
                    continue
                entries = [
                    (k, _as_float(old[k].get(metric)), _as_float(row.get(metric))) for k, row in new.items() if k in old
                ]
                entries.sort(key=lambda e: -_change(e[1], e[2]))
                self.changes[metric] = entries
            self.pass_fraction: tuple[float | None, float | None] = (
                _pass_fraction(before.rows),
                _pass_fraction(after.rows),
            )
            """Fraction of passing cases before and after (``None`` without a pass predicate)."""

        def format(self, *, top: int = 3, digits: int = 4) -> str:
            """The differences as text, with the ``top`` largest changes per metric."""
            before, after = self.pass_fraction
            lines = [
                f"pass fraction: {_cell(before, digits)} -> {_cell(after, digits)}; "
                f"{len(self.regressions)} regression(s), {len(self.fixes)} fix(es)"
            ]
            if self.regressions:
                lines.append("regressions: " + ", ".join(map(str, self.regressions)))
            if self.fixes:
                lines.append("fixes: " + ", ".join(map(str, self.fixes)))
            if self.added or self.removed:
                lines.append(f"added: {len(self.added)}, removed: {len(self.removed)}")
            for metric, entries in self.changes.items():
                changed = [e for e in entries if _change(e[1], e[2]) > 0.0][:top]
                if changed:
                    parts = [f"{k}: {_cell(a, digits)} -> {_cell(b, digits)}" for k, a, b in changed]
                    lines.append(f"{metric}: " + "; ".join(parts))
            return "\n".join(lines)

        def __str__(self) -> str:
            return self.format()

    def evaluate(
        self,
        candidates: Sequence[Any] | Mapping[Any, Any],
        scenarios: Sequence[Any] | Mapping[Any, Any] | None = None,
        *,
        frames: int,
        score: Callable[[dict[str, np.ndarray], list[tuple[Any, Any]]], Mapping[str, Any]],
        setup: Callable[[BatchRollout.WorldSetup, Any, Any], None] | None = None,
        control: Any = None,
        record: Mapping[str, Any] | None = None,
        every: int = 1,
        passed: Callable[[dict[str, Any], Any, Any], bool] | None = None,
        worst: Mapping[str, str] | None = None,
        build: Callable[[Any], ModelBuilder | Model] | None = None,
        initial_state: State | None = None,
        initial_model: Model | None = None,
        initial_world: int = 0,
    ) -> BatchRollout.Evaluation:
        """Run every candidate in every scenario, one world per case, and score the cases.

        Each case (candidate, scenario) gets a world. Every world starts from
        the same state (``initial_state``, else the model's initial state),
        then ``setup(world, candidate, scenario)`` sets the case's values
        through a :class:`WorldSetup`: model values (e.g. friction or gains),
        start-state values (e.g. perturbed poses), constant controls, and
        per-frame schedules. Writes of all worlds of a batch are applied
        together. The rollout runs ``frames`` frames with ``control`` (shared
        schedules and control functions, see :meth:`run`) and ``record``,
        and ``score(records, cases)`` returns metrics per world: a mapping of
        metric names to sequences with one value per case of the batch,
        where ``records`` holds the batch's records (``[T, case, ...]``) and
        ``cases`` its ``(candidate, scenario)`` values. ``passed(metrics,
        candidate, scenario)`` decides whether a case passes.

        Cases are grouped into batches:

        - Cases run in the worlds of this rollout, :attr:`world_count` at a
          time; unused worlds of the last batch run unchanged and are not
          scored.
        - A case whose setup sets a model attribute that the solver does not
          take per world (for example a solver option, or a value it reads
          only when it is constructed) runs in a separate model that has the
          value in every world. Values equal to the model's are not a change.
        - With ``build``, each candidate's world is ``build(candidate)``
          (one world, as for the constructor), and candidates whose builds
          differ run in separate models.

        Separate models are kept for later calls (up to 8). Model and control
        values that setups changed are restored afterwards. The state is
        left at the end of the last batch.

        Args:
            candidates: Candidates as a sequence (keyed by index) or a mapping
                of keys to candidates.
            scenarios: Scenarios as a sequence or mapping, or ``None`` for one
                scenario (``None``).
            frames: Frames to run per case.
            score: Metrics per case from the records.
            setup: Per-case setup of a world.
            control: Schedules and control functions shared by all cases
                (see :meth:`run`); a case's own schedule of the same
                attribute and labels replaces the shared one.
            record: Probes (see :meth:`run`).
            every: Record every ``every`` frames.
            passed: Pass predicate of a case.
            worst: Which end of a metric is worse, ``"max"`` or ``"min"``,
                for the summary.
            build: Per-candidate builds, for candidates that change the
                model's structure or values the solver shares.
            initial_state: State every case starts from.
            initial_model: Model of ``initial_state`` if it is not this
                rollout's (see :meth:`reset`).
            initial_world: World of ``initial_state`` to start from.

        Returns:
            The table, per-candidate summary, and batches.
        """
        candidate_keys, candidate_values = _keyed(candidates, "candidates")
        scenario_keys, scenario_values = _keyed(scenarios, "scenarios")
        if worst is not None and any(value not in ("max", "min") for value in worst.values()):
            raise ValueError("worst values must be 'max' or 'min'")
        cases = [(c, s) for c in range(len(candidate_keys)) for s in range(len(scenario_keys))]

        # Build groups: this rollout, or one rollout per distinct candidate build.
        groups: list[tuple[BatchRollout, str, list[tuple[int, int]]]] = []
        if build is None:
            groups.append((self, "rollout", cases))
        else:
            by_fingerprint: dict[str, tuple[Any, list[int]]] = {}
            for c, value in enumerate(candidate_values):
                source = _call_build(functools.partial(build, value))
                model = source.one_world
                if model is None:
                    model = source.builder.finalize(device=self.device)
                entry = by_fingerprint.setdefault(_fingerprint(model), (source, []))
                entry[1].append(c)
            for fingerprint, (source, members) in by_fingerprint.items():
                group_cases = [case for case in cases if case[0] in members]
                world_count = min(self.world_count, len(group_cases))
                rollout = self._sibling(("build", fingerprint, world_count), source, world_count)
                groups.append((rollout, f"of candidate {candidate_keys[members[0]]}", group_cases))

        rows: dict[tuple[int, int], dict[str, Any]] = {}
        batches: list[dict[str, Any]] = []
        metrics: list[str] = []
        touched: dict[int, tuple[BatchRollout, dict]] = {}
        try:
            for rollout, build_name, group_cases in groups:
                rollout.reset(initial_state, model=initial_model, world=initial_world)
                start = rollout.model.state()
                start.assign(rollout.state_0)
                setups = {}
                for case in group_cases:
                    world = BatchRollout.WorldSetup(rollout, start)
                    if setup is not None:
                        setup(world, candidate_values[case[0]], scenario_values[case[1]])
                    setups[case] = world
                for target, shared, reason, subgroup in rollout._split_shared(setups, group_cases):
                    base = touched.setdefault(id(target), (target, {}))[1]
                    for first in range(0, len(subgroup), target.world_count):
                        chunk = subgroup[first : first + target.world_count]
                        chunk_control = target._apply_setups(
                            chunk, setups, control, base, initial_state, initial_model, initial_world
                        )
                        records = target.run(frames, control=chunk_control, record=record, every=every)
                        active = {name: array[:, : len(chunk)] for name, array in records.items()}
                        chunk_values = [(candidate_values[c], scenario_values[s]) for c, s in chunk]
                        result = score(active, chunk_values)
                        if not isinstance(result, Mapping):
                            raise TypeError(f"score must return a mapping of metric names, got {type(result)!r}")
                        for metric in result:
                            if metric not in metrics:
                                metrics.append(metric)
                        for index, (c, s) in enumerate(chunk):
                            row = {"candidate": candidate_keys[c], "scenario": scenario_keys[s], "batch": len(batches)}
                            for metric, values in result.items():
                                sequence = np.asarray(values, dtype=object)
                                if sequence.ndim == 0 or sequence.shape[0] != len(chunk):
                                    raise ValueError(
                                        f"score returned {np.shape(values)} values of '{metric}' for a batch of "
                                        f"{len(chunk)} cases"
                                    )
                                row[metric] = _plain(sequence[index])
                            metric_values = {m: row[m] for m in result}
                            row["passed"] = (
                                bool(passed(metric_values, candidate_values[c], scenario_values[s]))
                                if passed is not None
                                else None
                            )
                            rows[(c, s)] = row
                        batches.append(
                            {
                                "worlds": len(chunk),
                                "world_count": target.world_count,
                                "cases": [(candidate_keys[c], scenario_keys[s]) for c, s in chunk],
                                "build": build_name,
                                "shared": {name: value for name, _, value in shared},
                                "reason": reason,
                            }
                        )
        finally:
            for target, base in touched.values():
                target._restore(base)
        ordered = [rows[case] for case in cases]
        return BatchRollout.Evaluation(ordered, candidate_keys, scenario_keys, batches, metrics, worst)

    def _sibling(self, key: Any, source: _BuildSource, world_count: int) -> BatchRollout:
        """A rollout of another build or with shared values, with this rollout's solver, pipeline, and steps."""
        rollout = self._siblings.get(key)
        if rollout is None:
            rollout = BatchRollout(
                source,
                world_count,
                solver=self._solver_factory,
                pipeline=self._pipeline_factory,
                dt=self.dt,
                substeps=self.substeps,
                device=self.device,
                capture=self._capture_enabled,
            )
            self._siblings[key] = rollout
            while len(self._siblings) > 8:
                self._siblings.popitem(last=False)
        else:
            self._siblings.move_to_end(key)
        return rollout

    def _split_shared(self, setups: dict, cases: list) -> list[tuple[BatchRollout, tuple, str | None, list]]:
        """Group cases by the model values they need in every world (attributes the solver shares)."""
        verdicts: dict[tuple, str | None] = {}
        current: dict[tuple, np.ndarray] = {}
        keys: dict[tuple[int, int], tuple] = {}
        for case in cases:
            shared = []
            for name, labels, value in setups[case]._model_edits:
                selection = _selection_key(name, labels)
                if selection not in verdicts:
                    verdicts[selection] = self._shared_reason(name, labels)
                if verdicts[selection] is None:
                    continue
                if selection not in current:
                    current[selection] = self._current_values(name, labels)
                base = current[selection]
                try:
                    unchanged = np.array_equal(np.broadcast_to(value, base.shape), base)
                except ValueError:
                    unchanged = False
                if not unchanged:
                    shared.append((selection, name, labels, value))
            shared.sort(key=lambda item: repr(item[0]))
            keys[case] = tuple(
                (selection, np.asarray(value).dtype.str, np.asarray(value).tobytes(), np.shape(value))
                for selection, _, _, value in shared
            )
            setups[case]._shared = [(name, labels, value) for _, name, labels, value in shared]
            # Edits of shared attributes are applied by the batch's model (or are no change); none per world.
            setups[case]._skip = {
                _selection_key(name, labels)
                for name, labels, _ in setups[case]._model_edits
                if verdicts[_selection_key(name, labels)] is not None
            }
        groups: dict[tuple, list] = {}
        for case in cases:
            groups.setdefault(keys[case], []).append(case)
        result = []
        for key, members in groups.items():
            shared = tuple(setups[members[0]]._shared)
            if not key:
                result.append((self, (), None, members))
                continue
            reasons = []
            for name, labels, value in shared:
                reasons.append(f"{name}={_describe(value)} ({verdicts[_selection_key(name, labels)]})")
            world_count = min(self.world_count, len(members))
            base = self._source
            source = _BuildSource(base.builder, base.one_world, base.options, base.presets + shared)
            rollout = self._sibling(("shared", key, world_count), source, world_count)
            result.append((rollout, shared, "; ".join(reasons), members))
        return result

    def _shared_reason(self, name: str, labels: Any) -> str | None:
        """Why per-world values of ``name`` cannot share this model, or ``None`` if they can."""
        name = name.replace(".", ":", 1)
        try:
            frequency = self.model.get_attribute_frequency(name) if name != "gravity" else Frequency.WORLD
        except KeyError:
            raise KeyError(f"Model has no attribute '{name}' with a known frequency") from None
        if frequency == Frequency.ONCE:
            return "one value for all worlds of a model"
        self.view._check_model_attribute(name, frequency)
        rows = self.view.get_indices(name, labels)
        try:
            self.solver.check_world_values(name, rows.ravel())
        except ValueError as error:
            return str(error)
        return None

    def _current_values(self, name: str, labels: Any) -> np.ndarray:
        name = name.replace(".", ":", 1)
        frequency = self.model.get_attribute_frequency(name) if name != "gravity" else Frequency.WORLD
        if frequency == Frequency.ONCE:
            owner, leaf = _attribute_owner(self.model, name)
            return getattr(owner, leaf).numpy()
        return self.view.get_attribute(name, self.model, labels=labels, worlds=[0])[0]

    def _apply_setups(self, chunk, setups, control, base, initial_state, initial_model, initial_world) -> list:
        """Write the setups of a chunk of cases (case ``i`` in world ``i``), reset, and return the chunk's control."""
        writes: dict[str, dict[tuple, list]] = {"model": {}, "control": {}, "state": {}, "schedule": {}}
        for world, case in enumerate(chunk):
            setup = setups[case]
            shared = setup._skip
            for kind, edits in (
                ("model", setup._model_edits),
                ("control", setup._control_edits),
                ("state", setup._state_edits),
                ("schedule", setup._schedule_edits),
            ):
                for name, labels, value in edits:
                    selection = _selection_key(name, labels)
                    if kind == "model" and selection in shared:
                        continue
                    writes[kind].setdefault(selection, [name, labels, {}])[2][world] = value

        # Model and control values: the base value in every world whose case does not set one. Keys that an earlier
        # chunk wrote are written back to the base value.
        for kind, target in (("model", self.model), ("control", self.control)):
            kind_writes = writes[kind]
            for (entry_kind, *selection), (name, labels, _) in base.items():
                if entry_kind == kind and tuple(selection) not in kind_writes:
                    kind_writes[tuple(selection)] = [name, labels, {}]
            for selection, (name, labels, per_world) in kind_writes.items():
                entry = base.get((kind, *selection))
                if entry is None:
                    entry = (name, labels, self.view.get_attribute(name, target, labels=labels))
                    base[(kind, *selection)] = entry
                values = entry[2].copy()
                for world, value in per_world.items():
                    values[world] = np.broadcast_to(value, values.shape[1:])
                self.view.set_attribute(name, target, values, labels=labels)

        # Model values such as joint_q define the initial state, so the reset follows the model writes.
        self.reset(initial_state, model=initial_model, world=initial_world)
        state_writes = []
        for name, labels, per_world in writes["state"].values():
            worlds = sorted(per_world)
            values = self.view.get_attribute(name, self.state_0, labels=labels, worlds=worlds)
            for index, world in enumerate(worlds):
                values[index] = np.broadcast_to(per_world[world], values.shape[1:])
            state_writes.append((name, labels, values, worlds))
        if state_writes:
            self._write_states(state_writes)

        # Schedules: the shared ones in every world, replaced in a world by its case's own.
        shared_schedules, controllers = self._split_control(control)
        schedules = {
            selection: (name, labels, values) for selection, (name, labels, values) in shared_schedules.items()
        }
        for selection, (name, labels, edits) in writes["schedule"].items():
            constant = self.view.get_attribute(name, self.control, labels=labels)  # [W, k, ...]
            row_shape = constant.shape[1:]
            per_world = {world: self._schedule_rows(values, row_shape) for world, values in edits.items()}
            frames = max(values.shape[0] for values in per_world.values())
            shared = shared_schedules.get(selection)
            if shared is not None:
                shared_values = self._schedule_rows(shared[2], row_shape)
                frames = max(frames, shared_values.shape[0])
                fill = np.broadcast_to(_hold(shared_values, frames)[:, None], (frames, *constant.shape)).copy()
            else:
                fill = np.broadcast_to(constant[None], (frames, *constant.shape)).copy()
            for world, values in per_world.items():
                fill[:, world] = _hold(values, frames)
            schedules[selection] = (name, labels, fill)
        return [{(name, labels): values for name, labels, values in schedules.values()}, *controllers]

    @staticmethod
    def _schedule_rows(values: np.ndarray, row_shape: tuple) -> np.ndarray:
        """Schedule values of one world as ``[F, k, ...]`` (``[F]`` is accepted for a single scalar row)."""
        values = np.asarray(values)
        if values.ndim == 1 and row_shape == (1,):
            values = values[:, None]
        if values.shape[1:] != row_shape:
            try:
                values = np.broadcast_to(
                    values.reshape(values.shape[0], *([1] * (len(row_shape) + 1 - values.ndim)), *values.shape[1:]),
                    (values.shape[0], *row_shape),
                )
            except ValueError:
                raise ValueError(
                    f"schedule rows of shape {values.shape[1:]} do not broadcast to {row_shape} (the selected rows)"
                ) from None
        return values

    @staticmethod
    def _split_control(control: Any) -> tuple[dict, list]:
        """Shared schedules by selection (as NumPy arrays) and control functions."""
        if control is None:
            return {}, []
        items = control if isinstance(control, (list, tuple)) else [control]
        schedules, controllers = {}, []
        for item in items:
            if callable(item):
                controllers.append(item)
            elif isinstance(item, Mapping):
                for key, values in item.items():
                    name, labels = _split_key(key)
                    host = values.numpy() if wp.types.is_array(values) else np.asarray(values)
                    schedules[_selection_key(name, labels)] = (name, labels, host)
            else:
                raise TypeError(f"control items must be mappings of schedules or functions, got {type(item)!r}")
        return schedules, controllers

    def _restore(self, base: dict) -> None:
        for (kind, *_), (name, labels, values) in base.items():
            target = self.model if kind == "model" else self.control
            self.view.set_attribute(name, target, values, labels=labels)


# ----------------------------------------------------------------------------------------------------------------------
# Formatting helpers


def _hold(values: np.ndarray, frames: int) -> np.ndarray:
    """Schedule rows extended to ``frames`` rows by holding the last row."""
    values = np.asarray(values)
    if values.shape[0] < frames:
        values = np.concatenate([values, np.repeat(values[-1:], frames - values.shape[0], axis=0)])
    return values


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _change(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return math.inf
    if a == b:
        return 0.0
    return abs(b - a)


def _pass_fraction(rows: list[dict]) -> float | None:
    judged = [row["passed"] for row in rows if row["passed"] is not None]
    return sum(bool(v) for v in judged) / len(judged) if judged else None


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _cell(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "NO"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i]) for i in range(len(header))
    ]
    lines = ["  ".join(text.ljust(width) for text, width in zip(header, widths, strict=True))]
    lines += ["  ".join(text.ljust(width) for text, width in zip(row, widths, strict=True)) for row in rows]
    return "\n".join(lines)
