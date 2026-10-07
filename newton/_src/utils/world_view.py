# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-world access to model, state, and control attributes selected by label."""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from warp.types import is_array

from ..geometry import GeoType
from ..geometry.utils import compute_shape_radius
from ..sim import Control, Model, ModelFlags, State
from .selection import get_name_from_label, match_labels

if TYPE_CHECKING:
    from ..solvers import SolverBase

Frequency = Model.AttributeFrequency

# Shape types whose collision data ModelBuilder.finalize() derives from shape_scale.
_SCALE_BAKED_SHAPE_TYPES = (int(GeoType.MESH), int(GeoType.CONVEX_MESH), int(GeoType.HFIELD), int(GeoType.GAUSSIAN))

# Attributes that define model structure rather than per-entity values.
_STRUCTURAL_ATTRIBUTES = frozenset(
    {"shape_type", "shape_source_ptr", "shape_heightfield_index", "shape_edge_range", "joint_type", "joint_dof_dim"}
)

# State arrays without an attribute spec on the model.
_STATE_FREQUENCIES = {"particle_f": Frequency.PARTICLE}

# Derived inverse arrays kept consistent with the edited attribute.
_INVERSE_ATTRIBUTES = {
    "body_mass": "body_inv_mass",
    "particle_mass": "particle_inv_mass",
    "body_inertia": "body_inv_inertia",
}


@wp.kernel(enable_backward=False)
def _copy_rows_u32_kernel(
    src: wp.array2d[wp.uint32],
    src_rows: wp.array[wp.int32],
    dst: wp.array2d[wp.uint32],
    dst_rows: wp.array[wp.int32],
):
    i, j = wp.tid()
    dst[dst_rows[i], j] = src[src_rows[i], j]


@wp.kernel(enable_backward=False)
def _copy_rows_u8_kernel(
    src: wp.array2d[wp.uint8],
    src_rows: wp.array[wp.int32],
    dst: wp.array2d[wp.uint8],
    dst_rows: wp.array[wp.int32],
):
    i, j = wp.tid()
    dst[dst_rows[i], j] = src[src_rows[i], j]


def _row_words(array: wp.array) -> wp.array:
    """Reinterpret the rows of a contiguous array as 4-byte (or 1-byte) words, shape [rows, words]."""
    if not array.is_contiguous:
        raise ValueError("WorldView needs contiguous attribute arrays")
    row_bytes = array.strides[0]
    dtype, words = (wp.uint32, row_bytes // 4) if row_bytes % 4 == 0 else (wp.uint8, row_bytes)
    return wp.array(ptr=array.ptr, dtype=dtype, shape=(array.shape[0], words), device=array.device, copy=False)


def _value_shape(array: wp.array) -> tuple[int, ...]:
    """NumPy shape of one row of ``array``, e.g. ``(3,)`` for ``vec3`` and ``(7,)`` for ``transform``."""
    return (*array.shape[1:], *getattr(array.dtype, "_shape_", ()))


def _pattern_key(labels: Any) -> Any:
    if labels is None or isinstance(labels, (str, re.Pattern)):
        return labels
    if isinstance(labels, (list, tuple)) and (
        all(isinstance(item, str) for item in labels) or all(isinstance(item, (int, np.integer)) for item in labels)
    ):
        return tuple(item if isinstance(item, str) else int(item) for item in labels)
    raise TypeError(
        "labels must be a glob string, a list of glob strings, a compiled regular expression, or a list of "
        f"model indices; got {labels!r}"
    )


def _attribute_owner(source: Model | State | Control, name: str) -> tuple[Any, str]:
    if ":" in name:
        namespace, leaf = name.split(":", 1)
        return getattr(source, namespace, None), leaf
    return source, name


class WorldView:
    """Per-world access to the rows of model, state, and control attributes.

    The view selects the rows of an attribute in each world by label pattern
    and reads or writes them as arrays of shape ``[world, row, ...]``. It works
    for any attribute whose rows belong to worlds: bodies, shapes, joints,
    joint DOFs and coordinates, articulations, particles, mimic constraints,
    per-world values such as ``gravity`` and ``mujoco:<option>``, and custom
    frequencies with a world or articulation assignment (for example
    ``mujoco:actuator`` and ``mujoco:tendon``). Unlike
    :class:`~newton.selection.ArticulationView`, the selected rows need not
    belong to identical articulations, and static shapes of a world can be
    selected as well.

    Global entities (world ``-1``) are shared by all worlds and cannot take
    per-world values; patterns that match them raise :class:`ValueError`. In a
    model with a single world, global rows count as rows of world ``0``.

    Typical use is to evaluate many candidates in one replicated model (see
    :meth:`newton.ModelBuilder.replicate`): set one candidate per world with
    :meth:`set_attribute`, clone a start state into every world with
    :meth:`copy_state`, step all worlds together, and read per-world results
    with :meth:`get_attribute`.

    Example:

    .. code-block:: python

        view = newton.selection.WorldView(model)
        # One friction coefficient per world for all shapes labeled "*pad*".
        view.set_attribute(
            "shape_material_mu", model, np.linspace(0.2, 1.0, model.world_count), labels="*pad*", solver=solver
        )
        view.copy_state(state_0, start_state)
        ...
        heights = view.get_attribute("body_q", state_0, labels="*ball")[:, 0, 2]

    Rows are selected with the label patterns or model indices described in
    :ref:`label-matching`. As in :meth:`newton.Model.find_bodies`, a pattern
    matches either the full label or its last path component (the text after
    the last ``/``), so ``"left_finger"`` selects ``"robot/left_finger"``.
    Rows of joint DOFs and joint coordinates are selected by the labels of
    their joints. Within a world, selected rows keep their model order, which
    is the same in every replicated world.

    Args:
        model: The model whose worlds are accessed.
        solver: Solver that :meth:`set_attribute` checks and notifies for
            model edits when its ``solver`` argument is ``None``.
    """

    def __init__(self, model: Model, solver: SolverBase | None = None):
        self.model = model
        """The model whose worlds this view accesses."""
        self.solver = solver
        """Solver that :meth:`set_attribute` checks and notifies for model edits by default, or ``None``."""
        self.world_count = int(model.world_count)
        """Number of worlds in :attr:`model`."""
        self._layouts: dict[Any, tuple[np.ndarray, list[str] | None]] = {}
        self._selections: dict[tuple[Any, Any], tuple[np.ndarray, np.ndarray]] = {}
        # Device row indices and copy plans, cached by value.
        self._device_rows: dict[Any, Any] = {}
        self._source_views: dict[int, WorldView] = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Public API

    def get_indices(
        self,
        name: str,
        labels: str | list[str] | re.Pattern[str] | list[int] | None = None,
        worlds: Sequence[int] | slice | int | None = None,
    ) -> np.ndarray:
        """Return the model indices of the rows of an attribute selected in each world.

        Args:
            name: Attribute name, e.g. ``"joint_target_ke"`` (indices of joint
                DOFs), ``"body_q"`` (indices of bodies), or ``"mujoco:gravcomp"``.
            labels: Label pattern or model indices selecting the rows, or
                ``None`` for every row that belongs to a world.
            worlds: World indices, a slice of worlds, or ``None`` for all worlds.

        Returns:
            Integer array of shape ``[len(worlds), k]`` with the ``k`` selected
            row indices of each world.
        """
        name, frequency = self._resolve(name)
        return self._rows(name, frequency, labels, worlds)

    def get_attribute(
        self,
        name: str,
        source: Model | State | Control,
        labels: str | list[str] | re.Pattern[str] | list[int] | None = None,
        worlds: Sequence[int] | slice | int | None = None,
    ) -> np.ndarray:
        """Read the selected rows of an attribute in each world.

        Args:
            name: Attribute name, e.g. ``"body_q"`` or ``"mujoco:gravcomp"``.
            source: The model, state, or control that holds the attribute.
            labels: Label pattern or model indices selecting the rows, or
                ``None`` for every row that belongs to a world.
            worlds: World indices, a slice of worlds, or ``None`` for all worlds.

        Returns:
            Array of shape ``[len(worlds), k, *value_shape]``, where
            ``value_shape`` is the NumPy shape of one value, e.g. ``(7,)`` for
            transforms.
        """
        name, frequency = self._resolve(name)
        array = self._array(source, name)
        rows = self._rows(name, frequency, labels, worlds)
        world_count, row_count = rows.shape
        self._check_rows_fit(name, array, rows)
        buffer = wp.empty((rows.size, *array.shape[1:]), dtype=array.dtype, device=array.device)
        self._copy_rows(array, self._device_index(rows), buffer, None)
        return buffer.numpy().reshape(world_count, row_count, *_value_shape(array))

    def set_attribute(
        self,
        name: str,
        target: Model | State | Control,
        values: Any,
        *,
        labels: str | list[str] | re.Pattern[str] | list[int] | None = None,
        worlds: Sequence[int] | slice | int | None = None,
        solver: SolverBase | None = None,
    ) -> int:
        """Write per-world values to the selected rows of an attribute.

        The first axis of ``values`` holds one entry per selected world. Each
        entry broadcasts against the ``k`` selected rows of its world with
        NumPy rules, so a value per world has shape ``[len(worlds)]`` and a
        value per world and row has shape ``[len(worlds), k]`` (plus the value
        shape, e.g. ``3`` for ``vec3`` attributes). A scalar sets every
        selected row. A Warp array with the attribute's dtype and
        ``len(worlds) * k`` rows is copied on the device without a host
        round-trip.

        Writing ``body_mass``, ``body_inertia``, or ``particle_mass`` also
        updates the matching inverse array. Writing ``shape_scale`` also
        updates :attr:`~newton.Model.shape_collision_radius`; it raises for
        mesh, convex-mesh, heightfield, Gaussian, and SDF-backed shapes, whose
        collision data :meth:`~newton.ModelBuilder.finalize` computes from
        their scale.

        When ``target`` is the model and a solver is given (or bound to the
        view), the view first calls
        :meth:`~newton.solvers.SolverBase.check_world_values`, which raises
        for attributes the solver shares across worlds, reads only at
        construction, or does not read, and after writing calls
        :meth:`~newton.solvers.SolverBase.notify_model_changed` with the
        returned flags.

        Args:
            name: Attribute name, e.g. ``"shape_material_mu"``,
                ``"joint_target_ke"``, ``"joint_q"``, or ``"mujoco:gravcomp"``.
            target: The model, state, or control that holds the attribute.
            values: Per-world values; see above.
            labels: Label pattern or model indices selecting the rows, or
                ``None`` for every row that belongs to a world.
            worlds: World indices, a slice of worlds, or ``None`` for all worlds.
            solver: Solver to check and notify when ``target`` is the model;
                ``None`` for :attr:`solver`.

        Returns:
            :class:`~newton.ModelFlags` bits that cover the edit when ``target``
            is the model, else ``0``.

        Raises:
            ValueError: If the attribute cannot take per-world values, the
                pattern matches global rows or different row counts per world,
                ``values`` does not broadcast, or ``solver`` does not use
                per-world values of the selected rows.
        """
        name, frequency = self._resolve(name)
        solver = self.solver if solver is None else solver
        is_model = isinstance(target, Model)
        if is_model and target is not self.model:
            raise ValueError("target is a different Model than the one this WorldView was created for")
        array = self._array(target, name)
        if is_model:
            self._check_model_attribute(name, frequency)
        rows = self._rows(name, frequency, labels, worlds)
        self._check_rows_fit(name, array, rows)
        row_index = self._device_index(rows)

        if is_model and name == "shape_scale":
            self._check_shape_scale_rows(rows.ravel())
        if is_model and solver is not None:
            solver.check_world_values(name, rows.ravel())

        buffer = self._values(values, array, rows.shape)
        self._copy_rows(buffer, None, array, row_index)

        if is_model:
            self._update_derived(name, buffer, rows, row_index)
            flags = ModelFlags.from_attributes(name)
            if solver is not None:
                solver.notify_model_changed(flags)
            return flags
        return 0

    def copy_state(
        self,
        dst: State,
        src: State,
        src_world: int = 0,
        worlds: Sequence[int] | slice | int | None = None,
        src_model: Model | None = None,
    ) -> None:
        """Copy the state of one world into worlds of ``dst``.

        Copies every per-entity array present in both states (for example
        :attr:`~newton.State.joint_q`, :attr:`~newton.State.joint_qd`,
        :attr:`~newton.State.body_q`, :attr:`~newton.State.body_qd`, and
        particle arrays). The source may be a state of another model whose
        worlds have the same layout, e.g. the single-world plant of a planner
        with many worlds. Values are copied as they are, so worlds that
        :meth:`~newton.ModelBuilder.replicate` placed apart with ``spacing``
        receive the source world's positions. Solver-internal data such as
        warm starts is not copied. After the first call with the same
        arguments, the copy only launches kernels.

        Args:
            dst: State of this view's model to write.
            src: State to read.
            src_world: World of ``src`` to copy.
            worlds: Worlds of ``dst`` to write; ``None`` for all worlds.
            src_model: Model of ``src``; ``None`` if it is this view's model.

        Raises:
            ValueError: If the source world's row count of an array differs
                from the row count of the destination worlds.
        """
        source_view = self if src_model is None or src_model is self.model else self._source_view(src_model)
        if not 0 <= src_world < source_view.world_count:
            raise ValueError(f"src_world {src_world} is out of range [0, {source_view.world_count})")
        dst_arrays, src_arrays = self._state_arrays(dst), self._state_arrays(src)
        names = tuple(name for name, array in dst_arrays.items() if array.size and name in src_arrays)
        worlds_key = (worlds.start, worlds.stop, worlds.step) if isinstance(worlds, slice) else worlds
        if worlds_key is not None and not isinstance(worlds_key, (int, tuple)):
            worlds_key = tuple(np.asarray(worlds_key).ravel().tolist())
        plan_key = ("copy", id(source_view), src_world, worlds_key, names)
        plan = self._device_rows.get(plan_key)
        if plan is None:
            plan = []
            for name in names:
                frequency = _STATE_FREQUENCIES.get(name) or self.model.get_attribute_frequency(name)
                dst_rows = self._rows(name, frequency, None, worlds)
                src_rows = source_view._rows(name, frequency, None, [src_world])
                if src_rows.shape[1] != dst_rows.shape[1]:
                    raise ValueError(
                        f"State.{name.replace(':', '.')}: source world {src_world} has {src_rows.shape[1]} rows, "
                        f"destination worlds have {dst_rows.shape[1]}"
                    )
                self._check_rows_fit(name, dst_arrays[name], dst_rows)
                source_view._check_rows_fit(name, src_arrays[name], src_rows)
                src_index = self._device_index(np.tile(src_rows[0], dst_rows.shape[0]))
                plan.append((name, src_index, self._device_index(dst_rows)))
            self._device_rows[plan_key] = plan
        for name, src_index, dst_index in plan:
            self._copy_rows(src_arrays[name], src_index, dst_arrays[name], dst_index)

    # ------------------------------------------------------------------------------------------------------------------
    # Selection

    def _resolve(self, name: str) -> tuple[str, Any]:
        name = name.replace(".", ":", 1)
        if name == "gravity":
            return name, Frequency.WORLD
        frequency = _STATE_FREQUENCIES.get(name)
        if frequency is None:
            try:
                frequency = self.model.get_attribute_frequency(name)
            except KeyError:
                raise KeyError(f"Model has no attribute '{name}' with a known frequency") from None
        return name, frequency

    def _layout(self, frequency: Any) -> tuple[np.ndarray, list[str] | None]:
        """World of each row and the labels used to select rows of a frequency."""
        layout = self._layouts.get(frequency)
        if layout is not None:
            return layout
        model = self.model
        if frequency == Frequency.WORLD:
            layout = (np.arange(self.world_count), None)
        elif frequency in (Frequency.JOINT_DOF, Frequency.JOINT_COORD):
            starts = (model.joint_qd_start if frequency == Frequency.JOINT_DOF else model.joint_q_start).numpy()
            counts = np.diff(starts)
            joint_world = model.joint_world.numpy()
            layout = (
                np.repeat(joint_world, counts),
                [label for label, n in zip(model.joint_label, counts, strict=True) for _ in range(n)],
            )
        elif isinstance(frequency, str):
            layout = self._custom_layout(frequency)
        else:
            entity = {
                Frequency.BODY: "body",
                Frequency.SHAPE: "shape",
                Frequency.JOINT: "joint",
                Frequency.ARTICULATION: "articulation",
                Frequency.PARTICLE: "particle",
                Frequency.CONSTRAINT_MIMIC: "constraint_mimic",
            }.get(frequency)
            if entity is None:
                raise ValueError(f"Rows of frequency {getattr(frequency, 'name', frequency)} have no world assignment")
            world = getattr(model, f"{entity}_world")
            layout = (
                np.zeros(0, dtype=np.int32) if world is None else world.numpy(),
                getattr(model, f"{entity}_label", None),
            )
        self._layouts[frequency] = layout
        return layout

    def _custom_layout(self, frequency: str) -> tuple[np.ndarray, list[str] | None]:
        model = self.model
        owner, leaf = _attribute_owner(model, frequency)
        world = getattr(owner, f"{leaf}_world", None) if owner is not None else None
        if isinstance(world, wp.array):
            row_world = world.numpy()
        elif frequency in model.custom_frequency_articulation:
            owner = model.custom_frequency_articulation[frequency].numpy()
            articulation_world = model.articulation_world.numpy()
            row_world = np.where(owner >= 0, articulation_world[np.maximum(owner, 0)], -1)
        else:
            raise ValueError(f"Rows of custom frequency '{frequency}' have no world assignment")
        labels = None
        label_name = model.custom_frequency_label_attributes.get(frequency)
        if label_name is not None:
            owner, leaf = _attribute_owner(model, label_name)
            labels = getattr(owner, leaf, None)
        return row_world, labels

    def _selection(self, frequency: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
        """Selected rows sorted by world and the start of each world's rows (``world_count + 1`` entries)."""
        key = (frequency, _pattern_key(labels))
        selection = self._selections.get(key)
        if selection is not None:
            return selection
        row_world, row_labels = self._layout(frequency)
        domain = getattr(frequency, "name", frequency)
        pattern = _pattern_key(labels)
        if labels is None:
            matched = np.arange(row_world.shape[0])
        elif pattern and isinstance(pattern[0], int):
            matched = np.unique(np.asarray(pattern, dtype=np.int64))
            if matched.size != len(pattern) or matched[0] < 0 or matched[-1] >= row_world.shape[0]:
                raise ValueError(f"Indices must be unique and in [0, {row_world.shape[0]}), got {list(pattern)}")
        else:
            if row_labels is None:
                raise ValueError(f"Rows of frequency {domain} have no labels; pass labels=None or model indices")
            # Same matching as Model.find_bodies(): the full label or its last path component.
            names = [get_name_from_label(label) for label in row_labels]
            matched = np.union1d(match_labels(row_labels, labels), match_labels(names, labels)).astype(np.int64)
            if matched.size == 0:
                patterns = labels if isinstance(labels, list) else [labels]
                close = {
                    name
                    for item in patterns
                    for name in difflib.get_close_matches(str(getattr(item, "pattern", item)), sorted(set(names)), n=3)
                }
                hint = f"; closest names: {', '.join(sorted(close))}" if close else ""
                examples = ", ".join(repr(label) for label in row_labels[:5])
                raise KeyError(f"No {domain} labels match {labels!r} (labels include {examples}){hint}")
        worlds = row_world[matched].astype(np.int64)
        if self.world_count == 1:
            worlds = np.where(worlds < 0, 0, worlds)
        elif labels is not None and np.any(worlds < 0):
            shared = [row_labels[i] if row_labels is not None else int(i) for i in matched[worlds < 0][:5]]
            raise ValueError(
                f"Pattern {labels!r} matches global {domain} rows (world -1), which all worlds share and which "
                f"cannot take per-world values: {shared}"
            )
        local = worlds >= 0
        matched, worlds = matched[local], worlds[local]
        order = np.argsort(worlds, kind="stable")
        matched = matched[order]
        starts = np.zeros(self.world_count + 1, dtype=np.int64)
        np.cumsum(np.bincount(worlds, minlength=self.world_count), out=starts[1:])
        selection = (matched, starts)
        self._selections[key] = selection
        return selection

    def _world_indices(self, worlds: Sequence[int] | slice | int | None) -> np.ndarray:
        if worlds is None:
            return np.arange(self.world_count)
        if isinstance(worlds, slice):
            indices = np.arange(self.world_count)[worlds]
            if indices.size == 0:
                raise ValueError(f"worlds={worlds} selects no world")
            return indices
        indices = np.atleast_1d(np.asarray(worlds, dtype=np.int64))
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("worlds must be a non-empty sequence of world indices")
        if np.any(indices < 0) or np.any(indices >= self.world_count):
            raise ValueError(f"World indices must be in [0, {self.world_count}), got {indices.tolist()}")
        if np.unique(indices).size != indices.size:
            raise ValueError("World indices must be unique")
        return indices

    def _rows(self, name: str, frequency: Any, labels: Any, worlds: Any) -> np.ndarray:
        matched, starts = self._selection(frequency, labels)
        world_indices = self._world_indices(worlds)
        counts = starts[1:] - starts[:-1]
        selected = counts[world_indices]
        if np.any(selected != selected[0]):
            per_world = dict(zip(world_indices.tolist(), selected.tolist(), strict=True))
            raise ValueError(f"'{name}' rows matching {labels!r} differ in count between worlds: {per_world}")
        if selected[0] == 0:
            raise KeyError(f"No '{name}' rows matching {labels!r} in the selected worlds")
        return matched[starts[world_indices][:, None] + np.arange(selected[0])[None, :]]

    def _source_view(self, model: Model) -> WorldView:
        view = self._source_views.get(id(model))
        if view is None or view.model is not model:
            view = WorldView(model)
            self._source_views[id(model)] = view
        return view

    # ------------------------------------------------------------------------------------------------------------------
    # Arrays

    def _array(self, source: Model | State | Control, name: str) -> wp.array:
        owner, leaf = _attribute_owner(source, name)
        array = getattr(owner, leaf, None) if owner is not None else None
        if not isinstance(array, wp.array):
            raise AttributeError(f"{type(source).__name__} has no array '{name.replace(':', '.')}'")
        return array

    @staticmethod
    def _state_arrays(state: State) -> dict[str, wp.array]:
        arrays = {}
        for name, value in state.__dict__.items():
            if isinstance(value, wp.array):
                arrays[name] = value
            elif isinstance(value, Model.AttributeNamespace):
                for leaf, array in value.__dict__.items():
                    if isinstance(array, wp.array):
                        arrays[f"{name}:{leaf}"] = array
        return arrays

    def _device_index(self, rows: np.ndarray) -> wp.array:
        """Row indices on the model's device, cached by value."""
        rows = np.ascontiguousarray(rows, dtype=np.int32).ravel()
        key = ("index", rows.tobytes())
        index = self._device_rows.get(key)
        if index is None:
            index = wp.array(rows, dtype=wp.int32, device=self.model.device)
            self._device_rows[key] = index
        return index

    @staticmethod
    def _check_rows_fit(name: str, array: wp.array, rows: np.ndarray) -> None:
        if rows.size and int(rows.max()) >= array.shape[0]:
            raise ValueError(f"'{name}' has {array.shape[0]} rows, but the model layout selects row {int(rows.max())}")

    def _copy_rows(self, src: wp.array, src_rows: wp.array | None, dst: wp.array, dst_rows: wp.array | None) -> None:
        """``dst[dst_rows[i]] = src[src_rows[i]]``; ``None`` row indices mean ``0, 1, 2, ...``."""
        count = (src_rows if src_rows is not None else dst_rows).shape[0]
        if count == 0:
            return
        src_words, dst_words = _row_words(src), _row_words(dst)
        if src_words.dtype != dst_words.dtype or src_words.shape[1] != dst_words.shape[1]:
            raise ValueError("Source and destination rows have different sizes")
        identity = self._device_rows.get(("identity", count))
        if identity is None:
            identity = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=dst.device)
            self._device_rows[("identity", count)] = identity
        kernel = _copy_rows_u32_kernel if src_words.dtype == wp.uint32 else _copy_rows_u8_kernel
        wp.launch(
            kernel,
            dim=(count, src_words.shape[1]),
            inputs=[
                src_words,
                identity if src_rows is None else src_rows,
                dst_words,
                identity if dst_rows is None else dst_rows,
            ],
            device=dst.device,
        )

    def _values(self, values: Any, array: wp.array, shape: tuple[int, int]) -> wp.array:
        """Values as a contiguous array with ``shape[0] * shape[1]`` rows of ``array``'s dtype."""
        world_count, row_count = shape
        row_shape = array.shape[1:]
        if is_array(values):
            if values.dtype != array.dtype:
                raise TypeError(f"values has dtype {values.dtype}, expected {array.dtype}")
            if values.device != array.device:
                raise ValueError(f"values is on {values.device}, expected {array.device}")
            if values.size != world_count * row_count * int(np.prod(row_shape, dtype=np.int64)):
                raise ValueError(f"values has {values.size} elements, expected {world_count} worlds x {row_count} rows")
            if not values.is_contiguous:
                raise ValueError("values must be contiguous")
            return wp.array(
                ptr=values.ptr,
                dtype=array.dtype,
                shape=(world_count * row_count, *row_shape),
                device=array.device,
                copy=False,
            )
        values = np.asarray(values)
        value_shape = (row_count, *_value_shape(array))
        if values.ndim == 0:
            full = np.broadcast_to(values, (world_count, *value_shape))
        else:
            if values.shape[0] != world_count:
                raise ValueError(
                    f"values has {values.shape[0]} entries along its first axis, expected one per selected world "
                    f"({world_count})"
                )
            if values.ndim - 1 > len(value_shape):
                raise ValueError(f"values entries have shape {values.shape[1:]}, expected at most {value_shape}")
            expanded = values.reshape(world_count, *([1] * (len(value_shape) - values.ndim + 1)), *values.shape[1:])
            try:
                full = np.broadcast_to(expanded, (world_count, *value_shape))
            except ValueError:
                raise ValueError(
                    f"values entries of shape {values.shape[1:]} do not broadcast to {value_shape} "
                    f"({row_count} selected rows per world)"
                ) from None
        flat = np.ascontiguousarray(full).reshape(world_count * row_count, *_value_shape(array))
        return wp.array(flat, dtype=array.dtype, device=array.device)

    # ------------------------------------------------------------------------------------------------------------------
    # Model edits

    def _check_model_attribute(self, name: str, frequency: Any) -> None:
        if frequency == Frequency.ONCE:
            raise ValueError(f"'{name}' holds one value that all worlds share")
        spec = self.model._attribute_spec(name)
        if (
            name in _STRUCTURAL_ATTRIBUTES
            or name.startswith("_")
            or (spec is not None and (spec.references is not None or spec.compaction_policy != "generic"))
        ):
            raise ValueError(
                f"'{name}' defines the model structure; build worlds with different values in the ModelBuilder"
            )

    def _check_shape_scale_rows(self, rows: np.ndarray) -> None:
        model = self.model
        baked = np.isin(model.shape_type.numpy()[rows], _SCALE_BAKED_SHAPE_TYPES)
        sdf_index = getattr(model, "_shape_sdf_index", None)
        if sdf_index is not None and sdf_index.shape[0] == model.shape_count:
            baked |= sdf_index.numpy()[rows] >= 0
        if np.any(baked):
            names = [model.shape_label[i] for i in rows[baked][:5]]
            raise ValueError(
                "ModelBuilder.finalize() computes the collision data of mesh, convex-mesh, heightfield, Gaussian, "
                f"and SDF-backed shapes from their scale; shape_scale of {names} cannot change after finalize()"
            )

    def _update_derived(self, name: str, buffer: wp.array, rows: np.ndarray, row_index: wp.array) -> None:
        """Keep inverse masses, inverse inertias, and collision radii consistent with an edit."""
        model = self.model
        inverse_name = _INVERSE_ATTRIBUTES.get(name)
        if name == "shape_scale":
            scales = buffer.numpy()
            shape_type = model.shape_type.numpy()
            flat = rows.ravel()
            derived = np.array(
                [
                    compute_shape_radius(int(shape_type[shape]), scale, model.shape_source[shape])
                    for shape, scale in zip(flat, scales, strict=True)
                ],
                dtype=np.float32,
            )
            target = model.shape_collision_radius
        elif inverse_name is not None and isinstance(getattr(model, inverse_name, None), wp.array):
            target = getattr(model, inverse_name)
            values = buffer.numpy().astype(np.float64)
            if name == "body_inertia":
                derived = np.zeros_like(values)
                nonzero = np.any(values.reshape(len(values), -1) != 0.0, axis=1)
                if np.any(nonzero):
                    derived[nonzero] = np.linalg.inv(values[nonzero])
            else:
                derived = np.divide(1.0, values, out=np.zeros_like(values), where=values > 0.0)
        else:
            return
        derived_array = wp.array(derived.astype(np.float32), dtype=target.dtype, device=target.device)
        self._copy_rows(derived_array, None, target, row_index)
