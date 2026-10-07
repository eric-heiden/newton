# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learn which control and application arrays a simulation step writes.

Reset and restore rewind the state and only those arrays, so inputs that cells
set and that stepping does not overwrite (command schedules, perturbations,
targets) survive them. Each step is bracketed by device-side checksums of the
arrays not yet known to be written; the comparison runs on the device and its
flags are read only when a reset or restore needs them, so stepping adds no
host synchronization. Arrays already known to be written are no longer
checksummed.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

from ..sim.model import Model


@wp.kernel
def _mark_written_kernel(
    before: wp.array2d[wp.uint32],
    after: wp.array2d[wp.uint32],
    written: wp.array[wp.int32],
):
    segment = wp.tid()
    if before[segment, 0] != after[segment, 0] or before[segment, 1] != after[segment, 1]:
        written[segment] = 1
    # Ready for the next step's checksums, which accumulate atomically.
    for k in range(2):
        before[segment, k] = wp.uint32(0)
        after[segment, k] = wp.uint32(0)


def public_arrays(owner: Any, label: str) -> dict[str, wp.array]:
    """Public Warp arrays of ``owner`` and of its attribute namespaces, as ``label.<name>``."""
    result = {}
    if owner is None:
        return result
    for name, value in vars(owner).items():
        if name.startswith("_"):
            continue
        if isinstance(value, wp.array):
            result[f"{label}.{name}"] = value
        elif isinstance(value, Model.AttributeNamespace):
            for child, array in vars(value).items():
                if not child.startswith("_") and isinstance(array, wp.array):
                    result[f"{label}.{name}.{child}"] = array
    return result


def _trackable(array: wp.array) -> bool:
    return array.ptr is not None and array.size > 0 and array.is_contiguous


def rewind_arrays(arrays: dict[str, Any], saved: dict[str, np.ndarray], learned: set[str] | None, label: str) -> dict:
    """Write saved contents back into the arrays that steps write; keep the others.

    Args:
        arrays: Current arrays by name (``None`` or other objects when an attribute was replaced).
        saved: Host copies by name, as taken by a snapshot.
        learned: Qualified names (``label.name``) that steps have written; ``None`` rewinds every array.
        label: Prefix of the qualified names.

    Returns:
        ``rewound``: written arrays whose contents changed and were restored; ``kept``: arrays that
        differ from the snapshot although no step wrote them (cell edits), left as they are.
    """
    rewound, kept = [], []
    for name, data in saved.items():
        qualified = f"{label}.{name}"
        array = arrays.get(name)
        current = array.numpy() if isinstance(array, wp.array) else None
        if current is None or current.shape != data.shape or current.dtype != data.dtype:
            if learned is not None and qualified not in learned:
                kept.append(qualified)
            continue
        if current.tobytes() == data.tobytes():
            continue
        if learned is None or qualified in learned or not _trackable(array):
            array.assign(data)
            if learned is None or qualified in learned:
                rewound.append(qualified)
        else:
            kept.append(qualified)
    return {"rewound": rewound, "kept": kept}


class _DeviceTable:
    """Checksum buffers of the tracked arrays that live on one CUDA device."""

    def __init__(self, labels: list[str], arrays: list[wp.array], device):
        from .solverview import _hash_segments_kernel  # noqa: PLC0415

        self.kernel = _hash_segments_kernel
        self.labels = labels
        self.device = device
        item = [4 if array.capacity % 4 == 0 else 1 for array in arrays]
        self.pointers = wp.array([array.ptr for array in arrays], dtype=wp.uint64, device=device)
        self.counts = wp.array(
            [array.capacity // size for array, size in zip(arrays, item, strict=True)], dtype=wp.int32, device=device
        )
        self.item = wp.array(item, dtype=wp.int32, device=device)
        self.before = wp.zeros((len(arrays), 2), dtype=wp.uint32, device=device)
        self.after = wp.zeros((len(arrays), 2), dtype=wp.uint32, device=device)
        self.written = wp.zeros(len(arrays), dtype=wp.int32, device=device)
        self.dirty = False

    def hash(self, out: wp.array2d[wp.uint32]) -> None:
        wp.launch(
            self.kernel,
            dim=(len(self.labels), 512),
            inputs=[self.pointers, self.counts, self.item, out],
            device=self.device,
        )

    def read(self) -> set[str]:
        if not self.dirty:
            return set()
        flags = self.written.numpy()
        self.written.zero_()
        self.dirty = False
        return {label for label, flag in zip(self.labels, flags, strict=True) if flag}


class StepWrites:
    """Which public arrays of the registered sources the simulation's steps have written.

    Sources are named objects (``"control"``, ``"example"``) whose public Warp arrays are tracked
    as ``"<label>.<name>"``. Call :meth:`before` and :meth:`after` around each step (nested calls
    count once); :meth:`learned` returns the labels written by any step since :meth:`clear`.
    Arrays that cannot be checksummed (non-contiguous views) are reported by :meth:`untracked`.
    """

    def __init__(self):
        self._sources: dict[str, Callable[[], Any]] = {}
        self._learned: set[str] = set()
        self._depth = 0
        self._identity: tuple | None = None
        self._tables: list[_DeviceTable] = []
        self._host: list[tuple[str, wp.array]] = []
        self._host_before: list[int] | None = None
        self._active = False
        self.failed: str | None = None
        """Why learning stopped (reset and restore then rewind every array), or ``None``."""

    def add_source(self, label: str, getter: Callable[[], Any]) -> None:
        """Track the public arrays of ``getter()`` as ``label.<name>``."""
        self._sources[label] = getter
        self._identity = None

    def clear(self) -> None:
        """Forget what was learned, e.g. when the scene is replaced."""
        self._learned.clear()
        self._identity = None
        self._tables = []
        self._host = []
        self._host_before = None
        self._active = False
        self.failed = None

    def arrays(self) -> dict[str, wp.array]:
        """Current public arrays of all sources by label."""
        result = {}
        for label, getter in self._sources.items():
            try:
                owner = getter()
            except Exception:
                owner = None
            result.update(public_arrays(owner, label))
        return result

    def _collect(self) -> None:
        """Flush the pending flags and rebuild the checksum tables if the tracked arrays changed."""
        arrays = [
            (label, array) for label, array in self.arrays().items() if label not in self._learned and _trackable(array)
        ]
        identity = tuple((label, id(array), array.ptr, array.capacity) for label, array in arrays)
        if identity == self._identity:
            return
        self._flush()
        self._identity = identity
        self._tables = []
        self._host = []
        groups: dict[str, list[tuple[str, wp.array]]] = {}
        for label, array in arrays:
            if array.device.is_cuda:
                groups.setdefault(str(array.device), []).append((label, array))
            else:
                self._host.append((label, array))
        for items in groups.values():
            device = items[0][1].device
            self._tables.append(_DeviceTable([label for label, _ in items], [array for _, array in items], device))

    def _flush(self) -> None:
        for table in self._tables:
            self._learned |= table.read()

    def _fail(self, error: Exception) -> None:
        # Learning must never fail a step; reset then rewinds every array, as without learning.
        self.failed = f"{type(error).__name__}: {error}"
        self._tables, self._host, self._active = [], [], False

    def before(self) -> None:
        """Checksum the arrays not yet known to be written, before a step."""
        self._depth += 1
        if self._depth > 1 or self.failed or not self._sources:
            return
        self._active = False
        try:
            self._collect()
            # Launches inside an application's graph capture would be recorded, not run.
            if any(getattr(table.device, "is_capturing", False) for table in self._tables):
                return
            for table in self._tables:
                table.hash(table.before)
            self._host_before = [
                zlib.crc32(np.ascontiguousarray(array.numpy()).view(np.uint8)) for _, array in self._host
            ]
            self._active = True
        except Exception as error:
            self._fail(error)

    def after(self) -> None:
        """Mark the arrays whose checksums changed during the step as written."""
        self._depth = max(0, self._depth - 1)
        if self._depth or not self._active:
            return
        self._active = False
        try:
            for table in self._tables:
                table.hash(table.after)
                wp.launch(
                    _mark_written_kernel,
                    dim=len(table.labels),
                    inputs=[table.before, table.after, table.written],
                    device=table.device,
                )
                table.dirty = True
            for (label, array), before in zip(self._host, self._host_before or [], strict=False):
                if zlib.crc32(np.ascontiguousarray(array.numpy()).view(np.uint8)) != before:
                    self._learned.add(label)
                    # Arrays learned on the host side need no more checksums.
                    self._identity = None
            self._host_before = None
        except Exception as error:
            self._fail(error)

    def learned(self) -> set[str] | None:
        """Labels of the arrays a step has written since :meth:`clear`, or ``None`` if learning failed.

        Reads the pending device flags (one synchronization).
        """
        if self.failed:
            return None
        try:
            self._flush()
        except Exception as error:
            self._fail(error)
            return None
        # Learned arrays leave the tables at the next step.
        self._identity = None
        return set(self._learned)
