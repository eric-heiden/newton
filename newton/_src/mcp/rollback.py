# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Undo points that return a live session to its state before a failed operation.

An :class:`UndoPoint` copies the simulation's mutable arrays on their device
(state, control, model, and arrays the application registers) and remembers the
session bindings, time, and frame. If the operation fails, :meth:`UndoPoint.rollback`
rebinds replaced objects, writes back only the arrays whose contents changed,
notifies the solver about restored model fields, and resets solver caches when
the state moved.
"""

from __future__ import annotations

import functools
import sysconfig
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import warp as wp

from ..sim.enums import ModelFlags
from ..sim.model import Model

_BINDINGS = ("model", "solver", "state", "state_next", "control", "collision_pipeline", "contacts")
_MISSING = object()

_BUDGET = 256 * 1024 * 1024
"""Device bytes copied per group (state and application arrays; model arrays) for one undo point."""

_FIELD_FLAGS: dict[str, ModelFlags] = {
    "body_q": ModelFlags.BODY_PROPERTIES,
    "body_qd": ModelFlags.BODY_PROPERTIES,
    "body_flags": ModelFlags.BODY_PROPERTIES,
    "gravity": ModelFlags.MODEL_PROPERTIES,
}
_FREQUENCY_FLAGS: dict[str, int] = {
    "ONCE": ModelFlags.MODEL_PROPERTIES,
    "WORLD": ModelFlags.MODEL_PROPERTIES,
    "JOINT": ModelFlags.JOINT_PROPERTIES,
    "JOINT_COORD": ModelFlags.JOINT_PROPERTIES,
    "JOINT_DOF": ModelFlags.JOINT_DOF_PROPERTIES,
    "BODY": ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES,
    "SHAPE": ModelFlags.SHAPE_PROPERTIES,
    "CONSTRAINT_MIMIC": ModelFlags.CONSTRAINT_PROPERTIES,
}


def model_flags(model: Model, fields: list[str]) -> int:
    """Notification flags for restored model fields (``"model.<name>"`` or ``"model.<namespace>.<name>"``)."""
    flags = 0
    for qualified in fields:
        field = qualified.split(".", 1)[1]
        name = field.rsplit(".", 1)[-1]
        if name in _FIELD_FLAGS:
            flags |= _FIELD_FLAGS[name]
            continue
        try:
            frequency = model.get_attribute_frequency(field.replace(".", ":", 1) if "." in field else field)
        except (KeyError, AttributeError):
            frequency = None
        if isinstance(frequency, str):
            # Custom solver frequencies, e.g. "mujoco:actuator" or "mujoco:tendon".
            kind = frequency.lower()
            flags |= (
                ModelFlags.ACTUATOR_PROPERTIES
                if "actuator" in kind
                else ModelFlags.TENDON_PROPERTIES
                if "tendon" in kind
                else ModelFlags.CONSTRAINT_PROPERTIES
                if "eq" in kind
                else ModelFlags.SHAPE_PROPERTIES
                if "pair" in kind or "geom" in kind
                else ModelFlags.ALL
            )
        else:
            flags |= _FREQUENCY_FLAGS.get(getattr(frequency, "name", ""), ModelFlags.ALL)
    return int(flags)


class ArrayCopies:
    """Device copies of the public Warp arrays held by a few objects, restorable in place.

    Args:
        budget: Maximum bytes copied; objects beyond it keep only their array identities.
    """

    def __init__(self, budget: int = _BUDGET):
        self.budget = budget
        self.bytes = 0
        self.slots: list[tuple[str, Any, str, wp.array]] = []
        self.copies: dict[int, tuple[str, wp.array, wp.array]] = {}
        self.uncovered: list[str] = []

    def capture(self, label: str, owner: Any) -> None:
        """Copy the public array attributes of ``owner`` (and its attribute namespaces) as ``label.<name>``."""
        if owner is None:
            return
        items = []
        for name, value in vars(owner).items():
            if name.startswith("_"):
                continue
            if isinstance(value, wp.array):
                items.append((f"{label}.{name}", owner, name, value))
            elif isinstance(value, Model.AttributeNamespace):
                for child, array in vars(value).items():
                    if not child.startswith("_") and isinstance(array, wp.array):
                        items.append((f"{label}.{name}.{child}", value, child, array))
        self.slots.extend(items)
        fresh = {id(array): (qualified, array) for qualified, _, _, array in items if id(array) not in self.copies}
        fresh = {key: item for key, item in fresh.items() if item[1].ptr is not None and item[1].size}
        size = sum(array.capacity for _, array in fresh.values())
        if self.bytes + size > self.budget:
            self.uncovered.append(f"{label} arrays ({size / 2**20:.0f} MiB)")
            return
        self.bytes += size
        for key, (qualified, array) in fresh.items():
            self.copies[key] = (qualified, array, wp.clone(array, requires_grad=False))

    def restore(self) -> list[str]:
        """Rebind replaced arrays and write back changed contents.

        Returns:
            Qualified names of the restored arrays.
        """
        restored = []
        for qualified, owner, name, original in self.slots:
            if getattr(owner, name, _MISSING) is not original:
                setattr(owner, name, original)
                restored.append(qualified)
        for qualified, original, copy in self.copies.values():
            if original.numpy().tobytes() != copy.numpy().tobytes():
                original.assign(copy)
                if qualified not in restored:
                    restored.append(qualified)
        return restored


def _differs(current: Any, saved: Any) -> bool:
    if current is saved:
        return False
    scalar = (int, float, bool, str, type(None))
    return type(current) is not type(saved) or type(current) not in scalar or current != saved


def restore_attributes(current: dict, saved: dict, label: str) -> list[str]:
    """Return a namespace dictionary to its saved bindings (shallow) and list the names that changed."""
    changed = []
    for name in [name for name in current if name not in saved]:
        del current[name]
        changed.append(f"{label}.{name}")
    for name, value in saved.items():
        if _differs(current.get(name, _MISSING), value):
            current[name] = value
            changed.append(f"{label}.{name}")
    return changed


# Python creates these in a class's __dict__ on first read (e.g. typing.get_type_hints), not only on assignment.
_LAZY_CLASS_ATTRIBUTES = frozenset({"__annotations__", "__annotate__", "__annotate_func__", "__annotations_cache__"})


def restore_class_attributes(cls: type, saved: dict, label: str) -> list[str]:
    """Return a class to its saved attributes (shallow), e.g. a method a cell rebound, and list what changed.

    Attributes that cannot be set or deleted on ``cls`` are listed with ``(not restored)``. Annotation
    attributes that Python adds when they are first read are left in place.
    """
    changed = []
    current = vars(cls)
    for name in [name for name in current if name not in saved and name not in _LAZY_CLASS_ATTRIBUTES]:
        try:
            delattr(cls, name)
            changed.append(f"{label}.{name}")
        except (AttributeError, TypeError):
            changed.append(f"{label}.{name} (not restored)")
    for name, value in saved.items():
        if _differs(current.get(name, _MISSING), value):
            try:
                setattr(cls, name, value)
                changed.append(f"{label}.{name}")
            except (AttributeError, TypeError):
                changed.append(f"{label}.{name} (not restored)")
    return changed


class UndoPoint:
    """Everything needed to return ``session`` to the moment this object was created.

    The session's ``undo_callback(session, copies)`` may register application arrays with
    ``copies.capture(label, owner)`` and returns a function that restores the application's
    Python attributes and returns the qualified names it changed.
    """

    def __init__(self, session, *, budget: int = _BUDGET):
        self.session = session
        self.generation = session._scene_generation
        self.time, self.frame = session.time, session.frame
        self.bindings = {name: getattr(session, name) for name in _BINDINGS}
        self.state = ArrayCopies(budget)
        self.state.capture("state", session.state)
        self.state.capture("control", session.control)
        self.model = ArrayCopies(budget)
        self.model.capture("model", session.model)
        callback = getattr(session, "undo_callback", None)
        self.application: Callable[[], list[str]] | None = callback(session, self.state) if callback else None

    @property
    def uncovered(self) -> list[str]:
        """Array groups too large to copy; their contents are not rolled back."""
        return self.state.uncovered + self.model.uncovered

    def rollback(self) -> dict:
        """Restore the session; returns what changed (``restored`` names, ``time``/``frame`` moves, flags)."""
        session = self.session
        if session._scene_generation != self.generation:
            return {"replaced": True}
        restored = self.application() if self.application is not None else []
        bindings = [name for name, value in self.bindings.items() if getattr(session, name) is not value]
        for name in bindings:
            setattr(session, name, self.bindings[name])
        state_fields = self.state.restore()
        model_fields = self.model.restore()
        flags = 0
        if model_fields:
            flags = model_flags(session.model, model_fields)
            session.solver.notify_model_changed(flags)
        moved = (session.time, session.frame) != (self.time, self.frame)
        report = {
            "restored": restored + [name for name in state_fields if name not in restored] + model_fields,
            "bindings": bindings,
            "time": (session.time, self.time),
            "frame": (session.frame, self.frame),
            "moved": moved,
            "flags": flags,
        }
        # Solver caches only need a reset when the state itself was rewritten or rebound.
        if moved or bindings or any(name.startswith(("state.", "control.")) for name in state_fields):
            session._resync(self.time, self.frame)
        return report


def summarize(report: dict, uncovered: list[str]) -> str:
    """One-paragraph description of a rollback report."""
    if report.get("replaced"):
        return "The scene was rebuilt during this call, so it was not rolled back."
    names = report["restored"] or [f"session.{name}" for name in report["bindings"]]
    if not names and not report["moved"]:
        text = "No simulation data changed (state, control, model arrays, application attributes), so nothing was restored."
    else:
        groups: dict[str, list[str]] = {}
        for name in names:
            root, _, rest = name.partition(".")
            groups.setdefault(root, []).append(rest)
        parts = []
        for root, fields in groups.items():
            shown = ", ".join(fields[:8]) + (f", +{len(fields) - 8} more" if len(fields) > 8 else "")
            parts.append(f"{root} ({shown})")
        now, before = report["time"]
        frame_now, frame_before = report["frame"]
        moved = f" from t={now:.6g} s (frame {frame_now})" if report["moved"] else ""
        text = f"The simulation was rolled back{moved} to t={before:.6g} s (frame {frame_before})"
        text += f"; restored {', '.join(parts)}." if parts else "."
        if report["flags"]:
            flags = [flag.name for flag in ModelFlags if flag != ModelFlags.ALL and report["flags"] & flag]
            text += f" Restored model fields were notified to the solver ({'|'.join(flags)})."
    if uncovered:
        text += f" Not covered by rollback: {', '.join(uncovered)}."
    return text


@functools.cache
def _library_roots() -> tuple[str, ...]:
    import newton  # noqa: PLC0415

    roots = {str(Path(newton.__file__).parent / "_src"), str(Path(wp.__file__).parent)}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(path)
    return tuple(sorted(roots))


def user_frame(filename: str) -> bool:
    """Whether a traceback frame belongs to user code (scripts, cells) rather than Newton, Warp, or installed packages."""
    return not filename.startswith(_library_roots())


def describe_exception(error: BaseException, *, limit: int = 8) -> str:
    """``Type: message`` plus the user-code frames of the traceback as ``file:line in function: source``."""
    lines = [f"{type(error).__name__}: {str(error)[:4096]}"]
    if isinstance(error, SyntaxError) and error.filename:
        lines.append(f"  {error.filename}:{error.lineno}: {(error.text or '').strip()[:200]}")
    frames = traceback.extract_tb(error.__traceback__)
    for frame in [frame for frame in frames if user_frame(frame.filename)][-limit:]:
        lines.append(f"  {frame.filename}:{frame.lineno} in {frame.name}: {(frame.line or '').strip()[:200]}")
    if frames and not user_frame(frames[-1].filename) and "/_src/mcp/" not in frames[-1].filename:
        last = frames[-1]
        lines.append(f"  (raised in {last.filename}:{last.lineno} in {last.name})")
    return "\n".join(lines)
