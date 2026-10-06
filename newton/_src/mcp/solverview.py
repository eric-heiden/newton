# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""What a solver integrates, and which live model edits reached it.

:func:`solver_params` reports the values a solver uses next to the Newton model
arrays they come from and the :class:`~newton.ModelFlags` category that
refreshes them. :class:`ModelWatch` checksums the model arrays solvers read so
the host can notice edits that no ``notify_model_changed`` call covered, and
state which edited fields the current solver configuration does not read.
"""

from __future__ import annotations

import contextlib
import fnmatch
import functools
import time
import weakref
import zlib
from typing import Any

import numpy as np
import warp as wp

from ..sim.enums import BodyFlags, JointTargetMode, ModelFlags, _covered_model_flags
from ..solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from ..solvers.mujoco.solver_mujoco import _MJW_BATCHED_MODEL_FIELDS, _MJW_BATCHED_OPTION_FIELDS, SolverMuJoCo

_F = ModelFlags

FIELD_FLAGS: dict[str, ModelFlags] = {
    **dict.fromkeys(("joint_X_p", "joint_X_c", "joint_axis"), _F.JOINT_PROPERTIES),
    "joint_velocity_limit": _F.JOINT_DOF_PROPERTIES,
    **dict.fromkeys(
        (
            "joint_target_ke",
            "joint_target_kd",
            "joint_target_mode",
            "joint_damping",
            "joint_friction",
            "joint_effort_limit",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_limit_lower",
            "joint_limit_upper",
            "mujoco.solimplimit",
            "mujoco.solreflimit",
            "mujoco.solreflimit_mode",
            "mujoco.limit_margin",
            "mujoco.dof_passive_stiffness",
            "mujoco.solreffriction",
            "mujoco.solimpfriction",
        ),
        _F.JOINT_DOF_FORCE_PROPERTIES,
    ),
    "joint_armature": _F.JOINT_DOF_INERTIAL_PROPERTIES,
    **dict.fromkeys(("mujoco.dof_ref", "mujoco.dof_springref"), _F.JOINT_REFERENCE_POSE_PROPERTIES),
    "body_flags": _F.BODY_PROPERTIES,
    **dict.fromkeys(
        ("body_mass", "body_inv_mass", "body_com", "body_inertia", "body_inv_inertia", "mujoco.gravcomp"),
        _F.BODY_INERTIAL_PROPERTIES,
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
            "shape_transform",
            "shape_scale",
            "shape_margin",
            "shape_gap",
            "mujoco.geom_solimp",
            "mujoco.geom_solmix",
            "mujoco.solref",
            "mujoco.solref_mode",
            "mujoco.pair_solref",
            "mujoco.pair_solreffriction",
            "mujoco.pair_solimp",
            "mujoco.pair_margin",
            "mujoco.pair_gap",
            "mujoco.pair_friction",
        ),
        _F.SHAPE_PROPERTIES,
    ),
    "gravity": _F.MODEL_PROPERTIES,
    **dict.fromkeys(
        (
            "joint_mimic_coeffs",
            "constraint_mimic_coef0",
            "constraint_mimic_coef1",
            "constraint_mimic_enabled",
            "mujoco.eq_solref",
            "mujoco.eq_solimp",
            "mujoco.equality_constraint_anchor",
            "mujoco.equality_constraint_relpose",
            "mujoco.equality_constraint_polycoef",
            "mujoco.equality_constraint_torquescale",
            "mujoco.equality_constraint_enabled",
        ),
        _F.CONSTRAINT_PROPERTIES,
    ),
    **dict.fromkeys(
        (
            "mujoco.tendon_stiffness",
            "mujoco.tendon_damping",
            "mujoco.tendon_frictionloss",
            "mujoco.tendon_range",
            "mujoco.tendon_margin",
            "mujoco.tendon_solref_limit",
            "mujoco.tendon_solimp_limit",
            "mujoco.tendon_solref_friction",
            "mujoco.tendon_solimp_friction",
            "mujoco.tendon_armature",
            "mujoco.tendon_actuator_force_range",
        ),
        _F.TENDON_PROPERTIES,
    ),
    **dict.fromkeys(
        (
            "mujoco.actuator_gainprm",
            "mujoco.actuator_biasprm",
            "mujoco.actuator_dynprm",
            "mujoco.actuator_ctrlrange",
            "mujoco.actuator_forcerange",
            "mujoco.actuator_actrange",
            "mujoco.actuator_gear",
            "mujoco.actuator_cranklength",
        ),
        _F.ACTUATOR_PROPERTIES,
    ),
}
"""Model fields that solvers read, mapped to the :class:`~newton.ModelFlags` category that refreshes them."""

MUJOCO_CONSTRUCTION_ONLY: frozenset[str] = frozenset(
    {
        "joint_target_mode",
        "mujoco.condim",
        "mujoco.geom_priority",
        "mujoco.contype",
        "mujoco.conaffinity",
        "mujoco.pair_condim",
        "mujoco.ctrl_source",
        "mujoco.ctrl_type",
        "mujoco.actuator_trntype",
        "mujoco.actuator_trnid",
        "mujoco.actuator_gaintype",
        "mujoco.actuator_biastype",
        "mujoco.actuator_dyntype",
        "mujoco.jnt_actgravcomp",
        "mujoco.tendon_springlength",
        *(
            f"mujoco.{name}"
            for name in (
                "iterations",
                "ls_iterations",
                "ccd_iterations",
                "sdf_iterations",
                "sdf_initpoints",
                "solver",
                "integrator",
                "cone",
                "jacobian",
                "impratio",
                "tolerance",
                "ls_tolerance",
                "ccd_tolerance",
                "sleep_tolerance",
                "density",
                "viscosity",
                "wind",
                "magnetic",
            )
        ),
    }
)
"""Fields :class:`~newton.solvers.SolverMuJoCo` reads only when it is constructed; no ``ModelFlags`` refreshes them."""

MUJOCO_NOT_READ: dict[str, str] = {
    "joint_velocity_limit": "SolverMuJoCo does not read joint_velocity_limit",
    "shape_material_restitution": "SolverMuJoCo does not read shape_material_restitution",
    "body_inv_mass": "SolverMuJoCo reads body_mass, not body_inv_mass",
    "body_inv_inertia": "SolverMuJoCo reads body_inertia, not body_inv_inertia",
}
"""Fields :class:`~newton.solvers.SolverMuJoCo` never reads, with the fact to report."""

WATCHED_FIELDS: tuple[str, ...] = tuple(sorted({*FIELD_FLAGS, *MUJOCO_CONSTRUCTION_ONLY}))

_SOLREF_MODES = {
    SOLREF_MODE_FORCE_SPACE: "FORCE_SPACE",
    SOLREF_MODE_RAW: "RAW",
    SOLREF_MODE_MJCF_DEFAULT: "MJCF_DEFAULT",
}
_JOINT_TARGET = int(SolverMuJoCo.CtrlSource.JOINT_TARGET)
_ACTUATOR_DIRECT_FIELDS = (
    "mujoco.actuator_gainprm",
    "mujoco.actuator_biasprm",
    "mujoco.actuator_dynprm",
    "mujoco.actuator_forcerange",
    "mujoco.actuator_actrange",
    "mujoco.actuator_gear",
    "mujoco.actuator_cranklength",
)


def flag_names(flags: int) -> str:
    """``"JOINT_DOF_PROPERTIES | SHAPE_PROPERTIES"`` for a flag mask."""
    names = [flag.name for flag in ModelFlags if flag != ModelFlags.ALL and int(flags) & int(flag)]
    return " | ".join(names) if names else "0"


def flag_expression(flags: int) -> str:
    """``"newton.ModelFlags.A | newton.ModelFlags.B"`` for a flag mask."""
    return " | ".join(f"newton.ModelFlags.{name}" for name in flag_names(flags).split(" | "))


def inferred_flags(fields) -> int:
    """Union of the :class:`~newton.ModelFlags` categories of the given model fields."""
    flags = 0
    for field in fields:
        flags |= int(FIELD_FLAGS.get(field, 0))
    return flags


def model_field(model, name: str):
    """Model array for ``"joint_target_ke"`` or ``"mujoco.solref"``, or ``None``."""
    obj = model
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_mujoco(solver) -> bool:
    return isinstance(solver, SolverMuJoCo) and getattr(solver, "mjw_model", None) is not None


def _rows_text(rows, limit: int = 8) -> str:
    """``"1-6, 8, 10-13"`` for row indices, with at most ``limit`` runs."""
    rows = sorted({int(r) for r in rows})
    runs = []
    for row in rows:
        if runs and row == runs[-1][1] + 1:
            runs[-1][1] = row
        else:
            runs.append([row, row])
    text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs[:limit])
    return text + (f", ... ({len(rows)} rows)" if len(runs) > limit else "")


# ---------------------------------------------------------------------------------------------------------------------
# notify_model_changed recording


_WATCHES: weakref.WeakSet = weakref.WeakSet()


def _record_notify(solver, flags) -> None:
    try:
        value = int(flags)
    except (TypeError, ValueError):
        return
    for watch in list(_WATCHES):
        watch.record(solver, value)


def _wrap_notify(cls) -> None:
    """Make ``cls.notify_model_changed`` record its solver and flags for active watches (idempotent)."""
    method = getattr(cls, "notify_model_changed", None)
    if method is None or getattr(method, "_newton_mcp_records", False):
        return

    @functools.wraps(method)
    def notify_model_changed(self, flags, *args, **kwargs):
        result = method(self, flags, *args, **kwargs)
        # Recorded after the call: notify_model_changed may rewrite model arrays itself.
        _record_notify(self, flags)
        return result

    notify_model_changed._newton_mcp_records = True
    cls.notify_model_changed = notify_model_changed


def _wrap_solver_classes(solver=None) -> None:
    from ..solvers.solver import SolverBase  # noqa: PLC0415

    pending, seen = [SolverBase], set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if "notify_model_changed" in cls.__dict__:
            _wrap_notify(cls)
        pending.extend(cls.__subclasses__())
    if solver is not None and hasattr(type(solver), "notify_model_changed"):
        _wrap_notify(type(solver))


# ---------------------------------------------------------------------------------------------------------------------
# Checksums


@wp.kernel
def _hash_segments_kernel(
    pointers: wp.array[wp.uint64],
    counts: wp.array[wp.int32],
    item_bytes: wp.array[wp.int32],
    out: wp.array2d[wp.uint32],
):
    segment, lane = wp.tid()
    count = counts[segment]
    lanes = wp.int32(512)
    h0 = wp.uint32(0)
    h1 = wp.uint32(0)
    i = lane
    if item_bytes[segment] == 4:
        words = wp.array(ptr=pointers[segment], shape=(count,), dtype=wp.uint32)
        while i < count:
            value = words[i]
            x = (value ^ (wp.uint32(i) * wp.uint32(0x9E3779B9))) * wp.uint32(0x85EBCA6B)
            h0 = h0 + (x ^ (x >> wp.uint32(13)))
            h1 = h1 + (value + wp.uint32(i)) * wp.uint32(0xC2B2AE35)
            i += lanes
    else:
        data = wp.array(ptr=pointers[segment], shape=(count,), dtype=wp.uint8)
        while i < count:
            value = wp.uint32(data[i])
            x = (value ^ (wp.uint32(i) * wp.uint32(0x9E3779B9))) * wp.uint32(0x85EBCA6B)
            h0 = h0 + (x ^ (x >> wp.uint32(13)))
            h1 = h1 + (value + wp.uint32(i)) * wp.uint32(0xC2B2AE35)
            i += lanes
    wp.atomic_add(out, segment, 0, h0)
    wp.atomic_add(out, segment, 1, h1)


def _host_rows(array: wp.array) -> np.ndarray:
    """Raw bytes of an array as ``[rows, bytes per row]`` (``NaN`` payloads compare exactly)."""
    values = np.ascontiguousarray(array.numpy())
    rows = values.shape[0] if values.ndim else 1
    return values.view(np.uint8).reshape(rows, -1) if values.size else np.zeros((rows, 0), np.uint8)


class ModelWatch:
    """Detect edits to the model arrays solvers read, between and inside trusted-execution cells.

    The session calls :meth:`begin` before a cell runs, wraps stepping operations in
    :meth:`stepping`, and calls :meth:`end` after the cell. Each check compares device-side
    checksums of the watched arrays with the previous baseline. A changed array counts as
    applied when ``notify_model_changed`` was called on the session's solver with its
    :class:`~newton.ModelFlags` category after the array's last change; other changed arrays
    are notified (``mode="notify"``) or only reported (``mode="report"``). ``mode="off"``
    disables checks.

    Args:
        session: Owning :class:`SimulationSession`.
    """

    _MODES = ("notify", "report", "off")
    copy_budget = 64 * 1024 * 1024
    """Bytes of host copies kept to name changed rows; larger arrays are reported without rows."""

    def __init__(self, session):
        self._session = weakref.ref(session)
        self._mode = "notify"
        self.notified = 0
        """Flags passed to ``notify_model_changed`` of the session's solver since the last baseline."""
        self.timings: list[float] = []
        """Wall time [s] of recent checksum passes (bounded), for overhead measurements."""
        self._model = None
        self._fields: list[tuple[str, wp.array]] = []
        self._segments = None
        self._baseline: dict | None = None
        self._pending = None
        self._revision = None
        self._copies: dict[str, np.ndarray] = {}
        self._notifications: list[tuple[Any, int, dict | None]] = []
        self._notifying = False
        self._depth = 0
        self.active = False
        self._notes: list[str] = []
        self._facts: set[str] = set()
        """Facts about edited fields already reported for the current model; each is reported once."""
        _WATCHES.add(self)

    @property
    def mode(self) -> str:
        """``"notify"`` (default), ``"report"``, or ``"off"``."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in self._MODES:
            raise ValueError(f"watch mode must be one of {self._MODES}")
        self._mode = value
        self._baseline = None
        self._pending = None

    def reset(self) -> None:
        """Forget the baseline (scene replaced); inside a cell, start a new one so later edits in it are checked."""
        self._model = None
        self._baseline = None
        self._pending = None
        self._copies.clear()
        self._segments = None
        if self.active:
            # E.g. persist() or a rebuild dispatched from a cell.
            self._guard(self._start)

    def settle(self) -> None:
        """Keep the baseline valid across a revision change that did not touch the model (the end of a cell)."""
        session = self._session()
        if session is not None and self._baseline is not None:
            self._revision = session.revision

    def record(self, solver, flags: int) -> None:
        """Remember a ``notify_model_changed(flags)`` call on ``solver`` made while a cell runs."""
        if not self.active or self._notifying or self._baseline is None:
            return
        session = self._session()
        if session is None:
            return
        if solver is session.solver:
            self.notified |= int(flags)
        digests = None
        if solver is session.solver or getattr(solver, "model", None) is session.model:
            # Checksums at the call: an array edited after it was not covered by it.
            with contextlib.suppress(Exception):
                digests = self._digests()
        self._notifications.append((solver, int(flags), digests))

    # -- checksums --------------------------------------------------------------------------------------------------

    def _discover(self, model) -> None:
        if model is not self._model:
            self._facts.clear()
        self._model = model
        fields = []
        for name in WATCHED_FIELDS:
            array = model_field(model, name)
            if isinstance(array, wp.array) and array.size and array.is_contiguous:
                fields.append((name, array))
        self._fields = fields
        self._segments = None
        self._copies.clear()

    def _identity(self) -> tuple:
        model = self._model
        current = []
        for name, _ in self._fields:
            live = model_field(model, name)
            current.append((name, id(live), getattr(live, "ptr", None), getattr(live, "capacity", None)))
        return tuple(current)

    def _launch(self) -> tuple:
        """Start a checksum pass over the watched arrays; :meth:`_read` returns its digests."""
        self._resolve_pending()
        model = self._session().model
        if model is not self._model:
            self._discover(model)
        identity = self._identity()
        if self._segments is None or self._segments[0] != identity:
            # Arrays were replaced (or first use): rebuild the list and the device pointer table.
            fields = []
            for name, *_ in identity:
                array = model_field(model, name)
                if isinstance(array, wp.array) and array.size and array.is_contiguous:
                    fields.append((name, array))
            self._fields = fields
            identity = self._identity()
            device = model.device
            table = None
            if device.is_cuda and fields:
                item = [4 if array.capacity % 4 == 0 else 1 for _, array in fields]
                table = (
                    wp.array([array.ptr for _, array in fields], dtype=wp.uint64, device=device),
                    wp.array(
                        [array.capacity // size for (_, array), size in zip(fields, item, strict=True)],
                        dtype=wp.int32,
                        device=device,
                    ),
                    wp.array(item, dtype=wp.int32, device=device),
                    wp.zeros((len(fields), 2), dtype=wp.uint32, device=device),
                )
            self._segments = (identity, table)
        identity, table = self._segments
        if table is None:
            return identity, [
                zlib.crc32(np.ascontiguousarray(array.numpy()).view(np.uint8)) for _, array in self._fields
            ]
        pointers, counts, item, out = table
        out.zero_()
        wp.launch(
            _hash_segments_kernel, dim=(len(self._fields), 512), inputs=[pointers, counts, item, out], device=out.device
        )
        return identity, out

    @staticmethod
    def _read(handle: tuple) -> dict[str, tuple]:
        identity, hashes = handle
        if isinstance(hashes, wp.array):
            hashes = [tuple(row) for row in hashes.numpy().tolist()]
        return {name: (key, digest) for (name, *key), digest in zip(identity, hashes, strict=True)}

    def _digests(self) -> dict[str, tuple]:
        started = time.perf_counter()
        digests = self._read(self._launch())
        self.timings = [*self.timings[-63:], time.perf_counter() - started]
        return digests

    def _resolve_pending(self) -> None:
        """Make the checksums launched after the last stepping operation the baseline."""
        if self._pending is None:
            return
        handle, self._pending = self._pending, None
        digests = self._read(handle)
        if self._baseline is not None:
            for name, value in digests.items():
                if self._baseline.get(name) != value:
                    # The application's own step changed it; its rows are not known for the next report.
                    self._copies.pop(name, None)
        self._baseline = digests

    def _refresh_copy(self, name: str) -> np.ndarray | None:
        """Changed rows of ``name`` against its host copy, updating the copy (``None`` if unknown)."""
        array = model_field(self._model, name)
        if not isinstance(array, wp.array):
            self._copies.pop(name, None)
            return None
        old = self._copies.get(name)
        used = sum(copy.nbytes for key, copy in self._copies.items() if key != name)
        if used + array.capacity > self.copy_budget:
            self._copies.pop(name, None)
            return None
        new = _host_rows(array)
        self._copies[name] = new.copy()
        if old is None or old.shape != new.shape:
            return None
        return np.flatnonzero(np.any(old != new, axis=1))

    def _set_baseline(self, digests: dict, *, changed=()) -> None:
        if self._baseline is None:
            # First baseline: keep host copies so later checks can name the changed rows.
            for name, _ in self._fields:
                self._refresh_copy(name)
        else:
            for name in changed:
                self._refresh_copy(name)
        self._baseline = digests
        self._revision = self._session().revision
        self._forget_notifications()

    def _forget_notifications(self) -> None:
        self.notified = 0
        self._notifications.clear()

    def _changes(self) -> tuple[dict, list[str]]:
        digests = self._digests()
        changed = [name for name, value in digests.items() if self._baseline.get(name) != value]
        return digests, changed

    # -- session hooks ----------------------------------------------------------------------------------------------

    def _guard(self, action, *args) -> None:
        """Run a check; a failure disables checks for the rest of the cell instead of failing the cell."""
        try:
            action(*args)
        except Exception as error:
            self.active = False
            self._baseline = None
            self._pending = None
            self._notes.append(f"Model-edit checks stopped for this cell: {type(error).__name__}: {error}")

    def _start(self) -> None:
        session = self._session()
        _wrap_solver_classes(session.solver)
        self._resolve_pending()
        if self._baseline is None or self._model is not session.model:
            self._baseline = None
            self._set_baseline(self._digests())
        elif self._revision != session.revision:
            digests, changed = self._changes()
            self._set_baseline(digests, changed=changed)
        self._forget_notifications()

    def begin(self) -> None:
        """Start a cell: refresh the baseline if operations since the last cell may have edited the model."""
        self._notes = []
        self.active = self._mode != "off"
        if self.active:
            self._guard(self._start)

    def end(self) -> str | None:
        """Finish a cell: check for edits made after the last stepping operation and return the notes."""
        if self.active and self._depth == 0:
            self._guard(self._check, "in this cell")
        if self.active:
            # Nothing else runs before the next cell, so the baseline stays valid across this revision.
            self._revision = self._session().revision
        self.active = False
        self._notifications.clear()
        notes, self._notes = self._notes, []
        return "\n".join(notes) or None

    def abort(self) -> None:
        """Finish a cell whose failure invalidated the scene; the next cell starts from a fresh baseline."""
        self.active = False
        self._baseline = None
        self._pending = None
        self._notifications.clear()
        self._notes = []

    @contextlib.contextmanager
    def stepping(self, operation: str):
        """Check before a stepping operation inside a cell and re-baseline after it.

        Re-baselining after the operation keeps model edits made by the application's own
        ``step()`` from being attributed to the cell's code.
        """
        outermost = self.active and self._depth == 0
        if outermost:
            self._guard(self._check, f"before {operation}")
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if outermost and self.active:
                self._guard(self._rebaseline)

    def _rebaseline(self) -> None:
        if self._baseline is None:
            self._start()
            return
        # Read at the next check, which waits for the device anyway; this keeps stepping asynchronous.
        self._pending = self._launch()
        self._revision = self._session().revision
        self._forget_notifications()

    def _covered(self, name: str, digest: tuple, solver) -> bool:
        """Whether a notify_model_changed call on ``solver`` covered ``name`` after its last change."""
        flag = int(FIELD_FLAGS[name])
        return any(
            other is solver
            and _covered_model_flags(flags) & flag
            and (recorded is None or recorded.get(name) == digest)
            for other, flags, recorded in self._notifications
        )

    def _uncounted(self, names: list[str], digests: dict, solver) -> str:
        """Why notify_model_changed calls with the right flags did not cover ``names``."""
        reasons = []
        for name in names:
            flag = int(FIELD_FLAGS[name])
            for other, flags, recorded in self._notifications:
                if not _covered_model_flags(flags) & flag:
                    continue
                if other is not solver:
                    reason = f"a call on a {type(other).__name__} that is not the session's solver"
                elif recorded is not None and recorded.get(name) != digests[name]:
                    reason = f"a call before the last edit of model.{name}"
                else:
                    continue
                if reason not in reasons:
                    reasons.append(reason)
        # The flags used are only news when some missing category was never passed to the session's solver.
        missing = inferred_flags(names)
        own = flag_names(self.notified) if self.notified and missing & ~_covered_model_flags(self.notified) else None
        parts = ([f"calls in this interval used {own}"] if own else []) + reasons[:4]
        return f" ({'; '.join(parts)})" if parts else ""

    def _check(self, where: str) -> None:
        session = self._session()
        if session.sync_callback is not None:
            session.sync_callback(session)
        if self._baseline is None and self._pending is None:
            self._start()
            return
        digests, changed = self._changes()
        if not changed:
            self._forget_notifications()
            return
        rows = {name: self._refresh_copy(name) for name in changed}
        solver = session.solver
        mujoco = _is_mujoco(solver)
        flagged = [n for n in changed if n in FIELD_FLAGS and not (mujoco and n in MUJOCO_CONSTRUCTION_ONLY)]
        uncovered = [name for name in flagged if not self._covered(name, digests[name], solver)]
        missing = inferred_flags(uncovered)
        # Edits a notify_model_changed call covered are not reported: only facts the cell cannot see itself.
        lines = []
        if missing:
            described = []
            for name in uncovered:
                if self._baseline.get(name, (None,))[0] != digests[name][0]:
                    size = "new array object"
                else:
                    size = f"{len(rows[name])} rows" if rows[name] is not None else "rows not tracked"
                described.append(f"model.{name} [{size}]")
            line = (
                f"{', '.join(described)} changed {where}; no notify_model_changed call covered "
                f"{flag_names(missing)}{self._uncounted(uncovered, digests, solver)}; "
            )
            if self._mode == "notify":
                self._notifying = True
                try:
                    solver.notify_model_changed(missing)
                    line += f"the host called solver.notify_model_changed({flag_expression(missing)})."
                except Exception as error:  # the cell's edits stay; report instead of failing the cell
                    line += f"the host's notify_model_changed raised {type(error).__name__}: {error}"
                finally:
                    self._notifying = False
                # notify_model_changed may rewrite model arrays itself (e.g. MuJoCo solref modes).
                notified = self._digests()
                for name, value in notified.items():
                    if digests.get(name) != value and name not in rows:
                        self._refresh_copy(name)
                digests = notified
            else:
                line += f"the solver keeps its previous values until notify_model_changed({flag_expression(missing)})."
            lines.append(line)
        if mujoco:
            facts = [fact for fact in mujoco_change_facts(session.model, solver, rows) if fact not in self._facts]
            self._facts.update(facts)
            lines += facts
        for line in lines:
            if line not in self._notes:
                self._notes.append(line)
        self._baseline = digests
        self._revision = session.revision
        self._forget_notifications()


# ---------------------------------------------------------------------------------------------------------------------
# MuJoCo facts about edited fields


def _mujoco_actuator_maps(solver) -> dict | None:
    if solver.mjc_actuator_ctrl_source is None:
        return None
    return {
        "source": solver.mjc_actuator_ctrl_source.numpy(),
        "newton": solver.mjc_actuator_to_newton_idx.numpy(),
        "custom": (
            solver.mjc_actuator_to_newton_actuator_idx.numpy()
            if solver.mjc_actuator_to_newton_actuator_idx is not None
            else None
        ),
        "joint_effort": (
            solver._actuator_uses_joint_effort_limit.numpy()
            if getattr(solver, "_actuator_uses_joint_effort_limit", None) is not None
            else None
        ),
    }


def mujoco_change_facts(model, solver, rows: dict[str, np.ndarray | None]) -> list[str]:
    """Facts about edited fields that :class:`~newton.solvers.SolverMuJoCo` does not read as configured.

    Args:
        model: Model the solver was built from.
        solver: :class:`~newton.solvers.SolverMuJoCo` instance.
        rows: Edited field name (``"mujoco.solref"``) to changed row indices, or ``None`` when unknown.

    Returns:
        One sentence per fact.
    """
    facts = []
    world_count = max(1, int(model.world_count))
    for name in rows:
        if name in MUJOCO_NOT_READ:
            facts.append(MUJOCO_NOT_READ[name] + ".")
        elif name == "joint_target_mode":
            facts.append(
                "model.joint_target_mode decides which MuJoCo actuators exist when SolverMuJoCo is constructed; "
                "afterwards JOINT_DOF_FORCE_PROPERTIES only reads world 0's mode to decide whether a position actuator "
                "also takes joint_target_kd."
            )
        elif name in MUJOCO_CONSTRUCTION_ONLY:
            facts.append(f"model.{name} is read when SolverMuJoCo is constructed; no ModelFlags refreshes it.")
        elif name == "shape_material_kf" and getattr(solver, "_use_mujoco_contacts", True):
            facts.append(
                "shape_material_kf is read only with use_mujoco_contacts=False; this solver uses MuJoCo contacts."
            )

    def all_rows(name: str) -> np.ndarray:
        changed = rows.get(name)
        if changed is not None:
            return changed
        array = model_field(model, name)
        return np.arange(array.shape[0]) if array is not None else np.zeros(0, dtype=int)

    maps = _mujoco_actuator_maps(solver)
    actuator_fields = [name for name in _ACTUATOR_DIRECT_FIELDS if name in rows]
    if actuator_fields and maps is not None and maps["custom"] is not None:
        total = int(model.custom_frequency_counts.get("mujoco:actuator", 0))
        per_world = total // world_count if total % world_count == 0 else total
        sources = {}
        for actuator, custom in enumerate(maps["custom"]):
            if custom >= 0:
                sources.setdefault(int(custom), set()).add(int(maps["source"][actuator]))
        for name in actuator_fields:
            changed = all_rows(name)
            template = changed % per_world if per_world else changed
            joint_target = [
                int(r) for r, t in zip(changed, template, strict=True) if sources.get(int(t)) == {_JOINT_TARGET}
            ]
            unmapped = [int(r) for r, t in zip(changed, template, strict=True) if int(t) not in sources]
            if joint_target:
                facts.append(
                    f"model.{name} rows {_rows_text(joint_target)} belong to actuators with ctrl_source JOINT_TARGET, "
                    "whose gain and bias come from model.joint_target_ke/kd (JOINT_DOF_FORCE_PROPERTIES); "
                    "notify_model_changed copies only actuator_ctrlrange of these rows into MuJoCo."
                )
            if unmapped:
                facts.append(f"model.{name} rows {_rows_text(unmapped)} have no MuJoCo actuator.")
    for name in ("joint_target_ke", "joint_target_kd"):
        if name not in rows:
            continue
        if maps is None:
            facts.append(
                f"model.{name} rows {_rows_text(all_rows(name))} drive no MuJoCo actuator (this SolverMuJoCo has no "
                "actuators: joint_target_mode was NONE or EFFORT for every DOF at construction)."
            )
            continue
        dofs_per_world = model.joint_dof_count // world_count
        driven = set()
        for source, index in zip(maps["source"], maps["newton"], strict=True):
            if source == _JOINT_TARGET and index != -1:
                driven.add(int(index) if index >= 0 else -(int(index) + 2))
        changed = all_rows(name)
        idle = [int(r) for r in changed if int(r) % max(dofs_per_world, 1) not in driven]
        if idle:
            facts.append(
                f"model.{name} rows {_rows_text(idle)} drive no MuJoCo actuator (joint_target_mode NONE or EFFORT "
                "at construction, or the DOF is driven by a CTRL_DIRECT actuator)."
            )
    mujoco_attrs = getattr(model, "mujoco", None)
    shape_mode = getattr(mujoco_attrs, "solref_mode", None)
    if shape_mode is not None:
        mode = shape_mode.numpy()
        if "mujoco.solref" in rows:
            changed = all_rows("mujoco.solref")
            ignored = changed[mode[changed] != SOLREF_MODE_RAW]
            if len(ignored):
                facts.append(
                    f"model.mujoco.solref rows {_rows_text(ignored)} are shapes whose mujoco.solref_mode is not RAW; "
                    "their geom solref comes from shape_material_ke/kd."
                )
        for name in ("shape_material_ke", "shape_material_kd"):
            if name in rows:
                changed = all_rows(name)
                ignored = changed[mode[changed] == SOLREF_MODE_RAW]
                if len(ignored):
                    facts.append(
                        f"model.{name} rows {_rows_text(ignored)} are shapes with mujoco.solref_mode RAW; their geom "
                        "solref comes from model.mujoco.solref."
                    )
    limit_mode = getattr(mujoco_attrs, "solreflimit_mode", None)
    if limit_mode is not None:
        mode = limit_mode.numpy()
        if "mujoco.solreflimit" in rows:
            changed = all_rows("mujoco.solreflimit")
            ignored = changed[mode[changed] != SOLREF_MODE_RAW]
            if len(ignored):
                facts.append(
                    f"model.mujoco.solreflimit rows {_rows_text(ignored)} are DOFs whose mujoco.solreflimit_mode is "
                    "not RAW; their limit solref comes from joint_limit_ke/kd (FORCE_SPACE) or MuJoCo's default "
                    "(MJCF_DEFAULT)."
                )
        for name in ("joint_limit_ke", "joint_limit_kd"):
            if name in rows:
                changed = all_rows(name)
                ignored = changed[mode[changed] == SOLREF_MODE_RAW]
                if len(ignored):
                    facts.append(
                        f"model.{name} rows {_rows_text(ignored)} are DOFs with mujoco.solreflimit_mode RAW; their "
                        "limit solref comes from model.mujoco.solreflimit."
                    )
    return facts


# ---------------------------------------------------------------------------------------------------------------------
# solver_params


def _round(value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype.kind == "f":
        if array.ndim == 0:
            return float(f"{float(array):.6g}")
        return [float(f"{float(x):.6g}") for x in array.reshape(-1)]
    if array.dtype.kind in "iub":
        return array.item() if array.ndim == 0 else array.reshape(-1).tolist()
    return value


def _leaf(label: str | None) -> str:
    return label.rsplit("/", 1)[-1] if label else ""


def _matcher(select):
    """Predicate on a full label; ``None`` matches everything.

    Patterns are globs matched against the full label or its last path component; a
    pattern without wildcards also matches as a substring of the last component.
    """
    if select is None:
        return lambda label: True
    patterns = select if isinstance(select, list | tuple) else [select]
    if not all(isinstance(p, str) for p in patterns):
        raise ValueError("select must be a label pattern or a list of label patterns")

    def match(label: str) -> bool:
        leaf = _leaf(label)
        for pattern in patterns:
            if fnmatch.fnmatchcase(label, pattern) or fnmatch.fnmatchcase(leaf, pattern):
                return True
            if not any(c in pattern for c in "*?[") and pattern in leaf:
                return True
        return False

    return match


def dof_labels(model) -> list[str]:
    """Label per joint DOF: the joint label, with ``:axis`` for multi-DOF joints."""
    starts = model.joint_qd_start.numpy()
    labels = model.joint_label or [str(j) for j in range(model.joint_count)]
    result = [""] * model.joint_dof_count
    for joint in range(model.joint_count):
        begin, end = int(starts[joint]), int(starts[joint + 1])
        for dof in range(begin, end):
            result[dof] = labels[joint] if end - begin == 1 else f"{labels[joint]}:{dof - begin}"
    return result


def _dof_world(model) -> np.ndarray:
    starts = model.joint_qd_start.numpy()
    return np.repeat(model.joint_world.numpy(), np.diff(starts))


_KINDS = ("actuator", "joint", "geom", "body", "equality", "option")


def solver_params(model, solver, kind: str, select=None, *, world: int = 0, limit: int = 64) -> dict:
    """What the solver integrates for one kind of entity, and where each value comes from.

    Args:
        model: Model the solver was built from.
        solver: Solver instance.
        kind: ``"actuator"``, ``"joint"``, ``"geom"``, ``"body"``, ``"equality"`` or ``"option"``.
        select: Label pattern(s) selecting rows; see :func:`_matcher`.
        world: World whose rows are reported.
        limit: Maximum number of rows.

    Returns:
        ``{"solver", "kind", "world", "rows", ...}`` with per-row compiled values, their ``from``
        sources, the current Newton values, and ``pending`` names where the two differ.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    if isinstance(world, bool) or not isinstance(world, int) or not 0 <= world < max(1, model.world_count):
        raise ValueError(f"world must be an integer in [0, {max(1, model.world_count) - 1}]")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    match = _matcher(select)
    if _is_mujoco(solver):
        result = _MuJoCoParams(model, solver, world).run(kind, match, limit)
    else:
        result = _generic_params(model, solver, kind, match, world, limit)
    result = {"solver": type(solver).__name__, "kind": kind, "world": world, **result}
    rows = result.get("rows")
    if isinstance(rows, list):
        result["row_count"] = len(rows)
        result["rows_truncated"] = result.pop("_truncated", False)
    return result


def _limited(rows: list, limit: int) -> tuple[list, bool]:
    return rows[:limit], len(rows) > limit


def _generic_params(model, solver, kind, match, world, limit) -> dict:
    name = type(solver).__name__
    unsupported = (
        f"{name} does not expose compiled solver parameters; rows list the Newton model values, which this "
        "solver may read directly or ignore."
    )
    rows = []
    if kind in ("actuator", "joint"):
        labels = dof_labels(model)
        worlds = _dof_world(model)
        fields = (
            ("joint_target_mode", "joint_target_ke", "joint_target_kd", "joint_effort_limit")
            if kind == "actuator"
            else (
                "joint_armature",
                "joint_damping",
                "joint_friction",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_limit_ke",
                "joint_limit_kd",
                "joint_effort_limit",
                "joint_velocity_limit",
            )
        )
        values = {f: model_field(model, f).numpy() for f in fields if model_field(model, f) is not None}
        for dof, label in enumerate(labels):
            if worlds[dof] not in (world, -1) or not match(label):
                continue
            rows.append({"dof": dof, "label": _leaf(label), **{f: _round(v[dof]) for f, v in values.items()}})
    elif kind == "geom":
        fields = [
            f
            for f in (
                "shape_material_mu",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_mu_torsional",
                "shape_material_mu_rolling",
                "shape_material_restitution",
                "shape_margin",
                "shape_gap",
            )
            if model_field(model, f) is not None
        ]
        values = {f: model_field(model, f).numpy() for f in fields}
        worlds = model.shape_world.numpy()
        for shape, label in enumerate(model.shape_label or []):
            if worlds[shape] not in (world, -1) or not match(label):
                continue
            rows.append({"shape": shape, "label": _leaf(label), **{f: _round(v[shape]) for f, v in values.items()}})
    elif kind == "body":
        mass, com, inertia = model.body_mass.numpy(), model.body_com.numpy(), model.body_inertia.numpy()
        worlds = model.body_world.numpy()
        for body, label in enumerate(model.body_label or []):
            if worlds[body] not in (world, -1) or not match(label):
                continue
            rows.append(
                {
                    "body": body,
                    "label": _leaf(label),
                    "body_mass": _round(mass[body]),
                    "body_com": _round(com[body]),
                    "body_inertia_diag": _round(np.diag(inertia[body])),
                }
            )
    elif kind == "equality":
        return {"rows": [], "unsupported": f"{name}: equality constraint parameters are not reported for this solver."}
    else:
        options = {
            key: value
            for key, value in vars(solver).items()
            if not key.startswith("_") and type(value) in (int, float, bool, str)
        }
        return {"options": options, "unsupported": unsupported}
    rows, truncated = _limited(rows, limit)
    return {"rows": rows, "_truncated": truncated, "unsupported": unsupported}


class _MuJoCoParams:
    """Compiled MuJoCo values next to their Newton sources (mirrors ``SolverMuJoCo._update_*``)."""

    def __init__(self, model, solver, world: int):
        self.model, self.solver = model, solver
        self.cpu = bool(getattr(solver, "use_mujoco_cpu", False))
        nworld = solver.mjc_geom_to_newton_shape.shape[0] if solver.mjc_geom_to_newton_shape is not None else 1
        self.world = world if world < nworld else 0
        self.world_count = max(1, int(model.world_count))
        self.mujoco = getattr(model, "mujoco", None)

    def compiled(self, name: str) -> np.ndarray:
        """``name`` of the MuJoCo model for the selected world (CPU backend: ``mj_model``)."""
        if self.cpu and hasattr(self.solver.mj_model, name):
            return np.asarray(getattr(self.solver.mj_model, name))
        values = getattr(self.solver.mjw_model, name).numpy()
        return values[self.world % values.shape[0]] if values.ndim >= 2 else values

    def per_world(self, name: str) -> bool:
        return name in _MJW_BATCHED_MODEL_FIELDS

    def attr(self, name: str):
        array = getattr(self.mujoco, name, None) if self.mujoco is not None else None
        return array.numpy() if isinstance(array, wp.array) else None

    def run(self, kind: str, match, limit: int) -> dict:
        return getattr(self, f"_{kind}")(match, limit)

    @staticmethod
    def _pending(row: dict, checks: list[tuple[str, Any, Any]]) -> None:
        pending = [
            name
            for name, compiled, newton in checks
            if newton is not None and not np.allclose(np.asarray(compiled), np.asarray(newton), rtol=1e-5, atol=1e-7)
        ]
        if pending:
            row["pending"] = pending

    # -- actuators --------------------------------------------------------------------------------------------------

    def _actuator(self, match, limit: int) -> dict:
        solver, model = self.solver, self.model
        maps = _mujoco_actuator_maps(solver)
        result = {
            "flags": {
                "model.joint_target_ke/kd": "JOINT_DOF_FORCE_PROPERTIES",
                "model.joint_effort_limit": "JOINT_DOF_FORCE_PROPERTIES",
                "model.mujoco.actuator_*": "ACTUATOR_PROPERTIES",
            },
            "per_world": {
                name: self.per_world(name)
                for name in (
                    "actuator_gainprm",
                    "actuator_biasprm",
                    "actuator_ctrlrange",
                    "actuator_forcerange",
                    "jnt_actfrcrange",
                )
            },
            "joint_actfrcrange": "MuJoCo clamps the sum of all actuator forces on a joint to the joint's actfrcrange, "
            "after clamping each actuator to its own forcerange",
            "construction_only": [
                "model.joint_target_mode (which JOINT_TARGET actuators exist; POSITION vs POSITION_VELOCITY is read "
                "from world 0)",
                "model.mujoco.ctrl_source",
                "model.mujoco.actuator_gaintype/biastype/dyntype/trntype",
                "forcerange of JOINT_TARGET actuators on non-ball joints",
            ],
            "not_read": "rows of model.mujoco.actuator_gainprm/biasprm/dynprm/forcerange/actrange/gear/cranklength "
            "that belong to JOINT_TARGET actuators",
        }
        if maps is None:
            return {**result, "rows": []}
        gain, bias = self.compiled("actuator_gainprm"), self.compiled("actuator_biasprm")
        ctrl_limited = np.asarray(solver.mj_model.actuator_ctrllimited).astype(bool)
        force_limited = np.asarray(solver.mj_model.actuator_forcelimited).astype(bool)
        ctrlrange, forcerange = self.compiled("actuator_ctrlrange"), self.compiled("actuator_forcerange")
        dofs_per_world = model.joint_dof_count // self.world_count
        offset = self.world * dofs_per_world
        labels = dof_labels(model)
        ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        effort, mode = model.joint_effort_limit.numpy(), model.joint_target_mode.numpy()
        total = int(model.custom_frequency_counts.get("mujoco:actuator", 0))
        per_world = total // self.world_count if total and total % self.world_count == 0 else total
        names = {
            f: self.attr(f)
            for f in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange")
        }
        actuator_labels = getattr(self.mujoco, "actuator_label", None) if self.mujoco is not None else None
        mj = solver.mj_model
        trn_type, trn_id = np.asarray(mj.actuator_trntype), np.asarray(mj.actuator_trnid)
        jnt_type, jnt_force_limited = np.asarray(mj.jnt_type), np.asarray(mj.jnt_actfrclimited).astype(bool)
        jnt_actfrcrange = self.compiled("jnt_actfrcrange")
        jnt_dof = solver.mjc_jnt_to_newton_dof.numpy() if solver.mjc_jnt_to_newton_dof is not None else None
        jnt_dof = jnt_dof[self.world % jnt_dof.shape[0]] if jnt_dof is not None else None
        rows = []
        for actuator in range(len(maps["source"])):
            source, index = int(maps["source"][actuator]), int(maps["newton"][actuator])
            custom = int(maps["custom"][actuator]) if maps["custom"] is not None else -1
            k = self.world * per_world + custom if custom >= 0 else -1
            row: dict[str, Any] = {"actuator": actuator}
            sources: dict[str, str] = {}
            checks = []
            if source == _JOINT_TARGET:
                if index == -1:
                    continue
                position = index >= 0
                dof = offset + (index if position else -(index + 2))
                label = labels[dof]
                if not match(label):
                    continue
                row.update(label=_leaf(label), ctrl_source="JOINT_TARGET", type="position" if position else "velocity")
                row["dof"] = dof
                if position:
                    sources["gainprm[0]"] = f"model.joint_target_ke[{dof}]"
                    sources["biasprm[1]"] = f"-model.joint_target_ke[{dof}]"
                    checks += [("gainprm[0]", gain[actuator][0], ke[dof]), ("biasprm[1]", bias[actuator][1], -ke[dof])]
                    if int(mode[dof - offset]) == JointTargetMode.POSITION:  # world 0's mode decides
                        sources["biasprm[2]"] = f"-model.joint_target_kd[{dof}]"
                        checks.append(("biasprm[2]", bias[actuator][2], -kd[dof]))
                else:
                    sources["gainprm[0]"] = f"model.joint_target_kd[{dof}]"
                    sources["biasprm[2]"] = f"-model.joint_target_kd[{dof}]"
                    checks += [("gainprm[0]", gain[actuator][0], kd[dof]), ("biasprm[2]", bias[actuator][2], -kd[dof])]
                if maps["joint_effort"] is not None and maps["joint_effort"][actuator]:
                    sources["forcerange"] = f"+-model.joint_effort_limit[{dof}]"
                    checks.append(("forcerange", forcerange[actuator], [-effort[dof], effort[dof]]))
                else:
                    sources["forcerange"] = "set at construction"
                if k >= 0:
                    sources["ctrlrange"] = f"model.mujoco.actuator_ctrlrange[{k}]"
                    if names["actuator_ctrlrange"] is not None:
                        checks.append(("ctrlrange", ctrlrange[actuator], names["actuator_ctrlrange"][k]))
                row["model"] = {
                    f"joint_target_ke[{dof}]": _round(ke[dof]),
                    f"joint_target_kd[{dof}]": _round(kd[dof]),
                }
            else:
                label = (
                    actuator_labels[k] if isinstance(actuator_labels, list) and 0 <= k < len(actuator_labels) else ""
                )
                target = _actuator_target_label(self.mujoco, k)
                if not (match(label) or (target and match(target))):
                    continue
                row.update(label=_leaf(label or target) or f"actuator {actuator}", ctrl_source="CTRL_DIRECT")
                if k >= 0:
                    for field in ("gainprm", "biasprm", "dynprm", "ctrlrange", "forcerange", "actrange", "gear"):
                        sources[field] = f"model.mujoco.actuator_{field}[{k}]"
                    sources["ctrl"] = "control.mujoco.ctrl"
                    if names["actuator_gainprm"] is not None:
                        checks.append(("gainprm", gain[actuator], names["actuator_gainprm"][k]))
                    model_bias = names["actuator_biasprm"][k] if names["actuator_biasprm"] is not None else None
                    if model_bias is not None:
                        # A positive biasprm[2] on a position shortcut is a damping ratio MuJoCo resolves to -kd.
                        compare = 2 if model_bias[2] > 0 else len(model_bias)
                        checks.append(("biasprm", bias[actuator][:compare], model_bias[:compare]))
                    for field, key, values in (
                        ("actuator_ctrlrange", "ctrlrange", ctrlrange),
                        ("actuator_forcerange", "forcerange", forcerange),
                    ):
                        if names[field] is not None:
                            checks.append((key, values[actuator], names[field][k]))
            row["gainprm"] = _round(gain[actuator][:3])
            row["biasprm"] = _round(bias[actuator][:3])
            row["ctrlrange"] = _round(ctrlrange[actuator]) if ctrl_limited[actuator] else "unlimited"
            row["forcerange"] = _round(forcerange[actuator]) if force_limited[actuator] else "unlimited"
            joint = int(trn_id[actuator][0]) if int(trn_type[actuator]) in (0, 1) else -1  # mjTRN_JOINT(INPARENT)
            if 0 <= joint < len(jnt_force_limited) and jnt_force_limited[joint]:
                row["joint_actfrcrange"] = _round(jnt_actfrcrange[joint])
                joint_dof = int(jnt_dof[joint]) if jnt_dof is not None else -1
                if joint_dof >= 0 and int(jnt_type[joint]) in (2, 3):  # slide, hinge
                    sources["joint_actfrcrange"] = f"+-model.joint_effort_limit[{joint_dof}]"
                    checks.append(
                        ("joint_actfrcrange", jnt_actfrcrange[joint], [-effort[joint_dof], effort[joint_dof]])
                    )
            checks = [
                check
                for check in checks
                if not (check[0] == "ctrlrange" and not ctrl_limited[actuator])
                and not (check[0] == "forcerange" and not force_limited[actuator])
            ]
            row["from"] = sources
            self._pending(row, checks)
            rows.append(row)
        rows, truncated = _limited(rows, limit)
        return {**result, "rows": rows, "_truncated": truncated}

    # -- joints -----------------------------------------------------------------------------------------------------

    def _joint(self, match, limit: int) -> dict:
        solver, model = self.solver, self.model
        mj = solver.mj_model
        dof_map = solver.mjc_dof_to_newton_dof.numpy()
        dof_map = dof_map[self.world % dof_map.shape[0]]
        labels = dof_labels(model)
        armature, damping = self.compiled("dof_armature"), self.compiled("dof_damping")
        friction = self.compiled("dof_frictionloss")
        jnt_range, jnt_solref = self.compiled("jnt_range"), self.compiled("jnt_solref")
        jnt_solimp, stiffness = self.compiled("jnt_solimp"), self.compiled("jnt_stiffness")
        actfrcrange = self.compiled("jnt_actfrcrange")
        newton = {
            name: model_field(model, name).numpy()
            for name in (
                "joint_armature",
                "joint_damping",
                "joint_friction",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_limit_ke",
                "joint_limit_kd",
                "joint_effort_limit",
            )
        }
        limit_mode, raw_limit = self.attr("solreflimit_mode"), self.attr("solreflimit")
        passive, ref = self.attr("dof_passive_stiffness"), self.attr("dof_ref")
        invweight = self.compiled("dof_invweight0")
        kinematic = model.body_flags.numpy() & int(BodyFlags.KINEMATIC) if model.body_count else np.zeros(0, dtype=int)
        dof_body = solver.newton_dof_to_body.numpy() if solver.newton_dof_to_body is not None else None
        rows = []
        for mj_dof in range(mj.nv):
            dof = int(dof_map[mj_dof])
            if dof < 0 or not match(labels[dof]):
                continue
            joint = int(mj.dof_jntid[mj_dof])
            jtype = int(mj.jnt_type[joint])
            row = {
                "dof": dof,
                "label": _leaf(labels[dof]),
                "type": ("free", "ball", "slide", "hinge")[jtype],
                "armature": _round(armature[mj_dof]),
                "damping": _round(damping[mj_dof]),
                "frictionloss": _round(friction[mj_dof]),
            }
            sources = {
                "armature": f"model.joint_armature[{dof}]",
                "damping": f"model.joint_damping[{dof}]",
                "frictionloss": f"model.joint_friction[{dof}]",
            }
            is_kinematic = dof_body is not None and dof_body[dof] >= 0 and kinematic[dof_body[dof]]
            checks = [
                ("damping", damping[mj_dof], newton["joint_damping"][dof]),
                ("frictionloss", friction[mj_dof], newton["joint_friction"][dof]),
            ]
            if is_kinematic:
                sources["armature"] = "1e10 (kinematic body)"
            else:
                checks.append(("armature", armature[mj_dof], newton["joint_armature"][dof]))
            if jtype in (2, 3):  # slide, hinge: one DOF per MuJoCo joint
                shift = float(ref[dof]) if ref is not None else 0.0
                row.update(
                    limited=bool(mj.jnt_limited[joint]),
                    range=_round(jnt_range[joint]),
                    solref=_round(jnt_solref[joint]),
                    solimp=_round(jnt_solimp[joint][:3]),
                    stiffness=_round(stiffness[joint]),
                    actfrcrange=_round(actfrcrange[joint]),
                )
                sources["range"] = f"model.joint_limit_lower/upper[{dof}]" + (" + mujoco.dof_ref" if shift else "")
                sources["actfrcrange"] = f"+-model.joint_effort_limit[{dof}]"
                sources["solimp"] = f"model.mujoco.solimplimit[{dof}]"
                sources["stiffness"] = f"model.mujoco.dof_passive_stiffness[{dof}]"
                checks += [
                    (
                        "range",
                        jnt_range[joint],
                        [newton["joint_limit_lower"][dof] + shift, newton["joint_limit_upper"][dof] + shift],
                    ),
                    (
                        "actfrcrange",
                        actfrcrange[joint],
                        [-newton["joint_effort_limit"][dof], newton["joint_effort_limit"][dof]],
                    ),
                ]
                if passive is not None:
                    checks.append(("stiffness", stiffness[joint], passive[dof]))
                mode = int(limit_mode[dof]) if limit_mode is not None else SOLREF_MODE_FORCE_SPACE
                row["solref_mode"] = _SOLREF_MODES.get(mode, str(mode))
                if mode == SOLREF_MODE_RAW:
                    sources["solref"] = f"model.mujoco.solreflimit[{dof}]"
                    if raw_limit is not None:
                        checks.append(("solref", jnt_solref[joint], raw_limit[dof]))
                elif mode == SOLREF_MODE_MJCF_DEFAULT:
                    sources["solref"] = "MuJoCo default (0.02, 1) until joint_limit_ke/kd change"
                else:
                    sources["solref"] = (
                        f"model.joint_limit_ke/kd[{dof}] scaled by dof_invweight0 * (1 - solimp[1]), "
                        "as (timeconst, dampratio)"
                    )
                    ke, kd = float(newton["joint_limit_ke"][dof]), float(newton["joint_limit_kd"][dof])
                    width = float(jnt_solimp[joint][1])
                    weight = float(invweight[mj_dof])
                    factor = weight * (1.0 - width) if weight > 0.0 and width < 1.0 else 1.0
                    if ke > 0.0 and kd > 0.0:
                        ke, kd = max(ke * factor, 2.2e-16), max(kd * factor, 2.2e-16)
                        checks.append(("solref", jnt_solref[joint], [2.0 / kd, kd / 2.0 * np.sqrt(1.0 / ke)]))
            row["from"] = sources
            self._pending(row, checks)
            rows.append(row)
        rows, truncated = _limited(rows, limit)
        return {
            "flags": {
                "model.joint_armature": "JOINT_DOF_INERTIAL_PROPERTIES",
                "model.mujoco.dof_ref, dof_springref": "JOINT_REFERENCE_POSE_PROPERTIES",
                "other model.joint_* DOF and model.mujoco joint attributes": "JOINT_DOF_FORCE_PROPERTIES",
                "all of these": "JOINT_DOF_PROPERTIES",
            },
            "per_world": {
                name: self.per_world(name)
                for name in ("dof_armature", "dof_damping", "dof_frictionloss", "jnt_range", "jnt_solref")
            },
            "not_read": ["model.joint_velocity_limit"],
            "rows": rows,
            "_truncated": truncated,
        }

    # -- geoms ------------------------------------------------------------------------------------------------------

    def _geom(self, match, limit: int) -> dict:
        solver, model = self.solver, self.model
        geom_map = solver.mjc_geom_to_newton_shape.numpy()
        geom_map = geom_map[self.world % geom_map.shape[0]]
        labels = model.shape_label or [str(s) for s in range(model.shape_count)]
        friction, solref = self.compiled("geom_friction"), self.compiled("geom_solref")
        solimp, solmix = self.compiled("geom_solimp"), self.compiled("geom_solmix")
        margin, gap = self.compiled("geom_margin"), self.compiled("geom_gap")
        priority, condim = np.asarray(solver.mj_model.geom_priority), np.asarray(solver.mj_model.geom_condim)
        mu = model.shape_material_mu.numpy()
        mu_t, mu_r = model.shape_material_mu_torsional.numpy(), model.shape_material_mu_rolling.numpy()
        ke, kd = model.shape_material_ke.numpy(), model.shape_material_kd.numpy()
        shape_gap = model.shape_gap.numpy() if model.shape_gap is not None else None
        mode, raw = self.attr("solref_mode"), self.attr("solref")
        model_solmix = self.attr("geom_solmix")
        zero_margin = bool(
            getattr(solver, "_use_mujoco_contacts", True) and getattr(solver, "_zero_margins_for_native_ccd", False)
        )
        rows = []
        for geom, mapped in enumerate(geom_map):
            shape = int(mapped)
            if shape < 0 or not match(labels[shape]):
                continue
            shape_mode = int(mode[shape]) if mode is not None and raw is not None else SOLREF_MODE_FORCE_SPACE
            row = {
                "geom": geom,
                "shape": shape,
                "label": _leaf(labels[shape]),
                "friction": _round(friction[geom]),
                "solref": _round(solref[geom]),
                "solref_mode": _SOLREF_MODES.get(shape_mode, str(shape_mode)),
                "solimp": _round(solimp[geom][:3]),
                "solmix": _round(solmix[geom]),
                "priority": int(priority[geom]),
                "condim": int(condim[geom]),
                "margin": _round(margin[geom]),
                "gap": _round(gap[geom]),
            }
            sources = {
                "friction": f"model.shape_material_mu/mu_torsional/mu_rolling[{shape}]",
                "solimp": f"model.mujoco.geom_solimp[{shape}]",
                "solmix": f"model.mujoco.geom_solmix[{shape}]",
                "priority": "model.mujoco.geom_priority (construction only, shared by all worlds)",
                "condim": "model.mujoco.condim (construction only, shared by all worlds)",
                "margin": "0: margins are zeroed for MuJoCo collision with box or mesh geoms"
                if zero_margin
                else f"model.shape_margin[{shape}]",
                "gap": f"model.shape_gap[{shape}]",
            }
            checks = [("friction", friction[geom], [mu[shape], mu_t[shape], mu_r[shape]])]
            if shape_mode == SOLREF_MODE_RAW:
                sources["solref"] = f"model.mujoco.solref[{shape}]"
                checks.append(("solref", solref[geom], raw[shape]))
            else:
                sources["solref"] = (
                    f"(2 / kd, kd / 2 * sqrt(1 / ke)) of model.shape_material_ke/kd[{shape}]"
                    if ke[shape] > 0 and kd[shape] > 0
                    else "MuJoCo default (0.02, 1): shape_material_ke or kd is not positive"
                )
                expected = (
                    [2.0 / kd[shape], kd[shape] / 2.0 * np.sqrt(1.0 / ke[shape])]
                    if ke[shape] > 0 and kd[shape] > 0
                    else [0.02, 1.0]
                )
                checks.append(("solref", solref[geom], expected))
            if model_solmix is not None:
                checks.append(("solmix", solmix[geom], model_solmix[shape]))
            if shape_gap is not None:
                checks.append(("gap", gap[geom], shape_gap[shape]))
            row["from"] = sources
            self._pending(row, checks)
            rows.append(row)
        rows, truncated = _limited(rows, limit)
        return {
            "flags": {"model.shape_* and model.mujoco geom attributes": "SHAPE_PROPERTIES"},
            "per_world": {name: self.per_world(name) for name in ("geom_friction", "geom_solref", "geom_priority")},
            "not_read": ["model.shape_material_restitution"]
            + (
                ["model.shape_material_kf (read only with use_mujoco_contacts=False)"]
                if getattr(solver, "_use_mujoco_contacts", True)
                else []
            ),
            "rows": rows,
            "_truncated": truncated,
        }

    # -- bodies -----------------------------------------------------------------------------------------------------

    def _body(self, match, limit: int) -> dict:
        solver, model = self.solver, self.model
        body_map = solver.mjc_body_to_newton.numpy()
        body_map = body_map[self.world % body_map.shape[0]]
        labels = model.body_label or [str(b) for b in range(model.body_count)]
        mass, ipos = self.compiled("body_mass"), self.compiled("body_ipos")
        inertia, gravcomp = self.compiled("body_inertia"), self.compiled("body_gravcomp")
        newton_mass, newton_com = model.body_mass.numpy(), model.body_com.numpy()
        newton_inertia = model.body_inertia.numpy()
        newton_gravcomp = self.attr("gravcomp")
        rows = []
        for mj_body, mapped in enumerate(body_map):
            body = int(mapped)
            if body < 0 or not match(labels[body]):
                continue
            row = {
                "body": body,
                "label": _leaf(labels[body]),
                "mass": _round(mass[mj_body]),
                "ipos": _round(ipos[mj_body]),
                "inertia": _round(inertia[mj_body]),
                "gravcomp": _round(gravcomp[mj_body]),
                "from": {
                    "mass": f"model.body_mass[{body}]",
                    "ipos": f"model.body_com[{body}]",
                    "inertia": f"principal moments of model.body_inertia[{body}]",
                    "gravcomp": f"model.mujoco.gravcomp[{body}]" if newton_gravcomp is not None else "0",
                },
            }
            checks = [
                ("mass", mass[mj_body], newton_mass[body]),
                ("ipos", ipos[mj_body], newton_com[body]),
                ("inertia", np.sort(inertia[mj_body]), np.sort(np.linalg.eigvalsh(newton_inertia[body]))),
            ]
            if newton_gravcomp is not None:
                checks.append(("gravcomp", gravcomp[mj_body], newton_gravcomp[body]))
            self._pending(row, checks)
            rows.append(row)
        rows, truncated = _limited(rows, limit)
        return {
            "flags": {
                "model.body_mass/body_com/body_inertia/mujoco.gravcomp": "BODY_INERTIAL_PROPERTIES",
                "model.body_flags": "BODY_PROPERTIES",
            },
            "per_world": {name: self.per_world(name) for name in ("body_mass", "body_inertia", "body_gravcomp")},
            "not_read": ["model.body_inv_mass", "model.body_inv_inertia"],
            "rows": rows,
            "_truncated": truncated,
        }

    # -- equality constraints ---------------------------------------------------------------------------------------

    def _equality(self, match, limit: int) -> dict:
        solver = self.solver
        mj = solver.mj_model
        if mj.neq == 0:
            return {"rows": [], "flags": {"model.mujoco.eq_* and equality_constraint_*": "CONSTRAINT_PROPERTIES"}}
        solref, solimp = self.compiled("eq_solref"), self.compiled("eq_solimp")
        active = solver.mjw_data.eq_active.numpy() if not self.cpu else np.asarray(solver.mj_data.eq_active)[None]
        active = active[self.world % active.shape[0]]
        maps = {
            "equality": solver.mjc_eq_to_newton_eq,
            "loop joint": solver.mjc_eq_to_newton_jnt,
            "mimic": solver.mjc_eq_to_newton_mimic,
            "joint mimic": solver.mjc_eq_to_newton_joint_mimic,
        }
        maps = {key: value.numpy()[self.world % value.shape[0]] for key, value in maps.items() if value is not None}
        eq_labels = getattr(self.mujoco, "equality_constraint_label", None) if self.mujoco is not None else None
        model_solref, model_solimp = self.attr("eq_solref"), self.attr("eq_solimp")
        joint_labels = self.model.joint_label or []
        rows = []
        for eq in range(mj.neq):
            row = {
                "equality": eq,
                "type": ("CONNECT", "WELD", "JOINT", "TENDON", "FLEX", "DISTANCE")[int(mj.eq_type[eq])]
                if int(mj.eq_type[eq]) < 6
                else int(mj.eq_type[eq]),
                "active": bool(active[eq]),
                "solref": _round(solref[eq]),
                "solimp": _round(solimp[eq][:3]),
            }
            label, sources, checks = f"equality {eq}", {}, []
            if maps.get("equality") is not None and maps["equality"][eq] >= 0:
                index = int(maps["equality"][eq])
                if isinstance(eq_labels, list) and index < len(eq_labels):
                    label = eq_labels[index]
                sources = {
                    "solref": f"model.mujoco.eq_solref[{index}]",
                    "solimp": f"model.mujoco.eq_solimp[{index}]",
                    "active": f"model.mujoco.equality_constraint_enabled[{index}]",
                    "data": f"model.mujoco.equality_constraint_anchor/relpose/polycoef/torquescale[{index}]",
                }
                if model_solref is not None:
                    checks.append(("solref", solref[eq], model_solref[index]))
                if model_solimp is not None:
                    checks.append(("solimp", solimp[eq], model_solimp[index]))
            elif maps.get("loop joint") is not None and maps["loop joint"][eq] >= 0:
                joint = int(maps["loop joint"][eq])
                label = joint_labels[joint] if joint < len(joint_labels) else label
                sources = {"data": f"loop joint {joint}: model.joint_X_p/X_c (JOINT_PROPERTIES)"}
            elif maps.get("joint mimic") is not None and maps["joint mimic"][eq] >= 0:
                joint = int(maps["joint mimic"][eq])
                label = joint_labels[joint] if joint < len(joint_labels) else label
                sources = {"data": f"model.joint_mimic_coeffs of joint {joint}"}
            elif maps.get("mimic") is not None and maps["mimic"][eq] >= 0:
                sources = {"data": f"model.constraint_mimic_coef0/coef1/enabled[{int(maps['mimic'][eq])}]"}
            if not match(label):
                continue
            row["label"] = _leaf(label)
            row["from"] = sources
            self._pending(row, checks)
            rows.append(row)
        rows, truncated = _limited(rows, limit)
        return {
            "flags": {"model.mujoco.eq_*, equality_constraint_*, joint_mimic_coeffs": "CONSTRAINT_PROPERTIES"},
            "per_world": {name: self.per_world(name) for name in ("eq_solref", "eq_solimp", "eq_data")},
            "rows": rows,
            "_truncated": truncated,
        }

    # -- options ----------------------------------------------------------------------------------------------------

    def _option(self, match, limit: int) -> dict:
        solver = self.solver
        mujoco = solver._mujoco
        mj = solver.mj_model
        options = {}
        for name, enum in (
            ("solver", mujoco.mjtSolver),
            ("integrator", mujoco.mjtIntegrator),
            ("cone", mujoco.mjtCone),
            ("jacobian", mujoco.mjtJacobian),
        ):
            value = int(getattr(mj.opt, name))
            options[name] = enum(value).name.split("_", 1)[-1].lower()
        for name in ("iterations", "ls_iterations", "ccd_iterations", "sdf_iterations", "sdf_initpoints"):
            options[name] = int(getattr(mj.opt, name))
        opt = solver.mjw_model.opt if not self.cpu else None

        def value(name: str):
            if opt is not None and hasattr(opt, name):
                data = getattr(opt, name)
                if isinstance(data, wp.array):
                    data = data.numpy()
                    data = data[self.world % data.shape[0]] if data.ndim >= 1 and data.shape[0] > 1 else data
                    data = data.reshape(-1) if np.ndim(data) else data
                    if np.size(data) == 1:
                        data = np.asarray(data).reshape(()).item()
                return _round(data)
            return _round(getattr(mj.opt, name))

        if opt is not None and hasattr(opt, "impratio_invsqrt"):
            invsqrt = value("impratio_invsqrt")
            options["impratio"] = _round(1.0 / float(invsqrt) ** 2) if invsqrt else None
        else:
            options["impratio"] = _round(mj.opt.impratio)
        for name in (
            "tolerance",
            "ls_tolerance",
            "ccd_tolerance",
            "gravity",
            "wind",
            "magnetic",
            "density",
            "viscosity",
        ):
            options[name] = value(name)
        disabled = [
            name.removeprefix("mjDSBL_").lower()
            for name, bit in mujoco.mjtDisableBit.__members__.items()
            if int(mj.opt.disableflags) & int(bit)
        ]
        enabled = [
            name.removeprefix("mjENBL_").lower()
            for name, bit in mujoco.mjtEnableBit.__members__.items()
            if int(mj.opt.enableflags) & int(bit)
        ]
        options["disabled"] = disabled
        options["enabled"] = enabled
        data = solver.mjw_data
        nworld = int(getattr(data, "nworld", 1)) if data is not None else 1
        model_values = {}
        for name in (
            "impratio",
            "tolerance",
            "ls_tolerance",
            "iterations",
            "ls_iterations",
            "cone",
            "solver",
            "integrator",
        ):
            values = self.attr(name)
            if values is not None and values.size:
                model_values[f"model.mujoco.{name}"] = _round(
                    values[self.world % len(values)] if values.ndim else values
                )
        return {
            "options": options,
            "timestep": "the dt passed to step() (opt.timestep is overwritten on every step)",
            "refsafe": "refsafe" not in disabled,
            "buffers": {
                "naconmax": int(getattr(data, "naconmax", 0) or 0) if data is not None else None,
                "nconmax_per_world": int(getattr(data, "naconmax", 0) or 0) // max(nworld, 1)
                if data is not None
                else None,
                "njmax": int(getattr(data, "njmax", 0) or 0) if data is not None else None,
                "nworld": nworld,
            },
            "contacts": "MuJoCo collision" if getattr(solver, "_use_mujoco_contacts", True) else "Newton contacts",
            "backend": "mujoco (CPU)" if self.cpu else "mujoco_warp",
            "per_world": sorted(_MJW_BATCHED_OPTION_FIELDS),
            "refreshed_by": {"gravity": "MODEL_PROPERTIES (from model.gravity)"},
            "construction_only": "every other option: constructor argument, else model.mujoco.<option>, else default",
            "model_attributes": model_values,
        }


def _actuator_target_label(mujoco_attrs, index: int) -> str:
    labels = getattr(mujoco_attrs, "actuator_target_label", None) if mujoco_attrs is not None else None
    return labels[index] if isinstance(labels, list) and 0 <= index < len(labels) and labels[index] else ""
