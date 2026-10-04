# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reports on what a solver integrates and on signs of a failing simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..sim import Contacts, Model, State
    from ..solvers.solver import SolverBase


def report_solver_params(
    solver: SolverBase,
    kind: str,
    select: str | list[str] | None = None,
    *,
    world: int = 0,
    limit: int = 64,
) -> dict[str, Any]:
    """Report the values a solver integrates for one kind of entity and where each comes from.

    For :class:`~newton.solvers.SolverMuJoCo`, each row holds the compiled MuJoCo value, ``from`` (the
    model array and index it is computed from), and ``pending`` (values whose model array differs from
    the compiled value, i.e. edits that no ``notify_model_changed()`` call has applied). The report also
    lists the :class:`~newton.ModelFlags` that refresh each source, whether the MuJoCo field can differ
    between worlds, and fields that are read only at construction or not at all. Other solvers report the
    model values and state that compiled values are not available.

    .. note::
        Experimental: the layout of the report may change without a deprecation period.

    Args:
        solver: Solver to inspect; its ``model`` is the model reported.
        kind: ``"actuator"``, ``"joint"``, ``"geom"``, ``"body"``, ``"equality"``, or ``"option"``.
        select: Label glob, or list of globs, matched against full labels and their last path
            component; a pattern without wildcards also matches as a substring of the last component.
            ``None`` selects every row.
        world: World whose rows are reported.
        limit: Maximum number of rows.

    Returns:
        ``solver``, ``kind``, ``world``, ``rows`` (``options`` for ``kind="option"``), ``row_count``,
        ``rows_truncated``, and the facts above.
    """
    from ..mcp.solverview import solver_params  # noqa: PLC0415

    return solver_params(solver.model, solver, kind, select, world=world, limit=limit)


def report_health(
    model: Model,
    state: State,
    solver: SolverBase | None = None,
    *,
    contacts: Contacts | None = None,
    initial_state: State | None = None,
    per_world: bool = True,
    twins: bool = False,
    penetration: float = 0.01,
    twins_tolerance: float = 1e-3,
    limit: int = 8,
) -> dict[str, Any]:
    """Check a state and solver for non-finite values, runaway speeds, full buffers, and deep penetration.

    :class:`~newton.solvers.SolverMuJoCo` adds per-world checks of its own data, its ``njmax`` and
    contact buffers, and the shape pairs of its deepest contacts; for other solvers, ``contacts`` from
    the collision pipeline are checked instead.

    .. note::
        Experimental: the layout of the report may change without a deprecation period.

    Args:
        model: Model of ``state``.
        state: State to inspect.
        solver: Solver whose data and buffers to inspect, or ``None``.
        contacts: Collision-pipeline contacts, or ``None``.
        initial_state: State at the start; with ``twins``, joint coordinates are compared as
            displacements from it (otherwise only joint velocities are compared).
        per_world: Name the worlds behind each finding.
        twins: Compare the joint states of worlds built identical and list the worlds that deviate
            from the per-coordinate median by more than ``twins_tolerance``.
        penetration: Overlap [m] above which contacts are reported by shape pair.
        twins_tolerance: Deviation [m or rad, and m/s or rad/s for velocities] that counts as disagreeing.
        limit: Maximum number of shape pairs listed.

    Returns:
        ``ok``, ``warnings``, ``stats``, ``checked`` (what was inspected), and, where they apply,
        ``worlds`` (world indices per finding), ``penetration`` (shape pairs with depth [m]), and
        ``unsupported`` (checks this solver does not allow).
    """
    from ..mcp.diagnostics import health_report  # noqa: PLC0415

    joint_q = getattr(initial_state, "joint_q", None)
    initial = {"joint_q": joint_q.numpy()} if joint_q is not None else None
    return health_report(
        model,
        state,
        solver,
        contacts=contacts,
        initial=initial,
        per_world=per_world,
        twins=twins,
        penetration=penetration,
        twins_tolerance=twins_tolerance,
        limit=limit,
    )
