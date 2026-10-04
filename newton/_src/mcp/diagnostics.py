# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact solver-level diagnostics for trusted execution.

These helpers answer questions agents otherwise answer by reading solver
source: which contact parameters the solver actually uses after material
combination, and whether the simulation shows common failure signs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _label(labels, index: int) -> str:
    if index < 0:
        return "world"
    return labels[index] if labels is not None and index < len(labels) else str(index)


def solver_contacts(session, *, limit: int = 20) -> dict:
    """Active solver contacts grouped by shape pair, with effective parameters.

    For :class:`~newton.solvers.SolverMuJoCo` this reports what MuJoCo
    integrates after geom priority and material mixing: ``solref``
    (positive: time constant [s] and damping ratio; negative: stiffness and
    damping), ``solimp``, friction (sliding, torsional, rolling), and the
    signed distance [m]; ``active`` counts rows inside the contact margin.
    Authored Newton shape materials of both sides are listed for comparison.
    Other solvers report Newton collision contacts.

    Args:
        session: Live :class:`SimulationSession`.
        limit: Maximum number of shape pairs returned.

    Returns:
        ``{"source", "count", "pairs": [...]}`` sorted by contact count.
    """
    model, solver = session.model, session.solver
    shape_labels = getattr(model, "shape_label", None)
    body_labels = getattr(model, "body_label", None)
    shape_body = model.shape_body.numpy() if model.shape_count else np.zeros(0, dtype=int)
    material = {
        name: getattr(model, f"shape_material_{name}").numpy()
        for name in ("ke", "kd", "mu")
        if getattr(model, f"shape_material_{name}", None) is not None
    }
    priority = solmix = None
    mujoco_attrs = getattr(model, "mujoco", None)
    if mujoco_attrs is not None and getattr(mujoco_attrs, "geom_priority", None) is not None:
        priority = mujoco_attrs.geom_priority.numpy()
    if mujoco_attrs is not None and getattr(mujoco_attrs, "geom_solmix", None) is not None:
        solmix = mujoco_attrs.geom_solmix.numpy()

    def resolution(a: int, b: int) -> dict:
        """How MuJoCo combines the two shapes' parameters (mirrors mujoco_warp contact_params)."""
        if priority is None or min(a, b) < 0:
            return {}
        pa, pb = int(priority[a]), int(priority[b])
        if pa != pb:
            winner = a if pa > pb else b
            return {
                "decided_by": f"shape {winner} ({_label(shape_labels, winner)}): higher geom_priority, other side ignored"
            }
        wa = float(solmix[a]) if solmix is not None else 1.0
        wb = float(solmix[b]) if solmix is not None else 1.0
        weight = 0.5 if wa + wb <= 0.0 else wa / (wa + wb)
        entry = {"decided_by": f"mixed: {weight:.2f} x shape {a} + {1 - weight:.2f} x shape {b}; friction = max"}
        if "ke" in material and "kd" in material:
            entry["mixed_ke"] = weight * float(material["ke"][a]) + (1 - weight) * float(material["ke"][b])
            entry["mixed_kd"] = weight * float(material["kd"][a]) + (1 - weight) * float(material["kd"][b])
        return entry

    def side(shape: int) -> dict:
        body = int(shape_body[shape]) if 0 <= shape < len(shape_body) else -1
        entry = {"shape": int(shape), "label": _label(shape_labels, shape), "body": _label(body_labels, body)}
        for name, values in material.items():
            if 0 <= shape < len(values):
                entry[name] = float(values[shape])
        if priority is not None and 0 <= shape < len(priority):
            entry["priority"] = int(priority[shape])
        return entry

    pairs: dict[tuple, dict[str, Any]] = {}
    data = getattr(solver, "mjw_data", None)
    geom_map = getattr(solver, "mjc_geom_to_newton_shape", None)
    if data is not None and geom_map is not None:
        source = "mujoco"
        count = int(data.nacon.numpy()[0])
        contact = data.contact
        geoms = contact.geom.numpy()[:count]
        worlds = contact.worldid.numpy()[:count]
        dist = contact.dist.numpy()[:count]
        solref = contact.solref.numpy()[:count]
        solimp = contact.solimp.numpy()[:count]
        friction = contact.friction.numpy()[:count]
        margin = contact.includemargin.numpy()[:count] if hasattr(contact, "includemargin") else np.zeros(count)
        mapping = geom_map.numpy()
        for i in range(count):
            world = int(worlds[i]) if mapping.shape[0] > 1 else 0
            a, b = (int(mapping[world, g]) if g >= 0 else -1 for g in geoms[i])
            key = (min(a, b), max(a, b))
            row = pairs.get(key)
            if row is None:
                row = pairs[key] = {
                    "shapes": [side(key[0]), side(key[1])],
                    **resolution(*key),
                    "count": 0,
                    "active": 0,
                    "min_dist": float(dist[i]),
                    "solref": np.round(solref[i], 6).tolist(),
                    "solimp": np.round(solimp[i], 6).tolist(),
                    "friction": np.round(friction[i], 6).tolist(),
                }
            row["count"] += 1
            row["active"] += int(dist[i] < margin[i])
            row["min_dist"] = min(row["min_dist"], float(dist[i]))
    else:
        source = "newton"
        contacts = session.contacts
        count = int(contacts.rigid_contact_count.numpy()[0]) if contacts is not None else 0
        count = min(count, contacts.rigid_contact_max) if contacts is not None else 0
        if count:
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            for a, b in zip(shape0, shape1, strict=True):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                row = pairs.setdefault(key, {"shapes": [side(key[0]), side(key[1])], "count": 0})
                row["count"] += 1
    rows = sorted(pairs.values(), key=lambda r: (-r.get("active", r["count"]), -r["count"]))
    result = {"source": source, "count": count, "pairs": rows[:limit], "pairs_truncated": len(rows) > limit}
    if source == "mujoco":
        result["rules"] = (
            "solref/solimp/friction are the values MuJoCo integrates. decided_by names the material source: a "
            "higher geom_priority wins outright, equal priorities mix by solmix. By default each shape's ke/kd map to "
            "solref = (2 / kd, kd / 2 * sqrt(1 / ke)); force-space shapes combine ke/kd with the effective mass "
            "(docs/solvers/mujoco.rst, 'Shape-material contact stiffness and damping'). Edit model.shape_material_* "
            "or model.mujoco.geom_* and call solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)."
        )
    return result


_STATE_ROWS = {
    "body_q": "body",
    "body_qd": "body",
    "joint_q": "coord",
    "joint_qd": "dof",
    "particle_q": "particle",
    "particle_qd": "particle",
}


def _row_worlds(model, kind: str) -> np.ndarray | None:
    """World index of each row of a state array of the given kind, or ``None``."""
    if kind == "body":
        return model.body_world.numpy() if model.body_count else None
    if kind == "particle":
        return model.particle_world.numpy() if getattr(model, "particle_world", None) is not None else None
    starts = (model.joint_q_start if kind == "coord" else model.joint_qd_start).numpy()
    return np.repeat(model.joint_world.numpy(), np.diff(starts)) if model.joint_count else None


def _worlds_text(worlds, limit: int = 8) -> str:
    worlds = sorted(int(w) for w in worlds)
    text = ", ".join(str(w) for w in worlds[:limit])
    return f"[{text}{', ...' if len(worlds) > limit else ''}] ({len(worlds)} worlds)"


def _static_bodies(model) -> np.ndarray:
    """Bodies joined to the world through joints without degrees of freedom only."""
    static = np.zeros(model.body_count, dtype=bool)
    if not model.joint_count or not model.body_count:
        return static
    child, parent = model.joint_child.numpy(), model.joint_parent.numpy()
    fixed = np.diff(model.joint_qd_start.numpy()) == 0
    moving = np.zeros(model.body_count, dtype=bool)
    moving[child[~fixed & (child >= 0)]] = True
    fixed_child, fixed_parent = child[fixed & (child >= 0)], parent[fixed & (child >= 0)]
    while True:
        attached = (fixed_parent < 0) | static[np.maximum(fixed_parent, 0)]
        new = fixed_child[attached & ~moving[fixed_child] & ~static[fixed_child]]
        if not len(new):
            return static
        static[new] = True


def _skipped_pairs(model, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of contacts between two static shapes, and between shapes the model filters from colliding.

    Contact forces cannot separate two shapes that are static (on no body, or on bodies joined to the
    world without degrees of freedom), and filtered pairs never collide in Newton, so neither
    overlap is a sign of failure. Unmapped shapes (``-1``) are never skipped.
    """
    from ..geometry.flags import ShapeFlags  # noqa: PLC0415

    valid = (first >= 0) & (second >= 0)
    a, b = np.where(valid, first, 0).astype(np.int64), np.where(valid, second, 0).astype(np.int64)
    body = model.shape_body.numpy()
    shape_static = (body < 0) | _static_bodies(model)[np.maximum(body, 0)]
    static = valid & shape_static[a] & shape_static[b]
    collide = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0
    world, group = model.shape_world.numpy(), model.shape_collision_group.numpy()
    group_a, group_b = group[a], group[b]
    groups_collide = (
        (group_a != 0)
        & (group_b != 0)
        & np.where(group_a > 0, (group_a == group_b) | (group_b < 0), group_a != group_b)
    )
    filtered = (
        ~(collide[a] & collide[b])
        | ((body[a] == body[b]) & (body[a] >= 0))
        | ((world[a] != world[b]) & (world[a] >= 0) & (world[b] >= 0))
        | ~groups_collide
        | model.shape_collision_filter_mask(np.stack([a, b], axis=1))
    )
    return static, valid & ~static & filtered


def _skip_note(stats: dict, static: np.ndarray, filtered: np.ndarray) -> None:
    if static.any() or filtered.any():
        stats["penetration_skipped_contacts"] = {
            "static_pairs": int(static.sum()),
            "filtered_pairs": int(filtered.sum()),
        }


def _penetration_pairs(pairs: dict, depth: np.ndarray, first, second, worlds, labels, threshold: float) -> None:
    """Accumulate the deepest overlap [m] per shape pair for contacts deeper than ``threshold``."""
    for i in np.flatnonzero(depth > threshold):
        a, b = int(first[i]), int(second[i])
        key = (min(a, b), max(a, b))
        entry = pairs.setdefault(
            key,
            {"shapes": [_final(_label(labels, key[0])), _final(_label(labels, key[1]))], "depth": 0.0, "worlds": set()},
        )
        entry["depth"] = max(entry["depth"], float(depth[i]))
        entry["worlds"].add(int(worlds[i]))


def _mujoco_health(model, solver, report: dict, *, per_world: bool, threshold: float, limit: int) -> None:
    warnings, stats, worlds = report["warnings"], report["stats"], report["worlds"]
    labels = getattr(model, "shape_label", None)
    cpu = bool(getattr(solver, "use_mujoco_cpu", False))
    report["checked"] += ["MuJoCo qpos/qvel/qacc", "MuJoCo contact and constraint buffers", "MuJoCo penetration"]
    if cpu:
        data = solver.mj_data
        arrays = {name: np.asarray(getattr(data, name))[None] for name in ("qpos", "qvel", "qacc")}
        count = int(data.ncon)
        contact = data.contact
        geoms, dist = np.asarray(contact.geom)[:count], np.asarray(contact.dist)[:count]
        contact_worlds = np.zeros(count, dtype=int)
        report["unsupported"].append("MuJoCo CPU backend: buffers grow dynamically; capacities are not checked")
    else:
        data = solver.mjw_data
        arrays = {name: getattr(data, name).numpy() for name in ("qpos", "qvel", "qacc")}
        nacon, capacity = int(data.nacon.numpy()[0]), int(data.naconmax)
        count = min(nacon, capacity)
        stats["solver_contacts"], stats["solver_contact_capacity"] = nacon, capacity
        if nacon >= capacity:
            warnings.append(
                f"MuJoCo contact buffer full: nacon {nacon} >= naconmax {capacity} (shared by all worlds, "
                "SolverMuJoCo nconmax per world); contacts beyond it are dropped"
            )
        geoms = data.contact.geom.numpy()[:count]
        dist = data.contact.dist.numpy()[:count]
        contact_worlds = data.contact.worldid.numpy()[:count]
        nworld = int(data.nworld)
        if per_world and count:
            per = np.bincount(contact_worlds, minlength=nworld)
            stats["max_contacts_per_world"] = int(per.max())
            stats["nconmax_per_world"] = capacity // max(nworld, 1)
        nefc, njmax = data.nefc.numpy(), int(data.njmax)
        stats["solver_constraint_rows_max"], stats["njmax"] = int(nefc.max()), njmax
        full = np.flatnonzero(nefc >= njmax)
        if len(full):
            worlds["constraint_buffer_full"] = full[:64].tolist()
            warnings.append(f"MuJoCo constraint rows reached njmax {njmax} in worlds {_worlds_text(full)}")
        overflow = getattr(data, "overflow", None)
        if overflow is not None:
            _overflow_flags(overflow.numpy(), report)
        _overflow_counts(solver, stats)
        if getattr(data, "solver_niter", None) is not None:
            niter = data.solver_niter.numpy()
            cap = int(solver.mj_model.opt.iterations)
            stats["solver_iterations_max"], stats["solver_iterations_cap"] = int(niter.max()), cap
            capped = np.flatnonzero(niter >= cap)
            if per_world and len(capped):
                worlds["iteration_cap"] = capped[:64].tolist()
    for name, values in arrays.items():
        bad = np.flatnonzero(~np.isfinite(values.reshape(values.shape[0], -1)).all(axis=1))
        if len(bad):
            worlds.setdefault("nonfinite", set()).update(bad.tolist())
            warnings.append(f"MuJoCo {name} non-finite in worlds {_worlds_text(bad)}")
    if count:
        geom_map = solver.mjc_geom_to_newton_shape.numpy()
        rows = np.minimum(contact_worlds, geom_map.shape[0] - 1)
        first = np.where(geoms[:, 0] >= 0, geom_map[rows, np.maximum(geoms[:, 0], 0)], -1)
        second = np.where(geoms[:, 1] >= 0, geom_map[rows, np.maximum(geoms[:, 1], 0)], -1)
        static, filtered = _skipped_pairs(model, first, second)
        _skip_note(stats, static, filtered)
        keep = ~(static | filtered)
        stats["deepest_penetration"] = float(max(0.0, -dist[keep].min())) if keep.any() else 0.0
        _penetration_pairs(
            report["_pairs"], -dist[keep], first[keep], second[keep], contact_worlds[keep], labels, threshold
        )


_ITERATION_FLAGS = ("ITERATIONS", "LS_ITERATIONS")


def _overflow_flags(flags: np.ndarray, report: dict) -> None:
    """Per-world MuJoCo Warp overflow bits; they stay set from data creation until a data reset."""
    raised = np.flatnonzero(flags)
    if not len(raised):
        return
    import mujoco_warp

    names = {int(bit): bit.name for bit in mujoco_warp.OverflowType}
    per_world = {int(w): [name for bit, name in names.items() if int(flags[w]) & bit] for w in raised}
    report["worlds"]["overflow_flags"] = {str(w): per_world[w] for w in list(per_world)[:64]}
    by_flag: dict[str, list[int]] = {}
    for world, raised_names in per_world.items():
        for name in raised_names:
            by_flag.setdefault(name, []).append(world)
    buffers = {name: worlds for name, worlds in by_flag.items() if name not in _ITERATION_FLAGS}
    if buffers:
        listed = "; ".join(f"{name} in worlds {_worlds_text(worlds)}" for name, worlds in buffers.items())
        report["warnings"].append(f"MuJoCo Warp overflow flags (set since the solver data was created): {listed}")
    for name in _ITERATION_FLAGS:
        if name in by_flag:
            report["stats"][f"worlds_flagged_{name.lower()}"] = len(by_flag[name])


def _overflow_counts(solver, stats: dict) -> None:
    """(world, step) pairs per MuJoCo Warp overflow type since the solver was created (``SolverMuJoCo.step``)."""
    counts = getattr(solver, "_overflow_counts", None)
    if counts is None:
        return
    import mujoco_warp

    names = {int(flag).bit_length() - 1: flag.name for flag in mujoco_warp.OverflowType}
    raised = {names.get(bit, f"bit {bit}"): int(count) for bit, count in enumerate(counts.numpy()) if count}
    if raised:
        stats["overflow_counts"] = raised


def _newton_contacts_health(model, state, contacts, report: dict, *, threshold: float) -> None:
    from ..sim.contact_kinematics import eval_rigid_contact_kinematics  # noqa: PLC0415

    report["checked"].append("collision-pipeline contacts: buffer and penetration")
    count = int(contacts.rigid_contact_count.numpy()[0])
    capacity = int(contacts.rigid_contact_max)
    report["stats"]["contacts"], report["stats"]["contact_capacity"] = count, capacity
    if capacity and count >= capacity:
        report["warnings"].append(f"Newton contact buffer full: {count} >= rigid_contact_max {capacity}")
    count = min(count, capacity)
    if not count:
        return
    import warp as wp  # noqa: PLC0415

    distance = wp.empty(capacity, dtype=float, device=model.device)
    point0 = wp.empty(capacity, dtype=wp.vec3, device=model.device)
    point1 = wp.empty_like(point0)
    eval_rigid_contact_kinematics(
        model, state, contacts, out_distance=distance, out_point0_world=point0, out_point1_world=point1
    )
    depth = -distance.numpy()[:count]
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_world = model.shape_world.numpy()
    worlds = np.maximum(shape_world[np.maximum(shape0, 0)], shape_world[np.maximum(shape1, 0)])
    static, filtered = _skipped_pairs(model, shape0, shape1)
    _skip_note(report["stats"], static, filtered)
    keep = ~(static | filtered)
    report["stats"]["deepest_penetration"] = float(max(0.0, depth[keep].max())) if keep.any() else 0.0
    _penetration_pairs(
        report["_pairs"],
        depth[keep],
        shape0[keep],
        shape1[keep],
        worlds[keep],
        getattr(model, "shape_label", None),
        threshold,
    )


def _twins(model, state, initial: dict | None, report: dict, tolerance: float) -> None:
    """Flag worlds whose joint state deviates from the per-coordinate median over all worlds."""
    if model.world_count < 2 or not model.joint_count:
        report["unsupported"].append("twins: needs at least two worlds with joints")
        return
    deviation = np.zeros(model.world_count)
    compared = []
    for name, kind in (("joint_q", "coord"), ("joint_qd", "dof")):
        array = getattr(state, name, None)
        if array is None or not array.size:
            continue
        values = array.numpy().astype(np.float64)
        if name == "joint_q":
            reference = initial.get(name) if initial else None
            if reference is None or reference.shape != values.shape:
                continue
            # Displacement from the start removes per-world placement offsets of the roots.
            values = values - reference
        row_world = _row_worlds(model, kind)
        local = row_world >= 0
        counts = np.bincount(row_world[local], minlength=model.world_count)
        if counts.min() != counts.max():
            report["unsupported"].append("twins: worlds are not structurally identical")
            return
        per_world = values[local][np.argsort(row_world[local], kind="stable")].reshape(model.world_count, -1)
        spread = np.abs(per_world - np.median(per_world, axis=0))
        deviation = np.maximum(deviation, np.nan_to_num(spread, nan=np.inf).max(axis=1))
        compared.append(name)
    if not compared:
        report["unsupported"].append("twins: no joint state to compare")
        return
    report["checked"].append(f"twins: {' and '.join(compared)} against the median over worlds")
    report["stats"]["twins_max_deviation"] = float(deviation.max())
    outliers = np.flatnonzero(deviation > tolerance)
    if len(outliers):
        report["worlds"]["twins_disagree"] = outliers[np.argsort(-deviation[outliers])][:64].tolist()
        report["warnings"].append(
            f"worlds {_worlds_text(outliers)} deviate from the median over worlds by more than {tolerance:g} "
            f"(max {deviation.max():.3g}) in {' or '.join(compared)}"
        )


def health(
    session,
    solver=None,
    state=None,
    *,
    per_world: bool = True,
    twins: bool = False,
    penetration: float = 0.01,
    twins_tolerance: float = 1e-3,
    limit: int = 8,
) -> dict:
    """:func:`health_report` for a live session: its solver, state, contacts, and initial state by default.

    Args:
        session: Live :class:`SimulationSession`.
        solver: Solver to inspect (default: the session's).
        state: State to inspect (default: the session's).
        per_world: See :func:`health_report`.
        twins: See :func:`health_report`.
        penetration: See :func:`health_report`.
        twins_tolerance: See :func:`health_report`.
        limit: See :func:`health_report`.
    """
    solver = session.solver if solver is None else solver
    state = session.state if state is None else state
    return health_report(
        getattr(solver, "model", None) or session.model,
        state,
        solver,
        contacts=session.contacts if solver is session.solver else None,
        initial=session._initial.get("state") if state is session.state else None,
        per_world=per_world,
        twins=twins,
        penetration=penetration,
        twins_tolerance=twins_tolerance,
        limit=limit,
    )


def health_report(
    model,
    state,
    solver=None,
    *,
    contacts=None,
    initial: dict | None = None,
    per_world: bool = True,
    twins: bool = False,
    penetration: float = 0.01,
    twins_tolerance: float = 1e-3,
    limit: int = 8,
) -> dict:
    """Check for non-finite state, runaway velocities, solver buffer overflow, and deep penetration.

    Args:
        model: Model of ``state`` (and of ``solver``).
        state: State to inspect.
        solver: Solver to inspect, or ``None``. Any solver works; MuJoCo solvers add checks of
            their own data and buffers.
        contacts: Collision-pipeline contacts the solver uses (checked for solvers without their own
            contact buffers), or ``None``.
        initial: Joint coordinates at the start (``{"joint_q": array}``), which ``twins`` compares
            displacements against; without it only velocities are compared.
        per_world: Name the worlds behind each finding.
        twins: Also compare the worlds' joint states with each other, for scenes whose worlds were
            built identical: worlds deviating from the per-coordinate median are listed.
        penetration: Overlap [m] above which contacts are reported by shape pair. Contacts between
            two static shapes and between shapes the model filters from colliding are skipped (see
            :func:`_skipped_pairs`).
        twins_tolerance: Deviation [m, rad, m/s or rad/s] above which a world counts as disagreeing.
        limit: Maximum number of shape pairs listed.

    Returns:
        ``{"ok", "warnings", "stats", "checked", "worlds", "penetration", "unsupported"}``; ``checked``
        lists what was inspected and ``unsupported`` what could not be for this solver.
    """
    report = {
        "warnings": [],
        "stats": {},
        "checked": ["state: non-finite values and body speeds"],
        "worlds": {},
        "unsupported": [],
        "_pairs": {},
    }
    warnings, stats, worlds = report["warnings"], report["stats"], report["worlds"]
    for name, kind in _STATE_ROWS.items():
        array = getattr(state, name, None)
        if array is None or array.size == 0:
            continue
        values = array.numpy()
        bad = ~np.isfinite(values.reshape(values.shape[0], -1)).all(axis=1)
        if not bad.any():
            continue
        row_world = _row_worlds(model, kind) if per_world else None
        if row_world is not None and len(row_world) == len(values):
            affected = np.unique(row_world[bad])
            worlds.setdefault("nonfinite", set()).update(affected.tolist())
            warnings.append(f"state.{name} non-finite in {int(bad.sum())} rows, worlds {_worlds_text(affected)}")
        else:
            warnings.append(f"state.{name} contains non-finite values")
    if getattr(state, "body_qd", None) is not None and state.body_qd.size:
        qd = state.body_qd.numpy()
        linear = np.nan_to_num(np.linalg.norm(qd[:, :3], axis=1), nan=0.0)
        angular = np.nan_to_num(np.linalg.norm(qd[:, 3:], axis=1), nan=0.0)
        stats["max_body_speed"] = float(linear.max())
        stats["max_body_angular_speed"] = float(angular.max())
        fastest = int(np.argmax(linear))
        stats["fastest_body"] = _label(getattr(model, "body_label", None), fastest)
        runaway = (linear > 50.0) | (angular > 200.0)
        if runaway.any():
            body_world = model.body_world.numpy()
            affected = np.unique(body_world[runaway])
            if per_world:
                worlds["runaway"] = affected[:64].tolist()
            warnings.append(
                f"runaway body velocity (max {stats['max_body_speed']:.3g} m/s at {stats['fastest_body']}) in worlds "
                f"{_worlds_text(affected)}"
            )
    if getattr(solver, "mjw_model", None) is not None and hasattr(solver, "mjc_geom_to_newton_shape"):
        _mujoco_health(model, solver, report, per_world=per_world, threshold=penetration, limit=limit)
    else:
        name = type(solver).__name__ if solver is not None else "No solver given"
        report["unsupported"].append(f"{name}: no solver contact or constraint buffers to check")
        if contacts is not None and getattr(contacts, "rigid_contact_max", 0):
            _newton_contacts_health(model, state, contacts, report, threshold=penetration)
    if twins:
        _twins(model, state, initial, report, twins_tolerance)
    pairs = sorted(report.pop("_pairs").values(), key=lambda entry: -entry["depth"])
    if pairs:
        report["penetration"] = [
            {"shapes": p["shapes"], "depth": round(p["depth"], 6), "worlds": sorted(p["worlds"])[:16]}
            for p in pairs[:limit]
        ]
        listed = "; ".join(f"{p['shapes'][0]} | {p['shapes'][1]} {p['depth'] * 1000:.1f} mm" for p in pairs[:3])
        warnings.append(f"deep penetration (> {penetration * 1000:g} mm) in {len(pairs)} shape pairs: {listed}")
    if "nonfinite" in worlds:
        worlds["nonfinite"] = sorted(worlds["nonfinite"])[:64]
    if not per_world:
        report.pop("worlds")
    elif not worlds:
        report.pop("worlds")
    if not report["unsupported"]:
        report.pop("unsupported")
    return {"ok": not warnings, **report}


def _final(label: str | None) -> str:
    return label.rsplit("/", 1)[-1] if label else ""


def _select_shapes(model, selector) -> np.ndarray:
    """Boolean mask over shapes for a selector.

    Selectors: a label substring (matched against the last path component of each shape's
    label and of its body's label), a shape index, ``{"shape": index or substring}``,
    ``{"body": index or substring}``, or a list of these.
    """
    count = model.shape_count
    mask = np.zeros(count, dtype=bool)
    shape_body = model.shape_body.numpy() if count else np.zeros(0, dtype=int)
    shape_labels = [_final(label) for label in (model.shape_label or [""] * count)]
    body_labels = [_final(label) for label in (model.body_label or [""] * model.body_count)]
    items = selector if isinstance(selector, list) else [selector]
    for item in items:
        if isinstance(item, bool):
            raise ValueError("contact selectors are labels, shape indices, or {'shape'|'body': ...} dictionaries")
        if isinstance(item, (int, np.integer)):
            if not 0 <= int(item) < count:
                raise ValueError(f"shape index {item} out of range")
            mask[int(item)] = True
        elif isinstance(item, str):
            for i in range(count):
                body = int(shape_body[i])
                if item in shape_labels[i] or (body >= 0 and item in body_labels[body]):
                    mask[i] = True
        elif isinstance(item, dict) and len(item) == 1 and next(iter(item)) in ("shape", "body"):
            kind, value = next(iter(item.items()))
            if kind == "shape":
                if isinstance(value, str):
                    mask |= np.array([value in label for label in shape_labels], dtype=bool)
                else:
                    mask[int(value)] = True
            else:
                bodies = (
                    [b for b, label in enumerate(body_labels) if value in label]
                    if isinstance(value, str)
                    else [int(value)]
                )
                mask |= np.isin(shape_body, bodies)
        else:
            raise ValueError(f"unsupported contact selector {item!r}")
    if not mask.any():
        raise ValueError(f"selector {selector!r} matches no shape")
    return mask


def _contact_rows(session) -> dict:
    """Current contacts with world points, normals (shape 0 to 1), distances, and solver forces when available."""
    model, solver, state = session.model, session.solver, session.state
    data = getattr(solver, "mjw_data", None)
    if data is not None and hasattr(solver, "update_contacts"):
        import newton  # noqa: PLC0415

        count = int(data.nacon.numpy()[0])
        report = getattr(session, "_contact_report", None)
        if report is None or report.rigid_contact_max < data.naconmax:
            report = newton.Contacts(data.naconmax, 0, requested_attributes={"force"}, device=model.device)
            session._contact_report = report
        solver.update_contacts(report, state)
        count = min(count, report.rigid_contact_max)
        return {
            "source": "solver (MuJoCo contact set)",
            "shape0": report.rigid_contact_shape0.numpy()[:count],
            "shape1": report.rigid_contact_shape1.numpy()[:count],
            "normal": report.rigid_contact_normal.numpy()[:count],
            "point": data.contact.pos.numpy()[:count],
            "distance": data.contact.dist.numpy()[:count],
            "force": report.force.numpy()[:count, :3],
        }
    contacts = session.contacts
    if contacts is None:
        raise ValueError("No contacts available: the scene exposes neither solver contacts nor Newton Contacts")
    import warp as wp  # noqa: PLC0415

    from ..sim.contact_kinematics import eval_rigid_contact_kinematics  # noqa: PLC0415

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=model.device)
    point0 = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
    point1 = wp.empty_like(point0)
    eval_rigid_contact_kinematics(
        model, state, contacts, out_distance=distance, out_point0_world=point0, out_point1_world=point1
    )
    force = None
    if solver is not None and hasattr(solver, "update_contacts"):
        try:
            solver.update_contacts(contacts, state)
            if contacts.force is not None:
                force = contacts.force.numpy()[:count, :3]
            elif contacts.rigid_contact_force is not None:
                force = contacts.rigid_contact_force.numpy()[:count]
        except NotImplementedError:
            force = None
    return {
        "source": "collision pipeline",
        "shape0": contacts.rigid_contact_shape0.numpy()[:count],
        "shape1": contacts.rigid_contact_shape1.numpy()[:count],
        "normal": contacts.rigid_contact_normal.numpy()[:count],
        "point": 0.5 * (point0.numpy()[:count] + point1.numpy()[:count]),
        "distance": distance.numpy()[:count],
        "force": force,
    }


def contacts_between(session, a, b=None, *, detail: bool = False) -> dict:
    """Contact summary between shape sets ``a`` and ``b`` (or ``a`` and everything else) at the current state.

    Forces are those the solver applied in its last step (:meth:`~newton.solvers.SolverBase.update_contacts`),
    as seen by ``a``: ``normal_force`` [N] is the summed compressive normal force and
    ``tangential_force`` [N] the summed friction force. ``slip_max``/``slip_mean`` [m/s] are the
    relative tangential speeds at the contact points (zero for a firm grasp), ``penetration`` [m]
    is the deepest overlap and ``gap`` [m] the smallest signed distance. Contacts with a nonpositive
    distance count as ``touching``.

    Args:
        session: Live :class:`SimulationSession`.
        a: Selector for the first set (see :func:`_select_shapes`).
        b: Selector for the second set; ``None`` means every other shape, including static ones.
        detail: Also list contact counts and normal forces per body pair (``by_body``).

    Returns:
        Flat dictionary of scalars (``nan`` where the source has no data), plus ``by_body`` with ``detail``.
    """
    model, state = session.model, session.state
    in_a = _select_shapes(model, a)
    in_b = _select_shapes(model, b) if b is not None else ~in_a
    rows = _contact_rows(session)
    s0, s1 = rows["shape0"].astype(int), rows["shape1"].astype(int)
    valid = (s0 >= 0) & (s1 >= 0) & (s0 < model.shape_count) & (s1 < model.shape_count)
    in_a_ext = np.append(in_a, False)
    in_b_ext = np.append(in_b, b is None)
    s0_ = np.where(valid, s0, model.shape_count)
    s1_ = np.where(valid, s1, model.shape_count)
    forward = in_a_ext[s0_] & in_b_ext[s1_]
    backward = in_a_ext[s1_] & in_b_ext[s0_] & ~forward
    selected = np.flatnonzero(forward | backward)
    sign = np.where(backward[selected], -1.0, 1.0)[:, None]
    normal = rows["normal"][selected] * sign  # from a toward b
    distance = rows["distance"][selected] if rows["distance"] is not None else None
    touching = distance <= 0.0 if distance is not None else np.ones(len(selected), dtype=bool)
    result = {"count": int(len(selected)), "touching": int(touching.sum())}
    nan = float("nan")
    if rows["force"] is not None and len(selected):
        force = rows["force"][selected] * sign  # on a's body by b's body
        along = np.einsum("ij,ij->i", force, normal)
        result["normal_force"] = float(np.clip(-along, 0.0, None).sum())
        result["tangential_force"] = float(np.linalg.norm(force - along[:, None] * normal, axis=1).sum())
    else:
        result["normal_force"] = result["tangential_force"] = nan if rows["force"] is None else 0.0
    if len(selected):
        shape_body = model.shape_body.numpy()
        side_a = np.where(backward[selected], s1[selected], s0[selected])
        side_b = np.where(backward[selected], s0[selected], s1[selected])
        points = rows["point"][selected]
        velocity = _point_velocity(model, state, shape_body[side_b], points) - _point_velocity(
            model, state, shape_body[side_a], points
        )
        tangential = velocity - np.einsum("ij,ij->i", velocity, normal)[:, None] * normal
        slip = np.linalg.norm(tangential, axis=1)
        result["slip_max"] = float(slip[touching].max()) if touching.any() else 0.0
        result["slip_mean"] = float(slip[touching].mean()) if touching.any() else 0.0
    else:
        result["slip_max"] = result["slip_mean"] = 0.0
    if distance is not None and len(selected):
        result["penetration"] = float(max(0.0, -distance.min()))
        result["gap"] = float(distance.min())
    else:
        result["penetration"] = 0.0 if distance is not None else nan
        result["gap"] = nan
    if detail:
        body_labels = model.body_label or []
        shape_body = model.shape_body.numpy()
        pairs: dict[str, dict] = {}
        for k, index in enumerate(selected):
            first, second = (s1[index], s0[index]) if backward[index] else (s0[index], s1[index])
            key = f"{_final(_label(body_labels, int(shape_body[first])))} | {_final(_label(body_labels, int(shape_body[second])))}"
            entry = pairs.setdefault(key, {"count": 0, "touching": 0, "normal_force": 0.0})
            entry["count"] += 1
            entry["touching"] += int(touching[k])
            if rows["force"] is not None:
                f = rows["force"][index] * sign[k]
                entry["normal_force"] += float(max(0.0, -np.dot(f, normal[k])))
        result["by_body"] = pairs
        result["source"] = rows["source"]
    return result


def _point_velocity(model, state, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """World velocity [m/s] of the material point of each body at ``points``; zero for static shapes (body -1)."""
    out = np.zeros_like(points, dtype=np.float64)
    moving = bodies >= 0
    if not moving.any() or state.body_qd is None:
        return out
    body_q = state.body_q.numpy()[bodies[moving]]
    body_qd = state.body_qd.numpy()[bodies[moving]]
    com_local = model.body_com.numpy()[bodies[moving]]
    position, quaternion = body_q[:, :3], body_q[:, 3:]
    u, w = quaternion[:, :3], quaternion[:, 3:4]
    t = 2.0 * np.cross(u, com_local)
    com_world = position + com_local + w * t + np.cross(u, t)
    out[moving] = body_qd[:, :3] + np.cross(body_qd[:, 3:], points[moving] - com_world)
    return out
