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
    priority = None
    mujoco_attrs = getattr(model, "mujoco", None)
    if mujoco_attrs is not None and getattr(mujoco_attrs, "geom_priority", None) is not None:
        priority = mujoco_attrs.geom_priority.numpy()

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
    return {"source": source, "count": count, "pairs": rows[:limit], "pairs_truncated": len(rows) > limit}


def health(session) -> dict:
    """Quick checks for non-finite state, runaway velocities, penetration, and solver buffer overflow.

    Returns:
        ``{"ok": bool, "warnings": [...], "stats": {...}}``.
    """
    warnings, stats = [], {}
    state, model = session.state, session.model
    for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
        array = getattr(state, name, None)
        if array is None or array.size == 0:
            continue
        values = array.numpy()
        if not np.isfinite(values).all():
            warnings.append(f"state.{name} contains non-finite values")
    if getattr(state, "body_qd", None) is not None and state.body_qd.size:
        qd = state.body_qd.numpy()
        linear = np.linalg.norm(qd[:, :3], axis=1)
        angular = np.linalg.norm(qd[:, 3:], axis=1)
        stats["max_body_speed"] = float(np.nanmax(linear))
        stats["max_body_angular_speed"] = float(np.nanmax(angular))
        fastest = int(np.nanargmax(linear))
        stats["fastest_body"] = _label(getattr(model, "body_label", None), fastest)
        if stats["max_body_speed"] > 50.0 or stats["max_body_angular_speed"] > 200.0:
            warnings.append(f"runaway body velocity (max {stats['max_body_speed']:.3g} m/s at {stats['fastest_body']})")
    data = getattr(session.solver, "mjw_data", None)
    if data is not None:
        nacon = int(data.nacon.numpy()[0])
        stats["solver_contacts"] = nacon
        capacity = getattr(data, "naconmax", None)
        if capacity:
            stats["solver_contact_capacity"] = int(capacity)
            if nacon >= int(capacity):
                warnings.append("MuJoCo contact buffer full (nacon >= naconmax); raise nconmax")
        njmax = getattr(data, "njmax", None)
        if njmax and getattr(data, "nefc", None) is not None:
            nefc = int(data.nefc.numpy().max())
            stats["solver_constraint_rows"] = nefc
            if nefc >= int(njmax):
                warnings.append("MuJoCo constraint buffer full (nefc >= njmax); raise njmax")
        if nacon:
            dist = data.contact.dist.numpy()[:nacon]
            stats["deepest_penetration"] = float(max(0.0, -dist.min()))
            if -dist.min() > 0.01:
                warnings.append(f"deep penetration {-dist.min() * 1000:.1f} mm")
        if getattr(data, "solver_niter", None) is not None:
            stats["solver_iterations"] = int(data.solver_niter.numpy().max())
    return {"ok": not warnings, "warnings": warnings, "stats": stats}
