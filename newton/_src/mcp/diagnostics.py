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
