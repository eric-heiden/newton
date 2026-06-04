# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def _xfem_smoothstep(value: float):
    x = wp.min(1.0, wp.max(0.0, value))
    return x * x * (3.0 - 2.0 * x)


@wp.func
def _safe_normalized(v: wp.vec3):
    length = wp.length(v)
    if length <= 1.0e-8:
        return wp.vec3(0.0, 0.0, 1.0)
    return v / length


@wp.func
def _signed_side(value: float):
    side = -1.0
    if value >= 0.0:
        side = 1.0
    return side


@wp.func
def _knife_edge_process_weight(
    q: wp.vec3,
    edge_points: wp.array(dtype=wp.vec3),
    edge_point_count: int,
    center_y: float,
    half_width_y: float,
    process_width: float,
):
    best_d2 = float(1.0e12)
    for i in range(edge_point_count - 1):
        a = edge_points[i]
        b = edge_points[i + 1]
        abx = b[0] - a[0]
        abz = b[2] - a[2]
        denom = abx * abx + abz * abz
        t = float(0.0)
        if denom > 1.0e-12:
            t = ((q[0] - a[0]) * abx + (q[2] - a[2]) * abz) / denom
            t = wp.min(1.0, wp.max(0.0, t))
        cx = a[0] + t * abx
        cz = a[2] + t * abz
        dx = q[0] - cx
        dz = q[2] - cz
        best_d2 = wp.min(best_d2, dx * dx + dz * dz)

    y_out = wp.max(0.0, wp.abs(q[1] - center_y) - half_width_y)
    distance = wp.sqrt(best_d2 + y_out * y_out)
    return wp.max(0.0, 1.0 - distance / wp.max(process_width, 1.0e-6))


@wp.kernel
def apply_xfem_knife_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_damage: wp.array(dtype=float),
    particle_cut_side: wp.array(dtype=float),
    particle_enrichment_q: wp.array(dtype=wp.vec3),
    particle_enrichment_qd: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=wp.vec3),
    force_accum: wp.array(dtype=float),
    knife_edge_points: wp.array(dtype=wp.vec3),
    knife_edge_point_count: int,
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_y: float,
    half_width_z: float,
    process_width: float,
    dt: float,
    particle_area: float,
    fracture_energy: float,
    yield_stress: float,
    max_damage_rate: float,
    separation_stiffness: float,
    separation_speed: float,
    force_scale: float,
    knife_friction_mu: float,
    friction_velocity_scale: float,
    knife_velocity: wp.vec3,
    knife_tangent: wp.vec3,
    max_enrichment: float,
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    q = particle_q[tid]
    v = particle_qd[tid]
    old_damage = particle_damage[tid]
    y_rel = q[1] - center_y
    z_rel = q[2] - center_z

    front_weight = _knife_edge_process_weight(
        q,
        knife_edge_points,
        knife_edge_point_count,
        center_y,
        half_width_y,
        process_width,
    )
    in_cut_wake = q[0] <= front_x + process_width and wp.abs(z_rel) <= half_width_z
    active = front_weight > 0.0

    side = _signed_side(y_rel)
    if active or (old_damage > 0.0 and in_cut_wake):
        particle_cut_side[tid] = side

    new_damage = old_damage
    if active:
        tangent = _safe_normalized(knife_tangent)
        normal_dir = wp.vec3(0.0, side, 0.0)

        saw_speed = wp.abs(wp.dot(knife_velocity, tangent))
        saw_boost = 1.0 + 0.18 * wp.min(saw_speed, 3.0)
        delta_damage = max_damage_rate * dt * front_weight * saw_boost * (1.0 - old_damage)
        new_damage = wp.min(1.0, old_damage + delta_damage)
        particle_damage[tid] = new_damage

        damage_rate = delta_damage / wp.max(dt, 1.0e-6)
        normal_force = force_scale * (
            yield_stress * particle_area * front_weight * (1.0 - old_damage)
            + fracture_energy * particle_area / wp.max(process_width, 1.0e-6) * damage_rate
        )
        rel_tangent_speed = wp.dot(v - knife_velocity, tangent)
        friction_force_signed = -knife_friction_mu * wp.abs(normal_force) * wp.tanh(
            rel_tangent_speed / wp.max(friction_velocity_scale, 1.0e-6)
        )
        friction_force = tangent * friction_force_signed
        tangent_speed_delta = wp.dot(knife_velocity, tangent) - wp.dot(v, tangent)
        friction_drag_fraction = wp.min(1.0, knife_friction_mu * front_weight * 24.0 * dt)
        friction_drag_velocity = tangent * (tangent_speed_delta * friction_drag_fraction)

        particle_f[tid] = particle_f[tid] + normal_dir * normal_force + friction_force
        drag_force_equiv = float(0.0)
        if particle_inv_mass[tid] > 0.0:
            drag_force_equiv = wp.abs(tangent_speed_delta * friction_drag_fraction) / (
                particle_inv_mass[tid] * wp.max(dt, 1.0e-6)
            )
            particle_qd[tid] = (
                v
                + normal_dir * (separation_speed * delta_damage)
                + friction_force * particle_inv_mass[tid] * dt
                + friction_drag_velocity
            )

        target_enrichment = normal_dir * (max_enrichment * _xfem_smoothstep(new_damage))
        enrich_q = particle_enrichment_q[tid]
        enrich_qd = particle_enrichment_qd[tid]
        enrich_qd = enrich_qd * 0.92 + (target_enrichment - enrich_q) * separation_stiffness * dt
        enrich_q = enrich_q + enrich_qd * dt
        enrich_len = wp.length(enrich_q)
        if enrich_len > max_enrichment:
            enrich_q = enrich_q * (max_enrichment / enrich_len)
        particle_enrichment_q[tid] = enrich_q
        particle_enrichment_qd[tid] = enrich_qd

        wp.atomic_add(force_accum, 0, wp.abs(normal_force) + wp.abs(friction_force_signed) + drag_force_equiv)
        wp.atomic_add(force_accum, 1, 1.0)
        wp.atomic_add(force_accum, 3, wp.abs(normal_force))
        wp.atomic_add(force_accum, 4, wp.abs(friction_force_signed) + drag_force_equiv)

    wp.atomic_add(force_accum, 2, new_damage)
    if new_damage > 1.0e-4:
        wp.atomic_add(force_accum, 5, 1.0)

    particle_colors[tid] = wp.vec3(
        0.18 + 0.76 * new_damage,
        0.58 * (1.0 - new_damage) + 0.18 * new_damage,
        0.42 * (1.0 - new_damage) + 0.08 * new_damage,
    )


@wp.kernel
def classify_xfem_tets_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_damage: wp.array(dtype=float),
    particle_cut_side: wp.array(dtype=float),
    tet_indices: wp.array2d(dtype=wp.int32),
    tet_cut_state: wp.array(dtype=wp.int32),
    tet_damage: wp.array(dtype=float),
    tet_cut_weight: wp.array(dtype=float),
    knife_edge_points: wp.array(dtype=wp.vec3),
    knife_edge_point_count: int,
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_y: float,
    half_width_z: float,
    process_width: float,
    damage_threshold: float,
):
    tid = wp.tid()
    i = tet_indices[tid, 0]
    j = tet_indices[tid, 1]
    k = tet_indices[tid, 2]
    l = tet_indices[tid, 3]

    qi = particle_q[i]
    qj = particle_q[j]
    qk = particle_q[k]
    ql = particle_q[l]

    yi = qi[1] - center_y
    yj = qj[1] - center_y
    yk = qk[1] - center_y
    yl = ql[1] - center_y
    min_y = wp.min(wp.min(yi, yj), wp.min(yk, yl))
    max_y = wp.max(wp.max(yi, yj), wp.max(yk, yl))
    straddles_cut = min_y <= 0.0 and max_y >= 0.0

    centroid = (qi + qj + qk + ql) * 0.25
    front_weight = _knife_edge_process_weight(
        centroid,
        knife_edge_points,
        knife_edge_point_count,
        center_y,
        half_width_y,
        process_width,
    )
    z_in = front_weight > 0.0 or wp.abs(centroid[2] - center_z) <= half_width_z
    wake_weight = _xfem_smoothstep((front_x + process_width - centroid[0]) / wp.max(process_width, 1.0e-6))
    weight = wp.max(front_weight, wake_weight)

    mean_damage = 0.25 * (
        particle_damage[i] + particle_damage[j] + particle_damage[k] + particle_damage[l]
    )
    old_tet_damage = tet_damage[tid]
    new_tet_damage = old_tet_damage
    state = tet_cut_state[tid]
    if straddles_cut and z_in and weight > 0.0:
        new_tet_damage = wp.max(old_tet_damage, mean_damage)
        if new_tet_damage >= damage_threshold:
            state = 2
        elif new_tet_damage > 1.0e-4 or front_weight > 0.0:
            state = 1

        if particle_cut_side[i] == 0.0:
            particle_cut_side[i] = _signed_side(yi)
        if particle_cut_side[j] == 0.0:
            particle_cut_side[j] = _signed_side(yj)
        if particle_cut_side[k] == 0.0:
            particle_cut_side[k] = _signed_side(yk)
        if particle_cut_side[l] == 0.0:
            particle_cut_side[l] = _signed_side(yl)

    tet_damage[tid] = new_tet_damage
    tet_cut_state[tid] = state
    tet_cut_weight[tid] = weight


@wp.kernel
def degrade_xfem_tets_kernel(
    tet_cut_state: wp.array(dtype=wp.int32),
    tet_damage: wp.array(dtype=float),
    tet_materials: wp.array2d(dtype=float),
    base_tet_materials: wp.array2d(dtype=float),
    residual_stiffness: float,
):
    tid = wp.tid()
    state = tet_cut_state[tid]
    damage = tet_damage[tid]
    softening = 1.0
    if state != 0:
        softening = residual_stiffness + (1.0 - residual_stiffness) * wp.max(0.0, 1.0 - damage)
    tet_materials[tid, 0] = base_tet_materials[tid, 0] * softening
    tet_materials[tid, 1] = base_tet_materials[tid, 1] * softening
    tet_materials[tid, 2] = base_tet_materials[tid, 2]


@wp.kernel
def apply_xfem_post_constraints_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    rest_particle_q: wp.array(dtype=wp.vec3),
    particle_damage: wp.array(dtype=float),
    particle_cut_side: wp.array(dtype=float),
    particle_enrichment_q: wp.array(dtype=wp.vec3),
    front_x: float,
    center_y: float,
    process_width: float,
    max_visual_gap: float,
    table_z: float,
    table_glue_depth: float,
    table_glue_strength: float,
    table_friction: float,
    dt: float,
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    q = particle_q[tid]
    qd = particle_qd[tid]
    rest = rest_particle_q[tid]
    inv_mass = particle_inv_mass[tid]

    damage = particle_damage[tid]
    side = particle_cut_side[tid]
    if inv_mass > 0.0 and damage > 1.0e-4 and side != 0.0 and q[0] <= front_x + 2.0 * process_width:
        min_sep = max_visual_gap * _xfem_smoothstep(damage)
        current_sep = side * (q[1] - center_y)
        if current_sep < min_sep:
            q[1] = center_y + side * min_sep
            if side * qd[1] < 0.0:
                qd[1] = 0.0
        q = q + particle_enrichment_q[tid] * (0.12 * damage)

    glue = wp.min(1.0, wp.max(0.0, table_glue_strength))
    if glue > 0.0 and rest[2] <= table_z + table_glue_depth:
        q = q * (1.0 - glue) + rest * glue
        qd = qd * (1.0 - glue)
    elif q[2] < table_z:
        q[2] = table_z
        if qd[2] < 0.0:
            qd[2] = 0.0
        damp = wp.max(0.0, 1.0 - table_friction * dt)
        qd[0] = qd[0] * damp
        qd[1] = qd[1] * damp

    particle_q[tid] = q
    particle_qd[tid] = qd
