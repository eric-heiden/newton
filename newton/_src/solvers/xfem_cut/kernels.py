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
def _cut_path_center_y(
    x: float,
    center_y: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    if wp.abs(path_amplitude_y) <= 1.0e-12:
        return center_y
    wavelength = wp.max(wp.abs(path_wavelength_x), 1.0e-6)
    phase = 6.28318530718 * (x - path_origin_x) / wavelength + path_phase
    return center_y + path_amplitude_y * wp.sin(phase)


@wp.func
def _cut_path_slope_y(
    x: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    if wp.abs(path_amplitude_y) <= 1.0e-12:
        return float(0.0)
    wavelength = wp.max(wp.abs(path_wavelength_x), 1.0e-6)
    phase = 6.28318530718 * (x - path_origin_x) / wavelength + path_phase
    return path_amplitude_y * (6.28318530718 / wavelength) * wp.cos(phase)


@wp.func
def _cut_path_tangent_xy(
    x: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    slope = _cut_path_slope_y(x, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    return _safe_normalized(wp.vec3(1.0, slope, 0.0))


@wp.func
def _cut_path_normal_xy(
    x: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    tangent = _cut_path_tangent_xy(x, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    return _safe_normalized(wp.vec3(-tangent[1], tangent[0], 0.0))


@wp.func
def _cut_path_signed_y(
    q: wp.vec3,
    center_y: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    center = _cut_path_center_y(
        q[0],
        center_y,
        path_amplitude_y,
        path_wavelength_x,
        path_phase,
        path_origin_x,
    )
    slope = _cut_path_slope_y(q[0], path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    return (q[1] - center) / wp.sqrt(1.0 + slope * slope)


@wp.func
def _knife_edge_process_weight(
    q: wp.vec3,
    edge_points: wp.array[wp.vec3],
    edge_point_count: int,
    center_y: float,
    half_width_y: float,
    process_width: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
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

    signed_distance = _cut_path_signed_y(
        q,
        center_y,
        path_amplitude_y,
        path_wavelength_x,
        path_phase,
        path_origin_x,
    )
    y_out = wp.max(0.0, wp.abs(signed_distance) - half_width_y)
    distance = wp.sqrt(best_d2 + y_out * y_out)
    return wp.max(0.0, 1.0 - distance / wp.max(process_width, 1.0e-6))


@wp.kernel
def apply_xfem_knife_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_damage: wp.array[float],
    particle_cut_side: wp.array[float],
    particle_enrichment_q: wp.array[wp.vec3],
    particle_enrichment_qd: wp.array[wp.vec3],
    particle_colors: wp.array[wp.vec3],
    force_accum: wp.array[float],
    knife_edge_points: wp.array[wp.vec3],
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
    max_knife_velocity_delta: float,
    knife_velocity: wp.vec3,
    knife_tangent: wp.vec3,
    max_enrichment: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    q = particle_q[tid]
    v = particle_qd[tid]
    old_damage = particle_damage[tid]
    signed_distance = _cut_path_signed_y(q, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    z_rel = q[2] - center_z

    front_weight = _knife_edge_process_weight(
        q,
        knife_edge_points,
        knife_edge_point_count,
        center_y,
        half_width_y,
        process_width,
        path_amplitude_y,
        path_wavelength_x,
        path_phase,
        path_origin_x,
    )
    in_cut_wake = q[0] <= front_x + process_width and wp.abs(z_rel) <= half_width_z
    active = front_weight > 0.0

    side = _signed_side(signed_distance)
    if active or (old_damage > 0.0 and in_cut_wake):
        particle_cut_side[tid] = side

    tangent = _safe_normalized(knife_tangent)
    normal_dir = _cut_path_normal_xy(q[0], path_amplitude_y, path_wavelength_x, path_phase, path_origin_x) * side
    new_damage = old_damage
    delta_damage = float(0.0)
    normal_force = float(0.0)
    if active:
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

    wake_weight = _xfem_smoothstep((front_x + process_width - q[0]) / wp.max(process_width, 1.0e-6))
    side_wall_distance = wp.abs(signed_distance)
    near_cut_wall = _xfem_smoothstep(
        (half_width_y + process_width - side_wall_distance) / wp.max(process_width, 1.0e-6)
    )
    tangential_coupling = wp.max(front_weight, wake_weight * near_cut_wall * _xfem_smoothstep(new_damage))
    if tangential_coupling > 0.0:
        rel_tangent_speed = wp.dot(v - knife_velocity, tangent)
        friction_force_signed = (
            -knife_friction_mu
            * wp.abs(normal_force)
            * wp.tanh(rel_tangent_speed / wp.max(friction_velocity_scale, 1.0e-6))
        )
        force_scale_factor = float(1.0)
        applied_force = normal_dir * normal_force + tangent * friction_force_signed
        if particle_inv_mass[tid] > 0.0 and max_knife_velocity_delta > 0.0:
            max_force = max_knife_velocity_delta / (particle_inv_mass[tid] * wp.max(dt, 1.0e-6))
            applied_force_length = wp.length(applied_force)
            if applied_force_length > max_force:
                force_scale_factor = max_force / wp.max(applied_force_length, 1.0e-8)
                applied_force = applied_force * force_scale_factor

        tangent_speed_delta = wp.dot(knife_velocity, tangent) - wp.dot(v, tangent)
        friction_drag_fraction = wp.min(1.0, knife_friction_mu * tangential_coupling * 85.0 * dt)
        friction_drag_velocity = tangent * (tangent_speed_delta * friction_drag_fraction)
        direct_velocity_delta = normal_dir * (separation_speed * delta_damage) + friction_drag_velocity
        velocity_delta = direct_velocity_delta
        direct_velocity_scale = float(1.0)

        particle_f[tid] = particle_f[tid] + applied_force
        drag_force_equiv = float(0.0)
        if particle_inv_mass[tid] > 0.0:
            force_velocity_delta = applied_force * particle_inv_mass[tid] * dt
            velocity_delta = velocity_delta + force_velocity_delta
            if max_knife_velocity_delta > 0.0:
                velocity_delta_length = wp.length(velocity_delta)
                if velocity_delta_length > max_knife_velocity_delta:
                    direct_velocity_scale = max_knife_velocity_delta / velocity_delta_length
                    velocity_delta = velocity_delta * direct_velocity_scale

            drag_force_equiv = (
                wp.length(direct_velocity_delta) * direct_velocity_scale / (particle_inv_mass[tid] * wp.max(dt, 1.0e-6))
            )
            particle_qd[tid] = v + velocity_delta

        wp.atomic_add(
            force_accum,
            0,
            force_scale_factor * (wp.abs(normal_force) + wp.abs(friction_force_signed)) + drag_force_equiv,
        )
        wp.atomic_add(force_accum, 1, tangential_coupling)
        wp.atomic_add(force_accum, 3, force_scale_factor * wp.abs(normal_force))
        wp.atomic_add(force_accum, 4, force_scale_factor * wp.abs(friction_force_signed) + drag_force_equiv)

    if active:
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
    particle_q: wp.array[wp.vec3],
    particle_damage: wp.array[float],
    particle_cut_side: wp.array[float],
    tet_indices: wp.array2d[wp.int32],
    tet_cut_state: wp.array[wp.int32],
    tet_damage: wp.array[float],
    tet_cut_weight: wp.array[float],
    knife_edge_points: wp.array[wp.vec3],
    knife_edge_point_count: int,
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_y: float,
    half_width_z: float,
    process_width: float,
    damage_threshold: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
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

    yi = _cut_path_signed_y(qi, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    yj = _cut_path_signed_y(qj, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    yk = _cut_path_signed_y(qk, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    yl = _cut_path_signed_y(ql, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
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
        path_amplitude_y,
        path_wavelength_x,
        path_phase,
        path_origin_x,
    )
    z_in = front_weight > 0.0 or wp.abs(centroid[2] - center_z) <= half_width_z
    wake_weight = _xfem_smoothstep((front_x + process_width - centroid[0]) / wp.max(process_width, 1.0e-6))
    weight = wp.max(front_weight, wake_weight)

    mean_damage = 0.25 * (particle_damage[i] + particle_damage[j] + particle_damage[k] + particle_damage[l])
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
    tet_cut_state: wp.array[wp.int32],
    tet_damage: wp.array[float],
    tet_materials: wp.array2d[float],
    base_tet_materials: wp.array2d[float],
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


@wp.func
def _rest_segment_crosses_cloth_cut(
    qi: wp.vec3,
    qj: wp.vec3,
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_z: float,
    process_width: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    yi = _cut_path_signed_y(qi, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    yj = _cut_path_signed_y(qj, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    crosses = yi * yj < 0.0
    mid = (qi + qj) * 0.5
    reached = mid[0] <= front_x + process_width
    z_in = wp.abs(mid[2] - center_z) <= half_width_z + process_width
    return crosses and reached and z_in


@wp.func
def _rest_triangle_crosses_cloth_cut(
    q0: wp.vec3,
    q1: wp.vec3,
    q2: wp.vec3,
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_z: float,
    process_width: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    y0 = _cut_path_signed_y(q0, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    y1 = _cut_path_signed_y(q1, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    y2 = _cut_path_signed_y(q2, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    min_y = wp.min(y0, wp.min(y1, y2))
    max_y = wp.max(y0, wp.max(y1, y2))
    centroid = (q0 + q1 + q2) / 3.0
    reached = centroid[0] <= front_x + process_width
    z_in = wp.abs(centroid[2] - center_z) <= half_width_z + process_width
    return min_y < 0.0 and max_y > 0.0 and reached and z_in


@wp.kernel
def cut_xfem_cloth_springs_kernel(
    rest_particle_q: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    base_spring_stiffness: wp.array[float],
    base_spring_damping: wp.array[float],
    spring_cut_state: wp.array[wp.int32],
    cut_counts: wp.array[wp.int32],
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_z: float,
    process_width: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    qi = rest_particle_q[i]
    qj = rest_particle_q[j]
    mid = (qi + qj) * 0.5
    zero_rest_seam = wp.length(qi - qj) <= 1.0e-7
    seam_reached = (
        zero_rest_seam
        and mid[0] <= front_x + process_width
        and wp.abs(mid[2] - center_z) <= half_width_z + process_width
        and wp.abs(_cut_path_signed_y(mid, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x))
        <= process_width
    )
    should_cut = seam_reached or _rest_segment_crosses_cloth_cut(
        qi,
        qj,
        front_x,
        center_y,
        center_z,
        half_width_z,
        process_width,
        path_amplitude_y,
        path_wavelength_x,
        path_phase,
        path_origin_x,
    )
    if should_cut and spring_cut_state[tid] == 0:
        spring_cut_state[tid] = 1
        wp.atomic_add(cut_counts, 0, 1)

    if spring_cut_state[tid] != 0:
        spring_stiffness[tid] = 0.0
        spring_damping[tid] = 0.0
    else:
        spring_stiffness[tid] = base_spring_stiffness[tid]
        spring_damping[tid] = base_spring_damping[tid]


@wp.kernel
def cut_xfem_cloth_edges_kernel(
    rest_particle_q: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_bending_properties: wp.array2d[float],
    base_edge_bending_properties: wp.array2d[float],
    edge_cut_state: wp.array[wp.int32],
    cut_counts: wp.array[wp.int32],
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_z: float,
    process_width: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    tid = wp.tid()
    i = edge_indices[tid, 0]
    j = edge_indices[tid, 1]
    k = edge_indices[tid, 2]
    l = edge_indices[tid, 3]
    if i < 0 or j < 0 or k < 0 or l < 0:
        return

    qi = rest_particle_q[i]
    qj = rest_particle_q[j]
    qk = rest_particle_q[k]
    ql = rest_particle_q[l]
    yi = _cut_path_signed_y(qi, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    yj = _cut_path_signed_y(qj, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    yk = _cut_path_signed_y(qk, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    yl = _cut_path_signed_y(ql, center_y, path_amplitude_y, path_wavelength_x, path_phase, path_origin_x)
    min_y = wp.min(wp.min(yi, yj), wp.min(yk, yl))
    max_y = wp.max(wp.max(yi, yj), wp.max(yk, yl))
    centroid = (qi + qj + qk + ql) * 0.25
    reached = centroid[0] <= front_x + process_width
    z_in = wp.abs(centroid[2] - center_z) <= half_width_z + process_width
    should_cut = min_y < 0.0 and max_y > 0.0 and reached and z_in
    if should_cut and edge_cut_state[tid] == 0:
        edge_cut_state[tid] = 1
        wp.atomic_add(cut_counts, 1, 1)

    if edge_cut_state[tid] != 0:
        edge_bending_properties[tid, 0] = 0.0
        edge_bending_properties[tid, 1] = 0.0
    else:
        edge_bending_properties[tid, 0] = base_edge_bending_properties[tid, 0]
        edge_bending_properties[tid, 1] = base_edge_bending_properties[tid, 1]


@wp.kernel
def cut_xfem_cloth_triangles_kernel(
    rest_particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_materials: wp.array2d[float],
    base_tri_materials: wp.array2d[float],
    tri_cut_state: wp.array[wp.int32],
    cut_counts: wp.array[wp.int32],
    front_x: float,
    center_y: float,
    center_z: float,
    half_width_z: float,
    process_width: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    tid = wp.tid()
    i = tri_indices[tid, 0]
    j = tri_indices[tid, 1]
    k = tri_indices[tid, 2]
    should_cut = _rest_triangle_crosses_cloth_cut(
        rest_particle_q[i],
        rest_particle_q[j],
        rest_particle_q[k],
        front_x,
        center_y,
        center_z,
        half_width_z,
        process_width,
        path_amplitude_y,
        path_wavelength_x,
        path_phase,
        path_origin_x,
    )
    if should_cut and tri_cut_state[tid] == 0:
        tri_cut_state[tid] = 1
        wp.atomic_add(cut_counts, 2, 1)

    tri_materials[tid, 0] = base_tri_materials[tid, 0]
    tri_materials[tid, 1] = base_tri_materials[tid, 1]
    tri_materials[tid, 2] = base_tri_materials[tid, 2]
    tri_materials[tid, 3] = base_tri_materials[tid, 3]
    tri_materials[tid, 4] = base_tri_materials[tid, 4]


@wp.kernel
def enforce_xfem_cloth_seam_collision_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    rest_particle_q: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    spring_cut_state: wp.array[wp.int32],
    seam_collision_thickness: float,
    seam_collision_damping: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
):
    tid = wp.tid()
    if spring_cut_state[tid] == 0 or seam_collision_thickness <= 0.0:
        return

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    rest_i = rest_particle_q[i]
    rest_j = rest_particle_q[j]
    if wp.length(rest_i - rest_j) > 1.0e-7:
        return

    wi = float(0.0)
    wj = float(0.0)
    if (particle_flags[i] & ParticleFlags.ACTIVE) != 0 and particle_inv_mass[i] > 0.0:
        wi = particle_inv_mass[i]
    if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and particle_inv_mass[j] > 0.0:
        wj = particle_inv_mass[j]
    weight_sum = wi + wj
    if weight_sum <= 0.0:
        return

    mid_rest = (rest_i + rest_j) * 0.5
    normal = _cut_path_normal_xy(
        mid_rest[0],
        path_amplitude_y,
        path_wavelength_x,
        path_phase,
        path_origin_x,
    )

    qi = particle_q[i]
    qj = particle_q[j]
    gap = wp.dot(qj - qi, normal)
    if gap < seam_collision_thickness:
        deficit = seam_collision_thickness - gap
        qi = qi - normal * (deficit * wi / weight_sum)
        qj = qj + normal * (deficit * wj / weight_sum)

        qdi = particle_qd[i]
        qdj = particle_qd[j]
        rel_normal_velocity = wp.dot(qdj - qdi, normal)
        damping = wp.min(1.0, wp.max(0.0, seam_collision_damping))
        if rel_normal_velocity < 0.0 and damping > 0.0:
            velocity_correction = -rel_normal_velocity * damping
            qdi = qdi - normal * (velocity_correction * wi / weight_sum)
            qdj = qdj + normal * (velocity_correction * wj / weight_sum)
            particle_qd[i] = qdi
            particle_qd[j] = qdj

        particle_q[i] = qi
        particle_q[j] = qj


@wp.kernel
def apply_xfem_cloth_wind_kernel(
    particle_q: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    rest_particle_q: wp.array[wp.vec3],
    strength: float,
    frequency_hz: float,
    time: float,
    wind_direction: wp.vec3,
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[tid] <= 0.0:
        return
    rest = rest_particle_q[tid]
    phase = 6.28318530718 * frequency_hz * time + 8.0 * rest[0] + 3.0 * rest[1]
    gust = 0.55 + 0.45 * wp.sin(phase)
    particle_f[tid] = particle_f[tid] + _safe_normalized(wind_direction) * (strength * gust)


@wp.kernel
def apply_xfem_post_constraints_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    rest_particle_q: wp.array[wp.vec3],
    particle_damage: wp.array[float],
    particle_cut_side: wp.array[float],
    particle_enrichment_q: wp.array[wp.vec3],
    front_x: float,
    center_y: float,
    process_width: float,
    max_visual_gap: float,
    path_amplitude_y: float,
    path_wavelength_x: float,
    path_phase: float,
    path_origin_x: float,
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
        signed_distance = _cut_path_signed_y(
            q,
            center_y,
            path_amplitude_y,
            path_wavelength_x,
            path_phase,
            path_origin_x,
        )
        current_sep = side * signed_distance
        if current_sep < min_sep:
            normal_dir = (
                _cut_path_normal_xy(
                    q[0],
                    path_amplitude_y,
                    path_wavelength_x,
                    path_phase,
                    path_origin_x,
                )
                * side
            )
            q = q + normal_dir * (min_sep - current_sep)
            normal_velocity = wp.dot(qd, normal_dir)
            if normal_velocity < 0.0:
                qd = qd - normal_dir * normal_velocity
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
