# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from ...math.spatial import velocity_at_point
from ...sim import BodyFlags


@wp.kernel
def predict_body_contact_velocities(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_flags: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    if (body_flags[tid] & BodyFlags.KINEMATIC) != 0 or body_inv_mass[tid] <= 0.0:
        body_qd_out[tid] = body_qd[tid]
        return

    q = wp.transform_get_rotation(body_q[tid])
    v0 = wp.spatial_top(body_qd[tid])
    w0 = wp.spatial_bottom(body_qd[tid])
    f0 = wp.spatial_top(body_f[tid])
    t0 = wp.spatial_bottom(body_f[tid])

    world_idx = body_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

    v1 = v0 + (f0 * body_inv_mass[tid] + world_g) * dt

    wb = wp.quat_rotate_inv(q, w0)
    tb = wp.quat_rotate_inv(q, t0) - wp.cross(wb, body_inertia[tid] * wb)
    w1 = wp.quat_rotate(q, wb + body_inv_inertia[tid] * tb * dt)

    body_qd_out[tid] = wp.spatial_vector(v1, w1)


@wp.func
def _is_dynamic_body(
    body_index: int,
    body_flags: wp.array(dtype=wp.int32),
    body_inv_mass: wp.array(dtype=float),
) -> bool:
    return body_index >= 0 and (body_flags[body_index] & BodyFlags.KINEMATIC) == 0 and body_inv_mass[body_index] > 0.0


@wp.func
def _world_inv_inertia_action(
    body_q: wp.array(dtype=wp.transform),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_index: int,
    torque_world: wp.vec3,
) -> wp.vec3:
    q = wp.transform_get_rotation(body_q[body_index])
    torque_body = wp.quat_rotate_inv(q, torque_world)
    return wp.quat_rotate(q, body_inv_inertia[body_index] * torque_body)


@wp.func
def _contact_inv_mass(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_flags: wp.array(dtype=wp.int32),
    body_index: int,
    lever_arm: wp.vec3,
    direction: wp.vec3,
) -> float:
    if not _is_dynamic_body(body_index, body_flags, body_inv_mass):
        return 0.0

    torque_world = wp.cross(lever_arm, direction)
    ang_delta = _world_inv_inertia_action(body_q, body_inv_inertia, body_index, torque_world)
    return body_inv_mass[body_index] + wp.dot(direction, wp.cross(ang_delta, lever_arm))


@wp.func
def _angular_inv_mass(
    body_q: wp.array(dtype=wp.transform),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_flags: wp.array(dtype=wp.int32),
    body_index: int,
    axis_world: wp.vec3,
) -> float:
    if not _is_dynamic_body(body_index, body_flags, body_inv_mass):
        return 0.0

    ang_delta = _world_inv_inertia_action(body_q, body_inv_inertia, body_index, axis_world)
    return wp.dot(axis_world, ang_delta)


@wp.func
def _average_shape_property(
    shape_a: int,
    shape_b: int,
    values: wp.array(dtype=float),
) -> float:
    total = 0.0
    count = 0

    if shape_a >= 0:
        total += values[shape_a]
        count += 1
    if shape_b >= 0:
        total += values[shape_b]
        count += 1

    if count == 0:
        return 0.0
    return total / float(count)


@wp.func
def _world_contact_point(
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    point_local: wp.vec3,
) -> wp.vec3:
    if body_index >= 0:
        return wp.transform_point(body_q[body_index], point_local)
    return point_local


@wp.func
def _body_com_world(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_index: int,
) -> wp.vec3:
    return wp.transform_point(body_q[body_index], body_com[body_index])


@wp.kernel
def match_contact_warmstart(
    contact_count: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    prev_contact_count: wp.array(dtype=int),
    prev_contact_point_id: wp.array(dtype=int),
    prev_contact_impulse: wp.array(dtype=wp.vec3),
    prev_contact_anchor0: wp.array(dtype=wp.vec3),
    prev_contact_anchor1: wp.array(dtype=wp.vec3),
    warmstart_scale: float,
    # outputs
    warmstart_contact_impulse: wp.array(dtype=wp.vec3),
    contact_anchor0: wp.array(dtype=wp.vec3),
    contact_anchor1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        warmstart_contact_impulse[tid] = wp.vec3()
        contact_anchor0[tid] = wp.vec3()
        contact_anchor1[tid] = wp.vec3()
        return

    point_id = contact_point_id[tid]
    current_anchor0 = contact_point0[tid] + contact_offset0[tid]
    current_anchor1 = contact_point1[tid] + contact_offset1[tid]
    if point_id < 0:
        warmstart_contact_impulse[tid] = wp.vec3()
        contact_anchor0[tid] = current_anchor0
        contact_anchor1[tid] = current_anchor1
        return

    matched_impulse = wp.vec3()
    matched_anchor0 = current_anchor0
    matched_anchor1 = current_anchor1
    prev_count = prev_contact_count[0]
    for prev_idx in range(prev_count):
        if prev_contact_point_id[prev_idx] == point_id:
            matched_impulse = prev_contact_impulse[prev_idx] * warmstart_scale
            matched_anchor0 = prev_contact_anchor0[prev_idx]
            matched_anchor1 = prev_contact_anchor1[prev_idx]
            break

    warmstart_contact_impulse[tid] = matched_impulse
    contact_anchor0[tid] = matched_anchor0
    contact_anchor1[tid] = matched_anchor1


@wp.func
def _apply_body_impulse(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_flags: wp.array(dtype=wp.int32),
    body_index: int,
    lever_arm: wp.vec3,
    impulse_world: wp.vec3,
    body_impulses: wp.array(dtype=wp.spatial_vector),
    body_delta_qd: wp.array(dtype=wp.spatial_vector),
):
    if not _is_dynamic_body(body_index, body_flags, body_inv_mass):
        return

    torque_world = wp.cross(lever_arm, impulse_world)
    delta_v = impulse_world * body_inv_mass[body_index]
    delta_w = _world_inv_inertia_action(body_q, body_inv_inertia, body_index, torque_world)
    delta_qd = wp.spatial_vector(delta_v, delta_w)
    impulse_wrench = wp.spatial_vector(impulse_world, torque_world)

    wp.atomic_add(body_delta_qd, body_index, delta_qd)
    wp.atomic_add(body_impulses, body_index, impulse_wrench)


@wp.func
def _apply_body_angular_impulse(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_flags: wp.array(dtype=wp.int32),
    body_index: int,
    angular_impulse_world: wp.vec3,
    body_impulses: wp.array(dtype=wp.spatial_vector),
    body_delta_qd: wp.array(dtype=wp.spatial_vector),
):
    if not _is_dynamic_body(body_index, body_flags, body_inv_mass):
        return

    delta_w = _world_inv_inertia_action(body_q, body_inv_inertia, body_index, angular_impulse_world)
    delta_qd = wp.spatial_vector(wp.vec3(), delta_w)
    impulse_wrench = wp.spatial_vector(wp.vec3(), angular_impulse_world)

    wp.atomic_add(body_delta_qd, body_index, delta_qd)
    wp.atomic_add(body_impulses, body_index, impulse_wrench)


@wp.kernel
def apply_warmstart_body_contact_impulses(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_flags: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_margin0: wp.array(dtype=float),
    contact_margin1: wp.array(dtype=float),
    warmstart_contact_impulse: wp.array(dtype=wp.vec3),
    separation_slop: float,
    inv_dt: float,
    # outputs
    body_impulses: wp.array(dtype=wp.spatial_vector),
    body_delta_qd: wp.array(dtype=wp.spatial_vector),
    contact_force: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    impulse = warmstart_contact_impulse[tid]
    if wp.length_sq(impulse) == 0.0:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    n = contact_normal[tid]
    if wp.length_sq(n) == 0.0:
        return

    point_a = _world_contact_point(body_q, body_a, contact_point0[tid]) + n * contact_margin0[tid]
    point_b = _world_contact_point(body_q, body_b, contact_point1[tid]) - n * contact_margin1[tid]
    separation = wp.dot(n, point_b - point_a)
    if separation > separation_slop:
        return

    com_a = point_a
    com_b = point_b
    if body_a >= 0:
        com_a = _body_com_world(body_q, body_com, body_a)
    if body_b >= 0:
        com_b = _body_com_world(body_q, body_com, body_b)

    r_a = point_a - com_a
    r_b = point_b - com_b

    _apply_body_impulse(
        body_q, body_inv_mass, body_inv_inertia, body_flags, body_a, r_a, -impulse, body_impulses, body_delta_qd
    )
    _apply_body_impulse(
        body_q, body_inv_mass, body_inv_inertia, body_flags, body_b, r_b, impulse, body_impulses, body_delta_qd
    )

    if contact_force:
        contact_force[tid] = contact_force[tid] + impulse * inv_dt


@wp.kernel
def store_contact_history(
    contact_count: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
    contact_force: wp.array(dtype=wp.vec3),
    contact_anchor0: wp.array(dtype=wp.vec3),
    contact_anchor1: wp.array(dtype=wp.vec3),
    dt: float,
    # outputs
    prev_contact_point_id: wp.array(dtype=int),
    prev_contact_impulse: wp.array(dtype=wp.vec3),
    prev_contact_anchor0: wp.array(dtype=wp.vec3),
    prev_contact_anchor1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        prev_contact_point_id[tid] = -1
        prev_contact_impulse[tid] = wp.vec3()
        prev_contact_anchor0[tid] = wp.vec3()
        prev_contact_anchor1[tid] = wp.vec3()
        return

    prev_contact_point_id[tid] = contact_point_id[tid]
    prev_contact_impulse[tid] = contact_force[tid] * dt
    prev_contact_anchor0[tid] = contact_anchor0[tid]
    prev_contact_anchor1[tid] = contact_anchor1[tid]


@wp.kernel
def eval_body_contact_impulses(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_flags: wp.array(dtype=wp.int32),
    shape_material_ka: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_material_mu_torsional: wp.array(dtype=float),
    shape_material_mu_rolling: wp.array(dtype=float),
    shape_material_restitution: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_margin0: wp.array(dtype=float),
    contact_margin1: wp.array(dtype=float),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_anchor0: wp.array(dtype=wp.vec3),
    contact_anchor1: wp.array(dtype=wp.vec3),
    rigid_contact_friction_scale: wp.array(dtype=float),
    baumgarte: float,
    penetration_slop: float,
    restitution_scale: float,
    restitution_velocity_threshold: float,
    static_friction_velocity_threshold: float,
    static_friction_scale: float,
    static_friction_anchor_gain: float,
    static_friction_anchor_break_distance: float,
    dt: float,
    inv_dt: float,
    # outputs
    body_impulses: wp.array(dtype=wp.spatial_vector),
    body_delta_qd: wp.array(dtype=wp.spatial_vector),
    contact_force: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        if contact_force:
            contact_force[tid] = wp.vec3()
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        if contact_force:
            contact_force[tid] = wp.vec3()
        return

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    n = contact_normal[tid]
    if wp.length_sq(n) == 0.0:
        if contact_force:
            contact_force[tid] = wp.vec3()
        return

    adhesion = _average_shape_property(shape_a, shape_b, shape_material_ka)
    mu = _average_shape_property(shape_a, shape_b, shape_material_mu)
    mu_torsional = _average_shape_property(shape_a, shape_b, shape_material_mu_torsional)
    mu_rolling = _average_shape_property(shape_a, shape_b, shape_material_mu_rolling)
    restitution = restitution_scale * _average_shape_property(shape_a, shape_b, shape_material_restitution)

    if rigid_contact_friction_scale:
        friction_scale = rigid_contact_friction_scale[tid]
        if friction_scale > 0.0:
            mu *= friction_scale
            mu_torsional *= friction_scale
            mu_rolling *= friction_scale

    point_a = _world_contact_point(body_q, body_a, contact_point0[tid])
    point_b = _world_contact_point(body_q, body_b, contact_point1[tid])

    margin_a = contact_margin0[tid]
    margin_b = contact_margin1[tid]
    point_a = point_a + n * margin_a
    point_b = point_b - n * margin_b

    separation = wp.dot(n, point_b - point_a)
    if separation >= adhesion:
        if contact_force:
            contact_force[tid] = wp.vec3()
        return

    penetration = wp.max(-separation, 0.0)
    bias = baumgarte * wp.max(penetration - penetration_slop, 0.0) * inv_dt

    com_a = point_a
    com_b = point_b
    if body_a >= 0:
        com_a = _body_com_world(body_q, body_com, body_a)
    if body_b >= 0:
        com_b = _body_com_world(body_q, body_com, body_b)

    r_a = point_a - com_a
    r_b = point_b - com_b

    current_anchor0 = contact_point0[tid] + contact_offset0[tid]
    current_anchor1 = contact_point1[tid] + contact_offset1[tid]
    anchor0_local = contact_anchor0[tid]
    anchor1_local = contact_anchor1[tid]
    anchor0_world = _world_contact_point(body_q, body_a, anchor0_local)
    anchor1_world = _world_contact_point(body_q, body_b, anchor1_local)
    friction_delta = anchor1_world - anchor0_world
    friction_delta_t = friction_delta - n * wp.dot(n, friction_delta)
    friction_delta_t_len = wp.length(friction_delta_t)
    if friction_delta_t_len > static_friction_anchor_break_distance:
        anchor0_local = current_anchor0
        anchor1_local = current_anchor1
        contact_anchor0[tid] = anchor0_local
        contact_anchor1[tid] = anchor1_local
        friction_delta_t = wp.vec3()
        friction_delta_t_len = 0.0

    v_a = wp.vec3()
    v_b = wp.vec3()
    if body_a >= 0:
        v_a = velocity_at_point(body_qd[body_a], r_a)
    if body_b >= 0:
        v_b = velocity_at_point(body_qd[body_b], r_b)

    rel_v = v_b - v_a
    vn = wp.dot(n, rel_v)

    target_vn = bias
    closing_speed = -vn
    if closing_speed > restitution_velocity_threshold:
        target_vn = wp.max(target_vn, closing_speed * restitution)

    normal_inv_mass = _contact_inv_mass(body_q, body_inv_mass, body_inv_inertia, body_flags, body_a, r_a, n)
    normal_inv_mass += _contact_inv_mass(body_q, body_inv_mass, body_inv_inertia, body_flags, body_b, r_b, n)

    if normal_inv_mass <= 0.0:
        if contact_force:
            contact_force[tid] = wp.vec3()
        return

    lambda_n = (target_vn - vn) / normal_inv_mass
    if lambda_n < 0.0:
        lambda_n = 0.0

    vt = rel_v - n * vn
    tangent_impulse = wp.vec3()
    vt_len = wp.length(vt)
    friction_drive = vt + friction_delta_t * (static_friction_anchor_gain * inv_dt)
    friction_drive_len = wp.length(friction_drive)

    if friction_drive_len > 0.0 and mu > 0.0 and lambda_n > 0.0:
        tangent = friction_drive / friction_drive_len
        tangent_inv_mass = _contact_inv_mass(body_q, body_inv_mass, body_inv_inertia, body_flags, body_a, r_a, tangent)
        tangent_inv_mass += _contact_inv_mass(
            body_q, body_inv_mass, body_inv_inertia, body_flags, body_b, r_b, tangent
        )

        if tangent_inv_mass > 0.0:
            friction_limit = mu * lambda_n
            if vt_len < static_friction_velocity_threshold:
                friction_limit *= static_friction_scale
            unclamped_lambda_t = friction_drive_len / tangent_inv_mass
            lambda_t = wp.min(unclamped_lambda_t, friction_limit)
            tangent_impulse = -tangent * lambda_t
            if unclamped_lambda_t > friction_limit and vt_len > static_friction_velocity_threshold:
                contact_anchor0[tid] = current_anchor0
                contact_anchor1[tid] = current_anchor1

    impulse = n * lambda_n + tangent_impulse

    w_a = wp.vec3()
    w_b = wp.vec3()
    if body_a >= 0:
        w_a = wp.spatial_bottom(body_qd[body_a])
    if body_b >= 0:
        w_b = wp.spatial_bottom(body_qd[body_b])

    delta_w = w_b - w_a

    if mu_torsional > 0.0 and lambda_n > 0.0:
        spin_speed = wp.dot(delta_w, n)
        if wp.abs(spin_speed) > 0.0:
            torsional_inv_mass = _angular_inv_mass(body_q, body_inv_inertia, body_inv_mass, body_flags, body_a, n)
            torsional_inv_mass += _angular_inv_mass(
                body_q, body_inv_inertia, body_inv_mass, body_flags, body_b, n
            )

            if torsional_inv_mass > 0.0:
                lambda_torsion = wp.min(wp.abs(spin_speed) / torsional_inv_mass, mu_torsional * lambda_n)
                torsional_impulse = -n * (wp.sign(spin_speed) * lambda_torsion)
                _apply_body_angular_impulse(
                    body_q,
                    body_inv_mass,
                    body_inv_inertia,
                    body_flags,
                    body_a,
                    -torsional_impulse,
                    body_impulses,
                    body_delta_qd,
                )
                _apply_body_angular_impulse(
                    body_q,
                    body_inv_mass,
                    body_inv_inertia,
                    body_flags,
                    body_b,
                    torsional_impulse,
                    body_impulses,
                    body_delta_qd,
                )

    if mu_rolling > 0.0 and lambda_n > 0.0:
        delta_w_t = delta_w - n * wp.dot(delta_w, n)
        roll_speed = wp.length(delta_w_t)
        if roll_speed > 0.0:
            roll_axis = delta_w_t / roll_speed
            rolling_inv_mass = _angular_inv_mass(
                body_q, body_inv_inertia, body_inv_mass, body_flags, body_a, roll_axis
            )
            rolling_inv_mass += _angular_inv_mass(
                body_q, body_inv_inertia, body_inv_mass, body_flags, body_b, roll_axis
            )

            if rolling_inv_mass > 0.0:
                lambda_roll = wp.min(roll_speed / rolling_inv_mass, mu_rolling * lambda_n)
                rolling_impulse = -roll_axis * lambda_roll
                _apply_body_angular_impulse(
                    body_q,
                    body_inv_mass,
                    body_inv_inertia,
                    body_flags,
                    body_a,
                    -rolling_impulse,
                    body_impulses,
                    body_delta_qd,
                )
                _apply_body_angular_impulse(
                    body_q,
                    body_inv_mass,
                    body_inv_inertia,
                    body_flags,
                    body_b,
                    rolling_impulse,
                    body_impulses,
                    body_delta_qd,
                )

    if contact_force:
        contact_force[tid] = contact_force[tid] + impulse * inv_dt

    if wp.length_sq(impulse) == 0.0:
        return

    _apply_body_impulse(body_q, body_inv_mass, body_inv_inertia, body_flags, body_a, r_a, -impulse, body_impulses, body_delta_qd)
    _apply_body_impulse(body_q, body_inv_mass, body_inv_inertia, body_flags, body_b, r_b, impulse, body_impulses, body_delta_qd)


@wp.kernel
def add_spatial_vectors(
    src: wp.array(dtype=wp.spatial_vector),
    delta: wp.array(dtype=wp.spatial_vector),
    # outputs
    dst: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    dst[tid] = src[tid] + delta[tid]


@wp.kernel
def add_scaled_spatial_vectors(
    src: wp.array(dtype=wp.spatial_vector),
    delta: wp.array(dtype=wp.spatial_vector),
    scale: float,
    # outputs
    dst: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    dst[tid] = src[tid] + delta[tid] * scale
