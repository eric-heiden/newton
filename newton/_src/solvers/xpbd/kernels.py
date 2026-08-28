# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ...math import (
    vec_abs,
    vec_leaky_max,
    vec_leaky_min,
    vec_max,
    vec_min,
    velocity_at_point,
)
from ...sim import BodyFlags, JointType
from ...sim.contacts import contact_surface_point, contact_surface_separation
from ..vbd.tri_mesh_collision import (
    TriMeshCollisionInfo,
    get_edge_colliding_edges,
    get_edge_colliding_edges_count,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
)


@wp.kernel
def eval_triangle_aerodynamics(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[int],
    air_velocity: wp.vec3,
    air_density: float,
    drag_coefficient: float,
    lift_coefficient: float,
    dt: float,
    particle_f: wp.array[wp.vec3],
):
    """Accumulate flat-plate aerodynamic drag and lift on each cloth triangle."""
    triangle = wp.tid()
    i = tri_indices[triangle, 0]
    j = tri_indices[triangle, 1]
    k = tri_indices[triangle, 2]

    normal = wp.cross(particle_x[j] - particle_x[i], particle_x[k] - particle_x[i])
    normal_length = wp.length(normal)
    if normal_length < 1.0e-12:
        return
    area = 0.5 * normal_length
    normal = normal / normal_length

    # Velocity of the triangle through the surrounding air.
    relative_velocity = (particle_v[i] + particle_v[j] + particle_v[k]) / 3.0 - air_velocity
    speed = wp.length(relative_velocity)
    if speed < 1.0e-6:
        return
    flow = relative_velocity / speed

    # Only the projected frontal area of an inclined plate meets the flow, so a
    # triangle moving edge-on produces almost no force.
    cos_theta = wp.dot(normal, flow)
    dynamic_pressure = 0.5 * air_density * speed * speed
    force = -drag_coefficient * dynamic_pressure * area * wp.abs(cos_theta) * flow
    if lift_coefficient != 0.0:
        # Component of the face normal perpendicular to the flow; the extra
        # cos_theta gives the flat-plate sin(a)cos(a) angle-of-attack response.
        perpendicular = normal - cos_theta * flow
        force -= lift_coefficient * dynamic_pressure * area * cos_theta * perpendicular

    share = force / 3.0
    share_magnitude = wp.length(share)
    indices = wp.vec3i(i, j, k)
    for local_index in range(3):
        particle = indices[local_index]
        if (particle_flags[particle] & ParticleFlags.ACTIVE) == 0:
            continue
        applied = share
        # Aerodynamic drag can only bring the cloth to rest relative to the
        # air. Letting a large coefficient or time step overshoot that would
        # turn drag into an energy source.
        velocity_change = share_magnitude * particle_invmass[particle] * dt
        if velocity_change > speed and velocity_change > 0.0:
            applied = share * (speed / velocity_change)
        wp.atomic_add(particle_f, particle, applied)


@wp.kernel
def damp_particle_velocities(
    particle_x_prev: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    damping: float,
    dt: float,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
):
    """Apply exponential particle drag while keeping prediction consistent."""
    particle = wp.tid()
    if (particle_flags[particle] & ParticleFlags.ACTIVE) == 0:
        return
    velocity = particle_v[particle] * wp.exp(-damping * dt)
    particle_v[particle] = velocity
    particle_x[particle] = particle_x_prev[particle] + velocity * dt


@wp.kernel
def copy_kinematic_body_state_kernel(
    body_flags: wp.array[wp.int32],
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
):
    """Copy prescribed maximal state through the solve for kinematic bodies."""
    tid = wp.tid()
    if (body_flags[tid] & int(BodyFlags.KINEMATIC)) == 0:
        return
    body_q_out[tid] = body_q_in[tid]
    body_qd_out[tid] = body_qd_in[tid]


@wp.kernel
def apply_particle_shape_restitution(
    particle_v_new: wp.array[wp.vec3],
    particle_x_old: wp.array[wp.vec3],
    particle_v_old: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_qd_prev: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    particle_ka: float,
    restitution: float,
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    particle_v_out: wp.array[wp.vec3],
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    v_new = particle_v_new[particle_index]
    px = particle_x_old[particle_index]
    v_old = particle_v_old[particle_index]

    X_wb = wp.transform_identity()
    X_wb_prev = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_wb_prev = body_q_prev[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # lever arm from previous pose (consistent with apply_rigid_restitution)
    bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[tid])
    r = bx_prev - wp.transform_point(X_wb_prev, X_com)

    # compute body velocity at the contact point
    bv_contact = wp.transform_vector(X_wb_prev, contact_body_vel[tid])
    bv_old = bv_contact
    bv_new = bv_contact
    if body_index >= 0:
        bv_old = velocity_at_point(body_qd_prev[body_index], r) + bv_contact
        bv_new = velocity_at_point(body_qd[body_index], r) + bv_contact

    rel_vel_old = wp.dot(n, v_old - bv_old)
    rel_vel_new = wp.dot(n, v_new - bv_new)

    if rel_vel_old < 0.0:
        dv = n * (-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0))

        wp.atomic_add(particle_v_out, particle_index, dv)


@wp.kernel
def count_particle_shape_contact_bodies(
    particle_x: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_m_inv: wp.array[float],
    body_flags: wp.array[wp.int32],
    shape_body: wp.array[int],
    shape_material_ka: wp.array[float],
    shape_margin: wp.array[float],
    particle_ka: float,
    contact_count: wp.array[int],
    contact_indices: wp.array[wp.vec3i],
    contact_barycentric: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    particle_contact_counts: wp.array[int],
    body_contact_counts: wp.array[int],
):
    """Count active soft-feature contacts incident on each particle and body."""
    contact = wp.tid()
    if contact >= min(contact_count[0], contact_max):
        return

    corners = contact_indices[contact]
    bary = contact_barycentric[contact]
    soft_position = wp.vec3(0.0)
    soft_radius = float(0.0)
    has_active_particle = False
    has_proxy_particle = False
    for local_index in range(3):
        particle = corners[local_index]
        if particle >= 0:
            soft_position += bary[local_index] * particle_x[particle]
            soft_radius = wp.max(soft_radius, particle_radius[particle])
            particle_flag = particle_flags[particle]
            if (particle_flag & ParticleFlags.ACTIVE) != 0:
                has_active_particle = True
            if (particle_flag & ParticleFlags.PROXY) != 0:
                has_proxy_particle = True
    if not has_active_particle:
        return

    shape = contact_shape[contact]
    body = shape_body[shape]
    if has_proxy_particle:
        if body < 0:
            return
        if (body_flags[body] & int(BodyFlags.PROXY)) != 0 or body_m_inv[body] == 0.0:
            return

    X_wb = wp.transform_identity()
    if body >= 0:
        X_wb = body_q[body]
    body_position = wp.transform_point(X_wb, contact_body_pos[contact])
    gap = wp.dot(contact_normal[contact], soft_position - body_position) - soft_radius - shape_margin[shape]
    adhesion = particle_ka + shape_material_ka[shape]
    if gap > adhesion:
        return

    for local_index in range(3):
        particle = corners[local_index]
        if particle >= 0 and bary[local_index] != 0.0:
            wp.atomic_add(particle_contact_counts, particle, 1)
    if body >= 0:
        wp.atomic_add(body_contact_counts, body, 1)


@wp.kernel
def solve_particle_shape_contacts(
    particle_x: wp.array[wp.vec3],
    particle_x_prev: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_margin: wp.array[float],
    shape_material_ka: wp.array[float],
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array[int],
    contact_indices: wp.array[wp.vec3i],
    contact_barycentric: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    particle_contact_counts: wp.array[int],
    body_contact_counts: wp.array[int],
    integrate_with_external_rigid_solver: bool,
    max_depenetration_velocity: float,
    dt: float,
    relaxation: float,
    lambdas: wp.array[float],
    split_lambdas: wp.array[float],
    friction_lambdas: wp.array[float],
    # outputs
    delta: wp.array[wp.vec3],
    body_delta: wp.array[wp.spatial_vector],
    particle_split_delta: wp.array[wp.vec3],
    body_split_velocity_delta: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    corners = contact_indices[tid]
    bary = contact_barycentric[tid]

    px = wp.vec3(0.0)
    px_prev = wp.vec3(0.0)
    soft_radius = float(0.0)
    w1 = float(0.0)
    has_active_particle = False
    has_proxy_particle = False
    for local_index in range(3):
        particle = corners[local_index]
        if particle >= 0:
            weight = bary[local_index]
            px += weight * particle_x[particle]
            px_prev += weight * particle_x_prev[particle]
            soft_radius = wp.max(soft_radius, particle_radius[particle])
            particle_flag = particle_flags[particle]
            if (particle_flag & ParticleFlags.ACTIVE) != 0:
                has_active_particle = True
                w1 += particle_invmass[particle] * weight * weight
            if (particle_flag & ParticleFlags.PROXY) != 0:
                has_proxy_particle = True
    if not has_active_particle:
        return
    if has_proxy_particle:
        if body_index < 0:
            return
        if (body_flags[body_index] & int(BodyFlags.PROXY)) != 0:
            return
        if body_m_inv[body_index] == 0.0:
            return

    X_wb = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    contact_radius = soft_radius + shape_margin[shape_index]
    c = wp.dot(n, px - bx) - contact_radius

    adhesion = particle_ka + shape_material_ka[shape_index]
    if c > adhesion:
        return

    # Limit the positional target for contacts that were already penetrating at
    # the beginning of the step. This avoids converting a large initial overlap
    # into a single, explosive separating velocity.
    X_wb_prev = wp.transform_identity()
    if body_index >= 0:
        X_wb_prev = body_q_prev[body_index]
    bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[tid])
    c_prev = wp.dot(n, px_prev - bx_prev) - contact_radius
    min_separation = 0.0
    if c_prev < 0.0:
        if max_depenetration_velocity >= 0.0:
            min_separation = wp.min(0.0, c_prev + max_depenetration_velocity * dt)
        else:
            max_correction = 0.25 * contact_radius
            min_separation = wp.min(0.0, c_prev + max_correction)
    c -= min_separation

    # Use the realized contact-point motion so externally integrated and proxy
    # bodies transfer tangential motion to particles. Fall back to body_qd when
    # the poses are unchanged but a prescribed velocity is available.
    bv = (bx - bx_prev) / dt
    if body_index >= 0 and wp.length_sq(bx - bx_prev) < 1.0e-16:
        body_v_s = body_qd[body_index]
        body_w = wp.spatial_bottom(body_v_s)
        body_v = wp.spatial_top(body_v_s)
        bv = body_v + wp.cross(body_w, r)
    bv += wp.transform_vector(X_wb, contact_body_vel[tid])
    # Evaluate tangential slip from the current contact displacement rather
    # than the pre-solve velocity. Position iterations update ``px`` and the
    # body pose, so this measure converges toward a no-slip state instead of
    # reapplying the same stale friction correction on every iteration.
    v = (px - px_prev) / dt - bv

    # compute inverse masses
    w2 = 0.0
    if body_index >= 0 and not integrate_with_external_rigid_solver:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return

    # Hard XPBD contact. A positive material adhesion distance makes the
    # constraint bilateral while the contact remains inside that distance.
    # Keeping the multiplier across iterations avoids repeatedly treating the
    # full penetration or separation as a new impulse.
    lambda_old = lambdas[tid]
    if adhesion <= 0.0 and c >= 0.0 and lambda_old <= 0.0:
        return
    dlambda = -c / denom
    lambda_new = lambda_old + dlambda
    if adhesion <= 0.0:
        lambda_new = wp.max(lambda_new, 0.0)
    dlambda = (lambda_new - lambda_old) * relaxation
    lambda_new = lambda_old + dlambda
    lambdas[tid] = lambda_new

    # Only the part of the correction that repairs overlap present at the
    # beginning of the step is non-physical split depenetration. Corrections
    # that arrest this step's incoming motion must remain in the velocity so a
    # resting contact can balance gravity without a separate support impulse.
    split_dlambda = 0.0
    if c_prev < 0.0:
        recovery_distance = wp.max(min_separation - c_prev, 0.0)
        split_lambda_new = wp.min(lambda_new, recovery_distance / denom)
        split_dlambda = split_lambda_new - split_lambdas[tid]
        split_lambdas[tid] = split_lambda_new

    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])
    vn = wp.dot(n, v)
    vt = v - n * vn
    tangent_speed = wp.length(vt)
    delta_f = wp.vec3(0.0)
    if tangent_speed > 1.0e-8 and mu > 0.0:
        # Accumulate the tangential multiplier over the step and apply only its
        # increment, mirroring the normal direction. Re-applying the full
        # Coulomb-bounded impulse every iteration would make the friction force
        # scale with the iteration count rather than with mu.
        friction_lambda_old = friction_lambdas[tid]
        friction_lambda_new = wp.min(
            friction_lambda_old + tangent_speed * dt / denom * relaxation,
            mu * wp.abs(lambda_new),
        )
        # A shrinking normal impulse must not drive the tangential correction
        # backwards; that would inject energy instead of removing slip.
        friction_lambda_new = wp.max(friction_lambda_new, friction_lambda_old)
        friction_lambdas[tid] = friction_lambda_new
        delta_f = -vt / tangent_speed * (friction_lambda_new - friction_lambda_old)
    delta_total = n * dlambda + delta_f

    for local_index in range(3):
        particle = corners[local_index]
        if particle >= 0:
            weight = bary[local_index]
            inverse_mass = particle_invmass[particle]
            particle_contact_count = float(wp.max(particle_contact_counts[particle], 1))
            wp.atomic_add(delta, particle, inverse_mass * weight * delta_total / particle_contact_count)
            if split_dlambda != 0.0:
                wp.atomic_add(
                    particle_split_delta,
                    particle,
                    inverse_mass * weight * n * split_dlambda / particle_contact_count,
                )

    if body_index >= 0 and not integrate_with_external_rigid_solver:
        # apply_body_deltas() treats body_delta as a velocity-like correction:
        # it multiplies by inverse mass/inertia and dt to update the body pose.
        # delta_total is a positional contact correction, matching the particle
        # path above, so convert it to the body-delta convention here.
        body_contact_count = float(wp.max(body_contact_counts[body_index], 1))
        delta_v = delta_total / (dt * body_contact_count)
        delta_w = wp.cross(r, delta_v)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_v, delta_w))
        if split_dlambda != 0.0:
            normal_velocity_impulse = n * split_dlambda / (dt * body_contact_count)
            q = wp.transform_get_rotation(X_wb)
            torque = wp.cross(r, -normal_velocity_impulse)
            angular_delta = wp.quat_rotate(q, body_I_inv[body_index] * wp.quat_rotate_inv(q, torque))
            wp.atomic_add(
                body_split_velocity_delta,
                body_index,
                wp.spatial_vector(-body_m_inv[body_index] * normal_velocity_impulse, angular_delta),
            )


@wp.kernel
def solve_particle_shape_contact_velocities(
    particle_x: wp.array[wp.vec3],
    particle_x_prev: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_v_prev: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_margin: wp.array[float],
    particle_mu: float,
    restitution: float,
    contact_count: wp.array[int],
    contact_indices: wp.array[wp.vec3i],
    contact_barycentric: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    particle_contact_counts: wp.array[int],
    body_contact_counts: wp.array[int],
    integrate_with_external_rigid_solver: bool,
    dt: float,
    relaxation: float,
    lambdas: wp.array[float],
    friction_lambdas: wp.array[float],
    particle_velocity_delta: wp.array[wp.vec3],
    body_velocity_delta: wp.array[wp.spatial_vector],
    particle_static_friction_limit: wp.array[wp.vec4],
):
    """Remove artificial separating velocity and enforce Coulomb friction."""
    tid = wp.tid()
    count = min(contact_max, contact_count[0])
    if tid >= count or wp.abs(lambdas[tid]) <= 1.0e-12:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    corners = contact_indices[tid]
    bary = contact_barycentric[tid]
    soft_position_prev = wp.vec3(0.0)
    soft_velocity = wp.vec3(0.0)
    soft_velocity_prev = wp.vec3(0.0)
    soft_radius = float(0.0)
    w1 = float(0.0)
    has_active_particle = False
    has_proxy_particle = False
    for local_index in range(3):
        particle = corners[local_index]
        if particle >= 0:
            weight = bary[local_index]
            soft_position_prev += weight * particle_x_prev[particle]
            soft_velocity += weight * particle_v[particle]
            soft_velocity_prev += weight * particle_v_prev[particle]
            soft_radius = wp.max(soft_radius, particle_radius[particle])
            particle_flag = particle_flags[particle]
            if (particle_flag & ParticleFlags.ACTIVE) != 0:
                has_active_particle = True
                w1 += particle_invmass[particle] * weight * weight
            if (particle_flag & ParticleFlags.PROXY) != 0:
                has_proxy_particle = True
    if not has_active_particle:
        return
    if has_proxy_particle:
        if body_index < 0:
            return
        if (body_flags[body_index] & int(BodyFlags.PROXY)) != 0 or body_m_inv[body_index] == 0.0:
            return

    X_wb = wp.transform_identity()
    X_wb_prev = wp.transform_identity()
    X_com = wp.vec3(0.0)
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_wb_prev = body_q_prev[body_index]
        X_com = body_com[body_index]

    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)
    n = contact_normal[tid]

    # Prescribed and externally integrated bodies are best represented by the
    # realized pose delta. Dynamic XPBD bodies use their solved velocity so an
    # equal-and-opposite velocity impulse can be harvested by proxy coupling.
    local_velocity = wp.transform_vector(X_wb, contact_body_vel[tid])
    body_motion_velocity = (bx - bx_prev) / dt + local_velocity
    if body_index >= 0 and wp.length_sq(bx - bx_prev) < 1.0e-16:
        body_v_s = body_qd[body_index]
        body_motion_velocity = wp.spatial_top(body_v_s) + wp.cross(wp.spatial_bottom(body_v_s), r) + local_velocity
    body_current_velocity = body_motion_velocity
    if body_index >= 0 and not integrate_with_external_rigid_solver and body_m_inv[body_index] > 0.0:
        body_v_s = body_qd[body_index]
        body_current_velocity = wp.spatial_top(body_v_s) + wp.cross(wp.spatial_bottom(body_v_s), r) + local_velocity

    v_current = soft_velocity - body_current_velocity
    v_previous = soft_velocity_prev - body_motion_velocity
    vn_current = wp.dot(n, v_current)
    vn_previous = wp.dot(n, v_previous)

    w2 = 0.0
    if body_index >= 0 and not integrate_with_external_rigid_solver:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, body_I_inv[body_index] * rot_angular)
    denominator = w1 + w2
    if denominator <= 0.0:
        return

    contact_radius = soft_radius + shape_margin[shape_index]
    previous_gap = wp.dot(n, soft_position_prev - bx_prev) - contact_radius
    desired_vn = wp.max(vn_previous, 0.0)
    if lambdas[tid] < 0.0:
        desired_vn = 0.0
    # Treat only material overlap as pre-existing penetration. Contact points
    # generated exactly on the surface can have a tiny negative gap from
    # floating-point roundoff; those are new impacts and must retain their
    # restitution response.
    overlap_epsilon = 1.0e-4 * contact_radius
    if previous_gap < -overlap_epsilon:
        desired_vn = 0.0
    elif vn_previous < 0.0:
        desired_vn = -restitution * vn_previous
    normal_impulse_scalar = (desired_vn - vn_current) / denominator * relaxation
    impulse = n * normal_impulse_scalar

    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])
    # The Coulomb budget is per step, not per pass. The positional solve
    # already spent ``friction_lambdas``; spending the full budget again here
    # would double the friction force a material coefficient can produce.
    friction_budget = wp.max(mu * wp.abs(lambdas[tid]) - friction_lambdas[tid], 0.0) / dt
    tangent_velocity = v_current - n * vn_current
    tangent_speed = wp.length(tangent_velocity)
    if tangent_speed > 1.0e-8:
        tangent = tangent_velocity / tangent_speed
        tangent_denominator = w1
        if body_index >= 0 and not integrate_with_external_rigid_solver:
            angular_tangent = wp.cross(r, tangent)
            q = wp.transform_get_rotation(X_wb)
            rot_angular_tangent = wp.quat_rotate_inv(q, angular_tangent)
            tangent_denominator += body_m_inv[body_index] + wp.dot(
                rot_angular_tangent, body_I_inv[body_index] * rot_angular_tangent
            )
        if tangent_denominator > 0.0:
            tangent_impulse = wp.min(
                tangent_speed / tangent_denominator * relaxation,
                friction_budget,
            )
            impulse -= tangent * tangent_impulse

    for local_index in range(3):
        particle = corners[local_index]
        if particle >= 0:
            weight = bary[local_index]
            particle_contact_count = float(wp.max(particle_contact_counts[particle], 1))
            wp.atomic_add(
                particle_velocity_delta,
                particle,
                particle_invmass[particle] * weight * impulse / particle_contact_count,
            )
            if body_index < 0 and mu > 0.0:
                # A Jacobi contact pass can leave a small tangential residual
                # when several redundant edge/face records support one cloth
                # vertex. Accumulate the actual Coulomb velocity budget so the
                # final per-particle application can resolve the static branch
                # exactly without damping unconstrained motion.
                static_limit = particle_invmass[particle] * wp.abs(weight) * friction_budget / particle_contact_count
                wp.atomic_add(
                    particle_static_friction_limit,
                    particle,
                    wp.vec4(n[0] * static_limit, n[1] * static_limit, n[2] * static_limit, static_limit),
                )
    if body_index >= 0 and not integrate_with_external_rigid_solver:
        impulse /= float(wp.max(body_contact_counts[body_index], 1))
        q = wp.transform_get_rotation(X_wb)
        torque = wp.cross(r, -impulse)
        angular_delta = wp.quat_rotate(q, body_I_inv[body_index] * wp.quat_rotate_inv(q, torque))
        wp.atomic_add(
            body_velocity_delta,
            body_index,
            wp.spatial_vector(-body_m_inv[body_index] * impulse, angular_delta),
        )


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_x_rest: wp.array[wp.vec3],
    particle_in_mesh: wp.array[int],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    deltas: wp.array[wp.vec3],
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    particle_flag = particle_flags[i]
    if (particle_flag & ParticleFlags.ACTIVE) == 0:
        return
    is_proxy = particle_flag & ParticleFlags.PROXY

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        neighbor_flag = particle_flags[index]
        if (
            (neighbor_flag & ParticleFlags.ACTIVE) != 0
            and (is_proxy == 0 or ((neighbor_flag & ParticleFlags.PROXY) == 0 and particle_invmass[index] > 0.0))
            and index != i
        ):
            # Two mesh vertices that already overlap in the rest pose are
            # neighbors on the same surface, not a colliding pair. A cloth
            # spaced closer than its own particle diameter would otherwise have
            # every vertex repel its neighbors and tear itself apart. Free
            # particles keep colliding even when they start overlapped.
            if particle_in_mesh[i] != 0 and particle_in_mesh[index] != 0:
                rest_separation = wp.length(particle_x_rest[i] - particle_x_rest[index])
                if rest_separation < radius + particle_radius[index]:
                    continue

            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0 and d > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = vrel - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


@wp.kernel
def count_vertex_triangle_self_contacts(
    particle_x: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    collision_radius: float,
    counts: wp.array[int],
):
    """Count active vertex-triangle contact incidence per particle."""
    vertex = wp.tid()
    collision_info = collision_info_array[0]
    collision_count = get_vertex_colliding_triangles_count(collision_info, vertex)
    for collision_index in range(collision_count):
        triangle = get_vertex_colliding_triangles(collision_info, vertex, collision_index)
        t0 = tri_indices[triangle, 0]
        t1 = tri_indices[triangle, 1]
        t2 = tri_indices[triangle, 2]
        closest, _bary, _feature = triangle_closest_point(
            particle_x[t0], particle_x[t1], particle_x[t2], particle_x[vertex]
        )
        if wp.length_sq(particle_x[vertex] - closest) >= collision_radius * collision_radius:
            continue
        wp.atomic_add(counts, vertex, 1)
        wp.atomic_add(counts, t0, 1)
        wp.atomic_add(counts, t1, 1)
        wp.atomic_add(counts, t2, 1)


@wp.kernel
def count_edge_edge_self_contacts(
    particle_x: wp.array[wp.vec3],
    edge_indices: wp.array2d[int],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    collision_radius: float,
    edge_parallel_epsilon: float,
    counts: wp.array[int],
):
    """Count unique active edge-edge contact incidence per particle."""
    edge0 = wp.tid()
    collision_info = collision_info_array[0]
    collision_count = get_edge_colliding_edges_count(collision_info, edge0)
    for collision_index in range(collision_count):
        edge1 = get_edge_colliding_edges(collision_info, edge0, collision_index)
        if edge1 > edge0:
            e00 = edge_indices[edge0, 2]
            e01 = edge_indices[edge0, 3]
            e10 = edge_indices[edge1, 2]
            e11 = edge_indices[edge1, 3]
            closest_parameters = wp.closest_point_edge_edge(
                particle_x[e00],
                particle_x[e01],
                particle_x[e10],
                particle_x[e11],
                edge_parallel_epsilon,
            )
            s = closest_parameters[0]
            t = closest_parameters[1]
            point0 = particle_x[e00] + (particle_x[e01] - particle_x[e00]) * s
            point1 = particle_x[e10] + (particle_x[e11] - particle_x[e10]) * t
            if wp.length_sq(point0 - point1) >= collision_radius * collision_radius:
                continue
            wp.atomic_add(counts, e00, 1)
            wp.atomic_add(counts, e01, 1)
            wp.atomic_add(counts, e10, 1)
            wp.atomic_add(counts, e11, 1)


@wp.func
def triangle_pair_is_rest_filtered(
    particle_x_rest: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    triangle0: int,
    triangle1: int,
    exclusion_radius: float,
):
    """Return whether two triangles belong to the same local rest-pose patch."""
    if exclusion_radius <= 0.0:
        return False

    a0 = particle_x_rest[tri_indices[triangle0, 0]]
    a1 = particle_x_rest[tri_indices[triangle0, 1]]
    a2 = particle_x_rest[tri_indices[triangle0, 2]]
    b0 = particle_x_rest[tri_indices[triangle1, 0]]
    b1 = particle_x_rest[tri_indices[triangle1, 1]]
    b2 = particle_x_rest[tri_indices[triangle1, 2]]
    exclusion_radius_sq = exclusion_radius * exclusion_radius

    closest, _bary, _feature = triangle_closest_point(b0, b1, b2, a0)
    if wp.length_sq(a0 - closest) < exclusion_radius_sq:
        return True
    closest, _bary, _feature = triangle_closest_point(b0, b1, b2, a1)
    if wp.length_sq(a1 - closest) < exclusion_radius_sq:
        return True
    closest, _bary, _feature = triangle_closest_point(b0, b1, b2, a2)
    if wp.length_sq(a2 - closest) < exclusion_radius_sq:
        return True
    closest, _bary, _feature = triangle_closest_point(a0, a1, a2, b0)
    if wp.length_sq(b0 - closest) < exclusion_radius_sq:
        return True
    closest, _bary, _feature = triangle_closest_point(a0, a1, a2, b1)
    if wp.length_sq(b1 - closest) < exclusion_radius_sq:
        return True
    closest, _bary, _feature = triangle_closest_point(a0, a1, a2, b2)
    return wp.length_sq(b2 - closest) < exclusion_radius_sq


@wp.kernel
def count_triangle_intersection_self_contacts(
    particle_x: wp.array[wp.vec3],
    particle_world: wp.array[int],
    tri_indices: wp.array2d[int],
    intersecting_triangles: wp.array[int],
    intersecting_triangle_counts: wp.array[int],
    intersecting_triangle_offsets: wp.array[int],
    counts: wp.array[int],
):
    """Count exact triangle intersections used for self-contact recovery."""
    triangle0 = wp.tid()
    offset = intersecting_triangle_offsets[triangle0]
    capacity = intersecting_triangle_offsets[triangle0 + 1] - offset
    intersection_count = wp.min(intersecting_triangle_counts[triangle0], capacity)
    for intersection_index in range(intersection_count):
        triangle1 = intersecting_triangles[offset + intersection_index]
        if triangle1 <= triangle0:
            continue

        a0 = tri_indices[triangle0, 0]
        a1 = tri_indices[triangle0, 1]
        a2 = tri_indices[triangle0, 2]
        b0 = tri_indices[triangle1, 0]
        b1 = tri_indices[triangle1, 1]
        b2 = tri_indices[triangle1, 2]
        if particle_world[a0] != particle_world[b0]:
            continue
        if not wp.intersect_tri_tri(
            particle_x[a0],
            particle_x[a1],
            particle_x[a2],
            particle_x[b0],
            particle_x[b1],
            particle_x[b2],
        ):
            continue

        wp.atomic_add(counts, a0, 1)
        wp.atomic_add(counts, a1, 1)
        wp.atomic_add(counts, a2, 1)
        wp.atomic_add(counts, b0, 1)
        wp.atomic_add(counts, b1, 1)
        wp.atomic_add(counts, b2, 1)


@wp.kernel
def solve_triangle_intersection_self_contacts(
    particle_x: wp.array[wp.vec3],
    particle_x_prev: wp.array[wp.vec3],
    particle_x_rest: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_world: wp.array[int],
    tri_indices: wp.array2d[int],
    intersecting_triangles: wp.array[int],
    intersecting_triangle_counts: wp.array[int],
    intersecting_triangle_offsets: wp.array[int],
    collision_radius: float,
    rest_shape_exclusion_radius: float,
    max_depenetration_velocity: float,
    dt: float,
    relaxation: float,
    prefer_shortest_local_exit: bool,
    contact_counts: wp.array[int],
    contact_normals: wp.array[wp.vec3],
    contact_zero_normal_velocity: wp.array[int],
    deltas: wp.array[wp.vec3],
    split_deltas: wp.array[wp.vec3],
):
    """Recover exact triangle intersections missed by discrete feature proximity."""
    triangle0 = wp.tid()
    offset = intersecting_triangle_offsets[triangle0]
    capacity = intersecting_triangle_offsets[triangle0 + 1] - offset
    intersection_count = wp.min(intersecting_triangle_counts[triangle0], capacity)
    for intersection_index in range(intersection_count):
        triangle1 = intersecting_triangles[offset + intersection_index]
        if triangle1 <= triangle0:
            continue

        a0_index = tri_indices[triangle0, 0]
        a1_index = tri_indices[triangle0, 1]
        a2_index = tri_indices[triangle0, 2]
        b0_index = tri_indices[triangle1, 0]
        b1_index = tri_indices[triangle1, 1]
        b2_index = tri_indices[triangle1, 2]
        if particle_world[a0_index] != particle_world[b0_index]:
            continue
        is_local_rest_pair = triangle_pair_is_rest_filtered(
            particle_x_rest,
            tri_indices,
            triangle0,
            triangle1,
            rest_shape_exclusion_radius,
        )
        target_radius = collision_radius
        if is_local_rest_pair:
            # Nearby material triangles need inversion protection, but applying
            # the full cloth thickness would inflate an undeformed local patch.
            target_radius = 0.05 * collision_radius

        a0 = particle_x[a0_index]
        a1 = particle_x[a1_index]
        a2 = particle_x[a2_index]
        b0 = particle_x[b0_index]
        b1 = particle_x[b1_index]
        b2 = particle_x[b2_index]
        a0_prev = particle_x_prev[a0_index]
        a1_prev = particle_x_prev[a1_index]
        a2_prev = particle_x_prev[a2_index]
        b0_prev = particle_x_prev[b0_index]
        b1_prev = particle_x_prev[b1_index]
        b2_prev = particle_x_prev[b2_index]

        if not wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2):
            continue

        normal_a = wp.cross(a1 - a0, a2 - a0)
        normal_b = wp.cross(b1 - b0, b2 - b0)
        normal_a_prev = wp.cross(a1_prev - a0_prev, a2_prev - a0_prev)
        normal_b_prev = wp.cross(b1_prev - b0_prev, b2_prev - b0_prev)
        length_a = wp.length(normal_a)
        length_b = wp.length(normal_b)
        length_a_prev = wp.length(normal_a_prev)
        length_b_prev = wp.length(normal_b_prev)
        if length_a <= 1.0e-8 or length_b <= 1.0e-8:
            continue
        normal_a /= length_a
        normal_b /= length_b
        if length_a_prev > 1.0e-8:
            normal_a_prev /= length_a_prev
            if wp.dot(normal_a, normal_a_prev) < 0.0:
                normal_a = -normal_a
        else:
            normal_a_prev = normal_a
        if length_b_prev > 1.0e-8:
            normal_b_prev /= length_b_prev
            if wp.dot(normal_b, normal_b_prev) < 0.0:
                normal_b = -normal_b
        else:
            normal_b_prev = normal_b

        center_a_prev = (a0_prev + a1_prev + a2_prev) / 3.0
        center_b_prev = (b0_prev + b1_prev + b2_prev) / 3.0
        was_intersecting = wp.intersect_tri_tri(
            a0_prev,
            a1_prev,
            a2_prev,
            b0_prev,
            b1_prev,
            b2_prev,
        )
        use_shortest_local_exit = prefer_shortest_local_exit and is_local_rest_pair and was_intersecting
        if not was_intersecting or is_local_rest_pair:
            contact_zero_normal_velocity[offset + intersection_index] = 1
        # Move each triangle back to the ordering from the beginning of the
        # step. On the final iteration, a local pair that is still inverted has
        # no useful historical ordering, so move it through the shorter exit.
        side_a = 1.0
        if wp.dot(center_a_prev - center_b_prev, normal_b_prev) < 0.0:
            side_a = -1.0
        distance_a0 = wp.dot(a0 - b0, normal_b)
        distance_a1 = wp.dot(a1 - b0, normal_b)
        distance_a2 = wp.dot(a2 - b0, normal_b)
        shift_a = wp.max(target_radius - wp.min(distance_a0, wp.min(distance_a1, distance_a2)), 0.0)
        if side_a < 0.0:
            shift_a = wp.min(-target_radius - wp.max(distance_a0, wp.max(distance_a1, distance_a2)), 0.0)
        if use_shortest_local_exit:
            positive_shift = wp.max(
                target_radius - wp.min(distance_a0, wp.min(distance_a1, distance_a2)),
                0.0,
            )
            negative_shift = wp.min(
                -target_radius - wp.max(distance_a0, wp.max(distance_a1, distance_a2)),
                0.0,
            )
            if wp.abs(negative_shift) < wp.abs(positive_shift):
                shift_a = negative_shift
            else:
                shift_a = positive_shift
        relative_correction_a = normal_b * shift_a

        # Candidate 1 moves B relative to A, then converts that motion to A-B.
        side_b = 1.0
        if wp.dot(center_b_prev - center_a_prev, normal_a_prev) < 0.0:
            side_b = -1.0
        distance_b0 = wp.dot(b0 - a0, normal_a)
        distance_b1 = wp.dot(b1 - a0, normal_a)
        distance_b2 = wp.dot(b2 - a0, normal_a)
        shift_b = wp.max(target_radius - wp.min(distance_b0, wp.min(distance_b1, distance_b2)), 0.0)
        if side_b < 0.0:
            shift_b = wp.min(-target_radius - wp.max(distance_b0, wp.max(distance_b1, distance_b2)), 0.0)
        if use_shortest_local_exit:
            positive_shift = wp.max(
                target_radius - wp.min(distance_b0, wp.min(distance_b1, distance_b2)),
                0.0,
            )
            negative_shift = wp.min(
                -target_radius - wp.max(distance_b0, wp.max(distance_b1, distance_b2)),
                0.0,
            )
            if wp.abs(negative_shift) < wp.abs(positive_shift):
                shift_b = negative_shift
            else:
                shift_b = positive_shift
        relative_correction_b = -normal_a * shift_b

        relative_correction = relative_correction_a
        if wp.length_sq(relative_correction_b) < wp.length_sq(relative_correction_a):
            relative_correction = relative_correction_b

        correction_length = wp.length(relative_correction)
        if correction_length <= 1.0e-8:
            continue

        # Exact recovery is an emergency depenetration path. Keep its configured
        # cap even for a newly crossed pair: contact compression can make the
        # apparent pair motion much larger than the physical separation needed,
        # so using that motion as a correction limit creates positive feedback.
        if max_depenetration_velocity >= 0.0:
            max_correction = max_depenetration_velocity * dt
            if correction_length > max_correction:
                relative_correction *= max_correction / correction_length
                correction_length = max_correction

        normal = relative_correction / correction_length
        contact_normals[offset + intersection_index] = normal

        inverse_mass_sum = (
            particle_invmass[a0_index]
            + particle_invmass[a1_index]
            + particle_invmass[a2_index]
            + particle_invmass[b0_index]
            + particle_invmass[b1_index]
            + particle_invmass[b2_index]
        )
        if inverse_mass_sum <= 0.0:
            continue
        # A vertex shared by several intersecting pairs would otherwise receive
        # each pair's full separation, and that overshoot creates new
        # intersections faster than recovery removes them. A single shared
        # Jacobi scale divides the work while keeping the impulse on the two
        # triangles equal and opposite.
        correction_count_sum = int(0)
        all_indices = wp.vec3i(a0_index, a1_index, a2_index)
        all_indices_b = wp.vec3i(b0_index, b1_index, b2_index)
        for local_index in range(3):
            correction_count_sum += wp.max(contact_counts[all_indices[local_index]], 1)
            correction_count_sum += wp.max(contact_counts[all_indices_b[local_index]], 1)
        jacobi_scale = 6.0 / float(correction_count_sum)
        scale = 3.0 * relaxation * jacobi_scale / inverse_mass_sum
        indices = wp.vec4i(a0_index, a1_index, a2_index, b0_index)
        signs = wp.vec4(1.0, 1.0, 1.0, -1.0)
        for local_index in range(4):
            particle = indices[local_index]
            correction = relative_correction * signs[local_index] * particle_invmass[particle] * scale
            wp.atomic_add(deltas, particle, correction)
            if was_intersecting and not is_local_rest_pair:
                wp.atomic_add(split_deltas, particle, correction)
        indices_b = wp.vec2i(b1_index, b2_index)
        for local_index in range(2):
            particle = indices_b[local_index]
            correction = -relative_correction * particle_invmass[particle] * scale
            wp.atomic_add(deltas, particle, correction)
            if was_intersecting and not is_local_rest_pair:
                wp.atomic_add(split_deltas, particle, correction)


@wp.kernel
def compute_particle_displacements(
    particle_x_start: wp.array[wp.vec3],
    particle_x_end: wp.array[wp.vec3],
    displacements: wp.array[wp.vec3],
):
    """Compute particle trajectories for conservative self-contact truncation."""
    particle = wp.tid()
    displacements[particle] = particle_x_end[particle] - particle_x_start[particle]


@wp.kernel
def solve_triangle_intersection_self_contact_velocities(
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    tri_indices: wp.array2d[int],
    intersecting_triangles: wp.array[int],
    intersecting_triangle_counts: wp.array[int],
    intersecting_triangle_offsets: wp.array[int],
    contact_counts: wp.array[int],
    contact_normals: wp.array[wp.vec3],
    contact_zero_normal_velocity: wp.array[int],
    velocity_deltas: wp.array[wp.vec3],
):
    """Remove closing normal velocity after exact triangle-intersection recovery."""
    triangle0 = wp.tid()
    offset = intersecting_triangle_offsets[triangle0]
    capacity = intersecting_triangle_offsets[triangle0 + 1] - offset
    intersection_count = wp.min(intersecting_triangle_counts[triangle0], capacity)
    for intersection_index in range(intersection_count):
        triangle1 = intersecting_triangles[offset + intersection_index]
        if triangle1 <= triangle0:
            continue
        normal = contact_normals[offset + intersection_index]
        if wp.length_sq(normal) <= 0.0:
            continue

        a0 = tri_indices[triangle0, 0]
        a1 = tri_indices[triangle0, 1]
        a2 = tri_indices[triangle0, 2]
        b0 = tri_indices[triangle1, 0]
        b1 = tri_indices[triangle1, 1]
        b2 = tri_indices[triangle1, 2]
        velocity_a = (particle_v[a0] + particle_v[a1] + particle_v[a2]) / 3.0
        velocity_b = (particle_v[b0] + particle_v[b1] + particle_v[b2]) / 3.0
        closing_velocity = wp.dot(velocity_a - velocity_b, normal)
        if closing_velocity >= 0.0 and contact_zero_normal_velocity[offset + intersection_index] == 0:
            continue

        inverse_mass_sum = (
            particle_invmass[a0]
            + particle_invmass[a1]
            + particle_invmass[a2]
            + particle_invmass[b0]
            + particle_invmass[b1]
            + particle_invmass[b2]
        )
        if inverse_mass_sum <= 0.0:
            continue

        indices = wp.vec4i(a0, a1, a2, b0)
        signs = wp.vec4(1.0, 1.0, 1.0, -1.0)
        for local_index in range(4):
            particle = indices[local_index]
            correction = (
                -3.0
                * particle_invmass[particle]
                * signs[local_index]
                * normal
                * closing_velocity
                / inverse_mass_sum
                / float(wp.max(contact_counts[particle], 1))
            )
            wp.atomic_add(velocity_deltas, particle, correction)
        indices_b = wp.vec2i(b1, b2)
        for local_index in range(2):
            particle = indices_b[local_index]
            correction = (
                3.0
                * particle_invmass[particle]
                * normal
                * closing_velocity
                / inverse_mass_sum
                / float(wp.max(contact_counts[particle], 1))
            )
            wp.atomic_add(velocity_deltas, particle, correction)


@wp.kernel
def solve_vertex_triangle_self_contacts(
    particle_x: wp.array[wp.vec3],
    particle_x_prev: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    tri_indices: wp.array2d[int],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    collision_radius: float,
    friction_coefficient: float,
    max_depenetration_velocity: float,
    dt: float,
    relaxation: float,
    contact_counts: wp.array[int],
    lambdas: wp.array[float],
    split_lambdas: wp.array[float],
    friction_lambdas: wp.array[float],
    deltas: wp.array[wp.vec3],
    split_deltas: wp.array[wp.vec3],
):
    """Solve vertex-triangle cloth self-contact candidates."""
    vertex = wp.tid()
    collision_info = collision_info_array[0]
    collision_count = get_vertex_colliding_triangles_count(collision_info, vertex)

    for collision_index in range(collision_count):
        triangle = get_vertex_colliding_triangles(collision_info, vertex, collision_index)
        t0 = tri_indices[triangle, 0]
        t1 = tri_indices[triangle, 1]
        t2 = tri_indices[triangle, 2]
        p = particle_x[vertex]
        a = particle_x[t0]
        b = particle_x[t1]
        c = particle_x[t2]
        closest, bary, _feature = triangle_closest_point(a, b, c, p)
        diff = p - closest
        distance = wp.length(diff)

        p_prev = particle_x_prev[vertex]
        a_prev = particle_x_prev[t0]
        b_prev = particle_x_prev[t1]
        c_prev = particle_x_prev[t2]
        closest_prev = bary[0] * a_prev + bary[1] * b_prev + bary[2] * c_prev
        diff_prev = p_prev - closest_prev

        tri_normal = wp.cross(b - a, c - a)
        tri_normal_prev = wp.cross(b_prev - a_prev, c_prev - a_prev)
        normal_prev_length = wp.length(tri_normal_prev)
        if normal_prev_length > 1.0e-8:
            tri_normal_prev /= normal_prev_length
        else:
            tri_normal_prev = wp.vec3(0.0, 0.0, 1.0)

        side = 1.0
        if wp.dot(diff_prev, tri_normal_prev) < 0.0:
            side = -1.0

        normal_length = wp.length(tri_normal)
        if normal_length > 1.0e-8:
            tri_normal = tri_normal / normal_length * side
        else:
            tri_normal = tri_normal_prev * side

        normal = tri_normal
        if distance > 1.0e-8:
            normal = diff / distance
            if wp.dot(normal, tri_normal) < 0.0:
                normal = -normal

        signed_distance = wp.dot(diff, normal)
        previous_distance = wp.length(diff_prev)
        target_distance = collision_radius
        if max_depenetration_velocity >= 0.0:
            target_distance = wp.min(collision_radius, previous_distance + max_depenetration_velocity * dt)

        weights = wp.vec4(1.0, -bary[0], -bary[1], -bary[2])
        indices = wp.vec4i(vertex, t0, t1, t2)
        denominator = float(0.0)
        # A shared Jacobi scale preserves the equal-and-opposite impulse when
        # the four vertices have different contact valences.
        correction_count_sum = int(0)
        for local_index in range(4):
            particle = indices[local_index]
            weighted_inverse_mass = particle_invmass[particle] * weights[local_index] * weights[local_index]
            denominator += weighted_inverse_mass
            correction_count_sum += wp.max(contact_counts[particle], 1)
        jacobi_scale = 4.0 / float(correction_count_sum)
        split_denominator = denominator * jacobi_scale

        lambda_index = collision_info.vertex_colliding_triangles_offsets[vertex] + collision_index
        lambda_old = lambdas[lambda_index]
        constraint = signed_distance - target_distance
        if denominator > 0.0 and (constraint < 0.0 or lambda_old > 0.0):
            dlambda = -constraint / denominator
            lambda_new = wp.max(lambda_old + dlambda, 0.0)
            dlambda = (lambda_new - lambda_old) * relaxation
            lambda_new = lambda_old + dlambda
            lambdas[lambda_index] = lambda_new

            recovery_distance = wp.max(target_distance - previous_distance, 0.0)
            split_lambda_new = wp.min(lambda_new, recovery_distance / split_denominator)
            split_dlambda = split_lambda_new - split_lambdas[lambda_index]
            split_lambdas[lambda_index] = split_lambda_new

            relative_displacement = p - p_prev
            relative_displacement -= bary[0] * (a - a_prev)
            relative_displacement -= bary[1] * (b - b_prev)
            relative_displacement -= bary[2] * (c - c_prev)
            tangent_displacement = relative_displacement - normal * wp.dot(relative_displacement, normal)
            tangent_distance = wp.length(tangent_displacement)
            tangent_impulse = wp.vec3(0.0)
            if tangent_distance > 1.0e-8 and friction_coefficient > 0.0:
                # Accumulate over the step and apply only the increment, so the
                # tangential impulse is bounded by mu * lambda_n once per step
                # instead of once per iteration.
                friction_lambda_old = friction_lambdas[lambda_index]
                friction_lambda_new = wp.min(
                    friction_lambda_old + tangent_distance / denominator * relaxation,
                    friction_coefficient * lambda_new,
                )
                friction_lambda_new = wp.max(friction_lambda_new, friction_lambda_old)
                friction_lambdas[lambda_index] = friction_lambda_new
                tangent_impulse = -tangent_displacement / tangent_distance * (friction_lambda_new - friction_lambda_old)

            impulse = normal * dlambda + tangent_impulse
            for local_index in range(4):
                particle = indices[local_index]
                correction = particle_invmass[particle] * weights[local_index] * impulse * jacobi_scale
                cap_scale = 1.0
                correction_length = wp.length(correction)
                max_correction = 0.25 * collision_radius
                if max_depenetration_velocity >= 0.0:
                    max_correction = max_depenetration_velocity * dt
                if correction_length > max_correction and correction_length > 0.0:
                    cap_scale = max_correction / correction_length
                    correction *= cap_scale
                wp.atomic_add(deltas, particle, correction)
                if split_dlambda != 0.0:
                    split_correction = (
                        particle_invmass[particle]
                        * weights[local_index]
                        * normal
                        * split_dlambda
                        * jacobi_scale
                        * cap_scale
                    )
                    wp.atomic_add(split_deltas, particle, split_correction)


@wp.kernel
def solve_edge_edge_self_contacts(
    particle_x: wp.array[wp.vec3],
    particle_x_prev: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    edge_indices: wp.array2d[int],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    collision_radius: float,
    friction_coefficient: float,
    max_depenetration_velocity: float,
    edge_parallel_epsilon: float,
    dt: float,
    relaxation: float,
    contact_counts: wp.array[int],
    lambdas: wp.array[float],
    split_lambdas: wp.array[float],
    friction_lambdas: wp.array[float],
    deltas: wp.array[wp.vec3],
    split_deltas: wp.array[wp.vec3],
):
    """Solve unique edge-edge cloth self-contact candidates."""
    edge0 = wp.tid()
    collision_info = collision_info_array[0]
    collision_count = get_edge_colliding_edges_count(collision_info, edge0)

    for collision_index in range(collision_count):
        edge1 = get_edge_colliding_edges(collision_info, edge0, collision_index)
        if edge1 <= edge0:
            continue

        e00 = edge_indices[edge0, 2]
        e01 = edge_indices[edge0, 3]
        e10 = edge_indices[edge1, 2]
        e11 = edge_indices[edge1, 3]
        x00 = particle_x[e00]
        x01 = particle_x[e01]
        x10 = particle_x[e10]
        x11 = particle_x[e11]
        closest_parameters = wp.closest_point_edge_edge(x00, x01, x10, x11, edge_parallel_epsilon)
        s = closest_parameters[0]
        t = closest_parameters[1]
        point0 = x00 + (x01 - x00) * s
        point1 = x10 + (x11 - x10) * t
        diff = point0 - point1
        distance = wp.length(diff)

        x00_prev = particle_x_prev[e00]
        x01_prev = particle_x_prev[e01]
        x10_prev = particle_x_prev[e10]
        x11_prev = particle_x_prev[e11]
        point0_prev = x00_prev + (x01_prev - x00_prev) * s
        point1_prev = x10_prev + (x11_prev - x10_prev) * t
        diff_prev = point0_prev - point1_prev
        previous_distance = wp.length(diff_prev)

        normal = wp.vec3(0.0, 0.0, 1.0)
        if distance > 1.0e-8:
            normal = diff / distance
            if previous_distance > 1.0e-8 and wp.dot(normal, diff_prev) < 0.0:
                normal = -normal
        elif previous_distance > 1.0e-8:
            normal = diff_prev / previous_distance

        signed_distance = wp.dot(diff, normal)
        target_distance = collision_radius
        if max_depenetration_velocity >= 0.0:
            target_distance = wp.min(collision_radius, previous_distance + max_depenetration_velocity * dt)

        weights = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        indices = wp.vec4i(e00, e01, e10, e11)
        denominator = float(0.0)
        # Per-vertex divisors inject momentum when the two edges have
        # different contact valences.
        correction_count_sum = int(0)
        for local_index in range(4):
            particle = indices[local_index]
            weighted_inverse_mass = particle_invmass[particle] * weights[local_index] * weights[local_index]
            denominator += weighted_inverse_mass
            correction_count_sum += wp.max(contact_counts[particle], 1)
        jacobi_scale = 4.0 / float(correction_count_sum)
        split_denominator = denominator * jacobi_scale

        lambda_index = collision_info.edge_colliding_edges_offsets[edge0] + collision_index
        lambda_old = lambdas[lambda_index]
        constraint = signed_distance - target_distance
        if denominator > 0.0 and (constraint < 0.0 or lambda_old > 0.0):
            dlambda = -constraint / denominator
            lambda_new = wp.max(lambda_old + dlambda, 0.0)
            dlambda = (lambda_new - lambda_old) * relaxation
            lambda_new = lambda_old + dlambda
            lambdas[lambda_index] = lambda_new

            recovery_distance = wp.max(target_distance - previous_distance, 0.0)
            split_lambda_new = wp.min(lambda_new, recovery_distance / split_denominator)
            split_dlambda = split_lambda_new - split_lambdas[lambda_index]
            split_lambdas[lambda_index] = split_lambda_new

            relative_displacement = (point0 - point0_prev) - (point1 - point1_prev)
            tangent_displacement = relative_displacement - normal * wp.dot(relative_displacement, normal)
            tangent_distance = wp.length(tangent_displacement)
            tangent_impulse = wp.vec3(0.0)
            if tangent_distance > 1.0e-8 and friction_coefficient > 0.0:
                # Accumulate over the step and apply only the increment, so the
                # tangential impulse is bounded by mu * lambda_n once per step
                # instead of once per iteration.
                friction_lambda_old = friction_lambdas[lambda_index]
                friction_lambda_new = wp.min(
                    friction_lambda_old + tangent_distance / denominator * relaxation,
                    friction_coefficient * lambda_new,
                )
                friction_lambda_new = wp.max(friction_lambda_new, friction_lambda_old)
                friction_lambdas[lambda_index] = friction_lambda_new
                tangent_impulse = -tangent_displacement / tangent_distance * (friction_lambda_new - friction_lambda_old)

            impulse = normal * dlambda + tangent_impulse
            for local_index in range(4):
                particle = indices[local_index]
                correction = particle_invmass[particle] * weights[local_index] * impulse * jacobi_scale
                cap_scale = 1.0
                correction_length = wp.length(correction)
                max_correction = 0.25 * collision_radius
                if max_depenetration_velocity >= 0.0:
                    max_correction = max_depenetration_velocity * dt
                if correction_length > max_correction and correction_length > 0.0:
                    cap_scale = max_correction / correction_length
                    correction *= cap_scale
                wp.atomic_add(deltas, particle, correction)
                if split_dlambda != 0.0:
                    split_correction = (
                        particle_invmass[particle]
                        * weights[local_index]
                        * normal
                        * split_dlambda
                        * jacobi_scale
                        * cap_scale
                    )
                    wp.atomic_add(split_deltas, particle, split_correction)


@wp.kernel
def solve_vertex_triangle_self_contact_velocities(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    tri_indices: wp.array2d[int],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    friction_coefficient: float,
    dt: float,
    relaxation: float,
    contact_counts: wp.array[int],
    lambdas: wp.array[float],
    friction_lambdas: wp.array[float],
    velocity_deltas: wp.array[wp.vec3],
):
    """Stabilize normal motion and apply cloth self-contact friction."""
    vertex = wp.tid()
    collision_info = collision_info_array[0]
    collision_count = get_vertex_colliding_triangles_count(collision_info, vertex)

    for collision_index in range(collision_count):
        lambda_index = collision_info.vertex_colliding_triangles_offsets[vertex] + collision_index
        normal_support = lambdas[lambda_index] / dt
        if normal_support <= 0.0:
            continue
        # The Coulomb budget is per step, not per pass: subtract what the
        # positional solve already spent instead of granting a second full one.
        friction_budget = (
            wp.max(friction_coefficient * lambdas[lambda_index] - friction_lambdas[lambda_index], 0.0) / dt
        )

        triangle = get_vertex_colliding_triangles(collision_info, vertex, collision_index)
        t0 = tri_indices[triangle, 0]
        t1 = tri_indices[triangle, 1]
        t2 = tri_indices[triangle, 2]
        closest, bary, _feature = triangle_closest_point(
            particle_x[t0], particle_x[t1], particle_x[t2], particle_x[vertex]
        )
        diff = particle_x[vertex] - closest
        distance = wp.length(diff)
        if distance <= 1.0e-8:
            diff = wp.cross(particle_x[t1] - particle_x[t0], particle_x[t2] - particle_x[t0])
            distance = wp.length(diff)
            if distance <= 1.0e-8:
                continue
        normal = diff / distance

        weights = wp.vec4(1.0, -bary[0], -bary[1], -bary[2])
        indices = wp.vec4i(vertex, t0, t1, t2)
        relative_velocity = wp.vec3(0.0)
        denominator = float(0.0)
        correction_count_sum = int(0)
        for local_index in range(4):
            particle = indices[local_index]
            weight = weights[local_index]
            relative_velocity += weight * particle_v[particle]
            denominator += particle_invmass[particle] * weight * weight
            correction_count_sum += wp.max(contact_counts[particle], 1)
        if denominator <= 0.0:
            continue

        normal_velocity = wp.dot(relative_velocity, normal)
        impulse = wp.vec3(0.0)
        if normal_velocity < 0.0:
            impulse = -normal * normal_velocity / denominator
        tangent_velocity = relative_velocity - normal * normal_velocity
        tangent_speed = wp.length(tangent_velocity)
        if tangent_speed > 1.0e-8 and friction_coefficient > 0.0:
            tangent_impulse = wp.min(
                tangent_speed / denominator,
                friction_budget,
            )
            impulse -= tangent_velocity / tangent_speed * tangent_impulse
        impulse *= relaxation

        for local_index in range(4):
            particle = indices[local_index]
            wp.atomic_add(
                velocity_deltas,
                particle,
                particle_invmass[particle] * weights[local_index] * impulse * 4.0 / float(correction_count_sum),
            )


@wp.kernel
def solve_edge_edge_self_contact_velocities(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    edge_indices: wp.array2d[int],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    friction_coefficient: float,
    edge_parallel_epsilon: float,
    dt: float,
    relaxation: float,
    contact_counts: wp.array[int],
    lambdas: wp.array[float],
    friction_lambdas: wp.array[float],
    velocity_deltas: wp.array[wp.vec3],
):
    """Stabilize normal motion and apply edge-edge self-contact friction."""
    edge0 = wp.tid()
    collision_info = collision_info_array[0]
    collision_count = get_edge_colliding_edges_count(collision_info, edge0)

    for collision_index in range(collision_count):
        edge1 = get_edge_colliding_edges(collision_info, edge0, collision_index)
        if edge1 <= edge0:
            continue
        lambda_index = collision_info.edge_colliding_edges_offsets[edge0] + collision_index
        normal_support = lambdas[lambda_index] / dt
        if normal_support <= 0.0:
            continue
        # The Coulomb budget is per step, not per pass: subtract what the
        # positional solve already spent instead of granting a second full one.
        friction_budget = (
            wp.max(friction_coefficient * lambdas[lambda_index] - friction_lambdas[lambda_index], 0.0) / dt
        )

        e00 = edge_indices[edge0, 2]
        e01 = edge_indices[edge0, 3]
        e10 = edge_indices[edge1, 2]
        e11 = edge_indices[edge1, 3]
        closest_parameters = wp.closest_point_edge_edge(
            particle_x[e00],
            particle_x[e01],
            particle_x[e10],
            particle_x[e11],
            edge_parallel_epsilon,
        )
        s = closest_parameters[0]
        t = closest_parameters[1]
        point0 = particle_x[e00] + (particle_x[e01] - particle_x[e00]) * s
        point1 = particle_x[e10] + (particle_x[e11] - particle_x[e10]) * t
        diff = point0 - point1
        distance = wp.length(diff)
        if distance <= 1.0e-8:
            continue
        normal = diff / distance

        weights = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        indices = wp.vec4i(e00, e01, e10, e11)
        relative_velocity = wp.vec3(0.0)
        denominator = float(0.0)
        correction_count_sum = int(0)
        for local_index in range(4):
            particle = indices[local_index]
            weight = weights[local_index]
            relative_velocity += weight * particle_v[particle]
            denominator += particle_invmass[particle] * weight * weight
            correction_count_sum += wp.max(contact_counts[particle], 1)
        if denominator <= 0.0:
            continue

        normal_velocity = wp.dot(relative_velocity, normal)
        impulse = wp.vec3(0.0)
        if normal_velocity < 0.0:
            impulse = -normal * normal_velocity / denominator
        tangent_velocity = relative_velocity - normal * normal_velocity
        tangent_speed = wp.length(tangent_velocity)
        if tangent_speed > 1.0e-8 and friction_coefficient > 0.0:
            tangent_impulse = wp.min(
                tangent_speed / denominator,
                friction_budget,
            )
            impulse -= tangent_velocity / tangent_speed * tangent_impulse
        impulse *= relaxation

        for local_index in range(4):
            particle = indices[local_index]
            wp.atomic_add(
                velocity_deltas,
                particle,
                particle_invmass[particle] * weights[local_index] * impulse * 4.0 / float(correction_count_sum),
            )


@wp.kernel
def apply_particle_velocity_deltas(
    particle_flags: wp.array[wp.int32],
    deltas: wp.array[wp.vec3],
    static_friction_limits: wp.array[wp.vec4],
    max_velocity: float,
    particle_qd: wp.array[wp.vec3],
):
    """Apply a post-position particle velocity correction."""
    particle = wp.tid()
    if (particle_flags[particle] & ParticleFlags.ACTIVE) != 0:
        velocity = particle_qd[particle] + deltas[particle]
        static_limit = static_friction_limits[particle]
        capacity = static_limit[3]
        normal_sum = wp.vec3(static_limit[0], static_limit[1], static_limit[2])
        coherent_capacity = wp.length(normal_sum)
        if capacity > 0.0 and coherent_capacity > 0.95 * capacity:
            normal = normal_sum / coherent_capacity
            tangent_velocity = velocity - normal * wp.dot(normal, velocity)
            tangent_speed = wp.length(tangent_velocity)
            if tangent_speed <= coherent_capacity + 1.0e-8:
                velocity -= tangent_velocity
        speed = wp.length(velocity)
        if speed > max_velocity:
            velocity *= max_velocity / speed
        particle_qd[particle] = velocity


@wp.kernel
def remove_particle_split_velocity(
    particle_flags: wp.array[wp.int32],
    particle_qd_prev: wp.array[wp.vec3],
    split_deltas: wp.array[wp.vec3],
    dt: float,
    particle_qd: wp.array[wp.vec3],
):
    """Remove only velocity changes caused by overlap recovery."""
    particle = wp.tid()
    if (particle_flags[particle] & ParticleFlags.ACTIVE) != 0:
        split_velocity = split_deltas[particle] / dt
        split_speed_sq = wp.length_sq(split_velocity)
        if split_speed_sq > 0.0:
            velocity_change = particle_qd[particle] - particle_qd_prev[particle]
            scale = wp.clamp(wp.dot(velocity_change, split_velocity) / split_speed_sq, 0.0, 1.0)
            particle_qd[particle] -= scale * split_velocity


@wp.kernel
def remove_body_split_velocity(
    body_qd_prev: wp.array[wp.spatial_vector],
    split_velocity_deltas: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Remove only body velocity changes caused by overlap recovery."""
    body = wp.tid()
    split_velocity = split_velocity_deltas[body]
    split_linear = wp.spatial_top(split_velocity)
    split_angular = wp.spatial_bottom(split_velocity)
    split_speed_sq = wp.length_sq(split_linear) + wp.length_sq(split_angular)
    if split_speed_sq > 0.0:
        velocity_change = body_qd[body] - body_qd_prev[body]
        change_linear = wp.spatial_top(velocity_change)
        change_angular = wp.spatial_bottom(velocity_change)
        projected_change = wp.dot(change_linear, split_linear) + wp.dot(change_angular, split_angular)
        scale = wp.clamp(projected_change / split_speed_sq, 0.0, 1.0)
        body_qd[body] -= scale * split_velocity


@wp.kernel
def solve_springs(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    invmass: wp.array[float],
    spring_indices: wp.array[int],
    spring_rest_lengths: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    dt: float,
    lambdas: wp.array[float],
    delta: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)

    if l == 0.0:
        return

    n = xij / l

    c = l - rest
    grad_c_xi = n
    grad_c_xj = -1.0 * n

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj

    # Note strict inequality for damping -- 0 damping is ok
    if denom <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_c_dot_v = dt * wp.dot(grad_c_xi, vij)  # Note: dt because from the paper we want x_i - x^n, not v...
    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

    dxi = wi * dlambda * grad_c_xi
    dxj = wj * dlambda * grad_c_xj

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)


@wp.kernel
def count_bending_constraints(
    indices: wp.array2d[int],
    counts: wp.array[int],
):
    """Count valid dihedral constraints incident on each particle."""
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]
    if i == -1 or j == -1 or k == -1 or l == -1:
        return
    wp.atomic_add(counts, i, 1)
    wp.atomic_add(counts, j, 1)
    wp.atomic_add(counts, k, 1)
    wp.atomic_add(counts, l, 1)


@wp.kernel
def bending_constraint(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    invmass: wp.array[float],
    indices: wp.array2d[int],
    rest: wp.array[float],
    rest_length: wp.array[float],
    bending_properties: wp.array2d[float],
    particle_bending_constraint_counts: wp.array[int],
    dt: float,
    lambdas: wp.array[float],
    delta: wp.array[wp.vec3],
):
    tid = wp.tid()
    eps = 1.0e-6

    ke = bending_properties[tid, 0] * rest_length[tid]
    kd = bending_properties[tid, 1] * rest_length[tid]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    w1 = invmass[i]
    w2 = invmass[j]
    w3 = invmass[k]
    w4 = invmass[l]

    n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
    n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2
    e = x4 - x3

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)

    # Check for degenerate cases
    if n1_length < eps or n2_length < eps or e_length < eps:
        return

    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    e_hat = e / e_length

    cos_theta = wp.dot(n1_hat, n2_hat)
    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    c = theta - rest_angle
    if c > wp.pi:
        c -= 2.0 * wp.pi
    elif c < -wp.pi:
        c += 2.0 * wp.pi

    # Closed-form derivatives of the signed dihedral angle. These are
    # algebraically equivalent to differentiating the normalized face normals.
    grad_x1 = -n1_hat * e_length / n1_length
    grad_x2 = -n2_hat * e_length / n2_length
    grad_x3 = -n1_hat * wp.dot(x1 - x4, e_hat) / n1_length - n2_hat * wp.dot(x2 - x4, e_hat) / n2_length
    grad_x4 = -n1_hat * wp.dot(x3 - x1, e_hat) / n1_length - n2_hat * wp.dot(x3 - x2, e_hat) / n2_length

    denominator = (
        w1 * wp.length_sq(grad_x1)
        + w2 * wp.length_sq(grad_x2)
        + w3 * wp.length_sq(grad_x3)
        + w4 * wp.length_sq(grad_x4)
    )

    # Note strict inequality for damping -- 0 damping is ok
    if denominator <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_dot_v = dt * (wp.dot(grad_x1, v1) + wp.dot(grad_x2, v2) + wp.dot(grad_x3, v3) + wp.dot(grad_x4, v4))

    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

    count0 = wp.max(particle_bending_constraint_counts[i], 1)
    count1 = wp.max(particle_bending_constraint_counts[j], 1)
    count2 = wp.max(particle_bending_constraint_counts[k], 1)
    count3 = wp.max(particle_bending_constraint_counts[l], 1)
    delta0 = w1 * dlambda * grad_x1 / float(wp.max(count0, 1))
    delta1 = w2 * dlambda * grad_x2 / float(wp.max(count1, 1))
    delta2 = w3 * dlambda * grad_x3 / float(wp.max(count2, 1))
    delta3 = w4 * dlambda * grad_x4 / float(wp.max(count3, 1))

    # Jacobi averaging attenuates each particle correction by its incident
    # constraint count. Apply the same effective relaxation to the persistent
    # multiplier; otherwise the compliance term records a correction that was
    # never applied and suppresses all subsequent bending iterations.
    normalized_denominator = (
        w1 * wp.length_sq(grad_x1) / float(count0)
        + w2 * wp.length_sq(grad_x2) / float(count1)
        + w3 * wp.length_sq(grad_x3) / float(count2)
        + w4 * wp.length_sq(grad_x4) / float(count3)
    )
    lambdas[tid] = lambdas[tid] + dlambda * normalized_denominator / denominator

    wp.atomic_add(delta, i, delta0)
    wp.atomic_add(delta, j, delta1)
    wp.atomic_add(delta, k, delta2)
    wp.atomic_add(delta, l, delta3)


@wp.kernel
def solve_triangles(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    indices: wp.array2d[int],
    rest_pose: wp.array[wp.mat22],
    rest_area: wp.array[float],
    materials: wp.array2d[float],
    vertex_triangle_adjacency_offsets: wp.array[int],
    dt: float,
    relaxation: float,
    lambdas: wp.array2d[float],
    delta: wp.array[wp.vec3],
):
    """Solve isotropic membrane strain constraints for one triangle."""
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]

    pose = rest_pose[tid]
    d10 = pose[0, 0]
    d11 = pose[0, 1]
    d20 = pose[1, 0]
    d21 = pose[1, 1]

    x10 = x1 - x0
    x20 = x2 - x0
    f0 = x10 * d10 + x20 * d20
    f1 = x10 * d11 + x20 * d21

    # Green strain is split into one trace and two deviatoric constraints:
    #
    #   C_trace = E00 + E11
    #   C_diff  = E00 - E11
    #   C_shear = 2 E01
    #
    # This diagonalizes the isotropic St. Venant-Kirchhoff membrane energy,
    # giving stiffnesses lambda + mu, mu, and mu, respectively.
    e00 = 0.5 * (wp.dot(f0, f0) - 1.0)
    e11 = 0.5 * (wp.dot(f1, f1) - 1.0)
    e01_2 = wp.dot(f0, f1)

    df0_0 = -d10 - d20
    df0_1 = d10
    df0_2 = d20
    df1_0 = -d11 - d21
    df1_1 = d11
    df1_2 = d21

    grad_e00_0 = f0 * df0_0
    grad_e00_1 = f0 * df0_1
    grad_e00_2 = f0 * df0_2
    grad_e11_0 = f1 * df1_0
    grad_e11_1 = f1 * df1_1
    grad_e11_2 = f1 * df1_2

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]
    area = rest_area[tid]

    constraint = float(0.0)
    stiffness = float(0.0)
    grad0 = wp.vec3(0.0)
    grad1 = wp.vec3(0.0)
    grad2 = wp.vec3(0.0)

    for term in range(3):
        if term == 0:
            constraint = e00 + e11
            stiffness = area * (k_lambda + k_mu)
            grad0 = grad_e00_0 + grad_e11_0
            grad1 = grad_e00_1 + grad_e11_1
            grad2 = grad_e00_2 + grad_e11_2
        elif term == 1:
            constraint = e00 - e11
            stiffness = area * k_mu
            grad0 = grad_e00_0 - grad_e11_0
            grad1 = grad_e00_1 - grad_e11_1
            grad2 = grad_e00_2 - grad_e11_2
        else:
            constraint = e01_2
            stiffness = area * k_mu
            grad0 = f1 * df0_0 + f0 * df1_0
            grad1 = f1 * df0_1 + f0 * df1_1
            grad2 = f1 * df0_2 + f0 * df1_2

        denominator = w0 * wp.length_sq(grad0) + w1 * wp.length_sq(grad1) + w2 * wp.length_sq(grad2)
        if stiffness > 0.0 and denominator > 0.0:
            alpha = 1.0 / (stiffness * dt * dt)
            gamma = wp.max(k_damp, 0.0) * area / (stiffness * dt)
            grad_dot_v = dt * (wp.dot(grad0, v0) + wp.dot(grad1, v1) + wp.dot(grad2, v2))
            dlambda = -(constraint + alpha * lambdas[tid, term] + gamma * grad_dot_v) / (
                (1.0 + gamma) * denominator + alpha
            )
            dlambda *= relaxation

            count0 = (vertex_triangle_adjacency_offsets[i + 1] - vertex_triangle_adjacency_offsets[i]) >> 1
            count1 = (vertex_triangle_adjacency_offsets[j + 1] - vertex_triangle_adjacency_offsets[j]) >> 1
            count2 = (vertex_triangle_adjacency_offsets[k + 1] - vertex_triangle_adjacency_offsets[k]) >> 1
            count0 = wp.max(count0, 1)
            count1 = wp.max(count1, 1)
            count2 = wp.max(count2, 1)
            normalized_denominator = (
                w0 * wp.length_sq(grad0) / float(count0)
                + w1 * wp.length_sq(grad1) / float(count1)
                + w2 * wp.length_sq(grad2) / float(count2)
            )
            lambdas[tid, term] = lambdas[tid, term] + dlambda * normalized_denominator / denominator
            wp.atomic_add(delta, i, w0 * dlambda * grad0 / float(count0))
            wp.atomic_add(delta, j, w1 * dlambda * grad1 / float(count1))
            wp.atomic_add(delta, k, w2 * dlambda * grad2 / float(count2))


@wp.kernel
def solve_tetrahedra(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    indices: wp.array2d[int],
    rest_matrix: wp.array[wp.mat33],
    activation: wp.array[float],
    materials: wp.array2d[float],
    dt: float,
    relaxation: float,
    delta: wp.array[wp.vec3],
):
    # Tetrahedral XPBD constraint solve.
    #
    # ModelBuilder stores rest_matrix as inv(Dm), where
    # Dm = [x1_0 - x0_0, x2_0 - x0_0, x3_0 - x0_0] in the rest pose.  Each
    # iteration rebuilds Ds from the current particle positions and computes the
    # deformation gradient
    #
    #     F = Ds * inv(Dm).
    #
    # The material is the same compressible Neo-Hookean-style split used by the
    # FEM path: a distortional term controlled by the first Lame parameter
    # k_mu, and a volume term controlled by the second Lame parameter k_lambda.
    # In XPBD form these are solved as two scalar constraints:
    #
    #     C_dev = trace(F^T F) - 3
    #     C_vol = det(F) - 1 + activation
    #
    # Their gradients are dC/dF = 2F for C_dev and cof(F) for C_vol.  The chain
    # rule dF/dx contributes inv(Dm)^T, giving the per-particle gradients below.
    #
    # A tetrahedron's energy scales with rest volume V0, so the XPBD compliance
    # for a material stiffness k is 1 / (V0 * k).  Since rest_matrix is inv(Dm),
    # det(rest_matrix) * 6 = 1 / V0.
    #
    # Damping uses XPBD's compliant Rayleigh term:
    #
    #     gamma = k_damp / (k * dt)
    #     dlambda = -(C + gamma * dt * grad(C).dot(v))
    #               / ((1 + gamma) * sum_i(w_i |grad_i C|^2) + alpha)
    #
    # The solver does not persist lambdas for this constraint, so each iteration
    # computes a local multiplier and accumulates relaxed position corrections.
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = rest_matrix[tid]
    inv_QT = wp.transpose(Dm)

    inv_rest_volume = wp.determinant(Dm) * 6.0
    if inv_rest_volume <= 0.0 or k_mu <= 0.0 or k_lambda <= 0.0:
        return

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)

    C = float(0.0)
    dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    compliance = float(0.0)
    stiffness = float(0.0)

    num_terms = 2
    for term in range(0, num_terms):
        if term == 0:
            # deviatoric, stable
            C = tr - 3.0
            dC = F * 2.0
            compliance = inv_rest_volume / k_mu
            stiffness = k_mu
        elif term == 1:
            # volume conservation
            C = wp.determinant(F) - 1.0 + act
            dC = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            compliance = inv_rest_volume / k_lambda
            stiffness = k_lambda

        if C != 0.0:
            dP = dC * inv_QT
            grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
            grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
            grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
            grad0 = -grad1 - grad2 - grad3

            w = (
                wp.dot(grad0, grad0) * w0
                + wp.dot(grad1, grad1) * w1
                + wp.dot(grad2, grad2) * w2
                + wp.dot(grad3, grad3) * w3
            )

            if w > 0.0:
                alpha = compliance / dt / dt
                gamma = float(0.0)
                grad_dot_v = float(0.0)
                if k_damp > 0.0 and stiffness > 0.0:
                    gamma = k_damp / (stiffness * dt)
                    grad_dot_v = dt * (wp.dot(grad0, v0) + wp.dot(grad1, v1) + wp.dot(grad2, v2) + wp.dot(grad3, v3))
                dlambda = -1.0 * (C + gamma * grad_dot_v) / ((1.0 + gamma) * w + alpha)

                wp.atomic_add(delta, i, w0 * dlambda * grad0 * relaxation)
                wp.atomic_add(delta, j, w1 * dlambda * grad1 * relaxation)
                wp.atomic_add(delta, k, w2 * dlambda * grad2 * relaxation)
                wp.atomic_add(delta, l, w3 * dlambda * grad3 * relaxation)
                # wp.atomic_add(particle.num_corr, id0, 1)
                # wp.atomic_add(particle.num_corr, id1, 1)
                # wp.atomic_add(particle.num_corr, id2, 1)
                # wp.atomic_add(particle.num_corr, id3, 1)

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    # grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    # grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    # grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    # grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    # delta0 = grad0 * multiplier
    # delta1 = grad1 * multiplier
    # delta2 = grad2 * multiplier
    # delta3 = grad3 * multiplier

    # # hydrostatic part
    # J = wp.determinant(F)

    # C_vol = J - alpha
    # # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    # s = inv_rest_volume / 6.0
    # grad1 = wp.cross(x20, x30) * s
    # grad2 = wp.cross(x30, x10) * s
    # grad3 = wp.cross(x10, x20) * s
    # grad0 = -(grad1 + grad2 + grad3)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    # delta0 += grad0 * multiplier
    # delta1 += grad1 * multiplier
    # delta2 += grad2 * multiplier
    # delta3 += grad3 * multiplier

    # # # apply forces
    # # wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    # # wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    # # wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    # # wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def solve_tetrahedra2(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    indices: wp.array2d[int],
    pose: wp.array[wp.mat33],
    activation: wp.array[float],
    materials: wp.array2d[float],
    dt: float,
    relaxation: float,
    delta: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    # act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # C_sqrt
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # r_s = wp.sqrt(abs(tr - 3.0))
    # C = r_s

    # if (r_s == 0.0):
    #     return

    # if (tr < 3.0):
    #     r_s = 0.0 - r_s

    # dCdx = F*wp.transpose(Dm)*(1.0/r_s)
    # alpha = 1.0 + k_mu / k_lambda

    # C_Neo
    r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    if r_s == 0.0:
        return
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # if (tr < 3.0):
    #     r_s = -r_s
    r_s_inv = 1.0 / r_s
    C = r_s
    dCdx = F * wp.transpose(Dm) * r_s_inv
    alpha = 1.0 + k_mu / k_lambda

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    delta0 = grad0 * multiplier
    delta1 = grad1 * multiplier
    delta2 = grad2 * multiplier
    delta3 = grad3 * multiplier

    # hydrostatic part
    J = wp.determinant(F)

    C_vol = J - alpha
    # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    s = inv_rest_volume / 6.0
    grad1 = wp.cross(x20, x30) * s
    grad2 = wp.cross(x30, x10) * s
    grad3 = wp.cross(x10, x20) * s
    grad0 = -(grad1 + grad2 + grad3)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    delta0 += grad0 * multiplier
    delta1 += grad1 * multiplier
    delta2 += grad2 * multiplier
    delta3 += grad3 * multiplier

    # apply forces
    wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def apply_particle_deltas(
    x_orig: wp.array[wp.vec3],
    x_pred: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    delta: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0) / dt

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag
        x_new = x0 + v_new * dt

    x_out[tid] = x_new
    v_out[tid] = v_new


@wp.kernel
def apply_body_deltas(
    q_in: wp.array[wp.transform],
    qd_in: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_I: wp.array[wp.mat33],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    deltas: wp.array[wp.spatial_vector],
    constraint_inv_weights: wp.array[float],
    dt: float,
    # outputs
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    inv_m = body_inv_m[tid]
    if inv_m == 0.0:
        q_out[tid] = q_in[tid]
        qd_out[tid] = qd_in[tid]
        return
    inv_I = body_inv_I[tid]

    tf = q_in[tid]
    delta = deltas[tid]

    v0 = wp.spatial_top(qd_in[tid])
    w0 = wp.spatial_bottom(qd_in[tid])

    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    weight = 1.0
    if constraint_inv_weights:
        inv_weight = constraint_inv_weights[tid]
        if inv_weight > 0.0:
            weight = 1.0 / inv_weight

    dp = wp.spatial_top(delta) * (inv_m * weight)
    dq = wp.spatial_bottom(delta) * weight

    wb = wp.quat_rotate_inv(q0, w0)
    dwb = inv_I * wp.quat_rotate_inv(q0, dq)
    # coriolis forces delta from dwb = (wb + dwb) I (wb + dwb) - wb I wb
    tb = wp.cross(dwb, body_I[tid] * (wb + dwb)) + wp.cross(wb, body_I[tid] * dwb)
    dw1 = wp.quat_rotate(q0, dwb - dt * inv_I * tb)

    # update orientation
    q1 = q0 + 0.5 * wp.quat(dw1 * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # update position
    com = body_com[tid]
    x_com = p0 + wp.quat_rotate(q0, com)
    p1 = x_com + dp * dt
    p1 -= wp.quat_rotate(q1, com)

    q_out[tid] = wp.transform(p1, q1)

    # update linear and angular velocity
    v1 = v0 + dp
    w1 = w0 + dw1

    # XXX this improves gradient stability
    if wp.length(v1) < 1e-4:
        v1 = wp.vec3(0.0)
    if wp.length(w1) < 1e-4:
        w1 = wp.vec3(0.0)

    qd_out[tid] = wp.spatial_vector(v1, w1)


@wp.kernel
def apply_body_delta_velocities(
    deltas: wp.array[wp.spatial_vector],
    qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    wp.atomic_add(qd_out, tid, deltas[tid])


@wp.kernel
def apply_joint_forces(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    joint_f: wp.array[float],
    dt: float,
    body_f: wp.array[wp.spatial_vector],
    joint_impulse: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    type = joint_type[tid]
    if not joint_enabled[tid]:
        return
    if type == JointType.FIXED or type == JointType.ROD:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    # # local joint rotations
    # q_p = wp.transform_get_rotation(X_wp)
    # q_c = wp.transform_get_rotation(X_wc)

    # joint properties (for 1D joints)
    qd_start = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    if type == JointType.FREE or type == JointType.DISTANCE:
        f_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
        t_total = wp.vec3(joint_f[qd_start + 3], joint_f[qd_start + 4], joint_f[qd_start + 5])
        # Interpret free-joint forces as spatial wrench at the COM (same as body_f).
        # Avoid adding a moment arm that would introduce torque for pure forces.
        wp.atomic_add(body_f, id_c, wp.spatial_vector(f_total, t_total))
        if id_p >= 0:
            wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total))
        # Record the contribution to the inbound joint wrench (used to populate
        # ``State.body_parent_f``).  For FREE joints this is a diagnostic only;
        # for DISTANCE joints the constraint solver adds its own contribution.
        # Convention: positive = wrench transmitted parent->child at child COM.
        if joint_impulse:
            wp.atomic_add(joint_impulse, tid, wp.spatial_vector(f_total, t_total) * dt)
        return
    elif type == JointType.BALL:
        t_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])

    elif type == JointType.REVOLUTE or type == JointType.PRISMATIC or type == JointType.D6:
        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a dynamic for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            f = joint_f[qd_start + 0]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            f = joint_f[qd_start + 1]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            f = joint_f[qd_start + 2]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p

        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            f = joint_f[qd_start + lin_axis_count + 0]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            f = joint_f[qd_start + lin_axis_count + 1]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            f = joint_f[qd_start + lin_axis_count + 2]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p

    else:
        print("joint type not handled in apply_joint_forces")

    # write forces
    child_wrench_at_com = wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total))
    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))
    wp.atomic_add(body_f, id_c, child_wrench_at_com)

    # Record the joint-f contribution to the inbound joint wrench (used to
    # populate ``State.body_parent_f``).  We accumulate the child-side spatial
    # wrench (linear ``[N]``, torque ``[N·m]`` at the child COM, world frame)
    # multiplied by ``dt`` so that the same `impulse / dt` conversion applied
    # in :func:`convert_joint_impulse_to_parent_f` recovers the wrench.
    if joint_impulse:
        wp.atomic_add(joint_impulse, tid, child_wrench_at_com * dt)


@wp.func
def update_joint_axis_limits(axis: wp.vec3, limit_lower: float, limit_upper: float, input_limits: wp.spatial_vector):
    # update the 3D linear/angular limits (spatial_vector [lower, upper]) given the axis vector and limits
    lo_temp = axis * limit_lower
    up_temp = axis * limit_upper
    lo = vec_min(lo_temp, up_temp)
    up = vec_max(lo_temp, up_temp)
    input_lower = wp.spatial_top(input_limits)
    input_upper = wp.spatial_bottom(input_limits)
    lower = vec_min(input_lower, lo)
    upper = vec_max(input_upper, up)
    return wp.spatial_vector(lower, upper)


@wp.func
def update_joint_axis_weighted_target(
    axis: wp.vec3, target: float, weight: float, input_target_weight: wp.spatial_vector
):
    axis_targets = wp.spatial_top(input_target_weight)
    axis_weights = wp.spatial_bottom(input_target_weight)

    weighted_axis = axis * weight
    axis_targets += weighted_axis * target  # weighted target (to be normalized later by sum of weights)
    axis_weights += vec_abs(weighted_axis)

    return wp.spatial_vector(axis_targets, axis_weights)


@wp.func
def compute_linear_correction_3d(
    dx: wp.vec3,
    r1: wp.vec3,
    r2: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    c = wp.length(dx)
    if c == 0.0:
        # print("c == 0.0 in positional correction")
        return 0.0

    n = wp.normalize(dx)

    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)

    # Eq. 2-3 (make sure to project into the frame of the body)
    r1xn = wp.quat_rotate_inv(q1, wp.cross(r1, n))
    r2xn = wp.quat_rotate_inv(q2, wp.cross(r2, n))

    w1 = m_inv1 + wp.dot(r1xn, I_inv1 * r1xn)
    w2 = m_inv2 + wp.dot(r2xn, I_inv2 * r2xn)
    w = w1 + w2
    if w == 0.0:
        return 0.0
    alpha = compliance
    gamma = compliance * damping

    # Eq. 4-5
    d_lambda = -c - alpha * lambda_in
    # TODO consider damping for velocity correction?
    # delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if w + alpha > 0.0:
        d_lambda /= w * (dt + gamma) + alpha / dt

    return d_lambda


@wp.func
def compute_angular_correction_3d(
    corr: wp.vec3,
    q1: wp.quat,
    q2: wp.quat,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    relaxation: float,
    dt: float,
):
    # compute and apply the correction impulse for an angular constraint
    theta = wp.length(corr)
    if theta == 0.0:
        return 0.0

    n = wp.normalize(corr)

    # project variables to body rest frame as they are in local matrix
    n1 = wp.quat_rotate_inv(q1, n)
    n2 = wp.quat_rotate_inv(q2, n)

    # Eq. 11-12
    w1 = wp.dot(n1, I_inv1 * n1)
    w2 = wp.dot(n2, I_inv2 * n2)
    w = w1 + w2
    if w == 0.0:
        return 0.0

    # Eq. 13-14
    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w * dt + alpha_tilde / dt)
    # TODO consider lambda_prev?
    # p = d_lambda * n * relaxation

    # Eq. 15-16
    return d_lambda


@wp.kernel
def solve_simple_body_joints(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    joint_target: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_linear_compliance: float,
    joint_angular_compliance: float,
    angular_relaxation: float,
    linear_relaxation: float,
    dt: float,
    deltas: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    type = joint_type[tid]

    if not joint_enabled[tid]:
        return
    if type == JointType.FREE:
        return
    if type == JointType.DISTANCE:
        return
    if type == JointType.D6:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    # rel_pose = wp.transform_inverse(X_wp) * X_wc
    # rel_p = wp.transform_get_translation(rel_pose)

    # joint connection points
    # x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # linear_compliance = joint_linear_compliance
    angular_compliance = joint_angular_compliance
    damping = 0.0

    axis_start = joint_qd_start[tid]
    # mode = joint_dof_mode[axis_start]

    # local joint rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    inertial_q_p = wp.transform_get_rotation(pose_p)
    inertial_q_c = wp.transform_get_rotation(pose_c)

    # joint properties (for 1D joints)
    axis = joint_axis[axis_start]

    if type == JointType.FIXED:
        limit_lower = 0.0
        limit_upper = 0.0
    else:
        limit_lower = joint_limit_lower[axis_start]
        limit_upper = joint_limit_upper[axis_start]

    # linear_alpha_tilde = linear_compliance / dt / dt
    angular_alpha_tilde = angular_compliance / dt / dt

    # prevent division by zero
    # linear_alpha_tilde = wp.max(linear_alpha_tilde, 1e-6)
    # angular_alpha_tilde = wp.max(angular_alpha_tilde, 1e-6)

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    # handle angular constraints
    if type == JointType.REVOLUTE:
        # align joint axes
        a_p = wp.quat_rotate(q_p, axis)
        a_c = wp.quat_rotate(q_c, axis)
        # Eq. 20
        corr = wp.cross(a_p, a_c)
        ncorr = wp.normalize(corr)

        angular_relaxation = 0.2
        # angular_correction(
        #     corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
        #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
        lambda_n = compute_angular_correction_3d(
            corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, angular_alpha_tilde, damping, dt
        )
        lambda_n *= angular_relaxation
        ang_delta_p -= lambda_n * ncorr
        ang_delta_c += lambda_n * ncorr

        # limit joint angles (Alg. 3)
        pi = 3.14159265359
        two_pi = 2.0 * pi
        if limit_lower > -two_pi or limit_upper < two_pi:
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0] * a[0] / h, -a[0] * a[1] / h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            # b = c  # TODO verify

            # joint axis
            n = wp.quat_rotate(q_p, a)
            # the axes n1 and n2 are aligned with the two bodies
            n1 = wp.quat_rotate(q_p, b)
            n2 = wp.quat_rotate(q_c, b)

            phi = wp.asin(wp.dot(wp.cross(n1, n2), n))
            # print("phi")
            # print(phi)
            if wp.dot(n1, n2) < 0.0:
                phi = pi - phi
            if phi > pi:
                phi -= two_pi
            if phi < -pi:
                phi += two_pi
            if phi < limit_lower or phi > limit_upper:
                phi = wp.clamp(phi, limit_lower, limit_upper)
                # print("clamped phi")
                # print(phi)
                # rot = wp.quat(phi, n[0], n[1], n[2])
                # rot = wp.quat(n, phi)
                rot = wp.quat_from_axis_angle(n, phi)
                n1 = wp.quat_rotate(rot, n1)
                corr = wp.cross(n1, n2)
                # print("corr")
                # print(corr)
                # TODO expose
                # angular_alpha_tilde = 0.0001 / dt / dt
                # angular_relaxation = 0.5
                # TODO fix this constraint
                # angular_correction(
                #     corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
                #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
                lambda_n = compute_angular_correction_3d(
                    corr,
                    inertial_q_p,
                    inertial_q_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    angular_alpha_tilde,
                    damping,
                    dt,
                )
                lambda_n *= angular_relaxation
                ncorr = wp.normalize(corr)
                ang_delta_p -= lambda_n * ncorr
                ang_delta_c += lambda_n * ncorr

        # handle joint targets
        target_ke = joint_target_ke[axis_start]
        # target_kd = joint_target_kd[axis_start]
        target = joint_target[axis_start]
        if target_ke > 0.0:
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0] * a[0] / h, -a[0] * a[1] / h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            b = c

            q = wp.quat_from_axis_angle(a_p, target)
            b_target = wp.quat_rotate(q, wp.quat_rotate(q_p, b))
            b2 = wp.quat_rotate(q_c, b)
            # Eq. 21
            d_target = wp.cross(b_target, b2)

            target_compliance = 1.0 / target_ke  # / dt / dt
            # angular_correction(
            #     d_target, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
            #     target_compliance, angular_relaxation, deltas, id_p, id_c)
            lambda_n = compute_angular_correction_3d(
                d_target, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, target_compliance, damping, dt
            )
            lambda_n *= angular_relaxation
            ncorr = wp.normalize(d_target)
            # TODO fix
            ang_delta_p -= lambda_n * ncorr
            ang_delta_c += lambda_n * ncorr

    if (type == JointType.FIXED) or (type == JointType.PRISMATIC):
        # align the mutual orientations of the two bodies
        # Eq. 18-19
        q = q_p * wp.quat_inverse(q_c)
        corr = -2.0 * wp.vec3(q[0], q[1], q[2])
        # angular_correction(
        #     -corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
        #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
        lambda_n = compute_angular_correction_3d(
            corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, angular_alpha_tilde, damping, dt
        )
        lambda_n *= angular_relaxation
        ncorr = wp.normalize(corr)
        ang_delta_p -= lambda_n * ncorr
        ang_delta_c += lambda_n * ncorr

    # handle positional constraints

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # compute error between the joint attachment points on both bodies
    # delta x is the difference of point r_2 minus point r_1 (Fig. 3)
    dx = x_c - x_p

    # rotate the error vector into the joint frame
    q_dx = q_p
    # q_dx = q_c
    # q_dx = wp.transform_get_rotation(pose_p)
    dx = wp.quat_rotate_inv(q_dx, dx)

    lower_pos_limits = wp.vec3(0.0)
    upper_pos_limits = wp.vec3(0.0)
    if type == JointType.PRISMATIC:
        lower_pos_limits = axis * limit_lower
        upper_pos_limits = axis * limit_upper

    # compute linear constraint violations
    corr = wp.vec3(0.0)
    zero = wp.vec3(0.0)
    corr -= vec_leaky_min(zero, upper_pos_limits - dx)
    corr -= vec_leaky_max(zero, lower_pos_limits - dx)

    # if (type == JointType.PRISMATIC):
    #     if mode == JointMode.TARGET_POSITION:
    #         target = wp.clamp(target, limit_lower, limit_upper)
    #         if target_ke > 0.0:
    #             err = dx - target * axis
    #             compliance = 1.0 / target_ke
    #         damping = axis_damping[dim]
    #     elif mode == JointMode.TARGET_VELOCITY:
    #         if target_ke > 0.0:
    #             err = (derr - target) * dt
    #             compliance = 1.0 / target_ke
    #         damping = axis_damping[dim]

    # rotate correction vector into world frame
    corr = wp.quat_rotate(q_dx, corr)

    lambda_in = 0.0
    linear_alpha = joint_linear_compliance
    lambda_n = compute_linear_correction_3d(
        corr, r_p, r_c, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, lambda_in, linear_alpha, damping, dt
    )
    lambda_n *= linear_relaxation
    n = wp.normalize(corr)

    lin_delta_p -= n * lambda_n
    lin_delta_c += n * lambda_n
    ang_delta_p -= wp.cross(r_p, n) * lambda_n
    ang_delta_c += wp.cross(r_c, n) * lambda_n

    if id_p >= 0:
        wp.atomic_add(deltas, id_p, wp.spatial_vector(lin_delta_p, ang_delta_p))
    if id_c >= 0:
        wp.atomic_add(deltas, id_c, wp.spatial_vector(lin_delta_c, ang_delta_c))


@wp.kernel
def solve_body_joints(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_linear_compliance: float,
    joint_angular_compliance: float,
    angular_relaxation: float,
    linear_relaxation: float,
    dt: float,
    deltas: wp.array[wp.spatial_vector],
    joint_impulse: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    type = joint_type[tid]

    if not joint_enabled[tid]:
        return
    if type == JointType.FREE:
        return
    # if type == JointType.FIXED:
    #     return
    # if type == JointType.REVOLUTE:
    #     return
    # if type == JointType.PRISMATIC:
    #     return
    # if type == JointType.BALL:
    #     return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    vel_p = wp.vec3(0.0)
    omega_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
        vel_p = wp.spatial_top(body_qd[id_p])
        omega_p = wp.spatial_bottom(body_qd[id_p])

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    vel_c = wp.spatial_top(body_qd[id_c])
    omega_c = wp.spatial_bottom(body_qd[id_c])

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    rel_pose = wp.transform_inverse(X_wp) * X_wc
    rel_p = wp.transform_get_translation(rel_pose)

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    linear_compliance = joint_linear_compliance
    angular_compliance = joint_angular_compliance

    axis_start = joint_qd_start[tid]
    target_axis_start = joint_target_q_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    world_com_p = wp.transform_point(pose_p, com_p)
    world_com_c = wp.transform_point(pose_c, com_c)

    # handle positional constraints
    if type == JointType.DISTANCE:
        r_p = x_p - world_com_p
        r_c = x_c - world_com_c
        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]
        if lower < 0.0 and upper < 0.0:
            # no limits
            return
        anchor_delta = x_c - x_p
        d = wp.length(anchor_delta)
        err = 0.0
        if lower >= 0.0 and d < lower:
            err = d - lower
        elif upper >= 0.0 and d > upper:
            err = d - upper

        if wp.abs(err) > 1e-9:
            # compute gradients
            if d > 1e-9:
                linear_c = anchor_delta / d
            else:
                com_delta = world_com_c - world_com_p
                if wp.length_sq(com_delta) > 1e-18:
                    linear_c = wp.normalize(com_delta)
                else:
                    # The parent joint frame supplies a stable direction when the geometry cannot.
                    linear_c = wp.transform_vector(X_wp, wp.vec3(1.0, 0.0, 0.0))
            linear_p = -linear_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )
            lambda_in = 0.0
            compliance = linear_compliance
            ke = joint_target_ke[axis_start]
            if ke > 0.0:
                compliance = 1.0 / ke
            damping = joint_target_kd[axis_start]
            d_lambda = compute_positional_correction(
                err,
                derr,
                pose_p,
                pose_c,
                m_inv_p,
                m_inv_c,
                I_inv_p,
                I_inv_c,
                linear_p,
                linear_c,
                angular_p,
                angular_c,
                lambda_in,
                compliance,
                damping,
                dt,
            )

            lin_delta_p += linear_p * (d_lambda * linear_relaxation)
            ang_delta_p += angular_p * (d_lambda * angular_relaxation)
            lin_delta_c += linear_c * (d_lambda * linear_relaxation)
            ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    else:
        # compute joint target, stiffness, damping
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        axis_target_pos_ke = wp.spatial_vector()
        axis_target_vel_kd = wp.spatial_vector()
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if lin_axis_count > 0:
            axis = joint_axis[axis_start]
            lo_temp = axis * joint_limit_lower[axis_start]
            up_temp = axis * joint_limit_upper[axis_start]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            ke = joint_target_ke[axis_start]
            kd = joint_target_kd[axis_start]
            target_pos = joint_target_q[target_axis_start]
            target_vel = joint_target_qd[axis_start]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if lin_axis_count > 1:
            axis_idx = axis_start + 1
            target_axis_idx = target_axis_start + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if lin_axis_count > 2:
            axis_idx = axis_start + 2
            target_axis_idx = target_axis_start + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)

        axis_target_pos = wp.spatial_top(axis_target_pos_ke)
        axis_stiffness = wp.spatial_bottom(axis_target_pos_ke)
        axis_target_vel = wp.spatial_top(axis_target_vel_kd)
        axis_damping = wp.spatial_bottom(axis_target_vel_kd)
        for i in range(3):
            if axis_stiffness[i] > 0.0:
                axis_target_pos[i] /= axis_stiffness[i]
        for i in range(3):
            if axis_damping[i] > 0.0:
                axis_target_vel[i] /= axis_damping[i]
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        # A relative offset can contain both valid joint motion and error. For example,
        # a prismatic joint may be 0.5 m along its free axis and 1 m off it.
        # Start from the current offset so unconstrained coordinates keep their extension.
        projected_rel_p = rel_p
        for dim in range(3):
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            # Limit violations project to the nearest admissible boundary.
            if rel_p[dim] < lower:
                projected_rel_p[dim] = lower
            elif rel_p[dim] > upper:
                projected_rel_p[dim] = upper
            # A position-driven coordinate projects to its target. Locked coordinates
            # have zero-width limits above and therefore already project to zero.
            elif axis_stiffness[dim] > 0.0:
                projected_rel_p[dim] = wp.clamp(axis_target_pos[dim], lower, upper)

        frame_p = wp.quat_to_matrix(wp.transform_get_rotation(X_wp))
        # Use the admissible point for the parent lever arm: the parent anchor would
        # discard valid extension, while the child anchor would include separation error.
        r_p = wp.transform_point(X_wp, projected_rel_p) - world_com_p
        r_c = x_c - world_com_c

        # for loop will be unrolled, so we can modify local variables
        for dim in range(3):
            e = rel_p[dim]

            # compute gradients
            linear_c = wp.vec3(frame_p[0, dim], frame_p[1, dim], frame_p[2, dim])
            linear_p = -linear_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )

            err = 0.0
            compliance = linear_compliance
            damping = 0.0

            target_vel = axis_target_vel[dim]
            derr_rel = derr - target_vel

            # consider joint limits irrespective of axis mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
            elif e > upper:
                err = e - upper
            else:
                target_pos = axis_target_pos[dim]
                target_pos = wp.clamp(target_pos, lower, upper)

                if axis_stiffness[dim] > 0.0:
                    err = e - target_pos
                    compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif axis_damping[dim] > 0.0:
                    compliance = 1.0 / axis_damping[dim]
                    damping = axis_damping[dim]

            if wp.abs(err) > 1e-9 or wp.abs(derr_rel) > 1e-9:
                lambda_in = 0.0
                d_lambda = compute_positional_correction(
                    err,
                    derr_rel,
                    pose_p,
                    pose_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    linear_p,
                    linear_c,
                    angular_p,
                    angular_c,
                    lambda_in,
                    compliance,
                    damping,
                    dt,
                )

                lin_delta_p += linear_p * (d_lambda * linear_relaxation)
                ang_delta_p += angular_p * (d_lambda * angular_relaxation)
                lin_delta_c += linear_c * (d_lambda * linear_relaxation)
                ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    if type == JointType.FIXED or type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        # handle angular constraints

        # local joint rotations
        q_p = wp.transform_get_rotation(X_wp)
        q_c = wp.transform_get_rotation(X_wc)

        # make quats lie in same hemisphere
        if wp.dot(q_p, q_c) < 0.0:
            q_c *= -1.0

        rel_q = wp.quat_inverse(q_p) * q_c

        qtwist = wp.normalize(wp.quat(rel_q[0], 0.0, 0.0, rel_q[3]))
        qswing = rel_q * wp.quat_inverse(qtwist)

        # decompose to a compound rotation each axis
        s = wp.sqrt(rel_q[0] * rel_q[0] + rel_q[3] * rel_q[3])
        invs = 1.0 / s
        invscube = invs * invs * invs

        # handle axis-angle joints

        # rescale twist from quaternion space to angular
        err_0 = 2.0 * wp.asin(wp.clamp(qtwist[0], -1.0, 1.0))
        err_1 = qswing[1]
        err_2 = qswing[2]
        # analytic gradients of swing-twist decomposition
        grad_0 = wp.quat(invs - rel_q[0] * rel_q[0] * invscube, 0.0, 0.0, -(rel_q[3] * rel_q[0]) * invscube)
        grad_1 = wp.quat(
            -rel_q[3] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
            rel_q[3] * invs,
            -rel_q[0] * invs,
            rel_q[0] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
        )
        grad_2 = wp.quat(
            rel_q[3] * (rel_q[3] * rel_q[1] - rel_q[0] * rel_q[2]) * invscube,
            rel_q[0] * invs,
            rel_q[3] * invs,
            rel_q[0] * (rel_q[2] * rel_q[0] - rel_q[3] * rel_q[1]) * invscube,
        )
        grad_0 *= 2.0 / wp.abs(qtwist[3])
        # grad_0 *= 2.0 / wp.sqrt(1.0-qtwist[0]*qtwist[0])	# derivative of asin(x) = 1/sqrt(1-x^2)

        # rescale swing
        swing_sq = qswing[3] * qswing[3]
        # if swing axis magnitude close to zero vector, just treat in quaternion space
        angularEps = 1.0e-4
        if swing_sq + angularEps < 1.0:
            d = wp.sqrt(1.0 - qswing[3] * qswing[3])
            theta = 2.0 * wp.acos(wp.clamp(qswing[3], -1.0, 1.0))
            scale = theta / d

            err_1 *= scale
            err_2 *= scale

            grad_1 *= scale
            grad_2 *= scale

        errs = wp.vec3(err_0, err_1, err_2)
        grad_x = wp.vec3(grad_0[0], grad_1[0], grad_2[0])
        grad_y = wp.vec3(grad_0[1], grad_1[1], grad_2[1])
        grad_z = wp.vec3(grad_0[2], grad_1[2], grad_2[2])
        grad_w = wp.vec3(grad_0[3], grad_1[3], grad_2[3])

        # compute joint target, stiffness, damping
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        axis_target_pos_ke = wp.spatial_vector()  # [weighted_target_pos, ke_weights]
        axis_target_vel_kd = wp.spatial_vector()  # [weighted_target_vel, kd_weights]
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if ang_axis_count > 0:
            axis_idx = axis_start + lin_axis_count
            target_axis_idx = target_axis_start + lin_axis_count
            axis = joint_axis[axis_idx]
            lo_temp = axis * joint_limit_lower[axis_idx]
            up_temp = axis * joint_limit_upper[axis_idx]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if ang_axis_count > 1:
            axis_idx = axis_start + lin_axis_count + 1
            target_axis_idx = target_axis_start + lin_axis_count + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if ang_axis_count > 2:
            axis_idx = axis_start + lin_axis_count + 2
            target_axis_idx = target_axis_start + lin_axis_count + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)

        axis_target_pos = wp.spatial_top(axis_target_pos_ke)
        axis_stiffness = wp.spatial_bottom(axis_target_pos_ke)
        axis_target_vel = wp.spatial_top(axis_target_vel_kd)
        axis_damping = wp.spatial_bottom(axis_target_vel_kd)
        for i in range(3):
            if axis_stiffness[i] > 0.0:
                axis_target_pos[i] /= axis_stiffness[i]
        for i in range(3):
            if axis_damping[i] > 0.0:
                axis_target_vel[i] /= axis_damping[i]
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        # if type == JointType.D6:
        #     wp.printf("axis_target: %f %f %f\t axis_stiffness: %f %f %f\t axis_damping: %f %f %f\t axis_limits_lower: %f %f %f \t axis_limits_upper: %f %f %f\n",
        #               axis_target[0], axis_target[1], axis_target[2],
        #               axis_stiffness[0], axis_stiffness[1], axis_stiffness[2],
        #               axis_damping[0], axis_damping[1], axis_damping[2],
        #               axis_limits_lower[0], axis_limits_lower[1], axis_limits_lower[2],
        #               axis_limits_upper[0], axis_limits_upper[1], axis_limits_upper[2])
        #     # wp.printf("wp.sqrt(1.0-qtwist[0]*qtwist[0]) = %f\n", wp.sqrt(1.0-qtwist[0]*qtwist[0]))

        for dim in range(3):
            e = errs[dim]

            # analytic gradients of swing-twist decomposition
            grad = wp.quat(grad_x[dim], grad_y[dim], grad_z[dim], grad_w[dim])

            quat_c = 0.5 * q_p * grad * wp.quat_inverse(q_c)
            angular_c = wp.vec3(quat_c[0], quat_c[1], quat_c[2])
            angular_p = -angular_c
            # time derivative of the constraint
            derr = wp.dot(angular_p, omega_p) + wp.dot(angular_c, omega_c)

            err = 0.0
            compliance = angular_compliance
            damping = 0.0

            target_vel = axis_target_vel[dim]
            angular_c_len = wp.length(angular_c)
            derr_rel = derr - target_vel * angular_c_len

            # consider joint limits irrespective of mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
            elif e > upper:
                err = e - upper
            else:
                target_pos = axis_target_pos[dim]
                target_pos = wp.clamp(target_pos, lower, upper)

                if axis_stiffness[dim] > 0.0:
                    err = e - target_pos
                    compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif axis_damping[dim] > 0.0:
                    damping = axis_damping[dim]
                    compliance = 1.0 / axis_damping[dim]

            d_lambda = (
                compute_angular_correction(
                    err, derr_rel, pose_p, pose_c, I_inv_p, I_inv_c, angular_p, angular_c, 0.0, compliance, damping, dt
                )
                * angular_relaxation
            )

            # update deltas
            ang_delta_p += angular_p * d_lambda
            ang_delta_c += angular_c * d_lambda

    if id_p >= 0:
        wp.atomic_add(deltas, id_p, wp.spatial_vector(lin_delta_p, ang_delta_p))
    if id_c >= 0:
        wp.atomic_add(deltas, id_c, wp.spatial_vector(lin_delta_c, ang_delta_c))

    # Optionally accumulate the child-side spatial impulse for this joint.
    # The convention matches `body_parent_f`: incoming joint wrench in world
    # frame, referenced to the child body's COM (see `r_c` above which is
    # measured from the child COM).
    if joint_impulse:
        wp.atomic_add(joint_impulse, tid, wp.spatial_vector(lin_delta_c, ang_delta_c))


@wp.func
def compute_contact_constraint_delta(
    err: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3,
    linear_b: wp.vec3,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    relaxation: float,
    dt: float,
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a) * m_inv_a
    denom += wp.length_sq(linear_b) * m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    delta_lambda = -err
    if denom > 0.0:
        delta_lambda /= dt * denom

    return delta_lambda * relaxation


@wp.func
def compute_positional_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3,
    linear_b: wp.vec3,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a) * m_inv_a
    denom += wp.length_sq(linear_b) * m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        delta_lambda /= (dt + gamma) * denom + alpha / dt

    return delta_lambda


@wp.func
def compute_angular_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    denom = 0.0

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        delta_lambda /= (dt + gamma) * denom + alpha / dt

    return delta_lambda


@wp.kernel
def solve_body_contact_positions(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_material_mu_torsional: wp.array[float],
    shape_material_mu_rolling: wp.array[float],
    relaxation: float,
    dt: float,
    # outputs
    deltas: wp.array[wp.spatial_vector],
    contact_inv_weight: wp.array[float],
    contact_impulse: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    if body_a == body_b:
        return

    # find body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    # compute body position in world space
    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])

    n = contact_normal[tid]
    d = contact_surface_separation(bx_a, bx_b, n, contact_thickness0[tid], contact_thickness1[tid])

    if d >= 0.0:
        return

    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # center of mass in body frame
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    # body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    # angular velocities
    omega_a = wp.vec3(0.0)
    omega_b = wp.vec3(0.0)
    # contact offset in body frame
    offset_a = contact_offset0[tid]
    offset_b = contact_offset1[tid]

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        com_a = body_com[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        omega_a = wp.spatial_bottom(body_qd[body_a])

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        com_b = body_com[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        omega_b = wp.spatial_bottom(body_qd[body_b])

    # use average contact material properties
    mat_nonzero = 0
    mu = 0.0
    mu_torsional = 0.0
    mu_rolling = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        mu += shape_material_mu[shape_a]
        mu_torsional += shape_material_mu_torsional[shape_a]
        mu_rolling += shape_material_mu_rolling[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        mu += shape_material_mu[shape_b]
        mu_torsional += shape_material_mu_torsional[shape_b]
        mu_rolling += shape_material_mu_rolling[shape_b]
    if mat_nonzero > 0:
        mu /= float(mat_nonzero)
        mu_torsional /= float(mat_nonzero)
        mu_rolling /= float(mat_nonzero)

    r_a = bx_a - wp.transform_point(X_wb_a, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    angular_a = -wp.cross(r_a, n)
    angular_b = wp.cross(r_b, n)

    if contact_inv_weight:
        if body_a >= 0:
            wp.atomic_add(contact_inv_weight, body_a, 1.0)
        if body_b >= 0:
            wp.atomic_add(contact_inv_weight, body_b, 1.0)

    lambda_n = compute_contact_constraint_delta(
        d, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, -n, n, angular_a, angular_b, relaxation, dt
    )

    lin_delta_a = -n * lambda_n
    lin_delta_b = n * lambda_n
    ang_delta_a = angular_a * lambda_n
    ang_delta_b = angular_b * lambda_n

    # linear friction
    if mu > 0.0:
        # add on displacement from surface offsets, this ensures we include any rotational effects due to thickness from feature
        # need to use the current rotation to account for friction due to angular effects (e.g.: slipping contact)
        bx_a = contact_surface_point(X_wb_a, contact_point0[tid], offset_a)
        bx_b = contact_surface_point(X_wb_b, contact_point1[tid], offset_b)

        # update delta
        delta = bx_b - bx_a
        friction_delta = delta - wp.dot(n, delta) * n

        r_a = bx_a - wp.transform_point(X_wb_a, com_a)
        r_b = bx_b - wp.transform_point(X_wb_b, com_b)

        # Add only prescribed kinematic surface motion here.
        # Dynamic-body tangential motion is already reflected in the
        # positional slip `delta`; adding full relative velocity would
        # double-count ordinary ground friction and destabilize contacts.
        rel_v_kin_t = wp.vec3(0.0)
        if body_a >= 0 and (body_flags[body_a] & int(BodyFlags.KINEMATIC)) != 0:
            v_a = velocity_at_point(body_qd[body_a], r_a)
            rel_v_kin_t = rel_v_kin_t - (v_a - wp.dot(n, v_a) * n)
        if body_b >= 0 and (body_flags[body_b] & int(BodyFlags.KINEMATIC)) != 0:
            v_b = velocity_at_point(body_qd[body_b], r_b)
            rel_v_kin_t = rel_v_kin_t + (v_b - wp.dot(n, v_b) * n)
        friction_delta += rel_v_kin_t * dt

        perp = wp.normalize(friction_delta)

        angular_a = -wp.cross(r_a, perp)
        angular_b = wp.cross(r_b, perp)

        err = wp.length(friction_delta)

        if err > 0.0:
            lambda_fr = compute_contact_constraint_delta(
                err,
                X_wb_a,
                X_wb_b,
                m_inv_a,
                m_inv_b,
                I_inv_a,
                I_inv_b,
                -perp,
                perp,
                angular_a,
                angular_b,
                relaxation,
                dt,
            )

            # limit friction based on incremental normal force, good approximation to limiting on total force
            lambda_fr = wp.max(lambda_fr, -lambda_n * mu)

            lin_delta_a -= perp * lambda_fr
            lin_delta_b += perp * lambda_fr

            ang_delta_a += angular_a * lambda_fr
            ang_delta_b += angular_b * lambda_fr

    delta_omega = omega_b - omega_a

    if mu_torsional > 0.0:
        err = wp.dot(delta_omega, n) * dt

        if wp.abs(err) > 0.0:
            lin = wp.vec3(0.0)
            lambda_torsion = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -n, n, relaxation, dt
            )

            lambda_torsion = wp.clamp(lambda_torsion, -lambda_n * mu_torsional, lambda_n * mu_torsional)

            ang_delta_a -= n * lambda_torsion
            ang_delta_b += n * lambda_torsion

    if mu_rolling > 0.0:
        delta_omega -= wp.dot(n, delta_omega) * n
        err = wp.length(delta_omega) * dt
        if err > 0.0:
            lin = wp.vec3(0.0)
            roll_n = wp.normalize(delta_omega)
            lambda_roll = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -roll_n, roll_n, relaxation, dt
            )

            lambda_roll = wp.max(lambda_roll, -lambda_n * mu_rolling)

            ang_delta_a -= roll_n * lambda_roll
            ang_delta_b += roll_n * lambda_roll

    if body_a >= 0:
        wp.atomic_add(deltas, body_a, wp.spatial_vector(lin_delta_a, ang_delta_a))
    if body_b >= 0:
        wp.atomic_add(deltas, body_b, wp.spatial_vector(lin_delta_b, ang_delta_b))

    if contact_impulse:
        wp.atomic_add(contact_impulse, tid, wp.spatial_vector(lin_delta_a, ang_delta_a))


@wp.kernel
def accumulate_weighted_contact_impulse(
    contact_count: wp.array[int],
    contact_impulse_iter: wp.array[wp.spatial_vector],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_body: wp.array[int],
    constraint_inv_weight: wp.array[float],
    # output (accumulated across iterations)
    contact_impulse: wp.array[wp.spatial_vector],
):
    """Scale per-contact impulse from one iteration by 1/N and accumulate.

    ``constraint_inv_weight[body]`` holds the number of active contacts on
    each body for the current iteration.  ``apply_body_deltas`` divides the
    positional correction by that count, so the raw impulse stored per contact
    is N times too large relative to what was actually applied.

    When only one body is dynamic (the other is kinematic / ground), the
    weight is simply ``1/N_dynamic``.  When both bodies are dynamic the
    solver applies ``1/N_a`` to body A and ``1/N_b`` to body B, so there is
    no single exact scalar.  We use the harmonic mean ``2/(N_a + N_b)`` which
    is symmetric with respect to body ordering and reduces to ``1/N`` when
    both counts are equal.
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    impulse = contact_impulse_iter[tid]

    weight = 1.0
    if constraint_inv_weight:
        n_a = 0.0
        n_b = 0.0
        shape_a = contact_shape0[tid]
        if shape_a >= 0:
            body_a = shape_body[shape_a]
            if body_a >= 0:
                n_a = constraint_inv_weight[body_a]
        shape_b = contact_shape1[tid]
        if shape_b >= 0:
            body_b = shape_body[shape_b]
            if body_b >= 0:
                n_b = constraint_inv_weight[body_b]
        n_sum = n_a + n_b
        if n_sum > 0.0:
            if n_a == 0.0:
                weight = 1.0 / n_b
            elif n_b == 0.0:
                weight = 1.0 / n_a
            else:
                weight = 2.0 / n_sum

    scaled = wp.spatial_vector(
        wp.spatial_top(impulse) * weight,
        wp.spatial_bottom(impulse) * weight,
    )
    wp.atomic_add(contact_impulse, tid, scaled)


@wp.kernel
def convert_contact_impulse_to_force(
    contact_count: wp.array[int],
    contact_impulse: wp.array[wp.spatial_vector],
    dt: float,
    # output
    contact_force: wp.array[wp.spatial_vector],
):
    """Convert accumulated per-contact spatial impulse to ``contacts.force`` spatial vectors.

    The XPBD lambda convention used in this solver already absorbs one power
    of ``dt`` (see ``compute_contact_constraint_delta``), so dividing the
    accumulated impulse by the substep ``dt`` yields force [N] and torque [N·m].
    The linear component includes normal and friction forces; the angular
    component includes torsional and rolling friction torques.

    The impulse is expected to already include the 1/N contact-weighting
    correction (applied by ``accumulate_weighted_contact_impulse`` each
    iteration).
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        contact_force[tid] = wp.spatial_vector()
        return

    inv_dt = 1.0 / dt
    impulse = contact_impulse[tid]
    f = wp.spatial_top(impulse) * inv_dt
    tau = wp.spatial_bottom(impulse) * inv_dt
    contact_force[tid] = wp.spatial_vector(f, tau)


@wp.kernel
def convert_joint_impulse_to_parent_f(
    joint_impulse: wp.array[wp.spatial_vector],
    joint_enabled: wp.array[bool],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    dt: float,
    # output
    body_parent_f: wp.array[wp.spatial_vector],
):
    """Convert accumulated child-side joint impulse to ``state.body_parent_f``.

    The accumulated ``joint_impulse[joint_id]`` contains two contributions:

    * The XPBD constraint correction accumulated by ``solve_body_joints`` over
      every iteration.  The lambda convention used there already absorbs one
      power of ``dt`` (see ``compute_positional_correction`` /
      ``compute_angular_correction``), so dividing by the substep ``dt``
      yields the constraint reaction wrench.
    * The body-frame contribution from ``Control.joint_f`` recorded by
      ``apply_joint_forces``, pre-multiplied by ``dt`` for the same
      conversion to compose correctly.

    The result is the **total** wrench transmitted from the parent through the
    inbound joint to the child, expressed in world frame at the child body's
    COM (linear ``[N]``, torque ``[N·m]``).  This matches the convention used
    by :class:`SolverFeatherstone` and :class:`SolverMuJoCo`.

    Free joints and disabled joints contribute zero (their bodies inherit the
    zero-init from the caller).  Multiple joints sharing the same child body
    accumulate atomically, so loop-closure topologies remain race-free.
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return
    if joint_type[tid] == JointType.FREE:
        return

    id_c = joint_child[tid]
    if id_c < 0:
        return

    inv_dt = 1.0 / dt
    impulse = joint_impulse[tid]
    f = wp.spatial_top(impulse) * inv_dt
    tau = wp.spatial_bottom(impulse) * inv_dt
    wp.atomic_add(body_parent_f, id_c, wp.spatial_vector(f, tau))


@wp.kernel
def update_body_velocities(
    poses: wp.array[wp.transform],
    poses_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    dt: float,
    qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    pose = poses[tid]
    pose_prev = poses_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)

    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    # Update body velocities according to Alg. 2
    # XXX we consider the body COM as the origin of the body frame
    x_com = x + wp.quat_rotate(q, body_com[tid])
    x_com_prev = x_prev + wp.quat_rotate(q_prev, body_com[tid])

    # XXX consider the velocity of the COM
    v = (x_com - x_com_prev) / dt
    dq = q * wp.quat_inverse(q_prev)

    omega = 2.0 / dt * wp.vec3(dq[0], dq[1], dq[2])
    if dq[3] < 0.0:
        omega = -omega

    qd_out[tid] = wp.spatial_vector(v, omega)


@wp.kernel
def apply_rigid_restitution(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_qd_prev: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    body_world: wp.array[wp.int32],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_material_restitution: wp.array[float],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_inv_weight: wp.array[float],
    gravity: wp.array[wp.vec3],
    dt: float,
    # outputs
    deltas: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1

    # use average contact material properties
    mat_nonzero = 0
    restitution = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        restitution /= float(mat_nonzero)
    if body_a == body_b:
        return

    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # body to world transform
    X_wb_a_prev = wp.transform_identity()
    X_wb_b_prev = wp.transform_identity()
    # center of mass in body frame
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    # previous velocity at contact points
    v_a = wp.vec3(0.0)
    v_b = wp.vec3(0.0)
    # new velocity at contact points
    v_a_new = wp.vec3(0.0)
    v_b_new = wp.vec3(0.0)
    # inverse mass used to compute the impulse
    inv_mass = 0.0

    if body_a >= 0:
        X_wb_a_prev = body_q_prev[body_a]
        # X_wb_a = body_q[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        com_a = body_com[body_a]

    if body_b >= 0:
        X_wb_b_prev = body_q_prev[body_b]
        # X_wb_b = body_q[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        com_b = body_com[body_b]

    # compute body position in world space
    bx_a = contact_surface_point(X_wb_a_prev, contact_point0[tid], contact_offset0[tid])
    bx_b = contact_surface_point(X_wb_b_prev, contact_point1[tid], contact_offset1[tid])

    n = contact_normal[tid]
    d = wp.dot(n, bx_b - bx_a)
    if d >= 0.0:
        return

    r_a = bx_a - wp.transform_point(X_wb_a_prev, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b_prev, com_b)

    rxn_a = wp.vec3(0.0)
    rxn_b = wp.vec3(0.0)
    if body_a >= 0:
        world_idx_a = body_world[body_a]
        world_a_g = gravity[world_idx_a]
        v_a = velocity_at_point(body_qd_prev[body_a], r_a) + world_a_g * dt
        v_a_new = velocity_at_point(body_qd[body_a], r_a)
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        rxn_a = wp.quat_rotate_inv(q_a, wp.cross(r_a, n))
        # Eq. 2
        inv_mass_a = m_inv_a + wp.dot(rxn_a, I_inv_a * rxn_a)
        inv_mass += inv_mass_a
    if body_b >= 0:
        world_idx_b = body_world[body_b]
        world_b_g = gravity[world_idx_b]
        v_b = velocity_at_point(body_qd_prev[body_b], r_b) + world_b_g * dt
        v_b_new = velocity_at_point(body_qd[body_b], r_b)
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        rxn_b = wp.quat_rotate_inv(q_b, wp.cross(r_b, n))
        # Eq. 3
        inv_mass_b = m_inv_b + wp.dot(rxn_b, I_inv_b * rxn_b)
        inv_mass += inv_mass_b

    if inv_mass == 0.0:
        return

    # Eq. 29 — relative velocity of B w.r.t. A along the A-to-B normal
    rel_vel_old = wp.dot(n, v_b - v_a)
    rel_vel_new = wp.dot(n, v_b_new - v_a_new)

    if rel_vel_old >= 0.0:
        return

    # Eq. 34
    dv = (-rel_vel_new - restitution * rel_vel_old) / inv_mass

    # Eq. 33 — push A in -n direction, B in +n direction
    if body_a >= 0:
        dv_a = -dv
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        dq = wp.quat_rotate(q_a, I_inv_a * rxn_a * dv_a)
        wp.atomic_add(deltas, body_a, wp.spatial_vector(n * m_inv_a * dv_a, dq))

    if body_b >= 0:
        dv_b = dv
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        dq = wp.quat_rotate(q_b, I_inv_b * rxn_b * dv_b)
        wp.atomic_add(deltas, body_b, wp.spatial_vector(n * m_inv_b * dv_b, dq))
