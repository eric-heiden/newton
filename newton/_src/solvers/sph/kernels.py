# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...core.types import Axis
from ...geometry import GeoType, ParticleFlags, ShapeFlags
from ...geometry.kernels import (
    sample_sdf_grad_heightfield,
    sdf_box,
    sdf_box_grad,
    sdf_capsule,
    sdf_capsule_grad,
    sdf_cone,
    sdf_cone_grad,
    sdf_cylinder,
    sdf_cylinder_grad,
    sdf_ellipsoid,
    sdf_ellipsoid_grad,
    sdf_plane,
    sdf_sphere,
    sdf_sphere_grad,
)
from ...math import velocity_at_point
from ...sim import BodyFlags
from ...utils.heightfield import HeightfieldData

PI = wp.constant(3.141592653589793)
EPS = wp.constant(1.0e-6)
DIFFUSE_FREE_SLOT_SCAN = wp.constant(32)


@wp.func
def _is_active_particle(flags: wp.int32) -> bool:
    return (flags & ParticleFlags.ACTIVE) != 0


@wp.func
def _hash01(seed: float) -> float:
    value = wp.sin(seed) * 43758.5453123
    return value - wp.floor(value)


@wp.func
def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = wp.min(wp.max((x - edge0) / wp.max(edge1 - edge0, EPS), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def _vec4_xyz(v: wp.vec4) -> wp.vec3:
    return wp.vec3(v[0], v[1], v[2])


@wp.func
def _normalize_or(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    length = wp.length(v)
    result = fallback
    if length > EPS:
        result = v / length
    return result


@wp.func
def _orthogonal_axis(axis: wp.vec3) -> wp.vec3:
    seed = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(axis[2]) > 0.92:
        seed = wp.vec3(0.0, 1.0, 0.0)
    return _normalize_or(wp.cross(seed, axis), wp.vec3(1.0, 0.0, 0.0))


@wp.func
def _covariance_mul(
    cxx: float,
    cxy: float,
    cxz: float,
    cyy: float,
    cyz: float,
    czz: float,
    v: wp.vec3,
) -> wp.vec3:
    return wp.vec3(
        cxx * v[0] + cxy * v[1] + cxz * v[2],
        cxy * v[0] + cyy * v[1] + cyz * v[2],
        cxz * v[0] + cyz * v[1] + czz * v[2],
    )


@wp.func
def _poly6_kernel(r2: float, h: float) -> float:
    h2 = h * h
    result = float(0.0)
    if r2 < h2:
        x = h2 - r2
        h9 = h2 * h2 * h2 * h2 * h
        result = 315.0 / (64.0 * PI * h9) * x * x * x
    return result


@wp.func
def _spiky_gradient(r_vec: wp.vec3, r: float, h: float) -> wp.vec3:
    result = wp.vec3(0.0)
    if r > EPS and r < h:
        h6 = h * h * h * h * h * h
        x = h - r
        result = (-45.0 / (PI * h6) * x * x / r) * r_vec
    return result


@wp.func
def _viscosity_laplacian(r: float, h: float) -> float:
    result = float(0.0)
    if r < h:
        h6 = h * h * h * h * h * h
        result = 45.0 / (PI * h6) * (h - r)
    return result


@wp.func
def _cohesion_kernel(r: float, h: float) -> float:
    result = float(0.0)
    if r > EPS and r < h:
        q = 1.0 - r / h
        result = q * q * q
    return result


@wp.func
def _diffuse_visual_neighbors(
    neighbors: int,
    speed: float,
    smoothing_length: float,
    diffuse_ballistic: int,
) -> float:
    n = float(neighbors)
    ballistic_n = float(diffuse_ballistic)
    speed_scale = speed / wp.max(smoothing_length * 18.0, EPS)
    speed01 = _smoothstep(0.35, 1.25, speed_scale)
    ballistic = 1.0 - _smoothstep(ballistic_n - 1.0, ballistic_n + 2.0, n)
    submerged = _smoothstep(ballistic_n, ballistic_n + 8.0, n)

    spray_bias = ballistic * speed01 * wp.min(2.5 + n * 0.45, 5.5)
    bubble_bias = submerged * (1.0 + 2.2 * (1.0 - speed01))
    visual_neighbors = n - spray_bias + bubble_bias
    return wp.min(wp.max(visual_neighbors, 0.0), 28.0)


@wp.func
def _reserve_diffuse_slot(request: int, diffuse_slot_state: wp.array[wp.int32]) -> int:
    capacity = diffuse_slot_state.shape[0]
    scan_count = wp.min(capacity, int(DIFFUSE_FREE_SLOT_SCAN))
    slot = int(-1)
    offset = int(0)
    while offset < scan_count:
        candidate = (request + offset) % capacity
        old_state = wp.atomic_cas(diffuse_slot_state, candidate, 0, 1)
        if old_state == 0:
            slot = candidate
            break
        offset += 1

    if slot == -1:
        slot = request % capacity
        diffuse_slot_state[slot] = 1

    return slot


@wp.func
def _clamp_to_bounds(
    x: wp.vec3,
    v: wp.vec3,
    radius: float,
    lower: wp.vec3,
    upper: wp.vec3,
    damping: float,
):
    vx = v[0]
    vy = v[1]
    vz = v[2]
    px = x[0]
    py = x[1]
    pz = x[2]

    lo_x = lower[0] + radius
    lo_y = lower[1] + radius
    lo_z = lower[2] + radius
    hi_x = upper[0] - radius
    hi_y = upper[1] - radius
    hi_z = upper[2] - radius

    if px < lo_x:
        px = lo_x
        if vx < 0.0:
            vx = -vx * damping
    elif px > hi_x:
        px = hi_x
        if vx > 0.0:
            vx = -vx * damping

    if py < lo_y:
        py = lo_y
        if vy < 0.0:
            vy = -vy * damping
    elif py > hi_y:
        py = hi_y
        if vy > 0.0:
            vy = -vy * damping

    if pz < lo_z:
        pz = lo_z
        if vz < 0.0:
            vz = -vz * damping
    elif pz > hi_z:
        pz = hi_z
        if vz > 0.0:
            vz = -vz * damping

    return wp.vec3(px, py, pz), wp.vec3(vx, vy, vz)


@wp.func
def _clamp_velocity_delta(v: wp.vec3, v_ref: wp.vec3, max_acceleration: float, dt: float) -> wp.vec3:
    max_delta = wp.max(max_acceleration, 0.0) * wp.max(dt, 0.0)
    dv = v - v_ref
    dv_len = wp.length(dv)
    if dv_len > max_delta:
        v = v_ref + dv * (max_delta / dv_len)
    return v


@wp.func
def _apply_sleep_threshold(v: wp.vec3, sleep_threshold: float) -> wp.vec3:
    threshold = wp.max(sleep_threshold, 0.0)
    if threshold > 0.0 and wp.dot(v, v) < threshold * threshold:
        v = wp.vec3(0.0)
    return v


@wp.func
def _solid_pressure_axis(
    coord: float, lower: float, upper: float, radius: float, support: float, strength: float
) -> float:
    axis_accel = float(0.0)

    lower_dist = coord - (lower + radius)
    if lower_dist < support:
        q = wp.min(wp.max(1.0 - lower_dist / support, 0.0), 1.0)
        axis_accel += strength * q * q * (0.5 + 0.5 * q)

    upper_dist = (upper - radius) - coord
    if upper_dist < support:
        q = wp.min(wp.max(1.0 - upper_dist / support, 0.0), 1.0)
        axis_accel -= strength * q * q * (0.5 + 0.5 * q)

    return axis_accel


@wp.func
def _solid_pressure_bounds_accel(
    x: wp.vec3,
    radius: float,
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    smoothing_length: float,
    rest_density: float,
    solid_pressure: float,
) -> wp.vec3:
    support = wp.max(smoothing_length, radius + EPS)
    strength = wp.max(solid_pressure, 0.0) * wp.max(rest_density, EPS) * support
    return wp.vec3(
        _solid_pressure_axis(x[0], bounds_lower[0], bounds_upper[0], radius, support, strength),
        _solid_pressure_axis(x[1], bounds_lower[1], bounds_upper[1], radius, support, strength),
        _solid_pressure_axis(x[2], bounds_lower[2], bounds_upper[2], radius, support, strength),
    )


@wp.func
def _eval_primitive_shape_sdf(geo_type: int, x_local: wp.vec3, geo_scale: wp.vec3):
    d = float(1.0e8)
    n = wp.vec3(0.0, 0.0, 1.0)
    supported = False

    if geo_type == GeoType.SPHERE:
        d = sdf_sphere(x_local, geo_scale[0])
        n = sdf_sphere_grad(x_local, geo_scale[0])
        supported = True

    if geo_type == GeoType.BOX:
        d = sdf_box(x_local, geo_scale[0], geo_scale[1], geo_scale[2])
        n = sdf_box_grad(x_local, geo_scale[0], geo_scale[1], geo_scale[2])
        supported = True

    if geo_type == GeoType.CAPSULE:
        d = sdf_capsule(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_capsule_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        supported = True

    if geo_type == GeoType.CYLINDER:
        d = sdf_cylinder(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_cylinder_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        supported = True

    if geo_type == GeoType.CONE:
        d = sdf_cone(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_cone_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        supported = True

    if geo_type == GeoType.ELLIPSOID:
        d = sdf_ellipsoid(x_local, geo_scale)
        n = sdf_ellipsoid_grad(x_local, geo_scale)
        supported = True

    if geo_type == GeoType.PLANE:
        d = sdf_plane(x_local, geo_scale[0] * 0.5, geo_scale[1] * 0.5)
        n = wp.vec3(0.0, 0.0, 1.0)
        supported = True

    return d, n, supported


@wp.func
def _collide_point_with_shapes(
    x: wp.vec3,
    v: wp.vec3,
    radius: float,
    particle_mass: float,
    particle_world_id: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_margin: wp.array[float],
    shape_world: wp.array[wp.int32],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_count: int,
    boundary_damping: float,
    collision_distance: float,
    collision_margin: float,
    shape_restitution: float,
    shape_friction: float,
    shape_adhesion: float,
    dt: float,
    body_feedback: int,
):
    for shape_index in range(shape_count):
        if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
            continue

        shape_world_id = shape_world[shape_index]
        if particle_world_id != -1 and shape_world_id != -1 and particle_world_id != shape_world_id:
            continue

        body_index = shape_body[shape_index]
        X_wb = wp.transform_identity()
        if body_index >= 0:
            X_wb = body_q[body_index]

        X_ws = wp.transform_multiply(X_wb, shape_transform[shape_index])
        X_sw = wp.transform_inverse(X_ws)
        x_local = wp.transform_point(X_sw, x)

        geo_type = shape_type[shape_index]
        geo_scale = shape_scale[shape_index]
        d, n_local, supported = _eval_primitive_shape_sdf(geo_type, x_local, geo_scale)
        mesh_v_local = wp.vec3(0.0)
        if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
            min_scale = wp.min(wp.min(wp.abs(geo_scale[0]), wp.abs(geo_scale[1])), wp.abs(geo_scale[2]))
            if min_scale > EPS:
                mesh = shape_source_ptr[shape_index]
                base_clearance = radius
                if collision_distance >= 0.0:
                    base_clearance = collision_distance
                query_radius = (base_clearance + shape_margin[shape_index] + wp.max(collision_margin, 0.0)) / min_scale
                query = wp.mesh_query_point_sign_parity(mesh, wp.cw_div(x_local, geo_scale), query_radius)
                if query.result:
                    shape_p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                    shape_v = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
                    shape_p = wp.cw_mul(shape_p, geo_scale)
                    mesh_v_local = wp.cw_mul(shape_v, geo_scale)

                    delta = x_local - shape_p
                    delta_len = wp.length(delta)
                    if delta_len > EPS:
                        d = delta_len * query.sign
                        n_local = delta / delta_len * query.sign
                    else:
                        d = 0.0
                        n_local = _normalize_or(x_local, wp.vec3(0.0, 0.0, 1.0))
                    supported = True
        if geo_type == GeoType.HFIELD:
            hfield_index = shape_heightfield_index[shape_index]
            if hfield_index >= 0:
                d, n_local = sample_sdf_grad_heightfield(
                    heightfield_data[hfield_index],
                    heightfield_elevations,
                    x_local,
                )
                supported = True
        if not supported:
            continue

        base_clearance = radius
        if collision_distance >= 0.0:
            base_clearance = collision_distance
        clearance = wp.max(base_clearance, 0.0) + shape_margin[shape_index]
        contact_radius = clearance + wp.max(collision_margin, 0.0)
        if d >= contact_radius:
            continue

        n_world = wp.transform_vector(X_ws, n_local)
        normal_len = wp.length(n_world)
        if normal_len > EPS:
            n_world = n_world / normal_len
        else:
            shape_center = wp.transform_get_translation(X_ws)
            n_world = _normalize_or(x - shape_center, wp.vec3(0.0, 0.0, 1.0))

        penetration = clearance - d
        if penetration > 0.0:
            x = x + n_world * penetration

        body_v = wp.vec3(0.0)
        if body_index >= 0:
            body_v = velocity_at_point(
                body_qd[body_index],
                x - wp.transform_point(body_q[body_index], body_com[body_index]),
            )
        body_v += wp.transform_vector(X_ws, mesh_v_local)

        v_before = v
        rel_v = v - body_v
        normal_speed = wp.dot(rel_v, n_world)
        if normal_speed < 0.0:
            restitution = wp.min(wp.max(shape_restitution, 0.0), 1.0)
            rel_v = rel_v - n_world * ((1.0 + restitution) * normal_speed)
            v = body_v + rel_v

        if shape_friction > 0.0:
            rel_v = v - body_v
            tangent_v = rel_v - n_world * wp.dot(rel_v, n_world)
            friction_blend = wp.min(wp.max(shape_friction * dt, 0.0), 1.0)
            rel_v = rel_v - tangent_v * friction_blend
            v = body_v + rel_v

        if shape_adhesion > 0.0:
            rel_v = v - body_v
            separating_speed = wp.max(wp.dot(rel_v, n_world), 0.0)
            tangent_v = rel_v - n_world * wp.dot(rel_v, n_world)
            adhesion_blend = wp.min(wp.max(shape_adhesion * dt, 0.0), 1.0)
            rel_v = rel_v - n_world * (separating_speed * adhesion_blend)
            rel_v = rel_v - tangent_v * (0.25 * adhesion_blend)
            v = body_v + rel_v

        if body_feedback != 0 and body_index >= 0 and body_index < body_f.shape[0] and body_index < body_flags.shape[0]:
            if (body_flags[body_index] & BodyFlags.KINEMATIC) == 0:
                velocity_delta = v - v_before
                if wp.dot(velocity_delta, velocity_delta) > EPS * EPS:
                    particle_impulse = particle_mass * velocity_delta
                    body_force = -particle_impulse / wp.max(dt, EPS)
                    contact_point = x - n_world * clearance
                    body_origin = wp.transform_point(body_q[body_index], body_com[body_index])
                    r = contact_point - body_origin
                    wp.atomic_add(body_f, body_index, wp.spatial_vector(body_force, wp.cross(r, body_force)))

    return x, v


@wp.kernel
def compute_sph_density_pressure(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    smoothing_length: float,
    rest_density: float,
    gas_constant: float,
    out_density: wp.array[float],
    out_pressure: wp.array[float],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if not _is_active_particle(particle_flags[i]):
        out_density[i] = 0.0
        out_pressure[i] = 0.0
        return

    xi = particle_q[i]
    density = float(0.0)
    query = wp.hash_grid_query(grid, xi, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if _is_active_particle(particle_flags[j]):
            r = xi - particle_q[j]
            density += particle_mass[j] * _poly6_kernel(wp.dot(r, r), smoothing_length)

    density = wp.max(density, EPS)
    out_density[i] = density
    out_pressure[i] = wp.max(gas_constant * (density - rest_density), 0.0)


@wp.kernel
def compute_sph_vorticity(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    density: wp.array[float],
    smoothing_length: float,
    out_vorticity: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if not _is_active_particle(particle_flags[i]):
        out_vorticity[i] = wp.vec3(0.0)
        return

    xi = particle_q[i]
    vi = particle_qd[i]
    omega = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i and _is_active_particle(particle_flags[j]):
            r_vec = xi - particle_q[j]
            r = wp.length(r_vec)
            if r > EPS and r < smoothing_length:
                rho_j = wp.max(density[j], EPS)
                grad = _spiky_gradient(r_vec, r, smoothing_length)
                omega += particle_mass[j] / rho_j * wp.cross(particle_qd[j] - vi, grad)

    out_vorticity[i] = omega


@wp.kernel
def integrate_sph_particles(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    buoyancy: float,
    density: wp.array[float],
    pressure: wp.array[float],
    vorticity: wp.array[wp.vec3],
    smoothing_length: float,
    particle_collision_margin: float,
    rest_density: float,
    viscosity: float,
    particle_friction: float,
    cohesion: float,
    surface_tension: float,
    vorticity_confinement: float,
    solid_pressure: float,
    free_surface_drag: float,
    dissipation: float,
    velocity_damping: float,
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    boundary_damping: float,
    max_acceleration: float,
    max_velocity: float,
    sleep_threshold: float,
    dt: float,
    out_q: wp.array[wp.vec3],
    out_qd: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    xi = particle_q[i]
    vi = particle_qd[i]

    if not _is_active_particle(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        out_q[i] = xi
        out_qd[i] = vi
        return

    rho_i = wp.max(density[i], EPS)
    p_i = pressure[i]
    accel = particle_f[i] * particle_inv_mass[i]

    world_idx = particle_world[i]
    accel += gravity[wp.max(world_idx, 0)] * buoyancy
    if solid_pressure > 0.0:
        accel += _solid_pressure_bounds_accel(
            xi,
            particle_radius[i],
            bounds_lower,
            bounds_upper,
            smoothing_length,
            rest_density,
            solid_pressure,
        )

    contact_radius = wp.max(smoothing_length + particle_collision_margin, smoothing_length)
    query = wp.hash_grid_query(grid, xi, contact_radius)
    j = int(0)
    color_normal = wp.vec3(0.0)
    color_laplacian = float(0.0)
    omega_i = vorticity[i]
    eta = wp.vec3(0.0)
    contact_count = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i and _is_active_particle(particle_flags[j]):
            xj = particle_q[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < contact_radius and r > EPS:
                contact_count += 1
                if particle_friction > 0.0:
                    n_ij = r_vec / r
                    rel_v = vi - particle_qd[j]
                    tangent_v = rel_v - n_ij * wp.dot(rel_v, n_ij)
                    contact_weight = wp.min(wp.max(1.0 - r / contact_radius, 0.0), 1.0)
                    accel -= particle_friction * contact_weight * tangent_v

                if r < smoothing_length:
                    rho_j = wp.max(density[j], EPS)
                    p_j = pressure[j]
                    m_j = particle_mass[j]
                    grad = _spiky_gradient(r_vec, r, smoothing_length)
                    accel += -m_j * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * grad
                    accel += viscosity * m_j * (particle_qd[j] - vi) / rho_j * _viscosity_laplacian(r, smoothing_length)
                    if cohesion > 0.0:
                        accel += (
                            cohesion
                            * m_j
                            / rho_j
                            * _cohesion_kernel(r, smoothing_length)
                            * (xj - xi)
                            / smoothing_length
                        )
                    if surface_tension > 0.0:
                        color_normal += m_j / rho_j * grad
                        color_laplacian += m_j / rho_j * _viscosity_laplacian(r, smoothing_length)
                    if vorticity_confinement > 0.0:
                        eta += (wp.length(vorticity[j]) - wp.length(omega_i)) * grad

    if surface_tension > 0.0:
        normal_len = wp.length(color_normal)
        if normal_len > 0.015:
            accel += -surface_tension * color_laplacian * color_normal / normal_len

    if vorticity_confinement > 0.0:
        eta_len = wp.length(eta)
        omega_len = wp.length(omega_i)
        if eta_len > EPS and omega_len > EPS:
            accel += vorticity_confinement * wp.cross(eta / eta_len, omega_i)

    if free_surface_drag > 0.0:
        surface_deficit = (wp.max(rest_density, EPS) - rho_i) / wp.max(rest_density, EPS)
        surface_factor = wp.min(wp.max(surface_deficit / 0.18, 0.0), 1.0)
        accel -= vi * (free_surface_drag * surface_factor)

    if dissipation > 0.0 and contact_count > 0:
        contact_fraction = wp.min(float(contact_count) / 16.0, 1.0)
        accel -= vi * (dissipation * contact_fraction)

    accel_len = wp.length(accel)
    if accel_len > max_acceleration:
        accel *= max_acceleration / accel_len

    v_new = (vi + accel * dt) * wp.max(0.0, 1.0 - velocity_damping * dt)
    v_mag = wp.length(v_new)
    if v_mag > max_velocity:
        v_new *= max_velocity / v_mag
    v_new = _apply_sleep_threshold(v_new, sleep_threshold)

    x_new = xi + v_new * dt
    x_new, v_new = _clamp_to_bounds(
        x_new,
        v_new,
        particle_radius[i],
        bounds_lower,
        bounds_upper,
        boundary_damping,
    )

    out_q[i] = x_new
    out_qd[i] = v_new


@wp.kernel
def collide_sph_particles_with_shapes(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_margin: wp.array[float],
    shape_world: wp.array[wp.int32],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_count: int,
    boundary_damping: float,
    collision_distance: float,
    collision_margin: float,
    shape_restitution: float,
    shape_friction: float,
    shape_adhesion: float,
    dt: float,
    body_feedback: int,
):
    i = wp.tid()
    if not _is_active_particle(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        return

    x = particle_q[i]
    v = particle_qd[i]
    radius = particle_radius[i]
    particle_world_id = particle_world[i]

    x, v = _collide_point_with_shapes(
        x,
        v,
        radius,
        particle_mass[i],
        particle_world_id,
        body_q,
        body_qd,
        body_f,
        body_com,
        body_flags,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_flags,
        shape_margin,
        shape_world,
        shape_heightfield_index,
        heightfield_data,
        heightfield_elevations,
        shape_count,
        boundary_damping,
        collision_distance,
        collision_margin,
        shape_restitution,
        shape_friction,
        shape_adhesion,
        dt,
        body_feedback,
    )

    particle_q[i] = x
    particle_qd[i] = v


@wp.kernel
def sleep_sph_particles(
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    sleep_threshold: float,
):
    i = wp.tid()
    if not _is_active_particle(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        return

    particle_qd[i] = _apply_sleep_threshold(particle_qd[i], sleep_threshold)


@wp.kernel
def collide_sph_diffuse_particles_with_shapes(
    diffuse_q: wp.array[wp.vec4],
    diffuse_qd: wp.array[wp.vec4],
    diffuse_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_margin: wp.array[float],
    shape_world: wp.array[wp.int32],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_count: int,
    diffuse_radius: float,
    boundary_damping: float,
    collision_distance: float,
    collision_margin: float,
    shape_restitution: float,
    shape_friction: float,
    shape_adhesion: float,
):
    i = wp.tid()
    q_life = diffuse_q[i]
    life = q_life[3]
    if life <= 0.0:
        return

    v_neighbors = diffuse_qd[i]
    x = _vec4_xyz(q_life)
    v = _vec4_xyz(v_neighbors)

    x, v = _collide_point_with_shapes(
        x,
        v,
        diffuse_radius,
        0.0,
        diffuse_world[i],
        body_q,
        body_qd,
        body_f,
        body_com,
        body_flags,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_flags,
        shape_margin,
        shape_world,
        shape_heightfield_index,
        heightfield_data,
        heightfield_elevations,
        shape_count,
        boundary_damping,
        collision_distance,
        collision_margin,
        shape_restitution,
        shape_friction,
        shape_adhesion,
        1.0,
        0,
    )

    diffuse_q[i] = wp.vec4(x[0], x[1], x[2], life)
    diffuse_qd[i] = wp.vec4(v[0], v[1], v[2], v_neighbors[3])


@wp.kernel
def compute_sph_render_particles(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    smoothing_length: float,
    render_smoothing: float,
    anisotropy_scale: float,
    anisotropy_min: float,
    anisotropy_max: float,
    out_render_q: wp.array[wp.vec3],
    out_anisotropy: wp.array[wp.vec4],
    out_anisotropy_secondary: wp.array[wp.vec4],
    out_anisotropy_tertiary: wp.array[wp.vec4],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    xi = particle_q[i]
    if not _is_active_particle(particle_flags[i]):
        out_render_q[i] = xi
        out_anisotropy[i] = wp.vec4(1.0, 0.0, 0.0, 0.0)
        out_anisotropy_secondary[i] = wp.vec4(0.0, 1.0, 0.0, 0.0)
        out_anisotropy_tertiary[i] = wp.vec4(0.0, 0.0, 1.0, 0.0)
        return

    h = wp.max(smoothing_length, EPS)
    h2 = h * h
    weighted_center = wp.vec3(0.0)
    separation = wp.vec3(0.0)
    weight_sum = float(0.0)
    neighbor_count = int(0)

    query = wp.hash_grid_query(grid, xi, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if not _is_active_particle(particle_flags[j]):
            continue

        xj = particle_q[j]
        r_vec = xi - xj
        r2 = wp.dot(r_vec, r_vec)
        if r2 < h2:
            w = wp.pow(wp.max(1.0 - r2 / h2, 0.0), 2.0)
            weighted_center += xj * w
            weight_sum += w
            neighbor_count += 1

            if j != i and r2 > EPS:
                r = wp.sqrt(r2)
                separation += (r_vec / r) * (w * (1.0 - r / h))

    render_q = xi
    center = xi
    if weight_sum > EPS:
        center = weighted_center / weight_sum
        smoothing = wp.min(wp.max(render_smoothing, 0.0), 1.0)
        render_q = xi * (1.0 - smoothing) + center * smoothing

    out_render_q[i] = render_q

    axis = _normalize_or(separation, _normalize_or(particle_qd[i], wp.vec3(1.0, 0.37, 0.19)))
    side_axis = _orthogonal_axis(axis)
    depth_axis = _normalize_or(wp.cross(axis, side_axis), _orthogonal_axis(side_axis))
    stretch = float(1.0)
    side_scale = float(1.0)
    depth_scale = float(1.0)
    if neighbor_count >= 4 and anisotropy_scale > 0.0 and weight_sum > EPS:
        cxx = float(0.0)
        cxy = float(0.0)
        cxz = float(0.0)
        cyy = float(0.0)
        cyz = float(0.0)
        czz = float(0.0)

        query_cov = wp.hash_grid_query(grid, xi, h)
        k = int(0)
        while wp.hash_grid_query_next(query_cov, k):
            if not _is_active_particle(particle_flags[k]):
                continue

            xk = particle_q[k]
            r_vec = xi - xk
            r2 = wp.dot(r_vec, r_vec)
            if r2 < h2:
                w = wp.pow(wp.max(1.0 - r2 / h2, 0.0), 2.0)
                d = xk - center
                cxx += d[0] * d[0] * w
                cxy += d[0] * d[1] * w
                cxz += d[0] * d[2] * w
                cyy += d[1] * d[1] * w
                cyz += d[1] * d[2] * w
                czz += d[2] * d[2] * w

        inv_weight = 1.0 / wp.max(weight_sum, EPS)
        cxx *= inv_weight
        cxy *= inv_weight
        cxz *= inv_weight
        cyy *= inv_weight
        cyz *= inv_weight
        czz *= inv_weight

        regularizer = h2 * 0.0025
        cxx += regularizer
        cyy += regularizer
        czz += regularizer

        axis = _normalize_or(_covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, axis), axis)
        axis = _normalize_or(_covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, axis), axis)
        axis = _normalize_or(_covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, axis), axis)
        axis = _normalize_or(_covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, axis), axis)

        cov_axis = _covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, axis)
        major_var = wp.max(wp.dot(axis, cov_axis), 0.0)
        trace = wp.max(cxx + cyy + czz, 0.0)
        side_axis = _orthogonal_axis(axis)
        cov_side = _covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, side_axis)
        side_axis = _normalize_or(cov_side - axis * wp.dot(cov_side, axis), side_axis)
        cov_side = _covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, side_axis)
        side_axis = _normalize_or(cov_side - axis * wp.dot(cov_side, axis), side_axis)
        cov_side = _covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, side_axis)
        side_axis = _normalize_or(cov_side - axis * wp.dot(cov_side, axis), side_axis)
        cov_side = _covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, side_axis)
        side_var = wp.max(wp.dot(side_axis, cov_side), 0.0)
        depth_axis = _normalize_or(wp.cross(axis, side_axis), _orthogonal_axis(side_axis))
        cov_depth = _covariance_mul(cxx, cxy, cxz, cyy, cyz, czz, depth_axis)
        depth_var = wp.max(wp.dot(depth_axis, cov_depth), 0.0)
        minor_var = wp.max((trace - major_var) * 0.5, 0.0)
        if side_var > 0.0 or depth_var > 0.0:
            minor_var = wp.max((side_var + depth_var) * 0.5, 0.0)
        major_spread = wp.sqrt(major_var)
        minor_spread = wp.sqrt(minor_var)
        eccentricity = wp.max((major_spread - minor_spread) / wp.max(0.45 * h, EPS), 0.0)
        min_axis_scale = wp.max(anisotropy_min, 0.01)
        max_axis_scale = wp.max(anisotropy_max, min_axis_scale)
        major_min_scale = wp.max(min_axis_scale, 1.0)
        major_max_scale = wp.max(max_axis_scale, major_min_scale)
        stretch = 1.0 + anisotropy_scale * wp.min(eccentricity, major_max_scale - 1.0)
        stretch = wp.min(wp.max(stretch, major_min_scale), major_max_scale)
        stretch_strength = wp.min(wp.max((stretch - 1.0) / wp.max(major_max_scale - 1.0, EPS), 0.0), 1.0)
        minor_min_scale = wp.min(min_axis_scale, 1.0)
        minor_span = 1.0 - minor_min_scale
        side_scale = wp.min(wp.max(1.0 - stretch_strength * minor_span * 0.70, min_axis_scale), max_axis_scale)
        depth_scale = wp.min(wp.max(1.0 - stretch_strength * minor_span, min_axis_scale), max_axis_scale)

    out_anisotropy[i] = wp.vec4(axis[0], axis[1], axis[2], stretch)
    out_anisotropy_secondary[i] = wp.vec4(side_axis[0], side_axis[1], side_axis[2], side_scale)
    out_anisotropy_tertiary[i] = wp.vec4(depth_axis[0], depth_axis[1], depth_axis[2], depth_scale)


@wp.kernel
def compute_pbf_lambdas(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    smoothing_length: float,
    rest_density: float,
    relaxation_epsilon: float,
    out_density: wp.array[float],
    out_lambda: wp.array[float],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if not _is_active_particle(particle_flags[i]):
        out_density[i] = 0.0
        out_lambda[i] = 0.0
        return

    xi = particle_q[i]
    inv_rest_density = 1.0 / wp.max(rest_density, EPS)
    density = float(0.0)
    grad_i = wp.vec3(0.0)
    grad_sum = float(0.0)

    query = wp.hash_grid_query(grid, xi, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if not _is_active_particle(particle_flags[j]):
            continue

        r_vec = xi - particle_q[j]
        r2 = wp.dot(r_vec, r_vec)
        density += particle_mass[j] * _poly6_kernel(r2, smoothing_length)

        if j != i:
            r = wp.sqrt(r2)
            grad_j = -particle_mass[j] * inv_rest_density * _spiky_gradient(r_vec, r, smoothing_length)
            grad_sum += wp.dot(grad_j, grad_j)
            grad_i -= grad_j

    grad_sum += wp.dot(grad_i, grad_i)
    constraint = wp.max(density * inv_rest_density - 1.0, 0.0)
    out_density[i] = density
    out_lambda[i] = -constraint / (grad_sum + relaxation_epsilon)


@wp.kernel
def solve_pbf_deltas(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    lambdas: wp.array[float],
    smoothing_length: float,
    rest_density: float,
    artificial_pressure: float,
    artificial_radius: float,
    artificial_power: float,
    max_delta: float,
    out_delta: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if not _is_active_particle(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        out_delta[i] = wp.vec3(0.0)
        return

    xi = particle_q[i]
    inv_rest_density = 1.0 / wp.max(rest_density, EPS)
    lambda_i = lambdas[i]
    delta = wp.vec3(0.0)
    w_q = _poly6_kernel(artificial_radius * artificial_radius, smoothing_length)

    query = wp.hash_grid_query(grid, xi, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_particle(particle_flags[j]):
            continue

        r_vec = xi - particle_q[j]
        r = wp.length(r_vec)
        if r <= EPS or r >= smoothing_length:
            continue

        corr = float(0.0)
        if artificial_pressure > 0.0 and w_q > EPS:
            w = _poly6_kernel(wp.dot(r_vec, r_vec), smoothing_length)
            corr = -artificial_pressure * wp.pow(w / w_q, artificial_power)

        grad = _spiky_gradient(r_vec, r, smoothing_length)
        delta += (lambda_i + lambdas[j] + corr) * particle_mass[j] * inv_rest_density * grad

    delta_len = wp.length(delta)
    if max_delta > 0.0 and delta_len > max_delta:
        delta *= max_delta / delta_len

    out_delta[i] = delta


@wp.kernel
def apply_pbf_deltas(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    deltas: wp.array[wp.vec3],
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    relaxation: float,
):
    i = wp.tid()
    if not _is_active_particle(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        return

    x = particle_q[i] + deltas[i] * relaxation
    v = wp.vec3(0.0)
    x, v = _clamp_to_bounds(x, v, particle_radius[i], bounds_lower, bounds_upper, 0.0)
    particle_q[i] = x


@wp.kernel
def finalize_pbf_velocities(
    prev_q: wp.array[wp.vec3],
    prev_qd: wp.array[wp.vec3],
    projected_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    max_acceleration: float,
    max_velocity: float,
    dt: float,
    out_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if not _is_active_particle(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        return

    v = (projected_q[i] - prev_q[i]) / wp.max(dt, EPS)
    v = _clamp_velocity_delta(v, prev_qd[i], max_acceleration, dt)
    v_len = wp.length(v)
    if v_len > max_velocity:
        v *= max_velocity / v_len
    out_qd[i] = v


@wp.kernel
def smooth_sph_velocities(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    smoothing_length: float,
    xsph_strength: float,
    max_acceleration: float,
    max_velocity: float,
    dt: float,
    out_qd: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    vi = particle_qd[i]
    if not _is_active_particle(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        out_qd[i] = vi
        return

    xi = particle_q[i]
    h = wp.max(smoothing_length, EPS)
    h2 = h * h
    weighted_velocity = wp.vec3(0.0)
    weight_sum = float(0.0)

    query = wp.hash_grid_query(grid, xi, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if not _is_active_particle(particle_flags[j]):
            continue

        r_vec = xi - particle_q[j]
        r2 = wp.dot(r_vec, r_vec)
        if r2 < h2:
            w = wp.pow(wp.max(1.0 - r2 / h2, 0.0), 2.0)
            weighted_velocity += particle_qd[j] * w
            weight_sum += w

    v = vi
    if weight_sum > EPS:
        strength = wp.min(wp.max(xsph_strength, 0.0), 1.0)
        v = vi * (1.0 - strength) + (weighted_velocity / weight_sum) * strength

    v = _clamp_velocity_delta(v, vi, max_acceleration, dt)
    v_len = wp.length(v)
    if v_len > max_velocity:
        v *= max_velocity / v_len
    out_qd[i] = v


@wp.kernel
def copy_sph_velocities(
    in_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    out_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if _is_active_particle(particle_flags[i]) and particle_inv_mass[i] != 0.0:
        out_qd[i] = in_qd[i]


@wp.kernel
def update_sph_diffuse_particles(
    grid: wp.uint64,
    fluid_q: wp.array[wp.vec3],
    fluid_qd: wp.array[wp.vec3],
    fluid_flags: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    smoothing_length: float,
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    boundary_damping: float,
    diffuse_lifetime: float,
    diffuse_drag: float,
    diffuse_buoyancy: float,
    diffuse_ballistic: int,
    dt: float,
    diffuse_q: wp.array[wp.vec4],
    diffuse_qd: wp.array[wp.vec4],
    diffuse_world: wp.array[wp.int32],
    diffuse_slot_state: wp.array[wp.int32],
):
    tid = wp.tid()
    q_life = diffuse_q[tid]
    life = q_life[3]
    if life <= 0.0:
        diffuse_slot_state[tid] = 0
        return

    x = _vec4_xyz(q_life)
    v = _vec4_xyz(diffuse_qd[tid])
    world_idx = diffuse_world[tid]
    g = gravity[wp.max(world_idx, 0)]

    weighted_v = wp.vec3(0.0)
    weight_sum = float(0.0)
    neighbors = int(0)

    query = wp.hash_grid_query(grid, x, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if not _is_active_particle(fluid_flags[j]):
            continue

        r_vec = x - fluid_q[j]
        r2 = wp.dot(r_vec, r_vec)
        if r2 < smoothing_length * smoothing_length:
            w = _poly6_kernel(r2, smoothing_length)
            weighted_v += fluid_qd[j] * w
            weight_sum += w
            neighbors += 1

    if neighbors >= diffuse_ballistic and weight_sum > EPS:
        target_v = weighted_v / weight_sum
        blend = wp.min(wp.max(diffuse_drag * dt, 0.0), 1.0)
        v = v * (1.0 - blend) + target_v * blend
        # Buoyancy cancels a fraction of gravity (1 = neutrally buoyant); it
        # must not become a net upward thrust or foam launches out of the
        # water and hovers above the surface.
        v += g * ((1.0 - diffuse_buoyancy) * dt)
    else:
        v += g * dt

    x = x + v * dt
    x, v = _clamp_to_bounds(x, v, 0.0, bounds_lower, bounds_upper, boundary_damping)

    decay = dt / wp.max(diffuse_lifetime, EPS)
    life = wp.max(life - decay, 0.0)
    diffuse_q[tid] = wp.vec4(x[0], x[1], x[2], life)
    visual_neighbors = _diffuse_visual_neighbors(neighbors, wp.length(v), smoothing_length, diffuse_ballistic)
    diffuse_qd[tid] = wp.vec4(v[0], v[1], v[2], visual_neighbors)
    if life > 0.0:
        diffuse_slot_state[tid] = 1
    else:
        diffuse_slot_state[tid] = 0


@wp.kernel
def advance_sph_diffuse_seed(frame_seed: wp.array[wp.int32]):
    # Device-side spawn seed so the randomness keeps advancing inside a
    # captured CUDA graph, where Python-side counters are frozen.
    frame_seed[0] = frame_seed[0] + 1


@wp.kernel
def spawn_sph_diffuse_particles(
    grid: wp.uint64,
    fluid_q: wp.array[wp.vec3],
    fluid_qd: wp.array[wp.vec3],
    fluid_flags: wp.array[wp.int32],
    fluid_world: wp.array[wp.int32],
    density: wp.array[float],
    smoothing_length: float,
    rest_density: float,
    diffuse_threshold: float,
    diffuse_spawn_probability: float,
    diffuse_jitter: float,
    diffuse_surface_density_ratio: float,
    diffuse_ballistic: int,
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    boundary_damping: float,
    frame_seed: wp.array[wp.int32],
    diffuse_spawn_counter: wp.array[wp.int32],
    diffuse_q: wp.array[wp.vec4],
    diffuse_qd: wp.array[wp.vec4],
    diffuse_world: wp.array[wp.int32],
    diffuse_slot_state: wp.array[wp.int32],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1 or not _is_active_particle(fluid_flags[i]):
        return

    xi = fluid_q[i]
    vi = fluid_qd[i]
    speed_sq = wp.dot(vi, vi)
    divergence = float(0.0)
    neighbors = int(0)
    separation = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_particle(fluid_flags[j]):
            continue

        r_vec = xi - fluid_q[j]
        r = wp.length(r_vec)
        if r > EPS and r < smoothing_length:
            rel_v = vi - fluid_qd[j]
            divergence += wp.max(wp.dot(rel_v, r_vec / r), 0.0)
            separation += (r_vec / r) * (1.0 - r / smoothing_length)
            neighbors += 1

    is_surface = density[i] < rest_density * diffuse_surface_density_ratio or neighbors < diffuse_ballistic

    # Wave-crest weighting: `separation` approximates the outward surface
    # normal, so velocity aligned with it marks breaking crests, plunging
    # sheets, and beach run-up fronts. Fast laminar bulk flow has little
    # separating motion and no crest alignment, so it stays foam-free.
    speed = wp.sqrt(speed_sq)
    crest = float(0.0)
    separation_len = wp.length(separation)
    if separation_len > EPS and speed > EPS:
        crest = wp.max(wp.dot(separation / separation_len, vi / speed), 0.0)
    potential = divergence * (0.3 + 0.7 * crest) + 0.5 * speed_sq * crest * crest
    threshold = wp.max(diffuse_threshold, EPS)
    if potential <= threshold:
        return

    surface_scale = float(1.0)
    if not is_surface:
        # Keep only a trickle of submerged bubbles; whitewater belongs to the
        # free surface.
        surface_scale = 0.06

    probability = wp.min(diffuse_spawn_probability * surface_scale * (potential / threshold - 1.0), 1.0)
    seed = float((i + 1) * 928371 + (frame_seed[0] + 17) * 68917)
    if _hash01(seed) > probability:
        return

    request = wp.atomic_add(diffuse_spawn_counter, 0, 1)
    slot = _reserve_diffuse_slot(request, diffuse_slot_state)
    a = _hash01(seed + 13.0) * 6.28318530718
    z = _hash01(seed + 29.0) * 2.0 - 1.0
    rxy = wp.sqrt(wp.max(1.0 - z * z, 0.0))
    jitter_dir = wp.vec3(wp.cos(a) * rxy, wp.sin(a) * rxy, z)
    jitter = jitter_dir * (diffuse_jitter * _hash01(seed + 47.0))

    normal = separation
    normal_len = wp.length(normal)
    if normal_len > EPS:
        normal /= normal_len
    else:
        normal = jitter_dir

    spray_speed = wp.sqrt(wp.max(speed_sq, 0.0)) * (0.10 + 0.22 * surface_scale)
    spray_speed += wp.min(divergence, threshold * 4.0) / wp.max(float(neighbors), 1.0) * 0.16
    tangent = jitter_dir - normal * wp.dot(jitter_dir, normal)
    tangent_len = wp.length(tangent)
    if tangent_len > EPS:
        tangent /= tangent_len
    else:
        tangent = jitter_dir
    velocity_jitter = normal * spray_speed + tangent * (spray_speed * 0.22 * _hash01(seed + 61.0))

    spawn_x = xi + jitter
    spawn_v = vi + velocity_jitter
    spawn_x, spawn_v = _clamp_to_bounds(spawn_x, spawn_v, 0.0, bounds_lower, bounds_upper, boundary_damping)

    initial_life = 0.35 + 0.65 * _hash01(seed + 83.0)
    diffuse_q[slot] = wp.vec4(spawn_x[0], spawn_x[1], spawn_x[2], initial_life)
    visual_neighbors = _diffuse_visual_neighbors(neighbors, wp.length(spawn_v), smoothing_length, diffuse_ballistic)
    diffuse_qd[slot] = wp.vec4(
        spawn_v[0],
        spawn_v[1],
        spawn_v[2],
        visual_neighbors,
    )
    diffuse_world[slot] = fluid_world[i]
