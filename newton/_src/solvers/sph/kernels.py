# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags

PI = wp.constant(3.141592653589793)
EPS = wp.constant(1.0e-6)


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

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        out_density[i] = 0.0
        out_pressure[i] = 0.0
        return

    xi = particle_q[i]
    density = float(0.0)
    query = wp.hash_grid_query(grid, xi, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0:
            r = xi - particle_q[j]
            density += particle_mass[j] * _poly6_kernel(wp.dot(r, r), smoothing_length)

    density = wp.max(density, EPS)
    out_density[i] = density
    out_pressure[i] = wp.max(gas_constant * (density - rest_density), 0.0)


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
    density: wp.array[float],
    pressure: wp.array[float],
    smoothing_length: float,
    viscosity: float,
    velocity_damping: float,
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    boundary_damping: float,
    max_velocity: float,
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

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[i] == 0.0:
        out_q[i] = xi
        out_qd[i] = vi
        return

    rho_i = wp.max(density[i], EPS)
    p_i = pressure[i]
    accel = particle_f[i] * particle_inv_mass[i]

    world_idx = particle_world[i]
    accel += gravity[wp.max(world_idx, 0)]

    query = wp.hash_grid_query(grid, xi, smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i and (particle_flags[j] & ParticleFlags.ACTIVE) != 0:
            xj = particle_q[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < smoothing_length and r > EPS:
                rho_j = wp.max(density[j], EPS)
                p_j = pressure[j]
                m_j = particle_mass[j]
                grad = _spiky_gradient(r_vec, r, smoothing_length)
                accel += -m_j * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * grad
                accel += viscosity * m_j * (particle_qd[j] - vi) / rho_j * _viscosity_laplacian(r, smoothing_length)

    v_new = (vi + accel * dt) * wp.max(0.0, 1.0 - velocity_damping * dt)
    v_mag = wp.length(v_new)
    if v_mag > max_velocity:
        v_new *= max_velocity / v_mag

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
