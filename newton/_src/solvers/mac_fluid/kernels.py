# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the MAC fluid solver step stages.

Sign conventions for impulse bookkeeping: impulses accumulated into the
diagnostics buffers are momentum *added to the fluid*; per-body wrench
impulses are the equal-and-opposite reaction applied to the rigid body.
"""

from __future__ import annotations

import warp as wp

from .boundary import body_point_velocity, face_solid_owner
from .grid import (
    CELL_FLUID,
    face_position,
    is_solid_cell,
    sample_u,
    sample_v,
    sample_velocity,
    sample_w,
)

# diagnostics vector slots (momentum bookkeeping, world frame)
DIAG_V_MOMENTUM_PRE = wp.constant(0)  # fluid momentum after advection [kg m/s]
DIAG_V_MOMENTUM_POST = wp.constant(1)  # fluid momentum after projection [kg m/s]
DIAG_V_IMPULSE_EXTERNAL = wp.constant(2)  # gravity + external impulse [kg m/s]
DIAG_V_IMPULSE_VISCOUS = wp.constant(3)  # viscous boundary impulse on fluid [kg m/s]
DIAG_V_IMPULSE_PRESSURE = wp.constant(4)  # pressure boundary impulse on fluid [kg m/s]
DIAG_V_COUNT = 5

# diagnostics scalar slots
DIAG_S_DIV_L2_PRE = wp.constant(0)  # sum of squared divergence before projection
DIAG_S_DIV_LINF_PRE = wp.constant(1)
DIAG_S_DIV_L2_POST = wp.constant(2)  # sum of squared divergence after projection
DIAG_S_DIV_LINF_POST = wp.constant(3)
DIAG_S_PRESSURE_RESIDUAL = wp.constant(4)  # squared residual norm of the pressure solve
DIAG_S_NOSLIP_MAX = wp.constant(5)  # max tangential slip at solid boundaries [m/s]
DIAG_S_NOSLIP_SUM = wp.constant(6)
DIAG_S_NOSLIP_COUNT = wp.constant(7)
DIAG_S_COUNT = 8


@wp.func
def _face_neighbor_cells(axis: int, i: int, j: int, k: int):
    """Indices of the lower neighbor cell of face ``(i, j, k)``; upper is ``(i, j, k)``."""
    ia = i
    ja = j
    ka = k
    if axis == 0:
        ia = i - 1
    elif axis == 1:
        ja = j - 1
    else:
        ka = k - 1
    return ia, ja, ka


@wp.func
def _is_pure_fluid_face(cell_label: wp.array3d[wp.int32], axis: int, i: int, j: int, k: int) -> bool:
    ia, ja, ka = _face_neighbor_cells(axis, i, j, k)
    if is_solid_cell(cell_label, ia, ja, ka):
        return False
    return not is_solid_cell(cell_label, i, j, k)


# ---------------------------------------------------------------------------
# advection (semi-Lagrangian, RK2 backtrace, trilinear MAC interpolation)
# ---------------------------------------------------------------------------


@wp.func
def _backtrace(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    origin: wp.vec3,
    dx: float,
    pos: wp.vec3,
    dt: float,
) -> wp.vec3:
    vel = sample_velocity(u, v, w, origin, dx, pos)
    mid = pos - 0.5 * dt * vel
    vel_mid = sample_velocity(u, v, w, origin, dx, mid)
    return pos - dt * vel_mid


@wp.kernel(enable_backward=False)
def advect_u_kernel(
    origin: wp.vec3,
    dx: float,
    dt: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    q: wp.array3d[float],
    q_out: wp.array3d[float],
):
    """Advect x-face quantity ``q`` through the (u, v, w) velocity field."""
    i, j, k = wp.tid()
    if not _is_pure_fluid_face(cell_label, 0, i, j, k):
        q_out[i, j, k] = q[i, j, k]
        return
    pos = face_position(origin, dx, 0, i, j, k)
    back = _backtrace(u, v, w, origin, dx, pos, dt)
    q_out[i, j, k] = sample_u(q, origin, dx, back)


@wp.kernel(enable_backward=False)
def advect_v_kernel(
    origin: wp.vec3,
    dx: float,
    dt: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    q: wp.array3d[float],
    q_out: wp.array3d[float],
):
    """Advect y-face quantity ``q`` through the (u, v, w) velocity field."""
    i, j, k = wp.tid()
    if not _is_pure_fluid_face(cell_label, 1, i, j, k):
        q_out[i, j, k] = q[i, j, k]
        return
    pos = face_position(origin, dx, 1, i, j, k)
    back = _backtrace(u, v, w, origin, dx, pos, dt)
    q_out[i, j, k] = sample_v(q, origin, dx, back)


@wp.kernel(enable_backward=False)
def advect_w_kernel(
    origin: wp.vec3,
    dx: float,
    dt: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    q: wp.array3d[float],
    q_out: wp.array3d[float],
):
    """Advect z-face quantity ``q`` through the (u, v, w) velocity field."""
    i, j, k = wp.tid()
    if not _is_pure_fluid_face(cell_label, 2, i, j, k):
        q_out[i, j, k] = q[i, j, k]
        return
    pos = face_position(origin, dx, 2, i, j, k)
    back = _backtrace(u, v, w, origin, dx, pos, dt)
    q_out[i, j, k] = sample_w(q, origin, dx, back)


# MacCormack correction (Selle et al. 2008): after a forward semi-Lagrangian
# pass (q_hat) and a backward pass of q_hat (q_tilde), the corrected value is
# q_hat + (q - q_tilde) / 2, clamped to the range of the original field around
# the backtraced point so the scheme stays unconditionally stable.


@wp.func
def _corner_bounds(field: wp.array3d[float], lx: float, ly: float, lz: float):
    """Min/max of the 8 interpolation corners at component-space coordinates."""
    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]
    lx = wp.clamp(lx, 0.0, float(nx - 1))
    ly = wp.clamp(ly, 0.0, float(ny - 1))
    lz = wp.clamp(lz, 0.0, float(nz - 1))
    i0 = wp.clamp(int(lx), 0, wp.max(nx - 2, 0))
    j0 = wp.clamp(int(ly), 0, wp.max(ny - 2, 0))
    k0 = wp.clamp(int(lz), 0, wp.max(nz - 2, 0))
    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)

    lo = field[i0, j0, k0]
    hi = lo
    for n in range(1, 8):
        ii = i1
        if n & 1 == 0:
            ii = i0
        jj = j1
        if (n >> 1) & 1 == 0:
            jj = j0
        kk = k1
        if (n >> 2) & 1 == 0:
            kk = k0
        val = field[ii, jj, kk]
        lo = wp.min(lo, val)
        hi = wp.max(hi, val)
    return lo, hi


@wp.func
def _maccormack_correct(
    axis: int,
    i: int,
    j: int,
    k: int,
    origin: wp.vec3,
    dx: float,
    dt: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    q: wp.array3d[float],
    q_hat: wp.array3d[float],
    q_tilde: wp.array3d[float],
    q_out: wp.array3d[float],
):
    if not _is_pure_fluid_face(cell_label, axis, i, j, k):
        q_out[i, j, k] = q[i, j, k]
        return

    corrected = q_hat[i, j, k] + 0.5 * (q[i, j, k] - q_tilde[i, j, k])

    # clamp to the original-field bounds around the backtraced point
    pos = face_position(origin, dx, axis, i, j, k)
    back = _backtrace(u, v, w, origin, dx, pos, dt)
    g = (back - origin) / dx
    if axis == 0:
        lo, hi = _corner_bounds(q, g[0], g[1] - 0.5, g[2] - 0.5)
    elif axis == 1:
        lo, hi = _corner_bounds(q, g[0] - 0.5, g[1], g[2] - 0.5)
    else:
        lo, hi = _corner_bounds(q, g[0] - 0.5, g[1] - 0.5, g[2])

    q_out[i, j, k] = wp.clamp(corrected, lo, hi)


@wp.kernel(enable_backward=False)
def maccormack_correct_u_kernel(
    origin: wp.vec3,
    dx: float,
    dt: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    q_hat: wp.array3d[float],
    q_tilde: wp.array3d[float],
    q_out: wp.array3d[float],
):
    i, j, k = wp.tid()
    _maccormack_correct(0, i, j, k, origin, dx, dt, cell_label, u, v, w, u, q_hat, q_tilde, q_out)


@wp.kernel(enable_backward=False)
def maccormack_correct_v_kernel(
    origin: wp.vec3,
    dx: float,
    dt: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    q_hat: wp.array3d[float],
    q_tilde: wp.array3d[float],
    q_out: wp.array3d[float],
):
    i, j, k = wp.tid()
    _maccormack_correct(1, i, j, k, origin, dx, dt, cell_label, u, v, w, v, q_hat, q_tilde, q_out)


@wp.kernel(enable_backward=False)
def maccormack_correct_w_kernel(
    origin: wp.vec3,
    dx: float,
    dt: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    q_hat: wp.array3d[float],
    q_tilde: wp.array3d[float],
    q_out: wp.array3d[float],
):
    i, j, k = wp.tid()
    _maccormack_correct(2, i, j, k, origin, dx, dt, cell_label, u, v, w, w, q_hat, q_tilde, q_out)


# ---------------------------------------------------------------------------
# external forces (gravity + uniform external acceleration)
# ---------------------------------------------------------------------------


@wp.func
def _add_force(
    axis: int,
    i: int,
    j: int,
    k: int,
    accel: wp.vec3,
    dt: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    vel: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
):
    if not _is_pure_fluid_face(cell_label, axis, i, j, k):
        return
    a = accel[axis]
    if a == 0.0:
        return
    vel[i, j, k] = vel[i, j, k] + a * dt
    imp = wp.vec3(0.0)
    imp[axis] = mass_face * a * dt
    wp.atomic_add(diag_vec, DIAG_V_IMPULSE_EXTERNAL, imp)


@wp.kernel(enable_backward=False)
def add_forces_u_kernel(
    gravity: wp.array[wp.vec3],
    external_accel: wp.array[wp.vec3],
    dt: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
):
    i, j, k = wp.tid()
    accel = gravity[0] + external_accel[0]
    _add_force(0, i, j, k, accel, dt, mass_face, cell_label, u, diag_vec)


@wp.kernel(enable_backward=False)
def add_forces_v_kernel(
    gravity: wp.array[wp.vec3],
    external_accel: wp.array[wp.vec3],
    dt: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    v: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
):
    i, j, k = wp.tid()
    accel = gravity[0] + external_accel[0]
    _add_force(1, i, j, k, accel, dt, mass_face, cell_label, v, diag_vec)


@wp.kernel(enable_backward=False)
def add_forces_w_kernel(
    gravity: wp.array[wp.vec3],
    external_accel: wp.array[wp.vec3],
    dt: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    w: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
):
    i, j, k = wp.tid()
    accel = gravity[0] + external_accel[0]
    _add_force(2, i, j, k, accel, dt, mass_face, cell_label, w, diag_vec)


# ---------------------------------------------------------------------------
# viscosity (explicit diffusion on the staggered components)
# ---------------------------------------------------------------------------


@wp.func
def _viscous_neighbor(
    axis: int,
    ni: int,
    nj: int,
    nk: int,
    vel: wp.array3d[float],
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
):
    """Value, constrained flag, and owner label of one same-component neighbor face.

    Out-of-array neighbor faces are static no-slip walls with zero velocity.
    """
    nx = vel.shape[0]
    ny = vel.shape[1]
    nz = vel.shape[2]
    if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz:
        return 0.0, True, int(-1)

    ia, ja, ka = _face_neighbor_cells(axis, ni, nj, nk)
    owner = face_solid_owner(cell_label, cell_sdf, ia, ja, ka, ni, nj, nk)
    return vel[ni, nj, nk], owner != CELL_FLUID, owner


@wp.func
def _diffuse_face(
    axis: int,
    i: int,
    j: int,
    k: int,
    origin: wp.vec3,
    dx: float,
    coeff: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    vel: wp.array3d[float],
    vel_out: wp.array3d[float],
    body_impulse: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    diag_vec: wp.array[wp.vec3],
):
    center = vel[i, j, k]
    if not _is_pure_fluid_face(cell_label, axis, i, j, k):
        vel_out[i, j, k] = center
        return

    lap = float(0.0)
    boundary_exchange = float(0.0)

    for n in range(6):
        d = n // 2  # neighbor direction axis
        ni = i
        nj = j
        nk = k
        step = 2 * (n % 2) - 1  # -1, +1
        if d == 0:
            ni += step
        elif d == 1:
            nj += step
        else:
            nk += step

        nb_val, nb_constrained, nb_owner = _viscous_neighbor(axis, ni, nj, nk, vel, cell_label, cell_sdf)
        delta = nb_val - center
        lap += delta

        if nb_constrained:
            # momentum exchanged with the solid through this neighbor face
            exchange = coeff * delta  # velocity change contributed to this face
            boundary_exchange += exchange
            if nb_owner >= 0:
                imp = wp.vec3(0.0)
                imp[axis] = -mass_face * exchange  # reaction impulse on the body
                pos = face_position(origin, dx, axis, ni, nj, nk)
                com_world = wp.transform_point(body_q[nb_owner], body_com[nb_owner])
                r = pos - com_world
                wp.atomic_add(body_impulse, nb_owner, wp.spatial_vector(imp, wp.cross(r, imp)))

    vel_out[i, j, k] = center + coeff * lap

    if boundary_exchange != 0.0:
        imp = wp.vec3(0.0)
        imp[axis] = mass_face * boundary_exchange
        wp.atomic_add(diag_vec, DIAG_V_IMPULSE_VISCOUS, imp)


@wp.kernel(enable_backward=False)
def diffuse_u_kernel(
    origin: wp.vec3,
    dx: float,
    coeff: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    u: wp.array3d[float],
    u_out: wp.array3d[float],
    body_impulse: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    diag_vec: wp.array[wp.vec3],
):
    i, j, k = wp.tid()
    _diffuse_face(
        0,
        i,
        j,
        k,
        origin,
        dx,
        coeff,
        mass_face,
        cell_label,
        cell_sdf,
        u,
        u_out,
        body_impulse,
        body_q,
        body_com,
        diag_vec,
    )


@wp.kernel(enable_backward=False)
def diffuse_v_kernel(
    origin: wp.vec3,
    dx: float,
    coeff: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    v: wp.array3d[float],
    v_out: wp.array3d[float],
    body_impulse: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    diag_vec: wp.array[wp.vec3],
):
    i, j, k = wp.tid()
    _diffuse_face(
        1,
        i,
        j,
        k,
        origin,
        dx,
        coeff,
        mass_face,
        cell_label,
        cell_sdf,
        v,
        v_out,
        body_impulse,
        body_q,
        body_com,
        diag_vec,
    )


@wp.kernel(enable_backward=False)
def diffuse_w_kernel(
    origin: wp.vec3,
    dx: float,
    coeff: float,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    w: wp.array3d[float],
    w_out: wp.array3d[float],
    body_impulse: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    diag_vec: wp.array[wp.vec3],
):
    i, j, k = wp.tid()
    _diffuse_face(
        2,
        i,
        j,
        k,
        origin,
        dx,
        coeff,
        mass_face,
        cell_label,
        cell_sdf,
        w,
        w_out,
        body_impulse,
        body_q,
        body_com,
        diag_vec,
    )


# ---------------------------------------------------------------------------
# divergence and pressure right-hand side
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def divergence_kernel(
    dx: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    divergence: wp.array3d[float],
    diag_scalar: wp.array[float],
    diag_pre: int,
):
    i, j, k = wp.tid()
    if cell_label[i, j, k] != CELL_FLUID:
        divergence[i, j, k] = 0.0
        return

    div = (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]) / dx
    divergence[i, j, k] = div
    if diag_pre != 0:
        wp.atomic_add(diag_scalar, DIAG_S_DIV_L2_PRE, div * div)
        wp.atomic_max(diag_scalar, DIAG_S_DIV_LINF_PRE, wp.abs(div))
    else:
        wp.atomic_add(diag_scalar, DIAG_S_DIV_L2_POST, div * div)
        wp.atomic_max(diag_scalar, DIAG_S_DIV_LINF_POST, wp.abs(div))


@wp.kernel(enable_backward=False)
def pressure_rhs_kernel(
    cell_label: wp.array3d[wp.int32],
    divergence: wp.array3d[float],
    div_sum: wp.array[float],
    fluid_cell_count: wp.array[wp.int32],
    b: wp.array[float],
):
    i, j, k = wp.tid()
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    idx = (i * ny + j) * nz + k
    if cell_label[i, j, k] != CELL_FLUID:
        b[idx] = 0.0
        return
    # Null-space / compatibility handling for the all-Neumann closed domain:
    # distribute the net divergence uniformly over the fluid cells so the
    # right-hand side is orthogonal to the constant kernel vector.
    count = wp.max(fluid_cell_count[0], 1)
    mean_div = div_sum[0] / float(count)
    b[idx] = -(divergence[i, j, k] - mean_div)


# ---------------------------------------------------------------------------
# pressure gradient application and pressure wrench on solids
# ---------------------------------------------------------------------------


@wp.func
def _apply_gradient(
    axis: int,
    i: int,
    j: int,
    k: int,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    q: wp.array[float],
    vel: wp.array3d[float],
):
    if not _is_pure_fluid_face(cell_label, axis, i, j, k):
        return
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    ia, ja, ka = _face_neighbor_cells(axis, i, j, k)
    idx_lo = (ia * ny + ja) * nz + ka
    idx_hi = (i * ny + j) * nz + k
    vel[i, j, k] = vel[i, j, k] - (q[idx_hi] - q[idx_lo]) / dx


@wp.kernel(enable_backward=False)
def apply_gradient_u_kernel(
    dx: float,
    cell_label: wp.array3d[wp.int32],
    q: wp.array[float],
    u: wp.array3d[float],
):
    i, j, k = wp.tid()
    _apply_gradient(0, i, j, k, dx, cell_label, q, u)


@wp.kernel(enable_backward=False)
def apply_gradient_v_kernel(
    dx: float,
    cell_label: wp.array3d[wp.int32],
    q: wp.array[float],
    v: wp.array3d[float],
):
    i, j, k = wp.tid()
    _apply_gradient(1, i, j, k, dx, cell_label, q, v)


@wp.kernel(enable_backward=False)
def apply_gradient_w_kernel(
    dx: float,
    cell_label: wp.array3d[wp.int32],
    q: wp.array[float],
    w: wp.array3d[float],
):
    i, j, k = wp.tid()
    _apply_gradient(2, i, j, k, dx, cell_label, q, w)


@wp.kernel(enable_backward=False)
def pressure_wrench_kernel(
    origin: wp.vec3,
    dx: float,
    rho: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    q: wp.array[float],
    pressure: wp.array3d[float],
    inv_dt: float,
    body_impulse: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    diag_vec: wp.array[wp.vec3],
):
    """Accumulate the pressure surface impulse on solids and store the pressure field.

    For every interface face between a fluid cell and a solid cell, the fluid
    pushes the solid with ``p * A * n`` (n pointing from fluid into solid); the
    projection applies the opposite momentum change to the fluid interior.
    ``q`` is the dt-scaled pressure ``q = p * dt / rho``, so the impulse over
    the step is ``rho * q * A * n``.
    """
    i, j, k = wp.tid()
    if cell_label[i, j, k] != CELL_FLUID:
        pressure[i, j, k] = 0.0
        return

    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    idx = (i * ny + j) * nz + k
    q_f = q[idx]
    pressure[i, j, k] = q_f * rho * inv_dt  # actual pressure p = q * rho / dt

    area = dx * dx
    imp_total = wp.vec3(0.0)

    for n in range(6):
        d = n // 2
        step = 2 * (n % 2) - 1
        ni = i
        nj = j
        nk = k
        if d == 0:
            ni += step
        elif d == 1:
            nj += step
        else:
            nk += step

        if not is_solid_cell(cell_label, ni, nj, nk):
            continue

        # outward normal from the fluid cell into the solid neighbor
        normal = wp.vec3(0.0)
        normal[d] = float(step)
        imp = rho * q_f * area * normal  # impulse applied to the solid over dt
        imp_total += imp

        owner = int(-1)
        nx_c = cell_label.shape[0]
        ny_c = cell_label.shape[1]
        nz_c = cell_label.shape[2]
        if ni >= 0 and nj >= 0 and nk >= 0 and ni < nx_c and nj < ny_c and nk < nz_c:
            owner = cell_label[ni, nj, nk]

        if owner >= 0:
            # face position between the two cells
            fi = i
            fj = j
            fk = k
            if step > 0:
                if d == 0:
                    fi = i + 1
                elif d == 1:
                    fj = j + 1
                else:
                    fk = k + 1
            pos = face_position(origin, dx, d, fi, fj, fk)
            com_world = wp.transform_point(body_q[owner], body_com[owner])
            r = pos - com_world
            wp.atomic_add(body_impulse, owner, wp.spatial_vector(imp, wp.cross(r, imp)))

    if wp.length_sq(imp_total) > 0.0:
        # reaction on the fluid
        wp.atomic_add(diag_vec, DIAG_V_IMPULSE_PRESSURE, -imp_total)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


@wp.func
def _momentum_face(
    axis: int,
    i: int,
    j: int,
    k: int,
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    vel: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
    slot: int,
):
    if not _is_pure_fluid_face(cell_label, axis, i, j, k):
        return
    p = wp.vec3(0.0)
    p[axis] = mass_face * vel[i, j, k]
    wp.atomic_add(diag_vec, slot, p)


@wp.kernel(enable_backward=False)
def momentum_u_kernel(
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    u: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
    slot: int,
):
    i, j, k = wp.tid()
    _momentum_face(0, i, j, k, mass_face, cell_label, u, diag_vec, slot)


@wp.kernel(enable_backward=False)
def momentum_v_kernel(
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    v: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
    slot: int,
):
    i, j, k = wp.tid()
    _momentum_face(1, i, j, k, mass_face, cell_label, v, diag_vec, slot)


@wp.kernel(enable_backward=False)
def momentum_w_kernel(
    mass_face: float,
    cell_label: wp.array3d[wp.int32],
    w: wp.array3d[float],
    diag_vec: wp.array[wp.vec3],
    slot: int,
):
    i, j, k = wp.tid()
    _momentum_face(2, i, j, k, mass_face, cell_label, w, diag_vec, slot)


@wp.func
def _noslip_error_face(
    axis: int,
    i: int,
    j: int,
    k: int,
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    vel: wp.array3d[float],
    diag_scalar: wp.array[float],
):
    """Tangential-slip error for pure fluid faces bordering a rigid body."""
    if not _is_pure_fluid_face(cell_label, axis, i, j, k):
        return

    # look for a rigid-body cell among the tangential neighbor cells
    ia, ja, ka = _face_neighbor_cells(axis, i, j, k)
    owner = int(CELL_FLUID)
    nx = cell_label.shape[0]
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    for n in range(12):
        d = n // 4  # 0..2, skip the face axis below
        if d == axis:
            continue
        m = n % 4
        # neighbors of both adjacent cells in tangential direction d
        ci = ia
        cj = ja
        ck = ka
        if m >= 2:
            ci = i
            cj = j
            ck = k
        step = 2 * (m % 2) - 1
        if d == 0:
            ci += step
        elif d == 1:
            cj += step
        else:
            ck += step
        if ci < 0 or cj < 0 or ck < 0 or ci >= nx or cj >= ny or ck >= nz:
            continue
        label = cell_label[ci, cj, ck]
        if label >= 0:
            owner = label

    if owner < 0:
        return

    pos = face_position(origin, dx, axis, i, j, k)
    solid_vel = body_point_velocity(owner, pos, body_q, body_qd, body_com)
    err = wp.abs(vel[i, j, k] - solid_vel[axis])
    wp.atomic_max(diag_scalar, DIAG_S_NOSLIP_MAX, err)
    wp.atomic_add(diag_scalar, DIAG_S_NOSLIP_SUM, err)
    wp.atomic_add(diag_scalar, DIAG_S_NOSLIP_COUNT, 1.0)


@wp.kernel(enable_backward=False)
def noslip_error_u_kernel(
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    u: wp.array3d[float],
    diag_scalar: wp.array[float],
):
    i, j, k = wp.tid()
    _noslip_error_face(0, i, j, k, origin, dx, cell_label, body_q, body_qd, body_com, u, diag_scalar)


@wp.kernel(enable_backward=False)
def noslip_error_v_kernel(
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    v: wp.array3d[float],
    diag_scalar: wp.array[float],
):
    i, j, k = wp.tid()
    _noslip_error_face(1, i, j, k, origin, dx, cell_label, body_q, body_qd, body_com, v, diag_scalar)


@wp.kernel(enable_backward=False)
def noslip_error_w_kernel(
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    w: wp.array3d[float],
    diag_scalar: wp.array[float],
):
    i, j, k = wp.tid()
    _noslip_error_face(2, i, j, k, origin, dx, cell_label, body_q, body_qd, body_com, w, diag_scalar)
