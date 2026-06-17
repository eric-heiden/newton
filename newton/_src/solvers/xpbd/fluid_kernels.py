# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-based fluid (PBF) kernels for :class:`~newton.solvers.SolverXPBD`.

Implements the SPH smoothing kernels and the density-constraint, cohesion,
viscosity, and vorticity passes of Macklin & Müller, "Position Based Fluids"
(2013). Kept separate from the general XPBD constraint kernels in
:mod:`~newton._src.solvers.xpbd.kernels` so the fluid solve can be read and
maintained on its own.
"""

import warp as wp

from ...geometry import ParticleFlags

PI = wp.constant(3.141592653589793)


@wp.func
def poly6_kernel(r_sq: float, h: float) -> float:
    """Poly6 smoothing kernel used for fluid density estimation."""
    h_sq = h * h
    result = float(0.0)
    if r_sq < h_sq:
        x = h_sq - r_sq
        h9 = h_sq * h_sq * h_sq * h_sq * h
        result = 315.0 / (64.0 * PI * h9) * x * x * x
    return result


@wp.func
def spiky_kernel_gradient(r_vec: wp.vec3, r: float, h: float) -> wp.vec3:
    """Gradient of the spiky kernel used for fluid density constraint gradients."""
    result = wp.vec3(0.0)
    if r > 1.0e-6 and r < h:
        h6 = h * h * h * h * h * h
        x = h - r
        result = (-45.0 / (PI * h6) * x * x / r) * r_vec
    return result


@wp.func
def cohesion_kernel(r: float, h: float) -> float:
    """Normalized Akinci-style cohesion spline.

    Returns +1 at the maximum-attraction distance ``r = h/2``, falls to zero at
    ``r = h``, and turns negative (repulsive) below ``r ~ 0.27 h`` so isolated
    particle clusters reach a stable spacing instead of collapsing to a point.
    See Akinci et al., "Versatile Surface Tension and Adhesion for SPH Fluids" (2013).
    """
    q = r / h
    result = float(0.0)
    if q < 1.0:
        a = 1.0 - q
        s = a * a * a * q * q * q
        if q <= 0.5:
            result = 2.0 * s - 1.0 / 64.0
        else:
            result = s
        result *= 64.0
    return result


@wp.func
def _pseudo_random_offset(idx: int) -> wp.vec3:
    # fixed per-index pseudo-random vector in [-0.5, 0.5]^3
    state = wp.rand_init(idx + 1)
    return wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)


@wp.func
def coincidence_separation_dir(i: int, j: int) -> wp.vec3:
    """Deterministic antisymmetric unit vector to separate near-coincident particles.

    When two fluid particles are pushed to almost the same position the spiky
    kernel's gradient direction (``r_vec / |r_vec|``) is numerically meaningless,
    so the density constraint can no longer drive them apart and they fuse into a
    stuck "super particle". A fixed per-index pseudo-random offset gives a stable,
    antisymmetric (``dir(i, j) == -dir(j, i)``) direction so the pair drifts apart
    consistently across iterations instead of collapsing to a point.
    """
    d = _pseudo_random_offset(i) - _pseudo_random_offset(j)
    n = wp.length(d)
    if n < 1.0e-6:
        return wp.vec3(0.0, 0.0, 1.0)
    return d / n


@wp.kernel
def compute_fluid_lambdas(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_invmass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    smoothing_length: float,
    rest_density: float,
    relaxation_epsilon: float,
    max_neighbors: int,
    rest_distance: float,
    # outputs
    fluid_density: wp.array[float],
    fluid_lambda: wp.array[float],
):
    """Compute SPH densities and density-constraint Lagrange multipliers.

    Implements Eqs. 8-11 of Macklin & Müller, "Position Based Fluids" (2013),
    generalized with per-particle masses: the constraint for fluid particle i is
    ``C_i = rho_i / rho_0 - 1`` and the denominator accumulates the
    inverse-mass-weighted squared constraint gradients of all participating
    particles. The constraint acts on compression only; under-dense particles
    (free surfaces, sparse splashes) are handled by the bounded cohesion term in
    :func:`solve_fluid_deltas` because the raw attractive branch diverges for
    near-isolated particles whose density deficit saturates at ``-1`` while
    their gradient denominator vanishes.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    flags = particle_flags[i]
    if (flags & ParticleFlags.ACTIVE) == 0 or (flags & ParticleFlags.FLUID) == 0:
        fluid_density[i] = 0.0
        fluid_lambda[i] = 0.0
        return

    x = particle_x[i]
    h = smoothing_length
    inv_rest_density = 1.0 / rest_density

    density = particle_mass[i] * poly6_kernel(0.0, h)
    grad_i = wp.vec3(0.0)
    grad_sum = float(0.0)

    # Cap the neighbors processed per particle. A momentary over-compressed
    # clump (e.g. fluid slammed into a corner) can hold an order of magnitude
    # more neighbors than the bulk; one such particle stalls its whole 32-lane
    # warp. Capping above the bulk count leaves normal fluid untouched while
    # bounding that tail. ``max_neighbors <= 0`` disables the cap.
    n_acc = int(0)
    query = wp.hash_grid_query(grid, x, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        flags_j = particle_flags[j]
        if (flags_j & ParticleFlags.ACTIVE) == 0 or (flags_j & ParticleFlags.FLUID) == 0:
            continue
        r_vec = x - particle_x[j]
        r_sq = wp.dot(r_vec, r_vec)
        if r_sq >= h * h:
            continue
        density += particle_mass[j] * poly6_kernel(r_sq, h)
        # density above keeps the true (max) weight; the gradient below needs a
        # well-defined direction, so substitute one for near-coincident pairs
        r = wp.sqrt(r_sq)
        if r < 0.05 * rest_distance:
            r_vec = 0.05 * rest_distance * coincidence_separation_dir(i, j)
            r = 0.05 * rest_distance
        grad_j = -(particle_mass[j] * inv_rest_density) * spiky_kernel_gradient(r_vec, r, h)
        grad_sum += particle_invmass[j] * wp.dot(grad_j, grad_j)
        grad_i -= grad_j
        n_acc += 1
        if max_neighbors > 0 and n_acc >= max_neighbors:
            break

    grad_sum += particle_invmass[i] * wp.dot(grad_i, grad_i)
    fluid_density[i] = density

    c = wp.max(density * inv_rest_density - 1.0, 0.0)
    fluid_lambda[i] = -c / (grad_sum + relaxation_epsilon)


@wp.kernel
def solve_fluid_deltas(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_invmass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    fluid_lambda: wp.array[float],
    smoothing_length: float,
    rest_density: float,
    cohesion_step: float,
    max_delta: float,
    relaxation: float,
    max_neighbors: int,
    rest_distance: float,
    # outputs
    deltas: wp.array[wp.vec3],
):
    """Accumulate density-constraint position corrections for fluid particles.

    In addition to the incompressibility correction, each neighbor pair receives
    a bounded cohesion bias of at most ``cohesion_step`` meters per iteration
    along the pair direction, following the sign of :func:`cohesion_kernel`:
    attraction at mid-range, short-range repulsion. This produces the
    surface-tension-like coagulation of splashes without the divergence of
    constraint-based attraction for near-isolated particles.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    flags = particle_flags[i]
    if (flags & ParticleFlags.ACTIVE) == 0 or (flags & ParticleFlags.FLUID) == 0:
        return
    w_i = particle_invmass[i]
    if w_i == 0.0:
        return

    x = particle_x[i]
    h = smoothing_length
    inv_rest_density = 1.0 / rest_density
    lambda_i = fluid_lambda[i]

    min_sep = 0.05 * rest_distance
    min_dist = 0.5 * rest_distance

    # Density correction (summed, standard PBF) and the cohesion bias and the
    # minimum-distance push are accumulated separately because they need
    # different normalization: the density term is summed for incompressibility,
    # while cohesion is averaged per neighbor so the surface-tension pull stays
    # bounded, and the contact-like separation is applied at full strength.
    delta = wp.vec3(0.0)
    cohesion = wp.vec3(0.0)
    separation = wp.vec3(0.0)
    num_neighbors = int(0)

    query = wp.hash_grid_query(grid, x, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        flags_j = particle_flags[j]
        if (flags_j & ParticleFlags.ACTIVE) == 0 or (flags_j & ParticleFlags.FLUID) == 0:
            continue
        r_vec = x - particle_x[j]
        r_sq = wp.dot(r_vec, r_vec)
        if r_sq >= h * h:
            continue
        r = wp.sqrt(r_sq)
        num_neighbors += 1

        # near-coincident particles have no meaningful pair direction; substitute
        # a deterministic one (see compute_fluid_lambdas)
        if r < min_sep:
            r_vec = min_sep * coincidence_separation_dir(i, j)
            r = min_sep

        grad = spiky_kernel_gradient(r_vec, r, h)
        delta += (lambda_i + fluid_lambda[j]) * (particle_mass[j] * inv_rest_density) * grad * w_i

        if cohesion_step > 0.0:
            # bounded position bias toward (or away from) the neighbor
            cohesion += (-cohesion_step * cohesion_kernel(r, h) / r) * r_vec

        # short-range repulsion: push the pair apart to the minimum distance,
        # split evenly (equal fluid masses). Only fires when over-compressed, so
        # the rest lattice (nearest neighbor at ~rest_distance) is untouched.
        if r < min_dist:
            separation += (0.5 * (min_dist - r) / r) * r_vec

        # Bound the worst-case loop in over-compressed clumps (see
        # compute_fluid_lambdas); must use the same cap so the averaging below
        # matches the density estimate.
        if max_neighbors > 0 and num_neighbors >= max_neighbors:
            break

    if num_neighbors == 0:
        return

    # Standard PBF position correction (Macklin & Müller 2013, Eq. 12): the
    # per-particle lambda already carries the gradient-sum normalization, so the
    # correction is the raw sum over neighbors -- NOT additionally divided by the
    # neighbor count. Dividing again (an over-conservative Jacobi averaging)
    # weakened incompressibility by ~the neighbor count, so a tall column of fine
    # particles collapsed into a dense slug instead of holding its volume. The
    # max-delta clamp below bounds the per-iteration step for stability instead.
    delta_len = wp.length(delta)
    if delta_len > max_delta:
        delta *= max_delta / delta_len

    # average the cohesion bias over the contributing neighbors so the
    # surface-tension pull stays bounded regardless of neighborhood size (the
    # density term above is intentionally left summed)
    cohesion = cohesion / float(num_neighbors)

    # bound the un-averaged separation push too, then apply all three
    sep_len = wp.length(separation)
    if sep_len > max_delta:
        separation *= max_delta / sep_len

    wp.atomic_add(deltas, i, delta * relaxation + cohesion + separation)


@wp.kernel
def compute_fluid_vorticity(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    fluid_density: wp.array[float],
    smoothing_length: float,
    # outputs
    fluid_vorticity: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    flags = particle_flags[i]
    if (flags & ParticleFlags.ACTIVE) == 0 or (flags & ParticleFlags.FLUID) == 0:
        fluid_vorticity[i] = wp.vec3(0.0)
        return

    x = particle_x[i]
    v = particle_v[i]
    h = smoothing_length
    omega = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        flags_j = particle_flags[j]
        if (flags_j & ParticleFlags.ACTIVE) == 0 or (flags_j & ParticleFlags.FLUID) == 0:
            continue
        r_vec = x - particle_x[j]
        r_sq = wp.dot(r_vec, r_vec)
        if r_sq >= h * h:
            continue
        rho_j = wp.max(fluid_density[j], 1.0e-6)
        grad = spiky_kernel_gradient(r_vec, wp.sqrt(r_sq), h)
        omega += particle_mass[j] / rho_j * wp.cross(particle_v[j] - v, grad)

    fluid_vorticity[i] = omega


@wp.kernel
def solve_fluid_velocities(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_invmass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    fluid_density: wp.array[float],
    fluid_vorticity: wp.array[wp.vec3],
    smoothing_length: float,
    viscosity: float,
    vorticity_confinement: float,
    dt: float,
    # outputs
    v_out: wp.array[wp.vec3],
):
    """Post-projection velocity pass: XSPH viscosity and vorticity confinement.

    Non-fluid particles pass their velocity through unchanged so the output
    buffer can replace the particle velocity array wholesale.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    v = particle_v[i]
    flags = particle_flags[i]
    if (flags & ParticleFlags.ACTIVE) == 0 or (flags & ParticleFlags.FLUID) == 0 or particle_invmass[i] == 0.0:
        v_out[i] = v
        return

    x = particle_x[i]
    h = smoothing_length
    omega_i = fluid_vorticity[i]

    rho_i = wp.max(fluid_density[i], 1.0e-6)
    weight_sum = particle_mass[i] / rho_i * poly6_kernel(0.0, h)
    v_weighted = v * weight_sum
    eta = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        flags_j = particle_flags[j]
        if (flags_j & ParticleFlags.ACTIVE) == 0 or (flags_j & ParticleFlags.FLUID) == 0:
            continue
        r_vec = x - particle_x[j]
        r_sq = wp.dot(r_vec, r_vec)
        if r_sq >= h * h:
            continue
        rho_j = wp.max(fluid_density[j], 1.0e-6)
        w = particle_mass[j] / rho_j * poly6_kernel(r_sq, h)
        v_weighted += particle_v[j] * w
        weight_sum += w
        if vorticity_confinement > 0.0:
            grad = spiky_kernel_gradient(r_vec, wp.sqrt(r_sq), h)
            eta += (wp.length(fluid_vorticity[j]) - wp.length(omega_i)) * grad

    v_new = v
    if viscosity > 0.0 and weight_sum > 1.0e-6:
        v_new = v + viscosity * (v_weighted / weight_sum - v)

    if vorticity_confinement > 0.0:
        eta_len = wp.length(eta)
        if eta_len > 1.0e-6:
            v_new += vorticity_confinement * dt * wp.cross(eta / eta_len, omega_i)

    v_out[i] = v_new
