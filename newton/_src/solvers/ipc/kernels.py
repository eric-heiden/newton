# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the experimental cloth IPC solver."""

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def barrier_value(s: float):
    """Evaluate the normalized IPC barrier for a squared-distance ratio."""
    value = float(0.0)
    if s < 1.0:
        t = s - 1.0
        value = -(t * t) * wp.log(s)
    return value


@wp.func
def barrier_first_derivative(s: float):
    """Evaluate the first derivative of :func:`barrier_value`."""
    value = float(0.0)
    if s < 1.0:
        t = s - 1.0
        value = -(2.0 * t * wp.log(s) + t * t / s)
    return value


@wp.func
def barrier_second_derivative(s: float):
    """Evaluate the second derivative of :func:`barrier_value`."""
    value = float(0.0)
    if s < 1.0:
        value = -2.0 * wp.log(s) - 3.0 + 2.0 / s + 1.0 / (s * s)
    return value


@wp.kernel
def initialize_step(
    dt: float,
    normal: wp.vec3,
    plane_offset: float,
    minimum_separation: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[int],
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    pd_diagonal: wp.array[float],
    x_previous: wp.array[wp.vec3],
    x_predictor: wp.array[wp.vec3],
    x_current: wp.array[wp.vec3],
    static_diagonal: wp.array[float],
    invalid: wp.array[int],
    minimum_gap: wp.array[float],
):
    particle = wp.tid()
    x = particle_q[particle]
    x_previous[particle] = x
    x_current[particle] = x

    mass = particle_mass[particle]
    active = (particle_flags[particle] & ParticleFlags.ACTIVE) != 0 and mass > 0.0
    if active:
        world = particle_world[particle]
        x_predictor[particle] = (
            x + particle_qd[particle] * dt + (gravity[world] + particle_f[particle] / mass) * (dt * dt)
        )
        static_diagonal[particle] = mass / (dt * dt) + pd_diagonal[particle]

        gap = wp.dot(normal, x) - plane_offset - minimum_separation
        wp.atomic_min(minimum_gap, 0, gap)
        if gap <= 0.0 or wp.isnan(gap):
            wp.atomic_max(invalid, 0, 1)
    else:
        x_predictor[particle] = x
        static_diagonal[particle] = 0.0


@wp.kernel
def finish_initialization(
    invalid: wp.array[int],
    active: wp.array[int],
    status: wp.array[int],
    running_status: int,
    invalid_status: int,
):
    if invalid[0] != 0:
        active[0] = 0
        status[0] = invalid_status
    else:
        active[0] = 1
        status[0] = running_status


@wp.kernel
def initialize_rhs(
    dt: float,
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    x_current: wp.array[wp.vec3],
    x_predictor: wp.array[wp.vec3],
    solve_active: wp.array[int],
    rhs: wp.array[wp.vec3],
):
    particle = wp.tid()
    mass = particle_mass[particle]
    if solve_active[0] != 0 and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0 and mass > 0.0:
        rhs[particle] = (x_predictor[particle] - x_current[particle]) * mass / (dt * dt)
    else:
        rhs[particle] = wp.vec3(0.0)


@wp.func
def triangle_deformation_gradient(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, inv_dm: wp.mat22):
    x01 = x1 - x0
    x02 = x2 - x0
    fu = x01 * inv_dm[0, 0] + x02 * inv_dm[1, 0]
    fv = x01 * inv_dm[0, 1] + x02 * inv_dm[1, 1]
    return fu, fv


@wp.kernel
def add_triangle_forces(
    x: wp.array[wp.vec3],
    triangle_area: wp.array[float],
    triangle_pose: wp.array[wp.mat22],
    triangle_indices: wp.array2d[int],
    triangle_stiffness: wp.array[wp.vec3],
    solve_active: wp.array[int],
    rhs: wp.array[wp.vec3],
):
    triangle = wp.tid()
    if solve_active[0] == 0:
        return

    indices = wp.vec3i(
        triangle_indices[triangle, 0],
        triangle_indices[triangle, 1],
        triangle_indices[triangle, 2],
    )
    inv_dm = triangle_pose[triangle]
    fu, fv = triangle_deformation_gradient(
        x[indices[0]],
        x[indices[1]],
        x[indices[2]],
        inv_dm,
    )
    length_u = wp.length(fu)
    length_v = wp.length(fv)
    direction_u = wp.vec3(0.0)
    direction_v = wp.vec3(0.0)
    if length_u > 1.0e-8:
        direction_u = fu / length_u
    if length_v > 1.0e-8:
        direction_v = fv / length_v

    derivative_u = wp.vec3(
        -inv_dm[0, 0] - inv_dm[1, 0],
        inv_dm[0, 0],
        inv_dm[1, 0],
    )
    derivative_v = wp.vec3(
        -inv_dm[0, 1] - inv_dm[1, 1],
        inv_dm[0, 1],
        inv_dm[1, 1],
    )
    stiffness = triangle_stiffness[triangle]
    shear = wp.dot(fu, fv)
    area = triangle_area[triangle]
    for local in range(3):
        gradient = area * (
            stiffness[0] * (length_u - 1.0) * derivative_u[local] * direction_u
            + stiffness[1] * (length_v - 1.0) * derivative_v[local] * direction_v
            + stiffness[2] * shear * (derivative_u[local] * fv + derivative_v[local] * fu)
        )
        wp.atomic_sub(rhs, indices[local], gradient)


@wp.func
def bending_weight(cotangent: wp.vec4):
    return wp.vec4(
        -cotangent[0] - cotangent[2],
        -cotangent[1] - cotangent[3],
        cotangent[2] + cotangent[3],
        cotangent[0] + cotangent[1],
    )


@wp.kernel
def add_bending_forces(
    x: wp.array[wp.vec3],
    edge_rest_area: wp.array[float],
    edge_bending_cotangent: wp.array[wp.vec4],
    edge_indices: wp.array2d[int],
    edge_bending_properties: wp.array2d[float],
    solve_active: wp.array[int],
    rhs: wp.array[wp.vec3],
):
    edge = wp.tid()
    if solve_active[0] == 0 or edge_indices[edge, 0] < 0 or edge_indices[edge, 1] < 0:
        return

    indices = wp.vec4i(
        edge_indices[edge, 0],
        edge_indices[edge, 1],
        edge_indices[edge, 2],
        edge_indices[edge, 3],
    )
    weight = bending_weight(edge_bending_cotangent[edge])
    stiffness = edge_bending_properties[edge, 0] * 3.0 / edge_rest_area[edge]
    weighted_position = wp.vec3(0.0)
    for local in range(4):
        weighted_position += weight[local] * x[indices[local]]
    for local in range(4):
        wp.atomic_sub(rhs, indices[local], stiffness * weight[local] * weighted_position)


@wp.kernel
def add_barrier_forces(
    x: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    normal: wp.vec3,
    plane_offset: float,
    minimum_separation: float,
    contact_distance: float,
    barrier_stiffness: float,
    solve_active: wp.array[int],
    rhs: wp.array[wp.vec3],
    contact_hessian: wp.array[float],
):
    particle = wp.tid()
    contact_hessian[particle] = 0.0
    if solve_active[0] == 0 or (particle_flags[particle] & ParticleFlags.ACTIVE) == 0 or particle_mass[particle] <= 0.0:
        return

    gap = wp.dot(normal, x[particle]) - plane_offset - minimum_separation
    if gap > 0.0 and gap < contact_distance:
        inverse_distance_squared = 1.0 / (contact_distance * contact_distance)
        s = gap * gap * inverse_distance_squared
        ds_dd = 2.0 * gap * inverse_distance_squared
        first = barrier_first_derivative(s)
        second = barrier_second_derivative(s)
        gradient = barrier_stiffness * first * ds_dd
        hessian = barrier_stiffness * (second * ds_dd * ds_dd + first * 2.0 * inverse_distance_squared)
        wp.atomic_sub(rhs, particle, gradient * normal)
        contact_hessian[particle] = wp.max(hessian, 0.0)


@wp.kernel
def mask_rhs(
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    solve_active: wp.array[int],
    rhs: wp.array[wp.vec3],
):
    particle = wp.tid()
    if solve_active[0] == 0 or (particle_flags[particle] & ParticleFlags.ACTIVE) == 0 or particle_mass[particle] <= 0.0:
        rhs[particle] = wp.vec3(0.0)


@wp.kernel
def maximum_residual_squared(rhs: wp.array[wp.vec3], residual_squared: wp.array[float]):
    """Reduce the particle force residual with an infinity-over-blocks norm."""
    particle = wp.tid()
    wp.atomic_max(residual_squared, 0, wp.dot(rhs[particle], rhs[particle]))


@wp.kernel
def array_inner_tiled(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    count: int,
    output_index: int,
    output: wp.array[float],
):
    """Reduce a vector-array dot product with one atomic add per tile."""
    block, lane = wp.tid()
    index = block * wp.block_dim() + lane
    value = float(0.0)
    if index < count:
        value = wp.dot(a[index], b[index])
    tile_sum = wp.tile_sum(wp.tile(value))
    if lane == 0:
        wp.atomic_add(output, output_index, wp.tile_extract(tile_sum, 0))


@wp.kernel
def prepare_rank_one_preconditioner(
    static_diagonal: wp.array[float],
    contact_hessian: wp.array[float],
    normal: wp.vec3,
    particle_flags: wp.array[int],
    solve_active: wp.array[int],
    inverse_diagonal: wp.array[wp.mat33],
):
    particle = wp.tid()
    diagonal = static_diagonal[particle]
    if solve_active[0] != 0 and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0 and diagonal > 0.0:
        inverse = wp.identity(3, float) / diagonal
        hessian = contact_hessian[particle]
        if hessian > 0.0:
            inverse -= wp.outer(normal, normal) * (hessian / (diagonal * (diagonal + hessian)))
        inverse_diagonal[particle] = inverse
    else:
        inverse_diagonal[particle] = wp.mat33(0.0)


@wp.kernel
def prepare_dense_preconditioner(
    static_diagonal: wp.array[float],
    contact_hessian: wp.array[float],
    normal: wp.vec3,
    particle_flags: wp.array[int],
    solve_active: wp.array[int],
    inverse_diagonal: wp.array[wp.mat33],
):
    particle = wp.tid()
    diagonal = static_diagonal[particle]
    if solve_active[0] != 0 and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0 and diagonal > 0.0:
        matrix = wp.identity(3, float) * diagonal + wp.outer(normal, normal) * contact_hessian[particle]
        inverse_diagonal[particle] = wp.inverse(matrix)
    else:
        inverse_diagonal[particle] = wp.mat33(0.0)


@wp.kernel
def multiply_barrier_hessian(
    vector: wp.array[wp.vec3],
    contact_hessian: wp.array[float],
    normal: wp.vec3,
    product: wp.array[wp.vec3],
):
    particle = wp.tid()
    product[particle] = contact_hessian[particle] * wp.dot(normal, vector[particle]) * normal


@wp.kernel
def update_convergence(
    absolute_tolerance: float,
    relative_tolerance: float,
    residual_squared: wp.array[float],
    reference_residual: wp.array[float],
    residual: wp.array[float],
    newton_iterations: wp.array[int],
    solve_active: wp.array[int],
    status: wp.array[int],
    converged_status: int,
):
    if solve_active[0] == 0:
        return
    value = wp.sqrt(wp.max(residual_squared[0], 0.0))
    residual[0] = value
    if newton_iterations[0] == 0:
        reference_residual[0] = value
    tolerance = absolute_tolerance + relative_tolerance * reference_residual[0]
    if value <= tolerance:
        solve_active[0] = 0
        status[0] = converged_status


@wp.kernel
def mask_direction(
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    solve_active: wp.array[int],
    direction: wp.array[wp.vec3],
):
    particle = wp.tid()
    if solve_active[0] == 0 or (particle_flags[particle] & ParticleFlags.ACTIVE) == 0 or particle_mass[particle] <= 0.0:
        direction[particle] = wp.vec3(0.0)


@wp.kernel
def validate_direction(
    rhs_dot_direction: wp.array[float],
    solve_active: wp.array[int],
    status: wp.array[int],
    breakdown_status: int,
):
    if solve_active[0] != 0:
        value = rhs_dot_direction[0]
        if value <= 0.0 or wp.isnan(value) or wp.isinf(value):
            solve_active[0] = 0
            status[0] = breakdown_status


@wp.kernel
def bound_plane_step(
    x: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    normal: wp.vec3,
    plane_offset: float,
    minimum_separation: float,
    ccd_safety: float,
    solve_active: wp.array[int],
    alpha: wp.array[float],
):
    particle = wp.tid()
    if solve_active[0] == 0 or (particle_flags[particle] & ParticleFlags.ACTIVE) == 0 or particle_mass[particle] <= 0.0:
        return
    projected_direction = wp.dot(normal, direction[particle])
    if projected_direction < 0.0:
        gap = wp.dot(normal, x[particle]) - plane_offset - minimum_separation
        bound = ccd_safety * gap / -projected_direction
        wp.atomic_min(alpha, 0, wp.max(bound, 0.0))


@wp.kernel
def add_particle_energy(
    dt: float,
    x: wp.array[wp.vec3],
    x_predictor: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    normal: wp.vec3,
    plane_offset: float,
    minimum_separation: float,
    contact_distance: float,
    barrier_stiffness: float,
    energy: wp.array[float],
):
    particle = wp.tid()
    mass = particle_mass[particle]
    if (particle_flags[particle] & ParticleFlags.ACTIVE) == 0 or mass <= 0.0:
        return
    displacement = x[particle] - x_predictor[particle]
    value = 0.5 * mass * wp.dot(displacement, displacement) / (dt * dt)
    gap = wp.dot(normal, x[particle]) - plane_offset - minimum_separation
    if gap > 0.0 and gap < contact_distance:
        s = gap * gap / (contact_distance * contact_distance)
        value += barrier_stiffness * barrier_value(s)
    elif gap <= 0.0:
        value = wp.float32(3.4028235e38)
    wp.atomic_add(energy, 0, value)


@wp.kernel
def add_triangle_energy(
    x: wp.array[wp.vec3],
    triangle_area: wp.array[float],
    triangle_pose: wp.array[wp.mat22],
    triangle_indices: wp.array2d[int],
    triangle_stiffness: wp.array[wp.vec3],
    energy: wp.array[float],
):
    triangle = wp.tid()
    i = triangle_indices[triangle, 0]
    j = triangle_indices[triangle, 1]
    k = triangle_indices[triangle, 2]
    fu, fv = triangle_deformation_gradient(x[i], x[j], x[k], triangle_pose[triangle])
    stiffness = triangle_stiffness[triangle]
    stretch_u = wp.length(fu) - 1.0
    stretch_v = wp.length(fv) - 1.0
    shear = wp.dot(fu, fv)
    value = (
        0.5
        * triangle_area[triangle]
        * (stiffness[0] * stretch_u * stretch_u + stiffness[1] * stretch_v * stretch_v + stiffness[2] * shear * shear)
    )
    wp.atomic_add(energy, 0, value)


@wp.kernel
def add_bending_energy(
    x: wp.array[wp.vec3],
    edge_rest_area: wp.array[float],
    edge_bending_cotangent: wp.array[wp.vec4],
    edge_indices: wp.array2d[int],
    edge_bending_properties: wp.array2d[float],
    energy: wp.array[float],
):
    edge = wp.tid()
    if edge_indices[edge, 0] < 0 or edge_indices[edge, 1] < 0:
        return
    weight = bending_weight(edge_bending_cotangent[edge])
    weighted_position = wp.vec3(0.0)
    for local in range(4):
        weighted_position += weight[local] * x[edge_indices[edge, local]]
    stiffness = edge_bending_properties[edge, 0] * 3.0 / edge_rest_area[edge]
    wp.atomic_add(energy, 0, 0.5 * stiffness * wp.dot(weighted_position, weighted_position))


@wp.kernel
def prepare_candidate(
    x: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    alpha: wp.array[float],
    solve_active: wp.array[int],
    accepted: wp.array[int],
    candidate: wp.array[wp.vec3],
):
    particle = wp.tid()
    if (
        solve_active[0] != 0
        and accepted[0] == 0
        and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0
        and particle_mass[particle] > 0.0
    ):
        candidate[particle] = x[particle] + alpha[0] * direction[particle]
    else:
        candidate[particle] = x[particle]


@wp.kernel
def begin_line_search(
    solve_active: wp.array[int],
    accepted: wp.array[int],
    line_search_active: wp.array[int],
    current_iterations: wp.array[int],
):
    accepted[0] = 0
    current_iterations[0] = 0
    line_search_active[0] = solve_active[0]


@wp.kernel
def validate_candidate(
    candidate: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    normal: wp.vec3,
    plane_offset: float,
    minimum_separation: float,
    solve_active: wp.array[int],
    accepted: wp.array[int],
    invalid: wp.array[int],
):
    particle = wp.tid()
    if (
        solve_active[0] != 0
        and accepted[0] == 0
        and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0
        and particle_mass[particle] > 0.0
    ):
        gap = wp.dot(normal, candidate[particle]) - plane_offset - minimum_separation
        if gap <= 0.0 or wp.isnan(gap):
            wp.atomic_max(invalid, 0, 1)


@wp.kernel
def accept_candidate(
    armijo: float,
    energy_tolerance: float,
    current_energy: wp.array[float],
    candidate_energy: wp.array[float],
    rhs_dot_direction: wp.array[float],
    candidate_invalid: wp.array[int],
    solve_active: wp.array[int],
    alpha: wp.array[float],
    accepted: wp.array[int],
    line_search_active: wp.array[int],
    current_iterations: wp.array[int],
    line_search_iterations: wp.array[int],
    max_line_search_iterations: int,
):
    if solve_active[0] == 0 or line_search_active[0] == 0:
        return
    current_iterations[0] += 1
    line_search_iterations[0] += 1
    trial_energy = candidate_energy[0]
    threshold = current_energy[0] - armijo * alpha[0] * rhs_dot_direction[0]
    if (
        candidate_invalid[0] == 0
        and not wp.isnan(trial_energy)
        and not wp.isinf(trial_energy)
        and trial_energy <= threshold + energy_tolerance
    ):
        accepted[0] = 1
        line_search_active[0] = 0
    else:
        alpha[0] *= 0.5
        if current_iterations[0] >= max_line_search_iterations:
            line_search_active[0] = 0


@wp.kernel
def commit_candidate(
    candidate: wp.array[wp.vec3],
    accepted: wp.array[int],
    x: wp.array[wp.vec3],
):
    particle = wp.tid()
    if accepted[0] != 0:
        x[particle] = candidate[particle]


@wp.kernel
def finish_line_search(
    accepted: wp.array[int],
    solve_active: wp.array[int],
    status: wp.array[int],
    newton_iterations: wp.array[int],
    line_search_exhausted_status: int,
    max_newton_iterations: int,
    newton_exhausted_status: int,
):
    if solve_active[0] != 0:
        if accepted[0] == 0:
            solve_active[0] = 0
            status[0] = line_search_exhausted_status
        else:
            newton_iterations[0] += 1
            if newton_iterations[0] >= max_newton_iterations:
                solve_active[0] = 0
                status[0] = newton_exhausted_status


@wp.kernel
def finish_newton(
    solve_active: wp.array[int],
    status: wp.array[int],
    exhausted_status: int,
):
    if solve_active[0] != 0:
        solve_active[0] = 0
        status[0] = exhausted_status


@wp.kernel
def commit_step(
    dt: float,
    velocity_damping: float,
    converged_status: int,
    status: wp.array[int],
    normal: wp.vec3,
    plane_offset: float,
    minimum_separation: float,
    particle_mass: wp.array[float],
    particle_flags: wp.array[int],
    x_previous: wp.array[wp.vec3],
    x_current: wp.array[wp.vec3],
    velocity_previous: wp.array[wp.vec3],
    position_out: wp.array[wp.vec3],
    velocity_out: wp.array[wp.vec3],
    minimum_gap: wp.array[float],
):
    particle = wp.tid()
    converged = status[0] == converged_status
    dynamic = (particle_flags[particle] & ParticleFlags.ACTIVE) != 0 and particle_mass[particle] > 0.0
    if converged:
        position = x_current[particle]
        position_out[particle] = position
        if dynamic:
            velocity_out[particle] = velocity_damping * (position - x_previous[particle]) / dt
            gap = wp.dot(normal, position) - plane_offset - minimum_separation
            wp.atomic_min(minimum_gap, 0, gap)
        else:
            velocity_out[particle] = wp.vec3(0.0)
    else:
        position_out[particle] = x_previous[particle]
        velocity_out[particle] = velocity_previous[particle]


@wp.kernel
def record_step_status(
    converged_status: int,
    status: wp.array[int],
    failed_steps: wp.array[int],
):
    """Latch the number of failed steps so a later substep cannot hide one."""
    if status[0] != converged_status:
        failed_steps[0] += 1
