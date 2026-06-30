# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-based fluid (PBF) kernels for :class:`~newton.solvers.SolverXPBD`.

Implements the smoothing kernels and the density-constraint, cohesion,
viscosity, and vorticity passes of Macklin & Müller, "Position Based Fluids"
(2013). Kept separate from the general XPBD constraint kernels in
:mod:`~newton._src.solvers.xpbd.kernels` so the fluid solve can be read and
maintained on its own.
"""

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
    See Akinci et al. (2013).
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


@wp.func
def _is_active_fluid_particle(flags: wp.int32) -> bool:
    return (flags & (ParticleFlags.ACTIVE | ParticleFlags.FLUID)) == (ParticleFlags.ACTIVE | ParticleFlags.FLUID)


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
    if capacity == 0:
        return int(-1)

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
        x_wb = wp.transform_identity()
        if body_index >= 0:
            x_wb = body_q[body_index]

        x_ws = wp.transform_multiply(x_wb, shape_transform[shape_index])
        x_sw = wp.transform_inverse(x_ws)
        x_local = wp.transform_point(x_sw, x)

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

        n_world = wp.transform_vector(x_ws, n_local)
        normal_len = wp.length(n_world)
        if normal_len > EPS:
            n_world = n_world / normal_len
        else:
            shape_center = wp.transform_get_translation(x_ws)
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
        body_v += wp.transform_vector(x_ws, mesh_v_local)

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
def compute_fluid_lambdas(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_invmass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    smoothing_length: float,
    rest_density: float,
    relaxation_epsilon: float,
    max_neighbors: int,
    rest_distance: float,
    # outputs
    fluid_density: wp.array[float],
    fluid_lambda: wp.array[float],
):
    """Compute smoothing-kernel densities and density-constraint Lagrange multipliers.

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
    world_id = particle_world[i]
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
    query = wp.hash_grid_query(grid, x, h, world_id)
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
    particle_world: wp.array[wp.int32],
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
    world_id = particle_world[i]
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

    query = wp.hash_grid_query(grid, x, h, world_id)
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
    particle_world: wp.array[wp.int32],
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
    world_id = particle_world[i]
    h = smoothing_length
    omega = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, h, world_id)
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
    particle_world: wp.array[wp.int32],
    fluid_density: wp.array[float],
    fluid_vorticity: wp.array[wp.vec3],
    smoothing_length: float,
    viscosity: float,
    vorticity_confinement: float,
    dt: float,
    # outputs
    v_out: wp.array[wp.vec3],
):
    """Post-projection velocity pass for viscosity and vorticity confinement.

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
    world_id = particle_world[i]
    h = smoothing_length
    omega_i = fluid_vorticity[i]

    rho_i = wp.max(fluid_density[i], 1.0e-6)
    weight_sum = particle_mass[i] / rho_i * poly6_kernel(0.0, h)
    v_weighted = v * weight_sum
    eta = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, h, world_id)
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


@wp.kernel
def sample_fluid_render_positions(
    particle_q: wp.array[wp.vec3],
    particle_count: int,
    render_count: int,
    out_render_q: wp.array[wp.vec3],
):
    tid = wp.tid()
    source = wp.min((tid * particle_count) // render_count, particle_count - 1)
    out_render_q[tid] = particle_q[source]


@wp.kernel
def sample_fluid_render_particles(
    render_q: wp.array[wp.vec3],
    anisotropy: wp.array[wp.vec4],
    anisotropy_secondary: wp.array[wp.vec4],
    anisotropy_tertiary: wp.array[wp.vec4],
    particle_count: int,
    render_count: int,
    out_render_q: wp.array[wp.vec3],
    out_anisotropy: wp.array[wp.vec4],
    out_anisotropy_secondary: wp.array[wp.vec4],
    out_anisotropy_tertiary: wp.array[wp.vec4],
):
    tid = wp.tid()
    source = wp.min((tid * particle_count) // render_count, particle_count - 1)
    out_render_q[tid] = render_q[source]
    out_anisotropy[tid] = anisotropy[source]
    out_anisotropy_secondary[tid] = anisotropy_secondary[source]
    out_anisotropy_tertiary[tid] = anisotropy_tertiary[source]


@wp.kernel
def compute_fluid_render_particles(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
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
    if not _is_active_fluid_particle(particle_flags[i]):
        out_render_q[i] = xi
        out_anisotropy[i] = wp.vec4(1.0, 0.0, 0.0, 0.0)
        out_anisotropy_secondary[i] = wp.vec4(0.0, 1.0, 0.0, 0.0)
        out_anisotropy_tertiary[i] = wp.vec4(0.0, 0.0, 1.0, 0.0)
        return

    world_id = particle_world[i]
    h = wp.max(smoothing_length, EPS)
    h2 = h * h
    weighted_center = wp.vec3(0.0)
    separation = wp.vec3(0.0)
    weight_sum = float(0.0)
    neighbor_count = int(0)
    moment_xx = float(0.0)
    moment_xy = float(0.0)
    moment_xz = float(0.0)
    moment_yy = float(0.0)
    moment_yz = float(0.0)
    moment_zz = float(0.0)

    query = wp.hash_grid_query(grid, xi, h, world_id)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if not _is_active_fluid_particle(particle_flags[j]):
            continue

        xj = particle_q[j]
        r_vec = xi - xj
        r2 = wp.dot(r_vec, r_vec)
        if r2 < h2:
            falloff = wp.max(1.0 - r2 / h2, 0.0)
            w = falloff * falloff
            weighted_center += xj * w
            weight_sum += w
            neighbor_count += 1
            neighbor_delta = xj - xi
            moment_xx += neighbor_delta[0] * neighbor_delta[0] * w
            moment_xy += neighbor_delta[0] * neighbor_delta[1] * w
            moment_xz += neighbor_delta[0] * neighbor_delta[2] * w
            moment_yy += neighbor_delta[1] * neighbor_delta[1] * w
            moment_yz += neighbor_delta[1] * neighbor_delta[2] * w
            moment_zz += neighbor_delta[2] * neighbor_delta[2] * w

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
        inv_weight = 1.0 / wp.max(weight_sum, EPS)
        center_delta = center - xi
        cxx = wp.max(moment_xx * inv_weight - center_delta[0] * center_delta[0], 0.0)
        cxy = moment_xy * inv_weight - center_delta[0] * center_delta[1]
        cxz = moment_xz * inv_weight - center_delta[0] * center_delta[2]
        cyy = wp.max(moment_yy * inv_weight - center_delta[1] * center_delta[1], 0.0)
        cyz = moment_yz * inv_weight - center_delta[1] * center_delta[2]
        czz = wp.max(moment_zz * inv_weight - center_delta[2] * center_delta[2], 0.0)

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
        side_axis = _normalize_or(cov_side - axis * wp.dot(cov_side, axis), side_axis)
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
def update_fluid_diffuse_particles(
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

    query = wp.hash_grid_query(grid, x, smoothing_length, world_idx)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if not _is_active_fluid_particle(fluid_flags[j]):
            continue

        r_vec = x - fluid_q[j]
        r2 = wp.dot(r_vec, r_vec)
        if r2 < smoothing_length * smoothing_length:
            w = poly6_kernel(r2, smoothing_length)
            weighted_v += fluid_qd[j] * w
            weight_sum += w
            neighbors += 1

    if neighbors >= diffuse_ballistic and weight_sum > EPS:
        target_v = weighted_v / weight_sum
        blend = wp.min(wp.max(diffuse_drag * dt, 0.0), 1.0)
        v = v * (1.0 - blend) + target_v * blend
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
def advance_fluid_diffuse_seed(frame_seed: wp.array[wp.int32]):
    frame_seed[0] = frame_seed[0] + 1


@wp.kernel
def spawn_fluid_diffuse_particles(
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
    if i == -1 or not _is_active_fluid_particle(fluid_flags[i]):
        return

    xi = fluid_q[i]
    vi = fluid_qd[i]
    world_id = fluid_world[i]
    speed_sq = wp.dot(vi, vi)
    divergence = float(0.0)
    neighbors = int(0)
    separation = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, smoothing_length, world_id)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_fluid_particle(fluid_flags[j]):
            continue

        r_vec = xi - fluid_q[j]
        r = wp.length(r_vec)
        if r > EPS and r < smoothing_length:
            rel_v = vi - fluid_qd[j]
            divergence += wp.max(wp.dot(rel_v, r_vec / r), 0.0)
            separation += (r_vec / r) * (1.0 - r / smoothing_length)
            neighbors += 1

    is_surface = density[i] < rest_density * diffuse_surface_density_ratio or neighbors < diffuse_ballistic

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
        surface_scale = 0.06

    probability = wp.min(diffuse_spawn_probability * surface_scale * (potential / threshold - 1.0), 1.0)
    seed = float((i + 1) * 928371 + (frame_seed[0] + 17) * 68917)
    if _hash01(seed) > probability:
        return

    request = wp.atomic_add(diffuse_spawn_counter, 0, 1)
    slot = _reserve_diffuse_slot(request, diffuse_slot_state)
    if slot < 0:
        return
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
    diffuse_qd[slot] = wp.vec4(spawn_v[0], spawn_v[1], spawn_v[2], visual_neighbors)
    diffuse_world[slot] = world_id


@wp.kernel
def collide_fluid_diffuse_particles_with_shapes(
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
    dt: float,
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
        dt,
        0,
    )

    diffuse_q[i] = wp.vec4(x[0], x[1], x[2], life)
    diffuse_qd[i] = wp.vec4(v[0], v[1], v[2], v_neighbors[3])
