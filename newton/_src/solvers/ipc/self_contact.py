# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared surface queries, barrier stencils, and conservative advancement.

The BVH layout and AABB builders are shared with VBD's mesh collision queries.
Unlike a discrete penalty query, the search query includes swept primitive
bounds. All triangle edges (including boundaries) participate. Only incident
primitive pairs are excluded; there is no rest-distance exclusion.
"""

import numpy as np
import warp as wp

from ...geometry.kernels import compute_edge_aabbs, compute_tri_aabbs
from .kernels import barrier_first_derivative, barrier_second_derivative, barrier_value

mat43 = wp.types.matrix(shape=(4, 3), dtype=wp.float32)


@wp.struct
class PairBuffer:
    indices: wp.array[wp.vec4i]
    kind: wp.array[int]
    count: wp.array[int]
    overflow: wp.array[int]
    weights: wp.array[wp.vec4]
    normal: wp.array[wp.vec3]
    curvature: wp.array[float]


@wp.func
def segment_coordinate(a: wp.vec3d, b: wp.vec3d, p: wp.vec3d):
    e = b - a
    t = wp.float64(0.0)
    if wp.dot(e, e) > wp.float64(0.0):
        t = wp.clamp(wp.dot(p - a, e) / wp.dot(e, e), wp.float64(0.0), wp.float64(1.0))
    return t


@wp.func
def contact_features(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, kind: int):
    """Return closest-point difference and its four envelope-theorem weights.

    Evaluate all boundary features, including collapsed and parallel edges.
    Double precision avoids float32 cancellation in almost parallel queries.
    """
    weights = wp.vec4d(1.0, -1.0, 0.0, 0.0)
    diff = a - b
    if kind == 0:
        vertices = wp.matrix_from_rows(b, c, d)
        best = wp.float64(1.0e300)
        for i in range(3):
            j = (i + 1) % 3
            edge_a, edge_b = vertices[i], vertices[j]
            t = segment_coordinate(edge_a, edge_b, a)
            delta = a - (edge_a + t * (edge_b - edge_a))
            distance = wp.dot(delta, delta)
            if distance < best:
                best = distance
                diff = delta
                weights = wp.vec4d(1.0, 0.0, 0.0, 0.0)
                weights[i + 1] = t - wp.float64(1.0)
                weights[j + 1] = -t
        n = wp.cross(c - b, d - b)
        n2 = wp.dot(n, n)
        if n2 > wp.float64(0.0):
            u = wp.dot(wp.cross(c - a, d - a), n) / n2
            v = wp.dot(wp.cross(d - a, b - a), n) / n2
            w = wp.float64(1.0) - u - v
            if u >= wp.float64(0.0) and v >= wp.float64(0.0) and w >= wp.float64(0.0):
                weights = wp.vec4d(1.0, -u, -v, -w)
                diff = (wp.dot(a - b, n) / n2) * n
    else:
        e, f = b - a, d - c
        best = wp.float64(1.0e300)
        for endpoint in range(4):
            s, t = wp.float64(0.0), wp.float64(0.0)
            if endpoint < 2:
                if endpoint == 1:
                    s = wp.float64(1.0)
                t = segment_coordinate(c, d, a + s * e)
            else:
                if endpoint == 3:
                    t = wp.float64(1.0)
                s = segment_coordinate(a, b, c + t * f)
            delta = a + s * e - c - t * f
            distance = wp.dot(delta, delta)
            if distance < best:
                best = distance
                diff = delta
                weights = wp.vec4d(wp.float64(1.0) - s, s, t - wp.float64(1.0), -t)
        n = wp.cross(e, f)
        n2 = wp.dot(n, n)
        if n2 > wp.float64(0.0):
            s = wp.dot(wp.cross(c - a, f), n) / n2
            t = wp.dot(wp.cross(c - a, e), n) / n2
            if s >= wp.float64(0.0) and s <= wp.float64(1.0) and t >= wp.float64(0.0) and t <= wp.float64(1.0):
                diff = a + s * e - c - t * f
                weights = wp.vec4d(wp.float64(1.0) - s, s, t - wp.float64(1.0), -t)
    return diff, weights


@wp.func
def edge_mollifier(x: wp.array[wp.vec3], rest: wp.array[wp.vec3], ids: wp.vec4i, kind: int):
    value = float(1.0)
    gradient = mat43(0.0)
    if kind == 1:
        e, f = x[ids[1]] - x[ids[0]], x[ids[3]] - x[ids[2]]
        er, fr = rest[ids[1]] - rest[ids[0]], rest[ids[3]] - rest[ids[2]]
        epsilon = 1.0e-3 * wp.dot(er, er) * wp.dot(fr, fr)
        cross = wp.cross(e, f)
        cross2 = wp.dot(cross, cross)
        if epsilon > 0.0 and cross2 < epsilon:
            ratio = cross2 / epsilon
            value = ratio * (2.0 - ratio)
            factor = 4.0 * (1.0 - ratio) / epsilon
            ge = factor * wp.cross(f, cross)
            gf = factor * wp.cross(cross, e)
            gradient[0] = -ge
            gradient[1] = ge
            gradient[2] = -gf
            gradient[3] = gf
    return value, gradient


@wp.kernel
def swept_bounds(
    x: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    alpha: float,
    primitives: wp.array2d[int],
    first: int,
    size: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    i = wp.tid()
    lo, hi = wp.vec3(1.0e30), wp.vec3(-1.0e30)
    for j in range(size):
        vertex = primitives[i, first + j]
        p, q = x[vertex], x[vertex] + alpha * direction[vertex]
        lo = wp.min(lo, wp.min(p, q))
        hi = wp.max(hi, wp.max(p, q))
    # Include float32 rounding of intermediate trial positions in the bounds.
    padding = wp.vec3(1.0e-6 * wp.max(wp.length(lo), wp.length(hi)))
    lower[i] = lo - padding
    upper[i] = hi + padding


@wp.func
def append_pair(buffer: PairBuffer, ids: wp.vec4i, kind: int, mass: wp.array[float]):
    if mass[ids[0]] + mass[ids[1]] + mass[ids[2]] + mass[ids[3]] > 0.0:
        offset = wp.atomic_add(buffer.count, 0, 1)
        if offset < buffer.indices.shape[0]:
            buffer.indices[offset] = ids
            buffer.kind[offset] = kind
        else:
            wp.atomic_max(buffer.overflow, 0, 1)


@wp.kernel
def query_pairs(
    x: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    alpha: float,
    radius: float,
    triangles: wp.array2d[int],
    edges: wp.array2d[int],
    mass: wp.array[float],
    triangle_bvh: wp.uint64,
    edge_bvh: wp.uint64,
    edge_lower: wp.array[wp.vec3],
    edge_upper: wp.array[wp.vec3],
    buffer: PairBuffer,
):
    i = wp.tid()
    padding = wp.vec3(radius)
    target = int(0)
    if i < x.shape[0]:
        p, q = x[i], x[i] + alpha * direction[i]
        query = wp.bvh_query_aabb(triangle_bvh, wp.min(p, q) - padding, wp.max(p, q) + padding)
        while wp.bvh_query_next(query, target):
            a, b, c = triangles[target, 0], triangles[target, 1], triangles[target, 2]
            if i != a and i != b and i != c:
                append_pair(buffer, wp.vec4i(i, a, b, c), 0, mass)
    else:
        edge = i - x.shape[0]
        a, b = edges[edge, 2], edges[edge, 3]
        query = wp.bvh_query_aabb(edge_bvh, edge_lower[edge] - padding, edge_upper[edge] + padding)
        while wp.bvh_query_next(query, target):
            c, d = edges[target, 2], edges[target, 3]
            if target > edge and a != c and a != d and b != c and b != d:
                append_pair(buffer, wp.vec4i(a, b, c, d), 1, mass)


@wp.kernel
def query_pairs_reference(
    x: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    alpha: float,
    radius: float,
    triangles: wp.array2d[int],
    edges: wp.array2d[int],
    mass: wp.array[float],
    triangle_lower: wp.array[wp.vec3],
    triangle_upper: wp.array[wp.vec3],
    edge_lower: wp.array[wp.vec3],
    edge_upper: wp.array[wp.vec3],
    buffer: PairBuffer,
):
    """Use an exhaustive AABB reference on CPU, without BVH traversal."""
    index = wp.tid()
    vertex_pairs = x.shape[0] * triangles.shape[0]
    ids = wp.vec4i(0)
    kind = int(0)
    lo, hi, target_lo, target_hi = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
    if index < vertex_pairs:
        vertex = index // triangles.shape[0]
        face = index % triangles.shape[0]
        a, b, c = triangles[face, 0], triangles[face, 1], triangles[face, 2]
        if vertex == a or vertex == b or vertex == c:
            return
        ids = wp.vec4i(vertex, a, b, c)
        p, q = x[vertex], x[vertex] + alpha * direction[vertex]
        lo, hi = wp.min(p, q), wp.max(p, q)
        target_lo = triangle_lower[face]
        target_hi = triangle_upper[face]
    else:
        pair = index - vertex_pairs
        i, j = pair // edges.shape[0], pair % edges.shape[0]
        if i >= j:
            return
        a, b, c, d = edges[i, 2], edges[i, 3], edges[j, 2], edges[j, 3]
        if a == c or a == d or b == c or b == d:
            return
        ids, kind = wp.vec4i(a, b, c, d), 1
        lo = edge_lower[i]
        hi = edge_upper[i]
        target_lo = edge_lower[j]
        target_hi = edge_upper[j]
    for axis in range(3):
        if lo[axis] - radius > target_hi[axis] or hi[axis] + radius < target_lo[axis]:
            return
    append_pair(buffer, ids, kind, mass)


@wp.kernel
def fail_overflow(buffer: PairBuffer, active: wp.array[int], status: wp.array[int]):
    if buffer.overflow[0] != 0:
        active[0] = 0
        status[0] = 6


@wp.kernel
def validate_intersections(
    x: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    edges: wp.array2d[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    bvh: wp.uint64,
    invalid: wp.array[int],
):
    """Reject an initially pierced surface, not just zero PT/EE distances."""
    edge = wp.tid()
    i, j = edges[edge, 2], edges[edge, 3]
    p, q = wp.vec3d(x[i]), wp.vec3d(x[j])
    query = wp.bvh_query_aabb(bvh, lower[edge], upper[edge])
    face = int(0)
    while wp.bvh_query_next(query, face):
        ia, ib, ic = triangles[face, 0], triangles[face, 1], triangles[face, 2]
        if i != ia and i != ib and i != ic and j != ia and j != ib and j != ic:
            a, b, c = wp.vec3d(x[ia]), wp.vec3d(x[ib]), wp.vec3d(x[ic])
            normal = wp.cross(b - a, c - a)
            denominator = wp.dot(normal, q - p)
            if denominator != wp.float64(0.0):
                t = wp.dot(normal, a - p) / denominator
                if t >= wp.float64(0.0) and t <= wp.float64(1.0):
                    hit = p + t * (q - p)
                    u = wp.dot(wp.cross(b - hit, c - hit), normal)
                    v = wp.dot(wp.cross(c - hit, a - hit), normal)
                    w = wp.dot(wp.cross(a - hit, b - hit), normal)
                    if u >= wp.float64(0.0) and v >= wp.float64(0.0) and w >= wp.float64(0.0):
                        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def evaluate_pairs(
    x: wp.array[wp.vec3],
    rest: wp.array[wp.vec3],
    buffer: PairBuffer,
    thickness: float,
    activation: float,
    stiffness: float,
    guard: float,
    energy: wp.array[float],
    invalid: wp.array[int],
    minimum: wp.array[float],
):
    pair = wp.tid()
    if pair >= wp.min(buffer.count[0], buffer.indices.shape[0]):
        return
    ids = buffer.indices[pair]
    diff, _weights = contact_features(
        wp.vec3d(x[ids[0]]), wp.vec3d(x[ids[1]]), wp.vec3d(x[ids[2]]), wp.vec3d(x[ids[3]]), buffer.kind[pair]
    )
    distance = float(wp.length(diff))
    gap = distance - thickness
    wp.atomic_min(minimum, 0, gap)
    if not wp.isfinite(gap) or gap <= guard:
        wp.atomic_max(invalid, 0, 1)
        return
    if gap < activation:
        s = gap * gap / (activation * activation)
        mollifier, _gradient = edge_mollifier(x, rest, ids, buffer.kind[pair])
        wp.atomic_add(energy, 0, mollifier * stiffness * barrier_value(s))


@wp.kernel
def assemble_pairs(
    x: wp.array[wp.vec3],
    rest: wp.array[wp.vec3],
    buffer: PairBuffer,
    thickness: float,
    activation: float,
    stiffness: float,
    active: wp.array[int],
    rhs: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
):
    pair = wp.tid()
    if pair >= wp.min(buffer.count[0], buffer.indices.shape[0]):
        return
    buffer.curvature[pair] = 0.0
    if active[0] == 0:
        return
    ids = buffer.indices[pair]
    delta, bary = contact_features(
        wp.vec3d(x[ids[0]]), wp.vec3d(x[ids[1]]), wp.vec3d(x[ids[2]]), wp.vec3d(x[ids[3]]), buffer.kind[pair]
    )
    distance = float(wp.length(delta))
    gap = distance - thickness
    if gap <= 0.0 or gap >= activation:
        return
    normal = wp.vec3(delta / wp.float64(distance))
    weights = wp.vec4(bary)
    s = gap * gap / (activation * activation)
    ds = 2.0 * gap / (activation * activation)
    energy = stiffness * barrier_value(s)
    first = stiffness * barrier_first_derivative(s) * ds
    second = stiffness * (
        barrier_second_derivative(s) * ds * ds + barrier_first_derivative(s) * 2.0 / (activation * activation)
    )
    mollifier, gradient = edge_mollifier(x, rest, ids, buffer.kind[pair])
    curvature = wp.max(0.0, mollifier * second)
    buffer.normal[pair] = normal
    buffer.weights[pair] = weights
    buffer.curvature[pair] = curvature
    for j in range(4):
        wp.atomic_sub(rhs, ids[j], mollifier * first * weights[j] * normal + energy * gradient[j])
        wp.atomic_add(diagonal, ids[j], curvature * weights[j] * weights[j] * wp.outer(normal, normal))


@wp.kernel
def multiply_pairs(buffer: PairBuffer, vector: wp.array[wp.vec3], result: wp.array[wp.vec3]):
    pair = wp.tid()
    if pair >= wp.min(buffer.count[0], buffer.indices.shape[0]) or buffer.curvature[pair] == 0.0:
        return
    ids, weights, normal = buffer.indices[pair], buffer.weights[pair], buffer.normal[pair]
    value = wp.vec3(0.0)
    for j in range(4):
        value += weights[j] * vector[ids[j]]
    product = buffer.curvature[pair] * wp.dot(normal, value) * normal
    for j in range(4):
        wp.atomic_add(result, ids[j], weights[j] * product)


@wp.kernel
def invert_blocks(
    diagonal: wp.array[wp.mat33],
    static_diagonal: wp.array[float],
    plane_hessian: wp.array[float],
    normal: wp.vec3,
    inverse: wp.array[wp.mat33],
):
    i = wp.tid()
    if static_diagonal[i] > 0.0:
        inverse[i] = wp.inverse(
            diagonal[i] + static_diagonal[i] * wp.identity(3, float) + plane_hessian[i] * wp.outer(normal, normal)
        )
    else:
        inverse[i] = wp.mat33(0.0)


@wp.kernel
def bound_pairs(
    x: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    buffer: PairBuffer,
    thickness: float,
    guard: float,
    safety: float,
    max_alpha: float,
    iterations: int,
    active: wp.array[int],
    alpha: wp.array[float],
):
    """Advance using a Lipschitz bound on primitive distance, never past it.

    At exhaustion return the already certified prefix, not the proposed end.
    Distance queries are float64; the guard also covers float32 trial rounding.
    This is conservative advancement, not a polynomial time-of-impact root finder.
    """
    pair = wp.tid()
    if pair >= wp.min(buffer.count[0], buffer.indices.shape[0]) or active[0] == 0:
        return
    ids, kind = buffer.indices[pair], buffer.kind[pair]
    a, b, c, d = wp.vec3d(x[ids[0]]), wp.vec3d(x[ids[1]]), wp.vec3d(x[ids[2]]), wp.vec3d(x[ids[3]])
    pa, pb, pc, pd = (
        wp.vec3d(direction[ids[0]]),
        wp.vec3d(direction[ids[1]]),
        wp.vec3d(direction[ids[2]]),
        wp.vec3d(direction[ids[3]]),
    )
    mean = (pa + pb + pc + pd) / wp.float64(4.0)
    scale = wp.max(wp.max(wp.length(a), wp.length(b)), wp.max(wp.length(c), wp.length(d)))
    motion = wp.max(wp.max(wp.length(pa), wp.length(pb)), wp.max(wp.length(pc), wp.length(pd)))
    rounding = wp.float64(guard) + wp.float64(1.0e-6) * (scale + wp.float64(max_alpha) * motion)
    pa, pb, pc, pd = pa - mean, pb - mean, pc - mean, pd - mean
    speed = wp.length(pa) + wp.max(wp.length(pb), wp.max(wp.length(pc), wp.length(pd)))
    if kind == 1:
        speed = wp.max(wp.length(pa), wp.length(pb)) + wp.max(wp.length(pc), wp.length(pd))
    if speed == wp.float64(0.0):
        return
    time = wp.float64(0.0)
    for _iteration in range(iterations):
        diff, _weights = contact_features(a + time * pa, b + time * pb, c + time * pc, d + time * pd, kind)
        gap = wp.length(diff) - wp.float64(thickness) - rounding
        if not wp.isfinite(gap) or gap <= wp.float64(0.0):
            break
        increment = wp.float64(safety) * gap / speed
        if time + increment >= wp.float64(max_alpha):
            time = wp.float64(max_alpha)
            break
        if increment <= wp.float64(1.0e-8) * wp.max(wp.float64(1.0), time):
            break
        time += increment
    # Rounding down avoids extending the certified prefix on float conversion.
    wp.atomic_min(alpha, 0, float(time * wp.float64(0.9999999)))


class SelfContact:
    """Own fixed-capacity contact storage and graph-stable surface BVHs."""

    def __init__(self, model, config):
        self.model, self.config, self.device = model, config, model.device
        tris = model.tri_indices.numpy()
        edges = np.unique(np.sort(np.concatenate((tris[:, :2], tris[:, 1:], tris[:, ::2])), axis=1), axis=0)
        edge_indices = np.full((len(edges), 4), -1, dtype=np.int32)
        edge_indices[:, 2:] = edges
        self.edges = wp.array(edge_indices, dtype=int, device=self.device)
        self.rest = wp.clone(model.particle_q)
        self.zero_direction = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self.minimum = wp.full(1, 1.0e30, dtype=float, device=self.device)
        self.scratch_energy = wp.zeros(1, dtype=float, device=self.device)
        self.lower_tri = wp.zeros(model.tri_count, dtype=wp.vec3, device=self.device)
        self.upper_tri = wp.zeros_like(self.lower_tri)
        self.lower_edge = wp.zeros(len(edges), dtype=wp.vec3, device=self.device)
        self.upper_edge = wp.zeros_like(self.lower_edge)
        wp.launch(
            compute_tri_aabbs,
            dim=model.tri_count,
            inputs=[model.particle_q, model.tri_indices, self.lower_tri, self.upper_tri],
            device=self.device,
        )
        wp.launch(
            compute_edge_aabbs,
            dim=len(edges),
            inputs=[model.particle_q, self.edges, self.lower_edge, self.upper_edge],
            device=self.device,
        )
        self.triangle_bvh = wp.Bvh(self.lower_tri, self.upper_tri)
        self.edge_bvh = wp.Bvh(self.lower_edge, self.upper_edge)
        self.capacity = config.self_contact_capacity or max(4096, 32 * (model.particle_count + len(edges)))
        self.current, self.swept = self._buffer(), self._buffer()

    def _buffer(self):
        buffer = PairBuffer()
        for name, dtype, size in (
            ("indices", wp.vec4i, self.capacity),
            ("kind", int, self.capacity),
            ("weights", wp.vec4, self.capacity),
            ("normal", wp.vec3, self.capacity),
            ("curvature", float, self.capacity),
            ("count", int, 1),
            ("overflow", int, 1),
        ):
            setattr(buffer, name, wp.zeros(size, dtype=dtype, device=self.device))
        return buffer

    def query(self, x, direction=None):
        swept = direction is not None
        direction = direction if swept else self.zero_direction
        alpha = self.config.initial_step_size if swept else 0.0
        buffer = self.swept if swept else self.current
        buffer.count.zero_()
        buffer.overflow.zero_()
        for primitives, first, size, lower, upper, bvh in (
            (self.model.tri_indices, 0, 3, self.lower_tri, self.upper_tri, self.triangle_bvh),
            (self.edges, 2, 2, self.lower_edge, self.upper_edge, self.edge_bvh),
        ):
            wp.launch(
                swept_bounds,
                dim=len(lower),
                inputs=[x, direction, alpha, primitives, first, size, lower, upper],
                device=self.device,
            )
            bvh.refit()
        radius = self.config.self_contact_thickness + self.config.self_contact_distance + self.config.self_contact_guard
        if self.device.is_cuda:
            wp.launch(
                query_pairs,
                dim=self.model.particle_count + len(self.edges),
                inputs=[
                    x,
                    direction,
                    alpha,
                    radius,
                    self.model.tri_indices,
                    self.edges,
                    self.model.particle_mass,
                    self.triangle_bvh.id,
                    self.edge_bvh.id,
                    self.lower_edge,
                    self.upper_edge,
                    buffer,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                query_pairs_reference,
                dim=self.model.particle_count * self.model.tri_count + len(self.edges) ** 2,
                inputs=[
                    x,
                    direction,
                    alpha,
                    radius,
                    self.model.tri_indices,
                    self.edges,
                    self.model.particle_mass,
                    self.lower_tri,
                    self.upper_tri,
                    self.lower_edge,
                    self.upper_edge,
                    buffer,
                ],
                device=self.device,
            )

    def check_overflow(self, active, status, *, swept=False):
        wp.launch(
            fail_overflow, dim=1, inputs=[self.swept if swept else self.current, active, status], device=self.device
        )

    def validate_initial(self, x, invalid):
        wp.launch(
            validate_intersections,
            dim=len(self.edges),
            inputs=[
                x,
                self.model.tri_indices,
                self.edges,
                self.lower_edge,
                self.upper_edge,
                self.triangle_bvh.id,
                invalid,
            ],
            device=self.device,
        )

    def energy(self, x, energy, invalid, *, swept=False):
        c = self.config
        wp.launch(
            evaluate_pairs,
            dim=self.capacity,
            inputs=[
                x,
                self.rest,
                self.swept if swept else self.current,
                c.self_contact_thickness,
                c.self_contact_distance,
                c.self_contact_stiffness,
                c.self_contact_guard,
                energy,
                invalid,
                self.minimum,
            ],
            device=self.device,
        )

    def assemble(self, x, active, rhs):
        c = self.config
        self.diagonal.zero_()
        wp.launch(
            assemble_pairs,
            dim=self.capacity,
            inputs=[
                x,
                self.rest,
                self.current,
                c.self_contact_thickness,
                c.self_contact_distance,
                c.self_contact_stiffness,
                active,
                rhs,
                self.diagonal,
            ],
            device=self.device,
        )

    def multiply(self, vector, result):
        wp.launch(multiply_pairs, dim=self.capacity, inputs=[self.current, vector, result], device=self.device)

    def bound(self, x, direction, active, alpha):
        c = self.config
        wp.launch(
            bound_pairs,
            dim=self.capacity,
            inputs=[
                x,
                direction,
                self.swept,
                c.self_contact_thickness,
                c.self_contact_guard,
                c.ccd_safety,
                c.initial_step_size,
                c.self_contact_ccd_iterations,
                active,
                alpha,
            ],
            device=self.device,
        )
