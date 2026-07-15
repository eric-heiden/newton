# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-boundary rasterization for the MAC fluid solver.

Each step, the union signed-distance field of the collider shapes is sampled
at cell centers to classify cells as fluid or solid, and the rigid-body
velocity field is sampled at grid faces adjacent to solid cells. Bodies are
treated as immersed voxelized boundaries: the classification is binary
(cut-cell face fractions are future work), so the no-slip boundary is
resolved to ``O(dx)``.
"""

from __future__ import annotations

import warp as wp

from ...geometry.kernels import (
    sdf_box,
    sdf_capsule,
    sdf_cone,
    sdf_cylinder,
    sdf_ellipsoid,
    sdf_mesh,
    sdf_sphere,
)
from ...geometry.types import Axis, GeoType
from .grid import CELL_FLUID, CELL_STATIC_SOLID, cell_center, face_position, is_solid_cell

__all__ = ["supported_collider_shapes"]


def supported_collider_shapes(model) -> list[int]:
    """Return indices of model shapes the fluid solver can rasterize."""
    supported = {
        int(GeoType.SPHERE),
        int(GeoType.BOX),
        int(GeoType.CAPSULE),
        int(GeoType.CYLINDER),
        int(GeoType.CONE),
        int(GeoType.ELLIPSOID),
        int(GeoType.MESH),
    }
    shape_type = model.shape_type.numpy()
    return [i for i in range(int(model.shape_count)) if int(shape_type[i]) in supported]


@wp.func
def _shape_sdf(
    geo: int,
    scale: wp.vec3,
    mesh_id: wp.uint64,
    x_local: wp.vec3,
    max_dist: float,
) -> float:
    """Signed distance of ``x_local`` (shape frame) to one collider shape."""
    if geo == GeoType.SPHERE:
        return sdf_sphere(x_local, scale[0])
    if geo == GeoType.BOX:
        return sdf_box(x_local, scale[0], scale[1], scale[2])
    if geo == GeoType.CAPSULE:
        return sdf_capsule(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.CYLINDER:
        return sdf_cylinder(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.CONE:
        return sdf_cone(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.ELLIPSOID:
        return sdf_ellipsoid(x_local, scale)
    if geo == GeoType.MESH and mesh_id != wp.uint64(0):
        # Query in the mesh frame; accurate for uniform scale, conservative
        # (smallest scale magnitude) for nonuniform scale.
        min_scale = wp.min(wp.abs(scale))
        if min_scale > 0.0:
            local = wp.cw_div(x_local, scale)
            return sdf_mesh(mesh_id, local, max_dist / min_scale) * min_scale
    return max_dist


@wp.func
def _collider_sdf(
    pos: wp.vec3,
    collider_shapes: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    body_q: wp.array[wp.transform],
    max_dist: float,
):
    """Minimum signed distance over all collider shapes at world ``pos``.

    Returns the distance and the body index of the closest shape
    (``CELL_STATIC_SOLID`` for static shapes).
    """
    min_sdf = max_dist
    min_body = int(CELL_STATIC_SOLID)

    for c in range(collider_shapes.shape[0]):
        s = collider_shapes[c]
        body = shape_body[s]
        shape_x = shape_transform[s]
        if body >= 0:
            shape_x = body_q[body] * shape_x
        x_local = wp.transform_point(wp.transform_inverse(shape_x), pos)
        d = _shape_sdf(shape_type[s], shape_scale[s], shape_source_ptr[s], x_local, max_dist)
        if d < min_sdf:
            min_sdf = d
            if body >= 0:
                min_body = body
            else:
                min_body = int(CELL_STATIC_SOLID)

    return min_sdf, min_body


@wp.func
def body_point_velocity(
    body: int,
    pos: wp.vec3,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
) -> wp.vec3:
    """Rigid-body velocity at world position ``pos`` [m/s]."""
    qd = body_qd[body]
    v_com = wp.spatial_top(qd)
    omega = wp.spatial_bottom(qd)
    com_world = wp.transform_point(body_q[body], body_com[body])
    return v_com + wp.cross(omega, pos - com_world)


@wp.kernel(enable_backward=False)
def rasterize_colliders_kernel(
    origin: wp.vec3,
    dx: float,
    collider_shapes: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    body_q: wp.array[wp.transform],
    max_dist: float,
    # outputs
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    fluid_cell_count: wp.array[wp.int32],
):
    i, j, k = wp.tid()
    pos = cell_center(origin, dx, i, j, k)
    sdf, body = _collider_sdf(
        pos,
        collider_shapes,
        shape_body,
        shape_transform,
        shape_type,
        shape_scale,
        shape_source_ptr,
        body_q,
        max_dist,
    )
    cell_sdf[i, j, k] = sdf
    if sdf < 0.0:
        cell_label[i, j, k] = body
    else:
        cell_label[i, j, k] = CELL_FLUID
        wp.atomic_add(fluid_cell_count, 0, 1)


@wp.func
def face_solid_owner(
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    ia: int,
    ja: int,
    ka: int,
    ib: int,
    jb: int,
    kb: int,
) -> int:
    """Owner label for a face between cells A and B.

    Returns ``CELL_FLUID`` for pure fluid faces; otherwise the label of the
    solid neighbor (the deeper one when both are solid). Out-of-domain
    neighbors are static walls.
    """
    solid_a = is_solid_cell(cell_label, ia, ja, ka)
    solid_b = is_solid_cell(cell_label, ib, jb, kb)
    if not solid_a and not solid_b:
        return CELL_FLUID

    nx = cell_label.shape[0]
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]

    label_a = int(CELL_STATIC_SOLID)
    sdf_a = -1.0e9
    if ia >= 0 and ja >= 0 and ka >= 0 and ia < nx and ja < ny and ka < nz:
        label_a = cell_label[ia, ja, ka]
        sdf_a = cell_sdf[ia, ja, ka]

    label_b = int(CELL_STATIC_SOLID)
    sdf_b = -1.0e9
    if ib >= 0 and jb >= 0 and kb >= 0 and ib < nx and jb < ny and kb < nz:
        label_b = cell_label[ib, jb, kb]
        sdf_b = cell_sdf[ib, jb, kb]

    if solid_a and solid_b:
        # prefer the deeper (more negative sdf) solid neighbor
        if sdf_a <= sdf_b:
            return label_a
        return label_b
    if solid_a:
        return label_a
    return label_b


@wp.func
def _update_face_solid(
    axis: int,
    i: int,
    j: int,
    k: int,
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    vel: wp.array3d[float],
    vel_solid: wp.array3d[float],
):
    ia = i
    ja = j
    ka = k
    if axis == 0:
        ia = i - 1
    elif axis == 1:
        ja = j - 1
    else:
        ka = k - 1

    owner = face_solid_owner(cell_label, cell_sdf, ia, ja, ka, i, j, k)
    if owner == CELL_FLUID:
        return

    u_s = float(0.0)
    if owner >= 0:
        pos = face_position(origin, dx, axis, i, j, k)
        u_s = body_point_velocity(owner, pos, body_q, body_qd, body_com)[axis]

    vel_solid[i, j, k] = u_s
    vel[i, j, k] = u_s


@wp.kernel(enable_backward=False)
def update_faces_u_kernel(
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    u: wp.array3d[float],
    u_solid: wp.array3d[float],
):
    i, j, k = wp.tid()
    _update_face_solid(0, i, j, k, origin, dx, cell_label, cell_sdf, body_q, body_qd, body_com, u, u_solid)


@wp.kernel(enable_backward=False)
def update_faces_v_kernel(
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    v: wp.array3d[float],
    v_solid: wp.array3d[float],
):
    i, j, k = wp.tid()
    _update_face_solid(1, i, j, k, origin, dx, cell_label, cell_sdf, body_q, body_qd, body_com, v, v_solid)


@wp.kernel(enable_backward=False)
def update_faces_w_kernel(
    origin: wp.vec3,
    dx: float,
    cell_label: wp.array3d[wp.int32],
    cell_sdf: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    w: wp.array3d[float],
    w_solid: wp.array3d[float],
):
    i, j, k = wp.tid()
    _update_face_solid(2, i, j, k, origin, dx, cell_label, cell_sdf, body_q, body_qd, body_com, w, w_solid)
