# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA-only particle-vs-shape soft contacts via texture SDFs.

This module is intentionally separate from :mod:`newton._src.geometry.kernels`
so that its kernel -- which references CUDA-only ``wp.Texture3D`` data through
:class:`~newton._src.geometry.sdf_texture.TextureSDFData` -- is only ever
compiled for CUDA. The shared :func:`~newton._src.geometry.kernels.create_soft_contacts`
kernel (which also runs on CPU) skips shapes that have a texture SDF; those
shapes are handled here instead, with one cheap SDF sample per particle rather
than a per-triangle mesh query.
"""

import warp as wp

from .flags import ParticleFlags
from .kernels import counter_increment
from .sdf_contact import safe_sdf_scale_inverse, scale_sdf_result_to_world
from .sdf_texture import TextureSDFData, texture_sample_sdf_grad


@wp.kernel
def create_soft_contacts_sdf(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_world: wp.array[int],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_data: wp.array[TextureSDFData],
    sdf_shape_indices: wp.array[int],
    num_sdf_shapes: int,
    replay_tid_offset: int,
    margin: float,
    soft_contact_max: int,
    # outputs
    soft_contact_count: wp.array[int],
    soft_contact_particle: wp.array[int],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    soft_contact_tids: wp.array[int],
):
    """Generate one soft contact per (particle, SDF-shape) pair by sampling the
    shape's texture SDF for signed distance and gradient.

    Launched over ``particle_count * num_sdf_shapes``; ``sdf_shape_indices`` maps
    the compact shape slot to the model shape index, so the launch only covers
    shapes that actually carry a texture SDF.
    """
    tid = wp.tid()
    particle_index = tid // num_sdf_shapes
    slot = tid % num_sdf_shapes
    shape_index = sdf_shape_indices[slot]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    sdf_idx = shape_sdf_index[shape_index]
    if sdf_idx < 0 or sdf_idx >= texture_sdf_data.shape[0]:
        return
    texture_sdf = texture_sdf_data[sdf_idx]
    if texture_sdf.coarse_texture.width == 0:
        return

    # Skip collision between different worlds (unless one is global)
    particle_world_id = particle_world[particle_index]
    shape_world_id = shape_world[shape_index]
    if particle_world_id != -1 and shape_world_id != -1 and particle_world_id != shape_world_id:
        return

    rigid_index = shape_body[shape_index]
    px = particle_q[particle_index]
    radius = particle_radius[particle_index]

    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_q[rigid_index]
    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)

    # particle position in shape-local space
    x_local = wp.transform_point(X_sw, px)

    # The SDF lives in unscaled mesh space; unscale the query point unless the
    # shape scale was baked into the SDF, then scale the result back.
    geo_scale = shape_scale[shape_index]
    sdf_scale = geo_scale
    if texture_sdf.scale_baked:
        sdf_scale = wp.vec3(1.0, 1.0, 1.0)
    inv_sdf_scale, min_sdf_scale = safe_sdf_scale_inverse(sdf_scale)
    x_unscaled = wp.cw_mul(x_local, inv_sdf_scale)

    d_unscaled, grad_unscaled = texture_sample_sdf_grad(texture_sdf, x_unscaled)
    d, n = scale_sdf_result_to_world(d_unscaled, grad_unscaled, sdf_scale, inv_sdf_scale, min_sdf_scale)

    if d < margin + radius:
        replay_tid = replay_tid_offset + tid
        index = counter_increment(soft_contact_count, 0, soft_contact_tids, replay_tid)

        if index < soft_contact_max:
            body_pos = wp.transform_point(X_bs, x_local - n * d)
            world_normal = wp.transform_vector(X_ws, n)

            soft_contact_shape[index] = shape_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = wp.vec3(0.0)
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal
