# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched camera frustum line generation for viewers."""

from __future__ import annotations

import math

import warp as wp

from ..core.cameras import CameraPinhole
from ..geometry.flags import CameraFlags

# segment topology: near rect (4), far rect (4), connecting edges (4)
_SEGMENTS_PER_CAMERA = 12


@wp.kernel(enable_backward=False)
def _frustum_lines_kernel(
    camera_transform: wp.array[wp.transform],
    camera_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    half_extents: wp.array[wp.vec4],  # (near_hw, near_hh, far_hw, far_hh)
    depths: wp.array[wp.vec2],  # (near, depth)
    visible: wp.array[wp.int32],
    camera_world: wp.array[wp.int32],
    world_offsets: wp.array[wp.vec3],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    cam = wp.tid()
    # Camera-to-world live: through the attached body when body >= 0, else the
    # stored transform is already world-space.
    body = camera_body[cam]
    if body >= 0:
        xf = body_q[body] * camera_transform[cam]
    else:
        xf = camera_transform[cam]
    offset = wp.vec3(0.0, 0.0, 0.0)
    world = camera_world[cam]
    if world >= 0 and world < world_offsets.shape[0]:
        offset = world_offsets[world]
    he = half_extents[cam]
    d = depths[cam]
    scale = float(visible[cam])  # 0 collapses hidden cameras to degenerate lines
    # corners: near plane z=-near, far plane z=-depth (camera looks along -Z)
    for i in range(4):
        sx = wp.where(i == 0 or i == 3, -1.0, 1.0)
        sy = wp.where(i < 2, 1.0, -1.0)
        n = wp.transform_point(xf, wp.vec3(sx * he[0], sy * he[1], -d[0]) * scale) + offset
        f = wp.transform_point(xf, wp.vec3(sx * he[2], sy * he[3], -d[1]) * scale) + offset
        j = (i + 1) % 4
        sxn = wp.where(j == 0 or j == 3, -1.0, 1.0)
        syn = wp.where(j < 2, 1.0, -1.0)
        n2 = wp.transform_point(xf, wp.vec3(sxn * he[0], syn * he[1], -d[0]) * scale) + offset
        f2 = wp.transform_point(xf, wp.vec3(sxn * he[2], syn * he[3], -d[1]) * scale) + offset
        base = cam * 12
        starts[base + i] = n
        ends[base + i] = n2
        starts[base + 4 + i] = f
        ends[base + 4 + i] = f2
        starts[base + 8 + i] = n
        ends[base + 8 + i] = f


class CameraFrustums:
    """Generates batched frustum line segments for all cameras of a model."""

    def __init__(self, model, depth: float = 0.5):
        self.model = model
        self.depth = depth
        n = model.camera_count
        device = model.device
        self.starts = wp.zeros(n * _SEGMENTS_PER_CAMERA, dtype=wp.vec3, device=device)
        self.ends = wp.zeros(n * _SEGMENTS_PER_CAMERA, dtype=wp.vec3, device=device)
        self._empty_offsets = wp.zeros(0, dtype=wp.vec3, device=device)
        # Bindable placeholder when the model has no bodies; the kernel only
        # reads body_q for body-attached cameras, which cannot exist then.
        self._empty_body_q = wp.zeros(0, dtype=wp.transform, device=device)
        # host-precomputed per-camera params (projections are static)
        half_extents = []
        depths = []
        visible = []
        flags = model.camera_flags.numpy()
        indices = model.camera_projection_index.numpy()
        for i in range(n):
            proj = model.camera_projections[indices[i]]
            far = min(proj.far, depth)
            near = min(proj.near, far)
            if isinstance(proj, CameraPinhole):
                tan_h = 0.5 * proj.vertical_aperture / proj.focal_length
                tan_w = 0.5 * proj.horizontal_aperture / proj.focal_length
            else:
                tan_h = tan_w = math.tan(math.radians(30.0))  # generic display frustum
            half_extents.append((near * tan_w, near * tan_h, far * tan_w, far * tan_h))
            depths.append((near, far))
            visible.append(1 if flags[i] & int(CameraFlags.VISIBLE) else 0)
        self._half_extents = wp.array(half_extents, dtype=wp.vec4, device=device)
        self._depths = wp.array(depths, dtype=wp.vec2, device=device)
        self._visible = wp.array(visible, dtype=wp.int32, device=device)

    def update(self, state, world_offsets: wp.array | None):
        """Recomputes frustum lines from the current camera poses.

        Body-attached cameras are composed with ``state.body_q`` (falling back
        to ``model.body_q``) live inside the kernel; no world transforms are
        stored.
        """
        if state is not None and state.body_q is not None:
            body_q = state.body_q
        elif self.model.body_q is not None:
            body_q = self.model.body_q
        else:
            body_q = self._empty_body_q
        offsets = world_offsets if world_offsets is not None else self._empty_offsets
        wp.launch(
            _frustum_lines_kernel,
            dim=self.model.camera_count,
            inputs=[
                self.model.camera_transform,
                self.model.camera_body,
                body_q,
                self._half_extents,
                self._depths,
                self._visible,
                self.model.camera_world,
                offsets,
            ],
            outputs=[self.starts, self.ends],
            device=self.model.device,
        )
