# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for experimental soft-object cutting examples.

The examples in this package use a small process-zone cutting model around
Newton's existing solvers. It is intentionally factored out so future tracks
can reuse the same knife trajectory, material parameters, plotting, and video
capture while swapping the solver integration strategy.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp


@dataclass(frozen=True)
class KnifeProfile:
    """Axis-aligned blade front moving through a soft block.

    The blade front advances along x. The blade is centered at ``center_y`` and
    ``center_z`` and affects particles inside the y/z blade window and inside a
    finite cohesive process zone around the front.
    """

    start_x: float = -0.08
    speed: float = 0.35
    center_y: float = 0.0
    center_z: float = 0.0
    half_width_y: float = 0.06
    half_width_z: float = 0.34
    process_width: float = 0.045

    def x_at(self, time: float) -> float:
        return self.start_x + self.speed * time

    def signed_distance_x(self, points: np.ndarray, time: float) -> np.ndarray:
        points = np.asarray(points)
        return points[..., 0] - self.x_at(time)

    def cut_weights(self, points: np.ndarray, time: float) -> np.ndarray:
        points = np.asarray(points)
        phi = np.abs(self.signed_distance_x(points, time))
        in_front = np.clip(1.0 - phi / max(self.process_width, 1.0e-12), 0.0, 1.0)
        in_y = np.abs(points[..., 1] - self.center_y) <= self.half_width_y
        in_z = np.abs(points[..., 2] - self.center_z) <= self.half_width_z
        return np.where(in_y & in_z, in_front, 0.0).astype(np.float32)

    def blade_segments(self, time: float, tail: float = 0.16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return line segments that draw the blade rectangle and short handle."""

        x = self.x_at(time)
        y0 = self.center_y - self.half_width_y
        y1 = self.center_y + self.half_width_y
        z0 = self.center_z - self.half_width_z
        z1 = self.center_z + self.half_width_z
        points = np.array(
            [
                [x, y0, z0],
                [x, y1, z0],
                [x, y1, z1],
                [x, y0, z1],
                [x - tail, self.center_y, z1],
            ],
            dtype=np.float32,
        )
        starts = np.array([points[0], points[1], points[2], points[3], points[4]], dtype=np.float32)
        ends = np.array([points[1], points[2], points[3], points[0], [x, self.center_y, z1]], dtype=np.float32)
        colors = np.tile(np.array([[0.95, 0.95, 0.98]], dtype=np.float32), (starts.shape[0], 1))
        return starts, ends, colors


@dataclass(frozen=True)
class CutMaterial:
    """Minimal material model for a cohesive process-zone cut."""

    fracture_energy: float = 80.0
    yield_stress: float = 1.5e4
    damping: float = 0.03
    max_damage_rate: float = 14.0
    separation_speed: float = 0.22
    force_scale: float = 1.0


@dataclass(frozen=True)
class ParticleCutUpdate:
    damage: np.ndarray
    force: float
    active_count: int
    mean_damage: float


@dataclass(frozen=True)
class AdaptiveRemeshStats:
    active_x_segments: int
    surface_vertex_count: int
    surface_triangle_count: int
    wall_vertex_count: int
    wall_triangle_count: int
    coarse_dx: float
    min_active_dx: float
    max_active_dx: float


@wp.func
def _cutting_smoothstep(value: float):
    x = wp.min(1.0, wp.max(0.0, value))
    return x * x * (3.0 - 2.0 * x)


@wp.func
def _cutting_gap_at(x: float, knife_x: float, max_gap: float, front_width: float):
    return max_gap * _cutting_smoothstep((knife_x - x) / front_width)


@wp.func
def _cutting_deform_from_particles(
    rest_point: wp.vec3,
    rest_particle_points: wp.array(dtype=wp.vec3),
    particle_points: wp.array(dtype=wp.vec3),
    particle_count: int,
):
    if particle_count <= 0:
        return rest_point

    weighted_delta = wp.vec3(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    for i in range(particle_count):
        rest_particle = rest_particle_points[i]
        current_particle = particle_points[i]
        d = rest_point - rest_particle
        d2 = wp.dot(d, d)
        weight = 1.0 / wp.max(d2, 1.0e-10)
        weighted_delta = weighted_delta + (current_particle - rest_particle) * weight
        weight_sum = weight_sum + weight

    return rest_point + weighted_delta / wp.max(weight_sum, 1.0e-10)


@wp.func
def _cutting_write_quad(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    quad_index: int,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    rest_particle_points: wp.array(dtype=wp.vec3),
    particle_points: wp.array(dtype=wp.vec3),
    particle_count: int,
):
    vertex = quad_index * 4
    index = quad_index * 6
    points[vertex + 0] = _cutting_deform_from_particles(a, rest_particle_points, particle_points, particle_count)
    points[vertex + 1] = _cutting_deform_from_particles(b, rest_particle_points, particle_points, particle_count)
    points[vertex + 2] = _cutting_deform_from_particles(c, rest_particle_points, particle_points, particle_count)
    points[vertex + 3] = _cutting_deform_from_particles(d, rest_particle_points, particle_points, particle_count)
    indices[index + 0] = vertex + 0
    indices[index + 1] = vertex + 1
    indices[index + 2] = vertex + 2
    indices[index + 3] = vertex + 0
    indices[index + 4] = vertex + 2
    indices[index + 5] = vertex + 3


@wp.func
def _cutting_write_empty_quad(points: wp.array(dtype=wp.vec3), indices: wp.array(dtype=wp.int32), quad_index: int):
    vertex = quad_index * 4
    index = quad_index * 6
    zero = wp.vec3(0.0, 0.0, 0.0)
    points[vertex + 0] = zero
    points[vertex + 1] = zero
    points[vertex + 2] = zero
    points[vertex + 3] = zero
    indices[index + 0] = 0
    indices[index + 1] = 0
    indices[index + 2] = 0
    indices[index + 3] = 0
    indices[index + 4] = 0
    indices[index + 5] = 0


@wp.kernel
def _build_adaptive_cut_surface_kernel(
    x_segments: wp.array(dtype=wp.vec2),
    active_segment_count: int,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    rest_particle_points: wp.array(dtype=wp.vec3),
    particle_points: wp.array(dtype=wp.vec3),
    particle_count: int,
    block_lo: wp.vec3,
    block_hi: wp.vec3,
    knife_x: float,
    knife_center_y: float,
    max_gap: float,
    front_width: float,
    height_segments: int,
    quads_per_x_segment: int,
    cap_quad_offset: int,
):
    quad = wp.tid()
    dz = (block_hi[2] - block_lo[2]) / float(height_segments)

    if quad < cap_quad_offset:
        segment_id = quad / quads_per_x_segment
        local = quad - segment_id * quads_per_x_segment
        if segment_id >= active_segment_count:
            _cutting_write_empty_quad(points, indices, quad)
            return

        side_id = local / (height_segments + 2)
        face_id = local - side_id * (height_segments + 2)
        side = -1.0
        if side_id == 1:
            side = 1.0

        segment = x_segments[segment_id]
        x0 = segment[0]
        x1 = segment[1]
        y_outer = block_lo[1]
        if side > 0.0:
            y_outer = block_hi[1]
        y_cut0 = knife_center_y + side * 0.5 * _cutting_gap_at(x0, knife_x, max_gap, front_width)
        y_cut1 = knife_center_y + side * 0.5 * _cutting_gap_at(x1, knife_x, max_gap, front_width)

        if face_id == 0:
            _cutting_write_quad(
                points,
                indices,
                quad,
                wp.vec3(x0, y_outer, block_hi[2]),
                wp.vec3(x1, y_outer, block_hi[2]),
                wp.vec3(x1, y_cut1, block_hi[2]),
                wp.vec3(x0, y_cut0, block_hi[2]),
                rest_particle_points,
                particle_points,
                particle_count,
            )
        elif face_id == 1:
            _cutting_write_quad(
                points,
                indices,
                quad,
                wp.vec3(x0, y_cut0, block_lo[2]),
                wp.vec3(x1, y_cut1, block_lo[2]),
                wp.vec3(x1, y_outer, block_lo[2]),
                wp.vec3(x0, y_outer, block_lo[2]),
                rest_particle_points,
                particle_points,
                particle_count,
            )
        else:
            z_id = face_id - 2
            z0 = block_lo[2] + float(z_id) * dz
            z1 = z0 + dz
            _cutting_write_quad(
                points,
                indices,
                quad,
                wp.vec3(x0, y_outer, z0),
                wp.vec3(x1, y_outer, z0),
                wp.vec3(x1, y_outer, z1),
                wp.vec3(x0, y_outer, z1),
                rest_particle_points,
                particle_points,
                particle_count,
            )
        return

    cap_id = quad - cap_quad_offset
    cap_count = 4 * height_segments
    if cap_id >= cap_count:
        _cutting_write_empty_quad(points, indices, quad)
        return

    end_id = cap_id / (2 * height_segments)
    rem = cap_id - end_id * 2 * height_segments
    side_id = rem / height_segments
    z_id = rem - side_id * height_segments

    side = -1.0
    if side_id == 1:
        side = 1.0
    x = block_lo[0]
    if end_id == 1:
        x = block_hi[0]
    y_outer = block_lo[1]
    if side > 0.0:
        y_outer = block_hi[1]
    y_cut = knife_center_y + side * 0.5 * _cutting_gap_at(x, knife_x, max_gap, front_width)
    z0 = block_lo[2] + float(z_id) * dz
    z1 = z0 + dz

    if end_id == 0:
        _cutting_write_quad(
            points,
            indices,
            quad,
            wp.vec3(x, y_outer, z0),
            wp.vec3(x, y_cut, z0),
            wp.vec3(x, y_cut, z1),
            wp.vec3(x, y_outer, z1),
            rest_particle_points,
            particle_points,
            particle_count,
        )
    else:
        _cutting_write_quad(
            points,
            indices,
            quad,
            wp.vec3(x, y_cut, z0),
            wp.vec3(x, y_outer, z0),
            wp.vec3(x, y_outer, z1),
            wp.vec3(x, y_cut, z1),
            rest_particle_points,
            particle_points,
            particle_count,
        )


@wp.kernel
def _build_adaptive_cut_wall_kernel(
    x_segments: wp.array(dtype=wp.vec2),
    active_segment_count: int,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    rest_particle_points: wp.array(dtype=wp.vec3),
    particle_points: wp.array(dtype=wp.vec3),
    particle_count: int,
    block_lo: wp.vec3,
    block_hi: wp.vec3,
    knife_x: float,
    knife_center_y: float,
    max_gap: float,
    front_width: float,
    height_segments: int,
):
    quad = wp.tid()
    quads_per_segment = 2 * height_segments
    segment_id = quad / quads_per_segment
    local = quad - segment_id * quads_per_segment
    if segment_id >= active_segment_count:
        _cutting_write_empty_quad(points, indices, quad)
        return

    side_id = local / height_segments
    z_id = local - side_id * height_segments
    side = -1.0
    if side_id == 1:
        side = 1.0

    segment = x_segments[segment_id]
    x0 = segment[0]
    x1 = segment[1]
    dz = (block_hi[2] - block_lo[2]) / float(height_segments)
    z0 = block_lo[2] + float(z_id) * dz
    z1 = z0 + dz
    y_cut0 = knife_center_y + side * 0.5 * _cutting_gap_at(x0, knife_x, max_gap, front_width)
    y_cut1 = knife_center_y + side * 0.5 * _cutting_gap_at(x1, knife_x, max_gap, front_width)

    _cutting_write_quad(
        points,
        indices,
        quad,
        wp.vec3(x0, y_cut0, z0),
        wp.vec3(x1, y_cut1, z0),
        wp.vec3(x1, y_cut1, z1),
        wp.vec3(x0, y_cut0, z1),
        rest_particle_points,
        particle_points,
        particle_count,
    )


class SplitCuboidRenderMesh:
    """Render-only cuboid remesh with duplicated seam vertices and cut walls.

    This is deliberately a visualization layer. It keeps a fixed topology so it
    can be updated every frame, while the vertices around the knife path are
    duplicated and separated to show a visible slit behind the blade.
    """

    def __init__(
        self,
        block_lo: tuple[float, float, float] | np.ndarray,
        block_hi: tuple[float, float, float] | np.ndarray,
        knife: KnifeProfile,
        max_gap: float = 0.12,
        segments: int = 48,
        front_width: float | None = None,
        motion_sample_count: int = 16,
    ):
        self.block_lo = np.asarray(block_lo, dtype=np.float32)
        self.block_hi = np.asarray(block_hi, dtype=np.float32)
        self.knife = knife
        self.max_gap = float(max_gap)
        self.segments = int(max(2, segments))
        self.front_width = float(front_width if front_width is not None else max(2.0 * knife.process_width, 1.0e-4))
        self.motion_sample_count = int(max(1, motion_sample_count))
        self.x_values = np.linspace(self.block_lo[0], self.block_hi[0], self.segments + 1, dtype=np.float32)

        surface_points, wall_points = self.build_points(time=0.0)
        self.surface_points_np = surface_points
        self.wall_points_np = wall_points
        self.surface_indices_np = self._quad_indices(len(surface_points) // 4)
        self.wall_indices_np = self._quad_indices(len(wall_points) // 4)

        self.surface_points_wp: wp.array | None = None
        self.wall_points_wp: wp.array | None = None
        self.surface_indices_wp: wp.array | None = None
        self.wall_indices_wp: wp.array | None = None

    @staticmethod
    def _quad_indices(quad_count: int) -> np.ndarray:
        indices = np.empty(quad_count * 6, dtype=np.int32)
        for q in range(quad_count):
            v = q * 4
            indices[q * 6 : q * 6 + 6] = [v, v + 1, v + 2, v, v + 2, v + 3]
        return indices

    @staticmethod
    def _smoothstep(value: float) -> float:
        x = min(1.0, max(0.0, value))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _append_quad(vertices: list[list[float]], a, b, c, d):
        vertices.extend([list(a), list(b), list(c), list(d)])

    def gap_at(self, x: float, time: float) -> float:
        knife_x = self.knife.x_at(time)
        return self.max_gap * self._smoothstep((knife_x - float(x)) / self.front_width)

    def build_points(
        self,
        time: float,
        rest_particle_points: np.ndarray | None = None,
        particle_points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = self.block_lo
        hi = self.block_hi
        z0 = float(lo[2])
        z1 = float(hi[2])
        surface: list[list[float]] = []
        walls: list[list[float]] = []
        gaps = np.array([self.gap_at(float(x), time) for x in self.x_values], dtype=np.float32)

        for side in (-1.0, 1.0):
            y_outer = float(lo[1] if side < 0.0 else hi[1])
            for i in range(self.segments):
                x0 = float(self.x_values[i])
                x1 = float(self.x_values[i + 1])
                y_cut0 = self.knife.center_y + side * 0.5 * float(gaps[i])
                y_cut1 = self.knife.center_y + side * 0.5 * float(gaps[i + 1])

                self._append_quad(surface, (x0, y_outer, z1), (x1, y_outer, z1), (x1, y_cut1, z1), (x0, y_cut0, z1))
                self._append_quad(surface, (x0, y_cut0, z0), (x1, y_cut1, z0), (x1, y_outer, z0), (x0, y_outer, z0))
                self._append_quad(surface, (x0, y_outer, z0), (x1, y_outer, z0), (x1, y_outer, z1), (x0, y_outer, z1))
                self._append_quad(walls, (x0, y_cut0, z0), (x1, y_cut1, z0), (x1, y_cut1, z1), (x0, y_cut0, z1))

                if i == 0:
                    self._append_quad(
                        surface,
                        (x0, y_outer, z0),
                        (x0, y_cut0, z0),
                        (x0, y_cut0, z1),
                        (x0, y_outer, z1),
                    )
                if i == self.segments - 1:
                    self._append_quad(
                        surface,
                        (x1, y_cut1, z0),
                        (x1, y_outer, z0),
                        (x1, y_outer, z1),
                        (x1, y_cut1, z1),
                    )

        surface_np = np.asarray(surface, dtype=np.float32)
        walls_np = np.asarray(walls, dtype=np.float32)
        if rest_particle_points is not None or particle_points is not None:
            surface_np = self._deform_points_with_particles(surface_np, rest_particle_points, particle_points)
            walls_np = self._deform_points_with_particles(walls_np, rest_particle_points, particle_points)
        return surface_np, walls_np

    def _deform_points_with_particles(
        self,
        points: np.ndarray,
        rest_particle_points: np.ndarray | None,
        particle_points: np.ndarray | None,
    ) -> np.ndarray:
        if rest_particle_points is None or particle_points is None:
            raise ValueError("rest_particle_points and particle_points must be provided together")
        if points.size == 0:
            return points

        rest = np.asarray(rest_particle_points, dtype=np.float32)
        current = np.asarray(particle_points, dtype=np.float32)
        if rest.shape != current.shape or rest.ndim != 2 or rest.shape[1] != 3:
            raise ValueError("particle motion arrays must have matching shape (N, 3)")
        if rest.shape[0] == 0:
            return points

        k = min(self.motion_sample_count, rest.shape[0])
        deltas = current - rest
        d2 = np.sum((points[:, None, :] - rest[None, :, :]) ** 2, axis=2)
        if k == rest.shape[0]:
            nearest_d2 = d2
            nearest_deltas = np.broadcast_to(deltas[None, :, :], (points.shape[0], rest.shape[0], 3))
        else:
            nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
            nearest_d2 = np.take_along_axis(d2, nearest, axis=1)
            nearest_deltas = deltas[nearest]

        weights = 1.0 / np.maximum(nearest_d2, 1.0e-12)
        sampled_delta = np.sum(nearest_deltas * weights[:, :, None], axis=1) / np.sum(weights, axis=1)[:, None]
        return (points + sampled_delta).astype(np.float32)

    def _ensure_device_arrays(self, device):
        if self.surface_points_wp is not None:
            return
        self.surface_points_wp = wp.array(self.surface_points_np, dtype=wp.vec3, device=device)
        self.wall_points_wp = wp.array(self.wall_points_np, dtype=wp.vec3, device=device)
        self.surface_indices_wp = wp.array(self.surface_indices_np, dtype=wp.int32, device=device)
        self.wall_indices_wp = wp.array(self.wall_indices_np, dtype=wp.int32, device=device)

    def log(
        self,
        viewer,
        device,
        time: float,
        prefix: str = "/cutting/render_split",
        surface_color: tuple[float, float, float] = (0.18, 0.62, 0.95),
        wall_color: tuple[float, float, float] = (0.95, 0.32, 0.42),
        rest_particle_points: np.ndarray | None = None,
        particle_points: np.ndarray | None = None,
    ):
        self._ensure_device_arrays(device)
        self.surface_points_np, self.wall_points_np = self.build_points(
            time,
            rest_particle_points=rest_particle_points,
            particle_points=particle_points,
        )
        assert self.surface_points_wp is not None
        assert self.wall_points_wp is not None
        assert self.surface_indices_wp is not None
        assert self.wall_indices_wp is not None
        self.surface_points_wp.assign(self.surface_points_np)
        self.wall_points_wp.assign(self.wall_points_np)
        viewer.log_mesh(
            f"{prefix}/surface",
            self.surface_points_wp,
            self.surface_indices_wp,
            hidden=False,
            backface_culling=False,
            color=surface_color,
            roughness=0.68,
        )
        viewer.log_mesh(
            f"{prefix}/cut_walls",
            self.wall_points_wp,
            self.wall_indices_wp,
            hidden=self.knife.x_at(time) < self.block_lo[0],
            backface_culling=False,
            color=wall_color,
            roughness=0.82,
        )


class AdaptiveCutSurfaceRemesher:
    """Warp-backed render remesher with local refinement near the knife front.

    The remesher uses fixed-capacity buffers so the geometry can be regenerated
    every frame without changing allocation size. A small host-side schedule
    decides which coarse x cells need refinement; Warp kernels emit duplicated
    seam vertices, cut-wall triangles, and particle-motion deformation.
    """

    def __init__(
        self,
        block_lo: tuple[float, float, float] | np.ndarray,
        block_hi: tuple[float, float, float] | np.ndarray,
        knife: KnifeProfile,
        max_gap: float = 0.12,
        base_segments: int = 24,
        refine_factor: int = 4,
        refine_band: float | None = None,
        height_segments: int = 6,
        front_width: float | None = None,
    ):
        self.block_lo = np.asarray(block_lo, dtype=np.float32)
        self.block_hi = np.asarray(block_hi, dtype=np.float32)
        self.knife = knife
        self.max_gap = float(max_gap)
        self.base_segments = int(max(2, base_segments))
        self.refine_factor = int(max(1, refine_factor))
        self.refine_band = float(refine_band if refine_band is not None else 2.0 * knife.process_width)
        self.height_segments = int(max(1, height_segments))
        self.front_width = float(front_width if front_width is not None else max(2.0 * knife.process_width, 1.0e-4))

        self.coarse_dx = float((self.block_hi[0] - self.block_lo[0]) / self.base_segments)
        self.max_x_segments = self.base_segments * self.refine_factor
        self.quads_per_x_segment = 2 * (self.height_segments + 2)
        self.surface_cap_quads = 4 * self.height_segments
        self.surface_max_quads = self.max_x_segments * self.quads_per_x_segment + self.surface_cap_quads
        self.wall_max_quads = self.max_x_segments * 2 * self.height_segments

        self.x_segments_np = np.zeros((self.max_x_segments, 2), dtype=np.float32)
        self.empty_particles_np = np.zeros((0, 3), dtype=np.float32)

        self.x_segments_wp: wp.array | None = None
        self.surface_points_wp: wp.array | None = None
        self.surface_indices_wp: wp.array | None = None
        self.wall_points_wp: wp.array | None = None
        self.wall_indices_wp: wp.array | None = None
        self.empty_particles_wp: wp.array | None = None
        self.device_key: str | None = None
        self.last_stats = AdaptiveRemeshStats(
            active_x_segments=0,
            surface_vertex_count=0,
            surface_triangle_count=0,
            wall_vertex_count=0,
            wall_triangle_count=0,
            coarse_dx=self.coarse_dx,
            min_active_dx=self.coarse_dx,
            max_active_dx=self.coarse_dx,
        )

    @staticmethod
    def _as_vec3_wp(points: np.ndarray | wp.array | None, device) -> tuple[wp.array | None, int]:
        if points is None:
            return None, 0
        if isinstance(points, wp.array):
            return points, len(points)
        points_np = np.asarray(points, dtype=np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise ValueError("particle points must have shape (N, 3)")
        return wp.array(points_np, dtype=wp.vec3, device=device), int(points_np.shape[0])

    def _ensure_device_arrays(self, device):
        device_key = str(device)
        if self.surface_points_wp is not None and self.device_key == device_key:
            return
        self.device_key = device_key
        self.x_segments_wp = wp.array(self.x_segments_np, dtype=wp.vec2, device=device)
        self.surface_points_wp = wp.empty(self.surface_max_quads * 4, dtype=wp.vec3, device=device)
        self.surface_indices_wp = wp.empty(self.surface_max_quads * 6, dtype=wp.int32, device=device)
        self.wall_points_wp = wp.empty(self.wall_max_quads * 4, dtype=wp.vec3, device=device)
        self.wall_indices_wp = wp.empty(self.wall_max_quads * 6, dtype=wp.int32, device=device)
        self.empty_particles_wp = wp.array(self.empty_particles_np, dtype=wp.vec3, device=device)

    def _build_x_segments(self, time: float) -> tuple[int, float, float]:
        knife_x = self.knife.x_at(time)
        coarse_nodes = np.linspace(self.block_lo[0], self.block_hi[0], self.base_segments + 1, dtype=np.float32)
        count = 0
        min_dx = float("inf")
        max_dx = 0.0
        self.x_segments_np.fill(0.0)

        for i in range(self.base_segments):
            x0 = float(coarse_nodes[i])
            x1 = float(coarse_nodes[i + 1])
            center = 0.5 * (x0 + x1)
            splits = self.refine_factor if abs(center - knife_x) <= self.refine_band else 1
            dx = (x1 - x0) / splits
            for j in range(splits):
                if count >= self.max_x_segments:
                    break
                a = x0 + j * dx
                b = x0 + (j + 1) * dx
                self.x_segments_np[count, 0] = a
                self.x_segments_np[count, 1] = b
                min_dx = min(min_dx, b - a)
                max_dx = max(max_dx, b - a)
                count += 1

        if count == 0:
            min_dx = self.coarse_dx
            max_dx = self.coarse_dx
        return count, float(min_dx), float(max_dx)

    def update(
        self,
        device,
        time: float,
        rest_particle_points: np.ndarray | wp.array | None = None,
        particle_points: np.ndarray | wp.array | None = None,
    ) -> AdaptiveRemeshStats:
        self._ensure_device_arrays(device)
        active_x_segments, min_dx, max_dx = self._build_x_segments(time)
        assert self.x_segments_wp is not None
        assert self.surface_points_wp is not None
        assert self.surface_indices_wp is not None
        assert self.wall_points_wp is not None
        assert self.wall_indices_wp is not None
        assert self.empty_particles_wp is not None

        rest_wp, rest_count = self._as_vec3_wp(rest_particle_points, device)
        current_wp, current_count = self._as_vec3_wp(particle_points, device)
        if rest_wp is None and current_wp is None:
            rest_wp = self.empty_particles_wp
            current_wp = self.empty_particles_wp
            particle_count = 0
        elif rest_wp is None or current_wp is None or rest_count != current_count:
            raise ValueError("rest_particle_points and particle_points must be provided together with equal length")
        else:
            particle_count = rest_count

        self.x_segments_wp.assign(self.x_segments_np)
        block_lo = wp.vec3(float(self.block_lo[0]), float(self.block_lo[1]), float(self.block_lo[2]))
        block_hi = wp.vec3(float(self.block_hi[0]), float(self.block_hi[1]), float(self.block_hi[2]))
        knife_x = float(self.knife.x_at(time))

        wp.launch(
            _build_adaptive_cut_surface_kernel,
            dim=self.surface_max_quads,
            inputs=[
                self.x_segments_wp,
                active_x_segments,
                self.surface_points_wp,
                self.surface_indices_wp,
                rest_wp,
                current_wp,
                particle_count,
                block_lo,
                block_hi,
                knife_x,
                float(self.knife.center_y),
                self.max_gap,
                self.front_width,
                self.height_segments,
                self.quads_per_x_segment,
                active_x_segments * self.quads_per_x_segment,
            ],
            device=device,
        )
        wp.launch(
            _build_adaptive_cut_wall_kernel,
            dim=self.wall_max_quads,
            inputs=[
                self.x_segments_wp,
                active_x_segments,
                self.wall_points_wp,
                self.wall_indices_wp,
                rest_wp,
                current_wp,
                particle_count,
                block_lo,
                block_hi,
                knife_x,
                float(self.knife.center_y),
                self.max_gap,
                self.front_width,
                self.height_segments,
            ],
            device=device,
        )

        active_surface_quads = active_x_segments * self.quads_per_x_segment + self.surface_cap_quads
        active_wall_quads = active_x_segments * 2 * self.height_segments
        self.last_stats = AdaptiveRemeshStats(
            active_x_segments=active_x_segments,
            surface_vertex_count=active_surface_quads * 4,
            surface_triangle_count=active_surface_quads * 2,
            wall_vertex_count=active_wall_quads * 4,
            wall_triangle_count=active_wall_quads * 2,
            coarse_dx=self.coarse_dx,
            min_active_dx=min_dx,
            max_active_dx=max_dx,
        )
        return self.last_stats

    def log(
        self,
        viewer,
        device,
        time: float,
        prefix: str = "/cutting/adaptive_remesh",
        surface_color: tuple[float, float, float] = (0.18, 0.62, 0.95),
        wall_color: tuple[float, float, float] = (0.95, 0.32, 0.42),
        rest_particle_points: np.ndarray | wp.array | None = None,
        particle_points: np.ndarray | wp.array | None = None,
    ) -> AdaptiveRemeshStats:
        stats = self.update(
            device,
            time,
            rest_particle_points=rest_particle_points,
            particle_points=particle_points,
        )
        assert self.surface_points_wp is not None
        assert self.surface_indices_wp is not None
        assert self.wall_points_wp is not None
        assert self.wall_indices_wp is not None
        viewer.log_mesh(
            f"{prefix}/surface",
            self.surface_points_wp,
            self.surface_indices_wp,
            hidden=False,
            backface_culling=False,
            color=surface_color,
            roughness=0.68,
        )
        viewer.log_mesh(
            f"{prefix}/cut_walls",
            self.wall_points_wp,
            self.wall_indices_wp,
            hidden=self.knife.x_at(time) < self.block_lo[0],
            backface_culling=False,
            color=wall_color,
            roughness=0.82,
        )
        return stats


@dataclass
class RuntimeStats:
    solver: str
    frame_count: int
    sim_seconds: float
    wall_seconds: float
    mean_step_ms: float
    mean_render_ms: float
    fps: float
    peak_force_n: float
    mean_force_n: float
    force_impulse_ns: float
    final_mean_damage: float
    hardware: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def summarize_remesh_history(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {
            "frame_count": 0.0,
            "mean_active_x_segments": 0.0,
            "max_active_x_segments": 0.0,
            "mean_surface_triangles": 0.0,
            "max_surface_triangles": 0.0,
            "mean_wall_triangles": 0.0,
            "max_wall_triangles": 0.0,
            "min_active_dx": 0.0,
            "max_active_dx": 0.0,
        }

    active_x = np.array([row["active_x_segments"] for row in history], dtype=np.float32)
    surface_triangles = np.array([row["surface_triangle_count"] for row in history], dtype=np.float32)
    wall_triangles = np.array([row["wall_triangle_count"] for row in history], dtype=np.float32)
    min_dx = np.array([row["min_active_dx"] for row in history], dtype=np.float32)
    max_dx = np.array([row["max_active_dx"] for row in history], dtype=np.float32)
    return {
        "frame_count": float(len(history)),
        "mean_active_x_segments": float(np.mean(active_x)),
        "max_active_x_segments": float(np.max(active_x)),
        "mean_surface_triangles": float(np.mean(surface_triangles)),
        "max_surface_triangles": float(np.max(surface_triangles)),
        "mean_wall_triangles": float(np.mean(wall_triangles)),
        "max_wall_triangles": float(np.max(wall_triangles)),
        "min_active_dx": float(np.min(min_dx)),
        "max_active_dx": float(np.max(max_dx)),
    }


def compute_particle_cut_update(
    points: np.ndarray,
    damage: np.ndarray,
    knife: KnifeProfile,
    material: CutMaterial,
    dt: float,
    particle_volume: float,
    time: float = 0.0,
) -> ParticleCutUpdate:
    """Compute a NumPy reference update for the shared cutting model."""

    points = np.asarray(points, dtype=np.float32)
    damage = np.asarray(damage, dtype=np.float32)
    weights = knife.cut_weights(points, time)
    active = weights > 0.0
    damage_increment = material.max_damage_rate * dt * weights * (1.0 - damage)
    new_damage = np.clip(damage + damage_increment, 0.0, 1.0)

    area = max(float(particle_volume), 1.0e-18) ** (2.0 / 3.0)
    process_width = max(float(knife.process_width), 1.0e-9)
    damage_rate = np.divide(new_damage - damage, max(dt, 1.0e-9))
    yield_force = material.yield_stress * area * weights * (1.0 - damage)
    fracture_force = material.fracture_energy * area / process_width * damage_rate
    force = float(material.force_scale * np.sum((yield_force + fracture_force)[active]))

    return ParticleCutUpdate(
        damage=new_damage.astype(np.float32),
        force=force,
        active_count=int(np.count_nonzero(active)),
        mean_damage=float(np.mean(new_damage)) if len(new_damage) else 0.0,
    )


def summarize_force_profile(times: np.ndarray, forces: np.ndarray, damage: np.ndarray) -> dict[str, float]:
    times = np.asarray(times, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    damage = np.asarray(damage, dtype=np.float64)
    if len(forces) == 0:
        return {
            "peak_force_n": 0.0,
            "mean_force_n": 0.0,
            "force_impulse_ns": 0.0,
            "final_mean_damage": 0.0,
        }

    impulse = float(np.trapezoid(forces, times)) if len(forces) > 1 else 0.0
    return {
        "peak_force_n": float(np.max(forces)),
        "mean_force_n": float(np.mean(forces)),
        "force_impulse_ns": impulse,
        "final_mean_damage": float(damage[-1]) if len(damage) else 0.0,
    }


@wp.kernel
def apply_mpm_knife_cut_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    damage: wp.array[wp.float32],
    colors: wp.array[wp.vec3],
    accum: wp.array[wp.float32],
    knife_x: float,
    center_y: float,
    center_z: float,
    half_width_y: float,
    half_width_z: float,
    process_width: float,
    dt: float,
    particle_area: float,
    fracture_energy: float,
    yield_stress: float,
    max_damage_rate: float,
    separation_speed: float,
    force_scale: float,
):
    tid = wp.tid()
    q = particle_q[tid]
    phi = q[0] - knife_x
    y_rel = q[1] - center_y
    z_rel = q[2] - center_z
    weight = wp.max(0.0, 1.0 - wp.abs(phi) / wp.max(process_width, 1.0e-6))

    active = weight > 0.0 and wp.abs(y_rel) <= half_width_y and wp.abs(z_rel) <= half_width_z
    old_damage = damage[tid]
    new_damage = old_damage

    if active:
        delta_damage = max_damage_rate * dt * weight * (1.0 - old_damage)
        new_damage = wp.min(1.0, old_damage + delta_damage)
        damage[tid] = new_damage

        side = wp.where(y_rel >= 0.0, 1.0, -1.0)
        v = particle_qd[tid]
        particle_qd[tid] = v + wp.vec3(0.0, side * separation_speed * delta_damage, 0.0)

        damage_rate = delta_damage / wp.max(dt, 1.0e-6)
        force = force_scale * (
            yield_stress * particle_area * weight * (1.0 - old_damage)
            + fracture_energy * particle_area / wp.max(process_width, 1.0e-6) * damage_rate
        )
        wp.atomic_add(accum, 0, force)
        wp.atomic_add(accum, 1, 1.0)

    wp.atomic_add(accum, 2, new_damage)
    colors[tid] = wp.vec3(
        0.15 + 0.82 * new_damage,
        0.48 * (1.0 - new_damage) + 0.16 * new_damage,
        0.86 * (1.0 - new_damage) + 0.08 * new_damage,
    )


@wp.kernel
def apply_vbd_knife_cut_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    damage: wp.array[wp.float32],
    colors: wp.array[wp.vec3],
    accum: wp.array[wp.float32],
    knife_x: float,
    center_y: float,
    center_z: float,
    half_width_y: float,
    half_width_z: float,
    process_width: float,
    dt: float,
    particle_area: float,
    fracture_energy: float,
    yield_stress: float,
    max_damage_rate: float,
    separation_speed: float,
    force_scale: float,
):
    tid = wp.tid()
    q = particle_q[tid]
    phi = q[0] - knife_x
    y_rel = q[1] - center_y
    z_rel = q[2] - center_z
    weight = wp.max(0.0, 1.0 - wp.abs(phi) / wp.max(process_width, 1.0e-6))

    active = weight > 0.0 and wp.abs(y_rel) <= half_width_y and wp.abs(z_rel) <= half_width_z
    old_damage = damage[tid]
    new_damage = old_damage

    if active:
        delta_damage = max_damage_rate * dt * weight * (1.0 - old_damage)
        new_damage = wp.min(1.0, old_damage + delta_damage)
        damage[tid] = new_damage

        side = wp.where(y_rel >= 0.0, 1.0, -1.0)
        damage_rate = delta_damage / wp.max(dt, 1.0e-6)
        force = force_scale * (
            yield_stress * particle_area * weight * (1.0 - old_damage)
            + fracture_energy * particle_area / wp.max(process_width, 1.0e-6) * damage_rate
        )
        particle_f[tid] = particle_f[tid] + wp.vec3(0.0, side * force, 0.0)
        particle_qd[tid] = particle_qd[tid] + wp.vec3(0.0, side * separation_speed * delta_damage, 0.0)
        wp.atomic_add(accum, 0, force)
        wp.atomic_add(accum, 1, 1.0)

    wp.atomic_add(accum, 2, new_damage)
    colors[tid] = wp.vec3(
        0.15 + 0.82 * new_damage,
        0.48 * (1.0 - new_damage) + 0.16 * new_damage,
        0.86 * (1.0 - new_damage) + 0.08 * new_damage,
    )


@wp.kernel
def degrade_cut_tets_kernel(
    particle_damage: wp.array[wp.float32],
    tet_indices: wp.array2d[wp.int32],
    tet_materials: wp.array2d[wp.float32],
    base_tet_materials: wp.array2d[wp.float32],
    damage_threshold: float,
    residual_stiffness: float,
):
    tid = wp.tid()
    i = tet_indices[tid, 0]
    j = tet_indices[tid, 1]
    k = tet_indices[tid, 2]
    l = tet_indices[tid, 3]
    mean_damage = 0.25 * (particle_damage[i] + particle_damage[j] + particle_damage[k] + particle_damage[l])
    if mean_damage > damage_threshold:
        softening = residual_stiffness + (1.0 - residual_stiffness) * wp.max(0.0, 1.0 - mean_damage)
        tet_materials[tid, 0] = base_tet_materials[tid, 0] * softening
        tet_materials[tid, 1] = base_tet_materials[tid, 1] * softening
        tet_materials[tid, 2] = base_tet_materials[tid, 2]


def launch_mpm_knife_cut(
    state,
    damage: wp.array,
    colors: wp.array,
    accum: wp.array,
    knife: KnifeProfile,
    material: CutMaterial,
    dt: float,
    particle_volume: float,
    time_value: float,
    device,
):
    accum.zero_()
    wp.launch(
        apply_mpm_knife_cut_kernel,
        dim=state.particle_count,
        inputs=[
            state.particle_q,
            state.particle_qd,
            damage,
            colors,
            accum,
            knife.x_at(time_value),
            knife.center_y,
            knife.center_z,
            knife.half_width_y,
            knife.half_width_z,
            knife.process_width,
            dt,
            max(particle_volume, 1.0e-18) ** (2.0 / 3.0),
            material.fracture_energy,
            material.yield_stress,
            material.max_damage_rate,
            material.separation_speed,
            material.force_scale,
        ],
        device=device,
    )


def launch_vbd_knife_cut(
    state,
    damage: wp.array,
    colors: wp.array,
    accum: wp.array,
    knife: KnifeProfile,
    material: CutMaterial,
    dt: float,
    particle_volume: float,
    time_value: float,
    device,
):
    accum.zero_()
    wp.launch(
        apply_vbd_knife_cut_kernel,
        dim=state.particle_count,
        inputs=[
            state.particle_q,
            state.particle_qd,
            state.particle_f,
            damage,
            colors,
            accum,
            knife.x_at(time_value),
            knife.center_y,
            knife.center_z,
            knife.half_width_y,
            knife.half_width_z,
            knife.process_width,
            dt,
            max(particle_volume, 1.0e-18) ** (2.0 / 3.0),
            material.fracture_energy,
            material.yield_stress,
            material.max_damage_rate,
            material.separation_speed,
            material.force_scale,
        ],
        device=device,
    )


def launch_cut_tet_degradation(
    model,
    damage: wp.array,
    base_tet_materials: wp.array,
    damage_threshold: float = 0.18,
    residual_stiffness: float = 0.08,
):
    if model.tet_count == 0 or model.tet_indices is None or model.tet_materials is None:
        return
    wp.launch(
        degrade_cut_tets_kernel,
        dim=model.tet_count,
        inputs=[
            damage,
            model.tet_indices,
            model.tet_materials,
            base_tet_materials,
            damage_threshold,
            residual_stiffness,
        ],
        device=model.device,
    )


class ForceHistory:
    def __init__(self):
        self.times: list[float] = []
        self.forces: list[float] = []
        self.active_counts: list[float] = []
        self.mean_damage: list[float] = []

    def append_from_accum(self, time_value: float, accum: wp.array, particle_count: int):
        values = accum.numpy()
        self.append_values(
            time_value,
            float(values[0]),
            float(values[1]),
            float(values[2]) / max(float(particle_count), 1.0),
        )

    def append_values(self, time_value: float, force: float, active_count: float, mean_damage: float):
        self.times.append(float(time_value))
        self.forces.append(float(force))
        self.active_counts.append(float(active_count))
        self.mean_damage.append(float(mean_damage))

    def summary(self) -> dict[str, float]:
        return summarize_force_profile(np.array(self.times), np.array(self.forces), np.array(self.mean_damage))

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "time_s": self.times,
            "force_n": self.forces,
            "active_particles": self.active_counts,
            "mean_damage": self.mean_damage,
        }

    def write_csv(self, path: str | Path):
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            f.write("time_s,force_n,active_particles,mean_damage\n")
            for row in zip(self.times, self.forces, self.active_counts, self.mean_damage, strict=True):
                f.write(f"{row[0]:.8f},{row[1]:.8f},{row[2]:.0f},{row[3]:.8f}\n")


class StepTimer:
    def __init__(self):
        self.step_times: list[float] = []
        self.render_times: list[float] = []
        self._start = time.perf_counter()

    def time_step(self, fn):
        start = time.perf_counter()
        result = fn()
        self.step_times.append(time.perf_counter() - start)
        return result

    def time_render(self, fn):
        start = time.perf_counter()
        result = fn()
        self.render_times.append(time.perf_counter() - start)
        return result

    @property
    def wall_seconds(self) -> float:
        return time.perf_counter() - self._start

    def build_stats(
        self, solver: str, frame_count: int, sim_seconds: float, force_history: ForceHistory
    ) -> RuntimeStats:
        summary = force_history.summary()
        wall = self.wall_seconds
        return RuntimeStats(
            solver=solver,
            frame_count=frame_count,
            sim_seconds=float(sim_seconds),
            wall_seconds=float(wall),
            mean_step_ms=float(1.0e3 * np.mean(self.step_times)) if self.step_times else 0.0,
            mean_render_ms=float(1.0e3 * np.mean(self.render_times)) if self.render_times else 0.0,
            fps=float(frame_count / wall) if wall > 0.0 else 0.0,
            peak_force_n=summary["peak_force_n"],
            mean_force_n=summary["mean_force_n"],
            force_impulse_ns=summary["force_impulse_ns"],
            final_mean_damage=summary["final_mean_damage"],
            hardware=collect_hardware_details(),
        )


def collect_hardware_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "warp_version": getattr(wp, "__version__", "unknown"),
    }
    try:
        device = wp.get_device()
        details["warp_device"] = str(device)
        details["is_cuda"] = bool(device.is_cuda)
        if device.is_cuda:
            details["cuda_arch"] = getattr(device, "arch", None)
            details["device_name"] = getattr(device, "name", str(device))
            details["total_memory_bytes"] = int(getattr(device, "total_memory", 0))
    except Exception as exc:  # pragma: no cover - diagnostic only
        details["warp_device_error"] = str(exc)

    if shutil.which("nvidia-smi"):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            details["nvidia_smi"] = output.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            details["nvidia_smi_error"] = str(exc)
    return details


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_force_plot(path: str | Path, history: ForceHistory, title: str):
    path = Path(path)
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return None

    fig, ax_force = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax_damage = ax_force.twinx()
    ax_force.plot(history.times, history.forces, color="#b91c1c", linewidth=2.0, label="knife force")
    ax_damage.plot(history.times, history.mean_damage, color="#1d4ed8", linewidth=1.8, label="mean damage")
    ax_force.set_xlabel("time [s]")
    ax_force.set_ylabel("force [N]", color="#b91c1c")
    ax_damage.set_ylabel("mean damage", color="#1d4ed8")
    ax_force.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax_force.set_title(title)
    lines = ax_force.get_lines() + ax_damage.get_lines()
    ax_force.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_json(path: str | Path, payload: Any):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def encode_mp4(frames: list[np.ndarray], path: str | Path, fps: float = 30.0) -> Path | None:
    """Encode RGB frames to MP4 if an optional encoder is available."""

    path = Path(path)
    if not frames:
        return None

    try:
        import imageio.v3 as iio  # noqa: PLC0415

        iio.imwrite(path, np.asarray(frames), fps=fps, codec="libx264", macro_block_size=16)
        return path
    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            return None
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(tmp_path / f"frame_{i:05d}.png")
        subprocess.check_call(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "frame_%05d.png"),
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                str(path),
            ]
        )
    return path


def save_first_frame(frames: list[np.ndarray], path: str | Path) -> Path | None:
    if not frames:
        return None
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None
    path = Path(path)
    Image.fromarray(frames[0]).save(path)
    return path


def capture_viewer_frame(viewer) -> np.ndarray | None:
    if not hasattr(viewer, "get_frame"):
        return None
    image = viewer.get_frame(render_ui=True)
    if image is None:
        return None
    frame = image.numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def export_artifacts(
    output_dir: str | Path,
    solver_name: str,
    frames: list[np.ndarray],
    history: ForceHistory,
    stats: RuntimeStats,
    fps: float,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    artifacts: dict[str, str] = {}
    video_path = output_dir / f"{solver_name}_cutting.mp4"
    if encode_mp4(frames, video_path, fps=fps) is not None:
        artifacts["video"] = str(video_path)
    first_frame = save_first_frame(frames, output_dir / f"{solver_name}_first_frame.png")
    if first_frame is not None:
        artifacts["first_frame"] = str(first_frame)

    plot_path = write_force_plot(
        output_dir / f"{solver_name}_force_profile.png", history, f"{solver_name.upper()} knife cut"
    )
    if plot_path is not None:
        artifacts["force_plot"] = str(plot_path)

    csv_path = output_dir / f"{solver_name}_force_profile.csv"
    history.write_csv(csv_path)
    artifacts["force_csv"] = str(csv_path)

    stats_path = output_dir / f"{solver_name}_runtime_stats.json"
    stats_path.write_text(stats.to_json() + "\n", encoding="utf-8")
    artifacts["runtime_stats"] = str(stats_path)

    write_json(output_dir / f"{solver_name}_force_history.json", history.to_dict())
    return artifacts


def export_remesh_artifacts(output_dir: str | Path, solver_name: str, history: list[dict[str, float]]) -> Path | None:
    if not history:
        return None
    output_dir = ensure_dir(output_dir)
    path = output_dir / f"{solver_name}_adaptive_remesh_stats.json"
    write_json(path, {"summary": summarize_remesh_history(history), "frames": history})
    return path


def add_cutting_artifact_args(parser):
    parser.add_argument(
        "--artifact-dir", type=str, default=None, help="Directory for MP4, force plot, and stats output."
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Capture ViewerGL.get_frame() frames and encode MP4."
    )
    parser.add_argument("--record-fps", type=float, default=30.0, help="Output video frame rate.")
    return parser


def run_cutting_example(example, args, solver_name: str):
    viewer = example.viewer
    frames: list[np.ndarray] = []
    timer = StepTimer()
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    frame_count = int(args.num_frames)
    for _ in range(frame_count):
        if not viewer.is_running():
            break
        if viewer.should_step():
            timer.time_step(example.step)
        timer.time_render(example.render)
        if args.record_video:
            frame = capture_viewer_frame(viewer)
            if frame is not None:
                frames.append(frame)

    if args.test and hasattr(example, "test_final"):
        example.test_final()

    stats = timer.build_stats(solver_name, len(timer.step_times), example.sim_time, example.force_history)
    artifacts = {}
    if args.artifact_dir:
        artifacts = export_artifacts(
            args.artifact_dir, solver_name, frames, example.force_history, stats, args.record_fps
        )
        if hasattr(example, "remesh_history"):
            remesh_path = export_remesh_artifacts(args.artifact_dir, solver_name, example.remesh_history)
            if remesh_path is not None:
                artifacts["adaptive_remesh_stats"] = str(remesh_path)
        print(json.dumps({"artifacts": artifacts, "stats": asdict(stats)}, indent=2, sort_keys=True))

    viewer.close()
    return artifacts, stats


def scalar_from_accum(accum: wp.array, index: int) -> float:
    return float(accum.numpy()[index])


def estimate_particle_volume_from_grid(extents: tuple[float, float, float], particle_count: int) -> float:
    return math.prod(extents) / max(float(particle_count), 1.0)
