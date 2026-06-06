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

MAX_KNIFE_EDGE_POINTS = 32


@dataclass(frozen=True)
class KnifeProfile:
    """Rigid knife with a blade-edge spline moving through a soft block.

    The knife advances along x. The cutting geometry is a piecewise-linear edge
    spline in the knife-local frame, with local points offset from
    ``(x_at(t), center_y, center_z)``. The default is a straight edge spanning
    ``half_width_z``. Forces and damage use a compact SDF-like process zone
    around this rigid edge rather than an invisible front plane.
    """

    start_x: float = -0.08
    speed: float = 0.35
    center_y: float = 0.0
    center_z: float = 0.0
    half_width_y: float = 0.06
    half_width_z: float = 0.34
    process_width: float = 0.045
    edge_control_points: tuple[tuple[float, float, float], ...] = ()
    blade_spine_depth: float = 0.18
    handle_length: float = 0.20
    handle_height: float = 0.075
    cut_path_amplitude_y: float = 0.0
    cut_path_wavelength_x: float = 1.0
    cut_path_phase: float = 0.0
    cut_path_origin_x: float = 0.0

    def x_at(self, time: float) -> float:
        return self.start_x + self.speed * time

    def center_y_at_x(self, x: float | np.ndarray) -> float | np.ndarray:
        if abs(float(self.cut_path_amplitude_y)) <= 1.0e-12:
            if np.isscalar(x):
                return float(self.center_y)
            return np.full_like(np.asarray(x, dtype=np.float32), float(self.center_y), dtype=np.float32)
        x_np = np.asarray(x, dtype=np.float32)
        wavelength = max(abs(float(self.cut_path_wavelength_x)), 1.0e-6)
        phase = 2.0 * np.pi * (x_np - float(self.cut_path_origin_x)) / wavelength + float(self.cut_path_phase)
        values = float(self.center_y) + float(self.cut_path_amplitude_y) * np.sin(phase)
        if np.isscalar(x):
            return float(values)
        return values.astype(np.float32, copy=False)

    def path_slope_at_x(self, x: float | np.ndarray) -> float | np.ndarray:
        if abs(float(self.cut_path_amplitude_y)) <= 1.0e-12:
            if np.isscalar(x):
                return 0.0
            return np.zeros_like(np.asarray(x, dtype=np.float32), dtype=np.float32)
        x_np = np.asarray(x, dtype=np.float32)
        wavelength = max(abs(float(self.cut_path_wavelength_x)), 1.0e-6)
        phase = 2.0 * np.pi * (x_np - float(self.cut_path_origin_x)) / wavelength + float(self.cut_path_phase)
        values = float(self.cut_path_amplitude_y) * (2.0 * np.pi / wavelength) * np.cos(phase)
        if np.isscalar(x):
            return float(values)
        return values.astype(np.float32, copy=False)

    def path_tangent_at_x(self, x: float | np.ndarray) -> np.ndarray:
        slope = np.asarray(self.path_slope_at_x(x), dtype=np.float32)
        tangent = np.stack(
            [np.ones_like(slope, dtype=np.float32), slope, np.zeros_like(slope, dtype=np.float32)],
            axis=-1,
        )
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1.0e-8)
        if np.isscalar(x):
            return tangent.reshape(3).astype(np.float32, copy=False)
        return tangent.astype(np.float32, copy=False)

    def path_normal_at_x(self, x: float | np.ndarray) -> np.ndarray:
        tangent = self.path_tangent_at_x(x)
        normal = np.stack([-tangent[..., 1], tangent[..., 0], np.zeros_like(tangent[..., 0])], axis=-1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-8)
        return normal.astype(np.float32, copy=False)

    def signed_cut_y(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        slope = self.path_slope_at_x(points[..., 0])
        return (points[..., 1] - self.center_y_at_x(points[..., 0])) / np.sqrt(1.0 + np.asarray(slope) ** 2)

    def signed_distance_x(self, points: np.ndarray, time: float) -> np.ndarray:
        points = np.asarray(points)
        return points[..., 0] - self.x_at(time)

    def _local_edge_points(self) -> np.ndarray:
        if self.edge_control_points:
            points = np.asarray(self.edge_control_points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
                raise ValueError("edge_control_points must have shape (N, 3) with N >= 2")
            return points[:MAX_KNIFE_EDGE_POINTS].copy()
        return np.array(
            [
                [0.0, 0.0, -self.half_width_z],
                [0.0, 0.0, self.half_width_z],
            ],
            dtype=np.float32,
        )

    def edge_points(
        self,
        time: float,
        *,
        front_x: float | None = None,
        center_y: float | None = None,
        center_z: float | None = None,
    ) -> np.ndarray:
        """Return world-space points of the rigid blade-edge polyline."""

        origin = np.array(
            [
                self.x_at(time) if front_x is None else float(front_x),
                self.center_y if center_y is None else float(center_y),
                self.center_z if center_z is None else float(center_z),
            ],
            dtype=np.float32,
        )
        return self._local_edge_points() + origin

    def edge_distances(
        self,
        points: np.ndarray,
        time: float,
        *,
        front_x: float | None = None,
        center_y: float | None = None,
        center_z: float | None = None,
    ) -> np.ndarray:
        """Distance from points to the blade process zone around the edge spline."""

        points = np.asarray(points, dtype=np.float32)
        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        edge = self.edge_points(time, front_x=front_x, center_y=center_y, center_z=center_z)
        best_d2 = np.full(flat_points.shape[0], np.inf, dtype=np.float32)

        pxz = flat_points[:, [0, 2]]
        for i in range(edge.shape[0] - 1):
            a = edge[i]
            b = edge[i + 1]
            axz = a[[0, 2]]
            bxz = b[[0, 2]]
            ab = bxz - axz
            denom = float(np.dot(ab, ab))
            if denom <= 1.0e-12:
                closest = np.broadcast_to(axz, pxz.shape)
            else:
                t = np.clip(np.sum((pxz - axz) * ab, axis=1) / denom, 0.0, 1.0)
                closest = axz + t[:, None] * ab
            d = pxz - closest
            best_d2 = np.minimum(best_d2, np.sum(d * d, axis=1))

        if center_y is None:
            signed_distance = self.signed_cut_y(flat_points)
        else:
            path = KnifeProfile(
                start_x=self.start_x,
                speed=self.speed,
                center_y=float(center_y),
                center_z=self.center_z,
                half_width_y=self.half_width_y,
                half_width_z=self.half_width_z,
                process_width=self.process_width,
                edge_control_points=self.edge_control_points,
                blade_spine_depth=self.blade_spine_depth,
                handle_length=self.handle_length,
                handle_height=self.handle_height,
                cut_path_amplitude_y=self.cut_path_amplitude_y,
                cut_path_wavelength_x=self.cut_path_wavelength_x,
                cut_path_phase=self.cut_path_phase,
                cut_path_origin_x=self.cut_path_origin_x,
            )
            signed_distance = path.signed_cut_y(flat_points)
        y_out = np.maximum(np.abs(signed_distance) - self.half_width_y, 0.0)
        distance = np.sqrt(best_d2 + y_out * y_out)
        return distance.reshape(original_shape)

    def cut_weights(self, points: np.ndarray, time: float) -> np.ndarray:
        distance = self.edge_distances(points, time)
        return np.clip(1.0 - distance / max(self.process_width, 1.0e-12), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _box_mesh(
        center: np.ndarray,
        half_extent: np.ndarray,
        vertex_offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        hx, hy, hz = half_extent
        cx, cy, cz = center
        vertices = np.array(
            [
                [cx - hx, cy - hy, cz - hz],
                [cx + hx, cy - hy, cz - hz],
                [cx + hx, cy + hy, cz - hz],
                [cx - hx, cy + hy, cz - hz],
                [cx - hx, cy - hy, cz + hz],
                [cx + hx, cy - hy, cz + hz],
                [cx + hx, cy + hy, cz + hz],
                [cx - hx, cy + hy, cz + hz],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int32,
        )
        return vertices, indices + int(vertex_offset)

    def blade_mesh(
        self,
        time: float,
        *,
        front_x: float | None = None,
        center_y: float | None = None,
        center_z: float | None = None,
        include_handle: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a triangle mesh for the rigid knife blade and optional handle."""

        edge = self.edge_points(time, front_x=front_x, center_y=center_y, center_z=center_z)
        x_front = self.x_at(time) if front_x is None else float(front_x)
        tangent = self.path_tangent_at_x(x_front)
        normal = self.path_normal_at_x(x_front)
        spine_depth = max(float(self.blade_spine_depth), 1.0e-4)
        half_y = max(float(self.half_width_y), 1.0e-4)
        left_spine = edge - spine_depth * tangent - half_y * normal
        right_spine = edge - spine_depth * tangent + half_y * normal

        vertices = np.vstack([edge, left_spine, right_spine]).astype(np.float32, copy=False)
        n = edge.shape[0]
        triangles: list[list[int]] = []
        for i in range(n - 1):
            e0 = i
            e1 = i + 1
            l0 = n + i
            l1 = n + i + 1
            r0 = 2 * n + i
            r1 = 2 * n + i + 1
            triangles.extend(
                [
                    [e0, l0, l1],
                    [e0, l1, e1],
                    [e0, e1, r1],
                    [e0, r1, r0],
                    [l0, r0, r1],
                    [l0, r1, l1],
                ]
            )

        triangles.extend([[0, 2 * n, n], [n - 1, n + n - 1, 3 * n - 1]])
        indices = np.asarray(triangles, dtype=np.int32)

        if include_handle:
            y_center = self.center_y if center_y is None else float(center_y)
            z_top = float(np.max(edge[:, 2]))
            handle_length = max(float(self.handle_length), 1.0e-4)
            handle_center = np.array(
                [
                    x_front - spine_depth - 0.5 * handle_length,
                    y_center,
                    z_top + 0.52 * max(float(self.handle_height), 1.0e-4),
                ],
                dtype=np.float32,
            )
            handle_half_extent = np.array(
                [
                    0.5 * handle_length,
                    max(1.45 * half_y, 0.018),
                    0.5 * max(float(self.handle_height), 1.0e-4),
                ],
                dtype=np.float32,
            )
            handle_vertices, handle_indices = self._box_mesh(handle_center, handle_half_extent, vertices.shape[0])
            vertices = np.vstack([vertices, handle_vertices])
            indices = np.vstack([indices, handle_indices])

        return vertices.astype(np.float32, copy=False), indices.astype(np.int32, copy=False)

    def blade_segments(self, time: float, tail: float = 0.16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return line segments that draw the rigid blade outline and edge."""

        edge = self.edge_points(time)
        tangent = self.path_tangent_at_x(self.x_at(time))
        normal = self.path_normal_at_x(self.x_at(time))
        left_spine = edge - float(tail) * tangent - float(self.half_width_y) * normal
        right_spine = edge - float(tail) * tangent + float(self.half_width_y) * normal
        starts = []
        ends = []
        for i in range(edge.shape[0] - 1):
            starts.extend([edge[i], left_spine[i], right_spine[i]])
            ends.extend([edge[i + 1], left_spine[i + 1], right_spine[i + 1]])
        starts.extend([edge[0], edge[0], edge[-1], edge[-1], left_spine[0], left_spine[-1]])
        ends.extend([left_spine[0], right_spine[0], left_spine[-1], right_spine[-1], right_spine[0], right_spine[-1]])
        starts = np.asarray(starts, dtype=np.float32)
        ends = np.asarray(ends, dtype=np.float32)
        colors = np.tile(np.array([[0.95, 0.95, 0.98]], dtype=np.float32), (starts.shape[0], 1))
        return starts, ends, colors


def log_knife_mesh(
    viewer,
    device,
    knife: KnifeProfile,
    time: float,
    prefix: str = "/cutting/knife",
    blade_color: tuple[float, float, float] = (0.56, 0.61, 0.68),
    edge_color: tuple[float, float, float] = (0.06, 0.07, 0.09),
):
    """Log the rigid knife as a shaded mesh plus a dark cutting edge."""

    vertices, indices = knife.blade_mesh(time)
    edge = knife.edge_points(time)
    viewer.log_mesh(
        f"{prefix}/rigid_blade",
        wp.array(vertices, dtype=wp.vec3, device=device),
        wp.array(indices.reshape(-1), dtype=wp.int32, device=device),
        hidden=False,
        backface_culling=False,
        color=blade_color,
        roughness=0.36,
        opacity=1.0,
    )
    if edge.shape[0] > 1:
        starts = edge[:-1]
        ends = edge[1:]
        colors = np.tile(np.asarray(edge_color, dtype=np.float32), (starts.shape[0], 1))
        viewer.log_lines(
            f"{prefix}/edge",
            wp.array(starts, dtype=wp.vec3, device=device),
            wp.array(ends, dtype=wp.vec3, device=device),
            wp.array(colors, dtype=wp.vec3, device=device),
            width=0.022,
        )


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


@dataclass(frozen=True)
class TetCutWallRenderStats:
    active_x_segments: int
    surface_vertex_count: int
    surface_triangle_count: int
    wall_vertex_count: int
    wall_triangle_count: int
    coarse_dx: float
    min_active_dx: float
    max_active_dx: float
    active_tet_count: int


@wp.func
def _cutting_smoothstep(value: float):
    x = wp.min(1.0, wp.max(0.0, value))
    return x * x * (3.0 - 2.0 * x)


@wp.func
def _cutting_gap_at(x: float, knife_x: float, max_gap: float, front_width: float):
    return max_gap * _cutting_smoothstep((knife_x - x) / front_width)


def _cutting_visual_opening_y(
    x: float, side: float, knife_x: float, center_y: float, max_gap: float, front_width: float
):
    return center_y + side * _cutting_gap_at(x, knife_x, max_gap, front_width)


@wp.func
def _cutting_visual_opening_y_wp(
    x: float, side: float, knife_x: float, center_y: float, max_gap: float, front_width: float
):
    return center_y + side * max_gap * _cutting_smoothstep((knife_x - x) / front_width)


@wp.func
def _cutting_deform_from_particles(
    rest_point: wp.vec3,
    rest_particle_points: wp.array[wp.vec3],
    particle_points: wp.array[wp.vec3],
    particle_count: int,
    side_center_y: float,
    side_hint: float,
):
    if particle_count <= 0:
        return rest_point

    weighted_delta = wp.vec3(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    rest_side = side_hint
    if rest_side == 0.0:
        rest_side = -1.0
        if rest_point[1] >= side_center_y:
            rest_side = 1.0
    for i in range(particle_count):
        rest_particle = rest_particle_points[i]
        current_particle = particle_points[i]
        d = rest_point - rest_particle
        d2 = wp.dot(d, d)
        weight = 1.0 / wp.max(d2, 1.0e-10)
        particle_side = -1.0
        if rest_particle[1] >= side_center_y:
            particle_side = 1.0
        if rest_side * particle_side < 0.0:
            weight = 0.0
        weighted_delta = weighted_delta + (current_particle - rest_particle) * weight
        weight_sum = weight_sum + weight

    return rest_point + weighted_delta / wp.max(weight_sum, 1.0e-10)


@wp.func
def _cutting_write_quad(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    quad_index: int,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    rest_particle_points: wp.array[wp.vec3],
    particle_points: wp.array[wp.vec3],
    particle_count: int,
    side_center_y: float,
    side_hint: float,
):
    vertex = quad_index * 4
    index = quad_index * 6
    points[vertex + 0] = _cutting_deform_from_particles(
        a, rest_particle_points, particle_points, particle_count, side_center_y, side_hint
    )
    points[vertex + 1] = _cutting_deform_from_particles(
        b, rest_particle_points, particle_points, particle_count, side_center_y, side_hint
    )
    points[vertex + 2] = _cutting_deform_from_particles(
        c, rest_particle_points, particle_points, particle_count, side_center_y, side_hint
    )
    points[vertex + 3] = _cutting_deform_from_particles(
        d, rest_particle_points, particle_points, particle_count, side_center_y, side_hint
    )
    indices[index + 0] = vertex + 0
    indices[index + 1] = vertex + 1
    indices[index + 2] = vertex + 2
    indices[index + 3] = vertex + 0
    indices[index + 4] = vertex + 2
    indices[index + 5] = vertex + 3


@wp.func
def _cutting_write_empty_quad(points: wp.array[wp.vec3], indices: wp.array[wp.int32], quad_index: int):
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
    x_segments: wp.array[wp.vec2],
    active_segment_count: int,
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    rest_particle_points: wp.array[wp.vec3],
    particle_points: wp.array[wp.vec3],
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
        y_cut0 = _cutting_visual_opening_y_wp(x0, side, knife_x, knife_center_y, max_gap, front_width)
        y_cut1 = _cutting_visual_opening_y_wp(x1, side, knife_x, knife_center_y, max_gap, front_width)

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
                knife_center_y,
                side,
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
                knife_center_y,
                side,
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
                knife_center_y,
                side,
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
    y_cut = _cutting_visual_opening_y_wp(x, side, knife_x, knife_center_y, max_gap, front_width)
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
            knife_center_y,
            side,
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
            knife_center_y,
            side,
        )


@wp.kernel
def _build_adaptive_cut_wall_kernel(
    x_segments: wp.array[wp.vec2],
    wall_segment_count: int,
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    rest_particle_points: wp.array[wp.vec3],
    particle_points: wp.array[wp.vec3],
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
    if segment_id >= wall_segment_count:
        _cutting_write_empty_quad(points, indices, quad)
        return

    side_id = local / height_segments
    z_id = local - side_id * height_segments
    side = -1.0
    if side_id == 1:
        side = 1.0

    segment = x_segments[segment_id]
    x0 = segment[0]
    x1 = wp.min(segment[1], knife_x)
    if x1 <= x0:
        _cutting_write_empty_quad(points, indices, quad)
        return
    dz = (block_hi[2] - block_lo[2]) / float(height_segments)
    z0 = block_lo[2] + float(z_id) * dz
    z1 = z0 + dz
    y_cut0 = _cutting_visual_opening_y_wp(x0, side, knife_x, knife_center_y, max_gap, front_width)
    y_cut1 = _cutting_visual_opening_y_wp(x1, side, knife_x, knife_center_y, max_gap, front_width)

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
        knife_center_y,
        side,
    )


class SplitCuboidRenderMesh:
    """Render-only cuboid remesh with duplicated seam vertices and cut walls.

    This is deliberately a visualization layer. It keeps a fixed topology so it
    can be updated every frame, while the vertices around the knife path are
    duplicated at zero kerf to expose internal cut faces without shrinking the
    exterior volume.
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
    def _append_quad(vertices: list[list[float]], side_hints: list[float], side_hint: float, a, b, c, d):
        vertices.extend([list(a), list(b), list(c), list(d)])
        side_hints.extend([float(side_hint)] * 4)

    @staticmethod
    def _append_empty_quad(vertices: list[list[float]], side_hints: list[float]):
        vertices.extend([[0.0, 0.0, 0.0]] * 4)
        side_hints.extend([0.0] * 4)

    def gap_at(self, x: float, time: float) -> float:
        return _cutting_gap_at(x, self.knife.x_at(time), self.max_gap, self.front_width)

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
        surface_side_hints: list[float] = []
        walls: list[list[float]] = []
        wall_side_hints: list[float] = []
        knife_x = self.knife.x_at(time)

        for side in (-1.0, 1.0):
            y_outer = float(lo[1] if side < 0.0 else hi[1])
            for i in range(self.segments):
                x0 = float(self.x_values[i])
                x1 = float(self.x_values[i + 1])
                y_cut0 = self.knife.center_y + side * self.gap_at(x0, time)
                y_cut1 = self.knife.center_y + side * self.gap_at(x1, time)

                self._append_quad(
                    surface,
                    surface_side_hints,
                    side,
                    (x0, y_outer, z1),
                    (x1, y_outer, z1),
                    (x1, y_cut1, z1),
                    (x0, y_cut0, z1),
                )
                self._append_quad(
                    surface,
                    surface_side_hints,
                    side,
                    (x0, y_cut0, z0),
                    (x1, y_cut1, z0),
                    (x1, y_outer, z0),
                    (x0, y_outer, z0),
                )
                self._append_quad(
                    surface,
                    surface_side_hints,
                    side,
                    (x0, y_outer, z0),
                    (x1, y_outer, z0),
                    (x1, y_outer, z1),
                    (x0, y_outer, z1),
                )
                if x0 < knife_x:
                    wall_x1 = min(x1, knife_x)
                    self._append_quad(
                        walls,
                        wall_side_hints,
                        side,
                        (x0, y_cut0, z0),
                        (wall_x1, y_cut1, z0),
                        (wall_x1, y_cut1, z1),
                        (x0, y_cut0, z1),
                    )
                else:
                    self._append_empty_quad(walls, wall_side_hints)

                if i == 0:
                    self._append_quad(
                        surface,
                        surface_side_hints,
                        side,
                        (x0, y_outer, z0),
                        (x0, y_cut0, z0),
                        (x0, y_cut0, z1),
                        (x0, y_outer, z1),
                    )
                if i == self.segments - 1:
                    self._append_quad(
                        surface,
                        surface_side_hints,
                        side,
                        (x1, y_cut1, z0),
                        (x1, y_outer, z0),
                        (x1, y_outer, z1),
                        (x1, y_cut1, z1),
                    )

        surface_np = np.asarray(surface, dtype=np.float32)
        walls_np = np.asarray(walls, dtype=np.float32)
        if rest_particle_points is not None or particle_points is not None:
            surface_np = self._deform_points_with_particles(
                surface_np, rest_particle_points, particle_points, np.asarray(surface_side_hints, dtype=np.float32)
            )
            walls_np = self._deform_points_with_particles(
                walls_np, rest_particle_points, particle_points, np.asarray(wall_side_hints, dtype=np.float32)
            )
        return surface_np, walls_np

    def _deform_points_with_particles(
        self,
        points: np.ndarray,
        rest_particle_points: np.ndarray | None,
        particle_points: np.ndarray | None,
        side_hints: np.ndarray | None = None,
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
        if side_hints is not None:
            rest_sides = np.where(rest[:, 1] >= self.knife.center_y, 1.0, -1.0).astype(np.float32)
            point_sides = np.asarray(side_hints, dtype=np.float32).reshape(-1)
            point_sides = np.where(
                point_sides == 0.0,
                np.where(points[:, 1] >= self.knife.center_y, 1.0, -1.0),
                point_sides,
            )
            if k == rest.shape[0]:
                nearest_sides = np.broadcast_to(rest_sides[None, :], (points.shape[0], rest.shape[0]))
            else:
                nearest_sides = rest_sides[nearest]
            weights = np.where(nearest_sides * point_sides[:, None] < 0.0, 0.0, weights)
        weight_sum = np.maximum(np.sum(weights, axis=1), 1.0e-12)
        sampled_delta = np.sum(nearest_deltas * weights[:, :, None], axis=1) / weight_sum[:, None]
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
        surface_opacity: float = 0.72,
        wall_opacity: float = 1.0,
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
            opacity=surface_opacity,
        )
        viewer.log_mesh(
            f"{prefix}/cut_walls",
            self.wall_points_wp,
            self.wall_indices_wp,
            hidden=self.knife.x_at(time) < self.block_lo[0],
            backface_culling=False,
            color=wall_color,
            roughness=0.82,
            opacity=wall_opacity,
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

    def _build_x_segments(self, time: float) -> tuple[int, int, float, float]:
        knife_x = self.knife.x_at(time)
        coarse_nodes = np.linspace(self.block_lo[0], self.block_hi[0], self.base_segments + 1, dtype=np.float32)
        count = 0
        wall_count = 0
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
                if a < knife_x:
                    wall_count += 1
                count += 1

        if count == 0:
            min_dx = self.coarse_dx
            max_dx = self.coarse_dx
        return count, wall_count, float(min_dx), float(max_dx)

    def update(
        self,
        device,
        time: float,
        rest_particle_points: np.ndarray | wp.array | None = None,
        particle_points: np.ndarray | wp.array | None = None,
    ) -> AdaptiveRemeshStats:
        self._ensure_device_arrays(device)
        active_x_segments, wall_x_segments, min_dx, max_dx = self._build_x_segments(time)
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
                wall_x_segments,
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
        active_wall_quads = wall_x_segments * 2 * self.height_segments
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
        surface_opacity: float = 0.72,
        wall_opacity: float = 1.0,
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
            opacity=surface_opacity,
        )
        viewer.log_mesh(
            f"{prefix}/cut_walls",
            self.wall_points_wp,
            self.wall_indices_wp,
            hidden=self.knife.x_at(time) < self.block_lo[0],
            backface_culling=False,
            color=wall_color,
            roughness=0.82,
            opacity=wall_opacity,
        )
        return stats


class TetMeshCutSurfaceRenderer:
    """Render actual soft-mesh surface triangles plus zero-kerf internal cut faces.

    The exterior mesh uses the Newton model's surface triangles directly, so it
    follows particle motion exactly and never deletes volume. The internal wall
    is generated by intersecting the rest tetrahedra with the material cut plane
    behind the knife front, then interpolating those intersection vertices from
    the current particle positions.
    """

    _TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    def __init__(
        self,
        rest_points: np.ndarray,
        tet_indices: np.ndarray,
        surface_indices: np.ndarray,
        knife: KnifeProfile,
        nominal_edge_length: float,
        max_wall_triangles: int | None = None,
        max_visual_gap: float = 0.04,
        front_width: float | None = None,
    ):
        self.rest_points = np.asarray(rest_points, dtype=np.float32)
        self.tet_indices = np.asarray(tet_indices, dtype=np.int32).reshape(-1, 4)
        self.base_surface_triangles_np = np.asarray(surface_indices, dtype=np.int32).reshape(-1, 3)
        self.knife = knife
        self.nominal_edge_length = float(max(nominal_edge_length, 1.0e-8))
        self.max_visual_gap = float(max_visual_gap)
        self.front_width = float(front_width if front_width is not None else max(2.0 * knife.process_width, 1.0e-4))
        self.max_surface_triangles = int(max(1, 3 * len(self.base_surface_triangles_np)))
        self.max_wall_triangles = int(max_wall_triangles or max(1, 4 * len(self.tet_indices)))

        self.surface_points_np = np.zeros((self.max_surface_triangles * 3, 3), dtype=np.float32)
        self.surface_indices_np = np.arange(self.max_surface_triangles * 3, dtype=np.int32)
        self.wall_points_np = np.zeros((self.max_wall_triangles * 3, 3), dtype=np.float32)
        self.wall_indices_np = np.arange(self.max_wall_triangles * 3, dtype=np.int32)
        self.surface_points_wp: wp.array | None = None
        self.surface_indices_wp: wp.array | None = None
        self.wall_points_wp: wp.array | None = None
        self.wall_indices_wp: wp.array | None = None
        self.device_key: str | None = None
        self.last_stats = TetCutWallRenderStats(
            active_x_segments=0,
            surface_vertex_count=0,
            surface_triangle_count=0,
            wall_vertex_count=0,
            wall_triangle_count=0,
            coarse_dx=self.nominal_edge_length,
            min_active_dx=self.nominal_edge_length,
            max_active_dx=self.nominal_edge_length,
            active_tet_count=0,
        )

    def _ensure_device_arrays(self, device):
        device_key = str(device)
        if self.wall_points_wp is not None and self.device_key == device_key:
            return
        self.device_key = device_key
        self.surface_points_wp = wp.array(self.surface_points_np, dtype=wp.vec3, device=device)
        self.surface_indices_wp = wp.array(self.surface_indices_np, dtype=wp.int32, device=device)
        self.wall_points_wp = wp.array(self.wall_points_np, dtype=wp.vec3, device=device)
        self.wall_indices_wp = wp.array(self.wall_indices_np, dtype=wp.int32, device=device)

    @staticmethod
    def _current_points_np(points: np.ndarray | wp.array) -> np.ndarray:
        if isinstance(points, wp.array):
            return points.numpy().astype(np.float32, copy=False)
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _append_unique(
        rest_points: list[np.ndarray],
        current_points: list[np.ndarray],
        rest_point: np.ndarray,
        current_point: np.ndarray,
        tol: float,
    ):
        for existing in rest_points:
            if float(np.linalg.norm(existing - rest_point)) <= tol:
                return
        rest_points.append(rest_point.astype(np.float32, copy=False))
        current_points.append(current_point.astype(np.float32, copy=False))

    @classmethod
    def _tet_plane_polygon(
        cls,
        rest_tet: np.ndarray,
        render_tet: np.ndarray,
        plane_y: float,
        keep_positive: bool,
        eps: float = 1.0e-7,
    ) -> np.ndarray | None:
        signed = rest_tet[:, 1] - plane_y
        if np.all(signed > eps) or np.all(signed < -eps):
            return None

        rest_hits: list[np.ndarray] = []
        current_hits: list[np.ndarray] = []
        for i, j in cls._TET_EDGES:
            si = float(signed[i])
            sj = float(signed[j])
            if abs(si) <= eps and abs(sj) <= eps:
                cls._append_unique(rest_hits, current_hits, rest_tet[i], render_tet[i], eps)
                cls._append_unique(rest_hits, current_hits, rest_tet[j], render_tet[j], eps)
            elif abs(si) <= eps:
                cls._append_unique(rest_hits, current_hits, rest_tet[i], render_tet[i], eps)
            elif abs(sj) <= eps:
                cls._append_unique(rest_hits, current_hits, rest_tet[j], render_tet[j], eps)
            elif si * sj < 0.0:
                alpha = si / (si - sj)
                rest_point = rest_tet[i] + alpha * (rest_tet[j] - rest_tet[i])
                i_inside = si >= -eps if keep_positive else si <= eps
                side_index = i if i_inside else j
                current_point = rest_point + (render_tet[side_index] - rest_tet[side_index])
                cls._append_unique(rest_hits, current_hits, rest_point, current_point, eps)

        if len(current_hits) < 3:
            return None

        rest_poly = np.asarray(rest_hits, dtype=np.float32)
        current_poly = np.asarray(current_hits, dtype=np.float32)
        center = np.mean(rest_poly[:, [0, 2]], axis=0)
        angles = np.arctan2(rest_poly[:, 2] - center[1], rest_poly[:, 0] - center[0])
        order = np.argsort(angles)
        return current_poly[order]

    def _apply_visual_opening(self, polygon: np.ndarray, side: float, knife_x: float) -> np.ndarray:
        if polygon is None or polygon.size == 0 or self.max_visual_gap <= 0.0:
            return polygon
        opened = np.asarray(polygon, dtype=np.float32).copy()
        for i in range(opened.shape[0]):
            opened_y = _cutting_visual_opening_y(
                float(opened[i, 0]),
                side,
                knife_x,
                float(self.knife.center_y),
                self.max_visual_gap,
                self.front_width,
            )
            if side > 0.0:
                opened[i, 1] = max(float(opened[i, 1]), opened_y)
            else:
                opened[i, 1] = min(float(opened[i, 1]), opened_y)
        return opened

    def _append_surface_triangle(self, triangle_count: int, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
        if triangle_count >= self.max_surface_triangles:
            return triangle_count
        if float(np.linalg.norm(np.cross(b - a, c - a))) <= 1.0e-10:
            return triangle_count
        vertex = triangle_count * 3
        self.surface_points_np[vertex + 0] = a
        self.surface_points_np[vertex + 1] = b
        self.surface_points_np[vertex + 2] = c
        return triangle_count + 1

    def _append_surface_polygon(self, triangle_count: int, polygon: np.ndarray | None) -> int:
        if polygon is None or polygon.shape[0] < 3:
            return triangle_count
        for i in range(1, polygon.shape[0] - 1):
            triangle_count = self._append_surface_triangle(triangle_count, polygon[0], polygon[i], polygon[i + 1])
        return triangle_count

    @staticmethod
    def _clip_triangle_by_side(
        rest_tri: np.ndarray,
        render_tri: np.ndarray,
        plane_y: float,
        keep_positive: bool,
        eps: float = 1.0e-7,
    ) -> np.ndarray | None:
        out_render: list[np.ndarray] = []

        def inside(point: np.ndarray) -> bool:
            signed = float(point[1] - plane_y)
            return signed >= -eps if keep_positive else signed <= eps

        def seam_point(
            prev_rest: np.ndarray,
            prev_render: np.ndarray,
            prev_inside: bool,
            curr_rest: np.ndarray,
            curr_render: np.ndarray,
            curr_inside: bool,
        ) -> np.ndarray:
            s0 = float(prev_rest[1] - plane_y)
            s1 = float(curr_rest[1] - plane_y)
            alpha = 0.0 if abs(s0 - s1) <= eps else s0 / (s0 - s1)
            rest_hit = prev_rest + alpha * (curr_rest - prev_rest)
            if prev_inside and not curr_inside:
                return (rest_hit + (prev_render - prev_rest)).astype(np.float32)
            if curr_inside and not prev_inside:
                return (rest_hit + (curr_render - curr_rest)).astype(np.float32)
            return (prev_render + alpha * (curr_render - prev_render)).astype(np.float32)

        for edge_id in range(3):
            prev_id = (edge_id + 2) % 3
            curr_rest = rest_tri[edge_id]
            curr_render = render_tri[edge_id]
            prev_rest = rest_tri[prev_id]
            prev_render = render_tri[prev_id]
            curr_inside = inside(curr_rest)
            prev_inside = inside(prev_rest)

            if curr_inside != prev_inside:
                out_render.append(seam_point(prev_rest, prev_render, prev_inside, curr_rest, curr_render, curr_inside))
            if curr_inside:
                out_render.append(curr_render.astype(np.float32, copy=False))

        if len(out_render) < 3:
            return None
        return np.asarray(out_render, dtype=np.float32)

    def _surface_triangle_in_cut_wake(
        self,
        rest_tri: np.ndarray,
        knife_x: float,
        z_lo: float,
        z_hi: float,
    ) -> bool:
        if float(np.min(rest_tri[:, 0])) > knife_x:
            return False
        if float(np.max(rest_tri[:, 2])) < z_lo or float(np.min(rest_tri[:, 2])) > z_hi:
            return False
        return True

    def _update_surface_mesh(
        self,
        render_np: np.ndarray,
        hidden_anchor: np.ndarray,
        knife_x: float,
        z_lo: float,
        z_hi: float,
        render_negative_np: np.ndarray | None = None,
        render_positive_np: np.ndarray | None = None,
    ) -> int:
        self.surface_points_np[:, :] = hidden_anchor
        surface_triangles = 0
        plane_y = float(self.knife.center_y)
        eps = 1.0e-7
        negative_np = render_np if render_negative_np is None else render_negative_np
        positive_np = render_np if render_positive_np is None else render_positive_np
        for tri in self.base_surface_triangles_np:
            rest_tri = self.rest_points[tri]
            signed = rest_tri[:, 1] - plane_y
            touches_or_crosses = np.min(signed) <= eps and np.max(signed) >= -eps
            if touches_or_crosses and self._surface_triangle_in_cut_wake(rest_tri, knife_x, z_lo, z_hi):
                negative = self._clip_triangle_by_side(rest_tri, negative_np[tri], plane_y, keep_positive=False)
                positive = self._clip_triangle_by_side(rest_tri, positive_np[tri], plane_y, keep_positive=True)
                surface_triangles = self._append_surface_polygon(
                    surface_triangles, self._apply_visual_opening(negative, -1.0, knife_x)
                )
                surface_triangles = self._append_surface_polygon(
                    surface_triangles, self._apply_visual_opening(positive, 1.0, knife_x)
                )
            else:
                side = 1.0 if float(np.mean(rest_tri[:, 1])) >= plane_y else -1.0
                side_render_np = positive_np if side > 0.0 else negative_np
                render_tri = side_render_np[tri]
                surface_triangles = self._append_surface_triangle(
                    surface_triangles, render_tri[0], render_tri[1], render_tri[2]
                )
        return surface_triangles

    def update(
        self,
        current_points: np.ndarray | wp.array,
        time: float,
        front_x: float | None = None,
        center_z: float | None = None,
        enrichment_points: np.ndarray | wp.array | None = None,
    ) -> TetCutWallRenderStats:
        current_np = self._current_points_np(current_points)
        if current_np.shape != self.rest_points.shape:
            raise ValueError("current_points must match rest point shape")
        if enrichment_points is None:
            render_np = current_np
            render_negative_np = current_np
            render_positive_np = current_np
        else:
            enrichment_np = self._current_points_np(enrichment_points)
            if enrichment_np.shape != self.rest_points.shape:
                raise ValueError("enrichment_points must match rest point shape")
            # X-FEM rendering needs virtual nodes on both sides of the cut.  Do not
            # interpolate one enriched position across a straddling face: that
            # collapses the discontinuity and makes the opened material look like
            # missing/deleted elements.  Instead render the negative and positive
            # clipped pieces with side-specific Heaviside enrichment.
            side_gap = np.abs(enrichment_np[:, 1:2])
            side_gap = np.maximum(side_gap, np.linalg.norm(enrichment_np, axis=1, keepdims=True) * 0.35)
            side_gap = np.minimum(side_gap, self.max_visual_gap)
            y_axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
            render_negative_np = current_np - side_gap * y_axis
            render_positive_np = current_np + side_gap * y_axis
            rest_side = np.where(self.rest_points[:, 1:2] >= float(self.knife.center_y), 1.0, -1.0).astype(np.float32)
            render_np = np.where(rest_side > 0.0, render_positive_np, render_negative_np)

        hidden_anchor = np.mean(current_np, axis=0, dtype=np.float64).astype(np.float32)
        self.surface_points_np[:, :] = hidden_anchor
        self.wall_points_np[:, :] = hidden_anchor
        knife_x = float(self.knife.x_at(time) if front_x is None else front_x)
        knife_center_z = float(self.knife.center_z if center_z is None else center_z)
        z_lo = knife_center_z - self.knife.half_width_z - self.knife.process_width
        z_hi = knife_center_z + self.knife.half_width_z + self.knife.process_width

        surface_triangles = self._update_surface_mesh(
            render_np,
            hidden_anchor,
            knife_x,
            z_lo,
            z_hi,
            render_negative_np=render_negative_np,
            render_positive_np=render_positive_np,
        )
        wall_triangles = 0
        active_tets = 0
        for tet in self.tet_indices:
            rest_tet = self.rest_points[tet]
            if float(np.mean(rest_tet[:, 0])) > knife_x:
                continue
            if float(np.max(rest_tet[:, 2])) < z_lo or float(np.min(rest_tet[:, 2])) > z_hi:
                continue

            polygons: dict[float, np.ndarray] = {}
            for side, side_render_np in ((-1.0, render_negative_np), (1.0, render_positive_np)):
                side_polygon = self._tet_plane_polygon(
                    rest_tet,
                    side_render_np[tet],
                    float(self.knife.center_y),
                    keep_positive=side > 0.0,
                )
                if side_polygon is not None:
                    polygons[side] = side_polygon
            if not polygons:
                continue
            active_tets += 1

            for side in (-1.0, 1.0):
                polygon = polygons.get(side)
                if polygon is None:
                    continue
                opened_polygon = self._apply_visual_opening(polygon, side, knife_x)
                for i in range(1, opened_polygon.shape[0] - 1):
                    if wall_triangles >= self.max_wall_triangles:
                        break
                    vertex = wall_triangles * 3
                    if side < 0.0:
                        self.wall_points_np[vertex + 0] = opened_polygon[0]
                        self.wall_points_np[vertex + 1] = opened_polygon[i + 1]
                        self.wall_points_np[vertex + 2] = opened_polygon[i]
                    else:
                        self.wall_points_np[vertex + 0] = opened_polygon[0]
                        self.wall_points_np[vertex + 1] = opened_polygon[i]
                        self.wall_points_np[vertex + 2] = opened_polygon[i + 1]
                    wall_triangles += 1
                if wall_triangles >= self.max_wall_triangles:
                    break
            if wall_triangles >= self.max_wall_triangles:
                break

        self.last_stats = TetCutWallRenderStats(
            active_x_segments=active_tets,
            surface_vertex_count=int(surface_triangles * 3),
            surface_triangle_count=int(surface_triangles),
            wall_vertex_count=int(wall_triangles * 3),
            wall_triangle_count=int(wall_triangles),
            coarse_dx=self.nominal_edge_length,
            min_active_dx=self.nominal_edge_length,
            max_active_dx=self.nominal_edge_length,
            active_tet_count=int(active_tets),
        )
        return self.last_stats

    def log(
        self,
        viewer,
        device,
        time: float,
        current_points: wp.array,
        prefix: str = "/cutting/tet_cut_surface",
        surface_color: tuple[float, float, float] = (0.18, 0.62, 0.95),
        wall_color: tuple[float, float, float] = (0.95, 0.32, 0.42),
        surface_opacity: float = 0.72,
        wall_opacity: float = 1.0,
        front_x: float | None = None,
        center_z: float | None = None,
        enrichment_points: wp.array | np.ndarray | None = None,
    ) -> TetCutWallRenderStats:
        self._ensure_device_arrays(device)
        stats = self.update(
            current_points,
            time,
            front_x=front_x,
            center_z=center_z,
            enrichment_points=enrichment_points,
        )
        assert self.surface_points_wp is not None
        assert self.surface_indices_wp is not None
        assert self.wall_points_wp is not None
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
            roughness=0.72,
            opacity=surface_opacity,
        )
        viewer.log_mesh(
            f"{prefix}/cut_walls",
            self.wall_points_wp,
            self.wall_indices_wp,
            hidden=stats.wall_triangle_count == 0,
            backface_culling=False,
            color=wall_color,
            roughness=0.86,
            opacity=wall_opacity,
        )
        return stats


class ShellCutSurfaceRenderer:
    """Render a thin triangle sheet with an opened X-FEM-style cut wake.

    This renderer is the shell counterpart of :class:`TetMeshCutSurfaceRenderer`.
    It keeps the full cloth surface visible, clips triangles that cross the cut
    plane behind the knife into side-specific polygons, and duplicates the seam
    vertices so the two paper sides can separate without deleting material.
    """

    def __init__(
        self,
        rest_points: np.ndarray,
        surface_indices: np.ndarray,
        knife: KnifeProfile,
        nominal_edge_length: float,
        max_surface_triangles: int | None = None,
        max_wall_triangles: int | None = None,
        max_visual_gap: float = 0.04,
        front_width: float | None = None,
        render_seam_edges: bool = True,
        render_surface_edges: bool = False,
        cut_refine_factor: int = 1,
    ):
        self.rest_points = np.asarray(rest_points, dtype=np.float32)
        self.base_surface_triangles_np = np.asarray(surface_indices, dtype=np.int32).reshape(-1, 3)
        self.knife = knife
        self.nominal_edge_length = float(max(nominal_edge_length, 1.0e-8))
        self.max_visual_gap = float(max_visual_gap)
        self.front_width = float(front_width if front_width is not None else max(2.0 * knife.process_width, 1.0e-4))
        self.render_seam_edges = bool(render_seam_edges)
        self.render_surface_edges = bool(render_surface_edges)
        self.cut_refine_factor = max(1, int(cut_refine_factor))
        self.seam_lift = 0.04 * self.nominal_edge_length
        self.max_surface_triangles = int(max_surface_triangles or max(1, 8 * len(self.base_surface_triangles_np)))
        self.max_wall_triangles = int(max_wall_triangles or max(1, 4 * len(self.base_surface_triangles_np)))
        self.max_edge_segments = int(max(1, 3 * self.max_surface_triangles))

        self.surface_points_np = np.zeros((self.max_surface_triangles * 3, 3), dtype=np.float32)
        self.surface_indices_np = np.arange(self.max_surface_triangles * 3, dtype=np.int32)
        self.wall_points_np = np.zeros((self.max_wall_triangles * 3, 3), dtype=np.float32)
        self.wall_indices_np = np.arange(self.max_wall_triangles * 3, dtype=np.int32)
        self.edge_starts_np = np.zeros((self.max_edge_segments, 3), dtype=np.float32)
        self.edge_ends_np = np.zeros((self.max_edge_segments, 3), dtype=np.float32)
        self.surface_points_wp: wp.array | None = None
        self.surface_indices_wp: wp.array | None = None
        self.wall_points_wp: wp.array | None = None
        self.wall_indices_wp: wp.array | None = None
        self.edge_starts_wp: wp.array | None = None
        self.edge_ends_wp: wp.array | None = None
        self.device_key: str | None = None
        self.last_edge_segment_count = 0
        self.last_stats = TetCutWallRenderStats(
            active_x_segments=0,
            surface_vertex_count=0,
            surface_triangle_count=0,
            wall_vertex_count=0,
            wall_triangle_count=0,
            coarse_dx=self.nominal_edge_length,
            min_active_dx=self.nominal_edge_length,
            max_active_dx=self.nominal_edge_length,
            active_tet_count=0,
        )

    def _ensure_device_arrays(self, device):
        device_key = str(device)
        if self.wall_points_wp is not None and self.device_key == device_key:
            return
        self.device_key = device_key
        self.surface_points_wp = wp.array(self.surface_points_np, dtype=wp.vec3, device=device)
        self.surface_indices_wp = wp.array(self.surface_indices_np, dtype=wp.int32, device=device)
        self.wall_points_wp = wp.array(self.wall_points_np, dtype=wp.vec3, device=device)
        self.wall_indices_wp = wp.array(self.wall_indices_np, dtype=wp.int32, device=device)
        self.edge_starts_wp = wp.array(self.edge_starts_np, dtype=wp.vec3, device=device)
        self.edge_ends_wp = wp.array(self.edge_ends_np, dtype=wp.vec3, device=device)

    @staticmethod
    def _current_points_np(points: np.ndarray | wp.array) -> np.ndarray:
        if isinstance(points, wp.array):
            return points.numpy().astype(np.float32, copy=False)
        return np.asarray(points, dtype=np.float32)

    def _apply_visual_opening(self, polygon: np.ndarray | None, side: float, knife_x: float) -> np.ndarray | None:
        if polygon is None or polygon.size == 0 or self.max_visual_gap <= 0.0:
            return polygon
        opened = np.asarray(polygon, dtype=np.float32).copy()
        for i in range(opened.shape[0]):
            opened_y = _cutting_visual_opening_y(
                float(opened[i, 0]),
                side,
                knife_x,
                float(self.knife.center_y_at_x(float(opened[i, 0]))),
                self.max_visual_gap,
                self.front_width,
            )
            if side > 0.0:
                opened[i, 1] = max(float(opened[i, 1]), opened_y)
            else:
                opened[i, 1] = min(float(opened[i, 1]), opened_y)
        return opened

    def _append_surface_triangle(self, triangle_count: int, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
        if triangle_count >= self.max_surface_triangles:
            return triangle_count
        if float(np.linalg.norm(np.cross(b - a, c - a))) <= 1.0e-10:
            return triangle_count
        vertex = triangle_count * 3
        self.surface_points_np[vertex + 0] = a
        self.surface_points_np[vertex + 1] = b
        self.surface_points_np[vertex + 2] = c
        return triangle_count + 1

    def _append_surface_polygon(self, triangle_count: int, polygon: np.ndarray | None) -> int:
        if polygon is None or polygon.shape[0] < 3:
            return triangle_count
        for i in range(1, polygon.shape[0] - 1):
            triangle_count = self._append_surface_triangle(triangle_count, polygon[0], polygon[i], polygon[i + 1])
        return triangle_count

    def _append_edge_segment(self, edge_count: int, a: np.ndarray, b: np.ndarray) -> int:
        if edge_count >= self.max_edge_segments:
            return edge_count
        if float(np.linalg.norm(b - a)) <= 1.0e-8:
            return edge_count
        self.edge_starts_np[edge_count] = a
        self.edge_ends_np[edge_count] = b
        return edge_count + 1

    def _append_wall_quad(self, wall_count: int, a: np.ndarray, b: np.ndarray) -> int:
        if wall_count + 1 >= self.max_wall_triangles:
            return wall_count
        lift = np.asarray((0.0, 0.0, self.seam_lift), dtype=np.float32)
        vertex = wall_count * 3
        self.wall_points_np[vertex + 0] = a
        self.wall_points_np[vertex + 1] = b
        self.wall_points_np[vertex + 2] = b + lift
        wall_count += 1
        vertex = wall_count * 3
        self.wall_points_np[vertex + 0] = a
        self.wall_points_np[vertex + 1] = b + lift
        self.wall_points_np[vertex + 2] = a + lift
        return wall_count + 1

    @staticmethod
    def _interpolate_triangle(triangle: np.ndarray, u: float, v: float) -> np.ndarray:
        return (triangle[0] + float(u) * (triangle[1] - triangle[0]) + float(v) * (triangle[2] - triangle[0])).astype(
            np.float32
        )

    def _iter_refined_triangle_patches(
        self,
        rest_tri: np.ndarray,
        current_tri: np.ndarray,
        negative_tri: np.ndarray,
        positive_tri: np.ndarray,
        refine_factor: int,
    ):
        factor = max(1, int(refine_factor))
        if factor <= 1:
            yield rest_tri, current_tri, negative_tri, positive_tri
            return

        inv = 1.0 / float(factor)
        rest_grid: dict[tuple[int, int], np.ndarray] = {}
        current_grid: dict[tuple[int, int], np.ndarray] = {}
        negative_grid: dict[tuple[int, int], np.ndarray] = {}
        positive_grid: dict[tuple[int, int], np.ndarray] = {}
        for i in range(factor + 1):
            for j in range(factor + 1 - i):
                u = float(i) * inv
                v = float(j) * inv
                key = (i, j)
                rest_grid[key] = self._interpolate_triangle(rest_tri, u, v)
                current_grid[key] = self._interpolate_triangle(current_tri, u, v)
                negative_grid[key] = self._interpolate_triangle(negative_tri, u, v)
                positive_grid[key] = self._interpolate_triangle(positive_tri, u, v)

        for i in range(factor):
            for j in range(factor - i):
                tri_a = ((i, j), (i + 1, j), (i, j + 1))
                yield (
                    np.asarray([rest_grid[key] for key in tri_a], dtype=np.float32),
                    np.asarray([current_grid[key] for key in tri_a], dtype=np.float32),
                    np.asarray([negative_grid[key] for key in tri_a], dtype=np.float32),
                    np.asarray([positive_grid[key] for key in tri_a], dtype=np.float32),
                )
                if i + j < factor - 1:
                    tri_b = ((i + 1, j), (i + 1, j + 1), (i, j + 1))
                    yield (
                        np.asarray([rest_grid[key] for key in tri_b], dtype=np.float32),
                        np.asarray([current_grid[key] for key in tri_b], dtype=np.float32),
                        np.asarray([negative_grid[key] for key in tri_b], dtype=np.float32),
                        np.asarray([positive_grid[key] for key in tri_b], dtype=np.float32),
                    )

    @staticmethod
    def _deduplicate_polygon(
        rest_polygon: list[np.ndarray],
        render_polygon: list[np.ndarray],
        eps: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if len(rest_polygon) < 3:
            return None, None

        clean_rest: list[np.ndarray] = []
        clean_render: list[np.ndarray] = []
        for rest_point, render_point in zip(rest_polygon, render_polygon, strict=True):
            if clean_rest and float(np.linalg.norm(rest_point - clean_rest[-1])) <= eps:
                clean_rest[-1] = rest_point.astype(np.float32, copy=False)
                clean_render[-1] = render_point.astype(np.float32, copy=False)
            else:
                clean_rest.append(rest_point.astype(np.float32, copy=False))
                clean_render.append(render_point.astype(np.float32, copy=False))

        if len(clean_rest) >= 2 and float(np.linalg.norm(clean_rest[0] - clean_rest[-1])) <= eps:
            clean_rest.pop()
            clean_render.pop()
        if len(clean_rest) < 3:
            return None, None
        return np.asarray(clean_rest, dtype=np.float32), np.asarray(clean_render, dtype=np.float32)

    def _clip_polygon_by_scalar(
        self,
        rest_polygon: np.ndarray | None,
        render_polygon: np.ndarray | None,
        scalar_fn,
        keep_positive: bool,
        eps: float = 1.0e-7,
        lock_to_inside_displacement: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if rest_polygon is None or render_polygon is None or rest_polygon.shape[0] < 3:
            return None, None
        out_rest: list[np.ndarray] = []
        out_render: list[np.ndarray] = []

        def inside(value: float) -> bool:
            return value >= -eps if keep_positive else value <= eps

        count = int(rest_polygon.shape[0])
        prev_rest = rest_polygon[-1]
        prev_render = render_polygon[-1]
        prev_value = float(scalar_fn(prev_rest))
        prev_inside = inside(prev_value)

        for index in range(count):
            curr_rest = rest_polygon[index]
            curr_render = render_polygon[index]
            curr_value = float(scalar_fn(curr_rest))
            curr_inside = inside(curr_value)

            if curr_inside != prev_inside:
                denom = prev_value - curr_value
                alpha = 0.0 if abs(denom) <= eps else float(np.clip(prev_value / denom, 0.0, 1.0))
                hit_rest = prev_rest + alpha * (curr_rest - prev_rest)
                if lock_to_inside_displacement and prev_inside and not curr_inside:
                    hit_render = hit_rest + (prev_render - prev_rest)
                elif lock_to_inside_displacement and curr_inside and not prev_inside:
                    hit_render = hit_rest + (curr_render - curr_rest)
                else:
                    hit_render = prev_render + alpha * (curr_render - prev_render)
                out_rest.append(hit_rest.astype(np.float32))
                out_render.append(hit_render.astype(np.float32))
            if curr_inside:
                out_rest.append(curr_rest.astype(np.float32, copy=False))
                out_render.append(curr_render.astype(np.float32, copy=False))

            prev_rest = curr_rest
            prev_render = curr_render
            prev_value = curr_value
            prev_inside = curr_inside

        return self._deduplicate_polygon(out_rest, out_render, eps)

    def _clip_triangle_by_side(
        self,
        rest_tri: np.ndarray,
        render_tri: np.ndarray,
        keep_positive: bool,
        eps: float = 1.0e-7,
    ) -> np.ndarray | None:
        _rest, render = self._clip_polygon_by_scalar(
            rest_tri,
            render_tri,
            lambda point: float(self.knife.signed_cut_y(np.asarray(point, dtype=np.float32)[None, :])[0]),
            keep_positive=keep_positive,
            eps=eps,
            lock_to_inside_displacement=True,
        )
        return render

    def _plane_segment(
        self,
        rest_tri: np.ndarray,
        render_tri: np.ndarray,
        keep_positive: bool,
        eps: float = 1.0e-7,
    ):
        if rest_tri is None or render_tri is None or rest_tri.shape[0] < 3:
            return None
        signed = self.knife.signed_cut_y(rest_tri)
        hits: list[np.ndarray] = []
        for edge_id in range(int(rest_tri.shape[0])):
            next_id = (edge_id + 1) % int(rest_tri.shape[0])
            s0 = float(signed[edge_id])
            s1 = float(signed[next_id])
            p0 = render_tri[edge_id]
            p1 = render_tri[next_id]
            r0 = rest_tri[edge_id]
            r1 = rest_tri[next_id]
            if abs(s0) <= eps:
                hits.append(p0.astype(np.float32, copy=False))
            if s0 * s1 < 0.0:
                alpha = s0 / (s0 - s1)
                rest_hit = r0 + alpha * (r1 - r0)
                use_start = s0 >= -eps if keep_positive else s0 <= eps
                side_rest = r0 if use_start else r1
                side_render = p0 if use_start else p1
                hits.append((rest_hit + (side_render - side_rest)).astype(np.float32))
            elif abs(s1) <= eps:
                hits.append(p1.astype(np.float32, copy=False))
        if len(hits) < 2:
            return None
        unique: list[np.ndarray] = []
        for hit in hits:
            if not any(float(np.linalg.norm(hit - existing)) <= eps for existing in unique):
                unique.append(hit)
        if len(unique) < 2:
            return None
        points = np.asarray(unique, dtype=np.float32)
        order = np.lexsort((points[:, 2], points[:, 0]))
        return points[order[[0, -1]]]

    def _front_scalar_fn(self, knife_x: float):
        front_origin = np.asarray(
            [knife_x, float(self.knife.center_y_at_x(knife_x)), self.knife.center_z],
            dtype=np.float32,
        )
        tangent = np.asarray(self.knife.path_tangent_at_x(knife_x), dtype=np.float32)

        def scalar(point: np.ndarray) -> float:
            return float(np.dot(np.asarray(point, dtype=np.float32) - front_origin, tangent))

        return scalar

    def _update_edge_overlay(self, hidden_anchor: np.ndarray, surface_triangles: int, edge_count: int = 0) -> int:
        if edge_count <= 0:
            self.edge_starts_np[:, :] = hidden_anchor
            self.edge_ends_np[:, :] = hidden_anchor
        for tri_id in range(min(int(surface_triangles), self.max_surface_triangles)):
            base = tri_id * 3
            a = self.surface_points_np[base + 0]
            b = self.surface_points_np[base + 1]
            c = self.surface_points_np[base + 2]
            for p0, p1 in ((a, b), (b, c), (c, a)):
                edge_count = self._append_edge_segment(edge_count, p0, p1)
                if edge_count >= self.max_edge_segments:
                    return edge_count
        return edge_count

    def _surface_triangle_in_cut_wake(
        self,
        rest_tri: np.ndarray,
        knife_x: float,
        z_lo: float,
        z_hi: float,
    ) -> bool:
        if float(np.min(rest_tri[:, 0])) > knife_x:
            return False
        if float(np.max(rest_tri[:, 2])) < z_lo or float(np.min(rest_tri[:, 2])) > z_hi:
            return False
        return True

    def update(
        self,
        current_points: np.ndarray | wp.array,
        time: float,
        front_x: float | None = None,
        center_z: float | None = None,
        enrichment_points: np.ndarray | wp.array | None = None,
        triangle_cut_state: np.ndarray | wp.array | None = None,
    ) -> TetCutWallRenderStats:
        current_np = self._current_points_np(current_points)
        if current_np.shape != self.rest_points.shape:
            raise ValueError("current_points must match rest point shape")
        rest_signed = self.knife.signed_cut_y(self.rest_points)[:, None]
        rest_side = np.where(rest_signed >= 0.0, 1.0, -1.0).astype(np.float32)
        cut_state_np = None
        if triangle_cut_state is not None:
            if isinstance(triangle_cut_state, wp.array):
                cut_state_np = triangle_cut_state.numpy().astype(np.int32, copy=False)
            else:
                cut_state_np = np.asarray(triangle_cut_state, dtype=np.int32)
            if cut_state_np.shape[0] < self.base_surface_triangles_np.shape[0]:
                raise ValueError("triangle_cut_state must cover all shell triangles")
        if enrichment_points is None:
            render_negative_np = current_np
            render_positive_np = current_np
        else:
            enrichment_np = self._current_points_np(enrichment_points)
            if enrichment_np.shape != self.rest_points.shape:
                raise ValueError("enrichment_points must match rest point shape")
            opening = np.minimum(np.linalg.norm(enrichment_np, axis=1, keepdims=True), self.max_visual_gap)
            normals = np.asarray(self.knife.path_normal_at_x(self.rest_points[:, 0]), dtype=np.float32)
            render_negative_np = current_np - opening * normals
            render_positive_np = current_np + opening * normals
        render_np = np.where(rest_side > 0.0, render_positive_np, render_negative_np)

        hidden_anchor = np.mean(current_np, axis=0, dtype=np.float64).astype(np.float32)
        self.surface_points_np[:, :] = hidden_anchor
        self.wall_points_np[:, :] = hidden_anchor
        self.edge_starts_np[:, :] = hidden_anchor
        self.edge_ends_np[:, :] = hidden_anchor
        knife_x = float(self.knife.x_at(time) if front_x is None else front_x)
        knife_center_z = float(self.knife.center_z if center_z is None else center_z)
        z_lo = knife_center_z - self.knife.half_width_z - self.knife.process_width
        z_hi = knife_center_z + self.knife.half_width_z + self.knife.process_width
        eps = 1.0e-7

        surface_triangles = 0
        wall_triangles = 0
        edge_segments = 0
        active_triangles = 0
        front_scalar = self._front_scalar_fn(knife_x)

        def cut_scalar(point: np.ndarray) -> float:
            return float(self.knife.signed_cut_y(np.asarray(point, dtype=np.float32)[None, :])[0])

        for tri_id, tri in enumerate(self.base_surface_triangles_np):
            rest_tri = self.rest_points[tri]
            signed = self.knife.signed_cut_y(rest_tri)
            in_wake = self._surface_triangle_in_cut_wake(rest_tri, knife_x, z_lo, z_hi)
            solver_cut = True if cut_state_np is None else int(cut_state_np[tri_id]) != 0
            if in_wake and solver_cut:
                emitted_from_source = False
                for patch_rest, patch_current, patch_negative, patch_positive in self._iter_refined_triangle_patches(
                    rest_tri,
                    current_np[tri],
                    render_negative_np[tri],
                    render_positive_np[tri],
                    self.cut_refine_factor,
                ):
                    _current_rest, current_render = self._clip_polygon_by_scalar(
                        patch_rest,
                        patch_current,
                        front_scalar,
                        keep_positive=True,
                        eps=eps,
                    )
                    negative_rest, negative_render = self._clip_polygon_by_scalar(
                        patch_rest,
                        patch_negative,
                        front_scalar,
                        keep_positive=False,
                        eps=eps,
                    )
                    positive_rest, positive_render = self._clip_polygon_by_scalar(
                        patch_rest,
                        patch_positive,
                        front_scalar,
                        keep_positive=False,
                        eps=eps,
                    )

                    negative = None
                    positive = None
                    if negative_rest is not None and negative_render is not None:
                        _negative_rest, negative = self._clip_polygon_by_scalar(
                            negative_rest,
                            negative_render,
                            cut_scalar,
                            keep_positive=False,
                            eps=eps,
                            lock_to_inside_displacement=True,
                        )
                    if positive_rest is not None and positive_render is not None:
                        _positive_rest, positive = self._clip_polygon_by_scalar(
                            positive_rest,
                            positive_render,
                            cut_scalar,
                            keep_positive=True,
                            eps=eps,
                            lock_to_inside_displacement=True,
                        )

                    if negative is not None or positive is not None:
                        emitted_from_source = True
                    surface_triangles = self._append_surface_polygon(
                        surface_triangles, self._apply_visual_opening(negative, -1.0, knife_x)
                    )
                    surface_triangles = self._append_surface_polygon(
                        surface_triangles, self._apply_visual_opening(positive, 1.0, knife_x)
                    )
                    surface_triangles = self._append_surface_polygon(surface_triangles, current_render)
                    if self.render_seam_edges:
                        for side, side_rest, side_render in (
                            (-1.0, negative_rest, negative_render),
                            (1.0, positive_rest, positive_render),
                        ):
                            segment = self._plane_segment(side_rest, side_render, keep_positive=side > 0.0)
                            segment = self._apply_visual_opening(segment, side, knife_x)
                            if segment is not None:
                                edge_segments = self._append_edge_segment(edge_segments, segment[0], segment[1])
                if emitted_from_source:
                    active_triangles += 1
            else:
                side = 1.0 if float(np.mean(signed)) >= 0.0 else -1.0
                side_render_np = render_positive_np if side > 0.0 else render_negative_np
                if cut_state_np is not None and not solver_cut:
                    render_tri = current_np[tri]
                elif in_wake:
                    render_tri = side_render_np[tri]
                else:
                    render_tri = render_np[tri]
                surface_triangles = self._append_surface_triangle(
                    surface_triangles, render_tri[0], render_tri[1], render_tri[2]
                )

        self.last_edge_segment_count = int(edge_segments)
        self.last_stats = TetCutWallRenderStats(
            active_x_segments=int(active_triangles),
            surface_vertex_count=int(surface_triangles * 3),
            surface_triangle_count=int(surface_triangles),
            wall_vertex_count=int(wall_triangles * 3),
            wall_triangle_count=int(wall_triangles),
            coarse_dx=self.nominal_edge_length,
            min_active_dx=self.nominal_edge_length,
            max_active_dx=self.nominal_edge_length,
            active_tet_count=0,
        )
        return self.last_stats

    def log(
        self,
        viewer,
        device,
        time: float,
        current_points: wp.array,
        prefix: str = "/cutting/shell_cut_surface",
        surface_color: tuple[float, float, float] = (0.94, 0.94, 0.90),
        wall_color: tuple[float, float, float] = (0.65, 0.16, 0.16),
        surface_opacity: float = 0.92,
        wall_opacity: float = 0.95,
        front_x: float | None = None,
        center_z: float | None = None,
        enrichment_points: wp.array | np.ndarray | None = None,
        triangle_cut_state: wp.array | np.ndarray | None = None,
    ) -> TetCutWallRenderStats:
        self._ensure_device_arrays(device)
        stats = self.update(
            current_points,
            time,
            front_x=front_x,
            center_z=center_z,
            enrichment_points=enrichment_points,
            triangle_cut_state=triangle_cut_state,
        )
        assert self.surface_points_wp is not None
        assert self.surface_indices_wp is not None
        assert self.wall_points_wp is not None
        assert self.wall_indices_wp is not None
        assert self.edge_starts_wp is not None
        assert self.edge_ends_wp is not None
        self.surface_points_wp.assign(self.surface_points_np)
        self.wall_points_wp.assign(self.wall_points_np)
        edge_count = self.last_edge_segment_count
        if self.render_surface_edges:
            hidden_anchor = np.mean(self._current_points_np(current_points), axis=0, dtype=np.float64).astype(
                np.float32
            )
            edge_count = self._update_edge_overlay(hidden_anchor, stats.surface_triangle_count, edge_count=edge_count)
        self.edge_starts_wp.assign(self.edge_starts_np)
        self.edge_ends_wp.assign(self.edge_ends_np)
        viewer.log_mesh(
            f"{prefix}/surface",
            self.surface_points_wp,
            self.surface_indices_wp,
            hidden=False,
            backface_culling=False,
            color=surface_color,
            roughness=0.78,
            opacity=surface_opacity,
        )
        viewer.log_mesh(
            f"{prefix}/tear_edges",
            self.wall_points_wp,
            self.wall_indices_wp,
            hidden=True,
            backface_culling=False,
            color=wall_color,
            roughness=0.86,
            opacity=wall_opacity,
        )
        if self.render_seam_edges or self.render_surface_edges:
            viewer.log_lines(
                f"{prefix}/cut_edges",
                self.edge_starts_wp,
                self.edge_ends_wp,
                wall_color if self.render_seam_edges else (0.08, 0.10, 0.12),
                width=0.004,
                hidden=edge_count == 0,
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


@wp.func
def _knife_edge_process_weight(
    q: wp.vec3,
    edge_points: wp.array[wp.vec3],
    edge_point_count: int,
    center_y: float,
    half_width_y: float,
    process_width: float,
):
    best_d2 = float(1.0e12)
    for i in range(edge_point_count - 1):
        a = edge_points[i]
        b = edge_points[i + 1]
        abx = b[0] - a[0]
        abz = b[2] - a[2]
        denom = abx * abx + abz * abz
        t = float(0.0)
        if denom > 1.0e-12:
            t = ((q[0] - a[0]) * abx + (q[2] - a[2]) * abz) / denom
            t = wp.min(1.0, wp.max(0.0, t))
        cx = a[0] + t * abx
        cz = a[2] + t * abz
        dx = q[0] - cx
        dz = q[2] - cz
        best_d2 = wp.min(best_d2, dx * dx + dz * dz)

    y_out = wp.max(0.0, wp.abs(q[1] - center_y) - half_width_y)
    distance = wp.sqrt(best_d2 + y_out * y_out)
    return wp.max(0.0, 1.0 - distance / wp.max(process_width, 1.0e-6))


@wp.kernel
def apply_mpm_knife_cut_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    damage: wp.array[wp.float32],
    colors: wp.array[wp.vec3],
    accum: wp.array[wp.float32],
    knife_edge_points: wp.array[wp.vec3],
    knife_edge_point_count: int,
    center_y: float,
    half_width_y: float,
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
    y_rel = q[1] - center_y
    weight = _knife_edge_process_weight(
        q,
        knife_edge_points,
        knife_edge_point_count,
        center_y,
        half_width_y,
        process_width,
    )

    active = weight > 0.0
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
    knife_edge_points: wp.array[wp.vec3],
    knife_edge_point_count: int,
    center_y: float,
    half_width_y: float,
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
    y_rel = q[1] - center_y
    weight = _knife_edge_process_weight(
        q,
        knife_edge_points,
        knife_edge_point_count,
        center_y,
        half_width_y,
        process_width,
    )

    active = weight > 0.0
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
    edge_points_np = knife.edge_points(time_value)
    edge_points = wp.array(edge_points_np, dtype=wp.vec3, device=device)
    wp.launch(
        apply_mpm_knife_cut_kernel,
        dim=state.particle_count,
        inputs=[
            state.particle_q,
            state.particle_qd,
            damage,
            colors,
            accum,
            edge_points,
            int(edge_points_np.shape[0]),
            knife.center_y,
            knife.half_width_y,
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
    edge_points_np = knife.edge_points(time_value)
    edge_points = wp.array(edge_points_np, dtype=wp.vec3, device=device)
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
            edge_points,
            int(edge_points_np.shape[0]),
            knife.center_y,
            knife.half_width_y,
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
        self.normal_forces: list[float] = []
        self.friction_forces: list[float] = []

    def append_from_accum(self, time_value: float, accum: wp.array, particle_count: int):
        values = accum.numpy()
        self.append_values(
            time_value,
            float(values[0]),
            float(values[1]),
            float(values[2]) / max(float(particle_count), 1.0),
        )

    def append_values(
        self,
        time_value: float,
        force: float,
        active_count: float,
        mean_damage: float,
        normal_force: float | None = None,
        friction_force: float | None = None,
    ):
        self.times.append(float(time_value))
        self.forces.append(float(force))
        self.active_counts.append(float(active_count))
        self.mean_damage.append(float(mean_damage))
        if normal_force is not None or friction_force is not None:
            self.normal_forces.append(float(normal_force or 0.0))
            self.friction_forces.append(float(friction_force or 0.0))

    def summary(self) -> dict[str, float]:
        return summarize_force_profile(np.array(self.times), np.array(self.forces), np.array(self.mean_damage))

    def to_dict(self) -> dict[str, list[float]]:
        payload = {
            "time_s": self.times,
            "force_n": self.forces,
            "active_particles": self.active_counts,
            "mean_damage": self.mean_damage,
        }
        if self.normal_forces or self.friction_forces:
            payload["normal_force_n"] = self.normal_forces
            payload["friction_force_n"] = self.friction_forces
        return payload

    def write_csv(self, path: str | Path):
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            has_components = bool(self.normal_forces or self.friction_forces)
            if has_components:
                f.write("time_s,force_n,normal_force_n,friction_force_n,active_particles,mean_damage\n")
                for i, row in enumerate(
                    zip(self.times, self.forces, self.active_counts, self.mean_damage, strict=True)
                ):
                    normal = self.normal_forces[i] if i < len(self.normal_forces) else 0.0
                    friction = self.friction_forces[i] if i < len(self.friction_forces) else 0.0
                    f.write(f"{row[0]:.8f},{row[1]:.8f},{normal:.8f},{friction:.8f},{row[2]:.0f},{row[3]:.8f}\n")
            else:
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
    if history.normal_forces:
        ax_force.plot(history.times, history.normal_forces, color="#f97316", linewidth=1.4, label="normal")
    if history.friction_forces:
        ax_force.plot(history.times, history.friction_forces, color="#7c3aed", linewidth=1.4, label="friction")
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

        iio.imwrite(path, np.asarray(frames), fps=fps, codec="libx264", macro_block_size=1)
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


def capture_viewer_frame(viewer, render_ui: bool = True) -> np.ndarray | None:
    if not hasattr(viewer, "get_frame"):
        return None
    image = viewer.get_frame(render_ui=render_ui)
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
