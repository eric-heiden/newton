# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton


@dataclass(frozen=True)
class ParticleGridConfig:
    """Resolved uniform particle-grid configuration."""

    spacing: float
    dimensions: tuple[int, int, int]
    particle_count: int

    @property
    def radius(self) -> float:
        return 0.5 * self.spacing


def parse_particle_count(value: str) -> int:
    """Parse a positive target particle count for example CLIs."""
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("particle count must be an integer") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("particle count must be positive")
    return count


def grid_dimensions(size: Sequence[float], spacing: float, minimum: Sequence[int] = (1, 1, 1)) -> tuple[int, int, int]:
    """Return integer dimensions for a uniform grid contained in ``size``."""
    if spacing <= 0.0:
        raise ValueError("particle spacing must be positive")
    if len(size) != 3 or len(minimum) != 3:
        raise ValueError("size and minimum must have three components")
    return tuple(
        max(int(np.floor(float(extent) / spacing + 1.0e-9)), int(lower))
        for extent, lower in zip(size, minimum, strict=True)
    )


def resolve_particle_spacing(
    target_count: int,
    reference_spacing: float,
    count_particles: Callable[[float], int],
    *,
    iterations: int = 28,
) -> tuple[float, int]:
    """Find the spacing whose realizable particle count is nearest a target."""
    if target_count < 1:
        raise ValueError("target particle count must be positive")
    if reference_spacing <= 0.0:
        raise ValueError("reference spacing must be positive")

    reference_count = max(int(count_particles(reference_spacing)), 1)
    guess = reference_spacing * (reference_count / target_count) ** (1.0 / 3.0)
    lower = guess
    upper = guess
    lower_count = max(int(count_particles(lower)), 0)
    upper_count = lower_count

    while lower_count < target_count:
        upper = lower
        upper_count = lower_count
        lower *= 0.5
        lower_count = max(int(count_particles(lower)), 0)

    while upper_count > target_count:
        lower = upper
        lower_count = upper_count
        upper *= 2.0
        upper_count = max(int(count_particles(upper)), 0)

    best_spacing = lower
    best_count = lower_count

    def consider(spacing: float, count: int) -> None:
        nonlocal best_spacing, best_count
        error = abs(count - target_count)
        best_error = abs(best_count - target_count)
        if error < best_error or (error == best_error and count <= target_count < best_count):
            best_spacing = spacing
            best_count = count

    consider(upper, upper_count)
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_count = max(int(count_particles(midpoint)), 0)
        consider(midpoint, midpoint_count)
        if midpoint_count >= target_count:
            lower = midpoint
        else:
            upper = midpoint

    return best_spacing, best_count


def resolve_particle_grid(
    target_count: int,
    size: Sequence[float],
    reference_spacing: float,
    minimum: Sequence[int] = (1, 1, 1),
) -> ParticleGridConfig:
    """Resolve a fixed-volume Cartesian grid from a target particle count."""

    def count_particles(spacing: float) -> int:
        return int(np.prod(grid_dimensions(size, spacing, minimum), dtype=np.int64))

    spacing, particle_count = resolve_particle_spacing(target_count, reference_spacing, count_particles)
    return ParticleGridConfig(spacing, grid_dimensions(size, spacing, minimum), particle_count)


def cylinder_particle_count(spacing: float, inner_radius: float, floor_height: float, fill_height: float) -> int:
    """Count cubic-lattice points inside a cylindrical fluid fill."""
    particle_radius = 0.5 * spacing
    radial_limit = inner_radius - particle_radius
    lower = floor_height + particle_radius
    if radial_limit <= 0.0 or fill_height < lower:
        return 0
    dimension_xy = max(int(2.0 * radial_limit / spacing) + 1, 1)
    dimension_z = max(int((fill_height - lower) / spacing) + 1, 1)
    axis = -radial_limit + spacing * np.arange(dimension_xy)
    radial_sq = axis[:, None] * axis[:, None] + axis[None, :] * axis[None, :]
    return int(np.count_nonzero(radial_sq < radial_limit * radial_limit)) * dimension_z


def cylinder_particle_positions(
    spacing: float,
    inner_radius: float,
    floor_height: float,
    fill_height: float,
) -> np.ndarray:
    """Create cubic-lattice points inside a cylindrical fluid fill."""
    particle_radius = 0.5 * spacing
    radial_limit = inner_radius - particle_radius
    lower = floor_height + particle_radius
    dimension_xy = max(int(2.0 * radial_limit / spacing) + 1, 1)
    dimension_z = max(int((fill_height - lower) / spacing) + 1, 1)
    axis_xy = -radial_limit + spacing * np.arange(dimension_xy)
    axis_z = lower + spacing * np.arange(dimension_z)
    grid_x, grid_y, grid_z = np.meshgrid(axis_xy, axis_xy, axis_z, indexing="ij")
    points = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=1)
    return points[points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1] < radial_limit * radial_limit]


def add_tank_walls(
    builder: newton.ModelBuilder,
    half_x: float,
    half_y: float,
    height: float,
    thickness: float,
    color: tuple[float, float, float],
    opacity: float,
) -> tuple[int, ...]:
    """Add four flush rectangular tank walls around the given inner bounds."""
    half_thickness = 0.5 * thickness
    center_z = 0.5 * height
    outer_half_x = half_x + thickness
    walls = []

    for side in (-1.0, 1.0):
        walls.append(
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(side * (half_x + half_thickness), 0.0, center_z),
                    wp.quat_identity(),
                ),
                hx=half_thickness,
                hy=half_y,
                hz=center_z,
                color=color,
                opacity=opacity,
                label=f"tank_wall_x_{side:+.0f}",
            )
        )

    # End walls span the complete outer width and meet the side-wall ends
    # without a gap or overlapping transparent geometry.
    for side in (-1.0, 1.0):
        walls.append(
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(0.0, side * (half_y + half_thickness), center_z),
                    wp.quat_identity(),
                ),
                hx=outer_half_x,
                hy=half_thickness,
                hz=center_z,
                color=color,
                opacity=opacity,
                label=f"tank_wall_y_{side:+.0f}",
            )
        )

    return tuple(walls)


def ignore_shapes_for_picking(viewer, shape_count: int, shape_indices: Iterable[int]) -> None:
    """Mark selected model shapes as transparent to viewer picking."""
    picking = getattr(viewer, "picking", None)
    if picking is None or not hasattr(picking, "set_pickable_shapes"):
        return

    indices = np.asarray(tuple(shape_indices), dtype=np.int32)
    if indices.size == 0:
        return
    if np.any(indices < 0) or np.any(indices >= shape_count):
        raise ValueError("Pick-ignored shape indices must be valid model shape indices.")

    mask = np.ones(int(shape_count), dtype=np.int32)
    mask[indices] = 0
    picking.set_pickable_shapes(mask)
