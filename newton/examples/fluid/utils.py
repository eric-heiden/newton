# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


@wp.kernel
def _prepare_particle_display(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[int],
    worlds: wp.array[int],
    world_offsets: wp.array[wp.vec3],
    visible_worlds: wp.array[int],
    display_positions: wp.array[wp.vec3],
    display_radii: wp.array[float],
):
    i = wp.tid()
    position = positions[i]
    world = worlds[i]
    if world >= 0 and world < world_offsets.shape[0]:
        position += world_offsets[world]
    display_positions[i] = position
    radius = float(0.0)
    visible = world < 0 or visible_worlds.shape[0] == 0
    if world >= 0 and world < visible_worlds.shape[0]:
        visible = visible_worlds[world] != 0
    if visible and (flags[i] & newton.ParticleFlags.ACTIVE) != 0:
        radius = radii[i]
    display_radii[i] = radius


class FluidParticleRenderer:
    """Draw blue fluid points with display-only world offsets.

    The stock particle logger does not apply world offsets. Keep a persistent
    display buffer instead of moving simulation particles, and mask inactive
    particles with zero radii to avoid host readbacks for stream compaction.
    """

    def __init__(self, model: newton.Model):
        self.model = model
        self.positions = wp.clone(model.particle_q)
        self.radii = wp.clone(model.particle_radius)
        self.colors = wp.full(model.particle_count, wp.vec3(0.12, 0.48, 0.82), device=model.device)
        self._no_offsets = wp.empty(0, dtype=wp.vec3, device=model.device)
        self._all_worlds = wp.empty(0, dtype=int, device=model.device)

    def log_state(self, viewer, state: newton.State) -> None:
        """Log rigid geometry and offset particles, preserving the visibility toggle."""
        if isinstance(viewer, ViewerNull):
            return
        show_particles = viewer.show_particles
        viewer.show_particles = False
        try:
            viewer.log_state(state)
        finally:
            viewer.show_particles = show_particles
        if show_particles:
            offsets = viewer.world_offsets
            # ViewerBase maintains this device mask when set_visible_worlds changes.
            visible_worlds = getattr(viewer, "_visible_worlds_mask", None)
            wp.launch(
                _prepare_particle_display,
                dim=self.model.particle_count,
                inputs=[
                    state.particle_q,
                    self.model.particle_radius,
                    self.model.particle_flags,
                    self.model.particle_world,
                    self._no_offsets if offsets is None else offsets,
                    self._all_worlds if visible_worlds is None else visible_worlds,
                    self.positions,
                    self.radii,
                ],
                device=self.model.device,
            )
        viewer.log_points(
            "/model/fluid_particles", self.positions, radii=self.radii, colors=self.colors, hidden=not show_particles
        )


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
    """Return grid dimensions with the requested per-axis minimum counts."""
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("particle spacing must be positive")
    if len(size) != 3 or len(minimum) != 3:
        raise ValueError("size and minimum must have three components")
    if any(not np.isfinite(extent) or extent <= 0.0 for extent in size):
        raise ValueError("grid size must contain positive finite extents")
    if any(int(lower) != lower or lower < 1 for lower in minimum):
        raise ValueError("minimum grid dimensions must be positive integers")
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
    if not np.isfinite(reference_spacing) or reference_spacing <= 0.0:
        raise ValueError("reference spacing must be positive")

    reference_count = max(int(count_particles(reference_spacing)), 1)
    guess = reference_spacing * (reference_count / target_count) ** (1.0 / 3.0)
    lower = guess
    upper = guess
    lower_count = max(int(count_particles(lower)), 0)
    upper_count = lower_count

    for _ in range(64):
        if lower_count >= target_count:
            break
        upper = lower
        upper_count = lower_count
        lower *= 0.5
        lower_count = max(int(count_particles(lower)), 0)

    else:
        raise ValueError("particle count cannot reach the target at smaller spacing")

    for _ in range(64):
        if upper_count <= target_count:
            break
        lower = upper
        lower_count = upper_count
        upper *= 2.0
        upper_count = max(int(count_particles(upper)), 0)

    else:
        raise ValueError("particle count cannot reach the target at larger spacing")

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
    """Resolve a Cartesian grid, allowing the minimum grid to exceed the target."""
    if target_count < 1:
        raise ValueError("target particle count must be positive")
    grid_dimensions(size, reference_spacing, minimum)
    target_count = max(target_count, int(np.prod(minimum)))

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
    if radial_limit <= 0.0 or fill_height < lower:
        return np.empty((0, 3), dtype=np.float64)
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


def step_simulation(example, graph_key=None) -> None:
    """Advance one fluid frame, preserving state buffers across graph replays."""

    def simulate():
        initial_state = example.state_0
        example.simulate()
        # Graph replays keep the pointers recorded during capture. An odd
        # substep count needs a copy back to the original input buffer.
        if example.state_0 is not initial_state:
            initial_state.assign(example.state_0)
            example.state_0, example.state_1 = example.state_1, example.state_0

    if not example.use_cuda_graph:
        simulate()
        return
    if example.graph is None or graph_key != getattr(example, "_graph_key", None):
        states = example.state_0, example.state_1
        try:
            with wp.ScopedCapture(device=example.model.device) as capture:
                simulate()
            example.graph = capture.graph
            example._graph_key = graph_key
        except Exception as exc:
            # Capture records kernels without running them, but Python state
            # swaps happen immediately and must be undone before fallback.
            example.state_0, example.state_1 = states
            example.graph = None
            example.use_cuda_graph = False
            warnings.warn(f"CUDA graph capture failed; running uncaptured: {exc}", stacklevel=2)
            simulate()
            return
    wp.capture_launch(example.graph)


def build_cup_mesh(inner_radius: float, wall_thickness: float, height: float, *, segments: int = 48) -> newton.Mesh:
    """Closed solid of revolution: a cylindrical cup with an open cavity."""
    ri = inner_radius
    ro = inner_radius + wall_thickness
    t = wall_thickness
    profile = [(ro, 0.0), (ro, height), (ri, height), (ri, t)]
    vertices = []
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        c, sn = np.cos(angle), np.sin(angle)
        for r, z in profile:
            vertices.append((r * c, r * sn, z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    cavity_center = len(vertices)
    vertices.append((0.0, 0.0, t))

    rows = len(profile)
    indices = []
    for i in range(segments):
        j = (i + 1) % segments
        for k in range(rows - 1):
            a = i * rows + k
            b = i * rows + k + 1
            c0 = j * rows + k
            d = j * rows + k + 1
            indices += [a, c0, b, b, c0, d]
        indices += [i * rows + 0, bottom_center, j * rows + 0]
        indices += [i * rows + rows - 1, j * rows + rows - 1, cavity_center]

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
    )


def validate_simulation_args(args, *, supported_solvers: tuple[str, ...]) -> None:
    """Reject unsupported backends and invalid timesteps before building a scene."""
    if args.solver not in supported_solvers:
        raise ValueError(f"This scene supports {', '.join(supported_solvers)}; got {args.solver!r}")
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    if args.substeps < 1:
        raise ValueError("substeps must be positive")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not np.isfinite(args.rest_density) or args.rest_density <= 0.0:
        raise ValueError("rest_density must be positive and finite")
