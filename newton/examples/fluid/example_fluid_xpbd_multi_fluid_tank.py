# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Multi-Fluid Tank
#
# Three XPBD fluid particle blocks with different particle densities and
# ViewerGL materials slosh together in one tank with floating rigid bodies.
# The fluids share the same global XPBD viscosity/cohesion settings, but their
# masses and screen-space fluid materials differ. Grab the boxes or fluids with
# the mouse to stir the tank and watch the logged fluid batches interact.
#
# Command: python -m newton.examples fluid_xpbd_multi_fluid_tank
#
###########################################################################

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.fluid.utils import (
    add_tank_walls,
    ignore_shapes_for_picking,
    parse_particle_count,
    resolve_particle_grid,
)

_REFERENCE_SPACING = 0.02
_REFERENCE_FILL_HEIGHT = 18 * _REFERENCE_SPACING


class Example:
    class FluidSpec(NamedTuple):
        label: str
        density_factor: float
        color: tuple[float, float, float, float]
        absorption: tuple[float, float, float] | None
        ior: float
        reflectance: float
        specular_intensity: float
        specular_power: float
        blur_radius_world: float
        radius_scale: float

    class FluidSegment(NamedTuple):
        label: str
        start: int
        count: int
        color: tuple[float, float, float, float]
        absorption: tuple[float, float, float] | None
        ior: float
        reflectance: float
        specular_intensity: float
        specular_power: float
        blur_radius_world: float
        radius_scale: float

    FLUID_SPECS = (
        FluidSpec(
            label="green",
            density_factor=1.0,
            color=(0.05, 0.68, 0.28, 0.76),
            absorption=None,
            ior=1.333,
            reflectance=0.10,
            specular_intensity=1.25,
            specular_power=420.0,
            blur_radius_world=0.035,
            radius_scale=1.8,
        ),
        FluidSpec(
            label="white",
            density_factor=1.35,
            color=(0.94, 0.95, 0.92, 0.55),
            absorption=None,
            ior=1.48,
            reflectance=0.05,
            specular_intensity=0.9,
            specular_power=180.0,
            blur_radius_world=0.045,
            radius_scale=1.95,
        ),
        FluidSpec(
            label="red",
            density_factor=0.82,
            color=(0.90, 0.10, 0.14, 0.66),
            absorption=(0.12, 1.35, 1.15),
            ior=1.47,
            reflectance=0.04,
            specular_intensity=0.8,
            specular_power=220.0,
            blur_radius_world=0.04,
            radius_scale=1.9,
        ),
    )

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        fluid_size = (
            args.tank_size[0] - _REFERENCE_SPACING,
            args.tank_size[1] - _REFERENCE_SPACING,
            _REFERENCE_FILL_HEIGHT,
        )
        particle_grid = resolve_particle_grid(
            args.particle_count,
            fluid_size,
            _REFERENCE_SPACING,
            minimum=(2 * max(2, int(args.fluid_count)), 2, 2),
        )
        spacing = particle_grid.spacing
        radius = 0.5 * spacing
        self.particle_radius = radius
        fluid_count = max(2, min(int(args.fluid_count), len(self.FLUID_SPECS)))
        density_factors = tuple(args.fluid_density_factors)

        self.tank_half_x = 0.5 * args.tank_size[0]
        self.tank_half_y = 0.5 * args.tank_size[1]
        wall_height = args.tank_size[2]
        self.wall_height = wall_height
        wall_thickness = 0.1

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        builder.default_particle_radius = radius
        builder.default_shape_cfg.mu = 0.2

        dim_x_total, dim_y, dim_z = particle_grid.dimensions
        dim_x_parts = [dim_x_total // fluid_count] * fluid_count
        for i in range(dim_x_total % fluid_count):
            dim_x_parts[i] += 1

        x0 = -0.5 * (dim_x_total - 1) * spacing
        y0 = -0.5 * (dim_y - 1) * spacing
        self.fluid_segments: list[Example.FluidSegment] = []
        cursor = 0
        for i, (base_spec, dim_x) in enumerate(zip(self.FLUID_SPECS[:fluid_count], dim_x_parts, strict=True)):
            density_factor = float(density_factors[i % len(density_factors)])
            mass = args.rest_density * density_factor * spacing**3
            start = builder.particle_count
            lateral_speed = 0.055 * (float(i) - 0.5 * float(fluid_count - 1))
            builder.add_particle_grid(
                pos=wp.vec3(x0 + cursor * spacing, y0, radius),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, lateral_speed, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=spacing,
                cell_y=spacing,
                cell_z=spacing,
                mass=mass,
                jitter=0.04 * spacing,
                radius_mean=radius,
                flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
            )
            self.fluid_segments.append(
                self.FluidSegment(
                    label=base_spec.label,
                    start=start,
                    count=builder.particle_count - start,
                    color=base_spec.color,
                    absorption=base_spec.absorption,
                    ior=base_spec.ior,
                    reflectance=base_spec.reflectance,
                    specular_intensity=base_spec.specular_intensity,
                    specular_power=base_spec.specular_power,
                    blur_radius_world=base_spec.blur_radius_world,
                    radius_scale=base_spec.radius_scale,
                )
            )
            cursor += dim_x

        wall_color = (0.62, 0.72, 0.78)
        self.wall_shapes = add_tank_walls(
            builder,
            self.tank_half_x,
            self.tank_half_y,
            wall_height,
            wall_thickness,
            wall_color,
            args.wall_opacity,
        )
        builder.add_ground_plane()

        colors = (
            (1.0, 0.78, 0.05),
            (0.10, 0.88, 0.35),
            (0.95, 0.18, 0.26),
            (0.20, 0.50, 1.0),
            (0.82, 0.35, 1.0),
        )
        self.box_bodies = []
        fractions = tuple(args.box_density_fractions)
        fluid_top = dim_z * spacing
        for i in range(args.box_count):
            column = i % 3
            row = i // 3
            x = (column - 1) * 0.34
            y = -0.18 + row * 0.32
            half = args.box_half_extent * (0.85 + 0.12 * (i % 3))
            q = wp.quat_from_axis_angle(wp.vec3(0.2, 0.8, 0.1), 0.2 * float(i))
            density = float(fractions[i % len(fractions)]) * args.rest_density
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, y, fluid_top + 0.25 + 0.05 * float(i)), q),
                label=f"multi_fluid_cube_{i}",
            )
            builder.add_shape_box(
                body,
                hx=half,
                hy=half,
                hz=half,
                cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=0.2),
                color=colors[i % len(colors)],
            )
            self.box_bodies.append(body)

        self.model = builder.finalize()
        self.model.particle_max_velocity = 0.5 * radius / self.sim_dt
        self.model.soft_contact_mu = 0.1

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=spacing,
            fluid_rest_density=args.rest_density,
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
            max_diffuse_particles=args.foam_max_particles,
            diffuse_lifetime=args.foam_lifetime,
            diffuse_threshold=0.7,
            diffuse_spawn_probability=args.foam_spawn_probability,
            diffuse_buoyancy=args.foam_buoyancy,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        self.render_smoothing = args.render_smoothing
        self.render_anisotropy_scale = args.render_anisotropy_scale
        self.foam_radius = args.foam_radius
        self.foam_stretch = args.foam_stretch
        self.foam_opacity = args.foam_opacity

        self.viewer.set_model(self.model)
        ignore_shapes_for_picking(self.viewer, self.model.shape_count, self.wall_shapes)
        self.viewer.picking_enabled = True
        self._apply_picking_params(args.pick_stiffness, args.pick_damping)
        use_fluid_surface = args.render_mode == "fluid" and getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)

        self.graph = None
        self.use_cuda_graph = wp.get_device(self.model.device).is_cuda
        self._graph_key = None

    def _graph_key_tuple(self):
        return (
            round(self.solver.fluid_viscosity, 6),
            round(self.solver.fluid_cohesion, 6),
            round(self.solver.diffuse_spawn_probability, 3),
            round(self.solver.diffuse_lifetime, 3),
        )

    def _apply_picking_params(self, stiffness, damping):
        picking = getattr(self.viewer, "picking", None)
        if picking is None:
            return
        picking.pick_stiffness = float(stiffness)
        picking.pick_damping = float(damping)
        state = picking.pick_state.numpy()
        state[0]["pick_stiffness"] = float(stiffness)
        state[0]["pick_damping"] = float(damping)
        picking.pick_state.assign(state)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.use_cuda_graph:
            key = self._graph_key_tuple()
            if self.graph is None or key != self._graph_key:
                try:
                    with wp.ScopedCapture() as capture:
                        self.simulate()
                    self.graph = capture.graph
                    self._graph_key = key
                    wp.capture_launch(self.graph)
                except Exception as exc:
                    warnings.warn(f"CUDA graph capture failed; running uncaptured: {exc}", stacklevel=2)
                    self.use_cuda_graph = False
                    self.graph = None
                    self.simulate()
            else:
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def gui(self, ui):
        _, self.solver.fluid_viscosity = ui.slider_float("Viscosity", self.solver.fluid_viscosity, 0.0, 1.0, "%.2f")
        changed, cohesion = ui.slider_float("Cohesion", self.solver.fluid_cohesion, 0.0, 1.0, "%.2f")
        if changed:
            self.solver.fluid_cohesion = cohesion
        _, self.render_smoothing = ui.slider_float("Render Smoothing", self.render_smoothing, 0.0, 1.0, "%.2f")
        _, self.render_anisotropy_scale = ui.slider_float(
            "Anisotropy Scale", self.render_anisotropy_scale, 0.0, 3.0, "%.2f"
        )
        _, self.foam_stretch = ui.slider_float("Foam Stretch", self.foam_stretch, 0.0, 4.0, "%.2f")
        _, self.foam_radius = ui.slider_float("Foam Size", self.foam_radius, 0.001, 0.02, "%.3f")
        _, self.foam_opacity = ui.slider_float("Foam Opacity", self.foam_opacity, 0.0, 2.5, "%.2f")
        _, self.solver.diffuse_spawn_probability = ui.slider_float(
            "Foam Amount", self.solver.diffuse_spawn_probability, 0.0, 1.0, "%.2f"
        )
        _, self.solver.diffuse_lifetime = ui.slider_float(
            "Foam Lifetime", self.solver.diffuse_lifetime, 0.25, 6.0, "%.2f"
        )

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise ValueError("XPBD multi-fluid particles contain non-finite state")
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Rigid boxes contain non-finite transforms")
        tolerance = 1.0e-5
        if q[:, 2].min() < self.particle_radius - tolerance:
            raise ValueError("Fluid penetrated the tank floor")
        below_rim = q[:, 2] <= self.wall_height - self.particle_radius
        if np.any(np.abs(q[below_rim, 0]) > self.tank_half_x - self.particle_radius + tolerance) or np.any(
            np.abs(q[below_rim, 1]) > self.tank_half_y - self.particle_radius + tolerance
        ):
            raise ValueError("Fluid escaped the tank walls")
        margin = 0.3
        box_q = body_q[self.box_bodies]
        if np.any(np.abs(box_q[:, 0]) > self.tank_half_x + margin) or np.any(
            np.abs(box_q[:, 1]) > self.tank_half_y + margin
        ):
            raise ValueError("Boxes escaped the tank walls")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        show_fluid = getattr(self.viewer, "show_fluid", False)
        if show_fluid:
            self.viewer.show_fluid = False
        try:
            self.viewer.log_state(self.state_0)
        finally:
            if show_fluid:
                self.viewer.show_fluid = show_fluid

        if show_fluid and not self.viewer.show_particles:
            self._log_fluid_surface()
        else:
            self._hide_fluid_surface()
            if self.solver.diffuse_positions is not None:
                self.viewer.log_fluid_diffuse("/model/fluid/diffuse", None)
        self.viewer.end_frame()

    def _hide_fluid_surface(self):
        for segment in self.fluid_segments:
            self.viewer.log_fluid(f"/model/fluid/{segment.label}", None)

    def _log_fluid_surface(self):
        self.solver.update_render_particles(
            self.state_0,
            smoothing=self.render_smoothing,
            anisotropy_scale=self.render_anisotropy_scale,
        )
        for segment in self.fluid_segments:
            sl = slice(segment.start, segment.start + segment.count)
            self.viewer.log_fluid(
                f"/model/fluid/{segment.label}",
                self.solver.render_positions[sl],
                radii=self.model.particle_radius[sl],
                radius_scale=segment.radius_scale,
                color=segment.color,
                absorption=segment.absorption,
                ior=segment.ior,
                reflectance=segment.reflectance,
                specular_intensity=segment.specular_intensity,
                specular_power=segment.specular_power,
                blur_radius_world=segment.blur_radius_world,
                anisotropy=self.solver.render_anisotropy[sl],
                anisotropy_secondary=self.solver.render_anisotropy_secondary[sl],
                anisotropy_tertiary=self.solver.render_anisotropy_tertiary[sl],
                hidden=False,
            )
        if getattr(self.viewer, "show_fluid_diffuse", False) and self.solver.diffuse_positions is not None:
            self.viewer.log_fluid_diffuse(
                "/model/fluid/diffuse",
                self.solver.diffuse_positions,
                self.solver.diffuse_velocities,
                radius=self.foam_radius,
                color=(0.96, 0.96, 0.90, self.foam_opacity),
                motion_blur_scale=self.foam_stretch,
                lifetime=self.solver.diffuse_lifetime,
                surface_bias=0.045,
                hidden=False,
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=2)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")

        parser.add_argument("--tank-size", type=float, nargs=3, default=(1.8, 1.2, 0.6))
        parser.add_argument("--wall-opacity", type=float, default=0.3)
        parser.add_argument(
            "--particle-count",
            type=parse_particle_count,
            default=95_000,
            help="Target total fluid particle count; spacing and grid dimensions are derived automatically.",
        )
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--fluid-count", type=int, choices=(2, 3), default=3)
        parser.add_argument("--fluid-density-factors", type=float, nargs="+", default=(1.0, 1.35, 0.82))
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--box-count", type=int, default=5)
        parser.add_argument("--box-half-extent", type=float, default=0.085)
        parser.add_argument(
            "--box-density-fractions",
            type=float,
            nargs="+",
            default=(0.30, 0.55, 0.85, 0.42, 1.60),
            help="Box densities as fractions of the base fluid rest density; values above 1 tend to sink.",
        )

        parser.add_argument("--cohesion", type=float, default=0.55)
        parser.add_argument("--viscosity", type=float, default=0.06)
        parser.add_argument("--pick-stiffness", type=float, default=160.0)
        parser.add_argument("--pick-damping", type=float, default=30.0)

        parser.add_argument("--foam-max-particles", type=int, default=30000)
        parser.add_argument("--foam-lifetime", type=float, default=0.65)
        parser.add_argument("--foam-spawn-probability", type=float, default=1.0)
        parser.add_argument("--foam-buoyancy", type=float, default=1.0)
        parser.add_argument("--foam-radius", type=float, default=0.009)
        parser.add_argument("--foam-stretch", type=float, default=1.25, help="Foam velocity elongation factor.")
        parser.add_argument("--foam-opacity", type=float, default=1.6)
        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument("--render-anisotropy-scale", type=float, default=1.0)

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.7, -1.8, 1.55))
        parser.add_argument("--camera-pitch", type=float, default=-33.0)
        parser.add_argument("--camera-yaw", type=float, default=132.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
