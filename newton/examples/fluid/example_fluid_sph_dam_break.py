# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSPH


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        self.bounds_lower = wp.vec3(args.bounds_lower)
        self.bounds_upper = wp.vec3(args.bounds_upper)
        self.particle_radius = args.radius
        self.show_bounds = args.show_bounds
        self.diffuse_particle_radius = args.fluid_diffuse_radius
        self.diffuse_alpha = args.fluid_diffuse_alpha
        self.diffuse_motion_blur_scale = args.fluid_diffuse_motion_blur

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.default_particle_radius = args.radius

        mass = args.rest_density * args.spacing**3
        builder.add_particle_grid(
            pos=wp.vec3(args.emit_lower),
            rot=wp.quat_identity(),
            vel=wp.vec3(args.initial_velocity),
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            dim_z=args.dim_z,
            cell_x=args.spacing,
            cell_y=args.spacing,
            cell_z=args.spacing,
            mass=mass,
            jitter=args.jitter,
            radius_mean=args.radius,
            radius_std=0.0,
        )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0), color=tuple(args.ground_color))

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver = SolverSPH(
            self.model,
            smoothing_length=args.smoothing_length,
            rest_density=args.rest_density,
            gas_constant=args.gas_constant,
            viscosity=args.viscosity,
            particle_friction=args.particle_friction,
            particle_collision_margin=args.particle_collision_margin,
            cohesion=args.cohesion,
            surface_tension=args.surface_tension,
            vorticity_confinement=args.vorticity_confinement,
            solid_pressure=args.solid_pressure,
            buoyancy=args.buoyancy,
            xsph_strength=args.xsph_strength,
            free_surface_drag=args.free_surface_drag,
            dissipation=args.dissipation,
            velocity_damping=args.velocity_damping,
            sleep_threshold=args.sleep_threshold,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
            boundary_damping=args.boundary_damping,
            shape_collision_distance=args.shape_collision_distance,
            shape_collision_margin=args.shape_collision_margin,
            shape_restitution=args.shape_restitution,
            shape_friction=args.shape_friction,
            shape_adhesion=args.shape_adhesion,
            max_velocity=args.max_velocity,
            max_acceleration=args.max_acceleration,
            pbf_iterations=args.pbf_iterations,
            pbf_relaxation=args.pbf_relaxation,
            pbf_artificial_pressure=args.pbf_artificial_pressure,
            max_diffuse_particles=args.fluid_diffuse_max_particles,
            diffuse_threshold=args.fluid_diffuse_threshold,
            diffuse_lifetime=args.fluid_diffuse_lifetime,
            diffuse_drag=args.fluid_diffuse_drag,
            diffuse_buoyancy=args.fluid_diffuse_buoyancy,
            diffuse_ballistic=args.fluid_diffuse_ballistic,
            diffuse_spawn_probability=args.fluid_diffuse_spawn_probability,
            render_smoothing=args.fluid_render_smoothing,
            render_anisotropy_scale=args.fluid_render_anisotropy_scale,
            render_anisotropy_min=args.fluid_render_anisotropy_min,
            render_anisotropy_max=args.fluid_render_anisotropy_max,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = args.render_mode == "particles"
        self.viewer.show_fluid = args.render_mode == "fluid"
        self.viewer.show_fluid_diffuse = args.show_diffuse
        self.viewer.fluid_color = tuple(args.fluid_color)
        self.viewer.fluid_deep_color = tuple(args.fluid_deep_color)
        self.viewer.fluid_color_gradient_strength = args.fluid_color_gradient_strength
        self.viewer.fluid_opacity = args.fluid_opacity
        self.viewer.fluid_radius_scale = args.fluid_radius_scale
        self.viewer.fluid_thickness_scale = args.fluid_thickness_scale
        self.viewer.fluid_smoothing_iterations = args.fluid_smoothing_iterations
        self.viewer.fluid_smoothing_radius = args.fluid_smoothing_radius
        self.viewer.fluid_smoothing_depth_edge_falloff = args.fluid_smoothing_depth_edge_falloff
        self.viewer.fluid_smoothing_max_samples = args.fluid_smoothing_max_samples
        self.viewer.fluid_reflection_strength = args.fluid_reflection_strength
        self.viewer.fluid_refraction_strength = args.fluid_refraction_strength
        self.viewer.fluid_env_map_strength = args.fluid_env_map_strength
        self.viewer.fluid_env_reflection_lod = args.fluid_env_reflection_lod
        self.viewer.fluid_env_color_preserve = args.fluid_env_color_preserve
        self.viewer.fluid_absorption_strength = args.fluid_absorption_strength
        self.viewer.fluid_depth_visualization_strength = args.fluid_depth_visualization_strength
        self.viewer.fluid_caustic_strength = args.fluid_caustic_strength
        self.viewer.fluid_caustic_scale = args.fluid_caustic_scale
        self.viewer.fluid_floor_caustic_strength = args.fluid_floor_caustic_strength
        self.viewer.fluid_surface_shadow_strength = args.fluid_surface_shadow_strength
        self.viewer.fluid_foam_strength = args.fluid_foam_strength
        self.viewer.fluid_foam_scale = args.fluid_foam_scale
        self.viewer.fluid_diffuse_expansion = args.fluid_diffuse_expansion
        self.viewer.fluid_diffuse_inscatter = args.fluid_diffuse_inscatter
        self.viewer.fluid_diffuse_outscatter = args.fluid_diffuse_outscatter
        self.viewer.fluid_diffuse_shadow_strength = args.fluid_diffuse_shadow_strength
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)
        self._configure_render_environment(args)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.graph = None
        self.capture_graph = args.capture_graph
        self.capture()

    def capture(self):
        if not self.capture_graph:
            return
        if not wp.get_device().is_cuda:
            warnings.warn("SPH graph capture is only available on CUDA devices.", stacklevel=2)
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        except Exception as exc:
            warnings.warn(f"SPH graph capture failed; falling back to uncaptured stepping: {exc}", stacklevel=2)
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        lower = np.array(self.bounds_lower, dtype=np.float32)
        upper = np.array(self.bounds_upper, dtype=np.float32)
        radius = self.particle_radius
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()

        if not np.all(np.isfinite(q)):
            raise ValueError("SPH particles contain non-finite positions")
        if not np.all(np.isfinite(qd)):
            raise ValueError("SPH particles contain non-finite velocities")
        if np.any(q < lower - 2.0 * radius) or np.any(q > upper + 2.0 * radius):
            raise ValueError("SPH particles escaped the configured bounds")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        show_fluid = self.viewer.show_fluid
        if show_fluid:
            self.viewer.show_fluid = False
        try:
            self.viewer.log_state(self.state_0)
        finally:
            self.viewer.show_fluid = show_fluid
        self._log_fluid_surface()
        self._log_diffuse_particles()
        if self.show_bounds:
            self._log_bounds()
        else:
            self.viewer.log_lines("/fluid/bounds", None, None, None)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.viewer.show_fluid = imgui.checkbox("Fluid Surface", self.viewer.show_fluid)
        if self.viewer.show_fluid:
            self.viewer.show_particles = False
        _changed, self.viewer.show_fluid_diffuse = imgui.checkbox("Diffuse Spray", self.viewer.show_fluid_diffuse)
        _, self.viewer.fluid_diffuse_inscatter = imgui.slider_float(
            "Diffuse Inscatter", self.viewer.fluid_diffuse_inscatter, 0.0, 1.5, "%.2f"
        )
        _, self.viewer.fluid_diffuse_outscatter = imgui.slider_float(
            "Diffuse Outscatter", self.viewer.fluid_diffuse_outscatter, 0.0, 1.2, "%.2f"
        )
        _, self.viewer.fluid_diffuse_shadow_strength = imgui.slider_float(
            "Diffuse Shadows", self.viewer.fluid_diffuse_shadow_strength, 0.0, 1.2, "%.2f"
        )
        _, self.viewer.fluid_diffuse_expansion = imgui.slider_float(
            "Diffuse Expansion", self.viewer.fluid_diffuse_expansion, 0.0, 1.5, "%.2f"
        )
        _changed, self.viewer.show_particles = imgui.checkbox("Raw Particles", self.viewer.show_particles)
        if self.viewer.show_particles:
            self.viewer.show_fluid = False
        _, self.solver.particle_friction = imgui.slider_float(
            "Particle Friction", self.solver.particle_friction, 0.0, 8.0, "%.2f"
        )
        _, self.solver.particle_collision_margin = imgui.slider_float(
            "Particle Margin", self.solver.particle_collision_margin, 0.0, 0.05, "%.4f"
        )
        _, self.solver.shape_friction = imgui.slider_float(
            "Shape Friction", self.solver.shape_friction, 0.0, 4.0, "%.2f"
        )
        if self.solver.shape_collision_distance is None:
            self.solver.shape_collision_distance = self.particle_radius
        _, self.solver.shape_collision_distance = imgui.slider_float(
            "Shape Distance", self.solver.shape_collision_distance, 0.0, 0.10, "%.4f"
        )
        _, self.solver.shape_collision_margin = imgui.slider_float(
            "Shape Margin", self.solver.shape_collision_margin, 0.0, 0.05, "%.4f"
        )
        _, self.solver.shape_restitution = imgui.slider_float(
            "Shape Restitution", self.solver.shape_restitution or 0.0, 0.0, 1.0, "%.2f"
        )
        _, self.solver.shape_adhesion = imgui.slider_float(
            "Shape Adhesion", self.solver.shape_adhesion, 0.0, 4.0, "%.2f"
        )
        _, self.solver.dissipation = imgui.slider_float("Dissipation", self.solver.dissipation, 0.0, 8.0, "%.2f")
        _, self.solver.sleep_threshold = imgui.slider_float(
            "Sleep Threshold", self.solver.sleep_threshold, 0.0, 1.0, "%.2f"
        )
        _, self.solver.max_acceleration = imgui.slider_float(
            "Max Acceleration", self.solver.max_acceleration, 1.0, 300.0, "%.1f"
        )
        _, self.solver.render_smoothing = imgui.slider_float(
            "Render Smoothing", self.solver.render_smoothing, 0.0, 1.0, "%.2f"
        )
        _, self.solver.render_anisotropy_scale = imgui.slider_float(
            "Anisotropy Scale", self.solver.render_anisotropy_scale, 0.0, 3.0, "%.2f"
        )
        _, self.solver.render_anisotropy_min = imgui.slider_float(
            "Anisotropy Min", self.solver.render_anisotropy_min, 0.01, 1.0, "%.2f"
        )
        _, self.solver.render_anisotropy_max = imgui.slider_float(
            "Anisotropy Max", self.solver.render_anisotropy_max, 1.0, 4.0, "%.2f"
        )
        self.solver.render_anisotropy_max = max(self.solver.render_anisotropy_max, self.solver.render_anisotropy_min)

    def _log_fluid_surface(self):
        if (
            not self.viewer.show_fluid
            or getattr(self.viewer, "fluids", None) is None
            or not self.solver.render_buffers_valid
            or self.solver.render_positions is None
            or self.solver.render_anisotropy is None
            or self.solver.render_anisotropy_secondary is None
            or self.solver.render_anisotropy_tertiary is None
        ):
            return

        self.viewer.log_fluid(
            "/model/fluid",
            self.state_0.particle_q,
            radii=self.model.particle_radius,
            color=self.viewer.fluid_color,
            deep_color=self.viewer.fluid_deep_color,
            color_gradient_strength=self.viewer.fluid_color_gradient_strength,
            opacity=self.viewer.fluid_opacity,
            radius_scale=self.viewer.fluid_radius_scale,
            thickness_scale=self.viewer.fluid_thickness_scale,
            smoothing_iterations=self.viewer.fluid_smoothing_iterations,
            smoothing_radius=self.viewer.fluid_smoothing_radius,
            smoothing_depth_edge_falloff=self.viewer.fluid_smoothing_depth_edge_falloff,
            smoothing_max_samples=self.viewer.fluid_smoothing_max_samples,
            reflection_strength=self.viewer.fluid_reflection_strength,
            refraction_strength=self.viewer.fluid_refraction_strength,
            env_map_strength=self.viewer.fluid_env_map_strength,
            env_reflection_lod=self.viewer.fluid_env_reflection_lod,
            env_color_preserve=self.viewer.fluid_env_color_preserve,
            absorption_strength=self.viewer.fluid_absorption_strength,
            depth_visualization_strength=self.viewer.fluid_depth_visualization_strength,
            caustic_strength=self.viewer.fluid_caustic_strength,
            caustic_scale=self.viewer.fluid_caustic_scale,
            floor_caustic_strength=self.viewer.fluid_floor_caustic_strength,
            surface_shadow_strength=self.viewer.fluid_surface_shadow_strength,
            foam_strength=self.viewer.fluid_foam_strength,
            foam_scale=self.viewer.fluid_foam_scale,
            hidden=False,
            render_points=self.solver.render_positions,
            anisotropy=self.solver.render_anisotropy,
            anisotropy_secondary=self.solver.render_anisotropy_secondary,
            anisotropy_tertiary=self.solver.render_anisotropy_tertiary,
        )

    def _log_diffuse_particles(self):
        if not self.viewer.show_fluid or not self.viewer.show_fluid_diffuse or self.solver.diffuse_positions is None:
            self.viewer.log_fluid_diffuse("/model/fluid/diffuse", None, hidden=True)
            return

        foam_strength = max(self.viewer.fluid_foam_strength, 0.0)
        foam_radius_scale = max(self.viewer.fluid_foam_scale, 1.0) / 111.1
        self.viewer.log_fluid_diffuse(
            "/model/fluid/diffuse",
            self.solver.diffuse_positions,
            self.solver.diffuse_velocities,
            radius=self.diffuse_particle_radius * max(0.2, foam_radius_scale**0.5),
            color=self.viewer.fluid_diffuse_color,
            alpha=self.diffuse_alpha * foam_strength,
            motion_blur_scale=self.diffuse_motion_blur_scale,
            expansion=self.viewer.fluid_diffuse_expansion,
            inscatter=self.viewer.fluid_diffuse_inscatter,
            outscatter=self.viewer.fluid_diffuse_outscatter,
            shadow_strength=self.viewer.fluid_diffuse_shadow_strength,
            hidden=False,
        )

    def _log_bounds(self):
        lower = self.bounds_lower
        upper = self.bounds_upper
        corners = [
            wp.vec3(lower[0], lower[1], lower[2]),
            wp.vec3(upper[0], lower[1], lower[2]),
            wp.vec3(upper[0], upper[1], lower[2]),
            wp.vec3(lower[0], upper[1], lower[2]),
            wp.vec3(lower[0], lower[1], upper[2]),
            wp.vec3(upper[0], lower[1], upper[2]),
            wp.vec3(upper[0], upper[1], upper[2]),
            wp.vec3(lower[0], upper[1], upper[2]),
        ]
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        starts = wp.array([corners[i] for i, _j in edges], dtype=wp.vec3, device=self.model.device)
        ends = wp.array([corners[j] for _i, j in edges], dtype=wp.vec3, device=self.model.device)
        colors = wp.full(len(edges), value=wp.vec3(0.25, 0.55, 0.75), dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/fluid/bounds", starts, ends, colors)

    def _configure_render_environment(self, args):
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is None:
            return

        renderer._env_intensity = float(args.environment_intensity)
        renderer.fluid_shadow_size = args.fluid_shadow_size
        env_path = getattr(renderer, "_env_path", None)
        if env_path is not None and hasattr(renderer, "set_environment_map"):
            renderer.set_environment_map(env_path, intensity=args.environment_intensity)
            renderer._env_path = None

        sun = np.asarray(args.sun_direction, dtype=np.float32)
        sun_norm = float(np.linalg.norm(sun))
        if sun_norm > 1.0e-6:
            renderer._sun_direction = sun / sun_norm

        if args.beach_lighting:
            renderer.sky_upper = (0.48, 0.74, 1.0)
            renderer.sky_lower = (0.68, 0.76, 0.78)
            renderer.ambient_sky = (0.86, 0.92, 1.0)
            renderer.ambient_ground = (0.48, 0.50, 0.52)
            renderer.exposure = 1.08
            renderer.specular_scale = 4.00

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")
        parser.add_argument("--capture-graph", action="store_true", help="Capture the SPH substeps in a CUDA graph.")
        parser.add_argument("--show-bounds", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--show-diffuse", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--beach-lighting", action=argparse.BooleanOptionalAction, default=True)

        parser.add_argument("--dim-x", type=int, default=14)
        parser.add_argument("--dim-y", type=int, default=9)
        parser.add_argument("--dim-z", type=int, default=8)
        parser.add_argument("--spacing", type=float, default=0.055)
        parser.add_argument("--radius", type=float, default=0.045)
        parser.add_argument("--jitter", type=float, default=0.002)
        parser.add_argument("--emit-lower", type=float, nargs=3, default=(-0.62, -0.28, 0.08))
        parser.add_argument("--initial-velocity", type=float, nargs=3, default=(0.6, 0.0, 0.0))

        parser.add_argument("--smoothing-length", type=float, default=0.105)
        parser.add_argument("--rest-density", type=float, default=420.0)
        parser.add_argument("--gas-constant", type=float, default=45.0)
        parser.add_argument("--viscosity", type=float, default=0.07)
        parser.add_argument("--particle-friction", type=float, default=0.16)
        parser.add_argument("--particle-collision-margin", type=float, default=0.0023)
        parser.add_argument("--cohesion", type=float, default=0.08)
        parser.add_argument("--surface-tension", type=float, default=0.000001)
        parser.add_argument("--vorticity-confinement", type=float, default=0.00012)
        parser.add_argument("--solid-pressure", type=float, default=0.35)
        parser.add_argument("--buoyancy", type=float, default=1.0)
        parser.add_argument("--xsph-strength", type=float, default=0.08)
        parser.add_argument("--free-surface-drag", type=float, default=0.16)
        parser.add_argument("--dissipation", type=float, default=0.28)
        parser.add_argument("--velocity-damping", type=float, default=0.014)
        parser.add_argument("--sleep-threshold", type=float, default=0.0)
        parser.add_argument("--boundary-damping", type=float, default=0.15)
        parser.add_argument("--shape-collision-distance", type=float, default=0.045)
        parser.add_argument("--shape-collision-margin", type=float, default=0.0023)
        parser.add_argument("--shape-restitution", type=float, default=0.0)
        parser.add_argument("--shape-friction", type=float, default=0.16)
        parser.add_argument("--shape-adhesion", type=float, default=0.15)
        parser.add_argument("--max-velocity", type=float, default=5.0)
        parser.add_argument("--max-acceleration", type=float, default=105.0)
        parser.add_argument("--pbf-iterations", type=int, default=3)
        parser.add_argument("--pbf-relaxation", type=float, default=0.75)
        parser.add_argument("--pbf-artificial-pressure", type=float, default=0.002)
        parser.add_argument("--fluid-diffuse-max-particles", type=int, default=4096)
        parser.add_argument("--fluid-diffuse-threshold", type=float, default=1.2)
        parser.add_argument("--fluid-diffuse-lifetime", type=float, default=1.8)
        parser.add_argument("--fluid-diffuse-drag", type=float, default=0.95)
        parser.add_argument("--fluid-diffuse-buoyancy", type=float, default=0.18)
        parser.add_argument("--fluid-diffuse-ballistic", type=int, default=7)
        parser.add_argument("--fluid-diffuse-spawn-probability", type=float, default=0.22)
        parser.add_argument("--fluid-render-smoothing", type=float, default=0.45)
        parser.add_argument("--fluid-render-anisotropy-scale", type=float, default=0.82)
        parser.add_argument("--fluid-render-anisotropy-min", type=float, default=0.1)
        parser.add_argument("--fluid-render-anisotropy-max", type=float, default=2.0)
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--bounds-lower", type=float, nargs=3, default=(-0.75, -0.45, 0.0))
        parser.add_argument("--bounds-upper", type=float, nargs=3, default=(0.75, 0.45, 1.0))
        parser.add_argument("--ground-color", type=float, nargs=3, default=(0.54, 0.55, 0.53))
        parser.add_argument("--environment-intensity", type=float, default=3.15)
        parser.add_argument("--fluid-shadow-size", type=int, default=2048)

        parser.add_argument("--fluid-color", type=float, nargs=3, default=(0.10, 0.50, 0.80))
        parser.add_argument("--fluid-deep-color", type=float, nargs=3, default=(0.01, 0.09, 0.34))
        parser.add_argument("--fluid-color-gradient-strength", type=float, default=0.20)
        parser.add_argument("--fluid-opacity", type=float, default=1.00)
        parser.add_argument("--fluid-radius-scale", type=float, default=1.34)
        parser.add_argument("--fluid-thickness-scale", type=float, default=0.55)
        parser.add_argument("--fluid-smoothing-iterations", type=int, default=7)
        parser.add_argument("--fluid-smoothing-radius", type=float, default=3.83)
        parser.add_argument("--fluid-smoothing-depth-edge-falloff", type=float, default=5.5)
        parser.add_argument("--fluid-smoothing-max-samples", type=int, default=4)
        parser.add_argument("--fluid-reflection-strength", type=float, default=0.528)
        parser.add_argument("--fluid-refraction-strength", type=float, default=0.038)
        parser.add_argument("--fluid-env-map-strength", type=float, default=1.02)
        parser.add_argument("--fluid-env-reflection-lod", type=float, default=1.8)
        parser.add_argument("--fluid-env-color-preserve", type=float, default=0.57)
        parser.add_argument("--fluid-absorption-strength", type=float, default=1.2)
        parser.add_argument("--fluid-depth-visualization-strength", type=float, default=2.13)
        parser.add_argument("--fluid-caustic-strength", type=float, default=3.03)
        parser.add_argument("--fluid-caustic-scale", type=float, default=37.1)
        parser.add_argument("--fluid-floor-caustic-strength", type=float, default=1.15)
        parser.add_argument("--fluid-surface-shadow-strength", type=float, default=0.35)
        parser.add_argument("--fluid-foam-strength", type=float, default=0.99)
        parser.add_argument("--fluid-foam-scale", type=float, default=5.0)
        parser.add_argument("--fluid-diffuse-radius", type=float, default=0.016)
        parser.add_argument("--fluid-diffuse-alpha", type=float, default=0.36)
        parser.add_argument("--fluid-diffuse-motion-blur", type=float, default=0.12)
        parser.add_argument("--fluid-diffuse-expansion", type=float, default=1.23)
        parser.add_argument("--fluid-diffuse-inscatter", type=float, default=0.60)
        parser.add_argument("--fluid-diffuse-outscatter", type=float, default=0.70)
        parser.add_argument("--fluid-diffuse-shadow-strength", type=float, default=0.62)

        parser.add_argument("--sun-direction", type=float, nargs=3, default=(0.78, -0.56, 0.20))
        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.15, -1.35, 0.78))
        parser.add_argument("--camera-pitch", type=float, default=-20.0)
        parser.add_argument("--camera-yaw", type=float, default=126.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
