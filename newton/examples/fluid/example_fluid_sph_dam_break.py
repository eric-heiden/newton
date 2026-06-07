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
            velocity_damping=args.velocity_damping,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
            boundary_damping=args.boundary_damping,
            max_velocity=args.max_velocity,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = args.render_mode == "particles"
        self.viewer.show_fluid = args.render_mode == "fluid"
        self.viewer.fluid_color = tuple(args.fluid_color)
        self.viewer.fluid_deep_color = tuple(args.fluid_deep_color)
        self.viewer.fluid_color_gradient_strength = args.fluid_color_gradient_strength
        self.viewer.fluid_opacity = args.fluid_opacity
        self.viewer.fluid_radius_scale = args.fluid_radius_scale
        self.viewer.fluid_thickness_scale = args.fluid_thickness_scale
        self.viewer.fluid_smoothing_iterations = args.fluid_smoothing_iterations
        self.viewer.fluid_smoothing_radius = args.fluid_smoothing_radius
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
        self.viewer.fluid_foam_strength = args.fluid_foam_strength
        self.viewer.fluid_foam_scale = args.fluid_foam_scale
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
        self.viewer.log_state(self.state_0)
        if self.show_bounds:
            self._log_bounds()
        else:
            self.viewer.log_lines("/fluid/bounds", None, None, None)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.viewer.show_fluid = imgui.checkbox("Fluid Surface", self.viewer.show_fluid)
        if self.viewer.show_fluid:
            self.viewer.show_particles = False
        _changed, self.viewer.show_particles = imgui.checkbox("Raw Particles", self.viewer.show_particles)
        if self.viewer.show_particles:
            self.viewer.show_fluid = False

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
        env_path = getattr(renderer, "_env_path", None)
        if env_path is not None and hasattr(renderer, "set_environment_map"):
            renderer.set_environment_map(env_path, intensity=args.environment_intensity)
            renderer._env_path = None

        if args.beach_lighting:
            renderer.sky_upper = (0.48, 0.74, 1.0)
            renderer.sky_lower = (0.68, 0.76, 0.78)
            renderer.ambient_sky = (0.86, 0.92, 1.0)
            renderer.ambient_ground = (0.56, 0.50, 0.36)
            renderer.exposure = 1.12
            renderer.specular_scale = 1.35

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")
        parser.add_argument("--capture-graph", action="store_true", help="Capture the SPH substeps in a CUDA graph.")
        parser.add_argument("--show-bounds", action=argparse.BooleanOptionalAction, default=True)
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
        parser.add_argument("--velocity-damping", type=float, default=0.02)
        parser.add_argument("--boundary-damping", type=float, default=0.15)
        parser.add_argument("--max-velocity", type=float, default=4.0)
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--bounds-lower", type=float, nargs=3, default=(-0.75, -0.45, 0.0))
        parser.add_argument("--bounds-upper", type=float, nargs=3, default=(0.75, 0.45, 1.0))
        parser.add_argument("--ground-color", type=float, nargs=3, default=(0.68, 0.61, 0.44))
        parser.add_argument("--environment-intensity", type=float, default=1.35)

        parser.add_argument("--fluid-color", type=float, nargs=3, default=(0.10, 0.98, 0.92))
        parser.add_argument("--fluid-deep-color", type=float, nargs=3, default=(0.0, 0.13, 0.58))
        parser.add_argument("--fluid-color-gradient-strength", type=float, default=0.88)
        parser.add_argument("--fluid-opacity", type=float, default=0.64)
        parser.add_argument("--fluid-radius-scale", type=float, default=1.55)
        parser.add_argument("--fluid-thickness-scale", type=float, default=1.8)
        parser.add_argument("--fluid-smoothing-iterations", type=int, default=8)
        parser.add_argument("--fluid-smoothing-radius", type=float, default=2.0)
        parser.add_argument("--fluid-reflection-strength", type=float, default=0.14)
        parser.add_argument("--fluid-refraction-strength", type=float, default=0.055)
        parser.add_argument("--fluid-env-map-strength", type=float, default=0.52)
        parser.add_argument("--fluid-env-reflection-lod", type=float, default=0.0)
        parser.add_argument("--fluid-env-color-preserve", type=float, default=0.85)
        parser.add_argument("--fluid-absorption-strength", type=float, default=1.55)
        parser.add_argument("--fluid-depth-visualization-strength", type=float, default=0.55)
        parser.add_argument("--fluid-caustic-strength", type=float, default=0.78)
        parser.add_argument("--fluid-caustic-scale", type=float, default=155.0)
        parser.add_argument("--fluid-floor-caustic-strength", type=float, default=0.65)
        parser.add_argument("--fluid-foam-strength", type=float, default=0.12)
        parser.add_argument("--fluid-foam-scale", type=float, default=55.0)

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.15, -1.35, 0.78))
        parser.add_argument("--camera-pitch", type=float, default=-20.0)
        parser.add_argument("--camera-yaw", type=float, default=126.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
