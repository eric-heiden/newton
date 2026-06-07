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

ParticleFlags = newton.ParticleFlags


@wp.func
def _sign_nonzero(x: float) -> float:
    result = float(1.0)
    if x < 0.0:
        result = -1.0
    return result


@wp.kernel
def apply_body_suspension_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_ids: wp.array[int],
    target_heights: wp.array[float],
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    stiffness: float,
    damping: float,
    wall_stiffness: float,
    wall_damping: float,
):
    tid = wp.tid()
    body = body_ids[tid]
    if body < 0:
        return

    x = wp.transform_get_translation(body_q[body])
    v = wp.spatial_top(body_qd[body])
    force = wp.vec3(0.0)
    mass = body_mass[body]

    force[2] += mass * (stiffness * (target_heights[tid] - x[2]) - damping * v[2])

    margin = float(0.18)
    if x[0] < bounds_lower[0] + margin:
        force[0] += wall_stiffness * (bounds_lower[0] + margin - x[0]) - wall_damping * v[0]
    elif x[0] > bounds_upper[0] - margin:
        force[0] += wall_stiffness * (bounds_upper[0] - margin - x[0]) - wall_damping * v[0]

    if x[1] < bounds_lower[1] + margin:
        force[1] += wall_stiffness * (bounds_lower[1] + margin - x[1]) - wall_damping * v[1]
    elif x[1] > bounds_upper[1] - margin:
        force[1] += wall_stiffness * (bounds_upper[1] - margin - x[1]) - wall_damping * v[1]

    if x[2] < bounds_lower[2] + margin:
        force[2] += wall_stiffness * (bounds_lower[2] + margin - x[2]) - wall_damping * v[2]
    elif x[2] > bounds_upper[2] - margin:
        force[2] += wall_stiffness * (bounds_upper[2] - margin - x[2]) - wall_damping * v[2]

    wp.atomic_add(body_f, body, wp.spatial_vector(force, wp.vec3(0.0)))


@wp.kernel
def apply_particle_box_coupling(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    box_body_ids: wp.array[int],
    box_half_extents: wp.array[wp.vec3],
    contact_distance: float,
    stiffness: float,
    damping: float,
    splash_velocity_gain: float,
):
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_q[i]
    v = particle_qd[i]
    radius = particle_radius[i]
    particle_force = wp.vec3(0.0)

    for box_idx in range(box_body_ids.shape[0]):
        body = box_body_ids[box_idx]
        X_wb = body_q[body]
        X_bw = wp.transform_inverse(X_wb)
        local = wp.transform_point(X_bw, x)
        half = box_half_extents[box_idx]

        cx = wp.min(wp.max(local[0], -half[0]), half[0])
        cy = wp.min(wp.max(local[1], -half[1]), half[1])
        cz = wp.min(wp.max(local[2], -half[2]), half[2])
        closest_local = wp.vec3(cx, cy, cz)
        delta_local = local - closest_local
        dist = wp.length(delta_local)
        normal_local = wp.vec3(0.0)
        penetration = radius + contact_distance - dist

        inside = (
            local[0] >= -half[0]
            and local[0] <= half[0]
            and local[1] >= -half[1]
            and local[1] <= half[1]
            and local[2] >= -half[2]
            and local[2] <= half[2]
        )
        if inside:
            dx = half[0] - wp.abs(local[0])
            dy = half[1] - wp.abs(local[1])
            dz = half[2] - wp.abs(local[2])
            if dx <= dy and dx <= dz:
                normal_local = wp.vec3(_sign_nonzero(local[0]), 0.0, 0.0)
                closest_local = wp.vec3(_sign_nonzero(local[0]) * half[0], local[1], local[2])
                penetration = radius + contact_distance + dx
            elif dy <= dz:
                normal_local = wp.vec3(0.0, _sign_nonzero(local[1]), 0.0)
                closest_local = wp.vec3(local[0], _sign_nonzero(local[1]) * half[1], local[2])
                penetration = radius + contact_distance + dy
            else:
                normal_local = wp.vec3(0.0, 0.0, _sign_nonzero(local[2]))
                closest_local = wp.vec3(local[0], local[1], _sign_nonzero(local[2]) * half[2])
                penetration = radius + contact_distance + dz
        elif dist > 1.0e-6:
            normal_local = delta_local / dist

        if penetration > 0.0:
            normal = wp.quat_rotate(wp.transform_get_rotation(X_wb), normal_local)
            closest_world = wp.transform_point(X_wb, closest_local)
            body_v = wp.spatial_top(body_qd[body])
            rel_n = wp.dot(v - body_v, normal)
            magnitude = stiffness * penetration - damping * rel_n
            if magnitude > 0.0:
                force = normal * magnitude
                particle_force += force
                body_force = -force * splash_velocity_gain
                r = closest_world - wp.transform_point(X_wb, body_com[body])
                wp.atomic_add(body_f, body, wp.spatial_vector(body_force, wp.cross(r, body_force)))

    particle_f[i] = particle_f[i] + particle_force


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
        self.show_bounds = args.show_bounds
        self.particle_radius = args.radius
        self.coupling_stiffness = args.coupling_stiffness
        self.coupling_damping = args.coupling_damping
        self.suspension_stiffness = args.suspension_stiffness
        self.suspension_damping = args.suspension_damping
        self.splash_velocity_gain = args.splash_velocity_gain
        self.pick_stiffness = args.pick_stiffness
        self.pick_damping = args.pick_damping
        self.show_box_guides = args.show_box_guides

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.default_particle_radius = args.radius
        builder.default_shape_cfg.mu = 0.25

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

        self.box_body_ids: list[int] = []
        self.box_half_extents: list[wp.vec3] = []
        self.box_target_heights: list[float] = []
        self.box_guide_colors: list[wp.vec3] = []
        self._add_boxes(builder, args)

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.box_body_ids_wp = wp.array(self.box_body_ids, dtype=int, device=self.model.device)
        self.box_half_extents_wp = wp.array(self.box_half_extents, dtype=wp.vec3, device=self.model.device)
        self.box_target_heights_wp = wp.array(self.box_target_heights, dtype=float, device=self.model.device)

        self.sph_solver = SolverSPH(
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
        self.rigid_integrator = newton.solvers.SolverSemiImplicit(self.model, angular_damping=args.angular_damping)

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        self._apply_picking_params()
        self.viewer.show_particles = args.render_mode == "particles"
        self.viewer.show_fluid = args.render_mode == "fluid"
        self._apply_fluid_args(args)
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)
        self._configure_render_environment(args)
        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 0.35

        self.graph = None
        self.capture_graph = args.capture_graph
        self.capture()

    def _add_boxes(self, builder: newton.ModelBuilder, args):
        spacing = 0.32
        colors = (
            (1.0, 0.78, 0.05),
            (0.10, 0.88, 0.35),
            (0.95, 0.18, 0.26),
            (0.20, 0.50, 1.0),
            (0.82, 0.35, 1.0),
        )
        for i in range(args.box_count):
            column = i % 3
            row = i // 3
            x = (column - 1) * spacing
            y = -0.18 + row * 0.34
            z = args.box_height + 0.035 * ((i % 2) * 2 - 1)
            hx = args.box_half_extent * (1.0 + 0.14 * (i % 2))
            hy = args.box_half_extent * (0.85 + 0.10 * ((i + 1) % 3))
            hz = args.box_half_extent * (0.95 + 0.10 * (i % 3))
            q = wp.quat_from_axis_angle(wp.vec3(0.2, 0.8, 0.1), 0.20 * float(i))
            body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), q), mass=args.box_mass, label=f"water_cube_{i}")
            builder.add_shape_box(
                body,
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=newton.ModelBuilder.ShapeConfig(density=args.box_density, mu=0.18),
                color=colors[i % len(colors)],
            )
            self.box_body_ids.append(body)
            self.box_half_extents.append(wp.vec3(hx, hy, hz))
            self.box_target_heights.append(z)
            self.box_guide_colors.append(wp.vec3(colors[i % len(colors)]))

    def _apply_fluid_args(self, args):
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

    def _configure_render_environment(self, args):
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is None:
            return

        renderer._env_intensity = float(args.environment_intensity)
        env_path = getattr(renderer, "_env_path", None)
        if env_path is not None and hasattr(renderer, "set_environment_map"):
            renderer.set_environment_map(env_path, intensity=args.environment_intensity)
            renderer._env_path = None

        renderer.sky_upper = (0.42, 0.74, 1.0)
        renderer.sky_lower = (0.72, 0.82, 0.78)
        renderer.ambient_sky = (0.92, 0.96, 1.0)
        renderer.ambient_ground = (0.66, 0.58, 0.40)
        renderer.exposure = args.exposure
        renderer.specular_scale = args.specular_scale
        renderer.diffuse_scale = args.diffuse_scale

    def _apply_picking_params(self):
        picking = getattr(self.viewer, "picking", None)
        if picking is None:
            return
        picking.pick_stiffness = float(self.pick_stiffness)
        picking.pick_damping = float(self.pick_damping)
        state = picking.pick_state.numpy()
        state[0]["pick_stiffness"] = float(self.pick_stiffness)
        state[0]["pick_damping"] = float(self.pick_damping)
        picking.pick_state.assign(state)

    def capture(self):
        self.graph = None
        if not self.capture_graph:
            return
        if not wp.get_device().is_cuda:
            warnings.warn("SPH interactive graph capture is only available on CUDA devices.", stacklevel=2)
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        except Exception as exc:
            warnings.warn(f"Interactive SPH graph capture failed; falling back to uncaptured stepping: {exc}", stacklevel=2)
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                kernel=apply_body_suspension_forces,
                dim=len(self.box_body_ids),
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.model.body_mass,
                    self.box_body_ids_wp,
                    self.box_target_heights_wp,
                    self.bounds_lower,
                    self.bounds_upper,
                    self.suspension_stiffness,
                    self.suspension_damping,
                    90.0,
                    9.0,
                ],
                device=self.model.device,
            )
            wp.launch(
                kernel=apply_particle_box_coupling,
                dim=self.model.particle_count,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.state_0.particle_f,
                    self.model.particle_radius,
                    self.model.particle_flags,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.model.body_com,
                    self.box_body_ids_wp,
                    self.box_half_extents_wp,
                    self.particle_radius * 1.75,
                    self.coupling_stiffness,
                    self.coupling_damping,
                    self.splash_velocity_gain,
                ],
                device=self.model.device,
            )
            self.sph_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.rigid_integrator.integrate_bodies(self.model, self.state_0, self.state_1, self.sim_dt, 0.03)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.show_box_guides:
            self._log_box_guides()
        if self.show_bounds:
            self._log_bounds()
        else:
            self.viewer.log_lines("/fluid/bounds", None, None, None)
        self.viewer.end_frame()

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
        colors = wp.full(len(edges), value=wp.vec3(0.18, 0.72, 0.94), dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/fluid/bounds", starts, ends, colors)

    def _log_box_guides(self):
        edges = (
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
        )
        body_q = self.state_0.body_q.numpy()
        starts = []
        ends = []
        colors = []
        for box_idx, body in enumerate(self.box_body_ids):
            q = body_q[body]
            p = q[:3]
            quat = q[3:7]
            x, y, z, w = quat
            rot = np.array(
                [
                    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                    [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                    [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
                ],
                dtype=np.float32,
            )
            h = self.box_half_extents[box_idx]
            corners = np.array(
                [
                    [-h[0], -h[1], -h[2]],
                    [h[0], -h[1], -h[2]],
                    [h[0], h[1], -h[2]],
                    [-h[0], h[1], -h[2]],
                    [-h[0], -h[1], h[2]],
                    [h[0], -h[1], h[2]],
                    [h[0], h[1], h[2]],
                    [-h[0], h[1], h[2]],
                ],
                dtype=np.float32,
            )
            world = corners @ rot.T + p
            for i, j in edges:
                starts.append(wp.vec3(world[i]))
                ends.append(wp.vec3(world[j]))
                colors.append(self.box_guide_colors[box_idx])

        self.viewer.log_lines(
            "/fluid/box_guides",
            wp.array(starts, dtype=wp.vec3, device=self.model.device),
            wp.array(ends, dtype=wp.vec3, device=self.model.device),
            wp.array(colors, dtype=wp.vec3, device=self.model.device),
        )

    def gui(self, ui):
        _changed, self.viewer.show_fluid = ui.checkbox("Fluid Surface", self.viewer.show_fluid)
        if self.viewer.show_fluid:
            self.viewer.show_particles = False
        _changed, self.viewer.show_particles = ui.checkbox("Raw Particles", self.viewer.show_particles)
        if self.viewer.show_particles:
            self.viewer.show_fluid = False

        ui.separator()
        ui.text("Water Shader")
        self.viewer.fluid_color = self._slider_color(ui, "Shallow Color", self.viewer.fluid_color)
        self.viewer.fluid_deep_color = self._slider_color(ui, "Deep Color", self.viewer.fluid_deep_color)
        _, self.viewer.fluid_color_gradient_strength = ui.slider_float(
            "Color Gradient", self.viewer.fluid_color_gradient_strength, 0.0, 1.0, "%.2f"
        )
        _, self.viewer.fluid_opacity = ui.slider_float("Opacity", self.viewer.fluid_opacity, 0.05, 1.0, "%.2f")
        _, self.viewer.fluid_radius_scale = ui.slider_float("Radius Scale", self.viewer.fluid_radius_scale, 0.2, 4.0, "%.2f")
        _, self.viewer.fluid_thickness_scale = ui.slider_float(
            "Thickness Scale", self.viewer.fluid_thickness_scale, 0.1, 6.0, "%.2f"
        )
        _, self.viewer.fluid_smoothing_iterations = ui.slider_int(
            "Smoothing Iterations", self.viewer.fluid_smoothing_iterations, 0, 30
        )
        _, self.viewer.fluid_smoothing_radius = ui.slider_float(
            "Smoothing Radius", self.viewer.fluid_smoothing_radius, 0.2, 6.0, "%.2f"
        )
        _, self.viewer.fluid_absorption_strength = ui.slider_float(
            "Depth Absorption", self.viewer.fluid_absorption_strength, 0.0, 5.0, "%.2f"
        )
        _, self.viewer.fluid_depth_visualization_strength = ui.slider_float(
            "Depth Cues", self.viewer.fluid_depth_visualization_strength, 0.0, 2.5, "%.2f"
        )
        _, self.viewer.fluid_reflection_strength = ui.slider_float(
            "Screen Reflection", self.viewer.fluid_reflection_strength, 0.0, 0.6, "%.3f"
        )
        _, self.viewer.fluid_refraction_strength = ui.slider_float(
            "Refraction", self.viewer.fluid_refraction_strength, 0.0, 0.18, "%.3f"
        )
        _, self.viewer.fluid_env_map_strength = ui.slider_float(
            "Env Reflection", self.viewer.fluid_env_map_strength, 0.0, 2.0, "%.2f"
        )
        _, self.viewer.fluid_env_reflection_lod = ui.slider_float(
            "Env Reflection LOD", self.viewer.fluid_env_reflection_lod, 0.0, 5.0, "%.2f"
        )
        _, self.viewer.fluid_env_color_preserve = ui.slider_float(
            "Env Color Preserve", self.viewer.fluid_env_color_preserve, 0.0, 1.0, "%.2f"
        )
        _, self.viewer.fluid_caustic_strength = ui.slider_float(
            "Surface Caustics", self.viewer.fluid_caustic_strength, 0.0, 4.0, "%.2f"
        )
        _, self.viewer.fluid_floor_caustic_strength = ui.slider_float(
            "Floor Caustics", self.viewer.fluid_floor_caustic_strength, 0.0, 5.0, "%.2f"
        )
        _, self.viewer.fluid_caustic_scale = ui.slider_float(
            "Caustic Scale", self.viewer.fluid_caustic_scale, 20.0, 420.0, "%.1f"
        )
        _, self.viewer.fluid_foam_strength = ui.slider_float("Foam", self.viewer.fluid_foam_strength, 0.0, 1.2, "%.2f")
        _, self.viewer.fluid_foam_scale = ui.slider_float("Foam Scale", self.viewer.fluid_foam_scale, 5.0, 160.0, "%.1f")

        ui.separator()
        ui.text("Interaction")
        _, self.coupling_stiffness = ui.slider_float("Water-Box Stiffness", self.coupling_stiffness, 0.0, 2200.0, "%.1f")
        _, self.coupling_damping = ui.slider_float("Water-Box Damping", self.coupling_damping, 0.0, 80.0, "%.1f")
        _, self.splash_velocity_gain = ui.slider_float("Splash Gain", self.splash_velocity_gain, 0.0, 4.0, "%.2f")
        _, self.suspension_stiffness = ui.slider_float("Suspension", self.suspension_stiffness, 0.0, 120.0, "%.1f")
        _, self.suspension_damping = ui.slider_float("Suspension Damping", self.suspension_damping, 0.0, 35.0, "%.1f")
        changed_stiff, self.pick_stiffness = ui.slider_float("Pick Stiffness", self.pick_stiffness, 0.0, 250.0, "%.1f")
        changed_damp, self.pick_damping = ui.slider_float("Pick Damping", self.pick_damping, 0.0, 60.0, "%.1f")
        if changed_stiff or changed_damp:
            self._apply_picking_params()

        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            ui.separator()
            ui.text("Render Environment")
            _, renderer._env_intensity = ui.slider_float("Env Intensity", renderer._env_intensity, 0.0, 4.0, "%.2f")
            _, renderer.exposure = ui.slider_float("Exposure", renderer.exposure, 0.2, 2.5, "%.2f")
            _, renderer.specular_scale = ui.slider_float("Specular Scale", renderer.specular_scale, 0.0, 4.0, "%.2f")

    @staticmethod
    def _slider_color(ui, label: str, value: tuple[float, float, float]) -> tuple[float, float, float]:
        changed, color = ui.slider_float3(label, [float(value[0]), float(value[1]), float(value[2])], 0.0, 1.0, "%.2f")
        if not changed:
            return value
        return (float(color[0]), float(color[1]), float(color[2]))

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("SPH particles contain non-finite positions")
        if not np.all(np.isfinite(qd)):
            raise ValueError("SPH particles contain non-finite velocities")
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Rigid bodies contain non-finite transforms")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=5)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")
        parser.add_argument("--capture-graph", action="store_true", help="Capture fixed-parameter SPH substeps.")
        parser.add_argument("--show-bounds", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--show-box-guides",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Draw colored guide outlines around the pickable rigid bodies.",
        )

        parser.add_argument("--dim-x", type=int, default=28)
        parser.add_argument("--dim-y", type=int, default=18)
        parser.add_argument("--dim-z", type=int, default=14)
        parser.add_argument("--spacing", type=float, default=0.043)
        parser.add_argument("--radius", type=float, default=0.034)
        parser.add_argument("--jitter", type=float, default=0.0015)
        parser.add_argument("--emit-lower", type=float, nargs=3, default=(-0.58, -0.34, 0.08))
        parser.add_argument("--initial-velocity", type=float, nargs=3, default=(0.0, 0.0, 0.0))

        parser.add_argument("--smoothing-length", type=float, default=0.092)
        parser.add_argument("--rest-density", type=float, default=460.0)
        parser.add_argument("--gas-constant", type=float, default=70.0)
        parser.add_argument("--viscosity", type=float, default=0.10)
        parser.add_argument("--velocity-damping", type=float, default=0.015)
        parser.add_argument("--boundary-damping", type=float, default=0.20)
        parser.add_argument("--max-velocity", type=float, default=5.5)
        parser.add_argument("--gravity", type=float, default=-3.5)
        parser.add_argument("--bounds-lower", type=float, nargs=3, default=(-0.72, -0.48, 0.0))
        parser.add_argument("--bounds-upper", type=float, nargs=3, default=(0.72, 0.48, 0.88))
        parser.add_argument("--ground-color", type=float, nargs=3, default=(0.72, 0.63, 0.42))

        parser.add_argument("--box-count", type=int, default=5)
        parser.add_argument("--box-half-extent", type=float, default=0.105)
        parser.add_argument("--box-height", type=float, default=0.46)
        parser.add_argument("--box-mass", type=float, default=1.4)
        parser.add_argument("--box-density", type=float, default=65.0)
        parser.add_argument("--pick-stiffness", type=float, default=120.0)
        parser.add_argument("--pick-damping", type=float, default=18.0)
        parser.add_argument("--coupling-stiffness", type=float, default=760.0)
        parser.add_argument("--coupling-damping", type=float, default=28.0)
        parser.add_argument("--splash-velocity-gain", type=float, default=1.45)
        parser.add_argument("--suspension-stiffness", type=float, default=38.0)
        parser.add_argument("--suspension-damping", type=float, default=9.5)

        parser.add_argument("--fluid-color", type=float, nargs=3, default=(0.02, 0.96, 0.86))
        parser.add_argument("--fluid-deep-color", type=float, nargs=3, default=(0.0, 0.035, 0.36))
        parser.add_argument("--fluid-color-gradient-strength", type=float, default=0.96)
        parser.add_argument("--fluid-opacity", type=float, default=0.56)
        parser.add_argument("--fluid-radius-scale", type=float, default=2.35)
        parser.add_argument("--fluid-thickness-scale", type=float, default=2.55)
        parser.add_argument("--fluid-smoothing-iterations", type=int, default=18)
        parser.add_argument("--fluid-smoothing-radius", type=float, default=3.25)
        parser.add_argument("--fluid-reflection-strength", type=float, default=0.26)
        parser.add_argument("--fluid-refraction-strength", type=float, default=0.082)
        parser.add_argument("--fluid-env-map-strength", type=float, default=1.18)
        parser.add_argument("--fluid-env-reflection-lod", type=float, default=0.0)
        parser.add_argument("--fluid-env-color-preserve", type=float, default=0.92)
        parser.add_argument("--fluid-absorption-strength", type=float, default=2.65)
        parser.add_argument("--fluid-depth-visualization-strength", type=float, default=1.15)
        parser.add_argument("--fluid-caustic-strength", type=float, default=1.75)
        parser.add_argument("--fluid-caustic-scale", type=float, default=225.0)
        parser.add_argument("--fluid-floor-caustic-strength", type=float, default=2.25)
        parser.add_argument("--fluid-foam-strength", type=float, default=0.18)
        parser.add_argument("--fluid-foam-scale", type=float, default=38.0)

        parser.add_argument("--environment-intensity", type=float, default=2.25)
        parser.add_argument("--exposure", type=float, default=1.18)
        parser.add_argument("--diffuse-scale", type=float, default=1.05)
        parser.add_argument("--specular-scale", type=float, default=1.85)
        parser.add_argument("--angular-damping", type=float, default=0.04)
        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.18, -1.28, 0.74))
        parser.add_argument("--camera-pitch", type=float, default=-18.0)
        parser.add_argument("--camera-yaw", type=float, default=132.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
