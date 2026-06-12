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


@wp.kernel
def deactivate_particles_inside_box(
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    box_xform: wp.transform,
    box_half_extents: wp.vec3,
    clearance: float,
):
    tid = wp.tid()
    local = wp.transform_point(wp.transform_inverse(box_xform), particle_q[tid])
    half = box_half_extents + wp.vec3(clearance)
    if wp.abs(local[0]) <= half[0] and wp.abs(local[1]) <= half[1] and wp.abs(local[2]) <= half[2]:
        particle_flags[tid] = wp.int32(0)


@wp.kernel
def drive_wave_paddle(
    paddle_time: wp.array[float],
    dt: float,
    paddle_body: int,
    base_pos: wp.vec3,
    amplitude: float,
    angular_frequency: float,
    start_ramp: float,
    body_q_0: wp.array[wp.transform],
    body_qd_0: wp.array[wp.spatial_vector],
    body_q_1: wp.array[wp.transform],
    body_qd_1: wp.array[wp.spatial_vector],
):
    # The clock lives on the device so the paddle keeps moving inside a
    # captured CUDA graph.
    t = paddle_time[0] + dt
    paddle_time[0] = t

    ramp = wp.min(t * start_ramp, 1.0)
    offset = amplitude * ramp * wp.sin(angular_frequency * t)
    velocity = amplitude * ramp * angular_frequency * wp.cos(angular_frequency * t)
    xform = wp.transform(base_pos + wp.vec3(offset, 0.0, 0.0), wp.quat_identity())
    qd = wp.spatial_vector(wp.vec3(velocity, 0.0, 0.0), wp.vec3(0.0))
    body_q_0[paddle_body] = xform
    body_qd_0[paddle_body] = qd
    body_q_1[paddle_body] = xform
    body_qd_1[paddle_body] = qd


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
        self.paddle_amplitude = args.paddle_amplitude
        self.paddle_frequency = 2.0 * np.pi / max(args.paddle_period, 1.0e-3)
        self.fluid_color = tuple(args.fluid_color)
        self.fluid_ior = args.fluid_ior
        self.fluid_blur_radius = args.fluid_blur_radius
        self.fluid_radius_scale = args.fluid_radius_scale
        self.foam_color = tuple(args.foam_color)
        self.foam_radius = args.foam_radius
        self.foam_motion_blur = args.foam_motion_blur
        self.foam_lifetime = args.fluid_diffuse_lifetime

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.default_particle_radius = args.radius

        mass = args.rest_density * args.spacing**3
        builder.add_particle_grid(
            pos=wp.vec3(args.emit_lower),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
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
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.2), color=tuple(args.ground_color))

        # Sloped beach the waves run up and break on.
        beach_angle = float(np.deg2rad(args.beach_angle_deg))
        beach_length = args.beach_length
        beach_half_thickness = 0.10
        beach_start_x = args.beach_start
        beach_center = wp.vec3(
            beach_start_x + 0.5 * beach_length * np.cos(beach_angle),
            0.0,
            0.5 * beach_length * np.sin(beach_angle) - beach_half_thickness / np.cos(beach_angle),
        )
        beach_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -beach_angle)
        self.beach_xform = wp.transform(beach_center, beach_q)
        self.beach_half_extents = wp.vec3(
            0.5 * beach_length,
            0.5 * (self.bounds_upper[1] - self.bounds_lower[1]) + 0.2,
            beach_half_thickness,
        )
        builder.add_shape_box(
            -1,
            xform=self.beach_xform,
            hx=self.beach_half_extents[0],
            hy=self.beach_half_extents[1],
            hz=self.beach_half_extents[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.4),
            color=(0.83, 0.74, 0.55),
            label="beach",
        )

        # Kinematic paddle at the deep end; its transform is animated directly.
        self.paddle_base_pos = wp.vec3(args.paddle_x, 0.0, args.paddle_height)
        self.paddle_half_extents = wp.vec3(
            0.06,
            0.5 * (self.bounds_upper[1] - self.bounds_lower[1]) + 0.1,
            args.paddle_height + 0.1,
        )
        self.paddle_body = builder.add_body(
            xform=wp.transform(self.paddle_base_pos, wp.quat_identity()),
            is_kinematic=True,
            label="wave_paddle",
        )
        builder.add_shape_box(
            self.paddle_body,
            hx=self.paddle_half_extents[0],
            hy=self.paddle_half_extents[1],
            hz=self.paddle_half_extents[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.1),
            color=(0.35, 0.38, 0.42),
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.paddle_time = wp.zeros(1, dtype=float, device=self.model.device)

        wp.launch(
            kernel=deactivate_particles_inside_box,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self.model.particle_flags,
                self.beach_xform,
                self.beach_half_extents,
                args.radius,
            ],
            device=self.model.device,
        )

        self.sph_solver = SolverSPH(
            self.model,
            smoothing_length=args.smoothing_length,
            rest_density=args.rest_density,
            viscosity=args.viscosity,
            particle_friction=args.particle_friction,
            cohesion=args.cohesion,
            surface_tension=args.surface_tension,
            xsph_strength=args.xsph_strength,
            free_surface_drag=args.free_surface_drag,
            dissipation=args.dissipation,
            velocity_damping=args.velocity_damping,
            sleep_threshold=args.sleep_threshold,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
            boundary_damping=args.boundary_damping,
            shape_collision_distance=args.radius,
            shape_friction=args.shape_friction,
            shape_adhesion=args.shape_adhesion,
            max_velocity=args.max_velocity,
            max_acceleration=args.max_acceleration,
            pbf_iterations=args.pbf_iterations,
            pbf_relaxation=args.pbf_relaxation,
            pbf_artificial_pressure=args.pbf_artificial_pressure,
            shape_collision_body_feedback=False,
            max_diffuse_particles=args.fluid_diffuse_max_particles,
            diffuse_threshold=args.fluid_diffuse_threshold,
            diffuse_lifetime=args.fluid_diffuse_lifetime,
            diffuse_drag=args.fluid_diffuse_drag,
            diffuse_buoyancy=args.fluid_diffuse_buoyancy,
            diffuse_ballistic=args.fluid_diffuse_ballistic,
            diffuse_spawn_probability=args.fluid_diffuse_spawn_probability,
            render_smoothing=args.fluid_render_smoothing,
            render_anisotropy_scale=args.fluid_render_anisotropy_scale,
            render_update_interval=args.fluid_render_update_interval,
            diffuse_update_interval=args.fluid_diffuse_update_interval,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_fluid = True
        self.viewer.show_fluid_diffuse = True
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)
        self._configure_render_environment(args)

        self.graph = None
        self.capture_graph = args.capture_graph
        self.capture()

    def _configure_render_environment(self, args):
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is None:
            return

        renderer._env_intensity = float(args.environment_intensity)
        sun = np.asarray(args.sun_direction, dtype=np.float32)
        sun_norm = float(np.linalg.norm(sun))
        if sun_norm > 1.0e-6:
            renderer._sun_direction = sun / sun_norm
        renderer.sky_upper = (0.42, 0.74, 1.0)
        renderer.sky_lower = (0.72, 0.82, 0.78)
        renderer.ambient_sky = (0.92, 0.96, 1.0)
        renderer.ambient_ground = (0.48, 0.50, 0.52)
        renderer.exposure = args.exposure

    def capture(self):
        self.graph = None
        if not self.capture_graph:
            return
        if not wp.get_device().is_cuda:
            self.capture_graph = False
            warnings.warn("Wave pool graph capture is only available on CUDA devices.", stacklevel=2)
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        except Exception as exc:
            self.capture_graph = False
            warnings.warn(f"Wave pool graph capture failed; falling back to uncaptured stepping: {exc}", stacklevel=2)
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            wp.launch(
                kernel=drive_wave_paddle,
                dim=1,
                inputs=[
                    self.paddle_time,
                    self.sim_dt,
                    self.paddle_body,
                    self.paddle_base_pos,
                    self.paddle_amplitude,
                    self.paddle_frequency,
                    0.5,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.sph_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

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
        self.viewer.end_frame()

    def _log_fluid_surface(self):
        if (
            not self.viewer.show_fluid
            or getattr(self.viewer, "fluids", None) is None
            or not self.sph_solver.render_buffers_valid
            or self.sph_solver.render_positions is None
        ):
            return

        self.viewer.log_fluid(
            "/model/fluid",
            self.sph_solver.render_positions,
            radii=self.model.particle_radius,
            radius_scale=self.fluid_radius_scale,
            color=self.fluid_color,
            ior=self.fluid_ior,
            blur_radius_world=self.fluid_blur_radius,
            anisotropy=self.sph_solver.render_anisotropy,
            anisotropy_secondary=self.sph_solver.render_anisotropy_secondary,
            anisotropy_tertiary=self.sph_solver.render_anisotropy_tertiary,
            hidden=False,
        )

    def _log_diffuse_particles(self):
        if (
            not self.viewer.show_fluid
            or not self.viewer.show_fluid_diffuse
            or self.sph_solver.diffuse_positions is None
        ):
            self.viewer.log_fluid_diffuse("/model/fluid/diffuse", None, hidden=True)
            return

        self.viewer.log_fluid_diffuse(
            "/model/fluid/diffuse",
            self.sph_solver.diffuse_positions,
            self.sph_solver.diffuse_velocities,
            radius=self.foam_radius,
            color=self.foam_color,
            motion_blur_scale=self.foam_motion_blur,
            lifetime=self.foam_lifetime,
            hidden=False,
        )

    def gui(self, ui):
        _changed, self.viewer.show_fluid = ui.checkbox("Fluid Surface", self.viewer.show_fluid)
        _changed, self.viewer.show_fluid_diffuse = ui.checkbox("Foam", self.viewer.show_fluid_diffuse)
        _changed, self.viewer.show_particles = ui.checkbox("Raw Particles", self.viewer.show_particles)
        ui.separator()
        ui.text("Wave Generator (takes effect on restart when graph capture is on)")
        _, self.paddle_amplitude = ui.slider_float("Paddle Amplitude", self.paddle_amplitude, 0.0, 0.4, "%.2f")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("SPH particles contain non-finite positions")
        if not np.all(np.isfinite(qd)):
            raise ValueError("SPH particles contain non-finite velocities")

        active = (self.model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
        speeds = np.linalg.norm(qd[active], axis=1)
        if float(speeds.max()) <= 0.05:
            raise ValueError("Wave pool fluid is static; the paddle generated no waves")

        if self.sph_solver.diffuse_positions is not None:
            alive_foam = int(np.count_nonzero(self.sph_solver.diffuse_positions.numpy()[:, 3] > 0.0))
            if alive_foam == 0:
                raise ValueError("Breaking waves spawned no diffuse foam particles")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=3)
        parser.add_argument(
            "--capture-graph",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the SPH substeps in a CUDA graph.",
        )

        parser.add_argument("--dim-x", type=int, default=115)
        parser.add_argument("--dim-y", type=int, default=44)
        parser.add_argument("--dim-z", type=int, default=14)
        parser.add_argument("--spacing", type=float, default=0.028)
        parser.add_argument("--radius", type=float, default=0.021)
        parser.add_argument("--jitter", type=float, default=0.0015)
        parser.add_argument("--emit-lower", type=float, nargs=3, default=(-1.98, -0.66, 0.05))

        parser.add_argument("--smoothing-length", type=float, default=0.0602)
        parser.add_argument("--rest-density", type=float, default=460.0)
        parser.add_argument("--viscosity", type=float, default=0.010)
        parser.add_argument("--particle-friction", type=float, default=0.02)
        parser.add_argument("--cohesion", type=float, default=0.020)
        parser.add_argument("--surface-tension", type=float, default=0.0000006)
        parser.add_argument("--xsph-strength", type=float, default=0.04)
        parser.add_argument("--free-surface-drag", type=float, default=0.05)
        parser.add_argument("--dissipation", type=float, default=0.10)
        parser.add_argument("--velocity-damping", type=float, default=0.010)
        parser.add_argument("--sleep-threshold", type=float, default=0.0)
        parser.add_argument("--boundary-damping", type=float, default=0.10)
        parser.add_argument("--shape-friction", type=float, default=0.30)
        parser.add_argument("--shape-adhesion", type=float, default=0.0)
        parser.add_argument("--max-velocity", type=float, default=6.5)
        parser.add_argument("--max-acceleration", type=float, default=90.0)
        parser.add_argument("--pbf-iterations", type=int, default=3)
        parser.add_argument("--pbf-relaxation", type=float, default=0.60)
        parser.add_argument("--pbf-artificial-pressure", type=float, default=0.002)
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--bounds-lower", type=float, nargs=3, default=(-2.30, -0.72, 0.0))
        parser.add_argument("--bounds-upper", type=float, nargs=3, default=(2.60, 0.72, 1.8))
        parser.add_argument("--ground-color", type=float, nargs=3, default=(0.54, 0.55, 0.53))

        parser.add_argument("--paddle-x", type=float, default=-2.20)
        parser.add_argument("--paddle-height", type=float, default=0.55)
        parser.add_argument("--paddle-amplitude", type=float, default=0.26)
        parser.add_argument("--paddle-period", type=float, default=2.0)
        parser.add_argument("--beach-start", type=float, default=0.20)
        parser.add_argument("--beach-length", type=float, default=2.6)
        parser.add_argument("--beach-angle-deg", type=float, default=13.0)

        parser.add_argument("--fluid-diffuse-max-particles", type=int, default=14000)
        parser.add_argument("--fluid-diffuse-threshold", type=float, default=3.0)
        parser.add_argument("--fluid-diffuse-lifetime", type=float, default=1.8)
        parser.add_argument("--fluid-diffuse-drag", type=float, default=0.9)
        parser.add_argument("--fluid-diffuse-buoyancy", type=float, default=1.0)
        parser.add_argument("--fluid-diffuse-ballistic", type=int, default=9)
        parser.add_argument("--fluid-diffuse-spawn-probability", type=float, default=0.30)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))
        parser.add_argument("--fluid-ior", type=float, default=1.0)
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.075)
        parser.add_argument("--foam-color", type=float, nargs=4, default=(0.9, 0.95, 1.0, 1.6))
        parser.add_argument("--foam-radius", type=float, default=0.016)
        parser.add_argument("--foam-motion-blur", type=float, default=1.0)
        parser.add_argument("--fluid-render-smoothing", type=float, default=0.8)
        parser.add_argument("--fluid-render-anisotropy-scale", type=float, default=0.82)
        parser.add_argument("--fluid-render-update-interval", type=int, default=2)
        parser.add_argument("--fluid-diffuse-update-interval", type=int, default=2)

        parser.add_argument("--environment-intensity", type=float, default=3.15)
        parser.add_argument("--exposure", type=float, default=1.08)
        parser.add_argument("--sun-direction", type=float, nargs=3, default=(0.78, -0.56, 0.34))
        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.90, -1.95, 0.90))
        parser.add_argument("--camera-pitch", type=float, default=-13.0)
        parser.add_argument("--camera-yaw", type=float, default=136.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
