# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Wave Pool
#
# A kinematic paddle at the deep end of a walled pool drives waves in
# XPBD position-based fluid that travel down the tank and break on a
# sloped beach. The paddle is a kinematic rigid body whose transform is
# animated directly; the fluid interacts with it and the beach through
# the standard XPBD particle-shape contact pipeline.
#
# Command: python -m newton.examples fluid_xpbd_wave_pool
#
###########################################################################

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

import newton
import newton.examples

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

        spacing = args.spacing
        radius = 0.5 * spacing
        mass = args.rest_density * spacing**3

        self.pool_half_y = 0.5 * args.pool_width
        self.paddle_amplitude = args.paddle_amplitude
        self.paddle_frequency = 2.0 * np.pi / max(args.paddle_period, 1.0e-3)

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        builder.default_particle_radius = radius
        builder.default_shape_cfg.mu = 0.2

        builder.add_particle_grid(
            pos=wp.vec3(args.emit_lower[0], -0.5 * (args.dim_y - 1) * spacing, args.emit_lower[1]),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            dim_z=args.dim_z,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=mass,
            jitter=0.05 * spacing,
            radius_mean=radius,
            flags=ParticleFlags.ACTIVE | ParticleFlags.FLUID,
        )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.2))

        # sloped beach the waves run up and break on
        beach_angle = float(np.deg2rad(args.beach_angle_deg))
        beach_length = args.beach_length
        beach_half_thickness = 0.10
        beach_center = wp.vec3(
            args.beach_start + 0.5 * beach_length * np.cos(beach_angle),
            0.0,
            0.5 * beach_length * np.sin(beach_angle) - beach_half_thickness / np.cos(beach_angle),
        )
        beach_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -beach_angle)
        self.beach_xform = wp.transform(beach_center, beach_q)
        self.beach_half_extents = wp.vec3(0.5 * beach_length, self.pool_half_y + 0.2, beach_half_thickness)
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

        # side walls and a back wall behind the paddle keep the pool contained;
        # the walls run past the end of the beach so breaking waves cannot
        # spill off its sides
        wall_color = (0.62, 0.72, 0.78)
        wall_height = 0.55
        beach_end = args.beach_start + beach_length * float(np.cos(beach_angle))
        pool_center_x = 0.5 * (args.paddle_x - 0.5 + beach_end + 0.2)
        pool_half_x = 0.5 * (beach_end + 0.2 - (args.paddle_x - 0.5))
        for sy in (-1.0, 1.0):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(pool_center_x, sy * (self.pool_half_y + 0.05), 0.5 * wall_height), wp.quat_identity()
                ),
                hx=pool_half_x,
                hy=0.05,
                hz=0.5 * wall_height,
                color=wall_color,
                opacity=args.wall_opacity,
            )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(args.paddle_x - 0.35, 0.0, 0.5 * wall_height), wp.quat_identity()),
            hx=0.05,
            hy=self.pool_half_y + 0.1,
            hz=0.5 * wall_height,
            color=wall_color,
            opacity=args.wall_opacity,
        )

        # kinematic paddle at the deep end; its transform is animated directly
        self.paddle_base_pos = wp.vec3(args.paddle_x, 0.0, args.paddle_height)
        self.paddle_body = builder.add_body(
            xform=wp.transform(self.paddle_base_pos, wp.quat_identity()),
            is_kinematic=True,
            label="wave_paddle",
        )
        builder.add_shape_box(
            self.paddle_body,
            hx=0.06,
            hy=self.pool_half_y,
            hz=args.paddle_height + 0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.1),
            color=(0.35, 0.38, 0.42),
        )

        # small low-density primitives that bob on the passing waves; their
        # buoyancy is emergent from the unified XPBD fluid + contact solve
        water_top = args.emit_lower[1] + (args.dim_z - 1) * spacing
        self.float_bodies = self._add_floats(builder, args, water_top)

        self.model = builder.finalize()
        self.model.particle_max_velocity = 0.5 * radius / self.sim_dt
        self.model.soft_contact_mu = 0.1

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.paddle_time = wp.zeros(1, dtype=float, device=self.model.device)

        # remove spawned particles that started inside the beach ramp
        wp.launch(
            kernel=deactivate_particles_inside_box,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self.model.particle_flags,
                self.beach_xform,
                self.beach_half_extents,
                radius,
            ],
            device=self.model.device,
        )

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=spacing,
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
            max_diffuse_particles=args.foam_max_particles,
            diffuse_lifetime=args.foam_lifetime,
            diffuse_threshold=1.2,
            diffuse_spawn_probability=0.5,
            # Cranking the paddle amplitude slams water into momentary clumps at
            # many times the rest neighbor count (~30-80), and a single such
            # particle stalls its whole warp -- the cause of the FPS collapse at
            # high amplitude. Capping above the settled bulk leaves calm water
            # untouched but bounds those clumps, so the frame rate holds up no
            # matter how violent the paddle gets.
            fluid_max_neighbors=args.max_neighbors,
            # keep a yanked float from diverging if the user grabs and flings it
            body_max_velocity=12.0,
            body_max_angular_velocity=40.0,
        )

        self.fluid_color = tuple(args.fluid_color)
        self.fluid_radius_scale = args.fluid_radius_scale
        self.fluid_blur_radius = args.fluid_blur_radius
        self.render_smoothing = args.render_smoothing
        self.render_anisotropy_scale = args.render_anisotropy_scale

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        use_fluid_surface = args.render_mode == "fluid" and getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)

        # CUDA graph capture: the whole substep loop (paddle, collide, picking,
        # solve) is replayed as a single graph, eliminating per-substep launch
        # overhead. Recaptured only when a GUI-tunable scalar baked into the
        # kernels changes.
        self.graph = None
        self.use_cuda_graph = wp.get_device(self.model.device).is_cuda
        self._graph_key = None

    def _graph_key_tuple(self):
        return (round(self.paddle_amplitude, 6), round(self.solver.fluid_viscosity, 6))

    def simulate(self):
        self.solver.reorder_particles(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
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
        _, self.paddle_amplitude = ui.slider_float("Paddle Amplitude", self.paddle_amplitude, 0.0, 0.4, "%.2f")
        _, self.solver.fluid_viscosity = ui.slider_float("Viscosity", self.solver.fluid_viscosity, 0.0, 1.0, "%.2f")

    def test_final(self):
        active = (self.model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
        q = self.state_0.particle_q.numpy()[active]
        qd = self.state_0.particle_qd.numpy()[active]
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise ValueError("XPBD fluid particles contain non-finite state")
        if np.abs(q[:, 1]).max() > self.pool_half_y + 0.3:
            raise ValueError("Fluid escaped the pool side walls")
        mean_speed = float(np.linalg.norm(qd, axis=1).mean())
        if mean_speed < 1.0e-3:
            raise ValueError("Wave pool fluid is static; the paddle generated no waves")

        if self.float_bodies:
            float_q = self.state_0.body_q.numpy()[self.float_bodies]
            if not np.all(np.isfinite(float_q)):
                raise ValueError("Floating primitives contain non-finite transforms")
            if np.abs(float_q[:, 1]).max() > self.pool_half_y + 0.3:
                raise ValueError("A floating primitive escaped the pool side walls")
            # low-density primitives must float, not sink through the pool floor
            if float_q[:, 2].min() < 0.0:
                raise ValueError("A floating primitive sank through the pool floor")

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
        # Hide the fluid surface while debugging with raw particles, so toggling
        # "Show Particles" in the GUI leaves only the particles visible.
        if show_fluid and not self.viewer.show_particles:
            self._log_fluid_surface()
        elif self.solver.diffuse_positions is not None:
            # the foam is emitted by _log_fluid_surface; when that is skipped,
            # clear the last foam batch so it doesn't stay frozen on screen
            self.viewer.log_fluid_diffuse("/model/fluid/diffuse", None)
        self.viewer.end_frame()

    def _log_fluid_surface(self):
        self.solver.update_render_particles(
            self.state_0,
            smoothing=self.render_smoothing,
            anisotropy_scale=self.render_anisotropy_scale,
        )
        self.viewer.log_fluid(
            "/model/fluid",
            self.solver.render_positions,
            radii=self.model.particle_radius,
            radius_scale=self.fluid_radius_scale,
            color=self.fluid_color,
            blur_radius_world=self.fluid_blur_radius,
            anisotropy=self.solver.render_anisotropy,
            anisotropy_secondary=self.solver.render_anisotropy_secondary,
            anisotropy_tertiary=self.solver.render_anisotropy_tertiary,
            hidden=False,
        )
        if getattr(self.viewer, "show_fluid_diffuse", False) and self.solver.diffuse_positions is not None:
            self.viewer.log_fluid_diffuse(
                "/model/fluid/diffuse",
                self.solver.diffuse_positions,
                self.solver.diffuse_velocities,
                radius=0.006,
                color=(0.9, 0.95, 1.0, 1.1),
                motion_blur_scale=3.0,
                lifetime=self.solver.diffuse_lifetime,
                surface_bias=0.035,
                hidden=False,
            )

    @staticmethod
    def _add_floats(builder, args, water_top):
        """Drop a few small, low-density primitives that float on the waves.

        Densities are given as fractions of the fluid rest density (``< 1``
        floats); the buoyancy is emergent from the unified XPBD fluid-density and
        particle-shape contact solve -- no special coupling force is applied, as
        in the interactive-tank example.
        """
        rng = np.random.default_rng(5)
        size = args.float_size
        # (primitive kind, density as a fraction of water, RGB color)
        specs = (
            ("sphere", 0.45, (0.95, 0.55, 0.20)),
            ("box", 0.55, (0.25, 0.65, 0.95)),
            ("capsule", 0.35, (0.95, 0.85, 0.30)),
            ("sphere", 0.60, (0.40, 0.85, 0.45)),
            ("box", 0.40, (0.92, 0.35, 0.55)),
            ("capsule", 0.50, (0.62, 0.45, 0.95)),
        )
        bodies = []
        n = max(int(args.float_count), 0)
        for i in range(n):
            kind, fraction, color = specs[i % len(specs)]
            # scatter along the deep-to-mid pool (clear of the paddle and beach),
            # alternating across the width so the waves rock each one differently
            x = -1.6 + 1.6 * (float(i) / float(max(n - 1, 1)))
            y = (0.22 if i % 2 == 0 else -0.22) + float(rng.uniform(-0.05, 0.05))
            z = water_top + 0.12 + 0.05 * float(i)
            cfg = newton.ModelBuilder.ShapeConfig(density=fraction * args.rest_density, mu=0.3)
            if kind == "capsule":
                # the capsule's length is along local +Z; lay it on its side so
                # it floats like a log instead of bobbing end-up
                axis = wp.vec3(0.0, 1.0, 0.0) if i % 4 == 0 else wp.vec3(1.0, 0.0, 0.0)
                q = wp.quat_from_axis_angle(axis, 0.5 * np.pi)
            else:
                q = wp.quat_from_axis_angle(wp.vec3(0.3, 0.7, 0.2), float(rng.uniform(0.0, 0.5)))
            body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), q), label=f"float_{i}")
            if kind == "sphere":
                builder.add_shape_sphere(body, radius=size, cfg=cfg, color=color)
            elif kind == "box":
                builder.add_shape_box(body, hx=size, hy=size, hz=0.7 * size, cfg=cfg, color=color)
            else:
                builder.add_shape_capsule(body, radius=0.65 * size, half_height=size, cfg=cfg, color=color)
            bodies.append(body)
        return bodies

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        # ~100k particles at 2 PBF iterations (real-time standard). The long pool
        # keeps spacing coarser than the tank/dam-break scenes for the same count.
        parser.add_argument("--substeps", type=int, default=2)
        parser.add_argument("--iterations", type=int, default=2)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")

        parser.add_argument("--dim-x", type=int, default=123)
        parser.add_argument("--dim-y", type=int, default=43)
        parser.add_argument("--dim-z", type=int, default=19)
        parser.add_argument("--spacing", type=float, default=0.0211)
        parser.add_argument("--emit-lower", type=float, nargs=2, default=(-2.05, 0.025))
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--pool-width", type=float, default=0.95)
        parser.add_argument("--wall-opacity", type=float, default=0.3)
        parser.add_argument("--beach-angle-deg", type=float, default=8.0)
        parser.add_argument("--beach-length", type=float, default=2.4)
        parser.add_argument("--beach-start", type=float, default=0.35)
        parser.add_argument("--paddle-x", type=float, default=-2.20)
        parser.add_argument("--paddle-height", type=float, default=0.30)
        parser.add_argument("--paddle-amplitude", type=float, default=0.16)
        parser.add_argument("--paddle-period", type=float, default=1.5)

        parser.add_argument("--cohesion", type=float, default=0.6)
        parser.add_argument("--viscosity", type=float, default=0.03)
        # Cap fluid neighbors above the settled bulk (~80) so high-amplitude
        # clumps can't stall a warp; 0 disables the cap.
        parser.add_argument("--max-neighbors", type=int, default=128)

        # small low-density primitives (spheres/boxes/capsules) that float on the waves
        parser.add_argument("--float-count", type=int, default=6)
        parser.add_argument("--float-size", type=float, default=0.08)

        parser.add_argument("--foam-max-particles", type=int, default=20000)
        parser.add_argument("--foam-lifetime", type=float, default=1.8)
        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument("--render-anisotropy-scale", type=float, default=1.0)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.034)

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(0.9, -2.4, 1.5))
        parser.add_argument("--camera-pitch", type=float, default=-26.0)
        parser.add_argument("--camera-yaw", type=float, default=112.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
