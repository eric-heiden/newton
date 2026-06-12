# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Interactive Tank
#
# A walled tank of XPBD position-based fluid with rigid boxes of varying
# density dropped in. Unlike the SPH interactive tank, no hand-tuned
# buoyancy, drag, or coupling forces are needed: boxes lighter than
# water float and denser boxes sink purely through the unified XPBD
# solve of fluid density constraints and particle-shape contacts.
# Grab and drag the boxes (or the water itself) with the mouse.
#
# Command: python -m newton.examples fluid_xpbd_interactive_tank
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


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

        self.tank_half_x = 0.5 * args.tank_size[0]
        self.tank_half_y = 0.5 * args.tank_size[1]
        wall_height = args.tank_size[2]
        wall_thickness = 0.05

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        builder.default_particle_radius = radius
        builder.default_shape_cfg.mu = 0.2

        dim_x = max(int((2.0 * self.tank_half_x - 2.0 * radius) / spacing), 2)
        dim_y = max(int((2.0 * self.tank_half_y - 2.0 * radius) / spacing), 2)
        builder.add_particle_grid(
            pos=wp.vec3(-0.5 * (dim_x - 1) * spacing, -0.5 * (dim_y - 1) * spacing, radius),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=args.dim_z,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=mass,
            jitter=0.05 * spacing,
            radius_mean=radius,
            flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
        )

        # glass tank walls
        wall_color = (0.62, 0.72, 0.78)
        hx = self.tank_half_x + wall_thickness
        for sx in (-1.0, 1.0):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(sx * (self.tank_half_x + wall_thickness), 0.0, 0.5 * wall_height), wp.quat_identity()
                ),
                hx=wall_thickness,
                hy=self.tank_half_y,
                hz=0.5 * wall_height,
                color=wall_color,
                opacity=args.wall_opacity,
            )
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(0.0, sx * (self.tank_half_y + wall_thickness), 0.5 * wall_height), wp.quat_identity()
                ),
                hx=hx,
                hy=wall_thickness,
                hz=0.5 * wall_height,
                color=wall_color,
                opacity=args.wall_opacity,
            )
        builder.add_ground_plane()

        # boxes with densities given as fractions of the fluid rest density:
        # fractions below 1 float, fractions above 1 sink
        colors = (
            (1.0, 0.78, 0.05),
            (0.10, 0.88, 0.35),
            (0.95, 0.18, 0.26),
            (0.20, 0.50, 1.0),
            (0.82, 0.35, 1.0),
        )
        self.box_bodies = []
        fractions = tuple(args.box_density_fractions)
        water_top = args.dim_z * spacing
        for i in range(args.box_count):
            column = i % 3
            row = i // 3
            x = (column - 1) * 0.34
            y = -0.16 + row * 0.32
            half = args.box_half_extent * (0.85 + 0.12 * (i % 3))
            q = wp.quat_from_axis_angle(wp.vec3(0.2, 0.8, 0.1), 0.2 * float(i))
            density = float(fractions[i % len(fractions)]) * args.rest_density
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, y, water_top + 0.25 + 0.05 * float(i)), q),
                label=f"water_cube_{i}",
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
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        self.fluid_color = tuple(args.fluid_color)
        self.fluid_radius_scale = args.fluid_radius_scale
        self.fluid_blur_radius = args.fluid_blur_radius
        self.render_smoothing = args.render_smoothing
        self.render_anisotropy_scale = args.render_anisotropy_scale

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        self._apply_picking_params(args.pick_stiffness, args.pick_damping)
        use_fluid_surface = args.render_mode == "fluid" and getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)

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
        self.simulate()
        self.sim_time += self.frame_dt

    def gui(self, ui):
        _, self.solver.fluid_viscosity = ui.slider_float("Viscosity", self.solver.fluid_viscosity, 0.0, 1.0, "%.2f")
        changed, cohesion = ui.slider_float("Cohesion", self.solver.fluid_cohesion, 0.0, 1.0, "%.2f")
        if changed:
            self.solver.fluid_cohesion = cohesion
            self.solver._update_fluid_settings()
        _, self.render_smoothing = ui.slider_float("Render Smoothing", self.render_smoothing, 0.0, 1.0, "%.2f")
        _, self.render_anisotropy_scale = ui.slider_float(
            "Anisotropy Scale", self.render_anisotropy_scale, 0.0, 3.0, "%.2f"
        )
        _, self.fluid_blur_radius = ui.slider_float("Smoothing Radius", self.fluid_blur_radius, 0.0, 0.25, "%.3f")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise ValueError("XPBD fluid particles contain non-finite state")
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Rigid boxes contain non-finite transforms")
        margin = 0.3
        if np.any(np.abs(q[:, 0]) > self.tank_half_x + margin) or np.any(np.abs(q[:, 1]) > self.tank_half_y + margin):
            raise ValueError("Fluid escaped the tank walls")
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
        if show_fluid:
            self._log_fluid_surface()
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

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=3)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")

        parser.add_argument("--tank-size", type=float, nargs=3, default=(1.7, 1.2, 0.55))
        parser.add_argument("--wall-opacity", type=float, default=0.3)
        parser.add_argument("--dim-z", type=int, default=8)
        parser.add_argument("--spacing", type=float, default=0.04)
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--box-count", type=int, default=5)
        parser.add_argument("--box-half-extent", type=float, default=0.085)
        parser.add_argument(
            "--box-density-fractions",
            type=float,
            nargs="+",
            default=(0.30, 0.55, 0.85, 0.42, 1.60),
            help="Box densities as fractions of the fluid rest density; values above 1 sink.",
        )

        parser.add_argument("--cohesion", type=float, default=0.5)
        parser.add_argument("--viscosity", type=float, default=0.05)
        parser.add_argument("--pick-stiffness", type=float, default=160.0)
        parser.add_argument("--pick-damping", type=float, default=30.0)

        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument("--render-anisotropy-scale", type=float, default=1.0)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.07)

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.7, -1.8, 1.55))
        parser.add_argument("--camera-pitch", type=float, default=-33.0)
        parser.add_argument("--camera-yaw", type=float, default=132.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
