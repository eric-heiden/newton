# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Dam Break
#
# A water column collapses under gravity, surges across an unbounded
# ground plane, splashes against a static pillar, and carries a light
# dynamic box along with the wave. The fluid is simulated as a
# position-based fluid (PBF) inside SolverXPBD: particles flagged with
# ParticleFlags.FLUID generate SPH density constraints instead of
# pairwise contacts, so the fluid two-way couples with rigid bodies and
# is not confined to solver bounds.
#
# Command: python -m newton.examples fluid_xpbd_dam_break
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

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        builder.default_particle_radius = radius

        builder.add_particle_grid(
            pos=wp.vec3(-1.2, -0.5 * args.dim_y * spacing, radius),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            dim_z=args.dim_z,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=mass,
            jitter=0.1 * spacing,
            radius_mean=radius,
            flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
        )

        # heavy pillar the wave splashes against (dynamic so it can be picked)
        self.pillar_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.25), wp.quat_identity()),
        )
        builder.add_shape_box(
            body=self.pillar_body,
            hx=0.1,
            hy=0.1,
            hz=0.25,
            cfg=newton.ModelBuilder.ShapeConfig(density=2000.0, mu=0.4),
        )

        # light dynamic box that floats on the wave
        box_half = 0.07
        self.box_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.25, box_half), wp.quat_identity()),
        )
        builder.add_shape_box(
            body=self.box_body,
            hx=box_half,
            hy=box_half,
            hz=box_half,
            cfg=newton.ModelBuilder.ShapeConfig(density=250.0, mu=0.3),
        )

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.1))

        self.model = builder.finalize()
        # CFL-style velocity clamp (half a particle radius per substep)
        self.model.particle_max_velocity = 0.5 * radius / self.sim_dt
        self.model.soft_contact_mu = 0.1

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=spacing,
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
            fluid_vorticity_confinement=args.vorticity_confinement,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        self.fluid_color = tuple(args.fluid_color)
        self.fluid_ior = args.fluid_ior
        self.fluid_blur_radius = args.fluid_blur_radius
        self.fluid_radius_scale = args.fluid_radius_scale
        self.render_smoothing = args.render_smoothing
        self.render_anisotropy_scale = args.render_anisotropy_scale

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        use_fluid_surface = args.render_mode == "fluid" and getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.viewer.set_camera(pos=wp.vec3(1.6, -2.3, 1.1), pitch=-17.0, yaw=121.0)

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

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("XPBD fluid particles contain non-finite positions")
        if not np.all(np.isfinite(qd)):
            raise ValueError("XPBD fluid particles contain non-finite velocities")
        radius = float(self.model.particle_max_radius)
        if q[:, 2].min() < -2.0 * radius:
            raise ValueError("XPBD fluid particles fell through the ground plane")
        if np.abs(q[:, :2]).max() > 10.0:
            raise ValueError("XPBD fluid particles dispersed unrealistically far")

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
            ior=self.fluid_ior,
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
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--iterations", type=int, default=3)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")

        parser.add_argument("--dim-x", type=int, default=14)
        parser.add_argument("--dim-y", type=int, default=20)
        parser.add_argument("--dim-z", type=int, default=24)
        parser.add_argument("--spacing", type=float, default=0.05)
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--cohesion", type=float, default=1.0)
        parser.add_argument("--viscosity", type=float, default=0.05)
        parser.add_argument("--vorticity-confinement", type=float, default=0.0)

        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument("--render-anisotropy-scale", type=float, default=1.0)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))
        parser.add_argument("--fluid-ior", type=float, default=1.0)
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
