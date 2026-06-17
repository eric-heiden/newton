# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Cup
#
# A single rigid cup sitting on the ground, filled with XPBD position-based
# fluid, that you grab and swing around with the mouse. It is the minimal
# "fluid in a moving container" scene -- the common robotics case of a
# gripper carrying a cup -- stripped of the arm/IK so it is easy to profile
# and tune. The cup carries a texture SDF, so the water collides with it via
# one cheap SDF sample per particle, and the whole substep loop is captured
# in a CUDA graph for high frame rates.
#
# Command: python -m newton.examples fluid_xpbd_cup
#
###########################################################################

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

# Cache the cooked cup SDF on disk so repeated runs skip the voxelization.
_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_cup_sdf"


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
        self.inner_radius = args.cup_inner_radius
        self.cup_height = args.cup_height
        wall_thickness = args.wall_thickness

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        builder.default_particle_radius = radius
        builder.default_shape_cfg.mu = 0.2

        # dynamic ceramic cup resting on the ground; grab it with the mouse
        self.cup_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            label="cup",
        )
        cup_mesh = self._build_cup_mesh(self.inner_radius, wall_thickness, self.cup_height)
        # An SDF on the cup lets the water collide with it via one cheap sample
        # per particle instead of a per-triangle mesh query.
        cup_mesh.build_sdf(
            max_resolution=args.sdf_resolution,
            narrow_band_range=(-0.03, 0.03),
            margin=0.02,
            cache_dir=_SDF_CACHE_DIR,
        )
        builder.add_shape_mesh(
            self.cup_body,
            mesh=cup_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=args.cup_density, mu=0.4),
            color=(0.8, 0.85, 0.92),
            opacity=args.cup_opacity,
        )

        self._fill_water(builder, args, wall_thickness)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))

        self.model = builder.finalize()
        # The water leaks when the cup is moved because the default CFL cap
        # (half a particle radius per substep) is far slower than the cup: the
        # water can't keep up, so the moving wall leaves it behind and it ends
        # up outside the cup. Cap it instead just under the wall-crossing speed
        # -- fast enough to ride along with the cup, slow enough that it cannot
        # cross the thin wall in one substep -- and clamp the cup just under
        # that so it can never outrun the water (see the solver below).
        wall_cross_speed = 0.5 * wall_thickness / self.sim_dt
        self._water_max_velocity = 0.85 * wall_cross_speed
        self._cup_max_velocity = 0.6 * wall_cross_speed
        self.model.particle_max_velocity = self._water_max_velocity
        self.model.soft_contact_mu = 0.3
        # roomier hash grid: a hard swing can fling water well past the cup, and
        # the default 128^3 grid would alias far cells onto the cup region
        with wp.ScopedDevice(self.model.device):
            self.model.particle_grid = wp.HashGrid(256, 256, 256)
            self.model.particle_grid.reserve(self.model.particle_count)

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=spacing,
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
            fluid_relaxation=args.relaxation,
            # bound the per-warp cost when a swing slams water into the cup wall
            fluid_max_neighbors=args.max_neighbors,
            # clamp the cup just under the water's max speed so it can never move
            # faster than the water can follow (which is what leaks it through
            # the wall); the angular cap keeps the rim from sweeping too fast
            body_max_velocity=self._cup_max_velocity,
            body_max_angular_velocity=10.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        self.fluid_color = tuple(args.fluid_color)
        self.fluid_radius_scale = args.fluid_radius_scale
        self.fluid_blur_radius = args.fluid_blur_radius
        self.render_smoothing = args.render_smoothing

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        self._apply_picking_params(args.pick_stiffness, args.pick_damping)
        use_fluid_surface = args.render_mode == "fluid" and getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)

        # Replay the whole substep loop (collide, picking, solve) from a CUDA
        # graph; recaptured only when a GUI-tunable solver scalar changes. Prime
        # the reorder scratch first -- it allocates on first use, which cannot
        # happen inside the capture.
        self.solver.reorder_particles(self.state_0)
        self.graph = None
        self.use_cuda_graph = wp.get_device(self.model.device).is_cuda
        self._graph_key = None

    @staticmethod
    def _build_cup_mesh(inner_radius, wall_thickness, height, segments=48):
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

    def _fill_water(self, builder, args, wall_thickness):
        """Fill the cup cavity with a column of fluid at rest spacing.

        Filling the full inner radius (not a narrower inset block) and starting
        at the rest spacing avoids the column collapsing into a dense slug at the
        bottom -- the water sits at rest density across the whole cross-section.
        """
        spacing = args.spacing
        radius = 0.5 * spacing
        r_max = self.inner_radius - radius
        z0 = wall_thickness + radius
        z1 = args.fill_height

        n_xy = max(int(2.0 * r_max / spacing) + 1, 1)
        n_z = max(int((z1 - z0) / spacing) + 1, 1)
        axis_xy = -r_max + spacing * np.arange(n_xy)
        axis_z = z0 + spacing * np.arange(n_z)
        gx, gy, gz = np.meshgrid(axis_xy, axis_xy, axis_z, indexing="ij")
        pts = np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1)

        # trim the square grid to the circular cavity
        pts = pts[pts[:, 0] ** 2 + pts[:, 1] ** 2 < r_max * r_max]
        rng = np.random.default_rng(0)
        pts += rng.uniform(-0.05 * spacing, 0.05 * spacing, size=pts.shape)

        mass = args.rest_density * spacing**3
        builder.add_particles(
            pos=pts.tolist(),
            vel=[(0.0, 0.0, 0.0)] * len(pts),
            mass=[mass] * len(pts),
            radius=[radius] * len(pts),
            flags=[int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID)] * len(pts),
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

    def _graph_key_tuple(self):
        return (round(self.solver.fluid_viscosity, 6), round(self.solver.fluid_cohesion, 6))

    def simulate(self):
        # spatial re-sort once per frame so the density solve's neighbor reads
        # stay cache-coherent after a swing churns the water
        self.solver.reorder_particles(self.state_0)
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
            self.solver._update_fluid_settings()
        _, self.render_smoothing = ui.slider_float("Render Smoothing", self.render_smoothing, 0.0, 1.0, "%.2f")
        _, self.fluid_blur_radius = ui.slider_float("Smoothing Radius", self.fluid_blur_radius, 0.0, 0.25, "%.3f")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(body_q)):
            raise ValueError("XPBD fluid or cup state contains non-finite values")
        # the water should rest inside the cup, not tunnel through the floor
        if q[:, 2].min() < -0.05:
            raise ValueError("water tunneled below the floor")
        # and it should fill a reasonable fraction of the cup cross-section,
        # rather than collapsing into a narrow slug at the bottom
        rmax = float(np.linalg.norm(q[:, :2], axis=1).max())
        if rmax < 0.5 * self.inner_radius:
            raise ValueError("water collapsed to a narrow column instead of filling the cup")

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
        self.viewer.end_frame()

    def _log_fluid_surface(self):
        self.solver.update_render_particles(self.state_0, smoothing=self.render_smoothing)
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
        parser.add_argument("--iterations", type=int, default=4)
        parser.add_argument("--max-neighbors", type=int, default=128)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--cup-inner-radius", type=float, default=0.06)
        parser.add_argument("--cup-height", type=float, default=0.11)
        parser.add_argument("--wall-thickness", type=float, default=0.008)
        parser.add_argument("--cup-density", type=float, default=2000.0)
        parser.add_argument("--cup-opacity", type=float, default=0.35)
        parser.add_argument("--sdf-resolution", type=int, default=192, help="Cup SDF grid resolution.")

        # ~50k water particles filling the cup. Coarser than a deep narrow column
        # would need, which keeps the solve cheap and the fill stable.
        parser.add_argument("--spacing", type=float, default=0.0024)
        parser.add_argument("--fill-height", type=float, default=0.085)
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--cohesion", type=float, default=0.4)
        parser.add_argument("--viscosity", type=float, default=0.0)
        # The summed (standard-PBF) density correction overshoots at full
        # strength in a small, fast-moving container: the water buzzes at the
        # velocity cap instead of settling. Under-relaxing the density push lets
        # it come to rest; the extra iterations recover incompressibility.
        parser.add_argument("--relaxation", type=float, default=0.3)
        parser.add_argument("--pick-stiffness", type=float, default=400.0)
        parser.add_argument("--pick-damping", type=float, default=40.0)

        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.02)

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(0.32, -0.32, 0.22))
        parser.add_argument("--camera-pitch", type=float, default=-22.0)
        parser.add_argument("--camera-yaw", type=float, default=135.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
