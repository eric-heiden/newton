# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Multiworld Cup
#
# Two XPBD fluid cups are built in separate simulation worlds at overlapping
# physics coordinates. The viewer applies a small world offset so both cups are
# visible and still overlap on screen. Grab either cup with right-click picking:
# the other world's cup and fluid should not react.
#
# Command: python -m newton.examples fluid_xpbd_multiworld_cup
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
from newton.examples.fluid.utils import (
    cylinder_particle_count,
    cylinder_particle_positions,
    parse_particle_count,
    resolve_particle_spacing,
)

# Share the same cooked SDF cache as the single-cup example.
_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_cup_sdf"
_REFERENCE_SPACING = 0.0028


@wp.kernel
def _build_world_mask(
    particle_world: wp.array[wp.int32],
    world_id: int,
    mask: wp.array[wp.int32],
):
    i = wp.tid()
    if particle_world[i] == world_id:
        mask[i] = wp.int32(1)
    else:
        mask[i] = wp.int32(0)


@wp.kernel
def _compact_world_positions(
    src: wp.array[wp.vec3],
    mask: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    world_offset: wp.vec3,
    dst: wp.array[wp.vec3],
):
    i = wp.tid()
    if mask[i] == wp.int32(1):
        dst[offsets[i]] = src[i] + world_offset


@wp.kernel
def _compact_world_radii(
    src: wp.array[wp.float32],
    mask: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    dst: wp.array[wp.float32],
):
    i = wp.tid()
    if mask[i] == wp.int32(1):
        dst[offsets[i]] = src[i]


@wp.kernel
def _compact_world_vec4(
    src: wp.array[wp.vec4],
    mask: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    dst: wp.array[wp.vec4],
):
    i = wp.tid()
    if mask[i] == wp.int32(1):
        dst[offsets[i]] = src[i]


def _points_to_body_frame(points, body_xform):
    delta = points - body_xform[:3]
    inverse_quat_vector = np.broadcast_to(-body_xform[3:6], delta.shape)
    cross = 2.0 * np.cross(inverse_quat_vector, delta)
    return delta + body_xform[6] * cross + np.cross(inverse_quat_vector, cross)


class Example:
    WORLD_COUNT = 2

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        target_per_world = max(args.particle_count // self.WORLD_COUNT, 1)
        spacing, _ = resolve_particle_spacing(
            target_per_world,
            _REFERENCE_SPACING,
            lambda candidate: cylinder_particle_count(
                candidate,
                args.cup_inner_radius,
                args.wall_thickness,
                args.fill_height,
            ),
        )
        radius = 0.5 * spacing
        self.particle_spacing = spacing
        self.particle_radius = radius
        self.inner_radius = args.cup_inner_radius
        self.cup_height = args.cup_height
        self.fill_height = args.fill_height
        wall_thickness = args.wall_thickness
        self.wall_thickness = wall_thickness

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        builder.default_particle_radius = radius
        builder.default_shape_cfg.mu = 0.2

        cup_mesh = self._build_cup_mesh(self.inner_radius, wall_thickness, self.cup_height)
        cup_mesh.build_sdf(
            max_resolution=args.sdf_resolution,
            narrow_band_range=(-0.03, 0.03),
            margin=0.02,
            cache_dir=_SDF_CACHE_DIR,
        )

        self.cup_bodies: list[int] = []
        cup_colors = (
            (0.65, 0.84, 0.94),
            (0.95, 0.63, 0.42),
        )
        for world in range(self.WORLD_COUNT):
            builder.begin_world(label=f"xpbd_cup_{world}")
            cup_body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                label=f"cup_{world}",
            )
            self.cup_bodies.append(cup_body)
            builder.add_shape_mesh(
                cup_body,
                mesh=cup_mesh,
                cfg=newton.ModelBuilder.ShapeConfig(density=args.cup_density, mu=0.4),
                color=cup_colors[world],
                opacity=args.cup_opacity,
            )
            self._fill_water(builder, args, wall_thickness)
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
            builder.end_world()

        self.model = builder.finalize(device=args.device)
        wall_cross_speed = 0.5 * wall_thickness / self.sim_dt
        self._water_max_velocity = 0.85 * wall_cross_speed
        self._cup_max_velocity = 0.6 * wall_cross_speed
        self.model.particle_max_velocity = self._water_max_velocity
        self.model.soft_contact_mu = 0.3
        with wp.ScopedDevice(self.model.device):
            self.model.particle_grid = wp.HashGrid(256, 256, 256)

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=spacing,
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
            fluid_relaxation=args.relaxation,
            fluid_max_neighbors=args.max_neighbors,
            body_max_velocity=self._cup_max_velocity,
            body_max_angular_velocity=10.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        self.fluid_colors = (tuple(args.fluid_color_0), tuple(args.fluid_color_1))
        self.fluid_radius_scale = args.fluid_radius_scale
        self.fluid_blur_radius = args.fluid_blur_radius
        self.render_smoothing = args.render_smoothing
        self._render_world_mask = wp.empty(self.model.particle_count, dtype=wp.int32, device=self.model.device)
        self._render_world_offsets = wp.empty(self.model.particle_count, dtype=wp.int32, device=self.model.device)
        self._world_render_cache: dict[int, dict[str, wp.array]] = {}

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((args.render_world_spacing, 0.0, 0.0))
        self.viewer.picking_enabled = True
        self._apply_picking_params(args.pick_stiffness, args.pick_damping)
        self.viewer.show_particles = False
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = True
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)

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
        spacing = self.particle_spacing
        radius = 0.5 * spacing
        pts = cylinder_particle_positions(spacing, self.inner_radius, wall_thickness, args.fill_height)
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
        _, self.fluid_blur_radius = ui.slider_float("Smoothing Radius", self.fluid_blur_radius, 0.0, 0.25, "%.3f")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        particle_world = self.model.particle_world.numpy()
        body_world = self.model.body_world.numpy()

        if self.model.world_count != self.WORLD_COUNT:
            raise ValueError("multiworld cup example must build exactly two worlds")
        if set(np.unique(particle_world).tolist()) != {0, 1} or set(np.unique(body_world).tolist()) != {0, 1}:
            raise ValueError("cups and fluids must be split across two worlds")
        world_counts = np.bincount(particle_world, minlength=self.WORLD_COUNT)[: self.WORLD_COUNT]
        if np.any(world_counts != world_counts[0]):
            raise ValueError("fluid particle counts must match across replicated worlds")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(body_q)):
            raise ValueError("XPBD multiworld cup state contains non-finite values")
        tolerance = 1.0e-4
        if q[:, 2].min() < self.particle_radius - tolerance:
            raise ValueError("multiworld cup water tunneled below the floor")
        for world in range(self.WORLD_COUNT):
            particles = q[particle_world == world]
            local_q = _points_to_body_frame(particles, body_q[self.cup_bodies[world]])
            radial = np.linalg.norm(local_q[:, :2], axis=1)
            if local_q[:, 2].min() < self.wall_thickness - self.particle_radius - tolerance:
                raise ValueError("multiworld cup water tunneled through a cup bottom")
            below_rim = local_q[:, 2] <= self.cup_height + self.particle_radius
            outer_radius = self.inner_radius + self.wall_thickness + self.particle_radius
            if np.any(radial[below_rim] > outer_radius + tolerance):
                raise ValueError("multiworld cup water tunneled through a cup wall")
            if float(np.percentile(radial, 95)) > 1.1 * self.inner_radius:
                raise ValueError("multiworld cup fluid is not inside its cup")
            minimum_fill_top = self.wall_thickness + 0.7 * (self.fill_height - self.wall_thickness)
            if float(local_q[:, 2].max()) < minimum_fill_top:
                raise ValueError("multiworld cup water over-compressed instead of retaining its fill height")

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
            self._log_fluid_surfaces()
        else:
            self._hide_fluid_surfaces()
        self.viewer.end_frame()

    def _hide_fluid_surfaces(self):
        self.viewer.log_fluid("/model/fluid", None)
        for world in range(self.WORLD_COUNT):
            self.viewer.log_fluid(f"/model/fluid/world_{world}", None)

    def _ensure_world_render_cache(self, world: int, count: int) -> dict[str, wp.array]:
        cache = self._world_render_cache.setdefault(world, {})
        specs = {
            "positions": wp.vec3,
            "radii": wp.float32,
            "anisotropy": wp.vec4,
            "anisotropy_secondary": wp.vec4,
            "anisotropy_tertiary": wp.vec4,
        }
        for name, dtype in specs.items():
            arr = cache.get(name)
            if arr is None or len(arr) != count:
                cache[name] = wp.empty(count, dtype=dtype, device=self.model.device)
        return cache

    def _visual_world_offset(self, world: int) -> wp.vec3:
        if self.viewer.world_offsets is None:
            return wp.vec3(0.0)
        offsets = self.viewer.world_offsets.numpy()
        if world < 0 or world >= len(offsets):
            return wp.vec3(0.0)
        offset = offsets[world]
        return wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))

    def _compact_world_render_particles(self, world: int) -> tuple[dict[str, wp.array], int] | tuple[None, int]:
        n = self.model.particle_count
        wp.launch(
            _build_world_mask,
            dim=n,
            inputs=[self.model.particle_world, world, self._render_world_mask],
            device=self.model.device,
        )
        wp.utils.array_scan(self._render_world_mask, self._render_world_offsets, inclusive=False)
        count = int(self._render_world_offsets[-1:].numpy()[0]) + int(self._render_world_mask[-1:].numpy()[0])
        if count == 0:
            return None, 0

        cache = self._ensure_world_render_cache(world, count)
        visual_offset = self._visual_world_offset(world)
        wp.launch(
            _compact_world_positions,
            dim=n,
            inputs=[
                self.solver.render_positions,
                self._render_world_mask,
                self._render_world_offsets,
                visual_offset,
                cache["positions"],
            ],
            device=self.model.device,
        )
        wp.launch(
            _compact_world_radii,
            dim=n,
            inputs=[self.model.particle_radius, self._render_world_mask, self._render_world_offsets, cache["radii"]],
            device=self.model.device,
        )
        for name, src in (
            ("anisotropy", self.solver.render_anisotropy),
            ("anisotropy_secondary", self.solver.render_anisotropy_secondary),
            ("anisotropy_tertiary", self.solver.render_anisotropy_tertiary),
        ):
            wp.launch(
                _compact_world_vec4,
                dim=n,
                inputs=[src, self._render_world_mask, self._render_world_offsets, cache[name]],
                device=self.model.device,
            )
        return cache, count

    def _log_fluid_surfaces(self):
        self.solver.update_render_particles(self.state_0, smoothing=self.render_smoothing)
        self.viewer.log_fluid("/model/fluid", None)

        for world in range(self.WORLD_COUNT):
            cache, count = self._compact_world_render_particles(world)
            if cache is None or count == 0:
                self.viewer.log_fluid(f"/model/fluid/world_{world}", None)
                continue
            self.viewer.log_fluid(
                f"/model/fluid/world_{world}",
                cache["positions"],
                radii=cache["radii"],
                radius_scale=self.fluid_radius_scale,
                color=self.fluid_colors[world],
                blur_radius_world=self.fluid_blur_radius,
                anisotropy=cache["anisotropy"],
                anisotropy_secondary=cache["anisotropy_secondary"],
                anisotropy_tertiary=cache["anisotropy_tertiary"],
                hidden=False,
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=8)
        parser.add_argument("--iterations", type=int, default=4)
        parser.add_argument("--max-neighbors", type=int, default=128)
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--cup-inner-radius", type=float, default=0.06)
        parser.add_argument("--cup-height", type=float, default=0.11)
        parser.add_argument("--wall-thickness", type=float, default=0.008)
        parser.add_argument("--cup-density", type=float, default=2000.0)
        parser.add_argument("--cup-opacity", type=float, default=0.38)
        parser.add_argument("--sdf-resolution", type=int, default=192, help="Cup SDF grid resolution.")

        parser.add_argument(
            "--particle-count",
            type=parse_particle_count,
            default=120_000,
            help="Target total fluid particle count across both worlds; particle size and fills are derived automatically.",
        )
        parser.add_argument("--fill-height", type=float, default=0.082)
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--cohesion", type=float, default=0.4)
        parser.add_argument("--viscosity", type=float, default=0.0)
        parser.add_argument("--relaxation", type=float, default=0.6)
        parser.add_argument("--pick-stiffness", type=float, default=400.0)
        parser.add_argument("--pick-damping", type=float, default=40.0)

        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument("--fluid-color-0", type=float, nargs=4, default=(0.05, 0.45, 0.72, 0.78))
        parser.add_argument("--fluid-color-1", type=float, nargs=4, default=(0.95, 0.48, 0.18, 0.55))
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.02)
        parser.add_argument(
            "--render-world-spacing",
            type=float,
            default=0.08,
            help="Visual offset between worlds; smaller than the cup diameter so the cups overlap on screen.",
        )

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(0.25, -0.32, 0.20))
        parser.add_argument("--camera-pitch", type=float, default=-23.0)
        parser.add_argument("--camera-yaw", type=float, default=135.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
