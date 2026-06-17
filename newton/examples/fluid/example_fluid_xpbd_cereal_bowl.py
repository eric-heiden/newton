# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Cereal Bowl
#
# A breakfast scene inspired by the NVIDIA Flex demos: torus-shaped
# cereal pieces float in a bowl of milk. The milk is a position-based
# fluid (PBF) inside SolverXPBD, two-way coupled with the rigid bodies,
# so the cereal bobs on the surface and the bowl reacts to the sloshing
# milk. Both the bowl and every cereal piece are dynamic bodies: drag
# them around with the mouse (right-click drag in ViewerGL) to stir the
# milk or tip the bowl. The milk is rendered as an opaque scattering
# liquid via the screen-space fluid material parameters of
# :meth:`newton.viewer.ViewerGL.log_fluid`.
#
# Command: python -m newton.examples fluid_xpbd_cereal_bowl
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

# Cache cooked SDFs on disk so repeated runs skip the (slow) voxelization.
_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_cereal_bowl_sdf"


def create_bowl_mesh(base_radius, rim_radius, height, thickness, segments=48, profile_steps=10):
    """Create a watertight bowl as a solid of revolution.

    The profile runs from the bottom-outer edge up the curved outer wall,
    over the flat rim, and back down the inner wall to the cavity floor;
    two center vertices cap the bottom disk and the cavity floor.

    Args:
        base_radius: Radius of the flat base [m].
        rim_radius: Outer radius at the rim [m].
        height: Rim height above the ground [m].
        thickness: Wall thickness [m].
        segments: Number of segments around the circumference.
        profile_steps: Number of samples along each curved wall.

    Returns:
        A watertight :class:`newton.Mesh`.
    """
    inner_base = max(base_radius - thickness, 0.25 * base_radius)

    profile = []
    # outer wall: quarter-ellipse from the base edge to the rim
    for i in range(profile_steps + 1):
        ang = 0.5 * np.pi * i / profile_steps
        r = base_radius + (rim_radius - base_radius) * np.sin(ang)
        z = height * (1.0 - np.cos(ang))
        profile.append((r, z))
    # flat rim
    profile.append((rim_radius - thickness, height))
    # inner wall back down to the cavity floor
    for i in range(1, profile_steps + 1):
        ang = 0.5 * np.pi * (1.0 - i / profile_steps)
        r = inner_base + (rim_radius - thickness - inner_base) * np.sin(ang)
        z = thickness + (height - thickness) * (1.0 - np.cos(ang))
        profile.append((r, z))

    vertices = []
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        c, s = np.cos(angle), np.sin(angle)
        for r, z in profile:
            vertices.append((r * c, r * s, z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    cavity_center = len(vertices)
    vertices.append((0.0, 0.0, thickness))

    rows = len(profile)
    indices = []
    for i in range(segments):
        j = (i + 1) % segments
        for k in range(rows - 1):
            a = i * rows + k
            b = i * rows + k + 1
            c0 = j * rows + k
            d = j * rows + k + 1
            # outward-facing winding along the revolved strips
            indices += [a, c0, b, b, c0, d]
        # bottom disk (faces down) and cavity floor (faces up)
        indices += [i * rows + 0, bottom_center, j * rows + 0]
        indices += [i * rows + rows - 1, j * rows + rows - 1, cavity_center]

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
    )


def create_torus_mesh(major_radius, minor_radius, segments_major=20, segments_minor=12):
    """Create a watertight torus mesh centered at the origin (axis = +Z)."""
    vertices = []
    for i in range(segments_major):
        theta = 2.0 * np.pi * i / segments_major
        ct, st = np.cos(theta), np.sin(theta)
        for j in range(segments_minor):
            phi = 2.0 * np.pi * j / segments_minor
            cp, sp = np.cos(phi), np.sin(phi)
            r = major_radius + minor_radius * cp
            vertices.append((r * ct, r * st, minor_radius * sp))

    indices = []
    for i in range(segments_major):
        i1 = (i + 1) % segments_major
        for j in range(segments_minor):
            j1 = (j + 1) % segments_minor
            a = i * segments_minor + j
            b = i1 * segments_minor + j
            c = i * segments_minor + j1
            d = i1 * segments_minor + j1
            indices += [a, b, d, a, d, c]

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
    )


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

        self.bowl_height = args.bowl_height
        self.bowl_rim_radius = args.bowl_rim_radius

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        builder.default_particle_radius = radius

        # dynamic ceramic bowl
        self.bowl_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            label="bowl",
        )
        bowl_mesh = create_bowl_mesh(
            base_radius=args.bowl_base_radius,
            rim_radius=args.bowl_rim_radius,
            height=args.bowl_height,
            thickness=args.bowl_thickness,
        )
        # Build an SDF on the bowl so the ~100k milk particles collide with it
        # through one cheap SDF sample each, instead of a per-triangle mesh query
        # against every particle (which makes the soft-contact count explode).
        bowl_mesh.build_sdf(
            max_resolution=args.sdf_resolution,
            narrow_band_range=(-0.03, 0.03),
            margin=0.02,
            cache_dir=_SDF_CACHE_DIR,
        )
        builder.add_shape_mesh(
            self.bowl_body,
            mesh=bowl_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=args.bowl_density, mu=0.5),
            color=(0.92, 0.93, 0.96),
        )

        # torus cereal pieces: a visual mesh plus a ring of sphere colliders
        # (sphere contacts are cheap and robust against the bowl mesh and the
        # fluid particles, and give the body a torus-like inertia)
        self.cereal_bodies = self._add_cereal(builder, args)

        # milk: a cylinder of fluid particles trimmed to the bowl cavity
        self._add_milk(builder, args)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))

        self.model = builder.finalize()
        # Fixed physical velocity cap: at this resolution the usual
        # half-a-radius-per-substep CFL clamp would throttle the milk to slow
        # motion; the bowl walls span many radii so a few radii/substep is stable.
        self.model.particle_max_velocity = args.max_velocity
        # Grippy fluid-shape friction so milk that spills out crawls to a stop
        # rather than gliding into a wide thin film. A spilled film is the worst
        # case for the PBF solve: when the bowl is thrown and milk escapes onto
        # the floor, particles created in spatial order scatter across a >1 m
        # sheet, so the hash-grid neighbor reads lose locality and the solve cost
        # multiplies (~5x). High friction plus the velocity cap below keep a
        # spill compact (~0.8 m) so the solve stays cheap under aggressive
        # picking. Milk does cling to ceramic, so this also reads naturally.
        self.model.soft_contact_mu = 1.0
        self.model.rigid_contact_max = 32768
        # A roomier hash grid: when the bowl is tipped and milk spills, the
        # particles spread past the bowl. 256^3 covers ~1.4 m at this smoothing
        # length, comfortably enclosing a friction-contained spill so far cells
        # never alias onto the bowl region.
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
            # Throwing the bowl can momentarily crush milk into a corner at many
            # times rest density; such a particle has an order of magnitude more
            # neighbors than the bulk (~200) and stalls its whole warp. Capping
            # above the bulk count leaves the in-bowl fluid untouched but bounds
            # that tail, roughly doubling the framerate when the milk disperses.
            fluid_max_neighbors=args.max_neighbors,
            # A light cereal slammed by the hard-thrown bowl can otherwise pick
            # up a divergent contact-correction velocity (>200 m/s), tunnel, and
            # blow up to NaN -- which then poisons the milk it touches. These
            # caps keep dynamic bodies sane while still allowing brisk throws.
            body_max_velocity=args.body_max_velocity,
            body_max_angular_velocity=args.body_max_angular_velocity,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # With SDF colliders each particle makes ~1 contact, so the default
        # shape_count*particle_count soft-contact capacity is hugely over-sized;
        # cap it so the XPBD particle-shape solve doesn't iterate empty slots.
        soft_contact_max = min(self.model.shape_count * self.model.particle_count, 6 * self.model.particle_count)
        self._collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_max=soft_contact_max, broad_phase="explicit"
        )
        self.contacts = self.model.contacts(collision_pipeline=self._collision_pipeline)

        self.fluid_color = tuple(args.fluid_color)
        self.fluid_blur_radius = 2.5 * spacing
        self.render_smoothing = args.render_smoothing

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        use_fluid_surface = args.render_mode == "fluid" and getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.viewer.set_camera(pos=wp.vec3(0.28, -0.33, 0.24), pitch=-30.0, yaw=140.0)

        # Prime the spatial-reorder scratch now: reorder_particles() allocates
        # its sort/gather buffers on first use, and that allocation cannot happen
        # inside the CUDA graph capture below.
        self.solver.reorder_particles(self.state_0)

        # Replay the substep loop from a CUDA graph (eliminates per-substep launch
        # overhead, which dominates this uncaptured mesh+SDF scene).
        self.graph = None
        self.use_cuda_graph = wp.get_device(self.model.device).is_cuda

    @staticmethod
    def _add_cereal(builder, args):
        """Drop rings of torus cereal above the milk surface.

        Each cereal is a single torus mesh carrying an analytic-quality SDF, so it
        collides as a proper torus with both the milk (one SDF sample per nearby
        particle) and the other rigids -- no multi-sphere approximation.
        """
        # coarse collision mesh: the SDF (voxelized at cereal_sdf_resolution) keeps
        # the milk-facing shape smooth, while the few triangles keep torus-torus
        # and torus-bowl rigid contact counts low
        torus_mesh = create_torus_mesh(
            args.cereal_major_radius, args.cereal_minor_radius, segments_major=12, segments_minor=8
        )
        # one SDF shared across every cereal (the builder deduplicates by mesh)
        torus_mesh.build_sdf(
            max_resolution=args.cereal_sdf_resolution,
            narrow_band_range=(-args.cereal_minor_radius, args.cereal_minor_radius),
            margin=0.5 * args.cereal_minor_radius,
            cache_dir=_SDF_CACHE_DIR,
        )
        rng = np.random.default_rng(7)
        # golden-tan palette with slight per-piece variation
        base_color = np.array([0.82, 0.6, 0.3])

        bodies = []
        per_layer = 7
        layers = max(1, args.cereal_count // per_layer + (args.cereal_count % per_layer > 0))
        idx = 0
        for layer in range(layers):
            z = args.bowl_height + 0.03 + 0.035 * layer
            for k in range(per_layer):
                if idx >= args.cereal_count:
                    break
                if k == 0:
                    pos = np.array([0.0, 0.0, z])
                else:
                    ang = 2.0 * np.pi * (k - 1) / (per_layer - 1) + 0.5 * layer
                    ring_r = 0.45 * args.bowl_rim_radius
                    pos = np.array([ring_r * np.cos(ang), ring_r * np.sin(ang), z])
                pos[:2] += rng.uniform(-0.004, 0.004, size=2)
                rot = wp.quat_rpy(
                    float(rng.uniform(-0.4, 0.4)),
                    float(rng.uniform(-0.4, 0.4)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                )
                body = builder.add_body(
                    xform=wp.transform(wp.vec3(*pos), rot),
                    label=f"cereal_{idx}",
                )
                color = tuple(np.clip(base_color + rng.uniform(-0.06, 0.06, size=3), 0.0, 1.0))
                # the torus carries a texture SDF and collides as a proper torus
                # with both the milk (one SDF sample per nearby particle) and the
                # other rigids (a coarse collision mesh keeps the torus-torus and
                # torus-bowl contact counts low)
                builder.add_shape_mesh(
                    body,
                    mesh=torus_mesh,
                    cfg=newton.ModelBuilder.ShapeConfig(density=args.cereal_density, mu=0.3),
                    color=color,
                )
                bodies.append(body)
                idx += 1
        return bodies

    @staticmethod
    def _add_milk(builder, args):
        """Fill the bowl cavity with fluid particles up to the fill height."""
        spacing = args.spacing
        radius = 0.5 * spacing
        thickness = args.bowl_thickness
        height = args.bowl_height
        inner_rim = args.bowl_rim_radius - thickness
        inner_base = max(args.bowl_base_radius - thickness, 0.25 * args.bowl_base_radius)

        lo = np.array([-inner_rim, -inner_rim, thickness + radius])
        hi = np.array([inner_rim, inner_rim, args.fill_height])
        counts = np.maximum(((hi - lo) / spacing).astype(int) + 1, 1)
        axes = [lo[d] + spacing * np.arange(counts[d]) for d in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T

        rng = np.random.default_rng(11)
        points += rng.uniform(-0.1 * spacing, 0.1 * spacing, size=points.shape)

        # keep points inside the bowl cavity with a margin: invert the inner
        # wall profile of create_bowl_mesh to get the cavity radius at z
        z = points[:, 2]
        cos_ang = np.clip(1.0 - (z - thickness) / (height - thickness), 0.0, 1.0)
        sin_ang = np.sqrt(1.0 - cos_ang**2)
        cavity_r = inner_base + (inner_rim - inner_base) * sin_ang - spacing
        r_xy = np.linalg.norm(points[:, :2], axis=1)
        points = points[r_xy < cavity_r]

        mass = args.rest_density * spacing**3
        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[mass] * len(points),
            radius=[radius] * len(points),
            flags=[int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID)] * len(points),
        )

    def simulate(self):
        # Re-sort the milk into spatial order once per frame. Throwing the bowl
        # churns the milk out of its original layout, so the PBF density solve's
        # hash-grid neighbor reads lose cache locality and slow down; this
        # restores it (a pure relabel, see SolverXPBD.reorder_particles).
        self.solver.reorder_particles(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.use_cuda_graph:
            if self.graph is None:
                try:
                    with wp.ScopedCapture() as capture:
                        self.simulate()
                    self.graph = capture.graph
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

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("milk particles contain non-finite positions")
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("bodies contain non-finite transforms")
        # the milk should still mostly be inside the bowl
        radius = float(self.model.particle_max_radius)
        in_bowl = (q[:, 2] > -2.0 * radius) & (np.linalg.norm(q[:, :2], axis=1) < 2.0 * self.bowl_rim_radius)
        if np.count_nonzero(in_bowl) < 0.8 * len(q):
            raise ValueError("most of the milk left the bowl")
        # the cereal should float on the milk rather than sink to the bowl floor
        cereal_z = body_q[self.cereal_bodies, 2]
        if cereal_z.mean() < 0.3 * self.bowl_height:
            raise ValueError("cereal sank to the bottom of the bowl")
        if cereal_z.max() > self.bowl_height + 0.2:
            raise ValueError("cereal was ejected from the bowl")

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
        self.solver.update_render_particles(
            self.state_0,
            smoothing=self.render_smoothing,
        )
        # milk: opaque scattering body with a soft, broad sheen
        self.viewer.log_fluid(
            "/model/fluid",
            self.solver.render_positions,
            radii=self.model.particle_radius,
            radius_scale=1.8,
            color=self.fluid_color,
            reflectance=0.015,
            specular_intensity=0.18,
            specular_power=60.0,
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
        # ~100k milk particles. The bowl and cereal carry texture SDFs so milk
        # collides via one cheap SDF sample per particle (2 PBF iterations).
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--iterations", type=int, default=4)
        # Cap fluid neighbors above the in-bowl bulk (~200) so over-compressed
        # clumps from a hard throw can't stall a warp; 0 disables the cap.
        parser.add_argument("--max-neighbors", type=int, default=256)
        # Keep dynamic bodies from diverging when slammed by a hard throw; 0 disables.
        parser.add_argument("--body-max-velocity", type=float, default=20.0)
        parser.add_argument("--body-max-angular-velocity", type=float, default=50.0)
        # Capped low so a hard yank of the bowl can't fling milk into a wide
        # thin film (see soft_contact_mu); 1.5 m/s still slosh-es freely at the
        # bowl scale while keeping a spill compact and the PBF solve cheap.
        parser.add_argument("--max-velocity", type=float, default=1.5)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--bowl-base-radius", type=float, default=0.05)
        parser.add_argument("--bowl-rim-radius", type=float, default=0.12)
        parser.add_argument("--bowl-height", type=float, default=0.07)
        parser.add_argument("--bowl-thickness", type=float, default=0.008)
        parser.add_argument("--bowl-density", type=float, default=2000.0)
        parser.add_argument("--sdf-resolution", type=int, default=256, help="Bowl SDF grid resolution.")

        parser.add_argument("--cereal-count", type=int, default=21)
        parser.add_argument("--cereal-major-radius", type=float, default=0.016)
        parser.add_argument("--cereal-minor-radius", type=float, default=0.007)
        # light, like puffed cereal: heavier pieces sink through the coarse milk
        # and poke through the thin bowl floor
        parser.add_argument("--cereal-density", type=float, default=150.0)
        parser.add_argument("--cereal-sdf-resolution", type=int, default=64, help="Per-cereal torus SDF resolution.")

        # ~30k milk particles. Finer milk over-compresses ("jams") against the
        # walls when the bowl is dragged, which stalls the neighbor grid; this
        # resolution stays fast and contained under picking. The fill leaves
        # freeboard so a slosh has somewhere to go instead of jamming.
        parser.add_argument("--spacing", type=float, default=0.003)
        parser.add_argument("--fill-height", type=float, default=0.045)
        parser.add_argument("--rest-density", type=float, default=1000.0)
        # Milk has weak surface tension; the previous 1.0 was unphysically sticky
        # and, being unrelaxed, overpowered an under-relaxed density push and made
        # the milk collapse. A physical value both calms it and unlocks --relaxation.
        parser.add_argument("--cohesion", type=float, default=0.3)
        parser.add_argument("--viscosity", type=float, default=0.05)
        # The summed (standard-PBF) density correction overshoots at full strength,
        # leaving the milk buzzing instead of settling. Under-relaxing it lets the
        # milk come to rest; --iterations compensates for the gentler push.
        parser.add_argument("--relaxation", type=float, default=0.3)

        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument(
            "--fluid-color",
            type=float,
            nargs=4,
            default=(0.93, 0.91, 0.86, 0.06),
            help="Fluid albedo (rgb) and transmittance (a); the default is opaque milk",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
