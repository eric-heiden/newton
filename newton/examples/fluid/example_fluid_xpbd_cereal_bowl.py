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
from newton.examples.fluid.utils import parse_particle_count, resolve_particle_spacing

# Cache cooked SDFs on disk so repeated runs skip the (slow) voxelization.
_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_cereal_bowl_sdf"
_REFERENCE_SPACING = 0.003


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


def _milk_particle_count(spacing, base_radius, rim_radius, height, thickness, fill_height):
    radius = 0.5 * spacing
    inner_rim = rim_radius - thickness
    inner_base = max(base_radius - thickness, 0.25 * base_radius)
    lower = np.array([-inner_rim, -inner_rim, thickness + radius])
    upper = np.array([inner_rim, inner_rim, fill_height])
    dimensions = np.maximum(((upper - lower) / spacing).astype(int) + 1, 1)
    axis_x = lower[0] + spacing * np.arange(dimensions[0])
    axis_y = lower[1] + spacing * np.arange(dimensions[1])
    axis_z = lower[2] + spacing * np.arange(dimensions[2])
    radial_sq = np.sort((axis_x[:, None] ** 2 + axis_y[None, :] ** 2).ravel())
    cos_angle = np.clip(1.0 - (axis_z - thickness) / (height - thickness), 0.0, 1.0)
    sin_angle = np.sqrt(1.0 - cos_angle**2)
    cavity_radius = inner_base + (inner_rim - inner_base) * sin_angle - spacing
    return int(np.searchsorted(radial_sq, cavity_radius * cavity_radius, side="left").sum())


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        spacing, _ = resolve_particle_spacing(
            args.particle_count,
            _REFERENCE_SPACING,
            lambda candidate: _milk_particle_count(
                candidate,
                args.bowl_base_radius,
                args.bowl_rim_radius,
                args.bowl_height,
                args.bowl_thickness,
                args.fill_height,
            ),
        )
        radius = 0.5 * spacing

        self.bowl_height = args.bowl_height
        self.bowl_rim_radius = args.bowl_rim_radius
        self.cereal_major_radius = args.cereal_major_radius
        self.cereal_minor_radius = args.cereal_minor_radius
        self.cereal_outer_radius = args.cereal_major_radius + args.cereal_minor_radius
        self.cereal_should_float = args.cereal_density < args.rest_density
        self.body_max_velocity = args.body_max_velocity
        self.body_max_angular_velocity = args.body_max_angular_velocity

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

        # The torus SDF handles milk contact; analytic capsules provide robust
        # rigid contact while preserving each ring's hole.
        self.cereal_bodies = self._add_cereal(builder, args, spacing)

        # milk: a cylinder of fluid particles trimmed to the bowl cavity
        self._add_milk(builder, args, spacing)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.03))

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
        self.model.rigid_contact_max = 65536
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
            # Keep pathological contact corrections bounded without clipping
            # interactive throws or ordinary ring rolling.
            body_max_velocity=args.body_max_velocity,
            body_max_angular_velocity=args.body_max_angular_velocity,
            rigid_contact_relaxation=args.rigid_contact_relaxation,
            angular_damping=args.angular_damping,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # Only the bowl and nearby cereal proxies can contact a given particle;
        # avoid allocating/iterating the full particle-by-shape cross product.
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
    def _add_cereal(builder, args, spacing):
        """Place rings of torus cereal near the milk surface.

        Each cereal uses its torus mesh for rendering, inertia, and milk contact.
        Tangent capsules form a compound torus for robust rigid contact without
        the redundant manifolds of concave mesh contact.
        """
        torus_mesh = create_torus_mesh(
            args.cereal_major_radius, args.cereal_minor_radius, segments_major=12, segments_minor=8
        )
        torus_mesh.build_sdf(
            target_voxel_size=0.5 * spacing,
            narrow_band_range=(-2.0 * spacing, 2.0 * spacing),
            margin=2.0 * spacing,
            cache_dir=_SDF_CACHE_DIR,
        )
        collision_segments = max(args.cereal_collision_segments, 3)
        rng = np.random.default_rng(7)
        # golden-tan palette with slight per-piece variation
        base_color = np.array([0.82, 0.6, 0.3])

        bodies = []
        outer_radius = args.cereal_major_radius + args.cereal_minor_radius
        lattice_spacing = 2.0 * outer_radius
        usable_center_radius = max(args.bowl_rim_radius - outer_radius, 0.0)
        axial_radius = max(int(usable_center_radius / lattice_spacing), 0)
        hex_slots = []
        for axial_q in range(-axial_radius, axial_radius + 1):
            for axial_r in range(-axial_radius, axial_radius + 1):
                if max(abs(axial_q), abs(axial_r), abs(-axial_q - axial_r)) <= axial_radius:
                    x = lattice_spacing * (axial_q + 0.5 * axial_r)
                    y = lattice_spacing * (0.5 * np.sqrt(3.0) * axial_r)
                    hex_slots.append((x, y))
        hex_slots.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], np.arctan2(p[1], p[0])))

        # Start on the milk instead of dropping a tall artificial stack. Slots
        # are generated center-out so the default scene remains a calm layer;
        # unusually large cereal counts still stack only after the slot set is
        # exhausted.
        layer_capacity = len(hex_slots)
        base_z = args.fill_height + args.cereal_minor_radius + spacing
        for idx in range(args.cereal_count):
            layer, slot = divmod(idx, layer_capacity)
            x, y = hex_slots[slot]
            if layer:
                # Upper layers sit over triangular gaps rather than directly
                # above another ring, avoiding a high-energy vertical impact.
                x += 0.5 * lattice_spacing
                y += np.sqrt(3.0) / 6.0 * lattice_spacing
            layer_angle = 0.35 * layer
            cos_angle = np.cos(layer_angle)
            sin_angle = np.sin(layer_angle)
            pos = np.array(
                [
                    cos_angle * x - sin_angle * y,
                    sin_angle * x + cos_angle * y,
                    base_z + 1.5 * outer_radius * layer,
                ]
            )
            rot = wp.quat_rpy(
                float(rng.uniform(-0.2, 0.2)),
                float(rng.uniform(-0.2, 0.2)),
                float(rng.uniform(0.0, 2.0 * np.pi)),
            )
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*pos), rot),
                label=f"cereal_{idx}",
            )
            color = tuple(np.clip(base_color + rng.uniform(-0.06, 0.06, size=3), 0.0, 1.0))
            torus_cfg = newton.ModelBuilder.ShapeConfig(
                density=args.cereal_density,
                mu=0.2,
                has_shape_collision=False,
                has_particle_collision=True,
            )
            builder.add_shape_mesh(
                body,
                mesh=torus_mesh,
                cfg=torus_cfg,
                color=color,
            )

            rigid_collider_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=0.03,
                mu_torsional=0.0,
                mu_rolling=0.0,
                has_particle_collision=False,
                is_visible=False,
            )
            capsule_half_height = args.cereal_major_radius * np.sin(np.pi / collision_segments)
            for segment in range(collision_segments):
                angle = 2.0 * np.pi * segment / collision_segments
                radial = wp.vec3(np.cos(angle), np.sin(angle), 0.0)
                tangent = wp.vec3(-np.sin(angle), np.cos(angle), 0.0)
                builder.add_shape_capsule(
                    body,
                    xform=wp.transform(
                        radial * args.cereal_major_radius,
                        wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), tangent),
                    ),
                    radius=args.cereal_minor_radius,
                    half_height=capsule_half_height,
                    cfg=rigid_collider_cfg,
                )
            bodies.append(body)
        return bodies

    @staticmethod
    def _add_milk(builder, args, spacing):
        """Fill the bowl cavity with fluid particles up to the fill height."""
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

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("milk particles contain non-finite positions")
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("bodies contain non-finite transforms")
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_qd)):
            raise ValueError("bodies contain non-finite velocities")

        bowl_q = body_q[self.bowl_body]
        milk_local = self._points_to_body_frame(q, bowl_q)
        cereal_local = self._points_to_body_frame(body_q[self.cereal_bodies, :3], bowl_q)

        # Validate against the moving bowl, not the world-space ground. The old
        # checks allowed spilled milk and cereal to settle nearby and still pass.
        radius = float(self.model.particle_max_radius)
        if milk_local[:, 2].min() < radius - 5.0e-4:
            raise ValueError("milk tunneled through the bowl floor")
        milk_r = np.linalg.norm(milk_local[:, :2], axis=1)
        in_bowl = (milk_local[:, 2] < self.bowl_height + 2.0 * radius) & (milk_r < self.bowl_rim_radius + 2.0 * radius)
        if np.count_nonzero(in_bowl) < 0.99 * len(q):
            raise ValueError("more than one percent of the milk left the bowl")

        cereal_r = np.linalg.norm(cereal_local[:, :2], axis=1)
        if np.any(cereal_r > self.bowl_rim_radius + self.cereal_outer_radius):
            raise ValueError("cereal left the bowl")
        torus_axes_world = self._rotate_vectors(
            body_q[self.cereal_bodies, 3:7],
            np.broadcast_to((0.0, 0.0, 1.0), (len(self.cereal_bodies), 3)),
        )
        bowl_q_inv = np.array([-bowl_q[3], -bowl_q[4], -bowl_q[5], bowl_q[6]])
        torus_axes_local = self._rotate_vectors(
            np.broadcast_to(bowl_q_inv, (len(self.cereal_bodies), 4)),
            torus_axes_world,
        )
        vertical_extent = self.cereal_minor_radius + self.cereal_major_radius * np.sqrt(
            np.maximum(1.0 - torus_axes_local[:, 2] ** 2, 0.0)
        )
        if np.any(cereal_local[:, 2] < vertical_extent - 2.0e-3):
            raise ValueError("cereal tunneled through the bowl floor")
        if cereal_local[:, 2].max() > self.bowl_height + 2.0 * self.cereal_outer_radius:
            raise ValueError("cereal was ejected from the bowl")
        if self.cereal_should_float and self.sim_time > 0.5:
            milk_surface = np.percentile(milk_local[in_bowl, 2], 95)
            if np.median(cereal_local[:, 2]) <= milk_surface:
                raise ValueError("low-density cereal failed to float on the milk")

        cereal_speed = np.linalg.norm(body_qd[self.cereal_bodies, :3], axis=1)
        if self.body_max_velocity > 0.0 and np.percentile(cereal_speed, 95) > 1.05 * self.body_max_velocity:
            raise ValueError("cereal linear velocity exceeded the configured cap")
        cereal_angular_speed = np.linalg.norm(body_qd[self.cereal_bodies, 3:], axis=1)
        if (
            self.body_max_angular_velocity > 0.0
            and np.percentile(cereal_angular_speed, 95) > 1.05 * self.body_max_angular_velocity
        ):
            raise ValueError("cereal angular velocity exceeded the configured cap")

    @staticmethod
    def _points_to_body_frame(points, body_q):
        relative = points - body_q[:3]
        q_xyz_inv = -body_q[3:6]
        twice_cross = 2.0 * np.cross(q_xyz_inv, relative)
        return relative + body_q[6] * twice_cross + np.cross(q_xyz_inv, twice_cross)

    @staticmethod
    def _rotate_vectors(quaternions, vectors):
        twice_cross = 2.0 * np.cross(quaternions[:, :3], vectors)
        return vectors + quaternions[:, 3:4] * twice_cross + np.cross(quaternions[:, :3], twice_cross)

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
            radius_scale=1.9,
            color=self.fluid_color,
            reflectance=0.025,
            specular_intensity=0.16,
            specular_power=45.0,
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
        # Eight substeps keep rigid motion below the bowl wall thickness, while
        # four relaxed PBF iterations keep the ~30k-particle milk incompressible.
        parser.add_argument("--substeps", type=int, default=8)
        parser.add_argument("--iterations", type=int, default=4)
        # Cap fluid neighbors above the in-bowl bulk (~200) so over-compressed
        # clumps from a hard throw can't stall a warp; 0 disables the cap.
        parser.add_argument("--max-neighbors", type=int, default=256)
        # Safety caps stay above interactive throws and normal cereal rolling
        # speeds; 0 disables.
        parser.add_argument("--body-max-velocity", type=float, default=4.0)
        parser.add_argument("--body-max-angular-velocity", type=float, default=180.0)
        parser.add_argument("--rigid-contact-relaxation", type=float, default=0.65)
        parser.add_argument("--angular-damping", type=float, default=0.0)
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

        parser.add_argument("--cereal-count", type=int, default=19)
        parser.add_argument("--cereal-major-radius", type=float, default=0.016)
        parser.add_argument("--cereal-minor-radius", type=float, default=0.007)
        parser.add_argument("--cereal-collision-segments", type=int, default=8)
        # Light like puffed cereal; the torus SDF displaces its actual milk volume.
        parser.add_argument("--cereal-density", type=float, default=150.0)
        parser.add_argument(
            "--particle-count",
            type=parse_particle_count,
            default=30_000,
            help="Target milk particle count; spacing, radius, mass, and the carved fill grid are derived automatically.",
        )
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
        parser.add_argument("--relaxation", type=float, default=0.6)

        parser.add_argument("--render-smoothing", type=float, default=0.7)
        parser.add_argument(
            "--fluid-color",
            type=float,
            nargs=4,
            default=(0.97, 0.98, 0.965, 0.015),
            help="Fluid albedo (rgb) and transmittance (a); the default is opaque milk",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
