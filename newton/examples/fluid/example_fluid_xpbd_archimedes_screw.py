# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Fluid XPBD Archimedes Screw
#
# A powered Archimedes screw lifts water through a transparent circular
# tube and a smooth descending bend onto a paddle wheel whose
# revolute joint has no actuator: without viewer interaction, all wheel
# motion comes from the fluid's two-way particle contact forces. Right-drag
# picking can apply an external disturbance to brake or accelerate the wheel.
# Water falling from the wheel returns to the basin for recirculation.
#
# Command: python -m newton.examples fluid_xpbd_archimedes_screw
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
from newton.examples.fluid.utils import ignore_shapes_for_picking, parse_particle_count, resolve_particle_grid

_REFERENCE_SPACING = 0.014
_FLUID_SIZE = (1.90, 0.50, 0.25)
_PIPE_INNER_RADIUS = 0.17
_PIPE_OUTER_RADIUS = 0.188
_PIPE_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_archimedes_pipe_sdf"
_SCREW_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_archimedes_screw_sdf"
_WHEEL_CENTER = (0.80, 0.0, 0.50)


@wp.kernel
def drive_archimedes_screw(
    motion: wp.array[float],
    dt: float,
    body: int,
    base_pos: wp.vec3,
    base_rot: wp.quat,
    speed: float,
    ramp_duration: float,
    body_q_0: wp.array[wp.transform],
    body_qd_0: wp.array[wp.spatial_vector],
    body_q_1: wp.array[wp.transform],
    body_qd_1: wp.array[wp.spatial_vector],
):
    """Rotate the powered screw while keeping its inclined axis fixed."""
    t = motion[0] + dt
    ramp = wp.min(t / wp.max(ramp_duration, dt), 1.0)
    angular_speed = speed * ramp
    angle = motion[1] + angular_speed * dt
    motion[0] = t
    motion[1] = angle

    local_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
    rotation = wp.mul(base_rot, local_rotation)
    world_axis = wp.quat_rotate(base_rot, wp.vec3(1.0, 0.0, 0.0))
    xform = wp.transform(base_pos, rotation)
    velocity = wp.spatial_vector(wp.vec3(0.0), world_axis * angular_speed)
    body_q_0[body] = xform
    body_qd_0[body] = velocity
    body_q_1[body] = xform
    body_qd_1[body] = velocity


def _build_helical_flight(length: float, inner_radius: float, outer_radius: float, turns: float, thickness: float):
    """Build a closed helical ribbon around the local X axis."""
    segments = max(int(32 * turns), 64)
    vertices = []

    for i in range(segments + 1):
        u = i / segments
        x = length * (u - 0.5)
        phase = -2.0 * np.pi * turns * u
        radial = np.array((0.0, np.cos(phase), np.sin(phase)), dtype=np.float64)
        phase_rate = -2.0 * np.pi * turns / length
        mid_radius = 0.5 * (inner_radius + outer_radius)
        tangent = np.array(
            (1.0, -mid_radius * phase_rate * np.sin(phase), mid_radius * phase_rate * np.cos(phase)),
            dtype=np.float64,
        )
        normal = np.cross(tangent, radial)
        normal /= np.linalg.norm(normal)
        offset = 0.5 * thickness * normal
        inner = np.array((x, inner_radius * radial[1], inner_radius * radial[2]))
        outer = np.array((x, outer_radius * radial[1], outer_radius * radial[2]))
        vertices.extend((inner + offset, outer + offset, inner - offset, outer - offset))

    indices = []
    for i in range(segments):
        a = 4 * i
        b = 4 * (i + 1)
        ip0, op0, im0, om0 = a, a + 1, a + 2, a + 3
        ip1, op1, im1, om1 = b, b + 1, b + 2, b + 3

        indices.extend((ip0, ip1, op1, ip0, op1, op0))
        indices.extend((im0, om1, im1, im0, om0, om1))
        indices.extend((op0, op1, om1, op0, om1, om0))
        indices.extend((ip0, im0, im1, ip0, im1, ip1))

    ip0, op0, im0, om0 = 0, 1, 2, 3
    end = 4 * segments
    ip1, op1, im1, om1 = end, end + 1, end + 2, end + 3
    indices.extend((ip0, op0, om0, ip0, om0, im0))
    indices.extend((ip1, im1, om1, ip1, om1, op1))

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        compute_inertia=False,
        is_solid=True,
    )


def _build_swept_pipe(centerline: np.ndarray, tangents: np.ndarray, radial_segments: int = 32) -> newton.Mesh:
    """Build a watertight circular shell around a sampled centerline."""
    vertices = []
    for center, tangent in zip(centerline, tangents, strict=True):
        unit_tangent = tangent / np.linalg.norm(tangent)
        lateral = np.array((0.0, 1.0, 0.0))
        normal = np.cross(unit_tangent, lateral)
        normal /= np.linalg.norm(normal)
        for radius in (_PIPE_OUTER_RADIUS, _PIPE_INNER_RADIUS):
            for j in range(radial_segments):
                phase = 2.0 * np.pi * j / radial_segments
                radial = np.cos(phase) * lateral + np.sin(phase) * normal
                vertices.append(center + radius * radial)

    ring_stride = 2 * radial_segments
    indices = []
    for i in range(len(centerline) - 1):
        ring_0 = i * ring_stride
        ring_1 = (i + 1) * ring_stride
        for j in range(radial_segments):
            j_next = (j + 1) % radial_segments

            outer_0 = ring_0 + j
            outer_0_next = ring_0 + j_next
            outer_1 = ring_1 + j
            outer_1_next = ring_1 + j_next
            indices.extend((outer_0, outer_0_next, outer_1_next, outer_0, outer_1_next, outer_1))

            inner_0 = ring_0 + radial_segments + j
            inner_0_next = ring_0 + radial_segments + j_next
            inner_1 = ring_1 + radial_segments + j
            inner_1_next = ring_1 + radial_segments + j_next
            indices.extend((inner_0, inner_1, inner_1_next, inner_0, inner_1_next, inner_0_next))

    start_outer = 0
    start_inner = radial_segments
    end_outer = (len(centerline) - 1) * ring_stride
    end_inner = end_outer + radial_segments
    for j in range(radial_segments):
        j_next = (j + 1) % radial_segments
        indices.extend(
            (
                start_outer + j,
                start_inner + j_next,
                start_outer + j_next,
                start_outer + j,
                start_inner + j,
                start_inner + j_next,
            )
        )
        indices.extend(
            (
                end_outer + j,
                end_outer + j_next,
                end_inner + j_next,
                end_outer + j,
                end_inner + j_next,
                end_inner + j,
            )
        )

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        compute_inertia=False,
        is_solid=True,
    )


def _build_delivery_pipe(base_pos: np.ndarray, screw_angle: float, sdf_resolution: int) -> newton.Mesh:
    """Build the screw tube and its tangent-continuous delivery bend."""
    screw_axis = np.array((np.cos(screw_angle), 0.0, np.sin(screw_angle)))
    straight_parameters = np.linspace(-0.56, 0.54, 13)
    straight = base_pos + straight_parameters[:, None] * screw_axis
    straight_tangents = np.repeat(screw_axis[None, :], len(straight), axis=0)

    bend_radius = 0.30
    outlet_angle = float(np.deg2rad(-28.0))
    curvature = -1.0 / bend_radius
    bend_angles = np.linspace(screw_angle, outlet_angle, 25)[1:]
    bend = []
    bend_tangents = []
    for angle in bend_angles:
        offset = np.array(
            (
                (np.sin(angle) - np.sin(screw_angle)) / curvature,
                0.0,
                (np.cos(screw_angle) - np.cos(angle)) / curvature,
            )
        )
        bend.append(straight[-1] + offset)
        bend_tangents.append((np.cos(angle), 0.0, np.sin(angle)))

    bend = np.asarray(bend)
    bend_tangents = np.asarray(bend_tangents)
    outlet_length = (0.52 - bend[-1, 0]) / np.cos(outlet_angle)
    outlet_parameters = np.linspace(0.0, outlet_length, 11)[1:]
    outlet_tangent = np.array((np.cos(outlet_angle), 0.0, np.sin(outlet_angle)))
    outlet = bend[-1] + outlet_parameters[:, None] * outlet_tangent
    outlet_tangents = np.repeat(outlet_tangent[None, :], len(outlet), axis=0)

    centerline = np.vstack((straight, bend, outlet))
    tangents = np.vstack((straight_tangents, bend_tangents, outlet_tangents))
    pipe = _build_swept_pipe(centerline, tangents)
    pipe.build_sdf(
        max_resolution=sdf_resolution,
        narrow_band_range=(-0.04, 0.04),
        margin=0.025,
        cache_dir=_PIPE_SDF_CACHE_DIR,
    )
    return pipe


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        # Reserve extra lattice sites because the initial packing removes points
        # intersecting the pump hardware; the remaining active count stays above
        # the user-facing target.
        particle_grid = resolve_particle_grid(int(np.ceil(1.40 * args.particle_count)), _FLUID_SIZE, _REFERENCE_SPACING)
        spacing = particle_grid.spacing
        radius = particle_grid.radius
        self.particle_radius = radius
        self.particle_render_radius = 1.05 * radius
        mass = args.rest_density * spacing**3
        dim_x, dim_y, dim_z = particle_grid.dimensions
        screw_angle = float(np.deg2rad(args.screw_angle))
        self.screw_base_pos = wp.vec3(-0.40, 0.0, 0.47)
        self.screw_base_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -screw_angle)

        builder = newton.ModelBuilder(up_axis="Z", gravity=(0.0, 0.0, args.gravity))
        builder.rigid_gap = 0.003
        builder.default_particle_radius = radius
        builder.default_shape_cfg.mu = 0.15

        floor_height = 0.08
        self._fill_reservoir(
            builder,
            args,
            spacing,
            radius,
            mass,
            (dim_x, dim_y, dim_z),
            floor_height,
            screw_angle,
        )

        non_pickable_shapes = []
        non_pickable_shapes.extend(self._add_return_basin(builder, args))

        self.screw_speed = args.screw_speed
        self.screw_body = self._add_screw(builder, args)
        non_pickable_shapes.append(self._add_delivery_pipe(builder, args, screw_angle))

        self.wheel_body, self.wheel_joint = self._add_passive_wheel(builder, args)
        # The revolute joint remains unactuated; right-drag picking is an explicit
        # external force that lets the user brake or accelerate the wheel.
        self.wheel_shapes = tuple(i for i, body in enumerate(builder.shape_body) if body == self.wheel_body)
        non_pickable_shapes.extend(self._add_wheel_support(builder))

        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.model.particle_max_velocity = 0.8 * args.wall_thickness / self.sim_dt
        self.model.soft_contact_mu = 0.15

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=spacing,
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
            fluid_relaxation=args.relaxation,
            fluid_max_neighbors=args.max_neighbors,
            joint_linear_relaxation=1.0,
            joint_angular_relaxation=1.0,
            # The wheel is the only dynamic body, so angular drag represents
            # passive bearing loss without affecting the powered screw.
            angular_damping=args.wheel_damping,
            body_max_velocity=8.0,
            body_max_angular_velocity=30.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.screw_motion = wp.zeros(2, dtype=float, device=self.model.device)

        self.fluid_color = tuple(args.fluid_color)
        self.particle_render_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(*self.fluid_color[:3]),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.fluid_radius_scale = args.fluid_radius_scale
        self.fluid_blur_radius = args.fluid_blur_radius
        self.render_smoothing = args.render_smoothing
        self.render_anisotropy_scale = args.render_anisotropy_scale
        self.render_particle_limit = args.render_particle_limit

        self.viewer.set_model(self.model)
        ignore_shapes_for_picking(self.viewer, self.model.shape_count, non_pickable_shapes)
        self.viewer.picking_enabled = True
        use_fluid_surface = args.render_mode == "fluid" and getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)

        self.solver.reorder_particles(self.state_0)
        self.graph = None
        self.use_cuda_graph = wp.get_device(self.model.device).is_cuda
        self._graph_key = None

    def _fill_reservoir(self, builder, args, spacing, radius, mass, dimensions, floor_height, screw_angle):
        """Pack water around, but never inside, the initially stationary screw."""
        dim_x, dim_y, dim_z = dimensions
        grid_x, grid_y, grid_z = np.meshgrid(
            -1.04 + spacing * np.arange(dim_x),
            -0.5 * (dim_y - 1) * spacing + spacing * np.arange(dim_y),
            floor_height + radius + spacing * np.arange(dim_z),
            indexing="ij",
        )
        points = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=1)
        rng = np.random.default_rng(0)
        points += rng.uniform(-0.04 * spacing, 0.04 * spacing, size=points.shape)

        delta = points - np.asarray(self.screw_base_pos)
        c = np.cos(screw_angle)
        s = np.sin(screw_angle)
        local_x = c * delta[:, 0] + s * delta[:, 2]
        local_y = delta[:, 1]
        local_z = -s * delta[:, 0] + c * delta[:, 2]
        radial = np.sqrt(local_y * local_y + local_z * local_z)
        within_length = np.abs(local_x) <= 0.5 * args.screw_length + radius

        shaft_clearance = args.screw_shaft_radius + 1.25 * radius
        overlaps_shaft = within_length & (radial < shaft_clearance)

        phase = -2.0 * np.pi * args.screw_turns * (local_x / args.screw_length + 0.5)
        particle_phase = np.arctan2(local_z, local_y)
        phase_delta = np.arctan2(np.sin(particle_phase - phase), np.cos(particle_phase - phase))
        surface_distance = np.abs(phase_delta) * np.maximum(radial, args.screw_shaft_radius)
        flight_clearance = 0.75 * args.wall_thickness + 1.25 * radius
        overlaps_flight = (
            within_length
            & (radial >= args.screw_shaft_radius - radius)
            & (radial <= args.screw_radius + radius)
            & (surface_distance < flight_clearance)
        )
        within_pipe = np.abs(local_x) <= 0.56 + radius
        overlaps_pipe_wall = (
            within_pipe & (radial >= _PIPE_INNER_RADIUS - radius) & (radial <= _PIPE_OUTER_RADIUS + radius)
        )
        wheel_delta = points - np.array(_WHEEL_CENTER)
        overlaps_wheel = (np.abs(wheel_delta[:, 1]) <= 0.17 + radius) & (
            np.linalg.norm(wheel_delta[:, (0, 2)], axis=1) <= 0.25 + radius
        )
        points = points[~(overlaps_shaft | overlaps_flight | overlaps_pipe_wall | overlaps_wheel)]

        flags = int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID)
        builder.add_particles(
            pos=points.tolist(),
            vel=[(0.0, 0.0, 0.0)] * len(points),
            mass=[mass] * len(points),
            radius=[radius] * len(points),
            flags=[flags] * len(points),
        )

    @staticmethod
    def _add_return_basin(builder, args):
        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.1)
        visual_cfg = newton.ModelBuilder.ShapeConfig()
        visual_cfg.mark_as_site()
        wall_color = (0.38, 0.48, 0.52)
        wall_opacity = args.wall_opacity
        wall_half_height = 0.24
        shapes = []

        return_angle = float(np.deg2rad(1.2))
        floor_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -return_angle)
        shapes.append(
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.04, 0.0, 0.03), floor_q),
                hx=1.12,
                hy=0.31,
                hz=0.025,
                cfg=wall_cfg,
                color=(0.34, 0.39, 0.40),
                label="return_floor",
            )
        )
        for side in (-1.0, 1.0):
            shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(0.04, side * 0.30, wall_half_height), wp.quat_identity()),
                    hx=1.12,
                    hy=0.025,
                    hz=wall_half_height,
                    cfg=wall_cfg,
                    color=wall_color,
                    opacity=0.0,
                    label=f"return_side_{side:+.0f}",
                )
            )
            shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(0.04, side * 0.30, wall_half_height), wp.quat_identity()),
                    hx=1.12,
                    hy=0.025,
                    hz=wall_half_height,
                    cfg=visual_cfg,
                    color=wall_color,
                    opacity=wall_opacity,
                    label=f"return_side_visual_{side:+.0f}",
                )
            )
        for side in (-1.0, 1.0):
            shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(0.04 + side * 1.145, 0.0, wall_half_height), wp.quat_identity()),
                    hx=0.025,
                    hy=0.325,
                    hz=wall_half_height,
                    cfg=wall_cfg,
                    color=wall_color,
                    opacity=0.0,
                    label=f"return_end_{side:+.0f}",
                )
            )
            shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(0.04 + side * 1.145, 0.0, wall_half_height), wp.quat_identity()),
                    hx=0.025,
                    hy=0.325,
                    hz=wall_half_height,
                    cfg=visual_cfg,
                    color=wall_color,
                    opacity=wall_opacity,
                    label=f"return_end_visual_{side:+.0f}",
                )
            )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.2))
        return shapes

    def _add_screw(self, builder, args):
        body = builder.add_link(
            xform=wp.transform(self.screw_base_pos, self.screw_base_rot),
            is_kinematic=True,
            label="powered_archimedes_screw",
        )
        flight = _build_helical_flight(
            args.screw_length,
            args.screw_shaft_radius,
            args.screw_radius,
            args.screw_turns,
            args.wall_thickness,
        )
        flight.build_sdf(
            max_resolution=args.screw_sdf_resolution,
            narrow_band_range=(-0.035, 0.035),
            margin=0.025,
            cache_dir=_SCREW_SDF_CACHE_DIR,
        )
        screw_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.2)
        builder.add_shape_mesh(
            body,
            mesh=flight,
            cfg=screw_cfg,
            color=(0.74, 0.42, 0.12),
            label="helical_flight",
        )
        shaft_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-0.5 * np.pi))
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(wp.vec3(0.0), shaft_q),
            radius=args.screw_shaft_radius,
            half_height=0.5 * args.screw_length + 0.025,
            cfg=screw_cfg,
            color=(0.28, 0.29, 0.30),
            label="screw_shaft",
        )
        return body

    def _add_delivery_pipe(self, builder, args, angle):
        pipe = _build_delivery_pipe(np.asarray(self.screw_base_pos), angle, args.pipe_sdf_resolution)
        return builder.add_shape_mesh(
            -1,
            mesh=pipe,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.04),
            color=(0.48, 0.56, 0.59),
            opacity=0.14,
            label="continuous_round_delivery_pipe",
        )

    @staticmethod
    def _add_passive_wheel(builder, args):
        center = wp.vec3(*_WHEEL_CENTER)
        wheel_mass = max(0.01386775 * args.wheel_density, 0.01)
        axial_inertia = 0.03 * wheel_mass
        radial_inertia = 0.55 * axial_inertia
        body = builder.add_link(
            xform=wp.transform(center, wp.quat_identity()),
            mass=wheel_mass,
            inertia=wp.mat33(
                radial_inertia,
                0.0,
                0.0,
                0.0,
                axial_inertia,
                0.0,
                0.0,
                0.0,
                radial_inertia,
            ),
            com=wp.vec3(0.0),
            lock_inertia=True,
            label="passive_water_wheel",
        )
        wheel_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.1)
        color = (0.24, 0.43, 0.66)

        paddle_count = args.paddle_count
        for i in range(paddle_count):
            phase = 2.0 * np.pi * i / paddle_count
            paddle_center = wp.vec3(0.15 * np.cos(phase), 0.0, 0.15 * np.sin(phase))
            paddle_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-phase))
            builder.add_shape_box(
                body,
                xform=wp.transform(paddle_center, paddle_q),
                hx=0.085,
                hy=0.145,
                hz=0.012,
                cfg=wheel_cfg,
                color=color,
                label=f"wheel_paddle_{i}",
            )

        axle_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(-0.5 * np.pi))
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(wp.vec3(0.0), axle_q),
            radius=0.045,
            half_height=0.16,
            cfg=wheel_cfg,
            color=(0.18, 0.22, 0.25),
            label="wheel_hub",
        )

        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            parent_xform=wp.transform(center, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(0.0, 1.0, 0.0),
            target_ke=0.0,
            target_kd=0.0,
            damping=0.0,
            friction=0.0,
            limit_lower=-1.0e6,
            limit_upper=1.0e6,
            limit_ke=0.0,
            limit_kd=0.0,
            actuator_mode=newton.JointTargetMode.NONE,
            label="passive_wheel_bearing",
        )
        builder.add_articulation([joint], label="passive_water_wheel")
        return body, joint

    @staticmethod
    def _add_wheel_support(builder):
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mark_as_site()
        shapes = []
        color = (0.28, 0.26, 0.23)
        for side in (-1.0, 1.0):
            y = side * 0.205
            shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(_WHEEL_CENTER[0], y, 0.5 * _WHEEL_CENTER[2]), wp.quat_identity()),
                    hx=0.035,
                    hy=0.025,
                    hz=0.5 * _WHEEL_CENTER[2],
                    cfg=cfg,
                    color=color,
                    label=f"wheel_support_{side:+.0f}",
                )
            )
        return shapes

    def _graph_key_tuple(self):
        return (round(self.screw_speed, 6), round(self.solver.fluid_viscosity, 6))

    def simulate(self):
        self.solver.reorder_particles(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                kernel=drive_archimedes_screw,
                dim=1,
                inputs=[
                    self.screw_motion,
                    self.sim_dt,
                    self.screw_body,
                    self.screw_base_pos,
                    self.screw_base_rot,
                    self.screw_speed,
                    1.2,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # Dense particle contacts can leave visible residual error in a
            # finite-iteration maximal-coordinate solve. Project through the
            # passive revolute articulation while preserving its free DOF.
            newton.eval_ik(self.model, self.state_0, self.model.joint_q, self.model.joint_qd)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

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
        _, self.screw_speed = ui.slider_float("Screw Speed", self.screw_speed, 0.0, 24.0, "%.1f rad/s")
        _, self.solver.fluid_viscosity = ui.slider_float("Viscosity", self.solver.fluid_viscosity, 0.0, 1.0, "%.2f")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise ValueError("Archimedes screw fluid contains non-finite state")
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            raise ValueError("Archimedes screw mechanism contains non-finite state")
        if q[:, 2].min() < self.particle_radius - 1.0e-4:
            raise ValueError("Water tunneled below the return basin")

        wheel_dof = int(self.model.joint_qd_start.numpy()[self.wheel_joint])
        wheel_mode = int(self.model.joint_target_mode.numpy()[wheel_dof])
        if wheel_mode != int(newton.JointTargetMode.NONE):
            raise ValueError("Paddle wheel must remain fully passive")
        wheel_ke = float(self.model.joint_target_ke.numpy()[wheel_dof])
        wheel_kd = float(self.model.joint_target_kd.numpy()[wheel_dof])
        if wheel_ke != 0.0 or wheel_kd != 0.0:
            raise ValueError("Paddle wheel must not have position or velocity drive gains")
        wheel_flags = int(self.model.body_flags.numpy()[self.wheel_body])
        if wheel_flags & int(newton.BodyFlags.KINEMATIC):
            raise ValueError("Paddle wheel must remain a dynamic body")
        picking = getattr(self.viewer, "picking", None)
        if picking is not None and len(picking.shape_pickable) > 0:
            pickable = picking.shape_pickable.numpy()
            if not np.all(pickable[list(self.wheel_shapes)]):
                raise ValueError("Paddle wheel shapes must remain available for right-drag picking")
        if self.sim_time > 2.5 and float(q[:, 2].max()) < 0.45:
            raise ValueError("Archimedes screw failed to lift water into the round delivery pipe")
        wheel_coord = int(self.model.joint_q_start.numpy()[self.wheel_joint])
        if self.sim_time > 3.5 and abs(float(self.model.joint_q.numpy()[wheel_coord])) < 0.05:
            raise ValueError("Passive paddle wheel did not rotate under water force")
        wheel_position_error = np.linalg.norm(body_q[self.wheel_body, :3] - np.array(_WHEEL_CENTER))
        wheel_axis_error = np.hypot(body_q[self.wheel_body, 3], body_q[self.wheel_body, 5])
        if wheel_position_error > 1.0e-4 or wheel_axis_error > 1.0e-4:
            raise ValueError("Paddle wheel escaped its revolute bearing")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        show_fluid = getattr(self.viewer, "show_fluid", False)
        show_particles = self.viewer.show_particles
        if show_fluid:
            self.viewer.show_fluid = False
        if show_particles:
            self.viewer.show_particles = False
        try:
            self.viewer.log_state(self.state_0)
        finally:
            if show_fluid:
                self.viewer.show_fluid = show_fluid
            if show_particles:
                self.viewer.show_particles = show_particles
        self.viewer.log_points(
            "/model/fluid_particles",
            points=self.state_0.particle_q,
            radii=self.particle_render_radius,
            colors=self.particle_render_colors,
            hidden=not show_particles,
        )
        if show_fluid and not show_particles:
            self._log_fluid_surface()
        self.viewer.end_frame()

    def _log_fluid_surface(self):
        self.solver.update_render_particles(
            self.state_0,
            smoothing=self.render_smoothing,
            anisotropy_scale=self.render_anisotropy_scale,
            max_particles=self.render_particle_limit,
        )
        self.viewer.log_fluid(
            "/model/fluid",
            self.solver.render_positions,
            radii=self.model.particle_max_radius,
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
        # Match the interactive tank's real-time PBF budget. The particle speed
        # cap keeps the faster pump contacts inside a particle-scale timestep.
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=2)
        parser.add_argument("--max-neighbors", type=int, default=128)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument(
            "--particle-count",
            type=parse_particle_count,
            default=100_000,
            help="Target water particle count; spacing and grid dimensions are derived automatically.",
        )
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--cohesion", type=float, default=0.5)
        parser.add_argument("--viscosity", type=float, default=0.2)
        parser.add_argument("--relaxation", type=float, default=0.6)

        parser.add_argument("--screw-length", type=float, default=1.05)
        parser.add_argument("--screw-radius", type=float, default=0.164)
        parser.add_argument("--screw-shaft-radius", type=float, default=0.038)
        parser.add_argument("--screw-turns", type=float, default=2.5)
        parser.add_argument("--screw-angle", type=float, default=31.0, help="Screw incline above horizontal [degrees].")
        parser.add_argument("--screw-speed", type=float, default=14.0, help="Powered screw angular speed [rad/s].")
        parser.add_argument("--wall-thickness", type=float, default=0.018)
        parser.add_argument("--wall-opacity", type=float, default=0.35)
        parser.add_argument("--screw-sdf-resolution", type=int, default=256, help="Helical flight SDF resolution.")
        parser.add_argument("--pipe-sdf-resolution", type=int, default=256, help="Round delivery pipe SDF resolution.")

        parser.add_argument("--paddle-count", type=int, default=10)
        parser.add_argument("--wheel-density", type=float, default=80.0)
        parser.add_argument(
            "--wheel-damping",
            type=float,
            default=1.5,
            help="Passive angular drag on the wheel bearing [1/s].",
        )

        parser.add_argument("--render-smoothing", type=float, default=0.6)
        parser.add_argument("--render-anisotropy-scale", type=float, default=1.0)
        parser.add_argument(
            "--render-particle-limit",
            type=int,
            default=0,
            help="Maximum particles used for fluid rendering; 0 renders all particles without temporal subsampling.",
        )
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.035)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))
        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.85, -2.75, 1.25))
        parser.add_argument("--camera-pitch", type=float, default=-18.0)
        parser.add_argument("--camera-yaw", type=float, default=122.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
