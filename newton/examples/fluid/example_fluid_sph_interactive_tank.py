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
def deactivate_particles_overlapping_boxes(
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    box_body_ids: wp.array[int],
    box_half_extents: wp.array[wp.vec3],
    clearance: float,
):
    tid = wp.tid()
    x = particle_q[tid]
    for box_idx in range(box_body_ids.shape[0]):
        body = box_body_ids[box_idx]
        X_bw = wp.transform_inverse(body_q[body])
        local = wp.transform_point(X_bw, x)
        half = box_half_extents[box_idx] + wp.vec3(clearance)
        if wp.abs(local[0]) <= half[0] and wp.abs(local[1]) <= half[1] and wp.abs(local[2]) <= half[2]:
            particle_flags[tid] = wp.int32(0)
            return


@wp.kernel
def reset_box_water_heights(
    box_water_height_raw: wp.array[float],
    floor_height: float,
):
    tid = wp.tid()
    box_water_height_raw[tid] = floor_height


@wp.kernel
def measure_box_water_heights(
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    box_body_ids: wp.array[int],
    box_half_extents: wp.array[wp.vec3],
    sample_clearance: float,
    sample_margin: float,
    box_water_height_raw: wp.array[float],
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_q[tid]
    for box_idx in range(box_body_ids.shape[0]):
        body = box_body_ids[box_idx]
        c = wp.transform_get_translation(body_q[body])
        half = box_half_extents[box_idx]
        dx = wp.abs(x[0] - c[0])
        dy = wp.abs(x[1] - c[1])
        # Sample the free surface in a ring around the box footprint so
        # particles splashed onto or climbing the box do not inflate the
        # measured water level (which would levitate the box).
        inside_outer = dx <= half[0] + sample_margin and dy <= half[1] + sample_margin
        outside_inner = dx > half[0] + sample_clearance or dy > half[1] + sample_clearance
        if inside_outer and outside_inner:
            wp.atomic_max(box_water_height_raw, box_idx, x[2])


@wp.kernel
def smooth_box_water_heights(
    box_water_height_raw: wp.array[float],
    box_water_height: wp.array[float],
    blend_up: float,
    blend_down: float,
):
    tid = wp.tid()
    raw = box_water_height_raw[tid]
    smoothed = box_water_height[tid]
    # Rise slowly so brief splashes in the sampling ring cannot inflate the
    # perceived water level (and levitate the box), but fall quickly when the
    # surface recedes.
    blend = wp.where(raw > smoothed, blend_up, blend_down)
    box_water_height[tid] = wp.lerp(smoothed, raw, blend)


@wp.kernel
def apply_body_buoyancy_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_ids: wp.array[int],
    box_half_extents: wp.array[wp.vec3],
    box_water_height: wp.array[float],
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    gravity: wp.vec3,
    rest_density: float,
    buoyancy_scale: float,
    linear_drag: float,
    quadratic_drag: float,
    angular_drag: float,
    floor_stiffness: float,
    floor_damping: float,
    floor_friction: float,
    wall_stiffness: float,
    wall_damping: float,
):
    tid = wp.tid()
    body = body_ids[tid]
    if body < 0:
        return

    X_wb = body_q[body]
    x = wp.transform_point(X_wb, body_com[body])
    q = wp.transform_get_rotation(X_wb)
    v = wp.spatial_top(body_qd[body])
    w = wp.spatial_bottom(body_qd[body])
    mass = body_mass[body]
    half = box_half_extents[tid]
    water_level = box_water_height[tid]

    axis_x = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    axis_y = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    axis_z = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    cell_volume = half[0] * half[1] * half[2]
    cell_half_up = 0.5 * (wp.abs(axis_x[2]) * half[0] + wp.abs(axis_y[2]) * half[1] + wp.abs(axis_z[2]) * half[2])
    cell_half_up = wp.max(cell_half_up, 1.0e-4)

    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    submerged = float(0.0)

    # Archimedes per octant: each of the 8 cells contributes buoyancy and drag
    # at its own center, which yields net lift plus a natural righting torque,
    # and the locally measured water height couples the box to passing waves.
    for corner in range(8):
        sx = float(1.0)
        sy = float(1.0)
        sz = float(1.0)
        if (corner & 1) == 0:
            sx = -1.0
        if (corner & 2) == 0:
            sy = -1.0
        if (corner & 4) == 0:
            sz = -1.0

        r = axis_x * (sx * half[0] * 0.5) + axis_y * (sy * half[1] * 0.5) + axis_z * (sz * half[2] * 0.5)
        cell_center = x + r
        depth_fraction = (water_level - cell_center[2] + cell_half_up) / (2.0 * cell_half_up)
        depth_fraction = wp.min(wp.max(depth_fraction, 0.0), 1.0)
        if depth_fraction > 0.0:
            displaced_mass = rest_density * cell_volume * depth_fraction
            cell_buoyancy = -gravity * (displaced_mass * buoyancy_scale)
            force += cell_buoyancy
            torque += wp.cross(r, cell_buoyancy)

            point_v = v + wp.cross(w, r)
            drag = -point_v * (displaced_mass * (linear_drag + quadratic_drag * wp.length(point_v)))
            force += drag
            torque += wp.cross(r, drag)
            submerged += depth_fraction * 0.125

    torque -= w * (mass * angular_drag * submerged)

    # Sphere-approximated box-box repulsion so floating boxes do not raft into
    # each other (the rigid integrator in this example resolves no contacts).
    # Each box applies its own half of the symmetric pair force.
    my_radius = 0.40 * (half[0] + half[1] + half[2])
    for other_idx in range(box_half_extents.shape[0]):
        if other_idx != tid:
            other_body = body_ids[other_idx]
            if other_body >= 0:
                other_half = box_half_extents[other_idx]
                other_center = wp.transform_point(body_q[other_body], body_com[other_body])
                radius_sum = my_radius + 0.40 * (other_half[0] + other_half[1] + other_half[2])
                delta = x - other_center
                dist = wp.length(delta)
                if dist > 1.0e-6 and dist < radius_sum:
                    n = delta / dist
                    rel_v = v - wp.spatial_top(body_qd[other_body])
                    repulsion = mass * (220.0 * (radius_sum - dist) - 10.0 * wp.dot(rel_v, n))
                    if repulsion > 0.0:
                        force += n * repulsion

    # Tank floor contact at the true corners. The rigid integrator in this
    # example does not resolve contacts, so denser-than-water boxes need a
    # floor reaction to rest on once they sink.
    floor_z = bounds_lower[2]
    for corner in range(8):
        sx = float(1.0)
        sy = float(1.0)
        sz = float(1.0)
        if (corner & 1) == 0:
            sx = -1.0
        if (corner & 2) == 0:
            sy = -1.0
        if (corner & 4) == 0:
            sz = -1.0

        r = axis_x * (sx * half[0]) + axis_y * (sy * half[1]) + axis_z * (sz * half[2])
        corner_pos = x + r
        penetration = floor_z - corner_pos[2]
        if penetration > 0.0:
            corner_v = v + wp.cross(w, r)
            normal_force = (mass * 0.125) * (floor_stiffness * penetration - floor_damping * corner_v[2])
            normal_force = wp.max(normal_force, 0.0)
            contact_force = wp.vec3(0.0, 0.0, normal_force)
            contact_force -= wp.vec3(corner_v[0], corner_v[1], 0.0) * (floor_friction * mass * 0.125)
            force += contact_force
            torque += wp.cross(r, contact_force)

    margin = float(0.18)
    if x[0] < bounds_lower[0] + margin:
        force[0] += mass * (wall_stiffness * (bounds_lower[0] + margin - x[0]) - wall_damping * v[0])
    elif x[0] > bounds_upper[0] - margin:
        force[0] += mass * (wall_stiffness * (bounds_upper[0] - margin - x[0]) - wall_damping * v[0])

    if x[1] < bounds_lower[1] + margin:
        force[1] += mass * (wall_stiffness * (bounds_lower[1] + margin - x[1]) - wall_damping * v[1])
    elif x[1] > bounds_upper[1] - margin:
        force[1] += mass * (wall_stiffness * (bounds_upper[1] - margin - x[1]) - wall_damping * v[1])

    if x[2] > bounds_upper[2] - margin:
        force[2] += mass * (wall_stiffness * (bounds_upper[2] - margin - x[2]) - wall_damping * v[2])

    wp.atomic_add(body_f, body, wp.spatial_vector(force, torque))


@wp.kernel
def copy_body_kinematics(
    src_body_q: wp.array[wp.transform],
    src_body_qd: wp.array[wp.spatial_vector],
    dst_body_q: wp.array[wp.transform],
    dst_body_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    dst_body_q[tid] = src_body_q[tid]
    dst_body_qd[tid] = src_body_qd[tid]


@wp.kernel
def add_body_wrenches(
    dst_body_f: wp.array[wp.spatial_vector],
    src_body_f: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    dst_body_f[tid] = dst_body_f[tid] + src_body_f[tid]


@wp.kernel
def clamp_body_water_velocities(
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_ids: wp.array[int],
    max_linear_speed: float,
    max_angular_speed: float,
    max_torque: float,
):
    tid = wp.tid()
    body = body_ids[tid]
    if body < 0:
        return

    qd = body_qd[body]
    v = wp.spatial_top(qd)
    w = wp.spatial_bottom(qd)

    speed = wp.length(v)
    if max_linear_speed > 0.0 and speed > max_linear_speed:
        v *= max_linear_speed / speed

    angular_speed = wp.length(w)
    if max_angular_speed > 0.0 and angular_speed > max_angular_speed:
        w *= max_angular_speed / angular_speed

    body_qd[body] = wp.spatial_vector(v, w)

    wrench = body_f[body]
    force = wp.spatial_top(wrench)
    torque = wp.spatial_bottom(wrench)
    torque_mag = wp.length(torque)
    if max_torque > 0.0 and torque_mag > max_torque:
        torque *= max_torque / torque_mag
        body_f[body] = wp.spatial_vector(force, torque)


@wp.kernel
def apply_particle_box_coupling(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_mass: wp.array[float],
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
    mass = particle_mass[i]
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
            r = closest_world - wp.transform_point(X_wb, body_com[body])
            body_v = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), r)
            rel_n = wp.dot(v - body_v, normal)
            magnitude = stiffness * penetration - damping * rel_n
            if magnitude > 0.0:
                magnitude = wp.min(magnitude, 80.0)
                force = normal * (magnitude * mass)
                particle_force += force
                body_feedback_scale = splash_velocity_gain
                body_force = -force * body_feedback_scale
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
        self.gravity = wp.vec3(0.0, 0.0, float(args.gravity))
        if args.water_level is None:
            self.water_level = float(args.emit_lower[2] + args.spacing * max(args.dim_z - 1, 0) + args.radius)
        else:
            self.water_level = float(args.water_level)
        self.show_bounds = args.show_bounds
        self.particle_radius = args.radius
        self.rest_density = args.rest_density
        self.coupling_stiffness = args.coupling_stiffness
        self.coupling_damping = args.coupling_damping
        self.buoyancy_scale = args.buoyancy_scale
        self.box_linear_drag = args.box_linear_drag
        self.box_quadratic_drag = args.box_quadratic_drag
        self.box_angular_drag = args.box_angular_drag
        self.box_floor_stiffness = args.box_floor_stiffness
        self.box_floor_damping = args.box_floor_damping
        self.box_floor_friction = args.box_floor_friction
        self.box_wall_stiffness = args.box_wall_stiffness
        self.box_wall_damping = args.box_wall_damping
        self.box_max_linear_speed = args.box_max_linear_speed
        self.box_max_angular_speed = args.box_max_angular_speed
        self.box_max_torque = args.box_max_torque
        self.splash_velocity_gain = args.splash_velocity_gain
        self.pick_stiffness = args.pick_stiffness
        self.pick_damping = args.pick_damping
        self.show_box_guides = args.show_box_guides
        self.fluid_color = tuple(args.fluid_color)
        self.fluid_ior = args.fluid_ior
        self.fluid_blur_radius = args.fluid_blur_radius
        self.fluid_radius_scale = args.fluid_radius_scale
        self.foam_color = tuple(args.foam_color)
        self.foam_radius = args.foam_radius
        self.foam_motion_blur = args.foam_motion_blur

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
        self.box_densities: list[float] = []
        self.box_guide_colors: list[wp.vec3] = []
        self._add_boxes(builder, args)

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.box_body_ids_wp = wp.array(self.box_body_ids, dtype=int, device=self.model.device)
        self.box_half_extents_wp = wp.array(self.box_half_extents, dtype=wp.vec3, device=self.model.device)
        # Per-box local water surface heights measured from the SPH particles;
        # the smoothed array drives buoyancy so boxes ride passing waves.
        box_count = max(len(self.box_body_ids), 1)
        self.box_water_height_raw = wp.full(box_count, value=self.water_level, dtype=float, device=self.model.device)
        self.box_water_height = wp.full(box_count, value=self.water_level, dtype=float, device=self.model.device)
        self._carve_initial_box_volumes(args.fluid_carve_clearance)

        self.sph_solver = SolverSPH(
            self.model,
            smoothing_length=args.smoothing_length,
            rest_density=args.rest_density,
            gas_constant=args.gas_constant,
            viscosity=args.viscosity,
            particle_friction=args.particle_friction,
            particle_collision_margin=args.particle_collision_margin,
            cohesion=args.cohesion,
            surface_tension=args.surface_tension,
            vorticity_confinement=args.vorticity_confinement,
            solid_pressure=args.solid_pressure,
            buoyancy=args.buoyancy,
            xsph_strength=args.xsph_strength,
            free_surface_drag=args.free_surface_drag,
            dissipation=args.dissipation,
            velocity_damping=args.velocity_damping,
            sleep_threshold=args.sleep_threshold,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
            boundary_damping=args.boundary_damping,
            shape_collision_distance=args.shape_collision_distance,
            shape_collision_margin=args.shape_collision_margin,
            shape_restitution=args.shape_restitution,
            shape_friction=args.shape_friction,
            shape_adhesion=args.shape_adhesion,
            max_velocity=args.max_velocity,
            max_acceleration=args.max_acceleration,
            pbf_iterations=args.pbf_iterations,
            pbf_relaxation=args.pbf_relaxation,
            pbf_artificial_pressure=args.pbf_artificial_pressure,
            # The example applies its own displaced-volume buoyancy plus scaled
            # splash feedback; full solver contact reactions would double-count
            # buoyancy and let the coarse particle bed carry dense boxes.
            shape_collision_body_feedback=args.sph_body_feedback,
            max_diffuse_particles=args.fluid_diffuse_max_particles,
            diffuse_threshold=args.fluid_diffuse_threshold,
            diffuse_lifetime=args.fluid_diffuse_lifetime,
            diffuse_drag=args.fluid_diffuse_drag,
            diffuse_buoyancy=args.fluid_diffuse_buoyancy,
            diffuse_ballistic=args.fluid_diffuse_ballistic,
            diffuse_spawn_probability=args.fluid_diffuse_spawn_probability,
            render_smoothing=args.fluid_render_smoothing,
            render_anisotropy_scale=args.fluid_render_anisotropy_scale,
            render_anisotropy_min=args.fluid_render_anisotropy_min,
            render_anisotropy_max=args.fluid_render_anisotropy_max,
            render_update_interval=args.fluid_render_update_interval,
            diffuse_update_interval=args.fluid_diffuse_update_interval,
        )
        self.rigid_integrator = newton.solvers.SolverSemiImplicit(self.model, angular_damping=args.angular_damping)

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        self._apply_picking_params()
        self.viewer.show_particles = args.render_mode == "particles"
        self.viewer.show_fluid = args.render_mode == "fluid"
        self.viewer.show_fluid_diffuse = args.show_diffuse
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
        density_fractions = tuple(args.box_density_fractions)
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
            # The box mass comes purely from its shape density, expressed as a
            # fraction of the fluid rest density: fractions below 1 float (the
            # smaller, the higher they ride), fractions above 1 sink.
            density = float(density_fractions[i % len(density_fractions)]) * args.rest_density
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, y, z), q),
                label=f"water_cube_{i}",
            )
            builder.add_shape_box(
                body,
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=0.18),
                color=colors[i % len(colors)],
            )
            self.box_body_ids.append(body)
            self.box_half_extents.append(wp.vec3(hx, hy, hz))
            self.box_densities.append(density)
            self.box_guide_colors.append(wp.vec3(colors[i % len(colors)]))

    def _carve_initial_box_volumes(self, clearance: float):
        if clearance < 0.0 or len(self.box_body_ids) == 0:
            return
        wp.launch(
            kernel=deactivate_particles_overlapping_boxes,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self.model.particle_flags,
                self.state_0.body_q,
                self.box_body_ids_wp,
                self.box_half_extents_wp,
                clearance,
            ],
            device=self.model.device,
        )

    def _configure_render_environment(self, args):
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is None:
            return

        renderer._env_intensity = float(args.environment_intensity)
        env_path = getattr(renderer, "_env_path", None)
        if env_path is not None and hasattr(renderer, "set_environment_map"):
            renderer.set_environment_map(env_path, intensity=args.environment_intensity)
            renderer._env_path = None

        sun = np.asarray(args.sun_direction, dtype=np.float32)
        sun_norm = float(np.linalg.norm(sun))
        if sun_norm > 1.0e-6:
            renderer._sun_direction = sun / sun_norm

        renderer.sky_upper = (0.42, 0.74, 1.0)
        renderer.sky_lower = (0.72, 0.82, 0.78)
        renderer.ambient_sky = (0.92, 0.96, 1.0)
        renderer.ambient_ground = (0.48, 0.50, 0.52)
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
        self._captured_params = None
        if not self.capture_graph:
            return
        if not wp.get_device().is_cuda:
            self.capture_graph = False
            warnings.warn("SPH interactive graph capture is only available on CUDA devices.", stacklevel=2)
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self._captured_params = self._graph_params()
        except Exception as exc:
            self.capture_graph = False
            warnings.warn(
                f"Interactive SPH graph capture failed; falling back to uncaptured stepping: {exc}",
                stacklevel=2,
            )
            self.graph = None

    def _graph_params(self):
        # Every Python-side value baked into the captured launch sequence;
        # changing any of them (e.g. via a GUI slider) requires a re-capture.
        solver = self.sph_solver
        return (
            self.coupling_stiffness,
            self.coupling_damping,
            self.splash_velocity_gain,
            self.buoyancy_scale,
            self.box_linear_drag,
            self.box_quadratic_drag,
            self.box_angular_drag,
            self.box_floor_stiffness,
            self.box_floor_damping,
            self.box_floor_friction,
            self.box_wall_stiffness,
            self.box_wall_damping,
            self.box_max_linear_speed,
            self.box_max_angular_speed,
            self.box_max_torque,
            solver.particle_friction,
            solver.particle_collision_margin,
            solver.cohesion,
            solver.surface_tension,
            solver.vorticity_confinement,
            solver.solid_pressure,
            solver.buoyancy,
            solver.xsph_strength,
            solver.xsph_enabled,
            solver.free_surface_drag,
            solver.dissipation,
            solver.sleep_threshold,
            solver.shape_friction,
            solver.shape_collision_distance,
            solver.shape_collision_margin,
            solver.shape_restitution,
            solver.shape_adhesion,
            solver.max_acceleration,
            solver.render_smoothing,
            solver.render_anisotropy_scale,
            solver.render_anisotropy_min,
            solver.render_anisotropy_max,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                kernel=reset_box_water_heights,
                dim=len(self.box_body_ids),
                inputs=[
                    self.box_water_height_raw,
                    float(self.bounds_lower[2]),
                ],
                device=self.model.device,
            )
            wp.launch(
                kernel=measure_box_water_heights,
                dim=self.model.particle_count,
                inputs=[
                    self.state_0.particle_q,
                    self.model.particle_flags,
                    self.state_0.body_q,
                    self.box_body_ids_wp,
                    self.box_half_extents_wp,
                    self.particle_radius * 1.5,
                    self.particle_radius * 6.0,
                    self.box_water_height_raw,
                ],
                device=self.model.device,
            )
            wp.launch(
                kernel=smooth_box_water_heights,
                dim=len(self.box_body_ids),
                inputs=[
                    self.box_water_height_raw,
                    self.box_water_height,
                    0.04,
                    0.30,
                ],
                device=self.model.device,
            )
            wp.launch(
                kernel=apply_body_buoyancy_forces,
                dim=len(self.box_body_ids),
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.model.body_com,
                    self.model.body_mass,
                    self.box_body_ids_wp,
                    self.box_half_extents_wp,
                    self.box_water_height,
                    self.bounds_lower,
                    self.bounds_upper,
                    self.gravity,
                    self.rest_density,
                    self.buoyancy_scale,
                    self.box_linear_drag,
                    self.box_quadratic_drag,
                    self.box_angular_drag,
                    self.box_floor_stiffness,
                    self.box_floor_damping,
                    self.box_floor_friction,
                    self.box_wall_stiffness,
                    self.box_wall_damping,
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
                    self.model.particle_mass,
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
            wp.launch(
                kernel=clamp_body_water_velocities,
                dim=len(self.box_body_ids),
                inputs=[
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.box_body_ids_wp,
                    self.box_max_linear_speed,
                    self.box_max_angular_speed,
                    self.box_max_torque,
                ],
                device=self.model.device,
            )
            wp.launch(
                kernel=copy_body_kinematics,
                dim=self.model.body_count,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.sph_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.launch(
                kernel=add_body_wrenches,
                dim=self.model.body_count,
                inputs=[
                    self.state_0.body_f,
                    self.state_1.body_f,
                ],
                device=self.model.device,
            )
            wp.launch(
                kernel=clamp_body_water_velocities,
                dim=len(self.box_body_ids),
                inputs=[
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.box_body_ids_wp,
                    self.box_max_linear_speed,
                    self.box_max_angular_speed,
                    self.box_max_torque,
                ],
                device=self.model.device,
            )
            self.rigid_integrator.integrate_bodies(self.model, self.state_0, self.state_1, self.sim_dt, 0.03)
            wp.launch(
                kernel=clamp_body_water_velocities,
                dim=len(self.box_body_ids),
                inputs=[
                    self.state_1.body_qd,
                    self.state_1.body_f,
                    self.box_body_ids_wp,
                    self.box_max_linear_speed,
                    self.box_max_angular_speed,
                    0.0,
                ],
                device=self.model.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.capture_graph and (self.graph is None or self._graph_params() != self._captured_params):
            self.capture()
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
        if self.show_box_guides:
            self._log_box_guides()
        if self.show_bounds:
            self._log_bounds()
        else:
            self.viewer.log_lines("/fluid/bounds", None, None, None)
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
            hidden=False,
        )

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
        _changed, self.viewer.show_fluid_diffuse = ui.checkbox("Diffuse Spray", self.viewer.show_fluid_diffuse)
        _changed, self.viewer.show_particles = ui.checkbox("Raw Particles", self.viewer.show_particles)
        if self.viewer.show_particles:
            self.viewer.show_fluid = False

        _, self.sph_solver.particle_friction = ui.slider_float(
            "Particle Friction", self.sph_solver.particle_friction, 0.0, 8.0, "%.2f"
        )
        _, self.sph_solver.particle_collision_margin = ui.slider_float(
            "Particle Margin", self.sph_solver.particle_collision_margin, 0.0, 0.05, "%.4f"
        )
        _, self.sph_solver.cohesion = ui.slider_float("Cohesion", self.sph_solver.cohesion, 0.0, 2.0, "%.3f")
        _, self.sph_solver.surface_tension = ui.slider_float(
            "Surface Tension", self.sph_solver.surface_tension, 0.0, 0.00002, "%.6f"
        )
        _, self.sph_solver.vorticity_confinement = ui.slider_float(
            "Vorticity", self.sph_solver.vorticity_confinement, 0.0, 0.0004, "%.6f"
        )
        _, self.sph_solver.solid_pressure = ui.slider_float(
            "Solid Pressure", self.sph_solver.solid_pressure, 0.0, 1.0, "%.2f"
        )
        _, self.sph_solver.buoyancy = ui.slider_float("Buoyancy", self.sph_solver.buoyancy, -1.0, 2.0, "%.2f")
        _, self.sph_solver.xsph_strength = ui.slider_float("XSPH", self.sph_solver.xsph_strength, 0.0, 0.35, "%.3f")
        self.sph_solver.xsph_enabled = self.sph_solver.xsph_strength > 0.0
        _, self.sph_solver.free_surface_drag = ui.slider_float(
            "Free Surface Drag", self.sph_solver.free_surface_drag, 0.0, 2.0, "%.2f"
        )
        _, self.sph_solver.dissipation = ui.slider_float("Dissipation", self.sph_solver.dissipation, 0.0, 8.0, "%.2f")
        _, self.sph_solver.sleep_threshold = ui.slider_float(
            "Sleep Threshold", self.sph_solver.sleep_threshold, 0.0, 1.0, "%.2f"
        )
        _, self.sph_solver.shape_friction = ui.slider_float(
            "Shape Friction", self.sph_solver.shape_friction, 0.0, 4.0, "%.2f"
        )
        if self.sph_solver.shape_collision_distance is None:
            self.sph_solver.shape_collision_distance = self.particle_radius
        _, self.sph_solver.shape_collision_distance = ui.slider_float(
            "Shape Distance", self.sph_solver.shape_collision_distance, 0.0, 0.10, "%.4f"
        )
        _, self.sph_solver.shape_collision_margin = ui.slider_float(
            "Shape Margin", self.sph_solver.shape_collision_margin, 0.0, 0.05, "%.4f"
        )
        _, self.sph_solver.shape_restitution = ui.slider_float(
            "Shape Restitution", self.sph_solver.shape_restitution or 0.0, 0.0, 1.0, "%.2f"
        )
        _, self.sph_solver.shape_adhesion = ui.slider_float(
            "Shape Adhesion", self.sph_solver.shape_adhesion, 0.0, 4.0, "%.2f"
        )
        _, self.sph_solver.max_acceleration = ui.slider_float(
            "Max Acceleration", self.sph_solver.max_acceleration, 1.0, 300.0, "%.1f"
        )
        _, self.sph_solver.render_smoothing = ui.slider_float(
            "Render Smoothing", self.sph_solver.render_smoothing, 0.0, 1.0, "%.2f"
        )
        _, self.sph_solver.render_anisotropy_scale = ui.slider_float(
            "Anisotropy Scale", self.sph_solver.render_anisotropy_scale, 0.0, 3.0, "%.2f"
        )
        _, self.sph_solver.render_anisotropy_min = ui.slider_float(
            "Anisotropy Min", self.sph_solver.render_anisotropy_min, 0.01, 1.0, "%.2f"
        )
        _, self.sph_solver.render_anisotropy_max = ui.slider_float(
            "Anisotropy Max", self.sph_solver.render_anisotropy_max, 1.0, 4.0, "%.2f"
        )
        self.sph_solver.render_anisotropy_max = max(
            self.sph_solver.render_anisotropy_max, self.sph_solver.render_anisotropy_min
        )

        ui.separator()
        ui.text("Water Shader")
        changed, rgb = ui.slider_float3(
            "Water Color", [self.fluid_color[0], self.fluid_color[1], self.fluid_color[2]], 0.0, 1.0, "%.2f"
        )
        if changed:
            self.fluid_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), self.fluid_color[3])
        _, opacity = ui.slider_float("Opacity", self.fluid_color[3], 0.0, 1.0, "%.2f")
        self.fluid_color = (self.fluid_color[0], self.fluid_color[1], self.fluid_color[2], float(opacity))
        _, self.fluid_ior = ui.slider_float("IOR", self.fluid_ior, 0.5, 2.5, "%.2f")
        _, self.fluid_blur_radius = ui.slider_float("Smoothing Radius", self.fluid_blur_radius, 0.0, 0.25, "%.3f")
        _, self.foam_radius = ui.slider_float("Foam Radius", self.foam_radius, 0.002, 0.08, "%.3f")
        _, self.foam_motion_blur = ui.slider_float("Foam Motion Blur", self.foam_motion_blur, 0.0, 4.0, "%.2f")

        ui.separator()
        ui.text("Interaction")
        _, self.coupling_stiffness = ui.slider_float(
            "Water-Box Stiffness", self.coupling_stiffness, 0.0, 2200.0, "%.1f"
        )
        _, self.coupling_damping = ui.slider_float("Water-Box Damping", self.coupling_damping, 0.0, 80.0, "%.1f")
        _, self.splash_velocity_gain = ui.slider_float("Splash Gain", self.splash_velocity_gain, 0.0, 4.0, "%.2f")
        _, self.buoyancy_scale = ui.slider_float("Buoyancy Scale", self.buoyancy_scale, 0.0, 2.0, "%.2f")
        _, self.box_linear_drag = ui.slider_float("Box Linear Drag", self.box_linear_drag, 0.0, 10.0, "%.1f")
        _, self.box_quadratic_drag = ui.slider_float("Box Quadratic Drag", self.box_quadratic_drag, 0.0, 25.0, "%.1f")
        _, self.box_angular_drag = ui.slider_float("Box Angular Drag", self.box_angular_drag, 0.0, 8.0, "%.1f")
        _, self.box_max_linear_speed = ui.slider_float("Box Max Speed", self.box_max_linear_speed, 0.5, 8.0, "%.1f")
        _, self.box_max_angular_speed = ui.slider_float(
            "Box Max Angular Speed", self.box_max_angular_speed, 1.0, 30.0, "%.1f"
        )
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

        box_q = body_q[self.box_body_ids]
        margin = 0.35
        if np.any(box_q[:, 0] < self.bounds_lower[0] - margin) or np.any(box_q[:, 0] > self.bounds_upper[0] + margin):
            raise ValueError("Boxes escaped the tank bounds along x")
        if np.any(box_q[:, 1] < self.bounds_lower[1] - margin) or np.any(box_q[:, 1] > self.bounds_upper[1] + margin):
            raise ValueError("Boxes escaped the tank bounds along y")
        if np.any(box_q[:, 2] < self.bounds_lower[2] - margin) or np.any(box_q[:, 2] > self.bounds_upper[2] + margin):
            raise ValueError("Boxes escaped the tank bounds along z")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=3)
        parser.add_argument("--render-mode", choices=["fluid", "particles"], default="fluid")
        parser.add_argument(
            "--capture-graph",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the SPH substeps in a CUDA graph (re-captured automatically when sliders change).",
        )
        parser.add_argument("--show-bounds", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--show-diffuse", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--show-box-guides",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Draw colored guide outlines around the pickable rigid bodies.",
        )

        parser.add_argument("--dim-x", type=int, default=80)
        parser.add_argument("--dim-y", type=int, default=50)
        parser.add_argument("--dim-z", type=int, default=14)
        parser.add_argument("--spacing", type=float, default=0.034)
        parser.add_argument("--radius", type=float, default=0.0255)
        parser.add_argument("--jitter", type=float, default=0.0015)
        parser.add_argument("--emit-lower", type=float, nargs=3, default=(-1.24, -0.78, 0.06))
        parser.add_argument("--initial-velocity", type=float, nargs=3, default=(0.0, 0.0, 0.0))

        parser.add_argument("--smoothing-length", type=float, default=0.0731)
        parser.add_argument("--rest-density", type=float, default=460.0)
        parser.add_argument("--gas-constant", type=float, default=44.0)
        parser.add_argument("--viscosity", type=float, default=0.010)
        parser.add_argument("--particle-friction", type=float, default=0.10)
        parser.add_argument("--particle-collision-margin", type=float, default=0.0015)
        parser.add_argument("--cohesion", type=float, default=0.035)
        parser.add_argument("--surface-tension", type=float, default=0.0000006)
        parser.add_argument("--vorticity-confinement", type=float, default=0.0)
        parser.add_argument("--solid-pressure", type=float, default=0.08)
        parser.add_argument("--buoyancy", type=float, default=1.0)
        parser.add_argument("--xsph-strength", type=float, default=0.06)
        parser.add_argument("--free-surface-drag", type=float, default=0.12)
        parser.add_argument("--dissipation", type=float, default=0.25)
        parser.add_argument("--velocity-damping", type=float, default=0.02)
        parser.add_argument("--sleep-threshold", type=float, default=0.01)
        parser.add_argument("--boundary-damping", type=float, default=0.10)
        parser.add_argument("--shape-collision-distance", type=float, default=0.0255)
        parser.add_argument("--shape-collision-margin", type=float, default=0.0015)
        parser.add_argument("--shape-restitution", type=float, default=0.0)
        parser.add_argument("--shape-friction", type=float, default=0.25)
        parser.add_argument("--shape-adhesion", type=float, default=0.18)
        parser.add_argument("--max-velocity", type=float, default=6.0)
        parser.add_argument("--max-acceleration", type=float, default=70.0)
        parser.add_argument(
            "--sph-body-feedback",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Apply full SPH shape-contact reactions to rigid bodies (in addition to the example's buoyancy model).",
        )
        parser.add_argument("--pbf-iterations", type=int, default=3)
        parser.add_argument("--pbf-relaxation", type=float, default=0.60)
        parser.add_argument("--pbf-artificial-pressure", type=float, default=0.002)
        parser.add_argument("--fluid-diffuse-max-particles", type=int, default=16000)
        parser.add_argument("--fluid-diffuse-threshold", type=float, default=0.55)
        parser.add_argument("--fluid-diffuse-lifetime", type=float, default=2.8)
        parser.add_argument("--fluid-diffuse-drag", type=float, default=0.92)
        parser.add_argument("--fluid-diffuse-buoyancy", type=float, default=0.9)
        parser.add_argument("--fluid-diffuse-ballistic", type=int, default=9)
        parser.add_argument("--fluid-diffuse-spawn-probability", type=float, default=0.45)
        parser.add_argument("--fluid-render-smoothing", type=float, default=0.8)
        parser.add_argument("--fluid-render-anisotropy-scale", type=float, default=0.82)
        parser.add_argument("--fluid-render-anisotropy-min", type=float, default=0.1)
        parser.add_argument("--fluid-render-anisotropy-max", type=float, default=2.0)
        parser.add_argument("--fluid-render-update-interval", type=int, default=2)
        parser.add_argument("--fluid-diffuse-update-interval", type=int, default=2)
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--bounds-lower", type=float, nargs=3, default=(-1.35, -0.92, 0.0))
        parser.add_argument("--bounds-upper", type=float, nargs=3, default=(1.55, 1.00, 1.72))
        parser.add_argument("--ground-color", type=float, nargs=3, default=(0.54, 0.55, 0.53))
        parser.add_argument("--water-level", type=float, default=None)

        parser.add_argument("--box-count", type=int, default=5)
        parser.add_argument("--box-half-extent", type=float, default=0.115)
        parser.add_argument("--box-height", type=float, default=0.70)
        parser.add_argument(
            "--box-density-fractions",
            type=float,
            nargs="+",
            default=(0.30, 0.55, 0.85, 0.42, 1.60),
            help="Box densities as fractions of the fluid rest density; values above 1 sink.",
        )
        parser.add_argument("--fluid-carve-clearance", type=float, default=0.055)
        parser.add_argument("--pick-stiffness", type=float, default=72.0)
        parser.add_argument("--pick-damping", type=float, default=22.0)
        parser.add_argument("--coupling-stiffness", type=float, default=90.0)
        parser.add_argument("--coupling-damping", type=float, default=30.0)
        parser.add_argument("--splash-velocity-gain", type=float, default=0.15)
        parser.add_argument("--buoyancy-scale", type=float, default=1.0)
        parser.add_argument("--box-linear-drag", type=float, default=2.0)
        parser.add_argument("--box-quadratic-drag", type=float, default=8.0)
        parser.add_argument("--box-angular-drag", type=float, default=3.0)
        parser.add_argument("--box-floor-stiffness", type=float, default=600.0)
        parser.add_argument("--box-floor-damping", type=float, default=20.0)
        parser.add_argument("--box-floor-friction", type=float, default=8.0)
        parser.add_argument("--box-wall-stiffness", type=float, default=60.0)
        parser.add_argument("--box-wall-damping", type=float, default=5.0)
        parser.add_argument("--box-max-linear-speed", type=float, default=4.0)
        parser.add_argument("--box-max-angular-speed", type=float, default=14.0)
        parser.add_argument("--box-max-torque", type=float, default=0.0)

        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))
        parser.add_argument("--fluid-ior", type=float, default=1.0)
        parser.add_argument("--fluid-radius-scale", type=float, default=1.8)
        parser.add_argument("--fluid-blur-radius", type=float, default=0.09)
        parser.add_argument("--foam-color", type=float, nargs=4, default=(0.9, 0.95, 1.0, 0.8))
        parser.add_argument("--foam-radius", type=float, default=0.018)
        parser.add_argument("--foam-motion-blur", type=float, default=1.0)

        parser.add_argument("--environment-intensity", type=float, default=3.15)
        parser.add_argument("--exposure", type=float, default=1.08)
        parser.add_argument("--diffuse-scale", type=float, default=1.05)
        parser.add_argument("--specular-scale", type=float, default=4.00)
        parser.add_argument("--sun-direction", type=float, nargs=3, default=(0.78, -0.56, 0.20))
        parser.add_argument("--angular-damping", type=float, default=0.04)
        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.45, -1.55, 1.30))
        parser.add_argument("--camera-pitch", type=float, default=-27.0)
        parser.add_argument("--camera-yaw", type=float, default=132.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
