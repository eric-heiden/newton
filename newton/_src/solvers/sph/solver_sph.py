# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase
from .kernels import (
    advance_sph_diffuse_seed,
    apply_pbf_deltas,
    collide_sph_diffuse_particles_with_shapes,
    collide_sph_particles_with_shapes,
    compute_pbf_lambdas,
    compute_sph_density_pressure,
    compute_sph_render_particles,
    compute_sph_vorticity,
    copy_sph_velocities,
    finalize_pbf_velocities,
    integrate_sph_particles,
    sleep_sph_particles,
    smooth_sph_velocities,
    solve_pbf_deltas,
    spawn_sph_diffuse_particles,
    update_sph_diffuse_particles,
)


def _vec3(value: Sequence[float] | wp.vec3 | None, default: tuple[float, float, float]) -> wp.vec3:
    if value is None:
        return wp.vec3(*default)
    return wp.vec3(float(value[0]), float(value[1]), float(value[2]))


class SolverSPH(SolverBase):
    """Weakly-compressible SPH prototype for Newton particle fluids.

    The solver advances :attr:`newton.State.particle_q` and
    :attr:`newton.State.particle_qd` using Warp kernels and a
    :class:`warp.HashGrid` neighbor search. It is intentionally small: it
    provides density, pressure, viscosity, gravity, external particle forces,
    and optional axis-aligned world bounds. It is meant as a first fluid solver
    and a reusable baseline for solver developers, not as a production-grade
    incompressible fluid method.

    Args:
        model: Model containing particles to simulate.
        smoothing_length: SPH kernel radius [m]. If ``None``, twice the
            model's maximum particle radius is used.
        rest_density: Fluid rest density [kg/m^3].
        gas_constant: Pressure stiffness for weak compressibility.
        viscosity: Kinematic viscosity coefficient.
        particle_friction: Flex-style particle contact friction [1/s] that
            damps tangential relative velocity between nearby particles.
        particle_collision_margin: Additional neighbor search distance [m]
            used by contact-style particle friction and dissipation without
            expanding the density/pressure kernel support.
        cohesion: Neighbor attraction strength for Flex-style fluid strands.
        surface_tension: Free-surface contraction strength.
        vorticity_confinement: Rotational energy injection strength for
            counteracting numerical damping.
        solid_pressure: Pressure scale applied near solver bounds to mimic
            missing solid-neighbor support.
        buoyancy: Gravity scale for fluid particles.
        xsph_strength: Post-projection velocity smoothing strength in
            ``[0, 1]``. Values greater than zero reduce particle velocity
            noise while preserving the local flow average.
        free_surface_drag: Drag rate [1/s] applied to low-density boundary
            particles.
        dissipation: Flex-style contact dissipation rate [1/s], scaled by
            local particle neighbor count to calm dense fluid without damping
            isolated spray as strongly.
        velocity_damping: Linear velocity damping coefficient [1/s].
        sleep_threshold: Speed below which active particles are treated as
            settled and have their velocity zeroed [m/s]. Set to ``0`` to
            disable sleeping.
        bounds_lower: Optional lower world bounds [m].
        bounds_upper: Optional upper world bounds [m].
        boundary_damping: Normal velocity restitution when clamped to bounds.
        max_velocity: Velocity clamp [m/s].
        max_acceleration: Acceleration clamp [m/s²]. If ``None``, a large
            value is used, matching Flex's opt-out behavior for avoiding
            sudden pressure/contact pops only when requested.
        pbf_iterations: Number of position-based density projection iterations
            applied after force integration. Values greater than zero make the
            fluid less compressible, closer to Flex/PBF behavior.
        pbf_relaxation: Per-iteration multiplier for projected position
            corrections.
        pbf_relaxation_epsilon: Constraint denominator regularizer.
        pbf_artificial_pressure: Tensile-instability correction strength.
        pbf_artificial_radius: Kernel distance [m] used as the artificial
            pressure reference. If ``None``, ``0.3 * smoothing_length`` is used.
        pbf_artificial_power: Exponent for the artificial pressure term.
        pbf_max_delta: Maximum position correction per iteration [m]. If
            ``None``, ``0.5 * smoothing_length`` is used.
        max_diffuse_particles: Maximum secondary foam/spray particles. Set to
            ``0`` to disable the diffuse particle layer.
        diffuse_threshold: Trapped-air/wave-crest potential needed to emit a
            diffuse particle. The potential favors surface particles whose
            velocity points out of the fluid (breaking crests, splashes).
        diffuse_lifetime: Diffuse particle lifetime [s].
        diffuse_drag: Blend rate toward neighboring fluid velocity [1/s].
        diffuse_buoyancy: Fraction of gravity canceled while a diffuse
            particle is inside the fluid neighborhood.
        diffuse_ballistic: Neighbor count below which a diffuse particle falls
            ballistically instead of advecting with the fluid.
        diffuse_spawn_probability: Per-step spawn probability multiplier once
            the diffuse potential exceeds ``diffuse_threshold``.
        diffuse_jitter: Spawn offset radius [m]. If ``None``, a fraction of the
            smoothing length is used.
        diffuse_surface_density_ratio: Density threshold used to classify
            surface particles as diffuse emitters.
        render_smoothing: Strength of Flex-style smoothed render particle
            positions in ``[0, 1]``. Set to ``0`` to keep raw particle centers.
        render_anisotropy_scale: Stretch multiplier for render ellipsoids.
            Set to ``0`` to output spherical render particles.
        render_anisotropy_min: Minimum ellipsoid axis scale as a fraction of
            particle radius.
        render_anisotropy_max: Maximum ellipsoid axis scale as a fraction of
            particle radius.
        render_update_interval: Number of solver steps between render buffer
            updates. Values greater than one reduce visual-buffer cost for
            large fluids while leaving simulation particles updated every step.
        diffuse_update_interval: Number of solver steps between diffuse
            foam/spray updates. Values greater than one reduce secondary
            particle cost for large fluids.
        shape_collision: Whether SPH and diffuse particles should collide
            with primitive, mesh, and heightfield model shapes marked with
            ``ShapeFlags.COLLIDE_PARTICLES``.
        shape_collision_distance: Distance particles maintain from shape
            surfaces [m]. If ``None``, each particle's radius is used.
        shape_collision_margin: Additional contact search distance [m] used
            to find nearby shapes without increasing maintained clearance.
        shape_restitution: Coefficient of restitution for particle-shape
            collisions. If ``None``, ``boundary_damping`` is used for
            backwards compatibility.
        shape_friction: Flex-style contact friction rate [1/s] that damps
            fluid velocity tangent to shape surfaces.
        shape_adhesion: Flex-style contact adhesion rate [1/s] that damps
            fluid velocity peeling away from shape surfaces.
        shape_collision_body_feedback: Whether main SPH particle contacts
            with dynamic shape bodies should accumulate equal-and-opposite
            wrenches into ``state_out.body_f``. Diffuse particles remain
            visual-only and do not contribute body feedback.
    """

    def __init__(
        self,
        model: Model,
        smoothing_length: float | None = None,
        rest_density: float = 1000.0,
        gas_constant: float = 2000.0,
        viscosity: float = 0.05,
        particle_friction: float = 0.0,
        particle_collision_margin: float = 0.0,
        cohesion: float = 0.0,
        surface_tension: float = 0.0,
        vorticity_confinement: float = 0.0,
        solid_pressure: float = 0.0,
        buoyancy: float = 1.0,
        xsph_strength: float = 0.0,
        free_surface_drag: float = 0.0,
        dissipation: float = 0.0,
        velocity_damping: float = 0.0,
        sleep_threshold: float = 0.0,
        bounds_lower: Sequence[float] | wp.vec3 | None = None,
        bounds_upper: Sequence[float] | wp.vec3 | None = None,
        boundary_damping: float = 0.5,
        max_velocity: float | None = None,
        max_acceleration: float | None = None,
        pbf_iterations: int = 0,
        pbf_relaxation: float = 1.0,
        pbf_relaxation_epsilon: float = 1.0e-5,
        pbf_artificial_pressure: float = 0.01,
        pbf_artificial_radius: float | None = None,
        pbf_artificial_power: float = 4.0,
        pbf_max_delta: float | None = None,
        max_diffuse_particles: int = 0,
        diffuse_threshold: float = 2.0,
        diffuse_lifetime: float = 1.5,
        diffuse_drag: float = 0.8,
        diffuse_buoyancy: float = 0.25,
        diffuse_ballistic: int = 8,
        diffuse_spawn_probability: float = 0.35,
        diffuse_jitter: float | None = None,
        diffuse_surface_density_ratio: float = 0.92,
        render_smoothing: float = 0.45,
        render_anisotropy_scale: float = 0.82,
        render_anisotropy_min: float = 0.1,
        render_anisotropy_max: float = 2.0,
        render_update_interval: int = 1,
        diffuse_update_interval: int = 1,
        shape_collision: bool = True,
        shape_collision_distance: float | None = None,
        shape_collision_margin: float = 0.0,
        shape_restitution: float | None = None,
        shape_friction: float = 0.0,
        shape_adhesion: float = 0.0,
        shape_collision_body_feedback: bool = True,
    ):
        super().__init__(model=model)
        default_h = 2.0 * model.particle_max_radius if model.particle_max_radius > 0.0 else 0.1
        self.smoothing_length = float(default_h if smoothing_length is None else smoothing_length)
        self.rest_density = float(rest_density)
        self.gas_constant = float(gas_constant)
        self.viscosity = float(viscosity)
        self.particle_friction = float(max(particle_friction, 0.0))
        self.particle_collision_margin = float(max(particle_collision_margin, 0.0))
        self.cohesion = float(max(cohesion, 0.0))
        self.surface_tension = float(max(surface_tension, 0.0))
        self.vorticity_confinement = float(max(vorticity_confinement, 0.0))
        self.solid_pressure = float(max(solid_pressure, 0.0))
        self.buoyancy = float(buoyancy)
        self.xsph_strength = float(min(max(xsph_strength, 0.0), 1.0))
        self.xsph_enabled = self.xsph_strength > 0.0
        self.free_surface_drag = float(max(free_surface_drag, 0.0))
        self.dissipation = float(max(dissipation, 0.0))
        self.velocity_damping = float(velocity_damping)
        self.sleep_threshold = float(max(sleep_threshold, 0.0))
        self.bounds_lower = _vec3(bounds_lower, (-1.0e8, -1.0e8, -1.0e8))
        self.bounds_upper = _vec3(bounds_upper, (1.0e8, 1.0e8, 1.0e8))
        self.boundary_damping = float(boundary_damping)
        self.max_velocity = float(model.particle_max_velocity if max_velocity is None else max_velocity)
        self.max_acceleration = float(1.0e8 if max_acceleration is None else max(max_acceleration, 0.0))
        self.pbf_iterations = max(int(pbf_iterations), 0)
        self.pbf_enabled = self.pbf_iterations > 0
        self.pbf_relaxation = float(max(pbf_relaxation, 0.0))
        self.pbf_relaxation_epsilon = float(max(pbf_relaxation_epsilon, 0.0))
        self.pbf_artificial_pressure = float(max(pbf_artificial_pressure, 0.0))
        default_artificial_radius = 0.3 * self.smoothing_length
        self.pbf_artificial_radius = float(
            default_artificial_radius if pbf_artificial_radius is None else max(pbf_artificial_radius, 0.0)
        )
        self.pbf_artificial_power = float(max(pbf_artificial_power, 1.0))
        default_max_delta = 0.5 * self.smoothing_length
        self.pbf_max_delta = float(default_max_delta if pbf_max_delta is None else max(pbf_max_delta, 0.0))
        self.max_diffuse_particles = max(int(max_diffuse_particles), 0)
        self.diffuse_threshold = float(max(diffuse_threshold, 0.0))
        self.diffuse_lifetime = float(max(diffuse_lifetime, 1.0e-6))
        self.diffuse_drag = float(max(diffuse_drag, 0.0))
        self.diffuse_buoyancy = float(max(diffuse_buoyancy, 0.0))
        self.diffuse_ballistic = max(int(diffuse_ballistic), 1)
        self.diffuse_spawn_probability = float(max(diffuse_spawn_probability, 0.0))
        default_jitter = 0.12 * self.smoothing_length
        self.diffuse_jitter = float(default_jitter if diffuse_jitter is None else max(diffuse_jitter, 0.0))
        self.diffuse_surface_density_ratio = float(max(diffuse_surface_density_ratio, 0.0))
        self.diffuse_enabled = self.max_diffuse_particles > 0
        self.render_smoothing = float(min(max(render_smoothing, 0.0), 1.0))
        self.render_anisotropy_scale = float(max(render_anisotropy_scale, 0.0))
        self.render_anisotropy_min = float(max(render_anisotropy_min, 0.01))
        self.render_anisotropy_max = float(max(render_anisotropy_max, self.render_anisotropy_min))
        self.render_update_interval = max(int(render_update_interval), 1)
        self.diffuse_update_interval = max(int(diffuse_update_interval), 1)
        self.render_enabled = self.render_smoothing > 0.0 or self.render_anisotropy_scale > 0.0
        self.shape_collision = bool(shape_collision)
        self.shape_collision_distance = (
            None if shape_collision_distance is None else float(max(shape_collision_distance, 0.0))
        )
        self.shape_collision_margin = float(max(shape_collision_margin, 0.0))
        self.shape_restitution = None if shape_restitution is None else float(min(max(shape_restitution, 0.0), 1.0))
        self.shape_friction = float(max(shape_friction, 0.0))
        self.shape_adhesion = float(max(shape_adhesion, 0.0))
        self.shape_collision_body_feedback = bool(shape_collision_body_feedback)
        self.render_buffers_valid = False
        self._step_index = 0
        self._capacity = 0
        self._empty_body_q: wp.array[wp.transform] | None = None
        self._empty_body_qd: wp.array[wp.spatial_vector] | None = None
        self._empty_body_f: wp.array[wp.spatial_vector] | None = None
        self._empty_body_com: wp.array[wp.vec3] | None = None
        self._empty_body_flags: wp.array[wp.int32] | None = None
        self.particle_density: wp.array[wp.float32] | None = None
        self.particle_pressure: wp.array[wp.float32] | None = None
        self.particle_vorticity: wp.array[wp.vec3] | None = None
        self.particle_velocity_smooth: wp.array[wp.vec3] | None = None
        self.pbf_lambdas: wp.array[wp.float32] | None = None
        self.pbf_deltas: wp.array[wp.vec3] | None = None
        self.render_positions: wp.array[wp.vec3] | None = None
        self.render_anisotropy: wp.array[wp.vec4] | None = None
        self.render_anisotropy_secondary: wp.array[wp.vec4] | None = None
        self.render_anisotropy_tertiary: wp.array[wp.vec4] | None = None
        self.diffuse_positions: wp.array[wp.vec4] | None = None
        self.diffuse_velocities: wp.array[wp.vec4] | None = None
        self.diffuse_worlds: wp.array[wp.int32] | None = None
        self.diffuse_slot_states: wp.array[wp.int32] | None = None
        self.diffuse_spawn_counter: wp.array[wp.int32] | None = None
        self.diffuse_frame_seed: wp.array[wp.int32] | None = None
        self._ensure_particle_storage()
        self._ensure_diffuse_storage()

    def _particle_neighbor_search_radius(self) -> float:
        return max(self.smoothing_length, self.smoothing_length + self.particle_collision_margin)

    def _shape_collision_distance_value(self) -> float:
        if self.shape_collision_distance is None:
            return -1.0
        return self.shape_collision_distance

    def _shape_restitution_value(self) -> float:
        if self.shape_restitution is None:
            return self.boundary_damping
        return self.shape_restitution

    def _ensure_particle_storage(self) -> None:
        model = self.model
        n = model.particle_count
        if n == self._capacity:
            return
        self._capacity = n
        self.particle_density = wp.empty(n, dtype=wp.float32, device=model.device)
        self.particle_pressure = wp.empty(n, dtype=wp.float32, device=model.device)
        self.particle_vorticity = wp.empty(n, dtype=wp.vec3, device=model.device)
        self.particle_velocity_smooth = wp.empty(n, dtype=wp.vec3, device=model.device)
        self.pbf_lambdas = wp.empty(n, dtype=wp.float32, device=model.device) if self.pbf_enabled else None
        self.pbf_deltas = wp.empty(n, dtype=wp.vec3, device=model.device) if self.pbf_enabled else None
        self.render_positions = wp.empty(n, dtype=wp.vec3, device=model.device) if self.render_enabled else None
        self.render_anisotropy = wp.empty(n, dtype=wp.vec4, device=model.device) if self.render_enabled else None
        self.render_anisotropy_secondary = (
            wp.empty(n, dtype=wp.vec4, device=model.device) if self.render_enabled else None
        )
        self.render_anisotropy_tertiary = (
            wp.empty(n, dtype=wp.vec4, device=model.device) if self.render_enabled else None
        )
        self.render_buffers_valid = False
        if n:
            with wp.ScopedDevice(model.device):
                if model.particle_grid is None:
                    model.particle_grid = wp.HashGrid(128, 128, 128)
                model.particle_grid.reserve(n)

    def _ensure_diffuse_storage(self) -> None:
        if not self.diffuse_enabled:
            self.diffuse_positions = None
            self.diffuse_velocities = None
            self.diffuse_worlds = None
            self.diffuse_slot_states = None
            self.diffuse_spawn_counter = None
            self.diffuse_frame_seed = None
            return

        model = self.model
        if (
            self.diffuse_positions is not None
            and self.diffuse_worlds is not None
            and self.diffuse_slot_states is not None
            and len(self.diffuse_positions) == self.max_diffuse_particles
        ):
            return

        self.diffuse_positions = wp.zeros(self.max_diffuse_particles, dtype=wp.vec4, device=model.device)
        self.diffuse_velocities = wp.zeros(self.max_diffuse_particles, dtype=wp.vec4, device=model.device)
        self.diffuse_worlds = wp.zeros(self.max_diffuse_particles, dtype=wp.int32, device=model.device)
        self.diffuse_slot_states = wp.zeros(self.max_diffuse_particles, dtype=wp.int32, device=model.device)
        self.diffuse_spawn_counter = wp.zeros(1, dtype=wp.int32, device=model.device)
        self.diffuse_frame_seed = wp.zeros(1, dtype=wp.int32, device=model.device)

    def clear_diffuse_particles(self) -> None:
        """Deactivate all secondary foam/spray particles."""
        self._ensure_diffuse_storage()
        if self.diffuse_positions is not None:
            self.diffuse_positions.zero_()
        if self.diffuse_velocities is not None:
            self.diffuse_velocities.zero_()
        if self.diffuse_worlds is not None:
            self.diffuse_worlds.zero_()
        if self.diffuse_slot_states is not None:
            self.diffuse_slot_states.zero_()
        if self.diffuse_spawn_counter is not None:
            self.diffuse_spawn_counter.zero_()

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & ModelFlags.PARTICLE_PROPERTIES:
            self._ensure_particle_storage()
            self._ensure_diffuse_storage()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance particle fluid state by one time step.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Unused; accepted for solver API compatibility.
            contacts: Unused; accepted for solver API compatibility.
            dt: Time step [s].
        """
        model = self.model
        if model.particle_count == 0:
            return

        self._ensure_particle_storage()
        assert model.particle_grid is not None
        assert self.particle_density is not None
        assert self.particle_pressure is not None
        assert self.particle_vorticity is not None

        with wp.ScopedTimer("simulate", False):
            particle_search_radius = self._particle_neighbor_search_radius()
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(state_in.particle_q, radius=particle_search_radius)

            wp.launch(
                kernel=compute_sph_density_pressure,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_in.particle_q,
                    model.particle_mass,
                    model.particle_flags,
                    self.smoothing_length,
                    self.rest_density,
                    self.gas_constant,
                    self.particle_density,
                    self.particle_pressure,
                ],
                device=model.device,
            )

            wp.launch(
                kernel=compute_sph_vorticity,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_mass,
                    model.particle_flags,
                    self.particle_density,
                    self.smoothing_length,
                    self.particle_vorticity,
                ],
                device=model.device,
            )

            wp.launch(
                kernel=integrate_sph_particles,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_mass,
                    model.particle_inv_mass,
                    model.particle_radius,
                    model.particle_flags,
                    model.particle_world,
                    model.gravity,
                    self.buoyancy,
                    self.particle_density,
                    self.particle_pressure,
                    self.particle_vorticity,
                    self.smoothing_length,
                    self.particle_collision_margin,
                    self.rest_density,
                    self.viscosity,
                    self.particle_friction,
                    self.cohesion,
                    self.surface_tension,
                    self.vorticity_confinement,
                    self.solid_pressure,
                    self.free_surface_drag,
                    self.dissipation,
                    self.velocity_damping,
                    self.bounds_lower,
                    self.bounds_upper,
                    self.boundary_damping,
                    self.max_acceleration,
                    self.max_velocity,
                    self.sleep_threshold,
                    dt,
                    state_out.particle_q,
                    state_out.particle_qd,
                ],
                device=model.device,
            )

            if self.pbf_enabled:
                self._project_density(state_in, state_out, dt)

            self.xsph_enabled = self.xsph_strength > 0.0
            if self.xsph_enabled:
                self._smooth_velocities(state_out, dt)

            if self.shape_collision and model.shape_count > 0:
                self._collide_with_shapes(state_out, dt)

            if self.sleep_threshold > 0.0:
                self._sleep_particles(state_out)

            render_updated = False
            if self.render_enabled and (
                not self.render_buffers_valid or self._step_index % self.render_update_interval == 0
            ):
                self._update_render_particles(state_out)
                render_updated = True

            if self.diffuse_enabled and self._step_index % self.diffuse_update_interval == 0:
                self._step_diffuse_particles(state_out, dt * float(self.diffuse_update_interval), not render_updated)

        self._step_index += 1

    def _empty_body_arrays(
        self,
    ) -> tuple[
        wp.array[wp.transform],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
        wp.array[wp.vec3],
        wp.array[wp.int32],
    ]:
        model = self.model
        if self._empty_body_q is None:
            self._empty_body_q = wp.empty(0, dtype=wp.transform, device=model.device)
            self._empty_body_qd = wp.empty(0, dtype=wp.spatial_vector, device=model.device)
            self._empty_body_f = wp.empty(0, dtype=wp.spatial_vector, device=model.device)
            self._empty_body_com = wp.empty(0, dtype=wp.vec3, device=model.device)
            self._empty_body_flags = wp.empty(0, dtype=wp.int32, device=model.device)
        assert self._empty_body_q is not None
        assert self._empty_body_qd is not None
        assert self._empty_body_f is not None
        assert self._empty_body_com is not None
        assert self._empty_body_flags is not None
        return self._empty_body_q, self._empty_body_qd, self._empty_body_f, self._empty_body_com, self._empty_body_flags

    def _body_arrays(
        self, state: State
    ) -> tuple[
        wp.array[wp.transform],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
        wp.array[wp.vec3],
        wp.array[wp.int32],
    ]:
        model = self.model
        if model.body_count == 0:
            return self._empty_body_arrays()

        body_q = state.body_q if state.body_q is not None else model.body_q
        body_qd = state.body_qd if state.body_qd is not None else model.body_qd
        body_f = state.body_f
        body_com = model.body_com
        body_flags = model.body_flags
        if body_q is None or body_qd is None or body_com is None:
            return self._empty_body_arrays()
        if body_f is None:
            if self._empty_body_f is None:
                self._empty_body_f = wp.empty(0, dtype=wp.spatial_vector, device=model.device)
            body_f = self._empty_body_f
        if body_flags is None:
            if self._empty_body_flags is None:
                self._empty_body_flags = wp.empty(0, dtype=wp.int32, device=model.device)
            body_flags = self._empty_body_flags
        return body_q, body_qd, body_f, body_com, body_flags

    def _has_shape_collision_arrays(self) -> bool:
        model = self.model
        return (
            model.shape_transform is not None
            and model.shape_body is not None
            and model.shape_type is not None
            and model.shape_scale is not None
            and model.shape_source_ptr is not None
            and model.shape_flags is not None
            and model.shape_margin is not None
            and model.shape_world is not None
            and model.shape_heightfield_index is not None
            and model.heightfield_data is not None
            and model.heightfield_elevations is not None
        )

    def _collide_with_shapes(self, fluid_state: State, dt: float) -> None:
        model = self.model
        if not self._has_shape_collision_arrays():
            return

        assert model.shape_transform is not None
        assert model.shape_body is not None
        assert model.shape_type is not None
        assert model.shape_scale is not None
        assert model.shape_source_ptr is not None
        assert model.shape_flags is not None
        assert model.shape_margin is not None
        assert model.shape_world is not None
        assert model.shape_heightfield_index is not None
        assert model.heightfield_data is not None
        assert model.heightfield_elevations is not None
        body_q, body_qd, body_f, body_com, body_flags = self._body_arrays(fluid_state)
        body_feedback = int(
            self.shape_collision_body_feedback
            and len(body_f) == model.body_count
            and len(body_flags) == model.body_count
            and dt > 0.0
        )
        wp.launch(
            kernel=collide_sph_particles_with_shapes,
            dim=model.particle_count,
            inputs=[
                fluid_state.particle_q,
                fluid_state.particle_qd,
                model.particle_radius,
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                model.particle_world,
                body_q,
                body_qd,
                body_f,
                body_com,
                body_flags,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_source_ptr,
                model.shape_flags,
                model.shape_margin,
                model.shape_world,
                model.shape_heightfield_index,
                model.heightfield_data,
                model.heightfield_elevations,
                model.shape_count,
                self.boundary_damping,
                self._shape_collision_distance_value(),
                self.shape_collision_margin,
                self._shape_restitution_value(),
                self.shape_friction,
                self.shape_adhesion,
                dt,
                body_feedback,
            ],
            device=model.device,
        )

    def _sleep_particles(self, fluid_state: State) -> None:
        wp.launch(
            kernel=sleep_sph_particles,
            dim=self.model.particle_count,
            inputs=[
                fluid_state.particle_qd,
                self.model.particle_flags,
                self.model.particle_inv_mass,
                self.sleep_threshold,
            ],
            device=self.model.device,
        )

    def _collide_diffuse_with_shapes(self, fluid_state: State) -> None:
        model = self.model
        if not self._has_shape_collision_arrays():
            return

        assert self.diffuse_positions is not None
        assert self.diffuse_velocities is not None
        assert self.diffuse_worlds is not None
        assert model.shape_transform is not None
        assert model.shape_body is not None
        assert model.shape_type is not None
        assert model.shape_scale is not None
        assert model.shape_source_ptr is not None
        assert model.shape_flags is not None
        assert model.shape_margin is not None
        assert model.shape_world is not None
        assert model.shape_heightfield_index is not None
        assert model.heightfield_data is not None
        assert model.heightfield_elevations is not None
        body_q, body_qd, body_f, body_com, body_flags = self._body_arrays(fluid_state)
        diffuse_radius = 0.25 * self.smoothing_length
        wp.launch(
            kernel=collide_sph_diffuse_particles_with_shapes,
            dim=self.max_diffuse_particles,
            inputs=[
                self.diffuse_positions,
                self.diffuse_velocities,
                self.diffuse_worlds,
                body_q,
                body_qd,
                body_f,
                body_com,
                body_flags,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_source_ptr,
                model.shape_flags,
                model.shape_margin,
                model.shape_world,
                model.shape_heightfield_index,
                model.heightfield_data,
                model.heightfield_elevations,
                model.shape_count,
                diffuse_radius,
                self.boundary_damping,
                self._shape_collision_distance_value(),
                self.shape_collision_margin,
                self._shape_restitution_value(),
                self.shape_friction,
                self.shape_adhesion,
            ],
            device=model.device,
        )

    def _project_density(self, state_in: State, state_out: State, dt: float) -> None:
        model = self.model
        assert model.particle_grid is not None
        assert self.particle_density is not None
        assert self.pbf_lambdas is not None
        assert self.pbf_deltas is not None

        for _ in range(self.pbf_iterations):
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(state_out.particle_q, radius=self.smoothing_length)

            wp.launch(
                kernel=compute_pbf_lambdas,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_out.particle_q,
                    model.particle_mass,
                    model.particle_flags,
                    self.smoothing_length,
                    self.rest_density,
                    self.pbf_relaxation_epsilon,
                    self.particle_density,
                    self.pbf_lambdas,
                ],
                device=model.device,
            )

            wp.launch(
                kernel=solve_pbf_deltas,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_out.particle_q,
                    model.particle_mass,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.pbf_lambdas,
                    self.smoothing_length,
                    self.rest_density,
                    self.pbf_artificial_pressure,
                    self.pbf_artificial_radius,
                    self.pbf_artificial_power,
                    self.pbf_max_delta,
                    self.pbf_deltas,
                ],
                device=model.device,
            )

            wp.launch(
                kernel=apply_pbf_deltas,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    model.particle_radius,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.pbf_deltas,
                    self.bounds_lower,
                    self.bounds_upper,
                    self.pbf_relaxation,
                ],
                device=model.device,
            )

        wp.launch(
            kernel=finalize_pbf_velocities,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                model.particle_flags,
                model.particle_inv_mass,
                self.max_acceleration,
                self.max_velocity,
                dt,
                state_out.particle_qd,
            ],
            device=model.device,
        )

    def _smooth_velocities(self, fluid_state: State, dt: float) -> None:
        model = self.model
        assert model.particle_grid is not None
        assert self.particle_velocity_smooth is not None

        with wp.ScopedDevice(model.device):
            model.particle_grid.build(fluid_state.particle_q, radius=self.smoothing_length)

        wp.launch(
            kernel=smooth_sph_velocities,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                fluid_state.particle_q,
                fluid_state.particle_qd,
                model.particle_flags,
                model.particle_inv_mass,
                self.smoothing_length,
                self.xsph_strength,
                self.max_acceleration,
                self.max_velocity,
                dt,
                self.particle_velocity_smooth,
            ],
            device=model.device,
        )
        wp.launch(
            kernel=copy_sph_velocities,
            dim=model.particle_count,
            inputs=[
                self.particle_velocity_smooth,
                model.particle_flags,
                model.particle_inv_mass,
                fluid_state.particle_qd,
            ],
            device=model.device,
        )

    def _update_render_particles(self, fluid_state: State, rebuild_grid: bool = True) -> None:
        model = self.model
        assert model.particle_grid is not None
        assert self.render_positions is not None
        assert self.render_anisotropy is not None
        assert self.render_anisotropy_secondary is not None
        assert self.render_anisotropy_tertiary is not None

        if rebuild_grid:
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(fluid_state.particle_q, radius=self.smoothing_length)

        wp.launch(
            kernel=compute_sph_render_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                fluid_state.particle_q,
                fluid_state.particle_qd,
                model.particle_flags,
                self.smoothing_length,
                self.render_smoothing,
                self.render_anisotropy_scale,
                self.render_anisotropy_min,
                self.render_anisotropy_max,
                self.render_positions,
                self.render_anisotropy,
                self.render_anisotropy_secondary,
                self.render_anisotropy_tertiary,
            ],
            device=model.device,
        )
        self.render_buffers_valid = True

    def _step_diffuse_particles(self, fluid_state: State, dt: float, rebuild_grid: bool = True) -> None:
        model = self.model
        self._ensure_diffuse_storage()
        assert model.particle_grid is not None
        assert self.particle_density is not None
        assert self.diffuse_positions is not None
        assert self.diffuse_velocities is not None
        assert self.diffuse_worlds is not None
        assert self.diffuse_slot_states is not None
        assert self.diffuse_spawn_counter is not None

        if rebuild_grid:
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(fluid_state.particle_q, radius=self.smoothing_length)

        wp.launch(
            kernel=update_sph_diffuse_particles,
            dim=self.max_diffuse_particles,
            inputs=[
                model.particle_grid.id,
                fluid_state.particle_q,
                fluid_state.particle_qd,
                model.particle_flags,
                model.gravity,
                self.smoothing_length,
                self.bounds_lower,
                self.bounds_upper,
                self.boundary_damping,
                self.diffuse_lifetime,
                self.diffuse_drag,
                self.diffuse_buoyancy,
                self.diffuse_ballistic,
                dt,
                self.diffuse_positions,
                self.diffuse_velocities,
                self.diffuse_worlds,
                self.diffuse_slot_states,
            ],
            device=model.device,
        )

        wp.launch(
            kernel=advance_sph_diffuse_seed,
            dim=1,
            inputs=[self.diffuse_frame_seed],
            device=model.device,
        )
        wp.launch(
            kernel=spawn_sph_diffuse_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                fluid_state.particle_q,
                fluid_state.particle_qd,
                model.particle_flags,
                model.particle_world,
                self.particle_density,
                self.smoothing_length,
                self.rest_density,
                self.diffuse_threshold,
                self.diffuse_spawn_probability,
                self.diffuse_jitter,
                self.diffuse_surface_density_ratio,
                self.diffuse_ballistic,
                self.bounds_lower,
                self.bounds_upper,
                self.boundary_damping,
                self.diffuse_frame_seed,
                self.diffuse_spawn_counter,
                self.diffuse_positions,
                self.diffuse_velocities,
                self.diffuse_worlds,
                self.diffuse_slot_states,
            ],
            device=model.device,
        )

        if self.shape_collision and model.shape_count > 0:
            self._collide_diffuse_with_shapes(fluid_state)
