# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math

import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase
from .fluid_kernels import (
    advance_fluid_diffuse_seed,
    collide_fluid_diffuse_particles_with_shapes,
    compute_fluid_lambdas,
    compute_fluid_render_particles,
    compute_fluid_vorticity,
    sample_fluid_render_particles,
    sample_fluid_render_positions,
    solve_fluid_deltas,
    solve_fluid_velocities,
    spawn_fluid_diffuse_particles,
    update_fluid_diffuse_particles,
)
from .kernels import (
    accumulate_weighted_contact_impulse,
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_shape_restitution,
    apply_rigid_restitution,
    bending_constraint,
    clamp_body_motion,
    clamp_body_velocities,
    compute_morton_keys,
    compute_particle_bounds_min,
    convert_contact_impulse_to_force,
    convert_joint_impulse_to_parent_f,
    copy_kinematic_body_state_kernel,
    gather_float,
    gather_int32,
    gather_uint32,
    gather_vec3,
    solve_body_contact_positions,
    solve_body_joints,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    # solve_simple_body_joints,
    solve_springs,
    solve_tetrahedra,
    update_body_velocities,
)


def _poly6(r_sq: float, h: float) -> float:
    if r_sq >= h * h:
        return 0.0
    x = h * h - r_sq
    return 315.0 / (64.0 * math.pi * h**9) * x * x * x


class SolverXPBD(SolverBase):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU, Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    After constructing :class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Limitations:
        **Momentum conservation** -- When ``rigid_contact_con_weighting`` is
        enabled (the default), each body's positional correction is divided by
        its number of active contacts.  This improves convergence for stacking
        scenarios but means the solver does not conserve momentum at contacts.
        Reported per-contact forces (see :meth:`update_contacts`) are
        approximate: for contacts between two dynamic bodies the force is
        computed using the harmonic mean of the two bodies' contact counts,
        which is symmetric but not exact.

        **Reported parent-joint forces** (see :attr:`~newton.State.body_parent_f`,
        populated when the extended state attribute is requested) are
        approximate.  XPBD applies relaxation factors
        (``joint_linear_relaxation``, ``joint_angular_relaxation``) to each
        joint constraint correction, and with a finite ``iterations`` count
        residual constraint error remains at end-of-step, so the reported
        wrench is the *applied* constraint reaction rather than the exact
        wrench needed to enforce the joint perfectly.  The convention matches
        :class:`~newton.solvers.SolverFeatherstone` and
        :class:`~newton.solvers.SolverMuJoCo`: it is the spatial wrench
        transmitted from the parent through the inbound joint, in world frame
        at the child body's COM, **including** both the constraint reaction
        and the body-frame contribution of :attr:`~newton.Control.joint_f`.
        In equilibrium this wrench counters all applied forces (gravity,
        contacts, ``State.body_f``) by Newton's third law.

    Joint limitations:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE, D6.
          CABLE joints are not supported.
        - :attr:`~newton.Model.joint_enabled`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`, and
          :attr:`~newton.Control.joint_f` are supported.
          Joint limits are enforced as hard positional constraints (``joint_limit_ke``/``joint_limit_kd`` are not used).
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          and :attr:`~newton.Model.joint_target_mode` are not supported.
        - Equality and mimic constraints are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Fluids:
        Particles flagged with :attr:`newton.ParticleFlags.FLUID` are simulated
        as a position-based fluid (Macklin & Müller, "Position Based Fluids",
        2013): instead of pairwise contact constraints, fluid particle pairs
        generate density constraints that are solved inside the regular
        XPBD iteration loop, so fluids two-way couple with rigid bodies, cloth,
        and soft bodies. Fluid particles are
        free to travel anywhere and collide with shapes through the standard
        particle contact pipeline. A bounded cohesion term (``fluid_cohesion``)
        makes splashes coagulate into strands and droplets instead of dispersing
        into isolated particles; viscosity and vorticity confinement act on
        velocities after the position solve.

        An optional render-only foam/spray layer (``max_diffuse_particles``)
        spawns Flex-style diffuse particles from a trapped-air/wave-crest
        potential over the fluid surface. The results in
        :attr:`diffuse_positions` and :attr:`diffuse_velocities` can be passed
        to ``Viewer.log_fluid_diffuse()``.

        .. code-block:: python

            builder.add_particle_grid(
                ...,
                flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
            )
            solver = newton.solvers.SolverXPBD(model, iterations=3, fluid_rest_distance=0.05)

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverXPBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    @property
    def fluid_cohesion(self) -> float:
        """Fluid cohesion strength in ``[0, 1]``."""
        return self._fluid_cohesion

    @fluid_cohesion.setter
    def fluid_cohesion(self, value: float) -> None:
        self._fluid_cohesion = min(max(float(value), 0.0), 1.0)
        if hasattr(self, "_fluid_rest_distance_eff"):
            self._fluid_cohesion_step = 0.02 * self._fluid_rest_distance_eff * self._fluid_cohesion

    def __init__(
        self,
        model: Model,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        joint_linear_relaxation: float = 0.7,
        joint_angular_relaxation: float = 0.4,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        angular_damping: float = 0.0,
        enable_restitution: bool = False,
        fluid_rest_distance: float | None = None,
        fluid_smoothing_length: float | None = None,
        fluid_rest_density: float | None = None,
        fluid_cohesion: float = 1.0,
        fluid_viscosity: float = 0.0,
        fluid_vorticity_confinement: float = 0.0,
        fluid_relaxation: float = 1.0,
        fluid_max_neighbors: int = 0,
        body_max_velocity: float = 0.0,
        body_max_angular_velocity: float = 0.0,
        max_diffuse_particles: int = 0,
        diffuse_threshold: float = 2.0,
        diffuse_lifetime: float = 1.5,
        diffuse_drag: float = 0.8,
        diffuse_buoyancy: float = 0.25,
        diffuse_ballistic: int = 8,
        diffuse_spawn_probability: float = 0.35,
        diffuse_jitter: float | None = None,
        diffuse_surface_density_ratio: float = 0.92,
        diffuse_shape_collision: bool = True,
    ):
        """
        Args:
            model: Model to simulate.
            iterations: Number of XPBD constraint iterations per step.
            soft_body_relaxation: Relaxation factor for soft body (FEM) constraints.
            soft_contact_relaxation: Relaxation factor for particle contact constraints.
            joint_linear_relaxation: Relaxation factor for linear joint corrections.
            joint_angular_relaxation: Relaxation factor for angular joint corrections.
            joint_linear_compliance: Compliance of linear joint constraints [m/N].
            joint_angular_compliance: Compliance of angular joint constraints [rad/(N·m)].
            rigid_contact_relaxation: Relaxation factor for rigid contact constraints.
            rigid_contact_con_weighting: Whether to divide rigid contact corrections by the
                number of active contacts per body.
            angular_damping: Angular damping coefficient applied during body integration.
            enable_restitution: Whether to apply restitution at contacts.
            fluid_rest_distance: Rest spacing between fluid particles [m]. If ``None``,
                twice the maximum particle radius is used (touching spheres). Particles
                spawned on a grid with this spacing are exactly at rest density.
            fluid_smoothing_length: Smoothing-kernel support radius [m]. If ``None``,
                ``1.8 * fluid_rest_distance`` is used, mirroring the rest-distance to
                interaction-radius ratio used by Flex fluids.
            fluid_rest_density: Fluid rest density [kg/m³]. If ``None``, it is calibrated
                from the mean fluid particle mass so that a regular grid of particles at
                ``fluid_rest_distance`` spacing is exactly at rest.
            fluid_cohesion: How strongly the fluid holds together, in ``[0, 1]``.
                Scales a bounded Akinci-style cohesion term that attracts neighbors
                at mid-range and repels at short range, producing surface-tension-like
                coagulation of splashes into droplets and strands. ``0`` disables
                cohesion so splashes disperse into individual particles.
            fluid_viscosity: Viscosity in ``[0, 1]``: per-substep blend toward the
                kernel-weighted neighborhood velocity. Small values (``0.01``-``0.1``)
                suit water; values near ``1`` give honey-like behavior.
            fluid_vorticity_confinement: Vorticity confinement strength that re-injects
                rotational motion lost to the position solve. ``0`` disables it.
            fluid_relaxation: Per-iteration scale on fluid density corrections.
            fluid_max_neighbors: Cap on the neighbors each fluid particle processes
                in the density solve. ``0`` (default) processes all neighbors; a
                positive value bounds the per-particle cost so a momentary
                over-compressed clump cannot stall its whole warp. Set it above
                the bulk neighbor count (so the rest state is never truncated and
                stays consistent with the rest-density calibration).
            body_max_velocity: Per-substep cap on dynamic-body linear speed [m/s].
                ``0`` (default) disables it. A small positive value prevents a
                body slammed into deep penetration from receiving a divergent
                correction velocity that tunnels and blows up to NaN.
            body_max_angular_velocity: Per-substep cap on dynamic-body angular
                speed [rad/s]. ``0`` (default) disables it.
            max_diffuse_particles: Maximum secondary foam/spray particles. Set to
                ``0`` (default) to disable the diffuse particle layer. When enabled,
                foam state is written to :attr:`diffuse_positions` and
                :attr:`diffuse_velocities`, which can be passed to
                ``Viewer.log_fluid_diffuse()``.
            diffuse_threshold: Trapped-air/wave-crest potential needed to emit a
                diffuse particle. The potential favors surface particles whose
                neighborhood is rapidly compressing or cresting.
            diffuse_lifetime: Diffuse particle lifetime [s].
            diffuse_drag: Blend rate toward neighboring fluid velocity [1/s].
            diffuse_buoyancy: Fraction of gravity canceled while a diffuse
                particle has fluid neighbors.
            diffuse_ballistic: Neighbor count below which a diffuse particle falls
                ballistically (spray) instead of advecting with the fluid (foam).
            diffuse_spawn_probability: Per-step spawn probability multiplier once
                the diffuse potential exceeds ``diffuse_threshold``.
            diffuse_jitter: Spawn offset radius [m]. If ``None``, a fraction of the
                fluid smoothing length is used.
            diffuse_surface_density_ratio: Density threshold used to classify
                surface particles as diffuse emitters.
            diffuse_shape_collision: Whether diffuse particles collide with shapes.
        """
        super().__init__(model=model)
        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation

        self.joint_linear_relaxation = joint_linear_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation
        self.joint_linear_compliance = joint_linear_compliance
        self.joint_angular_compliance = joint_angular_compliance

        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting

        self.angular_damping = angular_damping

        self.enable_restitution = enable_restitution

        self.fluid_rest_distance = fluid_rest_distance
        self.fluid_smoothing_length = fluid_smoothing_length
        self.fluid_rest_density = fluid_rest_density
        self.fluid_cohesion = fluid_cohesion
        self.fluid_viscosity = min(max(float(fluid_viscosity), 0.0), 1.0)
        self.fluid_vorticity_confinement = max(float(fluid_vorticity_confinement), 0.0)
        self.fluid_relaxation = max(float(fluid_relaxation), 0.0)
        self.fluid_max_neighbors = int(fluid_max_neighbors)
        self.body_max_velocity = max(float(body_max_velocity), 0.0)
        self.body_max_angular_velocity = max(float(body_max_angular_velocity), 0.0)

        self.max_diffuse_particles = max(int(max_diffuse_particles), 0)
        self.diffuse_threshold = float(max(diffuse_threshold, 0.0))
        self.diffuse_lifetime = float(max(diffuse_lifetime, 1.0e-6))
        self.diffuse_drag = float(max(diffuse_drag, 0.0))
        self.diffuse_buoyancy = float(max(diffuse_buoyancy, 0.0))
        self.diffuse_ballistic = max(int(diffuse_ballistic), 1)
        self.diffuse_spawn_probability = float(max(diffuse_spawn_probability, 0.0))
        self.diffuse_jitter = diffuse_jitter
        self.diffuse_surface_density_ratio = float(max(diffuse_surface_density_ratio, 0.0))
        self.diffuse_shape_collision = bool(diffuse_shape_collision)
        self.diffuse_enabled = self.max_diffuse_particles > 0

        self.render_positions: wp.array[wp.vec3] | None = None
        self.render_anisotropy: wp.array[wp.vec4] | None = None
        self.render_anisotropy_secondary: wp.array[wp.vec4] | None = None
        self.render_anisotropy_tertiary: wp.array[wp.vec4] | None = None
        self._render_positions_full: wp.array[wp.vec3] | None = None
        self._render_anisotropy_full: wp.array[wp.vec4] | None = None
        self._render_anisotropy_secondary_full: wp.array[wp.vec4] | None = None
        self._render_anisotropy_tertiary_full: wp.array[wp.vec4] | None = None

        self.diffuse_positions: wp.array[wp.vec4] | None = None
        self.diffuse_velocities: wp.array[wp.vec4] | None = None
        self.diffuse_worlds: wp.array[wp.int32] | None = None
        self.diffuse_slot_states: wp.array[wp.int32] | None = None
        self.diffuse_spawn_counter: wp.array[wp.int32] | None = None
        self.diffuse_frame_seed: wp.array[wp.int32] | None = None
        self._diffuse_empty_bodies: tuple | None = None

        self._fluid_density: wp.array[wp.float32] | None = None
        self._fluid_lambda: wp.array[wp.float32] | None = None
        self._fluid_vorticity: wp.array[wp.vec3] | None = None
        self._all_fluid = False
        # Lazily allocated scratch for reorder_particles (spatial sort buffers
        # plus per-dtype gather destinations, keyed by array label).
        self._reorder_keys: wp.array[wp.int32] | None = None
        self._reorder_indices: wp.array[wp.int32] | None = None
        self._reorder_bounds: wp.array[wp.float32] | None = None
        self._reorder_scratch: dict = {}
        self._update_fluid_settings()
        self._ensure_diffuse_storage()
        self._ensure_particle_grid_ready()

        self.compute_body_velocity_from_position_delta = False

        self._init_kinematic_state()

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()
        if flags & ModelFlags.PARTICLE_PROPERTIES:
            self._update_fluid_settings()
            self._ensure_particle_grid_ready()

    def _update_fluid_settings(self) -> None:
        """Resolve fluid parameters and allocate fluid buffers if the model contains fluid particles."""
        model = self.model
        self._has_fluid = False
        self._all_fluid = False
        if model.particle_count == 0 or model.particle_flags is None:
            return

        flags = model.particle_flags.numpy()
        active_mask = (flags & int(ParticleFlags.ACTIVE)) != 0
        fluid_mask = active_mask & ((flags & int(ParticleFlags.FLUID)) != 0)
        if not fluid_mask.any():
            return
        # whether every active particle is fluid: lets the solver skip the
        # particle-particle contact kernel entirely (fluid-fluid pairs are
        # handled by the density constraint, so that kernel would be pure waste)
        self._all_fluid = not bool((active_mask & ~fluid_mask).any())

        rest_distance = self.fluid_rest_distance
        if rest_distance is None:
            rest_distance = 2.0 * model.particle_max_radius
        if rest_distance <= 0.0:
            raise ValueError("fluid_rest_distance must be positive (or particle radii must be nonzero)")

        h = self.fluid_smoothing_length
        if h is None:
            h = 1.8 * rest_distance
        if h <= rest_distance:
            raise ValueError("fluid_smoothing_length must be larger than fluid_rest_distance")

        fluid_masses = model.particle_mass.numpy()[fluid_mask]
        mean_mass = float(fluid_masses.mean())

        rest_density = self.fluid_rest_density
        if rest_density is None:
            if mean_mass <= 0.0:
                raise ValueError(
                    "fluid_rest_density cannot be calibrated from massless fluid particles; pass it explicitly"
                )
            # calibrate against a regular particle grid at rest spacing so that
            # grid-initialized fluid starts exactly at rest density
            n = int(math.ceil(h / rest_distance))
            lattice_sum = 0.0
            for ix in range(-n, n + 1):
                for iy in range(-n, n + 1):
                    for iz in range(-n, n + 1):
                        r_sq = float(ix * ix + iy * iy + iz * iz) * rest_distance * rest_distance
                        lattice_sum += _poly6(r_sq, h)
            rest_density = mean_mass * lattice_sum
        if rest_density <= 0.0:
            raise ValueError("fluid_rest_density must be positive")

        # scale-invariant CFM regularizer: a small fraction of the constraint-
        # gradient denominator of a particle in the bulk of the rest lattice
        if mean_mass > 0.0:
            n = int(math.ceil(h / rest_distance))
            grad_sq_sum = 0.0
            for ix in range(-n, n + 1):
                for iy in range(-n, n + 1):
                    for iz in range(-n, n + 1):
                        if ix == 0 and iy == 0 and iz == 0:
                            continue
                        r = math.sqrt(float(ix * ix + iy * iy + iz * iz)) * rest_distance
                        if r >= h:
                            continue
                        g = 45.0 / (math.pi * h**6) * (h - r) ** 2 * mean_mass / rest_density
                        grad_sq_sum += g * g / mean_mass
            relaxation_epsilon = 1.0e-3 * grad_sq_sum
        else:
            relaxation_epsilon = 1.0e-6

        self._has_fluid = True
        self._fluid_rest_distance_eff = rest_distance
        self._fluid_h = h
        self._fluid_rest_density_eff = rest_density
        self._fluid_eps = relaxation_epsilon
        self._fluid_max_delta = 0.5 * h
        # bound the per-pair cohesion bias to a small fraction of the rest
        # spacing per iteration so the term stays contractive (no overshoot
        # oscillation) regardless of neighborhood size, and weak enough that
        # the incompressibility correction dominates in the bulk
        self._fluid_cohesion_step = 0.02 * rest_distance * self.fluid_cohesion

        n = model.particle_count
        if self._fluid_density is None or len(self._fluid_density) != n:
            self._fluid_density = wp.zeros(n, dtype=wp.float32, device=model.device)
            self._fluid_lambda = wp.zeros(n, dtype=wp.float32, device=model.device)
            self._fluid_vorticity = wp.zeros(n, dtype=wp.vec3, device=model.device)

    def reorder_particles(self, state: State) -> None:
        """Sort fluid particles into spatial order to keep the density solve fast.

        Free fluid particles are created in spatial order, so the hash-grid
        neighbor gathers in the density constraint start cache-coherent. As the
        fluid spreads, consecutive particle indices scatter across memory and
        those neighbor reads lose locality, multiplying the solve cost. Sorting
        the particles back into Morton (Z-curve) order restores it.

        The reorder is a pure relabeling, so the simulation result is unchanged.
        It only runs when every active particle is a free fluid particle;
        reordering would otherwise scramble the index-based topology of cloth or
        soft bodies. Call it once per frame, before the substep loop. Every step
        is on device and CUDA-graph-capturable, so it may run inside a captured
        region.

        Args:
            state: State whose ``particle_q`` defines the sort order; its
                ``particle_q`` and ``particle_qd`` are permuted in place along
                with the model's per-particle arrays.
        """
        model = self.model
        n = model.particle_count
        if not self._all_fluid or not self._has_fluid or n <= 1:
            return

        dev = model.device
        if self._reorder_keys is None or len(self._reorder_indices) < 2 * n:
            # radix_sort_pairs sorts the first n entries using the rest as scratch
            self._reorder_keys = wp.empty(2 * n, dtype=wp.int32, device=dev)
            self._reorder_indices = wp.empty(2 * n, dtype=wp.int32, device=dev)
            self._reorder_bounds = wp.empty(3, dtype=wp.float32, device=dev)
            self._reorder_scratch = {}

        # Morton key per particle relative to the cloud's lower corner.
        self._reorder_bounds.fill_(1.0e30)
        wp.launch(compute_particle_bounds_min, dim=n, inputs=[state.particle_q, self._reorder_bounds], device=dev)
        wp.launch(
            compute_morton_keys,
            dim=n,
            inputs=[
                state.particle_q,
                self._reorder_bounds,
                1.0 / self._fluid_h,
                self._reorder_keys,
                self._reorder_indices,
            ],
            device=dev,
        )
        wp.utils.radix_sort_pairs(self._reorder_keys, self._reorder_indices, n)
        perm = self._reorder_indices  # first n entries: old indices in sorted order

        # Permute every per-particle array by the same permutation (pure relabel).
        for owner, name in (
            (state, "particle_q"),
            (state, "particle_qd"),
            (model, "particle_q"),
            (model, "particle_qd"),
            (model, "particle_colors"),
            (model, "particle_mass"),
            (model, "particle_inv_mass"),
            (model, "particle_radius"),
            (model, "particle_flags"),
            (model, "particle_world"),
        ):
            self._permute_particle_array(getattr(owner, name, None), perm, n, f"{type(owner).__name__}.{name}")

    def _permute_particle_array(self, arr, perm, n: int, label: str) -> None:
        """Gather ``arr`` by ``perm`` into cached scratch, then copy it back in place."""
        if arr is None or len(arr) != n:
            return
        dtype = arr.dtype
        if dtype == wp.vec3:
            kernel = gather_vec3
        elif dtype == wp.float32:
            kernel = gather_float
        elif dtype == wp.uint32:
            kernel = gather_uint32
        elif dtype == wp.int32:
            kernel = gather_int32
        else:
            return
        scratch = self._reorder_scratch.get(label)
        if scratch is None or len(scratch) != n or scratch.dtype != dtype:
            scratch = wp.empty(n, dtype=dtype, device=self.model.device)
            self._reorder_scratch[label] = scratch
        wp.launch(kernel, dim=n, inputs=[arr, perm, scratch], device=self.model.device)
        wp.copy(arr, scratch)

    def _build_particle_grid(self, particle_q: wp.array[wp.vec3], radius: float) -> None:
        model = self.model
        with wp.ScopedDevice(model.device):
            model.particle_grid.build(particle_q, radius=radius, groups=self._particle_grid_groups())

    def _particle_grid_groups(self) -> wp.array[wp.int32] | None:
        model = self.model
        return model.particle_world if model.world_count > 1 else None

    def _ensure_particle_grid_ready(self) -> None:
        """Reserve and initialize grouped-grid metadata outside graph capture."""
        model = self.model
        if model.particle_count <= 1 or model.particle_grid is None:
            return
        with wp.ScopedDevice(model.device):
            model.particle_grid.reserve(model.particle_count)
        if self._has_fluid:
            self._build_particle_grid(model.particle_q, self._fluid_h)

    def update_render_particles(
        self,
        state: State,
        smoothing: float = 0.5,
        anisotropy_scale: float = 1.0,
        anisotropy_min: float = 0.2,
        anisotropy_max: float = 2.0,
        max_particles: int = 0,
    ) -> None:
        """Compute smoothed, anisotropic render particles for fluid surface rendering.

        Mirrors Flex's smoothed-particle and anisotropy outputs: particle positions
        are Laplacian-smoothed toward their kernel-weighted neighborhood center, and
        per-particle ellipsoid axes are fit to the neighborhood covariance so that
        stretched splashes render as connected sheets and strands rather than
        individual spheres. Results are written to :attr:`render_positions`,
        :attr:`render_anisotropy`, :attr:`render_anisotropy_secondary`, and
        :attr:`render_anisotropy_tertiary`, which can be passed to
        ``Viewer.log_fluid()``.

        Args:
            state: State whose ``particle_q``/``particle_qd`` are used.
            smoothing: Position smoothing strength in ``[0, 1]``.
            anisotropy_scale: Stretch multiplier for the ellipsoid fit. ``0`` keeps spheres.
            anisotropy_min: Minimum ellipsoid axis scale as a fraction of particle radius.
            anisotropy_max: Maximum ellipsoid axis scale as a fraction of particle radius.
            max_particles: Maximum particles in the render output. ``0`` keeps all particles.
                When both ``smoothing`` and ``anisotropy_scale`` are zero, positions are sampled
                directly without a render-neighbor pass.
        """
        model = self.model
        if not self._has_fluid or model.particle_count == 0 or model.particle_grid is None:
            return

        n = model.particle_count
        render_count = n if max_particles <= 0 else min(int(max_particles), n)
        if smoothing <= 0.0 and anisotropy_scale <= 0.0:
            if render_count == n:
                self.render_positions = state.particle_q
            else:
                if self.render_positions is None or len(self.render_positions) != render_count:
                    self.render_positions = wp.empty(render_count, dtype=wp.vec3, device=model.device)
                wp.launch(
                    sample_fluid_render_positions,
                    dim=render_count,
                    inputs=[state.particle_q, n, render_count],
                    outputs=[self.render_positions],
                    device=model.device,
                )
            self.render_anisotropy = None
            self.render_anisotropy_secondary = None
            self.render_anisotropy_tertiary = None
            return

        if self._render_positions_full is None or len(self._render_positions_full) != n:
            self._render_positions_full = wp.empty(n, dtype=wp.vec3, device=model.device)
            self._render_anisotropy_full = wp.empty(n, dtype=wp.vec4, device=model.device)
            self._render_anisotropy_secondary_full = wp.empty(n, dtype=wp.vec4, device=model.device)
            self._render_anisotropy_tertiary_full = wp.empty(n, dtype=wp.vec4, device=model.device)

        h = self._fluid_h
        # Hash-grid storage is scratch: every simulation step rebuilds it before
        # querying. Reusing it here avoids a second full grid allocation and,
        # crucially, keeps CUDA graph replay compatible with grouped hash grids;
        # interleaving builds from a second grid can invalidate captured grouped
        # query metadata in Warp.
        self._build_particle_grid(state.particle_q, h)

        wp.launch(
            kernel=compute_fluid_render_particles,
            dim=n,
            inputs=[
                model.particle_grid.id,
                state.particle_q,
                state.particle_qd,
                model.particle_flags,
                model.particle_world,
                h,
                smoothing,
                anisotropy_scale,
                anisotropy_min,
                anisotropy_max,
                self._render_positions_full,
                self._render_anisotropy_full,
                self._render_anisotropy_secondary_full,
                self._render_anisotropy_tertiary_full,
            ],
            device=model.device,
        )

        if render_count == n:
            self.render_positions = self._render_positions_full
            self.render_anisotropy = self._render_anisotropy_full
            self.render_anisotropy_secondary = self._render_anisotropy_secondary_full
            self.render_anisotropy_tertiary = self._render_anisotropy_tertiary_full
            return

        if (
            self.render_positions is None
            or len(self.render_positions) != render_count
            or self.render_anisotropy is None
            or len(self.render_anisotropy) != render_count
            or self.render_anisotropy_secondary is None
            or len(self.render_anisotropy_secondary) != render_count
            or self.render_anisotropy_tertiary is None
            or len(self.render_anisotropy_tertiary) != render_count
        ):
            self.render_positions = wp.empty(render_count, dtype=wp.vec3, device=model.device)
            self.render_anisotropy = wp.empty(render_count, dtype=wp.vec4, device=model.device)
            self.render_anisotropy_secondary = wp.empty(render_count, dtype=wp.vec4, device=model.device)
            self.render_anisotropy_tertiary = wp.empty(render_count, dtype=wp.vec4, device=model.device)
        wp.launch(
            sample_fluid_render_particles,
            dim=render_count,
            inputs=[
                self._render_positions_full,
                self._render_anisotropy_full,
                self._render_anisotropy_secondary_full,
                self._render_anisotropy_tertiary_full,
                n,
                render_count,
            ],
            outputs=[
                self.render_positions,
                self.render_anisotropy,
                self.render_anisotropy_secondary,
                self.render_anisotropy_tertiary,
            ],
            device=model.device,
        )

    def _ensure_diffuse_storage(self) -> None:
        if not self.diffuse_enabled:
            return
        if self.diffuse_positions is not None and len(self.diffuse_positions) == self.max_diffuse_particles:
            return
        device = self.model.device
        self.diffuse_positions = wp.zeros(self.max_diffuse_particles, dtype=wp.vec4, device=device)
        self.diffuse_velocities = wp.zeros(self.max_diffuse_particles, dtype=wp.vec4, device=device)
        self.diffuse_worlds = wp.zeros(self.max_diffuse_particles, dtype=wp.int32, device=device)
        self.diffuse_slot_states = wp.zeros(self.max_diffuse_particles, dtype=wp.int32, device=device)
        self.diffuse_spawn_counter = wp.zeros(1, dtype=wp.int32, device=device)
        self.diffuse_frame_seed = wp.zeros(1, dtype=wp.int32, device=device)

    def clear_diffuse_particles(self) -> None:
        """Kill all live diffuse foam/spray particles."""
        if self.diffuse_positions is not None:
            self.diffuse_positions.zero_()
        if self.diffuse_velocities is not None:
            self.diffuse_velocities.zero_()
        if self.diffuse_slot_states is not None:
            self.diffuse_slot_states.zero_()
        if self.diffuse_worlds is not None:
            self.diffuse_worlds.zero_()
        if self.diffuse_spawn_counter is not None:
            self.diffuse_spawn_counter.zero_()
        if self.diffuse_frame_seed is not None:
            self.diffuse_frame_seed.zero_()

    def _step_diffuse_particles(self, state_out: State, dt: float) -> None:
        """Advance, spawn, and collide the diffuse foam/spray layer.

        Runs over the final particle state of a step. Spawning reuses the
        fluid densities computed by the last constraint iteration; the hash
        grid is rebuilt over the final positions because
        :func:`spawn_fluid_diffuse_particles` iterates particles through
        ``wp.hash_grid_point_id``.
        """
        model = self.model
        h = self._fluid_h
        jitter = 0.12 * h if self.diffuse_jitter is None else max(float(self.diffuse_jitter), 0.0)
        # effectively unbounded: XPBD fluids have no solver bounds
        bounds_lower = wp.vec3(-1.0e9, -1.0e9, -1.0e9)
        bounds_upper = wp.vec3(1.0e9, 1.0e9, 1.0e9)

        self._build_particle_grid(state_out.particle_q, h)

        wp.launch(
            kernel=update_fluid_diffuse_particles,
            dim=self.max_diffuse_particles,
            inputs=[
                model.particle_grid.id,
                state_out.particle_q,
                state_out.particle_qd,
                model.particle_flags,
                model.gravity,
                h,
                bounds_lower,
                bounds_upper,
                0.0,
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
            kernel=advance_fluid_diffuse_seed,
            dim=1,
            inputs=[self.diffuse_frame_seed],
            device=model.device,
        )
        wp.launch(
            kernel=spawn_fluid_diffuse_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state_out.particle_q,
                state_out.particle_qd,
                model.particle_flags,
                model.particle_world,
                self._fluid_density,
                h,
                self._fluid_rest_density_eff,
                self.diffuse_threshold,
                self.diffuse_spawn_probability,
                jitter,
                self.diffuse_surface_density_ratio,
                self.diffuse_ballistic,
                bounds_lower,
                bounds_upper,
                0.0,
                self.diffuse_frame_seed,
                self.diffuse_spawn_counter,
                self.diffuse_positions,
                self.diffuse_velocities,
                self.diffuse_worlds,
                self.diffuse_slot_states,
            ],
            device=model.device,
        )

        if self.diffuse_shape_collision and model.shape_count > 0 and model.shape_transform is not None:
            if model.body_count > 0:
                body_q, body_qd, body_f = state_out.body_q, state_out.body_qd, state_out.body_f
                body_com, body_flags = model.body_com, model.body_flags
            else:
                if self._diffuse_empty_bodies is None:
                    self._diffuse_empty_bodies = (
                        wp.empty(0, dtype=wp.transform, device=model.device),
                        wp.empty(0, dtype=wp.spatial_vector, device=model.device),
                        wp.empty(0, dtype=wp.spatial_vector, device=model.device),
                        wp.empty(0, dtype=wp.vec3, device=model.device),
                        wp.empty(0, dtype=wp.int32, device=model.device),
                    )
                body_q, body_qd, body_f, body_com, body_flags = self._diffuse_empty_bodies
            wp.launch(
                kernel=collide_fluid_diffuse_particles_with_shapes,
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
                    0.25 * h,
                    0.0,
                    0.5 * self._fluid_rest_distance_eff,
                    0.0,
                    0.0,
                    0.2,
                    0.0,
                    dt,
                ],
                device=model.device,
            )

    def copy_kinematic_body_state(self, model: Model, state_in: State, state_out: State):
        if model.body_count == 0:
            return
        wp.launch(
            kernel=copy_kinematic_body_state_kernel,
            dim=model.body_count,
            inputs=[model.body_flags, state_in.body_q, state_in.body_qd],
            outputs=[state_out.body_q, state_out.body_qd],
            device=model.device,
        )

    def _apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    def _apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.body_q = new_body_q
                state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float) -> None:
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model
        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_q_init = None
        body_qd_init = None
        body_deltas = None

        rigid_contact_inv_weight = None

        contact_impulse = None
        contact_impulse_iter = None

        if contacts:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros(model.body_count, dtype=float, device=model.device)
            rigid_contact_inv_weight_init = None

            if contacts.force is not None:
                contact_impulse = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device)
                contact_impulse_iter = wp.zeros(
                    contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device
                )

        # Optional per-joint accumulated child-side spatial impulse, used to
        # populate ``state_out.body_parent_f`` after the iteration loop.
        joint_impulse = None
        if state_out.body_parent_f is not None and model.joint_count > 0:
            joint_impulse = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd

                self.particle_q_init = wp.clone(state_in.particle_q)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)

                # Build/update the particle hash grid for particle-particle contact queries
                if model.particle_count > 1 and model.particle_grid is not None:
                    # Search radius must cover the maximum interaction distance used by the contact query
                    search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                    if self._has_fluid:
                        search_radius = max(search_radius, self._fluid_h)
                    self._build_particle_grid(state_out.particle_q, search_radius)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd)

                body_f_tmp = state_in.body_f
                if model.joint_count:
                    # Avoid accumulating joint_f into the persistent state body_f buffer.
                    body_f_tmp = wp.clone(state_in.body_f)
                    # ``joint_impulse`` (may be ``None`` when ``body_parent_f``
                    # was not requested) accumulates both the joint_f wrench
                    # contribution recorded here and the constraint-correction
                    # contribution added by :func:`solve_body_joints` inside
                    # the iteration loop.  Together they recover the total
                    # wrench transmitted to the child body, matching the
                    # :attr:`State.body_parent_f` convention.
                    wp.launch(
                        kernel=apply_joint_forces,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_type,
                            model.joint_enabled,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_X_c,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_axis,
                            control.joint_f,
                            dt,
                        ],
                        outputs=[body_f_tmp, joint_impulse],
                        device=model.device,
                    )

                if body_f_tmp is state_in.body_f:
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_tmp
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                    state_in.body_f = body_f_prev

                # Integration includes external picking forces. Bound the
                # resulting pose before constraints see it so one force spike
                # cannot cross a thin collider in a single substep.
                if self.body_max_velocity > 0.0 or self.body_max_angular_velocity > 0.0:
                    wp.launch(
                        kernel=clamp_body_motion,
                        dim=model.body_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            self.body_inv_mass_effective,
                            self.body_max_velocity,
                            self.body_max_angular_velocity,
                            dt,
                        ],
                        outputs=[state_out.body_q, state_out.body_qd],
                        device=model.device,
                    )

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    wp.int32(0),
                                    wp.int32(0),
                                    wp.int32(1),
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    self.body_inv_mass_effective,
                                    self.body_inv_inertia_effective,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        # Particle-particle contacts. Skipped entirely when every
                        # active particle is fluid: fluid-fluid pairs are resolved
                        # by the density constraint, so this kernel would only burn
                        # a grid query per particle to reject everything.
                        if model.particle_max_radius > 0.0 and model.particle_count > 1 and not self._all_fluid:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            assert model.particle_grid is not None
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_world,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # position-based fluid density constraints
                        if self._has_fluid and model.particle_count > 1 and model.particle_grid is not None:
                            wp.launch(
                                kernel=compute_fluid_lambdas,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    model.particle_mass,
                                    model.particle_inv_mass,
                                    model.particle_flags,
                                    model.particle_world,
                                    self._fluid_h,
                                    self._fluid_rest_density_eff,
                                    self._fluid_eps,
                                    self.fluid_max_neighbors,
                                    self._fluid_rest_distance_eff,
                                ],
                                outputs=[self._fluid_density, self._fluid_lambda],
                                device=model.device,
                            )
                            wp.launch(
                                kernel=solve_fluid_deltas,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    model.particle_mass,
                                    model.particle_inv_mass,
                                    model.particle_flags,
                                    model.particle_world,
                                    self._fluid_lambda,
                                    self._fluid_h,
                                    self._fluid_rest_density_eff,
                                    self._fluid_cohesion_step,
                                    self._fluid_max_delta,
                                    self.fluid_relaxation,
                                    self.fluid_max_neighbors,
                                    self._fluid_rest_distance_eff,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=solve_springs,
                                dim=model.spring_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.spring_indices,
                                    model.spring_rest_length,
                                    model.spring_stiffness,
                                    model.spring_damping,
                                    dt,
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # tetrahedral FEM
                        if model.tet_count:
                            wp.launch(
                                kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tet_indices,
                                    model.tet_poses,
                                    control.tet_activations,
                                    model.tet_materials,
                                    dt,
                                    self.soft_body_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        particle_q, particle_qd = self._apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                        # Re-project after the pressure solve so nonpenetration
                        # takes precedence. Dynamic shapes receive this reaction;
                        # otherwise pressure-induced impulses lose momentum.
                        if self._has_fluid and model.shape_count and contacts is not None:
                            particle_deltas.zero_()
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    wp.int32(ParticleFlags.FLUID),
                                    wp.int32(0),
                                    wp.int32(1),
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    self.body_inv_mass_effective,
                                    self.body_inv_inertia_effective,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    1.0,
                                ],
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )
                            particle_q, particle_qd = self._apply_particle_deltas(
                                model, state_in, state_out, particle_deltas, dt
                            )

                            # A dynamic object can squeeze fluid against a fixed
                            # boundary. Resolve fixed contacts once more so an
                            # opposing dynamic contact cannot average away the
                            # container constraint.
                            particle_deltas.zero_()
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    wp.int32(ParticleFlags.FLUID),
                                    wp.int32(1),
                                    wp.int32(0),
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    self.body_inv_mass_effective,
                                    self.body_inv_inertia_effective,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    1.0,
                                ],
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )
                            particle_q, particle_qd = self._apply_particle_deltas(
                                model, state_in, state_out, particle_deltas, dt
                            )

                    # Apply fluid/soft-body reaction as its own weighted
                    # manifold. Mixing these corrections with rigid contacts
                    # makes the result depend on how many rigid points happen
                    # to be generated for the same body.
                    if model.body_count and model.particle_count and model.shape_count and contacts is not None:
                        body_q, body_qd = self._apply_body_deltas(model, state_in, state_out, body_deltas, dt)
                        body_deltas.zero_()

                    # handle rigid bodies
                    # ----------------------------

                    # Solve rigid contact constraints
                    if model.body_count and contacts is not None:
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()

                        if contact_impulse_iter is not None:
                            contact_impulse_iter.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=contacts.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_flags,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.shape_body,
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_point0,
                                contacts.rigid_contact_point1,
                                contacts.rigid_contact_offset0,
                                contacts.rigid_contact_offset1,
                                contacts.rigid_contact_normal,
                                contacts.rigid_contact_margin0,
                                contacts.rigid_contact_margin1,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                model.shape_material_mu,
                                model.shape_material_mu_torsional,
                                model.shape_material_mu_rolling,
                                self.rigid_contact_relaxation,
                                dt,
                            ],
                            outputs=[
                                body_deltas,
                                rigid_contact_inv_weight,
                                contact_impulse_iter,
                            ],
                            device=model.device,
                        )

                        if contact_impulse_iter is not None:
                            wp.launch(
                                kernel=accumulate_weighted_contact_impulse,
                                dim=contacts.rigid_contact_max,
                                inputs=[
                                    contacts.rigid_contact_count,
                                    contact_impulse_iter,
                                    contacts.rigid_contact_shape0,
                                    contacts.rigid_contact_shape1,
                                    model.shape_body,
                                    rigid_contact_inv_weight,
                                ],
                                outputs=[contact_impulse],
                                device=model.device,
                            )

                        # if model.rigid_contact_count.numpy()[0] > 0:
                        #     print("rigid_contact_count:", model.rigid_contact_count.numpy().flatten())
                        #     # print("rigid_active_contact_distance:", rigid_active_contact_distance.numpy().flatten())
                        #     # print("rigid_active_contact_point0:", rigid_active_contact_point0.numpy().flatten())
                        #     # print("rigid_active_contact_point1:", rigid_active_contact_point1.numpy().flatten())
                        #     print("body_deltas:", body_deltas.numpy().flatten())

                        # print(rigid_active_contact_distance.numpy().flatten())

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

                        body_q, body_qd = self._apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

                    if model.joint_count:
                        if requires_grad:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.joint_type,
                                model.joint_enabled,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_X_p,
                                model.joint_X_c,
                                model.joint_limit_lower,
                                model.joint_limit_upper,
                                model.joint_qd_start,
                                model.joint_target_q_start,
                                model.joint_dof_dim,
                                model.joint_axis,
                                control.joint_target_q,
                                control.joint_target_qd,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                self.joint_linear_compliance,
                                self.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas, joint_impulse],
                            device=model.device,
                        )

                        body_q, body_qd = self._apply_body_deltas(model, state_in, state_out, body_deltas, dt)

            # post-projection fluid velocity pass: viscosity and vorticity confinement
            if (
                model.particle_count
                and self._has_fluid
                and model.particle_grid is not None
                and (self.fluid_viscosity > 0.0 or self.fluid_vorticity_confinement > 0.0)
            ):
                if self.fluid_vorticity_confinement > 0.0:
                    wp.launch(
                        kernel=compute_fluid_vorticity,
                        dim=model.particle_count,
                        inputs=[
                            model.particle_grid.id,
                            particle_q,
                            particle_qd,
                            model.particle_mass,
                            model.particle_flags,
                            model.particle_world,
                            self._fluid_density,
                            self._fluid_h,
                        ],
                        outputs=[self._fluid_vorticity],
                        device=model.device,
                    )
                new_particle_qd = wp.empty_like(particle_qd)
                wp.launch(
                    kernel=solve_fluid_velocities,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        particle_q,
                        particle_qd,
                        model.particle_mass,
                        model.particle_inv_mass,
                        model.particle_flags,
                        model.particle_world,
                        self._fluid_density,
                        self._fluid_vorticity,
                        self._fluid_h,
                        self.fluid_viscosity,
                        self.fluid_vorticity_confinement,
                        dt,
                    ],
                    outputs=[new_particle_qd],
                    device=model.device,
                )
                particle_qd = new_particle_qd

            self._contact_impulse = contact_impulse
            self._contact_impulse_capacity = contacts.rigid_contact_max if contacts is not None else 0
            self._last_dt = dt

            # Populate optional ``state_out.body_parent_f`` (incoming joint
            # wrench per body) from the per-joint accumulated child-side
            # impulse.  Bodies without an inbound joint (roots / free bodies)
            # remain zero-initialized, matching MuJoCo's behavior.
            if state_out.body_parent_f is not None:
                state_out.body_parent_f.zero_()
                if joint_impulse is not None:
                    wp.launch(
                        kernel=convert_joint_impulse_to_parent_f,
                        dim=model.joint_count,
                        inputs=[
                            joint_impulse,
                            model.joint_enabled,
                            model.joint_type,
                            model.joint_child,
                            dt,
                        ],
                        outputs=[state_out.body_parent_f],
                        device=model.device,
                    )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                if particle_qd.ptr != state_out.particle_qd.ptr:
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            if self.enable_restitution and contacts is not None:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_q_init,
                            body_qd,
                            body_qd_init,
                            model.body_com,
                            model.shape_body,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                        ],
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )

                if model.body_count:
                    body_deltas.zero_()

                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            self.body_inv_mass_effective,
                            self.body_inv_inertia_effective,
                            model.body_world,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_restitution,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            if model.body_count:
                self.copy_kinematic_body_state(model, state_in, state_out)

                # Stabilize dynamic bodies: clamp velocity and reset any
                # non-finite component before it can tunnel or poison contacts.
                if self.body_max_velocity > 0.0 or self.body_max_angular_velocity > 0.0:
                    wp.launch(
                        kernel=clamp_body_velocities,
                        dim=model.body_count,
                        inputs=[
                            self.body_inv_mass_effective,
                            self.body_max_velocity,
                            self.body_max_angular_velocity,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            # secondary foam/spray layer over the final fluid state (render
            # only, no feedback on the simulation)
            if (
                self.diffuse_enabled
                and self._has_fluid
                and model.particle_count
                and model.particle_grid is not None
                and not requires_grad
            ):
                self._ensure_diffuse_storage()
                self._step_diffuse_particles(state_out, dt)

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Populate ``contacts.force`` from XPBD contact impulses accumulated during the last :meth:`step`.

        Both force [N] and torque [N·m] components are written.  The torque
        includes torsional and rolling friction contributions that cannot be
        reconstructed from the linear force alone.

        When ``rigid_contact_con_weighting`` is enabled, the raw per-contact
        impulse is scaled to reflect the ``1/N`` correction that
        ``apply_body_deltas`` applies.  For contacts between a dynamic and a
        kinematic body, ``N`` is the dynamic body's contact count.  For
        contacts between two dynamic bodies, the harmonic mean
        ``2/(N_a + N_b)`` is used so that the reported force is symmetric with
        respect to body ordering.  This is an approximation -- the solver
        applies ``1/N_a`` and ``1/N_b`` independently to each side, so no
        single scalar can exactly represent both.

        Args:
            contacts: :class:`Contacts` object whose :attr:`~Contacts.force` buffer will be written.
                Must have been created with ``"force"`` in its requested attributes and must
                match the :class:`Contacts` instance (same ``rigid_contact_max``) passed to
                the preceding :meth:`step`.
            state: Unused (accepted for API compatibility with :class:`SolverBase`).

        Raises:
            ValueError: If ``contacts.force`` is ``None`` (not requested), if no step has been run yet,
                or if the contacts capacity does not match the one used in the last :meth:`step`.
        """
        if contacts.force is None:
            raise ValueError(
                "contacts.force is not allocated. Call model.request_contact_attributes('force') "
                "before creating the Contacts object."
            )
        if not hasattr(self, "_contact_impulse") or self._contact_impulse is None:
            raise ValueError("No contact impulse data available. Call step() before update_contacts().")
        if contacts.rigid_contact_max != self._contact_impulse_capacity:
            raise ValueError(
                f"Contacts capacity mismatch: update_contacts() received rigid_contact_max="
                f"{contacts.rigid_contact_max}, but step() used {self._contact_impulse_capacity}. "
                f"Pass the same Contacts instance to both step() and update_contacts()."
            )

        contacts.force.zero_()

        wp.launch(
            kernel=convert_contact_impulse_to_force,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                self._contact_impulse,
                self._last_dt,
            ],
            outputs=[contacts.force],
            device=self.model.device,
        )
