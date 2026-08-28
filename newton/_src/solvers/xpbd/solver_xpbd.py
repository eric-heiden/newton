# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelFlags, State
from ...utils.deprecation import deprecate_nonkeyword_arguments
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from ..vbd.particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    apply_planar_truncation_parallel_by_collision,
    apply_truncation_ts,
)
from ..vbd.tri_mesh_collision import TriMeshCollisionDetector, TriMeshCollisionInfo
from . import kernels
from .kernels import (
    accumulate_weighted_contact_impulse,
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_velocity_deltas,
    apply_rigid_restitution,
    bending_constraint,
    compute_particle_displacements,
    convert_contact_impulse_to_force,
    convert_joint_impulse_to_parent_f,
    copy_kinematic_body_state_kernel,
    count_bending_constraints,
    count_edge_edge_self_contacts,
    count_particle_shape_contact_bodies,
    count_triangle_intersection_self_contacts,
    count_vertex_triangle_self_contacts,
    damp_particle_velocities,
    eval_triangle_aerodynamics,
    remove_body_split_velocity,
    remove_particle_split_velocity,
    solve_body_contact_positions,
    solve_body_joints,
    solve_edge_edge_self_contact_velocities,
    solve_edge_edge_self_contacts,
    solve_particle_particle_contacts,
    solve_particle_shape_contact_velocities,
    solve_particle_shape_contacts,
    # solve_simple_body_joints,
    solve_springs,
    solve_tetrahedra,
    solve_triangle_intersection_self_contact_velocities,
    solve_triangle_intersection_self_contacts,
    solve_triangles,
    solve_vertex_triangle_self_contact_velocities,
    solve_vertex_triangle_self_contacts,
    update_body_velocities,
)


class SolverXPBD(SolverBase, CouplingInterface):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Miles Macklin, Kier Storey, Michelle Lu, Pierre Terdiman, Nuttapong Chentanez, Stefan Jeschke, and Matthias Müller. 2019. Small Steps in Physics Simulation. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '19). Association for Computing Machinery, New York, NY, USA, Article 39, 1-7. https://doi.org/10.1145/3309486.3340247
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

        **Cloth self-contact buffers** -- Vertex-triangle, edge-edge, and intersecting-triangle
        candidate buffers have fixed capacities. Increase
        ``particle_vertex_contact_buffer_size`` or
        ``particle_edge_contact_buffer_size`` or ``particle_triangle_contact_buffer_size`` for unusually dense folds;
        candidates beyond those capacities are discarded.

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
          ROD joints are not supported.
        - :attr:`~newton.Model.joint_enabled`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`, and
          :attr:`~newton.Control.joint_f` are supported.
          Joint limits are enforced as hard positional constraints (``joint_limit_ke``/``joint_limit_kd`` are not used).
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          and :attr:`~newton.Model.joint_target_mode` are not supported.
        - Equality and mimic constraints are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverXPBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    @deprecate_nonkeyword_arguments
    def __init__(
        self,
        model: Model,
        *,
        iterations: int = 2,
        integrate_with_external_rigid_solver: bool = False,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        soft_contact_max_depenetration_velocity: float | None = None,
        particle_enable_self_contact: bool = False,
        particle_enable_triangle_intersection_recovery: bool = False,
        particle_self_contact_relaxation: float | None = None,
        particle_triangle_intersection_relaxation: float | None = None,
        particle_self_contact_radius: float | None = None,
        particle_self_contact_margin: float | None = None,
        particle_max_depenetration_velocity: float | None = None,
        particle_vertex_contact_buffer_size: int = 32,
        particle_edge_contact_buffer_size: int = 64,
        particle_triangle_contact_buffer_size: int = 32,
        particle_edge_parallel_epsilon: float = 1.0e-5,
        particle_topological_contact_filter_threshold: int = 2,
        particle_rest_shape_contact_exclusion_radius: float = 0.0,
        joint_linear_relaxation: float = 0.7,
        joint_angular_relaxation: float = 0.4,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        angular_damping: float = 0.0,
        particle_damping: float = 0.0,
        air_density: float = 0.0,
        air_drag_coefficient: float = 1.0,
        air_lift_coefficient: float = 0.0,
        air_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        enable_restitution: bool = False,
        deterministic: wp.DeterministicMode | None = None,
    ):
        """Initialize the XPBD solver.

        Args:
            model: Simulation model to integrate.
            iterations: Number of constraint-solver iterations per time step. Defaults to 2.
            integrate_with_external_rigid_solver: Whether rigid bodies are integrated by an external solver for
                one-way rigid-body-to-particle coupling. Defaults to ``False``.
            soft_body_relaxation: Relaxation factor applied to soft-body constraint corrections
                [dimensionless]. Defaults to 0.9.
            soft_contact_relaxation: Relaxation factor applied to particle-particle and particle-shape contact
                corrections [dimensionless]. Defaults to 0.9.
            soft_contact_max_depenetration_velocity: Maximum particle-shape depenetration velocity [m/s].
                ``None`` limits each step to one quarter of the particle radius.
            particle_enable_self_contact: Whether to solve cloth vertex-triangle and edge-edge self-contact.
                Defaults to ``False``.
            particle_enable_triangle_intersection_recovery: Whether to recover exact intersecting-triangle pairs
                that discrete feature proximity misses. This also conservatively truncates nearby particle
                trajectories before the constraint solve. The robust path is more expensive and strongly dissipative.
                Defaults to ``False``.
            particle_self_contact_relaxation: Relaxation factor applied to cloth self-contact corrections
                [dimensionless]. ``None`` uses ``soft_contact_relaxation``.
            particle_triangle_intersection_relaxation: Relaxation factor applied to exact intersecting-triangle
                recovery [dimensionless]. ``None`` uses ``particle_self_contact_relaxation``.
            particle_self_contact_radius: Cloth self-contact separation distance [m]. ``None`` uses the model's
                maximum particle radius.
            particle_self_contact_margin: Cloth self-contact candidate-generation distance [m]. ``None`` uses
                twice ``particle_self_contact_radius``.
            particle_max_depenetration_velocity: Maximum cloth self-contact depenetration velocity [m/s].
                ``None`` limits each step to one quarter of ``particle_self_contact_radius``.
            particle_vertex_contact_buffer_size: Fixed vertex-triangle candidate capacity per particle.
                Defaults to 32.
            particle_edge_contact_buffer_size: Fixed edge-edge candidate capacity per edge. Defaults to 64.
            particle_triangle_contact_buffer_size: Fixed intersecting-triangle candidate capacity per triangle.
                Defaults to 32.
            particle_edge_parallel_epsilon: Threshold for treating cloth collision edges as parallel.
                Defaults to 1e-5.
            particle_topological_contact_filter_threshold: Topological distance in rings at or below which cloth
                self-contact candidates are discarded. Defaults to 2.
            particle_rest_shape_contact_exclusion_radius: Rest-pose distance below which cloth self-contact
                candidates are discarded [m]. Defaults to 0.0.
            joint_linear_relaxation: Relaxation factor applied to linear joint constraint corrections
                [dimensionless]. Defaults to 0.7.
            joint_angular_relaxation: Relaxation factor applied to angular joint constraint corrections
                [dimensionless]. Defaults to 0.4.
            joint_linear_compliance: Compliance shared by linear joint constraints [m/N]. Defaults to 0.0.
            joint_angular_compliance: Compliance shared by angular joint constraints [rad/(N·m)]. Defaults to 0.0.
            rigid_contact_relaxation: Relaxation factor applied to rigid contact constraint corrections
                [dimensionless]. Defaults to 0.8.
            rigid_contact_con_weighting: Whether to divide each rigid body's contact correction by its number of
                active contacts. Defaults to ``True``.
            angular_damping: Rigid-body angular velocity damping coefficient [1/s]. Defaults to 0.0.
            particle_damping: Particle velocity damping coefficient [1/s]. Defaults to 0.0.
            air_density: Density of the surrounding air [kg/m^3]. ``0.0`` (default) disables the
                aerodynamic model entirely. Use ``1.225`` for air at sea level.
            air_drag_coefficient: Flat-plate drag coefficient applied to each cloth triangle's
                projected frontal area [dimensionless]. Defaults to 1.0.
            air_lift_coefficient: Flat-plate lift coefficient [dimensionless]. Defaults to 0.0,
                which yields pure drag.
            air_velocity: Ambient air velocity in world frame [m/s], i.e. wind. Cloth experiences
                drag from its motion *relative* to this velocity. Defaults to ``(0.0, 0.0, 0.0)``.
            enable_restitution: Whether to apply restitution to rigid and particle-shape contacts after the
                positional solve. Defaults to ``False``.
            deterministic: Opt-in determinism for this solver's atomic-emitting
                kernel module. Pass a :class:`warp.DeterministicMode`, or
                ``None`` (default) to inherit the current
                ``wp.config.deterministic`` mode.
        """
        super().__init__(model=model)
        effective_deterministic = deterministic if deterministic is not None else wp.config.deterministic
        self._set_module_options(
            {
                "deterministic": effective_deterministic,
                "deterministic_max_records": 0,
            },
            module=kernels,
        )

        self.iterations = iterations
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation
        self.soft_contact_max_depenetration_velocity = soft_contact_max_depenetration_velocity

        if particle_enable_triangle_intersection_recovery and not particle_enable_self_contact:
            raise ValueError("Triangle-intersection recovery requires particle_enable_self_contact=True")
        if particle_damping < 0.0:
            raise ValueError("particle_damping must be non-negative")

        self.particle_enable_self_contact = particle_enable_self_contact
        self.particle_enable_triangle_intersection_recovery = particle_enable_triangle_intersection_recovery
        self.particle_conservative_bound_relaxation = 0.85
        self.particle_self_contact_relaxation = (
            soft_contact_relaxation if particle_self_contact_relaxation is None else particle_self_contact_relaxation
        )
        self.particle_triangle_intersection_relaxation = (
            self.particle_self_contact_relaxation
            if particle_triangle_intersection_relaxation is None
            else particle_triangle_intersection_relaxation
        )
        self.particle_self_contact_radius = (
            model.particle_max_radius if particle_self_contact_radius is None else particle_self_contact_radius
        )
        self.particle_self_contact_margin = (
            2.0 * self.particle_self_contact_radius
            if particle_self_contact_margin is None
            else particle_self_contact_margin
        )
        self.particle_max_depenetration_velocity = particle_max_depenetration_velocity
        self.particle_edge_parallel_epsilon = particle_edge_parallel_epsilon
        self.particle_rest_shape_contact_exclusion_radius = particle_rest_shape_contact_exclusion_radius
        self.particle_triangle_intersection_rest_shape_exclusion_radius = max(
            particle_rest_shape_contact_exclusion_radius,
            self.particle_self_contact_margin,
        )

        self.trimesh_collision_detector = None
        self.trimesh_collision_info = None
        self.vertex_triangle_contact_lambdas = None
        self.vertex_triangle_contact_split_lambdas = None
        self.vertex_triangle_contact_friction_lambdas = None
        self.edge_edge_contact_lambdas = None
        self.edge_edge_contact_split_lambdas = None
        self.edge_edge_contact_friction_lambdas = None
        self.self_contact_constraint_counts = None
        self.triangle_intersection_constraint_counts = None
        self.triangle_intersection_contact_normals = None
        self.triangle_intersection_contact_zero_normal_velocity = None
        self.particle_self_contact_displacements = None
        self.particle_self_contact_truncation_ts = None
        self.particle_self_contact_evaluation_kernel_launch_size = None
        if particle_enable_self_contact:
            if model.tri_count == 0:
                raise ValueError("Cloth self-contact requires at least one triangle")
            if self.particle_self_contact_radius <= 0.0:
                raise ValueError("particle_self_contact_radius must be positive")
            if self.particle_self_contact_margin < self.particle_self_contact_radius:
                raise ValueError("particle_self_contact_margin must be at least particle_self_contact_radius")
            self.trimesh_collision_detector = TriMeshCollisionDetector(
                model,
                vertex_collision_buffer_pre_alloc=particle_vertex_contact_buffer_size,
                edge_collision_buffer_pre_alloc=particle_edge_contact_buffer_size,
                triangle_triangle_collision_buffer_pre_alloc=particle_triangle_contact_buffer_size,
                edge_edge_parallel_epsilon=particle_edge_parallel_epsilon,
                topological_contact_filter_threshold=particle_topological_contact_filter_threshold,
            )
            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info],
                dtype=TriMeshCollisionInfo,
                device=model.device,
            )
            self.vertex_triangle_contact_lambdas = wp.zeros(
                self.trimesh_collision_detector.vertex_colliding_triangles.size // 2,
                dtype=float,
                device=model.device,
            )
            self.vertex_triangle_contact_split_lambdas = wp.zeros_like(self.vertex_triangle_contact_lambdas)
            self.vertex_triangle_contact_friction_lambdas = wp.zeros_like(self.vertex_triangle_contact_lambdas)
            self.edge_edge_contact_lambdas = wp.zeros(
                self.trimesh_collision_detector.edge_colliding_edges.size // 2,
                dtype=float,
                device=model.device,
            )
            self.edge_edge_contact_split_lambdas = wp.zeros_like(self.edge_edge_contact_lambdas)
            self.edge_edge_contact_friction_lambdas = wp.zeros_like(self.edge_edge_contact_lambdas)
            self.self_contact_constraint_counts = wp.zeros(model.particle_count, dtype=wp.int32, device=model.device)
            if particle_enable_triangle_intersection_recovery:
                # Allocate exact intersection buffers eagerly so recovery remains graph-capturable.
                self.trimesh_collision_detector.triangle_triangle_intersection_detection()
                self.triangle_intersection_constraint_counts = wp.zeros(
                    model.particle_count, dtype=wp.int32, device=model.device
                )
                self.triangle_intersection_contact_normals = wp.zeros(
                    self.trimesh_collision_detector.triangle_intersecting_triangles.shape,
                    dtype=wp.vec3,
                    device=model.device,
                )
                self.triangle_intersection_contact_zero_normal_velocity = wp.zeros(
                    self.trimesh_collision_detector.triangle_intersecting_triangles.shape,
                    dtype=wp.int32,
                    device=model.device,
                )
                self.particle_self_contact_displacements = wp.empty_like(model.particle_q)
                self.particle_self_contact_truncation_ts = wp.ones(
                    model.particle_count, dtype=float, device=model.device
                )
                self.particle_self_contact_evaluation_kernel_launch_size = max(
                    model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                    model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                )

        self.joint_linear_relaxation = joint_linear_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation
        self.joint_linear_compliance = joint_linear_compliance
        self.joint_angular_compliance = joint_angular_compliance

        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting

        self.angular_damping = angular_damping
        self.particle_damping = particle_damping

        if air_density < 0.0:
            raise ValueError("air_density must be non-negative")
        self.air_density = air_density
        self.air_drag_coefficient = air_drag_coefficient
        self.air_lift_coefficient = air_lift_coefficient
        self.air_velocity = wp.vec3(*air_velocity)
        # Aerodynamic force is accumulated into a scratch copy so the caller's
        # State.particle_f keeps holding only the forces the caller applied.
        self.particle_f_aero = wp.empty_like(model.particle_qd) if (model.particle_count and model.tri_count) else None

        self.enable_restitution = enable_restitution

        self.compute_body_velocity_from_position_delta = False

        self._init_kinematic_state()

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        self.particle_q_init = wp.empty_like(model.particle_q) if model.particle_count else None
        self.particle_qd_init = wp.empty_like(model.particle_qd) if model.particle_count else None
        self.particle_deltas = wp.empty_like(model.particle_q) if model.particle_count else None
        self.particle_split_deltas = wp.empty_like(model.particle_q) if model.particle_count else None
        self.particle_velocity_deltas = wp.empty_like(model.particle_qd) if model.particle_count else None
        self.particle_static_friction_limits = (
            wp.empty(model.particle_count, dtype=wp.vec4, device=model.device) if model.particle_count else None
        )
        self.body_q_init = wp.empty_like(model.body_q) if model.body_count else None
        self.body_qd_init = wp.empty_like(model.body_qd) if model.body_count else None
        self.body_deltas = wp.empty_like(model.body_qd) if model.body_count else None
        self.body_split_velocity_deltas = wp.empty_like(model.body_qd) if model.body_count else None
        self.body_velocity_deltas = wp.empty_like(model.body_qd) if model.body_count else None
        self.soft_contact_lambdas = None
        self.soft_contact_split_lambdas = None
        self.soft_contact_friction_lambdas = None
        self.soft_contact_lambda_capacity = 0
        self.soft_contact_body_counts = (
            wp.zeros(model.body_count, dtype=int, device=model.device) if model.body_count else None
        )
        self.soft_contact_particle_counts = (
            wp.zeros(model.particle_count, dtype=int, device=model.device) if model.particle_count else None
        )
        self.spring_constraint_lambdas = wp.zeros_like(model.spring_rest_length) if model.spring_count else None
        self.edge_constraint_lambdas = wp.zeros_like(model.edge_rest_angle) if model.edge_count else None
        self.particle_bending_constraint_counts = None
        if model.edge_count:
            self.particle_bending_constraint_counts = wp.zeros(
                model.particle_count, dtype=wp.int32, device=model.device
            )
            wp.launch(
                kernel=count_bending_constraints,
                dim=model.edge_count,
                inputs=[model.edge_indices],
                outputs=[self.particle_bending_constraint_counts],
                device=model.device,
            )
        self.triangle_constraint_lambdas = (
            wp.zeros(shape=(model.tri_count, 3), dtype=float, device=model.device) if model.tri_count else None
        )
        self.vertex_triangle_adjacency_offsets = None
        if model.tri_count:
            if model.soft_mesh_adjacency is None:
                raise ValueError("Triangle constraints require soft-mesh adjacency data")
            adjacency = model.soft_mesh_adjacency.init_vertex_adjacency(model.particle_count)
            adjacency_offsets = adjacency.v_adj_tris_offsets
            if hasattr(adjacency_offsets, "to"):
                self.vertex_triangle_adjacency_offsets = adjacency_offsets.to(model.device)
            else:
                self.vertex_triangle_adjacency_offsets = wp.array(
                    adjacency_offsets, dtype=wp.int32, device=model.device
                )

        # Marks vertices belonging to a triangle mesh so the particle-particle
        # fallback can tell surface neighbors from free colliding particles.
        self.particle_in_mesh = None
        if model.particle_count:
            in_mesh = np.zeros(model.particle_count, dtype=np.int32)
            if model.tri_count:
                in_mesh[np.unique(model.tri_indices.numpy())] = 1
            self.particle_in_mesh = wp.array(in_mesh, dtype=wp.int32, device=model.device)

        if model.particle_count > 1 and model.particle_grid is not None:
            # reserve space for the particle hash grid
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

    def rebuild_bvh(self, state: State) -> None:
        """Rebuild the cloth self-contact acceleration structures from a state.

        Args:
            state: State providing particle positions for the rebuilt BVHs.
        """
        if self.trimesh_collision_detector is not None:
            self.trimesh_collision_detector.rebuild(state.particle_q)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh cached body data after model properties change.

        Effective inverse masses and inertia tensors are refreshed when
        :attr:`~newton.ModelFlags.BODY_PROPERTIES` or
        :attr:`~newton.ModelFlags.BODY_INERTIAL_PROPERTIES` is set. Other flags are ignored.

        Args:
            flags: Bitmask of :class:`~newton.ModelFlags` or custom ``int`` bits indicating which model properties
                changed.
        """
        self._apply_module_options()
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()

    @override
    def coupling_supports_inertial_property_refresh(self) -> bool:
        """Return whether inertial properties can be refreshed during graph capture.

        Returns:
            ``True`` because :meth:`notify_model_changed` refreshes the derived inertial buffers with device work.
        """
        return True

    @override
    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        """Return whether XPBD consumes edge and face soft contacts."""
        return True

    def copy_kinematic_body_state(self, model: Model, state_in: State, state_out: State):
        """Copy kinematic body poses and velocities from an input state to an output state.

        Args:
            model: Simulation model that owns the body data.
            state_in: State containing the source kinematic body poses and velocities.
            state_out: State that receives the kinematic body poses and velocities.
        """
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
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation state by one time step using XPBD.

        Args:
            state_in: State at the beginning of the time step.
            state_out: State that receives the simulation result.
            control: Control inputs. If ``None``, the model's default control values are used.
            contacts: Contact data populated by :meth:`~newton.CollisionPipeline.collide` and allocated with
                :meth:`~newton.CollisionPipeline.contacts`. If ``None``, rigid and particle-shape contact handling
                is skipped; particle-particle contacts and model constraints are still solved.
            dt: Time step size [s].
        """
        self._apply_module_options()
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None
        particle_split_deltas = None

        body_q = None
        body_qd = None
        body_q_init = None
        body_qd_init = None
        body_deltas = None
        body_split_velocity_deltas = None

        rigid_contact_inv_weight = None

        contact_impulse = None
        contact_impulse_iter = None
        soft_contact_lambdas = None
        soft_contact_split_lambdas = None
        soft_contact_friction_lambdas = None

        if contacts:
            if self.rigid_contact_con_weighting and not self.integrate_with_external_rigid_solver:
                rigid_contact_inv_weight = wp.zeros(model.body_count, dtype=float, device=model.device)
            rigid_contact_inv_weight_init = None

            if contacts.force is not None and not self.integrate_with_external_rigid_solver:
                contact_impulse = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device)
                contact_impulse_iter = wp.zeros(
                    contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device
                )
            if model.particle_count and contacts.soft_contact_max > 0:
                if requires_grad:
                    soft_contact_lambdas = wp.zeros(
                        contacts.soft_contact_max,
                        dtype=float,
                        device=model.device,
                        requires_grad=True,
                    )
                    soft_contact_split_lambdas = wp.zeros_like(soft_contact_lambdas)
                    soft_contact_friction_lambdas = wp.zeros_like(soft_contact_lambdas)
                else:
                    if self.soft_contact_lambda_capacity != contacts.soft_contact_max:
                        self.soft_contact_lambdas = wp.zeros(
                            contacts.soft_contact_max,
                            dtype=float,
                            device=model.device,
                        )
                        self.soft_contact_split_lambdas = wp.zeros_like(self.soft_contact_lambdas)
                        self.soft_contact_friction_lambdas = wp.zeros_like(self.soft_contact_lambdas)
                        self.soft_contact_lambda_capacity = contacts.soft_contact_max
                    else:
                        self.soft_contact_lambdas.zero_()
                        self.soft_contact_split_lambdas.zero_()
                        self.soft_contact_friction_lambdas.zero_()
                    soft_contact_lambdas = self.soft_contact_lambdas
                    soft_contact_split_lambdas = self.soft_contact_split_lambdas
                    soft_contact_friction_lambdas = self.soft_contact_friction_lambdas

                self.soft_contact_particle_counts.zero_()
                if model.body_count:
                    self.soft_contact_body_counts.zero_()

        stabilize_particle_contacts = self.trimesh_collision_detector is not None or soft_contact_lambdas is not None

        # Optional per-joint accumulated child-side spatial impulse, used to
        # populate ``state_out.body_parent_f`` after the iteration loop.
        joint_impulse = None
        if (
            state_out.body_parent_f is not None
            and model.joint_count > 0
            and not self.integrate_with_external_rigid_solver
        ):
            joint_impulse = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd

                if requires_grad:
                    self.particle_q_init = wp.clone(state_in.particle_q)
                    particle_deltas = wp.empty_like(state_out.particle_qd)
                else:
                    self.particle_q_init.assign(state_in.particle_q)
                    particle_deltas = self.particle_deltas
                if stabilize_particle_contacts:
                    if requires_grad:
                        self.particle_qd_init = wp.clone(state_in.particle_qd)
                        particle_split_deltas = wp.zeros_like(state_out.particle_q)
                    else:
                        self.particle_qd_init.assign(state_in.particle_qd)
                        particle_split_deltas = self.particle_split_deltas
                        particle_split_deltas.zero_()

                if self.particle_enable_triangle_intersection_recovery:
                    # Use unfiltered nearby features only to build conservative
                    # trajectory bounds. Ordinary contact still uses the
                    # configured topological filter below.
                    self.trimesh_collision_detector.refit(state_in.particle_q)
                    self.trimesh_collision_detector.vertex_triangle_collision_detection(
                        self.particle_self_contact_margin,
                        use_topological_filter=False,
                    )
                    self.trimesh_collision_detector.edge_edge_collision_detection(
                        self.particle_self_contact_margin,
                        use_topological_filter=False,
                    )

                if self.air_density > 0.0 and model.tri_count:
                    self.particle_f_aero.assign(state_in.particle_f)
                    wp.launch(
                        kernel=eval_triangle_aerodynamics,
                        dim=model.tri_count,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_inv_mass,
                            model.particle_flags,
                            model.tri_indices,
                            self.air_velocity,
                            self.air_density,
                            self.air_drag_coefficient,
                            self.air_lift_coefficient,
                            dt,
                        ],
                        outputs=[self.particle_f_aero],
                        device=model.device,
                    )
                    particle_f_prev = state_in.particle_f
                    state_in.particle_f = self.particle_f_aero
                    self.integrate_particles(model, state_in, state_out, dt)
                    state_in.particle_f = particle_f_prev
                else:
                    self.integrate_particles(model, state_in, state_out, dt)
                if self.particle_enable_triangle_intersection_recovery:
                    wp.launch(
                        kernel=compute_particle_displacements,
                        dim=model.particle_count,
                        inputs=[
                            state_in.particle_q,
                            state_out.particle_q,
                        ],
                        outputs=[self.particle_self_contact_displacements],
                        device=model.device,
                    )
                    self.particle_self_contact_truncation_ts.fill_(1.0)
                    wp.launch(
                        kernel=apply_planar_truncation_parallel_by_collision,
                        dim=self.particle_self_contact_evaluation_kernel_launch_size,
                        inputs=[
                            state_in.particle_q,
                            self.particle_self_contact_displacements,
                            model.tri_indices,
                            model.edge_indices,
                            self.trimesh_collision_info,
                            self.particle_edge_parallel_epsilon,
                            self.particle_conservative_bound_relaxation,
                        ],
                        outputs=[self.particle_self_contact_truncation_ts],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=apply_truncation_ts,
                        dim=model.particle_count,
                        inputs=[
                            state_in.particle_q,
                            self.particle_self_contact_displacements,
                            self.particle_self_contact_truncation_ts,
                            0.5 * self.particle_conservative_bound_relaxation * self.particle_self_contact_margin,
                        ],
                        outputs=[self.particle_self_contact_displacements, None],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=apply_particle_deltas,
                        dim=model.particle_count,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_q,
                            model.particle_flags,
                            self.particle_self_contact_displacements,
                            dt,
                            model.particle_max_velocity,
                        ],
                        outputs=[state_out.particle_q, state_out.particle_qd],
                        device=model.device,
                    )
                if self.particle_damping > 0.0:
                    wp.launch(
                        kernel=damp_particle_velocities,
                        dim=model.particle_count,
                        inputs=[
                            state_in.particle_q,
                            model.particle_flags,
                            self.particle_damping,
                            dt,
                        ],
                        outputs=[state_out.particle_q, state_out.particle_qd],
                        device=model.device,
                    )

                # Build/update the particle hash grid for particle-particle contact queries
                if (
                    model.particle_count > 1
                    and model.particle_grid is not None
                    and self.trimesh_collision_detector is None
                ):
                    # Search radius must cover the maximum interaction distance used by the contact query
                    search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                    with wp.ScopedDevice(model.device):
                        model.particle_grid.build(state_out.particle_q, radius=search_radius)

                if self.trimesh_collision_detector is not None:
                    self.trimesh_collision_detector.refit(state_out.particle_q)
                    self.trimesh_collision_detector.vertex_triangle_collision_detection(
                        self.particle_self_contact_margin,
                        min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                        min_distance_filtering_ref_pos=model.particle_q,
                    )
                    self.trimesh_collision_detector.edge_edge_collision_detection(
                        self.particle_self_contact_margin,
                        min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                        min_distance_filtering_ref_pos=model.particle_q,
                    )
                    self.vertex_triangle_contact_lambdas.zero_()
                    self.vertex_triangle_contact_split_lambdas.zero_()
                    self.vertex_triangle_contact_friction_lambdas.zero_()
                    self.edge_edge_contact_lambdas.zero_()
                    self.edge_edge_contact_split_lambdas.zero_()
                    self.edge_edge_contact_friction_lambdas.zero_()
                    self.self_contact_constraint_counts.zero_()
                    wp.launch(
                        kernel=count_vertex_triangle_self_contacts,
                        dim=model.particle_count,
                        inputs=[
                            state_out.particle_q,
                            model.tri_indices,
                            self.trimesh_collision_info,
                            self.particle_self_contact_radius,
                        ],
                        outputs=[self.self_contact_constraint_counts],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=count_edge_edge_self_contacts,
                        dim=model.edge_count,
                        inputs=[
                            state_out.particle_q,
                            model.edge_indices,
                            self.trimesh_collision_info,
                            self.particle_self_contact_radius,
                            self.particle_edge_parallel_epsilon,
                        ],
                        outputs=[self.self_contact_constraint_counts],
                        device=model.device,
                    )

            if model.body_count and (
                soft_contact_lambdas is not None
                or self.compute_body_velocity_from_position_delta
                or self.enable_restitution
            ):
                if requires_grad:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)
                else:
                    self.body_q_init.assign(state_in.body_q)
                    self.body_qd_init.assign(state_in.body_qd)
                    body_q_init = self.body_q_init
                    body_qd_init = self.body_qd_init
                if soft_contact_lambdas is not None:
                    if requires_grad:
                        body_split_velocity_deltas = wp.zeros_like(state_out.body_qd)
                    else:
                        body_split_velocity_deltas = self.body_split_velocity_deltas
                        body_split_velocity_deltas.zero_()

            if model.body_count and self.integrate_with_external_rigid_solver:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                body_deltas = self.body_deltas

            if model.body_count and not self.integrate_with_external_rigid_solver:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                body_deltas = wp.empty_like(state_out.body_qd) if requires_grad else self.body_deltas

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

            if soft_contact_lambdas is not None:
                wp.launch(
                    kernel=count_particle_shape_contact_bodies,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        state_out.particle_q,
                        model.particle_radius,
                        model.particle_flags,
                        state_out.body_q,
                        self.body_inv_mass_effective,
                        model.body_flags,
                        model.shape_body,
                        model.shape_material_ka,
                        model.shape_margin,
                        model.particle_adhesion,
                        contacts.soft_contact_count,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_barycentric,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_max,
                    ],
                    outputs=[self.soft_contact_particle_counts, self.soft_contact_body_counts],
                    device=model.device,
                )

            spring_constraint_lambdas = None
            if model.spring_count:
                if requires_grad:
                    spring_constraint_lambdas = wp.zeros_like(model.spring_rest_length)
                else:
                    spring_constraint_lambdas = self.spring_constraint_lambdas
                    spring_constraint_lambdas.zero_()
            edge_constraint_lambdas = None
            if model.edge_count:
                if requires_grad:
                    edge_constraint_lambdas = wp.zeros_like(model.edge_rest_angle)
                else:
                    edge_constraint_lambdas = self.edge_constraint_lambdas
                    edge_constraint_lambdas.zero_()
            triangle_constraint_lambdas = None
            if model.tri_count:
                if requires_grad:
                    triangle_constraint_lambdas = wp.zeros(
                        shape=(model.tri_count, 3),
                        dtype=float,
                        device=model.device,
                        requires_grad=True,
                    )
                else:
                    triangle_constraint_lambdas = self.triangle_constraint_lambdas
                    triangle_constraint_lambdas.zero_()

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
                        if model.shape_count and contacts is not None and soft_contact_lambdas is not None:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    self.particle_q_init,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_q_init,
                                    body_qd,
                                    model.body_com,
                                    self.body_inv_mass_effective,
                                    self.body_inv_inertia_effective,
                                    model.body_flags,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.shape_margin,
                                    model.shape_material_ka,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_indices,
                                    contacts.soft_contact_barycentric,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    self.soft_contact_particle_counts,
                                    self.soft_contact_body_counts,
                                    self.integrate_with_external_rigid_solver,
                                    (
                                        -1.0
                                        if self.soft_contact_max_depenetration_velocity is None
                                        else self.soft_contact_max_depenetration_velocity
                                    ),
                                    dt,
                                    self.soft_contact_relaxation,
                                    soft_contact_lambdas,
                                    soft_contact_split_lambdas,
                                    soft_contact_friction_lambdas,
                                ],
                                # outputs
                                outputs=[
                                    particle_deltas,
                                    body_deltas,
                                    particle_split_deltas,
                                    body_split_velocity_deltas,
                                ],
                                device=model.device,
                            )

                        if (
                            model.particle_max_radius > 0.0
                            and model.particle_count > 1
                            and self.trimesh_collision_detector is None
                        ):
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            assert model.particle_grid is not None
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    model.particle_q,
                                    self.particle_in_mesh,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        if self.trimesh_collision_detector is not None:
                            if self.particle_max_depenetration_velocity is None:
                                max_depenetration_velocity = 0.25 * self.particle_self_contact_radius / dt
                            else:
                                max_depenetration_velocity = self.particle_max_depenetration_velocity

                            wp.launch(
                                kernel=solve_vertex_triangle_self_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    particle_q,
                                    self.particle_q_init,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tri_indices,
                                    self.trimesh_collision_info,
                                    self.particle_self_contact_radius,
                                    model.particle_mu,
                                    max_depenetration_velocity,
                                    dt,
                                    self.particle_self_contact_relaxation,
                                    self.self_contact_constraint_counts,
                                    self.vertex_triangle_contact_lambdas,
                                    self.vertex_triangle_contact_split_lambdas,
                                    self.vertex_triangle_contact_friction_lambdas,
                                ],
                                outputs=[particle_deltas, particle_split_deltas],
                                device=model.device,
                            )
                            wp.launch(
                                kernel=solve_edge_edge_self_contacts,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    self.particle_q_init,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    self.trimesh_collision_info,
                                    self.particle_self_contact_radius,
                                    model.particle_mu,
                                    max_depenetration_velocity,
                                    self.particle_edge_parallel_epsilon,
                                    dt,
                                    self.particle_self_contact_relaxation,
                                    self.self_contact_constraint_counts,
                                    self.edge_edge_contact_lambdas,
                                    self.edge_edge_contact_split_lambdas,
                                    self.edge_edge_contact_friction_lambdas,
                                ],
                                outputs=[particle_deltas, particle_split_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
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
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_rest_length,
                                    model.edge_bending_properties,
                                    self.particle_bending_constraint_counts,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # triangle membrane constraints
                        if model.tri_count:
                            wp.launch(
                                kernel=solve_triangles,
                                dim=model.tri_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tri_indices,
                                    model.tri_poses,
                                    model.tri_areas,
                                    model.tri_materials,
                                    self.vertex_triangle_adjacency_offsets,
                                    dt,
                                    self.soft_body_relaxation,
                                    triangle_constraint_lambdas,
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
                        if self.particle_enable_triangle_intersection_recovery:
                            # Regular XPBD constraints can create intersections
                            # after the initial predicted-position query.
                            if i >= self.iterations - 2:
                                self.trimesh_collision_detector.refit(particle_q)
                                self.trimesh_collision_detector.triangle_triangle_intersection_detection()
                            self.triangle_intersection_contact_normals.zero_()
                            self.triangle_intersection_contact_zero_normal_velocity.zero_()
                            self.triangle_intersection_constraint_counts.zero_()
                            wp.launch(
                                kernel=count_triangle_intersection_self_contacts,
                                dim=model.tri_count,
                                inputs=[
                                    particle_q,
                                    model.particle_world,
                                    model.tri_indices,
                                    self.trimesh_collision_detector.triangle_intersecting_triangles,
                                    self.trimesh_collision_detector.triangle_intersecting_triangles_count,
                                    self.trimesh_collision_detector.triangle_intersecting_triangles_offsets,
                                ],
                                outputs=[self.triangle_intersection_constraint_counts],
                                device=model.device,
                            )
                            if requires_grad:
                                particle_deltas = wp.zeros_like(particle_deltas)
                            else:
                                particle_deltas.zero_()
                            wp.launch(
                                kernel=solve_triangle_intersection_self_contacts,
                                dim=model.tri_count,
                                inputs=[
                                    particle_q,
                                    self.particle_q_init,
                                    model.particle_q,
                                    model.particle_inv_mass,
                                    model.particle_world,
                                    model.tri_indices,
                                    self.trimesh_collision_detector.triangle_intersecting_triangles,
                                    self.trimesh_collision_detector.triangle_intersecting_triangles_count,
                                    self.trimesh_collision_detector.triangle_intersecting_triangles_offsets,
                                    self.particle_self_contact_radius,
                                    self.particle_triangle_intersection_rest_shape_exclusion_radius,
                                    max_depenetration_velocity,
                                    dt,
                                    self.particle_triangle_intersection_relaxation,
                                    i == self.iterations - 1,
                                    self.triangle_intersection_constraint_counts,
                                ],
                                outputs=[
                                    self.triangle_intersection_contact_normals,
                                    self.triangle_intersection_contact_zero_normal_velocity,
                                    particle_deltas,
                                    particle_split_deltas,
                                ],
                                device=model.device,
                            )
                            particle_q, particle_qd = self._apply_particle_deltas(
                                model, state_in, state_out, particle_deltas, dt
                            )

                    # handle rigid bodies
                    # ----------------------------

                    # Solve rigid contact constraints
                    if model.body_count and contacts is not None and not self.integrate_with_external_rigid_solver:
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

                    if model.joint_count and not self.integrate_with_external_rigid_solver:
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

            self._contact_impulse = contact_impulse
            self._contact_impulse_capacity = contacts.rigid_contact_max if contacts is not None else 0
            self._last_dt = dt

            # Populate optional ``state_out.body_parent_f`` (incoming joint
            # wrench per body) from the per-joint accumulated child-side
            # impulse.  Bodies without an inbound joint (roots / free bodies)
            # remain zero-initialized, matching MuJoCo's behavior.
            if state_out.body_parent_f is not None and not self.integrate_with_external_rigid_solver:
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
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count and not self.integrate_with_external_rigid_solver:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if (
                self.compute_body_velocity_from_position_delta
                and model.body_count
                and not requires_grad
                and not self.integrate_with_external_rigid_solver
            ):
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

            if particle_split_deltas is not None:
                wp.launch(
                    kernel=remove_particle_split_velocity,
                    dim=model.particle_count,
                    inputs=[model.particle_flags, self.particle_qd_init, particle_split_deltas, dt],
                    outputs=[state_out.particle_qd],
                    device=model.device,
                )
            if body_split_velocity_deltas is not None and not self.integrate_with_external_rigid_solver:
                wp.launch(
                    kernel=remove_body_split_velocity,
                    dim=model.body_count,
                    inputs=[body_qd_init, body_split_velocity_deltas],
                    outputs=[state_out.body_qd],
                    device=model.device,
                )

            # Split depenetration removes only pre-existing overlap recovery.
            # Contact still needs a final impulse pass for inelastic
            # self-contact stabilization, restitution, moving-surface
            # friction, and rigid-body feedback.
            if model.particle_count and (
                soft_contact_lambdas is not None or self.trimesh_collision_detector is not None
            ):
                if requires_grad:
                    particle_velocity_deltas = wp.zeros_like(state_out.particle_qd)
                    particle_static_friction_limits = wp.zeros(
                        model.particle_count,
                        dtype=wp.vec4,
                        device=model.device,
                    )
                else:
                    particle_velocity_deltas = self.particle_velocity_deltas
                    particle_velocity_deltas.zero_()
                    particle_static_friction_limits = self.particle_static_friction_limits
                    particle_static_friction_limits.zero_()

                body_velocity_deltas = self.body_velocity_deltas
                if body_velocity_deltas is not None:
                    if requires_grad:
                        body_velocity_deltas = wp.zeros_like(state_out.body_qd)
                    else:
                        body_velocity_deltas.zero_()

                if contacts is not None and soft_contact_lambdas is not None:
                    wp.launch(
                        kernel=solve_particle_shape_contact_velocities,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            state_out.particle_q,
                            self.particle_q_init,
                            state_out.particle_qd,
                            self.particle_qd_init,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            state_out.body_q,
                            body_q_init,
                            state_out.body_qd,
                            model.body_com,
                            self.body_inv_mass_effective,
                            self.body_inv_inertia_effective,
                            model.body_flags,
                            model.shape_body,
                            model.shape_material_mu,
                            model.shape_margin,
                            model.soft_contact_mu,
                            model.soft_contact_restitution if self.enable_restitution else 0.0,
                            contacts.soft_contact_count,
                            contacts.soft_contact_indices,
                            contacts.soft_contact_barycentric,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                            self.soft_contact_particle_counts,
                            self.soft_contact_body_counts,
                            self.integrate_with_external_rigid_solver,
                            dt,
                            1.0,
                            soft_contact_lambdas,
                            soft_contact_friction_lambdas,
                        ],
                        outputs=[
                            particle_velocity_deltas,
                            body_velocity_deltas,
                            particle_static_friction_limits,
                        ],
                        device=model.device,
                    )

                if self.trimesh_collision_detector is not None:
                    wp.launch(
                        kernel=solve_vertex_triangle_self_contact_velocities,
                        dim=model.particle_count,
                        inputs=[
                            state_out.particle_q,
                            state_out.particle_qd,
                            model.particle_inv_mass,
                            model.tri_indices,
                            self.trimesh_collision_info,
                            model.particle_mu,
                            dt,
                            1.0,
                            self.self_contact_constraint_counts,
                            self.vertex_triangle_contact_lambdas,
                            self.vertex_triangle_contact_friction_lambdas,
                        ],
                        outputs=[particle_velocity_deltas],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=solve_edge_edge_self_contact_velocities,
                        dim=model.edge_count,
                        inputs=[
                            state_out.particle_q,
                            state_out.particle_qd,
                            model.particle_inv_mass,
                            model.edge_indices,
                            self.trimesh_collision_info,
                            model.particle_mu,
                            self.particle_edge_parallel_epsilon,
                            dt,
                            1.0,
                            self.self_contact_constraint_counts,
                            self.edge_edge_contact_lambdas,
                            self.edge_edge_contact_friction_lambdas,
                        ],
                        outputs=[particle_velocity_deltas],
                        device=model.device,
                    )
                    if self.particle_enable_triangle_intersection_recovery:
                        wp.launch(
                            kernel=solve_triangle_intersection_self_contact_velocities,
                            dim=model.tri_count,
                            inputs=[
                                state_out.particle_qd,
                                model.particle_inv_mass,
                                model.tri_indices,
                                self.trimesh_collision_detector.triangle_intersecting_triangles,
                                self.trimesh_collision_detector.triangle_intersecting_triangles_count,
                                self.trimesh_collision_detector.triangle_intersecting_triangles_offsets,
                                self.triangle_intersection_constraint_counts,
                                self.triangle_intersection_contact_normals,
                                self.triangle_intersection_contact_zero_normal_velocity,
                            ],
                            outputs=[particle_velocity_deltas],
                            device=model.device,
                        )

                wp.launch(
                    kernel=apply_particle_velocity_deltas,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_flags,
                        particle_velocity_deltas,
                        particle_static_friction_limits,
                        model.particle_max_velocity,
                    ],
                    outputs=[state_out.particle_qd],
                    device=model.device,
                )
                if body_velocity_deltas is not None and not self.integrate_with_external_rigid_solver:
                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[body_velocity_deltas],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            if self.enable_restitution and contacts is not None:
                if model.body_count and not self.integrate_with_external_rigid_solver:
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

            if model.body_count and not self.integrate_with_external_rigid_solver:
                self.copy_kinematic_body_state(model, state_in, state_out)

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
        self._apply_module_options()
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
