# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import IntEnum

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from ..style3d.builder import PDMatrixBuilder
from ..style3d.kernels import deactivate_zero_mass_particles_kernel
from ..style3d.linear_solver import PcgSolver, SparseMatrixELL
from ..style3d.solver_style3d import SolverStyle3D
from . import kernels
from .self_contact import SelfContact, invert_blocks


@dataclass
class _Diagnostics:
    """Device-resident diagnostics for one IPC solve."""

    status: wp.array
    residual: wp.array
    minimum_gap: wp.array
    newton_iterations: wp.array
    line_search_iterations: wp.array
    failed_steps: wp.array


class _TiledPcgSolver(PcgSolver):
    """PCG variant whose prepared tile reductions are valid in conditional graphs."""

    def __init__(self, dim: int, device, max_iterations: int):
        super().__init__(dim, device, maxIter=max_iterations)
        self.block_dim = 256 if wp.get_device(device).is_cuda else 1
        self.block_count = (dim + self.block_dim - 1) // self.block_dim

    def solve(self, *args, **kwargs):
        self.rTz.zero_()
        self.pTAp.zero_()
        super().solve(*args, **kwargs)

    def _array_inner(self, a: wp.array, b: wp.array, output: wp.array, output_index: int) -> None:
        wp.launch(
            kernels.array_inner_tiled,
            dim=(self.block_count, self.block_dim),
            block_dim=self.block_dim,
            inputs=[a, b, self.dim, output_index],
            outputs=[output],
            device=self.device,
        )

    def step3_update_rTz(self, iter: int):
        self._array_inner(self.r, self.z, self.rTz, iter)

    def step6_update_pTAp(self, iter: int):
        self._array_inner(self.p, self.Ap, self.pTAp, iter)


class SolverIPC(SolverBase):
    r"""Experimental incremental-potential-contact solver for cloth and particles.

    The solver minimizes a backward-Euler objective for Style3D cloth with the
    normalized logarithmic IPC barrier, a fixed plane, and finite-thickness
    triangle-surface self-contact. Swept BVH queries and conservative advancement
    bound every search step; Armijo search evaluates the actual barrier energy.

    This API is experimental and may change without the normal deprecation
    period. Friction, tetrahedra,
    dynamic rigid bodies, multiple worlds, and differentiation are not yet
    supported. The ``contacts`` argument to :meth:`step` must be ``None``;
    contact geometry is owned by this solver rather than Newton's reduced
    contact-point representation. CUDA graph capture is supported. CPU APIC
    execution is supported eagerly, but nested conditional capture is not yet
    supported because its replay does not currently match eager execution.

    Call :meth:`register_custom_attributes` before adding cloth, and use
    :func:`newton.solvers.style3d.add_cloth_grid` or
    :func:`newton.solvers.style3d.add_cloth_mesh` to populate the required
    anisotropic cloth data.

    The nonlinear solve uses a fixed sparse bending operator, state-dependent
    positive-semidefinite membrane blocks, and matrix-free barrier factors.
    Cloth uses assembled 3x3 block-Jacobi inverses; plane-only particle blocks
    retain the Sherman-Morrison fast path. Edge barriers are mollified near
    parallelism. Contact discretization remains mesh-dependent, and there is
    no strain-limiting energy. Initialize with nonintersecting surfaces whose
    nonincident primitive gaps exceed the configured thickness and guard.

    CUDA uses swept BVHs; CPU uses an exhaustive reference broad phase. CCD
    evaluates distances in float64 and reserves a scale-aware float32 rounding
    allowance. This is not an interval-arithmetic collision certificate.
    """

    class Status(IntEnum):
        """Terminal status of the most recent step."""

        NOT_RUN = 0
        """No step has run."""
        CONVERGED = 1
        """The objective converged and the output state was committed."""
        INVALID_INITIAL_STATE = 2
        """Initial plane separation, surface separation, or intersections are invalid."""
        LINEAR_BREAKDOWN = 3
        """The linear solve did not produce a finite descent direction."""
        LINE_SEARCH_EXHAUSTED = 4
        """No feasible Armijo step was found within the line-search budget."""
        NEWTON_EXHAUSTED = 5
        """The nonlinear residual did not converge within the iteration budget."""
        CONTACT_OVERFLOW = 6
        """A contact query exceeded prepared capacity; the step was rolled back."""

    @dataclass
    class Config:
        """Configuration for :class:`SolverIPC`.

        A floating-point configuration value is fixed when a CUDA graph is
        recorded. Recapture after changing scalar tolerances or the time step.
        Reconstruct after changing topology, self-contact enablement/capacity,
        or ``use_projective_hessian`` because these determine prepared storage.
        """

        plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
        """Unit normal of the fixed collision plane [dimensionless]."""
        plane_offset: float = 0.0
        """Plane offset in ``dot(plane_normal, x) = plane_offset`` [m]."""
        minimum_separation: float = 0.0
        """Minimum particle-plane surface separation [m]."""
        contact_distance: float = 0.02
        """Barrier activation distance above ``minimum_separation`` [m]."""
        barrier_stiffness: float = 1.0e-3
        """Normalized barrier energy scale per particle [J]."""
        max_newton_iterations: int = 20
        """Maximum nonlinear updates per step."""
        max_pcg_iterations: int = 32
        """Fixed PCG iterations per nonlinear update."""
        max_line_search_iterations: int = 16
        """Maximum Armijo backtracking trials per nonlinear update."""
        absolute_tolerance: float = 1.0e-4
        """Absolute force-residual tolerance [N]."""
        relative_tolerance: float = 1.0e-5
        """Residual tolerance relative to the first nonlinear iterate."""
        armijo: float = 1.0e-4
        """Armijo sufficient-decrease coefficient [dimensionless]."""
        energy_tolerance: float = 1.0e-8
        """Absolute float32 roundoff allowance in line-search energy [J]."""
        initial_step_size: float = 1.0
        """Initial Armijo trial multiplier for the stabilized search direction."""
        ccd_safety: float = 0.9
        """Fraction of the exact point-plane collision step bound to use."""
        velocity_damping: float = 0.998
        """Velocity multiplier applied after a converged step."""
        preconditioner: str = "rank_one"
        """Plane-only block inverse: ``"rank_one"`` or ``"dense"``; cloth uses full blocks."""
        graph_mode: str = "conditional"
        """Use nested device-conditional loops or a fixed unrolled schedule."""
        enable_self_contact: bool = True
        """Enable point-triangle and edge-edge barriers on triangle surfaces."""
        self_contact_thickness: float = 0.001
        """Minimum nonincident surface-primitive separation [m]."""
        self_contact_distance: float = 0.005
        """Self-contact barrier activation distance above thickness [m]."""
        self_contact_stiffness: float = 0.001
        """Normalized barrier energy scale per primitive stencil [J]."""
        self_contact_capacity: int = 0
        """Maximum stored pairs per query; zero selects a topology-sized estimate."""
        self_contact_guard: float = 1.0e-6
        """Additional spatial safety allowance for float32 trial rounding [m]."""
        self_contact_ccd_iterations: int = 64
        """Conservative advancement budget; exhaustion retains only a safe prefix."""
        use_projective_hessian: bool = False
        """Use the legacy fixed membrane metric instead of state-dependent PSD blocks."""

    def __init__(self, model: Model, *, config: Config | None = None):
        """Prepare fixed cloth topology and graph-stable solve storage.

        Args:
            model: Model containing particles and optional Style3D cloth.
            config: Solver and fixed-plane configuration.
        """
        super().__init__(model)
        self.config = config if config is not None else self.Config()
        self._validate_configuration()

        if model.world_count > 1:
            raise NotImplementedError("SolverIPC currently supports one local world or global particles only")
        if model.body_count > 0:
            raise NotImplementedError("SolverIPC currently supports fixed planes but not body state")
        if model.tri_count > 0 and not hasattr(model, "style3d"):
            raise AttributeError(
                "IPC cloth attributes are missing. Call SolverIPC.register_custom_attributes() "
                "and add cloth with newton.solvers.style3d helpers."
            )

        normal = wp.vec3(*self.config.plane_normal)
        self._plane_normal = wp.normalize(normal)
        self._particle_flags = wp.clone(model.particle_flags)
        self._refresh_particle_flags()

        self.pd_non_diagonals = SparseMatrixELL()
        self.pd_diagonal = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self._precompute_cloth_operator()

        count = model.particle_count
        self._linear_solver = _TiledPcgSolver(count, self.device, self.config.max_pcg_iterations)
        self._reduction_block_dim = 256 if self.device.is_cuda else 1
        self._reduction_block_count = (count + self._reduction_block_dim - 1) // self._reduction_block_dim
        self._x_previous = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self._x_predictor = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self._x_current = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self._x_candidate = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self._rhs = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self._direction = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self._static_diagonal = wp.zeros(count, dtype=float, device=self.device)
        self._contact_hessian = wp.zeros(count, dtype=float, device=self.device)
        self._inverse_diagonal = wp.zeros(count, dtype=wp.mat33, device=self.device)
        self._barrier_product = wp.zeros(count, dtype=wp.vec3, device=self.device)

        self._solve_active = wp.zeros(1, dtype=int, device=self.device)
        self._invalid = wp.zeros(1, dtype=int, device=self.device)
        self._accepted = wp.zeros(1, dtype=int, device=self.device)
        self._line_search_active = wp.zeros(1, dtype=int, device=self.device)
        self._line_search_current_iterations = wp.zeros(1, dtype=int, device=self.device)
        self._alpha = wp.ones(1, dtype=float, device=self.device)
        self._current_energy = wp.zeros(1, dtype=float, device=self.device)
        self._candidate_energy = wp.zeros(1, dtype=float, device=self.device)
        self._residual_squared = wp.zeros(1, dtype=float, device=self.device)
        self._reference_residual = wp.zeros(1, dtype=float, device=self.device)
        self._rhs_dot_direction = wp.zeros(1, dtype=float, device=self.device)

        self.diagnostics = _Diagnostics(
            status=wp.full(1, int(self.Status.NOT_RUN), dtype=int, device=self.device),
            residual=wp.zeros(1, dtype=float, device=self.device),
            minimum_gap=wp.zeros(1, dtype=float, device=self.device),
            newton_iterations=wp.zeros(1, dtype=int, device=self.device),
            line_search_iterations=wp.zeros(1, dtype=int, device=self.device),
            failed_steps=wp.zeros(1, dtype=int, device=self.device),
        )
        """Device arrays describing convergence, feasibility, and iteration counts."""
        self._self_contact = (
            SelfContact(model, self.config) if self.config.enable_self_contact and model.tri_count else None
        )
        self._metric_diagonal = (
            self._self_contact.diagonal
            if self._self_contact is not None
            else wp.zeros(count, dtype=wp.mat33, device=self.device)
        )

    def _validate_configuration(self) -> None:
        config = self.config
        for field in fields(config):
            value = getattr(config, field.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if not all(math.isfinite(value) for value in config.plane_normal):
            raise ValueError("plane_normal must be finite")
        if config.self_contact_thickness < 0.0 or config.self_contact_capacity < 0:
            raise ValueError("self-contact thickness and capacity must be nonnegative")
        if min(config.self_contact_distance, config.self_contact_stiffness, config.self_contact_guard) <= 0.0:
            raise ValueError("self-contact distance, stiffness, and guard must be positive")
        if config.self_contact_ccd_iterations < 1:
            raise ValueError("self_contact_ccd_iterations must be >= 1")
        normal_length = sum(component * component for component in config.plane_normal) ** 0.5
        if normal_length <= 0.0:
            raise ValueError("plane_normal must be nonzero")
        if config.contact_distance <= 0.0:
            raise ValueError("contact_distance must be > 0")
        if config.barrier_stiffness <= 0.0:
            raise ValueError("barrier_stiffness must be > 0")
        if config.max_newton_iterations < 1:
            raise ValueError("max_newton_iterations must be >= 1")
        if config.max_pcg_iterations < 1:
            raise ValueError("max_pcg_iterations must be >= 1")
        if config.max_line_search_iterations < 1:
            raise ValueError("max_line_search_iterations must be >= 1")
        if config.absolute_tolerance < 0.0 or config.relative_tolerance < 0.0:
            raise ValueError("residual tolerances must be nonnegative")
        if not 0.0 < config.armijo < 1.0:
            raise ValueError("armijo must be in (0, 1)")
        if config.energy_tolerance < 0.0:
            raise ValueError("energy_tolerance must be nonnegative")
        if config.initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be > 0")
        if not 0.0 < config.ccd_safety < 1.0:
            raise ValueError("ccd_safety must be in (0, 1)")
        if not 0.0 <= config.velocity_damping <= 1.0:
            raise ValueError("velocity_damping must be in [0, 1]")
        if config.preconditioner not in ("rank_one", "dense"):
            raise ValueError("preconditioner must be 'rank_one' or 'dense'")
        if config.graph_mode not in ("conditional", "unrolled"):
            raise ValueError("graph_mode must be 'conditional' or 'unrolled'")

    def _refresh_particle_flags(self) -> None:
        wp.copy(self._particle_flags, self.model.particle_flags)
        if self.model.particle_count > 0:
            wp.launch(
                deactivate_zero_mass_particles_kernel,
                dim=self.model.particle_count,
                inputs=[self.model.particle_mass],
                outputs=[self._particle_flags],
                device=self.device,
            )

    def _precompute_cloth_operator(self) -> None:
        builder = PDMatrixBuilder(self.model.particle_count)
        if self.model.tri_count > 0 and self.config.use_projective_hessian:
            builder.add_stretch_constraints(
                self.model.tri_indices.numpy().tolist(),
                self.model.tri_poses.numpy().tolist(),
                self.model.style3d.tri_aniso_ke.numpy().tolist(),
                self.model.tri_areas.numpy().tolist(),
            )
        if self.model.edge_count > 0:
            builder.add_bend_constraints(
                self.model.edge_indices.numpy().tolist(),
                self.model.edge_bending_properties.numpy().tolist(),
                self.model.style3d.edge_rest_area.numpy().tolist(),
                self.model.style3d.edge_bending_cot.numpy().tolist(),
            )
        self.pd_diagonal, self.pd_non_diagonals.num_nz, self.pd_non_diagonals.nz_ell = builder.finalize(self.device)

    def _add_elastic_forces(self) -> None:
        if self.model.tri_count > 0:
            wp.launch(
                kernels.add_triangle_forces,
                dim=self.model.tri_count,
                inputs=[
                    self._x_current,
                    self.model.tri_areas,
                    self.model.tri_poses,
                    self.model.tri_indices,
                    self.model.style3d.tri_aniso_ke,
                    self._solve_active,
                ],
                outputs=[self._rhs],
                device=self.device,
            )
        if self.model.edge_count > 0:
            wp.launch(
                kernels.add_bending_forces,
                dim=self.model.edge_count,
                inputs=[
                    self._x_current,
                    self.model.style3d.edge_rest_area,
                    self.model.style3d.edge_bending_cot,
                    self.model.edge_indices,
                    self.model.edge_bending_properties,
                    self._solve_active,
                ],
                outputs=[self._rhs],
                device=self.device,
            )

    def _accumulate_energy(self, x: wp.array, energy: wp.array, dt: float, *, swept: bool = False) -> None:
        if self._self_contact is not None:
            self._self_contact.energy(x, energy, self._invalid, swept=swept)
        wp.launch(
            kernels.add_particle_energy,
            dim=self.model.particle_count,
            inputs=[
                dt,
                x,
                self._x_predictor,
                self.model.particle_mass,
                self._particle_flags,
                self._plane_normal,
                self.config.plane_offset,
                self.config.minimum_separation,
                self.config.contact_distance,
                self.config.barrier_stiffness,
            ],
            outputs=[energy],
            device=self.device,
        )
        if self.model.tri_count > 0:
            wp.launch(
                kernels.add_triangle_energy,
                dim=self.model.tri_count,
                inputs=[
                    x,
                    self.model.tri_areas,
                    self.model.tri_poses,
                    self.model.tri_indices,
                    self.model.style3d.tri_aniso_ke,
                ],
                outputs=[energy],
                device=self.device,
            )
        if self.model.edge_count > 0:
            wp.launch(
                kernels.add_bending_energy,
                dim=self.model.edge_count,
                inputs=[
                    x,
                    self.model.style3d.edge_rest_area,
                    self.model.style3d.edge_bending_cot,
                    self.model.edge_indices,
                    self.model.edge_bending_properties,
                ],
                outputs=[energy],
                device=self.device,
            )

    def _multiply_barrier_hessian(self, vector: wp.array) -> wp.array:
        wp.launch(
            kernels.multiply_barrier_hessian,
            dim=self.model.particle_count,
            inputs=[vector, self._contact_hessian, self._plane_normal],
            outputs=[self._barrier_product],
            device=self.device,
        )
        if self._self_contact is not None:
            self._self_contact.multiply(vector, self._barrier_product)
        if self.model.tri_count and not self.config.use_projective_hessian:
            wp.launch(
                kernels.multiply_membrane_metric,
                dim=self.model.tri_count,
                inputs=[
                    self._x_current,
                    self.model.tri_indices,
                    self.model.tri_poses,
                    self.model.style3d.tri_aniso_ke,
                    self.model.tri_areas,
                    vector,
                    self._barrier_product,
                ],
                device=self.device,
            )
        return self._barrier_product

    def _array_inner(self, a: wp.array, b: wp.array, output: wp.array) -> None:
        output.zero_()
        wp.launch(
            kernels.array_inner_tiled,
            dim=(self._reduction_block_count, self._reduction_block_dim),
            block_dim=self._reduction_block_dim,
            inputs=[a, b, self.model.particle_count, 0],
            outputs=[output],
            device=self.device,
        )

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one transactional IPC step.

        Args:
            state_in: Committed input state. It is never modified.
            state_out: Output state. It receives the converged iterate or an
                exact rollback copy when the step fails.
            control: Unused control input.
            contacts: Must be ``None``; IPC owns its primitive contact query.
            dt: Fixed positive time step [s]. Recapture if it changes.
        """
        del control
        if contacts is not None:
            raise ValueError("SolverIPC owns primitive collision queries; contacts must be None")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be > 0")

        self._refresh_particle_flags()
        self._invalid.zero_()
        self._solve_active.zero_()
        self._reference_residual.zero_()
        self.diagnostics.residual.zero_()
        self.diagnostics.minimum_gap.fill_(3.4028235e38)
        self.diagnostics.newton_iterations.zero_()
        self.diagnostics.line_search_iterations.zero_()
        self.diagnostics.status.fill_(int(self.Status.NOT_RUN))

        wp.launch(
            kernels.initialize_step,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self._plane_normal,
                self.config.plane_offset,
                self.config.minimum_separation,
                self.model.gravity,
                self.model.particle_world,
                self.model.particle_mass,
                self._particle_flags,
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                self.pd_diagonal,
            ],
            outputs=[
                self._x_previous,
                self._x_predictor,
                self._x_current,
                self._static_diagonal,
                self._invalid,
                self.diagnostics.minimum_gap,
            ],
            device=self.device,
        )
        if self._self_contact is not None:
            self._self_contact.query(self._x_current)
            self._self_contact.validate_initial(self._x_current, self._invalid)
            self._self_contact.minimum.fill_(1.0e30)
            self._self_contact.scratch_energy.zero_()
            self._self_contact.energy(self._x_current, self._self_contact.scratch_energy, self._invalid)
        wp.launch(
            kernels.finish_initialization,
            dim=1,
            inputs=[
                self._invalid,
                self._solve_active,
                self.diagnostics.status,
                int(self.Status.NOT_RUN),
                int(self.Status.INVALID_INITIAL_STATE),
            ],
            device=self.device,
        )

        if self._self_contact is not None:
            self._self_contact.check_overflow(self._solve_active, self.diagnostics.status)

        def run_line_search_iteration():
            wp.launch(
                kernels.prepare_candidate,
                dim=self.model.particle_count,
                inputs=[
                    self._x_current,
                    self._direction,
                    self.model.particle_mass,
                    self._particle_flags,
                    self._alpha,
                    self._solve_active,
                    self._accepted,
                ],
                outputs=[self._x_candidate],
                device=self.device,
            )
            self._invalid.zero_()
            wp.launch(
                kernels.validate_candidate,
                dim=self.model.particle_count,
                inputs=[
                    self._x_candidate,
                    self.model.particle_mass,
                    self._particle_flags,
                    self._plane_normal,
                    self.config.plane_offset,
                    self.config.minimum_separation,
                    self._solve_active,
                    self._accepted,
                ],
                outputs=[self._invalid],
                device=self.device,
            )
            self._candidate_energy.zero_()
            self._accumulate_energy(self._x_candidate, self._candidate_energy, dt, swept=True)
            wp.launch(
                kernels.accept_candidate,
                dim=1,
                inputs=[
                    self.config.armijo,
                    self.config.energy_tolerance,
                    self._current_energy,
                    self._candidate_energy,
                    self._rhs_dot_direction,
                    self._invalid,
                    self._solve_active,
                    self._alpha,
                    self._accepted,
                    self._line_search_active,
                    self._line_search_current_iterations,
                    self.diagnostics.line_search_iterations,
                    self.config.max_line_search_iterations,
                ],
                device=self.device,
            )
            wp.launch(
                kernels.commit_candidate,
                dim=self.model.particle_count,
                inputs=[self._x_candidate, self._accepted],
                outputs=[self._x_current],
                device=self.device,
            )

        def run_newton_iteration():
            if self._self_contact is not None:
                self._self_contact.query(self._x_current)
                self._self_contact.check_overflow(self._solve_active, self.diagnostics.status)
            wp.launch(
                kernels.initialize_rhs,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    self.model.particle_mass,
                    self._particle_flags,
                    self._x_current,
                    self._x_predictor,
                    self._solve_active,
                ],
                outputs=[self._rhs],
                device=self.device,
            )
            self._add_elastic_forces()
            self._metric_diagonal.zero_()
            if self._self_contact is not None:
                self._self_contact.assemble(self._x_current, self._solve_active, self._rhs)
            if self.model.tri_count and not self.config.use_projective_hessian:
                wp.launch(
                    kernels.add_membrane_diagonal,
                    dim=self.model.tri_count,
                    inputs=[
                        self._x_current,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.style3d.tri_aniso_ke,
                        self.model.tri_areas,
                        self._metric_diagonal,
                    ],
                    device=self.device,
                )
            wp.launch(
                kernels.add_barrier_forces,
                dim=self.model.particle_count,
                inputs=[
                    self._x_current,
                    self.model.particle_mass,
                    self._particle_flags,
                    self._plane_normal,
                    self.config.plane_offset,
                    self.config.minimum_separation,
                    self.config.contact_distance,
                    self.config.barrier_stiffness,
                    self._solve_active,
                ],
                outputs=[self._rhs, self._contact_hessian],
                device=self.device,
            )
            wp.launch(
                kernels.mask_rhs,
                dim=self.model.particle_count,
                inputs=[self.model.particle_mass, self._particle_flags, self._solve_active],
                outputs=[self._rhs],
                device=self.device,
            )

            self._residual_squared.zero_()
            wp.launch(
                kernels.maximum_residual_squared,
                dim=self.model.particle_count,
                inputs=[self._rhs],
                outputs=[self._residual_squared],
                device=self.device,
            )
            wp.launch(
                kernels.update_convergence,
                dim=1,
                inputs=[
                    self.config.absolute_tolerance,
                    self.config.relative_tolerance,
                    self._residual_squared,
                    self._reference_residual,
                    self.diagnostics.residual,
                    self.diagnostics.newton_iterations,
                    self._solve_active,
                    self.diagnostics.status,
                    int(self.Status.CONVERGED),
                    int(self.Status.LINEAR_BREAKDOWN),
                ],
                device=self.device,
            )
            wp.launch(
                kernels.mask_rhs,
                dim=self.model.particle_count,
                inputs=[self.model.particle_mass, self._particle_flags, self._solve_active],
                outputs=[self._rhs],
                device=self.device,
            )

            preconditioner_kernel = (
                kernels.prepare_rank_one_preconditioner
                if self.config.preconditioner == "rank_one"
                else kernels.prepare_dense_preconditioner
            )
            wp.launch(
                preconditioner_kernel,
                dim=self.model.particle_count,
                inputs=[
                    self._static_diagonal,
                    self._contact_hessian,
                    self._plane_normal,
                    self._particle_flags,
                    self._solve_active,
                ],
                outputs=[self._inverse_diagonal],
                device=self.device,
            )
            if self._self_contact is not None or (self.model.tri_count and not self.config.use_projective_hessian):
                wp.launch(
                    invert_blocks,
                    dim=self.model.particle_count,
                    inputs=[
                        self._metric_diagonal,
                        self._static_diagonal,
                        self._contact_hessian,
                        self._plane_normal,
                        self._inverse_diagonal,
                    ],
                    device=self.device,
                )
            self._linear_solver.solve(
                self.pd_non_diagonals,
                self._static_diagonal,
                None,
                self._rhs,
                self._inverse_diagonal,
                self._direction,
                self.config.max_pcg_iterations,
                self._multiply_barrier_hessian,
            )
            wp.launch(
                kernels.mask_direction,
                dim=self.model.particle_count,
                inputs=[self.model.particle_mass, self._particle_flags, self._solve_active],
                outputs=[self._direction],
                device=self.device,
            )
            self._array_inner(self._rhs, self._direction, self._rhs_dot_direction)
            wp.launch(
                kernels.validate_direction,
                dim=1,
                inputs=[
                    self._rhs_dot_direction,
                    self._solve_active,
                    self.diagnostics.status,
                    int(self.Status.LINEAR_BREAKDOWN),
                ],
                device=self.device,
            )

            if self._self_contact is not None:
                self._self_contact.query(self._x_current, self._direction)
                self._self_contact.check_overflow(self._solve_active, self.diagnostics.status, swept=True)
            self._current_energy.zero_()
            self._accumulate_energy(self._x_current, self._current_energy, dt, swept=True)
            self._alpha.fill_(self.config.initial_step_size)
            wp.launch(
                kernels.bound_plane_step,
                dim=self.model.particle_count,
                inputs=[
                    self._x_current,
                    self._direction,
                    self.model.particle_mass,
                    self._particle_flags,
                    self._plane_normal,
                    self.config.plane_offset,
                    self.config.minimum_separation,
                    self.config.ccd_safety,
                    self._solve_active,
                ],
                outputs=[self._alpha],
                device=self.device,
            )
            if self._self_contact is not None:
                self._self_contact.bound(self._x_current, self._direction, self._solve_active, self._alpha)
            wp.launch(
                kernels.begin_line_search,
                dim=1,
                inputs=[
                    self._solve_active,
                    self._accepted,
                    self._line_search_active,
                    self._line_search_current_iterations,
                ],
                device=self.device,
            )

            if self.config.graph_mode == "conditional":
                wp.capture_while(self._line_search_active, run_line_search_iteration)
            else:
                for _ in range(self.config.max_line_search_iterations):
                    run_line_search_iteration()

            wp.launch(
                kernels.finish_line_search,
                dim=1,
                inputs=[
                    self._accepted,
                    self._solve_active,
                    self.diagnostics.status,
                    self.diagnostics.newton_iterations,
                    int(self.Status.LINE_SEARCH_EXHAUSTED),
                    self.config.max_newton_iterations,
                    int(self.Status.NEWTON_EXHAUSTED),
                ],
                device=self.device,
            )

        if self.config.graph_mode == "conditional":
            wp.capture_while(self._solve_active, run_newton_iteration)
        else:
            for _ in range(self.config.max_newton_iterations):
                run_newton_iteration()

        wp.launch(
            kernels.finish_newton,
            dim=1,
            inputs=[
                self._solve_active,
                self.diagnostics.status,
                int(self.Status.NEWTON_EXHAUSTED),
            ],
            device=self.device,
        )
        self.diagnostics.minimum_gap.fill_(3.4028235e38)
        wp.launch(
            kernels.commit_step,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.config.velocity_damping,
                int(self.Status.CONVERGED),
                self.diagnostics.status,
                self._plane_normal,
                self.config.plane_offset,
                self.config.minimum_separation,
                self.model.particle_mass,
                self._particle_flags,
                self._x_previous,
                self._x_current,
                state_in.particle_qd,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd, self.diagnostics.minimum_gap],
            device=self.device,
        )
        wp.launch(
            kernels.record_step_status,
            dim=1,
            inputs=[
                int(self.Status.CONVERGED),
                self.diagnostics.status,
                self.diagnostics.failed_steps,
            ],
            device=self.device,
        )

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register the cloth attributes shared with :class:`SolverStyle3D`."""
        SolverStyle3D.register_custom_attributes(builder)
