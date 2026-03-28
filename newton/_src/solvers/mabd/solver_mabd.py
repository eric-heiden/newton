# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, State, eval_jacobian, eval_mass_matrix
from ..contact import RigidContactResolver
from ..featherstone.kernels import (
    compute_com_transforms,
    compute_spatial_inertia,
    convert_body_force_com_to_origin,
    copy_kinematic_joint_state,
    eval_fk_with_velocity_conversion,
    eval_rigid_id,
    eval_rigid_tau,
    integrate_generalized_joints,
    zero_kinematic_body_forces,
    zero_kinematic_joint_qdd,
)
from ..flags import SolverNotifyFlags
from ..semi_implicit.kernels_contact import (
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from ..semi_implicit.kernels_muscle import eval_muscle_forces
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from . import kernels as mabd_kernels
from .kernels import (
    compute_body_q_com,
    create_cholesky_solve_kernel,
    create_inertia_matrix_kernel,
    create_inertia_matrix_solve_kernel,
    eval_rigid_jacobian_batched,
    eval_rigid_mass_batched,
    factor_articulation_mass_matrices,
    integrate_unarticulated_bodies,
    mark_articulated_bodies,
    set_effective_joint_armature,
    solve_articulation_accelerations,
)


class SolverMABD(SolverBase):
    """GPU-native articulated rigid-body solver inspired by the M-ABD tree solve.

    This implementation focuses on the robotics-relevant tree-articulation case
    from Sec. 5.3 of the M-ABD paper and keeps the runtime path fully in Warp.
    It integrates Newton's existing rigid articulation/contact model without
    introducing new affine-body state on :class:`~newton.Model`.

    The current scope is intentionally pragmatic:

    - Tree articulations in reduced coordinates are solved through a per-articulation
      dense mass-matrix solve on the GPU.
    - Free rigid bodies and particles still use Newton's existing body/particle
      force pipelines.
    - Equality constraints, mimic constraints, and graph/loop solvers from the
      paper are not implemented here.
    """

    _KINEMATIC_ARMATURE = 1.0e10

    def __init__(
        self,
        model: Model,
        angular_damping: float = 0.05,
        friction_smoothing: float = 1.0,
        enable_tri_contact: bool = True,
        min_mass_matrix_diagonal: float = 1.0e-1,
        use_tile_gemm: bool = True,
        fuse_cholesky: bool = True,
        tile_block_dim: int = 128,
        rigid_contact_method: str = "impulse",
        impulse_contact_iterations: int = 4,
        impulse_contact_baumgarte: float = 0.3,
        impulse_contact_penetration_slop: float = 5.0e-4,
        impulse_contact_restitution_scale: float = 0.0,
        impulse_contact_restitution_velocity_threshold: float = 0.75,
    ):
        super().__init__(model)

        if model.joint_count and model.articulation_count == 0:
            raise ValueError(
                "SolverMABD expects reduced-coordinate articulations for jointed systems. "
                "Add articulations with ModelBuilder.add_articulation() or use a maximal-coordinate solver."
            )

        self.angular_damping = angular_damping
        self.friction_smoothing = friction_smoothing
        self.enable_tri_contact = enable_tri_contact
        self.min_mass_matrix_diagonal = min_mass_matrix_diagonal
        self.use_tile_gemm = use_tile_gemm
        self.fuse_cholesky = fuse_cholesky
        self.tile_block_dim = tile_block_dim
        self.rigid_contact_resolver = RigidContactResolver(
            model=model,
            method=rigid_contact_method,
            friction_smoothing=friction_smoothing,
            impulse_iterations=impulse_contact_iterations,
            impulse_baumgarte=impulse_contact_baumgarte,
            impulse_penetration_slop=impulse_contact_penetration_slop,
            impulse_restitution_scale=impulse_contact_restitution_scale,
            impulse_restitution_velocity_threshold=impulse_contact_restitution_velocity_threshold,
        )

        self._configure_tile_path()
        self._allocate_buffers()
        self._refresh_solver_metadata()

        if self._tile_path_enabled:
            wp.load_module(module=mabd_kernels, device=wp.get_device(model.device))

    def _configure_tile_path(self) -> None:
        model = self.model

        self._tile_path_enabled = False
        self._tile_joint_count = 0
        self._tile_dof_count = 0
        self.eval_inertia_matrix_tile_kernel = None
        self.eval_cholesky_solve_tile_kernel = None
        self.eval_inertia_matrix_solve_tile_kernel = None

        if not self.use_tile_gemm or model.articulation_count == 0 or model.joint_count == 0:
            return

        if not wp.get_device(model.device).is_cuda:
            return

        articulation_start = model.articulation_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        joint_counts = np.diff(articulation_start)
        dof_counts = np.asarray(
            [joint_qd_start[articulation_start[i + 1]] - joint_qd_start[articulation_start[i]] for i in range(model.articulation_count)],
            dtype=np.int32,
        )

        if joint_counts.size == 0 or dof_counts.size == 0:
            return

        joint_count = int(joint_counts[0])
        dof_count = int(dof_counts[0])

        if joint_count <= 0 or dof_count <= 0:
            return

        if not (np.all(joint_counts == joint_count) and np.all(dof_counts == dof_count)):
            return

        self._tile_joint_count = joint_count
        self._tile_dof_count = dof_count
        self._tile_path_enabled = True

        if self.fuse_cholesky:
            self.eval_inertia_matrix_solve_tile_kernel = create_inertia_matrix_solve_kernel(
                joint_count,
                dof_count,
            )
        else:
            self.eval_inertia_matrix_tile_kernel = create_inertia_matrix_kernel(
                joint_count,
                dof_count,
            )
            self.eval_cholesky_solve_tile_kernel = create_cholesky_solve_kernel(
                dof_count,
            )

    def _allocate_buffers(self) -> None:
        model = self.model

        if model.body_count:
            self.body_X_com = wp.empty(
                model.body_count,
                dtype=wp.transform,
                device=model.device,
                requires_grad=False,
            )
            wp.launch(
                compute_com_transforms,
                dim=model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )

            self.body_q_com = wp.empty_like(model.body_q, requires_grad=False)
            self.body_I_m = wp.empty(
                model.body_count,
                dtype=wp.spatial_matrix,
                device=model.device,
                requires_grad=False,
            )
            self.body_I_s = wp.empty(
                model.body_count,
                dtype=wp.spatial_matrix,
                device=model.device,
                requires_grad=False,
            )
            self.body_v_s = wp.empty(
                model.body_count,
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=False,
            )
            self.body_a_s = wp.empty(
                model.body_count,
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=False,
            )
            self.body_f_s = wp.zeros(
                model.body_count,
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=False,
            )
            self.body_ft_s = wp.zeros(
                model.body_count,
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=False,
            )
            self.body_f_origin = wp.zeros_like(model.body_qd, requires_grad=False)
            self.body_articulated = wp.zeros(
                model.body_count,
                dtype=wp.bool,
                device=model.device,
                requires_grad=False,
            )

            wp.launch(
                compute_spatial_inertia,
                dim=model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )

        if model.joint_count:
            self.joint_S_s = wp.empty(
                model.joint_dof_count,
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=False,
            )
            self.joint_tau = wp.zeros_like(model.joint_qd, requires_grad=False)
            self.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=False)
            self.joint_armature_effective = wp.empty_like(model.joint_armature, requires_grad=False)

            if self._tile_path_enabled:
                row_count = self._tile_joint_count * 6
                dof_count = self._tile_dof_count

                self.tile_J = wp.zeros(
                    (model.articulation_count, row_count, dof_count),
                    dtype=float,
                    device=model.device,
                    requires_grad=False,
                )
                self.tile_M = wp.zeros(
                    (model.articulation_count, row_count, row_count),
                    dtype=float,
                    device=model.device,
                    requires_grad=False,
                )
                self.tile_H = None
                self.tile_L = None

                if not self.fuse_cholesky:
                    self.tile_H = wp.zeros(
                        (model.articulation_count, dof_count, dof_count),
                        dtype=float,
                        device=model.device,
                        requires_grad=False,
                    )

                self.joint_tau_tiled = self.joint_tau.reshape((model.articulation_count, dof_count))
                self.joint_qdd_tiled = self.joint_qdd.reshape((model.articulation_count, dof_count))
                self.joint_armature_effective_tiled = self.joint_armature_effective.reshape(
                    (model.articulation_count, dof_count)
                )
            else:
                max_links = model.max_joints_per_articulation
                max_dofs = model.max_dofs_per_articulation
                self.J = wp.zeros(
                    (model.articulation_count, max_links * 6, max_dofs),
                    dtype=float,
                    device=model.device,
                    requires_grad=False,
                )
                self.H = wp.zeros(
                    (model.articulation_count, max_dofs, max_dofs),
                    dtype=float,
                    device=model.device,
                    requires_grad=False,
                )
                self.L = wp.zeros(
                    (model.articulation_count, max_dofs, max_dofs),
                    dtype=float,
                    device=model.device,
                    requires_grad=False,
                )

    def _refresh_solver_metadata(self) -> None:
        model = self.model

        if model.body_count:
            self.body_articulated.zero_()
            if model.joint_count:
                wp.launch(
                    mark_articulated_bodies,
                    dim=model.joint_count,
                    inputs=[model.joint_child],
                    outputs=[self.body_articulated],
                    device=model.device,
                )

        if model.joint_count:
            wp.launch(
                set_effective_joint_armature,
                dim=model.joint_count,
                inputs=[
                    model.joint_child,
                    model.body_flags,
                    model.joint_qd_start,
                    model.joint_armature,
                    self._KINEMATIC_ARMATURE,
                ],
                outputs=[self.joint_armature_effective],
                device=model.device,
            )

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.JOINT_DOF_PROPERTIES):
            self._refresh_solver_metadata()

        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES and self.model.body_count:
            wp.launch(
                compute_com_transforms,
                dim=self.model.body_count,
                inputs=[self.model.body_com],
                outputs=[self.body_X_com],
                device=self.model.device,
            )
            wp.launch(
                compute_spatial_inertia,
                dim=self.model.body_count,
                inputs=[self.model.body_inertia, self.model.body_mass],
                outputs=[self.body_I_m],
                device=self.model.device,
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
        model = self.model

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            particle_f = state_in.particle_f if state_in.particle_count else None
            body_f_work = state_in.body_f if state_in.body_count else None

            if model.joint_count:
                eval_fk_with_velocity_conversion(model, state_in.joint_q, state_in.joint_qd, state_in)
                wp.launch(
                    compute_body_q_com,
                    dim=model.body_count,
                    inputs=[state_in.body_q, self.body_X_com],
                    outputs=[self.body_q_com],
                    device=model.device,
                )

            eval_spring_forces(model, state_in, particle_f)
            eval_triangle_forces(model, state_in, control, particle_f)
            eval_bending_forces(model, state_in, particle_f)
            eval_tetrahedra_forces(model, state_in, control, particle_f)
            eval_particle_contact_forces(model, state_in, particle_f)

            if self.enable_tri_contact:
                eval_triangle_contact_forces(model, state_in, particle_f)

            if False:
                eval_muscle_forces(model, state_in, control, body_f_work)

            if body_f_work is not None:
                self.rigid_contact_resolver.resolve(state_in, contacts, dt, body_f_out=body_f_work)
            if particle_f is not None and body_f_work is not None:
                eval_particle_body_contact_forces(
                    model,
                    state_in,
                    contacts,
                    particle_f,
                    body_f_work,
                    body_f_in_world_frame=False,
                )

            if model.body_count:
                wp.launch(
                    zero_kinematic_body_forces,
                    dim=model.body_count,
                    inputs=[model.body_flags],
                    outputs=[body_f_work],
                    device=model.device,
                )

            self.integrate_particles(model, state_in, state_out, dt)

            if model.joint_count:
                self.body_f_origin.assign(body_f_work)
                wp.launch(
                    convert_body_force_com_to_origin,
                    dim=model.body_count,
                    inputs=[state_in.body_q, self.body_X_com],
                    outputs=[self.body_f_origin],
                    device=model.device,
                )

                self.body_ft_s.zero_()
                wp.launch(
                    eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_dof_dim,
                        self.body_I_m,
                        state_in.body_q,
                        self.body_q_com,
                        model.joint_X_p,
                        model.body_world,
                        model.gravity,
                    ],
                    outputs=[
                        self.joint_S_s,
                        self.body_I_s,
                        self.body_v_s,
                        self.body_f_s,
                        self.body_a_s,
                    ],
                    device=model.device,
                )

                self.joint_tau.zero_()
                wp.launch(
                    eval_rigid_tau,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        control.joint_target_pos,
                        control.joint_target_vel,
                        state_in.joint_q,
                        state_in.joint_qd,
                        control.joint_f,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_limit_ke,
                        model.joint_limit_kd,
                        self.joint_S_s,
                        self.body_f_s,
                        self.body_f_origin,
                        dt,
                    ],
                    outputs=[self.body_ft_s, self.joint_tau],
                    device=model.device,
                )

                self.joint_qdd.zero_()

                if self._tile_path_enabled:
                    wp.launch(
                        eval_rigid_jacobian_batched,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_ancestor,
                            model.joint_qd_start,
                            self.joint_S_s,
                        ],
                        outputs=[self.tile_J],
                        device=model.device,
                    )
                    wp.launch(
                        eval_rigid_mass_batched,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_child,
                            self.body_I_s,
                        ],
                        outputs=[self.tile_M],
                        device=model.device,
                    )

                    if self.fuse_cholesky:
                        wp.launch_tiled(
                            self.eval_inertia_matrix_solve_tile_kernel,
                            dim=model.articulation_count,
                            inputs=[
                                self.tile_J,
                                self.tile_M,
                                self.joint_armature_effective_tiled,
                                self.joint_tau_tiled,
                                self.min_mass_matrix_diagonal,
                            ],
                            outputs=[self.joint_qdd_tiled],
                            device=model.device,
                            block_dim=self.tile_block_dim,
                        )
                    else:
                        wp.launch_tiled(
                            self.eval_inertia_matrix_tile_kernel,
                            dim=model.articulation_count,
                            inputs=[self.tile_J, self.tile_M],
                            outputs=[self.tile_H],
                            device=model.device,
                            block_dim=self.tile_block_dim,
                        )
                        wp.launch_tiled(
                            self.eval_cholesky_solve_tile_kernel,
                            dim=model.articulation_count,
                            inputs=[
                                self.tile_H,
                                self.joint_armature_effective_tiled,
                                self.joint_tau_tiled,
                                self.min_mass_matrix_diagonal,
                            ],
                            outputs=[self.joint_qdd_tiled],
                            device=model.device,
                            block_dim=self.tile_block_dim,
                        )
                else:
                    eval_jacobian(model, state_in, J=self.J, joint_S_s=self.joint_S_s)
                    eval_mass_matrix(model, state_in, H=self.H, J=self.J, body_I_s=self.body_I_s)

                    wp.launch(
                        factor_articulation_mass_matrices,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_qd_start,
                            self.joint_armature_effective,
                            self.H,
                            self.min_mass_matrix_diagonal,
                        ],
                        outputs=[self.L],
                        device=model.device,
                    )
                    wp.launch(
                        solve_articulation_accelerations,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_qd_start,
                            self.L,
                            self.joint_tau,
                        ],
                        outputs=[self.joint_qdd],
                        device=model.device,
                    )
                wp.launch(
                    zero_kinematic_joint_qdd,
                    dim=model.joint_count,
                    inputs=[model.joint_child, model.body_flags, model.joint_qd_start],
                    outputs=[self.joint_qdd],
                    device=model.device,
                )
                wp.launch(
                    integrate_generalized_joints,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        self.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )
                wp.launch(
                    copy_kinematic_joint_state,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_child,
                        model.body_flags,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

            if model.body_count:
                wp.launch(
                    integrate_unarticulated_bodies,
                    dim=model.body_count,
                    inputs=[
                        self.body_articulated,
                        state_in.body_q,
                        state_in.body_qd,
                        body_f_work,
                        model.body_com,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.body_flags,
                        model.body_world,
                        model.gravity,
                        self.angular_damping,
                        dt,
                    ],
                    outputs=[state_out.body_q, state_out.body_qd],
                    device=model.device,
                )

            if model.joint_count:
                eval_fk_with_velocity_conversion(model, state_out.joint_q, state_out.joint_qd, state_out)

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        return

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        return
