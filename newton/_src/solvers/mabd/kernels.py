# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from ...sim import BodyFlags
from ..solver import integrate_rigid_body


@wp.kernel
def compute_body_q_com(
    body_q: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    body_q_com[tid] = body_q[tid] * body_X_com[tid]


@wp.kernel
def mark_articulated_bodies(
    joint_child: wp.array(dtype=wp.int32),
    body_articulated: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    child = joint_child[tid]
    if child >= 0:
        body_articulated[child] = True


@wp.kernel
def set_effective_joint_armature(
    joint_child: wp.array(dtype=wp.int32),
    body_flags: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    model_joint_armature: wp.array(dtype=float),
    kinematic_armature: float,
    effective_joint_armature: wp.array(dtype=float),
):
    tid = wp.tid()

    dof_start = joint_qd_start[tid]
    dof_end = joint_qd_start[tid + 1]
    child = joint_child[tid]
    is_kinematic = (body_flags[child] & BodyFlags.KINEMATIC) != 0

    for i in range(dof_start, dof_end):
        effective_joint_armature[i] = kinematic_armature if is_kinematic else model_joint_armature[i]


@wp.kernel
def integrate_unarticulated_bodies(
    body_articulated: wp.array(dtype=wp.bool),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_flags: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    angular_damping: float,
    dt: float,
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    if body_articulated[tid]:
        body_q_new[tid] = body_q[tid]
        body_qd_new[tid] = body_qd[tid]
        return

    if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:
        body_q_new[tid] = body_q[tid]
        body_qd_new[tid] = body_qd[tid]
        return

    world_idx = body_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

    q_new, qd_new = integrate_rigid_body(
        body_q[tid],
        body_qd[tid],
        body_f[tid],
        body_com[tid],
        body_inertia[tid],
        body_inv_mass[tid],
        body_inv_inertia[tid],
        world_g,
        angular_damping,
        dt,
    )

    body_q_new[tid] = q_new
    body_qd_new[tid] = qd_new


@wp.kernel
def eval_rigid_jacobian_batched(
    articulation_start: wp.array(dtype=wp.int32),
    joint_ancestor: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    J: wp.array(dtype=float, ndim=3),
):
    art_idx = wp.tid()

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]
    joint_count = joint_end - joint_start
    if joint_count == 0:
        return

    articulation_dof_start = joint_qd_start[joint_start]

    for i in range(joint_count):
        row_start = i * 6

        j = joint_start + i
        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_end = joint_qd_start[j + 1]
            joint_dof_count = joint_dof_end - joint_dof_start

            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof
                S = joint_S_s[joint_dof_start + dof]

                for k in range(6):
                    J[art_idx, row_start + k, col] = S[k]

            j = joint_ancestor[j]


@wp.kernel
def eval_rigid_mass_batched(
    articulation_start: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    M: wp.array(dtype=float, ndim=3),
):
    art_idx = wp.tid()

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]
    joint_count = joint_end - joint_start
    if joint_count == 0:
        return

    for link_idx in range(joint_count):
        child = joint_child[joint_start + link_idx]
        I_s = body_I_s[child]
        row_start = link_idx * 6

        for i in range(6):
            for j in range(6):
                M[art_idx, row_start + i, row_start + j] = I_s[i, j]


def create_inertia_matrix_kernel(num_joints: int, num_dofs: int):
    @wp.kernel
    def eval_inertia_matrix_tile(
        J_arr: wp.array3d(dtype=float),
        M_arr: wp.array3d(dtype=float),
        H_arr: wp.array3d(dtype=float),
    ):
        articulation = wp.tid()

        H = wp.tile_zeros(shape=(num_dofs, num_dofs), dtype=float)

        # Accumulate H = sum_i J_i^T M_i J_i using 6x6 spatial inertia blocks.
        for i in range(int(num_joints)):
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))
            J_body = wp.tile_load(J_arr[articulation], shape=(6, num_dofs), offset=(i * 6, 0))
            P_body = wp.tile_matmul(M_body, J_body)
            wp.tile_matmul(wp.tile_transpose(J_body), P_body, H)

        wp.tile_store(H_arr[articulation], H)

    return eval_inertia_matrix_tile


def create_cholesky_solve_kernel(num_dofs: int):
    @wp.kernel
    def eval_cholesky_solve_tile(
        H_arr: wp.array3d(dtype=float),
        R_arr: wp.array2d(dtype=float),
        tau_arr: wp.array2d(dtype=float),
        diagonal_regularization: float,
        qdd_arr: wp.array2d(dtype=float),
    ):
        articulation = wp.tid()

        H = wp.tile_load(H_arr[articulation], shape=(num_dofs, num_dofs), storage="shared")
        R = wp.tile_load(R_arr[articulation], shape=num_dofs, storage="shared")

        if diagonal_regularization > 0.0:
            R = wp.tile_map(
                wp.add,
                R,
                wp.tile_full(shape=num_dofs, value=diagonal_regularization, dtype=float),
            )

        L = wp.tile_cholesky(wp.tile_diag_add(H, R))

        tau = wp.tile_load(tau_arr[articulation], shape=num_dofs, storage="shared")
        qdd = wp.tile_cholesky_solve(L, tau)
        wp.tile_store(qdd_arr[articulation], qdd)

    return eval_cholesky_solve_tile


def create_inertia_matrix_solve_kernel(num_joints: int, num_dofs: int):
    @wp.kernel
    def eval_inertia_matrix_solve_tile(
        J_arr: wp.array3d(dtype=float),
        M_arr: wp.array3d(dtype=float),
        R_arr: wp.array2d(dtype=float),
        tau_arr: wp.array2d(dtype=float),
        diagonal_regularization: float,
        qdd_arr: wp.array2d(dtype=float),
    ):
        articulation = wp.tid()

        H = wp.tile_zeros(shape=(num_dofs, num_dofs), dtype=float)

        # Form H directly in tile memory to avoid materializing P or a separate factor kernel.
        for i in range(int(num_joints)):
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))
            J_body = wp.tile_load(J_arr[articulation], shape=(6, num_dofs), offset=(i * 6, 0))
            P_body = wp.tile_matmul(M_body, J_body)
            wp.tile_matmul(wp.tile_transpose(J_body), P_body, H)

        R = wp.tile_load(R_arr[articulation], shape=num_dofs, storage="shared")
        if diagonal_regularization > 0.0:
            R = wp.tile_map(
                wp.add,
                R,
                wp.tile_full(shape=num_dofs, value=diagonal_regularization, dtype=float),
            )

        L = wp.tile_cholesky(wp.tile_diag_add(H, R))
        tau = wp.tile_load(tau_arr[articulation], shape=num_dofs, storage="shared")
        qdd = wp.tile_cholesky_solve(L, tau)
        wp.tile_store(qdd_arr[articulation], qdd)

    return eval_inertia_matrix_solve_tile


@wp.kernel
def factor_articulation_mass_matrices(
    articulation_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_armature: wp.array(dtype=float),
    mass_matrix: wp.array(dtype=float, ndim=3),
    min_diagonal: float,
    cholesky_factor: wp.array(dtype=float, ndim=3),
):
    art_idx = wp.tid()

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]
    if joint_start == joint_end:
        return

    dof_start = joint_qd_start[joint_start]
    dof_end = joint_qd_start[joint_end]
    dof_count = dof_end - dof_start

    for i in range(dof_count):
        for j in range(dof_count):
            cholesky_factor[art_idx, i, j] = 0.0

    for j in range(dof_count):
        s = mass_matrix[art_idx, j, j] + joint_armature[dof_start + j]

        for k in range(j):
            r = cholesky_factor[art_idx, j, k]
            s -= r * r

        s = wp.max(s, min_diagonal)
        s = wp.sqrt(s)
        inv_s = 1.0 / s

        cholesky_factor[art_idx, j, j] = s

        for i in range(j + 1, dof_count):
            t = mass_matrix[art_idx, i, j]
            for k in range(j):
                t -= cholesky_factor[art_idx, i, k] * cholesky_factor[art_idx, j, k]
            cholesky_factor[art_idx, i, j] = t * inv_s


@wp.kernel
def solve_articulation_accelerations(
    articulation_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    cholesky_factor: wp.array(dtype=float, ndim=3),
    joint_tau: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
):
    art_idx = wp.tid()

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]
    if joint_start == joint_end:
        return

    dof_start = joint_qd_start[joint_start]
    dof_end = joint_qd_start[joint_end]
    dof_count = dof_end - dof_start

    for i in range(dof_count):
        s = joint_tau[dof_start + i]
        for j in range(i):
            s -= cholesky_factor[art_idx, i, j] * joint_qdd[dof_start + j]
        joint_qdd[dof_start + i] = s / cholesky_factor[art_idx, i, i]

    for i in range(dof_count - 1, -1, -1):
        s = joint_qdd[dof_start + i]
        for j in range(i + 1, dof_count):
            s -= cholesky_factor[art_idx, j, i] * joint_qdd[dof_start + j]
        joint_qdd[dof_start + i] = s / cholesky_factor[art_idx, i, i]
