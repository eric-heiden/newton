# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free pressure Poisson solver for the MAC fluid solver.

Solves ``A q = b`` with ``A = -div(grad(.))`` discretized with the standard
7-point stencil over fluid cells and homogeneous Neumann conditions at solid
and wall faces. ``q`` is the dt-scaled pressure ``q = p * dt / rho`` [m^2/s].

The solver is a Jacobi-preconditioned conjugate-gradient loop with a fixed
iteration count and device-resident scalars, so a step never synchronizes
with the host and is compatible with CUDA graph capture. In the closed
all-Neumann domain the operator is singular with the constant vector as its
null space; the right-hand side is made compatible upstream (see
``pressure_rhs_kernel``), which keeps CG within the range space.
"""

from __future__ import annotations

import warp as wp

from .grid import CELL_FLUID, is_solid_cell

__all__ = ["PressureSolver"]


@wp.kernel(enable_backward=False)
def _build_diagonal_kernel(
    dx: float,
    cell_label: wp.array3d[wp.int32],
    diag: wp.array[float],
):
    i, j, k = wp.tid()
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    idx = (i * ny + j) * nz + k

    if cell_label[i, j, k] != CELL_FLUID:
        # identity row for solid cells keeps the flat system well-defined
        diag[idx] = 1.0
        return

    n_open = int(0)
    for n in range(6):
        d = n // 2
        step = 2 * (n % 2) - 1
        ni = i
        nj = j
        nk = k
        if d == 0:
            ni += step
        elif d == 1:
            nj += step
        else:
            nk += step
        if not is_solid_cell(cell_label, ni, nj, nk):
            n_open += 1

    if n_open == 0:
        diag[idx] = 1.0
    else:
        diag[idx] = float(n_open) / (dx * dx)


@wp.kernel(enable_backward=False)
def _laplacian_apply_kernel(
    dx: float,
    cell_label: wp.array3d[wp.int32],
    x: wp.array[float],
    out: wp.array[float],
):
    i, j, k = wp.tid()
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    idx = (i * ny + j) * nz + k

    if cell_label[i, j, k] != CELL_FLUID:
        out[idx] = x[idx]
        return

    inv_dx2 = 1.0 / (dx * dx)
    acc = float(0.0)
    x_c = x[idx]
    for n in range(6):
        d = n // 2
        step = 2 * (n % 2) - 1
        ni = i
        nj = j
        nk = k
        if d == 0:
            ni += step
        elif d == 1:
            nj += step
        else:
            nk += step
        if not is_solid_cell(cell_label, ni, nj, nk):
            nidx = (ni * ny + nj) * nz + nk
            acc += (x_c - x[nidx]) * inv_dx2

    out[idx] = acc


@wp.kernel(enable_backward=False)
def _jacobi_precondition_kernel(
    r: wp.array[float],
    diag: wp.array[float],
    z: wp.array[float],
):
    i = wp.tid()
    z[i] = r[i] / diag[i]


# Relative squared-residual threshold below which the CG update freezes.
# Iterating past float32 convergence on the semi-definite Neumann system
# amplifies roundoff and diverges, so converged iterations become no-ops
# (alpha = beta = 0) while the launch count stays fixed for graph capture.
_CG_FREEZE_REL_SQ = wp.constant(1.0e-12)


@wp.func
def _cg_frozen(rz: float, rz0: float) -> bool:
    return rz <= _CG_FREEZE_REL_SQ * rz0 or rz <= 0.0


@wp.kernel(enable_backward=False)
def _cg_alpha_kernel(
    rz: wp.array[float],
    rz0: wp.array[float],
    s_dot_y: wp.array[float],
    alpha: wp.array[float],
):
    denom = s_dot_y[0]
    if denom > 0.0 and not _cg_frozen(rz[0], rz0[0]):
        alpha[0] = rz[0] / denom
    else:
        alpha[0] = 0.0


@wp.kernel(enable_backward=False)
def _cg_update_q_r_kernel(
    alpha: wp.array[float],
    s: wp.array[float],
    y: wp.array[float],
    q: wp.array[float],
    r: wp.array[float],
):
    i = wp.tid()
    a = alpha[0]
    q[i] = q[i] + a * s[i]
    r[i] = r[i] - a * y[i]


@wp.kernel(enable_backward=False)
def _cg_beta_kernel(
    rz_new: wp.array[float],
    rz0: wp.array[float],
    rz: wp.array[float],
    beta: wp.array[float],
):
    denom = rz[0]
    if denom > 0.0 and not _cg_frozen(rz_new[0], rz0[0]):
        beta[0] = rz_new[0] / denom
    else:
        beta[0] = 0.0
    rz[0] = rz_new[0]


@wp.kernel(enable_backward=False)
def _cg_update_s_kernel(
    beta: wp.array[float],
    z: wp.array[float],
    s: wp.array[float],
):
    i = wp.tid()
    s[i] = z[i] + beta[0] * s[i]


@wp.kernel(enable_backward=False)
def _remove_mean_kernel(
    cell_label: wp.array3d[wp.int32],
    q_sum: wp.array[float],
    fluid_cell_count: wp.array[wp.int32],
    q: wp.array[float],
):
    i, j, k = wp.tid()
    if cell_label[i, j, k] != CELL_FLUID:
        return
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    idx = (i * ny + j) * nz + k
    count = wp.max(fluid_cell_count[0], 1)
    q[idx] = q[idx] - q_sum[0] / float(count)


class PressureSolver:
    """Fixed-iteration Jacobi-preconditioned CG for the pressure Poisson system."""

    def __init__(self, shape: tuple[int, int, int], iterations: int, device):
        self.shape = shape
        self.iterations = int(iterations)
        n = shape[0] * shape[1] * shape[2]
        with wp.ScopedDevice(device):
            self.q = wp.zeros(n, dtype=float)
            self.b = wp.zeros(n, dtype=float)
            self.diag = wp.zeros(n, dtype=float)
            self._r = wp.zeros(n, dtype=float)
            self._z = wp.zeros(n, dtype=float)
            self._s = wp.zeros(n, dtype=float)
            self._y = wp.zeros(n, dtype=float)
            self._rz = wp.zeros(1, dtype=float)
            self._rz0 = wp.zeros(1, dtype=float)
            self._rz_new = wp.zeros(1, dtype=float)
            self._s_dot_y = wp.zeros(1, dtype=float)
            self._alpha = wp.zeros(1, dtype=float)
            self._beta = wp.zeros(1, dtype=float)
            self._q_sum = wp.zeros(1, dtype=float)
            self.residual_sq = wp.zeros(1, dtype=float)
        self.device = self.q.device

    def build(self, dx: float, cell_label: wp.array3d[wp.int32]):
        """Rebuild the preconditioner diagonal for the current cell labels."""
        wp.launch(
            _build_diagonal_kernel,
            dim=self.shape,
            inputs=[dx, cell_label, self.diag],
            device=self.device,
        )

    def solve(self, dx: float, cell_label: wp.array3d[wp.int32], fluid_cell_count: wp.array[wp.int32]):
        """Run the fixed CG iteration for the current right-hand side ``b``.

        The result is written to ``q`` (zero initial guess, mean removed over
        fluid cells) and the final squared residual norm to ``residual_sq``.
        """
        n = self.q.shape[0]
        dev = self.device

        self.q.zero_()
        # r = b (zero initial guess)
        wp.copy(self._r, self.b)
        wp.launch(_jacobi_precondition_kernel, dim=n, inputs=[self._r, self.diag, self._z], device=dev)
        wp.copy(self._s, self._z)
        wp.utils.array_inner(self._r, self._z, out=self._rz)
        wp.copy(self._rz0, self._rz)

        for _ in range(self.iterations):
            wp.launch(_laplacian_apply_kernel, dim=self.shape, inputs=[dx, cell_label, self._s, self._y], device=dev)
            wp.utils.array_inner(self._s, self._y, out=self._s_dot_y)
            wp.launch(_cg_alpha_kernel, dim=1, inputs=[self._rz, self._rz0, self._s_dot_y, self._alpha], device=dev)
            wp.launch(_cg_update_q_r_kernel, dim=n, inputs=[self._alpha, self._s, self._y, self.q, self._r], device=dev)
            wp.launch(_jacobi_precondition_kernel, dim=n, inputs=[self._r, self.diag, self._z], device=dev)
            wp.utils.array_inner(self._r, self._z, out=self._rz_new)
            wp.launch(_cg_beta_kernel, dim=1, inputs=[self._rz_new, self._rz0, self._rz, self._beta], device=dev)
            wp.launch(_cg_update_s_kernel, dim=n, inputs=[self._beta, self._z, self._s], device=dev)

        # final residual norm and null-space cleanup; q is zero on solid cells,
        # so a deterministic whole-array sum equals the fluid-cell sum
        wp.utils.array_inner(self._r, self._r, out=self.residual_sq)
        wp.utils.array_sum(self.q, out=self._q_sum)
        wp.launch(
            _remove_mean_kernel,
            dim=self.shape,
            inputs=[cell_label, self._q_sum, fluid_cell_count, self.q],
            device=dev,
        )
