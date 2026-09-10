# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Damped Gauss-Newton shooting, with GPU finite differences and line search."""

import math

import mujoco_warp as mjw
import warp as wp

from newton.examples.robot.wbc_mpc import (
    WholeBodyMPC,
    _finish,
    _min_cost,
    _min_index,
    _select,
    _shift,
)


@wp.kernel
def _differences(center: wp.array2d[float], epsilon: float, proposals: wp.array3d[float]):
    world, k, j = wp.tid()
    value = center[k, j]
    if world > 0 and (world - 1) // 2 == k * center.shape[1] + j:
        sign = float(1.0)
        if world % 2 == 0:
            sign = -1.0
        value += sign * epsilon
    proposals[world, k, j] = wp.clamp(value, -0.7, 0.7)


@wp.kernel
def _jacobian(residual: wp.array2d[float], count: int, epsilon: float, jacobian: wp.array2d[float]):
    i, r = wp.tid()
    value = float(0.0)
    if i < count and r < residual.shape[1]:
        value = (residual[2 * i + 1, r] - residual[2 * i + 2, r]) / (2.0 * epsilon)
    elif i == count and r < residual.shape[1]:
        value = residual[0, r]
    if not wp.isfinite(value):
        value = 0.0
    jacobian[i, r] = value


@wp.kernel
def _gram(jacobian: wp.array2d[float], gram: wp.array2d[float]):
    i, j = wp.tid()
    total = wp.tile_zeros(shape=(16, 16), dtype=float)
    for k in range(jacobian.shape[1] // 32):
        a = wp.tile_load(jacobian, shape=(16, 32), offset=(16 * i, 32 * k))
        b = wp.tile_load(jacobian, shape=(16, 32), offset=(16 * j, 32 * k))
        total += wp.tile_matmul(a, wp.tile_transpose(b))
    wp.tile_store(gram, total, offset=(16 * i, 16 * j))


@wp.kernel
def _system(gram: wp.array2d[float], count: int, damping: float, matrix: wp.array2d[float], rhs: wp.array[float]):
    i, j = wp.tid()
    value = float(0.0)
    if i < count and j < count:
        value = gram[i, j]
    if i == j:
        value += damping * (value + 1.0)
    matrix[i, j] = value
    if j == 0:
        rhs[i] = 0.0
        if i < count:
            rhs[i] = -gram[i, count]


def _solve_kernel(size):
    @wp.kernel(module="unique", enable_backward=False)
    def solve(matrix: wp.array2d[float], rhs: wp.array[float], direction: wp.array[float]):
        factor = wp.tile_load(matrix, shape=(size, size), storage="shared")
        wp.tile_cholesky_inplace(factor)
        b = wp.tile_load(rhs, shape=(size,))
        x = wp.tile_cholesky_solve(factor, b)
        wp.tile_store(direction, x)

    return solve


@wp.kernel
def _max_step(direction: wp.array[float], maximum: wp.array[float]):
    i = wp.tid()
    wp.atomic_max(maximum, 0, wp.abs(direction[i]))


@wp.kernel
def _lines(
    center: wp.array2d[float],
    direction: wp.array[float],
    maximum: wp.array[float],
    trust: float,
    proposals: wp.array3d[float],
):
    world, k, j = wp.tid()
    value = center[k, j]
    if world == 1:
        value = 0.0
    elif world >= 2:
        delta = direction[k * center.shape[1] + j]
        if wp.isfinite(delta):
            value += wp.pow(0.5, float(world - 2)) * wp.min(1.0, trust / wp.max(maximum[0], 1.0e-8)) * delta
    proposals[world, k, j] = wp.clamp(value, -0.7, 0.7)


@wp.kernel
def _coordinate_fallback(
    proposals: wp.array3d[float],
    candidate_cost: wp.array[float],
    best: wp.array[int],
    line_cost: wp.array[float],
    center: wp.array2d[float],
    plan: wp.array2d[float],
):
    k, j = wp.tid()
    if candidate_cost[0] < line_cost[0] and candidate_cost[0] < 1.0e19:
        value = proposals[best[0], k, j]
        center[k, j] = value
        plan[k, j] = value


@wp.kernel
def _merge_minimum(candidate_cost: wp.array[float], minimum: wp.array[float]):
    minimum[0] = wp.min(candidate_cost[0], minimum[0])


class WholeBodyGaussNewton(WholeBodyMPC):
    """Optimize the knot vector with central differences and damped least squares.

    Every line-search candidate is re-simulated with the complete cost.
    Pose and non-foot force residuals share the exact objective with sampling.
    Capture includes both rollout batches and the linear solve.
    """

    def __init__(
        self, model, kp, kd, reference, *, epsilon=0.03, damping=0.1, trust=0.2, coordinate_search=False, **kwargs
    ):
        count = kwargs.get("knots", 4) * model.nu
        if count >= 128:
            raise ValueError("The tiled Gauss-Newton solve supports at most 127 knot parameters")
        if any(not math.isfinite(x) or x <= 0 for x in (epsilon, damping, trust)):
            raise ValueError("Gauss-Newton epsilon, damping and trust radius must be finite and positive")
        kwargs["samples"] = 2 * count + 1
        super().__init__(model, kp, kd, reference, **kwargs)
        self.epsilon, self.damping, self.trust = epsilon, damping, trust
        self.coordinate_search = coordinate_search
        self.count = count
        self.size = max(16, 1 << count.bit_length())
        self.width = self.residual_width
        self.diff_data, self.diff_proposals, self.diff_costs = self.data, self.proposals, self.costs
        with wp.ScopedDevice(self.device):
            self.residual = wp.zeros((self.samples, self.steps * self.width))
            self.jacobian = wp.zeros((self.size, (self.steps * self.width + 31) // 32 * 32))
            self.gram, self.matrix = wp.zeros((self.size, self.size)), wp.zeros((self.size, self.size))
            self.rhs, self.direction, self.maximum = wp.zeros(self.size), wp.zeros(self.size), wp.zeros(1)
            self.line_data = mjw.make_data(self.cpu_model, nworld=8, nconmax=96, njmax=192)
            self.line_proposals = wp.zeros((8, self.plan.shape[0], model.nu))
            self.line_costs = wp.zeros(8)
            self.difference_minimum, self.difference_best = wp.zeros(1), wp.zeros(1, dtype=int)
        self.solve_kernel = _solve_kernel(self.size)

    def select(self):
        self.minimum.fill_(float("inf"))
        self.best.fill_(self.samples)
        wp.launch(
            _min_cost, self.samples, inputs=[self.costs, self.data.overflow, self.statistics], outputs=[self.minimum]
        )
        wp.launch(_min_index, self.samples, inputs=[self.costs, self.minimum], outputs=[self.best])
        wp.launch(
            _select, self.plan.shape, inputs=[self.proposals, self.minimum, self.best], outputs=[self.center, self.plan]
        )

    def optimize(self, q, v, clock):
        wp.launch(_shift, self.plan.shape, inputs=[self.plan, clock, self.last, self.spacing], outputs=[self.center])
        for r in range(self.rounds):
            self.record_traces = self.traces is not None and r == self.rounds - 1
            self.data, self.proposals, self.costs = self.diff_data, self.diff_proposals, self.diff_costs
            self.samples = self.proposals.shape[0]
            self.save_residuals = True
            wp.launch(_differences, self.proposals.shape, inputs=[self.center, self.epsilon], outputs=[self.proposals])
            self.rollout(q, v, clock)
            self.minimum.fill_(float("inf"))
            wp.launch(
                _min_cost,
                self.samples,
                inputs=[self.costs, self.data.overflow, self.statistics],
                outputs=[self.minimum],
            )
            if self.coordinate_search:
                wp.copy(self.difference_minimum, self.minimum)
                self.difference_best.fill_(self.samples)
                wp.launch(_min_index, self.samples, inputs=[self.costs, self.minimum], outputs=[self.difference_best])
            wp.launch(
                _jacobian,
                self.jacobian.shape,
                inputs=[self.residual, self.count, self.epsilon],
                outputs=[self.jacobian],
            )
            wp.launch_tiled(
                _gram,
                dim=(self.size // 16, self.size // 16),
                inputs=[self.jacobian],
                outputs=[self.gram],
                block_dim=128,
            )
            wp.launch(
                _system,
                self.matrix.shape,
                inputs=[self.gram, self.count, self.damping],
                outputs=[self.matrix, self.rhs],
            )
            wp.launch_tiled(
                self.solve_kernel, dim=1, inputs=[self.matrix, self.rhs], outputs=[self.direction], block_dim=128
            )
            self.maximum.zero_()
            wp.launch(_max_step, self.count, inputs=[self.direction], outputs=[self.maximum])
            self.data, self.proposals, self.costs = self.line_data, self.line_proposals, self.line_costs
            self.samples = self.proposals.shape[0]
            self.save_residuals = False
            wp.launch(
                _lines,
                self.proposals.shape,
                inputs=[self.center, self.direction, self.maximum, self.trust],
                outputs=[self.proposals],
            )
            self.rollout(q, v, clock)
            self.select()
            if self.coordinate_search:
                # Reuse the evaluated coordinate probes when the local solve stalls.
                wp.launch(
                    _coordinate_fallback,
                    self.plan.shape,
                    inputs=[self.diff_proposals, self.difference_minimum, self.difference_best, self.minimum],
                    outputs=[self.center, self.plan],
                )
                wp.launch(_merge_minimum, 1, inputs=[self.difference_minimum], outputs=[self.minimum])
        wp.launch(_finish, 1, inputs=[clock, self.minimum], outputs=[self.last, self.iteration, self.failure_count])
        if self.traces is not None:
            self.traces.finish(self, q, clock)
