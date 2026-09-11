# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""DIAL-MPC annealing on the shared joint-offset command basis.

Implements the horizon/iteration noise schedule and normalized softmax update
from LeCAR-Lab/dial-mpc. The final mean is physically evaluated for diagnostics
and prediction paths. All search operations run in a CUDA graph.
"""

import math

import mujoco_warp as mjw
import warp as wp

from newton.examples.robot.wbc_mpc import WholeBodyMPC, _finish, _min_cost, _shift


@wp.kernel
def _dial_propose(
    center: wp.array2d[float],
    iteration: wp.array[int],
    seed: int,
    round_index: int,
    sigma: float,
    horizon_decay: float,
    round_decay: float,
    proposals: wp.array3d[float],
):
    world, k, j = wp.tid()
    value = center[k, j]
    if world > 0 and k > 0:
        rng = wp.rand_init(
            seed + 7919 * iteration[0] + 104729 * round_index, (world * center.shape[0] + k) * center.shape[1] + j
        )
        scale = sigma * wp.pow(horizon_decay, float(center.shape[0] - 1 - k))
        scale *= wp.pow(round_decay, float(round_index))
        value += scale * wp.randn(rng)
    proposals[world, k, j] = wp.clamp(value, -0.7, 0.7)


@wp.kernel
def _dial_moments(costs: wp.array[float], minimum: wp.array[float], moments: wp.array[float]):
    i = wp.tid()
    if costs[i] < 1.0e19:
        delta = costs[i] - minimum[0]
        wp.atomic_add(moments, 0, 1.0)
        wp.atomic_add(moments, 1, delta)
        wp.atomic_add(moments, 2, delta * delta)


@wp.kernel
def _dial_weights(
    costs: wp.array[float],
    minimum: wp.array[float],
    moments: wp.array[float],
    temperature: float,
    weights: wp.array[float],
):
    i = wp.tid()
    n = wp.max(moments[0], 1.0)
    variance = wp.max(0.0, moments[2] / n - wp.pow(moments[1] / n, 2.0))
    scale = wp.max(1.0e-6, wp.sqrt(variance) * temperature)
    weights[i] = 0.0
    if costs[i] < 1.0e19:
        weights[i] = wp.exp(-(costs[i] - minimum[0]) / scale)


@wp.kernel
def _dial_update(proposals: wp.array3d[float], weights: wp.array[float], center: wp.array2d[float]):
    k, j = wp.tid()
    total, normalizer = float(0.0), float(0.0)
    for i in range(weights.shape[0]):
        total += weights[i] * proposals[i, k, j]
        normalizer += weights[i]
    if normalizer > 0.0:
        center[k, j] = total / normalizer


class WholeBodyDial(WholeBodyMPC):
    """DIAL-MPC updates with a fixed first knot and a shifted previous plan.

    The dynamics, task objective, PD parameterization and interpolation are
    shared with the other Newton controllers. No learned diffusion model is
    involved. Unlike the upstream implementation, the returned mean gets one
    extra rollout so its displayed future and acceptance cost are measured.
    """

    def __init__(self, *args, horizon_decay=0.9, round_decay=0.5, initial_rounds=10, dial_temperature=0.06, **kwargs):
        if not all(math.isfinite(x) and 0 < x <= 1 for x in (horizon_decay, round_decay)):
            raise ValueError("DIAL noise decay factors must be in (0, 1]")
        if not math.isfinite(dial_temperature) or dial_temperature <= 0 or initial_rounds < 1:
            raise ValueError("DIAL temperature and initial iteration count must be positive")
        super().__init__(*args, **kwargs)
        self.horizon_decay, self.round_decay = horizon_decay, round_decay
        self.initial_rounds, self.dial_temperature = initial_rounds, dial_temperature
        self.sigma = kwargs.get("noise", 0.12)
        self.search_data, self.search_proposals, self.search_costs = self.data, self.proposals, self.costs
        with wp.ScopedDevice(self.device):
            self.mean_data = mjw.make_data(self.cpu_model, nworld=1, nconmax=48, njmax=192)
            self.mean_proposals = self.plan.reshape((1, *self.plan.shape))
            self.mean_costs = wp.zeros(1)
            self.moments = wp.zeros(3)
            self.weights = wp.zeros(self.samples)
        self.trace_batches = ((self.search_data, self.search_costs), (self.mean_data, self.mean_costs))
        self.started = False

    def optimize(self, q, v, clock):
        wp.launch(_shift, self.plan.shape, inputs=[self.plan, clock, self.last, self.spacing], outputs=[self.center])
        self.data, self.proposals, self.costs = self.search_data, self.search_proposals, self.search_costs
        self.samples = self.proposals.shape[0]
        for r in range(self.rounds):
            self.record_traces = self.traces is not None and r == self.rounds - 1
            wp.launch(
                _dial_propose,
                self.proposals.shape,
                inputs=[self.center, self.iteration, self.seed, r, self.sigma, self.horizon_decay, self.round_decay],
                outputs=[self.proposals],
            )
            self.rollout(q, v, clock)
            self.minimum.fill_(float("inf"))
            wp.launch(
                _min_cost,
                self.samples,
                inputs=[self.costs, self.data.overflow, self.statistics],
                outputs=[self.minimum],
            )
            self.moments.zero_()
            wp.launch(_dial_moments, self.samples, inputs=[self.costs, self.minimum], outputs=[self.moments])
            wp.launch(
                _dial_weights,
                self.samples,
                inputs=[self.costs, self.minimum, self.moments, self.dial_temperature],
                outputs=[self.weights],
            )
            wp.launch(_dial_update, self.plan.shape, inputs=[self.proposals, self.weights], outputs=[self.center])
        wp.copy(self.plan, self.center)
        self.data, self.proposals, self.costs = self.mean_data, self.mean_proposals, self.mean_costs
        self.samples = 1
        self.rollout(q, v, clock)
        wp.copy(self.minimum, self.costs)
        self.best.zero_()
        wp.launch(_finish, 1, inputs=[clock, self.minimum], outputs=[self.last, self.iteration, self.failure_count])
        if self.traces is not None:
            self.traces.finish(self, q, clock)

    def capture(self, q, v, clock):
        super().capture(q, v, clock)
        regular = self.graph
        rounds = self.rounds
        self.rounds = self.initial_rounds
        super().capture(q, v, clock)
        self.initial_graph, self.graph = self.graph, regular
        self.rounds = rounds
        self.started = False

    def solve(self):
        wp.capture_launch(self.graph if self.started else self.initial_graph)
        self.started = True
