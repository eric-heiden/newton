# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Sketched Gauss-Newton shooting using MuJoCo Warp PR #1535 adjoints.

Random residual projections approximate J.T @ J; a separate adjoint computes
the exact gradient of the pose/velocity objective. Multiple independently
perturbed starts combine sampling with local updates. Contact-force penalties
remain in candidate acceptance, but are not differentiated by this adapter.
"""

import math

import mujoco_warp as mjw
import warp as wp
from mujoco_warp._src.support import jac_dof

from newton.examples.robot.wbc_mpc import (
    WholeBodyMPC,
    _finish,
    _min_cost,
    _min_index,
    _reset,
    _select,
    _shift,
    _targets,
)


@wp.kernel
def _starts(
    center: wp.array2d[float],
    iteration: wp.array[int],
    seed: int,
    sketch: int,
    noise: float,
    proposals: wp.array3d[float],
):
    world, k, j = wp.tid()
    start = world // (sketch + 1)
    value = center[k, j]
    if start > 0:
        rng = wp.rand_init(seed + 7919 * iteration[0], (start * center.shape[0] + k) * center.shape[1] + j)
        value += noise * wp.pow(0.9, float(center.shape[0] - 1 - k)) * wp.randn(rng)
    proposals[world, k, j] = wp.clamp(value, -0.7, 0.7)


@wp.kernel
def _projection_seed(
    sketch: int, stage: int, seed: int, values: wp.array2d[float], residual: wp.array2d[float], costs: wp.array[float]
):
    world, column = wp.tid()
    row = world % (sketch + 1)
    value = float(0.0)
    if row < sketch:
        rng = wp.rand_init(seed + 3001 * stage, row * residual.shape[1] + column)
        value = (2.0 * float(wp.randi(rng, 0, 2)) - 1.0) / wp.sqrt(float(sketch))
    if row == sketch:
        value = 2.0 * values[world, column]
    residual[world, column] = value
    if column == 0:
        costs[world] = 0.0


@wp.kernel
def _body_pose_vjp(
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    dof_bodyid: wp.array[int],
    ancestors: wp.array2d[int],
    subtree_com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    gp: wp.array2d[wp.vec3],
    gq: wp.array2d[wp.quat],
    dof_gradient: wp.array2d[float],
):
    world, dof = wp.tid()
    value = float(0.0)
    for body in range(1, xpos.shape[1]):
        raw, grad = xquat[world, body], gq[world, body]
        xyz, gxyz = wp.vec3(raw[1], raw[2], raw[3]), wp.vec3(grad[1], grad[2], grad[3])
        omega = 0.5 * (raw[0] * gxyz + wp.cross(xyz, gxyz) - grad[0] * xyz)
        jp, jr = jac_dof(
            body_parentid, body_rootid, dof_bodyid, ancestors, subtree_com, cdof, xpos[world, body], body, dof, world
        )
        value += wp.dot(jp, gp[world, body]) + wp.dot(jr, omega)
    dof_gradient[world, dof] = value


@wp.kernel
def _lift_gradient(qpos: wp.array2d[float], dof_gradient: wp.array2d[float], gradient: wp.array2d[float]):
    world, j = wp.tid()
    if j < 3:
        gradient[world, j] += dof_gradient[world, j]
    elif j >= 7:
        gradient[world, j] += dof_gradient[world, j - 1]
    else:
        q = wp.quat(qpos[world, 4], qpos[world, 5], qpos[world, 6], qpos[world, 3])
        g = wp.quat(dof_gradient[world, 3], dof_gradient[world, 4], dof_gradient[world, 5], 0.0)
        value = 2.0 * q * g
        gradient[world, j] += value[(j - 4 + 4) % 4]


@wp.kernel
def _sum_gradient(local: wp.array2d[float], future: wp.array2d[float], output: wp.array2d[float]):
    w, i = wp.tid()
    output[w, i] = local[w, i] + future[w, i]


@wp.kernel
def _control_vjp(ctrl: wp.array2d[float], elapsed: float, spacing: float, gradient: wp.array3d[float]):
    w, j = wp.tid()
    phase = elapsed / spacing
    k = wp.min(int(phase), gradient.shape[1] - 2)
    a = phase - float(k)
    gradient[w, k, j] += (1.0 - a) * ctrl[w, j]
    gradient[w, k + 1, j] += a * ctrl[w, j]


@wp.kernel
def _sketch_system(
    gradient: wp.array3d[float], sketch: int, size: int, damping: float, matrix: wp.array2d[float], rhs: wp.array[float]
):
    start, i, j = wp.tid()
    count = gradient.shape[1] * gradient.shape[2]
    nu = gradient.shape[2]
    value = float(0.0)
    if i < count and j < count:
        for r in range(sketch):
            w = start * (sketch + 1) + r
            value += gradient[w, i // nu, i % nu] * gradient[w, j // nu, j % nu]
    if i == j:
        value += damping * (value + 1.0)
    matrix[start * size + i, j] = value
    if j == 0:
        rhs[start * size + i] = 0.0
        if i < count:
            rhs[start * size + i] = -0.5 * gradient[start * (sketch + 1) + sketch, i // nu, i % nu]


def _solve_batched(size):
    @wp.kernel(module="unique", enable_backward=False)
    def solve(matrix: wp.array2d[float], rhs: wp.array[float], direction: wp.array2d[float]):
        start = wp.tid()
        factor = wp.tile_load(matrix, shape=(size, size), offset=(start * size, 0), storage="shared")
        wp.tile_cholesky_inplace(factor)
        b = wp.tile_load(rhs, shape=(size,), offset=(start * size,))
        x = wp.tile_cholesky_solve(factor, b)
        wp.tile_store(direction, wp.tile_reshape(x, shape=(1, size)), offset=(start, 0))

    return solve


@wp.kernel
def _step_size(direction: wp.array2d[float], maximum: wp.array[float]):
    start, i = wp.tid()
    wp.atomic_max(maximum, start, wp.abs(direction[start, i]))


@wp.kernel
def _analytic_lines(
    centers: wp.array3d[float],
    direction: wp.array2d[float],
    maximum: wp.array[float],
    sketch: int,
    trust: float,
    proposals: wp.array3d[float],
):
    w, k, j = wp.tid()
    start, line = w // 8, w % 8
    value = centers[start * (sketch + 1), k, j]
    if line == 1:
        value = 0.0
    elif line >= 2:
        delta = direction[start, k * centers.shape[2] + j]
        if wp.isfinite(delta):
            value += wp.pow(0.5, float(line - 2)) * wp.min(1.0, trust / wp.max(maximum[start], 1.0e-8)) * delta
    proposals[w, k, j] = wp.clamp(value, -0.7, 0.7)


class WholeBodyAdjoint(WholeBodyMPC):
    """Optimize random projections of the residual Jacobian with analytic VJPs.

    Requires the pinned experimental MuJoCo Warp PR #1535. The contact set is
    locally frozen by that backend. Only pose/velocity costs are differentiated;
    all candidate evaluations retain the complete non-foot-contact penalty.
    """

    def __init__(self, *args, sketch=16, starts=1, start_noise=0.05, damping=0.1, trust=0.2, **kwargs):
        if not hasattr(mjw, "enable_grad"):
            raise RuntimeError("Analytic MPC requires MuJoCo Warp PR #1535; see g1_wbc.md")
        if sketch < 1 or starts < 1 or not all(math.isfinite(x) and x > 0 for x in (damping, trust)):
            raise ValueError("Invalid analytic MPC sketch, starts, damping, or trust")
        if not math.isfinite(start_noise) or start_noise < 0:
            raise ValueError("Start noise must be finite and nonnegative")
        kwargs["samples"] = starts * (sketch + 1)
        mjw.enable_grad()
        super().__init__(*args, **kwargs)
        self.sketch, self.starts, self.start_noise = sketch, starts, start_noise
        self.damping, self.trust = damping, trust
        self.count = self.plan.size
        if self.count >= 128:
            raise ValueError("The tiled solve supports at most 127 knot parameters")
        self.size = max(16, 1 << self.count.bit_length())
        self.gradient_proposals, self.gradient_costs = self.proposals, self.costs
        self.gradient_data = self.data
        with wp.ScopedDevice(self.device):

            def data(nworld):
                return mjw.make_data(self.cpu_model, nworld=nworld, nconmax=48, njmax=192)

            self.states = [self.data] + [data(self.samples) for _ in range(self.steps)]
            self.task_data = data(self.samples)
            for array in (self.task_data.qpos, self.task_data.qvel, self.task_data.xpos, self.task_data.xquat):
                array.requires_grad = True
            self.residual = wp.zeros((self.samples, self.residual_width), requires_grad=True)
            self.stage_costs = wp.zeros(self.samples, requires_grad=True)
            self.gradient = wp.zeros_like(self.proposals)
            self.future_q, self.future_v = wp.zeros_like(self.data.qpos), wp.zeros_like(self.data.qvel)
            self.dof_gradient = wp.zeros_like(self.data.qvel)
            self.bc = mjw.create_backward_context(self.model, self.data)
            self.line_data = data(starts * 8)
            self.line_proposals = wp.zeros((starts * 8, *self.plan.shape))
            self.line_costs = wp.zeros(starts * 8)
            self.body_force = wp.zeros((max(self.samples, starts * 8), self.cpu_model.nbody))
            self.matrix = wp.zeros((starts * self.size, self.size))
            self.rhs = wp.zeros(starts * self.size)
            self.direction, self.maximum = wp.zeros((starts, self.size)), wp.zeros(starts)
        self.solve_kernel = _solve_batched(self.size)
        self.trace_batches = ((self.gradient_data, self.gradient_costs), (self.line_data, self.line_costs))

    def task_state(self, state):
        """Compute post-step task kinematics without overwriting the IFT cache."""
        self.data = self.task_data
        wp.copy(self.data.qpos, state.qpos)
        wp.copy(self.data.qvel, state.qvel)
        self.data.contact, self.data.efc = state.contact, state.efc
        self.data.nacon, self.data.nefc, self.data.overflow = state.nacon, state.nefc, state.overflow
        mjw.kinematics(self.model, self.data)
        mjw.com_pos(self.model, self.data)

    def differentiate(self, q, v, clock):
        self.samples = self.gradient_proposals.shape[0]
        self.proposals, self.costs = self.gradient_proposals, self.gradient_costs
        self.data = self.states[0]
        wp.launch(
            _reset,
            (self.samples, self.cpu_model.nq),
            inputs=[q, v, clock],
            outputs=[
                self.data.qpos,
                self.data.qvel,
                self.data.qacc_warmstart,
                self.data.time,
                self.costs,
                self.data.overflow,
            ],
        )
        if self.record_traces:
            self.traces.record(self, 0, batch_offset=0)
        tapes = []
        self.save_residuals = False
        for step in range(self.steps):
            before, after = self.states[step : step + 2]
            wp.launch(
                _targets,
                (self.samples, self.cpu_model.nu),
                inputs=[
                    self.proposals,
                    self.qref,
                    self.vref,
                    clock,
                    self.fps,
                    step * self.dt,
                    self.spacing,
                    self.kp,
                    self.kd,
                ],
                outputs=[before.ctrl],
            )
            with wp.Tape() as tape:
                mjw.step(self.model, before, after)
            tapes.append(tape)
            self.task_state(after)
            self.score(clock, step)
            if self.record_traces:
                self.traces.record(self, step + 1, batch_offset=0)
        self.future_q.zero_()
        self.future_v.zero_()
        self.gradient.zero_()
        self.costs, self.save_residuals = self.stage_costs, True
        for step in reversed(range(self.steps)):
            before, after = self.states[step : step + 2]
            self.task_state(after)
            self.stage_costs.zero_()
            self.residual.zero_()
            with wp.Tape() as task_tape:
                self.score(clock, step, pose_only=True, residual_step=0)
            # A fresh kernel tape registers its arrays only on backward().
            # Explicitly clear shared task buffers before seeding this stage.
            for array in (
                self.task_data.qpos,
                self.task_data.qvel,
                self.task_data.xpos,
                self.task_data.xquat,
                self.residual,
                self.stage_costs,
            ):
                array.grad.zero_()
            wp.launch(
                _projection_seed,
                self.residual.shape,
                inputs=[self.sketch, step, self.seed, self.residual],
                outputs=[self.residual.grad, self.stage_costs.grad],
            )
            task_tape.backward()
            m, d = self.model, self.task_data
            wp.launch(
                _body_pose_vjp,
                (self.samples, self.cpu_model.nv),
                inputs=[
                    m.body_parentid,
                    m.body_rootid,
                    m.dof_bodyid,
                    m.body_isdofancestor,
                    d.subtree_com,
                    d.cdof,
                    d.xpos,
                    d.xquat,
                    d.xpos.grad,
                    d.xquat.grad,
                ],
                outputs=[self.dof_gradient],
            )
            wp.launch(_lift_gradient, d.qpos.shape, inputs=[d.qpos, self.dof_gradient], outputs=[d.qpos.grad])
            tapes[step].zero()
            wp.launch(_sum_gradient, d.qpos.shape, inputs=[d.qpos.grad, self.future_q], outputs=[after.qpos.grad])
            wp.launch(_sum_gradient, d.qvel.shape, inputs=[d.qvel.grad, self.future_v], outputs=[after.qvel.grad])
            with mjw.backward_context(self.bc):
                tapes[step].backward()
            wp.launch(
                _control_vjp,
                before.ctrl.shape,
                inputs=[before.ctrl.grad, step * self.dt, self.spacing],
                outputs=[self.gradient],
            )
            wp.copy(self.future_q, before.qpos.grad)
            wp.copy(self.future_v, before.qvel.grad)
        self.save_residuals = False

    def optimize(self, q, v, clock):
        wp.launch(_shift, self.plan.shape, inputs=[self.plan, clock, self.last, self.spacing], outputs=[self.center])
        for r in range(self.rounds):
            self.record_traces = self.traces is not None and r == self.rounds - 1
            wp.launch(
                _starts,
                self.gradient_proposals.shape,
                inputs=[self.center, self.iteration, self.seed, self.sketch, self.start_noise],
                outputs=[self.gradient_proposals],
            )
            self.differentiate(q, v, clock)
            wp.launch(
                _sketch_system,
                (self.starts, self.size, self.size),
                inputs=[self.gradient, self.sketch, self.size, self.damping],
                outputs=[self.matrix, self.rhs],
            )
            wp.launch_tiled(
                self.solve_kernel,
                dim=self.starts,
                inputs=[self.matrix, self.rhs],
                outputs=[self.direction],
                block_dim=128,
            )
            self.maximum.zero_()
            wp.launch(_step_size, self.direction.shape, inputs=[self.direction], outputs=[self.maximum])
            wp.launch(
                _analytic_lines,
                self.line_proposals.shape,
                inputs=[self.gradient_proposals, self.direction, self.maximum, self.sketch, self.trust],
                outputs=[self.line_proposals],
            )
            self.data, self.proposals, self.costs = self.line_data, self.line_proposals, self.line_costs
            self.samples = self.proposals.shape[0]
            self.rollout(q, v, clock)
            self.minimum.fill_(float("inf"))
            self.best.fill_(self.samples)
            wp.launch(
                _min_cost,
                self.samples,
                inputs=[self.costs, self.data.overflow, self.statistics],
                outputs=[self.minimum],
            )
            wp.launch(_min_index, self.samples, inputs=[self.costs, self.minimum], outputs=[self.best])
            wp.launch(
                _select,
                self.plan.shape,
                inputs=[self.proposals, self.minimum, self.best],
                outputs=[self.center, self.plan],
            )
        wp.launch(_finish, 1, inputs=[clock, self.minimum], outputs=[self.last, self.iteration, self.failure_count])
        if self.traces is not None:
            self.traces.finish(self, q, clock)
