# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Small derivative-free full-body MPC using MuJoCo's threaded rollouts."""

import copy

import mujoco
import numpy as np
from mujoco import rollout


class WholeBodyMPC:
    """Optimize a warm-started spline of PD target offsets without contact labels."""

    def __init__(
        self, model, kp, kd, reference, *, samples=128, horizon=0.5, seed=123, threads=4, prediction_dt=0.01, rounds=2
    ):
        self.model = copy.deepcopy(model)
        self.model.opt.timestep = prediction_dt
        self.model.opt.iterations = 20
        self.model.opt.ls_iterations = 10
        self.reference = reference
        self.kp, self.kd = kp, kd
        self.samples = samples
        self.rounds = rounds
        if samples < 2 or rounds < 1 or prediction_dt <= 0 or horizon < prediction_dt:
            raise ValueError("MPC needs >=2 samples, >=1 round, and horizon >= positive prediction_dt")
        self.horizon = horizon
        self.steps = round(horizon / self.model.opt.timestep)
        self.rng = np.random.default_rng(seed)
        self.knots = np.zeros((4, model.nu))
        self.times = np.arange(self.steps) * self.model.opt.timestep
        self.knot_times = np.linspace(0, horizon, len(self.knots))
        self.basis = np.array([np.interp(self.times, self.knot_times, np.eye(4)[i]) for i in range(4)]).T
        self.noise_scale = np.ones(model.nu) * 0.12
        self.noise_scale[15:] = 0.04
        self.data = [mujoco.MjData(self.model) for _ in range(threads)]
        self.runner = rollout.Rollout(nthread=threads)
        self.model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp
        self.model.actuator_biasprm[:, 2] = -kd
        self.model.actuator_ctrllimited[:] = False
        self.model.actuator_forcelimited[:] = True
        self.model.actuator_forcerange[:] = self.model.jnt_actfrcrange[1:]
        self.last_time = None
        self.best_cost = np.inf
        self.failures = 0

    def solve(self, q, v, t):
        if self.last_time is not None:
            shift = t - self.last_time
            self.knots = np.array(
                [
                    np.interp(self.knot_times + shift, self.knot_times, self.knots[:, i])
                    for i in range(self.knots.shape[1])
                ]
            ).T
        self.last_time = t
        reference = [self.reference.sample(t + s) for s in self.times]
        qref = np.array([x[0] for x in reference])
        vref = np.array([x[1] for x in reference])
        nominal = qref[:, 7:] + self.kd / self.kp * vref[:, 6:]
        # Rollouts return the state AFTER each step, while controls act before it.
        future = [self.reference.sample(t + s + self.model.opt.timestep) for s in self.times]
        qref = np.array([x[0] for x in future])
        vref = np.array([x[1] for x in future])
        initial = np.r_[0.0, q, v]
        for scale in np.geomspace(1.0, 0.35, self.rounds):
            proposals = self.knots + self.rng.normal(size=(self.samples, *self.knots.shape)) * self.noise_scale * scale
            proposals[0] = self.knots
            proposals[1] = 0.0
            offsets = np.einsum("tk,nkj->ntj", self.basis, proposals)
            control = nominal[None] + offsets
            states, _ = self.runner.rollout(self.model, self.data, initial[None], control, chunk_size=1)
            predicted = states[:, :, 1 : 1 + self.model.nq]
            velocity = states[:, :, 1 + self.model.nq :]
            # Dimensionless squared tracking costs with explicit physical scales.
            cost = np.sum((predicted[:, :, :3] - qref[None, :, :3]) ** 2, axis=2) / 0.1**2
            dot = np.sum(predicted[:, :, 3:7] * qref[None, :, 3:7], axis=2)
            cost += (1 - np.clip(dot * dot, 0, 1)) / 0.15**2
            cost += np.mean((predicted[:, :, 7:] - qref[None, :, 7:]) ** 2, axis=2) / 0.3**2
            cost += 0.1 * np.sum((velocity[:, :, :3] - vref[None, :, :3]) ** 2, axis=2)
            cost += 0.05 * np.mean(offsets**2, axis=2)
            total = np.mean(cost, axis=1) + cost[:, -1]
            total[~np.isfinite(total)] = np.inf
            # MuJoCo can reset unstable simulations to finite states. Reject them.
            valid_time = np.all(np.isclose(states[:, :, 0], self.times + self.model.opt.timestep, atol=1e-7), axis=1)
            total[~valid_time] = np.inf
            best = int(np.argmin(total))
            if not np.isfinite(total[best]):
                self.knots[:] = 0
                self.failures += 1
                break
            self.knots = proposals[best]
            self.best_cost = float(total[best])
        return self.knots[0].copy()
