# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""GPU predictive sampling: the complete optimizer is one CUDA graph."""

import copy

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


@wp.func
def reference_value(values: wp.array2d[float], t: float, fps: float, column: int):
    frame = wp.clamp(t * fps, 0.0, float(values.shape[0] - 1))
    lower = wp.min(int(frame), values.shape[0] - 2)
    return wp.lerp(values[lower, column], values[lower + 1, column], frame - float(lower))


@wp.func
def reference_rotation(values: wp.array2d[float], t: float, fps: float):
    frame = wp.clamp(t * fps, 0.0, float(values.shape[0] - 1))
    i = wp.min(int(frame), values.shape[0] - 2)
    a = wp.quat(values[i, 4], values[i, 5], values[i, 6], values[i, 3])
    b = wp.quat(values[i + 1, 4], values[i + 1, 5], values[i + 1, 6], values[i + 1, 3])
    return wp.quat_slerp(a, b, frame - float(i))


@wp.func
def reference_velocity(values: wp.array2d[float], t: float, fps: float, column: int):
    value = float(0.0)
    if t * fps < float(values.shape[0] - 1):
        value = reference_value(values, t, fps, column)
    return value


@wp.func
def spline(plan: wp.array2d[float], t: float, spacing: float, j: int):
    phase = wp.clamp(t / spacing, 0.0, float(plan.shape[0] - 1))
    lo = wp.min(int(phase), plan.shape[0] - 2)
    return wp.lerp(plan[lo, j], plan[lo + 1, j], phase - float(lo))


@wp.kernel
def _shift(
    plan: wp.array2d[float], clock: wp.array[float], last: wp.array[float], spacing: float, center: wp.array2d[float]
):
    k, j = wp.tid()
    shift = wp.max(0.0, clock[0] - last[0])
    center[k, j] = spline(plan, float(k) * spacing + shift, spacing, j)


@wp.kernel
def _propose(
    center: wp.array2d[float],
    iteration: wp.array[int],
    seed: int,
    round_index: int,
    scale: float,
    noise: wp.array[float],
    plan: wp.array2d[float],
    blend: bool,
    proposals: wp.array3d[float],
):
    world, k, j = wp.tid()
    value = center[k, j]
    if world == 1:
        value = 0.0
    elif world == 2 and blend:
        value = plan[k, j]
    elif world > 1:
        # Antithetic samples share a random draw with opposite signs.
        index = ((world - 2) // 2 * center.shape[0] + k) * center.shape[1] + j
        rng = wp.rand_init(seed + 7919 * iteration[0] + 104729 * round_index, index)
        sign = float(1.0)
        if world % 2 == 1:
            sign = -1.0
        value += sign * wp.randn(rng) * noise[j] * scale
    proposals[world, k, j] = wp.clamp(value, -0.7, 0.7)


@wp.kernel
def _reset(
    q: wp.array2d[float],
    v: wp.array2d[float],
    clock: wp.array[float],
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    warm: wp.array2d[float],
    times: wp.array[float],
    costs: wp.array[float],
    overflow: wp.array[int],
):
    world, i = wp.tid()
    qpos[world, i] = q[0, i]
    if i < v.shape[1]:
        qvel[world, i] = v[0, i]
        warm[world, i] = 0.0
    if i == 0:
        times[world] = clock[0]
        costs[world] = 0.0
        overflow[world] = 0


@wp.kernel
def _targets(
    proposals: wp.array3d[float],
    qref: wp.array2d[float],
    vref: wp.array2d[float],
    clock: wp.array[float],
    fps: float,
    elapsed: float,
    spacing: float,
    kp: wp.array[float],
    kd: wp.array[float],
    ctrl: wp.array2d[float],
):
    world, j = wp.tid()
    phase = elapsed / spacing
    k = wp.min(int(phase), proposals.shape[1] - 2)
    offset = wp.lerp(proposals[world, k, j], proposals[world, k + 1, j], phase - float(k))
    t = clock[0] + elapsed
    ctrl[world, j] = (
        reference_value(qref, t, fps, j + 7) + offset + kd[j] / kp[j] * reference_velocity(vref, t, fps, j + 6)
    )


@wp.func
def residual_term(value: float, world: int, column: int, weight: float, record: bool, residual: wp.array2d[float]):
    if record:
        residual[world, column] = wp.sqrt(weight) * value
    return value * value


@wp.func
def rotation_error(current: wp.quat, desired: wp.quat):
    rotation = wp.quat_inverse(desired) * current
    sign = float(1.0)
    if rotation[3] < 0.0:
        sign = -1.0
    vector = wp.vec3(rotation[0], rotation[1], rotation[2])
    length = wp.length(vector)
    # Half the SO(3) logarithm retains a useful gradient near a half turn.
    factor = wp.atan2(length, wp.abs(rotation[3])) / wp.max(length, 1.0e-8)
    return sign * factor * vector


@wp.func
def sole_position(position: wp.vec3, rotation: wp.quat):
    center = position + wp.quat_rotate(rotation, wp.vec3(0.035, 0.0, -0.035))
    height = float(1.0e10)
    for i in range(2):
        for j in range(2):
            corner = position + wp.quat_rotate(
                rotation, wp.vec3(-0.05 + 0.17 * float(i), -0.025 + 0.05 * float(j), -0.035)
            )
            height = wp.min(height, corner[2])
    return wp.vec3(center[0], center[1], height)


@wp.kernel
def _score(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    qref: wp.array2d[float],
    vref: wp.array2d[float],
    bodyref: wp.array2d[float],
    rotationref: wp.array2d[float],
    tracked: wp.array[int],
    clock: wp.array[float],
    fps: float,
    elapsed: float,
    weight: float,
    joint_scale: float,
    root_scale: float,
    rotation_scale: float,
    foot_weight: float,
    hand_weight: float,
    hand_clearance: float,
    foot_vertical: float,
    foot_rotation: float,
    angular_weight: float,
    sole_tracking: bool,
    hand_position: float,
    hand_rotation: float,
    joint_velocity: float,
    step: int,
    width: int,
    record: bool,
    residual: wp.array2d[float],
    overflow: wp.array[int],
    nefc: wp.array[int],
    nacon: wp.array[int],
    statistics: wp.array[int],
    costs: wp.array[float],
):
    world = wp.tid()
    t = clock[0] + elapsed
    off = step * width
    cost = float(0.0)
    for j in range(3):
        value = (qpos[world, j] - reference_value(qref, t, fps, j)) / root_scale
        cost += residual_term(value, world, off + j, weight, record, residual)
        value = wp.sqrt(0.1) * (qvel[world, j] - reference_velocity(vref, t, fps, j))
        cost += residual_term(value, world, off + 6 + j, weight, record, residual)
    current = wp.quat(qpos[world, 4], qpos[world, 5], qpos[world, 6], qpos[world, 3])
    desired = reference_rotation(qref, t, fps)
    rotation = rotation_error(current, desired)
    for j in range(3):
        cost += residual_term(rotation[j] / rotation_scale, world, off + 3 + j, weight, record, residual)
    nu = qpos.shape[1] - 7
    for j in range(nu):
        value = (qpos[world, j + 7] - reference_value(qref, t, fps, j + 7)) / (joint_scale * wp.sqrt(float(nu)))
        cost += residual_term(value, world, off + 9 + j, weight, record, residual)
    feet = wp.min(2, tracked.shape[0])
    for k in range(tracked.shape[0]):
        p = xpos[world, tracked[k]]
        if k < 2:
            # MuJoCo Warp stores body quaternions in scalar-first order.
            raw = xquat[world, tracked[k]]
            actual = wp.quat(raw[1], raw[2], raw[3], raw[0])
            if sole_tracking:
                p = sole_position(p, actual)
            for j in range(3):
                scale = foot_weight
                if j == 2:
                    scale *= foot_vertical
                value = wp.sqrt(scale) * (p[j] - reference_value(bodyref, t, fps, 3 * k + j))
                cost += residual_term(value, world, off + 9 + nu + 3 * k + j, weight, record, residual)
            target = wp.quat(
                reference_value(rotationref, t, fps, 4 * k + 1),
                reference_value(rotationref, t, fps, 4 * k + 2),
                reference_value(rotationref, t, fps, 4 * k + 3),
                reference_value(rotationref, t, fps, 4 * k),
            )
            error = rotation_error(actual, wp.normalize(target))
            for j in range(3):
                value = wp.sqrt(foot_rotation) * error[j]
                cost += residual_term(value, world, off + 9 + nu + 3 * feet + 3 * k + j, weight, record, residual)
        else:
            clearance = wp.min(hand_clearance, reference_value(bodyref, t, fps, 3 * k + 2))
            value = wp.sqrt(hand_weight) * wp.max(0.0, clearance - p[2])
            cost += residual_term(value, world, off + 9 + nu + 6 * feet + k - 2, weight, record, residual)
    hands = wp.max(0, tracked.shape[0] - 2)
    actual_omega = wp.quat_rotate(current, wp.vec3(qvel[world, 3], qvel[world, 4], qvel[world, 5]))
    desired_omega = wp.quat_rotate(
        desired,
        wp.vec3(
            reference_velocity(vref, t, fps, 3),
            reference_velocity(vref, t, fps, 4),
            reference_velocity(vref, t, fps, 5),
        ),
    )
    for j in range(3):
        value = wp.sqrt(angular_weight) * (actual_omega[j] - desired_omega[j])
        cost += residual_term(value, world, off + 9 + nu + 6 * feet + hands + j, weight, record, residual)
    extra = off + 12 + nu + 6 * feet + hands
    for j in range(nu):
        value = wp.sqrt(joint_velocity) * (qvel[world, j + 6] - reference_velocity(vref, t, fps, j + 6))
        cost += residual_term(value, world, extra + j, weight, record, residual)
    for k in range(hands):
        body = tracked[k + 2]
        p = xpos[world, body]
        raw = xquat[world, body]
        actual = wp.quat(raw[1], raw[2], raw[3], raw[0])
        target = wp.quat(
            reference_value(rotationref, t, fps, 4 * (k + 2) + 1),
            reference_value(rotationref, t, fps, 4 * (k + 2) + 2),
            reference_value(rotationref, t, fps, 4 * (k + 2) + 3),
            reference_value(rotationref, t, fps, 4 * (k + 2)),
        )
        error = rotation_error(actual, wp.normalize(target))
        for j in range(3):
            value = wp.sqrt(hand_position) * (p[j] - reference_value(bodyref, t, fps, 3 * (k + 2) + j))
            cost += residual_term(value, world, extra + nu + 6 * k + j, weight, record, residual)
            value = wp.sqrt(hand_rotation) * error[j]
            cost += residual_term(value, world, extra + nu + 6 * k + 3 + j, weight, record, residual)
    if not wp.isfinite(cost) or overflow[world] != 0:
        cost = 1.0e20
    costs[world] += weight * cost
    wp.atomic_max(statistics, 3, nefc[world])
    if world == 0:
        wp.atomic_max(statistics, 4, nacon[0])


@wp.kernel
def _min_cost(costs: wp.array[float], overflow: wp.array[int], statistics: wp.array[int], minimum: wp.array[float]):
    i = wp.tid()
    wp.atomic_min(minimum, 0, costs[i])
    wp.atomic_add(statistics, 0, 1)
    if costs[i] >= 1.0e19:
        wp.atomic_add(statistics, 1, 1)
    if overflow[i] & 1536:
        wp.atomic_add(statistics, 2, 1)


@wp.kernel
def _min_index(costs: wp.array[float], minimum: wp.array[float], best: wp.array[int]):
    i = wp.tid()
    if costs[i] == minimum[0]:
        wp.atomic_min(best, 0, i)


@wp.kernel
def _select(
    proposals: wp.array3d[float],
    minimum: wp.array[float],
    best: wp.array[int],
    center: wp.array2d[float],
    plan: wp.array2d[float],
):
    k, j = wp.tid()
    # Keep the shifted last plan if every new rollout is invalid.
    value = center[k, j]
    if minimum[0] < 1.0e19 and best[0] < proposals.shape[0]:
        value = proposals[best[0], k, j]
    center[k, j] = value
    plan[k, j] = value


@wp.kernel
def _finish(
    clock: wp.array[float],
    minimum: wp.array[float],
    last: wp.array[float],
    iteration: wp.array[int],
    failures: wp.array[int],
):
    last[0] = clock[0]
    iteration[0] += 1
    if minimum[0] >= 1.0e19:
        failures[0] += 1


@wp.kernel
def apply_pd(
    q: wp.array2d[float],
    v: wp.array2d[float],
    clock: wp.array[float],
    qref: wp.array2d[float],
    vref: wp.array2d[float],
    fps: float,
    plan: wp.array2d[float],
    last: wp.array[float],
    spacing: float,
    kp: wp.array[float],
    kd: wp.array[float],
    limits: wp.array[float],
    force: wp.array[float],
):
    j = wp.tid()
    target = reference_value(qref, clock[0], fps, j + 7) + spline(plan, clock[0] - last[0], spacing, j)
    velocity = reference_velocity(vref, clock[0], fps, j + 6)
    force[j + 6] = wp.clamp(kp[j] * (target - q[0, j + 7]) + kd[j] * (velocity - v[0, j + 6]), -limits[j], limits[j])


@wp.kernel
def apply_targets(
    clock: wp.array[float],
    qref: wp.array2d[float],
    vref: wp.array2d[float],
    fps: float,
    plan: wp.array2d[float],
    last: wp.array[float],
    spacing: float,
    kp: wp.array[float],
    kd: wp.array[float],
    target: wp.array[float],
):
    j = wp.tid()
    q = reference_value(qref, clock[0], fps, j + 7) + spline(plan, clock[0] - last[0], spacing, j)
    v = reference_velocity(vref, clock[0], fps, j + 6)
    # A position drive embeds the desired velocity in its position input.
    target[j + target.shape[0] - kp.shape[0]] = q + kd[j] / kp[j] * v


class WholeBodyMPC:
    """Annealed predictive sampling with GPU-only search and MuJoCo Warp prediction.

    Bind live MuJoCo Warp state arrays with capture(). Initialization may use
    CPU reference preprocessing; each subsequent solve is one CUDA graph replay.
    """

    def __init__(
        self,
        model,
        kp,
        kd,
        reference,
        *,
        samples=1024,
        horizon=0.5,
        seed=123,
        prediction_dt=0.01,
        rounds=2,
        knots=4,
        noise=0.12,
        joint_scale=0.3,
        root_scale=0.08,
        rotation_scale=0.15,
        foot_weight=0.0,
        foot_vertical=1.0,
        foot_rotation=0.0,
        angular_weight=0.0,
        sole_tracking=False,
        hand_weight=10000.0,
        hand_clearance=0.2,
        hand_position=0.0,
        hand_rotation=0.0,
        joint_velocity=0.0,
        iterations=20,
        temperature=0.2,
        nonfoot_weight=1000.0,
        device="cuda:0",
    ):
        if samples < 2 or rounds < 1 or knots < 2 or prediction_dt <= 0 or horizon < prediction_dt:
            raise ValueError("Invalid MPC sampling or horizon configuration")
        scales = np.array(
            [
                prediction_dt,
                horizon,
                noise,
                joint_scale,
                foot_weight,
                hand_weight,
                hand_clearance,
                temperature,
                nonfoot_weight,
                foot_vertical,
                foot_rotation,
                angular_weight,
                hand_position,
                hand_rotation,
                joint_velocity,
            ]
        )
        if not np.isfinite(scales).all() or noise < 0 or joint_scale <= 0 or np.any(scales[4:] < 0) or iterations < 1:
            raise ValueError("MPC costs, noise and solver settings must be finite and nonnegative")
        if not all(np.isfinite(x) and x > 0 for x in (root_scale, rotation_scale)):
            raise ValueError("Root position and rotation scales must be finite and positive")
        if model.nv != model.nu + 6 or model.nq != model.nv + 1:
            raise ValueError("MPC expects one free root and one actuator per scalar joint")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("Sampling MPC requires a CUDA device")
        self.cpu_model = copy.deepcopy(model)
        m = self.cpu_model
        m.opt.timestep = prediction_dt
        m.opt.iterations = iterations
        m.opt.tolerance = 1e-4
        m.opt.ls_iterations = 50
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_biasprm[:, 2] = -kd
        m.actuator_ctrllimited[:] = False
        m.actuator_forcelimited[:] = True
        m.actuator_forcerange[:] = m.jnt_actfrcrange[1:]
        self.samples, self.rounds, self.seed = samples, rounds, seed
        self.temperature = temperature
        self.hand_position, self.hand_rotation, self.joint_velocity = hand_position, hand_rotation, joint_velocity
        self.hand_clearance = hand_clearance
        self.root_scale, self.rotation_scale = root_scale, rotation_scale
        self.sole_tracking = sole_tracking
        self.nonfoot_weight = nonfoot_weight
        self.foot_vertical, self.foot_rotation, self.angular_weight = foot_vertical, foot_rotation, angular_weight
        self.steps = round(horizon / prediction_dt)
        self.dt, self.spacing, self.fps = prediction_dt, horizon / (knots - 1), reference.fps
        self.joint_scale, self.foot_weight, self.hand_weight = joint_scale, foot_weight, hand_weight
        tracked = []
        for part in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"):
            tracked.extend(i for i in range(m.nbody) if m.body(i).name.endswith(part))
        data = mujoco.MjData(m)
        bodyref = np.zeros((len(reference.qpos), 3 * len(tracked)))
        rotationref = np.zeros((len(reference.qpos), 4 * len(tracked)))
        self.state_width = 2 * m.nu + 12 + 6 * min(2, len(tracked)) + 7 * max(0, len(tracked) - 2)
        self.residual_width = self.state_width + m.nbody
        for i, q in enumerate(reference.qpos):
            data.qpos[:] = q
            mujoco.mj_kinematics(m, data)
            bodyref[i] = data.xpos[tracked].ravel()
            rotationref[i] = data.xquat[tracked].ravel()
            if sole_tracking:
                local = np.array([[x, y, -0.035] for x in (-0.05, 0.12) for y in (-0.025, 0.025)])
                for k, body in enumerate(tracked[:2]):
                    corners = data.xpos[body] + local @ data.xmat[body].reshape(3, 3).T
                    bodyref[i, 3 * k : 3 * k + 3] = [*corners.mean(axis=0)[:2], corners[:, 2].min()]
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(m)
            self.model.opt.warn_overflow = False
            self.data = mjw.make_data(m, nworld=samples, nconmax=48, njmax=192)
            self.qref = wp.array(reference.qpos, dtype=float)
            self.vref = wp.array(reference.velocity, dtype=float)
            self.bodyref = wp.array(bodyref, dtype=float)
            self.rotationref = wp.array(rotationref, dtype=float)
            self.residual = wp.zeros((1, 0))
            self.body_force = wp.zeros((samples, m.nbody))
            self.tracked = wp.array(tracked, dtype=int)
            self.kp, self.kd = wp.array(kp, dtype=float), wp.array(kd, dtype=float)
            self.limits = wp.array(m.jnt_actfrcrange[1:, 1], dtype=float)
            scales = np.full(m.nu, noise)
            scales[15:] *= 0.4
            self.noise = wp.array(scales, dtype=float)
            self.plan, self.center = wp.zeros((knots, m.nu)), wp.zeros((knots, m.nu))
            self.proposals = wp.zeros((samples, knots, m.nu))
            self.costs, self.minimum = wp.zeros(samples), wp.zeros(1)
            self.best = wp.zeros(1, dtype=int)
            self.iteration, self.failure_count = wp.zeros(1, dtype=int), wp.zeros(1, dtype=int)
            self.last = wp.zeros(1)
            self.statistics = wp.zeros(5, dtype=int)
        self.graph = None
        self.save_residuals = False

    def optimize(self, q, v, clock):
        wp.launch(_shift, self.plan.shape, inputs=[self.plan, clock, self.last, self.spacing], outputs=[self.center])
        for r in range(self.rounds):
            scale = 0.35 ** (r / max(1, self.rounds - 1))
            wp.launch(
                _propose,
                self.proposals.shape,
                inputs=[self.center, self.iteration, self.seed, r, scale, self.noise, self.plan, self.temperature > 0],
                outputs=[self.proposals],
            )
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
            if self.temperature > 0 and r < self.rounds - 1:
                wp.launch(
                    _refine,
                    self.plan.shape,
                    inputs=[self.proposals, self.costs, self.minimum, self.temperature],
                    outputs=[self.center],
                )
        wp.launch(_finish, 1, inputs=[clock, self.minimum], outputs=[self.last, self.iteration, self.failure_count])

    def rollout(self, q, v, clock):
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
        for step in range(self.steps):
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
                outputs=[self.data.ctrl],
            )
            mjw.step(self.model, self.data)
            if self.foot_weight or self.hand_weight or self.foot_rotation or self.hand_position or self.hand_rotation:
                mjw.kinematics(self.model, self.data)
            weight = 1.0 / self.steps + float(step == self.steps - 1)
            wp.launch(
                _score,
                self.samples,
                inputs=[
                    self.data.qpos,
                    self.data.qvel,
                    self.data.xpos,
                    self.data.xquat,
                    self.qref,
                    self.vref,
                    self.bodyref,
                    self.rotationref,
                    self.tracked,
                    clock,
                    self.fps,
                    (step + 1) * self.dt,
                    weight,
                    self.joint_scale,
                    self.root_scale,
                    self.rotation_scale,
                    self.foot_weight,
                    self.hand_weight,
                    self.hand_clearance,
                    self.foot_vertical,
                    self.foot_rotation,
                    self.angular_weight,
                    self.sole_tracking,
                    self.hand_position,
                    self.hand_rotation,
                    self.joint_velocity,
                    step,
                    self.residual_width,
                    self.save_residuals,
                    self.residual,
                    self.data.overflow,
                    self.data.nefc,
                    self.data.nacon,
                    self.statistics,
                ],
                outputs=[self.costs],
            )
            self.body_force.zero_()
            if self.nonfoot_weight and self.tracked.shape[0] >= 2:
                wp.launch(
                    _contact_cost,
                    self.data.contact.geom.shape[0],
                    inputs=[
                        self.data.contact.geom,
                        self.data.contact.efc_address,
                        self.data.nacon,
                        self.data.contact.worldid,
                        self.data.contact.dim,
                        self.data.efc.force,
                        self.model.geom_bodyid,
                        self.tracked,
                        int(self.cpu_model.opt.cone),
                    ],
                    outputs=[self.body_force],
                )
            wp.launch(
                _force_score,
                self.samples,
                inputs=[
                    self.body_force,
                    self.nonfoot_weight,
                    step,
                    self.residual_width,
                    self.state_width,
                    weight,
                    self.save_residuals,
                    self.residual,
                ],
                outputs=[self.costs],
            )

    def capture(self, q, v, clock):
        with wp.ScopedDevice(self.device):
            self.optimize(q, v, clock)
            with wp.ScopedCapture() as capture:
                self.optimize(q, v, clock)
            self.graph = capture.graph
            self.plan.zero_()
            self.center.zero_()
            self.iteration.zero_()
            self.failure_count.zero_()
            self.statistics.zero_()
            self.last.zero_()

    def solve(self):
        wp.capture_launch(self.graph)

    def apply(self, data, control, *, native_pd=False):
        if native_pd:
            wp.launch(
                apply_targets,
                self.cpu_model.nu,
                inputs=[
                    data.time,
                    self.qref,
                    self.vref,
                    self.fps,
                    self.plan,
                    self.last,
                    self.spacing,
                    self.kp,
                    self.kd,
                ],
                outputs=[control.joint_target_q],
                device=self.device,
            )
            return
        wp.launch(
            apply_pd,
            self.cpu_model.nu,
            inputs=[
                data.qpos,
                data.qvel,
                data.time,
                self.qref,
                self.vref,
                self.fps,
                self.plan,
                self.last,
                self.spacing,
                self.kp,
                self.kd,
                self.limits,
            ],
            outputs=[control.joint_f],
            device=self.device,
        )


@wp.kernel
def audit_step(
    clock: wp.array[float],
    geom: wp.array[wp.vec2i],
    address: wp.array2d[int],
    nacon: wp.array[int],
    dimensions: wp.array[int],
    cone: int,
    forces: wp.array2d[float],
    bodies: wp.array[int],
    feet: wp.array[int],
    torque: wp.array[float],
    limits: wp.array[float],
    overflow: wp.array[int],
    nefc: wp.array[int],
    row: int,
    audit: wp.array2d[float],
    touched: wp.array[int],
):
    ground = float(0.0)
    nonfoot = float(0.0)
    for i in range(wp.min(nacon[0], geom.shape[0])):
        pair = geom[i]
        a, b = bodies[pair[0]], bodies[pair[1]]
        if a == 0 or b == 0:
            efc = address[i, 0]
            if efc >= 0 and efc < forces.shape[1]:
                f = wp.max(0.0, forces[0, efc])
                if cone == 0 and dimensions[i] > 1:
                    f = float(0.0)
                    for k in range(wp.min(2 * (dimensions[i] - 1), forces.shape[1] - efc)):
                        f += wp.max(0.0, forces[0, efc + k])
                ground += f
                body = wp.max(a, b)
                if body != feet[0] and body != feet[1]:
                    nonfoot += f
                    if f > 1.0:
                        touched[body] = 1
    peak = float(0.0)
    for j in range(limits.shape[0]):
        peak = wp.max(peak, wp.abs(torque[j + 6]) / limits[j])
    audit[row, 0] = clock[0]
    audit[row, 1] = ground
    audit[row, 2] = nonfoot
    audit[row, 3] = peak
    audit[row, 4] = float(overflow[0])
    audit[row, 5] = float(nefc[0])
    audit[row, 6] = float(nacon[0])


@wp.kernel
def _refine(
    proposals: wp.array3d[float],
    costs: wp.array[float],
    minimum: wp.array[float],
    temperature: float,
    center: wp.array2d[float],
):
    k, j = wp.tid()
    total = float(0.0)
    weight_sum = float(0.0)
    for world in range(proposals.shape[0]):
        weight = wp.exp(-(costs[world] - minimum[0]) / temperature)
        total += weight * proposals[world, k, j]
        weight_sum += weight
    if minimum[0] < 1.0e19:
        center[k, j] = total / wp.max(weight_sum, 1.0e-8)


@wp.kernel
def _contact_cost(
    geom: wp.array[wp.vec2i],
    address: wp.array2d[int],
    nacon: wp.array[int],
    worlds: wp.array[int],
    dimensions: wp.array[int],
    forces: wp.array2d[float],
    bodies: wp.array[int],
    feet: wp.array[int],
    cone: int,
    body_force: wp.array2d[float],
):
    i = wp.tid()
    if i < nacon[0]:
        pair = geom[i]
        a, b = bodies[pair[0]], bodies[pair[1]]
        body = wp.max(a, b)
        if (a == 0 or b == 0) and body != feet[0] and body != feet[1]:
            world, efc = worlds[i], address[i, 0]
            if efc >= 0 and efc < forces.shape[1]:
                force = wp.max(0.0, forces[world, efc])
                if cone == 0 and dimensions[i] > 1:
                    force = float(0.0)
                    for k in range(wp.min(2 * (dimensions[i] - 1), forces.shape[1] - efc)):
                        force += wp.max(0.0, forces[world, efc + k])
                if wp.isfinite(force):
                    wp.atomic_add(body_force, world, body, force)


@wp.kernel
def _force_score(
    body_force: wp.array2d[float],
    penalty: float,
    step: int,
    width: int,
    state_width: int,
    weight: float,
    record: bool,
    residual: wp.array2d[float],
    costs: wp.array[float],
):
    world = wp.tid()
    cost = float(0.0)
    for body in range(body_force.shape[1]):
        value = wp.sqrt(penalty) * body_force[world, body] / 300.0
        cost += residual_term(value, world, step * width + state_width + body, weight, record, residual)
    costs[world] += weight * cost
