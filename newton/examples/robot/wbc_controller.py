# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental contact-aware inverse-dynamics QP for a floating-base robot."""

import mujoco
import numpy as np
from scipy import sparse
from scipy.signal import butter, sosfiltfilt


class MotionReference:
    """Sample a z-up MuJoCo qpos CSV (root quaternion is wxyz) in seconds."""

    def __init__(self, model, qpos, *, fps=30.0):
        self.model = model
        self.qpos = np.asarray(qpos, dtype=float).copy()
        if self.qpos.ndim != 2 or self.qpos.shape[1] != model.nq or len(self.qpos) < 2:
            raise ValueError(f"Expected at least two rows of {model.nq} qpos values")
        if not np.isfinite(self.qpos).all() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Motion and fps must be finite; fps must be positive")
        norms = np.linalg.norm(self.qpos[:, 3:7], axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("Root quaternions must be nonzero")
        self.qpos[:, 3:7] /= norms[:, None]
        self.fps = fps
        self.duration = (len(self.qpos) - 1) / fps
        self.velocity = np.zeros((len(qpos), model.nv))
        for i in range(len(qpos)):
            lo, hi = max(0, i - 1), min(len(qpos) - 1, i + 1)
            mujoco.mj_differentiatePos(model, self.velocity[i], (hi - lo) / fps, self.qpos[lo], self.qpos[hi])
        self.acceleration = np.gradient(self.velocity, 1 / fps, axis=0)

    def sample(self, t):
        if t >= self.duration:
            return self.qpos[-1].copy(), np.zeros(self.model.nv), np.zeros(self.model.nv)
        t = np.clip(t, 0, self.duration) * self.fps
        i = min(int(t), len(self.qpos) - 2)
        a = t - i
        q = self.qpos[i].copy()
        delta = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, delta, 1.0, q, self.qpos[i + 1])
        mujoco.mj_integratePos(self.model, q, delta, a)
        v = (1 - a) * self.velocity[i] + a * self.velocity[i + 1]
        acc = (1 - a) * self.acceleration[i] + a * self.acceleration[i + 1]
        return q, v, acc


def measure_foot_tracking(model, reference, times, poses, points):
    """Measure sole tracking offline, without changing the simulated trajectory.

    Each foot is identified by its body in ``points``. Clearance is the lowest
    supplied sole corner. Reference clearance above 3 cm defines swing; actual
    clearance above 2 cm counts as lifted. Values are in metres and seconds.
    """
    bodies = sorted({body for body, _ in points})
    data = mujoco.MjData(model)
    measured, desired = [], []
    for t, q in zip(times, poses, strict=True):
        for configuration, output in ((q, measured), (reference.sample(t)[0], desired)):
            data.qpos[:] = configuration
            mujoco.mj_kinematics(model, data)
            feet = []
            for body in bodies:
                local = np.array([p for b, p in points if b == body])
                world = data.xpos[body] + local @ data.xmat[body].reshape(3, 3).T
                feet.append(np.r_[world.mean(axis=0), world[:, 2].min()])
            output.append(feet)
    measured, desired = np.asarray(measured), np.asarray(desired)
    swing = desired[:, :, 3] > 0.03
    stance = desired[:, :, 3] < 0.01
    return {
        "foot_position_rmse": float(np.sqrt(np.mean(np.sum((measured[:, :, :3] - desired[:, :, :3]) ** 2, axis=2)))),
        "foot_height_rmse": float(np.sqrt(np.mean((measured[:, :, 3] - desired[:, :, 3]) ** 2))),
        "swing_height_rmse": float(np.sqrt(np.mean((measured[:, :, 3][swing] - desired[:, :, 3][swing]) ** 2)))
        if swing.any()
        else None,
        "swing_recall": float(np.mean(measured[:, :, 3][swing] > 0.02)) if swing.any() else None,
        "false_lift_fraction": float(np.mean(measured[:, :, 3][stance] > 0.03)) if stance.any() else None,
        "reference_swing_fraction": float(swing.mean()),
    }


def measure_motion_tracking(model, reference, times, poses):
    """Offline wrist errors and motion diagnostics, independent of the optimizer.

    Velocity error differentiates the recorded joint-angle error at the median
    recording interval. Oscillation is its zero-phase fourth-order Butterworth
    high-pass component above 6 Hz, omitting 0.1 s at each end. These diagnostics
    supplement pose/contact errors; they are not a perceptual quality score.
    """
    bodies = [
        i for i in range(model.nbody) if model.body(i).name.endswith(("left_wrist_yaw_link", "right_wrist_yaw_link"))
    ]
    if not bodies or len(times) < 25:
        return {}
    desired = np.array([reference.sample(t)[0] for t in times])
    data = mujoco.MjData(model)
    positions = []
    for configurations in (poses, desired):
        values = []
        for q in configurations:
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            values.append(data.xpos[bodies].copy())
        positions.append(np.asarray(values))
    error = positions[0] - positions[1]
    joint = np.asarray(poses)[:, 7:] - desired[:, 7:]
    dt = float(np.median(np.diff(times)))
    velocity = np.gradient(joint, dt, axis=0)[2:-2]
    trim = max(1, round(0.1 / dt))
    highpass = sosfiltfilt(butter(4, 6, fs=1 / dt, btype="highpass", output="sos"), joint, axis=0)[trim:-trim]
    return {
        "hand_position_rmse": float(np.sqrt(np.mean(np.sum(error**2, axis=2)))),
        "hand_position_p95": float(np.percentile(np.linalg.norm(error, axis=2), 95)),
        "joint_velocity_error_rms": float(np.sqrt(np.mean(velocity**2))),
        "joint_highpass_rms": float(np.sqrt(np.mean(highpass**2))),
        "arm_highpass_rms": float(np.sqrt(np.mean(highpass[:, 15:] ** 2))),
        "arm_joint_rmse": float(np.sqrt(np.mean(joint[:, 15:] ** 2))),
        "leg_joint_rmse": float(np.sqrt(np.mean(joint[:, :12] ** 2))),
    }


class WholeBodyQP:
    """Optimize accelerations and unilateral friction-limited support forces.

    The first six generalized velocities must form a free joint. All remaining
    joints are actuated hinges. The output is torque in N m, with no base wrench.
    Contact points are supplied as (body index, body-local position in m).
    """

    def __init__(self, model, points, torque_limits, *, friction=0.6):
        self.model = model
        self.points = points
        self.limits = np.asarray(torque_limits)
        self.friction = friction
        self.nv = model.nv
        self.nf = 3 * len(points)
        self.nx = self.nv + self.nf
        self.force_scale = model.body_mass.sum() * 9.81
        self.data = mujoco.MjData(model)
        self.reference = mujoco.MjData(model)
        self.future = mujoco.MjData(model)
        self.mass = np.zeros((self.nv, self.nv))
        self.qp = None
        self.failures = 0
        self.last_solution = np.zeros(self.nx)
        self.status = "uninitialized"
        self.residual = np.inf
        self.kp = model.dof_armature[6:] * (2 * np.pi * 10) ** 2
        self.kd = 4 * model.dof_armature[6:] * (2 * np.pi * 10)

    def kinematics(self, data):
        positions, jacobians = [], []
        for body, local in self.points:
            p = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
            j = np.zeros((3, self.nv))
            mujoco.mj_jac(self.model, data, j, None, p, body)
            positions.append(p)
            jacobians.append(j)
        return np.array(positions), np.array(jacobians)

    def solve(self, q, v, target, *, dynamics=None):
        qref, vref, aref = target
        d, r, future = self.data, self.reference, self.future
        d.qpos[:], d.qvel[:] = q, v
        r.qpos[:], r.qvel[:] = qref, vref
        mujoco.mj_forward(self.model, d)
        mujoco.mj_forward(self.model, r)
        positions, jac = self.kinematics(d)
        pref, jref = self.kinematics(r)
        eps = 1e-4
        future.qpos[:], future.qvel[:] = q, v
        mujoco.mj_integratePos(self.model, future.qpos, v, eps)
        mujoco.mj_forward(self.model, future)
        _, jnext = self.kinematics(future)
        jdotv = ((jnext - jac) / eps) @ v
        # Reference foot accelerations include the convective Jacobian term.
        future.qpos[:], future.qvel[:] = qref, vref
        mujoco.mj_integratePos(self.model, future.qpos, vref, eps)
        mujoco.mj_forward(self.model, future)
        _, jnext = self.kinematics(future)
        foot_acc = jref @ aref + ((jnext - jref) / eps) @ vref
        if dynamics is None:
            mujoco.mj_fullM(self.model, d, self.mass)
            mass, bias = self.mass, d.qfrc_bias - d.qfrc_passive
        else:
            mass, bias = dynamics
        jt = jac.reshape(-1, self.nv).T * self.force_scale
        # Forces are normalized by body weight to improve QP conditioning.
        dynamics_matrix = np.column_stack((mass, -jt))
        p = np.eye(self.nx) * 1e-5
        p[self.nv :, self.nv :] *= 100
        linear = np.zeros(self.nx)

        def task(j, desired, weight):
            a = np.zeros((len(desired), self.nx))
            a[:, : self.nv] = j
            p[:] += weight * a.T @ a
            linear[:] -= weight * a.T @ desired

        error = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.model, error, 1.0, q, qref)
        desired = aref + 100 * error + 20 * (vref - v)
        task(np.eye(self.nv)[6:], desired[6:], 1.0)
        task(np.eye(self.nv)[:3], desired[:3], 80.0)
        task(np.eye(self.nv)[3:6], desired[3:6], 40.0)
        support = (pref[:, 2] < 0.035) & (positions[:, 2] < 0.065)
        for i in range(len(self.points)):
            if support[i]:
                acc = -40 * (jac[i] @ v) - jdotv[i]
                acc[2] -= 400 * positions[i, 2]
                task(jac[i], acc, 1000.0)
            else:
                acc = foot_acc[i] + 400 * (pref[i] - positions[i]) + 40 * (jref[i] @ vref - jac[i] @ v) - jdotv[i]
                task(jac[i], acc, 10.0)
        # Floating-base dynamics are hard equalities; actuator torques are bounded.
        rows = [dynamics_matrix]
        lower = [-bias - np.r_[np.zeros(6), self.limits]]
        upper = [-bias + np.r_[np.zeros(6), self.limits]]
        for i, active in enumerate(support):
            cone = np.zeros((5, self.nx))
            col = self.nv + 3 * i
            mu = self.friction / np.sqrt(2.0)  # Inscribed square in the circular friction cone.
            cone[:, col : col + 3] = [[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu], [0, 0, 1]]
            rows.append(cone)
            lower.append(np.array([-np.inf] * 4 + [0]))
            upper.append(np.array([0] * 4 + [2.0 if active else 0]))
        a = np.vstack(rows)
        lo, hi = np.concatenate(lower), np.concatenate(upper)
        scale = np.r_[np.full(6, self.force_scale), self.limits]
        a[: self.nv] /= scale[:, None]
        lo[: self.nv] /= scale
        hi[: self.nv] /= scale
        # Keep the dense sparsity pattern across contact switches for warm starts.
        if self.qp is None:
            try:
                import osqp  # noqa: PLC0415 - optional dependency used only by the QP
            except ImportError as exc:
                raise ImportError("The QP controller needs OSQP: uv sync --extra wbc") from exc
            self.pcols = np.repeat(np.arange(self.nx), np.arange(1, self.nx + 1))
            self.prows = np.concatenate([np.arange(i + 1) for i in range(self.nx)])
            ps = sparse.csc_matrix((p[self.prows, self.pcols], (self.prows, self.pcols)), shape=p.shape)
            acols = np.repeat(np.arange(self.nx), a.shape[0])
            arows = np.tile(np.arange(a.shape[0]), self.nx)
            aa = sparse.csc_matrix((a.T.ravel(), (arows, acols)), shape=a.shape)
            self.qp = osqp.OSQP()
            self.qp.setup(
                P=ps,
                q=linear,
                A=aa,
                l=lo,
                u=hi,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000,
                polishing=True,
            )
        else:
            self.qp.update(Px=p[self.prows, self.pcols], q=linear, Ax=a.T.ravel(), l=lo, u=hi)
        result = self.qp.solve(raise_error=False)
        self.status = result.info.status
        self.residual = np.inf
        if result.x is not None and np.isfinite(result.x).all():
            self.residual = float(max(np.max(lo - a @ result.x), np.max(a @ result.x - hi), 0))
        if result.info.status_val in (1, 2) and self.residual < 0.005:
            x = result.x
            self.last_solution = x
            torque = (dynamics_matrix @ x + bias)[6:]
        else:
            self.failures += 1
            self.last_solution[:] = 0
            torque = self.kp * error[6:] + self.kd * (vref[6:] - v[6:])
        return np.clip(torque, -self.limits, self.limits)
