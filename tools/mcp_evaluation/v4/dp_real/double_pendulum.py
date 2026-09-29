# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Torque-driven double pendulum replaying recorded motor torques from a real robot.

The recordings come from the DFKI underactuated double pendulum (design C.0,
system-identification runs). Each CSV row holds ``time [s], q1, q2 [rad],
qd1, qd2 [rad/s], tau1, tau2 [N m]``, with ``q = 0`` hanging straight down and
``tau`` the torque measured at each joint's motor output.

The model is evaluated by multiple shooting: the recording is cut into short
windows, every window becomes one Newton world that starts from the measured
state and is driven open loop by the measured torques, and the predicted joint
angles are compared with the recording.

Run: ``python double_pendulum.py --viewer null --trajectory train_07.csv``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

# ---------------- Model parameters (tunable) ----------------
# Nominal values from the CAD geometry: point masses at the link ends.
L1 = 0.2  # shoulder-to-elbow distance [m] (measured, fixed)
PARAMS = {
    "m1": 0.6,  # link 1 mass [kg]
    "m2": 0.6,  # link 2 mass [kg]
    "r1": 0.2,  # link 1 center of mass distance from the shoulder [m]
    "r2": 0.3,  # link 2 center of mass distance from the elbow [m]
    "I1": 0.0,  # link 1 inertia about its center of mass, around the joint axis [kg m^2]
    "I2": 0.0,  # link 2 inertia about its center of mass, around the joint axis [kg m^2]
    "armature1": 0.0,  # reflected motor inertia [kg m^2]
    "armature2": 0.0,
    "damping1": 0.0,  # viscous joint damping [N m s/rad]
    "damping2": 0.0,
    "friction1": 0.0,  # Coulomb joint friction [N m]
    "friction2": 0.0,
}
# ------------------------------------------------------------

SIM_DT = 0.002  # [s]
HORIZON = 0.5  # window length [s]
STRIDE = 0.25  # spacing between window starts [s]


def load_trajectory(path) -> np.ndarray:
    """Recording as an array of rows ``(t, q1, q2, qd1, qd2, tau1, tau2)``."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def add_pendulum(builder: newton.ModelBuilder, params: dict | None = None) -> None:
    """Add one double pendulum hanging from a fixed shoulder at the origin."""
    p = PARAMS if params is None else params
    axis = newton.Axis.Y
    # Thin-rod inertia tensor (symmetric about the link's long axis z); only the
    # component around the joint axis (y) affects the planar motion.
    for i, (parent_offset, com, mass, inertia) in enumerate(
        (((0.0, 0.0, 0.0), p["r1"], p["m1"], p["I1"]), ((0.0, 0.0, -L1), p["r2"], p["m2"], p["I2"]))
    ):
        body = builder.add_link(
            xform=wp.transform_identity(),
            mass=mass,
            com=wp.vec3(0.0, 0.0, -com),
            inertia=wp.mat33(inertia + 1e-6, 0.0, 0.0, 0.0, inertia + 1e-6, 0.0, 0.0, 0.0, 1e-6),
            label=f"link{i + 1}",
        )
        builder.add_shape_capsule(
            body,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * (L1 if i == 0 else 0.3)), wp.quat_identity()),
            radius=0.01,
            half_height=0.5 * (L1 if i == 0 else 0.3),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
        )
        joint = builder.add_joint_revolute(
            parent=-1 if i == 0 else body - 1,
            child=body,
            axis=axis,
            parent_xform=wp.transform(wp.vec3(*parent_offset), wp.quat_identity()),
            armature=p[f"armature{i + 1}"],
            damping=p[f"damping{i + 1}"],
            friction=p[f"friction{i + 1}"],
            target_ke=0.0,
            target_kd=0.0,
            label=f"joint{i + 1}",
        )
        if i == 1:
            builder.add_articulation([joint - 1, joint])


def build_model(num_worlds: int, params: dict | None = None) -> newton.Model:
    """Model with ``num_worlds`` independent copies of the pendulum."""
    pendulum = newton.ModelBuilder()
    add_pendulum(pendulum, params)
    builder = newton.ModelBuilder()
    builder.replicate(pendulum, num_worlds)
    return builder.finalize()


def make_solver(model: newton.Model) -> newton.solvers.SolverBase:
    return newton.solvers.SolverMuJoCo(model, integrator="implicitfast", disable_contacts=True)


def windows(data: np.ndarray, horizon: float = HORIZON, stride: float = STRIDE) -> np.ndarray:
    """Start times [s] of the evaluation windows that fit inside the recording."""
    return np.arange(data[0, 0], data[-1, 0] - horizon, stride)


def rollout(model, solver, data: np.ndarray, starts: np.ndarray, horizon: float = HORIZON):
    """Simulate one window per world; returns the times [s] and predicted angles [rad].

    Returns:
        ``(times, q)`` with ``times`` of shape ``[steps + 1]`` relative to each
        window start and ``q`` of shape ``[steps + 1, num_worlds, 2]``.
    """
    steps = round(horizon / SIM_DT)
    state_0, state_1, control = model.state(), model.state(), model.control()
    start_rows = np.array([np.searchsorted(data[:, 0], s) for s in starts])
    q0 = data[start_rows, 1:3]
    qd0 = data[start_rows, 3:5]
    state_0.joint_q.assign(q0.reshape(-1).astype(np.float32))
    state_0.joint_qd.assign(qd0.reshape(-1).astype(np.float32))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    times = np.arange(steps + 1) * SIM_DT
    # Zero-order hold on the recorded torque samples.
    rows = np.searchsorted(data[:, 0], data[start_rows, 0][None, :] + times[:-1, None], side="right") - 1
    torques = data[rows, 5:7].reshape(steps, -1).astype(np.float32)
    q = [q0]
    for k in range(steps):
        control.joint_f.assign(torques[k])
        solver.step(state_0, state_1, control, None, SIM_DT)
        state_0, state_1 = state_1, state_0
        q.append(state_0.joint_q.numpy().reshape(-1, 2))
    return times, np.asarray(q)


def window_errors(data: np.ndarray, starts: np.ndarray, times: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Angle error [rad] of each window, RMS over time and both joints."""
    t0 = data[np.array([np.searchsorted(data[:, 0], s) for s in starts]), 0]
    measured = np.stack(
        [np.stack([np.interp(t0 + times, data[:, 0], data[:, 1 + j]) for j in range(2)], axis=-1) for t0 in t0],
        axis=1,
    )
    return np.sqrt(np.mean((q - measured) ** 2, axis=(0, 2)))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        paths = args.trajectory or sorted(str(p) for p in Path(__file__).parent.glob("train_*.csv"))
        self.trajectories = {Path(p).stem: load_trajectory(p) for p in paths}
        self.data = self.trajectories[Path(paths[0]).stem]
        self.starts = windows(self.data)
        self.model = build_model(len(self.starts))
        self.solver = make_solver(self.model)
        self.frame_dt = HORIZON
        self.sim_dt = SIM_DT
        self.sim_time = 0.0
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.viewer.set_model(self.model)

    def evaluate(self) -> dict:
        """Mean window RMSE [rad] for every loaded trajectory."""
        result = {}
        for name, data in self.trajectories.items():
            starts = windows(data)
            model = build_model(len(starts))
            times, q = rollout(model, make_solver(model), data, starts)
            result[name] = float(np.mean(window_errors(data, starts, times, q)))
        return result

    def step(self):
        _, q = rollout(self.model, self.solver, self.data, self.starts)
        self.state_0.joint_q.assign(q[-1].reshape(-1).astype(np.float32))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        errors = self.evaluate()
        assert all(np.isfinite(e) for e in errors.values())

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--trajectory", action="append", help="Recording CSV (repeatable)")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    print(example.evaluate())
