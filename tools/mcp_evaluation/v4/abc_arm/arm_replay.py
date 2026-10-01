# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Real YAM arm replaying its teleoperation commands from ABC-130k logs.

The logs come from ABC-130k (https://abc.bot/#data): bimanual YAM stations,
each 6-DoF arm teleoperated by streaming joint-position commands from a leader
arm at about 30 Hz. ``logs/*.csv`` hold one arm each, with rows ``time [s],
q1..q6 [rad], qd1..qd6 [rad/s], tau1..tau6 [N m], cmd1..cmd6 [rad], grip,
grip_cmd``: measured joint positions, velocities, and motor torques, the
commanded joint positions, and the measured and commanded gripper opening
(0 closed, 1 open). Both arms are the same model (``yam_arm.xml``).

The model is evaluated by multiple shooting: every log is cut into short
windows, every window becomes one Newton world that starts from the measured
state and is driven open loop by the logged commands through the arm's
position controller, and the predicted joint angles are compared with the log.

Run: ``python arm_replay.py --viewer null --log logs/01_left_distribute_texas_hold_em_gaming_equipmen.csv``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

HERE = Path(__file__).resolve().parent
ARM = HERE / "yam_arm.xml"
ARM_JOINTS = [f"left_joint{j}" for j in range(1, 7)]
GRIPPER_TRAVEL = 0.0475  # finger travel [m] at a gripper opening of 1

# ---------------- Model parameters (tunable) ----------------
# Nominal values from the ABC simulation model (abc_sim, MJCF position actuators).
PARAMS = {
    "kp": [40.0, 40.0, 40.0, 20.0, 10.0, 10.0],  # position gain per joint [N m/rad]
    "kd": [2.5, 2.5, 2.5, 0.5, 1.0, 1.0],  # velocity gain per joint [N m s/rad]
    "armature": [0.032, 0.032, 0.032, 0.0018, 0.0018, 0.0018],  # reflected rotor inertia [kg m^2]
    "friction": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Coulomb joint friction [N m]
    "damping": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # viscous joint damping [N m s/rad]
    "effort_limit": [28.0, 28.0, 28.0, 10.0, 10.0, 10.0],  # actuator torque limit [N m]
    "command_delay": 0.0,  # latency from a logged command to the controller [s]
}
# ------------------------------------------------------------

SIM_DT = 0.002  # [s]
HORIZON = 1.0  # window length [s]
STRIDE = 0.5  # spacing between window starts [s]
COLUMNS = {"time": 0, "q": slice(1, 7), "qd": slice(7, 13), "tau": slice(13, 19), "cmd": slice(19, 25), "grip": 25}
COLUMNS["grip_cmd"] = 26


def load_log(path) -> np.ndarray:
    """Arm log as an array of rows (see the module docstring for the columns)."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def build_model(num_worlds: int, params: dict | None = None) -> newton.Model:
    """Model with ``num_worlds`` independent copies of the arm, mounted at the origin."""
    p = PARAMS if params is None else params
    arm = newton.ModelBuilder()
    arm.add_mjcf(str(ARM))
    starts = arm.joint_qd_start
    for j, name in enumerate(ARM_JOINTS):
        dof = starts[[label.rsplit("/", 1)[-1] for label in arm.joint_label].index(name)]
        arm.joint_target_ke[dof] = p["kp"][j]
        arm.joint_target_kd[dof] = p["kd"][j]
        arm.joint_armature[dof] = p["armature"][j]
        arm.joint_friction[dof] = p["friction"][j]
        arm.joint_damping[dof] = p["damping"][j]
        arm.joint_effort_limit[dof] = p["effort_limit"][j]
    builder = newton.ModelBuilder()
    builder.replicate(arm, num_worlds)
    return builder.finalize()


def make_solver(model: newton.Model) -> newton.solvers.SolverBase:
    return newton.solvers.SolverMuJoCo(model, integrator="implicitfast", disable_contacts=True)


def windows(data: np.ndarray, horizon: float = HORIZON, stride: float = STRIDE) -> np.ndarray:
    """Start times [s] of the evaluation windows that fit inside the log."""
    return np.arange(data[0, 0], data[-1, 0] - horizon, stride)


def _coordinates(model: newton.Model) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-world coordinate indices of the six arm joints and the two finger slides, and coordinates per world."""
    names = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    starts = model.joint_q_start.numpy()
    arm = np.array([starts[names.index(name)] for name in ARM_JOINTS])
    fingers = np.array([starts[names.index(name)] for name in ("left_left_finger", "left_right_finger")])
    return arm, fingers, model.joint_coord_count // model.world_count


def rollout(
    model, solver, segments: list[tuple[np.ndarray, float]], params: dict | None = None, horizon: float = HORIZON
):
    """Simulate one window per world; ``segments`` lists ``(log, start time)`` per world.

    Returns:
        ``(times, q)`` with ``times`` of shape ``[steps + 1]`` relative to each
        window start and ``q`` of shape ``[steps + 1, num_worlds, 6]`` (arm joints [rad]).
    """
    p = PARAMS if params is None else params
    steps = round(horizon / SIM_DT)
    arm, fingers, coords = _coordinates(model)
    worlds = len(segments)
    state_0, state_1, control = model.state(), model.state(), model.control()
    q0 = model.joint_q.numpy().reshape(worlds, coords)
    qd0 = np.zeros((worlds, model.joint_dof_count // worlds), dtype=np.float32)
    times = np.arange(steps + 1) * SIM_DT
    targets = np.zeros((steps, worlds, coords), dtype=np.float32)
    targets[:] = q0[None]
    for w, (data, start) in enumerate(segments):
        row = np.searchsorted(data[:, 0], start)
        q0[w, arm] = data[row, COLUMNS["q"]]
        qd0[w, arm] = data[row, COLUMNS["qd"]]
        opening = np.clip(data[row, COLUMNS["grip"]], 0.0, 1.0) * GRIPPER_TRAVEL
        q0[w, fingers] = (opening, -opening)
        # Zero-order hold on the logged commands, delayed by the controller latency.
        rows = np.searchsorted(data[:, 0], data[row, 0] + times[:-1] - p["command_delay"], side="right") - 1
        rows = np.clip(rows, 0, len(data) - 1)
        targets[:, w, arm] = data[rows][:, COLUMNS["cmd"]]
        targets[:, w, fingers[0]] = np.clip(data[rows, COLUMNS["grip_cmd"]], 0.0, 1.0) * GRIPPER_TRAVEL
    state_0.joint_q.assign(q0.reshape(-1).astype(np.float32))
    state_0.joint_qd.assign(qd0.reshape(-1))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    # The command schedule, step counter, and predicted trajectory live on the device, so one
    # simulation step is a fixed launch sequence that is captured once as a CUDA graph.
    schedule = wp.array(targets.reshape(steps, -1), dtype=wp.float32, device=model.device)
    history = wp.zeros((steps + 1, worlds * coords), dtype=wp.float32, device=model.device)
    counter = wp.zeros(1, dtype=wp.int32, device=model.device)
    wp.copy(history[0], state_0.joint_q)

    def step():
        wp.launch(_load_command, dim=worlds * coords, inputs=[schedule, counter], outputs=[control.joint_target_q])
        solver.step(state_0, state_1, control, None, SIM_DT)
        state_0.assign(state_1)
        wp.launch(_record_step, dim=worlds * coords, inputs=[state_0.joint_q, counter], outputs=[history])
        # Advance the step counter in its own launch: incrementing it inside _record_step races with the
        # threads of that launch that have not read it yet.
        wp.launch(_advance_counter, dim=1, inputs=[counter])

    if model.device.is_cuda:
        with wp.ScopedCapture() as capture:
            step()
        wp.copy(history[0], state_0.joint_q)
        for _ in range(steps):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(steps):
            step()
    return times, history.numpy().reshape(steps + 1, worlds, coords)[:, :, arm]


@wp.kernel
def _load_command(schedule: wp.array2d[wp.float32], counter: wp.array[wp.int32], target: wp.array[wp.float32]):
    i = wp.tid()
    target[i] = schedule[counter[0], i]


@wp.kernel
def _record_step(q: wp.array[wp.float32], counter: wp.array[wp.int32], history: wp.array2d[wp.float32]):
    i = wp.tid()
    history[counter[0] + 1, i] = q[i]


@wp.kernel
def _advance_counter(counter: wp.array[wp.int32]):
    counter[0] = counter[0] + 1


def window_errors(segments: list[tuple[np.ndarray, float]], times: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Joint-angle error [rad] of each window, RMS over time and the six arm joints."""
    measured = np.stack(
        [
            np.stack([np.interp(start + times, data[:, 0], data[:, 1 + j]) for j in range(6)], axis=-1)
            for data, start in segments
        ],
        axis=1,
    )
    return np.sqrt(np.mean((q - measured) ** 2, axis=(0, 2)))


def evaluate(logs: dict[str, np.ndarray], params: dict | None = None) -> dict[str, float]:
    """Mean window RMSE [rad] per log, all windows simulated in one batch."""
    segments, owners = [], []
    for name, data in logs.items():
        for start in windows(data):
            segments.append((data, start))
            owners.append(name)
    model = build_model(len(segments), params)
    times, q = rollout(model, make_solver(model), segments, params)
    errors = window_errors(segments, times, q)
    owners = np.asarray(owners)
    return {name: float(np.mean(errors[owners == name])) for name in logs}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        paths = args.log or sorted(str(p) for p in (HERE / "logs").glob("*.csv"))[:1]
        self.logs = {Path(p).stem: load_log(p) for p in paths}
        self.data = next(iter(self.logs.values()))
        self.segments = [(self.data, start) for start in windows(self.data)]
        self.model = build_model(len(self.segments))
        self.solver = make_solver(self.model)
        self.frame_dt = HORIZON
        self.sim_dt = SIM_DT
        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)

    def evaluate(self) -> dict[str, float]:
        return evaluate(self.logs)

    def step(self):
        _, q = rollout(self.model, self.solver, self.segments)
        arm, _, coords = _coordinates(self.model)
        joint_q = self.model.joint_q.numpy().reshape(len(self.segments), coords)
        joint_q[:, arm] = q[-1]
        self.state_0.joint_q.assign(joint_q.reshape(-1))
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
        parser.add_argument("--log", action="append", help="Arm log CSV (repeatable)")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    print(example.evaluate())
