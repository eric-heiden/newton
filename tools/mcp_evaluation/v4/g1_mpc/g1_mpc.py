# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 humanoid tracking a reference motion (starter for a controller).

The robot is the floating-base Unitree G1 with 29 actuated joints, standing on
the floor of its MJCF and simulated by SolverMuJoCo with 2 ms steps. A
controller runs at 100 Hz: every 10 ms it reads the robot's state and returns
one :class:`Command` for the next control period. Each joint actuator then
applies, at every 2 ms physics step,

    tau = clip(kp * (q - q_joint) + kd * (qd - qd_joint) + tau_ff, -limit, limit)

with the MJCF torque limit of its joint. The plant (``build_model``,
``make_solver``, :class:`Robot`) is fixed; the controller is the part to develop.

Reference motions are MuJoCo-qpos CSV clips at 30 fps: root position [m], root
quaternion (w, x, y, z), and the 29 joint angles [rad] in MJCF order. The robot
starts at rest at the clip's first frame with its feet 2 mm above the floor.

The baseline :class:`Controller` servos every joint toward the reference angles.
It ignores balance and falls within about a second.

Run: ``python g1_mpc.py --motion walk.csv --viewer null`` (the whole clip; ``--num-frames N``
runs N control periods) prints the tracking report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples

SIM_DT = 0.002  # physics step [s]
CONTROL_DT = 0.01  # control period [s] (100 Hz)
SUBSTEPS = round(CONTROL_DT / SIM_DT)
JOINT_COUNT = 29

# Reflected actuator inertia per joint (BeyondMimic G1 configuration) [kg m^2].
LEG = [0.010177520, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450]
ARM = [0.003609725] * 5 + [0.00425] * 2
ARMATURE = np.array(LEG * 2 + [0.010177520, 0.007219450, 0.007219450] + ARM * 2)

# Accepted actuator gains: kp [N m/rad] and kd [N m s/rad] per joint.
KP_RANGE = (1.0, 10000.0)
KD_RANGE = (0.0, 500.0)

# The robot has fallen when the root drops below this height [m], its up axis tilts more than 45 degrees
# (cosine below 0.7), or anything but the feet touches the floor.
FALL_HEIGHT_M = 0.55
FALL_UP_COS = 0.7
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# Foot contact spheres (centres in the ankle roll link frame [m]) and their radius [m].
SOLE_POINTS = np.array([[-0.05, -0.025, -0.03], [-0.05, 0.025, -0.03], [0.12, -0.03, -0.03], [0.12, 0.03, -0.03]])
SOLE_RADIUS = 0.005
SOLE_CENTER = np.array([0.035, 0.0, -0.035])
LIFT_REFERENCE_M = 0.03  # lift recall: reference sole clearance above this [m] ...
LIFT_ROBOT_M = 0.02  # ... counts when the robot's sole clearance is above this [m]
JITTER_HZ = 6.0  # jitter: joint-error RMS above this frequency [Hz]


def asset_path() -> str:
    return str(newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml")


def build_model(joint_q: np.ndarray | None = None) -> newton.Model:
    """The G1 on the MJCF floor with torque-limited joint actuators, optionally posed at ``joint_q``."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(asset_path(), collapse_fixed_joints=True)
    # Robot self-collision is excluded; the feet and body still collide with the floor.
    robot_shapes = [i for i, body in enumerate(builder.shape_body) if body >= 0]
    for i, shape in enumerate(robot_shapes):
        for other in robot_shapes[i + 1 :]:
            builder.add_shape_collision_filter_pair(shape, other)
    builder.joint_armature[6:] = ARMATURE.tolist()
    # MuJoCo position servos whose gains and targets Robot.apply sets from each Command.
    builder.joint_target_ke[6:] = [100.0] * JOINT_COUNT
    builder.joint_target_kd[6:] = [2.0] * JOINT_COUNT
    builder.joint_target_mode[6:] = [int(newton.JointTargetMode.POSITION)] * JOINT_COUNT
    builder.custom_attributes["mujoco:ctrl_source"].values = [
        int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET)
    ] * JOINT_COUNT
    builder.custom_attributes["mujoco:actuator_has_forcerange"].values = [True] * JOINT_COUNT
    builder.custom_attributes["mujoco:actuator_forcelimited"].values = [1] * JOINT_COUNT
    builder.custom_attributes["mujoco:actuator_forcerange"].values = [
        (-limit, limit) for limit in builder.joint_effort_limit[6:]
    ]
    if joint_q is not None:
        builder.joint_q[:] = np.asarray(joint_q, dtype=float).tolist()
    return builder.finalize()


def make_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    return newton.solvers.SolverMuJoCo(
        model, njmax=192, nconmax=64, use_mujoco_contacts=True, integrator="implicitfast", iterations=50
    )


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def body_index(model: newton.Model, name: str) -> int:
    return [label.rsplit("/", 1)[-1] for label in model.body_label].index(name)


class MotionClip:
    """A reference clip, linearly interpolated in time (root quaternion by normalized lerp).

    With a model, the clip is shifted vertically so that the lowest foot contact point of its first frame
    is 2 mm above the floor (the robot starts there).
    """

    def __init__(self, path, model: newton.Model | None = None, fps: float = 30.0):
        self.qpos = np.loadtxt(path, delimiter=",", ndmin=2)
        if self.qpos.shape[1] != 7 + JOINT_COUNT or len(self.qpos) < 2:
            raise ValueError("Expected rows of 36 qpos values (root pos, root quat wxyz, 29 joints)")
        self.fps = fps
        self.duration = (len(self.qpos) - 1) / fps
        self.z_offset = 0.0
        if model is not None:
            self.z_offset = floor_offset(model, self.sample(0.0))

    def sample(self, t: float) -> np.ndarray:
        """Reference joint_q in Newton layout (root pos, root quat xyzw, joints) at time ``t`` [s]."""
        x = np.clip(t, 0.0, self.duration) * self.fps
        i = min(int(x), len(self.qpos) - 2)
        a = x - i
        q = (1 - a) * self.qpos[i] + a * self.qpos[i + 1]
        q[3:7] /= np.linalg.norm(q[3:7])
        q[2] += self.z_offset
        return np.concatenate([q[:3], quat_wxyz_to_xyzw(q[3:7]), q[7:]])


def floor_offset(model: newton.Model, joint_q: np.ndarray) -> float:
    """Vertical shift [m] that puts the lowest foot contact point of ``joint_q`` 2 mm above the floor."""
    state = model.state()
    newton.eval_fk(model, wp.array(joint_q, dtype=float, device=model.device), model.joint_qd, state)
    body_q = state.body_q.numpy().astype(np.float64)
    lowest = np.inf
    for name in FEET:
        pose = body_q[body_index(model, name)]
        points = pose[:3] + SOLE_POINTS @ _rotation(pose[3:7]).T
        lowest = min(lowest, float(points[:, 2].min()) - SOLE_RADIUS)
    return float(0.002 - lowest)


@dataclass
class Command:
    """Actuator command for one control period: per-joint arrays of 29 values in MJCF joint order (scalars
    broadcast). Each actuator applies ``clip(kp (q - q_joint) + kd (qd - qd_joint) + tau, -limit, limit)``
    at every physics step of the period."""

    q: np.ndarray
    """Position targets [rad]."""
    kp: np.ndarray | float
    """Stiffness [N m/rad], within KP_RANGE."""
    kd: np.ndarray | float
    """Damping [N m s/rad], within KD_RANGE."""
    qd: np.ndarray | float = 0.0
    """Velocity targets [rad/s]."""
    tau: np.ndarray | float = 0.0
    """Feedforward torques [N m]."""


def command_arrays(command) -> dict[str, np.ndarray]:
    """The command's fields as float64 arrays of 29 values; raises ValueError for invalid commands."""
    out = {}
    for name in ("q", "kp", "kd", "qd", "tau"):
        value = getattr(command, name, 0.0)
        value = np.broadcast_to(np.asarray(0.0 if value is None else value, dtype=np.float64), (JOINT_COUNT,))
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Command.{name} is not finite")
        out[name] = value.copy()
    for name, (low, high) in (("kp", KP_RANGE), ("kd", KD_RANGE)):
        if np.any(out[name] < low) or np.any(out[name] > high):
            raise ValueError(f"Command.{name} must be within [{low:g}, {high:g}]")
    return out


class Robot:
    """The simulated G1 (the plant): model, solver, state, and the actuator command interface."""

    def __init__(self, start_q: np.ndarray):
        self.model = build_model(start_q)
        self.solver = make_solver(self.model)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.effort_limit = self.model.joint_effort_limit.numpy()[6:].copy()
        # MuJoCo actuator -> actuated joint (0..28), for writing servo gains without a full model update.
        self._actuator_joint = self.solver.mjc_actuator_to_newton_idx.numpy() - 6
        self._gains = None
        self._target = self.model.joint_q.numpy().copy()
        self.graph = None
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """One control period of physics steps."""
        for i in range(SUBSTEPS):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, SIM_DT)
            if SUBSTEPS % 2 == 1 and i == SUBSTEPS - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def apply(self, command) -> None:
        """Set the actuators from a :class:`Command` (the servo target absorbs qd and tau, which is exact)."""
        c = command_arrays(command)
        gains = np.concatenate([c["kp"], c["kd"]])
        if self._gains is None or not np.array_equal(gains, self._gains):
            ke, kd = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
            ke[6:], kd[6:] = c["kp"], c["kd"]
            self.model.joint_target_ke.assign(ke)
            self.model.joint_target_kd.assign(kd)
            # Same effect as solver.notify_model_changed(JOINT_DOF_PROPERTIES) on the servos, which is much slower.
            gain, bias = self.solver.mjw_model.actuator_gainprm, self.solver.mjw_model.actuator_biasprm
            gain_np, bias_np = gain.numpy(), bias.numpy()
            gain_np[:, :, 0] = c["kp"][self._actuator_joint]
            bias_np[:, :, 1] = -c["kp"][self._actuator_joint]
            bias_np[:, :, 2] = -c["kd"][self._actuator_joint]
            gain.assign(gain_np)
            bias.assign(bias_np)
            self._gains = gains
        self._target[7:] = c["q"] + (c["kd"] * c["qd"] + c["tau"]) / c["kp"]
        self.control.joint_target_q.assign(self._target)

    def advance(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle [rad] between two unit quaternions (xyzw)."""
    return float(2.0 * np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def _rotation(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw / np.linalg.norm(quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def foot_points(body_q: np.ndarray, foot: int) -> tuple[np.ndarray, float]:
    """Sole centre [m] and sole clearance above the floor [m] of a foot body."""
    rotation = _rotation(body_q[foot, 3:7])
    clearance = float(np.min((body_q[foot, :3] + SOLE_POINTS @ rotation.T)[:, 2]) - SOLE_RADIUS)
    return body_q[foot, :3] + rotation @ SOLE_CENTER, clearance


def jitter(errors: np.ndarray, dt: float, cutoff: float = JITTER_HZ, trim: float = 0.1) -> float:
    """RMS [rad] of the joint errors' content above ``cutoff`` [Hz] (zero-phase, ``trim`` [s] cut at both ends)."""
    k = round(trim / dt)
    errors = errors[k : len(errors) - k]
    if len(errors) < 16:
        return float("nan")
    errors = errors - errors.mean(axis=0)
    spectrum = np.fft.rfft(errors, axis=0)
    spectrum[np.fft.rfftfreq(len(errors), dt) <= cutoff] = 0.0
    return float(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(errors), axis=0) ** 2)))


class TrackingReport:
    """Tracking errors against a reference clip, sampled after every control period."""

    def __init__(self, robot: Robot, motion: MotionClip):
        self.robot, self.motion = robot, motion
        model = robot.model
        self.scratch = model.state()
        self.joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=model.device)
        self.feet = [body_index(model, name) for name in FEET]
        self.wrists = [body_index(model, name) for name in WRISTS]
        mj = robot.solver.mj_model
        foot_ids = {i for i in range(mj.nbody) if mj.body(i).name.endswith(FEET)}
        # MuJoCo geoms that may touch the floor: those of the feet.
        self.foot_geom = np.array([int(mj.geom_bodyid[g]) in foot_ids for g in range(mj.ngeom)])
        self.robot_geom = np.array([int(mj.geom_bodyid[g]) != 0 for g in range(mj.ngeom)])
        self.rows = []
        self.fall_time = None
        self.fall_reason = None

    def non_foot_contact(self) -> bool:
        """Whether a robot geom other than the feet touches the floor."""
        data = self.robot.solver.mjw_data
        count = min(int(data.nacon.numpy()[0]), data.contact.geom.shape[0])
        if count == 0:
            return False
        geoms = data.contact.geom.numpy()[:count]
        touching = data.contact.dist.numpy()[:count] <= 0.0
        robot = self.robot_geom[geoms]
        other = robot & ~self.foot_geom[geoms]
        return bool(np.any(touching & other.any(axis=1)))

    def update(self, t: float, state: newton.State) -> bool:
        """Record the errors of ``state`` at time ``t`` [s]; returns whether the robot is still on its feet.

        Recording stops at a fall. Samples at or after ``t`` are dropped first (the clock was rewound)."""
        while self.rows and self.rows[-1]["t"] >= t - 1e-9:
            self.rows.pop()
        if self.fall_time is not None and self.fall_time >= t - 1e-9:
            self.fall_time = self.fall_reason = None
        if self.fall_time is not None:
            return False
        q = state.joint_q.numpy().astype(np.float64)
        body_q = state.body_q.numpy().astype(np.float64)
        ref = self.motion.sample(t)
        self.joint_q.assign(ref)
        newton.eval_fk(self.robot.model, self.joint_q, self.robot.model.joint_qd, self.scratch)
        ref_body_q = self.scratch.body_q.numpy().astype(np.float64)
        finite = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(body_q)))
        quat = q[3:7] / np.linalg.norm(q[3:7]) if finite else ref[3:7]
        soles = [foot_points(body_q, f) for f in self.feet]
        ref_soles = [foot_points(ref_body_q, f) for f in self.feet]
        row = {
            "t": t,
            "finite": finite,
            "root_err": float(np.linalg.norm(q[:3] - ref[:3])),
            "root_rot_err": _quat_angle(quat, ref[3:7]),
            "joint_err": q[7:] - ref[7:],
            "wrist_err2": float(np.mean(np.sum((body_q[self.wrists, :3] - ref_body_q[self.wrists, :3]) ** 2, axis=1))),
            "sole_err2": float(np.mean([np.sum((a[0] - b[0]) ** 2) for a, b in zip(soles, ref_soles, strict=True)])),
            "lift": [(b[1] > LIFT_REFERENCE_M, a[1] > LIFT_ROBOT_M) for a, b in zip(soles, ref_soles, strict=True)],
        }
        self.rows.append(row)
        reason = None
        if not finite:
            reason = "state not finite"
        elif q[2] < FALL_HEIGHT_M:
            reason = f"root height {q[2]:.2f} m"
        elif _rotation(quat)[2, 2] < FALL_UP_COS:
            reason = "root tilted over"
        elif self.non_foot_contact():
            reason = "non-foot floor contact"
        if reason is not None:
            self.fall_time, self.fall_reason = t, reason
        return reason is None

    def summary(self) -> dict:
        """Metrics over the recorded samples, up to a fall (errors are infinite if the state stopped being finite)."""
        rows = self.rows
        finite = bool(rows) and all(r["finite"] for r in rows)
        inf = float("inf")

        def rms(values):
            return float(np.sqrt(np.mean(np.square(values)))) if finite else inf

        lifts = [robot for r in rows for ref, robot in r["lift"] if ref]
        errors = np.array([r["joint_err"] for r in rows]) if finite else None
        return {
            "upright": self.fall_time is None and finite,
            "fall_time_s": self.fall_time,
            "fall_reason": self.fall_reason,
            "survival": 1.0 if self.fall_time is None else self.fall_time / self.motion.duration,
            "root_rmse_m": rms([r["root_err"] for r in rows]),
            "root_rot_rmse_deg": float(np.degrees(rms([r["root_rot_err"] for r in rows]))),
            "joint_rmse_rad": rms(errors) if finite else inf,
            "wrist_rmse_m": float(np.sqrt(np.mean([r["wrist_err2"] for r in rows]))) if finite else inf,
            "sole_rmse_m": float(np.sqrt(np.mean([r["sole_err2"] for r in rows]))) if finite else inf,
            "lift_recall": float(np.mean(lifts)) if lifts else None,
            "jitter_mrad": 1000.0 * jitter(errors, CONTROL_DT) if finite else inf,
            "root_err_max_m": max(r["root_err"] for r in rows) if finite else inf,
            "samples": len(rows),
        }


class Controller:
    """Baseline: servo every joint toward the reference angle one control period ahead (no balance).

    ``model`` is a copy of the plant's model posed at the start (``build_model(motion.sample(0.0))``), the
    controller's to use for planning; the simulated robot is a separate model the controller never sees.
    ``motion`` is the reference clip (a :class:`MotionClip` with the floor shift applied).
    """

    def __init__(self, model: newton.Model, motion: MotionClip):
        self.motion = motion
        natural_frequency = 2.0 * np.pi * 10.0
        self.kp = ARMATURE * natural_frequency**2
        self.kd = 4.0 * ARMATURE * natural_frequency

    def compute(self, t: float, joint_q: np.ndarray, joint_qd: np.ndarray) -> Command:
        """Command for the control period starting at ``t`` [s], given the state then: joint_q (root position [m],
        root quaternion xyzw, 29 joint angles [rad]) and joint_qd (root COM linear [m/s] and angular [rad/s]
        velocity in the world frame, 29 joint velocities [rad/s])."""
        reference = self.motion.sample(t + CONTROL_DT)
        return Command(q=reference[7:], kp=self.kp, kd=self.kd)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = CONTROL_DT
        self.sim_dt = SIM_DT
        self.sim_time = 0.0
        # The robot starts at rest at the clip's first frame; the controller plans on its own copy of the model.
        self.motion = MotionClip(args.motion, build_model())
        start = self.motion.sample(0.0)
        self.robot = Robot(start)
        planning_model = build_model(start)
        self.controller = Controller(planning_model, MotionClip(args.motion, planning_model))
        self.report = TrackingReport(self.robot, self.motion)
        self.compute_seconds = 0.0
        self.viewer.set_model(self.model)

    # The live host and viewers look for these on the example.
    @property
    def model(self):
        return self.robot.model

    @property
    def solver(self):
        return self.robot.solver

    @property
    def state_0(self):
        return self.robot.state_0

    @property
    def state_1(self):
        return self.robot.state_1

    @property
    def control(self):
        return self.robot.control

    def capture(self):
        self.robot.capture()

    def step(self):
        # Read the state first: the copy waits for the last physics step, which is not the controller's time.
        joint_q = self.state_0.joint_q.numpy().astype(np.float64)
        joint_qd = self.state_0.joint_qd.numpy().astype(np.float64)
        started = time.perf_counter()
        command = self.controller.compute(self.sim_time, joint_q, joint_qd)
        self.compute_seconds += time.perf_counter() - started
        self.robot.apply(command)
        self.robot.advance()
        self.sim_time += self.frame_dt
        self.report.update(self.sim_time, self.state_0)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def print_report(self):
        summary = self.report.summary()
        print(f"tracked {self.sim_time:.2f} of {self.motion.duration:.2f} s")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print(f"  controller_seconds: {self.compute_seconds:.2f}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--motion", type=str, default="walk.csv", help="Reference clip (MuJoCo qpos CSV, 30 fps)")
        parser.set_defaults(num_frames=None)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    known, _ = parser.parse_known_args()
    if known.num_frames is None:
        parser.set_defaults(num_frames=round(MotionClip(known.motion).duration / CONTROL_DT))
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    started = time.perf_counter()
    newton.examples.run(example, args)
    example.print_report()
    print(f"  wall_seconds: {time.perf_counter() - started:.2f}")
