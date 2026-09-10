# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Track Kimodo G1 CSV motions with bounded joint torques in SolverMuJoCo.

Run ``uv run --extra wbc -m newton.examples robot_g1_wbc`` for standing balance.
Add ``--motion walk.csv --controller mpc`` for a native Kimodo G1 reference
(30 Hz, xyz + wxyz + 29 joint angles). The CPU sampling MPC is slower than
real time at its default budget. The faster QP handles standing gestures but
does not reliably track dynamic locomotion. Neither controller guarantees
feasibility or balance for an arbitrary kinematic reference.
"""

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.utils
from newton.examples.robot.wbc_controller import MotionReference, WholeBodyQP
from newton.examples.robot.wbc_mpc import WholeBodyMPC


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if not np.isfinite(args.slowdown) or args.slowdown <= 0:
            raise ValueError("slowdown must be finite and positive")
        self.fps = 50
        self.frame_dt = 1 / self.fps
        self.sim_time = 0.0
        self.sim_dt = 0.002
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        asset = newton.utils.download_asset("unitree_g1")
        builder.add_mjcf(
            str(asset / "mjcf/g1_29dof_rev_1_0.xml"), collapse_fixed_joints=True, enable_self_collisions=False
        )
        builder.joint_target_ke[:] = [0.0] * builder.joint_dof_count
        builder.joint_target_kd[:] = [0.0] * builder.joint_dof_count
        builder.joint_target_mode[:] = [0] * builder.joint_dof_count
        # Reflected actuator inertia from the public BeyondMimic G1 configuration.
        leg = [0.010177520, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450]
        arm = [0.003609725] * 5 + [0.00425] * 2
        armature = np.array(leg * 2 + [0.010177520, 0.007219450, 0.007219450] + arm * 2)
        builder.joint_armature[6:] = armature.tolist()
        self.kp = armature * (2 * np.pi * 10) ** 2
        self.kd = 4 * armature * (2 * np.pi * 10)
        builder.add_ground_plane()
        self.model = builder.finalize(device="cpu")
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            use_mujoco_contacts=True,
            integrator="implicitfast",
            cone="elliptic",
            iterations=50,
        )
        self.mj = self.solver.mj_model
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.points = []
        for side in ("left", "right"):
            body = next(i for i in range(self.mj.nbody) if self.mj.body(i).name.endswith(f"{side}_ankle_roll_link"))
            for x in (-0.05, 0.12):
                for y in (-0.025, 0.025):
                    self.points.append((body, np.array([x, y, -0.035])))
        self.limits = self.mj.jnt_actfrcrange[1:, 1].copy()
        self.wbc = WholeBodyQP(self.mj, self.points, self.limits)
        if args.motion:
            qpos = np.loadtxt(args.motion, delimiter=",")
        else:
            q = self.solver.mj_data.qpos.copy()
            q[[7, 13]] = -0.2
            q[[10, 16]] = 0.4
            q[[11, 17]] = -0.2
            qpos = np.tile(q, (301, 1))
        self.motion = MotionReference(self.mj, qpos, fps=args.motion_fps / args.slowdown)
        qpos = self.motion.qpos
        d = mujoco.MjData(self.mj)
        d.qpos[:] = qpos[0]
        mujoco.mj_forward(self.mj, d)
        points, _ = self.wbc.kinematics(d)
        self.floor_shift = 0.002 - points[:, 2].min()
        qpos[:, 2] += self.floor_shift
        self.mpc = None
        if args.controller == "mpc":
            self.mpc = WholeBodyMPC(
                self.mj,
                self.kp,
                self.kd,
                self.motion,
                samples=args.mpc_samples,
                prediction_dt=args.prediction_dt,
                rounds=args.mpc_rounds,
                seed=args.seed,
            )
        q, v, _ = self.motion.sample(0)
        nq = q.copy()
        nq[3:7] = q[[4, 5, 6, 3]]
        mujoco.mj_forward(self.mj, d)
        rot = d.xmat[1].reshape(3, 3)
        omega = rot @ v[3:6]
        nv = v.copy()
        nv[:3] += np.cross(omega, rot @ self.mj.body_ipos[1])
        nv[3:6] = omega
        self.state_0.joint_q.assign(nq.astype(np.float32))
        self.state_0.joint_qd.assign(nv.astype(np.float32))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.q, self.v = q, v
        self.rows, self.poses = [], []
        self.saturated_steps = 0
        self.physics_steps = 0
        self.contact_rows = []
        self.nonfoot_bodies = set()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.5, -3.0, 1.5), pitch=-10.0, yaw=130.0)

    def step(self):
        for _ in range(self.args.control_rate // self.fps):
            target = self.motion.sample(self.sim_time)
            start = time.perf_counter()
            if self.args.controller == "qp":
                tau = self.wbc.solve(self.q, self.v, target)
            elif self.args.controller == "mpc":
                offset = self.mpc.solve(self.q, self.v, self.sim_time)
                tau = self.kp * offset
            else:
                tau = np.zeros_like(self.limits)
            controller_ms = (time.perf_counter() - start) * 1000
            command_q = self.q[7:].copy()
            command_v = self.v[6:].copy()
            command_a = self.wbc.last_solution[6 : self.mj.nv].copy()
            peak_torque_ratio = 0.0
            for substep in range(round(1 / self.args.control_rate / self.sim_dt)):
                if self.args.controller in ("pd", "mpc"):
                    qtarget, vtarget, _ = self.motion.sample(self.sim_time + substep * self.sim_dt)
                    applied = tau + self.kp * (qtarget[7:] - self.q[7:]) + self.kd * (vtarget[6:] - self.v[6:])
                else:
                    elapsed = substep * self.sim_dt
                    qtarget = command_q + elapsed * command_v + 0.5 * elapsed**2 * command_a
                    vtarget = command_v + elapsed * command_a
                    applied = tau + self.kp * (qtarget - self.q[7:]) + self.kd * (vtarget - self.v[6:])
                applied = np.clip(applied, -self.limits, self.limits)
                peak_torque_ratio = max(peak_torque_ratio, float(np.max(np.abs(applied) / self.limits)))
                self.saturated_steps += int(np.any(np.abs(applied) >= self.limits * 0.999))
                self.physics_steps += 1
                self.control.joint_f.assign(np.r_[np.zeros(6), applied].astype(np.float32))
                self.state_0.clear_forces()
                self.viewer.apply_forces(self.state_0)
                previous_time = self.solver.mj_data.time
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                if self.solver.mj_data.time < previous_time + self.sim_dt * 0.9:
                    raise RuntimeError("MuJoCo reset an unstable simulation")
                self.state_0, self.state_1 = self.state_1, self.state_0
                self.q = self.solver.mj_data.qpos.copy()
                self.v = self.solver.mj_data.qvel.copy()
                if not np.isfinite(self.q).all() or not np.isfinite(self.v).all():
                    raise RuntimeError("Non-finite simulation state")
                ground_force, nonfoot_force = 0.0, 0.0
                for index, contact in enumerate(self.solver.mj_data.contact):
                    bodies = self.mj.geom_bodyid[[contact.geom1, contact.geom2]]
                    if 0 not in bodies:
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.mj, self.solver.mj_data, index, force)
                    ground_force += max(0.0, force[0])
                    body = self.mj.body(int(max(bodies))).name
                    if not body.endswith(("left_ankle_roll_link", "right_ankle_roll_link")):
                        nonfoot_force += max(0.0, force[0])
                        if force[0] > 1.0:
                            self.nonfoot_bodies.add(body)
                self.contact_rows.append([self.sim_time + (substep + 1) * self.sim_dt, ground_force, nonfoot_force])
            self.q = self.solver.mj_data.qpos.copy()
            self.v = self.solver.mj_data.qvel.copy()
            self.sim_time += 1 / self.args.control_rate
            error = np.zeros(self.mj.nv)
            current_target = self.motion.sample(self.sim_time)[0]
            mujoco.mj_differentiatePos(self.mj, error, 1, self.q, current_target)
            self.rows.append(
                [
                    self.sim_time,
                    self.q[2],
                    np.linalg.norm(error[:3]),
                    np.linalg.norm(error[3:6]),
                    np.sqrt(np.mean(error[6:] ** 2)),
                    controller_ms,
                    peak_torque_ratio,
                    self.wbc.residual if self.args.controller == "qp" else 0.0,
                ]
            )
            self.poses.append(self.q.copy())

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        assert np.isfinite(self.q).all(), "Non-finite simulation state"
        if not self.args.motion:
            assert self.q[2] > 0.5, "Standing robot fell"

    def save(self):
        if not self.args.output:
            return
        path = Path(self.args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = np.array(self.rows)
        timing = rows[min(10, len(rows) - 1) :, 5]
        final_up = float(self.solver.mj_data.xmat[1].reshape(3, 3)[2, 2])
        contacts = np.array(self.contact_rows)
        summary = {
            "motion": self.args.motion or "stand",
            "controller": self.args.controller,
            "duration": self.sim_time,
            "floor_shift": self.floor_shift,
            "min_height": float(rows[:, 1].min()),
            "root_rmse": float(np.sqrt(np.mean(rows[:, 2] ** 2))),
            "joint_rmse": float(np.sqrt(np.mean(rows[:, 4] ** 2))),
            "rotation_rmse": float(np.sqrt(np.mean(rows[:, 3] ** 2))),
            "controller_ms_median": float(np.median(timing)),
            "controller_ms_p95": float(np.percentile(timing, 95)),
            "deadline_miss_fraction": float(np.mean(timing > 1000 / self.args.control_rate)),
            "saturation_fraction": self.saturated_steps / self.physics_steps,
            "max_torque_ratio": float(rows[:, 6].max()),
            "qp_failures": self.wbc.failures,
            "mpc_failures": self.mpc.failures if self.mpc else 0,
            "final_height": float(self.q[2]),
            "final_up": final_up,
            "recovered": bool(self.q[2] > 0.55 and final_up > 0.7),
            "nonfoot_contact_seconds": float(np.sum(contacts[:, 2] > 1.0) * self.sim_dt),
            "nonfoot_impulse": float(np.sum(contacts[:, 2]) * self.sim_dt),
            "nonfoot_bodies": sorted(self.nonfoot_bodies),
            "config": vars(self.args),
        }
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(
            path.with_suffix(".npz"), rows=rows, qpos=self.poses, reference=self.motion.qpos, contacts=contacts
        )
        print(json.dumps(summary))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--motion", type=str, default=None, help="Kimodo G1 MuJoCo qpos CSV")
        parser.add_argument("--motion-fps", type=float, default=30.0)
        parser.add_argument("--slowdown", type=float, default=1.0)
        parser.add_argument("--controller", choices=("qp", "pd", "mpc"), default="qp")
        parser.add_argument("--mpc-samples", type=int, default=128)
        parser.add_argument("--mpc-rounds", type=int, default=2)
        parser.add_argument("--prediction-dt", type=float, default=0.01)
        parser.add_argument("--control-rate", type=int, choices=(50, 100), default=100)
        parser.add_argument("--seed", type=int, default=123)
        parser.add_argument("--output", type=str, default=None, help="Output stem for measured JSON and NPZ")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.save()
