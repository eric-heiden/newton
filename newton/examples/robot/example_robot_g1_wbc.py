# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Track Kimodo G1 CSV motions with bounded joint actuation in SolverMuJoCo.

Run ``uv run --extra wbc -m newton.examples robot_g1_wbc`` for standing balance.
Add ``--motion walk.csv`` for a native Kimodo G1 reference
(30 Hz, xyz + wxyz + 29 joint angles). Gauss-Newton shooting is the default;
``--controller mpc`` selects predictive sampling. Both complete optimizers,
including their MuJoCo Warp rollouts, execute as CUDA graphs. The CPU QP handles
standing gestures but does not reliably track dynamic locomotion. No mode
guarantees feasibility or balance for an arbitrary kinematic reference.
See ``g1_wbc.md`` for parameters and measured limitations.
"""

import argparse
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
from newton.examples.robot.wbc_controller import (
    MotionReference,
    WholeBodyQP,
    measure_foot_tracking,
    measure_head_tracking,
    measure_motion_tracking,
)
from newton.examples.robot.wbc_mpc import WholeBodyMPC, audit_step
from newton.examples.robot.wbc_mpc_adjoint import WholeBodyAdjoint
from newton.examples.robot.wbc_mpc_dial import WholeBodyDial
from newton.examples.robot.wbc_mpc_gn import WholeBodyGaussNewton
from newton.examples.robot.wbc_rollouts import G1_TRACE_BODIES, RolloutTraces, draw_rollouts


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if args.show_rollouts and args.controller not in ("mpc", "mpc-gn", "mpc-dial", "mpc-adjoint", "mpc-hybrid"):
            raise ValueError("Rollout visualization requires an MPC controller")
        if args.rollout_count < 1 or args.rollout_count > 16:
            raise ValueError("Display between 1 and 16 rollout candidates")
        self.trace_frames, self.trace_frame = [], None
        # Gauss-Newton uses finer prediction and explicit limb tracking.
        # Sampling retains its broader, lower-gain starting configuration.
        defaults = {
            "gain_scale": (1.0, 4.0),
            "joint_scale": (0.3, 0.15),
            "root_scale": (0.08, 0.04),
            "rotation_scale": (0.15, 0.1),
            "foot_weight": (0.0, 300.0),
            "foot_vertical": (1.0, 6.0),
            "foot_rotation": (0.0, 10.0),
            "angular_weight": (0.0, 0.2),
            "hand_position": (0.0, 300.0),
            "hand_rotation": (0.0, 3.0),
            "head_position": (0.0, 100.0),
            "head_rotation": (0.0, 300.0),
            "joint_velocity": (0.0, 0.02),
            "arm_velocity_scale": (1.0, 5.0),
            "prediction_dt": (0.01, 0.005),
            "mpc_rounds": (2, 1),
            "actuation": ("torque", "pd"),
            "gn_coordinate_search": (False, True),
        }
        if args.controller == "mpc-dial" and args.mpc_rounds is None:
            args.mpc_rounds = 2
        for name, values in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, values[int(args.controller in ("mpc-gn", "mpc-dial", "mpc-adjoint", "mpc-hybrid"))])
        if not np.isfinite(args.slowdown) or args.slowdown <= 0:
            raise ValueError("slowdown must be finite and positive")
        self.fps = 50
        self.step_rate = max(args.control_rate, self.fps)
        if args.control_rate < self.fps and args.controller not in (
            "mpc",
            "mpc-gn",
            "mpc-dial",
            "mpc-adjoint",
            "mpc-hybrid",
        ):
            raise ValueError("Replanning below 50 Hz is supported by the GPU MPC modes")
        self.frame_dt = 1 / self.fps
        self.sim_time = 0.0
        self.sim_dt = 0.002
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        asset = newton.utils.download_asset("unitree_g1")
        builder.add_mjcf(
            str(asset / "mjcf/g1_29dof_rev_1_0.xml"), collapse_fixed_joints=True, enable_self_collisions=True
        )
        # The importer hides colliders, including the floor; display that plane.
        for i, shape_type in enumerate(builder.shape_type):
            if shape_type == newton.GeoType.PLANE:
                builder.shape_flags[i] |= newton.ShapeFlags.VISIBLE
                builder.shape_color[i] = (0.125, 0.125, 0.15)
        # Exclude robot self-contact, but retain the floor supplied by the MJCF.
        robot_shapes = [
            i
            for i, body in enumerate(builder.shape_body)
            if body >= 0 and builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES
        ]
        for i, shape in enumerate(robot_shapes):
            for other in robot_shapes[i + 1 :]:
                builder.add_shape_collision_filter_pair(shape, other)
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
        if not np.isfinite(args.gain_scale) or args.gain_scale <= 0:
            raise ValueError("gain-scale must be finite and positive")
        self.kp *= args.gain_scale
        self.kd *= np.sqrt(args.gain_scale)
        self.native_pd = args.actuation == "pd" and args.controller in (
            "mpc",
            "mpc-gn",
            "mpc-dial",
            "mpc-adjoint",
            "mpc-hybrid",
        )
        if self.native_pd:
            builder.joint_target_ke[6:] = self.kp.tolist()
            builder.joint_target_kd[6:] = self.kd.tolist()
            builder.joint_target_mode[6:] = [int(newton.JointTargetMode.POSITION)] * len(self.kp)
            # Route the imported motor definitions through Newton's PD drives.
            builder.custom_attributes["mujoco:ctrl_source"].values = [
                int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET)
            ] * len(self.kp)
            builder.custom_attributes["mujoco:actuator_has_forcerange"].values = [True] * len(self.kp)
            builder.custom_attributes["mujoco:actuator_forcelimited"].values = [1] * len(self.kp)
            builder.custom_attributes["mujoco:actuator_forcerange"].values = [
                (-limit, limit) for limit in builder.joint_effort_limit[6:]
            ]
        # The MJCF already supplies the textured ground plane.
        self.gpu = args.controller in ("mpc", "mpc-gn", "mpc-dial", "mpc-adjoint", "mpc-hybrid")
        if self.gpu and not wp.get_device(args.device or "cuda:0").is_cuda:
            raise ValueError("MPC requires CUDA; use --controller qp for the CPU baseline")
        self.model = builder.finalize(device=(args.device or "cuda:0") if self.gpu else "cpu")
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=not self.gpu,
            update_data_interval=0 if self.gpu else 1,
            njmax=192,
            nconmax=64,
            jacobian="dense",
            use_mujoco_contacts=True,
            integrator="implicitfast",
            cone=args.cone,
            iterations=50,
            tolerance=1e-4 if self.gpu else 1e-8,
        )
        self.mj = self.solver.mj_model
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.applied_torque = self.solver.mjw_data.qfrc_actuator.flatten() if self.native_pd else self.control.joint_f
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
        if self.gpu:
            controller = {
                "mpc": WholeBodyMPC,
                "mpc-gn": WholeBodyGaussNewton,
                "mpc-dial": WholeBodyDial,
                "mpc-adjoint": WholeBodyAdjoint,
                "mpc-hybrid": WholeBodyAdjoint,
            }[args.controller]
            extra = (
                {
                    "epsilon": args.gn_epsilon,
                    "damping": args.gn_damping,
                    "trust": args.gn_trust,
                    "coordinate_search": args.gn_coordinate_search,
                }
                if args.controller == "mpc-gn"
                else {}
            )
            if args.controller == "mpc-dial":
                extra = {
                    "horizon_decay": args.dial_horizon_decay,
                    "round_decay": args.dial_round_decay,
                    "initial_rounds": args.dial_initial_rounds,
                    "dial_temperature": args.dial_temperature,
                }
            if args.controller in ("mpc-adjoint", "mpc-hybrid"):
                if args.adjoint_starts is None:
                    args.adjoint_starts = 4 if args.controller == "mpc-hybrid" else 1
                extra = {
                    "sketch": args.adjoint_sketch,
                    "starts": args.adjoint_starts,
                    "start_noise": args.adjoint_noise,
                    "damping": args.gn_damping,
                    "trust": args.gn_trust,
                }
            self.mpc = controller(
                self.mj,
                self.kp,
                self.kd,
                self.motion,
                samples=args.mpc_samples,
                horizon=args.horizon,
                knots=args.mpc_knots,
                temperature=args.temperature,
                nonfoot_weight=args.nonfoot_weight,
                noise=args.noise,
                joint_scale=args.joint_scale,
                root_scale=args.root_scale,
                rotation_scale=args.rotation_scale,
                foot_weight=args.foot_weight,
                foot_vertical=args.foot_vertical,
                foot_rotation=args.foot_rotation,
                angular_weight=args.angular_weight,
                sole_tracking=args.foot_task == "sole",
                hand_weight=args.hand_weight,
                hand_clearance=args.hand_clearance,
                hand_position=args.hand_position,
                hand_rotation=args.hand_rotation,
                joint_velocity=args.joint_velocity,
                arm_velocity_scale=args.arm_velocity_scale,
                head_position=args.head_position,
                head_rotation=args.head_rotation,
                iterations=args.prediction_iterations,
                device=self.model.device,
                prediction_dt=args.prediction_dt,
                rounds=args.mpc_rounds,
                seed=args.seed,
                **extra,
            )
            if args.show_rollouts:
                self.mpc.traces = RolloutTraces(
                    self.mpc,
                    G1_TRACE_BODIES if args.rollout_torso else G1_TRACE_BODIES[:-1],
                    horizon=args.rollout_horizon,
                    stride=args.rollout_stride,
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
        self.max_plant_constraints = 0
        self.max_plant_contacts = 0
        self.nonfoot_bodies = set()
        if self.gpu:
            self.model.joint_q.assign(self.state_0.joint_q)
            self.model.joint_qd.assign(self.state_0.joint_qd)
            self.solver.reset(self.state_0)
            self.mpc.capture(self.solver.mjw_data.qpos, self.solver.mjw_data.qvel, self.solver.mjw_data.time)
            with wp.ScopedDevice(self.model.device):
                self.audit = wp.zeros((round(1 / self.step_rate / self.sim_dt), 7))
                self.touched = wp.zeros(self.mj.nbody, dtype=int)
                self.geom_bodies = wp.array(self.mj.geom_bodyid, dtype=int)
                self.foot_bodies = wp.array(sorted({point[0] for point in self.points}), dtype=int)
                self.start_event, self.end_event = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                self.gpu_physics()
                self.solver.reset(self.state_0)
                self.solver.mjw_data.time.zero_()
                with wp.ScopedCapture() as capture:
                    self.gpu_physics()
                self.physics_graph = capture.graph
                self.solver.reset(self.state_0)
                self.solver.mjw_data.time.zero_()
                self.touched.zero_()
        self.viewer.set_model(self.model)
        self.camera_root = self.q[:3].copy()
        self.viewer.set_camera(wp.vec3(1.8, -1.8, 1.35), pitch=-12.0, yaw=135.0)

    def gpu_physics(self):
        data = self.solver.mjw_data
        for i in range(self.audit.shape[0]):
            self.mpc.apply(data, self.control, native_pd=self.native_pd)
            wp.copy(self.state_1.body_f, self.state_0.body_f)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.launch(
                audit_step,
                1,
                inputs=[
                    data.time,
                    data.contact.geom,
                    data.contact.efc_address,
                    data.nacon,
                    data.contact.dim,
                    int(self.mj.opt.cone),
                    data.efc.force,
                    self.geom_bodies,
                    self.foot_bodies,
                    self.applied_torque,
                    self.mpc.limits,
                    data.overflow,
                    data.nefc,
                    i,
                ],
                outputs=[self.audit, self.touched],
            )
            if i == self.audit.shape[0] - 1 and self.audit.shape[0] % 2:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def gpu_control(self):
        start = time.perf_counter()
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        replan = self.physics_steps % round(1 / self.args.control_rate / self.sim_dt) == 0
        wp.record_event(self.start_event)
        if replan:
            self.mpc.solve()
        wp.record_event(self.end_event)
        wp.capture_launch(self.physics_graph)
        data = self.solver.mjw_data
        self.q, self.v = data.qpos.numpy()[0], data.qvel.numpy()[0]
        audit = self.audit.numpy()
        if not np.isfinite(self.q).all() or not np.isfinite(self.v).all() or np.any(audit[:, 4]):
            raise RuntimeError("Non-finite or unconverged MuJoCo Warp plant step")
        self.max_plant_constraints = max(self.max_plant_constraints, int(audit[:, 5].max()))
        self.max_plant_contacts = max(self.max_plant_contacts, int(audit[:, 6].max()))
        self.saturated_steps += int(np.sum(audit[:, 3] >= 0.999))
        self.physics_steps += len(audit)
        self.contact_rows.extend(audit[:, :3].tolist())
        self.sim_time = float(audit[-1, 0])
        error = np.zeros(self.mj.nv)
        mujoco.mj_differentiatePos(self.mj, error, 1, self.q.astype(float), self.motion.sample(self.sim_time)[0])
        self.rows.append(
            [
                self.sim_time,
                self.q[2],
                np.linalg.norm(error[:3]),
                np.linalg.norm(error[3:6]),
                np.sqrt(np.mean(error[6:] ** 2)),
                wp.get_event_elapsed_time(self.start_event, self.end_event) if replan else 0.0,
                float(audit[:, 3].max()),
                0.0,
                (time.perf_counter() - start) * 1000,
            ]
        )
        self.poses.append(self.q.copy())

    def step(self):
        if self.gpu:
            for _ in range(self.step_rate // self.fps):
                self.gpu_control()
            if self.mpc.traces is not None:
                self.trace_frame = self.mpc.traces.snapshot(self.args.rollout_count)
                if self.args.output:
                    self.trace_frames.append(self.trace_frame)
            return
        for _ in range(self.args.control_rate // self.fps):
            target = self.motion.sample(self.sim_time)
            start = time.perf_counter()
            if self.args.controller == "qp":
                tau = self.wbc.solve(self.q, self.v, target)
            else:
                tau = np.zeros_like(self.limits)
            controller_ms = (time.perf_counter() - start) * 1000
            command_q = self.q[7:].copy()
            command_v = self.v[6:].copy()
            command_a = self.wbc.last_solution[6 : self.mj.nv].copy()
            peak_torque_ratio = 0.0
            for substep in range(round(1 / self.args.control_rate / self.sim_dt)):
                if self.args.controller == "pd":
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
        if hasattr(self.viewer, "camera") and not self.args.fixed_camera:
            delta = 0.2 * (self.q[:3] - self.camera_root)
            delta[2] = 0.0
            camera = self.viewer.camera
            self.viewer.set_camera(camera.pos + wp.vec3(*delta), camera.pitch, camera.yaw)
            self.camera_root += delta
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.trace_frame is not None:
            draw_rollouts(
                self.viewer,
                self.trace_frame["positions"],
                self.mpc.traces.offsets,
                elapsed=self.sim_time - self.trace_frame["time"],
            )
        self.viewer.end_frame()

    def test_final(self):
        assert np.count_nonzero(self.mj.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1, "Duplicate ground plane"
        robot_bodies = self.mj.nbody - 1
        assert self.mj.nexclude == robot_bodies * (robot_bodies - 1) // 2, "Robot self-contact was not excluded"
        assert np.isfinite(self.q).all(), "Non-finite simulation state"
        if self.native_pd:
            assert self.mj.nu == len(self.limits), "Duplicated PD actuators"
            np.testing.assert_allclose(self.mj.actuator_forcerange[:, 1], self.limits)
            assert max(row[6] for row in self.rows) <= 1.0001, "Actuator force limit exceeded"
            np.testing.assert_allclose(self.solver.mjw_data.qfrc_actuator.numpy()[0, :6], 0)
        if not self.args.motion:
            assert self.q[2] > 0.5, "Standing robot fell"

    def save(self):
        if not self.args.output:
            return
        path = Path(self.args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = np.array(self.rows)
        timing = rows[:, 5]
        if self.gpu:
            timing = timing[timing > 0]
        timing = timing[min(10, len(timing) - 1) :]
        # Include physics-only subframes when comparing with a replan deadline.
        period = self.step_rate // self.args.control_rate
        wall = np.add.reduceat(rows[:, 8], np.arange(0, len(rows), period)) if self.gpu else None
        if wall is not None:
            wall = wall[min(10, len(wall) - 1) :]
        d = mujoco.MjData(self.mj)
        d.qpos[:] = self.q
        mujoco.mj_kinematics(self.mj, d)
        final_up = float(d.xmat[1].reshape(3, 3)[2, 2])
        if self.gpu:
            self.nonfoot_bodies = {self.mj.body(i).name for i in np.flatnonzero(self.touched.numpy())}
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
            "mpc_failures": int(self.mpc.failure_count.numpy()[0]) if self.mpc else 0,
            "search_statistics": self.mpc.statistics.numpy().tolist() if self.gpu else None,
            "plant_peak_constraints": self.max_plant_constraints,
            "plant_peak_contacts": self.max_plant_contacts,
            "backend": "mujoco_warp_cuda_graph" if self.gpu else "mujoco_cpu",
            "control_wall_ms_median": float(np.median(wall)) if self.gpu else None,
            "control_wall_ms_p95": float(np.percentile(wall, 95)) if self.gpu else None,
            "control_period_deadline_miss_fraction": (
                float(np.mean(wall > 1000 / self.args.control_rate)) if self.gpu else None
            ),
            "final_height": float(self.q[2]),
            "final_up": final_up,
            "recovered": bool(self.q[2] > 0.55 and final_up > 0.7),
            "nonfoot_contact_seconds": float(np.sum(contacts[:, 2] > 1.0) * self.sim_dt),
            "nonfoot_impulse": float(np.sum(contacts[:, 2]) * self.sim_dt),
            "nonfoot_bodies": sorted(self.nonfoot_bodies),
            "config": vars(self.args),
        }
        summary.update(measure_foot_tracking(self.mj, self.motion, rows[:, 0], self.poses, self.points))
        summary.update(measure_motion_tracking(self.mj, self.motion, rows[:, 0], self.poses))
        summary.update(measure_head_tracking(self.mj, self.motion, rows[:, 0], self.poses))
        Path(f"{path}.json").write_text(json.dumps(summary, indent=2) + "\n")
        traces = {}
        if self.trace_frames:
            traces = {f"trace_{key}": np.array([f[key] for f in self.trace_frames]) for key in self.trace_frames[0]}
            traces.update(trace_offsets=self.mpc.traces.offsets, trace_bodies=self.mpc.traces.names)
        np.savez_compressed(
            f"{path}.npz", rows=rows, qpos=self.poses, reference=self.motion.qpos, contacts=contacts, **traces
        )
        print(json.dumps(summary))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fixed-camera", action="store_true", help="Disable horizontal camera following")
        parser.add_argument(
            "--show-rollouts", action="store_true", help="Draw and optionally record actual GPU futures"
        )
        parser.add_argument(
            "--rollout-count", type=int, default=4, help="Selected plan plus display alternatives (1-16)"
        )
        parser.add_argument(
            "--rollout-horizon", type=float, default=0.4, help="Displayed prediction duration in seconds"
        )
        parser.add_argument("--rollout-stride", type=int, default=4, help="Record every Nth prediction step")
        parser.add_argument("--rollout-torso", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--motion", type=str, default=None, help="Kimodo G1 MuJoCo qpos CSV")
        parser.add_argument("--motion-fps", type=float, default=30.0)
        parser.add_argument("--slowdown", type=float, default=1.0)
        parser.add_argument(
            "--controller",
            choices=("qp", "pd", "mpc", "mpc-gn", "mpc-dial", "mpc-adjoint", "mpc-hybrid"),
            default="mpc-gn",
        )
        parser.add_argument("--actuation", choices=("torque", "pd"))
        parser.add_argument(
            "--gn-coordinate-search",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Reuse improving finite-difference trials as coordinate-search candidates",
        )
        parser.add_argument("--gn-epsilon", type=float, default=0.03)
        parser.add_argument("--gn-damping", type=float, default=0.1)
        parser.add_argument("--gn-trust", type=float, default=0.2)
        parser.add_argument("--adjoint-sketch", type=int, default=16)
        parser.add_argument("--adjoint-starts", type=int)
        parser.add_argument("--adjoint-noise", type=float, default=0.05)
        parser.add_argument("--dial-temperature", type=float, default=0.06)
        parser.add_argument("--dial-horizon-decay", type=float, default=0.9)
        parser.add_argument("--dial-round-decay", type=float, default=0.5)
        parser.add_argument("--dial-initial-rounds", type=int, default=10)
        parser.add_argument("--mpc-samples", type=int, default=1024)
        parser.add_argument("--mpc-rounds", type=int, help="Search iterations; compute cost scales with this count")
        parser.add_argument("--prediction-dt", type=float)
        parser.add_argument("--control-rate", type=int, choices=(10, 25, 50, 100), default=100)
        parser.add_argument("--nonfoot-weight", type=float, default=1000.0)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--cone", choices=("elliptic", "pyramidal"), default="pyramidal")
        parser.add_argument("--prediction-iterations", type=int, default=20)
        parser.add_argument("--mpc-knots", type=int, default=4)
        parser.add_argument("--horizon", type=float, default=0.5)
        parser.add_argument("--noise", type=float, default=0.12)
        parser.add_argument("--joint-scale", type=float)
        parser.add_argument("--gain-scale", type=float, help="PD stiffness scale; damping scales by its square root")
        parser.add_argument("--foot-weight", type=float)
        parser.add_argument("--foot-task", choices=("ankle", "sole"), default="sole")
        parser.add_argument("--root-scale", type=float)
        parser.add_argument("--rotation-scale", type=float)
        parser.add_argument("--foot-vertical", type=float, help="Foot vertical position weight multiplier")
        parser.add_argument("--foot-rotation", type=float)
        parser.add_argument("--angular-weight", type=float, help="World-frame root angular velocity weight")
        parser.add_argument(
            "--hand-position", type=float, help="Wrist position tracking weight in inverse metres squared"
        )
        parser.add_argument("--hand-rotation", type=float, help="Wrist half-angle rotation tracking weight")
        parser.add_argument("--head-position", type=float, help="Head-center position tracking weight")
        parser.add_argument("--head-rotation", type=float, help="Head half-angle rotation tracking weight")
        parser.add_argument("--joint-velocity", type=float, help="Per-joint velocity tracking weight")
        parser.add_argument("--arm-velocity-scale", type=float, help="G1 arm multiplier for joint-velocity tracking")
        parser.add_argument("--hand-clearance", type=float, default=0.2)
        parser.add_argument("--hand-weight", type=float, default=10000.0)
        parser.add_argument("--seed", type=int, default=123)
        parser.add_argument("--output", type=str, default=None, help="Output stem for measured JSON and NPZ")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.save()
