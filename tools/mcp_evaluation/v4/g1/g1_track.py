# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 following a Kimodo reference motion with joint-space PD actuators.

The floating-base G1 stands on the ground and tracks the 29 reference joint
angles of a MuJoCo-qpos CSV clip (root xyz, root quaternion wxyz, 29 joints at
30 fps). Actuators are MuJoCo position servos with torque limits from the
robot's MJCF; the controller sets their gains and targets every frame.

Run: ``python g1_track.py --motion wave.csv --viewer null --num-frames 250``
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

# Reflected actuator inertia per joint (BeyondMimic G1 configuration) [kg m^2].
LEG = [0.010177520, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450]
ARM = [0.003609725] * 5 + [0.00425] * 2
ARMATURE = np.array(LEG * 2 + [0.010177520, 0.007219450, 0.007219450] + ARM * 2)


def quat_wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


class MotionClip:
    """Linear interpolation of a qpos CSV in time (root quaternion by normalized lerp)."""

    def __init__(self, path, fps=30.0):
        self.qpos = np.loadtxt(path, delimiter=",")
        if self.qpos.ndim != 2 or self.qpos.shape[1] != 36:
            raise ValueError("Expected rows of 36 qpos values (root pos, root quat wxyz, 29 joints)")
        self.fps = fps
        self.duration = (len(self.qpos) - 1) / fps

    def sample(self, t):
        """Reference joint_q in Newton layout (root pos, root quat xyzw, joints) at time t [s]."""
        x = np.clip(t, 0.0, self.duration) * self.fps
        i = min(int(x), len(self.qpos) - 2)
        a = x - i
        q = (1 - a) * self.qpos[i] + a * self.qpos[i + 1]
        q[3:7] /= np.linalg.norm(q[3:7])
        return np.concatenate([q[:3], quat_wxyz_to_xyzw(q[3:7]), q[7:]])


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.motion = MotionClip(args.motion)

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        asset = newton.utils.download_asset("unitree_g1")
        builder.add_mjcf(str(asset / "mjcf/g1_29dof_rev_1_0.xml"), collapse_fixed_joints=True)
        # Robot self-collision is excluded; feet and body still collide with the floor.
        robot_shapes = [i for i, body in enumerate(builder.shape_body) if body >= 0]
        for i, shape in enumerate(robot_shapes):
            for other in robot_shapes[i + 1 :]:
                builder.add_shape_collision_filter_pair(shape, other)
        builder.joint_armature[6:] = ARMATURE.tolist()

        # ---------------- Controller (tunable) ----------------
        # Joint-space PD servos: stiffness kp [N m/rad] and damping kd [N m s/rad]
        # per actuated joint, here derived from a 10 Hz natural frequency.
        natural_frequency = 2.0 * np.pi * 10.0
        self.kp = ARMATURE * natural_frequency**2
        self.kd = 4.0 * ARMATURE * natural_frequency
        # -------------------------------------------------------

        builder.joint_target_ke[6:] = self.kp.tolist()
        builder.joint_target_kd[6:] = self.kd.tolist()
        builder.joint_target_mode[6:] = [int(newton.JointTargetMode.POSITION)] * 29
        builder.custom_attributes["mujoco:ctrl_source"].values = [
            int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET)
        ] * 29
        builder.custom_attributes["mujoco:actuator_has_forcerange"].values = [True] * 29
        builder.custom_attributes["mujoco:actuator_forcelimited"].values = [1] * 29
        builder.custom_attributes["mujoco:actuator_forcerange"].values = [
            (-limit, limit) for limit in builder.joint_effort_limit[6:]
        ]

        # Start at the first reference frame with the feet just touching the ground.
        q0 = self.motion.sample(0.0)
        builder.joint_q[:] = q0.tolist()
        self.model = builder.finalize()
        self.floor_offset = self._floor_offset(q0)
        self.model.joint_q.assign(self._shifted(q0))
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=192,
            nconmax=64,
            use_mujoco_contacts=True,
            integrator="implicitfast",
            iterations=50,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _floor_offset(self, q0):
        """Vertical shift [m] that puts the lowest foot collision point 2 mm above the floor."""
        state = self.model.state()
        self.model.joint_q.assign(q0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        body_q = state.body_q.numpy()
        labels = [label.rsplit("/", 1)[-1] for label in self.model.body_label]
        lowest = np.inf
        for side in ("left", "right"):
            pose = body_q[labels.index(f"{side}_ankle_roll_link")]
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:7]))).reshape(3, 3)
            for x in (-0.05, 0.12):
                for y in (-0.025, 0.025):
                    lowest = min(lowest, (pose[:3] + rotation @ np.array([x, y, -0.035]))[2])
        return 0.002 - lowest

    def _shifted(self, q):
        q = q.copy()
        q[2] += self.floor_offset
        return q

    def reference(self, t):
        """Reference joint_q (with the floor shift applied) at time t [s]."""
        return self._shifted(self.motion.sample(t))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # ---------------- Controller (tunable) ----------------
        # Position targets for the servos: the reference joint angles one frame ahead.
        target = self.reference(self.sim_time + self.frame_dt)
        self.control.joint_target_q.assign(target)
        # -------------------------------------------------------
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--motion", type=str, required=False, default="wave.csv", help="Kimodo G1 qpos CSV")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
