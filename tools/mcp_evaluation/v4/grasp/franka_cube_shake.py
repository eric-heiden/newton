# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Franka cube grasp and shake-test environment.

The scene and controller follow Newton's ``brick_stacking`` example: a fixed
FR3 is position controlled by analytical IK while a MuJoCo solver advances the
articulation and a Newton collision pipeline handles contacts.
"""

from __future__ import annotations

import enum
import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

CUBE_SIZE = 0.04
CUBE_DENSITY = 500.0
GRIPPER_OPEN = 0.035
GRIPPER_CLOSED = -0.01


class Phase(enum.IntEnum):
    """Phases of the scripted grasp-and-shake task."""

    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    SHAKE = 4


@wp.kernel(enable_backward=False)
def update_task_targets(
    phase_durations: wp.array[wp.float32],
    phase_index: wp.array[wp.int32],
    phase_time: wp.array[wp.float32],
    frame_dt: wp.float32,
    phase_start_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: wp.int32,
    cube_index: wp.int32,
    approach_height: wp.float32,
    lift_height: wp.float32,
    shake_amplitude: wp.float32,
    shake_frequency: wp.float32,
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_interpolated: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_interpolated: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    """Create a smooth Cartesian target for the current task phase."""
    phase = phase_index[0]
    phase_time[0] = phase_time[0] + frame_dt

    duration = phase_durations[phase]
    t_linear = wp.min(1.0, phase_time[0] / duration)
    t = t_linear * t_linear * (3.0 - 2.0 * t_linear)

    ee_start = phase_start_body_q[ee_index]
    start_pos = wp.transform_get_translation(ee_start)
    start_quat = wp.transform_get_rotation(ee_start)
    cube_pos = wp.transform_get_translation(body_q[cube_index])
    down_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    target_pos = start_pos
    target_quat = down_quat
    gripper_pos = GRIPPER_OPEN

    if phase == Phase.APPROACH.value:
        target_pos = cube_pos + wp.vec3(0.0, 0.0, approach_height)
    elif phase == Phase.DESCEND.value:
        target_pos = cube_pos
    elif phase == Phase.GRASP.value:
        target_pos = start_pos
        target_quat = start_quat
        gripper_pos = GRIPPER_OPEN * (1.0 - t) + GRIPPER_CLOSED * t
    elif phase == Phase.LIFT.value:
        target_pos = start_pos + wp.vec3(0.0, 0.0, lift_height)
        target_quat = start_quat
        gripper_pos = GRIPPER_CLOSED
    elif phase == Phase.SHAKE.value:
        # A three-axis Lissajous-style path exercises the grasp in more than
        # one direction without introducing discontinuities at frame edges.
        omega_t = 2.0 * wp.pi * shake_frequency * phase_time[0]
        target_pos = start_pos + wp.vec3(
            shake_amplitude * wp.sin(omega_t),
            0.65 * shake_amplitude * wp.sin(2.0 * omega_t),
            0.35 * shake_amplitude * wp.sin(1.5 * omega_t),
        )
        target_quat = start_quat
        gripper_pos = GRIPPER_CLOSED
        t = 1.0
    ee_pos_target[0] = target_pos
    ee_pos_interpolated[0] = start_pos * (1.0 - t) + target_pos * t
    ee_rot_target[0] = target_quat[:4]
    ee_rot_interpolated[0] = wp.quat_slerp(start_quat, target_quat, t)[:4]
    gripper_target[0, 0] = gripper_pos
    gripper_target[0, 1] = gripper_pos


@wp.kernel(enable_backward=False)
def advance_task_phase(
    phase_durations: wp.array[wp.float32],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    ee_index: wp.int32,
    # outputs
    phase_index: wp.array[wp.int32],
    phase_time: wp.array[wp.float32],
    phase_start_body_q: wp.array[wp.transform],
):
    """Advance settled setup phases, then remain in SHAKE forever."""
    phase = phase_index[0]
    if phase >= Phase.SHAKE.value or phase_time[0] < phase_durations[phase]:
        return

    ee_q = body_q[ee_index]
    ee_pos = wp.transform_get_translation(ee_q)
    ee_quat = wp.transform_get_rotation(ee_q)
    target_quat = wp.quaternion(ee_rot_target[0][:3], ee_rot_target[0][3])

    pos_error = wp.length(ee_pos_target[0] - ee_pos)
    quat_error = ee_quat * wp.quat_inverse(target_quat)
    rot_error = wp.degrees(2.0 * wp.acos(wp.clamp(wp.abs(quat_error[3]), 0.0, 1.0)))

    # Setup phases wait until the arm has reached their final pose. SHAKE is
    # handled by the early return above and continues until the viewer closes.
    ready = pos_error < 0.005 and rot_error < 2.0
    if ready:
        phase_index[0] = phase + 1
        phase_time[0] = 0.0
        for body_index in range(wp.len(body_q)):
            phase_start_body_q[body_index] = body_q[body_index]


def create_parser():
    """Create the standard Newton example CLI with shake-test options."""
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=900)
    parser.add_argument(
        "--shake-amplitude",
        type=float,
        default=0.03,
        help="Peak side-to-side shake displacement in meters.",
    )
    parser.add_argument(
        "--shake-frequency",
        type=float,
        default=1.0,
        help="Base shake frequency in hertz.",
    )
    return parser


class FrankaCubeShake:
    """Newton example/environment that grasps, lifts, and shakes one cube."""

    create_parser = staticmethod(create_parser)

    def __init__(self, viewer, args=None):
        newton.use_coord_layout_targets = True
        if args is None:
            args = newton.examples.default_args(create_parser())
        if args.shake_amplitude < 0.0:
            raise ValueError("shake amplitude must be non-negative")
        if args.shake_frequency <= 0.0:
            raise ValueError("shake frequency must be positive")
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cube_size = CUBE_SIZE
        self.robot_base_pos = wp.vec3(-0.5, -0.5, 0.0)
        self.cube_start_pos = wp.vec3(0.0, -0.44, 0.5 * self.cube_size)

        self.approach_height = 0.10
        self.lift_height = 0.20
        self.shake_amplitude = float(args.shake_amplitude)
        self.shake_frequency = float(args.shake_frequency)

        robot_builder = self._build_franka()
        self.model_ik = robot_builder.finalize()
        self.ee_index = self._body_index(self.model_ik, "fr3_hand_tcp")

        # Starting at the approach pose makes the grasp deterministic and keeps
        # the first visible motion focused on the object.
        initial_arm_q = self._solve_initial_approach()
        robot_builder.joint_q[:7] = initial_arm_q.tolist()
        robot_builder.joint_q[7:9] = [GRIPPER_OPEN, GRIPPER_OPEN]
        robot_builder.joint_target_q[:9] = robot_builder.joint_q[:9]

        scene = newton.ModelBuilder()
        scene.add_builder(robot_builder)
        self.cube_index = self._add_cube(scene)
        scene.add_ground_plane(
            color=(0.45, 0.45, 0.45),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, gap=0.005),
        )
        self.model = scene.finalize()

        contact_max = 4096
        self.model.rigid_contact_max = contact_max
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=contact_max,
            broad_phase="nxn",
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=15,
            ls_iterations=100,
            nconmax=contact_max,
            njmax=contact_max * 2,
            cone="elliptic",
            impratio=50.0,
            use_mujoco_contacts=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_q[:9], self.model.joint_q[:9])

        self._setup_ik()
        self._setup_task()

        self.viewer.set_model(self.model)
        # Viewer picking feeds drag forces into ``viewer.apply_forces`` in the
        # simulation loop, so the robot and cube can be perturbed interactively.
        self.viewer.picking_enabled = True
        camera_pos = wp.vec3(0.35, -0.85, 0.30)
        self.viewer.set_camera(pos=camera_pos, pitch=-25.0, yaw=135.0)
        self._capture_graphs()

    @staticmethod
    def _body_index(model, short_label: str) -> int:
        for index, label in enumerate(model.body_label):
            if label.rsplit("/", 1)[-1] == short_label:
                return index
        raise ValueError(f"body {short_label!r} not found in Franka model")

    def _build_franka(self):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.005
        # Compliant robot contacts avoid jitter when the hand touches the table.
        builder.default_shape_cfg.ke = 250.0
        builder.default_shape_cfg.kd = 10.0
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        ready_q = [0.0, 0.5, 0.0, -1.5, 0.0, 2.0, math.pi / 4.0, GRIPPER_OPEN, GRIPPER_OPEN]
        builder.joint_q[:9] = ready_q
        builder.joint_target_q[:9] = ready_q
        builder.joint_target_ke[:9] = [400.0] * 7 + [400.0, 400.0]
        builder.joint_target_kd[:9] = [40.0] * 7 + [40.0, 40.0]
        builder.joint_effort_limit[:9] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        joint_gravity = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        joint_gravity.values = joint_gravity.values or {}
        for dof_index in range(7):
            joint_gravity.values[dof_index] = True

        body_gravity = builder.custom_attributes["mujoco:gravcomp"]
        body_gravity.values = body_gravity.values or {}
        for body_index in range(2, 14):
            body_gravity.values[body_index] = 1.0

        # Give finger contacts priority so both pads maintain a firm grasp.
        solimp = builder.custom_attributes.get("mujoco:geom_solimp")
        priority = builder.custom_attributes.get("mujoco:geom_priority")
        if solimp is not None and priority is not None:
            solimp.values = solimp.values or {}
            priority.values = priority.values or {}
            for shape_index, body_index in enumerate(builder.shape_body):
                if body_index in (12, 13):
                    solimp.values[shape_index] = (0.7, 0.95, 0.0001, 0.5, 2.0)
                    priority.values[shape_index] = 1

        return builder

    def _add_cube(self, scene) -> int:
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=CUBE_DENSITY,
            ke=8.0e4,
            kd=8.0e2,
            mu=1.2,
        )
        cube_index = scene.add_body(
            xform=wp.transform(self.cube_start_pos, wp.quat_identity()),
            label="shake_cube",
        )
        scene.add_shape_box(
            body=cube_index,
            hx=0.5 * self.cube_size,
            hy=0.5 * self.cube_size,
            hz=0.5 * self.cube_size,
            cfg=cube_cfg,
            color=(0.15, 0.45, 0.9),
        )
        return cube_index

    def _solve_initial_approach(self) -> np.ndarray:
        target_pos = self.cube_start_pos + wp.vec3(0.0, 0.0, self.approach_height)
        down_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
        dof_count = self.model_ik.joint_coord_count
        seed = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 2.0, math.pi / 4.0, GRIPPER_OPEN, GRIPPER_OPEN], dtype=np.float32)
        joint_q = wp.array(seed.reshape(1, dof_count), dtype=wp.float32)
        solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[
                ik.IKObjectivePosition(
                    link_index=self.ee_index,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([target_pos], dtype=wp.vec3),
                ),
                ik.IKObjectiveRotation(
                    link_index=self.ee_index,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([down_quat[:4]], dtype=wp.vec4),
                ),
                ik.IKObjectiveJointLimit(
                    joint_limit_lower=self.model_ik.joint_limit_lower[:dof_count],
                    joint_limit_upper=self.model_ik.joint_limit_upper[:dof_count],
                    weight=10.0,
                ),
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        for _ in range(30):
            solver.step(joint_q, joint_q, iterations=24)
        return joint_q.flatten().numpy()[:7]

    def _setup_ik(self):
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        ee_q = state.body_q.numpy()[self.ee_index]
        self.position_objective = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*ee_q[:3])], dtype=wp.vec3),
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*ee_q[3:7])], dtype=wp.vec4),
        )

        dof_count = self.model_ik.joint_coord_count
        limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.clone(self.model_ik.joint_limit_lower[:dof_count]),
            joint_limit_upper=wp.clone(self.model_ik.joint_limit_upper[:dof_count]),
            weight=10.0,
        )
        self.joint_q_ik = wp.clone(self.model.joint_q[:dof_count].reshape((1, dof_count)))
        self.ik_iterations = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective, limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _setup_task(self):
        self.phase_durations = wp.array(
            [0.25, 1.0, 0.75, 1.5, 1.0],
            dtype=wp.float32,
        )
        self.phase_index = wp.zeros(1, dtype=wp.int32)
        self.phase_time = wp.zeros(1, dtype=wp.float32)
        self.phase_start_body_q = wp.clone(self.state_0.body_q)
        self.ee_pos_target = wp.zeros(1, dtype=wp.vec3)
        self.ee_pos_interpolated = wp.zeros(1, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(1, dtype=wp.vec4)
        self.ee_rot_interpolated = wp.zeros(1, dtype=wp.vec4)
        self.gripper_target = wp.zeros((1, 2), dtype=wp.float32)

    def _set_joint_targets(self):
        wp.launch(
            update_task_targets,
            dim=1,
            inputs=[
                self.phase_durations,
                self.phase_index,
                self.phase_time,
                self.frame_dt,
                self.phase_start_body_q,
                self.state_0.body_q,
                self.ee_index,
                self.cube_index,
                self.approach_height,
                self.lift_height,
                self.shake_amplitude,
                self.shake_frequency,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_interpolated,
                self.ee_rot_target,
                self.ee_rot_interpolated,
                self.gripper_target,
            ],
        )
        self.position_objective.set_target_positions(self.ee_pos_interpolated)
        self.rotation_objective.set_target_rotations(self.ee_rot_interpolated)
        if self.ik_graph is not None:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)

        wp.copy(self.control.joint_target_q[:7], self.joint_q_ik.flatten()[:7])
        wp.copy(self.control.joint_target_q[7:9], self.gripper_target.flatten())
        wp.launch(
            advance_task_phase,
            dim=1,
            inputs=[
                self.phase_durations,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.ee_index,
            ],
            outputs=[self.phase_index, self.phase_time, self.phase_start_body_q],
        )

    def _simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_graphs(self):
        self.sim_graph = None
        self.ik_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.sim_graph = capture.graph
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)
            self.ik_graph = capture.graph

    def reset(self):
        """Restore the initial scene and restart the scripted task."""
        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_q[:9], self.model.joint_q[:9])
        dof_count = self.model_ik.joint_coord_count
        self.joint_q_ik = wp.clone(self.model.joint_q[:dof_count].reshape((1, dof_count)))
        self._setup_task()
        self._capture_graphs()

    def step(self):
        """Advance the controller and physics by one 60 Hz frame."""
        self._set_joint_targets()
        if self.sim_graph is not None:
            wp.capture_launch(self.sim_graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Assert that the ongoing shake has started without dropping the cube."""
        phase = int(self.phase_index.numpy()[0])
        if phase != Phase.SHAKE.value:
            raise ValueError(f"shake sequence incomplete: reached {Phase(phase).name}, expected SHAKE")
        shake_time = float(self.phase_time.numpy()[0])
        if shake_time < 5.0:
            raise ValueError(f"shake ran for only {shake_time:.2f} s; expected at least 5.0 s")

        body_q = self.state_0.body_q.numpy()
        cube_pos = body_q[self.cube_index, :3]
        ee_pos = body_q[self.ee_index, :3]
        if not np.all(np.isfinite(cube_pos)):
            raise ValueError(f"cube has a non-finite pose: {cube_pos}")

        separation = float(np.linalg.norm(cube_pos - ee_pos))
        if separation > 0.08:
            raise ValueError(f"cube was dropped during shake: cube/TCP separation is {separation:.3f} m")
        if cube_pos[2] < self.cube_size:
            raise ValueError(f"cube was not lifted after shake: cube height is {cube_pos[2]:.3f} m")


# Newton's runner and example browser conventionally look for ``Example``.
Example = FrankaCubeShake


def main():
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(FrankaCubeShake(viewer, args), args)


if __name__ == "__main__":
    main()
