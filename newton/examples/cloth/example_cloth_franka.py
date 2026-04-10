# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
#
# The simulation runs in centimeter scale for better numerical behavior
# of the VBD solver. A vis_state is used to convert back to meter scale
# for visualization.
#
# Command: python -m newton.examples cloth_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import ModelBuilder, State
from newton.solvers import SolverFeatherstone, SolverVBD


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


class Example:
    def __init__(self, viewer, args):
        # parameters
        #   simulation (centimeter scale)
        self.add_cloth = True
        self.add_robot = True
        self.show_trajectory = getattr(args, "show_trajectory", False)
        self.show_targets = getattr(args, "show_targets", False)
        self.pause_on_keyframe = getattr(args, "pause_on_keyframe", False)
        self._current_keyframe = 0
        self._keyframe_elapsed = 0.0
        self._ee_error = 0.0
        self._finger_error = 0.0
        self._keyframe_advanced = False
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # visualization: simulation in cm, viewer in meters
        self.viz_scale = 0.01

        #   contact (cm scale)
        #       body-cloth contact
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 1.5
        #       self-contact
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 0.1

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 0.01
        self.robot_contact_mu = 1.5

        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-6

        self.bending_ke = 5
        self.bending_kd = 1e-2

        self.scene = ModelBuilder(gravity=-981.0)

        self.viewer = viewer

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # add a table (cm scale)
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0)
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(
                self.table_pos_cm,
                wp.quat_identity(),
            ),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        # add the T-shirt
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")

        shirt_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = shirt_mesh.vertices
        mesh_indices = shirt_mesh.indices
        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                pos=wp.vec3(0.0, 70.0, 30.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.02,
                scale=1.0,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )

            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Hide the table box from automatic shape rendering -- the GL viewer
        # bakes primitive dimensions into the mesh and ignores shape_scale,
        # so we render it manually at meter scale in render() instead.
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Pre-compute meter-scale table viz data
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        # Explicit collision pipeline for cloth-body contacts with custom margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_ik()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=self.robot_contact_ke,
            )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        # Pre-allocate visualization arrays
        n_targets = len(self.targets)
        target_positions_cm = self.targets[:, :3]
        self.viz_target_points = wp.array((target_positions_cm * self.viz_scale).astype(np.float32), dtype=wp.vec3)
        self.viz_target_radii = wp.array(np.full(n_targets, 0.008, dtype=np.float32))
        self.viz_target_colors = wp.array(np.tile([0.2, 1.0, 0.2], (n_targets, 1)).astype(np.float32), dtype=wp.vec3)
        max_frames = getattr(args, "num_frames", 4800)
        self.viz_trajectory_points = wp.zeros(max_frames, dtype=wp.vec3)
        self.viz_trajectory_radii = wp.array(np.full(max_frames, 0.003, dtype=np.float32))
        self.viz_trajectory_colors = wp.array(
            np.tile([1.0, 0.2, 0.2], (max_frames, 1)).astype(np.float32), dtype=wp.vec3
        )
        self.trajectory_count = 0

        # Visualization state for meter-scale rendering
        self.viz_state = self.model.state()

        # Pre-compute scaled shape data for meter-scale visualization.
        # Two paths need updating:
        #   1) The GL viewer's CUDA path reads model.shape_transform / model.shape_scale
        #      directly, so we swap them temporarily in render().
        #   2) The base viewer path caches shapes.xforms / shapes.scales during
        #      set_model(), so we permanently scale those cached copies here.
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Scale the viewer's cached shape instance data (base viewer / GL fallback path)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        # gravity in cm/s²
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def set_up_ik(self):
        """Set up IK solver for end-effector control."""
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=self.ee_link_offset,
            target_positions=wp.array([wp.vec3(*self.targets[0][:3])], dtype=wp.vec3),
        )

        target_quat = self.targets[0][3:7]
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=self.ee_link_rotation,
            target_rotations=wp.array([wp.vec4(*target_quat)], dtype=wp.vec4),
        )

        self.joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        self.ik_joint_q = wp.zeros((1, self.model.joint_coord_count), dtype=float)

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            lambda_initial=0.1,
        )

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-50.0, -50.0, 0.0),
                wp.quat_identity(),
            ),
            floating=False,
            scale=100,  # URDF is in meters, scale to cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
        builder.joint_q[7] = 3.2  # left finger open (0.8 * 4.0 cm)
        builder.joint_q[8] = 3.2  # right finger open (0.8 * 4.0 cm)

        # Gripper activation as a fraction of the finger joint range (0-1).
        # Scaled by the joint upper limit (4 cm at scale=100) to get finger
        # positions: close = 0.4 cm/finger, open = 3.2 cm/finger.
        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8

        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position [cm], 4D quaternion), gripper activation
                # top left (tilted gripper for +x reachability)
                [6.5, 31.0, -60.0, 23.0, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                [2, 31.0, -60.0, 21.25, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                [2, 31.0, -60.0, 21.25, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [2, 26.0, -60.0, 26.0, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [2, 12.0, -60.0, 31.0, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [3, -6.0, -60.0, 31.0, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [1, -6.0, -60.0, 31.0, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                [1, -6.0, -60.0, 35.0, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                # bottom left (tilted gripper for +x reachability)
                [2, 15.0, -33.0, 31.0, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.25, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                [2, 15.0, -33.0, 21.25, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                [2, 15.0, -33.0, 21.25, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [2, 15.0, -33.0, 28.0, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [3, -2.0, -33.0, 28.0, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [1, -2.0, -33.0, 31.0, 0.9239, 0, 0.3827, 0, clamp_close_activation_val],
                [1, -2.0, -33.0, 31.0, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                [1, -2.0, -33.0, 35.0, 0.9239, 0, 0.3827, 0, clamp_open_activation_val],
                # top right
                [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 21.25, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 21.25, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 21.25, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -18.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [1, 5.0, -60.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [3, -18.0, -30.0, 21.25, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -18.0, -30.0, 21.25, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -18.0, -30.0, 21.25, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [1, -3.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom (tilted gripper toward +y for reachability)
                [2, 0.0, -20.0, 30.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_open_activation_val],
                [2, 0.0, -20.0, 21.25, 0.6533, 0.6533, 0.2706, -0.2706, clamp_open_activation_val],
                [2, 0.0, -20.0, 21.25, 0.6533, 0.6533, 0.2706, -0.2706, clamp_open_activation_val],
                [1, 0.0, -20.0, 21.25, 0.6533, 0.6533, 0.2706, -0.2706, clamp_close_activation_val],
                [2, 0.0, -20.0, 21.25, 0.6533, 0.6533, 0.2706, -0.2706, clamp_close_activation_val],
                [2, 0.0, -20.0, 35.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_close_activation_val],
                [1, 0.0, -30.0, 35.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_close_activation_val],
                [1.5, 0.0, -30.0, 35.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_close_activation_val],
                [1, 0.0, -40.0, 31.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 31.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_open_activation_val],
                [1, 0.0, -40.0, 35.0, 0.6533, 0.6533, 0.2706, -0.2706, clamp_open_activation_val],
                [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        # Offset from the end-effector body (link8) to the gripper tip.
        # The translation (22 cm along z) reaches the fingertip from link8.
        # The rotation accounts for the collapsed fr3_hand_joint fixed joint,
        # which introduces a -pi/4 yaw between link8 and the hand frame
        # (see fr3_franka_hand.urdf).  Without this rotation the IK drives
        # toward link8's frame and the gripper ends up with a constant
        # 45-degree yaw offset from the desired orientation.
        self.ee_link_offset = wp.vec3(0.0, 0.0, 21.04)
        self.ee_link_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 4.0)

    def generate_control_joint_qd(self, state_in: State):
        # Deferred pause: if a keyframe advanced last frame, pause now
        # so the viewer shows the state *before* the new keyframe starts.
        if self._keyframe_advanced:
            self._current_keyframe += 1
            self._keyframe_elapsed = 0.0
            self._keyframe_advanced = False

        # After the key poses sequence ends, hold position with zero velocity
        if self._current_keyframe >= len(self.targets):
            self.target_joint_qd.zero_()
            return

        # Compute EE tip error and finger error for convergence check
        body_q_np = state_in.body_q.numpy()
        ee_tf = body_q_np[self.endeffector_id]
        ee_xform = wp.transform(*ee_tf)
        ee_tip = wp.transform_point(ee_xform, self.ee_link_offset)
        target_pos = self.targets[self._current_keyframe][:3]
        self._ee_error = float(np.linalg.norm(np.array([ee_tip[0], ee_tip[1], ee_tip[2]]) - target_pos))

        current_q = state_in.joint_q.numpy()
        finger_target = float(self.targets[self._current_keyframe][-1]) * 4.0
        self._finger_error = max(abs(current_q[-2] - finger_target), abs(current_q[-1] - finger_target))

        # Advance to next keyframe only when the soft time limit has elapsed,
        # the EE tip has converged, AND the fingers have reached their target.
        soft_limit = float(self.transition_duration[self._current_keyframe])
        if self._keyframe_elapsed >= soft_limit and self._ee_error < 0.1 and self._finger_error < 0.1:
            self._keyframe_advanced = True
            if self.pause_on_keyframe and hasattr(self.viewer, "_paused"):
                self.viewer._paused = True
            return

        self._keyframe_elapsed += self.frame_dt

        current_interval = self._current_keyframe

        # Interpolate toward the current keyframe target using elapsed time
        # as a fraction of the soft time limit.
        alpha = float(np.clip(self._keyframe_elapsed / soft_limit, 0.0, 1.0))

        target_cur = self.targets[current_interval]
        target_prev = self.targets[current_interval - 1] if current_interval > 0 else target_cur
        target = (1.0 - alpha) * target_prev + alpha * target_cur

        # Update IK targets from interpolated pose
        self.pos_obj.set_target_position(0, wp.vec3(float(target[0]), float(target[1]), float(target[2])))
        self.rot_obj.set_target_rotation(
            0, wp.vec4(float(target[3]), float(target[4]), float(target[5]), float(target[6]))
        )

        # Seed IK with current joint positions
        ik_flat = self.ik_joint_q.reshape((self.model.joint_coord_count,))
        wp.copy(ik_flat, state_in.joint_q)

        # Solve IK for target joint positions
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=24)

        # Compute joint velocities from position difference
        current_q = state_in.joint_q.numpy()
        target_q = ik_flat.numpy()
        delta_q = target_q - current_q

        # Apply gripper finger control. Activation values (0-1) are scaled to
        # the finger joint range (0-4 cm at scale=100).
        finger_pos_target = target[-1] * 4.0
        delta_q[-2] = 1.0 * (finger_pos_target - current_q[-2])
        delta_q[-1] = 1.0 * (finger_pos_target - current_q[-1])

        self.target_joint_qd.assign(delta_q)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        # Scale particle and body positions from cm to meters for visualization
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Swap model shape data to meter-scale for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        # Accumulate EE tip position for trajectory visualization (circular buffer)
        if self.add_robot:
            ee_body_q = self.viz_state.body_q.numpy()[self.endeffector_id]
            ee_xform = wp.transform(*ee_body_q)
            tip_pos = wp.transform_point(ee_xform, self.ee_link_offset * self.viz_scale)
            buf_size = self.viz_trajectory_points.shape[0]
            idx = self.trajectory_count % buf_size
            wp.copy(
                self.viz_trajectory_points,
                wp.array([tip_pos], dtype=wp.vec3),
                dest_offset=idx,
                count=1,
            )
            self.trajectory_count += 1

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        # Render the table box manually at meter scale
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )
        if self.show_trajectory and self.trajectory_count > 0:
            n = min(self.trajectory_count, self.viz_trajectory_points.shape[0])
            self.viewer.log_points(
                "/ee_trajectory",
                self.viz_trajectory_points[:n],
                radii=self.viz_trajectory_radii[:n],
                colors=self.viz_trajectory_colors[:n],
            )
        if self.show_targets:
            self.viewer.log_points(
                "/targets",
                self.viz_target_points,
                radii=self.viz_target_radii,
                colors=self.viz_target_colors,
            )
        self.viewer.end_frame()

        # Restore simulation shape data
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def render_ui(self, imgui):
        _changed, self.show_trajectory = imgui.checkbox("Show Trajectory", self.show_trajectory)
        _changed, self.show_targets = imgui.checkbox("Show Targets", self.show_targets)
        _changed, self.pause_on_keyframe = imgui.checkbox("Pause on Keyframe", self.pause_on_keyframe)
        imgui.text(f"EE error: {self._ee_error:.2f} cm")
        imgui.text(f"Finger error: {self._finger_error:.3f} cm")
        imgui.text(f"Keyframe: {self._current_keyframe} / {len(self.targets)}")
        imgui.text(f"Elapsed: {self._keyframe_elapsed:.2f} s")
        if self._current_keyframe < len(self.transition_duration):
            imgui.text(f"Soft limit: {self.transition_duration[self._current_keyframe]:.1f} s")

    def test_final(self):
        p_lower = wp.vec3(-36.0, -95.0, -5.0)
        p_upper = wp.vec3(36.0, 5.0, 56.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=4800)
    parser.add_argument("--show-trajectory", action="store_true", help="Draw EE tip trajectory as spheres")
    parser.add_argument("--show-targets", action="store_true", help="Draw keyframe target positions as spheres")
    parser.add_argument("--pause-on-keyframe", action="store_true", help="Pause viewer at each keyframe transition")
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
