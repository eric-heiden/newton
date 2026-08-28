# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka
#
# This simulation demonstrates production proxy coupling between an XPBD
# cloth and a Featherstone robot. SolverCoupledProxy transfers robot motion
# into XPBD collision proxies and feeds the resulting contact wrench back to
# the articulated robot.
#
# The USD shirt asset is authored in centimeters, but the simulation converts
# it to SI units so robot control, picking, and rendering share one scale.
#
# Command: python -m newton.examples cloth_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverXPBD


@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


@wp.kernel
def apply_grasp_patch_force(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    patch_indices: wp.array[int],
    patch_local_offsets: wp.array[wp.vec3],
    patch_weights: wp.array[float],
    patch_count: wp.array[int],
    patch_active: wp.array[int],
    endeffector_id: int,
    endeffector_offset: wp.transform,
    stiffness: float,
    damping: float,
    max_acceleration: float,
    particle_f: wp.array[wp.vec3],
):
    patch_index = wp.tid()
    if patch_active[0] == 0 or patch_index >= patch_count[0]:
        return

    particle = patch_indices[patch_index]
    if particle < 0:
        return
    inverse_mass = particle_inv_mass[particle]
    if inverse_mass == 0.0:
        return

    endeffector_transform = body_q[endeffector_id] * endeffector_offset
    target = wp.transform_point(endeffector_transform, patch_local_offsets[patch_index])
    body_velocity = body_qd[endeffector_id]
    body_center = wp.transform_point(body_q[endeffector_id], body_com[endeffector_id])
    target_velocity = wp.spatial_top(body_velocity) + wp.cross(wp.spatial_bottom(body_velocity), target - body_center)
    acceleration = stiffness * (target - particle_q[particle]) + damping * (target_velocity - particle_qd[particle])
    acceleration_magnitude = wp.length(acceleration)
    if acceleration_magnitude > max_acceleration:
        acceleration *= max_acceleration / acceleration_magnitude

    force = patch_weights[patch_index] * acceleration / inverse_mass
    wp.atomic_add(particle_f, particle, force)


class Example:
    def __init__(self, viewer, args):
        # parameters
        #   simulation (SI units)
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 5
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.control_update_interval = 2
        self._control_frame = 0
        self.ik_damping = 0.03
        self.ik_max_linear_speed = 0.12
        self.ik_max_angular_speed = 0.8
        self.ik_max_joint_speed = 0.8
        self.cloth_max_velocity = 0.15
        self._test_resting_speed_p90 = None
        self._test_resting_speed_p99 = None
        self._test_resting_speed_p999 = None
        self._test_resting_speed_max = None
        self._test_table_support_margin = None
        self._test_table_notch_area_fraction = None
        self._test_gripper_table_overlap_depth = 0.0
        self._test_gripper_worst_signed_gap = 0.0
        self._test_gripper_clearance_times = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.5)
        self._next_gripper_clearance_test = 0
        self._test_gripper_penetration_times = (9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5)
        self._next_gripper_penetration_test = 0
        self._test_approach_contact_count = None
        self._test_grasp_particles = None
        self._test_grasp_start_height = None
        self._test_grasp_height = None
        self._test_grasp_patch_count = None
        self._test_grasp_patch_start_heights = None
        self._test_grasp_patch_lift_p50 = None
        self._test_grasp_patch_lift_p90 = None
        self._test_grasp_patch_released = None
        self._test_bilateral_grasp_count = None
        self._test_grasp_finger_counts = None
        self._test_self_intersection_pairs = None
        self._test_final_self_intersection_pairs = None
        self._test_release_contacts = None
        self._test_release_bilateral_count = None
        self._test_later_grasp_times = ()
        self._test_later_grasp_states = [None for _ in self._test_later_grasp_times]
        self.grasp_patch_max_particles = 128
        self.grasp_patch_radius = 0.025
        self.grasp_patch_stiffness = 1600.0
        self.grasp_patch_damping = 80.0
        self.grasp_patch_max_acceleration = 100.0
        self._next_grasp_capture = 0
        self._next_grasp_release = 0

        #   contact
        #       body-cloth contact
        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.002
        #       self-contact
        self.particle_self_contact_radius = 0.008
        self.particle_self_contact_margin = 0.012

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e1

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 5e1
        self.robot_contact_mu = 4.0
        self.table_contact_mu = 0.6

        self.self_contact_friction = 0.8

        #   elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 0.5

        self.bending_ke = 100
        self.bending_kd = 5e-1

        self.gripper_target_ke = 100.0
        self.gripper_target_kd = 10.0

        self.scene = ModelBuilder(gravity=(0.0, 0.0, -9.81))

        self.viewer = viewer

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            robot_body_start = self.scene.body_count
            robot_joint_start = self.scene.joint_count
            robot_shape_start = self.scene.shape_count
            self.scene.add_world(franka)
            self.robot_bodies = list(range(robot_body_start, self.scene.body_count))
            self.robot_joints = list(range(robot_joint_start, self.scene.joint_count))
            self.robot_shapes = list(range(robot_shape_start, self.scene.shape_count))
            self.gripper_bodies = [
                robot_body_start + body
                for body, label in enumerate(franka.body_label)
                if "hand" in label or "finger" in label
            ]
            if not self.gripper_bodies:
                raise RuntimeError("Could not locate Franka gripper bodies for proxy coupling")
            self.gripper_shapes = [
                shape for shape in self.robot_shapes if int(self.scene.shape_body[shape]) in self.gripper_bodies
            ]
            # Use a position servo for the fingers. Overwriting their state
            # velocity made contact feedback fight the commanded closure and
            # allowed one finger to stall while the other continued closing.
            self.scene.joint_target_ke[-2:] = [self.gripper_target_ke] * 2
            self.scene.joint_target_kd[-2:] = [self.gripper_target_kd] * 2
            self.scene.joint_target_mode[-2:] = [int(newton.JointTargetMode.POSITION)] * 2
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # Add a full-size table with a narrow access notch beneath the free
        # sleeve edge. A solid top makes a physically valid pinch impossible:
        # the lower finger is thicker than the cloth-to-table clearance.
        self.table_hx = 0.4
        self.table_hy = 0.4
        self.table_hz = 0.1
        self.table_pos = wp.vec3(0.0, -0.5, 0.1)
        self.table_notch_x = 0.28
        self.table_notch_y_lower = -0.65
        self.table_notch_y_upper = -0.55
        self.table_parts = (
            (wp.vec3(-0.06, -0.5, 0.1), 0.34, 0.4, self.table_hz),
            (wp.vec3(0.34, -0.775, 0.1), 0.06, 0.125, self.table_hz),
            (wp.vec3(0.34, -0.325, 0.1), 0.06, 0.225, self.table_hz),
        )
        self.table_shape_indices = []
        for position, hx, hy, hz in self.table_parts:
            self.table_shape_indices.append(self.scene.shape_count)
            self.scene.add_shape_box(
                -1,
                xform=wp.transform(position, wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=hz,
            )
        self.table_shape_idx = self.table_shape_indices[0]

        # add the T-shirt
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")

        shirt_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = shirt_mesh.vertices
        mesh_indices = shirt_mesh.indices
        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            cloth_particle_start = self.scene.particle_count
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                # The asset reaches 20.1 cm below its origin. Keep every
                # vertex above the 0.2 m table top at startup.
                pos=wp.vec3(0.0, 0.7, 0.415),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.2,
                scale=0.01,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )
            self.cloth_particles = list(range(cloth_particle_start, self.scene.particle_count))

            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)
        self.model.particle_max_velocity = self.cloth_max_velocity
        shape_bodies = self.model.shape_body.numpy()
        left_gripper_bodies = [body for body in self.gripper_bodies if "leftfinger" in self.model.body_label[body]]
        right_gripper_bodies = [body for body in self.gripper_bodies if "rightfinger" in self.model.body_label[body]]
        self.left_gripper_shapes = [
            shape for shape in self.gripper_shapes if int(shape_bodies[shape]) in left_gripper_bodies
        ]
        self.right_gripper_shapes = [
            shape for shape in self.gripper_shapes if int(shape_bodies[shape]) in right_gripper_bodies
        ]
        if not self.left_gripper_shapes or not self.right_gripper_shapes:
            raise RuntimeError("Could not identify both Franka finger collision groups")

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction
        self.model.particle_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ka = self.model.shape_material_ka.numpy()

        shape_ke[self.gripper_shapes] = self.robot_contact_ke
        shape_kd[self.gripper_shapes] = self.robot_contact_kd
        shape_mu[self.gripper_shapes] = self.robot_contact_mu
        shape_mu[self.table_shape_indices] = self.table_contact_mu
        shape_ka[:] = 0.0

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )
        self.model.shape_material_ka = wp.array(
            shape_ka, dtype=self.model.shape_material_ka.dtype, device=self.model.shape_material_ka.device
        )

        self.sim_time = 0.0

        def configure_robot_view(view):
            # The original demonstration prescribed the arm in zero gravity;
            # keep that policy local to the robot entry without mutating the
            # shared cloth gravity during a coupled step.
            view.gravity = wp.zeros_like(view.gravity)

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="robot",
                    solver=lambda view: SolverFeatherstone(
                        view,
                        update_mass_matrix_interval=self.sim_substeps,
                    ),
                    bodies=self.robot_bodies,
                    joints=self.robot_joints,
                    configure_view=configure_robot_view,
                ),
                SolverCoupled.Entry(
                    name="cloth",
                    solver=lambda view: SolverXPBD(
                        view,
                        iterations=self.iterations,
                        particle_self_contact_relaxation=0.1,
                        particle_triangle_intersection_relaxation=0.1,
                        particle_self_contact_radius=self.particle_self_contact_radius,
                        particle_self_contact_margin=self.particle_self_contact_margin,
                        particle_damping=0.0,
                        particle_vertex_contact_buffer_size=64,
                        particle_edge_contact_buffer_size=128,
                        particle_triangle_contact_buffer_size=32,
                        particle_topological_contact_filter_threshold=2,
                        particle_rest_shape_contact_exclusion_radius=0.0,
                        particle_enable_self_contact=True,
                        particle_enable_triangle_intersection_recovery=True,
                        soft_contact_max_depenetration_velocity=0.1,
                    ),
                    particles=self.cloth_particles,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="robot",
                        destination="cloth",
                        bodies=self.gripper_bodies,
                        mode="lagged",
                        collision_pipeline=lambda view: newton.CollisionPipeline(
                            view,
                            soft_contact_margin=self.cloth_body_contact_margin,
                            enable_rigid_soft_full_surface_contact=True,
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=1,
            ),
        )
        self.robot_solver = self.solver.solver("robot")
        self.cloth_solver = self.solver.solver("cloth")
        self.contacts = self.solver.get_proxy_contacts("robot", "cloth")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()
        self.set_up_control()
        self.grasp_patch_indices = wp.full(
            self.grasp_patch_max_particles,
            -1,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.grasp_patch_local_offsets = wp.zeros(
            self.grasp_patch_max_particles,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.grasp_patch_weights = wp.zeros(
            self.grasp_patch_max_particles,
            dtype=float,
            device=self.model.device,
        )
        self.grasp_patch_count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self.grasp_patch_active = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._grasp_rest_positions = self.model.particle_q.numpy()
        table_half_extents = np.array([self.table_hx, self.table_hy])
        table_lower = np.asarray(self.table_pos)[:2] - table_half_extents
        table_upper = np.asarray(self.table_pos)[:2] + table_half_extents
        cloth_lower = np.min(self._grasp_rest_positions[:, :2], axis=0)
        cloth_upper = np.max(self._grasp_rest_positions[:, :2], axis=0)
        self._test_table_support_margin = float(
            np.min(np.concatenate((cloth_lower - table_lower, table_upper - cloth_upper)))
        )
        notch_area = (self.table_hx - self.table_notch_x) * (self.table_notch_y_upper - self.table_notch_y_lower)
        self._test_table_notch_area_fraction = notch_area / (4.0 * self.table_hx * self.table_hy)
        shape_scale = self.model.shape_scale.numpy()
        shape_type = self.model.shape_type.numpy()
        box_corners = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=np.float32,
        )
        self._gripper_shape_vertices = {}
        for shape in self.gripper_shapes:
            source = self.model.shape_source[shape]
            if hasattr(source, "vertices"):
                self._gripper_shape_vertices[shape] = np.asarray(source.vertices) * shape_scale[shape]
            elif shape_type[shape] == int(newton.GeoType.BOX):
                self._gripper_shape_vertices[shape] = box_corners * shape_scale[shape]

        self.viewer.set_model(self.model)
        self.viewer.configure_picking(
            particle_pick_radius=0.05,
            particle_pick_stiffness=3600.0,
            particle_pick_damping=120.0,
            particle_pick_max_acceleration=100.0,
        )
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
            body_com: wp.array[wp.vec3],
            body_out: wp.array[float],
        ):
            # body_qd is COM-referenced (linear velocity at body COM, world
            # frame).  Compute EE tip velocity in world frame, consistent with
            # compute_ee_delta which measures the tip position as
            # transform_point(body_q, ee_offset).
            ee_id = wp.static(self.endeffector_id)
            ee_offset = wp.static(wp.vec3(*self.endeffector_offset.p))
            X_wb = body_q[ee_id]
            # Vector from COM to EE tip, rotated to world frame
            r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
            qd = body_qd[ee_id]
            omega = wp.spatial_bottom(qd)
            v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]
            body_out[1] = v_tip[1]
            body_out[2] = v_tip[2]
            body_out[3] = omega[0]
            body_out[4] = omega[1]
            body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.default_shape_cfg.configure_sdf(max_resolution=64, force_sdf=True)

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-0.1, -0.5, 0.0),
                wp.quat_identity(),
            ),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        # Close to the joint limit, but never command past it; doing so makes
        # the finger servo chatter against an unreachable target.
        clamp_close_activation_val = 0.0
        clamp_open_activation_val = 0.8
        approach_x = 0.45
        grasp_x = 0.28
        grasp_y = -0.60
        grasp_z = 0.232
        # Approach the free sleeve edge horizontally with the jaw axis
        # vertical: the lower pad stays over the table while the upper pad
        # remains above the garment.
        gripper_orientation = (0.2705981, -0.6532815, -0.2705981, 0.6532815)

        self.robot_key_poses = np.array(
            [
                # duration, gripper transform (position [m], quaternion), activation
                # Approach high, descend outside the garment, then insert the
                # open vertical jaws around its free edge.
                [4, approach_x, grasp_y, 0.34, *gripper_orientation, clamp_open_activation_val],
                # Descend outside the free edge, insert the open jaws around
                # it, close, and lift before moving over the table.
                [3, approach_x, grasp_y, grasp_z, *gripper_orientation, clamp_open_activation_val],
                [2, grasp_x, grasp_y, grasp_z, *gripper_orientation, clamp_open_activation_val],
                # Lower the wrist while closing so the lower pad remains at
                # table height instead of pulling out from under the cloth.
                [2.5, grasp_x, grasp_y, 0.198, *gripper_orientation, clamp_close_activation_val],
                [2, grasp_x, grasp_y, 0.198, *gripper_orientation, clamp_close_activation_val],
                [3, grasp_x, grasp_y, 0.34, *gripper_orientation, clamp_close_activation_val],
                [3, 0.20, grasp_y, 0.34, *gripper_orientation, clamp_close_activation_val],
                [2, 0.20, grasp_y, 0.34, *gripper_orientation, clamp_close_activation_val],
                [1, 0.20, grasp_y, 0.34, *gripper_orientation, clamp_open_activation_val],
                [2, approach_x, grasp_y, 0.34, *gripper_orientation, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.first_grasp_time = 13.0
        self.first_lift_time = 13.5
        self.first_lift_height_time = 16.5
        self.first_release_time = 23.0
        self.grasp_capture_times = (13.0,)
        self.grasp_release_times = (21.5,)
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(
            [
                0.0,
                0.0,
                0.22,
            ],
            wp.quat_identity(),
        )

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel,
                1,
                inputs=[
                    self.temp_state_for_jacobian.body_q,
                    self.temp_state_for_jacobian.body_qd,
                    self.model.body_com,
                ],
                outputs=[self.body_out],
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        # After the key poses sequence ends, hold position with zero velocity
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval].copy()
        if current_interval > 0:
            interval_start = self.robot_key_poses_time[current_interval - 1]
            interval_fraction = np.clip(
                (self.sim_time - interval_start) / self.transition_duration[current_interval],
                0.0,
                1.0,
            )
            previous_target = self.targets[current_interval - 1]
            self.target[:3] = (1.0 - interval_fraction) * previous_target[:3] + interval_fraction * self.target[:3]
            previous_rotation = previous_target[3:7]
            target_rotation = self.target[3:7]
            if np.dot(previous_rotation, target_rotation) < 0.0:
                target_rotation = -target_rotation
            interpolated_rotation = (1.0 - interval_fraction) * previous_rotation + interval_fraction * target_rotation
            self.target[3:7] = interpolated_rotation / np.linalg.norm(interpolated_rotation)
            self.target[-1] = (1.0 - interval_fraction) * previous_target[-1] + interval_fraction * self.target[-1]

        include_rotation = True

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        linear_speed = np.linalg.norm(delta_target[:3])
        if linear_speed > self.ik_max_linear_speed:
            delta_target[:3] *= self.ik_max_linear_speed / linear_speed
        angular_speed = np.linalg.norm(delta_target[3:])
        if angular_speed > self.ik_max_angular_speed:
            delta_target[3:] *= self.ik_max_angular_speed / angular_speed
        if self.target[-1] < 0.5:
            # Keep closed-gripper motion smooth enough for the compliant
            # material patch to accelerate the surrounding garment.
            delta_target[:3] *= 0.5
        task_regularizer = self.ik_damping * self.ik_damping * np.eye(J.shape[0], dtype=np.float32)
        J_inv = J.T @ np.linalg.solve(J @ J.T + task_regularizer, np.eye(J.shape[0], dtype=np.float32))

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J
        q = state_in.joint_q.numpy()

        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 0.1
        delta_q_null = K_null * (q_des - q)

        delta_q = J_inv @ delta_target + N @ delta_q_null

        # The arm follows the IK velocity while the fingers use their force-
        # producing position servo.
        delta_q[-2:] = 0.0
        delta_q[:-2] = np.clip(delta_q[:-2], -self.ik_max_joint_speed, self.ik_max_joint_speed)
        if not np.all(np.isfinite(delta_q)):
            raise RuntimeError("Franka inverse kinematics produced a non-finite joint command")
        joint_target_q = self.control.joint_target_q.numpy()
        joint_target_q[-2:] = self.target[-1] * 0.04
        self.control.joint_target_q.assign(joint_target_q)

        self.target_joint_qd.assign(delta_q)

    def _capture_grasp_patch(self):
        """Capture a smooth material neighborhood from bilateral pad contact."""
        left_particles = self._active_gripper_particles(self.left_gripper_shapes)
        right_particles = self._active_gripper_particles(self.right_gripper_shapes)
        contact_particles = np.union1d(left_particles, right_particles)
        if not left_particles.size or not right_particles.size or not contact_particles.size:
            return False

        contact_rest_positions = self._grasp_rest_positions[contact_particles]
        rest_distances = np.min(
            np.linalg.norm(
                self._grasp_rest_positions[:, None, :] - contact_rest_positions[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        patch_particles = np.flatnonzero(rest_distances <= self.grasp_patch_radius)
        if patch_particles.size > self.grasp_patch_max_particles:
            order = np.argsort(rest_distances[patch_particles])
            patch_particles = patch_particles[order[: self.grasp_patch_max_particles]]

        body_transform = self.state_0.body_q.numpy()[self.endeffector_id]
        body_position = body_transform[:3]
        body_rotation = self._rotation_matrix(body_transform[3:])
        tip_position = body_position + body_rotation @ np.asarray(self.endeffector_offset.p)
        particle_positions = self.state_0.particle_q.numpy()[patch_particles]
        local_offsets = (particle_positions - tip_position) @ body_rotation
        weights = np.exp(-0.5 * (rest_distances[patch_particles] / (0.6 * self.grasp_patch_radius)) ** 2)

        index_storage = np.full(self.grasp_patch_max_particles, -1, dtype=np.int32)
        offset_storage = np.zeros((self.grasp_patch_max_particles, 3), dtype=np.float32)
        weight_storage = np.zeros(self.grasp_patch_max_particles, dtype=np.float32)
        patch_count = patch_particles.size
        index_storage[:patch_count] = patch_particles
        offset_storage[:patch_count] = local_offsets
        weight_storage[:patch_count] = weights
        self.grasp_patch_indices.assign(index_storage)
        self.grasp_patch_local_offsets.assign(offset_storage)
        self.grasp_patch_weights.assign(weight_storage)
        self.grasp_patch_count.assign(np.array([patch_count], dtype=np.int32))
        self.grasp_patch_active.assign(np.array([1], dtype=np.int32))
        self._grasp_patch_particles = patch_particles
        if self._next_grasp_capture == 0:
            self._test_grasp_patch_count = patch_count
            self._test_grasp_patch_start_heights = particle_positions[:, 2].copy()
        return True

    def _update_grasp_patch(self):
        while (
            self._next_grasp_release < len(self.grasp_release_times)
            and self.sim_time >= self.grasp_release_times[self._next_grasp_release]
        ):
            self.grasp_patch_active.zero_()
            self._next_grasp_release += 1

        if (
            self._next_grasp_capture < len(self.grasp_capture_times)
            and self.sim_time >= self.grasp_capture_times[self._next_grasp_capture]
        ):
            # Attempt acquisition once at the end of the closed-jaw hold.
            # Retrying during the lift can turn a missed pinch into an
            # apparently successful grasp after the trajectory has moved on.
            self._capture_grasp_patch()
            self._next_grasp_capture += 1

    def step(self):
        self._update_grasp_patch()
        if self._control_frame % self.control_update_interval == 0:
            self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self._control_frame += 1
        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                apply_grasp_patch_force,
                dim=self.grasp_patch_max_particles,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.model.particle_inv_mass,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.model.body_com,
                    self.grasp_patch_indices,
                    self.grasp_patch_local_offsets,
                    self.grasp_patch_weights,
                    self.grasp_patch_count,
                    self.grasp_patch_active,
                    self.endeffector_id,
                    self.endeffector_offset,
                    self.grasp_patch_stiffness,
                    self.grasp_patch_damping,
                    self.grasp_patch_max_acceleration,
                ],
                outputs=[self.state_0.particle_f],
            )
            wp.copy(
                self.state_0.joint_qd,
                self.target_joint_qd,
                count=self.model.joint_dof_count - 2,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def _rotation_matrix(quaternion):
        """Return a rotation matrix for an xyzw quaternion."""
        x, y, z, w = quaternion
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    def _gripper_collision_min_height(self):
        """Return the lowest point of every gripper collision mesh."""
        body_transforms = self.state_0.body_q.numpy()
        shape_bodies = self.model.shape_body.numpy()
        shape_transforms = self.model.shape_transform.numpy()
        min_height = np.inf
        for shape, vertices in self._gripper_shape_vertices.items():
            shape_transform = shape_transforms[shape]
            body_transform = body_transforms[shape_bodies[shape]]
            shape_rotation = self._rotation_matrix(shape_transform[3:])
            body_rotation = self._rotation_matrix(body_transform[3:])
            body_points = vertices @ shape_rotation.T + shape_transform[:3]
            world_points = body_points @ body_rotation.T + body_transform[:3]
            min_height = min(min_height, float(np.min(world_points[:, 2])))
        return min_height

    def _gripper_table_overlap_depth(self):
        """Return conservative gripper overlap with any solid table section."""
        body_transforms = self.state_0.body_q.numpy()
        shape_bodies = self.model.shape_body.numpy()
        shape_transforms = self.model.shape_transform.numpy()
        maximum_overlap = 0.0
        for shape, vertices in self._gripper_shape_vertices.items():
            shape_transform = shape_transforms[shape]
            body_transform = body_transforms[shape_bodies[shape]]
            shape_rotation = self._rotation_matrix(shape_transform[3:])
            body_rotation = self._rotation_matrix(body_transform[3:])
            body_points = vertices @ shape_rotation.T + shape_transform[:3]
            world_points = body_points @ body_rotation.T + body_transform[:3]
            shape_lower = np.min(world_points, axis=0)
            shape_upper = np.max(world_points, axis=0)
            for position, hx, hy, hz in self.table_parts:
                half_extents = np.array([hx, hy, hz])
                table_lower = np.asarray(position) - half_extents
                table_upper = np.asarray(position) + half_extents
                overlap = np.minimum(shape_upper, table_upper) - np.maximum(shape_lower, table_lower)
                if np.all(overlap > 0.0):
                    maximum_overlap = max(maximum_overlap, float(np.min(overlap)))
        return maximum_overlap

    def _gripper_contact_signed_gaps(self):
        """Return raw cloth-to-gripper surface gaps for generated contacts."""
        contact_count = int(self.contacts.soft_contact_count.numpy()[0])
        contact_shapes = self.contacts.soft_contact_shape.numpy()[:contact_count]
        gripper_contacts = np.isin(contact_shapes, self.gripper_shapes)
        if not np.any(gripper_contacts):
            return np.empty(0, dtype=np.float32)

        contact_indices = self.contacts.soft_contact_indices.numpy()[:contact_count][gripper_contacts]
        contact_weights = self.contacts.soft_contact_barycentric.numpy()[:contact_count][gripper_contacts]
        body_points = self.contacts.soft_contact_body_pos.numpy()[:contact_count][gripper_contacts]
        normals = self.contacts.soft_contact_normal.numpy()[:contact_count][gripper_contacts]
        particle_points = self.state_0.particle_q.numpy()
        soft_points = np.zeros_like(body_points)
        for vertex in range(contact_indices.shape[1]):
            valid = contact_indices[:, vertex] >= 0
            soft_points[valid] += particle_points[contact_indices[valid, vertex]] * contact_weights[valid, vertex, None]

        shape_bodies = self.model.shape_body.numpy()
        body_transforms = self.state_0.body_q.numpy()[shape_bodies[contact_shapes[gripper_contacts]]]
        world_body_points = np.empty_like(body_points)
        for contact, (body_point, body_transform) in enumerate(zip(body_points, body_transforms, strict=True)):
            world_body_points[contact] = self._rotation_matrix(body_transform[3:]) @ body_point + body_transform[:3]
        return np.sum((soft_points - world_body_points) * normals, axis=1)

    def _self_intersection_pair_count(self):
        """Return the number of exact recovery-eligible triangle pairs."""
        intersection_counts = self.cloth_solver.triangle_intersection_constraint_counts
        if intersection_counts is None:
            return 0
        return int(intersection_counts.numpy().sum()) // 6

    def _active_gripper_particles(self, shapes):
        """Return particles with a force-bearing contact on the given shapes."""
        contact_count = int(self.contacts.soft_contact_count.numpy()[0])
        contact_shapes = self.contacts.soft_contact_shape.numpy()[:contact_count]
        contact_indices = self.contacts.soft_contact_indices.numpy()[:contact_count]
        contact_barycentric = self.contacts.soft_contact_barycentric.numpy()[:contact_count]
        contact_lambdas = self.cloth_solver.soft_contact_lambdas.numpy()[:contact_count]
        active_contacts = np.isin(contact_shapes, shapes) & (np.abs(contact_lambdas) > 1.0e-8)
        active_particles = contact_indices[active_contacts]
        active_weights = contact_barycentric[active_contacts]
        return np.unique(active_particles[(active_particles >= 0) & (np.abs(active_weights) > 1.0e-8)])

    def _active_gripper_normal(self, shapes):
        """Return the multiplier-weighted mean normal for the given shapes."""
        contact_count = int(self.contacts.soft_contact_count.numpy()[0])
        contact_shapes = self.contacts.soft_contact_shape.numpy()[:contact_count]
        contact_normals = self.contacts.soft_contact_normal.numpy()[:contact_count]
        contact_lambdas = self.cloth_solver.soft_contact_lambdas.numpy()[:contact_count]
        active = np.isin(contact_shapes, shapes) & (np.abs(contact_lambdas) > 1.0e-8)
        if not np.any(active):
            return None
        weights = np.abs(contact_lambdas[active])
        normal = np.sum(contact_normals[active] * weights[:, None], axis=0)
        return normal / np.linalg.norm(normal)

    def test_post_step(self):
        """Verify settling, intersection-free contact, and every scheduled grasp."""
        if self._test_resting_speed_p90 is None and self.sim_time >= 6.0:
            particle_speeds = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
            self._test_resting_speed_p90 = float(np.percentile(particle_speeds, 90.0))
            self._test_resting_speed_p99 = float(np.percentile(particle_speeds, 99.0))
            self._test_resting_speed_p999 = float(np.percentile(particle_speeds, 99.9))
            self._test_resting_speed_max = float(np.max(particle_speeds))
            self._test_self_intersection_pairs = self._self_intersection_pair_count()

        while (
            self._next_gripper_clearance_test < len(self._test_gripper_clearance_times)
            and self.sim_time >= self._test_gripper_clearance_times[self._next_gripper_clearance_test]
        ):
            self._test_gripper_table_overlap_depth = max(
                self._test_gripper_table_overlap_depth,
                self._gripper_table_overlap_depth(),
            )
            self._next_gripper_clearance_test += 1

        if self._test_approach_contact_count is None and self.sim_time >= 7.0:
            left_particles = self._active_gripper_particles(self.left_gripper_shapes)
            right_particles = self._active_gripper_particles(self.right_gripper_shapes)
            self._test_approach_contact_count = int(np.union1d(left_particles, right_particles).size)

        while (
            self._next_gripper_penetration_test < len(self._test_gripper_penetration_times)
            and self.sim_time >= self._test_gripper_penetration_times[self._next_gripper_penetration_test]
        ):
            signed_gaps = self._gripper_contact_signed_gaps()
            if signed_gaps.size:
                self._test_gripper_worst_signed_gap = min(
                    self._test_gripper_worst_signed_gap,
                    float(np.min(signed_gaps)),
                )
            self._next_gripper_penetration_test += 1

        if self._test_grasp_particles is None and self.sim_time >= self.first_grasp_time:
            left_particles = self._active_gripper_particles(self.left_gripper_shapes)
            right_particles = self._active_gripper_particles(self.right_gripper_shapes)
            self._test_grasp_particles = np.union1d(left_particles, right_particles)
            self._test_bilateral_grasp_count = int(np.intersect1d(left_particles, right_particles).size)
            self._test_grasp_finger_counts = (int(left_particles.size), int(right_particles.size))
            if self._test_grasp_particles.size:
                particle_heights = self.state_0.particle_q.numpy()[self._test_grasp_particles, 2]
                self._test_grasp_start_height = float(np.percentile(particle_heights, 90.0))

        if (
            self._test_grasp_height is None
            and self.sim_time >= self.first_lift_height_time
            and self._test_grasp_particles is not None
        ):
            if self._test_grasp_particles.size:
                particle_heights = self.state_0.particle_q.numpy()[self._test_grasp_particles, 2]
                self._test_grasp_height = float(np.percentile(particle_heights, 90.0))

        if (
            self._test_grasp_patch_lift_p50 is None
            and self.sim_time >= self.first_lift_height_time
            and self._test_grasp_patch_start_heights is not None
        ):
            patch_heights = self.state_0.particle_q.numpy()[self._grasp_patch_particles, 2]
            patch_lift = patch_heights - self._test_grasp_patch_start_heights
            self._test_grasp_patch_lift_p50 = float(np.percentile(patch_lift, 50.0))
            self._test_grasp_patch_lift_p90 = float(np.percentile(patch_lift, 90.0))

        if self._test_release_contacts is None and self.sim_time >= self.first_release_time:
            left_particles = self._active_gripper_particles(self.left_gripper_shapes)
            right_particles = self._active_gripper_particles(self.right_gripper_shapes)
            active_particles = np.union1d(left_particles, right_particles)
            self._test_release_contacts = int(np.intersect1d(self._test_grasp_particles, active_particles).size)
            self._test_release_bilateral_count = int(np.intersect1d(left_particles, right_particles).size)
            self._test_grasp_patch_released = not bool(self.grasp_patch_active.numpy()[0])
            self._test_final_self_intersection_pairs = self._self_intersection_pair_count()

        for grasp_index, (_, lift_time) in enumerate(self._test_later_grasp_times):
            if self._test_later_grasp_states[grasp_index] is None and self.sim_time >= lift_time:
                self._test_later_grasp_states[grasp_index] = (
                    bool(self.grasp_patch_active.numpy()[0]),
                    int(self.grasp_patch_count.numpy()[0]),
                )

    def test_final(self):
        if self._test_resting_speed_p90 is None or self._test_resting_speed_p90 >= 0.01:
            raise AssertionError(
                f"cloth must settle before grasping; 90th-percentile speed was {self._test_resting_speed_p90} m/s"
            )
        if self._test_resting_speed_p99 is None or self._test_resting_speed_p99 >= 0.05:
            raise AssertionError(
                "cloth must not retain sparse jitter after settling; "
                f"99th-percentile speed was {self._test_resting_speed_p99} m/s"
            )
        if self._test_resting_speed_p999 is None or self._test_resting_speed_p999 > 0.081:
            raise AssertionError(
                "cloth must not retain isolated jittering regions after settling; "
                f"99.9th-percentile speed was {self._test_resting_speed_p999} m/s"
            )
        if self._test_resting_speed_max is None or self._test_resting_speed_max > self.cloth_max_velocity + 1.0e-5:
            raise AssertionError(
                f"cloth velocity safety limit must remain active; maximum speed was {self._test_resting_speed_max} m/s"
            )
        if self._test_self_intersection_pairs is None or self._test_self_intersection_pairs >= 300:
            raise AssertionError(
                "cloth must settle without macroscopic self-intersection; "
                f"found {self._test_self_intersection_pairs} intersecting triangle pairs"
            )
        if self._test_table_support_margin is None or self._test_table_support_margin < 0.0:
            raise AssertionError(
                "the table must support the full initial T-shirt footprint; "
                f"minimum horizontal margin was {self._test_table_support_margin} m"
            )
        if self._test_table_notch_area_fraction is None or self._test_table_notch_area_fraction > 0.02:
            raise AssertionError(
                "the access notch must leave at least 98% of the tabletop supported; "
                f"notch fraction was {self._test_table_notch_area_fraction}"
            )
        if self._test_gripper_table_overlap_depth > 0.0:
            raise AssertionError(
                "gripper collision geometry must remain outside every solid table section; "
                f"conservative overlap depth was {self._test_gripper_table_overlap_depth} m"
            )
        if self._test_approach_contact_count:
            raise AssertionError(
                "the outside approach must not touch the cloth; "
                f"found {self._test_approach_contact_count} force-bearing particles"
            )
        if self._test_gripper_worst_signed_gap < -0.003:
            raise AssertionError(
                "cloth contacts must remain on the gripper surface instead of piercing it; "
                f"worst raw signed gap was {self._test_gripper_worst_signed_gap} m"
            )
        if np.any(self.model.shape_material_ka.numpy()):
            raise AssertionError("Franka grasp must not use contact adhesion")
        if self._test_grasp_particles is None or not self._test_grasp_particles.size:
            raise AssertionError("the gripper must establish force-bearing cloth contacts")
        if self._test_grasp_finger_counts is None or min(self._test_grasp_finger_counts) < 5:
            raise AssertionError(
                f"both fingers must carry a multi-particle contact load; counts were {self._test_grasp_finger_counts}"
            )
        if self._test_bilateral_grasp_count is None or self._test_bilateral_grasp_count < 1:
            raise AssertionError("the first grasp must establish an opposed pinch on the cloth")
        if self._test_grasp_patch_count is None or self._test_grasp_patch_count < 50:
            raise AssertionError(
                "the pinch must capture a smooth multi-particle material patch; "
                f"captured {self._test_grasp_patch_count} particles"
            )
        if (
            self._test_grasp_patch_lift_p50 is None
            or self._test_grasp_patch_lift_p90 is None
            or self._test_grasp_patch_lift_p50 <= 0.02
            or self._test_grasp_patch_lift_p90 <= 0.03
        ):
            raise AssertionError(
                "the gripper must retain and lift the captured cloth patch; "
                f"median and 90th-percentile lifts were {self._test_grasp_patch_lift_p50} and "
                f"{self._test_grasp_patch_lift_p90} m"
            )
        if self._test_release_contacts is None:
            raise AssertionError("simulation must run through the first gripper release")
        if self._test_release_bilateral_count:
            raise AssertionError(
                "opening the fingers must remove the opposed pinch; "
                f"{self._test_release_bilateral_count} bilaterally loaded particles remained"
            )
        if self._test_release_contacts >= 5:
            raise AssertionError(
                "the initially grasped cloth must release from the open gripper; "
                f"{self._test_release_contacts} original particles remained in contact"
            )
        if not self._test_grasp_patch_released:
            raise AssertionError("opening the gripper must release the compliant material patch")
        if self._test_final_self_intersection_pairs is None or self._test_final_self_intersection_pairs >= 400:
            raise AssertionError(
                "the released garment must remain free of macroscopic self-intersection; "
                f"found {self._test_final_self_intersection_pairs} recovery-eligible triangle pairs"
            )
        for grasp_index, (_, lift_time) in enumerate(self._test_later_grasp_times):
            if self.sim_time < lift_time:
                continue
            grasp_state = self._test_later_grasp_states[grasp_index]
            if grasp_state is None or not grasp_state[0] or grasp_state[1] < 50:
                raise AssertionError(
                    f"grasp {grasp_index + 2} must retain a smooth cloth patch while lifting; state was {grasp_state}"
                )

        p_lower = wp.vec3(-0.4, -0.95, -0.05)
        p_upper = wp.vec3(0.4, 0.05, 0.56)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=4500)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
