# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SolverMuJoCo joint-coordinate bridge.

:meth:`SolverMuJoCo.convert_joint_coords_to_mujoco` and
:meth:`SolverMuJoCo.convert_joint_coords_from_mujoco` are checked against
MuJoCo's own kinematics on a model MuJoCo compiles from MJCF directly, against
the solver's internal conversion kernels, and for exact round trips.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import JointType, ModelFlags
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_test_devices

try:
    import mujoco
except ImportError:  # pragma: no cover - mujoco is a test dependency
    mujoco = None


# Free root with a rotated, offset inertial frame; ball shoulder; hinge and
# slide joints with ``ref``; a body with two hinges (a Newton D6 joint).
MJCF = """<mujoco model="bridge">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="torso" pos="0.1 -0.2 1.0" quat="0.9238795 0.0 0.3826834 0.0">
      <freejoint name="root"/>
      <inertial pos="0.05 -0.02 0.1" quat="0.9659258 0.2588190 0 0" mass="4" diaginertia="0.2 0.3 0.25"/>
      <body name="upper_arm" pos="0.2 0.1 0" quat="0.8775826 0 0 0.4794255">
        <joint name="shoulder" type="ball"/>
        <inertial pos="0 0.15 0" mass="1" diaginertia="0.02 0.01 0.02"/>
        <body name="forearm" pos="0 0.3 0">
          <joint name="elbow" type="hinge" axis="1 0 0" ref="0.3"/>
          <inertial pos="0 0.1 0.02" mass="0.6" diaginertia="0.01 0.005 0.01"/>
          <body name="hand" pos="0 0.2 0">
            <joint name="wrist" type="slide" axis="0 1 0" ref="0.05"/>
            <inertial pos="0 0.03 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
          </body>
        </body>
      </body>
      <body name="thigh" pos="0 -0.1 -0.2">
        <joint name="hip_y" type="hinge" axis="0 1 0"/>
        <joint name="hip_x" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 -0.2" mass="2" diaginertia="0.05 0.05 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _random_qpos(mj_model, rng, batch):
    """Random MuJoCo positions with unit quaternions, shape [batch, nq]."""
    qpos = rng.normal(scale=0.5, size=(batch, mj_model.nq))
    for j in range(mj_model.njnt):
        adr = mj_model.jnt_qposadr[j]
        if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qpos[:, adr + 3 : adr + 7] = _unit(rng.normal(size=(batch, 4)))
        elif mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
            qpos[:, adr : adr + 4] = _unit(rng.normal(size=(batch, 4)))
    return qpos


def _random_joint_q(model, rng, batch):
    """Random Newton joint coordinates with unit quaternions, shape [batch, joint_coord_count]."""
    q = rng.normal(scale=0.5, size=(batch, model.joint_coord_count))
    q_start = model.joint_q_start.numpy()
    for j, jtype in enumerate(model.joint_type.numpy()):
        if jtype == JointType.FREE:
            q[:, q_start[j] + 3 : q_start[j] + 7] = _unit(rng.normal(size=(batch, 4)))
        elif jtype == JointType.BALL:
            q[:, q_start[j] : q_start[j] + 4] = _unit(rng.normal(size=(batch, 4)))
    return q


def _newton_body_states(model, joint_q, joint_qd):
    """Newton forward kinematics: body_q [m, xyzw] and body_qd (COM linear, angular, world frame)."""
    state = model.state()
    state.joint_q.assign(np.asarray(joint_q, dtype=np.float32))
    state.joint_qd.assign(np.asarray(joint_qd, dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state.body_q.numpy().astype(np.float64), state.body_qd.numpy().astype(np.float64)


def _mujoco_body_states(mj_model, qpos, qvel):
    """MuJoCo kinematics per MuJoCo body: (xpos, xquat as xyzw, COM linear velocity, angular velocity), world frame."""
    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(mj_model, data)
    vel = np.zeros((mj_model.nbody, 6))
    for body in range(mj_model.nbody):
        # mjOBJ_BODY: velocity of the body's inertial (COM) frame, [angular, linear] in world orientation.
        mujoco.mj_objectVelocity(mj_model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel[body], 0)
    return data.xpos.copy(), data.xquat[:, [1, 2, 3, 0]].copy(), vel[:, 3:], vel[:, :3]


def _quat_distance(a, b):
    return np.minimum(np.abs(a - b).max(axis=-1), np.abs(a + b).max(axis=-1))


class _BridgeAssertions:
    def assert_kinematics_match(self, model, mj_model, newton_bodies, joint_q, joint_qd, qpos, qvel, atol=1e-5):
        """Newton FK of (joint_q, joint_qd) matches MuJoCo kinematics of (qpos, qvel) for the given bodies."""
        body_q, body_qd = _newton_body_states(model, joint_q, joint_qd)
        xpos, xquat, v_com, w = _mujoco_body_states(mj_model, qpos, qvel)
        for mj_body, newton_body in newton_bodies.items():
            np.testing.assert_allclose(body_q[newton_body, :3], xpos[mj_body], atol=atol)
            self.assertLess(_quat_distance(body_q[newton_body, 3:], xquat[mj_body]), atol)
            np.testing.assert_allclose(body_qd[newton_body, :3], v_com[mj_body], atol=atol)
            np.testing.assert_allclose(body_qd[newton_body, 3:], w[mj_body], atol=atol)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestJointCoordsNativeMuJoCo(_BridgeAssertions, unittest.TestCase):
    """Compare against a model that MuJoCo compiles from the MJCF itself."""

    @classmethod
    def setUpClass(cls):
        cls.native = mujoco.MjModel.from_xml_string(MJCF)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(MJCF)
        cls.model = builder.finalize(device="cpu")
        cls.solver = SolverMuJoCo(cls.model, use_mujoco_cpu=True, disable_contacts=True)
        labels = [label.rsplit("/", 1)[-1] for label in cls.model.body_label]
        cls.bodies = {cls.native.body(name).id: labels.index(name) for name in labels}

    def test_solver_model_layout_matches_native(self):
        # The comparisons below rely on the solver's MuJoCo model using the native qpos layout.
        mj_model = self.solver.mj_model
        self.assertEqual((mj_model.nq, mj_model.nv), (self.native.nq, self.native.nv))
        np.testing.assert_array_equal(mj_model.jnt_type, self.native.jnt_type)
        np.testing.assert_array_equal(mj_model.jnt_qposadr, self.native.jnt_qposadr)
        self.assertIn(JointType.D6, self.model.joint_type.numpy())

    def test_from_mujoco_matches_native_kinematics(self):
        rng = np.random.default_rng(0)
        qpos = _random_qpos(self.native, rng, 4)
        qvel = rng.normal(size=(4, self.native.nv))
        joint_q, joint_qd = self.solver.convert_joint_coords_from_mujoco(qpos, qvel)
        self.assertEqual(joint_q.shape, (4, self.model.joint_coord_count))
        self.assertEqual(joint_qd.shape, (4, self.model.joint_dof_count))
        for i in range(4):
            with self.subTest(sample=i):
                self.assert_kinematics_match(
                    self.model, self.native, self.bodies, joint_q[i], joint_qd[i], qpos[i], qvel[i]
                )

    def test_to_mujoco_matches_native_kinematics(self):
        rng = np.random.default_rng(1)
        joint_q = _random_joint_q(self.model, rng, 3)
        joint_qd = rng.normal(size=(3, self.model.joint_dof_count))
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        self.assertEqual(qpos.shape, (3, self.native.nq))
        for i in range(3):
            with self.subTest(sample=i):
                self.assert_kinematics_match(
                    self.model, self.native, self.bodies, joint_q[i], joint_qd[i], qpos[i], qvel[i]
                )

    def test_scalar_joints_add_ref(self):
        joint_q = self.model.joint_q.numpy().astype(np.float64)
        qpos, _ = self.solver.convert_joint_coords_to_mujoco(joint_q)
        # Newton's default coordinates are relative to the authored pose, MuJoCo's qpos0 is absolute.
        np.testing.assert_allclose(qpos[7:], self.native.qpos0[7:], atol=1e-6)
        for name, ref in (("elbow", 0.3), ("wrist", 0.05)):
            adr = self.native.joint(name).qposadr[0]
            self.assertAlmostEqual(qpos[adr], ref, places=6)

    def test_round_trip_is_exact(self):
        rng = np.random.default_rng(2)
        qpos = _random_qpos(self.native, rng, 5)
        qvel = rng.normal(size=(5, self.native.nv))
        joint_q, joint_qd = self.solver.convert_joint_coords_from_mujoco(qpos, qvel)
        qpos2, qvel2 = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        np.testing.assert_allclose(qpos2, qpos, atol=1e-12)
        np.testing.assert_allclose(qvel2, qvel, atol=1e-12)


def _build_mixed_model(world_count=1, device="cpu"):
    """Free root with non-identity joint frames, a ball joint with a rotated child anchor, a D6 joint,
    a revolute joint with ``mujoco:dof_ref``, a prismatic joint, joints in non-depth-first order,
    and a loop joint."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMuJoCo.register_custom_attributes(b)

    def inertia(scale):
        return wp.mat33(np.diag([1.0, 1.5, 2.0]) * scale)

    root = b.add_link(mass=2.0, com=wp.vec3(0.1, -0.05, 0.2), inertia=inertia(0.1))
    arm = b.add_link(mass=1.0, com=wp.vec3(0.0, 0.1, 0.0), inertia=inertia(0.02))
    leg = b.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, -0.1), inertia=inertia(0.02))
    hand = b.add_link(mass=0.5, com=wp.vec3(0.05, 0.0, 0.0), inertia=inertia(0.01))
    foot = b.add_link(mass=0.5, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia(0.01))
    cfg = newton.ModelBuilder.JointDofConfig
    joints = [
        b.add_joint_free(
            root,
            parent_xform=wp.transform((0.3, 0.1, 0.5), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7)),
            child_xform=wp.transform((0.05, 0.0, -0.1), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)),
        ),
        b.add_joint_ball(
            root,
            arm,
            parent_xform=wp.transform((0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.9)),
        ),
        # The D6 leg comes before the arm's child, so Newton's joint order is not depth-first.
        b.add_joint_d6(
            root,
            leg,
            linear_axes=[cfg(axis=(1.0, 0.0, 0.0))],
            angular_axes=[cfg(axis=(0.0, 0.0, 1.0)), cfg(axis=(0.0, 1.0, 0.0))],
            parent_xform=wp.transform((0.0, -0.3, 0.0), wp.quat_identity()),
        ),
        b.add_joint_revolute(
            arm,
            hand,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, 0.3, 0.0), wp.quat_identity()),
            custom_attributes={"mujoco:dof_ref": 0.4},
        ),
        b.add_joint_prismatic(leg, foot, axis=(0.0, 0.0, 1.0)),
    ]
    b.add_articulation(joints)
    loop = b.add_joint_revolute(hand, foot, axis=(1.0, 0.0, 0.0))
    b.joint_articulation[loop] = -1
    b.joint_q[b.joint_q_start[loop]] = 0.25
    if world_count == 1:
        return b.finalize(device=device)
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(b, world_count)
    return scene.finalize(device=device)


def _make_solver(model, **kwargs):
    with warnings.catch_warnings():
        # The non-depth-first joint order and the loop joint are intentional.
        warnings.simplefilter("ignore", UserWarning)
        return SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True, **kwargs)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestJointCoordsBuilderModel(_BridgeAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _build_mixed_model()
        cls.solver = _make_solver(cls.model)
        cls.loop_joint = cls.model.joint_count - 1
        body_map = cls.solver.mjc_body_to_newton.numpy()[0]
        cls.bodies = {mj: int(n) for mj, n in enumerate(body_map) if n >= 0}

    def _random_state(self, seed, batch=None):
        rng = np.random.default_rng(seed)
        joint_q = _random_joint_q(self.model, rng, 1 if batch is None else batch)
        joint_qd = rng.normal(size=(len(joint_q), self.model.joint_dof_count))
        if batch is None:
            return joint_q[0], joint_qd[0]
        return joint_q, joint_qd

    def test_joint_order_differs_from_mujoco(self):
        # Precondition for test_kinematics_match_mujoco: the bridge must reorder coordinates.
        mj_q_start = self.solver.mj_q_start.numpy()[: self.loop_joint]
        self.assertFalse(np.all(np.diff(mj_q_start) >= 0))

    def test_kinematics_match_mujoco(self):
        joint_q, joint_qd = self._random_state(3)
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        self.assert_kinematics_match(self.model, self.solver.mj_model, self.bodies, joint_q, joint_qd, qpos, qvel)

    def test_matches_solver_kernels(self):
        joint_q, joint_qd = self._random_state(4)
        state = self.model.state()
        state.joint_q.assign(joint_q.astype(np.float32))
        state.joint_qd.assign(joint_qd.astype(np.float32))
        data = mujoco.MjData(self.solver.mj_model)
        self.solver._update_mjc_data(data, self.model, state)
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(state.joint_q, state.joint_qd)
        np.testing.assert_allclose(qpos, data.qpos, atol=1e-5)
        np.testing.assert_allclose(qvel, data.qvel, atol=1e-5)

        out = self.model.state()
        self.solver._update_newton_state(self.model, out, data, state_prev=state)
        joint_q2, joint_qd2 = self.solver.convert_joint_coords_from_mujoco(data.qpos, data.qvel)
        np.testing.assert_allclose(joint_q2, out.joint_q.numpy(), atol=1e-5)
        np.testing.assert_allclose(joint_qd2, out.joint_qd.numpy(), atol=1e-5)

    def test_round_trip_is_exact(self):
        joint_q, joint_qd = self._random_state(5, batch=6)
        joint_q = joint_q.reshape(2, 3, -1)
        joint_qd = joint_qd.reshape(2, 3, -1)
        joint_q[..., self.model.joint_q_start.numpy()[self.loop_joint]] = 0.25  # loop joint: from the model
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        self.assertEqual(qpos.shape, (2, 3, self.solver.mj_model.nq))
        self.assertEqual(qvel.shape, (2, 3, self.solver.mj_model.nv))
        joint_q2, joint_qd2 = self.solver.convert_joint_coords_from_mujoco(qpos, qvel)
        np.testing.assert_allclose(joint_q2, joint_q, atol=1e-12)
        loop_dof = self.model.joint_qd_start.numpy()[self.loop_joint]
        joint_qd[..., loop_dof] = 0.0
        np.testing.assert_allclose(joint_qd2, joint_qd, atol=1e-12)

    def test_loop_joint_coordinates_come_from_model(self):
        qpos = _random_qpos(self.solver.mj_model, np.random.default_rng(6), 2)
        qvel = np.ones((2, self.solver.mj_model.nv))
        joint_q, joint_qd = self.solver.convert_joint_coords_from_mujoco(qpos, qvel)
        q_index = self.model.joint_q_start.numpy()[self.loop_joint]
        qd_index = self.model.joint_qd_start.numpy()[self.loop_joint]
        np.testing.assert_allclose(joint_q[:, q_index], 0.25, atol=1e-7)
        np.testing.assert_allclose(joint_qd[:, qd_index], 0.0)

    def test_inputs_and_shapes(self):
        joint_q, joint_qd = self._random_state(7)
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q)
        self.assertIsNone(qvel)
        self.assertEqual(qpos.dtype, np.float64)

        # Warp arrays and lists are accepted.
        state = self.model.state()
        state.joint_q.assign(joint_q.astype(np.float32))
        qpos_wp, _ = self.solver.convert_joint_coords_to_mujoco(state.joint_q)
        qpos_list, _ = self.solver.convert_joint_coords_to_mujoco(joint_q.tolist())
        np.testing.assert_allclose(qpos_wp, qpos, atol=1e-6)
        np.testing.assert_array_equal(qpos_list, qpos)

        # Batch dimensions broadcast: one configuration with several velocities. The bridge is
        # linear in the velocities, so unit velocities give the map qvel = T @ joint_qd.
        dofs = self.model.joint_dof_count
        _, columns = self.solver.convert_joint_coords_to_mujoco(joint_q, np.eye(dofs))
        T = columns.T
        _, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        np.testing.assert_allclose(T @ joint_qd, qvel, atol=1e-12)

        with self.assertRaises(ValueError):
            self.solver.convert_joint_coords_to_mujoco(joint_q[:-1])
        with self.assertRaises(ValueError):
            self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd[:-1])
        with self.assertRaises(ValueError):
            self.solver.convert_joint_coords_to_mujoco(np.zeros((2, len(joint_q))), np.zeros((3, dofs)))
        with self.assertRaises(ValueError):
            self.solver.convert_joint_coords_from_mujoco(qpos[:-1])

    def test_does_not_modify_model_or_solver_data(self):
        model_q = self.model.joint_q.numpy().copy()
        data_qpos = self.solver.mj_data.qpos.copy()
        data_qvel = self.solver.mj_data.qvel.copy()
        warp_qpos = self.solver.mjw_data.qpos.numpy().copy()
        joint_q, joint_qd = self._random_state(8)
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        self.solver.convert_joint_coords_from_mujoco(qpos, qvel)
        np.testing.assert_array_equal(self.model.joint_q.numpy(), model_q)
        np.testing.assert_array_equal(self.solver.mj_data.qpos, data_qpos)
        np.testing.assert_array_equal(self.solver.mj_data.qvel, data_qvel)
        np.testing.assert_array_equal(self.solver.mjw_data.qpos.numpy(), warp_qpos)

    def test_reads_runtime_model_edits(self):
        joint_q, joint_qd = self._random_state(9)
        model = _build_mixed_model()
        solver = _make_solver(model)
        solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        com = model.body_com.numpy()
        com[0] = (-0.2, 0.1, 0.05)
        model.body_com.assign(com)
        model.mujoco.dof_ref.assign(model.mujoco.dof_ref.numpy() + 0.1)
        state = model.state()
        state.joint_q.assign(joint_q.astype(np.float32))
        state.joint_qd.assign(joint_qd.astype(np.float32))
        data = mujoco.MjData(solver.mj_model)
        solver._update_mjc_data(data, model, state)
        qpos, qvel = solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        np.testing.assert_allclose(qpos, data.qpos, atol=1e-5)
        np.testing.assert_allclose(qvel, data.qvel, atol=1e-5)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestJointCoordsMultiWorld(unittest.TestCase):
    """One MuJoCo world per Newton world, in the layout of ``mjw_data.qpos``."""

    @classmethod
    def setUpClass(cls):
        cls.world_count = 3
        cls.model = _build_mixed_model(cls.world_count)
        # World-specific COM offsets and root joint frames.
        bodies_per_world = cls.model.body_count // cls.world_count
        joints_per_world = cls.model.joint_count // cls.world_count
        com = cls.model.body_com.numpy()
        X_p = cls.model.joint_X_p.numpy()
        for world in range(cls.world_count):
            com[world * bodies_per_world] += (0.03 * world, -0.02 * world, 0.01)
            X_p[world * joints_per_world, :3] += (0.0, 0.5 * world, 0.0)
        cls.model.body_com.assign(com)
        cls.model.joint_X_p.assign(X_p)
        cls.solver = _make_solver(cls.model, separate_worlds=True)

    def test_layout_matches_warp_data(self):
        rng = np.random.default_rng(10)
        joint_q = _random_joint_q(self.model, rng, 1)[0]
        joint_qd = rng.normal(size=self.model.joint_dof_count)
        nq, nv = self.solver.mj_model.nq, self.solver.mj_model.nv
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        self.assertEqual(qpos.shape, (self.world_count * nq,))
        self.assertEqual(qvel.shape, (self.world_count * nv,))

        state = self.model.state()
        state.joint_q.assign(joint_q.astype(np.float32))
        state.joint_qd.assign(joint_qd.astype(np.float32))
        self.solver._update_mjc_data(self.solver.mjw_data, self.model, state)
        warp_qpos = self.solver.mjw_data.qpos.numpy()
        warp_qvel = self.solver.mjw_data.qvel.numpy()
        np.testing.assert_allclose(qpos.reshape(self.world_count, nq), warp_qpos, atol=1e-5)
        np.testing.assert_allclose(qvel.reshape(self.world_count, nv), warp_qvel, atol=1e-5)

        # [world_count, nq] input as in mjw_data, also as Warp arrays and with batch dimensions.
        out = self.model.state()
        self.solver._update_newton_state(self.model, out, self.solver.mjw_data, state_prev=state)
        joint_q2, joint_qd2 = self.solver.convert_joint_coords_from_mujoco(
            self.solver.mjw_data.qpos, self.solver.mjw_data.qvel
        )
        np.testing.assert_allclose(joint_q2, out.joint_q.numpy(), atol=1e-5)
        np.testing.assert_allclose(joint_qd2, out.joint_qd.numpy(), atol=1e-5)
        batch_q, batch_qd = self.solver.convert_joint_coords_from_mujoco(
            np.stack([warp_qpos, warp_qpos]), np.stack([warp_qvel, warp_qvel])
        )
        self.assertEqual(batch_q.shape, (2, self.model.joint_coord_count))
        np.testing.assert_array_equal(batch_q[1], joint_q2)
        np.testing.assert_array_equal(batch_qd[0], joint_qd2)

    def test_worlds_differ(self):
        # Identical joint coordinates in every world give different MuJoCo states where the worlds differ.
        coords = self.model.joint_coord_count // self.world_count
        dofs = self.model.joint_dof_count // self.world_count
        joint_q = np.tile(_random_joint_q(self.model, np.random.default_rng(11), 1)[0][:coords], self.world_count)
        joint_qd = np.tile(np.random.default_rng(12).normal(size=dofs), self.world_count)
        qpos, qvel = self.solver.convert_joint_coords_to_mujoco(joint_q, joint_qd)
        qpos = qpos.reshape(self.world_count, -1)
        qvel = qvel.reshape(self.world_count, -1)
        self.assertGreater(np.abs(qpos[1, :3] - qpos[0, :3]).max(), 0.1)
        self.assertGreater(np.abs(qvel[1, :3] - qvel[0, :3]).max(), 1e-3)
        np.testing.assert_allclose(qpos[1, 7:], qpos[0, 7:], atol=1e-12)


def test_matches_mujoco_warp_data(test, device):
    """The bridge agrees with the MuJoCo Warp backend's per-step conversion on ``device``."""
    model = _build_mixed_model(world_count=2, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = SolverMuJoCo(model, disable_contacts=True)
    rng = np.random.default_rng(14)
    joint_q = _random_joint_q(model, rng, 1)[0]
    joint_qd = rng.normal(size=model.joint_dof_count)
    state = model.state()
    state.joint_q.assign(joint_q.astype(np.float32))
    state.joint_qd.assign(joint_qd.astype(np.float32))
    solver._update_mjc_data(solver.mjw_data, model, state)
    qpos, qvel = solver.convert_joint_coords_to_mujoco(state.joint_q, state.joint_qd)
    np.testing.assert_allclose(qpos.reshape(solver.mjw_data.qpos.shape), solver.mjw_data.qpos.numpy(), atol=1e-5)
    np.testing.assert_allclose(qvel.reshape(solver.mjw_data.qvel.shape), solver.mjw_data.qvel.numpy(), atol=1e-5)

    out = model.state()
    solver._update_newton_state(model, out, solver.mjw_data, state_prev=state)
    joint_q2, joint_qd2 = solver.convert_joint_coords_from_mujoco(solver.mjw_data.qpos, solver.mjw_data.qvel)
    np.testing.assert_allclose(joint_q2, out.joint_q.numpy(), atol=1e-5)
    np.testing.assert_allclose(joint_qd2, out.joint_qd.numpy(), atol=1e-5)


class TestJointCoordsDevices(unittest.TestCase):
    pass


if mujoco is not None:
    add_function_test(
        TestJointCoordsDevices,
        "test_matches_mujoco_warp_data",
        test_matches_mujoco_warp_data,
        devices=get_test_devices(),
    )


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestCpuMuJoCoModel(unittest.TestCase):
    """The CPU ``mj_model`` with bridged states reproduces :meth:`SolverMuJoCo.step`."""

    # Position servos on the scalar DOFs (elbow, wrist, hip_y, hip_x); the ball shoulder (DOFs 6-8) is passive.
    SERVO_DOFS = slice(9, 13)

    def _model(self):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(MJCF)
        builder.joint_target_mode[:] = [int(newton.JointTargetMode.NONE)] * builder.joint_dof_count
        builder.joint_target_ke[self.SERVO_DOFS] = [40.0, 30.0, 60.0, 50.0]
        builder.joint_target_kd[self.SERVO_DOFS] = [2.0, 1.0, 3.0, 2.5]
        builder.joint_target_mode[self.SERVO_DOFS] = [int(newton.JointTargetMode.POSITION)] * 4
        return builder.finalize(device="cpu")

    def test_mj_step_reproduces_solver_step(self):
        model = self._model()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        rng = np.random.default_rng(13)
        joint_q = _random_joint_q(model, rng, 1)[0]
        joint_qd = rng.normal(scale=0.3, size=model.joint_dof_count)
        state_in, state_out = model.state(), model.state()
        state_in.joint_q.assign(joint_q.astype(np.float32))
        state_in.joint_qd.assign(joint_qd.astype(np.float32))
        control = model.control()
        # Scalar position targets share the joint_q layout (joint_target_q_start == joint_q_start).
        target = joint_q.copy()
        target[11:15] += rng.normal(scale=0.3, size=4)
        control.joint_target_q.assign(target.astype(np.float32))
        solver.step(state_in, state_out, control, None, 0.002)

        data = mujoco.MjData(solver.mj_model)
        data.qpos[:], data.qvel[:] = solver.convert_joint_coords_to_mujoco(state_in.joint_q, state_in.joint_qd)
        # A position actuator's ctrl is its joint's target in MuJoCo's qpos convention (target + ref).
        target_qpos, _ = solver.convert_joint_coords_to_mujoco(control.joint_target_q)
        mj_model = solver.mj_model
        dof_to_qpos = {
            int(dof): int(mj_model.jnt_qposadr[mj_model.dof_jntid[mj_dof]])
            for mj_dof, dof in enumerate(solver.mjc_dof_to_newton_dof.numpy()[0])
        }
        data.ctrl[:] = [target_qpos[dof_to_qpos[int(dof)]] for dof in solver.mjc_actuator_to_newton_idx.numpy()]
        mujoco.mj_step(mj_model, data)
        joint_q2, joint_qd2 = solver.convert_joint_coords_from_mujoco(data.qpos, data.qvel)
        np.testing.assert_allclose(joint_q2, state_out.joint_q.numpy(), atol=1e-5)
        np.testing.assert_allclose(joint_qd2, state_out.joint_qd.numpy(), atol=1e-4)

    def test_actuator_gains_follow_model(self):
        model = self._model()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        mj_model = solver.mj_model
        # Position actuators map to their Newton DOF index (velocity actuators are encoded as -(dof + 2)).
        dofs = solver.mjc_actuator_to_newton_idx.numpy()
        np.testing.assert_array_equal(dofs, np.arange(9, 13))

        def check(ke, kd):
            np.testing.assert_allclose(mj_model.actuator_gainprm[:, 0], ke[dofs], rtol=1e-6)
            np.testing.assert_allclose(mj_model.actuator_biasprm[:, 1], -ke[dofs], rtol=1e-6)
            np.testing.assert_allclose(mj_model.actuator_biasprm[:, 2], -kd[dofs], rtol=1e-6)

        check(model.joint_target_ke.numpy(), model.joint_target_kd.numpy())
        ke = model.joint_target_ke.numpy() * 3.0
        kd = model.joint_target_kd.numpy() + 1.0
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        check(ke, kd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
