# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for JointType.MIMIC inline forward kinematics evaluation."""

import math
import unittest

import numpy as np
import warp as wp

import newton


class TestMimicJointFK(unittest.TestCase):
    """Verify that MIMIC joints evaluate FK inline in a single kernel launch."""

    def _build_leader_follower_chain(self, leader_angle=0.5, coef0=0.0, coef1=1.0):
        """Build a simple chain: world -> body0 (revolute leader) -> body1 (mimic follower).

        Returns (model, state) with the leader joint set to `leader_angle`.
        """
        builder = newton.ModelBuilder()

        b0 = builder.add_link(mass=1.0)
        b1 = builder.add_link(mass=1.0)

        leader_joint = builder.add_joint_revolute(
            parent=-1,
            child=b0,
            axis=(0, 0, 1),
            label="leader",
        )

        follower_joint = builder.add_joint_mimic(
            parent=b0,
            child=b1,
            leader_joint=leader_joint,
            coef0=coef0,
            coef1=coef1,
            parent_xform=((1, 0, 0), (0, 0, 0, 1)),
            label="follower",
        )

        builder.add_articulation([leader_joint, follower_joint])

        model = builder.finalize()
        state = model.state()

        # Set the leader revolute joint's q value using its q_start
        leader_q_start = model.joint_q_start.numpy()[leader_joint]
        jq = state.joint_q.numpy()
        jq[leader_q_start] = leader_angle
        state.joint_q = wp.array(jq, dtype=wp.float32)

        return model, state, leader_joint

    def test_mimic_joint_type_registered(self):
        """MIMIC joint type value exists and has 0 DOFs."""
        self.assertEqual(newton.JointType.MIMIC, 8)
        dof_count, coord_count = newton.JointType.MIMIC.dof_count(0)
        self.assertEqual(dof_count, 0)
        self.assertEqual(coord_count, 0)

    def test_mimic_joint_fk_identity(self):
        """Mimic with coef0=0, coef1=1 copies leader angle exactly."""
        angle = 0.7
        model, state, _leader_joint = self._build_leader_follower_chain(leader_angle=angle, coef0=0.0, coef1=1.0)

        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        body_q = state.body_q.numpy()
        # body0 is child of revolute about Z at `angle`
        b0_quat = body_q[0][3:]
        expected_b0 = np.array([0, 0, math.sin(angle / 2), math.cos(angle / 2)])
        np.testing.assert_allclose(b0_quat, expected_b0, atol=1e-5)

        # body1 is child of mimic that copies leader angle about same axis
        # Total rotation from world = angle (leader) + angle (mimic copy) = 2*angle
        b1_quat = body_q[1][3:]
        expected_angle = 2.0 * angle
        expected_b1 = np.array([0, 0, math.sin(expected_angle / 2), math.cos(expected_angle / 2)])
        np.testing.assert_allclose(b1_quat, expected_b1, atol=1e-5)

    def test_mimic_joint_fk_scaled(self):
        """Mimic with coef1=0.5 applies half the leader angle."""
        angle = 1.0
        model, state, _ = self._build_leader_follower_chain(leader_angle=angle, coef0=0.0, coef1=0.5)

        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        body_q = state.body_q.numpy()
        # body1 should be rotated by angle + 0.5*angle = 1.5*angle from world
        b1_quat = body_q[1][3:]
        expected_angle = angle + 0.5 * angle
        expected_b1 = np.array([0, 0, math.sin(expected_angle / 2), math.cos(expected_angle / 2)])
        np.testing.assert_allclose(b1_quat, expected_b1, atol=1e-5)

    def test_mimic_joint_fk_with_offset(self):
        """Mimic with coef0=0.2 adds offset to the derived angle."""
        angle = 0.4
        offset = 0.2
        model, state, _ = self._build_leader_follower_chain(leader_angle=angle, coef0=offset, coef1=1.0)

        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        body_q = state.body_q.numpy()
        # body1 rotation from world = angle + (offset + 1.0*angle)
        b1_quat = body_q[1][3:]
        mimic_angle = offset + 1.0 * angle
        expected_angle = angle + mimic_angle
        expected_b1 = np.array([0, 0, math.sin(expected_angle / 2), math.cos(expected_angle / 2)])
        np.testing.assert_allclose(b1_quat, expected_b1, atol=1e-5)

    def test_mimic_joint_zero_dof(self):
        """MIMIC joints contribute 0 DOFs and 0 coordinates."""
        builder = newton.ModelBuilder()

        b0 = builder.add_link(mass=1.0)
        b1 = builder.add_link(mass=1.0)

        leader = builder.add_joint_revolute(parent=-1, child=b0, axis=(0, 0, 1))
        follower = builder.add_joint_mimic(parent=b0, child=b1, leader_joint=leader)
        builder.add_articulation([leader, follower])

        model = builder.finalize()
        # The revolute contributes 1 DOF; the mimic contributes 0
        leader_dofs = model.joint_qd_start.numpy()[leader + 1] - model.joint_qd_start.numpy()[leader]
        follower_dofs = model.joint_qd_start.numpy()[follower + 1] - model.joint_qd_start.numpy()[follower]
        self.assertEqual(leader_dofs, 1)
        self.assertEqual(follower_dofs, 0)

    def test_multi_follower_chain(self):
        """Multiple mimic joints in a chain, all driven by a single leader."""
        builder = newton.ModelBuilder()

        bodies = []
        for _i in range(4):
            b = builder.add_link(mass=1.0)
            bodies.append(b)

        leader = builder.add_joint_revolute(parent=-1, child=bodies[0], axis=(0, 0, 1))
        joints = [leader]
        for i in range(1, 4):
            j = builder.add_joint_mimic(
                parent=bodies[i - 1],
                child=bodies[i],
                leader_joint=leader,
                coef0=0.0,
                coef1=1.0,
                parent_xform=((1, 0, 0), (0, 0, 0, 1)),
            )
            joints.append(j)

        builder.add_articulation(joints)
        model = builder.finalize()
        state = model.state()

        angle = 0.3
        leader_q_start = model.joint_q_start.numpy()[leader]
        jq = state.joint_q.numpy()
        jq[leader_q_start] = angle
        state.joint_q = wp.array(jq, dtype=wp.float32)

        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        body_q = state.body_q.numpy()
        # Each body accumulates one more `angle` rotation from the mimic
        for idx in range(4):
            cumulative_angle = (idx + 1) * angle
            expected_quat = np.array([0, 0, math.sin(cumulative_angle / 2), math.cos(cumulative_angle / 2)])
            np.testing.assert_allclose(
                body_q[idx][3:], expected_quat, atol=1e-5, err_msg=f"Body {idx} orientation mismatch"
            )

    def test_mimic_joint_uses_follower_axis(self):
        """A mimic joint derives q from the leader but rotates about its own axis."""
        builder = newton.ModelBuilder()

        b0 = builder.add_link(mass=1.0)
        b1 = builder.add_link(mass=1.0)

        leader = builder.add_joint_revolute(parent=-1, child=b0, axis=(0, 0, 1), label="leader_z")
        follower = builder.add_joint_mimic(
            parent=b0,
            child=b1,
            leader_joint=leader,
            axis=(1, 0, 0),
            mimic_type=newton.JointType.REVOLUTE,
            label="follower_x",
        )
        builder.add_articulation([leader, follower])

        model = builder.finalize()
        state = model.state()

        angle = 0.4
        jq = state.joint_q.numpy()
        jq[model.joint_q_start.numpy()[leader]] = angle
        state.joint_q.assign(jq)

        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        qz = np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])
        qx = np.array([math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)])
        expected = np.array(
            [
                qz[3] * qx[0] + qz[0] * qx[3] + qz[1] * qx[2] - qz[2] * qx[1],
                qz[3] * qx[1] - qz[0] * qx[2] + qz[1] * qx[3] + qz[2] * qx[0],
                qz[3] * qx[2] + qz[0] * qx[1] - qz[1] * qx[0] + qz[2] * qx[3],
                qz[3] * qx[3] - qz[0] * qx[0] - qz[1] * qx[1] - qz[2] * qx[2],
            ]
        )
        np.testing.assert_allclose(state.body_q.numpy()[1][3:], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
