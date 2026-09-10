# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Physical constraints and quaternion conventions in the optional WBC example."""

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") and importlib.util.find_spec("scipy"), "Requires simulation extras"
)
class TestG1WBC(unittest.TestCase):
    def setUp(self):
        # Keep optional example dependencies out of test discovery.
        import mujoco

        from newton.examples.robot.wbc_controller import MotionReference, WholeBodyQP  # noqa: PLC0415
        from newton.examples.robot.wbc_mpc import WholeBodyMPC  # noqa: PLC0415

        self.reference_type, self.qp_type, self.mpc_type = MotionReference, WholeBodyQP, WholeBodyMPC

        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco><option gravity="0 0 -9.81"/>
          <worldbody><body pos="0 0 0.8"><freejoint/>
            <geom type="box" size=".1 .1 .1" mass="10"/>
            <body><joint name="hinge" armature=".01" actuatorfrcrange="-20 20"/>
              <geom type="sphere" size=".05" mass="1"/>
            </body>
          </body></worldbody><actuator><motor joint="hinge"/></actuator>
        </mujoco>""")

    def test_reference_shortest_rotation_and_endpoint(self):
        q = np.tile(self.model.qpos0, (3, 1))
        q[1, 3:7] *= -1  # Same rotation, different quaternion sign.
        q[:, 0] = [0, 1, 2]
        ref = self.reference_type(self.model, q, fps=2)
        pos, vel, _ = ref.sample(0.25)
        self.assertAlmostEqual(pos[0], 0.5)
        self.assertAlmostEqual(np.linalg.norm(pos[3:7]), 1)
        np.testing.assert_allclose(vel[3:6], 0, atol=1e-10)
        np.testing.assert_allclose(ref.sample(2)[1], 0)
        np.testing.assert_allclose(ref.sample(2)[0], q[-1])
        for fps in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.reference_type(self.model, q, fps=fps)

    @unittest.skipUnless(importlib.util.find_spec("osqp"), "Install the wbc extra for QP tests")
    def test_qp_unactuated_base_and_contact_force_balance(self):
        points = [(1, np.array([x, y, -0.8])) for x in (-0.1, 0.1) for y in (-0.1, 0.1)]
        qp = self.qp_type(self.model, points, np.array([20.0]))
        q, v = self.model.qpos0.copy(), np.zeros(self.model.nv)
        tau = qp.solve(q, v, (q, v, v))
        self.assertEqual(qp.failures, 0)
        self.assertLess(qp.residual, 0.005)
        self.assertLessEqual(np.max(np.abs(tau)), 20)
        a = qp.last_solution[: self.model.nv]
        force = qp.last_solution[self.model.nv :].reshape(-1, 3) * qp.force_scale
        self.assertTrue(np.all(force[:, 2] >= -1e-5))
        self.assertAlmostEqual(force[:, 2].sum(), 11 * 9.81, delta=0.1)
        self.assertLess(np.linalg.norm(a[:6]), 0.02)
        # This detects a fictitious external wrench on the floating base.
        _, jac = qp.kinematics(qp.data)
        residual = qp.mass @ a + qp.data.qfrc_bias - jac.reshape(-1, self.model.nv).T @ force.ravel()
        np.testing.assert_allclose(residual[:6], 0, atol=0.01)

    def test_mpc_seed_and_finite_command(self):
        q = self.model.qpos0.copy()
        ref = self.reference_type(self.model, np.tile(q, (3, 1)))
        outputs = []
        for _ in range(2):
            mpc = self.mpc_type(
                self.model,
                np.array([30.0]),
                np.array([3.0]),
                ref,
                samples=4,
                horizon=0.03,
                rounds=1,
                threads=1,
                seed=123,
            )
            self.addCleanup(mpc.runner.close)
            outputs.append(mpc.solve(q, np.zeros(self.model.nv), 0))
            self.assertEqual(mpc.failures, 0)
        np.testing.assert_array_equal(*outputs)
        self.assertTrue(np.isfinite(outputs).all())


if __name__ == "__main__":
    unittest.main()
