# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Physical constraints and quaternion conventions in the optional WBC example."""

import importlib.util
import unittest

import numpy as np
import warp as wp


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
        """Verify quaternion sign continuity and the held endpoint."""
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

    def test_foot_clearance_measurement(self):
        """A known suppressed swing must be distinguished from pose matching."""
        from newton.examples.robot.wbc_controller import measure_foot_tracking  # noqa: PLC0415

        poses = np.tile(self.model.qpos0, (3, 1))
        poses[:, 2] = [0.1, 0.15, 0.1]
        ref = self.reference_type(self.model, poses, fps=10)
        points = [(1, np.array([0.0, 0.0, -0.1]))]
        actual = poses.copy()
        actual[:, 2] = 0.1
        metrics = measure_foot_tracking(self.model, ref, [0, 0.1, 0.2], actual, points)
        self.assertEqual(metrics["swing_recall"], 0.0)
        self.assertAlmostEqual(metrics["swing_height_rmse"], 0.05)
        self.assertAlmostEqual(metrics["foot_height_rmse"], 0.05 / np.sqrt(3))

    @unittest.skipUnless(wp.is_cuda_available(), "Gauss-Newton requires CUDA")
    def test_gauss_newton_graph_solve(self):
        """Check the captured damped solve against independent NumPy algebra."""
        from newton.examples.robot.wbc_mpc_gn import _gram, _solve_kernel, _system  # noqa: PLC0415

        rng = np.random.default_rng(42)
        jac = rng.normal(size=(4, 64)).astype(np.float32)
        residual = rng.normal(size=64).astype(np.float32)
        augmented = np.zeros((16, 64), dtype=np.float32)
        augmented[:4], augmented[4] = jac, residual
        h = jac @ jac.T
        expected = np.linalg.solve(h + 0.1 * np.diag(h.diagonal() + 1), -jac @ residual)
        with wp.ScopedDevice("cuda:0"):
            a = wp.array(augmented)
            gram, matrix, rhs, direction = wp.zeros((16, 16)), wp.zeros((16, 16)), wp.zeros(16), wp.zeros(16)
            solve = _solve_kernel(16)
            with wp.ScopedCapture() as capture:
                wp.launch_tiled(_gram, dim=(1, 1), inputs=[a], outputs=[gram], block_dim=128)
                wp.launch(_system, (16, 16), inputs=[gram, 4, 0.1], outputs=[matrix, rhs])
                wp.launch_tiled(solve, dim=1, inputs=[matrix, rhs], outputs=[direction], block_dim=128)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(direction.numpy()[:4], expected, atol=1e-5, rtol=1e-4)

    @unittest.skipUnless(wp.is_cuda_available(), "Gauss-Newton requires CUDA")
    def test_gauss_newton_rollout_descent(self):
        """A captured shooting update must reduce a physical tracking objective."""
        from newton.examples.robot.wbc_mpc_gn import WholeBodyGaussNewton  # noqa: PLC0415

        poses = np.tile(self.model.qpos0, (31, 1))
        poses[:, -1] = 0.4
        reference = self.reference_type(self.model, poses)
        with wp.ScopedDevice("cuda:0"):
            mpc = WholeBodyGaussNewton(
                self.model,
                np.array([30.0]),
                np.array([3.0]),
                reference,
                horizon=0.1,
                rounds=1,
                hand_weight=0,
                nonfoot_weight=0,
            )
            q = wp.array(self.model.qpos0[None], dtype=float)
            v, clock = wp.zeros((1, self.model.nv)), wp.zeros(1)
            mpc.capture(q, v, clock)
            mpc.solve()
            costs = mpc.costs.numpy()
            self.assertLess(float(mpc.minimum.numpy()[0]), costs[0] - 1e-4)
            # Residuals used for the finite differences equal the actual cost here.
            np.testing.assert_allclose(
                np.sum(mpc.residual.numpy() ** 2, axis=1), mpc.diff_costs.numpy(), rtol=1e-4, atol=1e-5
            )

    @unittest.skipUnless(importlib.util.find_spec("osqp"), "Install the wbc extra for QP tests")
    def test_qp_unactuated_base_and_contact_force_balance(self):
        """Verify that ground forces support the unactuated floating base."""
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

    @unittest.skipUnless(wp.is_cuda_available(), "Sampling MPC requires CUDA")
    def test_mpc_graph_replay_and_native_prediction(self):
        """Compare graph rollouts to native dynamics and verify seeded replay."""
        import mujoco

        q = self.model.qpos0.copy()
        poses = np.tile(q, (31, 1))
        poses[:, -1] = np.linspace(0, 0.6, len(poses))
        ref = self.reference_type(self.model, poses)
        outputs = []
        with wp.ScopedDevice("cuda:0"):
            live_q = wp.array(q[None], dtype=float)
            live_v = wp.zeros((1, self.model.nv))
            clock = wp.zeros(1)
            for _ in range(2):
                mpc = self.mpc_type(
                    self.model,
                    np.array([30.0]),
                    np.array([3.0]),
                    ref,
                    samples=8,
                    horizon=0.03,
                    prediction_dt=0.01,
                    rounds=1,
                    seed=123,
                    foot_weight=0,
                    hand_weight=0,
                )
                mpc.capture(live_q, live_v, clock)
                self.assertIsNotNone(mpc.graph)
                mpc.solve()
                first = mpc.proposals.numpy()
                # The optimizer must choose an evaluated candidate, including the warm plan.
                costs = mpc.costs.numpy()
                best = int(mpc.best.numpy()[0])
                np.testing.assert_array_equal(mpc.plan.numpy(), first[best])
                self.assertEqual(best, int(np.argmin(costs)))
                self.assertLessEqual(costs[best], costs[1])
                # Independent native MuJoCo integration checks the graph's actual dynamics.
                predicted = mpc.data.qpos.numpy()
                for world in range(mpc.samples):
                    data = mujoco.MjData(mpc.cpu_model)
                    data.qpos[:] = q
                    for step in range(mpc.steps):
                        t = step * mpc.dt
                        target, velocity, _ = ref.sample(t)
                        offset = np.interp(t, np.linspace(0, 0.03, 4), first[world, :, 0])
                        data.ctrl[0] = target[-1] + offset + 0.1 * velocity[-1]
                        mujoco.mj_step(mpc.cpu_model, data)
                    np.testing.assert_allclose(predicted[world], data.qpos, atol=2e-5)
                mpc.solve()
                self.assertFalse(np.array_equal(first[2:], mpc.proposals.numpy()[2:]))
                self.assertEqual(int(mpc.iteration.numpy()[0]), 2)
                self.assertEqual(int(mpc.failure_count.numpy()[0]), 0)
                outputs.append(mpc.plan.numpy())
                np.testing.assert_array_equal(live_q.numpy()[0], q.astype(np.float32))
            np.testing.assert_array_equal(*outputs)

    @unittest.skipUnless(wp.is_cuda_available(), "Sampling MPC requires CUDA")
    def test_nonfoot_force_cost_against_native_contacts(self):
        """Verify that GPU landing costs exclude feet and decode pyramidal forces."""
        import mujoco

        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option cone="pyramidal" integrator="implicitfast"/>
        <worldbody><geom type="plane" size="1 1 .1"/>
        <body name="pelvis" pos="0 0 .04"><freejoint/>
          <geom type="sphere" size=".01" pos="0 0 .1" mass="1"/>
          <body name="left_ankle_roll_link" pos="-.2 0 0"><geom size=".05" mass="1"/></body>
          <body name="right_ankle_roll_link" pos=".2 0 0"><geom size=".05" mass="1"/></body>
          <body name="right_wrist_yaw_link" pos="0 .2 0">
            <joint name="hinge" armature=".01" actuatorfrcrange="-20 20"/>
            <geom size=".05" mass="1"/>
          </body>
        </body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>""")
        ref = self.reference_type(model, np.tile(model.qpos0, (31, 1)))
        costs = []
        with wp.ScopedDevice("cuda:0"):
            q = wp.array(model.qpos0[None], dtype=float)
            v, clock = wp.zeros((1, model.nv)), wp.zeros(1)
            for penalty in (0.0, 1000.0):
                mpc = self.mpc_type(
                    model,
                    np.array([30.0]),
                    np.array([3.0]),
                    ref,
                    samples=4,
                    horizon=0.03,
                    prediction_dt=0.01,
                    rounds=1,
                    nonfoot_weight=penalty,
                )
                mpc.capture(q, v, clock)
                mpc.solve()
                costs.append(mpc.costs.numpy())
            proposals = mpc.proposals.numpy()
            expected = []
            for world in range(mpc.samples):
                data = mujoco.MjData(mpc.cpu_model)
                accumulated = 0.0
                for step in range(mpc.steps):
                    data.ctrl[0] = np.interp(step * mpc.dt, np.linspace(0, 0.03, 4), proposals[world, :, 0])
                    mujoco.mj_step(mpc.cpu_model, data)
                    for index, contact in enumerate(data.contact):
                        bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                        if 0 in bodies and 4 in bodies:
                            force = np.zeros(6)
                            mujoco.mj_contactForce(mpc.cpu_model, data, index, force)
                            accumulated += (1 / mpc.steps + float(step == mpc.steps - 1)) * 1000 * (force[0] / 300) ** 2
                expected.append(accumulated)
            self.assertGreater(min(expected), 0.1)
            np.testing.assert_allclose(costs[1] - costs[0], expected, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
