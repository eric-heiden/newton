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

    def test_dial_annealing_and_normalized_update(self):
        """Check horizon annealing and softmax refinement against independent algebra."""
        from newton.examples.robot.wbc_mpc_dial import (  # noqa: PLC0415
            _dial_moments,
            _dial_propose,
            _dial_update,
            _dial_weights,
        )

        with wp.ScopedDevice("cpu"):
            center = wp.zeros((4, 2))
            base, annealed = wp.zeros((10, 4, 2)), wp.zeros((10, 4, 2))
            iteration = wp.zeros(1, dtype=int)
            for output, horizon, decay in [(base, 1.0, 1.0), (annealed, 0.5, 0.25)]:
                wp.launch(
                    _dial_propose,
                    output.shape,
                    inputs=[center, iteration, 42, 2, 0.01, horizon, decay],
                    outputs=[output],
                )
            expected = base.numpy() * (0.5 ** np.arange(3, -1, -1))[None, :, None] * 0.25**2
            np.testing.assert_allclose(annealed.numpy(), expected, atol=1e-8)
            np.testing.assert_array_equal(base.numpy()[:, 0], 0)
            costs = wp.array([4.0, 6.0, 8.0, 1e20], dtype=float)
            minimum, moments, weights = wp.array([4.0], dtype=float), wp.zeros(3), wp.zeros(4)
            values = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
            result = wp.zeros((2, 2))
            wp.launch(_dial_moments, 4, inputs=[costs, minimum], outputs=[moments])
            wp.launch(_dial_weights, 4, inputs=[costs, minimum, moments, 0.5], outputs=[weights])
            wp.launch(_dial_update, (2, 2), inputs=[wp.array(values), weights], outputs=[result])
            expected_weights = np.exp(-np.array([0, 2, 4]) / (np.std([4, 6, 8]) * 0.5))
            expected = np.einsum("i,ijk->jk", expected_weights / expected_weights.sum(), values[:3])
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)
            self.assertEqual(weights.numpy()[-1], 0)

    @unittest.skipUnless(wp.is_cuda_available(), "DIAL-MPC requires CUDA")
    def test_dial_graph_and_mean_future(self):
        """Verify the executed DIAL mean has its own native-matching physical future."""
        import mujoco

        from newton.examples.robot.wbc_mpc_dial import WholeBodyDial  # noqa: PLC0415
        from newton.examples.robot.wbc_rollouts import RolloutTraces  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(
            """<mujoco><worldbody><body name="base" pos="0 0 .8"><freejoint/><geom size=".1" mass="10"/><body><joint name="j" armature=".01" actuatorfrcrange="-20 20"/><geom pos=".2 0 0" size=".05" mass="1"/></body></body></worldbody><actuator><motor joint="j"/></actuator></mujoco>"""
        )
        reference = self.reference_type(model, np.tile(model.qpos0, (3, 1)))
        with wp.ScopedDevice("cuda:0"):
            mpc = WholeBodyDial(
                model,
                np.array([30.0]),
                np.array([3.0]),
                reference,
                samples=8,
                horizon=0.02,
                rounds=2,
                initial_rounds=3,
                hand_weight=0,
                nonfoot_weight=0,
            )
            mpc.traces = RolloutTraces(mpc, ("base",), horizon=0.02, stride=1)
            mpc.capture(wp.array(model.qpos0[None], dtype=float), wp.zeros((1, model.nv)), wp.zeros(1))
            for _ in range(2):
                mpc.solve()
                self.assertEqual(mpc.traces.selected.numpy()[0], 8)
                np.testing.assert_allclose(mpc.minimum.numpy(), mpc.mean_costs.numpy())
                data = mujoco.MjData(mpc.cpu_model)
                data.qpos[:] = mpc.mean_data.qpos.numpy()[0]
                mujoco.mj_kinematics(mpc.cpu_model, data)
                np.testing.assert_allclose(mpc.traces.positions.numpy()[8, -1, 0], data.xpos[1], atol=1e-6)
                self.assertTrue(np.isfinite(mpc.plan.numpy()).all())

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

    def test_custom_batches_keep_candidate_alternatives(self):
        """DIAL's executed mean must not hide all sampled alternative futures."""
        from types import SimpleNamespace  # noqa: PLC0415

        import mujoco

        from newton.examples.robot.wbc_rollouts import RolloutTraces  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="base"><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        )
        with wp.ScopedDevice("cpu"):
            mpc = SimpleNamespace(
                cpu_model=model,
                device="cpu",
                steps=1,
                dt=0.01,
                trace_batches=((object(), wp.zeros(3)), (object(), wp.zeros(1))),
            )
            traces = RolloutTraces(mpc, ("base",), horizon=0.01, stride=1)
            paths = np.broadcast_to(np.arange(4, dtype=np.float32)[:, None, None, None], (4, 2, 1, 3)).copy()
            traces.positions.assign(paths)
            traces.costs.assign(np.array([4, 3, 2, 1], dtype=np.float32))
            traces.selected.fill_(3)
            frame = traces.snapshot(3)
            np.testing.assert_array_equal(frame["indices"], [3, 0, 1])
            np.testing.assert_array_equal(frame["positions"], paths[[3, 0, 1]])

    @unittest.skipUnless(wp.is_cuda_available(), "Analytic MPC requires CUDA")
    def test_adjoint_multistep_gradient_and_graph(self):
        """Compare repeated adjoints with physical finite differences, then test descent."""
        import mujoco
        import mujoco_warp as mjw

        from newton.examples.robot.wbc_mpc_adjoint import WholeBodyAdjoint  # noqa: PLC0415
        from newton.examples.robot.wbc_rollouts import RolloutTraces  # noqa: PLC0415

        if not hasattr(mjw, "enable_grad"):
            self.skipTest("Requires the optional MuJoCo Warp PR #1535")
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="base" pos="0 0 .8"><freejoint/>'
            '<geom size=".1" mass="10"/><body name="torso_link" pos=".2 0 0">'
            '<joint name="j" armature=".01" actuatorfrcrange="-20 20"/>'
            '<geom pos=".2 0 0" size=".05" mass="1"/></body></body></worldbody>'
            '<actuator><motor joint="j"/></actuator></mujoco>'
        )
        poses = np.tile(model.qpos0, (31, 1))
        poses[:, -1] = 0.3
        reference = self.reference_type(model, poses)
        with wp.ScopedDevice("cuda:0"):
            mpc = WholeBodyAdjoint(
                model,
                np.array([30.0]),
                np.array([3.0]),
                reference,
                sketch=4,
                starts=2,
                horizon=0.03,
                rounds=1,
                head_position=100,
                head_rotation=300,
                hand_weight=0,
                nonfoot_weight=0,
            )
            q = wp.array(model.qpos0[None], dtype=float)
            v, clock = wp.zeros((1, model.nv)), wp.zeros(1)
            mpc.traces = RolloutTraces(mpc, ("head",), horizon=0.03, stride=1)
            mpc.capture(q, v, clock)
            mpc.record_traces = False
            center = np.full(mpc.plan.shape, 0.05, dtype=np.float32)
            mpc.gradient_proposals.assign(np.broadcast_to(center, mpc.gradient_proposals.shape).copy())
            mpc.differentiate(q, v, clock)
            first = mpc.gradient.numpy().copy()
            mpc.differentiate(q, v, clock)
            np.testing.assert_allclose(mpc.gradient.numpy(), first, atol=1e-5, rtol=1e-4)
            direction = np.array([[0.1], [-0.7], [0.5], [0.2]], dtype=np.float32)
            epsilon = 0.003
            plans = np.broadcast_to(center, mpc.line_proposals.shape).copy()
            plans[1] += epsilon * direction
            plans[2] -= epsilon * direction
            mpc.data, mpc.proposals, mpc.costs = mpc.line_data, mpc.line_proposals, mpc.line_costs
            mpc.samples = mpc.line_proposals.shape[0]
            mpc.line_proposals.assign(plans)
            mpc.rollout(q, v, clock)
            costs = mpc.line_costs.numpy()
            finite_difference = (costs[1] - costs[2]) / (2 * epsilon)
            analytic = np.sum(first[mpc.sketch] * direction)
            np.testing.assert_allclose(analytic, finite_difference, rtol=0.02, atol=1e-3)
            mpc.solve()
            self.assertLess(float(mpc.minimum.numpy()[0]), float(mpc.line_costs.numpy()[0]) - 1e-5)
            selected = int(mpc.traces.selected.numpy()[0])
            self.assertGreaterEqual(selected, mpc.gradient_proposals.shape[0])
            self.assertTrue(np.isfinite(mpc.traces.positions.numpy()[selected]).all())
            # A fully force-saturated actuator cannot respond to a small target change.
            mpc.record_traces = False
            mpc.gradient_proposals.fill_(3.0)
            mpc.differentiate(q, v, clock)
            for state in mpc.states[1:]:
                np.testing.assert_allclose(state.actuator_force.numpy(), 20.0, atol=1e-5)
            np.testing.assert_allclose(mpc.gradient.numpy(), 0.0, atol=1e-7)
            plans.fill(3.0)
            plans[1] += epsilon * direction
            plans[2] -= epsilon * direction
            mpc.data, mpc.proposals, mpc.costs = mpc.line_data, mpc.line_proposals, mpc.line_costs
            mpc.samples = mpc.line_proposals.shape[0]
            mpc.line_proposals.assign(plans)
            mpc.rollout(q, v, clock)
            np.testing.assert_array_equal(mpc.line_costs.numpy()[1:3], mpc.line_costs.numpy()[0])

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

    def test_rotation_residual_near_half_turn(self):
        """Large-angle residuals retain slope and ignore quaternion sign."""
        from newton.examples.robot.wbc_mpc import rotation_error  # noqa: PLC0415

        @wp.kernel(module="unique")
        def evaluate(quats: wp.array[wp.quat], errors: wp.array[wp.vec3]):
            i = wp.tid()
            errors[i] = rotation_error(quats[i], wp.quat_identity())

        angles = np.deg2rad([0, 1, 170, 175])
        quats = np.zeros((4, 4), dtype=np.float32)
        quats[:, 1], quats[:, 3] = np.sin(angles / 2), np.cos(angles / 2)
        with wp.ScopedDevice("cpu"):
            q = wp.array(np.concatenate([quats, -quats]), dtype=wp.quat)
            output = wp.zeros(8, dtype=wp.vec3)
            wp.launch(evaluate, 8, inputs=[q], outputs=[output])
            expected = np.zeros((8, 3))
            expected[:, 1] = np.tile(angles / 2, 2)
            np.testing.assert_allclose(output.numpy(), expected, atol=1e-6)

    def test_rotation_residual_identity_derivative(self):
        """The half-angle rotation residual must retain its slope at exact tracking."""
        from newton.examples.robot.wbc_mpc import rotation_error  # noqa: PLC0415

        @wp.kernel(module="unique")
        def evaluate(angles: wp.array[float], output: wp.array[float]):
            i = wp.tid()
            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angles[i])
            output[i] = rotation_error(rotation, wp.quat_identity())[1]

        with wp.ScopedDevice("cpu"):
            angles = wp.array([0.0, 1e-9, -1e-9, 0.2], dtype=float, requires_grad=True)
            output = wp.zeros(4, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch(evaluate, 4, inputs=[angles], outputs=[output])
            output.grad.fill_(1.0)
            tape.backward()
            np.testing.assert_allclose(output.numpy(), angles.numpy() / 2, atol=1e-11)
            np.testing.assert_allclose(angles.grad.numpy(), 0.5, atol=1e-6)

    def test_sole_task_geometry(self):
        """Pitch and roll must change the lowest corner, not just ankle height."""
        from scipy.spatial.transform import Rotation

        from newton.examples.robot.wbc_mpc import sole_position  # noqa: PLC0415

        @wp.kernel(module="unique")
        def evaluate(quats: wp.array[wp.quat], errors: wp.array[wp.vec3]):
            i = wp.tid()
            errors[i] = sole_position(wp.vec3(0.1, -0.2, 0.3), quats[i])

        rotations = Rotation.from_euler("xyz", [[0, 0, 0], [30, 0, 0], [0, -45, 20]], degrees=True)
        local = np.array([[x, y, -0.035] for x in (-0.05, 0.12) for y in (-0.025, 0.025)])
        expected = []
        for rotation in rotations:
            corners = rotation.apply(local) + np.array([0.1, -0.2, 0.3])
            expected.append([*corners.mean(axis=0)[:2], corners[:, 2].min()])
        with wp.ScopedDevice("cpu"):
            q = wp.array(rotations.as_quat(), dtype=wp.quat)
            output = wp.zeros(3, dtype=wp.vec3)
            wp.launch(evaluate, 3, inputs=[q], outputs=[output])
            np.testing.assert_allclose(output.numpy(), expected, atol=1e-6)

    def test_motion_oscillation_diagnostic(self):
        """A 10 Hz joint oscillation is measured; a slow motion is rejected."""
        import mujoco

        from newton.examples.robot.wbc_controller import measure_motion_tracking  # noqa: PLC0415

        bodies = []
        for j in range(29):
            name = "left_wrist_yaw_link" if j == 28 else f"link{j}"
            bodies.append(f'<body name="{name}"><joint/><geom size=".01" mass=".1"/></body>')
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom size=".1"/>' + "".join(bodies) + "</body></worldbody></mujoco>"
        )
        times = np.arange(0, 5, 0.01)
        poses = np.tile(model.qpos0, (len(times), 1))
        reference = self.reference_type(model, np.tile(model.qpos0, (151, 1)))
        values = []
        for frequency in (2, 10):
            poses[:, -1] = 0.02 * np.sin(2 * np.pi * frequency * times)
            values.append(measure_motion_tracking(model, reference, times, poses))
        expected = 0.02 / np.sqrt(2 * 29)
        self.assertAlmostEqual(values[1]["joint_highpass_rms"], expected, delta=expected * 0.05)
        self.assertLess(values[0]["joint_highpass_rms"], expected * 0.05)
        expected_velocity = 0.02 * np.sin(2 * np.pi * 10 * 0.01) / 0.01 / np.sqrt(2 * 29)
        self.assertAlmostEqual(values[1]["joint_velocity_error_rms"], expected_velocity, delta=0.001)

    def test_hand_pose_and_joint_velocity_cost(self):
        """Known wrist offsets and speeds have an independent quadratic oracle."""
        from newton.examples.robot.wbc_mpc import _score  # noqa: PLC0415

        with wp.ScopedDevice("cpu"):
            q = np.zeros((1, 36), dtype=np.float32)
            q[0, 3] = 1
            v = np.zeros((1, 35), dtype=np.float32)
            v[0, [6, 34]] = [1, -2]
            positions = np.zeros((1, 4, 3), dtype=np.float32)
            positions[0, 2] = [0.2, -0.1, 0.3]
            positions[0, 3] = [-0.1, 0.2, 0.1]
            rotations = np.tile([1, 0, 0, 0], (1, 4, 1)).astype(np.float32)
            rotations[0, 2] = [np.cos(0.3), 0, np.sin(0.3), 0]
            rotationref = np.tile([1, 0, 0, 0], (2, 4)).astype(np.float32)
            width = 96  # 29 joints, 2 feet, 2 hands; no force residuals.
            residual, costs = wp.zeros((1, width)), wp.zeros(1)
            common = [
                wp.array(q),
                wp.array(v),
                wp.array(positions, dtype=wp.vec3),
                wp.array(rotations, dtype=wp.quat),
                wp.array(np.repeat(q, 2, axis=0)),
                wp.zeros((2, 35)),
                wp.zeros((2, 12)),
                wp.array(rotationref),
                wp.array([0, 1, 2, 3], dtype=int),
                wp.zeros(1),
                30.0,
                0.0,
                0.4,
                0.15,
                0.04,
                0.1,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                False,
            ]
            for hand_position, hand_rotation, joint_velocity in [(100.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 0.1)]:
                costs.zero_()
                wp.launch(
                    _score,
                    1,
                    inputs=[
                        *common,
                        hand_position,
                        hand_rotation,
                        joint_velocity,
                        5.0,
                        0,
                        width,
                        True,
                        residual,
                        wp.zeros(1, dtype=int),
                        wp.zeros(1, dtype=int),
                        wp.zeros(1, dtype=int),
                        wp.zeros(5, dtype=int),
                    ],
                    outputs=[costs],
                )
                expected = 0.4 * (hand_position * 0.2 + hand_rotation * 0.3**2 + joint_velocity * 21)
                self.assertAlmostEqual(float(costs.numpy()[0]), expected, places=5)
                self.assertAlmostEqual(float(np.sum(residual.numpy() ** 2)), expected, places=5)

    def test_head_pose_objective_and_measurement(self):
        """Known head translation and tilt agree with an independent rotation oracle."""
        import mujoco
        from scipy.spatial.transform import Rotation

        from newton.examples.robot.wbc_controller import G1_HEAD_OFFSET, measure_head_tracking  # noqa: PLC0415
        from newton.examples.robot.wbc_mpc import _head_score  # noqa: PLC0415

        actual = Rotation.from_euler("y", 0.6)
        target = Rotation.from_euler("y", 0.1)
        origin = np.array([0.1, -0.2, 0.8])
        desired = np.array([0.0, 0.0, 0.8]) + target.apply(G1_HEAD_OFFSET)
        position = origin + actual.apply(G1_HEAD_OFFSET)
        expected = 0.4 * (100 * np.sum((position - desired) ** 2) + 80 * 0.25**2)
        with wp.ScopedDevice("cpu"):
            raw = actual.as_quat()[[3, 0, 1, 2]]
            reference = np.r_[desired, target.as_quat()[[3, 0, 1, 2]]]
            residual, costs = wp.zeros((1, 6)), wp.zeros(1)
            wp.launch(
                _head_score,
                1,
                inputs=[
                    wp.array([[origin]], dtype=wp.vec3),
                    wp.array([[raw]], dtype=wp.quat),
                    wp.array(np.tile(reference, (2, 1)), dtype=float),
                    0,
                    wp.vec3(*G1_HEAD_OFFSET),
                    wp.zeros(1),
                    30.0,
                    0.0,
                    100.0,
                    80.0,
                    0.4,
                    0,
                    True,
                    residual,
                ],
                outputs=[costs],
            )
            self.assertAlmostEqual(float(costs.numpy()[0]), expected, places=5)
            self.assertAlmostEqual(float(np.sum(residual.numpy() ** 2)), expected, places=5)
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="torso_link"><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        )
        qr, qa = model.qpos0.copy(), model.qpos0.copy()
        qr[:3], qr[3:7] = [0, 0, 0.8], reference[3:]
        qa[:3], qa[3:7] = origin, raw
        ref = self.reference_type(model, np.tile(qr, (2, 1)))
        metrics = measure_head_tracking(model, ref, [0, 0.01], [qa, qa])
        self.assertAlmostEqual(metrics["head_position_rmse"], np.linalg.norm(position - desired), places=6)
        self.assertAlmostEqual(metrics["head_rotation_rms_deg"], np.rad2deg(0.5), places=6)
        self.assertAlmostEqual(metrics["head_downward_bias_deg"], np.rad2deg(0.5), places=6)

    @unittest.skipUnless(wp.is_cuda_available(), "Gauss-Newton requires CUDA")
    def test_head_graph_residuals_and_futures(self):
        """The head task enters the full graph cost and records a rotated local point."""
        import mujoco

        from newton.examples.robot.wbc_controller import G1_HEAD_OFFSET  # noqa: PLC0415
        from newton.examples.robot.wbc_mpc_gn import WholeBodyGaussNewton  # noqa: PLC0415
        from newton.examples.robot.wbc_rollouts import RolloutTraces  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(
            """<mujoco><worldbody><body name="torso_link" pos="0 0 .8"><freejoint/><geom size=".1" mass="10"/><body><joint name="j" armature=".01" actuatorfrcrange="-20 20"/><geom pos=".2 0 0" size=".05" mass="1"/></body></body></worldbody><actuator><motor joint="j"/></actuator></mujoco>"""
        )
        model.qpos0[3:7] = [np.cos(0.3), 0, np.sin(0.3), 0]
        ref = self.reference_type(model, np.tile(model.qpos0, (2, 1)))
        with wp.ScopedDevice("cuda:0"):
            mpc = WholeBodyGaussNewton(
                model,
                np.array([30.0]),
                np.array([3.0]),
                ref,
                horizon=0.03,
                rounds=1,
                head_position=100,
                head_rotation=300,
                hand_weight=0,
                nonfoot_weight=0,
            )
            mpc.traces = RolloutTraces(mpc, ("head",), horizon=0.03, stride=1)
            mpc.capture(wp.array(model.qpos0[None], dtype=float), wp.zeros((1, model.nv)), wp.zeros(1))
            mpc.solve()
            np.testing.assert_allclose(np.sum(mpc.residual.numpy() ** 2, axis=1), mpc.diff_costs.numpy(), rtol=1e-4)
            data = mujoco.MjData(model)
            for world, q in enumerate(mpc.diff_data.qpos.numpy()):
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                expected = data.xpos[1] + data.xmat[1].reshape(3, 3) @ G1_HEAD_OFFSET
                np.testing.assert_allclose(mpc.traces.positions.numpy()[world, -1, 0], expected, atol=1e-6)

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
    def test_gauss_newton_coordinate_fallback(self):
        """A useful physical probe survives a deliberately stalled line search."""
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
                trust=1e-6,
                coordinate_search=True,
            )
            q = wp.array(self.model.qpos0[None], dtype=float)
            mpc.capture(q, wp.zeros((1, self.model.nv)), wp.zeros(1))
            mpc.solve()
            costs = mpc.diff_costs.numpy()
            index = int(np.argmin(costs))
            self.assertLess(costs[index], float(mpc.line_costs.numpy().min()) - 1e-4)
            self.assertAlmostEqual(float(mpc.minimum.numpy()[0]), float(costs[index]), places=5)
            np.testing.assert_array_equal(mpc.plan.numpy(), mpc.diff_proposals.numpy()[index])

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
                joint_velocity=0.1,
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

    @unittest.skipUnless(wp.is_cuda_available(), "Rollout visualization requires CUDA")
    def test_recorded_candidate_futures(self):
        """Every saved future matches native dynamics, including a selected probe."""
        import mujoco

        from newton.examples.robot.wbc_mpc_gn import WholeBodyGaussNewton  # noqa: PLC0415
        from newton.examples.robot.wbc_rollouts import RolloutTraces  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body name="root" pos="0 0 .8"><freejoint/>
          <geom size=".1" mass="10"/>
          <body name="tip" pos=".2 0 0"><joint name="j" armature=".01" actuatorfrcrange="-20 20"/>
            <geom size=".05" pos=".1 0 0" mass="1"/>
          </body></body></worldbody><actuator><motor joint="j"/></actuator></mujoco>
        """)
        poses = np.tile(model.qpos0, (31, 1))
        poses[:, -1] = 0.4
        reference = self.reference_type(model, poses)
        cases = [(self.mpc_type, {}), (WholeBodyGaussNewton, {"coordinate_search": False})]
        cases.append((WholeBodyGaussNewton, {"coordinate_search": True, "trust": 1e-6}))
        with wp.ScopedDevice("cuda:0"):
            q, v, clock = wp.array(model.qpos0[None], dtype=float), wp.zeros((1, model.nv)), wp.zeros(1)
            for controller, extra in cases:
                baseline = None
                for record in (False, True):
                    mpc = controller(
                        model,
                        np.array([30.0]),
                        np.array([3.0]),
                        reference,
                        horizon=0.1,
                        rounds=2,
                        samples=8,
                        hand_weight=0,
                        nonfoot_weight=0,
                        **extra,
                    )
                    if record:
                        mpc.traces = RolloutTraces(mpc, ("root", "tip"), horizon=0.1, stride=3)
                    mpc.capture(q, v, clock)
                    for _ in range(2):
                        mpc.solve()
                    if not record:
                        baseline = mpc.plan.numpy()
                        continue
                    np.testing.assert_array_equal(mpc.plan.numpy(), baseline)
                    trace = mpc.traces
                    predictions = trace.positions.numpy()
                    proposals = mpc.proposals.numpy()
                    if trace.line_offset:
                        proposals = np.concatenate([mpc.diff_proposals.numpy(), proposals])
                    index = int(trace.selected.numpy()[0])
                    np.testing.assert_array_equal(proposals[index], mpc.plan.numpy())
                    if extra.get("coordinate_search"):
                        self.assertLess(index, trace.line_offset, "Test must exercise a selected coordinate probe")
                    self.assertAlmostEqual(float(trace.costs.numpy()[index]), float(mpc.minimum.numpy()[0]), places=5)
                    for world, proposal in enumerate(proposals):
                        data = mujoco.MjData(mpc.cpu_model)
                        data.qpos[:] = model.qpos0
                        expected = []
                        for step in range(mpc.steps + 1):
                            mujoco.mj_kinematics(mpc.cpu_model, data)
                            if step in trace.step_index:
                                expected.append(data.xpos[[1, 2]].copy())
                            if step < mpc.steps:
                                offset = np.interp(step * mpc.dt, np.arange(4) * mpc.spacing, proposal[:, 0])
                                data.ctrl[0] = 0.4 + offset
                                mujoco.mj_step(mpc.cpu_model, data)
                        np.testing.assert_allclose(predictions[world], expected, atol=2e-5)
                    snapshot = trace.snapshot(4)
                    self.assertEqual(snapshot["indices"][0], index)
                    self.assertEqual(len(set(snapshot["indices"])), 4)
                    np.testing.assert_array_equal(snapshot["positions"], predictions[snapshot["indices"]])
                    # An invalid final search must not highlight a stale plan.
                    mpc.costs.fill_(1e20)
                    if trace.line_offset:
                        mpc.diff_costs.fill_(1e20)
                    trace.finish(mpc, q, clock)
                    self.assertEqual(int(trace.selected.numpy()[0]), -1)
                    self.assertTrue(np.isnan(trace.snapshot()["positions"]).all())

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
            <geom size=".05" mass=".5" pos="-.06 0 0"/>
            <geom size=".05" mass=".5" pos=".06 0 0"/>
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
                    normal_force = 0.0
                    for index, contact in enumerate(data.contact):
                        bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                        if 0 in bodies and 4 in bodies:
                            force = np.zeros(6)
                            mujoco.mj_contactForce(mpc.cpu_model, data, index, force)
                            normal_force += force[0]
                    accumulated += (1 / mpc.steps + float(step == mpc.steps - 1)) * 1000 * (normal_force / 300) ** 2
                expected.append(accumulated)
            self.assertGreater(min(expected), 0.1)
            np.testing.assert_allclose(costs[1] - costs[0], expected, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
