# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton.examples.robot import example_robot_param_estimation as example
from newton.examples.robot.param_estimation_utils import (
    estimate_relative_errors,
    make_excitation_trajectory,
    positive_to_raw,
    raw_to_positive,
)


class TestParamEstimationUtils(unittest.TestCase):
    def test_positive_parameter_round_trip_preserves_values(self):
        values = np.array([0.1, 1.5, 25.0], dtype=np.float32)

        recovered = raw_to_positive(positive_to_raw(values))

        np.testing.assert_allclose(recovered, values, rtol=1e-6, atol=1e-6)

    def test_positive_parameter_encoding_clamps_nonphysical_values(self):
        values = np.array([-1.0, 0.0, 1e-9], dtype=np.float32)

        recovered = raw_to_positive(positive_to_raw(values, minimum=1e-4))

        np.testing.assert_allclose(recovered, np.full(3, 1e-4, dtype=np.float32), rtol=1e-6, atol=1e-6)

    def test_excitation_trajectory_is_deterministic_and_joint_specific(self):
        first = make_excitation_trajectory(steps=12, dof_count=4, dt=0.01, seed=7)
        second = make_excitation_trajectory(steps=12, dof_count=4, dt=0.01, seed=7)

        self.assertEqual(first.shape, (12, 4))
        np.testing.assert_allclose(first, second)
        self.assertGreater(float(np.max(np.abs(first[:, 0] - first[:, 1]))), 1e-3)

    def test_relative_errors_report_fractional_parameter_error(self):
        estimates = {"payload_mass": 1.2, "friction_scale": 0.8, "armature_scale": 1.5}
        truth = {"payload_mass": 1.0, "friction_scale": 1.0, "armature_scale": 2.0}

        errors = estimate_relative_errors(estimates, truth)

        self.assertAlmostEqual(errors["payload_mass"], 0.2)
        self.assertAlmostEqual(errors["friction_scale"], 0.2)
        self.assertAlmostEqual(errors["armature_scale"], 0.25)

    def test_rollout_finite_difference_gradient_produces_descent_step(self):
        device = "cpu"
        steps = 80
        dof_count = 2
        dt = np.float32(1.0 / 120.0)
        q0 = wp.array(np.array([0.0, -0.45], dtype=np.float32), dtype=wp.float32, device=device)
        qd0 = wp.zeros(dof_count, dtype=wp.float32, device=device)
        torque = wp.array(make_excitation_trajectory(steps, dof_count, float(dt), seed=3), dtype=wp.float32, device=device)
        base_inertia = wp.array(np.array([1.0, 0.85], dtype=np.float32), dtype=wp.float32, device=device)
        base_armature = wp.array(np.array([0.12, 0.11], dtype=np.float32), dtype=wp.float32, device=device)
        base_friction = wp.array(np.array([0.18, 0.14], dtype=np.float32), dtype=wp.float32, device=device)
        payload_lever = wp.array(np.array([0.34, 0.29], dtype=np.float32), dtype=wp.float32, device=device)
        truth_raw = wp.array(positive_to_raw(np.array([1.3, 1.4, 0.8], dtype=np.float32)), dtype=wp.float32, device=device)
        observed = wp.zeros((steps + 1, dof_count), dtype=wp.float32, device=device)
        observed_qd = wp.zeros((steps + 1, dof_count), dtype=wp.float32, device=device)
        wp.launch(
            example.simulate_joint_trajectory_kernel,
            dim=dof_count,
            inputs=[
                torque,
                base_inertia,
                base_armature,
                base_friction,
                payload_lever,
                truth_raw,
                q0,
                qd0,
                dt,
                steps,
            ],
            outputs=[observed, observed_qd],
            device=device,
        )

        def loss_for(raw_values: np.ndarray) -> float:
            raw = wp.array(raw_values, dtype=wp.float32, device=device)
            q_pred = wp.zeros_like(observed)
            qd_pred = wp.zeros_like(observed)
            loss = wp.zeros(1, dtype=wp.float32, device=device)
            wp.launch(
                example.rollout_loss_kernel,
                dim=1,
                inputs=[
                    torque,
                    observed,
                    base_inertia,
                    base_armature,
                    base_friction,
                    payload_lever,
                    raw,
                    q0,
                    qd0,
                    dt,
                    steps,
                    dof_count,
                    np.float32(1.0 / (steps * dof_count)),
                ],
                outputs=[q_pred, qd_pred, loss],
                device=device,
            )
            return float(loss.numpy()[0])

        raw_np = positive_to_raw(np.array([0.5, 0.7, 1.6], dtype=np.float32))
        finite_difference = np.zeros(3, dtype=np.float32)
        eps = 1.0e-3
        for i in range(3):
            plus = raw_np.copy()
            minus = raw_np.copy()
            plus[i] += eps
            minus[i] -= eps
            finite_difference[i] = (loss_for(plus) - loss_for(minus)) / (2.0 * eps)

        before = loss_for(raw_np)
        after = loss_for(raw_np - 0.05 * finite_difference)

        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
