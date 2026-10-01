# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise device-resident task boundaries in the batched fluid cup example."""

import unittest

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.fluid.example_fluid_cup_transfer import Example
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_example(device, *, particles=128, substeps=2, iterations=1, auto_reset=False, max_steps=1000):
    args = Example.create_parser().parse_args(
        [
            "--world-count",
            "2",
            "--particle-count",
            str(particles),
            "--substeps",
            str(substeps),
            "--iterations",
            str(iterations),
            "--auto-reset" if auto_reset else "--no-auto-reset",
            "--episode-steps",
            str(max_steps),
        ]
    )
    with wp.ScopedDevice(device):
        return Example(newton.viewer.ViewerNull(), args)


def test_task_layout(test, device):
    """Expose stable batched buffers and overlapping physical world coordinates."""
    example = _make_example(device)
    test.assertEqual(example.model.world_count, 2)
    test.assertEqual(example.actions.shape, (2, 4))
    test.assertEqual(example.observations.shape, (2, 20))
    count = example.particles_per_world
    positions = example.state_0.particle_q.numpy()
    np.testing.assert_array_equal(positions[:count], positions[count:])
    np.testing.assert_array_equal(example.model.particle_world.numpy(), np.repeat([0, 1], count))
    np.testing.assert_allclose(example.observations.numpy()[:, 17], 1.0)
    test.assertTrue(np.isfinite(example.observations.numpy()).all())
    with test.assertRaises(ValueError):
        example.reset(wp.zeros(2, dtype=bool, device=device))
    with test.assertRaises(ValueError):
        example.set_actions(wp.zeros((2, 3), dtype=float, device=device))


def test_independent_actions_and_partial_reset(test, device):
    """Preserve another world's state and graph replay when one episode resets."""
    example = _make_example(device)
    actions = wp.array([[0.0, 0.8, 0.0, 1.0], [0.0, -0.4, 0.0, -1.0]], dtype=float, device=device)
    example.set_actions(actions)
    for _ in range(9):
        example.step()
    baseline_particles = example.state_0.particle_q.numpy().copy()
    baseline_bodies = example.state_0.body_q.numpy().copy()
    example.reset()
    example.set_actions(actions)
    for _ in range(6):
        example.step()
    targets = example.targets.numpy()
    test.assertGreater(targets[0, 1], targets[1, 1] + 0.01)
    test.assertLess(example.aperture.numpy()[0], example.aperture.numpy()[1])
    identities = (example.actions.ptr, example.observations.ptr, example.state_0.particle_q.ptr)
    before_particles = example.state_0.particle_q.numpy().copy()
    before_bodies = example.state_0.body_q.numpy().copy()
    before_targets = targets.copy()
    before_steps = example.episode_steps.numpy().copy()
    before_episodes = example.episode_count.numpy().copy()
    mask = wp.array([True, False, False], dtype=bool, device=device)
    example.reset(mask)
    test.assertFalse(example.reference_controller)
    count, bodies = example.particles_per_world, example.bodies_per_world
    np.testing.assert_array_equal(example.state_0.particle_q.numpy()[count:], before_particles[count:])
    np.testing.assert_array_equal(example.state_0.body_q.numpy()[bodies:], before_bodies[bodies:])
    np.testing.assert_array_equal(example.targets.numpy()[1], before_targets[1])
    np.testing.assert_array_equal(example.episode_steps.numpy(), [0, before_steps[1]])
    np.testing.assert_array_equal(example.episode_count.numpy(), before_episodes + np.array([1, 0]))
    for state in (example.state_0, example.state_1):
        np.testing.assert_array_equal(
            state.particle_q.numpy()[:count], example.initial_state.particle_q.numpy()[:count]
        )
    for _ in range(3):
        example.step()
    np.testing.assert_allclose(
        example.state_0.particle_q.numpy()[count:], baseline_particles[count:], atol=5.0e-6, rtol=0.0
    )
    np.testing.assert_allclose(example.state_0.body_q.numpy()[bodies:], baseline_bodies[bodies:], atol=1.0e-6, rtol=0.0)
    test.assertEqual(identities, (example.actions.ptr, example.observations.ptr, example.state_0.particle_q.ptr))
    test.assertIsNotNone(example.graph)


def test_automatic_reset_terminal_observation(test, device):
    """Preserve fresh policy actions across selective automatic episode resets."""
    example = _make_example(device, auto_reset=True, max_steps=3)
    example.set_actions(wp.zeros((2, 4), dtype=float, device=device))
    example.step()
    example.reset(wp.array([False, True, False], dtype=bool, device=device))
    episodes = example.episode_count.numpy().copy()
    for _ in range(2):
        example.step()
    np.testing.assert_array_equal(example.truncated.numpy(), [True, False])
    np.testing.assert_allclose(example.observations.numpy()[:, 19], [1.0, 2.0 / 3.0])
    np.testing.assert_array_equal(example.episode_steps.numpy(), [3, 2])
    previous_targets = example.targets.numpy().copy()
    actions = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, -1.0, 0.0, -1.0]], dtype=np.float32)
    example.set_actions(wp.array(actions, dtype=float, device=device))
    example.step()
    np.testing.assert_array_equal(example.episode_steps.numpy(), [1, 3])
    np.testing.assert_array_equal(example.episode_count.numpy(), episodes + np.array([1, 0]))
    np.testing.assert_array_equal(example.truncated.numpy(), [False, True])
    test.assertFalse(example.reference_controller)
    np.testing.assert_array_equal(example.actions.numpy(), actions)
    expected = previous_targets.copy()
    expected[0] = np.array(example.home)
    expected[:, 1] += actions[:, 1] * example.action_speed * example.frame_dt
    np.testing.assert_allclose(example.targets.numpy(), expected, atol=1.0e-6, rtol=0.0)
    test.assertLess(example.aperture.numpy()[0], 0.04)


def test_reward_prefers_release(test, device):
    """Give the best nonterminal hold no reward so delaying release cannot pay."""
    example = _make_example(device)
    offset = np.array(example.goal) - np.array(example.start)
    example.state_0.particle_q.assign(example.state_0.particle_q.numpy() + offset)
    example.cup_pos.assign(np.tile(np.array(example.goal), (2, 1)))
    example.lifted.fill_(True)
    example.attached.assign([True, False])
    example._observe()
    np.testing.assert_allclose(example.observations.numpy()[:, 17], 1.0)
    np.testing.assert_allclose(example.rewards.numpy(), [0.0, 5.0])
    np.testing.assert_allclose(example.observations.numpy()[:, 18], 1.0)
    np.testing.assert_array_equal(example.terminated.numpy(), [False, True])
    example.state_0.particle_q.assign(example.state_0.particle_q.numpy() + np.array([1.0, 0.0, 0.0]))
    example.episode_steps.fill_(61)
    example._observe()
    test.assertTrue((example.rewards.numpy() < -5.0).all())
    test.assertTrue(example.terminated.numpy().all())
    test.assertFalse(example.successes.numpy().any())


def test_bounded_actions(test, device):
    """Prevent invalid policy outputs from poisoning the IK and fluid state."""
    example = _make_example(device)
    actions = wp.array([[1.0e6, np.nan, np.inf, np.nan], [-1.0e6, -np.inf, np.nan, -1.0]], dtype=float, device=device)
    example.set_actions(actions)
    for _ in range(5):
        example.step()
    test.assertTrue(np.isfinite(example.observations.numpy()).all())
    targets = example.targets.numpy()
    np.testing.assert_allclose(targets[:, 1:], np.tile(np.array(example.home)[1:], (2, 1)), atol=1.0e-6)
    test.assertTrue(
        np.all(np.abs(targets[:, 0] - example.home[0]) <= 5.0 * example.frame_dt * example.action_speed + 1.0e-6)
    )


def test_reference_transfer(test, device):
    """Complete a lift, transport, and release with retained water in both worlds."""
    example = _make_example(device, particles=1000, substeps=6, iterations=3)
    for _ in range(840):
        example.step()
    test.assertTrue(example.lifted.numpy().all())
    test.assertFalse(example.attached.numpy().any())
    test.assertTrue(example.successes.numpy().all())
    np.testing.assert_allclose(example.cup_pos.numpy(), np.tile(np.array(example.goal), (2, 1)), atol=0.015, rtol=0.0)
    test.assertTrue((example.retained.numpy() >= 0.95 * example.particles_per_world).all())
    example.test_final()


class TestFluidCupTransfer(unittest.TestCase):
    pass


add_function_test(TestFluidCupTransfer, "test_task_layout", test_task_layout, devices=["cpu"])
add_function_test(TestFluidCupTransfer, "test_reward_prefers_release", test_reward_prefers_release, devices=["cpu"])
cuda_devices = [device for device in get_test_devices() if device.is_cuda]
add_function_test(
    TestFluidCupTransfer,
    "test_independent_actions_and_partial_reset",
    test_independent_actions_and_partial_reset,
    devices=cuda_devices,
)
add_function_test(
    TestFluidCupTransfer,
    "test_automatic_reset_terminal_observation",
    test_automatic_reset_terminal_observation,
    devices=cuda_devices,
)
add_function_test(TestFluidCupTransfer, "test_bounded_actions", test_bounded_actions, devices=cuda_devices)
add_function_test(TestFluidCupTransfer, "test_reference_transfer", test_reference_transfer, devices=cuda_devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
