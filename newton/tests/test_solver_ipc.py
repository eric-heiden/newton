# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _particle_case(device, *, height=0.1, config=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    newton.solvers.SolverIPC.register_custom_attributes(builder)
    builder.add_particle((0.0, 0.0, height), (0.0, 0.0, 0.0), 1.0)
    model = builder.finalize(device=device)
    if config is None:
        config = newton.solvers.SolverIPC.Config(
            contact_distance=0.2,
            barrier_stiffness=0.1,
            max_newton_iterations=12,
            max_pcg_iterations=2,
            max_line_search_iterations=8,
            absolute_tolerance=1.0e-5,
        )
    return model, newton.solvers.SolverIPC(model, config=config)


def _barrier_first(s):
    if s >= 1.0:
        return 0.0
    return -(2.0 * (s - 1.0) * math.log(s) + (s - 1.0) ** 2 / s)


def test_particle_matches_scalar_reference(test, device):
    """Match an independent scalar barrier equilibrium."""
    model, solver = _particle_case(device)
    state_in, state_out = model.state(), model.state()
    input_position = state_in.particle_q.numpy().copy()
    dt = 0.1
    solver.step(state_in, state_out, None, None, dt)

    inertial_target = 0.1 - 9.81 * dt * dt
    distance = solver.config.contact_distance
    stiffness = solver.config.barrier_stiffness

    def derivative(x):
        s = x * x / (distance * distance)
        return (x - inertial_target) / (dt * dt) + stiffness * _barrier_first(s) * 2.0 * x / (
            distance * distance
        )

    lower, upper = 1.0e-8, distance
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if derivative(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    expected = 0.5 * (lower + upper)

    test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.CONVERGED))
    test.assertAlmostEqual(float(state_out.particle_q.numpy()[0, 2]), expected, delta=2.0e-5)
    test.assertGreater(float(solver.diagnostics.minimum_gap.numpy()[0]), 0.0)
    np.testing.assert_array_equal(state_in.particle_q.numpy(), input_position)


def test_cloth_force_matches_energy_gradient(test, device):
    """Match cloth forces to finite differences of the accepted energy."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverIPC.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        indices=(0, 1, 2),
        density=1.0,
        tri_aniso_ke=wp.vec3(3.0, 4.0, 2.0),
        edge_aniso_ke=wp.vec3(0.0),
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(
        model,
        config=newton.solvers.SolverIPC.Config(plane_offset=-10.0),
    )
    positions = model.particle_q.numpy()
    positions[0] += (-0.03, 0.02, 0.04)
    positions[1] += (0.08, 0.05, -0.02)
    positions[2] += (0.04, -0.06, 0.03)
    solver._x_current.assign(positions)
    solver._solve_active.fill_(1)
    solver._rhs.zero_()
    solver._add_elastic_forces()
    analytic_force = solver._rhs.numpy().reshape(-1)

    def energy(q):
        solver._x_current.assign(q)
        solver._x_predictor.assign(q)
        solver._current_energy.zero_()
        solver._accumulate_energy(solver._x_current, solver._current_energy, 1.0)
        return float(solver._current_energy.numpy()[0])

    epsilon = 2.0e-4
    finite_difference_force = np.zeros(positions.size, dtype=np.float64)
    for coordinate in range(positions.size):
        q_plus = positions.copy().reshape(-1)
        q_minus = positions.copy().reshape(-1)
        q_plus[coordinate] += epsilon
        q_minus[coordinate] -= epsilon
        finite_difference_force[coordinate] = -(energy(q_plus.reshape(-1, 3)) - energy(q_minus.reshape(-1, 3))) / (
            2.0 * epsilon
        )

    np.testing.assert_allclose(analytic_force, finite_difference_force, rtol=1.5e-2, atol=2.0e-3)


def test_invalid_initial_state_rolls_back(test, device):
    """Reject penetration and preserve the committed state exactly."""
    model, solver = _particle_case(device, height=0.0)
    state_in, state_out = model.state(), model.state()
    expected_q = state_in.particle_q.numpy().copy()
    expected_qd = state_in.particle_qd.numpy().copy()

    solver.step(state_in, state_out, None, None, 0.1)

    test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.INVALID_INITIAL_STATE))
    np.testing.assert_array_equal(state_out.particle_q.numpy(), expected_q)
    np.testing.assert_array_equal(state_out.particle_qd.numpy(), expected_qd)


def test_newton_exhaustion_rolls_back(test, device):
    """Rollback a feasible step that exhausts its nonlinear budget."""
    config = newton.solvers.SolverIPC.Config(
        contact_distance=0.2,
        barrier_stiffness=0.1,
        max_newton_iterations=1,
        max_pcg_iterations=1,
        max_line_search_iterations=4,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    model, solver = _particle_case(device, config=config)
    state_in, state_out = model.state(), model.state()
    expected = state_in.particle_q.numpy().copy()

    solver.step(state_in, state_out, None, None, 0.1)

    test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.NEWTON_EXHAUSTED))
    np.testing.assert_array_equal(state_out.particle_q.numpy(), expected)


def test_factorizations_agree(test, device):
    """Make rank-one and dense block factorizations agree."""
    outputs = []
    for preconditioner in ("rank_one", "dense"):
        config = newton.solvers.SolverIPC.Config(
            contact_distance=0.2,
            barrier_stiffness=0.1,
            max_newton_iterations=12,
            max_pcg_iterations=2,
            max_line_search_iterations=8,
            absolute_tolerance=1.0e-5,
            preconditioner=preconditioner,
        )
        model, solver = _particle_case(device, config=config)
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, None, None, 0.1)
        test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.CONVERGED))
        outputs.append(state_out.particle_q.numpy())

    np.testing.assert_allclose(outputs[0], outputs[1], rtol=2.0e-6, atol=2.0e-7)


def test_captured_replay_stays_feasible(test, device):
    """Keep a long two-step graph replay finite and above the plane."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("CPU APIC nested conditional replay is not supported")
    config = newton.solvers.SolverIPC.Config(
        contact_distance=0.2,
        barrier_stiffness=0.1,
        max_newton_iterations=32,
        max_pcg_iterations=2,
        max_line_search_iterations=8,
        absolute_tolerance=1.0e-4,
    )
    model, solver = _particle_case(device, height=0.5, config=config)
    state_a, state_b = model.state(), model.state()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_a, state_b, None, None, 1.0 / 240.0)
        solver.step(state_b, state_a, None, None, 1.0 / 240.0)

    minimum_gap = math.inf
    for _ in range(60):
        wp.capture_launch(capture.graph)
        status = int(solver.diagnostics.status.numpy()[0])
        test.assertEqual(status, int(solver.Status.CONVERGED))
        minimum_gap = min(minimum_gap, float(solver.diagnostics.minimum_gap.numpy()[0]))

    q = state_a.particle_q.numpy()
    qd = state_a.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertTrue(np.isfinite(qd).all())
    test.assertGreater(minimum_gap, 0.0)


def test_eager_and_captured_step_agree(test, device):
    """Match an eager contact solve with a captured CUDA replay."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("CPU APIC nested conditional replay is not supported")

    eager_model, eager_solver = _particle_case(device)
    eager_in, eager_out = eager_model.state(), eager_model.state()
    eager_solver.step(eager_in, eager_out, None, None, 0.1)

    graph_model, graph_solver = _particle_case(device)
    graph_in, graph_out = graph_model.state(), graph_model.state()
    with wp.ScopedCapture(device=device) as capture:
        graph_solver.step(graph_in, graph_out, None, None, 0.1)
    wp.capture_launch(capture.graph)

    test.assertEqual(
        int(graph_solver.diagnostics.status.numpy()[0]),
        int(eager_solver.diagnostics.status.numpy()[0]),
    )
    np.testing.assert_allclose(graph_out.particle_q.numpy(), eager_out.particle_q.numpy(), rtol=2.0e-6, atol=2.0e-7)
    np.testing.assert_allclose(graph_out.particle_qd.numpy(), eager_out.particle_qd.numpy(), rtol=2.0e-6, atol=2.0e-7)


def test_failure_count_is_persistent(test, device):
    """Retain an earlier failure after a later successful step."""
    config = newton.solvers.SolverIPC.Config(
        contact_distance=0.2,
        barrier_stiffness=0.1,
        max_newton_iterations=1,
        max_pcg_iterations=1,
        max_line_search_iterations=4,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    model, solver = _particle_case(device, config=config)
    state_in, state_out = model.state(), model.state()
    solver.step(state_in, state_out, None, None, 0.1)
    test.assertEqual(int(solver.diagnostics.failed_steps.numpy()[0]), 1)

    solver.config.absolute_tolerance = 1.0e6
    solver.step(state_in, state_out, None, None, 0.1)
    test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.CONVERGED))
    test.assertEqual(int(solver.diagnostics.failed_steps.numpy()[0]), 1)


devices = get_test_devices()


class TestSolverIPC(unittest.TestCase):
    pass


add_function_test(TestSolverIPC, "test_particle_matches_scalar_reference", test_particle_matches_scalar_reference, devices=devices)
add_function_test(TestSolverIPC, "test_cloth_force_matches_energy_gradient", test_cloth_force_matches_energy_gradient, devices=devices)
add_function_test(TestSolverIPC, "test_invalid_initial_state_rolls_back", test_invalid_initial_state_rolls_back, devices=devices)
add_function_test(TestSolverIPC, "test_newton_exhaustion_rolls_back", test_newton_exhaustion_rolls_back, devices=devices)
add_function_test(TestSolverIPC, "test_factorizations_agree", test_factorizations_agree, devices=devices)
add_function_test(TestSolverIPC, "test_captured_replay_stays_feasible", test_captured_replay_stays_feasible, devices=devices)
add_function_test(TestSolverIPC, "test_eager_and_captured_step_agree", test_eager_and_captured_step_agree, devices=devices)
add_function_test(TestSolverIPC, "test_failure_count_is_persistent", test_failure_count_is_persistent, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
