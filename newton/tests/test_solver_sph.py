# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSPH
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestSolverSPH(unittest.TestCase):
    pass


def _build_two_particle_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.08
    builder.add_particle(pos=(-0.03, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.08)
    builder.add_particle(pos=(0.03, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.08)
    return builder.finalize(device=device)


def test_sph_exports_public_solver(test, device):
    model = _build_two_particle_model(device)
    solver = SolverSPH(model, smoothing_length=0.16)
    test.assertIsInstance(solver, newton.solvers.SolverBase)


def test_sph_computes_positive_density(test, device):
    model = _build_two_particle_model(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(model, smoothing_length=0.16, rest_density=10.0, gas_constant=0.0)

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    density = solver.particle_density.numpy()
    test.assertEqual(density.shape[0], 2)
    test.assertTrue(np.all(np.isfinite(density)))
    test.assertTrue(np.all(density > 0.0))


def test_sph_pressure_separates_overlapping_particles(test, device):
    model = _build_two_particle_model(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.18,
        rest_density=8.0,
        gas_constant=10.0,
        viscosity=0.0,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    initial_distance = np.linalg.norm(state_0.particle_q.numpy()[1] - state_0.particle_q.numpy()[0])
    for _ in range(8):
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    final_distance = np.linalg.norm(state_0.particle_q.numpy()[1] - state_0.particle_q.numpy()[0])
    test.assertGreater(final_distance, initial_distance)


def test_sph_respects_world_bounds(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(0.0, 0.0, 0.02), vel=(0.0, 0.0, -1.0), mass=1.0, radius=0.05)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.12,
        gas_constant=0.0,
        bounds_lower=(-1.0, -1.0, 0.0),
        bounds_upper=(1.0, 1.0, 1.0),
        boundary_damping=0.25,
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=0.2)
    q = state_1.particle_q.numpy()[0]
    qd = state_1.particle_qd.numpy()[0]

    test.assertGreaterEqual(q[2], 0.05 - 1.0e-6)
    test.assertGreater(qd[2], 0.0)


def test_sph_cuda_graph_capture(test, device):
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")

    builder = newton.ModelBuilder(gravity=-9.81)
    builder.default_particle_radius = 0.05
    builder.add_particle_grid(
        pos=wp.vec3(-0.2, -0.1, 0.1),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.2, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        dim_z=3,
        cell_x=0.06,
        cell_y=0.06,
        cell_z=0.06,
        mass=0.1,
        jitter=0.0,
        radius_mean=0.05,
        radius_std=0.0,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.11,
        rest_density=200.0,
        gas_constant=20.0,
        viscosity=0.05,
        bounds_lower=(-1.0, -1.0, 0.0),
        bounds_upper=(1.0, 1.0, 1.0),
        max_velocity=5.0,
    )

    # Compile kernels and allocate solver/grid internals before capture.
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 240.0)
    state_0, state_1 = state_1, state_0
    wp.synchronize()

    with wp.ScopedCapture(device=device) as capture:
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 240.0)

    wp.capture_launch(capture.graph)
    wp.synchronize()
    test.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))


devices = get_test_devices(mode="basic")
add_function_test(TestSolverSPH, "test_sph_exports_public_solver", test_sph_exports_public_solver, devices=devices)
add_function_test(TestSolverSPH, "test_sph_computes_positive_density", test_sph_computes_positive_density, devices=devices)
add_function_test(
    TestSolverSPH,
    "test_sph_pressure_separates_overlapping_particles",
    test_sph_pressure_separates_overlapping_particles,
    devices=devices,
)
add_function_test(TestSolverSPH, "test_sph_respects_world_bounds", test_sph_respects_world_bounds, devices=devices)
add_function_test(TestSolverSPH, "test_sph_cuda_graph_capture", test_sph_cuda_graph_capture, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
