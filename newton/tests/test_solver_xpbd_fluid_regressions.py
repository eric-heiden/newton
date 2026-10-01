# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for fluid particle ownership and numerical invariants."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.tests.unittest_utils import add_function_test, get_test_devices

FLUID_FLAGS = newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID


def _build_particles(positions, masses=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for i, position in enumerate(positions):
        builder.add_particle(
            pos=position,
            vel=(0.0, 0.0, 0.0),
            mass=1.0 if masses is None else masses[i],
            radius=0.025,
            flags=FLUID_FLAGS,
        )
    return builder


def test_fluid_unequal_mass_pressure_conserves_momentum(test, device):
    """Preserve the center of mass when pressure separates unequal masses."""
    model = _build_particles([(0.0, 0.0, 0.0), (0.04, 0.0, 0.0)], [1.0, 2.0]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=1,
        fluid_rest_distance=0.05,
        fluid_rest_density=5000.0,
        fluid_cohesion=0.0,
    )
    state_in, state_out = model.state(), model.state()
    initial = state_in.particle_q.numpy().copy()
    solver.step(state_in, state_out, None, None, 0.01)
    displacement = state_out.particle_q.numpy() - initial
    test.assertGreater(float(np.linalg.norm(displacement)), 1.0e-5)
    np.testing.assert_allclose((displacement * np.array([[1.0], [2.0]])).sum(axis=0), 0.0, atol=1.0e-7)


def test_fluid_unequal_mass_separation_conserves_momentum(test, device):
    """Split short-range separation according to particle inverse masses."""
    model = _build_particles([(0.0, 0.0, 0.0), (0.01, 0.0, 0.0)], [1.0, 2.0]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=1,
        fluid_rest_distance=0.05,
        fluid_rest_density=1.0e10,
        fluid_cohesion=0.0,
    )
    state_in, state_out = model.state(), model.state()
    initial = state_in.particle_q.numpy().copy()
    solver.step(state_in, state_out, None, None, 0.01)
    displacement = state_out.particle_q.numpy() - initial
    test.assertGreater(float(np.linalg.norm(displacement)), 1.0e-5)
    np.testing.assert_allclose((displacement * np.array([[1.0], [2.0]])).sum(axis=0), 0.0, atol=1.0e-7)


def test_fluid_reorder_preserves_forces(test, device):
    """Keep externally applied forces attached to their particles after sorting."""
    model = _build_particles([(0.2, 0.0, 0.0), (0.0, 0.0, 0.0)]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05)
    state = model.state()
    state.particle_f.assign([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    before = np.concatenate([state.particle_q.numpy(), state.particle_f.numpy()], axis=1)
    solver.reorder_particles(state)
    after = np.concatenate([state.particle_q.numpy(), state.particle_f.numpy()], axis=1)
    np.testing.assert_array_equal(after, before[::-1])


def test_fluid_reorder_preserves_spring_topology(test, device):
    """Skip sorting when particle indices are referenced by constraints."""
    builder = _build_particles([(0.2, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])
    builder.add_spring(0, 1, ke=100.0, kd=0.0, control=0.0)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05)
    state = model.state()
    before = state.particle_q.numpy().copy()
    solver.reorder_particles(state)
    np.testing.assert_array_equal(state.particle_q.numpy(), before)


def test_fluid_single_particle_viscosity_preserves_velocity(test, device):
    """Keep an isolated particle's velocity when viscosity has no neighbors."""
    model = _build_particles([(0.0, 0.0, 0.0)]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05, fluid_viscosity=1.0)
    state_in, state_out = model.state(), model.state()
    state_in.particle_qd.assign([(1.0, 2.0, 3.0)])
    solver.step(state_in, state_out, None, None, 0.01)
    np.testing.assert_allclose(state_out.particle_qd.numpy(), [[1.0, 2.0, 3.0]], atol=1.0e-6)


def test_fluid_viscosity_uses_projected_neighbors(test, device):
    """Find fluid neighbors brought together by the position constraints."""
    speeds = []
    for viscosity in (0.0, 1.0):
        builder = _build_particles([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        builder.add_spring(0, 1, ke=1.0e10, kd=0.0, control=0.0)
        builder.spring_rest_length[0] = 0.04
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=1,
            fluid_rest_distance=0.05,
            fluid_cohesion=0.0,
            fluid_viscosity=viscosity,
        )
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, None, None, 0.01)
        speeds.append(float(np.linalg.norm(state_out.particle_qd.numpy())))
    test.assertLess(speeds[1], 0.8 * speeds[0])


def test_fluid_rejects_unsupported_gradients(test, device):
    """Reject gradient tracking instead of returning incomplete fluid derivatives."""
    model = _build_particles([(0.0, 0.0, 0.0), (0.04, 0.0, 0.0)]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05)
    with test.assertRaisesRegex(NotImplementedError, "fluid.*gradient"):
        solver.step(model.state(requires_grad=True), model.state(requires_grad=True), None, None, 0.01)


def test_fluid_rejects_nonfinite_material_parameters(test, device):
    """Reject non-finite fluid settings before they contaminate the solve."""
    model = _build_particles([(0.0, 0.0, 0.0), (0.04, 0.0, 0.0)]).finalize(device=device)
    for name in (
        "fluid_rest_distance",
        "fluid_smoothing_length",
        "fluid_rest_density",
        "fluid_cohesion",
        "fluid_viscosity",
        "fluid_vorticity_confinement",
        "fluid_relaxation",
        "fluid_max_neighbors",
        "body_max_velocity",
        "body_max_angular_velocity",
    ):
        for value in (float("nan"), float("inf")):
            with test.subTest(parameter=name, value=value):
                with test.assertRaisesRegex(ValueError, name):
                    newton.solvers.SolverXPBD(model, **{name: value})


def test_nonfluid_contacts_preserve_jacobi_order(test, device):
    """Keep ordinary particle/body corrections in the original Jacobi projection."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    builder.add_particle(pos=(0.0, 0.0, 0.14), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.02)
    model = builder.finalize(device=device)
    model.soft_contact_mu = 0.0
    solver = newton.solvers.SolverXPBD(model, iterations=1)
    state_in, state_out = model.state(), model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, 0.01)

    # Main's Jacobi pass measures both constraints at the predicted pose:
    # a 5 cm sphere/ground overlap and a 3 cm particle/sphere overlap.
    body_inv_mass = float(model.body_inv_mass.numpy()[body])
    particle_correction = 0.9 * 0.03 / (1.0 + body_inv_mass)
    expected_body_z = 0.05 + 0.8 * 0.05 - body_inv_mass * particle_correction
    test.assertAlmostEqual(float(state_out.body_q.numpy()[body, 2]), expected_body_z, delta=1.0e-6)
    test.assertAlmostEqual(float(state_out.particle_q.numpy()[0, 2]), 0.14 + particle_correction, delta=1.0e-6)


class TestSolverXPBDFluidRegressions(unittest.TestCase):
    pass


for _function in (
    test_fluid_unequal_mass_pressure_conserves_momentum,
    test_fluid_unequal_mass_separation_conserves_momentum,
    test_fluid_reorder_preserves_forces,
    test_fluid_reorder_preserves_spring_topology,
    test_fluid_single_particle_viscosity_preserves_velocity,
    test_fluid_viscosity_uses_projected_neighbors,
    test_fluid_rejects_unsupported_gradients,
    test_fluid_rejects_nonfinite_material_parameters,
    test_nonfluid_contacts_preserve_jacobi_order,
):
    add_function_test(
        TestSolverXPBDFluidRegressions, _function.__name__, _function, devices=get_test_devices(), check_output=False
    )
del _function


if __name__ == "__main__":
    unittest.main(verbosity=2)
