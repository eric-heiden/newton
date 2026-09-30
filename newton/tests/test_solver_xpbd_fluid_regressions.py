# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for fluid particle ownership and numerical invariants."""

import unittest

import numpy as np

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


def test_fluid_single_particle_render_position(test, device):
    """Render a lone fluid particle at its actual position."""
    model = _build_particles([(1.0, 2.0, 3.0)]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05)
    solver.update_render_particles(model.state())
    np.testing.assert_allclose(solver.render_positions.numpy(), [[1.0, 2.0, 3.0]], atol=1.0e-6)


def test_fluid_render_fast_path_hides_nonfluid_particles(test, device):
    """Preserve fluid visibility filtering when render smoothing is disabled."""
    builder = _build_particles([(0.0, 0.0, 0.0)])
    builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.025)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05)
    solver.update_render_particles(model.state(), smoothing=0.0, anisotropy_scale=0.0)
    test.assertIsNotNone(solver.render_anisotropy)
    test.assertEqual(float(solver.render_anisotropy.numpy()[1, 3]), 0.0)


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


def test_fluid_disable_clears_render_outputs(test, device):
    """Discard stale fluid visuals after all fluid particle flags are removed."""
    model = _build_particles([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05, max_diffuse_particles=1)
    state = model.state()
    solver.update_render_particles(state)
    solver.diffuse_positions.assign([(0.0, 0.0, 0.0, 1.0)])
    model.particle_flags.fill_(int(newton.ParticleFlags.ACTIVE))
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    solver.update_render_particles(state)
    test.assertIsNone(solver.render_positions)
    test.assertEqual(float(solver.diffuse_positions.numpy()[0, 3]), 0.0)


def test_fluid_rejects_unsupported_gradients(test, device):
    """Reject gradient tracking instead of returning incomplete fluid derivatives."""
    model = _build_particles([(0.0, 0.0, 0.0), (0.04, 0.0, 0.0)]).finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=0.05)
    with test.assertRaisesRegex(NotImplementedError, "fluid.*gradient"):
        solver.step(model.state(requires_grad=True), model.state(requires_grad=True), None, None, 0.01)


def test_fluid_rejects_nonfinite_material_parameters(test, device):
    """Reject non-finite fluid settings before they contaminate the solve."""
    model = _build_particles([(0.0, 0.0, 0.0), (0.04, 0.0, 0.0)]).finalize(device=device)
    for name in ("fluid_rest_distance", "fluid_smoothing_length", "fluid_rest_density", "fluid_cohesion"):
        for value in (float("nan"), float("inf")):
            with test.subTest(parameter=name, value=value):
                with test.assertRaisesRegex(ValueError, name):
                    newton.solvers.SolverXPBD(model, **{name: value})


class TestSolverXPBDFluidRegressions(unittest.TestCase):
    pass


for _function in (
    test_fluid_unequal_mass_pressure_conserves_momentum,
    test_fluid_unequal_mass_separation_conserves_momentum,
    test_fluid_reorder_preserves_forces,
    test_fluid_reorder_preserves_spring_topology,
    test_fluid_single_particle_viscosity_preserves_velocity,
    test_fluid_single_particle_render_position,
    test_fluid_render_fast_path_hides_nonfluid_particles,
    test_fluid_viscosity_uses_projected_neighbors,
    test_fluid_disable_clears_render_outputs,
    test_fluid_rejects_unsupported_gradients,
    test_fluid_rejects_nonfinite_material_parameters,
):
    add_function_test(
        TestSolverXPBDFluidRegressions, _function.__name__, _function, devices=get_test_devices(), check_output=False
    )
del _function


if __name__ == "__main__":
    unittest.main(verbosity=2)
