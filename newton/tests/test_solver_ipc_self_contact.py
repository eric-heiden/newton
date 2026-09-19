# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for surface contact, including swept tunneling."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipc import kernels
from newton._src.solvers.ipc.self_contact import contact_features, query_pairs_reference
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _scene(device, *, enabled=True, capacity=0, square=False, height=0.2):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverIPC.register_custom_attributes(builder)
    vertices = [(0.0, 0.0, 0.1), (1.0, 0.0, 0.1), (0.0, 1.0, 0.1)]
    indices = [0, 1, 2]
    if square:
        vertices.append((1.0, 1.0, 0.1))
        indices.extend([1, 3, 2])
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=indices,
        density=1.0,
        tri_aniso_ke=wp.vec3(0.0),
        edge_aniso_ke=wp.vec3(0.0),
    )
    for i in range(len(vertices)):
        builder.particle_mass[i] = 0.0
    builder.add_particle((0.25, 0.25, height), (0.0, 0.0, -10.0), 1.0)
    model = builder.finalize(device=device)
    config = newton.solvers.SolverIPC.Config(
        plane_offset=-10.0,
        enable_self_contact=enabled,
        self_contact_distance=0.05,
        self_contact_stiffness=0.01,
        self_contact_capacity=capacity,
        max_newton_iterations=100,
        max_pcg_iterations=8,
        max_line_search_iterations=32,
        absolute_tolerance=0.001,
        relative_tolerance=1.0e-5,
        velocity_damping=1.0,
    )
    return model, newton.solvers.SolverIPC(model, config=config)


def test_swept_tunneling(test, device):
    """Stop a fast particle at a triangle that the plane-only solver crosses."""
    for enabled in (False, True):
        model, solver = _scene(device, enabled=enabled)
        a, b = model.state(), model.state()
        solver.step(a, b, None, None, 0.02)
        test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.CONVERGED))
        height = b.particle_q.numpy()[-1, 2]
        if enabled:
            test.assertGreater(height, 0.101)
        else:
            test.assertLess(height, 0.1)
        np.testing.assert_array_equal(a.particle_q.numpy()[:-1], b.particle_q.numpy()[:-1])


def test_initial_thickness_and_overflow(test, device):
    """Roll back invalid thickness and contact-buffer overflow exactly."""
    for options, expected in (
        ({"height": 0.1005}, "INVALID_INITIAL_STATE"),
        ({"square": True, "capacity": 1}, "CONTACT_OVERFLOW"),
    ):
        model, solver = _scene(device, **options)
        a, b = model.state(), model.state()
        solver.step(a, b, None, None, 0.02)
        test.assertEqual(solver.Status(int(solver.diagnostics.status.numpy()[0])).name, expected)
        np.testing.assert_array_equal(a.particle_q.numpy(), b.particle_q.numpy())
        np.testing.assert_array_equal(a.particle_qd.numpy(), b.particle_qd.numpy())


def test_capture_surface_contact(test, device):
    """Match eager surface contact with replayed nested CUDA graphs."""
    if not device.is_cuda:
        test.skipTest("CPU APIC nested conditional replay is not supported")
    model, solver = _scene(device)
    a, b = model.state(), model.state()
    solver.step(a, b, None, None, 0.02)
    expected = b.particle_q.numpy().copy()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(a, b, None, None, 0.02)
    for _ in range(3):
        wp.capture_launch(capture.graph)
        test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.CONVERGED))
        np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=1.0e-6)


def test_contact_gradient_and_operator(test, device):
    """Check barrier force by finite differences and operator symmetry."""
    model, solver = _scene(device, height=0.13)
    contact = solver._self_contact
    x = wp.clone(model.particle_q)
    contact.query(x)
    active = wp.ones(1, dtype=int, device=device)
    rhs = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
    contact.assemble(x, active, rhs)
    analytic = rhs.numpy()
    q = x.numpy().copy()
    numeric = np.zeros_like(q)
    energy = wp.zeros(1, dtype=float, device=device)
    invalid = wp.zeros(1, dtype=int, device=device)
    for i in range(len(q)):
        for j in range(3):
            values = []
            for sign in (1, -1):
                shifted = q.copy()
                shifted[i, j] += sign * 1.0e-5
                x.assign(shifted)
                energy.zero_()
                contact.energy(x, energy, invalid)
                values.append(energy.numpy()[0])
            numeric[i, j] = -(values[0] - values[1]) / 2.0e-5
    np.testing.assert_allclose(analytic, numeric, atol=3.0e-4, rtol=0.01)
    rng = np.random.default_rng(42)
    v, w = [wp.array(rng.normal(size=q.shape), dtype=wp.vec3, device=device) for _ in range(2)]
    av, aw = [wp.zeros_like(v) for _ in range(2)]
    contact.multiply(v, av)
    contact.multiply(w, aw)
    test.assertAlmostEqual(float(np.sum(v.numpy() * aw.numpy())), float(np.sum(w.numpy() * av.numpy())), delta=1.0e-4)
    test.assertGreaterEqual(float(np.sum(v.numpy() * av.numpy())), 0.0)


@wp.kernel
def _distance_probe(points: wp.array2d[wp.vec3d], kinds: wp.array[int], result: wp.array[wp.float64]):
    i = wp.tid()
    diff, _weights = contact_features(points[i, 0], points[i, 1], points[i, 2], points[i, 3], kinds[i])
    result[i] = wp.length(diff)


def test_parallel_and_degenerate_features(test, device):
    """Resolve parallel, crossing, and collapsed features without NaNs."""
    points = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0.2, 0.1, 0], [0.8, 0.1, 0]],
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]],
            [[0.2, 0.2, 0.1], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0.2, 0.1, 0], [0, 0, 0], [1, 0, 0], [0.5, 0, 0]],
        ],
        dtype=np.float64,
    )
    result = wp.zeros(5, dtype=wp.float64, device=device)
    wp.launch(
        _distance_probe,
        dim=5,
        inputs=[
            wp.array(points, dtype=wp.vec3d, device=device),
            wp.array([1, 1, 1, 0, 0], dtype=int, device=device),
            result,
        ],
        device=device,
    )
    np.testing.assert_allclose(result.numpy(), [0.1, 0.0, 1.0, 0.1, 0.1], atol=1.0e-12)


def test_swept_query_matches_exhaustive(test, device):
    """Compare swept BVH candidates with exhaustive AABB enumeration."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverIPC.register_custom_attributes(builder)
    rng = np.random.default_rng(9)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        vertices=rng.uniform(-1, 1, (24, 3)),
        indices=np.arange(24),
        density=1.0,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    contact = solver._self_contact
    direction = wp.array(rng.uniform(-0.3, 0.3, (24, 3)), dtype=wp.vec3, device=device)
    for moving in (False, True):
        contact.query(model.particle_q, direction if moving else None)
        actual = contact.swept if moving else contact.current
        reference = contact.current if moving else contact.swept
        reference.count.zero_()
        reference.overflow.zero_()
        c = solver.config
        wp.launch(
            query_pairs_reference,
            dim=24 * model.tri_count + len(contact.edges) ** 2,
            inputs=[
                model.particle_q,
                direction,
                c.initial_step_size if moving else 0.0,
                c.self_contact_thickness + c.self_contact_distance + c.self_contact_guard,
                model.tri_indices,
                contact.edges,
                model.particle_mass,
                contact.lower_tri,
                contact.upper_tri,
                contact.lower_edge,
                contact.upper_edge,
                reference,
            ],
            device=device,
        )

        def pairs(buffer):
            count = int(buffer.count.numpy()[0])
            test.assertEqual(int(buffer.overflow.numpy()[0]), 0)
            return {
                tuple(row) for row in np.column_stack((buffer.kind.numpy()[:count], buffer.indices.numpy()[:count]))
            }

        test.assertEqual(pairs(actual), pairs(reference))


def test_edge_ccd_and_mollifier(test, device):
    """Bound a moving edge crossing and differentiate the mollified barrier."""
    _model, solver = _scene(device)
    contact = solver._self_contact
    contact.swept.indices.assign(np.tile([0, 1, 2, 3], (contact.capacity, 1)))
    contact.swept.kind.fill_(1)
    contact.swept.count.fill_(1)
    active = wp.ones(1, dtype=int, device=device)
    for offset in (0.0, 1000.0):
        q = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0.1], [0, 1, 0.1]], dtype=np.float32) + offset
        x = wp.array(q, dtype=wp.vec3, device=device)
        direction = wp.array([[0, 0, 0], [0, 0, 0], [0, 0, -0.2], [0, 0, -0.2]], dtype=wp.vec3, device=device)
        alpha = wp.ones(1, dtype=float, device=device)
        contact.bound(x, direction, active, alpha)
        step = float(alpha.numpy()[0])
        test.assertGreater(step, 0.0)
        test.assertLess(step, 0.5)
        end = q + step * direction.numpy()
        test.assertGreater(float(end[2, 2] - end[0, 2]), solver.config.self_contact_thickness)
    q = np.array([[-0.1, 0, 0], [0.1, 0, 0], [-0.1, 0.02, 0.001], [0.1, 0.022, 0.001]], dtype=np.float32)
    x.assign(q)
    contact.rest.assign(q)
    contact.current.indices.assign(np.tile([0, 1, 2, 3], (contact.capacity, 1)))
    contact.current.kind.fill_(1)
    contact.current.count.fill_(1)
    rhs = wp.zeros_like(x)
    contact.assemble(x, active, rhs)
    analytic = rhs.numpy().copy()
    numerical = np.zeros_like(q)
    energy, invalid = wp.zeros(1, dtype=float, device=device), wp.zeros(1, dtype=int, device=device)
    for i in range(4):
        for j in range(3):
            values = []
            for sign in (1, -1):
                shifted = q.copy()
                shifted[i, j] += sign * 1.0e-6
                x.assign(shifted)
                energy.zero_()
                contact.energy(x, energy, invalid)
                values.append(float(energy.numpy()[0]))
            numerical[i, j] = -(values[0] - values[1]) / 2.0e-6
    np.testing.assert_allclose(analytic, numerical, atol=0.002, rtol=0.02)


def test_membrane_metric(test, device):
    """Match the membrane metric to the rest-state force Jacobian."""
    model, solver = _scene(device)
    model.style3d.tri_aniso_ke.assign([[500.0, 500.0, 250.0]])
    q = model.particle_q.numpy().copy()
    solver._x_current.assign(q)
    solver._solve_active.fill_(1)
    vector = wp.array(np.random.default_rng(4).normal(size=q.shape), dtype=wp.vec3, device=device)
    result = wp.zeros_like(vector)
    wp.launch(
        kernels.multiply_membrane_metric,
        dim=model.tri_count,
        inputs=[
            solver._x_current,
            model.tri_indices,
            model.tri_poses,
            model.style3d.tri_aniso_ke,
            model.tri_areas,
            vector,
            result,
        ],
        device=device,
    )
    forces = []
    for sign in (1, -1):
        solver._x_current.assign(q + sign * 1.0e-4 * vector.numpy())
        solver._rhs.zero_()
        solver._add_elastic_forces()
        forces.append(solver._rhs.numpy().copy())
    np.testing.assert_allclose(result.numpy(), -(forces[0] - forces[1]) / 2.0e-4, atol=0.3, rtol=0.002)
    test.assertGreater(float(np.sum(vector.numpy() * result.numpy())), 0.0)


class TestSolverIPCSelfContact(unittest.TestCase):
    pass


def test_initial_pierced_surface(test, device):
    """Reject an edge piercing a face even with positive vertex distances."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverIPC.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        vertices=[(0, 0, 0.1), (1, 0, 0.1), (0, 1, 0.1), (2, 0, 0.1), (3, 0, 0.1), (2, 1, 0.1)],
        indices=[0, 1, 2, 3, 4, 5],
        density=1.0,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, config=newton.solvers.SolverIPC.Config(plane_offset=-10.0))
    a, b = model.state(), model.state()
    positions = a.particle_q.numpy().copy()
    positions[3:] = [(0.25, 0.25, -0.1), (0.25, 0.25, 0.3), (0.75, 0.25, 0.3)]
    a.particle_q.assign(positions)
    solver.step(a, b, None, None, 0.01)
    test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.INVALID_INITIAL_STATE))
    np.testing.assert_array_equal(a.particle_q.numpy(), b.particle_q.numpy())


def test_nonfinite_input(test, device):
    """Reject nonfinite velocities rather than declaring infinite residuals converged."""
    model, solver = _scene(device)
    a, b = model.state(), model.state()
    velocity = a.particle_qd.numpy().copy()
    velocity[-1, 2] = np.inf
    a.particle_qd.assign(velocity)
    solver.step(a, b, None, None, 0.02)
    test.assertEqual(int(solver.diagnostics.status.numpy()[0]), int(solver.Status.INVALID_INITIAL_STATE))
    np.testing.assert_array_equal(b.particle_qd.numpy(), velocity)
    with test.assertRaises(ValueError):
        solver.step(a, b, None, None, float("nan"))


for func in (
    test_swept_tunneling,
    test_initial_thickness_and_overflow,
    test_capture_surface_contact,
    test_contact_gradient_and_operator,
    test_parallel_and_degenerate_features,
    test_swept_query_matches_exhaustive,
    test_edge_ccd_and_mollifier,
    test_membrane_metric,
    test_initial_pierced_surface,
    test_nonfinite_input,
):
    add_function_test(TestSolverIPCSelfContact, func.__name__, func, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
