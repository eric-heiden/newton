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


def _cube_mesh(h: float) -> newton.Mesh:
    vertices = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0,
            2,
            1,
            0,
            3,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
            0,
            1,
            5,
            0,
            5,
            4,
            3,
            7,
            6,
            3,
            6,
            2,
            0,
            4,
            7,
            0,
            7,
            3,
            1,
            2,
            6,
            1,
            6,
            5,
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices, indices, compute_inertia=False)


def test_sph_exports_public_solver(test, device):
    model = _build_two_particle_model(device)
    solver = SolverSPH(model, smoothing_length=0.16)
    test.assertIsInstance(solver, newton.solvers.SolverBase)
    test.assertIsNone(solver.diffuse_positions)
    test.assertIsNotNone(solver.particle_vorticity)
    test.assertIsNotNone(solver.particle_velocity_smooth)
    test.assertIsNotNone(solver.render_positions)
    test.assertIsNotNone(solver.render_anisotropy)
    test.assertIsNotNone(solver.render_anisotropy_secondary)
    test.assertIsNotNone(solver.render_anisotropy_tertiary)
    test.assertFalse(solver.render_buffers_valid)


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


def test_sph_pbf_projection_separates_without_pressure(test, device):
    model = _build_two_particle_model(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.16,
        rest_density=8.0,
        gas_constant=0.0,
        viscosity=0.0,
        pbf_iterations=4,
        pbf_relaxation=0.8,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    initial_distance = np.linalg.norm(state_0.particle_q.numpy()[1] - state_0.particle_q.numpy()[0])
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)
    projected_distance = np.linalg.norm(state_1.particle_q.numpy()[1] - state_1.particle_q.numpy()[0])
    velocities = state_1.particle_qd.numpy()

    test.assertTrue(solver.pbf_enabled)
    test.assertGreater(projected_distance, initial_distance + 0.02)
    test.assertTrue(np.all(np.isfinite(velocities)))


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


def test_sph_collides_with_particle_shapes(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(-1.0, 0.0, 0.0), mass=0.1, radius=0.02)
    builder.add_shape_sphere(body=-1, radius=0.1)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        boundary_damping=0.25,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    q = state_1.particle_q.numpy()[0]
    qd = state_1.particle_qd.numpy()[0]
    test.assertGreaterEqual(q[0], 0.12 - 1.0e-5)
    test.assertGreater(qd[0], 0.0)


def test_sph_shape_collision_accumulates_dynamic_body_feedback(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    body = builder.add_body(mass=1.0)
    builder.add_shape_sphere(body=body, radius=0.1)
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(-1.0, 0.0, 0.0), mass=0.1, radius=0.02)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_hit = model.state()
    state_disabled = model.state()

    solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        boundary_damping=0.25,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )
    solver.step(state_0, state_hit, control=None, contacts=None, dt=1.0 / 120.0)

    disabled_solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        boundary_damping=0.25,
        shape_collision_body_feedback=False,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )
    disabled_solver.step(state_0, state_disabled, control=None, contacts=None, dt=1.0 / 120.0)

    body_force = state_hit.body_f.numpy()[body]
    disabled_force = state_disabled.body_f.numpy()[body]
    test.assertGreater(state_hit.particle_qd.numpy()[0, 0], 0.0)
    test.assertLess(float(body_force[0]), 0.0)
    test.assertGreater(float(np.linalg.norm(body_force[:3])), 1.0)
    test.assertTrue(np.allclose(disabled_force, 0.0))


def test_sph_shape_adhesion_damps_fluid_and_diffuse_rebound(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(-1.0, 0.0, 0.0), mass=0.1, radius=0.02)
    builder.add_shape_sphere(body=-1, radius=0.1)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_rebound = model.state()
    state_adhesion = model.state()

    common_kwargs = {
        "smoothing_length": 0.08,
        "rest_density": 50.0,
        "gas_constant": 0.0,
        "viscosity": 0.0,
        "boundary_damping": 0.25,
        "max_diffuse_particles": 1,
        "diffuse_threshold": 1.0e9,
        "bounds_lower": (-10.0, -10.0, -10.0),
        "bounds_upper": (10.0, 10.0, 10.0),
    }
    rebound_solver = SolverSPH(model, **common_kwargs)
    adhesion_solver = SolverSPH(model, shape_adhesion=240.0, **common_kwargs)
    for solver in (rebound_solver, adhesion_solver):
        test.assertIsNotNone(solver.diffuse_positions)
        test.assertIsNotNone(solver.diffuse_velocities)
        test.assertIsNotNone(solver.diffuse_worlds)
        solver.diffuse_positions.assign(np.array([[0.04, 0.0, 0.0, 1.0]], dtype=np.float32))
        solver.diffuse_velocities.assign(np.array([[-1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        solver.diffuse_worlds.assign(np.array([-1], dtype=np.int32))

    rebound_solver.step(state_0, state_rebound, control=None, contacts=None, dt=1.0 / 120.0)
    adhesion_solver.step(state_0, state_adhesion, control=None, contacts=None, dt=1.0 / 120.0)

    rebound_velocity = state_rebound.particle_qd.numpy()[0, 0]
    adhesion_velocity = state_adhesion.particle_qd.numpy()[0, 0]
    rebound_diffuse_velocity = rebound_solver.diffuse_velocities.numpy()[0, 0]
    adhesion_diffuse_velocity = adhesion_solver.diffuse_velocities.numpy()[0, 0]

    test.assertGreater(rebound_velocity, 0.0)
    test.assertGreater(rebound_diffuse_velocity, 0.0)
    test.assertLess(abs(float(adhesion_velocity)), abs(float(rebound_velocity)))
    test.assertLess(abs(float(adhesion_diffuse_velocity)), abs(float(rebound_diffuse_velocity)))
    test.assertAlmostEqual(float(adhesion_velocity), 0.0, delta=1.0e-5)
    test.assertAlmostEqual(float(adhesion_diffuse_velocity), 0.0, delta=1.0e-5)


def test_sph_shape_friction_damps_fluid_and_diffuse_slide(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(0.0, 1.0, 0.0), mass=0.1, radius=0.02)
    builder.add_shape_sphere(body=-1, radius=0.1)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_slide = model.state()
    state_friction = model.state()

    common_kwargs = {
        "smoothing_length": 0.08,
        "rest_density": 50.0,
        "gas_constant": 0.0,
        "viscosity": 0.0,
        "boundary_damping": 0.25,
        "max_diffuse_particles": 1,
        "diffuse_threshold": 1.0e9,
        "bounds_lower": (-10.0, -10.0, -10.0),
        "bounds_upper": (10.0, 10.0, 10.0),
    }
    slide_solver = SolverSPH(model, **common_kwargs)
    friction_solver = SolverSPH(model, shape_friction=240.0, **common_kwargs)
    for solver in (slide_solver, friction_solver):
        test.assertIsNotNone(solver.diffuse_positions)
        test.assertIsNotNone(solver.diffuse_velocities)
        test.assertIsNotNone(solver.diffuse_worlds)
        solver.diffuse_positions.assign(np.array([[0.04, 0.0, 0.0, 1.0]], dtype=np.float32))
        solver.diffuse_velocities.assign(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
        solver.diffuse_worlds.assign(np.array([-1], dtype=np.int32))

    slide_solver.step(state_0, state_slide, control=None, contacts=None, dt=1.0 / 120.0)
    friction_solver.step(state_0, state_friction, control=None, contacts=None, dt=1.0 / 120.0)

    slide_q = state_slide.particle_q.numpy()[0]
    friction_q = state_friction.particle_q.numpy()[0]
    slide_v = state_slide.particle_qd.numpy()[0]
    friction_v = state_friction.particle_qd.numpy()[0]
    slide_n = slide_q / np.linalg.norm(slide_q)
    friction_n = friction_q / np.linalg.norm(friction_q)
    slide_tangent_speed = np.linalg.norm(slide_v - slide_n * np.dot(slide_v, slide_n))
    friction_tangent_speed = np.linalg.norm(friction_v - friction_n * np.dot(friction_v, friction_n))

    slide_diffuse_q = slide_solver.diffuse_positions.numpy()[0, :3]
    friction_diffuse_q = friction_solver.diffuse_positions.numpy()[0, :3]
    slide_diffuse_v = slide_solver.diffuse_velocities.numpy()[0, :3]
    friction_diffuse_v = friction_solver.diffuse_velocities.numpy()[0, :3]
    slide_diffuse_n = slide_diffuse_q / np.linalg.norm(slide_diffuse_q)
    friction_diffuse_n = friction_diffuse_q / np.linalg.norm(friction_diffuse_q)
    slide_diffuse_tangent_speed = np.linalg.norm(
        slide_diffuse_v - slide_diffuse_n * np.dot(slide_diffuse_v, slide_diffuse_n)
    )
    friction_diffuse_tangent_speed = np.linalg.norm(
        friction_diffuse_v - friction_diffuse_n * np.dot(friction_diffuse_v, friction_diffuse_n)
    )

    test.assertGreater(float(slide_tangent_speed), 0.5)
    test.assertGreater(float(slide_diffuse_tangent_speed), 0.5)
    test.assertLess(float(friction_tangent_speed), 1.0e-4)
    test.assertLess(float(friction_diffuse_tangent_speed), 1.0e-4)


def test_sph_shape_collision_distance_separates_from_margin(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.02)
    builder.add_shape_sphere(body=-1, radius=0.1)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_margin = model.state()
    state_distance = model.state()
    common_kwargs = {
        "smoothing_length": 0.08,
        "rest_density": 50.0,
        "gas_constant": 0.0,
        "viscosity": 0.0,
        "max_diffuse_particles": 1,
        "diffuse_threshold": 1.0e9,
        "boundary_damping": 0.25,
        "bounds_lower": (-10.0, -10.0, -10.0),
        "bounds_upper": (10.0, 10.0, 10.0),
    }
    margin_solver = SolverSPH(model, shape_collision_margin=0.05, **common_kwargs)
    distance_solver = SolverSPH(
        model,
        shape_collision_distance=0.05,
        shape_collision_margin=0.05,
        **common_kwargs,
    )
    for solver in (margin_solver, distance_solver):
        test.assertIsNotNone(solver.diffuse_positions)
        test.assertIsNotNone(solver.diffuse_velocities)
        test.assertIsNotNone(solver.diffuse_worlds)
        solver.diffuse_positions.assign(np.array([[0.04, 0.0, 0.0, 1.0]], dtype=np.float32))
        solver.diffuse_velocities.assign(np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        solver.diffuse_worlds.assign(np.array([-1], dtype=np.int32))

    margin_solver.step(state_0, state_margin, control=None, contacts=None, dt=1.0 / 120.0)
    distance_solver.step(state_0, state_distance, control=None, contacts=None, dt=1.0 / 120.0)

    margin_position = state_margin.particle_q.numpy()[0]
    distance_position = state_distance.particle_q.numpy()[0]
    margin_diffuse_position = margin_solver.diffuse_positions.numpy()[0, :3]
    distance_diffuse_position = distance_solver.diffuse_positions.numpy()[0, :3]

    test.assertIsNone(margin_solver.shape_collision_distance)
    test.assertAlmostEqual(margin_solver.shape_collision_margin, 0.05)
    test.assertAlmostEqual(distance_solver.shape_collision_distance, 0.05)
    test.assertGreaterEqual(float(margin_position[0]), 0.12 - 1.0e-5)
    test.assertLess(float(margin_position[0]), 0.13)
    test.assertGreaterEqual(float(margin_diffuse_position[0]), 0.12 - 1.0e-5)
    test.assertLess(float(margin_diffuse_position[0]), 0.13)
    test.assertGreaterEqual(float(distance_position[0]), 0.15 - 1.0e-5)
    test.assertGreaterEqual(float(distance_diffuse_position[0]), 0.15 - 1.0e-5)


def test_sph_shape_restitution_overrides_boundary_damping(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(-1.0, 0.0, 0.0), mass=0.1, radius=0.02)
    builder.add_shape_sphere(body=-1, radius=0.1)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_sticky = model.state()
    state_bouncy = model.state()
    common_kwargs = {
        "smoothing_length": 0.08,
        "rest_density": 50.0,
        "gas_constant": 0.0,
        "viscosity": 0.0,
        "boundary_damping": 0.0,
        "max_diffuse_particles": 1,
        "diffuse_threshold": 1.0e9,
        "bounds_lower": (-10.0, -10.0, -10.0),
        "bounds_upper": (10.0, 10.0, 10.0),
    }
    sticky_solver = SolverSPH(model, shape_restitution=0.0, **common_kwargs)
    bouncy_solver = SolverSPH(model, shape_restitution=0.6, **common_kwargs)
    for solver in (sticky_solver, bouncy_solver):
        test.assertIsNotNone(solver.diffuse_positions)
        test.assertIsNotNone(solver.diffuse_velocities)
        test.assertIsNotNone(solver.diffuse_worlds)
        solver.diffuse_positions.assign(np.array([[0.04, 0.0, 0.0, 1.0]], dtype=np.float32))
        solver.diffuse_velocities.assign(np.array([[-1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        solver.diffuse_worlds.assign(np.array([-1], dtype=np.int32))

    sticky_solver.step(state_0, state_sticky, control=None, contacts=None, dt=1.0 / 120.0)
    bouncy_solver.step(state_0, state_bouncy, control=None, contacts=None, dt=1.0 / 120.0)

    sticky_velocity = state_sticky.particle_qd.numpy()[0, 0]
    bouncy_velocity = state_bouncy.particle_qd.numpy()[0, 0]
    sticky_diffuse_velocity = sticky_solver.diffuse_velocities.numpy()[0, 0]
    bouncy_diffuse_velocity = bouncy_solver.diffuse_velocities.numpy()[0, 0]

    test.assertAlmostEqual(sticky_solver.boundary_damping, 0.0)
    test.assertAlmostEqual(sticky_solver.shape_restitution, 0.0)
    test.assertAlmostEqual(bouncy_solver.shape_restitution, 0.6)
    test.assertAlmostEqual(float(sticky_velocity), 0.0, delta=1.0e-5)
    test.assertAlmostEqual(float(sticky_diffuse_velocity), 0.0, delta=1.0e-5)
    test.assertGreater(float(bouncy_velocity), 0.5)
    test.assertGreater(float(bouncy_diffuse_velocity), 0.5)


def test_sph_shape_collision_honors_box_sdf_and_opt_out(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.0, 0.0, 0.03), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.02)
    builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_hit = model.state()
    state_disabled = model.state()

    solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )
    solver.step(state_0, state_hit, control=None, contacts=None, dt=1.0 / 120.0)

    disabled_solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        shape_collision=False,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )
    disabled_solver.step(state_0, state_disabled, control=None, contacts=None, dt=1.0 / 120.0)

    test.assertGreaterEqual(state_hit.particle_q.numpy()[0, 2], 0.12 - 1.0e-5)
    test.assertLess(state_disabled.particle_q.numpy()[0, 2], 0.1)


def test_sph_collides_with_mesh_shapes(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.0, 0.0, 0.095), vel=(0.0, 0.0, -0.1), mass=0.1, radius=0.02)
    builder.add_shape_mesh(body=-1, mesh=_cube_mesh(0.1))
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        boundary_damping=0.25,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    q = state_1.particle_q.numpy()[0]
    qd = state_1.particle_qd.numpy()[0]
    test.assertGreaterEqual(q[2], 0.12 - 1.0e-5)
    test.assertGreater(qd[2], 0.0)


def test_sph_collides_with_heightfield_shapes(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.0, 0.0, 0.01), vel=(0.0, 0.0, -1.0), mass=0.1, radius=0.02)
    heightfield = newton.Heightfield(
        data=np.zeros((3, 3), dtype=np.float32),
        nrow=3,
        ncol=3,
        hx=1.0,
        hy=1.0,
        min_z=0.0,
        max_z=0.0,
    )
    builder.add_shape_heightfield(heightfield=heightfield)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        boundary_damping=0.25,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    q = state_1.particle_q.numpy()[0]
    qd = state_1.particle_qd.numpy()[0]
    test.assertGreaterEqual(q[2], 0.02 - 1.0e-5)
    test.assertGreater(qd[2], 0.0)


def test_sph_diffuse_particles_collide_with_shapes(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.3, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.02)
    builder.add_shape_sphere(body=-1, radius=0.1)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        max_diffuse_particles=1,
        diffuse_threshold=1.0e9,
        boundary_damping=0.25,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )
    test.assertIsNotNone(solver.diffuse_positions)
    test.assertIsNotNone(solver.diffuse_velocities)
    test.assertIsNotNone(solver.diffuse_worlds)
    solver.diffuse_positions.assign(np.array([[0.04, 0.0, 0.0, 1.0]], dtype=np.float32))
    solver.diffuse_velocities.assign(np.array([[-1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    solver.diffuse_worlds.assign(np.array([-1], dtype=np.int32))

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    diffuse_position = solver.diffuse_positions.numpy()[0]
    diffuse_velocity = solver.diffuse_velocities.numpy()[0]
    test.assertGreaterEqual(diffuse_position[0], 0.12 - 1.0e-5)
    test.assertGreater(diffuse_velocity[0], 0.0)


def test_sph_diffuse_particles_collide_with_heightfield_shapes(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.02
    builder.add_particle(pos=(0.5, 0.0, 0.5), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.02)
    heightfield = newton.Heightfield(
        data=np.zeros((3, 3), dtype=np.float32),
        nrow=3,
        ncol=3,
        hx=1.0,
        hy=1.0,
        min_z=0.0,
        max_z=0.0,
    )
    builder.add_shape_heightfield(heightfield=heightfield)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.08,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        max_diffuse_particles=1,
        diffuse_threshold=1.0e9,
        boundary_damping=0.25,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )
    test.assertIsNotNone(solver.diffuse_positions)
    test.assertIsNotNone(solver.diffuse_velocities)
    test.assertIsNotNone(solver.diffuse_worlds)
    solver.diffuse_positions.assign(np.array([[0.0, 0.0, 0.01, 1.0]], dtype=np.float32))
    solver.diffuse_velocities.assign(np.array([[0.0, 0.0, -1.0, 0.0]], dtype=np.float32))
    solver.diffuse_worlds.assign(np.array([-1], dtype=np.int32))

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    diffuse_position = solver.diffuse_positions.numpy()[0]
    diffuse_velocity = solver.diffuse_velocities.numpy()[0]
    test.assertGreaterEqual(diffuse_position[2], 0.02 - 1.0e-5)
    test.assertGreater(diffuse_velocity[2], 0.0)


def test_sph_diffuse_particles_spawn_from_fast_fluid(test, device):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.default_particle_radius = 0.04
    builder.add_particle_grid(
        pos=wp.vec3(-0.08, -0.08, 0.2),
        rot=wp.quat_identity(),
        vel=wp.vec3(3.0, 0.5, 0.0),
        dim_x=3,
        dim_y=3,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.1,
        jitter=0.0,
        radius_mean=0.04,
        radius_std=0.0,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.12,
        rest_density=50.0,
        gas_constant=0.0,
        max_diffuse_particles=64,
        diffuse_threshold=0.1,
        diffuse_spawn_probability=1.0,
        bounds_lower=(-1.0, -1.0, 0.0),
        bounds_upper=(1.0, 1.0, 1.0),
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

    test.assertIsNotNone(solver.diffuse_positions)
    test.assertIsNotNone(solver.diffuse_velocities)
    diffuse_positions = solver.diffuse_positions.numpy()
    diffuse_velocities = solver.diffuse_velocities.numpy()
    live = diffuse_positions[:, 3] > 0.0
    test.assertGreater(int(np.count_nonzero(live)), 0)
    test.assertTrue(np.all(np.isfinite(diffuse_positions)))
    test.assertTrue(np.all(np.isfinite(diffuse_velocities)))
    diffuse_lifetimes = diffuse_positions[live, 3]
    test.assertGreaterEqual(float(diffuse_lifetimes.min()), 0.35 - 1.0e-6)
    test.assertLess(float(diffuse_lifetimes.max()), 1.0)
    test.assertTrue(np.all(diffuse_velocities[live, 3] >= 0.0))
    max_diffuse_speed = float(np.linalg.norm(diffuse_velocities[live, :3], axis=1).max())
    test.assertGreater(max_diffuse_speed, float(np.linalg.norm([3.0, 0.5, 0.0])) + 0.01)


def test_sph_diffuse_particles_spawn_inside_bounds(test, device):
    bounds_lower = np.array([-0.12, -0.12, 0.0], dtype=np.float32)
    bounds_upper = np.array([0.18, 0.18, 0.36], dtype=np.float32)
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.035
    builder.add_particle_grid(
        pos=wp.vec3(-0.04, -0.04, 0.14),
        rot=wp.quat_identity(),
        vel=wp.vec3(2.5, 0.8, 0.2),
        dim_x=3,
        dim_y=3,
        dim_z=2,
        cell_x=0.045,
        cell_y=0.045,
        cell_z=0.045,
        mass=0.1,
        jitter=0.0,
        radius_mean=0.035,
        radius_std=0.0,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.12,
        rest_density=50.0,
        gas_constant=0.0,
        max_diffuse_particles=64,
        diffuse_threshold=0.05,
        diffuse_spawn_probability=1.0,
        diffuse_jitter=0.35,
        bounds_lower=tuple(float(v) for v in bounds_lower),
        bounds_upper=tuple(float(v) for v in bounds_upper),
        boundary_damping=0.2,
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

    test.assertIsNotNone(solver.diffuse_positions)
    diffuse_positions = solver.diffuse_positions.numpy()
    live_positions = diffuse_positions[diffuse_positions[:, 3] > 0.0, :3]
    test.assertGreater(live_positions.shape[0], 0)
    test.assertTrue(np.all(live_positions >= bounds_lower - 1.0e-6))
    test.assertTrue(np.all(live_positions <= bounds_upper + 1.0e-6))


def test_sph_diffuse_spawn_prefers_free_slots(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    # Two particles separating fast: the trapped-air/wave-crest potential
    # requires separating relative motion with outward-pointing velocity, so an
    # isolated fast particle alone does not spawn foam.
    builder.add_particle(pos=(0.0, 0.0, 0.2), vel=(2.0, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(-0.05, 0.0, 0.2), vel=(-2.0, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.12,
        rest_density=50.0,
        gas_constant=0.0,
        max_diffuse_particles=4,
        diffuse_threshold=0.1,
        diffuse_spawn_probability=1.0,
        bounds_lower=(-1.0, -1.0, 0.0),
        bounds_upper=(1.0, 1.0, 1.0),
    )
    test.assertIsNotNone(solver.diffuse_positions)
    test.assertIsNotNone(solver.diffuse_velocities)
    test.assertIsNotNone(solver.diffuse_slot_states)

    solver.diffuse_positions.assign(
        np.array(
            [
                [0.55, 0.0, 0.2, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    solver.diffuse_velocities.assign(np.zeros((4, 4), dtype=np.float32))
    solver.diffuse_slot_states.assign(np.array([1, 0, 0, 0], dtype=np.int32))

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    diffuse_positions = solver.diffuse_positions.numpy()
    diffuse_slot_states = solver.diffuse_slot_states.numpy()
    live = diffuse_positions[:, 3] > 0.0
    test.assertGreaterEqual(int(np.count_nonzero(live)), 2)
    test.assertGreater(float(diffuse_positions[0, 3]), 0.0)
    test.assertGreater(float(diffuse_positions[0, 0]), 0.50)
    test.assertEqual(int(diffuse_slot_states[0]), 1)
    test.assertTrue(np.any((diffuse_slot_states[1:] == 1) & (diffuse_positions[1:, 3] > 0.0)))


def test_sph_diffuse_visual_density_separates_spray_and_foam(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.035
    builder.add_particle_grid(
        pos=wp.vec3(-0.045, -0.045, -0.045),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        dim_z=3,
        cell_x=0.045,
        cell_y=0.045,
        cell_z=0.045,
        mass=0.1,
        jitter=0.0,
        radius_mean=0.035,
        radius_std=0.0,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=50.0,
        gas_constant=0.0,
        max_diffuse_particles=2,
        diffuse_threshold=1.0e9,
        diffuse_ballistic=6,
        bounds_lower=(-2.0, -2.0, -2.0),
        bounds_upper=(2.0, 2.0, 2.0),
    )
    test.assertIsNotNone(solver.diffuse_positions)
    test.assertIsNotNone(solver.diffuse_velocities)
    test.assertIsNotNone(solver.diffuse_worlds)
    solver.diffuse_positions.assign(
        np.array(
            [
                [0.75, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    solver.diffuse_velocities.assign(
        np.array(
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    solver.diffuse_worlds.assign(np.array([-1, -1], dtype=np.int32))

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    visual_density = solver.diffuse_velocities.numpy()[:, 3]
    test.assertLess(float(visual_density[0]), 1.0)
    test.assertGreater(float(visual_density[1]), float(solver.diffuse_ballistic))


def test_sph_flex_style_force_terms_are_finite_and_cohesive(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(-0.05, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.05, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.0, 0.07, 0.0), vel=(-0.2, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.14,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        cohesion=8.0,
        surface_tension=1.0e-6,
        vorticity_confinement=1.0e-5,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    initial_distance = np.linalg.norm(state_0.particle_q.numpy()[1] - state_0.particle_q.numpy()[0])
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

    final_distance = np.linalg.norm(state_1.particle_q.numpy()[1] - state_1.particle_q.numpy()[0])
    vorticity = solver.particle_vorticity.numpy()
    q = state_1.particle_q.numpy()
    qd = state_1.particle_qd.numpy()

    test.assertLess(final_distance, initial_distance)
    test.assertTrue(np.all(np.isfinite(vorticity)))
    test.assertTrue(np.all(np.isfinite(q)))
    test.assertTrue(np.all(np.isfinite(qd)))
    test.assertGreater(float(np.linalg.norm(vorticity, axis=1).max()), 0.0)


def test_sph_free_surface_drag_damps_low_density_particles(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_drag = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=5000.0,
        gas_constant=0.0,
        viscosity=0.0,
        free_surface_drag=3.0,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_drag, control=None, contacts=None, dt=1.0 / 60.0)

    initial_speed = float(np.linalg.norm(state_0.particle_qd.numpy()[0]))
    damped_speed = float(np.linalg.norm(state_drag.particle_qd.numpy()[0]))
    test.assertGreater(solver.free_surface_drag, 0.0)
    test.assertLess(damped_speed, initial_speed)


def test_sph_dissipation_damps_particles_with_neighbors(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(-0.04, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_dissipated = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        dissipation=16.0,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_dissipated, control=None, contacts=None, dt=0.1)

    speeds = np.linalg.norm(state_dissipated.particle_qd.numpy(), axis=1)
    test.assertGreater(solver.dissipation, 0.0)
    test.assertLess(float(speeds[0]), 1.0)
    test.assertLess(float(speeds[1]), 1.0)
    test.assertAlmostEqual(float(speeds[2]), 1.0, delta=1.0e-6)
    test.assertLess(float(max(speeds[0], speeds[1])), float(speeds[2]) - 0.05)


def test_sph_particle_friction_damps_neighbor_tangential_velocity(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(-0.04, 0.0, 0.0), vel=(0.0, 1.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(0.0, -1.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_none = model.state()
    state_friction = model.state()
    common_kwargs = {
        "smoothing_length": 0.13,
        "rest_density": 50.0,
        "gas_constant": 0.0,
        "viscosity": 0.0,
        "bounds_lower": (-10.0, -10.0, -10.0),
        "bounds_upper": (10.0, 10.0, 10.0),
    }
    SolverSPH(model, **common_kwargs).step(state_0, state_none, control=None, contacts=None, dt=0.1)
    friction_solver = SolverSPH(model, particle_friction=6.0, **common_kwargs)
    friction_solver.step(state_0, state_friction, control=None, contacts=None, dt=0.1)

    no_friction_relative_speed = np.linalg.norm(state_none.particle_qd.numpy()[0] - state_none.particle_qd.numpy()[1])
    friction_relative_speed = np.linalg.norm(
        state_friction.particle_qd.numpy()[0] - state_friction.particle_qd.numpy()[1]
    )
    test.assertGreater(friction_solver.particle_friction, 0.0)
    test.assertGreater(float(no_friction_relative_speed), 1.9)
    test.assertLess(float(friction_relative_speed), float(no_friction_relative_speed))
    test.assertGreater(float(friction_relative_speed), 0.0)


def test_sph_particle_collision_margin_damps_near_contacts(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(-0.08, 0.0, 0.0), vel=(0.0, 1.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.08, 0.0, 0.0), vel=(0.0, -1.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_no_margin = model.state()
    state_margin = model.state()
    common_kwargs = {
        "smoothing_length": 0.13,
        "rest_density": 50.0,
        "gas_constant": 0.0,
        "viscosity": 0.0,
        "particle_friction": 6.0,
        "bounds_lower": (-10.0, -10.0, -10.0),
        "bounds_upper": (10.0, 10.0, 10.0),
    }
    SolverSPH(model, particle_collision_margin=0.0, **common_kwargs).step(
        state_0, state_no_margin, control=None, contacts=None, dt=0.1
    )
    margin_solver = SolverSPH(model, particle_collision_margin=0.08, **common_kwargs)
    margin_solver.step(state_0, state_margin, control=None, contacts=None, dt=0.1)

    no_margin_relative_speed = np.linalg.norm(
        state_no_margin.particle_qd.numpy()[0] - state_no_margin.particle_qd.numpy()[1]
    )
    margin_relative_speed = np.linalg.norm(state_margin.particle_qd.numpy()[0] - state_margin.particle_qd.numpy()[1])
    test.assertAlmostEqual(margin_solver.particle_collision_margin, 0.08)
    test.assertGreater(float(no_margin_relative_speed), 1.9)
    test.assertLess(float(margin_relative_speed), float(no_margin_relative_speed) - 0.1)
    test.assertGreater(float(margin_relative_speed), 0.0)


def test_sph_sleep_threshold_freezes_slow_particles(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.05, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.3, 0.0, 0.0), vel=(0.2, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_sleep = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        sleep_threshold=0.1,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_sleep, control=None, contacts=None, dt=0.1)

    positions = state_sleep.particle_q.numpy()
    velocities = state_sleep.particle_qd.numpy()
    test.assertAlmostEqual(solver.sleep_threshold, 0.1)
    np.testing.assert_allclose(velocities[0], np.zeros(3, dtype=np.float32), atol=1.0e-6)
    np.testing.assert_allclose(positions[0], np.zeros(3, dtype=np.float32), atol=1.0e-6)
    test.assertGreater(float(np.linalg.norm(velocities[1])), 0.15)
    test.assertGreater(float(positions[1, 0]), 0.31)


def test_sph_max_acceleration_limits_force_and_xsph_velocity_delta(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_limited = model.state()
    state_0.particle_f.assign(np.array([[1000.0, 0.0, 0.0]], dtype=np.float32))
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=1000.0,
        gas_constant=0.0,
        viscosity=0.0,
        max_velocity=100.0,
        max_acceleration=3.0,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )
    dt = 0.1

    solver.step(state_0, state_limited, control=None, contacts=None, dt=dt)

    force_delta_v = np.linalg.norm(state_limited.particle_qd.numpy()[0] - state_0.particle_qd.numpy()[0])
    test.assertAlmostEqual(solver.max_acceleration, 3.0)
    test.assertLessEqual(float(force_delta_v), 3.0 * dt + 1.0e-6)
    test.assertGreater(float(force_delta_v), 0.25)

    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(-0.03, 0.0, 0.0), vel=(0.1, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.03, 0.0, 0.0), vel=(-0.1, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_smooth = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        xsph_strength=1.0,
        max_velocity=100.0,
        max_acceleration=0.5,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_smooth, control=None, contacts=None, dt=dt)

    xsph_delta_v = np.linalg.norm(state_smooth.particle_qd.numpy() - state_0.particle_qd.numpy(), axis=1)
    test.assertTrue(np.all(xsph_delta_v <= 0.5 * dt + 1.0e-6))
    test.assertGreater(float(np.max(xsph_delta_v)), 0.04)


def test_sph_solid_pressure_pushes_particles_away_from_bounds(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(0.0, 0.0, 0.05), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_solid = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=1000.0,
        gas_constant=0.0,
        viscosity=0.0,
        solid_pressure=1.0,
        bounds_lower=(-10.0, -10.0, 0.0),
        bounds_upper=(10.0, 10.0, 1.0),
    )

    solver.step(state_0, state_solid, control=None, contacts=None, dt=1.0 / 60.0)

    test.assertGreater(solver.solid_pressure, 0.0)
    test.assertGreater(state_solid.particle_qd.numpy()[0, 2], 0.0)
    test.assertGreater(state_solid.particle_q.numpy()[0, 2], state_0.particle_q.numpy()[0, 2])


def test_sph_buoyancy_scales_fluid_gravity(test, device):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(0.0, 0.0, 0.2), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_scaled = model.state()
    buoyancy = 0.25
    dt = 1.0 / 30.0
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=1000.0,
        gas_constant=0.0,
        viscosity=0.0,
        buoyancy=buoyancy,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    solver.step(state_0, state_scaled, control=None, contacts=None, dt=dt)

    expected_velocity = model.gravity.numpy()[0] * buoyancy * dt
    test.assertAlmostEqual(solver.buoyancy, buoyancy)
    test.assertTrue(np.allclose(state_scaled.particle_qd.numpy()[0], expected_velocity, rtol=1.0e-5, atol=1.0e-6))


def test_sph_xsph_velocity_smoothing_reduces_local_velocity_noise(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    builder.add_particle(pos=(-0.04, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.04, 0.0, 0.0), vel=(-1.0, 0.0, 0.0), mass=0.1, radius=0.04)
    builder.add_particle(pos=(0.0, 0.06, 0.0), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.04)
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        xsph_strength=0.75,
        bounds_lower=(-10.0, -10.0, -10.0),
        bounds_upper=(10.0, 10.0, 10.0),
    )

    initial_relative_speed = np.linalg.norm(state_0.particle_qd.numpy()[0] - state_0.particle_qd.numpy()[1])
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 240.0)

    smoothed_velocities = state_1.particle_qd.numpy()
    final_relative_speed = np.linalg.norm(smoothed_velocities[0] - smoothed_velocities[1])

    test.assertTrue(solver.xsph_enabled)
    test.assertIsNotNone(solver.particle_velocity_smooth)
    test.assertTrue(np.all(np.isfinite(smoothed_velocities)))
    test.assertLess(final_relative_speed, initial_relative_speed)


def test_sph_render_buffers_smooth_and_stretch_particles(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    for i in range(7):
        builder.add_particle(
            pos=(-0.12 + 0.04 * i, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.1,
            radius=0.04,
            flags=int(newton.ParticleFlags.ACTIVE) if i != 3 else 0,
        )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        render_smoothing=0.45,
        render_anisotropy_scale=0.82,
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    test.assertTrue(solver.render_buffers_valid)
    test.assertIsNotNone(solver.render_positions)
    test.assertIsNotNone(solver.render_anisotropy)
    test.assertIsNotNone(solver.render_anisotropy_secondary)
    test.assertIsNotNone(solver.render_anisotropy_tertiary)
    render_positions = solver.render_positions.numpy()
    anisotropy = solver.render_anisotropy.numpy()
    anisotropy_secondary = solver.render_anisotropy_secondary.numpy()
    anisotropy_tertiary = solver.render_anisotropy_tertiary.numpy()

    test.assertEqual(render_positions.shape, (model.particle_count, 3))
    test.assertEqual(anisotropy.shape, (model.particle_count, 4))
    test.assertEqual(anisotropy_secondary.shape, (model.particle_count, 4))
    test.assertEqual(anisotropy_tertiary.shape, (model.particle_count, 4))
    test.assertTrue(np.all(np.isfinite(render_positions)))
    test.assertTrue(np.all(np.isfinite(anisotropy)))
    test.assertTrue(np.all(np.isfinite(anisotropy_secondary)))
    test.assertTrue(np.all(np.isfinite(anisotropy_tertiary)))
    test.assertEqual(float(anisotropy[3, 3]), 0.0)
    test.assertEqual(float(anisotropy_secondary[3, 3]), 0.0)
    test.assertEqual(float(anisotropy_tertiary[3, 3]), 0.0)
    test.assertGreater(float(np.max(anisotropy[[0, 1, 2, 4, 5, 6], 3])), 1.0)
    test.assertLessEqual(float(np.max(anisotropy[[0, 1, 2, 4, 5, 6], 3])), 2.0 + 1.0e-5)
    active = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    test.assertTrue(np.all(anisotropy_secondary[active, 3] > 0.0))
    test.assertTrue(np.all(anisotropy_tertiary[active, 3] > 0.0))
    test.assertLessEqual(float(np.max(anisotropy_secondary[active, 3])), 1.0)
    test.assertLessEqual(float(np.max(anisotropy_tertiary[active, 3])), 1.0)
    axis_dot = np.sum(anisotropy[active, :3] * anisotropy_secondary[active, :3], axis=1)
    axis_depth_dot = np.sum(anisotropy[active, :3] * anisotropy_tertiary[active, :3], axis=1)
    side_depth_dot = np.sum(anisotropy_secondary[active, :3] * anisotropy_tertiary[active, :3], axis=1)
    test.assertLess(float(np.max(np.abs(axis_dot))), 0.35)
    test.assertLess(float(np.max(np.abs(axis_depth_dot))), 0.35)
    test.assertLess(float(np.max(np.abs(side_depth_dot))), 0.35)
    test.assertGreater(float(np.linalg.norm(render_positions[0] - state_1.particle_q.numpy()[0])), 1.0e-4)


def test_sph_render_anisotropy_clamps_match_solver_settings(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = 0.04
    for i in range(7):
        builder.add_particle(
            pos=(-0.12 + 0.04 * i, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.1,
            radius=0.04,
        )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(
        model,
        smoothing_length=0.13,
        rest_density=50.0,
        gas_constant=0.0,
        viscosity=0.0,
        render_smoothing=0.0,
        render_anisotropy_scale=6.0,
        render_anisotropy_min=0.25,
        render_anisotropy_max=1.25,
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 120.0)

    test.assertIsNotNone(solver.render_anisotropy)
    anisotropy = solver.render_anisotropy.numpy()
    stretched = anisotropy[:, 3]
    test.assertGreater(float(np.max(stretched)), 1.20)
    test.assertTrue(np.all(stretched <= 1.25 + 1.0e-5))
    test.assertTrue(np.all(solver.render_anisotropy_secondary.numpy()[:, 3] >= 0.25 - 1.0e-5))
    test.assertTrue(np.all(solver.render_anisotropy_tertiary.numpy()[:, 3] >= 0.25 - 1.0e-5))


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
add_function_test(
    TestSolverSPH, "test_sph_computes_positive_density", test_sph_computes_positive_density, devices=devices
)
add_function_test(
    TestSolverSPH,
    "test_sph_pressure_separates_overlapping_particles",
    test_sph_pressure_separates_overlapping_particles,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_pbf_projection_separates_without_pressure",
    test_sph_pbf_projection_separates_without_pressure,
    devices=devices,
)
add_function_test(TestSolverSPH, "test_sph_respects_world_bounds", test_sph_respects_world_bounds, devices=devices)
add_function_test(
    TestSolverSPH,
    "test_sph_collides_with_particle_shapes",
    test_sph_collides_with_particle_shapes,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_shape_collision_honors_box_sdf_and_opt_out",
    test_sph_shape_collision_honors_box_sdf_and_opt_out,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_shape_collision_accumulates_dynamic_body_feedback",
    test_sph_shape_collision_accumulates_dynamic_body_feedback,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_shape_adhesion_damps_fluid_and_diffuse_rebound",
    test_sph_shape_adhesion_damps_fluid_and_diffuse_rebound,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_shape_friction_damps_fluid_and_diffuse_slide",
    test_sph_shape_friction_damps_fluid_and_diffuse_slide,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_shape_collision_distance_separates_from_margin",
    test_sph_shape_collision_distance_separates_from_margin,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_shape_restitution_overrides_boundary_damping",
    test_sph_shape_restitution_overrides_boundary_damping,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_collides_with_mesh_shapes",
    test_sph_collides_with_mesh_shapes,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_collides_with_heightfield_shapes",
    test_sph_collides_with_heightfield_shapes,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_diffuse_particles_collide_with_shapes",
    test_sph_diffuse_particles_collide_with_shapes,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_diffuse_particles_collide_with_heightfield_shapes",
    test_sph_diffuse_particles_collide_with_heightfield_shapes,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_diffuse_particles_spawn_from_fast_fluid",
    test_sph_diffuse_particles_spawn_from_fast_fluid,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_diffuse_particles_spawn_inside_bounds",
    test_sph_diffuse_particles_spawn_inside_bounds,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_diffuse_spawn_prefers_free_slots",
    test_sph_diffuse_spawn_prefers_free_slots,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_diffuse_visual_density_separates_spray_and_foam",
    test_sph_diffuse_visual_density_separates_spray_and_foam,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_flex_style_force_terms_are_finite_and_cohesive",
    test_sph_flex_style_force_terms_are_finite_and_cohesive,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_free_surface_drag_damps_low_density_particles",
    test_sph_free_surface_drag_damps_low_density_particles,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_dissipation_damps_particles_with_neighbors",
    test_sph_dissipation_damps_particles_with_neighbors,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_particle_friction_damps_neighbor_tangential_velocity",
    test_sph_particle_friction_damps_neighbor_tangential_velocity,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_particle_collision_margin_damps_near_contacts",
    test_sph_particle_collision_margin_damps_near_contacts,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_sleep_threshold_freezes_slow_particles",
    test_sph_sleep_threshold_freezes_slow_particles,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_max_acceleration_limits_force_and_xsph_velocity_delta",
    test_sph_max_acceleration_limits_force_and_xsph_velocity_delta,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_solid_pressure_pushes_particles_away_from_bounds",
    test_sph_solid_pressure_pushes_particles_away_from_bounds,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_buoyancy_scales_fluid_gravity",
    test_sph_buoyancy_scales_fluid_gravity,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_xsph_velocity_smoothing_reduces_local_velocity_noise",
    test_sph_xsph_velocity_smoothing_reduces_local_velocity_noise,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_render_buffers_smooth_and_stretch_particles",
    test_sph_render_buffers_smooth_and_stretch_particles,
    devices=devices,
)
add_function_test(
    TestSolverSPH,
    "test_sph_render_anisotropy_clamps_match_solver_settings",
    test_sph_render_anisotropy_clamps_match_solver_settings,
    devices=devices,
)
add_function_test(TestSolverSPH, "test_sph_cuda_graph_capture", test_sph_cuda_graph_capture, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
