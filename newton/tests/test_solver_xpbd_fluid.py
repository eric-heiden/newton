# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the position-based fluid (PBF) support in the XPBD solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.xpbd.fluid_kernels import (
    _reserve_diffuse_slot,
    compute_fluid_lambdas,
    spawn_fluid_diffuse_particles,
)
from newton._src.solvers.xpbd.kernels import apply_particle_deltas, clamp_body_motion, clamp_body_velocities
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

SPACING = 0.05
RADIUS = 0.025
REST_DENSITY = 1000.0
PARTICLE_MASS = REST_DENSITY * SPACING**3
FLUID_FLAGS = newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID


@wp.kernel
def _reserve_diffuse_slot_kernel(
    request: int,
    slot_states: wp.array[wp.int32],
    result: wp.array[wp.int32],
):
    result[0] = _reserve_diffuse_slot(request, slot_states)


def _build_fluid_grid(device, dims=(6, 6, 6), spacing=SPACING, gravity=0.0, ground=False, z0=0.0, fluid=True):
    builder = newton.ModelBuilder(up_axis="Z", gravity=gravity)
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, z0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dims[0],
        dim_y=dims[1],
        dim_z=dims[2],
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=RADIUS,
        flags=FLUID_FLAGS if fluid else newton.ParticleFlags.ACTIVE,
    )
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_overlapping_world_fluid_grids(device, dims=(4, 4, 4)):
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.default_particle_radius = RADIUS
    for _ in range(2):
        builder.begin_world()
        builder.add_particle_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dims[0],
            dim_y=dims[1],
            dim_z=dims[2],
            cell_x=SPACING,
            cell_y=SPACING,
            cell_z=SPACING,
            mass=PARTICLE_MASS,
            jitter=0.0,
            radius_mean=RADIUS,
            flags=FLUID_FLAGS,
        )
        builder.end_world()
    return builder.finalize(device=device)


def _simulate(model, steps, ground=False, iterations=3, dt=1.0 / 240.0, **solver_kwargs):
    solver = newton.solvers.SolverXPBD(model, iterations=iterations, fluid_rest_distance=SPACING, **solver_kwargs)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts() if ground else None
    for _ in range(steps):
        state_0.clear_forces()
        if ground:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0, solver


def _densities(q, solver):
    """Poly6 density of each particle, normalized by the solver's rest density."""
    h = solver._fluid_h
    d = np.linalg.norm(q[None, :, :] - q[:, None, :], axis=-1)
    w = np.where(d < h, (h * h - d**2) ** 3, 0.0) * 315.0 / (64.0 * np.pi * h**9)
    return PARTICLE_MASS * w.sum(axis=1) / solver._fluid_rest_density_eff


def test_fluid_rest_lattice_stays_at_rest(test, device):
    """A particle grid at rest spacing without gravity must stay bounded.

    The rest density is calibrated from the lattice, so the density constraint is
    inactive in the bulk (see :func:`test_fluid_rest_lattice_exact_without_cohesion`,
    which holds it to machine precision with cohesion off). With the default
    (maximum) cohesion the surface tension does reshape the small cube toward a
    rounder blob -- a few times the rest spacing of contraction -- but it must
    stay contractive and bounded, never gaining energy and blowing up.
    """
    model = _build_fluid_grid(device)
    q0 = model.particle_q.numpy().copy()
    state, _solver = _simulate(model, 240)

    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    drift = np.linalg.norm(q - q0, axis=1).max()
    max_speed = np.linalg.norm(qd, axis=1).max()
    # cohesion reshapes the cube but must stay bounded (a blow-up runs away far
    # past this and also trips the speed check); the speed bound is the real
    # "no energy gain" guard
    test.assertLess(drift, 0.25, f"rest lattice drifted {drift:.4f} m")
    test.assertLess(max_speed, 1.0, f"rest lattice reached {max_speed:.3f} m/s")


def test_fluid_rest_lattice_exact_without_cohesion(test, device):
    """Without cohesion the calibrated rest lattice is an exact equilibrium."""
    model = _build_fluid_grid(device)
    q0 = model.particle_q.numpy().copy()
    state, _solver = _simulate(model, 120, fluid_cohesion=0.0)

    drift = np.linalg.norm(state.particle_q.numpy() - q0, axis=1).max()
    test.assertLess(drift, 1.0e-4, f"rest lattice drifted {drift:.6f} m without cohesion")


def test_fluid_compressed_block_decompresses(test, device):
    """A block compressed to 2x rest density must expand toward rest density."""
    model = _build_fluid_grid(device, spacing=0.8 * SPACING)
    state, solver = _simulate(model, 480, fluid_cohesion=0.5, fluid_viscosity=0.05)

    q = state.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    rho = _densities(q, solver)
    initial_ratio = (SPACING / (0.8 * SPACING)) ** 3  # ~1.95
    test.assertGreater(initial_ratio, 1.9)
    test.assertLess(
        float(np.percentile(rho, 90)),
        1.25,
        "compressed fluid did not decompress toward rest density",
    )


def test_fluid_drop_forms_cohesive_puddle(test, device):
    """A fluid cube dropped on the ground must settle into a compact puddle.

    With cohesion the puddle stays bound (surface tension); without it the
    particles disperse into a thin monolayer. Both must remain stable.
    """
    model = _build_fluid_grid(device, gravity=-9.81, ground=True, z0=0.3)
    state, _solver = _simulate(model, 480, ground=True, fluid_cohesion=1.0, fluid_viscosity=0.05)
    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertTrue(np.isfinite(qd).all())
    test.assertGreater(q[:, 2].min(), -RADIUS, "particles fell through the ground")
    test.assertLess(np.linalg.norm(qd, axis=1).max(), 1.0, "puddle did not settle")
    spread_cohesive = np.linalg.norm(q[:, :2] - q[:, :2].mean(axis=0), axis=1).max()
    test.assertLess(spread_cohesive, 0.6, "cohesive puddle dispersed too far")

    model = _build_fluid_grid(device, gravity=-9.81, ground=True, z0=0.3)
    state, _solver = _simulate(model, 480, ground=True, fluid_cohesion=0.0, fluid_viscosity=0.05)
    q0 = state.particle_q.numpy()
    test.assertTrue(np.isfinite(q0).all())
    spread_loose = np.linalg.norm(q0[:, :2] - q0[:, :2].mean(axis=0), axis=1).max()
    test.assertGreater(
        spread_loose,
        spread_cohesive,
        "cohesion should reduce how far the splash disperses",
    )


def test_fluid_density_corrections_do_not_cancel_shape_contacts(test, device):
    """A strong density correction must not push fluid through a collider.

    The compressed block expands in every direction. If density and contact
    corrections share one Jacobi sum, the downward density correction cancels
    part of the ground correction and moves particle centers inside the plane.
    """
    compressed_spacing = 0.7 * SPACING
    model = _build_fluid_grid(
        device,
        dims=(6, 6, 6),
        spacing=compressed_spacing,
        ground=True,
        z0=RADIUS,
    )
    dt = 1.0 / 120.0
    model.particle_max_velocity = 0.5 * RADIUS / dt
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=2,
        fluid_rest_distance=SPACING,
        fluid_cohesion=0.0,
    )
    state_0, state_1 = model.state(), model.state()
    contacts = model.contacts()

    minimum_height = float("inf")
    for _ in range(5):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
        minimum_height = min(minimum_height, float(state_0.particle_q.numpy()[:, 2].min()))

    test.assertGreaterEqual(
        minimum_height,
        RADIUS - 1.0e-6,
        f"density correction pushed fluid {RADIUS - minimum_height:.6f} m into the ground",
    )


def test_fluid_pressure_reprojection_conserves_momentum(test, device):
    """A rigid body must receive the reaction from a post-pressure contact correction."""
    particle_mass = 0.125
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.default_particle_radius = RADIUS
    builder.add_particles(
        pos=[wp.vec3(0.0), wp.vec3(-0.01, 0.0, 0.0)],
        vel=[wp.vec3(0.0)] * 2,
        mass=[particle_mass] * 2,
        radius=[RADIUS] * 2,
        flags=[int(FLUID_FLAGS)] * 2,
    )
    body = builder.add_body(xform=wp.transform(wp.vec3(0.075, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(
        body,
        hx=0.05,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0),
    )
    model = builder.finalize(device=device)
    model.particle_max_velocity = 100.0

    state_0, state_1 = model.state(), model.state()
    contacts = model.contacts()
    model.collide(state_0, contacts)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=1,
        soft_contact_relaxation=1.0,
        fluid_rest_distance=SPACING,
        fluid_rest_density=100.0,
        fluid_cohesion=0.0,
        fluid_relaxation=1.0,
    )
    solver.step(state_0, state_1, None, contacts, 1.0 / 120.0)

    particle_momentum = particle_mass * state_1.particle_qd.numpy()[:, 0].sum()
    body_velocity = float(state_1.body_qd.numpy()[body, 0])
    body_momentum = float(model.body_mass.numpy()[body]) * body_velocity
    test.assertGreater(body_velocity, 0.05, "fluid pressure did not react against the dynamic body")
    test.assertAlmostEqual(float(particle_momentum + body_momentum), 0.0, places=5)


def test_fluid_drop_settles_without_viscosity(test, device):
    """Contact projection must dissipate boundary buzz without XSPH viscosity."""
    model = _build_fluid_grid(device, gravity=-9.81, ground=True, z0=0.3)
    model.particle_max_velocity = 3.0
    state, _solver = _simulate(
        model,
        480,
        ground=True,
        fluid_cohesion=0.5,
        fluid_viscosity=0.0,
    )

    q = state.particle_q.numpy()
    speed = np.linalg.norm(state.particle_qd.numpy(), axis=1)
    test.assertGreaterEqual(float(q[:, 2].min()), RADIUS - 1.0e-6)
    test.assertLess(float(speed.mean()), 0.01, "fluid retained excessive boundary jitter")
    test.assertLess(float(np.percentile(speed, 95)), 0.02, "fluid failed to settle without viscosity")


def test_fluid_pair_coheres_without_oscillation(test, device):
    """Two separated fluid particles must pull together to a stable spacing.

    Near-isolated particles have a saturated density deficit; constraint-based
    attraction diverges for them, so this guards the bounded cohesion term.
    """
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.add_particles(
        pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.5 * SPACING, 0.0, 0.0)],
        vel=[wp.vec3(0.0)] * 2,
        mass=[PARTICLE_MASS] * 2,
        radius=[RADIUS] * 2,
        flags=[int(FLUID_FLAGS)] * 2,
    )
    model = builder.finalize(device=device)
    state, _solver = _simulate(model, 480)

    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    dist = float(np.linalg.norm(q[1] - q[0]))
    test.assertLess(dist, 1.5 * SPACING, "pair did not cohere")
    test.assertGreater(dist, 0.1 * SPACING, "pair collapsed to a point")
    test.assertLess(np.linalg.norm(qd, axis=1).max(), 0.5, "pair oscillates")


def test_fluid_pairs_skip_contact_constraints(test, device):
    """Fluid-fluid pairs must not generate XPBD contact constraints.

    Fluid particles rest at less than two collision radii apart; if the contact
    kernel also acted on them it would fight the density constraint and push
    them to 2*radius spacing.
    """

    def run(fluid):
        builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
        flags = FLUID_FLAGS if fluid else newton.ParticleFlags.ACTIVE
        # closer than 2*radius: a contact constraint would push them apart
        builder.add_particles(
            pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.2 * RADIUS, 0.0, 0.0)],
            vel=[wp.vec3(0.0)] * 2,
            mass=[PARTICLE_MASS] * 2,
            radius=[RADIUS] * 2,
            flags=[int(flags)] * 2,
        )
        model = builder.finalize(device=device)
        state, _solver = _simulate(model, 60, fluid_cohesion=0.0)
        q = state.particle_q.numpy()
        return float(np.linalg.norm(q[1] - q[0]))

    dist_fluid = run(fluid=True)
    dist_solid = run(fluid=False)
    test.assertGreaterEqual(dist_solid, 2.0 * RADIUS - 1.0e-4, "solid contact should separate the pair")
    test.assertLess(dist_fluid, 2.0 * RADIUS - 1.0e-4, "fluid pair must not be separated by contact constraints")


def test_fluid_render_particles(test, device):
    """update_render_particles fills smoothed positions and ellipsoid axes."""
    model = _build_fluid_grid(device)
    state, solver = _simulate(model, 10)

    solver.update_render_particles(state, smoothing=0.5, anisotropy_scale=1.0)
    test.assertIsNotNone(solver.render_positions)
    render_q = solver.render_positions.numpy()
    test.assertEqual(render_q.shape, (model.particle_count, 3))
    test.assertTrue(np.isfinite(render_q).all())
    # smoothed positions must stay near the simulated positions
    err = np.linalg.norm(render_q - state.particle_q.numpy(), axis=1).max()
    test.assertLess(err, solver._fluid_h)
    for aniso in (
        solver.render_anisotropy,
        solver.render_anisotropy_secondary,
        solver.render_anisotropy_tertiary,
    ):
        a = aniso.numpy()
        test.assertEqual(a.shape, (model.particle_count, 4))
        test.assertTrue(np.isfinite(a).all())
        test.assertGreater(a[:, 3].min(), 0.0)


def test_fluid_render_particle_limit_and_fast_path(test, device):
    model = _build_fluid_grid(device, dims=(4, 4, 4))
    state = model.state()
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=SPACING)

    solver.update_render_particles(state, smoothing=0.0, anisotropy_scale=0.0, max_particles=10)
    source = state.particle_q.numpy()
    indices = np.minimum(np.arange(10) * model.particle_count // 10, model.particle_count - 1)
    np.testing.assert_allclose(solver.render_positions.numpy(), source[indices], atol=0.0)
    test.assertIsNone(solver.render_anisotropy)
    test.assertIsNone(solver.render_anisotropy_secondary)
    test.assertIsNone(solver.render_anisotropy_tertiary)

    solver.update_render_particles(state, smoothing=0.5, anisotropy_scale=1.0, max_particles=10)
    test.assertEqual(solver.render_positions.shape, (10,))
    test.assertEqual(solver.render_anisotropy.shape, (10,))


def test_fluid_diffuse_particles_spawn_and_expire(test, device):
    """A splashing drop must emit diffuse foam particles that age and expire."""
    model = _build_fluid_grid(device, gravity=-9.81, ground=True, z0=0.4)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=3,
        fluid_rest_distance=SPACING,
        fluid_cohesion=0.5,
        fluid_viscosity=0.05,
        max_diffuse_particles=2000,
        diffuse_threshold=1.0,
        diffuse_lifetime=0.5,
    )
    test.assertIsNotNone(solver.diffuse_positions)

    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 1.0 / 240.0

    def alive_count():
        return int(np.count_nonzero(solver.diffuse_positions.numpy()[:, 3] > 0.0))

    # drop and splash: foam must spawn around the impact
    for _ in range(120):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
    spawned = int(solver.diffuse_spawn_counter.numpy()[0])
    test.assertGreater(spawned, 0, "splash did not emit diffuse particles")
    alive_after_splash = alive_count()
    test.assertGreater(alive_after_splash, 0, "no live diffuse particles after the splash")

    diffuse_q = solver.diffuse_positions.numpy()
    live = diffuse_q[:, 3] > 0.0
    test.assertTrue(np.isfinite(diffuse_q[live]).all())
    test.assertGreater(diffuse_q[live][:, 2].min(), -2.0 * RADIUS, "diffuse particles fell through the ground")

    # once the fluid settles, spawning stops and the foam expires
    for _ in range(360):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
    test.assertLess(alive_count(), alive_after_splash, "diffuse particles did not expire")


def test_fluid_diffuse_disabled_by_default(test, device):
    """Without max_diffuse_particles the foam layer stays unallocated."""
    model = _build_fluid_grid(device)
    _state, solver = _simulate(model, 5)
    test.assertIsNone(solver.diffuse_positions)
    test.assertFalse(solver.diffuse_enabled)


def _watertight_box_mesh(hx, hy, hz):
    """A watertight (outward-wound) box mesh centered at the origin."""
    v = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # bottom (-z)
            [4, 5, 6],
            [4, 6, 7],  # top (+z)
            [0, 1, 5],
            [0, 5, 4],  # -y
            [2, 3, 7],
            [2, 7, 6],  # +y
            [1, 2, 6],
            [1, 6, 5],  # +x
            [3, 0, 4],
            [3, 4, 7],  # -x
        ],
        dtype=np.int32,
    ).flatten()
    return newton.Mesh(v, f)


def test_fluid_sdf_mesh_contains_particles(test, device):
    """A mesh with a texture SDF should contain fluid via the SDF soft-contact path.

    Exercises ``create_soft_contacts_sdf`` (CUDA-only): fluid dropped onto a
    static SDF box slab must rest on top instead of tunneling through, and the
    slab must be flagged as carrying an SDF.
    """
    if not wp.get_device(device).is_cuda:
        test.skipTest("texture SDFs require CUDA")

    slab_top = 0.1
    mesh = _watertight_box_mesh(0.5, 0.5, 0.5 * slab_top)
    mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.1, 0.1), margin=0.05)

    builder = newton.ModelBuilder(up_axis="Z", gravity=-9.81)
    builder.default_particle_radius = RADIUS
    slab = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * slab_top), wp.quat_identity()))
    builder.add_shape_mesh(
        slab,
        mesh=mesh,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=True,
        ),
    )
    builder.body_flags[slab] = int(newton.BodyFlags.KINEMATIC)
    builder.add_particle_grid(
        pos=wp.vec3(-0.1, -0.1, slab_top + RADIUS),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=5,
        dim_y=5,
        dim_z=5,
        cell_x=SPACING,
        cell_y=SPACING,
        cell_z=SPACING,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=RADIUS,
        flags=FLUID_FLAGS,
    )
    model = builder.finalize(device=device)
    test.assertGreaterEqual(int(model._shape_sdf_index.numpy()[0]), 0, "slab mesh should carry an SDF")

    solver = newton.solvers.SolverXPBD(model, iterations=3, fluid_rest_distance=SPACING)
    state_0, state_1 = model.state(), model.state()
    contacts = model.contacts()
    dt = 1.0 / 120.0
    for _ in range(120):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    q = state_0.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    # particles must rest on top of the slab, not tunnel through it
    test.assertGreater(q[:, 2].min(), slab_top - RADIUS, "fluid tunneled through the SDF slab")


def test_fluid_max_neighbors_truncates_density(test, device):
    """``fluid_max_neighbors`` caps the neighbor loop: a cap below the local
    count lowers the density estimate, while a cap above it is a no-op.

    ``compute_fluid_lambdas`` writes one entry per thread with no atomics, so
    it is deterministic and can be compared bit-exactly across launches.
    """
    model = _build_fluid_grid(device, dims=(6, 6, 6))
    solver = newton.solvers.SolverXPBD(model, iterations=1, fluid_rest_distance=SPACING)
    state = model.state()
    h = solver._fluid_h
    model.particle_grid.build(state.particle_q, radius=h, groups=model.particle_world)
    n = model.particle_count

    def densities_with_cap(cap):
        density = wp.zeros(n, dtype=wp.float32, device=device)
        lam = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            compute_fluid_lambdas,
            dim=n,
            inputs=[
                model.particle_grid.id,
                state.particle_q,
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                model.particle_world,
                h,
                solver._fluid_rest_density_eff,
                solver._fluid_eps,
                cap,
                solver._fluid_rest_distance_eff,
            ],
            outputs=[density, lam],
            device=device,
        )
        return density.numpy()

    uncapped = densities_with_cap(0)
    truncated = densities_with_cap(2)
    above_bulk = densities_with_cap(100000)

    # truncation can only drop the density, and must drop it somewhere
    test.assertTrue(np.all(truncated <= uncapped + 1e-5))
    test.assertLess(float(truncated.max()), float(uncapped.max()))
    # a cap above every particle's neighbor count changes nothing
    test.assertEqual(float(np.abs(above_bulk - uncapped).max()), 0.0)


def test_fluid_coincident_particles_separate(test, device):
    """Near-coincident fluid particles must be driven apart, not fuse into a
    stuck "super particle".

    Two particles a micron apart are under-dense (so the compression-only density
    constraint is inactive) and have an undefined pair direction, so only the
    un-averaged minimum-separation repulsion can pull them apart.
    """
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.default_particle_radius = RADIUS
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=1.0e-6,  # the two particles start essentially coincident
        cell_y=SPACING,
        cell_z=SPACING,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=RADIUS,
        flags=FLUID_FLAGS,
    )
    model = builder.finalize(device=device)
    state, _ = _simulate(model, steps=30, iterations=3)
    q = state.particle_q.numpy()
    sep = float(np.linalg.norm(q[0] - q[1]))
    test.assertTrue(np.all(np.isfinite(q)))
    test.assertGreater(sep, 0.25 * SPACING, "coincident fluid particles failed to separate")


def test_fluid_overlapping_worlds_do_not_interact(test, device):
    """Identical fluids in different worlds may occupy the same coordinates.

    If the hash grid ignores ``particle_world``, each particle double-counts its
    twin in the other world and the rest lattice immediately moves. With grouped
    queries, each world sees only itself and remains at rest.
    """
    model = _build_overlapping_world_fluid_grids(device)
    q_initial = model.particle_q.numpy().copy()
    state, _ = _simulate(model, steps=10, iterations=3, fluid_cohesion=0.0, fluid_viscosity=0.0)

    q = state.particle_q.numpy()
    worlds = model.particle_world.numpy()
    q0 = q[worlds == 0]
    q1 = q[worlds == 1]
    q0_initial = q_initial[worlds == 0]

    test.assertTrue(np.isfinite(q).all())
    test.assertEqual(q0.shape, q1.shape)
    test.assertLess(float(np.abs(q0 - q1).max()), 1.0e-6)
    test.assertLess(float(np.abs(q0 - q0_initial).max()), 1.0e-5)


def test_body_velocity_clamp_and_sanitize(test, device):
    """``clamp_body_velocities`` caps dynamic-body linear/angular speed, zeroes
    any non-finite component, and leaves static bodies (inv_mass 0) untouched.
    """
    # row layout: [linear xyz (spatial_top), angular xyz (spatial_bottom)]
    qd_np = np.array(
        [
            [1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0],  # dynamic: huge linear + angular
            [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],  # dynamic: non-finite linear
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],  # static: must be untouched
        ],
        dtype=np.float32,
    )
    qd = wp.array(qd_np, dtype=wp.spatial_vector, device=device)
    inv_mass = wp.array([1.0, 1.0, 0.0], dtype=float, device=device)

    wp.launch(clamp_body_velocities, dim=3, inputs=[inv_mass, 10.0, 20.0], outputs=[qd], device=device)
    out = qd.numpy()

    test.assertTrue(np.all(np.isfinite(out)))
    test.assertAlmostEqual(float(np.linalg.norm(out[0, :3])), 10.0, places=4)  # linear clamped
    test.assertAlmostEqual(float(np.linalg.norm(out[0, 3:])), 20.0, places=4)  # angular clamped
    test.assertEqual(float(np.abs(out[1, :3]).max()), 0.0)  # non-finite -> zero
    test.assertTrue(np.array_equal(out[2], qd_np[2]))  # static body untouched


def test_body_motion_clamp_limits_predicted_pose(test, device):
    """The pre-solve clamp must pull an over-fast predicted pose back within
    the linear and angular distance allowed for one substep.
    """
    dt = 0.1
    q_prev = wp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform, device=device)
    q = wp.array([[100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=wp.transform, device=device)
    qd = wp.array([[1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0]], dtype=wp.spatial_vector, device=device)
    com = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_mass = wp.ones(1, dtype=float, device=device)

    wp.launch(
        clamp_body_motion,
        dim=1,
        inputs=[q_prev, com, inv_mass, 2.0, 4.0, dt],
        outputs=[q, qd],
        device=device,
    )

    q_out = q.numpy()[0]
    qd_out = qd.numpy()[0]
    angle = 2.0 * np.arccos(np.clip(abs(q_out[6]), 0.0, 1.0))
    test.assertAlmostEqual(float(np.linalg.norm(q_out[:3])), 2.0 * dt, places=5)
    test.assertLessEqual(float(angle), 4.0 * dt + 3.0e-3)
    test.assertAlmostEqual(float(np.linalg.norm(qd_out[:3])), 2.0, places=5)
    test.assertAlmostEqual(float(np.linalg.norm(qd_out[3:])), 4.0, places=5)


def test_particle_delta_self_heals_nan(test, device):
    """A non-finite position correction must be reset to the pre-step position
    at rest rather than propagated by :func:`apply_particle_deltas`."""
    x0 = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    xp = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    flags = wp.array([int(FLUID_FLAGS), int(FLUID_FLAGS)], dtype=wp.int32, device=device)
    delta = wp.array([[0.01, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=wp.vec3, device=device)
    x_out = wp.zeros(2, dtype=wp.vec3, device=device)
    v_out = wp.zeros(2, dtype=wp.vec3, device=device)

    wp.launch(
        apply_particle_deltas,
        dim=2,
        inputs=[x0, xp, flags, delta, 1.0 / 60.0, 100.0],
        outputs=[x_out, v_out],
        device=device,
    )
    xo = x_out.numpy()
    vo = v_out.numpy()
    test.assertTrue(np.all(np.isfinite(xo)) and np.all(np.isfinite(vo)))
    # the healed particle returns to its pre-step position at rest
    test.assertTrue(np.allclose(xo[1], [1.0, 0.0, 0.0]))
    test.assertEqual(float(np.abs(vo[1]).max()), 0.0)


def _particle_records(model, state):
    """Full per-particle record (q, qd, mass, radius, flags) for relabel checks."""
    return np.concatenate(
        [
            state.particle_q.numpy(),
            state.particle_qd.numpy(),
            model.particle_mass.numpy()[:, None],
            model.particle_radius.numpy()[:, None],
            model.particle_flags.numpy()[:, None].astype(np.float64),
        ],
        axis=1,
    )


def test_fluid_reorder_is_pure_relabel(test, device):
    """reorder_particles must permute particles without changing the data.

    The set of per-particle records must be bit-identical before and after, the
    order must actually change (a lattice's row-major order differs from Morton
    order), and the q/qd/mass coupling must stay intact.
    """
    model = _build_fluid_grid(device, dims=(5, 5, 5), gravity=-9.81, ground=True, z0=0.3)
    state, solver = _simulate(model, steps=15, ground=True)

    before = _particle_records(model, state)
    solver.reorder_particles(state)
    after = _particle_records(model, state)

    # the multiset of records is unchanged (sort each lexicographically, compare)
    sorted_before = before[np.lexsort(before.T[::-1])]
    sorted_after = after[np.lexsort(after.T[::-1])]
    test.assertEqual(float(np.abs(sorted_before - sorted_after).max()), 0.0)
    # the order genuinely changed (otherwise the test proves nothing)
    test.assertTrue(bool(np.any(np.abs(before - after).max(axis=1) > 0.0)))
    # a step after reorder must still integrate to a finite state
    state_1 = model.state()
    contacts = model.contacts()
    state.clear_forces()
    model.collide(state, contacts)
    solver.step(state, state_1, None, contacts, 1.0 / 240.0)
    test.assertTrue(np.isfinite(state_1.particle_q.numpy()).all())


def test_fluid_reorder_noop_when_not_all_fluid(test, device):
    """reorder_particles must leave non-fluid scenes untouched (it would
    otherwise scramble the index-based topology of cloth/soft bodies)."""
    model = _build_fluid_grid(device, dims=(4, 4, 4), fluid=False)
    solver = newton.solvers.SolverXPBD(model, iterations=2, fluid_rest_distance=SPACING)
    state = model.state()
    before = state.particle_q.numpy().copy()
    solver.reorder_particles(state)
    test.assertEqual(float(np.abs(before - state.particle_q.numpy()).max()), 0.0)


def test_fluid_render_particles_ignore_non_fluid_neighbors(test, device):
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.default_particle_radius = RADIUS
    builder.add_particle(
        pos=(0.0, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=PARTICLE_MASS,
        radius=RADIUS,
        flags=FLUID_FLAGS,
    )
    builder.add_particle(
        pos=(0.5 * SPACING, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=PARTICLE_MASS,
        radius=RADIUS,
        flags=newton.ParticleFlags.ACTIVE,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=SPACING)
    state = model.state()

    solver.update_render_particles(state, smoothing=1.0)
    render_q = solver.render_positions.numpy()
    anisotropy = solver.render_anisotropy.numpy()

    np.testing.assert_allclose(render_q[0], state.particle_q.numpy()[0], atol=1.0e-7)
    test.assertEqual(float(anisotropy[1, 3]), 0.0)


def test_inactive_fluid_flags_do_not_enable_solver(test, device):
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.default_particle_radius = RADIUS
    builder.add_particle(
        pos=(0.0, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=PARTICLE_MASS,
        radius=RADIUS,
        flags=newton.ParticleFlags.FLUID,
    )
    builder.add_particle(
        pos=(SPACING, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=PARTICLE_MASS,
        radius=RADIUS,
        flags=newton.ParticleFlags.ACTIVE,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=SPACING)

    test.assertFalse(solver._has_fluid)
    test.assertFalse(solver._all_fluid)


def test_fluid_cohesion_assignment_updates_derived_coefficient(test, device):
    model = _build_fluid_grid(device, dims=(2, 2, 2))
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=SPACING, fluid_cohesion=1.0)

    solver.fluid_cohesion = 0.25

    test.assertEqual(solver.fluid_cohesion, 0.25)
    test.assertAlmostEqual(solver._fluid_cohesion_step, 0.02 * SPACING * 0.25)


def test_fluid_render_particles_reuse_simulation_hash_grid(test, device):
    model = _build_fluid_grid(device, dims=(3, 3, 3))
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=SPACING)
    grid_id = model.particle_grid.id

    solver.update_render_particles(model.state())

    test.assertEqual(model.particle_grid.id, grid_id)


def test_fluid_render_update_does_not_interfere_with_capture(test, device):
    """Render-grid builds between graph launches must not alter simulation."""

    def run(render_updates):
        model = _build_fluid_grid(device, dims=(4, 4, 4), gravity=-9.81, ground=True, z0=0.2)
        solver = newton.solvers.SolverXPBD(model, iterations=2, fluid_rest_distance=SPACING)
        state_0, state_1 = model.state(), model.state()
        contacts = model.contacts()

        with wp.ScopedCapture(device=device) as capture:
            # Two steps return the persistent state to the same pair of buffers,
            # so each graph launch advances from the previous launch's output.
            for _ in range(2):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, None, contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0

        for _ in range(30):
            wp.capture_launch(capture.graph)
            if render_updates:
                solver.update_render_particles(state_0)
        return state_0.particle_q.numpy()

    reference = run(False)
    rendered = run(True)
    np.testing.assert_allclose(rendered, reference, atol=1.0e-6, rtol=0.0)


def test_diffuse_emission_ignores_non_fluid_particles(test, device):
    q = wp.array([(-0.5 * SPACING, 0.0, 0.0), (0.5 * SPACING, 0.0, 0.0)], dtype=wp.vec3, device=device)
    qd = wp.array([(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    flags = wp.array([int(newton.ParticleFlags.ACTIVE)] * 2, dtype=wp.int32, device=device)
    worlds = wp.zeros(2, dtype=wp.int32, device=device)
    density = wp.zeros(2, dtype=wp.float32, device=device)
    grid = wp.HashGrid(8, 8, 8, device=device)
    grid.reserve(2)
    grid.build(q, radius=2.0 * SPACING, groups=worlds)

    capacity = 8
    diffuse_q = wp.zeros(capacity, dtype=wp.vec4, device=device)
    diffuse_qd = wp.zeros(capacity, dtype=wp.vec4, device=device)
    diffuse_world = wp.zeros(capacity, dtype=wp.int32, device=device)
    slot_states = wp.zeros(capacity, dtype=wp.int32, device=device)
    frame_seed = wp.zeros(1, dtype=wp.int32, device=device)
    spawn_counter = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(
        spawn_fluid_diffuse_particles,
        dim=2,
        inputs=[
            grid.id,
            q,
            qd,
            flags,
            worlds,
            density,
            2.0 * SPACING,
            REST_DENSITY,
            1.0e-3,
            10.0,
            0.0,
            1.0,
            8,
            wp.vec3(-10.0),
            wp.vec3(10.0),
            0.0,
            frame_seed,
            spawn_counter,
            diffuse_q,
            diffuse_qd,
            diffuse_world,
            slot_states,
        ],
        device=device,
    )

    test.assertEqual(float(diffuse_q.numpy()[:, 3].max()), 0.0)


def test_diffuse_slot_reservation_does_not_overwrite_live_particle(test, device):
    slot_states = wp.ones(1, dtype=wp.int32, device=device)
    result = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(_reserve_diffuse_slot_kernel, dim=1, inputs=[0, slot_states], outputs=[result], device=device)

    test.assertEqual(int(result.numpy()[0]), -1)


def test_diffuse_shape_friction_scales_with_timestep(test, device):
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.default_particle_radius = RADIUS
    builder.add_particle(
        pos=(0.0, 0.0, 1.0),
        vel=(0.0, 0.0, 0.0),
        mass=PARTICLE_MASS,
        radius=RADIUS,
        flags=FLUID_FLAGS,
    )
    builder.add_particle(
        pos=(10.0, 0.0, 1.0),
        vel=(0.0, 0.0, 0.0),
        mass=PARTICLE_MASS,
        radius=RADIUS,
        flags=FLUID_FLAGS,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(
        model,
        fluid_rest_distance=SPACING,
        max_diffuse_particles=1,
        diffuse_lifetime=100.0,
        diffuse_drag=0.0,
        diffuse_spawn_probability=0.0,
    )
    state = model.state()
    solver.diffuse_positions.assign([(0.0, 0.0, 0.01, 1.0)])
    solver.diffuse_velocities.assign([(1.0, 0.0, 0.0, 0.0)])
    solver.diffuse_slot_states.fill_(1)

    solver._step_diffuse_particles(state, dt=0.1)
    velocity = solver.diffuse_velocities.numpy()[0, :3]

    test.assertAlmostEqual(float(velocity[0]), 0.98, places=5)
    test.assertAlmostEqual(float(velocity[1]), 0.0, places=6)
    test.assertAlmostEqual(float(velocity[2]), 0.0, places=6)


def test_fluid_hash_grid_is_capture_ready(test, device):
    model = _build_overlapping_world_fluid_grids(device, dims=(3, 3, 3))
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=SPACING)
    state = model.state()

    with wp.ScopedCapture(device=device) as capture:
        solver._build_particle_grid(state.particle_q, solver._fluid_h)
    wp.capture_launch(capture.graph)


devices = get_test_devices()


class TestSolverXPBDFluid(unittest.TestCase):
    pass


for _name in (
    "test_fluid_rest_lattice_stays_at_rest",
    "test_fluid_rest_lattice_exact_without_cohesion",
    "test_fluid_compressed_block_decompresses",
    "test_fluid_drop_forms_cohesive_puddle",
    "test_fluid_density_corrections_do_not_cancel_shape_contacts",
    "test_fluid_pressure_reprojection_conserves_momentum",
    "test_fluid_drop_settles_without_viscosity",
    "test_fluid_pair_coheres_without_oscillation",
    "test_fluid_pairs_skip_contact_constraints",
    "test_fluid_render_particles",
    "test_fluid_render_particle_limit_and_fast_path",
    "test_fluid_diffuse_particles_spawn_and_expire",
    "test_fluid_diffuse_disabled_by_default",
    "test_fluid_sdf_mesh_contains_particles",
    "test_fluid_reorder_is_pure_relabel",
    "test_fluid_reorder_noop_when_not_all_fluid",
    "test_fluid_render_particles_ignore_non_fluid_neighbors",
    "test_inactive_fluid_flags_do_not_enable_solver",
    "test_fluid_cohesion_assignment_updates_derived_coefficient",
    "test_fluid_render_particles_reuse_simulation_hash_grid",
    "test_diffuse_emission_ignores_non_fluid_particles",
    "test_diffuse_slot_reservation_does_not_overwrite_live_particle",
    "test_diffuse_shape_friction_scales_with_timestep",
    "test_fluid_max_neighbors_truncates_density",
    "test_fluid_coincident_particles_separate",
    "test_fluid_overlapping_worlds_do_not_interact",
    "test_body_velocity_clamp_and_sanitize",
    "test_body_motion_clamp_limits_predicted_pose",
    "test_particle_delta_self_heals_nan",
):
    add_function_test(
        TestSolverXPBDFluid,
        _name,
        globals()[_name],
        devices=devices,
        check_output=False,
    )

add_function_test(
    TestSolverXPBDFluid,
    "test_fluid_hash_grid_is_capture_ready",
    test_fluid_hash_grid_is_capture_ready,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestSolverXPBDFluid,
    "test_fluid_render_update_does_not_interfere_with_capture",
    test_fluid_render_update_does_not_interfere_with_capture,
    devices=get_cuda_test_devices(),
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
