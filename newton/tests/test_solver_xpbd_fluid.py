# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the position-based fluid (PBF) support in the XPBD solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

SPACING = 0.05
RADIUS = 0.025
REST_DENSITY = 1000.0
PARTICLE_MASS = REST_DENSITY * SPACING**3
FLUID_FLAGS = newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID


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
    """Poly6 SPH density of each particle, normalized by the solver's rest density."""
    h = solver._fluid_h
    d = np.linalg.norm(q[None, :, :] - q[:, None, :], axis=-1)
    w = np.where(d < h, (h * h - d**2) ** 3, 0.0) * 315.0 / (64.0 * np.pi * h**9)
    return PARTICLE_MASS * w.sum(axis=1) / solver._fluid_rest_density_eff


def test_fluid_rest_lattice_stays_at_rest(test, device):
    """A particle grid at rest spacing without gravity must not gain energy.

    The rest density is calibrated from the lattice, so the density constraint
    should be inactive in the bulk and the cohesion term must stay contractive
    (no oscillation blow-up).
    """
    model = _build_fluid_grid(device)
    q0 = model.particle_q.numpy().copy()
    state, _solver = _simulate(model, 240)

    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    drift = np.linalg.norm(q - q0, axis=1).max()
    max_speed = np.linalg.norm(qd, axis=1).max()
    # surface particles relax slightly under cohesion; the bulk must not move
    test.assertLess(drift, 0.1, f"rest lattice drifted {drift:.4f} m")
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


devices = get_test_devices()


class TestSolverXPBDFluid(unittest.TestCase):
    pass


for _name in (
    "test_fluid_rest_lattice_stays_at_rest",
    "test_fluid_rest_lattice_exact_without_cohesion",
    "test_fluid_compressed_block_decompresses",
    "test_fluid_drop_forms_cohesive_puddle",
    "test_fluid_pair_coheres_without_oscillation",
    "test_fluid_pairs_skip_contact_constraints",
    "test_fluid_render_particles",
):
    add_function_test(
        TestSolverXPBDFluid,
        _name,
        globals()[_name],
        devices=devices,
        check_output=False,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
