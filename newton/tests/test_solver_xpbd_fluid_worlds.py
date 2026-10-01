# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify independent fluid environments, captured rollout, and selective reset."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_FLAGS = newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID


def _model(device, speeds, *, coincident=False, global_particles=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -1.0))
    positions = (
        [(0.0, 0.0, 0.0), (0.000001, 0.0, 0.0)]
        if coincident
        else [(0.036 * x, 0.036 * y, 0.036 * z) for x in range(2) for y in range(2) for z in range(2)]
    )
    for speed in speeds:
        builder.begin_world()
        for x, y, z in positions:
            builder.add_particle(pos=(x, y, z), vel=(speed - y, x, 0.0), mass=1.0, radius=0.025, flags=_FLAGS)
        builder.end_world()
    if global_particles:
        for position in positions:
            builder.add_particle(pos=position, vel=(3.0, 0.0, 0.0), mass=1.0, radius=0.025, flags=_FLAGS)
    return builder.finalize(device=device)


def _solver(model):
    return newton.solvers.SolverXPBD(
        model,
        iterations=2,
        fluid_rest_distance=0.05,
        fluid_cohesion=0.2,
        fluid_viscosity=0.05,
        fluid_vorticity_confinement=0.05,
    )


def _advance(solver, a, b, count):
    # An even substep count returns the result to the original state buffers.
    for _ in range(count):
        solver.step(a, b, None, None, 1.0 / 240.0)
        a, b = b, a
    return a, b


@wp.kernel
def _reset_particles(
    worlds: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    initial_q: wp.array[wp.vec3],
    initial_qd: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
):
    i = wp.tid()
    if mask[worlds[i]]:
        q[i] = initial_q[i]
        qd[i] = initial_qd[i]
        forces[i] = wp.vec3(0.0)


def test_fluid_coincident_worlds_match_single(test, device):
    """Separate coincident particles identically regardless of environment batch index."""
    model = _model(device, [0.0, 0.0, 0.0], coincident=True)
    single = _model(device, [0.0], coincident=True)
    a, _ = _advance(_solver(model), model.state(), model.state(), 4)
    reference, _ = _advance(_solver(single), single.state(), single.state(), 4)
    q, qd = a.particle_q.numpy().reshape(3, 2, 3), a.particle_qd.numpy().reshape(3, 2, 3)
    test.assertGreater(float(np.linalg.norm(q[0, 1] - q[0, 0])), 0.02)
    for world in range(3):
        np.testing.assert_allclose(q[world], reference.particle_q.numpy(), atol=2.0e-6, rtol=2.0e-5)
        np.testing.assert_allclose(qd[world], reference.particle_qd.numpy(), atol=2.0e-5, rtol=2.0e-5)


def test_fluid_overlapping_rollouts_match_independent_worlds(test, device):
    """Match independent dynamics with overlapping, differently moving fluid worlds."""
    speeds = [0.0, 0.3, -0.2]
    model = _model(device, speeds)
    a, _ = _advance(_solver(model), model.state(), model.state(), 12)
    for world, speed in enumerate(speeds):
        single = _model(device, [speed])
        reference, _ = _advance(_solver(single), single.state(), single.state(), 12)
        np.testing.assert_allclose(
            a.particle_q.numpy()[world * 8 : (world + 1) * 8], reference.particle_q.numpy(), atol=2.0e-6, rtol=2.0e-5
        )
        np.testing.assert_allclose(
            a.particle_qd.numpy()[world * 8 : (world + 1) * 8], reference.particle_qd.numpy(), atol=3.0e-5, rtol=3.0e-5
        )


def test_fluid_global_group_does_not_couple_local_world(test, device):
    """Keep global fluid particles separate with both one and several local worlds."""
    reference_model = _model(device, [0.0])
    reference, _ = _advance(_solver(reference_model), reference_model.state(), reference_model.state(), 4)
    for worlds in (1, 2):
        model = _model(device, [0.0] * worlds, global_particles=True)
        state, _ = _advance(_solver(model), model.state(), model.state(), 4)
        np.testing.assert_allclose(state.particle_q.numpy()[:8], reference.particle_q.numpy(), atol=2.0e-6, rtol=2.0e-5)


def _check_selective_reset(test, device, captured):
    model = _model(device, [0.0, 0.3, -0.2])
    solver = _solver(model)
    a, b = model.state(), model.state()
    mask = wp.array([False, True, False, False], dtype=wp.bool, device=device)

    def reset_and_step():
        for state in (a, b):
            wp.launch(
                _reset_particles,
                dim=model.particle_count,
                inputs=[model.particle_world, mask, model.particle_q, model.particle_qd],
                outputs=[state.particle_q, state.particle_qd, state.particle_f],
                device=device,
            )
        solver.reset(a, world_mask=mask)
        _advance(solver, a, b, 4)

    if captured:
        with wp.ScopedCapture(device=device) as capture:
            reset_and_step()

        def advance():
            wp.capture_launch(capture.graph)
    else:
        advance = reset_and_step
    _advance(solver, a, b, 8)
    advance()
    # Reset world restarts at its initial state; other worlds continue uninterrupted.
    for world, speed in enumerate((0.0, 0.3, -0.2)):
        single = _model(device, [speed])
        count = 4 if world == 1 else 12
        reference, _ = _advance(_solver(single), single.state(), single.state(), count)
        np.testing.assert_allclose(
            a.particle_q.numpy()[world * 8 : (world + 1) * 8], reference.particle_q.numpy(), atol=2.0e-6, rtol=2.0e-5
        )
        np.testing.assert_allclose(
            a.particle_qd.numpy()[world * 8 : (world + 1) * 8], reference.particle_qd.numpy(), atol=3.0e-5, rtol=3.0e-5
        )
    # Change the same device mask before graph replay, including an empty reset.
    mask.fill_(False)
    advance()
    test.assertTrue(np.isfinite(a.particle_q.numpy()).all())
    mask.assign([True, True, True, False])
    advance()
    for world, speed in enumerate((0.0, 0.3, -0.2)):
        single = _model(device, [speed])
        reference, _ = _advance(_solver(single), single.state(), single.state(), 4)
        np.testing.assert_allclose(
            a.particle_q.numpy()[world * 8 : (world + 1) * 8], reference.particle_q.numpy(), atol=2.0e-6, rtol=2.0e-5
        )


def test_fluid_selective_reset(test, device):
    """Reset selected fluid worlds without disturbing neighboring rollouts."""
    _check_selective_reset(test, device, False)


def test_fluid_selective_reset_captured(test, device):
    """Replay selective resets from a changing device mask inside a CUDA graph."""
    _check_selective_reset(test, device, True)


def test_fluid_interleaved_graphs_keep_grid_metadata(test, device):
    """Preserve grid ownership and radii when replaying two captured solvers."""
    rollouts = []
    for speeds, radius in (([0.0], 0.08), ([0.0, 0.3, -0.2], 0.11)):
        model = _model(device, speeds)
        # Force step() to build at a radius larger than the fluid kernel support.
        model.particle_cohesion = 0.12
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=2,
            fluid_rest_distance=0.05,
            fluid_smoothing_length=radius,
            fluid_viscosity=0.05,
        )
        a, b = model.state(), model.state()
        _advance(solver, a, b, 12)
        expected_q, expected_qd = a.particle_q.numpy(), a.particle_qd.numpy()
        for state in (a, b):
            wp.copy(state.particle_q, model.particle_q)
            wp.copy(state.particle_qd, model.particle_qd)
        with wp.ScopedCapture(device=device) as capture:
            # Exercise different cell widths within a single graph as well.
            solver._build_particle_grid(a.particle_q, 0.25)
            _advance(solver, a, b, 4)
        rollouts.append((solver, a, b, capture.graph, expected_q, expected_qd))
    for _ in range(3):
        for _, _, _, graph, _, _ in rollouts:
            wp.capture_launch(graph)
    for _, state, _, _, expected_q, expected_qd in rollouts:
        np.testing.assert_allclose(state.particle_q.numpy(), expected_q, atol=2.0e-6, rtol=2.0e-5)
        np.testing.assert_allclose(state.particle_qd.numpy(), expected_qd, atol=3.0e-5, rtol=3.0e-5)


def test_fluid_reorder_preserves_local_global_ranges(test, device):
    """Preserve collision and reset index ranges with local and global particles."""
    model = _model(device, [0.0], global_particles=True)
    state = model.state()
    positions = state.particle_q.numpy()
    positions[:8, 0] += 1.0  # Force global particles to sort before local particles.
    state.particle_q.assign(positions)
    initial_q = state.particle_q.numpy().copy()
    initial_worlds = model.particle_world.numpy().copy()
    _solver(model).reorder_particles(state)
    np.testing.assert_array_equal(model.particle_world.numpy(), initial_worlds)
    np.testing.assert_array_equal(state.particle_q.numpy(), initial_q)


class TestSolverXPBDFluidWorlds(unittest.TestCase):
    pass


for function in (
    test_fluid_coincident_worlds_match_single,
    test_fluid_overlapping_rollouts_match_independent_worlds,
    test_fluid_global_group_does_not_couple_local_world,
    test_fluid_selective_reset,
    test_fluid_reorder_preserves_local_global_ranges,
):
    add_function_test(
        TestSolverXPBDFluidWorlds, function.__name__, function, devices=get_test_devices(), check_output=False
    )
add_function_test(
    TestSolverXPBDFluidWorlds,
    test_fluid_selective_reset_captured.__name__,
    test_fluid_selective_reset_captured,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestSolverXPBDFluidWorlds,
    test_fluid_interleaved_graphs_keep_grid_metadata.__name__,
    test_fluid_interleaved_graphs_keep_grid_metadata,
    devices=get_cuda_test_devices(),
    check_output=False,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
