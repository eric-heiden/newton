# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the XPBD solver.

Includes tests for particle-particle friction using relative velocity correctly.
"""

import math
import unittest
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.xpbd.kernels import apply_rigid_restitution
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_particle_particle_friction_uses_relative_velocity(test, device):
    """
    Test that particle-particle friction correctly uses relative velocity.

    This test verifies the fix for the bug where friction was computed using
    absolute velocity instead of relative velocity:
        WRONG: vt = v - n * vn        (uses absolute velocity v)
        RIGHT: vt = vrel - n * vn     (uses relative velocity vrel)

    Setup:
    - Two particles in contact (overlapping slightly)
    - Both particles moving with the same tangential velocity
    - With friction coefficient > 0

    Expected behavior:
    - Since relative tangential velocity is zero, friction should not
      affect their relative motion
    - Both particles should continue moving together at the same velocity
      (modulo normal contact forces)

    If the bug existed (using absolute velocity), the friction would
    incorrectly compute a non-zero tangential component and try to
    slow down both particles differently.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    # Two particles that are slightly overlapping (in contact)
    # Positioned along X axis, both at y=0, z=0
    particle_radius = 0.5
    overlap = 0.1  # small overlap to ensure contact
    separation = 2.0 * particle_radius - overlap

    pos = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(separation, 0.0, 0.0),
    ]

    # Both particles moving with the same tangential velocity (along Z axis)
    # The contact normal will be along X axis, so Z velocity is tangential
    tangential_velocity = 10.0
    vel = [
        wp.vec3(0.0, 0.0, tangential_velocity),
        wp.vec3(0.0, 0.0, tangential_velocity),
    ]

    mass = [1.0, 1.0]
    radius = [particle_radius, particle_radius]

    builder.add_particles(pos=pos, vel=vel, mass=mass, radius=radius)

    model = builder.finalize(device=device)

    # Disable gravity so we only see friction effects
    model.set_gravity((0.0, 0.0, 0.0))

    # Set particle-particle friction coefficient (XPBD particle-particle contact uses model.particle_mu)
    model.particle_mu = 1.0  # high friction
    model.particle_cohesion = 0.0

    # Use XPBD solver which uses the solve_particle_particle_contacts kernel
    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=20,
    )

    state0 = model.state()
    state1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    # Apply equal and opposite forces to keep the particles in sustained contact.
    # Without this, the initial overlap may be resolved in ~1 iteration and friction becomes hard to observe,
    # making the test flaky across devices/precision.
    press_force = 50.0
    assert state0.particle_f is not None
    state0.particle_f.assign(
        wp.array(
            [
                wp.vec3(wp.float32(press_force), wp.float32(0.0), wp.float32(0.0)),
                wp.vec3(wp.float32(-press_force), wp.float32(0.0), wp.float32(0.0)),
            ],
            dtype=wp.vec3,
            device=device,
        )
    )

    dt = 1.0 / 60.0
    num_steps = 60

    # Store initial relative velocity
    initial_vel = state0.particle_qd.numpy().copy()
    initial_relative_z_vel = initial_vel[0, 2] - initial_vel[1, 2]

    # Run simulation
    for _ in range(num_steps):
        collision_pipeline.collide(state0, contacts)
        control = model.control()
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    # Get final velocities
    final_vel = state0.particle_qd.numpy()
    final_relative_z_vel = final_vel[0, 2] - final_vel[1, 2]

    # The key assertion: relative tangential velocity should remain near zero
    # since both particles started with the same tangential velocity
    test.assertAlmostEqual(
        initial_relative_z_vel,
        0.0,
        places=5,
        msg="Initial relative tangential velocity should be zero",
    )
    test.assertAlmostEqual(
        final_relative_z_vel,
        0.0,
        places=3,
        msg="Final relative tangential velocity should remain near zero "
        "(friction should not affect particles moving together)",
    )

    # Also verify both particles still have similar Z velocities
    # (they should move together, not be affected differently by friction)
    test.assertAlmostEqual(
        final_vel[0, 2],
        final_vel[1, 2],
        places=3,
        msg="Both particles should have the same tangential velocity after simulation",
    )


def test_distance_joint_limits(test, device):
    """Enforce distance-joint bounds from separated and coincident anchors."""

    def solve(initial_distance, min_distance, max_distance):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(
            xform=wp.transform(wp.vec3(initial_distance, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_shape_sphere(body, radius=0.1)
        joint = builder.add_joint_distance(
            -1,
            body,
            parent_xform=wp.transform(
                wp.vec3(0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            min_distance=min_distance,
            max_distance=max_distance,
        )
        builder.add_articulation([joint])

        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverXPBD(model, iterations=10)
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        return state_out.body_q.numpy()[body, :3]

    test.assertGreaterEqual(np.linalg.norm(solve(0.25, 1.0, -1.0)), 0.99)
    np.testing.assert_allclose(solve(0.0, 1.0, -1.0), (0.0, 1.0, 0.0), atol=0.01)
    test.assertLessEqual(np.linalg.norm(solve(2.0, -1.0, 1.0)), 1.01)


def test_ball_joint_recovers_from_large_anchor_separation(test, device):
    """Recover a ball joint from a large off-axis anchor separation."""
    capsule_radius = 0.0625
    capsule_half_height = 0.25
    capsule_half_extent = capsule_radius + capsule_half_height

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    shape_xform = wp.transform(
        wp.vec3(0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi),
    )

    parent = builder.add_link()
    builder.add_shape_capsule(
        parent,
        xform=shape_xform,
        radius=capsule_radius,
        half_height=capsule_half_height,
    )
    child = builder.add_link(xform=wp.transform(wp.vec3(2.0 * capsule_half_extent, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_capsule(
        child,
        xform=shape_xform,
        radius=capsule_radius,
        half_height=capsule_half_height,
    )

    root_joint = builder.add_joint_free(child=parent)
    ball_joint = builder.add_joint_ball(
        parent=parent,
        child=child,
        parent_xform=wp.transform(wp.vec3(capsule_half_extent, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-capsule_half_extent, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root_joint, ball_joint])

    model = builder.finalize(device=device)
    state0 = model.state()
    state1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    body_q = state0.body_q.numpy()
    body_q[child, :3] += np.array((1.0, 1.0, 0.0), dtype=np.float32)
    state0.body_q.assign(body_q)

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    solver.step(state0, state1, None, None, 1.0 / 240.0)

    body_q = state1.body_q.numpy()

    parent_anchor = wp.transform_point(
        wp.transform(*body_q[parent]),
        wp.vec3(capsule_half_extent, 0.0, 0.0),
    )
    child_anchor = wp.transform_point(
        wp.transform(*body_q[child]),
        wp.vec3(-capsule_half_extent, 0.0, 0.0),
    )
    anchor_gap = float(wp.length(child_anchor - parent_anchor))

    test.assertLess(
        anchor_gap,
        0.5,
        msg=f"Ball joint did not recover from a large anchor separation: gap={anchor_gap:.6g} m",
    )


def test_prismatic_joint_recovers_from_large_transverse_separation(test, device):
    """Recover transverse errors while retaining a valid prismatic coordinate."""
    capsule_radius = 0.0625
    capsule_half_height = 0.25

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    shape_xform = wp.transform(
        wp.vec3(0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi),
    )

    parent = builder.add_link()
    builder.add_shape_capsule(
        parent,
        xform=shape_xform,
        radius=capsule_radius,
        half_height=capsule_half_height,
    )
    child = builder.add_link()
    builder.add_shape_capsule(
        child,
        xform=shape_xform,
        radius=capsule_radius,
        half_height=capsule_half_height,
    )

    root_joint = builder.add_joint_free(child=parent)
    prismatic_joint = builder.add_joint_prismatic(
        parent=parent,
        child=child,
        axis=newton.Axis.X,
        limit_lower=-2.0,
        limit_upper=2.0,
    )
    builder.add_articulation([root_joint, prismatic_joint])

    model = builder.finalize(device=device)
    state0 = model.state()
    state1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    body_q = state0.body_q.numpy()
    body_q[child, :3] += np.array((0.5, 1.0, 1.0), dtype=np.float32)
    state0.body_q.assign(body_q)

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    solver.step(state0, state1, None, None, 1.0 / 240.0)

    body_q = state1.body_q.numpy()
    relative_offset = wp.transform_point(
        wp.transform_inverse(wp.transform(*body_q[parent])),
        wp.vec3(*body_q[child, :3]),
    )
    joint_position = float(relative_offset[0])
    transverse_gap = float(np.hypot(relative_offset[1], relative_offset[2]))

    test.assertLess(
        transverse_gap,
        0.5,
        msg=f"Prismatic joint did not recover from transverse separation: gap={transverse_gap:.6g} m",
    )
    test.assertGreaterEqual(
        joint_position,
        -2.0,
        msg=f"Prismatic joint violated its lower limit: position={joint_position:.6g} m",
    )
    test.assertLessEqual(
        joint_position,
        2.0,
        msg=f"Prismatic joint violated its upper limit: position={joint_position:.6g} m",
    )


def test_prismatic_joint_retains_extension_in_parent_moment_arm(test, device):
    """Retain valid prismatic extension in the parent moment arm."""
    extension = 0.5
    transverse_error = 0.1

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link()
    builder.add_shape_sphere(parent, radius=0.25)
    child = builder.add_link()
    builder.add_shape_sphere(child, radius=0.25)

    root_joint = builder.add_joint_free(child=parent)
    prismatic_joint = builder.add_joint_prismatic(
        parent=parent,
        child=child,
        axis=newton.Axis.X,
        limit_lower=-2.0,
        limit_upper=2.0,
        damping=0.0,
    )
    builder.add_articulation([root_joint, prismatic_joint])

    model = builder.finalize(device=device)
    state0 = model.state()
    state1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    body_q = state0.body_q.numpy()
    body_q[child, :2] += np.array((extension, transverse_error), dtype=np.float32)
    state0.body_q.assign(body_q)

    solver = newton.solvers.SolverXPBD(model, iterations=2, angular_damping=0.0)
    solver.step(state0, state1, None, None, 1.0 / 240.0)

    parent_rotation_z = abs(float(state1.body_q.numpy()[parent, 5]))
    test.assertGreater(
        parent_rotation_z,
        0.01,
        msg="Prismatic correction omitted torque from the valid joint extension",
    )


def test_optional_control_and_contacts(test, device):
    """Test that XPBD accepts omitted control and contact data.

    The ground-plane shape catches attempts to access a missing contact buffer,
    while the falling particle verifies that non-contact integration still runs.
    """
    builder = newton.ModelBuilder(up_axis="Y")
    builder.add_particle(pos=(0.0, 1.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model)
    state_in = model.state()
    state_out = model.state()

    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 60.0)

    test.assertLess(float(state_out.particle_q.numpy()[0, 1]), 1.0)


def test_particle_particle_friction_with_relative_motion(test, device):
    """
    Test that friction DOES affect particles with different tangential velocities.

    This is the complementary test - when particles have different tangential
    velocities, friction should work to equalize them.

    Notes on test design:
    - Particle-particle friction in XPBD is applied during constraint projection while particles are in contact.
      If particles are not kept in sustained contact, you may only get a single contact correction and the
      effect of friction can be near-zero and noisy.
    - To make this robust, we apply equal-and-opposite forces along the contact normal so the particles stay
      pressed together while sliding tangentially, and we compare against a mu=0 baseline.
    """
    # Keep this test to a single time step with guaranteed initial penetration.
    # XPBD's particle-particle friction term is limited by the *incremental* normal correction (penetration error),
    # so once the overlap is resolved to touching, friction can become effectively zero. A long multi-step
    # "relative velocity must decrease" assertion is therefore inherently flaky.

    particle_radius = 0.5
    overlap = 0.1
    separation = 2.0 * particle_radius - overlap

    dt = 1.0 / 30.0  # larger dt to make frictional slip correction clearly measurable

    def run(mu: float) -> float:
        builder = newton.ModelBuilder(up_axis="Y")

        pos = [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(separation, 0.0, 0.0),
        ]

        # Different tangential velocities along Z (tangent to the X-axis contact normal).
        vel = [
            wp.vec3(0.0, 0.0, 10.0),
            wp.vec3(0.0, 0.0, 0.0),
        ]

        mass = [1.0, 1.0]
        radius = [particle_radius, particle_radius]

        builder.add_particles(pos=pos, vel=vel, mass=mass, radius=radius)

        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))
        model.particle_mu = mu
        model.particle_cohesion = 0.0

        solver = newton.solvers.SolverXPBD(model=model, iterations=30)

        state0 = model.state()
        state1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        # One step: measure tangential slip (relative z displacement).
        collision_pipeline.collide(state0, contacts)
        control = model.control()
        solver.step(state0, state1, control, contacts, dt)

        q1 = state1.particle_q.numpy()
        return float(abs(q1[0, 2] - q1[1, 2]))

    slip_no_friction = run(mu=0.0)
    slip_with_friction = run(mu=1.0)

    # With mu=0, slip should be close to v_rel * dt (~10 * dt).
    test.assertGreater(
        slip_no_friction,
        0.2,
        msg="With mu=0, relative tangential slip over one step should be significant",
    )
    test.assertLess(
        slip_with_friction,
        slip_no_friction * 0.95,
        msg="With mu>0, particle-particle friction should reduce tangential slip over one step vs mu=0 baseline",
    )


def test_xpbd_particle_particle_contact_nan_guard(test, device):
    builder = newton.ModelBuilder(up_axis="Y")

    particle_radius = 0.5
    builder.add_particles(
        pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
        vel=[wp.vec3(0.0), wp.vec3(0.0)],
        mass=[1.0, 1.0],
        radius=[particle_radius, particle_radius],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.particle_mu = 1.0
    model.particle_cohesion = 0.0

    solver = newton.solvers.SolverXPBD(model=model, iterations=1)
    state0 = model.state()
    state1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    solver.step(state0, state1, model.control(), contacts, 1.0 / 60.0)

    test.assertTrue(
        np.all(np.isfinite(state1.particle_q.numpy())),
        msg="Exact-overlap particle contact must not write non-finite particle positions.",
    )
    test.assertTrue(
        np.all(np.isfinite(state1.particle_qd.numpy())),
        msg="Exact-overlap particle contact must not write non-finite particle velocities.",
    )


def test_xpbd_particle_particle_tiny_separation_contact_remains_active(test, device):
    builder = newton.ModelBuilder(up_axis="Y")

    particle_radius = 0.5
    separation = 5.0e-9
    builder.add_particles(
        pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(separation, 0.0, 0.0)],
        vel=[wp.vec3(0.0), wp.vec3(0.0)],
        mass=[1.0, 1.0],
        radius=[particle_radius, particle_radius],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.particle_mu = 1.0
    model.particle_cohesion = 0.0

    solver = newton.solvers.SolverXPBD(model=model, iterations=1)
    state0 = model.state()
    state1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    solver.step(state0, state1, model.control(), contacts, 1.0 / 60.0)

    particle_q = state1.particle_q.numpy()
    final_separation = float(np.linalg.norm(particle_q[1] - particle_q[0]))

    test.assertTrue(
        np.all(np.isfinite(particle_q)),
        msg="Tiny nonzero-separation particle contact must not write non-finite particle positions.",
    )
    test.assertTrue(
        np.all(np.isfinite(state1.particle_qd.numpy())),
        msg="Tiny nonzero-separation particle contact must not write non-finite particle velocities.",
    )
    test.assertGreater(
        final_separation,
        0.1,
        msg="Tiny but nonzero particle separation has a valid normal and should keep the contact active.",
    )


def test_particle_shape_restitution_correct_particle(test, device):
    """
    Regression test for the bug where apply_particle_shape_restitution wrote
    restitution velocity to particle_v_out[tid] (contact index) instead of
    particle_v_out[particle_index].

    Setup:
    - Particle 0 ("decoy"): high above the ground (y=10), zero velocity, no contact.
    - Particle 1 ("bouncer"): at the ground surface with downward velocity, will contact.
    - The first contact has tid=0 but contact_particle[0] = 1.
    - With the old bug, restitution dv was written to particle 0 (the decoy).
    - After fix, restitution dv is written to particle 1 (the bouncer).

    Assert: particle 1's y-velocity should be positive (bouncing up) and
    particle 0's y-velocity should remain near zero.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    particle_radius = 0.1

    # Particle 0: decoy, far above the ground — should never contact
    builder.add_particle(pos=(0.0, 10.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=particle_radius)

    # Particle 1: at ground level with downward velocity — will contact
    builder.add_particle(pos=(0.0, particle_radius, 0.0), vel=(0.0, -5.0, 0.0), mass=1.0, radius=particle_radius)

    # Add a ground plane so particle 1 can bounce
    builder.add_ground_plane()

    model = builder.finalize(device=device)

    # Disable gravity so decoy particle stays at rest
    model.set_gravity((0.0, 0.0, 0.0))

    # Enable restitution
    model.soft_contact_restitution = 1.0

    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=10,
        enable_restitution=True,
    )

    state0 = model.state()
    state1 = model.state()

    dt = 1.0 / 60.0

    # Run a single step — enough for the contact + restitution pass
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state0, contacts)
    control = model.control()
    solver.step(state0, state1, control, contacts, dt)

    vel = state1.particle_qd.numpy()

    # Particle 0 (decoy, no contact): y-velocity should be ~0
    test.assertAlmostEqual(
        float(vel[0, 1]),
        0.0,
        places=2,
        msg="Decoy particle (no contact) should have zero y-velocity; restitution was incorrectly applied to it",
    )

    # Particle 1 (bouncer): y-velocity should be positive (bouncing up)
    test.assertGreater(
        float(vel[1, 1]),
        0.0,
        msg="Bouncing particle should have positive y-velocity after restitution",
    )


def test_particle_shape_restitution_accounts_for_body_velocity(test, device):
    """
    Regression test for the bug where apply_particle_shape_restitution
    did not account for the rigid body velocity at the contact point when
    computing relative velocity for restitution (#1273).

    Setup:
    - A rigid box moving upward at 5 m/s.
    - A stationary particle sitting just above the top face of the box.
    - Restitution = 1.0, gravity disabled.

    Without the fix, the kernel computes relative velocity from the
    particle velocity alone (ignoring the approaching body), so the
    approaching normal velocity appears zero and no restitution impulse
    is applied — the particle stays nearly at rest.

    With the fix, the kernel correctly subtracts the body velocity at
    the contact point, detects the closing velocity, and applies a
    restitution impulse that launches the particle upward.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    # Add a dynamic rigid box centered at origin
    body_id = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_shape_box(body=body_id, hx=1.0, hy=0.5, hz=1.0)

    # Add a stationary particle just above the box's top face (y=0.5)
    particle_radius = 0.1
    builder.add_particle(
        pos=(0.0, 0.5 + particle_radius, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=1.0,
        radius=particle_radius,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_restitution = 1.0

    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=10,
        enable_restitution=True,
    )

    state0 = model.state()
    state1 = model.state()

    # Give the rigid body an upward velocity so it approaches the particle
    body_vel = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    state0.body_qd.assign(wp.array(body_vel, dtype=wp.spatial_vector, device=device))

    dt = 1.0 / 60.0
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state0, contacts)
    control = model.control()
    solver.step(state0, state1, control, contacts, dt)

    vel = state1.particle_qd.numpy()

    # Without the fix, the position solver alone gives the particle ~5 m/s.
    # With the fix, restitution adds another ~5 m/s on top (elastic bounce
    # against a body moving at 5 m/s), yielding ~10 m/s total.
    test.assertGreater(
        float(vel[0, 1]),
        7.0,
        msg=f"Particle should receive restitution impulse from the moving body (expected ~10 m/s, got {float(vel[0, 1]):.2f})",
    )


def test_rigid_restitution_surface_gate_does_not_double_count_thickness(test, device):
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    body_qd_prev = wp.array([wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device)
    body_qd = wp.array([wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device)
    body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    body_m_inv = wp.array([1.0], dtype=float, device=device)
    body_I_inv = wp.array(
        [wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)],
        dtype=wp.mat33,
        device=device,
    )
    body_world = wp.array([0], dtype=wp.int32, device=device)

    shape_body = wp.array([0], dtype=wp.int32, device=device)
    contact_count = wp.array([1], dtype=wp.int32, device=device)
    contact_normal = wp.array([wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device=device)
    contact_shape0 = wp.array([0], dtype=wp.int32, device=device)
    contact_shape1 = wp.array([-1], dtype=wp.int32, device=device)
    restitution = wp.array([1.0], dtype=float, device=device)

    contact_point0 = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    contact_offset0 = wp.array([wp.vec3(0.0, 0.05, 0.0)], dtype=wp.vec3, device=device)
    contact_point1 = wp.array([wp.vec3(0.0, 0.06, 0.0)], dtype=wp.vec3, device=device)
    contact_offset1 = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    contact_inv_weight = wp.array([1.0], dtype=float, device=device)
    gravity = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    deltas = wp.zeros(1, dtype=wp.spatial_vector, device=device)

    wp.launch(
        apply_rigid_restitution,
        dim=1,
        inputs=[
            body_q,
            body_qd,
            body_q,
            body_qd_prev,
            body_com,
            body_m_inv,
            body_I_inv,
            body_world,
            shape_body,
            contact_count,
            contact_normal,
            contact_shape0,
            contact_shape1,
            restitution,
            contact_point0,
            contact_point1,
            contact_offset0,
            contact_offset1,
            contact_inv_weight,
            gravity,
            1.0 / 60.0,
        ],
        outputs=[deltas],
        device=device,
    )

    np.testing.assert_allclose(deltas.numpy()[0], np.zeros(6), atol=1.0e-6)


def test_articulation_contact_drift(test, device):
    """
    Regression test for articulated bodies drifting laterally on the ground (#2030).

    When joints are solved before contacts in the XPBD iteration loop, joint
    corrections displace bodies laterally and contact friction can't fully
    counteract the displacement. Over many steps, the residual accumulates
    into visible sliding.

    Setup:
    - Load a quadruped URDF on its side on the ground plane.
    - Let it settle for 2 seconds, then simulate for 3 more seconds.
    - Check that the root body hasn't drifted laterally.
    """
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.armature = 0.01
    builder.default_joint_cfg.target_ke = 2000.0
    builder.default_joint_cfg.target_kd = 1.0
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e2
    builder.default_shape_cfg.mu = 1.0

    # Place the quadruped on its side (rotated 90 degrees around X axis)
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.PI * 0.5)
    builder.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), rot),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
    )
    armature_inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.01
    for i in range(builder.body_count):
        builder.body_inertia[i] = builder.body_inertia[i] + armature_inertia

    builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
    builder.joint_target_q[-12:] = builder.joint_q[-12:]
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    fps = 100
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # Let the quadruped settle after drop (2 seconds)
    for _ in range(200):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    initial_x = float(body_q[0][0])
    initial_y = float(body_q[0][1])

    # Simulate for 3 more seconds
    for _ in range(300):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    final_x = float(body_q[0][0])
    final_y = float(body_q[0][1])

    drift_x = abs(final_x - initial_x)
    drift_y = abs(final_y - initial_y)
    drift_xy = float(np.hypot(drift_x, drift_y))

    # The root body should not drift more than 1 cm laterally over 3 seconds
    # (Z is up, so X and Y are the lateral axes)
    # Without the fix, Y drifts ~5.9 mm/s → ~1.8 cm over 3 seconds.
    max_drift = 0.01
    test.assertLess(
        drift_xy,
        max_drift,
        msg=(
            f"Root body drifted {drift_xy:.4f} m laterally over 3 seconds "
            f"(dx={drift_x:.4f}, dy={drift_y:.4f}, max allowed: {max_drift})"
        ),
    )


def test_xpbd_contact_force_static_equilibrium(test, device):
    """Steady-state contact-force regression suite for XPBD.

    Four scenarios run together in a single model so they share one settle phase
    and one averaging window. Each scenario is placed far apart on the X axis so
    contact pairs never mix between scenarios:

    - small sphere on plane (Fz = -mg)
    - heavy sphere on plane (Fz = -mg, mass-independent)
    - box on plane (4 corner contacts; summed Fz = -mg, regression for the
      ``rigid_contact_con_weighting`` N*mg inflation bug)
    - mini pyramid (two bottom cubes + one top cube; ground reaction on each
      bottom cube = own weight + half the top cube ≈ 1.5*mg)
    """
    gravity = 9.81

    sphere_radius = 0.25
    sphere_density = 1000.0
    sphere_mass = sphere_density * (4.0 / 3.0) * np.pi * sphere_radius**3

    heavy_radius = 0.5
    heavy_density = 2000.0
    heavy_mass = heavy_density * (4.0 / 3.0) * np.pi * heavy_radius**3

    box_h = 0.5
    box_density = 1000.0
    box_mass = box_density * (2.0 * box_h) ** 3

    cube_h = 0.5
    cube_density = 1000.0
    cube_mass = cube_density * (2.0 * cube_h) ** 3
    cube_mg = cube_mass * gravity

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    ground_shape = 0

    builder.default_shape_cfg.density = sphere_density
    sphere_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_radius), wp.quat_identity()))
    builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)

    builder.default_shape_cfg.density = heavy_density
    heavy_body = builder.add_body(xform=wp.transform(wp.vec3(10.0, 0.0, heavy_radius), wp.quat_identity()))
    builder.add_shape_sphere(body=heavy_body, radius=heavy_radius)

    builder.default_shape_cfg.density = box_density
    box_body = builder.add_body(xform=wp.transform(wp.vec3(20.0, 0.0, box_h), wp.quat_identity()))
    builder.add_shape_box(body=box_body, hx=box_h, hy=box_h, hz=box_h)

    builder.default_shape_cfg.density = cube_density
    pyramid_x = 30.0
    cube_left_body = builder.add_body(xform=wp.transform(wp.vec3(pyramid_x - cube_h, 0.0, cube_h), wp.quat_identity()))
    builder.add_shape_box(body=cube_left_body, hx=cube_h, hy=cube_h, hz=cube_h)
    cube_right_body = builder.add_body(xform=wp.transform(wp.vec3(pyramid_x + cube_h, 0.0, cube_h), wp.quat_identity()))
    builder.add_shape_box(body=cube_right_body, hx=cube_h, hy=cube_h, hz=cube_h)
    cube_top_body = builder.add_body(xform=wp.transform(wp.vec3(pyramid_x, 0.0, 3.0 * cube_h), wp.quat_identity()))
    builder.add_shape_box(body=cube_top_body, hx=cube_h, hy=cube_h, hz=cube_h)

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=32, rigid_contact_con_weighting=True)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    num_substeps = 8
    sub_dt = dt / num_substeps
    settle_steps = 200  # max needed across scenarios (pyramid stack)
    avg_steps = 60

    for _ in range(settle_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in

    shape_body_np = model.shape_body.numpy()

    sphere_force = np.zeros(3)
    heavy_force = np.zeros(3)
    box_force = np.zeros(3)
    cube_left_fz_on_body = 0.0
    cube_right_fz_on_body = 0.0

    for _ in range(avg_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)

        nc = int(contacts.rigid_contact_count.numpy()[0])
        if nc == 0:
            continue
        forces = contacts.force.numpy()[:nc, :3]
        s0 = contacts.rigid_contact_shape0.numpy()[:nc]
        s1 = contacts.rigid_contact_shape1.numpy()[:nc]

        box_step_count = 0
        for ci in range(nc):
            # ``contacts.force`` is force on body0 by body1. Sum into a "force-on-ground"
            # bucket regardless of which side ground was recorded as: flip sign when
            # ground is shape1 so the final values consistently match -mg downward.
            if s0[ci] == ground_shape:
                other_shape = s1[ci]
                f = forces[ci]
            elif s1[ci] == ground_shape:
                other_shape = s0[ci]
                f = -forces[ci]
            else:
                continue  # body-body contact (top cube against bottom cubes); not asserted
            if other_shape < 0:
                continue
            other_body = shape_body_np[other_shape]
            if other_body == sphere_body:
                sphere_force += f
            elif other_body == heavy_body:
                heavy_force += f
            elif other_body == box_body:
                box_force += f
                box_step_count += 1
            elif other_body == cube_left_body:
                cube_left_fz_on_body += -f[2]
            elif other_body == cube_right_body:
                cube_right_fz_on_body += -f[2]

        test.assertGreater(box_step_count, 1, "Box should generate multiple ground contact points")

    sphere_force /= avg_steps
    heavy_force /= avg_steps
    box_force /= avg_steps
    cube_left_fz_on_body /= avg_steps
    cube_right_fz_on_body /= avg_steps

    np.testing.assert_allclose(
        sphere_force[2],
        -sphere_mass * gravity,
        rtol=0.05,
        err_msg="Sphere on plane: vertical contact force should match -mg",
    )
    np.testing.assert_allclose(
        sphere_force[0], 0.0, atol=0.5, err_msg="Sphere on plane: horizontal X force should be ~0"
    )
    np.testing.assert_allclose(
        sphere_force[1], 0.0, atol=0.5, err_msg="Sphere on plane: horizontal Y force should be ~0"
    )

    np.testing.assert_allclose(
        heavy_force[2],
        -heavy_mass * gravity,
        rtol=0.05,
        err_msg="Heavy sphere on plane: vertical contact force should match -mg",
    )
    np.testing.assert_allclose(
        heavy_force[0], 0.0, atol=0.5, err_msg="Heavy sphere on plane: horizontal X force should be ~0"
    )
    np.testing.assert_allclose(
        heavy_force[1], 0.0, atol=0.5, err_msg="Heavy sphere on plane: horizontal Y force should be ~0"
    )

    np.testing.assert_allclose(
        box_force[2],
        -box_mass * gravity,
        rtol=0.10,
        err_msg="Box on plane: total vertical contact force over multiple contacts should match -mg, not N*mg",
    )
    np.testing.assert_allclose(box_force[0], 0.0, atol=1.0, err_msg="Box on plane: horizontal X force should be ~0")
    np.testing.assert_allclose(box_force[1], 0.0, atol=1.0, err_msg="Box on plane: horizontal Y force should be ~0")

    np.testing.assert_allclose(
        cube_left_fz_on_body,
        1.5 * cube_mg,
        rtol=0.15,
        err_msg=f"Pyramid: ground reaction on left bottom cube should be ~1.5*mg={1.5 * cube_mg:.0f}, got {cube_left_fz_on_body:.0f}",
    )
    np.testing.assert_allclose(
        cube_right_fz_on_body,
        1.5 * cube_mg,
        rtol=0.15,
        err_msg=f"Pyramid: ground reaction on right bottom cube should be ~1.5*mg={1.5 * cube_mg:.0f}, got {cube_right_fz_on_body:.0f}",
    )


def test_xpbd_contact_force_zero_when_no_contact(test, device):
    """A sphere in free-fall (no ground) should produce zero contact force."""
    radius = 0.25

    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=radius)
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    state_in.clear_forces()
    collision_pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, dt)
    solver.update_contacts(contacts, state_out)

    ncontacts = int(contacts.rigid_contact_count.numpy()[0])
    if ncontacts > 0:
        forces = contacts.force.numpy()[:ncontacts]
        np.testing.assert_allclose(forces, 0.0, atol=1e-6, err_msg="No contact force expected in free-fall")


def test_xpbd_contact_force_zero_when_not_touching(test, device):
    """A sphere near a ground plane with a large gap: contact pair exists but force is zero."""
    radius = 0.25
    gap = 1.0
    # Place sphere so it's within the gap (contact pair generated) but not penetrating.
    # Ground is at z=0, sphere center at z = radius + 0.5*gap (well above surface).
    z = radius + 0.5 * gap

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = gap
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=radius)
    model = builder.finalize(device=device)
    model.set_gravity(wp.vec3(0.0, 0.0, 0.0))
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    state_in.clear_forces()
    collision_pipeline.collide(state_in, contacts)

    ncontacts = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(ncontacts, 0, "Gap should cause a contact pair to be generated")

    solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)
    solver.update_contacts(contacts, state_out)

    forces = contacts.force.numpy()[:ncontacts, :3]
    np.testing.assert_allclose(
        forces,
        0.0,
        atol=1e-6,
        err_msg="Contact pair within gap but not touching should report zero force",
    )


def test_xpbd_update_contacts_requires_force_attribute(test, device):
    """update_contacts should raise ValueError when contacts.force is not allocated."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=0.25)
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    state_in.clear_forces()
    collision_pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)

    test.assertIsNone(contacts.force)
    with test.assertRaises(ValueError):
        solver.update_contacts(contacts)


def _build_single_body_pendulum(joint_kind: str, parent_kinematic: bool, gravity: float):
    """Build a single dynamic body suspended from world (or a kinematic body).

    Returns ``(model, child_body_index)``.  The child body is offset 1 m below
    the joint anchor along -Z, so steady-state requires the joint to support
    its weight along +Z.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity), up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    if parent_kinematic:
        parent_link = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_box(parent_link, hx=0.05, hy=0.05, hz=0.05)
        # Replace the default DYNAMIC flag with KINEMATIC.
        builder.body_flags[parent_link] = int(newton.BodyFlags.KINEMATIC)
    else:
        parent_link = -1

    child_link = builder.add_link()
    builder.add_shape_box(child_link, hx=0.1, hy=0.1, hz=0.1)

    parent_xform = wp.transform_identity()
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())

    if joint_kind == "revolute":
        joint = builder.add_joint_revolute(
            parent_link,
            child_link,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=wp.vec3(0.0, 1.0, 0.0),
        )
    elif joint_kind == "ball":
        joint = builder.add_joint_ball(
            parent_link,
            child_link,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    elif joint_kind == "fixed":
        joint = builder.add_joint_fixed(
            parent_link,
            child_link,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    else:
        raise ValueError(f"Unsupported joint kind: {joint_kind}")

    builder.add_articulation([joint])
    return builder, child_link


def _run_single_body_steady_state(test, device, joint_kind: str, parent_kinematic: bool):
    """Settle a single-body pendulum and return the time-averaged parent force."""
    gravity = 9.81
    builder, child_link = _build_single_body_pendulum(joint_kind, parent_kinematic, gravity)
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverXPBD(model, iterations=8)
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    test.assertIsNotNone(state_in.body_parent_f)

    dt = 1.0 / 60.0
    num_substeps = 8
    sub_dt = dt / num_substeps
    settle_steps = 60
    avg_steps = 30

    for _ in range(settle_steps):
        for _ in range(num_substeps):
            solver.step(state_in, state_out, None, None, sub_dt)
            state_in, state_out = state_out, state_in

    parent_f_avg = np.zeros(6)
    for _ in range(avg_steps):
        for _ in range(num_substeps):
            solver.step(state_in, state_out, None, None, sub_dt)
            state_in, state_out = state_out, state_in
        parent_f_avg += state_in.body_parent_f.numpy()[child_link]
    parent_f_avg /= avg_steps

    weight = float(model.body_mass.numpy()[child_link]) * gravity
    return parent_f_avg, weight


def _assert_simple_decoupled_pendulum(test, parent_f_avg, weight, label):
    """Tight assertions for the simple decoupled single-body case.

    With one dynamic body and a kinematic / world parent, XPBD's joint solve
    is fully decoupled and converges essentially exactly.  Only a small
    first-order integration bias remains (~0.05% of ``m*g``).  We assert a
    1% tolerance on the vertical reaction and small absolute tolerances on
    the orthogonal components.
    """
    np.testing.assert_allclose(
        parent_f_avg[2],
        weight,
        rtol=0.01,
        err_msg=f"{label}: vertical parent force should match m*g",
    )
    np.testing.assert_allclose(
        parent_f_avg[:2], 0.0, atol=0.1, err_msg=f"{label}: horizontal parent force should be ~0"
    )
    np.testing.assert_allclose(
        parent_f_avg[3:6], 0.0, atol=0.1, err_msg=f"{label}: parent torque about COM should be ~0"
    )


def test_xpbd_parent_force_revolute_to_world(test, device):
    """Single body on a revolute joint to world should give ~exact m*g reaction."""
    parent_f_avg, weight = _run_single_body_steady_state(test, device, joint_kind="revolute", parent_kinematic=False)
    _assert_simple_decoupled_pendulum(test, parent_f_avg, weight, "revolute-to-world")


def test_xpbd_parent_force_revolute_to_kinematic(test, device):
    """Single body on a revolute joint to a kinematic parent should give ~exact m*g reaction."""
    parent_f_avg, weight = _run_single_body_steady_state(test, device, joint_kind="revolute", parent_kinematic=True)
    _assert_simple_decoupled_pendulum(test, parent_f_avg, weight, "revolute-to-kinematic")


def test_xpbd_parent_force_ball_to_world(test, device):
    """Single body on a ball joint to world should give ~exact m*g reaction."""
    parent_f_avg, weight = _run_single_body_steady_state(test, device, joint_kind="ball", parent_kinematic=False)
    _assert_simple_decoupled_pendulum(test, parent_f_avg, weight, "ball-to-world")


def test_xpbd_parent_force_ball_to_kinematic(test, device):
    """Single body on a ball joint to a kinematic parent should give ~exact m*g reaction."""
    parent_f_avg, weight = _run_single_body_steady_state(test, device, joint_kind="ball", parent_kinematic=True)
    _assert_simple_decoupled_pendulum(test, parent_f_avg, weight, "ball-to-kinematic")


def test_xpbd_parent_force_fixed_to_world(test, device):
    """Single body on a fixed joint to world should give ~exact m*g reaction."""
    parent_f_avg, weight = _run_single_body_steady_state(test, device, joint_kind="fixed", parent_kinematic=False)
    _assert_simple_decoupled_pendulum(test, parent_f_avg, weight, "fixed-to-world")


def test_xpbd_parent_force_fixed_to_kinematic(test, device):
    """Single body on a fixed joint to a kinematic parent should give ~exact m*g reaction."""
    parent_f_avg, weight = _run_single_body_steady_state(test, device, joint_kind="fixed", parent_kinematic=True)
    _assert_simple_decoupled_pendulum(test, parent_f_avg, weight, "fixed-to-kinematic")


def test_xpbd_parent_force_chain_weight_propagation(test, device):
    """Steady-state parent-force test on a 2-link chain.

    Two links hang from a revolute joint to ground, with a second revolute
    joint between them.  After settling, the upper joint must support the
    weight of *both* links, while the lower joint must support only the
    second link.  This verifies that joint reactions propagate correctly up
    the kinematic chain.
    """
    gravity = 9.81

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity), up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    link0 = builder.add_link()
    builder.add_shape_box(link0, hx=0.1, hy=0.1, hz=0.1)
    joint0 = builder.add_joint_revolute(
        -1,
        link0,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    link1 = builder.add_link()
    builder.add_shape_box(link1, hx=0.1, hy=0.1, hz=0.1)
    joint1 = builder.add_joint_revolute(
        link0,
        link1,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint0, joint1])
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverXPBD(model, iterations=32)
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    masses = model.body_mass.numpy()
    weight_total = float(masses[0] + masses[1]) * gravity
    weight_link1 = float(masses[1]) * gravity

    dt = 1.0 / 60.0
    num_substeps = 8
    sub_dt = dt / num_substeps
    settle_steps = 120
    avg_steps = 60

    for _ in range(settle_steps):
        for _ in range(num_substeps):
            solver.step(state_in, state_out, None, None, sub_dt)
            state_in, state_out = state_out, state_in

    parent_f_avg = np.zeros((2, 6))
    for _ in range(avg_steps):
        for _ in range(num_substeps):
            solver.step(state_in, state_out, None, None, sub_dt)
            state_in, state_out = state_out, state_in
        parent_f_avg += state_in.body_parent_f.numpy()
    parent_f_avg /= avg_steps

    np.testing.assert_allclose(
        parent_f_avg[0, 2],
        weight_total,
        rtol=0.10,
        err_msg="Chain: upper joint should support total weight of both links",
    )
    np.testing.assert_allclose(
        parent_f_avg[1, 2],
        weight_link1,
        rtol=0.10,
        err_msg="Chain: lower joint should support only the second link's weight",
    )


def test_xpbd_parent_force_not_allocated(test, device):
    """``body_parent_f`` is None when not requested, and ``step`` runs without it."""
    builder = newton.ModelBuilder()
    link = builder.add_link()
    builder.add_shape_sphere(link, radius=0.1)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()

    test.assertIsNone(state_in.body_parent_f)
    test.assertIsNone(state_out.body_parent_f)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver.step(state_in, state_out, None, None, 1.0 / 60.0)

    test.assertIsNone(state_out.body_parent_f)


def test_xpbd_parent_force_zero_for_free_body(test, device):
    """A body with only a free joint should report zero parent force.

    ``solve_body_joints`` returns early for ``JointType.FREE``, so no
    constraint impulse accumulates and ``body_parent_f`` should remain at
    its zero-init value for the free body.
    """
    builder = newton.ModelBuilder()
    builder.request_state_attributes("body_parent_f")
    link = builder.add_link()
    builder.add_shape_sphere(link, radius=0.1)
    joint = builder.add_joint_free(child=link)
    builder.add_articulation([joint])
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)

    parent_f = state_out.body_parent_f.numpy()[0]
    np.testing.assert_allclose(
        parent_f,
        0.0,
        atol=1e-6,
        err_msg="Free-joint body should have zero parent force",
    )


def test_xpbd_parent_f_centripetal_zero_g(test, device):
    """Two free bodies on a hinge, zero gravity, in steady-state rotation.

    Initial state: both bodies are in rigid-body rotation about the hinge
    axis (which coincides with the system COM since masses are equal),
    so the only wrench the hinge needs to transmit per body is the
    centripetal force:

        |F_body| == m * omega^2 * r_perp     (toward the rotation axis)

    Unlike single-body solvers that report inverse-dynamics reactions in
    one step, XPBD is position-based: in one substep the constraint sees
    only the O(dt^2) curvature deviation, so the reported per-step
    reaction is dt-suppressed.  We therefore run for many sub-steps and
    take a time-average of ``body_parent_f`` over a full rotation -- the
    average converges to the analytical centripetal magnitude because
    angular momentum (and ω) are conserved.
    """
    omega = 5.0  # rad/s about Y

    # add_link (NOT add_body) so we control the joint topology and avoid
    # the implicit free joints that ``add_body`` would create.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    body_1 = builder.add_link()
    builder.add_shape_box(body_1, hx=0.25, hy=0.05, hz=0.05)
    body_2 = builder.add_link()
    builder.add_shape_box(body_2, hx=0.25, hy=0.05, hz=0.05)

    joint_free = builder.add_joint_free(child=body_1)
    joint_rev = builder.add_joint_revolute(
        body_1,
        body_2,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint_free, joint_rev])

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=16,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=1.0,
        joint_linear_compliance=0.0,
        joint_angular_compliance=0.0,
        angular_damping=0.0,
        enable_restitution=False,
    )

    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    # For rigid-body rotation about Y axis through (0.5, 0, 0):
    #   body_1 COM @ (0,0,0):  v = omega x r = (0, 0,  0.5*omega)
    #   body_2 COM @ (1,0,0):  v = omega x r = (0, 0, -0.5*omega)
    qd = state_in.body_qd.numpy()
    qd[body_1, :3] = (0.0, 0.0, 0.5 * omega)
    qd[body_1, 3:6] = (0.0, omega, 0.0)
    qd[body_2, :3] = (0.0, 0.0, -0.5 * omega)
    qd[body_2, 3:6] = (0.0, omega, 0.0)
    state_in.body_qd.assign(qd)

    # Run for a fraction of a revolution and average the *magnitude* of
    # the reported wrench (the vector rotates with the body in world
    # frame, so a direct vector-average would cancel to zero).
    dt = 1.0 / 240.0
    num_steps = 240  # ~one revolution at omega=5 rad/s
    num_substeps = 4
    sub_dt = dt / num_substeps

    f_lin_mags = []
    f_tau_mags = []
    for _ in range(num_steps):
        for _ in range(num_substeps):
            solver.step(state_in, state_out, None, None, sub_dt)
            state_in, state_out = state_out, state_in
        pf2 = state_in.body_parent_f.numpy()[body_2]
        f_lin_mags.append(float(np.linalg.norm(pf2[:3])))
        f_tau_mags.append(float(np.linalg.norm(pf2[3:6])))

    f_lin_mag_avg = float(np.mean(f_lin_mags))
    f_tau_mag_avg = float(np.mean(f_tau_mags))

    m_body2 = float(model.body_mass.numpy()[body_2])
    r_perp = 0.5  # body_2 COM offset from hinge axis
    expected_force = m_body2 * omega * omega * r_perp  # m * omega^2 * r

    # Centripetal magnitude: time-averaged |F| should match m*omega^2*r.
    np.testing.assert_allclose(
        f_lin_mag_avg,
        expected_force,
        rtol=0.10,
        err_msg=(
            f"body_2 time-avg |F| should match m*omega^2*r = {expected_force:.3f} N; got |F|={f_lin_mag_avg:.3f} N"
        ),
    )

    # A pure centripetal force passes through the body's COM trajectory,
    # so the constraint exerts negligible torque about the COM.
    test.assertLess(
        f_tau_mag_avg,
        0.10 * expected_force * r_perp,
        msg=f"body_2 time-avg |tau| about COM should be ~0; got |tau|={f_tau_mag_avg:.3f}",
    )


def test_xpbd_parent_f_consistent_across_solvers(test, device):
    """XPBD's ``body_parent_f`` must match MuJoCo / Featherstone for a static pendulum.

    Identical scene, identical initial state, one step.  The three solvers
    use different integration schemes but report wrenches in the same
    documented frame (world frame, at child COM).  Disagreement larger
    than a few percent on the dominant component indicates a convention
    or accumulation bug rather than legitimate per-solver discretization
    difference (which is bounded for a static configuration).
    """

    def _build():
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        builder.request_state_attributes("body_parent_f")
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            -1,
            link,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 1.0, 0.0),
        )
        builder.add_articulation([joint])
        return builder.finalize(device=device)

    dt = 5e-3
    results = {}
    for name, make_solver in [
        ("xpbd", lambda m: newton.solvers.SolverXPBD(m, iterations=8)),
        ("mujoco", lambda m: newton.solvers.SolverMuJoCo(m, use_mujoco_cpu=False)),
        ("featherstone", newton.solvers.SolverFeatherstone),
    ]:
        model = _build()
        solver = make_solver(model)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        solver.step(state_0, state_1, None, None, dt)
        results[name] = state_1.body_parent_f.numpy()[0]

    mg = float(model.body_mass.numpy()[0]) * 9.81
    for name, parent_f in results.items():
        np.testing.assert_allclose(parent_f[2], mg, rtol=0.05, err_msg=f"{name}: |F_z| should be ~m*g")

    # Cross-solver agreement: XPBD must be within 10% of MuJoCo on every
    # spatial component (5% would be tight for the off-axis components
    # given the different integration orders).
    np.testing.assert_allclose(
        results["xpbd"],
        results["mujoco"],
        atol=0.5,
        rtol=0.10,
        err_msg=(
            "XPBD and MuJoCo disagree on body_parent_f for a static pendulum:\n"
            f"  xpbd   = {results['xpbd']}\n"
            f"  mujoco = {results['mujoco']}"
        ),
    )


def _build_two_body_one_joint(joint_kind: str, device):
    """Two free-floating bodies connected by a single joint, gravity=0.

    The parent is attached to the world by a FREE joint; the child by the
    inner joint under test.  With gravity off and no contacts, the only
    impulse exchanged is at the inner joint, so Newton's 2nd law on the
    child becomes an exact algebraic identity against ``body_parent_f``.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")
    parent = builder.add_link()
    builder.add_shape_box(parent, hx=0.2, hy=0.1, hz=0.1)
    child = builder.add_link()
    builder.add_shape_box(child, hx=0.2, hy=0.1, hz=0.1)
    j_free = builder.add_joint_free(child=parent)
    parent_xform = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
    child_xform = wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity())
    if joint_kind == "revolute":
        j_inner = builder.add_joint_revolute(
            parent, child, parent_xform=parent_xform, child_xform=child_xform, axis=wp.vec3(0.0, 0.0, 1.0)
        )
    elif joint_kind == "ball":
        j_inner = builder.add_joint_ball(parent, child, parent_xform=parent_xform, child_xform=child_xform)
    elif joint_kind == "fixed":
        j_inner = builder.add_joint_fixed(parent, child, parent_xform=parent_xform, child_xform=child_xform)
    elif joint_kind == "prismatic":
        j_inner = builder.add_joint_prismatic(
            parent, child, parent_xform=parent_xform, child_xform=child_xform, axis=wp.vec3(0.0, 0.0, 1.0)
        )
    else:
        raise ValueError(joint_kind)
    builder.add_articulation([j_free, j_inner])
    return builder.finalize(device=device), parent, child


def _quat_to_R(q):
    x, y, z, w = q
    xx, yy, zz, xy, xz, yz, wx, wy, wz = x * x, y * y, z * z, x * y, x * z, y * z, w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )


def _newton_second_law_on_child(joint_kind, ic, *, dt, iters, device):
    """Run one step with gravity=0, return (F_reported, F_expected, tau_reported, tau_expected, dP_total).

    F_expected = m_c * (v_after - v_before) / dt  --  the actual linear force XPBD applied to the child.
    tau_expected = (R*I*R^T * w_after - R*I*R^T * w_before) / dt  --  rate of change of angular momentum
    about the child's COM, world frame.
    """
    model, parent, child = _build_two_body_one_joint(joint_kind, device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=iters,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=1.0,
        joint_linear_compliance=0.0,
        joint_angular_compliance=0.0,
        angular_damping=0.0,
        enable_restitution=False,
    )
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    qd = state_in.body_qd.numpy()
    if "v_parent" in ic:
        qd[parent, :3] = ic["v_parent"]
    if "w_parent" in ic:
        qd[parent, 3:6] = ic["w_parent"]
    if "v_child" in ic:
        qd[child, :3] = ic["v_child"]
    if "w_child" in ic:
        qd[child, 3:6] = ic["w_child"]
    state_in.body_qd.assign(qd)

    # XPBD writes back to state_in (body_q/body_qd are updated in place inside
    # ``_apply_body_deltas``), so snapshot before the step.
    body_q_before = state_in.body_q.numpy().copy()
    body_qd_before = state_in.body_qd.numpy().copy()

    solver.step(state_in, state_out, None, None, dt)

    qd_out = state_out.body_qd.numpy()
    q_out = state_out.body_q.numpy()

    mass = model.body_mass.numpy()
    I_body = model.body_inertia.numpy()

    v_in, w_in = body_qd_before[child, :3], body_qd_before[child, 3:6]
    v_out, w_out = qd_out[child, :3], qd_out[child, 3:6]
    R_in = _quat_to_R(body_q_before[child, 3:7])
    R_out = _quat_to_R(q_out[child, 3:7])

    F_expected = mass[child] * (v_out - v_in) / dt
    L_in = (R_in @ I_body[child] @ R_in.T) @ w_in
    L_out = (R_out @ I_body[child] @ R_out.T) @ w_out
    tau_expected = (L_out - L_in) / dt

    parent_f = state_out.body_parent_f.numpy()[child]
    F_reported, tau_reported = parent_f[:3], parent_f[3:6]

    # System linear momentum drift (independent check on the solver, not the diagnostic).
    dP = np.zeros(3)
    for i in range(model.body_count):
        if mass[i] == 0.0:
            continue
        dP += mass[i] * (qd_out[i, :3] - body_qd_before[i, :3])

    return F_reported, F_expected, tau_reported, tau_expected, dP


def test_xpbd_parent_f_newton_second_law_zero_g(test, device):
    """Gold-standard self-consistency for ``body_parent_f`` under XPBD.

    Setup: two free-floating bodies, one joint between them, gravity=0,
    no contacts, no ``joint_f``.  The only impulse on the child comes from
    the joint, so Newton's second law gives an exact algebraic identity:

        body_parent_f[child].linear  * dt  ==  m_c * (v_after - v_before)
        body_parent_f[child].angular * dt  ==  R*I*R^T * w_after - R*I*R^T * w_before

    This test bypasses every approximation that complicates other verifications:
    no reference solver is invoked, no closed-form physics is assumed, no
    convergence is needed for the identity to hold.  ``body_parent_f`` is the
    *applied* constraint reaction; if it does not match the actual change in
    child momentum, the reporting is broken by construction.

    System linear momentum is also checked — gravity=0 + no contacts means
    the joint applies equal-and-opposite impulses, so total ``Δp = 0``.
    """
    cases = [
        # (joint_kind, initial conditions) -- chosen to exercise different
        # constraint axes (linear vs angular, single-DOF vs multi-DOF locked).
        ("revolute", {"v_child": (0.0, 1.0, 0.0)}),  # linear mismatch at joint
        ("revolute", {"w_child": (0.0, 0.0, 2.0)}),  # spin about joint axis -- free, no force
        ("revolute", {"w_parent": (0.0, 0.0, 2.0), "w_child": (0.0, 0.0, 2.0)}),  # rigid-body spin
        ("ball", {"v_child": (0.0, 1.0, 0.0)}),
        ("ball", {"w_child": (0.5, 0.5, 0.5)}),
        ("fixed", {"v_child": (0.0, 1.0, 0.0)}),
        ("fixed", {"w_parent": (0.0, 0.0, 1.0), "w_child": (0.0, 0.0, 1.0)}),
        ("prismatic", {"v_child": (0.0, 1.0, 0.0)}),  # perpendicular to joint axis -> force
    ]

    dt = 1e-3
    iters = 32

    for joint_kind, ic in cases:
        with test.subTest(joint_kind=joint_kind, ic=tuple(ic.keys())):
            F_rep, F_exp, tau_rep, tau_exp, dP = _newton_second_law_on_child(
                joint_kind, ic, dt=dt, iters=iters, device=device
            )

            # Linear law: holds to floating-point precision because both
            # sides are direct readouts of the same XPBD impulse (no
            # integration, no quaternion coupling).
            np.testing.assert_allclose(
                F_rep,
                F_exp,
                rtol=1e-4,
                atol=1.0,
                err_msg=(
                    f"{joint_kind} ic={list(ic.keys())}: "
                    f"body_parent_f[child].linear must equal m_c * dv_c / dt.\n"
                    f"  reported = {F_rep}\n"
                    f"  expected = {F_exp}"
                ),
            )

            # Angular law: same identity in principle, but the orientation
            # rotates over dt so I_world is taken at slightly different
            # frames at the two endpoints.  Allow 1% slack on the dominant
            # component plus a small absolute floor.
            np.testing.assert_allclose(
                tau_rep,
                tau_exp,
                rtol=0.01,
                atol=1.0,
                err_msg=(
                    f"{joint_kind} ic={list(ic.keys())}: "
                    f"body_parent_f[child].angular must equal dL_c / dt at child COM.\n"
                    f"  reported = {tau_rep}\n"
                    f"  expected = {tau_exp}"
                ),
            )

            # System linear momentum conservation -- the joint exerts
            # equal-and-opposite linear forces on the two bodies.  Holds
            # at machine precision for the underlying solver, independent
            # of the body_parent_f reporting.
            np.testing.assert_allclose(
                dP,
                0.0,
                atol=1e-5,
                err_msg=f"{joint_kind} ic={list(ic.keys())}: system linear momentum must be conserved",
            )


def test_xpbd_aligned_box_stack_remains_stable(test, device):
    """Keep an aligned five-box stack upright through the reported collapse window.

    Exact face alignment accumulates solver-scale quaternion drift. MPR must
    retain four-point face manifolds as that drift crosses zero, or the stack
    loses support and collapses.
    """
    builder = newton.ModelBuilder(up_axis="Y")
    builder.add_ground_plane()
    box_count = 5
    half_extent = 0.5
    for box_index in range(box_count):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, half_extent + box_index, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body, hx=half_extent, hy=half_extent, hz=half_extent)

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=4)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = collision_pipeline.contacts()

    frame_dt = 1.0 / 60.0
    substep_count = 4
    substep_dt = frame_dt / substep_count
    for _ in range(180):
        for _ in range(substep_count):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, substep_dt)
            state_in, state_out = state_out, state_in

    body_transforms = state_in.body_q.numpy()
    positions = body_transforms[:, :3]
    orientations = body_transforms[:, 3:]
    expected_heights = half_extent + np.arange(box_count)

    test.assertTrue(np.all(np.isfinite(body_transforms)), "Stack state must remain finite")
    np.testing.assert_allclose(positions[:, 1], expected_heights, atol=2.0e-2)
    test.assertLess(
        float(np.max(np.linalg.norm(positions[:, (0, 2)], axis=1))),
        1.0e-2,
        "Stack boxes must not slide laterally out of alignment",
    )
    test.assertLess(
        float(np.max(np.linalg.norm(orientations[:, (0, 2)], axis=1))),
        1.0e-3,
        "Stack boxes must remain upright",
    )


def test_xpbd_spring_compliance_is_iteration_independent(test, device):
    """Keep an isolated compliant spring response independent of iteration count."""

    def solve(iterations: int) -> float:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        builder.add_spring(0, 1, ke=100.0, kd=0.0, control=0.0)

        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        positions = state_in.particle_q.numpy()
        positions[1, 0] = 2.0
        state_in.particle_q.assign(positions)

        solver = newton.solvers.SolverXPBD(model, iterations=iterations)
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        solved_positions = state_out.particle_q.numpy()
        return float(np.linalg.norm(solved_positions[1] - solved_positions[0]))

    one_iteration_length = solve(1)
    many_iteration_length = solve(5)
    test.assertAlmostEqual(one_iteration_length, many_iteration_length, places=5)


def test_xpbd_cloth_triangle_resists_membrane_strain(test, device):
    """Restore membrane strain in cloth created without auxiliary springs."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        density=1.0,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=0.0,
        edge_ke=0.0,
        add_springs=False,
        particle_radius=0.0,
    )
    builder.particle_mass[0] = 0.0
    builder.particle_mass[1] = 0.0

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    positions = state_in.particle_q.numpy()
    positions[2] = (0.5, 1.8, 0.0)
    state_in.particle_q.assign(positions)

    def strain_error(q: np.ndarray) -> float:
        f0 = q[1] - q[0]
        f1 = q[2] - q[0]
        e00 = 0.5 * (np.dot(f0, f0) - 1.0)
        e11 = 0.5 * (np.dot(f1, f1) - 1.0)
        e01 = np.dot(f0, f1)
        return float(np.linalg.norm((e00 + e11, e00 - e11, e01)))

    initial_error = strain_error(positions)
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    final_error = strain_error(state_out.particle_q.numpy())

    test.assertLess(final_error, 0.5 * initial_error)


def test_xpbd_connected_cloth_constraints_continue_converging(test, device):
    """Continue converging Jacobi-averaged compliant cloth constraints."""

    def solve(iterations: int) -> float:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=4,
            dim_y=4,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=0.0,
            edge_ke=5.0,
            edge_kd=0.0,
            particle_radius=0.0,
        )
        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        positions = state_in.particle_q.numpy()
        positions[12, 2] = 1.0
        state_in.particle_q.assign(positions)

        solver = newton.solvers.SolverXPBD(model, iterations=iterations)
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        heights = state_out.particle_q.numpy()[:, 2]
        return float(np.sqrt(np.mean(np.square(heights - np.mean(heights)))))

    one_iteration_error = solve(1)
    five_iteration_error = solve(5)
    test.assertLess(five_iteration_error, 0.8 * one_iteration_error)


def test_xpbd_cloth_self_contact_separates_surfaces(test, device):
    """Separate disconnected cloth surfaces with triangle-mesh self-contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)]
    common_args = {
        "rot": wp.quat_identity(),
        "scale": 1.0,
        "vel": wp.vec3(0.0),
        "vertices": vertices,
        "indices": [0, 1, 2],
        "density": 1.0,
        "tri_ke": 1.0e4,
        "tri_ka": 1.0e4,
        "edge_ke": 0.0,
        "particle_radius": 0.0,
    }
    builder.add_cloth_mesh(pos=wp.vec3(0.0), **common_args)
    builder.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.05), **common_args)
    for particle in range(3):
        builder.particle_mass[particle] = 0.0

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=5,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.1,
        particle_self_contact_margin=0.2,
        particle_topological_contact_filter_threshold=1,
    )
    for _ in range(10):
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        state_in, state_out = state_out, state_in

    positions = state_in.particle_q.numpy()
    velocities = state_in.particle_qd.numpy()
    separation = float(np.mean(positions[3:, 2]) - np.mean(positions[:3, 2]))
    np.testing.assert_allclose(separation, 0.1, atol=1.0e-4)
    np.testing.assert_allclose(velocities[3:], 0.0, atol=1.0e-4)


def test_xpbd_cloth_self_contact_prevents_triangle_tunneling(test, device):
    """Prevent a triangle trajectory from tunneling through another triangle."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cloth_args = {
        "rot": wp.quat_identity(),
        "scale": 1.0,
        "density": 1.0,
        "tri_ke": 1.0e-6,
        "tri_ka": 1.0e-6,
        "tri_kd": 0.0,
        "edge_ke": 0.0,
        "particle_radius": 0.0,
    }
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-1.0, -1.0, 0.0), wp.vec3(1.0, -1.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        **cloth_args,
    )
    for particle in range(3):
        builder.particle_mass[particle] = 0.0
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-0.5, -0.5, 0.1), wp.vec3(0.5, -0.5, 0.1), wp.vec3(0.0, 0.5, 0.1)],
        indices=[0, 1, 2],
        **cloth_args,
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    velocities = state_in.particle_qd.numpy()
    velocities[5, 2] = -12.0
    state_in.particle_qd.assign(velocities)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=5,
        particle_enable_self_contact=True,
        particle_enable_triangle_intersection_recovery=True,
        particle_self_contact_radius=0.02,
        particle_self_contact_margin=0.05,
        particle_max_depenetration_velocity=12.0,
        particle_topological_contact_filter_threshold=1,
    )

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    detector = solver.trimesh_collision_detector
    detector.refit(state_out.particle_q)
    detector.triangle_triangle_intersection_detection()
    intersection_count = int(detector.triangle_intersecting_triangles_count.numpy().sum())
    test.assertEqual(intersection_count, 0, "self-contact must not leave intersecting triangle pairs")
    test.assertGreater(
        float(np.min(state_out.particle_q.numpy()[3:, 2])),
        0.0,
        "conservative self-contact must stop a fully traversing triangle before it changes sides",
    )


def test_xpbd_cloth_self_contact_recovers_intersection_created_by_constraints(test, device):
    """Recover a triangle intersection created during XPBD constraint iterations."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cloth_args = {
        "rot": wp.quat_identity(),
        "scale": 1.0,
        "density": 1.0,
        "tri_ke": 1.0e-6,
        "tri_ka": 1.0e-6,
        "tri_kd": 0.0,
        "edge_ke": 0.0,
        "particle_radius": 0.0,
    }
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-1.0, -1.0, 0.0), wp.vec3(1.0, -1.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        **cloth_args,
    )
    for particle in range(3):
        builder.particle_mass[particle] = 0.0
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-0.5, -0.5, 0.04), wp.vec3(0.5, -0.5, 0.04), wp.vec3(0.0, 0.5, 0.04)],
        indices=[0, 1, 2],
        **cloth_args,
    )
    anchor = builder.add_particle(wp.vec3(0.0, 0.5, 0.1), wp.vec3(0.0), mass=0.0, radius=0.0)
    builder.add_spring(5, anchor, ke=1.0e8, kd=0.0, control=0.0)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    positions = state_in.particle_q.numpy()
    positions[anchor] = (0.0, 0.5, -0.2)
    state_in.particle_q.assign(positions)
    state_out.particle_q.assign(positions)

    solver = newton.solvers.SolverXPBD(
        model,
        iterations=1,
        particle_enable_self_contact=True,
        particle_enable_triangle_intersection_recovery=True,
        particle_self_contact_relaxation=1.0,
        particle_triangle_intersection_relaxation=1.0,
        particle_self_contact_radius=0.02,
        particle_self_contact_margin=0.05,
        particle_max_depenetration_velocity=100.0,
        particle_topological_contact_filter_threshold=1,
    )

    detector = solver.trimesh_collision_detector
    detector.refit(state_in.particle_q)
    detector.triangle_triangle_intersection_detection()
    test.assertEqual(int(detector.triangle_intersecting_triangles_count.numpy().sum()), 0)

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    detector.refit(state_out.particle_q)
    detector.triangle_triangle_intersection_detection()
    intersection_count = int(detector.triangle_intersecting_triangles_count.numpy().sum())
    test.assertEqual(intersection_count, 0, "self-contact must catch intersections created inside the solve")


def test_xpbd_triangle_intersection_recovery_honors_depenetration_limit(test, device):
    """Limit exact recovery of a pre-existing triangle intersection."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cloth_args = {
        "rot": wp.quat_identity(),
        "scale": 1.0,
        "density": 1.0,
        "tri_ke": 1.0e-6,
        "tri_ka": 1.0e-6,
        "tri_kd": 0.0,
        "edge_ke": 0.0,
        "particle_radius": 0.0,
    }
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-1.0, -1.0, 0.0), wp.vec3(1.0, -1.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        **cloth_args,
    )
    for particle in range(3):
        builder.particle_mass[particle] = 0.0
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-0.5, -0.5, 0.1), wp.vec3(0.5, -0.5, 0.1), wp.vec3(0.0, 0.5, 0.1)],
        indices=[0, 1, 2],
        **cloth_args,
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    positions = state_in.particle_q.numpy()
    positions[5, 2] = -0.1
    state_in.particle_q.assign(positions)
    state_out.particle_q.assign(positions)
    dt = 1.0 / 60.0
    max_depenetration_velocity = 0.1
    ballistic_positions = positions.copy()
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=1,
        particle_enable_self_contact=True,
        particle_enable_triangle_intersection_recovery=True,
        particle_self_contact_radius=0.02,
        particle_self_contact_margin=0.05,
        particle_max_depenetration_velocity=max_depenetration_velocity,
        particle_topological_contact_filter_threshold=1,
    )

    solver.step(state_in, state_out, None, None, dt)
    correction = np.linalg.norm(state_out.particle_q.numpy() - ballistic_positions, axis=1)
    test.assertGreater(int(solver.trimesh_collision_detector.triangle_intersecting_triangles_count.numpy().sum()), 0)
    test.assertLessEqual(
        float(np.max(correction)),
        max_depenetration_velocity * dt + 1.0e-5,
        "exact recovery must not replace its configured limit with the pair's full crossing motion",
    )


def test_xpbd_cloth_self_contact_recovers_local_triangle_inversion(test, device):
    """Recover intersecting local triangles without applying full cloth thickness."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(2.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(1.0, 1.0, 0.0),
            wp.vec3(2.0, 1.0, 0.0),
        ],
        indices=[0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4],
        density=1.0,
        tri_ke=1.0e-6,
        tri_ka=1.0e-6,
        edge_ke=0.0,
        particle_radius=0.0,
    )
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    positions = state_in.particle_q.numpy()
    positions[2] = (0.0, 0.0, -0.02)
    positions[4] = (1.0, 0.0, 0.02)
    positions[5] = (0.0, 1.0, 0.02)
    state_in.particle_q.assign(positions)
    state_out.particle_q.assign(positions)

    solver = newton.solvers.SolverXPBD(
        model,
        iterations=10,
        particle_enable_self_contact=True,
        particle_enable_triangle_intersection_recovery=True,
        particle_self_contact_relaxation=0.5,
        particle_triangle_intersection_relaxation=0.5,
        particle_self_contact_radius=0.1,
        particle_self_contact_margin=1.0,
        particle_topological_contact_filter_threshold=2,
    )
    detector = solver.trimesh_collision_detector
    detector.refit(state_in.particle_q)
    detector.triangle_triangle_intersection_detection()
    test.assertGreater(int(detector.triangle_intersecting_triangles_count.numpy().sum()), 0)

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    detector.refit(state_out.particle_q)
    detector.triangle_triangle_intersection_detection()
    intersection_count = int(detector.triangle_intersecting_triangles_count.numpy().sum())
    test.assertEqual(intersection_count, 0, "local exact recovery must remove the inversion")


def test_xpbd_cloth_self_contact_relaxation_is_independent(test, device):
    """Control cloth self-contact correction without changing the shared contact relaxation."""

    def solve(self_contact_relaxation: float) -> tuple[float, newton.solvers.SolverXPBD]:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)]
        cloth_args = {
            "rot": wp.quat_identity(),
            "scale": 1.0,
            "vel": wp.vec3(0.0),
            "vertices": vertices,
            "indices": [0, 1, 2],
            "density": 1.0,
            "tri_ke": 1.0e4,
            "tri_ka": 1.0e4,
            "edge_ke": 0.0,
            "particle_radius": 0.0,
        }
        builder.add_cloth_mesh(pos=wp.vec3(0.0), **cloth_args)
        builder.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.05), **cloth_args)
        for particle in range(3):
            builder.particle_mass[particle] = 0.0

        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=1,
            soft_contact_relaxation=0.9,
            particle_enable_self_contact=True,
            particle_self_contact_relaxation=self_contact_relaxation,
            particle_self_contact_radius=0.1,
            particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
        )
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        positions = state_out.particle_q.numpy()
        separation = float(np.mean(positions[3:, 2]) - np.mean(positions[:3, 2]))
        return separation, solver

    low_separation, low_solver = solve(0.2)
    high_separation, high_solver = solve(0.8)

    test.assertGreater(high_separation, low_separation + 0.005)
    test.assertEqual(low_solver.soft_contact_relaxation, high_solver.soft_contact_relaxation)
    test.assertEqual(low_solver.particle_self_contact_relaxation, 0.2)
    test.assertEqual(high_solver.particle_self_contact_relaxation, 0.8)


def test_xpbd_cloth_overlap_recovery_does_not_accumulate_velocity(test, device):
    """Avoid accumulating velocity when elasticity opposes overlap recovery."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)]
    common_args = {
        "rot": wp.quat_identity(),
        "scale": 1.0,
        "vel": wp.vec3(0.0),
        "vertices": vertices,
        "indices": [0, 1, 2],
        "density": 1.0,
        "tri_ke": 1.0e4,
        "tri_ka": 1.0e4,
        "edge_ke": 0.0,
        "particle_radius": 0.0,
    }
    builder.add_cloth_mesh(pos=wp.vec3(0.0), **common_args)
    builder.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.05), **common_args)
    for particle in range(3):
        builder.particle_mass[particle] = 0.0

    # Tether the moving surface inside the contact thickness. Each step asks
    # self-contact to recover overlap while the springs oppose that recovery.
    for vertex in vertices:
        builder.add_particle(
            pos=(vertex[0], vertex[1], vertex[2] + 0.05),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
            radius=0.0,
        )
    for particle in range(3):
        builder.add_spring(particle + 3, particle + 6, ke=1.0e5, kd=10.0, control=0.0)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=5,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.1,
        particle_self_contact_margin=0.2,
        particle_topological_contact_filter_threshold=1,
    )
    for _ in range(60):
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        state_in, state_out = state_out, state_in

    moving_velocities = state_in.particle_qd.numpy()[3:6]
    np.testing.assert_allclose(moving_velocities, 0.0, atol=1.0e-4)


def test_xpbd_cloth_self_contact_friction_reduces_slip(test, device):
    """Reduce tangential slip between overlapping cloth surfaces."""

    def simulate(friction: float) -> float:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)]
        common_args = {
            "rot": wp.quat_identity(),
            "scale": 1.0,
            "vertices": vertices,
            "indices": [0, 1, 2],
            "density": 1.0,
            "tri_ke": 1.0e4,
            "tri_ka": 1.0e4,
            "edge_ke": 0.0,
            "particle_radius": 0.0,
        }
        builder.add_cloth_mesh(pos=wp.vec3(0.0), vel=wp.vec3(0.0), **common_args)
        builder.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.05), vel=wp.vec3(1.0, 0.0, 0.0), **common_args)
        for particle in range(3):
            builder.particle_mass[particle] = 0.0

        model = builder.finalize(device=device)
        model.particle_mu = friction
        state_in = model.state()
        state_out = model.state()
        initial_positions = state_in.particle_q.numpy().copy()
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=5,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.1,
            particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
        )
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        final_positions = state_out.particle_q.numpy()
        return float(np.mean(final_positions[3:6, 0] - initial_positions[3:6, 0]))

    slip_without_friction = simulate(0.0)
    slip_partial_friction = simulate(0.5)
    slip_with_friction = simulate(1.0)
    test.assertGreater(slip_without_friction, 0.01)
    # A high coefficient saturates into near-complete stick.
    test.assertLess(slip_with_friction, 0.2 * slip_without_friction)
    # Slip must vary continuously with the coefficient. A friction impulse that
    # ignores its Coulomb bound would clamp every non-zero coefficient to the
    # same near-zero slip and erase this ordering.
    test.assertLess(slip_partial_friction, 0.75 * slip_without_friction)
    test.assertGreater(slip_partial_friction, 1.5 * slip_with_friction)


def test_xpbd_cloth_self_contact_velocity_stabilization_is_inelastic(test, device):
    """Stop closing self-contact velocity independently of position relaxation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    vertices = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)]
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=[0, 1, 2],
        density=1.0,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        edge_ke=0.0,
        particle_radius=0.0,
    )
    for particle in range(3):
        builder.particle_mass[particle] = 0.0
    moving_particle = builder.add_particle(
        pos=(0.25, 0.25, 0.11),
        vel=(0.0, 0.0, -1.0),
        mass=1.0,
        radius=0.0,
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=1,
        particle_enable_self_contact=True,
        particle_self_contact_relaxation=0.2,
        particle_self_contact_radius=0.1,
        particle_self_contact_margin=0.2,
        particle_topological_contact_filter_threshold=1,
    )
    solver.step(state_in, state_out, None, None, 0.02)

    test.assertGreaterEqual(
        float(state_out.particle_qd.numpy()[moving_particle, 2]),
        -1.0e-5,
        "inelastic self-contact must remove the full closing velocity",
    )


def test_xpbd_external_rigid_solver_preserves_body_state(test, device):
    """Preserve externally integrated bodies while resolving particle contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
    body = builder.add_body()
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_particle(pos=(1.0, 0.6, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    body_q = state_out.body_q.numpy()
    body_q[body, :3] = (1.0, 0.0, 0.0)
    state_out.body_q.assign(body_q)

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_out, contacts)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=4,
        integrate_with_external_rigid_solver=True,
    )
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    np.testing.assert_allclose(state_out.body_q.numpy()[body, :3], (1.0, 0.0, 0.0), atol=1.0e-6)
    test.assertGreater(float(state_out.particle_q.numpy()[0, 1]), 0.6)


def test_xpbd_shape_adhesion_pulls_particle(test, device):
    """Apply a shape's adhesion distance to a nearby particle."""

    def solve(adhesion: float) -> float:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
        builder.add_shape_box(
            -1,
            hx=1.0,
            hy=0.5,
            hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(ka=adhesion),
        )
        builder.add_particle(pos=(0.0, 0.7, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.3)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)
        test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)

        solver = newton.solvers.SolverXPBD(model, iterations=5)
        solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
        return float(state_out.particle_q.numpy()[0, 1])

    position_without_adhesion = solve(0.0)
    position_with_adhesion = solve(0.2)
    test.assertAlmostEqual(position_without_adhesion, 0.7, places=5)
    test.assertAlmostEqual(position_with_adhesion, 0.6, places=4)


def test_xpbd_soft_contact_honors_shape_margin(test, device):
    """Keep a soft particle outside the rigid shape's collision margin."""

    def solve(shape_margin: float) -> float:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
        builder.add_shape_box(
            -1,
            hx=1.0,
            hy=0.5,
            hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(margin=shape_margin),
        )
        builder.add_particle(pos=(0.0, 0.65, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.2)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)

        solver = newton.solvers.SolverXPBD(model, iterations=8)
        solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
        return float(state_out.particle_q.numpy()[0, 1])

    test.assertAlmostEqual(solve(0.0), 0.65, places=5)
    test.assertGreater(solve(0.1), 0.69)


def test_xpbd_full_surface_soft_contact_blocks_triangle_interior(test, device):
    """Block a rigid sphere that contacts the interior of a coarse cloth triangle."""

    def solve(full_surface: bool):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_cloth_grid(
            pos=wp.vec3(-0.25, -0.25, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.5,
            cell_y=0.5,
            mass=1.0,
            particle_radius=0.01,
        )
        for particle in range(len(builder.particle_mass)):
            builder.particle_mass[particle] = 0.0

        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()))
        builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = 100.0
        shape_cfg.margin = 0.01
        builder.add_shape_sphere(body, radius=0.1, cfg=shape_cfg)

        model = builder.finalize(device=device)
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_margin=0.03,
            enable_rigid_soft_full_surface_contact=full_surface,
        )
        contacts = collision_pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        collision_pipeline.collide(state_in, contacts)
        contact_count = int(contacts.soft_contact_count.numpy()[0])
        has_feature_contact = bool(np.any(contacts.soft_contact_particle.numpy()[:contact_count] < 0))

        solver = newton.solvers.SolverXPBD(model, iterations=8)
        test.assertTrue(solver.coupling_supports_full_surface_soft_contacts())
        solver.step(state_in, state_out, None, contacts, 1.0 / 120.0)
        return contact_count, has_feature_contact, state_out.body_q.numpy()[body], state_out.body_qd.numpy()[body]

    vertex_count, _, vertex_position, vertex_velocity = solve(False)
    feature_count, has_feature_contact, feature_position, feature_velocity = solve(True)

    test.assertEqual(vertex_count, 0)
    test.assertGreater(feature_count, 0)
    test.assertTrue(has_feature_contact)
    test.assertAlmostEqual(float(vertex_velocity[2]), -1.0, places=5)
    test.assertGreater(float(feature_velocity[2]), -0.1)
    test.assertGreater(float(feature_position[2]), float(vertex_position[2]) + 0.005)


def test_xpbd_particle_contact_limits_depenetration_velocity(test, device):
    """Limit correction velocity for a particle that starts deeply penetrating."""

    def solve(max_velocity: float | None) -> float:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
        builder.add_ground_plane()
        builder.add_particle(pos=(0.0, -1.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)

        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=4,
            soft_contact_max_depenetration_velocity=max_velocity,
        )
        solver.step(state_in, state_out, None, contacts, 0.1)
        return float(state_out.particle_q.numpy()[0, 1] - state_in.particle_q.numpy()[0, 1])

    automatic_displacement = solve(None)
    explicit_displacement = solve(1.0)
    test.assertGreater(automatic_displacement, 0.0)
    test.assertLessEqual(automatic_displacement, 0.025 + 1.0e-5)
    test.assertGreater(explicit_displacement, automatic_displacement)
    test.assertLessEqual(explicit_displacement, 0.1 + 1.0e-5)


def test_xpbd_particle_contact_depenetration_does_not_launch_particle(test, device):
    """Recover deep overlap without creating persistent separating velocity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
    builder.add_ground_plane()
    builder.add_particle(pos=(0.0, -0.1, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=5)

    heights = []
    for _ in range(8):
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
        heights.append(float(state_out.particle_q.numpy()[0, 1]))
        np.testing.assert_allclose(state_out.particle_qd.numpy()[0], 0.0, atol=1.0e-5)
        state_in, state_out = state_out, state_in

    test.assertTrue(all(b >= a for a, b in pairwise(heights)))
    np.testing.assert_allclose(heights[-1], 0.2, atol=1.0e-5)


def test_xpbd_external_body_motion_transfers_friction(test, device):
    """Transfer tangential motion from an externally integrated body through friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
    body = builder.add_body()
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_particle(pos=(0.0, 0.6, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)

    model = builder.finalize(device=device)
    model.particle_mu = 1.0
    state_in = model.state()
    state_out = model.state()
    dt = 1.0 / 60.0
    body_q = state_out.body_q.numpy()
    body_q[body, 0] = dt
    state_out.body_q.assign(body_q)
    body_qd = state_out.body_qd.numpy()
    body_qd[body, 0] = 1.0
    state_out.body_qd.assign(body_qd)

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_in, contacts)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=5,
        integrate_with_external_rigid_solver=True,
    )
    solver.step(state_in, state_out, None, contacts, dt)

    np.testing.assert_allclose(state_out.particle_qd.numpy()[0], (1.0, 0.0, 0.0), atol=1.0e-4)
    np.testing.assert_allclose(state_out.body_qd.numpy()[body, :3], (1.0, 0.0, 0.0), atol=1.0e-6)


def test_xpbd_external_body_velocity_transfers_friction_without_pose_delta(test, device):
    """Use prescribed body velocity when a lagged proxy pose is unchanged."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
    body = builder.add_body()
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_particle(pos=(0.0, 0.6, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)

    model = builder.finalize(device=device)
    model.particle_mu = 1.0
    state_in = model.state()
    state_out = model.state()
    body_qd = state_out.body_qd.numpy()
    body_qd[body, 0] = 1.0
    state_out.body_qd.assign(body_qd)

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_in, contacts)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=5,
        integrate_with_external_rigid_solver=True,
    )
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    test.assertGreater(float(state_out.particle_qd.numpy()[0, 0]), 0.9)


def test_xpbd_static_soft_contact_enters_static_friction_branch(test, device):
    """Stop low-speed cloth slip supported by redundant surface contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
    builder.add_cloth_grid(
        pos=wp.vec3(-0.25, -0.25, 0.005),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.01, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.5,
        cell_y=0.5,
        mass=1.0,
        tri_ke=1.0e-6,
        tri_ka=1.0e-6,
        edge_ke=0.0,
        particle_radius=0.01,
    )
    model = builder.finalize(device=device)
    model.particle_mu = 1.0
    model.soft_contact_mu = 1.0
    state_in = model.state()
    state_out = model.state()
    collision_pipeline = newton.CollisionPipeline(
        model,
        soft_contact_margin=0.02,
        enable_rigid_soft_full_surface_contact=True,
    )
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_in, contacts)
    solver = newton.solvers.SolverXPBD(model, iterations=2)

    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    tangent_speed = np.linalg.norm(state_out.particle_qd.numpy()[:, :2], axis=1)
    np.testing.assert_allclose(tangent_speed, 0.0, atol=1.0e-6)


def test_xpbd_particle_damping_is_time_step_independent(test, device):
    """Apply exponential particle damping to velocity and predicted position."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=1.0, radius=0.0)
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    dt = 0.1
    damping = 2.0

    solver = newton.solvers.SolverXPBD(model, iterations=1, particle_damping=damping)
    solver.step(state_in, state_out, None, None, dt)

    expected_velocity = np.exp(-damping * dt)
    np.testing.assert_allclose(state_out.particle_qd.numpy()[0], (expected_velocity, 0.0, 0.0), atol=1.0e-6)
    np.testing.assert_allclose(
        state_out.particle_q.numpy()[0],
        (expected_velocity * dt, 0.0, 0.0),
        atol=1.0e-6,
    )


def test_xpbd_inactive_shape_candidates_do_not_dilute_friction(test, device):
    """Exclude inactive broad-phase candidates from contact averaging."""

    def solve(inactive_shape_count: int) -> float:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        for i in range(inactive_shape_count):
            builder.add_shape_box(
                -1,
                xform=wp.transform((0.0, -0.5 - 0.05 * i, 0.0), wp.quat_identity()),
                hx=0.5,
                hy=0.25,
                hz=0.5,
            )
        builder.add_particle(pos=(0.0, 0.6, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)

        model = builder.finalize(device=device)
        model.particle_mu = 1.0
        state_in = model.state()
        state_out = model.state()
        body_qd = state_out.body_qd.numpy()
        body_qd[body, 0] = 1.0
        state_out.body_qd.assign(body_qd)

        collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=2.0)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)
        test.assertGreaterEqual(int(contacts.soft_contact_count.numpy()[0]), inactive_shape_count + 1)
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=5,
            integrate_with_external_rigid_solver=True,
        )
        solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
        return float(state_out.particle_qd.numpy()[0, 0])

    reference_velocity = solve(0)
    cluttered_velocity = solve(8)
    test.assertGreater(reference_velocity, 0.9)
    test.assertAlmostEqual(cluttered_velocity, reference_velocity, places=4)


def test_xpbd_fully_coupled_rigid_cloth_contact(test, device):
    """Exchange tangential momentum when XPBD owns both a rigid body and cloth."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
    body = builder.add_body(mass=10.0, inertia=wp.mat33(np.eye(3)))
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.6, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-0.1, 0.0, -0.1), wp.vec3(0.1, 0.0, -0.1), wp.vec3(0.0, 0.0, 0.1)],
        indices=[0, 1, 2],
        density=1.0,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        edge_ke=0.0,
        particle_radius=0.2,
    )

    model = builder.finalize(device=device)
    model.particle_mu = 1.0
    state_in = model.state()
    state_out = model.state()
    body_qd = state_in.body_qd.numpy()
    body_qd[body, 0] = 1.0
    state_in.body_qd.assign(body_qd)

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_in, contacts)
    solver = newton.solvers.SolverXPBD(model, iterations=5)
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    cloth_velocity = np.mean(state_out.particle_qd.numpy(), axis=0)
    test.assertGreater(float(cloth_velocity[0]), 0.5)
    test.assertLess(abs(float(cloth_velocity[1])), 1.0e-3)
    test.assertLess(float(state_out.body_qd.numpy()[body, 0]), 1.0)


def test_xpbd_many_particle_contacts_do_not_amplify_body_delta(test, device):
    """Normalize simultaneous particle contacts sharing one rigid body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Y")
    body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_shape_box(
        body,
        hx=0.75,
        hy=0.5,
        hz=0.75,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    for z in np.linspace(-0.5, 0.5, 5):
        for x in np.linspace(-0.5, 0.5, 5):
            builder.add_particle(pos=(x, 0.55, z), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_in, contacts)
    solver = newton.solvers.SolverXPBD(model, iterations=5)
    solver.compute_body_velocity_from_position_delta = True
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    body_position = state_out.body_q.numpy()[body, :3]
    body_velocity = state_out.body_qd.numpy()[body]
    test.assertTrue(np.all(np.isfinite(body_position)))
    test.assertTrue(np.all(np.isfinite(body_velocity)))
    test.assertGreater(float(body_position[1]), -0.03)
    test.assertLess(float(np.linalg.norm(body_velocity)), 0.1)


devices = get_test_devices()


class TestSolverXPBD(unittest.TestCase):
    pass


def _incline_slide_distance(device, friction: float, iterations: int, frames: int = 60) -> float:
    """Slide a cloth patch down a 40-degree incline and return its travel [m]."""
    theta = math.radians(40.0)
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = friction
    cfg.ka = 0.0
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)
    builder.add_shape_plane(body=-1, xform=wp.transform(wp.vec3(0.0), rot), width=60.0, length=60.0, cfg=cfg)
    size = 8 * 0.05
    builder.add_cloth_grid(
        pos=wp.vec3(
            -0.5 * size * math.cos(theta) + 0.02 * math.sin(theta),
            -0.5 * size,
            0.5 * size * math.sin(theta) + 0.02 * math.cos(theta),
        ),
        rot=rot,
        vel=wp.vec3(0.0),
        dim_x=8,
        dim_y=8,
        cell_x=0.05,
        cell_y=0.05,
        mass=1.0e-3,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=2.0,
        edge_ke=1.0,
        edge_kd=0.1,
        particle_radius=0.01,
    )
    model = builder.finalize(device=device)
    model.soft_contact_mu = friction
    model.particle_mu = friction
    model.shape_material_ka.zero_()

    solver = newton.solvers.SolverXPBD(model, iterations=iterations)
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.03, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    start = state_0.particle_q.numpy().mean(axis=0)
    for _ in range(frames):
        for _ in range(4):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
    end = state_0.particle_q.numpy().mean(axis=0)
    downhill = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    return float(np.dot(end - start, downhill))


def test_xpbd_cloth_friction_is_independent_of_iteration_count(test, device):
    """Keep cloth-shape friction a material property, not a solver-iteration count.

    A friction impulse that is re-applied at its full Coulomb bound on every
    iteration makes the effective friction force scale with ``iterations``, so
    the same material slides very different distances at different iteration
    counts.
    """
    distances = [_incline_slide_distance(device, friction=0.3, iterations=n) for n in (1, 2, 4, 8)]
    reference = distances[0]
    test.assertGreater(reference, 0.1, "the patch must actually slide at mu below tan(theta)")
    for iterations, distance in zip((1, 2, 4, 8), distances, strict=True):
        test.assertAlmostEqual(
            distance / reference,
            1.0,
            delta=0.15,
            msg=f"slide distance changed by more than 15% at {iterations} iterations",
        )


def test_xpbd_cloth_friction_obeys_coulomb_threshold(test, device):
    """Hold a cloth patch only when mu exceeds tan of the incline angle.

    Spending the Coulomb budget once in the positional solve and again in the
    velocity pass doubles the achievable friction force, which lets the patch
    stick well below the analytic threshold.
    """
    tan_theta = math.tan(math.radians(40.0))
    below = _incline_slide_distance(device, friction=0.5 * tan_theta, iterations=8)
    above = _incline_slide_distance(device, friction=1.2 * tan_theta, iterations=8)
    # Free-fall reference for the same duration, used to bound the sliding case.
    frictionless = _incline_slide_distance(device, friction=0.0, iterations=8)

    test.assertGreater(below, 0.25 * frictionless, "mu below tan(theta) must not arrest the patch")
    test.assertLess(above, 0.05 * frictionless, "mu above tan(theta) must hold the patch")


def test_xpbd_triangle_intersection_recovery_stays_bounded(test, device):
    """Keep intersection recovery stable when one vertex joins many intersecting pairs.

    Recovery separates each intersecting triangle pair. Without a shared Jacobi
    scale, a vertex belonging to several pairs receives every pair's full
    separation at once, and that overshoot creates more intersections than it
    removes.
    """
    # Zero gravity keeps every bit of residual motion attributable to the
    # recovery impulses rather than to bulk free fall.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    # Two crumpled layers seeded close together produce many-to-many triangle
    # intersections through a single shared region.
    rng = np.random.default_rng(7)
    for layer in range(2):
        vertices = []
        indices = []
        stride = 9
        for y in range(stride):
            for x in range(stride):
                vertices.append(wp.vec3(x * 0.05, y * 0.05, layer * 0.004 + float(rng.uniform(-0.004, 0.004))))
        for y in range(stride - 1):
            for x in range(stride - 1):
                v0 = y * stride + x
                indices.extend((v0, v0 + 1, v0 + stride, v0 + 1, v0 + 1 + stride, v0 + stride))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=vertices,
            indices=indices,
            density=0.2,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=0.0,
            edge_ke=0.01,
            edge_kd=0.0,
            particle_radius=0.01,
        )

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=4,
        particle_enable_self_contact=True,
        particle_enable_triangle_intersection_recovery=True,
        particle_self_contact_radius=0.02,
        particle_self_contact_margin=0.04,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(120):
        solver.rebuild_bvh(state_0)
        for _ in range(4):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

    probe = TriMeshCollisionDetector(model)
    probe.refit(state_0.particle_q)
    probe.triangle_triangle_intersection_detection()
    remaining = int(probe.triangle_intersecting_triangles_count.numpy().sum() // 2)
    positions = state_0.particle_q.numpy()
    speeds = np.linalg.norm(state_0.particle_qd.numpy(), axis=1)

    test.assertTrue(np.all(np.isfinite(positions)), "recovery must not produce non-finite positions")
    test.assertEqual(remaining, 0, f"recovery must resolve every triangle intersection ({remaining} left)")
    # Unbounded overshoot shows up as runaway speed long before it shows up as
    # a non-finite position.
    test.assertLess(
        float(np.max(speeds)),
        2.0,
        f"intersection recovery must not inject energy (max speed {float(np.max(speeds)):.3f} m/s)",
    )


def test_xpbd_cloth_vertices_do_not_self_repel_without_self_contact(test, device):
    """Keep a plain cloth at rest when its vertex spacing is below the particle diameter.

    The particle-particle fallback treats every vertex as a sphere. Without a
    rest-pose exclusion it collides each vertex with its own immediate mesh
    neighbors, and that permanent overlap fights the stretch constraints until
    the surface tears itself apart.
    """
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = 1.0
    cfg.ka = 0.0
    builder.add_ground_plane(cfg=cfg)
    # Spacing 0.025 m is deliberately below the 0.03 m particle diameter.
    builder.add_cloth_grid(
        pos=wp.vec3(-0.2, -0.2, 0.2),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=16,
        dim_y=16,
        cell_x=0.025,
        cell_y=0.025,
        mass=2.0e-4,
        tri_ke=8.0e3,
        tri_ka=8.0e3,
        tri_kd=0.0,
        edge_ke=0.002,
        edge_kd=0.0,
        particle_radius=0.015,
    )
    model = builder.finalize(device=device)
    model.particle_mu = 1.0
    model.soft_contact_mu = 1.0

    solver = newton.solvers.SolverXPBD(model, iterations=6)
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.04, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(240):
        for _ in range(4):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

    positions = state_0.particle_q.numpy()
    speeds = np.linalg.norm(state_0.particle_qd.numpy(), axis=1)
    rest = model.particle_q.numpy()
    test.assertTrue(np.all(np.isfinite(positions)), "cloth positions must remain finite")
    test.assertLess(
        float(np.max(speeds)), 0.2, f"settled cloth must be at rest (max speed {float(np.max(speeds)):.3f} m/s)"
    )
    # Self-repulsion inflates the patch well beyond its rest footprint.
    rest_extent = float(np.max(rest[:, 0]) - np.min(rest[:, 0]))
    final_extent = float(np.max(positions[:, 0]) - np.min(positions[:, 0]))
    test.assertLess(final_extent, 1.25 * rest_extent, "cloth must not stretch itself apart by vertex self-repulsion")


def _aerodynamic_patch(device, gravity, air_density, air_velocity, frames, lift=0.0, drag=1.0):
    """Simulate a free cloth patch lying in the XY plane and return (drift, velocity)."""
    builder = newton.ModelBuilder(gravity=gravity)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.25, -0.25, 2.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=16,
        dim_y=16,
        cell_x=0.03125,
        cell_y=0.03125,
        mass=5.0e-4,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=1.0,
        edge_ke=0.5,
        edge_kd=0.05,
        particle_radius=0.005,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=4,
        air_density=air_density,
        air_drag_coefficient=drag,
        air_lift_coefficient=lift,
        air_velocity=air_velocity,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    start = state_0.particle_q.numpy().mean(axis=0)
    for _ in range(frames):
        for _ in range(4):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
    drift = state_0.particle_q.numpy().mean(axis=0) - start
    velocity = state_0.particle_qd.numpy().mean(axis=0)
    return drift, velocity


def test_xpbd_air_drag_reduces_fall_speed(test, device):
    """Slow a falling cloth patch with aerodynamic drag and reach a terminal speed."""
    _, free_fall = _aerodynamic_patch(device, (0.0, 0.0, -9.81), 0.0, (0.0, 0.0, 0.0), frames=90)
    _, light = _aerodynamic_patch(device, (0.0, 0.0, -9.81), 1.225, (0.0, 0.0, 0.0), frames=90)
    _, heavy = _aerodynamic_patch(device, (0.0, 0.0, -9.81), 20.0, (0.0, 0.0, 0.0), frames=90)

    # 1.5 s of unresisted free fall.
    test.assertAlmostEqual(free_fall[2], -9.81 * 1.5, delta=0.2)
    test.assertGreater(light[2], free_fall[2], "drag must slow the fall")
    test.assertGreater(heavy[2], light[2], "denser air must slow the fall further")
    test.assertLess(heavy[2], 0.0, "the patch must still fall")


def test_xpbd_air_drag_is_disabled_by_zero_density(test, device):
    """Leave the simulation untouched when air density is zero."""
    drift, velocity = _aerodynamic_patch(device, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 5.0), frames=60)
    np.testing.assert_allclose(drift, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(velocity, 0.0, atol=1.0e-6)


def test_xpbd_air_drag_follows_angle_of_attack(test, device):
    """Push cloth with face-on wind and leave edge-on wind without effect.

    The patch lies in the XY plane, so wind along +Z meets its full frontal
    area while wind along +X slides past edge-on and projects to zero area.
    """
    face_on, face_velocity = _aerodynamic_patch(device, (0.0, 0.0, 0.0), 1.225, (0.0, 0.0, 3.0), frames=60)
    edge_on, edge_velocity = _aerodynamic_patch(device, (0.0, 0.0, 0.0), 1.225, (3.0, 0.0, 0.0), frames=60)

    test.assertGreater(face_velocity[2], 1.0, "face-on wind must accelerate the patch downwind")
    # Drag can only approach the wind speed, never exceed it.
    test.assertLess(face_velocity[2], 3.0, "drag must not accelerate cloth past the air velocity")
    test.assertGreater(face_on[2], 0.5, "the patch must travel downwind")
    np.testing.assert_allclose(edge_velocity, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(edge_on, 0.0, atol=1.0e-6)


add_function_test(
    TestSolverXPBD,
    "test_xpbd_spring_compliance_is_iteration_independent",
    test_xpbd_spring_compliance_is_iteration_independent,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_triangle_resists_membrane_strain",
    test_xpbd_cloth_triangle_resists_membrane_strain,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_connected_cloth_constraints_continue_converging",
    test_xpbd_connected_cloth_constraints_continue_converging,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_self_contact_separates_surfaces",
    test_xpbd_cloth_self_contact_separates_surfaces,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_self_contact_prevents_triangle_tunneling",
    test_xpbd_cloth_self_contact_prevents_triangle_tunneling,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_self_contact_recovers_intersection_created_by_constraints",
    test_xpbd_cloth_self_contact_recovers_intersection_created_by_constraints,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_triangle_intersection_recovery_honors_depenetration_limit",
    test_xpbd_triangle_intersection_recovery_honors_depenetration_limit,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_self_contact_recovers_local_triangle_inversion",
    test_xpbd_cloth_self_contact_recovers_local_triangle_inversion,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_self_contact_relaxation_is_independent",
    test_xpbd_cloth_self_contact_relaxation_is_independent,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_external_rigid_solver_preserves_body_state",
    test_xpbd_external_rigid_solver_preserves_body_state,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_shape_adhesion_pulls_particle",
    test_xpbd_shape_adhesion_pulls_particle,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_soft_contact_honors_shape_margin",
    test_xpbd_soft_contact_honors_shape_margin,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_full_surface_soft_contact_blocks_triangle_interior",
    test_xpbd_full_surface_soft_contact_blocks_triangle_interior,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_contact_limits_depenetration_velocity",
    test_xpbd_particle_contact_limits_depenetration_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_overlap_recovery_does_not_accumulate_velocity",
    test_xpbd_cloth_overlap_recovery_does_not_accumulate_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_self_contact_friction_reduces_slip",
    test_xpbd_cloth_self_contact_friction_reduces_slip,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_self_contact_velocity_stabilization_is_inelastic",
    test_xpbd_cloth_self_contact_velocity_stabilization_is_inelastic,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_contact_depenetration_does_not_launch_particle",
    test_xpbd_particle_contact_depenetration_does_not_launch_particle,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_external_body_motion_transfers_friction",
    test_xpbd_external_body_motion_transfers_friction,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_external_body_velocity_transfers_friction_without_pose_delta",
    test_xpbd_external_body_velocity_transfers_friction_without_pose_delta,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_static_soft_contact_enters_static_friction_branch",
    test_xpbd_static_soft_contact_enters_static_friction_branch,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_damping_is_time_step_independent",
    test_xpbd_particle_damping_is_time_step_independent,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_inactive_shape_candidates_do_not_dilute_friction",
    test_xpbd_inactive_shape_candidates_do_not_dilute_friction,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_fully_coupled_rigid_cloth_contact",
    test_xpbd_fully_coupled_rigid_cloth_contact,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_many_particle_contacts_do_not_amplify_body_delta",
    test_xpbd_many_particle_contacts_do_not_amplify_body_delta,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_particle_particle_friction_uses_relative_velocity",
    test_particle_particle_friction_uses_relative_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_distance_joint_limits",
    test_distance_joint_limits,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_ball_joint_recovers_from_large_anchor_separation",
    test_ball_joint_recovers_from_large_anchor_separation,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_prismatic_joint_recovers_from_large_transverse_separation",
    test_prismatic_joint_recovers_from_large_transverse_separation,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_prismatic_joint_retains_extension_in_parent_moment_arm",
    test_prismatic_joint_retains_extension_in_parent_moment_arm,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_optional_control_and_contacts",
    test_optional_control_and_contacts,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_particle_particle_friction_with_relative_motion",
    test_particle_particle_friction_with_relative_motion,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_particle_contact_nan_guard",
    test_xpbd_particle_particle_contact_nan_guard,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_particle_tiny_separation_contact_remains_active",
    test_xpbd_particle_particle_tiny_separation_contact_remains_active,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_particle_shape_restitution_correct_particle",
    test_particle_shape_restitution_correct_particle,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_particle_shape_restitution_accounts_for_body_velocity",
    test_particle_shape_restitution_accounts_for_body_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_rigid_restitution_surface_gate_does_not_double_count_thickness",
    test_rigid_restitution_surface_gate_does_not_double_count_thickness,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_articulation_contact_drift",
    test_articulation_contact_drift,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_static_equilibrium",
    test_xpbd_contact_force_static_equilibrium,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_zero_when_no_contact",
    test_xpbd_contact_force_zero_when_no_contact,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_zero_when_not_touching",
    test_xpbd_contact_force_zero_when_not_touching,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_update_contacts_requires_force_attribute",
    test_xpbd_update_contacts_requires_force_attribute,
    devices=devices,
    check_output=False,
)

for _joint_kind in ("revolute", "ball", "fixed"):
    for _parent in ("world", "kinematic"):
        _name = f"test_xpbd_parent_force_{_joint_kind}_to_{_parent}"
        add_function_test(
            TestSolverXPBD,
            _name,
            globals()[_name],
            devices=devices,
            check_output=False,
        )

add_function_test(
    TestSolverXPBD,
    "test_xpbd_parent_force_chain_weight_propagation",
    test_xpbd_parent_force_chain_weight_propagation,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_parent_force_not_allocated",
    test_xpbd_parent_force_not_allocated,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_parent_force_zero_for_free_body",
    test_xpbd_parent_force_zero_for_free_body,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_parent_f_centripetal_zero_g",
    test_xpbd_parent_f_centripetal_zero_g,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_parent_f_consistent_across_solvers",
    test_xpbd_parent_f_consistent_across_solvers,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_parent_f_newton_second_law_zero_g",
    test_xpbd_parent_f_newton_second_law_zero_g,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_aligned_box_stack_remains_stable",
    test_xpbd_aligned_box_stack_remains_stable,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_friction_is_independent_of_iteration_count",
    test_xpbd_cloth_friction_is_independent_of_iteration_count,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_friction_obeys_coulomb_threshold",
    test_xpbd_cloth_friction_obeys_coulomb_threshold,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_triangle_intersection_recovery_stays_bounded",
    test_xpbd_triangle_intersection_recovery_stays_bounded,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_xpbd_cloth_vertices_do_not_self_repel_without_self_contact",
    test_xpbd_cloth_vertices_do_not_self_repel_without_self_contact,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_xpbd_air_drag_reduces_fall_speed",
    test_xpbd_air_drag_reduces_fall_speed,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_air_drag_is_disabled_by_zero_density",
    test_xpbd_air_drag_is_disabled_by_zero_density,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_air_drag_follows_angle_of_attack",
    test_xpbd_air_drag_follows_angle_of_attack,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
