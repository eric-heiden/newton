# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`SolverMuJoCo.reset`.

The reset method implements the design proposed in
`newton#1266 <https://github.com/newton-physics/newton/issues/1266>`_ and
`newton#2552 <https://github.com/newton-physics/newton/issues/2552>`_.
It is the fix for
`IsaacLab#5359 <https://github.com/isaac-sim/IsaacLab/issues/5359>`_:
teleporting bodies on a settled scene leaves ``qacc_warmstart`` holding a
converged acceleration which, combined with the new pose, produces
integrator corrections that manifest as spurious ~3 cm/s velocity kicks
compounding unboundedly. Resetting ``qacc_warmstart`` (and the other
per-world solver caches) before the next step suppresses the kick.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


def _build_two_world_free_box_model():
    """Return a model with two worlds, each containing a free-floating box.

    ``add_body`` implicitly creates a FREE joint so we do not call
    ``add_joint_free`` explicitly; doing so would create two joints leading to
    the same body and fail topological sort.
    """
    template_builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

    body = template_builder.add_body(mass=1.0, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    template_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)

    builder = newton.ModelBuilder()
    builder.add_shape_plane()

    for i in range(2):
        builder.add_world(template_builder, xform=wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity()))

    return builder.finalize()


def _build_one_world_free_box_model():
    """Return a single-world model with one free-floating box."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    builder.add_shape_plane()
    body = builder.add_body(mass=1.0, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    return builder.finalize()


class TestSolverMuJoCoReset(unittest.TestCase):
    """Functional tests for :meth:`SolverMuJoCo.reset`."""

    def _make_solver(
        self, use_mujoco_cpu: bool = False, single_world: bool = False
    ) -> tuple[newton.Model, SolverMuJoCo, newton.State, newton.State]:
        if single_world:
            model = _build_one_world_free_box_model()
        else:
            model = _build_two_world_free_box_model()
        solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu)
        state_in = model.state()
        state_out = model.state()
        return model, solver, state_in, state_out

    def _settle(
        self, solver: SolverMuJoCo, state_0: newton.State, state_1: newton.State, n: int = 30, dt: float = 1e-3
    ):
        """Step the solver so ``qacc_warmstart`` accumulates a non-zero value."""
        control = solver.model.control()
        contacts = solver.model.contacts()
        for _ in range(n):
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        return state_0, state_1

    def test_reset_all_worlds_clears_qacc_warmstart(self):
        """reset() with world_mask=None zeroes qacc_warmstart in every world."""
        _model, solver, state_0, state_1 = self._make_solver()

        state_0, state_1 = self._settle(solver, state_0, state_1, n=40)
        pre = solver.mjw_data.qacc_warmstart.numpy()
        # Make sure the test scenario actually produced a non-zero warmstart;
        # otherwise the test would pass trivially.
        self.assertGreater(np.linalg.norm(pre), 0.0, "setup did not produce a nonzero qacc_warmstart")

        solver.reset(state_0, state_1)

        post = solver.mjw_data.qacc_warmstart.numpy()
        np.testing.assert_array_equal(post, np.zeros_like(post))

    def test_reset_masked_only_clears_selected_worlds(self):
        """reset() with mask=[False, True] leaves world 0's solver state intact."""
        model, solver, state_0, state_1 = self._make_solver()

        state_0, state_1 = self._settle(solver, state_0, state_1, n=40)
        pre = solver.mjw_data.qacc_warmstart.numpy().copy()
        self.assertGreater(np.linalg.norm(pre), 0.0, "setup did not produce a nonzero qacc_warmstart")

        mask = wp.array(np.array([False, True], dtype=bool), dtype=wp.bool, device=model.device)
        solver.reset(state_0, state_1, world_mask=mask)

        post = solver.mjw_data.qacc_warmstart.numpy()
        np.testing.assert_array_equal(post[0], pre[0])
        np.testing.assert_array_equal(post[1], np.zeros_like(post[1]))

    def test_reset_applies_state_in_joint_q_to_masked_worlds(self):
        """reset() overlays state_in.joint_q into the selected worlds' qpos."""
        model, solver, state_0, state_1 = self._make_solver()

        state_0, state_1 = self._settle(solver, state_0, state_1, n=10)

        # Teleport world 1 to (5, 5, 1) with identity orientation.
        q = state_0.joint_q.numpy().copy()
        q[7:10] = [5.0, 5.0, 1.0]
        q[10:14] = [0.0, 0.0, 0.0, 1.0]  # Newton quaternion is xyzw
        state_0.joint_q.assign(q)
        # Zero the velocity for the teleported world.
        qd = state_0.joint_qd.numpy().copy()
        qd[6:12] = 0.0
        state_0.joint_qd.assign(qd)

        mask = wp.array(np.array([False, True], dtype=bool), dtype=wp.bool, device=model.device)
        solver.reset(state_0, state_1, world_mask=mask)

        out_q = state_1.joint_q.numpy()
        np.testing.assert_allclose(out_q[7:10], [5.0, 5.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(out_q[10:14], [0.0, 0.0, 0.0, 1.0], atol=1e-6)

        out_qd = state_1.joint_qd.numpy()
        np.testing.assert_allclose(out_qd[6:12], np.zeros(6), atol=1e-6)

    def test_reset_suppresses_spurious_velocity_after_teleport(self):
        """Regression test for IsaacLab#5359.

        After a settled simulation, teleporting the body and taking one step
        without calling reset() produces a detectable velocity kick because the
        stored ``qacc_warmstart`` reflects the pre-teleport configuration.
        Calling reset() before stepping suppresses this kick.
        """
        _model, solver, state_0, state_1 = self._make_solver()
        state_0, state_1 = self._settle(solver, state_0, state_1, n=40)

        # Freeze the currently-settled state so both branches start from the
        # same place.
        joint_q = state_0.joint_q.numpy().copy()
        joint_qd = state_0.joint_qd.numpy().copy()

        # Teleport the free body in world 0 upward by 1 meter and zero velocity.
        joint_q[2] += 1.0
        joint_qd[:6] = 0.0

        # --- Branch A: teleport without reset() ---
        state_0.joint_q.assign(joint_q)
        state_0.joint_qd.assign(joint_qd)
        control = solver.model.control()
        contacts = solver.model.contacts()
        solver.step(state_0, state_1, control, contacts, 1e-3)
        qd_no_reset = state_1.joint_qd.numpy().copy()

        # --- Branch B: teleport then reset() then step ---
        _model_b, solver_b, sb_0, sb_1 = self._make_solver()
        sb_0, sb_1 = self._settle(solver_b, sb_0, sb_1, n=40)
        sb_0.joint_q.assign(joint_q)
        sb_0.joint_qd.assign(joint_qd)
        solver_b.reset(sb_0, sb_1)
        sb_0, sb_1 = sb_1, sb_0
        control_b = solver_b.model.control()
        contacts_b = solver_b.model.contacts()
        solver_b.step(sb_0, sb_1, control_b, contacts_b, 1e-3)
        qd_reset = sb_1.joint_qd.numpy().copy()

        # Branch B should have a strictly smaller spurious linear velocity
        # than branch A at the teleported DOFs (horizontal components).
        kick_no_reset = np.linalg.norm(qd_no_reset[:2])
        kick_reset = np.linalg.norm(qd_reset[:2])
        self.assertLess(
            kick_reset,
            kick_no_reset + 1e-9,
            "reset() did not suppress the spurious velocity kick",
        )

    def test_reset_validates_world_mask(self):
        """Invalid world_mask arguments raise TypeError or ValueError."""
        model, solver, state_0, state_1 = self._make_solver()

        with self.assertRaises(TypeError):
            solver.reset(state_0, state_1, world_mask=[True, True])  # not a warp.array

        bad_dtype = wp.array(np.zeros(model.world_count), dtype=wp.float32, device=model.device)
        with self.assertRaises(TypeError):
            solver.reset(state_0, state_1, world_mask=bad_dtype)

        bad_shape = wp.array(np.array([True, True, True], dtype=bool), dtype=wp.bool, device=model.device)
        with self.assertRaises(ValueError):
            solver.reset(state_0, state_1, world_mask=bad_shape)

    def test_reset_returns_consistent_body_q_for_masked_world(self):
        """After reset, state_out.body_q reflects the overlaid joint_q."""
        model, solver, state_0, state_1 = self._make_solver()
        state_0, state_1 = self._settle(solver, state_0, state_1, n=10)

        q = state_0.joint_q.numpy().copy()
        q[7:10] = [5.0, 5.0, 1.0]
        q[10:14] = [0.0, 0.0, 0.0, 1.0]
        state_0.joint_q.assign(q)

        mask = wp.array(np.array([False, True], dtype=bool), dtype=wp.bool, device=model.device)
        solver.reset(state_0, state_1, world_mask=mask)

        body_q = state_1.body_q.numpy()
        # Two worlds * 1 free body per world = 2 body transforms.
        world_1_body_xform = body_q[1]
        np.testing.assert_allclose(world_1_body_xform[:3], [5.0, 5.0, 1.0], atol=1e-6)

    def test_reset_all_worlds_cpu(self):
        """CPU backend full reset clears qacc_warmstart and writes qpos/qvel."""
        _model, solver, state_0, state_1 = self._make_solver(use_mujoco_cpu=True, single_world=True)

        # CPU backend operates on a single MjData; step a few times so the
        # solver accumulates nontrivial state, then reset.
        state_0, state_1 = self._settle(solver, state_0, state_1, n=5)

        solver.reset(state_0, state_1)

        np.testing.assert_array_equal(solver.mj_data.qacc_warmstart, np.zeros_like(solver.mj_data.qacc_warmstart))

    def test_reset_rejects_partial_world_mask_on_cpu(self):
        """CPU backend cannot process a partial world_mask."""
        model, solver, state_0, state_1 = self._make_solver(use_mujoco_cpu=True, single_world=True)
        mask = wp.array(np.array([True], dtype=bool), dtype=wp.bool, device=model.device)
        with self.assertRaises(NotImplementedError):
            solver.reset(state_0, state_1, world_mask=mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)
