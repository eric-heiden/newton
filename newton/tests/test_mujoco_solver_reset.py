# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`SolverMuJoCo.reset`."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverStateFlags


def _apply_state_attributes(builder: newton.ModelBuilder, attributes: tuple[str, ...]) -> None:
    """Request optional state attributes on a builder."""
    if attributes:
        builder.request_state_attributes(*attributes)


def _build_two_world_free_box_model(state_attributes: tuple[str, ...] = ()) -> newton.Model:
    """Return a two-world model with one free-floating box per world."""
    template_builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

    body = template_builder.add_body(mass=1.0, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    template_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)

    builder = newton.ModelBuilder()
    _apply_state_attributes(builder, state_attributes)
    builder.add_shape_plane()

    for i in range(2):
        builder.add_world(template_builder, xform=wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity()))

    return builder.finalize()


def _build_two_world_kinematic_box_model(state_attributes: tuple[str, ...] = ()) -> newton.Model:
    """Return a two-world model with one kinematic free box per world."""
    template_builder = newton.ModelBuilder(gravity=0.0)
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

    body = template_builder.add_body(
        mass=1.0,
        xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        is_kinematic=True,
    )
    template_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)

    builder = newton.ModelBuilder(gravity=0.0)
    _apply_state_attributes(builder, state_attributes)

    for i in range(2):
        builder.add_world(template_builder, xform=wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity()))

    return builder.finalize()


def _build_one_world_free_box_model(state_attributes: tuple[str, ...] = ()) -> newton.Model:
    """Return a single-world model with one free-floating box."""
    builder = newton.ModelBuilder()
    _apply_state_attributes(builder, state_attributes)
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    builder.add_shape_plane()
    body = builder.add_body(mass=1.0, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    return builder.finalize()


class TestSolverMuJoCoReset(unittest.TestCase):
    """Contract tests for :meth:`SolverMuJoCo.reset`."""

    def _make_solver(
        self,
        *,
        use_mujoco_cpu: bool = False,
        single_world: bool = False,
        kinematic_worlds: bool = False,
        state_attributes: tuple[str, ...] = (),
    ) -> tuple[newton.Model, SolverMuJoCo, newton.State, newton.State]:
        if kinematic_worlds:
            model = _build_two_world_kinematic_box_model(state_attributes)
        elif single_world:
            model = _build_one_world_free_box_model(state_attributes)
        else:
            model = _build_two_world_free_box_model(state_attributes)

        solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu)
        state_in = model.state()
        state_out = model.state()
        return model, solver, state_in, state_out

    def _settle(
        self,
        solver: SolverMuJoCo,
        state_0: newton.State,
        state_1: newton.State,
        *,
        n: int = 30,
        dt: float = 1e-3,
    ) -> tuple[newton.State, newton.State]:
        """Step the solver until warm-start data is non-trivial."""
        control = solver.model.control()
        contacts = solver.model.contacts()
        for _ in range(n):
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        return state_0, state_1

    def test_reset_all_worlds_clears_qacc_warmstart(self):
        """reset() with ``world_mask=None`` zeroes qacc_warmstart everywhere."""
        _model, solver, state_0, state_1 = self._make_solver()

        state_0, state_1 = self._settle(solver, state_0, state_1, n=40)
        pre = solver.mjw_data.qacc_warmstart.numpy()
        self.assertGreater(np.linalg.norm(pre), 0.0, "setup did not produce a nonzero qacc_warmstart")

        solver.reset(state_0)

        post = solver.mjw_data.qacc_warmstart.numpy()
        np.testing.assert_array_equal(post, np.zeros_like(post))

    def test_reset_joint_flags_update_state_in_place(self):
        """Joint flags re-apply ``joint_q`` / ``joint_qd`` into the same state."""
        _model, solver, state_0, state_1 = self._make_solver()

        state_0, state_1 = self._settle(solver, state_0, state_1, n=10)

        joint_q = state_0.joint_q.numpy().copy()
        joint_q[7:10] = [5.0, 5.0, 1.0]
        joint_q[10:14] = [0.0, 0.0, 0.0, 1.0]
        state_0.joint_q.assign(joint_q)

        joint_qd = state_0.joint_qd.numpy().copy()
        joint_qd[6:12] = 0.0
        state_0.joint_qd.assign(joint_qd)

        solver.reset(state_0, flags=SolverStateFlags.JOINT_POS | SolverStateFlags.JOINT_VEL)

        np.testing.assert_allclose(state_0.joint_q.numpy()[7:14], joint_q[7:14], atol=1e-6)
        np.testing.assert_allclose(state_0.joint_qd.numpy()[6:12], joint_qd[6:12], atol=1e-6)

    def test_reset_body_flags_recompute_joint_state(self):
        """Body flags recover joint coordinates via inverse kinematics."""
        _model, solver, state_0, state_1 = self._make_solver(single_world=True)

        state_0, state_1 = self._settle(solver, state_0, state_1, n=5)

        body_q = state_0.body_q.numpy().copy()
        body_q[0, :3] = [2.0, -1.0, 1.5]
        body_q[0, 3:] = [0.0, 0.0, 0.0, 1.0]
        state_0.body_q.assign(body_q)

        body_qd = state_0.body_qd.numpy().copy()
        body_qd[0] = [0.5, -0.25, 0.75, 0.1, 0.2, -0.3]
        state_0.body_qd.assign(body_qd)

        solver.reset(state_0, flags=SolverStateFlags.BODY_POS | SolverStateFlags.BODY_VEL)

        np.testing.assert_allclose(state_0.body_q.numpy()[0], body_q[0], atol=1e-6)
        np.testing.assert_allclose(state_0.joint_q.numpy()[:7], body_q[0], atol=1e-6)
        np.testing.assert_allclose(state_0.joint_qd.numpy()[:6], body_qd[0], atol=1e-6)

    def test_reset_preserves_unreset_kinematic_world_state(self):
        """Masked resets keep untouched kinematic worlds on the cached solver state."""
        model, solver, state_0, _state_1 = self._make_solver(kinematic_worlds=True)

        original_q = state_0.joint_q.numpy().copy()
        edited_q = original_q.copy()
        edited_q[:3] = [9.0, 9.0, 9.0]
        edited_q[7:10] = [5.0, 5.0, 1.0]
        state_0.joint_q.assign(edited_q)

        mask = wp.array(np.array([False, True], dtype=bool), dtype=wp.bool, device=model.device)
        solver.reset(state_0, world_mask=mask, flags=SolverStateFlags.JOINT_POS)

        result_q = state_0.joint_q.numpy()
        np.testing.assert_allclose(result_q[:7], original_q[:7], atol=1e-6)
        np.testing.assert_allclose(result_q[7:14], edited_q[7:14], atol=1e-6)

    def test_reset_clears_extended_outputs(self):
        """reset() clears derived outputs invalidated by the solver reset."""
        state_attributes = ("body_qdd", "body_parent_f", "mujoco:qfrc_actuator")
        _model, solver, state_0, state_1 = self._make_solver(single_world=True, state_attributes=state_attributes)

        control = solver.model.control()
        contacts = solver.model.contacts()
        solver.step(state_0, state_1, control, contacts, 1e-3)
        state_0, state_1 = state_1, state_0
        self.assertGreater(np.linalg.norm(state_0.body_qdd.numpy()), 0.0)

        state_0.body_parent_f.assign(np.ones_like(state_0.body_parent_f.numpy()))
        state_0.mujoco.qfrc_actuator.assign(np.ones_like(state_0.mujoco.qfrc_actuator.numpy()))

        solver.reset(state_0)

        np.testing.assert_array_equal(state_0.body_qdd.numpy(), np.zeros_like(state_0.body_qdd.numpy()))
        np.testing.assert_array_equal(state_0.body_parent_f.numpy(), np.zeros_like(state_0.body_parent_f.numpy()))
        np.testing.assert_array_equal(
            state_0.mujoco.qfrc_actuator.numpy(),
            np.zeros_like(state_0.mujoco.qfrc_actuator.numpy()),
        )

    def test_reset_validates_world_mask(self):
        """Invalid ``world_mask`` arguments raise ``TypeError`` or ``ValueError``."""
        model, solver, state_0, _state_1 = self._make_solver()

        with self.assertRaises(TypeError):
            solver.reset(state_0, world_mask=[True, True])  # not a warp.array

        bad_dtype = wp.array(np.zeros(model.world_count), dtype=wp.float32, device=model.device)
        with self.assertRaises(TypeError):
            solver.reset(state_0, world_mask=bad_dtype)

        bad_shape = wp.array(np.array([True, True, True], dtype=bool), dtype=wp.bool, device=model.device)
        with self.assertRaises(ValueError):
            solver.reset(state_0, world_mask=bad_shape)

    def test_reset_all_worlds_cpu_accepts_true_mask(self):
        """CPU reset accepts an all-``True`` mask as a full reset."""
        model, solver, state_0, state_1 = self._make_solver(use_mujoco_cpu=True, single_world=True)

        state_0, state_1 = self._settle(solver, state_0, state_1, n=5)

        mask = wp.array(np.array([True], dtype=bool), dtype=wp.bool, device=model.device)
        solver.reset(state_0, world_mask=mask)

        np.testing.assert_array_equal(solver.mj_data.qacc_warmstart, np.zeros_like(solver.mj_data.qacc_warmstart))

    def test_reset_rejects_partial_world_mask_on_cpu(self):
        """CPU reset rejects masks that are not equivalent to a full reset."""
        model, solver, state_0, _state_1 = self._make_solver(use_mujoco_cpu=True, single_world=True)

        mask = wp.array(np.array([False], dtype=bool), dtype=wp.bool, device=model.device)
        with self.assertRaises(NotImplementedError):
            solver.reset(state_0, world_mask=mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)
