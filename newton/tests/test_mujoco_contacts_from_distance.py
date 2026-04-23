# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Contacts From Distance (CFD) gradient surrogate.

CFD is an opt-in extension of :class:`~newton.solvers.SolverMuJoCo` based on
Paulus et al., "Differentiable Simulation of Hard Contacts with Soft
Gradients for Learning and Control" (arXiv:2506.14186). The module lives at
``newton._src.solvers.mujoco.contacts_from_distance`` and exposes the public
:class:`newton.solvers.ContactsFromDistance` dataclass.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import ContactsFromDistance, SolverMuJoCo, SolverNotifyFlags


def _build_two_spheres_model(
    gap: float = 0.02,
    radius: float = 0.05,
) -> newton.Model:
    """Two free spheres separated by ``gap`` meters along the x-axis."""
    builder = newton.ModelBuilder()
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

    center_offset = radius + 0.5 * gap
    body_a = builder.add_body(
        xform=wp.transform((-center_offset, 0.0, 0.5), wp.quat_identity()),
        mass=0.1,
    )
    builder.add_shape_sphere(body=body_a, radius=radius, cfg=shape_cfg)

    body_b = builder.add_body(
        xform=wp.transform((center_offset, 0.0, 0.5), wp.quat_identity()),
        mass=0.1,
    )
    builder.add_shape_sphere(body=body_b, radius=radius, cfg=shape_cfg)

    builder.add_ground_plane()
    return builder.finalize()


class TestContactsFromDistanceConfig(unittest.TestCase):
    """Dataclass-level validation and helpers."""

    def test_defaults(self):
        cfd = ContactsFromDistance()
        self.assertEqual(cfd.width, 0.05)
        self.assertEqual(cfd.dmin, 0.0)
        self.assertEqual(cfd.dmax, 0.01)
        self.assertEqual(cfd.midpoint, 0.5)
        self.assertEqual(cfd.power, 2.0)
        self.assertTrue(cfd.enabled)
        self.assertTrue(cfd.straight_through)

    def test_rejects_non_positive_width(self):
        with self.assertRaises(ValueError):
            ContactsFromDistance(width=0.0)
        with self.assertRaises(ValueError):
            ContactsFromDistance(width=-0.1)

    def test_rejects_invalid_impedance(self):
        with self.assertRaises(ValueError):
            ContactsFromDistance(dmin=-0.01)
        with self.assertRaises(ValueError):
            ContactsFromDistance(dmin=0.5, dmax=0.2)
        with self.assertRaises(ValueError):
            ContactsFromDistance(dmax=1.0)

    def test_rejects_invalid_midpoint(self):
        with self.assertRaises(ValueError):
            ContactsFromDistance(midpoint=0.0)
        with self.assertRaises(ValueError):
            ContactsFromDistance(midpoint=1.0)

    def test_rejects_power_below_one(self):
        with self.assertRaises(ValueError):
            ContactsFromDistance(power=0.5)

    def test_rejects_non_positive_timeconst(self):
        with self.assertRaises(ValueError):
            ContactsFromDistance(timeconst=0.0)

    def test_rejects_non_positive_damping_ratio(self):
        with self.assertRaises(ValueError):
            ContactsFromDistance(damping_ratio=0.0)

    def test_solimp_vec_encodes_width(self):
        cfd = ContactsFromDistance(width=0.08, midpoint=0.4, power=3.0)
        solimp = cfd.solimp_vec()
        self.assertAlmostEqual(solimp[0], 0.0)
        self.assertAlmostEqual(solimp[1], 0.01)
        self.assertAlmostEqual(solimp[2], 0.16)
        self.assertAlmostEqual(solimp[3], 0.4)
        self.assertAlmostEqual(solimp[4], 3.0)

    def test_solref_vec(self):
        cfd = ContactsFromDistance(timeconst=0.05, damping_ratio=1.2)
        solref = cfd.solref_vec()
        self.assertAlmostEqual(solref[0], 0.05)
        self.assertAlmostEqual(solref[1], 1.2)


class TestContactsFromDistanceSolver(unittest.TestCase):
    """End-to-end wiring through :class:`SolverMuJoCo`."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _with_cfd_solver(self, **kwargs) -> SolverMuJoCo:
        """Build a solver with a CFD config, silencing the straight-through warning."""
        model = _build_two_spheres_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return SolverMuJoCo(
                model,
                cfd=ContactsFromDistance(**kwargs),
                use_mujoco_contacts=True,
            )

    def test_no_cfd_leaves_model_untouched(self):
        model = _build_two_spheres_model()
        solver = SolverMuJoCo(model, use_mujoco_contacts=True)
        self.assertIsNone(solver.cfd)
        geom_margin = solver.mjw_model.geom_margin.numpy()
        np.testing.assert_array_equal(geom_margin, np.zeros_like(geom_margin))

    def test_cfd_extends_geom_margin(self):
        solver = self._with_cfd_solver(width=0.05, straight_through=False)
        self.assertIsNotNone(solver.cfd)
        geom_margin = solver.mjw_model.geom_margin.numpy()
        self.assertTrue(
            np.all(geom_margin >= 0.05 - 1e-6),
            f"CFD should extend geom_margin to at least width=0.05, got {geom_margin}",
        )

    def test_cfd_overwrites_solref_and_solimp(self):
        solver = self._with_cfd_solver(width=0.04, straight_through=False)
        solref = solver.mjw_model.geom_solref.numpy()
        solimp = solver.mjw_model.geom_solimp.numpy()
        # solref[0]=timeconst=0.02, solref[1]=damping_ratio=1.0 by default.
        self.assertTrue(np.allclose(solref[..., 0], 0.02))
        self.assertTrue(np.allclose(solref[..., 1], 1.0))
        # solimp[2] = 2 * width = 0.08.
        self.assertTrue(np.allclose(solimp[..., 2], 0.08))

    def test_straight_through_warns(self):
        model = _build_two_spheres_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SolverMuJoCo(
                model,
                cfd=ContactsFromDistance(straight_through=True),
                use_mujoco_contacts=True,
            )
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("straight_through=True" in msg for msg in messages),
            f"expected straight_through warning, got {messages}",
        )

    def test_set_cfd_enables_and_disables(self):
        model = _build_two_spheres_model()
        solver = SolverMuJoCo(model, use_mujoco_contacts=True)
        baseline_margin = solver.mjw_model.geom_margin.numpy().copy()

        solver.set_cfd(ContactsFromDistance(width=0.03, straight_through=False))
        self.assertIsNotNone(solver.cfd)
        active_margin = solver.mjw_model.geom_margin.numpy()
        self.assertTrue(np.all(active_margin >= 0.03 - 1e-6))

        solver.set_cfd(None)
        self.assertIsNone(solver.cfd)
        restored_margin = solver.mjw_model.geom_margin.numpy()
        np.testing.assert_array_equal(restored_margin, baseline_margin)

    def test_disabled_cfd_is_no_op(self):
        model = _build_two_spheres_model()
        solver = SolverMuJoCo(
            model,
            cfd=ContactsFromDistance(enabled=False),
            use_mujoco_contacts=True,
        )
        self.assertIsNone(solver.cfd)
        geom_margin = solver.mjw_model.geom_margin.numpy()
        np.testing.assert_array_equal(geom_margin, np.zeros_like(geom_margin))

    def test_cfd_requires_mujoco_contacts(self):
        model = _build_two_spheres_model()
        with self.assertRaises(ValueError):
            SolverMuJoCo(
                model,
                cfd=ContactsFromDistance(straight_through=False),
                use_mujoco_contacts=False,
            )

    def test_notify_shape_properties_preserves_cfd(self):
        solver = self._with_cfd_solver(width=0.06, straight_through=False)
        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)
        geom_margin = solver.mjw_model.geom_margin.numpy()
        self.assertTrue(
            np.all(geom_margin >= 0.06 - 1e-6),
            f"CFD margins should survive notify_model_changed, got {geom_margin}",
        )
        solimp = solver.mjw_model.geom_solimp.numpy()
        self.assertTrue(np.allclose(solimp[..., 2], 0.12))

    def test_short_rollout_with_cfd(self):
        """A few steps with CFD active should run without NaNs or errors."""
        model = _build_two_spheres_model()
        solver = self._with_cfd_solver(width=0.05, straight_through=False)

        state_in = model.state()
        state_out = model.state()
        control = model.control()

        for _ in range(3):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, 1.0 / 240.0)
            state_in, state_out = state_out, state_in

        body_q = state_in.body_q.numpy()
        self.assertFalse(np.any(np.isnan(body_q)), "CFD rollout produced NaN body_q")


if __name__ == "__main__":
    unittest.main()
