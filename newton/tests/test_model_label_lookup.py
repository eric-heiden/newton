# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model.find_* label lookups and the SolverMuJoCo Newton-to-MuJoCo id mappings."""

import importlib.util
import re
import unittest

import numpy as np
import warp as wp

import newton

_HAS_MUJOCO = bool(importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"))


def _robot_model(worlds: int = 2) -> newton.Model:
    """Replicated robot: free root body ``robot/base`` and a revolute ``robot/elbow`` to ``robot/arm``."""
    template = newton.ModelBuilder()
    base = template.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), label="robot/base")
    arm = template.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.5), wp.quat_identity()), label="robot/arm")
    template.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1, label="robot/base_box")
    template.add_shape_box(arm, hx=0.05, hy=0.05, hz=0.25, label="robot/arm_box")
    root = template.add_joint_free(base, label="robot/root")
    elbow = template.add_joint_revolute(
        base,
        arm,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        label="robot/elbow",
    )
    template.add_articulation([root, elbow], label="robot")
    builder = newton.ModelBuilder()
    builder.add_ground_plane(label="ground")
    builder.replicate(template, worlds)
    return builder.finalize(device="cpu")


class TestModelLabelLookup(unittest.TestCase):
    def setUp(self):
        self.model = _robot_model()

    def test_leaf_names_match_in_every_world_or_one(self):
        """A last path component finds one entity per world; ``world`` restricts the search."""
        model = self.model
        joint_world = model.joint_world.numpy()
        elbows = model.find_joints("elbow")
        self.assertEqual(len(elbows), 2)
        self.assertEqual(joint_world[elbows].tolist(), [0, 1])
        self.assertEqual(model.find_joints("elbow", world=1), [elbows[1]])
        self.assertEqual(model.find_joints("robot/elbow"), elbows)  # full label
        self.assertEqual([model.joint_label[j] for j in elbows], ["robot/elbow", "robot/elbow"])

    def test_dofs_and_coords_cover_each_joint(self):
        """DOF and coordinate lookups expand joints by joint_qd_start and joint_q_start."""
        model = self.model
        qd_start, q_start = model.joint_qd_start.numpy(), model.joint_q_start.numpy()
        (root,) = model.find_joints("root", world=1)
        self.assertEqual(model.find_joint_dofs("root", world=1), list(range(qd_start[root], qd_start[root + 1])))
        self.assertEqual(len(model.find_joint_dofs("root", world=1)), 6)
        self.assertEqual(len(model.find_joint_coords("root", world=1)), 7)
        elbow_dofs = model.find_joint_dofs("elbow")
        self.assertEqual(elbow_dofs, [int(qd_start[j]) for j in model.find_joints("elbow")])
        self.assertEqual(model.find_joint_coords("elbow", world=0), [int(q_start[model.find_joints("elbow")[0]])])
        self.assertEqual(model.find_joint_dofs([root]), model.find_joint_dofs("root", world=1))
        self.assertEqual(model.find_joint_dofs(["root", "elbow"], world=0), list(range(0, 7)))

    def test_patterns_regex_lists_and_global_world(self):
        """Globs, regular expressions, and pattern lists select in model order; world -1 holds global shapes."""
        model = self.model
        body_world = model.body_world.numpy()
        bodies = model.find_bodies("robot/*", world=1)
        self.assertEqual(body_world[bodies].tolist(), [1, 1])
        self.assertEqual(model.find_bodies(re.compile(r"(base|arm)"), world=1), bodies)
        self.assertEqual(model.find_bodies(["arm", "base"], world=1), bodies)
        self.assertEqual(model.find_shapes("ground", world=-1), [model.shape_label.index("ground")])
        self.assertEqual(len(model.find_shapes("*_box")), 4)
        self.assertEqual(model.find_bodies([3, 1, 3]), [3, 1])
        self.assertEqual(model.find_bodies([]), [])

    def test_errors_name_the_problem(self):
        """Unmatched patterns name close labels; indices and worlds are validated."""
        model = self.model
        with self.assertRaisesRegex(KeyError, r"no joint label or name matches 'elbw' in world 0.*elbow"):
            model.find_joints("elbw", world=0)
        with self.assertRaisesRegex(KeyError, "'ground'"):
            model.find_shapes("ground", world=0)
        with self.assertRaisesRegex(KeyError, "'hand'"):
            model.find_bodies(["arm", "hand"])
        with self.assertRaisesRegex(ValueError, "world must be None when joint indices are given"):
            model.find_joints([0], world=0)
        with self.assertRaisesRegex(ValueError, r"out of range"):
            model.find_bodies([model.body_count])
        with self.assertRaisesRegex(ValueError, r"world must be None or an integer in \[-1, 1\]"):
            model.find_joints("elbow", world=2)


@unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
class TestSolverMuJoCoNewtonToMuJoCoIds(unittest.TestCase):
    def test_newton_to_mujoco_ids_invert_the_mujoco_to_newton_maps(self):
        """Each Newton body, DOF, and shape maps to the MuJoCo id whose row in every world maps back to it."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        model = _robot_model()
        solver = SolverMuJoCo(model)
        mj = solver.mj_model
        for newton_to_mjc, mjc_to_newton, worlds in (
            (solver.newton_body_to_mjc_body, solver.mjc_body_to_newton, model.body_world),
            (solver.newton_dof_to_mjc_dof, solver.mjc_dof_to_newton_dof, None),
            (solver.newton_shape_to_mjc_geom, solver.mjc_geom_to_newton_shape, model.shape_world),
        ):
            forward, inverse = newton_to_mjc.numpy(), mjc_to_newton.numpy()
            world = worlds.numpy() if worlds is not None else np.repeat([0, 1], len(forward) // 2)
            for index, mjc_id in enumerate(forward):
                if mjc_id < 0:
                    continue
                rows = range(inverse.shape[0]) if world[index] < 0 else [world[index]]
                for row in rows:
                    self.assertEqual(inverse[row, mjc_id], index)
        # Both elbows share one MuJoCo joint id, whose compiled name is the flattened label.
        mjc_dofs = solver.newton_dof_to_mjc_dof.numpy()[model.find_joint_dofs("elbow")]
        self.assertEqual(mjc_dofs[0], mjc_dofs[1])
        self.assertEqual(mj.joint(int(mj.dof_jntid[mjc_dofs[0]])).name, "robot_elbow")
        bodies = solver.newton_body_to_mjc_body.numpy()[model.find_bodies("arm")]
        self.assertEqual({mj.body(int(body)).name for body in bodies}, {"robot_arm"})
        (ground_geom,) = solver.newton_shape_to_mjc_geom.numpy()[model.find_shapes("ground")]
        self.assertGreaterEqual(ground_geom, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
