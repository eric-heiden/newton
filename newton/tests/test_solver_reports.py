# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""newton.utils.report_solver_params and newton.utils.report_health outside any live session."""

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
import newton.utils

_HAS_MUJOCO = bool(importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"))

_MJCF = """<mujoco><worldbody>
<body name="arm" pos="0 0 1"><joint name="hinge" type="hinge" axis="0 1 0" range="-60 60"/>
<geom name="link" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/></body>
</worldbody><actuator><position name="servo" joint="hinge" kp="40" kv="2"/></actuator></mujoco>"""


class TestSolverReports(unittest.TestCase):
    @unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
    def test_solver_params_name_sources_and_unapplied_edits(self):
        """Report a MuJoCo servo's compiled gain, its model source, and an edit no notification applied."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(_MJCF)
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model)
        model.joint_target_ke.fill_(80.0)
        report = newton.utils.report_solver_params(solver, "actuator", select="hinge")
        (row,) = report["rows"]
        self.assertEqual(report["solver"], "SolverMuJoCo")
        self.assertEqual(row["gainprm"][0], 40.0)
        self.assertEqual(row["from"]["gainprm[0]"], "model.joint_target_ke[0]")
        self.assertIn("gainprm[0]", row["pending"])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        (row,) = newton.utils.report_solver_params(solver, "actuator")["rows"]
        self.assertEqual(row["gainprm"][0], 80.0)
        self.assertNotIn("pending", row)

    def test_health_names_non_finite_worlds_and_diverging_twins(self):
        """Name the world with non-finite state and the world that left its identical twins, with no solver."""
        template = newton.ModelBuilder()
        body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        template.add_shape_sphere(body, radius=0.1)
        builder = newton.ModelBuilder()
        builder.replicate(template, 3)
        model = builder.finalize(device="cpu")
        initial, state = model.state(), model.state()
        self.assertTrue(newton.utils.report_health(model, state)["ok"])
        q = state.joint_q.numpy()
        q[model.joint_q_start.numpy()[1]] += 0.5
        state.joint_q.assign(q)
        report = newton.utils.report_health(model, state, twins=True, initial_state=initial)
        self.assertEqual(report["worlds"]["twins_disagree"], [1])
        body_q = state.body_q.numpy()
        body_q[2, 0] = np.nan
        state.body_q.assign(body_q)
        report = newton.utils.report_health(model, state)
        self.assertFalse(report["ok"])
        self.assertEqual(report["worlds"]["nonfinite"], [2])
        self.assertIn("No solver given", report["unsupported"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
