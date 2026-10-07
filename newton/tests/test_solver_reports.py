# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""newton.utils.report_solver_params and newton.utils.report_health outside any live session."""

import importlib.util
import re
import unittest
import warnings

import numpy as np
import warp as wp

import newton
import newton.utils
from newton.tests.unittest_utils import StdOutCapture

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

    @unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
    def test_solver_params_kind_aliases(self):
        """'shape' and 'contact' report the geom rows, plurals name their kind, and unknown kinds list the kinds."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(_MJCF)
        solver = SolverMuJoCo(builder.finalize(device="cpu"))
        geoms = newton.utils.report_solver_params(solver, "geom")
        for kind, expected in (("shape", "geom"), ("Contact", "geom"), ("joints", "joint"), ("bodies", "body")):
            with self.subTest(kind=kind):
                report = newton.utils.report_solver_params(solver, kind)
                self.assertEqual(report["kind"], expected)
                if expected == "geom":
                    self.assertEqual(report["rows"], geoms["rows"])
        with self.assertRaisesRegex(ValueError, r"kind must be one of .*'shape' or 'contact' for 'geom'.*got 'tendon'"):
            newton.utils.report_solver_params(solver, "tendon")

    @unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
    def test_actuator_rows_show_the_joint_effort_limit(self):
        """An effort limit on the joint's actfrcrange is reported on the joint's actuators, not as unlimited."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(
            _MJCF.replace('range="-60 60"', 'range="-60 60" actuatorfrcrange="-12 12"').replace(
                "</actuator>", '<motor name="torque" joint="hinge"/></actuator>'
            )
        )
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model)
        report = newton.utils.report_solver_params(solver, "actuator")
        self.assertIn("jnt_actfrcrange", report["per_world"])
        self.assertEqual(len(report["rows"]), 2)
        for row in report["rows"]:
            with self.subTest(actuator=row["label"]):
                self.assertEqual(row["forcerange"], "unlimited")
                self.assertEqual(row["joint_actfrcrange"], [-12.0, 12.0])
                self.assertEqual(row["from"]["joint_actfrcrange"], "+-model.joint_effort_limit[0]")
        model.joint_effort_limit.fill_(5.0)
        for row in newton.utils.report_solver_params(solver, "actuator")["rows"]:
            self.assertIn("joint_actfrcrange", row["pending"])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        for row in newton.utils.report_solver_params(solver, "actuator")["rows"]:
            self.assertEqual(row["joint_actfrcrange"], [-5.0, 5.0])
            self.assertNotIn("pending", row)

    @unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
    def test_inactive_torsional_and_rolling_friction(self):
        """Friction a shape's condim leaves out triggers one warning and is marked on the shape's geom row."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane()
        balls = {
            "default": (None, None, 3),
            "torsional3": (0.02, None, 3),
            "rolling4": (None, 0.01, 4),
            "both6": (0.02, 0.01, 6),
            "zero3": (0.0, 0.0, 3),
        }
        for i, (name, (torsional, rolling, condim)) in enumerate(balls.items()):
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.mu_torsional = cfg.mu_torsional if torsional is None else torsional
            cfg.mu_rolling = cfg.mu_rolling if rolling is None else rolling
            body = builder.add_body(xform=wp.transform(wp.vec3(float(i), 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_sphere(
                body, radius=0.1, cfg=cfg, label=f"{name}/ball", custom_attributes={"mujoco:condim": condim}
            )
        model = builder.finalize(device="cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = SolverMuJoCo(model)
        messages = [str(w.message) for w in caught if "mu_torsional or mu_rolling" in str(w.message)]
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("SolverMuJoCo: 2 shapes", messages[0])
        self.assertIn("'torsional3/ball' (condim 3", messages[0])
        self.assertIn("'rolling4/ball' (condim 4", messages[0])
        self.assertNotIn("default/ball", messages[0])
        self.assertNotIn("both6/ball", messages[0])
        report = newton.utils.report_solver_params(solver, "geom", select="*/ball")
        inactive = {
            model.shape_label[row["shape"]].split("/")[0]: row.get("friction_inactive") for row in report["rows"]
        }
        self.assertEqual(
            inactive,
            {
                "default": ["torsional", "rolling"],
                "torsional3": ["torsional", "rolling"],
                "rolling4": ["rolling"],
                "both6": None,
                "zero3": None,
            },
        )
        self.assertIn("condim >= 4", report["friction_inactive"])

    @unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
    def test_health_skips_overlap_between_static_shapes(self):
        """A welded base overlapping the ground does not fail the health check; a penetrating free body does."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        base = builder.add_link(label="base")
        upright = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi)
        # 30 mm into the ground; unfiltered, so MuJoCo collides the (mocap) base with the plane.
        builder.add_shape_capsule(base, xform=wp.transform(wp.vec3(), upright), radius=0.03, half_height=0.1)
        weld = builder.add_joint_fixed(-1, base, collision_filter_parent=False)
        arm = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()), label="arm")
        builder.add_shape_box(arm, hx=0.1, hy=0.02, hz=0.02)
        hinge = builder.add_joint_revolute(
            base, arm, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()), axis=(0.0, 1.0, 0.0)
        )
        builder.add_articulation([weld, hinge])
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model)
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, model.control(), None, 0.002)
        self.assertGreater(int(solver.mjw_data.nacon.numpy()[0]), 0)
        report = newton.utils.report_health(model, state_1, solver)
        self.assertTrue(report["ok"], report)
        self.assertNotIn("penetration", report)
        self.assertGreater(report["stats"]["penetration_skipped_contacts"]["static_pairs"], 0)
        self.assertEqual(report["stats"]["deepest_penetration"], 0.0)

    def test_health_skips_pairs_filtered_from_colliding(self):
        """Explicit filter pairs, shared bodies, groups, worlds, and non-colliding shapes are skipped; others are kept."""
        from newton._src.mcp.diagnostics import _skipped_pairs  # noqa: PLC0415

        template = newton.ModelBuilder()
        body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        a = template.add_shape_sphere(body, radius=0.1)
        b = template.add_shape_sphere(body, radius=0.1)
        other = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.1), wp.quat_identity()))
        c = template.add_shape_sphere(other, radius=0.1)
        d = template.add_shape_sphere(other, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(collision_group=2))
        e = template.add_shape_sphere(other, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))
        f = template.add_shape_sphere(other, radius=0.1)
        template.add_shape_collision_filter_pair(a, f)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(template, 2)
        model = builder.finalize(device="cpu")
        per_world = template.shape_count
        offset = model.shape_count - 2 * per_world  # the ground plane comes first

        def shape(index, world=0):
            return offset + world * per_world + index

        pairs = {
            "kept": (shape(a), shape(c)),
            "kept_ground": (shape(c), 0),
            "same_body": (shape(a), shape(b)),
            "group": (shape(a), shape(d)),
            "no_collision": (shape(a), shape(e)),
            "explicit": (shape(a), shape(f)),
            "other_world": (shape(a), shape(c, world=1)),
            "unmapped": (-1, shape(c)),
        }
        first = np.array([p[0] for p in pairs.values()])
        second = np.array([p[1] for p in pairs.values()])
        static, filtered = _skipped_pairs(model, first, second)
        self.assertFalse(static.any())
        self.assertEqual(
            {name for name, skip in zip(pairs, filtered, strict=True) if skip},
            {"same_body", "group", "no_collision", "explicit", "other_world"},
        )

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


@unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
class TestMuJoCoWarpOverflowCounts(unittest.TestCase):
    def test_iteration_limits_print_once_and_are_counted(self):
        """MuJoCo Warp's per-world, per-step iteration-limit prints become one line per type and a count."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        template = newton.ModelBuilder()
        tilted = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)
        cube = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), tilted))
        template.add_shape_box(cube, hx=0.05, hy=0.05, hz=0.05)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(template, 2)
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, iterations=1, ls_iterations=1)
        state_0, state_1 = model.state(), model.state()
        steps = 5
        capture = StdOutCapture()
        capture.begin()
        try:
            for _ in range(steps):
                solver.step(state_0, state_1, model.control(), None, 0.002)
                state_0, state_1 = state_1, state_0
            wp.synchronize()
        finally:
            output = capture.end()
        lines = [line for line in output.splitlines() if not line.startswith("Module ")]
        once = " (printed once per solver, newton.utils.report_health() counts every occurrence)"
        self.assertEqual(
            sorted(re.sub(r"in world \d+", "in world W", line) for line in lines),
            [
                "SolverMuJoCo: MuJoCo Warp linesearch iteration limit (ls_iterations 1) reached in world W" + once,
                "SolverMuJoCo: MuJoCo Warp solver iteration limit (iterations 1) reached in world W" + once,
            ],
            output,
        )
        report = newton.utils.report_health(model, state_0, solver)
        # The contact keeps a one-iteration linesearch from converging in every world and step.
        self.assertEqual(report["stats"]["overflow_counts"]["LS_ITERATIONS"], steps * model.world_count)
        self.assertGreater(report["stats"]["overflow_counts"]["ITERATIONS"], 0)
        self.assertEqual(report["worlds"]["overflow_flags"]["1"], ["ITERATIONS", "LS_ITERATIONS"])

    def test_newton_contact_overflow_prints_once_and_counts_losses_per_world(self):
        """Newton contacts past naconmax print one line per solver and are counted per world in report_health."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        template = newton.ModelBuilder()
        for i in range(3):
            box = template.add_body(xform=wp.transform(wp.vec3(0.3 * i, 0.0, 0.049), wp.quat_identity()))
            template.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(template, 4)
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, nconmax=6)
        naconmax = int(solver.mjw_data.naconmax)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        pipeline.collide(state_0, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, naconmax)
        # The worlds of the contacts the conversion drops: each contact has one box, of one world.
        shape_body = model.shape_body.numpy()
        pairs = np.stack([contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()], axis=1)
        boxes = np.max(shape_body[pairs[naconmax:count]], axis=1)
        expected = np.bincount(model.body_world.numpy()[boxes], minlength=model.world_count)

        collisions, substeps = 2, 3
        capture = StdOutCapture()
        capture.begin()
        try:
            for collision in range(collisions):
                if collision:
                    pipeline.collide(state_0, contacts)
                for _ in range(substeps):
                    solver.step(state_0, state_1, control, contacts, 0.002)
                    state_0, state_1 = state_1, state_0
            wp.synchronize()
        finally:
            output = capture.end()
        lines = [line for line in output.splitlines() if "Newton contacts" in line]
        self.assertEqual(len(lines), 1, output)
        self.assertIn(f"{count} Newton contacts exceed the MuJoCo Warp contact buffer (naconmax {naconmax}", lines[0])
        self.assertNotIn("exceeded MJWarp limit", output)

        report = newton.utils.report_health(model, state_0, solver)
        overflow = report["stats"]["newton_contact_overflow"]
        # One count per contact set: substeps reusing a contact set do not count again.
        self.assertEqual(overflow["contact_sets"], collisions)
        self.assertGreaterEqual(overflow["max_contacts"], count)
        self.assertEqual(overflow["naconmax"], naconmax)
        if int(contacts.rigid_contact_count.numpy()[0]) == count:
            self.assertEqual(overflow["contacts_lost"], collisions * (count - naconmax))
            self.assertEqual(
                report["stats"]["contacts_lost_per_world"],
                {str(w): collisions * int(n) for w, n in enumerate(expected) if n},
            )
        self.assertEqual(report["worlds"]["contacts_lost"], [int(w) for w in np.flatnonzero(expected)])
        self.assertTrue(any("contacts were dropped in worlds" in warning for warning in report["warnings"]))

        # No overflow, no report.
        roomy = SolverMuJoCo(model, use_mujoco_contacts=False)
        roomy.step(state_0, state_1, control, contacts, 0.002)
        self.assertNotIn("newton_contact_overflow", newton.utils.report_health(model, state_1, roomy)["stats"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
