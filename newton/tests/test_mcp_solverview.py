# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""solver_params, model-edit detection around trusted execution, and per-world health checks."""

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.mcp.session import SimulationSession as _Session
from newton._src.mcp.solverview import FIELD_FLAGS, flag_names
from newton.mcp import ExampleHost, SimulationSession

_HAS_MUJOCO = bool(importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"))

# A position servo (JOINT_TARGET), a motor (CTRL_DIRECT), a free cube above a floor.
_MJCF = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
<body name="arm" pos="0 0 1"><joint name="hinge" type="hinge" axis="0 1 0" range="-60 60"/>
<geom name="link" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/></body>
<body name="wheel" pos="1 0 1"><joint name="spin" type="hinge" axis="0 1 0"/>
<geom name="disc" type="cylinder" size="0.1 0.02" mass="0.5"/></body>
<body name="box" pos="0.5 0 0.2"><freejoint/><geom name="cube" type="box" size="0.05 0.05 0.05" mass="0.2"/></body>
<geom name="floor" type="plane" size="2 2 0.1"/>
</worldbody><actuator><position name="servo" joint="hinge" kp="40" kv="2"/>
<motor name="drive" joint="spin" gear="2"/></actuator></mujoco>"""


def _mujoco_model(device: str, worlds: int = 2, *, gravity: bool = True):
    from newton.solvers import SolverMuJoCo  # noqa: PLC0415

    template = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(template)
    template.add_mjcf(_MJCF if gravity else _MJCF.replace("0 0 -9.81", "0 0 0"))
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template, worlds)
    return builder.finalize(device=device)


def _devices() -> list[str]:
    return ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]


def _session(model, solver, directory, **kwargs):
    session = SimulationSession(model, solver, dt=0.01, allow_execute=True, artifact_directory=directory, **kwargs)
    return session


def _hinge_actuator(rows):
    return next(row for row in rows if row["label"] == "hinge")


@unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
class TestMcpSolverParams(unittest.TestCase):
    def setUp(self):
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = _mujoco_model("cpu")
        self.solver = SolverMuJoCo(self.model)
        self.session = _session(self.model, self.solver, self.directory.name)
        self.addCleanup(self.session.close)

    def test_actuator_rows_name_sources_and_pending_edits(self):
        """Report each actuator's compiled gains with the model array they come from and unapplied edits."""
        report = self.session.solver_params("actuator")
        self.assertEqual(report["solver"], "SolverMuJoCo")
        servo = _hinge_actuator(report["rows"])
        self.assertEqual(servo["ctrl_source"], "JOINT_TARGET")
        self.assertEqual(servo["gainprm"][0], 40.0)
        self.assertEqual(servo["from"]["gainprm[0]"], f"model.joint_target_ke[{servo['dof']}]")
        self.assertNotIn("pending", servo)
        motor = next(row for row in report["rows"] if row["ctrl_source"] == "CTRL_DIRECT")
        self.assertTrue(motor["from"]["gainprm"].startswith("model.mujoco.actuator_gainprm["))
        self.assertIn("JOINT_TARGET", report["not_read"])
        # An edit without notify_model_changed is visible as pending, and disappears once applied.
        self.model.joint_target_ke.fill_(80.0)
        servo = _hinge_actuator(self.session.solver_params("actuator")["rows"])
        self.assertEqual(servo["gainprm"][0], 40.0)
        self.assertIn("gainprm[0]", servo["pending"])
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        servo = _hinge_actuator(self.session.solver_params("actuator")["rows"])
        self.assertEqual(servo["gainprm"][0], 80.0)
        self.assertNotIn("pending", servo)

    def test_selection_world_and_other_kinds(self):
        """Select rows by label pattern and world, and report joints, geoms, bodies and options."""
        dofs_per_world = self.model.joint_dof_count // 2
        world0 = self.session.solver_params("joint", select="hin*")["rows"]
        world1 = self.session.solver_params("joint", select="hinge", world=1)["rows"]
        self.assertEqual([row["label"] for row in world0], ["hinge"])
        self.assertEqual(world1[0]["dof"], world0[0]["dof"] + dofs_per_world)
        self.assertEqual(world0[0]["from"]["damping"], f"model.joint_damping[{world0[0]['dof']}]")
        geoms = self.session.solver_params("geom", select="cube")
        self.assertEqual([row["label"] for row in geoms["rows"]], ["cube"])
        cube = geoms["rows"][0]["shape"]
        self.assertAlmostEqual(geoms["rows"][0]["friction"][0], float(self.model.shape_material_mu.numpy()[cube]), 5)
        self.assertFalse(geoms["per_world"]["geom_priority"])
        self.assertIn("model.shape_material_restitution", geoms["not_read"])
        bodies = self.session.solver_params("body", select="box")["rows"]
        self.assertAlmostEqual(bodies[0]["mass"], 0.2, places=5)
        self.assertEqual(bodies[0]["from"]["gravcomp"], f"model.mujoco.gravcomp[{bodies[0]['body']}]")
        options = self.session.solver_params("option")
        self.assertEqual(options["options"]["iterations"], 100)
        self.assertTrue(options["refsafe"])
        self.assertIn("gravity", options["refreshed_by"])
        self.assertEqual(self.session.solver_params("equality")["rows"], [])
        with self.assertRaises(ValueError):
            self.session.solver_params("tendons")
        with self.assertRaises(ValueError):
            self.session.solver_params("joint", world=2)

    def test_geom_solref_follows_material_and_pending(self):
        """Show the solref computed from shape_material_ke/kd and flag material edits not yet applied."""
        row = self.session.solver_params("geom", select="cube")["rows"][0]
        shape = row["shape"]
        ke, kd = self.model.shape_material_ke.numpy()[shape], self.model.shape_material_kd.numpy()[shape]
        if ke > 0 and kd > 0:
            np.testing.assert_allclose(row["solref"], [2.0 / kd, kd / 2.0 * np.sqrt(1.0 / ke)], rtol=1e-5)
        mu = self.model.shape_material_mu.numpy()
        mu[shape] = 0.123
        self.model.shape_material_mu.assign(mu)
        row = self.session.solver_params("geom", select="cube")["rows"][0]
        self.assertEqual(row["pending"], ["friction"])

    def test_other_solvers_report_model_values(self):
        """Fall back to Newton model arrays and say that compiled values are unavailable."""
        builder = newton.ModelBuilder()
        body = builder.add_link()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label="block")
        joint = builder.add_joint_revolute(-1, body, target_ke=5.0, label="pivot")
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        session = _session(model, newton.solvers.SolverXPBD(model), self.directory.name)
        self.addCleanup(session.close)
        report = session.solver_params("actuator")
        self.assertIn("does not expose compiled solver parameters", report["unsupported"])
        self.assertEqual(report["rows"][0]["joint_target_ke"], 5.0)
        self.assertEqual(session.solver_params("geom", select="block")["rows"][0]["label"], "block")
        self.assertIn("options", session.solver_params("option"))


@unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
class TestMcpModelWatch(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def make(self, device: str):
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        model = _mujoco_model(device)
        solver = SolverMuJoCo(model)
        session = _session(model, solver, self.directory.name)
        self.addCleanup(session.close)
        return session

    @staticmethod
    def gain(session) -> float:
        return _hinge_actuator(session.solver_params("actuator")["rows"])["gainprm"][0]

    def test_edits_before_rollout_are_notified_with_inferred_flags(self):
        """Notify edits no notify_model_changed covered before the cell's rollout runs, and say so."""
        for device in _devices():
            with self.subTest(device=device):
                session = self.make(device)
                result = session.dispatch("execute", {"code": "model.joint_target_ke.fill_(80.0)\nr = rollout(2)"})
                self.assertIn("before rollout()", result["note"])
                self.assertIn("model.joint_target_ke", result["note"])
                self.assertIn(
                    "solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)", result["note"]
                )
                self.assertEqual(self.gain(session), 80.0)
                # Without the check the rollout runs with the compiled gain of 40.
                session.watch.mode = "off"
                result = session.dispatch("execute", {"code": "model.joint_target_ke.fill_(20.0)\nr = rollout(2)"})
                self.assertNotIn("note", result)
                self.assertEqual(self.gain(session), 80.0)

    def test_report_mode_and_covered_notifications(self):
        """Report without notifying, and stay silent when the cell's own notify_model_changed covered an edit."""
        session = self.make("cpu")
        session.watch.mode = "report"
        result = session.dispatch("execute", {"code": "model.joint_target_ke.fill_(70.0)"})
        self.assertIn("keeps its previous values", result["note"])
        self.assertEqual(self.gain(session), 40.0)
        session.watch.mode = "notify"
        code = "model.joint_target_ke.fill_(60.0)\nsolver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)"
        result = session.dispatch("execute", {"code": code})
        self.assertNotIn("note", result)
        self.assertEqual(self.gain(session), 60.0)
        # The narrow force flag covers the gain edit as well.
        code = "model.joint_target_ke.fill_(50.0)\nsolver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)"
        result = session.dispatch("execute", {"code": code})
        self.assertNotIn("note", result)
        self.assertEqual(self.gain(session), 50.0)
        self.assertNotIn("note", session.dispatch("execute", {"code": "x = 1"}))

    def test_edits_written_into_the_solver_directly_are_not_reported(self):
        """Stay silent when the cell also wrote the compiled values itself, as controllers that set gains do."""
        session = self.make("cpu")
        servo = _hinge_actuator(session.solver_params("actuator")["rows"])["actuator"]
        code = (
            "model.joint_target_ke.fill_(70.0)\n"
            "gain, bias = solver.mjw_model.actuator_gainprm.numpy(), solver.mjw_model.actuator_biasprm.numpy()\n"
            f"gain[:, {servo}, 0] = 70.0\nbias[:, {servo}, 1] = -70.0\n"
            "solver.mjw_model.actuator_gainprm.assign(gain)\nsolver.mjw_model.actuator_biasprm.assign(bias)\n"
            "rollout(1)"
        )
        result = session.dispatch("execute", {"code": code})
        self.assertNotIn("no notify_model_changed call covered", result.get("note") or "")
        self.assertEqual(self.gain(session), 70.0)
        # Writing only one world's compiled values leaves the edit pending, which is reported and notified.
        code = (
            "model.joint_target_ke.fill_(30.0)\n"
            "gain = solver.mjw_model.actuator_gainprm.numpy()\n"
            f"gain[0, {servo}, 0] = 30.0\n"
            "solver.mjw_model.actuator_gainprm.assign(gain)\n"
            "rollout(1)"
        )
        result = session.dispatch("execute", {"code": code})
        self.assertIn("no notify_model_changed call covered", result["note"])

    def test_wrong_flag_is_reported_and_completed(self):
        """Complete a gravcomp edit notified with BODY_PROPERTIES, which does not refresh inertial data."""
        session = self.make("cpu")
        code = (
            "g = model.mujoco.gravcomp.numpy(); g[:] = 1.0; model.mujoco.gravcomp.assign(g)\n"
            "solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)"
        )
        note = session.dispatch("execute", {"code": code})["note"]
        self.assertIn("calls in this interval used BODY_PROPERTIES", note)
        self.assertIn("newton.ModelFlags.BODY_INERTIAL_PROPERTIES", note)
        self.assertTrue(all(row["gravcomp"] == 1.0 for row in session.solver_params("body")["rows"]))

    def test_notifications_count_only_on_the_live_solver_after_the_edit(self):
        """Do not count a notify_model_changed call made before the edit or on another solver as covering it."""
        session = self.make("cpu")
        flag = "newton.ModelFlags.JOINT_DOF_PROPERTIES"
        code = f"solver.notify_model_changed({flag})\nmodel.joint_damping.fill_(5.0)\nrollout(1)"
        note = session.dispatch("execute", {"code": code})["note"]
        self.assertIn("a call before the last edit of model.joint_damping", note)
        self.assertIn("the host called solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)", note)
        joints = session.solver_params("joint")["rows"]
        self.assertTrue(joints and all(row["damping"] == 5.0 and "pending" not in row for row in joints))
        code = (
            "scratch = newton.solvers.SolverMuJoCo(model)\nmodel.joint_damping.fill_(7.0)\n"
            f"scratch.notify_model_changed({flag})\nrollout(1)"
        )
        note = session.dispatch("execute", {"code": code})["note"]
        self.assertIn("a call on a SolverMuJoCo that is not the session's solver", note)
        self.assertTrue(all(row["damping"] == 7.0 for row in session.solver_params("joint")["rows"]))
        code = f"model.joint_damping.fill_(9.0)\nsolver.notify_model_changed({flag})\nrollout(1)"
        self.assertNotIn("note", session.dispatch("execute", {"code": code}))

    def test_gain_edits_on_a_solver_without_actuators_are_stated(self):
        """State that joint target gains drive nothing when the MuJoCo model has no actuators."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        template = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template)
        template.add_mjcf(_MJCF.split("<actuator>", maxsplit=1)[0] + "</mujoco>")
        model = template.finalize(device="cpu")
        session = _session(model, SolverMuJoCo(model), self.directory.name)
        self.addCleanup(session.close)
        self.assertEqual(session.solver_params("actuator")["rows"], [])
        note = session.dispatch("execute", {"code": "model.joint_target_ke.fill_(123.0)"})["note"]
        self.assertIn("model.joint_target_ke rows", note)
        self.assertIn("drive no MuJoCo actuator (this SolverMuJoCo has no actuators", note)

    def test_facts_name_fields_the_solver_does_not_read(self):
        """State that JOINT_TARGET actuators ignore mujoco.actuator_gainprm and non-RAW shapes ignore mujoco.solref."""
        session = self.make("cpu")
        servo = _hinge_actuator(session.solver_params("actuator")["rows"])
        code = textwrap.dedent(
            """
            gain = model.mujoco.actuator_gainprm.numpy(); gain[:, 0] *= 3; model.mujoco.actuator_gainprm.assign(gain)
            solref = model.mujoco.solref.numpy(); solref[:] = (0.05, 1.0); model.mujoco.solref.assign(solref)
            model.joint_velocity_limit.fill_(1.0)
            solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES | newton.ModelFlags.SHAPE_PROPERTIES)
            """
        )
        note = session.dispatch("execute", {"code": code})["note"]
        self.assertIn("ctrl_source JOINT_TARGET", note)
        self.assertIn("mujoco.solref_mode is not RAW", note)
        self.assertIn("SolverMuJoCo does not read joint_velocity_limit", note)
        self.assertEqual(_hinge_actuator(session.solver_params("actuator")["rows"])["gainprm"], servo["gainprm"])

    def test_edits_made_by_the_application_step_are_not_attributed(self):
        """Exclude model edits the application's own step makes from the cell's report."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        model = _mujoco_model("cpu")
        solver = SolverMuJoCo(model)

        def step(session, dt):
            model.joint_damping.assign(model.joint_damping.numpy() + 0.01)
            solver.step(session.state, session.state_next, session.control, None, dt)
            session.state, session.state_next = session.state_next, session.state

        session = _session(model, solver, self.directory.name, step_callback=step)
        self.addCleanup(session.close)
        result = session.dispatch("execute", {"code": "r = rollout(3)"})
        self.assertNotIn("note", result)
        result = session.dispatch("execute", {"code": "r = rollout(3)\nmodel.joint_armature.fill_(0.01)"})
        self.assertIn("model.joint_armature", result["note"])
        self.assertNotIn("joint_damping", result["note"])

    def test_inferred_flags_agree_with_structured_edits(self):
        """Keep the structured edit operation's flag inference consistent with the watch table."""
        for field, flag in _Session._EDIT_FLAGS.items():
            self.assertEqual(FIELD_FLAGS[field], flag, field)
        self.assertEqual(
            flag_names(newton.ModelFlags.SHAPE_PROPERTIES | newton.ModelFlags.MODEL_PROPERTIES),
            "SHAPE_PROPERTIES | MODEL_PROPERTIES",
        )

    def test_other_solvers_are_notified_too(self):
        """Notify any solver class and record the notify_model_changed calls of new solver objects."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        session = _session(model, newton.solvers.SolverXPBD(model), self.directory.name)
        self.addCleanup(session.close)
        note = session.dispatch("execute", {"code": "model.shape_material_mu.fill_(0.3)\nr = rollout(1)"})["note"]
        self.assertIn("newton.ModelFlags.SHAPE_PROPERTIES", note)
        result = session.dispatch("execute", {"code": "model.body_mass.fill_(2.0)\nsolver.notify_model_changed(8)"})
        self.assertNotIn("note", result)


_HOSTED_SCRIPT = textwrap.dedent(
    """
    import warp as wp

    import newton
    from newton.solvers import SolverMuJoCo


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            SolverMuJoCo.register_custom_attributes(builder)
            link = builder.add_link()
            builder.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
            joint = builder.add_joint_revolute(-1, link, axis=newton.Axis.Z, target_ke=5.0, target_kd=1.0)
            builder.add_articulation([joint])
            builder.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
            self.model = builder.finalize()
            self.solver = SolverMuJoCo(self.model, disable_contacts=True)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.control.joint_target_q.fill_(1.0)
            self.frame_dt = 0.02
            self.graph = None
            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

        def simulate(self):
            for _ in range(2):
                self.solver.step(self.state_0, self.state_1, self.control, None, 0.5 * self.frame_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

        def step(self):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
    """
)


_GRAVITY_SCRIPT = textwrap.dedent(
    """
    import newton


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_shape_sphere(builder.add_body(), radius=0.1)
            self.model = builder.finalize(device="cpu")
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.frame_dt = 0.1
            self.ticks = 0

        def step(self):
            # The application's own model edit, every frame.
            self.ticks += 1
            self.model.gravity.assign([[0.0, 0.0, -9.81 * (self.ticks % 2)]])
            self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    """
)


class TestMcpHostedStepChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "gravity_steps.py"
        script.write_text(_GRAVITY_SCRIPT)
        self.host = ExampleHost(script)
        self.session = self.host.session(artifact_directory=directory.name)
        self.addCleanup(self.session.close)

    def execute(self, code: str) -> dict:
        return self.session.dispatch("execute", {"code": code})

    def test_model_edits_of_the_application_step_are_not_attributed_to_cells(self):
        """Check around example.step() called from a cell as around a dispatched step."""
        self.assertNotIn("note", self.execute("for _ in range(3):\n    example.step()"))
        self.assertNotIn("note", self.execute("session.dispatch('step', {'count': 3})"))
        note = self.execute("model.body_mass.fill_(2.0)\nexample.step()")["note"]
        self.assertIn("model.body_mass [1 rows] changed before example.step(); no notify_model_changed call", note)
        self.assertNotIn("gravity", note)

    def test_stepping_checks_skip_settings_comparison_and_leave_checksums_unread(self):
        """Keep the per-step cost of the checks to a binding refresh and an asynchronous checksum launch."""
        deep = [0]
        fingerprint = self.host.fingerprint

        def counted(**kwargs):
            deep[0] += kwargs.get("deep", True)
            return fingerprint(**kwargs)

        self.host.fingerprint = counted
        self.execute(
            "pending = []\nfor _ in range(10):\n    session.dispatch('step')\n"
            "    pending.append(session.watch._pending is not None)"
        )
        # host.sync (first step of a batch) and end_batch per dispatched step, plus one after the cell.
        self.assertLessEqual(deep[0], 2 * 10 + 1)
        self.assertEqual(self.execute("pending")["result"], [True] * 10)


@unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
class TestMcpHostedModelWatch(unittest.TestCase):
    def test_graph_captured_example_sees_unnotified_gain_edits(self):
        """Make an un-notified gain edit change a CUDA-graph-captured MuJoCo rollout in the same cell."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "servo.py"
        script.write_text(_HOSTED_SCRIPT)
        session = ExampleHost(script).session(artifact_directory=directory.name)
        self.addCleanup(session.close)
        code = textwrap.dedent(
            """
            angles = {}
            for ke in (5.0, 50.0):
                model.joint_target_ke.fill_(ke)
                rollout(10, start=True)
                angles[ke] = float(state.joint_q.numpy()[0])
            angles
            """
        )
        result = session.dispatch("execute", {"code": code})
        angles = {float(k): v for k, v in result["result"].items()}
        self.assertGreater(angles[50.0], angles[5.0] + 0.05)
        self.assertIn("newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES", result["note"])


@unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
class TestMcpHealthPerWorld(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_nonfinite_worlds_twins_and_any_solver(self):
        """Name the worlds with non-finite state or diverging twins, for a solver passed in explicitly."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        model = _mujoco_model("cpu", worlds=3, gravity=False)
        session = _session(model, SolverMuJoCo(model), self.directory.name)
        self.addCleanup(session.close)
        other = newton.solvers.SolverXPBD(model)
        state = model.state()
        report = session.health(solver=other, state=state, twins=True)
        self.assertTrue(report["ok"], report)
        self.assertIn("SolverXPBD: no solver contact or constraint buffers to check", report["unsupported"])
        body_q = state.body_q.numpy()
        body_q[model.body_world.numpy() == 2, 0] = np.nan
        state.body_q.assign(body_q)
        report = session.health(solver=other, state=state)
        self.assertFalse(report["ok"])
        self.assertEqual(report["worlds"]["nonfinite"], [2])
        joint_q = session.state.joint_q.numpy()
        coord_world = np.repeat(model.joint_world.numpy(), np.diff(model.joint_q_start.numpy()))
        joint_q[np.flatnonzero(coord_world == 1)[0]] += 0.1
        session.state.joint_q.assign(joint_q)
        report = session.health(twins=True)
        self.assertEqual(report["worlds"]["twins_disagree"], [1])
        self.assertAlmostEqual(report["stats"]["twins_max_deviation"], 0.1, places=5)

    def test_penetrating_shape_pairs_are_named(self):
        """Report the shape pair and world behind a deep MuJoCo contact."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        model = _mujoco_model("cpu", worlds=1)
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        state = model.state()
        body_q = state.body_q.numpy()
        box = model.body_label.index(next(label for label in model.body_label if label.endswith("box")))
        body_q[box, 2] = 0.02  # cube half-size 0.05: 30 mm into the floor
        state.body_q.assign(body_q)
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        session = _session(model, solver, self.directory.name, state=state)
        self.addCleanup(session.close)
        session.dispatch("step", {"count": 1})
        report = session.health()
        self.assertFalse(report["ok"])
        pair = report["penetration"][0]
        self.assertEqual(sorted(pair["shapes"]), ["cube", "floor"])
        self.assertGreater(pair["depth"], 0.01)
        self.assertEqual(pair["worlds"], [0])
        # The MuJoCo CPU backend reports its compiled values from mj_model.
        self.assertEqual(_hinge_actuator(session.solver_params("actuator")["rows"])["gainprm"][0], 40.0)
        self.assertEqual(session.solver_params("option")["backend"], "mujoco (CPU)")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_mujoco_warp_buffer_overflow_per_world(self):
        """Name the worlds whose constraint rows reach njmax and report a full contact buffer."""
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        model = _mujoco_model("cuda:0", worlds=2)
        state = model.state()
        body_q = state.body_q.numpy()
        boxes = [i for i, label in enumerate(model.body_label) if label.endswith("box")]
        body_q[boxes[1], 2] = 0.04  # only world 1's cube touches the floor
        state.body_q.assign(body_q)
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        solver = SolverMuJoCo(model, njmax=2, nconmax=1)
        session = _session(model, solver, self.directory.name, state=state)
        self.addCleanup(session.close)
        session.dispatch("step", {"count": 1})
        report = session.health()
        self.assertFalse(report["ok"])
        self.assertEqual(report["worlds"]["constraint_buffer_full"], [1])
        self.assertTrue(any("contact buffer full" in warning for warning in report["warnings"]), report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
