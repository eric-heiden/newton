# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interplay of the live MCP features: rollback, model-edit checks, and the instructions agents receive."""

import importlib.util
import re
import tempfile
import types
import unittest
from pathlib import Path

import newton
import newton.examples
from newton._src.mcp import protocol
from newton.mcp import ExampleHost, SimulationSession

_HAS_MUJOCO = bool(importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"))

_MJCF = """<mujoco><worldbody>
<body name="arm" pos="0 0 1"><joint name="hinge" type="hinge" axis="0 1 0"/>
<geom name="link" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/></body>
</worldbody><actuator><position name="servo" joint="hinge" kp="40" kv="2"/></actuator></mujoco>"""


@unittest.skipUnless(_HAS_MUJOCO, "Requires sim extra")
class TestMcpRollbackWithEditChecks(unittest.TestCase):
    def setUp(self):
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(_MJCF)
        self.model = builder.finalize(device="cpu")
        self.session = SimulationSession(
            self.model,
            SolverMuJoCo(self.model),
            dt=0.01,
            allow_execute=True,
            artifact_directory=self.directory.name,
        )
        self.addCleanup(self.session.close)

    def execute(self, code: str) -> dict:
        return self.session.dispatch("execute", {"code": code})

    def servo_gain(self) -> tuple[float, bool]:
        row = next(row for row in self.session.solver_params("actuator")["rows"] if row["label"] == "hinge")
        return row["gainprm"][0], "pending" in row

    def test_failed_cell_undoes_an_edit_the_host_notified(self):
        """A failed cell restores an edit that was auto-notified before its rollout, and the solver follows."""
        with self.assertRaisesRegex(RuntimeError, r"ValueError.*rolled back.*joint_target_ke") as raised:
            self.execute("model.joint_target_ke.fill_(80.0)\nrollout(2)\nraise ValueError('after the rollout')")
        self.assertIn("JOINT_DOF_PROPERTIES", str(raised.exception))
        self.assertEqual(float(self.model.joint_target_ke.numpy()[0]), 40.0)
        self.assertEqual(self.servo_gain(), (40.0, False))
        # The restored values are the new baseline: the next cell reports no edits.
        self.assertNotIn("note", self.execute("rollout(1)\nNone"))
        # A later edit without notify_model_changed is still caught and applied.
        result = self.execute("model.joint_target_ke.fill_(20.0)\nrollout(1)\nNone")
        self.assertRegex(result["note"], r"model\.joint_target_ke .*notify_model_changed")
        self.assertEqual(self.servo_gain(), (20.0, False))


_HINGE_SCRIPT = """
import newton


class Example:
    def __init__(self, viewer, args):
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([builder.add_joint_revolute(-1, body, axis=(0.0, 1.0, 0.0))])
        self.model = builder.finalize(device="cpu")
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.frame_dt = 0.01

    def step(self):
        self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
"""


class TestMcpRebuildInsideACell(unittest.TestCase):
    def test_edits_after_a_rebuild_in_the_same_cell_are_checked(self):
        """A rebuild dispatched from a cell (as persist() does) restarts the model-edit checks on the new model."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "hinge.py"
        script.write_text(_HINGE_SCRIPT)
        session = ExampleHost(script).session(artifact_directory=directory.name)
        self.addCleanup(session.close)
        result = session.dispatch(
            "execute", {"code": "session.dispatch('rebuild', {})\nmodel.joint_target_ke.fill_(7.0)\nNone"}
        )
        self.assertNotIn("stopped", result["note"])
        self.assertRegex(result["note"], r"model\.joint_target_ke .*changed in this cell")


class TestMcpInstructions(unittest.TestCase):
    """The text agents receive: tool semantics for every helper, no removed names, no workflow advice."""

    def setUp(self):
        script = Path(newton.examples.get_source_directory()) / "basic" / "example_basic_pendulum.py"
        self.host = ExampleHost(script)
        # The guide only needs the example's frame time; no scene is built.
        self.host.example = types.SimpleNamespace(frame_dt=1.0 / 60.0)
        self.lean = protocol._INSTRUCTIONS_LEAN.replace("<<RTX>>", "")
        self.guide = self.host.guide(2, 4)
        tools = {tool["name"]: tool for tool in protocol._tools(240.0)}
        self.execute_description = tools["newton_execute"]["description"]
        self.descriptions = self.execute_description + "\n" + tools["newton_rebuild"]["description"]

    def test_every_helper_is_described(self):
        """Name each documented helper in the instructions or the hosted guide, with its arguments."""
        undocumented = ("persist", "persist_source")
        for name in SimulationSession._HELPERS:
            if name not in undocumented:
                self.assertRegex(self.lean + self.guide, rf"\b{name}\(", name)
        # Worker pools, jobs and writing values back to the script work, but the guide does not describe them.
        for pattern in (r"\bpersist(_source)?\(", r"\bworkers\b", r"\bjobs\."):
            self.assertNotRegex(self.lean + self.guide, pattern)
        for fact in ("overrides", "code", "example.reset()", "only files on disk persist"):
            self.assertIn(fact, self.guide)

    def test_execute_description_states_the_reply_limit(self):
        """State the reply limit as a fact when the adapter has one, and nothing without it."""
        self.assertIn("Calls reply within 240 s", self.execute_description)
        unlimited = {tool["name"]: tool for tool in protocol._tools(None)}["newton_execute"]["description"]
        self.assertNotIn("reply within", unlimited)
        self.assertNotIn("<<", unlimited)

    def test_removed_features_and_advice_are_absent(self):
        """Drop removed helpers, structured operations, and workflow coaching from the instructions."""
        text = "\n".join((self.lean, protocol._INSTRUCTIONS, self.guide, self.descriptions))
        for removed in (
            *("compare_images", "plot=", "recovery", "'query'", "'edit'", "recapture()"),
            *("fresh(", "swap_solver", "diff_model"),
        ):
            self.assertNotIn(removed, text)
        self.assertNotRegex(self.execute_description, r"\b(query|edit|contacts|collide|record|play|pause)\b")
        advice = r"(?i)\b(prefer|instead of|efficient|use it only|batch many|should|rather than|re-simulat)"
        self.assertIsNone(re.search(advice, text), re.search(advice, text))

    def test_hosted_guide_fits_the_instruction_budget(self):
        """Keep the hosted guide and the lean instructions short: agents re-read them on every turn."""
        self.assertLess(len(self.guide), 1000)
        self.assertLess(len(self.lean) + len(self.guide), 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
