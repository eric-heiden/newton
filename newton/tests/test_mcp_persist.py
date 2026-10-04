# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise writing live values and definitions back into hosted scripts, and model diffs."""

import ast
import contextlib
import io
import linecache
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.mcp import ExampleHost, SimulationSession

_SCRIPT = textwrap.dedent(
    '''
    import numpy as np

    import newton

    # Tuned by hand; keep these comments.
    PARAMS = {
        "speed": 1.0,  # m/s
        "gains": [1.0, 2.0, 3.0],  # per axis
        "name": "push",
    }
    SUBSTEPS = 2
    NOTE: str = "é unicode stays"


    def helper(x):
        return 2 * x


    @staticmethod
    def decorated():
        return 1


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body(label="ball")
            builder.add_shape_sphere(body, radius=0.1, label="ball_shape")
            self.model = builder.finalize()
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.speed = PARAMS["speed"]
            self.frame_dt = 0.1

        def step(self):
            """Advance one frame."""
            self.state_0.body_qd.assign(np.array([[self.speed, 0, 0, 0, 0, 0]], dtype=np.float32))
            self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    '''
).lstrip()


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign | ast.AnnAssign)]
    for node in nodes:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class _Base(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.script = self.root / "push.py"
        self.script.write_text(_SCRIPT, encoding="utf-8")
        builder = newton.ModelBuilder()
        builder.add_shape_sphere(builder.add_body(label="ball"), radius=0.1, label="ball_shape")
        model = builder.finalize(device="cpu")
        self.session = SimulationSession(
            model,
            newton.solvers.SolverXPBD(model),
            allow_execute=True,
            artifact_directory=self.root / "artifacts",
            invalidate_on_error=False,
        )
        self.session.source_path = self.script
        self.addCleanup(self.session.close)

    def execute(self, code: str) -> dict:
        return self.session.dispatch("execute", {"code": code})

    def backups(self) -> list[Path]:
        directory = self.root / "artifacts" / "persist"
        return sorted(directory.iterdir()) if directory.exists() else []


class TestMcpPersist(_Base):
    def test_rewrites_only_the_changed_entries_and_keeps_a_backup(self):
        """Patch one dictionary value in place, keep comments and other bytes, and back up the original."""
        original = self.script.read_bytes()
        result = self.execute(
            "persist('PARAMS', {'speed': np.float32(2.5), 'gains': [1.0, 2.0, 3.0], 'name': 'push'}, rebuild=False)"
        )
        expected = _SCRIPT.replace('"speed": 1.0,  # m/s', '"speed": 2.5,  # m/s')
        self.assertEqual(self.script.read_text(encoding="utf-8"), expected)
        self.assertTrue(result["result"]["changed"])
        self.assertEqual(result["result"]["line"], 6)
        self.assertIn('-    "speed": 1.0,  # m/s', result["stdout"])
        self.assertIn('+    "speed": 2.5,  # m/s', result["stdout"])
        self.assertEqual(self.backups()[0].read_bytes(), original)
        self.assertEqual(Path(result["result"]["backup"]), self.backups()[0])
        # Writing the same value again is a no-op without a second backup.
        result = self.execute(
            "persist('PARAMS', {'speed': 2.5, 'gains': [1.0, 2.0, 3.0], 'name': 'push'}, rebuild=False)"
        )
        self.assertFalse(result["result"]["changed"])
        self.assertEqual(len(self.backups()), 1)

    def test_whole_literal_is_reformatted_when_its_structure_changes(self):
        """Replace a dictionary whose keys changed, keeping its multi-line layout and the rest of the file."""
        value = {"speed": 0.5, "gains": np.linspace(0.0, 1.0, 40, dtype=np.float32), "extra": (1,), "flag": None}
        with contextlib.redirect_stdout(io.StringIO()):
            self.session.persist("PARAMS", value, rebuild=False)
        text = self.script.read_text(encoding="utf-8")
        written = _literal(self.script, "PARAMS")
        self.assertEqual(written["gains"], [float(str(x)) for x in value["gains"]])
        self.assertEqual(written["extra"], (1,))
        self.assertIsNone(written["flag"])
        self.assertTrue(all(len(line) <= 120 for line in text.splitlines()))
        prefix, suffix = _SCRIPT.split("PARAMS = {", 1)[0], _SCRIPT.split("}\nSUBSTEPS", 1)[1]
        self.assertTrue(text.startswith(prefix + "PARAMS = {\n"))
        self.assertTrue(text.endswith("}\nSUBSTEPS" + suffix))
        # Scalars, Warp vectors, and annotated assignments.
        self.execute("persist('SUBSTEPS', np.int64(8), rebuild=False)")
        self.execute("persist('NOTE', 'it\\'s \"quoted\"', rebuild=False)")
        self.assertEqual(_literal(self.script, "SUBSTEPS"), 8)
        self.assertEqual(_literal(self.script, "NOTE"), 'it\'s "quoted"')
        self.assertIn("NOTE: str = ", self.script.read_text(encoding="utf-8"))
        self.execute("persist('SUBSTEPS', wp.vec3(1.0, 2.0, 0.1), rebuild=False)")
        self.assertEqual(_literal(self.script, "SUBSTEPS"), [1.0, 2.0, 0.1])

    def test_value_defaults_to_the_module_attribute(self):
        """Write the hosted module's current value when no value is given."""
        module = type(sys)("hosted")
        module.SUBSTEPS = 6
        self.session.namespace["module"] = module
        self.execute("persist('SUBSTEPS', rebuild=False)")
        self.assertEqual(_literal(self.script, "SUBSTEPS"), 6)
        with self.assertRaisesRegex(RuntimeError, "Give a value"):
            self.execute("persist('MISSING_NAME', rebuild=False)")

    def test_refusals_leave_the_file_untouched(self):
        """Refuse missing, repeated, chained, non-literal targets and unrepresentable values without writing."""
        self.script.write_text(
            _SCRIPT
            + textwrap.dedent(
                """
                TWICE = 1
                if True:
                    TWICE = 2
                A = B = 3
                COMPUTED = dict(a=1)
                SCALE = 1.0 / 60.0
                """
            ),
            encoding="utf-8",
        )
        original = self.script.read_bytes()
        cases = {
            "persist('PARAM', {}, rebuild=False)": "no assignment named 'PARAM' at module level; similar names: PARAMS",
            "persist('TWICE', 3, rebuild=False)": "bound 2 times",
            "persist('A', 4, rebuild=False)": "not a plain NAME",
            "persist('COMPUTED', {'a': 2}, rebuild=False)": "is not a literal",
            "persist('SCALE', 0.5, rebuild=False)": "is not a literal",
            "persist('helper', 1, rebuild=False)": "not a plain NAME",
            "persist('SUBSTEPS', object(), rebuild=False)": "cannot be written as a literal",
            "persist('SUBSTEPS', float('nan'), rebuild=False)": "no literal form",
            "persist('SUBSTEPS', 3, check='1')": "rebuild callback",
            "persist('SUBSTEPS', 3, rebuild=False, check='1')": "needs rebuild=True",
            "persist('not valid', 3)": "identifier",
        }
        for code, message in cases.items():
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, message):
                self.execute(code)
        self.assertEqual(self.script.read_bytes(), original)
        self.assertEqual(self.backups(), [])

    def test_line_endings_and_encoding_are_preserved(self):
        """Keep CRLF line endings, a byte-order mark, and non-ASCII text outside the edited span."""
        source = "﻿# Ünïcode header\r\nPARAMS = {\r\n    'k': 1,  # ß\r\n}\r\nOTHER = 'ö'\r\n"
        self.script.write_bytes(source.encode("utf-8"))
        self.execute("persist('PARAMS', {'k': 2}, rebuild=False)")
        self.assertEqual(self.script.read_bytes(), source.replace("'k': 1", "'k': 2").encode("utf-8"))
        self.execute("persist('PARAMS', {'k': 2, 'new': ['x', 'y']}, rebuild=False)")
        data = self.script.read_bytes()
        # The rewritten literal keeps the file's line endings and its single-quote style.
        expected = "PARAMS = {\r\n    'k': 2,\r\n    'new': ['x', 'y'],\r\n}\r\nOTHER = 'ö'\r\n"
        self.assertEqual(data, b"\xef\xbb\xbf# \xc3\x9cn\xc3\xafcode header\r\n" + expected.encode("utf-8"))

    def test_workers_are_rebuilt_after_the_main_scene(self):
        """Rebuild the scene through the rebuild callback, then broadcast a rebuild to worker sessions."""
        calls = []

        class Pool:
            count = 2

            def broadcast(self, code):
                calls.append(code)
                return [None, None]

        def rebuild(session, **_):
            calls.append("main")
            return {"model": session.model, "solver": session.solver}

        self.session.rebuild_callback = rebuild
        self.session.workers = Pool()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.session.persist("SUBSTEPS", 3, check="session.revision > 0")
        self.assertEqual(calls, ["main", "session.dispatch('rebuild', {})"])
        self.assertEqual(result["workers_rebuilt"], 2)
        self.assertTrue(result["check"]["reproduced"])


class TestMcpPersistSource(_Base):
    def test_replaces_functions_and_methods_with_cell_source(self):
        """Replace a top-level function and a class method, re-indenting code but not string contents."""
        result = self.execute(
            "def helper(x):\n    '''Triple.'''\n    return 3 * x\npersist_source(helper, rebuild=False)"
        )
        self.assertEqual(result["result"]["target"], "helper")
        self.assertIn("+    return 3 * x", result["stdout"])
        text = self.script.read_text(encoding="utf-8")
        self.assertIn("def helper(x):\n    '''Triple.'''\n    return 3 * x\n\n\n@staticmethod", text)
        before = text
        result = self.execute(
            textwrap.dedent(
                """
                def my_step(self):
                    banner = '''first
                second'''
                    self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt * GAIN)
                    self.state_0, self.state_1 = self.state_1, self.state_0
                persist_source(my_step, target="Example.step", rebuild=False)
                """
            )
        )
        self.assertEqual(result["result"]["names_not_defined_in_script"], ["GAIN"])
        text = self.script.read_text(encoding="utf-8")
        self.assertIn("    def step(self):\n        banner = '''first\nsecond'''\n        self.solver.step(", text)
        self.assertNotIn("Advance one frame", text)
        self.assertTrue(text.startswith(before.split("    def step(self):")[0]))
        # Decorators of the replaced definition go too; the new source brings its own (here a Warp function).
        result = self.execute("@wp.func\ndef decorated():\n    return 2\npersist_source(decorated, rebuild=False)")
        text = self.script.read_text(encoding="utf-8")
        self.assertNotIn("@staticmethod", text)
        self.assertIn("@wp.func\ndef decorated():\n    return 2\n", text)
        self.assertEqual(result["result"]["names_not_defined_in_script"], ["wp"])
        self.assertEqual(len(self.backups()), 3)

    def test_replaces_classes_defined_in_the_same_cell(self):
        """Read a cell class's source in the defining cell and replace the script's class with it."""
        result = self.execute(
            textwrap.dedent(
                '''
                class Example:
                    """Rewritten."""

                    def __init__(self, viewer, args):
                        self.speed = 2.0

                persist_source(Example, rebuild=False)
                '''
            )
        )
        self.assertEqual(result["result"]["kind"], "ClassDef")
        text = self.script.read_text(encoding="utf-8")
        self.assertTrue(
            text.endswith(
                'class Example:\n    """Rewritten."""\n\n    def __init__(self, viewer, args):\n        self.speed = 2.0\n'
            )
        )
        self.assertTrue(text.startswith(_SCRIPT.split("class Example:")[0]))

    def test_refuses_missing_repeated_and_unsupported_targets(self):
        """Refuse unknown or repeated targets, non-definitions, and lambdas without writing."""
        self.script.write_text(
            _SCRIPT + "\n\ndef helper(x):\n    return x\n\n\nvalue = 1; other = 2\n", encoding="utf-8"
        )
        original = self.script.read_bytes()
        self.execute("def helpr():\n    pass\ndef value():\n    pass\ndef step(self):\n    pass")
        cases = {
            "persist_source(helpr, rebuild=False)": "no def or class named 'helpr' at the top level; similar names: helper",
            "persist_source(value, rebuild=False)": "bound by a Assign statement",
            "persist_source(helpr, target='helper', rebuild=False)": "bound 2 times",
            "persist_source(step, target='Missing.step', rebuild=False)": "no class named 'Missing' at the top level",
            "persist_source(step, target='Example.stop', rebuild=False)": "no def or class named 'stop' in class Example",
            "persist_source(lambda: 1, rebuild=False)": "function or class",
            "persist_source(print, rebuild=False)": "function or class",
            "persist_source('undefined_name', rebuild=False)": "not defined in the workspace",
        }
        for code, message in cases.items():
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, message):
                self.execute(code)
        self.assertEqual(self.script.read_bytes(), original)
        self.assertEqual(self.backups(), [])


class TestMcpCellSource(_Base):
    def test_cell_classes_have_source_and_still_pickle(self):
        """Give classes (and nested classes) defined in cells a source file, and keep them picklable."""
        result = self.execute(
            textwrap.dedent(
                """
                from __future__ import annotations
                import inspect, pickle, typing
                from dataclasses import dataclass

                @dataclass
                class Gains:
                    kp: float = 2.0
                    scale: Scale | None = None

                    class Inner:
                        pass

                Scale = float
                same_cell = inspect.getsource(Gains)
                copy = pickle.loads(pickle.dumps(Gains(3.0)))
                assert typing.get_type_hints(Gains)["scale"] == float | None
                (same_cell.splitlines()[:2], copy.kp, inspect.getsource(Gains.Inner).strip(), Gains.__module__)
                """
            )
        )
        lines, kp, inner, module_name = result["result"]
        self.assertEqual(lines, ["@dataclass", "class Gains:"])
        self.assertEqual(kp, 3.0)
        self.assertEqual(inner, "class Inner:\n        pass")
        self.assertIn(module_name, sys.modules)
        self.assertEqual(self.execute("pickle.loads(pickle.dumps(Gains.Inner)) is Gains.Inner")["result"], True)
        # Sources of live definitions outlast the 64-cell window; unused cells are dropped.
        filename = self.execute("inspect.getsourcefile(Gains)")["result"]
        dropped = self.execute("def dropped():\n    pass\ndropped.__code__.co_filename")["result"]
        self.execute("del dropped")
        for index in range(70):
            self.execute(f"counter = {index}")
        self.assertTrue(linecache.getlines(filename))
        self.assertFalse(linecache.getlines(dropped))
        self.assertEqual(self.execute("inspect.getsource(Gains).splitlines()[1]")["result"], "class Gains:")
        self.session.dispatch("execute", {"code": "", "reset_namespace": True})
        self.assertNotIn(module_name, sys.modules)
        self.assertFalse(linecache.getlines(filename))


class TestMcpDiffModel(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        robot = newton.ModelBuilder()
        link = robot.add_link(label="arm")
        robot.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1, label="box")
        hinge = robot.add_joint_revolute(-1, link, label="hinge")
        robot.add_articulation([hinge], label="robot")
        builder = newton.ModelBuilder()
        builder.replicate(robot, 2)
        model = builder.finalize(device="cpu")
        self.session = SimulationSession(
            model, newton.solvers.SolverXPBD(model), allow_execute=True, artifact_directory=self.directory.name
        )
        self.addCleanup(self.session.close)

    def test_reports_changed_rows_by_label_with_inferred_flags(self):
        """Key changed rows by label and world, infer ModelFlags, and reset the baseline on rebuild."""
        self.assertEqual(self.session.diff_model()["fields"], {})
        old_ke = float(self.session.model.joint_target_ke.numpy()[1])
        self.session.dispatch(
            "edit",
            {
                "patches": [
                    {"field": "joint_target_ke", "indices": [1], "values": [150.0]},
                    {"field": "body_mass", "indices": [0], "values": [2.0]},
                ]
            },
        )
        model = self.session.model
        mu = model.shape_material_mu.numpy()
        mu[:] = 0.25
        model.shape_material_mu.assign(mu)
        model.joint_X_p = wp.clone(model.joint_X_p)
        result = self.session.diff_model()
        fields = result["fields"]
        self.assertEqual(fields["joint_target_ke"]["values"], {"hinge@1": [old_ke, 150.0]})
        self.assertEqual(fields["joint_target_ke"]["flag"], "JOINT_DOF_PROPERTIES")
        self.assertEqual(fields["shape_material_mu"]["changed_rows"], 2)
        self.assertEqual(set(fields["shape_material_mu"]["values"]), {"box@0", "box@1"})
        self.assertEqual(fields["body_mass"]["values"]["arm@0"][1], 2.0)
        self.assertIn("body_inv_mass", fields)
        self.assertIn("body_inertia", fields)
        self.assertEqual(fields["joint_X_p"], {"replaced": True, "flag": "JOINT_PROPERTIES"})
        self.assertEqual(
            result["flags"],
            ["JOINT_PROPERTIES", "JOINT_DOF_PROPERTIES", "BODY_INERTIAL_PROPERTIES", "SHAPE_PROPERTIES"],
        )
        self.assertNotIn("fields_without_flag", result)
        # since="last" compares with the previous call.
        self.assertEqual(self.session.diff_model(since="last")["fields"], {})
        old_mu = model.particle_mu
        model.particle_mu = 0.7
        later = self.session.diff_model(since="last", limit=0)
        self.assertEqual(later["fields"], {"particle_mu": {"values": [old_mu, 0.7], "flag": None}})
        self.assertEqual(later["fields_without_flag"], ["particle_mu"])
        self.session.replace(model, self.session.solver)
        self.assertEqual(self.session.diff_model()["fields"], {})

    def test_large_arrays_are_tracked_by_digest(self):
        """Report changes of arrays beyond the baseline budget without row details."""
        from newton._src.mcp import persist  # noqa: PLC0415

        budget = persist._BASELINE_BYTES
        persist._BASELINE_BYTES = 0
        self.addCleanup(setattr, persist, "_BASELINE_BYTES", budget)
        self.session.replace(self.session.model, self.session.solver)
        ke = self.session.model.joint_target_ke
        ke.assign(np.full(ke.shape, 3.0, dtype=np.float32))
        field = self.session.diff_model()["fields"]["joint_target_ke"]
        self.assertIn("rows were not compared", field["changed"])


class TestMcpPersistHosted(unittest.TestCase):
    def test_persist_rebuilds_and_reports_the_check(self):
        """Persist a live value, rebuild the hosted script, and compare a check expression."""
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "push.py"
            script.write_text(_SCRIPT, encoding="utf-8")
            session = ExampleHost(script).session(artifact_directory=directory)
            try:
                code = "module.PARAMS['speed'] = 2.5\nexample.speed = 2.5\npersist('PARAMS', check='example.speed')"
                result = session.dispatch("execute", {"code": code})["result"]
                self.assertTrue(result["rebuilt"])
                self.assertEqual(result["check"]["live"], 2.5)
                self.assertEqual(result["check"]["rebuilt"], 2.5)
                self.assertTrue(result["check"]["reproduced"])
                self.assertEqual(session.dispatch("execute", {"code": "example.speed"})["result"], 2.5)
                code = "example.speed = 4.0\npersist('PARAMS', {**module.PARAMS, 'speed': 3.0}, check='example.speed')"
                check = session.dispatch("execute", {"code": code})["result"]["check"]
                self.assertFalse(check["reproduced"])
                self.assertAlmostEqual(check["max_abs_difference"], 1.0)
                # A check that raises aborts before anything is written.
                code = "def helper(x):\n    return undefined_name\npersist_source(helper, check='helper(1)')"
                with self.assertRaisesRegex(RuntimeError, "undefined_name"):
                    session.dispatch("execute", {"code": code})
                self.assertIn("return 2 * x", script.read_text(encoding="utf-8"))
                # A definition that breaks the script reports the failed rebuild; the backup stays.
                code = "class Example:\n    def __init__(self, viewer, args):\n        raise ValueError('broken')\n"
                code += "persist_source(Example)"
                with self.assertRaisesRegex(RuntimeError, "rebuilding it failed"):
                    session.dispatch("execute", {"code": code})
                self.assertIn("raise ValueError('broken')", script.read_text(encoding="utf-8"))
                self.assertEqual(len(list((Path(directory) / "persist").iterdir())), 3)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
