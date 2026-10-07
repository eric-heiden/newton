# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live batched evaluation: evaluate, branch, and checkpoints with Python objects in a hosted script."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.mcp.batch import _structure_key
from newton.mcp import ExampleHost
from newton.tests.unittest_utils import add_function_test, get_test_devices

_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import warp as wp

    import newton

    DEVICE = None
    SUBSTEPS = 2
    GAIN = 2.0
    SIZE = 0.05
    EXTRA_BODY = False
    WORLDS = 1
    UNRELATED = 0
    LAZY_SOLVER = False


    class Controller:
        \"\"\"Pushes the puck toward a target speed; its integral term and history are its state.\"\"\"

        def __init__(self, device):
            self.gain = GAIN
            self.target = 1.0
            self.integral = 0.0
            self.history = []
            self.settings = {"limit": 5.0}
            self.trace = np.zeros(2)
            self.buffer = wp.zeros(3, dtype=float, device=device)

        def compute(self, speed):
            self.integral += self.target - speed
            self.history.append(speed)
            self.trace[0] += 1.0
            return float(np.clip(self.gain * (self.target - speed) + 0.1 * self.integral, -5.0, 5.0))


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder()
            builder.add_ground_plane(label="floor")
            for _ in range(WORLDS):
                world = newton.ModelBuilder()
                puck = world.add_body(xform=wp.transform((0.0, 0.0, SIZE), wp.quat_identity()), label="puck")
                world.add_shape_box(puck, hx=SIZE, hy=SIZE, hz=SIZE, label="puck_geom")
                if EXTRA_BODY:
                    ball = world.add_body(xform=wp.transform((1.0, 0.0, 0.2), wp.quat_identity()), label="ball")
                    world.add_shape_sphere(ball, radius=0.05, label="ball_geom")
                builder.add_world(world)
            self.model = builder.finalize(device=DEVICE)
            if LAZY_SOLVER:
                # A solver class created while the scene is built, as by a solver module imported on first use.
                class LazySolver(newton.solvers.SolverXPBD):
                    def __init__(self, model, *, iterations=1, flag=False):
                        super().__init__(model, iterations=iterations)
                        self.flag = flag

                self.solver = LazySolver(self.model, iterations=3, flag=True)
            else:
                self.solver = newton.solvers.SolverXPBD(self.model, iterations=4)
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.frame_dt = 1.0 / 60.0
            self.sim_dt = self.frame_dt / SUBSTEPS
            self.controller = Controller(self.model.device)
            self.sim_time = 0.0

        def step(self):
            force = self.controller.compute(float(self.state_0.body_qd.numpy()[0, 0])) if self.controller.gain else 0.0
            for _ in range(SUBSTEPS):
                self.state_0.clear_forces()
                if force:
                    f = self.state_0.body_f.numpy()
                    f[0, 0] = force
                    self.state_0.body_f.assign(f)
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.frame_dt
    """
)

_HELPERS = textwrap.dedent(
    """
    def setup(world, mu, push):
        world.set_model("shape_material_mu", mu, labels=["puck_geom", "floor"])
        world.set_state("joint_qd", [push, 0.0, 0.0, 0.0, 0.0, 0.0], labels="puck*")

    def score(records, cases):
        x = records["puck"][:, :, 0, 0]
        return {"slide": x[-1] - x[0]}

    PUCK = {"puck": ("body_q", "puck")}
    """
)


class _Hosted:
    """A hosted copy of the test script with ``overrides`` (a new directory per instance)."""

    def __init__(self, test, device, overrides=None):
        self.directory = tempfile.TemporaryDirectory()
        test.addCleanup(self.directory.cleanup)
        self.script = Path(self.directory.name) / "puck.py"
        self.script.write_text(_SCRIPT)
        self.host = ExampleHost(self.script, overrides={"DEVICE": str(device), **(overrides or {})})
        self.session = self.host.session(artifact_directory=self.directory.name)
        test.addCleanup(self.session.close)
        self.execute(_HELPERS)

    def execute(self, code: str) -> dict:
        return self.session.dispatch("execute", {"code": code})

    def value(self, code: str):
        return self.execute(code)["result"]


def test_evaluate_reuses_models_and_copies_live_values(test, device):
    hosted = _Hosted(test, device)
    session = hosted.session
    body_q = session.state.body_q.numpy().copy()
    call = "evaluate({'mu=0.1': 0.1, 'mu=0.8': 0.8}, {'slow': 1.0, 'fast': 2.5}, frames=20, setup=setup, score=score, record=PUCK, every=10)"
    text = hosted.value(f"first = {call}\nfirst")
    test.assertIsInstance(text, str)
    test.assertIn("evaluate: 4 case(s) from the live state at t=0 s; 20 frames of 0.0166667 s (0.00833333 s x 2)", text)
    test.assertIn("model: 4 world(s), built in", text)
    test.assertIn("(first use); solver SolverXPBD(model, iterations=4); collision CollisionPipeline(model)", text)
    rows = hosted.value("[dict(r) for r in first.rows]")
    slide = {(row["candidate"], row["scenario"]): row["slide"] for row in rows}
    test.assertLess(slide[("mu=0.8", "fast")], slide[("mu=0.1", "fast")])
    test.assertLess(slide[("mu=0.1", "slow")], slide[("mu=0.1", "fast")])
    # The live session did not move.
    test.assertEqual((session.time, session.frame), (0.0, 0))
    np.testing.assert_array_equal(session.state.body_q.numpy(), body_q)

    again = hosted.value(f"second = {call}\nsecond")
    test.assertIn("model: 4 world(s), reused;", again)
    if wp.get_device(device).is_cuda:
        test.assertIn("(cached CUDA graphs)", again)
    test.assertEqual(hosted.value("[dict(r) for r in second.rows]"), rows)

    # A live edit of the scene's friction reaches the kept worlds.
    one = "evaluate([None], None, frames=20, setup=lambda w, c, s: w.set_state('joint_qd', [2.0, 0, 0, 0, 0, 0], labels='puck*'), score=score, record=PUCK)"
    before = hosted.value(f"{one}.rows[0]['slide']")
    result = hosted.execute(
        "model.shape_material_mu.fill_(0.9)\nsolver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)\n" + one
    )
    test.assertIn("reused", result["result"])
    test.assertIn("live model values copied into the worlds: shape_material_mu", result["result"])
    test.assertNotIn("note", result)
    test.assertLess(hosted.value("_.rows[0]['slide']"), before)  # _ is the last evaluation


def test_rebuilds_keep_models_of_the_same_scene(test, device):
    hosted = _Hosted(test, device)
    call = "evaluate([0.5], [1.0, 2.0], frames=10, setup=setup, score=score, record=PUCK).facts"
    test.assertIn("first use", hosted.value(call)["model"])
    # A rebuild that changes code but not the scene reuses the model.
    device_override = {"DEVICE": str(device)}  # a rebuild's overrides replace the active set
    hosted.session.dispatch("rebuild", {"overrides": {**device_override, "UNRELATED": 1}})
    test.assertEqual(hosted.value(call)["model"], "reused")
    # Changed values are copied into the kept worlds.
    hosted.session.dispatch("rebuild", {"overrides": {**device_override, "SIZE": 0.06}})
    facts = hosted.value(call)
    test.assertEqual(facts["model"], "reused")
    test.assertIn("shape_scale", facts["copied"])
    # Another structure builds a new model and says why.
    hosted.session.dispatch("rebuild", {"overrides": {**device_override, "EXTRA_BODY": True}})
    test.assertRegex(hosted.value(call)["model"], r"built in .* \(the scene's structure or integer values changed")
    # So does a batch that needs more worlds than the kept model has.
    more = call.replace("[1.0, 2.0]", "[1.0, 2.0, 3.0, 4.0, 5.0]")
    test.assertIn("the kept model has 2 worlds", hosted.value(more)["model"])


def test_branch_from_the_live_state_and_a_checkpoint(test, device):
    hosted = _Hosted(test, device, {"GAIN": 0.0})
    session = hosted.session
    hosted.execute(
        "session.dispatch('step', {'count': 6})\ncheckpoint('early')\nsession.dispatch('step', {'count': 4})"
    )
    body_q = session.state.body_q.numpy().copy()
    time_before = session.time
    result = hosted.value(
        "b = branch(3, lambda w, i: w.set_model('shape_material_mu', [0.0, 0.3, 0.9][i], labels=['puck_geom', 'floor']), "
        "frames=12, record=PUCK, every=4, score=lambda r: {'x': r['puck'][-1, :, 0, 0]})\nb"
    )
    test.assertIn("branch: 3 variant(s) from the live state at t=0.166667 s; 12 frames", result)
    test.assertEqual(hosted.value("list(b.records['puck'].shape)"), [4, 3, 1, 7])
    test.assertAlmostEqual(hosted.value("float(b.t[0])"), time_before, places=6)
    test.assertEqual(session.time, time_before)
    np.testing.assert_array_equal(session.state.body_q.numpy(), body_q)

    # Variants without edits continue the live trajectory as the session's own steps do.
    hosted.execute(
        "same = branch(2, frames=8, record=PUCK, start='early')\n"
        "live = rollout(8, record={'x': lambda: state.body_q.numpy()[0, :3].copy()}, start='early')"
    )
    branched = np.asarray(hosted.value("same.records['puck'][-1, :, 0, :3].tolist()"))
    live = np.asarray(hosted.value("live['x'][-1].tolist()"))
    np.testing.assert_allclose(branched[0], branched[1], atol=0.0)
    np.testing.assert_allclose(branched[0], live, atol=2e-4)
    test.assertIn("checkpoint 'early' (t=0.1 s)", hosted.value("same.format()"))


def test_branches_start_from_the_saved_body_state(test, device):
    """Worlds start from a checkpoint's body poses and velocities, not the model's (vector-valued arrays too)."""
    hosted = _Hosted(test, device)
    session = hosted.session
    hosted.execute("session.dispatch('step', {'count': 6})\ncheckpoint('moving')")
    body_q = session.state.body_q.numpy().copy()
    body_qd = session.state.body_qd.numpy().copy()
    test.assertGreater(abs(body_q[0, 0]), 1.0e-3)  # the controller pushed the puck
    hosted.execute("session.dispatch('step', {'count': 4})")
    hosted.execute("b = branch(2, frames=1, start='moving', record={'q': 'body_q', 'qd': 'body_qd'})")
    for name, expected in (("q", body_q), ("qd", body_qd)):
        start = np.asarray(hosted.value(f"b.records[{name!r}][0].tolist()"))
        for variant in range(2):
            np.testing.assert_allclose(start[variant], expected, atol=1.0e-6)
    x0 = hosted.value(
        "evaluate([None], None, frames=1, start='moving', record={'q': 'body_q'}, "
        "score=lambda r, c: {'x0': r['q'][0, :, 0, 0]}).rows[0]['x0']"
    )
    test.assertAlmostEqual(x0, float(body_q[0, 0]), places=6)


def test_checkpoint_restores_python_objects_in_place(test, device):
    hosted = _Hosted(test, device)
    hosted.execute("session.dispatch('step', {'count': 3})")
    saved = hosted.value("checkpoint('c', include=['example.controller'])['objects']")
    test.assertEqual(saved["saved"], ["example.controller"])
    test.assertEqual(saved["arrays"], 2)
    hosted.execute(
        "c = example.controller\nbuffer = c.buffer\n"
        "c.gain = 7.0; c.history.append(99.0); c.settings['limit'] = 1.0; c.extra = 1; c.trace[:] = 5.0\n"
        "c.buffer.fill_(3.0); c.integral = -1.0\nsession.dispatch('step', {'count': 2})"
    )
    status = hosted.value("session.dispatch('restore', {'name': 'c'})")
    test.assertEqual(status["frame"], 3)
    for name in ("extra", "gain", "integral", "history", "settings", "buffer", "trace"):
        test.assertTrue(any(name in entry for entry in status["objects_restored"]), name)
    c = hosted.host.example.controller
    test.assertEqual((c.gain, len(c.history), c.settings["limit"], hasattr(c, "extra")), (2.0, 3, 5.0, False))
    test.assertEqual(c.trace.tolist(), [3.0, 0.0])
    test.assertEqual(c.buffer.numpy().tolist(), [0.0, 0.0, 0.0])
    test.assertTrue(hosted.value("buffer is example.controller.buffer"))
    # rollout(start=...) restores them too.
    hosted.execute("example.controller.gain = 4.0\nrollout(2, start='c')")
    test.assertEqual(c.gain, 2.0)
    with test.assertRaisesRegex(RuntimeError, "has no variable 'missing'"):
        hosted.execute("checkpoint('d', include=['missing'])")


def test_sequential_branch_resumes_python_objects(test, device):
    hosted = _Hosted(test, device)
    session = hosted.session
    hosted.execute("session.dispatch('step', {'count': 4})\ncheckpoint('c', include=['example.controller'])")
    hosted.execute("session.dispatch('step', {'count': 3})")
    controller = hosted.host.example.controller
    state = (controller.integral, len(controller.history), session.time)
    mu = float(session.model.shape_material_mu.numpy()[0])
    text = hosted.value(
        "def vary(world, i):\n"
        "    example.controller.gain = [1.0, 4.0, 4.0][i]\n"
        "    world.set_model('shape_material_mu', 0.2, labels=['puck_geom', 'floor'])\n"
        "s = branch(3, vary, start='c', frames=6, sequential=True, record={**PUCK, 'integral': lambda: example.controller.integral})\n"
        "s"
    )
    test.assertIn("ran one after another through example.step()", text)
    test.assertIn("example.controller (from the checkpoint)", text)
    integral = np.asarray(hosted.value("s.records['integral'].tolist()"))
    puck = np.asarray(hosted.value("s.records['puck'][:, :, 0, 0].tolist()"))
    # Every variant resumes the controller as the checkpoint saved it.
    test.assertEqual(integral.shape, (7, 3))
    np.testing.assert_allclose(integral[0], integral[0, 0])
    test.assertNotAlmostEqual(integral[-1, 0], integral[-1, 1])
    np.testing.assert_array_equal(puck[:, 1], puck[:, 2])
    # The session, the controller, and the model are as before the call.
    test.assertEqual((controller.integral, len(controller.history), session.time), state)
    test.assertEqual(controller.gain, 2.0)
    test.assertEqual(float(session.model.shape_material_mu.numpy()[0]), mu)
    with test.assertRaisesRegex(RuntimeError, "schedules apply to branches run as worlds"):
        hosted.execute("branch(1, lambda w, i: w.set_schedule('joint_target_q', [0.0]), frames=1, sequential=True)")


def test_scene_problems_are_reported(test, device):
    hosted = _Hosted(test, device, {"WORLDS": 2})
    with test.assertRaisesRegex(RuntimeError, "the hosted scene has 2 worlds"):
        hosted.execute("evaluate([1], None, frames=1, score=score)")
    hosted = _Hosted(test, device)
    hosted.execute("session.solver = example.solver = newton.solvers.SolverXPBD(model, iterations=2)")
    with test.assertRaisesRegex(RuntimeError, "session.solver was replaced"):
        hosted.execute("evaluate([1], None, frames=1, score=score, record=PUCK)")
    text = hosted.value(
        "evaluate([1], None, frames=2, score=score, record=PUCK, solver=lambda m: newton.solvers.SolverXPBD(m, iterations=2))"
    )
    test.assertIn("solver <lambda>", text)


def test_solver_classes_created_while_building_are_recorded(test, device):
    hosted = _Hosted(test, device, {"LAZY_SOLVER": True})
    test.assertEqual(hosted.session.scene_source.solver.label(), "LazySolver(model, iterations=3, flag=True)")
    test.assertEqual(
        hosted.value("evaluate([1], None, frames=1, score=score, record=PUCK).facts['solver']"),
        hosted.session.scene_source.solver.label(),
    )
    # The recording wrappers are gone after the build.
    test.assertFalse(hasattr(newton.solvers.SolverXPBD.__init__, "_newton_mcp_records"))


def test_structure_key_ignores_acceleration_data(test, device):
    def build():
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for i in range(60):
            body = builder.add_body(xform=wp.transform((0.1 * i, 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.04)
            builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
        return builder.finalize(device=device)

    # finalize() fills the shape BVH arrays in an order that can differ between identical builds.
    keys = {_structure_key(build()) for _ in range(4)}
    test.assertEqual(len(keys), 1)


def test_cells_can_bind_helper_names(test, device):
    hosted = _Hosted(test, device)
    result = hosted.execute("def evaluate(x):\n    return x + 1")
    test.assertIn("evaluate", result["workspace"]["variables"])
    # The cell's own function stays bound; the helper remains a session method.
    test.assertEqual(hosted.value("evaluate(2)"), 3)
    facts = hosted.value("session.evaluate([1], None, frames=1, score=score, record=PUCK).facts")
    test.assertIn("first use", facts["model"])
    hosted.execute("del evaluate")
    test.assertIn("evaluate: 1 case(s)", hosted.value("evaluate([1], None, frames=1, score=score, record=PUCK)"))


class TestMcpBatch(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _func in list(globals().items()):
    if _name.startswith("test_") and callable(_func):
        add_function_test(TestMcpBatch, _name, _func, devices=devices)


def _request(process, message: dict) -> dict:
    process.stdin.write((json.dumps(message) + "\n").encode())
    process.stdin.flush()
    return json.loads(process.stdout.readline())


class TestMcpBatchStdio(unittest.TestCase):
    def test_batch_helpers_over_stdio(self):
        """Host a script, attach the stdio MCP server, and run evaluate, branch and checkpoint as an agent does."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "puck.py"
        script.write_text(_SCRIPT)
        connection = Path(directory.name) / "session.json"
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        host = subprocess.Popen(
            [
                *(sys.executable, "-m", "newton.mcp", "host", str(script), "--connection-file", str(connection)),
                *("--overrides", json.dumps({"DEVICE": "cpu"})),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self.addCleanup(host.wait, 30)
        self.addCleanup(host.terminate)
        deadline = time.monotonic() + 180
        while not connection.with_suffix(".ready").exists():
            self.assertIsNone(host.poll(), "the host exited")
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.1)
        server = subprocess.Popen(
            [sys.executable, "-m", "newton.mcp", "--connect", str(connection), "--profile", "lean"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self.addCleanup(server.wait, 30)
        self.addCleanup(server.terminate)
        self.addCleanup(server.stdout.close)
        self.addCleanup(server.stdin.close)
        initialized = _request(server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        instructions = initialized["result"]["instructions"]
        for helper in ("evaluate(", "branch(", "checkpoint(name, include="):
            self.assertIn(helper, instructions)

        def execute(code: str, request_id: int) -> str:
            response = _request(
                server,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": "newton_execute", "arguments": {"code": code}},
                },
            )
            self.assertFalse(response["result"]["isError"], response["result"]["content"][0]["text"])
            return json.loads(response["result"]["content"][0]["text"]).get("result")

        execute(_HELPERS, 2)
        table = execute(
            "evaluate({'mu=0.1': 0.1, 'mu=0.8': 0.8}, {'slow': 1.0, 'fast': 2.5}, frames=20, setup=setup, "
            "score=score, record=PUCK, passed=lambda m, c, s: m['slide'] < 0.5, worst={'slide': 'max'})",
            3,
        )
        self.assertIn("evaluate: 4 case(s) from the live state at t=0 s", table)
        self.assertRegex(table, r"mu=0\.8 +2/2")
        self.assertRegex(table, r"mu=0\.1 +1/2 +fast")
        execute("session.dispatch('step', {'count': 5})", 4)
        saved = execute("checkpoint('c', include=['example.controller'])", 5)
        self.assertEqual(saved["objects"]["saved"], ["example.controller"])
        branched = execute(
            "branch(2, lambda w, i: setattr(example.controller, 'gain', [1.0, 3.0][i]), start='c', frames=5, "
            "sequential=True, record={'integral': lambda: example.controller.integral})",
            6,
        )
        self.assertIn("branch: 2 variant(s) from checkpoint 'c'", branched)
        self.assertRegex(branched, r"\n1 +-?\d")
        warm = execute(
            "evaluate([0.5, 0.6], [1.0, 2.0], frames=10, setup=setup, score=score, record=PUCK).facts['model']", 7
        )
        self.assertEqual(warm, "reused")


if __name__ == "__main__":
    unittest.main(verbosity=2)
