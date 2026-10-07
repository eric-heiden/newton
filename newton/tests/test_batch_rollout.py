# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`newton.utils.BatchRollout` and :func:`newton.utils.compare_trajectories`."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices
from newton.utils import BatchRollout, compare_trajectories

DT = 1.0 / 240.0


def _puck(mujoco: bool = False) -> newton.ModelBuilder:
    """One world: a puck on a floor and a position-driven hinge arm."""
    builder = newton.ModelBuilder()
    if mujoco:
        SolverMuJoCo.register_custom_attributes(builder)
    builder.add_ground_plane(label="floor")
    puck = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), label="puck")
    builder.add_shape_box(puck, hx=0.05, hy=0.05, hz=0.05, label="puck_geom")
    arm = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), label="arm")
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        parent_xform=wp.transform((2.0, 0.0, 1.0), wp.quat_identity()),
        axis=(0.0, 1.0, 0.0),
        target_ke=100.0,
        target_kd=5.0,
        label="hinge",
    )
    builder.add_articulation([hinge], label="arm")
    builder.add_shape_capsule(arm, radius=0.02, half_height=0.1, label="arm_geom")
    return builder


def _xpbd(model):
    return newton.solvers.SolverXPBD(model)


def _mujoco(model):
    return SolverMuJoCo(model)


def _push(rollout, speeds):
    speeds = np.asarray(speeds, dtype=np.float32)
    velocity = np.zeros((len(speeds), 6), dtype=np.float32)
    velocity[:, 0] = speeds
    rollout.set_state("joint_qd", velocity, labels="puck*", worlds=list(range(len(speeds))))


def test_build_forms(test, device):
    # A ModelBuilder, a function returning one, and a function returning the Model it finalized.
    for build in (_puck(), _puck, lambda: _puck().finalize(device=device)):
        rollout = BatchRollout(build, 3, solver=_xpbd, dt=DT, device=device)
        test.assertEqual(rollout.model.world_count, 3)
        test.assertEqual(rollout.model.body_count, 6)
        np.testing.assert_array_equal(rollout.model.body_world.numpy(), [0, 0, 1, 1, 2, 2])

    # Edits a build function makes to its Model after finalize() reach every world.
    def edited():
        model = _puck().finalize(device=device)
        mu = model.shape_material_mu.numpy()
        mu[model.find_shapes("puck_geom")] = 0.123
        model.shape_material_mu.assign(mu)
        return model

    rollout = BatchRollout(edited, 2, solver=_xpbd, dt=DT, device=device)
    test.assertIn("shape_material_mu", rollout.copied_attributes)
    np.testing.assert_allclose(rollout.view.get_attribute("shape_material_mu", rollout.model, "puck_geom"), 0.123)

    def two_worlds():
        scene = newton.ModelBuilder()
        scene.replicate(_puck(), 2)
        return scene.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "one world"):
        BatchRollout(two_worlds, 2, solver=_xpbd, dt=DT, device=device)


def test_run_records_and_schedules(test, device):
    rollout = BatchRollout(_puck, 3, solver=_xpbd, pipeline=newton.CollisionPipeline, dt=DT, substeps=2, device=device)
    frames, every = 12, 4
    schedule = np.arange(10, dtype=np.float32)[:, None, None] + np.array([0.0, 100.0, 200.0])[None, :, None]
    records = rollout.run(
        frames,
        control={("joint_target_q", "hinge"): schedule},
        record={"puck": ("body_q", "puck"), "target": ("joint_target_q", "hinge"), "q": "joint_q"},
        every=every,
    )
    test.assertEqual(records["puck"].shape, (frames // every + 1, 3, 1, 7))
    test.assertEqual(records["q"].shape, (frames // every + 1, 3, 8))
    np.testing.assert_allclose(rollout.record_time, np.arange(4) * every * 2 * DT)
    # Row r > 0 is recorded after frame r * every - 1, which applied schedule row min(r * every - 1, 9).
    targets = records["target"][:, :, 0]
    for r in range(1, 4):
        row = min(r * every - 1, 9)
        np.testing.assert_allclose(targets[r], [row, row + 100.0, row + 200.0])
    # The pucks rest on the floor; row 0 is the start state.
    np.testing.assert_allclose(records["puck"][0, :, 0, 2], 0.05, atol=1e-6)
    test.assertEqual(rollout.frames_done, frames)
    np.testing.assert_array_equal(rollout.frame_index.numpy(), [frames])

    # A run continues from the current state; reset starts over.
    more = rollout.run(4, record={"target": ("joint_target_q", "hinge")})
    np.testing.assert_allclose(more["target"][-1, :, 0], [9.0, 109.0, 209.0])
    rollout.reset()
    test.assertEqual(rollout.frames_done, 0)
    again = rollout.run(1, control={("joint_target_q", "hinge"): schedule}, record={"t": ("joint_target_q", "hinge")})
    np.testing.assert_allclose(again["t"][1, :, 0], [0.0, 100.0, 200.0])

    # The same values in every world: [F, k].
    rollout.reset()
    same = rollout.run(
        2, control={("joint_target_q", "hinge"): [[0.5], [0.7]]}, record={"t": ("joint_target_q", "hinge")}
    )
    np.testing.assert_allclose(same["t"][2, :, 0], 0.7)
    with test.assertRaisesRegex(ValueError, "schedule"):
        rollout.run(1, control={("joint_target_q", "hinge"): np.zeros((3, 2, 2))})


def test_control_functions_and_function_probes(test, device):
    rollout = BatchRollout(_puck, 2, solver=_xpbd, pipeline=newton.CollisionPipeline, dt=DT, device=device)
    puck_dofs = wp.array(rollout.view.get_indices("joint_qd", "puck*")[:, 0].astype(np.int32), device=device)
    height = wp.zeros(rollout.world_count, dtype=float, device=device)
    puck_bodies = wp.array(rollout.view.get_indices("body_q", "puck")[:, 0].astype(np.int32), device=device)

    @wp.kernel
    def push(frame: wp.array[wp.int32], dofs: wp.array[wp.int32], force: wp.array[float]):
        w = wp.tid()
        force[dofs[w]] = float(w + 1) * 5.0 * wp.where(frame[0] < 20, 1.0, 0.0)

    @wp.kernel
    def heights(body_q: wp.array[wp.transform], bodies: wp.array[wp.int32], out: wp.array[float]):
        w = wp.tid()
        out[w] = wp.transform_get_translation(body_q[bodies[w]])[0]

    def controller(r):
        wp.launch(
            push, dim=r.world_count, inputs=[r.frame_index, puck_dofs], outputs=[r.control.joint_f], device=device
        )

    def probe(r):
        wp.launch(heights, dim=r.world_count, inputs=[r.state.body_q, puck_bodies], outputs=[height], device=device)
        return height

    records = rollout.run(40, control=controller, record={"x": probe, "body": ("body_q", "puck")}, every=10)
    test.assertEqual(records["x"].shape, (5, 2))
    np.testing.assert_allclose(records["x"], records["body"][:, :, 0, 0], atol=1e-6)
    test.assertGreater(records["x"][-1, 1], records["x"][-1, 0])
    test.assertGreater(records["x"][-1, 0], 0.0)


def test_graph_matches_eager(test, device):
    results = []
    for capture in (True, False):
        rollout = BatchRollout(
            _puck, 3, solver=_xpbd, pipeline=newton.CollisionPipeline, dt=DT, substeps=3, device=device, capture=capture
        )
        rollout.view.set_attribute("shape_material_mu", rollout.model, [0.1, 0.5, 1.0], labels=["puck_geom", "floor"])
        _push(rollout, [2.0, 2.0, 2.0])
        results.append(rollout.run(30, record={"puck": ("body_q", "puck")}, every=5)["puck"])
        # A second run with the same plan replays the captured graph.
        rollout.reset()
        _push(rollout, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(rollout.run(30, record={"puck": ("body_q", "puck")}, every=5)["puck"], results[-1])
        test.assertEqual(len(rollout._graphs), 1 if capture else 0)
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)
    x = results[0][-1, :, 0, 0]
    test.assertGreater(x[0], x[1])
    test.assertGreater(x[1], x[2])


def test_reset_and_branch(test, device):
    rollout = BatchRollout(_puck, 3, solver=_xpbd, pipeline=newton.CollisionPipeline, dt=DT, device=device)
    rollout.view.set_attribute("shape_material_mu", rollout.model, [0.2, 0.5, 0.8], labels=["puck_geom", "floor"])
    record = {"puck": ("body_q", "puck")}
    _push(rollout, [2.0, 2.0, 2.0])
    full = rollout.run(40, record=record, every=40)["puck"]
    rollout.reset()
    _push(rollout, [2.0, 2.0, 2.0])
    rollout.run(20)
    saved = rollout.model.state()
    saved.assign(rollout.state)
    x_saved = saved.body_q.numpy()[rollout.view.get_indices("body_q", "puck")[:, 0], 0]

    # World by world: continues where the saved state was (XPBD keeps no internal state).
    rollout.reset(saved)
    resumed = rollout.run(20, record=record, every=20)["puck"]
    np.testing.assert_allclose(resumed[-1], full[-1], atol=1e-5)

    # One world into every world.
    rollout.reset(saved, world=2)
    branched = rollout.run(0, record=record)["puck"]
    np.testing.assert_allclose(branched[0, :, 0, 0], x_saved[2])

    # A state of a single-world model with the same layout, with and without that model.
    plant = _puck().finalize(device=device)
    plant_state = plant.state()
    plant_q = plant.joint_q.numpy()
    plant_q[0] = 0.3
    plant_state.joint_q.assign(plant_q)
    newton.eval_fk(plant, plant_state.joint_q, plant_state.joint_qd, plant_state)
    for model in (plant, None):
        rollout.reset(plant_state, model=model)
        np.testing.assert_allclose(rollout.run(0, record=record)["puck"][0, :, 0, 0], 0.3, atol=1e-6)
    with test.assertRaisesRegex(ValueError, "layout"):
        other = newton.ModelBuilder()
        other.add_body(label="lonely")
        rollout.reset(other.finalize(device=device).state())


def test_set_state_keeps_bodies_consistent(test, device):
    rollout = BatchRollout(_puck, 3, solver=_xpbd, dt=DT, device=device)
    before = rollout.state.body_q.numpy().copy()
    q = rollout.view.get_attribute("joint_q", rollout.state, labels="puck*", worlds=[1])
    q[0, 0] += 0.25
    rollout.set_state("joint_q", q, labels="puck*", worlds=[1])
    after = rollout.state.body_q.numpy()
    pucks = rollout.view.get_indices("body_q", "puck")[:, 0]
    np.testing.assert_allclose(after[pucks[1], 0], before[pucks[1], 0] + 0.25, atol=1e-6)
    np.testing.assert_allclose(after[pucks[[0, 2]]], before[pucks[[0, 2]]])


def test_view_refuses_shared_attributes(test, device):
    rollout = BatchRollout(lambda: _puck(mujoco=True), 2, solver=_mujoco, dt=DT, device=device)
    with test.assertRaisesRegex(ValueError, "only when it is constructed"):
        rollout.view.set_attribute("mujoco:condim", rollout.model, [1, 3], labels="puck_geom")
    rollout.view.set_attribute("shape_material_mu", rollout.model, [0.2, 0.4], labels="puck_geom")


def _setup(world, mu, scenario):
    world.set_model("shape_material_mu", mu, labels=["puck_geom", "floor"])
    world.set_state("joint_qd", [scenario["push"], 0.0, 0.0, 0.0, 0.0, 0.0], labels="puck*")
    if scenario.get("condim") is not None:
        world.set_model("mujoco:condim", scenario["condim"], labels=["puck_geom", "floor"])


def _score(records, cases):
    x = records["puck"][:, :, 0, 0]
    return {"slide": x[-1] - x[0], "moving": np.abs(x[-1] - x[-2]) > 1e-4}


def _passed(metrics, mu, scenario):
    if scenario.get("control"):
        return metrics["slide"] > 0.45  # the frictionless control must slide
    return metrics["slide"] < 0.5


def test_evaluate_table_summary_and_compare(test, device):
    rollout = BatchRollout(_puck, 8, solver=_xpbd, pipeline=newton.CollisionPipeline, dt=DT, substeps=2, device=device)
    mu_before = rollout.model.shape_material_mu.numpy().copy()
    scenarios = {"slow": {"push": 1.0}, "fast": {"push": 2.5}, "no_friction": {"push": 1.0, "control": True}}

    def setup(world, mu, scenario):
        _setup(world, 0.0 if scenario.get("control") else mu, scenario)

    kwargs = {
        "frames": 60,
        "setup": setup,
        "record": {"puck": ("body_q", "puck")},
        "every": 10,
        "score": _score,
        "passed": _passed,
        "worst": {"slide": "max"},
    }
    result = rollout.evaluate({"low": 0.1, "high": 0.8}, scenarios, **kwargs)
    test.assertEqual(len(result.rows), 6)
    test.assertEqual([row["scenario"] for row in result.rows[:3]], ["slow", "fast", "no_friction"])
    test.assertEqual(len(result.batches), 1)
    rows = {(row["candidate"], row["scenario"]): row for row in result.rows}
    test.assertTrue(rows[("high", "slow")]["passed"])
    test.assertTrue(rows[("high", "no_friction")]["passed"])
    test.assertFalse(rows[("low", "fast")]["passed"])
    test.assertGreater(rows[("low", "fast")]["slide"], rows[("high", "fast")]["slide"])
    summary = {entry["candidate"]: entry for entry in result.summary}
    test.assertEqual(summary["high"]["passed"], 3)
    test.assertEqual(summary["low"]["failed"], ["fast"])
    test.assertAlmostEqual(summary["low"]["pass_fraction"], 2.0 / 3.0)
    test.assertEqual(summary["low"]["worst"]["slide"][1], "fast")
    test.assertEqual(result.best("slide", reduce="max"), "high")
    test.assertIn("failed scenarios", str(result))
    # Values that setups changed are restored.
    np.testing.assert_array_equal(rollout.model.shape_material_mu.numpy(), mu_before)

    # A regression check against the earlier table.
    later = rollout.evaluate({"low": 0.8, "high": 0.1}, scenarios, **kwargs)
    diff = later.compare(result)
    test.assertEqual(diff.fixes, [("low", "fast")])
    test.assertEqual(diff.regressions, [("high", "fast")])
    test.assertEqual({entry[0] for entry in diff.changes["slide"][:2]}, {("low", "fast"), ("high", "fast")})
    test.assertIn("1 regression(s), 1 fix(es)", str(diff))

    # More cases than worlds run in chunks with the same results.
    chunked = BatchRollout(_puck, 4, solver=_xpbd, pipeline=newton.CollisionPipeline, dt=DT, substeps=2, device=device)
    split = chunked.evaluate({"low": 0.1, "high": 0.8}, scenarios, **kwargs)
    test.assertEqual([batch["worlds"] for batch in split.batches], [4, 2])
    np.testing.assert_allclose(
        [row["slide"] for row in split.rows], [row["slide"] for row in result.rows], rtol=1e-4, atol=1e-5
    )


def test_evaluate_groups_shared_values(test, device):
    rollout = BatchRollout(lambda: _puck(mujoco=True), 4, solver=_mujoco, dt=DT, substeps=2, device=device)
    scenarios = {"push": {"push": 2.0}, "frictionless": {"push": 2.0, "condim": 1}, "same": {"push": 2.0, "condim": 3}}
    result = rollout.evaluate(
        [0.5],
        scenarios,
        frames=40,
        setup=_setup,
        record={"puck": ("body_q", "puck")},
        every=40,
        score=_score,
    )
    # condim=3 equals the model's value and shares the batch; condim=1 needs a model of its own.
    test.assertEqual(len(result.batches), 2)
    test.assertEqual(result.batches[0]["cases"], [(0, "push"), (0, "same")])
    test.assertIn("mujoco:condim=1", result.batches[1]["reason"])
    test.assertIn("only when it is constructed", result.batches[1]["reason"])
    slides = {row["scenario"]: row["slide"] for row in result.rows}
    test.assertAlmostEqual(slides["push"], slides["same"], places=5)
    test.assertGreater(slides["frictionless"], 1.5 * slides["push"])
    # The separate model is kept for later calls.
    test.assertEqual(len(rollout._siblings), 1)
    rollout.evaluate([0.5], scenarios, frames=1, setup=_setup, score=lambda r, c: {"n": [0] * len(c)})
    test.assertEqual(len(rollout._siblings), 1)


def test_evaluate_per_candidate_builds(test, device):
    def build(size):
        builder = _puck()
        builder.shape_scale[builder.shape_label.index("puck_geom")] = (size, size, size)
        return builder

    rollout = BatchRollout(_puck, 4, solver=_xpbd, pipeline=newton.CollisionPipeline, dt=DT, device=device)
    result = rollout.evaluate(
        [0.05, 0.05, 0.08],
        None,
        frames=1,
        build=build,
        score=lambda records, cases: {"half_size": [size for size, _ in cases]},
    )
    test.assertEqual(len(result.batches), 2)
    test.assertEqual([batch["worlds"] for batch in result.batches], [2, 1])
    test.assertEqual(result.batches[1]["build"], "of candidate 2")


def _mu_seen(rollout):
    """A score that reads the puck friction each case ran with (the model during its batch)."""

    def score(records, cases):
        mu = rollout.view.get_attribute("shape_material_mu", rollout.model, labels="puck_geom")
        return {"mu": mu[: len(cases), 0]}

    return score


def test_evaluate_overlapping_selections(test, device):
    """A case's writes apply in its order, and no case's values carry over to later batches or calls."""

    def setup(world, case, _):
        if case == "all":
            world.set_model("shape_material_mu", 0.1)
        elif case == "puck":
            world.set_model("shape_material_mu", 2.0, labels="puck_geom")
        elif case == "all_then_puck":
            world.set_model("shape_material_mu", 0.1)
            world.set_model("shape_material_mu", 2.0, labels=["puck_geom"])
        elif case == "puck_then_all":
            world.set_model("shape_material_mu", 2.0, labels="puck_geom")
            world.set_model("shape_material_mu", 0.1)

    cases = ["all", "puck", "none", "all_then_puck", "puck_then_all"]
    expected = [0.1, 2.0, 1.0, 2.0, 0.1]
    for world_count in (1, 2, 5):
        rollout = BatchRollout(_puck, world_count, solver=_xpbd, dt=DT, device=device)
        before = rollout.model.shape_material_mu.numpy().copy()
        result = rollout.evaluate(cases, frames=1, setup=setup, score=_mu_seen(rollout))
        np.testing.assert_allclose([row["mu"] for row in result.rows], expected, rtol=1e-6)
        np.testing.assert_array_equal(rollout.model.shape_material_mu.numpy(), before)
        again = rollout.evaluate(list(range(world_count)), frames=1, score=_mu_seen(rollout))
        np.testing.assert_allclose([row["mu"] for row in again.rows], 1.0)


def test_evaluate_schedules_and_control_do_not_carry_over(test, device):
    """A case's schedule and constant controls stay in its batch; the control is restored afterwards."""
    rollout = BatchRollout(_puck, 1, solver=_xpbd, dt=DT, device=device)
    target = rollout.view.get_indices("joint_target_q", "hinge")[0, 0]

    def setup(world, case, _):
        if case == "swing":
            world.set_schedule("joint_target_q", np.full((5, 1), 1.0), labels="hinge")
        elif case == "hold":
            world.set_control("joint_target_q", 0.5, labels=["hinge"])

    def push(r):  # a control function that writes the control every step
        r.control.joint_f.fill_(0.0)

    before = rollout.control.joint_target_q.numpy().copy()
    result = rollout.evaluate(
        ["none", "swing", "none_again", "hold", "none_last"],
        frames=10,
        setup=setup,
        control=push,
        record={"t": ("joint_target_q", "hinge")},
        score=lambda records, cases: {"target": records["t"][-1, :, 0]},
    )
    test.assertEqual([row["target"] for row in result.rows], [0.0, 1.0, 0.0, 0.5, 0.0])
    np.testing.assert_array_equal(rollout.control.joint_target_q.numpy(), before)
    test.assertEqual(rollout.control.joint_target_q.numpy()[target], 0.0)

    # Labels given as lists or tuples, in shared and per-case schedules.
    shared = {("joint_target_q", ("hinge",)): np.full((3, 1), 0.25, dtype=np.float32)}
    for control, labels in ((shared, ["hinge"]), (None, ("hinge",))):
        result = rollout.evaluate(
            [0.0, 0.75],
            frames=3,
            control=control,
            setup=lambda world, value, _, labels=labels: world.set_schedule("joint_target_q", [[value]], labels=labels),
            record={"t": ("joint_target_q", "hinge")},
            score=lambda records, cases: {"target": records["t"][-1, :, 0]},
        )
        test.assertEqual([row["target"] for row in result.rows], [0.0, 0.75])


def test_body_state_edits_reach_joint_coordinate_solvers(test, device):
    """Edits of body_q and body_qd move the bodies for solvers that integrate joint coordinates."""
    rollout = BatchRollout(lambda: _puck(mujoco=True), 2, solver=_mujoco, dt=DT, device=device)

    def setup(world, shift, _):
        q = world.get_state("body_q", "puck")
        q[0, 0] += shift
        world.set_state("body_q", q, labels="puck")
        world.set_state("body_qd", [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], labels="puck")

    result = rollout.evaluate(
        [0.0, 0.5],
        frames=3,
        setup=setup,
        record={"puck": ("body_q", "puck"), "q": ("joint_q", "puck*")},
        score=lambda records, cases: {
            "start": records["puck"][0, :, 0, 0],
            "end": records["puck"][-1, :, 0, 0],
            "joint": records["q"][0, :, 0],
        },
    )
    rows = {row["candidate"]: row for row in result.rows}
    test.assertAlmostEqual(rows[1]["start"], 0.5, places=5)
    test.assertAlmostEqual(rows[1]["joint"], 0.5, places=5)
    test.assertAlmostEqual(rows[1]["end"], 0.5, places=3)
    test.assertAlmostEqual(rows[0]["end"], 0.0, places=3)

    # set_state(): a body velocity reaches the joint velocities; other articulations keep their state.
    rollout.reset()
    arm_before = rollout.view.get_attribute("joint_q", rollout.state, "hinge").copy()
    velocity = np.zeros((1, 1, 6), dtype=np.float32)
    velocity[..., 0] = 1.0
    rollout.set_state("body_qd", velocity, labels="puck", worlds=[1])
    qd = rollout.view.get_attribute("joint_qd", rollout.state, "puck*")
    np.testing.assert_allclose(qd[:, 0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(rollout.view.get_attribute("joint_q", rollout.state, "hinge"), arm_before)
    x = rollout.run(30, record={"puck": ("body_q", "puck")})["puck"][-1, :, 0, 0]
    test.assertGreater(x[1], x[0] + 0.02)


def test_evaluate_separate_models_start_like_this_rollout(test, device):
    """Separate models take world 0's model values and control, also after the rollout's values change."""
    rollout = BatchRollout(lambda: _puck(mujoco=True), 2, solver=_mujoco, dt=DT, substeps=2, device=device)
    record = {"arm": ("joint_q", "hinge"), "puck": ("body_q", "puck")}

    def setup(world, case, _):
        world.set_state("joint_qd", [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], labels="puck*")
        if case == "separate":
            world.set_model("mujoco:condim", 4, labels=["puck_geom", "floor"])  # torsional friction: no slide change

    def score(records, cases):
        x = records["puck"][:, :, 0, 0]
        return {"arm": records["arm"][-1, :, 0], "slide": x[-1] - x[0]}

    cases = {"same": "same", "separate": "separate"}

    def run():
        result = rollout.evaluate(cases, frames=40, setup=setup, record=record, score=score)
        test.assertIsNotNone(result.batches[-1]["reason"])
        return {row["candidate"]: row for row in result.rows}

    # The rollout's control drives the hinge to 1 rad in every world, also in the separate model.
    target = rollout.control.joint_target_q.numpy()
    target[rollout.view.get_indices("joint_target_q", "hinge").ravel()] = 1.0
    rollout.control.joint_target_q.assign(target)
    rows = run()
    test.assertGreater(rows["same"]["arm"], 0.3)
    test.assertAlmostEqual(rows["separate"]["arm"], rows["same"]["arm"], places=4)
    test.assertAlmostEqual(rows["separate"]["slide"], rows["same"]["slide"], delta=0.01)
    separate = next(iter(rollout._siblings.values()))

    # A later edit of the rollout's friction reaches the kept separate model.
    rollout.view.set_attribute("shape_material_mu", rollout.model, 0.05, labels=["puck_geom", "floor"])
    slippery = run()
    test.assertIs(next(iter(rollout._siblings.values())), separate)
    test.assertIn("shape_material_mu", rollout._stats["synced"])
    test.assertGreater(slippery["same"]["slide"], 1.5 * rows["same"]["slide"])
    test.assertAlmostEqual(slippery["separate"]["slide"], slippery["same"]["slide"], delta=0.01)

    # initial_control: world 0 of another control in every world of every model; the rollout keeps its own.
    plant = _puck(mujoco=True).finalize(device=device)
    plant_control = plant.control()
    still = np.zeros(plant.joint_coord_count, dtype=np.float32)
    plant_control.joint_target_q.assign(still)
    result = rollout.evaluate(
        cases,
        frames=40,
        setup=setup,
        record=record,
        score=score,
        initial_control=plant_control,
        initial_model=plant,
    )
    np.testing.assert_allclose([row["arm"] for row in result.rows], 0.0, atol=1e-4)
    np.testing.assert_array_equal(rollout.control.joint_target_q.numpy(), target)


def test_evaluate_keeps_the_separate_models_of_a_call(test, device):
    """A call with more separate models than the cache limit keeps them, so repeating it builds none."""
    rollout = BatchRollout(lambda: _puck(mujoco=True), 2, solver=_mujoco, dt=DT, device=device)
    values = np.linspace(1.5, 4.0, 10)

    def setup(world, index, _):
        world.set_model("mujoco:impratio", values[index])

    kwargs = {"frames": 1, "setup": setup, "score": lambda records, cases: {"n": [0] * len(cases)}}
    rollout.evaluate(list(range(10)), **kwargs)
    kept = dict(rollout._siblings)
    test.assertEqual(len(kept), 10)
    rollout.evaluate(list(range(10)), **kwargs)
    test.assertEqual(len(rollout._siblings), 10)
    test.assertTrue(all(rollout._siblings[key] is model for key, model in kept.items()))
    # A smaller call afterwards trims the cache to its limit.
    rollout.evaluate([0], **kwargs)
    test.assertEqual(len(rollout._siblings), 8)


def test_buffers_scalars_and_speculative_contacts(test, device):
    """Probe buffers are reused and bounded, scalar model settings are copied, and speculative pipelines run."""

    def build():
        model = _puck().finalize(device=device)
        model.soft_contact_ke = 123.0
        model.particle_mu = 0.9
        return model

    rollout = BatchRollout(
        build,
        2,
        solver=_xpbd,
        pipeline=lambda model: newton.CollisionPipeline(model, speculative_contact_gap_max=0.01),
        dt=DT,
        device=device,
    )
    test.assertEqual((rollout.model.soft_contact_ke, rollout.model.particle_mu), (123.0, 0.9))
    test.assertIn("soft_contact_ke", rollout.copied_attributes)
    for _ in range(5):
        rollout.reset()
        records = rollout.run(5, record={"q": lambda r: r.state.body_q})  # a new function every run
    test.assertEqual(records["q"].shape[0], 6)
    test.assertEqual(len(rollout._probes), 1)
    for index in range(40):
        rollout.run(1, record={f"probe{index}": "body_q"})
    test.assertLessEqual(len(rollout._probes), 32)


def test_value_sync_ignores_ids_offset_per_world(test, device):
    """IDs that replication offsets per world (MJCF collision mask domains) are structure, not world values."""
    from newton._src.utils.batch_rollout import _ValueSync  # noqa: PLC0415

    mjcf = """<mujoco><worldbody><body name="box" pos="0 0 0.2"><freejoint/>
    <geom type="box" size="0.05 0.05 0.05" contype="2" conaffinity="2"/></body></worldbody></mujoco>"""

    def build():
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        builder.add_ground_plane()
        return builder.finalize(device=device)

    one_world = build()
    rollout = BatchRollout(build, 3, solver=_mujoco, dt=DT, device=device)
    domains = rollout.model.mujoco.collision_mask_domain.numpy()
    test.assertGreater(len(set(domains[domains >= 0].tolist())), 1)  # one domain per world
    test.assertEqual(_ValueSync(rollout, one_world).changed(), [])
    with test.assertRaisesRegex(ValueError, "structure"):
        rollout.view.set_attribute("mujoco:collision_mask_domain", rollout.model, 0, labels="*")


def test_compare_trajectories(test, device):
    del device
    t = np.linspace(0.0, 1.0, 11)
    reference = {"q": np.zeros((11, 3)), "x": np.zeros(11), "only_ref": np.zeros(11)}
    q = np.zeros((11, 3))
    q[6:, 1] = 0.2  # joint 1 diverges at t = 0.6
    q[:, 2] = 0.01
    result = compare_trajectories(
        reference, {"q": q, "x": np.full(11, 0.1)}, times=t, tolerance={"q": 0.1}, labels={"q": ["a", "b", "c"]}
    )
    test.assertEqual(result.signals, ["q", "x"])
    test.assertEqual(result.missing, ["only_ref"])
    test.assertAlmostEqual(result.divergence_time["q"], 0.6)
    test.assertTrue(np.isnan(result.divergence_time["x"]))
    test.assertAlmostEqual(result.max_error["q"], 0.2)
    test.assertAlmostEqual(result.max_time["q"], 0.6)
    test.assertEqual(result.contributors["q"][0][0], "b")
    test.assertAlmostEqual(result.rmse["x"], 0.1)
    test.assertAlmostEqual(result.objective({"q": 0.0}), 0.1)
    test.assertIn("diverges at 0.6", str(result))

    # A batched candidate sampled at other times is interpolated to the reference times.
    fine_t = np.linspace(0.0, 1.0, 101)
    batched = np.stack([fine_t, 2.0 * fine_t], axis=1)[:, :, None]  # [T, 2 worlds, 1]
    result = compare_trajectories({"x": t[:, None]}, {"x": batched}, times=t, candidate_times=fine_t, tolerance=0.25)
    test.assertTrue(result.batched)
    np.testing.assert_allclose(result.rmse["x"], [0.0, np.sqrt(np.mean(t**2))], atol=1e-12)
    np.testing.assert_allclose(result.divergence_time["x"], [np.nan, 0.3])
    test.assertEqual(result.objective().shape, (2,))

    # A non-finite candidate diverges where it stops being finite.
    broken = np.zeros((11, 1))
    broken[4:] = np.nan
    result = compare_trajectories({"x": np.zeros((11, 1))}, {"x": broken}, times=t)
    test.assertAlmostEqual(result.divergence_time["x"], 0.4)
    test.assertTrue(np.isinf(result.rmse["x"]))
    with test.assertRaisesRegex(ValueError, "candidate_times"):
        compare_trajectories({"x": np.zeros(11)}, {"x": np.zeros(12)}, times=t)


class TestBatchRollout(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _func in (
    ("test_build_forms", test_build_forms),
    ("test_run_records_and_schedules", test_run_records_and_schedules),
    ("test_control_functions_and_function_probes", test_control_functions_and_function_probes),
    ("test_reset_and_branch", test_reset_and_branch),
    ("test_set_state_keeps_bodies_consistent", test_set_state_keeps_bodies_consistent),
    ("test_view_refuses_shared_attributes", test_view_refuses_shared_attributes),
    ("test_evaluate_table_summary_and_compare", test_evaluate_table_summary_and_compare),
    ("test_evaluate_groups_shared_values", test_evaluate_groups_shared_values),
    ("test_evaluate_per_candidate_builds", test_evaluate_per_candidate_builds),
    ("test_evaluate_overlapping_selections", test_evaluate_overlapping_selections),
    ("test_evaluate_schedules_and_control_do_not_carry_over", test_evaluate_schedules_and_control_do_not_carry_over),
    ("test_body_state_edits_reach_joint_coordinate_solvers", test_body_state_edits_reach_joint_coordinate_solvers),
    ("test_evaluate_separate_models_start_like_this_rollout", test_evaluate_separate_models_start_like_this_rollout),
    ("test_evaluate_keeps_the_separate_models_of_a_call", test_evaluate_keeps_the_separate_models_of_a_call),
    ("test_buffers_scalars_and_speculative_contacts", test_buffers_scalars_and_speculative_contacts),
    ("test_value_sync_ignores_ids_offset_per_world", test_value_sync_ignores_ids_offset_per_world),
):
    add_function_test(TestBatchRollout, _name, _func, devices=devices)
add_function_test(
    TestBatchRollout, "test_graph_matches_eager", test_graph_matches_eager, devices=get_cuda_test_devices()
)
add_function_test(TestBatchRollout, "test_compare_trajectories", test_compare_trajectories)


if __name__ == "__main__":
    unittest.main(verbosity=2)
