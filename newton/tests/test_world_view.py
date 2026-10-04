# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`newton.selection.WorldView` and solver per-world value checks."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import WorldView
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _template(mujoco: bool = False) -> newton.ModelBuilder:
    """One world: a two-link arm, a free box on a static table, and a static mesh."""
    builder = newton.ModelBuilder()
    if mujoco:
        SolverMuJoCo.register_custom_attributes(builder)
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), label="arm/base")
    tip = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.01), label="arm/tip")
    j0 = builder.add_joint_revolute(parent=-1, child=base, axis=(0.0, 0.0, 1.0), label="arm/shoulder")
    j1 = builder.add_joint_revolute(parent=base, child=tip, axis=(0.0, 1.0, 0.0), label="arm/elbow")
    builder.add_articulation([j0, j1], label="arm")
    builder.add_shape_capsule(base, radius=0.02, half_height=0.1, label="arm/base_geom")
    builder.add_shape_capsule(tip, radius=0.02, half_height=0.1, label="arm/pad")
    box = builder.add_body(xform=wp.transform((0.5, 0.0, 0.1), wp.quat_identity()), label="box")
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, label="box_geom")
    builder.add_shape_box(
        -1, xform=wp.transform((0.5, 0.0, -0.05), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05, label="table"
    )
    mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_inertia=False)
    builder.add_shape_mesh(-1, xform=wp.transform((1.5, 0.0, 0.05), wp.quat_identity()), mesh=mesh, label="obstacle")
    return builder


def _replicated(device, world_count: int, mujoco: bool = False, ground: bool = True) -> newton.Model:
    scene = newton.ModelBuilder()
    if mujoco:
        SolverMuJoCo.register_custom_attributes(scene)
    if ground:
        scene.add_ground_plane()
    scene.replicate(_template(mujoco), world_count)
    return scene.finalize(device=device)


def test_indices_per_world(test, device):
    model = _replicated(device, 3)
    view = WorldView(model)
    shape_world = model.shape_world.numpy()

    pads = view.get_indices("shape_material_mu", "*pad")
    test.assertEqual(pads.shape, (3, 1))
    test.assertEqual([model.shape_label[i] for i in pads[:, 0]], ["arm/pad"] * 3)
    np.testing.assert_array_equal(shape_world[pads[:, 0]], [0, 1, 2])

    # Static shapes of a world are selectable.
    tables = view.get_indices("shape_material_mu", "table")
    np.testing.assert_array_equal(shape_world[tables[:, 0]], [0, 1, 2])

    # Joint DOFs and coordinates are selected by joint label, in model order.
    dofs = view.get_indices("joint_target_ke", "arm/*")
    np.testing.assert_array_equal(dofs, np.arange(3)[:, None] * 8 + np.array([0, 1]))
    coords = view.get_indices("joint_q", "box*")
    test.assertEqual(coords.shape, (3, 7))
    np.testing.assert_array_equal(coords[1], 9 + 2 + np.arange(7))

    # A subset of worlds keeps the requested order.
    np.testing.assert_array_equal(view.get_indices("body_mass", "box", worlds=[2, 0]), [[8], [2]])

    with test.assertRaisesRegex(ValueError, "global SHAPE rows"):
        view.get_indices("shape_material_mu", "*")
    with test.assertRaises(KeyError):
        view.get_indices("body_mass", "missing*")
    np.testing.assert_array_equal(view.get_indices("body_mass", [8, 2, 5]), [[2], [5], [8]])
    with test.assertRaisesRegex(ValueError, "differ in count"):
        view.get_indices("body_mass", [2])
    with test.assertRaises(TypeError):
        view.get_indices("body_mass", [0.5])

    # In a single-world model, global rows belong to world 0.
    single = _template().finalize(device=device)
    np.testing.assert_array_equal(WorldView(single).get_indices("shape_material_mu", "table"), [[3]])


def test_heterogeneous_worlds(test, device):
    scene = newton.ModelBuilder()
    for has_ball in (True, False, True):
        world = newton.ModelBuilder()
        world.add_shape_box(world.add_body(label="box"), hx=0.1, hy=0.1, hz=0.1, label="box_geom")
        if has_ball:
            world.add_shape_sphere(world.add_body(label="ball"), radius=0.1, label="ball_geom")
        scene.add_world(world)
    model = scene.finalize(device=device)
    view = WorldView(model)
    with test.assertRaisesRegex(ValueError, "differ in count"):
        view.get_indices("body_mass", "ball")
    np.testing.assert_array_equal(view.get_indices("body_mass", "ball", worlds=[0, 2]), [[1], [4]])
    view.set_attribute("body_mass", model, [2.0, 3.0], labels="ball", worlds=[0, 2])
    np.testing.assert_allclose(model.body_mass.numpy()[[1, 4]], [2.0, 3.0])


def test_set_get_attribute(test, device):
    model = _replicated(device, 4)
    view = WorldView(model)
    mu = model.shape_material_mu.numpy().copy()

    flags = view.set_attribute("shape_material_mu", model, [0.1, 0.2, 0.3, 0.4], labels=["*pad", "table"])
    test.assertEqual(flags, int(newton.ModelFlags.SHAPE_PROPERTIES))
    got = view.get_attribute("shape_material_mu", model, labels=["*pad", "table"])
    test.assertEqual(got.shape, (4, 2))
    np.testing.assert_allclose(got, np.repeat([[0.1], [0.2], [0.3], [0.4]], 2, axis=1), rtol=1e-6)
    # Rows outside the selection keep their values.
    changed = view.get_indices("shape_material_mu", ["*pad", "table"]).ravel()
    unchanged = np.setdiff1d(np.arange(model.shape_count), changed)
    np.testing.assert_array_equal(model.shape_material_mu.numpy()[unchanged], mu[unchanged])

    # One value per world and row.
    ke = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
    test.assertEqual(
        view.set_attribute("joint_target_ke", model, ke, labels="arm/*"), int(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    )
    np.testing.assert_allclose(model.joint_target_ke.numpy().reshape(4, 8)[:, :2], ke)

    # Vector values broadcast per world; a subset of worlds leaves the others alone.
    before = model.shape_transform.numpy().copy()
    xforms = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    view.set_attribute("shape_transform", model, xforms, labels="table", worlds=[1, 3])
    got = view.get_attribute("shape_transform", model, labels="table")
    np.testing.assert_allclose(got[[1, 3], 0], xforms)
    np.testing.assert_allclose(got[[0, 2], 0], before[view.get_indices("shape_transform", "table")[[0, 2], 0]])

    # A scalar sets every selected row; a Warp array is copied on the device.
    view.set_attribute("joint_armature", model, 0.5, labels="arm/*")
    np.testing.assert_allclose(model.joint_armature.numpy().reshape(4, 8)[:, :2], 0.5)
    damping = wp.array(np.arange(8, dtype=np.float32), dtype=wp.float32, device=device)
    view.set_attribute("joint_damping", model, damping, labels="arm/*")
    np.testing.assert_allclose(model.joint_damping.numpy().reshape(4, 8)[:, :2].ravel(), np.arange(8))

    with test.assertRaisesRegex(ValueError, "one per selected world"):
        view.set_attribute("shape_material_mu", model, [0.1, 0.2], labels="*pad")
    with test.assertRaisesRegex(ValueError, "do not broadcast"):
        view.set_attribute("joint_target_ke", model, np.ones((4, 3)), labels="arm/*")


def test_world_values_and_state(test, device):
    model = _replicated(device, 3)
    view = WorldView(model)
    gravity = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0], [0.0, 0.0, -3.0]])
    test.assertEqual(view.set_attribute("gravity", model, gravity), int(newton.ModelFlags.MODEL_PROPERTIES))
    np.testing.assert_allclose(model.gravity.numpy()[:3], gravity)

    state = model.state()
    control = model.control()
    test.assertEqual(view.set_attribute("joint_q", state, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], labels="arm/*"), 0)
    np.testing.assert_allclose(state.joint_q.numpy().reshape(3, 9)[:, :2], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    view.set_attribute("joint_target_q", control, [1.0, 2.0, 3.0], labels="arm/elbow")
    np.testing.assert_allclose(view.get_attribute("joint_target_q", control, "arm/elbow")[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(view.get_attribute("body_q", state, "box")[:, 0, :3], [[0.5, 0.0, 0.1]] * 3)


def test_derived_arrays(test, device):
    model = _replicated(device, 3)
    view = WorldView(model)
    test.assertEqual(
        view.set_attribute("body_mass", model, [1.0, 2.0, 4.0], labels="box"),
        int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES),
    )
    rows = view.get_indices("body_mass", "box")[:, 0]
    np.testing.assert_allclose(model.body_inv_mass.numpy()[rows], [1.0, 0.5, 0.25])
    view.set_attribute("body_inertia", model, [np.eye(3) * k for k in (1.0, 2.0, 4.0)], labels="box")
    np.testing.assert_allclose(model.body_inv_inertia.numpy()[rows][:, 0, 0], [1.0, 0.5, 0.25])

    view.set_attribute("shape_scale", model, [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], labels="box_geom")
    shapes = view.get_indices("shape_scale", "box_geom")[:, 0]
    np.testing.assert_allclose(
        model.shape_collision_radius.numpy()[shapes], np.sqrt(3) * np.array([0.1, 0.2, 0.3]), rtol=1e-6
    )
    with test.assertRaisesRegex(ValueError, "computes the collision data"):
        view.set_attribute("shape_scale", model, [2.0, 2.0, 2.0], labels="obstacle")


def test_refuses_structural_attributes(test, device):
    model = _replicated(device, 2, mujoco=True)
    view = WorldView(model)
    with test.assertRaisesRegex(ValueError, "all worlds share"):
        view.set_attribute("mujoco:iterations", model, 10)
    with test.assertRaisesRegex(ValueError, "model structure"):
        view.set_attribute("shape_body", model, 0, labels="box_geom")
    with test.assertRaisesRegex(ValueError, "model structure"):
        view.set_attribute("shape_type", model, 1, labels="box_geom")
    other = _replicated(device, 2, mujoco=True)
    with test.assertRaisesRegex(ValueError, "different Model"):
        view.set_attribute("body_mass", other, 1.0, labels="box")


def test_copy_state(test, device):
    model = _replicated(device, 4)
    view = WorldView(model)
    state = model.state()
    q = state.joint_q.numpy().reshape(4, 9)
    q[2, :2] = [0.7, -0.3]
    q[2, 4] = 1.25  # box z
    state.joint_q.assign(q.ravel())
    qd = state.joint_qd.numpy().reshape(4, 8)
    qd[2] = np.arange(8)
    state.joint_qd.assign(qd.ravel())
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    clone = model.state()
    view.copy_state(clone, state, src_world=2, worlds=[0, 1])
    q_clone = clone.joint_q.numpy().reshape(4, 9)
    np.testing.assert_allclose(q_clone[:2], np.repeat(q[2:3], 2, axis=0))
    np.testing.assert_allclose(q_clone[2:], model.joint_q.numpy().reshape(4, 9)[2:])
    np.testing.assert_allclose(clone.joint_qd.numpy().reshape(4, 8)[:2], np.repeat(qd[2:3], 2, axis=0))
    np.testing.assert_allclose(
        clone.body_q.numpy().reshape(4, 3, 7)[:2], np.repeat(state.body_q.numpy().reshape(4, 3, 7)[2:3], 2, axis=0)
    )
    # Repeated calls reuse the cached plan.
    view.copy_state(clone, state, src_world=2)
    np.testing.assert_allclose(clone.joint_q.numpy().reshape(4, 9), np.repeat(q[2:3], 4, axis=0))

    # From a single-world model with the same layout, e.g. a plant into a planner.
    plant = _template().finalize(device=device)
    plant_state = plant.state()
    plant_q = plant_state.joint_q.numpy()
    plant_q[:2] = [0.4, 0.5]
    plant_state.joint_q.assign(plant_q)
    planner_state = model.state()
    view.copy_state(planner_state, plant_state, src_model=plant)
    np.testing.assert_allclose(planner_state.joint_q.numpy().reshape(4, 9), np.tile(plant_q, (4, 1)))

    smaller = newton.ModelBuilder()
    smaller.add_body(label="box")
    small = smaller.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "source world 0 has"):
        view.copy_state(model.state(), small.state(), src_model=small)


def test_model_flags_from_attributes(test, device):
    flags = newton.ModelFlags
    test.assertEqual(flags.from_attributes("body_mass", "mujoco:gravcomp"), int(flags.BODY_INERTIAL_PROPERTIES))
    test.assertEqual(flags.from_attributes("mujoco.gravcomp"), int(flags.BODY_INERTIAL_PROPERTIES))
    test.assertEqual(
        flags.from_attributes("joint_target_ke", "shape_material_mu"),
        int(flags.JOINT_DOF_PROPERTIES | flags.SHAPE_PROPERTIES),
    )
    test.assertEqual(flags.from_attributes("gravity"), int(flags.MODEL_PROPERTIES))
    test.assertEqual(flags.from_attributes("mujoco:tendon_stiffness"), int(flags.TENDON_PROPERTIES))
    test.assertEqual(flags.from_attributes("unknown_attribute"), int(flags.ALL))
    test.assertEqual(flags.from_attributes(), 0)


def test_mujoco_check_world_values(test, device):
    model = _replicated(device, 2, mujoco=True)
    view = WorldView(model)
    solver = SolverMuJoCo(model)

    # Per-world values reach every MuJoCo world after the view notifies the solver.
    view.set_attribute("shape_material_mu", model, [0.3, 0.9], labels="box_geom", solver=solver)
    shapes = view.get_indices("shape_material_mu", "box_geom")[:, 0]
    geoms = [int(np.nonzero(solver.mjc_geom_to_newton_shape.numpy()[w] == shapes[w])[0][0]) for w in range(2)]
    friction = solver.mjw_model.geom_friction.numpy()
    np.testing.assert_allclose([friction[w, geoms[w], 0] for w in range(2)], [0.3, 0.9], rtol=1e-6)
    view.set_attribute("mujoco:gravcomp", model, [0.0, 1.0], labels="arm/*", solver=solver)
    gravcomp = solver.mjw_model.body_gravcomp.numpy()
    test.assertEqual(gravcomp.shape[0], 2)
    test.assertEqual(np.count_nonzero(gravcomp[0]), 0)
    test.assertEqual(np.count_nonzero(gravcomp[1] == 1.0), 2)

    refusals = {
        "mujoco:condim": ("box_geom", 1, "only when it is constructed"),
        "joint_target_mode": ("arm/*", 0, "only when it is constructed"),
        "mujoco:tolerance": (None, 1e-6, "solver option"),
        "shape_material_restitution": ("box_geom", 0.5, "does not read"),
        "body_inv_mass": ("box", 1.0, "reads body_mass"),
        "shape_scale": ("obstacle", 1.0, None),
    }
    for name, (labels, value, message) in refusals.items():
        with test.subTest(name=name):
            before = view.get_attribute(name, model, labels=labels)
            with test.assertRaises(ValueError) as caught:
                view.set_attribute(name, model, value, labels=labels, solver=solver)
            if message is not None:
                test.assertIn(message, str(caught.exception))
            np.testing.assert_array_equal(view.get_attribute(name, model, labels=labels), before)

    # The solver can be asked directly, for rows of any attribute.
    with test.assertRaisesRegex(ValueError, "mesh, convex-mesh, and heightfield"):
        solver.check_world_values("shape_scale", view.get_indices("shape_scale", "obstacle").ravel())
    solver.check_world_values("shape_scale", view.get_indices("shape_scale", "box_geom").ravel())
    # The arm joints have no joint-target actuators (joint_target_mode NONE), so their gains are not read.
    with test.assertRaisesRegex(ValueError, "no position actuator for DOFs of arm/elbow, arm/shoulder"):
        view.set_attribute("joint_target_ke", model, [10.0, 20.0], labels="arm/*", solver=solver)


_SERVO_MJCF = """
<mujoco>
  <worldbody>
    <body name="link">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="servo" joint="hinge" kp="50"/>
    <motor name="motor" joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


def test_mujoco_joint_target_actuator_rows(test, device):
    world = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(world)
    world.add_mjcf(_SERVO_MJCF)
    scene = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(world, 2)
    model = scene.finalize(device=device)
    solver = SolverMuJoCo(model)
    view = WorldView(model)

    with test.assertRaisesRegex(ValueError, "JOINT_TARGET actuators"):
        view.set_attribute("mujoco:actuator_gainprm", model, 1.0, labels="*servo", solver=solver)
    gain = np.zeros((2, 10), dtype=np.float32)
    gain[:, 0] = [3.0, 7.0]
    view.set_attribute("mujoco:actuator_gainprm", model, gain, labels="*motor", solver=solver)
    view.set_attribute("joint_target_ke", model, [10.0, 90.0], labels="*hinge", solver=solver)
    solver.check_world_values("joint_target_kd")
    compiled = solver.mjw_model.actuator_gainprm.numpy()[:, :, 0]
    for world, values in enumerate(([10.0, 3.0], [90.0, 7.0])):
        np.testing.assert_allclose(np.sort(compiled[world]), np.sort(values))


def test_candidates_match_sequential_runs(test, device):
    """Candidates set per world give the same rollouts as one candidate per single-world model."""
    candidates = np.array([0.0, 0.5, 1.0])

    def build(world_count):
        world = newton.ModelBuilder()
        body = world.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), label="puck")
        world.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, label="puck_geom")
        world.add_ground_plane(label="floor")
        scene = newton.ModelBuilder()
        scene.replicate(world, world_count)
        return scene.finalize(device=device)

    def rollout(model, values):
        view = WorldView(model)
        solver = newton.solvers.SolverXPBD(model)
        view.set_attribute("shape_material_mu", model, values, labels="puck_geom", solver=solver)
        state_0, state_1 = model.state(), model.state()
        view.set_attribute("joint_qd", state_0, [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(values), labels="puck*")
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        for _ in range(60):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, None, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        return view.get_attribute("body_q", state_0, labels="puck")[:, 0, 0]

    batched = rollout(build(len(candidates)), candidates)
    sequential = np.concatenate([rollout(build(1), [mu]) for mu in candidates])
    np.testing.assert_allclose(batched, sequential, rtol=1e-5, atol=1e-6)
    test.assertGreater(batched[0], batched[1])
    test.assertGreater(batched[1], batched[2])


def test_labels_match_like_model_find(test, device):
    model = _replicated(device, 2)
    view = WorldView(model)

    # A bare name matches the last path component, as Model.find_shapes() does.
    pads = view.get_indices("shape_material_mu", "pad")
    np.testing.assert_array_equal(pads.ravel(), model.find_shapes("pad"))
    np.testing.assert_array_equal(view.get_indices("joint_target_ke", "elbow").ravel(), model.find_joint_dofs("elbow"))
    np.testing.assert_array_equal(
        view.get_indices("joint_q", ["shoulder", "elbow"]).ravel(), model.find_joint_coords("arm/*")
    )
    np.testing.assert_array_equal(
        view.get_indices("body_mass", ["base", "arm/tip"]).ravel(), model.find_bodies(["base", "tip"])
    )
    with test.assertRaisesRegex(KeyError, "closest names: elbow"):
        view.get_indices("joint_target_ke", "elbw")


class TestWorldView(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _func in (
    ("test_indices_per_world", test_indices_per_world),
    ("test_heterogeneous_worlds", test_heterogeneous_worlds),
    ("test_labels_match_like_model_find", test_labels_match_like_model_find),
    ("test_set_get_attribute", test_set_get_attribute),
    ("test_world_values_and_state", test_world_values_and_state),
    ("test_derived_arrays", test_derived_arrays),
    ("test_refuses_structural_attributes", test_refuses_structural_attributes),
    ("test_copy_state", test_copy_state),
    ("test_model_flags_from_attributes", test_model_flags_from_attributes),
    ("test_mujoco_check_world_values", test_mujoco_check_world_values),
    ("test_mujoco_joint_target_actuator_rows", test_mujoco_joint_target_actuator_rows),
    ("test_candidates_match_sequential_runs", test_candidates_match_sequential_runs),
):
    add_function_test(TestWorldView, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
