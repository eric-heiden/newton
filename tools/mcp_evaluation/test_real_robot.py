# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check real-data provenance, physical parameterization and Newton candidate parity."""

import io
import pickle
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

from .real_robot import RealRobotScenario, initial_config, prepare_windows
from .real_robot_data import _DataFrameUnpickler, load_reference, prepare_geometry
from .real_robot_model import make_builder, physical_coefficients, validate_config
from .real_robot_regressor import inverse_dynamics_matrix, prepare_regressor
from .rollout import make_session


def reference_arrays() -> dict:
    """Create labeled numeric fixture data, never an identified study parameter set."""
    times = np.linspace(0.0, 4.0, 220)
    q = np.sin(times[:, None] + np.arange(7)) * 0.1
    return {
        "time": times,
        "q": q,
        "qd": np.cos(times[:, None] + np.arange(7)) * 0.1,
        "qdd": -q,
        "tau": np.zeros_like(q),
        "episode_offsets": np.array([0, len(times)]),
        "episode_ids": np.array([901]),
    }


def make_fixture_geometry(path: Path) -> None:
    """Create a small seven-link test chain without external assets or nominal inertias."""
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    parent = ET.SubElement(world, "body", name="link0")
    for joint in range(1, 8):
        parent = ET.SubElement(parent, "body", name=f"link{joint}", pos="0 0 0.12")
        ET.SubElement(
            parent,
            "joint",
            name=f"joint{joint}",
            type="hinge",
            axis=("0 1 0" if joint % 2 else "1 0 0"),
            limited="false",
        )
    ET.ElementTree(root).write(path, encoding="unicode")


class TestRealRobotInputs(unittest.TestCase):
    """Reject altered observations and invalid physical candidates before simulation."""

    def test_clean_initialization_and_shapes(self):
        """Every replicate starts from identical homogeneous dynamics and independent lists."""
        first, second = initial_config(0), initial_config(8)
        self.assertEqual(first, second)
        self.assertEqual(first["mass"], [1.0] * 7)
        for key in ("viscous", "coulomb", "torque_bias", "armature"):
            self.assertEqual(first[key], [0.0] * 7)
        first["mass"][0] = 2.0
        self.assertEqual(second["mass"][0], 1.0)
        with self.assertRaises(ValueError):
            initial_config(-1)

    def test_physical_consistency_rejects_shortcuts(self):
        """Reject negative or nonphysical tensors, omitted coefficients and nonfinite values."""
        invalid = []
        for key, value in (
            ("mass", [0.0] * 7),
            ("com", [[0.6, 0, 0]] * 7),
            ("armature", [-0.1] * 7),
            ("viscous", [np.nan] * 7),
            ("inertia", [np.diag([0.001, 0.001, 0.1]).tolist()] * 7),
            ("inertia", [np.eye(3).tolist()] * 7),
        ):
            invalid.append(initial_config() | {key: value})
        invalid.extend([{}, initial_config() | {"controller_gain": 100}])
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_config(candidate)
        asymmetric = initial_config()
        asymmetric["inertia"][0][0][1] = 0.001
        with self.assertRaises(ValueError):
            validate_config(asymmetric)

    def test_parallel_axis_coefficient_mapping(self):
        """Map full off-diagonal COM tensors into the documented linear basis correctly."""
        config = initial_config()
        config["mass"][0] = 2.0
        config["com"][0] = [0.1, -0.05, 0.02]
        coefficients = physical_coefficients(config)
        self.assertEqual(coefficients.shape, (98,))
        np.testing.assert_allclose(coefficients[:4], [2.0, 0.2, -0.1, 0.04])
        np.testing.assert_allclose(coefficients[7:10], [0.01, -0.004, 0.002])

    def test_reference_integrity_and_irregular_interpolation(self):
        """Require complete finite episodes and preserve timestamp-based window targets."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            arrays = reference_arrays()
            np.savez(path, **arrays)
            loaded = load_reference(path)
            windows = prepare_windows(loaded)
            self.assertEqual(windows["q"].shape, (600, 7))
            self.assertEqual(windows["initial_q"].shape, (12, 7))
            np.testing.assert_allclose(
                windows["q"][0, 0], np.interp(arrays["time"][30] + 0.002, arrays["time"], arrays["q"][:, 0])
            )
            for changed in (
                arrays | {"hidden_model_mass": np.ones(7)},
                arrays | {"time": arrays["time"][::-1]},
                arrays | {"episode_offsets": np.array([1, 220])},
                arrays | {"tau": np.full((220, 7), np.nan)},
            ):
                np.savez(path, **changed)
                with self.assertRaises(ValueError):
                    load_reference(path)

    def test_pickle_rejects_arbitrary_global_constructors(self):
        """Do not execute arbitrary pickle globals during publisher-data conversion."""
        with self.assertRaises(pickle.UnpicklingError):
            _DataFrameUnpickler(io.BytesIO(b"cos\nsystem\n.")).load()

    def test_geometry_copy_removes_authored_dynamics(self):
        """Preserve kinematics while removing inertia, losses and controller values."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xml"
            source.write_text(
                '<mujoco><default><joint damping="123" armature="9" frictionloss="8"/><geom density="900" friction="3 2 1"/><general forcerange="-87 87"/></default><worldbody><body name="link1" pos="0 0 1"><joint name="j1" range="-2 2"/><inertial mass="9" pos=".1 .2 .3" diaginertia="1 2 3"/><geom type="sphere" size=".1" mass="99"/></body></worldbody><actuator><motor joint="j1" gear="33"/></actuator><keyframe><key qpos=".7"/></keyframe></mujoco>'
            )
            copied = prepare_geometry(source, root / "geometry")
            tree = ET.parse(copied)
            self.assertIsNone(tree.find(".//inertial"))
            self.assertIsNone(tree.find("actuator"))
            self.assertIsNone(tree.find("keyframe"))
            self.assertIsNone(tree.find(".//general"))
            self.assertEqual(tree.find("worldbody/body").get("pos"), "0 0 1")
            for joint in tree.findall(".//joint"):
                self.assertFalse({"damping", "armature", "frictionloss"} & set(joint.attrib))
            for geom in tree.findall(".//geom"):
                self.assertEqual(geom.get("density"), "0")
                self.assertNotIn("mass", geom.attrib)


class TestRealRobotPhysics(unittest.TestCase):
    """Exercise public Newton calculations on a small asset-independent robot fixture."""

    @classmethod
    def setUpClass(cls):
        """Prepare one numeric fixture and immutable matrix for all physical checks."""
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.geometry = cls.root / "geometry.xml"
        make_fixture_geometry(cls.geometry)
        cls.reference = cls.root / "training.npz"
        np.savez(cls.reference, **reference_arrays())
        cls.regressor = cls.root / "training-regressor.npz"
        prepare_regressor(cls.reference, cls.geometry, cls.regressor)

    @classmethod
    def tearDownClass(cls):
        """Remove only this test's temporary inputs."""
        cls.directory.cleanup()

    def scenario(self, config=None):
        """Construct the same fixture through the production scenario API."""
        return RealRobotScenario(config, reference_file=self.reference, geometry_file=self.geometry)

    def test_regressor_matches_direct_newton_full_tensors(self):
        """Check full COM/off-diagonal coefficients against an independently constructed model."""
        config = initial_config()
        config["mass"] = np.linspace(0.8, 2.0, 7).tolist()
        config["com"] = [[0.04, -0.02, 0.03]] * 7
        config["inertia"] = [[[0.02, 0.002, -0.001], [0.002, 0.018, 0.001], [-0.001, 0.001, 0.015]]] * 7
        config["viscous"], config["coulomb"], config["torque_bias"], config["armature"] = [
            [value] * 7 for value in (0.2, 0.1, -0.02, 0.04)
        ]
        q, qd, qdd = np.linspace(-0.3, 0.4, 7), np.linspace(-0.7, 0.6, 7), np.linspace(0.8, -1, 7)
        matrix = inverse_dynamics_matrix(self.geometry, q[None], qd[None], qdd[None])
        with wp.ScopedDevice("cpu"):
            builder, _ = make_builder(self.geometry, config, visual=False)
            model = builder.finalize(device="cpu")
            state = model.state()
            state.joint_q.assign(q.astype(np.float32))
            state.joint_qd.assign(qd.astype(np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            mass = wp.zeros((1, 7, 7), dtype=float)
            coriolis, gravity, force = (wp.zeros(7, dtype=float) for _ in range(3))
            newton.eval_inverse_dynamics_passive(
                model, state, mass_matrix=mass, coriolis_force=coriolis, gravity_force=gravity
            )
            newton.eval_inverse_dynamics_force(
                model,
                state,
                mass_matrix=mass,
                joint_qdd=wp.array(qdd.astype(np.float32), dtype=float),
                coriolis_force=coriolis,
                gravity_force=gravity,
                joint_f=force,
            )
            expected = force.numpy() + 0.2 * qd + 0.1 * np.sign(qd) - 0.02 + 0.04 * qdd
        np.testing.assert_allclose(matrix @ physical_coefficients(config), expected, atol=1e-5, rtol=2e-5)

    def test_live_full_inertia_edit_and_reset_match_fresh(self):
        """Detect missing native COM, inertia, friction, damping or armature notifications."""
        config = initial_config()
        config.update(
            mass=[2.0] * 7,
            com=[[0.01, -0.02, 0.015]] * 7,
            inertia=[[[0.02, 0.001, 0], [0.001, 0.018, 0], [0, 0, 0.016]]] * 7,
            viscous=[0.3] * 7,
            coulomb=[0.15] * 7,
            armature=[0.08] * 7,
            torque_bias=[0.1] * 7,
        )
        live, fresh = self.scenario(), self.scenario(config)
        live.apply_config(config)
        live.rollout()
        fresh.rollout()
        np.testing.assert_allclose(live.q_trace, fresh.q_trace, atol=2e-6, rtol=0)
        np.testing.assert_allclose(live.qd_trace, fresh.qd_trace, atol=2e-5, rtol=0)
        previous_q, previous_qd = np.asarray(live.q_trace).copy(), np.asarray(live.qd_trace).copy()
        live.rollout()
        np.testing.assert_array_equal(live.q_trace, previous_q)
        np.testing.assert_array_equal(live.qd_trace, previous_qd)
        self.assertEqual(live.metrics()["sample_count"], 600)
        self.assertTrue(live.metrics()["finite"])
        live.save_trace(self.root / "candidate.npz")
        with np.load(self.root / "candidate.npz", allow_pickle=False) as data:
            self.assertEqual(data["reference_q"].shape, (600, 7))
            self.assertEqual(data["torque_prediction"].shape, (150, 7))

    def test_incomplete_candidate_cannot_pass(self):
        """Do not allow a short prefix to stand in for complete recorded windows."""
        scenario = self.scenario()
        self.assertFalse(scenario.metrics()["success"])
        for _ in range(50):
            scenario.step()
        result = scenario.metrics()
        self.assertFalse(result["success"])
        self.assertEqual(result["expected_frames"], 600)
        self.assertEqual(result["per_episode"][0]["sample_count"], 50)

    def test_regressor_hash_prevents_replaced_inputs(self):
        """Reject a matrix that differs from its fixed provenance manifest."""
        copy = self.root / "changed-regressor.npz"
        copy.write_bytes(self.regressor.read_bytes() + b"changed")
        copy.with_suffix(".manifest.json").write_text(self.regressor.with_suffix(".manifest.json").read_text())
        with self.assertRaisesRegex(ValueError, "regressor_sha256"):
            RealRobotScenario(reference_file=self.reference, geometry_file=self.geometry, regressor_file=copy)

    def test_pooled_success_does_not_hide_one_failed_recording(self):
        """Reject one inaccurate recording even when the pooled error passes every threshold."""
        scenario = self.scenario()
        scenario.episodes = [901, 902]
        scenario.steps = scenario.frame = 1200
        for key in ("q", "qd"):
            scenario.windows[key] = np.concatenate([scenario.windows[key]] * 2)
        scenario.windows["episode_ids"] = np.repeat([901, 902], 600)
        scenario.matrix = np.zeros((2100, 98))
        scenario.torque_target = np.zeros(2100)
        scenario.torque_episode_ids = np.repeat([901, 902], 150)
        scenario.q_trace = scenario.windows["q"].copy()
        scenario.q_trace[:600] += 0.03
        scenario.qd_trace = scenario.windows["qd"].copy()
        result = scenario.metrics()
        self.assertLess(result["max_joint_position_rmse_rad"], 0.025)
        self.assertFalse(result["per_episode"][0]["success"])
        self.assertTrue(result["per_episode"][1]["success"])
        self.assertFalse(result["success"])

    def test_nonzero_checkpoint_requires_complete_candidate_reset(self):
        """Reject resumed diagnostics with missing history and recover through a frame-zero reset."""
        scenario = self.scenario()
        session = make_session(scenario, self.root / "observations")
        session.dispatch("checkpoint", {"name": "start"})
        session.dispatch("step", {"count": 51})
        session.dispatch("checkpoint", {"name": "moving"})
        session.dispatch("step", {"count": 2})
        with self.assertRaisesRegex(ValueError, "complete measurement history"):
            session.dispatch("restore", {"name": "moving"})
        with self.assertRaisesRegex(ValueError, "Reset the complete candidate"):
            scenario.metrics()
        with self.assertRaisesRegex(ValueError, "Reset the complete candidate"):
            scenario.step()
        session.dispatch("restore", {"name": "start"})
        session.dispatch("step", {"count": 51})
        first = np.asarray(scenario.q_trace).copy()
        session.dispatch("reset")
        session.dispatch("step", {"count": 51})
        np.testing.assert_array_equal(first, scenario.q_trace)
        self.assertEqual(scenario.metrics()["sample_count"], 51)

    def test_nonfinite_candidates_retain_equal_work_across_paths(self):
        """Retain a failed frame and complete identical fixed work through fresh and session paths."""
        candidates = []
        for live in (False, True):
            scenario = self.scenario()
            original_step = scenario.solver.step
            calls = []

            def injected_step(state, state_next, control, contacts, dt, _original=original_step, _calls=calls):
                _original(state, state_next, control, contacts, dt)
                _calls.append(1)
                if len(_calls) == 3:
                    q = state_next.joint_q.numpy().copy()
                    q[0] = np.nan
                    state_next.joint_q.assign(q)

            with patch.object(scenario.solver, "step", injected_step):
                if live:
                    make_session(scenario, self.root / "nonfinite-observations").dispatch(
                        "step", {"count": scenario.steps}
                    )
                else:
                    scenario.rollout()
            self.assertFalse(scenario.metrics()["finite"])
            self.assertFalse(scenario.metrics()["success"])
            self.assertEqual(scenario.frame, scenario.steps)
            candidates.append(scenario)
        np.testing.assert_array_equal(candidates[0].q_trace, candidates[1].q_trace)


if __name__ == "__main__":
    unittest.main()
