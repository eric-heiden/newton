# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import warp as wp

import newton
from newton import CameraFlags, CameraPinhole, Model, ModelBuilder


class TestCameraReplication(unittest.TestCase):
    def _make_env(self):
        env = ModelBuilder()
        body = env.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        env.add_shape_sphere(body)
        env.add_camera(body=body, label="wrist", projection=CameraPinhole.from_fov(math.radians(70.0)))
        env.add_camera(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), label="static")
        return env, body

    def test_replicate_offsets_cameras(self):
        """Verify replicate duplicates cameras with per-world body offsets and transforms."""
        env, _ = self._make_env()
        scene = ModelBuilder()
        scene.replicate(env, world_count=3, spacing=(10.0, 0.0, 0.0))
        model = scene.finalize()
        self.assertEqual(model.camera_count, 6)
        self.assertEqual(model.camera_world.numpy().tolist(), [0, 0, 1, 1, 2, 2])
        bodies = model.camera_body.numpy()
        self.assertEqual(bodies[0], 0)
        self.assertEqual(bodies[2], 1)
        self.assertEqual(bodies[4], 2)
        # static cameras in later worlds are offset by the per-world spacing:
        # world 0 keeps the source position; other worlds must differ from it
        # (compute_world_offsets may arrange worlds in a grid, so don't assert
        # a specific axis — assert distinctness and that world 0 is unchanged)
        xf = model.camera_transform.numpy()
        # compute_world_offsets centres the grid, so the middle world (world index 1)
        # receives an identity offset and its static cam stays at source x=1.0
        self.assertAlmostEqual(float(xf[3][0]), 1.0, places=5)  # world 1 static cam (centre)
        positions = {tuple(round(float(v), 4) for v in xf[i][:3]) for i in (1, 3, 5)}
        self.assertEqual(len(positions), 3)
        # a single shared projection object across all replicas
        self.assertEqual(len(model.camera_projections), 2)

    def test_add_world_label_prefix(self):
        """Verify add_world applies the label prefix to camera labels."""
        env, _ = self._make_env()
        scene = ModelBuilder()
        scene.add_world(env, label_prefix="env0")
        model = scene.finalize()
        self.assertIn("env0/wrist", model.camera_label)


class TestCameraSchema(unittest.TestCase):
    def test_camera_attribute_frequency_registered(self):
        """Verify the CAMERA attribute frequency exists as the next value after WORLD."""
        freq = Model.AttributeFrequency.CAMERA
        self.assertEqual(int(freq), int(Model.AttributeFrequency.WORLD) + 1)

    def test_camera_attribute_frequencies(self):
        """Verify per-camera attributes are registered with the CAMERA frequency."""
        model = ModelBuilder().finalize()
        freq = model.attribute_frequency
        for name in ("camera_label", "camera_transform", "camera_body", "camera_world", "camera_flags"):
            self.assertEqual(freq[name], Model.AttributeFrequency.CAMERA)
        for name in ("camera_projection_index", "camera_resolution"):
            self.assertEqual(freq[name], Model.AttributeFrequency.CAMERA)
        self.assertEqual(freq["camera_projections"], Model.AttributeFrequency.ONCE)

    def test_camera_flags(self):
        """Verify CameraFlags bitmask values."""
        self.assertEqual(int(newton.CameraFlags.ENABLED), 1)
        self.assertEqual(int(newton.CameraFlags.VISIBLE), 2)

    def test_empty_model_has_camera_fields(self):
        """Verify a finalized empty model exposes zeroed camera fields."""
        model = ModelBuilder().finalize()
        self.assertEqual(model.camera_count, 0)
        self.assertEqual(model.camera_label, [])
        self.assertEqual(model.camera_projections, [])


class TestAddCamera(unittest.TestCase):
    def test_add_camera_defaults(self):
        """Verify add_camera with defaults creates a world-fixed enabled camera."""
        builder = ModelBuilder()
        cam = builder.add_camera()
        self.assertEqual(cam, 0)
        self.assertEqual(builder.camera_count, 1)
        model = builder.finalize()
        self.assertEqual(model.camera_count, 1)
        self.assertEqual(model.camera_body.numpy()[0], -1)
        self.assertEqual(model.camera_world.numpy()[0], -1)
        self.assertEqual(model.camera_label, ["camera_0"])
        self.assertTrue(model.camera_flags.numpy()[0] & int(CameraFlags.ENABLED))
        self.assertEqual(list(model.camera_resolution.numpy()[0]), [-1, -1])

    def test_add_camera_world_and_body(self):
        """Verify cameras record the current world and validate body indices."""
        builder = ModelBuilder()
        builder.begin_world()
        body = builder.add_body()
        cam = builder.add_camera(body=body, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        builder.end_world()
        model = builder.finalize()
        self.assertEqual(model.camera_world.numpy()[cam], 0)
        self.assertEqual(model.camera_body.numpy()[cam], body)
        with self.assertRaises(ValueError):
            ModelBuilder().add_camera(body=5)  # invalid body index

    def test_projection_dedup(self):
        """Verify equal projections collapse to one entry in camera_projections."""
        builder = ModelBuilder()
        shared = CameraPinhole.from_fov(math.radians(60.0))
        builder.add_camera(projection=shared)
        builder.add_camera(projection=CameraPinhole.from_fov(math.radians(60.0)))
        builder.add_camera(projection=CameraPinhole.from_fov(math.radians(90.0)))
        model = builder.finalize()
        self.assertEqual(len(model.camera_projections), 2)
        idx = model.camera_projection_index.numpy()
        self.assertEqual(idx[0], idx[1])
        self.assertNotEqual(idx[0], idx[2])

    def test_camera_world_start(self):
        """Verify camera_world_start follows the standard world-start layout."""
        builder = ModelBuilder()
        builder.add_camera()  # global
        builder.begin_world()
        builder.add_camera()
        builder.add_camera()
        builder.end_world()
        model = builder.finalize()
        # layout: [start_w0, start_global_tail, total]; global head count = start_w0
        starts = model.camera_world_start.numpy().tolist()
        self.assertEqual(starts[-1], 3)
        self.assertEqual(starts[1] - starts[0], 2)  # world 0 has 2 cameras

    def test_camera_custom_attribute(self):
        """Verify CAMERA-frequency custom attributes flow from builder to model."""
        builder = ModelBuilder()
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="exposure",
                dtype=wp.float32,
                frequency=Model.AttributeFrequency.CAMERA,
                default=1.0,
            )
        )
        builder.add_camera(custom_attributes={"exposure": 2.5})
        builder.add_camera()
        model = builder.finalize()
        vals = model.exposure.numpy()
        self.assertAlmostEqual(float(vals[0]), 2.5)
        self.assertAlmostEqual(float(vals[1]), 1.0)

    def test_add_builder_camera_frequency_custom_attribute(self):
        """Merge two sub-builders that each have a camera with a CAMERA-frequency custom attribute."""
        sub = ModelBuilder()
        sub.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="exposure",
                dtype=wp.float32,
                frequency=Model.AttributeFrequency.CAMERA,
                default=1.0,
            )
        )
        sub.add_camera(custom_attributes={"exposure": 2.5})

        main = ModelBuilder()
        main.add_builder(sub)
        main.add_builder(sub)
        model = main.finalize()

        self.assertEqual(model.camera_count, 2)
        vals = model.exposure.numpy()
        self.assertAlmostEqual(float(vals[0]), 2.5)
        self.assertAlmostEqual(float(vals[1]), 2.5)


if __name__ == "__main__":
    unittest.main()
