# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import warp as wp

import newton
from newton import CameraFlags, CameraPinhole, Model, ModelBuilder


class TestCameraSchema(unittest.TestCase):
    def test_camera_attribute_frequency_registered(self):
        """Verify the CAMERA attribute frequency and its count attribute mapping exist."""
        freq = Model.AttributeFrequency.CAMERA
        self.assertEqual(int(freq), 16)
        self.assertEqual(Model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS[freq], "camera_count")

    def test_camera_core_attribute_specs(self):
        """Verify camera attribute specs are registered with correct references."""
        specs = Model._CORE_ATTRIBUTE_SPECS
        self.assertEqual(specs["camera_body"].references, Model.AttributeFrequency.BODY)
        self.assertEqual(specs["camera_world"].references, Model.AttributeFrequency.WORLD)
        self.assertEqual(specs["camera_world_start"].compaction_policy, "world_start")
        self.assertEqual(specs["camera_resolution"].row_width, 2)
        for name in ("camera_label", "camera_transform", "camera_flags", "camera_projection_index"):
            self.assertEqual(specs[name].frequency, Model.AttributeFrequency.CAMERA)

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


if __name__ == "__main__":
    unittest.main()
