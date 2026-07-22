# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
from newton import Model, ModelBuilder


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


if __name__ == "__main__":
    unittest.main()
