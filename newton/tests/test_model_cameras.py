# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
from newton import Model, ModelBuilder


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


if __name__ == "__main__":
    unittest.main()
