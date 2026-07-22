# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from newton import ModelBuilder

MJCF = """
<mujoco>
  <worldbody>
    <camera name="overview" pos="0 -3 2" fovy="60" resolution="640 480"/>
    <body name="torso" pos="0 0 1">
      <geom type="sphere" size="0.1"/>
      <camera name="head_cam" pos="0 0 0.3" fovy="90"/>
      <camera name="tracker" mode="trackcom" pos="1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


class TestImportMjcfCameras(unittest.TestCase):
    def _parse(self):
        builder = ModelBuilder()
        builder.add_mjcf(MJCF)
        return builder, builder.finalize()

    def test_cameras_imported(self):
        """Verify MJCF camera elements are imported as model cameras."""
        _, model = self._parse()
        self.assertEqual(model.camera_count, 3)
        labels = model.camera_label
        self.assertTrue(any(label.endswith("overview") for label in labels))

    def test_fovy_and_resolution(self):
        """Verify fovy [deg] maps to the projection fov [rad] and resolution to the hint."""
        _, model = self._parse()
        i = next(i for i, label in enumerate(model.camera_label) if label.endswith("overview"))
        proj = model.camera_projections[model.camera_projection_index.numpy()[i]]
        self.assertAlmostEqual(proj.fov, math.radians(60.0), places=5)
        self.assertEqual(list(model.camera_resolution.numpy()[i]), [640, 480])

    def test_body_attachment(self):
        """Verify body-level cameras attach to the parent body with a relative transform."""
        _, model = self._parse()
        i = next(i for i, label in enumerate(model.camera_label) if label.endswith("head_cam"))
        self.assertGreaterEqual(int(model.camera_body.numpy()[i]), 0)
        self.assertAlmostEqual(float(model.camera_transform.numpy()[i][2]), 0.3, places=5)

    def test_tracking_mode_warns_and_imports_fixed(self):
        """Verify non-fixed camera modes import as fixed cameras with a warning."""
        with self.assertWarns(UserWarning):
            _, model = self._parse()
        self.assertEqual(model.camera_count, 3)

    def test_camera_mode_preserved_as_custom_attribute(self):
        """Verify the original MJCF camera mode string is preserved on the model."""
        _, model = self._parse()
        i = next(i for i, label in enumerate(model.camera_label) if label.endswith("tracker"))
        # attribute is namespaced under mjcf; registered lazily by the importer
        modes = model.mjcf.camera_mode
        self.assertEqual(modes[i], "trackcom")


MJCF_FRAME_CAMERA = """
<mujoco>
  <worldbody>
    <frame pos="0 0 1">
      <camera name="framed" pos="0 0 0.5"/>
    </frame>
  </worldbody>
</mujoco>
"""


class TestImportMjcfFrameCameras(unittest.TestCase):
    def test_frame_nested_camera_composed_transform(self):
        """Import a camera inside a frame and verify its transform is composed with the frame offset."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_FRAME_CAMERA)
        model = builder.finalize()
        self.assertEqual(model.camera_count, 1)
        labels = model.camera_label
        self.assertTrue(any(label.endswith("framed") for label in labels))
        i = next(i for i, label in enumerate(labels) if label.endswith("framed"))
        # frame pos=(0,0,1) + camera pos=(0,0,0.5) → z == 1.5
        self.assertAlmostEqual(float(model.camera_transform.numpy()[i][2]), 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
