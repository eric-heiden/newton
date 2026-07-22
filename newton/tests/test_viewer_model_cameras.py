# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import warp as wp

import newton
from newton import CameraPinhole, ModelBuilder
from newton._src.viewer.camera_frustums import CameraFrustums


class TestCameraFrustums(unittest.TestCase):
    def _model(self):
        builder = ModelBuilder()
        builder.add_camera(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
            projection=CameraPinhole.from_fov(math.radians(60.0), aspect=1.0),
        )
        return builder.finalize()

    def test_frustum_line_counts(self):
        """Verify 12 line segments are generated per camera."""
        model = self._model()
        frustums = CameraFrustums(model)
        frustums.update(model.state(), world_offsets=None)
        self.assertEqual(frustums.starts.shape[0], 12)
        self.assertEqual(frustums.ends.shape[0], 12)

    def test_frustum_apex_near_camera(self):
        """Verify frustum geometry is positioned at the camera's world pose."""
        model = self._model()
        frustums = CameraFrustums(model)
        frustums.update(model.state(), world_offsets=None)
        starts = frustums.starts.numpy()
        # all segment endpoints lie within frustum depth of the camera position
        for p in starts:
            self.assertLess(abs(float(p[2]) - 2.0), 1.0)

    def test_viewer_null_show_cameras_smoke(self):
        """Verify ViewerNull renders a state with show_cameras enabled without error."""
        model = self._model()
        viewer = newton.viewer.ViewerNull()
        viewer.set_model(model)
        viewer.show_cameras = True
        viewer.begin_frame(0.0)
        viewer.log_state(model.state())
        viewer.end_frame()


if __name__ == "__main__":
    unittest.main()
