# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton import CameraPinhole, ModelBuilder
from newton._src.core.cameras import xform_to_pitch_yaw
from newton._src.viewer.camera import Camera
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

    def test_viewer_null_camera_frustum_depth_rebuild(self):
        """Changing camera_frustum_depth between log_state calls must rebuild frustums."""
        model = self._model()
        viewer = newton.viewer.ViewerNull()
        viewer.set_model(model)
        viewer.show_cameras = True
        viewer.begin_frame(0.0)
        viewer.log_state(model.state())
        viewer.camera_frustum_depth = 1.5
        viewer.log_state(model.state())
        viewer.end_frame()
        self.assertEqual(viewer._camera_frustums.depth, 1.5)


class TestViewFromCamera(unittest.TestCase):
    def test_xform_to_pitch_yaw_identity_z_up(self):
        """Verify camera forward pointing along world -Z (downward in Z-up space) yields pitch -90."""
        xf = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
        pos, pitch, _yaw = xform_to_pitch_yaw(xf, up_axis=2)
        self.assertAlmostEqual(float(pos[2]), 3.0, places=5)
        self.assertAlmostEqual(pitch, -90.0, places=3)

    def test_xform_to_pitch_yaw_round_trip_y_up(self):
        """Verify xform_to_pitch_yaw pitch/yaw agree with viewer Camera.look_at for Y-up."""
        # Camera at (0, 2, 5) looking at (1, 0, 0) with Y-up.
        cam_pos = (0.0, 2.0, 5.0)
        target = (1.0, 0.0, 0.0)
        cam = Camera(up_axis=1)
        cam.pos = cam._as_vec3(cam_pos)
        cam.look_at(target)
        expected_pitch = cam.pitch
        expected_yaw = cam.yaw

        # Build a warp transform whose -Z forward points at the same target.
        direction = np.array(target, dtype=np.float64) - np.array(cam_pos, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        # Compute quaternion that rotates (0,0,-1) to direction.
        z_neg = np.array([0.0, 0.0, -1.0])
        axis = np.cross(z_neg, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-8:
            q = wp.quat_identity()
        else:
            axis /= axis_len
            angle = math.acos(float(np.clip(np.dot(z_neg, direction), -1.0, 1.0)))
            q = wp.quat_from_axis_angle(wp.vec3(*axis.tolist()), angle)
        xf = wp.transform(cam_pos, q)
        _pos, pitch, yaw = xform_to_pitch_yaw(xf, up_axis=1)
        self.assertAlmostEqual(pitch, expected_pitch, places=3)
        self.assertAlmostEqual(yaw, expected_yaw, places=3)

    def test_set_camera_from_model_by_label(self):
        """Verify set_camera_from_model resolves labels and positions the viewport camera."""
        builder = ModelBuilder()
        builder.add_camera(xform=wp.transform((5.0, 0.0, 1.0), wp.quat_identity()), label="cam_a")
        model = builder.finalize()
        viewer = newton.viewer.ViewerNull()
        viewer.set_model(model)
        viewer.set_camera_from_model("cam_a")  # must not raise
        with self.assertRaises(KeyError):
            viewer.set_camera_from_model("missing")

    def test_set_camera_from_model_by_index(self):
        """Verify set_camera_from_model accepts integer camera index."""
        builder = ModelBuilder()
        builder.add_camera(xform=wp.transform((3.0, 1.0, 2.0), wp.quat_identity()), label="cam0")
        model = builder.finalize()
        viewer = newton.viewer.ViewerNull()
        viewer.set_model(model)
        viewer.set_camera_from_model(0)  # must not raise

    def test_set_camera_from_model_no_model(self):
        """Verify set_camera_from_model is a no-op when no model is set."""
        viewer = newton.viewer.ViewerNull()
        viewer.set_camera_from_model(0)  # must not raise

    def test_xform_to_pitch_yaw_exported(self):
        """Verify xform_to_pitch_yaw is accessible from newton._src.core.cameras."""
        self.assertTrue(callable(xform_to_pitch_yaw))

    def test_set_camera_from_model_adopts_fov(self):
        """Verify set_camera_from_model sets camera.fov to degrees(projection.fov)."""

        class _FakeViewport:
            def __init__(self):
                self.fov = 45.0

        class _ViewerWithCamera(newton.viewer.ViewerNull):
            def __init__(self):
                super().__init__()
                self.camera = _FakeViewport()

        fov_rad = math.radians(60.0)
        builder = ModelBuilder()
        builder.add_camera(
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            projection=newton.CameraPinhole.from_fov(fov_rad, aspect=1.0),
        )
        model = builder.finalize()
        viewer = _ViewerWithCamera()
        viewer.set_model(model)
        viewer.set_camera_from_model(0)
        self.assertAlmostEqual(viewer.camera.fov, math.degrees(fov_rad), places=3)


class TestCameraMathUnification(unittest.TestCase):
    def test_basis_matches_viewport_camera(self):
        """Verify pitch_yaw_to_basis reproduces the viewport Camera vectors for all up axes."""
        from newton._src.core.cameras import pitch_yaw_to_basis

        for up_axis in (0, 1, 2):
            for pitch, yaw in [(0.0, 0.0), (-30.0, 45.0), (60.0, -120.0), (-89.0, 179.0)]:
                cam = Camera(up_axis=up_axis)
                cam.pitch = pitch
                cam.yaw = yaw
                front, right, up = pitch_yaw_to_basis(pitch, yaw, up_axis)
                expected_front = cam.get_front()
                for k in range(3):
                    self.assertAlmostEqual(
                        float(front[k]), float(expected_front[k]), places=5,
                        msg=f"up_axis={up_axis} pitch={pitch} yaw={yaw}",
                    )

    def test_fov_to_focal_length_roundtrip(self):
        """Verify fov/focal conversion inverts CameraPinhole.from_fov."""
        from newton._src.core.cameras import fov_to_focal_length

        fov = math.radians(50.0)
        proj = newton.CameraPinhole.from_fov(fov)
        self.assertAlmostEqual(
            fov_to_focal_length(fov, proj.vertical_aperture), proj.focal_length, places=5
        )

    def test_basis_right_up_orthonormal(self):
        """Verify pitch_yaw_to_basis returns orthonormal (front, right, up) for all axes."""
        from newton._src.core.cameras import pitch_yaw_to_basis

        for up_axis in (0, 1, 2):
            for pitch, yaw in [(0.0, 0.0), (-30.0, 45.0), (60.0, -120.0), (-89.0, 179.0)]:
                front, right, up = pitch_yaw_to_basis(pitch, yaw, up_axis)
                # Each vector should be unit length.
                for name, v in [("front", front), ("right", right), ("up", up)]:
                    length = math.sqrt(sum(float(v[k]) ** 2 for k in range(3)))
                    self.assertAlmostEqual(length, 1.0, places=5,
                                          msg=f"{name} not unit: up_axis={up_axis} pitch={pitch} yaw={yaw}")
                # Vectors should be mutually orthogonal.
                dot_fr = sum(float(front[k]) * float(right[k]) for k in range(3))
                dot_fu = sum(float(front[k]) * float(up[k]) for k in range(3))
                dot_ru = sum(float(right[k]) * float(up[k]) for k in range(3))
                self.assertAlmostEqual(dot_fr, 0.0, places=5,
                                       msg=f"front·right non-zero: up_axis={up_axis} pitch={pitch} yaw={yaw}")
                self.assertAlmostEqual(dot_fu, 0.0, places=5,
                                       msg=f"front·up non-zero: up_axis={up_axis} pitch={pitch} yaw={yaw}")
                self.assertAlmostEqual(dot_ru, 0.0, places=5,
                                       msg=f"right·up non-zero: up_axis={up_axis} pitch={pitch} yaw={yaw}")


if __name__ == "__main__":
    unittest.main()
