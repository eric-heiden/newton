# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import warp as wp

import newton
from newton._src.core.cameras import compute_camera_rays


class TestComputeCameraRays(unittest.TestCase):
    def test_pinhole_center_ray_points_forward(self):
        """Verify the center pixel of a symmetric pinhole projection looks along -Z.

        For a 64-wide image with pixel-center convention, pixel 32 maps to
        u = 32.5/64 = 0.5078, which is half a pixel off-axis.  The z-component
        check uses places=2; the x-component check uses places=1 to accommodate
        this half-pixel offset (the off-axis angle is ~0.009, within 1 dp).
        """
        proj = newton.CameraPinhole.from_fov(math.radians(60.0), aspect=1.0)
        rays = compute_camera_rays(proj, 64, 64)
        self.assertEqual(rays.shape, (64, 64, 2))
        d = rays.numpy()[32, 32, 1]
        self.assertAlmostEqual(float(d[2]), -1.0, places=2)
        self.assertAlmostEqual(float(d[0]), 0.0, places=1)

    def test_pinhole_corner_ray_matches_fov(self):
        """Verify the top edge ray elevation is close to half the vertical fov."""
        fov = math.radians(90.0)
        proj = newton.CameraPinhole.from_fov(fov, aspect=1.0)
        rays = compute_camera_rays(proj, 200, 200)
        d = rays.numpy()[0, 100, 1]  # top row, center column
        elevation = math.atan2(float(d[1]), -float(d[2]))
        self.assertAlmostEqual(elevation, fov / 2.0, delta=math.radians(1.0))

    def test_ray_origins_zero_and_directions_normalized(self):
        """Verify pinhole ray origins are zero and directions unit length."""
        proj = newton.CameraPinhole.from_fov(math.radians(45.0))
        rays = compute_camera_rays(proj, 16, 8).numpy()
        self.assertAlmostEqual(float(abs(rays[:, :, 0]).max()), 0.0, places=6)
        norms = (rays[:, :, 1] ** 2).sum(axis=-1) ** 0.5
        self.assertAlmostEqual(float(norms.min()), 1.0, places=4)
        self.assertAlmostEqual(float(norms.max()), 1.0, places=4)

    def test_custom_rays_returned_verbatim(self):
        """Verify CameraCustomRays bundles are returned as-is and shape mismatches raise."""
        bundle = wp.zeros((4, 8, 2), dtype=wp.vec3f)
        proj = newton.CameraCustomRays(rays=bundle)
        self.assertIs(compute_camera_rays(proj, 8, 4), bundle)
        with self.assertRaises(ValueError):
            compute_camera_rays(proj, 16, 16)

    def test_fisheye_opencv_generates_bundle(self):
        """Verify the OpenCV fisheye projection generates a bundle with a forward center ray."""
        proj = newton.CameraFisheyeOpenCV(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
        rays = compute_camera_rays(proj, 64, 64)
        d = rays.numpy()[32, 32, 1]
        self.assertLess(float(d[2]), -0.99)

    def test_fisheye_opencv_calibration_size_produces_different_rays(self):
        """Assert that a 2x calibration size yields rays distinct from the default (None) calibration."""
        width, height = 32, 32
        proj_default = newton.CameraFisheyeOpenCV(fx=100.0, fy=100.0, cx=16.0, cy=16.0)
        proj_calib = newton.CameraFisheyeOpenCV(
            fx=100.0,
            fy=100.0,
            cx=16.0,
            cy=16.0,
            image_width=float(width * 2),
            image_height=float(height * 2),
        )
        rays_default = compute_camera_rays(proj_default, width, height).numpy()
        rays_calib = compute_camera_rays(proj_calib, width, height).numpy()
        self.assertFalse(
            (rays_default == rays_calib).all(),
            "Expected rays to differ when calibration size is 2x the render size",
        )


if __name__ == "__main__":
    unittest.main()
