# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import warp as wp

import newton


class TestCameraProjections(unittest.TestCase):
    def test_pinhole_from_fov_roundtrip(self):
        """Verify CameraPinhole.from_fov produces a projection whose fov property returns the input."""
        fov = math.radians(60.0)
        proj = newton.CameraPinhole.from_fov(fov)
        self.assertAlmostEqual(proj.fov, fov, places=6)

    def test_pinhole_from_fov_aspect_sets_horizontal_aperture(self):
        """Verify the aspect argument scales the horizontal aperture relative to the vertical one."""
        proj = newton.CameraPinhole.from_fov(math.radians(45.0), aspect=2.0)
        self.assertAlmostEqual(proj.horizontal_aperture, 2.0 * proj.vertical_aperture, places=6)

    def test_parametric_equality_and_hash(self):
        """Verify equal parametric projections compare and hash equal for dedup."""
        a = newton.CameraPinhole.from_fov(math.radians(60.0))
        b = newton.CameraPinhole.from_fov(math.radians(60.0))
        c = newton.CameraPinhole.from_fov(math.radians(90.0))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)

    def test_projection_immutable(self):
        """Verify projection descriptors are frozen."""
        proj = newton.CameraPinhole.from_fov(math.radians(60.0))
        with self.assertRaises(Exception):
            proj.focal_length = 1.0

    def test_custom_rays_identity_equality(self):
        """Verify CameraCustomRays compares by object identity, not ray contents."""
        rays1 = wp.zeros((4, 8, 2), dtype=wp.vec3f)
        rays2 = wp.zeros((4, 8, 2), dtype=wp.vec3f)
        a = newton.CameraCustomRays(rays=rays1)
        b = newton.CameraCustomRays(rays=rays2)
        self.assertEqual(a, a)
        self.assertNotEqual(a, b)
        self.assertEqual(a.resolution, (8, 4))  # (width, height)

    def test_custom_rays_validates_shape(self):
        """Verify CameraCustomRays rejects arrays that are not (H, W, 2) of vec3f."""
        with self.assertRaises(ValueError):
            newton.CameraCustomRays(rays=wp.zeros((4, 8), dtype=wp.vec3f))

    def test_fisheye_classes_exist(self):
        """Verify fisheye projection descriptors construct and are hashable."""
        f = newton.CameraFisheyeOpenCV(fx=300.0, fy=300.0, cx=320.0, cy=240.0)
        self.assertIsInstance(hash(f), int)


if __name__ == "__main__":
    unittest.main()
