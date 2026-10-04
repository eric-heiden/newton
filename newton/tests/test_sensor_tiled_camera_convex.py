# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices, ignore_sensor_tiled_camera_deprecation


def setUpModule():
    # SensorTiledCamera is deprecated; these tests exercise it intentionally.
    ignore_sensor_tiled_camera_deprecation()


def test_convex_hull_is_rendered(test: unittest.TestCase, device):
    """Convex-hull shapes show up in renders like the mesh they are built from."""
    depths = []
    for convex in (True, False):
        builder = newton.ModelBuilder()
        mesh = newton.Mesh.create_box(0.2, 0.2, 0.1, compute_inertia=False)
        if convex:
            builder.add_shape_convex_hull(-1, mesh=mesh)
        else:
            builder.add_shape_mesh(-1, mesh=mesh)
        model = builder.finalize(device=device)
        test.assertEqual(
            int(model.shape_type.numpy()[0]), int(newton.GeoType.CONVEX_MESH if convex else newton.GeoType.MESH)
        )
        sensor = SensorTiledCamera(model)
        rays = sensor.utils.compute_camera_rays_pinhole(16, 16, camera_fovs=math.radians(30.0))
        eye = wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity())
        transforms = wp.array([[eye]], dtype=wp.transformf, device=device)
        depth = sensor.utils.create_depth_image_output(16, 16)
        sensor.update(model.state(), transforms, rays, depth_image=depth)
        depths.append(depth.numpy()[0, 0])
    # The camera looks straight down at the top face (z = 0.1), 1.9 m away.
    test.assertGreater(depths[0][8, 8], 0.0)
    np.testing.assert_allclose(depths[0][8, 8], 1.9, atol=1e-3)
    np.testing.assert_allclose(depths[0], depths[1], atol=1e-3)


class TestSensorTiledCameraConvex(unittest.TestCase):
    pass


add_function_test(
    TestSensorTiledCameraConvex,
    "test_convex_hull_is_rendered",
    test_convex_hull_is_rendered,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
