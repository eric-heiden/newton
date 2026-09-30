# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_transparent_shapes_are_not_rendered(test: unittest.TestCase, device):
    """A fully transparent shape in front of the camera does not occlude what is behind it."""
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.5)
    # An invisible helper (MJCF rgba alpha 0) between the camera and the sphere.
    builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 1.5), wp.quat_identity()), hx=1.0, hy=1.0, hz=0.05, opacity=0.0
    )
    model = builder.finalize(device=device)
    state = model.state()
    sensor = SensorTiledCamera(model)
    rays = sensor.utils.compute_camera_rays_pinhole(16, 16, camera_fovs=math.radians(40.0))
    camera = wp.array([[wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())]], dtype=wp.transformf, device=device)
    depth = sensor.utils.create_depth_image_output(16, 16)
    sensor.update(state, camera, rays, depth_image=depth)
    # The center ray hits the sphere top at z = 0.5, i.e. 2.5 m from the camera, not the box at 1.4 m.
    np.testing.assert_allclose(depth.numpy()[0, 0, 8, 8], 2.5, atol=0.05)


class TestSensorTiledCameraOpacity(unittest.TestCase):
    pass


add_function_test(
    TestSensorTiledCameraOpacity,
    "test_transparent_shapes_are_not_rendered",
    test_transparent_shapes_are_not_rendered,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
