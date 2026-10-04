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


def _center_radiance(up_axis: newton.Axis, configure, device) -> np.ndarray:
    """Linear color at the image center of a white box's upward-facing side, seen from above."""
    builder = newton.ModelBuilder(up_axis=up_axis)
    builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.5, color=(1.0, 1.0, 1.0))
    model = builder.finalize(device=device)
    state = model.state()
    sensor = SensorTiledCamera(model)
    sensor.default_render_config.enable_shadows = False
    configure(sensor)
    rays = sensor.utils.compute_camera_rays_pinhole(9, 9, camera_fovs=math.radians(20.0))
    # The camera looks along its -Z axis; turn it to look down the model's up axis.
    if up_axis == newton.Axis.Z:
        eye = wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())
    else:
        eye = wp.transform(wp.vec3(0.0, 3.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi))
    transforms = wp.array([[eye]], dtype=wp.transformf, device=device)
    hdr = sensor.utils.create_hdr_color_image_output(9, 9)
    sensor.update(state, transforms, rays, hdr_color_image=hdr)
    return hdr.numpy()[0, 0, 4, 4]


def test_light_color_scales_direct_light(test: unittest.TestCase, device):
    def configure(sensor):
        sensor.utils.set_ambient_light(wp.vec3f(0.0))
        sensor.utils.create_default_light(
            enable_shadows=False, direction=wp.vec3f(0.0, 0.0, -1.0), color=wp.vec3f(0.5, 0.25, 0.0)
        )

    np.testing.assert_allclose(_center_radiance(newton.Axis.Z, configure, device), [0.5, 0.25, 0.0], atol=1e-4)


def test_ambient_light_follows_up_axis(test: unittest.TestCase, device):
    def configure(sensor):
        sensor.utils.set_ambient_light(wp.vec3f(0.3, 0.3, 0.3), wp.vec3f(0.1, 0.1, 0.1))

    # Upward-facing surfaces receive the sky color for Z-up and Y-up models alike.
    for up_axis in (newton.Axis.Z, newton.Axis.Y):
        np.testing.assert_allclose(_center_radiance(up_axis, configure, device), [0.3, 0.3, 0.3], atol=1e-4)


def test_default_light_shines_down_for_y_up(test: unittest.TestCase, device):
    def configure(sensor):
        sensor.utils.set_ambient_light(wp.vec3f(0.0))
        sensor.utils.create_default_light(enable_shadows=False)

    # The default light hits the upward-facing side at 1/sqrt(3) for either up axis.
    for up_axis in (newton.Axis.Z, newton.Axis.Y):
        np.testing.assert_allclose(_center_radiance(up_axis, configure, device), [0.57735] * 3, atol=1e-4)


class TestSensorTiledCameraLighting(unittest.TestCase):
    pass


for name, function in (
    ("test_light_color_scales_direct_light", test_light_color_scales_direct_light),
    ("test_ambient_light_follows_up_axis", test_ambient_light_follows_up_axis),
    ("test_default_light_shines_down_for_y_up", test_default_light_shines_down_for_y_up),
):
    add_function_test(TestSensorTiledCameraLighting, name, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
