# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_cloth_renders_triangle_colors_from_both_sides(test: unittest.TestCase, device):
    """A colored cloth sheet shows Model.tri_color from above and from below."""
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.25,
        cell_y=0.25,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    model.tri_color.assign(np.tile([[1.0, 0.0, 0.0]], (model.tri_count, 1)).astype(np.float32))
    state = model.state()
    model.bvh_refit_particles(state)
    sensor = SensorTiledCamera(model)
    sensor.default_render_config.enable_shadows = False
    sensor.utils.set_ambient_light(wp.vec3f(1.0))
    rays = sensor.utils.compute_camera_rays_pinhole(8, 8, camera_fovs=math.radians(20.0))
    above = wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())
    below = wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi))
    for eye in (above, below):
        hdr = sensor.utils.create_hdr_color_image_output(8, 8)
        sensor.update(state, wp.array([[eye]], dtype=wp.transformf, device=device), rays, hdr_color_image=hdr)
        center = hdr.numpy()[0, 0, 4, 4]
        test.assertGreater(center[0], 0.5)
        test.assertLess(center[1], 0.05)


class TestSensorTiledCameraCloth(unittest.TestCase):
    pass


add_function_test(
    TestSensorTiledCameraCloth,
    "test_cloth_renders_triangle_colors_from_both_sides",
    test_cloth_renders_triangle_colors_from_both_sides,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
