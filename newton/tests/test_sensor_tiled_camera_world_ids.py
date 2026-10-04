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


def test_render_selected_worlds_matches_full_render(test: unittest.TestCase, device):
    """Rendering a subset of worlds reproduces those worlds of a full render at a fraction of the cost."""
    blueprint = newton.ModelBuilder()
    body = blueprint.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    blueprint.add_shape_box(body, hx=0.2, hy=0.3, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(blueprint, 4)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    state = model.state()
    # Give each world a different height so the worlds render differently.
    body_q = state.body_q.numpy()
    body_q[:, 2] = [0.3, 0.5, 0.7, 0.9]
    state.body_q.assign(body_q)
    model.bvh_refit_shapes(state)

    sensor = SensorTiledCamera(model)
    rays = sensor.utils.compute_camera_rays_pinhole(32, 24, camera_fovs=math.radians(60.0))
    eye = wp.transform(wp.vec3(2.0, -2.0, 1.5), wp.quat_rpy(0.9, 0.0, 0.8))
    full_transforms = wp.array([[eye] * model.world_count], dtype=wp.transformf, device=device)
    full_depth = sensor.utils.create_depth_image_output(32, 24)
    sensor.update(state, full_transforms, rays, depth_image=full_depth)

    world_ids = wp.array([2, 0], dtype=wp.int32, device=device)
    subset_transforms = wp.array([[eye, eye]], dtype=wp.transformf, device=device)
    subset_depth = sensor.utils.create_depth_image_output(32, 24, world_count=2)
    sensor.update(state, subset_transforms, rays, depth_image=subset_depth, world_ids=world_ids)

    full = full_depth.numpy()
    subset = subset_depth.numpy()
    test.assertEqual(subset.shape, (2, 1, 24, 32))
    np.testing.assert_allclose(subset[0], full[2], atol=1e-5)
    np.testing.assert_allclose(subset[1], full[0], atol=1e-5)
    test.assertFalse(np.allclose(full[2], full[0]))


class TestSensorTiledCameraWorldIds(unittest.TestCase):
    pass


add_function_test(
    TestSensorTiledCameraWorldIds,
    "test_render_selected_worlds_matches_full_render",
    test_render_selected_worlds_matches_full_render,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
