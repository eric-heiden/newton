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


def _center_left_albedo(device, texture_transform) -> np.ndarray:
    """Albedo where u = 0.25 on a quad textured red for u < 0.5 and green for u >= 0.5."""
    texture = np.zeros((8, 16, 4), dtype=np.uint8)
    texture[..., 3] = 255
    texture[:, :8, 0] = 255
    texture[:, 8:, 1] = 255
    mesh = newton.Mesh(
        np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
        np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        compute_inertia=False,
        texture=texture,
        texture_transform=texture_transform,
    )
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=mesh, color=(1.0, 1.0, 1.0))
    model = builder.finalize(device=device)
    sensor = SensorTiledCamera(model, load_textures=True)
    sensor.default_render_config.enable_textures = True
    rays = sensor.utils.compute_camera_rays_pinhole(16, 16, camera_fovs=math.radians(60.0))
    eye = wp.array([[wp.transform(wp.vec3(0.0, 0.0, 1.8), wp.quat_identity())]], dtype=wp.transformf, device=device)
    albedo = sensor.utils.create_albedo_image_output(16, 16)
    sensor.update(model.state(), eye, rays, albedo_image=albedo)
    packed = albedo.numpy()[0, 0, 8, 4]
    return np.array([packed & 255, (packed >> 8) & 255, (packed >> 16) & 255])


def test_texture_transform_offsets_uvs(test: unittest.TestCase, device):
    identity = _center_left_albedo(device, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    shifted = _center_left_albedo(device, ((1.0, 0.0, 0.5), (0.0, 1.0, 0.0)))
    test.assertGreater(identity[0], 200)
    test.assertLess(identity[1], 50)
    test.assertGreater(shifted[1], 200)
    test.assertLess(shifted[0], 50)


class TestSensorTiledCameraTextureTransform(unittest.TestCase):
    pass


add_function_test(
    TestSensorTiledCameraTextureTransform,
    "test_texture_transform_offsets_uvs",
    test_texture_transform_offsets_uvs,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
