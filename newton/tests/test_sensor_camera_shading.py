# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _render(camera: SensorCamera, model: newton.Model, eye: wp.transformf, size: int, fov_deg: float, **outputs):
    """Render one view of world 0 into freshly allocated ``outputs`` (channel name -> dtype); returns numpy images."""
    device = model.device
    rays = SensorCamera.compute_camera_rays_pinhole(size, size, camera_fov=math.radians(fov_deg), device=device)
    transforms = wp.array([eye], dtype=wp.transformf, device=device)
    images = {name: wp.zeros((1, size, size), dtype=dtype, device=device) for name, dtype in outputs.items()}
    state = model.state()
    model.bvh_refit_shapes(state)
    model.bvh_refit_particles(state)
    camera.update(state, transforms, rays, **{f"{name}_image": image for name, image in images.items()})
    return {name: image.numpy()[0] for name, image in images.items()}


def _center_radiance(up_axis: newton.Axis, configure, device) -> np.ndarray:
    """Linear color at the image center of a white box's upward-facing side, seen from above."""
    builder = newton.ModelBuilder(up_axis=up_axis)
    builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.5, color=(1.0, 1.0, 1.0))
    model = builder.finalize(device=device)
    camera = SensorCamera(model)
    configure(camera)
    # The camera looks along its -Z axis; turn it to look down the model's up axis.
    if up_axis == newton.Axis.Z:
        eye = wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())
    else:
        eye = wp.transform(wp.vec3(0.0, 3.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi))
    return _render(camera, model, eye, 9, 20.0, hdr_color=wp.vec3f)["hdr_color"][4, 4]


def test_light_color_scales_direct_light(test: unittest.TestCase, device):
    def configure(camera):
        camera.set_ambient_light(wp.vec3f(0.0))
        camera.create_default_light(
            enable_shadows=False, direction=wp.vec3f(0.0, 0.0, -1.0), color=wp.vec3f(0.5, 0.25, 0.0)
        )

    np.testing.assert_allclose(_center_radiance(newton.Axis.Z, configure, device), [0.5, 0.25, 0.0], atol=1e-4)


def test_ambient_light_follows_up_axis(test: unittest.TestCase, device):
    def configure(camera):
        camera.set_ambient_light(wp.vec3f(0.3, 0.3, 0.3), wp.vec3f(0.1, 0.1, 0.1))

    # Upward-facing surfaces receive the sky color for Z-up and Y-up models alike.
    for up_axis in (newton.Axis.Z, newton.Axis.Y):
        np.testing.assert_allclose(_center_radiance(up_axis, configure, device), [0.3, 0.3, 0.3], atol=1e-4)


def test_default_ambient_light_is_unchanged(test: unittest.TestCase, device):
    # The default sky radiance on an upward-facing Z-up surface, without lights.
    radiance = _center_radiance(newton.Axis.Z, lambda camera: None, device)
    np.testing.assert_allclose(radiance, [0.2, 0.2, 0.225], atol=1e-4)


def test_default_light_shines_down_for_y_up(test: unittest.TestCase, device):
    def configure(camera):
        camera.set_ambient_light(wp.vec3f(0.0))
        camera.create_default_light(enable_shadows=False)

    # The default light hits the upward-facing side at 1/sqrt(3) for either up axis.
    for up_axis in (newton.Axis.Z, newton.Axis.Y):
        np.testing.assert_allclose(_center_radiance(up_axis, configure, device), [0.57735] * 3, atol=1e-4)


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
        eye = wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity())
        depths.append(_render(SensorCamera(model), model, eye, 16, 30.0, depth=wp.float32)["depth"])
    # The camera looks straight down at the top face (z = 0.1), 1.9 m away.
    np.testing.assert_allclose(depths[0][8, 8], 1.9, atol=1e-3)
    np.testing.assert_allclose(depths[0], depths[1], atol=1e-3)


def test_transparent_shapes_are_not_rendered(test: unittest.TestCase, device):
    """A fully transparent shape in front of the camera does not occlude what is behind it."""
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.5)
    # An invisible helper (MJCF rgba alpha 0) between the camera and the sphere.
    builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 1.5), wp.quat_identity()), hx=1.0, hy=1.0, hz=0.05, opacity=0.0
    )
    model = builder.finalize(device=device)
    eye = wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())
    depth = _render(SensorCamera(model), model, eye, 16, 40.0, depth=wp.float32)["depth"]
    # The center ray hits the sphere top at z = 0.5, i.e. 2.5 m from the camera, not the box at 1.4 m.
    np.testing.assert_allclose(depth[8, 8], 2.5, atol=0.05)


class TestSensorCameraShading(unittest.TestCase):
    pass


for _name, _function in (
    ("test_light_color_scales_direct_light", test_light_color_scales_direct_light),
    ("test_ambient_light_follows_up_axis", test_ambient_light_follows_up_axis),
    ("test_default_ambient_light_is_unchanged", test_default_ambient_light_is_unchanged),
    ("test_default_light_shines_down_for_y_up", test_default_light_shines_down_for_y_up),
    ("test_convex_hull_is_rendered", test_convex_hull_is_rendered),
    ("test_transparent_shapes_are_not_rendered", test_transparent_shapes_are_not_rendered),
):
    add_function_test(TestSensorCameraShading, _name, _function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
