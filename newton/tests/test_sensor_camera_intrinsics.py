# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrated camera geometry: SensorCamera.Intrinsics and body-mounted camera transforms."""

import json
import math
import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices

Intrinsics = SensorCamera.Intrinsics

# A RealSense color calibration in OpenCV order (k1, k2, p1, p2, k3), at its 640x480 stream size.
_REALSENSE_K = [434.879, 0.0, 322.866, 0.0, 434.325, 239.466, 0.0, 0.0, 1.0]
_REALSENSE_D = [-0.05412, 0.06110, -0.000978, -0.0000880, -0.02060]
# Every OpenCV pinhole coefficient, mild enough to stay one-to-one over the image.
_OPENCV_D = [-0.12, 0.03, 0.002, -0.003, 0.004, 0.01, -0.002, 0.001, 0.001, -0.0005, 0.0008, 0.0002]
# A camera 1.2 m above the origin, looking down and turned about the vertical (Newton frame: -Z forward, +Y up).
_POSE = [
    0.05,
    -0.1,
    1.2,
    *wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, -0.1, 1.0)), 0.7)
    * wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.1),
]


def _rotation(quaternion):
    """Rotation matrix of an xyzw quaternion, computed without Warp."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _opencv_extrinsics(pose):
    """OpenCV world-to-camera rotation and translation of a camera pose in the Newton camera frame."""
    pose = np.asarray(pose, dtype=np.float64)
    # The OpenCV camera frame keeps x and flips y (down) and z (forward).
    rotation = np.diag([1.0, -1.0, -1.0]) @ _rotation(pose[3:]).T
    return rotation, -rotation @ pose[:3]


def _cv_project_points(points, pose, camera_matrix, coefficients):
    """Transcription of the pinhole model documented for ``cv2.projectPoints`` (12 coefficients, OpenCV order)."""
    rotation, translation = _opencv_extrinsics(pose)
    camera = np.asarray(points, dtype=np.float64) @ rotation.T + translation
    x, y = camera[:, 0] / camera[:, 2], camera[:, 1] / camera[:, 2]
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = coefficients
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2**2
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r2**2
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=-1)


def _inverse_brown_conrady_points(pixels, pose, camera_matrix, coefficients, depth):
    """World points at OpenCV depth ``depth`` seen at ``pixels``: the Brown-Conrady polynomial undistorts pixels."""
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    k1, k2, p1, p2, k3 = coefficients
    x = (pixels[:, 0] - K[0, 2]) / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    ux = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    uy = y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    camera = np.stack([ux, uy, np.ones_like(ux)], axis=-1) * np.asarray(depth)[:, None]
    rotation, translation = _opencv_extrinsics(pose)
    return (camera - translation) @ rotation


def _scene_points(count=200, seed=3):
    """Points on and above a table at z = 0.75 m below the test camera."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(-0.3, 0.3, count), rng.uniform(-0.25, 0.25, count), rng.uniform(0.0, 0.6, count)]
    )


def _pixel_grid(width, height):
    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([xs, ys], axis=-1).astype(np.float64)


class TestSensorCameraIntrinsics(unittest.TestCase):
    def test_project_matches_opencv_reference(self):
        """Projection with every OpenCV coefficient matches the documented cv2.projectPoints model."""
        camera = Intrinsics.from_camera_matrix(_REALSENSE_K, _OPENCV_D, width=640, height=480)
        points = _scene_points()
        pixels, depth = camera.project(points, _POSE)
        expected = _cv_project_points(points, _POSE, _REALSENSE_K, _OPENCV_D)
        np.testing.assert_allclose(pixels, expected, atol=1.0e-9)
        rotation, translation = _opencv_extrinsics(_POSE)
        np.testing.assert_allclose(depth, (points @ rotation.T + translation)[:, 2], atol=1.0e-12)
        # The camera pose may also be a (position, quaternion) pair or a Warp transform.
        for pose in ((_POSE[:3], _POSE[3:]), wp.transform(*_POSE)):
            np.testing.assert_allclose(camera.project(points, pose)[0], expected, atol=1.0e-3)

    def test_inverse_brown_conrady_matches_reference(self):
        """RealSense unprojection applies the polynomial to pixels; projection inverts it."""
        camera = Intrinsics.from_camera_matrix(
            _REALSENSE_K, _REALSENSE_D, width=640, height=480, distortion_model="inverse_brown_conrady"
        )
        self.assertEqual(camera.distortion_model, Intrinsics.DistortionModel.INVERSE_BROWN_CONRADY)
        pixels = np.array([[0.0, 0.0], [639.0, 0.0], [0.0, 479.0], [639.0, 479.0], [322.866, 239.466], [100.3, 351.7]])
        depth = np.array([0.8, 1.0, 1.2, 0.9, 1.1, 1.3])
        expected = _inverse_brown_conrady_points(pixels, _POSE, _REALSENSE_K, _REALSENSE_D, depth)
        np.testing.assert_allclose(camera.unproject_to_depth(pixels, _POSE, forward_depth=depth), expected, atol=1e-12)
        directions = camera.unproject(pixels, _POSE)
        offsets = expected - np.asarray(_POSE[:3])
        np.testing.assert_allclose(directions, offsets / np.linalg.norm(offsets, axis=-1, keepdims=True), atol=1e-12)
        projected, forward_depth = camera.project(expected, _POSE)
        np.testing.assert_allclose(projected, pixels, atol=1.0e-7)
        np.testing.assert_allclose(forward_depth, depth, atol=1.0e-12)

    def test_round_trips(self):
        """Project, unproject to depth, rays, and planes invert each other for both distortion models."""
        for model, coefficients in (("opencv", _OPENCV_D), ("inverse_brown_conrady", _REALSENSE_D)):
            with self.subTest(model=model):
                camera = Intrinsics.from_camera_matrix(
                    _REALSENSE_K, coefficients, width=640, height=480, distortion_model=model
                )
                points = _scene_points()
                pixels, depth = camera.project(points, _POSE)
                self.assertTrue(np.isfinite(pixels).all())
                np.testing.assert_allclose(
                    camera.unproject_to_depth(pixels, _POSE, forward_depth=depth), points, atol=1e-9
                )
                offsets = points - np.asarray(_POSE[:3])
                np.testing.assert_allclose(
                    camera.unproject(pixels, _POSE),
                    offsets / np.linalg.norm(offsets, axis=-1, keepdims=True),
                    atol=1e-9,
                )
                # Every pixel center of the image unprojects onto the table plane and projects back to itself.
                grid = _pixel_grid(640, 480)[::7, ::9]
                table = camera.unproject_to_plane(grid, _POSE, plane=(0.0, 0.0, 1.0, -0.75))
                self.assertEqual(table.shape, (*grid.shape[:2], 3))
                np.testing.assert_allclose(table[..., 2], 0.75, atol=1e-12)
                np.testing.assert_allclose(camera.project(table, _POSE)[0], grid, atol=1e-7)
                # Without a camera transform, points and directions are in the camera frame.
                local = camera.unproject_to_depth(grid, forward_depth=2.0)
                np.testing.assert_allclose(local[..., 2], -2.0, atol=1e-12)
                np.testing.assert_allclose(camera.project(local)[0], grid, atol=1e-7)

    def test_single_points_and_invalid_projections(self):
        """Single points keep their shape; points behind the camera or past the distortion fold have no pixel."""
        camera = Intrinsics(640, 480, 400.0, 400.0, 319.5, 239.5)
        pixel, depth = camera.project([0.1, -0.05, -2.0])
        self.assertEqual(pixel.shape, (2,))
        np.testing.assert_allclose(pixel, [319.5 + 20.0, 239.5 + 10.0])
        self.assertAlmostEqual(float(depth), 2.0)
        np.testing.assert_allclose(camera.unproject([319.5, 239.5]), [0.0, 0.0, -1.0])
        # k1 = -0.3 makes r * (1 + k1 r^2) fold back at r = 1.054, so the point at r = 2 would land inside the image.
        folding = Intrinsics(640, 480, 300.0, 300.0, 319.5, 239.5, k1=-0.3)
        pixels, depth = folding.project([[0.1, 0.0, -1.0], [2.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.isfinite(pixels[0]).all())
        self.assertTrue(np.isnan(pixels[1:]).all())
        np.testing.assert_allclose(depth, [1.0, 1.0, -1.0])
        # Rays parallel to or pointing away from a plane do not meet it.
        hits = camera.unproject_to_plane([[319.5, 239.5], [319.5, 479.0]], plane=(0.0, 1.0, 0.0, -1.0))
        self.assertTrue(np.isnan(hits).all())
        hit = camera.unproject_to_plane([319.5, 239.5], plane=(0.0, 0.0, 1.0, 3.0))
        np.testing.assert_allclose(hit, [0.0, 0.0, -3.0])

    def test_resize_keeps_pixel_edges(self):
        """Resampled intrinsics move pixel centers with the image edges, as OpenCV pixel coordinates do."""
        camera = Intrinsics.from_camera_matrix(
            _REALSENSE_K, _REALSENSE_D, width=640, height=480, distortion_model="inverse_brown_conrady"
        )
        half = camera.resize(320, 240)
        self.assertEqual((half.width, half.height), (320, 240))
        self.assertAlmostEqual(half.fx, camera.fx / 2)
        points = _scene_points(20)
        full, _ = camera.project(points, _POSE)
        np.testing.assert_allclose(half.project(points, _POSE)[0], (full + 0.5) / 2 - 0.5, atol=1e-9)
        np.testing.assert_allclose(half.resize(640, 480).project(points, _POSE)[0], full, atol=1e-9)

    def test_from_fov_and_camera_matrix(self):
        """Constructors parse OpenCV coefficient orders, model names, and reject malformed calibrations."""
        fov = Intrinsics.from_fov(64, 48, math.radians(60.0))
        self.assertAlmostEqual(fov.fy, 24.0 / math.tan(math.radians(30.0)))
        self.assertEqual((fov.cx, fov.cy), (31.5, 23.5))
        K = np.array(_REALSENSE_K).reshape(3, 3)
        for count in (4, 5, 8, 12):
            camera = Intrinsics.from_camera_matrix(K, _OPENCV_D[:count], width=640, height=480)
            ordered = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")
            self.assertEqual([getattr(camera, name) for name in ordered], [*_OPENCV_D[:count], *[0.0] * (12 - count)])
        camera = Intrinsics.from_camera_matrix(_REALSENSE_K, width=640.0, height=480)
        self.assertEqual((camera.width, camera.fx, camera.cy), (640, 434.879, 239.466))
        self.assertEqual(camera, Intrinsics(640, 480, 434.879, 434.325, 322.866, 239.466))
        self.assertEqual(
            Intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0, distortion_model="INVERSE_BROWN_CONRADY").distortion_model,
            Intrinsics.DistortionModel.INVERSE_BROWN_CONRADY,
        )
        for arguments, message in (
            ({"camera_matrix": [1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}, "zero skew"),
            ({"camera_matrix": _REALSENSE_K[:6]}, "9 values"),
            ({"camera_matrix": _REALSENSE_K, "distortion": [0.1] * 6}, "4, 5, 8, or 12"),
            (
                {"camera_matrix": _REALSENSE_K, "distortion": _OPENCV_D, "distortion_model": "inverse_brown_conrady"},
                "k1, k2, k3, p1, p2",
            ),
            ({"camera_matrix": _REALSENSE_K, "distortion_model": "fisheye"}, "distortion_model"),
            ({"camera_matrix": [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}, "positive"),
            ({"camera_matrix": _REALSENSE_K, "width": 640.5}, "positive integer"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Intrinsics.from_camera_matrix(**{"width": 640, "height": 480, **arguments})
        with self.assertRaisesRegex(ValueError, "camera_fov"):
            Intrinsics.from_fov(64, 48, math.pi)
        with self.assertRaisesRegex(ValueError, "camera_transform"):
            camera.project([0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "plane"):
            camera.unproject_to_plane([1.0, 2.0], plane=(0.0, 0.0, 0.0, 1.0))

    def test_from_dict_reads_calibration_formats(self):
        """OpenCV camera.json, ROS, and RealSense calibrations give the intrinsics of from_camera_matrix."""
        expected = Intrinsics.from_camera_matrix(
            _REALSENSE_K, _REALSENSE_D, width=640, height=480, distortion_model="inverse_brown_conrady"
        )
        K3 = np.array(_REALSENSE_K).reshape(3, 3).tolist()
        fx, cx, fy, cy = _REALSENSE_K[0], _REALSENSE_K[2], _REALSENSE_K[4], _REALSENSE_K[5]
        top = {
            "width": 640,
            "height": 480,
            "K": _REALSENSE_K,
            "D": _REALSENSE_D,
            "distortion_model": "inverse_brown_conrady",
            "position": [0.0, 0.0, 1.5],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        wrist = {**top, "K": K3, "body": "wrist", "body_rotation_xyzw": [1.0, 0.0, 0.0, 0.0]}
        cameras = {"top": top, "wrist": wrist, "convention": "camera looks along -Z", "time_base": "state clock"}
        formats = {
            "camera.json camera": (cameras, "top"),
            "nested camera matrix": (cameras, "wrist"),
            "single camera": ({"top": top, "convention": "-Z"}, None),
            "flat": (top, None),
            "RealSense rs2_intrinsics": (
                {
                    "width": 640,
                    "height": 480,
                    "ppx": cx,
                    "ppy": cy,
                    "fx": fx,
                    "fy": fy,
                    "model": "distortion.inverse_brown_conrady",
                    "coeffs": _REALSENSE_D,
                },
                None,
            ),
            "ROS calibration YAML": (
                {
                    "image_width": 640,
                    "image_height": 480,
                    "camera_name": "top",
                    "camera_matrix": {"rows": 3, "cols": 3, "data": _REALSENSE_K},
                    "distortion_model": "Inverse Brown Conrady",
                    "distortion_coefficients": {"rows": 1, "cols": 5, "data": _REALSENSE_D},
                },
                None,
            ),
            "separate values": (
                {
                    "width": 640,
                    "height": 480,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    **dict(zip(("k1", "k2", "p1", "p2", "k3"), _REALSENSE_D, strict=True)),
                    "K": _REALSENSE_K,
                    "distortion_model": Intrinsics.DistortionModel.INVERSE_BROWN_CONRADY,
                },
                None,
            ),
        }
        for name, (calibration, camera) in formats.items():
            with self.subTest(format=name):
                self.assertEqual(Intrinsics.from_dict(calibration, camera), expected)

        # ROS CameraInfo messages (lowercase k and d) and OpenCV model names.
        ros = {"width": 64, "height": 48, "k": [50.0, 0.0, 31.5, 0.0, 50.0, 23.5, 0.0, 0.0, 1.0]}
        for model in ("plumb_bob", "rational_polynomial", "OPENCV", "brown_conrady", "radtan"):
            camera = Intrinsics.from_dict({**ros, "d": _OPENCV_D[:8], "distortion_model": model})
            self.assertEqual(camera, Intrinsics.from_camera_matrix(ros["k"], _OPENCV_D[:8], width=64, height=48))
        self.assertEqual(
            Intrinsics.from_dict({**ros, "d": [0.0] * 5, "distortion_model": "none"}),
            Intrinsics(64, 48, 50.0, 50.0, 31.5, 23.5),
        )
        # The image size from keywords, when the calibration has none.
        bare = {"fx": 50.0, "fy": 50.0, "cx": 31.5, "cy": 23.5}
        self.assertEqual(Intrinsics.from_dict(bare, width=64, height=48), Intrinsics(64, 48, 50.0, 50.0, 31.5, 23.5))
        self.assertEqual(Intrinsics.from_dict(ros, width=64, height=48).width, 64)

        for calibration, arguments, message in (
            (cameras, {}, r"holds cameras \['top', 'wrist'\]"),
            (cameras, {"camera": "left"}, "no camera 'left'"),
            ({"position": [0, 0, 1]}, {}, "no camera matrix"),
            (bare, {}, "no image width"),
            ({**bare, "width": 64}, {"height": 48, "width": 32}, "resize"),
            ({"fx": 50.0, "width": 64, "height": 48}, {}, "lacks fy, cx, cy"),
            ({**ros, "fx": 51.0}, {}, "disagrees"),
            ({**ros, "K": _REALSENSE_K}, {}, "'K' and 'k' disagree"),
            ({**ros, "D": [0.1, 0, 0, 0], "k1": 0.2}, {}, "k1 0.2 disagrees"),
            ({**ros, "k": _REALSENSE_K[:6]}, {}, "9 values"),
            ({**ros, "D": [0.1] * 6}, {}, "4, 5, 8, or 12"),
            ({**ros, "D": "none"}, {}, "must hold numbers"),
            ({**ros, "distortion_model": "equidistant"}, {}, "fisheye"),
            ({**ros, "distortion_model": "modified_brown_conrady"}, {}, "not a pinhole distortion model"),
            ({**ros, "distortion_model": "none", "D": [0.1, 0, 0, 0]}, {}, "nonzero"),
            ({**ros, "distortion_model": 1.5}, {}, "model name"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Intrinsics.from_dict(calibration, **arguments)
        with self.assertRaisesRegex(TypeError, "mapping"):
            Intrinsics.from_dict([1.0, 2.0])

    def test_from_json_and_pixels_to_plane(self):
        """A camera.json file maps pixels to points on a table plane and back."""
        calibration = {
            "overhead": {
                "width": 640,
                "height": 480,
                "K": _REALSENSE_K,
                "D": _REALSENSE_D,
                "distortion_model": "inverse_brown_conrady",
                "position": _POSE[:3],
                "rotation_xyzw": _POSE[3:],
            },
            "convention": "the camera looks along -Z with +Y up",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "camera.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(calibration, file)
            camera = Intrinsics.from_json(path)
            self.assertEqual(camera, Intrinsics.from_dict(calibration["overhead"]))
            self.assertEqual(Intrinsics.from_json(path, camera="overhead"), camera)
            with self.assertRaisesRegex(ValueError, "camera.json: calibration has no camera 'top'"):
                Intrinsics.from_json(path, camera="top")
        entry = calibration["overhead"]
        pose = wp.transform(wp.vec3(*entry["position"]), wp.quat(*entry["rotation_xyzw"]))
        # The centroid of a mask region: columns are image x and rows image y.
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:260, 300:380] = True
        rows, columns = np.nonzero(mask)
        pixels = np.array([[columns.mean(), rows.mean()], [10.0, 20.0], [630.0, 470.0]])
        on_table = camera.unproject_to_plane(pixels, pose, plane=(0.0, 0.0, 1.0, -0.75))
        np.testing.assert_allclose(on_table[:, 2], 0.75, atol=1.0e-12)
        back, forward_depth = camera.project(on_table, pose)
        np.testing.assert_allclose(back, pixels, atol=1.0e-6)
        self.assertTrue((forward_depth > 0.0).all())


def test_rays_match_pinhole_helper(test, device):
    """from_fov rays equal compute_camera_rays_pinhole, and projection sends each ray to its pixel center."""
    width, height, fov = 40, 30, math.radians(55.0)
    camera = Intrinsics.from_fov(width, height, fov)
    rays = camera.compute_camera_rays(device=device).numpy()
    expected = SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=fov, device=device).numpy()
    np.testing.assert_allclose(rays, expected, atol=1.0e-6)
    pixels, _ = camera.project(rays[:, :, 1].astype(np.float64))
    np.testing.assert_allclose(pixels, _pixel_grid(width, height), atol=1.0e-4)


def test_rays_match_opencv_helper(test, device):
    """OpenCV rays come from compute_camera_rays_pinhole_opencv and equal unproject at pixel centers."""
    camera = Intrinsics.from_camera_matrix(_REALSENSE_K, _OPENCV_D, width=640, height=480)
    for width, height in ((640, 480), (160, 120)):
        with test.subTest(size=(width, height)):
            rays = camera.compute_camera_rays(width, height, device=device).numpy()
            np.testing.assert_allclose(rays[:, :, 0], 0.0)
            expected = camera.resize(width, height).unproject(_pixel_grid(width, height))
            # The Warp solver converges to about 1e-6 in normalized coordinates.
            np.testing.assert_allclose(rays[:, :, 1], expected, atol=5.0e-6)
    # The helper samples pixel i at calibration coordinate i + 0.5, so the same rays need the principal point moved.
    helper = SensorCamera.compute_camera_rays_pinhole_opencv(
        640,
        480,
        camera.fx,
        camera.fy,
        camera.cx + 0.5,
        camera.cy + 0.5,
        **{
            name: getattr(camera, name)
            for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4")
        },
        device=device,
    )
    np.testing.assert_array_equal(camera.compute_camera_rays(device=device).numpy(), helper.numpy())
    # Pixels whose inverse fails, or whose only preimage lies past the fold radius (folded back from the far side
    # of the optical axis), are zero rays in the bundle and NaN directions from unproject.
    folding = Intrinsics(64, 48, 20.0, 20.0, 31.5, 23.5, k1=-0.3)
    rays = folding.compute_camera_rays(device=device).numpy()[:, :, 1]
    directions = folding.unproject(_pixel_grid(64, 48))
    invalid = np.isnan(directions).any(axis=-1)
    test.assertTrue(invalid.any() and not invalid.all())
    np.testing.assert_array_equal(np.all(rays == 0.0, axis=-1), invalid)
    np.testing.assert_allclose(rays[~invalid], directions[~invalid], atol=1.0e-5)


def test_rays_match_inverse_brown_conrady(test, device):
    """RealSense rays apply the polynomial at pixel centers, at the calibration size and resampled."""
    camera = Intrinsics.from_camera_matrix(
        _REALSENSE_K, _REALSENSE_D, width=640, height=480, distortion_model="inverse_brown_conrady"
    )
    for width, height in ((640, 480), (96, 72)):
        with test.subTest(size=(width, height)):
            out = wp.zeros((height, width, 2), dtype=wp.vec3f, device=device)
            rays = camera.compute_camera_rays(width, height, out_rays=out, device=device)
            test.assertIs(rays, out)
            expected = camera.resize(width, height).unproject(_pixel_grid(width, height))
            np.testing.assert_allclose(rays.numpy()[:, :, 1], expected, atol=2.0e-6)


def test_render_projects_to_pixel_centers(test, device):
    """Spheres rendered through a distorted body-mounted camera sit where project() puts their centers."""
    camera = Intrinsics.from_camera_matrix(
        _REALSENSE_K, _REALSENSE_D, width=640, height=480, distortion_model="inverse_brown_conrady"
    ).resize(160, 120)
    builder = newton.ModelBuilder()
    body_pose = wp.transform(
        wp.vec3(0.2, -0.1, 0.9), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.4, 0.2)), 2.6)
    )
    body = builder.add_body(xform=body_pose)
    # The camera's optical frame (+Z forward, +Y down) in the body frame; flipped to the Newton camera frame.
    optical = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.2)
    offset = wp.transform(wp.vec3(0.02, -0.01, 0.03), optical * wp.quat(1.0, 0.0, 0.0, 0.0))
    pose = np.asarray(wp.transform_multiply(body_pose, offset), dtype=np.float64)
    # Spheres 0.5 m in front of the camera near the image corners, where the distortion moves them most.
    targets = np.array([[12.0, 10.0], [148.0, 12.0], [10.0, 108.0], [150.0, 110.0], [80.0, 60.0]])
    centers = camera.unproject_to_depth(targets, pose, forward_depth=0.5)
    shapes = [
        builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(*c), wp.quat_identity()), radius=0.012) for c in centers
    ]
    model = builder.finalize(device=device)
    state = model.state()
    transforms = SensorCamera.compute_camera_transforms_body(state.body_q, [body], [offset])
    np.testing.assert_allclose(transforms.numpy()[0], pose, atol=1.0e-6)
    sensor = SensorCamera(model)
    rays = camera.compute_camera_rays(device=device)
    image = sensor.create_shape_index_image_output(1, camera.width, camera.height)
    model.bvh_refit_shapes(state)
    sensor.update(state, transforms, rays, shape_index_image=image)
    shape_index = image.numpy()[0]
    projected, _ = camera.project(centers, pose)
    np.testing.assert_allclose(projected, targets, atol=1.0e-6)
    # The lens moves the corner spheres by more than a pixel, so an undistorted render would miss the targets.
    pinhole = Intrinsics(camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy)
    test.assertTrue(np.all(np.linalg.norm(pinhole.project(centers[:4], pose)[0] - targets[:4], axis=-1) > 1.0))
    for shape, target in zip(shapes, targets, strict=True):
        ys, xs = np.nonzero(shape_index == shape)
        test.assertGreater(len(xs), 8)
        np.testing.assert_allclose([xs.mean(), ys.mean()], target, atol=0.25)


def test_camera_transforms_body(test, device):
    """Mounted cameras compose body poses with their offsets; -1 keeps a world pose; outputs can be reused."""
    rng = np.random.default_rng(5)
    poses = []
    for _ in range(3):
        q = rng.normal(size=4)
        poses.append([*rng.uniform(-1.0, 1.0, 3), *(q / np.linalg.norm(q))])
    body_q = wp.array(np.asarray(poses, dtype=np.float32), dtype=wp.transformf, device=device)
    offsets = [
        wp.transform(wp.vec3(0.1, 0.0, 0.2), wp.quat(1.0, 0.0, 0.0, 0.0)),
        ([0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 1.0]),
    ]
    offsets.append([0.5, 0.5, 2.0, *wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3)])
    transforms = SensorCamera.compute_camera_transforms_body(body_q, [2, 0, -1], offsets).numpy()
    for view, (body, xform) in enumerate(zip((2, 0, -1), offsets, strict=True)):
        if isinstance(xform, tuple):
            offset = np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in xform])
        else:
            offset = np.asarray(xform, dtype=np.float64)
        if body < 0:
            position, rotation = offset[:3], _rotation(offset[3:])
        else:
            parent = np.asarray(poses[body])
            position = parent[:3] + _rotation(parent[3:]) @ offset[:3]
            rotation = _rotation(parent[3:]) @ _rotation(offset[3:])
        np.testing.assert_allclose(transforms[view, :3], position, atol=1.0e-5)
        np.testing.assert_allclose(_rotation(transforms[view, 3:]), rotation, atol=1.0e-5)
    # Device arrays and a preallocated output: the identity offset gives the body poses.
    bodies = wp.array([1, 2], dtype=wp.int32, device=device)
    out = wp.zeros(2, dtype=wp.transformf, device=device)
    result = SensorCamera.compute_camera_transforms_body(body_q, bodies, out_transforms=out)
    test.assertIs(result, out)
    np.testing.assert_allclose(out.numpy(), np.asarray(poses, dtype=np.float32)[[1, 2]], atol=1.0e-6)
    with test.assertRaisesRegex(ValueError, "body indices"):
        SensorCamera.compute_camera_transforms_body(body_q, [3])
    with test.assertRaisesRegex(ValueError, "entries"):
        SensorCamera.compute_camera_transforms_body(body_q, [0, 1], offsets)
    with test.assertRaisesRegex(TypeError, "body_q"):
        SensorCamera.compute_camera_transforms_body(np.zeros((2, 7)), [0])


for _name, _function in (
    ("test_rays_match_pinhole_helper", test_rays_match_pinhole_helper),
    ("test_rays_match_opencv_helper", test_rays_match_opencv_helper),
    ("test_rays_match_inverse_brown_conrady", test_rays_match_inverse_brown_conrady),
    ("test_render_projects_to_pixel_centers", test_render_projects_to_pixel_centers),
    ("test_camera_transforms_body", test_camera_transforms_body),
):
    add_function_test(TestSensorCameraIntrinsics, _name, _function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
