# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise real CPU sensor observations and optional attached OpenGL capture."""

import base64
import importlib.util
import json
import math
import os
import struct
import tempfile
import threading
import unittest
import warnings
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.mcp.imaging import encode_png as _encode_png
from newton._src.mcp.observation import (
    ObservationRenderer,
    _intrinsics,
    _project,
)
from newton._src.mcp.protocol import TOOLS
from newton.mcp import SimulationSession
from newton.solvers import SolverXPBD

# Rotates a camera's local -Z (its viewing direction) onto world +Y, keeping +Z up.
_LOOK_ALONG_Y = [float(v) for v in wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)]
_MAGENTA = (255, 0, 255)
# A RealSense calibration at its full 848x480 stream size, as robot wrist cameras record.
_REALSENSE_848 = {
    "fx": 434.196,
    "fy": 433.593,
    "cx": 420.881,
    "cy": 235.403,
    "image_width": 848,
    "image_height": 480,
    "distortion_model": "inverse_brown_conrady",
    "k1": -0.05299,
    "k2": 0.05932,
    "k3": -0.01946,
    "p1": 0.00041,
    "p2": 0.00046,
}
# Orientation of a ROS optical frame (+Z forward, +Y down) in the observation camera frame (-Z forward, +Y up).
_OPTICAL = [1.0, 0.0, 0.0, 0.0]


def _decode_png(result):
    data = base64.b64decode(result["image_base64"])
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    width, height = struct.unpack(">II", data[16:24])
    offset, payload = 8, b""
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        if kind == b"IDAT":
            payload += data[offset + 8 : offset + 8 + length]
        offset += length + 12
    rows = np.frombuffer(zlib.decompress(payload), dtype=np.uint8).reshape(height, width * 3 + 1)
    if np.any(rows[:, 0]):
        raise ValueError("Unexpected PNG filter")
    return rows[:, 1:].reshape(height, width, 3)


def _assert_nearly_equal_images(test, a, b, max_pixels=2):
    """Allow a couple of silhouette pixels to flip under float32 rounding of equivalent camera poses."""
    test.assertEqual(a.shape, b.shape)
    test.assertLessEqual(int(np.any(a != b, axis=-1).sum()), max_pixels)


def _hit_centroid(image):
    """Mean image coordinates [x, y] (pixel centers at integers) of the non-black pixels of a shape_index image."""
    ys, xs = np.nonzero(np.asarray(image).any(axis=-1))
    return np.array([xs.mean(), ys.mean()])


def _has_color(image, color):
    return bool(np.all(np.asarray(image) == color, axis=-1).any())


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


def _mounted_pose(body_pose, offset):
    """World position [m] and rotation matrix of a camera at ``offset`` in the frame of a body at ``body_pose``."""
    body_pose, offset = np.asarray(body_pose, dtype=np.float64), np.asarray(offset, dtype=np.float64)
    rotation = _rotation(body_pose[3:])
    return body_pose[:3] + rotation @ offset[:3], rotation @ _rotation(offset[3:])


def _matrix_quaternion(rotation):
    """An xyzw quaternion of a proper rotation matrix (trace > -1)."""
    w = 0.5 * math.sqrt(max(1.0 + np.trace(rotation), 0.0))
    return [
        (rotation[2, 1] - rotation[1, 2]) / (4 * w),
        (rotation[0, 2] - rotation[2, 0]) / (4 * w),
        (rotation[1, 0] - rotation[0, 1]) / (4 * w),
        w,
    ]


# Undistorted normalized image coordinates (x right, y down) near the corners of an 848x480 RealSense image, where
# its distortion moves points most, and at the center.
_CORNERS = [(-0.85, -0.47), (0.85, -0.47), (-0.85, 0.47), (0.85, 0.47), (0.0, 0.0)]
# Looks down from 1 m above the origin, so the corner spheres at z = 0.6 m sit 0.4 m in front of the camera.
_CORNER_CAMERA = {"pose": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "width": 848, "height": 480, "environment": False}


def _corner_spheres():
    """Small red spheres at :data:`_CORNERS` 0.4 m in front of :data:`_CORNER_CAMERA`."""
    builder = newton.ModelBuilder()
    for x, y in _CORNERS:
        center = wp.vec3(0.4 * x, -0.4 * y, 0.6)
        builder.add_shape_sphere(
            -1, xform=wp.transform(center, wp.quat_identity()), radius=0.012, color=(0.9, 0.05, 0.05)
        )
    return builder.finalize(device="cpu")


def _corner_centroids(mask):
    """Centroids [px] of the corner spheres' silhouettes in a boolean image mask."""
    ys, xs = np.nonzero(mask)
    result = []
    for x, y in _CORNERS:
        near = (np.abs(xs - 424 - 434 * x) < 60) & (np.abs(ys - 240 - 434 * y) < 60)
        result.append([xs[near].mean(), ys[near].mean()])
    return np.array(result)


def _red(image):
    """Pixels redder than any gray or black background."""
    image = np.asarray(image, dtype=int)
    return image[..., 0] - np.maximum(image[..., 1], image[..., 2]) > 25


def _pinhole(intrinsics):
    """The calibration without its lens distortion."""
    return {key: intrinsics[key] for key in ("fx", "fy", "cx", "cy", "image_width", "image_height")}


class TestMcpObservation(unittest.TestCase):
    def setUp(self):
        """Build an actual CPU sphere scene and temporary artifact directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_sphere(body, radius=0.5, color=(1.0, 0.0, 0.0))
        self.model = builder.finalize(device="cpu")
        self.session = SimpleNamespace(
            model=self.model,
            state=self.model.state(),
            viewer=None,
            time=0.0,
            frame=0,
            revision=0,
            artifact_directory=Path(self.directory.name),
        )
        self.renderer = ObservationRenderer(self.session)
        self.addCleanup(self.renderer.close)
        self.camera = {"width": 65, "height": 65, "eye": [0, 0, 3], "target": [0, 0, 0], "up": [0, 1, 0]}

    def test_metric_depth_and_png(self):
        """Preserve metric radial and forward depth while encoding valid PNGs."""
        radial = self.renderer.observe(channel="depth", raw=True, **self.camera)
        forward = self.renderer.observe(channel="forward_depth", raw=True, **self.camera)
        with np.load(radial["raw_artifact"]) as artifact:
            ray_depth = artifact["depth"]
        with np.load(forward["raw_artifact"]) as artifact:
            forward_depth = artifact["forward_depth"]
        self.assertAlmostEqual(float(ray_depth[32, 32]), 2.5, places=5)
        self.assertAlmostEqual(float(forward_depth[32, 32]), 2.5, places=5)
        self.assertGreater(float(ray_depth[32, 38]), float(forward_depth[32, 38]))
        self.assertEqual(radial["depth_stats"]["units"], "m")
        self.assertEqual(_decode_png(radial).shape, (65, 65, 3))
        self.assertEqual(_decode_png(radial)[0, 0].tolist(), [0, 0, 0])

    def test_camera_pose_and_movement(self):
        """Match xyzw poses to look-at cameras and move the actual rendered view."""
        first = self.renderer.observe(channel="shape_index", pick=[[32, 32]], **self.camera)
        pose = self.renderer.observe(width=65, height=65, channel="shape_index", pose=[0, 0, 3, 0, 0, 0, 1])
        np.testing.assert_array_equal(_decode_png(first), _decode_png(pose))
        self.assertEqual(first["picks"][0]["shape_id"], 0)
        away = self.renderer.observe(
            width=65, height=65, channel="shape_index", pose=[0, 0, 3, 0, 1, 0, 0], pick=[[32, 32]]
        )
        self.assertIsNone(away["picks"][0]["shape_id"])
        closer = self.renderer.observe(channel="depth", **(self.camera | {"eye": [0, 0, 2]}))
        self.assertAlmostEqual(closer["depth_stats"]["min"], 1.5, places=5)

    def test_image_up_and_camera_roll(self):
        """Keep positive camera Y at the top and honor quaternion roll."""
        transforms = self.session.state.body_q.numpy()
        transforms[0, :3] = [-0.5, 0.5, 0.0]
        self.session.state.body_q.assign(transforms)
        result = self.renderer.observe(channel="shape_index", **self.camera)
        ys, xs = np.nonzero(np.any(_decode_png(result), axis=-1))
        self.assertLess(float(xs.mean()), 32)
        self.assertLess(float(ys.mean()), 32)
        rolled = self.renderer.observe(channel="shape_index", width=65, height=65, pose=[0, 0, 3, 0, 0, 1, 0])
        ys, xs = np.nonzero(np.any(_decode_png(rolled), axis=-1))
        self.assertGreater(float(xs.mean()), 32)
        self.assertGreater(float(ys.mean()), 32)

    def test_refit_after_body_motion_and_cache_reuse(self):
        """Refit acceleration bounds after motion while retaining sensor buffers."""
        first = self.renderer.observe(channel="depth", **self.camera)
        sensor, rays, output = self.renderer._sensor, self.renderer._rays, self.renderer._outputs["depth"]
        transforms = self.session.state.body_q.numpy()
        transforms[0, 2] = 1.0
        self.session.state.body_q.assign(transforms)
        moved = self.renderer.observe(channel="depth", **self.camera)
        self.assertAlmostEqual(first["depth_stats"]["min"] - moved["depth_stats"]["min"], 1.0, places=5)
        self.assertIs(self.renderer._sensor, sensor)
        self.assertIs(self.renderer._rays, rays)
        self.assertIs(self.renderer._outputs["depth"], output)

    def test_observations_emit_no_deprecation_warnings(self):
        """Sensor renders, logged overlay meshes, and calibrated cameras use only current Newton APIs."""
        quad = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float32)
        self.session.overlay_callback = lambda session: [("quad", quad, np.array([0, 1, 2, 0, 2, 3]), (0, 1, 0))]
        focal = 32.5 / math.tan(math.radians(30.0))
        ideal = {"fx": focal, "fy": focal, "cx": 32.0, "cy": 32.0}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.renderer.observe(**self.camera)
            self.renderer.observe(channel="depth", intrinsics={**ideal, "k1": -0.1}, **self.camera)
            self.renderer.observe(
                intrinsics={**ideal, "distortion_model": "inverse_brown_conrady", "k1": 0.1}, **self.camera
            )
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(messages, [])

    def test_hiding_shape_after_finalize(self):
        """Clearing a shape's VISIBLE flag on the finalized model removes it from the next observation."""
        first = self.renderer.observe(channel="depth", **self.camera)
        self.assertGreater(first["depth_stats"]["valid_count"], 0)
        flags = self.model.shape_flags.numpy()
        flags[0] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags.assign(flags)
        hidden = self.renderer.observe(channel="depth", **self.camera)
        self.assertEqual(hidden["depth_stats"]["valid_count"], 0)

    def test_environment_leaves_cloth_untouched(self):
        """The ground checker applies to plane shapes only, not to cloth pixels with sentinel shape ids."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, -1.0, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=8,
            dim_y=8,
            cell_x=0.25,
            cell_y=0.25,
            mass=0.1,
        )
        self.session.model = builder.finalize(device="cpu")
        self.session.state = self.session.model.state()
        camera = {"eye": [0.0, 0.0, 4.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0], "width": 32, "height": 32}
        plain = _decode_png(self.renderer.observe(environment=False, antialias=False, **camera))
        dressed = _decode_png(self.renderer.observe(environment=True, antialias=False, **camera))
        np.testing.assert_array_equal(dressed[12:20, 12:20], plain[12:20, 12:20])

    def test_albedo_normal_and_fixed_depth_range(self):
        """Return known albedo, world normals, and explicitly normalized depth."""
        albedo = self.renderer.observe(channel="albedo", **self.camera)
        normal = self.renderer.observe(channel="normal", **self.camera)
        depth = self.renderer.observe(channel="depth", depth_range=[0, 5], **self.camera)
        np.testing.assert_allclose(_decode_png(albedo)[32, 32], [255, 0, 0], atol=1)
        np.testing.assert_allclose(_decode_png(normal)[32, 32], [127, 127, 255], atol=1)
        np.testing.assert_allclose(_decode_png(depth)[32, 32], [152, 152, 152], atol=1)

    def test_limits_and_unsupported_features(self):
        """Budget pixels per observed world and reject unsupported options before rendering."""
        original_count = self.model.world_count
        self.model.world_count = 100
        try:
            # Only the observed world is rendered, so many worlds no longer multiply the budget.
            result = self.renderer.observe(width=64, height=48, antialias=False, view="iso")
            self.assertEqual(result["aggregate_pixels"], 64 * 48)
        finally:
            self.model.world_count = original_count
            self.renderer.invalidate()
        for options in (
            {"wireframe": True},
            {"backend": "viewer"},
            {"width": True},
            {"world_id": -1},
            {"fov_y": float("nan")},
            {"pose": [0] * 7},
            {"eye": [0, 0, 0]},
            {"pick": [[65, 0]]},
            {"depth_range": [3, 1]},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.renderer.observe(**(self.camera | options))

    def test_owner_thread(self):
        """Reject rendering from a thread that does not own the simulation."""
        errors = []

        def run():
            try:
                self.renderer.observe(**self.camera)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("owner thread", errors[0])

    def test_world_selection(self):
        """Render selected-world geometry and retain shared global geometry."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for color in ((1, 0, 0), (0, 1, 0)):
            world = newton.ModelBuilder()
            body = world.add_body(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()))
            world.add_shape_sphere(body, radius=0.5, color=color)
            builder.add_world(world)
        self.session.model = builder.finalize(device="cpu")
        self.session.state = self.session.model.state()
        red = self.renderer.observe(channel="albedo", world_id=0, pick=[[32, 32], [0, 0]], **self.camera)
        green = self.renderer.observe(channel="albedo", world_id=1, pick=[[32, 32], [0, 0]], **self.camera)
        np.testing.assert_allclose(_decode_png(red)[32, 32], [255, 0, 0], atol=1)
        np.testing.assert_allclose(_decode_png(green)[32, 32], [0, 255, 0], atol=1)
        self.assertNotEqual(red["picks"][0]["shape_id"], green["picks"][0]["shape_id"])
        self.assertIsNotNone(red["picks"][1]["shape_id"])
        self.assertEqual(red["picks"][1]["shape_id"], green["picks"][1]["shape_id"])
        self.assertEqual(green["aggregate_pixels"], 65 * 65)

    def test_cpu_texture_toggle(self):
        """Render a real CPU mesh texture and disable it through the same sensor."""
        mesh = newton.Mesh(
            np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
            np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
            compute_inertia=False,
            texture=np.full((2, 2, 4), [0, 255, 0, 255], dtype=np.uint8),
        )
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=mesh, color=(1, 1, 1))
        self.session.model = builder.finalize(device="cpu")
        self.session.state = self.session.model.state()
        textured = self.renderer.observe(channel="albedo", textures=True, **self.camera)
        plain = self.renderer.observe(channel="albedo", textures=False, **self.camera)
        default = self.renderer.observe(channel="albedo", **self.camera)
        np.testing.assert_allclose(_decode_png(textured)[32, 32], [0, 255, 0], atol=1)
        np.testing.assert_allclose(_decode_png(plain)[32, 32], [255, 255, 255], atol=1)
        # Textured models render their textures unless the caller turns them off.
        np.testing.assert_allclose(_decode_png(default)[32, 32], [0, 255, 0], atol=1)

    def test_recording_stride_stop_and_timestamps(self):
        """Write a bounded sequence with reproducible frame times and stop state."""
        started = self.renderer.record(action="start", every_steps=2, max_frames=3, **self.camera)
        self.assertTrue(started["active"])
        for frame in range(1, 7):
            self.session.frame = frame
            self.session.time = frame * 0.01
            self.renderer.after_step()
        stopped = self.renderer.record(action="status")
        self.assertFalse(stopped["active"])
        self.assertEqual(stopped["frame_count"], 3)
        self.assertEqual(stopped["stop_reason"], "frame_limit")
        manifest = json.loads((Path(stopped["directory"]) / "manifest.json").read_text())
        self.assertEqual([frame["time"] for frame in manifest["frames"]], [0.0, 0.02, 0.04])
        self.assertEqual(len(list(Path(stopped["directory"]).glob("*.png"))), 3)
        restarted = self.renderer.record(action="start", **self.camera)
        self.assertTrue(restarted["active"])
        self.assertFalse(self.renderer.record(action="stop")["active"])

    def test_contact_overlay_depth_policy(self):
        """Project real world contact support points and hide occluded markers."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.49), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        state = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        points0 = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
        points1 = wp.empty_like(points0)
        newton.eval_rigid_contact_kinematics(model, state, contacts, out_point0_world=points0, out_point1_world=points1)
        count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
        self.assertGreater(count, 0)
        normal = contacts.rigid_contact_normal.numpy()[:count]
        surface0 = points0.numpy()[:count] + normal * contacts.rigid_contact_margin0.numpy()[:count, None]
        surface1 = points1.numpy()[:count] - normal * contacts.rigid_contact_margin1.numpy()[:count, None]
        rows = [{"surface0": a.tolist(), "surface1": b.tolist()} for a, b in zip(surface0, surface1, strict=True)]
        self.session.model, self.session.state = model, state
        self.session.contact_data = lambda **kwargs: {"rows": rows, "source": "generated"}
        options = self.camera | {"contacts": True, "target": [0, 0, 0.0]}
        visible = self.renderer.observe(**options)
        always = self.renderer.observe(**options, contact_depth="always")
        self.assertEqual(visible["contacts"]["drawn"], 0)
        self.assertGreater(always["contacts"]["drawn"], 0)
        self.assertFalse(np.array_equal(_decode_png(visible), _decode_png(always)))

    def test_contact_markers_follow_calibrated_intrinsics(self):
        """Contact markers use the calibrated principal point, like the rendered image."""
        top = {"surface0": [0.0, 0.0, 0.5], "surface1": [0.0, 0.0, 0.5]}
        self.session.contact_data = lambda **kwargs: {"rows": [top], "source": "test"}
        focal = 32.5 / math.tan(math.radians(30.0))
        shifted = {"fx": focal, "fy": focal, "cx": 22.0, "cy": 32.0}
        result = self.renderer.observe(contacts=True, contact_depth="always", intrinsics=shifted, **self.camera)
        self.assertEqual(result["contacts"]["drawn"], 1)
        ys, xs = np.nonzero(np.all(_decode_png(result) == (255, 32, 224), axis=-1))
        # The point lies on the optical axis, so its marker is centered on the principal point's pixel.
        np.testing.assert_allclose([xs.mean(), ys.mean()], [22.0, 32.0], atol=0.01)

    def test_recording_byte_budget_and_capture_error(self):
        """Stop at the byte budget and contain recording failures during physics."""
        self.renderer.MAX_RECORD_BYTES = 1
        result = self.renderer.record(action="start", **self.camera)
        self.assertFalse(result["active"])
        self.assertEqual(result["frame_count"], 0)
        self.assertEqual(result["stop_reason"], "byte_limit")
        self.renderer.MAX_RECORD_BYTES = 1024 * 1024
        self.renderer.record(action="start", **self.camera)
        render = self.renderer._render_sensor

        def failing_render(*args, **kwargs):
            raise RuntimeError("capture failed")

        self.renderer._render_sensor = failing_render
        try:
            self.renderer.after_step()
        finally:
            self.renderer._render_sensor = render
        result = self.renderer.record(action="status")
        self.assertFalse(result["active"])
        self.assertIn("capture_error", result["stop_reason"])

    def test_session_global_contact_surface_overlay(self):
        """Render actual global contacts at their physical surfaces through the session."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.48), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5)
        model = builder.finalize(device="cpu")
        session = SimulationSession(model, SolverXPBD(model), artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        rows = session.contact_data(refresh=True, world=0, include_global=True)["rows"]
        self.assertGreater(len(rows), 0)
        self.assertAlmostEqual(rows[0]["surface1"][2], -0.02, places=5)
        plain = session.dispatch("observe", self.camera)
        overlay = session.dispatch("observe", self.camera | {"contacts": True, "contact_depth": "always"})
        self.assertGreater(overlay["contacts"]["considered"], 0)
        self.assertGreater(overlay["contacts"]["drawn"], 0)
        self.assertFalse(np.array_equal(_decode_png(plain), _decode_png(overlay)))

    @unittest.skipUnless(
        os.environ.get("NEWTON_MCP_TEST_GL") == "1", "Set NEWTON_MCP_TEST_GL=1 for attached GL capture"
    )
    def test_attached_viewer_framing_and_restoration(self):
        """Capture actual CPU ViewerGL images while restoring settings and framing."""
        from newton.viewer import ViewerGL  # noqa: PLC0415

        with wp.ScopedDevice("cpu"):
            viewer = ViewerGL(width=128, height=96, headless=True)
        self.addCleanup(viewer.close)
        viewer.set_model(self.model)
        self.session.viewer = viewer
        camera = viewer.camera
        camera_state = vars(camera).copy()
        settings = viewer.renderer.draw_shadows, viewer.renderer.draw_wireframe
        transforms = self.session.state.body_q.numpy()
        transforms[0, :3] = [-0.5, 0.5, 0]
        self.session.state.body_q.assign(transforms)
        options = self.camera | {"width": 96, "height": 64}
        sensor = self.renderer.observe(channel="shape_index", **options)
        gl = self.renderer.observe(backend="viewer", shadows=False, **options)
        wireframe = self.renderer.observe(backend="viewer", wireframe=True, **options)
        self.assertEqual(sensor["backend"], "sensor")
        self.assertEqual(gl["source_resolution"], [128, 96])
        self.assertEqual(_decode_png(gl).shape, (64, 96, 3))
        self.assertFalse(np.array_equal(_decode_png(gl), _decode_png(wireframe)))
        rgb = _decode_png(gl)
        ys, xs = np.nonzero((rgb[..., 0] > rgb[..., 1] * 1.5) & (rgb[..., 0] > rgb[..., 2] * 1.5))
        sensor_y, sensor_x = np.nonzero(np.any(_decode_png(sensor), axis=-1))
        self.assertAlmostEqual(float(xs.mean()), float(sensor_x.mean()), delta=2)
        self.assertAlmostEqual(float(ys.mean()), float(sensor_y.mean()), delta=2)
        self.assertIs(viewer.camera, camera)
        self.assertEqual(vars(camera), camera_state)
        self.assertEqual((viewer.renderer.draw_shadows, viewer.renderer.draw_wireframe), settings)
        self.assertIsNone(viewer._visible_worlds)

    def test_auto_framing_centers_scene(self):
        """Frame the scene from view presets when no camera is given."""
        transforms = self.session.state.body_q.numpy()
        transforms[0, :3] = [2.0, -1.0, 0.5]
        self.session.state.body_q.assign(transforms)
        for view in ("iso", "top", "front"):
            result = self.renderer.observe(channel="shape_index", width=64, height=64, view=view)
            ys, xs = np.nonzero(np.any(_decode_png(result), axis=-1))
            self.assertAlmostEqual(float(xs.mean()), 31.5, delta=3)
            self.assertAlmostEqual(float(ys.mean()), 31.5, delta=3)
            # The sphere should fill a sizeable but not overflowing part of the image.
            self.assertGreater(len(xs), 64 * 64 * 0.05)
            self.assertLess(len(xs), 64 * 64 * 0.9)
            self.assertEqual(result["camera"]["auto_framed"]["view"], view)
        with self.assertRaises(ValueError):
            self.renderer.observe(view="diagonal")

    def test_color_is_antialiased_and_environment_is_drawn(self):
        """Supersample color edges, draw a sky behind the scene, and checker ground planes."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5, color=(1.0, 0.0, 0.0))
        self.session.model = builder.finalize(device="cpu")
        self.session.state = self.session.model.state()
        camera = {"eye": [3.0, -3.0, 1.0], "target": [0.0, 0.0, 0.5], "width": 96, "height": 64, "shadows": False}
        plain = self.renderer.observe(**camera, antialias=False, environment=False)
        rich = self.renderer.observe(**camera)
        self.assertEqual(rich["settings"]["supersample"], 2)
        self.assertIn("ground checker cells", rich["settings"]["environment"])
        plain_image, rich_image = _decode_png(plain).astype(int), _decode_png(rich).astype(int)
        # The top rows show sky instead of the clear color.
        self.assertGreater(rich_image[0, :, 2].mean(), rich_image[0, :, 0].mean())
        self.assertFalse(np.array_equal(plain_image[0], rich_image[0]))
        # Anti-aliasing produces intermediate colors along the sphere silhouette.
        self.assertGreater(
            len(np.unique(rich_image.reshape(-1, 3), axis=0)), len(np.unique(plain_image.reshape(-1, 3), axis=0))
        )

    def test_calibrated_intrinsics_match_pinhole_and_shift_principal_point(self):
        """Render calibrated OpenCV intrinsics; an ideal camera matches the equivalent fov_y render."""
        focal = 32.5 / math.tan(math.radians(30.0))
        ideal = {"fx": focal, "fy": focal, "cx": 32.0, "cy": 32.0}
        pinhole = self.renderer.observe(channel="depth", raw=True, fov_y=60.0, **self.camera)
        calibrated = self.renderer.observe(channel="depth", raw=True, intrinsics=ideal, **self.camera)
        self.assertAlmostEqual(calibrated["camera"]["fov_y"], 60.0, places=4)
        with np.load(pinhole["raw_artifact"]) as a, np.load(calibrated["raw_artifact"]) as b:
            np.testing.assert_allclose(a["depth"], b["depth"], atol=1e-4)
        # Moving the principal point shifts the sphere in the image.
        shifted = self.renderer.observe(channel="depth", raw=True, intrinsics={**ideal, "cx": 22.0}, **self.camera)
        with np.load(shifted["raw_artifact"]) as c:
            hit = np.nonzero(c["depth"][32] > 0)[0]
        self.assertLess(hit.mean(), 32.0 - 5.0)
        with self.assertRaises(ValueError):
            self.renderer.observe(intrinsics={"fx": 1.0}, **self.camera)

    def test_inverse_brown_conrady_distortion(self):
        """RealSense distortion maps distorted pixels straight to rays and matches the ideal camera at zero."""
        focal = 32.5 / math.tan(math.radians(30.0))
        ideal = {"fx": focal, "fy": focal, "cx": 32.0, "cy": 32.0}
        realsense = {**ideal, "distortion_model": "inverse_brown_conrady", "k1": 0.0}
        pinhole = self.renderer.observe(channel="depth", raw=True, intrinsics=ideal, **self.camera)
        zero = self.renderer.observe(channel="depth", raw=True, intrinsics=realsense, **self.camera)
        with np.load(pinhole["raw_artifact"]) as a, np.load(zero["raw_artifact"]) as b:
            np.testing.assert_allclose(a["depth"], b["depth"], atol=1e-4)
        self.renderer.observe(channel="depth", intrinsics={**realsense, "k1": 0.2}, **self.camera)
        # The top-left pixel is centered at image coordinates (0, 0).
        x, y = (0.0 - 32.0) / focal, (0.0 - 32.0) / focal
        r2 = x * x + y * y
        expected = np.array([x * (1 + 0.2 * r2), -y * (1 + 0.2 * r2), -1.0])
        np.testing.assert_allclose(self.renderer._rays.numpy()[0, 0, 1], expected / np.linalg.norm(expected), atol=1e-6)
        with self.assertRaises(ValueError):
            self.renderer.observe(intrinsics={**realsense, "k4": 0.1}, **self.camera)

    @unittest.skipUnless(
        importlib.util.find_spec("ovrtx") is not None and wp.is_cuda_available(), "requires ovrtx and CUDA"
    )
    def test_rtx_backend_renders_selected_world(self):
        """Path-trace one world with the rtx backend and rebuild after a color edit."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.3, color=(0.9, 0.1, 0.1))
        self.session.model = builder.finalize(device="cuda:0")
        self.session.state = self.session.model.state()
        camera = {"eye": [2.0, -2.0, 1.2], "target": [0.0, 0.0, 0.4], "width": 64, "height": 48, "samples": 4}
        first = self.renderer.observe(backend="rtx", **camera)
        image = _decode_png(first).astype(int)
        self.assertEqual(image.shape, (48, 64, 3))
        self.assertTrue(first["renderer_rebuilt"])
        center = image[20:28, 28:36].mean(axis=(0, 1))
        self.assertGreater(center[0], center[1] + 20)
        again = self.renderer.observe(backend="rtx", **camera)
        self.assertFalse(again["renderer_rebuilt"])
        colors = self.session.model.shape_color.numpy()
        colors[-1] = (0.1, 0.1, 0.9)
        self.session.model.shape_color.assign(colors)
        recolored = self.renderer.observe(backend="rtx", **camera)
        self.assertTrue(recolored["renderer_rebuilt"])
        center = _decode_png(recolored).astype(int)[20:28, 28:36].mean(axis=(0, 1))
        self.assertGreater(center[2], center[0] + 20)

    def test_multi_view_grid_and_reference_comparison(self):
        """Tile several views in one image and compare a render against a reference photo."""
        grid = self.renderer.observe(views=["top", {"label": "custom", **self.camera}], width=40, height=30)
        self.assertEqual(len(grid["views"]), 2)
        # Two views sit side by side; the row is as tall as the larger view.
        self.assertEqual(_decode_png(grid).shape[:2], (65, 40 + 4 + 65))
        four = self.renderer.observe(views=["iso", "top", "front", "right"], width=40, height=30)
        self.assertEqual(_decode_png(four).shape[:2], (2 * 30 + 4, 2 * 40 + 4))
        single = self.renderer.observe(**self.camera)
        reference = Path(self.directory.name) / "reference.png"
        reference.write_bytes(base64.b64decode(single["image_base64"]))
        camera = {k: v for k, v in self.camera.items() if k not in ("width", "height")}
        same = self.renderer.observe(reference=str(reference), **camera)
        self.assertEqual(same["reference"]["mismatch_fraction"], 0.0)
        self.assertEqual(_decode_png(same).shape, (65, 3 * 65 + 2 * 4, 3))
        transforms = self.session.state.body_q.numpy()
        transforms[0, 0] = 0.4
        self.session.state.body_q.assign(transforms)
        moved = self.renderer.observe(reference=str(reference), **camera)
        self.assertGreater(moved["reference"]["mismatch_fraction"], 0.05)
        with self.assertRaises(ValueError):
            self.renderer.observe(reference=[str(reference)])

    def test_filmstrip_advances_and_compares(self):
        """Capture labeled frames at simulation times and compare them with reference images."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.2)
        model = builder.finalize(device="cpu")
        session = SimulationSession(model, SolverXPBD(model), dt=0.01, artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        camera = {"eye": [0.0, -6.0, 1.0], "target": [0.0, 0.0, 1.0], "width": 48, "height": 32}
        strip = session.dispatch("filmstrip", {"times": [0.0, 0.1, 0.3], **camera})
        self.assertEqual(strip["times"], [0.0, 0.1, 0.3])
        self.assertEqual(strip["steps_advanced"], 30)
        self.assertAlmostEqual(session.time, 0.3)
        self.assertEqual(_decode_png(strip).shape[:2], (32, 3 * 48 + 2 * 4))
        paths = []
        for index, t in enumerate((0.0, 0.1, 0.3)):
            session.dispatch("reset")
            session.dispatch("step", {"count": round(t / 0.01)}) if t else None
            frame = session.dispatch("observe", camera)
            paths.append(str(Path(self.directory.name) / f"frame-{index}.png"))
            Path(paths[-1]).write_bytes(base64.b64decode(frame["image_base64"]))
        compared = session.dispatch(
            "filmstrip", {"times": [0.0, 0.1, 0.3], "reset": True, "references": [paths], **camera}
        )
        self.assertEqual([row["mismatch_fraction"] for row in compared["mismatch"]], [0.0, 0.0, 0.0])
        self.assertEqual(_decode_png(compared).shape[0], 3 * 32 + 2 * 4)
        # Recorded video: an (N, H, W, 3) array with extra frames that stride skips, a mask, and edge panels.
        frames = np.stack(
            [_decode_png({"image_base64": base64.b64encode(Path(p).read_bytes()).decode()}) for p in paths]
        )
        padded = np.stack([frames[0], frames[0], frames[1], frames[1], frames[2], frames[2]])
        mask = np.ones(frames.shape[1:3], dtype=bool)
        mask[:, :8] = False
        video = session.dispatch(
            "filmstrip",
            {
                "times": [0.0, 0.05, 0.1, 0.2, 0.3, 0.35],
                "reset": True,
                "references": padded,
                "stride": 2,
                "mask": mask,
                "comparison": "edges",
                **camera,
            },
        )
        self.assertEqual(video["times"], [0.0, 0.1, 0.3])
        self.assertEqual([row["mismatch_fraction"] for row in video["mismatch"]], [0.0, 0.0, 0.0])
        self.assertGreater(video["metrics_mean"]["ssim"], 0.99)
        directory = Path(self.directory.name) / "frames"
        directory.mkdir()
        for index, path in enumerate(paths):
            (directory / f"{index:03d}.png").write_bytes(Path(path).read_bytes())
        # Unsorted times keep their references.
        shuffled = session.dispatch(
            "filmstrip",
            {"times": [0.3, 0.0, 0.1], "reset": True, "references": [paths[2], paths[0], paths[1]], **camera},
        )
        self.assertEqual(shuffled["times"], [0.0, 0.1, 0.3])
        self.assertEqual([row["mismatch_fraction"] for row in shuffled["mismatch"]], [0.0, 0.0, 0.0])
        from_directory = session.dispatch(
            "filmstrip", {"times": [0.0, 0.1, 0.3], "reset": True, "references": str(directory), **camera}
        )
        self.assertEqual([row["mismatch_fraction"] for row in from_directory["mismatch"]], [0.0, 0.0, 0.0])
        counted = session.dispatch("filmstrip", {"count": 2, "every_steps": 5, "reset": True, "view": "front"})
        self.assertEqual(counted["times"], [0.0, 0.05])
        with self.assertRaisesRegex(ValueError, "precede"):
            session.dispatch("filmstrip", {"times": [0.0]})

    def test_filmstrip_pages_fit_the_display_size(self):
        """Long filmstrips wrap into bands and pages that fit the display size, and show() returns every page."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.2)
        model = builder.finalize(device="cpu")
        session = SimulationSession(
            model, SolverXPBD(model), dt=0.01, artifact_directory=self.directory.name, allow_execute=True
        )
        self.addCleanup(session.close)
        camera = {"eye": [0.0, -6.0, 1.0], "target": [0.0, 0.0, 1.0]}
        size = {"width": 96, "height": 64}
        times = [round(0.02 * k, 2) for k in range(10)]
        views = [{"label": "a", **camera}, {"label": "b", "eye": [6.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]}]
        original = ObservationRenderer.DISPLAY_EDGE
        ObservationRenderer.DISPLAY_EDGE = 300
        self.addCleanup(setattr, ObservationRenderer, "DISPLAY_EDGE", original)
        strip = session.dispatch("filmstrip", {"times": times, "reset": True, "views": views, **size})
        # Bands of 3 times (100 px columns), 2 bands per page (144 px bands): 10 times on 2 pages, full-size frames.
        self.assertEqual(strip["pages"], 2)
        self.assertNotIn("thumbnail_scale", strip)
        self.assertIn("3 times per band, 2 bands per page", strip["layout"])
        pages = [_decode_png(strip), *(_decode_png(page) for page in strip["images"])]
        for page in pages:
            self.assertLessEqual(max(page.shape[:2]), 300)
        self.assertEqual(pages[0].shape[:2], (2 * (2 * 64 + 4) + 12, 3 * 96 + 2 * 4))
        # One page at most: frames shrink instead.
        single = session.dispatch("filmstrip", {"times": times, "reset": True, "views": views, **size, "max_pages": 1})
        self.assertNotIn("images", single)
        self.assertEqual(single["thumbnail_scale"], 0.5)
        self.assertLessEqual(max(_decode_png(single).shape[:2]), 300)
        shown = session.dispatch(
            "execute",
            {
                "code": f"show(session.dispatch('filmstrip', {{'times': {times}, 'reset': True, 'views': {views}, "
                "'width': 96, 'height': 64}), 'strip')"
            },
        )
        self.assertEqual(len(shown["images"]), 2)

    def test_projection_matches_camera_rays(self):
        """Overlay projection inverts the renderer's rays for pinhole, OpenCV, and inverse Brown-Conrady cameras."""
        camera = {"width": 40, "height": 30, "eye": [1.0, -2.5, 0.8], "target": [0.1, 0.0, 0.2]}
        calibration = {"fx": 30.0, "fy": 32.0, "cx": 22.0, "cy": 13.0}
        opencv = {**calibration, "k1": -0.12, "k2": 0.03, "p1": 0.002, "p2": -0.003, "k4": 0.01, "s1": 0.001}
        realsense = {**calibration, "distortion_model": "inverse_brown_conrady", "k1": 0.1, "k2": -0.02, "p1": 0.002}
        ys, xs = np.mgrid[0:30, 0:40]
        centers = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float64)
        for options in ({}, {"intrinsics": opencv}, {"intrinsics": realsense}):
            with self.subTest(options=options):
                metadata = self.renderer.observe(channel="depth", **camera, **options)
                directions = self.renderer._rays.numpy().reshape(30, 40, 2, 3)[:, :, 1].reshape(-1, 3)
                pose = np.asarray(metadata["camera"]["pose"], dtype=np.float64)
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=np.float64).reshape(3, 3)
                points = pose[:3] + 2.0 * directions.astype(np.float64) @ rotation.T
                pixels, depth = _project(
                    points, pose, 40, 30, metadata["camera"]["fov_y"], metadata["camera"].get("intrinsics")
                )
                np.testing.assert_allclose(pixels, centers, atol=1.0e-3)
                self.assertTrue(np.all(depth > 0.0))
        # Points behind the camera, or past the radius where the distortion polynomial folds back, have no pixel.
        folding = {
            "fx": 30.0,
            "fy": 30.0,
            "cx": 20.0,
            "cy": 15.0,
            "image_width": 40.0,
            "image_height": 30.0,
            "k1": -0.3,
        }
        pixels, _ = _project(
            [[0.1, 0.0, -1.0], [2.0, 0.0, -1.0], [0.0, 0.0, 1.0]], [0, 0, 0, 0, 0, 0, 1], 40, 30, 60.0, folding
        )
        self.assertTrue(np.isfinite(pixels[0]).all())
        self.assertTrue(np.isnan(pixels[1:]).all())

    def test_camera_body_follows_its_body(self):
        """A camera mounted on a body renders like the equivalent fixed pose and moves with the body."""
        builder = newton.ModelBuilder()
        target = builder.add_body(xform=wp.transform_identity(), label="scene/target")
        builder.add_shape_sphere(target, radius=0.5)
        builder.add_body(xform=wp.transform(wp.vec3(0.0, -3.0, 0.0), wp.quat(*_LOOK_ALONG_Y)), label="scene/mount")
        model = builder.finalize(device="cpu")
        self.session.model, self.session.state = model, model.state()
        camera = {"width": 48, "height": 32, "channel": "shape_index"}

        def fixed(position):
            return _decode_png(self.renderer.observe(pose=[*position, *_LOOK_ALONG_Y], **camera))

        mounted = self.renderer.observe(camera_body="mount", **camera)
        _assert_nearly_equal_images(self, _decode_png(mounted), fixed([0.0, -3.0, 0.0]))
        self.assertEqual(mounted["camera"]["mount"]["body"], 1)
        self.assertEqual(mounted["camera"]["mount"]["label"], "scene/mount")
        np.testing.assert_allclose(_hit_centroid(_decode_png(mounted)), [23.5, 15.5], atol=0.5)
        # camera_offset is a pose in the body frame, whose x axis is world x here.
        offset = self.renderer.observe(camera_body="scene/mount", camera_offset=[0.5, 0, 0, 0, 0, 0, 1], **camera)
        _assert_nearly_equal_images(self, _decode_png(offset), fixed([0.5, -3.0, 0.0]))
        transforms = self.session.state.body_q.numpy()
        transforms[1, :3] = [0.0, -2.0, 0.4]
        self.session.state.body_q.assign(transforms)
        moved = self.renderer.observe(camera_body=1, **camera)
        np.testing.assert_allclose(moved["camera"]["pose"][:3], [0.0, -2.0, 0.4], atol=1.0e-6)
        _assert_nearly_equal_images(self, _decode_png(moved), fixed([0.0, -2.0, 0.4]))
        for bad in (
            {"camera_body": "mount", "eye": [0.0, 0.0, 3.0]},
            {"camera_body": "mount", "view": "top"},
            {"camera_offset": [0, 0, 0, 0, 0, 0, 1]},
            {"camera_body": "missing"},
            {"camera_body": 7},
            {"camera_body": "mount", "camera_offset": [0, 0, 0, 0, 0, 0, 0]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.renderer.observe(**camera, **bad)

    def test_body_labels_resolve_in_the_observed_world(self):
        """Replicated worlds share labels: world_id picks the camera body and overlay bodies, ambiguity fails."""
        robot = newton.ModelBuilder()
        link = robot.add_body(xform=wp.transform_identity(), label="arm/link")
        robot.add_shape_sphere(link, radius=0.3)
        robot.add_body(xform=wp.transform(wp.vec3(0.0, -3.0, 0.0), wp.quat(*_LOOK_ALONG_Y)), label="arm/link_camera")
        builder = newton.ModelBuilder()
        builder.replicate(robot, 2, spacing=(1.0, 0.0, 0.0))
        model = builder.finalize(device="cpu")
        self.session.model, self.session.state = model, model.state()
        camera = {"width": 48, "height": 32, "channel": "shape_index"}
        mounted = [self.renderer.observe(camera_body="link_camera", world_id=w, **camera) for w in (0, 1)]
        self.assertEqual([m["camera"]["mount"]["body"] for m in mounted], [1, 3])
        # Each world's camera sees its own link centered (shape ids, and so the colors, differ per world).
        for result in mounted:
            np.testing.assert_allclose(_hit_centroid(_decode_png(result)), [23.5, 15.5], atol=0.5)
        self.assertEqual(
            self.renderer.observe(camera_body="arm/link", world_id=1, **camera)["camera"]["mount"]["body"], 2
        )
        # Replication centers the worlds at x = -0.5 and 0.5, so a camera on the y axis sees them on either side.
        fixed = {"eye": [0.0, -4.0, 0.0], "target": [0.0, 0.0, 0.0], **camera}
        for world in (0, 1):
            plain = _decode_png(self.renderer.observe(world_id=world, **fixed))
            marked = self.renderer.observe(world_id=world, overlay={"link": {"body": "link"}}, **fixed)
            np.testing.assert_allclose(marked["overlay"]["link"], _hit_centroid(plain), atol=1.0)
            self.assertEqual(marked["overlay"]["link"][0] > 23.5, world == 1)
        with self.assertRaisesRegex(ValueError, "matches 2 bodies"):
            self.renderer.observe(camera_body="lin", **camera)
        with self.assertRaisesRegex(ValueError, "belongs to world 0"):
            self.renderer.observe(camera_body=1, world_id=1, **camera)

    def test_observe_overlay_marks_simulated_and_reference_images(self):
        """Overlay markers appear on both panels, report pixel coordinates, and leave the metrics unchanged."""
        plain = self.renderer.observe(**self.camera)
        self.assertFalse(_has_color(_decode_png(plain), _MAGENTA))
        reference = Path(self.directory.name) / "reference.png"
        reference.write_bytes(base64.b64decode(plain["image_base64"]))
        camera = {k: v for k, v in self.camera.items() if k not in ("width", "height")}
        overlay = {"center": [0.0, 0.0, 0.0], "rim": {"body": 0, "point": [0.5, 0.0, 0.0]}}
        marked = self.renderer.observe(reference=str(reference), overlay=overlay, **camera)
        self.assertEqual(marked["reference"]["mismatch_fraction"], 0.0)
        self.assertEqual(marked["overlay"]["center"], [32.0, 32.0])
        focal = 32.5 / math.tan(math.radians(30.0))
        np.testing.assert_allclose(marked["overlay"]["rim"], [32.0 + focal * 0.5 / 3.0, 32.0], atol=0.06)
        image = _decode_png(marked)
        self.assertTrue(_has_color(image[:, :65], _MAGENTA))
        self.assertTrue(_has_color(image[:, 69:134], _MAGENTA))
        with self.assertRaisesRegex(ValueError, "allow_execute"):
            self.renderer.observe(overlay={"center": "np.zeros(3)"}, **self.camera)
        with self.assertRaisesRegex(ValueError, "world point"):
            self.renderer.observe(overlay={"bad": [1.0, 2.0]}, **self.camera)

    def test_filmstrip_mounted_camera_and_overlay(self):
        """A mounted camera follows a falling body to every capture time; overlays mark it in every frame."""
        builder = newton.ModelBuilder()
        ball = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), label="ball")
        builder.add_shape_sphere(ball, radius=0.2)
        model = builder.finalize(device="cpu")
        session = SimulationSession(model, SolverXPBD(model), dt=0.01, artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        realsense = {"fx": 40.0, "fy": 40.0, "cx": 20.0, "cy": 14.0, "distortion_model": "inverse_brown_conrady"}
        chase = {"camera_body": "ball", "camera_offset": [0.0, -3.0, 0.0, *_LOOK_ALONG_Y], "intrinsics": realsense}
        fixed = {"eye": [0.0, -6.0, 1.2], "target": [0.0, 0.0, 1.2], "fov_y": 30.0}
        size = {"width": 48, "height": 32}
        times = [0.0, 0.3, 0.6]
        strip = session.dispatch(
            "filmstrip",
            {
                "times": times,
                "reset": True,
                "views": [{"label": "chase", **chase}, {"label": "fixed", **fixed}],
                "overlay": {"ball": {"body": "ball"}},
                **size,
            },
        )
        rows = {
            view: [r["pixels"]["ball"] for r in strip["overlay"] if r["view"] == view] for view in ("chase", "fixed")
        }
        self.assertEqual([r["time"] for r in strip["overlay"]], [0.0, 0.0, 0.3, 0.3, 0.6, 0.6])
        # The mounted camera keeps the ball on its optical axis while it falls; the fixed camera sees it drop.
        np.testing.assert_allclose(rows["chase"], [[20.0, 14.0]] * 3, atol=0.05)
        self.assertGreater(rows["fixed"][2][1], rows["fixed"][0][1] + 10.0)
        for spec, pixel in ((chase, rows["chase"][-1]), (fixed, rows["fixed"][-1])):
            rendered = session.render(channel="shape_index", **spec, **size)
            np.testing.assert_allclose(_hit_centroid(rendered), pixel, atol=1.0)
        # Against references, markers go on both rows after scoring, so identical frames still match exactly.
        frames = []
        session.dispatch("reset")
        for t in times:
            while session.time < t - 0.005:
                session.dispatch("step", {"count": 1})
            frames.append(session.render(**chase, **size))
        self.assertFalse(any(_has_color(frame, _MAGENTA) for frame in frames))
        compared = session.dispatch(
            "filmstrip",
            {
                "times": times,
                "reset": True,
                "references": frames,
                "overlay": {"ball": {"body": "ball"}},
                **chase,
                **size,
            },
        )
        self.assertEqual([row["mismatch_fraction"] for row in compared["mismatch"]], [0.0, 0.0, 0.0])
        grid = _decode_png(compared)
        for column in range(3):
            left = column * (48 + 4)
            self.assertTrue(_has_color(grid[0:32, left : left + 48], _MAGENTA))
            self.assertTrue(_has_color(grid[36:68, left : left + 48], _MAGENTA))
        # Expressions and callables are evaluated at every capture time; expressions need allow_execute.
        expression = {"top": "state.body_q.numpy()[0, :3] + np.array([0.0, 0.0, 0.2])"}
        with self.assertRaisesRegex(ValueError, "allow_execute"):
            session.dispatch("filmstrip", {"times": [0.0], "reset": True, "overlay": expression, **fixed, **size})
        session.allow_execute = True
        evaluated = session.dispatch(
            "filmstrip",
            {"times": times, "reset": True, "overlay": {**expression, "fn": lambda: [0.0, 0.0, 1.2]}, **fixed, **size},
        )
        tops = [row["pixels"]["top"] for row in evaluated["overlay"]]
        self.assertGreater(tops[2][1], tops[0][1] + 10.0)
        self.assertEqual([row["pixels"]["fn"] for row in evaluated["overlay"]], [[23.5, 15.5]] * 3)
        with self.assertRaisesRegex(ValueError, "camera_offset requires camera_body"):
            session.dispatch(
                "filmstrip", {"times": [0.0], "reset": True, "camera_offset": [0, 0, 0, 0, 0, 0, 1], **size}
            )
        # Recordings follow the mounted camera as well; the manifest keeps overlay callables as text.
        session.dispatch("reset")
        options = {**chase, **size, "overlay": {"ball": {"body": "ball"}, "fn": lambda: [0.0, 0.0, 1.2]}}
        status = session.dispatch("record", {"action": "start", "every_steps": 30, **options})
        session.dispatch("step", {"count": 60})
        session.dispatch("record", {"action": "stop"})
        manifest = json.loads((Path(status["directory"]) / "manifest.json").read_text())
        self.assertEqual(len(manifest["frames"]), 3)
        for frame in manifest["frames"]:
            self.assertEqual(frame["camera"]["mount"]["label"], "ball")
            np.testing.assert_allclose(frame["overlay"]["ball"], [20.0, 14.0], atol=0.05)

    def test_mounted_realsense_camera_matches_its_rays(self):
        """A body-mounted 848x480 RealSense camera takes its pose from the body and projects like its rays."""
        builder = newton.ModelBuilder()
        tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.4, 0.2)), 2.6)
        body_pose = [0.3, -0.2, 0.9, *tilt]
        builder.add_body(xform=wp.transform(*body_pose), label="arm/wrist_camera")
        lens = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.2) * wp.quat(*_OPTICAL)
        offset = [0.02, -0.01, 0.03, *lens]
        position, rotation = _mounted_pose(body_pose, offset)
        # Small spheres 0.4 m in front of the camera near the image corners and on the optical axis.
        centers = [position + rotation @ (0.4 * np.array([x, -y, -1.0])) for x, y in _CORNERS]
        spheres = [
            builder.add_shape_sphere(-1, xform=wp.transform(c, wp.quat_identity()), radius=0.006) for c in centers
        ]
        model = builder.finalize(device="cpu")
        self.session.model, self.session.state = model, model.state()
        camera = {
            "camera_body": "wrist_camera",
            "camera_offset": offset,
            "intrinsics": _REALSENSE_848,
            "width": 848,
            "height": 480,
        }
        overlay = {f"sphere{i}": center.tolist() for i, center in enumerate(centers)}
        mounted = self.renderer.observe(channel="shape_index", raw=True, overlay=overlay, **camera)
        self.assertEqual((mounted["width"], mounted["height"]), (848, 480))
        pose = np.asarray(mounted["camera"]["pose"], dtype=np.float64)
        np.testing.assert_allclose(pose[:3], position, atol=1.0e-6)
        np.testing.assert_allclose(_rotation(pose[3:]), rotation, atol=1.0e-6)
        self.assertEqual(mounted["camera"]["mount"]["label"], "arm/wrist_camera")
        intrinsics = mounted["camera"]["intrinsics"]
        self.assertEqual(intrinsics["distortion_model"], "inverse_brown_conrady")
        # Points along every pixel's ray project back to that pixel's center.
        directions = self.renderer._rays.numpy()[:, :, 1].reshape(-1, 3).astype(np.float64)
        pixels, depth = _project(pose[:3] + 0.5 * directions @ _rotation(pose[3:]).T, pose, 848, 480, 0.0, intrinsics)
        ys, xs = np.mgrid[0:480, 0:848]
        np.testing.assert_allclose(pixels, np.stack([xs.ravel(), ys.ravel()], axis=-1), atol=1.0e-3)
        self.assertTrue(np.all(depth > 0.0))
        # Overlay rings sit on the rendered spheres, which the same camera without distortion would miss.
        pinhole = _pinhole(intrinsics)
        with np.load(mounted["raw_artifact"]) as artifact:
            shape_index = artifact["shape_index"]
        for i, shape in enumerate(spheres):
            with self.subTest(sphere=i):
                ys, xs = np.nonzero(shape_index == shape)
                rendered = np.array([xs.mean(), ys.mean()])
                marked = np.asarray(mounted["overlay"][f"sphere{i}"])
                np.testing.assert_allclose(marked, rendered, atol=0.35)
                if _CORNERS[i] != (0.0, 0.0):
                    undistorted = _project(centers[i], pose, 848, 480, 0.0, pinhole)[0][0]
                    self.assertGreater(np.linalg.norm(undistorted - rendered), 3.0)

    def test_filmstrip_pages_mounted_realsense_camera_with_references(self):
        """A filmstrip follows a falling, turning wrist camera at 848x480 and pages frames against references."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        # The body's +Z, the camera's optical axis, points down; the body falls and turns about the vertical.
        down = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
        wrist = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.2), down),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
            label="wrist",
        )
        builder.joint_qd[-6:] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5]
        for x, y, color in ((0.0, 0.0, (0.9, 0.2, 0.1)), (0.25, 0.1, (0.1, 0.7, 0.2)), (-0.2, -0.15, (0.2, 0.3, 0.9))):
            builder.add_shape_box(
                -1, xform=wp.transform(wp.vec3(x, y, 0.05), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.05, color=color
            )
        model = builder.finalize(device="cpu")
        session = SimulationSession(model, SolverXPBD(model), dt=0.01, artifact_directory=self.directory.name)
        self.addCleanup(session.close)
        offset = [0.0, 0.0, 0.05, *_OPTICAL]
        common = {"intrinsics": _REALSENSE_848, "antialias": False}
        times = [round(0.05 * k, 2) for k in range(8)]
        # References: renders from fixed poses composed from the simulated body pose at each time.
        directory = Path(self.directory.name) / "wrist"
        directory.mkdir()
        poses = []
        for index, t in enumerate(times):
            while session.time < t - 0.005:
                session.dispatch("step", {"count": 1})
            position, rotation = _mounted_pose(session.state.body_q.numpy()[wrist], offset)
            poses.append([*position, *_matrix_quaternion(rotation)])
            frame = session.render(pose=poses[-1], width=848, height=480, **common)
            (directory / f"{index:03d}.png").write_bytes(_encode_png(frame))
        box = [0.25, 0.1, 0.1]
        strip = session.dispatch(
            "filmstrip",
            {
                "times": times,
                "reset": True,
                "references": str(directory),
                "camera_body": "wrist",
                "camera_offset": offset,
                "overlay": {"box": box},
                **common,
            },
        )
        self.assertEqual(strip["times"], times)
        # The mounted camera reproduces every fixed-pose reference.
        for row in strip["mismatch"]:
            self.assertLessEqual(row["mismatch_fraction"], 1.0e-4)
        # Its markers follow the body: the projection through each time's composed pose, sweeping across the image.
        intrinsics = _intrinsics(_REALSENSE_848, 848, 480)
        expected = [_project(box, pose, 848, 480, 0.0, intrinsics)[0][0] for pose in poses]
        # Reported pixels are rounded to 0.1 px.
        np.testing.assert_allclose([row["pixels"]["box"] for row in strip["overlay"]], expected, atol=0.06)
        self.assertGreater(np.linalg.norm(expected[-1] - expected[0]), 40.0)
        # Eight 848x480 times with references fit two pages at half scale: 2 bands of 3 times, then 2 times.
        self.assertEqual(strip["pages"], 2)
        self.assertEqual(strip["thumbnail_scale"], 0.5)
        self.assertIn("3 times per band, 2 bands per page", strip["layout"])
        pages = [_decode_png(strip), *(_decode_png(page) for page in strip["images"])]
        band = 3 * 240 + 2 * 4
        self.assertEqual(pages[0].shape[:2], (2 * band + 12, 3 * 424 + 2 * 4))
        self.assertEqual(pages[1].shape[:2], (band, 2 * 424 + 4))
        for page in pages:
            self.assertLessEqual(max(page.shape[:2]), ObservationRenderer.DISPLAY_EDGE)

    def test_tool_schemas_expose_mounted_cameras_and_overlays(self):
        """The structured observe, filmstrip, and record tools accept camera_body, camera_offset, and intrinsics."""
        schemas = {tool["name"]: tool["inputSchema"]["properties"] for tool in TOOLS}
        for name in ("newton_observe", "newton_filmstrip", "newton_record"):
            for key in ("camera_body", "camera_offset", "intrinsics"):
                self.assertIn(key, schemas[name], f"{name} lacks {key}")
        self.assertIn("overlay", schemas["newton_observe"])
        self.assertIn("overlay", schemas["newton_filmstrip"])


if __name__ == "__main__":
    unittest.main()
