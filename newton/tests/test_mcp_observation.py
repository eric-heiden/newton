# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise real CPU sensor observations and optional attached OpenGL capture."""

import base64
import json
import os
import struct
import tempfile
import threading
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.mcp.observation import ObservationRenderer
from newton.mcp import SimulationSession
from newton.solvers import SolverXPBD


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

    def test_albedo_normal_and_fixed_depth_range(self):
        """Return known albedo, world normals, and explicitly normalized depth."""
        albedo = self.renderer.observe(channel="albedo", **self.camera)
        normal = self.renderer.observe(channel="normal", **self.camera)
        depth = self.renderer.observe(channel="depth", depth_range=[0, 5], **self.camera)
        np.testing.assert_allclose(_decode_png(albedo)[32, 32], [255, 0, 0], atol=1)
        np.testing.assert_allclose(_decode_png(normal)[32, 32], [127, 127, 255], atol=1)
        np.testing.assert_allclose(_decode_png(depth)[32, 32], [152, 152, 152], atol=1)

    def test_limits_and_unsupported_features(self):
        """Reject aggregate allocations and unsupported options before rendering."""
        original_count = self.model.world_count
        self.model.world_count = 100
        try:
            with self.assertRaisesRegex(ValueError, "aggregate pixels"):
                self.renderer.observe(width=1024, height=1024)
            self.assertIsNone(self.renderer._sensor)
        finally:
            self.model.world_count = original_count
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
        self.assertEqual(green["aggregate_pixels"], 2 * 65 * 65)

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
        np.testing.assert_allclose(_decode_png(textured)[32, 32], [0, 255, 0], atol=1)
        np.testing.assert_allclose(_decode_png(plain)[32, 32], [255, 255, 255], atol=1)

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

    def test_recording_byte_budget_and_capture_error(self):
        """Stop at the byte budget and contain recording failures during physics."""
        self.renderer.MAX_RECORD_BYTES = 1
        result = self.renderer.record(action="start", **self.camera)
        self.assertFalse(result["active"])
        self.assertEqual(result["frame_count"], 0)
        self.assertEqual(result["stop_reason"], "byte_limit")
        self.renderer.MAX_RECORD_BYTES = 1024 * 1024
        self.renderer.record(action="start", **self.camera)
        self.session.model.world_count = 10000
        try:
            self.renderer.after_step()
        finally:
            self.session.model.world_count = 1
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


if __name__ == "__main__":
    unittest.main()
