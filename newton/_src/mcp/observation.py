# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded camera observations for an application-owned simulation session."""

from __future__ import annotations

import base64
import copy
import json
import math
import struct
import threading
import uuid
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton.sensors import SensorTiledCamera


def _integer(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _vector(name: str, value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _camera_pose(eye, target, up, pose, up_axis: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if pose is not None:
        if any(value is not None for value in (eye, target, up)):
            raise ValueError("pose and eye/target/up are mutually exclusive")
        pose = _vector("pose", pose, 7)
        norm = np.linalg.norm(pose[3:])
        if norm < 1.0e-12:
            raise ValueError("pose quaternion must be nonzero")
        quaternion = pose[3:] / norm
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*quaternion)), dtype=np.float64).reshape(3, 3)
        return pose[:3], rotation, [*pose[:3].tolist(), *quaternion.tolist()]
    eye = _vector("eye", (3.0, -3.0, 2.0) if eye is None else eye, 3)
    target = _vector("target", (0.0, 0.0, 0.0) if target is None else target, 3)
    up = _vector("up", np.eye(3)[up_axis] if up is None else up, 3)
    forward = target - eye
    if np.linalg.norm(forward) < 1.0e-12:
        raise ValueError("eye and target must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1.0e-12:
        raise ValueError("up must not be parallel to the viewing direction")
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(right, forward), -forward))
    quaternion = wp.quat_from_matrix(wp.mat33(*rotation.flatten()))
    return eye, rotation, [*eye.tolist(), *list(quaternion)]


def _png(rgb: np.ndarray) -> bytes:
    """Encode top-left RGB bytes without an optional imaging dependency."""
    height, width, _ = rgb.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    scanlines = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    scanlines[:, 1:] = rgb.reshape(height, width * 3)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines.tobytes()))
        + chunk(b"IEND", b"")
    )


class ObservationRenderer:
    """Render on the session owner thread using cached sensor allocations.

    Camera poses use position [m] followed by an xyzw quaternion. Local -Z is
    forward and +Y is up; image rows start at the top. ``fov_y`` is the vertical
    field of view [deg], converted to radians for the sensor ray helper.

    Sensor observations are the default even when a ViewerGL is attached.
    Viewer captures reuse its native framebuffer and resize the returned RGB
    image, preserving the requested camera aspect ratio. They never create a
    window or change the application's camera. The viewer backend supports
    color, shadows, world selection, and real mesh wireframe; texture toggles,
    raw sensor channels, picking, and depth-tested overlays require the sensor.
    """

    CHANNELS = ("color", "albedo", "depth", "forward_depth", "normal", "shape_index")
    MAX_PIXELS = 4_194_304
    MAX_RECORD_BYTES = 256 * 1024 * 1024

    class _CaptureCamera:
        """Adapt a camera pose to ViewerGL without modifying its live camera."""

        def __init__(self, camera, eye, rotation, width, height, fov_y):
            from pyglet.math import Vec3

            self._camera = copy.copy(camera)
            self.pos = Vec3(*eye)
            self._rotation = rotation
            self.width, self.height, self.fov = width, height, fov_y

        def __getattr__(self, name):
            return getattr(self._camera, name)

        def get_up(self):
            from pyglet.math import Vec3

            return Vec3(*self._rotation[:, 1])

        def get_view_matrix(self, scaling=1.0):
            from pyglet.math import Mat4, Vec3

            position = self.pos / scaling
            return np.asarray(
                Mat4.look_at(position, position - Vec3(*self._rotation[:, 2]), self.get_up()), dtype=np.float32
            )

        def get_projection_matrix(self):
            from pyglet.math import Mat4

            return np.asarray(
                Mat4.perspective_projection(self.width / self.height, self.near, self.far, self.fov), dtype=np.float32
            )

    def __init__(self, session):
        self.session = session
        self._owner_thread = threading.get_ident()
        self._sensor = None
        self._sensor_model = None
        self._buffer_key = None
        self._rays = None
        self._transforms = None
        self._outputs = {}
        self._recording = None
        self._last_recording = {"active": False, "frame_count": 0}

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Observations must run on the simulation/viewer owner thread")

    def invalidate(self):
        """Discard cached geometry and buffers after model edits or replacement."""
        self._check_thread()
        self._sensor = self._sensor_model = self._buffer_key = self._rays = self._transforms = None
        self._outputs = {}

    def observe(
        self,
        *,
        backend: str = "sensor",
        channel: str = "color",
        width: int = 640,
        height: int = 480,
        world_id: int = 0,
        eye=None,
        target=None,
        up=None,
        pose=None,
        fov_y: float = 60.0,
        shadows: bool = True,
        textures: bool | None = None,
        wireframe: bool = False,
        contacts: bool = False,
        contact_depth: str = "visible",
        depth_range=None,
        raw: bool = False,
        pick=None,
    ) -> dict:
        """Return a PNG and bounded metadata for the current simulation state.

        Depth channels retain meters in raw NPZ artifacts. PNG depth maps the
        selected near/far range to grayscale 255/50, with misses black; omitted
        ranges use valid per-frame extrema. Normal RGB maps world components
        [-1, 1] to [0, 255], with misses black. Shape IDs use a deterministic
        hash palette and retain uint32 IDs in artifacts (0xFFFFFFFF is a miss).
        Color and albedo are display/sRGB. Contact markers use the midpoint of
        world contact surfaces, including margins, with optional depth occlusion.
        """
        self._check_thread()
        model = self.session.model
        width = _integer("width", width, 1, 2048)
        height = _integer("height", height, 1, 2048)
        world_id = _integer("world_id", world_id, 0, model.world_count - 1)
        if backend not in ("sensor", "viewer"):
            raise ValueError("backend must be 'sensor' or 'viewer'")
        if channel not in self.CHANNELS:
            raise ValueError(f"channel must be one of {self.CHANNELS}")
        for name, value in (("shadows", shadows), ("wireframe", wireframe), ("contacts", contacts), ("raw", raw)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if textures is not None and not isinstance(textures, bool):
            raise ValueError("textures must be a boolean or null")
        if not isinstance(fov_y, (int, float)) or isinstance(fov_y, bool) or not 1.0 <= fov_y <= 175.0:
            raise ValueError("fov_y must be finite and in [1, 175] degrees")
        if contact_depth not in ("visible", "always"):
            raise ValueError("contact_depth must be 'visible' or 'always'")
        if depth_range is not None:
            depth_range = _vector("depth_range", depth_range, 2)
            if not 0 <= depth_range[0] < depth_range[1]:
                raise ValueError("depth_range must satisfy 0 <= near < far")
            if channel not in ("depth", "forward_depth"):
                raise ValueError("depth_range applies only to depth channels")
        if pick is not None:
            if not isinstance(pick, (list, tuple)) or len(pick) > 32:
                raise ValueError("pick must contain at most 32 [x, y] pixels")
            for pixel in pick:
                if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
                    raise ValueError("pick pixels must be [x, y] pairs")
                _integer("pick x", pixel[0], 0, width - 1)
                _integer("pick y", pixel[1], 0, height - 1)
        aggregate_pixels = model.world_count * width * height if backend == "sensor" else width * height
        if aggregate_pixels > self.MAX_PIXELS:
            raise ValueError(
                f"Observation needs {aggregate_pixels} aggregate pixels (worlds x cameras x width x height); "
                f"limit is {self.MAX_PIXELS}. Reduce resolution."
            )
        eye, rotation, camera_pose = _camera_pose(eye, target, up, pose, int(model.up_axis))
        metadata = {
            "backend": backend,
            "channel": channel,
            "width": width,
            "height": height,
            "world_id": world_id,
            "camera": {
                "pose": camera_pose,
                "fov_y": float(fov_y),
                "convention": "xyzw, -Z forward, +Y up, top-left",
                "coordinates": "simulation world coordinates [m]",
            },
            "time": float(self.session.time),
            "frame": int(getattr(self.session, "frame", 0)),
            "revision": int(getattr(self.session, "revision", 0)),
            "aggregate_pixels": aggregate_pixels,
            "settings": {"shadows": shadows, "textures": textures, "wireframe": wireframe},
        }
        if backend == "sensor":
            if wireframe:
                raise ValueError("SensorTiledCamera does not support mesh wireframe; use an attached ViewerGL backend")
            arrays = self._render_sensor(
                width, height, fov_y, camera_pose, world_id, channel, shadows, bool(textures), contacts, pick
            )
            rgb, channel_metadata = self._colorize(arrays[channel], channel, depth_range)
            metadata.update(channel_metadata)
            if pick is not None:
                metadata["picks"] = self._pick(arrays["shape_index"], pick)
            depth = arrays.get("forward_depth")
        else:
            if channel != "color" or textures is not None or raw or pick is not None:
                raise ValueError(
                    "Viewer backend supports color only; textures, raw channels and picking require sensor"
                )
            if contacts and contact_depth != "always":
                raise ValueError("Viewer contact overlays require contact_depth='always'; sensor supports occlusion")
            rgb, source_size = self._render_viewer(width, height, fov_y, eye, rotation, world_id, shadows, wireframe)
            metadata.update({"source_resolution": source_size, "normalization": "display RGB; nearest resize"})
            depth = None
            arrays = {}
        if contacts:
            metadata["contacts"] = self._overlay_contacts(rgb, eye, rotation, fov_y, world_id, contact_depth, depth)
        if raw:
            directory = Path(self.session.artifact_directory)
            directory.mkdir(parents=True, exist_ok=True)
            artifact = directory / f"observation-{uuid.uuid4().hex}.npz"
            np.savez_compressed(artifact, **arrays, camera_pose=camera_pose, fov_y=fov_y)
            metadata["raw_artifact"] = str(artifact)
        metadata["image_base64"] = base64.b64encode(_png(rgb)).decode("ascii")
        metadata["mime_type"] = "image/png"
        return metadata

    def _render_sensor(self, width, height, fov_y, pose, world_id, channel, shadows, textures, contacts, pick):
        model, state = self.session.model, self.session.state
        if self._sensor is None or self._sensor_model is not model:
            self.invalidate()
            config = SensorTiledCamera.RenderConfig(enable_shadows=True)
            self._sensor = SensorTiledCamera(model, default_render_config=config, load_textures=True)
            self._sensor.utils.create_default_light(enable_shadows=True)
            self._sensor_model = model
        key = (width, height, fov_y)
        if key != self._buffer_key:
            self._outputs = {}
            self._rays = self._sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(fov_y))
            self._transforms = wp.empty((1, model.world_count), dtype=wp.transform, device=model.device)
            self._buffer_key = key
        transform = np.broadcast_to(np.asarray(pose, dtype=np.float32), (1, model.world_count, 7)).copy()
        self._transforms.assign(transform)
        needed = {channel}
        if contacts:
            needed.add("forward_depth")
        if pick is not None:
            needed.add("shape_index")
        # Retain only this request's outputs so channel changes cannot grow the cache without bound.
        self._outputs = {name: value for name, value in self._outputs.items() if name in needed}
        for name in needed:
            if name not in self._outputs:
                create = getattr(self._sensor.utils, f"create_{name}_image_output")
                self._outputs[name] = create(width, height)
        model.bvh_refit_shapes(state)
        model.bvh_refit_particles(state)
        config = SensorTiledCamera.RenderConfig(enable_shadows=shadows, enable_textures=textures)
        self._sensor.update(
            state,
            self._transforms,
            self._rays,
            render_config=config,
            **{f"{name}_image": output for name, output in self._outputs.items()},
        )
        return {name: output[world_id, 0].numpy() for name, output in self._outputs.items()}

    @staticmethod
    def _colorize(values, channel, depth_range):
        if channel in ("color", "albedo"):
            rgb = np.stack([(values >> shift) & 255 for shift in (0, 8, 16)], axis=-1).astype(np.uint8)
            return rgb, {"normalization": "display/sRGB RGB bytes"}
        if channel in ("depth", "forward_depth"):
            valid = np.isfinite(values) & (values > 0)
            samples = values[valid]
            stats = {"valid_count": int(samples.size), "miss_count": int(values.size - samples.size), "units": "m"}
            stats.update(
                {
                    name: float(function(samples)) if samples.size else None
                    for name, function in (("min", np.min), ("max", np.max), ("mean", np.mean))
                }
            )
            near, far = depth_range if depth_range is not None else (stats["min"] or 0.0, stats["max"] or 1.0)
            gray = np.zeros(values.shape, dtype=np.uint8)
            gray[valid] = (255.0 - 205.0 * np.clip((values[valid] - near) / max(far - near, 1.0e-6), 0, 1)).astype(
                np.uint8
            )
            return np.repeat(gray[..., None], 3, axis=-1), {
                "normalization": "near=255, far=50, misses=0; raw depth in meters",
                "depth_range": [float(near), float(far)],
                "depth_stats": stats,
            }
        if channel == "normal":
            rgb = (np.clip(values * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
            rgb[np.linalg.norm(values, axis=-1) == 0] = 0
            return rgb, {"normalization": "world normal [-1, 1] mapped to RGB [0, 255]; misses black"}
        hashed = ((values.astype(np.uint64) + 1) * 2654435761) & 0xFFFFFF
        rgb = np.stack([(hashed >> shift) & 255 for shift in (16, 8, 0)], axis=-1).astype(np.uint8)
        rgb[values == 0xFFFFFFFF] = 0
        return rgb, {"normalization": "shape ID hash palette; miss=0xFFFFFFFF (black)"}

    def _pick(self, shape_indices, pixels):
        model = self.session.model
        shape_body = model.shape_body.numpy()
        names = getattr(model, "shape_label", None)
        rows = []
        for x, y in pixels:
            index = int(shape_indices[y, x])
            row = {"pixel": [x, y], "shape_id": None if index == 0xFFFFFFFF else index}
            if 0 <= index < model.shape_count:
                row["body_id"] = int(shape_body[index])
                if names is not None:
                    row["label"] = str(names[index])[:256]
            rows.append(row)
        return rows

    def _overlay_contacts(self, rgb, eye, rotation, fov_y, world_id, policy, depth):
        data = self.session.contact_data(refresh=False, limit=256, world=world_id, include_global=True)
        height, width = rgb.shape[:2]
        focal = height / (2 * math.tan(math.radians(fov_y) / 2))
        drawn = 0
        for row in data["rows"]:
            point = (np.asarray(row["surface0"]) + np.asarray(row["surface1"])) * 0.5
            camera = (point - eye) @ rotation
            distance = -camera[2]
            if not np.isfinite(camera).all() or distance <= 0:
                continue
            x = int(math.floor(width / 2 + focal * camera[0] / distance))
            y = int(math.floor(height / 2 - focal * camera[1] / distance))
            if not 0 <= x < width or not 0 <= y < height:
                continue
            if policy == "visible":
                tolerance = max(0.01, distance * 0.005)
                if depth[y, x] > 0 and distance > depth[y, x] + tolerance:
                    continue
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx * dx + dy * dy <= 9 and 0 <= x + dx < width and 0 <= y + dy < height:
                        rgb[y + dy, x + dx] = (255, 32, 224)
            drawn += 1
        return {
            "source": data.get("source", "generated rigid contacts"),
            "considered": len(data["rows"]),
            "total_matched": data.get("total_matched", len(data["rows"])),
            "drawn": drawn,
            "policy": policy,
            "position": "midpoint of world contact surfaces from eval_rigid_contact_kinematics plus normal margins",
            "occlusion_tolerance": "max(0.01 m, 0.005 * forward distance)" if policy == "visible" else None,
        }

    def _render_viewer(self, width, height, fov_y, eye, rotation, world_id, shadows, wireframe):
        from newton.viewer import ViewerGL  # noqa: PLC0415

        viewer = self.session.viewer
        if not isinstance(viewer, ViewerGL):
            raise ValueError("backend='viewer' requires an attached ViewerGL; no GL fallback is created")
        if viewer.model is not self.session.model:
            raise ValueError("Attached ViewerGL must use the session model")
        renderer = viewer.renderer
        source_width, source_height = renderer._screen_width, renderer._screen_height
        if source_width * source_height > self.MAX_PIXELS:
            raise ValueError("Attached ViewerGL framebuffer exceeds the observation pixel budget")
        visible_worlds = None if viewer._visible_worlds is None else set(viewer._visible_worlds)
        camera = viewer.camera
        renderer_camera = getattr(renderer, "camera", None)
        draw_shadows, draw_wireframe = renderer.draw_shadows, renderer.draw_wireframe
        try:
            viewer.set_visible_worlds([world_id])
            offset = np.zeros(3) if viewer.world_offsets is None else viewer.world_offsets.numpy()[world_id]
            capture_camera = self._CaptureCamera(camera, eye + offset, rotation, width, height, fov_y)
            renderer.draw_shadows, renderer.draw_wireframe = shadows, wireframe
            viewer.log_state(self.session.state)
            renderer.render(capture_camera, viewer.objects, viewer.lines, viewer.wireframe_shapes, viewer.arrows)
            rgb = viewer.get_frame().numpy()
        finally:
            renderer.draw_shadows, renderer.draw_wireframe = draw_shadows, draw_wireframe
            viewer.set_visible_worlds(visible_worlds)
            viewer.camera = camera
            renderer.camera = renderer_camera
            viewer.log_state(self.session.state)
        if (source_width, source_height) != (width, height):
            xs = np.minimum(((np.arange(width) + 0.5) * source_width / width).astype(int), source_width - 1)
            ys = np.minimum(((np.arange(height) + 0.5) * source_height / height).astype(int), source_height - 1)
            rgb = rgb[ys[:, None], xs]
        return rgb.copy(), [source_width, source_height]

    def record(self, *, action: str = "status", every_steps: int = 1, max_frames: int = 300, **options) -> dict:
        """Start, stop, or inspect a bounded PNG sequence with simulation timestamps."""
        self._check_thread()
        if action not in ("start", "stop", "status"):
            raise ValueError("record action must be 'start', 'stop' or 'status'")
        if action == "start":
            if self._recording is not None:
                raise ValueError("A recording is already active")
            every_steps = _integer("every_steps", every_steps, 1, 1_000_000)
            max_frames = _integer("max_frames", max_frames, 1, 1000)
            if options.get("raw"):
                raise ValueError("Recording stores PNG frames and metadata; raw NPZ requires an individual observation")
            result = self.observe(**options)
            directory = Path(self.session.artifact_directory) / f"recording-{uuid.uuid4().hex}"
            directory.mkdir(parents=True)
            self._recording = {
                "active": True,
                "directory": str(directory),
                "every_steps": every_steps,
                "max_frames": max_frames,
                "steps": 0,
                "bytes": 0,
                "frames": [],
                "options": options,
            }
            try:
                self._append_frame(result)
            except Exception:
                self._finish_recording("capture_error")
                raise
        elif action == "stop" and self._recording is not None:
            self._finish_recording()
        recording = self._recording or self._last_recording
        return {key: value for key, value in recording.items() if key not in ("frames", "options", "steps")} | {
            "frame_count": len(recording["frames"]) if "frames" in recording else recording.get("frame_count", 0)
        }

    def _append_frame(self, result):
        recording = self._recording
        image = base64.b64decode(result.pop("image_base64"))
        if recording["bytes"] + len(image) > self.MAX_RECORD_BYTES:
            self._finish_recording("byte_limit")
            return
        filename = f"frame-{len(recording['frames']):06d}.png"
        (Path(recording["directory"]) / filename).write_bytes(image)
        recording["bytes"] += len(image)
        recording["frames"].append({"file": filename, **result})
        if len(recording["frames"]) >= recording["max_frames"]:
            self._finish_recording("frame_limit")
        else:
            self._write_manifest(recording)

    @staticmethod
    def _write_manifest(recording):
        path = Path(recording["directory"]) / "manifest.json"
        path.write_text(json.dumps(recording, indent=2), encoding="utf-8")

    def _finish_recording(self, reason="stopped"):
        recording = self._recording
        recording["active"] = False
        recording["stop_reason"] = reason
        self._last_recording = recording
        self._recording = None
        try:
            self._write_manifest(recording)
        except OSError as error:
            recording["manifest_error"] = str(error)[:512]

    def after_step(self):
        """Capture the scheduled frame without letting capture errors interrupt physics."""
        self._check_thread()
        if self._recording is None:
            return
        self._recording["steps"] += 1
        if self._recording["steps"] % self._recording["every_steps"] == 0:
            try:
                self._append_frame(self.observe(**self._recording["options"]))
            except Exception as error:
                self._finish_recording(f"capture_error: {str(error)[:512]}")

    def close(self):
        """Finish recording and release sensor references."""
        self._check_thread()
        if self._recording is not None:
            self._finish_recording("session_closed")
        self.invalidate()
