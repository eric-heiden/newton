# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Top-camera renderer and image score for the ABC station twin.

The verifier uses this exact module (do not edit it): it poses the robot from a
joint-log row, renders the model through the real top camera's calibrated
intrinsics and distortion (camera.json) at 640x480, 2x2 supersampled in linear
light, and scores the render against a recorded frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

HERE = Path(__file__).resolve().parent
# Finger travel [m] per unit of the logged gripper command (0 closed, 1 open).
GRIPPER_TRAVEL = 0.0475
DEFAULT_LOOK = {
    "light_direction": (-0.57735, 0.57735, -0.57735),
    "light_color": (1.0, 1.0, 1.0),
    "shadows": True,
    "ambient_sky": (0.2, 0.2, 0.225),
    "ambient_ground": (0.05, 0.05, 0.06),
    "exposure": 1.0,
}


def joint_positions(q: np.ndarray) -> dict[str, float]:
    """Map one 14-value joint-log row to station joint names.

    Row layout: left joints 1-6 [rad], left gripper [0..1], right joints 1-6 [rad], right gripper [0..1].
    """
    values = {}
    for side, offset in (("left", 0), ("right", 7)):
        for j in range(6):
            values[f"{side}_joint{j + 1}"] = float(q[offset + j])
        travel = float(np.clip(q[offset + 6], 0.0, 1.0)) * GRIPPER_TRAVEL
        values[f"{side}_left_finger"] = travel
        values[f"{side}_right_finger"] = -travel
    return values


def pose_robot(model: newton.Model, state: newton.State, q: np.ndarray) -> None:
    """Set the arm and finger joints from a joint-log row and evaluate forward kinematics into ``state``."""
    values = joint_positions(q)
    joint_q = model.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    found = set()
    for index, label in enumerate(model.joint_label):
        name = label.rsplit("/", 1)[-1]
        if name in values:
            joint_q[starts[index]] = values[name]
            found.add(name)
    missing = sorted(set(values) - found)
    if missing:
        raise ValueError(f"model is missing station joints {missing}")
    model.joint_q.assign(joint_q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)


def camera_rays(camera: dict, supersample: int) -> np.ndarray:
    """Camera-frame ray directions (x right, y up, looking along -z), shape [H*s, W*s, 3].

    Uses RealSense's inverse Brown-Conrady model, which maps distorted pixels directly to rays.
    """
    width, height = camera["width"], camera["height"]
    fx, _, cx, _, fy, cy = camera["K"][:6]
    k1, k2, p1, p2, k3 = camera["D"]
    u = (np.arange(width * supersample) + 0.5) / supersample
    v = (np.arange(height * supersample) + 0.5) / supersample
    x, y = np.meshgrid((u - cx) / fx, (v - cy) / fy)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    ux = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    uy = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    rays = np.stack([ux, -uy, -np.ones_like(ux)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


class TopCamera:
    """Renders a single-world station model through the calibrated top camera.

    Args:
        model: The station model.
        position: Camera position in the world frame [m].
        rotation: Camera orientation quaternion (x, y, z, w); the camera looks along its -Z axis, +Y is image up.
        look: Lighting, all colors linear RGB: ``light_direction`` (world direction the
            directional light travels), ``light_color``, ``shadows`` (bool), ``ambient_sky`` and
            ``ambient_ground`` (hemispheric ambient light on surfaces facing +Z and -Z), and
            ``exposure`` (gain applied before sRGB encoding).
        camera_file: Intrinsics and distortion (camera.json).
        supersample: Rays per pixel along each image axis.
    """

    def __init__(
        self,
        model: newton.Model,
        position,
        rotation,
        look: dict | None = None,
        camera_file: Path = HERE / "camera.json",
        supersample: int = 2,
    ):
        if model.world_count != 1:
            raise ValueError("the station twin renders single-world models")
        look = {**DEFAULT_LOOK, **(look or {})}
        self.camera = json.loads(Path(camera_file).read_text())
        self.width, self.height = self.camera["width"], self.camera["height"]
        self.supersample = supersample
        self.exposure = float(look["exposure"])
        self.model = model
        self.sensor = SensorTiledCamera(model)
        direction = np.asarray(look["light_direction"], dtype=np.float32)
        self.sensor.default_render_config.enable_shadows = bool(look["shadows"])
        self.sensor.utils.create_default_light(
            enable_shadows=bool(look["shadows"]),
            direction=wp.vec3f(*(direction / np.linalg.norm(direction))),
            color=wp.vec3f(*look["light_color"]),
        )
        self.sensor.utils.set_ambient_light(wp.vec3f(*look["ambient_sky"]), wp.vec3f(*look["ambient_ground"]))
        rays = camera_rays(self.camera, supersample).astype(np.float32)
        packed = np.zeros((1, *rays.shape[:2], 2, 3), dtype=np.float32)
        packed[0, :, :, 1] = rays
        self.rays = wp.array(packed, dtype=wp.vec3f, device=model.device)
        self.hdr = self.sensor.utils.create_hdr_color_image_output(
            self.width * supersample, self.height * supersample, camera_count=1
        )
        self.transforms = wp.array(
            [[wp.transformf(wp.vec3f(*position), wp.normalize(wp.quatf(*rotation)))]],
            dtype=wp.transformf,
            device=model.device,
        )

    def render(self, state: newton.State) -> np.ndarray:
        """Render ``state``; returns an sRGB image, shape [480, 640, 3], dtype uint8."""
        self.model.bvh_refit_shapes(state)
        self.sensor.update(state, self.transforms, self.rays, hdr_color_image=self.hdr)
        s = self.supersample
        linear = self.hdr.numpy()[0, 0].reshape(self.height, s, self.width, s, 3).mean(axis=(1, 3))
        return np.round(_linear_to_srgb(linear * self.exposure) * 255.0).astype(np.uint8)


def _pool(image: np.ndarray, factor: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    h, w = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    return image[:h, :w].reshape(h // factor, factor, w // factor, factor, *image.shape[2:]).mean(axis=(1, 3))


def _gray(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)[..., :3] @ np.array([0.299, 0.587, 0.114]) / 255.0


def _box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(x, radius, mode="reflect")
    table = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = 2 * radius + 1
    return (table[n:, n:] - table[:-n, n:] - table[n:, :-n] + table[:-n, :-n]) / (n * n)


def _edges(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return _box_mean(_box_mean(np.hypot(gx, gy), 1), 1)


def score(rendered: np.ndarray, recorded: np.ndarray) -> dict[str, float]:
    """Image agreement of a render with a recorded frame (both [480, 640, 3] uint8).

    Returns ``edge_ncc``, the normalized cross-correlation of smoothed Sobel edge magnitudes
    at quarter resolution (160x120; silhouettes and alignment); ``ssim``, the 7x7 structural
    similarity of luminance at half resolution (320x240); and ``color_psnr_db``, the PSNR of
    RGB at quarter resolution (colors and shading).
    """
    quarter_a, quarter_b = _pool(rendered, 4), _pool(recorded, 4)
    ea, eb = _edges(_gray(quarter_a)), _edges(_gray(quarter_b))
    ea, eb = ea - ea.mean(), eb - eb.mean()
    edge_ncc = float((ea * eb).sum() / (np.sqrt((ea * ea).sum() * (eb * eb).sum()) + 1e-12))
    a, b = _gray(_pool(rendered, 2)), _gray(_pool(recorded, 2))
    mu_a, mu_b = _box_mean(a, 3), _box_mean(b, 3)
    var_a = _box_mean(a * a, 3) - mu_a * mu_a
    var_b = _box_mean(b * b, 3) - mu_b * mu_b
    cov = _box_mean(a * b, 3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    ssim = float(np.mean(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))))
    mse = float(np.mean((quarter_a - quarter_b) ** 2))
    color_psnr_db = float(10.0 * np.log10(255.0**2 / max(mse, 1e-12)))
    return {"edge_ncc": edge_ncc, "ssim": ssim, "color_psnr_db": color_psnr_db}
