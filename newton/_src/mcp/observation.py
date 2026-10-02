# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded camera observations for an application-owned simulation session."""

from __future__ import annotations

import base64
import copy
import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton import ShapeFlags
from newton.sensors import SensorTiledCamera

from .imaging import compare as _compare_images
from .imaging import comparison_panel as _comparison_panel
from .imaging import draw_label, load_image, tile
from .imaging import encode_png as _png
from .imaging import image_metrics as _image_metrics
from .imaging import to_rgb as _rgb


def _integer(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _vector(name: str, value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


# View directions from the target toward the camera in a Z-up frame.
VIEW_PRESETS = {
    "iso": (1.0, -1.0, 0.8),
    "front": (0.0, -1.0, 0.3),
    "back": (0.0, 1.0, 0.3),
    "right": (1.0, 0.0, 0.3),
    "left": (-1.0, 0.0, 0.3),
    "top": (0.0, -0.001, 1.0),
}


def _from_z_up(vector, up_axis: int) -> np.ndarray:
    x, y, z = vector
    return np.asarray({2: (x, y, z), 1: (x, z, -y), 0: (z, x, y)}[up_axis], dtype=np.float64)


def _fit_distance(points, target, forward, right, up, tan_x, tan_y, margin=1.15):
    """Camera distance [m] from ``target`` along ``-forward`` that keeps all ``points`` inside the frustum."""
    rel = points - target
    lateral_x = np.abs(rel @ right) * margin / tan_x
    lateral_y = np.abs(rel @ up) * margin / tan_y
    toward_camera = rel @ -forward
    return float(np.max(toward_camera + np.maximum(lateral_x, lateral_y)))


def _camera_pose(eye, target, up, pose, up_axis: int, frame=None, view=None, fov_y: float = 60.0, aspect=4.0 / 3.0):
    """Return eye, camera rotation, and pose; ``frame`` = (center, radius[, points]) enables auto-framing."""
    if view is not None and view not in VIEW_PRESETS:
        raise ValueError(f"view must be one of {sorted(VIEW_PRESETS)}")
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
    if eye is None and frame is not None:
        center, radius = frame[0], frame[1]
        points = frame[2] if len(frame) > 2 else None
        center = center if target is None else _vector("target", target, 3)
        direction = _from_z_up(VIEW_PRESETS[view or "iso"], up_axis)
        direction /= np.linalg.norm(direction)
        if (view or "iso") == "top" and up is None:
            up = _from_z_up((0.0, 1.0, 0.0), up_axis)
        distance = 1.15 * max(radius, 1.0e-3) / math.sin(math.radians(fov_y) / 2.0)
        if points is not None and len(points) and target is None:
            # Fit the projected extent of the scene rather than its bounding sphere, which
            # leaves elongated scenes (arms, humanoids) small in the frame.
            view_up = up if up is not None else np.eye(3)[up_axis]
            forward = -direction
            right = np.cross(forward, view_up)
            right /= np.linalg.norm(right)
            cam_up = np.cross(right, forward)
            tan_y = math.tan(math.radians(fov_y) / 2.0)
            tan_x = tan_y * aspect
            for _ in range(2):
                distance = _fit_distance(points, center, forward, right, cam_up, tan_x, tan_y)
                depth = distance - (points - center) @ -forward
                x = ((points - center) @ right) / np.maximum(depth, 1.0e-6)
                y = ((points - center) @ cam_up) / np.maximum(depth, 1.0e-6)
                center = (
                    center
                    + right * 0.5 * (x.max() + x.min()) * distance
                    + cam_up * 0.5 * (y.max() + y.min()) * distance
                )
            distance = max(_fit_distance(points, center, forward, right, cam_up, tan_x, tan_y), 1.0e-3)
        eye = center + direction * distance
        target = center
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


_DISTORTION = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4")


def _intrinsics(value, width: int, height: int) -> dict | None:
    """Validate OpenCV pinhole intrinsics; returns keyword arguments for the sensor ray helper."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("intrinsics must be an object with fx, fy, cx, cy")
    unknown = set(value) - {"fx", "fy", "cx", "cy", "image_width", "image_height", "distortion_model", *_DISTORTION}
    if unknown:
        raise ValueError(f"Unknown intrinsics keys: {sorted(unknown)}")
    model = value.get("distortion_model", "opencv")
    if model not in ("opencv", "inverse_brown_conrady"):
        raise ValueError("intrinsics distortion_model must be 'opencv' or 'inverse_brown_conrady'")
    if model == "inverse_brown_conrady" and set(value) & {"k4", "k5", "k6", "s1", "s2", "s3", "s4"}:
        raise ValueError("inverse_brown_conrady takes k1, k2, k3, p1, p2")
    result = {} if model == "opencv" else {"distortion_model": model}
    for key in ("fx", "fy", "cx", "cy", "image_width", "image_height", *_DISTORTION):
        if key not in value:
            continue
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"intrinsics {key} must be a finite number")
        result[key] = float(number)
    missing = {"fx", "fy", "cx", "cy"} - set(result)
    if missing:
        raise ValueError(f"intrinsics require {sorted(missing)}")
    if result["fx"] <= 0.0 or result["fy"] <= 0.0:
        raise ValueError("intrinsics fx and fy must be positive")
    result.setdefault("image_width", float(width))
    result.setdefault("image_height", float(height))
    return result


def _inverse_brown_conrady_rays(width: int, height: int, intrinsics: dict) -> np.ndarray:
    """Camera rays for RealSense's inverse Brown-Conrady model, which maps distorted pixels directly to rays.

    Returns ray origins and directions in the sensor's camera frame (x right, y up, looking along -z),
    shape [1, height, width, 2, 3].
    """
    k = {name: intrinsics.get(name, 0.0) for name in ("k1", "k2", "k3", "p1", "p2")}
    u = (np.arange(width) + 0.5) / width * intrinsics["image_width"]
    v = (np.arange(height) + 0.5) / height * intrinsics["image_height"]
    x, y = np.meshgrid((u - intrinsics["cx"]) / intrinsics["fx"], (v - intrinsics["cy"]) / intrinsics["fy"])
    r2 = x * x + y * y
    radial = 1.0 + k["k1"] * r2 + k["k2"] * r2 * r2 + k["k3"] * r2 * r2 * r2
    ux = x * radial + 2.0 * k["p1"] * x * y + k["p2"] * (r2 + 2.0 * x * x)
    uy = y * radial + 2.0 * k["p2"] * x * y + k["p1"] * (r2 + 2.0 * y * y)
    directions = np.stack([ux, -uy, -np.ones_like(ux)], axis=-1)
    rays = np.zeros((1, height, width, 2, 3), dtype=np.float32)
    rays[0, :, :, 1] = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    return rays


def _distort_opencv(x: np.ndarray, y: np.ndarray, k: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply OpenCV's rational, tangential, and thin-prism distortion to normalized coordinates (y down)."""

    def radial(s):
        return (1.0 + k["k1"] * s + k["k2"] * s * s + k["k3"] * s**3) / (
            1.0 + k["k4"] * s + k["k5"] * s * s + k["k6"] * s**3
        )

    r2 = x * x + y * y
    xd = x * radial(r2) + 2.0 * k["p1"] * x * y + k["p2"] * (r2 + 2.0 * x * x) + k["s1"] * r2 + k["s2"] * r2 * r2
    yd = y * radial(r2) + k["p1"] * (r2 + 2.0 * y * y) + 2.0 * k["p2"] * x * y + k["s3"] * r2 + k["s4"] * r2 * r2
    # Past the radius where r * radial(r^2) stops growing, the polynomial folds points from outside
    # the calibrated field of view back into the image.
    radii = np.sqrt(np.nan_to_num(r2, nan=0.0))[:, None] * np.linspace(0.0, 1.0, 65)[None, :]
    monotonic = (np.diff(radii * radial(radii * radii), axis=1) > 0.0).all(axis=1) | (r2 == 0.0)
    return np.where(monotonic, xd, np.nan), np.where(monotonic, yd, np.nan)


def _distort_inverse_brown_conrady(x: np.ndarray, y: np.ndarray, k: dict) -> tuple[np.ndarray, np.ndarray]:
    """Distorted coordinates that the inverse Brown-Conrady map of :func:`_inverse_brown_conrady_rays` sends to (x, y)."""

    def undistort(xd, yd):
        r2 = xd * xd + yd * yd
        radial = 1.0 + k["k1"] * r2 + k["k2"] * r2 * r2 + k["k3"] * r2**3
        return (
            xd * radial + 2.0 * k["p1"] * xd * yd + k["p2"] * (r2 + 2.0 * xd * xd),
            yd * radial + 2.0 * k["p2"] * xd * yd + k["p1"] * (r2 + 2.0 * yd * yd),
            radial,
        )

    xd, yd = x.copy(), y.copy()
    for _ in range(100):
        ux, uy, radial = undistort(xd, yd)
        xd, yd = xd + (x - ux) / radial, yd + (y - uy) / radial
    ux, uy, _ = undistort(xd, yd)
    converged = np.hypot(ux - x, uy - y) <= 1.0e-9 * np.maximum(1.0, np.hypot(x, y))
    return np.where(converged, xd, np.nan), np.where(converged, yd, np.nan)


def _project(points, pose, width: int, height: int, fov_y: float, intrinsics: dict | None):
    """Image coordinates [px] and forward depth [m] of world points seen by an observation camera.

    Coordinates match the renderer's rays: x right, y down, and pixel ``i`` spans ``[i, i + 1)``.
    Points behind the camera or outside a distortion model's valid range are NaN.
    """
    pose = np.asarray(pose, dtype=np.float64)
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:7])), dtype=np.float64).reshape(3, 3)
    local = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - pose[:3]) @ rotation
    depth = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ahead = depth > 1.0e-9
        x = np.where(ahead, local[:, 0] / np.where(ahead, depth, 1.0), np.nan)
        y = np.where(ahead, -local[:, 1] / np.where(ahead, depth, 1.0), np.nan)
        if intrinsics is None:
            focal = height / (2.0 * math.tan(math.radians(fov_y) / 2.0))
            u, v = 0.5 * width + focal * x, 0.5 * height + focal * y
        else:
            k = {name: intrinsics.get(name, 0.0) for name in _DISTORTION}
            if intrinsics.get("distortion_model") == "inverse_brown_conrady":
                x, y = _distort_inverse_brown_conrady(x, y, k)
            else:
                x, y = _distort_opencv(x, y, k)
            u = (intrinsics["fx"] * x + intrinsics["cx"]) * width / intrinsics["image_width"]
            v = (intrinsics["fy"] * y + intrinsics["cy"]) * height / intrinsics["image_height"]
    return np.stack([u, v], axis=-1), depth


def _camera_directions(width: int, height: int, intrinsics: dict, device) -> np.ndarray:
    """Camera-frame ray directions [H, W, 3] that the sensor backend traces for calibrated ``intrinsics``.

    Directions use the sensor convention (x right, y up, looking along -z); pixels without a valid ray are zero.
    """
    if intrinsics.get("distortion_model") == "inverse_brown_conrady":
        return _inverse_brown_conrady_rays(width, height, intrinsics)[0, :, :, 1]
    from ..sensors.warp_raytrace.camera_utils import compute_camera_rays_pinhole_opencv_kernel  # noqa: PLC0415

    rays = wp.zeros((1, height, width, 2), dtype=wp.vec3f, device=device)
    calibration = [intrinsics[key] for key in ("image_width", "image_height", "fx", "fy", "cx", "cy")]
    coefficients = [intrinsics.get(key, 0.0) for key in _DISTORTION]
    wp.launch(
        compute_camera_rays_pinhole_opencv_kernel,
        dim=(height, width),
        inputs=[width, height, *calibration, *coefficients, 0, rays],
        device=device,
    )
    return rays.numpy()[0, :, :, 1]


def _is_square_pinhole(intrinsics: dict, width: int, height: int) -> bool:
    """Whether calibrated intrinsics, scaled to the output size, are an undistorted pinhole with square pixels."""
    if any(intrinsics.get(key, 0.0) for key in _DISTORTION):
        return False
    scale_x, scale_y = width / intrinsics["image_width"], height / intrinsics["image_height"]
    return math.isclose(intrinsics["fx"], intrinsics["fy"], rel_tol=1.0e-6) and math.isclose(
        scale_x, scale_y, rel_tol=1.0e-6
    )


def _pinhole_cover(width: int, height: int, intrinsics: dict, device, max_pixels: int) -> tuple[dict, np.ndarray]:
    """Square-pixel pinhole intrinsics whose image covers every ray of a calibrated camera, and where each ray lands.

    Renderers without lens models draw such a camera by rendering the pinhole and sampling it at the returned
    continuous pixel coordinates [H, W, 2] (pixel ``i`` spans ``[i, i + 1)``; NaN where the camera has no ray).
    The pinhole keeps the camera's pixel density at its principal point unless that exceeds ``max_pixels``.
    """
    directions = _camera_directions(width, height, intrinsics, device).astype(np.float64)
    ahead = directions[..., 2] < -1.0e-9
    if not ahead.any():
        raise ValueError("intrinsics give no valid camera rays")
    depth = np.where(ahead, -directions[..., 2], 1.0)
    x = np.where(ahead, directions[..., 0] / depth, np.nan)
    y = np.where(ahead, -directions[..., 1] / depth, np.nan)
    lower = np.array([np.nanmin(x), np.nanmin(y)])
    extent = np.array([np.nanmax(x), np.nanmax(y)]) - lower
    focal = native = max(
        intrinsics["fx"] * width / intrinsics["image_width"], intrinsics["fy"] * height / intrinsics["image_height"]
    )
    # One pixel of margin keeps bilinear samples inside the image.
    size = np.ceil(extent * focal + 2.0).astype(int)
    while size.prod() > max_pixels:
        focal *= 0.99 * math.sqrt(max_pixels / float(size.prod()))
        size = np.ceil(extent * focal + 2.0).astype(int)
    if focal < 0.5 * native:
        raise ValueError("The camera's rays span too wide a view to render through a pinhole; use backend='sensor'")
    cx, cy = (float(value) for value in 1.0 - lower * focal)
    pinhole = {
        "fx": focal,
        "fy": focal,
        "cx": cx,
        "cy": cy,
        "image_width": float(size[0]),
        "image_height": float(size[1]),
    }
    return pinhole, np.stack([x * focal + cx, y * focal + cy], axis=-1)


def _sample_bilinear(image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Bilinearly sample an RGB image at continuous pixel coordinates [H, W, 2]; NaN coordinates give black."""
    height, width = image.shape[:2]
    valid = np.isfinite(coordinates).all(axis=-1)
    u = np.clip(np.nan_to_num(coordinates[..., 0]) - 0.5, 0.0, width - 1.0)
    v = np.clip(np.nan_to_num(coordinates[..., 1]) - 0.5, 0.0, height - 1.0)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = np.minimum(u0 + 1, width - 1), np.minimum(v0 + 1, height - 1)
    du, dv = (u - u0)[..., None], (v - v0)[..., None]
    pixels = image.astype(np.float32)
    top = pixels[v0, u0] * (1.0 - du) + pixels[v0, u1] * du
    bottom = pixels[v1, u0] * (1.0 - du) + pixels[v1, u1] * du
    result = top * (1.0 - dv) + bottom * dv
    result[~valid] = 0.0
    return np.clip(result + 0.5, 0.0, 255.0).astype(np.uint8)


def _body_index(model, selector, world_id: int) -> int:
    """Index of the body selected by label or index within ``world_id`` (or among global bodies).

    Labels match exactly, then by their last path component, then as a substring; worlds
    replicated from one builder share labels, so the observed world disambiguates them.
    """
    if isinstance(selector, bool) or not isinstance(selector, (str, int, np.integer)):
        raise ValueError("Bodies are selected by label or index")
    if not model.body_count:
        raise ValueError("The model has no bodies")
    worlds = model.body_world.numpy() if model.body_world is not None else np.full(model.body_count, -1)
    if not isinstance(selector, str):
        index = _integer("body index", selector, 0, model.body_count - 1)
        if worlds[index] not in (world_id, -1):
            raise ValueError(f"Body {index} belongs to world {worlds[index]}, not world_id {world_id}")
        return index
    labels = [str(label) for label in (model.body_label or [])]
    candidates = [i for i in range(min(len(labels), model.body_count)) if worlds[i] in (world_id, -1)]
    for matches in (
        lambda label: label == selector,
        lambda label: label.rsplit("/", 1)[-1] == selector,
        lambda label: selector in label,
    ):
        found = [i for i in candidates if matches(labels[i])]
        if len(found) == 1:
            return found[0]
        if found:
            names = ", ".join(labels[i] for i in found[:8])
            raise ValueError(f"Body {selector!r} matches {len(found)} bodies in world {world_id}: {names}")
    raise ValueError(f"No body labeled {selector!r} in world {world_id}")


_MARKER_COLORS = (
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 128, 0),
    (80, 140, 255),
    (255, 255, 255),
    (255, 40, 40),
)


def _draw_markers(image: np.ndarray, markers: list, scale: int = 1) -> np.ndarray:
    """Copy of ``image`` with a labeled ring per projected point; ``scale`` divides the pixel coordinates."""
    image = image.copy()
    height, width = image.shape[:2]
    radius = max(4.0, min(width, height) / 40.0)
    text_scale = 2 if min(width, height) >= 480 else 1
    for index, (name, pixels, single) in enumerate(markers):
        color = _MARKER_COLORS[index % len(_MARKER_COLORS)]
        for point, (u, v) in enumerate(pixels / scale):
            if not (np.isfinite(u) and np.isfinite(v)) or not (0 <= u < width and 0 <= v < height):
                continue
            x0, x1 = max(int(u - radius - 3), 0), min(int(u + radius + 3), width)
            y0, y1 = max(int(v - radius - 3), 0), min(int(v + radius + 3), height)
            ys, xs = np.mgrid[y0:y1, x0:x1]
            distance = np.abs(np.hypot(xs + 0.5 - u, ys + 0.5 - v) - radius)
            region = image[y0:y1, x0:x1]
            region[distance <= 2.0] = 0
            region[distance <= 1.0] = color
            image[int(v), int(u)] = color
            text = name if single else f"{name}[{point}]"
            text_width = (len(text) * 6 + 2) * text_scale
            left = u + radius + 3 if u + radius + 3 + text_width <= width else u - radius - 3 - text_width
            top = min(max(v - 4.5 * text_scale, 0), height - 9 * text_scale)
            draw_label(image, text, int(left), int(top), text_scale, color)
    return image


def _marker_report(markers: list) -> dict:
    """Pixel coordinates [x, y] per overlay name (a list for point arrays), None where a point is not projectable."""
    report = {}
    for name, pixels, single in markers:
        coordinates = [
            [round(float(u), 1), round(float(v), 1)] if np.isfinite([u, v]).all() else None for u, v in pixels
        ]
        report[name] = coordinates[0] if single else coordinates
    return report


def _view_name(spec: dict, index: int) -> str:
    """Caption for a camera specification: its label, preset, mounting body, or list position."""
    if isinstance(spec.get("label"), str):
        return spec["label"]
    if spec.get("view"):
        return spec["view"]
    if spec.get("camera_body") is not None:
        return str(spec["camera_body"])
    return "auto" if spec.get("eye") is None and spec.get("pose") is None else f"view {index}"


def _reference_rows(references, views: int) -> list[list]:
    """Normalize filmstrip references to one list of images (paths or arrays) per view."""
    if isinstance(references, (str, Path)) and Path(references).is_dir():
        files = sorted(
            p for p in Path(references).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        )
        if not files:
            raise ValueError(f"No images in {references}")
        return [[str(f) for f in files]]
    if isinstance(references, np.ndarray):
        if references.ndim != 4:
            raise ValueError("Reference arrays must have shape (N, H, W, 3)")
        return [list(references)]
    if not isinstance(references, list) or not references:
        raise ValueError("references must be a list, an (N, H, W, 3) array, or a directory")
    if views == 1 and not (
        isinstance(references[0], list) or (isinstance(references[0], np.ndarray) and references[0].ndim == 4)
    ):
        return [list(references)]
    return [_reference_rows(row, 1)[0] if not isinstance(row, list) else row for row in references]


def _mask(value, shape) -> np.ndarray:
    """Boolean pixel mask from an array or image path (nonzero pixels selected)."""
    array = load_image(value).max(axis=-1) if isinstance(value, (str, Path)) else np.asarray(value)
    if array.shape[:2] != tuple(shape):
        raise ValueError(f"mask shape {array.shape[:2]} differs from the frame {tuple(shape)}")
    return array.astype(bool)


def _page_layout(
    row_heights: list[int], width: int, columns: int, edge: int, max_pages: int, gap: int = 4, band_gap: int = 12
) -> tuple[int, int, int]:
    """Shrink factor, columns per band, and bands per page for grid pages that fit ``edge`` x ``edge`` pixels.

    Takes the smallest shrink factor whose layout needs at most ``max_pages`` pages, shrinking frames to no less
    than 48 px wide.
    """
    scale = 1
    while True:
        column = -(-width // scale) + gap
        band = sum(-(-height // scale) for height in row_heights) + gap * (len(row_heights) - 1) + band_gap
        if column <= edge + gap and band <= edge + band_gap:
            per_band = min(columns, max(1, (edge + gap) // column))
            bands = max(1, (edge + band_gap) // band)
            bands = min(bands, -(-columns // per_band))
            # Stop before frames get narrower than 48 px, even if that needs more pages.
            if -(-columns // (per_band * bands)) <= max_pages or -(-width // (scale + 1)) < min(48, width):
                return scale, per_band, bands
        scale += 1


def _stack_bands(bands: list[np.ndarray], gap: int = 12) -> np.ndarray:
    """Stack band images vertically on white, left-aligned."""
    width = max(band.shape[1] for band in bands)
    page = np.full((sum(band.shape[0] for band in bands) + gap * (len(bands) - 1), width, 3), 255, dtype=np.uint8)
    top = 0
    for band in bands:
        page[top : top + band.shape[0], : band.shape[1]] = band
        top += band.shape[0] + gap
    return page


def _shrink(image: np.ndarray, factor: int) -> np.ndarray:
    h, w = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    return image[:h, :w].reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3)).astype(np.uint8)


def _checker_cell(scene_radius: float) -> float:
    """A round checker cell size [m] of roughly a quarter of the scene radius."""
    for cell in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        if cell >= 0.25 * scene_radius:
            return cell
    return 10.0


@wp.func
def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return wp.pow((c + 0.055) / 1.055, 2.4)


@wp.func
def _linear_to_srgb(c: float) -> float:
    c = wp.clamp(c, 0.0, 1.0)
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * wp.pow(c, 1.0 / 2.4) - 0.055


@wp.kernel(enable_backward=False)
def _finish_color(
    color: wp.array2d[wp.uint32],
    shape_index: wp.array2d[wp.uint32],
    depth: wp.array2d[wp.float32],
    rays: wp.array3d[wp.vec3f],
    rotation: wp.mat33f,
    eye: wp.vec3f,
    up: wp.vec3f,
    plane_inverse: wp.array[wp.transformf],
    plane_flag: wp.array[wp.int32],
    cell: float,
    environment: int,
    factor: int,
    out: wp.array2d[wp.uint32],
):
    """Sky for misses, checker on planes, then box-average ``factor`` x ``factor`` samples in linear light."""
    y, x = wp.tid()
    total = wp.vec3f(0.0)
    for sy in range(factor):
        for sx in range(factor):
            py = y * factor + sy
            px = x * factor + sx
            packed = color[py, px]
            c = wp.vec3f(
                _srgb_to_linear(float(packed & wp.uint32(255)) / 255.0),
                _srgb_to_linear(float((packed >> wp.uint32(8)) & wp.uint32(255)) / 255.0),
                _srgb_to_linear(float((packed >> wp.uint32(16)) & wp.uint32(255)) / 255.0),
            )
            if environment != 0:
                direction = wp.normalize(rotation * rays[py, px, 1])
                shape = shape_index[py, px]
                if shape == wp.uint32(0xFFFFFFFF):
                    elevation = wp.dot(direction, up)
                    horizon = wp.vec3f(0.62, 0.68, 0.76)
                    if elevation >= 0.0:
                        c = horizon + (wp.vec3f(0.22, 0.38, 0.66) - horizon) * wp.sqrt(wp.min(elevation, 1.0))
                    else:
                        c = horizon + (wp.vec3f(0.30, 0.29, 0.28) - horizon) * wp.min(-elevation * 4.0, 1.0)
                elif shape < wp.uint32(plane_flag.shape[0]):
                    # Unsigned compare: the sentinel ids of particles, cloth, and overlays (0xFFFFFFFx)
                    # are negative as int and must not index the shape arrays.
                    if plane_flag[int(shape)] != 0:
                        local = wp.transform_point(plane_inverse[int(shape)], eye + direction * depth[py, px])
                        parity = int(wp.floor(local[0] / cell) + wp.floor(local[1] / cell)) % 2
                        if parity != 0:
                            c = c * 0.78
            total = total + c
    total = total / float(factor * factor)
    r = wp.uint32(wp.round(_linear_to_srgb(total[0]) * 255.0))
    g = wp.uint32(wp.round(_linear_to_srgb(total[1]) * 255.0))
    b = wp.uint32(wp.round(_linear_to_srgb(total[2]) * 255.0))
    out[y, x] = r | (g << wp.uint32(8)) | (b << wp.uint32(16)) | wp.uint32(0xFF000000)


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
    # Clients downscale images to about this long edge (Claude: 1568 px), so filmstrip pages are laid out to fit it.
    DISPLAY_EDGE = 1568
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
        self._visibility_signature = None
        self._textured_model = None
        self._textured = False
        self._buffer_key = None
        self._rays = None
        self._transforms = None
        self._outputs = {}
        self._recording = None
        self._last_recording = {"active": False, "frame_count": 0}
        self._rtx = self._rtx_model = self._rtx_colors = self._rtx_world = None
        self._blender = self._blender_signature = None
        self._pinhole_cover = self._pinhole_cover_key = None

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Observations must run on the simulation/viewer owner thread")

    def invalidate(self):
        """Discard cached geometry and buffers after model edits or replacement."""
        self._check_thread()
        self._sensor = self._sensor_model = self._buffer_key = self._rays = self._transforms = None
        self._outputs = {}

    def _single(
        self,
        *,
        view: str | None = None,
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
        antialias: bool = True,
        environment: bool = True,
        intrinsics: dict | None = None,
        samples: int = 16,
        camera_body: str | int | None = None,
        camera_offset=None,
    ) -> dict:
        """Return a PNG and bounded metadata for the current simulation state.

        Depth channels retain meters in raw NPZ artifacts. PNG depth maps the
        selected near/far range to grayscale 255/50, with misses black; omitted
        ranges use valid per-frame extrema. Normal RGB maps world components
        [-1, 1] to [0, 255], with misses black. Shape IDs use a deterministic
        hash palette and retain uint32 IDs in artifacts (0xFFFFFFFF is a miss).
        Color and albedo are display/sRGB. Contact markers use the midpoint of
        world contact surfaces, including margins, with optional depth occlusion.
        ``intrinsics`` renders a calibrated OpenCV camera instead of ``fov_y``:
        ``fx``, ``fy``, ``cx``, ``cy`` [px] for an image of ``image_width`` x
        ``image_height`` (default: the output size, scaled to it otherwise),
        plus optional distortion ``k1``-``k6``, ``p1``, ``p2``, ``s1``-``s4``.
        ``distortion_model='inverse_brown_conrady'`` (RealSense cameras) reads
        ``k1``, ``k2``, ``k3``, ``p1``, ``p2`` as a distorted-to-undistorted map.
        The blender backends render distorted or non-square-pixel cameras as a
        pinhole covering their rays and resample it through those rays.
        Color images are supersampled (``antialias``) when the pixel budget
        allows, and ``environment`` adds a sky gradient behind the scene and a
        checker of known cell size on ground planes for scale and motion cues.
        ``camera_body`` (a body label or index, matched within ``world_id``)
        mounts the camera on that body, e.g. a wrist camera: ``camera_offset``
        is the camera pose in the body frame (position [m] and xyzw quaternion,
        default identity), and the body's pose at render time places it. A ROS
        optical frame (+Z forward, +Y down) needs ``camera_offset=[0, 0, 0, 1, 0, 0, 0]``.
        """
        self._check_thread()
        model = self.session.model
        width = _integer("width", width, 1, 2048)
        height = _integer("height", height, 1, 2048)
        world_id = _integer("world_id", world_id, 0, model.world_count - 1)
        mount = None
        if camera_body is not None:
            if any(value is not None for value in (eye, target, up, pose, view)):
                raise ValueError("camera_body places the camera; give camera_offset instead of eye/target/up/pose/view")
            body = _body_index(model, camera_body, world_id)
            if self.session.state.body_q is None:
                raise ValueError("camera_body needs body transforms in the state")
            offset = _vector("camera_offset", (0, 0, 0, 0, 0, 0, 1) if camera_offset is None else camera_offset, 7)
            if np.linalg.norm(offset[3:]) < 1.0e-12:
                raise ValueError("camera_offset quaternion must be nonzero")
            offset = np.concatenate([offset[:3], offset[3:] / np.linalg.norm(offset[3:])])
            body_pose = self.session.state.body_q.numpy()[body]
            pose = list(wp.transform_multiply(wp.transform(*body_pose), wp.transform(*offset)))
            labels = getattr(model, "body_label", None) or []
            mount = {
                "body": body,
                **({"label": str(labels[body])[:256]} if body < len(labels) else {}),
                "offset": [round(float(v), 6) for v in offset],
            }
        elif camera_offset is not None:
            raise ValueError("camera_offset requires camera_body")
        if backend not in ("sensor", "viewer", "rtx", "blender", "blender_cycles"):
            raise ValueError("backend must be 'sensor', 'viewer', 'rtx', 'blender', or 'blender_cycles'")
        samples = _integer("samples", samples, 1, 256)
        if channel not in self.CHANNELS:
            raise ValueError(f"channel must be one of {self.CHANNELS}")
        for name, value in (
            ("shadows", shadows),
            ("wireframe", wireframe),
            ("contacts", contacts),
            ("raw", raw),
            ("antialias", antialias),
            ("environment", environment),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if textures is not None and not isinstance(textures, bool):
            raise ValueError("textures must be a boolean or null")
        if not isinstance(fov_y, (int, float)) or isinstance(fov_y, bool) or not 1.0 <= fov_y <= 175.0:
            raise ValueError("fov_y must be finite and in [1, 175] degrees")
        intrinsics = _intrinsics(intrinsics, width, height)
        if intrinsics is not None:
            if backend not in ("sensor", "blender", "blender_cycles"):
                raise ValueError("intrinsics require the sensor or blender backend")
            # Framing and overlays use the equivalent vertical field of view.
            fov_y = math.degrees(2.0 * math.atan(0.5 * intrinsics["image_height"] / intrinsics["fy"]))
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
        # The sensor renders only the observed world, so the budget no longer scales with the world count.
        aggregate_pixels = width * height
        if aggregate_pixels > self.MAX_PIXELS:
            raise ValueError(
                f"Observation needs {aggregate_pixels} pixels (width x height); "
                f"limit is {self.MAX_PIXELS}. Reduce resolution."
            )
        frame = self._scene_frame(world_id) if pose is None and eye is None else None
        eye, rotation, camera_pose = _camera_pose(
            eye, target, up, pose, int(model.up_axis), frame=frame, view=view, fov_y=float(fov_y), aspect=width / height
        )
        metadata = {
            "backend": backend,
            "channel": channel,
            "width": width,
            "height": height,
            "world_id": world_id,
            "camera": {
                "pose": camera_pose,
                "fov_y": float(fov_y),
                **({"intrinsics": intrinsics} if intrinsics is not None else {}),
                **({"mount": mount} if mount is not None else {}),
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
            # Supersample color 2x2 when the budget allows; other channels stay exact per pixel.
            supersample = 2 if antialias and channel == "color" and 4 * aggregate_pixels <= self.MAX_PIXELS else 1
            arrays, environment_metadata = self._render_sensor(
                width,
                height,
                fov_y,
                camera_pose,
                world_id,
                channel,
                shadows,
                self._has_textures() if textures is None else textures,
                contacts,
                pick,
                supersample=supersample,
                intrinsics=intrinsics,
                environment=environment and channel == "color",
            )
            metadata["settings"].update(supersample=supersample, **environment_metadata)
            rgb, channel_metadata = self._colorize(arrays[channel], channel, depth_range)
            metadata.update(channel_metadata)
            if pick is not None:
                metadata["picks"] = self._pick(arrays["shape_index"], pick)
            depth = arrays.get("forward_depth")
        elif backend.startswith("blender"):
            if channel != "color" or raw or pick is not None or wireframe or contacts:
                raise ValueError("The blender backend renders color only; channels, picking, contacts need sensor")
            started = time.perf_counter()
            engine = "CYCLES" if backend == "blender_cycles" else "EEVEE"
            rgb, blender_metadata = self._render_blender(
                width, height, fov_y, camera_pose, world_id, samples, intrinsics, engine
            )
            metadata.update(blender_metadata)
            metadata["render_seconds"] = round(time.perf_counter() - started, 3)
            depth = None
            arrays = {}
        elif backend == "rtx":
            if channel != "color" or raw or pick is not None or wireframe:
                raise ValueError("The rtx backend renders color only; raw channels, picking and wireframe need sensor")
            if contacts and contact_depth != "always":
                raise ValueError("rtx contact overlays require contact_depth='always'; sensor supports occlusion")
            started = time.perf_counter()
            rgb, rtx_metadata = self._render_rtx(width, height, fov_y, camera_pose, world_id, samples)
            metadata.update(rtx_metadata)
            metadata["render_seconds"] = round(time.perf_counter() - started, 3)
            depth = None
            arrays = {}
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
            metadata["contacts"] = self._overlay_contacts(rgb, metadata["camera"], world_id, contact_depth, depth)
        if raw:
            directory = Path(self.session.artifact_directory)
            directory.mkdir(parents=True, exist_ok=True)
            artifact = directory / f"observation-{uuid.uuid4().hex}.npz"
            np.savez_compressed(artifact, **arrays, camera_pose=camera_pose, fov_y=fov_y)
            metadata["raw_artifact"] = str(artifact)
        if frame is not None:
            metadata["camera"]["auto_framed"] = {
                "view": view or "iso",
                "target": [round(float(v), 4) for v in frame[0]],
            }
        return rgb, metadata

    def observe(self, *, views=None, reference=None, label=None, comparison="mismatch", **options) -> dict:
        """Render one camera, a labeled multi-view grid, or a comparison against a reference image.

        Omitting ``eye``/``target``/``pose`` frames the current scene
        automatically from the ``view`` preset (default ``"iso"``). ``views`` is
        a list of preset names or per-view camera dictionaries sharing the other
        options. ``reference`` is an image path (a list aligned with ``views``)
        taken with the same camera; the result adds the reference and a
        mismatch panel (magenta = pixels differing by more than 24/255) plus
        pixel statistics and PSNR, SSIM, and edge NCC. ``comparison="edges"``
        (simulated edges magenta, reference edges green, overlap white) or
        ``"blend"`` suits real photos better than pixel mismatch. Width/height default to the reference size.

        ``overlay`` maps names to simulated world points [m] to mark, each a
        Python expression (needs ``allow_execute``) or callable returning a
        point or an ``(N, 3)`` array, ``{"body": label or index, "point": [x, y, z]}``
        (a point in the body frame, default its origin, resolved in ``world_id``),
        or fixed coordinates. The points are projected through the same camera,
        including ``intrinsics`` and distortion, and drawn as labeled rings on
        the simulated and reference images after the comparison metrics are
        computed; ``overlay`` in the result holds their pixel coordinates.
        """
        self._check_thread()
        if views is None:
            if isinstance(reference, list):
                raise ValueError("A reference list requires views; use one reference path for a single view")
            spec = dict(options)
            if isinstance(label, str):
                spec["label"] = label
            rows, labels, metadata = self._rows([spec], [reference], bool(label) or reference is not None, comparison)
            single = metadata["views"][0]
            image = rows[0][0] if reference is None and not label else tile(rows, labels)
            single["image_base64"] = base64.b64encode(_png(image)).decode("ascii")
            single["mime_type"] = "image/png"
            if reference is not None:
                single["layout"] = f"simulated | reference | {comparison}"
            return single
        if not isinstance(views, list) or not 1 <= len(views) <= 16:
            raise ValueError("views must be a list of 1 to 16 presets or camera dictionaries")
        specs = []
        for item in views:
            spec = dict(options)
            if isinstance(item, str):
                spec.update(view=item, eye=None, target=None, pose=None)
            elif isinstance(item, dict):
                spec.update(item)
            else:
                raise ValueError("Each view must be a preset name or a camera dictionary")
            specs.append(spec)
        references = reference if isinstance(reference, list) else [reference] * len(specs)
        if reference is not None and (not isinstance(reference, list) or len(reference) != len(specs)):
            raise ValueError("reference must be a list aligned with views")
        rows, labels, metadata = self._rows(specs, references, label is None or bool(label), comparison)
        if reference is None:
            # Without comparison panels, arrange single views in a near-square grid instead of a tall strip.
            columns = math.ceil(math.sqrt(len(rows)))
            cells = [row[0] for row in rows]
            names = [row[0] for row in labels]
            rows = [cells[i : i + columns] for i in range(0, len(cells), columns)]
            labels = [names[i : i + columns] for i in range(0, len(names), columns)]
            metadata["layout"] = f"views in row-major order, {columns} per row"
        else:
            metadata["layout"] = f"one row per view: simulated | reference | {comparison}"
        metadata["image_base64"] = base64.b64encode(_png(tile(rows, labels))).decode("ascii")
        metadata["mime_type"] = "image/png"
        return metadata

    def _rows(self, specs, references, label, comparison="mismatch"):
        rows, labels, views = [], [], []
        total_pixels = 0
        for index, (view_spec, reference) in enumerate(zip(specs, references, strict=True)):
            spec = dict(view_spec)
            name = spec.pop("label", None)
            name = name if isinstance(name, str) else None
            overlay = spec.pop("overlay", None)
            reference_rgb = None
            if reference is not None:
                if not isinstance(reference, str):
                    raise ValueError("reference must be an image file path")
                reference_rgb = load_image(reference)
                spec.setdefault("height", reference_rgb.shape[0])
                spec.setdefault("width", reference_rgb.shape[1])
            rgb, metadata = self._single(**spec)
            total_pixels += rgb.shape[0] * rgb.shape[1] * (1 if reference_rgb is None else 3)
            if total_pixels > self.MAX_PIXELS:
                raise ValueError(f"Combined image exceeds {self.MAX_PIXELS} pixels; reduce width/height or views")
            name = name or _view_name(spec, index)
            markers = self._overlay_markers(overlay, metadata) if overlay is not None else None
            row, row_labels = [rgb], [f"{name} t={self.session.time:.3f}" if label else ""]
            if reference_rgb is not None:
                stats = _compare_images(rgb, reference_rgb)[1]
                panel = _comparison_panel(rgb, reference_rgb, comparison)
                caption = {
                    "mismatch": f"mismatch {100 * stats['mismatch_fraction']:.1f}%",
                    "edges": f"edges ncc {stats['edge_ncc']:.2f}",
                    "blend": f"blend ssim {stats['ssim']:.2f}",
                }[comparison]
                row += [reference_rgb, panel]
                row_labels += ["reference", caption] if label else ["", ""]
                metadata["reference"] = {"path": reference, **stats}
            if markers is not None:
                # Markers go on after the metrics so they never affect the scores.
                row[: min(len(row), 2)] = [_draw_markers(image, markers) for image in row[:2]]
                metadata["overlay"] = _marker_report(markers)
            rows.append(row)
            labels.append(row_labels)
            views.append(metadata)
        summary = {
            "time": float(self.session.time),
            "frame": int(getattr(self.session, "frame", 0)),
            "revision": int(getattr(self.session, "revision", 0)),
            "views": views,
        }
        return rows, labels, summary

    MAX_OVERLAY_POINTS = 256

    def _check_overlay(self, overlay) -> None:
        """Validate the overlay mapping before any stepping."""
        if not isinstance(overlay, dict) or not 1 <= len(overlay) <= 16:
            raise ValueError("overlay must map 1 to 16 names to points")
        for name, source in overlay.items():
            if not isinstance(name, str) or not name:
                raise ValueError("overlay names must be nonempty strings")
            # Expressions are Python, so they follow the same permission as execute.
            if isinstance(source, str) and not getattr(self.session, "allow_execute", False):
                raise ValueError("overlay expressions need allow_execute; give {'body': ...} or coordinates instead")

    def _overlay_points(self, overlay, world_id: int) -> list:
        """Evaluate overlay sources on the current state as ``(name, world points [m] (N, 3), single)``."""
        self._check_overlay(overlay)
        session = self.session
        body_q = None
        result = []
        for name, source in overlay.items():
            if isinstance(source, dict):
                if "body" not in source or set(source) - {"body", "point"}:
                    raise ValueError(
                        "overlay bodies are {'body': label or index, 'point': [x, y, z] in the body frame}"
                    )
                if body_q is None:
                    if session.state.body_q is None:
                        raise ValueError("overlay bodies need body transforms in the state")
                    body_q = session.state.body_q.numpy()
                body = _body_index(session.model, source["body"], world_id)
                local = _vector("overlay point", source.get("point", (0.0, 0.0, 0.0)), 3)
                value = wp.transform_point(wp.transform(*body_q[body]), wp.vec3(*local))
            elif isinstance(source, str):
                scope = getattr(session, "_eval_scope", None)
                scope = (
                    scope()
                    if callable(scope)
                    else {"session": session, "model": session.model, "state": session.state, "np": np, "wp": wp}
                )
                value = eval(compile(source, f"<overlay:{name}>", "eval"), scope)
            elif callable(source):
                from .session import _session_callable  # noqa: PLC0415

                value = _session_callable(source)(session)
            else:
                value = source
            try:
                points = np.asarray(value.numpy() if isinstance(value, wp.array) else value, dtype=np.float64)
            except (TypeError, ValueError):
                points = np.zeros(0)
            if points.ndim not in (1, 2) or points.shape[-1] != 3:
                raise ValueError(f"overlay {name!r} must give a world point [x, y, z] [m] or an (N, 3) array of points")
            result.append((name, points.reshape(-1, 3), points.ndim == 1))
        if sum(len(points) for _, points, _ in result) > self.MAX_OVERLAY_POINTS:
            raise ValueError(f"overlay is limited to {self.MAX_OVERLAY_POINTS} points")
        return result

    def _overlay_markers(self, overlay, metadata: dict) -> list:
        """Project overlay points through the camera of an observation; returns ``(name, pixels, single)``."""
        camera = metadata["camera"]
        width, height = metadata["width"], metadata["height"]
        return [
            (
                name,
                _project(points, camera["pose"], width, height, camera["fov_y"], camera.get("intrinsics"))[0],
                single,
            )
            for name, points, single in self._overlay_points(overlay, metadata["world_id"])
        ]

    def _scene_frame(self, world_id: int) -> tuple[np.ndarray, float, np.ndarray]:
        """Center, bounding radius [m], and extent points of non-plane shapes and particles in one world."""
        from ..geometry.types import GeoType  # noqa: PLC0415

        model, state = self.session.model, self.session.state
        points, radii = [], []
        if model.shape_count:
            shape_type = model.shape_type.numpy()
            shape_body = model.shape_body.numpy()
            shape_world = model.shape_world.numpy()
            transforms = model.shape_transform.numpy()
            radius = model.shape_collision_radius.numpy()
            body_q = state.body_q.numpy() if state.body_q is not None and model.body_count else None
            scales = model.shape_scale.numpy()
            sources = getattr(model, "shape_source", None) or [None] * model.shape_count
            for i in range(model.shape_count):
                if shape_type[i] == GeoType.PLANE or shape_world[i] not in (world_id, -1):
                    continue
                pose = wp.transform(*transforms[i])
                if shape_body[i] >= 0 and body_q is not None:
                    pose = wp.transform_multiply(wp.transform(*body_q[shape_body[i]]), pose)
                vertices = getattr(sources[i], "vertices", None) if shape_type[i] == GeoType.MESH else None
                if vertices is not None and len(vertices):
                    # Mesh bounds are much tighter than the collision radius for elongated meshes.
                    local = np.asarray(vertices, dtype=np.float64) * scales[i]
                    corners = np.array(
                        [
                            [x, y, z]
                            for x in local[:, 0][[local[:, 0].argmin(), local[:, 0].argmax()]]
                            for y in (local[:, 1].min(), local[:, 1].max())
                            for z in (local[:, 2].min(), local[:, 2].max())
                        ]
                    )
                    for corner in corners:
                        points.append(np.asarray(wp.transform_point(pose, wp.vec3(*corner))))
                        radii.append(0.0)
                    continue
                points.append(np.asarray(wp.transform_get_translation(pose)))
                radii.append(float(min(radius[i], 1.0e3)))
        for _, vertices, _, _ in self._overlay_meshes():
            finite = vertices[np.isfinite(vertices).all(axis=1)]
            if len(finite):
                points += [finite.min(axis=0), finite.max(axis=0)]
                radii += [0.0, 0.0]
        if model.particle_count and state.particle_q is not None:
            particles = state.particle_q.numpy()
            worlds = model.particle_world.numpy() if model.particle_world is not None else None
            if worlds is not None:
                particles = particles[(worlds == world_id) | (worlds == -1)]
            particles = particles[np.isfinite(particles).all(axis=1)]
            if len(particles):
                points += [particles.min(axis=0), particles.max(axis=0)]
                radii += [0.0, 0.0]
        if not points:
            return np.zeros(3), 1.0, np.zeros((1, 3))
        points, radii = np.asarray(points, dtype=np.float64), np.asarray(radii)
        finite = np.isfinite(points).all(axis=1)
        points, radii = points[finite], radii[finite]
        lower = (points - radii[:, None]).min(axis=0)
        upper = (points + radii[:, None]).max(axis=0)
        # Each shape contributes points on its bounding sphere (axes and diagonals), which trace
        # its silhouette under perspective without the slack of one scene-wide bounding sphere.
        diagonals = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) / math.sqrt(3.0)
        offsets = np.concatenate([np.eye(3), -np.eye(3), diagonals])
        extent = (points[:, None, :] + radii[:, None, None] * offsets[None]).reshape(-1, 3)
        return 0.5 * (lower + upper), float(0.5 * np.linalg.norm(upper - lower)), extent

    def filmstrip(
        self,
        *,
        times=None,
        every_steps: int | None = None,
        count: int | None = None,
        reset: bool = False,
        restore: str | None = None,
        views=None,
        references=None,
        stride: int | None = None,
        mask=None,
        comparison: str = "mismatch",
        overlay: dict | None = None,
        max_pages: int = 4,
        **options,
    ) -> dict:
        """Advance the simulation and return labeled grids of frames over time.

        Columns are capture times; rows are views. ``times`` are absolute
        simulation times [s] at or after the current time (after the optional
        ``reset``/``restore``); the simulation steps to each one, so frames show
        the simulated state. Alternatively capture ``count`` frames every
        ``every_steps`` steps, starting with the current state.

        ``references`` adds a reference row and a comparison row beneath each
        simulated row, with per-frame metrics (PSNR, SSIM, edge NCC; within
        ``mask`` if given) and their mean. For one view it may be a list of image
        paths or arrays, an ``(N, H, W, 3)`` array, or a directory of images
        (sorted by name), one per time; for several views, one such list per view.
        Simulated frames are rendered at the reference size, so pass the real
        camera's ``pose`` and ``intrinsics`` to compare with recorded video.
        ``stride`` keeps every ``stride``-th time (and reference), e.g. to check a
        long recording a few frames at a time. ``comparison`` selects the third
        row: ``mismatch``, ``edges`` (for real photos), or ``blend``. ``mask`` is
        a boolean ``(H, W)`` array or image path, or one per time.

        Times wrap into bands (each band repeats the rows for its times) and bands into
        pages, so every page fits the size clients display images at; frames are
        shrunk only as far as needed to fit ``max_pages`` pages. The first page is
        ``image_base64``; further pages are in ``images``.

        Cameras on a ``camera_body`` follow that body to each capture time.
        ``overlay`` marks simulated points on the simulated and reference
        frames (see :meth:`observe`), evaluated at each capture time and
        projected through each view's camera; ``overlay`` in the result lists
        their pixel coordinates per time and view.
        """
        self._check_thread()
        session = self.session
        if reset and restore is not None:
            raise ValueError("Use either reset or restore")
        if reset:
            session.dispatch("reset")
        elif restore is not None:
            session.dispatch("restore", {"name": restore})
        if comparison not in ("mismatch", "edges", "blend"):
            raise ValueError("comparison must be 'mismatch', 'edges', or 'blend'")
        if overlay is not None:
            self._check_overlay(overlay)
        if references is not None:
            references = _reference_rows(references, 1 if views is None else len(views))
        per_frame_mask = isinstance(mask, list) and bool(mask) and not isinstance(mask[0], bool)
        if times is not None:
            times = [float(t) for t in np.asarray(times, dtype=float).reshape(-1)]
            if stride is not None:
                stride = _integer("stride", stride, 1, 100_000)
                times = times[::stride]
                references = None if references is None else [row[::stride] for row in references]
                mask = mask[::stride] if per_frame_mask else mask
            if not 1 <= len(times) <= 32:
                raise ValueError("times must list 1 to 32 simulation times [s] (use stride to subsample)")
            if references is not None and any(len(row) != len(times) for row in references):
                raise ValueError("references need one image per time (after stride)")
            order = np.argsort(times, kind="stable")
            targets = [times[i] for i in order]
            references = None if references is None else [[row[i] for i in order] for row in references]
            mask = [mask[i] for i in order] if per_frame_mask else mask
            if targets[0] < session.time - 1.0e-9:
                raise ValueError(f"times must not precede the current time {session.time}; pass reset=true")
        else:
            count = _integer("count", 6 if count is None else count, 1, 32)
            every_steps = _integer("every_steps", 10 if every_steps is None else every_steps, 1, 100_000)
            targets = None
        placement = ("eye", "target", "up", "pose", "view", "camera_body", "camera_offset")
        view_specs = (
            views
            if views is not None
            else [
                dict(options)
                if any(k in options for k in ("eye", "pose", "camera_body", "camera_offset"))
                else options.get("view", "iso")
            ]
        )
        if not isinstance(view_specs, list) or not 1 <= len(view_specs) <= 4:
            raise ValueError("views must list 1 to 4 presets or camera dictionaries")
        options = {k: v for k, v in options.items() if k not in placement} if views is None else options
        names = [_view_name(spec, row) if isinstance(spec, dict) else str(spec) for row, spec in enumerate(view_specs)]
        options.setdefault("width", 320)
        options.setdefault("height", 240)
        if references is not None and len(references) != len(view_specs):
            raise ValueError("references must hold one list of images per view")
        columns, captured_times, statistics, overlay_rows = [], [], [], []
        steps = 0
        for index in range(len(targets) if targets is not None else count):
            if targets is not None:
                while session.time < targets[index] - 0.5 * session.dt:
                    session.dispatch("step", {"count": 1})
                    steps += 1
            elif index:
                session.dispatch("step", {"count": every_steps})
                steps += every_steps
            column = []
            for row, spec in enumerate(view_specs):
                camera = dict(options)
                if isinstance(spec, str):
                    camera.update(view=spec)
                else:
                    camera.update(spec)
                camera.pop("label", None)
                view_overlay = camera.pop("overlay", overlay)
                reference = None if references is None else references[row][index]
                if reference is not None:
                    reference_rgb = load_image(reference) if isinstance(reference, (str, Path)) else _rgb(reference)
                    camera["height"], camera["width"] = reference_rgb.shape[:2]
                rgb, info = self._single(**camera)
                markers = None if view_overlay is None else self._overlay_markers(view_overlay, info)
                if markers is not None:
                    overlay_rows.append(
                        {"view": names[row], "time": round(float(session.time), 6), "pixels": _marker_report(markers)}
                    )
                column.append((rgb, None if reference is None else reference_rgb, markers))
            columns.append(column)
            captured_times.append(float(session.time))
        grid, labels, grid_markers = [], [], []
        for row, name in enumerate(names):
            grid.append([column[row][0] for column in columns])
            labels.append([f"{name} t={t:.3f}" for t in captured_times])
            grid_markers.append([column[row][2] for column in columns])
            if references is not None:
                grid.append([column[row][1] for column in columns])
                labels.append([f"reference t={t:.3f}" for t in captured_times])
                grid_markers.append([column[row][2] for column in columns])
                panels, row_stats = [], []
                for index, (column, t) in enumerate(zip(columns, captured_times, strict=True)):
                    simulated, reference_rgb, _ = column[row]
                    _, stats = _compare_images(simulated, reference_rgb)
                    frame_mask = mask[index] if per_frame_mask else mask
                    weights = None if frame_mask is None else _mask(frame_mask, simulated.shape[:2])
                    stats.update(_image_metrics(simulated, reference_rgb, weights))
                    panels.append(_comparison_panel(simulated, reference_rgb, comparison))
                    row_stats.append({"view": name, "time": round(t, 6), **stats})
                statistics.extend(row_stats)
                grid.append(panels)
                labels.append([f"{comparison} ssim {s['ssim']:.2f} ncc {s['edge_ncc']:.2f}" for s in row_stats])
                grid_markers.append([None] * len(columns))
        # Shrink thumbnails (box filter) only as far as needed for the pages to fit the display size; metrics stay
        # full size.
        scale, per_band, bands = _page_layout(
            [max(im.shape[0] for im in r) for r in grid],
            max(im.shape[1] for r in grid for im in r),
            len(columns),
            self.DISPLAY_EDGE,
            max(1, int(max_pages)),
        )
        if scale > 1:
            grid = [[_shrink(im, scale) for im in r] for r in grid]
        # Markers go on after the metrics and the downscale, so they stay crisp and never affect scores.
        grid = [
            [
                image if markers is None else _draw_markers(image, markers, scale)
                for image, markers in zip(r, m, strict=True)
            ]
            for r, m in zip(grid, grid_markers, strict=True)
        ]
        pages = []
        per_page = per_band * bands
        for first in range(0, len(columns), per_page):
            band_images = []
            for start in range(first, min(first + per_page, len(columns)), per_band):
                stop = min(start + per_band, len(columns))
                band_images.append(tile([r[start:stop] for r in grid], [r[start:stop] for r in labels]))
            page = _stack_bands(band_images)
            if page.shape[0] * page.shape[1] > self.MAX_PIXELS:
                raise ValueError("Filmstrip exceeds the pixel budget; reduce width/height, times, or views")
            pages.append(page)
        layout = "columns = times; rows = views" + (" (simulated, reference, mismatch)" if references else "")
        if len(pages) > 1 or per_band < len(columns):
            layout += f"; {per_band} times per band, {bands} bands per page"
        result = {
            "time": float(session.time),
            "frame": int(session.frame),
            "times": [round(t, 6) for t in captured_times],
            "steps_advanced": steps,
            "layout": layout,
            "image_base64": base64.b64encode(_png(pages[0])).decode("ascii"),
            "mime_type": "image/png",
        }
        if len(pages) > 1:
            result["pages"] = len(pages)
            result["images"] = [
                {"image_base64": base64.b64encode(_png(page)).decode("ascii"), "mime_type": "image/png"}
                for page in pages[1:]
            ]
        if overlay_rows:
            result["overlay"] = overlay_rows
        if statistics:
            result["mismatch"] = statistics
            keys = ("psnr_db", "ssim", "edge_ncc", "mismatch_fraction")
            result["metrics_mean"] = {k: round(float(np.mean([s[k] for s in statistics])), 4) for k in keys}
        if scale > 1:
            result["thumbnail_scale"] = 1.0 / scale
        return result

    def _render_sensor(
        self,
        width,
        height,
        fov_y,
        pose,
        world_id,
        channel,
        shadows,
        textures,
        contacts,
        pick,
        supersample=1,
        environment=False,
        intrinsics=None,
    ):
        """Render one camera; returns per-pixel arrays at ``width x height`` and environment metadata."""
        base_width, base_height = width, height
        width, height = width * supersample, height * supersample
        model, state = self.session.model, self.session.state
        if self._sensor is None or self._sensor_model is not model:
            self.invalidate()
            config = SensorTiledCamera.RenderConfig(enable_shadows=True)
            self._sensor = SensorTiledCamera(model, default_render_config=config, load_textures=True)
            self._sensor.utils.create_default_light(enable_shadows=True)
            self._sensor_model = model
            self._visibility_signature = None
        key = (width, height, fov_y, json.dumps(intrinsics, sort_keys=True))
        if key != self._buffer_key:
            self._outputs = {}
            if intrinsics is None:
                self._rays = self._sensor.utils.compute_camera_rays_pinhole(
                    width, height, camera_fovs=math.radians(fov_y)
                )
            elif intrinsics.get("distortion_model") == "inverse_brown_conrady":
                self._rays = wp.array(
                    _inverse_brown_conrady_rays(width, height, intrinsics), dtype=wp.vec3f, device=model.device
                )
            else:
                # The helper rescales the calibration to the (supersampled) output size.
                self._rays = self._sensor.utils.compute_camera_rays_pinhole_opencv(width, height, **intrinsics)
            self._transforms = wp.empty((1, 1), dtype=wp.transform, device=model.device)
            self._buffer_key = key
        self._transforms.assign(np.asarray(pose, dtype=np.float32).reshape(1, 1, 7))
        world_ids = wp.array([world_id], dtype=wp.int32, device=model.device)
        overlay = self._overlay_meshes() if channel == "color" else []
        needed = {channel}
        if contacts or overlay:
            needed.add("forward_depth")
        if pick is not None or environment:
            needed.add("shape_index")
        if environment:
            needed.add("depth")
        # Retain only this request's outputs so channel changes cannot grow the cache without bound.
        self._outputs = {name: value for name, value in self._outputs.items() if name in needed}
        for name in needed:
            if name not in self._outputs:
                create = getattr(self._sensor.utils, f"create_{name}_image_output")
                self._outputs[name] = create(width, height, world_count=1)
        # Shapes hidden or made transparent after finalize() only leave the render BVH on a rebuild.
        visible = model.shape_flags.numpy() & int(ShapeFlags.VISIBLE) != 0
        if model.shape_opacity is not None:
            visible &= model.shape_opacity.numpy() > 0.0
        signature = hash(visible.tobytes())
        if signature != self._visibility_signature:
            if self._visibility_signature is not None:
                model.bvh_build_shapes(state)
            self._visibility_signature = signature
        model.bvh_refit_shapes(state)
        model.bvh_refit_particles(state)
        config = SensorTiledCamera.RenderConfig(enable_shadows=shadows, enable_textures=textures)
        self._sensor.update(
            state,
            self._transforms,
            self._rays,
            render_config=config,
            world_ids=world_ids,
            **{f"{name}_image": output for name, output in self._outputs.items()},
        )
        if overlay:
            arrays = {name: output[0, 0].numpy() for name, output in self._outputs.items()}
            self._composite_overlay(arrays, overlay, width, height, pose, shadows)
            for name, values in arrays.items():
                self._outputs[name][0, 0].assign(values)
        environment_metadata = {}
        if channel == "color" and (environment or supersample > 1):
            color, environment_metadata = self._finish_color(
                pose, world_id, supersample, environment, base_width, base_height
            )
        arrays = {}
        for name, output in self._outputs.items():
            values = output[0, 0].numpy()
            arrays[name] = values[supersample // 2 :: supersample, supersample // 2 :: supersample][
                :base_height, :base_width
            ]
        if channel == "color" and (environment or supersample > 1):
            arrays["color"] = color
        return arrays, environment_metadata

    def _finish_color(self, pose, world_id, supersample, environment, width, height):
        """Sky, ground checker, and supersample resolve on the device; returns packed color and metadata."""
        from ..geometry.types import GeoType  # noqa: PLC0415

        model = self.session.model
        device = model.device
        metadata = {}
        plane_flag = np.zeros(max(model.shape_count, 1), dtype=np.int32)
        plane_inverse = np.zeros((max(model.shape_count, 1), 7), dtype=np.float32)
        cell = 1.0
        if environment:
            metadata["environment"] = "sky gradient behind the scene"
            if model.shape_count:
                types = model.shape_type.numpy()
                planes = np.flatnonzero(types == int(GeoType.PLANE))
                if len(planes):
                    plane_flag[planes] = 1
                    transforms = model.shape_transform.numpy()
                    for index in planes:
                        plane_inverse[index] = np.asarray(wp.transform_inverse(wp.transform(*transforms[index])))
                    cell = _checker_cell(self._scene_frame(world_id)[1])
                    metadata["environment"] += f"; ground checker cells {cell:g} m"
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:7])), dtype=np.float32).reshape(3, 3)
        dummy = wp.zeros((1, 1), dtype=wp.uint32, device=device)
        dummy_depth = wp.zeros((1, 1), dtype=wp.float32, device=device)
        out = wp.empty((height, width), dtype=wp.uint32, device=device)
        wp.launch(
            _finish_color,
            dim=(height, width),
            inputs=[
                self._outputs["color"][0, 0],
                self._outputs["shape_index"][0, 0] if environment else dummy,
                self._outputs["depth"][0, 0] if environment else dummy_depth,
                self._rays[0],
                wp.mat33f(*rotation.flatten()),
                wp.vec3f(*np.asarray(pose[:3], dtype=np.float32)),
                wp.vec3f(*np.eye(3, dtype=np.float32)[int(model.up_axis)]),
                wp.array(plane_inverse, dtype=wp.transformf, device=device),
                wp.array(plane_flag, dtype=wp.int32, device=device),
                float(cell),
                int(environment),
                int(supersample),
            ],
            outputs=[out],
            device=device,
        )
        return out.numpy(), metadata

    def _render_rtx(self, width, height, fov_y, pose, world_id, samples):
        """Path-trace the observed world with a lazily created, headless ViewerRTX."""
        model, state = self.session.model, self.session.state
        colors = model.shape_color.numpy() if getattr(model, "shape_color", None) is not None else None
        rebuilt = False
        if (
            self._rtx is None
            or self._rtx_model is not model
            or (colors is not None and not np.array_equal(colors, self._rtx_colors))
        ):
            # Materials are baked when the renderer is built, so appearance edits need a rebuild.
            self._close_rtx()
            try:
                import newton.viewer  # noqa: PLC0415

                viewer = newton.viewer.ViewerRTX(width=width, height=height, headless=True, async_rendering=False)
            except ImportError as error:
                raise ValueError(
                    "backend='rtx' needs the ovrtx package (pip install newton[rtx]) and an RTX GPU"
                ) from error
            viewer.set_model(model)
            if model.world_count > 1:
                viewer.set_world_offsets((0.0, 0.0, 0.0))
            self._rtx, self._rtx_model, self._rtx_colors, self._rtx_world = viewer, model, colors, None
            rebuilt = True
        if model.world_count > 1 and self._rtx_world != world_id:
            self._rtx.set_visible_worlds([world_id])
            self._rtx_world = world_id
        overlay = self._overlay_meshes()
        rgb = self._rtx._render_offscreen(state, pose, fov_y, width, height, samples=samples, meshes=overlay)
        return np.ascontiguousarray(rgb[..., :3], dtype=np.uint8), {
            "renderer": "OVRTX path tracer",
            "samples": samples,
            "renderer_rebuilt": rebuilt,
        }

    def blender_worker(self, world_id: int = 0):
        """The Blender render worker for the current model, started on first use.

        The worker is rebuilt when the model, its visible shapes, or shape colors change.
        """
        from .blender_bridge import BlenderRenderer, find_blender  # noqa: PLC0415

        model = self.session.model
        signature = (
            id(model),
            world_id,
            hash(model.shape_flags.numpy().tobytes()),
            hash(model.shape_color.numpy().tobytes()) if model.shape_color is not None else None,
        )
        if self._blender is not None and self._blender_signature == signature:
            return self._blender, False
        self._close_blender()
        blender = find_blender()
        if blender is None:
            raise ValueError("backend='blender' needs Blender; set NEWTON_BLENDER or put blender on PATH")
        directory = Path(self.session.artifact_directory) / "blender" if self.session.artifact_directory else None
        self._blender = BlenderRenderer(model, blender=blender, world_id=world_id, workdir=directory)
        self._blender_signature = signature
        return self._blender, True

    def _render_blender(self, width, height, fov_y, pose, world_id, samples, intrinsics, engine):
        worker, rebuilt = self.blender_worker(world_id)
        camera, coordinates = intrinsics, None
        if intrinsics is not None and not _is_square_pinhole(intrinsics, width, height):
            # Blender cameras are square-pixel pinholes without lens distortion.
            key = (width, height, json.dumps(intrinsics, sort_keys=True))
            if self._pinhole_cover_key != key:
                self._pinhole_cover = _pinhole_cover(
                    width, height, intrinsics, self.session.model.device, self.MAX_PIXELS
                )
                self._pinhole_cover_key = key
            camera, coordinates = self._pinhole_cover
        rgb, timing = worker.render(
            self.session.state,
            pose=pose,
            fov_y=fov_y,
            width=width if coordinates is None else int(camera["image_width"]),
            height=height if coordinates is None else int(camera["image_height"]),
            samples=samples,
            engine=engine,
            intrinsics=camera,
        )
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        metadata = {
            "renderer": f"Blender {engine.title()}",
            "samples": samples,
            "renderer_rebuilt": rebuilt,
            "blender_render_seconds": timing.get("render_s"),
            **({"startup_seconds": round(worker.startup_s, 2)} if rebuilt else {}),
        }
        if coordinates is not None:
            rgb = _sample_bilinear(rgb, coordinates)
            size = f"{int(camera['image_width'])}x{int(camera['image_height'])}"
            metadata["lens"] = f"resampled through the calibrated rays from a {size} pinhole render"
        return rgb, metadata

    def _close_blender(self):
        worker, self._blender = getattr(self, "_blender", None), None
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass

    def _close_rtx(self):
        viewer, self._rtx = getattr(self, "_rtx", None), None
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    def _has_textures(self) -> bool:
        """Whether any shape of the session model carries a texture (textures then render by default)."""
        model = self.session.model
        if self._textured_model is not model:
            sources = getattr(model, "shape_source", None) or []
            self._textured = any(getattr(source, "texture", None) is not None for source in sources)
            self._textured_model = model
        return self._textured

    def _overlay_meshes(self) -> list:
        callback = getattr(self.session, "overlay_callback", None)
        if callback is None:
            return []
        try:
            return callback(self.session)
        except Exception:
            return []

    def _composite_overlay(self, arrays, meshes, width, height, pose, shadows):
        """Draw application-logged meshes with the same camera and keep the nearer surface per pixel."""
        import newton  # noqa: PLC0415

        model = self.session.model
        builder = newton.ModelBuilder(up_axis=int(model.up_axis))
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        for name, points, indices, color in meshes:
            mesh = newton.Mesh(points, indices, compute_inertia=False)
            builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg, color=color, label=str(name))
        overlay = builder.finalize(device=model.device)
        sensor = SensorTiledCamera(
            overlay, default_render_config=SensorTiledCamera.RenderConfig(enable_shadows=shadows)
        )
        sensor.utils.create_default_light(enable_shadows=shadows)
        rays = self._rays  # the same camera rays as the main render, including any intrinsics
        transforms = wp.array(
            np.asarray(pose, dtype=np.float32).reshape(1, 1, 7), dtype=wp.transform, device=model.device
        )
        color = sensor.utils.create_color_image_output(width, height)
        depth = sensor.utils.create_forward_depth_image_output(width, height)
        state = overlay.state()
        overlay.bvh_refit_shapes(state)
        sensor.update(
            state,
            transforms,
            rays,
            render_config=SensorTiledCamera.RenderConfig(enable_shadows=shadows),
            color_image=color,
            forward_depth_image=depth,
        )
        overlay_depth = depth[0, 0].numpy()
        base_depth = arrays["forward_depth"]
        hit = np.isfinite(overlay_depth) & (overlay_depth > 0)
        base_hit = np.isfinite(base_depth) & (base_depth > 0)
        nearer = hit & (~base_hit | (overlay_depth < base_depth))
        arrays["color"] = np.where(nearer, color[0, 0].numpy(), arrays["color"])
        arrays["forward_depth"] = np.where(nearer, overlay_depth, base_depth)
        if "shape_index" in arrays:
            # Mark overlay pixels as hits that are not model shapes, so the sky pass keeps them.
            arrays["shape_index"] = np.where(nearer, np.uint32(0xFFFFFFFE), arrays["shape_index"])

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

    def _overlay_contacts(self, rgb, camera, world_id, policy, depth):
        data = self.session.contact_data(refresh=False, limit=256, world=world_id, include_global=True)
        height, width = rgb.shape[:2]
        points = [(np.asarray(row["surface0"]) + np.asarray(row["surface1"])) * 0.5 for row in data["rows"]]
        pixels, distances = _project(
            np.reshape(points, (-1, 3)), camera["pose"], width, height, camera["fov_y"], camera.get("intrinsics")
        )
        drawn = 0
        for (u, v), distance in zip(pixels, distances, strict=True):
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            x, y = int(math.floor(u)), int(math.floor(v))
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
        # Options may hold arrays or overlay callables, which have no JSON form.
        text = json.dumps(recording, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else repr(v))
        path.write_text(text, encoding="utf-8")

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
        self._close_rtx()
        self._close_blender()
