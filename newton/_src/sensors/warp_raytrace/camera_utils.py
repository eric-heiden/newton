# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import warp as wp

from newton._src.core.cameras import (  # noqa: F401
    _fisheye_direction_from_theta,
    compute_camera_rays_fisheye_ftheta_kernel,
    compute_camera_rays_fisheye_kannala_brandt_kernel,
    compute_camera_rays_fisheye_opencv_kernel,
    compute_camera_rays_pinhole_from_aperture_kernel,
)

if TYPE_CHECKING:
    from pxr import Usd, UsdGeom

    UsdCameraLike: TypeAlias = Usd.Prim | UsdGeom.Camera
    UsdTime: TypeAlias = Usd.TimeCode | float
else:
    UsdCameraLike: TypeAlias = Any
    UsdTime: TypeAlias = Any

UsdCameraInput: TypeAlias = UsdCameraLike | Sequence[UsdCameraLike]
UsdCameraGridInput: TypeAlias = UsdCameraInput | Sequence[Sequence[UsdCameraLike]]


def _is_camera_sequence(cameras: Any) -> bool:
    return isinstance(cameras, Sequence) and not isinstance(cameras, (str, bytes, bytearray))


def _camera_param_count(param: Any) -> int:
    if isinstance(param, numbers.Real):
        return 1
    if isinstance(param, list | tuple | np.ndarray):
        return int(np.asarray(param).size)
    return int(param.size)


def _camera_param_array(
    name: str,
    param: Any,
    camera_count: int,
    device: wp.Device,
) -> wp.array[wp.float32]:
    if isinstance(param, numbers.Real):
        return wp.full((camera_count,), value=float(param), dtype=wp.float32, device=device)

    if isinstance(param, list | tuple | np.ndarray):
        values = np.asarray(param, dtype=np.float32).reshape(-1)
        if values.size == camera_count:
            return wp.array(values, dtype=wp.float32, device=device)
        if values.size == 1:
            return wp.full((camera_count,), value=float(values[0]), dtype=wp.float32, device=device)
        raise ValueError(f"{name} must have length 1 or {camera_count}.")

    if param.size == 1:
        value = float(param.numpy().reshape(-1)[0])
        return wp.full((camera_count,), value=value, dtype=wp.float32, device=device)
    if param.size != camera_count:
        raise ValueError(f"{name} must have length 1 or {camera_count}.")
    if param.dtype != wp.float32 or param.device != device:
        return wp.array(param.numpy().reshape(-1).astype(np.float32), dtype=wp.float32, device=device)
    return param


def _validate_camera_ray_output(
    width: int,
    height: int,
    camera_count: int,
    out_rays: wp.array4d[wp.vec3f] | None,
    camera_index: int,
    device: wp.Device,
) -> tuple[wp.array4d[wp.vec3f], int]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    camera_index = int(camera_index)
    if camera_index < 0:
        raise ValueError("camera_index must be non-negative.")

    if out_rays is None:
        out_rays = wp.empty((camera_count, height, width, 2), dtype=wp.vec3f, device=device)
        camera_index = 0
    elif (
        out_rays.shape[0] < camera_index + camera_count
        or out_rays.shape[1] != height
        or out_rays.shape[2] != width
        or out_rays.shape[3] != 2
    ):
        raise ValueError("out_rays must have shape (out_camera_count, height, width, 2) with enough camera slots.")

    return out_rays, camera_index


def _coerce_usd_time(time: Any) -> Any:
    try:
        from pxr import Usd
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    if time is None:
        return Usd.TimeCode.Default()
    if isinstance(time, Usd.TimeCode):
        return time
    return Usd.TimeCode(float(time))


def _normalize_usd_cameras(cameras: UsdCameraInput) -> list[Any]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    if _is_camera_sequence(cameras):
        camera_items = list(cameras)
    else:
        camera_items = [cameras]

    if not camera_items:
        raise ValueError("At least one USD camera is required.")

    usd_cameras = []
    for camera in camera_items:
        if isinstance(camera, UsdGeom.Camera):
            usd_camera = camera
            prim = usd_camera.GetPrim()
        elif isinstance(camera, Usd.Prim):
            prim = camera
            if not prim.IsValid():
                raise TypeError("Expected a valid UsdGeom.Camera prim.")
            usd_camera = UsdGeom.Camera(prim)
        else:
            raise TypeError("Expected a UsdGeom.Camera or Usd.Prim.")

        if not prim.IsValid():
            raise TypeError("Expected a valid UsdGeom.Camera prim.")
        if not prim.IsA(UsdGeom.Camera):
            raise TypeError(f"Expected a UsdGeom.Camera prim, got {prim.GetPath()!r}.")
        usd_cameras.append(usd_camera)

    return usd_cameras


def compute_camera_rays_usd_pinhole(
    width: int,
    height: int,
    cameras: UsdCameraInput,
    *,
    device: wp.Device,
    time: UsdTime | None = None,
    out_rays: wp.array4d[wp.vec3f] | None = None,
    camera_index: int = 0,
) -> wp.array4d[wp.vec3f]:
    time_code = _coerce_usd_time(time)
    usd_cameras = _normalize_usd_cameras(cameras)
    camera_count = len(usd_cameras)
    out_rays, camera_index = _validate_camera_ray_output(width, height, camera_count, out_rays, camera_index, device)

    focal_lengths = []
    horizontal_apertures = []
    vertical_apertures = []
    horizontal_aperture_offsets = []
    vertical_aperture_offsets = []
    for usd_camera in usd_cameras:
        projection = str(usd_camera.GetProjectionAttr().Get(time_code))
        if projection != "perspective":
            prim = usd_camera.GetPrim()
            raise NotImplementedError(f"USD camera {prim.GetPath()} uses unsupported projection {projection!r}.")

        focal_lengths.append(float(usd_camera.GetFocalLengthAttr().Get(time_code)))
        horizontal_apertures.append(float(usd_camera.GetHorizontalApertureAttr().Get(time_code)))
        vertical_apertures.append(float(usd_camera.GetVerticalApertureAttr().Get(time_code)))
        horizontal_aperture_offsets.append(float(usd_camera.GetHorizontalApertureOffsetAttr().Get(time_code)))
        vertical_aperture_offsets.append(float(usd_camera.GetVerticalApertureOffsetAttr().Get(time_code)))

    wp.launch(
        kernel=compute_camera_rays_pinhole_from_aperture_kernel,
        dim=(camera_count, height, width),
        inputs=[
            width,
            height,
            wp.array(focal_lengths, dtype=wp.float32, device=device),
            wp.array(horizontal_apertures, dtype=wp.float32, device=device),
            wp.array(vertical_apertures, dtype=wp.float32, device=device),
            wp.array(horizontal_aperture_offsets, dtype=wp.float32, device=device),
            wp.array(vertical_aperture_offsets, dtype=wp.float32, device=device),
            camera_index,
            out_rays,
        ],
        device=device,
    )

    return out_rays


def compute_camera_transforms_usd(
    cameras: UsdCameraGridInput,
    *,
    world_count: int,
    device: wp.Device,
    target_up_axis: Any | None = None,
    time: UsdTime | None = None,
    xform: Any | None = None,
) -> wp.array2d[wp.transformf]:
    try:
        from pxr import UsdGeom
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    from ...core import Axis, quat_between_axes  # noqa: PLC0415
    from ...usd.utils import get_transform  # noqa: PLC0415

    time_code = _coerce_usd_time(time)
    xform_cache = UsdGeom.XformCache(time_code)
    scene_xform = wp.transform(*xform) if xform is not None else None

    def world_transform(usd_camera: Any) -> wp.transformf:
        transform = get_transform(usd_camera.GetPrim(), local=False, xform_cache=xform_cache)
        if target_up_axis is not None:
            stage_up_axis = Axis.from_string(str(UsdGeom.GetStageUpAxis(usd_camera.GetPrim().GetStage())))
            axis_xform = wp.transform(wp.vec3(0.0), quat_between_axes(stage_up_axis, target_up_axis))
            transform = axis_xform * transform
        if scene_xform is not None:
            transform = scene_xform * transform
        return transform

    is_per_world = _is_camera_sequence(cameras) and len(cameras) > 0 and _is_camera_sequence(cameras[0])

    if is_per_world:
        if len(cameras) != world_count:
            raise ValueError(
                f"compute_camera_transforms_usd: per-world cameras outer dimension {len(cameras)} "
                f"must match world_count {world_count}."
            )
        rows = [_normalize_usd_cameras(row) for row in cameras]
        camera_count = len(rows[0])
        for world_index, row in enumerate(rows):
            if len(row) != camera_count:
                raise ValueError(
                    f"compute_camera_transforms_usd: per-world cameras row {world_index} has "
                    f"{len(row)} cameras, expected {camera_count}."
                )
        transforms = [
            [world_transform(rows[world_index][camera_index]) for world_index in range(world_count)]
            for camera_index in range(camera_count)
        ]
    else:
        usd_cameras = _normalize_usd_cameras(cameras)
        transforms = [[world_transform(usd_camera)] * world_count for usd_camera in usd_cameras]

    return wp.array(
        transforms,
        dtype=wp.transformf,
        device=device,
    )


@wp.kernel(enable_backward=False)
def compute_camera_rays_pinhole(
    width: int,
    height: int,
    camera_fovs: wp.array[wp.float32],
    camera_index_start: int,
    out_rays: wp.array4d[wp.vec3f],
):
    camera_index, py, px = wp.tid()
    output_camera_index = camera_index_start + camera_index
    aspect_ratio = float(width) / float(height)
    u = (float(px) + 0.5) / float(width) - 0.5
    v = (float(py) + 0.5) / float(height) - 0.5
    h = wp.tan(camera_fovs[camera_index] / 2.0)
    ray_direction_camera_space = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    out_rays[output_camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[output_camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)
