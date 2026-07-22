# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera projection descriptors and shared camera math for Newton."""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

__all__ = [
    "CameraCustomRays",
    "CameraFisheyeFTheta",
    "CameraFisheyeKannalaBrandt",
    "CameraFisheyeOpenCV",
    "CameraPinhole",
    "CameraProjection",
    "compute_camera_rays",
    "compute_camera_rays_fisheye_ftheta_kernel",
    "compute_camera_rays_fisheye_kannala_brandt_kernel",
    "compute_camera_rays_fisheye_opencv_kernel",
    "compute_camera_rays_pinhole_from_aperture_kernel",
    "eval_camera_world_xforms",
    "xform_to_pitch_yaw",
]

# Defaults match UsdGeom.Camera fallback values (35mm film back).
_DEFAULT_FOCAL_LENGTH = 18.0
_DEFAULT_HORIZONTAL_APERTURE = 20.955
_DEFAULT_VERTICAL_APERTURE = 15.2908


@dataclass(frozen=True, kw_only=True)
class CameraProjection:
    """Base class for camera projection descriptors.

    Projection descriptors are immutable value objects: parametric projections
    compare by value so that cameras sharing identical projections are
    deduplicated at :meth:`ModelBuilder.finalize`.

    Camera space is -Z forward, +Y up, +X right.
    """

    near: float = 0.01
    """Near clipping distance hint [m]."""
    far: float = 1000.0
    """Far clipping distance hint [m]."""


@dataclass(frozen=True, kw_only=True)
class CameraPinhole(CameraProjection):
    """Perspective pinhole projection in physical (USD-style) form."""

    focal_length: float = _DEFAULT_FOCAL_LENGTH
    """Focal length in aperture units (USD convention: tenths of world units)."""
    horizontal_aperture: float = _DEFAULT_HORIZONTAL_APERTURE
    """Horizontal film aperture, same units as :attr:`focal_length`."""
    vertical_aperture: float = _DEFAULT_VERTICAL_APERTURE
    """Vertical film aperture, same units as :attr:`focal_length`."""
    horizontal_aperture_offset: float = 0.0
    """Horizontal aperture offset."""
    vertical_aperture_offset: float = 0.0
    """Vertical aperture offset."""

    @classmethod
    def from_fov(cls, fov: float, aspect: float | None = None, **kwargs) -> CameraPinhole:
        """Creates a pinhole projection from a vertical field of view.

        Args:
            fov: Vertical field of view [rad].
            aspect: Optional width/height aspect ratio. If given, the
                horizontal aperture is set to ``vertical_aperture * aspect``.
            **kwargs: Forwarded to the constructor (e.g. ``near``, ``far``).

        Returns:
            A :class:`CameraPinhole` whose :attr:`fov` equals ``fov``.
        """
        vertical_aperture = kwargs.pop("vertical_aperture", _DEFAULT_VERTICAL_APERTURE)
        focal_length = 0.5 * vertical_aperture / math.tan(0.5 * fov)
        horizontal_aperture = kwargs.pop("horizontal_aperture", _DEFAULT_HORIZONTAL_APERTURE)
        if aspect is not None:
            horizontal_aperture = vertical_aperture * aspect
        return cls(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            **kwargs,
        )

    @property
    def fov(self) -> float:
        """Vertical field of view [rad]."""
        return 2.0 * math.atan(0.5 * self.vertical_aperture / self.focal_length)


@dataclass(frozen=True, kw_only=True)
class CameraFisheyeOpenCV(CameraProjection):
    """OpenCV fisheye model: r = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)."""

    fx: float
    """Horizontal focal length [px]."""
    fy: float
    """Vertical focal length [px]."""
    cx: float
    """Principal point x-coordinate [px]."""
    cy: float
    """Principal point y-coordinate [px]."""
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    max_fov: float = 2.0 * math.pi
    """Maximum field of view [rad]; rays beyond it are marked invalid."""
    image_width: float | None = None
    """Calibration image width [px]; None means the render width."""
    image_height: float | None = None
    """Calibration image height [px]; None means the render height."""


@dataclass(frozen=True, kw_only=True)
class CameraFisheyeFTheta(CameraProjection):
    """F-theta fisheye model: r = k0 + k1*theta + k2*theta^2 + k3*theta^3 + k4*theta^4."""

    optical_center_x: float
    optical_center_y: float
    k0: float = 0.0
    k1: float = 1.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    max_fov: float = 2.0 * math.pi
    """Maximum field of view [rad]; rays beyond it are marked invalid."""
    nominal_width: float | None = None
    """Calibration image width [px]; None means the render width."""
    nominal_height: float | None = None
    """Calibration image height [px]; None means the render height."""


@dataclass(frozen=True, kw_only=True)
class CameraFisheyeKannalaBrandt(CameraProjection):
    """Kannala-Brandt K3 fisheye model: r = k0*theta + k1*theta^3 + k2*theta^5 + k3*theta^7."""

    optical_center_x: float
    optical_center_y: float
    k0: float = 1.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    max_fov: float = 2.0 * math.pi
    """Maximum field of view [rad]; rays beyond it are marked invalid."""
    nominal_width: float | None = None
    """Calibration image width [px]; None means the render width."""
    nominal_height: float | None = None
    """Calibration image height [px]; None means the render height."""


@dataclass(frozen=True, eq=False, kw_only=True)
class CameraCustomRays(CameraProjection):
    """User-provided camera-space ray bundle for custom camera models.

    Equality is object identity: pass the *same* instance to multiple cameras
    to share one ray bundle in memory. The image resolution is fixed by the
    ray array shape.
    """

    rays: wp.array
    """Camera-space rays, shape [height, width, 2] of ``wp.vec3f``: index 0 is the ray origin [m], index 1 the normalized direction."""

    def __post_init__(self):
        if self.rays.ndim != 3 or self.rays.shape[2] != 2 or self.rays.dtype != wp.vec3f:
            raise ValueError(
                f"CameraCustomRays.rays must have shape (height, width, 2) and dtype vec3f, "
                f"got shape {self.rays.shape} and dtype {self.rays.dtype}"
            )

    # Identity equality: two distinct CameraCustomRays instances are never equal
    # even if their ray arrays have identical contents, so multiple cameras can
    # share the same bundle by passing the same instance.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    @property
    def resolution(self) -> tuple[int, int]:
        """Image resolution as (width, height)."""
        return (self.rays.shape[1], self.rays.shape[0])


# ---------------------------------------------------------------------------
# Warp helper functions shared by ray-generation kernels
# ---------------------------------------------------------------------------


@wp.func
def _opencv_fisheye_radius(theta: wp.float32, k0: wp.float32, k1: wp.float32, k2: wp.float32, k3: wp.float32):
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    return theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8)


@wp.func
def _ftheta_radius(
    theta: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
):
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta2 * theta2
    return k0 + k1 * theta + k2 * theta2 + k3 * theta3 + k4 * theta4


@wp.func
def _kannala_brandt_k3_radius(
    theta: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
):
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta5 = theta3 * theta2
    theta7 = theta5 * theta2
    return k0 * theta + k1 * theta3 + k2 * theta5 + k3 * theta7


@wp.func
def _solve_opencv_fisheye_theta(
    radius: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    max_theta: wp.float32,
):
    if radius <= 1.0e-7:
        return wp.float32(0.0)

    # This endpoint check and the binary search assume r(theta) is monotonic.
    max_radius = _opencv_fisheye_radius(max_theta, k0, k1, k2, k3)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(24):
        mid = (lo + hi) * 0.5
        if _opencv_fisheye_radius(mid, k0, k1, k2, k3) < radius:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@wp.func
def _solve_ftheta_theta(
    radius: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    max_theta: wp.float32,
):
    if radius <= 1.0e-7:
        return wp.float32(0.0)

    # When k0 != 0 the polynomial has a nonzero floor at theta=0 (r(0) = k0).
    # Pixels inside that central circle are undefined by the model; return theta=0 (forward).
    min_radius = _ftheta_radius(0.0, k0, k1, k2, k3, k4)
    if radius <= min_radius:
        return wp.float32(0.0)

    # This endpoint check and the binary search assume r(theta) is monotonic.
    max_radius = _ftheta_radius(max_theta, k0, k1, k2, k3, k4)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(24):
        mid = (lo + hi) * 0.5
        if _ftheta_radius(mid, k0, k1, k2, k3, k4) < radius:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@wp.func
def _solve_kannala_brandt_k3_theta(
    radius: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    max_theta: wp.float32,
):
    if radius <= 1.0e-7:
        return wp.float32(0.0)

    # This endpoint check and the binary search assume r(theta) is monotonic.
    max_radius = _kannala_brandt_k3_radius(max_theta, k0, k1, k2, k3)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(24):
        mid = (lo + hi) * 0.5
        if _kannala_brandt_k3_radius(mid, k0, k1, k2, k3) < radius:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@wp.func
def _fisheye_direction_from_theta(x: wp.float32, y: wp.float32, radius: wp.float32, theta: wp.float32):
    # Valid fisheye rays are unit-length by construction; zero is reserved for invalid rays.
    if theta < 0.0:
        return wp.vec3f(0.0)
    if radius <= 1.0e-7:
        return wp.vec3f(0.0, 0.0, -1.0)

    sin_theta = wp.sin(theta)
    return wp.vec3f((x / radius) * sin_theta, (y / radius) * sin_theta, -wp.cos(theta))


# ---------------------------------------------------------------------------
# Ray-generation warp kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def compute_camera_rays_pinhole_from_aperture_kernel(
    width: int,
    height: int,
    focal_lengths: wp.array[wp.float32],
    horizontal_apertures: wp.array[wp.float32],
    vertical_apertures: wp.array[wp.float32],
    horizontal_aperture_offsets: wp.array[wp.float32],
    vertical_aperture_offsets: wp.array[wp.float32],
    camera_index_start: int,
    out_rays: wp.array4d[wp.vec3f],
):
    camera_index, py, px = wp.tid()
    output_camera_index = camera_index_start + camera_index
    u = (float(px) + 0.5) / float(width)
    v = (float(py) + 0.5) / float(height)
    film_x = (u - 0.5) * horizontal_apertures[camera_index] + horizontal_aperture_offsets[camera_index]
    film_y = (0.5 - v) * vertical_apertures[camera_index] + vertical_aperture_offsets[camera_index]
    focal_length = focal_lengths[camera_index]
    ray_direction_camera_space = wp.vec3f(film_x / focal_length, film_y / focal_length, -1.0)
    out_rays[output_camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[output_camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)


@wp.kernel(enable_backward=False)
def compute_camera_rays_fisheye_opencv_kernel(
    width: int,
    height: int,
    image_width: wp.float32,
    image_height: wp.float32,
    fx: wp.float32,
    fy: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    max_fov: wp.float32,
    camera_index: int,
    out_rays: wp.array4d[wp.vec3f],
):
    py, px = wp.tid()
    u = ((float(px) + 0.5) / float(width)) * image_width
    v = ((float(py) + 0.5) / float(height)) * image_height
    x = (u - cx) / fx
    y = -(v - cy) / fy
    radius = wp.sqrt(x * x + y * y)
    theta = _solve_opencv_fisheye_theta(
        radius,
        k1,
        k2,
        k3,
        k4,
        wp.min(max_fov * wp.float32(0.5), wp.float32(math.pi)),
    )
    ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space


@wp.kernel(enable_backward=False)
def compute_camera_rays_fisheye_ftheta_kernel(
    width: int,
    height: int,
    nominal_width: wp.float32,
    nominal_height: wp.float32,
    optical_center_x: wp.float32,
    optical_center_y: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    max_fov: wp.float32,
    camera_index: int,
    out_rays: wp.array4d[wp.vec3f],
):
    py, px = wp.tid()
    u = ((float(px) + 0.5) / float(width)) * nominal_width
    v = ((float(py) + 0.5) / float(height)) * nominal_height
    x = u - optical_center_x
    y = -(v - optical_center_y)
    radius = wp.sqrt(x * x + y * y)
    max_theta = wp.min(max_fov * 0.5, wp.float32(math.pi))
    theta = _solve_ftheta_theta(
        radius,
        k0,
        k1,
        k2,
        k3,
        k4,
        max_theta,
    )
    ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space


@wp.kernel(enable_backward=False)
def compute_camera_rays_fisheye_kannala_brandt_kernel(
    width: int,
    height: int,
    nominal_width: wp.float32,
    nominal_height: wp.float32,
    optical_center_x: wp.float32,
    optical_center_y: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    max_fov: wp.float32,
    camera_index: int,
    out_rays: wp.array4d[wp.vec3f],
):
    py, px = wp.tid()
    u = ((float(px) + 0.5) / float(width)) * nominal_width
    v = ((float(py) + 0.5) / float(height)) * nominal_height
    x = u - optical_center_x
    y = -(v - optical_center_y)
    radius = wp.sqrt(x * x + y * y)
    max_theta = wp.min(max_fov * 0.5, wp.float32(math.pi))
    theta = _solve_kannala_brandt_k3_theta(
        radius,
        k0,
        k1,
        k2,
        k3,
        max_theta,
    )
    ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def compute_camera_rays(
    projection: CameraProjection,
    width: int,
    height: int,
    out: wp.array | None = None,
    device=None,
) -> wp.array:
    """Generates a camera-space ray bundle for a projection descriptor.

    Args:
        projection: The projection descriptor.
        width: Image width [px].
        height: Image height [px].
        out: Optional preallocated output of shape [height, width, 2], dtype ``wp.vec3f``.
        device: Warp device for the output array.

    Returns:
        Rays of shape [height, width, 2]: index 0 origins [m], index 1 normalized directions.
    """
    if isinstance(projection, CameraCustomRays):
        if (width, height) != projection.resolution:
            raise ValueError(
                f"CameraCustomRays has fixed resolution {projection.resolution}, requested ({width}, {height})"
            )
        return projection.rays

    # Single allocation shared by all parametric branches below.
    bundle4d = wp.zeros((1, height, width, 2), dtype=wp.vec3f, device=device)

    if isinstance(projection, CameraPinhole):
        wp.launch(
            kernel=compute_camera_rays_pinhole_from_aperture_kernel,
            dim=(1, height, width),
            inputs=[
                width,
                height,
                wp.array([projection.focal_length], dtype=wp.float32, device=device),
                wp.array([projection.horizontal_aperture], dtype=wp.float32, device=device),
                wp.array([projection.vertical_aperture], dtype=wp.float32, device=device),
                wp.array([projection.horizontal_aperture_offset], dtype=wp.float32, device=device),
                wp.array([projection.vertical_aperture_offset], dtype=wp.float32, device=device),
                0,
                bundle4d,
            ],
            device=device,
        )
    elif isinstance(projection, CameraFisheyeOpenCV):
        # Calibration size falls back to render size when None.
        calib_w = wp.float32(width if projection.image_width is None else projection.image_width)
        calib_h = wp.float32(height if projection.image_height is None else projection.image_height)
        wp.launch(
            kernel=compute_camera_rays_fisheye_opencv_kernel,
            dim=(height, width),
            inputs=[
                width,
                height,
                calib_w,
                calib_h,
                wp.float32(projection.fx),
                wp.float32(projection.fy),
                wp.float32(projection.cx),
                wp.float32(projection.cy),
                wp.float32(projection.k1),
                wp.float32(projection.k2),
                wp.float32(projection.k3),
                wp.float32(projection.k4),
                wp.float32(projection.max_fov),
                0,
                bundle4d,
            ],
            device=device,
        )
    elif isinstance(projection, CameraFisheyeFTheta):
        # Calibration size falls back to render size when None.
        calib_w = wp.float32(width if projection.nominal_width is None else projection.nominal_width)
        calib_h = wp.float32(height if projection.nominal_height is None else projection.nominal_height)
        wp.launch(
            kernel=compute_camera_rays_fisheye_ftheta_kernel,
            dim=(height, width),
            inputs=[
                width,
                height,
                calib_w,
                calib_h,
                wp.float32(projection.optical_center_x),
                wp.float32(projection.optical_center_y),
                wp.float32(projection.k0),
                wp.float32(projection.k1),
                wp.float32(projection.k2),
                wp.float32(projection.k3),
                wp.float32(projection.k4),
                wp.float32(projection.max_fov),
                0,
                bundle4d,
            ],
            device=device,
        )
    elif isinstance(projection, CameraFisheyeKannalaBrandt):
        # Calibration size falls back to render size when None.
        calib_w = wp.float32(width if projection.nominal_width is None else projection.nominal_width)
        calib_h = wp.float32(height if projection.nominal_height is None else projection.nominal_height)
        wp.launch(
            kernel=compute_camera_rays_fisheye_kannala_brandt_kernel,
            dim=(height, width),
            inputs=[
                width,
                height,
                calib_w,
                calib_h,
                wp.float32(projection.optical_center_x),
                wp.float32(projection.optical_center_y),
                wp.float32(projection.k0),
                wp.float32(projection.k1),
                wp.float32(projection.k2),
                wp.float32(projection.k3),
                wp.float32(projection.max_fov),
                0,
                bundle4d,
            ],
            device=device,
        )
    else:
        raise TypeError(f"Unsupported projection type: {type(projection).__name__}")

    rays = bundle4d.reshape((height, width, 2))
    if out is not None:
        wp.copy(out, rays)
        return out
    return rays


# ---------------------------------------------------------------------------
# World-transform evaluation
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _eval_camera_world_xforms_kernel(
    camera_transform: wp.array[wp.transform],
    camera_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    out_xforms: wp.array[wp.transform],
):
    tid = wp.tid()
    body = camera_body[tid]
    if body >= 0:
        out_xforms[tid] = body_q[body] * camera_transform[tid]
    else:
        out_xforms[tid] = camera_transform[tid]


def eval_camera_world_xforms(model, state=None, out: wp.array | None = None) -> wp.array:
    """Evaluates the current world transform of every camera.

    Args:
        model: The :class:`~newton.Model` holding the cameras.
        state: Optional :class:`~newton.State`; its ``body_q`` is used for
            body-attached cameras. Falls back to ``model.body_q``.
        out: Optional preallocated output, shape [camera_count].

    Returns:
        Camera-to-world transforms [m, unit quaternion], shape [camera_count].
    """
    # Early return for zero cameras to avoid allocating when unnecessary.
    if model.camera_count == 0:
        if out is None:
            return wp.empty(0, dtype=wp.transform, device=model.device)
        else:
            return out

    if out is None:
        out = wp.empty(model.camera_count, dtype=wp.transform, device=model.device)
    # state.body_q is None when the model has no bodies; fall back to model.body_q
    # (an empty array) so the kernel still has a valid array to bind.
    if state is not None and state.body_q is not None:
        body_q = state.body_q
    else:
        body_q = model.body_q
    wp.launch(
        _eval_camera_world_xforms_kernel,
        dim=model.camera_count,
        inputs=[model.camera_transform, model.camera_body, body_q],
        outputs=[out],
        device=model.device,
    )
    return out


def xform_to_pitch_yaw(xform: wp.transform, up_axis: int) -> tuple[wp.vec3, float, float]:
    """Converts a camera world transform to viewport position, pitch, and yaw.

    This is the inverse of the viewer :class:`~newton._src.viewer.camera.Camera`
    orientation convention: given a camera-to-world transform whose camera
    space is -Z forward / +Y up, returns the position and the pitch/yaw angles
    used by the viewport camera.

    Args:
        xform: Camera-to-world transform (-Z forward, +Y up camera space).
        up_axis: World up axis (0=X, 1=Y, 2=Z).

    Returns:
        Tuple of (position [m], pitch [deg], yaw [deg]).
    """
    q = wp.transform_get_rotation(xform)
    forward = wp.quat_rotate(q, wp.vec3(0.0, 0.0, -1.0))
    fx, fy, fz = float(forward[0]), float(forward[1]), float(forward[2])
    if up_axis == 0:  # X up
        pitch = math.asin(max(-1.0, min(1.0, fx)))
        yaw = math.atan2(fz, fy)
    elif up_axis == 1:  # Y up
        pitch = math.asin(max(-1.0, min(1.0, fy)))
        yaw = math.atan2(fz, fx)
    else:  # Z up
        pitch = math.asin(max(-1.0, min(1.0, fz)))
        yaw = math.atan2(fy, fx)
    return wp.transform_get_translation(xform), math.degrees(pitch), math.degrees(yaw)
