# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrated pinhole camera geometry for :class:`~newton.sensors.SensorCamera`."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import numpy as np
import warp as wp

from ..core.types import Devicelike, Transform, Vec4

# Damped Newton inversion of the distortion polynomial, in normalized image coordinates.
_INVERSION_ITERATIONS = 50
_LINE_SEARCH_ITERATIONS = 12
_CONVERGENCE_TOLERANCE = 1.0e-13
_ACCEPTANCE_TOLERANCE = 1.0e-9
# The fold search covers normalized radii up to 100 (89.4 degrees off the optical axis).
_FOLD_SEARCH_RADIUS = 100.0
_FOLD_SEARCH_SAMPLES = 20001

_COEFFICIENTS = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4")
# OpenCV's distCoeffs order; RealSense coefficients use its first five entries.
_OPENCV_ORDER = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")


def transform_values(xform: Any, name: str = "camera_transform") -> np.ndarray:
    """Seven float64 values ``(px, py, pz, qx, qy, qz, qw)`` of a transform given in any supported form."""
    if isinstance(xform, tuple) and len(xform) == 2:
        values = np.concatenate(
            [np.asarray(xform[0], dtype=np.float64).reshape(-1), np.asarray(xform[1], dtype=np.float64).reshape(-1)]
        )
    else:
        values = np.asarray(xform, dtype=np.float64).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a transform: a position [m] and an xyzw quaternion (7 finite values)")
    if np.linalg.norm(values[3:]) < 1.0e-12:
        raise ValueError(f"{name} quaternion must be nonzero")
    return values


def _pose(camera_transform: Any) -> tuple[np.ndarray, np.ndarray]:
    """Position [m] and float64 rotation matrix of a camera transform (identity for ``None``)."""
    if camera_transform is None:
        return np.zeros(3), np.eye(3)
    values = transform_values(camera_transform)
    x, y, z, w = values[3:] / np.linalg.norm(values[3:])
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    return values[:3], rotation


def _vectors(values: Any, size: int, name: str) -> np.ndarray:
    """``values`` as a float64 array of shape ``(..., size)``."""
    if isinstance(values, wp.array):
        values = values.numpy()
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}), got {result.shape}")
    return result


def _image_size(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive integer [px]")
    if not math.isfinite(value) or value != int(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer [px], got {value}")
    return int(value)


@dataclasses.dataclass(frozen=True)
class Intrinsics:
    """Calibrated pinhole camera: image size, focal lengths, principal point, and lens distortion.

    Maps world points to image coordinates (:meth:`project`) and image
    coordinates to world rays, depths, or plane points (:meth:`unproject`,
    :meth:`unproject_to_depth`, :meth:`unproject_to_plane`) on the host in
    float64 NumPy, and builds the matching ray bundle for
    :meth:`SensorCamera.update() <newton.sensors.SensorCamera.update>` with
    :meth:`compute_camera_rays`, so renders and projections share one lens.

    Image coordinates [px] follow OpenCV: x points right, y points down, and
    the center of the top-left pixel is ``(0, 0)``. Pixel ``(i, j)`` (column
    ``i``, row ``j`` of an image array) is centered at ``(i, j)`` and covers
    ``[i - 0.5, i + 0.5) x [j - 0.5, j + 0.5)``. OpenCV and ROS
    ``CameraInfo`` calibrations use this convention.

    Camera transforms place the :class:`~newton.sensors.SensorCamera` camera
    frame in the world: the camera looks along its local -Z axis with +Y up
    and +X right. An OpenCV or ROS optical frame (+Z forward, +Y down) is that
    frame rotated by 180 degrees about X, so its orientation is
    ``q_optical * wp.quat(1.0, 0.0, 0.0, 0.0)`` in this convention. Use
    :meth:`SensorCamera.compute_camera_transforms_body()
    <newton.sensors.SensorCamera.compute_camera_transforms_body>` for cameras
    mounted on bodies.

    Normalized image coordinates are ``x = (u - cx) / fx`` and
    ``y = (v - cy) / fy``. :attr:`DistortionModel.OPENCV` maps the
    undistorted coordinates of a ray to the distorted coordinates of its
    pixel; :attr:`DistortionModel.INVERSE_BROWN_CONRADY` maps the other way.
    Each direction without a closed form is solved with damped Newton
    iteration.

    Example::

        from newton.sensors import SensorCamera

        camera = SensorCamera.Intrinsics.from_camera_matrix(
            K, D, width=640, height=480, distortion_model="inverse_brown_conrady"
        )
        pixels, forward_depth = camera.project(points, camera_transform)
        on_table = camera.unproject_to_plane(pixels, camera_transform, plane=(0.0, 0.0, 1.0, -0.75))
        rays = camera.compute_camera_rays(device=model.device)
    """

    class DistortionModel(IntEnum):
        """Lens distortion models of :class:`~newton.sensors.SensorCamera.Intrinsics`."""

        OPENCV = 0
        """OpenCV pinhole distortion (``cv2.projectPoints``): rational radial ``k1``-``k6``, tangential ``p1``,
        ``p2``, and thin-prism ``s1``-``s4`` coefficients map undistorted normalized coordinates to distorted ones.
        All-zero coefficients describe an ideal pinhole camera."""

        INVERSE_BROWN_CONRADY = 1
        """RealSense inverse Brown-Conrady model (``RS2_DISTORTION_INVERSE_BROWN_CONRADY``): the Brown-Conrady
        polynomial with ``k1``, ``k2``, ``k3``, ``p1``, and ``p2`` maps distorted normalized coordinates (a recorded
        pixel) to undistorted ones (its ray)."""

    width: int
    """Calibration image width [px]."""
    height: int
    """Calibration image height [px]."""
    fx: float
    """Horizontal focal length [px]."""
    fy: float
    """Vertical focal length [px]."""
    cx: float
    """Principal point x-coordinate [px]."""
    cy: float
    """Principal point y-coordinate [px]."""
    k1: float = 0.0
    """First radial distortion coefficient (numerator)."""
    k2: float = 0.0
    """Second radial distortion coefficient (numerator)."""
    k3: float = 0.0
    """Third radial distortion coefficient (numerator)."""
    k4: float = 0.0
    """First rational radial distortion coefficient (denominator); :attr:`DistortionModel.OPENCV` only."""
    k5: float = 0.0
    """Second rational radial distortion coefficient (denominator); :attr:`DistortionModel.OPENCV` only."""
    k6: float = 0.0
    """Third rational radial distortion coefficient (denominator); :attr:`DistortionModel.OPENCV` only."""
    p1: float = 0.0
    """First tangential distortion coefficient."""
    p2: float = 0.0
    """Second tangential distortion coefficient."""
    s1: float = 0.0
    """First thin-prism distortion coefficient; :attr:`DistortionModel.OPENCV` only."""
    s2: float = 0.0
    """Second thin-prism distortion coefficient; :attr:`DistortionModel.OPENCV` only."""
    s3: float = 0.0
    """Third thin-prism distortion coefficient; :attr:`DistortionModel.OPENCV` only."""
    s4: float = 0.0
    """Fourth thin-prism distortion coefficient; :attr:`DistortionModel.OPENCV` only."""
    distortion_model: DistortionModel = DistortionModel.OPENCV
    """Lens distortion model. Strings name a member case-insensitively, e.g. ``"inverse_brown_conrady"``."""

    def __post_init__(self):
        model = self.distortion_model
        if isinstance(model, str):
            try:
                model = Intrinsics.DistortionModel[model.upper()]
            except KeyError:
                names = ", ".join(m.name.lower() for m in Intrinsics.DistortionModel)
                raise ValueError(f"distortion_model must be one of {names}, got {model!r}") from None
        else:
            model = Intrinsics.DistortionModel(model)
        object.__setattr__(self, "distortion_model", model)
        object.__setattr__(self, "width", _image_size("width", self.width))
        object.__setattr__(self, "height", _image_size("height", self.height))
        for name in ("fx", "fy", "cx", "cy", *_COEFFICIENTS):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value}")
            object.__setattr__(self, name, float(value))
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("fx and fy must be positive")
        if model == Intrinsics.DistortionModel.INVERSE_BROWN_CONRADY:
            extra = [name for name in ("k4", "k5", "k6", "s1", "s2", "s3", "s4") if getattr(self, name) != 0.0]
            if extra:
                raise ValueError(f"inverse_brown_conrady takes k1, k2, k3, p1, p2; got nonzero {', '.join(extra)}")

    @classmethod
    def from_camera_matrix(
        cls,
        camera_matrix: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        distortion: Sequence[float] | np.ndarray | None = None,
        *,
        width: int,
        height: int,
        distortion_model: DistortionModel | str = DistortionModel.OPENCV,
    ) -> Intrinsics:
        """Create intrinsics from a camera matrix and OpenCV-ordered distortion coefficients.

        Args:
            camera_matrix: Camera matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``
                [px], as a 3x3 array or nine row-major values. Skew must be zero.
            distortion: Distortion coefficients in OpenCV order
                ``(k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]])``: 4, 5, 8,
                or 12 values. RealSense intrinsics list their five coefficients
                in the same order. If ``None``, the camera has no distortion.
            width: Calibration image width [px].
            height: Calibration image height [px].
            distortion_model: Model the coefficients belong to.

        Returns:
            The camera intrinsics.

        Raises:
            ValueError: If the matrix is not a zero-skew camera matrix, the
                coefficient count is unsupported, or a value is invalid.
        """
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        if matrix.size != 9:
            raise ValueError(f"camera_matrix must hold 9 values, got {matrix.size}")
        matrix = matrix.reshape(3, 3)
        if matrix[0, 1] != 0.0 or matrix[1, 0] != 0.0 or not np.array_equal(matrix[2], [0.0, 0.0, 1.0]):
            raise ValueError("camera_matrix must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (zero skew)")
        coefficients = np.zeros(0) if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1)
        if coefficients.size not in (0, 4, 5, 8, 12):
            raise ValueError(
                "distortion must hold 4, 5, 8, or 12 OpenCV coefficients "
                f"(k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]]), got {coefficients.size}"
            )
        named = dict(zip(_OPENCV_ORDER, coefficients.tolist(), strict=False))
        return cls(
            width,
            height,
            float(matrix[0, 0]),
            float(matrix[1, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
            **named,
            distortion_model=distortion_model,
        )

    @classmethod
    def from_fov(cls, width: int, height: int, camera_fov: float) -> Intrinsics:
        """Create an ideal pinhole camera from its vertical field of view.

        The principal point is the image center and pixels are square. Its
        rays match :meth:`SensorCamera.compute_camera_rays_pinhole()
        <newton.sensors.SensorCamera.compute_camera_rays_pinhole>` with the
        same ``camera_fov``.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fov: Vertical field of view [rad], in ``(0, pi)``.

        Returns:
            The camera intrinsics.
        """
        if not 0.0 < float(camera_fov) < math.pi:
            raise ValueError(f"camera_fov must be in (0, pi) radians, got {float(camera_fov)}")
        focal = 0.5 * float(height) / math.tan(0.5 * float(camera_fov))
        return cls(width, height, focal, focal, 0.5 * (float(width) - 1.0), 0.5 * (float(height) - 1.0))

    def resize(self, width: int, height: int) -> Intrinsics:
        """Return the intrinsics of this camera for its image resampled to ``width`` x ``height``.

        Focal lengths scale with the image size, and the principal point keeps
        its position relative to the image edges. Distortion coefficients act
        on normalized coordinates and do not change.

        Args:
            width: New image width [px].
            height: New image height [px].

        Returns:
            The resampled camera intrinsics.
        """
        width, height = _image_size("width", width), _image_size("height", height)
        scale_x, scale_y = width / self.width, height / self.height
        return dataclasses.replace(
            self,
            width=width,
            height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=(self.cx + 0.5) * scale_x - 0.5,
            cy=(self.cy + 0.5) * scale_y - 0.5,
        )

    def project(
        self,
        points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | wp.array[wp.vec3],
        camera_transform: Transform | Sequence[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to image coordinates.

        Points behind the camera, outside the range where the distortion model
        is one-to-one, or (for :attr:`DistortionModel.INVERSE_BROWN_CONRADY`)
        without a distorted preimage get NaN image coordinates. Coordinates of
        points outside the field of view are returned as computed, so check
        them against the image bounds ``[-0.5, width - 0.5) x [-0.5, height - 0.5)``.

        Args:
            points: World points [m], shape ``(..., 3)``.
            camera_transform: World pose of the camera as a transform, a
                ``(position, quaternion)`` pair, or seven values
                ``(px, py, pz, qx, qy, qz, qw)`` [m]. The camera looks along
                its local -Z axis with +Y up. If ``None``, ``points`` are in the
                camera frame.

        Returns:
            Image coordinates [px], shape ``(..., 2)``, and forward depth [m]
            (distance along the viewing axis, negative behind the camera),
            shape ``(...)``.
        """
        points = _vectors(points, 3, "points")
        position, rotation = _pose(camera_transform)
        local = (points - position) @ rotation
        forward_depth = -local[..., 2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ahead = forward_depth > 0.0
            safe_depth = np.where(ahead, forward_depth, 1.0)
            x = np.where(ahead, local[..., 0] / safe_depth, np.nan)
            y = np.where(ahead, -local[..., 1] / safe_depth, np.nan)
            if self.distortion_model == Intrinsics.DistortionModel.OPENCV:
                x, y = self._distort_checked(x, y)
            else:
                x, y = self._solve_distortion(x, y)
            pixels = np.stack([self.fx * x + self.cx, self.fy * y + self.cy], axis=-1)
        return pixels, forward_depth

    def unproject(
        self,
        pixels: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        camera_transform: Transform | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the world-space ray direction through each image coordinate.

        Each ray starts at the camera position. Pixels whose distortion
        inverse cannot be verified get NaN directions.

        Args:
            pixels: Image coordinates [px], shape ``(..., 2)``.
            camera_transform: World pose of the camera, as in :meth:`project`.
                If ``None``, directions are in the camera frame.

        Returns:
            Unit ray directions, shape ``(..., 3)``.
        """
        _, rotation = _pose(camera_transform)
        local = self._camera_directions(pixels)
        with np.errstate(invalid="ignore"):
            local = local / np.linalg.norm(local, axis=-1, keepdims=True)
        return local @ rotation.T

    def unproject_to_depth(
        self,
        pixels: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        camera_transform: Transform | Sequence[float] | np.ndarray | None = None,
        *,
        forward_depth: float | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Compute the world points seen at image coordinates with known forward depths.

        This inverts :meth:`project`. Forward depth is the distance along the
        viewing axis, as in a :class:`~newton.sensors.SensorCamera` forward-depth
        image or a librealsense depth frame.

        Args:
            pixels: Image coordinates [px], shape ``(..., 2)``.
            camera_transform: World pose of the camera, as in :meth:`project`.
                If ``None``, points are in the camera frame.
            forward_depth: Forward depth [m] per image coordinate, broadcastable
                to shape ``(...)``.

        Returns:
            World points [m], shape ``(..., 3)``; NaN where the distortion
            inverse cannot be verified.
        """
        position, rotation = _pose(camera_transform)
        local = self._camera_directions(pixels)
        depth = np.broadcast_to(np.asarray(forward_depth, dtype=np.float64), local.shape[:-1])
        return position + (local * depth[..., None]) @ rotation.T

    def unproject_to_plane(
        self,
        pixels: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        camera_transform: Transform | Sequence[float] | np.ndarray | None = None,
        *,
        plane: Vec4 | Sequence[float],
    ) -> np.ndarray:
        """Intersect the rays through image coordinates with a plane.

        Args:
            pixels: Image coordinates [px], shape ``(..., 2)``.
            camera_transform: World pose of the camera, as in :meth:`project`.
                If ``None``, the plane and the result are in the camera frame.
            plane: Plane equation ``(a, b, c, d)`` with ``a*x + b*y + c*z + d = 0``,
                as in :meth:`~newton.ModelBuilder.add_shape_plane`;
                ``(0.0, 0.0, 1.0, -h)`` is the plane ``z = h`` [m].

        Returns:
            World points [m] where the rays meet the plane, shape ``(..., 3)``;
            NaN for rays parallel to the plane, pointing away from it, or whose
            distortion inverse cannot be verified.
        """
        coefficients = np.asarray(plane, dtype=np.float64).reshape(-1)
        if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
            raise ValueError("plane must be four finite values (a, b, c, d) with a*x + b*y + c*z + d = 0")
        normal = coefficients[:3]
        if np.linalg.norm(normal) == 0.0:
            raise ValueError("plane normal (a, b, c) must be nonzero")
        position, _ = _pose(camera_transform)
        directions = self.unproject(pixels, camera_transform)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = -(normal @ position + coefficients[3]) / (directions @ normal)
            hit = np.isfinite(distance) & (distance > 0.0)
            return np.where(hit[..., None], position + distance[..., None] * directions, np.nan)

    def compute_camera_rays(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute the camera-space ray bundle of this camera for :meth:`SensorCamera.update()
        <newton.sensors.SensorCamera.update>`.

        Pixel ``(i, j)`` of the bundle looks through image coordinates
        ``(i, j)`` of :meth:`resize` ``(width, height)``, so points rendered at
        a pixel project to its center. :attr:`DistortionModel.OPENCV` uses
        :meth:`SensorCamera.compute_camera_rays_pinhole_opencv()
        <newton.sensors.SensorCamera.compute_camera_rays_pinhole_opencv>`.
        Pixels that :meth:`unproject` maps to NaN, because the distortion
        inverse cannot be verified or lies past the fold radius, receive a zero
        direction.

        Args:
            width: Output image width [px]. If ``None``, uses :attr:`width`.
            height: Output image height [px]. If ``None``, uses :attr:`height`.
            out_rays: Optional output buffer, shape ``(height, width, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, 2)`` of ``vec3f``.
        """
        from .sensor_camera import SensorCamera, _validate_camera_ray_output  # noqa: PLC0415
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width = self.width if width is None else _image_size("width", width)
        height = self.height if height is None else _image_size("height", height)
        # The ray kernels sample pixel i at calibration coordinate (i + 0.5) * scale, which treats
        # integer coordinates as pixel corners; moving the principal point by half a pixel samples
        # (i + 0.5) * scale - 0.5, the pixel center in OpenCV coordinates.
        cx, cy = self.cx + 0.5, self.cy + 0.5
        if self.distortion_model == Intrinsics.DistortionModel.OPENCV:
            out_rays = SensorCamera.compute_camera_rays_pinhole_opencv(
                width,
                height,
                self.fx,
                self.fy,
                cx,
                cy,
                image_width=float(self.width),
                image_height=float(self.height),
                **{name: getattr(self, name) for name in _COEFFICIENTS},
                out_rays=out_rays,
                device=device,
            )
            if math.isfinite(self._fold_radius):
                wp.launch(
                    kernel=camera_utils.mask_folded_camera_rays_kernel,
                    dim=(height, width),
                    inputs=[self._fold_radius**2, out_rays],
                    device=out_rays.device,
                )
            return out_rays
        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        wp.launch(
            kernel=camera_utils.compute_camera_rays_pinhole_inverse_brown_conrady_kernel,
            dim=(height, width),
            inputs=[
                width,
                height,
                float(self.width),
                float(self.height),
                self.fx,
                self.fy,
                cx,
                cy,
                self.k1,
                self.k2,
                self.k3,
                self.p1,
                self.p2,
                out_rays,
            ],
            device=device,
        )
        return out_rays

    def _camera_directions(self, pixels: Any) -> np.ndarray:
        """Camera-frame ray directions with unit forward depth, shape ``(..., 3)``."""
        pixels = _vectors(pixels, 2, "pixels")
        x = (pixels[..., 0] - self.cx) / self.fx
        y = (pixels[..., 1] - self.cy) / self.fy
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.distortion_model == Intrinsics.DistortionModel.OPENCV:
                x, y = self._solve_distortion(x, y)
            else:
                x, y = self._distort(x, y)
        return np.stack([x, -y, -np.ones_like(x)], axis=-1)

    def _radial(self, r2: np.ndarray) -> np.ndarray:
        numerator = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        denominator = 1.0 + r2 * (self.k4 + r2 * (self.k5 + r2 * self.k6))
        return numerator / denominator

    def _distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """OpenCV's distortion polynomial at normalized coordinates (y down)."""
        r2 = x * x + y * y
        radial = self._radial(r2)
        return (
            x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x) + self.s1 * r2 + self.s2 * r2 * r2,
            y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y + self.s3 * r2 + self.s4 * r2 * r2,
        )

    def _distort_jacobian(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        r2 = x * x + y * y
        numerator = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        denominator = 1.0 + r2 * (self.k4 + r2 * (self.k5 + r2 * self.k6))
        radial = numerator / denominator
        numerator_slope = self.k1 + r2 * (2.0 * self.k2 + 3.0 * self.k3 * r2)
        denominator_slope = self.k4 + r2 * (2.0 * self.k5 + 3.0 * self.k6 * r2)
        radial_slope = (numerator_slope * denominator - numerator * denominator_slope) / (denominator * denominator)
        radial_x, radial_y = 2.0 * x * radial_slope, 2.0 * y * radial_slope
        p1, p2, s1, s2, s3, s4 = self.p1, self.p2, self.s1, self.s2, self.s3, self.s4
        return (
            radial + x * radial_x + 2.0 * p1 * y + 6.0 * p2 * x + 2.0 * s1 * x + 4.0 * s2 * r2 * x,
            x * radial_y + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * s1 * y + 4.0 * s2 * r2 * y,
            y * radial_x + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * s3 * x + 4.0 * s4 * r2 * x,
            radial + y * radial_y + 6.0 * p1 * y + 2.0 * p2 * x + 2.0 * s3 * y + 4.0 * s4 * r2 * y,
        )

    def _radial_slope(self, r2: np.ndarray) -> np.ndarray:
        """Derivative of ``r * radial(r^2)`` with respect to ``r``; NaN past a pole of the rational model."""
        numerator = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        denominator = 1.0 + r2 * (self.k4 + r2 * (self.k5 + r2 * self.k6))
        numerator_slope = self.k1 + r2 * (2.0 * self.k2 + 3.0 * self.k3 * r2)
        denominator_slope = self.k4 + r2 * (2.0 * self.k5 + 3.0 * self.k6 * r2)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = numerator / denominator + 2.0 * r2 * (
                numerator_slope * denominator - numerator * denominator_slope
            ) / (denominator * denominator)
        return np.where(denominator > 0.0, slope, np.nan)

    @functools.cached_property
    def _fold_radius(self) -> float:
        """Normalized radius where ``r * radial(r^2)`` stops increasing; past it the polynomial folds back."""
        radii = np.linspace(0.0, _FOLD_SEARCH_RADIUS, _FOLD_SEARCH_SAMPLES)
        folded = ~(self._radial_slope(radii * radii) > 0.0)
        if not folded.any():
            return math.inf
        index = int(np.argmax(folded))
        low, high = float(radii[index - 1]), float(radii[index])
        for _ in range(60):
            middle = 0.5 * (low + high)
            if self._radial_slope(np.array(middle * middle)) > 0.0:
                low = middle
            else:
                high = middle
        return low

    def _distort_checked(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`_distort`, NaN past the fold radius."""
        inside = x * x + y * y < self._fold_radius**2
        xd, yd = self._distort(x, y)
        return np.where(inside, xd, np.nan), np.where(inside, yd, np.nan)

    def _solve_distortion(self, target_x: np.ndarray, target_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates inside the fold radius that :meth:`_distort` maps to the targets; NaN where none is found."""
        target_x, target_y = np.broadcast_arrays(np.asarray(target_x, dtype=np.float64), target_y)
        x, y = target_x.copy(), target_y.copy()
        stalled = ~(np.isfinite(x) & np.isfinite(y))
        with np.errstate(all="ignore"):
            for _ in range(_INVERSION_ITERATIONS):
                distorted_x, distorted_y = self._distort(x, y)
                residual_x, residual_y = distorted_x - target_x, distorted_y - target_y
                residual = residual_x * residual_x + residual_y * residual_y
                active = ~stalled & (residual > _CONVERGENCE_TOLERANCE**2)
                if not active.any():
                    break
                j00, j01, j10, j11 = self._distort_jacobian(x, y)
                determinant = j00 * j11 - j01 * j10
                step_x = (j11 * residual_x - j01 * residual_y) / determinant
                step_y = (j00 * residual_y - j10 * residual_x) / determinant
                stalled |= active & ~(np.isfinite(step_x) & np.isfinite(step_y))
                pending = active & ~stalled
                scale = 1.0
                for _ in range(_LINE_SEARCH_ITERATIONS):
                    candidate_x, candidate_y = x - scale * step_x, y - scale * step_y
                    candidate_dx, candidate_dy = self._distort(candidate_x, candidate_y)
                    candidate = (candidate_dx - target_x) ** 2 + (candidate_dy - target_y) ** 2
                    accept = pending & (candidate < residual)
                    x, y = np.where(accept, candidate_x, x), np.where(accept, candidate_y, y)
                    pending &= ~accept
                    if not pending.any():
                        break
                    scale *= 0.5
                stalled |= pending
            distorted_x, distorted_y = self._distort(x, y)
            residual = np.hypot(distorted_x - target_x, distorted_y - target_y)
            valid = (residual <= _ACCEPTANCE_TOLERANCE) & (x * x + y * y < self._fold_radius**2)
        return np.where(valid, x, np.nan), np.where(valid, y, np.nan)
