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
    """OpenCV fisheye model: r = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8).

    Pixel-unit intrinsics are interpreted at the render resolution, matching
    the existing ``compute_camera_rays_fisheye_opencv`` helper semantics.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    max_fov: float = math.pi
    """Maximum field of view [rad]; rays beyond it are marked invalid."""


@dataclass(frozen=True, kw_only=True)
class CameraFisheyeFTheta(CameraProjection):
    """F-theta fisheye model: r = k0 + k1*theta + k2*theta^2 + k3*theta^3 + k4*theta^4."""

    optical_center_x: float
    optical_center_y: float
    k0: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    max_fov: float = math.pi


@dataclass(frozen=True, kw_only=True)
class CameraFisheyeKannalaBrandt(CameraProjection):
    """Kannala-Brandt K3 fisheye model: r = k0*theta + k1*theta^3 + k2*theta^5 + k3*theta^7."""

    optical_center_x: float
    optical_center_y: float
    k0: float = 1.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    max_fov: float = math.pi


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
