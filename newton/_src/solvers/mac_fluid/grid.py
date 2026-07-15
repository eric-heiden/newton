# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Staggered MAC-grid storage and interpolation helpers.

Layout
------

The fluid domain is a dense uniform grid of ``nx * ny * nz`` cells with cell
size ``dx`` and lower corner ``origin``. Quantities are stored staggered:

- pressure and cell labels at cell centers, shape ``(nx, ny, nz)``
- x-velocity ``u`` at x-faces, shape ``(nx + 1, ny, nz)``
- y-velocity ``v`` at y-faces, shape ``(nx, ny + 1, nz)``
- z-velocity ``w`` at z-faces, shape ``(nx, ny, nz + 1)``

The world position of cell center ``(i, j, k)`` is
``origin + dx * (i + 0.5, j + 0.5, k + 0.5)``; the position of x-face
``(i, j, k)`` is ``origin + dx * (i, j + 0.5, k + 0.5)`` and similarly for the
other axes.

Cell labels partition the domain: ``CELL_FLUID`` marks fluid cells,
``CELL_STATIC_SOLID`` marks static solids (domain walls and static shapes),
and values ``>= 0`` mark cells inside a rigid body, storing the body index.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "CELL_FLUID",
    "CELL_STATIC_SOLID",
    "MACGridData",
]

# Cell label values (cell_label array). Values >= 0 store a rigid body index.
CELL_FLUID = wp.constant(-2)
CELL_STATIC_SOLID = wp.constant(-1)


@wp.func
def cell_center(origin: wp.vec3, dx: float, i: int, j: int, k: int) -> wp.vec3:
    """World position of the center of cell ``(i, j, k)``."""
    return origin + dx * wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)


@wp.func
def face_position(origin: wp.vec3, dx: float, axis: int, i: int, j: int, k: int) -> wp.vec3:
    """World position of face ``(i, j, k)`` of the given axis (0=x, 1=y, 2=z)."""
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    if axis == 0:
        p[0] = float(i)
    elif axis == 1:
        p[1] = float(j)
    else:
        p[2] = float(k)
    return origin + dx * p


@wp.func
def is_solid_cell(cell_label: wp.array3d[wp.int32], i: int, j: int, k: int) -> bool:
    """Whether cell ``(i, j, k)`` is solid; cells outside the grid count as solid walls."""
    nx = cell_label.shape[0]
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
        return True
    return cell_label[i, j, k] != CELL_FLUID


@wp.func
def cell_label_at(cell_label: wp.array3d[wp.int32], i: int, j: int, k: int) -> int:
    """Label of cell ``(i, j, k)``; cells outside the grid are static solid."""
    nx = cell_label.shape[0]
    ny = cell_label.shape[1]
    nz = cell_label.shape[2]
    if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
        return CELL_STATIC_SOLID
    return cell_label[i, j, k]


@wp.func
def _sample_component(field: wp.array3d[float], lx: float, ly: float, lz: float) -> float:
    """Clamped trilinear interpolation of one staggered component.

    ``(lx, ly, lz)`` are continuous sample coordinates in the component's own
    index space (i.e. sample point p maps to array index p exactly).
    """
    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]

    lx = wp.clamp(lx, 0.0, float(nx - 1))
    ly = wp.clamp(ly, 0.0, float(ny - 1))
    lz = wp.clamp(lz, 0.0, float(nz - 1))

    i0 = wp.clamp(int(lx), 0, wp.max(nx - 2, 0))
    j0 = wp.clamp(int(ly), 0, wp.max(ny - 2, 0))
    k0 = wp.clamp(int(lz), 0, wp.max(nz - 2, 0))

    fx = lx - float(i0)
    fy = ly - float(j0)
    fz = lz - float(k0)

    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)

    c00 = field[i0, j0, k0] * (1.0 - fx) + field[i1, j0, k0] * fx
    c10 = field[i0, j1, k0] * (1.0 - fx) + field[i1, j1, k0] * fx
    c01 = field[i0, j0, k1] * (1.0 - fx) + field[i1, j0, k1] * fx
    c11 = field[i0, j1, k1] * (1.0 - fx) + field[i1, j1, k1] * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def sample_u(u: wp.array3d[float], origin: wp.vec3, dx: float, pos: wp.vec3) -> float:
    """Interpolate the x-velocity component at world position ``pos``."""
    g = (pos - origin) / dx
    return _sample_component(u, g[0], g[1] - 0.5, g[2] - 0.5)


@wp.func
def sample_v(v: wp.array3d[float], origin: wp.vec3, dx: float, pos: wp.vec3) -> float:
    """Interpolate the y-velocity component at world position ``pos``."""
    g = (pos - origin) / dx
    return _sample_component(v, g[0] - 0.5, g[1], g[2] - 0.5)


@wp.func
def sample_w(w: wp.array3d[float], origin: wp.vec3, dx: float, pos: wp.vec3) -> float:
    """Interpolate the z-velocity component at world position ``pos``."""
    g = (pos - origin) / dx
    return _sample_component(w, g[0] - 0.5, g[1] - 0.5, g[2])


@wp.func
def sample_velocity(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    origin: wp.vec3,
    dx: float,
    pos: wp.vec3,
) -> wp.vec3:
    """Interpolate the full MAC velocity at world position ``pos``."""
    return wp.vec3(
        sample_u(u, origin, dx, pos),
        sample_v(v, origin, dx, pos),
        sample_w(w, origin, dx, pos),
    )


class MACGridData:
    """Fixed-size device storage for one MAC grid.

    All arrays are allocated once at construction so that solver steps are
    free of allocations and compatible with CUDA graph capture.
    """

    def __init__(self, resolution: tuple[int, int, int], device):
        nx, ny, nz = (int(r) for r in resolution)
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(f"Grid resolution must be positive, got {resolution}")
        self.shape = (nx, ny, nz)

        with wp.ScopedDevice(device):
            # velocity components and double buffers
            self.u = wp.zeros((nx + 1, ny, nz), dtype=float)
            self.v = wp.zeros((nx, ny + 1, nz), dtype=float)
            self.w = wp.zeros((nx, ny, nz + 1), dtype=float)
            self.u_tmp = wp.zeros_like(self.u)
            self.v_tmp = wp.zeros_like(self.v)
            self.w_tmp = wp.zeros_like(self.w)

            # checkpoint buffers for coupled-iteration restarts
            self.u_checkpoint = wp.zeros_like(self.u)
            self.v_checkpoint = wp.zeros_like(self.v)
            self.w_checkpoint = wp.zeros_like(self.w)

            # prescribed solid velocity at faces (valid where a neighbor cell is solid)
            self.u_solid = wp.zeros_like(self.u)
            self.v_solid = wp.zeros_like(self.v)
            self.w_solid = wp.zeros_like(self.w)

            # cell-centered fields
            self.cell_label = wp.zeros((nx, ny, nz), dtype=wp.int32)
            self.cell_sdf = wp.zeros((nx, ny, nz), dtype=float)
            self.pressure = wp.zeros((nx, ny, nz), dtype=float)
            self.divergence = wp.zeros((nx, ny, nz), dtype=float)

    # NOTE: velocity buffers are never reference-swapped; stages write to the
    # ``*_tmp`` buffers and copy back so that a captured CUDA graph replays
    # correctly with stable array identities.
