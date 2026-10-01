# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Work around the shared CUDA HashGrid descriptor in Warp 1.17.

Warp captures a host-to-device copy from one static descriptor shared by all
float32 grids. Building another grid changes that source, so graph replay can
read the wrong grid or freed memory. Snapshot the initialized descriptor once
and restore it on-device after each build. Storage ownership and point counts
must remain fixed for the captured graph's lifetime. The radius may change.

Remove this compatibility helper after Warp fixes the shared host descriptor:
https://github.com/NVIDIA/warp/blob/v1.17.0/warp/native/hashgrid.cpp
"""

import warp as wp


@wp.func_native("""
    static_assert(sizeof(wp::HashGrid_t<float>) <= 16 * sizeof(uint64_t),
                  "XPBD fluid hash-grid descriptor storage is too small");
    auto* descriptor = reinterpret_cast<wp::HashGrid_t<float>*>(grid);
    auto* saved = reinterpret_cast<wp::HashGrid_t<float>*>(storage.data);
    if (restore) {
        *descriptor = *saved;
        descriptor->cell_width = radius;
        descriptor->cell_width_inv = 1.0f / radius;
    } else {
        *saved = *descriptor;
    }
""")
def _copy_hash_grid_descriptor(grid: wp.uint64, storage: wp.array[wp.uint64], restore: bool, radius: float):
    """Keep the grid descriptor owned by its solver during CUDA graph replay."""
    ...


@wp.kernel
def copy_hash_grid_descriptor(
    grid: wp.uint64,
    storage: wp.array[wp.uint64],
    restore: bool,
    radius: float,
):
    _copy_hash_grid_descriptor(grid, storage, restore, radius)
