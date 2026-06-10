# SPDX-FileCopyrightText: Copyright (c) 2026 Eric Heiden
# SPDX-License-Identifier: Apache-2.0

"""Run one variant over the captured meshes.

Steps:
  1. ``apply_variant.py --variant <v>`` is run before this (separately).
  2. We set V2's ``wp.config.deterministic`` flag before importing newton.inertia.
  3. We load ``meshes.npz`` (262 records) and call ``compute_inertia_mesh``
     5 times per mesh, capturing the f64 outputs (mass, com, inertia, volume).
  4. We measure per-call wall-clock with cuda sync, and peak GPU MiB via nvidia-smi.
  5. Aggregated results are written to ``results/results_<variant>.json``.

The output schema (per mesh × per rerun):
    {
      "robot": str, "shape_index": int, "num_tris": int, "is_solid": bool,
      "ms": [t0, t1, t2, t3, t4],
      "outputs": [{"mass": float, "com": [x,y,z], "I": [[..],[..],[..]],
                   "volume": float, "hash": "..."}, ...],
    }
And top-level: variant, warp_version, gpu, started_at, finished_at,
host_mem_peak_mib, env_snapshot.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    MeshRecord,
    env_snapshot,
    gpu_mem_used_mib,
    hash_bytes,
    load_meshes,
    write_results,
)


def _enable_v2_determinism() -> str:
    """For V2 we must set wp.config.deterministic BEFORE importing inertia.

    Returns the actual mode used:
      - "warp_pr1355_native" if the installed Warp exposes DeterministicMode.RUN_TO_RUN
      - "emulation" if we are falling back to the V2 inertia.py snapshot
    """
    import warp as wp  # noqa: WPS433

    mode_enum = getattr(wp.config, "DeterministicMode", None)
    if mode_enum is None:
        print(
            "V2: wp.config.DeterministicMode missing — using V2 inertia.py snapshot "
            "(emulation of PR #1355 behavior). Real Warp build would require CUDA 12 "
            "toolkit which is not installed here.",
            file=sys.stderr,
        )
        return "emulation"
    wp.config.deterministic = mode_enum.RUN_TO_RUN
    if getattr(wp.config, "deterministic_debug", None) is not None:
        wp.config.deterministic_debug = False
    print(
        f"V2: wp.config.deterministic = {wp.config.deterministic} (native PR #1355 build)",
        file=sys.stderr,
    )
    return "warp_pr1355_native"


def _call_compute(
    m: MeshRecord,
) -> tuple[float, tuple[float, float, float], list[list[float]], float]:
    """Invoke compute_inertia_mesh on one MeshRecord; return (mass, com, I, V)."""
    from newton._src.geometry.inertia import compute_inertia_mesh

    mass, com, I, V = compute_inertia_mesh(
        m.density,
        m.vertices.astype(np.float32, copy=False),
        m.indices.astype(np.int32, copy=False),
        m.is_solid,
        m.thickness,
    )
    # mass is a Python float; com is wp.vec3; I is wp.mat33 (or numpy)
    com_t = tuple(float(c) for c in com)  # length-3 tuple
    if hasattr(I, "numpy_value"):
        I_arr = np.array(I.numpy_value, dtype=np.float64).reshape(3, 3)
    else:
        # wp.mat33 supports float access via index
        I_arr = np.array([[float(I[r, c]) for c in range(3)] for r in range(3)], dtype=np.float64)
    return float(mass), com_t, I_arr.tolist(), float(V)


def _hash_output(mass: float, com: tuple[float, float, float], I: list[list[float]], V: float) -> str:
    return hash_bytes([mass, list(com), I, V])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--reruns", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup calls discarded")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of meshes (debug only)")
    args = parser.parse_args()

    v2_mode = None
    if args.variant == "v2":
        v2_mode = _enable_v2_determinism()

    # Defer warp/newton imports until after determinism flag is set.
    import warp as wp  # noqa: WPS433

    print(f"Warp version: {wp.__version__}", file=sys.stderr)
    print(f"Variant: {args.variant}", file=sys.stderr)

    meshes = load_meshes()
    if args.limit > 0:
        meshes = meshes[: args.limit]
    print(f"Loaded {len(meshes)} mesh records", file=sys.stderr)

    # Per-mesh result rows.
    rows: list[dict] = []

    # ----- Phase 0: warmup pass — JIT, CUDA context, kernel cache ------------
    print("Warmup pass…", file=sys.stderr)
    for m in meshes:
        for _ in range(args.warmup):
            _call_compute(m)
    wp.synchronize_device()
    gc.collect()

    # Capture per-variant baseline AFTER warmup so context + JIT cost are excluded.
    gpu_mem_baseline = gpu_mem_used_mib()
    gpu_mem_peak = gpu_mem_baseline

    # ----- Phase 1: timed reruns --------------------------------------------
    started_at = _dt.datetime.now(_dt.UTC).isoformat()
    t0 = time.perf_counter()
    for i, m in enumerate(meshes):
        outputs: list[dict] = []
        times_ms: list[float] = []
        # Per-mesh memory delta — we measure the GPU memory just before and just
        # after each call so that variants which allocate per-call buffers
        # (V1's per-triangle device arrays) show up.
        per_mesh_peak_delta = 0.0
        for _rerun in range(args.reruns):
            mem_before_call = gpu_mem_used_mib()
            wp.synchronize_device()
            start = time.perf_counter()
            mass, com, I, V = _call_compute(m)
            wp.synchronize_device()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)
            outputs.append(
                {
                    "mass": mass,
                    "com": list(com),
                    "I": I,
                    "volume": V,
                    "hash": _hash_output(mass, com, I, V),
                }
            )
            mem_after_call = gpu_mem_used_mib()
            call_delta = mem_after_call - mem_before_call
            if call_delta > per_mesh_peak_delta:
                per_mesh_peak_delta = call_delta
            if mem_after_call > gpu_mem_peak:
                gpu_mem_peak = mem_after_call

        rows.append(
            {
                "robot": m.robot,
                "shape_index": m.shape_index,
                "num_tris": m.num_tris,
                "is_solid": m.is_solid,
                "density": m.density,
                "thickness": m.thickness,
                "ms": times_ms,
                "outputs": outputs,
                "mem_peak_delta_mib": per_mesh_peak_delta,
            }
        )

        if (i + 1) % 25 == 0:
            print(f"  [{args.variant}] {i + 1}/{len(meshes)} meshes done", file=sys.stderr)
        # Encourage Warp's memory pool to release between large meshes.
        if i % 32 == 31:
            gc.collect()

    finished_at = _dt.datetime.now(_dt.UTC).isoformat()
    elapsed_total = time.perf_counter() - t0

    payload = {
        "variant": args.variant,
        "v2_mode": v2_mode,
        "warp_version": wp.__version__,
        "warp_deterministic": str(getattr(wp.config, "deterministic", "not_guaranteed")),
        "warp_path": wp.__file__,
        "reruns": args.reruns,
        "warmup": args.warmup,
        "num_meshes": len(meshes),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_total_s": elapsed_total,
        "gpu_mem_baseline_mib": gpu_mem_baseline,
        "gpu_mem_peak_mib": gpu_mem_peak,
        "gpu_mem_delta_mib": gpu_mem_peak - gpu_mem_baseline,
        "rows": rows,
        "env": env_snapshot(),
    }
    out = write_results(args.variant, payload)
    print(f"Wrote {out} ({len(rows)} rows, peak GPU mem delta: {gpu_mem_peak - gpu_mem_baseline:.1f} MiB)")


if __name__ == "__main__":
    main()
