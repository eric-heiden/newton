# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a visual-first tiled sensing benchmark and export review artifacts."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

_BRICK_HALF_EXTENTS_M = np.array([0.016, 0.008, 0.0048], dtype=np.float32)
_BRICK_GAP_M = 0.0008
_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=_repo_root(),
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def default_tiled_sensing_results_dir() -> Path:
    """Return the default directory used for tiled sensing benchmark artifacts."""
    configured = os.environ.get("NEWTON_TILED_SENSING_RESULTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (_repo_root() / "benchmarks" / "results" / "tiled_sensing").resolve()


def default_tiled_sensing_runs_dir() -> Path:
    """Return the default directory used for tiled sensing benchmark run payloads."""
    return default_tiled_sensing_results_dir() / "runs"


def default_tiled_sensing_preview_dir() -> Path:
    """Return the default directory used for tiled sensing preview arrays."""
    return default_tiled_sensing_results_dir() / "previews"


def default_tiled_sensing_index_path() -> Path:
    """Return the aggregated tiled sensing benchmark index path."""
    return default_tiled_sensing_results_dir() / "index.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    world_count: int = 8
    steps: int = 6
    warmup_steps: int = 1
    resolution_width: int = 64
    resolution_height: int = 64
    tile_width: int = 8
    tile_height: int = 8
    camera_fov_rad: float = float(np.deg2rad(42.0))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Newton's visual-first tiled sensing benchmark and write structured artifacts."
    )
    parser.add_argument("--results-dir", type=Path, default=default_tiled_sensing_results_dir())
    parser.add_argument("--world-count", type=int, default=BenchmarkConfig.world_count)
    parser.add_argument("--steps", type=int, default=BenchmarkConfig.steps)
    parser.add_argument("--warmup-steps", type=int, default=BenchmarkConfig.warmup_steps)
    parser.add_argument("--width", type=int, default=BenchmarkConfig.resolution_width)
    parser.add_argument("--height", type=int, default=BenchmarkConfig.resolution_height)
    parser.add_argument("--tile-width", type=int, default=BenchmarkConfig.tile_width)
    parser.add_argument("--tile-height", type=int, default=BenchmarkConfig.tile_height)
    parser.add_argument("--device", type=str, default=None, help="Warp device alias, for example cpu or cuda:0.")
    return parser


def _validate_config(config: BenchmarkConfig) -> None:
    if config.world_count <= 0:
        raise ValueError("world_count must be positive")
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.resolution_width <= 0 or config.resolution_height <= 0:
        raise ValueError("resolution must be positive")
    if config.tile_width <= 0 or config.tile_height <= 0:
        raise ValueError("tile dimensions must be positive")


def _build_brick_scene(world_count: int) -> tuple[newton.Model, newton.State, list[int], list[int], list[float]]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    moving_brick_ids: list[int] = []
    stack_sizes: list[int] = []
    stack_top_heights: list[float] = []

    for world_id in range(world_count):
        builder.begin_world()
        stack_bricks = 2 + (world_id % 3)
        stack_sizes.append(stack_bricks)

        for brick_index in range(stack_bricks):
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        float(_BRICK_HALF_EXTENTS_M[2] + brick_index * (_BRICK_HALF_EXTENTS_M[2] * 2.0 + _BRICK_GAP_M)),
                    ),
                    wp.quat_identity(),
                ),
                label=f"stack_brick_{world_id}_{brick_index}",
            )
            builder.add_shape_box(
                body,
                hx=float(_BRICK_HALF_EXTENTS_M[0]),
                hy=float(_BRICK_HALF_EXTENTS_M[1]),
                hz=float(_BRICK_HALF_EXTENTS_M[2]),
            )

        stack_top_height = float(stack_bricks * (_BRICK_HALF_EXTENTS_M[2] * 2.0 + _BRICK_GAP_M))
        stack_top_heights.append(stack_top_height)
        moving_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(-0.08, 0.0, stack_top_height + 0.045),
                wp.quat_identity(),
            ),
            label=f"moving_brick_{world_id}",
        )
        moving_brick_ids.append(moving_body)
        builder.add_shape_box(
            moving_body,
            hx=float(_BRICK_HALF_EXTENTS_M[0]),
            hy=float(_BRICK_HALF_EXTENTS_M[1]),
            hz=float(_BRICK_HALF_EXTENTS_M[2]),
        )
        builder.end_world()

    builder.add_ground_plane()
    model = builder.finalize()
    state = model.state()
    return model, state, moving_brick_ids, stack_sizes, stack_top_heights


def _camera_transform() -> wp.transformf:
    return wp.transformf(
        wp.vec3f(0.18, -0.18, 0.12),
        wp.quat_rpy(float(np.deg2rad(65.0)), 0.0, float(np.deg2rad(45.0))),
    )


def _moving_brick_position(step: int, steps: int, world_id: int, stack_top_height: float) -> np.ndarray:
    progress = 0.0 if steps <= 1 else float(step) / float(steps - 1)
    lateral_bias = (world_id % 2) * 0.004 - 0.002
    x = -0.08 + progress * 0.08
    y = lateral_bias
    z = stack_top_height + 0.035 - progress * 0.014
    return np.array([x, y, z], dtype=np.float32)


def _update_state_for_step(
    state: newton.State,
    moving_brick_ids: list[int],
    stack_top_heights: list[float],
    step: int,
    steps: int,
) -> list[list[float]]:
    body_q = state.body_q.numpy()
    poses: list[list[float]] = []
    for world_id, body_id in enumerate(moving_brick_ids):
        position = _moving_brick_position(step, steps, world_id, stack_top_heights[world_id])
        body_q[body_id, :3] = position
        body_q[body_id, 3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        poses.append([*position.astype(float).tolist(), 0.0, 0.0, 0.0, 1.0])
    state.body_q.assign(body_q)
    return poses


def _make_runtime_packet() -> dict[str, Any]:
    return {
        "alignment": "internal_sensor_runtime_boundary",
        "input_fields": [
            "world_id",
            "body_q",
            "camera_transform",
            "camera_rays",
        ],
        "derived_observables": [
            "moving_brick_pose",
            "stack_top_height_m",
            "depth_hit_pixel_count",
            "color_nonzero_pixel_count",
            "mean_depth_m",
        ],
        "output_fields": [
            "color_image",
            "depth_image",
        ],
    }


def run_tiled_sensing_benchmark(
    config: BenchmarkConfig, *, device: str | None = None
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run the tiled sensing benchmark and return the structured payload plus previews."""
    _validate_config(config)
    wp.init()
    resolved_device = device or wp.get_preferred_device().alias

    with wp.ScopedDevice(resolved_device):
        model, state, moving_brick_ids, stack_sizes, stack_top_heights = _build_brick_scene(config.world_count)
        sensor = SensorTiledCamera(model=model)
        sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        sensor.render_config.tile_width = config.tile_width
        sensor.render_config.tile_height = config.tile_height
        sensor.render_config.enable_backface_culling = True
        sensor.utils.create_default_light(enable_shadows=False)
        sensor.utils.assign_random_colors_per_shape()
        sensor.utils.assign_checkerboard_material_to_all_shapes()

        camera_rays = sensor.utils.compute_pinhole_camera_rays(
            config.resolution_width,
            config.resolution_height,
            config.camera_fov_rad,
        )
        camera_transforms = wp.array(
            [[_camera_transform() for _ in range(config.world_count)]],
            dtype=wp.transformf,
            device=model.device,
        )
        color_image = sensor.utils.create_color_image_output(
            config.resolution_width,
            config.resolution_height,
            camera_count=1,
        )
        depth_image = sensor.utils.create_depth_image_output(
            config.resolution_width,
            config.resolution_height,
            camera_count=1,
        )

        for warmup_step in range(config.warmup_steps):
            _update_state_for_step(
                state,
                moving_brick_ids,
                stack_top_heights,
                min(warmup_step, config.steps - 1),
                config.steps,
            )
            sensor.update(
                state,
                camera_transforms,
                camera_rays,
                color_image=color_image,
                depth_image=depth_image,
                refit_bvh=True,
            )

        render_times_ms: list[float] = []
        observability_packets: list[dict[str, Any]] = []
        final_color = np.zeros(
            (config.world_count, 1, config.resolution_height, config.resolution_width), dtype=np.uint32
        )
        final_depth = np.zeros(
            (config.world_count, 1, config.resolution_height, config.resolution_width), dtype=np.float32
        )

        for step in range(config.steps):
            moving_poses = _update_state_for_step(state, moving_brick_ids, stack_top_heights, step, config.steps)
            start = time.perf_counter()
            sensor.update(
                state,
                camera_transforms,
                camera_rays,
                color_image=color_image,
                depth_image=depth_image,
                refit_bvh=True,
            )
            render_times_ms.append((time.perf_counter() - start) * 1000.0)

            color_np = color_image.numpy()
            depth_np = depth_image.numpy()
            final_color = color_np
            final_depth = depth_np

            for world_id in range(config.world_count):
                world_depth = depth_np[world_id, 0]
                world_color = color_np[world_id, 0]
                hit_mask = world_depth > 0.0
                hit_pixels = int(np.count_nonzero(hit_mask))
                color_nonzero = int(np.count_nonzero(world_color))
                mean_depth = float(np.mean(world_depth[hit_mask])) if hit_pixels else None
                min_depth = float(np.min(world_depth[hit_mask])) if hit_pixels else None
                max_depth = float(np.max(world_depth[hit_mask])) if hit_pixels else None
                observability_packets.append(
                    {
                        "world_id": world_id,
                        "step": step,
                        "time_s": round(step / 60.0, 6),
                        "camera_transform": list(
                            np.asarray(camera_transforms.numpy()[0, world_id], dtype=np.float32).astype(float)
                        ),
                        "moving_brick_pose": moving_poses[world_id],
                        "stack_brick_count": stack_sizes[world_id],
                        "stack_top_height_m": round(stack_top_heights[world_id], 6),
                        "depth_hit_pixel_count": hit_pixels,
                        "color_nonzero_pixel_count": color_nonzero,
                        "mean_depth_m": mean_depth,
                        "min_depth_m": min_depth,
                        "max_depth_m": max_depth,
                    }
                )

    pixel_count = config.world_count * config.resolution_width * config.resolution_height
    mean_render_time_ms = float(statistics.fmean(render_times_ms))
    total_render_time_ms = float(sum(render_times_ms))
    steps_per_second = float(1000.0 / mean_render_time_ms) if mean_render_time_ms > 0.0 else 0.0
    pixels_per_second = float(pixel_count * steps_per_second)

    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": _utc_now(),
        "benchmark": {
            "name": "visual_first_tiled_sensing",
            "schema_version": _SCHEMA_VERSION,
            "git_commit": _git_commit(),
            "device": resolved_device,
            "host": socket.gethostname(),
            "warp_version": getattr(wp, "__version__", None),
            "steps": config.steps,
            "warmup_steps": config.warmup_steps,
            "resolution": [config.resolution_width, config.resolution_height],
            "render_order": "tiled",
            "tile_shape": [config.tile_width, config.tile_height],
        },
        "scenario": {
            "name": "multi_world_brick_stacking",
            "world_count": config.world_count,
            "camera_count": 1,
            "stack_brick_counts": stack_sizes,
            "camera_fov_rad": config.camera_fov_rad,
            "brick_half_extents_m": _BRICK_HALF_EXTENTS_M.astype(float).tolist(),
        },
        "runtime": _make_runtime_packet(),
        "summary": {
            "total_render_time_ms": total_render_time_ms,
            "mean_render_time_ms": mean_render_time_ms,
            "min_render_time_ms": float(min(render_times_ms)),
            "max_render_time_ms": float(max(render_times_ms)),
            "steps_per_second": steps_per_second,
            "pixels_per_second": pixels_per_second,
            "preview_step": config.steps - 1,
        },
        "observability": {
            "world_packets": observability_packets,
        },
        "artifacts": {},
        "notes": [
            "This first pass keeps the sensing boundary internal and uses SensorTiledCamera directly.",
            "Preview arrays capture the final tiled render for review and downstream dashboard ingestion.",
        ],
    }
    return payload, final_color, final_depth


def build_tiled_sensing_index(run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an aggregated index for tiled sensing benchmark runs."""
    sorted_runs = sorted(
        run_payloads,
        key=lambda item: (item.get("generated_at") or "", item.get("run_id") or ""),
        reverse=True,
    )
    latest = sorted_runs[0] if sorted_runs else None
    summaries = [
        {
            "run_id": run.get("run_id"),
            "generated_at": run.get("generated_at"),
            "device": run.get("benchmark", {}).get("device"),
            "world_count": run.get("scenario", {}).get("world_count"),
            "mean_render_time_ms": run.get("summary", {}).get("mean_render_time_ms"),
            "steps_per_second": run.get("summary", {}).get("steps_per_second"),
        }
        for run in sorted_runs
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_count": len(sorted_runs),
        "latest_run": summaries[0] if summaries else None,
        "runs": summaries,
        "latest_runtime": latest.get("runtime") if latest else None,
    }


def write_tiled_sensing_artifacts(
    results_dir: Path,
    run_payload: dict[str, Any],
    color_preview: np.ndarray,
    depth_preview: np.ndarray,
) -> tuple[Path, Path]:
    """Write a tiled sensing run payload, previews, and refreshed index."""
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = results_dir / "runs"
    preview_dir = results_dir / "previews"
    runs_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(run_payload["run_id"])
    color_preview_path = preview_dir / f"{run_id}_color.npy"
    depth_preview_path = preview_dir / f"{run_id}_depth.npy"
    np.save(color_preview_path, color_preview)
    np.save(depth_preview_path, depth_preview)

    run_payload = json.loads(json.dumps(run_payload))
    run_payload["artifacts"] = {
        "color_preview": str(color_preview_path.relative_to(results_dir)),
        "depth_preview": str(depth_preview_path.relative_to(results_dir)),
    }

    run_path = runs_dir / f"{run_id}.json"
    run_path.write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")

    run_payloads = []
    for candidate in runs_dir.glob("*.json"):
        run_payloads.append(json.loads(candidate.read_text(encoding="utf-8")))
    index_payload = build_tiled_sensing_index(run_payloads)
    index_path = results_dir / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    return run_path, index_path


def main(argv: list[str] | None = None) -> int:
    """Run the tiled sensing benchmark CLI."""
    args = _make_parser().parse_args(argv)
    config = BenchmarkConfig(
        world_count=args.world_count,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        resolution_width=args.width,
        resolution_height=args.height,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
    )
    run_payload, color_preview, depth_preview = run_tiled_sensing_benchmark(config, device=args.device)
    run_path, index_path = write_tiled_sensing_artifacts(
        args.results_dir.resolve(), run_payload, color_preview, depth_preview
    )
    print(f"Wrote tiled sensing benchmark run to {run_path}")
    print(f"Updated tiled sensing benchmark index at {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
