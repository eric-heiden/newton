# Visual-First Tiled Sensing Benchmark Contract

This note defines the first executable benchmark contract for
[ROB-142](/ROB/issues/ROB-142). The goal is not a full dashboard product yet.
The goal is to make Newton's visual-first tiled sensing path reproducible,
inspectable, and obviously tied to the multi-world tiled-camera runtime.

## Scenario

The first implementation uses a deterministic multi-world brick-stacking scene:

- each world contains a small tower of rigid box bricks
- each world contains one moving top brick that approaches the tower over time
- one tiled camera renders all worlds every step
- the benchmark logs both render timings and simple per-world observable fields

This keeps the workload visibly tied to the tiled-camera path without requiring
robot-policy or task-planning infrastructure first.

## Required Artifact Sections

Every run should emit one JSON payload with these top-level sections:

1. `schema_version`
2. `run_id`
3. `generated_at`
4. `benchmark`
5. `scenario`
6. `runtime`
7. `summary`
8. `observability`
9. `artifacts`

## Runtime Alignment

The `runtime` section should reflect the internal solver-state to sensor-runtime
boundary rather than a public API:

- input fields:
  - `world_id`
  - `body_q`
  - `camera_transform`
  - `camera_rays`
- derived observables:
  - `moving_brick_pose`
  - `stack_top_height_m`
  - `depth_hit_pixel_count`
  - `color_nonzero_pixel_count`
  - `mean_depth_m`
- output fields:
  - `color_image`
  - `depth_image`

## Reviewability Requirements

Each run should also save preview arrays for the final step:

- `color_preview`
- `depth_preview`

These previews should be referenced from the JSON artifact through relative
paths so a dashboard or review tool can load them without guessing filenames.

## Minimal Summary Metrics

The `summary` section should include:

- `total_render_time_ms`
- `mean_render_time_ms`
- `min_render_time_ms`
- `max_render_time_ms`
- `steps_per_second`
- `pixels_per_second`
- `preview_step`

## Minimal Observability Packets

The `observability.world_packets` list should contain one packet per
`(world_id, step)` with:

- `world_id`
- `step`
- `time_s`
- `camera_transform`
- `moving_brick_pose`
- `stack_brick_count`
- `stack_top_height_m`
- `depth_hit_pixel_count`
- `color_nonzero_pixel_count`
- `mean_depth_m`
- `min_depth_m`
- `max_depth_m`

This is the minimum evidence needed to answer "did the tiled camera render the
multi-world brick-stacking scene, and what did it cost?" before broader
dashboard work.
