# Deterministic mesh inertia bench

Benchmark harness comparing four implementations of `compute_inertia_mesh` for
determinism, accuracy, runtime, and memory on real Menagerie USD assets.

See [`docs/superpowers/specs/2026-06-10-deterministic-mesh-inertia-design.md`](../../docs/superpowers/specs/2026-06-10-deterministic-mesh-inertia-design.md)
for the full design.

## Usage

```bash
# 1. Extract per-shape meshes from every USD asset once.
uv run python scripts/inertia_bench/collect_meshes.py

# 2. Run each variant (V0/V1/V3 from this repo, V2 = Warp PR1355 emulation).
for v in v0 v1 v2 v3; do
    uv run python scripts/inertia_bench/apply_variant.py --variant $v
    uv run python scripts/inertia_bench/run_variant.py --variant $v --reruns 5
done

# 3. Restore upstream main inertia.py before committing.
uv run python scripts/inertia_bench/apply_variant.py --variant v0

# 4. Render the HTML report into ../academic-website-reports/.
uv run python scripts/inertia_bench/report.py
```

## Variants

| Variant | Source | Behavior |
|---------|--------|----------|
| V0 | `newton-physics/newton@main` | `wp.atomic_add` into f32 scalar (non-deterministic). |
| V1 | commit `23416f8b` (Andrew Kaufman) | Per-triangle f32 outputs → numpy f64 host reduce. |
| V2 | NVIDIA/warp PR #1355 emulation | Same kernel + host-side f32 deterministic reduce (emulates run-to-run). |
| V3 | commit `9b523cf1` (twidmer) | `wp.tile_sum` in-kernel reduce, no atomics, no host work. |

## Notes

- V2 is an emulation because PR #1355's native CUB integration requires CUDA Toolkit 12.0+,
  which is not available on this benchmark host (only CUDA 11.5 from apt). The emulation matches
  the runtime functional behavior of PR #1355 on f32-typed atomic destinations
  (per-thread scatter + deterministic in-order f32 reduce).
- The harness monkey-patches `newton._src.geometry.inertia.compute_inertia_mesh` during
  `collect_meshes.py` to capture every call that occurs during `add_usd()` for each robot.
  This guarantees byte-identical inputs across variants.
- Results land in `scripts/inertia_bench/results/results_{variant}.json` and the rendered
  report goes to `~/repos/academic-website-reports/deterministic-mesh-inertia/`.
