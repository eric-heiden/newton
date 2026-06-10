# Deterministic Mesh Inertia Investigation

**Date:** 2026-06-10
**Author:** Eric Heiden (with Claude)
**Branch:** `eric-heiden/deterministic-mesh-inertia-investigation`
**Output:** `academic-website-reports/deterministic-mesh-inertia/` (titled "Deterministic mesh inertia")

## 1. Goal

Compare four implementations of Newton's `compute_inertia_mesh` for **determinism**, **accuracy**, **runtime**, and **memory footprint** on the real mesh assets Newton consumes when loading USD-based humanoid and menagerie robots.

The four variants are:

| Variant | Source | Strategy |
|--------|--------|----------|
| **V0 baseline** | newton-physics/newton@main (post-839284af) | GPU `wp.atomic_add` reduction into f32 scalars (current shipping code). |
| **V1 f64 host-reduce** | newton-physics/newton commit `23416f8bef00…` | Per-triangle f32 outputs to device buffer; host (numpy) reduces in f64. |
| **V2 Warp auto-determinism** | V0 kernel + Warp PR #1355 (`wp.config.deterministic = True`) | Same source as V0, but Warp lowers the atomics to scatter-sort-reduce. |
| **V3 tiled in-kernel reduce** | newton-physics/newton commit `9b523cf170…` | Single tile of 256 threads, in-kernel `wp.tile_sum`, no atomics. |

## 2. Specific questions to answer

1. **Does f64 host-side accumulation matter?** On the practical robot meshes (G1, H1, ~5 menagerie), does V1's f64 reduction produce *measurably different* `body_inertia` values from a deterministic f32 reduction (V3), or from the average of V0? Order-of-magnitude in ULPs.
2. **Does V2 (Warp auto-lowering) reproduce V1 / V3 determinism without code changes?** Verify bit-identity across reruns when `wp.config.deterministic = True` on V0's source.
3. **What is the performance cost of each variant?** Median + p90 runtime per mesh, peak GPU memory delta, end-to-end USD import time delta.
4. **Is V0 actually non-deterministic on these meshes?** Quantify the run-to-run drift in `body_com`, `body_inertia`, `body_inv_inertia` (max abs / max rel diff) for each robot.

## 3. Scope

**Assets** (decided with user):

- `unitree_h1` (humanoid, free joint, 25 bodies)
- `unitree_g1` 29-DOF with hands (humanoid)
- 5 menagerie samples chosen for size + collision-mesh diversity:
  - `franka_fr3` (small precise arm)
  - `shadow_hand` (many small meshes)
  - `apptronik_apollo` (large humanoid)
  - `booster_t1` (biped)
  - `google_robot` (mobile manipulator)

USD assets come from `newton.utils.download_asset()` (newton-assets repo) — same path as `test_menagerie_usd_mujoco.py`.

**Measurement:** inertia outputs only (decided with user) — for each mesh-shape body we capture `body_com`, `body_inertia`, `body_inv_inertia` and the intermediate `(volume, first_moment, second_moment)` exposed by `compute_inertia_mesh`. We do **not** run an integrator or FK; this isolates the kernel under test from solver-level non-determinism.

**Reruns:** 5 per (variant × asset) for both determinism and perf (decided with user).

## 4. Architecture

```
newton-mesh-inertia-investigation/        # worktree of newton fork (base: upstream/main)
  ├─ scripts/inertia_bench/
  │   ├─ run_variant.py          # invoked per variant; outputs results.json
  │   ├─ collect_meshes.py       # downloads + extracts per-body meshes once, caches
  │   ├─ apply_variant.py        # patches inertia.py for V0/V1/V3, sets env for V2
  │   ├─ report.py               # aggregates 4 results.json → HTML + plots
  │   └─ shared.py               # asset list, hash helpers, perf timer
  └─ docs/superpowers/specs/2026-06-10-deterministic-mesh-inertia-design.md  (this file)

warp-determinism-pr1355/                  # separate Warp clone for V2
  └─ (built once, used only for V2 run)
```

**Per-variant flow** (one Python process per variant, fresh Warp init):
1. Apply variant (checkout commit or patch `inertia.py` accordingly; for V2 set `wp.config.deterministic = True` before `wp.init`).
2. Load each cached mesh, call `compute_inertia_mesh(...)` 5 times.
3. Capture: hash of each result, max abs/rel diff across reruns, wall-clock per call (CUDA-synchronized), GPU memory peak via `wp.ScopedTimer` + `torch.cuda.max_memory_allocated` analog (we'll use `nvidia-smi` snapshots).
4. Write `results_{variant}.json`.

**Aggregation:** `report.py` loads all four JSONs, computes cross-variant tables (numerical accuracy: max ULP diff vs V1-f64-reference), and emits the HTML report with bar plots (Plotly, embedded JS) and color-coded tables.

## 5. Test harness — key design choices

- **Mesh extraction is shared.** `collect_meshes.py` runs once (under V0 environment) using `newton.ModelBuilder().add_usd(...)` and snapshots `(vertices, indices, is_solid, density)` per shape to `meshes.npz`. This means all four variants compute on bit-identical inputs.
- **Per-mesh hashing.** A variant is "deterministic" iff `hash(body_inertia_run_i) == hash(body_inertia_run_0)` for all i and all meshes. We hash the f64 bit pattern.
- **Reference.** V1 (f64 host-reduce) is the ground-truth reference for accuracy comparisons. Max ULP / max rel error vs V1 is reported for V0, V2, V3.
- **Warp version pinning.** V2 requires Warp PR #1355's branch. We clone Warp separately, build wheel, install into V2's venv. V0/V1/V3 use the shipped Warp 1.14.0 in the existing public-main-bench venv (verified working).
- **Perf isolation.** Each variant runs in its own Python process. `wp.synchronize_device()` before and after each call. Warm-up call discarded.

## 6. Report (HTML)

**Path:** `academic-website-reports/deterministic-mesh-inertia/index.html`
**Title card:** "Deterministic mesh inertia" — registered in root `index.html`.

**Sections:**
1. **Summary** — 2-paragraph TL;DR with a verdict per variant (deterministic? faster? more accurate?).
2. **Determinism table** — color-coded: green if all 5 reruns bit-identical, yellow if reruns differ by <1 ULP, red otherwise. Rows = robots × variants.
3. **Accuracy table** — max abs / rel diff of (V0, V2, V3) `body_inertia` against V1 reference, per robot.
4. **Runtime bar plot** — grouped bars: x = robot, y = ms median, group = variant. Error bars = p10/p90.
5. **GPU memory bar plot** — same layout, y = MiB peak delta.
6. **End-to-end USD import table** — wall-clock for the full `add_usd()` call per variant, per robot.
7. **Per-mesh deep dive** — for the largest mesh in each robot: triangle count, ULP drift, kernel-only ms.
8. **Reproducibility** — exact commits, Warp version/build, GPU model, environment.

**Implementation:** Plotly inline JS via CDN; minimalist CSS matching the existing `academic-website-reports/index.html` palette (`--accent: #0f766e`, `--paper: #f7f5f0`, `--panel: #fffdf8`).

## 7. Out of scope (YAGNI)

- Solver-level determinism (V2 only changes the inertia kernel; downstream solvers still atomic-accumulate).
- Multi-GPU / cross-architecture determinism (single L40 only).
- Hollow-mesh inertia variant (`compute_hollow_mesh_inertia`): the four variants all change it identically; covered only if at least one robot uses it.
- Backward-pass / differentiability impact (V3 sets `enable_backward=False`).
- ARM/CPU host-reduction variance.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Warp PR #1355 doesn't build cleanly. | Time-box build to 30 min. If it fails, document the failure mode and skip V2 numerically but keep its slot in the report. |
| Some menagerie USDs don't have mesh-collision bodies (cylinders only). | `collect_meshes.py` skips robots with zero mesh shapes and we report N/A. The 5 chosen robots all have mesh assets per inspection. |
| Tile-size 256 in V3 underutilizes large meshes (100k+ tris). | Already reflected in V3's design (single tile). We measure as-is — it is the change under test. |
| Run-to-run perf noise on shared GPU. | 5 reruns + report p10/median/p90. Reject runs with >2x noise vs median. |

## 9. Deliverables

1. Branch `eric-heiden/deterministic-mesh-inertia-investigation` pushed to `eric-heiden/newton` containing the test harness under `scripts/inertia_bench/`.
2. Separate Warp clone at `~/repos/warp-determinism-pr1355/` (not pushed anywhere).
3. Report at `~/repos/academic-website-reports/deterministic-mesh-inertia/` with the HTML, a JSON dump of raw results, and a thumbnail.
4. Updated root `index.html` of `academic-website-reports` linking the new card.
5. Commits and pushes to `gh-pages` (or the configured publishing branch) for `reports.eric-heiden.com`.
