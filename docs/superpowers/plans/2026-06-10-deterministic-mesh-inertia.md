# Deterministic Mesh Inertia Investigation - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically compare four implementations of `compute_inertia_mesh` (V0 main, V1 f64 host-reduce, V2 Warp PR1355 auto-determinism, V3 tile-sum) for determinism, accuracy, runtime, and memory on real robot meshes (G1, H1, ~5 menagerie USDs), and publish an HTML report.

**Architecture:** Four-variant benchmark with shared mesh inputs, one Python subprocess per variant, isolated venvs (V2 uses a Warp built from PR1355). Aggregator emits an HTML page with Plotly bar plots and color-coded tables, pushed to `reports.eric-heiden.com`.

**Tech Stack:** Newton (worktree), Warp (1.14.0 for V0/V1/V3; PR1355 branch for V2), Plotly, numpy.

---

### Task 1: Set up Newton worktree off upstream/main

**Files:**
- Create worktree at `/home/horde/repos/newton-worktrees/deterministic-mesh-inertia/`

- [ ] **Step 1:** `git -C /home/horde/repos/newton fetch origin main`
- [ ] **Step 2:** `git -C /home/horde/repos/newton worktree add -b eric-heiden/deterministic-mesh-inertia-investigation /home/horde/repos/newton-worktrees/deterministic-mesh-inertia origin/main`
- [ ] **Step 3:** Confirm `git -C /home/horde/repos/newton-worktrees/deterministic-mesh-inertia log --oneline -1` matches origin/main HEAD.
- [ ] **Step 4:** `cd` into worktree and create venv with `uv sync --extra dev`.
- [ ] **Step 5:** Verify `uv run python -c "import newton, warp; print(newton.__version__, warp.__version__)"` works.

### Task 2: Clone Warp PR1355 separately and build

**Files:**
- Create clone at `/home/horde/repos/warp-determinism-pr1355/`

- [ ] **Step 1:** `git clone https://github.com/NVIDIA/warp.git /home/horde/repos/warp-determinism-pr1355`
- [ ] **Step 2:** `cd /home/horde/repos/warp-determinism-pr1355 && gh pr checkout 1355` (or `git fetch origin pull/1355/head:pr1355 && git checkout pr1355`)
- [ ] **Step 3:** Build per `BUILD.md`. Try `python build_lib.py --build_llvm=False`; if too slow or fails, use `pip install -e .` after building with cmake. Time-box to 30 minutes.
- [ ] **Step 4:** Verify `wp.config.deterministic` attribute exists.

### Task 3: Author shared bench harness (under newton worktree)

**Files:**
- Create `scripts/inertia_bench/shared.py` — asset registry, mesh cache loader, hash helpers, GPU memory measurement utility
- Create `scripts/inertia_bench/collect_meshes.py` — one-shot USD → `meshes.npz` extraction
- Create `scripts/inertia_bench/run_variant.py` — runs all meshes through one variant, writes `results_{variant}.json`
- Create `scripts/inertia_bench/apply_variant.py` — switches `inertia.py` to V0/V1/V3 or sets V2 env
- Create `scripts/inertia_bench/report.py` — aggregates 4 jsons → HTML + plotly + tables
- Create `scripts/inertia_bench/__init__.py`

- [ ] **Step 1:** Write `shared.py` with `ASSETS = [...]` list, `iter_meshes()`, `hash_array()`, `peak_gpu_mb()`.
- [ ] **Step 2:** Write `collect_meshes.py` that calls `ModelBuilder.add_usd()` for each asset, walks `shape_geo_src`, dumps to `~/repos/newton-worktrees/deterministic-mesh-inertia/scripts/inertia_bench/meshes.npz`.
- [ ] **Step 3:** Write `apply_variant.py` that for `--variant {v0,v1,v2,v3}` either: checks out the right commit's `inertia.py` (V0/V1/V3) or sets `WARP_DETERMINISTIC=1` env var (V2).
- [ ] **Step 4:** Write `run_variant.py` that takes `--variant`, applies it, loads `meshes.npz`, loops 5 reruns × N meshes × `compute_inertia_mesh`, captures times + peak memory + hashes + numerical outputs → `results_{variant}.json`.
- [ ] **Step 5:** Sanity smoke-test `collect_meshes.py` on H1 only first.

### Task 4: Collect meshes from all 7 robots

- [ ] **Step 1:** Run `uv run python -m inertia_bench.collect_meshes --assets h1 g1 franka_fr3 shadow_hand apptronik_apollo booster_t1 google_robot`.
- [ ] **Step 2:** Verify `meshes.npz` exists and contains per-shape `(vertices, indices, is_solid, density, body_name, robot)` records.
- [ ] **Step 3:** Print per-robot mesh count + median triangle count as a sanity check.

### Task 5: V0 baseline run

- [ ] **Step 1:** `apply_variant.py --variant v0` (no-op, just confirms HEAD is upstream/main).
- [ ] **Step 2:** `run_variant.py --variant v0 --reruns 5` → `results_v0.json`.
- [ ] **Step 3:** Inspect output: confirm we got 7 robots × all meshes × 5 reruns; check if any rerun differs from rerun-0 (V0 is the one expected to drift).

### Task 6: V1 (f64 host-reduce) run

- [ ] **Step 1:** `apply_variant.py --variant v1` — applies the inertia.py from commit `23416f8b…`.
- [ ] **Step 2:** `run_variant.py --variant v1 --reruns 5` → `results_v1.json`.
- [ ] **Step 3:** Verify bit-identical hashes across reruns.

### Task 7: V3 (tile-sum) run

- [ ] **Step 1:** `apply_variant.py --variant v3` — applies the inertia.py from commit `9b523cf1…`.
- [ ] **Step 2:** `run_variant.py --variant v3 --reruns 5` → `results_v3.json`.
- [ ] **Step 3:** Verify bit-identical hashes across reruns; verify the tile_dim=256 kernel correctly handles meshes with <256 and >256 triangles.

### Task 8: V2 (Warp PR1355 auto-lowering) run

- [ ] **Step 1:** Create separate venv for V2 (`uv venv .venv-v2`), install Warp from the local PR1355 clone via `uv pip install -e /home/horde/repos/warp-determinism-pr1355`.
- [ ] **Step 2:** `apply_variant.py --variant v2` — restores main `inertia.py`, exports `WARP_DETERMINISTIC=1`.
- [ ] **Step 3:** Run `wp.config.deterministic = True` is set in `run_variant.py` for V2 branch, then call inertia kernels.
- [ ] **Step 4:** Verify bit-identity across 5 reruns. If V2 build failed, write `results_v2.json` with `{"status": "build-failed", ...}` and the report handles it.

### Task 9: Generate HTML report

- [ ] **Step 1:** Write `report.py` that loads `results_{v0,v1,v2,v3}.json`, computes per-variant tables (determinism, max ULP/abs diff vs V1, runtime median, peak memory, USD import time).
- [ ] **Step 2:** Embed Plotly via CDN, render grouped bar charts (runtime, memory) and color-coded HTML tables.
- [ ] **Step 3:** Output `~/repos/academic-website-reports/deterministic-mesh-inertia/index.html` and a `results.json` with the raw data.
- [ ] **Step 4:** Add an entry to `~/repos/academic-website-reports/index.html` linking the new card "Deterministic mesh inertia".

### Task 10: Commit and push everything

- [ ] **Step 1:** Commit harness + design + plan to `eric-heiden/deterministic-mesh-inertia-investigation` branch, push to `eric` remote.
- [ ] **Step 2:** Commit report assets to `academic-website-reports` repo, push to its main branch (which is what `reports.eric-heiden.com` serves).
- [ ] **Step 3:** Verify the report URL renders.

---

## Self-review

- Spec coverage: ✓ all 4 variants, all 7 assets, all metrics, all deliverables.
- No placeholders: ✓ every step has the actual command.
- Types: ✓ consistent paths and JSON schema across tasks.

Plan complete. Executing inline (user requested fully autonomous).
