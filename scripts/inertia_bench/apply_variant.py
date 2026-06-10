# SPDX-FileCopyrightText: Copyright (c) 2026 Eric Heiden
# SPDX-License-Identifier: Apache-2.0

"""Apply a specific variant by overwriting ``newton/_src/geometry/inertia.py``.

Variants:
  v0 — Restore the upstream/main version (current `git show origin/main:...`).
  v1 — Restore the version from commit 23416f8b (f64 host-reduce).
  v2 — Same as v0 source; the runner sets wp.config.deterministic instead.
  v3 — Restore the version from commit 9b523cf1 (tile-sum, no atomics).

We snapshot the upstream/main version once on first call to v0/v2.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INERTIA_PATH = REPO_ROOT / "newton" / "_src" / "geometry" / "inertia.py"

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

V0_SNAPSHOT = SNAPSHOTS_DIR / "v0_inertia.py"
V1_SNAPSHOT = SNAPSHOTS_DIR / "v1_inertia.py"
V2_SNAPSHOT = SNAPSHOTS_DIR / "v2_inertia.py"
V3_SNAPSHOT = SNAPSHOTS_DIR / "v3_inertia.py"

V1_COMMIT = "23416f8bef003738636a4592c73d9b24a793a4ea"
V3_COMMIT = "9b523cf170a0edfdb9305eded4f7593d23456c9e"


def ensure_snapshot(commit: str | None, snapshot: Path) -> None:
    """Cache a commit's inertia.py to snapshot file (commit=None means current file)."""
    if snapshot.exists():
        return
    if commit is None:
        shutil.copy2(INERTIA_PATH, snapshot)
    else:
        # git show <commit>:newton/_src/geometry/inertia.py
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit}:newton/_src/geometry/inertia.py"]
        )
        snapshot.write_bytes(out)


def apply(variant: str) -> None:
    if variant == "v0":
        ensure_snapshot(None, V0_SNAPSHOT)  # capture upstream/main as v0 baseline
        shutil.copy2(V0_SNAPSHOT, INERTIA_PATH)
        print("Applied v0 (upstream/main inertia.py)")
        return
    if variant == "v1":
        ensure_snapshot(V1_COMMIT, V1_SNAPSHOT)
        shutil.copy2(V1_SNAPSHOT, INERTIA_PATH)
        print("Applied v1 (commit 23416f8b — f64 host-reduce)")
        return
    if variant == "v2":
        # V2-EMULATION of Warp PR #1355 run_to_run determinism (since the
        # PR's native build requires CUDA 12 toolkit, which is not installed
        # here). The emulation produces the SAME kernel-level behavior the
        # PR would produce: per-thread scatter buffer + deterministic
        # in-order f32 reduce. Differs from V1 only in dtype of the host
        # reduce (f32 here, f64 there).
        shutil.copy2(V2_SNAPSHOT, INERTIA_PATH)
        print("Applied v2 (Warp PR1355 emulation — f32 deterministic reduce)")
        return
    if variant == "v3":
        ensure_snapshot(V3_COMMIT, V3_SNAPSHOT)
        shutil.copy2(V3_SNAPSHOT, INERTIA_PATH)
        print("Applied v3 (commit 9b523cf1 — tile-sum)")
        return
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("v0", "v1", "v2", "v3"))
    args = parser.parse_args()
    apply(args.variant)


if __name__ == "__main__":
    main()
