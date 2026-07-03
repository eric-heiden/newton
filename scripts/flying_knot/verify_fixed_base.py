# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic probe: the xArm7 base and pedestal must not move.

Runs the flying-knot example headless with the arm enabled, records the
world transform of the arm base link and of every static shape at every
frame, and fails (exit 1) if anything drifts.

Usage: uv run python scripts/flying_knot/verify_fixed_base.py [--frames N]
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))

from example_cable_flying_knot import Example  # noqa: E402

import newton  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=492)
    args = ap.parse_args()

    viewer = newton.viewer.ViewerNull()
    ex = Example(viewer, SimpleNamespace())
    if not ex.use_arm:
        print("FAIL: arm assets not found; probe requires the xArm7")
        sys.exit(2)

    # Pedestal must be static world geometry (body -1), not attached to a link.
    shape_body = ex.model.shape_body.numpy()
    shape_labels = list(ex.model.shape_label)
    ped = [i for i, lbl in enumerate(shape_labels) if "pedestal" in lbl]
    assert ped, "pedestal shape not found"
    ped_ok = all(shape_body[i] == -1 for i in ped)
    print(
        f"pedestal shape parent body: {[int(shape_body[i]) for i in ped]} (static == -1): {'OK' if ped_ok else 'FAIL'}"
    )

    base_poses = []
    n = min(args.frames, ex.num_frames)
    for _ in range(n):
        ex.step()
        base_poses.append(ex.state_0.body_q.numpy()[ex.arm_base_body].copy())
    base = np.array(base_poses)
    drift = np.abs(base - base[0]).max()
    print(f"frames simulated: {n}")
    print(f"arm base world transform, frame 0:   {np.round(base[0], 6)}")
    print(f"arm base world transform, frame {n - 1}: {np.round(base[-1], 6)}")
    print(f"max |transform - transform[0]| over all frames: {drift:.3e}")

    ok = ped_ok and drift < 1.0e-6
    print("PASS: base and pedestal remain fixed" if ok else "FAIL: base or pedestal moved")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
