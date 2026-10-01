# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a Blender look for the ABC station twin against held-out recorded frames.

Only the submitted ``look.py`` is used: the verifier builds the fixed twin from its
own copies of the task files, runs the look in a fresh Blender worker (then locks
post-processing), renders 5 held-out frames through the real top camera with the
robot posed from the log, and scores per-region colors and whole-image color PSNR.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private"))
# Held-out limits: mean region color distance [0-255] at most, color PSNR [dB] at least.
THRESHOLDS = {"region_color_error": 33.0, "color_psnr_db": 18.5}


def verify(script: Path) -> dict:
    started = time.perf_counter()
    look_code = (script.parent / "look.py").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy2(HERE / "look_common.py", work / "look_common.py")
        shutil.copy2(HERE.parent / "abc_twin" / "twin_render.py", work / "twin_render.py")
        for name in ("scene.json", "camera.json", "joint_log.npz"):
            shutil.copy2(PRIVATE / "abc_look" / name, work / name)
        shutil.copytree(PRIVATE / "abc_twin" / "station", work / "station")
        sys.path.insert(0, str(work))
        import look_common  # noqa: PLC0415
        import twin_render  # noqa: PLC0415

        heldout = np.load(PRIVATE / "abc_twin" / "heldout.npz")
        model = look_common.build_model(work)
        renderer = look_common.LookRenderer(model, look_code, work, workdir=work / "blender")
        try:
            checks = look_common.look_integrity(renderer)
            renders = [renderer.render(q) for q in heldout["q"]]
        finally:
            renderer.close()
        masks = [look_common.shape_masks(model, q, work) for q in heldout["q"]]
        recordings = list(heldout["image"])
        region, per_region = look_common.region_color_error(masks, renders, recordings)
        scores = [twin_render.score(r, i) for r, i in zip(renders, recordings, strict=True)]
    metrics = {
        "region_color_error": region,
        "color_psnr_db": float(np.mean([s["color_psnr_db"] for s in scores])),
        "ssim": float(np.mean([s["ssim"] for s in scores])),
        "edge_ncc": float(np.mean([s["edge_ncc"] for s in scores])),
    }
    failed = [name for name, ok in checks.items() if not ok]
    failed += [k for k in ("region_color_error",) if not metrics[k] <= THRESHOLDS[k]]
    failed += [k for k in ("color_psnr_db",) if not metrics[k] >= THRESHOLDS[k]]
    return {
        "task": "abc_look",
        "success": not failed,
        "integrity": checks,
        "failed_checks": failed,
        "metrics": metrics,
        "normalized_worst": {
            "region_color_error": metrics["region_color_error"] / THRESHOLDS["region_color_error"],
            "color_psnr_db": THRESHOLDS["color_psnr_db"] / max(metrics["color_psnr_db"], 1e-6),
        },
        "details": {
            "per_region": {str(k): round(v, 2) for k, v in per_region.items()},
            "seconds": time.perf_counter() - started,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.script.resolve())
    text = json.dumps(result, indent=2, default=float)
    if args.output:
        args.output.write_text(text + "\n")
    print(
        json.dumps(
            {"success": result["success"], "metrics": result["metrics"], "failed_checks": result["failed_checks"]}
        )
    )


if __name__ == "__main__":
    main()
