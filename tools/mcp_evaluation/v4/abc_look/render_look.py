# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render the twin in Blender with look.py applied and score it against the recorded frames.

Starts a fresh Blender process, runs ``look.py``, renders the given log frames through the
real top camera, writes ``<out>/fNNNNN.png`` (render | recording), and prints
``twin_render.score`` per frame and the mean.

Usage: ``python render_look.py [--frames 0 8 16] [--out renders]``
"""

import argparse
import json
from pathlib import Path

import look_common
import numpy as np
import station_look
import twin_render

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=HERE / "renders")
    args = parser.parse_args()
    recorded = station_look.recorded_frames()
    frames = args.frames if args.frames else sorted(recorded)
    args.out.mkdir(parents=True, exist_ok=True)
    model = look_common.build_model(HERE)
    log = np.load(HERE / "joint_log.npz")
    renderer = look_common.LookRenderer(model, (HERE / "look.py").read_text())
    print(json.dumps(look_common.look_integrity(renderer)))
    from PIL import Image

    scores = []
    try:
        for frame in frames:
            image = renderer.render(log["q"][frame])
            if frame in recorded:
                score = twin_render.score(image, recorded[frame])
                scores.append(score)
                print(frame, {k: round(v, 4) for k, v in score.items()})
                image = np.concatenate([image, recorded[frame]], axis=1)
            Image.fromarray(image).save(args.out / f"f{frame:05d}.png")
    finally:
        renderer.close()
    if scores:
        print("mean", {k: round(float(np.mean([s[k] for s in scores])), 4) for k in scores[0]})


if __name__ == "__main__":
    main()
