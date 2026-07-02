# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the flying-knot example with ViewerGL (headless) to an mp4 video.

Usage:
  uv run python scripts/flying_knot/record_video.py OUT.mp4 [--slowmo 4] [--width 1280]
      [--camera default|side|closeup] [example args...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--slowmo", type=float, default=1.0, help="output slowdown factor (frame duplication)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--start", type=float, default=0.0, help="start recording at sim time [s]")
    ap.add_argument("--end", type=float, default=1.0e9, help="stop recording at sim time [s]")
    ap.add_argument("--camera", default="default")
    args, extra = ap.parse_known_args()

    import imageio_ffmpeg

    import newton.examples
    import newton.viewer
    from example_cable_flying_knot import Example, add_arguments

    parser = newton.examples.create_parser()
    add_arguments(parser)
    ex_args = parser.parse_args(["--viewer", "gl", "--headless", *extra])

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    example = Example(viewer, ex_args)

    # Camera presets (tuned for the flying-knot scene).
    cam_target = np.array([0.72, -0.08, example.z_offset + 0.25])
    presets = {
        "default": (cam_target + np.array([2.3, -2.1, 0.55]), None),
        "side": (cam_target + np.array([0.3, -2.9, 0.35]), None),
        "closeup": (cam_target + np.array([1.5, -1.4, 0.2]), None),
        "front": (cam_target + np.array([2.9, 0.4, 0.4]), None),
    }
    pos, _ = presets[args.camera]
    d = cam_target - pos
    yaw = float(np.degrees(np.arctan2(d[1], d[0])))
    pitch = float(np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2]))))
    viewer.set_camera(wp.vec3(*pos.tolist()), pitch, yaw)

    writer = imageio_ffmpeg.write_frames(
        args.output,
        (args.width, args.height),
        fps=args.fps,
        quality=8,
        macro_block_size=8,
    )
    writer.send(None)

    n_frames = example.num_frames
    written = 0
    for _f in range(n_frames):
        example.step()
        example.render()
        t = example.sim_time
        if t < args.start or t > args.end:
            continue
        frame = example.viewer.get_frame().numpy()
        dup = max(1, round(args.slowmo))
        for _ in range(dup):
            writer.send(np.ascontiguousarray(frame))
            written += 1
    writer.close()
    print(f"wrote {args.output}: {written} frames at {args.fps} fps ({written / args.fps:.1f}s)")

    example.test_final()


if __name__ == "__main__":
    main()
