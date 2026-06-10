# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a short ViewerGL clip for the tricycle origin-offset report."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.robot.example_robot_tricycle_origin_offset import Example


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/featherstone_origin_offset/assets/tricycle_origin_offset.mp4"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--far-offset", type=float, default=100.0)
    parser.add_argument("--drive-torque", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    """Capture the tricycle example and encode it as MP4."""
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the report video")

    wp.set_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    example_args = argparse.Namespace(
        far_offset=args.far_offset,
        invariance_tolerance=0.25,
        print_metrics=False,
        drive_torque=args.drive_torque,
    )

    writer: subprocess.Popen[bytes] | None = None
    frame_buffer = None
    try:
        example = Example(viewer, example_args)
        frame_count = max(1, round(args.duration * args.fps))
        sim_steps_per_frame = max(1, round(example.fps / args.fps))

        for _ in range(frame_count):
            example.render()
            frame_buffer = viewer.get_frame(target_image=frame_buffer)
            frame = np.ascontiguousarray(frame_buffer.numpy())
            if writer is None:
                height, width = frame.shape[:2]
                writer = subprocess.Popen(
                    [
                        ffmpeg,
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        str(args.fps),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        "-crf",
                        "23",
                        "-preset",
                        "medium",
                        str(args.output),
                    ],
                    stdin=subprocess.PIPE,
                )
                assert writer.stdin is not None

            assert writer.stdin is not None
            writer.stdin.write(frame.tobytes())

            for _ in range(sim_steps_per_frame):
                sim_time = example.sim_time
                example.step()
                if example.sim_time == sim_time:
                    example.sim_time += example.frame_dt

        if writer is not None:
            assert writer.stdin is not None
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {writer.returncode}")
    finally:
        if writer is not None and writer.poll() is None:
            writer.kill()
        viewer.close()


if __name__ == "__main__":
    main()
