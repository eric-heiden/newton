# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the flying-knot example with ViewerGL (headless) to an mp4 video.

Usage:
  uv run python scripts/flying_knot/record_video.py OUT.mp4 [--slowmo 4] [--width 1280]
      [--camera default|side|closeup|front|track] [example args...]

The ``track`` camera follows the knot as it forms: each frame it targets the
smoothed centroid of the rope's self-contact cluster (fallback: rope centroid)
and verifies the cluster stays inside the frame (--verify-in-frame).
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
    ap.add_argument("--require-knot", action="store_true", dest="require_knot")
    ap.add_argument("--verify-in-frame", action="store_true", dest="verify_in_frame")
    ap.add_argument(
        "--max-knot-pos",
        type=float,
        default=None,
        dest="max_knot_pos",
        help="fail (exit 5) if the final knot sits below this arc-length fraction (0=handle, 1=tip)",
    )
    args, extra = ap.parse_known_args()

    import imageio_ffmpeg  # noqa: PLC0415
    from example_cable_flying_knot import Example, add_arguments  # noqa: PLC0415

    import newton.examples  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415

    parser = newton.examples.create_parser()
    add_arguments(parser)
    ex_args = parser.parse_args(["--viewer", "gl", "--headless", *extra])

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    example = Example(viewer, ex_args)

    # Camera presets (tuned for the flying-knot scene).
    presets = {
        "default": (np.array([0.72, -0.10, example.z_offset + 0.22]), np.array([1.55, -1.30, 0.18])),
        "side": (np.array([0.70, -0.05, example.z_offset + 0.28]), np.array([0.10, -1.95, 0.15])),
        "closeup": (np.array([0.85, -0.22, example.z_offset + 0.22]), np.array([1.15, 0.55, 0.12])),
        "front": (np.array([0.72, -0.10, example.z_offset + 0.35]), np.array([1.7, 0.3, 0.2])),
    }
    track_delta = np.array([0.30, -1.55, 0.18])
    if args.camera == "track":
        cam_target, delta = np.array([0.72, -0.10, example.z_offset + 0.2]), track_delta
    else:
        cam_target, delta = presets[args.camera]
    pos = cam_target + delta

    # Brighter studio-like backdrop so the rope reads clearly.
    r = viewer.renderer
    r.sky_upper = (0.62, 0.70, 0.80)
    r.sky_lower = (0.42, 0.44, 0.50)
    r.background_color = r.sky_upper

    def aim_camera(target, cam_pos):
        d = target - cam_pos
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        pitch = float(np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2]))))
        viewer.set_camera(wp.vec3(*cam_pos.tolist()), pitch, yaw)

    aim_camera(cam_target, pos)

    def knot_cluster(nodes, sep=4, thresh=0.05):
        """Indices of rope nodes in self-contact proximity (the knot/loop)."""
        diff = nodes[:, None, :] - nodes[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        n = len(nodes)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = (jj - ii >= sep) & (dist < thresh)
        return np.unique(np.concatenate([ii[mask], jj[mask]]))

    def ndc_coords(pts, cam_pos, target):
        """Normalized device coords of points for the current camera aim."""
        f = target - cam_pos
        f = f / np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        fov = getattr(viewer.camera, "fov", 45.0)
        tanv = np.tan(np.radians(fov / 2))
        aspect = args.width / args.height
        v = pts - cam_pos
        z = v @ f
        return (v @ r) / (z * tanv * aspect), (v @ u) / (z * tanv)

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
    smoothed = None
    ndc_max = []
    for _f in range(n_frames):
        example.step()
        nodes = example.rope_traj[-1]
        if args.camera == "track":
            cluster = knot_cluster(nodes)
            focus = nodes[cluster].mean(axis=0) if len(cluster) else nodes.mean(axis=0)
            if smoothed is None:
                smoothed = focus
            else:
                # Adaptive smoothing: follow tightly when the knot moves fast,
                # settle to a calm camera when it is slow.
                alpha = float(np.clip(0.08 + 2.5 * np.linalg.norm(focus - smoothed), 0.08, 0.45))
                smoothed = (1 - alpha) * smoothed + alpha * focus
            pos = smoothed + track_delta
            aim_camera(smoothed, pos)
        example.render()
        t = example.sim_time
        if t < args.start or t > args.end:
            continue
        if args.camera == "track":
            cluster = knot_cluster(nodes)
            pts = nodes[cluster] if len(cluster) else nodes
            x, y = ndc_coords(pts, pos, smoothed)
            ndc_max.append(max(np.abs(x).max(), np.abs(y).max()))
        frame = example.viewer.get_frame().numpy()
        dup = max(1, round(args.slowmo))
        for _ in range(dup):
            writer.send(np.ascontiguousarray(frame))
            written += 1
    writer.close()
    print(f"wrote {args.output}: {written} frames at {args.fps} fps ({written / args.fps:.1f}s)")

    if ndc_max:
        ndc_max = np.array(ndc_max)
        print(
            f"knot-cluster framing over {len(ndc_max)} recorded frames: "
            f"max |ndc| {ndc_max.max():.2f}, mean {ndc_max.mean():.2f}, "
            f"frames fully in view (<1.0): {(ndc_max < 1.0).sum()}/{len(ndc_max)}"
        )
        if args.verify_in_frame and (ndc_max.max() >= 0.95 or ndc_max.mean() >= 0.8):
            print("knot cluster left the frame; exiting 4 for retry")
            sys.exit(4)

    if args.max_knot_pos is not None:
        nodes = example.rope_traj[-1]
        idx = knot_cluster(nodes)
        seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        knot_pos = float(arc[idx].mean() / arc[-1]) if len(idx) else 1.0
        print(f"final knot arc-length position: {knot_pos:.2f} (0=handle, 1=tip)")
        if knot_pos > args.max_knot_pos:
            print(f"knot ended too close to the rope end (> {args.max_knot_pos}); exiting 5 for retry")
            sys.exit(5)

    example.test_final()
    if args.require_knot:
        m = example.knot_metrics(example.rope_traj[-1])
        if not (abs(m["writhe"]) > 2.0 and m["length_ratio"] < 0.95):
            print("no knot in this run; exiting 3 for retry")
            sys.exit(3)


if __name__ == "__main__":
    main()
