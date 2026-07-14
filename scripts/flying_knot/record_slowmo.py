# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""True slow-motion rendering of the flying knot from sub-frame sim states.

Unlike record_video.py --slowmo (which duplicates 60 Hz frames), this
renders every Nth simulation substep, so a 4x slow-motion video contains
genuine 240 Hz motion. The camera tracks the knot cluster with a
time-constant-based exponential smoother and biases the view toward the
robot so both stay in frame where possible; the knot's screen position is
verified every rendered frame.

Usage:
  uv run python scripts/flying_knot/record_slowmo.py OUT.mp4 --slowdown 4 \
      [--start 1.4] [--end 5.0] [example args...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))

ROBOT_ANCHOR = np.array([0.42, -0.22, 1.55])  # around the arm's elbow region


def knot_cluster(nodes, sep=4, thresh=0.05):
    """Indices of rope nodes in self-contact proximity (the knot/loop)."""
    diff = nodes[:, None, :] - nodes[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    n = len(nodes)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = (jj - ii >= sep) & (dist < thresh)
    return np.unique(np.concatenate([ii[mask], jj[mask]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--slowdown", type=float, default=4.0, help="Slow-motion factor (4 = quarter speed).")
    ap.add_argument("--start", type=float, default=1.4)
    ap.add_argument("--end", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--require-knot", action="store_true", dest="require_knot")
    args, extra = ap.parse_known_args()

    import imageio_ffmpeg  # noqa: PLC0415

    import newton.examples  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415
    from example_cable_flying_knot_mujoco import Example, add_arguments  # noqa: PLC0415

    parser = newton.examples.create_parser()
    add_arguments(parser)
    ex_args = parser.parse_args(["--viewer", "gl", "--headless", *extra])

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    example = Example(viewer, ex_args)

    stride = max(1, int(round(example.sim_substeps / args.slowdown)))
    capture_hz = example.fps * example.sim_substeps / stride
    dt_capture = 1.0 / capture_hz
    print(
        f"substeps/frame {example.sim_substeps}, stride {stride} -> capture {capture_hz:.0f} Hz, "
        f"output {args.fps} fps = {capture_hz / args.fps:.1f}x slowdown"
    )

    # Studio backdrop as in record_video.py.
    r = viewer.renderer
    r.sky_upper = (0.62, 0.70, 0.80)
    r.sky_lower = (0.42, 0.44, 0.50)
    r.background_color = r.sky_upper

    cam_delta = np.array([0.40, -2.15, 0.30])

    def aim_camera(target, cam_pos):
        d = target - cam_pos
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        pitch = float(np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2]))))
        viewer.set_camera(wp.vec3(*cam_pos.tolist()), pitch, yaw)

    def ndc_coords(pts, cam_pos, target):
        f = target - cam_pos
        f = f / np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        rt = np.cross(f, up)
        rt = rt / np.linalg.norm(rt)
        u = np.cross(rt, f)
        fov = getattr(viewer.camera, "fov", 45.0)
        tanv = np.tan(np.radians(fov / 2))
        aspect = args.width / args.height
        v = pts - cam_pos
        z = np.maximum(v @ f, 1e-6)
        return (v @ rt) / (z * tanv * aspect), (v @ u) / (z * tanv)

    writer = imageio_ffmpeg.write_frames(
        args.output, (args.width, args.height), fps=args.fps, quality=8, macro_block_size=8
    )
    writer.send(None)

    start_frame = int(args.start * example.fps)
    end_frame = min(example.num_frames, int(np.ceil(args.end * example.fps)))
    for _ in range(start_frame):
        example.step()

    smoothed = None
    written = 0
    ndc_max = []
    tau_calm, tau_fast = 0.45, 0.12  # camera smoothing time constants [s]
    for frame in range(start_frame, end_frame):
        for k in range(example.sim_substeps):
            example.substep()
            if k % stride != 0:
                continue
            nodes = example.rope_centerline(example.state_0.body_q.numpy())
            cluster = knot_cluster(nodes)
            focus = nodes[cluster].mean(axis=0) if len(cluster) else nodes[len(nodes) // 2]
            if smoothed is None:
                smoothed = focus.copy()
            speed = np.linalg.norm(focus - smoothed)
            tau = tau_fast if speed > 0.25 else tau_calm
            alpha = 1.0 - np.exp(-dt_capture / tau)
            smoothed = (1 - alpha) * smoothed + alpha * focus
            # Bias the look target toward the robot so both subjects share the
            # frame; the knot keeps priority through the tighter tracking.
            look = 0.72 * smoothed + 0.28 * ROBOT_ANCHOR
            pos = look + cam_delta
            aim_camera(look, pos)

            example.sim_time = frame * example.frame_dt + (k + 1) * example.sim_dt
            viewer.begin_frame(example.sim_time)
            viewer.log_state(example.state_0)
            viewer.end_frame()

            pts = nodes[cluster] if len(cluster) else nodes
            x, y = ndc_coords(pts, pos, look)
            ndc_max.append(max(np.abs(x).max(), np.abs(y).max()))

            img = viewer.get_frame().numpy()
            writer.send(np.ascontiguousarray(img))
            written += 1
        example.frame_index += 1
        example.rope_traj.append(example.rope_centerline(example.state_0.body_q.numpy()))
        if frame % 30 == 0:
            print(f"  t={frame / example.fps:.2f}s ({written} frames)", flush=True)
    example.sim_time = end_frame * example.frame_dt

    for _ in range(end_frame, example.num_frames):
        example.step()

    writer.close()
    ndc_max = np.array(ndc_max)
    print(f"wrote {args.output}: {written} frames at {args.fps} fps ({written / args.fps:.1f}s video)")
    print(
        f"knot framing: max |ndc| {ndc_max.max():.2f}, mean {ndc_max.mean():.2f}, "
        f"in view (<1.0): {(ndc_max < 1.0).sum()}/{len(ndc_max)}"
    )

    metrics = example.knot_metrics(example.rope_traj[-1])
    print(
        f"final rope metrics: writhe {metrics['writhe']:+.2f}, crossings {metrics['crossings']}, "
        f"end-to-end/arc {metrics['length_ratio']:.3f}"
    )
    if args.require_knot and not (abs(metrics["writhe"]) > 2.0 and metrics["length_ratio"] < 0.95):
        print("no knot in this rollout; exiting 3 for retry")
        sys.exit(3)
    if ndc_max.max() >= 1.0:
        print("warning: knot cluster left the frame on some frames")


if __name__ == "__main__":
    main()
