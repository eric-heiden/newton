# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record a substep-resolution USD rollout of the flying knot with ViewerUSD.

Every simulation substep inside the recording window is written as a USD
time sample, so the file can be retimed to extreme slow motion in a DCC
tool. The USD stage's timeCodesPerSecond matches the substep rate (e.g.
2880 for 48 substeps at 60 FPS), so default playback is real time.

Usage:
  uv run python scripts/flying_knot/usd_rollout.py OUT.usd \
      [--start 1.4] [--end 5.0] [--sample-every 1] [example args...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "newton" / "examples" / "cable"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--start", type=float, default=1.4, help="Recording window start [sim s].")
    ap.add_argument("--end", type=float, default=5.0, help="Recording window end [sim s].")
    ap.add_argument("--sample-every", type=int, default=1, dest="sample_every", help="Log every Nth substep.")
    args, extra = ap.parse_known_args()

    from example_cable_flying_knot_mujoco import Example, add_arguments

    import newton.examples
    import newton.viewer

    parser = newton.examples.create_parser()
    add_arguments(parser)
    ex_args = parser.parse_args(["--viewer", "null", *extra])

    # Build the example against a temporary null viewer, then attach ViewerUSD
    # once the substep rate is known (the searched-hires preset raises it).
    class _NullSink:
        def set_model(self, model):
            self.model = model

        def set_camera(self, *a, **k):
            pass

        def apply_forces(self, state):
            pass

    sink = _NullSink()
    example = Example(sink, ex_args)

    sample_hz = example.fps * example.sim_substeps / args.sample_every
    print(f"substep rate {example.fps * example.sim_substeps} Hz, sampling at {sample_hz:.0f} Hz")
    viewer = newton.viewer.ViewerUSD(output_path=args.output, fps=int(round(sample_hz)), num_frames=None)
    viewer.set_model(example.model)
    example.viewer = viewer

    start_frame = int(args.start * example.fps)
    n_window_frames = int(np.ceil((args.end - args.start) * example.fps))
    total_frames = min(example.num_frames, start_frame + n_window_frames)

    # Fast-forward to the window with the captured full-frame graph.
    for _ in range(start_frame):
        example.step()
    print(f"fast-forwarded to t={example.sim_time:.2f}s; recording to t={args.end:.2f}s")

    sample = 0
    frame = start_frame
    while frame < total_frames:
        for k in range(example.sim_substeps):
            example.substep()
            if (frame * example.sim_substeps + k) % args.sample_every == 0:
                t = (sample + 0.5) / sample_hz
                viewer.begin_frame(t)
                viewer.log_state(example.state_0)
                viewer.end_frame()
                sample += 1
        example.sim_time += example.frame_dt
        example.frame_index += 1
        example.rope_traj.append(example.rope_centerline(example.state_0.body_q.numpy()))
        frame += 1
        if frame % 30 == 0:
            print(f"  t={example.sim_time:.2f}s ({sample} samples)", flush=True)

    # Continue (unrecorded) to the end so the knot metrics reflect the final state.
    while frame < example.num_frames:
        example.step()
        frame += 1

    viewer.close()
    metrics = example.knot_metrics(example.rope_traj[-1])
    print(
        f"final rope metrics: writhe {metrics['writhe']:+.2f}, crossings {metrics['crossings']}, "
        f"end-to-end/arc {metrics['length_ratio']:.3f}"
    )
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"wrote {args.output}: {sample} time samples at {sample_hz:.0f} Hz ({size_mb:.1f} MB)")
    if not (abs(metrics["writhe"]) > 2.0 and metrics["length_ratio"] < 0.95):
        print("no knot in this rollout; exiting 3 for retry")
        sys.exit(3)


if __name__ == "__main__":
    main()
