# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render HRDexDB replay overlay videos with headless ViewerGL.

The simulated scene (robot + passive object) is rendered solid; the
ground-truth object pose is overlaid as a semi-transparent green ghost via
``viewer.log_shapes`` — it never participates in physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton

from dataset import load_episode  # noqa: E402
from replay import Replayer  # noqa: E402
from scene import SimParams  # noqa: E402


def render_episode(
    ep,
    params: SimParams | None = None,
    target_source: str = "cmd",
    substeps: int = 4,
    output: Path | str = "replay.mp4",
    width: int = 1600,
    height: int = 1000,
    fps: int = 50,
    camera: str = "front",
):
    import imageio_ffmpeg

    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    rep = Replayer(ep, params, target_source=target_source, substeps=substeps)
    viewer.set_model(rep.model)

    r = viewer.renderer
    r.sky_upper = (0.65, 0.72, 0.82)
    r.sky_lower = (0.42, 0.44, 0.50)
    r.background_color = r.sky_upper

    # Aim at the object's ground-truth trajectory midpoint.
    focus = ep.obj_poses[:, :3, 3].mean(axis=0)
    cam_offsets = {
        "front": np.array([0.85, 0.0, 0.45]),
        "side": np.array([0.1, -0.9, 0.4]),
        "top": np.array([0.3, 0.0, 1.0]),
        "closeup": np.array([0.45, -0.25, 0.25]),
    }
    cam_pos = focus + cam_offsets[camera]
    d = focus - cam_pos
    yaw = float(np.degrees(np.arctan2(d[1], d[0])))
    pitch = float(np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2]))))
    viewer.set_camera(wp.vec3(*cam_pos.tolist()), pitch, yaw)

    gt_color = wp.array([wp.vec3(0.1, 0.9, 0.2)], dtype=wp.vec3)
    gt_opacity = wp.array([0.4], dtype=wp.float32)
    mesh = rep.info.object_mesh_newton

    def gt_overlay(t: float):
        T = ep.obj_poses_at(np.array([t]))[0]
        from scene import _mat44_to_transform

        xf = wp.array([_mat44_to_transform(T)], dtype=wp.transform)
        viewer.log_shapes(
            "/gt_object",
            newton.GeoType.MESH,
            1.0,
            xf,
            colors=gt_color,
            opacities=gt_opacity,
            geo_src=mesh,
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (width, height), fps=fps, quality=8, macro_block_size=8)
    writer.send(None)

    control_hz = 1.0 / rep.control_dt
    render_every = max(1, round(control_hz / fps))

    n_steps = len(ep.t)
    rec_obj = wp.zeros((n_steps, 7), dtype=wp.float32)
    rec_q = wp.zeros((n_steps, len(rep.info.dof_map)), dtype=wp.float32)
    frames = 0
    for i in range(n_steps):
        rep._control_step(1, rec_obj, rec_q)
        if i % render_every == 0:
            viewer.begin_frame(float(ep.t[i]))
            viewer.log_state(rep.state_0)
            gt_overlay(float(ep.t[i]))
            viewer.end_frame()
            frame = viewer.get_frame().numpy()
            writer.send(np.ascontiguousarray(frame))
            frames += 1
    writer.close()
    print(f"wrote {output}: {frames} frames ({frames / fps:.1f}s)")
    return rec_obj.numpy(), rec_q.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="allegro_v5")
    parser.add_argument("--object", default="banana")
    parser.add_argument("--scene", default="2")
    parser.add_argument("--target-source", default="cmd", choices=["cmd", "meas"])
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--camera", default="front", choices=["front", "side", "top", "closeup"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ep = load_episode(args.hand, args.object, args.scene)
    raw = {k: v for k, v in json.loads(Path(args.params).read_text()).items() if not k.startswith("_")} if args.params else {}
    params = SimParams(**raw)
    out = args.output or f"videos/{args.hand}_{args.object}_{args.scene}_{args.target_source}_{args.camera}.mp4"
    render_episode(ep, params, target_source=args.target_source, output=out, camera=args.camera)


if __name__ == "__main__":
    main()
