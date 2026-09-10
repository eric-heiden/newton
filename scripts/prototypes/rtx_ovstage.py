# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare and replay source-scene poses through Newton's ovstage prototype.

Preparation uses Newton's existing USD importer and FK. Replay uses only
Newton pose arrays, ovstage and OVRTX; it does not run a solver or import USD
physics. The input USD must include its camera, lights and RenderProduct.
"""

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton


@wp.kernel
def _move(rest: wp.array[wp.transform], shift: float, poses: wp.array[wp.transform]):
    i = wp.tid()
    pose = rest[i]
    pose[0] += shift
    poses[i] = pose


def prepare(scene: Path, fixture: Path):
    from pxr import Gf, Usd, UsdGeom

    usd = Usd.Stage.Open(str(scene.resolve()))
    if usd is None:
        raise ValueError(f"Could not open {scene}")
    if UsdGeom.GetStageMetersPerUnit(usd) != 1.0 or UsdGeom.GetStageUpAxis(usd) != "Z":
        raise ValueError("This prototype fixture requires a meter-scale, Z-up USD scene")
    builder = newton.ModelBuilder()
    imported = builder.add_usd(usd, load_visual_shapes=False, skip_mesh_approximation=True)
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    poses = state.body_q.numpy()
    cache = UsdGeom.XformCache()
    paths, indices, offsets, worlds = [], [], [], []
    for path, body in imported["path_body_map"].items():
        if body < 0:
            continue
        pose = poses[body]
        body_world = Gf.Matrix4d(1)
        body_world.SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))
        body_world.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
        world = cache.GetLocalToWorldTransform(usd.GetPrimAtPath(path))
        paths.append(path)
        indices.append(body)
        offsets.append(np.asarray(world * body_world.GetInverse()))
        worlds.append(np.asarray(world))
    np.savez(fixture, body_q=poses, paths=paths, ids=indices, offsets=offsets, worlds=worlds)
    print(f"Prepared {len(paths)} driven prims / {model.body_count} Newton bodies", flush=True)


def replay(args):
    import ovrtx
    import ovstage  # noqa: PLC0415 - preparation does not require the rendering stack

    from newton.viewer import OvstageBodyBinding, ViewerRTX  # noqa: PLC0415

    fixture = np.load(args.fixture, allow_pickle=False)
    # Reconstruct a pose-array carrier with the same final body index space.
    # It deliberately contains no collision shapes or solver configuration.
    builder = newton.ModelBuilder()
    for pose in fixture["body_q"]:
        builder.add_body(xform=wp.transform(*pose))
    model = builder.finalize(device="cuda:0")
    state = model.state()
    rest = wp.clone(state.body_q)
    renderer = ovrtx.Renderer(config=ovrtx.RendererConfig(sync_mode=True))
    stage = ovstage.Stage("newton-source-scene")
    viewer = binding = None
    attached = False
    try:
        renderer.attach_ovstage(stage)
        attached = True
        ovstage.population.open_usd(
            stage, str(args.scene.resolve()), ordinal=1, domains=ovstage.PopulationDomain.ALL, time_code=0.0
        )
        stage.advance_write_floor(1).wait()
        binding = OvstageBodyBinding(
            stage,
            model,
            ordinal=1,
            prim_paths=fixture["paths"].tolist(),
            body_indices=fixture["ids"],
            body_local_transforms=fixture["offsets"],
        )
        viewer = ViewerRTX(
            stage=stage,
            renderer=renderer,
            render_product=args.render_product,
            headless=not args.window,
        )
        binding.write(state, ordinal=2)
        stage.advance_write_floor(2).wait()
        start = time.perf_counter()
        warm_frames = 0
        while time.perf_counter() - start < args.warmup or warm_frames < 15:
            viewer.render(ordinal=2)
            warm_frames += 1
        viewer.save_screenshot(str(args.output.with_suffix(".png")))
        frames, publication = [], []
        ordinal = 2
        for i in range(args.frames):
            ordinal += 1
            start = time.perf_counter()
            wp.launch(_move, model.body_count, inputs=[rest, 0.2 * np.sin(i * 0.11), state.body_q], device=model.device)
            binding.write(state, ordinal=ordinal)
            stage.advance_write_floor(ordinal).wait()
            publication.append((time.perf_counter() - start) * 1000)
            viewer.render(ordinal=ordinal)
            frames.append((time.perf_counter() - start) * 1000)
        viewer.save_screenshot(str(args.output.with_name(args.output.stem + "_moved.png")))
        result = {
            "scope": "Newton FK pose replay through prototype APIs; no solver, CPU window preview excluded when headless",
            "versions": {
                name: importlib.metadata.version(name) for name in ("newton", "ovrtx", "ovstage", "warp-lang")
            },
            "bodies": model.body_count,
            "targets": len(binding.prim_paths),
            "frames": len(frames),
            "median_frame_ms": float(np.median(frames)),
            "p95_frame_ms": float(np.percentile(frames, 95)),
            "median_publication_ms": float(np.median(publication)),
            "pxr_imported": "pxr" in sys.modules,
            "ovnewton_imported": "ovnewton" in sys.modules,
        }
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        if binding is not None:
            binding.close()
        if attached:
            renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("scene", type=Path)
    prep.add_argument("fixture", type=Path)
    render = commands.add_parser("replay")
    render.add_argument("scene", type=Path)
    render.add_argument("fixture", type=Path)
    render.add_argument("--render-product", required=True)
    render.add_argument("--output", type=Path, default=Path("ovstage-prototype.png"))
    render.add_argument("--frames", type=int, default=60)
    render.add_argument("--warmup", type=float, default=30)
    render.add_argument("--window", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.scene, args.fixture)
    else:
        if args.frames < 1 or args.warmup < 0:
            parser.error("frames must be positive and warmup must be nonnegative")
        replay(args)


if __name__ == "__main__":
    main()
