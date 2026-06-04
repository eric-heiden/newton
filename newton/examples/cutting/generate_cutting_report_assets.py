# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate report media for the experimental cutting examples."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton.examples.cutting.cutting_common import (
    AdaptiveCutSurfaceRemesher,
    KnifeProfile,
    RuntimeStats,
    SplitCuboidRenderMesh,
    TetMeshCutSurfaceRenderer,
    capture_viewer_frame,
    collect_hardware_details,
    encode_mp4,
    export_remesh_artifacts,
    save_first_frame,
    summarize_force_profile,
    write_force_plot,
    write_json,
)
from newton.examples.cutting.example_cutting_mpm import Example as MPMExample
from newton.examples.cutting.example_cutting_vbd import Example as VBDExample
from newton.examples.cutting.example_cutting_xfem import Example as XFEMExample


class _NullViewer:
    def set_model(self, model):
        self.model = model

    def apply_forces(self, state):
        return None

    def log_contacts(self, *args, **kwargs):
        return None

    def begin_frame(self, *args, **kwargs):
        return None

    def end_frame(self, *args, **kwargs):
        return None

    def log_mesh(self, *args, **kwargs):
        return None

    def log_points(self, *args, **kwargs):
        return None

    def log_lines(self, *args, **kwargs):
        return None


def _parse_example_args(example_cls, argv: list[str], viewer_name: str = "null", headless: bool = False):
    old_argv = sys.argv
    try:
        sys.argv = ["generate_cutting_report_assets", "--viewer", viewer_name, "--quiet"]
        if headless:
            sys.argv.append("--headless")
        sys.argv.extend(argv)
        parser = example_cls.create_parser()
        viewer, args = newton.examples.init(parser)
        return viewer, args
    finally:
        sys.argv = old_argv


def _as_numpy_vec3(values) -> np.ndarray:
    if values is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(values, wp.array):
        return values.numpy().astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def _particle_colors(example) -> np.ndarray:
    colors = getattr(example, "particle_colors", None)
    if colors is None and hasattr(example, "solver"):
        colors = getattr(example.solver, "particle_colors", None)
    if colors is None:
        return np.tile(np.array([[0.16, 0.42, 0.76]], dtype=np.float32), (example.model.particle_count, 1))
    return np.clip(_as_numpy_vec3(colors), 0.0, 1.0)


def _knife_profile_for_example(example, time_value: float) -> KnifeProfile | None:
    if hasattr(example, "_knife_state"):
        cfg = example.scenario
        front_x, center_z, _velocity = example._knife_state(time_value)
        base = getattr(example, "knife_profile", None)
        return KnifeProfile(
            start_x=front_x,
            speed=0.0,
            center_y=cfg.knife_center_y,
            center_z=center_z,
            half_width_y=cfg.knife_half_width_y,
            half_width_z=cfg.knife_half_width_z,
            process_width=cfg.process_width,
            edge_control_points=base.edge_control_points if base is not None else (),
        )

    knife = getattr(example, "knife", None)
    if knife is None:
        return None
    return knife


def _knife_geometry(example, time_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knife = _knife_profile_for_example(example, time_value)
    if knife is None:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
        )

    render_time = 0.0 if hasattr(example, "_knife_state") else time_value
    vertices, indices = knife.blade_mesh(render_time)
    edge_points = knife.edge_points(render_time)
    return vertices.astype(np.float32, copy=False), indices.astype(np.int32, copy=False), edge_points.astype(
        np.float32, copy=False
    )


def _snapshot_render_geometry(example) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    render_mesh = getattr(example, "render_split_mesh", None)
    if render_mesh is None:
        points = example.state_0.particle_q.numpy().astype(np.float32, copy=False)
        indices = (
            example.model.tri_indices.numpy().reshape(-1, 3).astype(np.int32, copy=False)
            if example.model.tri_indices is not None
            else np.zeros((0, 3), dtype=np.int32)
        )
        return {
            "surface_points": points,
            "surface_indices": indices,
            "wall_points": np.zeros((0, 3), dtype=np.float32),
            "wall_indices": np.zeros((0, 3), dtype=np.int32),
        }, {}

    if isinstance(render_mesh, TetMeshCutSurfaceRenderer):
        front_x = None
        center_z = None
        if hasattr(example, "_knife_state"):
            front_x, center_z, _velocity = example._knife_state(example.sim_time)
        stats = render_mesh.update(
            example.state_0.particle_q,
            example.sim_time,
            front_x=front_x,
            center_z=center_z,
            enrichment_points=getattr(getattr(example, "solver", None), "particle_enrichment_q", None),
        )
        surface_vertices = stats.surface_vertex_count
        wall_vertices = stats.wall_vertex_count
        return {
            "surface_points": render_mesh.surface_points_np[:surface_vertices].copy(),
            "surface_indices": np.arange(surface_vertices, dtype=np.int32).reshape(-1, 3),
            "wall_points": render_mesh.wall_points_np[:wall_vertices].copy(),
            "wall_indices": np.arange(wall_vertices, dtype=np.int32).reshape(-1, 3),
        }, asdict(stats)

    if isinstance(render_mesh, AdaptiveCutSurfaceRemesher):
        stats = render_mesh.update(
            example.model.device,
            example.sim_time,
            rest_particle_points=getattr(example, "render_rest_particle_q_wp", None),
            particle_points=example.state_0.particle_q,
        )
        surface_points = render_mesh.surface_points_wp.numpy()[: stats.surface_vertex_count]
        surface_indices = render_mesh.surface_indices_wp.numpy()[: stats.surface_triangle_count * 3].reshape(-1, 3)
        wall_points = render_mesh.wall_points_wp.numpy()[: stats.wall_vertex_count]
        wall_indices = render_mesh.wall_indices_wp.numpy()[: stats.wall_triangle_count * 3].reshape(-1, 3)
        return {
            "surface_points": surface_points.astype(np.float32, copy=False),
            "surface_indices": surface_indices.astype(np.int32, copy=False),
            "wall_points": wall_points.astype(np.float32, copy=False),
            "wall_indices": wall_indices.astype(np.int32, copy=False),
        }, asdict(stats)

    if isinstance(render_mesh, SplitCuboidRenderMesh):
        rest = getattr(example, "render_rest_particle_q", None)
        current = example.state_0.particle_q.numpy()
        surface_points, wall_points = render_mesh.build_points(
            example.sim_time,
            rest_particle_points=rest,
            particle_points=current,
        )
        return {
            "surface_points": surface_points,
            "surface_indices": render_mesh.surface_indices_np.reshape(-1, 3),
            "wall_points": wall_points,
            "wall_indices": render_mesh.wall_indices_np.reshape(-1, 3),
        }, {}

    raise TypeError(f"unsupported render mesh type: {type(render_mesh).__name__}")


def _snapshot(example) -> tuple[dict[str, object], dict[str, float]]:
    geometry, stats = _snapshot_render_geometry(example)
    knife_vertices, knife_indices, knife_edge_points = _knife_geometry(example, example.sim_time)
    particles = example.state_0.particle_q.numpy().astype(np.float32, copy=False)
    colors = _particle_colors(example)
    return {
        **geometry,
        "particles": particles.copy(),
        "particle_colors": colors.copy(),
        "knife_vertices": knife_vertices.copy(),
        "knife_indices": knife_indices.copy(),
        "knife_edge_points": knife_edge_points.copy(),
        "time": float(example.sim_time),
    }, stats


def _rgb_frame_from_figure(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3].copy()


def _plot_table(ax, bounds: np.ndarray):
    x0, y0, z0 = bounds[0]
    x1, y1, _z1 = bounds[1]
    xx = np.array([[x0, x1], [x0, x1]], dtype=np.float32)
    yy = np.array([[y0, y0], [y1, y1]], dtype=np.float32)
    zz = np.zeros_like(xx) + min(0.0, float(z0))
    ax.plot_surface(xx, yy, zz, color="#e5e7eb", alpha=0.36, linewidth=0, shade=False)


def _render_frame(
    snapshot: dict[str, object],
    history,
    frame_index: int,
    bounds: np.ndarray,
    title: str,
    surface_color: tuple[float, float, float],
    wall_color: tuple[float, float, float],
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    fig = plt.figure(figsize=(12.8, 7.2), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax_profile = fig.add_subplot(gs[0, 1])

    surface_points = snapshot["surface_points"]
    surface_indices = snapshot["surface_indices"]
    wall_points = snapshot["wall_points"]
    wall_indices = snapshot["wall_indices"]
    particles = snapshot["particles"]
    particle_colors = snapshot["particle_colors"]

    _plot_table(ax, bounds)
    if len(surface_indices):
        surface_tris = surface_points[surface_indices]
        collection = Poly3DCollection(
            surface_tris,
            facecolors=(*surface_color, 0.58),
            edgecolors=(0.08, 0.10, 0.13, 0.10),
            linewidths=0.05,
        )
        ax.add_collection3d(collection)
    if len(wall_indices):
        wall_tris = wall_points[wall_indices]
        wall_collection = Poly3DCollection(
            wall_tris,
            facecolors=(*wall_color, 0.88),
            edgecolors=(0.18, 0.04, 0.04, 0.18),
            linewidths=0.08,
        )
        ax.add_collection3d(wall_collection)

    if len(particles):
        stride = max(1, len(particles) // 1800)
        ax.scatter(
            particles[::stride, 0],
            particles[::stride, 1],
            particles[::stride, 2],
            c=particle_colors[::stride],
            s=8.0 if len(particles) < 1200 else 4.5,
            alpha=0.88,
            depthshade=False,
        )

    knife_vertices = snapshot["knife_vertices"]
    knife_indices = snapshot["knife_indices"]
    knife_edge_points = snapshot["knife_edge_points"]
    if len(knife_indices):
        knife_tris = knife_vertices[knife_indices]
        facecolors = np.tile(np.array([[0.52, 0.57, 0.64, 0.96]], dtype=np.float32), (len(knife_indices), 1))
        knife_collection = Poly3DCollection(
            knife_tris,
            facecolors=facecolors,
            edgecolors=(0.03, 0.04, 0.05, 0.28),
            linewidths=0.16,
        )
        ax.add_collection3d(knife_collection)
    if len(knife_edge_points) > 1:
        ax.plot(
            knife_edge_points[:, 0],
            knife_edge_points[:, 1],
            knife_edge_points[:, 2],
            color="#111827",
            linewidth=4.4,
            alpha=0.96,
        )

    margin = 0.06
    ax.set_xlim(float(bounds[0, 0] - margin), float(bounds[1, 0] + margin))
    ax.set_ylim(float(bounds[0, 1] - margin), float(bounds[1, 1] + margin))
    ax.set_zlim(float(min(-0.01, bounds[0, 2] - margin)), float(bounds[1, 2] + margin))
    extents = np.maximum(bounds[1] - bounds[0], 0.05)
    ax.set_box_aspect((float(extents[0]), float(extents[1]), float(max(extents[2], 0.18))))
    ax.view_init(elev=22.0, azim=-54.0)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(True, alpha=0.22)

    times = np.asarray(history.times, dtype=np.float32)
    forces = np.asarray(history.forces, dtype=np.float32)
    damage = np.asarray(history.mean_damage, dtype=np.float32)
    ax_profile.plot(times, forces, color="#b91c1c", linewidth=2.0, label="knife force")
    if history.normal_forces:
        ax_profile.plot(times, history.normal_forces, color="#f97316", linewidth=1.2, label="normal")
    if history.friction_forces:
        ax_profile.plot(times, history.friction_forces, color="#7c3aed", linewidth=1.2, label="friction")
    ax_damage = ax_profile.twinx()
    ax_damage.plot(times, damage, color="#1d4ed8", linewidth=1.6, label="mean damage")
    if len(times):
        current_time = float(snapshot["time"])
        ax_profile.axvline(current_time, color="#111827", linewidth=1.2, alpha=0.72)
        ax_profile.scatter([current_time], [forces[min(frame_index, len(forces) - 1)]], color="#b91c1c", s=24, zorder=4)
    ax_profile.set_title("Force profile", loc="left", fontsize=12, fontweight="bold")
    ax_profile.set_xlabel("time [s]")
    ax_profile.set_ylabel("force [N]", color="#b91c1c")
    ax_damage.set_ylabel("mean damage", color="#1d4ed8")
    ax_profile.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.75)
    ax_profile.set_xlim(float(times[0]) if len(times) else 0.0, float(times[-1]) if len(times) else 1.0)
    if len(forces):
        ax_profile.set_ylim(0.0, max(1.0, float(np.max(forces)) * 1.08))
    ax_damage.set_ylim(0.0, max(0.05, float(np.max(damage)) * 1.18 if len(damage) else 1.0))
    lines = ax_profile.get_lines() + ax_damage.get_lines()
    labels = [line.get_label() for line in lines if not line.get_label().startswith("_")]
    handles = [line for line in lines if not line.get_label().startswith("_")]
    ax_profile.legend(handles, labels, loc="upper right", fontsize=8)

    frame = _rgb_frame_from_figure(fig)
    plt.close(fig)
    return frame


def _bounds_from_snapshots(snapshots: list[dict[str, object]]) -> np.ndarray:
    mins = []
    maxs = []
    for snapshot in snapshots:
        for key in ("surface_points", "wall_points", "particles", "knife_vertices"):
            points = snapshot[key]
            if len(points):
                mins.append(np.min(points, axis=0))
                maxs.append(np.max(points, axis=0))
    if not mins:
        return np.array([[-1.0, -1.0, -0.1], [1.0, 1.0, 1.0]], dtype=np.float32)
    return np.vstack([np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)]).astype(np.float32)


def _write_png(frame: np.ndarray, path: Path):
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return
    Image.fromarray(frame).save(path)


def _make_runtime_stats(
    solver_name: str,
    frame_count: int,
    sim_seconds: float,
    step_times: list[float],
    render_times: list[float],
    force_history,
) -> RuntimeStats:
    summary = summarize_force_profile(
        np.array(force_history.times), np.array(force_history.forces), np.array(force_history.mean_damage)
    )
    wall_seconds = float(np.sum(step_times) + np.sum(render_times))
    return RuntimeStats(
        solver=solver_name,
        frame_count=frame_count,
        sim_seconds=sim_seconds,
        wall_seconds=wall_seconds,
        mean_step_ms=float(1.0e3 * np.mean(step_times)) if step_times else 0.0,
        mean_render_ms=float(1.0e3 * np.mean(render_times)) if render_times else 0.0,
        fps=float(frame_count / wall_seconds) if wall_seconds > 0.0 else 0.0,
        peak_force_n=summary["peak_force_n"],
        mean_force_n=summary["mean_force_n"],
        force_impulse_ns=summary["force_impulse_ns"],
        final_mean_damage=summary["final_mean_damage"],
        hardware=collect_hardware_details(),
    )


def _run_case(
    name: str,
    example_cls,
    argv: list[str],
    output_dir: Path,
    solver_name: str,
    title: str,
    surface_color: tuple[float, float, float],
    wall_color: tuple[float, float, float],
    frame_count: int,
    video_fps: float,
    viewer=None,
):
    owns_viewer = viewer is None
    if owns_viewer:
        viewer, args = _parse_example_args(
            example_cls,
            [*argv, "--num-frames", str(frame_count)],
            viewer_name="gl",
            headless=True,
        )
    else:
        parsed_viewer, args = _parse_example_args(
            example_cls,
            [*argv, "--num-frames", str(frame_count)],
            viewer_name="null",
            headless=False,
        )
        parsed_viewer.close()
        if hasattr(viewer, "clear_model"):
            viewer.clear_model()

    example = example_cls(viewer, args)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    frames: list[np.ndarray] = []
    step_times: list[float] = []
    render_times: list[float] = []
    try:
        for _frame in range(frame_count):
            step_start = time.perf_counter()
            example.step()
            step_times.append(time.perf_counter() - step_start)

            render_start = time.perf_counter()
            example.render()
            frame = capture_viewer_frame(viewer, render_ui=False)
            render_times.append(time.perf_counter() - render_start)
            if frame is not None:
                frames.append(frame)
    finally:
        if owns_viewer:
            viewer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    encode_mp4(frames, output_dir / f"{solver_name}_cutting.mp4", fps=video_fps)
    save_first_frame(frames, output_dir / f"{solver_name}_first_frame.png")
    if frames:
        _write_png(frames[min(44, len(frames) - 1)], output_dir / f"{solver_name}_frame_44.png")
        _write_png(frames[min(89, len(frames) - 1)], output_dir / f"{solver_name}_frame_89.png")
        _write_png(frames[-1], output_dir / f"{solver_name}_frame_{len(frames) - 1}.png")
    write_force_plot(output_dir / f"{solver_name}_force_profile.png", example.force_history, f"{title} knife cut")
    example.force_history.write_csv(output_dir / f"{solver_name}_force_profile.csv")
    write_json(output_dir / f"{solver_name}_force_history.json", example.force_history.to_dict())
    stats = _make_runtime_stats(
        solver_name,
        frame_count,
        float(example.sim_time),
        step_times,
        render_times,
        example.force_history,
    )
    (output_dir / f"{solver_name}_runtime_stats.json").write_text(stats.to_json() + "\n", encoding="utf-8")
    remesh_history = getattr(example, "remesh_history", [])
    remesh_path = export_remesh_artifacts(output_dir, solver_name, remesh_history)
    print(
        f"{name}: {frame_count} frames, {example.model.particle_count} particles, "
        f"{example.model.tet_count} tets, peak {stats.peak_force_n:.2f} N, "
        f"video {output_dir / f'{solver_name}_cutting.mp4'}"
    )
    if remesh_path is not None:
        print(f"{name}: remesh stats {remesh_path}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/horde/repos/academic-website-reports/cutting/assets/newton_baselines"),
    )
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--case",
        action="append",
        choices=["mpm", "vbd", "xfem_cuboid", "xfem_vegetable", "xfem_paper", "xfem_bread"],
        help="Generate one case; repeat for multiple. Defaults to all.",
    )
    return parser


def main():
    args = create_parser().parse_args()
    cases = set(args.case or ["mpm", "vbd", "xfem_cuboid", "xfem_vegetable", "xfem_paper", "xfem_bread"])
    root = args.output_root
    configs = {
        "mpm": (
            MPMExample,
            [],
            root / "mpm",
            "mpm",
            "MPM baseline",
            (0.18, 0.52, 0.92),
            (0.92, 0.23, 0.18),
        ),
        "vbd": (
            VBDExample,
            [],
            root / "vbd",
            "vbd",
            "Mesh VBD baseline",
            (0.95, 0.58, 0.18),
            (0.86, 0.18, 0.16),
        ),
        "xfem_cuboid": (
            XFEMExample,
            ["--scenario", "cuboid_slice"],
            root / "xfem" / "cuboid_slice",
            "xfem_cuboid_slice",
            "X-FEM cuboid slice",
            (0.97, 0.64, 0.28),
            (0.92, 0.24, 0.22),
        ),
        "xfem_vegetable": (
            XFEMExample,
            ["--scenario", "vegetable_sawing"],
            root / "xfem" / "vegetable_sawing",
            "xfem_vegetable_sawing",
            "X-FEM half-cylinder sawing",
            (0.36, 0.72, 0.31),
            (0.94, 0.78, 0.42),
        ),
        "xfem_paper": (
            XFEMExample,
            ["--scenario", "paper_tearing"],
            root / "xfem" / "paper_tearing",
            "xfem_paper_tearing",
            "X-FEM paper tearing",
            (0.94, 0.94, 0.90),
            (0.65, 0.16, 0.16),
        ),
        "xfem_bread": (
            XFEMExample,
            ["--scenario", "bread_tearing"],
            root / "xfem" / "bread_tearing",
            "xfem_bread_tearing",
            "X-FEM bread half-cylinder",
            (0.86, 0.65, 0.36),
            (0.98, 0.83, 0.55),
        ),
    }

    case_order = ["mpm", "vbd", "xfem_cuboid", "xfem_vegetable", "xfem_paper", "xfem_bread"]
    selected_cases = [case_name for case_name in case_order if case_name in cases]
    if not selected_cases:
        return

    import newton.viewer  # noqa: PLC0415

    shared_viewer = newton.viewer.ViewerGL(headless=True)
    try:
        for case_name in selected_cases:
            example_cls, argv, output_dir, solver_name, title, surface_color, wall_color = configs[case_name]
            _run_case(
                case_name,
                example_cls,
                argv,
                output_dir,
                solver_name,
                title,
                surface_color,
                wall_color,
                args.frames,
                args.video_fps,
                viewer=shared_viewer,
            )
    finally:
        shared_viewer.close()


if __name__ == "__main__":
    main()
