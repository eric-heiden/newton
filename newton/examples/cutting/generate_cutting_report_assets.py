# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate report media for the experimental cutting examples."""

from __future__ import annotations

import argparse
import csv
import json
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
    ShellCutSurfaceRenderer,
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
        front_x, center_y, center_z, _velocity = example._knife_state(time_value)
        base = getattr(example, "knife_profile", None)
        return KnifeProfile(
            start_x=front_x,
            speed=0.0,
            center_y=center_y,
            center_z=center_z,
            half_width_y=cfg.knife_half_width_y,
            half_width_z=cfg.knife_half_width_z,
            process_width=cfg.process_width,
            edge_control_points=base.edge_control_points if base is not None else (),
            blade_spine_depth=cfg.blade_spine_depth,
            cut_path_amplitude_y=cfg.cut_path_amplitude_y,
            cut_path_wavelength_x=cfg.cut_path_wavelength_x,
            cut_path_phase=cfg.cut_path_phase,
            cut_path_origin_x=cfg.cut_path_origin_x,
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
    return (
        vertices.astype(np.float32, copy=False),
        indices.astype(np.int32, copy=False),
        edge_points.astype(np.float32, copy=False),
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

    if isinstance(render_mesh, (TetMeshCutSurfaceRenderer, ShellCutSurfaceRenderer)):
        front_x = None
        center_z = None
        if hasattr(example, "_knife_state"):
            front_x, _center_y, center_z, _velocity = example._knife_state(example.sim_time)
        stats = render_mesh.update(
            example.state_0.particle_q,
            example.sim_time,
            front_x=front_x,
            center_z=center_z,
            enrichment_points=None,
            triangle_cut_state=getattr(getattr(example, "solver", None), "tri_cut_state", None),
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
    import matplotlib  # noqa: PLC0415

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


def _triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return np.zeros(0, dtype=np.float32)
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return (0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)).astype(np.float32, copy=False)


def _collect_cloth_quality_frame(example) -> dict[str, float | int | bool] | None:
    model = getattr(example, "model", None)
    if model is None or getattr(model, "tet_count", 0) != 0 or getattr(model, "tri_count", 0) == 0:
        return None
    if getattr(model, "tri_indices", None) is None:
        return None

    rest_wp = getattr(example, "render_rest_particle_q_wp", None)
    if rest_wp is None and hasattr(example, "solver"):
        rest_wp = getattr(example.solver, "rest_particle_q", None)
    if rest_wp is None:
        return None

    rest_points = rest_wp.numpy().astype(np.float32, copy=False)
    current_points = example.state_0.particle_q.numpy().astype(np.float32, copy=False)
    tri_indices = model.tri_indices.numpy().reshape(-1, 3).astype(np.int32, copy=False)
    rest_area = _triangle_areas(rest_points, tri_indices)
    current_area = _triangle_areas(current_points, tri_indices)
    area_ratio = current_area / np.maximum(rest_area, 1.0e-12)
    finite_ratio = area_ratio[np.isfinite(area_ratio)]

    frame: dict[str, float | int | bool] = {
        "time_s": float(getattr(example, "sim_time", 0.0)),
        "triangle_count": int(tri_indices.shape[0]),
        "finite_geometry": bool(np.isfinite(current_points).all() and np.isfinite(area_ratio).all()),
        "total_area_m2": float(np.sum(current_area)),
        "rest_area_m2": float(np.sum(rest_area)),
        "total_area_ratio": float(np.sum(current_area) / max(float(np.sum(rest_area)), 1.0e-12)),
        "max_triangle_area_ratio": float(np.max(finite_ratio)) if finite_ratio.size else 0.0,
        "p99_triangle_area_ratio": float(np.percentile(finite_ratio, 99.0)) if finite_ratio.size else 0.0,
        "min_triangle_area_m2": float(np.min(current_area)) if current_area.size else 0.0,
        "max_triangle_area_m2": float(np.max(current_area)) if current_area.size else 0.0,
        "released_seam_count": 0,
        "min_released_seam_gap_m": 0.0,
        "p05_released_seam_gap_m": 0.0,
        "mean_released_seam_gap_m": 0.0,
    }

    solver = getattr(example, "solver", None)
    if (
        solver is None
        or getattr(model, "spring_count", 0) == 0
        or getattr(model, "spring_indices", None) is None
        or getattr(solver, "spring_cut_state", None) is None
    ):
        return frame

    spring_indices = model.spring_indices.numpy().reshape(-1, 2).astype(np.int32, copy=False)
    spring_cut = solver.spring_cut_state.numpy().astype(np.int32, copy=False)
    rest_delta = rest_points[spring_indices[:, 0]] - rest_points[spring_indices[:, 1]]
    released = (np.linalg.norm(rest_delta, axis=1) <= 1.0e-7) & (spring_cut != 0)
    if not np.any(released):
        return frame

    seam_pairs = spring_indices[released]
    rest_mid = 0.5 * (rest_points[seam_pairs[:, 0]] + rest_points[seam_pairs[:, 1]])
    knife = getattr(example, "knife_profile", None)
    if knife is None:
        normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (seam_pairs.shape[0], 1))
    else:
        normals = np.asarray(knife.path_normal_at_x(rest_mid[:, 0]), dtype=np.float32)
    gaps = np.einsum(
        "ij,ij->i",
        current_points[seam_pairs[:, 1]] - current_points[seam_pairs[:, 0]],
        normals,
    )
    finite_gaps = gaps[np.isfinite(gaps)]
    if finite_gaps.size:
        frame["released_seam_count"] = int(finite_gaps.size)
        frame["min_released_seam_gap_m"] = float(np.min(finite_gaps))
        frame["p05_released_seam_gap_m"] = float(np.percentile(finite_gaps, 5.0))
        frame["mean_released_seam_gap_m"] = float(np.mean(finite_gaps))
    return frame


def _summarize_cloth_quality_history(
    frames: list[dict[str, float | int | bool]],
) -> dict[str, float | int | bool | None]:
    if not frames:
        return {}
    released_frames = [frame for frame in frames if int(frame.get("released_seam_count", 0)) > 0]
    return {
        "frame_count": int(len(frames)),
        "finite_geometry": bool(all(bool(frame.get("finite_geometry", False)) for frame in frames)),
        "max_total_area_ratio": float(max(float(frame["total_area_ratio"]) for frame in frames)),
        "max_triangle_area_ratio": float(max(float(frame["max_triangle_area_ratio"]) for frame in frames)),
        "max_p99_triangle_area_ratio": float(max(float(frame["p99_triangle_area_ratio"]) for frame in frames)),
        "max_released_seam_count": int(max(int(frame.get("released_seam_count", 0)) for frame in frames)),
        "min_released_seam_gap_m": (
            float(min(float(frame["min_released_seam_gap_m"]) for frame in released_frames))
            if released_frames
            else None
        ),
        "min_p05_released_seam_gap_m": (
            float(min(float(frame["p05_released_seam_gap_m"]) for frame in released_frames))
            if released_frames
            else None
        ),
    }


def _write_cloth_quality_stats(
    output_dir: Path, solver_name: str, frames: list[dict[str, float | int | bool]]
) -> Path | None:
    if not frames:
        return None
    path = output_dir / f"{solver_name}_cloth_quality_stats.json"
    write_json(path, {"summary": _summarize_cloth_quality_history(frames), "frames": frames})
    return path


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


def _write_benchmark_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "case",
        "geometry",
        "particle_count",
        "tet_count",
        "tri_count",
        "edge_count",
        "spring_count",
        "frame_count",
        "substeps",
        "iterations",
        "sim_seconds",
        "wall_seconds",
        "mean_step_ms",
        "sim_fps",
        "peak_force_n",
        "mean_force_n",
        "final_mean_damage",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_benchmark_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return

    labels = [f"{row['method']} {row['case']}" for row in rows]
    step_ms = [float(row["mean_step_ms"]) for row in rows]
    elements = [max(int(row["particle_count"]) + int(row["tet_count"]) + int(row["tri_count"]), 1) for row in rows]
    colors = [
        {"MPM": "#2563eb", "VBD": "#f97316", "X-FEM": "#16a34a"}.get(str(row["method"]), "#64748b") for row in rows
    ]

    fig, (ax_ms, ax_scale) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)
    ax_ms.bar(np.arange(len(rows)), step_ms, color=colors)
    ax_ms.set_xticks(np.arange(len(rows)), labels, rotation=45, ha="right", fontsize=7)
    ax_ms.set_ylabel("mean physics step [ms/frame]")
    ax_ms.set_title("Null-viewer physics cost")
    ax_ms.grid(axis="y", color="#d1d5db", alpha=0.65)

    for method in ("MPM", "VBD", "X-FEM"):
        xs = [elements[i] for i, row in enumerate(rows) if row["method"] == method]
        ys = [step_ms[i] for i, row in enumerate(rows) if row["method"] == method]
        if xs:
            ax_scale.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("particles + tets + triangles")
    ax_scale.set_ylabel("mean physics step [ms/frame]")
    ax_scale.set_title("Complexity trend")
    ax_scale.grid(True, which="both", color="#d1d5db", alpha=0.65)
    ax_scale.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_adaptive_remesh_plot(path: Path, root: Path) -> None:
    cases = [
        ("MPM cuboid", root / "mpm" / "mpm_adaptive_remesh_stats.json", "#2563eb"),
        ("VBD cuboid", root / "vbd" / "vbd_adaptive_remesh_stats.json", "#f97316"),
        ("X-FEM cuboid", root / "xfem" / "cuboid_slice" / "xfem_cuboid_slice_adaptive_remesh_stats.json", "#16a34a"),
        (
            "X-FEM vegetable",
            root / "xfem" / "vegetable_sawing" / "xfem_vegetable_sawing_adaptive_remesh_stats.json",
            "#84cc16",
        ),
        (
            "X-FEM paper shell",
            root / "xfem" / "paper_tearing" / "xfem_paper_tearing_adaptive_remesh_stats.json",
            "#64748b",
        ),
        (
            "X-FEM hanging cloth",
            root / "xfem" / "hanging_cloth_cutoff" / "xfem_hanging_cloth_cutoff_adaptive_remesh_stats.json",
            "#0ea5e9",
        ),
        (
            "X-FEM curved cloth",
            root / "xfem" / "curved_cloth_spline_cut" / "xfem_curved_cloth_spline_cut_adaptive_remesh_stats.json",
            "#7c3aed",
        ),
        (
            "X-FEM bread",
            root / "xfem" / "bread_tearing" / "xfem_bread_tearing_adaptive_remesh_stats.json",
            "#a16207",
        ),
    ]
    histories: list[tuple[str, list[dict[str, float]], str]] = []
    for label, stats_path, color in cases:
        if not stats_path.exists():
            continue
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        frames = data.get("frames", [])
        if frames:
            histories.append((label, frames, color))
    if not histories:
        return

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return

    fig, (ax_active, ax_tris) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)
    for label, frames, color in histories:
        times = np.asarray([float(frame.get("time_s", i)) for i, frame in enumerate(frames)], dtype=np.float32)
        active = np.asarray([float(frame.get("active_x_segments", 0.0)) for frame in frames], dtype=np.float32)
        surface = np.asarray([float(frame.get("surface_triangle_count", 0.0)) for frame in frames], dtype=np.float32)
        walls = np.asarray([float(frame.get("wall_triangle_count", 0.0)) for frame in frames], dtype=np.float32)
        ax_active.plot(times, active, color=color, linewidth=1.7, label=label)
        ax_tris.plot(times, surface, color=color, linewidth=1.4, label=f"{label} surface")
        ax_tris.plot(times, walls, color=color, linewidth=1.2, linestyle="--", alpha=0.78)

    ax_active.set_title("Active remesh/cut support")
    ax_active.set_xlabel("time [s]")
    ax_active.set_ylabel("active x segments, cut tets, or cut triangles")
    ax_active.grid(True, color="#d1d5db", alpha=0.65)
    ax_active.legend(fontsize=7)

    ax_tris.set_title("Generated render triangles")
    ax_tris.set_xlabel("time [s]")
    ax_tris.set_ylabel("triangle count")
    ax_tris.grid(True, color="#d1d5db", alpha=0.65)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _run_benchmark_case(
    method: str,
    case: str,
    example_cls,
    argv: list[str],
    frame_count: int,
) -> dict[str, object]:
    viewer, args = _parse_example_args(
        example_cls,
        [*argv, "--num-frames", str(frame_count), "--no-render-split-mesh"],
        viewer_name="null",
        headless=False,
    )
    try:
        example = example_cls(viewer, args)
        step_times: list[float] = []
        for _frame in range(frame_count):
            start = time.perf_counter()
            example.step()
            step_times.append(time.perf_counter() - start)
        wall_seconds = float(np.sum(step_times))
        summary = summarize_force_profile(
            np.array(example.force_history.times),
            np.array(example.force_history.forces),
            np.array(example.force_history.mean_damage),
        )
        row = {
            "method": method,
            "case": case,
            "geometry": getattr(getattr(example, "scenario", None), "geometry", "cuboid"),
            "particle_count": int(example.model.particle_count),
            "tet_count": int(example.model.tet_count),
            "tri_count": int(example.model.tri_count),
            "edge_count": int(example.model.edge_count),
            "spring_count": int(example.model.spring_count),
            "frame_count": int(frame_count),
            "substeps": int(getattr(example, "sim_substeps", 1)),
            "iterations": int(
                getattr(example, "iterations", getattr(getattr(example, "solver", None), "iterations", 0))
            ),
            "sim_seconds": float(example.sim_time),
            "wall_seconds": wall_seconds,
            "mean_step_ms": float(1.0e3 * np.mean(step_times)) if step_times else 0.0,
            "sim_fps": float(frame_count / wall_seconds) if wall_seconds > 0.0 else 0.0,
            "peak_force_n": summary["peak_force_n"],
            "mean_force_n": summary["mean_force_n"],
            "final_mean_damage": summary["final_mean_damage"],
        }
    finally:
        viewer.close()
    print(
        f"benchmark {method} {case}: {row['particle_count']} particles, {row['tet_count']} tets, "
        f"{row['tri_count']} tris, {row['mean_step_ms']:.2f} ms/frame"
    )
    return row


def _run_benchmark_sweep(root: Path, frame_count: int) -> list[dict[str, object]]:
    cases = [
        ("MPM", "coarse", MPMExample, ["--particles-per-cell", "1", "--voxel-size", "0.090"]),
        ("MPM", "default", MPMExample, ["--particles-per-cell", "2", "--voxel-size", "0.075"]),
        ("MPM", "fine", MPMExample, ["--particles-per-cell", "3", "--voxel-size", "0.070"]),
        (
            "VBD",
            "coarse",
            VBDExample,
            ["--dim-x", "10", "--dim-y", "5", "--dim-z", "4", "--iterations", "6"],
        ),
        (
            "VBD",
            "default",
            VBDExample,
            ["--dim-x", "14", "--dim-y", "7", "--dim-z", "6", "--iterations", "8"],
        ),
        (
            "VBD",
            "fine",
            VBDExample,
            ["--dim-x", "18", "--dim-y", "9", "--dim-z", "7", "--iterations", "8"],
        ),
        ("X-FEM", "cuboid", XFEMExample, ["--scenario", "cuboid_slice", "--iterations", "10"]),
        ("X-FEM", "paper shell", XFEMExample, ["--scenario", "paper_tearing", "--iterations", "10"]),
        ("X-FEM", "hanging cloth", XFEMExample, ["--scenario", "hanging_cloth_cutoff", "--iterations", "10"]),
        ("X-FEM", "curved cloth", XFEMExample, ["--scenario", "curved_cloth_spline_cut", "--iterations", "10"]),
        ("X-FEM", "bread half-cylinder", XFEMExample, ["--scenario", "bread_tearing", "--iterations", "10"]),
        ("X-FEM", "vegetable half-cylinder", XFEMExample, ["--scenario", "vegetable_sawing", "--iterations", "10"]),
    ]
    rows = [_run_benchmark_case(method, case, cls, argv, frame_count) for method, case, cls, argv in cases]
    path_json = root / "benchmark_results.json"
    write_json(
        path_json,
        {
            "hardware": collect_hardware_details(),
            "frame_count": frame_count,
            "rows": rows,
        },
    )
    _write_benchmark_csv(root / "benchmark_results.csv", rows)
    _write_benchmark_plot(root / "benchmark_results.png", rows)
    return rows


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
    cloth_quality_frames: list[dict[str, float | int | bool]] = []
    step_times: list[float] = []
    render_times: list[float] = []
    try:
        for _frame in range(frame_count):
            step_start = time.perf_counter()
            example.step()
            step_times.append(time.perf_counter() - step_start)
            cloth_quality_frame = _collect_cloth_quality_frame(example)
            if cloth_quality_frame is not None:
                cloth_quality_frames.append(cloth_quality_frame)

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
    cloth_quality_path = _write_cloth_quality_stats(output_dir, solver_name, cloth_quality_frames)
    print(
        f"{name}: {frame_count} frames, {example.model.particle_count} particles, "
        f"{example.model.tet_count} tets, peak {stats.peak_force_n:.2f} N, "
        f"video {output_dir / f'{solver_name}_cutting.mp4'}"
    )
    if remesh_path is not None:
        print(f"{name}: remesh stats {remesh_path}")
    if cloth_quality_path is not None:
        print(f"{name}: cloth quality stats {cloth_quality_path}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/horde/repos/academic-website-reports/cutting/assets/newton_baselines"),
    )
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument(
        "--benchmark-sweep",
        action="store_true",
        help="Run null-viewer physics benchmarks over several geometry complexities and write benchmark_results.json.",
    )
    parser.add_argument("--benchmark-frames", type=int, default=120)
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "mpm",
            "vbd",
            "xfem_cuboid",
            "xfem_vegetable",
            "xfem_paper",
            "xfem_hanging",
            "xfem_curved",
            "xfem_bread",
        ],
        help="Generate one case; repeat for multiple. Defaults to all.",
    )
    return parser


def main():
    args = create_parser().parse_args()
    cases = set(
        args.case
        or [
            "mpm",
            "vbd",
            "xfem_cuboid",
            "xfem_vegetable",
            "xfem_paper",
            "xfem_hanging",
            "xfem_curved",
            "xfem_bread",
        ]
    )
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
        "xfem_hanging": (
            XFEMExample,
            ["--scenario", "hanging_cloth_cutoff"],
            root / "xfem" / "hanging_cloth_cutoff",
            "xfem_hanging_cloth_cutoff",
            "X-FEM hanging cloth cut-off",
            (0.82, 0.88, 0.96),
            (0.50, 0.14, 0.14),
        ),
        "xfem_curved": (
            XFEMExample,
            ["--scenario", "curved_cloth_spline_cut"],
            root / "xfem" / "curved_cloth_spline_cut",
            "xfem_curved_cloth_spline_cut",
            "X-FEM curved cloth spline cut",
            (0.89, 0.91, 0.84),
            (0.62, 0.12, 0.14),
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

    case_order = [
        "mpm",
        "vbd",
        "xfem_cuboid",
        "xfem_vegetable",
        "xfem_paper",
        "xfem_hanging",
        "xfem_curved",
        "xfem_bread",
    ]
    selected_cases = [case_name for case_name in case_order if case_name in cases]
    if not selected_cases and not args.benchmark_sweep:
        return

    if args.benchmark_sweep:
        _run_benchmark_sweep(root, args.benchmark_frames)

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
    _write_adaptive_remesh_plot(root / "adaptive_remesh_profile.png", root)


if __name__ == "__main__":
    main()
