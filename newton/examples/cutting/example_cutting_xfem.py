# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Track C: X-FEM-style soft-object cutting scenarios."""

import argparse
import math
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.xfem_cut.kernels import apply_xfem_cloth_wind_kernel
from newton.examples.cutting.cutting_common import (
    AdaptiveCutSurfaceRemesher,
    ForceHistory,
    KnifeProfile,
    ShellCutSurfaceRenderer,
    SplitCuboidRenderMesh,
    TetMeshCutSurfaceRenderer,
    add_cutting_artifact_args,
    log_knife_mesh,
    run_cutting_example,
)


def _orient_positive_tets(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a = vertices[tets[:, 0]]
    b = vertices[tets[:, 1]]
    c = vertices[tets[:, 2]]
    d = vertices[tets[:, 3]]
    signed_volumes = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    flipped = signed_volumes < 0.0
    if np.any(flipped):
        tets = tets.copy()
        tets[flipped, 2], tets[flipped, 3] = tets[flipped, 3], tets[flipped, 2].copy()
    return tets


def build_half_cylinder_tet_mesh(length: float, radius: float, target_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Build an extruded half-cylinder tetrahedral mesh with a flat table face."""

    try:
        import gmsh  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("gmsh is required for half-cylinder cutting scenarios; install the examples extra") from exc

    length = float(length)
    radius = float(radius)
    target_edge = float(target_edge)
    if length <= 0.0 or radius <= 0.0 or target_edge <= 0.0:
        raise ValueError("length, radius, and target_edge must be positive")

    initialized = bool(gmsh.isInitialized()) if hasattr(gmsh, "isInitialized") else False
    if not initialized:
        gmsh.initialize(["-"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("newton_half_cylinder")

        p_left = gmsh.model.geo.addPoint(0.0, -radius, 0.0, target_edge)
        p_top = gmsh.model.geo.addPoint(0.0, 0.0, radius, target_edge)
        p_right = gmsh.model.geo.addPoint(0.0, radius, 0.0, target_edge)
        p_center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, target_edge)
        arc_left = gmsh.model.geo.addCircleArc(p_left, p_center, p_top)
        arc_right = gmsh.model.geo.addCircleArc(p_top, p_center, p_right)
        base = gmsh.model.geo.addLine(p_right, p_left)
        loop = gmsh.model.geo.addCurveLoop([arc_left, arc_right, base])
        section = gmsh.model.geo.addPlaneSurface([loop])
        layers = max(2, int(round(length / target_edge)))
        gmsh.model.geo.extrude([(2, section)], length, 0.0, 0.0, numElements=[layers], recombine=False)
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.55 * target_edge)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_edge)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)

        node_tags, coords, _params = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        vertices[np.abs(vertices) < 1.0e-12] = 0.0
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
        element_types, _element_tags, element_nodes = gmsh.model.mesh.getElements(3)
        tet_blocks = []
        for element_type, nodes in zip(element_types, element_nodes, strict=True):
            name, dim, order, node_count, *_ = gmsh.model.mesh.getElementProperties(element_type)
            if name.startswith("Tetrahedron") and dim == 3 and order == 1 and node_count == 4:
                tet_blocks.append(
                    np.asarray([tag_to_index[int(node)] for node in nodes], dtype=np.int32).reshape(-1, 4)
                )

        if not tet_blocks:
            raise RuntimeError("gmsh did not generate first-order tetrahedra for the half-cylinder mesh")
        tets = _orient_positive_tets(vertices, np.vstack(tet_blocks).astype(np.int32, copy=False))
        used_vertices = np.unique(tets.ravel())
        remap = np.full(vertices.shape[0], -1, dtype=np.int32)
        remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int32)
        vertices = vertices[used_vertices]
        tets = remap[tets]
        return vertices.astype(np.float32), tets
    finally:
        if not initialized:
            gmsh.finalize()


@dataclass(frozen=True)
class XFEMScenario:
    name: str
    block_pos: tuple[float, float, float]
    dim_x: int
    dim_y: int
    dim_z: int
    cell_x: float
    cell_y: float
    cell_z: float
    density: float
    k_mu: float
    k_lambda: float
    k_damp: float
    gravity: tuple[float, float, float]
    fix_left: bool
    knife_start_x: float
    knife_speed: float
    knife_center_y: float
    knife_center_z: float
    knife_half_width_y: float
    knife_half_width_z: float
    blade_spine_depth: float
    process_width: float
    saw_amplitude_z: float
    saw_frequency_hz: float
    fracture_energy: float
    yield_stress: float
    max_damage_rate: float
    separation_speed: float
    force_scale: float
    friction_mu: float
    table_z: float
    table_glue_depth: float
    table_glue_strength: float
    residual_stiffness: float
    damage_threshold: float
    max_visual_gap: float
    surface_color: tuple[float, float, float]
    wall_color: tuple[float, float, float]
    particle_color_scale: float
    camera_pos: tuple[float, float, float]
    camera_pitch: float
    camera_yaw: float
    camera_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fix_right: bool = False
    fix_top: bool = False
    fix_bottom: bool = False
    wind_strength: float = 0.0
    wind_frequency_hz: float = 0.0
    wind_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    up_axis: newton.Axis = newton.Axis.Z
    geometry: str = "grid"
    tet_target_edge: float = 0.052
    cut_path_amplitude_y: float = 0.0
    cut_path_wavelength_x: float = 1.0
    cut_path_phase: float = 0.0
    cut_path_origin_x: float = 0.0
    render_mesh_edges: bool = False
    render_seam_edges: bool = False
    visual_topology_only: bool = False
    render_cut_refine_factor: int = 1

    @property
    def block_size(self) -> np.ndarray:
        if self.geometry == "half_cylinder":
            radius = 0.5 * self.dim_y * self.cell_y
            return np.array([self.dim_x * self.cell_x, 2.0 * radius, radius], dtype=np.float32)
        if self.geometry == "cloth_grid":
            return np.array([self.dim_x * self.cell_x, self.dim_y * self.cell_y, 0.0], dtype=np.float32)
        return np.array(
            [self.dim_x * self.cell_x, self.dim_y * self.cell_y, self.dim_z * self.cell_z],
            dtype=np.float32,
        )

    @property
    def half_cylinder_radius(self) -> float:
        return 0.5 * self.dim_y * self.cell_y


SCENARIOS: dict[str, XFEMScenario] = {
    "cuboid_slice": XFEMScenario(
        name="cuboid_slice",
        block_pos=(-0.45, -0.20, 0.045),
        dim_x=14,
        dim_y=6,
        dim_z=5,
        cell_x=0.064,
        cell_y=0.064,
        cell_z=0.058,
        density=950.0,
        k_mu=5.2e4,
        k_lambda=9.0e4,
        k_damp=1.0e-3,
        gravity=(0.0, 0.0, -9.81),
        fix_left=False,
        knife_start_x=-0.52,
        knife_speed=0.72,
        knife_center_y=0.0,
        knife_center_z=0.22,
        knife_half_width_y=0.065,
        knife_half_width_z=0.24,
        blade_spine_depth=0.18,
        process_width=0.06,
        saw_amplitude_z=0.0,
        saw_frequency_hz=0.0,
        fracture_energy=105.0,
        yield_stress=1.9e4,
        max_damage_rate=13.0,
        separation_speed=0.25,
        force_scale=0.42,
        friction_mu=0.82,
        table_z=0.0,
        table_glue_depth=0.0,
        table_glue_strength=0.0,
        residual_stiffness=0.08,
        damage_threshold=0.2,
        max_visual_gap=0.055,
        surface_color=(0.97, 0.64, 0.28),
        wall_color=(0.92, 0.24, 0.22),
        particle_color_scale=0.7,
        camera_pos=(1.15, -1.35, 0.82),
        camera_pitch=-20.0,
        camera_yaw=130.0,
    ),
    "vegetable_sawing": XFEMScenario(
        name="vegetable_sawing",
        block_pos=(-0.43, -0.13, 0.018),
        dim_x=18,
        dim_y=5,
        dim_z=4,
        cell_x=0.047,
        cell_y=0.052,
        cell_z=0.055,
        density=760.0,
        k_mu=5.0e3,
        k_lambda=9.0e3,
        k_damp=2.0e-3,
        gravity=(0.0, 0.0, -9.81),
        fix_left=False,
        knife_start_x=-0.48,
        knife_speed=0.45,
        knife_center_y=0.0,
        knife_center_z=0.14,
        knife_half_width_y=0.06,
        knife_half_width_z=0.18,
        blade_spine_depth=0.18,
        process_width=0.055,
        saw_amplitude_z=0.065,
        saw_frequency_hz=2.4,
        fracture_energy=75.0,
        yield_stress=1.15e4,
        max_damage_rate=18.0,
        separation_speed=0.18,
        force_scale=0.46,
        friction_mu=1.35,
        table_z=0.0,
        table_glue_depth=0.024,
        table_glue_strength=0.55,
        residual_stiffness=0.05,
        damage_threshold=0.18,
        max_visual_gap=0.062,
        surface_color=(0.36, 0.72, 0.31),
        wall_color=(0.96, 0.38, 0.16),
        particle_color_scale=0.42,
        camera_pos=(1.05, -1.12, 0.74),
        camera_pitch=-27.0,
        camera_yaw=128.0,
        geometry="half_cylinder",
        tet_target_edge=0.033,
    ),
    "paper_tearing": XFEMScenario(
        name="paper_tearing",
        block_pos=(-0.56, -0.35, 0.026),
        dim_x=32,
        dim_y=20,
        dim_z=0,
        cell_x=0.035,
        cell_y=0.035,
        cell_z=0.0,
        density=0.045,
        k_mu=2.8e2,
        k_lambda=2.8e2,
        k_damp=3.5e-2,
        gravity=(0.0, 0.0, -1.2),
        fix_left=True,
        knife_start_x=-0.58,
        knife_speed=0.46,
        knife_center_y=0.0175,
        knife_center_z=0.035,
        knife_half_width_y=0.035,
        knife_half_width_z=0.090,
        blade_spine_depth=0.18,
        process_width=0.040,
        saw_amplitude_z=0.0,
        saw_frequency_hz=0.0,
        fracture_energy=24.0,
        yield_stress=1.6e3,
        max_damage_rate=20.0,
        separation_speed=0.075,
        force_scale=0.16,
        friction_mu=0.78,
        table_z=0.0,
        table_glue_depth=0.0,
        table_glue_strength=0.0,
        residual_stiffness=0.025,
        damage_threshold=0.14,
        max_visual_gap=0.034,
        surface_color=(0.94, 0.94, 0.90),
        wall_color=(0.65, 0.16, 0.16),
        particle_color_scale=0.18,
        camera_pos=(0.82, -1.12, 0.42),
        camera_pitch=-34.0,
        camera_yaw=126.0,
        geometry="cloth_grid",
        render_cut_refine_factor=2,
    ),
    "hanging_cloth_cutoff": XFEMScenario(
        name="hanging_cloth_cutoff",
        block_pos=(-0.56, -0.42, 0.02),
        dim_x=36,
        dim_y=28,
        dim_z=0,
        cell_x=0.030,
        cell_y=0.030,
        cell_z=0.0,
        density=0.035,
        k_mu=1.9e2,
        k_lambda=1.9e2,
        k_damp=2.0e-2,
        gravity=(0.0, -0.15, 0.0),
        fix_left=False,
        knife_start_x=-0.62,
        knife_speed=0.42,
        knife_center_y=-0.105,
        knife_center_z=0.02,
        knife_half_width_y=0.030,
        knife_half_width_z=0.075,
        blade_spine_depth=0.18,
        process_width=0.035,
        saw_amplitude_z=0.0,
        saw_frequency_hz=0.0,
        fracture_energy=20.0,
        yield_stress=1.4e3,
        max_damage_rate=24.0,
        separation_speed=0.0,
        force_scale=0.020,
        friction_mu=0.15,
        table_z=0.0,
        table_glue_depth=0.0,
        table_glue_strength=0.0,
        residual_stiffness=0.020,
        damage_threshold=0.12,
        max_visual_gap=0.020,
        surface_color=(0.82, 0.88, 0.96),
        wall_color=(0.50, 0.14, 0.14),
        particle_color_scale=0.14,
        camera_pos=(0.28, -0.36, 4.15),
        camera_pitch=0.0,
        camera_yaw=-90.0,
        camera_target_offset=(0.20, -0.48, 0.0),
        fix_top=True,
        wind_strength=0.00075,
        wind_frequency_hz=0.55,
        wind_direction=(1.0, 0.0, 0.0),
        up_axis=newton.Axis.Y,
        geometry="cloth_grid",
        render_seam_edges=False,
        render_cut_refine_factor=2,
    ),
    "curved_cloth_spline_cut": XFEMScenario(
        name="curved_cloth_spline_cut",
        block_pos=(-0.72, -0.48, 0.030),
        dim_x=54,
        dim_y=34,
        dim_z=0,
        cell_x=0.028,
        cell_y=0.028,
        cell_z=0.0,
        density=0.040,
        k_mu=1.8e2,
        k_lambda=1.8e2,
        k_damp=8.0e-2,
        gravity=(0.0, 0.0, -0.08),
        fix_left=True,
        knife_start_x=-0.75,
        knife_speed=0.31,
        knife_center_y=-0.01,
        knife_center_z=0.038,
        knife_half_width_y=0.020,
        knife_half_width_z=0.055,
        blade_spine_depth=0.075,
        process_width=0.026,
        saw_amplitude_z=0.0,
        saw_frequency_hz=0.0,
        fracture_energy=22.0,
        yield_stress=1.45e3,
        max_damage_rate=22.0,
        separation_speed=0.0,
        force_scale=0.002,
        friction_mu=0.03,
        table_z=0.0,
        table_glue_depth=0.0,
        table_glue_strength=0.0,
        residual_stiffness=0.024,
        damage_threshold=0.13,
        max_visual_gap=0.012,
        surface_color=(0.89, 0.91, 0.84),
        wall_color=(0.62, 0.12, 0.14),
        particle_color_scale=0.10,
        camera_pos=(0.82, -1.22, 0.58),
        camera_pitch=-31.0,
        camera_yaw=124.0,
        geometry="cloth_grid",
        fix_right=True,
        fix_top=True,
        fix_bottom=True,
        cut_path_amplitude_y=0.105,
        cut_path_wavelength_x=0.54,
        cut_path_phase=0.40,
        cut_path_origin_x=-0.72,
        render_mesh_edges=False,
        render_seam_edges=False,
        visual_topology_only=False,
        render_cut_refine_factor=3,
    ),
    "bread_tearing": XFEMScenario(
        name="bread_tearing",
        block_pos=(-0.40, -0.16, 0.03),
        dim_x=14,
        dim_y=5,
        dim_z=4,
        cell_x=0.055,
        cell_y=0.058,
        cell_z=0.047,
        density=410.0,
        k_mu=6.0e3,
        k_lambda=1.0e4,
        k_damp=3.0e-3,
        gravity=(0.0, 0.0, -9.81),
        fix_left=False,
        knife_start_x=-0.44,
        knife_speed=0.42,
        knife_center_y=0.0,
        knife_center_z=0.12,
        knife_half_width_y=0.075,
        knife_half_width_z=0.17,
        blade_spine_depth=0.18,
        process_width=0.07,
        saw_amplitude_z=0.05,
        saw_frequency_hz=1.8,
        fracture_energy=55.0,
        yield_stress=6.6e3,
        max_damage_rate=17.0,
        separation_speed=0.12,
        force_scale=0.22,
        friction_mu=1.25,
        table_z=0.0,
        table_glue_depth=0.065,
        table_glue_strength=0.85,
        residual_stiffness=0.04,
        damage_threshold=0.16,
        max_visual_gap=0.074,
        surface_color=(0.86, 0.65, 0.36),
        wall_color=(0.96, 0.46, 0.20),
        particle_color_scale=0.42,
        camera_pos=(0.95, -1.05, 0.68),
        camera_pitch=-27.0,
        camera_yaw=128.0,
        geometry="half_cylinder",
        tet_target_edge=0.038,
    ),
}


def _scenario_cut_center_y(cfg: XFEMScenario, x: float) -> float:
    if abs(cfg.cut_path_amplitude_y) <= 1.0e-12:
        return float(cfg.knife_center_y)
    wavelength = max(abs(cfg.cut_path_wavelength_x), 1.0e-6)
    phase = 2.0 * math.pi * (float(x) - cfg.cut_path_origin_x) / wavelength + cfg.cut_path_phase
    return float(cfg.knife_center_y + cfg.cut_path_amplitude_y * math.sin(phase))


def _scenario_cut_signed_y(cfg: XFEMScenario, point: np.ndarray) -> float:
    x = float(point[0])
    center_y = _scenario_cut_center_y(cfg, x)
    if abs(cfg.cut_path_amplitude_y) <= 1.0e-12:
        return float(point[1] - center_y)
    wavelength = max(abs(cfg.cut_path_wavelength_x), 1.0e-6)
    phase = 2.0 * math.pi * (x - cfg.cut_path_origin_x) / wavelength + cfg.cut_path_phase
    slope = cfg.cut_path_amplitude_y * (2.0 * math.pi / wavelength) * math.cos(phase)
    return float((point[1] - center_y) / math.sqrt(1.0 + slope * slope))


def _build_virtual_node_cloth_grid(
    cfg: XFEMScenario,
) -> tuple[list[wp.vec3], list[int], list[tuple[int, int]]]:
    """Build a cloth grid pre-split along the scenario cut path."""

    base_vertices: list[np.ndarray] = []

    def grid_index(x: int, y: int) -> int:
        return y * (cfg.dim_x + 1) + x

    for y in range(cfg.dim_y + 1):
        for x in range(cfg.dim_x + 1):
            base_vertices.append(
                np.asarray(
                    (
                        cfg.block_pos[0] + x * cfg.cell_x,
                        cfg.block_pos[1] + y * cfg.cell_y,
                        cfg.block_pos[2],
                    ),
                    dtype=np.float32,
                )
            )

    base_triangles: list[tuple[int, int, int]] = []
    for y in range(1, cfg.dim_y + 1):
        for x in range(1, cfg.dim_x + 1):
            v0 = grid_index(x - 1, y - 1)
            v1 = grid_index(x, y - 1)
            v2 = grid_index(x, y)
            v3 = grid_index(x - 1, y)
            base_triangles.append((v0, v1, v3))
            base_triangles.append((v1, v2, v3))

    signed = np.asarray([_scenario_cut_signed_y(cfg, point) for point in base_vertices], dtype=np.float64)
    snap_band = 0.10 * min(cfg.cell_x, cfg.cell_y)
    signed[np.abs(signed) <= snap_band] = 0.0
    vertices: list[np.ndarray] = []
    indices: list[int] = []
    seam_pairs: list[tuple[int, int]] = []
    original_vertices: dict[tuple[int, int], int] = {}
    intersections: dict[tuple[int, int], tuple[int, int]] = {}
    eps = 1.0e-8

    def append_vertex(point: np.ndarray) -> int:
        vertices.append(np.asarray(point, dtype=np.float32))
        return len(vertices) - 1

    def side_key(side: int) -> int:
        return 1 if side >= 0 else -1

    def original_vertex(vertex_id: int, side: int) -> int:
        key = (vertex_id, side_key(side))
        if key not in original_vertices:
            original_vertices[key] = append_vertex(base_vertices[vertex_id])
            if abs(float(signed[vertex_id])) <= eps:
                other_key = (vertex_id, -side_key(side))
                if other_key in original_vertices:
                    neg = original_vertices[(vertex_id, -1)]
                    pos = original_vertices[(vertex_id, 1)]
                    if neg != pos and (neg, pos) not in seam_pairs and (pos, neg) not in seam_pairs:
                        seam_pairs.append((neg, pos))
        return original_vertices[key]

    def intersection_vertices(a: int, b: int) -> tuple[int, int]:
        key = (min(a, b), max(a, b))
        if key in intersections:
            return intersections[key]
        sa = float(signed[a])
        sb = float(signed[b])
        denom = sa - sb
        alpha = 0.5 if abs(denom) <= eps else float(np.clip(sa / denom, 0.0, 1.0))
        point = base_vertices[a] + alpha * (base_vertices[b] - base_vertices[a])
        neg = append_vertex(point)
        pos = append_vertex(point)
        intersections[key] = (neg, pos)
        seam_pairs.append((neg, pos))
        return intersections[key]

    def vertex_for_side(vertex_id: int, side: int) -> int:
        if abs(float(signed[vertex_id])) <= eps:
            return original_vertex(vertex_id, side)
        return original_vertex(vertex_id, 1 if signed[vertex_id] > 0.0 else -1)

    def polygon_for_side(triangle: tuple[int, int, int], side: int) -> list[int]:
        keep_positive = side > 0
        polygon: list[int] = []
        for edge in range(3):
            current = triangle[edge]
            nxt = triangle[(edge + 1) % 3]
            s_current = float(signed[current])
            s_next = float(signed[nxt])
            current_inside = s_current >= -eps if keep_positive else s_current <= eps
            next_inside = s_next >= -eps if keep_positive else s_next <= eps
            if current_inside:
                polygon.append(vertex_for_side(current, side))
            if current_inside != next_inside:
                neg, pos = intersection_vertices(current, nxt)
                polygon.append(pos if keep_positive else neg)
        deduped: list[int] = []
        for vertex in polygon:
            if not deduped or vertex != deduped[-1]:
                deduped.append(vertex)
        if len(deduped) >= 2 and deduped[0] == deduped[-1]:
            deduped.pop()
        return deduped

    def append_polygon(polygon: list[int]) -> None:
        if len(polygon) < 3:
            return
        for i in range(1, len(polygon) - 1):
            tri = (polygon[0], polygon[i], polygon[i + 1])
            a, b, c = (vertices[index] for index in tri)
            if float(np.linalg.norm(np.cross(b - a, c - a))) <= 1.0e-12:
                continue
            indices.extend(tri)

    for triangle in base_triangles:
        tri_signed = signed[list(triangle)]
        if np.all(tri_signed >= -eps):
            append_polygon([original_vertex(vertex, 1) for vertex in triangle])
        elif np.all(tri_signed <= eps):
            append_polygon([original_vertex(vertex, -1) for vertex in triangle])
        else:
            append_polygon(polygon_for_side(triangle, -1))
            append_polygon(polygon_for_side(triangle, 1))

    return [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices], indices, seam_pairs


class Example:
    """X-FEM cut-cell/enrichment solver scenarios."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.scenario = SCENARIOS[args.scenario]
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.force_history = ForceHistory()
        self.remesh_history: list[dict[str, float]] = []

        cfg = self.scenario
        self.block_pos = np.array(cfg.block_pos, dtype=np.float32)
        self.block_size = cfg.block_size
        self.block_hi = self.block_pos + self.block_size

        builder = newton.ModelBuilder(up_axis=cfg.up_axis)
        if cfg.name != "hanging_cloth_cutoff":
            builder.add_ground_plane()
        self.generated_vertices: np.ndarray | None = None
        self.generated_tets: np.ndarray | None = None
        if cfg.geometry == "half_cylinder":
            radius = cfg.half_cylinder_radius
            self.generated_vertices, self.generated_tets = build_half_cylinder_tet_mesh(
                length=float(self.block_size[0]),
                radius=radius,
                target_edge=cfg.tet_target_edge,
            )
            mesh_pos = (float(cfg.block_pos[0]), float(cfg.block_pos[1] + radius), float(cfg.block_pos[2]))
            builder.add_soft_mesh(
                pos=wp.vec3(*mesh_pos),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=self.generated_vertices.tolist(),
                indices=self.generated_tets.reshape(-1).tolist(),
                density=cfg.density,
                k_mu=cfg.k_mu,
                k_lambda=cfg.k_lambda,
                k_damp=cfg.k_damp,
                particle_radius=args.particle_radius,
                label=f"xfem_{cfg.name}",
            )
        elif cfg.geometry == "grid":
            builder.add_soft_grid(
                pos=wp.vec3(*cfg.block_pos),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=cfg.dim_x,
                dim_y=cfg.dim_y,
                dim_z=cfg.dim_z,
                cell_x=cfg.cell_x,
                cell_y=cfg.cell_y,
                cell_z=cfg.cell_z,
                density=cfg.density,
                k_mu=cfg.k_mu,
                k_lambda=cfg.k_lambda,
                k_damp=cfg.k_damp,
                fix_left=cfg.fix_left,
                particle_radius=args.particle_radius,
            )
        elif cfg.geometry == "cloth_grid":
            start_particle = len(builder.particle_q)
            vertices, indices, seam_pairs = _build_virtual_node_cloth_grid(cfg)
            total_area = cfg.cell_x * cfg.cell_y * cfg.dim_x * cfg.dim_y
            total_mass = max(cfg.density, 1.0e-5) * (cfg.dim_x + 1) * (cfg.dim_y + 1)
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=vertices,
                indices=indices,
                density=total_mass / max(total_area, 1.0e-8),
                tri_ke=cfg.k_mu,
                tri_ka=cfg.k_lambda,
                tri_kd=cfg.k_damp,
                edge_ke=max(0.015 * cfg.k_mu, 1.0),
                edge_kd=cfg.k_damp,
                add_springs=False,
                particle_radius=args.particle_radius,
                label=f"xfem_{cfg.name}",
            )
            cohesive_ke = max(18.0 * cfg.k_mu, 2.5e3)
            cohesive_kd = max(6.0 * cfg.k_damp, 2.0e-2)
            for a, b in seam_pairs:
                builder.add_spring(start_particle + a, start_particle + b, cohesive_ke, cohesive_kd, control=0.0)

            x0 = float(cfg.block_pos[0])
            y0 = float(cfg.block_pos[1])
            x1 = x0 + cfg.dim_x * cfg.cell_x
            y1 = y0 + cfg.dim_y * cfg.cell_y
            for vertex_id, point in enumerate(vertices):
                idx = start_particle + vertex_id
                px = float(point[0])
                py = float(point[1])
                fixed = (
                    (cfg.fix_left and abs(px - x0) <= 1.0e-6)
                    or (cfg.fix_right and abs(px - x1) <= 1.0e-6)
                    or (cfg.fix_bottom and abs(py - y0) <= 1.0e-6)
                    or (cfg.fix_top and abs(py - y1) <= 1.0e-6)
                )
                if fixed:
                    builder.particle_flags[idx] = builder.particle_flags[idx] & ~newton.ParticleFlags.ACTIVE
                    builder.particle_mass[idx] = 0.0
        else:
            raise ValueError(f"Unsupported X-FEM geometry: {cfg.geometry}")
        builder.color()

        self.model = builder.finalize()
        self.model.set_gravity(cfg.gravity)
        self.model.soft_contact_ke = args.soft_contact_ke
        self.model.soft_contact_kd = args.soft_contact_kd
        self.model.soft_contact_mu = args.soft_contact_mu

        self.solver = newton.solvers.SolverXFEMCut(
            self.model,
            iterations=args.iterations,
            fracture_energy=cfg.fracture_energy,
            yield_stress=cfg.yield_stress,
            max_damage_rate=cfg.max_damage_rate,
            separation_speed=cfg.separation_speed,
            force_scale=cfg.force_scale,
            knife_friction_mu=cfg.friction_mu,
            residual_stiffness=cfg.residual_stiffness,
            damage_threshold=cfg.damage_threshold,
            max_visual_gap=cfg.max_visual_gap,
            table_z=cfg.table_z,
            table_glue_depth=cfg.table_glue_depth,
            table_glue_strength=cfg.table_glue_strength,
            table_friction=args.table_friction,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.render_rest_particle_q_wp = wp.clone(self.state_0.particle_q)

        self.knife_profile = KnifeProfile(
            start_x=cfg.knife_start_x,
            speed=cfg.knife_speed,
            center_y=cfg.knife_center_y,
            center_z=cfg.knife_center_z,
            half_width_y=cfg.knife_half_width_y,
            half_width_z=cfg.knife_half_width_z,
            process_width=cfg.process_width,
            blade_spine_depth=cfg.blade_spine_depth,
            cut_path_amplitude_y=cfg.cut_path_amplitude_y,
            cut_path_wavelength_x=cfg.cut_path_wavelength_x,
            cut_path_phase=cfg.cut_path_phase,
            cut_path_origin_x=cfg.cut_path_origin_x,
        )
        if args.render_split_mesh and cfg.geometry == "half_cylinder":
            if self.model.tet_indices is None or self.model.tri_indices is None:
                raise ValueError("half-cylinder X-FEM scenes require tetrahedra and generated surface triangles")
            self.render_split_mesh = TetMeshCutSurfaceRenderer(
                rest_points=self.render_rest_particle_q_wp.numpy(),
                tet_indices=self.model.tet_indices.numpy(),
                surface_indices=self.model.tri_indices.numpy(),
                knife=self.knife_profile,
                nominal_edge_length=cfg.tet_target_edge,
                max_visual_gap=cfg.max_visual_gap,
            )
        elif args.render_split_mesh and cfg.geometry == "cloth_grid":
            if self.model.tri_indices is None:
                raise ValueError("cloth-grid X-FEM scenes require generated surface triangles")
            self.render_split_mesh = ShellCutSurfaceRenderer(
                rest_points=self.render_rest_particle_q_wp.numpy(),
                surface_indices=self.model.tri_indices.numpy(),
                knife=self.knife_profile,
                nominal_edge_length=max(cfg.cell_x, cfg.cell_y),
                max_visual_gap=cfg.max_visual_gap,
                render_seam_edges=cfg.render_seam_edges,
                render_surface_edges=cfg.render_mesh_edges,
                cut_refine_factor=cfg.render_cut_refine_factor,
            )
        elif args.render_split_mesh and args.render_remesh_mode == "adaptive":
            self.render_split_mesh = AdaptiveCutSurfaceRemesher(
                self.block_pos,
                self.block_hi,
                self.knife_profile,
                max_gap=args.render_gap,
                base_segments=args.adaptive_remesh_base_segments,
                refine_factor=args.adaptive_remesh_refine_factor,
                refine_band=args.adaptive_remesh_refine_band,
                height_segments=args.adaptive_remesh_height_segments,
            )
        elif args.render_split_mesh:
            self.render_split_mesh = SplitCuboidRenderMesh(
                self.block_pos,
                self.block_hi,
                self.knife_profile,
                max_gap=args.render_gap,
                segments=args.render_mesh_segments,
            )
        else:
            self.render_split_mesh = None

        radius_np = self.model.particle_radius.numpy() if self.model.particle_count else np.array([0.01])
        self.render_particle_radius = float(np.mean(radius_np)) * cfg.particle_color_scale * args.render_particle_scale

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(*cfg.camera_pos), pitch=cfg.camera_pitch, yaw=cfg.camera_yaw)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                center = self.block_pos + 0.5 * self.block_size + np.array(cfg.camera_target_offset, dtype=np.float32)
                self.viewer.camera.look_at(wp.vec3(float(center[0]), float(center[1]), float(center[2])))

    def _knife_state(self, time_value: float) -> tuple[float, float, float, tuple[float, float, float]]:
        cfg = self.scenario
        front_x = cfg.knife_start_x + cfg.knife_speed * time_value
        center_y = float(self.knife_profile.center_y_at_x(front_x))
        velocity_y = 0.0
        if abs(cfg.cut_path_amplitude_y) > 1.0e-12:
            wavelength = max(abs(cfg.cut_path_wavelength_x), 1.0e-6)
            phase = 2.0 * math.pi * (front_x - cfg.cut_path_origin_x) / wavelength + cfg.cut_path_phase
            velocity_y = cfg.cut_path_amplitude_y * (2.0 * math.pi / wavelength) * math.cos(phase) * cfg.knife_speed
        if cfg.saw_frequency_hz <= 0.0 or cfg.saw_amplitude_z == 0.0:
            return front_x, center_y, cfg.knife_center_z, (cfg.knife_speed, velocity_y, 0.0)

        omega = 2.0 * math.pi * cfg.saw_frequency_hz
        phase = omega * time_value
        center_z = cfg.knife_center_z + cfg.saw_amplitude_z * math.sin(phase)
        velocity_z = cfg.saw_amplitude_z * omega * math.cos(phase)
        return front_x, center_y, center_z, (cfg.knife_speed, velocity_y, velocity_z)

    def simulate(self):
        cfg = self.scenario
        frame_force = 0.0
        frame_normal = 0.0
        frame_friction = 0.0
        frame_active = 0.0
        frame_damage = 0.0
        for substep in range(self.sim_substeps):
            substep_time = self.sim_time + substep * self.sim_dt
            front_x, blade_center_y, center_z, knife_velocity = self._knife_state(substep_time)
            if cfg.geometry == "cloth_grid":
                knife_tangent = tuple(float(v) for v in self.knife_profile.path_tangent_at_x(front_x))
            else:
                knife_tangent = (0.0, 0.0, 1.0)
            self.solver.set_knife_state(
                front_x=front_x,
                center_y=cfg.knife_center_y,
                center_z=center_z,
                half_width_y=cfg.knife_half_width_y,
                half_width_z=cfg.knife_half_width_z,
                process_width=cfg.process_width,
                knife_velocity=knife_velocity,
                knife_tangent=knife_tangent,
                edge_points=self.knife_profile.edge_points(
                    substep_time,
                    front_x=front_x,
                    center_y=blade_center_y,
                    center_z=center_z,
                ),
                cut_path_amplitude_y=cfg.cut_path_amplitude_y,
                cut_path_wavelength_x=cfg.cut_path_wavelength_x,
                cut_path_phase=cfg.cut_path_phase,
                cut_path_origin_x=cfg.cut_path_origin_x,
            )

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if cfg.wind_strength > 0.0 and self.solver.rest_particle_q is not None:
                wp.launch(
                    apply_xfem_cloth_wind_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_f,
                        self.model.particle_inv_mass,
                        self.model.particle_flags,
                        self.solver.rest_particle_q,
                        cfg.wind_strength,
                        cfg.wind_frequency_hz,
                        substep_time,
                        wp.vec3(*cfg.wind_direction),
                    ],
                    device=self.model.device,
                )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            values = self.solver.force_accum.numpy()
            frame_force += float(values[0])
            frame_active += float(values[1])
            frame_damage = float(values[2]) / max(float(self.model.particle_count), 1.0)
            frame_normal += float(values[3])
            frame_friction += float(values[4])
            self.state_0, self.state_1 = self.state_1, self.state_0
            if cfg.visual_topology_only:
                self.state_0.particle_q.assign(self.render_rest_particle_q_wp)
                self.state_0.particle_qd.zero_()

        inv_substeps = 1.0 / max(float(self.sim_substeps), 1.0)
        self.force_history.append_values(
            self.sim_time,
            frame_force * inv_substeps,
            frame_active * inv_substeps,
            frame_damage,
            normal_force=frame_normal * inv_substeps,
            friction_force=frame_friction * inv_substeps,
        )

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        cfg = self.scenario
        self.viewer.begin_frame(self.sim_time)
        if self.render_split_mesh is not None:
            if isinstance(self.render_split_mesh, ShellCutSurfaceRenderer):
                front_x, _center_y, center_z, _knife_velocity = self._knife_state(self.sim_time)
                stats = self.render_split_mesh.log(
                    self.viewer,
                    self.model.device,
                    self.sim_time,
                    current_points=self.state_0.particle_q,
                    prefix=f"/cutting/xfem_{cfg.name}/cut_surface",
                    surface_color=cfg.surface_color,
                    wall_color=cfg.wall_color,
                    front_x=front_x,
                    center_z=center_z,
                    enrichment_points=None,
                    triangle_cut_state=self.solver.tri_cut_state,
                )
                self.remesh_history.append({"time_s": self.sim_time, **asdict(stats)})
            elif isinstance(self.render_split_mesh, TetMeshCutSurfaceRenderer):
                front_x, _center_y, center_z, _knife_velocity = self._knife_state(self.sim_time)
                stats = self.render_split_mesh.log(
                    self.viewer,
                    self.model.device,
                    self.sim_time,
                    current_points=self.state_0.particle_q,
                    prefix=f"/cutting/xfem_{cfg.name}/cut_surface",
                    surface_color=cfg.surface_color,
                    wall_color=cfg.wall_color,
                    front_x=front_x,
                    center_z=center_z,
                    enrichment_points=self.solver.particle_enrichment_q,
                )
                self.remesh_history.append({"time_s": self.sim_time, **asdict(stats)})
            elif isinstance(self.render_split_mesh, AdaptiveCutSurfaceRemesher):
                stats = self.render_split_mesh.log(
                    self.viewer,
                    self.model.device,
                    self.sim_time,
                    prefix=f"/cutting/xfem_{cfg.name}/adaptive_remesh",
                    surface_color=cfg.surface_color,
                    wall_color=cfg.wall_color,
                    rest_particle_points=self.render_rest_particle_q_wp,
                    particle_points=self.state_0.particle_q,
                )
                self.remesh_history.append({"time_s": self.sim_time, **asdict(stats)})
            else:
                self.render_split_mesh.log(
                    self.viewer,
                    self.model.device,
                    self.sim_time,
                    prefix=f"/cutting/xfem_{cfg.name}/render_split",
                    surface_color=cfg.surface_color,
                    wall_color=cfg.wall_color,
                    rest_particle_points=self.render_rest_particle_q_wp.numpy(),
                    particle_points=self.state_0.particle_q.numpy(),
                )
        else:
            self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.render_particle_radius > 0.0:
            self.viewer.log_points(
                name=f"/cutting/xfem_{cfg.name}/damage_particles",
                points=self.state_0.particle_q,
                radii=self.render_particle_radius,
                colors=self.solver.particle_colors,
            )
        front_x, center_y, center_z, _knife_velocity = self._knife_state(self.sim_time)
        blade = KnifeProfile(
            start_x=front_x,
            speed=0.0,
            center_y=center_y,
            center_z=center_z,
            half_width_y=cfg.knife_half_width_y,
            half_width_z=cfg.knife_half_width_z,
            process_width=cfg.process_width,
            edge_control_points=self.knife_profile.edge_control_points,
            blade_spine_depth=cfg.blade_spine_depth,
            cut_path_amplitude_y=cfg.cut_path_amplitude_y,
            cut_path_wavelength_x=cfg.cut_path_wavelength_x,
            cut_path_phase=cfg.cut_path_phase,
            cut_path_origin_x=cfg.cut_path_origin_x,
        )
        log_knife_mesh(self.viewer, self.model.device, blade, 0.0, prefix=f"/cutting/xfem_{cfg.name}/knife")
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-2.0, -2.0, -1.0)
        p_upper = wp.vec3(2.0, 2.0, 2.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles remain finite and near the X-FEM cutting scene",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        if not self.force_history.forces:
            raise ValueError("X-FEM cutting example did not record a force profile")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_cutting_artifact_args(parser)
        parser.add_argument("--scenario", type=str, default="vegetable_sawing", choices=sorted(SCENARIOS))
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=16)
        parser.add_argument("--iterations", type=int, default=12)
        parser.add_argument("--particle-radius", type=float, default=0.014)
        parser.add_argument("--soft-contact-ke", type=float, default=1.0e3)
        parser.add_argument("--soft-contact-kd", type=float, default=1.0)
        parser.add_argument("--soft-contact-mu", type=float, default=0.75)
        parser.add_argument("--table-friction", type=float, default=1.1)
        parser.add_argument("--render-split-mesh", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--render-remesh-mode", type=str, default="adaptive", choices=["adaptive", "split"])
        parser.add_argument("--render-gap", type=float, default=0.12)
        parser.add_argument("--render-particle-scale", type=float, default=1.0)
        parser.add_argument("--render-mesh-segments", type=int, default=56)
        parser.add_argument("--adaptive-remesh-base-segments", type=int, default=28)
        parser.add_argument("--adaptive-remesh-refine-factor", type=int, default=4)
        parser.add_argument("--adaptive-remesh-refine-band", type=float, default=0.12)
        parser.add_argument("--adaptive-remesh-height-segments", type=int, default=7)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    solver_name = f"xfem_{args.scenario}"
    if args.artifact_dir:
        run_cutting_example(example, args, solver_name)
    else:
        newton.examples.run(example, args)
