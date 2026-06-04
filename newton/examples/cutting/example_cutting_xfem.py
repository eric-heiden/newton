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
from newton.examples.cutting.cutting_common import (
    AdaptiveCutSurfaceRemesher,
    ForceHistory,
    KnifeProfile,
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
        import gmsh
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
    geometry: str = "grid"
    tet_target_edge: float = 0.052

    @property
    def block_size(self) -> np.ndarray:
        if self.geometry == "half_cylinder":
            radius = 0.5 * self.dim_y * self.cell_y
            return np.array([self.dim_x * self.cell_x, 2.0 * radius, radius], dtype=np.float32)
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
        process_width=0.06,
        saw_amplitude_z=0.0,
        saw_frequency_hz=0.0,
        fracture_energy=105.0,
        yield_stress=1.9e4,
        max_damage_rate=13.0,
        separation_speed=0.25,
        force_scale=0.42,
        friction_mu=0.42,
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
        process_width=0.055,
        saw_amplitude_z=0.065,
        saw_frequency_hz=2.4,
        fracture_energy=75.0,
        yield_stress=1.15e4,
        max_damage_rate=18.0,
        separation_speed=0.18,
        force_scale=0.46,
        friction_mu=0.72,
        table_z=0.0,
        table_glue_depth=0.024,
        table_glue_strength=0.55,
        residual_stiffness=0.05,
        damage_threshold=0.18,
        max_visual_gap=0.045,
        surface_color=(0.36, 0.72, 0.31),
        wall_color=(0.94, 0.78, 0.42),
        particle_color_scale=0.55,
        camera_pos=(1.05, -1.18, 0.58),
        camera_pitch=-17.0,
        camera_yaw=128.0,
        geometry="half_cylinder",
        tet_target_edge=0.033,
    ),
    "paper_tearing": XFEMScenario(
        name="paper_tearing",
        block_pos=(-0.38, -0.13, 0.018),
        dim_x=14,
        dim_y=6,
        dim_z=1,
        cell_x=0.052,
        cell_y=0.041,
        cell_z=0.032,
        density=260.0,
        k_mu=1.6e3,
        k_lambda=2.2e3,
        k_damp=2.0e-3,
        gravity=(0.0, 0.0, -2.0),
        fix_left=True,
        knife_start_x=-0.42,
        knife_speed=0.55,
        knife_center_y=0.0,
        knife_center_z=0.024,
        knife_half_width_y=0.055,
        knife_half_width_z=0.075,
        process_width=0.05,
        saw_amplitude_z=0.012,
        saw_frequency_hz=3.0,
        fracture_energy=38.0,
        yield_stress=2.8e3,
        max_damage_rate=18.0,
        separation_speed=0.22,
        force_scale=0.24,
        friction_mu=0.5,
        table_z=0.0,
        table_glue_depth=0.0,
        table_glue_strength=0.0,
        residual_stiffness=0.025,
        damage_threshold=0.14,
        max_visual_gap=0.03,
        surface_color=(0.94, 0.94, 0.90),
        wall_color=(0.65, 0.16, 0.16),
        particle_color_scale=0.35,
        camera_pos=(0.68, -0.85, 0.34),
        camera_pitch=-26.0,
        camera_yaw=132.0,
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
        process_width=0.07,
        saw_amplitude_z=0.05,
        saw_frequency_hz=1.8,
        fracture_energy=55.0,
        yield_stress=6.6e3,
        max_damage_rate=17.0,
        separation_speed=0.12,
        force_scale=0.22,
        friction_mu=0.55,
        table_z=0.0,
        table_glue_depth=0.065,
        table_glue_strength=0.85,
        residual_stiffness=0.04,
        damage_threshold=0.16,
        max_visual_gap=0.06,
        surface_color=(0.86, 0.65, 0.36),
        wall_color=(0.98, 0.83, 0.55),
        particle_color_scale=0.5,
        camera_pos=(0.95, -1.12, 0.55),
        camera_pitch=-19.0,
        camera_yaw=128.0,
        geometry="half_cylinder",
        tet_target_edge=0.038,
    ),
}


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

        builder = newton.ModelBuilder()
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
                center = self.block_pos + 0.5 * self.block_size
                self.viewer.camera.look_at(wp.vec3(float(center[0]), float(center[1]), float(center[2])))

    def _knife_state(self, time_value: float) -> tuple[float, float, tuple[float, float, float]]:
        cfg = self.scenario
        front_x = cfg.knife_start_x + cfg.knife_speed * time_value
        if cfg.saw_frequency_hz <= 0.0 or cfg.saw_amplitude_z == 0.0:
            return front_x, cfg.knife_center_z, (cfg.knife_speed, 0.0, 0.0)

        omega = 2.0 * math.pi * cfg.saw_frequency_hz
        phase = omega * time_value
        center_z = cfg.knife_center_z + cfg.saw_amplitude_z * math.sin(phase)
        velocity_z = cfg.saw_amplitude_z * omega * math.cos(phase)
        return front_x, center_z, (cfg.knife_speed, 0.0, velocity_z)

    def simulate(self):
        cfg = self.scenario
        frame_force = 0.0
        frame_normal = 0.0
        frame_friction = 0.0
        frame_active = 0.0
        frame_damage = 0.0
        for substep in range(self.sim_substeps):
            substep_time = self.sim_time + substep * self.sim_dt
            front_x, center_z, knife_velocity = self._knife_state(substep_time)
            self.solver.set_knife_state(
                front_x=front_x,
                center_y=cfg.knife_center_y,
                center_z=center_z,
                half_width_y=cfg.knife_half_width_y,
                half_width_z=cfg.knife_half_width_z,
                process_width=cfg.process_width,
                knife_velocity=knife_velocity,
                knife_tangent=(0.0, 0.0, 1.0),
                edge_points=self.knife_profile.edge_points(substep_time, front_x=front_x, center_z=center_z),
            )

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            values = self.solver.force_accum.numpy()
            frame_force += float(values[0])
            frame_active += float(values[1])
            frame_damage = float(values[2]) / max(float(self.model.particle_count), 1.0)
            frame_normal += float(values[3])
            frame_friction += float(values[4])
            self.state_0, self.state_1 = self.state_1, self.state_0

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
            if isinstance(self.render_split_mesh, TetMeshCutSurfaceRenderer):
                front_x, center_z, _knife_velocity = self._knife_state(self.sim_time)
                stats = self.render_split_mesh.log(
                    self.viewer,
                    self.model.device,
                    self.sim_time,
                    current_points=self.state_0.particle_q,
                    prefix=f"/cutting/xfem_{cfg.name}/tet_cut_surface",
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
        front_x, center_z, _knife_velocity = self._knife_state(self.sim_time)
        blade = KnifeProfile(
            start_x=front_x,
            speed=0.0,
            center_y=cfg.knife_center_y,
            center_z=center_z,
            half_width_y=cfg.knife_half_width_y,
            half_width_z=cfg.knife_half_width_z,
            process_width=cfg.process_width,
            edge_control_points=self.knife_profile.edge_control_points,
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
