# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from ..vbd import SolverVBD
from ..xpbd import SolverXPBD
from .kernels import (
    apply_xfem_knife_kernel,
    apply_xfem_post_constraints_kernel,
    classify_xfem_tets_kernel,
    cut_xfem_cloth_edges_kernel,
    cut_xfem_cloth_triangles_kernel,
    degrade_xfem_tets_kernel,
    update_xfem_shell_enrichment_kernel,
)

__all__ = ["SolverXFEMCut"]


MAX_XFEM_KNIFE_EDGE_POINTS = 32


def _vec3_tuple(value: Sequence[float] | wp.vec3) -> tuple[float, float, float]:
    if isinstance(value, wp.vec3):
        return (float(value[0]), float(value[1]), float(value[2]))
    if len(value) != 3:
        raise ValueError("expected a 3-vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _resample_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    if points.shape[0] <= sample_count:
        return points.astype(np.float32, copy=True)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 1.0e-12:
        return np.repeat(points[:1], sample_count, axis=0).astype(np.float32, copy=False)

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, total_length, sample_count)
    samples = np.empty((sample_count, 3), dtype=np.float32)
    for i, target in enumerate(targets):
        segment = min(int(np.searchsorted(cumulative, target, side="right")) - 1, points.shape[0] - 2)
        segment = max(segment, 0)
        denom = max(float(segment_lengths[segment]), 1.0e-12)
        alpha = float((target - cumulative[segment]) / denom)
        samples[i] = (1.0 - alpha) * points[segment] + alpha * points[segment + 1]
    return samples


class SolverXFEMCut(SolverBase):
    """Prototype X-FEM-style cutting solver for soft solids and shell cloth.

    The solver owns cut and enrichment state, then delegates the underlying
    material response to the appropriate Newton solver: XPBD for tetrahedral
    soft solids and VBD for triangle shell cloth. Warp kernels add
    shifted-Heaviside cut classification, enriched side displacement, cut-cell
    quadrature descriptors, cohesive damage, online material/topology softening,
    knife friction, and table/glue projection around that base solve. This
    keeps the X-FEM data path explicit and extensible while preserving the
    material model expected by each mesh type.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 8,
        soft_body_relaxation: float = 0.9,
        fracture_energy: float = 95.0,
        yield_stress: float = 1.8e4,
        max_damage_rate: float = 14.0,
        separation_stiffness: float = 80.0,
        separation_speed: float = 0.22,
        force_scale: float = 0.42,
        knife_friction_mu: float = 0.55,
        friction_velocity_scale: float = 0.08,
        max_knife_velocity_delta: float | None = None,
        damage_threshold: float = 0.22,
        residual_stiffness: float = 0.08,
        max_enrichment: float = 0.045,
        max_visual_gap: float = 0.045,
        table_z: float = 0.0,
        table_glue_depth: float = 0.0,
        table_glue_strength: float = 0.0,
        table_friction: float = 0.8,
    ):
        super().__init__(model=model)

        self.iterations = int(iterations)
        self.fracture_energy = float(fracture_energy)
        self.yield_stress = float(yield_stress)
        self.max_damage_rate = float(max_damage_rate)
        self.separation_stiffness = float(separation_stiffness)
        self.separation_speed = float(separation_speed)
        self.force_scale = float(force_scale)
        self.knife_friction_mu = float(knife_friction_mu)
        self.friction_velocity_scale = float(friction_velocity_scale)
        self.damage_threshold = float(damage_threshold)
        self.residual_stiffness = float(residual_stiffness)
        self.max_enrichment = float(max_enrichment)
        self.max_visual_gap = float(max_visual_gap)
        self.table_z = float(table_z)
        self.table_glue_depth = float(table_glue_depth)
        self.table_glue_strength = float(table_glue_strength)
        self.table_friction = float(table_friction)

        self._uses_shell_cloth_solver = model.tet_count == 0 and model.tri_count > 0
        if max_knife_velocity_delta is None:
            self.max_knife_velocity_delta = 0.12 if self._uses_shell_cloth_solver else 0.0
        else:
            self.max_knife_velocity_delta = max(0.0, float(max_knife_velocity_delta))
        if self._uses_shell_cloth_solver:
            self._base_solver = SolverVBD(model, iterations=self.iterations, particle_enable_self_contact=False)
        else:
            self._base_solver = SolverXPBD(
                model,
                iterations=self.iterations,
                soft_body_relaxation=soft_body_relaxation,
                enable_restitution=False,
            )

        self.rest_particle_q = wp.clone(model.particle_q) if model.particle_count else None
        self.particle_damage = wp.zeros(model.particle_count, dtype=float, device=model.device)
        self.particle_cut_side = wp.zeros(model.particle_count, dtype=float, device=model.device)
        self.particle_enrichment_q = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.particle_enrichment_qd = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.particle_colors = wp.full(
            model.particle_count, wp.vec3(0.18, 0.58, 0.42), dtype=wp.vec3, device=model.device
        )

        self.tet_cut_state = wp.zeros(model.tet_count, dtype=wp.int32, device=model.device)
        self.tet_damage = wp.zeros(model.tet_count, dtype=float, device=model.device)
        self.tet_cut_weight = wp.zeros(model.tet_count, dtype=float, device=model.device)
        self.base_tet_materials = wp.clone(model.tet_materials) if model.tet_count and model.tet_materials else None

        self.tri_cut_state = wp.zeros(model.tri_count, dtype=wp.int32, device=model.device)
        self.base_tri_materials = (
            wp.clone(model.tri_materials) if model.tri_count and model.tri_materials is not None else None
        )
        self.edge_cut_state = wp.zeros(model.edge_count, dtype=wp.int32, device=model.device)
        self.base_edge_bending_properties = (
            wp.clone(model.edge_bending_properties)
            if model.edge_count and model.edge_bending_properties is not None
            else None
        )
        self.cloth_cut_counts = wp.zeros(3, dtype=wp.int32, device=model.device)
        self.shell_quad_triangle_indices = wp.zeros(0, dtype=wp.int32, device=model.device)
        self.shell_quad_barycentric = wp.zeros(0, dtype=wp.vec3, device=model.device)
        self.shell_quad_side = wp.zeros(0, dtype=wp.int32, device=model.device)
        self.shell_quad_area = wp.zeros(0, dtype=float, device=model.device)
        self.shell_quad_count = 0
        self.shell_quad_cut_triangle_count = 0
        self.shell_quad_total_area = 0.0

        self.force_accum = wp.zeros(6, dtype=float, device=model.device)
        self.knife_edge_points = wp.zeros(MAX_XFEM_KNIFE_EDGE_POINTS, dtype=wp.vec3, device=model.device)
        self.knife_edge_point_count = 2

        self.particle_area = self._estimate_particle_area()
        self.set_knife_state(front_x=-1.0, center_y=0.0, center_z=0.0)

    def set_shell_quadrature(
        self,
        *,
        triangle_indices: np.ndarray,
        barycentric_coords: np.ndarray,
        side: np.ndarray,
        area: np.ndarray,
        cut_triangle_count: int = 0,
    ) -> None:
        """Attach fixed side-aware cut-cell quadrature for shell X-FEM scenes."""

        tri_np = np.asarray(triangle_indices, dtype=np.int32).reshape(-1)
        bary_np = np.asarray(barycentric_coords, dtype=np.float32).reshape(-1, 3)
        side_np = np.asarray(side, dtype=np.int32).reshape(-1)
        area_np = np.asarray(area, dtype=np.float32).reshape(-1)
        count = int(tri_np.shape[0])
        if bary_np.shape[0] != count or side_np.shape[0] != count or area_np.shape[0] != count:
            raise ValueError("shell quadrature arrays must have matching lengths")
        if count and (np.any(tri_np < 0) or np.any(tri_np >= self.model.tri_count)):
            raise ValueError("shell quadrature triangle indices are out of range")
        if count and not np.all(np.isfinite(bary_np)):
            raise ValueError("shell quadrature barycentric coordinates must be finite")
        if count and not np.all(np.isfinite(area_np)):
            raise ValueError("shell quadrature areas must be finite")

        self.shell_quad_triangle_indices = wp.array(tri_np, dtype=wp.int32, device=self.model.device)
        self.shell_quad_barycentric = wp.array(bary_np, dtype=wp.vec3, device=self.model.device)
        self.shell_quad_side = wp.array(side_np, dtype=wp.int32, device=self.model.device)
        self.shell_quad_area = wp.array(area_np, dtype=float, device=self.model.device)
        self.shell_quad_count = count
        self.shell_quad_cut_triangle_count = int(cut_triangle_count)
        self.shell_quad_total_area = float(np.sum(area_np)) if count else 0.0

    def _estimate_particle_area(self) -> float:
        if self.model.particle_count == 0 or self.model.particle_q is None:
            return 1.0
        if self.model.tet_count == 0 and self.model.tri_count and self.model.tri_areas is not None:
            tri_areas = self.model.tri_areas.numpy()
            total_area = float(np.sum(tri_areas)) if tri_areas.size else 0.0
            return max(total_area / max(float(self.model.particle_count), 1.0), 1.0e-12)
        points = self.model.particle_q.numpy()
        if points.size == 0:
            return 1.0
        extents = np.maximum(np.ptp(points, axis=0), 1.0e-4)
        volume = float(math.prod(extents)) / max(float(points.shape[0]), 1.0)
        return max(volume, 1.0e-18) ** (2.0 / 3.0)

    def set_knife_state(
        self,
        *,
        front_x: float,
        center_y: float,
        center_z: float,
        half_width_y: float = 0.06,
        half_width_z: float = 0.24,
        process_width: float = 0.06,
        knife_velocity: Sequence[float] | wp.vec3 = (0.0, 0.0, 0.0),
        knife_tangent: Sequence[float] | wp.vec3 = (0.0, 0.0, 1.0),
        edge_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        cut_path_amplitude_y: float = 0.0,
        cut_path_wavelength_x: float = 1.0,
        cut_path_phase: float = 0.0,
        cut_path_origin_x: float = 0.0,
    ) -> None:
        """Set the current blade state used by the next :meth:`step` call."""

        tangent = np.asarray(_vec3_tuple(knife_tangent), dtype=np.float64)
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-12:
            tangent = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            tangent = tangent / norm

        self.knife_front_x = float(front_x)
        self.knife_center_y = float(center_y)
        self.knife_center_z = float(center_z)
        self.knife_half_width_y = float(half_width_y)
        self.knife_half_width_z = float(half_width_z)
        self.knife_process_width = float(process_width)
        self.knife_velocity = _vec3_tuple(knife_velocity)
        self.knife_tangent = (float(tangent[0]), float(tangent[1]), float(tangent[2]))
        self.cut_path_amplitude_y = float(cut_path_amplitude_y)
        self.cut_path_wavelength_x = float(cut_path_wavelength_x)
        self.cut_path_phase = float(cut_path_phase)
        self.cut_path_origin_x = float(cut_path_origin_x)

        if edge_points is None:
            edge_np = np.array(
                [
                    [self.knife_front_x, self.knife_center_y, self.knife_center_z - self.knife_half_width_z],
                    [self.knife_front_x, self.knife_center_y, self.knife_center_z + self.knife_half_width_z],
                ],
                dtype=np.float32,
            )
        else:
            edge_np = np.asarray(edge_points, dtype=np.float32)
            if edge_np.ndim != 2 or edge_np.shape[1] != 3 or edge_np.shape[0] < 2:
                raise ValueError("edge_points must have shape (N, 3) with N >= 2")
            edge_np = _resample_polyline(edge_np, MAX_XFEM_KNIFE_EDGE_POINTS)

        self.knife_edge_point_count = int(edge_np.shape[0])
        edge_buffer = np.zeros((MAX_XFEM_KNIFE_EDGE_POINTS, 3), dtype=np.float32)
        edge_buffer[: self.knife_edge_point_count] = edge_np
        self.knife_edge_points.assign(edge_buffer)

    def _classify_and_degrade(self, particle_q: wp.array) -> None:
        model = self.model
        if model.tet_count == 0 or model.tet_indices is None:
            return

        wp.launch(
            classify_xfem_tets_kernel,
            dim=model.tet_count,
            inputs=[
                particle_q,
                self.particle_damage,
                self.particle_cut_side,
                model.tet_indices,
                self.tet_cut_state,
                self.tet_damage,
                self.tet_cut_weight,
                self.knife_edge_points,
                self.knife_edge_point_count,
                self.knife_front_x,
                self.knife_center_y,
                self.knife_center_z,
                self.knife_half_width_y,
                self.knife_half_width_z,
                self.knife_process_width,
                self.damage_threshold,
                self.cut_path_amplitude_y,
                self.cut_path_wavelength_x,
                self.cut_path_phase,
                self.cut_path_origin_x,
            ],
            device=model.device,
        )

        if model.tet_materials is not None and self.base_tet_materials is not None:
            wp.launch(
                degrade_xfem_tets_kernel,
                dim=model.tet_count,
                inputs=[
                    self.tet_cut_state,
                    self.tet_damage,
                    model.tet_materials,
                    self.base_tet_materials,
                    self.residual_stiffness,
                ],
                device=model.device,
            )

    def _update_cloth_topology(self) -> None:
        model = self.model
        if self.rest_particle_q is None or not self._uses_shell_cloth_solver:
            return

        self.cloth_cut_counts.zero_()
        if (
            model.edge_count
            and model.edge_indices is not None
            and model.edge_bending_properties is not None
            and self.base_edge_bending_properties is not None
        ):
            wp.launch(
                cut_xfem_cloth_edges_kernel,
                dim=model.edge_count,
                inputs=[
                    self.rest_particle_q,
                    model.edge_indices,
                    model.edge_bending_properties,
                    self.base_edge_bending_properties,
                    self.edge_cut_state,
                    self.cloth_cut_counts,
                    self.knife_front_x,
                    self.knife_center_y,
                    self.knife_center_z,
                    self.knife_half_width_z,
                    self.knife_process_width,
                    self.cut_path_amplitude_y,
                    self.cut_path_wavelength_x,
                    self.cut_path_phase,
                    self.cut_path_origin_x,
                ],
                device=model.device,
            )

        if (
            model.tri_count
            and model.tri_indices is not None
            and model.tri_materials is not None
            and self.base_tri_materials is not None
        ):
            wp.launch(
                cut_xfem_cloth_triangles_kernel,
                dim=model.tri_count,
                inputs=[
                    self.rest_particle_q,
                    model.tri_indices,
                    model.tri_materials,
                    self.base_tri_materials,
                    self.tri_cut_state,
                    self.cloth_cut_counts,
                    self.knife_front_x,
                    self.knife_center_y,
                    self.knife_center_z,
                    self.knife_half_width_z,
                    self.knife_process_width,
                    self.cut_path_amplitude_y,
                    self.cut_path_wavelength_x,
                    self.cut_path_phase,
                    self.cut_path_origin_x,
                ],
                device=model.device,
            )

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (
            SolverNotifyFlags.BODY_PROPERTIES
            | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            | SolverNotifyFlags.MODEL_PROPERTIES
        ):
            self._base_solver.notify_model_changed(flags)

    @override
    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        model = self.model
        if control is None:
            control = model.control(clone_variables=False)

        self.force_accum.zero_()

        if model.particle_count:
            wp.launch(
                apply_xfem_knife_kernel,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.particle_damage,
                    self.particle_cut_side,
                    self.particle_enrichment_q,
                    self.particle_enrichment_qd,
                    self.particle_colors,
                    self.force_accum,
                    self.knife_edge_points,
                    self.knife_edge_point_count,
                    self.knife_front_x,
                    self.knife_center_y,
                    self.knife_center_z,
                    self.knife_half_width_y,
                    self.knife_half_width_z,
                    self.knife_process_width,
                    dt,
                    self.particle_area,
                    self.fracture_energy,
                    self.yield_stress,
                    self.max_damage_rate,
                    self.separation_stiffness,
                    self.separation_speed,
                    self.force_scale,
                    self.knife_friction_mu,
                    self.friction_velocity_scale,
                    self.max_knife_velocity_delta,
                    wp.vec3(*self.knife_velocity),
                    wp.vec3(*self.knife_tangent),
                    self.max_enrichment,
                    self.cut_path_amplitude_y,
                    self.cut_path_wavelength_x,
                    self.cut_path_phase,
                    self.cut_path_origin_x,
                ],
                device=model.device,
            )

        if model.tet_count:
            self._classify_and_degrade(state_in.particle_q)
        if model.tri_count or model.edge_count:
            self._update_cloth_topology()

        self._base_solver.step(state_in, state_out, control, contacts, dt)

        if self._uses_shell_cloth_solver and model.particle_count and self.rest_particle_q is not None:
            wp.launch(
                update_xfem_shell_enrichment_kernel,
                dim=model.particle_count,
                inputs=[
                    self.rest_particle_q,
                    model.particle_flags,
                    self.particle_enrichment_q,
                    self.particle_enrichment_qd,
                    self.knife_front_x,
                    self.knife_center_y,
                    self.knife_center_z,
                    self.knife_half_width_z,
                    self.knife_process_width,
                    dt,
                    self.max_visual_gap,
                    self.cut_path_amplitude_y,
                    self.cut_path_wavelength_x,
                    self.cut_path_phase,
                    self.cut_path_origin_x,
                ],
                device=model.device,
            )

        if model.particle_count and self.rest_particle_q is not None and not self._uses_shell_cloth_solver:
            wp.launch(
                apply_xfem_post_constraints_kernel,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.rest_particle_q,
                    self.particle_damage,
                    self.particle_cut_side,
                    self.particle_enrichment_q,
                    self.knife_front_x,
                    self.knife_center_y,
                    self.knife_process_width,
                    self.max_visual_gap,
                    self.cut_path_amplitude_y,
                    self.cut_path_wavelength_x,
                    self.cut_path_phase,
                    self.cut_path_origin_x,
                    self.table_z,
                    self.table_glue_depth,
                    self.table_glue_strength,
                    self.table_friction,
                    dt,
                ],
                device=model.device,
            )

        if model.tet_count:
            self._classify_and_degrade(state_out.particle_q)

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        self._base_solver.update_contacts(contacts, state)
