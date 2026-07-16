# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incompressible fluid solver on a staggered MAC grid."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import warp as wp

import newton

from ...core.types import override
from ...sim import StateFlags
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from . import kernels as K
from .boundary import (
    rasterize_colliders_kernel,
    supported_collider_shapes,
    update_faces_u_kernel,
    update_faces_v_kernel,
    update_faces_w_kernel,
)
from .grid import MACGridData
from .pressure import PressureSolver

__all__ = ["SolverMACFluid"]


@wp.kernel(enable_backward=False)
def _rewind_proxy_bodies_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    coupling_forces: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_qd: wp.array[wp.spatial_vector],
):
    local_body = wp.tid()
    proxy_global = body_local_to_proxy_global[local_body]
    if proxy_global < 0:
        return

    f = coupling_forces[proxy_global]
    delta_v = dt * body_inv_mass[local_body] * wp.spatial_top(f)
    rot = wp.transform_get_rotation(body_q[local_body])
    delta_w = dt * wp.quat_rotate(
        rot,
        body_inv_inertia[local_body] * wp.quat_rotate_inv(rot, wp.spatial_bottom(f)),
    )

    body_qd[local_body] = body_qd[local_body] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel(enable_backward=False)
def _harvest_proxy_wrenches_kernel(
    inv_dt: float,
    body_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    body_flags: wp.array[wp.int32],
    body_impulse: wp.array[wp.spatial_vector],
    out_body_f: wp.array[wp.spatial_vector],
):
    local_body = wp.tid()
    proxy_global = body_local_to_proxy_global[local_body]
    if proxy_global < 0 or proxy_global >= out_body_f.shape[0]:
        return
    if (body_flags[local_body] & proxy_flag) == 0:
        return
    wp.atomic_add(out_body_f, proxy_global, body_impulse[local_body] * inv_dt)


class SolverMACFluid(SolverBase, CouplingInterface):
    """Incompressible Newtonian fluid solver on a dense staggered MAC grid.

    .. experimental::

    The solver stores pressure at cell centers and velocity components on
    cell faces of a fixed uniform grid (see the module docstring of the MAC
    grid layout). Each :meth:`step` performs a conventional incompressible
    update: rigid boundary rasterization, semi-Lagrangian advection, gravity
    and external forces, explicit viscosity, moving-boundary velocity
    constraints, and a pressure projection solved with a fixed-iteration
    matrix-free conjugate-gradient method.

    The fluid domain is closed: the grid boundary acts as a static no-slip
    tank wall. Rigid bodies overlapping the grid are treated as moving
    immersed voxelized boundaries; the solver never integrates rigid bodies
    itself. Hydrodynamic wrenches are accumulated from pressure and viscous
    boundary impulses per body and exposed through :attr:`body_impulse` and
    the coupling hooks, so the solver can be used as a destination entry of
    :class:`newton.solvers.SolverCoupledProxy` with a rigid-body solver such
    as MuJoCo owning the articulation.

    All buffers are allocated at construction and :meth:`step` performs no
    host synchronization, so steps can be captured in a CUDA graph.
    Free surfaces, multiphase flow, sparse grids, and differentiability are
    out of scope.
    """

    @dataclass
    class Config:
        """Configuration for :class:`SolverMACFluid`."""

        resolution: tuple[int, int, int] = (32, 32, 32)
        """Grid resolution in cells per axis."""

        cell_size: float = 0.05
        """Uniform cell size [m]."""

        origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """World position of the lower corner of the grid [m]."""

        density: float = 1000.0
        """Fluid density [kg/m^3]."""

        kinematic_viscosity: float = 1.0e-6
        """Kinematic viscosity [m^2/s]. Integrated explicitly; the step raises
        if ``kinematic_viscosity * dt / cell_size**2 > 1/6`` (stability limit)."""

        advection: Literal["semi_lagrangian", "maccormack"] = "semi_lagrangian"
        """Velocity advection scheme. ``semi_lagrangian`` is the diffusive
        first-order default; ``maccormack`` adds a clamped second-order
        error-correction pass that preserves wakes and vortices much longer."""

        pressure_iterations: int = 100
        """Fixed conjugate-gradient iteration count for the pressure solve."""

        collider_shapes: Sequence[int] | None = None
        """Model shape indices treated as fluid boundaries. ``None`` selects
        all shapes with supported geometry types."""

        max_sdf_distance: float | None = None
        """Maximum collider SDF query distance [m]. Defaults to 3 cell sizes."""

        enable_timers: bool = False
        """Collect per-stage wall-clock timings into :attr:`timings`.
        Timing synchronizes the device and is not compatible with CUDA graph capture."""

    def __init__(self, model: newton.Model, config: Config | None = None):
        SolverBase.__init__(self, model)
        self.config = config or SolverMACFluid.Config()
        cfg = self.config

        if cfg.advection not in ("semi_lagrangian", "maccormack"):
            raise ValueError(f"Unsupported advection scheme {cfg.advection!r}")
        self.nx, self.ny, self.nz = (int(r) for r in cfg.resolution)
        self.dx = float(cfg.cell_size)
        self.origin = wp.vec3(*(float(o) for o in cfg.origin))
        self.density = float(cfg.density)
        self._max_sdf_distance = float(cfg.max_sdf_distance) if cfg.max_sdf_distance is not None else 3.0 * self.dx

        device = model.device
        self.grid = MACGridData((self.nx, self.ny, self.nz), device)
        self.pressure_solver = PressureSolver((self.nx, self.ny, self.nz), cfg.pressure_iterations, device)

        # collider shape selection (fixed at construction)
        supported = supported_collider_shapes(model)
        if cfg.collider_shapes is not None:
            requested = [int(s) for s in cfg.collider_shapes]
            unsupported = [s for s in requested if s not in supported]
            if unsupported:
                raise ValueError(f"Unsupported collider shapes for SolverMACFluid: {unsupported}")
            shapes = requested
        else:
            shapes = supported

        with wp.ScopedDevice(device):
            self.collider_shapes = wp.array(shapes, dtype=wp.int32)
            body_count = max(int(model.body_count), 1)
            self.body_impulse = wp.zeros(body_count, dtype=wp.spatial_vector)
            """Hydrodynamic impulse per body over the last step [N·s, N·m·s],
            world frame about each body's center of mass."""
            self._diag_vec = wp.zeros(K.DIAG_V_COUNT, dtype=wp.vec3)
            self._diag_scalar = wp.zeros(K.DIAG_S_COUNT, dtype=float)
            self._fluid_cell_count = wp.zeros(1, dtype=wp.int32)
            self._div_sum = wp.zeros(1, dtype=float)
            self._external_accel = wp.zeros(1, dtype=wp.vec3)
            # dummies passed to kernels when the model owns no bodies
            self._dummy_body_q = wp.zeros(1, dtype=wp.transform)
            self._dummy_body_qd = wp.zeros(1, dtype=wp.spatial_vector)
            self._dummy_body_com = wp.zeros(1, dtype=wp.vec3)

        self._restore_pending = False
        self._last_dt = 0.0
        self.timings: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def velocity_u(self) -> wp.array3d:
        """X-velocity at x-faces [m/s], shape ``(nx + 1, ny, nz)``."""
        return self.grid.u

    @property
    def velocity_v(self) -> wp.array3d:
        """Y-velocity at y-faces [m/s], shape ``(nx, ny + 1, nz)``."""
        return self.grid.v

    @property
    def velocity_w(self) -> wp.array3d:
        """Z-velocity at z-faces [m/s], shape ``(nx, ny, nz + 1)``."""
        return self.grid.w

    @property
    def pressure(self) -> wp.array3d:
        """Pressure at cell centers [Pa], shape ``(nx, ny, nz)``."""
        return self.grid.pressure

    @property
    def divergence(self) -> wp.array3d:
        """Post-projection velocity divergence at cell centers [1/s]."""
        return self.grid.divergence

    @property
    def cell_label(self) -> wp.array3d:
        """Cell labels: ``-2`` fluid, ``-1`` static solid, ``>= 0`` rigid body index."""
        return self.grid.cell_label

    def set_external_acceleration(self, accel) -> None:
        """Set a uniform external acceleration applied to the fluid [m/s^2]."""
        self._external_accel.assign([wp.vec3(*accel)])

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def _timer(self, name: str):
        return wp.ScopedTimer(name, active=self.config.enable_timers, print=False, synchronize=True, dict=self.timings)

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        del control, contacts
        cfg = self.config
        if dt <= 0.0:
            raise ValueError("SolverMACFluid.step requires dt > 0")

        visc_coeff = cfg.kinematic_viscosity * dt / (self.dx * self.dx)
        if visc_coeff > 1.0 / 6.0 + 1.0e-9:
            raise ValueError(
                f"Explicit viscosity is unstable: kinematic_viscosity * dt / cell_size**2 = {visc_coeff:.4g} "
                "exceeds 1/6. Reduce dt or the viscosity, or increase the cell size."
            )

        model = self.model
        g = self.grid
        self._last_dt = float(dt)
        mass_face = self.density * self.dx**3

        body_q = state_in.body_q if state_in.body_q is not None else self._dummy_body_q
        body_qd = state_in.body_qd if state_in.body_qd is not None else self._dummy_body_qd
        body_com = model.body_com if model.body_count > 0 else self._dummy_body_com

        with wp.ScopedDevice(model.device):
            # coupled-iteration state restoration: a proxy-iteration restart
            # repeats the same physical interval, so the velocity grid must
            # return to its beginning-of-step checkpoint instead of advancing.
            if self._restore_pending:
                wp.copy(g.u, g.u_checkpoint)
                wp.copy(g.v, g.v_checkpoint)
                wp.copy(g.w, g.w_checkpoint)
            else:
                # saving unconditionally keeps standalone and coupled stepping
                # uniform and graph-capture friendly (three device copies)
                wp.copy(g.u_checkpoint, g.u)
                wp.copy(g.v_checkpoint, g.v)
                wp.copy(g.w_checkpoint, g.w)
            self._restore_pending = False

            self.body_impulse.zero_()
            self._diag_vec.zero_()
            self._diag_scalar.zero_()
            self._fluid_cell_count.zero_()
            self._div_sum.zero_()

            # 1. rigid boundary rasterization
            with self._timer("boundary"):
                wp.launch(
                    rasterize_colliders_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        self.origin,
                        self.dx,
                        self.collider_shapes,
                        model.shape_body,
                        model.shape_transform,
                        model.shape_type,
                        model.shape_scale,
                        model.shape_source_ptr,
                        body_q,
                        self._max_sdf_distance,
                        g.cell_label,
                        g.cell_sdf,
                        self._fluid_cell_count,
                    ],
                )
                face_args = [self.origin, self.dx, g.cell_label, g.cell_sdf, body_q, body_qd, body_com]
                wp.launch(update_faces_u_kernel, dim=g.u.shape, inputs=[*face_args, g.u, g.u_solid])
                wp.launch(update_faces_v_kernel, dim=g.v.shape, inputs=[*face_args, g.v, g.v_solid])
                wp.launch(update_faces_w_kernel, dim=g.w.shape, inputs=[*face_args, g.w, g.w_solid])
                self.pressure_solver.build(self.dx, g.cell_label)

            # 2. advection (semi-Lagrangian RK2, optionally MacCormack-corrected)
            with self._timer("advection"):
                adv_args = [self.origin, self.dx, dt, g.cell_label, g.u, g.v, g.w]
                wp.launch(K.advect_u_kernel, dim=g.u.shape, inputs=[*adv_args, g.u, g.u_tmp])
                wp.launch(K.advect_v_kernel, dim=g.v.shape, inputs=[*adv_args, g.v, g.v_tmp])
                wp.launch(K.advect_w_kernel, dim=g.w.shape, inputs=[*adv_args, g.w, g.w_tmp])
                if cfg.advection == "maccormack":
                    # backward pass of the advected field through the same flow
                    back_args = [self.origin, self.dx, -dt, g.cell_label, g.u, g.v, g.w]
                    wp.launch(K.advect_u_kernel, dim=g.u.shape, inputs=[*back_args, g.u_tmp, g.u_tmp2])
                    wp.launch(K.advect_v_kernel, dim=g.v.shape, inputs=[*back_args, g.v_tmp, g.v_tmp2])
                    wp.launch(K.advect_w_kernel, dim=g.w.shape, inputs=[*back_args, g.w_tmp, g.w_tmp2])
                    # clamped error correction, written back over q_hat
                    wp.launch(
                        K.maccormack_correct_u_kernel,
                        dim=g.u.shape,
                        inputs=[*adv_args, g.u_tmp, g.u_tmp2, g.u_tmp],
                    )
                    wp.launch(
                        K.maccormack_correct_v_kernel,
                        dim=g.v.shape,
                        inputs=[*adv_args, g.v_tmp, g.v_tmp2, g.v_tmp],
                    )
                    wp.launch(
                        K.maccormack_correct_w_kernel,
                        dim=g.w.shape,
                        inputs=[*adv_args, g.w_tmp, g.w_tmp2, g.w_tmp],
                    )
                wp.copy(g.u, g.u_tmp)
                wp.copy(g.v, g.v_tmp)
                wp.copy(g.w, g.w_tmp)

            # fluid momentum after advection (diagnostics baseline)
            mom_args = [mass_face, g.cell_label]
            wp.launch(
                K.momentum_u_kernel, dim=g.u.shape, inputs=[*mom_args, g.u, self._diag_vec, int(K.DIAG_V_MOMENTUM_PRE)]
            )
            wp.launch(
                K.momentum_v_kernel, dim=g.v.shape, inputs=[*mom_args, g.v, self._diag_vec, int(K.DIAG_V_MOMENTUM_PRE)]
            )
            wp.launch(
                K.momentum_w_kernel, dim=g.w.shape, inputs=[*mom_args, g.w, self._diag_vec, int(K.DIAG_V_MOMENTUM_PRE)]
            )

            # 3. gravity and external forces
            with self._timer("forces"):
                force_args = [model.gravity, self._external_accel, dt, mass_face, g.cell_label]
                wp.launch(K.add_forces_u_kernel, dim=g.u.shape, inputs=[*force_args, g.u, self._diag_vec])
                wp.launch(K.add_forces_v_kernel, dim=g.v.shape, inputs=[*force_args, g.v, self._diag_vec])
                wp.launch(K.add_forces_w_kernel, dim=g.w.shape, inputs=[*force_args, g.w, self._diag_vec])

            # 4. viscosity (explicit diffusion, includes viscous wrench on solids)
            if visc_coeff > 0.0:
                with self._timer("viscosity"):
                    visc_args = [self.origin, self.dx, visc_coeff, mass_face, g.cell_label, g.cell_sdf]
                    wrench_args = [self.body_impulse, body_q, body_com, self._diag_vec]
                    wp.launch(K.diffuse_u_kernel, dim=g.u.shape, inputs=[*visc_args, g.u, g.u_tmp, *wrench_args])
                    wp.launch(K.diffuse_v_kernel, dim=g.v.shape, inputs=[*visc_args, g.v, g.v_tmp, *wrench_args])
                    wp.launch(K.diffuse_w_kernel, dim=g.w.shape, inputs=[*visc_args, g.w, g.w_tmp, *wrench_args])
                    wp.copy(g.u, g.u_tmp)
                    wp.copy(g.v, g.v_tmp)
                    wp.copy(g.w, g.w_tmp)

            # 5. pressure projection
            with self._timer("pressure_solve"):
                wp.launch(
                    K.divergence_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[self.dx, g.cell_label, g.u, g.v, g.w, g.divergence, self._diag_scalar, 1],
                )
                # deterministic reduction (atomic sums would make repeated
                # coupling iterations non-reproducible on GPU)
                wp.utils.array_sum(g.divergence, out=self._div_sum)
                wp.launch(
                    K.pressure_rhs_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[g.cell_label, g.divergence, self._div_sum, self._fluid_cell_count, self.pressure_solver.b],
                )
                self.pressure_solver.solve(self.dx, g.cell_label, self._fluid_cell_count)

            with self._timer("projection"):
                q = self.pressure_solver.q
                wp.launch(K.apply_gradient_u_kernel, dim=g.u.shape, inputs=[self.dx, g.cell_label, q, g.u])
                wp.launch(K.apply_gradient_v_kernel, dim=g.v.shape, inputs=[self.dx, g.cell_label, q, g.v])
                wp.launch(K.apply_gradient_w_kernel, dim=g.w.shape, inputs=[self.dx, g.cell_label, q, g.w])
                wp.launch(
                    K.pressure_wrench_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        self.origin,
                        self.dx,
                        self.density,
                        g.cell_label,
                        g.cell_sdf,
                        q,
                        g.pressure,
                        1.0 / dt,
                        self.body_impulse,
                        body_q,
                        body_com,
                        self._diag_vec,
                    ],
                )

            # 6. diagnostics
            with self._timer("diagnostics"):
                wp.launch(
                    K.divergence_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[self.dx, g.cell_label, g.u, g.v, g.w, g.divergence, self._diag_scalar, 0],
                )
                noslip_args = [self.origin, self.dx, g.cell_label, body_q, body_qd, body_com]
                wp.launch(K.noslip_error_u_kernel, dim=g.u.shape, inputs=[*noslip_args, g.u, self._diag_scalar])
                wp.launch(K.noslip_error_v_kernel, dim=g.v.shape, inputs=[*noslip_args, g.v, self._diag_scalar])
                wp.launch(K.noslip_error_w_kernel, dim=g.w.shape, inputs=[*noslip_args, g.w, self._diag_scalar])
                wp.launch(
                    K.momentum_u_kernel,
                    dim=g.u.shape,
                    inputs=[*mom_args, g.u, self._diag_vec, int(K.DIAG_V_MOMENTUM_POST)],
                )
                wp.launch(
                    K.momentum_v_kernel,
                    dim=g.v.shape,
                    inputs=[*mom_args, g.v, self._diag_vec, int(K.DIAG_V_MOMENTUM_POST)],
                )
                wp.launch(
                    K.momentum_w_kernel,
                    dim=g.w.shape,
                    inputs=[*mom_args, g.w, self._diag_vec, int(K.DIAG_V_MOMENTUM_POST)],
                )

            # rigid state passes through unchanged (bodies are owned elsewhere)
            if state_out is not state_in:
                if state_in.body_q is not None and state_out.body_q is not None:
                    wp.copy(state_out.body_q, state_in.body_q)
                if state_in.body_qd is not None and state_out.body_qd is not None:
                    wp.copy(state_out.body_qd, state_in.body_qd)

    @override
    def reset(self, state: newton.State, world_mask: wp.array | None = None, flags=None) -> None:
        del state, world_mask, flags
        g = self.grid
        for arr in (
            g.u,
            g.v,
            g.w,
            g.u_tmp,
            g.v_tmp,
            g.w_tmp,
            g.u_tmp2,
            g.v_tmp2,
            g.w_tmp2,
            g.u_checkpoint,
            g.v_checkpoint,
            g.w_checkpoint,
            g.u_solid,
            g.v_solid,
            g.w_solid,
            g.pressure,
            g.divergence,
        ):
            arr.zero_()
        self.body_impulse.zero_()
        self._restore_pending = False

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def read_diagnostics(self) -> dict:
        """Read solver diagnostics for the last step (synchronizes with the host).

        Returns:
            Dictionary with divergence norms before/after projection [1/s],
            pressure-solver residual, no-slip error [m/s], per-body
            hydrodynamic wrench [N, N·m], and the fluid momentum-balance
            (action-reaction) error [N·s].
        """
        s = self._diag_scalar.numpy()
        v = self._diag_vec.numpy()
        body_impulse = self.body_impulse.numpy()
        dt = self._last_dt if self._last_dt > 0.0 else 1.0
        fluid_cells = int(self._fluid_cell_count.numpy()[0])

        momentum_pre = v[int(K.DIAG_V_MOMENTUM_PRE)]
        momentum_post = v[int(K.DIAG_V_MOMENTUM_POST)]
        impulse_external = v[int(K.DIAG_V_IMPULSE_EXTERNAL)]
        impulse_viscous = v[int(K.DIAG_V_IMPULSE_VISCOUS)]
        impulse_pressure = v[int(K.DIAG_V_IMPULSE_PRESSURE)]
        balance = (momentum_post - momentum_pre) - (impulse_external + impulse_viscous + impulse_pressure)

        n = max(fluid_cells, 1)
        return {
            "fluid_cell_count": fluid_cells,
            "div_l2_pre": math.sqrt(s[int(K.DIAG_S_DIV_L2_PRE)] / n),
            "div_linf_pre": float(s[int(K.DIAG_S_DIV_LINF_PRE)]),
            "div_l2_post": math.sqrt(s[int(K.DIAG_S_DIV_L2_POST)] / n),
            "div_linf_post": float(s[int(K.DIAG_S_DIV_LINF_POST)]),
            "pressure_residual": math.sqrt(float(self.pressure_solver.residual_sq.numpy()[0])),
            "noslip_max": float(s[int(K.DIAG_S_NOSLIP_MAX)]),
            "noslip_mean": float(s[int(K.DIAG_S_NOSLIP_SUM)]) / max(s[int(K.DIAG_S_NOSLIP_COUNT)], 1.0),
            "body_wrench": body_impulse / dt,
            "momentum_balance_error": [float(b) for b in balance],
            "boundary_impulse_pressure": [float(b) for b in impulse_pressure],
            "boundary_impulse_viscous": [float(b) for b in impulse_viscous],
        }

    # ------------------------------------------------------------------
    # coupling hooks (see CouplingInterface)
    # ------------------------------------------------------------------

    @override
    def coupling_eval_gravity_acceleration(
        self,
        out_body_acceleration: wp.array | None,
        out_particle_acceleration: wp.array | None,
    ) -> None:
        # the fluid solver never integrates rigid bodies, so it applies no
        # gravity to proxy bodies
        if out_body_acceleration is not None:
            out_body_acceleration.zero_()
        if out_particle_acceleration is not None:
            super().coupling_eval_gravity_acceleration(None, out_particle_acceleration)

    @override
    def coupling_notify_input_state_update(
        self,
        state: newton.State,
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        del state, flags, dt
        if iteration_restart:
            # the coupler is repeating the same physical interval: restore the
            # beginning-of-step fluid state at the next step() call
            self._restore_pending = True

    @override
    def coupling_rewind_proxy_body(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: newton.State,
        coupling_forces: wp.array[wp.spatial_vector],
        body_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Remove lagged proxy feedback from the boundary velocities the fluid sees."""
        del body_gravity_acceleration
        if state.body_q is None or state.body_qd is None or body_local_to_proxy_global.shape[0] == 0:
            return
        wp.launch(
            _rewind_proxy_bodies_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                body_local_to_proxy_global,
                coupling_forces,
                state.body_q,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                state.body_qd,
            ],
            device=self.model.device,
        )

    @override
    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: newton.State,
        state_out: newton.State,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Convert accumulated immersed-boundary impulses to proxy-body wrenches."""
        del body_qd_before, state, state_out, contacts
        if dt <= 0.0:
            raise ValueError("MAC fluid proxy wrench harvesting requires a positive dt")
        out_body_f.zero_()
        if body_local_to_proxy_global.shape[0] == 0:
            return
        wp.launch(
            _harvest_proxy_wrenches_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                1.0 / dt,
                body_local_to_proxy_global,
                int(newton.BodyFlags.PROXY),
                self.model.body_flags,
                self.body_impulse,
                out_body_f,
            ],
            device=self.model.device,
        )
