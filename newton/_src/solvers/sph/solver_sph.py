# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase
from .kernels import compute_sph_density_pressure, integrate_sph_particles


def _vec3(value: Sequence[float] | wp.vec3 | None, default: tuple[float, float, float]) -> wp.vec3:
    if value is None:
        return wp.vec3(*default)
    return wp.vec3(float(value[0]), float(value[1]), float(value[2]))


class SolverSPH(SolverBase):
    """Weakly-compressible SPH prototype for Newton particle fluids.

    The solver advances :attr:`newton.State.particle_q` and
    :attr:`newton.State.particle_qd` using Warp kernels and a
    :class:`warp.HashGrid` neighbor search. It is intentionally small: it
    provides density, pressure, viscosity, gravity, external particle forces,
    and optional axis-aligned world bounds. It is meant as a first fluid solver
    and a reusable baseline for solver developers, not as a production-grade
    incompressible fluid method.

    Args:
        model: Model containing particles to simulate.
        smoothing_length: SPH kernel radius [m]. If ``None``, twice the
            model's maximum particle radius is used.
        rest_density: Fluid rest density [kg/m^3].
        gas_constant: Pressure stiffness for weak compressibility.
        viscosity: Kinematic viscosity coefficient.
        velocity_damping: Linear velocity damping coefficient [1/s].
        bounds_lower: Optional lower world bounds [m].
        bounds_upper: Optional upper world bounds [m].
        boundary_damping: Normal velocity restitution when clamped to bounds.
        max_velocity: Velocity clamp [m/s].
    """

    def __init__(
        self,
        model: Model,
        smoothing_length: float | None = None,
        rest_density: float = 1000.0,
        gas_constant: float = 2000.0,
        viscosity: float = 0.05,
        velocity_damping: float = 0.0,
        bounds_lower: Sequence[float] | wp.vec3 | None = None,
        bounds_upper: Sequence[float] | wp.vec3 | None = None,
        boundary_damping: float = 0.5,
        max_velocity: float | None = None,
    ):
        super().__init__(model=model)
        default_h = 2.0 * model.particle_max_radius if model.particle_max_radius > 0.0 else 0.1
        self.smoothing_length = float(default_h if smoothing_length is None else smoothing_length)
        self.rest_density = float(rest_density)
        self.gas_constant = float(gas_constant)
        self.viscosity = float(viscosity)
        self.velocity_damping = float(velocity_damping)
        self.bounds_lower = _vec3(bounds_lower, (-1.0e8, -1.0e8, -1.0e8))
        self.bounds_upper = _vec3(bounds_upper, (1.0e8, 1.0e8, 1.0e8))
        self.boundary_damping = float(boundary_damping)
        self.max_velocity = float(model.particle_max_velocity if max_velocity is None else max_velocity)
        self._capacity = 0
        self.particle_density: wp.array[wp.float32] | None = None
        self.particle_pressure: wp.array[wp.float32] | None = None
        self._ensure_particle_storage()

    def _ensure_particle_storage(self) -> None:
        model = self.model
        n = model.particle_count
        if n == self._capacity:
            return
        self._capacity = n
        self.particle_density = wp.empty(n, dtype=wp.float32, device=model.device)
        self.particle_pressure = wp.empty(n, dtype=wp.float32, device=model.device)
        if n:
            with wp.ScopedDevice(model.device):
                if model.particle_grid is None:
                    model.particle_grid = wp.HashGrid(128, 128, 128)
                model.particle_grid.reserve(n)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & ModelFlags.PARTICLE_PROPERTIES:
            self._ensure_particle_storage()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance particle fluid state by one time step.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Unused; accepted for solver API compatibility.
            contacts: Unused; accepted for solver API compatibility.
            dt: Time step [s].
        """
        model = self.model
        if model.particle_count == 0:
            return

        self._ensure_particle_storage()
        assert model.particle_grid is not None
        assert self.particle_density is not None
        assert self.particle_pressure is not None

        with wp.ScopedTimer("simulate", False):
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(state_in.particle_q, radius=self.smoothing_length)

            wp.launch(
                kernel=compute_sph_density_pressure,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_in.particle_q,
                    model.particle_mass,
                    model.particle_flags,
                    self.smoothing_length,
                    self.rest_density,
                    self.gas_constant,
                    self.particle_density,
                    self.particle_pressure,
                ],
                device=model.device,
            )

            wp.launch(
                kernel=integrate_sph_particles,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_mass,
                    model.particle_inv_mass,
                    model.particle_radius,
                    model.particle_flags,
                    model.particle_world,
                    model.gravity,
                    self.particle_density,
                    self.particle_pressure,
                    self.smoothing_length,
                    self.viscosity,
                    self.velocity_damping,
                    self.bounds_lower,
                    self.bounds_upper,
                    self.boundary_damping,
                    self.max_velocity,
                    dt,
                    state_out.particle_q,
                    state_out.particle_qd,
                ],
                device=model.device,
            )
