# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import warp as wp

from ...sim import Contacts, Model, State
from .kernels import (
    add_scaled_spatial_vectors,
    add_spatial_vectors,
    eval_body_contact_impulses,
    predict_body_contact_velocities,
)


class RigidContactResolver:
    """Shared rigid-contact backend for force- and impulse-based solvers."""

    def __init__(
        self,
        model: Model,
        method: str = "force",
        friction_smoothing: float = 1.0,
        impulse_iterations: int = 3,
        impulse_baumgarte: float = 0.25,
        impulse_penetration_slop: float = 1.0e-3,
        impulse_restitution_scale: float = 0.0,
        impulse_restitution_velocity_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.method = method
        self.friction_smoothing = friction_smoothing
        self.impulse_iterations = impulse_iterations
        self.impulse_baumgarte = impulse_baumgarte
        self.impulse_penetration_slop = impulse_penetration_slop
        self.impulse_restitution_scale = impulse_restitution_scale
        self.impulse_restitution_velocity_threshold = impulse_restitution_velocity_threshold

        self._validate_method()

        if model.body_count:
            self.body_qd_predicted = wp.empty_like(model.body_qd, requires_grad=False)
            self.body_qd_work_a = wp.empty_like(model.body_qd, requires_grad=False)
            self.body_qd_work_b = wp.empty_like(model.body_qd, requires_grad=False)
            self.body_qd_delta = wp.zeros_like(model.body_qd, requires_grad=False)
            self.body_impulses = wp.zeros_like(model.body_qd, requires_grad=False)

    def _validate_method(self) -> None:
        if self.method not in ("force", "impulse"):
            raise ValueError(f"Unknown rigid contact method '{self.method}'. Expected 'force' or 'impulse'.")

    def resolve(
        self,
        state: State,
        contacts: Contacts | None,
        dt: float,
        body_f_out: wp.array | None = None,
    ) -> None:
        if contacts is None or contacts.rigid_contact_max == 0 or self.model.body_count == 0:
            return

        if body_f_out is None:
            body_f_out = state.body_f

        if self.method == "force":
            eval_body_contact_forces = importlib.import_module(
                "newton._src.solvers.semi_implicit.kernels_contact"
            ).eval_body_contact_forces

            eval_body_contact_forces(
                self.model,
                state,
                contacts,
                friction_smoothing=self.friction_smoothing,
                force_in_world_frame=False,
                body_f_out=body_f_out,
            )
            return

        if dt <= 0.0:
            return

        wp.launch(
            kernel=predict_body_contact_velocities,
            dim=self.model.body_count,
            inputs=[
                state.body_q,
                state.body_qd,
                body_f_out,
                self.model.body_inertia,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                self.model.body_flags,
                self.model.body_world,
                self.model.gravity,
                dt,
            ],
            outputs=[self.body_qd_predicted],
            device=self.model.device,
        )

        self.body_qd_work_a.assign(self.body_qd_predicted)
        self.body_qd_delta.zero_()
        self.body_impulses.zero_()

        work_in = self.body_qd_work_a
        work_out = self.body_qd_work_b
        inv_dt = 1.0 / dt

        for _ in range(self.impulse_iterations):
            self.body_qd_delta.zero_()
            wp.launch(
                kernel=eval_body_contact_impulses,
                dim=contacts.rigid_contact_max,
                inputs=[
                    state.body_q,
                    work_in,
                    self.model.body_com,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                    self.model.body_flags,
                    self.model.shape_material_ka,
                    self.model.shape_material_mu,
                    self.model.shape_material_mu_torsional,
                    self.model.shape_material_mu_rolling,
                    self.model.shape_material_restitution,
                    self.model.shape_body,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    contacts.rigid_contact_friction,
                    self.impulse_baumgarte,
                    self.impulse_penetration_slop,
                    self.impulse_restitution_scale,
                    self.impulse_restitution_velocity_threshold,
                    dt,
                    inv_dt,
                ],
                outputs=[self.body_impulses, self.body_qd_delta, contacts.rigid_contact_force],
                device=self.model.device,
            )
            wp.launch(
                kernel=add_spatial_vectors,
                dim=self.model.body_count,
                inputs=[work_in, self.body_qd_delta],
                outputs=[work_out],
                device=self.model.device,
            )
            work_in, work_out = work_out, work_in

        wp.launch(
            kernel=add_scaled_spatial_vectors,
            dim=self.model.body_count,
            inputs=[body_f_out, self.body_impulses, inv_dt],
            outputs=[body_f_out],
            device=self.model.device,
        )
