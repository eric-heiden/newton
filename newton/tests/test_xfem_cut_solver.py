# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


def _make_soft_block_model(device, *, dim_x=3, dim_y=2, dim_z=2):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(-0.12, -0.08, 0.02),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x,
        dim_y=dim_y,
        dim_z=dim_z,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.06,
        density=900.0,
        k_mu=2.5e4,
        k_lambda=4.0e4,
        k_damp=0.0,
        particle_radius=0.01,
    )
    builder.color()
    model = builder.finalize(device=device)
    model.gravity.zero_()
    return model


class TestXFEMCutSolver(unittest.TestCase):
    def test_solver_is_exported(self):
        self.assertTrue(hasattr(newton.solvers, "SolverXFEMCut"))

    def test_cut_classification_marks_straddling_tets_and_enriched_nodes(self):
        device = wp.get_device()
        model = _make_soft_block_model(device)
        solver = newton.solvers.SolverXFEMCut(model, iterations=2)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        solver.set_knife_state(
            front_x=0.02,
            center_y=0.0,
            center_z=0.08,
            half_width_y=0.08,
            half_width_z=0.18,
            process_width=0.08,
        )
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)

        tet_state = solver.tet_cut_state.numpy()
        cut_side = solver.particle_cut_side.numpy()

        self.assertGreater(np.count_nonzero(tet_state), 0)
        self.assertGreater(np.count_nonzero(cut_side), 0)
        self.assertGreater(np.count_nonzero(cut_side > 0.0), 0)
        self.assertGreater(np.count_nonzero(cut_side < 0.0), 0)

    def test_damage_and_force_accumulate_under_sawing_knife(self):
        device = wp.get_device()
        model = _make_soft_block_model(device)
        solver = newton.solvers.SolverXFEMCut(
            model,
            iterations=2,
            fracture_energy=120.0,
            yield_stress=1.6e4,
            max_damage_rate=24.0,
            knife_friction_mu=0.65,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        for i in range(5):
            solver.set_knife_state(
                front_x=-0.02 + 0.025 * i,
                center_y=0.0,
                center_z=0.08,
                half_width_y=0.08,
                half_width_z=0.18,
                process_width=0.08,
                knife_velocity=(0.6, 0.0, (-1.0) ** i * 0.35),
                knife_tangent=(0.0, 0.0, 1.0),
            )
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        particle_damage = solver.particle_damage.numpy()
        force_values = solver.force_accum.numpy()
        enrichment = solver.particle_enrichment_q.numpy()

        self.assertGreater(float(np.max(particle_damage)), 0.05)
        self.assertGreater(float(force_values[0]), 0.0)
        self.assertGreater(float(force_values[3]), 0.0)
        self.assertGreater(float(force_values[4]), 0.0)
        self.assertGreater(float(np.max(np.linalg.norm(enrichment, axis=1))), 0.0)

    def test_solver_accepts_rigid_spline_knife_edge(self):
        device = wp.get_device()
        model = _make_soft_block_model(device)
        solver = newton.solvers.SolverXFEMCut(model, iterations=2)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        solver.set_knife_state(
            front_x=0.0,
            center_y=0.0,
            center_z=0.08,
            half_width_y=0.08,
            half_width_z=0.18,
            process_width=0.045,
            edge_points=[
                (-0.02, 0.0, -0.10),
                (0.04, 0.0, 0.08),
                (-0.02, 0.0, 0.26),
            ],
        )
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)

        edge_points = solver.knife_edge_points.numpy()[: solver.knife_edge_point_count]
        force_values = solver.force_accum.numpy()

        np.testing.assert_allclose(edge_points[1], np.array([0.04, 0.0, 0.08], dtype=np.float32), atol=1.0e-6)
        self.assertGreater(float(force_values[0]), 0.0)

    def test_table_glue_keeps_bottom_vertices_near_rest_height(self):
        device = wp.get_device()
        model = _make_soft_block_model(device, dim_x=2, dim_y=2, dim_z=2)
        solver = newton.solvers.SolverXFEMCut(
            model,
            iterations=1,
            table_z=0.0,
            table_glue_depth=0.025,
            table_glue_strength=1.0,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        q = state_0.particle_q.numpy()
        bottom = q[:, 2] <= 0.021
        q[bottom, 2] -= 0.05
        state_0.particle_q.assign(q)
        solver.set_knife_state(front_x=-1.0, center_y=0.0, center_z=0.08)

        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)

        q_after = state_1.particle_q.numpy()
        rest = solver.rest_particle_q.numpy()
        np.testing.assert_allclose(q_after[bottom, 2], rest[bottom, 2], atol=1.0e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
