# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import inspect

import numpy as np
import warp as wp

import newton
from newton.examples.cutting.cutting_common import (
    AdaptiveCutSurfaceRemesher,
    CutMaterial,
    KnifeProfile,
    ShellCutSurfaceRenderer,
    SplitCuboidRenderMesh,
    TetMeshCutSurfaceRenderer,
    compute_particle_cut_update,
    encode_mp4,
    summarize_force_profile,
)
from newton.examples.cutting.example_cutting_xfem import Example as XFEMExample
from newton.examples.cutting.example_cutting_xfem import SCENARIOS, build_half_cylinder_tet_mesh
from newton.examples.cutting import generate_cutting_report_assets
from newton.solvers import SolverXFEMCut
from newton._src.viewer.gl.opengl import MeshGL, RendererGL
from newton._src.solvers.xfem_cut.kernels import apply_xfem_knife_kernel
from newton.viewer import ViewerNull


class TestCuttingCommon(unittest.TestCase):
    def test_knife_cut_weights_are_localized_to_process_zone(self):
        knife = KnifeProfile(
            start_x=-0.1,
            speed=1.0,
            center_y=0.0,
            center_z=0.0,
            half_width_y=0.4,
            half_width_z=0.2,
            process_width=0.05,
        )
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.049, 0.0, 0.0],
                [0.08, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.3],
            ],
            dtype=np.float32,
        )

        weights = knife.cut_weights(points, time=0.1)

        self.assertAlmostEqual(weights[0], 1.0, places=5)
        self.assertGreater(weights[1], 0.0)
        self.assertEqual(weights[2], 0.0)
        self.assertEqual(weights[3], 0.0)
        self.assertEqual(weights[4], 0.0)

    def test_knife_edge_spline_drives_process_zone(self):
        knife = KnifeProfile(
            start_x=0.0,
            speed=0.0,
            center_y=0.0,
            center_z=0.0,
            half_width_y=0.04,
            half_width_z=0.2,
            process_width=0.025,
            edge_control_points=((0.0, 0.0, -0.2), (0.045, 0.0, 0.0), (0.0, 0.0, 0.2)),
        )
        points = np.array(
            [
                [0.045, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.045, 0.07, 0.0],
                [0.12, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        weights = knife.cut_weights(points, time=0.0)

        self.assertAlmostEqual(float(weights[0]), 1.0, places=5)
        self.assertEqual(float(weights[1]), 0.0)
        self.assertEqual(float(weights[2]), 0.0)
        self.assertEqual(float(weights[3]), 0.0)

    def test_knife_blade_mesh_has_visible_rigid_faces(self):
        knife = KnifeProfile(start_x=0.1, speed=0.0, half_width_y=0.05, half_width_z=0.18)

        vertices, indices = knife.blade_mesh(time=0.0)

        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(indices.shape[1], 3)
        self.assertGreaterEqual(vertices.shape[0], 6)
        self.assertGreaterEqual(indices.shape[0], 8)
        self.assertLess(float(np.min(vertices[:, 0])), knife.x_at(0.0) - 0.08)
        self.assertGreater(float(np.max(vertices[:, 1])), knife.center_y + knife.half_width_y * 0.9)
        self.assertLess(float(np.min(vertices[:, 1])), knife.center_y - knife.half_width_y * 0.9)
        self.assertLessEqual(float(np.max(vertices[:, 2])), knife.center_z + knife.half_width_z + 1.0e-6)

    def test_particle_damage_prefers_spline_edge_over_old_front_plane(self):
        points = np.array(
            [
                [0.045, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        damage = np.zeros(2, dtype=np.float32)
        knife = KnifeProfile(
            start_x=0.0,
            speed=0.0,
            half_width_y=0.04,
            half_width_z=0.2,
            process_width=0.025,
            edge_control_points=((0.0, 0.0, -0.2), (0.045, 0.0, 0.0), (0.0, 0.0, 0.2)),
        )
        material = CutMaterial(fracture_energy=25.0, yield_stress=2.0e3, max_damage_rate=10.0)

        update = compute_particle_cut_update(points, damage, knife, material, dt=0.02, particle_volume=1.0e-6)

        self.assertGreater(update.damage[0], 0.0)
        self.assertEqual(update.damage[1], 0.0)
        self.assertEqual(update.active_count, 1)

    def test_viewer_log_mesh_accepts_opacity_keyword(self):
        points = wp.array(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            dtype=wp.vec3,
        )
        indices = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32)
        viewer = ViewerNull(num_frames=1)

        viewer.log_mesh("transparent_triangle", points, indices, opacity=0.35)

    def test_report_generator_uses_viewergl_frame_capture(self):
        source = inspect.getsource(generate_cutting_report_assets._run_case)

        self.assertIn("capture_viewer_frame", source)
        self.assertNotIn("_render_frame(", source)
        self.assertIn("frame_{len(frames) - 1}.png", source)

    def test_report_generator_reuses_viewergl_across_cases(self):
        source = inspect.getsource(generate_cutting_report_assets.main)

        self.assertIn("shared_viewer", source)
        self.assertIn("shared_viewer.close()", source)

    def test_report_generator_defaults_to_six_second_viewergl_videos(self):
        parser = generate_cutting_report_assets.create_parser()
        args = parser.parse_args([])

        self.assertEqual(args.frames, 360)
        self.assertEqual(args.video_fps, 60.0)

    def test_report_generator_can_write_complexity_benchmarks(self):
        source = inspect.getsource(generate_cutting_report_assets)

        self.assertIn("--benchmark-sweep", source)
        self.assertIn("benchmark_results.json", source)
        self.assertIn("adaptive_remesh_profile.png", source)
        self.assertIn("particle_count", source)
        self.assertIn("tet_count", source)
        self.assertIn("tri_count", source)

    def test_paper_tearing_scenario_is_wide_cloth_sheet(self):
        cfg = SCENARIOS["paper_tearing"]

        self.assertEqual(cfg.geometry, "cloth_grid")
        self.assertGreater(cfg.dim_y * cfg.cell_y, 0.55)
        self.assertEqual(cfg.dim_z, 0)
        self.assertEqual(cfg.saw_amplitude_z, 0.0)
        self.assertEqual(cfg.saw_frequency_hz, 0.0)
        self.assertGreaterEqual(cfg.k_damp, 3.0e-2)

    def test_hanging_cloth_wind_is_sideways_for_y_up_scene(self):
        cfg = SCENARIOS["hanging_cloth_cutoff"]

        self.assertEqual(cfg.up_axis, newton.Axis.Y)
        self.assertAlmostEqual(float(cfg.wind_direction[1]), 0.0)
        self.assertGreater(np.linalg.norm(np.asarray(cfg.wind_direction, dtype=np.float32)), 0.0)
        self.assertTrue(cfg.render_seam_edges)
        self.assertLessEqual(cfg.max_visual_gap, 0.025)

    def test_curved_cloth_spline_cut_scenario_is_large_nonstraight_sheet(self):
        cfg = SCENARIOS["curved_cloth_spline_cut"]

        self.assertEqual(cfg.geometry, "cloth_grid")
        self.assertGreaterEqual(cfg.dim_x * cfg.dim_y, 1500)
        self.assertGreater(cfg.cut_path_amplitude_y, 0.05)
        self.assertGreater(cfg.cut_path_wavelength_x, 0.1)
        self.assertTrue(cfg.render_seam_edges)
        self.assertFalse(cfg.visual_topology_only)
        self.assertGreater(cfg.force_scale, 0.0)
        self.assertGreater(cfg.friction_mu, 0.0)
        self.assertGreater(cfg.separation_speed, 0.0)

    def test_knife_blade_mesh_yaws_with_curved_cut_path(self):
        knife = KnifeProfile(
            start_x=-0.4,
            speed=0.0,
            center_y=-0.02,
            center_z=0.0,
            half_width_y=0.03,
            half_width_z=0.1,
            cut_path_amplitude_y=0.12,
            cut_path_wavelength_x=0.5,
            cut_path_phase=0.25,
            cut_path_origin_x=-0.4,
        )

        vertices, _indices = knife.blade_mesh(time=0.0)
        tangent = knife.path_tangent_at_x(knife.x_at(0.0))
        spine_direction = vertices[2] - vertices[0]
        spine_direction = spine_direction / np.linalg.norm(spine_direction)

        self.assertGreater(abs(float(tangent[1])), 0.1)
        self.assertGreater(float(np.dot(spine_direction, -tangent)), 0.97)

    def test_xfem_shell_particle_area_uses_triangle_area(self):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, -0.5, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=0.1,
            tri_ke=1.0,
            tri_ka=1.0,
            tri_kd=0.0,
        )
        model = builder.finalize()

        solver = SolverXFEMCut(model)

        self.assertAlmostEqual(solver.particle_area, 0.25, places=5)

    def test_paper_tearing_cloth_stays_bounded_until_knife_engages(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "test_cutting_examples",
                "--viewer",
                "null",
                "--quiet",
                "--scenario",
                "paper_tearing",
                "--num-frames",
                "1",
                "--no-render-split-mesh",
            ]
            viewer, args = newton.examples.init(XFEMExample.create_parser())
            example = XFEMExample(viewer, args)
            for _ in range(24):
                example.step()
            points = example.state_0.particle_q.numpy()
            viewer.close()
        finally:
            sys.argv = old_argv

        self.assertLess(float(np.max(np.linalg.norm(points, axis=1))), 2.0)
        self.assertGreater(max(example.force_history.active_counts), 0.0)

    def test_xfem_cloth_cut_disables_constraints_online(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "test_cutting_examples",
                "--viewer",
                "null",
                "--quiet",
                "--scenario",
                "paper_tearing",
                "--num-frames",
                "1",
                "--substeps",
                "4",
                "--iterations",
                "4",
                "--no-render-split-mesh",
            ]
            viewer, args = newton.examples.init(XFEMExample.create_parser())
            example = XFEMExample(viewer, args)
            self.assertEqual(int(np.sum(example.solver.spring_cut_state.numpy())), 0)
            for _ in range(45):
                example.step()
            spring_cut = example.solver.spring_cut_state.numpy()
            edge_cut = example.solver.edge_cut_state.numpy()
            tri_cut = example.solver.tri_cut_state.numpy()
            spring_stiffness = example.model.spring_stiffness.numpy()
            edge_stiffness = example.model.edge_bending_properties.numpy()[:, 0]
            viewer.close()
        finally:
            sys.argv = old_argv

        self.assertGreater(int(np.sum(spring_cut)), 0)
        self.assertGreater(int(np.sum(edge_cut)), 0)
        self.assertGreater(int(np.sum(tri_cut)), 0)
        self.assertTrue(np.all(spring_stiffness[spring_cut > 0] == 0.0))
        self.assertTrue(np.all(edge_stiffness[edge_cut > 0] == 0.0))

    def test_hanging_cloth_cutoff_scenario_is_top_fixed_with_wind(self):
        cfg = SCENARIOS["hanging_cloth_cutoff"]

        self.assertEqual(cfg.geometry, "cloth_grid")
        self.assertTrue(cfg.fix_top)
        self.assertFalse(cfg.fix_left)
        self.assertEqual(cfg.up_axis, newton.Axis.Y)
        self.assertLess(cfg.gravity[1], 0.0)
        self.assertGreater(cfg.wind_strength, 0.0)

    def test_hanging_cloth_cutoff_remeshes_online(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "test_cutting_examples",
                "--viewer",
                "null",
                "--quiet",
                "--scenario",
                "hanging_cloth_cutoff",
                "--num-frames",
                "1",
                "--substeps",
                "4",
                "--iterations",
                "4",
                "--no-render-split-mesh",
            ]
            viewer, args = newton.examples.init(XFEMExample.create_parser())
            example = XFEMExample(viewer, args)
            initial_points = example.state_0.particle_q.numpy().copy()
            for _ in range(160):
                example.step()
            spring_cut = example.solver.spring_cut_state.numpy()
            edge_cut = example.solver.edge_cut_state.numpy()
            tri_cut = example.solver.tri_cut_state.numpy()
            points = example.state_0.particle_q.numpy()
            top_row = np.isclose(initial_points[:, 1], np.max(initial_points[:, 1]))
            lower_panel = initial_points[:, 1] < example.scenario.knife_center_y
            viewer.close()
        finally:
            sys.argv = old_argv

        self.assertEqual(example.model.up_axis, newton.Axis.Y)
        self.assertGreater(int(np.sum(spring_cut)), 0)
        self.assertGreater(int(np.sum(edge_cut)), 0)
        self.assertGreater(int(np.sum(tri_cut)), 0)
        np.testing.assert_allclose(points[top_row], initial_points[top_row], atol=1.0e-6)
        self.assertLess(float(np.mean(points[lower_panel, 1] - initial_points[lower_panel, 1])), -0.01)

    def test_curved_cloth_spline_cut_remeshes_online(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "test_cutting_examples",
                "--viewer",
                "null",
                "--quiet",
                "--scenario",
                "curved_cloth_spline_cut",
                "--num-frames",
                "1",
                "--substeps",
                "4",
                "--iterations",
                "4",
                "--no-render-split-mesh",
            ]
            viewer, args = newton.examples.init(XFEMExample.create_parser())
            example = XFEMExample(viewer, args)
            initial_points = example.state_0.particle_q.numpy().copy()
            for _ in range(70):
                example.step()
            spring_cut = example.solver.spring_cut_state.numpy()
            edge_cut = example.solver.edge_cut_state.numpy()
            tri_cut = example.solver.tri_cut_state.numpy()
            points = example.state_0.particle_q.numpy()
            solver_center_y = example.solver.knife_center_y
            max_force = max(example.force_history.forces)
            viewer.close()
        finally:
            sys.argv = old_argv

        self.assertGreater(int(np.sum(spring_cut)), 0)
        self.assertGreater(int(np.sum(edge_cut)), 0)
        self.assertGreater(int(np.sum(tri_cut)), 0)
        self.assertAlmostEqual(float(solver_center_y), SCENARIOS["curved_cloth_spline_cut"].knife_center_y)
        self.assertGreater(float(max_force), 0.0)
        self.assertGreater(float(np.max(np.linalg.norm(points - initial_points, axis=1))), 1.0e-4)
        self.assertLess(float(np.max(np.ptp(points, axis=0))), 1.6)

    def test_viewergl_sets_pyglet_headless_before_shader_import(self):
        source = inspect.getsource(RendererGL.__init__)

        self.assertIn('pyglet.options["headless"] = bool(headless)', source)
        self.assertLess(
            source.index('pyglet.options["headless"] = bool(headless)'),
            source.index("from pyglet.graphics.shader import Shader"),
        )

    def test_mp4_encoder_preserves_viewergl_frame_size(self):
        source = inspect.getsource(encode_mp4)

        self.assertIn("macro_block_size=1)", source)
        self.assertNotIn("macro_block_size=16", source)

    def test_viewergl_mesh_updates_index_buffer_each_frame(self):
        source = inspect.getsource(MeshGL.update)

        self.assertNotIn("only update indices the first time", source)
        self.assertIn("host_indices = self.indices.numpy()", source)

    def test_xfem_knife_kernel_has_direct_saw_friction_drag(self):
        source = inspect.getsource(apply_xfem_knife_kernel)

        self.assertIn("friction_drag_velocity", source)

    def test_particle_damage_monotonic_and_force_scales_with_toughness(self):
        points = np.array(
            [
                [0.0, -0.05, 0.0],
                [0.01, 0.05, 0.0],
                [0.25, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        damage = np.array([0.0, 0.5, 0.2], dtype=np.float32)
        knife = KnifeProfile(start_x=0.0, speed=0.0, half_width_y=0.2, half_width_z=0.2, process_width=0.05)
        soft = CutMaterial(fracture_energy=25.0, yield_stress=2.0e3, max_damage_rate=10.0)
        tough = CutMaterial(fracture_energy=100.0, yield_stress=2.0e3, max_damage_rate=10.0)

        soft_update = compute_particle_cut_update(points, damage, knife, soft, dt=0.02, particle_volume=1.0e-6)
        tough_update = compute_particle_cut_update(points, damage, knife, tough, dt=0.02, particle_volume=1.0e-6)

        self.assertTrue(np.all(soft_update.damage >= damage))
        self.assertEqual(soft_update.active_count, 2)
        self.assertEqual(soft_update.damage[2], damage[2])
        self.assertGreater(soft_update.force, 0.0)
        self.assertGreater(tough_update.force, soft_update.force)

    def test_force_profile_summary(self):
        summary = summarize_force_profile(
            times=np.array([0.0, 0.1, 0.2, 0.3]),
            forces=np.array([0.0, 2.0, 4.0, 0.0]),
            damage=np.array([0.0, 0.2, 0.5, 0.75]),
        )

        self.assertAlmostEqual(summary["peak_force_n"], 4.0)
        self.assertAlmostEqual(summary["mean_force_n"], 1.5)
        self.assertAlmostEqual(summary["force_impulse_ns"], 0.6)
        self.assertAlmostEqual(summary["final_mean_damage"], 0.75)

    def test_split_cuboid_render_mesh_opens_visual_seam_without_shrinking_volume(self):
        knife = KnifeProfile(start_x=-0.5, speed=1.0, center_y=0.0, process_width=0.1)
        mesh = SplitCuboidRenderMesh(
            block_lo=(-1.0, -0.5, 0.0),
            block_hi=(1.0, 0.5, 1.0),
            knife=knife,
            max_gap=0.2,
            segments=8,
        )

        surface, walls = mesh.build_points(time=1.5)

        self.assertEqual(surface.shape[1], 3)
        self.assertEqual(walls.shape[1], 3)
        self.assertAlmostEqual(float(np.min(surface[:, 1])), -0.5)
        self.assertAlmostEqual(float(np.max(surface[:, 1])), 0.5)
        self.assertLess(float(np.min(walls[:, 1])), -0.15)
        self.assertGreater(float(np.max(walls[:, 1])), 0.15)
        self.assertGreater(mesh.gap_at(-0.75, time=1.5), 0.0)
        self.assertEqual(mesh.gap_at(1.25, time=1.5), 0.0)

    def test_split_cuboid_render_mesh_follows_particle_motion(self):
        knife = KnifeProfile(start_x=-0.5, speed=0.0, center_y=0.0, process_width=0.1)
        mesh = SplitCuboidRenderMesh(
            block_lo=(-1.0, -0.5, 0.0),
            block_hi=(1.0, 0.5, 1.0),
            knife=knife,
            max_gap=0.2,
            segments=4,
        )
        xs = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
        ys = np.linspace(-0.5, 0.5, 3, dtype=np.float32)
        zs = np.linspace(0.0, 1.0, 3, dtype=np.float32)
        rest_particles = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
        translation = np.array([0.12, -0.04, 0.07], dtype=np.float32)
        moved_particles = rest_particles + translation

        static_surface, static_walls = mesh.build_points(time=1.0)
        moved_surface, moved_walls = mesh.build_points(
            time=1.0,
            rest_particle_points=rest_particles,
            particle_points=moved_particles,
        )

        np.testing.assert_allclose(
            moved_surface - static_surface, np.broadcast_to(translation, moved_surface.shape), atol=1.0e-5
        )
        np.testing.assert_allclose(
            moved_walls - static_walls, np.broadcast_to(translation, moved_walls.shape), atol=1.0e-5
        )

    def test_split_cuboid_zero_kerf_vertices_follow_their_own_side(self):
        knife = KnifeProfile(start_x=-0.5, speed=0.0, center_y=0.0, process_width=0.08)
        mesh = SplitCuboidRenderMesh(
            block_lo=(-0.5, -0.25, 0.0),
            block_hi=(0.5, 0.25, 0.4),
            knife=knife,
            segments=2,
        )
        rest_particles = np.array(
            [
                [-0.25, -0.25, 0.0],
                [-0.25, -0.25, 0.4],
                [-0.25, 0.25, 0.0],
                [-0.25, 0.25, 0.4],
                [0.25, -0.25, 0.0],
                [0.25, -0.25, 0.4],
                [0.25, 0.25, 0.0],
                [0.25, 0.25, 0.4],
            ],
            dtype=np.float32,
        )
        negative_delta = np.array([0.0, -0.04, -0.08], dtype=np.float32)
        positive_delta = np.array([0.0, 0.04, 0.08], dtype=np.float32)
        moved_particles = rest_particles.copy()
        moved_particles[rest_particles[:, 1] < 0.0] += negative_delta
        moved_particles[rest_particles[:, 1] > 0.0] += positive_delta

        static_surface, _static_walls = mesh.build_points(time=1.0)
        moved_surface, _moved_walls = mesh.build_points(
            time=1.0,
            rest_particle_points=rest_particles,
            particle_points=moved_particles,
        )

        negative_side_first_quad_delta = moved_surface[:4] - static_surface[:4]
        np.testing.assert_allclose(
            negative_side_first_quad_delta,
            np.broadcast_to(negative_delta, negative_side_first_quad_delta.shape),
            atol=2.0e-3,
        )

    def test_adaptive_cut_remesher_refines_near_knife(self):
        knife = KnifeProfile(start_x=-0.25, speed=0.5, center_y=0.0, process_width=0.08)
        remesher = AdaptiveCutSurfaceRemesher(
            block_lo=(-0.5, -0.25, 0.0),
            block_hi=(0.5, 0.25, 0.4),
            knife=knife,
            base_segments=8,
            refine_factor=4,
            refine_band=0.12,
            height_segments=3,
        )

        stats = remesher.update(wp.get_device(), time=0.5)

        self.assertGreater(stats.active_x_segments, remesher.base_segments)
        self.assertGreater(stats.surface_triangle_count, 0)
        self.assertGreater(stats.wall_triangle_count, 0)
        self.assertLess(stats.min_active_dx, stats.coarse_dx)

    def test_adaptive_cut_remesher_follows_particle_motion(self):
        knife = KnifeProfile(start_x=-0.5, speed=0.0, center_y=0.0, process_width=0.08)
        remesher = AdaptiveCutSurfaceRemesher(
            block_lo=(-0.5, -0.25, 0.0),
            block_hi=(0.5, 0.25, 0.4),
            knife=knife,
            base_segments=4,
            refine_factor=2,
            refine_band=0.1,
            height_segments=2,
        )
        xs = np.linspace(-0.5, 0.5, 4, dtype=np.float32)
        ys = np.linspace(-0.25, 0.25, 3, dtype=np.float32)
        zs = np.linspace(0.0, 0.4, 3, dtype=np.float32)
        rest_particles = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
        translation = np.array([0.08, -0.03, 0.05], dtype=np.float32)
        moved_particles = rest_particles + translation

        static_stats = remesher.update(wp.get_device(), time=1.5)
        static_surface = remesher.surface_points_wp.numpy()[: static_stats.surface_vertex_count].copy()
        moved_stats = remesher.update(
            wp.get_device(),
            time=1.5,
            rest_particle_points=rest_particles,
            particle_points=moved_particles,
        )
        moved_surface = remesher.surface_points_wp.numpy()[: moved_stats.surface_vertex_count]

        self.assertEqual(static_stats.surface_vertex_count, moved_stats.surface_vertex_count)
        np.testing.assert_allclose(
            moved_surface - static_surface, np.broadcast_to(translation, moved_surface.shape), atol=1.0e-5
        )

    def test_adaptive_cut_remesher_zero_kerf_vertices_follow_their_own_side(self):
        knife = KnifeProfile(start_x=-0.5, speed=0.0, center_y=0.0, process_width=0.08)
        remesher = AdaptiveCutSurfaceRemesher(
            block_lo=(-0.5, -0.25, 0.0),
            block_hi=(0.5, 0.25, 0.4),
            knife=knife,
            base_segments=2,
            refine_factor=1,
            refine_band=0.1,
            height_segments=2,
        )
        rest_particles = np.array(
            [
                [-0.25, -0.25, 0.0],
                [-0.25, -0.25, 0.4],
                [-0.25, 0.25, 0.0],
                [-0.25, 0.25, 0.4],
                [0.25, -0.25, 0.0],
                [0.25, -0.25, 0.4],
                [0.25, 0.25, 0.0],
                [0.25, 0.25, 0.4],
            ],
            dtype=np.float32,
        )
        negative_delta = np.array([0.0, -0.04, -0.08], dtype=np.float32)
        positive_delta = np.array([0.0, 0.04, 0.08], dtype=np.float32)
        moved_particles = rest_particles.copy()
        moved_particles[rest_particles[:, 1] < 0.0] += negative_delta
        moved_particles[rest_particles[:, 1] > 0.0] += positive_delta

        static_stats = remesher.update(wp.get_device(), time=1.0)
        static_surface = remesher.surface_points_wp.numpy()[: static_stats.surface_vertex_count].copy()
        moved_stats = remesher.update(
            wp.get_device(),
            time=1.0,
            rest_particle_points=rest_particles,
            particle_points=moved_particles,
        )
        moved_surface = remesher.surface_points_wp.numpy()[: moved_stats.surface_vertex_count]

        negative_side_first_quad_delta = moved_surface[:4] - static_surface[:4]
        np.testing.assert_allclose(
            negative_side_first_quad_delta,
            np.broadcast_to(negative_delta, negative_side_first_quad_delta.shape),
            atol=2.0e-3,
        )

    def test_adaptive_cut_remesher_opens_visual_seam_without_shrinking_volume(self):
        knife = KnifeProfile(start_x=-0.5, speed=1.0, center_y=0.0, process_width=0.08)
        remesher = AdaptiveCutSurfaceRemesher(
            block_lo=(-0.5, -0.25, 0.0),
            block_hi=(0.5, 0.25, 0.4),
            knife=knife,
            max_gap=0.2,
            base_segments=8,
            refine_factor=2,
            refine_band=0.1,
            height_segments=3,
        )

        stats = remesher.update(wp.get_device(), time=0.8)
        walls = remesher.wall_points_wp.numpy()[: stats.wall_vertex_count]
        surface = remesher.surface_points_wp.numpy()[: stats.surface_vertex_count]

        self.assertLess(float(np.min(walls[:, 1])), -0.15)
        self.assertGreater(float(np.max(walls[:, 1])), 0.15)
        self.assertAlmostEqual(float(np.min(surface[:, 1])), -0.25, places=6)
        self.assertAlmostEqual(float(np.max(surface[:, 1])), 0.25, places=6)

    def test_half_cylinder_tet_mesh_is_curved_and_oriented(self):
        vertices, tets = build_half_cylinder_tet_mesh(length=0.45, radius=0.12, target_edge=0.055)

        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(tets.shape[1], 4)
        self.assertGreater(vertices.shape[0], 120)
        self.assertGreater(tets.shape[0], 250)
        self.assertAlmostEqual(float(np.min(vertices[:, 2])), 0.0, places=5)
        self.assertGreater(float(np.max(vertices[:, 2])), 0.11)
        self.assertGreater(np.unique(np.round(vertices[:, 1], 3)).size, 8)

        a = vertices[tets[:, 0]]
        b = vertices[tets[:, 1]]
        c = vertices[tets[:, 2]]
        d = vertices[tets[:, 3]]
        signed_volumes = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
        self.assertTrue(np.all(signed_volumes > 0.0))

    def test_tet_cut_surface_renderer_splits_surface_triangles_in_cut_wake(self):
        rest_points = np.array(
            [
                [-0.2, -0.1, 0.0],
                [-0.2, 0.1, 0.0],
                [-0.2, -0.1, 0.2],
                [-0.1, 0.0, 0.1],
            ],
            dtype=np.float32,
        )
        tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
        surface = np.array([[0, 1, 2]], dtype=np.int32)
        knife = KnifeProfile(start_x=0.0, speed=0.0, center_y=0.0, center_z=0.1, half_width_z=0.25)
        renderer = TetMeshCutSurfaceRenderer(rest_points, tets, surface, knife, nominal_edge_length=0.1)

        stats = renderer.update(rest_points, time=0.0, front_x=0.1, center_z=0.1)
        rendered_points = renderer.surface_points_np[: stats.surface_vertex_count]
        rendered_indices = renderer.surface_indices_np[: stats.surface_triangle_count * 3].reshape(-1, 3)

        self.assertGreater(stats.surface_triangle_count, 1)
        for tri in rendered_points[rendered_indices]:
            self.assertFalse(np.min(tri[:, 1]) < -1.0e-6 and np.max(tri[:, 1]) > 1.0e-6)

    def test_tet_cut_surface_renderer_opens_plane_touching_surface_triangles(self):
        rest_points = np.array(
            [
                [-0.2, 0.0, 0.0],
                [-0.2, 0.1, 0.0],
                [-0.2, 0.0, 0.2],
                [-0.2, -0.1, 0.0],
                [-0.1, 0.0, 0.1],
            ],
            dtype=np.float32,
        )
        tets = np.array([[0, 1, 2, 4], [0, 2, 3, 4]], dtype=np.int32)
        surface = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        knife = KnifeProfile(
            start_x=0.0,
            speed=0.0,
            center_y=0.0,
            center_z=0.1,
            half_width_z=0.25,
            process_width=0.1,
        )
        renderer = TetMeshCutSurfaceRenderer(
            rest_points,
            tets,
            surface,
            knife,
            nominal_edge_length=0.1,
            max_visual_gap=0.04,
        )

        stats = renderer.update(rest_points, time=0.0, front_x=0.1, center_z=0.1)
        rendered_points = renderer.surface_points_np[: stats.surface_vertex_count]
        rendered_indices = renderer.surface_indices_np[: stats.surface_triangle_count * 3].reshape(-1, 3)

        self.assertEqual(stats.surface_triangle_count, 2)
        for tri in rendered_points[rendered_indices]:
            self.assertTrue(np.all(tri[:, 1] > 0.01) or np.all(tri[:, 1] < -0.01))

    def test_shell_cut_surface_renderer_splits_triangles_without_collapsing_sheet(self):
        rest_points = np.array(
            [
                [-0.4, -0.35, 0.0],
                [0.0, -0.35, 0.0],
                [0.4, -0.35, 0.0],
                [-0.4, 0.35, 0.0],
                [0.0, 0.35, 0.0],
                [0.4, 0.35, 0.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]], dtype=np.int32)
        knife = KnifeProfile(start_x=-0.3, speed=0.0, center_y=0.0, center_z=0.0, half_width_z=0.04)
        renderer = ShellCutSurfaceRenderer(rest_points, triangles, knife, nominal_edge_length=0.2, max_visual_gap=0.08)

        stats = renderer.update(rest_points, time=0.0, front_x=0.3, center_z=0.0)
        rendered_points = renderer.surface_points_np[: stats.surface_vertex_count]
        rendered_indices = renderer.surface_indices_np[: stats.surface_triangle_count * 3].reshape(-1, 3)

        self.assertGreater(stats.surface_triangle_count, triangles.shape[0])
        self.assertLess(float(np.min(rendered_points[:, 1])), -0.30)
        self.assertGreater(float(np.max(rendered_points[:, 1])), 0.30)
        for tri in rendered_points[rendered_indices]:
            self.assertFalse(np.min(tri[:, 1]) < -1.0e-5 and np.max(tri[:, 1]) > 1.0e-5)

    def test_shell_cut_surface_renderer_preserves_area_when_visual_gap_is_disabled(self):
        rest_points = np.array(
            [
                [-0.4, -0.35, 0.0],
                [0.0, -0.35, 0.0],
                [0.4, -0.35, 0.0],
                [-0.4, 0.35, 0.0],
                [0.0, 0.35, 0.0],
                [0.4, 0.35, 0.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]], dtype=np.int32)
        knife = KnifeProfile(start_x=-0.3, speed=0.0, center_y=0.0, center_z=0.0, half_width_z=0.04)
        renderer = ShellCutSurfaceRenderer(
            rest_points,
            triangles,
            knife,
            nominal_edge_length=0.2,
            max_visual_gap=0.0,
            render_seam_edges=True,
        )

        stats = renderer.update(
            rest_points,
            time=0.0,
            front_x=0.3,
            center_z=0.0,
            triangle_cut_state=np.ones(triangles.shape[0], dtype=np.int32),
        )
        rendered_points = renderer.surface_points_np[: stats.surface_vertex_count]
        rendered_indices = renderer.surface_indices_np[: stats.surface_triangle_count * 3].reshape(-1, 3)
        original_area = 0.5 * np.sum(
            np.linalg.norm(
                np.cross(
                    rest_points[triangles][:, 1] - rest_points[triangles][:, 0],
                    rest_points[triangles][:, 2] - rest_points[triangles][:, 0],
                ),
                axis=1,
            )
        )
        rendered_area = 0.5 * np.sum(
            np.linalg.norm(
                np.cross(
                    rendered_points[rendered_indices][:, 1] - rendered_points[rendered_indices][:, 0],
                    rendered_points[rendered_indices][:, 2] - rendered_points[rendered_indices][:, 0],
                ),
                axis=1,
            )
        )

        self.assertGreater(stats.surface_triangle_count, triangles.shape[0])
        self.assertGreater(stats.wall_triangle_count, 0)
        self.assertAlmostEqual(float(rendered_area), float(original_area), places=5)

    def test_shell_cut_surface_renderer_uses_online_cut_state(self):
        rest_points = np.array(
            [
                [-0.2, -0.1, 0.0],
                [0.2, -0.1, 0.0],
                [-0.2, 0.1, 0.0],
                [0.2, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        knife = KnifeProfile(start_x=0.0, speed=0.0, center_y=0.0, center_z=0.0, half_width_z=0.04)
        renderer = ShellCutSurfaceRenderer(
            rest_points,
            triangles,
            knife,
            nominal_edge_length=0.2,
            max_visual_gap=0.08,
            render_seam_edges=False,
        )

        uncut = np.zeros(triangles.shape[0], dtype=np.int32)
        uncut_stats = renderer.update(rest_points, time=0.0, front_x=0.3, center_z=0.0, triangle_cut_state=uncut)
        cut = np.ones(triangles.shape[0], dtype=np.int32)
        cut_stats = renderer.update(rest_points, time=0.0, front_x=0.3, center_z=0.0, triangle_cut_state=cut)
        rendered_points = renderer.surface_points_np[: cut_stats.surface_vertex_count]
        rendered_indices = renderer.surface_indices_np[: cut_stats.surface_triangle_count * 3].reshape(-1, 3)

        self.assertEqual(uncut_stats.surface_triangle_count, triangles.shape[0])
        self.assertGreater(cut_stats.surface_triangle_count, uncut_stats.surface_triangle_count)
        self.assertEqual(cut_stats.wall_triangle_count, 0)
        for tri in rendered_points[rendered_indices]:
            self.assertFalse(np.min(tri[:, 1]) < -1.0e-5 and np.max(tri[:, 1]) > 1.0e-5)

    def test_shell_cut_surface_renderer_seam_follows_cut_side_motion(self):
        rest_points = np.array(
            [
                [-0.2, -0.1, 0.0],
                [0.2, -0.1, 0.0],
                [-0.2, 0.1, 0.0],
                [0.2, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        current_points = rest_points.copy()
        current_points[rest_points[:, 1] < 0.0, 1] -= 1.0
        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        knife = KnifeProfile(start_x=0.0, speed=0.0, center_y=0.0, center_z=0.0, half_width_z=0.04)
        renderer = ShellCutSurfaceRenderer(
            rest_points,
            triangles,
            knife,
            nominal_edge_length=0.2,
            max_visual_gap=0.0,
            render_seam_edges=False,
        )

        stats = renderer.update(
            current_points,
            time=0.0,
            front_x=0.3,
            center_z=0.0,
            triangle_cut_state=np.ones(triangles.shape[0], dtype=np.int32),
        )
        rendered_points = renderer.surface_points_np[: stats.surface_vertex_count]

        self.assertGreater(stats.surface_triangle_count, triangles.shape[0])
        self.assertFalse(np.any((rendered_points[:, 1] > -0.9) & (rendered_points[:, 1] < -0.1)))

    def test_tet_cut_surface_renderer_surface_seam_follows_cut_side_motion(self):
        rest_points = np.array(
            [
                [-0.2, -0.1, 0.0],
                [-0.2, 0.1, 0.0],
                [-0.2, -0.1, 0.2],
                [-0.1, 0.0, 0.1],
            ],
            dtype=np.float32,
        )
        current_points = rest_points.copy()
        current_points[rest_points[:, 1] < 0.0, 1] -= 1.0
        tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
        surface = np.array([[0, 1, 2]], dtype=np.int32)
        knife = KnifeProfile(start_x=0.0, speed=0.0, center_y=0.0, center_z=0.1, half_width_z=0.25)
        renderer = TetMeshCutSurfaceRenderer(
            rest_points,
            tets,
            surface,
            knife,
            nominal_edge_length=0.1,
            max_visual_gap=0.0,
        )

        stats = renderer.update(current_points, time=0.0, front_x=0.1, center_z=0.1)
        rendered_points = renderer.surface_points_np[: stats.surface_vertex_count]

        self.assertGreater(stats.surface_triangle_count, 1)
        self.assertFalse(np.any((rendered_points[:, 1] > -0.9) & (rendered_points[:, 1] < -0.1)))

    def test_tet_cut_surface_renderer_wall_seam_follows_cut_side_motion(self):
        rest_points = np.array(
            [
                [-0.2, -0.1, 0.0],
                [-0.2, 0.1, 0.0],
                [-0.2, -0.1, 0.2],
                [-0.1, 0.1, 0.2],
            ],
            dtype=np.float32,
        )
        current_points = rest_points.copy()
        current_points[rest_points[:, 1] < 0.0, 1] -= 1.0
        tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
        surface = np.array([[0, 1, 2]], dtype=np.int32)
        knife = KnifeProfile(start_x=0.0, speed=0.0, center_y=0.0, center_z=0.1, half_width_z=0.25)
        renderer = TetMeshCutSurfaceRenderer(
            rest_points,
            tets,
            surface,
            knife,
            nominal_edge_length=0.1,
            max_visual_gap=0.0,
        )

        stats = renderer.update(current_points, time=0.0, front_x=0.1, center_z=0.1)
        rendered_points = renderer.wall_points_np[: stats.wall_vertex_count]

        self.assertGreater(stats.wall_triangle_count, 0)
        self.assertFalse(np.any((rendered_points[:, 1] > -0.9) & (rendered_points[:, 1] < -0.1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
