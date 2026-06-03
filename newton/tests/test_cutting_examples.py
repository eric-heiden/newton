# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import inspect

import numpy as np
import warp as wp

from newton.examples.cutting.cutting_common import (
    AdaptiveCutSurfaceRemesher,
    CutMaterial,
    KnifeProfile,
    SplitCuboidRenderMesh,
    compute_particle_cut_update,
    encode_mp4,
    summarize_force_profile,
)
from newton.examples.cutting.example_cutting_xfem import build_half_cylinder_tet_mesh
from newton.examples.cutting import generate_cutting_report_assets
from newton._src.viewer.gl.opengl import RendererGL
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

    def test_report_generator_reuses_viewergl_across_cases(self):
        source = inspect.getsource(generate_cutting_report_assets.main)

        self.assertIn("shared_viewer", source)
        self.assertIn("shared_viewer.close()", source)

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

    def test_split_cuboid_render_mesh_preserves_zero_kerf_without_particle_motion(self):
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
        np.testing.assert_allclose(walls[:, 1], np.zeros(walls.shape[0]), atol=1.0e-6)
        self.assertEqual(mesh.gap_at(-0.75, time=1.5), 0.0)

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

    def test_adaptive_cut_remesher_preserves_zero_kerf_without_particle_motion(self):
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

        np.testing.assert_allclose(walls[:, 1], np.zeros(walls.shape[0]), atol=1.0e-6)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
