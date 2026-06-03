# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from newton.examples.cutting.cutting_common import (
    CutMaterial,
    KnifeProfile,
    SplitCuboidRenderMesh,
    compute_particle_cut_update,
    summarize_force_profile,
)


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

    def test_split_cuboid_render_mesh_opens_only_behind_knife(self):
        knife = KnifeProfile(start_x=-0.5, speed=1.0, center_y=0.0, process_width=0.1)
        mesh = SplitCuboidRenderMesh(
            block_lo=(-1.0, -0.5, 0.0),
            block_hi=(1.0, 0.5, 1.0),
            knife=knife,
            max_gap=0.2,
            segments=8,
        )

        closed_surface, closed_walls = mesh.build_points(time=-1.0)
        open_surface, open_walls = mesh.build_points(time=1.5)

        self.assertEqual(closed_surface.shape, open_surface.shape)
        self.assertEqual(closed_walls.shape, open_walls.shape)
        self.assertEqual(mesh.gap_at(0.75, time=0.0), 0.0)
        self.assertGreater(mesh.gap_at(-0.75, time=0.0), 0.0)
        self.assertAlmostEqual(mesh.gap_at(-0.75, time=1.5), 0.2)
        self.assertGreater(np.max(np.abs(open_walls[:, 1])), np.max(np.abs(closed_walls[:, 1])))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
