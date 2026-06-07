# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


class _LogFluidProbe(ViewerNull):
    """Captures fluid and point logging calls for ViewerBase particle routing."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.logged_fluid = None
        self.logged_points = None

    def log_fluid(
        self,
        name,
        points,
        radii=None,
        color=(0.10, 0.98, 0.92),
        deep_color=(0.0, 0.13, 0.58),
        color_gradient_strength=0.88,
        opacity=0.64,
        radius_scale=1.0,
        thickness_scale=1.8,
        smoothing_iterations=8,
        smoothing_radius=2.0,
        reflection_strength=0.14,
        refraction_strength=0.055,
        env_map_strength=0.52,
        env_reflection_lod=0.0,
        env_color_preserve=0.85,
        absorption_strength=1.55,
        depth_visualization_strength=0.55,
        caustic_strength=0.78,
        caustic_scale=155.0,
        floor_caustic_strength=0.65,
        foam_strength=0.12,
        foam_scale=55.0,
        hidden=False,
    ):
        self.logged_fluid = {
            "name": name,
            "points": points,
            "radii": radii,
            "color": color,
            "deep_color": deep_color,
            "color_gradient_strength": color_gradient_strength,
            "opacity": opacity,
            "radius_scale": radius_scale,
            "thickness_scale": thickness_scale,
            "smoothing_iterations": smoothing_iterations,
            "smoothing_radius": smoothing_radius,
            "reflection_strength": reflection_strength,
            "refraction_strength": refraction_strength,
            "env_map_strength": env_map_strength,
            "env_reflection_lod": env_reflection_lod,
            "env_color_preserve": env_color_preserve,
            "absorption_strength": absorption_strength,
            "depth_visualization_strength": depth_visualization_strength,
            "caustic_strength": caustic_strength,
            "caustic_scale": caustic_scale,
            "floor_caustic_strength": floor_caustic_strength,
            "foam_strength": foam_strength,
            "foam_scale": foam_scale,
            "hidden": hidden,
        }

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        self.logged_points = {"name": name, "points": points, "radii": radii, "hidden": hidden}


class TestViewerFluid(unittest.TestCase):
    @staticmethod
    def _build_model(flags_list):
        builder = newton.ModelBuilder()
        for i, flag in enumerate(flags_list):
            builder.add_particle(
                pos=(float(i), 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=1.0,
                radius=0.1,
                flags=flag,
            )
        return builder.finalize(device="cpu")

    def test_show_fluid_routes_active_particles_to_log_fluid(self):
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active, 0, active])
        state = model.state()
        viewer = _LogFluidProbe()

        viewer.set_model(model)
        viewer.show_fluid = True
        viewer.show_particles = False
        viewer._log_particles(state)

        self.assertIsNotNone(viewer.logged_fluid)
        self.assertEqual(viewer.logged_fluid["name"], "/model/fluid")
        self.assertFalse(viewer.logged_fluid["hidden"])
        self.assertEqual(viewer.logged_fluid["radius_scale"], viewer.fluid_radius_scale)
        self.assertEqual(viewer.logged_fluid["smoothing_radius"], viewer.fluid_smoothing_radius)
        self.assertEqual(viewer.logged_fluid["deep_color"], viewer.fluid_deep_color)
        self.assertEqual(viewer.logged_fluid["color_gradient_strength"], viewer.fluid_color_gradient_strength)
        self.assertEqual(viewer.logged_fluid["env_map_strength"], viewer.fluid_env_map_strength)
        self.assertEqual(viewer.logged_fluid["env_reflection_lod"], viewer.fluid_env_reflection_lod)
        self.assertEqual(viewer.logged_fluid["env_color_preserve"], viewer.fluid_env_color_preserve)
        self.assertEqual(viewer.logged_fluid["absorption_strength"], viewer.fluid_absorption_strength)
        self.assertEqual(
            viewer.logged_fluid["depth_visualization_strength"], viewer.fluid_depth_visualization_strength
        )
        self.assertEqual(viewer.logged_fluid["caustic_scale"], viewer.fluid_caustic_scale)
        self.assertEqual(viewer.logged_fluid["floor_caustic_strength"], viewer.fluid_floor_caustic_strength)
        self.assertEqual(viewer.logged_fluid["foam_strength"], viewer.fluid_foam_strength)
        self.assertEqual(viewer.logged_fluid["foam_scale"], viewer.fluid_foam_scale)
        np.testing.assert_allclose(viewer.logged_fluid["points"].numpy()[:, 0], [0.0, 2.0], atol=1.0e-6)
        self.assertIsNotNone(viewer.logged_points)
        self.assertEqual(viewer.logged_points["name"], "/model/particles")
        self.assertIsNone(viewer.logged_points["points"])
        self.assertTrue(viewer.logged_points["hidden"])

    def test_show_fluid_clears_when_all_particles_inactive(self):
        model = self._build_model([0, 0])
        state = model.state()
        viewer = _LogFluidProbe()

        viewer.set_model(model)
        viewer.show_fluid = True
        viewer.show_particles = False
        viewer._log_particles(state)

        self.assertIsNotNone(viewer.logged_fluid)
        self.assertIsNone(viewer.logged_fluid["points"])
        self.assertTrue(viewer.logged_fluid["hidden"])

    def test_switching_from_fluid_to_particles_hides_fluid_batch(self):
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active])
        state = model.state()
        viewer = _LogFluidProbe()

        viewer.set_model(model)
        viewer.show_fluid = True
        viewer._log_particles(state)
        self.assertFalse(viewer.logged_fluid["hidden"])

        viewer.logged_fluid = None
        viewer.logged_points = None
        viewer.show_fluid = False
        viewer.show_particles = True
        viewer._log_particles(state)

        self.assertIsNotNone(viewer.logged_fluid)
        self.assertIsNone(viewer.logged_fluid["points"])
        self.assertTrue(viewer.logged_fluid["hidden"])
        self.assertIsNotNone(viewer.logged_points)
        self.assertEqual(viewer.logged_points["name"], "/model/particles")
        self.assertFalse(viewer.logged_points["hidden"])

    def test_default_log_fluid_falls_back_to_points(self):
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active])
        state = model.state()
        viewer = _LogFluidProbe()

        ViewerNull.log_fluid(viewer, "fallback", state.particle_q, radii=0.2, hidden=False)

        self.assertIsNotNone(viewer.logged_points)
        self.assertEqual(viewer.logged_points["name"], "fallback")
        self.assertFalse(viewer.logged_points["hidden"])

    def test_default_fluid_material_uses_visible_tropical_env_refraction(self):
        viewer = _LogFluidProbe()

        self.assertGreaterEqual(viewer.fluid_color_gradient_strength, 0.75)
        self.assertGreaterEqual(viewer.fluid_env_map_strength, 0.40)
        self.assertGreaterEqual(viewer.fluid_refraction_strength, 0.04)
        self.assertGreaterEqual(viewer.fluid_reflection_strength, 0.10)
        self.assertGreaterEqual(viewer.fluid_absorption_strength, 1.20)
        self.assertLessEqual(viewer.fluid_env_reflection_lod, 0.5)
        self.assertGreaterEqual(viewer.fluid_env_color_preserve, 0.75)
        self.assertGreaterEqual(viewer.fluid_depth_visualization_strength, 0.35)
        self.assertGreaterEqual(viewer.fluid_floor_caustic_strength, 0.35)

    def test_interactive_tank_parser_exposes_shader_and_picking_controls(self):
        from newton.examples.fluid.example_fluid_sph_interactive_tank import Example

        args = Example.create_parser().parse_args([])

        self.assertEqual(args.render_mode, "fluid")
        self.assertGreater(args.box_count, 0)
        self.assertGreater(args.pick_stiffness, 0.0)
        self.assertGreater(args.fluid_env_map_strength, 0.8)
        self.assertLessEqual(args.fluid_env_reflection_lod, 0.5)
        self.assertGreater(args.fluid_floor_caustic_strength, 1.0)
        self.assertGreater(args.fluid_depth_visualization_strength, 0.5)

    def test_interactive_tank_rollout_keeps_boxes_submerged_and_stable(self):
        from newton.examples.fluid.example_fluid_sph_interactive_tank import Example

        args = Example.create_parser().parse_args(["--viewer", "null", "--no-show-bounds", "--box-count", "3"])
        viewer = ViewerNull(num_frames=1)
        example = Example(viewer, args)

        max_speed = 0.0
        max_height = -np.inf
        for _frame in range(45):
            example.step()
            wp.synchronize()
            body_q = example.state_0.body_q.numpy()[example.box_body_ids]
            body_qd = example.state_0.body_qd.numpy()[example.box_body_ids]
            max_speed = max(max_speed, float(np.linalg.norm(body_qd[:, :3], axis=1).max()))
            max_height = max(max_height, float(body_q[:, 2].max()))

        particles = example.state_0.particle_q.numpy()
        body_q = example.state_0.body_q.numpy()[example.box_body_ids]
        targets = np.asarray(example.box_target_heights, dtype=np.float32)
        half_z = np.asarray([float(h[2]) for h in example.box_half_extents], dtype=np.float32)

        self.assertLess(max_speed, 4.0)
        self.assertLess(max_height, args.bounds_upper[2] - 0.15)
        self.assertLess(np.abs(body_q[:, 2] - targets).max(), 0.16)
        self.assertGreater(particles[:, 2].max(), float((body_q[:, 2] + half_z).max()) + 0.05)
        self.assertGreater(args.bounds_upper[2] - args.bounds_lower[2], 1.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
