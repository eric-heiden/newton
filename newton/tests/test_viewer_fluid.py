# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.fluid.example_fluid_sph_interactive_tank import Example
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
        color=(0.10, 0.50, 0.80),
        deep_color=(0.01, 0.09, 0.34),
        color_gradient_strength=0.20,
        opacity=1.00,
        radius_scale=1.34,
        thickness_scale=0.55,
        smoothing_iterations=7,
        smoothing_radius=3.83,
        smoothing_depth_edge_falloff=5.5,
        smoothing_max_samples=4,
        reflection_strength=0.528,
        refraction_strength=0.038,
        env_map_strength=1.02,
        env_reflection_lod=1.8,
        env_color_preserve=0.57,
        absorption_strength=1.2,
        depth_visualization_strength=2.13,
        caustic_strength=3.03,
        caustic_scale=37.1,
        floor_caustic_strength=1.15,
        surface_shadow_strength=0.35,
        foam_strength=0.99,
        foam_scale=5.0,
        hidden=False,
        render_points=None,
        anisotropy=None,
        anisotropy_secondary=None,
        anisotropy_tertiary=None,
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
            "smoothing_depth_edge_falloff": smoothing_depth_edge_falloff,
            "smoothing_max_samples": smoothing_max_samples,
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
            "surface_shadow_strength": surface_shadow_strength,
            "foam_strength": foam_strength,
            "foam_scale": foam_scale,
            "hidden": hidden,
            "render_points": render_points,
            "anisotropy": anisotropy,
            "anisotropy_secondary": anisotropy_secondary,
            "anisotropy_tertiary": anisotropy_tertiary,
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
        self.assertEqual(viewer.logged_fluid["smoothing_depth_edge_falloff"], viewer.fluid_smoothing_depth_edge_falloff)
        self.assertEqual(viewer.logged_fluid["smoothing_max_samples"], viewer.fluid_smoothing_max_samples)
        self.assertEqual(viewer.logged_fluid["deep_color"], viewer.fluid_deep_color)
        self.assertEqual(viewer.logged_fluid["color_gradient_strength"], viewer.fluid_color_gradient_strength)
        self.assertEqual(viewer.logged_fluid["env_map_strength"], viewer.fluid_env_map_strength)
        self.assertEqual(viewer.logged_fluid["env_reflection_lod"], viewer.fluid_env_reflection_lod)
        self.assertEqual(viewer.logged_fluid["env_color_preserve"], viewer.fluid_env_color_preserve)
        self.assertEqual(viewer.logged_fluid["absorption_strength"], viewer.fluid_absorption_strength)
        self.assertEqual(viewer.logged_fluid["depth_visualization_strength"], viewer.fluid_depth_visualization_strength)
        self.assertEqual(viewer.logged_fluid["caustic_scale"], viewer.fluid_caustic_scale)
        self.assertEqual(viewer.logged_fluid["floor_caustic_strength"], viewer.fluid_floor_caustic_strength)
        self.assertEqual(viewer.logged_fluid["surface_shadow_strength"], viewer.fluid_surface_shadow_strength)
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

    def test_default_fluid_material_matches_water_shader_preset(self):
        viewer = _LogFluidProbe()

        self.assertEqual(viewer.fluid_color, (0.10, 0.50, 0.80))
        self.assertEqual(viewer.fluid_deep_color, (0.01, 0.09, 0.34))
        self.assertAlmostEqual(viewer.fluid_color_gradient_strength, 0.20)
        self.assertAlmostEqual(viewer.fluid_opacity, 1.00)
        self.assertAlmostEqual(viewer.fluid_radius_scale, 1.34)
        self.assertAlmostEqual(viewer.fluid_thickness_scale, 0.55)
        self.assertEqual(viewer.fluid_smoothing_iterations, 7)
        self.assertAlmostEqual(viewer.fluid_smoothing_radius, 3.83)
        self.assertAlmostEqual(viewer.fluid_smoothing_depth_edge_falloff, 5.5)
        self.assertEqual(viewer.fluid_smoothing_max_samples, 4)
        self.assertAlmostEqual(viewer.fluid_reflection_strength, 0.528)
        self.assertAlmostEqual(viewer.fluid_refraction_strength, 0.038)
        self.assertAlmostEqual(viewer.fluid_env_map_strength, 1.02)
        self.assertAlmostEqual(viewer.fluid_env_reflection_lod, 1.8)
        self.assertAlmostEqual(viewer.fluid_env_color_preserve, 0.57)
        self.assertAlmostEqual(viewer.fluid_absorption_strength, 1.2)
        self.assertAlmostEqual(viewer.fluid_depth_visualization_strength, 2.13)
        self.assertAlmostEqual(viewer.fluid_caustic_strength, 3.03)
        self.assertAlmostEqual(viewer.fluid_floor_caustic_strength, 1.15)
        self.assertAlmostEqual(viewer.fluid_caustic_scale, 37.1)
        self.assertAlmostEqual(viewer.fluid_surface_shadow_strength, 0.35)
        self.assertAlmostEqual(viewer.fluid_foam_strength, 0.99)
        self.assertAlmostEqual(viewer.fluid_foam_scale, 5.0)
        self.assertAlmostEqual(viewer.fluid_diffuse_radius, 0.012)
        self.assertAlmostEqual(viewer.fluid_diffuse_alpha, 0.34)
        self.assertAlmostEqual(viewer.fluid_diffuse_motion_blur_scale, 0.12)
        self.assertAlmostEqual(viewer.fluid_diffuse_expansion, 1.23)
        self.assertAlmostEqual(viewer.fluid_diffuse_inscatter, 0.60)
        self.assertAlmostEqual(viewer.fluid_diffuse_outscatter, 0.70)
        self.assertAlmostEqual(viewer.fluid_diffuse_shadow_strength, 0.62)

    def test_interactive_tank_parser_exposes_shader_and_picking_controls(self):
        args = Example.create_parser().parse_args([])

        self.assertEqual(args.render_mode, "fluid")
        self.assertEqual(args.substeps, 4)
        self.assertEqual(args.dim_x, 68)
        self.assertEqual(args.dim_y, 44)
        self.assertEqual(args.dim_z, 12)
        self.assertGreater(args.box_count, 0)
        self.assertGreater(args.pick_stiffness, 0.0)
        self.assertTrue(args.show_diffuse)
        self.assertGreater(args.fluid_diffuse_max_particles, 0)
        self.assertEqual(args.fluid_diffuse_max_particles, 16000)
        self.assertGreater(args.cohesion, 0.0)
        self.assertGreater(args.particle_friction, 0.0)
        self.assertGreater(args.particle_collision_margin, 0.0)
        self.assertGreater(args.surface_tension, 0.0)
        self.assertEqual(args.vorticity_confinement, 0.0)
        self.assertGreater(args.solid_pressure, 0.0)
        self.assertAlmostEqual(args.buoyancy, 1.0)
        self.assertGreater(args.xsph_strength, 0.1)
        self.assertGreaterEqual(args.free_surface_drag, 0.4)
        self.assertGreater(args.dissipation, 0.5)
        self.assertGreater(args.sleep_threshold, 0.0)
        self.assertGreater(args.shape_collision_distance, 0.0)
        self.assertGreater(args.shape_collision_margin, 0.0)
        self.assertEqual(args.shape_restitution, 0.0)
        self.assertGreater(args.shape_friction, 0.0)
        self.assertGreater(args.shape_adhesion, 0.0)
        self.assertGreater(args.max_acceleration, 0.0)
        self.assertGreater(args.fluid_render_smoothing, 0.0)
        self.assertGreater(args.fluid_render_anisotropy_scale, 0.0)
        self.assertAlmostEqual(args.fluid_render_anisotropy_min, 0.1)
        self.assertGreater(args.fluid_render_anisotropy_max, args.fluid_render_anisotropy_min)
        self.assertEqual(args.fluid_render_update_interval, 2)
        self.assertEqual(args.fluid_diffuse_update_interval, 2)
        self.assertEqual(args.pbf_iterations, 4)
        self.assertGreater(args.fluid_diffuse_inscatter, 0.0)
        self.assertGreater(args.fluid_diffuse_outscatter, 0.0)
        self.assertGreater(args.fluid_diffuse_shadow_strength, 0.0)
        self.assertGreater(args.fluid_diffuse_expansion, 0.0)
        self.assertGreaterEqual(args.fluid_shadow_size, 2048)
        self.assertAlmostEqual(args.fluid_smoothing_depth_edge_falloff, 5.5)
        self.assertEqual(args.fluid_smoothing_max_samples, 4)
        self.assertAlmostEqual(args.fluid_env_map_strength, 1.02)
        self.assertAlmostEqual(args.fluid_env_reflection_lod, 1.8)
        self.assertGreater(args.fluid_floor_caustic_strength, 1.0)
        self.assertGreater(args.fluid_surface_shadow_strength, 0.0)
        self.assertGreater(args.fluid_depth_visualization_strength, 0.5)
        self.assertAlmostEqual(args.environment_intensity, 3.15)
        self.assertAlmostEqual(args.exposure, 1.08)
        self.assertAlmostEqual(args.specular_scale, 4.00)
        self.assertEqual(args.sun_direction, (0.78, -0.56, 0.20))
        self.assertIsNone(args.water_level)
        self.assertAlmostEqual(args.buoyancy_scale, 1.0)
        self.assertGreater(args.box_linear_drag, 0.0)
        self.assertGreater(args.box_quadratic_drag, 0.0)
        self.assertGreater(args.box_angular_drag, 0.0)
        self.assertGreater(args.box_floor_stiffness, 0.0)
        self.assertGreater(args.box_wall_stiffness, 0.0)
        self.assertGreater(min(args.box_density_fractions), 0.0)
        self.assertGreater(max(args.box_density_fractions), 1.0)
        self.assertGreater(args.box_max_linear_speed, 0.0)
        self.assertGreater(args.box_max_angular_speed, 0.0)
        self.assertEqual(args.box_max_torque, 0.0)

    def test_interactive_tank_rollout_floats_and_sinks_boxes_by_density(self):
        args = Example.create_parser().parse_args(
            [
                "--viewer",
                "null",
                "--no-show-bounds",
                "--box-count",
                "3",
                "--box-density-fractions",
                "0.30",
                "0.60",
                "1.60",
                "--spacing",
                "0.08",
                "--radius",
                "0.06",
                "--smoothing-length",
                "0.172",
                "--shape-collision-distance",
                "0.06",
                "--shape-collision-margin",
                "0.003",
                "--particle-collision-margin",
                "0.003",
                "--fluid-carve-clearance",
                "0.08",
                "--dim-x",
                "34",
                "--dim-y",
                "22",
                "--dim-z",
                "6",
                "--emit-lower",
                "-1.24",
                "-0.78",
                "0.06",
                "--fluid-diffuse-max-particles",
                "0",
                "--fluid-render-update-interval",
                "1",
            ]
        )
        viewer = ViewerNull(num_frames=1)
        example = Example(viewer, args)

        max_speed = 0.0
        max_angular_speed = 0.0
        max_height = -np.inf
        initial_body_q = example.state_0.body_q.numpy()[example.box_body_ids, 3:7].copy()
        for _frame in range(240):
            example.step()
            wp.synchronize()
            body_q = example.state_0.body_q.numpy()[example.box_body_ids]
            body_qd = example.state_0.body_qd.numpy()[example.box_body_ids]
            max_speed = max(max_speed, float(np.linalg.norm(body_qd[:, :3], axis=1).max()))
            max_angular_speed = max(max_angular_speed, float(np.linalg.norm(body_qd[:, 3:6], axis=1).max()))
            max_height = max(max_height, float(body_q[:, 2].max()))

        active_particles = (example.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        particles = example.state_0.particle_q.numpy()[active_particles]
        particle_qd = example.state_0.particle_qd.numpy()[active_particles]
        body_q = example.state_0.body_q.numpy()[example.box_body_ids]
        body_qd = example.state_0.body_qd.numpy()[example.box_body_ids]
        half_z = np.asarray([float(h[2]) for h in example.box_half_extents], dtype=np.float32)
        orientation_delta = np.linalg.norm(body_q[:, 3:7] - initial_body_q, axis=1)
        particle_rms_speed = float(np.sqrt(np.mean(np.sum(particle_qd * particle_qd, axis=1))))
        # The settled SPH surface sits above the analytic fill estimate at this
        # coarse resolution; judge buoyancy against the measured surface that
        # actually drives it.
        box_surface = example.box_water_height.numpy()

        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertLess(max_speed, 4.0)
        self.assertGreater(max_angular_speed, 0.05)
        self.assertGreater(float(orientation_delta.max()), 0.01)
        self.assertLess(particle_rms_speed, 0.10)
        self.assertLess(float(np.linalg.norm(body_qd[:, :3], axis=1).max()), 0.25)
        self.assertLess(max_height, args.bounds_upper[2] - 0.15)
        self.assertGreater(particles[:, 2].max(), example.water_level - 0.10)

        # Boxes lighter than water float partially submerged with a draft that
        # roughly tracks their density fraction.
        for box_idx, target_fraction in ((0, 0.30), (1, 0.60)):
            surface = float(box_surface[box_idx])
            bottom = float(body_q[box_idx, 2] - half_z[box_idx])
            top = float(body_q[box_idx, 2] + half_z[box_idx])
            self.assertLess(bottom, surface, f"floating box {box_idx} lost contact with the water")
            self.assertGreater(top, surface - 0.10, f"floating box {box_idx} is fully submerged")
            submerged_fraction = (surface - bottom) / (2.0 * float(half_z[box_idx]))
            self.assertLess(
                abs(submerged_fraction - target_fraction),
                0.35,
                f"floating box {box_idx} draft {submerged_fraction:.2f} far from density fraction",
            )

        # The denser-than-water box fully submerges and settles near the tank
        # floor (a contact-shell layer of particles remains beneath it at this
        # coarse test resolution).
        sinker_top = float(body_q[2, 2] + half_z[2])
        sinker_bottom = float(body_q[2, 2] - half_z[2])
        self.assertLess(sinker_top, float(box_surface[2]) + 0.02, "dense box did not fully submerge")
        self.assertLess(sinker_bottom, args.bounds_lower[2] + 0.25, "dense box did not sink to the tank floor")


if __name__ == "__main__":
    unittest.main(verbosity=2)
