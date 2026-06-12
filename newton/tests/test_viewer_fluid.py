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
        radius_scale=1.0,
        color=(0.113, 0.425, 0.55, 0.8),
        ior=1.0,
        blur_radius_world=None,
        anisotropy=None,
        anisotropy_secondary=None,
        anisotropy_tertiary=None,
        hidden=False,
    ):
        self.logged_fluid = {
            "name": name,
            "points": points,
            "radii": radii,
            "radius_scale": radius_scale,
            "color": color,
            "ior": ior,
            "blur_radius_world": blur_radius_world,
            "anisotropy": anisotropy,
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
        self.assertEqual(viewer.logged_fluid["color"], viewer.fluid_color)
        self.assertEqual(viewer.logged_fluid["ior"], viewer.fluid_ior)
        np.testing.assert_allclose(viewer.logged_fluid["points"].numpy()[:, 0], [0.0, 2.0], atol=1.0e-6)
        self.assertIsNotNone(viewer.logged_points)
        self.assertEqual(viewer.logged_points["name"], "/model/particles")
        self.assertIsNone(viewer.logged_points["points"])
        self.assertTrue(viewer.logged_points["hidden"])

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

    def test_interactive_tank_parser_defaults(self):
        args = Example.create_parser().parse_args([])

        self.assertEqual(args.render_mode, "fluid")
        self.assertEqual(args.substeps, 3)
        self.assertEqual(args.pbf_iterations, 3)
        self.assertGreater(args.box_count, 0)
        self.assertGreater(args.pick_stiffness, 0.0)
        self.assertTrue(args.show_diffuse)
        self.assertGreater(args.fluid_diffuse_max_particles, 0)
        self.assertAlmostEqual(args.buoyancy_scale, 1.0)
        self.assertGreater(args.box_linear_drag, 0.0)
        self.assertGreater(args.box_quadratic_drag, 0.0)
        self.assertGreater(args.box_floor_stiffness, 0.0)
        self.assertGreater(min(args.box_density_fractions), 0.0)
        self.assertGreater(max(args.box_density_fractions), 1.0)
        self.assertEqual(len(args.fluid_color), 4)
        self.assertGreater(args.fluid_radius_scale, 1.0)
        self.assertGreater(args.fluid_blur_radius, 0.0)
        self.assertGreater(args.foam_radius, 0.0)
        self.assertGreater(args.foam_motion_blur, 0.0)

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
        for _frame in range(240):
            example.step()
            wp.synchronize()
            body_qd = example.state_0.body_qd.numpy()[example.box_body_ids]
            max_speed = max(max_speed, float(np.linalg.norm(body_qd[:, :3], axis=1).max()))

        body_q = example.state_0.body_q.numpy()[example.box_body_ids]
        half_z = np.asarray([float(h[2]) for h in example.box_half_extents], dtype=np.float32)
        box_surface = example.box_water_height.numpy()

        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertLess(max_speed, 4.0)

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

        # The denser-than-water box fully submerges and settles near the floor.
        sinker_top = float(body_q[2, 2] + half_z[2])
        sinker_bottom = float(body_q[2, 2] - half_z[2])
        self.assertLess(sinker_top, float(box_surface[2]) + 0.02, "dense box did not fully submerge")
        self.assertLess(sinker_bottom, args.bounds_lower[2] + 0.25, "dense box did not sink to the tank floor")


if __name__ == "__main__":
    unittest.main(verbosity=2)
