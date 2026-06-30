# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton._src.viewer.gl.fluid import FluidBatch, FluidRenderer, _Program
from newton.viewer import ViewerNull


class _FluidMaterialProbe:
    def __init__(self, color=(0.113, 0.425, 0.55, 0.8), absorption=None, blur_radius_world=0.06):
        for attr in FluidBatch._MATERIAL_ATTRS:
            setattr(self, attr, getattr(self, "_default_" + attr)())
        self.color = color
        self.absorption = absorption
        self.blur_radius_world = blur_radius_world

    @staticmethod
    def _default_color():
        return (0.113, 0.425, 0.55, 0.8)

    @staticmethod
    def _default_absorption():
        return None

    @staticmethod
    def _default_ior():
        return 1.0

    @staticmethod
    def _default_reflectance():
        return 0.1

    @staticmethod
    def _default_specular_intensity():
        return 1.2

    @staticmethod
    def _default_specular_power():
        return 400.0

    @staticmethod
    def _default_blur_radius_world():
        return 0.06

    @staticmethod
    def _default_max_blur_radius():
        return 8.0

    @staticmethod
    def _default_shadow_opacity():
        return 0.5

    @staticmethod
    def _default_thickness_scale():
        return 4.0

    @staticmethod
    def _default_thickness_gain():
        return 0.0015


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


class _UniformProbeGL:
    def __init__(self):
        self.lookup_count = 0

    def glGetUniformLocation(self, program, name):
        self.lookup_count += 1
        return 7


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

    def test_fluid_renderer_groups_surface_batches_by_material(self):
        water = _FluidMaterialProbe(color=(0.1, 0.4, 0.6, 0.8))
        water_later = _FluidMaterialProbe(color=(0.1, 0.4, 0.6, 0.8))
        honey = _FluidMaterialProbe(color=(0.9, 0.5, 0.1, 0.45), absorption=(0.2, 1.1, 2.6))

        groups = FluidRenderer._surface_material_groups([water, honey, water_later])

        self.assertEqual(groups, [[water, water_later], [honey]])

    def test_program_caches_uniform_locations(self):
        gl = _UniformProbeGL()
        program = _Program.__new__(_Program)
        program._gl = gl
        program.program = type("ProgramProbe", (), {"id": 3})()
        program._uniform_locations = {}

        self.assertEqual(program._loc("projection"), 7)
        self.assertEqual(program._loc("projection"), 7)
        self.assertEqual(gl.lookup_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
