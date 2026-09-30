# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.viewer.gl.fluid import DiffuseBatch, FluidBatch, FluidRenderer, _Program
from newton._src.viewer.viewer import ViewerBase
from newton._src.viewer.viewer_gl import ViewerGL
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


class _LogFluidGLProbe(ViewerGL):
    """Exercise GL particle routing without creating an OpenGL context."""

    clear_model = ViewerBase.clear_model
    set_model = ViewerBase.set_model
    log_fluid = _LogFluidProbe.log_fluid
    log_points = _LogFluidProbe.log_points

    def __init__(self):
        ViewerBase.__init__(self)
        self.logged_fluid = None
        self.logged_points = None


class _BufferProbeGL:
    """Capture uploaded vertices without requiring an OpenGL context."""

    GLuint = ctypes.c_uint
    GL_ARRAY_BUFFER = 1
    GL_DYNAMIC_DRAW = 2
    GL_FLOAT = 3
    GL_FALSE = 0

    def __getattr__(self, name):
        if name.startswith("gl"):
            return lambda *args: None
        raise AttributeError(name)

    def glBufferSubData(self, target, offset, size, pointer):
        self.vertices = np.ctypeslib.as_array((ctypes.c_float * (size // 4)).from_address(pointer)).copy()


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
        """Send only active model particles to fluid surface rendering."""
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
        """Hide the previous fluid surface when switching to point rendering."""
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
        """Render fluid samples as points on backends without fluid support."""
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active])
        state = model.state()
        viewer = _LogFluidProbe()

        ViewerNull.log_fluid(viewer, "fallback", state.particle_q, radii=0.2, hidden=False)

        self.assertIsNotNone(viewer.logged_points)
        self.assertEqual(viewer.logged_points["name"], "fallback")
        self.assertFalse(viewer.logged_points["hidden"])

    def test_fluid_renderer_groups_surface_batches_by_material(self):
        """Combine batches sharing a material into one surface reconstruction."""
        water = _FluidMaterialProbe(color=(0.1, 0.4, 0.6, 0.8))
        water_later = _FluidMaterialProbe(color=(0.1, 0.4, 0.6, 0.8))
        honey = _FluidMaterialProbe(color=(0.9, 0.5, 0.1, 0.45), absorption=(0.2, 1.1, 2.6))

        groups = FluidRenderer._surface_material_groups([water, honey, water_later])

        self.assertEqual(groups, [[water, water_later], [honey]])

    def test_program_caches_uniform_locations(self):
        """Reuse uniform locations across updates of the same shader program."""
        gl = _UniformProbeGL()
        program = _Program.__new__(_Program)
        program._gl = gl
        program.program = type("ProgramProbe", (), {"id": 3})()
        program._uniform_locations = {}

        self.assertEqual(program._loc("projection"), 7)
        self.assertEqual(program._loc("projection"), 7)
        self.assertEqual(gl.lookup_count, 1)

    def test_gl_show_fluid_without_point_rendering(self):
        """Render automatic fluid surfaces when ordinary particles are hidden."""
        model = self._build_model([int(newton.ParticleFlags.ACTIVE)])
        viewer = _LogFluidGLProbe()
        viewer.set_model(model)
        viewer.show_fluid = True
        viewer.show_particles = False
        viewer._log_particles(model.state())
        self.assertIsNotNone(viewer.logged_fluid)
        self.assertFalse(viewer.logged_fluid["hidden"])
        np.testing.assert_array_equal(viewer.logged_fluid["points"].numpy(), model.particle_q.numpy())

    def test_hidden_layer_clears_fluid_surface(self):
        """Hide an existing fluid surface when its layer becomes invisible."""
        model = self._build_model([int(newton.ParticleFlags.ACTIVE)])
        viewer = _LogFluidGLProbe()
        viewer.set_model(model)
        viewer.show_fluid = True
        viewer.show_particles = True
        viewer._log_particles(model.state())
        viewer.layer.visible = False
        viewer._log_particles(model.state())
        self.assertTrue(viewer.logged_fluid["hidden"])
        self.assertIsNone(viewer.logged_fluid["points"])

    def test_inactive_particles_clear_fluid_surface(self):
        """Clear stale fluid geometry after every particle becomes inactive."""
        model = self._build_model([int(newton.ParticleFlags.ACTIVE)])
        viewer = _LogFluidProbe()
        viewer.set_model(model)
        viewer.show_fluid = True
        viewer._log_particles(model.state())
        model.particle_flags.zero_()
        viewer._log_particles(model.state())
        self.assertTrue(viewer.logged_fluid["hidden"])
        self.assertIsNone(viewer.logged_fluid["points"])

    def test_numpy_anisotropy_matches_warp_vertices(self):
        """Pack equivalent NumPy and Warp ellipsoid samples identically."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        axes = [np.tile([float(i == 0), float(i == 1), float(i == 2), 1.5], (2, 1)) for i in range(3)]
        axes[0][1, 3] = 0.0
        gl = _BufferProbeGL()
        batch = FluidBatch(gl, 2)
        batch.update(
            wp.array(points, dtype=wp.vec3, device="cpu"),
            0.2,
            2.0,
            *[wp.array(axis, dtype=wp.vec4, device="cpu") for axis in axes],
        )
        expected = gl.vertices.copy()
        batch.update(points, 0.2, 2.0, *axes)
        np.testing.assert_array_equal(gl.vertices, expected)

    def test_fallback_preserves_fluid_radius_scale_and_color(self):
        """Apply fluid radius scaling and material color in point-only viewers."""
        calls = []
        viewer = ViewerNull()
        viewer.log_points = lambda **kwargs: calls.append(kwargs)
        points = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu")
        viewer.log_fluid("water", points, radii=0.2, radius_scale=3.0, color=(0.1, 0.2, 0.3, 0.8))
        self.assertAlmostEqual(calls[0]["radii"], 0.6)
        np.testing.assert_allclose(calls[0]["colors"], (0.1, 0.2, 0.3))

    def test_numpy_fluid_points_work_with_warp_only_backends(self):
        """Convert NumPy positions for point backends requiring Warp arrays."""
        with wp.ScopedDevice("cpu"):
            viewer = ViewerNull()
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        radii = np.array([0.1, 0.2], dtype=np.float32)
        logged = []

        def log_points(name, points, radii, colors, hidden):
            logged.append((points.numpy(), np.asarray(radii)))
            self.assertEqual(points.dtype, wp.vec3)
            self.assertEqual(points.device, viewer.device)

        viewer.log_points = log_points
        for scale in (1.0, 2.0):
            viewer.log_fluid("water", points, radii=radii, radius_scale=scale)
            np.testing.assert_array_equal(logged[-1][0], points)
            np.testing.assert_allclose(logged[-1][1], radii * scale)

    def test_fluid_respects_disabled_cuda_interop(self):
        """Keep CUDA registration disabled for fluid and diffuse buffers."""
        if not wp.is_cuda_available():
            self.skipTest("Requires a CUDA device")
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.renderer = SimpleNamespace(gl=_BufferProbeGL())
        viewer.fluids = {}
        viewer.fluid_diffuse = {}
        viewer._qualify = lambda name: name
        viewer._layer_force_hidden = lambda: False
        viewer._enable_cuda_interop = ViewerGL.CudaInterop.NONE
        points = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        foam = wp.full(1, wp.vec4(1.0), device="cuda:0")
        with patch.object(wp, "RegisteredGLBuffer") as register:
            viewer.log_fluid("water", points)
            viewer.log_fluid_diffuse("foam", foam)
            viewer.fluid_diffuse["foam"].sort_for_view(np.eye(4))
            register.assert_not_called()

    def test_cuda_points_accept_numpy_particle_attributes(self):
        """Upload host radius and ellipsoid arrays with CUDA particle positions."""
        if not wp.is_cuda_available():
            self.skipTest("Requires a CUDA device")
        gl = _BufferProbeGL()
        batch = FluidBatch(gl, 1, enable_cuda_interop=False)
        points = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device="cuda:0")
        radii = np.array([0.2], dtype=np.float32)
        axes = [np.array([[float(i == 0), float(i == 1), float(i == 2), 1.0]]) for i in range(3)]
        batch.update(points, radii, 1.0, *axes)
        np.testing.assert_allclose(gl.vertices[:4], [1.0, 2.0, 3.0, 0.2])
        np.testing.assert_allclose(gl.vertices[[7, 11, 15]], [0.2, 0.2, 0.2])

    def test_diffuse_host_uploads_once_after_sorting(self):
        """Upload only the sorted live foam samples, once per vertex buffer."""
        gl = _BufferProbeGL()
        batch = DiffuseBatch(gl, 4)
        positions = np.array([[0.0, 0.0, -1.0, 0.5], [0.0, 0.0, -2.0, 0.5], [0.0, 0.0, -3.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        with patch.object(gl, "glBufferSubData", wraps=gl.glBufferSubData) as upload:
            batch.update(positions, velocities)
            batch.sort_for_view(np.eye(4))
        self.assertEqual(upload.call_count, 2)
        self.assertEqual(batch.count, 2)
        np.testing.assert_array_equal(batch._host_positions, positions[[1, 0]])
        np.testing.assert_array_equal(gl.vertices.reshape(-1, 4), velocities[[1, 0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
