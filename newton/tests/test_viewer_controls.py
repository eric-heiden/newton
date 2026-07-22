# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from newton._src.viewer.camera import Camera
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_gui import ViewerGui
from newton._src.viewer.viewer_null import ViewerNull
from newton._src.viewer.viewer_rtx import ViewerRTX


def _make_gl_state(paused: bool = False, step_requested: bool = False) -> "ViewerGL":
    # Lightweight stand-in with just the fields ViewerGL.should_step() needs.
    return SimpleNamespace(_paused=paused, _step_requested=step_requested)  # type: ignore[return-value]


class TestViewerBaseShouldStep(unittest.TestCase):
    """ViewerBase.should_step() defaults to not self.is_paused()."""

    def test_returns_true_when_not_paused(self):
        viewer = ViewerNull()
        self.assertTrue(viewer.should_step())

    def test_returns_true_on_repeated_calls(self):
        viewer = ViewerNull()
        for _ in range(3):
            self.assertTrue(viewer.should_step())


class TestViewerGLShouldStep(unittest.TestCase):
    """ViewerGL.should_step() state machine: running, paused, and single-step."""

    def test_returns_true_when_running(self):
        v = _make_gl_state(paused=False, step_requested=False)
        self.assertTrue(ViewerGL.should_step(v))

    def test_returns_false_when_paused(self):
        v = _make_gl_state(paused=True, step_requested=False)
        self.assertFalse(ViewerGL.should_step(v))

    def test_returns_true_once_after_step_request(self):
        v = _make_gl_state(paused=True, step_requested=True)
        self.assertTrue(ViewerGL.should_step(v))
        self.assertFalse(ViewerGL.should_step(v))

    def test_stale_request_cleared_when_running(self):
        # Reproduces the bug: . pressed while running, then SPACE to pause.
        # The flag must not survive into the paused state and fire a spurious step.
        v = _make_gl_state(paused=False, step_requested=True)
        ViewerGL.should_step(v)  # running frame — must clear the flag
        v._paused = True
        self.assertFalse(ViewerGL.should_step(v))

    def test_multiple_step_requests_fire_once_each(self):
        v = _make_gl_state(paused=True, step_requested=True)
        self.assertTrue(ViewerGL.should_step(v))
        v._step_requested = True
        self.assertTrue(ViewerGL.should_step(v))
        self.assertFalse(ViewerGL.should_step(v))


class TestModelCameraFollowing(unittest.TestCase):
    def _make_gui(self):
        viewer = SimpleNamespace(
            camera=Camera(),
            model=SimpleNamespace(camera_count=1),
            set_camera_from_model=Mock(),
            _camera_dirty=False,
        )
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        gui._selected_model_camera = -1
        gui._cam_vel = np.zeros(3, dtype=np.float32)
        gui._cam_speed = 4.0
        gui._cam_damp_tau = 0.083
        return gui

    def test_selected_camera_follows_without_toggle(self):
        """Verify selecting a model camera makes every frame follow it."""
        gui = self._make_gui()
        gui._selected_model_camera = 0

        gui.apply_camera_follow()

        gui._viewer.set_camera_from_model.assert_called_once_with(0)

    def test_selecting_camera_resets_keyboard_momentum(self):
        """Verify selecting a model camera snaps to it and clears residual motion."""
        gui = self._make_gui()
        gui._cam_vel[:] = 2.0

        gui._select_model_camera(0)

        self.assertEqual(gui._selected_model_camera, 0)
        np.testing.assert_array_equal(gui._cam_vel, np.zeros(3, dtype=np.float32))
        gui._viewer.set_camera_from_model.assert_called_once_with(0)

    def test_mouse_camera_input_detaches_selected_camera(self):
        """Verify manual mouse camera input clears the model-camera selection."""
        gui = self._make_gui()
        gui._selected_model_camera = 0

        gui.rotate_camera_from_drag(2.0, -1.0)

        self.assertEqual(gui._selected_model_camera, -1)

    def test_keyboard_camera_input_detaches_selected_camera(self):
        """Verify manual keyboard camera input clears the model-camera selection."""
        import pyglet

        gui = self._make_gui()
        gui._selected_model_camera = 0

        gui.update_camera_from_keys(0.016, lambda symbol: symbol == pyglet.window.key.W)

        self.assertEqual(gui._selected_model_camera, -1)

    def test_rtx_applies_camera_follow_before_render_update(self):
        """Verify RTX reapplies the selected model camera before updating its render camera."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._phase = ViewerRTX._PHASE_RENDER
        call_order = []
        viewer.gui = SimpleNamespace(apply_camera_follow=lambda: call_order.append("follow"))
        viewer._update_ovrtx_camera = lambda: call_order.append("camera")
        viewer._update_ovrtx_transforms = Mock()
        viewer._update_ovrtx_instance_visibility = Mock()
        viewer._update_ovrtx_line_batches = Mock()
        viewer._update_ovrtx_point_batches = Mock()
        viewer._update_ovrtx_mesh_points = Mock()
        viewer._render_and_display = Mock()

        viewer.end_frame()

        self.assertEqual(call_order, ["follow", "camera"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
