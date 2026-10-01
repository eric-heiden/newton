# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import unittest

_SCRIPT = """
import warp as wp
import newton
import newton.viewer

builder = newton.ModelBuilder()
builder.add_ground_plane()
builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.2)
model = builder.finalize()
viewer = newton.viewer.ViewerGL(width=64, height=48, headless=True)
viewer.set_model(model)
viewer.begin_frame(0.0)
viewer.log_state(model.state())
viewer.end_frame()
frame = viewer.get_frame().numpy()
assert frame.shape == (48, 64, 3), frame.shape
assert frame.max() > 0
viewer.close()
print("HEADLESS_OK")
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "EGL headless rendering is Linux-only")
class TestViewerGLHeadless(unittest.TestCase):
    def test_headless_viewer_needs_no_display(self):
        """A headless ViewerGL renders offscreen without an X display."""
        try:
            __import__("pyglet")
        except ImportError as exc:
            self.skipTest(f"pyglet not available: {exc}")
        env = {key: value for key, value in os.environ.items() if key not in ("DISPLAY", "WAYLAND_DISPLAY")}
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=600, check=False
        )
        output = result.stdout + result.stderr
        self.assertNotIn("NoSuchDisplayException", output)
        if "HEADLESS_OK" not in output and ("EGL" in output or "libEGL" in output or "ContextException" in output):
            self.skipTest("no usable EGL device")
        self.assertIn("HEADLESS_OK", output, output[-2000:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
