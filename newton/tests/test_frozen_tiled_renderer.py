# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import hashlib
import unittest
from pathlib import Path

import newton

# Directory of the frozen raytracer that backs the deprecated SensorTiledCamera.
_FROZEN_DIR = Path(newton.__file__).resolve().parent / "_src" / "sensors" / "warp_raytrace"

# SHA-256 of every tracked ``.py`` file under _FROZEN_DIR.
#
# This directory is FROZEN: it exists only to support SensorTiledCamera through
# its deprecation window and is intentionally NOT kept in sync with the active
# renderer in ``newton/_src/sensors/sensor_camera_render``. If this manifest goes
# out of date, someone edited (or added/removed) a file they most likely should not have.
#
# raytrace.py, render.py, render_context.py and utils.py carry the world selection,
# light color, ambient light, cloth, texture-transform and convex-hull fixes that
# predate the freeze; existing SensorTiledCamera scripts rely on them.
_FROZEN_MANIFEST = {
    "__init__.py": "2db0e00de28786b0dd6e8b192fea6f9c12cafde351d92f736f3fbd468a534205",
    "camera_utils.py": "b929e6e78070cf647a56a5e3fcfac098d93106320277b416ecf670d0ba963fe9",
    "gaussians.py": "5647dac21aa74d329d83bcc0a062b4b75ad9ecf1590450fad123f754e72b8cf7",
    "lighting.py": "c4091ad05a472c210573092435b0162fc092be09afc28029e80fb28e06d3c73a",
    "raytrace.py": "300a6a0a95e3047bbf9adb82e7abf8f19fb8308a72d8fe32db1883f3260e5689",
    "render.py": "c450b809a4a5c49d8e22f170c98063a41475e5b46b45091f0143aaf0301b61f4",
    "render_context.py": "3e4fb427d8d3c5a590f79cf0f2a7f1915e3a5d0106fb61387b64ca35fa1cdd37",
    "textures.py": "0f12c8e515c267f0962f340234d9114ff4a6b564234cc18f11ae3d24571c821c",
    "tiling.py": "299deed013bf215bd0c693d904cd9320795cfc8cbb4bc4bf71954288fe4122d0",
    "types.py": "b6fc4eabded2380665f7e71ddfb2f7ad0c465e216d11567f3f9d7dec0e798d49",
    "utils.py": "51f4ef9650c79cc584677701acc09fdb50c011e3960e50685d7c353df3727400",
}


class TestFrozenTiledRenderer(unittest.TestCase):
    def test_warp_raytrace_renderer_is_frozen(self) -> None:
        """Verify the deprecated SensorTiledCamera renderer is left unchanged.

        The ``warp_raytrace`` package is frozen for the duration of the
        SensorTiledCamera deprecation window and must not be edited; new
        rendering work belongs in ``newton/_src/sensors/sensor_camera_render``.
        Any change here (including formatting) fails this test until the pinned
        manifest is deliberately updated.
        """

        # Normalize line endings so a CRLF checkout does not false-trip the guard.
        def _hash(path: Path) -> str:
            data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            return hashlib.sha256(data).hexdigest()

        actual = {
            path.relative_to(_FROZEN_DIR).as_posix(): _hash(path)
            for path in _FROZEN_DIR.rglob("*.py")
            if "__pycache__" not in path.parts
        }

        expected_files = set(_FROZEN_MANIFEST)
        actual_files = set(actual)
        added = sorted(actual_files - expected_files)
        removed = sorted(expected_files - actual_files)
        changed = sorted(name for name in expected_files & actual_files if actual[name] != _FROZEN_MANIFEST[name])

        message = (
            "\n"
            "The frozen SensorTiledCamera renderer (newton/_src/sensors/warp_raytrace) was modified.\n"
            "This code path is deprecated and must not be changed -- put new rendering work in\n"
            "newton/_src/sensors/sensor_camera_render instead. If a change here is genuinely required, update\n"
            "_FROZEN_MANIFEST in this test in the same commit so the edit is explicit and reviewed.\n"
            f"  added:   {added}\n"
            f"  removed: {removed}\n"
            f"  changed: {changed}"
        )
        self.assertEqual(actual, _FROZEN_MANIFEST, message)


if __name__ == "__main__":
    unittest.main()
