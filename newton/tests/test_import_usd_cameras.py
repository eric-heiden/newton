# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import newton
from newton import ModelBuilder

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    HAS_USD = True
except ImportError:
    HAS_USD = False


@unittest.skipUnless(HAS_USD, "usd-core not available")
class TestImportUsdCameras(unittest.TestCase):
    def _make_stage(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        cam = UsdGeom.Camera.Define(stage, "/World/overview")
        cam.GetFocalLengthAttr().Set(24.0)
        cam.GetHorizontalApertureAttr().Set(20.955)
        cam.GetVerticalApertureAttr().Set(15.2908)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 500.0))
        UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(0.0, -3.0, 2.0))
        # a rigid body with a child camera
        body = UsdGeom.Xform.Define(stage, "/World/robot")
        UsdGeom.XformCommonAPI(body).SetTranslate(Gf.Vec3d(1.0, 0.0, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        cube = UsdGeom.Cube.Define(stage, "/World/robot/geom")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        head = UsdGeom.Camera.Define(stage, "/World/robot/head_cam")
        UsdGeom.XformCommonAPI(head).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.5))
        return stage

    def test_cameras_imported_with_attributes(self):
        """Verify USD camera prims import with focal length, apertures, and clipping range."""
        builder = ModelBuilder()
        result = builder.add_usd(self._make_stage())
        model = builder.finalize()
        self.assertEqual(model.camera_count, 2)
        self.assertIn("/World/overview", result["path_camera_map"])
        i = result["path_camera_map"]["/World/overview"]
        proj = model.camera_projections[model.camera_projection_index.numpy()[i]]
        self.assertAlmostEqual(proj.focal_length, 24.0, places=4)
        self.assertAlmostEqual(proj.near, 0.1, places=5)
        self.assertAlmostEqual(proj.far, 500.0, places=3)

    def test_camera_under_rigid_body_attaches(self):
        """Verify a camera prim under a rigid-body prim attaches with a body-relative transform."""
        builder = ModelBuilder()
        result = builder.add_usd(self._make_stage())
        model = builder.finalize()
        i = result["path_camera_map"]["/World/robot/head_cam"]
        self.assertGreaterEqual(int(model.camera_body.numpy()[i]), 0)
        self.assertAlmostEqual(float(model.camera_transform.numpy()[i][2]), 0.5, places=4)

    def test_load_cameras_false_skips(self):
        """Verify load_cameras=False imports no cameras."""
        builder = ModelBuilder()
        builder.add_usd(self._make_stage(), load_cameras=False)
        model = builder.finalize()
        self.assertEqual(model.camera_count, 0)


if __name__ == "__main__":
    unittest.main()
