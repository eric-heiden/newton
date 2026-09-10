# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""External scene ownership tests plus optional real ovstage transport tests."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.viewer import OvstageBodyBinding, ViewerRTX, ViewerUSD


class TestViewerRTXStage(unittest.TestCase):
    def make_viewer(self, **kwargs):
        renderer = mock.Mock()
        stage = mock.Mock()
        with (
            mock.patch.dict("sys.modules", {"ovrtx": SimpleNamespace()}),
            mock.patch.object(ViewerUSD, "__init__", side_effect=AssertionError("must not rebuild USD")),
        ):
            viewer = ViewerRTX(
                stage=stage,
                renderer=renderer,
                render_product="/Render/Camera",
                headless=True,
                **kwargs,
            )
        self.addCleanup(viewer.close)
        return viewer, stage, renderer

    def test_external_scene_is_consumed_without_rebuilding_or_publishing(self):
        """Consume an external scene without creating or committing scene data."""
        viewer, stage, renderer = self.make_viewer(num_frames=2)
        result = viewer.render(ordinal=7, delta_time=0.02)
        self.assertIs(result, renderer.step.return_value)
        renderer.step.assert_called_once_with(render_products={"/Render/Camera"}, delta_time=0.02, ordinal=7)
        stage.assert_not_called()
        self.assertEqual(stage.mock_calls, [])
        self.assertTrue(viewer.is_running())
        viewer.render(ordinal=7)
        self.assertFalse(viewer.is_running())

    def test_close_and_clear_do_not_destroy_external_resources(self):
        """Preserve caller resources across clear and repeated close operations."""
        viewer, stage, renderer = self.make_viewer()
        viewer.clear_model()
        viewer.close()
        viewer.close()
        self.assertEqual(stage.mock_calls, [])
        self.assertEqual(renderer.mock_calls, [])
        self.assertFalse(viewer.is_running())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            viewer.render(ordinal=1)

    def test_missing_or_decreasing_ordinal_is_rejected(self):
        """Reject invalid publication order before rendering."""
        viewer, _, renderer = self.make_viewer()
        with self.assertRaisesRegex(ValueError, "ordinal"):
            viewer.end_frame()
        viewer.end_frame(ordinal=3)
        with self.assertRaises(ValueError):
            viewer.render(ordinal=2)
        with self.assertRaises(ValueError):
            viewer.render(ordinal=True)
        with self.assertRaises(ValueError):
            viewer.render(ordinal=4, delta_time=float("nan"))
        self.assertEqual(renderer.step.call_count, 1)

    def test_logging_does_not_silently_publish_a_second_pose(self):
        """Reject implicit state publication from the native viewer."""
        viewer, stage, _ = self.make_viewer()
        with self.assertRaisesRegex(RuntimeError, "OvstageBodyBinding"):
            viewer.log_state(newton.State())
        self.assertEqual(stage.mock_calls, [])

    def test_set_model_does_not_extract_visual_geometry(self):
        """Associate a model without reconstructing source geometry."""
        viewer, stage, _ = self.make_viewer()
        with mock.patch.object(viewer, "_populate_shapes", side_effect=AssertionError("visual extraction")):
            model = object()
            viewer.set_model(model)
        self.assertIs(viewer.model, model)
        self.assertEqual(stage.mock_calls, [])

    def test_screenshot_reads_full_render_var_path_and_owns_pixels(self):
        """Copy the full-path render output before releasing its mapping."""
        viewer, _, renderer = self.make_viewer()
        original = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        mapping = mock.MagicMock()
        mapping.__enter__.return_value = original
        color = mock.Mock()
        color.map.return_value = mapping
        renderer.step.return_value = {
            "/Render/Camera": SimpleNamespace(frames=[SimpleNamespace(render_vars={"/Render/Vars/LdrColor": color})])
        }
        viewer.render(ordinal=1)
        with mock.patch.dict("sys.modules", {"ovrtx": SimpleNamespace(Device=SimpleNamespace(CPU="cpu"))}):
            pixels = viewer._capture_screenshot_pixels()
        np.testing.assert_array_equal(pixels, original)
        original[:] = 0
        self.assertTrue(pixels.any())
        mapping.__exit__.assert_called_once()

    def test_incomplete_external_configuration_is_rejected(self):
        """Reject an incomplete external renderer configuration."""
        with self.assertRaisesRegex(ValueError, "together"):
            ViewerRTX(stage=object(), headless=True)


@unittest.skipUnless(importlib.util.find_spec("ovstage"), "Requires optional ovstage 0.2")
class TestOvstageBodyBinding(unittest.TestCase):
    def setUp(self):
        import ovstage  # noqa: PLC0415 - optional suite, after skip check

        self.ovstage = ovstage
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        scene = Path(self.directory.name) / "scene.usda"
        scene.write_text("""#usda 1.0
def Xform "World" {
    double3 xformOp:translate = (100, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Xform "A" {
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Xform "B" {
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
""")
        self.stage = ovstage.Stage("newton-binding-test")
        self.addCleanup(self.stage.destroy)
        ovstage.population.open_usd(self.stage, str(scene), ordinal=1, domains=ovstage.PopulationDomain.ALL)
        self.stage.advance_write_floor(1).wait()

    def model(self, device="cpu"):
        builder = newton.ModelBuilder()
        builder.add_body(xform=wp.transform((3.0, 4.0, 5.0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.5)))
        builder.add_body(xform=wp.transform((-2.0, 1.0, 4.0), wp.quat_identity()))
        return builder.finalize(device=device)

    def binding(self, model, *, paths=("/World/B", "/World/A"), indices=(1, 0), offsets=None):
        binding = OvstageBodyBinding(
            self.stage,
            model,
            ordinal=1,
            prim_paths=paths,
            body_indices=indices,
            body_local_transforms=np.repeat(np.eye(4)[None], len(paths), axis=0) if offsets is None else offsets,
        )
        self.addCleanup(binding.close)
        return binding

    def read(self, binding, name, ordinal):
        result = {}
        token = binding._dictionary.intern_token(name)
        with self.stage.read_attributes(binding._query, [token], self.ovstage.OrdinalRange.latest(ordinal)) as read:
            read.wait()
            group = read.fetch_next()
            while group is not None:
                try:
                    data = np.from_dlpack(group.dlpack(0)).copy()
                    for i in range(group.prim_count):
                        result[group.prim_index(i)] = data[group.data_row_index(i)]
                finally:
                    self.stage.release_group(group)
                group = read.fetch_next()
        return np.asarray([result[i] for i in range(len(binding.prim_paths))])

    def test_cpu_affine_offsets_and_permuted_body_indices(self):
        """Preserve affine rest offsets and explicit body index order."""
        model = self.model()
        offsets = np.repeat(np.eye(4)[None], 2, axis=0)
        offsets[0, :3, :3] = np.diag([-2.0, 3.0, 4.0])
        offsets[1, 3, :3] = [0.2, -0.3, 0.4]
        binding = self.binding(model, offsets=offsets)
        state = model.state()
        binding.write(state, ordinal=2)
        self.stage.advance_write_floor(2).wait()
        poses = state.body_q.numpy()
        expected = []
        for i, body in enumerate([1, 0]):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:])), dtype=np.float64).reshape(3, 3)
            world = np.eye(4)
            world[:3, :3] = rotation.T
            world[3, :3] = poses[body, :3]
            expected.append(offsets[i] @ world)
        np.testing.assert_allclose(self.read(binding, "omni:xform", 2).reshape(-1, 4, 4), expected, atol=1e-6)
        reset = self.read(binding, "omni:resetXformStack", 2)
        self.assertEqual(reset.dtype, np.dtype(bool))
        self.assertTrue(reset.all())

    def test_one_body_can_drive_multiple_prims(self):
        """Allow multiple source prims to follow one Newton body."""
        model = self.model()
        binding = self.binding(model, indices=(0, 0))
        binding.write(model.state(), ordinal=2)
        self.stage.advance_write_floor(2).wait()
        actual = self.read(binding, "omni:xform", 2)
        np.testing.assert_allclose(actual[0], actual[1])

    def test_missing_prim_is_rejected_before_writes(self):
        """Reject missing destination transforms before publishing poses."""
        with self.assertRaisesRegex(ValueError, "no populated"):
            self.binding(self.model(), paths=("/World/Missing",), indices=(0,))

    def test_bad_mapping_and_state_are_rejected(self):
        """Reject incompatible mappings, states and binding reuse."""
        model = self.model()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.binding(model, paths=("/World/A", "/World/A"))
        with self.assertRaisesRegex(ValueError, "final model"):
            self.binding(model, indices=(0, 99))
        binding = self.binding(model)
        with self.assertRaisesRegex(ValueError, "State body poses"):
            binding.write(newton.State(), ordinal=2)
        binding.write(model.state(), ordinal=2)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            binding.write(model.state(), ordinal=2)
        self.stage.advance_write_floor(2).wait()
        binding.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            binding.write(model.state(), ordinal=3)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_cuda_tensor_publication(self):
        """Publish CUDA pose buffers and read the committed transforms."""
        model = self.model("cuda:0")
        binding = self.binding(model)
        binding.write(model.state(), ordinal=2)
        self.stage.advance_write_floor(2).wait()
        actual = self.read(binding, "omni:xform", 2).reshape(-1, 4, 4)
        np.testing.assert_allclose(actual[:, 3, :3], model.state().body_q.numpy()[[1, 0], :3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
