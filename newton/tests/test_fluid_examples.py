# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fluid example setup, graph replay, and world-separated rendering."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.fluid.example_fluid_xpbd_archimedes_screw import Example as ScrewExample
from newton.examples.fluid.example_fluid_xpbd_cup_transfer import Example as CupTransferExample
from newton.examples.fluid.example_fluid_xpbd_dam_break import Example as DamBreakExample
from newton.examples.fluid.example_fluid_xpbd_interactive_tank import Example as TankExample
from newton.examples.fluid.example_fluid_xpbd_multi_fluid_tank import Example as MultiFluidTankExample
from newton.examples.fluid.example_fluid_xpbd_multiworld_cup import Example as MultiworldCupExample
from newton.examples.fluid.example_fluid_xpbd_wave_pool import Example as WavePoolExample
from newton.examples.fluid.utils import (
    cylinder_particle_count,
    cylinder_particle_positions,
    grid_dimensions,
    resolve_particle_grid,
    resolve_particle_spacing,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices
from newton.viewer import ViewerNull


class TestFluidExampleSetup(unittest.TestCase):
    def test_cup_transfer_detects_spilled_water(self):
        """Reject water left outside the carried cup at the default speed."""
        with wp.ScopedDevice("cpu"):
            example = CupTransferExample.__new__(CupTransferExample)
            example.speed = 1.0
            example.cup_body = 0
            example.cup_inner_radius = 0.05
            example.cup_height = 0.13
            example.wall_thickness = 0.01
            example.model = SimpleNamespace(
                particle_max_radius=0.001,
                particle_flags=wp.array([int(newton.ParticleFlags.ACTIVE)] * 2, dtype=wp.int32),
            )
            example.state_0 = SimpleNamespace(
                particle_q=wp.array([[0.5, 0.3, 0.001], [0.5, 0.3, 0.03]], dtype=wp.vec3),
                particle_qd=wp.zeros(2, dtype=wp.vec3),
                body_q=wp.array([wp.transform((0.5, -0.3, 0.325), wp.quat_identity())], dtype=wp.transform),
                body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            )
            with self.assertRaisesRegex(ValueError, "carried cup"):
                example.test_final()
            example.state_0.particle_q.assign([[0.5, -0.3, 0.35], [0.5, -0.3, 0.36]])
            example.test_final()

    def test_wheel_rotation_returns_through_zero(self):
        """Accept a rotating wheel crossing zero while rejecting a stationary one."""
        with wp.ScopedDevice("cpu"):
            example = ScrewExample.__new__(ScrewExample)
            example.sim_time = 4.0
            example.particle_radius = 0.01
            example.wheel_body = 0
            example.wheel_joint = 0
            example.viewer = SimpleNamespace()
            example.state_0 = SimpleNamespace(
                particle_q=wp.array([[0.0, 0.0, 0.6]], dtype=wp.vec3),
                particle_qd=wp.zeros(1, dtype=wp.vec3),
                body_q=wp.array([wp.transform((0.80, 0.0, 0.50), wp.quat_identity())], dtype=wp.transform),
                body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            )
            example.model = SimpleNamespace(
                joint_qd_start=wp.array([0], dtype=wp.int32),
                joint_q_start=wp.array([0], dtype=wp.int32),
                joint_target_mode=wp.array([int(newton.JointTargetMode.NONE)], dtype=wp.int32),
                joint_target_ke=wp.zeros(1),
                joint_target_kd=wp.zeros(1),
                joint_q=wp.zeros(1),
                body_flags=wp.array([int(newton.BodyFlags.DYNAMIC)], dtype=wp.int32),
            )
            example.screw_motion = wp.array([4.0, 30.0, 0.2], dtype=float)
            example.test_final()
            example.screw_motion.assign([4.0, 30.0, 0.0])
            with self.assertRaisesRegex(ValueError, "did not rotate"):
                example.test_final()

    def test_initial_body_rotations(self):
        """Initialize tank and wave-pool bodies with unit quaternions."""
        with wp.ScopedDevice("cpu"):
            for example_type in (TankExample, MultiFluidTankExample, WavePoolExample):
                with self.subTest(example=example_type.__module__):
                    args = example_type.create_parser().parse_args(
                        ["--particle-count", "64", "--foam-max-particles", "0"]
                    )
                    example = example_type(ViewerNull(), args)
                    rotations = example.state_0.body_q.numpy()[:, 3:]
                    np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1.0, atol=1.0e-6)

    def test_target_below_minimum_grid(self):
        """Return the minimum realizable grid for a smaller target count."""
        grid = resolve_particle_grid(1, (1.0, 1.0, 1.0), 0.02, minimum=(2, 2, 2))
        self.assertEqual(grid.dimensions, (2, 2, 2))
        self.assertEqual(grid.particle_count, 8)

    def test_empty_cylinder_fill(self):
        """Keep the count and generated points empty below the cavity floor."""
        parameters = (0.02, 0.05, 0.03, 0.02)
        self.assertEqual(cylinder_particle_count(*parameters), 0)
        self.assertEqual(cylinder_particle_positions(*parameters).shape, (0, 3))

    def test_cylinder_count_matches_positions(self):
        """Match the resolved particle count to actual cylindrical fills."""
        for spacing in (0.004, 0.02, 0.1):
            parameters = spacing, 0.06, 0.008, 0.082
            self.assertEqual(cylinder_particle_count(*parameters), len(cylinder_particle_positions(*parameters)))

    def test_unreachable_particle_count(self):
        """Reject impossible spacing searches instead of looping forever."""
        with self.assertRaisesRegex(ValueError, "cannot reach the target"):
            resolve_particle_spacing(1, 0.02, lambda spacing: 8)
        with self.assertRaisesRegex(ValueError, "cannot reach the target"):
            resolve_particle_spacing(8, 0.02, lambda spacing: 0)

    def test_invalid_grid_dimensions(self):
        """Reject nonfinite spacing and nonpositive grid dimensions."""
        for spacing in (0.0, -1.0, np.inf, np.nan):
            with self.assertRaises(ValueError):
                grid_dimensions((1.0, 1.0, 1.0), spacing)
        for size in ((0.0, 1.0, 1.0), (np.inf, 1.0, 1.0)):
            with self.assertRaises(ValueError):
                grid_dimensions(size, 0.02)
        with self.assertRaises(ValueError):
            grid_dimensions((1.0, 1.0, 1.0), 0.02, minimum=(0, 2, 2))


def test_fluid_example_graph_matches_eager(test, device):
    """Advance every frame identically with odd and even captured substeps."""
    with wp.ScopedDevice(device):
        for substeps in (1, 2, 3):
            with test.subTest(substeps=substeps):
                args = DamBreakExample.create_parser().parse_args(
                    ["--particle-count", "64", "--substeps", str(substeps), "--foam-max-particles", "0"]
                )
                eager = DamBreakExample(ViewerNull(), args)
                eager.use_cuda_graph = False
                captured = DamBreakExample(ViewerNull(), args)
                for frame in range(3):
                    eager.step()
                    captured.step()
                    test.assertIsNotNone(captured.graph)
                    for attribute in ("particle_q", "particle_qd", "body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(captured.state_0, attribute).numpy(),
                            getattr(eager.state_0, attribute).numpy(),
                            rtol=1.0e-4,
                            atol=2.0e-5,
                            err_msg=f"{attribute} differs after frame {frame} with {substeps} substeps",
                        )


def test_multiworld_fluid_compaction(test, device):
    """Keep interleaved worlds separate while applying device-side offsets."""
    with wp.ScopedDevice(device):
        positions = np.arange(12, dtype=np.float32).reshape(4, 3)
        radii = np.arange(4, dtype=np.float32)
        anisotropy = np.arange(16, dtype=np.float32).reshape(4, 4)
        world_offsets = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]], dtype=np.float32)
        example = MultiworldCupExample.__new__(MultiworldCupExample)
        example.model = SimpleNamespace(
            device=wp.get_device(device),
            particle_count=4,
            particle_world=wp.array([1, 0, 1, 0], dtype=wp.int32),
            particle_radius=wp.array(radii),
        )
        example.solver = SimpleNamespace(
            render_positions=wp.array(positions, dtype=wp.vec3),
            render_anisotropy=wp.array(anisotropy, dtype=wp.vec4),
            render_anisotropy_secondary=wp.array(anisotropy + 1, dtype=wp.vec4),
            render_anisotropy_tertiary=wp.array(anisotropy + 2, dtype=wp.vec4),
        )
        example.viewer = SimpleNamespace(world_offsets=wp.array(world_offsets, dtype=wp.vec3))
        example._world_particle_counts = [2, 2]
        example._render_world_mask = wp.empty(4, dtype=wp.int32)
        example._render_world_offsets = wp.empty(4, dtype=wp.int32)
        example._world_render_cache = {}
        for world, indices in ((0, [1, 3]), (1, [0, 2])):
            cache, count = example._compact_world_render_particles(world)
            test.assertEqual(count, 2)
            np.testing.assert_allclose(cache["positions"].numpy(), positions[indices] + world_offsets[world])
            np.testing.assert_array_equal(cache["radii"].numpy(), radii[indices])
            np.testing.assert_array_equal(cache["anisotropy"].numpy(), anisotropy[indices])


class TestFluidExampleDevices(unittest.TestCase):
    pass


add_function_test(
    TestFluidExampleDevices,
    "test_fluid_example_graph_matches_eager",
    test_fluid_example_graph_matches_eager,
    devices=get_cuda_test_devices(),
    check_output=False,
)
add_function_test(
    TestFluidExampleDevices,
    "test_multiworld_fluid_compaction",
    test_multiworld_fluid_compaction,
    devices=get_test_devices(),
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
