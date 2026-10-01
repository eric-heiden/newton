# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fluid example setup, graph replay, and world-separated rendering."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.fluid.example_fluid_archimedes_screw import Example as ScrewExample
from newton.examples.fluid.example_fluid_dam_break import Example as DamBreakExample
from newton.examples.fluid.example_fluid_dam_break import _confine_tank
from newton.examples.fluid.example_fluid_tank import Example as TankExample
from newton.examples.fluid.example_fluid_wave_pool import Example as WavePoolExample
from newton.examples.fluid.example_fluid_wave_pool import drive_wave_paddle
from newton.examples.fluid.utils import (
    FluidParticleRenderer,
    cylinder_particle_count,
    cylinder_particle_positions,
    grid_dimensions,
    resolve_particle_grid,
    resolve_particle_spacing,
    validate_simulation_args,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices
from newton.viewer import ViewerNull


class TestFluidExampleSetup(unittest.TestCase):
    def test_particle_display_offsets(self):
        """Offset visible points without moving physics or allocating each frame."""
        with wp.ScopedDevice("cpu"):
            positions = np.arange(12, dtype=np.float32).reshape(4, 3)
            physical = wp.array(positions, dtype=wp.vec3)
            model = SimpleNamespace(
                device=wp.get_device("cpu"),
                particle_count=4,
                particle_q=physical,
                particle_radius=wp.array([0.01, 0.02, 0.03, 0.04], dtype=float),
                particle_world=wp.array([-1, 0, 1, 1], dtype=int),
                particle_flags=wp.array([1, 1, 0, 1], dtype=int),
            )
            state = SimpleNamespace(particle_q=physical)
            offsets = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)
            viewer = SimpleNamespace(show_particles=True, world_offsets=wp.array(offsets, dtype=wp.vec3))
            calls = []
            viewer.log_state = lambda state: self.assertFalse(viewer.show_particles)
            viewer.log_points = lambda *args, **kwargs: calls.append((args, kwargs))
            renderer = FluidParticleRenderer(model)
            pointers = renderer.positions.ptr, renderer.radii.ptr, renderer.colors.ptr
            renderer.log_state(viewer, state)
            expected = positions.copy()
            expected[1] += offsets[0]
            expected[2:] += offsets[1]
            np.testing.assert_allclose(renderer.positions.numpy(), expected)
            np.testing.assert_array_equal(physical.numpy(), positions)
            np.testing.assert_allclose(renderer.radii.numpy(), [0.01, 0.02, 0.0, 0.04])
            self.assertTrue(viewer.show_particles)
            self.assertFalse(calls[-1][1]["hidden"])
            viewer.show_particles = False
            renderer.log_state(viewer, state)
            self.assertTrue(calls[-1][1]["hidden"])
            self.assertFalse(viewer.show_particles)
            viewer.show_particles = True
            viewer.world_offsets = None
            renderer.log_state(viewer, state)
            np.testing.assert_array_equal(renderer.positions.numpy(), positions)
            self.assertEqual(pointers, (renderer.positions.ptr, renderer.radii.ptr, renderer.colors.ptr))
            # Use the real viewer's subset API to obtain its device visibility mask.
            builder = newton.ModelBuilder()
            for _ in range(2):
                builder.begin_world()
                builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
                builder.end_world()
            null_viewer = ViewerNull()
            null_viewer.set_model(builder.finalize(device="cpu"))
            null_viewer.set_visible_worlds([1])
            viewer._visible_worlds_mask = null_viewer._visible_worlds_mask
            renderer.log_state(viewer, state)
            np.testing.assert_allclose(renderer.radii.numpy(), [0.01, 0.0, 0.0, 0.04])
            null_viewer.set_visible_worlds(None)
            viewer._visible_worlds_mask = null_viewer._visible_worlds_mask
            renderer.log_state(viewer, state)
            np.testing.assert_allclose(renderer.radii.numpy(), [0.01, 0.02, 0.0, 0.04])
            # Null rendering should perform no geometry logging or display preparation.
            null_viewer.log_state = lambda state: self.fail("Null rendering should skip geometry logging")
            null_viewer.log_points = lambda *args, **kwargs: self.fail("Null rendering should skip points")
            renderer.log_state(null_viewer, state)
            null_viewer.close()

    def test_invalid_scene_parameters(self):
        """Reject invalid timesteps and unsupported solver combinations early."""
        args = TankExample.create_parser().parse_args([])
        for key, value in (
            ("fps", 0.0),
            ("fps", np.inf),
            ("substeps", 0),
            ("iterations", 0),
            ("rest_density", 0.0),
            ("rest_density", np.nan),
            ("solver", "mpm"),
        ):
            invalid = SimpleNamespace(**vars(args))
            setattr(invalid, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_simulation_args(invalid, supported_solvers=("xpbd",))

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
            for example_type in (TankExample, WavePoolExample):
                with self.subTest(example=example_type.__module__):
                    args = example_type.create_parser().parse_args(["--particle-count", "64"])
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
                    ["--particle-count", "64", "--substeps", str(substeps)]
                )
                eager = DamBreakExample(ViewerNull(), args)
                eager.use_cuda_graph = False
                captured = DamBreakExample(ViewerNull(), args)
                for frame in range(3):
                    eager.step()
                    captured.step()
                    test.assertIsNotNone(captured.graph)
                    for attribute in ("particle_q", "particle_qd"):
                        np.testing.assert_allclose(
                            getattr(captured.state_0, attribute).numpy(),
                            getattr(eager.state_0, attribute).numpy(),
                            rtol=1.0e-4,
                            atol=2.0e-5,
                            err_msg=f"{attribute} differs after frame {frame} with {substeps} substeps",
                        )


def test_dam_break_overlapping_worlds(test, device):
    """Keep identically initialized overlapping worlds equal to an isolated tank."""
    with wp.ScopedDevice(device):
        parser = DamBreakExample.create_parser()
        isolated = DamBreakExample(ViewerNull(), parser.parse_args(["--particle-count", "125"]))
        batched = DamBreakExample(ViewerNull(), parser.parse_args(["--particle-count", "125", "--world-count", "3"]))
        test.assertEqual(batched.model.world_count, 3)
        test.assertEqual(batched.model.particle_count, 3 * isolated.model.particle_count)
        # Compare before impact changes the active contact set. Neighbor
        # summation order differs between a single grid and world grids;
        # velocity reconstruction amplifies position roundoff by 1 / dt.
        for _ in range(2):
            isolated.step()
            batched.step()
        count = isolated.model.particle_count
        for attribute in ("particle_q", "particle_qd"):
            expected = getattr(isolated.state_0, attribute).numpy()
            actual = getattr(batched.state_0, attribute).numpy().reshape(3, count, 3)
            for world in range(3):
                tolerance = 1.0e-5 if attribute == "particle_q" else 2.0e-4
                np.testing.assert_allclose(actual[world], expected, rtol=1.0e-4, atol=tolerance)


def test_wave_paddle_velocity_matches_motion(test, device):
    """Match the prescribed paddle velocity to its ramped position derivative."""
    with wp.ScopedDevice(device):
        time = wp.array([0.4], dtype=float)
        poses = [wp.empty(1, dtype=wp.transform) for _ in range(2)]
        velocities = [wp.empty(1, dtype=wp.spatial_vector) for _ in range(2)]
        dt, amplitude, frequency, ramp_rate = 0.01, 0.16, 4.0, 0.5
        wp.launch(
            drive_wave_paddle,
            dim=1,
            inputs=[
                time,
                dt,
                0,
                wp.vec3(0.0),
                amplitude,
                frequency,
                ramp_rate,
                poses[0],
                velocities[0],
                poses[1],
                velocities[1],
            ],
        )
        t = 0.4 + dt
        expected = amplitude * ramp_rate * (np.sin(frequency * t) + t * frequency * np.cos(frequency * t))
        for velocity in velocities:
            test.assertAlmostEqual(float(velocity.numpy()[0, 0]), expected, delta=1.0e-6)


def test_mpm_tank_boundary_projection(test, device):
    """Correct escaped particles while preserving tangential and inward motion."""
    with wp.ScopedDevice(device):
        positions = wp.array([[-1.3, 0.2, -0.1], [0.0, 0.7, 0.4], [1.3, 0.0, 0.5], [0.0, 0.0, 2.0]], dtype=wp.vec3)
        velocities = wp.array([[-1.0, 2.0, -3.0], [1.0, -2.0, 3.0], [-1.0, 2.0, 3.0], [0.0, 0.0, 3.0]], dtype=wp.vec3)
        wp.launch(_confine_tank, dim=4, inputs=[positions, velocities, 1.2, 0.5])
        np.testing.assert_allclose(
            positions.numpy(), [[-1.2, 0.2, 0.0], [0.0, 0.5, 0.4], [1.2, 0.0, 0.5], [0.0, 0.0, 2.0]]
        )
        np.testing.assert_allclose(
            velocities.numpy(), [[0.0, 2.0, 0.0], [1.0, -2.0, 3.0], [-1.0, 2.0, 3.0], [0.0, 0.0, 3.0]]
        )


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
    "test_dam_break_overlapping_worlds",
    test_dam_break_overlapping_worlds,
    devices=get_test_devices(),
    check_output=False,
)

add_function_test(
    TestFluidExampleDevices,
    "test_wave_paddle_velocity_matches_motion",
    test_wave_paddle_velocity_matches_motion,
    devices=get_test_devices(),
    check_output=False,
)

add_function_test(
    TestFluidExampleDevices,
    "test_mpm_tank_boundary_projection",
    test_mpm_tank_boundary_projection,
    devices=get_test_devices(),
    check_output=False,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
