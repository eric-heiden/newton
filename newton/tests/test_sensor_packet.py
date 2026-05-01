# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sensors.sensor_packet import (
    SensorStepPacket,
    build_sensor_camera_query_batch,
    build_sensor_scene_tables,
    build_sensor_step_packet,
    render_tiled_camera_from_packet,
)
from newton.sensors import SensorTiledCamera


def _build_multiworld_rigid_model(*, device: str = "cpu") -> newton.Model:
    """Build a tiny rigid multi-world scene with one sphere per world."""
    blueprint = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body = blueprint.add_body(xform=wp.transform((0.0, 0.0, 0.08), wp.quat_identity()))
    blueprint.add_shape_sphere(body, radius=0.10, label="ball")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    builder.add_world(blueprint, xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_world(blueprint, xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()))

    with wp.ScopedDevice(device):
        return builder.finalize()


def _camera_queries(world_count: int) -> list[list[wp.transformf]]:
    """Return one camera per world with the same local view."""
    return [
        [wp.transformf(wp.vec3f(float(world_id), -1.5, 0.8), wp.quat_identity())] for world_id in range(world_count)
    ]


class TestSensorPacket(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = "cpu"

    def test_scene_tables_reference_model_arrays(self):
        model = _build_multiworld_rigid_model(device=self.device)

        scene = build_sensor_scene_tables(model)

        self.assertEqual(scene.world_count, model.world_count)
        self.assertIs(scene.shape_body, model.shape_body)
        self.assertIs(scene.shape_world, model.shape_world)
        self.assertIs(scene.shape_transform_local, model.shape_transform)
        self.assertIs(scene.body_world, model.body_world)

    def test_step_packet_batches_contacts_by_world(self):
        model = _build_multiworld_rigid_model(device=self.device)
        state = model.state()
        contacts = model.contacts()
        model.collide(state, contacts)

        scene = build_sensor_scene_tables(model)
        camera_queries = build_sensor_camera_query_batch(_camera_queries(model.world_count), device=model.device)
        packet = build_sensor_step_packet(scene, model, state, contacts=contacts, camera_queries=camera_queries)

        self.assertGreaterEqual(packet.contact_count, 2)

        world_start = packet.contact_world_start.numpy()
        self.assertEqual(world_start.shape[0], model.world_count + 1)
        self.assertEqual(int(world_start[0]), 0)
        self.assertEqual(int(world_start[-1]), packet.contact_count)

        shape_world = model.shape_world.numpy()
        shape0 = packet.contact_shape0.numpy()
        shape1 = packet.contact_shape1.numpy()
        flags = packet.contact_flags.numpy()
        for world_id in range(model.world_count):
            begin = int(world_start[world_id])
            end = int(world_start[world_id + 1])
            self.assertGreater(end - begin, 0)

            world_shape0 = shape_world[shape0[begin:end]]
            world_shape1 = shape_world[shape1[begin:end]]
            resolved = np.where(world_shape0 >= 0, world_shape0, world_shape1)
            np.testing.assert_array_equal(resolved, np.full(end - begin, world_id, dtype=np.int32))

        valid_mask = int(
            SensorStepPacket.ContactFlags.VALID_BODY_IDS
            | SensorStepPacket.ContactFlags.VALID_LOCAL_POINTS
            | SensorStepPacket.ContactFlags.VALID_NORMAL
            | SensorStepPacket.ContactFlags.VALID_MARGINS
            | SensorStepPacket.ContactFlags.BACKEND_NATIVE
        )
        self.assertTrue(np.all((flags & valid_mask) == valid_mask))
        self.assertEqual(packet.contact_handle.numpy().dtype, np.uint64)

    def test_tiled_camera_runtime_consumes_packet_queries(self):
        model = _build_multiworld_rigid_model(device=self.device)
        state = model.state()

        sensor = SensorTiledCamera(model)
        width = 24
        height = 16
        camera_rays = sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(55.0))

        camera_queries = build_sensor_camera_query_batch(_camera_queries(model.world_count), device=model.device)
        scene = build_sensor_scene_tables(model)
        packet = build_sensor_step_packet(scene, model, state, camera_queries=camera_queries)

        direct_transforms = wp.array(
            [[world_queries[0] for world_queries in _camera_queries(model.world_count)]],
            dtype=wp.transformf,
            device=model.device,
        )
        direct_depth = sensor.utils.create_depth_image_output(width, height, camera_count=1)
        packet_depth = sensor.utils.create_depth_image_output(width, height, camera_count=1)

        sensor.update(state, direct_transforms, camera_rays, depth_image=direct_depth)
        render_tiled_camera_from_packet(sensor, packet, camera_rays, depth_image=packet_depth)

        np.testing.assert_allclose(direct_depth.numpy(), packet_depth.numpy(), rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
