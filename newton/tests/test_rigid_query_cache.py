# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_data import ContactData
from newton._src.sim.collide import ContactWriterData, write_contact
from newton._src.sim.contacts import Contacts
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


@wp.kernel
def _write_rigid_query_cache_kernel(writer_data: ContactWriterData):
    tid = wp.tid()

    contact_data = ContactData()
    contact_data.contact_point_center = wp.vec3(0.0, 3.0 * float(tid), 0.0)
    contact_data.contact_normal_a_to_b = wp.vec3(1.0, 0.0, 0.0)
    contact_data.contact_distance = -0.2 if tid == 0 else 0.2
    contact_data.radius_eff_a = 0.5
    contact_data.radius_eff_b = 0.5
    contact_data.margin_a = 0.0
    contact_data.margin_b = 0.0
    contact_data.shape_a = 2 * tid
    contact_data.shape_b = 2 * tid + 1
    contact_data.gap_sum = 0.0
    contact_data.contact_stiffness = 0.0
    contact_data.contact_damping = 0.0
    contact_data.contact_friction_scale = 0.0

    write_contact(contact_data, writer_data, -1)


class TestRigidQueryCache(unittest.TestCase):
    pass


def test_rigid_query_cache_retains_inactive_queries(test, device):
    with wp.ScopedDevice(device):
        contacts = Contacts(4, 0, device=device, requested_attributes={"rigid_query"})

        writer_data = ContactWriterData()
        writer_data.contact_max = contacts.rigid_contact_max
        writer_data.body_q = wp.array([wp.transform_identity()], dtype=wp.transform)
        writer_data.shape_body = wp.full(4, -1, dtype=wp.int32)
        writer_data.shape_gap = wp.zeros(4, dtype=wp.float32)
        writer_data.contact_count = contacts.rigid_contact_count
        writer_data.out_shape0 = contacts.rigid_contact_shape0
        writer_data.out_shape1 = contacts.rigid_contact_shape1
        writer_data.out_point0 = contacts.rigid_contact_point0
        writer_data.out_point1 = contacts.rigid_contact_point1
        writer_data.out_offset0 = contacts.rigid_contact_offset0
        writer_data.out_offset1 = contacts.rigid_contact_offset1
        writer_data.out_normal = contacts.rigid_contact_normal
        writer_data.out_margin0 = contacts.rigid_contact_margin0
        writer_data.out_margin1 = contacts.rigid_contact_margin1
        writer_data.out_tids = contacts.rigid_contact_tids
        writer_data.query_count = contacts.rigid_query_count
        writer_data.query_shape0 = contacts.rigid_query_shape0
        writer_data.query_shape1 = contacts.rigid_query_shape1
        writer_data.query_point0 = contacts.rigid_query_point0
        writer_data.query_point1 = contacts.rigid_query_point1
        writer_data.query_normal = contacts.rigid_query_normal
        writer_data.query_distance = contacts.rigid_query_distance
        writer_data.query_active = contacts.rigid_query_active
        writer_data.out_stiffness = wp.zeros(0, dtype=wp.float32)
        writer_data.out_damping = wp.zeros(0, dtype=wp.float32)
        writer_data.out_friction = wp.zeros(0, dtype=wp.float32)

        wp.launch(_write_rigid_query_cache_kernel, dim=2, inputs=[writer_data], device=device)

        rigid_count = int(contacts.rigid_contact_count.numpy()[0])
        query_count = int(contacts.rigid_query_count.numpy()[0])
        test.assertEqual(rigid_count, 1)
        test.assertEqual(query_count, 2)

        rigid_pairs = {
            (int(s0), int(s1))
            for s0, s1 in zip(
                contacts.rigid_contact_shape0.numpy()[:rigid_count],
                contacts.rigid_contact_shape1.numpy()[:rigid_count],
                strict=True,
            )
        }
        test.assertEqual(rigid_pairs, {(0, 1)})

        query_shape0 = contacts.rigid_query_shape0.numpy()[:query_count]
        query_shape1 = contacts.rigid_query_shape1.numpy()[:query_count]
        query_distance = contacts.rigid_query_distance.numpy()[:query_count]
        query_active = contacts.rigid_query_active.numpy()[:query_count]
        query_normal = contacts.rigid_query_normal.numpy()[:query_count]
        query_point0 = contacts.rigid_query_point0.numpy()[:query_count]
        query_point1 = contacts.rigid_query_point1.numpy()[:query_count]

        query_records = {
            (int(s0), int(s1)): {
                "distance": float(distance),
                "active": int(active),
                "normal": normal,
                "point0": point0,
                "point1": point1,
            }
            for s0, s1, distance, active, normal, point0, point1 in zip(
                query_shape0,
                query_shape1,
                query_distance,
                query_active,
                query_normal,
                query_point0,
                query_point1,
                strict=True,
            )
        }

        active_record = query_records[(0, 1)]
        inactive_record = query_records[(2, 3)]

        test.assertLess(active_record["distance"], 0.0)
        test.assertEqual(active_record["active"], 1)
        test.assertAlmostEqual(float(active_record["normal"][0]), 1.0, places=4)
        np.testing.assert_allclose(active_record["point0"], np.array([-0.4, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(active_record["point1"], np.array([0.4, 0.0, 0.0], dtype=np.float32))

        test.assertGreater(inactive_record["distance"], 0.0)
        test.assertEqual(inactive_record["active"], 0)
        test.assertAlmostEqual(float(inactive_record["normal"][0]), 1.0, places=4)
        np.testing.assert_allclose(inactive_record["point0"], np.array([-0.6, 3.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(inactive_record["point1"], np.array([0.6, 3.0, 0.0], dtype=np.float32))


add_function_test(
    TestRigidQueryCache,
    "test_rigid_query_cache_retains_inactive_queries",
    test_rigid_query_cache_retains_inactive_queries,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
