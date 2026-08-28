# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD Cloth Self Contact
#
# A folded cloth patch starts with two overlapping layers. XPBD separates
# the layers without launching isolated particles or introducing springs.
#
# Command: python -m newton.examples cloth_xpbd_self_contact
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


def create_folded_grid(dim_x: int, dim_y: int, cell: float, fold_segments: int):
    """Create flat material coordinates and a rounded, isometric fold."""
    rest_vertices = []
    initial_vertices = []
    indices = []
    fold_start = (dim_y + 1 - fold_segments) // 2
    for y in range(dim_y + 1):
        if y <= fold_start:
            folded_position_y = y * cell
            height = 0.0
        else:
            folded_position_y = fold_start * cell
            height = 0.0
            transition_segments = min(y - fold_start, fold_segments)
            for segment in range(1, transition_segments + 1):
                angle = np.pi * segment / fold_segments
                folded_position_y += cell * np.cos(angle)
                height += cell * np.sin(angle)
            folded_position_y -= max(y - fold_start - fold_segments, 0) * cell
        for x in range(dim_x + 1):
            rest_vertices.append(wp.vec3(x * cell, y * cell, 0.0))
            initial_vertices.append(wp.vec3(x * cell, folded_position_y, height))

    stride = dim_x + 1
    for y in range(dim_y):
        for x in range(dim_x):
            v0 = y * stride + x
            v1 = v0 + 1
            v2 = v1 + stride
            v3 = v0 + stride
            indices.extend((v0, v1, v3, v1, v2, v3))
    return rest_vertices, initial_vertices, indices


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.contact_friction = 1.5
        self.bending_stiffness = 0.002

        self.dim_x = 48
        self.dim_y = 48
        cloth_position = np.array((-0.64, -0.32, 0.8), dtype=np.float32)
        self.cell_size = 1.28 / self.dim_x
        self.fold_segments = 3
        rest_vertices, initial_vertices, indices = create_folded_grid(
            self.dim_x, self.dim_y, self.cell_size, self.fold_segments
        )
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(cloth_position),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=rest_vertices,
            indices=indices,
            density=0.2,
            tri_ke=8.0e3,
            tri_ka=8.0e3,
            tri_kd=0.0,
            edge_ke=self.bending_stiffness,
            edge_kd=0.0,
            particle_radius=0.015,
        )
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.mu = self.contact_friction
        ground_cfg.ka = 0.0
        builder.add_ground_plane(cfg=ground_cfg)

        self.model = builder.finalize()
        self.model.particle_mu = self.contact_friction
        self.model.soft_contact_mu = self.contact_friction
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=6,
            particle_enable_self_contact=True,
            particle_enable_triangle_intersection_recovery=True,
            soft_contact_max_depenetration_velocity=0.3,
            particle_self_contact_relaxation=0.4,
            particle_triangle_intersection_relaxation=0.3,
            particle_self_contact_radius=0.026,
            particle_self_contact_margin=0.033,
            particle_max_depenetration_velocity=0.3,
            particle_vertex_contact_buffer_size=96,
            particle_edge_contact_buffer_size=192,
            particle_triangle_contact_buffer_size=96,
            particle_topological_contact_filter_threshold=2,
            particle_damping=0.0,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.04,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        folded_positions = np.asarray(initial_vertices, dtype=np.float32) + cloth_position
        self.state_0.particle_q.assign(folded_positions)
        self.state_1.particle_q.assign(folded_positions)
        self.control = self.model.control()
        self.initial_positions = self.state_0.particle_q.numpy().copy()
        self.maximum_bending_rest_angle = float(np.max(np.abs(self.model.edge_rest_angle.numpy())))
        rest_positions = self.model.particle_q.numpy()
        triangles = self.model.tri_indices.numpy()
        edge_length_errors = []
        for start, end in ((0, 1), (1, 2), (2, 0)):
            initial_lengths = np.linalg.norm(
                self.initial_positions[triangles[:, start]] - self.initial_positions[triangles[:, end]],
                axis=1,
            )
            rest_lengths = np.linalg.norm(
                rest_positions[triangles[:, start]] - rest_positions[triangles[:, end]],
                axis=1,
            )
            edge_length_errors.append(np.abs(initial_lengths - rest_lengths))
        self.maximum_initial_edge_length_error = float(np.max(edge_length_errors))

        stride = self.dim_x + 1
        fold_start = (self.dim_y + 1 - self.fold_segments) // 2
        lower_rows = np.arange(0, fold_start * stride, dtype=np.int32).reshape(-1, stride)
        upper_start = fold_start + self.fold_segments
        upper_rows = np.arange(upper_start * stride, (self.dim_y + 1) * stride, dtype=np.int32)
        upper_rows = upper_rows.reshape(-1, stride)[::-1]
        self.layer_pairs = np.stack((lower_rows.ravel(), upper_rows.ravel()), axis=1)

        self.viewer.set_model(self.model)
        self.viewer.configure_picking(
            particle_pick_radius=0.28,
            particle_pick_stiffness=3600.0,
            particle_pick_damping=120.0,
            particle_pick_max_acceleration=100.0,
        )
        self.viewer.set_camera(wp.vec3(2.2, -2.8, 2.0), -18.0, 135.0)
        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Separate folded layers without encoding a permanent material crease."""
        detector = self.solver.trimesh_collision_detector
        detector.refit(self.state_0.particle_q)
        detector.triangle_triangle_intersection_detection()
        intersection_count = int(detector.triangle_intersecting_triangles_count.numpy().sum() // 2)
        overflow_flags = detector.resize_flags.numpy()
        positions = self.state_0.particle_q.numpy()
        speeds = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
        pair_distances = np.linalg.norm(positions[self.layer_pairs[:, 0]] - positions[self.layer_pairs[:, 1]], axis=1)
        speed_p99 = float(np.percentile(speeds, 99.0))
        speed_max = float(np.max(speeds))
        assert self.maximum_bending_rest_angle < 1.0e-5, "the temporary fold must not become the bending rest pose"
        assert self.maximum_initial_edge_length_error < 1.0e-5, "the initial fold must not pre-strain the cloth"
        assert intersection_count == 0, f"folded cloth must not contain triangle intersections ({intersection_count})"
        assert not np.any(overflow_flags), f"self-contact candidate buffers must not overflow ({overflow_flags})"
        pair_distance_p25 = float(np.percentile(pair_distances, 25.0))
        assert pair_distance_p25 > 0.03, f"folded layers must remain separated (p25 {pair_distance_p25:.4f} m)"
        assert speed_p99 < 0.06, f"folded cloth must settle without sparse jitter (p99 {speed_p99:.4f} m/s)"
        assert speed_max < 0.1, f"no individual cloth particle may remain unstable (max {speed_max:.4f} m/s)"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
