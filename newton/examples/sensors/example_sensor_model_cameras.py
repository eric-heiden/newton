# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Model Cameras
#
# Shows how to define static and body-attached cameras with ModelBuilder and
# render their output with SensorBatchedCamera.
#
# Command: python -m newton.examples sensor_model_cameras
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.sensors import SensorBatchedCamera


ORBIT_RADIUS = 4.0
ORBIT_HEIGHT = 0.4
ORBIT_SPEED = 0.75


def look_at_transform(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> wp.transform:
    """Build a camera-to-parent transform looking from a position to a target."""
    position_np = np.asarray(position, dtype=np.float32)
    forward = np.asarray(target, dtype=np.float32) - position_np
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float32))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.column_stack((right, camera_up, -forward))
    orientation = wp.quat_from_matrix(wp.mat33(*rotation.reshape(-1)))
    return wp.transform(wp.vec3(*position_np), orientation)


@wp.kernel(enable_backward=False)
def animate_orbiting_body(
    time: wp.float32,
    radius: wp.float32,
    height: wp.float32,
    angular_speed: wp.float32,
    body_index: wp.int32,
    body_q: wp.array[wp.transform],
):
    angle = angular_speed * time
    position = wp.vec3(radius * wp.cos(angle), radius * wp.sin(angle), height)
    orientation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle + 0.5 * wp.pi)
    body_q[body_index] = wp.transform(position, orientation)


@wp.kernel(enable_backward=False)
def resolve_camera_transforms(
    camera_transform: wp.array[wp.transform],
    camera_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    camera_world_transform: wp.array[wp.transform],
):
    camera_index = wp.tid()
    body_index = camera_body[camera_index]
    if body_index >= 0:
        camera_world_transform[camera_index] = wp.transform_multiply(
            body_q[body_index], camera_transform[camera_index]
        )
    else:
        camera_world_transform[camera_index] = camera_transform[camera_index]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.time = 0.0
        self.time_delta = 1.0 / 60.0
        self.sensor_width = 256
        self.sensor_height = 256

        builder = newton.ModelBuilder()
        builder.begin_world()

        # Mark the circular trajectory so the moving camera's motion is easy
        # to read from both the overview and its rendered images.
        for segment in range(24):
            angle = 2.0 * math.pi * segment / 24.0
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    (ORBIT_RADIUS * math.cos(angle), ORBIT_RADIUS * math.sin(angle), 0.025),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle + 0.5 * math.pi),
                ),
                hx=0.45,
                hy=0.08,
                hz=0.025,
                color=(0.35, 0.38, 0.42),
            )

        landmark_colors = (
            (0.93, 0.33, 0.25),
            (0.95, 0.68, 0.20),
            (0.30, 0.75, 0.35),
            (0.20, 0.65, 0.88),
            (0.45, 0.40, 0.90),
            (0.85, 0.35, 0.75),
            (0.20, 0.78, 0.72),
            (0.75, 0.55, 0.30),
        )
        for landmark, color in enumerate(landmark_colors):
            angle = 2.0 * math.pi * landmark / len(landmark_colors)
            height = 0.7 + 0.15 * (landmark % 4)
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    (6.5 * math.cos(angle), 6.5 * math.sin(angle), height), wp.quat_identity()
                ),
                hx=0.35,
                hy=0.35,
                hz=height,
                color=color,
            )

        builder.add_shape_cylinder(
            -1,
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            radius=0.8,
            half_height=1.0,
            color=(0.75, 0.78, 0.82),
        )

        self.moving_body = builder.add_body(
            xform=wp.transform((ORBIT_RADIUS, 0.0, ORBIT_HEIGHT), wp.quat_identity()),
            label="orbiting_body",
            is_kinematic=True,
        )
        builder.add_shape_box(
            self.moving_body,
            hx=0.65,
            hy=0.30,
            hz=0.20,
            color=(0.15, 0.45, 0.90),
        )
        builder.add_shape_sphere(
            self.moving_body,
            xform=wp.transform((0.65, 0.0, 0.0), wp.quat_identity()),
            radius=0.24,
            color=(1.0, 0.35, 0.12),
        )

        static_projection = newton.CameraPinhole.from_fov(math.radians(50.0), aspect=1.0, far=30.0)
        body_projection = newton.CameraPinhole.from_fov(math.radians(70.0), aspect=1.0, far=30.0)
        builder.add_camera(
            xform=look_at_transform((8.5, -8.5, 7.0), (0.0, 0.0, 0.5)),
            projection=static_projection,
            resolution=(self.sensor_width, self.sensor_height),
            label="static_overview",
        )
        builder.add_camera(
            self.moving_body,
            xform=look_at_transform((0.25, 0.0, 0.75), (1.25, 0.0, 0.75)),
            projection=body_projection,
            resolution=(self.sensor_width, self.sensor_height),
            label="body_attached",
        )
        builder.end_world()

        ground_shape = builder.add_ground_plane(color=(0.55, 0.58, 0.62))

        self.model = builder.finalize()
        self.state = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.show_cameras = True
        self.viewer.set_camera(pos=wp.vec3(10.0, -10.0, 8.0), pitch=-25.0, yaw=135.0)

        self.sensor = SensorBatchedCamera(self.model)
        self.sensor.utils.create_default_light(enable_shadows=True)
        self.sensor.utils.assign_checkerboard_material(shape_indices=[ground_shape])

        projections = self.model.camera_projections
        if not all(isinstance(projection, newton.CameraPinhole) for projection in projections):
            raise TypeError("This example requires pinhole camera projections")
        camera_fovs = [projection.fov for projection in projections]
        self.camera_rays = self.sensor.utils.compute_pinhole_camera_rays(
            self.sensor_width, self.sensor_height, camera_fovs
        )
        self.camera_indices = self.sensor.utils.create_camera_indices(
            self.model.camera_world.numpy(), self.model.camera_projection_index.numpy()
        )
        self.camera_world_transform = wp.empty(
            self.model.camera_count, dtype=wp.transform, device=self.model.device
        )
        self.color_image = self.sensor.utils.create_color_image_output(
            self.model.camera_count, self.sensor_width, self.sensor_height
        )
        self.depth_image = self.sensor.utils.create_depth_image_output(
            self.model.camera_count, self.sensor_width, self.sensor_height
        )
        self.depth_rgba = wp.empty(
            (self.model.camera_count, self.sensor_height, self.sensor_width, 4),
            dtype=wp.uint8,
            device=self.color_image.device,
        )

        self._animate_body()
        self._resolve_camera_transforms()
        self.initial_camera_world_transform = self.camera_world_transform.numpy().copy()

    def _animate_body(self):
        wp.launch(
            animate_orbiting_body,
            dim=1,
            inputs=[self.time, ORBIT_RADIUS, ORBIT_HEIGHT, ORBIT_SPEED, self.moving_body],
            outputs=[self.state.body_q],
        )

    def _resolve_camera_transforms(self):
        wp.launch(
            resolve_camera_transforms,
            dim=self.model.camera_count,
            inputs=[self.model.camera_transform, self.model.camera_body, self.state.body_q],
            outputs=[self.camera_world_transform],
        )

    def step(self):
        self._animate_body()
        self.time += self.time_delta

    def render(self):
        self.render_sensors()

        self.viewer.begin_frame(self.time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self):
        self.model.bvh_refit_shapes(self.state)
        self.model.bvh_refit_particles(self.state)
        self._resolve_camera_transforms()
        self.sensor.update(
            self.state,
            self.camera_world_transform,
            self.camera_rays,
            self.camera_indices,
            color_image=self.color_image,
            depth_image=self.depth_image,
            clear_data=SensorBatchedCamera.GRAY_CLEAR_DATA,
        )

        color_rgba = self.sensor.utils.to_rgba_from_color(self.color_image)
        self.sensor.utils.to_rgba_from_depth(self.depth_image, depth_range=(0.0, 20.0), out_buffer=self.depth_rgba)
        self.viewer.log_image("model camera color", color_rgba)
        self.viewer.log_image("model camera depth", self.depth_rgba)

    def test_final(self):
        """Verify both model cameras render and the attached camera follows its tangent-aligned body."""
        self.render_sensors()

        assert self.model.camera_label == ["static_overview", "body_attached"]
        np.testing.assert_array_equal(self.model.camera_body.numpy(), [-1, self.moving_body])
        np.testing.assert_array_equal(self.model.camera_world.numpy(), [0, 0])

        camera_world_transform = self.camera_world_transform.numpy()
        np.testing.assert_allclose(camera_world_transform[0], self.initial_camera_world_transform[0])
        assert not np.allclose(camera_world_transform[1], self.initial_camera_world_transform[1])

        body_transform = self.state.body_q.numpy()[self.moving_body]
        radial = body_transform[:2] / np.linalg.norm(body_transform[:2])
        expected_tangent = np.array((-radial[1], radial[0], 0.0), dtype=np.float32)
        body_rotation = wp.quat(*body_transform[3:])
        body_forward = np.asarray(wp.quat_rotate(body_rotation, wp.vec3(1.0, 0.0, 0.0)))
        np.testing.assert_allclose(body_forward, expected_tangent, atol=1.0e-5)

        camera_rotation = wp.quat(*camera_world_transform[1, 3:])
        camera_forward = np.asarray(wp.quat_rotate(camera_rotation, wp.vec3(0.0, 0.0, -1.0)))
        np.testing.assert_allclose(camera_forward, expected_tangent, atol=1.0e-5)

        expected_image_shape = (self.model.camera_count, self.sensor_height, self.sensor_width)
        color_image = self.color_image.numpy()
        depth_image = self.depth_image.numpy()
        assert color_image.shape == expected_image_shape
        assert depth_image.shape == expected_image_shape
        assert color_image.min() < color_image.max()
        assert depth_image.min() < depth_image.max()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
