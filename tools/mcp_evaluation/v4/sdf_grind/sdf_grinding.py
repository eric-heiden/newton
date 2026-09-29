# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDF Grinding
#
# A cylindrical grinding wheel is driven kinematically across an ellipsoidal
# workpiece. The workpiece is a mesh shape whose collision geometry is a sparse
# texture SDF (``workpiece_sdf``, built by ``Mesh.build_sdf``); hydroelastic
# SDF-SDF collision reports the contact surface and normal load every frame,
# without a dynamics solver. The rendered workpiece surface is extracted from
# the same SDF.
#
# Command: python sdf_grinding.py
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples

WORKPIECE_RADII = (0.45, 0.25, 0.12)
WORKPIECE_RESOLUTION = 256
GRINDER_RADIUS = 0.13
GRINDER_HALF_WIDTH = 0.04
GRIND_DEPTH = 0.035
GRIND_FRAMES = 270
HYDROELASTIC_STIFFNESS = 1.0e8


@wp.kernel(enable_backward=False)
def _compute_normal_load(
    contact_count: wp.array[wp.int32],
    contact_distance: wp.array[wp.float32],
    contact_stiffness: wp.array[wp.float32],
    normal_load: wp.array[wp.float32],
):
    contact = wp.tid()
    if contact < contact_count[0]:
        normal_load[contact] = wp.max(-contact_distance[contact] * contact_stiffness[contact], 0.0)
    else:
        normal_load[contact] = 0.0


class Example:
    def __init__(self, viewer, _args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.frame = 0

        # A gently curved blank keeps fine SDF subgrids resident around the
        # entire machined surface, which makes this compact in-place demo
        # independent of Newton's linear-subgrid storage optimization.
        workpiece_mesh = newton.Mesh.create_ellipsoid(
            *WORKPIECE_RADII,
            num_latitudes=32,
            num_longitudes=64,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        self.workpiece_sdf = workpiece_mesh.build_sdf(
            max_resolution=WORKPIECE_RESOLUTION,
            narrow_band_range=(-0.08, 0.08),
            margin=0.04,
            texture_format="float32",
            paired_samples=False,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.005,
            kh=HYDROELASTIC_STIFFNESS,
            density=0.0,
            is_hydroelastic=True,
        )
        workpiece_cfg = shape_cfg.copy()
        workpiece_cfg.is_visible = False
        grinder_cfg = shape_cfg.copy()
        grinder_cfg.sdf_max_resolution = 64
        grinder_cfg.sdf_narrow_band_range = (-0.06, 0.06)
        grinder_cfg.sdf_padding = 0.01

        builder = newton.ModelBuilder()
        builder.sdf_texture_paired_samples = False
        builder.add_shape_mesh(
            body=-1,
            mesh=workpiece_mesh,
            cfg=workpiece_cfg,
            label="workpiece",
        )

        initial_pose = self._grinder_pose(0)
        self.grinder_body = builder.add_body(xform=initial_pose, label="grinder")
        builder.add_shape_cylinder(
            body=self.grinder_body,
            radius=GRINDER_RADIUS,
            half_height=GRINDER_HALF_WIDTH,
            cfg=grinder_cfg,
            color=(0.18, 0.20, 0.23),
            opacity=0.4,
            label="grinding_wheel",
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            sdf_hydroelastic_config=newton.geometry.HydroelasticSDF.Config(
                output_contact_surface=True,
            ),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.contact_surface = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
        self.contact_distance = wp.empty(
            self.contacts.rigid_contact_max,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.normal_load = wp.empty_like(self.contact_distance)
        self.total_normal_load = 0.0

        self.body_q = self.state_0.body_q.numpy()
        self._set_grinder_pose(initial_pose)
        self._update_workpiece_surface()

        self.viewer.set_model(self.model)
        self.viewer.show_hydro_contact_surface = True
        self.viewer.set_camera(pos=wp.vec3(0.8, -0.85, 0.5), pitch=-18.0, yaw=132.0)

    @staticmethod
    def _grinder_pose(frame: int) -> wp.transform:
        x_start = -0.36
        x_end = 0.36
        x = x_start + min(frame / GRIND_FRAMES, 1.0) * (x_end - x_start)
        radial_fraction = min((x / WORKPIECE_RADII[0]) ** 2, 1.0)
        surface_z = WORKPIECE_RADII[2] * math.sqrt(1.0 - radial_fraction)
        z = surface_z + GRINDER_RADIUS - GRIND_DEPTH
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
        return wp.transform(wp.vec3(x, 0.0, z), rotation)

    def _set_grinder_pose(self, pose: wp.transform) -> None:
        self.body_q[self.grinder_body] = np.asarray(pose)
        self.state_0.body_q.assign(self.body_q)

    def _update_workpiece_surface(self) -> None:
        surface = self.workpiece_sdf.extract_isomesh(device=self.model.device)
        if surface is None:
            raise RuntimeError("Grinding removed the complete workpiece.")
        self.workpiece_points = wp.array(surface.vertices, dtype=wp.vec3, device=self.model.device)
        self.workpiece_indices = wp.array(surface.indices.reshape(-1), dtype=wp.int32, device=self.model.device)

    def step(self):
        self.frame += 1
        grinder_pose = self._grinder_pose(self.frame)
        self._set_grinder_pose(grinder_pose)

        # Run hydroelastic SDF-SDF collision before editing the workpiece.
        self.collision_pipeline.collide(self.state_0, self.contacts)
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state_0,
            self.contacts,
            out_distance=self.contact_distance,
        )
        wp.launch(
            _compute_normal_load,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contact_distance,
                self.contacts.rigid_contact_stiffness,
                self.normal_load,
            ],
            device=self.model.device,
        )
        self.total_normal_load = float(wp.utils.array_sum(self.normal_load))
        # TODO: remove the material the grinding wheel sweeps through.
        self._update_workpiece_surface()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/workpiece_sdf",
            self.workpiece_points,
            self.workpiece_indices,
            color=(0.55, 0.62, 0.68),
            roughness=0.65,
            metallic=0.25,
            dynamic=True,
        )
        self.viewer.log_hydro_contact_surface(self.contact_surface)
        self.viewer.log_scalar("Hydroelastic normal load [N]", self.total_normal_load)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
