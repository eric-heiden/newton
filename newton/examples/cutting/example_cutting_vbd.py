# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mesh-based VBD cutting track for a soft cuboid and moving knife."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cutting.cutting_common import (
    CutMaterial,
    ForceHistory,
    KnifeProfile,
    add_cutting_artifact_args,
    estimate_particle_volume_from_grid,
    launch_cut_tet_degradation,
    launch_vbd_knife_cut,
    run_cutting_example,
)


class Example:
    """VBD soft-grid cutting track with cohesive damage and tet softening."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = args.iterations
        self.sim_time = 0.0

        self.block_pos = np.array(args.block_pos, dtype=np.float32)
        self.cell = np.array([args.cell_x, args.cell_y, args.cell_z], dtype=np.float32)
        self.dims = np.array([args.dim_x, args.dim_y, args.dim_z], dtype=np.int32)
        self.block_size = self.cell * self.dims

        self.knife = KnifeProfile(
            start_x=args.knife_start_x,
            speed=args.knife_speed,
            center_y=args.knife_center_y,
            center_z=args.knife_center_z,
            half_width_y=args.knife_half_width_y,
            half_width_z=args.knife_half_width_z,
            process_width=args.process_width,
        )
        self.material = CutMaterial(
            fracture_energy=args.fracture_energy,
            yield_stress=args.yield_stress,
            damping=args.damage_damping,
            max_damage_rate=args.max_damage_rate,
            separation_speed=args.separation_speed,
            force_scale=args.force_scale,
        )
        self.force_history = ForceHistory()

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_soft_grid(
            pos=wp.vec3(self.block_pos),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            dim_z=args.dim_z,
            cell_x=args.cell_x,
            cell_y=args.cell_y,
            cell_z=args.cell_z,
            density=args.density,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            particle_radius=args.particle_radius,
        )
        builder.color()

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)
        self.model.soft_contact_ke = args.soft_contact_ke
        self.model.soft_contact_kd = args.soft_contact_kd
        self.model.soft_contact_mu = args.soft_contact_mu

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=args.self_contact,
            particle_enable_tile_solve=args.tile_solve,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.damage = wp.zeros(self.model.particle_count, dtype=wp.float32, device=self.model.device)
        self.particle_colors = wp.full(
            self.model.particle_count, wp.vec3(0.15, 0.48, 0.86), dtype=wp.vec3, device=self.model.device
        )
        self.cut_accum = wp.zeros(3, dtype=wp.float32, device=self.model.device)
        self.base_tet_materials = wp.clone(self.model.tet_materials) if self.model.tet_materials is not None else None
        self.particle_volume = estimate_particle_volume_from_grid(tuple(self.block_size), self.model.particle_count)
        self.damage_threshold = args.tet_damage_threshold
        self.residual_stiffness = args.residual_stiffness

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.15, -1.35, 0.82), pitch=-20.0, yaw=130.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                center = self.block_pos + 0.5 * self.block_size
                self.viewer.camera.look_at(wp.vec3(float(center[0]), float(center[1]), float(center[2])))

    def simulate(self):
        frame_force = 0.0
        frame_active = 0.0
        frame_damage = 0.0
        for substep in range(self.sim_substeps):
            substep_time = self.sim_time + substep * self.sim_dt
            self.state_0.clear_forces()

            launch_vbd_knife_cut(
                self.state_0,
                self.damage,
                self.particle_colors,
                self.cut_accum,
                self.knife,
                self.material,
                self.sim_dt,
                self.particle_volume,
                substep_time,
                self.model.device,
            )
            values = self.cut_accum.numpy()
            frame_force += float(values[0])
            frame_active += float(values[1])
            frame_damage = float(values[2]) / max(float(self.model.particle_count), 1.0)

            if self.base_tet_materials is not None:
                launch_cut_tet_degradation(
                    self.model,
                    self.damage,
                    self.base_tet_materials,
                    damage_threshold=self.damage_threshold,
                    residual_stiffness=self.residual_stiffness,
                )

            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.force_history.append_values(
            self.sim_time,
            frame_force / max(float(self.sim_substeps), 1.0),
            frame_active / max(float(self.sim_substeps), 1.0),
            frame_damage,
        )

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_points(
            name="/cutting/vbd_damage_particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
        )
        starts, ends, colors = self.knife.blade_segments(self.sim_time)
        self.viewer.log_lines(
            "/cutting/knife",
            wp.array(starts, dtype=wp.vec3, device=self.model.device),
            wp.array(ends, dtype=wp.vec3, device=self.model.device),
            wp.array(colors, dtype=wp.vec3, device=self.model.device),
            width=0.018,
        )
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-2.0, -2.0, -1.0)
        p_upper = wp.vec3(2.0, 2.0, 2.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles remain finite and near the cutting scene",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        if not self.force_history.forces:
            raise ValueError("cutting VBD example did not record a force profile")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_cutting_artifact_args(parser)

        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=8)
        parser.add_argument("--block-pos", type=float, nargs=3, default=[-0.45, -0.22, 0.06])
        parser.add_argument("--dim-x", type=int, default=14)
        parser.add_argument("--dim-y", type=int, default=7)
        parser.add_argument("--dim-z", type=int, default=6)
        parser.add_argument("--cell-x", type=float, default=0.064)
        parser.add_argument("--cell-y", type=float, default=0.064)
        parser.add_argument("--cell-z", type=float, default=0.058)
        parser.add_argument("--density", type=float, default=950.0)
        parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81])
        parser.add_argument("--particle-radius", type=float, default=0.018)

        parser.add_argument("--knife-start-x", type=float, default=-0.55)
        parser.add_argument("--knife-speed", type=float, default=0.75)
        parser.add_argument("--knife-center-y", type=float, default=0.0)
        parser.add_argument("--knife-center-z", type=float, default=0.24)
        parser.add_argument("--knife-half-width-y", type=float, default=0.06)
        parser.add_argument("--knife-half-width-z", type=float, default=0.24)
        parser.add_argument("--process-width", type=float, default=0.06)

        parser.add_argument("--fracture-energy", type=float, default=95.0)
        parser.add_argument("--yield-stress", type=float, default=1.8e4)
        parser.add_argument("--damage-damping", type=float, default=0.03)
        parser.add_argument("--max-damage-rate", type=float, default=12.0)
        parser.add_argument("--separation-speed", type=float, default=0.26)
        parser.add_argument("--force-scale", type=float, default=0.42)
        parser.add_argument("--tet-damage-threshold", type=float, default=0.18)
        parser.add_argument("--residual-stiffness", type=float, default=0.08)

        parser.add_argument("--k-mu", type=float, default=5.5e4)
        parser.add_argument("--k-lambda", type=float, default=9.5e4)
        parser.add_argument("--k-damp", type=float, default=2.0e-3)
        parser.add_argument("--soft-contact-ke", type=float, default=1.0e3)
        parser.add_argument("--soft-contact-kd", type=float, default=1.0)
        parser.add_argument("--soft-contact-mu", type=float, default=0.8)
        parser.add_argument("--self-contact", action="store_true")
        parser.add_argument("--tile-solve", action="store_true")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.artifact_dir:
        run_cutting_example(example, args, "vbd")
    else:
        newton.examples.run(example, args)
