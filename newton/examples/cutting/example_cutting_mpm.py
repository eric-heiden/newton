# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM baseline for cutting a soft cuboid with a moving knife."""

import argparse
import warnings
from dataclasses import asdict

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cutting.cutting_common import (
    AdaptiveCutSurfaceRemesher,
    CutMaterial,
    ForceHistory,
    KnifeProfile,
    SplitCuboidRenderMesh,
    add_cutting_artifact_args,
    estimate_particle_volume_from_grid,
    launch_mpm_knife_cut,
    run_cutting_example,
)
from newton.solvers import SolverImplicitMPM


class Example:
    """Particle MPM cutting baseline with a cohesive process-zone knife model."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.block_lo = np.array(args.block_lo, dtype=np.float32)
        self.block_hi = np.array(args.block_hi, dtype=np.float32)
        self.block_size = self.block_hi - self.block_lo

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
        SolverImplicitMPM.register_custom_attributes(builder)
        self._emit_particles(builder, args)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        if hasattr(self.model.mpm, "young_modulus"):
            self.model.mpm.young_modulus.fill_(args.young_modulus)
        if hasattr(self.model.mpm, "poisson_ratio"):
            self.model.mpm.poisson_ratio.fill_(args.poisson_ratio)
        if hasattr(self.model.mpm, "damping"):
            self.model.mpm.damping.fill_(args.mpm_damping)
        if hasattr(self.model.mpm, "friction"):
            self.model.mpm.friction.fill_(args.friction)
        if hasattr(self.model.mpm, "tensile_yield_ratio"):
            self.model.mpm.tensile_yield_ratio.fill_(1.0)

        options = SolverImplicitMPM.Config()
        options.warmstart_mode = "particles"
        for key in vars(args):
            if hasattr(options, key):
                setattr(options, key, getattr(args, key))
        if not wp.get_device().is_cuda:
            options.grid_type = "dense"
        self.solver = SolverImplicitMPM(self.model, options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.damage = wp.zeros(self.model.particle_count, dtype=wp.float32, device=self.model.device)
        self.particle_colors = wp.full(
            self.model.particle_count, wp.vec3(0.15, 0.48, 0.86), dtype=wp.vec3, device=self.model.device
        )
        self.cut_accum = wp.zeros(3, dtype=wp.float32, device=self.model.device)
        self.particle_volume = estimate_particle_volume_from_grid(tuple(self.block_size), self.model.particle_count)
        self.remesh_history: list[dict[str, float]] = []
        self.render_rest_particle_q_wp = wp.clone(self.state_0.particle_q) if args.render_split_mesh else None
        self.render_split_mesh = None
        if args.render_split_mesh and args.render_remesh_mode == "adaptive":
            self.render_split_mesh = AdaptiveCutSurfaceRemesher(
                self.block_lo,
                self.block_hi,
                self.knife,
                max_gap=args.render_gap,
                base_segments=args.adaptive_remesh_base_segments,
                refine_factor=args.adaptive_remesh_refine_factor,
                refine_band=args.adaptive_remesh_refine_band,
                height_segments=args.adaptive_remesh_height_segments,
            )
        elif args.render_split_mesh:
            self.render_split_mesh = SplitCuboidRenderMesh(
                self.block_lo,
                self.block_hi,
                self.knife,
                max_gap=args.render_gap,
                segments=args.render_mesh_segments,
            )
        self.render_rest_particle_q = self.state_0.particle_q.numpy() if args.render_split_mesh else None
        self.render_particle_radius = float(np.mean(self.model.particle_radius.numpy())) * args.render_particle_scale

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.15, -1.35, 0.82), pitch=-20.0, yaw=130.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                center = 0.5 * (self.block_lo + self.block_hi)
                self.viewer.camera.look_at(wp.vec3(float(center[0]), float(center[1]), float(center[2])))

    def _emit_particles(self, builder: newton.ModelBuilder, args):
        particles_per_cell = args.particles_per_cell
        voxel_size = args.voxel_size
        particle_res = np.maximum(
            np.ceil(particles_per_cell * self.block_size / voxel_size).astype(int),
            2,
        )
        cell_size = self.block_size / (particle_res - 1)
        cell_volume = float(np.prod(cell_size))
        radius = 0.42 * float(np.cbrt(cell_volume))
        mass = cell_volume * args.density

        builder.add_particle_grid(
            pos=wp.vec3(self.block_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(particle_res[0]),
            dim_y=int(particle_res[1]),
            dim_z=int(particle_res[2]),
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=float(mass),
            jitter=0.0,
            radius_mean=radius,
            custom_attributes={"mpm:friction": args.friction},
        )

    def simulate(self):
        frame_force = 0.0
        frame_active = 0.0
        frame_damage = 0.0
        for substep in range(self.sim_substeps):
            substep_time = self.sim_time + substep * self.sim_dt
            launch_mpm_knife_cut(
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

            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=self.sim_dt)
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
        if self.render_split_mesh is not None:
            if isinstance(self.render_split_mesh, AdaptiveCutSurfaceRemesher):
                stats = self.render_split_mesh.log(
                    self.viewer,
                    self.model.device,
                    self.sim_time,
                    rest_particle_points=self.render_rest_particle_q_wp,
                    particle_points=self.state_0.particle_q,
                )
                self.remesh_history.append({"time_s": self.sim_time, **asdict(stats)})
            else:
                self.render_split_mesh.log(
                    self.viewer,
                    self.model.device,
                    self.sim_time,
                    rest_particle_points=self.render_rest_particle_q,
                    particle_points=self.state_0.particle_q.numpy(),
                )
        else:
            self.viewer.log_state(self.state_0)
        if self.render_particle_radius > 0.0:
            self.viewer.log_points(
                name="/model/particles",
                points=self.state_0.particle_q,
                radii=self.render_particle_radius,
                colors=self.particle_colors,
            )
        else:
            self.viewer.log_points(name="/model/particles", points=None, hidden=True)
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
        newton.examples.test_particle_state(
            self.state_0,
            "all particles have finite positions",
            lambda q, _qd: wp.length(q) < 10.0,
        )
        if not self.force_history.forces:
            raise ValueError("cutting MPM example did not record a force profile")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_cutting_artifact_args(parser)

        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=2)
        parser.add_argument("--block-lo", type=float, nargs=3, default=[-0.45, -0.22, 0.06])
        parser.add_argument("--block-hi", type=float, nargs=3, default=[0.45, 0.22, 0.42])
        parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81])
        parser.add_argument("--density", type=float, default=950.0)
        parser.add_argument("--particles-per-cell", type=int, default=2)
        parser.add_argument("--render-split-mesh", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--render-remesh-mode", type=str, default="adaptive", choices=["adaptive", "split"])
        parser.add_argument("--render-gap", type=float, default=0.14)
        parser.add_argument("--render-mesh-segments", type=int, default=56)
        parser.add_argument("--render-particle-scale", type=float, default=0.35)
        parser.add_argument("--adaptive-remesh-base-segments", type=int, default=24)
        parser.add_argument("--adaptive-remesh-refine-factor", type=int, default=4)
        parser.add_argument("--adaptive-remesh-refine-band", type=float, default=0.13)
        parser.add_argument("--adaptive-remesh-height-segments", type=int, default=6)

        parser.add_argument("--knife-start-x", type=float, default=-0.55)
        parser.add_argument("--knife-speed", type=float, default=0.75)
        parser.add_argument("--knife-center-y", type=float, default=0.0)
        parser.add_argument("--knife-center-z", type=float, default=0.24)
        parser.add_argument("--knife-half-width-y", type=float, default=0.055)
        parser.add_argument("--knife-half-width-z", type=float, default=0.24)
        parser.add_argument("--process-width", type=float, default=0.055)

        parser.add_argument("--fracture-energy", type=float, default=95.0)
        parser.add_argument("--yield-stress", type=float, default=1.8e4)
        parser.add_argument("--damage-damping", type=float, default=0.03)
        parser.add_argument("--max-damage-rate", type=float, default=13.0)
        parser.add_argument("--separation-speed", type=float, default=0.34)
        parser.add_argument("--force-scale", type=float, default=1.0)

        parser.add_argument("--young-modulus", type=float, default=7.0e4)
        parser.add_argument("--poisson-ratio", type=float, default=0.42)
        parser.add_argument("--mpm-damping", type=float, default=0.015)
        parser.add_argument("--friction", type=float, default=0.55)
        parser.add_argument("--voxel-size", type=float, default=0.075)
        parser.add_argument("--grid-type", type=str, default="sparse", choices=["sparse", "dense", "fixed"])
        parser.add_argument("--max-iterations", type=int, default=80)
        parser.add_argument("--tolerance", type=float, default=1.0e-5)
        parser.add_argument("--solver", type=str, default="auto")
        parser.add_argument("--integration-scheme", type=str, default="pic", choices=["pic", "gimp"])
        parser.add_argument("--velocity-basis", type=str, default="Q1")
        parser.add_argument("--strain-basis", type=str, default="P0")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.artifact_dir:
        run_cutting_example(example, args, "mpm")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            newton.examples.run(example, args)
