# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collapse a water column in isolated tanks using XPBD or implicit MPM.

Both solvers use the same particles and static colliders. Their material and
iteration parameters have different meanings; the example supplies separate
backend settings. World coordinates overlap in physics and are offset only
by the viewer, exercising solver isolation rather than spatial separation.
The MPM path additionally projects drifted particles into the rectangular
tank, as in the fluid_water_surface example; XPBD uses particle contacts.

Command: python -m newton.examples fluid_dam_break --solver xpbd --world-count 4
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.fluid.utils import (
    FluidParticleRenderer,
    add_tank_walls,
    parse_particle_count,
    resolve_particle_grid,
    step_simulation,
    validate_simulation_args,
)
from newton.solvers import SolverImplicitMPM, SolverXPBD


@wp.kernel
def _confine_tank(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    half_x: float,
    half_y: float,
):
    i = wp.tid()
    q = positions[i]
    qd = velocities[i]
    for axis in range(3):
        lower = float(0.0)
        upper = float(1.0e10)
        if axis == 0:
            lower, upper = -half_x, half_x
        elif axis == 1:
            lower, upper = -half_y, half_y
        if q[axis] < lower:
            q[axis] = lower
            qd[axis] = wp.max(qd[axis], 0.0)
        elif q[axis] > upper:
            q[axis] = upper
            qd[axis] = wp.min(qd[axis], 0.0)
    positions[i] = q
    velocities[i] = qd


class Example:
    def __init__(self, viewer, args):
        validate_simulation_args(args, supported_solvers=("xpbd", "mpm"))
        if args.world_count < 1:
            raise ValueError("world_count must be positive")
        if args.solver == "mpm" and not wp.get_device().is_cuda:
            raise ValueError("The MPM backend requires a CUDA device")
        self.viewer = viewer
        self.backend = args.solver
        self.world_count = args.world_count
        self.frame_dt = 1.0 / args.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.half_x, self.half_y = 1.2, 0.5

        grid = resolve_particle_grid(args.particle_count, (0.65, 0.86, 0.85), 0.035)
        spacing = grid.spacing
        radius = grid.radius
        world = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        if self.backend == "mpm":
            SolverImplicitMPM.register_custom_attributes(world)
        flags = newton.ParticleFlags.ACTIVE
        if self.backend == "xpbd":
            flags |= newton.ParticleFlags.FLUID
        world.add_particle_grid(
            pos=wp.vec3(-self.half_x + spacing, -0.5 * (grid.dimensions[1] - 1) * spacing, spacing),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=grid.dimensions[0],
            dim_y=grid.dimensions[1],
            dim_z=grid.dimensions[2],
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=args.rest_density * spacing**3,
            jitter=0.05 * spacing,
            radius_mean=radius,
            flags=flags,
        )
        add_tank_walls(world, self.half_x, self.half_y, 1.2, 0.08, (0.6, 0.7, 0.8), 0.25)
        # Use a finite floor per world so both backends receive local colliders.
        world.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
            hx=self.half_x + 0.08,
            hy=self.half_y + 0.08,
            hz=0.05,
            color=(0.35, 0.4, 0.45),
        )
        builder = newton.ModelBuilder()
        builder.replicate(world, self.world_count)
        self.model = builder.finalize()
        self.initial_max_x = float(self.model.particle_q.numpy()[:, 0].max())

        if self.backend == "xpbd":
            self.model.particle_max_velocity = 0.5 * radius / self.sim_dt
            self.model.soft_contact_mu = 0.05
            self.solver = SolverXPBD(
                self.model,
                iterations=args.iterations,
                fluid_rest_distance=spacing,
                fluid_cohesion=0.5,
                fluid_viscosity=args.xpbd_viscosity,
            )
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
        else:
            self.model.mpm.friction.fill_(0.0)
            self.model.mpm.tensile_yield_ratio.fill_(1.0)
            self.model.mpm.viscosity.fill_(args.mpm_viscosity)
            config = SolverImplicitMPM.Config()
            config.voxel_size = 2.0 * spacing
            config.max_iterations = args.mpm_iterations
            config.tolerance = 1.0e-3
            config.warmstart_mode = "particles"
            config.collider_basis = "pic"
            config.separate_worlds = self.world_count > 1
            self.solver = SolverImplicitMPM(self.model, config=config)
            self.solver.setup_collider(collider_projection_threshold=[0.0] * self.world_count)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        if self.backend == "mpm":
            self.state_0.mpm.particle_Jp.fill_(1.0)
        self.viewer.set_model(self.model)
        self.particle_renderer = FluidParticleRenderer(self.model)
        self.viewer.show_particles = True
        self.viewer.set_world_offsets((2.8, 1.4, 0.0))
        scale = max(1.0, np.sqrt(self.world_count))
        self.viewer.set_camera(pos=wp.vec3(2.8 * scale, -3.5 * scale, 2.3 * scale), pitch=-25.0, yaw=130.0)
        self.graph = None
        # MPM's dynamically allocated sparse grid runs eagerly; XPBD reserves
        # all buffers and captures the complete frame without host readbacks.
        self.use_cuda_graph = self.backend == "xpbd" and self.model.device.is_cuda

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.backend == "xpbd":
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            else:
                self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
                self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
                # At coarse MPM resolutions, interpolation can let particles
                # drift through thin box faces. Preserve the closed tank with
                # the same analytic containment used by fluid_water_surface.
                wp.launch(
                    _confine_tank,
                    dim=self.model.particle_count,
                    inputs=[self.state_1.particle_q, self.state_1.particle_qd, self.half_x, self.half_y],
                    device=self.model.device,
                )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        step_simulation(self)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.particle_renderer.log_state(self.viewer, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite state, tank containment, and spreading in every world."""
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise ValueError("Fluid state contains non-finite values")
        tolerance = 2.0 * self.model.particle_max_radius
        if np.any(q[:, 2] < -tolerance):
            raise ValueError("Fluid penetrated the tank floor")
        if np.any(np.abs(q[:, 0]) > self.half_x + tolerance) or np.any(np.abs(q[:, 1]) > self.half_y + tolerance):
            raise ValueError("Fluid escaped the tank walls")
        if self.sim_time > 1.0:
            positions = q.reshape(self.world_count, -1, 3)
            if np.any(positions[:, :, 0].max(axis=1) < self.initial_max_x + 0.2):
                raise ValueError("A water column did not spread")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--solver", choices=["xpbd", "mpm"], default="xpbd")
        parser.add_argument("--particle-count", type=parse_particle_count, default=10_000, help="Particles per world.")
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=2, help="XPBD iterations per substep.")
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--xpbd-viscosity", type=float, default=0.03, help="Dimensionless XSPH blend.")
        parser.add_argument("--mpm-viscosity", type=float, default=0.0, help="MPM dynamic viscosity [Pa s].")
        parser.add_argument("--mpm-iterations", type=int, default=50)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
