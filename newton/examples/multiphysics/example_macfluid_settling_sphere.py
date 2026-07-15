# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MAC Fluid Settling Sphere
#
# A rigid sphere with a density different from the fluid settles (or
# rises) inside a closed water tank. MuJoCo owns the rigid body and the
# staggered MAC-grid fluid solver treats it as a moving immersed boundary,
# feeding buoyancy and drag wrenches back through the experimental proxy
# coupling. The example records position, velocity, hydrodynamic force,
# divergence, and coupling diagnostics per frame.
#
# Command: python -m newton.examples macfluid_settling_sphere
#          python -m newton.examples macfluid_settling_sphere --sphere-density 500
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.macfluid_demo_utils import (
    FluidSliceVisualizer,
    MetricsRecorder,
    add_macfluid_args,
    capture_frame_graph,
    fluid_body_wrenches,
    log_tank_outline,
    make_coupled_fluid_solver,
)
from newton.solvers import SolverMACFluid, SolverMuJoCo

TANK = 1.0  # cubic tank edge length [m]
RADIUS = 0.12  # sphere radius [m]
FLUID_DENSITY = 1000.0


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sphere_density = args.sphere_density
        rising = self.sphere_density < FLUID_DENSITY
        start_z = 0.25 * TANK if rising else 0.8 * TANK

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        volume = 4.0 / 3.0 * np.pi * RADIUS**3
        self.sphere_mass = self.sphere_density * volume
        self.sphere = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, start_z), wp.quat_identity()))
        # mass and inertia follow from the shape density
        builder.add_shape_sphere(
            self.sphere, radius=RADIUS, cfg=newton.ModelBuilder.ShapeConfig(density=self.sphere_density)
        )

        # tank floor for rigid contact (a plane is invisible to the fluid;
        # the fluid's closed-domain wall coincides with it at z = 0)
        builder.add_ground_plane()
        if rising:
            # rigid lid overlapping the top fluid layers keeps a buoyant
            # sphere fully submerged (the sealed incompressible tank cannot
            # lose volume, so the sphere must stop below the fluid ceiling)
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.0, 0.0, TANK), wp.quat_identity()),
                hx=0.5 * TANK,
                hy=0.5 * TANK,
                hz=0.08,
            )

        self.model = builder.finalize()

        res = int(args.fluid_res)
        fluid_cfg = SolverMACFluid.Config(
            resolution=(res, res, res),
            cell_size=TANK / res,
            origin=(-0.5 * TANK, -0.5 * TANK, 0.0),
            density=FLUID_DENSITY,
            kinematic_viscosity=args.viscosity,
            pressure_iterations=args.pressure_iterations,
        )

        # a buoyant sphere pressed against the lid needs damped coupling
        # feedback (squeeze pressure and contact force are both stiff)
        if rising and args.proxy_relaxation_mode == "fixed" and args.proxy_relaxation == 1.0:
            args.proxy_relaxation = 0.5
            args.proxy_relaxation_mode = "aitken"

        self.dry = args.dry
        if self.dry:
            self.solver = SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=50)
            self.fluid = None
        else:
            self.solver, self.fluid = make_coupled_fluid_solver(
                self.model,
                fluid_cfg,
                rigid_bodies=[self.sphere],
                joints=list(range(self.model.joint_count)),
                args=args,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.metrics = MetricsRecorder(args.metrics_output)
        self.metrics.meta = {
            "example": "macfluid_settling_sphere",
            "sphere_density": self.sphere_density,
            "sphere_mass": self.sphere_mass,
            "radius": RADIUS,
            "fluid_density": FLUID_DENSITY,
            "fluid_res": res,
            "dry": self.dry,
        }

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.6, -1.6, 1.0), pitch=-15.0, yaw=135.0)
        self.slice_vis = FluidSliceVisualizer(self.fluid, axis=1) if self.fluid is not None else None

        # warm up one frame outside the graph (first-step MuJoCo allocations),
        # then restore the initial state so the captured run starts at t = 0
        self.graph = capture_frame_graph(self.model, self.simulate, warmup=self._warmup)

    def _warmup(self):
        self.simulate()
        self._reset_state()

    def _reset_state(self):
        self.state_0.body_q.assign(self.model.body_q)
        self.state_0.body_qd.zero_()
        self.state_0.joint_q.assign(self.model.joint_q)
        self.state_0.joint_qd.zero_()
        self.solver.reset(self.state_0)

    def simulate(self):
        self.state_0.clear_forces()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        if self.dry:
            # substep the rigid solver like the coupled entry does
            sub_dt = self.frame_dt / self.args.rigid_substeps
            for _ in range(self.args.rigid_substeps):
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
                wp.copy(self.state_0.body_q, self.state_1.body_q)
                wp.copy(self.state_0.body_qd, self.state_1.body_qd)
                wp.copy(self.state_0.joint_q, self.state_1.joint_q)
                wp.copy(self.state_0.joint_qd, self.state_1.joint_qd)
        else:
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._record_metrics()

    def _record_metrics(self):
        body_q = self.state_0.body_q.numpy()[self.sphere]
        body_qd = self.state_0.body_qd.numpy()[self.sphere]
        frame = {
            "time": self.sim_time,
            "position": body_q[:3],
            "velocity": body_qd[:3],
        }
        if self.fluid is not None:
            wrench = fluid_body_wrenches(self.solver, self.fluid, self.frame_dt, self.model.body_count)
            diag = self.fluid.read_diagnostics()
            frame.update(
                fluid_force=wrench[self.sphere, :3],
                fluid_torque=wrench[self.sphere, 3:],
                div_l2_pre=diag["div_l2_pre"],
                div_l2_post=diag["div_l2_post"],
                div_linf_post=diag["div_linf_post"],
                pressure_residual=diag["pressure_residual"],
                noslip_max=diag["noslip_max"],
                momentum_balance_error=diag["momentum_balance_error"],
            )
        self.metrics.record(**frame)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.fluid is not None:
            log_tank_outline(self.viewer, self.fluid)
        if self.slice_vis is not None:
            self.slice_vis.log(self.viewer, field="speed", scale=1.0)
        self.viewer.end_frame()

    def test_final(self):
        self.metrics.save()
        z = float(self.state_0.body_q.numpy()[self.sphere][2])
        vz = float(self.state_0.body_qd.numpy()[self.sphere][2])
        rising = self.sphere_density < FLUID_DENSITY

        if self.dry:
            return

        assert np.isfinite(self.state_0.body_qd.numpy()).all()
        if len(self.metrics.frames) < 180:
            # short smoke runs only check stability and bookkeeping below
            pass
        elif rising:
            assert z > 0.4 * TANK, f"buoyant sphere should rise, z = {z}"
        else:
            assert z < 0.6 * TANK, f"dense sphere should settle, z = {z}"
            # hydrodynamic force opposes the motion direction (drag) and
            # includes buoyancy: while sinking, the fluid pushes up
            frames = self.metrics.frames
            mid = frames[len(frames) // 2]
            assert mid["fluid_force"][2] > 0.0, "fluid must push up on a sinking sphere"

        assert abs(vz) < 2.0, f"velocity must stay bounded, vz = {vz}"

        diag = self.fluid.read_diagnostics()
        assert diag["div_linf_post"] < 1.0, f"projection must keep divergence small: {diag['div_linf_post']}"
        balance = np.abs(np.array(diag["momentum_balance_error"])).max()
        scale = FLUID_DENSITY * 9.81 * self.frame_dt  # impulse scale per unit volume
        assert balance < scale, f"fluid-rigid impulse bookkeeping must balance: {balance}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_macfluid_args(parser)
        parser.add_argument(
            "--sphere-density",
            type=float,
            default=1500.0,
            help="Sphere density [kg/m^3]; below 1000 the sphere rises, above it settles.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
