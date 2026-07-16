# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MAC Fluid Paddle
#
# A motor-driven two-blade paddle stirs a closed water tank. MuJoCo owns
# the articulation (one revolute joint with a velocity actuator) and the
# MAC-grid fluid solver returns the hydrodynamic reaction torque, so the
# paddle spins up slower and settles below its velocity target when wet.
# Run once with --dry for the rigid-only comparison.
#
# Command: python -m newton.examples macfluid_paddle
#          python -m newton.examples macfluid_paddle --dry
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
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

TANK_XY = 1.0  # tank footprint [m]
TANK_Z = 0.5  # tank height [m]
BLADE_HALF_LENGTH = 0.32
BLADE_HALF_THICKNESS = 0.025
BLADE_HALF_HEIGHT = 0.1
FLUID_DENSITY = 1000.0


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.omega_target = args.omega

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        pivot = wp.vec3(0.0, 0.0, 0.5 * TANK_Z)
        self.paddle = builder.add_link(xform=wp.transform(pivot, wp.quat_identity()))
        builder.add_shape_box(
            self.paddle,
            hx=BLADE_HALF_LENGTH,
            hy=BLADE_HALF_THICKNESS,
            hz=BLADE_HALF_HEIGHT,
            # a dense blade keeps body inertia above the hydrodynamic added
            # inertia, the stability requirement of staggered (weak) coupling
            cfg=newton.ModelBuilder.ShapeConfig(density=2500.0),
        )
        self.joint = builder.add_joint_revolute(
            parent=-1,
            child=self.paddle,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(pivot, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        # velocity servo: tau = kd * (omega_target - omega)
        builder.joint_target_mode[-1] = int(JointTargetMode.VELOCITY)
        builder.joint_target_kd[-1] = args.motor_gain
        builder.add_articulation([self.joint])

        self.model = builder.finalize()

        res = int(args.fluid_res)
        dx = TANK_XY / res
        fluid_cfg = SolverMACFluid.Config(
            resolution=(res, res, max(int(round(TANK_Z / dx)), 8)),
            cell_size=dx,
            origin=(-0.5 * TANK_XY, -0.5 * TANK_XY, 0.0),
            density=FLUID_DENSITY,
            kinematic_viscosity=args.viscosity,
            advection=args.advection,
            pressure_iterations=args.pressure_iterations,
        )

        self.dry = args.dry
        if self.dry:
            self.solver = SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=50)
            self.fluid = None
        else:
            self.solver, self.fluid = make_coupled_fluid_solver(
                self.model,
                fluid_cfg,
                rigid_bodies=[self.paddle],
                joints=list(range(self.model.joint_count)),
                args=args,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.collision_pipeline.contacts()

        # constant velocity target (device array is read inside the graph)
        target_qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        target_qd[0] = self.omega_target
        self.control.joint_target_qd.assign(target_qd)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.metrics = MetricsRecorder(args.metrics_output)
        self.metrics.meta = {
            "example": "macfluid_paddle",
            "omega_target": self.omega_target,
            "motor_gain": args.motor_gain,
            "fluid_res": res,
            "dry": self.dry,
        }

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.4, -1.4, 1.2), pitch=-30.0, yaw=135.0)
        self.slice_vis = FluidSliceVisualizer(self.fluid, axis=2) if self.fluid is not None else None

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
        joint_q = float(self.state_0.joint_q.numpy()[0])
        omega = float(self.state_0.joint_qd.numpy()[0])
        frame = {"time": self.sim_time, "joint_angle": joint_q, "joint_velocity": omega}
        if self.fluid is not None:
            wrench = fluid_body_wrenches(self.solver, self.fluid, self.frame_dt, self.model.body_count)
            diag = self.fluid.read_diagnostics()
            frame.update(
                fluid_torque_z=wrench[self.paddle, 5],
                fluid_force=wrench[self.paddle, :3],
                div_l2_post=diag["div_l2_post"],
                div_linf_post=diag["div_linf_post"],
                pressure_residual=diag["pressure_residual"],
                momentum_balance_error=diag["momentum_balance_error"],
            )
        self.metrics.record(**frame)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.fluid is not None:
            log_tank_outline(self.viewer, self.fluid)
        if self.slice_vis is not None:
            self.slice_vis.log(self.viewer, field="speed", scale=1.5)
        self.viewer.end_frame()

    def test_final(self):
        self.metrics.save()
        frames = self.metrics.frames
        late = [f["joint_velocity"] for f in frames[-30:]]
        omega_late = float(np.mean(late))
        long_run = len(frames) >= 120
        assert np.isfinite(self.state_0.joint_qd.numpy()).all()

        if self.dry:
            if long_run:
                assert abs(omega_late - self.omega_target) < 0.05 * self.omega_target, (
                    f"dry paddle should reach its velocity target, got {omega_late} vs {self.omega_target}"
                )
            return

        if long_run:
            # under fluid load the servo settles below its target
            assert 0.1 * self.omega_target < omega_late < 0.98 * self.omega_target, (
                f"wet paddle must spin below target under fluid load, got {omega_late} vs {self.omega_target}"
            )
            # the fluid reaction torque opposes the rotation
            tau_late = float(np.mean([f["fluid_torque_z"] for f in frames[-30:]]))
            assert tau_late < 0.0, f"reaction torque must oppose +z spin, got {tau_late}"
        diag = self.fluid.read_diagnostics()
        assert diag["div_linf_post"] < 1.0, f"divergence must stay small: {diag['div_linf_post']}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_macfluid_args(parser)
        parser.add_argument("--omega", type=float, default=4.0, help="Motor velocity target [rad/s].")
        parser.add_argument("--motor-gain", type=float, default=8.0, help="Velocity servo gain [N*m*s/rad].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
