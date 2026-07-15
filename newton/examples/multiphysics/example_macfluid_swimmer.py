# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MAC Fluid Swimmer
#
# A five-link articulated swimmer driven by sinusoidal joint targets
# undulates inside a closed water tank. MuJoCo owns the articulation and
# the MAC-grid fluid solver returns per-link hydrodynamic wrenches, so all
# propulsion arises from two-way fluid interaction: the traveling body
# wave pushes water backward and the swimmer moves forward. Reversing the
# wave (--reverse) reverses the swimming direction; a dry run (--dry)
# wiggles in place. Gravity is disabled to isolate propulsion.
#
# Command: python -m newton.examples macfluid_swimmer
#          python -m newton.examples macfluid_swimmer --reverse
#          python -m newton.examples macfluid_swimmer --dry
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
    make_coupled_fluid_solver,
)
from newton.solvers import SolverMACFluid, SolverMuJoCo

TANK = (2.0, 0.8, 0.6)  # tank extents [m]
NUM_LINKS = 5
LINK_HX = 0.085  # link half length [m]
LINK_HY = 0.025  # link half width [m]
LINK_HZ = 0.05  # link half height [m]
LINK_PITCH = 0.19  # link center spacing [m]
FLUID_DENSITY = 1000.0


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.amplitude = args.amplitude
        self.frequency = args.frequency
        self.phase_step = -args.phase_step if args.reverse else args.phase_step

        # gravity off: propulsion must come from the fluid alone
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)

        z0 = 0.5 * TANK[2]
        # start displaced so the swimmer has most of the tank ahead of it
        x_offset = 0.45 if args.reverse else -0.45
        self.links = []
        joints = []
        for i in range(NUM_LINKS):
            x = x_offset + (0.5 * (NUM_LINKS - 1) - i) * LINK_PITCH
            body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, z0), wp.quat_identity()))
            builder.add_shape_box(
                body,
                hx=LINK_HX,
                hy=LINK_HY,
                hz=LINK_HZ,
                # dense links keep body inertia above hydrodynamic added
                # inertia (stability requirement of staggered weak coupling)
                cfg=newton.ModelBuilder.ShapeConfig(density=2500.0),
            )
            self.links.append(body)

        joints.append(builder.add_joint_free(self.links[0]))
        for i in range(1, NUM_LINKS):
            j = builder.add_joint_revolute(
                parent=self.links[i - 1],
                child=self.links[i],
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(-0.5 * LINK_PITCH, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.5 * LINK_PITCH, 0.0, 0.0), wp.quat_identity()),
            )
            builder.joint_target_mode[-1] = int(JointTargetMode.POSITION)
            builder.joint_target_ke[-1] = args.joint_ke
            builder.joint_target_kd[-1] = args.joint_kd
            joints.append(j)
        builder.add_articulation(joints)

        self.model = builder.finalize()

        res = int(args.fluid_res)
        dx = TANK[0] / res
        fluid_cfg = SolverMACFluid.Config(
            resolution=(res, max(int(round(TANK[1] / dx)), 8), max(int(round(TANK[2] / dx)), 8)),
            cell_size=dx,
            origin=(-0.5 * TANK[0], -0.5 * TANK[1], 0.0),
            density=FLUID_DENSITY,
            kinematic_viscosity=args.viscosity,
            pressure_iterations=args.pressure_iterations,
        )

        self.dry = args.dry
        if self.dry:
            self.solver = SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=100)
            self.fluid = None
        else:
            self.solver, self.fluid = make_coupled_fluid_solver(
                self.model,
                fluid_cfg,
                rigid_bodies=self.links,
                joints=list(range(self.model.joint_count)),
                args=args,
                mujoco_kwargs={"njmax": 100},
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.collision_pipeline.contacts()

        # coordinate indices of the revolute drive targets in joint_target_q
        target_len = int(self.control.joint_target_q.shape[0])
        if target_len == int(self.model.joint_coord_count):
            starts = self.model.joint_q_start.numpy()
        else:
            starts = self.model.joint_qd_start.numpy()
        self._target_indices = np.array([starts[j] for j in range(1, NUM_LINKS)], dtype=np.int64)
        self._target_q = np.zeros(target_len, dtype=np.float32)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.metrics = MetricsRecorder(args.metrics_output)
        self.metrics.meta = {
            "example": "macfluid_swimmer",
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "phase_step": self.phase_step,
            "reverse": bool(args.reverse),
            "fluid_res": res,
            "dry": self.dry,
        }

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.8, -2.0, 1.4), pitch=-25.0, yaw=140.0)
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

    def _update_targets(self):
        # smooth start-up ramp avoids impulsive added-mass torques
        t = self.sim_time
        ramp = min(t / 1.0, 1.0)
        phase = 2.0 * np.pi * self.frequency * t
        targets = ramp * self.amplitude * np.sin(phase - self.phase_step * np.arange(1, NUM_LINKS))
        self._target_q[self._target_indices] = targets.astype(np.float32)
        self.control.joint_target_q.assign(self._target_q)

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
        # the control array is updated in place outside the captured graph
        self._update_targets()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._record_metrics()

    def _record_metrics(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        com = body_q[self.links, :3].mean(axis=0)
        frame = {
            "time": self.sim_time,
            "com": com,
            "head_position": body_q[self.links[0], :3],
            "com_velocity": body_qd[self.links, :3].mean(axis=0),
        }
        if self.fluid is not None:
            wrench = fluid_body_wrenches(self.solver, self.fluid, self.frame_dt, self.model.body_count)
            diag = self.fluid.read_diagnostics()
            frame.update(
                link_forces=wrench[self.links, :3],
                div_l2_post=diag["div_l2_post"],
                div_linf_post=diag["div_linf_post"],
                pressure_residual=diag["pressure_residual"],
                momentum_balance_error=diag["momentum_balance_error"],
            )
        self.metrics.record(**frame)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.slice_vis is not None:
            self.slice_vis.log(self.viewer, field="speed", scale=0.8)
        self.viewer.end_frame()

    def test_final(self):
        self.metrics.save()
        frames = self.metrics.frames
        disp = np.array(frames[-1]["com"]) - np.array(frames[0]["com"])

        if self.dry:
            if len(frames) >= 240:
                assert abs(disp[0]) < 0.1, f"dry swimmer must not translate, dx = {disp[0]}"
            return

        # propulsion must arise from fluid interaction; a run of at least a
        # few gait cycles yields clear net displacement along the tank
        if len(frames) >= 240:
            assert abs(disp[0]) > 0.02, f"swimmer must make way through the fluid, dx = {disp[0]}"
        assert abs(disp[1]) < 0.25, f"swimmer must stay near the tank centerline, dy = {disp[1]}"
        diag = self.fluid.read_diagnostics()
        assert diag["div_linf_post"] < 1.0, f"divergence must stay small: {diag['div_linf_post']}"
        assert np.isfinite(self.state_0.body_qd.numpy()).all()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_macfluid_args(parser)
        parser.add_argument("--amplitude", type=float, default=0.5, help="Joint target amplitude [rad].")
        parser.add_argument("--frequency", type=float, default=1.2, help="Gait frequency [Hz].")
        parser.add_argument("--phase-step", type=float, default=1.0, help="Phase lag per joint [rad].")
        parser.add_argument("--reverse", action="store_true", help="Reverse the traveling wave direction.")
        parser.add_argument("--joint-ke", type=float, default=30.0, help="Joint position gain [N*m/rad].")
        parser.add_argument("--joint-kd", type=float, default=1.5, help="Joint damping gain [N*m*s/rad].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
