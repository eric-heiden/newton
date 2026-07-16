# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MAC Fluid Swimmer
#
# Articulated multi-link swimmers driven by sinusoidal joint targets
# undulate inside a closed water tank. MuJoCo owns the articulations and
# the MAC-grid fluid solver returns per-link hydrodynamic wrenches, so all
# propulsion arises from two-way fluid interaction: the traveling body
# wave pushes water backward and the swimmer moves forward. Reversing the
# wave (--reverse) reverses the swimming direction; a dry run (--dry)
# wiggles in place. Gravity is disabled to isolate propulsion.
#
# The tank size, link count, and swimmer count are configurable for
# large-scale rollouts: several swimmers can share one fluid domain
# (--num-swimmers, optionally with per-swimmer gait frequencies), and
# --reverse-at makes a swimmer turn around mid-run by smoothly reversing
# its traveling wave.
#
# The tank walls are fluid boundaries only (no rigid collision geometry),
# so a swimmer that reaches the end of the tank passes out of the fluid
# and coasts force-free at constant momentum.
#
# Command: python -m newton.examples macfluid_swimmer
#          python -m newton.examples macfluid_swimmer --reverse
#          python -m newton.examples macfluid_swimmer --dry
#          python -m newton.examples macfluid_swimmer --tank-length 8 --fluid-res 384 --reverse-at 14
#          python -m newton.examples macfluid_swimmer --num-swimmers 3 --tank-width 2.4 \
#              --swimmer-frequencies 0.8,1.2,1.6 --tank-length 8 --fluid-res 384
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

# base link geometry (scaled by --swimmer-scale)
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

        self.tank = (args.tank_length, args.tank_width, args.tank_depth)
        self.scale = float(args.swimmer_scale)
        self.link_hx = LINK_HX * self.scale
        self.link_hy = LINK_HY * self.scale
        self.link_hz = LINK_HZ * self.scale
        self.link_pitch = LINK_PITCH * self.scale
        self.num_links = int(args.num_links)
        self.num_swimmers = int(args.num_swimmers)
        self.amplitude = args.amplitude
        self.phase_step = -args.phase_step if args.reverse else args.phase_step
        self.reverse_at = args.reverse_at
        if args.swimmer_frequencies:
            self.frequencies = [float(f) for f in args.swimmer_frequencies.split(",")]
            if len(self.frequencies) != self.num_swimmers:
                raise ValueError("--swimmer-frequencies must list one frequency per swimmer")
        else:
            self.frequencies = [args.frequency] * self.num_swimmers

        # gravity off: propulsion must come from the fluid alone
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)

        z0 = 0.5 * self.tank[2]
        # start displaced so the swimmers have most of the tank ahead of them
        if args.start_x is not None:
            x_offset = args.start_x
        else:
            x_offset = 0.225 * self.tank[0] if args.reverse else -0.225 * self.tank[0]

        self.links = []
        self.swimmer_links = []
        for s in range(self.num_swimmers):
            y = (s - 0.5 * (self.num_swimmers - 1)) * (self.tank[1] / max(self.num_swimmers, 1))
            links = []
            joints = []
            for i in range(self.num_links):
                x = x_offset + (0.5 * (self.num_links - 1) - i) * self.link_pitch
                body = builder.add_link(xform=wp.transform(wp.vec3(x, y, z0), wp.quat_identity()))
                builder.add_shape_box(
                    body,
                    hx=self.link_hx,
                    hy=self.link_hy,
                    hz=self.link_hz,
                    # dense links keep body inertia above hydrodynamic added
                    # inertia (stability requirement of staggered weak coupling)
                    cfg=newton.ModelBuilder.ShapeConfig(density=args.link_density),
                )
                links.append(body)

            joints.append(builder.add_joint_free(links[0]))
            for i in range(1, self.num_links):
                j = builder.add_joint_revolute(
                    parent=links[i - 1],
                    child=links[i],
                    axis=wp.vec3(0.0, 0.0, 1.0),
                    parent_xform=wp.transform(wp.vec3(-0.5 * self.link_pitch, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.5 * self.link_pitch, 0.0, 0.0), wp.quat_identity()),
                )
                builder.joint_target_mode[-1] = int(JointTargetMode.POSITION)
                builder.joint_target_ke[-1] = args.joint_ke
                builder.joint_target_kd[-1] = args.joint_kd
                joints.append(j)
            builder.add_articulation(joints)
            self.swimmer_links.append(links)
            self.links.extend(links)
        self._drive_joints = [
            [s * self.num_links + i for i in range(1, self.num_links)] for s in range(self.num_swimmers)
        ]

        self.model = builder.finalize()

        res = int(args.fluid_res)
        dx = self.tank[0] / res
        fluid_cfg = SolverMACFluid.Config(
            resolution=(
                res,
                max(int(round(self.tank[1] / dx)), 8),
                max(int(round(self.tank[2] / dx)), 8),
            ),
            cell_size=dx,
            origin=(-0.5 * self.tank[0], -0.5 * self.tank[1], 0.0),
            density=FLUID_DENSITY,
            kinematic_viscosity=args.viscosity,
            advection=args.advection,
            pressure_iterations=args.pressure_iterations,
        )

        njmax = 100 * self.num_swimmers
        self.dry = args.dry
        if self.dry:
            self.solver = SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=njmax)
            self.fluid = None
        else:
            self.solver, self.fluid = make_coupled_fluid_solver(
                self.model,
                fluid_cfg,
                rigid_bodies=self.links,
                joints=list(range(self.model.joint_count)),
                args=args,
                mujoco_kwargs={"njmax": njmax},
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
        self._target_indices = np.array([[starts[j] for j in joints] for joints in self._drive_joints], dtype=np.int64)
        self._target_q = np.zeros(target_len, dtype=np.float32)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.metrics = MetricsRecorder(args.metrics_output)
        self.metrics.meta = {
            "example": "macfluid_swimmer",
            "amplitude": self.amplitude,
            "frequencies": self.frequencies,
            "phase_step": self.phase_step,
            "reverse": bool(args.reverse),
            "reverse_at": self.reverse_at,
            "num_links": self.num_links,
            "swimmer_scale": self.scale,
            "link_density": args.link_density,
            "advection": args.advection,
            "num_swimmers": self.num_swimmers,
            "tank": list(self.tank),
            "fluid_res": res,
            "fluid_cells": int(np.prod(fluid_cfg.resolution)),
            "dry": self.dry,
        }

        self.viewer.set_model(self.model)
        length = self.tank[0]
        self.viewer.set_camera(pos=wp.vec3(0.9 * length, -1.0 * length, 0.7 * length), pitch=-25.0, yaw=140.0)
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
        i = np.arange(1, self.num_links)
        # Optional mid-run turnaround: dip the gait amplitude to zero and flip
        # the traveling-wave direction at the quiet point. Cross-blending the
        # two waves instead would pass through a standing-wave regime where
        # all joints load the fluid simultaneously, which can destabilize the
        # weak coupling.
        env = 1.0
        wave_sign = 1.0
        if self.reverse_at is not None:
            u = (t - self.reverse_at) / 2.0  # 2 s turnaround window
            if u >= 1.0:
                wave_sign = -1.0
            elif u >= 0.0:
                env = abs(2.0 * u - 1.0)
                wave_sign = 1.0 if u < 0.5 else -1.0
        for s in range(self.num_swimmers):
            phase = 2.0 * np.pi * self.frequencies[s] * t
            targets = ramp * env * self.amplitude * np.sin(phase - wave_sign * self.phase_step * i)
            self._target_q[self._target_indices[s]] = targets.astype(np.float32)
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
        if self.num_swimmers > 1:
            frame["swimmer_coms"] = np.array([body_q[links, :3].mean(axis=0) for links in self.swimmer_links])
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
        if self.fluid is not None:
            log_tank_outline(self.viewer, self.fluid)
        if self.slice_vis is not None:
            self.slice_vis.log(self.viewer, field=self.args.slice_field, scale=self.args.slice_scale)
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
        if len(frames) >= 240 and self.reverse_at is None:
            assert abs(disp[0]) > 0.02, f"swimmer must make way through the fluid, dx = {disp[0]}"
        if self.num_swimmers == 1:
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
        parser.add_argument(
            "--reverse-at",
            type=float,
            default=None,
            help="Smoothly reverse the traveling wave at this time [s] (turnaround).",
        )
        parser.add_argument("--joint-ke", type=float, default=30.0, help="Joint position gain [N*m/rad].")
        parser.add_argument("--joint-kd", type=float, default=1.5, help="Joint damping gain [N*m*s/rad].")
        parser.add_argument("--num-links", type=int, default=5, help="Links per swimmer.")
        parser.add_argument("--num-swimmers", type=int, default=1, help="Swimmers sharing the fluid domain.")
        parser.add_argument(
            "--swimmer-frequencies",
            type=str,
            default=None,
            help="Comma-separated per-swimmer gait frequencies [Hz]; overrides --frequency.",
        )
        parser.add_argument("--tank-length", type=float, default=2.0, help="Tank extent along x [m].")
        parser.add_argument("--tank-width", type=float, default=0.8, help="Tank extent along y [m].")
        parser.add_argument("--tank-depth", type=float, default=0.6, help="Tank extent along z [m].")
        parser.add_argument("--start-x", type=float, default=None, help="Initial swimmer center x [m].")
        parser.add_argument("--swimmer-scale", type=float, default=1.0, help="Scale factor for link geometry.")
        parser.add_argument(
            "--link-density",
            type=float,
            default=2500.0,
            help="Link density [kg/m^3]; must keep body inertia above hydrodynamic added inertia.",
        )
        parser.add_argument(
            "--slice-field",
            type=str,
            choices=["speed", "vorticity", "pressure"],
            default="speed",
            help="Fluid slice visualization field.",
        )
        parser.add_argument("--slice-scale", type=float, default=0.8, help="Color scale of the slice field.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
