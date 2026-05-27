"""Differentiable particle target with wall and floor contacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from ..harness import Scenario, ScenarioSnapshot


@wp.kernel
def _ball_cluster_com(
    particle_q: wp.array[wp.vec3],
    particles_per_world: int,
    com: wp.array[wp.vec3],
):
    tid = wp.tid()
    wp.atomic_add(com, tid // particles_per_world, particle_q[tid] / float(particles_per_world))


@wp.kernel
def _com_target_loss(
    com: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    loss: wp.array[wp.float32],
):
    tid = wp.tid()
    delta = com[tid] - targets[tid]
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


class DiffsimBallScenario(Scenario):
    """Diffsim ball target gradient inspired by ``example_diffsim_ball``."""

    id = "diffsim_ball"
    supported_solvers = ("semi_implicit",)

    # 7x7 lattice matches diffsim_cloth_com's particle-per-world count;
    # a single ball per world is too small for the COM-accumulator
    # atomics to surface non-determinism.
    LATTICE = 7

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        del builder

    def build(self, viewer: Any) -> None:
        self.horizon_steps = max(1, self.args.num_steps)
        self.horizon_substeps = max(1, self.args.substeps)
        self.sim_dt = 1.0 / self.args.fps / self.horizon_substeps
        self.loss_history: list[float] = []
        self.particles_per_world = self.LATTICE * self.LATTICE

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        ke = 1.0e4
        kd = 1.0e1
        kf = 0.0
        mu = 0.2
        shape_cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu)

        targets = []
        for world in range(self.args.world_count):
            x = float(world) * 3.0
            for i in range(self.LATTICE):
                for j in range(self.LATTICE):
                    dx = (i - (self.LATTICE - 1) * 0.5) * 0.06
                    dy = (j - (self.LATTICE - 1) * 0.5) * 0.06
                    scene.add_particle(
                        pos=wp.vec3(x + dx, -0.5 + dy, 1.0),
                        vel=wp.vec3(0.0, 5.0, -5.0),
                        mass=1.0,
                    )
            targets.append((x, -2.0, 1.5))
            scene.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(x, 2.0, 1.0), wp.quat_identity()),
                hx=1.0,
                hy=0.25,
                hz=1.0,
                cfg=shape_cfg,
            )

        scene.add_ground_plane(cfg=shape_cfg)
        self.model = scene.finalize(requires_grad=True)
        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 1.0

        self.solver = newton.solvers.SolverSemiImplicit(
            self.model,
            deterministic=self.args.solver_deterministic,
        )
        state_count = self.horizon_steps * self.horizon_substeps
        self.states = [self.model.state() for _ in range(state_count + 1)]
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=10.0,
            deterministic=True,
            requires_grad=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.states[0], self.contacts)

        self.targets = wp.array(targets, dtype=wp.vec3)
        self.com = wp.zeros(self.args.world_count, dtype=wp.vec3, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self._ran_backward = False

        self.viewer = viewer
        viewer.set_model(self.model)

    def _forward(self) -> wp.array[wp.float32]:
        assert self.solver is not None and self.control is not None
        self.com.zero_()
        self.loss.zero_()

        for step in range(self.horizon_steps):
            for substep in range(self.horizon_substeps):
                idx = step * self.horizon_substeps + substep
                self.states[idx].clear_forces()
                self.solver.step(
                    self.states[idx],
                    self.states[idx + 1],
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )

        wp.launch(
            _ball_cluster_com,
            dim=self.model.particle_count,
            inputs=[self.states[-1].particle_q, self.particles_per_world],
            outputs=[self.com],
        )
        wp.launch(
            _com_target_loss,
            dim=self.args.world_count,
            inputs=[self.com, self.targets],
            outputs=[self.loss],
        )
        return self.loss

    def _forward_backward(self) -> None:
        for state in self.states:
            state.particle_q.grad.zero_()
            state.particle_qd.grad.zero_()
        self.com.grad.zero_()
        self.loss.grad.zero_()
        tape = wp.Tape()
        with tape:
            loss = self._forward()
        tape.backward(loss)
        self.loss_history.append(float(self.loss.numpy()[0]))
        self._ran_backward = True

    def step(self) -> None:
        if not self._ran_backward:
            self._forward_backward()
        self.step_index += 1

    def render(self) -> None:
        state_index = min(self.step_index * self.horizon_substeps, len(self.states) - 1)
        state = self.states[state_index]
        self.viewer.begin_frame(self.step_index * (1.0 / self.args.fps))
        self.viewer.log_state(state)
        self.viewer.log_contacts(self.contacts, state)
        self.viewer.end_frame()

    def snapshot(self) -> ScenarioSnapshot:
        if not self._ran_backward:
            self._forward_backward()

        core = {
            "final_particle_q": self.states[-1].particle_q.numpy().copy(),
            "final_particle_qd": self.states[-1].particle_qd.numpy().copy(),
            "initial_particle_qd_grad": self.states[0].particle_qd.grad.numpy().copy(),
            "com": self.com.numpy().copy(),
            "loss": self.loss.numpy().copy(),
        }
        meta = {
            "scenario": self.id,
            "solver": self.args.solver.name,
            "world_count": self.args.world_count,
            "num_steps": self.args.num_steps,
            "seed": self.args.seed,
            "fps": self.args.fps,
            "substeps": self.args.substeps,
            "warp_version": wp.__version__,
            "wp_deterministic": wp.config.deterministic,
            "collision_pipeline_deterministic": bool(getattr(self.collision_pipeline, "deterministic", False)),
            "collision_pipeline_wp_deterministic": wp.config.deterministic,
            "custom_collision_pipeline": True,
            "graph_capture_enabled": False,
            "diffsim": True,
        }
        extras = {
            "loss_history": np.asarray(self.loss_history, dtype=np.float32),
            "target": self.targets.numpy().copy(),
        }
        return ScenarioSnapshot(core=core, extras=extras, meta=meta)
