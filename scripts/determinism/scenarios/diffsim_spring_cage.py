"""Differentiable spring rest-length target gradient."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from ..harness import Scenario, ScenarioSnapshot


@wp.kernel
def _spring_mesh_com(
    particle_q: wp.array[wp.vec3],
    free_per_world: int,
    com: wp.array[wp.vec3],
):
    tid = wp.tid()
    wp.atomic_add(com, tid // free_per_world, particle_q[tid] / float(free_per_world))


@wp.kernel
def _com_target_loss(
    com: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    loss: wp.array[wp.float32],
):
    tid = wp.tid()
    delta = com[tid] - targets[tid]
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


class DiffsimSpringCageScenario(Scenario):
    """Spring rest-length gradient inspired by ``example_diffsim_spring_cage``."""

    id = "diffsim_spring_cage"
    supported_solvers = ("semi_implicit",)

    # Single-particle upstream variant does not generate enough atomic
    # contention to surface non-determinism: a 7x7 free-particle mesh
    # bouncing on the ground plane provides spring, contact and COM
    # atomics on every step.
    FREE_DIM = 7

    _CAGE_POINTS: tuple[tuple[float, float, float], ...] = (
        (0.2, -0.7, 0.8),
        (1.1, 0.0, 0.2),
        (-1.2, 0.1, 0.1),
        (0.4, 0.6, 0.4),
        (-0.2, 0.7, -0.9),
        (0.1, -0.8, -0.8),
        (-0.8, -0.9, 0.2),
        (-0.1, 1.0, 0.4),
    )

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        del builder

    def build(self, viewer: Any) -> None:
        self.horizon_steps = max(1, self.args.num_steps)
        self.horizon_substeps = max(1, self.args.substeps)
        self.sim_dt = 1.0 / self.args.fps / self.horizon_substeps
        self.loss_history: list[float] = []

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        ke = 1.0e4
        kd = 1.0e1
        kf = 0.0
        mu = 0.2
        shape_cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu)

        self.free_per_world = self.FREE_DIM * self.FREE_DIM
        anchors_per_world = len(self._CAGE_POINTS)

        # Free particles for all worlds come first, then all anchors, so
        # the COM kernel can run over a contiguous prefix.
        targets: list[tuple[float, float, float]] = []
        free_base_per_world: list[int] = []
        for world in range(self.args.world_count):
            x = float(world) * 3.0
            free_base_per_world.append(scene.particle_count)
            for i in range(self.FREE_DIM):
                for j in range(self.FREE_DIM):
                    dx = (i - (self.FREE_DIM - 1) * 0.5) * 0.06
                    dy = (j - (self.FREE_DIM - 1) * 0.5) * 0.06
                    scene.add_particle(
                        (x + dx, -0.5 + dy, 1.0),
                        (0.0, 5.0, -5.0),
                        1.0,
                    )
            targets.append((x, -2.0, 1.5))

        anchor_base_per_world: list[int] = []
        for world in range(self.args.world_count):
            x = float(world) * 3.0
            anchor_base_per_world.append(scene.particle_count)
            for point in self._CAGE_POINTS:
                scene.add_particle((x + point[0], point[1], 1.0 + point[2]), (0.0, 0.0, 0.0), 0.0)

        for world in range(self.args.world_count):
            x = float(world) * 3.0
            scene.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(x, 2.0, 1.0), wp.quat_identity()),
                hx=1.0,
                hy=0.25,
                hz=1.0,
                cfg=shape_cfg,
            )
            for free_local in range(self.free_per_world):
                free_idx = free_base_per_world[world] + free_local
                for anchor_local in range(anchors_per_world):
                    anchor_idx = anchor_base_per_world[world] + anchor_local
                    scene.add_spring(free_idx, anchor_idx, 50.0, 1.0, 0.0)

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

        self.total_free = self.args.world_count * self.free_per_world
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
            _spring_mesh_com,
            dim=self.total_free,
            inputs=[self.states[-1].particle_q, self.free_per_world],
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
        self.model.spring_rest_length.grad.zero_()
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
            "spring_rest_length_grad": self.model.spring_rest_length.grad.numpy().copy(),
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
