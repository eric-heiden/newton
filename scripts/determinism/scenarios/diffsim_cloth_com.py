"""Differentiable cloth COM target with atomic COM accumulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from ..harness import Scenario, ScenarioSnapshot


@wp.kernel
def _cloth_com(
    particle_q: wp.array[wp.vec3],
    particles_per_world: int,
    com: wp.array[wp.vec3],
):
    tid = wp.tid()
    world = tid // particles_per_world
    wp.atomic_add(com, world, particle_q[tid] / float(particles_per_world))


@wp.kernel
def _com_target_loss(
    com: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    loss: wp.array[wp.float32],
):
    world = wp.tid()
    delta = com[world] - targets[world]
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


class DiffsimClothComScenario(Scenario):
    """Small cloth gradient scenario inspired by ``example_diffsim_cloth``."""

    id = "diffsim_cloth_com"
    supported_solvers = ("semi_implicit",)

    DIM_X = 6
    DIM_Y = 6

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        del builder

    def build(self, viewer: Any) -> None:
        self.horizon_steps = max(1, self.args.num_steps)
        self.horizon_substeps = max(1, self.args.substeps)
        self.sim_dt = 1.0 / self.args.fps / self.horizon_substeps
        self.loss_history: list[float] = []

        scene = newton.ModelBuilder(gravity=0.0)
        scene.default_particle_radius = 0.025
        targets = []

        for world in range(self.args.world_count):
            x = float(world) * 1.5
            scene.add_cloth_grid(
                pos=wp.vec3(x, 0.0, 0.0),
                vel=wp.vec3(0.0, 0.1, 0.1),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi * 0.75),
                dim_x=self.DIM_X,
                dim_y=self.DIM_Y,
                cell_x=1.0 / self.DIM_X,
                cell_y=1.0 / self.DIM_Y,
                mass=1.0,
                tri_ke=500.0,
                tri_ka=500.0,
                tri_kd=20.0,
                tri_lift=0.0,
                tri_drag=0.5,
            )
            targets.append((x, 2.0, 0.0))

        self.model = scene.finalize(requires_grad=True)
        self.solver = newton.solvers.SolverSemiImplicit(
            self.model,
            deterministic=self.args.solver_deterministic,
        )
        self.solver.enable_tri_contact = False
        state_count = self.horizon_steps * self.horizon_substeps
        self.states = [self.model.state() for _ in range(state_count + 1)]
        self.control = self.model.control()
        self.contacts = None
        # ``add_cloth_grid`` builds an inclusive vertex grid of size
        # ``(dim_x + 1) * (dim_y + 1)`` (see ``ModelBuilder.add_cloth_grid``),
        # not ``dim_x * dim_y``. Using the latter under-counts particles and
        # mis-routes their atomic contributions to the wrong world.
        self.particles_per_world = (self.DIM_X + 1) * (self.DIM_Y + 1)

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
            _cloth_com,
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
            "graph_capture_enabled": False,
            "diffsim": True,
        }
        extras = {
            "loss_history": np.asarray(self.loss_history, dtype=np.float32),
            "target": self.targets.numpy().copy(),
            "particles_per_world": self.particles_per_world,
        }
        return ScenarioSnapshot(core=core, extras=extras, meta=meta)
