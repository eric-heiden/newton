# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Open-loop replay of a real bimanual robot episode at the ABC YAM station.

``episode.npz`` holds the measured joints and the logged joint and gripper commands of
a real episode recorded at a bimanual station with two 6-DoF YAM arms (FORMAT.md
describes the files). ``build_model`` builds the station from ``station/`` (the ABC
simulator's MJCF: arms, grippers, table, and enclosure) in ``num_worlds`` identical
worlds, and ``Example`` drives every world with the logged commands as joint position
targets, open loop.

Command rule (verification replays submissions with the same rule): physics step
``k`` applies, at control time ``t[0] + k * dt``, the last command logged at or before
``t[0] + k * dt - command_delay`` (zero-order hold; the first command before that, the
last one after the episode ends). A gripper command ``g`` sets the left finger slide
target to ``clip(g, 0, 1) * 0.0475`` m and the right finger's to the negative value.
The arms start at the first measured joints and velocities, the fingers at the first
measured opening, and every other body where ``model.joint_q`` places it.

Run: ``python scene_replay.py --viewer null`` replays the whole episode
(``--seconds`` stops early, ``--num-worlds`` adds copies).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

HERE = Path(__file__).resolve().parent
STATION = HERE / "station" / "yam_bimanual_empty.xml"
EPISODE = HERE / "episode.npz"
SIDES = ("left", "right")
GRIPPER_TRAVEL = 0.0475  # finger slide [m] at gripper opening 1
FRAME_DT = 1.0 / 30.0  # example frame [s]

PARAMS = {
    "command_delay": 0.0,  # latency from a logged command to the joint targets [s]
    "dt": 0.002,  # physics step [s]
}


def build_model(num_worlds: int = 1) -> newton.Model:
    """The station in ``num_worlds`` identical worlds."""
    station = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(station)
    station.add_mjcf(str(STATION))
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for _ in range(num_worlds):
        builder.add_world(station)
    return builder.finalize()


def make_solver(model: newton.Model) -> newton.solvers.SolverBase:
    return newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)


def make_pipeline(model: newton.Model) -> newton.CollisionPipeline | None:
    """Collision pipeline the replay calls before every solver step (``None`` if the solver finds contacts)."""
    return newton.CollisionPipeline(model)


def load_episode(path: str | Path = EPISODE) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def command_targets(episode: dict, times: np.ndarray, delay: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per side: arm joint targets [T, 6] [rad] and left finger slide targets [T] [m] at control times [s]."""
    out = {}
    for side in SIDES:
        t, tg = episode[f"{side}_cmd_t"], episode[f"{side}_grip_cmd_t"]
        rows = np.clip(np.searchsorted(t, times - delay, side="right") - 1, 0, len(t) - 1)
        grip_rows = np.clip(np.searchsorted(tg, times - delay, side="right") - 1, 0, len(tg) - 1)
        finger = np.clip(episode[f"{side}_grip_cmd"][grip_rows], 0.0, 1.0) * GRIPPER_TRAVEL
        out[side] = (episode[f"{side}_cmd"][rows, :6], finger)
    return out


@wp.kernel
def apply_targets(
    schedule: wp.array2d[wp.float32],
    step: wp.array[wp.int32],
    index: wp.array[wp.int32],
    target: wp.array[wp.float32],
):
    j = wp.tid()
    target[index[j]] = schedule[wp.min(step[0], schedule.shape[0] - 1), j]


@wp.kernel
def advance(step: wp.array[wp.int32]):
    step[0] = step[0] + 1


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.episode = load_episode()
        self.model = build_model(args.num_worlds)
        self.solver = make_solver(self.model)
        self.collision_pipeline = make_pipeline(self.model)
        self.contacts = self.collision_pipeline.contacts() if self.collision_pipeline is not None else None
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()

        self.sim_dt = PARAMS["dt"]
        self.substeps = max(1, round(FRAME_DT / self.sim_dt))
        self.frame_dt = self.substeps * self.sim_dt
        self.sim_time = 0.0
        t = self.episode["left_t"]
        self.duration = float(t[-1] - t[0])
        times = t[0] + np.arange(math.ceil(self.duration / self.sim_dt - 1e-6)) * self.sim_dt
        targets = command_targets(self.episode, times, PARAMS["command_delay"])

        # Station joints of every world by MJCF name; targets use coordinate or DOF indexing.
        model = self.model
        names = [label.rsplit("/", 1)[-1] for label in model.joint_label]
        world = model.joint_world.numpy()
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        coords = self.control.joint_target_q.shape[0] == model.joint_coord_count
        self.q0, self.qd0 = model.joint_q.numpy().copy(), np.zeros(model.joint_dof_count, dtype=np.float32)
        columns, index = [], []
        for w in range(model.world_count):
            joint = {name: j for j, name in enumerate(names) if world[j] == w}
            for side in SIDES:
                arm, finger = targets[side]
                for k in range(6):
                    j = joint[f"{side}_joint{k + 1}"]
                    self.q0[q_start[j]] = self.episode[f"{side}_q"][0, k]
                    self.qd0[qd_start[j]] = self.episode[f"{side}_qd"][0, k]
                    columns.append(arm[:, k])
                    index.append(q_start[j] if coords else qd_start[j])
                opening = float(np.clip(self.episode[f"{side}_grip"][0], 0.0, 1.0)) * GRIPPER_TRAVEL
                for sign, name in ((1.0, "left_finger"), (-1.0, "right_finger")):
                    j = joint[f"{side}_{name}"]
                    self.q0[q_start[j]] = sign * opening
                    columns.append(sign * finger)
                    index.append(q_start[j] if coords else qd_start[j])
        self.schedule = wp.array(np.stack(columns, axis=-1).astype(np.float32), dtype=wp.float32)
        self.target_index = wp.array(np.asarray(index, dtype=np.int32), dtype=wp.int32)
        self.step_index = wp.zeros(1, dtype=wp.int32)  # an example array: MCP checkpoints rewind it
        self.reset()

        self.graph = None
        if wp.get_device().is_cuda:
            self.simulate()  # load kernels and let the solver allocate before the capture
            self.reset()
            self.capture()
        self.viewer.set_model(self.model)

    def reset(self):
        """Back to the first state sample: initial state and targets, command cursor 0."""
        self.state_0.joint_q.assign(self.q0)
        self.state_0.joint_qd.assign(self.qd0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.step_index.zero_()
        self.apply_targets()
        if hasattr(self.solver, "reset"):
            self.solver.reset(self.state_0, flags=newton.StateFlags.NONE)
        self.sim_time = 0.0

    def apply_targets(self):
        inputs = [self.schedule, self.step_index, self.target_index]
        wp.launch(apply_targets, self.schedule.shape[1], inputs, [self.control.joint_target_q])

    def simulate(self):
        """One physics step at the current command row."""
        self.apply_targets()
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0.assign(self.state_1)
        wp.launch(advance, 1, [self.step_index])

    def capture(self):
        """Record one physics step as a CUDA graph (again after replacing the solver or pipeline)."""
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def step(self):
        for _ in range(self.substeps):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        assert np.all(np.isfinite(self.state_0.body_q.numpy())), "the simulation diverged"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--num-worlds", type=int, default=1, help="Identical copies of the scene, one world each")
        parser.add_argument("--seconds", type=float, default=None, help="Replay only this long [s]")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    seconds = example.duration if args.seconds is None else min(args.seconds, example.duration)
    for _ in range(math.ceil(seconds / example.frame_dt - 1e-6)):
        if args.viewer != "null" and not viewer.is_running():
            break
        example.step()
        example.render()
    example.test_final()
    print(f"replayed {example.sim_time:.2f} s of {args.num_worlds} world(s)")
