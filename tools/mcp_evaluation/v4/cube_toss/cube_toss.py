# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Real acrylic cube tossed onto a table, replayed from its measured release state.

The recordings come from the ContactNets cube-toss dataset (DAIR Lab, BSD-3-Clause):
a 0.1048 m, 0.37 kg acrylic cube is tossed by hand onto a wooden table and
tracked at 148 Hz. ``tosses.npz`` holds, per toss, the measured position [m],
orientation (quaternion ``x, y, z, w``), linear velocity [m/s], and angular
velocity [rad/s] (both in the world frame), with the table surface at ``z = 0``.

Every toss becomes one Newton world: the cube starts from the first measured
frame and is simulated open loop, and the predicted poses are compared with the
recording (``evaluate``). ``Example.step`` advances all tosses by one camera frame.

Run: ``python cube_toss.py --viewer null``
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

HALF_WIDTH = 0.0524  # [m] (measured)
MASS = 0.37  # [kg] (measured)
INERTIA = 0.00081  # principal moment about the center [kg m^2] (measured)
FRAME_DT = 1.0 / 148.0  # camera frame period [s]

# ---------------- Contact model (tunable) ----------------
# Untuned defaults for an acrylic cube on wood.
PARAMS = {
    "mu": 0.5,  # sliding friction coefficient [-]
    "ke": 2.5e3,  # contact stiffness [N/m]
    "kd": 100.0,  # contact damping [N s/m]
    "mu_torsional": 0.005,  # torsional friction [m]
    "mu_rolling": 0.0001,  # rolling friction [m]
    "table_height": 0.0,  # table surface height [m]
}
SUBSTEPS = 10  # simulation steps per camera frame
# ---------------------------------------------------------


def load_tosses(path) -> list[dict]:
    """Recorded tosses as dicts of arrays ``t, pos, quat, vel, ang_vel``."""
    data = np.load(path)
    offsets = data["offsets"]
    return [
        {
            "t": np.arange(end - start) * FRAME_DT,
            **{key: data[key][start:end] for key in ("pos", "quat", "vel", "ang_vel")},
        }
        for start, end in itertools.pairwise(offsets)
    ]


def build_model(num_worlds: int, params: dict | None = None) -> newton.Model:
    """One cube on its own table in each of ``num_worlds`` worlds."""
    p = PARAMS if params is None else params
    # The cube's mass properties are measured, so its shape carries no density.
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, mu=p["mu"], ke=p["ke"], kd=p["kd"], mu_torsional=p["mu_torsional"], mu_rolling=p["mu_rolling"]
    )
    cube = newton.ModelBuilder()
    body = cube.add_body(mass=MASS, inertia=wp.mat33(np.eye(3) * INERTIA), label="cube")
    cube.add_shape_box(body, hx=HALF_WIDTH, hy=HALF_WIDTH, hz=HALF_WIDTH, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(cube, num_worlds)
    builder.add_ground_plane(height=p["table_height"], cfg=cfg)
    return builder.finalize()


def make_solver(model: newton.Model) -> newton.solvers.SolverBase:
    return newton.solvers.SolverMuJoCo(
        model, use_mujoco_contacts=False, integrator="implicitfast", cone="elliptic", nconmax=16, njmax=64
    )


def make_pipeline(model: newton.Model) -> newton.CollisionPipeline:
    return newton.CollisionPipeline(model)


def initial_state(model: newton.Model, tosses: list[dict]):
    """State with every world's cube at its toss's first measured frame."""
    state = model.state()
    joint_q = np.concatenate([np.concatenate([toss["pos"][0], toss["quat"][0]]) for toss in tosses])
    joint_qd = np.concatenate([np.concatenate([toss["vel"][0], toss["ang_vel"][0]]) for toss in tosses])
    state.joint_q.assign(joint_q.astype(np.float32))
    state.joint_qd.assign(joint_qd.astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def simulate_frame(model, solver, pipeline, contacts, state_0, state_1, control):
    """Advance one camera frame; the result is always written back to ``state_0``."""
    dt = FRAME_DT / SUBSTEPS
    current, scratch = state_0, state_1
    for _ in range(SUBSTEPS):
        current.clear_forces()
        pipeline.collide(current, contacts)
        solver.step(current, scratch, control, contacts, dt)
        current, scratch = scratch, current
    if current is not state_0:
        # Keeping the result in a fixed buffer lets one captured CUDA graph replay every frame.
        state_0.assign(current)


def rollout(model, solver, tosses: list[dict], frames: int) -> np.ndarray:
    """Simulate every toss (one per world) from its first frame.

    Returns:
        Predicted body poses ``[frames, num_worlds, 7]`` (position, quaternion xyzw)
        at the camera frame times.
    """
    state_0, state_1, control = initial_state(model, tosses), model.state(), model.control()
    pipeline = make_pipeline(model)
    contacts = pipeline.contacts()
    graph = None
    if model.device.is_cuda:
        with wp.ScopedCapture() as capture:
            simulate_frame(model, solver, pipeline, contacts, state_0, state_1, control)
        graph = capture.graph
    poses = [state_0.body_q.numpy()]
    for _ in range(frames - 1):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            simulate_frame(model, solver, pipeline, contacts, state_0, state_1, control)
        poses.append(state_0.body_q.numpy())
    return np.asarray(poses)


def toss_errors(tosses: list[dict], poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-toss mean position error [m] and mean orientation error [rad] over the recorded frames."""
    position, rotation = [], []
    for world, toss in enumerate(tosses):
        n = min(len(toss["t"]), len(poses))
        predicted = poses[:n, world]
        position.append(np.mean(np.linalg.norm(predicted[:, :3] - toss["pos"][:n], axis=1)))
        dot = np.abs(np.sum(predicted[:, 3:7] * toss["quat"][:n], axis=1))
        rotation.append(np.mean(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))
    return np.asarray(position), np.asarray(rotation)


def evaluate(tosses: list[dict], params: dict | None = None) -> dict:
    """Mean position [m] and orientation [rad] error over all tosses."""
    model = build_model(len(tosses), params)
    poses = rollout(model, make_solver(model), tosses, max(len(toss["t"]) for toss in tosses))
    position, rotation = toss_errors(tosses, poses)
    return {"position_m": float(np.mean(position)), "rotation_rad": float(np.mean(rotation))}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.tosses = load_tosses(args.data or Path(__file__).with_name("tosses.npz"))
        self.model = build_model(len(self.tosses))
        self.solver = make_solver(self.model)
        self.frame_dt = FRAME_DT
        self.sim_time = 0.0
        self.state_0, self.state_1 = initial_state(self.model, self.tosses), self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = make_pipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        simulate_frame(
            self.model, self.solver, self.collision_pipeline, self.contacts, self.state_0, self.state_1, self.control
        )

    def step(self):
        """Advance all tosses by one camera frame."""
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
        errors = evaluate(self.tosses[:20])
        assert np.isfinite(errors["position_m"])

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--data", type=str, default=None, help="Toss recordings (npz)")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    print(evaluate(example.tosses))
