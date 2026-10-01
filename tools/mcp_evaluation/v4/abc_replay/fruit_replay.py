# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical replay of a real bimanual pick-and-place episode from ABC-130k.

The recording comes from ABC-130k (https://abc.bot/#data), episode "place and
organize the fake fruits in the fruit bowl": two 6-DoF YAM arms with parallel
grippers, teleoperated by streaming joint-position commands from leader arms at
about 30 Hz, pick up three fake fruits one after another (a pear and an orange
with the left arm, a dark fruit with the right arm) and put them into a
wedge-shaped tray. ``episodes/main.npz`` holds the measured joints and gripper
openings and the logged commands; ``scenes/main.json`` holds the station layout
(arm bases, tray outline, and per fruit its size, start pose, grasping arm, and
the camera frames of the real grasp events). FORMAT.md describes the files and
the clocks; ``video_reference.npz`` and ``frames/`` are the recorded video and
the fruit positions tracked in it.

``build_model`` builds one Newton world per scene: the station MJCF (station/,
the ABC simulator's model, whose actuator gains, armature, joint friction,
effort limits, and gravity compensation it keeps) with the arm bases at the
scene's positions, a tray, and the fruits as free rigid bodies.
``replay_common.Replay`` drives every world open loop with the logged commands
(zero-order hold, delayed by ``PARAMS["command_delay"]``) as joint position
targets and steps the solver from ``make_solver`` with the collision pipeline
from ``make_pipeline``; ``replay_common.score`` compares the replay with the
real episode.

Run: ``python fruit_replay.py --viewer null`` replays the whole episode and
prints the metrics (``--seconds`` stops early, ``--num-worlds`` adds copies).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import replay_common as rc
import warp as wp

import newton
import newton.examples

HERE = Path(__file__).resolve().parent
STATION = HERE / "station" / "yam_bimanual_empty.xml"
FRAME_DT = 1.0 / 30.0  # camera frame period [s]

# ---------------- Replay parameters (tunable) ----------------
PARAMS = {
    "command_delay": 0.0,  # latency from a logged command to the joint targets [s]
    "dt": 0.002,  # physics step [s]
}
FRUIT_MASS = 0.05  # mass of each fruit [kg] (hollow plastic)
# Contacts are detected this far ahead of touching [m]; Newton's default (0.1 m) fills the contact
# buffers of large batches with distant shape pairs.
CONTACT_GAP = 0.01
START_CLEARANCE = 0.0005  # fruits start this far above their resting height [m]
TRAY_WALL = 0.008  # rim wall thickness [m]
FRUIT_COLORS = {"pear": (0.85, 0.75, 0.05), "orange": (0.95, 0.35, 0.03), "dark_fruit": (0.22, 0.07, 0.07)}
TRAY_COLOR = (0.80, 0.55, 0.36)


def build_station() -> newton.ModelBuilder:
    """One station from the MJCF (arm bases where the MJCF puts them)."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.rigid_gap = CONTACT_GAP
    # The tray reaches past the MJCF's back wall (x = 0.9 m); the real station's wall stands farther back.
    builder.add_mjcf(str(STATION), ignore_names=["back_wall_collision"])
    return builder


def add_tray(builder: newton.ModelBuilder, scene: dict) -> None:
    """Static flat tray on the table: a floor slab over the scene's sector footprint and a vertical rim."""
    tray, table_z = scene["tray"], scene["table_z"]
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    outline = rc.sector_polygon(tray)
    count = len(outline)
    floor = table_z + tray["floor_height_m"]
    vertices = np.vstack(
        [np.c_[outline, np.full(count, table_z + 0.0005)], np.c_[outline, np.full(count, floor)]]
    ).astype(np.float32)
    builder.add_shape_convex_hull(
        -1, mesh=newton.Mesh(vertices, _prism_indices(count)), cfg=cfg, color=TRAY_COLOR, label="tray_floor"
    )
    rim = tray["rim_height_m"]
    centroid = outline.mean(axis=0)
    for k in range(count):
        a, b = outline[k], outline[(k + 1) % count]
        edge = b - a
        length = float(np.hypot(*edge))
        inward = np.array([-edge[1], edge[0]]) / length
        if np.dot(inward, centroid - 0.5 * (a + b)) < 0.0:
            inward = -inward
        # The outline is the rim's outer edge.
        middle = 0.5 * (a + b) + inward * TRAY_WALL / 2
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(float(middle[0]), float(middle[1]), table_z + rim / 2),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.atan2(edge[1], edge[0])),
            ),
            hx=length / 2 + TRAY_WALL / 2,
            hy=TRAY_WALL / 2,
            hz=rim / 2,
            cfg=cfg,
            color=TRAY_COLOR,
            label=f"tray_rim_{k}",
        )


def add_fruits(builder: newton.ModelBuilder, scene: dict) -> None:
    """The fruits as free solid bodies of the scene's nominal sizes, resting at its start poses.

    Spheres for the round fruits; the pear is an ellipsoid lying on its side with its long axis along body x.
    Bodies are labelled with the fruit names.
    """
    for name in rc.FRUITS:
        fruit = scene["fruits"][name]
        size = fruit["size_m"]
        if fruit["shape"] == "pear":
            semi_axes = (size["length"] / 2, size["width"] / 2, size["width"] / 2)
        else:
            semi_axes = (size["diameter"] / 2,) * 3
        x, y, _ = fruit["start"]["pos"]
        body = builder.add_body(
            xform=wp.transform(
                wp.vec3(x, y, scene["table_z"] + semi_axes[2] + START_CLEARANCE),
                wp.quat(*fruit["start"]["quat_xyzw"]),
            ),
            label=name,
        )
        volume = 4.0 / 3.0 * math.pi * semi_axes[0] * semi_axes[1] * semi_axes[2]
        cfg = newton.ModelBuilder.ShapeConfig(density=FRUIT_MASS / volume)
        if semi_axes[0] == semi_axes[1] == semi_axes[2]:
            builder.add_shape_sphere(body, radius=semi_axes[0], cfg=cfg, color=FRUIT_COLORS[name], label=f"{name}_geom")
        else:
            rx, ry, rz = semi_axes
            builder.add_shape_ellipsoid(
                body, rx=rx, ry=ry, rz=rz, cfg=cfg, color=FRUIT_COLORS[name], label=f"{name}_geom"
            )


def build_world(station: newton.ModelBuilder, scene: dict) -> newton.ModelBuilder:
    """One world: a copy of the station with the scene's arm bases, the tray, and the fruits."""
    world = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(world)
    world.rigid_gap = CONTACT_GAP
    world.add_builder(station)
    rc.set_arm_bases(world, scene["bases"])
    add_tray(world, scene)
    add_fruits(world, scene)
    return world


def build_model(scenes: list[dict]) -> newton.Model:
    """Model with one world per scene; the worlds share one layout (shape types and counts)."""
    station = build_station()  # parsed once (about 1-2 s), copied into every world
    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    for scene in scenes:
        top.add_world(build_world(station, scene))
    return top.finalize()


def make_solver(model: newton.Model) -> newton.solvers.SolverBase:
    return newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)


def make_pipeline(model: newton.Model) -> newton.CollisionPipeline | None:
    return newton.CollisionPipeline(model)


# -----------------------------------------------------------


def _prism_indices(count: int) -> np.ndarray:
    """Triangles of a prism over a convex outline of ``count`` points (bottom ring, then top ring)."""
    triangles = []
    for k in range(1, count - 1):
        triangles += [(0, k + 1, k), (count, count + k, count + k + 1)]
    for k in range(count):
        a, b = k, (k + 1) % count
        triangles += [(a, b, count + b), (a, count + b, count + a)]
    return np.asarray(triangles, dtype=np.int32).reshape(-1)


def load_reference(name: str) -> dict | None:
    """Ground truth of an episode for :func:`replay_common.score` (``video_reference.npz`` for ``main``)."""
    path = HERE / ("video_reference.npz" if name == "main" else f"gt/{name}.npz")
    return rc.load_gt(path) if path.exists() else None


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.name = Path(args.episode).stem
        self.episode = rc.load_episode(args.episode)
        scene = rc.load_scene(self.name if args.scene is None else args.scene)
        # Extra worlds are copies of the scene; with --jitter their fruit starts are perturbed like the verifier's.
        rng = np.random.default_rng(args.seed)
        self.scenes = [scene] + [
            rc.jitter_scene(scene, rng) if args.jitter else scene for _ in range(args.num_worlds - 1)
        ]
        self.reference = load_reference(self.name)
        self.camera = rc.load_camera() if (HERE / "camera.json").exists() else None

        self.model = build_model(self.scenes)
        self.solver = make_solver(self.model)
        self.collision_pipeline = make_pipeline(self.model)
        self.replay = rc.Replay(
            self.model,
            self.solver,
            self.collision_pipeline,
            self.episode,
            dt=PARAMS["dt"],
            command_delay=PARAMS["command_delay"],
        )
        # The MCP host binds these, and its checkpoints rewind the replay's cursor and history arrays.
        self.state_0, self.state_1 = self.replay.state_0, self.replay.state_1
        self.control, self.contacts = self.replay.control, self.replay.contacts
        for attribute, array in self.replay.checkpoint_arrays().items():
            setattr(self, attribute, array)
        self.graph = self.replay.graph

        self.sim_dt = PARAMS["dt"]
        self.substeps = max(1, round(FRAME_DT / self.sim_dt))
        self.frame_dt = self.substeps * self.sim_dt
        self.sim_time = 0.0
        self.viewer.set_model(self.model)

    def capture(self):
        """Re-record the replay step, e.g. after replacing the solver or the collision pipeline."""
        self.replay.solver, self.replay.collision_pipeline = self.solver, self.collision_pipeline
        self.replay.contacts = self.contacts
        self.replay.capture()
        self.graph = self.replay.graph

    def step(self):
        self.replay.step(self.substeps)
        self.sim_time += self.frame_dt

    def run(self, seconds: float | None = None) -> float:
        """Replay to ``seconds`` after the episode start (default: the end); returns the wall time [s]."""
        wall = self.replay.run(seconds)
        self.sim_time = self.replay.sim_time
        return wall

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def evaluate(self) -> list[dict]:
        """Metrics of every world so far (:func:`replay_common.score`), one dict per world."""
        recordings = self.replay.recordings()
        return [
            rc.score(recording, scene, self.reference, self.camera)
            for recording, scene in zip(recordings, self.scenes, strict=True)
        ]

    def report(self, results: list[dict] | None = None) -> list[dict]:
        """Print per-world metrics and :func:`replay_common.check` failures."""
        results = self.evaluate() if results is None else results

        def show(value, scale=1.0, digits=1):
            return "-" if value is None else f"{value * scale:.{digits}f}"

        for world, result in enumerate(results):
            arm = result["arm_rmse_rad"]
            print(
                f"world {world}: all_held={result['all_held']} all_placed={result['all_placed']} "
                f"complete={result['complete']} arm_rmse_rad left={arm['left']:.4f} right={arm['right']:.4f}"
            )
            for name, m in result["fruits"].items():
                print(
                    f"  {name:10s} held={m['held_fraction']:.2f} lifted={m['lifted_fraction']:.2f} "
                    f"rise={show(m['max_rise_m'], 1000)}mm liftoff_err={show(m['liftoff_err_s'], 1000, 0)}ms "
                    f"release_err={show(m['release_err_s'], 1000, 0)}ms track={show(m['carry_track_err_m'], 100)}cm "
                    f"gap_err={show(m['grip_gap_err_mm'])}mm slip={show(m['slip_m'], 1000)}mm placed={m['placed']} "
                    f"tray_dist={show(m['tray_distance_m'], 100)}cm final_err={show(m['final_xy_err_m'], 100)}cm"
                )
            failed = rc.check(result)["failed"]
            print(f"  check: {'pass' if not failed else 'fail ' + ', '.join(failed)}")
        return results

    def test_final(self):
        results = self.report()
        for recording in self.replay.recordings():
            for name in rc.FRUITS:
                assert np.all(np.isfinite(recording[f"{name}_pos"])), f"{name} left the valid state space"
        assert all(len(result["fruits"]) == len(rc.FRUITS) for result in results)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--episode", default="main", help="Episode name (episodes/<name>.npz) or .npz path")
        parser.add_argument("--scene", default=None, help="Scene name or .json path (default: the episode's name)")
        parser.add_argument("--num-worlds", type=int, default=1, help="Copies of the scene, one world each")
        parser.add_argument("--jitter", action="store_true", help="Perturb the fruit starts of the extra copies")
        parser.add_argument("--seed", type=int, default=0, help="Seed of --jitter")
        parser.add_argument("--seconds", type=float, default=None, help="Replay only this long [s] (main run)")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    wall = example.run(args.seconds)
    print(f"replayed {example.sim_time:.2f} s of {len(example.scenes)} world(s) in {wall:.1f} s")
    example.report()
