# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical replay of a real pick-and-place episode from ABC-130k: put the screwdriver in the bin.

The recording comes from ABC-130k (https://abc.bot/#data), episode "put the
screwdriver in the bin": at a bimanual station of two 6-DoF YAM arms with
parallel grippers, teleoperated by streaming joint-position commands from
leader arms at about 30 Hz, the right arm picks up a screwdriver lying on the
table by its handle, carries it over, and drops it into a pink plastic bin.
``episodes/main.npz`` holds the measured joints, gripper openings and efforts,
and the logged commands; ``scenes/main.json`` holds the station layout (arm
bases, the bin, and the screwdriver: its handle profile, start pose, grasping
arm, and the camera frames of the real grasp events). FORMAT.md describes the
files and the clocks; ``gt/`` and ``frames/`` hold the ground truth and the
recorded top and wrist videos.

``build_model`` builds one Newton world per scene: the station MJCF (station/,
the ABC simulator's model, whose actuator gains, armature, joint friction,
effort limits, gravity compensation, gripper gains, and contact settings it
keeps; friction coefficients of the station shapes are clipped to the task's
material bounds) with the arm bases at the scene's positions, the bin, and the
screwdriver as a free rigid body. ``replay_common.Replay`` drives every world
open loop with the logged commands (zero-order hold, delayed by
``PARAMS["command_delay"]``) as joint position targets and steps the solver
from ``make_solver`` with the collision pipeline from ``make_pipeline``;
``replay_common.score`` compares the replay with the real episode.

Run: ``python screwdriver_replay.py --viewer null`` replays the whole episode
and prints the metrics (``--seconds`` stops early, ``--num-worlds`` adds
copies, ``--jitter`` perturbs their starts, ``--episode sib_1`` selects another
episode).
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
# Upper bounds of the station shapes' friction coefficients: the MJCF's (sliding 2-4, torsional 0.1 m,
# rolling 0.01 m) are clipped to the task's material bounds.
STATION_MATERIAL_MAX = {"mu": 1.5, "mu_torsional": 0.01, "mu_rolling": 0.001}
TABLE_MU = 0.5
# Screwdriver: rubber handle and steel shaft. Torsional friction [m] is a contact patch radius, rolling
# friction [m] a rolling-resistance length.
HANDLE_MATERIAL = {"mu": 1.0, "mu_torsional": 0.004, "mu_rolling": 0.0001}
SHAFT_MATERIAL = {"mu": 0.4, "mu_torsional": 0.0016, "mu_rolling": 0.0001}
BIN_MASS = 0.35  # [kg] (polypropylene)
BIN_MU = 0.4
# Contacts are detected this far ahead of touching [m]; Newton's default (0.1 m) fills the contact
# buffers of large batches with distant shape pairs.
CONTACT_GAP = 0.01
START_CLEARANCE = 0.0005  # the screwdriver starts this far above its resting height [m]
HANDLE_CAPSULES_MAX = 5  # capsules approximating the handle profile (the shaft is one more)
PROFILE_TOLERANCE = 0.0005  # largest radius difference [m] of a handle capsule from the profile, if possible
HANDLE_COLOR = (0.95, 0.80, 0.05)
SHAFT_COLOR = (0.75, 0.75, 0.78)
BIN_COLOR = (0.93, 0.66, 0.60)


def build_station() -> newton.ModelBuilder:
    """One station from the MJCF (arm bases where the MJCF puts them)."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.rigid_gap = CONTACT_GAP
    # The bin may reach past the MJCF's back wall (x = 0.9 m); the real station's wall stands farther back.
    builder.add_mjcf(str(STATION), ignore_names=["back_wall_collision"])
    for i, label in enumerate(builder.shape_label):
        builder.shape_material_mu[i] = min(builder.shape_material_mu[i], STATION_MATERIAL_MAX["mu"])
        builder.shape_material_mu_torsional[i] = min(
            builder.shape_material_mu_torsional[i], STATION_MATERIAL_MAX["mu_torsional"]
        )
        builder.shape_material_mu_rolling[i] = min(
            builder.shape_material_mu_rolling[i], STATION_MATERIAL_MAX["mu_rolling"]
        )
        if label.endswith("table_plane"):
            builder.shape_material_mu[i] = TABLE_MU
    return builder


def add_bin(builder: newton.ModelBuilder, scene: dict) -> None:
    """The bin as one dynamic body labelled ``bin``: a floor slab and four walls leaning out to the rim.

    Bin frame x is the bin's length (scene ``bin``: footprint centre and yaw, outer sizes at the table and at
    the rim, height, wall and floor thickness).
    """
    spec, table_z = scene["bin"], scene["table_z"]
    height, wall, floor = spec["height_m"], spec["wall_m"], spec["floor_m"]
    (lb, wb), (lt, wt) = spec["bottom_size_m"], spec["top_size_m"]
    parts = [((0.0, 0.0, floor / 2), wp.quat_identity(), (lb / 2, wb / 2, floor / 2))]  # (centre, rotation, half)
    for axis, (bottom, top, other_b, other_t) in enumerate(((lb, lt, wb, wt), (wb, wt, lb, lt))):
        lean = (top - bottom) / 2  # outward lean of a wall over the height
        tilt = math.atan2(lean, height)
        slant = math.hypot(lean, height)
        for sign in (-1.0, 1.0):
            centre = [0.0, 0.0, height / 2]
            centre[axis] = (bottom / 2 + lean / 2 - wall / 2 * math.cos(tilt)) * sign
            if axis == 0:  # walls at +-x, spanning y, tilted about y
                rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), sign * tilt)
                half = (wall / 2, max(other_b, other_t) / 2, slant / 2)
            else:  # walls at +-y, spanning x, tilted about x
                rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -sign * tilt)
                half = (max(other_b, other_t) / 2, wall / 2, slant / 2)
            parts.append((tuple(centre), rotation, half))
    cx, cy = spec["center_xy"]
    root = wp.transform(
        wp.vec3(cx, cy, table_z + 0.0005),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(spec["yaw_deg"])),
    )
    body = builder.add_body(xform=root, label=rc.BIN_BODY)
    volume = sum(8.0 * h[0] * h[1] * h[2] for _, _, h in parts)
    cfg = newton.ModelBuilder.ShapeConfig(density=BIN_MASS / volume, mu=BIN_MU)
    for k, (centre, rotation, half) in enumerate(parts):
        builder.add_shape_box(
            body,
            xform=wp.transform(wp.vec3(*centre), rotation),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=cfg,
            color=BIN_COLOR,
            label=f"bin_{k}",
        )


def handle_capsules(profile: np.ndarray) -> list[tuple[float, float, float]]:
    """Capsules ``(start x, end x, radius)`` [m] that follow a handle profile ``[[x, radius], ...]``.

    Each profile segment becomes a capsule of its mid radius (between the segment's largest and smallest
    radius); the worst-fitting pieces are halved until no piece's radius differs from the profile by more than
    :data:`PROFILE_TOLERANCE` or there are :data:`HANDLE_CAPSULES_MAX` capsules. All caps stay within the
    handle's ends (so the ends are rounded); a piece too short for that is left to its neighbours' caps.
    """
    x0, xn = float(profile[0, 0]), float(profile[-1, 0])
    pieces = [[float(profile[k, 0]), float(profile[k + 1, 0])] for k in range(len(profile) - 1)]

    def fit(piece):  # (start, end, radius, largest radius difference) of a piece's capsule
        radii = np.interp(np.linspace(piece[0], piece[1], 33), profile[:, 0], profile[:, 1])
        radius = 0.5 * (radii.max() + radii.min())
        return max(piece[0], x0 + radius), min(piece[1], xn - radius), radius, 0.5 * (radii.max() - radii.min())

    while True:
        kept = [(piece, fit(piece)) for piece in pieces if fit(piece)[1] > fit(piece)[0]]
        worst, (*_, error) = max(kept, key=lambda item: item[1][3])
        if len(kept) >= HANDLE_CAPSULES_MAX or error <= PROFILE_TOLERANCE:
            return [(float(a), float(b), float(r)) for _, (a, b, r, _) in kept]
        i, middle = pieces.index(worst), 0.5 * (worst[0] + worst[1])
        pieces[i : i + 1] = [[worst[0], middle], [middle, worst[1]]]


def add_screwdriver(builder: newton.ModelBuilder, scene: dict) -> None:
    """The screwdriver as one free body labelled ``screwdriver``, resting at the scene's start pose.

    The body frame is the scene's (origin at the grasp point on the axis, +x toward the tip). The handle is
    a chain of capsules along the profile (:func:`handle_capsules`), the shaft a capsule from the collar to
    the tip; the handle and the shaft have uniform densities that give the scene's mass and centre of mass.
    """
    obj = scene["objects"]["screwdriver"]
    geometry = obj["geometry"]
    profile = np.asarray(geometry["handle_profile"], dtype=np.float64)
    shaft_radius = float(geometry["shaft_radius_m"])
    collar, tip = float(profile[-1, 0]), float(obj["tip_local"][0])
    handle = handle_capsules(profile)
    shaft = (collar + shaft_radius, tip - shaft_radius, shaft_radius)  # end caps inside the shaft's ends

    def volume_and_centroid(start, end, radius):
        length = end - start
        return math.pi * radius * radius * length + 4.0 / 3.0 * math.pi * radius**3, 0.5 * (start + end)

    handle_parts = [volume_and_centroid(*part) for part in handle]
    handle_volume = sum(v for v, _ in handle_parts)
    handle_x = sum(v * x for v, x in handle_parts) / handle_volume
    shaft_volume, shaft_x = volume_and_centroid(*shaft)
    mass, com_x = float(geometry.get("mass_kg", 0.07)), float(obj["com_local"][0])
    handle_mass = mass * float(np.clip((shaft_x - com_x) / (shaft_x - handle_x), 0.05, 0.95))

    start = obj["start"]
    position = np.asarray(start["pos"], dtype=np.float64)
    position[2] += START_CLEARANCE
    body = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat(*start["quat_xyzw"])), label=rc.OBJECTS[0])
    along_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)  # capsule axis (z) -> body x
    # condim 6 makes MuJoCo apply the torsional and rolling friction.
    attributes = {"mujoco:condim": 6}
    parts = [
        (part, handle_mass / handle_volume, HANDLE_MATERIAL, HANDLE_COLOR, f"handle_{k}")
        for k, part in enumerate(handle)
    ]
    parts.append((shaft, (mass - handle_mass) / shaft_volume, SHAFT_MATERIAL, SHAFT_COLOR, "shaft"))
    for (x0, x1, radius), density, material, color, name in parts:
        builder.add_shape_capsule(
            body,
            xform=wp.transform(wp.vec3(0.5 * (x0 + x1), 0.0, 0.0), along_x),
            radius=radius,
            half_height=0.5 * (x1 - x0),
            cfg=newton.ModelBuilder.ShapeConfig(density=float(density), **material),
            color=color,
            label=f"screwdriver_{name}",
            custom_attributes=attributes,
        )


def build_world(station: newton.ModelBuilder, scene: dict) -> newton.ModelBuilder:
    """One world: a copy of the station with the scene's arm bases, the bin, and the screwdriver."""
    world = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(world)
    world.rigid_gap = CONTACT_GAP
    world.add_builder(station)
    rc.set_arm_bases(world, scene["bases"])
    add_bin(world, scene)
    add_screwdriver(world, scene)
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


def load_reference(name: str) -> dict | None:
    """Ground truth of an episode for :func:`replay_common.score` (``gt/<name>.npz``), if present."""
    path = HERE / "gt" / f"{name}.npz"
    return rc.load_gt(path) if path.exists() else None


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.name = Path(args.episode).stem
        self.episode = rc.load_episode(args.episode)
        scene = rc.load_scene(self.name if args.scene is None else args.scene)
        # Extra worlds are copies of the scene; with --jitter their screwdriver starts are perturbed like the
        # verifier's.
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
        """Print per-world metrics, :func:`replay_common.check` failures, and the ensemble summary."""
        results = self.evaluate() if results is None else results

        def show(value, scale=1.0, digits=1):
            return "-" if value is None else f"{value * scale:.{digits}f}"

        for world, result in enumerate(results):
            arm = result["arm_rmse_rad"]
            print(
                f"world {world}: held={result['all_held']} placed={result['all_placed']} "
                f"complete={result['complete']} arm_rmse_rad left={arm['left']:.4f} right={arm['right']:.4f}"
            )
            for name, m in result["objects"].items():
                print(
                    f"  {name} held={m['held_fraction']:.2f} rot={show(m['inhand_rot_deg'])}deg "
                    f"slip={show(m['slip_m'], 1000)}mm gap_err={show(m['grip_gap_err_mm'])}mm "
                    f"liftoff_err={show(m['liftoff_err_rows'], 1, 0)}rows track={show(m['carry_track_err_m'], 100)}cm "
                    f"moved={show(m['moved_before_grasp_m'], 100)}cm placed={m['placed']} "
                    f"tip_wall={show(m['tip_wall_m'], 100)}cm com_wall={show(m['com_wall_m'], 100)}cm "
                    f"speed={show(m['final_speed_mps'], 100, 2)}cm/s final_tip_err={show(m['final_tip_xy_err_m'], 100)}cm "
                    f"yaw_err={show(m['final_yaw_err_deg'])}deg max_rise={show(m['max_rise_m'], 100)}cm"
                )
            failed = rc.check(result)["failed"]
            print(f"  check: {'pass' if not failed else 'fail ' + ', '.join(failed)}")
        if len(results) > 1:
            summary = rc.summarize(results)
            arm = summary["arm_rmse_rad"]
            for name, m in summary["objects"].items():
                print(
                    f"summary ({summary['copies']} copies): {name} held {m['held']} placed {m['placed']} "
                    f"held+placed+rot<={rc.HELDOUT_ROT_MAX_DEG:g}deg {m['held_placed_rot_ok']}; medians: "
                    f"rot {show(m['inhand_rot_deg'])}deg slip {show(m['slip_m'], 1000)}mm "
                    f"gap_err {show(m['grip_gap_err_mm'])}mm arm_rmse left {arm['left']:.4f} right {arm['right']:.4f}"
                )
        return results

    def test_final(self):
        results = self.report()
        for recording in self.replay.recordings():
            for name in rc.OBJECTS:
                assert np.all(np.isfinite(recording[f"{name}_pos"])), f"{name} left the valid state space"
        assert all(len(result["objects"]) == len(rc.OBJECTS) for result in results)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--episode", default="main", help="Episode name (episodes/<name>.npz) or .npz path")
        parser.add_argument("--scene", default=None, help="Scene name or .json path (default: the episode's name)")
        parser.add_argument("--num-worlds", type=int, default=1, help="Copies of the scene, one world each")
        parser.add_argument("--jitter", action="store_true", help="Perturb the screwdriver starts of the extra copies")
        parser.add_argument("--seed", type=int, default=0, help="Seed of --jitter")
        parser.add_argument("--seconds", type=float, default=None, help="Replay only this long [s] (main run)")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    wall = example.run(args.seconds)
    print(f"replayed {example.sim_time:.2f} s of {len(example.scenes)} world(s) in {wall:.1f} s")
    example.report()
