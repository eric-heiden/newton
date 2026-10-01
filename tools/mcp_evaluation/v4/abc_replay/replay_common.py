# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fixed helpers for the ABC fruit-bowl physical replay task.

The task replays a real ABC-130k episode ("place and organize the fake fruits
in the fruit bowl") in Newton: two YAM arms are driven open loop by the
recorded teleoperation commands and must grasp, carry, and place three fake
fruits the way the real robot did. The starter script and the verifier share
this module (the verifier uses its own copy, so edits here have no effect on
verification). It provides:

- loading of episodes, scenes, ground truth, and camera calibration
  (:func:`load_episode`, :func:`load_scene`, :func:`load_gt`, :func:`load_camera`),
- the command schedule (:func:`command_schedule`): zero-order hold of the
  logged commands, delayed by the controller latency,
- :class:`Replay`, which writes the per-world joint targets and steps a
  submitted model/solver/collision pipeline under a CUDA graph, recording the
  simulation at the episode's sample times,
- :func:`score`, the replay metrics the verifier thresholds, and :func:`check`,
- :class:`StationRenderer` (:func:`render_top`, :func:`render_wrist`), which
  renders the station through the real cameras' calibrated intrinsics,
- :func:`contact_summary`, contact counts and forces between shape sets,
- :class:`StationFK`, forward kinematics of the measured joints (FK-consistent
  fruit starts and FK-attached carry tracks).

File formats and the time-base convention are described in FORMAT.md.
"""

from __future__ import annotations

import copy
import json
import math
import time
import warnings
import weakref
from pathlib import Path

import numpy as np
import warp as wp

import newton

HERE = Path(__file__).resolve().parent
STATION_XML = HERE / "station" / "yam_bimanual_empty.xml"
FRUITS = ("pear", "orange", "dark_fruit")
SIDES = ("left", "right")
FRAME_RATE = 30.0  # nominal camera and arm-state sample rate [Hz]
GRIPPER_TRAVEL = 0.0475  # finger slide travel [m] at a gripper opening of 1
# Finger pad geometry in the pad body frame (bodies "<side>_lf_down" and "<side>_rf_down"): the pad
# box centre, and the grasp point 25 mm from it toward the fingertip along the pad's long axis (+z).
PAD_CENTRE = (0.0, -0.0024, 0.046)
PAD_POINT = (0.0, -0.0024, 0.071)
# MJCF cameras look along -Z of a frame rotated 180 degrees about x from their body; Newton cameras
# look along their own -Z with +Y up, so a camera's pose is body pose * this rotation.
CAMERA_IN_BODY_XYZW = (1.0, 0.0, 0.0, 0.0)
CAMERA_BODY = {"left": "left_camera_frame", "right": "right_camera_frame"}
HAND_BODY = "{side}_link_6"

# Definitions used by score().
HOLD_MARGIN = 0.015  # held: fruit centre within fruit radius + this of the grasping arm's pad midpoint [m]
LIFT_THRESHOLD = 0.01  # lift-off: fruit centre this far above its start [m]
RELEASE_THRESHOLD = 0.01  # release: fruit this far from its in-hand position at the carry start [m]
PLACE_MARGIN = 0.01  # placed: fruit centre within the tray sector dilated by this [m]
REST_HEIGHT_TOLERANCE = 0.015  # placed: centre height within this of tray floor + fruit radius [m]
REST_SPEED = 0.02  # placed: mean speed over the last REST_WINDOW below this [m/s]
REST_WINDOW = 0.3  # [s]
CARRY_START_OFFSET = 3  # carry window: state samples after lift-off ...
CARRY_END_OFFSET = 2  # ... to state samples before release

# Per-rollout limits used by check(): the single-rollout counterparts of the verification gates, which apply
# them to ensembles (counts of copies and medians over copies).
DEFAULT_THRESHOLDS = {
    "held_fraction": 0.9,  # min, fraction of carry samples held
    "lifted_fraction": 0.9,  # min, fraction of carry samples at least half as high as the real fruit
    "moved_before_grasp_m": 0.02,  # max
    "liftoff_err_s": 0.11,  # max |sim - real| lift-off time (3 state samples at 30 Hz)
    "release_err_s": (-0.10, 0.10),  # allowed sim - real release time
    "carry_track_err_m": 0.0343,  # max median xy distance to the FK-attached ground truth while carried
    "grip_gap_err_mm": 5.0,  # max |median finger gap error| while holding
    "arm_rmse_rad": 0.0261,  # max whole-episode joint RMSE per arm
    "final_xy_err_m": 0.142,  # max distance of the final position from the real one
}

_EPISODE_KEYS = ("t", "q", "cmd_t", "cmd", "grip_t", "grip", "grip_cmd_t", "grip_cmd")
_EVENT_KEYS = ("cmd_close", "contact", "liftoff", "cmd_open", "release")


# ----------------------------------------------------------------------------- data


def _resolve(path: str | Path, folder: str, suffix: str) -> Path:
    path = Path(path)
    if path.exists() or path.suffix:
        return path
    return HERE / folder / f"{path}{suffix}"


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    """Load an episode (.npz, see FORMAT.md).

    Args:
        path: File path, or an episode name resolved as ``episodes/<name>.npz`` next to this module.

    Returns:
        Arrays by key, e.g. ``left_q`` [n, 6] measured joints [rad] at the state times ``left_t`` [s].
    """
    with np.load(_resolve(path, "episodes", ".npz")) as data:
        episode = {key: np.asarray(data[key]) for key in data.files}
    required = ["t_top"] + [f"{side}_{key}" for side in SIDES for key in _EPISODE_KEYS]
    missing = [key for key in required if key not in episode]
    if missing:
        raise KeyError(f"episode is missing {missing}")
    return episode


def episode_from_lerobot(
    timestamp: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    *,
    velocity: np.ndarray | None = None,
    torque: np.ndarray | None = None,
    time_offset: float = 0.098,
    rate: float = FRAME_RATE,
    frame_lag: float = -2.0 / FRAME_RATE,
) -> dict[str, np.ndarray]:
    """Episode dict from a LeRobot copy of an ABC episode (10 fps), linearly interpolated onto a 30 Hz grid.

    Measured joints and gripper come from ``state``, commands from ``action``; both are interpolated onto
    the grid, which the replay then holds and delays like logged commands. LeRobot rows are copies of the
    MCAP samples nearest to each 10 fps tick.

    Args:
        timestamp: LeRobot sample times [s], shape [K].
        state: Measured values, shape [K, 14]: left joints 1-6 [rad], left gripper [0..1], right joints, right gripper.
        action: Commanded values in the same layout, shape [K, 14].
        velocity: Measured velocities in the same layout [rad/s] (``observation.velocity``); ``None`` for none.
        torque: Measured torques in the same layout [N m] (``observation.torque``); ``None`` for none.
        time_offset: Added to LeRobot times to reach the clock of the MCAP recordings (seconds from the first
            top-camera frame) [s]: the state stream's start minus the top camera's, 0.097-0.115 s
            (``source_episodes.json``; 0.098 s for the main episode's copy).
        rate: Grid rate [Hz].
        frame_lag: Camera latency [s]: a top frame shows the arm state this long before its tick.
            LeRobot top frame k is MCAP top frame 3k + 3 (the frame nearest to the tick), which shows state
            sample 3k + 2 (time-base rule), so frames show the state 2/30 s *after* their tick (-2/30).

    Returns:
        Episode dict in the format of :func:`load_episode`, with ``t_top`` the LeRobot frame times on the
        state clock and ``top_state_index`` the state sample each 10 fps top frame shows.
    """
    timestamp = np.asarray(timestamp, dtype=np.float64)
    state, action = np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)
    ts = timestamp + time_offset
    grid = ts[0] + np.arange(int(math.floor((ts[-1] - ts[0]) * rate + 1e-6)) + 1) / rate
    episode = {"t_top": ts.copy()}
    for side, base in (("left", 0), ("right", 7)):
        for key in ("t", "cmd_t", "grip_t", "grip_cmd_t"):
            episode[f"{side}_{key}"] = grid.copy()
        episode[f"{side}_q"] = np.stack([np.interp(grid, ts, state[:, base + j]) for j in range(6)], axis=-1)
        episode[f"{side}_cmd"] = np.stack([np.interp(grid, ts, action[:, base + j]) for j in range(6)], axis=-1)
        episode[f"{side}_grip"] = np.interp(grid, ts, state[:, base + 6])
        episode[f"{side}_grip_cmd"] = np.interp(grid, ts, action[:, base + 6])
        for key, measured_values in (("qd", velocity), ("tau", torque)):
            if measured_values is not None:
                columns = np.asarray(measured_values, dtype=np.float64)[:, base : base + 7]
                episode[f"{side}_{key}"] = np.stack([np.interp(grid, ts, columns[:, j]) for j in range(7)], axis=-1)
    episode["top_state_index"] = np.clip(np.round((ts - frame_lag - grid[0]) * rate), 0, len(grid) - 1).astype(np.int64)
    return episode


def frame_rows(episode: dict, camera: str = "top") -> np.ndarray:
    """State sample (row of ``left_t``) shown by each frame of a camera stream.

    Uses ``<camera>_state_index`` when the episode has it; otherwise frame i shows sample i - 1 (the
    measured latency of the main episode's cameras), and frame 0 the first sample.

    Args:
        episode: Episode dict.
        camera: ``"top"``, ``"left"``, or ``"right"`` (frame times ``t_<camera>``).

    Returns:
        Row index per frame, shape [frames], dtype int64.
    """
    key = f"{camera}_state_index"
    if key in episode:
        return np.asarray(episode[key], dtype=np.int64)
    frames = len(episode[f"t_{camera}"])
    return np.clip(np.arange(frames) - 1, 0, len(episode["left_t"]) - 1).astype(np.int64)


def measured(episode: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Measured arm joints [rad], shape [n, 6], and gripper opening [0..1], shape [n], at the state times ``left_t``."""
    t = episode["left_t"]
    q, grip = episode[f"{side}_q"][:, :6], episode[f"{side}_grip"]
    ts, tg = episode[f"{side}_t"], episode[f"{side}_grip_t"]
    if len(ts) != len(t) or not np.allclose(ts, t):
        q = np.stack([np.interp(t, ts, q[:, j]) for j in range(6)], axis=-1)
    if len(tg) != len(t) or not np.allclose(tg, t):
        grip = np.interp(t, tg, grip)
    return np.asarray(q, dtype=np.float64), np.asarray(grip, dtype=np.float64)


def load_scene(path: str | Path) -> dict:
    """Load a scene (.json, see FORMAT.md).

    Args:
        path: File path, or a scene name resolved as ``scenes/<name>.json`` next to this module.
    """
    scene = json.loads(_resolve(path, "scenes", ".json").read_text())
    for name in FRUITS:
        fruit = scene["fruits"][name]
        if fruit["arm"] not in SIDES:
            raise ValueError(f"{name}: arm must be one of {SIDES}")
        missing = [key for key in _EVENT_KEYS if key not in fruit["events"]]
        if missing:
            raise KeyError(f"{name}: events missing {missing}")
    return scene


def load_gt(path: str | Path) -> dict[str, np.ndarray]:
    """Load ground truth (.npz, see FORMAT.md) as a dict of arrays."""
    with np.load(Path(path)) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def gt_from_tracker(meta: dict, trajectories: dict) -> dict[str, np.ndarray]:
    """Ground truth in :func:`score`'s format from the video tracker output (ground_truth.json, trajectories.npz).

    Args:
        meta: Parsed ground_truth.json.
        trajectories: Arrays of trajectories.npz.
    """
    gt = {}
    for name in FRUITS:
        for key in ("pos", "src", "phase", "uv", "fix", "radpx"):
            if f"{name}_{key}" in trajectories:
                gt[f"{name}_{key}"] = np.asarray(trajectories[f"{name}_{key}"])
        gt[f"{name}_rest_xyz"] = np.asarray(meta["objects"][name]["rest_initial_xyz"], dtype=np.float64)
        gt[f"{name}_final_xyz"] = np.asarray(meta["objects"][name]["final_xyz"], dtype=np.float64)
    for side in SIDES:
        for key in ("pad", "gap", "grip"):
            if f"{side}_{key}" in trajectories:
                gt[f"{side}_{key}"] = np.asarray(trajectories[f"{side}_{key}"])
    return gt


def load_camera(path: str | Path | dict | None = None) -> dict:
    """Load the camera calibration (camera.json, see FORMAT.md); dicts pass through."""
    if isinstance(path, dict):
        return path
    return json.loads(Path(HERE / "camera.json" if path is None else path).read_text())


def jitter_scene(
    scene: dict,
    rng: np.random.Generator,
    *,
    xy_sigma: float = 0.002,
    xy_clip: float = 0.004,
    yaw_deg: float = 5.0,
) -> dict:
    """Copy of a scene with perturbed fruit starts, as in the verifier's ensembles.

    Each start moves by N(0, ``xy_sigma``) per axis, clipped to ``xy_clip`` [m]; elongated fruits
    (``shape == "pear"``) also turn about z by up to ``yaw_deg`` [deg] (uniform).
    """
    scene = copy.deepcopy(scene)
    for name in FRUITS:
        start = scene["fruits"][name]["start"]
        start["pos"][:2] = (
            np.asarray(start["pos"][:2]) + np.clip(rng.normal(0.0, xy_sigma, 2), -xy_clip, xy_clip)
        ).tolist()
        if scene["fruits"][name].get("shape") == "pear":
            turn = _quat_about_z(math.radians(rng.uniform(-yaw_deg, yaw_deg)))
            start["quat_xyzw"] = _quat_multiply(turn, np.asarray(start["quat_xyzw"], dtype=np.float64)).tolist()
    return scene


# ----------------------------------------------------------------------------- geometry


def _leaf(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _quat_about_z(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from quaternions (x, y, z, w), shape [..., 4] -> [..., 3, 3]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def set_arm_bases(builder: newton.ModelBuilder, bases: dict) -> None:
    """Move the arm bases (bodies ``left_arm`` and ``right_arm``, welded to the world) to the scene's positions.

    Apply to a builder holding one station (before :meth:`~newton.ModelBuilder.add_world`).

    Args:
        builder: Builder after ``add_mjcf`` of the station.
        bases: ``{"left": [x, y, z], "right": [x, y, z]}`` base positions [m] (scene ``bases``).
    """
    leaves = [_leaf(label) for label in builder.body_label]
    for side in SIDES:
        body = leaves.index(f"{side}_arm")
        joint = builder.joint_child.index(body)
        rotation = builder.joint_X_p[joint].q
        builder.joint_X_p[joint] = wp.transform(wp.vec3(*[float(v) for v in bases[side]]), rotation)


def sector_distance(xy, tray: dict) -> float:
    """Distance [m] from a point to the tray's circular-sector footprint (0 inside).

    Args:
        xy: Point in the world xy plane [m].
        tray: Scene ``tray`` (``apex_xy``, ``yaw_deg``, ``radius_m``, ``half_angle_deg``).
    """
    apex = np.asarray(tray["apex_xy"], dtype=np.float64)
    yaw, half = math.radians(tray["yaw_deg"]), math.radians(tray["half_angle_deg"])
    radius = float(tray["radius_m"])
    d = np.asarray(xy, dtype=np.float64)[:2] - apex
    r = float(np.hypot(*d))
    angle = (math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    if abs(angle) <= half:
        return max(0.0, r - radius)
    distances = []
    for sign in (-1.0, 1.0):
        edge = radius * np.array([math.cos(yaw + sign * half), math.sin(yaw + sign * half)])
        s = float(np.clip(np.dot(d, edge) / radius**2, 0.0, 1.0))
        distances.append(float(np.linalg.norm(d - s * edge)))
    return min(distances)


def sector_polygon(tray: dict, arc_points: int = 16) -> np.ndarray:
    """Tray outline in the world xy plane [m]: the apex, then points along the arc, shape [arc_points + 2, 2]."""
    apex = np.asarray(tray["apex_xy"], dtype=np.float64)
    yaw, half = math.radians(tray["yaw_deg"]), math.radians(tray["half_angle_deg"])
    angles = yaw + np.linspace(-half, half, arc_points + 1)
    arc = apex + tray["radius_m"] * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return np.vstack([apex, arc])


class StationFK:
    """Forward kinematics of the station arms at measured joint positions (CPU).

    Args:
        station: Station MJCF path.
        bases: Scene ``bases``; ``None`` keeps the MJCF placement.
    """

    def __init__(self, station: str | Path = STATION_XML, bases: dict | None = None):
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(station), parse_visuals=False)
        if bases is not None:
            set_arm_bases(builder, bases)
        self.model = builder.finalize(device="cpu")
        self.state = self.model.state()
        joints = {_leaf(label): i for i, label in enumerate(self.model.joint_label)}
        bodies = {_leaf(label): i for i, label in enumerate(self.model.body_label)}
        starts = self.model.joint_q_start.numpy()
        self.arm_coords = {s: [int(starts[joints[f"{s}_joint{j + 1}"]]) for j in range(6)] for s in SIDES}
        self.finger_coords = {
            s: (int(starts[joints[f"{s}_left_finger"]]), int(starts[joints[f"{s}_right_finger"]])) for s in SIDES
        }
        self.pad_bodies = {s: (bodies[f"{s}_lf_down"], bodies[f"{s}_rf_down"]) for s in SIDES}
        self.hand_body = {s: bodies[HAND_BODY.format(side=s)] for s in SIDES}

    def pose(self, arm_q: dict, grip: dict | None = None) -> np.ndarray:
        """Body poses with the arms at the given joints.

        Args:
            arm_q: ``{side: 6 joint angles [rad]}``.
            grip: ``{side: gripper opening [0..1]}``; ``None`` leaves the fingers closed.

        Returns:
            Body transforms (position, quaternion xyzw), shape [body_count, 7].
        """
        q = self.model.joint_q.numpy().copy()
        for side in SIDES:
            q[self.arm_coords[side]] = arm_q[side][:6]
            if grip is not None:
                travel = float(np.clip(grip[side], 0.0, 1.0)) * GRIPPER_TRAVEL
                q[list(self.finger_coords[side])] = (travel, -travel)
        self.state.joint_q.assign(q.astype(np.float32))
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        return self.state.body_q.numpy().astype(np.float64)

    def pose_row(self, episode: dict, row: int) -> np.ndarray:
        """Body poses [body_count, 7] at measured state sample ``row``."""
        values = {side: measured(episode, side) for side in SIDES}
        return self.pose({s: values[s][0][row] for s in SIDES}, {s: values[s][1][row] for s in SIDES})

    def pads(self, body_q: np.ndarray, side: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Both finger pads of an arm: (pad box centre [m], unit long axis toward the tip, grasp point [m])."""
        out = []
        for body in self.pad_bodies[side]:
            rotation = quat_to_matrix(body_q[body, 3:])
            origin = body_q[body, :3]
            out.append((origin + rotation @ PAD_CENTRE, rotation[:, 2], origin + rotation @ PAD_POINT))
        return out

    def grasp_point(self, body_q: np.ndarray, side: str) -> np.ndarray:
        """Midpoint of the two pads' grasp points [m] (the point :func:`score` measures holding against)."""
        return 0.5 * sum(pad[2] for pad in self.pads(body_q, side))

    def fruit_starts(self, episode: dict, scene: dict) -> dict[str, dict]:
        """Fruit start poses consistent with the real grasps.

        A fruit's xy is where the midpoint of the grasping arm's two pad axes meets the fruit's centre height,
        averaged over the frames from finger contact to the frame before lift-off (the fruit rests on the
        table then). Elongated fruits (``shape == "pear"``) lie with their long axis (body x) perpendicular to
        the closing direction just before lift-off.

        Args:
            episode: Episode dict.
            scene: Scene dict (fruit ``arm``, ``events``, ``centre_height_m``, ``shape``, and ``table_z``).

        Returns:
            ``{name: {"pos": [x, y, z], "quat_xyzw": [x, y, z, w]}}`` with z = table + centre height [m].
        """
        rows = frame_rows(episode)
        out = {}
        for name in FRUITS:
            fruit = scene["fruits"][name]
            side, events = fruit["arm"], fruit["events"]
            height = scene["table_z"] + fruit["centre_height_m"]
            points = []
            for frame in range(events["contact"], max(events["liftoff"] - 1, events["contact"] + 1)):
                pads = self.pads(self.pose_row(episode, int(rows[frame])), side)
                on_plane = [centre + axis * (height - centre[2]) / axis[2] for centre, axis, _ in pads]
                points.append(0.5 * (on_plane[0] + on_plane[1]))
            xy = np.mean(points, axis=0)[:2]
            quat = np.array([0.0, 0.0, 0.0, 1.0])
            if fruit.get("shape") == "pear":
                pads = self.pads(self.pose_row(episode, int(rows[max(events["liftoff"] - 1, 0)])), side)
                closing = pads[1][0] - pads[0][0]
                quat = _quat_about_z(math.atan2(closing[1], closing[0]) + math.pi / 2)
            out[name] = {"pos": [float(xy[0]), float(xy[1]), float(height)], "quat_xyzw": quat.tolist()}
        return out

    def attached_tracks(self, episode: dict, scene: dict) -> dict[str, np.ndarray]:
        """Fruit centres per top frame if each fruit stayed rigidly in its hand from contact to release.

        The in-hand offset is the scene start relative to the hand (``<side>_link_6``) at the contact frame.
        Frames before contact hold the start; frames from release on are NaN.

        Returns:
            ``{name: positions [frames, 3] [m]}``.
        """
        rows = frame_rows(episode)
        out = {}
        for name in FRUITS:
            fruit = scene["fruits"][name]
            side, events = fruit["arm"], fruit["events"]
            start = np.asarray(fruit["start"]["pos"], dtype=np.float64)
            track = np.full((len(rows), 3), np.nan)
            track[: events["contact"]] = start
            body = self.hand_body[side]
            pose = self.pose_row(episode, int(rows[events["contact"]]))[body]
            local = quat_to_matrix(pose[3:]).T @ (start - pose[:3])
            for frame in range(events["contact"], min(events["release"], len(rows))):
                pose = self.pose_row(episode, int(rows[frame]))[body]
                track[frame] = pose[:3] + quat_to_matrix(pose[3:]) @ local
            out[name] = track
        return out


# ----------------------------------------------------------------------------- commands


def command_schedule(episode: dict, times: np.ndarray, delay: float = 0.0) -> dict[str, dict[str, np.ndarray]]:
    """Joint targets at control times: the logged commands held from their timestamps, delayed by ``delay``.

    Args:
        episode: Episode dict.
        times: Control times on the episode's state clock [s], shape [T].
        delay: Controller latency [s]: the command logged at t takes effect at t + delay.

    Returns:
        ``{side: {"q": arm targets [T, 6] [rad], "finger": left finger slide targets [T] [m]}}``; the
        gripper target is clip(g, 0, 1) * :data:`GRIPPER_TRAVEL` (the right finger mirrors it).
    """
    times = np.asarray(times, dtype=np.float64)
    out = {}
    for side in SIDES:
        t = episode[f"{side}_cmd_t"]
        rows = np.clip(np.searchsorted(t, times - delay, side="right") - 1, 0, len(t) - 1)
        tg = episode[f"{side}_grip_cmd_t"]
        grip_rows = np.clip(np.searchsorted(tg, times - delay, side="right") - 1, 0, len(tg) - 1)
        out[side] = {
            "q": np.asarray(episode[f"{side}_cmd"], dtype=np.float64)[rows, :6],
            "finger": np.clip(np.asarray(episode[f"{side}_grip_cmd"], dtype=np.float64)[grip_rows], 0.0, 1.0)
            * GRIPPER_TRAVEL,
        }
    return out


# ----------------------------------------------------------------------------- replay


class _Station:
    """Per-world indices of the station joints and bodies in a batched model."""

    def __init__(self, model: newton.Model):
        if not model.joint_label or not model.body_label:
            raise ValueError("the model needs joint and body labels (add_mjcf of the station)")
        joint_world, body_world = model.joint_world.numpy(), model.body_world.numpy()
        joints, bodies = {}, {}
        for i, label in enumerate(model.joint_label):
            joints.setdefault((int(joint_world[i]), _leaf(label)), []).append(i)
        for i, label in enumerate(model.body_label):
            bodies.setdefault((int(body_world[i]), _leaf(label)), []).append(i)

        def one(table, world, name, kind):
            found = table.get((world, name), [])
            if len(found) != 1:
                raise ValueError(f"world {world} must have exactly one {kind} named {name!r}, found {len(found)}")
            return found[0]

        self.worlds = model.world_count
        self.arm_joints, self.finger_joints, self.fruit_bodies = [], [], []
        self.pad_bodies, self.hand_body, self.camera_body, self.world_bodies = [], [], [], []
        for w in range(self.worlds):
            self.arm_joints.append({s: [one(joints, w, f"{s}_joint{j + 1}", "joint") for j in range(6)] for s in SIDES})
            self.finger_joints.append(
                {
                    s: (one(joints, w, f"{s}_left_finger", "joint"), one(joints, w, f"{s}_right_finger", "joint"))
                    for s in SIDES
                }
            )
            self.fruit_bodies.append({name: one(bodies, w, name, "body") for name in FRUITS})
            self.pad_bodies.append(
                {s: (one(bodies, w, f"{s}_lf_down", "body"), one(bodies, w, f"{s}_rf_down", "body")) for s in SIDES}
            )
            self.hand_body.append({s: one(bodies, w, HAND_BODY.format(side=s), "body") for s in SIDES})
            self.camera_body.append({s: bodies.get((w, CAMERA_BODY[s]), [None])[0] for s in SIDES})
            self.world_bodies.append(np.flatnonzero(body_world == w))


@wp.kernel
def _apply_targets(
    schedule: wp.array2d[wp.float32],
    step: wp.array[wp.int32],
    index: wp.array[wp.int32],
    target: wp.array[wp.float32],
):
    j = wp.tid()
    row = wp.min(step[0], schedule.shape[0] - 1)
    target[index[j]] = schedule[row, j]


@wp.kernel
def _record(
    body_q: wp.array[wp.transformf],
    joint_q: wp.array[wp.float32],
    step: wp.array[wp.int32],
    every: int,
    history_body_q: wp.array2d[wp.transformf],
    history_joint_q: wp.array2d[wp.float32],
):
    i = wp.tid()
    done = step[0] + 1  # steps completed once this step's state is written
    if done % every != 0:
        return
    row = done // every
    if row >= history_body_q.shape[0]:
        return
    if i < body_q.shape[0]:
        history_body_q[row, i] = body_q[i]
    if i < joint_q.shape[0]:
        history_joint_q[row, i] = joint_q[i]


@wp.kernel
def _advance(step: wp.array[wp.int32]):
    step[0] = step[0] + 1


class Replay:
    """Open-loop replay of recorded episodes on a batched station model, one episode per world.

    Every physics step loads each world's joint targets from the precomputed command schedule
    (:func:`command_schedule`), runs the collision pipeline and the solver, and records all body
    poses and joint coordinates every ``record_every`` steps. The step cursor (:attr:`step_index`),
    schedule, and history live in Warp arrays, so one step is a fixed launch sequence that is
    captured once as a CUDA graph (host-loop fallback on CPU or if capture fails). Each world's
    episode starts at its first arm-state sample, with the arms at the measured joints, the fingers
    at the measured opening, and the fruits where ``model.joint_q`` placed them.

    For MCP checkpoints, expose :meth:`checkpoint_arrays` as attributes of the example (the MCP
    snapshots the example's own Warp arrays) together with :attr:`state_0` and :attr:`control`.

    Args:
        model: Batched station model (one world per episode; joints and bodies labelled as in the station
            MJCF, fruits as bodies ``pear``, ``orange``, ``dark_fruit``).
        solver: Newton solver for ``model``.
        pipeline: Collision pipeline, or ``None`` if the solver detects contacts itself.
        episodes: One episode dict per world, or a single one for all worlds.
        dt: Physics step [s].
        command_delay: Controller latency [s], scalar or one per world.
        contacts: Contacts buffer; defaults to ``pipeline.contacts()`` (``None`` without a pipeline).
        record_dt: Recording period [s] (rounded to a multiple of ``dt``).
        use_graph: Capture the step as a CUDA graph when the device supports it.
    """

    def __init__(
        self,
        model: newton.Model,
        solver: newton.solvers.SolverBase,
        pipeline: newton.CollisionPipeline | None,
        episodes: list[dict] | dict,
        *,
        dt: float,
        command_delay: float | list[float] = 0.0,
        contacts: newton.Contacts | None = None,
        record_dt: float = 1.0 / 120.0,
        use_graph: bool = True,
    ):
        worlds = model.world_count
        self.episodes = [episodes] * worlds if isinstance(episodes, dict) else list(episodes)
        if len(self.episodes) != worlds:
            raise ValueError(f"{len(self.episodes)} episodes for {worlds} worlds")
        delays = [float(command_delay)] * worlds if np.isscalar(command_delay) else [float(d) for d in command_delay]
        if len(delays) != worlds:
            raise ValueError(f"{len(delays)} command delays for {worlds} worlds")
        self.model, self.solver, self.collision_pipeline = model, solver, pipeline
        self.contacts = contacts if contacts is not None else (pipeline.contacts() if pipeline is not None else None)
        self.dt = float(dt)
        self.command_delay = delays
        self.use_graph = use_graph
        self.station = _Station(model)
        self.t0 = [float(e["left_t"][0]) for e in self.episodes]
        self.durations = [float(e["left_t"][-1]) - t0 for e, t0 in zip(self.episodes, self.t0, strict=True)]
        self.record_every = max(1, round(record_dt / self.dt))
        # Whole recording periods, so the history covers the last state sample of the longest episode.
        steps = int(math.ceil(max(self.durations) / self.dt - 1e-6))
        self.total_steps = -(-steps // self.record_every) * self.record_every
        device = model.device

        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        coord_layout = self.control.joint_target_q.shape[0] == model.joint_coord_count
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        q0 = model.joint_q.numpy().copy()
        qd0 = np.zeros(model.joint_dof_count, dtype=np.float32)
        steps = np.arange(self.total_steps) * self.dt
        columns, target_index, driven_dofs = [], [], []
        for w, episode in enumerate(self.episodes):
            schedule = command_schedule(episode, self.t0[w] + steps, delays[w])
            for side in SIDES:
                q_meas, grip_meas = measured(episode, side)
                qd_meas = episode.get(f"{side}_qd")
                for j, joint in enumerate(self.station.arm_joints[w][side]):
                    q0[q_start[joint]] = q_meas[0, j]
                    if qd_meas is not None:
                        qd0[qd_start[joint]] = qd_meas[0, j]
                    columns.append(schedule[side]["q"][:, j])
                    target_index.append(q_start[joint] if coord_layout else qd_start[joint])
                    driven_dofs.append(qd_start[joint])
                travel = float(np.clip(grip_meas[0], 0.0, 1.0)) * GRIPPER_TRAVEL
                for sign, joint in zip((1.0, -1.0), self.station.finger_joints[w][side], strict=True):
                    q0[q_start[joint]] = sign * travel
                    # The MJCF equality mirrors the right finger; its target is kept consistent anyway.
                    columns.append(sign * schedule[side]["finger"])
                    target_index.append(q_start[joint] if coord_layout else qd_start[joint])
                    driven_dofs.append(qd_start[joint])
        self._q0, self._qd0 = q0.astype(np.float32), qd0
        self._driven_dofs = np.asarray(driven_dofs, dtype=np.int64)
        self._target_index_host = np.asarray(target_index, dtype=np.int64)
        schedule = np.stack(columns, axis=-1).astype(np.float32)
        self._first_targets = schedule[0]
        self.schedule = wp.array(schedule, dtype=wp.float32, device=device)
        self.target_index = wp.array(self._target_index_host.astype(np.int32), dtype=wp.int32, device=device)
        rows = self.total_steps // self.record_every + 1
        self.history_body_q = wp.zeros((rows, model.body_count), dtype=wp.transformf, device=device)
        self.history_joint_q = wp.zeros((rows, model.joint_coord_count), dtype=wp.float32, device=device)
        self.step_index = wp.zeros(1, dtype=wp.int32, device=device)
        self._record_threads = max(model.body_count, model.joint_coord_count)
        self.graph = None
        self.reset()
        if use_graph and device.is_cuda:
            # Load kernels and let the solver/pipeline allocate outside the capture, then start over.
            self._step_once()
            self.reset()
            self.capture()

    def checkpoint_arrays(self) -> dict[str, wp.array]:
        """Replay arrays that advance with the simulation (cursor and history), by suggested attribute name."""
        return {
            "replay_step": self.step_index,
            "replay_history_body_q": self.history_body_q,
            "replay_history_joint_q": self.history_joint_q,
        }

    def reset(self) -> None:
        """Rewind to the episode starts: initial state, first targets, cursor 0, solver caches cleared."""
        self.state_0.joint_q.assign(self._q0)
        self.state_0.joint_qd.assign(self._qd0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        target = self.control.joint_target_q.numpy()
        target[self._target_index_host] = self._first_targets
        self.control.joint_target_q.assign(target)
        target_qd = self.control.joint_target_qd.numpy()
        target_qd[self._driven_dofs] = 0.0
        self.control.joint_target_qd.assign(target_qd)
        self.step_index.zero_()
        wp.copy(self.history_body_q[0], self.state_0.body_q)
        wp.copy(self.history_joint_q[0], self.state_0.joint_q)
        if hasattr(self.solver, "reset"):
            self.solver.reset(self.state_0, flags=newton.StateFlags.NONE)
        if self.collision_pipeline is not None and hasattr(self.collision_pipeline, "reset_contact_matching"):
            self.collision_pipeline.reset_contact_matching()
        if self.contacts is not None:
            self.contacts.clear()

    def _step_once(self) -> None:
        device = self.model.device
        wp.launch(
            _apply_targets,
            dim=self.schedule.shape[1],
            inputs=[self.schedule, self.step_index, self.target_index],
            outputs=[self.control.joint_target_q],
            device=device,
        )
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
        self.state_0.assign(self.state_1)
        wp.launch(
            _record,
            dim=self._record_threads,
            inputs=[self.state_0.body_q, self.state_0.joint_q, self.step_index, self.record_every],
            outputs=[self.history_body_q, self.history_joint_q],
            device=device,
        )
        # The cursor advances in its own launch: incrementing it inside _record would race with the
        # threads of that launch that have not read it yet.
        wp.launch(_advance, dim=1, inputs=[self.step_index], device=device)

    def capture(self) -> bool:
        """(Re)capture one replay step as a CUDA graph, e.g. after swapping the solver.

        Returns:
            Whether steps now replay a graph (``False`` means the host loop is used).
        """
        self.graph = None
        if not (self.use_graph and self.model.device.is_cuda):
            return False
        try:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._step_once()
            self.graph = capture.graph
        except Exception as error:
            warnings.warn(f"CUDA graph capture failed, stepping without a graph: {error}", stacklevel=2)
            self.graph = None
        return self.graph is not None

    def step(self, count: int = 1) -> None:
        """Advance ``count`` physics steps (targets held at the last command once an episode has ended)."""
        for _ in range(int(count)):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._step_once()

    @property
    def steps_done(self) -> int:
        """Physics steps taken since :meth:`reset` (reads the device cursor)."""
        return int(self.step_index.numpy()[0])

    @property
    def sim_time(self) -> float:
        """Simulated time since the episode starts [s]."""
        return self.steps_done * self.dt

    @property
    def done(self) -> bool:
        """Whether the longest episode has been replayed completely."""
        return self.steps_done >= self.total_steps

    def run(self, seconds: float | None = None) -> float:
        """Step until ``seconds`` after the episode starts (default: the end of the longest episode).

        Returns:
            Wall-clock time of the stepping [s].
        """
        target = self.total_steps if seconds is None else min(self.total_steps, round(seconds / self.dt))
        started = time.perf_counter()
        self.step(max(0, target - self.steps_done))
        wp.synchronize_device(self.model.device)
        return time.perf_counter() - started

    def recordings(self, worlds: list[int] | None = None) -> list[dict[str, np.ndarray]]:
        """Recordings of the given worlds (default: all) at their episodes' state times, up to the current step.

        Each recording samples the simulation at the episode's arm-state times ``left_t`` (top frame i shows
        row ``top_row[i]``, see :func:`frame_rows`), interpolating the history (linear positions and joints,
        normalized-lerp rotations). Keys (see FORMAT.md): ``t``, ``top_row``, ``<fruit>_pos``,
        ``<fruit>_quat``, ``<side>_pad``, ``<side>_hand_pos``, ``<side>_hand_quat``, ``<side>_finger``,
        ``<side>_q``, ``<side>_q_real``, ``<side>_grip_real``, ``body_q``, ``body_index``, ``world``,
        ``complete``.
        """
        worlds = list(range(self.model.world_count)) if worlds is None else list(worlds)
        valid = min(self.steps_done // self.record_every + 1, self.history_body_q.shape[0])
        history_body = self.history_body_q.numpy()[:valid].astype(np.float64)
        history_coords = self.history_joint_q.numpy()[:valid].astype(np.float64)
        t_history = np.arange(valid) * self.record_every * self.dt
        q_start = self.model.joint_q_start.numpy()
        out = []
        for w in worlds:
            episode, station = self.episodes[w], self.station
            t = np.asarray(episode["left_t"], dtype=np.float64)
            rows = int(np.searchsorted(t - self.t0[w], t_history[-1] + 1e-9, side="right"))
            lower, upper, weight = _bracket(t_history, t[:rows] - self.t0[w])
            bodies = station.world_bodies[w]
            body_q = _interpolate_transforms(history_body[:, bodies], lower, upper, weight)
            column = {int(b): k for k, b in enumerate(bodies)}
            coords = history_coords[lower] * (1.0 - weight[:, None]) + history_coords[upper] * weight[:, None]
            rec = {
                "t": t[:rows],
                "top_row": frame_rows(episode),
                "body_q": body_q,
                "body_index": bodies,
                "world": np.int64(w),
                "complete": np.bool_(rows == len(t)),
            }
            for name in FRUITS:
                pose = body_q[:, column[station.fruit_bodies[w][name]]]
                rec[f"{name}_pos"], rec[f"{name}_quat"] = pose[:, :3], pose[:, 3:]
            for side in SIDES:
                points = []
                for body in station.pad_bodies[w][side]:
                    pose = body_q[:, column[body]]
                    points.append(pose[:, :3] + quat_to_matrix(pose[:, 3:]) @ np.asarray(PAD_POINT))
                rec[f"{side}_pad"] = 0.5 * (points[0] + points[1])
                hand = body_q[:, column[station.hand_body[w][side]]]
                rec[f"{side}_hand_pos"], rec[f"{side}_hand_quat"] = hand[:, :3], hand[:, 3:]
                rec[f"{side}_finger"] = coords[:, q_start[station.finger_joints[w][side][0]]]
                rec[f"{side}_q"] = coords[:, [q_start[j] for j in station.arm_joints[w][side]]]
                q_meas, grip_meas = measured(episode, side)
                rec[f"{side}_q_real"], rec[f"{side}_grip_real"] = q_meas[:rows], grip_meas[:rows]
            out.append(rec)
        return out

    def recording(self, world: int = 0) -> dict[str, np.ndarray]:
        """Recording of one world (see :meth:`recordings`)."""
        return self.recordings([world])[0]


def _bracket(t_history: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(t_history) == 1:
        zeros = np.zeros(len(t), dtype=np.int64)
        return zeros, zeros, np.zeros(len(t))
    upper = np.clip(np.searchsorted(t_history, t, side="left"), 1, len(t_history) - 1)
    lower = upper - 1
    weight = np.clip((t - t_history[lower]) / (t_history[upper] - t_history[lower]), 0.0, 1.0)
    return lower, upper, weight


def _interpolate_transforms(history: np.ndarray, lower, upper, weight) -> np.ndarray:
    a, b = history[lower], history[upper]
    w = weight[:, None, None]
    out = np.empty_like(a)
    out[..., :3] = a[..., :3] * (1.0 - w) + b[..., :3] * w
    qb = np.where(np.sum(a[..., 3:] * b[..., 3:], axis=-1, keepdims=True) < 0.0, -b[..., 3:], b[..., 3:])
    q = a[..., 3:] * (1.0 - w) + qb * w
    out[..., 3:] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return out


def recording_state(model: newton.Model, recording: dict, row: int, state: newton.State | None = None) -> newton.State:
    """State with the recorded world's bodies posed as in recording row ``row`` (for rendering).

    Args:
        model: The replayed model.
        recording: One world's recording.
        row: Recording row (state sample); frame i of the top camera shows ``recording["top_row"][i]``.
        state: State to update; a new one if ``None``.
    """
    state = model.state() if state is None else state
    body_q = state.body_q.numpy()
    body_q[recording["body_index"]] = recording["body_q"][row]
    state.body_q.assign(body_q)
    return state


# ----------------------------------------------------------------------------- scoring


def _gt_rows(values: np.ndarray | None, rows: np.ndarray, count: int) -> np.ndarray:
    """Per-frame ground truth [frames, k] placed on recording rows [count, k] (NaN where no frame maps)."""
    if values is None:
        return np.full((count, 3), np.nan)
    values = np.asarray(values, dtype=np.float64)
    out = np.full((count, *values.shape[1:]), np.nan)
    for frame, row in enumerate(rows[: len(values)]):
        if row < count:
            out[row] = values[frame]
    return out


def _first(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def _score_fruit(rec: dict, scene: dict, gt: dict, name: str, camera: dict | None) -> dict:
    fruit = scene["fruits"][name]
    side, events, radius = fruit["arm"], fruit["events"], float(fruit["centre_height_m"])
    t, top_row = rec["t"], np.asarray(rec["top_row"])
    count = len(t)
    position = np.asarray(rec[f"{name}_pos"], dtype=np.float64)
    start = position[0]
    hand_rotation = quat_to_matrix(rec[f"{side}_hand_quat"])
    in_hand = np.einsum("nji,nj->ni", hand_rotation, position - rec[f"{side}_hand_pos"])
    row = {key: int(top_row[min(events[key], len(top_row) - 1)]) for key in _EVENT_KEYS}
    m: dict = {"arm": side}

    close = min(row["cmd_close"], count - 1)
    m["moved_before_grasp_m"] = float(np.linalg.norm(position[: close + 1, :2] - start[:2], axis=1).max())
    m["max_rise_m"] = float(position[:, 2].max() - start[2])

    first, last = row["liftoff"] + CARRY_START_OFFSET, row["release"] - CARRY_END_OFFSET
    carry = np.arange(first, last + 1) if last < count else np.zeros(0, dtype=np.int64)
    distance = np.linalg.norm(position - rec[f"{side}_pad"], axis=1)
    m["held_fraction"] = float(np.mean(distance[carry] <= radius + HOLD_MARGIN)) if carry.size else 0.0
    m["held"] = bool(carry.size and m["held_fraction"] >= DEFAULT_THRESHOLDS["held_fraction"])
    m["slip_m"] = float(np.linalg.norm(in_hand[carry] - in_hand[first], axis=1).max()) if carry.size else None

    track = _gt_rows(gt.get(f"{name}_pos"), top_row, count)
    rest_real = gt.get(f"{name}_rest_xyz")
    rest_z = float(rest_real[2]) if rest_real is not None else float(track[0, 2])
    known = carry[np.isfinite(track[carry, 2])] if carry.size else carry
    if known.size:
        lifted = position[known, 2] - start[2] >= 0.5 * (track[known, 2] - rest_z)
        m["carry_track_err_m"] = float(np.median(np.linalg.norm(position[known, :2] - track[known, :2], axis=1)))
        m["carry_track_err_3d_m"] = float(np.median(np.linalg.norm(position[known] - track[known], axis=1)))
    else:
        # Without a real carry height, require a clear lift (2 cm).
        lifted = position[carry, 2] - start[2] >= 0.02
        m["carry_track_err_m"] = m["carry_track_err_3d_m"] = None
    m["lifted_fraction"] = float(np.mean(lifted)) if carry.size else 0.0

    lift = _first(position[close:, 2] - start[2] > LIFT_THRESHOLD)
    m["liftoff_err_s"] = None if lift is None else float(t[close + lift] - t[min(row["liftoff"], count - 1)])
    m["liftoff_err_rows"] = None if lift is None else int(close + lift - row["liftoff"])

    m["release_err_s"] = None
    if carry.size and m["held"]:
        moved = _first(np.linalg.norm(in_hand[first:] - in_hand[first], axis=1) > RELEASE_THRESHOLD)
        if moved is not None and row["release"] < count:
            m["release_err_s"] = float(t[first + moved] - t[row["release"]])

    hold = np.arange(row["liftoff"], min(row["cmd_open"], count))
    if hold.size:
        gap_sim = 2.0 * rec[f"{side}_finger"][hold]
        gap_real = 2.0 * np.clip(rec[f"{side}_grip_real"][hold], 0.0, 1.0) * GRIPPER_TRAVEL
        m["grip_gap_err_mm"] = float(1000.0 * np.median(gap_sim - gap_real))
    else:
        m["grip_gap_err_mm"] = None

    final = position[-1]
    tray = scene["tray"]
    window_start = max(0, int(np.searchsorted(t, t[-1] - REST_WINDOW, side="right")) - 1)
    elapsed = t[-1] - t[window_start]
    m["final_xyz"] = [float(v) for v in final]
    m["tray_distance_m"] = sector_distance(final[:2], tray)
    m["final_height_err_m"] = float(final[2] - (scene["table_z"] + tray["floor_height_m"] + radius))
    m["final_speed_mps"] = float(np.linalg.norm(final - position[window_start]) / elapsed) if elapsed > 0 else None
    m["placed"] = bool(
        rec.get("complete", True)
        and m["tray_distance_m"] <= PLACE_MARGIN
        and abs(m["final_height_err_m"]) <= REST_HEIGHT_TOLERANCE
        and m["final_speed_mps"] is not None
        and m["final_speed_mps"] < REST_SPEED
    )

    final_real = gt.get(f"{name}_final_xyz")
    if final_real is None and gt.get(f"{name}_pos") is not None:
        finite = np.flatnonzero(np.isfinite(gt[f"{name}_pos"][:, 0]))
        final_real = gt[f"{name}_pos"][finite[-1]] if len(finite) else None
    m["final_xy_err_m"] = None if final_real is None else float(np.linalg.norm(final[:2] - np.asarray(final_real)[:2]))
    start_real = rest_real if rest_real is not None else (track[0] if np.isfinite(track[0, 0]) else None)
    m["initial_xy_err_m"] = (
        None if start_real is None else float(np.linalg.norm(start[:2] - np.asarray(start_real)[:2]))
    )

    m["image_err_px_post"] = None
    if camera is not None and f"{name}_uv" in gt and f"{name}_fix" in gt:
        frames = [
            f
            for f in range(events["release"], min(len(top_row), len(gt[f"{name}_uv"])))
            if gt[f"{name}_fix"][f] and top_row[f] < count and np.isfinite(gt[f"{name}_uv"][f, 0])
        ]
        if frames:
            pixels = project_points(camera["top"], position[top_row[frames]])
            m["image_err_px_post"] = float(np.median(np.linalg.norm(pixels - gt[f"{name}_uv"][frames], axis=1)))
    return m


def score(recording: dict, scene: dict, gt: dict | None = None, camera: dict | None = None) -> dict:
    """Replay metrics of one world's recording against the real episode.

    All times are on the arm-state clock, sampled at the episode's state samples; scene event frames are
    converted with ``recording["top_row"]`` (top frame i shows state sample i - 1 in the main episode).
    Per fruit (``fruits[name]``):

    - ``moved_before_grasp_m``: largest xy displacement from the start up to the grasping arm's close command.
    - ``max_rise_m``: highest centre height above the start over the episode (negative controls).
    - ``held_fraction``/``held``: fraction of carry samples (lift-off + 3 to release - 2) with the fruit centre
      within its radius + 15 mm of the grasping arm's pad midpoint; held if at least 0.9.
    - ``lifted_fraction``: fraction of carry samples where the fruit is at least half as high above its start
      as the real one (ground-truth track; 2 cm without one).
    - ``slip_m``: largest drift of the fruit in the hand frame over the carry.
    - ``liftoff_err_s``/``liftoff_err_rows``: sim minus real lift-off (centre 1 cm above its start).
    - ``release_err_s``: sim release (fruit 1 cm from its in-hand position at the carry start) minus real release.
    - ``carry_track_err_m``: median xy distance to the ground-truth track over carry samples (FK-attached in
      the main episode); ``carry_track_err_3d_m`` includes height (report only).
    - ``grip_gap_err_mm``: median finger-gap error (sim minus measured) from lift-off to the open command.
    - ``placed``: complete episode, final centre within the tray sector + 10 mm, centre height within 15 mm of
      tray floor + radius, and slower than 2 cm/s over the last 0.3 s (``tray_distance_m``,
      ``final_height_err_m``, ``final_speed_mps``).
    - ``final_xy_err_m``, ``initial_xy_err_m``, ``image_err_px_post`` (report only; the last needs ``camera``
      and ground-truth image fixes).

    Args:
        recording: One world's recording (:meth:`Replay.recordings`).
        scene: The world's scene.
        gt: Ground truth (:func:`load_gt`); ``None`` scores without it.
        camera: Camera calibration (:func:`load_camera`) for the image metric.

    Returns:
        ``{"complete", "arm_rmse_rad": {side: rad}, "fruits": {name: metrics}, "all_held", "all_placed"}``.
    """
    gt = {} if gt is None else gt
    result = {"complete": bool(recording.get("complete", True)), "arm_rmse_rad": {}, "fruits": {}}
    for side in SIDES:
        error = np.asarray(recording[f"{side}_q"]) - np.asarray(recording[f"{side}_q_real"])
        result["arm_rmse_rad"][side] = float(np.sqrt(np.mean(error**2)))
    for name in FRUITS:
        result["fruits"][name] = _score_fruit(recording, scene, gt, name, camera)
    result["all_held"] = all(m["held"] for m in result["fruits"].values())
    result["all_placed"] = all(m["placed"] for m in result["fruits"].values())
    return result


def check(metrics: dict, thresholds: dict | None = None) -> dict:
    """Pass/fail of one scored rollout against per-rollout limits (default :data:`DEFAULT_THRESHOLDS`).

    Metrics that need ground truth the episode lacks (``None`` track or final errors) are skipped; timing
    metrics that are ``None`` because the event never happened fail.

    Returns:
        ``{"pass": bool, "failed": [names]}``.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    failed = [] if metrics["complete"] else ["complete"]
    failed += [f"arm_rmse_rad.{s}" for s, v in metrics["arm_rmse_rad"].items() if not v <= limits["arm_rmse_rad"]]
    low, high = limits["release_err_s"]
    for name, m in metrics["fruits"].items():
        tests = {
            "held": m["held_fraction"] >= limits["held_fraction"],
            "placed": m["placed"],
            "lifted_fraction": m["lifted_fraction"] >= limits["lifted_fraction"],
            "moved_before_grasp_m": m["moved_before_grasp_m"] <= limits["moved_before_grasp_m"],
            "liftoff_err_s": m["liftoff_err_s"] is not None and abs(m["liftoff_err_s"]) <= limits["liftoff_err_s"],
            "release_err_s": m["release_err_s"] is not None and low <= m["release_err_s"] <= high,
            "grip_gap_err_mm": m["grip_gap_err_mm"] is not None
            and abs(m["grip_gap_err_mm"]) <= limits["grip_gap_err_mm"],
        }
        if m["carry_track_err_m"] is not None:
            tests["carry_track_err_m"] = m["carry_track_err_m"] <= limits["carry_track_err_m"]
        if m["final_xy_err_m"] is not None:
            tests["final_xy_err_m"] = m["final_xy_err_m"] <= limits["final_xy_err_m"]
        failed += [f"{name}.{key}" for key, ok in tests.items() if not ok]
    return {"pass": not failed, "failed": failed}


def summarize(results: list[dict]) -> dict:
    """Ensemble summary of several scored rollouts of one episode: per-fruit held/placed counts and medians.

    Returns:
        ``{"copies", "arm_rmse_rad": {side: median}, "fruits": {name: {"held", "placed", <metric>: median}}}``.
    """
    out = {"copies": len(results), "arm_rmse_rad": {}, "fruits": {}}
    for side in SIDES:
        out["arm_rmse_rad"][side] = float(np.median([r["arm_rmse_rad"][side] for r in results]))
    for name in FRUITS:
        rows = [r["fruits"][name] for r in results]
        entry = {"held": sum(m["held"] for m in rows), "placed": sum(m["placed"] for m in rows)}
        for key, value in rows[0].items():
            if isinstance(value, float) or value is None:
                values = [m[key] for m in rows if m[key] is not None]
                entry[key] = float(np.median(values)) if values else None
        out["fruits"][name] = entry
    return out


# ----------------------------------------------------------------------------- cameras

DEFAULT_LOOK = {
    "light_direction": (-0.57735, 0.57735, -0.57735),
    "light_color": (1.0, 1.0, 1.0),
    "shadows": True,
    "ambient_sky": (0.2, 0.2, 0.225),
    "ambient_ground": (0.05, 0.05, 0.06),
    "exposure": 1.0,
}


def camera_rays(intrinsics: dict, supersample: int = 1) -> np.ndarray:
    """Camera-frame ray directions (x right, y up, looking along -z), shape [H*s, W*s, 3].

    Uses RealSense's inverse Brown-Conrady model, which maps distorted pixels directly to rays.
    """
    width, height = intrinsics["width"], intrinsics["height"]
    fx, _, cx, _, fy, cy = intrinsics["K"][:6]
    k1, k2, p1, p2, k3 = intrinsics["D"]
    u = (np.arange(width * supersample) + 0.5) / supersample
    v = (np.arange(height * supersample) + 0.5) / supersample
    x, y = np.meshgrid((u - cx) / fx, (v - cy) / fy)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    ux = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    uy = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    rays = np.stack([ux, -uy, -np.ones_like(ux)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def project_points(intrinsics: dict, points: np.ndarray, pose: tuple | None = None) -> np.ndarray:
    """Pixel coordinates of world points seen by a calibrated camera (inverse of :func:`camera_rays`).

    Args:
        intrinsics: Camera entry of camera.json (``K``, ``D``, and ``position``/``rotation_xyzw`` unless
            ``pose`` is given).
        points: World points [m], shape [..., 3].
        pose: Camera ``(position, rotation_xyzw)``; the camera looks along its -Z with +Y up.

    Returns:
        Pixel coordinates (u right, v down), shape [..., 2].
    """
    position, rotation = pose if pose is not None else (intrinsics["position"], intrinsics["rotation_xyzw"])
    local = (np.asarray(points, dtype=np.float64) - np.asarray(position)) @ quat_to_matrix(np.asarray(rotation))
    xu, yu = local[..., 0] / -local[..., 2], -local[..., 1] / -local[..., 2]
    k1, k2, p1, p2, k3 = intrinsics["D"]
    x, y = xu.copy(), yu.copy()
    for _ in range(20):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x = x - (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xu)
        y = y - (y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y) - yu)
    fx, _, cx, _, fy, cy = intrinsics["K"][:6]
    return np.stack([x * fx + cx, y * fy + cy], axis=-1)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


class StationRenderer:
    """Renders worlds of a station model through the real top and wrist cameras (SensorTiledCamera).

    The top camera uses the fitted pose in camera.json; a wrist camera follows its MJCF camera body
    (``left_camera_frame``/``right_camera_frame``). Images are 640x480 sRGB through each camera's
    calibrated intrinsics and distortion, supersampled in linear light.

    Args:
        model: Station model (any number of worlds).
        camera: Camera calibration (path or dict, default ``camera.json`` next to this module).
        supersample: Rays per pixel along each image axis.
        look: Lighting overrides (see :data:`DEFAULT_LOOK`).
    """

    def __init__(
        self,
        model: newton.Model,
        camera: str | Path | dict | None = None,
        *,
        supersample: int = 1,
        look: dict | None = None,
    ):
        from newton.sensors import SensorTiledCamera  # noqa: PLC0415

        self.model = model
        self.camera = load_camera(camera)
        self.supersample = int(supersample)
        look = {**DEFAULT_LOOK, **(look or {})}
        self.exposure = float(look["exposure"])
        self.sensor = SensorTiledCamera(model)
        direction = np.asarray(look["light_direction"], dtype=np.float32)
        self.sensor.default_render_config.enable_shadows = bool(look["shadows"])
        self.sensor.utils.create_default_light(
            enable_shadows=bool(look["shadows"]),
            direction=wp.vec3f(*(direction / np.linalg.norm(direction))),
            color=wp.vec3f(*look["light_color"]),
        )
        self.sensor.utils.set_ambient_light(wp.vec3f(*look["ambient_sky"]), wp.vec3f(*look["ambient_ground"]))
        self.station = _Station(model)
        self._buffers: dict[str, tuple] = {}
        self._shape_body = model.shape_body.numpy()

    def _outputs(self, name: str):
        if name not in self._buffers:
            intrinsics = self.camera[name]
            rays = camera_rays(intrinsics, self.supersample).astype(np.float32)
            packed = np.zeros((1, *rays.shape[:2], 2, 3), dtype=np.float32)
            packed[0, :, :, 1] = rays
            width, height = intrinsics["width"] * self.supersample, intrinsics["height"] * self.supersample
            self._buffers[name] = (
                wp.array(packed, dtype=wp.vec3f, device=self.model.device),
                self.sensor.utils.create_hdr_color_image_output(width, height, camera_count=1, world_count=1),
                self.sensor.utils.create_shape_index_image_output(width, height, camera_count=1, world_count=1),
            )
        return self._buffers[name]

    def render(self, state: newton.State, camera_name: str, pose: tuple, world: int = 0, *, masks: bool = False):
        """Render one world through a camera.json camera at ``pose``.

        Args:
            state: State to render (shape BVHs are refit to it).
            camera_name: ``"top"``, ``"left"``, or ``"right"`` (intrinsics and distortion).
            pose: Camera ``(position [m], rotation_xyzw)``; the camera looks along its -Z with +Y up.
            world: World to render.
            masks: Also return each fruit's visible pixels.

        Returns:
            sRGB image [H, W, 3] uint8, plus ``{fruit: bool mask [H, W]}`` with ``masks``.
        """
        rays, hdr, index = self._outputs(camera_name)
        transforms = wp.array(
            [
                [
                    wp.transformf(
                        wp.vec3f(*[float(v) for v in pose[0]]), wp.normalize(wp.quatf(*[float(v) for v in pose[1]]))
                    )
                ]
            ],
            dtype=wp.transformf,
            device=self.model.device,
        )
        self.model.bvh_refit_shapes(state)
        self.sensor.update(
            state,
            transforms,
            rays,
            hdr_color_image=hdr,
            shape_index_image=index if masks else None,
            world_ids=wp.array([world], dtype=wp.int32, device=self.model.device),
        )
        s = self.supersample
        intrinsics = self.camera[camera_name]
        height, width = intrinsics["height"], intrinsics["width"]
        linear = hdr.numpy()[0, 0].reshape(height, s, width, s, 3).mean(axis=(1, 3))
        image = np.round(_linear_to_srgb(linear * self.exposure) * 255.0).astype(np.uint8)
        if not masks:
            return image
        shapes = index.numpy()[0, 0][s // 2 :: s, s // 2 :: s].astype(np.int64)
        hit = shapes < len(self._shape_body)
        body = np.where(hit, self._shape_body[np.where(hit, shapes, 0)], -1)
        return image, {name: body == self.station.fruit_bodies[world][name] for name in FRUITS}

    def top_pose(self) -> tuple:
        """Fitted top-camera ``(position, rotation_xyzw)`` from camera.json."""
        top = self.camera["top"]
        return top["position"], top["rotation_xyzw"]

    def wrist_pose(self, state: newton.State, side: str, world: int = 0) -> tuple:
        """Wrist-camera ``(position, rotation_xyzw)`` of an arm in ``state``."""
        body = self.station.camera_body[world][side]
        if body is None:
            raise ValueError(f"world {world} has no body {CAMERA_BODY[side]!r}")
        pose = state.body_q.numpy()[body].astype(np.float64)
        rotation = _quat_multiply(pose[3:], np.asarray(CAMERA_IN_BODY_XYZW, dtype=np.float64))
        return pose[:3], rotation

    def render_top(self, state: newton.State, world: int = 0, *, masks: bool = False):
        """Top-camera image of one world (see :meth:`render`)."""
        return self.render(state, "top", self.top_pose(), world, masks=masks)

    def render_wrist(self, state: newton.State, side: str, world: int = 0, *, masks: bool = False):
        """Wrist-camera image of one arm in one world (see :meth:`render`)."""
        return self.render(state, side, self.wrist_pose(state, side, world), world, masks=masks)


_RENDERERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _renderer(model: newton.Model, camera) -> StationRenderer:
    renderer = _RENDERERS.get(model)
    if renderer is None or (camera is not None and load_camera(camera) != renderer.camera):
        renderer = StationRenderer(model, camera)
        _RENDERERS[model] = renderer
    return renderer


def render_top(model: newton.Model, state: newton.State, world: int = 0, *, camera=None, masks: bool = False):
    """Top-camera image [480, 640, 3] uint8 of one world (cached :class:`StationRenderer` per model)."""
    return _renderer(model, camera).render_top(state, world, masks=masks)


def render_wrist(
    model: newton.Model, state: newton.State, side: str, world: int = 0, *, camera=None, masks: bool = False
):
    """Wrist-camera image [480, 640, 3] uint8 of one arm in one world (cached :class:`StationRenderer` per model)."""
    return _renderer(model, camera).render_wrist(state, side, world, masks=masks)


# ----------------------------------------------------------------------------- contacts

_REPORTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _select_shapes(model: newton.Model, selector, world: int | None) -> np.ndarray:
    """Shape mask for a label substring (shape or body label), a shape index, or a list of these."""
    shape_labels = [_leaf(label) for label in model.shape_label]
    body_labels = [_leaf(label) for label in model.body_label]
    shape_body = model.shape_body.numpy()
    mask = np.zeros(model.shape_count, dtype=bool)
    for item in selector if isinstance(selector, list | tuple) else [selector]:
        if isinstance(item, int | np.integer):
            mask[int(item)] = True
        else:
            for i in range(model.shape_count):
                body = int(shape_body[i])
                if item in shape_labels[i] or (body >= 0 and item in body_labels[body]):
                    mask[i] = True
    if world is not None:
        mask &= np.isin(model.shape_world.numpy(), (world, -1))
    if not mask.any():
        raise ValueError(f"selector {selector!r} matches no shape")
    return mask


def contact_summary(
    model: newton.Model,
    state: newton.State,
    contacts: newton.Contacts | None,
    a,
    b=None,
    *,
    solver: newton.solvers.SolverBase | None = None,
    world: int | None = None,
) -> dict:
    """Contacts between shape sets ``a`` and ``b`` (or ``a`` and everything else) after the last step.

    With :class:`~newton.solvers.SolverMuJoCo` the contact set and forces are the solver's own
    (``update_contacts``), whichever pipeline produced them; otherwise the ``contacts`` buffer is read.

    Args:
        model: The model.
        state: Current state.
        contacts: Contacts the last step used (``None`` with solver-internal contacts).
        a: Shapes: label substring of a shape or its body (e.g. ``"left_lf_down"``, ``"pear"``), shape
            index, or a list of these.
        b: Second set; ``None`` means every other shape.
        solver: Solver that took the last step (for forces).
        world: Restrict both sets to one world (plus global shapes).

    Returns:
        ``count``, ``touching`` (distance <= 0), ``normal_force`` [N] and ``tangential_force`` [N] on ``a``
        (``nan`` without solver forces), ``penetration`` [m], ``slip_max`` [m/s] (relative tangential speed
        at touching contacts), and ``by_body`` (contact count and normal force per body pair).
    """
    in_a = _select_shapes(model, a, world)
    in_b = _select_shapes(model, b, world) if b is not None else ~in_a
    data = getattr(solver, "mjw_data", None)
    if data is not None and hasattr(solver, "update_contacts"):
        report = _REPORTS.get(solver)
        if report is None or report.rigid_contact_max < data.naconmax:
            report = newton.Contacts(data.naconmax, 0, requested_attributes={"force"}, device=model.device)
            _REPORTS[solver] = report
        solver.update_contacts(report, state)
        count = min(int(data.nacon.numpy()[0]), report.rigid_contact_max)
        shape0, shape1 = report.rigid_contact_shape0.numpy()[:count], report.rigid_contact_shape1.numpy()[:count]
        normal = report.rigid_contact_normal.numpy()[:count]
        point = data.contact.pos.numpy()[:count]
        distance = data.contact.dist.numpy()[:count]
        force = report.force.numpy()[:count, :3]
    elif contacts is not None:
        count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
        out_distance = wp.empty(contacts.rigid_contact_max, dtype=wp.float32, device=model.device)
        point0 = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
        point1 = wp.empty_like(point0)
        newton.eval_rigid_contact_kinematics(
            model, state, contacts, out_distance=out_distance, out_point0_world=point0, out_point1_world=point1
        )
        shape0, shape1 = contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
        normal = contacts.rigid_contact_normal.numpy()[:count]
        point = 0.5 * (point0.numpy()[:count] + point1.numpy()[:count])
        distance = out_distance.numpy()[:count]
        force = None
        if solver is not None and hasattr(solver, "update_contacts"):
            try:
                solver.update_contacts(contacts, state)
                if contacts.force is not None:
                    force = contacts.force.numpy()[:count, :3]
            except NotImplementedError:
                force = None
    else:
        raise ValueError("no contacts: pass the contacts buffer or a solver with its own contact set")
    shape0, shape1 = shape0.astype(np.int64), shape1.astype(np.int64)
    valid = (shape0 >= 0) & (shape1 >= 0) & (shape0 < model.shape_count) & (shape1 < model.shape_count)
    s0, s1 = np.where(valid, shape0, 0), np.where(valid, shape1, 0)
    forward = valid & in_a[s0] & in_b[s1]
    backward = valid & in_a[s1] & in_b[s0] & ~forward
    selected = np.flatnonzero(forward | backward)
    sign = np.where(backward[selected], -1.0, 1.0)[:, None]
    normal_ab = normal[selected] * sign  # from a toward b
    touching = distance[selected] <= 0.0
    result = {"count": int(len(selected)), "touching": int(touching.sum())}
    if force is not None and len(selected):
        on_a = force[selected] * sign
        along = np.einsum("ij,ij->i", on_a, normal_ab)
        result["normal_force"] = float(np.clip(-along, 0.0, None).sum())
        result["tangential_force"] = float(np.linalg.norm(on_a - along[:, None] * normal_ab, axis=1).sum())
    else:
        result["normal_force"] = result["tangential_force"] = 0.0 if force is not None else float("nan")
    result["penetration"] = float(max(0.0, -distance[selected].min())) if len(selected) else 0.0
    shape_body = model.shape_body.numpy()
    side_a = np.where(backward[selected], shape1[selected], shape0[selected])
    side_b = np.where(backward[selected], shape0[selected], shape1[selected])
    if len(selected) and touching.any():
        velocity = _point_velocity(model, state, shape_body[side_b], point[selected]) - _point_velocity(
            model, state, shape_body[side_a], point[selected]
        )
        tangential = velocity - np.einsum("ij,ij->i", velocity, normal_ab)[:, None] * normal_ab
        result["slip_max"] = float(np.linalg.norm(tangential, axis=1)[touching].max())
    else:
        result["slip_max"] = 0.0
    body_labels = [_leaf(label) for label in model.body_label]
    pairs: dict[str, dict] = {}
    for k in range(len(selected)):
        names = [body_labels[b] if b >= 0 else "static" for b in (shape_body[side_a[k]], shape_body[side_b[k]])]
        entry = pairs.setdefault(
            " | ".join(names), {"count": 0, "normal_force": 0.0 if force is not None else float("nan")}
        )
        entry["count"] += 1
        if force is not None:
            entry["normal_force"] += float(max(0.0, -np.dot(force[selected[k]] * sign[k], normal_ab[k])))
    result["by_body"] = pairs
    return result


def _point_velocity(model: newton.Model, state: newton.State, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """World velocity [m/s] of each body's material point at ``points`` (zero for static shapes)."""
    out = np.zeros_like(points, dtype=np.float64)
    moving = bodies >= 0
    if not moving.any() or state.body_qd is None:
        return out
    pose = state.body_q.numpy()[bodies[moving]].astype(np.float64)
    twist = state.body_qd.numpy()[bodies[moving]].astype(np.float64)
    com = pose[:, :3] + np.einsum("nij,nj->ni", quat_to_matrix(pose[:, 3:]), model.body_com.numpy()[bodies[moving]])
    out[moving] = twist[:, :3] + np.cross(twist[:, 3:], points[moving] - com)
    return out
