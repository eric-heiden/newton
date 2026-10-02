# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fixed helpers for the ABC screwdriver-in-bin physical replay task.

The task replays a real ABC-130k episode ("put the screwdriver in the bin") in
Newton: the right YAM arm, driven open loop by the recorded teleoperation
commands, picks up a screwdriver lying on the table by its handle, carries it,
and puts it into a pink plastic bin, the way the real robot did. The starter
script and the verifier share this module (the verifier uses its own copy, so
edits here have no effect on verification). It provides:

- loading of episodes, scenes, ground truth, and camera calibration
  (:func:`load_episode`, :func:`load_scene`, :func:`load_gt`, :func:`load_camera`),
  and :func:`episode_from_lerobot` for 10 fps LeRobot copies,
- the command schedule (:func:`command_schedule`): zero-order hold of the
  logged commands, delayed by the controller latency,
- :class:`Replay`, which writes the per-world joint targets and steps a
  submitted model/solver/collision pipeline under a CUDA graph, recording the
  simulation at the episode's sample times,
- :func:`score`, the replay metrics the verifier thresholds, :func:`check`, and
  :func:`summarize`,
- :class:`StationRenderer` (:func:`render_top`, :func:`render_wrist`), which
  renders the station through the real cameras' calibrated intrinsics,
  and :func:`project_points`,
- :func:`contact_summary`, contact counts and forces between shape sets,
- :class:`StationFK`, forward kinematics of the measured joints (FK-consistent
  object starts and FK-attached carry tracks),
- the screwdriver and bin geometry used by the metrics (:func:`handle_radius`,
  :func:`bin_inner_polygon`, :func:`polygon_signed_distance`).

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
OBJECTS = ("screwdriver",)
BIN_BODY = "bin"  # label of a dynamic bin body (a static bin has no body)
SIDES = ("left", "right")
FRAME_RATE = 30.0  # nominal camera and arm-state sample rate [Hz]
GRIPPER_TRAVEL = 0.0475  # finger slide travel [m] at a gripper opening of 1
# Finger pad geometry in the pad body frame (bodies "<side>_lf_down" and "<side>_rf_down"): the pad
# box centre, and the grasp point 25 mm from it toward the fingertip along the pad's long axis (+z).
PAD_CENTRE = (0.0, -0.0024, 0.046)
PAD_POINT = (0.0, -0.0024, 0.071)
# MJCF cameras look along -Z of a frame rotated 180 degrees about x from their body; Newton cameras
# look along their own -Z with +Y up, so a wrist camera's pose is body pose * this rotation (camera.json
# "body_rotation_xyzw" overrides it, and an optional "body_offset_m" shifts the camera in the body frame).
CAMERA_IN_BODY_XYZW = (1.0, 0.0, 0.0, 0.0)
CAMERA_BODY = {"left": "left_camera_frame", "right": "right_camera_frame"}
HAND_BODY = "{side}_link_6"

# Definitions used by score().
HOLD_MARGIN = 0.015  # held: grasp point within the grasp radius + this of the grasping arm's pad midpoint [m]
HELD_FRACTION_MIN = 0.9  # held: at least this fraction of the carry rows
LIFT_THRESHOLD = 0.01  # lift-off: grasp point this far above its start [m]
RELEASE_THRESHOLD = 0.01  # release: grasp point this far from its in-hand position at the carry start [m]
PLACE_MARGIN = 0.005  # placed: tip and centre of mass within the bin's inner wall polygon dilated by this [m]
RIM_CLEARANCE = 0.02  # placed: centre of mass at least this far below the bin rim [m]
REST_SPEED = 0.02  # placed: mean speed of the centre of mass over the last REST_WINDOW below this [m/s]
REST_WINDOW = 0.3  # [s]
CARRY_START_OFFSET = 3  # carry window: state rows after lift-off ...
CARRY_END_OFFSET = 2  # ... to the scene's open_cmd_row (at most this many state rows before release)
OPEN_CMD_RISE = 0.02  # open_cmd_row: last row before the gripper command rises this far above its hold value
HELDOUT_ROT_MAX_DEG = 12.0  # unseen episodes: a copy counts if held, placed, and rotated at most this in hand

# Per-rollout limits used by check(): the single-rollout counterparts of the verification gates, which apply
# them to ensembles (counts of copies and medians over copies).
DEFAULT_THRESHOLDS = {
    "held_fraction": HELD_FRACTION_MIN,  # min, fraction of carry rows held
    "inhand_rot_deg": 6.0,  # max rotation in the hand frame over the carry [deg]
    "slip_m": 0.005,  # max drift of the grasp point in the hand frame over the carry [m]
    "grip_gap_err_mm": 5.0,  # max |median finger-gap error| while holding [mm]
    "liftoff_err_rows": 3,  # max |sim - real| lift-off [state rows]
    "carry_track_err_m": 0.025,  # max median xy distance to the FK-attached carry track [m]
    "final_tip_xy_err_m": 0.025,  # max distance of the final tip from the real one [m]
    "moved_before_grasp_m": 0.02,  # max xy displacement before the close command [m]
    "arm_rmse_rad": 0.0346,  # max whole-episode joint RMSE per arm [rad]
}

_EPISODE_KEYS = ("t", "q", "cmd_t", "cmd", "grip_t", "grip", "grip_cmd_t", "grip_cmd")
_EVENT_KEYS = ("cmd_close", "contact", "liftoff", "cmd_open", "release")
_OBJECT_KEYS = ("arm", "events", "open_cmd_row", "start", "grasp_height_m", "tip_local", "com_local", "geometry")
_BIN_KEYS = ("center_xy", "yaw_deg", "top_size_m", "bottom_size_m", "height_m", "wall_m", "floor_m")


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
        Arrays by key, e.g. ``right_q`` [n, 6] measured joints [rad] at the state times ``right_t`` [s].
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
    cameras: tuple[str, ...] = ("top", "left", "right"),
) -> dict[str, np.ndarray]:
    """Episode dict from a LeRobot copy of an ABC episode (10 fps), linearly interpolated onto a 30 Hz grid.

    Measured joints and gripper come from ``state``, commands from ``action``; both are interpolated onto
    the grid, which the replay then holds and delays like logged commands. LeRobot rows are copies of the
    MCAP samples nearest to each 10 fps tick.

    Args:
        timestamp: LeRobot sample times [s], shape [K].
        state: Measured values, shape [K, 14]: left joints 1-6 [rad], left gripper [0..1], right joints,
            right gripper.
        action: Commanded values in the same layout, shape [K, 14].
        velocity: Measured velocities in the same layout [rad/s] (``observation.velocity``); ``None`` for none.
        torque: Measured efforts in the same layout [N·m] (``observation.torque``; column 6 of each arm is the
            gripper motor); ``None`` for none.
        time_offset: Added to LeRobot times to reach the clock of the MCAP recordings (seconds from the first
            top-camera frame) [s]: the state stream's start minus the top camera's.
        rate: Grid rate [Hz].
        frame_lag: Camera latency [s]: a frame shows the arm state this long before its tick. LeRobot frame k
            is MCAP frame 3k + 3 (the frame nearest to the tick), which shows state sample 3k + 2 (time-base
            rule), so frames show the state 2/30 s *after* their tick (-2/30).
        cameras: Camera streams that get frame times ``t_<camera>`` and ``<camera>_state_index`` (the wrist
            streams are assumed to follow the top camera's rule).

    Returns:
        Episode dict in the format of :func:`load_episode`, with ``t_<camera>`` the LeRobot frame times on the
        state clock and ``<camera>_state_index`` the state row each 10 fps frame shows.
    """
    timestamp = np.asarray(timestamp, dtype=np.float64)
    state, action = np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)
    ts = timestamp + time_offset
    grid = ts[0] + np.arange(int(math.floor((ts[-1] - ts[0]) * rate + 1e-6)) + 1) / rate
    episode = {}
    for side, base in (("left", 0), ("right", 7)):
        for key in ("t", "cmd_t", "grip_t", "grip_cmd_t"):
            episode[f"{side}_{key}"] = grid.copy()
        episode[f"{side}_q"] = np.stack([np.interp(grid, ts, state[:, base + j]) for j in range(6)], axis=-1)
        episode[f"{side}_cmd"] = np.stack([np.interp(grid, ts, action[:, base + j]) for j in range(6)], axis=-1)
        episode[f"{side}_grip"] = np.interp(grid, ts, state[:, base + 6])
        episode[f"{side}_grip_cmd"] = np.interp(grid, ts, action[:, base + 6])
        for key, values in (("qd", velocity), ("tau", torque)):
            if values is not None:
                columns = np.asarray(values, dtype=np.float64)[:, base : base + 7]
                episode[f"{side}_{key}"] = np.stack([np.interp(grid, ts, columns[:, j]) for j in range(7)], axis=-1)
    rows = np.clip(np.round((ts - frame_lag - grid[0]) * rate), 0, len(grid) - 1).astype(np.int64)
    for camera in ("top", *[c for c in cameras if c != "top"]):
        episode[f"t_{camera}"] = ts.copy()
        episode[f"{camera}_state_index"] = rows.copy()
    return episode


def frame_rows(episode: dict, camera: str = "top") -> np.ndarray:
    """State sample (row of ``left_t``) shown by each frame of a camera stream.

    Uses ``<camera>_state_index`` when the episode has it; otherwise frame i shows sample i - 1 (the
    measured latency of the main episode's top camera, assumed for its wrist cameras), and frame 0 the
    first sample.

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
    """Load a scene (.json, see FORMAT.md) and check its object and bin entries.

    Args:
        path: File path, or a scene name resolved as ``scenes/<name>.json`` next to this module.
    """
    scene = json.loads(_resolve(path, "scenes", ".json").read_text())
    for name in OBJECTS:
        obj = scene["objects"][name]
        missing = [key for key in _OBJECT_KEYS if key not in obj]
        if missing:
            raise KeyError(f"{name}: missing {missing}")
        if obj["arm"] not in SIDES:
            raise ValueError(f"{name}: arm must be one of {SIDES}")
        missing = [key for key in _EVENT_KEYS if key not in obj["events"]]
        if missing:
            raise KeyError(f"{name}: events missing {missing}")
    missing = [key for key in _BIN_KEYS if key not in scene["bin"]]
    if missing:
        raise KeyError(f"bin: missing {missing}")
    return scene


def load_gt(path: str | Path) -> dict[str, np.ndarray]:
    """Load ground truth (.npz, see FORMAT.md) as a dict of arrays.

    Args:
        path: File path, or an episode name resolved as ``gt/<name>.npz`` next to this module.
    """
    with np.load(_resolve(path, "gt", ".npz")) as data:
        return {key: np.asarray(data[key]) for key in data.files}


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
    """Copy of a scene with perturbed object starts, as in the verifier's ensembles.

    Each start (the body origin, which is the object's grasp point) moves by N(0, ``xy_sigma``) per axis,
    clipped to ``xy_clip`` [m], and turns about the world z axis through its origin by U(±``yaw_deg``) [deg].
    """
    scene = copy.deepcopy(scene)
    for name in OBJECTS:
        start = scene["objects"][name]["start"]
        start["pos"][:2] = (
            np.asarray(start["pos"][:2]) + np.clip(rng.normal(0.0, xy_sigma, 2), -xy_clip, xy_clip)
        ).tolist()
        if yaw_deg > 0.0:
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


def handle_radius(obj: dict, x) -> np.ndarray:
    """Radius [m] of an object's body of revolution at axial positions ``x`` [m] of its body frame.

    The handle follows the piecewise-linear ``geometry.handle_profile`` between its first and last station;
    from there to the tip the radius is ``geometry.shaft_radius_m``; outside the object it is 0.

    Args:
        obj: Scene object entry (``scene["objects"][name]``).
        x: Body-frame x positions [m] (+x toward the tip).
    """
    geometry = obj["geometry"]
    profile = np.asarray(geometry["handle_profile"], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    tip = float(obj["tip_local"][0])
    radius = np.interp(x, profile[:, 0], profile[:, 1])
    radius = np.where(x > profile[-1, 0], geometry["shaft_radius_m"], radius)
    return np.where((x < profile[0, 0]) | (x > tip), 0.0, radius)


def grasp_radius(obj: dict) -> float:
    """Handle radius at the grasp point [m]: ``grasp_radius_m`` if given, else the profile at x = 0."""
    if "grasp_radius_m" in obj:
        return float(obj["grasp_radius_m"])
    return float(handle_radius(obj, 0.0))


def polygon_signed_distance(points, polygon) -> np.ndarray:
    """Signed distance [m] from points to a convex polygon in the plane (negative inside).

    Args:
        points: Points [..., 2] [m].
        polygon: Convex polygon vertices [k, 2] [m], in either winding order.

    Returns:
        Euclidean distance to the outline outside, minus the distance to the nearest edge inside, shape [...].
    """
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edge = b - a
    area2 = float(np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]))
    outward = np.stack([edge[:, 1], -edge[:, 0]], axis=-1) * (1.0 if area2 > 0.0 else -1.0)
    outward /= np.linalg.norm(outward, axis=-1, keepdims=True)
    d = points[..., None, :] - a  # [..., k, 2]
    s = np.clip(np.einsum("...kj,kj->...k", d, edge) / np.einsum("kj,kj->k", edge, edge), 0.0, 1.0)
    nearest = np.linalg.norm(d - s[..., None] * edge, axis=-1).min(axis=-1)
    inside = np.all(np.einsum("...kj,kj->...k", d, outward) <= 0.0, axis=-1)
    return np.where(inside, -nearest, nearest)


def bin_frame(bin_spec: dict, table_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin frame ``(origin [3], rotation [3, 3])``: origin on the table under the bin centre, x along its length."""
    yaw = math.radians(bin_spec["yaw_deg"])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return np.array([*bin_spec["center_xy"], table_z], dtype=np.float64), rotation


def bin_rim_polygon(bin_spec: dict) -> np.ndarray:
    """Outer outline of the bin rim in the world xy plane [m], shape [4, 2] (counter-clockwise)."""
    origin, rotation = bin_frame(bin_spec, 0.0)
    half = 0.5 * np.asarray(bin_spec["top_size_m"], dtype=np.float64)
    corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64) * half
    return origin[:2] + corners @ rotation[:2, :2].T


def bin_inner_polygon(bin_spec: dict, height: float = 0.0, *, world: bool = False) -> np.ndarray:
    """Inner wall outline of the tapered bin at ``height`` above the table [m], shape [4, 2] (counter-clockwise).

    The outer outline tapers linearly from ``bottom_size_m`` at the table to ``top_size_m`` at the rim
    (``height_m``); the inner wall lies ``wall_m`` inside it. Heights are clipped to [0, ``height_m``].

    Args:
        bin_spec: Scene ``bin``.
        height: Height above the table [m].
        world: Return world xy coordinates instead of bin-frame coordinates.
    """
    f = float(np.clip(height, 0.0, bin_spec["height_m"])) / float(bin_spec["height_m"])
    bottom, top = np.asarray(bin_spec["bottom_size_m"]), np.asarray(bin_spec["top_size_m"])
    half = 0.5 * (bottom + f * (top - bottom)) - float(bin_spec["wall_m"])
    corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64) * half
    if not world:
        return corners
    origin, rotation = bin_frame(bin_spec, 0.0)
    return origin[:2] + corners @ rotation[:2, :2].T


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
        self.camera_body = {s: bodies[CAMERA_BODY[s]] for s in SIDES}

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

    def object_starts(self, episode: dict, scene: dict) -> dict[str, dict]:
        """Object start poses consistent with the real grasp.

        The grasp point's xy is where the midpoint of the grasping arm's two pad axes meets the grasp height
        (``table_z + grasp_height_m``), averaged over the top frames from finger contact to the frame before
        lift-off (the object rests on the table then). The object's long axis (body +x, toward the tip) is
        perpendicular to the closing direction one frame before lift-off, on the side of ``axis_hint_xy``,
        and pitched by ``rest_pitch_rad`` about the body y axis (positive lowers the tip).

        Args:
            episode: Episode dict.
            scene: Scene dict (object ``arm``, ``events``, ``grasp_height_m``, optional ``axis_hint_xy`` and
                ``rest_pitch_rad``, and ``table_z``).

        Returns:
            ``{name: {"pos": [x, y, z], "quat_xyzw": [x, y, z, w], "yaw_deg", "closing_xy"}}``; ``pos`` is the
            grasp point [m].
        """
        rows = frame_rows(episode)
        out = {}
        for name in OBJECTS:
            obj = scene["objects"][name]
            side, events = obj["arm"], obj["events"]
            height = scene["table_z"] + obj["grasp_height_m"]
            points = []
            for frame in range(events["contact"], max(events["liftoff"] - 1, events["contact"] + 1)):
                pads = self.pads(self.pose_row(episode, int(rows[frame])), side)
                on_plane = [centre + axis * (height - centre[2]) / axis[2] for centre, axis, _ in pads]
                points.append(0.5 * (on_plane[0] + on_plane[1]))
            xy = np.mean(points, axis=0)[:2]
            pads = self.pads(self.pose_row(episode, int(rows[max(events["liftoff"] - 1, 0)])), side)
            closing = pads[1][0] - pads[0][0]
            yaw = math.atan2(closing[1], closing[0]) + math.pi / 2
            hint = np.asarray(obj.get("axis_hint_xy", [math.cos(yaw), math.sin(yaw)]), dtype=np.float64)
            if np.dot(hint, [math.cos(yaw), math.sin(yaw)]) < 0.0:
                yaw += math.pi
            pitch = float(obj.get("rest_pitch_rad", 0.0))
            tilt = np.array([0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0)])
            quat = _quat_multiply(_quat_about_z(yaw), tilt)
            out[name] = {
                "pos": [float(xy[0]), float(xy[1]), float(height)],
                "quat_xyzw": quat.tolist(),
                "yaw_deg": math.degrees(yaw),
                "closing_xy": [float(closing[0]), float(closing[1])],
            }
        return out

    def attached_tracks(self, episode: dict, scene: dict) -> dict[str, np.ndarray]:
        """Object grasp points per top frame if each object stayed rigidly in its hand from contact to release.

        The in-hand offset is the object's grasp pose (``grasp_start``, else ``start``) relative to the hand
        (``<side>_link_6``) at the contact frame. Frames before contact hold that pose; frames from release on are
        NaN.

        Returns:
            ``{name: positions [frames, 3] [m]}``.
        """
        rows = frame_rows(episode)
        out = {}
        for name in OBJECTS:
            obj = scene["objects"][name]
            side, events = obj["arm"], obj["events"]
            start = np.asarray(obj.get("grasp_start", obj["start"])["pos"], dtype=np.float64)
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


def open_command_row(episode: dict, obj: dict) -> int:
    """State row where an object's carry window ends (the scene's ``open_cmd_row``).

    The last row before the grasping arm's logged gripper command (held from its timestamps, no delay) rises
    :data:`OPEN_CMD_RISE` above its hold value for the release: the hold value is the median command from the
    carry start ``row(liftoff) + 3`` to ``row(cmd_open)``, and the rise is the run of rows above it that lasts
    until ``row(release)`` (smaller excursions of the command during the carry do not end it).

    Args:
        episode: Episode dict.
        obj: The object's scene entry (``arm`` and ``events``).
    """
    side, top = obj["arm"], frame_rows(episode)
    row = {key: int(top[min(obj["events"][key], len(top) - 1)]) for key in ("liftoff", "cmd_open", "release")}
    first = row["liftoff"] + CARRY_START_OFFSET
    t = np.asarray(episode["left_t"], dtype=np.float64)
    stamps = np.asarray(episode[f"{side}_grip_cmd_t"], dtype=np.float64)
    rows = np.clip(np.searchsorted(stamps, t, side="right") - 1, 0, len(stamps) - 1)
    command = np.asarray(episode[f"{side}_grip_cmd"], dtype=np.float64)[rows]
    hold = float(np.median(command[first : max(first + 1, row["cmd_open"])]))
    above = command > hold + OPEN_CMD_RISE
    end = min(row["release"], len(t) - 1)
    if not above[end]:
        rising = np.flatnonzero(above[first:])
        return first + int(rising[0]) - 1 if rising.size else end
    while end > first and above[end - 1]:
        end -= 1
    return end - 1


# ----------------------------------------------------------------------------- replay


class _Station:
    """Per-world indices of the station joints and bodies, the objects, and an optional bin body."""

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
        self.arm_joints, self.finger_joints, self.object_bodies, self.bin_body = [], [], [], []
        self.pad_bodies, self.hand_body, self.camera_body, self.world_bodies = [], [], [], []
        for w in range(self.worlds):
            self.arm_joints.append({s: [one(joints, w, f"{s}_joint{j + 1}", "joint") for j in range(6)] for s in SIDES})
            self.finger_joints.append(
                {
                    s: (one(joints, w, f"{s}_left_finger", "joint"), one(joints, w, f"{s}_right_finger", "joint"))
                    for s in SIDES
                }
            )
            self.object_bodies.append({name: one(bodies, w, name, "body") for name in OBJECTS})
            bins = bodies.get((w, BIN_BODY), [])
            if len(bins) > 1:
                raise ValueError(f"world {w} has {len(bins)} bodies named {BIN_BODY!r}")
            self.bin_body.append(bins[0] if bins else None)
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
    at the measured opening, and the objects where ``model.joint_q`` placed them.

    For MCP checkpoints, expose :meth:`checkpoint_arrays` as attributes of the example (the MCP
    snapshots the example's own Warp arrays) together with :attr:`state_0` and :attr:`control`.

    Args:
        model: Batched station model (one world per episode; joints and bodies labelled as in the station
            MJCF, the object as a body labelled ``screwdriver``, a dynamic bin as a body labelled ``bin``).
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
        normalized-lerp rotations). Keys (see FORMAT.md): ``t``, ``top_row``, ``<side>_wrist_row``,
        ``<object>_pos``, ``<object>_quat``, ``bin_pos``/``bin_quat`` (dynamic bin only), ``<side>_pad``,
        ``<side>_hand_pos``, ``<side>_hand_quat``, ``<side>_camera_body_pos``, ``<side>_camera_body_quat``,
        ``<side>_finger``, ``<side>_q``, ``<side>_q_real``, ``<side>_grip_real``, ``body_q``, ``body_index``,
        ``world``, ``complete``.
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
            for name in OBJECTS:
                pose = body_q[:, column[station.object_bodies[w][name]]]
                rec[f"{name}_pos"], rec[f"{name}_quat"] = pose[:, :3], pose[:, 3:]
            if station.bin_body[w] is not None:
                pose = body_q[:, column[station.bin_body[w]]]
                rec["bin_pos"], rec["bin_quat"] = pose[:, :3], pose[:, 3:]
            for side in SIDES:
                points = []
                for body in station.pad_bodies[w][side]:
                    pose = body_q[:, column[body]]
                    points.append(pose[:, :3] + quat_to_matrix(pose[:, 3:]) @ np.asarray(PAD_POINT))
                rec[f"{side}_pad"] = 0.5 * (points[0] + points[1])
                hand = body_q[:, column[station.hand_body[w][side]]]
                rec[f"{side}_hand_pos"], rec[f"{side}_hand_quat"] = hand[:, :3], hand[:, 3:]
                if station.camera_body[w][side] is not None:
                    cam = body_q[:, column[station.camera_body[w][side]]]
                    rec[f"{side}_camera_body_pos"], rec[f"{side}_camera_body_quat"] = cam[:, :3], cam[:, 3:]
                if f"t_{side}" in episode:
                    rec[f"{side}_wrist_row"] = frame_rows(episode, side)
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


def _angle_deg(rotations: np.ndarray) -> np.ndarray:
    """Rotation angles [deg] of rotation matrices [..., 3, 3]."""
    cosine = (np.trace(rotations, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _bin_frame_final(rec: dict, scene: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Bin frame at the end of a recording: the scene's, moved like a dynamic bin body; and that body's shift [m]."""
    origin, rotation = bin_frame(scene["bin"], scene["table_z"])
    if "bin_pos" not in rec:
        return origin, rotation, 0.0
    p0, pf = np.asarray(rec["bin_pos"][0]), np.asarray(rec["bin_pos"][-1])
    r0, rf = quat_to_matrix(rec["bin_quat"][0]), quat_to_matrix(rec["bin_quat"][-1])
    motion = rf @ r0.T
    return motion @ (origin - p0) + pf, motion @ rotation, float(np.linalg.norm(pf - p0))


def wrist_camera_pose(recording: dict, side: str, row: int, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    """Wrist camera ``(position [m], rotation_xyzw)`` of an arm at recording row ``row``.

    Args:
        recording: One world's recording (needs ``<side>_camera_body_pos``/``_quat``).
        side: ``"left"`` or ``"right"``.
        row: Recording row.
        camera: Camera calibration (:func:`load_camera`); its ``<side>`` entry's mount is applied.
    """
    return _mount(camera[side], recording[f"{side}_camera_body_pos"][row], recording[f"{side}_camera_body_quat"][row])


def _mount(intrinsics: dict, body_pos, body_quat) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(intrinsics.get("body_rotation_xyzw", CAMERA_IN_BODY_XYZW), dtype=np.float64)
    offset = np.asarray(intrinsics.get("body_offset_m", (0.0, 0.0, 0.0)), dtype=np.float64)
    body_pos, body_quat = np.asarray(body_pos, dtype=np.float64), np.asarray(body_quat, dtype=np.float64)
    return body_pos + quat_to_matrix(body_quat) @ offset, _quat_multiply(body_quat, rotation)


def _score_object(rec: dict, scene: dict, gt: dict, name: str, camera: dict | None) -> dict:
    obj = scene["objects"][name]
    side, events, radius = obj["arm"], obj["events"], grasp_radius(obj)
    t, top_row = rec["t"], np.asarray(rec["top_row"])
    count = len(t)
    position = np.asarray(rec[f"{name}_pos"], dtype=np.float64)
    rotation = quat_to_matrix(rec[f"{name}_quat"])
    start = position[0]
    hand_rotation = quat_to_matrix(rec[f"{side}_hand_quat"])
    in_hand = np.einsum("nji,nj->ni", hand_rotation, position - rec[f"{side}_hand_pos"])
    row = {key: int(top_row[min(events[key], len(top_row) - 1)]) for key in _EVENT_KEYS}
    m: dict = {"arm": side}

    close = min(row["cmd_close"], count - 1)
    m["moved_before_grasp_m"] = float(np.linalg.norm(position[: close + 1, :2] - start[:2], axis=1).max())
    m["max_rise_m"] = float(position[:, 2].max() - start[2])

    # The carry ends before the gripper command starts to open, so the window does not depend on the command delay.
    first = row["liftoff"] + CARRY_START_OFFSET
    last = min(int(obj["open_cmd_row"]), row["release"] - CARRY_END_OFFSET)
    carry = np.arange(first, last + 1) if last < count else np.zeros(0, dtype=np.int64)
    distance = np.linalg.norm(position - rec[f"{side}_pad"], axis=1)
    m["held_fraction"] = float(np.mean(distance[carry] <= radius + HOLD_MARGIN)) if carry.size else 0.0
    m["held"] = bool(carry.size and m["held_fraction"] >= HELD_FRACTION_MIN)
    if carry.size:
        m["slip_m"] = float(np.linalg.norm(in_hand[carry] - in_hand[first], axis=1).max())
        # Object orientation in the hand frame, relative to the carry start: R_rel(n)^T R_rel(first).
        relative = np.einsum("nji,njk->nik", hand_rotation[carry], rotation[carry])
        m["inhand_rot_deg"] = float(_angle_deg(np.einsum("nji,jk->nik", relative, relative[0])).max())
    else:
        m["slip_m"] = m["inhand_rot_deg"] = None

    track = _gt_rows(gt.get(f"{name}_pos"), top_row, count)
    known = carry[np.isfinite(track[carry, 0])] if carry.size else carry
    if known.size:
        m["carry_track_err_m"] = float(np.median(np.linalg.norm(position[known, :2] - track[known, :2], axis=1)))
        m["carry_track_err_3d_m"] = float(np.median(np.linalg.norm(position[known] - track[known], axis=1)))
    else:
        m["carry_track_err_m"] = m["carry_track_err_3d_m"] = None

    lift = _first(position[close:, 2] - start[2] > LIFT_THRESHOLD)
    m["liftoff_err_rows"] = None if lift is None else int(close + lift - row["liftoff"])
    m["liftoff_err_s"] = None if lift is None else float(t[close + lift] - t[min(row["liftoff"], count - 1)])

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

    # Placement: tip and centre of mass inside the bin's inner wall polygon (+ margin) at their heights, the
    # centre of mass below the rim, and at rest. A dynamic bin's frame moves with its body.
    tip_local = np.asarray(obj["tip_local"], dtype=np.float64)
    com_local = np.asarray(obj["com_local"], dtype=np.float64)
    com = position + rotation @ com_local
    tip = position[-1] + rotation[-1] @ tip_local
    bin_origin, bin_rotation, m["bin_moved_m"] = _bin_frame_final(rec, scene)
    in_bin = (np.stack([tip, com[-1]]) - bin_origin) @ bin_rotation  # bin-frame coordinates [2, 3]
    walls = [
        float(polygon_signed_distance(p[:2], bin_inner_polygon(scene["bin"], p[2]))) for p in in_bin
    ]  # tip, centre of mass
    m["tip_wall_m"], m["com_wall_m"] = walls
    m["com_below_rim_m"] = float(scene["bin"]["height_m"] - in_bin[1, 2])
    window = np.flatnonzero(t >= t[-1] - REST_WINDOW - 1e-9)
    elapsed = float(t[-1] - t[window[0]]) if window.size else 0.0
    path = float(np.linalg.norm(np.diff(com[window], axis=0), axis=1).sum()) if window.size > 1 else 0.0
    m["final_speed_mps"] = path / elapsed if elapsed > 0.0 else None
    # The last term only rejects an object that sank through the bin floor and the table.
    m["in_bin"] = bool(max(walls) <= PLACE_MARGIN and m["com_below_rim_m"] >= RIM_CLEARANCE and in_bin[1, 2] > -0.01)
    m["placed"] = bool(
        rec.get("complete", True)
        and m["in_bin"]
        and m["final_speed_mps"] is not None
        and m["final_speed_mps"] < REST_SPEED
    )
    m["final_tip_xyz"] = [float(v) for v in tip]
    m["final_com_xyz"] = [float(v) for v in com[-1]]
    axis = rotation[-1][:, 0]
    m["final_yaw_deg"] = float(math.degrees(math.atan2(axis[1], axis[0])))
    real_yaw = gt.get(f"{name}_final_yaw_deg")
    m["final_yaw_err_deg"] = (
        None if real_yaw is None else float((m["final_yaw_deg"] - float(real_yaw) + 180.0) % 360.0 - 180.0)
    )
    real_tip = gt.get(f"{name}_final_tip_xyz")
    m["final_tip_xy_err_m"] = None if real_tip is None else float(np.linalg.norm(tip[:2] - np.asarray(real_tip)[:2]))

    # Report only: the handle collar in the grasping arm's wrist camera against its tracked image position,
    # over the wrist frames that show carry rows.
    m["wrist_collar_err_px"] = m["wrist_collar_v_range_px"] = None
    real_uv = gt.get(f"{name}_wrist_collar_uv")
    wrist_rows = rec.get(f"{side}_wrist_row")
    if camera is not None and real_uv is not None and wrist_rows is not None and "collar_local" in obj and carry.size:
        real_uv = np.asarray(real_uv, dtype=np.float64)
        frames = [
            f
            for f in range(min(len(real_uv), len(wrist_rows)))
            if np.all(np.isfinite(real_uv[f])) and first <= wrist_rows[f] <= last
        ]
        if frames and f"{side}_camera_body_pos" in rec:
            collar_local = np.asarray(obj["collar_local"], dtype=np.float64)
            pixels = []
            for f in frames:
                r = int(wrist_rows[f])
                pose = wrist_camera_pose(rec, side, r, camera)
                pixels.append(project_points(camera[side], position[r] + rotation[r] @ collar_local, pose))
            pixels = np.asarray(pixels)
            m["wrist_collar_err_px"] = float(np.median(np.linalg.norm(pixels - real_uv[frames], axis=1)))
            m["wrist_collar_v_range_px"] = float(np.ptp(pixels[:, 1]))
    return m


def score(recording: dict, scene: dict, gt: dict | None = None, camera: dict | None = None) -> dict:
    """Replay metrics of one world's recording against the real episode.

    All times are on the arm-state clock, sampled at the episode's state samples; scene event frames are
    converted with ``recording["top_row"]``. The carry window runs from ``row(liftoff) + 3`` to the scene's
    ``open_cmd_row`` (the last state row before the gripper command starts to open; at most ``row(release) - 2``).
    The grasp point is the object's body origin. Per object (``objects[name]``):

    - ``held_fraction``/``held``: fraction of carry rows with the grasp point within the grasp radius + 15 mm
      of the grasping arm's pad midpoint; held if at least 0.9.
    - ``inhand_rot_deg``: largest rotation of the object in the grasping hand's frame (``<side>_link_6``) over
      the carry rows, relative to the first carry row.
    - ``slip_m``: largest drift of the grasp point in the hand frame over the carry rows.
    - ``placed``: complete episode; final tip and centre of mass (``tip_local``, ``com_local``) inside the
      bin's inner wall polygon at their heights dilated by 5 mm (``tip_wall_m``, ``com_wall_m``: signed
      distances, negative inside); centre of mass at least 2 cm below the rim (``com_below_rim_m``); mean
      speed of the centre of mass under 2 cm/s over the last 0.3 s (``final_speed_mps``). A dynamic bin's
      frame moves with its body (``bin_moved_m``).
    - ``liftoff_err_rows`` (``liftoff_err_s``): first row after the close command with the grasp point 1 cm
      above its start, minus the real lift-off row.
    - ``carry_track_err_m``: median xy distance to the ground-truth track over carry rows
      (``carry_track_err_3d_m`` in 3D, report only).
    - ``grip_gap_err_mm``: median finger-gap error (sim minus measured opening) from lift-off to the open
      command.
    - ``final_tip_xy_err_m``, ``final_yaw_err_deg``: final tip position and axis yaw against the real ones.
    - ``moved_before_grasp_m``: largest xy displacement of the grasp point up to the close command.
    - ``max_rise_m``: highest grasp-point height above its start over the episode (negative controls).
    - Report only: ``release_err_s`` (sim release, the grasp point 1 cm from its in-hand position at the
      carry start, minus the real release) and ``wrist_collar_err_px`` / ``wrist_collar_v_range_px`` (the
      projected handle collar in the grasping arm's wrist camera; need ``camera`` and a tracked collar in
      ``gt``), over the wrist frames that show carry rows.

    Args:
        recording: One world's recording (:meth:`Replay.recordings`).
        scene: The world's scene.
        gt: Ground truth (:func:`load_gt`); ``None`` scores without it (track and final errors ``None``).
        camera: Camera calibration (:func:`load_camera`) for the wrist metric.

    Returns:
        ``{"complete", "arm_rmse_rad": {side: rad}, "objects": {name: metrics}, "all_held", "all_placed"}``.
    """
    gt = {} if gt is None else gt
    camera = None if camera is None else load_camera(camera)
    result = {"complete": bool(recording.get("complete", True)), "arm_rmse_rad": {}, "objects": {}}
    for side in SIDES:
        error = np.asarray(recording[f"{side}_q"]) - np.asarray(recording[f"{side}_q_real"])
        result["arm_rmse_rad"][side] = float(np.sqrt(np.mean(error**2)))
    for name in OBJECTS:
        result["objects"][name] = _score_object(recording, scene, gt, name, camera)
    result["all_held"] = all(m["held"] for m in result["objects"].values())
    result["all_placed"] = all(m["placed"] for m in result["objects"].values())
    return result


def check(metrics: dict, thresholds: dict | None = None) -> dict:
    """Pass/fail of one scored rollout against per-rollout limits (default :data:`DEFAULT_THRESHOLDS`).

    Metrics that need ground truth the episode lacks (``None`` track or final errors) are skipped; metrics
    that are ``None`` because the event never happened (lift-off, carry, gap) fail.

    Returns:
        ``{"pass": bool, "failed": [names]}``.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    failed = [] if metrics["complete"] else ["complete"]
    failed += [f"arm_rmse_rad.{s}" for s, v in metrics["arm_rmse_rad"].items() if not v <= limits["arm_rmse_rad"]]

    def at_most(value, limit, absolute=False):
        return value is not None and (abs(value) if absolute else value) <= limit

    for name, m in metrics["objects"].items():
        tests = {
            "held": m["held_fraction"] >= limits["held_fraction"],
            "placed": m["placed"],
            "inhand_rot_deg": at_most(m["inhand_rot_deg"], limits["inhand_rot_deg"]),
            "slip_m": at_most(m["slip_m"], limits["slip_m"]),
            "grip_gap_err_mm": at_most(m["grip_gap_err_mm"], limits["grip_gap_err_mm"], absolute=True),
            "liftoff_err_rows": at_most(m["liftoff_err_rows"], limits["liftoff_err_rows"], absolute=True),
            "moved_before_grasp_m": at_most(m["moved_before_grasp_m"], limits["moved_before_grasp_m"]),
        }
        for key in ("carry_track_err_m", "final_tip_xy_err_m"):
            if m[key] is not None:
                tests[key] = m[key] <= limits[key]
        failed += [f"{name}.{key}" for key, ok in tests.items() if not ok]
    return {"pass": not failed, "failed": failed}


def summarize(results: list[dict]) -> dict:
    """Ensemble summary of several scored rollouts of one episode: per-object counts and medians.

    Returns:
        ``{"copies", "arm_rmse_rad": {side: median}, "objects": {name: {"held", "placed", "held_placed_rot_ok",
        <metric>: median}}}``; ``held_placed_rot_ok`` counts copies held, placed, and rotated at most
        :data:`HELDOUT_ROT_MAX_DEG` in hand (the unseen-episode rule).
    """
    out = {"copies": len(results), "arm_rmse_rad": {}, "objects": {}}
    for side in SIDES:
        out["arm_rmse_rad"][side] = float(np.median([r["arm_rmse_rad"][side] for r in results]))
    for name in OBJECTS:
        rows = [r["objects"][name] for r in results]
        entry = {
            "held": sum(m["held"] for m in rows),
            "placed": sum(m["placed"] for m in rows),
            "held_placed_rot_ok": sum(
                bool(m["held"] and m["placed"] and m["inhand_rot_deg"] is not None)
                and m["inhand_rot_deg"] <= HELDOUT_ROT_MAX_DEG
                for m in rows
            ),
        }
        for key, value in rows[0].items():
            if (isinstance(value, float | int) and not isinstance(value, bool)) or value is None:
                values = [m[key] for m in rows if m[key] is not None and not isinstance(m[key], bool | list)]
                entry[key] = float(np.median(values)) if values else None
        out["objects"][name] = entry
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
    (``left_camera_frame``/``right_camera_frame``) with the mount in camera.json. Images have the size in
    camera.json (848x480) and are sRGB through each camera's calibrated intrinsics and distortion,
    supersampled in linear light.

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
        self._shape_world = model.shape_world.numpy()
        body_leaf = [_leaf(label) for label in model.body_label]
        # Bin shapes by label: shapes of a body labelled "bin", or static shapes labelled "bin...".
        self._bin_shapes = np.array(
            [
                (body >= 0 and body_leaf[body] == BIN_BODY) or (body < 0 and _leaf(label).startswith(BIN_BODY))
                for label, body in zip(model.shape_label, self._shape_body, strict=True)
            ],
            dtype=bool,
        )

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
            masks: Also return the visible pixels of each object and of the bin.

        Returns:
            sRGB image [H, W, 3] uint8, plus ``{object: bool mask [H, W], "bin": bool mask [H, W]}`` with
            ``masks``.
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
        hit = (shapes >= 0) & (shapes < len(self._shape_body))
        safe = np.where(hit, shapes, 0)
        body = np.where(hit, self._shape_body[safe], -1)
        masks_out = {name: body == self.station.object_bodies[world][name] for name in OBJECTS}
        masks_out["bin"] = hit & self._bin_shapes[safe] & np.isin(self._shape_world[safe], (world, -1))
        return image, masks_out

    def top_pose(self) -> tuple:
        """Fitted top-camera ``(position, rotation_xyzw)`` from camera.json."""
        top = self.camera["top"]
        return top["position"], top["rotation_xyzw"]

    def wrist_pose(self, state: newton.State, side: str, world: int = 0) -> tuple:
        """Wrist-camera ``(position, rotation_xyzw)`` of an arm in ``state`` (camera body pose and mount)."""
        body = self.station.camera_body[world][side]
        if body is None:
            raise ValueError(f"world {world} has no body {CAMERA_BODY[side]!r}")
        pose = state.body_q.numpy()[body].astype(np.float64)
        return _mount(self.camera[side], pose[:3], pose[3:])

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
    """Top-camera image [H, W, 3] uint8 of one world (cached :class:`StationRenderer` per model)."""
    return _renderer(model, camera).render_top(state, world, masks=masks)


def render_wrist(
    model: newton.Model, state: newton.State, side: str, world: int = 0, *, camera=None, masks: bool = False
):
    """Wrist-camera image [H, W, 3] uint8 of one arm in one world (cached :class:`StationRenderer` per model)."""
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
        a: Shapes: label substring of a shape or its body (e.g. ``"right_lf_down"``, ``"screwdriver"``),
            shape index, or a list of these.
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
