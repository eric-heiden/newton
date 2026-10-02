# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the ABC screwdriver-in-bin physical replay task files (``abc_bin``).

Sources (ABC-130k, https://abc.bot/#data, Apache-2.0), task "put the screwdriver in the bin":

- the val episode ``02eb2789`` from the Voxel51 MCAP mirror (top, left and right wrist cameras, arm and
  gripper logs at about 30 Hz): the main episode,
- the LeRobot 10 fps mirror (``Dario-Shit2/ABC-130k``, ``put_the_screwdriver_in_the_bin__480x848``, train and
  val subsets): the pool of episodes with the same station, top-camera pose, pink bin, and yellow/black
  screwdriver (08-27 afternoon and 08-28). Four are agent siblings; the held-out episodes and the remaining
  pool episodes (used only to fit the calibration) are named in the private spec only,
- the YAM station MJCF and meshes (``abc_sim/models``, as in abc_replay).

Stages (the decoders and Newton live in different environments)::

    # abc-data venv (mcap, mcap-protobuf-support, pyarrow, av, Pillow)
    python prepare_data.py extract --mcap EPISODE.mcap --lerobot TRAIN_DIR VAL_DIR --raw RAW --spec SPEC.json
    # trial venv (Newton, SciPy, Pillow); cwd = the newton checkout
    python prepare_data.py fit --raw RAW --station STATION --spec SPEC.json --out CALIBRATION.json
    python prepare_data.py build --raw RAW --station STATION --calibration CALIBRATION.json --spec SPEC.json \\
        --task TASK_DIR --private PRIVATE_DIR [--arm-logs DIR] [--sheets DIR]
    python prepare_data.py quality --private PRIVATE_DIR --reference REFERENCE.py [--names ...]

``extract`` decodes the MCAP and the pool's 10 fps logs, per-episode camera calibration, and top and right
wrist frames into a raw cache.

``fit`` measures the station calibration and writes it to a JSON file that ``build`` reads:

- the top camera pose and the right arm base (x offset and yaw; its y follows the measured 610 mm mount
  separation of the dataset metadata), maximizing the edge NCC of robot renders against 20 main-episode
  top frames minus a penalty on the distance between the kinematic grasp point and the image screwdriver axis
  on the clean starts (``camera_fit_observations`` of the spec),
- the right wrist camera mount (rotation and offset in its MJCF body frame) from the gripper silhouette in the
  wrist frames before the grasp,
- the bin dimensions (silhouettes in the first top frames, height fixed, medians over the pool),
- the handle collar row in the right wrist view during the hold, and the common bias that maps it to the
  grasp location along the axis (anchored on the clean starts), and the tapered handle profile fitted to the
  held gripper gaps.

``build`` writes one scene, ground truth, and episode per episode (main, the agent siblings, and every pool
episode in ``PRIVATE/candidates``), the agent dataset, the verifier's copies, the held-out set, and review
sheets (``--sheets``). The world frame is the station MJCF's turned about the fitted right base so that the
right arm has no yaw (scene ``bases`` carry positions only); the left base is moved with it (its 2.7 deg yaw is
not represented: the left arm only stands by).

``quality`` replays a reference solution (``build_model`` / ``make_solver`` / ``make_pipeline`` / ``PARAMS``)
on candidate episodes under the construction variants of the build plan (dt, bin floor, base correction,
handle profile, bin position along the approach, start yaw) and writes ``heldout/quality.json``: an episode
whose verdict (held, placed, and rotated at most ``replay_common.HELDOUT_ROT_MAX_DEG`` in hand) changes in any
variant is ``unreliable``.

The held-out ids live only in the private spec (``PRIVATE/heldout_spec.json``: ``pool``, ``heldout``,
``spare``, ``manual_checks``, ``camera_fit_observations``), so this file does not name them.
File formats are described in FORMAT.md.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    import replay_common as rc
except ImportError:  # the extract stage runs in an environment without Newton
    rc = None

HERE = Path(__file__).resolve().parent
MAIN = "02eb2789"
MAIN_UUID = "02eb2789-4234-4d4c-984e-08872ed00c1d"
# Agent siblings (08-28 recordings, whose wrist intrinsics match camera.json), chosen after the quality study
# among the episodes with stable reference verdicts, to spread the bin pose and the grasp location.
SIBLINGS = {"sib_1": "2e59ebb4", "sib_2": "a97b9225", "sib_3": "f98b76c8", "sib_4": "e8aeefdb"}
MCAP_CAMERAS = {"/top-camera": "top", "/left-wrist-camera": "left_wrist", "/right-wrist-camera": "right_wrist"}
LEROBOT_CAMERAS = {"top_camera": "top", "right_wrist_camera": "right_wrist"}
JPEG_QUALITY = 90

TABLE_Z = 0.75
MJCF_BASES = {"left": [0.2525, 0.31, 0.75], "right": [0.2525, -0.31, 0.75]}
MOUNT_SEPARATION = 0.61  # right base y = left base y - this: the dataset's measured mount (mount_y_mm = -610)
TIME_BASE = "top frame i shows arm-state sample top_state_index[i] (default i - 1); see FORMAT.md"

# Screwdriver (yellow/black handle, steel shaft). Lengths from the main episode's first top frame (handle 110-113 mm,
# shaft 94-99 mm, total 0.207 m); the handle profile from the held gaps (fit stage); masses assumed.
HANDLE_LENGTH = 0.110  # butt to collar [m]
SHAFT_LENGTH = 0.097  # collar to tip [m]
SHAFT_RADIUS = 0.003  # [m]
HANDLE_MASS, SHAFT_MASS = 0.045, 0.025  # [kg] (assumed; the replay tolerates 0.5-3x)
# In-hand model used to map the wrist collar row to the grasp location (the resting axis at the grasp).
COLLAR_MODEL_PITCH = 0.06  # [rad]
COLLAR_MODEL_HEIGHT = 0.0135  # [m] axis height at the grasp above the table
BIN_WALL = 0.004  # [m] (assumed)
BIN_FLOOR = 0.004  # [m] (assumed; the quality variants use 1 and 8 mm)
BIN_HEIGHT = 0.13  # [m] rim height: see the fit stage notes
CARRY_PAD = 3  # rows after lift-off / before release excluded from the hold windows
# The closing jaws push the real screwdriver about 1 cm before they grip it (main episode, right wrist view), while
# the grasp pose (the pad midpoint at contact) is the pose after the push. Scene starts are the grasp pose moved this
# far toward the butt along the axis: from there the simulated jaws catch the handle near the recorded grasp location
# (finger-gap error of the reference on main +0.9 mm, against +4.1 mm when starting at the grasp pose).
PRE_PUSH = 0.008  # [m]

# Results of checking the agent episodes by hand on the review sheets (tip direction on the first top frame, grasp
# in the wrist view, bin fit, final pose); held-out checks are in the private spec. The tip direction (from the
# wrist view: the shaft points up in the image) matched the first top frame in all 57 pool episodes.
SIBLING_CHECKS = {
    "main": {
        "verdict": "usable",
        "notes": [
            "the closing fingers push the screwdriver about 1 cm (right wrist view); the start is 8 mm behind the grasp pose "
            "(grasp_start) along the axis",
            "final pose from the last 25 top frames: tip spread 0.7 mm, yaw spread 0.19 deg",
        ],
    },
    "sib_1": {
        "verdict": "usable",
        "notes": [
            "the real bin moved 21 mm between the first and the last top frame (the final pose is measured in the moved bin)"
        ],
    },
    "sib_2": {"verdict": "usable", "notes": ["bin far from the robot (centre x 0.72 m)"]},
    "sib_3": {"verdict": "usable", "notes": ["grasped 29 mm from the collar (held gap 19.4 mm)"]},
    "sib_4": {"verdict": "usable", "notes": ["bin close to the right arm (centre y -0.13 m)"]},
}


# ----------------------------------------------------------------------------- extract


def _save_jpeg(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=JPEG_QUALITY)


def extract_mcap(path: Path, raw: Path) -> None:
    """Decode the main episode's MCAP: arm and gripper logs, camera info, and all frames as JPEG."""
    import av  # noqa: PLC0415
    from mcap.reader import make_reader  # noqa: PLC0415
    from mcap_protobuf.decoder import DecoderFactory  # noqa: PLC0415

    video = {name: [] for name in MCAP_CAMERAS.values()}
    cameras, series = {}, {}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        topics = [c.topic for c in reader.get_summary().channels.values() if not c.topic.endswith(".plot")]
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics, log_time_order=True):
            topic = channel.topic
            if topic in MCAP_CAMERAS:
                video[MCAP_CAMERAS[topic]].append((message.log_time, bytes(decoded.data)))
            elif topic.endswith("-info") and topic[:-5] in MCAP_CAMERAS:
                cameras[MCAP_CAMERAS[topic[:-5]].replace("_wrist", "")] = {
                    "width": decoded.width,
                    "height": decoded.height,
                    "distortion_model": decoded.distortion_model,
                    "K": list(decoded.K),
                    "D": list(decoded.D),
                }
            elif topic.endswith("arm-state"):
                series.setdefault(topic, []).append(
                    [
                        message.log_time,
                        *decoded.position[:6],
                        *decoded.velocity[:7],
                        *decoded.torque[:7],
                        *decoded.pose[:16],
                    ]
                )
            elif topic.endswith("arm-action"):
                series.setdefault(topic, []).append([message.log_time, *decoded.position[:6], *decoded.pose[:16]])
            elif topic.endswith("ee-state") or topic.endswith("ee-action"):
                series.setdefault(topic, []).append([message.log_time, decoded.position[0]])

    t0 = video["top"][0][0]
    out = {}
    for name, messages in video.items():
        codec = av.CodecContext.create("h264", "r")
        frames = []
        for _, data in messages:  # one access unit per message
            frames.extend(frame.to_ndarray(format="rgb24") for frame in codec.decode(av.Packet(data)))
        frames.extend(frame.to_ndarray(format="rgb24") for frame in codec.decode(None))
        if len(frames) != len(messages):
            raise RuntimeError(f"{name}: decoded {len(frames)} frames from {len(messages)} messages")
        for i, image in enumerate(frames):
            _save_jpeg(image, raw / "frames" / "main" / name / f"{i:03d}.jpg")
        out[f"t_{name.replace('_wrist', '')}"] = (np.array([m[0] for m in messages]) - t0) / 1e9
    for side in ("left", "right"):
        a = np.array(series[f"/{side}-arm-state"])
        out[f"{side}_t"] = (a[:, 0] - t0) / 1e9
        out[f"{side}_q"], out[f"{side}_qd"], out[f"{side}_tau"] = a[:, 1:7], a[:, 7:14], a[:, 14:21]
        out[f"{side}_pose"] = a[:, 21:37].reshape(-1, 4, 4)
        a = np.array(series[f"/{side}-arm-action"])
        out[f"{side}_cmd_t"], out[f"{side}_cmd"] = (a[:, 0] - t0) / 1e9, a[:, 1:7]
        out[f"{side}_cmd_pose"] = a[:, 7:23].reshape(-1, 4, 4)
        for topic, key in (("ee-state", "grip"), ("ee-action", "grip_cmd")):
            a = np.array(series[f"/{side}-{topic}"])
            out[f"{side}_{key}_t"], out[f"{side}_{key}"] = (a[:, 0] - t0) / 1e9, a[:, 1]
    raw.mkdir(parents=True, exist_ok=True)
    np.savez(raw / "main.npz", **out)
    (raw / "cameras.json").write_text(json.dumps(cameras, indent=1) + "\n")
    print(f"main: {len(out['t_top'])} top frames, {out['t_top'][-1]:.2f} s")


def _decode_range(path: Path, start: float, stop: float) -> list[np.ndarray]:
    """Frames of a video with presentation times in [start, stop) [s]."""
    import av  # noqa: PLC0415

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((start - 1.0) / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start - 1e-3:
                continue
            if t >= stop - 1e-3:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def extract_lerobot(roots: list[Path], uuids: list[str], raw: Path) -> None:
    """Copy the 10 fps logs and per-episode camera calibration of the given episodes (8-character uuid prefixes,
    searched in every LeRobot subset of ``roots``) and decode their top and right-wrist frames."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    subsets = []
    for root in roots:
        sources = json.loads((root / "meta" / "source_episodes.json").read_text())["episodes"]
        subsets.append(
            (
                root,
                pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict(),
                pq.read_table(root / "meta" / "episode_metadata.parquet").to_pydict(),
                {s["uuid"][len("episode_") :][:8]: s for s in sources},
            )
        )
    tables = {}
    for uuid in uuids:
        found = [s for s in subsets if uuid in s[3]]
        if not found:
            raise KeyError(f"{uuid}: not in any LeRobot subset")
        root, episodes, metadata, sources = found[0]
        source = sources[uuid]
        row = next(i for i, s in enumerate(metadata["source_episode_id"]) if s[8:16] == uuid)
        index = metadata["episode_index"][row]
        e = episodes["episode_index"].index(index)
        offset = (source["t0_ns"] - source["stream_span_ns"]["/top-camera"][0]) / 1e9
        key = (str(root), episodes["data/chunk_index"][e], episodes["data/file_index"][e])
        if key not in tables:
            tables[key] = pq.read_table(
                root / "data" / f"chunk-{key[1]:03d}" / f"file-{key[2]:03d}.parquet"
            ).to_pydict()
        table = tables[key]
        mask = np.asarray(table["episode_index"]) == index
        arrays = {
            "timestamp": np.asarray(table["timestamp"], dtype=np.float64)[mask],
            "frame_index": np.asarray(table["frame_index"])[mask],
            **{
                name.replace(".", "_"): np.asarray(table[name], dtype=np.float64)[mask]
                for name in ("observation.state", "action", "observation.velocity", "observation.torque")
            },
            "eef_pose_base": np.asarray(table["observation.eef_pose_base"], dtype=np.float64)[mask],
            "time_offset": np.float64(offset),
            "uuid": np.str_(source["uuid"]),
            "t0_ns": np.int64(source["t0_ns"]),
            "mount_y_mm": np.float64(source.get("mount_y_mm", np.nan)),
            "calibration_json": np.str_(json.dumps(source["calibration"])),
        }
        counts = {}
        for camera, name in LEROBOT_CAMERAS.items():
            chunk = episodes[f"videos/observation.images.{camera}/chunk_index"][e]
            file = episodes[f"videos/observation.images.{camera}/file_index"][e]
            video = root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
            if not video.exists():
                continue
            start = episodes[f"videos/observation.images.{camera}/from_timestamp"][e]
            stop = episodes[f"videos/observation.images.{camera}/to_timestamp"][e]
            frames = _decode_range(video, start, stop)[: int(mask.sum())]
            for i, image in enumerate(frames):
                _save_jpeg(image, raw / "frames" / uuid / name / f"{i:03d}.jpg")
            counts[name] = len(frames)
        (raw / "lerobot").mkdir(parents=True, exist_ok=True)
        np.savez(raw / "lerobot" / f"{uuid}.npz", **arrays)
        print(f"{uuid}: {mask.sum()} samples, offset {offset:.4f} s, frames {counts}", flush=True)


# ----------------------------------------------------------------------------- shared helpers


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float) + "\n")


def _write_npz(path: Path, arrays: dict, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compressed else np.savez)(path, **arrays)


def _round(values, digits=5) -> list[float]:
    return [round(float(v), digits) for v in values]


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _frames(raw: Path, name: str, camera: str = "top") -> list[Path]:
    return sorted((raw / "frames" / name / camera).glob("*.jpg"))


def _hsv(image: np.ndarray):
    x = image.astype(np.float32) / 255.0
    high, low = x.max(-1), x.min(-1)
    d = high - low + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(high == r, ((g - b) / d) % 6, np.where(high == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (high + 1e-6), high


def yellow_mask(image: np.ndarray) -> np.ndarray:
    """Yellow handle parts (collar ring, inserts, butt cap)."""
    h, s, v = _hsv(image)
    return (h > 35) & (h < 66) & (s > 0.42) & (v > 0.28)


def pink_mask_top(image: np.ndarray) -> np.ndarray:
    """The pink bin in a top frame: largest pink blob below the back wall, holes filled."""
    from scipy import ndimage

    h, s, v = _hsv(image)
    m = ((h < 40) | (h > 350)) & (s > 0.16) & (s < 0.65) & (v > 0.3)
    m[:110] = False
    m = ndimage.binary_opening(m, iterations=2)
    labels, count = ndimage.label(m)
    if count == 0:
        return m
    sizes = ndimage.sum(m, labels, range(1, count + 1))
    m = labels == (np.argmax(sizes) + 1)
    return ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3))


class Camera:
    """Pinhole camera with RealSense inverse Brown-Conrady intrinsics; Newton pose convention (looks along -Z, +Y up)."""

    def __init__(self, intrinsics: dict, rotation_xyzw, position):
        self.intrinsics = dict(intrinsics)
        self.K = np.asarray(intrinsics["K"], dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(intrinsics["D"], dtype=np.float64)
        self.width, self.height = int(intrinsics["width"]), int(intrinsics["height"])
        self.rotation_xyzw = np.asarray(rotation_xyzw, dtype=np.float64)
        self.R_wc = rc.quat_to_matrix(self.rotation_xyzw) @ np.diag([1.0, -1.0, -1.0])  # OpenCV axes
        self.c = np.asarray(position, dtype=np.float64)

    def _undistort(self, x, y):
        k1, k2, p1, p2, k3 = self.D
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
        return x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)

    def backproject_z(self, u, v, z):
        """Points [..., 3] where the rays through pixels (u, v) meet the plane at height z [m]."""
        x = (np.asarray(u, dtype=np.float64) - self.K[0, 2]) / self.K[0, 0]
        y = (np.asarray(v, dtype=np.float64) - self.K[1, 2]) / self.K[1, 1]
        ux, uy = self._undistort(x, y)
        d = np.stack([ux, uy, np.ones_like(ux)], axis=-1) @ self.R_wc.T
        s = (np.asarray(z, dtype=np.float64) - self.c[2]) / d[..., 2]
        return self.c + s[..., None] * d

    def project(self, points):
        """Pixels [..., 2] of world points [..., 3]."""
        pc = (np.asarray(points, dtype=np.float64) - self.c) @ self.R_wc
        xu, yu = pc[..., 0] / pc[..., 2], pc[..., 1] / pc[..., 2]
        x, y = xu.copy(), yu.copy()
        for _ in range(20):
            fx, fy = self._undistort(x, y)
            x, y = x - (fx - xu), y - (fy - yu)
        return np.stack([x * self.K[0, 0] + self.K[0, 2], y * self.K[1, 1] + self.K[1, 2]], axis=-1)


def _rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(m).as_quat()


class World:
    """The fitted station frame turned about the right base so that the right arm has no yaw.

    ``fitted`` coordinates are the camera fit's (MJCF world with the right base moved by ``right_base_xy`` and
    turned by ``right_base_yaw``); scene coordinates are these turned by ``-right_base_yaw`` about the right base.
    """

    def __init__(self, calibration: dict, left_anchor=None):
        self.pivot = np.array([*calibration["right_base_xy"], TABLE_Z], dtype=np.float64)
        self.rotation = _rotz(-float(calibration["right_base_yaw"]))
        top = calibration["top"]
        self.camera = Camera(
            top,
            _quat_from_matrix(self.rotation @ rc.quat_to_matrix(np.asarray(top["rotation_xyzw"]))),
            self.point(np.asarray(top["position"])),
        )
        base = np.asarray(MJCF_BASES["left"], dtype=np.float64)
        if left_anchor is None:
            left = self.point(base)
        else:
            # The left arm cannot carry the turn (scene bases are positions); place its base so that its gripper
            # (``left_anchor``, fitted frame, at its standby pose) lands where the turned frame puts it.
            anchor = np.asarray(left_anchor, dtype=np.float64)
            left = self.point(anchor) - (anchor - base)
            left[2] = base[2]
        self.bases = {"left": _round(left, 5), "right": _round(self.pivot, 5)}

    def point(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, dtype=np.float64) - self.pivot) @ self.rotation.T + self.pivot


# ----------------------------------------------------------------------------- episodes, holds, FK


def lerobot_episode(raw: Path, uuid: str) -> dict:
    data = np.load(raw / "lerobot" / f"{uuid}.npz")
    return rc.episode_from_lerobot(
        data["timestamp"],
        data["observation_state"],
        data["action"],
        velocity=data["observation_velocity"],
        torque=data["observation_torque"],
        time_offset=float(data["time_offset"]),
    )


def lerobot_meta(raw: Path, uuid: str) -> dict:
    data = np.load(raw / "lerobot" / f"{uuid}.npz")
    return {"uuid": str(data["uuid"])[len("episode_") :], "calibration": json.loads(str(data["calibration_json"]))}


def load_any_episode(raw: Path, name: str) -> dict:
    return dict(np.load(raw / "main.npz")) if name == "main" else lerobot_episode(raw, name)


def wrist_intrinsics(raw: Path, name: str) -> dict:
    """Right wrist intrinsics of an episode (the 08-27 and 08-28 recordings differ: fx 437.5 vs 434.2 px)."""
    if name == "main":
        return json.loads((raw / "cameras.json").read_text())["right"]
    cal = lerobot_meta(raw, name)["calibration"]["/right-wrist-camera-info"]
    return {
        "width": int(cal.get("width", 848)),
        "height": int(cal.get("height", 480)),
        "distortion_model": cal.get("distortion_model", "inverse_brown_conrady"),
        "K": [float(v) for v in cal["K"]],
        "D": [float(v) for v in cal["D"]],
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def find_holds(episode: dict, sides=("left", "right"), min_rows: int = 6) -> list[dict]:
    """Grasps on the state grid (abc_replay's rule): rows of the close command, finger contact, the open command,
    and release, from the measured opening and the gripper command. Lift-off is added from the start."""
    holds = []
    for side in sides:
        _, g = rc.measured(episode, side)
        c = np.interp(episode["left_t"], episode[f"{side}_grip_cmd_t"], episode[f"{side}_grip_cmd"])
        n = len(g)
        for a, b in _runs(c < 0.35):
            if b - a < min_rows:
                continue
            k = a
            while k > 0 and c[k - 1] < 0.95:
                k -= 1
            j = k + 1
            while j < b - 1 and not (g[j] < 0.95 and g[j + 1] >= g[j] - 0.004 and c[j] < g[j] - 0.05):
                j += 1
            g_hold = float(np.median(g[min(j + 3, b - 1) : b]))
            o = b
            while o < n - 1 and c[o] <= g_hold + 0.02:
                o += 1
            r = o
            while r < n - 1 and g[r] < g_hold + 0.05:
                r += 1
            holds.append(
                {"arm": side, "rows": {"cmd_close": k, "contact": j, "cmd_open": o, "release": r}, "g_hold": g_hold}
            )
    holds.sort(key=lambda h: h["rows"]["contact"])
    return holds


def rows_to_frames(rows: dict[str, int], top_rows: np.ndarray) -> dict[str, int]:
    """Top-frame index showing each state row (nearest)."""
    return {key: int(np.argmin(np.abs(top_rows - row))) for key, row in rows.items()}


def pre_push_start(obj: dict) -> None:
    """Keep the grasp pose as ``grasp_start`` and move ``start`` :data:`PRE_PUSH` toward the butt along the axis."""
    grasp = obj.get("grasp_start", obj["start"])
    rotation = rc.quat_to_matrix(np.asarray(grasp["quat_xyzw"], dtype=np.float64))
    axis = rotation[:2, 0] / np.linalg.norm(rotation[:2, 0])
    pos = np.asarray(grasp["pos"], dtype=np.float64)
    pos[:2] -= PRE_PUSH * axis
    obj["grasp_start"] = {"pos": list(grasp["pos"]), "quat_xyzw": list(grasp["quat_xyzw"])}
    obj["start"] = {"pos": _round(pos), "quat_xyzw": list(grasp["quat_xyzw"])}


class FKCache:
    """Measured-joint FK of one episode (StationFK of the scene bases), cached per state row."""

    BODIES = ("right_link_6", "right_camera_frame", "right_lf_down", "right_rf_down")

    def __init__(self, fk, episode: dict):
        self.fk, self.episode = fk, episode
        leaf = [label.rsplit("/", 1)[-1] for label in fk.model.body_label]
        self.index = [leaf.index(b) for b in self.BODIES]
        self._rows = {}

    def body_q(self, row: int) -> np.ndarray:
        if row not in self._rows:
            self._rows[row] = self.fk.pose_row(self.episode, int(row))
        return self._rows[row]

    def bodies(self, row: int) -> np.ndarray:
        return self.body_q(row)[self.index]


# ----------------------------------------------------------------------------- wrist collar track


def detect_collar(image: np.ndarray):
    """Handle collar in a right wrist frame (A-data's rule): the topmost sizeable yellow blob; also the shaft angle."""
    from scipy import ndimage

    h, s, v = _hsv(image)
    yel = ndimage.binary_opening((h > 35) & (h < 66) & (s > 0.45) & (v > 0.35), iterations=1)
    labels, count = ndimage.label(yel)
    if count == 0:
        return None
    cents = np.array(ndimage.center_of_mass(yel, labels, range(1, count + 1)))
    sizes = ndimage.sum(yel, labels, range(1, count + 1))
    ok = sizes > 30
    if not ok.any():
        return None
    k = int(np.argmin(np.where(ok, cents[:, 0], 1e9)))
    cv, cu = cents[k]
    return [float(cu), float(cv)]


def collar_track(raw: Path, name: str, episode: dict, rows: dict) -> dict:
    """Collar pixel per right wrist frame from contact to release (NaN elsewhere) and its hold-window median."""
    files = _frames(raw, name, "right_wrist")
    wrows = rc.frame_rows(episode, "right")
    uv = np.full((len(files), 2), np.nan)
    for f, path in enumerate(files):
        if rows["contact"] <= wrows[f] <= rows["release"]:
            d = detect_collar(_load_image(path))
            if d is not None:
                uv[f] = d
    hold = [
        f for f in range(len(files)) if rows["contact"] + 6 <= wrows[f] <= rows["release"] - 3 and np.isfinite(uv[f, 0])
    ]
    out = {"uv": uv, "hold_frames": hold}
    if hold:
        out["v_median"] = float(np.median(uv[hold, 1]))
        out["u_median"] = float(np.median(uv[hold, 0]))
        out["v_range"] = float(np.ptp(uv[hold, 1]))
    return out


def mount_pose(body: np.ndarray, mount: dict) -> tuple[np.ndarray, np.ndarray]:
    """Wrist camera (position, xyzw) of a camera body pose [7] with a camera.json-style mount."""
    rotation = np.asarray(mount.get("body_rotation_xyzw", rc.CAMERA_IN_BODY_XYZW), dtype=np.float64)
    offset = np.asarray(mount.get("body_offset_m", (0.0, 0.0, 0.0)), dtype=np.float64)
    return body[:3] + rc.quat_to_matrix(body[3:]) @ offset, rc._quat_multiply(body[3:], rotation)


D_GRID = np.linspace(-0.02, 0.14, 161)


def grasp_geometry(cache: FKCache, rows: dict, height: float) -> dict:
    """Grasp point (pad-axis midpoint at ``height`` over contact .. lift-off - 1), closing direction, lift-off row."""
    centres, axes = [], []
    for row in range(rows["contact"], rows["release"]):
        b = cache.bodies(row)
        pads = [
            (b[k, :3] + rc.quat_to_matrix(b[k, 3:]) @ np.asarray(rc.PAD_CENTRE), rc.quat_to_matrix(b[k, 3:])[:, 2])
            for k in (2, 3)
        ]
        centres.append([p[0] for p in pads])
        axes.append([p[1] for p in pads])
    centres, axes = np.asarray(centres), np.asarray(axes)
    mid_z = centres[:, :, 2].mean(1)
    rise = np.flatnonzero(mid_z > mid_z[0] + 0.01)
    lift = rows["contact"] + (int(rise[0]) if rise.size else 2)
    span = slice(0, max(lift - 1 - rows["contact"], 1))
    z = TABLE_Z + height
    on = centres[span] + axes[span] * ((z - centres[span][..., 2]) / axes[span][..., 2])[..., None]
    closing = centres[max(lift - 1 - rows["contact"], 0), 1] - centres[max(lift - 1 - rows["contact"], 0), 0]
    return {"grasp": np.array([*on.mean(1).mean(0)[:2], z]), "closing": closing, "lift_pad_row": int(lift)}


def collar_curve(raw: Path, name: str, episode: dict, cache: FKCache, rows: dict, mount: dict) -> dict:
    """Median projected collar row v(d) in the right wrist view on the hold frames, for collars d along the axis
    from the grasp point (in-hand model: the resting axis at the grasp, carried rigidly from contact); the axis
    sign puts the shaft up in the image, as in every real frame. Also the tip direction in the world (xy)."""
    g = grasp_geometry(cache, rows, COLLAR_MODEL_HEIGHT)
    yaw = math.atan2(g["closing"][1], g["closing"][0]) + math.pi / 2
    hand_c = cache.bodies(rows["contact"])[0]
    rh_c = rc.quat_to_matrix(hand_c[3:])
    intr = wrist_intrinsics(raw, name)
    wrows = rc.frame_rows(episode, "right")
    frames = [f for f in range(len(wrows)) if rows["contact"] + 6 <= wrows[f] <= rows["release"] - 3]
    best = None
    for sign in (0.0, math.pi):
        a = yaw + sign
        axis = np.array(
            [
                math.cos(COLLAR_MODEL_PITCH) * math.cos(a),
                math.cos(COLLAR_MODEL_PITCH) * math.sin(a),
                -math.sin(COLLAR_MODEL_PITCH),
            ]
        )
        local_p, local_a = rh_c.T @ (g["grasp"] - hand_c[:3]), rh_c.T @ axis
        uv = []
        for f in frames:
            b = cache.bodies(int(wrows[f]))
            rh = rc.quat_to_matrix(b[0, 3:])
            points = b[0, :3][None, :] + (local_p[None, :] + D_GRID[:, None] * local_a[None, :]) @ rh.T
            uv.append(rc.project_points(intr, points, mount_pose(b[1], mount)))
        med = np.median(np.asarray(uv), axis=0)
        slope = med[-1, 1] - med[len(D_GRID) // 2, 1]
        if best is None or slope < best[0]:
            best = (slope, a, med)
    k = intr["K"]
    v_main = (best[2][:, 1] - k[5]) / k[4]  # normalized image row (intrinsics-free)
    return {"v_norm": v_main, "tip_yaw": best[1], "grasp": g, "intrinsics": intr}


def common_curve(curves: list[np.ndarray]) -> np.ndarray:
    return np.median(np.asarray(curves), axis=0)


def d_from_collar(v_px: float, intr: dict, curve_norm: np.ndarray, bias_px_main: float, main_fy: float) -> float:
    """Grasp-to-collar distance [m] from a real collar row: the common normalized curve, bias in main pixels."""
    v_norm = (v_px - intr["K"][5]) / intr["K"][4] - bias_px_main / main_fy
    return float(np.interp(-v_norm, -curve_norm, D_GRID))


# ----------------------------------------------------------------------------- screwdriver geometry


def handle_profile_s(calibration: dict) -> np.ndarray:
    """Handle radius stations [[s, r], ...] with s the distance from the collar toward the butt [m]."""
    return np.asarray(calibration["handle_profile_s"], dtype=np.float64)


def screwdriver_geometry(d: float, profile_s: np.ndarray, cylinder_radius: float | None = None) -> dict:
    """Body-frame geometry (origin at the grasp point, +x toward the tip) for a grasp ``d`` from the collar."""
    if cylinder_radius is not None:
        profile = np.array([[d - HANDLE_LENGTH, cylinder_radius], [d, cylinder_radius]])
    else:
        profile = np.array([[d - s, r] for s, r in profile_s[::-1]])  # butt -> collar
    x_tip = d + SHAFT_LENGTH
    xs = np.linspace(profile[0, 0], profile[-1, 0], 400)
    area = np.pi * np.interp(xs, profile[:, 0], profile[:, 1]) ** 2
    handle_x = float(np.sum(area * xs) / np.sum(area))
    shaft_x = 0.5 * (d + x_tip)
    com = (HANDLE_MASS * handle_x + SHAFT_MASS * shaft_x) / (HANDLE_MASS + SHAFT_MASS)
    # Resting pose: the axis touches the table under the tip (shaft) and under the handle station that tilts it most.
    best = max(range(len(profile)), key=lambda i: (profile[i, 1] - SHAFT_RADIUS) / (x_tip - profile[i, 0]))
    xi, ri = profile[best]
    pitch = math.asin((ri - SHAFT_RADIUS) / math.hypot(x_tip - xi, ri - SHAFT_RADIUS))
    height = ri * math.cos(pitch) + xi * math.sin(pitch)
    radius = float(np.interp(0.0, profile[:, 0], profile[:, 1]))
    return {
        "geometry": {
            "handle_profile": [_round(p, 5) for p in profile],
            "shaft_radius_m": SHAFT_RADIUS,
            "length_m": round(HANDLE_LENGTH + SHAFT_LENGTH, 4),
            "mass_kg": HANDLE_MASS + SHAFT_MASS,
        },
        "butt_local": [round(d - HANDLE_LENGTH, 5), 0.0, 0.0],
        "collar_local": [round(d, 5), 0.0, 0.0],
        "tip_local": [round(x_tip, 5), 0.0, 0.0],
        "com_local": [round(com, 5), 0.0, 0.0],
        "grasp_radius_m": round(radius, 5),
        "grasp_height_m": round(height, 5),
        "rest_pitch_rad": round(pitch, 5),
    }


# ----------------------------------------------------------------------------- bin


def bin_corners(p) -> np.ndarray:
    cx, cy, yaw, length, width, height, taper = p
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for z, k in ((TABLE_Z, taper), (TABLE_Z + height, 1.0)):
        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            lx, ly = sx * length / 2 * k, sy * width / 2 * k
            pts.append([cx + c * lx - s * ly, cy + s * lx + c * ly, z])
    return np.array(pts)


def bin_silhouette(camera: Camera, p) -> np.ndarray:
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    uv = camera.project(bin_corners(p))
    hull = ConvexHull(uv)
    im = Image.new("L", (camera.width, camera.height), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, uv[i])) for i in hull.vertices], fill=1)
    return np.asarray(im, bool)


def fit_bin(camera: Camera, mask: np.ndarray, *, height: float, dims=None, init=None):
    """Tapered open box (centre xy, yaw, rim length/width, height, bottom/top scale) to a top-frame silhouette
    (IoU, Nelder-Mead; A-data fit_bin.py). ``height`` is held; ``dims`` = (length, width, scale) holds those too."""
    from scipy import optimize

    vv, uu = np.nonzero(mask)
    pts = camera.backproject_z(uu.astype(float), vv.astype(float), TABLE_Z + 0.05)[:, :2]
    centre = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    yaw0 = math.atan2(vecs[1, 1], vecs[0, 1])

    def full(q):
        if dims is None:
            return [q[0], q[1], q[2], q[3], q[4], height, q[5]]
        return [q[0], q[1], q[2], dims[0], dims[1], height, dims[2]]

    def loss(q):
        p = full(q)
        if not (0.15 < p[3] < 0.5 and 0.12 < p[4] < 0.4 and 0.6 < p[6] <= 1.0):
            return 2.0
        s = bin_silhouette(camera, p)
        return 1 - (s & mask).sum() / (s | mask).sum()

    best = None
    yaws = [yaw0, yaw0 + math.pi / 2] if init is None else [init[2]]
    for yaw in yaws:
        q0 = [centre[0], centre[1], yaw] if init is None else list(init[:3])
        if dims is None:
            q0 += [0.30, 0.23, 0.9]
        r = optimize.minimize(loss, q0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 3000})
        r = optimize.minimize(loss, r.x, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    p = np.asarray(full(best.x), dtype=float)
    if p[4] > p[3]:  # canonical: length >= width
        p = np.array([p[0], p[1], p[2] + math.pi / 2, p[4], p[3], p[5], p[6]])
    p[2] = (p[2] + math.pi / 2) % math.pi - math.pi / 2
    return p, 1.0 - float(best.fun)


def bin_entry(p, iou: float, floor: float = BIN_FLOOR, **extra) -> dict:
    top = [float(p[3]), float(p[4])]
    return {
        "center_xy": _round(p[:2], 5),
        "yaw_deg": round(math.degrees(p[2]), 3),
        "top_size_m": _round(top, 5),
        "bottom_size_m": _round([top[0] * p[6], top[1] * p[6]], 5),
        "height_m": round(float(p[5]), 4),
        "wall_m": BIN_WALL,
        "floor_m": floor,
        "top_fit_iou": round(iou, 4),
        **extra,
    }


# ----------------------------------------------------------------------------- fit: top camera and right base


def _edge_map(rgb):
    from scipy import ndimage

    g = rgb.astype(np.float32).mean(-1) / 255.0
    e = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    e = ndimage.gaussian_filter(e, 1.0)
    return (e - e.mean()) / (e.std() + 1e-6)


def _set_robot_q(model, labels, q_start, joint_q, worlds_rows, meas, per_world):
    for w, r in enumerate(worlds_rows):
        for jj in range(w * per_world, (w + 1) * per_world):
            name = labels[jj]
            for side in rc.SIDES:
                qs, grip = meas[side]
                for j in range(6):
                    if name == f"{side}_joint{j + 1}":
                        joint_q[q_start[jj]] = qs[r, j]
                g = float(np.clip(grip[r], 0, 1)) * rc.GRIPPER_TRAVEL
                if name == f"{side}_left_finger":
                    joint_q[q_start[jj]] = g
                if name == f"{side}_right_finger":
                    joint_q[q_start[jj]] = -g
    model.joint_q.assign(joint_q)


def camera_observations(spec: dict, raw: Path, station: Path) -> list[dict]:
    """Clean-start screwdriver observations: image axis pixels and the pad axes (MJCF bases) from contact to lift-off."""
    fk = rc.StationFK(station, MJCF_BASES)
    out = []
    for name, o in sorted(spec["camera_fit_observations"].items()):
        episode = load_any_episode(raw, name)
        rows = find_holds(episode, sides=("right",))[0]["rows"]
        cache = FKCache(fk, episode)
        centres, axes = [], []
        for row in range(rows["contact"], rows["release"]):
            b = cache.bodies(row)
            centres.append([b[k, :3] + rc.quat_to_matrix(b[k, 3:]) @ np.asarray(rc.PAD_CENTRE) for k in (2, 3)])
            axes.append([rc.quat_to_matrix(b[k, 3:])[:, 2] for k in (2, 3)])
            if len(centres) > 1 and np.mean([c[2] for c in centres[-1]]) > np.mean([c[2] for c in centres[0]]) + 0.01:
                break
        n = max(len(centres) - 2, 1)
        out.append(
            {
                "name": name,
                "butt_uv": np.asarray(o["butt_uv"]),
                "tip_uv": np.asarray(o["tip_uv"]),
                "centres": np.asarray(centres[:n]),
                "axes": np.asarray(axes[:n]),
            }
        )
    return out


def base_transform(dx: float, dy: float, dyaw: float):
    """(R, t) moving right-arm poses of the MJCF base by (dx, dy) and turning them by dyaw about the base vertical."""
    rot = _rotz(dyaw)
    origin = np.asarray(MJCF_BASES["right"], dtype=np.float64)
    return rot, origin + np.array([dx, dy, 0.0]) - rot @ origin


def observation_terms(camera: Camera, o: dict, transform) -> tuple[float, float]:
    """Distance of the kinematic grasp point from the image axis (table plane) and its position along the axis
    from the image butt [m]."""
    rot, t = transform
    c = o["centres"].reshape(-1, 3) @ rot.T + t
    a = o["axes"].reshape(-1, 3) @ rot.T
    z = TABLE_Z + COLLAR_MODEL_HEIGHT
    on = c + a * ((z - c[:, 2]) / a[:, 2])[:, None]
    p = on.mean(0)
    butt = camera.backproject_z(*o["butt_uv"], TABLE_Z + 0.0125)
    tip = camera.backproject_z(*o["tip_uv"], TABLE_Z + 0.004)
    axis = (tip - butt)[:2]
    axis /= np.linalg.norm(axis)
    w = p[:2] - butt[:2]
    return abs(float(w[0] * axis[1] - w[1] * axis[0])), float(w @ axis)


def fit_top_camera(
    raw: Path,
    station: Path,
    obs: list[dict],
    init: dict,
    *,
    weight: float,
    fix_dy: float,
    scales=(4, 2, 1),
    frames_count: int = 20,
    subset: int = -1,
    nsub: int = 3,
    fit_yaw: bool = True,
) -> dict:
    """Top camera (6 DoF) and right base x and yaw: edge NCC of robot renders minus ``weight`` x mean(min(perp,
    30 mm)) / 10 mm over the observations; coordinate descent, coarse to fine (A-data calib_camera.py)."""
    import warp as wp  # noqa: PLC0415
    from scipy.spatial.transform import Rotation

    import newton  # noqa: PLC0415
    from newton.sensors import SensorTiledCamera  # noqa: PLC0415

    episode = dict(np.load(raw / "main.npz"))
    intr = json.loads((raw / "cameras.json").read_text())["top"]
    files = _frames(raw, "main")
    frames = np.linspace(2, len(files) - 3, frames_count).round().astype(int)
    if subset >= 0:
        frames = np.arange(2 + subset, len(files) - 3, nsub)
        frames = frames[np.linspace(0, len(frames) - 1, min(frames_count, len(frames))).round().astype(int)]
    full = np.stack([_load_image(files[f]).astype(np.float32) for f in frames])
    rows = rc.frame_rows(episode)
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(station))
    for si, label in enumerate(builder.shape_label):
        body = builder.shape_body[si]
        bname = builder.body_label[body].rsplit("/", 1)[-1] if body >= 0 else ""
        if (
            not (bname.startswith("left_") or bname.startswith("right_"))
            or "camera_d405" in label
            or "camera_d405" in bname
        ):
            builder.shape_flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
    top = newton.ModelBuilder()
    top.replicate(builder, len(frames))
    model = top.finalize()
    state = model.state()
    labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    right_root = [k for k in range(model.joint_count) if labels[k] == "right_arm_joint"]
    X_p0 = model.joint_X_p.numpy().copy()
    meas = {side: rc.measured(episode, side) for side in rc.SIDES}
    _set_robot_q(
        model,
        labels,
        model.joint_q_start.numpy(),
        model.joint_q.numpy(),
        [int(rows[f]) for f in frames],
        meas,
        model.joint_count // len(frames),
    )

    def set_base(y):
        X = X_p0.copy()
        for k in right_root:
            X[k, 0] = MJCF_BASES["right"][0] + y[0]
            X[k, 1] = MJCF_BASES["right"][1] + y[1]
            X[k, 3:7] = [0.0, 0.0, math.sin(y[2] / 2), math.cos(y[2] / 2)]
        model.joint_X_p.assign(X)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        model.bvh_refit_shapes(state)

    sensor = SensorTiledCamera(model)
    sensor.default_render_config.enable_shadows = False
    sensor.utils.create_default_light(enable_shadows=False)
    clear = SensorTiledCamera.ClearData(clear_color=0xFF959895)
    p0 = np.asarray(init["position"], dtype=np.float64)
    r0 = Rotation.from_quat(init["rotation_xyzw"])
    y0 = np.array([init["right_base_offset"][0], fix_dy, init.get("right_base_yaw", 0.0)])
    base_cam = Camera(intr, [0, 0, 0, 1], [0, 0, 0])

    class Level:
        def __init__(self, s):
            self.w, self.h = intr["width"] // s, intr["height"] // s
            self.color = sensor.utils.create_color_image_output(self.w, self.h, camera_count=1)
            uu, vv = np.meshgrid((np.arange(self.w) + 0.5) * s, (np.arange(self.h) + 0.5) * s)
            x = (uu - base_cam.K[0, 2]) / base_cam.K[0, 0]
            y = (vv - base_cam.K[1, 2]) / base_cam.K[1, 1]
            ux, uy = base_cam._undistort(x, y)
            d = np.stack([ux, uy, np.ones_like(ux)], -1)
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            r = np.zeros((1, self.h, self.w, 2, 3), np.float32)
            r[0, :, :, 1] = d * np.array([1, -1, -1])
            self.rays = wp.array(r, dtype=wp.vec3f)
            small = full.reshape(len(frames), self.h, s, self.w, s, 3).mean(axis=(2, 4))
            self.real_edges = np.stack([_edge_map(x) for x in small])

    def pose(x):
        return p0 + x[:3], r0 * Rotation.from_rotvec(x[3:6])

    def score(level, x, detail=False):
        p, rot = pose(x)
        y = y0 + np.array([x[6], 0.0, x[7]])
        set_base(y)
        tf = wp.array([[wp.transformf(wp.vec3f(*p), wp.quatf(*rot.as_quat()))] * len(frames)], dtype=wp.transformf)
        sensor.update(state, tf, level.rays, color_image=level.color, clear_data=clear)
        imgs = level.color.numpy()[:, 0].view(np.uint8).reshape(len(frames), level.h, level.w, 4)[..., :3]
        ncc = float(np.mean([(_edge_map(imgs[i]) * level.real_edges[i]).mean() for i in range(len(frames))]))
        cam = Camera(intr, rot.as_quat(), p)
        tr = base_transform(*y)
        terms = [observation_terms(cam, o, tr) for o in obs]
        perp = np.array([t[0] for t in terms])
        value = ncc - weight * float(np.mean(np.minimum(perp, 0.03))) / 0.01
        return (value, ncc, terms) if detail else value

    x = np.zeros(8)
    dims = [0, 1, 2, 3, 4, 5, 6] + ([7] if fit_yaw else [])
    steps0 = {4: 0.08, 2: 0.03, 1: 0.01}
    for s in scales:
        level = Level(s)
        best = score(level, x)
        steps = np.full(8, steps0.get(s, 0.03))
        steps[6] *= 0.66
        steps[7] = math.radians(2.0) * steps0.get(s, 0.03) / 0.08
        for _ in range(6):
            improved = True
            while improved:
                improved = False
                for d in dims:
                    for sgn in (1, -1):
                        xt = x.copy()
                        xt[d] += sgn * steps[d]
                        st = score(level, xt)
                        if st > best + 1e-6:
                            best, x, improved = st, xt, True
            steps *= 0.5
        print(f"camera fit scale {s}: objective {best:.4f} x={np.round(x, 4).tolist()}", flush=True)
    value, ncc, terms = score(level, x, detail=True)
    p, rot = pose(x)
    y = y0 + np.array([x[6], 0.0, x[7]])
    return {
        "position": p.tolist(),
        "rotation_xyzw": rot.as_quat().tolist(),
        "right_base_offset": [float(y[0]), float(y[1]), 0.0],
        "right_base_yaw": float(y[2]),
        "right_base_xy": [MJCF_BASES["right"][0] + float(y[0]), MJCF_BASES["right"][1] + float(y[1])],
        "edge_ncc": ncc,
        "objective": value,
        "frames": frames.tolist(),
        "observations": {
            o["name"]: {"perp_mm": 1000 * t[0], "along_from_butt_mm": 1000 * t[1]}
            for o, t in zip(obs, terms, strict=True)
        },
        "perp_median_mm": 1000 * float(np.median([t[0] for t in terms])),
    }


# ----------------------------------------------------------------------------- fit: wrist mount (gripper silhouette)


def fit_wrist_mount(raw: Path, station: Path, world: World, names: list[str]) -> dict:
    """Right wrist camera mount: rotation [deg] and offset [mm] in the camera body frame that best overlay the rendered
    right gripper on the real dark gripper pixels (lower 55 % of the image) in the frames before the grasp."""
    import warp as wp  # noqa: PLC0415
    from scipy.spatial.transform import Rotation

    import newton  # noqa: PLC0415
    from newton.sensors import SensorTiledCamera  # noqa: PLC0415

    sub = 2
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(station))
    for si, label in enumerate(builder.shape_label):
        if "camera_d405" in label:
            builder.shape_flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
    rc.set_arm_bases(builder, world.bases)
    model = builder.finalize()
    state = model.state()
    body_leaf = [label.rsplit("/", 1)[-1] for label in model.body_label]
    shape_body = model.shape_body.numpy()
    is_arm = np.array([b >= 0 and body_leaf[b].startswith("right_") for b in shape_body] + [False])
    cam_body = body_leaf.index("right_camera_frame")
    labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    sensor = SensorTiledCamera(model)
    sensor.default_render_config.enable_shadows = False
    frames = []
    for name in names:
        episode = load_any_episode(raw, name)
        rows = find_holds(episode, sides=("right",))[0]["rows"]
        wrows = rc.frame_rows(episode, "right")
        meas = {side: rc.measured(episode, side) for side in rc.SIDES}
        intr = wrist_intrinsics(raw, name)
        rays = rc.camera_rays(intr, 1)[sub // 2 :: sub, sub // 2 :: sub].astype(np.float32)
        for f, path in enumerate(_frames(raw, name, "right_wrist")):
            if wrows[f] >= rows["contact"] - 3 or (name == "main" and f % 4):
                continue
            img = _load_image(path).astype(np.float32)[sub // 2 :: sub, sub // 2 :: sub]
            dark = img.mean(-1) < 60
            dark[: int(0.45 * dark.shape[0])] = False
            jq = model.joint_q.numpy()
            _set_robot_q(model, labels, model.joint_q_start.numpy(), jq, [int(wrows[f])], meas, model.joint_count)
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            frames.append((state.body_q.numpy().copy(), model.joint_q.numpy().copy(), dark, rays))
    print(f"wrist mount fit: {len(frames)} frames", flush=True)

    def evaluate(cands):
        total = np.zeros(len(cands))
        for body_q, joint_q, dark, rays in frames:
            model.joint_q.assign(joint_q)
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            model.bvh_refit_shapes(state)
            packed = np.zeros((len(cands), *rays.shape[:2], 2, 3), dtype=np.float32)
            packed[:, :, :, 1] = rays
            index = sensor.utils.create_shape_index_image_output(rays.shape[1], rays.shape[0], camera_count=len(cands))
            pose = body_q[cam_body]
            rb = Rotation.from_quat(pose[3:])
            base_q = rb * Rotation.from_quat(rc.CAMERA_IN_BODY_XYZW)
            tfs = []
            for c in cands:
                q = (base_q * Rotation.from_rotvec(np.radians(c[:3]))).as_quat()
                p = pose[:3] + rb.apply(np.asarray(c[3:]) / 1000.0)
                tfs.append([wp.transformf(wp.vec3f(*[float(v) for v in p]), wp.quatf(*[float(v) for v in q]))])
            sensor.update(
                state, wp.array(tfs, dtype=wp.transformf), wp.array(packed, dtype=wp.vec3f), shape_index_image=index
            )
            ids = index.numpy()[0].astype(np.int64)
            ids = np.where((ids >= 0) & (ids < len(is_arm) - 1), ids, len(is_arm) - 1)
            m = is_arm[ids]
            m[:, : int(0.45 * m.shape[1])] = False
            total += (m & dark[None]).sum(axis=(1, 2)) / np.maximum((m | dark[None]).sum(axis=(1, 2)), 1)
        return total / len(frames)

    x = np.zeros(6)
    steps = np.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    start = float(evaluate([x])[0])
    value = start
    for _ in range(4):
        improved = True
        while improved:
            improved = False
            cands = [x.copy()]
            for d in range(6):
                for s in (1, -1):
                    c = x.copy()
                    c[d] += s * steps[d]
                    cands.append(c)
            vals = evaluate(cands)
            j = int(np.argmax(vals))
            if vals[j] > vals[0] + 1e-4:
                x, value, improved = cands[j], float(vals[j]), True
                print(f"wrist mount x={np.round(x, 2).tolist()} iou {value:.4f}", flush=True)
        steps *= 0.5
    rotation = (Rotation.from_quat(rc.CAMERA_IN_BODY_XYZW) * Rotation.from_rotvec(np.radians(x[:3]))).as_quat()
    return {
        "rotation_deg": x[:3].tolist(),
        "offset_mm": x[3:].tolist(),
        "body_rotation_xyzw": _round(rotation, 6),
        "body_offset_m": _round(x[3:] / 1000.0, 5),
        "gripper_iou_mjcf": start,
        "gripper_iou_fitted": value,
        "episodes": names,
        "frames": len(frames),
    }


# ----------------------------------------------------------------------------- fit: bin dimensions, grasp, profile


def fit(args: argparse.Namespace) -> None:
    """The fit stage (see the module docstring)."""
    spec = json.loads(args.spec.read_text())
    raw, station = args.raw, args.station
    pool = sorted(spec["pool"])
    t0 = time.perf_counter()
    previous = json.loads(args.out.read_text()) if args.out.exists() else {}
    obs = camera_observations(spec, raw, station)
    intr = json.loads((raw / "cameras.json").read_text())["top"]
    if "camera_fit" in previous and not args.refit_camera:
        cam = previous["camera_fit"]
    else:
        # Stage 1: edges only (camera, right base x and yaw) from A-data's top-camera fit; stage 2: with the
        # screwdriver term. The right base y follows the measured mount separation.
        init = {
            "position": [0.06146875, 0.01046875, 1.7043125],
            "rotation_xyzw": [-0.19193903522771988, 0.19425660854448518, 0.6852092316348083, -0.6752126225679452],
            "right_base_offset": [-0.0044125, 0.0, 0.0],
        }
        dy = MJCF_BASES["left"][1] - MOUNT_SEPARATION - MJCF_BASES["right"][1]
        stage1 = fit_top_camera(raw, station, obs, init, weight=0.0, fix_dy=dy, scales=(4, 2, 1))
        # The objective is flat in the base yaw; stage 2 starts from the stage-1 yaw and from a -2 deg seed and
        # keeps the better result.
        seeds = [stage1, {**stage1, "right_base_yaw": math.radians(-2.0)}]
        if args.camera_seed is not None:
            seed = json.loads(args.camera_seed.read_text())
            seeds.append(
                {
                    "position": seed["position"],
                    "rotation_xyzw": seed.get("rotation_xyzw", seed.get("rotation_xyzw_newton")),
                    "right_base_offset": seed["right_base_offset"],
                    "right_base_yaw": seed["right_base_yaw"],
                }
            )
        runs = [fit_top_camera(raw, station, obs, seed, weight=0.03, fix_dy=dy, scales=(2, 1)) for seed in seeds]
        cam = max(runs, key=lambda r: r["objective"])
        cam["stage2_seeds"] = [
            {k: r[k] for k in ("objective", "edge_ncc", "perp_median_mm", "right_base_yaw")} for r in runs
        ]
        cam["stage1"] = {
            k: stage1[k]
            for k in ("position", "rotation_xyzw", "right_base_xy", "right_base_yaw", "edge_ncc", "perp_median_mm")
        }
    print(
        f"camera: ncc {cam['edge_ncc']:.4f} perp median {cam['perp_median_mm']:.1f} mm yaw {math.degrees(cam['right_base_yaw']):.2f} deg "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    calibration = {
        "top": {**intr, "position": cam["position"], "rotation_xyzw": cam["rotation_xyzw"]},
        "right_base_xy": cam["right_base_xy"],
        "right_base_yaw": cam["right_base_yaw"],
        "camera_fit": cam,
    }
    world = World(calibration)
    if "wrist_mount" in previous and not args.refit_mount:
        mount = previous["wrist_mount"]
    else:
        mount = fit_wrist_mount(raw, station, world, ["main", *spec["mount_fit_episodes"]])
    calibration["wrist_mount"] = mount
    print(
        f"wrist mount {mount['rotation_deg']} deg {mount['offset_mm']} mm, gripper IoU {mount['gripper_iou_mjcf']:.3f} -> "
        f"{mount['gripper_iou_fitted']:.3f} ({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

    # Bin dimensions: free fits of the first top frames with the height held, medians over the pool.
    free = {}
    for name in ["main", *pool]:
        p, iou = fit_bin(world.camera, pink_mask_top(_load_image(_frames(raw, name)[0])), height=BIN_HEIGHT)
        free[name] = {"p": p.tolist(), "iou": iou}
    dims = [float(np.median([free[n]["p"][k] for n in free])) for k in (3, 4, 6)]
    # The free fits are flat in the bottom/top scale; refine the common dimensions on the pose-only fits of a
    # subset (mean IoU), one dimension at a time.
    subset = ["main", *pool[:: max(1, len(pool) // 15)]]
    masks_ = {n: pink_mask_top(_load_image(_frames(raw, n)[0])) for n in subset}

    def mean_iou(dd):
        return float(
            np.mean(
                [fit_bin(world.camera, masks_[n], height=BIN_HEIGHT, dims=dd, init=free[n]["p"])[1] for n in subset]
            )
        )

    best = mean_iou(dims)
    for k, step in ((2, 0.02), (0, 0.005), (1, 0.003), (2, 0.01)):
        improved = True
        while improved:
            improved = False
            for sgn in (1, -1):
                trial = list(dims)
                trial[k] += sgn * step
                value = mean_iou(trial)
                if value > best + 1e-4:
                    dims, best, improved = trial, value, True
    print(
        f"bin dims refined: {np.round(dims, 4).tolist()} mean pose-fit IoU {best:.4f} on {len(subset)} episodes",
        flush=True,
    )
    calibration["bin"] = {
        "rim_length_m": dims[0],
        "rim_width_m": dims[1],
        "bottom_scale": dims[2],
        "height_m": BIN_HEIGHT,
        "free_fit_std": dict(
            zip(
                ("rim_length", "rim_width", "bottom_scale"),
                np.std([[free[n]["p"][k] for k in (3, 4, 6)] for n in free], axis=0).tolist(),
                strict=True,
            )
        ),
        "free_fit_iou_median": float(np.median([free[n]["iou"] for n in free])),
        "free_fit_medians": [float(np.median([free[n]["p"][k] for n in free])) for k in (3, 4, 6)],
        "refined_mean_iou": best,
    }
    print(f"bin dims {np.round(dims, 4).tolist()} ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Grasp location: common collar curve over the pool, bias anchored on the clean starts, then the profile.
    fk = rc.StationFK(station, world.bases)
    camera_json = {
        "right": {"body_rotation_xyzw": mount["body_rotation_xyzw"], "body_offset_m": mount["body_offset_m"]}
    }
    curves, per = [], {}
    for name in ["main", *pool]:
        episode = load_any_episode(raw, name)
        rows = find_holds(episode, sides=("right",))[0]["rows"]
        g_hold = find_holds(episode, sides=("right",))[0]["g_hold"]
        cache = FKCache(fk, episode)
        curve = collar_curve(raw, name, episode, cache, rows, camera_json["right"])
        track = collar_track(raw, name, episode, rows)
        curves.append(curve["v_norm"])
        per[name] = {
            "v_px": track.get("v_median"),
            "v_range_px": track.get("v_range"),
            "intrinsics": curve["intrinsics"],
            "gap_m": 2.0 * float(np.clip(g_hold, 0, 1)) * rc.GRIPPER_TRAVEL,
            "curve": curve["v_norm"],
        }
    common = common_curve(curves)
    main_k = per["main"]["intrinsics"]["K"]
    biases = {}
    for name, o in cam["observations"].items():
        d_top = HANDLE_LENGTH - o["along_from_butt_mm"] / 1000.0
        k = per[name]["intrinsics"]["K"]
        v_norm = (per[name]["v_px"] - k[5]) / k[4]
        biases[name] = float((v_norm - np.interp(d_top, D_GRID, common)) * main_k[4])
    bias = float(np.median(list(biases.values())))
    d = {n: d_from_collar(per[n]["v_px"], per[n]["intrinsics"], common, bias, main_k[4]) for n in per}
    names = [n for n in per if n != MAIN]
    s = np.array([d[n] for n in names])
    gap = np.array([per[n]["gap_m"] for n in names])
    knots = np.array([0.0, 0.055, 0.075])
    basis = np.stack([np.interp(s, knots, np.eye(3)[k]) for k in range(3)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, gap / 2.0, rcond=None)
    residual = gap / 2.0 - basis @ coef
    profile_s = [[0.0, coef[0]], [0.055, coef[1]], [0.075, coef[2]], [0.100, coef[2]], [HANDLE_LENGTH, 0.013]]
    v = np.array([per[n]["v_px"] for n in names])
    calibration["grasp"] = {
        "collar_model": {
            "pitch_rad": COLLAR_MODEL_PITCH,
            "axis_height_m": COLLAR_MODEL_HEIGHT,
            "d_grid_m": D_GRID.tolist(),
            "common_v_norm": common.tolist(),
            "main_fy": main_k[4],
        },
        "bias_px": bias,
        "bias_per_clean_start_px": biases,
        "bias_std_px": float(np.std(list(biases.values()))),
        "d_m": d,
        "gap_vs_collar_v": {
            "corr": float(np.corrcoef(v, gap)[0, 1]),
            "fit_mm": np.polyfit(v, 1000 * gap, 1).tolist(),
            "residual_std_mm": float(np.std(1000 * gap - np.polyval(np.polyfit(v, 1000 * gap, 1), v))),
        },
        "gap_vs_d_corr": float(np.corrcoef(s, gap)[0, 1]),
        "profile_residual_std_gap_mm": float(2000 * residual.std()),
        "d_mm_p10_p50_p90": (1000 * np.percentile(s, [10, 50, 90])).tolist(),
        "wrist_v_range_px": {n: per[n]["v_range_px"] for n in per},
    }
    calibration["handle_profile_s"] = [_round(p, 5) for p in profile_s]
    calibration["notes"] = [
        "camera: robot-edge NCC over 20 main top frames plus the clean-start screwdriver term (weight 0.03); right base y "
        "held at the measured 610 mm mount separation",
        "wrist mount: gripper silhouette before the grasp (bin silhouettes in the wrist view were not used: their IoU "
        "optimum sat at the +-2.5 cm search edge in 55 of 57 episodes, see the build report)",
        f"bin height {BIN_HEIGHT} m held (silhouette fits are flat in height; taller fits match the top silhouette and the "
        "rigid-carry consistency better)",
        "grasp location: collar row on the hold frames through the common in-hand curve; bias = median over the clean "
        "starts of (real row - model row at the top-image grasp location)",
    ]
    _write_json(args.out, calibration)
    print(
        f"bias {bias:.1f} px (std {calibration['grasp']['bias_std_px']:.1f}), profile {calibration['handle_profile_s']}, "
        f"gap residual {calibration['grasp']['profile_residual_std_gap_mm']:.2f} mm ({time.perf_counter() - t0:.0f}s)"
    )


# ----------------------------------------------------------------------------- build: final pose in the top frames


class RobotMasks:
    """Robot-only renders (shape index) of the station posed at measured joints, through the top camera."""

    def __init__(self, station: Path, bases: dict, camera: Camera):
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415
        from newton.sensors import SensorTiledCamera  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(station))
        rc.set_arm_bases(builder, bases)
        robot = []
        for si, label in enumerate(builder.shape_label):
            body = builder.shape_body[si]
            bname = builder.body_label[body].rsplit("/", 1)[-1] if body >= 0 else ""
            keep = (bname.startswith("left_") or bname.startswith("right_")) and "camera_d405" not in label
            robot.append(keep)
            if not keep:
                builder.shape_flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model = builder.finalize()
        self.state = self.model.state()
        self.robot = np.array([*robot, False])
        self.labels = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        self.sensor = SensorTiledCamera(self.model)
        rays = np.zeros((1, camera.height, camera.width, 2, 3), np.float32)
        rays[0, :, :, 1] = rc.camera_rays(camera.intrinsics, 1)
        self.rays = wp.array(rays, dtype=wp.vec3f)
        self.index = self.sensor.utils.create_shape_index_image_output(camera.width, camera.height, camera_count=1)
        self.tf = wp.array([[wp.transformf(wp.vec3f(*camera.c), wp.quatf(*camera.rotation_xyzw))]], dtype=wp.transformf)

    def mask(self, meas: dict, row: int) -> np.ndarray:
        import newton  # noqa: PLC0415

        _set_robot_q(
            self.model,
            self.labels,
            self.model.joint_q_start.numpy(),
            self.model.joint_q.numpy(),
            [row],
            meas,
            self.model.joint_count,
        )
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.bvh_refit_shapes(self.state)
        self.sensor.update(self.state, self.tf, self.rays, shape_index_image=self.index)
        ids = self.index.numpy()[0, 0].astype(np.int64)
        ids = np.where((ids >= 0) & (ids < len(self.robot) - 1), ids, len(self.robot) - 1)
        return self.robot[ids]


def detect_final(image: np.ndarray, robot: np.ndarray, camera: Camera, spec: dict):
    """The screwdriver lying in the bin in a top frame: yellow and locally dark pixels inside the projected bin rim,
    away from the robot; axis by PCA, handle end on the yellow side, tip at the far end of the mask."""
    from PIL import Image, ImageDraw
    from scipy import ndimage

    rim = rc.bin_rim_polygon(spec)
    uv = camera.project(np.c_[rim, np.full(len(rim), TABLE_Z + spec["height_m"])])
    region = Image.new("L", (camera.width, camera.height), 0)
    ImageDraw.Draw(region).polygon([tuple(p) for p in uv], fill=1)
    region = np.asarray(region, bool) & ~ndimage.binary_dilation(robot, iterations=4)
    gray = image.astype(np.float32).mean(-1)
    dark = (gray < ndimage.median_filter(gray, size=21) - 22) & region
    yel = yellow_mask(image) & region
    if yel.sum() < 15:
        return None
    labels, _ = ndimage.label(ndimage.binary_closing(dark | yel, iterations=1))
    ids = np.unique(labels[yel])
    ids = ids[ids > 0]
    if ids.size == 0:
        return None
    sizes = ndimage.sum(labels > 0, labels, ids)
    mask = labels == ids[int(np.argmax(sizes))]
    if mask.sum() < 40:
        return None
    vv, uu = np.nonzero(mask)
    points = np.stack([uu, vv], 1) + 0.5
    centre = points.mean(0)
    _, vecs = np.linalg.eigh(np.cov((points - centre).T))
    axis = vecs[:, 1]
    yc = np.array(np.nonzero(yel & mask))[::-1].mean(1) + 0.5
    if (yc - centre) @ axis > 0:
        axis = -axis  # +axis points from the handle toward the tip
    along = (points - centre) @ axis
    tip_uv = centre + np.percentile(along, 99.5) * axis
    clearance = float(ndimage.distance_transform_edt(~robot)[mask].min()) if robot.any() else 1e9
    floor_z = TABLE_Z + spec["floor_m"]
    tip = camera.backproject_z(*tip_uv, floor_z + SHAFT_RADIUS)
    handle = camera.backproject_z(*yc, floor_z + 0.012)
    return {
        "tip": tip,
        "handle": handle,
        "yaw_deg": math.degrees(math.atan2(*(tip - handle)[1::-1])),
        "length_m": float(np.linalg.norm((tip - handle)[:2])),
        "pixels": int(mask.sum()),
        "clearance_px": clearance,
        "tip_uv": tip_uv.tolist(),
        "handle_uv": yc.tolist(),
    }


def final_pose(
    raw: Path, name: str, episode: dict, rows: dict, masks: RobotMasks, camera: Camera, spec: dict, count: int
):
    """Median final tip and yaw over the last ``count`` top frames with a clean detection (after release)."""
    files = _frames(raw, name)
    top_rows = rc.frame_rows(episode)
    meas = {side: rc.measured(episode, side) for side in rc.SIDES}
    found = []
    for f in range(len(files) - 1, -1, -1):
        if top_rows[f] < rows["release"] + 3 or len(found) >= count:
            break
        det = detect_final(_load_image(files[f]), masks.mask(meas, int(top_rows[f])), camera, spec)
        # Handle centroid to tip: the shaft (97 mm) plus about 30-50 mm of handle.
        if det is not None and det["clearance_px"] >= 8 and 0.09 <= det["length_m"] <= 0.19:
            found.append({"frame": f, **det})
    if len(found) < 2:
        return None
    # The shaft is steel on a pink floor and is sometimes lost: keep the frames that see the longest screwdriver.
    lengths = np.array([d["length_m"] for d in found])
    found = [d for d in found if d["length_m"] >= np.percentile(lengths, 80) - 0.012]
    tips = np.array([d["tip"] for d in found])
    yaws = np.array([d["yaw_deg"] for d in found])
    ref = yaws[0]
    yaws = (yaws - ref + 180.0) % 360.0 - 180.0 + ref
    return {
        "tip": np.median(tips, axis=0),
        "handle": np.median(np.array([d["handle"] for d in found]), axis=0),
        "yaw_deg": float(np.median(yaws)),
        "frames": [d["frame"] for d in found],
        "tip_std_mm": float(1000 * np.linalg.norm(tips.std(0)[:2])),
        "yaw_std_deg": float(yaws.std()),
        "detections": found,
    }


# ----------------------------------------------------------------------------- build: scenes and ground truth


def find_liftoff(fk, episode: dict, rows: dict, side: str, start: np.ndarray) -> int:
    """First row after contact where the grasp point, rigidly attached to the hand from contact, is 1 cm higher."""
    body = fk.hand_body[side]
    pose = fk.pose_row(episode, rows["contact"])[body]
    local = rc.quat_to_matrix(pose[3:]).T @ (start - pose[:3])
    for row in range(rows["contact"], rows["release"]):
        pose = fk.pose_row(episode, row)[body]
        if (pose[:3] + rc.quat_to_matrix(pose[3:]) @ local)[2] > start[2] + 0.01:
            return row
    return rows["release"]


def carried_points(fk, episode: dict, scene: dict, row: int) -> dict:
    """Body-frame points of the screwdriver carried rigidly by the hand from contact, at state ``row``."""
    obj = scene["objects"]["screwdriver"]
    top_rows = rc.frame_rows(episode)
    contact = int(top_rows[obj["events"]["contact"]])
    body = fk.hand_body[obj["arm"]]
    hc, hr = fk.pose_row(episode, contact)[body], fk.pose_row(episode, row)[body]
    rhc, rhr = rc.quat_to_matrix(hc[3:]), rc.quat_to_matrix(hr[3:])
    grasp = obj.get("grasp_start", obj["start"])
    p0 = np.asarray(grasp["pos"], dtype=np.float64)
    r0 = rc.quat_to_matrix(np.asarray(grasp["quat_xyzw"], dtype=np.float64))
    out = {}
    for key in ("butt_local", "collar_local", "tip_local", "com_local"):
        local = np.asarray(obj[key], dtype=np.float64)
        out[key[: -len("_local")]] = hr[:3] + rhr @ (rhc.T @ (p0 + r0 @ local - hc[:3]))
    out["grasp"] = hr[:3] + rhr @ (rhc.T @ (p0 - hc[:3]))
    return out


def lowest_point(fk, episode: dict, scene: dict, row: int) -> float:
    """Lowest point of the rigidly carried handle and tip at ``row`` above the table [m]."""
    obj = scene["objects"]["screwdriver"]
    pts = carried_points(fk, episode, scene, row)
    xs = np.asarray(obj["geometry"]["handle_profile"])
    axis = pts["tip"] - pts["butt"]
    axis /= np.linalg.norm(axis)
    lows = []
    for x, r in xs:
        centre = pts["grasp"] + axis * x
        lows.append(centre[2] - r * math.sqrt(max(0.0, 1.0 - axis[2] ** 2)))
    lows.append(pts["tip"][2] - SHAFT_RADIUS)
    return float(min(lows) - TABLE_Z)


def bin_margin(spec: dict, point: np.ndarray, height: float) -> float:
    """Distance of a point (xy) inside the bin's inner wall at ``height`` above the table [m]; negative outside."""
    origin, rotation = rc.bin_frame(spec, TABLE_Z)
    local = (np.asarray(point, dtype=np.float64) - origin) @ rotation
    return -float(rc.polygon_signed_distance(local[:2], rc.bin_inner_polygon(spec, height)))


def carry_butt_offset(raw: Path, name: str, episode: dict, scene: dict, fk, camera: Camera):
    """Along-axis offset [m] of the real handle butt (yellow cap, top view) from the rigidly carried model butt during
    the carry (state rows lift-off + 3 .. release - 6), median over frames; negative: the real butt is closer to the
    grasp. None if fewer than 3 frames show the cap."""
    ev = scene["objects"]["screwdriver"]["event_rows"]
    top_rows = rc.frame_rows(episode)
    values = []
    for f, path in enumerate(_frames(raw, name)):
        if not (ev["liftoff"] + 3 <= top_rows[f] <= ev["release"] - 6):
            continue
        pts = carried_points(fk, episode, scene, int(top_rows[f]))
        ub, ut = camera.project(pts["butt"]), camera.project(pts["tip"])
        axis = (ub - ut) / np.linalg.norm(ub - ut)
        step = (pts["butt"] - pts["tip"]) / np.linalg.norm(pts["butt"] - pts["tip"]) * 0.01
        px_per_m = float(np.linalg.norm(camera.project(pts["butt"] + step) - ub)) / 0.01
        image = _load_image(path)
        u0, v0 = int(ub[0]) - 50, int(ub[1]) - 50
        if u0 < 0 or v0 < 0 or u0 + 100 > image.shape[1] or v0 + 100 > image.shape[0]:
            continue
        vv, uu = np.nonzero(yellow_mask(image[v0 : v0 + 100, u0 : u0 + 100]))
        rel = np.stack([uu + u0 + 0.5, vv + v0 + 0.5], 1) - ub
        along, perp = rel @ axis, rel @ np.array([-axis[1], axis[0]])
        keep = (np.abs(perp) <= 14) & (np.abs(along) <= 45)
        if keep.sum() >= 15:
            values.append(float(np.percentile(along[keep], 97)) / px_per_m)
    return (float(np.median(values)), len(values)) if len(values) >= 3 else (None, len(values))


def build_scene(raw: Path, name: str, label: str, fk, world: World, calibration: dict, masks: RobotMasks) -> tuple:
    """Scene, ground truth, episode, and checks of one episode (main MCAP or a 10 fps copy)."""
    episode = load_any_episode(raw, name)
    top_rows = rc.frame_rows(episode)
    holds = find_holds(episode)
    checks = {"holds": len(holds), "hold_arms": [h["arm"] for h in holds]}
    hold = next(h for h in holds if h["arm"] == "right")
    rows = dict(hold["rows"])
    cache = FKCache(fk, episode)
    mount = calibration["wrist_mount"]
    camera_right = {"body_rotation_xyzw": mount["body_rotation_xyzw"], "body_offset_m": mount["body_offset_m"]}
    curve = collar_curve(raw, name, episode, cache, rows, camera_right)
    track = collar_track(raw, name, episode, rows)
    grasp_cal = calibration["grasp"]
    common = np.asarray(grasp_cal["collar_model"]["common_v_norm"])
    if track.get("v_median") is None:
        raise RuntimeError(f"{name}: no collar in the wrist view")
    d = d_from_collar(
        track["v_median"], curve["intrinsics"], common, grasp_cal["bias_px"], grasp_cal["collar_model"]["main_fy"]
    )
    geo = screwdriver_geometry(d, handle_profile_s(calibration))
    gap = 2.0 * float(np.clip(hold["g_hold"], 0, 1)) * rc.GRIPPER_TRAVEL

    first = _load_image(_frames(raw, name)[0])
    dims = calibration["bin"]
    p, iou = fit_bin(
        world.camera,
        pink_mask_top(first),
        height=dims["height_m"],
        dims=(dims["rim_length_m"], dims["rim_width_m"], dims["bottom_scale"]),
    )
    p_last, _ = fit_bin(
        world.camera,
        pink_mask_top(_load_image(_frames(raw, name)[-1])),
        height=dims["height_m"],
        dims=(dims["rim_length_m"], dims["rim_width_m"], dims["bottom_scale"]),
        init=p,
    )
    tip_dir = [math.cos(curve["tip_yaw"]), math.sin(curve["tip_yaw"])]
    obj = {
        "arm": "right",
        "shape": "screwdriver",
        **{k: geo[k] for k in ("grasp_height_m", "grasp_radius_m", "rest_pitch_rad")},
        "axis_hint_xy": _round(tip_dir, 4),
        "geometry": geo["geometry"],
        **{k: geo[k] for k in ("butt_local", "collar_local", "tip_local", "com_local")},
        "grip_gap_m": round(gap, 4),
        "wrist_collar_v_px": round(track["v_median"], 1),
    }
    is_main = name == "main"
    scene = {
        "name": label,
        "episode": f"episodes/{label}.npz",
        "uuid": MAIN_UUID if is_main else lerobot_meta(raw, name)["uuid"],
        "source": "ABC-130k val, Voxel51 MCAP mirror (about 30 Hz)"
        if is_main
        else "ABC-130k LeRobot 10 fps copy, interpolated to 30 Hz",
        "frame_rate": 30.0 if is_main else 10.0,
        "table_z": TABLE_Z,
        "bases": copy.deepcopy(world.bases),
        "bin": bin_entry(p, iou, last_center_xy=_round(p_last[:2], 5), last_yaw_deg=round(math.degrees(p_last[2]), 3)),
        "objects": {"screwdriver": obj},
        "time_base": TIME_BASE,
    }
    # Events: holds from the gripper signals, lift-off from the start (the start averages up to lift-off).
    obj["events"] = rows_to_frames({**rows, "liftoff": rows["contact"] + 2}, top_rows)
    for _ in range(3):
        start = fk.object_starts(episode, scene)["screwdriver"]
        rows["liftoff"] = find_liftoff(fk, episode, rows, "right", np.asarray(start["pos"]))
        obj["events"] = rows_to_frames(rows, top_rows)
    obj["events"] = {k: obj["events"][k] for k in ("cmd_close", "contact", "liftoff", "cmd_open", "release")}
    obj["event_rows"] = {k: int(rows[k]) for k in ("cmd_close", "contact", "liftoff", "cmd_open", "release")}
    obj["open_cmd_row"] = rc.open_command_row(episode, obj)
    start = fk.object_starts(episode, scene)["screwdriver"]
    obj["start"] = {"pos": _round(start["pos"]), "quat_xyzw": _round(start["quat_xyzw"], 6)}
    obj["start_yaw_deg"] = round(start["yaw_deg"], 2)

    # Ground truth.
    tracks = fk.attached_tracks(episode, scene)
    count = len(top_rows)
    src = np.zeros(count, np.int8)
    src[: obj["events"]["contact"]] = 4
    src[obj["events"]["contact"] : obj["events"]["release"]] = 2
    gt = {
        "screwdriver_pos": tracks["screwdriver"],
        "screwdriver_src": src,
        "screwdriver_rest_xyz": np.asarray(obj["start"]["pos"]),
        "t": episode["t_top"],
        "frame": np.arange(count),
        "top_state_index": top_rows,
    }
    final = final_pose(raw, name, episode, rows, masks, world.camera, scene["bin"], 25 if is_main else 12)
    if final is not None:
        gt["screwdriver_final_tip_xyz"] = np.asarray(final["tip"])
        gt["screwdriver_final_yaw_deg"] = np.float64(final["yaw_deg"])
        yaw = math.radians(final["yaw_deg"])
        gt["screwdriver_final_xyz"] = (
            np.asarray(final["tip"]) - np.array([math.cos(yaw), math.sin(yaw), 0.0]) * geo["tip_local"][0]
        )
    private_gt = {**gt, "screwdriver_wrist_collar_uv": track["uv"]}

    # Checks.
    release = carried_points(fk, episode, scene, rows["release"])
    floor = scene["bin"]["floor_m"]
    checks.update(
        {
            "rows": {k: int(v) for k, v in rows.items()},
            "gap_mm": round(1000 * gap, 1),
            "profile_radius_at_grasp_mm": round(1000 * geo["grasp_radius_m"], 2),
            "gap_minus_profile_diameter_mm": round(1000 * (gap - 2 * geo["grasp_radius_m"]), 2),
            "grasp_to_collar_mm": round(1000 * d, 1),
            "wrist_collar_v_px": round(track["v_median"], 1),
            "wrist_collar_v_range_px": None if track.get("v_range") is None else round(track["v_range"], 1),
            "wrist_hold_frames": len(track["hold_frames"]),
            "bin_top_iou": round(iou, 4),
            "bin_moved_first_to_last_mm": round(1000 * float(np.linalg.norm(p_last[:2] - p[:2])), 1),
            "release_margin_mm": {
                k: round(1000 * bin_margin(scene["bin"], release[k], floor + 0.003), 1) for k in ("butt", "tip", "com")
            },
            "release_lowest_point_mm": round(1000 * lowest_point(fk, episode, scene, rows["release"]), 1),
            "start_yaw_deg": obj["start_yaw_deg"],
        }
    )
    margins = checks["release_margin_mm"]
    offset, frames = carry_butt_offset(raw, name, episode, scene, fk, world.camera)
    checks["carry_butt_offset_mm"] = None if offset is None else round(1000 * offset, 1)
    checks["carry_butt_frames"] = frames
    observed = margins["butt"] - (0.0 if offset is None else 1000 * offset)
    checks["release_margin_observed_mm"] = {"butt": round(observed, 1), "tip": margins["tip"]}
    checks["rigid_carry_consistent_model"] = bool(min(margins["butt"], margins["tip"]) >= 3.0)
    # The rigid carry with the handle where the top view shows it during the carry (the model butt sits a median
    # 1 cm farther out than the real one, see the build report).
    checks["rigid_carry_consistent"] = bool(min(observed, margins["tip"]) >= 3.0)
    if final is not None:
        checks["final"] = {
            "frames": final["frames"],
            "tip_std_mm": round(final["tip_std_mm"], 1),
            "yaw_std_deg": round(final["yaw_std_deg"], 2),
            "tip_margin_mm": round(1000 * bin_margin(scene["bin"], final["tip"], floor + 0.003), 1),
            "tip_vs_rigid_release_mm": _round(1000 * (final["tip"][:2] - release["tip"][:2]), 1),
            "yaw_vs_rigid_release_deg": round(
                (final["yaw_deg"] - math.degrees(math.atan2(*(release["tip"] - release["butt"])[1::-1])) + 180) % 360
                - 180,
                1,
            ),
        }
    else:
        checks["final"] = None
    # The ground truth and the checks above use the grasp pose; the replay starts before the jaws push the handle.
    pre_push_start(obj)
    gt["screwdriver_rest_xyz"] = private_gt["screwdriver_rest_xyz"] = np.asarray(obj["start"]["pos"])
    return scene, gt, private_gt, episode, checks, {"track": track, "curve": curve, "final": final, "d": d}


# ----------------------------------------------------------------------------- build: review sheets


def _draw_polygon(draw, uv, colour):
    draw.line([tuple(p) for p in np.vstack([uv, uv[:1]])], fill=colour, width=1)


def review_sheet(
    path: Path,
    raw: Path,
    name: str,
    fk,
    world: World,
    calibration: dict,
    scene: dict,
    episode: dict,
    checks: dict,
    extra: dict,
):
    """One row per episode: start (zoomed, the scene's screwdriver: butt yellow, grasp red, collar blue, tip cyan),
    first frame with the bin model (rim cyan, inner wall at the floor magenta), the release frame with the rigidly
    carried screwdriver, the last frame with the final detection, and the right wrist view at contact and mid-hold
    with the projected axis and collar (real collar track: white)."""
    from PIL import Image, ImageDraw

    camera = world.camera
    obj = scene["objects"]["screwdriver"]
    spec = scene["bin"]
    top_rows = rc.frame_rows(episode)
    files = _frames(raw, name)
    rim = camera.project(np.c_[rc.bin_rim_polygon(spec), np.full(4, TABLE_Z + spec["height_m"])])
    inner = camera.project(
        np.c_[rc.bin_inner_polygon(spec, spec["floor_m"], world=True), np.full(4, TABLE_Z + spec["floor_m"])]
    )
    start_r = rc.quat_to_matrix(np.asarray(obj["start"]["quat_xyzw"]))
    start_p = np.asarray(obj["start"]["pos"])
    points = {k: start_p + start_r @ np.asarray(obj[f"{k}_local"]) for k in ("butt", "collar", "tip")}
    points["grasp"] = start_p
    colours = {"butt": (255, 255, 0), "grasp": (255, 0, 0), "collar": (0, 128, 255), "tip": (0, 255, 255)}

    def mark(draw, uv, scale=1.0, offset=(0, 0)):
        p = {k: (np.asarray(v) - offset) * scale for k, v in uv.items()}
        draw.line([tuple(p["butt"]), tuple(p["tip"])], fill=(255, 0, 0), width=1)
        for k, c in colours.items():
            u, v = p[k]
            draw.ellipse([u - 3, v - 3, u + 3, v + 3], outline=c, width=1)

    tiles = []
    image = Image.open(files[0]).convert("RGB")
    uv = {k: camera.project(v) for k, v in points.items()}
    centre = uv["grasp"]
    box = (int(centre[0]) - 90, int(centre[1]) - 70, int(centre[0]) + 90, int(centre[1]) + 70)
    crop = image.crop(box).resize((540, 420), Image.LANCZOS)
    draw = ImageDraw.Draw(crop)
    mark(draw, uv, 3.0, box[:2])
    draw.text(
        (4, 4),
        f"{scene['name']} {name} start: gap {checks['gap_mm']} mm, collar {checks['grasp_to_collar_mm']} mm",
        fill=(255, 0, 0),
    )
    tiles.append(crop)
    for f, label in (
        (0, "first"),
        (int(np.argmin(np.abs(top_rows - checks["rows"]["release"]))), "release"),
        (len(files) - 1, "last"),
    ):
        image = Image.open(files[f]).convert("RGB")
        draw = ImageDraw.Draw(image)
        _draw_polygon(draw, rim, (0, 255, 255))
        _draw_polygon(draw, inner, (255, 0, 255))
        if label == "release":
            carried = carried_points(fk, episode, scene, checks["rows"]["release"])
            mark(draw, {k: camera.project(carried[k]) for k in colours})
        if label == "last" and extra["final"] is not None:
            u, v = camera.project(extra["final"]["tip"])
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=(0, 255, 0), width=2)
            u, v = camera.project(extra["final"]["handle"])
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=(255, 255, 0), width=2)
        bc_ = camera.project(np.array([*spec["center_xy"], TABLE_Z + spec["height_m"]]))
        crop_box = (int(bc_[0]) - 200, int(bc_[1]) - 150, int(bc_[0]) + 200, int(bc_[1]) + 150)
        crop = image.crop(crop_box).resize((560, 420), Image.LANCZOS)
        ImageDraw.Draw(crop).text(
            (4, 4),
            f"{label} f{f} margins {checks['release_margin_mm'] if label == 'release' else ''}",
            fill=(255, 0, 0),
        )
        tiles.append(crop)
    wfiles = _frames(raw, name, "right_wrist")
    wrows = rc.frame_rows(episode, "right")
    mount = calibration["wrist_mount"]
    intr = extra["curve"]["intrinsics"]
    hold = extra["track"]["hold_frames"]
    contact_f = int(np.argmin(np.abs(wrows - checks["rows"]["contact"])))
    for f in (contact_f, hold[len(hold) // 2] if hold else contact_f):
        if f >= len(wfiles):
            continue
        image = Image.open(wfiles[f]).convert("RGB")
        draw = ImageDraw.Draw(image)
        cache = FKCache(fk, episode)
        pose = mount_pose(cache.bodies(int(wrows[f]))[1], mount)
        carried = carried_points(fk, episode, scene, int(wrows[f]))
        mark(draw, {k: rc.project_points(intr, carried[k], pose) for k in colours})
        real = extra["track"]["uv"][f]
        if np.isfinite(real[0]):
            draw.ellipse([real[0] - 6, real[1] - 6, real[0] + 6, real[1] + 6], outline=(255, 255, 255), width=2)
        draw.text((4, 4), f"wrist f{f} row {wrows[f]} collar v {checks['wrist_collar_v_px']}", fill=(255, 0, 0))
        tiles.append(image.resize((560, 317)))
    height = 420
    sheet = Image.new("RGB", (sum(t.size[0] for t in tiles), height))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.size[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def tip_grid(path: Path, raw: Path, items: list[tuple[str, str, World, dict]]):
    """First-frame crops of many episodes with the scene's start screwdriver (butt yellow, tip cyan) for checking
    the tip direction by hand."""
    from PIL import Image, ImageDraw

    tiles = []
    for name, label, world, scene in items:
        obj = scene["objects"]["screwdriver"]
        r = rc.quat_to_matrix(np.asarray(obj["start"]["quat_xyzw"]))
        p = np.asarray(obj["start"]["pos"])
        uv = {k: world.camera.project(p + r @ np.asarray(obj[f"{k}_local"])) for k in ("butt", "tip")}
        uv["grasp"] = world.camera.project(p)
        image = Image.open(_frames(raw, name)[0]).convert("RGB")
        c = uv["grasp"]
        box = (int(c[0]) - 70, int(c[1]) - 55, int(c[0]) + 70, int(c[1]) + 55)
        crop = image.crop(box).resize((420, 330), Image.LANCZOS)
        draw = ImageDraw.Draw(crop)
        q = {k: (v - box[:2]) * 3 for k, v in uv.items()}
        draw.line([tuple(q["butt"]), tuple(q["tip"])], fill=(255, 0, 0), width=1)
        for k, col in (("butt", (255, 255, 0)), ("tip", (0, 255, 255)), ("grasp", (255, 0, 0))):
            u, v = q[k]
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=col, width=2)
        draw.text((4, 4), f"{label} {name}", fill=(255, 0, 0))
        tiles.append(crop)
    columns = 4
    sheet = Image.new("RGB", (columns * 420, ((len(tiles) + columns - 1) // columns) * 330))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % columns) * 420, (i // columns) * 330))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


# ----------------------------------------------------------------------------- build


def camera_json(raw: Path, world: World, calibration: dict) -> dict:
    mcap = json.loads((raw / "cameras.json").read_text())
    mount = calibration["wrist_mount"]
    fit_ = calibration["camera_fit"]
    return {
        "top": {
            **{k: mcap["top"][k] for k in ("width", "height", "distortion_model", "K", "D")},
            "position": _round(world.camera.c, 6),
            "rotation_xyzw": _round(world.camera.rotation_xyzw, 7),
        },
        "left": {**mcap["left"], "body": rc.CAMERA_BODY["left"], "body_rotation_xyzw": list(rc.CAMERA_IN_BODY_XYZW)},
        "right": {
            **mcap["right"],
            "body": rc.CAMERA_BODY["right"],
            "body_rotation_xyzw": mount["body_rotation_xyzw"],
            "body_offset_m": mount["body_offset_m"],
        },
        "time_base": "top frame i shows arm-state sample i-1 (episode left_t; 10 fps episodes: <camera>_state_index); "
        "wrist frames are assumed to follow the same rule",
        "convention": "position [m] and rotation_xyzw of the camera in the world; it looks along its -Z axis, +Y up",
        "top_fit": {
            "episode": MAIN,
            "method": "edge NCC of robot renders against 20 top frames, jointly with the right arm base (x, yaw) and "
            "the kinematic grasp point vs the image screwdriver on clean starts",
            "edge_ncc": round(fit_["edge_ncc"], 4),
            "grasp_vs_image_axis_median_mm": round(fit_["perp_median_mm"], 1),
            "note": "fitted on the main episode; the 10 fps episodes use the same pose (same station, camera unmoved)",
        },
        "right_mount_fit": {
            "method": "gripper silhouette in the right wrist frames before the grasp",
            "rotation_deg": mount["rotation_deg"],
            "offset_mm": mount["offset_mm"],
            "gripper_iou_mjcf_mount": round(mount["gripper_iou_mjcf"], 3),
            "gripper_iou_fitted_mount": round(mount["gripper_iou_fitted"], 3),
        },
    }


def _copy_frames(raw: Path, name: str, task: Path, label: str, cameras=("top", "right_wrist")) -> int:
    count = 0
    for camera in cameras:
        source = raw / "frames" / name / camera
        if source.exists():
            shutil.copytree(source, task / "frames" / label / camera, dirs_exist_ok=True)
            count += len(list(source.glob("*.jpg")))
    return count


def base_off_calibration(calibration: dict) -> dict:
    """The calibration without the base correction: A-data's original top-camera fit (right base offset only)."""
    out = copy.deepcopy(calibration)
    out["top"] = {
        **calibration["top"],
        "position": [0.06146875, 0.01046875, 1.7043125],
        "rotation_xyzw": [-0.19193903522771988, 0.19425660854448518, 0.6852092316348083, -0.6752126225679452],
    }
    out["right_base_xy"] = [0.2480875, -0.3094875]
    out["right_base_yaw"] = 0.0
    return out


def left_anchor(raw: Path, station: Path) -> list[float]:
    """Left gripper grasp point at the main episode's first sample, with the MJCF bases (the fitted frame)."""
    fk = rc.StationFK(station, MJCF_BASES)
    episode = load_any_episode(raw, "main")
    return fk.grasp_point(fk.pose_row(episode, 0), "left").tolist()


def build(args: argparse.Namespace) -> None:
    if rc is None:
        raise ImportError("the build stage needs Newton (replay_common)")
    raw, private, station = args.raw, args.private, args.station
    spec = json.loads(args.spec.read_text())
    calibration = json.loads(args.calibration.read_text())
    _write_json(private / "calibration.json", calibration)
    sheets = args.sheets
    names = ["main", *sorted(spec["pool"])] if not args.names else args.names.split(",")
    variants = [("", calibration)] + ([("_base_off", base_off_calibration(calibration))] if args.variants else [])
    all_checks = {}
    anchor = left_anchor(raw, station)
    for tag, cal in variants:
        world = World(cal, anchor)
        fk = rc.StationFK(station, world.bases)
        masks = RobotMasks(station, world.bases, world.camera)
        root = private / f"candidates{tag}"
        grid = []
        t0 = time.perf_counter()
        for name in names:
            label = "main" if name == "main" else name
            scene, _, private_gt, episode, checks, extra = build_scene(raw, name, label, fk, world, cal, masks)
            scene["episode"] = f"episodes/{label}.npz"
            _write_json(root / "scenes" / f"{label}.json", scene)
            _write_npz(root / "episodes" / f"{label}.npz", episode)
            _write_npz(root / "gt" / f"{label}.npz", private_gt, compressed=True)
            if not tag:
                all_checks[label] = checks
                grid.append((name, label, world, scene))
                if sheets is not None:
                    review_sheet(sheets / f"{label}.jpg", raw, name, fk, world, cal, scene, episode, checks, extra)
            print(
                f"{tag or 'nominal'} {label}: collar {checks['grasp_to_collar_mm']} mm, bin IoU {checks['bin_top_iou']}, "
                f"release margins {checks['release_margin_mm']}, final {None if checks['final'] is None else checks['final']['tip_vs_rigid_release_mm']} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        if not tag:
            _write_json(root / "checks.json", all_checks)
            if sheets is not None:
                for k in range(0, len(grid), 16):
                    tip_grid(sheets / f"tips_{k // 16}.jpg", raw, grid[k : k + 16])
    if args.names:
        return
    write_outputs(args, spec, calibration, all_checks)


def write_outputs(args: argparse.Namespace, spec: dict, calibration: dict, all_checks: dict) -> None:
    """Agent dataset, the verifier's copies, and the held-out set (plus spares) from the candidate files."""
    raw, task, private, station = args.raw, args.task, args.private, args.station
    world = World(calibration, left_anchor(raw, station))
    candidates = private / "candidates"
    quality_path = private / "heldout" / "quality.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    for destination in (task / "station", private / "station"):
        shutil.copytree(station.parent, destination, dirs_exist_ok=True)
    if args.arm_logs is not None:
        shutil.copytree(args.arm_logs, task / "arm_logs", dirs_exist_ok=True)
    _write_json(task / "camera.json", camera_json(raw, world, calibration))
    checks = {**SIBLING_CHECKS, **spec.get("manual_checks", {})}
    agent = {"main": "main", **SIBLINGS}
    for label, name in agent.items():
        scene = json.loads((candidates / "scenes" / f"{name}.json").read_text())
        scene["name"] = label
        scene["episode"] = f"episodes/{label}.npz"
        scene["notes"] = list(checks.get(label, {}).get("notes", []))
        _write_json(task / "scenes" / f"{label}.json", scene)
        (task / "episodes").mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates / "episodes" / f"{name}.npz", task / "episodes" / f"{label}.npz")
        gt = dict(np.load(candidates / "gt" / f"{name}.npz"))
        _write_npz(private / "gt" / f"{label}.npz", gt, compressed=True)
        _write_npz(
            task / "gt" / f"{label}.npz",
            {k: v for k, v in gt.items() if not k.endswith("wrist_collar_uv")},
            compressed=True,
        )
        _copy_frames(raw, name, task, label)
    # The verifier replays its own copies of the agent-facing episodes, scenes, and camera.
    for folder in ("episodes", "scenes"):
        shutil.copytree(task / folder, private / folder, dirs_exist_ok=True)
    shutil.copy2(task / "camera.json", private / "camera.json")
    verdicts = quality.get("verdicts", {})
    for group, folder in (("heldout", "heldout"), ("spare", "heldout_spare")):
        for uuid in spec.get(group, []):
            scene = json.loads((candidates / "scenes" / f"{uuid}.json").read_text())
            scene["episode"] = f"{folder}/episodes/{uuid}.npz"
            check = checks.get(uuid, {})
            scene["notes"] = list(check.get("notes", []))
            # Unreliable if the review by hand or the quality study (reference verdict flips) says so.
            found = (check.get("verdict"), verdicts.get(uuid))
            scene["verdict"] = "unreliable" if "unreliable" in found else (found[0] or found[1] or "unchecked")
            # The right wrist camera of the 08-27 recordings has other intrinsics than camera.json (report-only metric).
            scene["wrist_intrinsics_right"] = wrist_intrinsics(raw, uuid)
            _write_json(private / folder / "scenes" / f"{uuid}.json", scene)
            for kind in ("episodes", "gt"):
                (private / folder / kind).mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidates / kind / f"{uuid}.npz", private / folder / kind / f"{uuid}.npz")
    print(
        f"wrote {task} ({len(agent)} agent episodes) and {private} ({len(spec.get('heldout', []))} held out, "
        f"{len(spec.get('spare', []))} spare)"
    )


# ----------------------------------------------------------------------------- quality


def _load_reference(path: Path, workspace: Path, station_dir: Path):
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "replay_common.py", workspace / "replay_common.py")
    shutil.copy2(path, workspace / "reference_replay.py")
    if (workspace / "station").is_symlink() or (workspace / "station").exists():
        (workspace / "station").unlink()
    (workspace / "station").symlink_to(station_dir)
    sys.path.insert(0, str(workspace))
    spec = importlib.util.spec_from_file_location("abc_bin_reference", workspace / "reference_replay.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _turn_start(scene: dict, degrees: float) -> dict:
    scene = copy.deepcopy(scene)
    start = scene["objects"]["screwdriver"]["start"]
    a = math.radians(degrees)
    start["quat_xyzw"] = rc._quat_multiply(
        np.array([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]), np.asarray(start["quat_xyzw"])
    ).tolist()
    return scene


def _approach(gt: dict, scene: dict) -> np.ndarray:
    events = scene["objects"]["screwdriver"]["events"]
    pos = gt["screwdriver_pos"]
    a, b = pos[events["contact"]], pos[max(events["release"] - 1, events["contact"])]
    d = (b - a)[:2]
    return d / max(np.linalg.norm(d), 1e-9)


def quality(args: argparse.Namespace) -> None:
    """Replay the reference on candidate episodes under the construction variants; verdict stability."""
    private = args.private
    calibration = json.loads((private / "calibration.json").read_text())
    root = private / "candidates"
    names = args.names.split(",") if args.names else sorted(p.stem for p in (root / "scenes").glob("*.json"))
    reference = _load_reference(args.reference, args.workspace or private / "_quality_ws", args.station.parent)
    rcm = sys.modules["replay_common"]
    scenes = {n: rcm.load_scene(root / "scenes" / f"{n}.json") for n in names}
    episodes = {n: rcm.load_episode(root / "episodes" / f"{n}.npz") for n in names}
    gts = {n: rcm.load_gt(root / "gt" / f"{n}.npz") for n in names}
    station = args.station
    fks = {}

    def restart(scene, episode):
        key = json.dumps(scene["bases"])
        if key not in fks:
            fks[key] = rc.StationFK(station, scene["bases"])
        start = fks[key].object_starts(episode, scene)["screwdriver"]
        obj = scene["objects"]["screwdriver"]
        obj.pop("grasp_start", None)
        obj["start"] = {"pos": _round(start["pos"]), "quat_xyzw": _round(start["quat_xyzw"], 6)}
        pre_push_start(obj)
        return scene

    def cylinder(scene, episode):
        scene = copy.deepcopy(scene)
        obj = scene["objects"]["screwdriver"]
        geo = screwdriver_geometry(
            float(obj["collar_local"][0]), handle_profile_s(calibration), cylinder_radius=0.5 * obj["grip_gap_m"]
        )
        obj.update(geo)
        return restart(scene, episode)

    def with_bin(scene, **values):
        scene = copy.deepcopy(scene)
        scene["bin"].update(values)
        return scene

    variants = {
        "nominal": (lambda n: scenes[n], {}),
        "repeat": (lambda n: scenes[n], {}),
        "dt1": (lambda n: scenes[n], {"dt": 0.001}),
        "floor1": (lambda n: with_bin(scenes[n], floor_m=0.001), {}),
        "floor8": (lambda n: with_bin(scenes[n], floor_m=0.008), {}),
        "cylinder": (lambda n: cylinder(scenes[n], episodes[n]), {}),
        "bin_plus": (
            lambda n: with_bin(
                scenes[n],
                center_xy=(np.asarray(scenes[n]["bin"]["center_xy"]) + 0.01 * _approach(gts[n], scenes[n])).tolist(),
            ),
            {},
        ),
        "bin_minus": (
            lambda n: with_bin(
                scenes[n],
                center_xy=(np.asarray(scenes[n]["bin"]["center_xy"]) - 0.01 * _approach(gts[n], scenes[n])).tolist(),
            ),
            {},
        ),
        "yaw_plus": (lambda n: _turn_start(scenes[n], 5.0), {}),
        "yaw_minus": (lambda n: _turn_start(scenes[n], -5.0), {}),
    }
    off_root = private / "candidates_base_off"
    if off_root.exists():
        variants["base_off"] = (lambda n: rcm.load_scene(off_root / "scenes" / f"{n}.json"), {"gt_root": off_root})
    if args.cylinder_base:
        # The same study with the per-episode cylinder (radius = held gap / 2) as the nominal handle.
        tapered = dict(scenes)
        scenes.update({n: cylinder(scenes[n], episodes[n]) for n in names})
        variants["cylinder"] = (lambda n: tapered[n], {})
        if "base_off" in variants:
            variants["base_off"] = (
                lambda n: cylinder(rcm.load_scene(off_root / "scenes" / f"{n}.json"), episodes[n]),
                {"gt_root": off_root},
            )
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants.split(",")}

    def run(scene_list, episode_list, gt_list, params, batch=80):
        dt = params.get("dt", reference.PARAMS["dt"])
        tic = time.perf_counter()
        out = []
        for k in range(0, len(scene_list), batch):
            scenes_k, episodes_k, gts_k = scene_list[k : k + batch], episode_list[k : k + batch], gt_list[k : k + batch]
            model = reference.build_model(scenes_k)
            solver = reference.make_solver(model)
            pipeline = reference.make_pipeline(model)
            replay = rcm.Replay(
                model, solver, pipeline, episodes_k, dt=dt, command_delay=reference.PARAMS["command_delay"]
            )
            replay.run()
            out += [rcm.score(r, s, g) for r, s, g in zip(replay.recordings(), scenes_k, gts_k, strict=True)]
            del replay, solver, pipeline, model
        return out, time.perf_counter() - tic

    def brief(result):
        m = result["objects"]["screwdriver"]
        ok = bool(
            m["held"]
            and m["placed"]
            and m["inhand_rot_deg"] is not None
            and m["inhand_rot_deg"] <= rcm.HELDOUT_ROT_MAX_DEG
        )
        keys = (
            "held_fraction",
            "placed",
            "in_bin",
            "inhand_rot_deg",
            "slip_m",
            "grip_gap_err_mm",
            "final_tip_xy_err_m",
            "tip_wall_m",
            "com_wall_m",
            "final_speed_mps",
            "bin_moved_m",
            "liftoff_err_rows",
            "carry_track_err_m",
        )
        return {
            "ok": ok,
            **{k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in keys},
            "arm_rmse_right": round(result["arm_rmse_rad"]["right"], 4),
        }

    results, timing = {}, {}
    for vname, (make, params) in variants.items():
        gt_root = params.get("gt_root")
        scene_list = [make(n) for n in names]
        gt_list = [rcm.load_gt(gt_root / "gt" / f"{n}.npz") if gt_root else gts[n] for n in names]
        out, wall = run(scene_list, [episodes[n] for n in names], gt_list, params)
        results[vname] = {n: brief(r) for n, r in zip(names, out, strict=True)}
        timing[vname] = round(wall, 1)
        print(f"{vname}: {sum(v['ok'] for v in results[vname].values())}/{len(names)} ok ({wall:.0f}s)", flush=True)
    # Verifier-style held-out group: 1 nominal + 3 jittered copies.
    rng = np.random.default_rng(14)
    ens_names = [n for n in names for _ in range(4)]
    ens_scenes = [scenes[n] if k % 4 == 0 else rcm.jitter_scene(scenes[n], rng) for k, n in enumerate(ens_names)]
    out, wall = run(ens_scenes, [episodes[n] for n in ens_names], [gts[n] for n in ens_names], {})
    ensemble = {}
    for n, r in zip(ens_names, out, strict=True):
        ensemble.setdefault(n, []).append(brief(r))
    timing["ensemble"] = round(wall, 1)

    report = {
        "reference": str(args.reference),
        "variants": list(variants),
        "timing_s": timing,
        "episodes": {},
        "verdicts": {},
    }
    build_checks = json.loads((root / "checks.json").read_text()) if (root / "checks.json").exists() else {}
    for n in names:
        base = results["nominal"][n]["ok"]
        flips = [v for v in variants if v != "nominal" and results[v][n]["ok"] != base]
        copies = ensemble[n]
        entry = {
            "nominal": results["nominal"][n],
            "variant_ok": {v: results[v][n]["ok"] for v in variants},
            "variants": {
                v: {k: results[v][n][k] for k in ("ok", "placed", "inhand_rot_deg", "held_fraction", "bin_moved_m")}
                for v in variants
            },
            "flips": flips,
            "ensemble_ok": sum(c["ok"] for c in copies),
            "ensemble_rot_deg": [c["inhand_rot_deg"] for c in copies],
            "ensemble_placed": [c["placed"] for c in copies],
            "checks": build_checks.get(n),
        }
        reasons = []
        if flips:
            reasons.append("verdict flips under " + ", ".join(flips))
        if not base:
            reasons.append("the reference fails the nominal copy")
        if results["nominal"][n]["bin_moved_m"] is not None and results["nominal"][n]["bin_moved_m"] > 0.03:
            reasons.append(f"the bin moves {results['nominal'][n]['bin_moved_m']:.2f} m in the nominal replay")
        if build_checks.get(n) and not build_checks[n].get("rigid_carry_consistent", True):
            reasons.append("rigid carry ends outside the inner bin wall (3 mm margin)")
        entry["reasons"] = reasons
        verdict = "usable" if not reasons else "unreliable"
        report["episodes"][n] = entry
        report["verdicts"][n] = verdict
    usable = [n for n, v in report["verdicts"].items() if v == "usable"]
    report["summary"] = {
        "episodes": len(names),
        "usable": len(usable),
        "nominal_ok": sum(results["nominal"][n]["ok"] for n in names),
        "flip_counts": {
            v: sum(results[v][n]["ok"] != results["nominal"][n]["ok"] for n in names)
            for v in variants
            if v != "nominal"
        },
        "rigid_carry_inconsistent": sum(
            1 for n in names if build_checks.get(n) and not build_checks[n].get("rigid_carry_consistent", True)
        ),
        "rigid_carry_inconsistent_model": sum(
            1 for n in names if build_checks.get(n) and not build_checks[n].get("rigid_carry_consistent_model", True)
        ),
        "ensemble_ok_rate_usable": (sum(report["episodes"][n]["ensemble_ok"] for n in usable) / (4 * len(usable)))
        if usable
        else None,
    }
    path = args.out or private / "heldout" / "quality_candidates.json"
    _write_json(path, report)
    print(json.dumps(report["summary"], indent=1))


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("extract", help="decode the MCAP and LeRobot sources into a raw cache")
    p.add_argument("--mcap", type=Path, required=True, help="MCAP file of the main episode (Voxel51 mirror)")
    p.add_argument("--lerobot", type=Path, nargs="+", required=True, help="LeRobot 480x848 subsets of the task")
    p.add_argument("--raw", type=Path, required=True, help="raw cache directory")
    p.add_argument("--spec", type=Path, required=True, help="private spec (key 'pool': the episodes to extract)")
    p = sub.add_parser("fit", help="fit the station calibration")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--station", type=Path, required=True, help="station MJCF (yam_bimanual_empty.xml)")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="calibration JSON (reused parts are kept unless --refit-*)")
    p.add_argument("--refit-camera", action="store_true")
    p.add_argument("--refit-mount", action="store_true")
    p.add_argument("--camera-seed", type=Path, help="extra stage-2 start (an earlier fit); the best objective is kept")
    p = sub.add_parser("build", help="write the candidates, the task dataset, and the hidden held-out set")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--station", type=Path, required=True, help="station MJCF (its folder with assets is copied)")
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--task", type=Path, required=True, help="agent-facing dataset directory")
    p.add_argument("--private", type=Path, required=True, help="hidden directory")
    p.add_argument("--arm-logs", type=Path, help="abc_arm training logs to copy into arm_logs/")
    p.add_argument("--sheets", type=Path, help="directory for review sheets")
    p.add_argument("--names", default="", help="only build these candidates (no outputs)")
    p.add_argument("--variants", action="store_true", help="also build the base-correction-off candidates")
    p.add_argument("--outputs-only", action="store_true", help="only write the outputs from existing candidates")
    p = sub.add_parser("quality", help="verdict stability of the candidates under the construction variants")
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="reference solution (screwdriver_replay.py)")
    p.add_argument("--station", type=Path, required=True, help="station MJCF (yam_bimanual_empty.xml)")
    p.add_argument("--workspace", type=Path, help="scratch directory for the reference's imports")
    p.add_argument("--names", default="")
    p.add_argument("--variants", default="")
    p.add_argument("--cylinder-base", action="store_true", help="per-episode cylinder handles as the nominal")
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.stage == "extract":
        spec = json.loads(args.spec.read_text())
        extract_mcap(args.mcap, args.raw)
        extract_lerobot(args.lerobot, sorted({MAIN, *spec.get("pool", [])}), args.raw)
    elif args.stage == "fit":
        fit(args)
    elif args.stage == "build":
        if args.outputs_only:
            write_outputs(
                args,
                json.loads(args.spec.read_text()),
                json.loads(args.calibration.read_text()),
                json.loads((args.private / "candidates" / "checks.json").read_text()),
            )
        else:
            build(args)
    else:
        quality(args)


if __name__ == "__main__":
    main()
