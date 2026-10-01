# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the ABC fruit-bowl physical replay task files (``abc_replay``).

Sources (ABC-130k, https://abc.bot/#data, Apache-2.0):

- the training episode ``7067ee1f`` of "place and organize the fake fruits in the fruit bowl" from the
  Voxel51 MCAP mirror (top and wrist cameras, arm and gripper logs at about 30 Hz),
- ten episodes of the same station, fruits, and wedge tray from the LeRobot 10 fps mirror
  (``Dario-Shit2/ABC-130k``, ``train/place_and_organize_the_fake_fruits_in_the_fruit_bowl__480x640``):
  four siblings for the agent and six held out for the verifier; the LeRobot copy of the training
  episode (``val/...``) calibrates the colour model and checks the 10 fps pipeline,
- the video tracker output of the training episode (``ground_truth.json``, ``trajectories.npz``),
- the YAM station MJCF and meshes (``abc_sim/models``, as in the abc_twin task).

There are two stages because the decoders and Newton live in different environments:

``extract`` decodes the sources into a raw cache (needs mcap, mcap-protobuf-support, pyarrow, av, Pillow)::

    python prepare_data.py extract --mcap EPISODE.mcap --lerobot TRAIN_DIR --lerobot-val VAL_DIR \\
        [--lerobot-wrist WRIST_DIR] --heldout-spec SPEC.json --raw RAW_DIR

``build`` writes the agent-facing dataset and the hidden held-out set (needs Newton, SciPy, Pillow)::

    python prepare_data.py build --raw RAW_DIR --station STATION_DIR --tracker FRUITS_DIR \\
        --task TASK_DIR --private PRIVATE_DIR [--heldout-spec SPEC.json] [--arm-logs DIR] [--overlays DIR]

The held-out episode ids live only in the private spec (``PRIVATE/heldout_spec.json``: ``{"heldout": [8-character
uuid prefixes], "manual_checks": {...}}``), so this file does not name them.

The agent-facing dataset (``--task``) is the workspace layout of FORMAT.md. The hidden directory (``--private``,
``~/.newton-visual-private/abc_replay``) holds ``heldout/{episodes,scenes,gt}/<uuid>.{npz,json,npz}`` and
``heldout/quality.json`` (automatic and by-hand checks), the verifier's copies of ``station/``, ``episodes/``,
``scenes/``, ``gt/`` and ``camera.json``, ``thresholds.json`` (written by calibrate.py), and ``reference/``
(the reference solution). Held-out scenes carry the by-hand ``verdict`` of the episode and optional fruit
verdicts; the verifier does not gate on ``unreliable`` ones.

For every 10 fps episode the build derives the scene from the logs and the first and last top frames:
the hold timeline from the gripper signals, the fruit of each hold from the first-frame colour detections
nearest to the gripper, fruit starts from the pad midpoint at the grasp (:meth:`StationFK.fruit_starts`),
the tray sector pose from the first-frame tray silhouette, and the final fruit positions from the last
frame. ``--overlays`` writes images of all of these for checking by hand, and a quality table.
File formats are described in FORMAT.md.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np

try:
    import replay_common as rc
except ImportError:  # the extract stage runs in an environment without Newton
    rc = None

HERE = Path(__file__).resolve().parent
MAIN = "7067ee1f"
MAIN_UUID = "7067ee1f-371a-4956-bf25-25522babd0f7"
SIBLINGS = {"sib_1": "9167db86", "sib_2": "5a0847a4", "sib_3": "7f249316", "sib_4": "e840a86a"}
MCAP_CAMERAS = {"/top-camera": "top", "/left-wrist-camera": "left_wrist", "/right-wrist-camera": "right_wrist"}
LEROBOT_CAMERAS = {"top_camera": "top", "left_wrist_camera": "left_wrist", "right_wrist_camera": "right_wrist"}
JPEG_QUALITY = 90
# LeRobot timestamp + this = seconds from the first top-camera frame (state stream start minus top-camera
# start), used when the subset has no source_episodes.json (the val copy of the main episode).
DEFAULT_LEROBOT_OFFSET = 0.0968

TABLE_Z = 0.75
BASES = {"left": [0.2525, 0.31, 0.75], "right": [0.2546875, -0.300, 0.75]}  # right base y: video fit
TRAY_RIM = 0.018  # assumed rim height [m]
TRAY_FLOOR = 0.005  # assumed floor height above the table [m]
FRUIT_SIZES = {
    "pear": {
        "shape": "pear",
        "size_m": {"width": 0.051, "length": 0.081},
        "size_range_m": {"width": [0.044, 0.058], "length": [0.065, 0.092]},
        "centre_height_m": 0.0255,
    },
    "orange": {
        "shape": "sphere",
        "size_m": {"diameter": 0.058},
        "size_range_m": {"diameter": [0.052, 0.066]},
        "centre_height_m": 0.029,
    },
    "dark_fruit": {
        "shape": "sphere",
        "size_m": {"diameter": 0.069},
        "size_range_m": {"diameter": [0.060, 0.074]},
        "centre_height_m": 0.0345,
    },
}
MASS_RANGE_KG = [0.02, 0.15]
FRUIT_WIDTH = {"pear": 0.051, "orange": 0.058, "dark_fruit": 0.069}  # width across the grasp [m]
PAD_TIP = (0.0, -0.0024, 0.086)  # fingertip end of the pad box in the pad body frame (MJCF <side>_lf_down) [m]
TIME_BASE = "top frame i shows arm-state sample top_state_index[i] (default i - 1); see FORMAT.md"
# Results of checking every sibling by hand on the --overlays images (wrist view at each contact, top-view
# crops around each grasp with the FK pads, first-frame starts, tray fit, last-frame finals); fruit
# identities were confirmed in the wrist views for every hold. The held-out episodes and their checks are
# in the private --heldout-spec file. "verdict": usable, at_risk (the real grasp differs from what an
# FK-placed fruit at table height allows), or unreliable (do not gate on it); held-out checks may also give
# "fruit_verdicts" ({fruit: verdict}) for single fruits.
SIBLING_CHECKS = {
    "sib_1": {
        "verdict": "usable",
        "notes": [
            "right arm grasps the pear across its bulb (62.4 mm, wider than the nominal 51 mm waist)",
            "dark fruit held at 63.2 mm",
        ],
    },
    "sib_2": {
        "verdict": "at_risk",
        "notes": [
            "right arm grasps the pear by its neck (36.7 mm) and carries it hanging; the FK start centres the pear "
            "on the neck",
        ],
    },
    "sib_3": {
        "verdict": "usable",
        "notes": [
            "dark fruit held at 62.8 mm; its FK start is 5.4 cm (mostly +x) from the frame-0 image position",
            "the real dark fruit lands beside the orange in the tray; at the FK release point it lands on the orange",
        ],
    },
    "sib_4": {"verdict": "usable", "notes": []},
}


# ----------------------------------------------------------------------------- extract


def _save_jpeg(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=JPEG_QUALITY)


def extract_mcap(path: Path, raw: Path) -> None:
    """Decode the training episode's MCAP: arm and gripper logs, camera info, and all frames as JPEG."""
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


def extract_lerobot(root: Path, uuids: list[str], raw: Path, wrist_root: Path | None = None) -> None:
    """Copy the 10 fps logs of the given episodes (8-character uuid prefixes) and decode their frames."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
    metadata = pq.read_table(root / "meta" / "episode_metadata.parquet").to_pydict()
    source_path = root / "meta" / "source_episodes.json"
    sources = json.loads(source_path.read_text())["episodes"] if source_path.exists() else []
    tables = {}
    for uuid in uuids:
        row = next(i for i, s in enumerate(metadata["source_episode_id"]) if s[8:16] == uuid)
        index = metadata["episode_index"][row]
        e = episodes["episode_index"].index(index)
        source = next((s for s in sources if s["uuid"][8:16] == uuid), None)
        offset = DEFAULT_LEROBOT_OFFSET
        if source is not None:
            offset = (source["t0_ns"] - source["stream_span_ns"]["/top-camera"][0]) / 1e9
        key = (episodes["data/chunk_index"][e], episodes["data/file_index"][e])
        if key not in tables:
            tables[key] = pq.read_table(
                root / "data" / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.parquet"
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
            "uuid": np.str_(metadata["source_episode_id"][row]),
        }
        counts = {}
        for camera, name in LEROBOT_CAMERAS.items():
            chunk = episodes[f"videos/observation.images.{camera}/chunk_index"][e]
            file = episodes[f"videos/observation.images.{camera}/file_index"][e]
            relative = Path("videos") / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
            video = next((r / relative for r in (root, wrist_root) if r is not None and (r / relative).exists()), None)
            if video is None:
                continue
            start = episodes[f"videos/observation.images.{camera}/from_timestamp"][e]
            stop = episodes[f"videos/observation.images.{camera}/to_timestamp"][e]
            frames = _decode_range(video, start, stop)
            for i, image in enumerate(frames):
                _save_jpeg(image, raw / "frames" / uuid / name / f"{i:03d}.jpg")
            counts[name] = len(frames)
        (raw / "lerobot").mkdir(parents=True, exist_ok=True)
        np.savez(raw / "lerobot" / f"{uuid}.npz", **arrays)
        print(f"{uuid}: {mask.sum()} samples, offset {offset:.4f} s, frames {counts}")


# ----------------------------------------------------------------------------- build: images


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _frames(raw: Path, uuid: str, camera: str = "top") -> list[Path]:
    return sorted((raw / "frames" / uuid / camera).glob("*.jpg"))


class Camera:
    """Top camera: RealSense inverse Brown-Conrady intrinsics and the fitted pose (OpenCV axes)."""

    def __init__(self, intrinsics: dict, rotation_xyzw_newton, position):
        self.K = np.asarray(intrinsics["K"], dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(intrinsics["D"], dtype=np.float64)
        self.width, self.height = int(intrinsics["width"]), int(intrinsics["height"])
        # Newton cameras look along -Z with +Y up; OpenCV looks along +Z with +Y down.
        self.R_wc = rc.quat_to_matrix(np.asarray(rotation_xyzw_newton)) @ np.diag([1.0, -1.0, -1.0])
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


# Colour model: RGB-histogram Bayes classifier trained on the first and last top frames of the LeRobot
# copy of the training episode (same codec as the other 10 fps episodes), from hand-placed boxes.
COLOUR_CLASSES = ("background", "pear", "orange", "dark_fruit", "tray")
COLOUR_BINS = 32
COLOUR_BOXES = {  # class: (u0, v0, u1, v1) in the first and in the last frame
    "first": {
        "pear": (198, 213, 248, 247),
        "orange": (345, 265, 376, 295),
        "dark_fruit": (394, 216, 427, 245),
        "tray": (272, 147, 358, 224),
    },
    "last": {
        "orange": (284, 180, 312, 210),
        "pear": (305, 186, 333, 215),
        "dark_fruit": (318, 174, 346, 202),
        "tray": (275, 150, 360, 228),
    },
}
SHADING_GAINS = (0.35, 0.5, 0.65, 0.8, 1.0, 1.2)  # arm shadows mostly scale the intensity
TABLE_REGION = (slice(100, 400), slice(110, 560))  # rows, columns of the table in the top image


def _hsv(image):
    x = image.astype(np.float32) / 255.0
    high, low = x.max(-1), x.min(-1)
    d = high - low + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(high == r, ((g - b) / d) % 6, np.where(high == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (high + 1e-6), high


def _loose_rule(image, name):
    h, s, v = _hsv(image)
    if name == "pear":
        return (h > 35) & (h < 65) & (s > 0.5) & (v > 0.2)
    if name == "orange":
        return ((h < 28) | (h > 345)) & (s > 0.75) & (v > 0.2)
    if name == "dark_fruit":
        return _dark_rule(image)
    return (h > 12) & (h < 42) & (s > 0.25) & (s < 0.8) & (v > 0.3)


def _dark_rule(image):
    """Dark red-brown fruit: dim and red-dominant (the robot's blacks are green-tinted, cables bluish)."""
    x = image.astype(np.int32)
    s = x.sum(-1)
    return (s < 130) & (s > 20) & (x[..., 0] > 0.42 * s) & (x[..., 0] > x[..., 1] + 6) & (x[..., 0] > x[..., 2] + 6)


def train_colour_model(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    counts = np.zeros((len(COLOUR_CLASSES), COLOUR_BINS, COLOUR_BINS, COLOUR_BINS))

    def add(hist, pixels, gains=SHADING_GAINS):
        for gain in gains:
            q = np.clip(pixels.astype(np.float32) * gain, 0, 255).astype(np.int64) // (256 // COLOUR_BINS)
            np.add.at(hist, tuple(q.T), 1.0 / len(gains))

    for image, boxes in ((first, COLOUR_BOXES["first"]), (last, COLOUR_BOXES["last"])):
        taken = np.zeros(image.shape[:2], bool)
        fruit = np.zeros(image.shape[:2], bool)
        for name in ("pear", "orange", "dark_fruit", "tray"):
            u0, v0, u1, v1 = boxes[name]
            taken[v0:v1, u0:u1] = True
            mask = np.zeros_like(taken)
            mask[v0:v1, u0:u1] = _loose_rule(image[v0:v1, u0:u1], name)
            if name == "tray":
                mask &= ~fruit
            else:
                mask = ndimage.binary_opening(mask, iterations=1)
                fruit |= ndimage.binary_dilation(mask, iterations=2)
            add(counts[COLOUR_CLASSES.index(name)], image[mask])
        add(counts[0], image[~taken], (1.0,))
    counts = np.stack([ndimage.gaussian_filter(c, 1.0) for c in counts])
    likelihood = counts / counts.sum(axis=(1, 2, 3), keepdims=True)
    posterior = likelihood * np.array([0.97, 0.0075, 0.0075, 0.0075, 0.0075])[:, None, None, None]
    return (posterior / (posterior.sum(0, keepdims=True) + 1e-30)).astype(np.float32)


def classify(model: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Label image [H, W]: 0 background, 1 pear, 2 orange, 3 dark fruit, 4 tray (table region only)."""
    q = image // (256 // COLOUR_BINS)
    p = model[:, q[..., 0], q[..., 1], q[..., 2]]
    labels = p.argmax(0)
    labels[p.max(0) < 0.6] = 0
    # The histogram confuses the dark fruit with the robot's blacks; a chroma rule finds it instead.
    labels[labels == 3] = 0
    labels[_dark_rule(image) & ((labels == 0) | (labels == 4))] = 3
    region = np.zeros(labels.shape, bool)
    region[TABLE_REGION] = True
    labels[~region] = 0
    return labels


def detect_fruits(labels: np.ndarray, min_area: int = 40) -> dict[str, dict]:
    """Largest blob of each fruit class: pixel centroid, area, and mask."""
    from scipy import ndimage

    out = {}
    for k, name in enumerate(("pear", "orange", "dark_fruit"), start=1):
        mask = ndimage.binary_opening(labels == k, iterations=1)
        components, count = ndimage.label(mask)
        if count == 0:
            continue
        areas = ndimage.sum(mask, components, range(1, count + 1))
        best = int(np.argmax(areas)) + 1
        if areas[best - 1] < min_area:
            continue
        ys, xs = np.nonzero(components == best)
        out[name] = {
            "uv": [float(xs.mean() + 0.5), float(ys.mean() + 0.5)],
            "area": int(areas[best - 1]),
            "mask": components == best,
        }
    return out


def tray_silhouette(labels: np.ndarray) -> np.ndarray:
    """Tray pixels (with any fruit lying in it), holes filled, largest blob."""
    from scipy import ndimage

    mask = (labels == 4) | ((labels >= 1) & (labels <= 3))
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=3))
    components, count = ndimage.label(mask)
    if count == 0:
        return mask
    areas = ndimage.sum(mask, components, range(1, count + 1))
    return components == (int(np.argmax(areas)) + 1)


def _sector_points(apex, yaw, radius, half, n=24):
    angles = np.linspace(yaw - half, yaw + half, n)
    return np.vstack([apex, np.asarray(apex) + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)])


def _raster_prism(camera: Camera, polygon: np.ndarray, height: float) -> np.ndarray:
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    points = np.vstack(
        [np.c_[polygon, np.full(len(polygon), TABLE_Z)], np.c_[polygon, np.full(len(polygon), TABLE_Z + height)]]
    )
    uv = camera.project(points)
    hull = uv[ConvexHull(uv).vertices]
    image = Image.new("L", (camera.width, camera.height), 0)
    ImageDraw.Draw(image).polygon([tuple(p) for p in hull], fill=1)
    return np.asarray(image, bool)


def fit_tray(camera: Camera, silhouette: np.ndarray, radius: float, half: float) -> tuple[np.ndarray, float]:
    """Sector apex xy [m] and bisector yaw [rad] whose projected prism best overlaps the silhouette (IoU).

    The radius and half-angle are the training episode's (same tray). The search varies the sector's area
    centroid and its yaw about that centroid (rotating about the apex would couple the parameters), from
    a yaw scan with the centroid on the silhouette's back-projected centroid.
    """
    offset = 2 * radius * math.sin(half) / (3 * half)  # apex to area centroid [m]

    def apex(x):
        return x[:2] - offset * np.array([math.cos(x[2]), math.sin(x[2])])

    def iou(x):
        mask = _raster_prism(camera, _sector_points(apex(x), x[2], radius, half), TRAY_RIM)
        return float((mask & silhouette).sum() / max((mask | silhouette).sum(), 1))

    ys, xs = np.nonzero(silhouette)
    centroid = camera.backproject_z(xs + 0.5, ys + 0.5, TABLE_Z + 0.5 * TRAY_RIM)[:, :2].mean(0)
    scan = [(iou(np.array([*centroid, yaw])), yaw) for yaw in np.radians(np.arange(-180.0, 180.0, 5.0))]
    best = None
    for _, yaw in sorted(scan, reverse=True)[:3]:
        x = np.array([*centroid, yaw])
        value = iou(x)
        steps = np.array([0.01, 0.01, math.radians(4.0)])
        for _ in range(7):
            improved = True
            while improved:
                improved = False
                for d in range(3):
                    for sign in (1.0, -1.0):
                        trial = x.copy()
                        trial[d] += sign * steps[d]
                        v = iou(trial)
                        if v > value + 1e-6:
                            x, value, improved = trial, v, True
            steps *= 0.5
        if best is None or value > best[1]:
            best = (x, value)
    x, value = best
    return np.array([*apex(x), (x[2] + math.pi) % (2 * math.pi) - math.pi]), value


# ----------------------------------------------------------------------------- build: episodes and events


def lerobot_episode(raw: Path, uuid: str) -> dict[str, np.ndarray]:
    """Episode dict of a 10 fps LeRobot episode on a 30 Hz grid (see :func:`episode_from_lerobot`)."""
    data = np.load(raw / "lerobot" / f"{uuid}.npz")
    return rc.episode_from_lerobot(
        data["timestamp"],
        data["observation_state"],
        data["action"],
        velocity=data["observation_velocity"],
        torque=data["observation_torque"],
        time_offset=float(data["time_offset"]),
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is true."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def find_holds(episode: dict, min_rows: int = 6) -> list[dict]:
    """Grasps on the state grid: rows of the close command, finger contact, lift-off, open command, release.

    Same rules as the training episode's tracker, applied per arm to the measured opening ``g`` and the
    command ``c``: a hold is a run of ``c < 0.35`` (at least ``min_rows`` samples); the close command is
    where ``c`` last left 0.95 before it; contact is where the fingers stop closing while the command is
    still below the opening; the open command is where ``c`` rises above the held opening; release is
    where the fingers have opened 0.05 beyond it. Lift-off is filled in later from the fruit start.
    """
    holds = []
    for side in rc.SIDES:
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
            holds.append({"arm": side, "rows": {"cmd_close": k, "contact": j, "cmd_open": o, "release": r}})
    holds.sort(key=lambda h: h["rows"]["contact"])
    return holds


def rows_to_frames(rows: dict[str, int], top_rows: np.ndarray) -> dict[str, int]:
    """Top-frame index showing each state row (nearest)."""
    return {key: int(np.argmin(np.abs(top_rows - row))) for key, row in rows.items()}


def hold_geometry(fk, episode: dict, hold: dict, name: str) -> dict:
    """FK quantities of a hold for a candidate fruit: start xy (pad axes at the fruit's centre height,
    averaged from contact to the row before lift-off), the measured held gap, and the pad tip height."""
    side, rows = hold["arm"], hold["rows"]
    height = TABLE_Z + FRUIT_SIZES[name]["centre_height_m"]
    _, g = rc.measured(episode, side)
    last = max(rows.get("liftoff", rows["contact"] + 2) - 1, rows["contact"] + 1)
    points, tips = [], []
    for row in range(rows["contact"], last):
        body_q = fk.pose_row(episode, row)
        pads = fk.pads(body_q, side)
        points.append(np.mean([c + a * (height - c[2]) / a[2] for c, a, _ in pads], axis=0))
        for body in fk.pad_bodies[side]:
            rotation = rc.quat_to_matrix(body_q[body, 3:])
            tips.append((body_q[body, :3] + rotation @ np.asarray(PAD_TIP))[2])
    return {
        "xy": np.mean(points, axis=0)[:2],
        "gap_m": float(2 * np.median(g[rows["contact"] : rows["cmd_open"]]) * rc.GRIPPER_TRAVEL),
        "pad_tip_height_m": float(np.median(tips) - TABLE_Z),
    }


def find_liftoff(fk, episode: dict, hold: dict, start: np.ndarray) -> int:
    """First row after contact where the fruit, rigidly attached to the hand from contact, is 1 cm higher."""
    side, rows = hold["arm"], hold["rows"]
    body = fk.hand_body[side]
    pose = fk.pose_row(episode, rows["contact"])[body]
    local = rc.quat_to_matrix(pose[3:]).T @ (start - pose[:3])
    for row in range(rows["contact"], rows["release"]):
        pose = fk.pose_row(episode, row)[body]
        if (pose[:3] + rc.quat_to_matrix(pose[3:]) @ local)[2] > start[2] + 0.01:
            return row
    return rows["release"]


def chord_width(name: str, tip_height: float) -> float:
    """Width [m] of a fruit lying on the table across the pads, if the pad tips stop at ``tip_height``."""
    r = 0.5 * FRUIT_WIDTH[name]
    above = max(0.0, tip_height - FRUIT_SIZES[name]["centre_height_m"])
    return 2.0 * math.sqrt(max(r * r - above * above, 0.0))


# ----------------------------------------------------------------------------- build: scenes


def _tray_entry(apex, yaw_deg, radius, half_deg, **extra) -> dict:
    return {
        "apex_xy": [round(float(apex[0]), 4), round(float(apex[1]), 4)],
        "yaw_deg": round(float(yaw_deg), 2),
        "radius_m": float(radius),
        "half_angle_deg": float(half_deg),
        "rim_height_m": TRAY_RIM,
        "floor_height_m": TRAY_FLOOR,
        **extra,
    }


def _fruit_entry(name: str, arm: str, events: dict) -> dict:
    return {"arm": arm, **copy.deepcopy(FRUIT_SIZES[name]), "mass_range_kg": list(MASS_RANGE_KG), "events": events}


def _round(values, digits=5) -> list[float]:
    return [round(float(v), digits) for v in values]


def _set_starts(fk, episode: dict, scene: dict) -> None:
    for name, start in fk.fruit_starts(episode, scene).items():
        scene["fruits"][name]["start"] = {"pos": _round(start["pos"]), "quat_xyzw": _round(start["quat_xyzw"], 6)}


def main_scene(meta: dict, episode: dict, fk) -> dict:
    """Scene of the training episode: events and tray from the video tracker, FK-consistent starts."""
    tray = meta["tray"]
    scene = {
        "name": "main",
        "episode": "episodes/main.npz",
        "uuid": MAIN_UUID,
        "source": "ABC-130k val, Voxel51 MCAP mirror (about 30 Hz)",
        "frame_rate": 30.0,
        "table_z": TABLE_Z,
        "bases": copy.deepcopy(BASES),
        "tray": _tray_entry(
            tray["apex_xy"],
            tray["yaw_deg"],
            tray["radius_m"],
            tray["half_angle_deg"],
            iou=tray["frame0_iou"],
            end_apex_xy=tray["pose_end"][:2],
            end_yaw_deg=tray["pose_end"][2],
        ),
        "fruits": {},
        "order": list(rc.FRUITS),
        "time_base": TIME_BASE,
    }
    for name in rc.FRUITS:
        obj = meta["objects"][name]
        fruit = _fruit_entry(name, obj["arm"], {k: int(v) for k, v in obj["frames"].items()})
        fruit["grip_gap_m"] = round(float(obj["size"]["gripper_gap_hold_m"]), 4)
        fruit["start_image_xy"] = _round(obj["rest_initial_xyz"][:2], 4)
        fruit["final_image_xyz"] = _round(obj["final_xyz"], 4)
        scene["fruits"][name] = fruit
    scene["grasp_order"] = sorted(rc.FRUITS, key=lambda n: scene["fruits"][n]["events"]["contact"])
    _set_starts(fk, episode, scene)
    return scene


def build_lerobot_scene(raw, uuid, name, fk, camera, model, tray_shape, quality) -> tuple[dict, dict]:
    """Scene of a 10 fps episode from its logs and first and last top frames (see the module docstring).

    Returns the scene and the episode; ``quality`` receives the per-episode checks.
    """
    episode = lerobot_episode(raw, uuid)
    top_rows = rc.frame_rows(episode)
    frames = _frames(raw, uuid)
    first, last = _load_image(frames[0]), _load_image(frames[-1])
    det_first, det_last = detect_fruits(classify(model, first)), detect_fruits(classify(model, last))
    image_start = {
        n: camera.backproject_z(*d["uv"], TABLE_Z + FRUIT_SIZES[n]["centre_height_m"])[:2] for n, d in det_first.items()
    }

    holds = find_holds(episode)
    if len(holds) != 3:
        raise RuntimeError(f"{uuid}: expected three holds, found {len(holds)}")
    # The fruit of each hold: the assignment with the smallest total distance between the pad midpoint at the
    # grasp and the first-frame detections.
    geometry = [{n: hold_geometry(fk, episode, h, n) for n in rc.FRUITS} for h in holds]
    best = None
    for perm in itertools.permutations(rc.FRUITS):
        cost = sum(
            np.linalg.norm(geometry[k][n]["xy"] - image_start[n]) if n in image_start else 1.0
            for k, n in enumerate(perm)
        )
        if best is None or cost < best[0]:
            best = (cost, perm)
    assignment = best[1]

    scene = {
        "name": name,
        "episode": f"episodes/{name}.npz",
        "uuid": str(np.load(raw / "lerobot" / f"{uuid}.npz")["uuid"])[len("episode_") :],
        "source": "ABC-130k train, LeRobot 10 fps copy, interpolated to 30 Hz",
        "frame_rate": 10.0,
        "table_z": TABLE_Z,
        "bases": copy.deepcopy(BASES),
        "fruits": {},
        "order": list(rc.FRUITS),
        "grasp_order": list(assignment),
        "time_base": TIME_BASE,
    }
    checks = {"holds": []}
    for k, (hold, fruit_name) in enumerate(zip(holds, assignment, strict=True)):
        g = hold_geometry(fk, episode, hold, fruit_name)
        start = np.array([*g["xy"], TABLE_Z + FRUIT_SIZES[fruit_name]["centre_height_m"]])
        for _ in range(3):  # the start averages the rows up to lift-off, which depends on the start
            hold["rows"]["liftoff"] = find_liftoff(fk, episode, hold, start)
            g = hold_geometry(fk, episode, hold, fruit_name)
            start[:2] = g["xy"]
        rows = {k: int(v) for k, v in hold["rows"].items()}
        events = rows_to_frames(rows, top_rows)
        fruit = _fruit_entry(
            fruit_name, hold["arm"], {k: events[k] for k in ("cmd_close", "contact", "liftoff", "cmd_open", "release")}
        )
        fruit["event_rows"] = {k: rows[k] for k in ("cmd_close", "contact", "liftoff", "cmd_open", "release")}
        fruit["grip_gap_m"] = round(g["gap_m"], 4)
        if fruit_name in image_start:
            fruit["start_image_xy"] = _round(image_start[fruit_name], 4)
        if fruit_name in det_last:
            height = TABLE_Z + TRAY_FLOOR + FRUIT_SIZES[fruit_name]["centre_height_m"]
            fruit["final_image_xyz"] = _round(camera.backproject_z(*det_last[fruit_name]["uv"], height), 4)
        scene["fruits"][fruit_name] = fruit
        checks["holds"].append(
            {
                "arm": hold["arm"],
                "fruit": fruit_name,
                "rows": rows,
                "frames": fruit["events"],
                "gap_mm": round(1000 * g["gap_m"], 1),
                "expected_gap_mm": round(1000 * chord_width(fruit_name, g["pad_tip_height_m"]), 1),
                "pad_tip_height_mm": round(1000 * g["pad_tip_height_m"], 1),
                "row_start_xy": _round(g["xy"], 4),
                "pad_to_image_mm": {
                    n: round(1000 * float(np.linalg.norm(geometry[k][n]["xy"] - image_start[n])), 1)
                    for n in image_start
                },
            }
        )
    scene["fruits"] = {n: scene["fruits"][n] for n in rc.FRUITS}  # same key order in every scene
    silhouette = tray_silhouette(classify(model, first))
    tray, iou = fit_tray(camera, silhouette, tray_shape["radius_m"], math.radians(tray_shape["half_angle_deg"]))
    scene["tray"] = _tray_entry(
        tray[:2], math.degrees(tray[2]), tray_shape["radius_m"], tray_shape["half_angle_deg"], iou=round(iou, 4)
    )
    _set_starts(fk, episode, scene)

    # Checks: start vs image, frame-level vs row-level start, clearance between starts, finals in the tray.
    for name_ in rc.FRUITS:
        fruit = scene["fruits"][name_]
        hold = next(h for h in checks["holds"] if h["fruit"] == name_)
        start = np.asarray(fruit["start"]["pos"])
        hold["start_vs_row_start_mm"] = round(1000 * float(np.linalg.norm(start[:2] - hold["row_start_xy"])), 1)
        if "start_image_xy" in fruit:
            hold["start_vs_image_mm"] = round(1000 * float(np.linalg.norm(start[:2] - fruit["start_image_xy"])), 1)
        if "final_image_xyz" in fruit:
            hold["final_tray_margin_mm"] = round(1000 * sector_margin(fruit["final_image_xyz"], scene["tray"]), 1)
        release = fk.grasp_point(fk.pose_row(episode, hold["rows"]["release"]), fruit["arm"])
        hold["release_tray_margin_mm"] = round(1000 * sector_margin(release, scene["tray"]), 1)
        hold["pad_tip_above_centre_mm"] = round(hold["pad_tip_height_mm"] - 1000 * fruit["centre_height_m"], 1)
        hold["release_height_mm"] = round(1000 * float(release[2] - TABLE_Z), 1)
    checks["start_clearance_mm"] = start_clearances(scene)
    checks["tray_iou"] = round(iou, 4)
    checks["detected_first"] = sorted(det_first)
    checks["detected_last"] = sorted(det_last)
    checks["moved_before_grasp_mm"] = moved_before_grasp(raw, uuid, model, camera, scene, det_first)
    quality[name] = checks
    return scene, episode


def start_clearances(scene: dict) -> dict[str, float]:
    """Gap [mm] between each pair of fruit starts (bounding spheres for the round fruits, the pear's
    ellipse in the table plane), negative if they overlap."""
    out = {}
    for a, b in itertools.combinations(scene["fruits"], 2):
        pa, pb = (np.asarray(scene["fruits"][n]["start"]["pos"][:2]) for n in (a, b))
        d = pb - pa
        distance = float(np.linalg.norm(d))
        extent = 0.0
        for n, direction in ((a, d), (b, -d)):
            fruit = scene["fruits"][n]
            if fruit["shape"] == "pear":
                q = fruit["start"]["quat_xyzw"]
                yaw = 2.0 * math.atan2(q[2], q[3])
                angle = math.atan2(direction[1], direction[0]) - yaw
                ax, ay = 0.5 * fruit["size_m"]["length"], 0.5 * fruit["size_m"]["width"]
                extent += ax * ay / math.hypot(ay * math.cos(angle), ax * math.sin(angle))
            else:
                extent += 0.5 * fruit["size_m"]["diameter"]
        out[f"{a}-{b}"] = round(1000 * (distance - extent), 1)
    return out


def moved_before_grasp(raw, uuid, model, camera, scene, det_first) -> dict[str, float | None]:
    """Image displacement [mm at the fruit's centre height] of each fruit before its close command: the median
    position over the last three unoccluded frames (detected area within 10% of the first frame's) minus the
    median over the first three. A push that the fruit start does not show appears here; turning in place
    does not."""
    frames = _frames(raw, uuid)
    out = {}
    for name, fruit in scene["fruits"].items():
        if name not in det_first:
            out[name] = None
            continue
        height = TABLE_Z + fruit["centre_height_m"]
        points = []
        for k in range(fruit["events"]["cmd_close"]):
            det = detect_fruits(classify(model, _load_image(frames[k])))
            if name in det and abs(det[name]["area"] - det_first[name]["area"]) <= 0.1 * det_first[name]["area"]:
                points.append(camera.backproject_z(*det[name]["uv"], height)[:2])
        if len(points) < 4:
            out[name] = None
            continue
        points = np.asarray(points)
        moved = np.median(points[-3:], axis=0) - np.median(points[:3], axis=0)
        out[name] = round(1000 * float(np.linalg.norm(moved)), 1)
    return out


def sector_margin(xy, tray: dict) -> float:
    """Signed distance [m] from a point to the tray sector's outline: positive inside, negative outside."""
    outside = rc.sector_distance(xy, tray)
    if outside > 0.0:
        return -outside
    apex = np.asarray(tray["apex_xy"], dtype=np.float64)
    d = np.asarray(xy, dtype=np.float64)[:2] - apex
    margins = [float(tray["radius_m"]) - float(np.hypot(*d))]
    for sign in (-1.0, 1.0):
        angle = math.radians(tray["yaw_deg"] + sign * tray["half_angle_deg"])
        margins.append(abs(math.cos(angle) * d[1] - math.sin(angle) * d[0]))
    return min(margins)


# ----------------------------------------------------------------------------- build: ground truth


def lerobot_gt(fk, episode: dict, scene: dict, camera: Camera, raw: Path, uuid: str, model) -> dict:
    """Ground truth (:func:`load_gt` format) of a 10 fps episode: FK starts until contact, FK-attached
    centres until release, the last frame's detection, image centroids at the first and last frames."""
    rows = rc.frame_rows(episode)
    count = len(rows)
    tracks = fk.attached_tracks(episode, scene)
    frames = _frames(raw, uuid)
    detections = {k: detect_fruits(classify(model, _load_image(frames[k]))) for k in (0, count - 1)}
    gt = {"t": episode["t_top"], "frame": np.arange(count), "top_state_index": rows}
    for name in rc.FRUITS:
        fruit = scene["fruits"][name]
        events = fruit["events"]
        pos = tracks[name].copy()
        src = np.zeros(count, np.int8)
        src[: events["contact"]] = 4
        src[events["contact"] : events["release"]] = 2
        phase = np.full(count, 3, np.int8)
        phase[: events["contact"]] = 0
        phase[events["contact"] : events["liftoff"]] = 1
        phase[events["liftoff"] : events["release"]] = 2
        phase[-1] = 4
        uv = np.full((count, 2), np.nan)
        fix = np.zeros(count, bool)
        radpx = np.full(count, np.nan)
        for k, det in detections.items():
            if name in det:
                uv[k] = det[name]["uv"]
                fix[k] = True
                radpx[k] = math.sqrt(det[name]["area"] / math.pi)
        if "final_image_xyz" in fruit:
            pos[-1] = fruit["final_image_xyz"]
            src[-1] = 1
        gt[f"{name}_pos"], gt[f"{name}_src"], gt[f"{name}_phase"] = pos, src, phase
        gt[f"{name}_uv"], gt[f"{name}_fix"], gt[f"{name}_radpx"] = uv, fix, radpx
        start = np.asarray(fruit["start"]["pos"])
        image = fruit.get("start_image_xy")
        gt[f"{name}_rest_xyz"] = np.array([*image, start[2]]) if image is not None else start
        gt[f"{name}_start_fk_xyz"] = start
        if "final_image_xyz" in fruit:
            gt[f"{name}_final_xyz"] = np.asarray(fruit["final_image_xyz"])
    for side in rc.SIDES:
        _, grip = rc.measured(episode, side)
        gt[f"{side}_pad"] = np.array([fk.grasp_point(fk.pose_row(episode, int(r)), side) for r in rows])
        gt[f"{side}_grip"] = grip[rows]
        gt[f"{side}_gap"] = 2.0 * np.clip(grip[rows], 0.0, 1.0) * rc.GRIPPER_TRAVEL
    return gt


# ----------------------------------------------------------------------------- build: overlays


FRUIT_COLOURS = {"pear": (255, 235, 0), "orange": (255, 120, 0), "dark_fruit": (230, 0, 255)}


def _fruit_outline(fruit: dict, centre) -> np.ndarray:
    """Points [n, 3] on the fruit's outline in the horizontal plane through its centre."""
    angles = np.linspace(0.0, 2.0 * math.pi, 40)
    if fruit["shape"] == "pear":
        q = fruit["start"]["quat_xyzw"]
        yaw = 2.0 * math.atan2(q[2], q[3])
        a, b = 0.5 * fruit["size_m"]["length"], 0.5 * fruit["size_m"]["width"]
        local = np.stack([a * np.cos(angles), b * np.sin(angles)], axis=-1)
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        xy = local @ rotation.T
    else:
        r = 0.5 * fruit["size_m"]["diameter"]
        xy = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=-1)
    return np.c_[xy + np.asarray(centre)[:2], np.full(len(xy), centre[2])]


def overlay_sheet(path: Path, raw: Path, uuid: str, camera: Camera, scene: dict, episode: dict, fk, checks) -> None:
    """Check images of one 10 fps episode: first frame (tray fit, FK starts, detections), last frame (finals),
    each hold's lift-off frame (FK pads and the attached fruit), and the grasping wrist view at contact."""
    from PIL import Image, ImageDraw

    rows = rc.frame_rows(episode)
    frames = _frames(raw, uuid)
    tray_xy = rc.sector_polygon(scene["tray"])
    tray_uv = camera.project(np.c_[tray_xy, np.full(len(tray_xy), TABLE_Z + TRAY_RIM)])
    tiles = []

    def annotate_tray(draw):
        draw.line([tuple(p) for p in np.vstack([tray_uv, tray_uv[:1]])], fill=(0, 255, 255), width=1)

    image = Image.open(frames[0]).convert("RGB")
    draw = ImageDraw.Draw(image)
    annotate_tray(draw)
    for name, fruit in scene["fruits"].items():
        colour = FRUIT_COLOURS[name]
        outline = camera.project(_fruit_outline(fruit, fruit["start"]["pos"]))
        draw.line([tuple(p) for p in outline], fill=colour, width=1)
        u, v = camera.project(np.asarray(fruit["start"]["pos"]))
        draw.line([(u - 3, v), (u + 3, v)], fill=colour)
        draw.line([(u, v - 3), (u, v + 3)], fill=colour)
        if "start_image_xy" in fruit:
            iu, iv = camera.project(np.array([*fruit["start_image_xy"], fruit["start"]["pos"][2]]))
            draw.rectangle([iu - 2, iv - 2, iu + 2, iv + 2], outline=(0, 0, 0))
        draw.text(
            (u + 12, v - 6), f"{name[:4]} {fruit['arm'][0].upper()} {1000 * fruit['grip_gap_m']:.0f}mm", fill=colour
        )
    draw.text((4, 4), f"{scene['name']} {uuid} frame 0  tray IoU {scene['tray'].get('iou', 0):.3f}", fill=(255, 0, 0))
    tiles.append(image)

    image = Image.open(frames[-1]).convert("RGB")
    draw = ImageDraw.Draw(image)
    annotate_tray(draw)
    for name, fruit in scene["fruits"].items():
        if "final_image_xyz" in fruit:
            u, v = camera.project(np.asarray(fruit["final_image_xyz"]))
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=FRUIT_COLOURS[name], width=2)
            hold = next(h for h in checks["holds"] if h["fruit"] == name)
            draw.text((u + 8, v - 6), f"{name[:4]} {hold.get('final_tray_margin_mm')}", fill=FRUIT_COLOURS[name])
    draw.text((4, 4), f"last frame {len(frames) - 1}: finals, margin inside the tray [mm]", fill=(255, 0, 0))
    tiles.append(image)

    tracks = fk.attached_tracks(episode, scene)
    wrist = []
    for name in scene["grasp_order"]:
        fruit = scene["fruits"][name]
        k = min(fruit["events"]["liftoff"] + 1, len(frames) - 1)
        image = Image.open(frames[k]).convert("RGB")
        draw = ImageDraw.Draw(image)
        body_q = fk.pose_row(episode, int(rows[k]))
        for _, _, point in fk.pads(body_q, fruit["arm"]):
            u, v = camera.project(point)
            draw.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(0, 255, 0), width=2)
        if np.isfinite(tracks[name][k, 0]):
            outline = camera.project(_fruit_outline(fruit, tracks[name][k]))
            draw.line([tuple(p) for p in outline], fill=FRUIT_COLOURS[name], width=1)
        draw.text((4, 4), f"{name} {fruit['arm']} lift-off+1 frame {k} (row {rows[k]})", fill=(255, 0, 0))
        tiles.append(image)
        wrist_frames = _frames(raw, uuid, f"{fruit['arm']}_wrist")
        if wrist_frames:
            k = min(fruit["events"]["contact"], len(wrist_frames) - 1)
            image = Image.open(wrist_frames[k]).convert("RGB")
            ImageDraw.Draw(image).text((4, 4), f"{fruit['arm']} wrist, {name} contact frame {k}", fill=(255, 0, 0))
            wrist.append(image)
    tiles.append(Image.new("RGB", tiles[0].size))
    tiles += wrist
    scale = 0.6
    w, h = int(640 * scale), int(480 * scale)
    columns = 3
    sheet = Image.new("RGB", (columns * w, ((len(tiles) + columns - 1) // columns) * h))
    for i, tile in enumerate(tiles):
        sheet.paste(tile.resize((w, h)), ((i % columns) * w, (i // columns) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def grasp_sheet(path: Path, raw: Path, uuid: str, camera: Camera, scene: dict, episode: dict, fk) -> None:
    """One row per hold: top-frame crops around the grasp (close command, contact - 1, contact, lift-off,
    lift-off + 2) with the FK pad grasp points (green) and the fruit start outline, then the wrist view
    at contact."""
    from PIL import Image, ImageDraw

    rows = rc.frame_rows(episode)
    frames = _frames(raw, uuid)
    size, scale = 90, 2
    tile = 2 * size * scale
    sheet_rows = []
    for name in scene["grasp_order"]:
        fruit = scene["fruits"][name]
        events = fruit["events"]
        centre = camera.project(np.asarray(fruit["start"]["pos"]))
        box = (int(centre[0]) - size, int(centre[1]) - size, int(centre[0]) + size, int(centre[1]) + size)
        tiles = []
        keys = (
            ("close", events["cmd_close"]),
            ("contact-1", events["contact"] - 1),
            ("contact", events["contact"]),
            ("liftoff", events["liftoff"]),
            ("liftoff+2", events["liftoff"] + 2),
        )
        for label, frame in keys:
            k = int(np.clip(frame, 0, len(frames) - 1))
            image = Image.open(frames[k]).convert("RGB").crop(box).resize((tile, tile))
            draw = ImageDraw.Draw(image)
            outline = (camera.project(_fruit_outline(fruit, fruit["start"]["pos"])) - box[:2]) * scale
            draw.line([tuple(p) for p in outline], fill=FRUIT_COLOURS[name], width=1)
            body_q = fk.pose_row(episode, int(rows[k]))
            for _, _, point in fk.pads(body_q, fruit["arm"]):
                u, v = (camera.project(point) - box[:2]) * scale
                draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(0, 255, 0), width=2)
            draw.text((3, 3), f"{name} {fruit['arm']} {label} f{k}", fill=(255, 0, 0))
            tiles.append(image)
        wrist_frames = _frames(raw, uuid, f"{fruit['arm']}_wrist")
        if wrist_frames:
            k = int(np.clip(events["contact"], 0, len(wrist_frames) - 1))
            image = Image.open(wrist_frames[k]).convert("RGB").resize((int(tile * 4 / 3), tile))
            ImageDraw.Draw(image).text(
                (3, 3), f"{fruit['arm']} wrist f{k} gap {1000 * fruit['grip_gap_m']:.1f} mm", fill=(255, 0, 0)
            )
            tiles.append(image)
        sheet_rows.append(tiles)
    width = max(sum(t.size[0] for t in r) for r in sheet_rows)
    sheet = Image.new("RGB", (width, tile * len(sheet_rows)))
    for i, tiles in enumerate(sheet_rows):
        x = 0
        for t in tiles:
            sheet.paste(t, (x, i * tile))
            x += t.size[0]
    sheet.save(path, quality=88)


def start_zoom(path: Path, raw: Path, uuid: str, camera: Camera, scene: dict) -> None:
    """3x crops of the first frame around each fruit start with the FK outline (colour) and the detection (box)."""
    from PIL import Image, ImageDraw

    image = Image.open(_frames(raw, uuid)[0]).convert("RGB")
    crops = []
    for name, fruit in scene["fruits"].items():
        u, v = camera.project(np.asarray(fruit["start"]["pos"]))
        box = (int(u) - 40, int(v) - 40, int(u) + 40, int(v) + 40)
        crop = image.crop(box).resize((240, 240), Image.NEAREST)
        draw = ImageDraw.Draw(crop)
        outline = (camera.project(_fruit_outline(fruit, fruit["start"]["pos"])) - [box[0], box[1]]) * 3
        draw.line([tuple(p) for p in outline], fill=FRUIT_COLOURS[name], width=2)
        if "start_image_xy" in fruit:
            iu, iv = (
                camera.project(np.array([*fruit["start_image_xy"], fruit["start"]["pos"][2]])) - [box[0], box[1]]
            ) * 3
            draw.rectangle([iu - 4, iv - 4, iu + 4, iv + 4], outline=(0, 0, 0), width=2)
        draw.text((4, 4), name, fill=(255, 0, 0))
        crops.append(crop)
    sheet = Image.new("RGB", (240 * len(crops), 240))
    for i, crop in enumerate(crops):
        sheet.paste(crop, (240 * i, 0))
    sheet.save(path, quality=90)


# ----------------------------------------------------------------------------- build


def _write_npz(path: Path, arrays: dict, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compressed else np.savez)(path, **arrays)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def _copy_frames(raw: Path, uuid: str, task: Path, name: str, cameras) -> int:
    count = 0
    for camera in cameras:
        source = raw / "frames" / uuid / camera
        if source.exists():
            shutil.copytree(source, task / "frames" / name / camera, dirs_exist_ok=True)
            count += len(list(source.glob("*.jpg")))
    return count


def build(args: argparse.Namespace) -> None:
    if rc is None:
        raise ImportError("the build stage needs Newton (replay_common)")
    raw, task, private, overlays = args.raw, args.task, args.private, args.overlays
    meta = json.loads((args.tracker / "ground_truth.json").read_text())
    trajectories = dict(np.load(args.tracker / "trajectories.npz"))
    intrinsics = json.loads((raw / "cameras.json").read_text())

    for destination in (task / "station", private / "station"):
        shutil.copytree(args.station, destination, dirs_exist_ok=True)
    if args.arm_logs is not None:
        shutil.copytree(args.arm_logs, task / "arm_logs", dirs_exist_ok=True)
    fit = meta["camera"]
    camera_json = {
        "top": {**intrinsics["top"], "position": fit["position"], "rotation_xyzw": fit["rotation_xyzw_newton"]},
        **{
            side: {
                **intrinsics[side],
                "body": rc.CAMERA_BODY[side],
                "body_rotation_xyzw": list(rc.CAMERA_IN_BODY_XYZW),
            }
            for side in rc.SIDES
        },
        "time_base": "top frame i shows arm-state sample i-1 (episode left_t; 10 fps episodes: top_state_index); "
        "wrist frames are assumed to follow the same rule",
        "convention": "position [m] and rotation_xyzw of the camera in the world; it looks along its -Z axis, +Y up",
        "top_fit": {
            "episode": MAIN,
            "method": fit["fit"]["method"],
            "edge_ncc": round(fit["fit"]["edge_ncc"], 4),
            "note": "fitted on the training episode; the 10 fps episodes use the same pose",
        },
    }
    _write_json(task / "camera.json", camera_json)
    camera = Camera(intrinsics["top"], fit["rotation_xyzw_newton"], fit["position"])
    fk = rc.StationFK(args.station / "yam_bimanual_empty.xml", BASES)

    # Training episode (MCAP).
    episode = dict(np.load(raw / "main.npz"))
    _write_npz(task / "episodes" / "main.npz", episode)
    frames = _copy_frames(raw, "main", task, "main", ("top", "left_wrist", "right_wrist"))
    scene = main_scene(meta, episode, fk)
    _write_json(task / "scenes" / "main.json", scene)
    gt = rc.gt_from_tracker(meta, trajectories)
    _write_npz(private / "gt" / "main.npz", gt, compressed=True)
    events = {name: scene["fruits"][name]["events"] for name in rc.FRUITS}
    reference = {**trajectories, **gt, "events_json": np.str_(json.dumps(events))}
    reference["format_json"] = np.str_((args.tracker / "format.json").read_text())
    _write_npz(task / "video_reference.npz", reference, compressed=True)
    print(
        f"main: {len(episode['t_top'])} top frames, {frames} frames copied, starts "
        + ", ".join(f"{n} {scene['fruits'][n]['start']['pos'][:2]}" for n in rc.FRUITS)
    )

    # 10 fps episodes. The colour model comes from the LeRobot copy of the training episode.
    copy_frames = _frames(raw, MAIN)
    model = train_colour_model(_load_image(copy_frames[0]), _load_image(copy_frames[-1]))
    quality = {}
    tray_shape = scene["tray"]
    check_scene, _ = build_lerobot_scene(raw, MAIN, "main_10fps", fk, camera, model, tray_shape, quality)
    quality["main_10fps"]["vs_main_mcap"] = {
        name: {
            "start_mm": round(
                1000
                * float(
                    np.linalg.norm(
                        np.asarray(check_scene["fruits"][name]["start"]["pos"][:2])
                        - scene["fruits"][name]["start"]["pos"][:2]
                    )
                ),
                1,
            ),
            "arm": check_scene["fruits"][name]["arm"] == scene["fruits"][name]["arm"],
            # LeRobot frame k is MCAP top frame 3k + 3 (3k + 2 near the end).
            "events_mcap_frames": {k: 3 * v + 3 for k, v in check_scene["fruits"][name]["events"].items()},
            "events_main": scene["fruits"][name]["events"],
        }
        for name in rc.FRUITS
    }
    quality["main_10fps"]["tray_vs_main"] = {
        "apex_mm": round(
            1000 * float(np.linalg.norm(np.subtract(check_scene["tray"]["apex_xy"], tray_shape["apex_xy"]))), 1
        ),
        "yaw_deg": round(check_scene["tray"]["yaw_deg"] - tray_shape["yaw_deg"], 2),
    }
    if overlays is not None:
        check_episode = lerobot_episode(raw, MAIN)
        overlay_sheet(
            overlays / "main_10fps.jpg", raw, MAIN, camera, check_scene, check_episode, fk, quality["main_10fps"]
        )
        grasp_sheet(overlays / "main_10fps_grasps.jpg", raw, MAIN, camera, check_scene, check_episode, fk)

    spec = json.loads(args.heldout_spec.read_text()) if args.heldout_spec.exists() else {"heldout": []}
    checks = {**SIBLING_CHECKS, **spec.get("manual_checks", {})}
    for name, uuid in [*SIBLINGS.items(), *((u, u) for u in spec["heldout"])]:
        scene, episode = build_lerobot_scene(raw, uuid, name, fk, camera, model, tray_shape, quality)
        gt = lerobot_gt(fk, episode, scene, camera, raw, uuid, model)
        check = checks.get(name, {"verdict": "unchecked", "notes": []})
        quality[name]["manual_check"] = check
        scene["notes"] = list(check["notes"])
        if name in SIBLINGS:
            _write_npz(task / "episodes" / f"{name}.npz", episode)
            _write_json(task / "scenes" / f"{name}.json", scene)
            _copy_frames(raw, uuid, task, name, ("top",))
            for root in (task, private):
                _write_npz(root / "gt" / f"{name}.npz", gt, compressed=True)
        else:
            scene["episode"] = f"heldout/episodes/{uuid}.npz"
            scene["verdict"] = check["verdict"]
            for fruit, verdict in check.get("fruit_verdicts", {}).items():
                scene["fruits"][fruit]["verdict"] = verdict
            _write_npz(private / "heldout" / "episodes" / f"{uuid}.npz", episode)
            _write_json(private / "heldout" / "scenes" / f"{uuid}.json", scene)
            _write_npz(private / "heldout" / "gt" / f"{uuid}.npz", gt, compressed=True)
        if overlays is not None:
            overlay_sheet(overlays / f"{name}.jpg", raw, uuid, camera, scene, episode, fk, quality[name])
            start_zoom(overlays / f"{name}_starts.jpg", raw, uuid, camera, scene)
            grasp_sheet(overlays / f"{name}_grasps.jpg", raw, uuid, camera, scene, episode, fk)
        print(
            f"{name} ({uuid}): order {scene['grasp_order']}, arms "
            + ", ".join(f"{n} {scene['fruits'][n]['arm'][0]}" for n in scene["grasp_order"])
            + f", tray IoU {scene['tray']['iou']}"
        )
    # The verifier replays its own copies of the agent-facing episodes, scenes, and camera.
    for folder in ("episodes", "scenes"):
        shutil.copytree(task / folder, private / folder, dirs_exist_ok=True)
    shutil.copy2(task / "camera.json", private / "camera.json")
    _write_json(private / "heldout" / "quality.json", quality)
    if overlays is not None:
        _write_json(overlays / "quality.json", quality)


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("extract", help="decode the MCAP and LeRobot sources into a raw cache")
    p.add_argument("--mcap", type=Path, required=True, help="MCAP file of the training episode (Voxel51 mirror)")
    p.add_argument("--lerobot", type=Path, required=True, help="LeRobot train subset of the fruit-bowl task")
    p.add_argument("--lerobot-val", type=Path, help="LeRobot val subset (copy of the training episode)")
    p.add_argument("--lerobot-wrist", type=Path, help="extra root with the train subset's wrist videos")
    p.add_argument("--raw", type=Path, required=True, help="raw cache directory")
    p.add_argument("--heldout-spec", type=Path, help="private JSON with the held-out episode ids (key 'heldout')")
    p = sub.add_parser("build", help="write the task dataset and the hidden held-out set")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--station", type=Path, required=True, help="station directory (yam_bimanual_empty.xml, assets)")
    p.add_argument("--tracker", type=Path, required=True, help="ground_truth.json and trajectories.npz directory")
    p.add_argument("--task", type=Path, required=True, help="agent-facing dataset directory")
    p.add_argument("--private", type=Path, required=True, help="hidden directory (held-out set, verifier copies)")
    p.add_argument("--arm-logs", type=Path, help="abc_arm training logs to copy into arm_logs/")
    p.add_argument("--overlays", type=Path, help="directory for check images and the quality table")
    p.add_argument(
        "--heldout-spec",
        type=Path,
        help="private JSON with the held-out episode ids and their checks by hand (default PRIVATE/heldout_spec.json)",
    )
    args = parser.parse_args()
    if args.stage == "extract":
        extract_mcap(args.mcap, args.raw)
        heldout = json.loads(args.heldout_spec.read_text())["heldout"] if args.heldout_spec else []
        extract_lerobot(args.lerobot, [*SIBLINGS.values(), *heldout], args.raw, args.lerobot_wrist)
        if args.lerobot_val is not None:
            extract_lerobot(args.lerobot_val, [MAIN], args.raw)
    else:
        if args.heldout_spec is None:
            args.heldout_spec = args.private / "heldout_spec.json"
        build(args)


if __name__ == "__main__":
    main()
