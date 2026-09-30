# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the ABC station-twin task files from one ABC-130k episode.

Source: the "place the plates into the plastic bin on the countertop" episode
(``episode_6751bfe7``) of ABC-130k (https://abc.bot/#data), Voxel51 MCAP
mirror, and the YAM station MJCF from the ABC repository's ``abc_sim/models``.
The first 43 top-camera frames (1.4 s) are used: the arms move, the objects do not.
Even frames 0, 8, ..., 40 go to the agent; frames 4, 12, ..., 36 are held out.

Needs ``mcap``, ``mcap-protobuf-support``, ``av``, and ``Pillow`` (not Newton dependencies).

Usage: ``python prepare_data.py EPISODE_MCAP ABC_MODELS_DIR TASK_DIR HELDOUT_DIR``
"""

import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

WINDOW = 43
AGENT_FRAMES = list(range(0, WINDOW, 8))
HELDOUT_FRAMES = list(range(4, WINDOW, 8))
STATION_MESHES = [f"model2{suffix}.stl" for suffix in ["", *(f"__{i}" for i in range(2, 18))]]
STATION_MESHES += ["base_visual_gate.stl", "d405.stl"]


def read_episode(path: Path):
    video, arms, grippers, info = [], {}, {}, None
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        topics = [c.topic for c in reader.get_summary().channels.values() if not c.topic.endswith(".plot")]
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            if topic == "/top-camera":
                video.append((message.log_time, decoded.format, bytes(decoded.data)))
            elif topic == "/top-camera-info":
                info = {
                    "width": decoded.width,
                    "height": decoded.height,
                    "distortion_model": decoded.distortion_model,
                    "K": list(decoded.K),
                    "D": list(decoded.D),
                }
            elif topic in ("/left-arm-state", "/right-arm-state"):
                arms.setdefault(topic, []).append((message.log_time, list(decoded.position)[:6]))
            elif topic in ("/left-ee-state", "/right-ee-state"):
                grippers.setdefault(topic, []).append((message.log_time, list(decoded.position)[:1]))
    return video, arms, grippers, info


def aligned(messages, ticks):
    messages = sorted(messages)
    times = np.array([m[0] for m in messages], dtype=np.int64)
    index = np.clip(np.searchsorted(times, ticks, side="right") - 1, 0, len(times) - 1)
    return np.array([messages[i][1] for i in index], dtype=np.float64)


def main() -> None:
    from PIL import Image

    episode, models, task, heldout = (Path(p) for p in sys.argv[1:5])
    video, arms, grippers, info = read_episode(episode)
    ticks = np.array([v[0] for v in video[:WINDOW]], dtype=np.int64)
    codec = av.CodecContext.create({"h265": "hevc"}.get(video[0][1], video[0][1]), "r")
    frames = []
    for _, _, data in video:
        for packet in codec.parse(data):
            frames.extend(codec.decode(packet))
        if len(frames) >= WINDOW:
            break
    images = np.stack([frame.to_ndarray(format="rgb24") for frame in frames[:WINDOW]])
    q = np.concatenate(
        [
            aligned(arms["/left-arm-state"], ticks),
            aligned(grippers["/left-ee-state"], ticks),
            aligned(arms["/right-arm-state"], ticks),
            aligned(grippers["/right-ee-state"], ticks),
        ],
        axis=1,
    )
    time_s = (ticks - ticks[0]) * 1e-9

    (task / "frames").mkdir(parents=True, exist_ok=True)
    heldout.mkdir(parents=True, exist_ok=True)
    for i in AGENT_FRAMES:
        Image.fromarray(images[i]).save(task / "frames" / f"top_f{i:05d}.png")
    np.savez(task / "joint_log.npz", frame=np.arange(WINDOW), time=time_s, q=q)
    np.savez_compressed(
        heldout / "heldout.npz", frame=np.asarray(HELDOUT_FRAMES), image=images[HELDOUT_FRAMES], q=q[HELDOUT_FRAMES]
    )
    (task / "camera.json").write_text(json.dumps(info, indent=2) + "\n")

    station = task / "station"
    mesh_dir = station / "assets" / "i2rt_yam" / "assets"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(models / "yam_bimanual_empty.xml", station / "yam_bimanual_empty.xml")
    for name in STATION_MESHES:
        shutil.copyfile(models / "assets" / "i2rt_yam" / "assets" / name, mesh_dir / name)
    print(f"{len(AGENT_FRAMES)} agent frames, {len(HELDOUT_FRAMES)} held-out frames, {time_s[-1]:.2f} s window")


if __name__ == "__main__":
    main()
