# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the YAM arm-replay task files from ABC-130k episodes.

Source: the 40 validation episodes of ABC-130k (https://abc.bot/#data) in the
Voxel51 MCAP mirror, and the YAM station MJCF from the ABC repository's
``abc_sim/models``. Every episode yields one CSV per arm (the two arms are the
same model) with rows ``time [s], q1..q6 [rad], qd1..qd6 [rad/s], tau1..tau6
[N m], cmd1..cmd6 [rad], grip, grip_cmd`` at the 29.6 Hz log rate; ``cmd`` is
the teleoperation target streamed to the arm. Eight episodes are held out.

Needs ``mcap`` and ``mcap-protobuf-support`` (not Newton dependencies).

Usage: ``python prepare_data.py MCAP_ROOT ABC_MODELS_DIR TASK_DIR HELDOUT_DIR``
"""

import copy
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

HELDOUT_EPISODES = 8
ARM_MESHES = [f"model2{suffix}.stl" for suffix in ["", *(f"__{i}" for i in range(2, 18))]]


def read_episode(path: Path) -> dict:
    topics = [
        f"/{side}-{kind}" for side in ("left", "right") for kind in ("arm-state", "arm-action", "ee-state", "ee-action")
    ]
    messages = {topic: [] for topic in topics}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
            row = list(decoded.position)
            if channel.topic.endswith("arm-state"):
                row = row[:6] + list(decoded.velocity)[:6] + list(decoded.torque)[:6]
            messages[channel.topic].append((message.log_time, row[:18] if len(row) > 6 else row[:6]))
    return messages


def aligned(messages, ticks):
    messages = sorted(messages)
    times = np.array([m[0] for m in messages], dtype=np.int64)
    index = np.clip(np.searchsorted(times, ticks, side="right") - 1, 0, len(times) - 1)
    return np.array([messages[i][1] for i in index], dtype=np.float64)


def arm_rows(messages: dict, side: str) -> np.ndarray:
    state = sorted(messages[f"/{side}-arm-state"])
    ticks = np.array([m[0] for m in state], dtype=np.int64)
    columns = [
        ((ticks - ticks[0]) * 1e-9)[:, None],
        np.array([m[1] for m in state], dtype=np.float64),
        aligned(messages[f"/{side}-arm-action"], ticks),
        aligned(messages[f"/{side}-ee-state"], ticks)[:, :1],
        aligned(messages[f"/{side}-ee-action"], ticks)[:, :1],
    ]
    return np.concatenate(columns, axis=1)


def single_arm_mjcf(station: Path, out: Path) -> None:
    """Write the station's left arm, mounted at the origin, as a standalone MJCF."""
    root = ET.parse(station).getroot()
    arm = copy.deepcopy(root.find(".//body[@name='left_arm']"))
    arm.set("pos", "0 0 0")
    mujoco = ET.Element("mujoco", {"model": "yam_arm"})
    for tag in ("compiler", "option", "default"):
        mujoco.append(copy.deepcopy(root.find(tag)))
    mujoco.find("compiler").set("meshdir", "meshes")
    asset = ET.SubElement(mujoco, "asset")
    for element in root.find("asset"):
        if element.tag == "material" or (element.tag == "mesh" and element.get("file", "").startswith("model2")):
            asset.append(copy.deepcopy(element))
    asset.append(ET.Element("mesh", {"name": "camera_d405", "file": "d405.stl"}))
    worldbody = ET.SubElement(mujoco, "worldbody")
    worldbody.append(arm)
    equality = ET.SubElement(mujoco, "equality")
    equality.append(
        ET.Element("joint", {"joint1": "left_left_finger", "joint2": "left_right_finger", "polycoef": "0 -1 0 0 0"})
    )
    actuator = ET.SubElement(mujoco, "actuator")
    for element in root.find("actuator"):
        if element.get("name", "").startswith("left_"):
            actuator.append(copy.deepcopy(element))
    ET.indent(mujoco)
    out.write_text(ET.tostring(mujoco, encoding="unicode") + "\n")


def main() -> None:
    source, models, task, heldout = (Path(p) for p in sys.argv[1:5])
    episodes = sorted(source.rglob("*.mcap"))
    order = list(range(len(episodes)))
    random.Random(0).shuffle(order)
    held = set(order[:HELDOUT_EPISODES])
    columns = ["time"] + [f"{name}{j}" for name in ("q", "qd", "tau") for j in range(1, 7)]
    header = ",".join([*columns, *(f"cmd{j}" for j in range(1, 7)), "grip", "grip_cmd"])
    (task / "logs").mkdir(parents=True, exist_ok=True)
    heldout.mkdir(parents=True, exist_ok=True)
    total = {False: 0.0, True: 0.0}
    for number, path in enumerate(episodes):
        name = path.parent.parent.name[:40]
        messages = read_episode(path)
        directory = heldout if number in held else task / "logs"
        for side in ("left", "right"):
            rows = arm_rows(messages, side)
            np.savetxt(
                directory / f"{number:02d}_{side}_{name}.csv",
                rows,
                delimiter=",",
                header=header,
                comments="",
                fmt="%.6g",
            )
            total[number in held] += rows[-1, 0]
    single_arm_mjcf(models / "yam_bimanual_empty.xml", task / "yam_arm.xml")
    (task / "meshes").mkdir(exist_ok=True)
    for name in [*ARM_MESHES, "d405.stl"]:
        shutil.copyfile(models / "assets" / "i2rt_yam" / "assets" / name, task / "meshes" / name)
    print(
        f"{len(episodes) - len(held)} train episodes ({total[False]:.0f} s per arm), {len(held)} held out ({total[True]:.0f} s)"
    )


if __name__ == "__main__":
    main()
