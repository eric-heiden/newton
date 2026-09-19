# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render saved simulation states to a timestamped PNG sequence, without rerunning dynamics."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from .rollout import camera, make_session
from .scenarios import Scenario


def main() -> None:
    """Create portable recorded-state images and an explicit playback manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    trace = np.load(args.metrics.with_suffix(".npz"))
    scenario = Scenario(metrics["scenario"], metrics["config"], variant=metrics["variant"])
    session = make_session(scenario, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    interval = int(trace["trace_step_interval"])
    for index in range(0, len(trace["body_q"]), args.stride):
        session.state.body_q.assign(trace["body_q"][index])
        session.frame = (index + 1) * interval
        session.time = session.frame * float(trace["dt"])
        result = session.dispatch(
            "observe", {**camera(metrics["scenario"]), "width": args.width, "height": args.height}
        )
        name = f"frame-{len(frames):04d}.png"
        (args.output / name).write_bytes(base64.b64decode(result.pop("image_base64")))
        frames.append({"file": name, "simulation_time_s": session.time})
    manifest = {
        "source": str(args.metrics),
        "type": "rendered saved physical rollout, no dynamics rerun",
        "fps": 1 / (float(trace["dt"]) * interval * args.stride),
        "camera": camera(metrics["scenario"]),
        "frames": frames,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": len(frames), "fps": manifest["fps"]}))


if __name__ == "__main__":
    main()
