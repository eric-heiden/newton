"""Simulate the calibration scene with parameters from a JSON file and compare with the reference photos.

Usage:
    python sim.py [--params params.json] [--episodes EP1,EP2] [--out out]

For each training episode this renders every reference camera at every
reference time, saves the frames to --out, and writes
--out/compare_<episode>.png with rows per camera: simulated, reference, and
mismatch (magenta = pixels differing by more than 24/255). It prints a JSON
summary with pixel statistics. Edit params.json (or pass another file) and
rerun. The task API (tools.mcp_evaluation.visual) can also be used directly
from your own scripts.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, __ROOT__)

from tools.mcp_evaluation.visual.common import contact_sheet, load_png, mismatch, save_png
from tools.mcp_evaluation.visual.tasks import task_class

TASK = __TASK__
HERE = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, default=HERE / "params.json")
    parser.add_argument("--episodes", default=None, help="Comma-separated training episodes (default: all)")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    cls = task_class(TASK)
    params = json.loads(args.params.read_text()) if args.params.exists() else {}
    task = cls(params)
    built = time.perf_counter()
    episodes = args.episodes.split(",") if args.episodes else list(cls.TRAIN_EPISODES)
    summary = {"params": task.params, "episodes": {}}
    for episode in episodes:
        task.set_episode(episode)
        frames = task.rollout()
        rows, labels, stats = [], [], []
        for camera in cls.CAMERAS:
            sim_row, ref_row, diff_row = [], [], []
            for t in sorted(frames):
                name = f"{episode}_{camera.name}_t{t:.2f}.png"
                simulated = frames[t][camera.name]
                save_png(args.out / name, simulated)
                reference = load_png(HERE / "reference" / name)
                panel, s = mismatch(simulated, reference)
                stats.append({"camera": camera.name, "time_s": t, **s})
                sim_row.append(simulated)
                ref_row.append(reference)
                diff_row.append(panel)
            rows += [sim_row, ref_row, diff_row]
            labels += [
                [f"SIM {camera.name} t={t:.2f}" for t in sorted(frames)],
                [f"REFERENCE {camera.name} t={t:.2f}" for t in sorted(frames)],
                [f"MISMATCH {100 * x['mismatch_fraction']:.1f}%" for x in stats[-len(frames) :]],
            ]
        sheet = args.out / f"compare_{episode}.png"
        save_png(sheet, contact_sheet(rows, labels))
        summary["episodes"][episode] = {
            "compare_image": str(sheet),
            "mean_mismatch_fraction": round(sum(x["mismatch_fraction"] for x in stats) / len(stats), 5),
            "frames": stats,
        }
    summary["seconds"] = {
        "startup_and_build": round(built - started, 2),
        "total": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
