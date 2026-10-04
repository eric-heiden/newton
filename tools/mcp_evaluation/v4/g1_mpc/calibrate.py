# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the ``g1_mpc`` gates from verifier runs of the baseline and earlier G1 controllers.

``prepare`` writes calibration submissions under ``~/.newton-visual-private/g1_mpc/candidates``: the
starter's baseline, and the controllers of passing ``g1_hard``/``g1_track`` loop trials (i6 to i8), each
wrapped unchanged behind the g1_mpc ``Controller`` interface. A candidate's own Example builds its model and
solver, which serve only as its planning copy: its physics never steps (``solver.step`` is a no-op), and the
plant's state is copied into its state (joint_q, joint_qd, forward kinematics) and its MuJoCo data before
every update. Its servo gains and targets become the command.

Timing: the frame-style controllers (one target per 20 ms frame) run at their own rate, the command held for
two 10 ms control periods, which reproduces their original loop exactly. The whole-body controllers, which
updated every 4 ms inside the frame, update every 10 ms (the task's fixed control rate).

``run`` verifies each submission (``verify.py``, every clip in a fresh process), including the private
reference (``~/.newton-visual-private/g1_mpc/reference``, a sampling-based MPC that shows the dynamic clips are
feasible). ``native`` runs each earlier controller in its own original loop and physics (the same plant,
unchanged code, 50 Hz frames) on every clip and scores it with the starter's tracking report at the frame
times; it isolates the effect of the 100 Hz interface. ``table`` prints per-clip metrics of both; ``propose``
derives thresholds from the reference (:data:`GATE_RULE`) and prints every submission's verdicts.

Usage::

    python -m tools.mcp_evaluation.v4.g1_mpc.calibrate prepare
    python -m tools.mcp_evaluation.v4.g1_mpc.calibrate run [--only NAME ...] [--out DIR]
    python -m tools.mcp_evaluation.v4.g1_mpc.calibrate native [--only NAME ...] [--out DIR]
    python -m tools.mcp_evaluation.v4.g1_mpc.calibrate table [--out DIR]
    python -m tools.mcp_evaluation.v4.g1_mpc.calibrate propose [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "g1_mpc"
CANDIDATES_DIR = PRIVATE / "candidates"
LOOP = Path(os.environ.get("NEWTON_LOOP", "/home/horde/artifacts/newton-live-mcp-v4/loop"))
OUT = Path(os.environ.get("NEWTON_G1_MPC_CALIBRATION", "/home/horde/artifacts/newton-live-mcp-v5/research/g1_mpc"))
CLIPS = ("walk", "dance", "jumpjack", "wave", "high5")  # recorded clips (native runs); verify.py adds walk_slow

# Earlier passing controllers: name -> (trial, style, per-update call). "frame": their step() computes the
# target of one 20 ms frame; "update": their whole-body controller's own update call (every 4 ms originally).
CANDIDATES = {
    "i8_hard_astra_mcp_p0": ("i8/g1_hard-astra-mcp-p0", "update", "return ex.controller.compute(t)"),
    "i7_hard_astra_restart_p0": ("i7/g1_hard-astra-restart-p0", "update", "ex.controller.update(t)"),
    "i6_hard_astra_restart_p0": ("i6/g1_hard-astra-restart-p0", "update", "ex.controller.update(t)"),
    "i6_hard_astra_mcp_p1": ("i6/g1_hard-astra-mcp-p1", "update", "ex._controller(t)"),
    "i8_hard_opus_mcp_p0": ("i8/g1_hard-opus-mcp-p0", "frame", "ex.step()"),
    "i7_hard_opus_mcp_p1": ("i7/g1_hard-opus-mcp-p1", "frame", "ex.step()"),
    "i8_hard_opus_mcp_p1": ("i8/g1_hard-opus-mcp-p1", "frame", "ex.step()"),
    "i8_hard_opus_restart_p0": ("i8/g1_hard-opus-restart-p0", "frame", "ex.step()"),
    "i8_track_opus_restart_p1": ("i8/g1_track-opus-restart-p1", "frame", "ex.step()"),
}

ADAPTER = """

# ---------------------------------------------------------------------------------------------------------
# Calibration adapter: the controller of an earlier g1_track/g1_hard submission (cand_track.py, unchanged)
# under the g1_mpc interface. Its Example builds its own model and solver, used only as its planning copy:
# their physics never steps, and the plant's state is copied in before every update.
import newton.viewer  # noqa: E402

import cand_track as _cand  # noqa: E402

STYLE = "{style}"


def _update(ex, t):
    {update}


class MotionClip(MotionClip):  # noqa: F811
    def __init__(self, path, model=None, fps: float = 30.0):
        super().__init__(path, model, fps)
        self.path = str(path)


class Controller:  # noqa: F811
    def __init__(self, model, motion):
        parser = _cand.Example.create_parser()
        args, _ = parser.parse_known_args(["--motion", motion.path])
        args.viewer = "null"
        self.ex = ex = _cand.Example(newton.viewer.ViewerNull(num_frames=1 << 30), args)
        ex.graph = None
        ex.solver.step = lambda *args, **kwargs: None
        # Frame-style controllers keep their 20 ms frame (command held for two periods), as in their loop.
        self.hold = max(1, round(ex.frame_dt / CONTROL_DT)) if STYLE == "frame" else 1
        self.command = None

    def compute(self, t, joint_q, joint_qd):
        if self.command is not None and round(t / CONTROL_DT) % self.hold:
            return self.command
        ex = self.ex
        ex.state_0.joint_q.assign(joint_q)
        ex.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(ex.model, ex.state_0.joint_q, ex.state_0.joint_qd, ex.state_0)
        ex.solver._update_mjc_data(ex.solver.mjw_data, ex.model, ex.state_0)
        ex.sim_time = t
        target = _update(ex, t)
        if target is None:
            target = ex.control.joint_target_q.numpy()
        kp = ex.model.joint_target_ke.numpy()[6:].astype(np.float64)
        kd = ex.model.joint_target_kd.numpy()[6:].astype(np.float64)
        self.command = Command(q=np.asarray(target, dtype=np.float64)[7:], kp=kp, kd=kd)
        return self.command
"""


def _starter_without_main() -> str:
    text = (HERE / "g1_mpc.py").read_text()
    return text[: text.index('\nif __name__ == "__main__":')]


def prepare() -> None:
    """Write the baseline and the adapted earlier controllers as g1_mpc submissions."""
    clips = [HERE / f"{name}.csv" for name in ("walk", "dance", "jumpjack")]
    baseline = CANDIDATES_DIR / "baseline"
    shutil.rmtree(baseline, ignore_errors=True)
    baseline.mkdir(parents=True)
    shutil.copy2(HERE / "g1_mpc.py", baseline / "g1_mpc.py")
    for clip in clips:
        shutil.copy2(clip, baseline / clip.name)
    starter = _starter_without_main()
    for name, (trial, style, update) in CANDIDATES.items():
        target = CANDIDATES_DIR / name
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        shutil.copy2(LOOP / trial / "g1_track.py", target / "cand_track.py")
        (target / "g1_mpc.py").write_text(starter + ADAPTER.format(style=style, update=update))
        for clip in clips:
            shutil.copy2(clip, target / clip.name)
        (target / "SOURCE.txt").write_text(f"{LOOP / trial / 'g1_track.py'}\nstyle={style}\nupdate={update}\n")
    print(f"prepared {len(CANDIDATES) + 1} submissions in {CANDIDATES_DIR}")


def submissions() -> dict[str, Path]:
    out = {path.parent.name: path for path in sorted(CANDIDATES_DIR.glob("*/g1_mpc.py"))}
    reference = PRIVATE / "reference" / "g1_mpc.py"
    if reference.exists():
        out["reference"] = reference
    return out


def run(names: list[str] | None, out: Path, clips: list[str] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, script in submissions().items():
        if names and name not in names:
            continue
        output = out / f"{name}.json"
        command = [sys.executable, str(HERE / "verify.py"), str(script), "--output", str(output)]
        if clips:
            command += ["--clips", *clips]
        started = time.perf_counter()
        # As in a trial: Newton from this tree (the venv's own path entry may point elsewhere).
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        result = subprocess.run(command, capture_output=True, text=True, cwd=script.parent, env=env, check=False)
        (out / f"{name}.log").write_text(result.stdout + result.stderr)
        print(f"{name}: rc={result.returncode} {time.perf_counter() - started:.0f} s", flush=True)


# ----------------------------------------------------------------------------- native runs


def native_child(candidate: Path, clip: str, output: Path) -> None:
    """One earlier controller, unchanged, in its own loop and physics on one clip (run in a fresh process)."""
    import types  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton.viewer  # noqa: PLC0415

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(candidate))
    import cand_track  # noqa: PLC0415
    import g1_mpc as plant  # noqa: PLC0415

    wp.config.log_level = wp.LOG_WARNING
    path = PRIVATE / "clips" / f"{clip}.csv"
    parser = cand_track.Example.create_parser()
    args, _ = parser.parse_known_args(["--motion", str(path)])
    args.viewer = "null"
    started = time.perf_counter()
    ex = cand_track.Example(newton.viewer.ViewerNull(num_frames=1 << 30), args)
    setup = time.perf_counter() - started
    reference = plant.MotionClip(path, plant.build_model())
    report = plant.TrackingReport(types.SimpleNamespace(model=ex.model, solver=ex.solver), reference)
    frames = round(reference.duration / ex.frame_dt)
    started = time.perf_counter()
    for _ in range(frames):
        ex.step()
        if not report.update(ex.sim_time, ex.state_0):
            break
    rollout = time.perf_counter() - started
    summary = report.summary()
    # The report's jitter assumes 10 ms samples; these are frame samples.
    errors = np.array([row["joint_err"] for row in report.rows]) if summary["upright"] else None
    summary["jitter_mrad"] = 1000.0 * plant.jitter(errors, ex.frame_dt) if errors is not None else None
    summary["expected_samples"] = frames
    result = {"clip": clip, "metrics": summary, "timing": {"setup_s": setup, "rollout_s": rollout}}
    output.write_text(json.dumps(_jsonable(result), indent=1) + "\n")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else str(number)


def native(names: list[str] | None, out: Path, clips: list[str] | None = None) -> None:
    out = out / "native"
    out.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATES:
        if names and name not in names:
            continue
        results = {}
        for clip in clips or CLIPS:
            output = out / f"{name}.{clip}.json"
            command = [sys.executable, "-m", "tools.mcp_evaluation.v4.g1_mpc.calibrate", "native-child"]
            command += ["--candidate", str(CANDIDATES_DIR / name), "--clip", clip, "--output", str(output)]
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=env, check=False)
            if result.returncode != 0 or not output.exists():
                results[clip] = {"error": (result.stderr or result.stdout)[-1500:]}
            else:
                results[clip] = json.loads(output.read_text())
                output.unlink()
        (out / f"{name}.json").write_text(json.dumps(results, indent=1) + "\n")
        print(f"{name}: native done", flush=True)


# ----------------------------------------------------------------------------- tables


def _fmt(value, digits=3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, str):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return f"{value:.{digits}f}"


def _num(value):
    if value is None or isinstance(value, bool):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


COLUMNS = (
    ("survival", 2),
    ("root_rmse_m", 3),
    ("root_rot_rmse_deg", 1),
    ("joint_rmse_rad", 3),
    ("wrist_rmse_m", 3),
    ("sole_rmse_m", 3),
    ("lift_recall", 2),
    ("jitter_mrad", 1),
)


def load_results(out: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(out.glob("*.json"))}


def _row(name: str, clip: str, passed, metrics: dict, timing: dict, error=None) -> str:
    row = f"{name:26s} {clip:9s} {_fmt(passed):4s} "
    row += " ".join(f"{_fmt(_num(metrics.get(k)), d):>12s}" for k, d in COLUMNS)
    row += f" {_fmt(_num(timing.get('compute_s')), 1):>7s} {_fmt(_num(timing.get('rollout_s')), 1):>7s}"
    if error:
        row += f"  error: {str(error)[-160:]!r}"
    elif metrics.get("fall_reason"):
        row += f"  fell at {_fmt(_num(metrics.get('fall_time_s')), 2)} s ({metrics['fall_reason']})"
    return row


def table(out: Path) -> None:
    header = f"{'submission':26s} {'clip':9s} {'pass':4s} " + " ".join(f"{k[:12]:>12s}" for k, _ in COLUMNS)
    header += f" {'ctrl_s':>7s} {'roll_s':>7s}"
    print("verify.py (g1_mpc interface, 100 Hz)")
    print(header)
    for name, result in load_results(out).items():
        for clip, detail in (result.get("details") or {}).items():
            timing = (detail.get("details") or {}).get("timing") or {}
            m = detail.get("metrics") or {}
            print(_row(name, clip, bool(detail.get("success")), m, timing, detail.get("error")))
        metrics = result.get("metrics") or {}
        print(
            f"{name:26s} {'TOTAL':9s} {_fmt(bool(result.get('success'))):4s} clips passed "
            f"{metrics.get('clips_passed')}/{len(result.get('details') or {})}, score {_fmt(_num(metrics.get('score')))}"
        )
    natives = load_results(out / "native") if (out / "native").exists() else {}
    if natives:
        print("\nnative (unchanged original loop and physics, 50 Hz frames; jitter from frame samples)")
        print(header)
        for name, results in natives.items():
            for clip, r in results.items():
                print(_row(name, clip, None, r.get("metrics") or {}, r.get("timing") or {}, r.get("error")))


# Gate rule: threshold = factor x the private reference's worst clip, rounded up to the resolution. The factors
# leave room for a less tuned controller of the same class: the reference's own variants (15 Hz arm servos, stiffer
# legs, other seeds) reached joint RMSE up to 0.10 rad, sole RMSE up to 0.067 m, and root tilt RMSE up to 8 degrees
# on walk while tracking it, whereas controllers that keep their feet planted have sole RMSE of 0.11 m or more.
GATE_RULE = {
    "root_rmse_m": (2.5, 0.01),
    "root_rot_rmse_deg": (2.3, 1.0),
    "joint_rmse_rad": (1.5, 0.01),
    "sole_rmse_m": (1.5, 0.01),
}


def _clip_passes(detail: dict, thresholds: dict) -> bool:
    """A clip's verdict under other thresholds, recomputed from its metrics (integrity must hold)."""
    metrics, integrity = detail.get("metrics") or {}, detail.get("integrity") or {}
    if detail.get("error") or not integrity or not all(integrity.values()) or metrics.get("upright") is not True:
        return False
    return all(
        isinstance(_num(metrics.get(k)), float) and math.isfinite(_num(metrics[k])) and _num(metrics[k]) <= v
        for k, v in thresholds.items()
    )


def propose(out: Path) -> dict:
    """Thresholds from the reference (:data:`GATE_RULE`), then every submission's verdict per clip under them
    and under verify.THRESHOLDS."""
    sys.path.insert(0, str(ROOT))
    from tools.mcp_evaluation.v4.g1_mpc.verify import THRESHOLDS  # noqa: PLC0415

    results = load_results(out)
    reference = results.get("reference")
    proposal = {}
    if reference is not None:
        for key, (factor, step) in GATE_RULE.items():
            values = [_num((d.get("metrics") or {}).get(key)) for d in reference["details"].values()]
            worst = max(v for v in values if isinstance(v, float))
            proposal[key] = round(math.ceil(factor * worst / step - 1e-9) * step, 6)
            print(
                f"{key:18s} reference worst {worst:.4f} -> proposed {proposal[key]:g} (verify.py: {THRESHOLDS[key]:g})"
            )
    for label, thresholds in (("proposed", proposal), ("verify.py", THRESHOLDS)):
        if not thresholds:
            continue
        print(f"\nverdicts under the {label} thresholds {thresholds}")
        for name, result in results.items():
            verdicts = {clip: _clip_passes(d, thresholds) for clip, d in (result.get("details") or {}).items()}
            row = " ".join(f"{clip}={'pass' if ok else 'FAIL'}" for clip, ok in verdicts.items())
            print(f"  {name:26s} {sum(verdicts.values())}/{len(verdicts)}  {row}")
    return proposal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("prepare", "run", "native", "native-child", "table", "propose"))
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--out", type=Path, default=OUT / "calibration")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--clip")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.only, args.out, args.clips)
    elif args.action == "native":
        native(args.only, args.out, args.clips)
    elif args.action == "native-child":
        native_child(args.candidate, args.clip, args.output)
    elif args.action == "table":
        table(args.out)
    else:
        propose(args.out)


if __name__ == "__main__":
    main()
