# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run one harness-improvement-loop trial: a Newton script task with or without the live MCP.

In the ``mcp`` condition the workspace script runs live under
``python -m newton.mcp host`` and the agent also keeps its shell; in ``restart``
the agent edits and reruns scripts. The submitted script is verified in a fresh
process after the agent exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from tools.mcp_evaluation.v4 import trial_isolation as ti
from tools.mcp_evaluation.visual.run_visual import (
    MODELS,
    UNAVAILABLE,
    _agent_command,
    _stop,
    digest,
    mcp_available,
    parse_events,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"
# MCP tool profile: "lean" advertises only execute and rebuild to keep per-turn context small.
PROFILE = os.environ.get("NEWTON_MCP_PROFILE", "lean")
# Sibling live copies the MCP condition may use for parallel sweeps.
WORKERS = int(os.environ.get("NEWTON_MCP_WORKERS", "2"))
CUBE_DATA = Path(os.environ.get("NEWTON_CUBE_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/cube_toss_task"))
DP_DATA = Path(os.environ.get("NEWTON_DP_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/dp_real_task"))
ABC_DATA = Path(os.environ.get("NEWTON_ABC_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/abc_twin_task"))
ARM_DATA = Path(os.environ.get("NEWTON_ARM_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/abc_arm_task"))
LOOK_DATA = Path(os.environ.get("NEWTON_LOOK_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/abc_look_task"))
# Blender for the look task and the MCP's blender backend (inherited by hosts, agents, and verifiers).
os.environ.setdefault("NEWTON_BLENDER", "/home/horde/opt/blender-5.2.2-linux-x64/blender")
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private"))
# Agent workspaces get opaque names under a neutral root; the labeled run directory keeps the harness
# records (spec, command, transcript, host log, verification), which agents must not see.
TRIALS = Path(os.environ.get("NEWTON_TRIAL_ROOT", Path.home() / "trials"))
SANDBOX = os.environ.get("NEWTON_TRIAL_SANDBOX", "1") == "1"
START = "__START_UTC__"
# Claude -p waits this long for background shell jobs after the agent's last turn; set explicitly so it does
# not depend on the launcher's environment (v4 trials inherited 30 min from the operator session).
BG_WAIT_CEILING_MS = 1_800_000
"""Placeholder for the budget start time, filled in when the agent launches."""


def _task(name: str) -> dict:
    tasks = {
        "grasp_drift": {
            "files": {"franka_cube_shake.py": HERE / "grasp/franka_cube_shake.py"},
            "script": "franka_cube_shake.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/grasp/verify.py",
            "goal": """franka_cube_shake.py is a Newton simulation (SolverMuJoCo, Newton collision pipeline) of a Franka FR3 that grasps a 4 cm cube, lifts it, and shakes it along a Lissajous path. Problem: the grasp is not reliable. With a 10 cm / 1 Hz shake (`--shake-amplitude 0.1`) the cube slips and is dropped, and smaller shakes also let it creep.

Goal: change the scene so the grasp holds. Verification runs the script's Example in a fresh process and measures the cube position in the hand (TCP) frame from the start of the shake: over 11.5 s of 3 cm / 1 Hz shaking it must move less than 1 mm, and over 6 s of 10 cm / 1 Hz shaking it must stay in the hand and move less than 5 mm.
Constraints (checked): do not change the scripted motion (phases, durations, IK, shake path), joint gains or effort limits, gripper commands, the cube's size, density, or mass, gravity, or the timestep (60 Hz frames, 16 substeps), and do not add bodies, joints, or constraints between gripper and cube. Contact, material, collision, and solver settings may change. Keep the file runnable as a Newton example (`python franka_cube_shake.py --viewer null`). Report the root cause you found.""",
        },
        "g1_track": {
            "files": {name: HERE / "g1" / name for name in ("g1_track.py", "wave.csv", "high5.csv")},
            "script": "g1_track.py",
            "host_args": ["--motion", "wave.csv"],
            "verifier": "tools/mcp_evaluation/v4/g1/verify.py",
            "goal": """g1_track.py simulates a floating-base Unitree G1 humanoid (SolverMuJoCo, 50 Hz frames, 2 ms substeps) that should follow a Kimodo reference motion (wave.csv: root pose and 29 joint angles at 30 fps) using position servos with torque limits. Problem: with the current controller the robot falls.

Goal: set up the controller so the robot follows the reference motion without falling. Verification runs the script in a fresh process on wave.csv and on an unseen faster playback of the same clip: the root must stay upright (height above 0.6 m), root position RMSE at most 4 cm, and joint-angle RMSE at most 0.05 rad. It also reports a harder stretch clip (high5.csv) that is not required.
Constraints (checked): do not change the robot model (bodies, masses, armature, torque limits), the timestep, or the motion input and playback, and apply no forces or torques to the floating base. The controller (gains, targets, feedforward, feedback, anything in the step logic) may change. Keep the file runnable as a Newton example (`python g1_track.py --motion wave.csv --viewer null`).""",
        },
        "dp_real": {
            "files": {
                "double_pendulum.py": HERE / "dp_real/double_pendulum.py",
                **{f"train_{i:02d}.csv": DP_DATA / f"train_{i:02d}.csv" for i in range(10)},
            },
            "script": "double_pendulum.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/dp_real/verify.py",
            "goal": """double_pendulum.py models a real robot: the DFKI torque-controlled double pendulum (two quasi-direct-drive AK80-6 motors with 6:1 planetary gearing, one at the shoulder and one at the elbow; shoulder-to-elbow distance 0.2 m, second link 0.3 m). train_00.csv ... train_09.csv are recordings from the real hardware: measured joint angles and velocities and the measured motor torques (about 500 Hz). Problem: the model uses nominal CAD guesses, and its predictions drift quickly away from the recordings.

Goal: calibrate the model so it predicts the real robot. Verification imports the script's build_model(num_worlds) and make_solver(model) and runs its own multiple-shooting evaluation (the same protocol as the script's rollout/window_errors, 2 ms steps) on two held-out recordings of the same robot: a 20 s excitation run and a 75 s run with full swings. Each 0.5 s window starts from the measured state and is driven open loop by the measured torques. The mean joint-angle RMSE over the windows must be at most {heldout_10_rad} rad on the 20 s run and at most {heldout_11_rad} rad on the 75 s run. The starter scores about 0.14 and 0.34 rad.
Constraints (checked): keep the two revolute joints, the 0.2 m shoulder-to-elbow offset, gravity, positive masses and valid inertias, and the build_model/make_solver interface. Any physical parameter (masses, centers of mass, inertias, armature, damping, friction, ...) and the solver settings may change, and other Newton modeling features may be used inside build_model/make_solver. Keep the file runnable (`python double_pendulum.py --viewer null`).""",
        },
        "cube_toss": {
            "files": {"cube_toss.py": HERE / "cube_toss/cube_toss.py", "tosses.npz": CUBE_DATA / "tosses.npz"},
            "script": "cube_toss.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/cube_toss/verify.py",
            "goal": """cube_toss.py simulates real measurements: an acrylic cube (0.1048 m, 0.37 kg, inertia 0.00081 kg m^2, all measured) tossed by hand onto a wooden table and tracked by cameras at 148 Hz (ContactNets dataset). tosses.npz holds 400 recorded tosses (positions, orientations, and world-frame linear and angular velocities per frame). Each toss is one Newton world that starts from the first measured frame and runs open loop through the impacts, bounces, and slides. Problem: with the current contact model the simulated cubes bounce and slide far from the recordings.

Goal: calibrate the simulation so it reproduces the real tosses. Verification imports the script's build_model(num_worlds), make_solver(model), make_pipeline(model), and SUBSTEPS, and runs its own open-loop rollout (the same protocol as the script's rollout/evaluate) on 170 held-out tosses of the same cube. The mean over tosses of the time-averaged position error must be at most {position_m} m, and of the orientation error at most {rotation_rad} rad. The starter scores about 0.10 m and 1.07 rad.
Constraints (checked): keep one cube per world with the measured size, mass, and inertia, and keep gravity. Contact and material parameters, table height, collision settings, solver type and settings, and the number of substeps per frame (1 to 100) may change. Keep the file runnable (`python cube_toss.py --viewer null`).""",
        },
        "g1_hard": {
            "files": {name: HERE / "g1" / name for name in ("g1_track.py", "wave.csv", "high5.csv")},
            "script": "g1_track.py",
            "host_args": ["--motion", "high5.csv"],
            "verifier": "tools/mcp_evaluation/v4/g1/verify_hard.py",
            "seconds": 3600,
            "goal": """g1_track.py simulates a floating-base Unitree G1 humanoid (SolverMuJoCo, 50 Hz frames, 2 ms substeps) that should follow Kimodo reference motions (root pose and 29 joint angles at 30 fps) using position servos with torque limits. Problem: with the current controller the robot falls, and high5.csv, a clip in which the root turns and shifts its weight, defeats plain joint-space PD tuning.

Goal: set up a controller that follows both wave.csv and high5.csv without falling. Verification runs the script in a fresh process on wave.csv, on high5.csv, and on an unseen slower (0.9x) playback of high5.csv: the root must stay upright (height above 0.6 m), root position RMSE at most 4 cm, and joint-angle RMSE at most 0.05 rad on every run. The same controller code must handle every clip (select the clip only through --motion).
Constraints (checked): do not change the robot model (bodies, masses, armature, torque limits), the timestep, or the motion input and playback, and apply no forces or torques to the floating base. The controller (gains, targets, feedforward, feedback, estimation, anything in the step logic, and additional Python packages already installed) may change. Keep the file runnable as a Newton example (`python g1_track.py --motion high5.csv --viewer null`).""",
        },
        "sdf_grind": {
            "files": {"sdf_grinding.py": HERE / "sdf_grind/sdf_grinding.py"},
            "script": "sdf_grinding.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/sdf_grind/verify.py",
            "seconds": 2700,
            "goal": """sdf_grinding.py is a Newton scene for machining: a cylindrical grinding wheel is driven kinematically across an ellipsoidal workpiece (a static mesh shape whose collision geometry is a sparse texture SDF built by Mesh.build_sdf), and hydroelastic SDF-SDF collision reports the contact surface and normal load every frame without a dynamics solver. Problem: the wheel passes through the workpiece without removing any material, so the rendered and colliding workpiece never changes.

Goal: add material removal, a capability Newton does not provide out of the box. As the wheel moves, the material it sweeps through must disappear from the workpiece's collision geometry (the SDF attached to the workpiece mesh shape), and the rendered surface must follow. Verification runs the script's Example for the full pass (GRIND_FRAMES + 10 frames) in a fresh process and then checks the workpiece SDF: the removed volume must match the volume swept by the wheel within 15%; points inside the groove must be clear and points 2 cm below it must remain solid (at least 90% each); and with the wheel placed back into the finished groove, the hydroelastic normal load must be at most 25% of the load on an unground workpiece.
Constraints (checked): keep the workpiece shape, wheel size, tool path (_grinder_pose), GRIND_DEPTH, GRIND_FRAMES, hydroelastic stiffness, and the collision pipeline setup; the removal must act on the geometry the collision pipeline uses. Performance matters less than correctness, but a full pass should stay under a few minutes. Keep the file runnable (`python sdf_grinding.py --viewer null`).""",
        },
        "abc_twin": {
            "files": {
                "station_twin.py": HERE / "abc_twin/station_twin.py",
                "twin_render.py": HERE / "abc_twin/twin_render.py",
                **{name: ABC_DATA / name for name in ("camera.json", "joint_log.npz", "frames", "station")},
            },
            "script": "station_twin.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/abc_twin/verify.py",
            "seconds": 2700,
            "goal": """station_twin.py is a digital twin of a real robot station from the ABC-130k dataset (https://abc.bot): two YAM arms with parallel grippers in a white enclosure, filmed from above by a RealSense D405. joint_log.npz holds the measured joint positions for 43 frames (30 Hz) of an episode in which the arms move and the objects on the table stay still; frames/ holds six recorded top-camera frames (640x480) of that window, and camera.json the camera's calibrated intrinsics and distortion. twin_render.py poses the robot from the log and renders the model through that camera; the verifier uses the same renderer. Problem: the twin is the nominal CAD station. The camera mount is off, the enclosure does not match the real one, the table is empty, and colors and lighting do not match the recording.

Goal: make the twin reproduce the recording. Verification imports build_model, CAMERA_POSITION, CAMERA_ROTATION, and LOOK, renders 5 held-out frames of the same window (between the given ones) with the robot posed from the log, and scores them against the recorded frames with twin_render.score: the mean edge_ncc must be at least {edge_ncc}, ssim at least {ssim}, and color_psnr_db at least {color_psnr_db}. The starter scores about 0.37, 0.53, and 14.0 dB.
Constraints (checked): keep one world and the station's arm kinematics (link 1 to link 6 of each arm), and add at most 60 shapes. twin_render.py is fixed (the verifier uses its own copy), and build_model runs without the recorded frames. The camera pose, arm base placement, enclosure and table, objects (static shapes; any Newton geometry), shape colors, and LOOK may change. Keep the file runnable (`python station_twin.py --viewer null`).""",
        },
        "abc_arm": {
            "files": {
                "arm_replay.py": HERE / "abc_arm/arm_replay.py",
                **{name: ARM_DATA / name for name in ("yam_arm.xml", "meshes", "logs")},
            },
            "script": "arm_replay.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/abc_arm/verify.py",
            "goal": """arm_replay.py models a real robot arm from the ABC-130k dataset (https://abc.bot): the 6-DoF YAM arm of a bimanual teleoperation station, whose joints track position commands streamed from a leader arm at about 30 Hz. logs/ holds 64 recorded arm logs (32 episodes, both arms; about 87 minutes per arm) with measured joint positions, velocities, and motor torques, the commanded joint positions, and the gripper opening. Each log is cut into 1 s windows; each window is one Newton world that starts from the measured state and is driven open loop by the logged commands through the arm's joint position targets (rollout/window_errors). Problem: the model uses the nominal gains, armature, and friction of the ABC simulation model, and its predicted joint angles drift from the recordings.

Goal: calibrate the arm model so it predicts the real arm. Verification imports build_model(num_worlds), make_solver(model), and PARAMS["command_delay"], and runs its own multiple-shooting evaluation (1 s windows every 0.5 s, 2 ms steps, the logged commands applied as joint position targets after the command delay) on 16 held-out logs of 8 other episodes. The mean joint-angle RMSE must be at most {heldout_rad} rad. The starter scores about {starter_rad} rad.
Constraints (checked): keep one arm per world with the kinematics of yam_arm.xml (joint frames, axes, and types), gravity, nonnegative masses, and valid inertias; the command delay must be between 0 and 0.2 s. Controller gains, effort limits, armature, friction, damping, masses and inertias, gravity compensation, solver settings, and other Newton modeling features inside build_model/make_solver may change. Keep the file runnable (`python arm_replay.py --viewer null`).""",
        },
        "abc_look": {
            "files": {
                **{
                    name: HERE / "abc_look" / name
                    for name in ("station_look.py", "look.py", "render_look.py", "look_common.py")
                },
                "twin_render.py": HERE / "abc_twin/twin_render.py",
                **{
                    name: LOOK_DATA / name
                    for name in ("scene.json", "camera.json", "joint_log.npz", "frames", "station")
                },
            },
            "script": "station_look.py",
            "host_args": [],
            "verifier": "tools/mcp_evaluation/v4/abc_look/verify.py",
            "seconds": 2700,
            "goal": """station_look.py is a fixed digital twin of a real robot station from the ABC-130k dataset (https://abc.bot): two YAM arms in a white enclosure with plates, dishes, and an orange bin on the table, seen by the station's top camera (a RealSense D405). The geometry is fitted already: scene.json holds the camera pose and the objects, joint_log.npz the robot's measured joint positions for 43 frames, frames/ six recorded top-camera frames (640x480), and camera.json the calibrated intrinsics. The twin is rendered in Blender EEVEE (look_common.LookRenderer), and its appearance comes from look.py, bpy code that runs once in Blender after the scene is built. render_look.py renders frames with the current look.py in a fresh Blender process and scores them. Problem: look.py is empty, so every object on the table is neutral gray, and materials, lights, and exposure are Blender defaults.

Goal: write look.py so that the renders match the recording. Verification runs only look.py, in a fresh Blender worker on the same twin, renders 5 held-out frames of the same window (between the given ones), and compares them with the recorded frames: the mean color difference over shape regions (look_common.region_color_error, sRGB 0-255) must be at most {region_color_error}, and the color PSNR (twin_render.score) at least {color_psnr_db} dB. The starter scores about 46 and 17.6 dB.
Constraints (checked): appearance only. Materials, lights, world, and color management may change; adding or reshaping mesh objects, loading images, and compositing are not allowed. The geometry files, look_common.py, and twin_render.py are fixed (the verifier uses its own copies). Keep render_look.py working.""",
        },
    }
    if name == "abc_look":
        from tools.mcp_evaluation.v4.abc_look.verify import THRESHOLDS  # noqa: PLC0415

        tasks[name]["goal"] = tasks[name]["goal"].format(**THRESHOLDS)
    if name == "abc_arm":
        from tools.mcp_evaluation.v4.abc_arm.verify import STARTER_RAD, THRESHOLDS  # noqa: PLC0415

        tasks[name]["goal"] = tasks[name]["goal"].format(starter_rad=STARTER_RAD, **THRESHOLDS)
    if name == "abc_twin":
        from tools.mcp_evaluation.v4.abc_twin.verify import THRESHOLDS  # noqa: PLC0415

        tasks[name]["goal"] = tasks[name]["goal"].format(**THRESHOLDS)
    if name == "cube_toss":
        from tools.mcp_evaluation.v4.cube_toss.verify import THRESHOLDS  # noqa: PLC0415

        tasks[name]["goal"] = tasks[name]["goal"].format(**THRESHOLDS)
    if name == "dp_real":
        from tools.mcp_evaluation.v4.dp_real.verify import THRESHOLDS  # noqa: PLC0415

        tasks[name]["goal"] = tasks[name]["goal"].format(**THRESHOLDS)
    task = tasks[name]
    # Commands run once on the starter to build the compile-cache seed both conditions start from.
    task.setdefault(
        "warmup",
        [
            [task["script"], "--viewer", "null", "--num-frames", "1", *task["host_args"]],
            # Warp caches kernels per module name: also warm the MCP host's and an importer's names.
            ["-m", "tools.mcp_evaluation.v4.trial_isolation", "warm", task["script"], *task["host_args"]],
        ],
    )
    task.setdefault("private", [name])
    if name == "abc_look":
        task["warmup"].append(["render_look.py", "--frames", "0"])  # EEVEE shaders (GL cache: 23 s cold)
        task["private"] = ["abc_look", "abc_twin"]
    return task


def prompt_for(name: str, condition: str, workspace: Path, seconds: int, guide: str | None) -> str:
    task = _task(name)
    common = f"""You are working on a Newton physics simulation task.

Workspace: {workspace}
Newton source tree (read-only reference, including docs and examples): {ROOT}

{task["goal"]}

Deliverable: the edited {task["script"]} in the workspace, then a brief report. You have {seconds // 60} minutes, starting {START} (check with `date -u`); working efficiently matters. Do not modify files outside the workspace, do not look for other trials or hidden verification data, and do not use subagents.
"""
    run = f"uv run --no-sync --project {ROOT} python {task['script']} --viewer null --num-frames <N> {' '.join(task['host_args'])}".rstrip()
    if condition == "restart":
        return (
            common
            + f"""
Workflow: this is a script-based setup. Run the simulation with
  {run}
or write your own scripts that import the Example class; each run starts a fresh simulator process. There is no display; to look at the scene, render images yourself (for example with newton.sensors.SensorTiledCamera) and open them with your image-viewing tool.
"""
        )
    return (
        common
        + f"""
Workflow: {task["script"]} is already running live, hosted by the Newton MCP (`newton` server tools, e.g. newton_execute and newton_rebuild). Images returned by MCP tools appear directly in your context, and Python state persists between calls. After editing the script on disk, newton_rebuild reloads it in the live process. You may also run scripts ({run}) when that is more efficient; each such run starts a fresh process.
If no newton tools are available to you, reply only with {UNAVAILABLE} and stop.
{guide or ""}
"""
    )


def prepare(run_dir: Path, name: str, condition: str, model: str, seconds: int | None, phase: str) -> dict:
    """Lay out one trial: harness records in ``run_dir``, the agent's files under an opaque name."""
    if condition not in ("mcp", "restart"):
        raise ValueError("condition must be mcp or restart")
    task = _task(name)
    seconds = seconds or task.get("seconds", 1800)
    run_dir.mkdir(parents=True, exist_ok=False)
    trial_id = uuid.uuid4().hex[:12]
    sandbox_root = TRIALS / trial_id
    workspace = sandbox_root / "work"
    try:
        ti.copy_files(task["files"], workspace)
        # Both conditions start from the same compile caches: the starter's, built once per task and commit.
        seed = ti.seed_caches(name, task["files"], task["warmup"], PYTHON, ROOT)
        shutil.copytree(seed, sandbox_root / "caches")
    except BaseException:
        shutil.rmtree(sandbox_root, ignore_errors=True)
        raise
    (run_dir / "workspace").symlink_to(workspace)
    guide = None
    if condition == "mcp":
        guide = _host_guide(workspace, task)
    prompt = prompt_for(name, condition, workspace, seconds, guide)
    spec = {
        "task": name,
        "condition": condition,
        "model": model,
        **MODELS[model],
        "phase": phase,
        "budget_seconds": seconds,
        "trial_id": trial_id,
        "workspace": str(workspace),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "starter_sha256": {
            target: digest(workspace / target) for target in task["files"] if (workspace / target).is_file()
        },
        "cache_seed": str(seed),
        "mcp_workers": WORKERS if condition == "mcp" else None,
        "mcp_profile": PROFILE if condition == "mcp" else None,
        "sandbox": SANDBOX,
        "harness_version": os.environ.get("NEWTON_HARNESS_VERSION", "unversioned"),
        "claude_bg_wait_ceiling_ms": BG_WAIT_CEILING_MS,
    }
    info = ti.provenance(ROOT, PYTHON)
    (run_dir / "harness.diff").write_text(info.pop("diff"))
    spec.update(info)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return {"spec": spec, "prompt": prompt, "run_dir": run_dir}


def _host_guide(workspace: Path, task: dict) -> str:
    """The same application guide the MCP server sends in its instructions."""
    code = (
        "import sys; from newton.mcp import ExampleHost; "
        f"host = ExampleHost({str(workspace / task['script'])!r}, {task['host_args']!r}); "
        f"host.example = type('E', (), {{'frame_dt': '?'}})(); print(host.guide({WORKERS}))"
    )
    result = subprocess.run([str(PYTHON), "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    return result.stdout.strip()


def run_trial(prepared: dict, barrier: Path | None = None, parties: int = 2) -> dict:
    spec, run_dir = prepared["spec"], prepared["run_dir"]
    task = _task(spec["task"])
    trial_id = spec["trial_id"]
    workspace = Path(spec["workspace"])
    sandbox_root = workspace.parent
    (sandbox_root / ".mcp").mkdir()
    env = ti.trial_env(ROOT, sandbox_root / "caches", trial_id)
    env["MCP_TOOL_TIMEOUT"] = "300000"
    env["MAX_MCP_OUTPUT_TOKENS"] = "60000"
    env["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] = str(BG_WAIT_CEILING_MS)

    def contained(command: list[str], extra_ro: list[Path] = ()) -> list[str]:
        # The host runs the agent's code too, so it shares the agent's filesystem view, including /tmp.
        if not SANDBOX:
            return command
        return ti.sandbox(command, sandbox_root, ROOT, PRIVATE, extra_ro=list(extra_ro), extra_hidden=[TRIALS])

    host, sampler, agent = None, None, None
    # Mount namespaces of the trial's sandboxes: cleanup also finds detached jobs that cleared their env.
    namespaces: set[str] = set()

    def remember_namespace(process: subprocess.Popen) -> None:
        if not SANDBOX:
            return
        for _ in range(100):
            namespace = ti.mount_namespace(ti.cli_child(process.pid))
            if namespace is not None:
                namespaces.add(namespace)
                return
            if process.poll() is not None:
                return
            time.sleep(0.1)

    try:
        mcp, startup = None, 0.0
        started = time.time()
        if spec["condition"] == "mcp":
            connection = sandbox_root / ".mcp/connection.json"
            before = time.perf_counter()
            log = (run_dir / "host.log").open("w")
            host = subprocess.Popen(
                contained(
                    [
                        str(PYTHON),
                        *("-m", "newton.mcp", "host", task["script"], "--connection-file", str(connection)),
                        *("--artifacts", str(workspace / "observations"), "--workers", str(WORKERS)),
                        "--",
                        *task["host_args"],
                    ]
                ),
                cwd=workspace,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            remember_namespace(host)
            ready = connection.with_suffix(".ready")
            while not ready.exists():
                if host.poll() is not None or time.perf_counter() - before > 600:
                    raise RuntimeError(f"Host failed to start; see {run_dir / 'host.log'}")
                time.sleep(0.1)
            startup = time.perf_counter() - before
            mcp = {
                "command": str(PYTHON),
                # The task prompt already carries the host guide, so the server instructions omit it.
                "args": [
                    *("-m", "newton.mcp", "--connect", str(connection), "--profile", PROFILE, "--timeout", "300"),
                    "--no-app-guide",
                ],
                # Load the Newton tools up front instead of behind a tool-search round trip (Claude Code only).
                "alwaysLoad": True,
            }
        command = _agent_command(spec, workspace, mcp)
        if "--mcp-config" in command:
            # Inline the MCP config so no harness file (an empty server list in restart) sits in the workspace.
            index = command.index("--mcp-config") + 1
            config = Path(command[index])
            command[index] = config.read_text()
            config.unlink()
        command = contained(command)
        (run_dir / "command.json").write_text(json.dumps(command))
        # Both conditions launch together once both are ready, so the MCP host's startup never falls into the
        # restart agent's budget, and the stated start time is the moment the budget clock starts.
        budget_start = ti.pair_barrier(barrier, spec["condition"], parties) if barrier else time.time()
        prompt = prepared["prompt"].replace(START, time.strftime("%H:%M:%S UTC", time.gmtime(budget_start)))
        (workspace / "TASK.md").write_text(prompt)
        sampler = ti.ResourceSampler(run_dir / "resources.jsonl", trial_id)
        sampler.start()
        agent_start = time.perf_counter()
        timed_out = False
        with (
            (run_dir / "agent.jsonl").open("w") as out,
            (run_dir / "agent.times.jsonl").open("w") as times,
            (run_dir / "agent.stderr").open("w") as err,
        ):
            agent = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                cwd=workspace,
                env=env,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            def pump():
                for number, line in enumerate(agent.stdout):
                    out.write(line)
                    out.flush()
                    times.write(json.dumps({"line": number, "seconds": time.perf_counter() - agent_start}) + "\n")

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            threading.Thread(target=remember_namespace, args=(agent,), daemon=True).start()
            try:
                agent.stdin.write(prompt)
                agent.stdin.close()
                agent.wait(timeout=max(1.0, spec["budget_seconds"] - (time.time() - budget_start)))
            except subprocess.TimeoutExpired:
                timed_out = True
                # Interrupt the CLI itself, not bwrap: bwrap would die at once and take the CLI down before it
                # writes its final usage record.
                child = ti.cli_child(agent.pid) if SANDBOX else agent.pid
                try:
                    os.kill(child or agent.pid, signal.SIGINT)
                    agent.wait(timeout=20)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            finally:
                _stop(agent)
                reader.join(timeout=10)
        elapsed = time.perf_counter() - agent_start
    finally:
        if host is not None:
            _stop(host)
        # Shell commands run in their own sessions, so background jobs survive the agent's process group.
        orphans = ti.kill_tagged(trial_id, namespaces=namespaces)
        if sampler is not None:
            sampler.stop()
        if agent is None:
            # The trial never launched its agent (host failure, barrier timeout): keep the records, drop the rest.
            shutil.rmtree(sandbox_root, ignore_errors=True)
    load_end = os.getloadavg()
    activity = parse_events(run_dir / "agent.jsonl", spec["cli"])
    activity["mcp_available"] = mcp_available(run_dir / "agent.jsonl", spec) if mcp is not None else None
    verification = verify(workspace, run_dir, task, env, contained)
    samples = [json.loads(line) for line in (run_dir / "resources.jsonl").read_text().splitlines()]
    summary = {
        **spec,
        "started_unix": started,
        "budget_start_unix": budget_start,
        "application_startup_seconds": startup,
        "agent_seconds": elapsed,
        "total_seconds": elapsed + startup,
        "timed_out": timed_out,
        "agent_exit_code": agent.returncode,
        "orphans_killed": len(orphans),
        "load_average_end": load_end,
        "load_1min_mean": sum(s["load"][0] for s in samples) / len(samples) if samples else None,
        "trial_cpu_seconds": max((s["trial_cpu_s"] or 0.0 for s in samples), default=None),
        "max_live_trials": max((s["live_trials"] for s in samples), default=None),
        "private_reference": _mentions_private(workspace),
        **activity,
        "verification": verification,
        "success": bool(verification.get("success")) and not timed_out,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    # Archive the workspace with the records; drop the trial's caches and private temp directories.
    (run_dir / "workspace").unlink()
    shutil.move(str(workspace), str(run_dir / "workspace"))
    shutil.rmtree(sandbox_root, ignore_errors=True)
    return summary


def _mentions_private(workspace: Path) -> bool:
    """Whether any workspace source names the hidden verification data."""
    for path in workspace.rglob("*.py"):
        text = path.read_text(errors="replace")
        if "newton-visual-private" in text or "NEWTON_VISUAL_PRIVATE" in text:
            return True
    return False


def verify(workspace: Path, run_dir: Path, task: dict, env: dict, contained=None) -> dict:
    """Run the task's verifier on the submission, sandboxed with only this task's hidden data readable."""
    sandbox_root = workspace.parent
    output = sandbox_root / "verification.json"
    command = [str(PYTHON), str(ROOT / task["verifier"]), str(workspace / task["script"]), "--output", str(output)]
    if contained is not None:
        command = contained(command, extra_ro=[PRIVATE / name for name in task["private"] if (PRIVATE / name).exists()])
    try:
        result = subprocess.run(
            command, cwd=workspace, env=env, capture_output=True, text=True, timeout=1800, check=False
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "verification timed out"}
    finally:
        ti.kill_tagged(env["NEWTON_TRIAL_ID"])
    (run_dir / "verification.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0 or not output.exists():
        return {"success": False, "error": (result.stderr or result.stdout)[-2000:]}
    shutil.copyfile(output, run_dir / "verification.json")
    data = json.loads(output.read_text())
    return {k: data[k] for k in ("success", "integrity", "failed_checks", "metrics", "normalized_worst")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=(
            "grasp_drift",
            "g1_track",
            "dp_real",
            "cube_toss",
            "sdf_grind",
            "g1_hard",
            "abc_twin",
            "abc_arm",
            "abc_look",
        ),
    )
    parser.add_argument("--condition", required=True, choices=("mcp", "restart"))
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--workspace", type=Path, required=True, help="Run directory for the harness records")
    parser.add_argument("--seconds", type=int, default=None, help="Budget [s]; defaults to the task's (usually 1800)")
    parser.add_argument("--phase", default="loop")
    parser.add_argument("--barrier", type=Path, help="Shared directory that starts both conditions of a pair together")
    parser.add_argument("--parties", type=int, default=2)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    run_dir = args.workspace.resolve()
    prepared = prepare(run_dir, args.task, args.condition, args.model, args.seconds, args.phase)
    if not args.run:
        print(f"Prepared {run_dir}")
        return
    summary = run_trial(prepared, args.barrier, args.parties)
    attempt = 1
    while summary.get("mcp_available") is False and attempt < 3:
        # The partner already ran, so a retry runs alone; analyses must treat retried pairs separately.
        run_dir.rename(run_dir.with_name(f"{run_dir.name}.infra-failure-{attempt}"))
        prepared = prepare(run_dir, args.task, args.condition, args.model, args.seconds, args.phase)
        summary = run_trial(prepared)
        summary["infrastructure_retries"] = attempt
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        attempt += 1
    print(
        json.dumps(
            {k: summary[k] for k in ("success", "total_seconds", "tool_call_total")} | {"usage": summary["usage"]}
        )
    )


if __name__ == "__main__":
    sys.exit(main())
