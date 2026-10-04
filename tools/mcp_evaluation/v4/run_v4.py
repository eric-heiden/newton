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
import fcntl
import hashlib
import json
import os
import re
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
REPLAY_DATA = Path(
    os.environ.get("NEWTON_REPLAY_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/abc_replay_task")
)
BIN_DATA = Path(os.environ.get("NEWTON_BIN_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/abc_bin_task"))
SCRATCH_DATA = Path(
    os.environ.get("NEWTON_SCRATCH_DATA", "/home/horde/artifacts/newton-live-mcp-v4/datasets/abc_scratch_task")
)
PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private"))
# Agent workspaces get opaque names under a neutral root; the labeled run directory keeps the harness
# records (spec, command, transcript, host log, verification), which agents must not see.
TRIALS = Path(os.environ.get("NEWTON_TRIAL_ROOT", Path.home() / "trials"))
SANDBOX = os.environ.get("NEWTON_TRIAL_SANDBOX", "1") == "1"
VERIFY_LOCKS = Path(os.environ.get("NEWTON_VERIFY_LOCKS", Path.home() / ".cache" / "newton-verify-locks"))
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
        "g1_mpc": {
            "files": {name: HERE / "g1_mpc" / name for name in ("g1_mpc.py", "walk.csv", "dance.csv", "jumpjack.csv")},
            "script": "g1_mpc.py",
            "host_args": ["--motion", "walk.csv"],
            "verifier": "tools/mcp_evaluation/v4/g1_mpc/verify.py",
            "seconds": 5400,
            "private": ["g1_mpc"],
            # Six clips in fresh processes, each rollout capped at up to 2.5x its idle-machine limit under load.
            "verify_seconds": 3600,
            # One g1_mpc verification at a time: concurrent ones would slow each other's timed rollouts.
            "verify_lock": True,
            "warmup": [
                ["g1_mpc.py", "--viewer", "null", "--num-frames", "1", "--motion", "walk.csv"],
                ["-m", "tools.mcp_evaluation.v4.trial_isolation", "warm", "g1_mpc.py", "--motion", "walk.csv"],
                # Planning-solver variants agents try (both conditions get the same seed).
                [
                    "-m",
                    "tools.mcp_evaluation.v4.trial_isolation",
                    "warm-solvers",
                    "g1_mpc.py",
                    json.dumps(
                        [
                            {"use_mujoco_contacts": True, "integrator": "euler", "njmax": 192, "nconmax": 64},
                            {"use_mujoco_contacts": True, "cone": "elliptic", "njmax": 192, "nconmax": 64},
                            {"use_mujoco_contacts": False, "integrator": "implicitfast"},
                        ]
                    ),
                ],
            ],
            "goal": """g1_mpc.py simulates a floating-base Unitree G1 humanoid (29 actuated joints, SolverMuJoCo with 2 ms steps) that should track reference motions: walk.csv, dance.csv, and jumpjack.csv (Kimodo clips in MuJoCo qpos format: root position, root quaternion wxyz, and 29 joint angles at 30 fps). A controller runs at 100 Hz: every 10 ms it gets the robot's state and returns one Command for the joint actuators (position and velocity targets, stiffness, damping, and feedforward torque per joint; the actuators clip the torque to the MJCF limits). The script's tracking report prints the metrics below. Problem: the baseline controller only servos the joints toward the reference angles, and the robot falls within about a second.

Goal: develop a controller that makes the robot track the reference motions; investigate model predictive control methods for this. Verification imports the script's Controller and MotionClip and simulates its own copy of the starter's plant (build_model, make_solver, Robot, Command, and TrackingReport) in a fresh process per clip, on the three given clips, on two unseen clips, and on an unseen slower playback of walk.csv (at {slow_low} to {slow_high} times the original speed, drawn at verification): Controller(model, motion) is built once per clip (model is a copy of the robot model posed at the clip's start, for the controller's own use), then compute(t, joint_q, joint_qd) is called every 10 ms. On every clip the robot must stay on its feet for the whole clip (root above {fall_m} m, up axis within 45 degrees of vertical, nothing but the feet touching the floor), with root position RMSE at most {root_cm} cm, root orientation RMSE at most {rot_deg} degrees, joint-angle RMSE at most {joint_rad} rad, and sole position RMSE at most {sole_cm} cm; wrist position errors, foot-lift recall, and jitter are reported as well. Per clip, building the Controller may take at most {setup_s} s and the rollout (all compute() calls and the physics) at most {rollout_s} s of wall time on this machine when it is otherwise idle; verification measures the machine's load (fixed GPU and CPU workloads timed between compute() calls) and allows proportionally more time, at most {load_max}x.
Constraints (checked): the plant is fixed (robot model, masses, inertias, armature, torque limits, contacts, timestep, and the actuator law), and the controller acts on it only through the Command it returns (gains within the ranges in the script); it may build its own models and solvers, for example for planning, and use the installed Python packages. The controller runs in one Python thread: no threads or subprocesses (GPU and library-internal parallelism are fine), and it may not inspect the verifier (stack frames, garbage collector, raw memory, code objects, trace hooks) or read files outside the workspace. Use only the data in the workspace: do not download motion clips or any other data (the G1 model is already cached). Keep the Controller(model, motion) and compute(t, joint_q, joint_qd) interface and the file runnable (`python g1_mpc.py --motion walk.csv --viewer null` runs the whole clip and prints the tracking report; --num-frames N runs N control periods).""",
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
                    for name in (
                        "station_look.py",
                        "look.py",
                        "render_look.py",
                        "look_common.py",
                        "blender_bridge.py",
                        "blender_server.py",
                    )
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
        "abc_replay": {
            "files": {
                **{name: HERE / "abc_replay" / name for name in ("fruit_replay.py", "replay_common.py", "FORMAT.md")},
                **{
                    name: REPLAY_DATA / name
                    for name in ("station", "episodes", "scenes", "gt", "frames", "camera.json", "video_reference.npz")
                },
                "arm_logs": ARM_DATA / "logs",
            },
            "script": "fruit_replay.py",
            "host_args": [],
            "run_args": "[--seconds <S>] [--episode sib_1] [--num-worlds 8 --jitter]",
            "verifier": "tools/mcp_evaluation/v4/abc_replay/verify.py",
            "seconds": 3600,
            "private": ["abc_replay"],
            # One frame of the starter (its main ignores --num-frames and would replay the whole episode).
            "warmup": [
                ["fruit_replay.py", "--viewer", "null", "--seconds", "0.034"],
                ["-m", "tools.mcp_evaluation.v4.trial_isolation", "warm", "fruit_replay.py"],
                # Solver variants agents try (both conditions get the same seed).
                [
                    "-m",
                    "tools.mcp_evaluation.v4.trial_isolation",
                    "warm-solvers",
                    "fruit_replay.py",
                    json.dumps(
                        [
                            {"cone": "elliptic"},
                            {"use_mujoco_contacts": True},
                            {"use_mujoco_contacts": True, "cone": "elliptic"},
                        ]
                    ),
                ],
            ],
            "goal": """fruit_replay.py replays a real robot episode from the ABC-130k dataset (https://abc.bot) in Newton. At a bimanual station (two 6-DoF YAM arms with parallel grippers, filmed from above and from both wrists), a teleoperator picks up three fake fruits one after another, a pear and an orange with the left arm and a dark fruit with the right, and puts them into a wedge-shaped tray. episodes/main.npz holds the measured joint positions, velocities, and torques, the gripper openings, and the logged joint and gripper commands (about 30 Hz); scenes/main.json the station layout (arm bases, tray sector, and per fruit its size, start pose, grasping arm, and the video frames of the real grasp events); frames/main/ the recorded top and wrist videos and video_reference.npz the fruit positions tracked in them. episodes/, scenes/, gt/, and frames/ also hold four 10 fps episodes of the same station and fruits (sib_1 to sib_4), and arm_logs/ 64 recorded YAM arm logs (32 episodes of various tasks, both arms; measured joints, velocities, torques, and commands) for calibrating the arms. FORMAT.md describes the files, their clocks, and the fixed helpers in replay_common.py (Replay, score, check, rendering through the real cameras, contact_summary, StationFK). Each scene is one Newton world from build_model(scenes), driven open loop by the logged commands as joint position targets (replay_common.Replay) and stepped with make_solver(model) and make_pipeline(model). Problem: the starter keeps the ABC simulator's defaults, and none of the fruits is held: the grasps slip, so nothing is carried to the tray.

Goal: make the replay physically reproduce the whole episode: every fruit grasped, lifted, carried, and released into the tray the way the real robot did it. Verification imports build_model, make_solver, make_pipeline, and PARAMS and runs its own replay of the full timelines (its own copy of replay_common: Replay, score, and jitter_scene) in fresh processes, on the main episode in 8 copies (2 nominal, 6 with the fruit starts jittered by up to 4 mm and the pear heading by 5 degrees) and on unseen episodes of the same station and fruits in 4 copies each. Main episode, per fruit: held through the carry in at least {held} of 8 copies and resting in the tray at the end in at least {placed}; medians over the copies: lifted fraction at least {lifted}, carry-track error at most {track_cm} cm, lift-off error at most {liftoff} state samples, release error between {release_low} and +{release_high} s, movement before the grasp at most {moved_cm} cm, finger-gap error while holding at most {gap_mm} mm, and final position error at most {final_cm} cm; whole-episode joint RMSE at most {arm_rad} rad per arm. Unseen episodes: at least {heldout_pct}% of their fruit copies held and placed (fruits whose real grasp the recorded data cannot reproduce are excluded), and joint RMSE at most {heldout_arm_rad} rad per arm (mean over episodes). Two negative controls replay the main episode with the gripper commands forced open and with arm and fruit friction set to 0.02: no fruit may rise more than {control_cm} cm. Verification runs twice (a third time if they disagree) and the majority decides. The starter holds no fruit in any copy and scores about 0.040 rad joint RMSE.
Constraints (checked): keep the station's arm kinematics, finger collision geometry (within 0.5 mm), arm bases (within 3 mm of the scene's), table plane, gravity, and the command input; add no shapes, actuators, equality constraints, tendons, or contact pairs to the robot. Each fruit is one free, dynamic body labelled pear, orange, or dark_fruit, starting at the scene's start (within 2 mm, resting on the table, pear long axis within 5 degrees of the scene's heading), with 20 to 150 g, principal inertias between 0.8x a solid and 1.2x a hollow ellipsoid of its extents, at most 8 collision shapes spanning the size ranges in the scene, collisions with the fingers, table, tray, and the other fruits, and no joint drives, springs, damping, gravity compensation, or applied forces. Tunable: arm joint gains, armature, friction, damping, effort limits (at most 28 N m on joints 1-3 and 10 N m on joints 4-6), and gravity compensation (0 to 1) per link; gripper position gain (100 to 3000 N/m) and squeeze force (5 to 60 N); PARAMS["command_delay"] (0 to 0.2 s) and PARAMS["dt"] (0.25 to 2 ms); the solver (a Newton solver class, not a subclass) and its settings; newton.CollisionPipeline settings or MuJoCo's own contacts; materials (friction at most 1.5, torsional at most 0.02 m, rolling at most 0.005 m, restitution at most 0.8, margin at most 2 mm, contact gap at most 0.1 m, no adhesion; robot shapes may keep their MJCF values); fruit shapes, masses, and inertias within the bounds above; and the tray model (at most 40 shapes, static or on one dynamic body labelled tray of 0.1 to 1 kg, inside the scene's tray sector plus 15 mm and at most 35 mm above the table). The verification batch (about 44 worlds) must replay within about 400 s. replay_common.py is fixed (verification uses its own copy). Verification calls build_model, make_solver, and make_pipeline in a copy of the workspace without frames/, arm_logs/, gt/, and video_reference.npz, and passes only the scenes (geometry, starts, sizes; no episode paths or ids), so keep fitted values in the script or in a file next to it. The submission may not inspect the verifier (stack frames, garbage collector, raw memory, code objects, trace hooks), start processes, or read files outside the workspace while it is built. Use only the data in the workspace: do not download recordings or any other data. Keep the file runnable (`python fruit_replay.py --viewer null` replays the main episode and prints the metrics; `--episode sib_1`, `--num-worlds 8 --jitter`, and `--seconds` select episodes, ensembles, and shorter runs).""",
        },
        "abc_bin": {
            "files": {
                **{
                    name: HERE / "abc_bin" / name for name in ("screwdriver_replay.py", "replay_common.py", "FORMAT.md")
                },
                **{name: BIN_DATA / name for name in ("station", "episodes", "scenes", "gt", "frames", "camera.json")},
                "arm_logs": ARM_DATA / "logs",
            },
            "script": "screwdriver_replay.py",
            "host_args": [],
            "run_args": "[--seconds <S>] [--episode sib_1] [--num-worlds 8 --jitter]",
            "verifier": "tools/mcp_evaluation/v4/abc_bin/verify.py",
            "seconds": 3600,
            "private": ["abc_bin"],
            # One frame of the starter (its main ignores --num-frames and would replay the whole episode).
            "warmup": [
                ["screwdriver_replay.py", "--viewer", "null", "--seconds", "0.034"],
                ["-m", "tools.mcp_evaluation.v4.trial_isolation", "warm", "screwdriver_replay.py"],
                # Solver variants agents try (both conditions get the same seed).
                [
                    "-m",
                    "tools.mcp_evaluation.v4.trial_isolation",
                    "warm-solvers",
                    "screwdriver_replay.py",
                    json.dumps(
                        [
                            {"cone": "elliptic"},
                            {"use_mujoco_contacts": True},
                            {"use_mujoco_contacts": True, "cone": "elliptic"},
                        ]
                    ),
                ],
            ],
            "goal": """screwdriver_replay.py replays a real robot episode from the ABC-130k dataset (https://abc.bot) in Newton. At a bimanual station (two 6-DoF YAM arms with parallel grippers, filmed from above and from the wrists), the right arm picks up a screwdriver lying on the table by its handle, carries it over, and puts it into a pink plastic bin. episodes/main.npz holds the measured joint positions, velocities, and torques (including the gripper motor's effort), the gripper openings, and the logged joint and gripper commands (about 30 Hz); scenes/main.json the station layout (arm bases, the bin's pose and size, and the screwdriver: its tapered handle profile, start pose, grasping arm, and the video frames of the real grasp events); frames/main/ the recorded top and right wrist videos; gt/main.npz the screwdriver's carry track (rigidly attached to the hand) and its final pose in the bin from the top video. episodes/, scenes/, gt/, and frames/ also hold four 10 fps episodes of the same station, bin, and screwdriver (sib_1 to sib_4), and arm_logs/ 64 recorded YAM arm logs (32 episodes of various tasks, both arms; measured joints, velocities, torques, and commands) for calibrating the arms. FORMAT.md describes the files, their clocks, and the fixed helpers in replay_common.py (Replay, score, check, rendering through the real cameras, contact_summary, StationFK). Each scene is one Newton world from build_model(scenes), driven open loop by the logged commands as joint position targets (replay_common.Replay) and stepped with make_solver(model) and make_pipeline(model). Problem: the starter keeps the ABC simulator's arm and gripper defaults. The screwdriver ends up in the bin, but during the carry it swings about {starter_rot}° in the fingers (median over copies) and slips about {starter_slip_mm} mm, while the right wrist video shows the real one sitting still in the hand, and the arms track the recording at about {starter_arm_rad} rad joint RMSE. In the friction-{control_mu} control below, the starter's simulation diverges in its first step (MuJoCo Warp's pyramidal friction cone is unstable at such low friction), which counts as held.

Goal: make the replay physically reproduce the whole episode: the screwdriver grasped, lifted, carried without turning or slipping in the fingers, and placed in the bin the way the real robot did it. Verification imports build_model, make_solver, make_pipeline, and PARAMS and runs its own replay of the full timelines (its own copy of replay_common: Replay, score, and jitter_scene) in fresh processes, on the main episode in 8 copies (2 nominal, 6 with the screwdriver start jittered by up to 4 mm and 5 degrees) and on unseen episodes of the same station, bin, and screwdriver in 4 copies each. Main episode: the screwdriver held through the carry in at least {held} of 8 copies and resting in the bin at the end in at least {placed}; medians over the copies: in-hand rotation during the carry at most {rot}°, slip at most {slip_mm} mm, finger-gap error while holding at most {gap_mm} mm, lift-off error at most {liftoff} state samples, carry-track error at most {track_cm} cm, final tip error at most {final_cm} cm, and movement before the grasp at most {moved_cm} cm; whole-episode joint RMSE at most {arm_rad} rad per arm. Unseen episodes: at least {ho_pct}% of their copies held, placed, and rotated at most {ho_rot}° in the hand (episodes whose real grasp the recorded data cannot reproduce are excluded), and joint RMSE at most {ho_arm_rad} rad per arm (mean over episodes). Two negative controls replay the main episode with the gripper commands forced open (the screwdriver may not rise more than {control_cm} cm) and with the friction of every collision shape set to {control_mu} (the screwdriver may not be held in any copy; a copy whose simulation diverges counts as held). Verification runs twice (a third time if they disagree) and the majority decides.
Constraints (checked): keep the station's arm kinematics, finger collision geometry (within 0.5 mm), arm bases (within 3 mm of the scene's), table plane, gravity, and the command input; add no shapes, actuators, equality constraints, tendons, or contact pairs to the robot (the finger-mirror equality may be stiffened, not removed). The screwdriver is one free, dynamic body labelled screwdriver with at most 6 collision shapes: 200 to 215 mm long, handle radius within 1.5 mm of the scene's profile at its stations and within 2.5 mm between them, shaft radius 1.5 to 3.5 mm, 40 to 150 g, centre of mass on the axis, within the handle and within 15 mm of the scene's com_local, principal inertias between 0.8x a solid and 1.2x a hollow screwdriver of the scene's shape, starting at the scene's start (within 2 mm and 5 degrees, resting on the table), colliding with the fingers, table, and bin, and with no joint drives, springs, damping, gravity compensation, or applied forces. Tunable: arm joint gains, armature, friction, damping, effort limits (at most 28 N m on joints 1-3 and 10 N m on joints 4-6), and gravity compensation (0 to 1) per link; gripper position gain (100 to 30000 N/m) and pinch force (5 to 80 N per pad; the finger mirror splits the finger actuators' force between the two pads); PARAMS["command_delay"] (0 to 0.2 s) and PARAMS["dt"] (0.25 to 2 ms); the solver (a Newton solver class, not a subclass) and its settings; newton.CollisionPipeline settings or MuJoCo's own contacts; MuJoCo contact and finger-mirror stiffness (solref in standard form with a time constant of at least 2 dt and a damping ratio of 0.5 to 2, refsafe on); materials of every collision shape, finger pads included (friction at most 1.5, torsional at most 0.01 m, rolling at most 0.001 m, restitution at most 0.8, margin at most 2 mm, contact gap at most 0.1 m, no adhesion); and the bin model (at most 12 shapes with walls all around, static or on one dynamic body labelled bin of 0.2 to 0.6 kg, inside the scene's rim outline plus 10 mm, its top within 15 mm of the scene's height). The verification batch (68 worlds) must replay within about 400 s. replay_common.py is fixed (verification uses its own copy). Verification calls build_model, make_solver, and make_pipeline in a copy of the workspace without frames/, arm_logs/, and gt/, and passes only the scenes (geometry, starts, sizes, events; no episode paths, ids, or image measurements), so keep fitted values in the script or in a file next to it. The submission may not inspect the verifier (stack frames, garbage collector, raw memory, code objects, trace hooks), start processes, or read files outside the workspace while it is built. Use only the data in the workspace: do not download recordings or any other data. Keep the file runnable (`python screwdriver_replay.py --viewer null` replays the main episode and prints the metrics; `--episode sib_1`, `--num-worlds 8 --jitter`, and `--seconds` select episodes, ensembles, and shorter runs).""",
        },
        "abc_scratch": {
            "files": {
                **{name: HERE / "abc_scratch" / name for name in ("scene_replay.py", "FORMAT.md")},
                **{name: SCRATCH_DATA / name for name in ("episode.npz", "photos", "camera.json", "station")},
                "arm_logs": ARM_DATA / "logs",
            },
            "script": "scene_replay.py",
            "host_args": [],
            "run_args": "[--seconds <S>] [--num-worlds <N>]",
            "verifier": "tools/mcp_evaluation/v4/abc_scratch/verify.py",
            "seconds": 5400,
            "private": ["abc_scratch"],
            # One frame of the starter (its main ignores --num-frames and would replay the whole episode).
            "warmup": [
                ["scene_replay.py", "--viewer", "null", "--seconds", "0.034"],
                ["-m", "tools.mcp_evaluation.v4.trial_isolation", "warm", "scene_replay.py"],
                # Solver variants agents try (both conditions get the same seed).
                [
                    "-m",
                    "tools.mcp_evaluation.v4.trial_isolation",
                    "warm-solvers",
                    "scene_replay.py",
                    json.dumps(
                        [
                            {"cone": "elliptic"},
                            {"use_mujoco_contacts": True},
                            {"use_mujoco_contacts": True, "cone": "elliptic"},
                        ]
                    ),
                ],
            ],
            "goal": """scene_replay.py replays a real robot episode from the ABC-130k dataset (https://abc.bot) in Newton, open loop. At a bimanual station (two 6-DoF YAM arms with parallel grippers, filmed from above and from both wrists), a teleoperator picks up three fake fruits one after another, a pear and an orange with the left arm and a dark round fruit with the right, and puts them into a wedge-shaped wooden tray on the table. photos/ holds {photos} frames of the recording (the top camera at the start, before each grasp, after each release, and at the end; the wrist cameras at each grasp; photos/index.json gives the state sample each one shows); episode.npz the measured joint positions, velocities, and torques (including the gripper motor's effort), the gripper openings, and the logged joint and gripper commands (about 30 Hz); camera.json the calibrated cameras (intrinsics with RealSense distortion, the top camera's pose in the world, and the wrist cameras' mounts); station/ the ABC simulator's MJCF of the station (arms, grippers, table, and enclosure; no objects); and arm_logs/ 64 recorded YAM arm logs (32 episodes of various tasks, both arms; measured joints, velocities, torques, and commands) for calibrating the arms. FORMAT.md describes the files and their frames. scene_replay.py builds the station in identical worlds (build_model(num_worlds)), steps them with make_solver(model) and make_pipeline(model), and drives the arms with the logged commands as joint position targets (its docstring states the command rule). Problem: the scene is empty. Only the station is modeled, with the ABC simulator's arm and gripper defaults; the fruits and the tray are missing.

Goal: recreate the scene and its physics from the photos and the logs (the three fruits with their sizes, masses, start poses, and materials; the tray), so that the replay reproduces the real episode: every fruit grasped, carried, and released into the tray, coming to rest where the real one did. Verification imports build_model, make_solver, make_pipeline, and PARAMS and runs its own replay of the whole episode (the starter's command rule, its own loop) in fresh processes, in 8 copies: 2 nominal, and 6 with every fruit's start moved by up to {jitter_mm} mm and turned by up to {jitter_deg} degrees about the vertical. It takes as the fruits the free bodies that start closest to where the top camera saw the real fruits at the start (within {match_cm} cm). Per fruit, in at least {held} of 8 copies: held through the real carry (from 3 state samples after the real lift-off to 2 before the real release), that is, its centre within the real fruit's half-width plus {hold_mm} mm of the midpoint between the grasping gripper's two finger-pad grasp points (25 mm from each pad's centre toward its tip) for at least {held_pct}% of the carry. Per fruit, in at least {placed} of 8 copies: at rest in the real tray at the end of the episode (inside the tray's outline at the start or the end plus {place_cm} cm, its centre within {height_mm} mm of the height of the real fruit lying on the tray floor, slower than {speed_cm} cm/s over the last {window_s} s), and within {rest_cm} cm of where the real fruit came to rest after its release (its final position, or one it rested at before a later fruit pushed it). Two negative controls replay the episode with the gripper commands forced open and with the friction of every arm and fruit collision shape set to {control_mu}: no fruit may rise more than {control_cm} cm, and a copy whose simulation diverges fails its control (MuJoCo Warp's default pyramidal friction cone can diverge at such low friction). Verification runs twice (a third time if they disagree) and the majority decides.
Constraints (checked): keep the station's arm kinematics, finger collision geometry (within 0.5 mm), the MJCF's arm bases (within 3 mm), table plane, gravity, and the command input; add no shapes, actuators, equality constraints, tendons, or contact pairs to the robot, and keep its actuators plain servos (no bias force, unit gear). All worlds are identical copies. The fruits are exactly three more free, dynamic bodies (any labels), each starting at rest on the table (its lowest point within 1 mm below to 3 mm above it) within {start_cm} cm of where the top camera saw the real fruit, with {mass_g_low} to {mass_g_high} g, principal inertias between 0.8x a solid and 1.2x a thin-shell ellipsoid of its extents, its centre of mass within 2 cm of the centre of its collision geometry, at most {object_shapes} collision shapes whose extents are within 30% of the real fruit's size range (as tracked in the video), collisions with the finger pads, the table, the tray, and the other fruits, and no joint drives, springs, damping, joint friction, armature, gravity compensation, or applied forces. Everything else you add (the tray) is static or one more body (free or welded to the world) of {tray_kg_low} to {tray_kg_high} kg, with at most {tray_shapes} collision shapes inside the real tray's outline plus {tray_cm} cm and at most {tray_top_cm} cm above the table. Tunable: arm joint gains, armature, friction, damping, effort limits (at most 28 N m on joints 1-3 and 10 N m on joints 4-6), and gravity compensation (0 to 1) per link; gripper position gain (100 to 3000 N/m) and squeeze force (5 to 60 N); PARAMS["command_delay"] (0 to 0.2 s) and PARAMS["dt"] (0.25 to 2 ms); the solver (a Newton solver class, not a subclass) and its settings; newton.CollisionPipeline settings or MuJoCo's own contacts; MuJoCo contact stiffness (solref in standard form with a time constant of at least 2 dt and a damping ratio of 0.5 to 2, refsafe on); materials (friction at most 1.5, torsional at most 0.02 m, rolling at most 0.005 m, restitution at most 0.8, margin at most 2 mm, contact gap at most 0.1 m, no adhesion; robot shapes may keep their MJCF values). The verification batch (12 worlds) must replay within about {runtime_s} s. Verification calls build_model, make_solver, and make_pipeline in a copy of the workspace without photos/ and arm_logs/ (with its own episode.npz, camera.json, and station/; links to files outside the workspace are not allowed), so keep fitted values in the script or in a file next to it. It then re-applies the model to SolverMuJoCo (notify_model_changed with all flags) and requires the solver's own arrays to match the model it compiled, so set solver parameters through the model (including its model.mujoco attributes) and the solver's constructor, not by editing the solver's MuJoCo arrays. The submission may not inspect the verifier (stack frames, garbage collector, raw memory, code objects, trace hooks), start processes, or read files outside the workspace while it is built. Use only the data in the workspace: do not download recordings or any other data. Keep the file runnable (`python scene_replay.py --viewer null` replays the whole episode; `--seconds` and `--num-worlds` select shorter runs and more copies).""",
        },
    }
    if name == "abc_look":
        # The look task renders with Blender in both conditions (inherited by agents and verifiers).
        os.environ.setdefault("NEWTON_BLENDER", "/home/horde/opt/blender-5.2.2-linux-x64/blender")
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
    if name == "abc_replay":
        from tools.mcp_evaluation.v4.abc_replay.verify import THRESHOLDS as t  # noqa: PLC0415

        tasks[name]["goal"] = tasks[name]["goal"].format(
            held=t["main_held_min"],
            placed=t["main_placed_min"],
            lifted=f"{t['lifted_fraction_min']:g}",
            track_cm=f"{100 * t['carry_track_err_m_max']:.2f}",
            liftoff=f"{t['liftoff_err_rows_max']:g}",
            release_low=f"{t['release_err_s_min']:g}",
            release_high=f"{t['release_err_s_max']:g}",
            moved_cm=f"{100 * t['moved_before_grasp_m_max']:g}",
            gap_mm=f"{t['grip_gap_err_mm_max']:g}",
            final_cm=f"{100 * t['final_xy_err_m_max']:.1f}",
            arm_rad=f"{t['arm_rmse_rad_max']:g}",
            heldout_pct=f"{100 * t['heldout_fruit_rate_min']:.1f}",
            heldout_arm_rad=f"{t['heldout_arm_rmse_rad_max']:g}",
            control_cm=f"{100 * t['control_rise_m_max']:g}",
        )
    if name == "abc_bin":
        from tools.mcp_evaluation.v4.abc_bin import verify as bin_verify  # noqa: PLC0415

        CONTROL_MU, STARTER, t = bin_verify.CONTROL_MU, bin_verify.STARTER, bin_verify.THRESHOLDS

        tasks[name]["goal"] = tasks[name]["goal"].format(
            starter_rot=f"{STARTER['inhand_rot_deg']:.0f}",
            starter_slip_mm=f"{STARTER['slip_mm']:.0f}",
            starter_arm_rad=f"{STARTER['arm_rmse_rad']:.2f}",
            held=t["main_held_min"],
            placed=t["main_placed_min"],
            rot=f"{t['inhand_rot_deg_max']:g}",
            slip_mm=f"{1000 * t['slip_m_max']:g}",
            gap_mm=f"{t['grip_gap_err_mm_max']:g}",
            liftoff=f"{t['liftoff_err_rows_max']:g}",
            track_cm=f"{100 * t['carry_track_err_m_max']:.2f}",
            final_cm=f"{100 * t['final_tip_xy_err_m_max']:.1f}",
            moved_cm=f"{100 * t['moved_before_grasp_m_max']:g}",
            arm_rad=f"{t['arm_rmse_rad_max']:g}",
            ho_pct=f"{100 * t['heldout_rate_min']:.1f}",
            ho_rot=f"{t['heldout_rot_max_deg']:g}",
            ho_arm_rad=f"{t['heldout_arm_rmse_rad_max']:g}",
            control_cm=f"{100 * t['control_rise_m_max']:g}",
            control_mu=f"{CONTROL_MU:g}",
        )
    if name == "abc_scratch":
        from tools.mcp_evaluation.v4.abc_scratch import verify as scratch  # noqa: PLC0415

        t, b, core = scratch.THRESHOLDS, scratch.BOUNDS, scratch.core
        tasks[name]["goal"] = tasks[name]["goal"].format(
            photos=len(json.loads((SCRATCH_DATA / "photos" / "index.json").read_text())),
            jitter_mm=f"{1000 * scratch.JITTER_XY_M:g}",
            jitter_deg=f"{scratch.JITTER_YAW_DEG:g}",
            match_cm=f"{100 * b['match_m']:g}",
            held=t["main_held_min"],
            hold_mm=f"{1000 * core.HOLD_MARGIN:g}",
            held_pct=f"{100 * t['held_fraction_min']:g}",
            placed=t["main_placed_min"],
            place_cm=f"{100 * core.PLACE_MARGIN:g}",
            height_mm=f"{1000 * core.REST_HEIGHT_TOLERANCE:g}",
            speed_cm=f"{100 * core.REST_SPEED:g}",
            window_s=f"{core.REST_WINDOW:g}",
            rest_cm=f"{100 * t['rest_xy_err_m_max']:g}",
            control_mu=f"{scratch.CONTROL_MU:g}",
            control_cm=f"{100 * t['control_rise_m_max']:g}",
            start_cm=f"{100 * b['start_xy_m']:g}",
            mass_g_low=f"{1000 * b['object_mass_kg'][0]:g}",
            mass_g_high=f"{1000 * b['object_mass_kg'][1]:g}",
            object_shapes=b["object_shapes_max"],
            tray_kg_low=f"{b['tray_mass_kg'][0]:g}",
            tray_kg_high=f"{b['tray_mass_kg'][1]:g}",
            tray_shapes=b["tray_shapes_max"],
            tray_cm=f"{100 * b['tray_margin_m']:g}",
            tray_top_cm=f"{100 * b['tray_top_m']:g}",
            runtime_s=f"{scratch.RUNTIME_LIMIT_S:g}",
        )
    if name == "g1_mpc":
        from tools.mcp_evaluation.v4.g1_mpc import verify as mpc  # noqa: PLC0415

        t = mpc.THRESHOLDS
        tasks[name]["goal"] = tasks[name]["goal"].format(
            fall_m=f"{mpc.plant.FALL_HEIGHT_M:g}",
            root_cm=f"{100 * t['root_rmse_m']:g}",
            rot_deg=f"{t['root_rot_rmse_deg']:g}",
            joint_rad=f"{t['joint_rmse_rad']:g}",
            sole_cm=f"{100 * t['sole_rmse_m']:g}",
            setup_s=f"{mpc.SETUP_LIMIT_S:g}",
            rollout_s=f"{mpc.ROLLOUT_LIMIT_S:g}",
            load_max=f"{mpc.LOAD_FACTOR_MAX:g}",
            slow_low=f"{mpc.SCALED['walk_slow'][1][0]:g}",
            slow_high=f"{mpc.SCALED['walk_slow'][1][1]:g}",
        )
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


def _renderers() -> str:
    """Renderers available to both conditions when a task provides Blender (the look task)."""
    blender = os.environ.get("NEWTON_BLENDER")
    if not blender or not Path(blender).exists():
        return ""
    return (
        "Renderers: newton.sensors.SensorCamera (ray-cast, GPU) and Blender (headless, scriptable with bpy; "
        f"executable in $NEWTON_BLENDER = {blender}).\n"
    )


def _newton_tools() -> str:
    """Newton utilities that serve both conditions (the MCP guide names their live wrappers, so the prompt must)."""
    return (
        "Newton utilities: `python -m newton.examples.headless SCRIPT [script args] [--frames N] [--call EXPR] "
        "[--json OUT] [--timeout S]` runs an example script in a fresh process and reports the outcome as JSON; "
        "newton.utils.report_solver_params(solver, kind) reports the values a solver integrates and the model arrays "
        "they come from; newton.utils.report_health(model, state, solver) reports non-finite values, full buffers, "
        "and penetrating shape pairs per world.\n"
    )


def prompt_for(name: str, condition: str, workspace: Path, seconds: int, guide: str | None) -> str:
    task = _task(name)
    common = f"""You are working on a Newton physics simulation task.

Workspace: {workspace}
Newton source tree (read-only reference, including docs and examples): {ROOT}

{task["goal"]}

Deliverable: the edited {task["script"]} in the workspace, then a brief report. You have {seconds // 60} minutes, starting {START} (check with `date -u`); working efficiently matters. Do not modify files outside the workspace, do not look for other trials or hidden verification data, and do not use subagents.
{_renderers()}{_newton_tools()}"""
    run_args = task.get("run_args", "--num-frames <N>")
    run = f"uv run --no-sync --project {ROOT} python {task['script']} --viewer null {run_args} {' '.join(task['host_args'])}"
    run = run.rstrip()
    if condition == "restart":
        return (
            common
            + f"""
Workflow: this is a script-based setup. Run the simulation with
  {run}
or write your own scripts that import the Example class; each run starts a fresh simulator process. There is no display; to look at the scene, render images yourself (for example with newton.sensors.SensorCamera) and open them with your image-viewing tool.
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
    env = ti.trial_env(ROOT, sandbox_root / "caches", trial_id, ti.cli_homes(sandbox_root) if SANDBOX else None)
    env["MCP_TOOL_TIMEOUT"] = "300000"
    env["MAX_MCP_OUTPUT_TOKENS"] = "60000"
    env["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] = str(BG_WAIT_CEILING_MS)

    def contained(command: list[str], extra_ro: list[Path] = (), harness: bool = False) -> list[str]:
        # The host runs the agent's code too, so it shares the agent's filesystem view, including /tmp.
        if not SANDBOX:
            return command
        # Hide the run directory's parent too (other trials' records), wherever the iteration directory lives.
        hidden = [TRIALS, run_dir.parent]
        # The study's harness (verifiers, data generators) is only for the verifier itself.
        masked = [] if harness else [ROOT / "tools" / "mcp_evaluation"]
        return ti.sandbox(
            command, sandbox_root, ROOT, PRIVATE, extra_ro=list(extra_ro), extra_hidden=hidden, masked=masked
        )

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
        # restart agent's budget, and the stated start time is the moment the budget clock starts. The host's
        # (seeded, a few seconds) startup is outside both budgets; analyses charge it through total_seconds.
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
        "introspection": _introspection(workspace),
        "downloads": _downloads(run_dir / "agent.jsonl"),
        "api_failure": api_failure(run_dir / "agent.jsonl"),
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


# Frame, garbage-collector, and memory introspection: a submission could use it to forge its verification.
INTROSPECTION = re.compile(
    r"_getframe|_current_frames|currentframe|f_back|f_locals|tb_frame|gi_frame|get_referrers|get_objects"
    r"|\bctypes\b|addaudithook|settrace|setprofile|os\._exit|/proc/self|\bnonce\b"
)
# Fetching data from the network (the held-out episodes are public).
DOWNLOAD = re.compile(
    r"huggingface\.co|hf_hub|snapshot_download|hf_hub_download|datasets\.load_dataset|abc\.bot|voxel51"
    r"|\b(?:curl|wget)\b[^\n]*https?://|urllib\.request|requests\.get|git clone|\bhf download|huggingface-cli",
    re.IGNORECASE,
)


# API-side errors that end an agent's run (not the agent's doing): the trial is retried.
API_FAILURE = re.compile(
    r"capacity|overloaded|rate.?limit|too many requests|internal server error|service unavailable|\b(?:429|500|502|503|529)\b",
    re.IGNORECASE,
)


def api_failure(transcript: Path) -> str | None:
    """The API error that ended the agent's run (model at capacity, overloaded, rate limited), if any."""
    if not transcript.exists():
        return None
    for line in reversed(transcript.read_text(errors="replace").splitlines()[-20:]):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "turn.failed":  # Codex
            message = str((event.get("error") or {}).get("message"))
        elif event.get("type") == "result" and event.get("is_error"):  # Claude Code
            message = str(event.get("result"))
        else:
            continue
        return message[:300] if API_FAILURE.search(message) else None
    return None


def _introspection(workspace: Path) -> list[str]:
    """Introspection names in the workspace's Python sources, for review."""
    hits = []
    for path in sorted(workspace.rglob("*.py")):
        if path.name == "replay_common.py" or "__pycache__" in path.parts:
            continue
        for match in INTROSPECTION.finditer(path.read_text(errors="replace")):
            hits.append(f"{path.relative_to(workspace)}: {match.group(0)}")
    return hits[:20]


def _downloads(transcript: Path) -> list[str]:
    """Agent commands and code that fetch data from the network, for review."""
    if not transcript.exists():
        return []
    hits = []
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        # Only what the agent wrote (tool calls and messages), not file contents it read.
        text = json.dumps(_agent_inputs(event))
        hits += [match.group(0) for match in DOWNLOAD.finditer(text)]
    return sorted(set(hits))[:20]


def _agent_inputs(event: dict) -> list:
    """Tool inputs and commands the agent issued in one transcript event (Claude Code or Codex)."""
    out = []
    message = event.get("message") if isinstance(event.get("message"), dict) else None
    if event.get("type") == "assistant" and message:
        out += [
            c.get("input") for c in message.get("content") or [] if isinstance(c, dict) and c.get("type") == "tool_use"
        ]
    item = event.get("item") if isinstance(event.get("item"), dict) else None
    if item and item.get("type") in ("command_execution", "mcp_tool_call", "file_change"):
        out.append({key: item.get(key) for key in ("command", "arguments", "changes")})
    return out


def verify(workspace: Path, run_dir: Path, task: dict, env: dict, contained=None) -> dict:
    """Run the task's verifier on the submission, sandboxed with only this task's hidden data readable.

    Tasks with ``verify_lock`` verify one submission at a time on this machine (their verifiers time the
    submission); the wait does not count against the verifier's timeout (``verify_seconds``, default 1800 s).
    """
    if not task.get("verify_lock"):
        return _verify(workspace, run_dir, task, env, contained)
    VERIFY_LOCKS.mkdir(parents=True, exist_ok=True)
    with (VERIFY_LOCKS / f"{Path(task['verifier']).parent.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _verify(workspace, run_dir, task, env, contained)


def _verify(workspace: Path, run_dir: Path, task: dict, env: dict, contained=None) -> dict:
    sandbox_root = workspace.parent
    output = sandbox_root / "verification.json"
    command = [str(PYTHON), str(ROOT / task["verifier"]), str(workspace / task["script"]), "--output", str(output)]
    if contained is not None:
        private = [PRIVATE / name for name in task["private"] if (PRIVATE / name).exists()]
        command = contained(command, extra_ro=private, harness=True)
    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    # The submission runs inside the verifier, so its detached jobs are found by the sandbox's mount namespace.
    namespace = None
    for _ in range(100):
        namespace = ti.mount_namespace(ti.cli_child(process.pid)) if contained is not None else None
        if namespace is not None or process.poll() is not None or contained is None:
            break
        time.sleep(0.1)
    try:
        stdout, stderr = process.communicate(timeout=task.get("verify_seconds", 1800))
    except subprocess.TimeoutExpired:
        _stop(process)
        return {"success": False, "error": "verification timed out"}
    finally:
        ti.kill_tagged(env["NEWTON_TRIAL_ID"], namespaces={namespace} if namespace else set())
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    (run_dir / "verification.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0 or not output.exists():
        return {"success": False, "error": (result.stderr or result.stdout)[-2000:]}
    shutil.copyfile(output, run_dir / "verification.json")
    data = json.loads(output.read_text())
    return {k: data[k] for k in ("success", "integrity", "failed_checks", "metrics", "normalized_worst")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        help="JSON file with the other options (keys as below, e.g. task, condition, model, workspace, run); "
        "keeps them out of the runner's command line, which other processes on this machine can read",
    )
    parser.add_argument(
        "--task",
        choices=(
            "grasp_drift",
            "g1_track",
            "dp_real",
            "cube_toss",
            "sdf_grind",
            "g1_hard",
            "g1_mpc",
            "abc_twin",
            "abc_arm",
            "abc_look",
            "abc_replay",
            "abc_bin",
            "abc_scratch",
        ),
    )
    parser.add_argument("--condition", choices=("mcp", "restart"))
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--workspace", type=Path, help="Run directory for the harness records")
    parser.add_argument("--seconds", type=int, default=None, help="Budget [s]; defaults to the task's (usually 1800)")
    parser.add_argument("--phase", default="loop")
    parser.add_argument("--barrier", type=Path, help="Shared directory that starts both conditions of a pair together")
    parser.add_argument("--parties", type=int, default=2)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.spec is not None:
        for key, value in json.loads(args.spec.read_text()).items():
            setattr(args, key, Path(value) if key in ("workspace", "barrier") and value else value)
    missing = [name for name in ("task", "condition", "model", "workspace") if getattr(args, name) is None]
    if missing:
        parser.error(f"missing {', '.join(missing)} (pass them as options or in --spec)")
    run_dir = Path(args.workspace).resolve()
    prepared = prepare(run_dir, args.task, args.condition, args.model, args.seconds, args.phase)
    if not args.run:
        print(f"Prepared {run_dir}")
        return
    summary = run_trial(prepared, args.barrier, args.parties)
    attempt = 1
    while (summary.get("mcp_available") is False or summary.get("api_failure")) and attempt < 3:
        # Trials run one at a time (from h12), so a retry runs under the same conditions as its partner.
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
