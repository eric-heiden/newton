# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Identical physics, controls, and scoring for live and restart trials."""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.solvers

from .calibration import CALIBRATION_SPEC
from .recording import Recording, digest

ROOT = Path(__file__).resolve().parents[2]
MENAGERIE = Path(os.environ.get("NEWTON_EVAL_MENAGERIE", "/home/horde/artifacts/newton-live-mcp/menagerie"))
HUG_DATA = Path(os.environ.get("NEWTON_EVAL_HUG_DATA", "/home/horde/repos/manosim-newton/data"))
MANOSIM = Path(os.environ.get("NEWTON_EVAL_MANOSIM", "/home/horde/repos/manosim"))

SPECS = {
    "panda_calibration": CALIBRATION_SPEC,
    "panda": {
        "description": "Track a smooth seven-joint motion under gravity with the Menagerie Panda arm.",
        "initial": {"kp": 35.0, "kd": 0.2},
        "bounds": {"kp": [1.0, 5000.0], "kd": [0.0, 300.0]},
        "thresholds": {"tracking_rmse_rad": 0.045, "tracking_p95_rad": 0.09, "max_joint_speed_rad_s": 4.0},
        "duration_s": 3.0,
        "dt_s": 0.002,
    },
    "allegro": {
        "description": "Track smooth finger flexion and hold a final posture with the Menagerie Allegro hand.",
        "initial": {"kp": 0.12, "kd": 0.002},
        "bounds": {"kp": [0.02, 40.0], "kd": [0.0, 3.0]},
        "thresholds": {"tracking_rmse_rad": 0.055, "tracking_p95_rad": 0.11, "max_joint_speed_rad_s": 5.0},
        "duration_s": 3.0,
        "dt_s": 0.002,
    },
    "hug": {
        "description": "Reconstruct a recorded HUG MANO approach to a scanned softball, then tune dynamic replay stability and wrist tracking.",
        "initial": {
            "wrist_kp": 80.0,
            "wrist_kd": 1.0,
            "finger_kp": 12.0,
            "finger_kd": 0.005,
            "finger_effort": 0.8,
            "recording_fps": 30.0,
        },
        "bounds": {
            "wrist_kp": [20.0, 4000.0],
            "wrist_kd": [0.0, 150.0],
            "finger_kp": [0.1, 40.0],
            "finger_kd": [0.0, 3.0],
            "finger_effort": [0.02, 1.0],
            "recording_fps": [5.0, 60.0],
        },
        "thresholds": {
            "wrist_rmse_m": 0.03,
            "wrist_p95_m": 0.055,
            "finger_rmse_rad": 0.25,
            "max_joint_speed_rad_s": 40.0,
            "max_wrist_speed_m_s": 2.0,
            "max_object_speed_m_s": 4.0,
            "recording_timing_error_s": 0.001,
        },
        "duration_s": 3.0,
        "dt_s": 0.002,
    },
}


def _without_actuators(path: Path) -> str:
    tree = ET.parse(path)
    root = tree.getroot()
    for child in list(root):
        if child.tag in ("actuator", "keyframe"):
            root.remove(child)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attr in ("meshdir", "texturedir"):
        compiler.set(attr, str((path.parent / compiler.get(attr, ".")).resolve()))
    if path.name == "capsule_hand.xml":
        wrist = root.find("worldbody/body")
        for joint in list(wrist.findall("joint")):
            wrist.remove(joint)
        ET.SubElement(wrist, "freejoint", name="wrist_free")
    return ET.tostring(root, encoding="unicode")


def initial_config(name: str, variant: int = 0) -> dict[str, float]:
    """Return prespecified poor initial gains for one of two paired variants."""
    values = dict(SPECS[name]["initial"])
    if name == "panda_calibration":
        if variant < 0:
            raise ValueError("calibration variant must be nonnegative")
        return values
    if variant == 1:
        for key in values:
            values[key] *= 0.75 if "kp" in key else 1.25
    elif variant != 0:
        raise ValueError("variant must be 0 or 1")
    return values


class Scenario:
    """Own one deterministic Newton simulation and its measurement history."""

    def __init__(self, name: str, config: dict | None = None, *, variant: int = 0):
        start = time.perf_counter()
        self.name, self.variant = name, variant
        self.spec = SPECS[name]
        self.dt = self.spec["dt_s"]
        self.duration = self.spec["duration_s"]
        self.steps = round(self.duration / self.dt)
        self.recording = None
        self.provenance = {}
        self.config = initial_config(name, variant)
        wp.init()
        # CPU MuJoCo avoids per-process GPU solver compilation; both conditions use this backend.
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            if name in ("panda", "allegro"):
                relative = "franka_emika_panda/panda_nohand.xml" if name == "panda" else "wonik_allegro/right_hand.xml"
                path = MENAGERIE / relative
                base = wp.transform(wp.vec3(0.0, 0.0, 0.25 if name == "allegro" else 0.0), wp.quat_identity())
                builder.add_mjcf(_without_actuators(path), xform=base, enable_self_collisions=False)
                self.provenance = {
                    "source": "https://github.com/google-deepmind/mujoco_menagerie",
                    "revision": "8161bba264d7fa7c99ca301e91e7fb44737676ad",
                    "xml": str(path),
                    "sha256": digest(path),
                    "actuation": "Replace authored actuators with Newton position drives; retain inertias and geometry.",
                }
                if name == "panda":
                    self.home = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 1.3, -0.6])
                    self.amplitude = np.array([0.18, 0.12, -0.15, 0.1, 0.12, -0.12, 0.15])
                    limits = [87.0] * 4 + [12.0] * 3
                else:
                    self.home = np.array([0.0, 0.2, 0.3, 0.25] * 3 + [0.6, 0.1, 0.2, 0.25])
                    self.amplitude = np.array([0.08, 0.45, 0.4, 0.35] * 3 + [0.2, 0.15, 0.35, 0.35])
                    limits = [0.7] * 16
                builder.joint_q[:] = self.home.tolist()
                builder.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)] * builder.joint_dof_count
                builder.joint_effort_limit[:] = limits
                builder.add_ground_plane()
                self.hand_dofs = builder.joint_dof_count
            else:
                self.recording = Recording(HUG_DATA, variant=variant)
                path = MANOSIM / "assets/myhand/mano_rhand/capsule_hand.xml"
                builder.add_mjcf(_without_actuators(path), enable_self_collisions=False)
                if builder.joint_coord_count != 67 or builder.joint_dof_count != 51:
                    raise ValueError("MANO MJCF layout changed: expected free wrist plus 15 ball joints")
                builder.joint_q[:] = self.recording.sample(0).tolist()
                builder.joint_target_mode[:] = [0] * 6 + [int(newton.JointTargetMode.POSITION)] * 45
                builder.joint_damping[:6] = [5.0] * 6
                builder.joint_armature[:6] = [0.0] * 3 + [0.01] * 3
                self.hand_dofs = 51
                self.object_body = builder.body_count
                hand_shape_count = builder.shape_count
                builder.add_mjcf(
                    str(self.recording.assets / "object.xml"),
                    xform=wp.transform(self.recording.object_position, wp.quat_identity()),
                    enable_self_collisions=False,
                )
                ground = builder.add_ground_plane()
                # The source table is finite; exclude hand contact with its infinite support-plane approximation.
                for index in range(hand_shape_count):
                    builder.add_shape_collision_filter_pair(index, ground)
                self.provenance = dict(self.recording.provenance)
                self.provenance["hand_asset"] = {
                    "path": str(path),
                    "sha256": digest(path),
                    "revision": "75c77c065c6979a6b251362fc2fe344d77737291",
                }
                self.provenance["wrist_normalization"] = {
                    "source": "Three unlimited slides plus an unlimited ball joint",
                    "import": "One equivalent free joint to avoid current importer solreflimit DOF-index failure",
                    "damping": [5.0] * 6,
                    "armature": [0.0] * 3 + [0.01] * 3,
                    "drive": "Explicit world-space PD wrench because MuJoCo free joints cannot have position actuators; translational gains tunable, rotational kp=30 N m/rad and kd=2 N m s/rad; force cap100 N and torque cap10 N m.",
                }
                self.provenance["scene_setup"] = (
                    "Dynamic free MANO wrist, ball-joint fingers, scanned convex object collision, source mass/friction, inferred planar support, no hand self-collision or hand/plane contact. Object is passive."
                )
            self.model = builder.finalize(device="cpu")
            self.state = self.model.state()
            self.state_next = self.model.state()
            self.control = self.model.control()
            self.apply_config(config or self.config, notify=False)
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=True,
                integrator="implicitfast",
                solver="newton",
                iterations=50,
                ls_iterations=20,
                nconmax=1000,
                njmax=4000,
            )
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2000)
            self.contacts = self.pipeline.contacts()
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            self.state_next.assign(self.state)
            self.initial_state = self.model.state()
            self.initial_state.assign(self.state)
            self.target_state = self.model.state()
        self.clear_metrics()
        self.build_seconds = time.perf_counter() - start

    def apply_config(self, config: dict, *, notify: bool = True) -> dict:
        """Validate and apply a complete or partial configuration atomically."""
        values = dict(self.config)
        for key, value in config.items():
            if key not in self.spec["bounds"]:
                raise ValueError(f"Unknown parameter {key!r}; allowed: {list(self.spec['bounds'])}")
            low, high = self.spec["bounds"][key]
            if not np.isfinite(value) or not low <= float(value) <= high:
                raise ValueError(f"{key} must be finite and in [{low}, {high}]")
            values[key] = float(value)
        ke, kd = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
        if self.name == "hug":
            ke[6:51], kd[6:51] = values["finger_kp"], values["finger_kd"]
            effort = self.model.joint_effort_limit.numpy()
            effort[6:51] = values["finger_effort"]
            self.model.joint_effort_limit.assign(effort)
        else:
            ke[:], kd[:] = values["kp"], values["kd"]
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)
        if notify:
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.config = values
        return dict(values)

    def clear_metrics(self) -> None:
        """Start a new measured rollout without changing parameters."""
        self.frame = 0
        self.errors, self.max_speeds, self.object_speeds, self.object_positions = [], [], [], []
        self.wrist_speeds = []
        self.joint_errors, self.pose_trace, self.target_trace = [], [], []
        self.finite = True
        self.rollout_seconds = 0.0

    def reset(self) -> None:
        """Restore the exact initial physical state and clear solver history."""
        self.state.assign(self.initial_state)
        self.state_next.assign(self.initial_state)
        self.control.joint_f.zero_()
        self.control.joint_target_q.assign(self.initial_state.joint_q)
        self.control.joint_target_qd.zero_()
        self.solver.reset(self.state)
        self.clear_metrics()

    def target(self, time_s: float) -> np.ndarray:
        """Compute the same fixed target trajectory for both conditions."""
        if self.recording is not None:
            target = self.model.joint_q.numpy().copy()
            target[:67] = self.recording.sample(time_s * self.config["recording_fps"] / self.recording.fps)
            return target
        # Smooth start and stop, followed by a one-second hold.
        phase = min(time_s / 2.0, 1.0)
        blend = 0.5 - 0.5 * np.cos(np.pi * phase)
        return self.home + self.amplitude * blend * (1.0 + 0.08 * self.variant)

    def step(self, dt: float | None = None) -> None:
        """Advance real Newton dynamics and collect fixed quality measurements."""
        start = time.perf_counter()
        dt = self.dt if dt is None else dt
        target = self.target((self.frame + 1) * dt)
        self.control.joint_target_q.assign(target)
        self.state.clear_forces()
        if self.name == "hug":
            q, qd = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
            force = np.zeros(self.model.joint_dof_count)
            mass = float(self.model.body_mass.numpy()[: self.object_body].sum())
            force[:3] = self.config["wrist_kp"] * (target[:3] - q[:3]) - self.config["wrist_kd"] * qd[:3]
            force[2] += mass * 9.81
            rotation = wp.mul(wp.quat(target[3:7]), wp.quat_inverse(wp.quat(q[3:7])))
            axis, angle = wp.quat_to_axis_angle(rotation)
            if angle > np.pi:
                angle -= 2 * np.pi
            force[3:6] = 30.0 * np.asarray(axis) * float(angle) - 2.0 * qd[3:6]
            force[:3] = np.clip(force[:3], -100, 100)
            force[3:6] = np.clip(force[3:6], -10, 10)
            self.control.joint_f.assign(force)
        self.solver.step(self.state, self.state_next, self.control, self.contacts, dt)
        self.state, self.state_next = self.state_next, self.state
        self.frame += 1
        q, qd = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
        self.finite &= bool(np.isfinite(q).all() and np.isfinite(qd).all())
        self.max_speeds.append(float(np.max(np.abs(qd[3 : self.hand_dofs] if self.name == "hug" else qd))))
        if self.name == "hug":
            reference = self.recording.sample(self.frame * dt)
            self.errors.append(float(np.linalg.norm(q[:3] - reference[:3])))
            self.wrist_speeds.append(float(np.linalg.norm(qd[:3])))
            self.object_speeds.append(float(np.linalg.norm(qd[51:54])))
            self.object_positions.append(q[67:70].tolist())
            dots = np.sum(q[7:67].reshape(15, 4) * reference[7:67].reshape(15, 4), axis=1)
            self.joint_errors.append((2 * np.arccos(np.clip(np.abs(dots), 0, 1))).tolist())
        else:
            self.errors.append((q - target).tolist())
        if self.frame % 25 == 0:
            self.pose_trace.append(self.state.body_q.numpy().tolist())
            self.target_trace.append(target.tolist())
        self.rollout_seconds += time.perf_counter() - start

    def metrics(self) -> dict:
        """Return measurements and success only for a complete finite rollout."""
        errors = np.asarray(self.errors)
        result = {
            "scenario": self.name,
            "variant": self.variant,
            "frames": self.frame,
            "sample_count": len(self.errors),
            "expected_frames": self.steps,
            "simulation_time_s": self.frame * self.dt,
            "finite": self.finite,
            "config": dict(self.config),
            "build_seconds": self.build_seconds,
            "rollout_seconds": self.rollout_seconds,
            "thresholds": self.spec["thresholds"],
        }
        if errors.size:
            prefix = "wrist" if self.name == "hug" else "tracking"
            unit = "m" if self.name == "hug" else "rad"
            result[f"{prefix}_rmse_{unit}"] = float(np.sqrt(np.mean(errors**2)))
            result[f"{prefix}_p95_{unit}"] = float(np.percentile(np.abs(errors), 95))
            result["max_joint_speed_rad_s"] = max(self.max_speeds)
            if self.name == "hug":
                result["max_object_speed_m_s"] = max(self.object_speeds)
                result["max_wrist_speed_m_s"] = max(self.wrist_speeds)
                result["finger_rmse_rad"] = float(np.sqrt(np.mean(np.asarray(self.joint_errors) ** 2)))
                result["object_max_displacement_m"] = float(
                    np.max(np.linalg.norm(np.asarray(self.object_positions) - self.recording.object_position, axis=1))
                )
                result["recorded_frame_start"] = int(self.recording.indices[0])
                result["recording_timing_error_s"] = (
                    self.frame * self.dt * abs(self.config["recording_fps"] / self.recording.fps - 1.0)
                )
        result["success"] = bool(
            self.frame == self.steps
            and len(self.errors) == self.steps
            and self.finite
            and all(result.get(key, float("inf")) <= limit for key, limit in self.spec["thresholds"].items())
        )
        return result

    def rollout(self) -> dict:
        """Reset, run one complete experiment, and return identical scoring."""
        self.reset()
        for _ in range(self.steps):
            self.step()
            if not self.finite:
                break
        return self.metrics()

    def session_step(self, session, dt: float) -> None:
        """Bind the application's state swaps to the live session."""
        self.state, self.state_next = session.state, session.state_next
        self.step(dt)
        session.state, session.state_next = self.state, self.state_next
        if self.frame == self.steps and hasattr(self, "rollout_log"):
            with self.rollout_log.open("a") as stream:
                stream.write(json.dumps(self.metrics()) + "\n")

    def session_reset(self, session) -> None:
        """Clear application measurements after the session resets dynamics."""
        self.state, self.state_next = session.state, session.state_next
        self.clear_metrics()
        self.frame = session.frame
