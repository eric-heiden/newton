# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Synthetic dynamics identification on the real Menagerie Panda asset.

Reference parameters are supplied by an external study generator, never
embedded in the common forward-model source or reference trace archive.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.solvers

from .recording import digest


class CalibrationScenario:
    """Own the fixed controller and three public calibration trajectories."""

    bounds: ClassVar[dict[str, tuple[float, float]]] = {
        "payload_mass": (0.2, 2.0),
        "damping_multiplier": (0.5, 2.0),
        "joint_friction": (0.0, 0.3),
    }
    initial: ClassVar[dict[str, float]] = {
        "payload_mass": 0.3,
        "damping_multiplier": 0.6,
        "joint_friction": 0.02,
    }
    dt = 0.002
    duration = 3.0
    steps = 1500
    episode_steps = 1500
    payload_radius = 0.035
    home = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 1.3, -0.6])
    amplitude = np.array([0.24, 0.22, 0.18, 0.18, 0.18, 0.20, 0.25])

    @classmethod
    def validate_config(cls, config: dict) -> dict[str, float]:
        """Validate all physical parameters without changing the simulator."""
        if set(config) != set(cls.bounds):
            raise ValueError(f"A complete configuration requires {sorted(cls.bounds)}")
        values = {key: float(value) for key, value in config.items()}
        for key, (low, high) in cls.bounds.items():
            if not np.isfinite(values[key]) or not low <= values[key] <= high:
                raise ValueError(f"{key} must be finite and in [{low}, {high}]")
        return values

    def __init__(self, config: dict | None = None, *, reference_file: Path | None = None, variant: int = 0):
        from .scenarios import MENAGERIE, _without_actuators  # noqa: PLC0415

        start = time.perf_counter()
        self.name, self.variant = "panda_calibration", variant
        self.spec = CALIBRATION_SPEC
        self.reference_file = None if reference_file is None else Path(reference_file).resolve()
        self.reference_q = None
        self.episodes = (0,)
        if self.reference_file is not None:
            with np.load(self.reference_file, allow_pickle=False) as reference:
                self.episodes = tuple(int(value) for value in reference["episodes"])
                self.reference_q = np.asarray(reference["q"], dtype=np.float64).copy()
            if self.episodes not in ((0, 1), (2,)):
                raise ValueError("Reference episodes must be training (0, 1) or held-out verification (2,)")
            if self.reference_q.shape != (len(self.episodes), self.episode_steps, 7):
                raise ValueError("Reference q must have shape [episode_count, 1500, 7]")
            if not np.isfinite(self.reference_q).all():
                raise ValueError("Reference positions must be finite")
        self.steps = len(self.episodes) * self.episode_steps
        self.duration = self.steps * self.dt
        self.config = self.validate_config(self.initial if config is None else config)
        path = MENAGERIE / "franka_emika_panda/panda_nohand.xml"
        root = ET.fromstring(_without_actuators(path))
        attachment = root.find(".//body[@name='attachment']")
        if attachment is None:
            raise ValueError("The Menagerie Panda attachment body is required")
        payload = ET.SubElement(attachment, "body", name="calibration_payload", pos="0 0 0.05")
        inertia = 0.4 * self.config["payload_mass"] * self.payload_radius**2
        ET.SubElement(
            payload,
            "inertial",
            pos="0 0 0",
            mass=str(self.config["payload_mass"]),
            diaginertia=" ".join([str(inertia)] * 3),
        )
        ET.SubElement(payload, "geom", type="sphere", size=str(self.payload_radius), contype="0", conaffinity="0")
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            builder.add_mjcf(ET.tostring(root, encoding="unicode"), enable_self_collisions=False)
            if builder.joint_dof_count != 7:
                raise ValueError("The Panda calibration asset must have seven movable joints")
            self.base_damping = np.array(builder.joint_damping, dtype=np.float32)
            if np.any(self.base_damping <= 0):
                raise ValueError("The asset's authored viscous damping must be positive")
            builder.joint_q[:] = self.home.tolist()
            builder.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)] * 7
            builder.joint_target_ke[:] = [140.0, 140.0, 100.0, 100.0, 45.0, 35.0, 25.0]
            builder.joint_target_kd[:] = [10.0, 10.0, 8.0, 8.0, 3.0, 3.0, 2.0]
            builder.joint_effort_limit[:] = [87.0] * 4 + [12.0] * 3
            builder.joint_damping[:] = (self.base_damping * self.config["damping_multiplier"]).tolist()
            builder.joint_friction[:] = [self.config["joint_friction"]] * 7
            self.model = builder.finalize(device="cpu")
            matches = [i for i, label in enumerate(self.model.body_label) if label.endswith("calibration_payload")]
            if len(matches) != 1:
                raise ValueError("The payload body must remain separately identifiable")
            self.payload_body = matches[0]
            self.state, self.state_next = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_cpu=True, integrator="implicitfast", solver="newton", iterations=50
            )
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2000)
            self.contacts = self.pipeline.contacts()
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            self.initial_state = self.model.state()
            self.initial_state.assign(self.state)
        self.episode = 0
        self.provenance = {
            "source": "https://github.com/google-deepmind/mujoco_menagerie",
            "revision": "8161bba264d7fa7c99ca301e91e7fb44737676ad",
            "asset_sha256": digest(path),
            "reference_sha256": None if self.reference_file is None else digest(self.reference_file),
            "synthetic": True,
            "description": "Newton-generated joint observations on a real imported Panda asset; not physical-robot measurements.",
            "attachment": "A solid sphere of radius 0.035 m, 0.05 m along the source attachment's local z axis.",
            "observations": "Position observations have prescribed Gaussian noise with standard deviation 0.0002 rad.",
            "episodes": list(self.episodes),
        }
        self.reset()
        self.build_seconds = time.perf_counter() - start

    def apply_config(self, config: dict) -> dict[str, float]:
        """Update payload inertia and joint losses for the next candidate."""
        values = self.validate_config(self.config | config)
        mass = self.model.body_mass.numpy().copy()
        inverse_mass = self.model.body_inv_mass.numpy().copy()
        inertia = self.model.body_inertia.numpy().copy()
        inverse_inertia = self.model.body_inv_inertia.numpy().copy()
        index = self.payload_body
        mass[index] = values["payload_mass"]
        inverse_mass[index] = 1.0 / mass[index]
        inertia[index] = np.eye(3) * (0.4 * mass[index] * self.payload_radius**2)
        inverse_inertia[index] = np.linalg.inv(inertia[index])
        self.model.body_mass.assign(mass)
        self.model.body_inv_mass.assign(inverse_mass)
        self.model.body_inertia.assign(inertia)
        self.model.body_inv_inertia.assign(inverse_inertia)
        self.model.joint_damping.assign(self.base_damping * values["damping_multiplier"])
        self.model.joint_friction.assign(np.full(7, values["joint_friction"], dtype=np.float32))
        self.solver.notify_model_changed(
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES
        )
        self.config = values
        return dict(values)

    @classmethod
    def target(cls, time_s: float, episode: int) -> np.ndarray:
        """Return a prescribed bidirectional motion followed by a hold."""
        if episode not in (0, 1, 2):
            raise ValueError("episode must be 0, 1, or 2")
        phase = min(time_s / 2.5, 1.0)
        envelope = np.sin(np.pi * phase) ** 2
        frequency = (0.65, 1.0, 0.8)[episode]
        offsets = np.arange(7) * (0.35, -0.4, 0.55)[episode]
        wave = np.sin(2.0 * np.pi * frequency * time_s + offsets)
        return cls.home + cls.amplitude * envelope * wave

    def _reset_physics(self) -> None:
        """Clear physical history between episodes without erasing observations."""
        self.state.assign(self.initial_state)
        self.state_next.assign(self.initial_state)
        self.control.joint_f.zero_()
        self.control.joint_target_q.assign(self.initial_state.joint_q)
        self.control.joint_target_qd.zero_()
        self.solver.reset(self.state)

    def clear_metrics(self) -> None:
        """Discard measurement history without changing the physical state."""
        self.frame = 0
        self.q_trace, self.qd_trace, self.target_trace, self.body_trace = [], [], [], []
        self.pose_trace = self.body_trace
        self.errors = []
        self.finite = True
        self.rollout_seconds = 0.0
        self.last_trace_path = None

    def reset(self) -> None:
        """Restore the initial physical state and all candidate observations."""
        self._reset_physics()
        self.clear_metrics()

    def step(self) -> None:
        """Advance fixed physics and retain identical numerical observations."""
        start = time.perf_counter()
        if self.frame >= self.steps:
            raise ValueError("The candidate is complete; reset before another rollout")
        episode_index, episode_frame = divmod(self.frame, self.episode_steps)
        if self.frame > 0 and episode_frame == 0:
            self._reset_physics()
        self.episode = self.episodes[episode_index]
        target = self.target((episode_frame + 1) * self.dt, self.episode)
        self.control.joint_target_q.assign(target)
        self.state.clear_forces()
        self.solver.step(self.state, self.state_next, self.control, None, self.dt)
        self.state, self.state_next = self.state_next, self.state
        q, qd = self.state.joint_q.numpy().copy(), self.state.joint_qd.numpy().copy()
        self.finite &= bool(np.isfinite(q).all() and np.isfinite(qd).all())
        self.q_trace.append(q)
        self.qd_trace.append(qd)
        self.target_trace.append(target)
        if self.reference_q is not None:
            self.errors.append(q - self.reference_q[episode_index, episode_frame])
        self.frame += 1
        if self.frame % 25 == 0:
            self.body_trace.append(self.state.body_q.numpy().copy())
        self.rollout_seconds += time.perf_counter() - start

    def rollout_episode(self, episode: int) -> dict:
        """Run one episode and expose observations without any hidden answer."""
        self.target(0.0, episode)
        if self.reference_q is not None:
            raise ValueError("Episode-only generation is unavailable in a scored task")
        previous_episodes, previous_steps = self.episodes, self.steps
        try:
            self.episodes, self.steps = (episode,), self.episode_steps
            self.reset()
            for _ in range(self.steps):
                self.step()
                if not self.finite:
                    break
            return {
                "q": np.asarray(self.q_trace),
                "qd": np.asarray(self.qd_trace),
                "target_q": np.asarray(self.target_trace),
                "body_q": np.asarray(self.body_trace),
                "finite": self.finite,
                "frames": self.frame,
                "episode": episode,
            }
        finally:
            self.episodes, self.steps = previous_episodes, previous_steps

    def metrics(self) -> dict:
        """Score every complete episode against the supplied synthetic response."""
        errors = np.asarray(self.errors)
        result = {
            "scenario": self.name,
            "variant": self.variant,
            "config": dict(self.config),
            "frames": self.frame,
            "sample_count": len(self.errors),
            "expected_frames": self.steps,
            "simulation_time_s": self.frame * self.dt,
            "finite": self.finite,
            "build_seconds": self.build_seconds,
            "rollout_seconds": self.rollout_seconds,
            "thresholds": self.spec["thresholds"],
            "reference_sha256": self.provenance["reference_sha256"],
            "episodes": list(self.episodes),
            "trace_path": self.last_trace_path,
        }
        complete = self.frame == self.steps and len(self.errors) == self.steps and self.finite
        per_episode = []
        if errors.size:
            result["trajectory_rmse_rad"] = float(np.sqrt(np.mean(errors**2)))
            result["trajectory_p95_rad"] = float(np.percentile(np.abs(errors), 95))
            result["max_joint_speed_rad_s"] = float(np.max(np.abs(self.qd_trace)))
            result["per_joint_rmse_rad"] = np.sqrt(np.mean(errors**2, axis=0)).tolist()
            for index, episode in enumerate(self.episodes):
                segment = errors[index * self.episode_steps : (index + 1) * self.episode_steps]
                if len(segment) != self.episode_steps:
                    continue
                per_episode.append(
                    {
                        "episode": episode,
                        "trajectory_rmse_rad": float(np.sqrt(np.mean(segment**2))),
                        "trajectory_p95_rad": float(np.percentile(np.abs(segment), 95)),
                    }
                )
        result["per_episode"] = per_episode
        result["success"] = bool(
            complete
            and all(result.get(key, float("inf")) <= limit for key, limit in self.spec["thresholds"].items())
            and len(per_episode) == len(self.episodes)
            and all(
                item[key] <= self.spec["thresholds"][key]
                for item in per_episode
                for key in ("trajectory_rmse_rad", "trajectory_p95_rad")
            )
        )
        return result

    def save_trace(self, path: Path) -> None:
        """Export candidate observations for identical offline analysis access."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q=np.asarray(self.q_trace),
            qd=np.asarray(self.qd_trace),
            target_q=np.asarray(self.target_trace),
            body_q=np.asarray(self.body_trace),
            errors=np.asarray(self.errors),
            episodes=np.asarray(self.episodes),
            dt=self.dt,
            trace_step_interval=25,
        )
        self.last_trace_path = str(path.resolve())

    def rollout(self) -> dict:
        """Run a complete candidate, resetting between its prescribed episodes."""
        self.reset()
        for _ in range(self.steps):
            self.step()
            if not self.finite:
                break
        return self.metrics()

    def session_step(self, session, dt: float) -> None:
        """Advance the application through the same two-episode control path."""
        if dt != self.dt:
            raise ValueError("Calibration timestep is fixed")
        self.state, self.state_next = session.state, session.state_next
        self.step()
        session.state, session.state_next = self.state, self.state_next
        if self.frame == self.steps and hasattr(self, "rollout_log"):
            index = len(self.rollout_log.read_text().splitlines()) if self.rollout_log.exists() else 0
            self.save_trace(self.rollout_log.parent / f"candidate-{index + 1:02d}.npz")
            with self.rollout_log.open("a") as stream:
                stream.write(json.dumps(self.metrics()) + "\n")

    def session_reset(self, session) -> None:
        """Discard stale measurements after reset or checkpoint restoration."""
        self.state, self.state_next = session.state, session.state_next
        self.clear_metrics()
        self.frame = session.frame


CALIBRATION_SPEC = {
    "description": "Identify payload mass, viscous joint damping, and Coulomb joint friction from two synthetic Newton-generated Panda response traces, using fixed controls. Final verification tests a third response withheld from the agent.",
    "initial": dict(CalibrationScenario.initial),
    "bounds": dict(CalibrationScenario.bounds),
    "thresholds": {
        "trajectory_rmse_rad": 0.00035,
        "trajectory_p95_rad": 0.0008,
        "max_joint_speed_rad_s": 5.0,
    },
    "duration_s": 6.0,
    "dt_s": 0.002,
    "expected_frames": 3000,
    "episode_steps": 1500,
    "synthetic_position_noise_sd_rad": 0.0002,
}
