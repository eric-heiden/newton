# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fit full effective Panda dynamics to real recorded joint observations.

Candidates must explain measured torque and predict independent short forward
windows. Link parameters are physically constrained but not uniquely identifiable;
filtered measured joint torque is not a reconstructed motor command.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.solvers

from .real_robot_data import DATASET_DOI, file_digest, load_reference
from .real_robot_model import BOUNDS, make_builder, physical_coefficients, validate_config
from .real_robot_model import initial_config as _initial_config
from .real_robot_regressor import load_regressor

THRESHOLDS = {
    "max_joint_torque_rmse_nm": 0.5,
    "max_joint_torque_normalized_rmse": 0.5,
    "max_joint_position_rmse_rad": 0.025,
    "max_joint_velocity_rmse_rad_s": 0.5,
    "position_p95_rad": 0.05,
    "max_joint_speed_rad_s": 5.0,
}


def initial_config(variant: int = 0) -> dict:
    """Return the same clean placeholders for every independent agent replicate."""
    if not isinstance(variant, int) or variant < 0:
        raise ValueError("variant is a nonnegative independent replicate identifier")
    return _initial_config()


REAL_SPEC = {
    "description": "Identify seven full physical link inertias, masses, centers of mass and joint losses from real measured Panda recordings, starting from geometry and homogeneous placeholders. Fit measured torque and Newton short-horizon forward trajectories. Final verification uses every recording in the independent published test fold.",
    "initial": initial_config(),
    "bounds": dict(BOUNDS),
    "thresholds": dict(THRESHOLDS),
    "dt_s": 0.002,
    "duration_s": 3.6,
    "expected_frames": 1800,
    "window_steps": 50,
    "windows_per_episode": 12,
    "training_episodes": 3,
    "heldout_episodes": 16,
    "synthetic": False,
}


def prepare_windows(reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Interpolate the fixed 100 ms windows using the recorded irregular time base."""
    result = {key: [] for key in ("initial_q", "initial_qd", "q", "qd", "tau", "time", "episode_ids")}
    for episode, start, stop in zip(
        reference["episode_ids"], reference["episode_offsets"][:-1], reference["episode_offsets"][1:], strict=True
    ):
        timestamps = reference["time"][start:stop]
        for offset in np.linspace(30, stop - start - 37, 12, dtype=np.int64):
            result["initial_q"].append(reference["q"][start + offset])
            result["initial_qd"].append(reference["qd"][start + offset])
            end_times = timestamps[offset] + np.arange(1, 51) * 0.002
            mid_times = end_times - 0.001
            if end_times[-1] > timestamps[-1]:
                raise ValueError("The fixed forward window exceeds the recorded episode")
            for key, times in (("q", end_times), ("qd", end_times), ("tau", mid_times)):
                result[key].extend(
                    np.stack(
                        [np.interp(times, timestamps, reference[key][start:stop, joint]) for joint in range(7)], axis=1
                    )
                )
            result["time"].extend(end_times)
            result["episode_ids"].extend([episode] * 50)
    return {key: np.asarray(value) for key, value in result.items()}


class RealRobotScenario:
    """Own the identical measured-data physical checks used by every tool condition."""

    dt = 0.002
    window_steps = 50
    initial = REAL_SPEC["initial"]
    bounds = REAL_SPEC["bounds"]
    validate_config = staticmethod(validate_config)

    def __init__(
        self,
        config: dict | None = None,
        *,
        reference_file: Path | None = None,
        variant: int = 0,
        geometry_file: Path | None = None,
        regressor_file: Path | None = None,
    ):
        started = time.perf_counter()
        self.name, self.variant = "panda_real", variant
        self.config = validate_config(initial_config(variant) if config is None else config)
        if reference_file is None:
            raise ValueError("Real Panda identification requires an explicit measured reference archive")
        self.reference_file = Path(reference_file).resolve()
        self.geometry_file = Path(
            geometry_file
            or os.environ.get("NEWTON_EVAL_REAL_GEOMETRY")
            or self.reference_file.parent / "geometry/panda_geometry.xml"
        ).resolve()
        self.regressor_file = Path(
            regressor_file or self.reference_file.with_name(self.reference_file.stem + "-regressor.npz")
        ).resolve()
        self.reference = load_reference(self.reference_file)
        self.matrix, self.torque_target, self.torque_episode_ids = load_regressor(
            self.regressor_file, self.reference_file, self.geometry_file, self.reference
        )
        self.windows = prepare_windows(self.reference)
        self.episodes = self.reference["episode_ids"].tolist()
        self.steps = len(self.windows["time"])
        self.duration = self.steps * self.dt
        self.spec = REAL_SPEC | {"expected_frames": self.steps, "duration_s": self.duration}
        with wp.ScopedDevice("cpu"):
            builder, self.link_bodies = make_builder(self.geometry_file, self.config)
            builder.joint_q[:] = self.windows["initial_q"][0].tolist()
            self.model = builder.finalize(device="cpu")
            self.state, self.state_next = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_cpu=True, integrator="implicitfast", solver="newton", iterations=50
            )
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2000)
            self.contacts = self.pipeline.contacts()
        self.provenance = {
            "dataset_doi": DATASET_DOI,
            "reference_sha256": file_digest(self.reference_file),
            "regressor_sha256": file_digest(self.regressor_file),
            "geometry_sha256": file_digest(self.geometry_file),
            "synthetic": False,
            "description": "Real publisher-filtered measured Panda joint observations; effective dynamics, not unique physical identification or raw actuator-command recovery",
            "episodes": self.episodes,
        }
        self.reset()
        self.build_seconds = time.perf_counter() - started

    def apply_config(self, config: dict) -> dict:
        """Apply a complete physical candidate, then notify the native solver of every change."""
        values = validate_config(self.config | config)
        mass, inverse_mass = self.model.body_mass.numpy().copy(), self.model.body_inv_mass.numpy().copy()
        com = self.model.body_com.numpy().copy()
        inertia, inverse_inertia = self.model.body_inertia.numpy().copy(), self.model.body_inv_inertia.numpy().copy()
        for joint, body in enumerate(self.link_bodies):
            mass[body], inverse_mass[body] = values["mass"][joint], 1.0 / values["mass"][joint]
            com[body], inertia[body] = values["com"][joint], values["inertia"][joint]
            inverse_inertia[body] = np.linalg.inv(inertia[body])
        self.model.body_mass.assign(mass)
        self.model.body_inv_mass.assign(inverse_mass)
        self.model.body_com.assign(com)
        self.model.body_inertia.assign(inertia)
        self.model.body_inv_inertia.assign(inverse_inertia)
        self.model.joint_damping.assign(np.asarray(values["viscous"], dtype=np.float32))
        self.model.joint_friction.assign(np.asarray(values["coulomb"], dtype=np.float32))
        self.model.joint_armature.assign(np.asarray(values["armature"], dtype=np.float32))
        self.solver.notify_model_changed(
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES
        )
        self.config = values
        return dict(values)

    def _reset_window(self, window: int) -> None:
        """Set measured initial conditions and clear native history before a forward window."""
        self.state.joint_q.assign(self.windows["initial_q"][window].astype(np.float32))
        self.state.joint_qd.assign(self.windows["initial_qd"][window].astype(np.float32))
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.state.clear_forces()
        self.state_next.assign(self.state)
        self.control.joint_f.zero_()
        self.solver.reset(self.state, flags=newton.StateFlags.NONE)

    def clear_metrics(self) -> None:
        """Discard all previous candidate samples without altering physical parameters."""
        self.frame = 0
        self.q_trace, self.qd_trace, self.body_trace, self.applied_torque_trace = [], [], [], []
        self.pose_trace = self.body_trace
        self.finite = True
        self.rollout_seconds = 0.0
        self.last_trace_path = None
        self._invalid_history = False

    def reset(self) -> None:
        """Begin a new complete candidate at the first measured initial state."""
        self.clear_metrics()
        self._reset_window(0)

    def step(self) -> None:
        """Advance one fixed Newton step with interpolated measured generalized torque."""
        started = time.perf_counter()
        if self._invalid_history:
            raise ValueError("Reset the complete candidate after restoring a nonzero-frame checkpoint")
        if self.frame >= self.steps:
            raise ValueError("The candidate is complete; reset before another rollout")
        if self.frame % self.window_steps == 0:
            self._reset_window(self.frame // self.window_steps)
        torque = self.windows["tau"][self.frame] - np.asarray(self.config["torque_bias"])
        self.control.joint_f.assign(torque.astype(np.float32))
        self.state.clear_forces()
        self.solver.step(self.state, self.state_next, self.control, None, self.dt)
        self.state, self.state_next = self.state_next, self.state
        q, qd = self.state.joint_q.numpy().copy(), self.state.joint_qd.numpy().copy()
        self.finite &= bool(np.isfinite(q).all() and np.isfinite(qd).all())
        self.q_trace.append(q)
        self.qd_trace.append(qd)
        self.applied_torque_trace.append(torque)
        self.frame += 1
        if self.frame % 25 == 0:
            self.body_trace.append(self.state.body_q.numpy().copy())
        self.rollout_seconds += time.perf_counter() - started

    def metrics(self) -> dict:
        """Require all six physical quality thresholds both pooled and in every recording."""
        if self._invalid_history:
            raise ValueError("Reset the complete candidate after restoring a nonzero-frame checkpoint")
        result = {
            "scenario": self.name,
            "variant": self.variant,
            "config": dict(self.config),
            "frames": self.frame,
            "sample_count": len(self.q_trace),
            "expected_frames": self.steps,
            "simulation_time_s": self.frame * self.dt,
            "finite": self.finite,
            "build_seconds": self.build_seconds,
            "rollout_seconds": self.rollout_seconds,
            "thresholds": dict(THRESHOLDS),
            "episodes": self.episodes,
            "reference_sha256": self.provenance["reference_sha256"],
            "trace_path": self.last_trace_path,
        }
        torque_error = (self.matrix @ physical_coefficients(self.config) - self.torque_target).reshape(-1, 7)
        measured_torque = self.torque_target.reshape(-1, 7)
        normalized_error = np.empty_like(torque_error)
        for episode in self.episodes:
            selected = self.torque_episode_ids == episode
            normalized_error[selected] = torque_error[selected] / np.maximum(measured_torque[selected].std(axis=0), 0.5)
        count = len(self.q_trace)
        position_error = np.asarray(self.q_trace).reshape(-1, 7) - self.windows["q"][:count]
        velocity_error = np.asarray(self.qd_trace).reshape(-1, 7) - self.windows["qd"][:count]
        velocity = np.asarray(self.qd_trace).reshape(-1, 7)

        def score(torque_mask: np.ndarray, forward_mask: np.ndarray) -> dict:
            torque_rmse = np.sqrt(np.mean(torque_error[torque_mask] ** 2, axis=0))
            normalized_rmse = np.sqrt(np.mean(normalized_error[torque_mask] ** 2, axis=0))
            values = {
                "torque_rmse_per_joint_nm": torque_rmse.tolist(),
                "torque_normalized_rmse_per_joint": normalized_rmse.tolist(),
                "max_joint_torque_rmse_nm": float(torque_rmse.max()),
                "max_joint_torque_normalized_rmse": float(normalized_rmse.max()),
            }
            if np.any(forward_mask):
                position_rmse = np.sqrt(np.mean(position_error[forward_mask] ** 2, axis=0))
                velocity_rmse = np.sqrt(np.mean(velocity_error[forward_mask] ** 2, axis=0))
                values.update(
                    {
                        "position_rmse_per_joint_rad": position_rmse.tolist(),
                        "velocity_rmse_per_joint_rad_s": velocity_rmse.tolist(),
                        "max_joint_position_rmse_rad": float(position_rmse.max()),
                        "max_joint_velocity_rmse_rad_s": float(velocity_rmse.max()),
                        "position_p95_rad": float(np.percentile(np.abs(position_error[forward_mask]), 95)),
                        "max_joint_speed_rad_s": float(np.max(np.abs(velocity[forward_mask]))),
                    }
                )
            values["success"] = bool(all(values.get(key, float("inf")) <= limit for key, limit in THRESHOLDS.items()))
            return values

        result.update(score(np.ones(len(torque_error), dtype=bool), np.ones(count, dtype=bool)))
        result["per_episode"] = []
        for episode in self.episodes:
            forward_mask = self.windows["episode_ids"][:count] == episode
            item = score(self.torque_episode_ids == episode, forward_mask)
            item.update({"episode": episode, "sample_count": int(forward_mask.sum())})
            item["success"] &= item["sample_count"] == 600
            result["per_episode"].append(item)
        complete = self.frame == self.steps and count == self.steps and self.finite
        result["success"] = bool(
            complete and result["success"] and all(item["success"] for item in result["per_episode"])
        )
        return result

    def save_trace(self, path: Path) -> None:
        """Retain complete measured and simulated numeric traces for candidate auditing."""
        if self._invalid_history:
            raise ValueError("Reset the complete candidate after restoring a nonzero-frame checkpoint")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = len(self.q_trace)
        np.savez_compressed(
            path,
            q=np.asarray(self.q_trace),
            qd=np.asarray(self.qd_trace),
            reference_q=self.windows["q"][:count],
            reference_qd=self.windows["qd"][:count],
            measured_torque=self.windows["tau"][:count],
            applied_torque=np.asarray(self.applied_torque_trace),
            recorded_time=self.windows["time"][:count],
            episode_ids=self.windows["episode_ids"][:count],
            window_index=np.arange(count) // self.window_steps,
            body_q=np.asarray(self.body_trace),
            dt=self.dt,
            trace_step_interval=25,
            torque_prediction=(self.matrix @ physical_coefficients(self.config)).reshape(-1, 7),
            torque_target=self.torque_target.reshape(-1, 7),
            torque_episode_ids=self.torque_episode_ids,
        )
        self.last_trace_path = str(path.resolve())

    def rollout(self) -> dict:
        """Evaluate one full physical candidate and retain every measurement."""
        self.reset()
        for _ in range(self.steps):
            self.step()
        return self.metrics()

    def session_step(self, session, dt: float) -> None:
        """Run the identical fixed-window callback through the live session."""
        if dt != self.dt:
            raise ValueError("The measured-data evaluation timestep is fixed")
        self.state, self.state_next = session.state, session.state_next
        self.step()
        session.state, session.state_next = self.state, self.state_next
        if self.frame == self.steps and hasattr(self, "rollout_log"):
            index = len(self.rollout_log.read_text().splitlines()) if self.rollout_log.exists() else 0
            self.save_trace(self.rollout_log.parent / f"candidate-{index + 1:03d}.npz")
            with self.rollout_log.open("a") as stream:
                stream.write(json.dumps(self.metrics()) + "\n")

    def session_reset(self, session) -> None:
        """Clear measurement history after a session reset or checkpoint restore."""
        self.state, self.state_next = session.state, session.state_next
        self.clear_metrics()
        self.frame = session.frame
        if session.frame != 0:
            self._invalid_history = True
            raise ValueError(
                "Scored candidates require complete measurement history; restore a frame-zero checkpoint or reset"
            )
