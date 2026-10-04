# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The ``abc_scratch`` verifier's replay loop and per-object metrics (verifier only, not in the workspace).

Adapted from ``abc_replay/replay_common.py``: the same command rule as the starter ``scene_replay.py``
(zero-order hold of the logged commands, delayed by ``command_delay``; gripper ``clip(g, 0, 1) * 0.0475`` m on
the left finger, mirrored on the right), the same CUDA-graph step (targets, ``collide``, ``solver.step``,
``assign``, record, cursor), but without named objects: the recording holds every body of every world, and
:func:`score_object` measures one matched body against one real object of ``truth.json``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import warp as wp

import newton

SIDES = ("left", "right")
GRIPPER_TRAVEL = 0.0475  # finger slide [m] at gripper opening 1
# Grasp point of a finger pad in its body frame (bodies "<side>_lf_down", "<side>_rf_down"): 25 mm from the pad
# box centre toward the fingertip.
PAD_POINT = (0.0, -0.0024, 0.071)
HAND_BODY = "{side}_link_6"
EVENT_KEYS = ("cmd_close", "contact", "liftoff", "cmd_open", "release")

HOLD_MARGIN = 0.015  # held: object centre within its real centre height + this of the pad midpoint [m]
LIFT_THRESHOLD = 0.01  # lift-off: centre this far above its start [m]
RELEASE_THRESHOLD = 0.01  # release: centre this far from its in-hand position at the carry start [m]
PLACE_MARGIN = 0.01  # placed: centre within the real tray sector dilated by this [m]
REST_HEIGHT_TOLERANCE = 0.015  # placed: centre height within this of tray floor + real centre height [m]
REST_SPEED = 0.02  # placed: mean speed over the last REST_WINDOW below this [m/s]
REST_WINDOW = 0.3  # [s]
CARRY_START_OFFSET = 3  # carry window: state samples after lift-off ...
CARRY_END_OFFSET = 2  # ... to state samples before release


def load_episode(path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        episode = {key: np.asarray(data[key]) for key in data.files}
    keys = ("t", "q", "qd", "cmd_t", "cmd", "grip_t", "grip", "grip_cmd_t", "grip_cmd")
    missing = [f"{side}_{key}" for side in SIDES for key in keys if f"{side}_{key}" not in episode]
    if missing:
        raise KeyError(f"episode is missing {missing}")
    return episode


def frame_rows(episode: dict, frames: int) -> np.ndarray:
    """State row shown by each top-camera frame: frame i shows row i - 1 (frame 0 row 0)."""
    return np.clip(np.arange(frames) - 1, 0, len(episode["left_t"]) - 1).astype(np.int64)


def measured(episode: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Measured arm joints [n, 6] [rad] and gripper opening [n] on the state times ``left_t``."""
    t = episode["left_t"]
    q, grip = episode[f"{side}_q"][:, :6], episode[f"{side}_grip"]
    ts, tg = episode[f"{side}_t"], episode[f"{side}_grip_t"]
    if len(ts) != len(t) or not np.allclose(ts, t):
        q = np.stack([np.interp(t, ts, q[:, j]) for j in range(6)], axis=-1)
    if len(tg) != len(t) or not np.allclose(tg, t):
        grip = np.interp(t, tg, grip)
    return np.asarray(q, dtype=np.float64), np.asarray(grip, dtype=np.float64)


def command_schedule(episode: dict, times: np.ndarray, delay: float = 0.0) -> dict[str, dict[str, np.ndarray]]:
    """Per side: arm targets ``q`` [T, 6] [rad] and left finger slide targets ``finger`` [T] [m] at control times."""
    times = np.asarray(times, dtype=np.float64)
    out = {}
    for side in SIDES:
        t, tg = episode[f"{side}_cmd_t"], episode[f"{side}_grip_cmd_t"]
        rows = np.clip(np.searchsorted(t, times - delay, side="right") - 1, 0, len(t) - 1)
        grip_rows = np.clip(np.searchsorted(tg, times - delay, side="right") - 1, 0, len(tg) - 1)
        out[side] = {
            "q": np.asarray(episode[f"{side}_cmd"], dtype=np.float64)[rows, :6],
            "finger": np.clip(np.asarray(episode[f"{side}_grip_cmd"], dtype=np.float64)[grip_rows], 0.0, 1.0)
            * GRIPPER_TRAVEL,
        }
    return out


# ----------------------------------------------------------------------------- geometry


def _leaf(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def quat_about_z(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from quaternions (x, y, z, w), shape [..., 4] -> [..., 3, 3]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def sector_distance(xy, tray: dict, pose: dict | None = None) -> float:
    """Distance [m] from a point to a circular-sector footprint (0 inside).

    Args:
        xy: Point [m].
        tray: ``radius_m``, ``half_angle_deg``, and (unless ``pose`` is given) ``apex_xy``, ``yaw_deg``.
        pose: ``apex_xy`` and ``yaw_deg`` overriding the tray's.
    """
    pose = tray if pose is None else pose
    apex = np.asarray(pose["apex_xy"], dtype=np.float64)
    yaw, half = math.radians(pose["yaw_deg"]), math.radians(tray["half_angle_deg"])
    radius = float(tray["radius_m"])
    d = np.asarray(xy, dtype=np.float64)[:2] - apex
    r = float(np.hypot(*d))
    angle = (math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    if abs(angle) <= half:
        return max(0.0, r - radius)
    distances = []
    for sign in (-1.0, 1.0):
        edge = radius * np.array([math.cos(yaw + sign * half), math.sin(yaw + sign * half)])
        s = float(np.clip(np.dot(d, edge) / radius**2, 0.0, 1.0))
        distances.append(float(np.linalg.norm(d - s * edge)))
    return min(distances)


def tray_distance(xy, tray: dict) -> float:
    """Distance to the real tray footprint at the start or at the end of the episode, whichever is closer [m]."""
    return min(sector_distance(xy, tray, tray[key]) for key in ("start", "end"))


# ----------------------------------------------------------------------------- replay


class _Station:
    """Per-world indices of the station joints and bodies in a batched model."""

    def __init__(self, model: newton.Model):
        joint_world, body_world = model.joint_world.numpy(), model.body_world.numpy()
        joints, bodies = {}, {}
        for i, label in enumerate(model.joint_label):
            joints.setdefault((int(joint_world[i]), _leaf(label)), []).append(i)
        for i, label in enumerate(model.body_label):
            bodies.setdefault((int(body_world[i]), _leaf(label)), []).append(i)

        def one(table, world, name, kind):
            found = table.get((world, name), [])
            if len(found) != 1:
                raise ValueError(f"world {world} must have exactly one {kind} named {name!r}, found {len(found)}")
            return found[0]

        self.worlds = model.world_count
        self.arm_joints, self.finger_joints, self.pad_bodies, self.hand_body, self.world_bodies = [], [], [], [], []
        for w in range(self.worlds):
            self.arm_joints.append({s: [one(joints, w, f"{s}_joint{j + 1}", "joint") for j in range(6)] for s in SIDES})
            self.finger_joints.append(
                {
                    s: (one(joints, w, f"{s}_left_finger", "joint"), one(joints, w, f"{s}_right_finger", "joint"))
                    for s in SIDES
                }
            )
            self.pad_bodies.append(
                {s: (one(bodies, w, f"{s}_lf_down", "body"), one(bodies, w, f"{s}_rf_down", "body")) for s in SIDES}
            )
            self.hand_body.append({s: one(bodies, w, HAND_BODY.format(side=s), "body") for s in SIDES})
            self.world_bodies.append(np.flatnonzero(body_world == w).astype(np.int64))


@wp.kernel
def _apply_targets(
    schedule: wp.array2d[wp.float32],
    step: wp.array[wp.int32],
    index: wp.array[wp.int32],
    target: wp.array[wp.float32],
):
    j = wp.tid()
    row = wp.min(step[0], schedule.shape[0] - 1)
    target[index[j]] = schedule[row, j]


@wp.kernel
def _record(
    body_q: wp.array[wp.transformf],
    joint_q: wp.array[wp.float32],
    step: wp.array[wp.int32],
    every: int,
    history_body_q: wp.array2d[wp.transformf],
    history_joint_q: wp.array2d[wp.float32],
):
    i = wp.tid()
    done = step[0] + 1  # steps completed once this step's state is written
    if done % every != 0:
        return
    row = done // every
    if row >= history_body_q.shape[0]:
        return
    if i < body_q.shape[0]:
        history_body_q[row, i] = body_q[i]
    if i < joint_q.shape[0]:
        history_joint_q[row, i] = joint_q[i]


@wp.kernel
def _advance(step: wp.array[wp.int32]):
    step[0] = step[0] + 1


class Replay:
    """Open-loop replay of one episode per world on a batched station model (see the module docstring).

    Each world starts at its episode's first state sample: arms at the measured joints and velocities, fingers
    at the measured opening, every other body where ``model.joint_q`` places it. Physics step ``k`` applies the
    schedule row ``min(k, T - 1)`` at control time ``t0 + k * dt``.
    """

    def __init__(
        self,
        model: newton.Model,
        solver,
        pipeline,
        episodes: list[dict],
        *,
        dt: float,
        command_delay: float = 0.0,
        record_dt: float = 1.0 / 120.0,
        use_graph: bool = True,
    ):
        worlds = model.world_count
        if len(episodes) != worlds:
            raise ValueError(f"{len(episodes)} episodes for {worlds} worlds")
        self.episodes = list(episodes)
        self.model, self.solver, self.collision_pipeline = model, solver, pipeline
        self.contacts = pipeline.contacts() if pipeline is not None else None
        self.dt = float(dt)
        self.use_graph = use_graph
        self.station = _Station(model)
        self.t0 = [float(e["left_t"][0]) for e in self.episodes]
        durations = [float(e["left_t"][-1]) - t0 for e, t0 in zip(self.episodes, self.t0, strict=True)]
        self.record_every = max(1, round(record_dt / self.dt))
        steps = int(math.ceil(max(durations) / self.dt - 1e-6))
        self.total_steps = -(-steps // self.record_every) * self.record_every
        device = model.device

        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        coord_layout = self.control.joint_target_q.shape[0] == model.joint_coord_count
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        q0 = model.joint_q.numpy().copy()
        qd0 = np.zeros(model.joint_dof_count, dtype=np.float32)
        steps = np.arange(self.total_steps) * self.dt
        columns, target_index, driven_dofs = [], [], []
        for w, episode in enumerate(self.episodes):
            schedule = command_schedule(episode, self.t0[w] + steps, command_delay)
            for side in SIDES:
                q_meas, grip_meas = measured(episode, side)
                qd_meas = episode.get(f"{side}_qd")
                for j, joint in enumerate(self.station.arm_joints[w][side]):
                    q0[q_start[joint]] = q_meas[0, j]
                    if qd_meas is not None:
                        qd0[qd_start[joint]] = qd_meas[0, j]
                    columns.append(schedule[side]["q"][:, j])
                    target_index.append(q_start[joint] if coord_layout else qd_start[joint])
                    driven_dofs.append(qd_start[joint])
                travel = float(np.clip(grip_meas[0], 0.0, 1.0)) * GRIPPER_TRAVEL
                for sign, joint in zip((1.0, -1.0), self.station.finger_joints[w][side], strict=True):
                    q0[q_start[joint]] = sign * travel
                    columns.append(sign * schedule[side]["finger"])
                    target_index.append(q_start[joint] if coord_layout else qd_start[joint])
                    driven_dofs.append(qd_start[joint])
        self._q0, self._qd0 = q0.astype(np.float32), qd0
        self._driven_dofs = np.asarray(driven_dofs, dtype=np.int64)
        self._target_index_host = np.asarray(target_index, dtype=np.int64)
        schedule = np.stack(columns, axis=-1).astype(np.float32)
        self._first_targets = schedule[0]
        self.schedule = wp.array(schedule, dtype=wp.float32, device=device)
        self.target_index = wp.array(self._target_index_host.astype(np.int32), dtype=wp.int32, device=device)
        rows = self.total_steps // self.record_every + 1
        self.history_body_q = wp.zeros((rows, model.body_count), dtype=wp.transformf, device=device)
        self.history_joint_q = wp.zeros((rows, model.joint_coord_count), dtype=wp.float32, device=device)
        self.step_index = wp.zeros(1, dtype=wp.int32, device=device)
        self._record_threads = max(model.body_count, model.joint_coord_count)
        self.graph = None
        self.reset()
        if use_graph and device.is_cuda:
            self._step_once()
            self.reset()
            self.capture()

    def reset(self) -> None:
        self.state_0.joint_q.assign(self._q0)
        self.state_0.joint_qd.assign(self._qd0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        target = self.control.joint_target_q.numpy()
        target[self._target_index_host] = self._first_targets
        self.control.joint_target_q.assign(target)
        target_qd = self.control.joint_target_qd.numpy()
        target_qd[self._driven_dofs] = 0.0
        self.control.joint_target_qd.assign(target_qd)
        self.step_index.zero_()
        wp.copy(self.history_body_q[0], self.state_0.body_q)
        wp.copy(self.history_joint_q[0], self.state_0.joint_q)
        if hasattr(self.solver, "reset"):
            self.solver.reset(self.state_0, flags=newton.StateFlags.NONE)
        if self.collision_pipeline is not None and hasattr(self.collision_pipeline, "reset_contact_matching"):
            self.collision_pipeline.reset_contact_matching()
        if self.contacts is not None:
            self.contacts.clear()

    def _step_once(self) -> None:
        device = self.model.device
        wp.launch(
            _apply_targets,
            dim=self.schedule.shape[1],
            inputs=[self.schedule, self.step_index, self.target_index],
            outputs=[self.control.joint_target_q],
            device=device,
        )
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
        self.state_0.assign(self.state_1)
        wp.launch(
            _record,
            dim=self._record_threads,
            inputs=[self.state_0.body_q, self.state_0.joint_q, self.step_index, self.record_every],
            outputs=[self.history_body_q, self.history_joint_q],
            device=device,
        )
        wp.launch(_advance, dim=1, inputs=[self.step_index], device=device)

    def capture(self) -> bool:
        self.graph = None
        if not (self.use_graph and self.model.device.is_cuda):
            return False
        try:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._step_once()
            self.graph = capture.graph
        except Exception as error:
            warnings.warn(f"CUDA graph capture failed, stepping without a graph: {error}", stacklevel=2)
            self.graph = None
        return self.graph is not None

    def step(self, count: int = 1) -> None:
        for _ in range(int(count)):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._step_once()

    @property
    def steps_done(self) -> int:
        return int(self.step_index.numpy()[0])

    def recordings(self) -> list[dict[str, np.ndarray]]:
        """Per world, at its episode's state times up to the current step: ``t``, ``body_q`` [n, B, 7] (the
        world's bodies, global indices in ``body_index``), ``<side>_pad`` (pad grasp-point midpoint),
        ``<side>_hand_pos``/``_quat``, ``<side>_finger``, ``<side>_q``, ``<side>_q_real``, ``<side>_grip_real``,
        ``complete``."""
        valid = min(self.steps_done // self.record_every + 1, self.history_body_q.shape[0])
        history_body = self.history_body_q.numpy()[:valid].astype(np.float64)
        history_coords = self.history_joint_q.numpy()[:valid].astype(np.float64)
        t_history = np.arange(valid) * self.record_every * self.dt
        q_start = self.model.joint_q_start.numpy()
        out = []
        for w in range(self.model.world_count):
            episode, station = self.episodes[w], self.station
            t = np.asarray(episode["left_t"], dtype=np.float64)
            rows = int(np.searchsorted(t - self.t0[w], t_history[-1] + 1e-9, side="right"))
            lower, upper, weight = _bracket(t_history, t[:rows] - self.t0[w])
            bodies = station.world_bodies[w]
            body_q = _interpolate_transforms(history_body[:, bodies], lower, upper, weight)
            column = {int(b): k for k, b in enumerate(bodies)}
            coords = history_coords[lower] * (1.0 - weight[:, None]) + history_coords[upper] * weight[:, None]
            rec = {"t": t[:rows], "body_q": body_q, "body_index": bodies, "complete": bool(rows == len(t))}
            for side in SIDES:
                points = []
                for body in station.pad_bodies[w][side]:
                    pose = body_q[:, column[body]]
                    points.append(pose[:, :3] + quat_to_matrix(pose[:, 3:]) @ np.asarray(PAD_POINT))
                rec[f"{side}_pad"] = 0.5 * (points[0] + points[1])
                hand = body_q[:, column[station.hand_body[w][side]]]
                rec[f"{side}_hand_pos"], rec[f"{side}_hand_quat"] = hand[:, :3], hand[:, 3:]
                rec[f"{side}_finger"] = coords[:, q_start[station.finger_joints[w][side][0]]]
                rec[f"{side}_q"] = coords[:, [q_start[j] for j in station.arm_joints[w][side]]]
                q_meas, grip_meas = measured(episode, side)
                rec[f"{side}_q_real"], rec[f"{side}_grip_real"] = q_meas[:rows], grip_meas[:rows]
            out.append(rec)
        return out


def _bracket(t_history: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(t_history) == 1:
        zeros = np.zeros(len(t), dtype=np.int64)
        return zeros, zeros, np.zeros(len(t))
    upper = np.clip(np.searchsorted(t_history, t, side="left"), 1, len(t_history) - 1)
    lower = upper - 1
    weight = np.clip((t - t_history[lower]) / (t_history[upper] - t_history[lower]), 0.0, 1.0)
    return lower, upper, weight


def _interpolate_transforms(history: np.ndarray, lower, upper, weight) -> np.ndarray:
    a, b = history[lower], history[upper]
    w = weight[:, None, None]
    out = np.empty_like(a)
    out[..., :3] = a[..., :3] * (1.0 - w) + b[..., :3] * w
    qb = np.where(np.sum(a[..., 3:] * b[..., 3:], axis=-1, keepdims=True) < 0.0, -b[..., 3:], b[..., 3:])
    q = a[..., 3:] * (1.0 - w) + qb * w
    out[..., 3:] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return out


# ----------------------------------------------------------------------------- metrics


def _first(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def _gt_rows(values: np.ndarray, rows: np.ndarray, count: int) -> np.ndarray:
    out = np.full((count, 3), np.nan)
    for frame, row in enumerate(rows[: len(values)]):
        if row < count:
            out[row] = values[frame]
    return out


def object_track(rec: dict, body: int, centre_local: np.ndarray) -> np.ndarray:
    """World positions [n, 3] of a body's geometric centre (``centre_local`` in its body frame)."""
    column = int(np.flatnonzero(rec["body_index"] == body)[0])
    pose = rec["body_q"][:, column]
    return pose[:, :3] + quat_to_matrix(pose[:, 3:]) @ np.asarray(centre_local, dtype=np.float64)


def score_object(
    rec: dict, position: np.ndarray, truth: dict, tray: dict, table_z: float, gt_track: np.ndarray | None
) -> dict:
    """Metrics of one simulated object (centre track ``position`` [n, 3]) against one real object.

    Event frames of ``truth`` are top frames, converted to state rows (frame i shows row i - 1). The carry
    window runs from lift-off + 3 to release - 2 rows, inclusive.

    - ``held_fraction``: carry rows with the centre within the real centre height + 15 mm of the grasping
      arm's pad midpoint; ``max_rise_m``: highest centre above its start; ``moved_before_grasp_m``.
    - ``placed``: complete episode, final centre within the real tray footprint (start or end pose) + 10 mm,
      within 15 mm of the height of the real tray floor + centre height, and slower than 2 cm/s over the last
      0.3 s; ``final_xy_err_m`` to the real final centre, ``rest_xy_err_m`` to the nearest centre at which the
      real object rested after its release (before or after other objects pushed it).
    - report: ``carry_track_err_m`` (median xy distance to the tracked carry, FK-attached), ``liftoff_err_rows``,
      ``release_err_s``, ``grip_gap_err_mm``, ``slip_m``.
    """
    side, events, radius = truth["arm"], truth["events"], float(truth["centre_height_m"])
    t = rec["t"]
    count = len(t)
    frames = max(events.values()) + 1
    top_row = np.clip(np.arange(frames) - 1, 0, max(count - 1, 0))
    row = {key: int(top_row[events[key]]) for key in EVENT_KEYS}
    start = position[0]
    m: dict = {"start_xyz": [float(v) for v in start]}
    close = min(row["cmd_close"], count - 1)
    m["moved_before_grasp_m"] = float(np.linalg.norm(position[: close + 1, :2] - start[:2], axis=1).max())
    rise = position[:, 2] - start[2]
    m["max_rise_m"] = float(rise.max()) if np.all(np.isfinite(position)) else math.inf

    first, last = row["liftoff"] + CARRY_START_OFFSET, row["release"] - CARRY_END_OFFSET
    carry = np.arange(first, last + 1) if last < count else np.zeros(0, dtype=np.int64)
    distance = np.linalg.norm(position - rec[f"{side}_pad"], axis=1)
    m["held_fraction"] = float(np.mean(distance[carry] <= radius + HOLD_MARGIN)) if carry.size else 0.0
    hand_rotation = quat_to_matrix(rec[f"{side}_hand_quat"])
    in_hand = np.einsum("nji,nj->ni", hand_rotation, position - rec[f"{side}_hand_pos"])
    m["slip_m"] = float(np.linalg.norm(in_hand[carry] - in_hand[first], axis=1).max()) if carry.size else None

    m["carry_track_err_m"] = None
    if gt_track is not None and carry.size:
        track = _gt_rows(gt_track, np.clip(np.arange(len(gt_track)) - 1, 0, None), count)
        known = carry[np.isfinite(track[carry, 0])]
        if known.size:
            m["carry_track_err_m"] = float(np.median(np.linalg.norm(position[known, :2] - track[known, :2], axis=1)))

    lift = _first(rise[close:] > LIFT_THRESHOLD)
    m["liftoff_err_rows"] = None if lift is None else int(close + lift - row["liftoff"])
    m["release_err_s"] = None
    if carry.size:
        moved = _first(np.linalg.norm(in_hand[first:] - in_hand[first], axis=1) > RELEASE_THRESHOLD)
        if moved is not None and row["release"] < count:
            m["release_err_s"] = float(t[first + moved] - t[row["release"]])
    hold = np.arange(row["liftoff"], min(row["cmd_open"], count))
    m["grip_gap_err_mm"] = None
    if hold.size:
        gap_sim = 2.0 * rec[f"{side}_finger"][hold]
        gap_real = 2.0 * np.clip(rec[f"{side}_grip_real"][hold], 0.0, 1.0) * GRIPPER_TRAVEL
        m["grip_gap_err_mm"] = float(1000.0 * np.median(gap_sim - gap_real))

    final = position[-1]
    window = max(0, int(np.searchsorted(t, t[-1] - REST_WINDOW, side="right")) - 1)
    elapsed = t[-1] - t[window]
    m["final_xyz"] = [float(v) for v in final]
    finite = bool(np.all(np.isfinite(final)))
    m["tray_distance_m"] = tray_distance(final[:2], tray) if finite else math.inf
    m["final_height_err_m"] = float(final[2] - (table_z + tray["floor_height_m"] + radius))
    m["final_speed_mps"] = float(np.linalg.norm(final - position[window]) / elapsed) if elapsed > 0 else None
    m["placed"] = bool(
        rec["complete"]
        and finite
        and m["tray_distance_m"] <= PLACE_MARGIN
        and abs(m["final_height_err_m"]) <= REST_HEIGHT_TOLERANCE
        and m["final_speed_mps"] is not None
        and m["final_speed_mps"] < REST_SPEED
    )
    real = np.asarray(truth["final_xyz"], dtype=np.float64)
    m["final_xy_err_m"] = float(np.linalg.norm(final[:2] - real[:2])) if finite else math.inf
    rest = np.asarray(truth.get("rest_xy") or [real[:2]], dtype=np.float64)
    m["rest_xy_err_m"] = float(np.linalg.norm(rest - final[:2], axis=1).min()) if finite else math.inf
    return m
