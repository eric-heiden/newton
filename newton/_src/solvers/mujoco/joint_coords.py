# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side (float64) conversion between Newton joint coordinates and MuJoCo qpos/qvel.

Mirrors ``convert_warp_coords_to_mj_kernel`` and ``convert_mj_coords_to_warp_kernel``
in :mod:`.kernels` for arbitrary batches of states, without touching solver data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...sim import JointType

if TYPE_CHECKING:
    from ...sim import Model


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product over the last axis (broadcasting); cheaper than np.cross on small arrays."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions (broadcasting)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


_CONJ = np.array([-1.0, -1.0, -1.0, 1.0])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the unit xyzw quaternion ``q`` (broadcasting)."""
    u = q[..., :3]
    t = 2.0 * _cross(u, v)
    return v + q[..., 3:4] * t + _cross(u, t)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


_XYZW_TO_WXYZ = [3, 0, 1, 2]
_WXYZ_TO_XYZW = [1, 2, 3, 0]


def _as_host_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, wp.array):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension, got a scalar.")
    return array


def _spans(starts: list[int], count: int) -> np.ndarray:
    """Index matrix ``starts[:, None] + arange(count)``, shape [len(starts), count]."""
    return np.asarray(starts, dtype=np.int64).reshape(-1, 1) + np.arange(count)


class JointCoordinateMap:
    """Index tables relating a model's joint coordinates to the solver's compiled MuJoCo model.

    Joint frames, COM offsets, and ``mujoco:dof_ref`` are read from the model on
    every conversion, as the solver's kernels do, so runtime model edits apply.
    """

    def __init__(
        self,
        model: Model,
        mj_q_start: np.ndarray,
        mj_qd_start: np.ndarray,
        nq: int,
        nv: int,
        qpos0: np.ndarray,
    ):
        self.model = model
        # SolverMuJoCo maps every Newton world to its own MuJoCo world (a single-world
        # model is one MuJoCo world either way); world w's joints follow world w-1's.
        self.world_count = int(model.world_count)
        self.nq = int(nq)
        self.nv = int(nv)
        joints_per_world = len(mj_q_start)
        if joints_per_world * self.world_count != model.joint_count:
            raise ValueError(
                f"Cannot map {model.joint_count} joints onto {self.world_count} MuJoCo worlds "
                f"of {joints_per_world} joints each."
            )

        joint_type = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_child = model.joint_child.numpy()

        free = {key: [] for key in ("joint", "child", "q", "qd", "mq", "mqd")}
        ball = {key: [] for key in ("joint", "q", "qd", "mq", "mqd")}
        scalar = {key: [] for key in ("q", "qd", "mq", "mqd")}
        for world in range(self.world_count):
            for template_joint in range(joints_per_world):
                mq = int(mj_q_start[template_joint])
                if mq < 0:
                    continue  # loop joint: no MuJoCo coordinates
                mq += world * self.nq
                mqd = int(mj_qd_start[template_joint]) + world * self.nv
                joint = world * joints_per_world + template_joint
                jtype = int(joint_type[joint])
                q0 = int(joint_q_start[joint])
                qd0 = int(joint_qd_start[joint])
                if jtype == JointType.FREE:
                    entry = (joint, int(joint_child[joint]), q0, qd0, mq, mqd)
                    for key, value in zip(free, entry, strict=True):
                        free[key].append(value)
                elif jtype == JointType.BALL:
                    for key, value in zip(ball, (joint, q0, qd0, mq, mqd), strict=True):
                        ball[key].append(value)
                elif jtype in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6):
                    for axis in range(int(joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1])):
                        for key, value in zip(scalar, (q0, qd0, mq, mqd), strict=True):
                            scalar[key].append(value + axis)

        self.free_joint = np.asarray(free["joint"], dtype=np.int64)
        self.free_child = np.asarray(free["child"], dtype=np.int64)
        self.free_q_pos = _spans(free["q"], 3)
        self.free_q_rot = _spans(free["q"], 7)[:, 3:]
        self.free_qd_lin = _spans(free["qd"], 3)
        self.free_qd_ang = _spans(free["qd"], 6)[:, 3:]
        self.free_mq_pos = _spans(free["mq"], 3)
        self.free_mq_rot = _spans(free["mq"], 7)[:, 3:]
        self.free_mqd_lin = _spans(free["mqd"], 3)
        self.free_mqd_ang = _spans(free["mqd"], 6)[:, 3:]

        self.ball_joint = np.asarray(ball["joint"], dtype=np.int64)
        self.ball_q = _spans(ball["q"], 4)
        self.ball_qd = _spans(ball["qd"], 3)
        self.ball_mq = _spans(ball["mq"], 4)
        self.ball_mqd = _spans(ball["mqd"], 3)

        self.scalar_q, self.scalar_qd, self.scalar_mq, self.scalar_mqd = (
            np.asarray(scalar[key], dtype=np.int64) for key in ("q", "qd", "mq", "mqd")
        )

        # Coordinates without a counterpart keep defaults: MuJoCo's qpos0 on the MuJoCo
        # side and the model's joint_q / joint_qd (loop joints) on the Newton side.
        covered_qpos = np.zeros(self.world_count * self.nq, dtype=bool)
        for index in (self.scalar_mq, self.free_mq_pos, self.free_mq_rot, self.ball_mq):
            covered_qpos[index.ravel()] = True
        self.qpos_uncovered = np.flatnonzero(~covered_qpos)
        self.qpos_default = np.tile(np.asarray(qpos0, dtype=np.float64), self.world_count)[self.qpos_uncovered]

        covered_q = np.zeros(model.joint_coord_count, dtype=bool)
        for index in (self.scalar_q, self.free_q_pos, self.free_q_rot, self.ball_q):
            covered_q[index.ravel()] = True
        covered_qd = np.zeros(model.joint_dof_count, dtype=bool)
        for index in (self.scalar_qd, self.free_qd_lin, self.free_qd_ang, self.ball_qd):
            covered_qd[index.ravel()] = True
        self.joint_q_uncovered = np.flatnonzero(~covered_q)
        self.joint_qd_uncovered = np.flatnonzero(~covered_qd)

    # -- model parameters, read at call time ---------------------------------

    def _dof_ref(self) -> np.ndarray | float:
        mujoco_attrs = getattr(self.model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None
        if dof_ref is None:
            return 0.0
        return dof_ref.numpy()[self.scalar_qd].astype(np.float64)

    def _free_frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_p = self.model.joint_X_p.numpy()[self.free_joint].astype(np.float64)
        X_c = self.model.joint_X_c.numpy()[self.free_joint].astype(np.float64)
        # Renormalize the float32 rotations so that both directions are exact inverses.
        X_p[:, 3:] = _quat_normalize(X_p[:, 3:])
        X_c[:, 3:] = _quat_normalize(X_c[:, 3:])
        com = self.model.body_com.numpy()[self.free_child].astype(np.float64)
        return X_p, X_c, com

    def _ball_child_rotation(self) -> np.ndarray:
        return _quat_normalize(self.model.joint_X_c.numpy()[self.ball_joint, 3:].astype(np.float64))

    # -- shape handling ---------------------------------------------------------

    @staticmethod
    def _split(array: np.ndarray, size: int, world_shape: tuple[int, int] | None, name: str):
        """Return ``(rows [batch, size], batch_shape)``; also accepts a trailing ``world_shape``."""
        if array.shape[-1] == size:
            return array.reshape(-1, size), array.shape[:-1]
        if world_shape is not None and array.ndim >= 2 and array.shape[-2:] == world_shape:
            return array.reshape(-1, size), array.shape[:-2]
        expected = f"[..., {size}]"
        if world_shape is not None:
            expected += f" or [..., {world_shape[0]}, {world_shape[1]}]"
        raise ValueError(f"{name} has shape {array.shape}, expected {expected}.")

    @staticmethod
    def _broadcast(a: np.ndarray, a_batch, b: np.ndarray | None, b_batch, names: tuple[str, str]):
        """Broadcast the batch dimensions of two row-flattened arrays against each other."""
        if b is None or a_batch == b_batch:
            return a, b, a_batch
        try:
            batch = np.broadcast_shapes(a_batch, b_batch)
        except ValueError:
            raise ValueError(
                f"{names[0]} and {names[1]} have incompatible batch shapes {a_batch} and {b_batch}."
            ) from None
        a = np.broadcast_to(a.reshape((*a_batch, -1)), (*batch, a.shape[-1])).reshape(-1, a.shape[-1])
        b = np.broadcast_to(b.reshape((*b_batch, -1)), (*batch, b.shape[-1])).reshape(-1, b.shape[-1])
        return a, b, batch

    # -- conversions --------------------------------------------------------------

    def to_mujoco(self, joint_q: Any, joint_qd: Any | None) -> tuple[np.ndarray, np.ndarray | None]:
        model = self.model
        q, batch = self._split(_as_host_array(joint_q, "joint_q"), model.joint_coord_count, None, "joint_q")
        qd = None
        if joint_qd is not None:
            qd, qd_batch = self._split(_as_host_array(joint_qd, "joint_qd"), model.joint_dof_count, None, "joint_qd")
            q, qd, batch = self._broadcast(q, batch, qd, qd_batch, ("joint_q", "joint_qd"))

        rows = q.shape[0]
        qpos = np.empty((rows, self.world_count * self.nq))
        qpos[:, self.qpos_uncovered] = self.qpos_default
        qvel = np.zeros((rows, self.world_count * self.nv)) if qd is not None else None

        if len(self.scalar_q):
            qpos[:, self.scalar_mq] = q[:, self.scalar_q] + self._dof_ref()
            if qd is not None:
                qvel[:, self.scalar_mqd] = qd[:, self.scalar_qd]

        if len(self.free_joint):
            X_p, X_c, com = self._free_frames()
            joint_r = q[:, self.free_q_rot]
            # Child body world pose X_p * X_joint * inv(X_c); FREE parents are the world.
            inv_c_r = X_c[:, 3:] * _CONJ
            mid_p = q[:, self.free_q_pos] - _quat_rotate(_quat_mul(joint_r, inv_c_r), X_c[:, :3])
            world_r = _quat_mul(X_p[:, 3:], _quat_mul(joint_r, inv_c_r))
            qpos[:, self.free_mq_pos] = X_p[:, :3] + _quat_rotate(X_p[:, 3:], mid_p)
            qpos[:, self.free_mq_rot] = world_r[..., _XYZW_TO_WXYZ]
            if qd is not None:
                rot = _quat_normalize(world_r)
                v_com = _quat_rotate(X_p[:, 3:], qd[:, self.free_qd_lin])
                w_world = _quat_rotate(X_p[:, 3:], qd[:, self.free_qd_ang])
                qvel[:, self.free_mqd_lin] = v_com - _cross(w_world, _quat_rotate(rot, com))
                qvel[:, self.free_mqd_ang] = _quat_rotate(rot * _CONJ, w_world)

        if len(self.ball_joint):
            q_c = self._ball_child_rotation()
            r = q[:, self.ball_q]
            qpos[:, self.ball_mq] = _quat_mul(_quat_mul(q_c, r), q_c * _CONJ)[..., _XYZW_TO_WXYZ]
            if qd is not None:
                to_mj = _quat_mul(q_c, _quat_normalize(r) * _CONJ)
                qvel[:, self.ball_mqd] = _quat_rotate(to_mj, qd[:, self.ball_qd])

        qpos = qpos.reshape((*batch, -1))
        if qvel is not None:
            qvel = qvel.reshape((*batch, -1))
        return qpos, qvel

    def from_mujoco(self, qpos: Any, qvel: Any | None) -> tuple[np.ndarray, np.ndarray | None]:
        model = self.model
        nworld = self.world_count
        pos, batch = self._split(_as_host_array(qpos, "qpos"), nworld * self.nq, (nworld, self.nq), "qpos")
        vel = None
        if qvel is not None:
            vel, vel_batch = self._split(_as_host_array(qvel, "qvel"), nworld * self.nv, (nworld, self.nv), "qvel")
            pos, vel, batch = self._broadcast(pos, batch, vel, vel_batch, ("qpos", "qvel"))

        rows = pos.shape[0]
        joint_q = np.empty((rows, model.joint_coord_count))
        if len(self.joint_q_uncovered):
            joint_q[:, self.joint_q_uncovered] = model.joint_q.numpy()[self.joint_q_uncovered]
        joint_qd = None
        if vel is not None:
            joint_qd = np.empty((rows, model.joint_dof_count))
            if len(self.joint_qd_uncovered):
                joint_qd[:, self.joint_qd_uncovered] = model.joint_qd.numpy()[self.joint_qd_uncovered]

        if len(self.scalar_q):
            joint_q[:, self.scalar_q] = pos[:, self.scalar_mq] - self._dof_ref()
            if vel is not None:
                joint_qd[:, self.scalar_qd] = vel[:, self.scalar_mqd]

        if len(self.free_joint):
            X_p, X_c, com = self._free_frames()
            world_r = pos[:, self.free_mq_rot][..., _WXYZ_TO_XYZW]
            # X_joint = inv(X_p) * X_world * X_c.
            inv_p_r = X_p[:, 3:] * _CONJ
            mid_p = pos[:, self.free_mq_pos] + _quat_rotate(world_r, X_c[:, :3])
            joint_q[:, self.free_q_pos] = _quat_rotate(inv_p_r, mid_p - X_p[:, :3])
            joint_q[:, self.free_q_rot] = _quat_mul(inv_p_r, _quat_mul(world_r, X_c[:, 3:]))
            if vel is not None:
                rot = _quat_normalize(world_r)
                w_world = _quat_rotate(rot, vel[:, self.free_mqd_ang])
                v_com = vel[:, self.free_mqd_lin] + _cross(w_world, _quat_rotate(rot, com))
                joint_qd[:, self.free_qd_lin] = _quat_rotate(inv_p_r, v_com)
                joint_qd[:, self.free_qd_ang] = _quat_rotate(inv_p_r, w_world)

        if len(self.ball_joint):
            q_c = self._ball_child_rotation()
            to_anchor = _quat_mul(q_c * _CONJ, pos[:, self.ball_mq][..., _WXYZ_TO_XYZW])
            joint_q[:, self.ball_q] = _quat_mul(to_anchor, q_c)
            if vel is not None:
                joint_qd[:, self.ball_qd] = _quat_rotate(_quat_normalize(to_anchor), vel[:, self.ball_mqd])

        joint_q = joint_q.reshape((*batch, -1))
        if joint_qd is not None:
            joint_qd = joint_qd.reshape((*batch, -1))
        return joint_q, joint_qd
