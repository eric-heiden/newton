# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Optional observation of MPC candidates; never feeds back into the optimizer."""

import mujoco_warp as mjw
import numpy as np
import warp as wp

from newton.examples.robot.wbc_controller import G1_HEAD_OFFSET

# Body origins, plus a virtual head center rigidly attached to the torso.
G1_TRACE_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "head",
    "torso_link",
)
TRACE_COLORS = np.array(
    [(0.2, 0.55, 1.0), (0.3, 0.9, 0.35), (0.1, 0.9, 0.95), (1.0, 0.4, 0.18), (0.95, 0.8, 0.3), (0.8, 0.8, 0.85)]
)


@wp.kernel
def _record(
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    bodies: wp.array[int],
    local: wp.array[wp.vec3],
    offset: int,
    step: int,
    positions: wp.array3d[wp.vec3],
):
    world, body = wp.tid()
    raw = xquat[world, bodies[body]]
    rotation = wp.quat(raw[1], raw[2], raw[3], raw[0])
    positions[offset + world, step, body] = xpos[world, bodies[body]] + wp.quat_rotate(rotation, local[body])


@wp.kernel
def _selection(
    costs: wp.array[float],
    best: wp.array[int],
    difference_best: wp.array[int],
    offset: int,
    coordinate: bool,
    selected: wp.array[int],
):
    index = offset + best[0]
    value = float(1.0e20)
    if index < costs.shape[0]:
        value = costs[index]
    if coordinate and difference_best[0] < offset:
        probe = difference_best[0]
        if costs[probe] < value:
            index = probe
            value = costs[probe]
    selected[0] = -1
    if value < 1.0e19:
        selected[0] = index


class RolloutTraces:
    """Record the final search iteration inside its existing CUDA graph.

    Attach to ``mpc.traces`` before capture. Device writes observe the original
    physical rollouts, including the coordinate fallback. CPU readback and
    thinning happen only in snapshot(), after optimization, for visualization.
    """

    def __init__(self, mpc, bodies=G1_TRACE_BODIES, *, horizon=0.4, stride=4):
        if not np.isfinite(horizon) or horizon <= 0 or stride < 1:
            raise ValueError("Trace horizon and stride must be positive")
        self.names = tuple(bodies)
        ids, local = [], []
        for name in bodies:
            target = "torso_link" if name == "head" else name
            matches = [i for i in range(mpc.cpu_model.nbody) if mpc.cpu_model.body(i).name.endswith(target)]
            if len(matches) != 1:
                raise ValueError(f"Expected one trace body matching {name!r}")
            ids.append(matches[0])
            local.append(G1_HEAD_OFFSET if name == "head" else (0.0, 0.0, 0.0))
        if not ids:
            raise ValueError("At least one trace body is required")
        last = min(mpc.steps, max(1, int(np.floor(horizon / mpc.dt + 1e-6))))
        indices = sorted({0, last, *range(stride, last + 1, stride)})
        self.step_index = {step: i for i, step in enumerate(indices)}
        self.offsets = np.array(indices) * mpc.dt
        self.batches = getattr(mpc, "trace_batches", None)
        if self.batches is None:
            self.batches = (
                ((mpc.diff_data, mpc.diff_costs), (mpc.line_data, mpc.line_costs))
                if hasattr(mpc, "diff_data")
                else ((mpc.data, mpc.costs),)
            )
        self.line_offset = sum(cost.size for _, cost in self.batches[:-1])
        count = sum(cost.size for _, cost in self.batches)
        self.coordinate = bool(self.line_offset and getattr(mpc, "coordinate_search", False))
        with wp.ScopedDevice(mpc.device):
            self.bodies = wp.array(ids, dtype=int)
            self.local = wp.array(local, dtype=wp.vec3)
            self.positions = wp.zeros((count, len(indices), len(ids)), dtype=wp.vec3)
            self.costs = wp.zeros(count)
            self.selected = wp.full(1, -1, dtype=int)
            self.qpos = wp.zeros((1, mpc.cpu_model.nq))
            self.time = wp.zeros(1)

    def record(self, mpc, step, *, batch_offset=None):
        if step not in self.step_index:
            return
        if step == 0:
            mjw.kinematics(mpc.model, mpc.data)
        offset = 0
        for data, costs in self.batches:
            if data is mpc.data:
                break
            offset += costs.size
        if batch_offset is not None:
            offset = batch_offset
        wp.launch(
            _record,
            (mpc.samples, len(self.names)),
            inputs=[mpc.data.xpos, mpc.data.xquat, self.bodies, self.local, offset, self.step_index[step]],
            outputs=[self.positions],
        )

    def finish(self, mpc, q, clock):
        offset = 0
        for _, costs in self.batches:
            wp.copy(self.costs, costs, dest_offset=offset)
            offset += costs.size
        wp.launch(
            _selection,
            1,
            inputs=[
                self.costs,
                mpc.best,
                mpc.difference_best if self.coordinate else mpc.best,
                self.line_offset,
                self.coordinate,
            ],
            outputs=[self.selected],
        )
        wp.copy(self.qpos, q)
        wp.copy(self.time, clock)

    def snapshot(self, count=4):
        """Selected plan first, then spatially diverse valid candidates.

        Farthest-point thinning is solely a display choice, not an optimization
        step or a confidence interval. Missing candidates are padded with NaNs.
        """
        if count < 1:
            raise ValueError("Trace count must be positive")
        positions, costs = self.positions.numpy(), self.costs.numpy()
        selected = int(self.selected.numpy()[0])
        indices = np.full(count, -1, dtype=int)
        paths = np.full((count, *positions.shape[1:]), np.nan, dtype=np.float32)
        values = np.full(count, np.nan, dtype=np.float32)
        if selected >= 0:
            first = 0 if self.coordinate else self.line_offset
            valid = np.isfinite(positions).all(axis=(1, 2, 3)) & (costs < 1e19)
            valid[:first] = False
            distance = np.full(len(costs), np.inf)
            for i in range(min(count, int(valid.sum()))):
                indices[i], paths[i], values[i] = selected, positions[selected], costs[selected]
                valid[selected] = False
                distance = np.minimum(distance, np.mean((positions - positions[selected]) ** 2, axis=(1, 2, 3)))
                distance[~valid] = -np.inf
                selected = int(np.argmax(distance))
        return {
            "time": float(self.time.numpy()[0]),
            "qpos": self.qpos.numpy()[0],
            "positions": paths,
            "indices": indices,
            "costs": values,
        }


def draw_rollouts(viewer, positions, offsets, *, elapsed=0.0):
    """Draw solid selected futures and muted, dashed alternative futures.

    The optional elapsed time clips away already elapsed prediction segments.
    Used by both the live example and offline replay of recorded predictions.
    """
    starts, ends, colors = [], [], []
    for candidate in reversed(range(len(positions))):
        path = positions[candidate]
        for step in range(len(offsets) - 1):
            if offsets[step + 1] <= elapsed or (candidate and step % 2):
                continue
            alpha = np.clip((elapsed - offsets[step]) / (offsets[step + 1] - offsets[step]), 0, 1)
            for body in range(path.shape[1]):
                a, b = path[step, body], path[step + 1, body]
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    continue
                starts.append((1 - alpha) * a + alpha * b)
                ends.append(b)
                color = TRACE_COLORS[body % len(TRACE_COLORS)]
                colors.append(color if candidate == 0 else 0.45 * color + 0.22)
    if not starts:
        viewer.log_lines("/mpc/futures", None, None, None)
        return
    arrays = [wp.array(x, dtype=wp.vec3, device=viewer.device) for x in (starts, ends, colors)]
    viewer.log_lines("/mpc/futures", *arrays)
