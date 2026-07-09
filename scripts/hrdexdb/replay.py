# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dynamic replay of HRDexDB episodes in Newton with SolverMuJoCo.

The robot is driven exclusively through PD position targets
(:attr:`Control.joint_target_q`); the object is a passive free body. Metrics
compare the simulated object trajectory against the ground-truth 6D tracking.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton

from dataset import ARM_DOF, load_episode  # noqa: E402
from scene import SceneInfo, SimParams, build_scene  # noqa: E402

RESULTS_ROOT = Path(__file__).parent / "results"


@wp.kernel
def _set_targets(
    targets: wp.array2d(dtype=wp.float32),
    dof_map: wp.array(dtype=wp.int32),
    step: wp.array(dtype=wp.int32),
    coords_per_world: int,
    num_dofs: int,
    # outputs
    joint_target_q: wp.array(dtype=wp.float32),
):
    tid = wp.tid()  # one thread per (world, data dof)
    world = tid // num_dofs
    col = tid - world * num_dofs
    t = wp.min(step[0], targets.shape[0] - 1)
    joint_target_q[world * coords_per_world + dof_map[col]] = targets[t, col]


@wp.kernel
def _record_state(
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=wp.float32),
    dof_map: wp.array(dtype=wp.int32),
    step: wp.array(dtype=wp.int32),
    record_every: int,
    object_body: int,
    num_dofs: int,
    # outputs
    rec_obj: wp.array2d(dtype=wp.float32),
    rec_q: wp.array2d(dtype=wp.float32),
):
    i = step[0]
    if i % record_every != 0:
        return
    r = i / record_every
    if r >= rec_obj.shape[0]:
        return
    tf = body_q[object_body]
    for k in range(3):
        rec_obj[r, k] = tf[k]
    for k in range(4):
        rec_obj[r, 3 + k] = tf[3 + k]
    for d in range(num_dofs):
        rec_q[r, d] = joint_q[dof_map[d]]


@wp.kernel
def _record_state_batch(
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=wp.float32),
    dof_map: wp.array(dtype=wp.int32),
    step: wp.array(dtype=wp.int32),
    record_every: int,
    object_body: int,
    bodies_per_world: int,
    coords_per_world: int,
    num_dofs: int,
    # outputs
    rec_obj: wp.array3d(dtype=wp.float32),
    rec_q: wp.array3d(dtype=wp.float32),
):
    w = wp.tid()
    i = step[0]
    if i % record_every != 0:
        return
    r = i / record_every
    if r >= rec_obj.shape[0]:
        return
    tf = body_q[w * bodies_per_world + object_body]
    for k in range(7):
        rec_obj[r, w, k] = tf[k]
    for d in range(num_dofs):
        rec_q[r, w, d] = joint_q[w * coords_per_world + dof_map[d]]


@wp.kernel
def _advance_step(step: wp.array(dtype=wp.int32)):
    step[0] = step[0] + 1


@dataclass
class ReplayResult:
    t: np.ndarray
    obj_pos_sim: np.ndarray
    obj_quat_sim: np.ndarray
    obj_pos_gt: np.ndarray
    obj_quat_gt: np.ndarray
    joint_q_sim: np.ndarray
    joint_q_ref: np.ndarray
    metrics: dict

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t=self.t,
            obj_pos_sim=self.obj_pos_sim,
            obj_quat_sim=self.obj_quat_sim,
            obj_pos_gt=self.obj_pos_gt,
            obj_quat_gt=self.obj_quat_gt,
            joint_q_sim=self.joint_q_sim,
            joint_q_ref=self.joint_q_ref,
            metrics=json.dumps(self.metrics),
        )


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(q1 * q2, axis=-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def compute_metrics(
    t: np.ndarray,
    pos_sim: np.ndarray,
    quat_sim: np.ndarray,
    pos_gt: np.ndarray,
    quat_gt: np.ndarray,
    joint_q_sim: np.ndarray,
    joint_q_ref: np.ndarray,
    mesh_vertices: np.ndarray,
    grasp_success: bool,
    table_height: float,
) -> dict:
    from scipy.spatial.transform import Rotation as R

    pos_err = np.linalg.norm(pos_sim - pos_gt, axis=1)
    rot_err = quat_geodesic_deg(quat_sim, quat_gt)

    # ADD: mean vertex distance between sim and gt object pose (subsampled mesh)
    v = mesh_vertices[:: max(1, len(mesh_vertices) // 500)]
    Rs = R.from_quat(quat_sim).as_matrix()
    Rg = R.from_quat(quat_gt).as_matrix()
    add = np.array(
        [np.linalg.norm((v @ Rs[i].T + pos_sim[i]) - (v @ Rg[i].T + pos_gt[i]), axis=1).mean() for i in range(len(t))]
    )

    # Episodes are pick-and-place: judge lift over the whole trajectory and
    # compare peak heights rather than final position only.
    lift_thresh = table_height + 0.06
    sim_lifted = bool((pos_sim[:, 2] > lift_thresh).any())
    gt_lifted = bool((pos_gt[:, 2] > lift_thresh).any())
    z_peak_err = float(pos_sim[:, 2].max() - pos_gt[:, 2].max())

    joint_rmse_arm = float(np.sqrt(np.mean((joint_q_sim[:, :ARM_DOF] - joint_q_ref[:, :ARM_DOF]) ** 2)))
    joint_rmse_hand = float(np.sqrt(np.mean((joint_q_sim[:, ARM_DOF:] - joint_q_ref[:, ARM_DOF:]) ** 2)))

    return {
        "pos_rmse": float(np.sqrt(np.mean(pos_err**2))),
        "pos_err_final": float(pos_err[-1]),
        "pos_err_max": float(pos_err.max()),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_err**2))),
        "rot_err_final_deg": float(rot_err[-1]),
        "add_rmse": float(np.sqrt(np.mean(add**2))),
        "add_mean": float(add.mean()),
        "add_final": float(add[-1]),
        "sim_lifted": sim_lifted,
        "gt_lifted": gt_lifted,
        "z_peak_err": z_peak_err,
        "gt_grasp_success": bool(grasp_success),
        "lift_match": sim_lifted == gt_lifted,
        "joint_rmse_arm": joint_rmse_arm,
        "joint_rmse_hand": joint_rmse_hand,
    }


class Replayer:
    """Runs one episode dynamically; optionally renders through a viewer."""

    def __init__(
        self,
        ep,
        params: SimParams | None = None,
        target_source: str = "cmd",
        substeps: int = 4,
        max_faces: int = 8000,
        solver_kwargs: dict | None = None,
    ):
        newton.use_coord_layout_targets = True
        self.ep = ep
        self.params = params or SimParams()
        self.info: SceneInfo = build_scene(ep, self.params, max_faces=max_faces)
        self.model = self.info.model
        self.control_dt = float(ep.t[1] - ep.t[0])
        self.substeps = substeps
        self.sim_dt = self.control_dt / substeps

        self.targets_np = np.asarray(ep.q_cmd if target_source == "cmd" else ep.q_meas, dtype=np.float32)
        self.targets = wp.array2d(self.targets_np, dtype=wp.float32)
        self.dof_map_wp = wp.array(self.info.dof_map, dtype=wp.int32)
        self.step_idx = wp.zeros(1, dtype=wp.int32)

        kwargs = dict(
            solver="newton",
            integrator="implicitfast",
            njmax=800,
            nconmax=2000,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
        )
        if solver_kwargs:
            kwargs.update(solver_kwargs)
        self.solver = newton.solvers.SolverMuJoCo(self.model, **kwargs)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.obj_q_index = None  # body_q row of the object body
        self.graph = None

    def _control_step(self, record_every: int, rec_obj, rec_q):
        wp.launch(
            _set_targets,
            dim=len(self.info.dof_map),
            inputs=[self.targets, self.dof_map_wp, self.step_idx, 0, len(self.info.dof_map)],
            outputs=[self.control.joint_target_q],
        )
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            _record_state,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.state_0.joint_q,
                self.dof_map_wp,
                self.step_idx,
                record_every,
                self.info.object_body,
                len(self.info.dof_map),
            ],
            outputs=[rec_obj, rec_q],
        )
        wp.launch(_advance_step, dim=1, inputs=[self.step_idx])

    def palm_object_min_dist(self, samples: int = 40) -> float:
        """Minimum distance between the palm (via FK on the reference
        trajectory) and the ground-truth object center. Large values indicate
        a camera-to-robot calibration outlier: the recorded hand never comes
        near where the object supposedly is."""
        ep = self.ep
        idx = np.linspace(0, len(ep.t) - 1, samples).astype(int)
        state = self.model.state()
        qbuf = self.model.joint_q.numpy().copy()
        gt = ep.obj_poses_at(ep.t[idx])
        dists = []
        for j, i in enumerate(idx):
            for col, dof in enumerate(self.info.dof_map):
                qbuf[dof] = ep.q_meas[i, col]
            for f, leader_col, mult, off in self.info.mimic_dofs:
                qbuf[f] = mult * ep.q_meas[i, leader_col] + off
            q_wp = wp.array(qbuf, dtype=self.model.joint_q.dtype)
            newton.eval_fk(self.model, q_wp, self.model.joint_qd, state)
            palm = state.body_q.numpy()[self.info.palm_body][:3]
            dists.append(np.linalg.norm(palm - gt[j, :3, 3]))
        return float(min(dists))

    def run(self, record_every: int = 3, viewer=None, gt_overlay=None) -> ReplayResult:
        ep = self.ep
        n_steps = len(ep.t)
        n_rec = (n_steps + record_every - 1) // record_every
        rec_obj = wp.zeros((n_rec, 7), dtype=wp.float32)
        rec_q = wp.zeros((n_rec, len(self.info.dof_map)), dtype=wp.float32)
        self.step_idx.zero_()

        # An even number of substeps keeps state_0/state_1 identity stable
        # across graph replays; enforce for capture.
        use_graph = wp.get_device().is_cuda and viewer is None and self.substeps % 2 == 0
        t_start = time.perf_counter()
        if use_graph:
            with wp.ScopedCapture() as capture:
                self._control_step(record_every, rec_obj, rec_q)
            for _ in range(n_steps):
                wp.capture_launch(capture.graph)
        else:
            for i in range(n_steps):
                self._control_step(record_every, rec_obj, rec_q)
                if viewer is not None:
                    viewer.begin_frame(float(ep.t[i]))
                    viewer.log_state(self.state_0)
                    if gt_overlay is not None:
                        gt_overlay(float(ep.t[i]))
                    viewer.end_frame()
        obj = rec_obj.numpy()
        joint_rec = rec_q.numpy()
        self.wall_time = time.perf_counter() - t_start

        t = ep.t[::record_every][: len(obj)]
        pos_sim, quat_sim = obj[:, :3], obj[:, 3:]
        gt = ep.obj_poses_at(t)
        from scipy.spatial.transform import Rotation as R

        pos_gt = gt[:, :3, 3]
        quat_gt = R.from_matrix(gt[:, :3, :3]).as_quat()

        joint_q_sim = joint_rec
        ref = ep.q_meas[np.searchsorted(ep.t, t).clip(0, len(ep.t) - 1)]

        metrics = compute_metrics(
            t,
            pos_sim,
            quat_sim,
            pos_gt,
            quat_gt,
            joint_q_sim,
            ref,
            np.asarray(self.info.object_mesh_newton.vertices),
            ep.grasp_success,
            self.info.table_height,
        )
        metrics["wall_time"] = self.wall_time
        metrics["palm_obj_min_dist"] = self.palm_object_min_dist()
        metrics["calib_outlier"] = metrics["palm_obj_min_dist"] > 0.25
        return ReplayResult(t, pos_sim, quat_sim, pos_gt, quat_gt, joint_q_sim, ref, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="allegro_v5")
    parser.add_argument("--object", default="banana")
    parser.add_argument("--scene", default="2")
    parser.add_argument("--target-source", default="cmd", choices=["cmd", "meas"])
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--params", type=str, default=None, help="JSON file with SimParams")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    ep = load_episode(args.hand, args.object, args.scene)
    params = SimParams(**json.loads(Path(args.params).read_text())) if args.params else SimParams()
    rep = Replayer(ep, params, target_source=args.target_source, substeps=args.substeps)
    res = rep.run()
    print(json.dumps(res.metrics, indent=2))
    if args.save:
        res.save(Path(args.save))


if __name__ == "__main__":
    main()
