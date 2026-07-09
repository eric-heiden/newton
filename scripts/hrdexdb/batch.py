# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Population-parallel HRDexDB replay for parameter tuning.

Replicates one episode's scene across ``W`` worlds, applies a different
:class:`SimParams` candidate per world, and rolls all candidates out in a
single CUDA-graph-captured simulation. Used by the CMA-ES tuner.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from replay import _advance_step, _record_state_batch, _set_targets, compute_metrics
from scene import SimParams, build_scene

import newton
from newton import ModelFlags


class BatchReplayer:
    """Simulates one episode for a population of parameter candidates."""

    def __init__(self, ep, num_worlds: int, target_source: str = "cmd", substeps: int = 4, max_faces: int = 2000):
        newton.use_coord_layout_targets = True
        self.ep = ep
        self.num_worlds = num_worlds
        self.base_params = SimParams()
        self.info = build_scene(ep, self.base_params, num_worlds=num_worlds, max_faces=max_faces)
        self.model = self.info.model
        self.control_dt = float(ep.t[1] - ep.t[0])
        self.substeps = substeps
        self.sim_dt = self.control_dt / substeps

        m = self.model
        self.coords_per_world = m.joint_coord_count // num_worlds
        self.dofs_per_world = m.joint_dof_count // num_worlds
        self.bodies_per_world = m.body_count // num_worlds
        self.shapes_per_world = m.shape_count // num_worlds

        self.targets_np = np.asarray(ep.q_cmd if target_source == "cmd" else ep.q_meas, dtype=np.float32)
        self.targets = wp.array2d(self.targets_np, dtype=wp.float32)
        self.dof_map_wp = wp.array(self.info.dof_map, dtype=wp.int32)
        self.step_idx = wp.zeros(1, dtype=wp.int32)

        # Baseline copies for parameter scaling / reset.
        self._mass0 = m.body_mass.numpy().copy()
        self._inertia0 = m.body_inertia.numpy().copy()
        self._init_joint_q = None
        self._init_joint_qd = None

        self.solver = None
        self.graph = None
        self._rec_shapes = None

    def apply_params(self, params_list: list[SimParams]):
        assert len(params_list) == self.num_worlds
        m = self.model
        W = self.num_worlds

        ke = m.joint_target_ke.numpy().reshape(W, self.dofs_per_world)
        kd = m.joint_target_kd.numpy().reshape(W, self.dofs_per_world)
        arm_dofs = set(int(d) for d in self.info.dof_map[:6])
        hand_dofs = set(int(d) for d in self.info.dof_map[6:])
        # Mimic followers get armature (they are real robot DOFs) but no drive.
        robot_dofs = arm_dofs | hand_dofs | {int(f) for f, _, _, _ in self.info.mimic_dofs}
        armature = m.joint_armature.numpy().reshape(W, self.dofs_per_world)
        mu = m.shape_material_mu.numpy().reshape(W, self.shapes_per_world)
        mass = m.body_mass.numpy().reshape(W, self.bodies_per_world)
        inv_mass = m.body_inv_mass.numpy().reshape(W, self.bodies_per_world)
        inertia = m.body_inertia.numpy().reshape(W, self.bodies_per_world, 3, 3)
        inv_inertia = m.body_inv_inertia.numpy().reshape(W, self.bodies_per_world, 3, 3)

        mass0 = self._mass0.reshape(W, self.bodies_per_world)
        inertia0 = self._inertia0.reshape(W, self.bodies_per_world, 3, 3)
        obj = self.info.object_body  # local body index (world 0 == template)
        base_mass = self.base_params.object_mass

        for w, p in enumerate(params_list):
            for d in range(self.dofs_per_world):
                if d in arm_dofs:
                    ke[w, d], kd[w, d] = p.arm_ke, p.arm_kd
                elif d in hand_dofs:
                    ke[w, d], kd[w, d] = p.hand_ke, p.hand_kd
                if d in robot_dofs:
                    armature[w, d] = p.joint_armature
            mu[w, :] = p.friction
            scale = p.object_mass / base_mass
            mass[w, obj] = mass0[w, obj] * scale
            inv_mass[w, obj] = 1.0 / max(mass[w, obj], 1e-9)
            inertia[w, obj] = inertia0[w, obj] * scale
            inv_inertia[w, obj] = np.linalg.inv(inertia[w, obj] + np.eye(3) * 1e-12)

        m.joint_target_ke.assign(ke.reshape(-1))
        m.joint_target_kd.assign(kd.reshape(-1))
        m.joint_armature.assign(armature.reshape(-1))
        m.shape_material_mu.assign(mu.reshape(-1))
        m.body_mass.assign(mass.reshape(-1))
        m.body_inv_mass.assign(inv_mass.reshape(-1))
        m.body_inertia.assign(inertia.reshape(W * self.bodies_per_world, 3, 3))
        m.body_inv_inertia.assign(inv_inertia.reshape(W * self.bodies_per_world, 3, 3))

        if self.solver is not None:
            self.solver.notify_model_changed(
                ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.SHAPE_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES
            )

    def _ensure_solver(self):
        if self.solver is not None:
            return
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
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
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._init_joint_q = wp.clone(self.model.joint_q)
        self._init_joint_qd = wp.clone(self.model.joint_qd)
        self.reset()

    def reset(self):
        wp.copy(self.state_0.joint_q, self._init_joint_q)
        wp.copy(self.state_0.joint_qd, self._init_joint_qd)
        wp.copy(self.state_1.joint_q, self._init_joint_q)
        wp.copy(self.state_1.joint_qd, self._init_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        self.step_idx.zero_()

    def _control_step(self, record_every: int, rec_obj, rec_q):
        wp.launch(
            _set_targets,
            dim=self.num_worlds * len(self.info.dof_map),
            inputs=[self.targets, self.dof_map_wp, self.step_idx, self.coords_per_world, len(self.info.dof_map)],
            outputs=[self.control.joint_target_q],
        )
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            _record_state_batch,
            dim=self.num_worlds,
            inputs=[
                self.state_0.body_q,
                self.state_0.joint_q,
                self.dof_map_wp,
                self.step_idx,
                record_every,
                self.info.object_body,
                self.bodies_per_world,
                self.coords_per_world,
                len(self.info.dof_map),
            ],
            outputs=[rec_obj, rec_q],
        )
        wp.launch(_advance_step, dim=1, inputs=[self.step_idx])

    def run(self, params_list: list[SimParams], record_every: int = 3) -> list[dict]:
        """Roll out all candidates; returns per-world metric dicts."""
        assert self.substeps % 2 == 0, "even substeps required for graph capture"
        self._ensure_solver()  # solver must exist so apply_params can notify it
        self.apply_params(params_list)
        self.reset()

        ep = self.ep
        n_steps = len(ep.t)
        n_rec = (n_steps + record_every - 1) // record_every
        if self._rec_shapes != (n_rec, record_every):
            self.rec_obj = wp.zeros((n_rec, self.num_worlds, 7), dtype=wp.float32)
            self.rec_q = wp.zeros((n_rec, self.num_worlds, len(self.info.dof_map)), dtype=wp.float32)
            self.graph = None
            self._rec_shapes = (n_rec, record_every)
        else:
            self.rec_obj.zero_()
            self.rec_q.zero_()

        if self.graph is None and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._control_step(record_every, self.rec_obj, self.rec_q)
            self.graph = capture.graph

        for _ in range(n_steps):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._control_step(record_every, self.rec_obj, self.rec_q)

        obj = self.rec_obj.numpy()  # (n_rec, W, 7)
        qrec = self.rec_q.numpy()
        t = ep.t[::record_every][: obj.shape[0]]
        gt = ep.obj_poses_at(t)
        from scipy.spatial.transform import Rotation as R

        pos_gt = gt[:, :3, 3]
        quat_gt = R.from_matrix(gt[:, :3, :3]).as_quat()
        ref = ep.q_meas[np.searchsorted(ep.t, t).clip(0, len(ep.t) - 1)]
        verts = np.asarray(self.info.object_mesh_newton.vertices)

        results = []
        for w in range(self.num_worlds):
            results.append(
                compute_metrics(
                    t,
                    obj[:, w, :3],
                    obj[:, w, 3:],
                    pos_gt,
                    quat_gt,
                    qrec[:, w],
                    ref,
                    verts,
                    ep.grasp_success,
                    self.info.table_height,
                )
            )
        return results
