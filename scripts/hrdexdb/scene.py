# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton scene construction for HRDexDB replay.

Builds a fixed-base xArm6 + hand articulation from the HRDexDB URDFs plus a
fully passive free rigid body for the manipulated object. The object is never
attached or driven — it only interacts through contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp
from dataset import ARM_DOF, Episode

import newton
from newton import JointTargetMode

_MESH_CACHE: dict[tuple[Path, int], newton.Mesh] = {}


@dataclass
class SimParams:
    """Tunable simulation parameters (subject of CMA-ES tuning, per hand)."""

    arm_ke: float = 3000.0
    arm_kd: float = 100.0
    hand_ke: float = 20.0
    hand_kd: float = 1.0
    friction: float = 1.0
    object_mass: float = 0.20
    joint_armature: float = 0.01
    contact_ke: float = 1.0e3
    contact_kd: float = 1.0e2

    def to_vector(self) -> np.ndarray:
        return np.array(
            [self.arm_ke, self.arm_kd, self.hand_ke, self.hand_kd, self.friction, self.object_mass, self.joint_armature]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> SimParams:
        return cls(
            arm_ke=float(v[0]),
            arm_kd=float(v[1]),
            hand_ke=float(v[2]),
            hand_kd=float(v[3]),
            friction=float(v[4]),
            object_mass=float(v[5]),
            joint_armature=float(v[6]),
        )


@dataclass
class SceneInfo:
    model: newton.Model
    dof_map: np.ndarray
    """Newton coord indices for each data column (arm + actuated hand DOFs)."""
    mimic_dofs: list[tuple[int, int, float, float]]
    """(follower_coord, leader_data_col, multiplier, offset) for mimic joints."""
    object_body: int
    object_mesh_newton: newton.Mesh
    table_height: float
    robot_body_count: int
    joint_labels: list[str] = field(default_factory=list)
    palm_body: int | None = None


def load_object_mesh(path: Path, max_faces: int = 2000) -> newton.Mesh:
    """Load and (if needed) decimate the scanned object mesh."""
    key = (path, max_faces)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    import trimesh

    tm = trimesh.load(path, force="mesh", process=False)
    if len(tm.faces) > max_faces:
        tm = tm.simplify_quadric_decimation(face_count=max_faces)
    mesh = newton.Mesh(np.asarray(tm.vertices, dtype=np.float32), np.asarray(tm.faces, dtype=np.int32).flatten())
    _MESH_CACHE[key] = mesh
    return mesh


def _actuated_data_joints(builder: newton.ModelBuilder, hand: str) -> tuple[list[int], list[str]]:
    """Movable, non-mimic joint indices in the order the dataset stores DOFs.

    The dataset follows URDF document order (yourdfpy actuated joints), which
    matches Newton's dfs ordering for these serial-chain robots. Verified via
    joint labels at build time.
    """
    arm_names = [f"joint{i}" for i in range(1, 7)]
    if hand == "allegro_v5":
        hand_names = [f"joint_{i}_0" for i in range(16)]
    else:
        hand_names = [
            "right_thumb_1_joint",
            "right_thumb_2_joint",
            "right_index_1_joint",
            "right_middle_1_joint",
            "right_ring_1_joint",
            "right_little_1_joint",
        ]
    labels = list(builder.joint_label)
    order = []
    for n in arm_names + hand_names:
        matches = [i for i, label in enumerate(labels) if label == n or label.endswith("/" + n)]
        if len(matches) != 1:
            raise ValueError(f"Joint {n}: expected exactly one match, got {matches} in {labels}")
        order.append(matches[0])
    return order, labels


INSPIRE_MIMICS = {
    # follower: (leader, multiplier, offset) from xarm_inspire_f1_right.urdf
    "right_thumb_3_joint": ("right_thumb_2_joint", None, None),
    "right_thumb_4_joint": ("right_thumb_2_joint", None, None),
    "right_index_2_joint": ("right_index_1_joint", None, None),
    "right_middle_2_joint": ("right_middle_1_joint", None, None),
    "right_ring_2_joint": ("right_ring_1_joint", None, None),
    "right_little_2_joint": ("right_little_1_joint", None, None),
}


def build_scene(
    ep: Episode,
    params: SimParams | None = None,
    num_worlds: int = 1,
    max_faces: int = 2000,
) -> SceneInfo:
    params = params or SimParams()

    world = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(world)
    world.default_shape_cfg.ke = params.contact_ke
    world.default_shape_cfg.kd = params.contact_kd
    world.default_shape_cfg.mu = params.friction

    world.add_urdf(
        str(ep.urdf),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        parse_visuals_as_colliders=False,
        hide_visuals=False,
    )
    robot_body_count = world.body_count

    dof_joints, labels = _actuated_data_joints(world, ep.hand)

    # Map data columns -> newton coord indices; joints here are all 1-DOF.
    q_start = np.array(world.joint_q_start + [world.joint_coord_count])
    dof_map = np.array([q_start[j] for j in dof_joints], dtype=np.int32)

    # Mimic followers (URDF importer adds the kinematic coupling constraints;
    # we keep their drives off so the equality constraint governs them).
    mimic_dofs = []
    urdf_mimics = _parse_urdf_mimics(ep.urdf)
    data_cols = {}
    arm_hand_names = _data_joint_names(ep.hand)
    for col, name in enumerate(arm_hand_names):
        data_cols[name] = col
    for i, label in enumerate(labels):
        short = label.split("/")[-1]
        if short in urdf_mimics:
            leader, mult, off = urdf_mimics[short]
            if leader in data_cols:
                mimic_dofs.append((int(q_start[i]), data_cols[leader], mult, off))

    driven = set(int(d) for d in dof_map)
    for dof in range(world.joint_dof_count):
        if dof in driven:
            is_arm = dof in set(int(d) for d in dof_map[:ARM_DOF])
            world.joint_target_ke[dof] = params.arm_ke if is_arm else params.hand_ke
            world.joint_target_kd[dof] = params.arm_kd if is_arm else params.hand_kd
            world.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        else:
            world.joint_target_ke[dof] = 0.0
            world.joint_target_kd[dof] = 0.0
        world.joint_armature[dof] = params.joint_armature

    # Initial robot configuration = first measured sample (also as target).
    q0 = ep.q_meas[0]
    for col, dof in enumerate(dof_map):
        world.joint_q[dof] = float(q0[col])
        world.joint_target_q[dof] = float(q0[col])
    for follower, leader_col, mult, off in mimic_dofs:
        world.joint_q[follower] = mult * float(q0[leader_col]) + off

    # Passive object as a free rigid body at the first ground-truth pose.
    obj_mesh = load_object_mesh(ep.mesh, max_faces=max_faces)
    T0 = ep.obj_poses[0]
    xform0 = _mat44_to_transform(T0)
    # add_body creates the free joint automatically — the object is passive.
    body = world.add_body(xform=xform0, label="object")

    verts = np.asarray(obj_mesh.vertices)
    vol_est = _mesh_volume(obj_mesh)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=params.object_mass / max(vol_est, 1e-6),
        ke=params.contact_ke,
        kd=params.contact_kd,
        mu=params.friction,
    )
    world.add_shape_mesh(body, mesh=obj_mesh, cfg=cfg, color=(0.83, 0.48, 0.22), label="object_mesh")

    # Support plane at the object's initial resting height.
    v0 = verts @ T0[:3, :3].T + T0[:3, 3]
    table_height = float(v0[:, 2].min())
    plane_cfg = newton.ModelBuilder.ShapeConfig(ke=params.contact_ke, kd=params.contact_kd, mu=params.friction)
    world.add_shape_plane(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, table_height), wp.quat_identity()),
        width=0.0,
        length=0.0,
        cfg=plane_cfg,
        color=(0.80, 0.82, 0.86),
        label="table",
    )

    builder = world
    if num_worlds > 1:
        builder = newton.ModelBuilder()
        builder.replicate(world, num_worlds)

    model = builder.finalize()

    # The Inspire URDF spells palm as "plam"; fall back to the wrist link.
    palm_body = next(
        (
            i
            for i, label in enumerate(model.body_label[:robot_body_count])
            if "palm" in label.lower() or "plam" in label.lower()
        ),
        None,
    )
    if palm_body is None:
        palm_body = next(i for i, label in enumerate(model.body_label[:robot_body_count]) if label.endswith("link6"))
    return SceneInfo(
        model=model,
        dof_map=dof_map,
        mimic_dofs=mimic_dofs,
        object_body=body,
        object_mesh_newton=obj_mesh,
        table_height=table_height,
        robot_body_count=robot_body_count,
        joint_labels=labels,
        palm_body=palm_body,
    )


def _data_joint_names(hand: str) -> list[str]:
    arm = [f"joint{i}" for i in range(1, 7)]
    if hand == "allegro_v5":
        return arm + [f"joint_{i}_0" for i in range(16)]
    return arm + [
        "right_thumb_1_joint",
        "right_thumb_2_joint",
        "right_index_1_joint",
        "right_middle_1_joint",
        "right_ring_1_joint",
        "right_little_1_joint",
    ]


def _parse_urdf_mimics(urdf_path: Path) -> dict[str, tuple[str, float, float]]:
    import xml.etree.ElementTree as ET

    out = {}
    root = ET.parse(urdf_path).getroot()
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            out[j.attrib["name"]] = (
                m.attrib["joint"],
                float(m.get("multiplier", 1.0)),
                float(m.get("offset", 0.0)),
            )
    return out


def _mat44_to_transform(T: np.ndarray) -> wp.transform:
    from scipy.spatial.transform import Rotation as R

    u, _, vt = np.linalg.svd(T[:3, :3])
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    q = R.from_matrix(rot).as_quat()  # xyzw
    return wp.transform(wp.vec3(*T[:3, 3].tolist()), wp.quat(*q.tolist()))


def _mesh_volume(mesh: newton.Mesh) -> float:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)
    return vol


if __name__ == "__main__":
    from dataset import load_episode

    for hand, obj, scene in [("allegro_v5", "banana", "2"), ("inspire_f1", "apple", "2")]:
        ep = load_episode(hand, obj, scene)
        info = build_scene(ep)
        m = info.model
        print(
            f"{hand}: bodies={m.body_count} joints={m.joint_count} coords={m.joint_coord_count} shapes={m.shape_count}"
        )
        print(f"  dof_map={info.dof_map.tolist()}")
        print(f"  mimic={[(f, c, mu) for f, c, mu, _ in info.mimic_dofs]}")
        print(f"  table_height={info.table_height:.4f} object_body={info.object_body}")
