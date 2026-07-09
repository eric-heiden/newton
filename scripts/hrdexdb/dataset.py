# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""HRDexDB dataset access and episode loading.

Downloads only the lightweight, non-video episode payload (raw joint signals,
object 6D tracking, calibration) from the HuggingFace dataset
``HRDexDB/HRDexDB`` plus the scanned object meshes, and loads episodes into a
time-aligned representation ready for dynamic replay.

Signal conventions (verified empirically on the released v0 data):
- ``raw/arm/position.npy``: measured xArm6 joint angles [rad], ~130 Hz.
- ``raw/arm/action.npy``: commanded end-effector poses (4x4), NOT joint
  targets — the arm is therefore always driven from measured joint positions.
- Allegro V5 hand: ``raw/hand/{action,position}.npy`` 16-DOF [rad], ~97 Hz.
- Inspire F1 hand: ``raw/hand/right_{commands,joint_states}.npy`` raw counts,
  converted to 6-DOF qpos [rad] with the vendor mapping from
  github.com/snuvclab/HRDexDB.
- Object 6D poses: per-video-frame 4x4 in the capture (camera-world) frame,
  either ``object_6d_pose.npz`` (``frame_<i>`` keys) or
  ``object_6d/pose_*.txt``; robot frame = ``inv(C2R) @ pose``.
- All ``*time*``/timestamp arrays share one epoch clock.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ID = "HRDexDB/HRDexDB"
DATA_ROOT = Path(os.environ.get("HRDEXDB_ROOT", "~/datasets/hrdexdb")).expanduser()
ROBOT_ASSET_ROOT = Path(os.environ.get("HRDEXDB_ROBOTS", "~/repos/HRDexDB/assets/robots")).expanduser()

HANDS = ("allegro_v5", "inspire_f1")

ROBOT_URDFS = {
    "allegro_v5": ROBOT_ASSET_ROOT / "allegro_v5" / "xarm_allegro_v5.urdf",
    "inspire_f1": ROBOT_ASSET_ROOT / "xarm_inspire_f1_right.urdf",
}

ARM_DOF = 6
HAND_DOF = {"allegro_v5": 16, "inspire_f1": 6}


# ---------------------------------------------------------------------------
# Download helpers (targeted, per episode — the full tree listing of the HF
# repo takes tens of minutes due to per-frame pose txt files and videos).
# ---------------------------------------------------------------------------


def _api():
    from huggingface_hub import HfApi  # local import: heavy

    return HfApi()


def list_remote_scenes(hand: str, object_name: str) -> list[str]:
    """Scene ids available for a hand/object pair."""
    entries = _api().list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=f"{hand}/{object_name}")
    return sorted(e.path.rsplit("/", 1)[-1] for e in entries if type(e).__name__ == "RepoFolder")


def list_remote_objects(hand: str) -> list[str]:
    entries = _api().list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=hand)
    return sorted(e.path.rsplit("/", 1)[-1] for e in entries if type(e).__name__ == "RepoFolder")


def episode_dir(hand: str, object_name: str, scene: str | int) -> Path:
    return DATA_ROOT / hand / object_name / str(scene)


def mesh_path(object_name: str) -> Path:
    return DATA_ROOT / "assets" / "mesh" / object_name / f"{object_name}.obj"


def download_episode(hand: str, object_name: str, scene: str | int, workers: int = 8) -> bool:
    """Download all non-video files of one episode. Returns False if the
    episode is incomplete on the remote (missing arm/hand raw data)."""
    from huggingface_hub import hf_hub_download

    prefix = f"{hand}/{object_name}/{scene}"
    marker = episode_dir(hand, object_name, scene) / ".complete"
    if marker.exists():
        return marker.read_text().strip() == "ok"
    try:
        entries = _api().list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=prefix, recursive=True)
        files = [e.path for e in entries if type(e).__name__ == "RepoFile" and "/vid/" not in e.path]
    except Exception:
        return False

    names = {f.rsplit("/", 1)[-1] for f in files}
    has_arm = any("/raw/arm/" in f for f in files)
    has_hand = any("/raw/hand/" in f for f in files)
    has_obj = "object_6d_pose.npz" in names or any("/object_6d/" in f for f in files)
    complete = has_arm and has_hand and has_obj and "C2R.npy" in names

    if complete:

        def fetch(path):
            hf_hub_download(REPO_ID, path, repo_type="dataset", local_dir=DATA_ROOT)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, files))

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok" if complete else "incomplete")
    return complete


def download_mesh(object_name: str) -> Path:
    from huggingface_hub import hf_hub_download

    target = mesh_path(object_name)
    if not target.exists():
        hf_hub_download(
            REPO_ID, f"assets/mesh/{object_name}/{object_name}.obj", repo_type="dataset", local_dir=DATA_ROOT
        )
    return target


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    hand: str
    object_name: str
    scene: str
    t: np.ndarray
    """Uniform control clock [s], starts at 0."""
    q_meas: np.ndarray
    """Measured joint positions (T, arm+hand) [rad] on the control clock."""
    q_cmd: np.ndarray | None
    """Commanded joint positions (T, arm+hand) [rad]; arm part is measured
    (arm commands are EE poses), hand part from recorded commands."""
    obj_t: np.ndarray
    """Object tracking times [s] on the same clock."""
    obj_poses: np.ndarray
    """Object poses (F, 4, 4) in the robot base frame."""
    grasp_success: bool
    mesh: Path
    urdf: Path
    extras: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def obj_poses_at(self, times: np.ndarray) -> np.ndarray:
        """Interpolate ground-truth object poses at arbitrary times."""
        return _interp_poses(self.obj_t, self.obj_poses, np.asarray(times))


def _load(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=True), dtype=float)


def _interp(t_src, x_src, t_dst):
    x_src = np.asarray(x_src, dtype=float)
    flat = x_src.reshape(len(x_src), -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_dst),) + x_src.shape[1:])


def _interp_poses(t_src: np.ndarray, poses: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    t_dst = np.clip(t_dst, t_src[0], t_src[-1])
    trans = np.stack([np.interp(t_dst, t_src, poses[:, i, 3]) for i in range(3)], axis=1)
    # Orthonormalize before Slerp: released poses are float32 and not exactly SO(3).
    u, _, vt = np.linalg.svd(poses[:, :3, :3])
    rots = u @ vt
    det = np.linalg.det(rots)
    u[det < 0, :, -1] *= -1.0
    rots = u @ vt
    slerp = Slerp(t_src, R.from_matrix(rots))
    out = np.tile(np.eye(4), (len(t_dst), 1, 1))
    out[:, :3, :3] = slerp(t_dst).as_matrix()
    out[:, :3, 3] = trans
    return out


def inspire_f1_counts_to_qpos(raw: np.ndarray) -> np.ndarray:
    """Vendor mapping from Inspire F1 raw counts to 6-DOF qpos [rad]
    (thumb rotation, thumb bend, index, middle, ring, little)."""
    raw = np.asarray(raw, dtype=float)
    qpos = np.zeros((raw.shape[0], 6))
    qpos[:, 0] = (1800.0 - raw[:, 0]) * np.pi / 1800.0
    qpos[:, 1] = (1350.0 - raw[:, 1]) * np.pi / 1800.0
    for i in range(2, 6):
        qpos[:, i] = (1740.0 - raw[:, i]) * np.pi / 1800.0
    return qpos


def _load_object_poses(ep_dir: Path) -> np.ndarray:
    npz_path = ep_dir / "object_6d_pose.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        return np.stack([npz[f"frame_{i}"] for i in range(len(npz.files))]).astype(float)
    pose_dir = ep_dir / "object_6d"
    paths = sorted(pose_dir.glob("pose_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No object poses in {ep_dir}")
    return np.stack([np.loadtxt(p).reshape(4, 4) for p in paths])


def load_episode(hand: str, object_name: str, scene: str | int, control_hz: float = 100.0) -> Episode:
    ep_dir = episode_dir(hand, object_name, scene)
    arm_t = _load(ep_dir / "raw" / "arm" / "time.npy").reshape(-1)
    arm_q = _load(ep_dir / "raw" / "arm" / "position.npy")[:, :ARM_DOF]
    n = min(len(arm_t), len(arm_q))
    arm_t, arm_q = arm_t[:n], arm_q[:n]

    if hand == "allegro_v5":
        hand_t = _load(ep_dir / "raw" / "hand" / "time.npy").reshape(-1)
        hand_meas = _load(ep_dir / "raw" / "hand" / "position.npy")
        hand_cmd = _load(ep_dir / "raw" / "hand" / "action.npy")
        hand_cmd_t = hand_t
        if hand_cmd.ndim != 2 or len(hand_cmd) < 2:  # a few episodes ship empty logs
            hand_cmd = hand_meas
    elif hand == "inspire_f1":
        hand_t = _load(ep_dir / "raw" / "hand" / "right_joint_states_time.npy").reshape(-1)
        hand_meas = inspire_f1_counts_to_qpos(_load(ep_dir / "raw" / "hand" / "right_joint_states.npy"))
        cmd_raw = _load(ep_dir / "raw" / "hand" / "right_commands.npy")
        if cmd_raw.ndim == 2 and len(cmd_raw) > 1:
            hand_cmd_t = _load(ep_dir / "raw" / "hand" / "right_commands_time.npy").reshape(-1)
            hand_cmd = inspire_f1_counts_to_qpos(cmd_raw)
        else:  # a few episodes ship empty command logs — fall back to measured
            hand_cmd_t = hand_t
            hand_cmd = hand_meas
    else:
        raise ValueError(f"Unsupported hand: {hand}")

    m = min(len(hand_t), len(hand_meas))
    hand_t, hand_meas = hand_t[:m], hand_meas[:m]
    m = min(len(hand_cmd_t), len(hand_cmd))
    hand_cmd_t, hand_cmd = hand_cmd_t[:m], hand_cmd[:m]

    video_t = _load(ep_dir / "raw" / "timestamps" / "timestamp.npy").reshape(-1)
    obj_poses_world = _load_object_poses(ep_dir)
    f = min(len(video_t), len(obj_poses_world))
    video_t, obj_poses_world = video_t[:f], obj_poses_world[:f]
    c2r = _load(ep_dir / "C2R.npy")
    obj_poses = np.einsum("ij,tjk->tik", np.linalg.inv(c2r), obj_poses_world)

    # Uniform control clock over the overlap of all streams, zeroed at start.
    t0 = max(arm_t[0], hand_t[0], hand_cmd_t[0], video_t[0])
    t1 = min(arm_t[-1], hand_t[-1], hand_cmd_t[-1], video_t[-1])
    if t1 - t0 < 2.0:
        raise ValueError(f"Episode {hand}/{object_name}/{scene}: stream overlap too short ({t1 - t0:.2f}s)")
    t = np.arange(0.0, t1 - t0, 1.0 / control_hz)

    q_meas = np.concatenate([_interp(arm_t - t0, arm_q, t), _interp(hand_t - t0, hand_meas, t)], axis=1)
    q_cmd = np.concatenate([_interp(arm_t - t0, arm_q, t), _interp(hand_cmd_t - t0, hand_cmd, t)], axis=1)

    grasp = json.loads((ep_dir / "grasp_result.json").read_text())

    return Episode(
        hand=hand,
        object_name=object_name,
        scene=str(scene),
        t=t,
        q_meas=q_meas,
        q_cmd=q_cmd,
        obj_t=video_t - t0,
        obj_poses=obj_poses,
        grasp_success=bool(grasp.get("grasp_success", False)),
        mesh=mesh_path(object_name),
        urdf=ROBOT_URDFS[hand],
        extras={"grasp_result": grasp},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="allegro_v5", choices=HANDS)
    parser.add_argument("--object", default="banana")
    parser.add_argument("--scene", default="2")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download:
        ok = download_episode(args.hand, args.object, args.scene)
        print(f"download {'ok' if ok else 'INCOMPLETE'}")
        if ok:
            download_mesh(args.object)

    ep = load_episode(args.hand, args.object, args.scene)
    print(f"{ep.hand}/{ep.object_name}/{ep.scene}: {ep.duration:.2f}s, success={ep.grasp_success}")
    print(f"  q_meas {ep.q_meas.shape}  q_cmd {ep.q_cmd.shape}  obj {ep.obj_poses.shape}")
    print(
        f"  obj z [m]: start {ep.obj_poses[0, 2, 3]:.3f} min {ep.obj_poses[:, 2, 3].min():.3f} max {ep.obj_poses[:, 2, 3].max():.3f}"
    )
    print(f"  obj xy start: {np.round(ep.obj_poses[0, :2, 3], 3)}")
    print(
        f"  hand q range: meas [{ep.q_meas[:, ARM_DOF:].min():.2f}, {ep.q_meas[:, ARM_DOF:].max():.2f}]"
        f" cmd [{ep.q_cmd[:, ARM_DOF:].min():.2f}, {ep.q_cmd[:, ARM_DOF:].max():.2f}]"
    )
