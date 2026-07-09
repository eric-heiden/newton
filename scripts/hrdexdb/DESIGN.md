# HRDexDB Dynamic Replay in Newton (SolverMuJoCo) — Design

Approved design (2026-07-09).

## Goal

Quantify how faithfully Newton + `SolverMuJoCo` reproduces real-world dexterous
picking by replaying HRDexDB robot trajectories purely dynamically and
comparing the simulated passive object motion against ground-truth 6D object
tracking. Publish results as a report with overlay videos.

## Ground rules (no cheating)

- Object is a passive free rigid body. Never kinematically driven, never
  attached to the hand, no helper constraints.
- Robot driven ONLY via `Control.joint_target_q` / `joint_target_qd` PD targets
  from the recorded trajectory.
- Initial object pose from first ground-truth frame only.
- CMA-ES tunes one parameter set per hand (Allegro V5, Inspire F1), trained on
  a small set of training episodes, then FROZEN and evaluated on all episodes
  including held-out ones. No per-episode tuning.

## Data (HRDexDB, HuggingFace `HRDexDB/HRDexDB`)

- Hands: `allegro_v5` (91 objects), `inspire_f1` (92 objects), ~5 scenes each.
- Episode layout: `raw/arm/{action,position,velocity,torque,time}.npy`,
  `raw/hand/{action,position,(tactile),time}.npy`,
  `raw/timestamps/{timestamp,frame_id}.npy`, `object_6d_pose.npz`, `C2R.npy`,
  `grasp_result.json`, `cam_param/*.json`. Videos (`vid/`, ~500 MB/ep) skipped.
- Object poses are in camera/world frame; robot frame = `inv(C2R) @ pose`.
- Hand signal conversion (from snuvclab/HRDexDB `hrdexdb/common.py`):
  Allegro: 16-DOF direct; Inspire F1: 6-DOF via `(c - action) * pi / 1800`.
- URDFs from github.com/snuvclab/HRDexDB `assets/robots/`:
  `allegro_v5/xarm_allegro_v5.urdf`, `xarm_inspire_f1_right.urdf` (xArm + hand,
  fixed base).

## Embodiment

Full fixed-base xArm7 + hand articulation; PD targets on all joints
(arm + hand). Second visual-only object instance (collision disabled,
transparent/wireframe via PR #3053 opacity) shows ground truth.

## Components (`scripts/hrdexdb/` in branch `eric/hrdexdb-replay`)

1. `dataset.py` — download/list/load episodes; time-align hand+arm signals on
   the arm clock; convert hand actions to qpos; object pose sequence in robot
   frame; support-plane height inference from initial resting pose.
2. `scene.py` — ModelBuilder scene: robot from URDF, passive object mesh
   (convex decomposition if needed), ground plane, GT overlay instance.
3. `replay.py` — rollout with SolverMuJoCo; per-step `joint_target_q(d)`;
   metrics: object translational RMSE, geodesic rotation error, ADD,
   final-lift success, robot joint tracking error.
4. `tune.py` — CMA-ES (`cma` package) per hand over: PD gain scales (arm/hand),
   contact friction, object mass scale, joint friction/armature scale,
   solver iterations. Objective: mean object tracking error over training
   episodes.
5. `render.py` — headless ViewerGL video: sim robot + object (solid) + GT
   object (transparent), `viewer.get_frame()` -> imageio_ffmpeg mp4.
6. `make_report.py` + report assets — generates
   `academic-website-reports/HRDexDB/` (gh-pages of eric-heiden/academic-website):
   methodology, overlay videos, CMA-ES convergence, error curves, per-object
   stats, action-vs-position pilot, limitations.

## Experiments

1. Pilot (2 objects x 2 hands): `action.npy` vs `position.npy` as PD target
   source; pick better as headline.
2. CMA-ES per hand on ~5 training episodes.
3. Broad evaluation: 25+ objects per hand, 1-2 scenes each (~60-100 episodes),
   default vs tuned params, train vs holdout split reported.

## Deliverables

- Branch `eric/hrdexdb-replay` pushed to `eric-heiden/newton`.
- `HRDexDB/` report on `gh-pages` of `eric-heiden/academic-website`.
