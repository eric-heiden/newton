# Mimic Joints Clear-Cut Report

This branch treats representable mimic relationships as joints, not as a
parallel model-level mimic-constraint store. URDF and USD mimic metadata import
into zero-DOF `JointType.MIMIC` followers, FK derives follower transforms inline
from the leader coordinate, and the MuJoCo bridge lowers those MIMIC joints
internally when MuJoCo needs an equality row.

## Implementation

- `JointType.MIMIC` stores a zero-DOF follower joint in the normal joint tree.
- Per-joint metadata records `joint_mimic_leader`, `[offset, multiplier]`,
  follower scalar type, and follower axis.
- URDF `<mimic>` tags import as MIMIC joints by default. The old
  `mimic_constraints_as_joints` argument is retained only for call-site
  compatibility; passing `False` no longer restores separate constraint storage.
- USD `NewtonMimicAPI` and `PhysxMimicJointAPI` import as MIMIC joints, including
  pending follower conversion when the referenced leader is parsed later.
- `eval_fk()` and batched IK FK evaluation consume MIMIC joints directly.
- `SolverMuJoCo` exports a synthetic scalar MuJoCo follower joint plus an
  `mjEQ_JOINT` row for each Newton MIMIC joint. That equality row is
  solver-private lowering, not a Newton model mimic-constraint representation.
- `SolverFeatherstone` now evaluates MIMIC follower transforms and velocities in
  its FK refresh path and internal rigid setup kernels.
- `SolverXPBD` now treats zero-DOF MIMIC followers as an explicit no-force joint
  type instead of falling through the unhandled joint path.

The legacy `ModelBuilder.add_constraint_mimic()` API remains present for older
callers and existing tests, but importers and the report path no longer
recommend it.

## Benchmark Commands

Branch run, including ViewerGL MP4 captures:

```bash
PYOPENGL_PLATFORM=egl uv run --with numpy --with warp-lang --with usd-core \
  --with mujoco --with mujoco-warp --with pillow --with trimesh --with scipy \
  --with pyglet --with PyOpenGL --with imageio --with imageio-ffmpeg \
  reports/mimic_joints_prototype/benchmark_mimic_assets.py \
  --samples 20 --repeats 3 \
  --solver-samples 5 --solver-repeats 3 \
  --video-dir reports/mimic_joints_prototype/videos \
  --video-frames 48 --video-fps 24 --video-width 1280 --video-height 720 \
  --json-out reports/mimic_joints_prototype/asset_benchmark_branch.json
```

Main run used the same harness from this branch, with
`PYTHONPATH=/tmp/newton-main-benchmark`, and wrote
`asset_benchmark_main.json`.

## Branch vs Main FK

Timings are per-sample FK means on `cuda:0`.

| Asset | Branch coords / DOFs | Main coords / DOFs | Branch mimic joints / constraints | Main mimic joints / constraints | Branch FK mean | Main FK mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic Robotiq URDF | 1 / 1 | 6 / 6 | 5 / 0 | 0 / 5 | 305.0 us | 509.9 us |
| Robotiq 2F85 USD | 6 / 6 | 14 / 12 | 0 / 0 | 0 / 1 | 316.4 us | 460.5 us |
| Robotiq 2F85 MJCF | 6 / 6 | 14 / 12 | 0 / 0 | 0 / 1 | 342.1 us | 490.1 us |
| LEAP hand MJCF | 16 / 16 | 16 / 16 | 0 / 0 | 0 / 0 | 370.3 us | 324.6 us |
| Shadow hand MJCF | 24 / 24 | 24 / 24 | 0 / 0 | 0 / 0 | 332.9 us | 339.7 us |
| Unitree G1 with hands URDF | 43 / 43 | 43 / 43 | 0 / 0 | 0 / 0 | 384.4 us | 308.0 us |

The mimic-heavy Robotiq cases are the relevant comparison. The branch removes
extra follower coordinates and separate mimic-constraint storage from the
synthetic URDF case, and the real Robotiq USD/MJCF imports no longer carry the
main-branch mimic-constraint row. Assets without mimic relationships are
included as control cases; they show no coordinate-count change.

## Solver Step Timing

Timings are per-step means from the branch benchmark, `dt=1/240`, 5 steps x 3
repeats. Error cells are included intentionally.

| Asset | Semi-Implicit | XPBD | Featherstone | VBD | MuJoCo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic Robotiq URDF | 169.1 us | 513.4 us | 530.0 us | unsupported MIMIC | inertia error |
| Robotiq 2F85 USD | 159.4 us | 503.6 us | 815.3 us | 4383.3 us | 15423.0 us |
| Robotiq 2F85 MJCF | 189.6 us | 702.3 us | 829.1 us | 3947.4 us | inertia error |
| LEAP hand MJCF | 184.6 us | 609.3 us | 4365.0 us | 4437.0 us | 4325.4 us |
| Shadow hand MJCF | 176.6 us | 496.8 us | 13429.4 us | 4107.8 us | inertia error |
| Unitree G1 with hands URDF | 174.9 us | 498.6 us | 107075.3 us | 3947.8 us | inertia error |

Main, for comparison:

| Asset | Semi-Implicit | XPBD | Featherstone | VBD | MuJoCo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic Robotiq URDF | 179.9 us | 508.2 us | 523.9 us | 3683.4 us | inertia error |
| Robotiq 2F85 USD | 150.2 us | 461.0 us | 804.4 us | 4883.4 us | 29931.1 us |
| Robotiq 2F85 MJCF | 160.8 us | 502.8 us | 822.9 us | 5280.0 us | inertia error |
| LEAP hand MJCF | 170.1 us | 520.1 us | 4333.4 us | 3954.8 us | 3953.0 us |
| Shadow hand MJCF | 167.2 us | 546.3 us | 13363.5 us | 4009.5 us | inertia error |
| Unitree G1 with hands URDF | 121.9 us | 388.0 us | 106552.2 us | 3742.8 us | inertia error |

Solver support details:

- Semi-Implicit steps all branch assets in maximal coordinates.
- XPBD steps all branch assets and explicitly skips zero-DOF MIMIC followers in
  the joint-force kernel.
- Featherstone steps all branch assets and now refreshes MIMIC follower body
  transforms/velocities through its solver boundary.
- VBD still rejects a model containing real `JointType.MIMIC` rigid followers.
  That is the branch synthetic Robotiq row. The main row steps because main keeps
  those followers as ordinary scalar joints plus model-level mimic constraints.
- MuJoCo steps branch Robotiq USD after private lowering, but several assets
  still fail MuJoCo XML construction because moving bodies have mass/inertia
  below `mjMINVAL`. Those errors are recorded rather than filtered out.

## ViewerGL Captures

All captures below are MP4 videos rendered from real imported assets using
`ViewerGL.get_frame()` in headless EGL mode. The branch benchmark decoded the
first frame of each video and verified nonblank image statistics.

- [Robotiq 2F85 USD](videos/robotiq_2f85_v4_usd.mp4)
- [Robotiq 2F85 MJCF](videos/robotiq_2f85_v4_mjcf.mp4)
- [LEAP hand MJCF](videos/leap_hand_right_mjcf.mp4)
- [Shadow hand MJCF](videos/shadow_hand_right_mjcf.mp4)
- [Unitree G1 with hands URDF](videos/unitree_g1_with_hands_urdf.mp4)

No schematic captures are part of this report.

## Verdict

The clear cut is still the right direction: mimics are joints. Importers should
attach mimic metadata to joints, FK/IK should consume that directly, and solvers
should either consume `JointType.MIMIC` natively or lower it internally without
exposing a second importer-facing mimic representation. The remaining solver gap
is VBD rigid MIMIC support; it is now visible as an unsupported branch benchmark
row instead of being hidden behind a fake capture or skipped table entry.
