# Real measured Panda identification

This task fits a full seven-link Panda model from real recorded joint measurements,
using a geometry-only asset and explicitly poor homogeneous initial dynamics.
Every candidate must explain measured joint torques and predict 100 ms forward
windows in Newton. Independent publisher test recordings are used only by the
final verifier. This is a measured-data task; the earlier `panda_calibration` task
uses explicitly synthetic Newton-generated observations, and HUG uses recorded
human motion and scanned object geometry.

## Data and provenance

The source is the real Panda subset of
[LIP4RobotInverseDynamics, Zenodo version 12516500](https://zenodo.org/records/12516500),
DOI `10.5281/zenodo.12516500`. Pin `PUB-5510-LIP4RID.zip` to publisher MD5
`d3e29fbd280fc4cf08d008eb50559338` (541,538,288 bytes). The data and derived reference
archives are **CC-BY-SA-4.0**. Attribution: “Created by Mitsubishi Electric Research
Laboratories (MERL), 2024”; Giacomuzzo, Carli, Romeres, Dalla Libera. Preserve the
attribution, source link, license and conversion description when redistributing
references or regressors.

The [paper, section V-B](https://www.merl.com/publications/docs/TR2024-077.pdf)
describes manufacturer-ROS joint position, velocity and torque observations,
low-pass filtered at 4 Hz, with acceleration computed by acausal differentiation
of velocity. The released real-data estimator selects `tau_interp`, the same field
used here. Its [source repository](https://github.com/merlresearch/LIP4RobotInverseDynamics)
is pinned to `e1e30333f6ec1f90de33f0d75be8b12a20f9b7e8`; its AGPL implementation is
not copied into this independent preparation or evaluation code.

The publication describes 10 training records; the released archive/configuration
contains 13 training seeds and 16 test seeds. This study chooses the three lowest
numbered training seeds **2, 3, 4**, before fitting. The held-out split is **every**
published test seed: **21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37,
38**. Training recordings use 50 sinusoids; test recordings use 100. No sequences
or time windows are selected by their fitting error.

The released timestamps are irregular, approximately 17 ms apart; their actual
values are retained. No additional filtering, inferred control command, or
manufacturer dynamics is introduced. `tau_interp` is the publisher's filtered,
interpolated generalized joint-torque observation. The archive does not establish
the raw ROS topic, motor-current conversion or a recovered commanded motor torque.
Consequently, the fitted model describes **effective generalized dynamics** under
this convention. It does not establish unique physical link parameters, complete
actuator dynamics, contact dynamics or long-horizon open-loop accuracy. The
possible unmodeled gripper or other distal hardware is represented only through
effective distal-link properties.

## Reproduce preparation

Download the pinned archive separately; it is not bundled with Newton. Convert
only the numeric measurement columns, then create the geometry-only asset:

```bash
uv run --no-project --with pandas==2.2.3 python -m tools.mcp_evaluation.real_robot_data \
  --archive /data/PUB-5510-LIP4RID.zip --split training \
  --output /data/public/training.npz
uv run --no-sync python -m tools.mcp_evaluation.real_robot_data \
  --geometry-source /data/menagerie/franka_emika_panda/panda_nohand.xml \
  --output /data/public/geometry
uv run --no-sync python -m tools.mcp_evaluation.real_robot_regressor \
  --reference /data/public/training.npz \
  --geometry /data/public/geometry/panda_geometry.xml \
  --output /data/public/training-regressor.npz
```

The geometry source is MuJoCo Menagerie revision
`8161bba264d7fa7c99ca301e91e7fb44737676ad`, `franka_emika_panda/panda_nohand.xml`.
The sanitizer copies mesh files and the asset license, retaining transforms,
joint topology and visual geometry. It removes authored inertias, actuators,
initial configurations, damping, friction and armature. Geometry has zero density
and disabled collision; runtime link dynamics come only from the submitted
configuration. The original asset and publisher `_dynamics_components.pkl`
(manufacturer M/c/g), identified models and feasibility fit files must be outside
permitted agent inputs.

After freezing the split, thresholds and sampling rules, repeat the reference and
regressor preparation with `--split heldout`, writing `private/heldout.npz` and
`private/heldout-regressor.npz`. Keep those paths and their responses outside all
agent workspaces. The harness may record their digests without disclosing data.
No test-response inspection may be used to tune the quality criteria or task.

Each reference archive contains only these numeric arrays (`allow_pickle=False`):

| Array | Shape | Meaning |
|---|---|---|
| `time` | `[N]` | Original timestamp, seconds; increasing within each episode |
| `q`, `qd`, `qdd` | `[N,7]` | Joint angle, velocity, acceleration in rad, rad/s, rad/s² |
| `tau` | `[N,7]` | Publisher `tau_interp_1..7`, N·m |
| `episode_offsets` | `[E+1]` | Integer boundaries into concatenated arrays |
| `episode_ids` | `[E]` | Publisher seed identifiers |

Reference sidecar `*.manifest.json` records exact archive-member hashes, units,
field mapping, timestamp intervals, license, output digest and preparation time.
No author-provided inertial properties or manufacturer M/c/g are converted.

## Full physical configuration

`RealRobotScenario.initial`, `initial_config(variant)` and `REAL_SPEC` expose the
same initial values for every replicate. `variant` is only a replicate identifier.
Each moving link starts at 1 kg, COM `(0,0,0)` in its body frame, and COM inertia
`0.01 * identity` kg·m². All joint loss, bias and armature terms start at zero.
These values are independent of authored asset dynamics.

| Configuration field | Shape | Meaning and bounds |
|---|---|---|
| `mass` | `[7]` | kg; 0.05–10 |
| `com` | `[7,3]` | Body-frame center of mass, m; each component −0.4–0.4 |
| `inertia` | `[7,3,3]` | Symmetric body-frame COM inertia, kg·m² |
| `viscous` | `[7]` | N·m·s/rad; 0–5 |
| `coulomb` | `[7]` | N·m; 0–5 |
| `torque_bias` | `[7]` | N·m; −2–2 |
| `armature` | `[7]` | Effective reflected joint inertia, kg·m²; 0–1 |

COM inertia eigenvalues must be at least `1e-8` and satisfy physical triangle
inequalities. The trace of the second moment about the body origin must not exceed
`mass * (0.5 m)**2`. These broad constraints admit physically consistent full
tensors; they do not supply an identified parameter set. There are 98 independent
coefficients (70 link coefficients plus 28 joint terms), with substantial
unobservable combinations in this serial arm and these motions. Validation tests
predictive accuracy, not recovery of a claimed unique ground-truth parameter set.

## Immutable Newton regressor and equal offline access

All conditions receive the same raw training reference, sanitized asset, generic
construction source and immutable training regressor. `real_robot_regressor.py`
uses public Newton inverse-dynamics APIs and kinematics to construct linear basis
columns. It does not fit parameters. Its signed unit basis probes are algebraic
vectors; forward simulation candidates must pass physical validation.

The regressor NPZ contains `A[150*E*7,98]`, `b[150*E*7]`,
`sample_indices[150*E]`, and `sample_episode_ids[150*E]`. Rows are sample-major,
then joint-major. Samples are 150 equally spaced integer row indices from 30
through `episode_length-31`, separately for each record. The coefficient vector
contains, for links 1–7 in order:

```text
[m, h_x, h_y, h_z, I_origin_xx, I_origin_yy, I_origin_zz,
 I_origin_xy, I_origin_xz, I_origin_yz]
h = m * com
I_origin = I_com + m * (dot(com,com)*identity - outer(com,com))
```

Then come seven viscous, seven Coulomb, seven torque-bias and seven armature
coefficients. Their features are respectively `qd`, `sign(qd)`, `1` and `qdd`.
The regressor sidecar hashes the reference, geometry and matrix and records the
basis ordering and preparation time. Stored arrays use float64; Newton basis
calculations use float32. `physical_coefficients(config)` provides the generic
physical-to-linear mapping.

Common immutable preprocessing is measured separately, outside all agent timers.
Every condition may write fitting helpers and use identical NumPy/SciPy/CVXPY
versions. No fitting algorithm, fit solution or preferred regularization is
provided. A restart condition may perform efficient offline optimization on the
immutable matrix, but each submitted mutable Newton physical validation runs in a
fresh process. Live and IPython candidates use the same class and scoring code.
Record every candidate and its immutable configuration, including failed attempts.

## Forward checks and final verification

A candidate runs 12 fixed windows per recording. Window starts are equally spaced
integer row indices from 30 through `episode_length-37`. Each resets Newton to
measured q/qd, then runs 50 steps at 2 ms. Measured torque is linearly interpolated
at each step midpoint; q/qd references are interpolated at the step end. A fitted
torque bias is subtracted from measured input; viscous friction, Coulomb friction
and armature are native Newton/MuJoCo properties. There is no trajectory controller.

All six thresholds apply both pooled and to **each recording**:

| Metric | Maximum |
|---|---:|
| Worst-joint torque RMSE | 0.5 N·m |
| Worst-joint normalized torque RMSE | 0.5 |
| Worst-joint forward position RMSE | 0.025 rad |
| Worst-joint forward velocity RMSE | 0.5 rad/s |
| 95th percentile absolute position error | 0.05 rad |
| Maximum simulated joint speed | 5 rad/s |

Torque normalization uses each recorded joint's standard deviation on its 150
selected observations, floored at 0.5 N·m. Complete finite samples are mandatory:
1,800 training steps across 36 windows, and 9,600 held-out steps across 192 windows.
All paths attempt the full fixed sequence even after a nonfinite observation;
later window resets cannot clear that candidate's failure flag. Native solver
exceptions are surfaced as errors and retained by the runner.
The frozen study protocol predates test conversion; thresholds are not changed
based on held-out results. Final success requires a valid complete training result
for the submitted configuration, plus independent fresh processes over both the
training and all held-out records.

The same callback API supports all runners:

```python
from pathlib import Path
from tools.mcp_evaluation.real_robot import RealRobotScenario

scenario = RealRobotScenario(reference_file=Path('/data/public/training.npz'))
metrics = scenario.rollout()
scenario.apply_config(candidate)
metrics = scenario.rollout()
scenario.save_trace(Path('candidate.npz'))
```

The default regressor is the reference sibling `<stem>-regressor.npz`. The default
geometry is `reference.parent/geometry/panda_geometry.xml`; use explicit
`geometry_file`, `regressor_file`, or `NEWTON_EVAL_REAL_GEOMETRY` when the harness
copies references into isolated workspaces. `session_step` and `session_reset`
follow the shared `make_session` application callback API. Saved traces retain
measured input, bias-adjusted applied torque, simulated and reference q/qd,
recorded timestamps, episode/window IDs and torque predictions for audit.

Frame-zero checkpoints can reset a complete candidate. Restoring a nonzero-frame
checkpoint is explicitly rejected because the public physical checkpoint does not
include the application's accumulated measurements. The scenario requires a full
reset before further diagnostics or steps, preventing misaligned partial-history
metrics.

Run bounded, asset-independent checks with:

```bash
uv run --no-sync python -m unittest tools.mcp_evaluation.test_real_robot -v
```

## Independent agent comparison

Install the same optional fitting dependencies for all conditions:

```bash
uv pip install --python .venv/bin/python scipy==1.17.1 cvxpy==1.9.3
uv run --no-sync -m tools.mcp_evaluation.run_real_agents \
  --workspace /data/fresh-real-live --condition live --variant 0 \
  --public /data/public --heldout /data/private/heldout.npz \
  --phase confirmation --seconds 1200
```

Add `--run` to launch an independent Astra xhigh context. Use `restart`, `ipython`
or the separately disclosed `ipython_fixed` control in a fresh workspace with
identical data and budgets. Upstream IPython installation and its correction are
documented in [README.md](README.md). The 60-candidate budget includes every
completed failed candidate. Agent time and application startup are recorded
separately; immutable preprocessing and final verification are outside that timer.
The prompt permits numerical helper files and the same fitting libraries, but only
the candidate workflow may instantiate/advance a simulation. Restart candidates
exit after one physical evaluation. All final parameters use `config.json`.

Source commitments cover the full Newton Python tree and evaluation modules.
`integrity-manifest.json` retains these large hashes separately from the concise
scientific task. The verifier checks source, task, manifest, geometry and numeric
input integrity before opening private responses; modified inputs disqualify the
trial. Failures, timeouts, incomplete logs and raw agent usage remain artifacts.
These are trusted evaluation agents governed by input restrictions, not adversarial
sandboxes. Do not reuse scored workspaces or hide excluded/failed contexts.
