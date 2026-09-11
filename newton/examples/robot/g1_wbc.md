# G1 whole-body tracking

This experimental example loads Kimodo's native G1 CSV format: 30 Hz rows of
world position, scalar-first root quaternion, and 29 hinge angles. It uses a
free base and the original G1 torque limits. There is no training, phase
adjustment, external base force, or supplied contact schedule.

```bash
uv sync --extra wbc
uv run --extra wbc -m newton.examples robot_g1_wbc --motion walk.csv --num-frames 250
```

The default `mpc-gn` controller uses damped Gauss-Newton single shooting. Four
knots for each joint give 116 parameters. Each iteration evaluates 233 central
finite-difference trajectories and 8 line-search trajectories. MuJoCo Warp
rollouts, costs, Jacobian assembly, tiled matrix operations, Cholesky solve,
and candidate selection are all captured in one CUDA graph. There is no CPU
optimizer or CPU rollout fallback. CUDA is required. Initialization, reference
preprocessing, recording and visualization are outside that graph.

`--controller mpc` selects annealed predictive sampling. `--controller qp`
selects the inexpensive native MuJoCo CPU inverse-dynamics baseline. The QP
works for standing gestures; the tested dynamic motions require preview.

Additional comparisons share the head, wrist, sole, PD, and prediction defaults
of `mpc-gn`:

- `--controller mpc-dial`: DIAL-MPC horizon and iteration annealing, normalized
  softmax averaging, and a fixed first command knot. Defaults are 1024 samples,
  two refinement rounds (ten at initialization), noise 0.12 rad, temperature
  0.06, horizon decay 0.9 and round decay 0.5. This ports the published update
  onto the shared four **linear PD-offset knots**; it does not reproduce the
  upstream quadratic spline or normalized action space. The returned mean
  receives its own physical rollout for measured cost and prediction paths.
- `--controller mpc-adjoint`: experimental sketched Gauss-Newton using the
  MuJoCo Warp PR #1535 reverse derivatives. With `--adjoint-sketch 16`, 17
  identical forward lanes carry 16 residual projections and one unprojected
  pose/velocity gradient. Eight additional trajectories evaluate step lengths.
  The Hessian approximation is `(S J).T @ (S J)` for a Rademacher sketch `S`,
  while the right-hand side uses the full pose/velocity gradient. This reduces
  trajectory count, but backward dynamics can cost more than finite differences.
- `--controller mpc-hybrid`: the same analytic update at four parallel starts;
  the first is the shifted previous plan and the others add Gaussian offsets.
  `--adjoint-starts`, `--adjoint-sketch`, and `--adjoint-noise` set this budget.
  Every start receives a local solve and physical line search before selection.

DIAL works with the normal dependency. Analytic modes require the experimental
branch, pinned here without replacing the normal installed package:

```bash
git clone https://github.com/google-deepmind/mujoco_warp.git ../mujoco-warp-adjoint
git -C ../mujoco-warp-adjoint fetch origin pull/1535/head
git -C ../mujoco-warp-adjoint checkout 357a75d60a56d67d476942a1b6e54b3045ee8e87
PYTHONPATH=../mujoco-warp-adjoint uv run --extra wbc -m newton.examples robot_g1_wbc \
  --controller mpc-adjoint --motion walk.csv --show-rollouts
```

The adjoint adapter differentiates free-base/hinge state dynamics and body pose
costs. PR #1535 freezes collision witnesses locally and uses implicit contact
solver derivatives; it does not differentiate contact-mode changes. The
adapter also masks the PR’s control VJP at saturated actuator forces; a
physical saturation regression verifies the zero response. The
non-foot contact-force penalty participates in candidate acceptance but is
excluded from the derivative. Random projection rows approximate curvature;
these modes are not full-Jacobian analytic Gauss-Newton. The implementation
stores forward states and reuses their caches in reverse. Both passes, the
linear solve, and the line searches execute in one CUDA graph. `--seed` affects
DIAL, the analytic sketch, and hybrid starts. The adapter is experimental and
currently assumes a G1-style free root followed by scalar hinge joints.

The following defaults are chosen by controller; explicit flags override them.
The resolved configuration is included in every output JSON.

| Option | GN / DIAL / analytic | Legacy sampling / CPU baselines |
| --- | ---: | ---: |
| `--mpc-rounds` | 1 | 2 |
| `--prediction-dt` | 0.005 | 0.01 |
| `--actuation` | pd | torque |
| `--gn-coordinate-search` | enabled | disabled |
| `--gain-scale` | 4 | 1 |
| `--joint-scale` | 0.15 | 0.3 |
| `--root-scale` | 0.04 | 0.08 |
| `--rotation-scale` | 0.1 | 0.15 |
| `--foot-weight` | 300 | 0 |
| `--foot-vertical` | 6 | 1 |
| `--foot-rotation` | 10 | 0 |
| `--angular-weight` | 0.2 | 0 |
| `--hand-position` | 300 | 0 |
| `--hand-rotation` | 3 | 0 |
| `--head-position` | 100 | 0 |
| `--head-rotation` | 300 | 0 |
| `--joint-velocity` | 0.02 | 0 |
| `--arm-velocity-scale` | 5 | 1 |

The horizon is 0.5 s, replan rate 100 Hz, and plant
step 0.002 s. `--gain-scale` multiplies proportional gains and its square root
multiplies derivative gains; torque bounds remain unchanged. Rotation costs
use half of the shortest SO(3) logarithm. The default `--foot-task sole` penalizes the horizontal sole center and
minimum sole-corner height; `--foot-task ankle` uses ankle-body origins.
Reported foot metrics always use the four sole corners.

The G1 head is rigidly attached to the torso; there is no actuated neck.
`--head-position` tracks the head mesh center at torso-local coordinates
(0.00765, 0, 0.38513) m. `--head-rotation` tracks its orientation using the
half-angle SO(3) residual. These six residuals enter the same scalar cost and
Gauss-Newton Jacobian as the other tasks. Set both weights to zero for the
previous objective. Head metrics report center-position RMS, full rotation
angle RMS/p95 in degrees, and signed downward tilt error of the forward axis.
The target follows the reference even during inversion; this is not an upright
head constraint.

The `--hand-position`, `--hand-rotation` and `--joint-velocity`
weights add world-frame wrist position, wrist rotation, and per-joint velocity
tracking. They enter both the scalar rollout cost and Gauss-Newton residuals.
`--arm-velocity-scale` multiplies the velocity weight for the 14 G1 arm
joints, allowing stronger damping of arm tracking errors without applying the
same weight to takeoff joints. Hand clearance remains a separate safety cost. Zero weights recover the
previous objective. Output diagnostics include wrist position error, joint
velocity error from recorded pose differences, and the RMS component of
joint-angle error above 6 Hz. The latter uses a zero-phase fourth-order
Butterworth filter and excludes 0.1 s at each end; it is a diagnostic of rapid
corrections, not a perceptual quality score.

Gauss-Newton controls are `--gn-epsilon 0.03` (central-difference perturbation
in radians), `--gn-damping 0.1` and `--gn-trust 0.2` (maximum knot update in
radians). It accepts only evaluated candidates. Enabled `--gn-coordinate-search` also
compares the best finite-difference probe with the line-search result. This
adds a coordinate-search fallback without extra physical rollouts. Disable
it with `--no-gn-coordinate-search` for the original line-search-only solver. The tiled solve supports at
most 127 parameters. The default uses one iteration (241 trajectories per update) and a finer
5 ms prediction step. In the evaluated motions, that allocation improves
tracking relative to two iterations with 10 ms prediction steps at similar
compute cost. `--mpc-rounds 2` spends more compute; improvement is not guaranteed.
Sampling uses `--mpc-samples 1024`, `--noise 0.12` and
`--temperature 0.2`. `--seed` affects stochastic modes; finite-difference Gauss-Newton has no random
search. Floating-point contact reductions can affect repeatability in either
mode. Increasing samples, gains, weights, or iteration counts need not improve
closed-loop tracking.

The default Gauss-Newton actuation uses Newton's target interface with
force-limited native PD actuators. `--actuation torque` selects explicit
bounded torque PD. Both
modes retain 29 actuators and zero commanded base wrench. Native PD uses the
same integration formulation as the predictor, but prediction still uses a
coarser time step.

Use `--viewer null --output local/run` for measurements. This creates local
`run.json` and `run.npz` files; trajectory archives are not needed on the report
website. The viewer displays visual meshes, materials, the textured ground,
sky and shadows. `--fixed-camera` disables camera following.

Add `--show-rollouts` to inspect the actual candidate futures in ViewerGL:

```bash
uv run --extra wbc -m newton.examples robot_g1_wbc \
  --motion walk.csv --num-frames 250 --show-rollouts
```

Solid lines show the selected plan; muted dashed lines show three alternative
predictions. Blue/green mark left/right ankle origins, cyan/orange mark
left/right wrist origins, and yellow marks the head center. These are body
origins, not sole-clearance measurements. `--rollout-count 1` shows only the
selected plan; `--rollout-count 4` includes three alternatives, chosen for
spatial diversity from valid candidates. This display subset is not a
confidence interval and does not influence control. Gauss-Newton includes
both line-search candidates and, when enabled, coordinate probes. The solid
path follows whichever batch actually supplied the command.

`--rollout-horizon 0.4` trims the displayed future without changing the 0.5 s
optimization horizon. `--rollout-stride 4` records every fourth prediction step
(20 ms at the default 5 ms step), including both endpoints. Use
`--rollout-torso` to add a torso-origin trace; the head path remains visible. Already elapsed segments are
clipped in the live viewer. Predictions are in world coordinates and start at
the live planning state; they are not histories or the later realized motion.

Recording observes the existing final-iteration rollouts inside the same CUDA
graph. It adds no physical rollouts or CPU optimization. With the option off,
there are no trajectory buffers or recording kernels. With it on, display
readback, thinning and line rendering occur outside the optimizer. Adding
`--output local/walk` saves a local archive with `trace_time`, `trace_qpos`,
`trace_positions`, `trace_offsets`, `trace_bodies`, `trace_indices` and
`trace_costs`. Each trace is anchored at the recorded planning time and pose;
candidate zero is the selected plan. Indices address the final sampling batch,
or the concatenated derivative and line-search batches in Gauss-Newton.
Invalid searches have index -1 and NaN paths. Only the displayed subset is
saved, at the example's 50 Hz frame rate. The report's
[rendering script](https://reports.eric-heiden.com/g1-whole-body-control/tools/render_rollouts.py)
can replay this archive separately; archives stay local.

The improved ordinary-motion tracking does not imply reliable backflips.
Gauss-Newton can get trapped around saturated actuators and landing contacts.
Sampling remains useful for that case. Arbitrary kinematic references need
not conserve flight momentum. No mode is real time on the tested 24 GB MIG
partition at 100 Hz; synchronous simulation waits for each optimization.
Self-collision, state-estimation error, actuator delay, torque-speed limits,
terrain and long-duration motion have not been evaluated.

The [live report](https://reports.eric-heiden.com/g1-whole-body-control/) contains
motion sources, measurements, failed trials, reproduction configurations,
GPU graph inspection and the separate Newton inverse-dynamics audit.

Replanning at 10 or 25 Hz is available for GPU MPC experiments while the
inner PD and physics remain at 500 Hz. It is not a demonstrated real-time
configuration: lowering the replan rate caused falls in the tested motions.
Optimizer timing excludes frames that only replay physics.
