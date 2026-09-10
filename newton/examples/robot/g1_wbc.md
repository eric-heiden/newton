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

The following defaults are chosen by controller; explicit flags override them.
The resolved configuration is included in every output JSON.

| Option | Gauss-Newton | Sampling / CPU baselines |
| --- | ---: | ---: |
| `--mpc-rounds` | 2 | 2 |
| `--gain-scale` | 4 | 1 |
| `--joint-scale` | 0.15 | 0.3 |
| `--root-scale` | 0.04 | 0.08 |
| `--rotation-scale` | 0.1 | 0.15 |
| `--foot-weight` | 300 | 0 |
| `--foot-vertical` | 4 | 1 |
| `--foot-rotation` | 10 | 0 |
| `--angular-weight` | 0.2 | 0 |

The horizon is 0.5 s, prediction step 0.01 s, replan rate 100 Hz, and plant
step 0.002 s. `--gain-scale` multiplies proportional gains and its square root
multiplies derivative gains; torque bounds remain unchanged. Rotation costs
use half of the shortest SO(3) logarithm. The default `--foot-task sole` penalizes the horizontal sole center and
minimum sole-corner height; `--foot-task ankle` uses ankle-body origins.
Reported foot metrics always use the four sole corners.

Gauss-Newton controls are `--gn-epsilon 0.03` (central-difference perturbation
in radians), `--gn-damping 0.1` and `--gn-trust 0.2` (maximum knot update in
radians). It accepts only evaluated candidates. The tiled solve supports at
most 127 parameters. Use `--mpc-rounds 1` for approximately half the optimizer latency, with
lower swing accuracy in the tested jumping motion. Sampling uses `--mpc-samples 1024`, `--noise 0.12` and
`--temperature 0.2`. `--seed` affects sampling only; Gauss-Newton has no random
search. Floating-point contact reductions can affect repeatability in either
mode. Increasing samples, gains, weights, or iteration counts need not improve
closed-loop tracking.

The default actuation is explicit bounded torque PD. `--actuation pd` instead
uses Newton's target interface with force-limited native PD actuators. Both
modes retain 29 actuators and zero commanded base wrench. Native PD uses the
same integration formulation as the predictor, but prediction still uses a
coarser time step.

Use `--viewer null --output local/run` for measurements. This creates local
`run.json` and `run.npz` files; trajectory archives are not needed on the report
website. The viewer displays visual meshes, materials, the textured ground,
sky and shadows. `--fixed-camera` disables camera following.

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
