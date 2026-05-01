# Co-Simulation Validation Gate For Cloth And Cable Benchmarks

This note turns the benchmark guidance from
[ROB-114](/ROB/issues/ROB-114) into a concrete Newton execution target. It
keeps the [ROB-117](/ROB/issues/ROB-117) discipline: start with the smallest
benchmark slice that can reject a weak co-simulation path quickly, then expand
only after the first gate is passing.

## Recommended First Slice

Implement the cloth real2sim benchmark first.

Reasons:

- Newton already has a representative cloth-manipulation substrate in
  `newton/examples/cloth/example_cloth_franka.py`.
- The repo already has cloth-focused ASV coverage in
  `asv/benchmarks/simulation/bench_cloth.py`, which gives a natural place for
  repeatable harness work once the validation artifact exists.
- The cloth task exercises the hardest part of [ROB-114](/ROB/issues/ROB-114)
  first: contact-rich rigid-soft interaction under robot control.
- The cable task should follow after the logging and replay contract is working,
  because it is cheaper to iterate on imagined-rollout policy evaluation than on
  observability plumbing.

## Benchmark 1: Real2Sim Cloth Manipulation

Primary source scenario: `newton.examples.cloth.example_cloth_franka`

Goal: evaluate whether an additive learned-deformable residual improves
short-horizon prediction enough to justify integration work.

### Minimal Scenario Contract

- one Franka arm
- one shirt or towel cloth asset
- one table contact surface
- one grasp, lift, drag, and partial place sequence
- one short replay window of `1.0 s` to `1.5 s` after grasp closure
- one held-out replay window from the same task family with different initial
  cloth wrinkles or grasp offset

### State, Action, And Observation

- state:
  - robot joint positions and velocities
  - cloth particle positions and velocities
  - contact state summary for robot-cloth and cloth-table interactions
- action:
  - commanded end-effector pose or joint targets from the replay clip
- observation:
  - tracked cloth keypoints or downsampled particle set
  - robot end-effector pose
  - contact counts
  - end-effector wrench estimate or proxy contact impulse

### Forward-Simulation Pass Criteria

The benchmark passes the first forward gate only when all of the following hold
on both the calibration replay and the held-out replay:

- short-horizon cloth position RMSE at tracked points stays at or below
  `0.02 m` median and `0.04 m` p95
- contact-onset timing error stays within `2` rendered frames
- contact-count distribution drift stays at or below `0.15` Jensen-Shannon
  divergence over the replay window
- end-effector wrench or net contact-impulse error stays within `15%`
  normalized absolute error
- penetration budget stays within `5 mm` p95 and `10 mm` max

### Decision Rule

Stop the learned-cloth path if either replay fails the full gate above. Do not
expand model complexity, latent size, or training set size until the minimal
gate passes.

## Benchmark 2: Imagined-Rollout Cable Endpoint Placement

Primary source substrate: Newton rod and cable builders used by
`newton/examples/cable/`

Goal: test whether imagined rollouts are accurate enough to rank low-DoF cable
placement actions before Newton executes them.

### Minimal Scenario Contract

- one cable with one fixed root and one controlled endpoint
- planar workspace with one obstacle and one target pocket or target disk
- low-DoF action parameterization:
  - endpoint translation in `x`
  - endpoint translation in `y`
  - endpoint height
  - release time
- imagined horizon of `0.75 s` to `1.0 s`
- policy objective: place the cable tip inside the target region without
  excessive obstacle contact or penetration

### State, Action, And Observation

- state:
  - endpoint pose
  - reduced cable centerline samples
  - per-segment velocities
  - obstacle contact summary
- action:
  - four scalar placement controls listed above
- observation:
  - tip position
  - centerline samples
  - contact count and first-contact time
  - endpoint reaction force or aggregate contact impulse

### Forward-Simulation Pass Criteria

The cable benchmark passes the first imagined-rollout gate only when all of the
following hold on a fixed action set and on the best action chosen by the
learned model:

- tip-placement error stays at or below `0.03 m`
- centerline rollout error stays at or below `0.025 m` mean over sampled points
- first-contact timing error stays within `3` rendered frames
- contact-count distribution drift stays at or below `0.20` Jensen-Shannon
  divergence
- endpoint reaction-force or contact-impulse error stays within `20%`
  normalized absolute error
- penetration budget stays within `3 mm` p95 and `6 mm` max

### Newton Revalidation Rule For Policy Gains

The learned model may be used to rank candidate actions, but a gain only counts
as real if Newton re-execution confirms it.

The policy-improvement gate passes only when:

- the action selected by the learned model beats the Newton baseline on target
  success rate by at least `10%` relative over `32` evaluation seeds
- at least `90%` of that measured gain remains after replaying the chosen
  actions in Newton
- no selected action violates the penetration budget above

If Newton revalidation fails, treat the learned planner as a proposal generator
only and do not claim control improvement.

## Observability Payload

Every benchmark run must emit one replayable JSON artifact with enough detail to
compare Newton against the learned model world-by-world.

Required top-level sections:

1. `benchmark`
2. `scenario`
3. `real2sim_cloth`
4. `imagined_rollout_cable`
5. `observability`

### Per-World Packet Traces

Each world packet should log:

- `world_id`
- `step`
- `time`
- `action`
- `observed_state`
- `predicted_state`
- `newton_state`
- `contact_summary`
- `loss_terms`

### Latent And Residual Debug Fields

The artifact must include:

- latent confidence or ensemble uncertainty per step
- residual magnitude per step
- contact residual norms
- rollout divergence score against Newton
- discrete divergence alarms with threshold name and first failing step

### Replayability Requirements

The artifact must also capture:

- git commit
- device
- Warp version
- seed
- timing config
- scenario asset identifiers
- initial state snapshot
- control sequence

This keeps Newton-versus-model comparisons deterministic enough to debug rather
than hand-waving over aggregate metrics.

## What Goes In Tests First

These metrics belong in repo tests or checked benchmark artifacts first:

- replay completeness and schema checks
- short-horizon rollout error
- contact-onset timing
- contact-count distribution
- impulse or wrench error
- penetration budget
- Newton revalidation of learned policy gains

These are the go or no-go criteria and should fail loudly when they regress.

## What Can Wait For Dashboards

These metrics can land in dashboards or reports after the first gate exists:

- run-to-run trend charts
- percentile breakdowns beyond the pass-fail thresholds
- latency histograms
- latent-space visualizations
- scenario-family rollups across many assets

Dashboard work is useful only after the benchmark contract above is executable.

## Repo-Fit Guidance

Preferred implementation split:

- keep benchmark specs and sample artifacts under `benchmarks/`
- keep scenario builders, replay helpers, and artifact writers under
  `newton/_src/tools/` or another internal helper surface
- keep pass-fail regression checks in `newton/tests/` as `unittest` modules
- write generated evidence under `benchmarks/results/`

Avoid shipping the first validation gate as a public example. The first version
should be an internal benchmark and regression contract.
