# Real2Sim Grasp Identification Validation Gate

This note defines the minimum replay and gradient-validation gate for the
first Newton grasp-lift-identify benchmark described in
[ROB-123](/ROB/issues/ROB-123). The goal is to make the benchmark usable as a
regression gate, not a one-off debugging script.

## Scope

The first validation gate assumes a single graspable object, one gripper
trajectory, and one short identification loop over four scalar parameters:

- object mass [kg]
- fingertip-object friction coefficient [-]
- object-table friction coefficient [-]
- normal-compliance proxy [m/N]

Normal damping may be logged, but it should remain fixed unless the compliance
term alone cannot explain lift and release transients.

## Deterministic Replay Contract

Every benchmark run must write one JSON artifact with four sections:

1. `benchmark`
   - `name`
   - `schema_version`
   - `git_commit`
   - `device`
   - `warp_version`
   - `step_dt`
   - `substeps`
   - `seed`
2. `scenario`
   - `object_asset`
   - `object_scale`
   - `gripper_trajectory_source`
   - `observation_source`
   - `loss_weights`
3. `parameters`
   - `nominal`
   - `initial_guess`
   - `identified`
4. `replay`
   - `initial_state`
   - `steps`
   - `summary`

`initial_state` must include the object pose and twist, the gripper pose or
joint state, and any solver state that changes warm-started contact behavior.

Each `steps` entry must log:

- `step`
- `time`
- `gripper_command`
- `object_pose`
- `object_twist`
- `contact_count`
- `grasp_success`
- `slip_detected`
- `slip_margin`
- `loss_terms`

`summary` must include:

- `trajectory_rmse_m`
- `trajectory_rmse_rad`
- `slip_event_step`
- `terminal_pose_error_m`
- `terminal_pose_error_rad`
- `loss`

## Replay Pass-Fail Criteria

The replay gate should compare two forward runs on the same machine, same
commit, same device, and same seed.

A run passes deterministic replay only when all of the following hold:

- `grasp_success` matches exactly.
- `slip_event_step` differs by at most 1 simulation step.
- object position drift at every logged checkpoint stays at or below `1e-3 m`.
- object orientation drift at every logged checkpoint stays at or below
  `5e-3 rad`.
- final scalar loss differs by at most `1e-4` relative or `1e-6` absolute.

If these checks fail, the benchmark is not ready for identification work. Fix
replay first.

## Gradient-Validation Gate

The first gate should evaluate analytic gradients against centered finite
differences for the four primary parameters:

- `object_mass`
- `gripper_object_friction`
- `table_object_friction`
- `normal_compliance`

Use centered finite differences around the same replay checkpoint with these
default perturbations:

- mass: `max(1e-3, 0.01 * mass)`
- friction coefficients: `0.02`
- compliance: `max(1e-6, 0.05 * compliance)`

Use one scalar identification loss assembled from:

- object pose trajectory error during lift, hold, and perturb
- slip timing error
- terminal pose error
- grasp success classification penalty

The gate should run on two checkpoints only to stay lightweight:

- one checkpoint just after stable lift
- one checkpoint at perturbation onset

A parameter passes when:

- both analytic and finite-difference gradients are finite
- the gradient signs match whenever `abs(fd_grad) >= 1e-4`
- `abs(analytic_grad - fd_grad) / max(abs(fd_grad), 1e-4) <= 0.25`

The full benchmark passes the gradient gate when:

- at least 3 of 4 parameters pass at each checkpoint
- `object_mass` must pass at both checkpoints
- no checked gradient is `NaN` or `Inf`

If friction and compliance fail together, treat the result as a stop signal for
identification until contact gradients or logging improve.

## Repo-Fit Guidance

The validation path should remain internal-first:

- keep replay artifact writers and comparison helpers under `newton/_src/tools/`
  or `benchmarks/`
- keep the regression gate in `newton/tests/` as a `unittest` module
- write generated evidence under `benchmarks/results/grasp_identification/`
- do not ship the first gate as a public example

The preferred split is:

- scenario construction and replay logging in an internal benchmark helper
- one `unittest` that loads a checked-in lightweight fixture or replay manifest
- one optional script entrypoint for regenerating artifacts during development

This keeps the benchmark cheap enough for repeated validation while preserving a
clear path into the existing dashboard artifact flow.

## Main Risks Before Implementation

- Contact-mode flips can make finite differences noisy near slip onset even when
  the forward replay looks stable.
- Warm-start state, contact ordering, or device-specific kernels can break
  determinism before any gradient bug appears.
- Mass, friction, and compliance can partially trade off against each other,
  which can make a gradient look correct while the parameter remains weakly
  identifiable.
- Observation timestamps from external reconstruction can drift relative to the
  simulation clock and invalidate both replay and gradient checks.

Implementation should stop at the first failing risk above instead of expanding
scope into a larger benchmark redesign.
