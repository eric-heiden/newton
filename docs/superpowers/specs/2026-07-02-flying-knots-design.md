# Flying Knots in Newton — Design

**Date:** 2026-07-02
**Goal:** Replicate the "flying knot" (overhand knot tied mid-air by throwing a rope,
https://flying-knots.github.io/, Suresh & Atkeson, arXiv:2602.21302) as a Newton example
using `SolverVBD` with cable joints, driven by the paper's recorded throw motion, and
publish a report with ViewerGL-rendered videos under `academic-website-reports/flying-knots`.

## Source material

- Code: https://github.com/krish-suresh/flying_knots_public (cloned to `~/repos/flying_knots_public`).
  Ships **no recorded mocap/command data** (expects a local `$FLYING_KNOT_DATA`).
- Quantitative motion source: `fttraj.png` (paper Fig. "follow-through trajectory") — the executed
  end-effector x/y/z position trajectory of the 0.7 s overhand throw. Digitized by color extraction.
- Rope/sim parameters from the paper + configs: rope length 1.1 m, 11 mocap markers / particles,
  0.1 m segments, weighted tip ("each rope has a mass affixed to aid the formation of the knot"),
  8-knot Bézier command parametrization (Appendix D), xArm7 joint limits (Table V).
- Robot: xArm7 URDF + meshes vendored from `flying_knots_public/models/xarm_description`.

## Architecture

1. **Trajectory digitization** (`newton/examples/cable/assets or inline data`): extract the three
   curves from `fttraj.png` by plot-color masking, calibrate axes from known ranges, fit the paper's
   8-control-point Bézier per axis. Store control points as data in the example. Validation: overlay
   plot of digitized vs. Bézier fit committed to the report.
2. **IK stage** (offline, at example startup or precomputed): Newton `newton.ik` solver tracking the
   Bézier EE positions (position objective dominant, joint-limit objective; orientation lightly
   constrained or heuristic tangent-following) → smooth joint trajectory `q(t)` at sim rate.
   Fallback if arm can't track the whip: relax orientation entirely; last resort: floating handle
   (paper's own particle model is position-only driven).
3. **Example** `newton/examples/cable/example_cable_flying_knot.py`:
   - xArm7 loaded with `add_urdf`, driven **kinematically** (prescribed `joint_q` + `eval_fk` per step).
   - Rope: `ModelBuilder.add_rod` — 1.1 m, ~40+ capsules, ~4–6 mm radius, denser tip segment(s),
     root attached to the EE/handle frame (cable joint root on EE body).
   - `SolverVBD`, rope self-collision enabled, ground plane.
   - Phases: settle → throw (0.7 s, optionally time-scaled) → free flight/follow-through →
     verification (slow lift/pull of handle).
   - `test_final()`: knot metric — crossing analysis of the centerline projection and/or
     taut end-to-end distance < rope length after pulling.
4. **Tuning driver** (throwaway script, kept under report dir): headless sweeps over throw scale,
   timing, cable stiffness/damping, contact thickness/friction; JSON metrics per run.
5. **Report** `~/repos/academic-website-reports/flying-knots/index.html`: ViewerGL-rendered videos
   (real-time + slow-mo throw, knot close-up), digitized-vs-original trajectory figure, parameter
   tables, which in-flight Newton PRs (jumyungc: #3122, #3180, #3316, #3200) were needed, honest
   discussion of failures/limitations.

## Newton PRs in flight (JC Chang / jumyungc)

Start from `origin/main` (31b06fe0). Merge into the feature branch **only if demonstrably needed**:
- #3122 Split VBD cable stretch/shear and bend/twist constraints (stability at whip speeds)
- #3180 Cable utilities for separate rest and initial poses (initial dangling shape)
- #3316 Masked rigid resets to SolverVBD (kinematic driving)
- #3200 [Draft] Cable joint minimal eval fk/ik
Document usage + rationale in the report.

## Success criteria

- Example runs headless and with ViewerGL; a topological overhand knot forms in the hanging rope at
  the end of motion (paper's success definition) for the tuned configuration.
- Report page renders locally with embedded videos and explains method, results, deviations.

## Non-goals

- No ILC/learning loop replication (single tuned command only).
- No hardware, no Drake/Elastica model ports.
- Not necessarily an upstreamable PR — exploration branch; polish to Newton example conventions
  where cheap (Example class, test_final).
