# Flying Knots — Implementation Plan

Executed in `~/repos/newton-worktrees/flying-knots` (branch `flying-knots` off origin/main 31b06fe0).

## Phase 0 — Environment
- [ ] `uv sync --extra examples --extra dev` in worktree; verify GPU warp init.
- [ ] Verify a cable example runs headless (`--viewer null`).
- [ ] Verify ViewerGL frame capture works under xvfb (screenshot from an existing example).

## Phase 1 — Motion digitization
- [ ] Script `scripts/flying_knot/digitize_fttraj.py` (worktree-local): color-mask the solid x/y/z
      curves in fttraj.png, calibrate pixel→(t, pos) from axis ticks (t∈[0,0.7], pos∈[-0.2,0.8+]),
      export dense trajectory npy + overlay validation figure.
- [ ] Fit 8-knot Bézier (paper Appendix D) per axis; report max fit error; save control points.

## Phase 2 — xArm7 + IK
- [ ] Vendor xarm7 URDF + visual meshes into example assets (local path; note licensing BSD-3 xArm).
- [ ] Load in Newton, identify EE body index; sanity-render.
- [ ] Batch IK over trajectory samples with warm starts; position objective + joint limits
      (+ light orientation). Output joint trajectory; plot tracking error.
- [ ] Choose base placement + trajectory offset so the workspace fits (paper: base at their world
      frame; fttraj is hand position in robot/world frame; verify reachability, xArm7 reach ≈ 0.7 m).

## Phase 3 — Example + rope
- [ ] `newton/examples/cable/example_cable_flying_knot.py` (Example class, CLI-compatible).
- [ ] Rope via add_rod attached to EE handle; weighted tip; SolverVBD; self-collision.
- [ ] Phases: settle → throw → flight → verify (lift). test_final knot check.
- [ ] Knot metric: minimal crossing count of projected centerline + Gauss linking-style writhe,
      and taut end-to-end length test.

## Phase 4 — Tuning
- [ ] Headless driver sweeping: time scale, throw amplitude scale, rope stiffness/damping, radius,
      tip mass, contact friction/thickness, substeps/iterations. JSON results.
- [ ] If unstable/knot fails: try JC PRs (#3122 first), document effect.
- [ ] Lock in tuned defaults in the example.

## Phase 5 — Videos + report
- [ ] Frame-capture recorder using ViewerGL get_frame → mp4 (imageio-ffmpeg or ffmpeg).
- [ ] Shots: full throw realtime, 4x slow-mo, knot close-up, (optional) failure case, digitized
      trajectory overlay figure, IK tracking figure.
- [ ] `academic-website-reports/flying-knots/index.html` in house style (look at existing reports),
      link videos/figures, parameter tables, PR usage, limitations. Update top-level index if the
      site has a report listing.
- [ ] Commit worktree branch + report repo.
