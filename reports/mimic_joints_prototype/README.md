# Mimic Joints Prototype Report

This is the first report-ready pass for the API prototype that treats mimic relationships as zero-DOF joints. It is intentionally scoped to kinematics and importer behavior; full solver integration still needs separate design work.

## Prototype

- Added `JointType.MIMIC` as a zero-DOF follower joint.
- Added per-joint metadata for `joint_mimic_leader`, `[offset, multiplier]`, follower scalar type, and follower axis.
- Updated `eval_fk()` and batched IK FK evaluation so mimic followers are evaluated inline in the same articulation FK kernel.
- Added `ModelBuilder.add_joint_mimic()` and a URDF prototype switch, `mimic_constraints_as_joints=True`, that maps representable URDF `<mimic>` tags to mimic joints instead of the existing separate mimic-constraint arrays.

## Robotiq-Style Kinematics Example

The executable example is:

```bash
uv run --with numpy --with warp-lang --with matplotlib --with pillow \
  newton/examples/robot/example_robot_mimic_joint_kinematics.py \
  --samples 200 --repeats 5 \
  --json-out reports/mimic_joints_prototype/benchmark_results.json \
  --gif-out reports/mimic_joints_prototype/mimic_joint_kinematics.gif
```

The model is a minimal Robotiq-2F85-style two-finger kinematic tree: one driver joint and five follower joints. It isolates the mimic relationship from the real menagerie asset's tendon, contact, and closed-loop equality details.

![Mimic joint kinematics](mimic_joint_kinematics.gif)

## Initial Measurements

Generated on `cuda:0` with 20 samples and 2 repeats:

| Representation | DOFs | Mimic joints | Mimic constraints | FK timing mean | Extra propagation timing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mimic as joints | 1 | 5 | 0 | 64.35 us | none |
| Separate mimic constraints | 6 | 0 | 5 | 56.90 us with pre-propagated q | 207.51 us for host q propagation plus FK |

Interpretation: the single-kernel mimic-joint path does not yet win raw FK launch time in this tiny model, but it eliminates five independent coordinates and removes the separate "propagate mimic coordinates before FK" step needed when the current mimic-constraint representation is used for kinematics-only evaluation.

## Solver Adaptation Notes

For FK and IK target evaluation, mimic-as-joint is practical: the follower has no independent coordinate, and its transform can be derived inline from the leader coordinate in the FK kernel. That matches the way URDF and USD attach mimic metadata to joints.

For maximal-coordinate solvers such as XPBD and Kamino, mimic-as-joint is only a kinematic representation unless the solver is taught how follower constraint impulses map back to the leader coordinate. Closed-loop mechanisms, including the full Robotiq menagerie model, still need equality/connect constraints for loop closure and contact-consistent dynamics.

For generalized-coordinate solvers such as Featherstone, mimic followers reduce coordinates, but articulated inertia and forces from follower bodies must be accumulated through the leader's generalized coordinate. Treating the follower as a fixed zero-DOF joint without that mapping would move geometry correctly in FK but would not provide complete dynamics.

For MuJoCo, the existing separate mimic/equality representation maps naturally to `mjEQ_JOINT`. A mimic-joint front-end would need to lower back to MuJoCo equality constraints for dynamic stepping unless the MuJoCo bridge adds dependent-coordinate handling explicitly.

## Verdict

Use mimic-as-joint as the user-facing/importer representation for kinematics and IK, but keep the separate mimic-constraint concept as a solver lowering target for dynamics until XPBD/Kamino/Featherstone/MuJoCo all have explicit dependent-coordinate force and constraint handling.

The practical architecture is a hybrid: schemas and importers attach mimic metadata to joints, FK/IK consume that directly, and each solver either consumes the dependent joint natively or lowers it to the existing constraint/equality mechanism.

## Remaining Work

- Run larger timings on the real menagerie Robotiq assets and include CPU/GPU variance.
- Publish this report under `reports.eric-heiden.com` by porting the markdown/data/GIF into the `academic-website` `gh-pages` layout used by the determinism report.
- Add MP4/WebM video exports if the publishing environment has `ffmpeg`; this pass includes an animated GIF because the local environment only exposed the Pillow writer.
- Decide whether MJCF `equality/joint` couplings should have their own mimic-joint prototype path or remain solver equality constraints.
