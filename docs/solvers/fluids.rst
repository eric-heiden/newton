.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Fluids and batched environments
===============================

.. experimental::

   :attr:`newton.ParticleFlags.FLUID`, the ``fluid_*`` options of
   :class:`newton.solvers.SolverXPBD`, and its particle reordering operation
   are experimental. The cup-transfer example's training interface is an
   example-local interface, not a supported environment framework.

Choosing a fluid solver
-----------------------

:class:`newton.solvers.SolverXPBD` supports position-based fluids. Density
constraints resist compression, cohesion keeps droplets together, and fluid
particles exchange contact impulses with dynamic rigid bodies and non-fluid
particles. Viscosity blends neighboring velocities after each substep;
optional vorticity confinement restores swirling motion. This is an
approximate particle model, not a calibrated incompressible flow solver.

:class:`newton.solvers.SolverImplicitMPM` offers a different continuum material
model. The shared dam-break example accepts ``--solver xpbd`` or ``--solver
mpm`` while using the same tank and initial particle distribution. Material
parameters and numerical resolution have different meanings between these
solvers; matching a viscosity value does not match their physical behavior.
The dynamic cup, robot, screw, tank, and wave examples currently use XPBD.

The MPM viscous funnel and extracted water-surface demonstrations live in
``newton/examples/fluid`` as ``fluid_viscous`` and ``fluid_water_surface``.
The previous ``mpm_viscous`` and ``mpm_water_dam_break`` commands remain
compatible entry points. Granular, elastic, and snow examples remain under
``mpm`` because their principal subject is not liquid simulation.

Creating XPBD fluid
-------------------

Use the existing model builder and mark fluid particles explicitly:

.. code-block:: python

   import warp as wp
   import newton
   import newton.solvers

   spacing = 0.025
   builder = newton.ModelBuilder()
   builder.add_particle_grid(
       pos=wp.vec3(0.0, 0.0, 0.3),
       rot=wp.quat_identity(), vel=wp.vec3(0.0),
       dim_x=12, dim_y=12, dim_z=12,
       cell_x=spacing, cell_y=spacing, cell_z=spacing,
       mass=1000.0 * spacing**3, radius_mean=0.5 * spacing,
       jitter=0.0,
       flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
   )
   builder.add_ground_plane()
   model = builder.finalize()
   solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=spacing)
   collision_pipeline = newton.CollisionPipeline(model)
   contacts = collision_pipeline.contacts()
   state_0, state_1 = model.state(), model.state()
   for _ in range(120):
       state_0.clear_forces()
       collision_pipeline.collide(state_0, contacts)
       solver.step(state_0, state_1, None, contacts, 1.0 / 240.0)
       state_0, state_1 = state_1, state_0

The rest density is calibrated from mass and spacing unless supplied
explicitly. Change particle mass and spacing together when changing
resolution. A solver has one rest density and smoothing length; colored
particle populations do not define immiscible phases or independent fluid
materials. Fluid stepping does not support gradient tracking.

World isolation and training
----------------------------

Replicate a world builder with zero physical spacing to simulate independent
environments at identical coordinates. Grouped spatial queries use
``Model.particle_world`` to isolate fluid neighbors. Global fluid particles
(world ``-1``) form a separate group and do not couple local worlds; global
collider shapes still interact with every world. Set viewer world offsets
to display the environments separately; display offsets do not affect the
physics. The dam-break and cup-transfer examples use this arrangement.

Training loops should retain fixed-size device arrays for actions, states,
observations, and reset masks. CUDA graphs capture array addresses and scalar
arguments: update array contents in place, preserve the state-buffer
convention, and recapture if scalar configuration changes. The fluid examples
share the bookkeeping for odd and even numbers of substeps.

The batched cup-transfer scene uses inverse kinematics and an explicit
kinematic attachment abstraction. Its grasp gate uses the achieved robot
pose. The cup stays upright while attached, and release is permitted only
within 1.5 cm of the floor; a free falling cup is outside this abstraction.
It supports prescribed playback and externally supplied actions; it is not
a force-controlled grasp benchmark or a trained policy. Robot geometry is
visual-only; water collides with the cup and ground. The observations summarize
water retention and omit the internal fluid motion, so the task is partially
observable through these buffers.

Its persistent arrays stay on ``example.model.device``:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Array
     - Shape
     - Meaning
   * - ``actions``
     - ``(world_count, 4)``
     - Normalized XYZ target velocities and gripper closing velocity.
       Full-scale speeds are 0.25 m/s per Cartesian axis and 0.08 m/s
       for the fingers.
   * - ``observations``
     - ``(world_count, 20)``
     - End-effector and cup positions; target and goal errors; cup velocity;
       aperture, attachment, retained-water fraction, completed-lift flag,
       and episode progress. The class docstring specifies column ordering.
   * - ``rewards``
     - ``(world_count,)``
     - Timestep times (retained-water fraction minus one, minus twice goal
       distance [m]), plus five on successful placement or minus five on
       spill failure. Nonterminal rewards cannot be positive, so holding the
       cup cannot accumulate a reward for delaying
       release. This is an example reward, not a calibrated training objective.
   * - ``terminated``, ``truncated``
     - ``(world_count,)``
     - Task completion or excessive spilling, and the episode step limit.
   * - ``episode_steps``, ``episode_count``
     - ``(world_count,)``
     - Independent episode clocks and reset counters.

``example.set_actions(actions)`` copies a float32 Warp array of shape
``(world_count, 4)`` on the model device and disables the reference controller.
The controller flag is device-resident, so changing control mode does not
require graph recapture. ``example.step()`` advances all worlds and updates
the observation and reward buffers. A training adapter can share these arrays
with its tensor framework using Warp's interoperability facilities.

The supported training interface is the task arrays above. The achieved
kinematic joint configuration is stored in ``example.model.joint_q``
(``example.joint_q_ik`` contains the batched arm solution before the finger
aperture override). As in the standalone IK examples, ``state_0.joint_q``
does not track these prescribed joint updates, and robot-link velocities
are not reconstructed. Use the task's achieved pose and cup-velocity
observations; these state fields are not dynamic robot observations.

``example.reset(mask)`` resets selected worlds using a boolean Warp array of
shape ``(world_count + 1,)``. The final entry selects global entities under
Newton's reset convention; this example's bodies and particles all belong to
local worlds. Passing ``None`` resets every world. Reset preserves array
addresses and the trajectories of unselected worlds; explicit reset clears
actions for the selected worlds.

Automatic reset occurs at the start of the following step, leaving terminal
observations readable after the terminating step. Automatic reset preserves
newly submitted actions for the next episode. Use ``--no-auto-reset`` when
the training adapter owns reset timing. Reset states are deterministic; the
example does not add domain randomization or a policy-training dependency.

Performance and stability
-------------------------

.. code-block:: console

   uv run --extra examples -m newton.examples fluid_dam_break --world-count 8
   uv run --extra examples -m newton.examples fluid_dam_break --solver mpm --world-count 4
   uv run --extra examples -m newton.examples fluid_cup_transfer --world-count 4
   uv run --extra examples -m newton.examples fluid_archimedes_screw
   uv run --extra examples -m newton.examples fluid_wave_pool

``--particle-count`` is a per-world packing target rather than an exact
count. Increasing worlds multiplies both total particles and collision work.
Benchmark the actual workload at its intended world count and report both
batch frames per second and environment steps per second. A batch step
advances every world; environment steps per second equals batch FPS times
world count. Neither quantity includes policy-network inference unless the
benchmark explicitly adds it.

Use ``--benchmark --num-frames 300`` to measure simulation without rendering.
The ASV ``FluidXPBDWorlds`` and ``FluidCupTransferWorlds`` workloads track
captured stepping across world counts, including IK and observation/reward
updates in the robot workload. FPS is a benchmark result, not a unit-test
assertion, because hardware and contention change timing.

Keep displacement per substep small relative to particle spacing and
collider thickness. Velocity caps do not replace continuous collision
detection. Thin moving walls may need more substeps or thicker geometry.
Neighbor caps bound work in compressed regions but truncate density sums;
the default uncapped solve avoids this approximation.

Spatial sorting is optional and changes particle indices. It is restricted
to free-fluid scenes without topology or indexed custom data. Training
observations that depend on stable particle identities should leave it off.
The multiworld examples preserve particle identity.

Warp 1.17 shares the host descriptor used by captured hash-grid builds.
Building another model can otherwise change the descriptor read by a live
graph. XPBD fluids keep a private device snapshot and restore it after each
build. This temporary compatibility helper has tests for interleaved models,
different radii, and graph replay after resets. Captured models must retain
their grid storage and particle count for the graph's lifetime.

The XPBD examples use ordinary particle rendering and do not depend on the
separate fluid-surface renderer. ``fluid_water_surface`` demonstrates the
existing mesh-extraction path for MPM. Fluid solver options and example
controls are runtime-only Python configuration; this change does not add a
persistent USD fluid-material schema or promise its round-trip serialization.
