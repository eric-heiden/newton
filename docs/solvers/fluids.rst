.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Position-based fluids
=====================

.. experimental::

   :attr:`newton.ParticleFlags.FLUID`, the fluid and diffuse-particle options
   of :class:`newton.solvers.SolverXPBD`, and the fluid surface logging methods
   of :class:`newton.viewer.ViewerBase` may change without prior notice.

The XPBD solver projects particle density constraints together with its
ordinary contact constraints. Fluid particles can exchange forces with rigid
bodies and non-fluid particles. This is a position-based fluid model; it does
not provide a calibrated incompressible Navier--Stokes solver. Viscosity is a
per-substep velocity blend, rather than a dynamic viscosity in Pa·s.

Creating a fluid
---------------

Use the public particle builder and mark fluid particles explicitly:

.. code-block:: python

   import warp as wp
   import newton
   import newton.solvers

   spacing = 0.025  # m
   builder = newton.ModelBuilder()
   builder.add_particle_grid(
       pos=wp.vec3(0.0, 0.0, 0.3),
       rot=wp.quat_identity(),
       vel=wp.vec3(0.0),
       dim_x=12, dim_y=12, dim_z=12,
       cell_x=spacing, cell_y=spacing, cell_z=spacing,
       mass=1000.0 * spacing**3,
       radius_mean=0.5 * spacing,
       jitter=0.0,
       flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
   )
   builder.add_ground_plane()
   model = builder.finalize()
   solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=spacing)
   state_0, state_1 = model.state(), model.state()
   collision_pipeline = newton.CollisionPipeline(model)
   contacts = collision_pipeline.contacts()
   for _ in range(120):
       state_0.clear_forces()
       collision_pipeline.collide(state_0, contacts)
       solver.step(state_0, state_1, None, contacts, 1.0 / 240.0)
       state_0, state_1 = state_1, state_0

By default, the solver calibrates rest density from particle mass and spacing.
Increasing particle count in a fixed volume requires reducing spacing and
mass together. Merely increasing mass changes the calibration, not resolution.
The current fluid material uses one rest density and smoothing length per
solver; separate render colors do not define separate physical materials.

Stability and performance
-------------------------

Start with the defaults of the closest example and measure both simulation
and rendering. Keep displacement per substep small compared with collider
thickness and particle spacing; velocity caps are safeguards, not continuous
collision detection. Thin moving surfaces may require more substeps or thicker
collision geometry. Per-particle mesh contacts currently use mesh queries;
texture SDFs are used by rigid-shape and full-surface soft-body contacts.
Building texture SDFs requires CUDA; primitive-only scenes can run on CPU.

``fluid_max_neighbors=0`` visits all neighbors. A finite cap bounds the cost
of compressed clusters, but truncation changes the density estimate. Choose
a cap above the rest-state neighbor count. Higher constraint iterations and
substeps have different effects, and both increase cost.

``reorder_particles()`` improves locality for supported free-fluid scenes.
It changes particle indices and permutes model and state arrays in place.
Do not retain external index-based particle identities across this operation.
Use ordinary stepping for scenes with particle topology or custom indexed
data. Fluid simulation and its spatial sorting are not differentiable.

CUDA graph replay records array addresses and scalar arguments. An example
must preserve its input/output buffer convention on every replay, including
odd substep counts, and recapture when captured scalar parameters change.
The fluid examples share this bookkeeping in their local utilities.

Surface reconstruction, anisotropy, and secondary foam/spray are visual
effects. They add work independently of the physical particle solve. The
OpenGL surface renderer is optional; other viewers fall back to particles.
The bundled Flex source notices apply to the ported GLSL shaders; their
upstream license review is separate from the XPBD solver implementation.

Examples and measurements
------------------------

The screw and passive waterwheel are one scene. Run at least 240 frames to
exercise its final water-lift and wheel-rotation checks:

.. code-block:: console

   uv run --extra examples -m newton.examples fluid_xpbd_archimedes_screw --num-frames 300 --test
   uv run --extra examples -m newton.examples fluid_xpbd_archimedes_screw --num-frames 300 --test --benchmark
   uv run --extra examples -m newton.examples fluid_xpbd_dam_break --particle-count 10k --num-frames 300 --test --benchmark
   uv run --extra examples -m newton.examples fluid_xpbd_cup_transfer --num-frames 1100 --test --benchmark

``--benchmark`` selects the null viewer and excludes rendering. The ASV
``FastExampleFluidXPBD`` workloads measure captured simulation for dam break
and the screw at 10,000 and 100,000 requested particles. Report the hardware,
actual particle count, time step, substeps, and renderer when comparing runs.
Particle-count arguments are packing targets, so the resulting count can
differ from the requested count.

At its default speed, the robot cup-transfer example needs about 530 frames
to place the cup at the opposite station and 1,060 frames for a round trip.
A short smoke run does not exercise the complete grasp, carry, and release.

Configuration ownership
-----------------------

Fluid solver parameters, rendering parameters, and the example controls are
currently runtime-only Python configuration. This branch does not introduce
a persistent USD fluid-material schema or promise round-tripping these
settings through USD. A future authorable fluid-material contract needs an
explicit schema owner and importer/exporter tests before becoming supported.
