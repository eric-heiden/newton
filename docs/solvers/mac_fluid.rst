.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

MAC Fluid
=========

.. experimental::

:class:`~newton.solvers.SolverMACFluid` simulates a 3D incompressible
Newtonian fluid on a dense uniform staggered marker-and-cell (MAC) grid. It is
designed primarily as a *fluid* half of a two-way coupled simulation: an
articulated rigid-body solver such as :class:`~newton.solvers.SolverMuJoCo`
owns the bodies, joints, and actuators, and the fluid solver treats those
bodies as moving immersed boundaries, returning hydrodynamic wrenches through
the experimental :doc:`coupled-solver framework </api/newton_solvers>`.

Grid layout
-----------

The fluid occupies a closed axis-aligned box of ``nx * ny * nz`` cells with
uniform cell size ``dx``:

- pressure (and cell labels) live at cell centers,
- the x/y/z velocity components live on the corresponding cell faces
  (``(nx+1, ny, nz)`` x-faces, and so on).

The staggering makes the discrete divergence and pressure-gradient operators
adjoint to each other, which is what allows the pressure projection to remove
divergence without checkerboard pressure modes. The grid boundary acts as a
static no-slip tank wall; free surfaces are not modeled.

Simulation step
---------------

Each :meth:`~newton.solvers.SolverBase.step` performs, in order:

1. **Boundary rasterization** — the union signed-distance field of the model's
   collider shapes (sphere, box, capsule, cylinder, cone, ellipsoid, and
   triangle meshes) is sampled at cell centers from the current ``body_q``;
   faces adjacent to solid cells are constrained to the rigid-body velocity
   sampled from ``body_qd``.
2. **Advection** — semi-Lagrangian RK2 backtrace with trilinear MAC
   interpolation (unconditionally stable). The optional ``maccormack``
   scheme adds a clamped second-order error-correction pass that strongly
   reduces numerical dissipation, so wakes and vortices persist much
   longer (kinetic-energy retention roughly doubles over one second of
   inviscid evolution).
3. **Forces** — gravity from the model plus an optional uniform external
   acceleration.
4. **Viscosity** — explicit diffusion of each staggered component. The step
   raises if ``kinematic_viscosity * dt / cell_size**2 > 1/6``.
5. **Pressure projection** — a matrix-free 7-point Poisson solve with
   homogeneous Neumann conditions at solid faces, using Jacobi-preconditioned
   conjugate gradient with a *fixed* iteration count and device-resident
   scalars (no host synchronization; CUDA-graph capturable). The all-Neumann
   closed domain is singular, so the right-hand side is made compatible by
   removing its mean over fluid cells, and converged iterations freeze to
   avoid float32 breakdown.
6. **Diagnostics** — divergence norms before/after projection, pressure
   residual, no-slip error, per-body wrenches, and momentum-balance error are
   accumulated on-device and can be read with
   :meth:`~newton.solvers.SolverMACFluid.read_diagnostics`.

Fluid–rigid coupling
--------------------

Hydrodynamic wrenches are collected *natively* from the immersed-boundary
momentum exchange rather than by differencing rigid-body momentum:

- the **pressure surface impulse** ``rho * q * A * n`` is accumulated for
  every fluid/solid interface face (this carries buoyancy, form drag, and
  added-mass reactions), and
- the **viscous exchange impulse** is accumulated wherever the diffusion
  stencil couples a fluid face to a constrained face (skin friction).

Both accumulations apply the exact opposite momentum to the fluid interior, so
fluid–rigid action–reaction holds to floating-point roundoff in the discrete
system; the residual is exposed as the ``momentum_balance_error`` diagnostic.

Used as a destination entry of
``newton.solvers.experimental.coupled.SolverCoupledProxy``, the solver:

- receives proxy body poses and velocities from the rigid source each
  coupling pass,
- converts its accumulated per-body impulses to wrenches in
  ``coupling_harvest_proxy_wrenches``, and
- restores its beginning-of-step velocity grid when the coupler restarts a
  coupling iteration (``iteration_restart``), so repeated passes never advance
  the fluid by extra physical time.

The ``staggered`` proxy mode is recommended: the generic free-body velocity
rewind of ``lagged`` mode is inconsistent for joint-constrained bodies. Weak
(staggered) coupling is only stable when the rigid body's inertia exceeds the
hydrodynamic added inertia of its immersed surface; light thin plates require
denser bodies, feedback relaxation, or a future strongly-coupled scheme.

Example
-------

.. code-block:: python

    import newton
    from newton.solvers import SolverMACFluid

    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.15, cfg=newton.ModelBuilder.ShapeConfig(density=1500.0))
    model = builder.finalize()

    solver = SolverMACFluid(
        model,
        SolverMACFluid.Config(
            resolution=(48, 48, 48),
            cell_size=1.0 / 48.0,
            origin=(-0.5, -0.5, 0.0),
            density=1000.0,
            kinematic_viscosity=1.0e-4,
            pressure_iterations=120,
        ),
    )

    state_0, state_1 = model.state(), model.state()
    for _ in range(100):
        solver.step(state_0, state_1, None, None, dt=1.0 / 60.0)

    print(solver.read_diagnostics()["body_wrench"])  # buoyancy + drag per body

For two-way coupled setups see the ``macfluid_settling_sphere``,
``macfluid_paddle``, and ``macfluid_swimmer`` examples.

Limitations
-----------

- Dense uniform grids only (no sparse or adaptive grids).
- Closed domains only: no free surfaces, multiphase flow, or inflow/outflow.
- Binary voxelized boundaries: no-slip is resolved to ``O(dx)``; forces on
  bodies converge first-order with grid resolution.
- Explicit viscosity with the usual diffusion stability limit; effective
  resolution of fine wake structure is limited by grid spacing (use the
  ``maccormack`` advection option to minimize numerical dissipation).
- Weakly coupled to rigid solvers; bodies with hydrodynamic added mass larger
  than their inertia can destabilize the coupling.
- Not differentiable; no turbulence modeling.
