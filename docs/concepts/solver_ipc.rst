Experimental IPC cloth solver
============================

.. experimental::

   :class:`newton.solvers.SolverIPC` and its configuration are experimental.

``SolverIPC`` advances Style3D-authored cloth using a backward-Euler incremental
objective, normalized logarithmic contact barriers, and a bounded nonlinear
solve. It supports one fixed plane and nonincident point–triangle and edge–edge
self-contact. It does not yet implement friction, tetrahedra, dynamic rigid
bodies, multiple independent worlds, differentiation, or strain limiting.

Use :meth:`newton.solvers.SolverIPC.register_custom_attributes` before authoring
cloth with :mod:`newton.solvers.style3d`. The solver owns primitive queries;
pass ``None`` for ``contacts``. Existing reduced collision manifolds cannot
replace its swept primitive queries.

Surface contact
---------------

Self-contact is enabled by default for triangle meshes. ``self_contact_thickness``
sets minimum separation, ``self_contact_distance`` sets activation distance above
that separation, and ``self_contact_stiffness`` is a per-stencil energy scale in
joules. Only incident primitives are excluded. All triangle edges, including
boundary edges, participate; the bending-edge list need not be complete.
Initialize with nonintersecting geometry and thickness below nonincident rest
separations. The discrete barrier is mesh-dependent, not a convergent continuum
contact potential; fine meshes may need smaller thickness/activation distances.

CUDA uses swept BVHs built with the AABB machinery also used by VBD. Current
queries assemble forces and positive-semidefinite search factors. A separate
swept query covers the entire proposed search segment, including newly appearing
contacts. The line-search objective includes that complete swept set. Near-parallel
edge energies use a rest-scaled mollifier and its gradient; collision step bounds
still include these pairs even when their barrier is mollified.

Conservative advancement uses float64 distances and a speed bound for deforming
primitives. An iteration limit returns only the already-safe prefix. Additional
scale-aware spatial padding protects float32 trial rounding. This is not an
interval-arithmetic certificate. CPU uses exhaustive AABB queries as a reference;
CUDA is the intended performance path.

Storage and failure semantics
-----------------------------

``self_contact_capacity`` fixes each pair buffer's capacity; zero selects a
topology-sized estimate, not a worst-case guarantee. Overflow records
``CONTACT_OVERFLOW`` and prevents a state commit. Increase capacity outside
capture and construct a new solver if the workload exceeds it.

Only ``CONVERGED`` commits. Other statuses roll positions and velocities back
exactly; ``diagnostics.failed_steps`` retains earlier failures. Inspect it after
captured multi-step replay, not only the final status. A rolled-back step does
not advance physical time. Zero/invalid separation and initially pierced surfaces
are rejected. Prescribed particles remain fixed during a solve; externally moving
them between time steps requires separate swept boundary-motion handling.

Search operator and graphs
--------------------------

The membrane search matrix uses state-dependent positive-semidefinite stretch
blocks and Gauss–Newton shear factors. Bending retains a precomputed sparse
operator, and contact adds matrix-free normal factors. Full 3×3 block-Jacobi
inverses account for membrane and contact directions. ``use_projective_hessian``
retains the old fixed membrane operator for comparison. The Sherman–Morrison
preconditioner remains available for plane-only particle blocks.

All query and solve storage is prepared before CUDA capture. Changing capacities,
topology, configuration, or time step requires reconstruction or recapture as
appropriate. Nested CPU APIC graph replay is not supported; CPU eager execution
is supported. Warp Tile reductions reduce atomic contention in the captured PCG
path. A true-energy Armijo check remains authoritative regardless of search metric.

Example
-------

.. code-block:: console

   uv run --extra examples -m newton.examples cloth_folding --solver ipc
   uv run --extra examples -m newton.examples cloth_folding --solver vbd

Both runs release the same prefolded sheet above a fixed lower panel. Their
small-strain membrane tangents agree, but finite-strain energies, bending, contact
laws, and stopping policies differ; equal step rates do not establish equal
physical accuracy.
