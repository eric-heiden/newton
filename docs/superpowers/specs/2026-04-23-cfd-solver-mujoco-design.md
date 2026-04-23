# Contacts From Distance (CFD) for SolverMuJoCo

Design spec for integrating the **Contacts From Distance (CFD)** technique from
Paulus et al., "Differentiable Simulation of Hard Contacts with Soft Gradients
for Learning and Control" (arXiv:2506.14186), into Newton's `SolverMuJoCo`.

## Motivation

Gradient-based optimization through physics simulation stalls whenever a task
depends on *making* a contact that has not yet occurred. The contact force is
zero when bodies are separated, so the analytic gradient is also zero; an
optimizer has no incentive to move one body toward another.

The paper's **Contacts From Distance (CFD)** idea addresses this by letting
MuJoCo's penalty-based contact solver generate *virtual* contact forces for
non-colliding pairs within a user-configurable distance `width`. These forces
produce informative gradients even when the objects are apart.

## Goals and Non-Goals

**Goals**

- Provide a module that a user can turn on to obtain informative contact
  gradients between non-colliding bodies, without touching the rest of the
  codebase.
- Keep the module self-contained: enabling it requires passing a single
  configuration object to `SolverMuJoCo`.
- Match the paper's mechanism: modify MuJoCo's impedance, reference
  acceleration, and contact detection distance so that contacts are treated as
  "active" up to `r = width`.
- Preserve forward-pass realism via the paper's **straight-through trick**:
  the forward state update uses the normal (unmodified) MuJoCo step, the
  backward pass uses the CFD step. Expose this as an opt-in flag.
- Add unit tests and a diffsim example (billiard-style gradient).

**Non-Goals**

- **Adaptive integration (DiffMJX)**. That half of the paper relies on
  Diffrax/JAX infrastructure. Newton's `SolverMuJoCo` delegates integration to
  `mujoco_warp`, which already exposes multiple integrators. We will not port
  Diffrax.
- **Smooth collision detection**. The paper also smooths collision primitives
  (capsule, cylinder, box). That is a larger change to `mujoco_warp` /
  Newton's collision pipeline and is out of scope for this change.

## Public API

```python
from newton.solvers import SolverMuJoCo, ContactsFromDistance

cfd = ContactsFromDistance(
    width=0.05,
    dmin=0.0,
    dmax=0.01,
    midpoint=0.5,
    power=2.0,
    straight_through=True,
)

solver = SolverMuJoCo(model, cfd=cfd)
```

`ContactsFromDistance` is a dataclass with the following fields:

| Field               | Default | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `width`             | `0.05`  | Max signed distance [m] at which CFD virtual contacts are generated.        |
| `dmin`              | `0.0`   | Minimum impedance at the CFD boundary.                                      |
| `dmax`              | `0.01`  | Maximum virtual impedance at `r = 0`.                                       |
| `midpoint`          | `0.5`   | Normalized spline midpoint (MuJoCo `solimp[3]`).                            |
| `power`             | `2.0`   | Polynomial spline power (MuJoCo `solimp[4]`).                               |
| `timeconst`         | `0.02`  | Reference acceleration time constant [s] (MuJoCo `solref[0]`).              |
| `damping_ratio`     | `1.0`   | Reference acceleration damping ratio (MuJoCo `solref[1]`).                  |
| `straight_through`  | `True`  | If `True`, forward uses normal step, backward uses CFD step (recommended). |
| `enabled`           | `True`  | Master toggle — when `False`, the solver behaves like stock `SolverMuJoCo`. |

The config is bound to a single solver instance. Switching CFD off is done by
either recreating the solver or calling `solver.set_cfd(None)` (a helper we
provide for convenience).

## Mechanism

`SolverMuJoCo` already exposes `mjw_model.geom_margin`,
`mjw_model.geom_solref`, and `mjw_model.geom_solimp` — per-shape arrays
allocated on the device. MuJoCo's active-contact condition is
`signed_distance < margin`, and the impedance function is a spline
parameterized by `solimp`. We exploit this: **extending `geom_margin` to
`width` and softening the impedance spline** lets the existing MuJoCo
constraint solver produce small contact forces for pairs in `r ∈ (0, width]`.

Concretely, when CFD is enabled on a solver we:

1. Snapshot the original `geom_margin`, `geom_solref`, `geom_solimp` arrays.
2. Allocate a **CFD-modified copy** of those arrays:
   - `cfd_margin[i] = max(original_margin[i], cfd.width)`
   - `cfd_solimp[i] = (cfd.dmin, cfd.dmax, cfd.width * 2, cfd.midpoint, cfd.power)`
     — `width * 2` on the `solimp[2]` entry keeps the impedance soft both on the
     CFD side (positive `r`) and into penetration.
   - `cfd_solref[i] = (cfd.timeconst, cfd.damping_ratio)` — a softer reference
     acceleration so that CFD contact forces taper smoothly.
3. Keep a second `mjw_data` buffer so we can run two independent steps without
   interfering with each other's intermediate state.

A step then does:

- If `straight_through` is `False`: swap in the CFD-modified arrays on
  `mjw_model` before the step, run the step, and swap the originals back. The
  user will see a slight forward-pass bias (the "hover" effect described in
  the paper) but the implementation is simple and the gradient is usable.
- If `straight_through` is `True`: run **two** steps on the same initial
  `state_in`:
  1. Step with original parameters → `state_out_normal`.
  2. Step with CFD-modified parameters → `state_out_cfd`.
     Then combine via a straight-through kernel that writes
     `state_out[i] = state_out_normal[i]` in the forward pass, but whose
     hand-written adjoint routes the gradient to `state_out_cfd[i]` only.
     Arrays touched: `body_q`, `body_qd`, `joint_q`, `joint_qd`. Concretely
     the kernel acts component-wise on the backing float buffers so it works
     uniformly for `wp.transform` and `wp.spatial_vector` data.

The CFD-modified arrays are (re)computed whenever the solver is notified of
shape property changes (`SolverNotifyFlags.SHAPE_PROPERTIES`). This hooks into
existing `notify_model_changed` machinery so nothing extra is required from
users.

### Straight-through kernel

In Warp, we cannot use JAX's `stop_gradient`. Instead we register a custom
adjoint for a dedicated kernel.

```python
@wp.kernel
def cfd_straight_through_kernel(
    normal: wp.array(dtype=wp.float32),
    cfd: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    out[tid] = normal[tid]  # forward value is normal


# Manually registered adjoint: gradient flows only through cfd.
@wp.func_grad(cfd_straight_through_kernel)
def adj_cfd_straight_through_kernel(
    normal, cfd, out, adj_normal, adj_cfd, adj_out,
):
    tid = wp.tid()
    wp.atomic_add(adj_cfd, tid, adj_out[tid])
    # adj_normal is intentionally left at zero.
```

(If `wp.func_grad` turns out to be unsuitable for kernels in the target Warp
version, we fall back to a `wp.Tape` indirection: wrap the CFD step in its
own tape, immediately replay it into the outer tape as a
`record_func`, and arrange inputs so the outer tape only knows about CFD
dependencies. The choice is an implementation detail of the straight-through
helper; the public API stays the same.)

### `use_mujoco_contacts=False` path

When the user drives MuJoCo with Newton's own contacts (`use_mujoco_contacts
=False`), the `_convert_contacts_to_mjwarp` kernel copies per-contact
`solref`/`solimp` directly from Newton's `Contacts` struct into
`mjw_data.contact.*`. We extend CFD to also swap in CFD-modified values in
that path — concretely, the solver carries a second contact buffer conversion
that uses the CFD-modified `solref`/`solimp` view of `mjw_model`, so the two
step calls see consistent parameters.

## File Layout

- `newton/_src/solvers/mujoco/contacts_from_distance.py` — new module
  containing the `ContactsFromDistance` dataclass, CFD parameter-derivation
  helpers, and the straight-through Warp kernel.
- `newton/_src/solvers/mujoco/solver_mujoco.py` — modify `__init__` to accept
  `cfd`, allocate CFD-modified arrays plus a secondary `mjw_data`, and adjust
  `step` / `notify_model_changed`.
- `newton/_src/solvers/mujoco/__init__.py` — re-export
  `ContactsFromDistance`.
- `newton/_src/solvers/__init__.py` & `newton/solvers.py` — re-export
  `ContactsFromDistance` at the `newton.solvers` level.
- `newton/tests/test_mujoco_contacts_from_distance.py` — unit tests.
- `newton/examples/diffsim/example_diffsim_mjc_billiard.py` — minimal billiard
  gradient demo (skipped if `mujoco_warp` is not installed).
- `CHANGELOG.md` — entry under "Added".

## Test Plan

1. **Construction.** `ContactsFromDistance()` constructs with reasonable
   defaults; `SolverMuJoCo(model, cfd=cfd)` succeeds for a trivial model.
2. **No-op when disabled.** `ContactsFromDistance(enabled=False)` produces a
   solver that matches the default solver's `step` output bit-for-bit over a
   handful of steps.
3. **Margin extension.** With CFD enabled, `mjw_model.geom_margin` reflects
   `max(original, width)` after initialization.
4. **Gradient is non-zero for non-colliding bodies.** Two spheres a small
   distance apart, one with velocity zero. Apply a force `F` to sphere A for
   one step; measure `d(distance_B - target) / dF` via `wp.Tape`. Without CFD
   the gradient is zero; with CFD it is non-zero (and has the expected sign).
5. **Forward trajectory unchanged (straight-through).** With
   `straight_through=True`, running `step` N times and diffing the resulting
   `body_q` against a non-CFD solver's `body_q` yields near-zero difference.
6. **Forward trajectory is biased (straight-through off).** Same test with
   `straight_through=False` shows the expected hover bias proportional to
   `width`, documenting the trade-off.
7. **Gradient through a short rollout.** A 4-step rollout of a sphere driven
   by an initial velocity onto a plane; check that the gradient wrt initial
   velocity is finite and has the right sign, with and without CFD.

Tests use `unittest` and `get_cuda_test_devices` per `AGENTS.md`.

## Changelog Entry

> Added `newton.solvers.ContactsFromDistance` to enable Contacts From Distance
> gradients through `SolverMuJoCo`, implementing the CFD technique from
> Paulus et al. (arXiv:2506.14186).
