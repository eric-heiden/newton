# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contacts From Distance (CFD) for SolverMuJoCo.

Implements the gradient-surrogate mechanism from Paulus et al.,
"Differentiable Simulation of Hard Contacts with Soft Gradients for Learning
and Control", arXiv:2506.14186.

The paper proposes augmenting MuJoCo's contact impedance ``d(r)`` and position
reference acceleration ``aref`` so that *virtual* contact forces are produced
for signed distances ``r > 0`` (i.e. between separated bodies). Those virtual
contacts give gradient signal even when objects are not touching — an optimizer
trying to "make contact" sees a useful non-zero descent direction.

This module exposes the configuration dataclass and the Warp kernels that
rewrite MuJoCo's per-geom contact parameters to enable CFD. It is meant to be
used as an *opt-in* add-on to :class:`~newton.solvers.SolverMuJoCo`; by
default the solver still runs stock MuJoCo physics.

See ``docs/superpowers/specs/2026-04-23-cfd-solver-mujoco-design.md`` for the
full design discussion.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ...core.types import vec5


@dataclass
class ContactsFromDistance:
    """Configuration for the Contacts From Distance (CFD) gradient surrogate.

    CFD is an opt-in extension of MuJoCo's penalty-based contact model that
    lets the constraint solver produce small *virtual* contact forces for
    signed distances ``r`` in the range ``(0, width]``. Those virtual forces
    exist only to create informative gradients between non-colliding objects;
    their magnitude is controlled by :attr:`dmax` and :attr:`width`.

    Example:
        .. code-block:: python

            from newton.solvers import SolverMuJoCo, ContactsFromDistance

            cfd = ContactsFromDistance(width=0.05)
            solver = SolverMuJoCo(model, cfd=cfd)
    """

    width: float = 0.05
    """CFD distance [m]. Virtual contacts are generated for ``0 < r <= width``."""

    dmin: float = 0.0
    """Lower bound of MuJoCo's impedance spline for CFD contacts (``solimp[0]``)."""

    dmax: float = 0.01
    """Upper bound of MuJoCo's impedance spline for CFD contacts (``solimp[1]``).

    Values close to ``1`` produce stiff virtual contacts; the default
    (``0.01``) is intentionally soft so that forward-pass behavior in the
    CFD band is close to free flight.
    """

    midpoint: float = 0.5
    """Normalized spline midpoint (``solimp[3]``)."""

    power: float = 2.0
    """Polynomial spline power (``solimp[4]``)."""

    timeconst: float = 0.02
    """Reference-acceleration time constant [s] (``solref[0]``)."""

    damping_ratio: float = 1.0
    """Reference-acceleration damping ratio (``solref[1]``)."""

    straight_through: bool = True
    """Reserved for the paper's straight-through trick.

    When ``True``, the intent is to run the forward step with the original
    parameters and route gradients through a second step with CFD-modified
    parameters so that forward-pass realism is preserved. Full straight-through
    propagation through ``mujoco_warp``'s ``step`` is an ongoing effort: this
    initial implementation accepts the flag but applies the CFD parameters
    directly and emits a warning. Setting ``False`` makes the behavior
    explicit and silences the warning.
    """

    enabled: bool = True
    """Master on/off switch. When ``False`` the solver behaves like stock MuJoCo."""

    def __post_init__(self) -> None:
        if self.width <= 0.0:
            raise ValueError(f"ContactsFromDistance.width must be positive, got {self.width}")
        if not (0.0 <= self.dmin <= self.dmax < 1.0):
            raise ValueError(
                f"ContactsFromDistance requires 0 <= dmin ({self.dmin}) <= dmax ({self.dmax}) < 1"
            )
        if not (0.0 < self.midpoint < 1.0):
            raise ValueError(f"ContactsFromDistance.midpoint must be in (0, 1), got {self.midpoint}")
        if self.power < 1.0:
            raise ValueError(f"ContactsFromDistance.power must be >= 1, got {self.power}")
        if self.timeconst <= 0.0:
            raise ValueError(f"ContactsFromDistance.timeconst must be positive, got {self.timeconst}")
        if self.damping_ratio <= 0.0:
            raise ValueError(
                f"ContactsFromDistance.damping_ratio must be positive, got {self.damping_ratio}"
            )

    def solimp_vec(self) -> vec5:
        """Return the CFD-flavored ``solimp`` vector used to rewrite per-geom values.

        The ``solimp[2]`` (spline width) component is set to ``2 * self.width``
        so the MuJoCo impedance spline remains soft both on the CFD side of
        the contact boundary (``r > 0``) and in a small penetration band.
        """
        return vec5(self.dmin, self.dmax, 2.0 * self.width, self.midpoint, self.power)

    def solref_vec(self) -> wp.vec2:
        """Return the CFD-flavored ``solref`` vector."""
        return wp.vec2(self.timeconst, self.damping_ratio)


@wp.kernel
def apply_cfd_geom_params_kernel(
    # CFD parameters (broadcast).
    cfd_margin: float,
    cfd_solref: wp.vec2,
    cfd_solimp: vec5,
    # Original per-world, per-geom baseline margins.
    original_margin: wp.array2d[float],
    # Outputs: modified per-world, per-geom arrays.
    geom_margin: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
):
    """Rewrite a single ``(world, geom)`` entry with CFD-modified values.

    ``geom_margin`` is set to ``max(original, cfd_margin)`` so any
    user-specified margin is preserved if larger. ``geom_solref`` and
    ``geom_solimp`` are replaced wholesale with the CFD-flavored values.
    """
    world, geom = wp.tid()
    orig = original_margin[world, geom]
    geom_margin[world, geom] = wp.max(orig, cfd_margin)
    geom_solref[world, geom] = cfd_solref
    geom_solimp[world, geom] = cfd_solimp


@wp.kernel
def restore_cfd_geom_params_kernel(
    original_margin: wp.array2d[float],
    original_solref: wp.array2d[wp.vec2],
    original_solimp: wp.array2d[vec5],
    geom_margin: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
):
    """Restore per-geom contact parameters to their original values."""
    world, geom = wp.tid()
    geom_margin[world, geom] = original_margin[world, geom]
    geom_solref[world, geom] = original_solref[world, geom]
    geom_solimp[world, geom] = original_solimp[world, geom]


class _CFDState:
    """Internal bookkeeping: snapshots of the unmodified geom parameters.

    When a solver is configured with :class:`ContactsFromDistance`, we take a
    clone of the relevant per-geom arrays at construction time (and whenever
    the user calls :meth:`~newton.solvers.SolverBase.notify_model_changed`
    with shape-property flags). Those clones are the ground truth we restore
    to whenever CFD is temporarily disabled or the user removes CFD entirely.
    """

    def __init__(
        self,
        geom_margin: wp.array,
        geom_solref: wp.array,
        geom_solimp: wp.array,
    ) -> None:
        self.original_margin: wp.array | None = wp.clone(geom_margin)
        """Snapshot of ``mjw_model.geom_margin``, dtype float."""
        self.original_solref: wp.array | None = wp.clone(geom_solref)
        """Snapshot of ``mjw_model.geom_solref``, dtype :class:`wp.vec2`."""
        self.original_solimp: wp.array | None = wp.clone(geom_solimp)
        """Snapshot of ``mjw_model.geom_solimp``, dtype :class:`vec5`."""

    def refresh(
        self,
        geom_margin: wp.array,
        geom_solref: wp.array,
        geom_solimp: wp.array,
    ) -> None:
        """Re-snapshot originals after the solver-level geom params changed.

        Reallocates if shapes changed; otherwise reuses the existing buffers.
        """
        if (
            self.original_margin is None
            or self.original_margin.shape != geom_margin.shape
            or self.original_solref.shape != geom_solref.shape
            or self.original_solimp.shape != geom_solimp.shape
        ):
            self.original_margin = wp.clone(geom_margin)
            self.original_solref = wp.clone(geom_solref)
            self.original_solimp = wp.clone(geom_solimp)
        else:
            wp.copy(self.original_margin, geom_margin)
            wp.copy(self.original_solref, geom_solref)
            wp.copy(self.original_solimp, geom_solimp)


def apply_cfd_to_mjw_model(
    mjw_model,
    cfd_state: _CFDState,
    cfd: ContactsFromDistance,
    device,
) -> None:
    """Rewrite ``mjw_model`` per-geom contact parameters in-place with CFD values."""
    n_world = mjw_model.geom_margin.shape[0]
    n_geom = mjw_model.geom_margin.shape[1]
    if n_world == 0 or n_geom == 0:
        return
    wp.launch(
        apply_cfd_geom_params_kernel,
        dim=(n_world, n_geom),
        inputs=[
            float(cfd.width),
            cfd.solref_vec(),
            cfd.solimp_vec(),
            cfd_state.original_margin,
        ],
        outputs=[
            mjw_model.geom_margin,
            mjw_model.geom_solref,
            mjw_model.geom_solimp,
        ],
        device=device,
    )


def restore_cfd_original(
    mjw_model,
    cfd_state: _CFDState,
    device,
) -> None:
    """Restore ``mjw_model`` per-geom contact parameters from the snapshot."""
    n_world = mjw_model.geom_margin.shape[0]
    n_geom = mjw_model.geom_margin.shape[1]
    if n_world == 0 or n_geom == 0:
        return
    wp.launch(
        restore_cfd_geom_params_kernel,
        dim=(n_world, n_geom),
        inputs=[
            cfd_state.original_margin,
            cfd_state.original_solref,
            cfd_state.original_solimp,
        ],
        outputs=[
            mjw_model.geom_margin,
            mjw_model.geom_solref,
            mjw_model.geom_solimp,
        ],
        device=device,
    )
