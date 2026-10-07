# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-signal discrepancies between a reference trajectory and candidate trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, wp.array):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _component_names(shape: tuple[int, ...], labels: Sequence[str] | None) -> list[str]:
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if labels is not None:
        labels = [str(label) for label in labels]
        if len(labels) != count:
            raise ValueError(f"{len(labels)} labels for {count} components")
        return labels
    if len(shape) <= 1:
        return [str(i) for i in range(count)]
    return [",".join(str(int(i)) for i in np.unravel_index(c, shape)) for c in range(count)]


class TrajectoryComparison:
    """Discrepancies between a reference and a candidate, per named signal.

    Returned by :func:`compare_trajectories`. For a batched candidate (one
    trajectory per world), every per-signal value is an array with one entry
    per world.
    """

    def __init__(self):
        self.signals: list[str] = []
        """Names of the compared signals (present in both inputs)."""
        self.missing: list[str] = []
        """Names present in only one of the inputs."""
        self.batched: bool = False
        """Whether the candidate holds one trajectory per world."""
        self.times: np.ndarray = np.zeros(0)
        """Compared sample times [s], or sample indices without times, shape [T]."""
        self.time_unit: str = "s"
        """``"s"`` for times, ``"sample"`` for sample indices."""
        self.rmse: dict[str, float | np.ndarray] = {}
        """Root mean square error over samples and components."""
        self.max_error: dict[str, float | np.ndarray] = {}
        """Largest absolute error of any component."""
        self.max_time: dict[str, float | np.ndarray] = {}
        """Time of :attr:`max_error`."""
        self.divergence_time: dict[str, float | np.ndarray] = {}
        """First time a component's absolute error exceeds the signal's tolerance or is not finite; NaN if never."""
        self.tolerance: dict[str, float | None] = {}
        """Divergence tolerance per signal (``None``: only non-finite values diverge)."""
        self.contributors: dict[str, list] = {}
        """Components with the largest RMSE over time, largest first: ``[(component, rmse), ...]``, per world for a
        batched candidate. Components are the trailing axes of a signal, named by ``labels`` or by index."""

    def objective(self, weights: Mapping[str, float] | None = None) -> float | np.ndarray:
        """Weighted sum of the signals' RMSE, e.g. as the objective of a fit (one value per world when batched).

        Args:
            weights: Weight per signal; signals not listed weigh 1. Signals
                with weight 0 are left out.
        """
        weights = dict(weights or {})
        unknown = set(weights) - set(self.signals)
        if unknown:
            raise KeyError(f"no compared signals named {sorted(unknown)}; signals: {self.signals}")
        total = 0.0
        for name in self.signals:
            weight = float(weights.get(name, 1.0))
            if weight:
                total = total + weight * np.asarray(self.rmse[name])
        return float(total) if np.ndim(total) == 0 else np.asarray(total)

    def format(self, *, digits: int = 4) -> str:
        """The comparison as text: one line per signal."""

        def number(value):
            return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}g}"

        unit = "" if self.time_unit == "sample" else " s"
        lines = []
        for name in self.signals:
            if not self.batched:
                divergence = self.divergence_time[name]
                diverges = "no divergence" if math.isnan(divergence) else f"diverges at {number(divergence)}{unit}"
                top = ", ".join(f"{component} {number(value)}" for component, value in self.contributors[name])
                lines.append(
                    f"{name}: rmse {number(self.rmse[name])}, max {number(self.max_error[name])} at "
                    f"{number(self.max_time[name])}{unit}, {diverges}; largest: {top}"
                )
            else:
                rmse = np.asarray(self.rmse[name])
                worst = int(np.nanargmax(np.where(np.isfinite(rmse), rmse, np.inf))) if rmse.size else 0
                diverged = int(np.sum(~np.isnan(np.asarray(self.divergence_time[name]))))
                top = ", ".join(f"{component} {number(value)}" for component, value in self.contributors[name][worst])
                lines.append(
                    f"{name}: rmse min {number(float(np.nanmin(rmse)))} median {number(float(np.nanmedian(rmse)))} "
                    f"max {number(float(rmse[worst]))} (world {worst}); {diverged}/{rmse.size} worlds diverge; "
                    f"world {worst} largest: {top}"
                )
        if self.missing:
            lines.append("not compared (in one input only): " + ", ".join(self.missing))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def compare_trajectories(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    times: Any = None,
    candidate_times: Any = None,
    tolerance: float | Mapping[str, float] | None = None,
    labels: Mapping[str, Sequence[str]] | None = None,
    top: int = 3,
) -> TrajectoryComparison:
    """Compare named time series of a candidate against a reference.

    Signals are the names in both inputs, each an array whose first axis is
    time: shape ``[T, ...]`` in the reference and ``[T, ...]`` or, for one
    trajectory per world (e.g. records of :meth:`BatchRollout.run
    <newton.utils.BatchRollout.run>`), ``[T, world, ...]`` in the candidate.
    The remaining axes are the signal's components, e.g. joints. Use
    separate signals for quantities with different units (positions and
    rotations, joints of different types).

    Per signal, the comparison reports the RMSE over samples and
    components, the largest absolute component error and its time, the
    first time an error exceeds the tolerance, and the components with the
    largest RMSE. :meth:`TrajectoryComparison.objective` sums the RMSE as an
    objective, e.g. for a fit of parameters across worlds.

    Example:

    .. code-block:: python

        log = {"arm": measured_q}  # [T, 6] [rad]
        records = rollout.run(frames, record={"arm": ("joint_q", "arm/*")}, every=4)
        result = newton.utils.compare_trajectories(
            log, {"arm": records["arm"]}, times=log_t, candidate_times=rollout.record_time, tolerance=0.05
        )
        print(result)  # one line per signal
        best_world = int(np.argmin(result.objective()))

    Args:
        reference: Reference signals by name, e.g. a log.
        candidate: Candidate signals by name.
        times: Sample times [s] of the reference, shape [T]; also of the
            candidate unless ``candidate_times`` is given. Without times,
            both inputs must have the same number of samples and times are
            sample indices.
        candidate_times: Sample times [s] of the candidate. The candidate is
            interpolated linearly to the reference times within the time
            range both cover.
        tolerance: Absolute error at which a signal diverges, for all
            signals or per signal name.
        labels: Component names per signal, for the contributors.
        top: Number of contributors per signal.

    Returns:
        The per-signal discrepancies.
    """
    if candidate_times is not None and times is None:
        raise ValueError("candidate_times needs the reference times")
    result = TrajectoryComparison()
    result.signals = [name for name in reference if name in candidate]
    result.missing = sorted(set(reference) ^ set(candidate))
    if not result.signals:
        raise ValueError(f"no signal names in both inputs: {sorted(reference)} and {sorted(candidate)}")
    labels = dict(labels or {})

    reference_arrays = {name: _as_numpy(reference[name]) for name in result.signals}
    candidate_arrays = {name: _as_numpy(candidate[name]) for name in result.signals}
    sample_counts = {array.shape[0] for array in reference_arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"reference signals have different numbers of samples: {sorted(sample_counts)}")
    count = sample_counts.pop()

    batch = {name: candidate_arrays[name].ndim == reference_arrays[name].ndim + 1 for name in result.signals}
    if len(set(batch.values())) != 1:
        raise ValueError("either every candidate signal has a world axis after the time axis or none has")
    result.batched = next(iter(batch.values()))

    if times is None:
        result.times, result.time_unit = np.arange(count, dtype=np.float64), "sample"
        mismatched = [name for name in result.signals if candidate_arrays[name].shape[0] != count]
        if mismatched:
            raise ValueError(
                f"signals {mismatched} have {candidate_arrays[mismatched[0]].shape[0]} candidate samples and {count} "
                "reference samples; pass times (and candidate_times) to compare them on a common time base"
            )
        keep = slice(None)
    else:
        reference_times = np.asarray(times, dtype=np.float64)
        if reference_times.shape != (count,):
            raise ValueError(f"times has shape {reference_times.shape}, the reference has {count} samples")
        if candidate_times is None:
            keep = slice(None)
            mismatched = [name for name in result.signals if candidate_arrays[name].shape[0] != count]
            if mismatched:
                raise ValueError(f"signals {mismatched} need candidate_times: their sample counts differ")
        else:
            source_times = np.asarray(candidate_times, dtype=np.float64)
            span = 1e-9 * max(1.0, float(np.abs(reference_times).max(initial=0.0)))
            inside = (reference_times >= source_times[0] - span) & (reference_times <= source_times[-1] + span)
            keep = np.flatnonzero(inside)
            if keep.size == 0:
                raise ValueError("the reference and candidate times do not overlap")
            for name in result.signals:
                values = candidate_arrays[name]
                if values.shape[0] != source_times.shape[0]:
                    raise ValueError(
                        f"candidate '{name}' has {values.shape[0]} samples, candidate_times {source_times.shape[0]}"
                    )
                flat = values.reshape(values.shape[0], -1)
                resampled = np.empty((keep.size, flat.shape[1]))
                for column in range(flat.shape[1]):
                    resampled[:, column] = np.interp(reference_times[keep], source_times, flat[:, column])
                candidate_arrays[name] = resampled.reshape(keep.size, *values.shape[1:])
            for name in result.signals:
                reference_arrays[name] = reference_arrays[name][keep]
        result.times = reference_times[keep]

    for name in result.signals:
        ref, cand = reference_arrays[name], candidate_arrays[name]
        if not (times is not None and candidate_times is not None):
            cand = cand[keep]
        if result.batched:
            if cand.shape[2:] != ref.shape[1:]:
                raise ValueError(f"signal '{name}': candidate components {cand.shape[2:]} != reference {ref.shape[1:]}")
            worlds = cand.shape[1]
            error = (cand - ref[:, None]).reshape(cand.shape[0], worlds, -1)  # [T, W, C]
        else:
            if cand.shape != ref.shape:
                raise ValueError(f"signal '{name}': candidate shape {cand.shape} != reference {ref.shape}")
            error = (cand - ref).reshape(cand.shape[0], 1, -1)
        names = _component_names(ref.shape[1:], labels.get(name))
        magnitude = np.abs(error)
        magnitude[~np.isfinite(magnitude)] = np.inf
        per_sample = magnitude.max(axis=2)  # [T, W]
        squared = np.square(magnitude)
        rmse = np.sqrt(squared.mean(axis=(0, 2)))
        component_rmse = np.sqrt(squared.mean(axis=0))  # [W, C]
        peak = per_sample.argmax(axis=0)
        max_error = per_sample[peak, np.arange(per_sample.shape[1])]
        max_time = result.times[peak]
        limit = tolerance.get(name) if isinstance(tolerance, Mapping) else tolerance
        result.tolerance[name] = None if limit is None else float(limit)
        exceeded = ~np.isfinite(per_sample) if limit is None else (per_sample > float(limit))
        first = exceeded.argmax(axis=0)
        divergence = np.where(exceeded.any(axis=0), result.times[first], np.nan)
        order = np.argsort(-component_rmse, axis=1, kind="stable")[:, : max(0, int(top))]
        contributors = [
            [(names[c], float(component_rmse[w, c])) for c in order[w]] for w in range(component_rmse.shape[0])
        ]
        if result.batched:
            result.rmse[name], result.max_error[name] = rmse, max_error
            result.max_time[name], result.divergence_time[name] = max_time, divergence
            result.contributors[name] = contributors
        else:
            result.rmse[name], result.max_error[name] = float(rmse[0]), float(max_error[0])
            result.max_time[name], result.divergence_time[name] = float(max_time[0]), float(divergence[0])
            result.contributors[name] = contributors[0]
    return result
