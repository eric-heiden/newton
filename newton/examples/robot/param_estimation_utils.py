# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the Franka real-to-sim parameter-estimation prototype."""

from __future__ import annotations

import numpy as np


def positive_to_raw(values: np.ndarray, minimum: float = 1.0e-6) -> np.ndarray:
    """Map positive physical parameters to unconstrained log parameters.

    Args:
        values: Positive parameter values.
        minimum: Lower physical bound used before taking the logarithm.

    Returns:
        Log-encoded parameter vector.
    """

    values = np.asarray(values, dtype=np.float32)
    return np.log(np.maximum(values, np.float32(minimum))).astype(np.float32)


def raw_to_positive(raw_values: np.ndarray) -> np.ndarray:
    """Map unconstrained log parameters back to positive physical values.

    Args:
        raw_values: Log-encoded parameter vector.

    Returns:
        Positive parameter values.
    """

    return np.exp(np.asarray(raw_values, dtype=np.float32)).astype(np.float32)


def make_excitation_trajectory(
    steps: int,
    dof_count: int,
    dt: float,
    seed: int,
    amplitude: float = 4.0,
) -> np.ndarray:
    """Create a deterministic multi-sine torque trajectory.

    Args:
        steps: Number of simulation steps.
        dof_count: Number of actuated joints.
        dt: Time step [s].
        seed: Random seed for phases and small amplitude variation.
        amplitude: Nominal torque amplitude [N m].

    Returns:
        Joint torque sequence with shape ``(steps, dof_count)``.
    """

    rng = np.random.default_rng(seed)
    time = np.arange(steps, dtype=np.float32)[:, None] * np.float32(dt)
    joint = np.arange(dof_count, dtype=np.float32)[None, :]
    phase = rng.uniform(-np.pi, np.pi, size=(1, dof_count)).astype(np.float32)
    scale = rng.uniform(0.75, 1.25, size=(1, dof_count)).astype(np.float32)

    slow = np.sin((0.8 + 0.17 * joint) * time + phase)
    fast = 0.45 * np.sin((1.7 + 0.11 * joint) * time - 0.5 * phase)
    chirp = 0.25 * np.sin((0.04 * joint + 0.15) * time * time + 0.25 * phase)
    return (np.float32(amplitude) * scale * (slow + fast + chirp)).astype(np.float32)


def estimate_relative_errors(estimates: dict[str, float], truth: dict[str, float]) -> dict[str, float]:
    """Compute absolute fractional errors for shared scalar parameters.

    Args:
        estimates: Estimated physical parameters.
        truth: Ground-truth physical parameters.

    Returns:
        Mapping from parameter name to ``abs(estimate - truth) / abs(truth)``.
    """

    errors: dict[str, float] = {}
    for name, true_value in truth.items():
        denom = max(abs(float(true_value)), 1.0e-12)
        errors[name] = abs(float(estimates[name]) - float(true_value)) / denom
    return errors
