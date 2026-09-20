# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Construct an immutable Newton inverse-dynamics design matrix without fitting parameters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .real_robot_data import file_digest, load_reference
from .real_robot_model import INERTIA_COMPONENTS, initial_config, make_builder


def sample_indices(reference: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Select the prespecified 150 torque observations in every complete episode."""
    indices, episodes = [], []
    for episode, start, stop in zip(
        reference["episode_ids"], reference["episode_offsets"][:-1], reference["episode_offsets"][1:], strict=True
    ):
        indices.extend(start + np.linspace(30, stop - start - 31, 150, dtype=np.int64))
        episodes.extend([episode] * 150)
    return np.asarray(indices, dtype=np.int64), np.asarray(episodes, dtype=np.int64)


def inverse_dynamics_matrix(geometry_file: Path, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    """Evaluate the 98 linear coefficient columns with Newton's public inverse dynamics.

    Signed unit probes are algebraic basis vectors, not physically admissible
    simulation candidates. No recorded torque or fitted parameter enters this
    construction. Candidate forward simulations require physical tensors.
    """
    if q.ndim != 2 or q.shape[1] != 7 or qd.shape != q.shape or qdd.shape != q.shape or not len(q):
        raise ValueError("Inverse-dynamics observations must have identical nonempty [n, 7] shapes")
    if not all(np.isfinite(value).all() for value in (q, qd, qdd)):
        raise ValueError("Inverse-dynamics observations must be finite")
    with wp.ScopedDevice("cpu"):
        base, indices = make_builder(geometry_file, initial_config(), visual=False)
        builder = newton.ModelBuilder()
        for _ in range(len(q)):
            builder.add_builder(base)
        model = builder.finalize(device="cpu")
        state = model.state()
        state.joint_q.assign(q.astype(np.float32).ravel())
        state.joint_qd.assign(qd.astype(np.float32).ravel())
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        matrix = wp.zeros(
            (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation), dtype=float
        )
        coriolis = wp.zeros(model.joint_dof_count, dtype=float)
        gravity, force = wp.zeros_like(coriolis), wp.zeros_like(coriolis)
        acceleration = wp.array(qdd.astype(np.float32).ravel(), dtype=float)

        def evaluate(mass: np.ndarray, com: np.ndarray, inertia: np.ndarray) -> np.ndarray:
            model.body_mass.assign(np.tile(mass, len(q)).astype(np.float32))
            model.body_com.assign(np.tile(com, (len(q), 1)).astype(np.float32))
            model.body_inertia.assign(np.tile(inertia, (len(q), 1, 1)).astype(np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            newton.eval_inverse_dynamics_passive(
                model, state, mass_matrix=matrix, coriolis_force=coriolis, gravity_force=gravity
            )
            newton.eval_inverse_dynamics_force(
                model,
                state,
                mass_matrix=matrix,
                joint_qdd=acceleration,
                coriolis_force=coriolis,
                gravity_force=gravity,
                joint_f=force,
            )
            return force.numpy().reshape(-1, 7).astype(np.float64)

        columns = []
        for body in indices:
            mass, com, inertia = (
                np.zeros(base.body_count),
                np.zeros((base.body_count, 3)),
                np.zeros((base.body_count, 3, 3)),
            )
            mass[body] = 1.0
            mass_column = evaluate(mass, com, inertia)
            columns.append(mass_column)
            for axis in range(3):
                com[body] = np.eye(3)[axis]
                inertia[body] = np.outer(com[body], com[body]) - np.eye(3)
                columns.append(evaluate(mass, com, inertia) - mass_column)
            com[body], inertia[body] = 0.0, 0.0
            for a, b in INERTIA_COMPONENTS:
                inertia[body] = 0.0
                inertia[body, a, b] = inertia[body, b, a] = 1.0
                columns.append(evaluate(mass, com, inertia) - mass_column)
        for feature in (qd, np.sign(qd), np.ones_like(qd), qdd):
            for joint in range(7):
                column = np.zeros_like(q)
                column[:, joint] = feature[:, joint]
                columns.append(column)
        return np.stack(columns, axis=-1).reshape(-1, 98)


def prepare_regressor(reference_file: Path, geometry_file: Path, output_file: Path) -> dict:
    """Prepare identical immutable linear features for every evaluation condition."""
    started = time.perf_counter()
    reference = load_reference(reference_file)
    indices, episodes = sample_indices(reference)
    matrix = inverse_dynamics_matrix(
        geometry_file, reference["q"][indices], reference["qd"][indices], reference["qdd"][indices]
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file, A=matrix, b=reference["tau"][indices].ravel(), sample_indices=indices, sample_episode_ids=episodes
    )
    manifest = {
        "reference_sha256": file_digest(reference_file),
        "geometry_sha256": file_digest(geometry_file),
        "regressor_sha256": file_digest(output_file),
        "preparation_seconds": time.perf_counter() - started,
        "shape": list(matrix.shape),
        "construction": "Newton public inverse-dynamics unit basis; no fitting; identical immutable preparation outside every agent timer",
        "basis": "Link1..7 each [mass, mass*com_x, mass*com_y, mass*com_z, I_origin_xx, I_origin_yy, I_origin_zz, I_origin_xy, I_origin_xz, I_origin_yz]; then viscous1..7, Coulomb1..7, torque_bias1..7, armature1..7",
        "mapping": "I_origin=I_com+mass*(dot(com,com)*identity-outer(com,com)); joint columns are qd, sign(qd), 1, qdd",
        "rows": "C order: one 7-joint torque vector per sample; 150 fixed row samples per episode",
        "dtype": "float64 storage; Newton float32 inverse-dynamics basis evaluations",
        "data_license": "CC-BY-SA-4.0",
    }
    output_file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_regressor(
    regressor_file: Path, reference_file: Path, geometry_file: Path, reference: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verify immutable inputs and row alignment before candidate scoring."""
    manifest = json.loads(regressor_file.with_suffix(".manifest.json").read_text())
    for key, path in (
        ("reference_sha256", reference_file),
        ("geometry_sha256", geometry_file),
        ("regressor_sha256", regressor_file),
    ):
        if manifest[key] != file_digest(path):
            raise ValueError(f"Regressor provenance mismatch: {key}")
    indices, episodes = sample_indices(reference)
    with np.load(regressor_file, allow_pickle=False) as data:
        if set(data.files) != {"A", "b", "sample_indices", "sample_episode_ids"}:
            raise ValueError("Unexpected immutable regressor fields")
        matrix, target = data["A"].copy(), data["b"].copy()
        if not np.array_equal(data["sample_indices"], indices) or not np.array_equal(
            data["sample_episode_ids"], episodes
        ):
            raise ValueError("Regressor sample selection does not match the fixed protocol")
    if matrix.shape != (len(indices) * 7, 98) or target.shape != (len(indices) * 7,) or not np.isfinite(matrix).all():
        raise ValueError("Regressor arrays have invalid shapes or values")
    if not np.array_equal(target, reference["tau"][indices].ravel()):
        raise ValueError("Regressor target differs from recorded observations")
    return matrix, target, episodes


def main() -> None:
    """Prepare one training or held-out immutable design matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_regressor(args.reference, args.geometry, args.output)))


if __name__ == "__main__":
    main()
