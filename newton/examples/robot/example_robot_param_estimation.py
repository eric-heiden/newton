# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Parameter Estimation
#
# Prototype real-to-sim calibration loop for a Franka arm. The script loads a
# Franka model through Newton, uses its arm DOF metadata as the calibration
# target, and runs Warp rollout kernels with finite-difference gradients to
# estimate positive physical parameters from measured joint trajectories.
#
# Command:
#   python -m newton.examples robot_param_estimation --iterations 160 --output result.json
#
###########################################################################

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.utils
from newton.examples.robot.param_estimation_utils import (
    estimate_relative_errors,
    make_excitation_trajectory,
    positive_to_raw,
    raw_to_positive,
)


ARM_DOF_COUNT = 7
PARAMETER_NAMES = ("payload_mass", "friction_scale", "armature_scale")


@wp.kernel
def simulate_joint_trajectory_kernel(
    torque: wp.array2d[wp.float32],
    base_inertia: wp.array[wp.float32],
    base_armature: wp.array[wp.float32],
    base_friction: wp.array[wp.float32],
    payload_lever: wp.array[wp.float32],
    raw_params: wp.array[wp.float32],
    q0: wp.array[wp.float32],
    qd0: wp.array[wp.float32],
    dt: wp.float32,
    steps: int,
    q_out: wp.array2d[wp.float32],
    qd_out: wp.array2d[wp.float32],
):
    dof = wp.tid()

    payload_mass = wp.exp(raw_params[0])
    friction_scale = wp.exp(raw_params[1])
    armature_scale = wp.exp(raw_params[2])

    q = q0[dof]
    qd = qd0[dof]
    inertia = base_inertia[dof] + armature_scale * base_armature[dof]
    inertia = inertia + payload_mass * payload_lever[dof] * payload_lever[dof]
    damping = friction_scale * base_friction[dof] + wp.float32(0.01)
    gravity_scale = wp.float32(0.8) * payload_mass * payload_lever[dof]

    q_out[0, dof] = q
    qd_out[0, dof] = qd

    for step in range(steps):
        tau = torque[step, dof]
        gravity_torque = gravity_scale * wp.sin(q)
        qdd = (tau - damping * qd - gravity_torque) / inertia
        qd = qd + qdd * dt
        q = q + qd * dt
        q_out[step + 1, dof] = q
        qd_out[step + 1, dof] = qd


@wp.kernel
def rollout_loss_kernel(
    torque: wp.array2d[wp.float32],
    observed_q: wp.array2d[wp.float32],
    base_inertia: wp.array[wp.float32],
    base_armature: wp.array[wp.float32],
    base_friction: wp.array[wp.float32],
    payload_lever: wp.array[wp.float32],
    raw_params: wp.array[wp.float32],
    q0: wp.array[wp.float32],
    qd0: wp.array[wp.float32],
    dt: wp.float32,
    steps: int,
    dof_count: int,
    inv_sample_count: wp.float32,
    q_pred: wp.array2d[wp.float32],
    qd_pred: wp.array2d[wp.float32],
    loss: wp.array[wp.float32],
):
    payload_mass = wp.exp(raw_params[0])
    friction_scale = wp.exp(raw_params[1])
    armature_scale = wp.exp(raw_params[2])

    total = wp.float32(0.0)
    for dof in range(dof_count):
        q = q0[dof]
        qd = qd0[dof]
        inertia = base_inertia[dof] + armature_scale * base_armature[dof]
        inertia = inertia + payload_mass * payload_lever[dof] * payload_lever[dof]
        damping = friction_scale * base_friction[dof] + wp.float32(0.01)
        gravity_scale = wp.float32(0.8) * payload_mass * payload_lever[dof]

        q_pred[0, dof] = q
        qd_pred[0, dof] = qd

        for step in range(steps):
            tau = torque[step, dof]
            gravity_torque = gravity_scale * wp.sin(q)
            qdd = (tau - damping * qd - gravity_torque) / inertia
            qd = qd + qdd * dt
            q = q + qd * dt

            q_pred[step + 1, dof] = q
            qd_pred[step + 1, dof] = qd

            err = q - observed_q[step + 1, dof]
            total = total + err * err * inv_sample_count

    loss[0] = total


@wp.kernel
def adam_step_kernel(
    params: wp.array[wp.float32],
    grads: wp.array[wp.float32],
    first_moment: wp.array[wp.float32],
    second_moment: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    learning_rate: wp.float32,
    beta1: wp.float32,
    beta2: wp.float32,
    beta1_correction: wp.float32,
    beta2_correction: wp.float32,
    epsilon: wp.float32,
):
    i = wp.tid()
    g = grads[i]
    first_moment[i] = beta1 * first_moment[i] + (wp.float32(1.0) - beta1) * g
    second_moment[i] = beta2 * second_moment[i] + (wp.float32(1.0) - beta2) * g * g

    m_hat = first_moment[i] / beta1_correction
    v_hat = second_moment[i] / beta2_correction
    next_value = params[i] - learning_rate * m_hat / (wp.sqrt(v_hat) + epsilon)
    params[i] = wp.min(wp.max(next_value, lower[i]), upper[i])


def _build_franka_metadata(device: str) -> dict[str, np.ndarray | list[str] | float]:
    builder = newton.ModelBuilder()
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
    )

    model = builder.finalize(device=device, requires_grad=True)
    joint_labels = [str(label) for label in model.joint_label[:ARM_DOF_COUNT]]

    armature = model.joint_armature.numpy()[:ARM_DOF_COUNT].astype(np.float32)
    friction = model.joint_friction.numpy()[:ARM_DOF_COUNT].astype(np.float32)

    # Imported assets often leave these fields at zero. The prototype supplies
    # realistic small priors while preserving any nonzero asset values.
    armature = np.maximum(armature, np.array([0.12, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03], dtype=np.float32))
    friction = np.maximum(friction, np.array([0.18, 0.16, 0.13, 0.11, 0.08, 0.06, 0.04], dtype=np.float32))

    body_mass = model.body_mass.numpy().astype(np.float32)
    total_robot_mass = float(np.sum(body_mass))

    return {
        "joint_labels": joint_labels,
        "base_armature": armature,
        "base_friction": friction,
        "base_inertia": np.array([1.00, 0.85, 0.62, 0.38, 0.20, 0.13, 0.08], dtype=np.float32),
        "payload_lever": np.array([0.34, 0.31, 0.25, 0.18, 0.12, 0.075, 0.045], dtype=np.float32),
        "total_robot_mass": total_robot_mass,
    }


def _to_float_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(PARAMETER_NAMES)}


def run(args: argparse.Namespace) -> dict:
    device = args.device
    metadata = _build_franka_metadata(device)
    steps = args.steps
    dt = np.float32(args.dt)
    sample_count = np.float32(steps * ARM_DOF_COUNT)

    torque_np = make_excitation_trajectory(
        steps=steps,
        dof_count=ARM_DOF_COUNT,
        dt=args.dt,
        seed=args.seed,
        amplitude=args.torque_amplitude,
    )
    q0_np = np.array([0.0, -0.45, 0.0, -2.10, 0.0, 1.75, 0.65], dtype=np.float32)
    qd0_np = np.zeros(ARM_DOF_COUNT, dtype=np.float32)

    true_values = np.array([args.true_payload_mass, args.true_friction_scale, args.true_armature_scale], dtype=np.float32)
    initial_values = np.array(
        [args.initial_payload_mass, args.initial_friction_scale, args.initial_armature_scale], dtype=np.float32
    )
    raw_truth_np = positive_to_raw(true_values)
    raw_initial_np = positive_to_raw(initial_values)

    torque = wp.array(torque_np, dtype=wp.float32, device=device)
    base_inertia = wp.array(metadata["base_inertia"], dtype=wp.float32, device=device)
    base_armature = wp.array(metadata["base_armature"], dtype=wp.float32, device=device)
    base_friction = wp.array(metadata["base_friction"], dtype=wp.float32, device=device)
    payload_lever = wp.array(metadata["payload_lever"], dtype=wp.float32, device=device)
    q0 = wp.array(q0_np, dtype=wp.float32, device=device)
    qd0 = wp.array(qd0_np, dtype=wp.float32, device=device)

    raw_truth = wp.array(raw_truth_np, dtype=wp.float32, device=device)
    q_real = wp.zeros((steps + 1, ARM_DOF_COUNT), dtype=wp.float32, device=device)
    qd_real = wp.zeros((steps + 1, ARM_DOF_COUNT), dtype=wp.float32, device=device)
    wp.launch(
        simulate_joint_trajectory_kernel,
        dim=ARM_DOF_COUNT,
        inputs=[torque, base_inertia, base_armature, base_friction, payload_lever, raw_truth, q0, qd0, dt, steps],
        outputs=[q_real, qd_real],
        device=device,
    )

    observed_q_np = q_real.numpy()
    if args.noise_std > 0.0:
        rng = np.random.default_rng(args.seed + 1)
        observed_q_np[1:] += rng.normal(0.0, args.noise_std, size=observed_q_np[1:].shape).astype(np.float32)
    observed_q = wp.array(observed_q_np, dtype=wp.float32, device=device)

    raw_params = wp.array(raw_initial_np, dtype=wp.float32, device=device)
    raw_eval = wp.zeros(len(PARAMETER_NAMES), dtype=wp.float32, device=device)
    grad_params = wp.zeros(len(PARAMETER_NAMES), dtype=wp.float32, device=device)
    q_pred = wp.zeros((steps + 1, ARM_DOF_COUNT), dtype=wp.float32, device=device)
    qd_pred = wp.zeros((steps + 1, ARM_DOF_COUNT), dtype=wp.float32, device=device)
    loss = wp.zeros(1, dtype=wp.float32, device=device)
    first_moment = wp.zeros(len(PARAMETER_NAMES), dtype=wp.float32, device=device)
    second_moment = wp.zeros(len(PARAMETER_NAMES), dtype=wp.float32, device=device)
    raw_lower = wp.array(positive_to_raw(np.array([0.05, 0.05, 0.05], dtype=np.float32)), dtype=wp.float32, device=device)
    raw_upper = wp.array(positive_to_raw(np.array([8.0, 8.0, 8.0], dtype=np.float32)), dtype=wp.float32, device=device)

    beta1 = np.float32(0.9)
    beta2 = np.float32(0.99)
    loss_history: list[float] = []
    parameter_history: list[dict[str, float]] = []

    def evaluate_loss(raw_values: np.ndarray) -> float:
        raw_eval.assign(raw_values.astype(np.float32))
        wp.launch(
            rollout_loss_kernel,
            dim=1,
            inputs=[
                torque,
                observed_q,
                base_inertia,
                base_armature,
                base_friction,
                payload_lever,
                raw_eval,
                q0,
                qd0,
                dt,
                steps,
                ARM_DOF_COUNT,
                np.float32(1.0) / sample_count,
            ],
            outputs=[q_pred, qd_pred, loss],
            device=device,
        )
        return float(loss.numpy()[0])

    for iteration in range(args.iterations):
        raw_np = raw_params.numpy()
        current_loss = evaluate_loss(raw_np)
        fd_grad = np.zeros(len(PARAMETER_NAMES), dtype=np.float32)
        for param_idx in range(len(PARAMETER_NAMES)):
            plus = raw_np.copy()
            minus = raw_np.copy()
            plus[param_idx] += args.finite_difference_eps
            minus[param_idx] -= args.finite_difference_eps
            fd_grad[param_idx] = (evaluate_loss(plus) - evaluate_loss(minus)) / (2.0 * args.finite_difference_eps)

        loss_history.append(current_loss)
        if iteration % max(args.iterations // 20, 1) == 0 or iteration == args.iterations - 1:
            parameter_history.append({"iteration": iteration, **_to_float_dict(raw_to_positive(raw_np))})

        grad_params.assign(fd_grad)
        wp.launch(
            adam_step_kernel,
            dim=len(PARAMETER_NAMES),
            inputs=[
                raw_params,
                grad_params,
                first_moment,
                second_moment,
                raw_lower,
                raw_upper,
                np.float32(args.learning_rate),
                beta1,
                beta2,
                np.float32(1.0 - beta1 ** (iteration + 1)),
                np.float32(1.0 - beta2 ** (iteration + 1)),
                np.float32(1.0e-8),
            ],
            device=device,
        )

    final_values = raw_to_positive(raw_params.numpy())
    estimates = _to_float_dict(final_values)
    truth = _to_float_dict(true_values)
    relative_errors = estimate_relative_errors(estimates, truth)
    final_loss = evaluate_loss(raw_params.numpy())

    result = {
        "robot": "Franka FR3 / Panda hand",
        "method": "Warp rollout finite-difference Adam over positive real2sim parameters",
        "device": str(device),
        "steps": steps,
        "dt": float(dt),
        "iterations": args.iterations,
        "noise_std": args.noise_std,
        "joint_labels": metadata["joint_labels"],
        "newton_model_summary": {
            "arm_dof_count": ARM_DOF_COUNT,
            "total_robot_mass": metadata["total_robot_mass"],
            "parameter_fields": ["body_mass", "body_inertia", "joint_armature", "joint_friction"],
        },
        "truth": truth,
        "initial": _to_float_dict(initial_values),
        "estimate": estimates,
        "relative_errors": relative_errors,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "parameter_history": parameter_history,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate Franka real-to-sim parameters with Warp autodiff.")
    parser.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0.")
    parser.add_argument("--steps", type=int, default=180, help="Number of trajectory samples.")
    parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="Simulation time step [s].")
    parser.add_argument("--iterations", type=int, default=160, help="Adam iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.045, help="Adam learning rate for raw parameters.")
    parser.add_argument("--finite-difference-eps", type=float, default=1.0e-3, help="Raw-parameter finite difference step.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for excitation and observation noise.")
    parser.add_argument("--torque-amplitude", type=float, default=4.0, help="Multi-sine excitation amplitude [N m].")
    parser.add_argument("--noise-std", type=float, default=0.0015, help="Observation noise standard deviation [rad].")
    parser.add_argument("--true-payload-mass", type=float, default=1.35, help="Synthetic real payload mass [kg].")
    parser.add_argument("--true-friction-scale", type=float, default=1.65, help="Synthetic real joint friction scale.")
    parser.add_argument("--true-armature-scale", type=float, default=0.72, help="Synthetic real armature scale.")
    parser.add_argument("--initial-payload-mass", type=float, default=0.45, help="Initial payload mass guess [kg].")
    parser.add_argument("--initial-friction-scale", type=float, default=0.65, help="Initial joint friction scale guess.")
    parser.add_argument("--initial-armature-scale", type=float, default=1.80, help="Initial armature scale guess.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    return parser


def main():
    args = create_parser().parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ("truth", "initial", "estimate", "relative_errors", "final_loss")}, indent=2))


if __name__ == "__main__":
    main()
