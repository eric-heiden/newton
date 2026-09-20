# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Physical parameterization and geometry-only construction for measured Panda fitting."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton

PARAMETER_SHAPES = {
    "mass": (7,),
    "com": (7, 3),
    "inertia": (7, 3, 3),
    "viscous": (7,),
    "coulomb": (7,),
    "torque_bias": (7,),
    "armature": (7,),
}
INERTIA_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
BOUNDS = {
    "mass": [0.05, 10.0],
    "com": [-0.4, 0.4],
    "inertia": "Symmetric COM tensor; eigenvalues >=1e-8 kg*m^2 and physical triangle inequalities",
    "second_moment": "trace(0.5*trace(I_origin)*identity-I_origin) <= mass*(0.5 m)^2",
    "viscous": [0.0, 5.0],
    "coulomb": [0.0, 5.0],
    "torque_bias": [-2.0, 2.0],
    "armature": [0.0, 1.0],
}


def initial_config() -> dict:
    """Return homogeneous placeholders independent of authored robot dynamics."""
    values = {key: np.zeros(shape) for key, shape in PARAMETER_SHAPES.items()}
    values["mass"][:] = 1.0
    values["inertia"][:] = 0.01 * np.eye(3)
    return {key: value.tolist() for key, value in values.items()}


def validate_config(config: dict) -> dict:
    """Require finite full physical tensors and broad geometry-based bounds."""
    if set(config) != set(PARAMETER_SHAPES):
        raise ValueError(f"Configuration requires exactly {sorted(PARAMETER_SHAPES)}")
    values = {}
    for key, shape in PARAMETER_SHAPES.items():
        value = np.asarray(config[key], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite values with shape {shape}")
        if key != "inertia" and (np.any(value < BOUNDS[key][0]) or np.any(value > BOUNDS[key][1])):
            raise ValueError(f"{key} must lie within {BOUNDS[key]}")
        values[key] = value
    for joint, tensor in enumerate(values["inertia"]):
        if not np.allclose(tensor, tensor.T, rtol=0, atol=1e-10):
            raise ValueError("COM inertia must be symmetric")
        eigenvalues = np.linalg.eigvalsh(tensor)
        if eigenvalues[0] < 1e-8 or eigenvalues[-1] > np.sum(eigenvalues[:2]) + 1e-10:
            raise ValueError("COM inertia must be positive and satisfy the physical triangle inequalities")
        second_moment = 0.5 * np.trace(tensor) + values["mass"][joint] * np.dot(
            values["com"][joint], values["com"][joint]
        )
        if second_moment > values["mass"][joint] * 0.5**2 + 1e-10:
            raise ValueError("Link second moment exceeds the broad 0.5 m geometry bound")
    return {key: value.tolist() for key, value in values.items()}


def physical_coefficients(config: dict) -> np.ndarray:
    """Map physical properties to 70 link coefficients followed by 28 joint coefficients."""
    values = validate_config(config)
    coefficients = []
    for mass, center_values, tensor in zip(values["mass"], values["com"], values["inertia"], strict=True):
        center = np.asarray(center_values)
        origin_tensor = np.asarray(tensor) + mass * (np.dot(center, center) * np.eye(3) - np.outer(center, center))
        coefficients.extend([mass, *(mass * center), *(origin_tensor[a, b] for a, b in INERTIA_COMPONENTS)])
    for key in ("viscous", "coulomb", "torque_bias", "armature"):
        coefficients.extend(values[key])
    return np.asarray(coefficients)


def make_builder(geometry_file: Path, config: dict, *, visual: bool = True) -> tuple[newton.ModelBuilder, list[int]]:
    """Build seven moving links with explicit candidate dynamics, never inferred mesh mass."""
    values = validate_config(config)
    geometry_file = Path(geometry_file).resolve()
    root = ET.parse(geometry_file).getroot()
    if root.find(".//inertial") is not None or root.find("actuator") is not None:
        raise ValueError("Only the sanitized geometry-only Panda asset is permitted")
    for parent in root.iter():
        if parent.tag == "joint" and any(key in parent.attrib for key in ("damping", "frictionloss", "armature")):
            raise ValueError("Geometry input must not carry authored joint losses")
        for child in list(parent):
            if not visual and child.tag in ("geom", "asset"):
                parent.remove(child)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(geometry_file.parent))
    compiler.set("texturedir", str(geometry_file.parent))
    builder = newton.ModelBuilder()
    builder.add_mjcf(ET.tostring(root, encoding="unicode"), enable_self_collisions=False)
    if builder.joint_dof_count != 7:
        raise ValueError("The sanitized Panda must contain seven revolute degrees of freedom")
    indices = []
    for joint in range(1, 8):
        matches = [i for i, label in enumerate(builder.body_label) if label.rsplit("/", 1)[-1] == f"link{joint}"]
        if len(matches) != 1:
            raise ValueError(f"Expected one moving link{joint} in the sanitized geometry")
        indices.append(matches[0])
    for body in range(builder.body_count):
        builder.body_mass[body] = 0.0
        builder.body_com[body] = wp.vec3(0.0)
        builder.body_inertia[body] = wp.mat33(0.0)
    for joint, body in enumerate(indices):
        builder.body_mass[body] = values["mass"][joint]
        builder.body_com[body] = wp.vec3(values["com"][joint])
        builder.body_inertia[body] = wp.mat33(values["inertia"][joint])
    builder.joint_damping[:] = values["viscous"]
    builder.joint_friction[:] = values["coulomb"]
    builder.joint_armature[:] = values["armature"]
    builder.joint_target_ke[:] = [0.0] * 7
    builder.joint_target_kd[:] = [0.0] * 7
    builder.joint_effort_limit[:] = [1e6] * 7
    return builder, indices
