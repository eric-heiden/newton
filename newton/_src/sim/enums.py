# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings
from enum import EnumMeta, IntEnum


class ModelFlags(IntEnum):
    """Flags indicating which parts of the model have been updated.

    These flags are used with :meth:`~newton.solvers.SolverBase.notify_model_changed`
    to specify which properties have changed, allowing the solver to efficiently
    update only the necessary components.

    Categories overlap semantically, but each flag has its own bit. The broad
    :attr:`JOINT_DOF_PROPERTIES` includes force, inertial, and reference-pose
    updates. Testing the broad bit alone does not detect a narrower notification.

    Combine flags with ``|``, e.g. ``ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.BODY_PROPERTIES``;
    the result is an ``int`` that :meth:`~newton.solvers.SolverBase.notify_model_changed` accepts.
    :attr:`ALL` selects every flag. ``ALL`` is itself a member, so iterating over
    :class:`ModelFlags` yields it after the single flags. Only members convert back to
    :class:`ModelFlags`: ``ModelFlags(0)`` and ``ModelFlags(3)`` raise :class:`ValueError`.
    """

    JOINT_PROPERTIES = 1 << 0
    """Indicates joint property updates: joint_q, joint_X_p, joint_X_c, joint_axis."""

    JOINT_DOF_PROPERTIES = 1 << 1
    """Indicates all joint DOF updates, including force, armature, and reference-pose properties: joint_target_ke, joint_target_kd, joint_damping, joint_effort_limit, joint_armature, joint_friction, joint_limit_ke, joint_limit_kd, joint_limit_lower, joint_limit_upper."""

    BODY_PROPERTIES = 1 << 2
    """Indicates body property updates: body_q, body_qd, body_flags."""

    BODY_INERTIAL_PROPERTIES = 1 << 3
    """Indicates body inertial property updates: body_com, body_inertia, body_inv_inertia, body_mass, body_inv_mass."""

    SHAPE_PROPERTIES = 1 << 4
    """Indicates shape property updates: shape_transform, shape_scale, shape_collision_radius, shape_material_mu, shape_material_ke, shape_material_kd, rigid_contact_mu_torsional, rigid_contact_mu_rolling."""

    MODEL_PROPERTIES = 1 << 5
    """Indicates model property updates: gravity and other global parameters."""

    CONSTRAINT_PROPERTIES = 1 << 6
    """Indicates constraint property updates: equality constraints (mujoco.equality_constraint_anchor, mujoco.equality_constraint_relpose, mujoco.equality_constraint_polycoef, mujoco.equality_constraint_torquescale, mujoco.equality_constraint_enabled, mujoco.eq_solref, mujoco.eq_solimp) and mimic relationships (joint_mimic_coeffs and the deprecated constraint_mimic_coef0, constraint_mimic_coef1, constraint_mimic_enabled arrays)."""

    TENDON_PROPERTIES = 1 << 7
    """Indicates tendon properties: eg tendon_stiffness."""

    ACTUATOR_PROPERTIES = 1 << 8
    """Indicates actuator property updates: gains, biases, limits, etc."""

    JOINT_DOF_FORCE_PROPERTIES = 1 << 9
    """Indicates joint force updates: friction, damping, target gains/modes, effort limits, passive stiffness, and limit coefficients/bounds. Excludes armature and reference poses."""

    JOINT_DOF_INERTIAL_PROPERTIES = 1 << 10
    """Indicates joint_armature updates. MuJoCo recomputes constants; use at reset or for domain randomization rather than every step."""

    JOINT_REFERENCE_POSE_PROPERTIES = 1 << 11
    """Indicates joint reference-pose and spring-reference updates. Excludes joint transforms, force parameters, and armature. MuJoCo recomputes constants; use at reset or for domain randomization rather than every step."""

    ALL = (
        JOINT_PROPERTIES
        | JOINT_DOF_PROPERTIES
        | BODY_PROPERTIES
        | BODY_INERTIAL_PROPERTIES
        | SHAPE_PROPERTIES
        | MODEL_PROPERTIES
        | CONSTRAINT_PROPERTIES
        | TENDON_PROPERTIES
        | ACTUATOR_PROPERTIES
        | JOINT_DOF_FORCE_PROPERTIES
        | JOINT_DOF_INERTIAL_PROPERTIES
        | JOINT_REFERENCE_POSE_PROPERTIES
    )
    """Indicates all property updates."""

    @classmethod
    def _missing_(cls, value: object):
        # Combinations are plain ints; name the spellings that work instead of Enum's bare
        # "is not a valid" message (e.g. for ``~ModelFlags(0)`` meant as "all flags").
        raise ValueError(
            f"{value!r} is not a ModelFlags member: members are single flags and ALL. "
            "Combine flags with | (the result is an int that notify_model_changed() accepts), "
            "and use ModelFlags.ALL for every flag."
        )

    @classmethod
    def from_attributes(cls, *names: str) -> int:
        """Return the flags that cover edits of the given model attributes.

        Pass the result to :meth:`~newton.solvers.SolverBase.notify_model_changed`
        after editing these :class:`~newton.Model` arrays. Custom attributes may
        be named with ``":"`` or ``"."`` after their namespace, e.g.
        ``"mujoco:gravcomp"`` or ``"mujoco.gravcomp"``.

        Example:

        .. code-block:: python

            model.body_mass.assign(masses)
            solver.notify_model_changed(newton.ModelFlags.from_attributes("body_mass"))

        Args:
            names: Model attribute names.

        Returns:
            Bit mask of :class:`ModelFlags`. Each attribute maps to its narrowest
            category, e.g. ``joint_friction`` to :attr:`JOINT_DOF_FORCE_PROPERTIES`
            rather than :attr:`JOINT_DOF_PROPERTIES`. Attributes without a
            documented category map to :attr:`ALL`.
        """
        flags = 0
        for name in names:
            flags |= int(_MODEL_FLAGS_BY_ATTRIBUTE.get(name.replace(".", ":", 1), cls.ALL))
        return flags


_JOINT_DOF_SUBCATEGORIES = int(
    ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    | ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
    | ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES
)


def _covered_model_flags(flags: int) -> int:
    """Categories a ``notify_model_changed(flags)`` call refreshes.

    The broad :attr:`ModelFlags.JOINT_DOF_PROPERTIES` also covers the narrow joint DOF categories.
    """
    flags = int(flags)
    return flags | _JOINT_DOF_SUBCATEGORIES if flags & int(ModelFlags.JOINT_DOF_PROPERTIES) else flags


def _model_flags_by_attribute() -> dict[str, ModelFlags]:
    flags = ModelFlags
    categories = {
        flags.JOINT_PROPERTIES: ("joint_q", "joint_X_p", "joint_X_c", "joint_axis"),
        # DOF attributes outside the narrow categories keep the full joint DOF update.
        flags.JOINT_DOF_PROPERTIES: (
            "joint_qd",
            "joint_f",
            "joint_target_q",
            "joint_target_qd",
            "joint_velocity_limit",
        ),
        flags.JOINT_DOF_FORCE_PROPERTIES: (
            "joint_target_ke",
            "joint_target_kd",
            "joint_target_mode",
            "joint_damping",
            "joint_friction",
            "joint_effort_limit",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_limit_lower",
            "joint_limit_upper",
            "mujoco:solimplimit",
            "mujoco:solreflimit",
            "mujoco:solreflimit_mode",
            "mujoco:limit_margin",
            "mujoco:dof_passive_stiffness",
            "mujoco:solreffriction",
            "mujoco:solimpfriction",
        ),
        flags.JOINT_DOF_INERTIAL_PROPERTIES: ("joint_armature",),
        flags.JOINT_REFERENCE_POSE_PROPERTIES: ("mujoco:dof_ref", "mujoco:dof_springref"),
        flags.BODY_PROPERTIES: ("body_q", "body_qd", "body_flags"),
        flags.BODY_INERTIAL_PROPERTIES: (
            "body_mass",
            "body_inv_mass",
            "body_com",
            "body_inertia",
            "body_inv_inertia",
            "mujoco:gravcomp",
        ),
        flags.SHAPE_PROPERTIES: (
            "shape_transform",
            "shape_scale",
            "shape_collision_radius",
            "shape_margin",
            "shape_gap",
            "shape_material_mu",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_ka",
            "shape_material_kh",
            "shape_material_restitution",
            "shape_material_mu_torsional",
            "shape_material_mu_rolling",
            "mujoco:geom_solimp",
            "mujoco:geom_solmix",
            "mujoco:solref",
            "mujoco:solref_mode",
            "mujoco:pair_solref",
            "mujoco:pair_solreffriction",
            "mujoco:pair_solimp",
            "mujoco:pair_margin",
            "mujoco:pair_gap",
            "mujoco:pair_friction",
        ),
        flags.MODEL_PROPERTIES: ("gravity",),
        flags.CONSTRAINT_PROPERTIES: (
            "joint_mimic_coeffs",
            "constraint_mimic_coef0",
            "constraint_mimic_coef1",
            "constraint_mimic_enabled",
            "mujoco:eq_solref",
            "mujoco:eq_solimp",
            "mujoco:equality_constraint_anchor",
            "mujoco:equality_constraint_relpose",
            "mujoco:equality_constraint_polycoef",
            "mujoco:equality_constraint_torquescale",
            "mujoco:equality_constraint_enabled",
        ),
        flags.TENDON_PROPERTIES: (
            "mujoco:tendon_stiffness",
            "mujoco:tendon_damping",
            "mujoco:tendon_frictionloss",
            "mujoco:tendon_range",
            "mujoco:tendon_margin",
            "mujoco:tendon_solref_limit",
            "mujoco:tendon_solimp_limit",
            "mujoco:tendon_solref_friction",
            "mujoco:tendon_solimp_friction",
            "mujoco:tendon_armature",
            "mujoco:tendon_actuator_force_range",
        ),
        flags.ACTUATOR_PROPERTIES: (
            "mujoco:actuator_gainprm",
            "mujoco:actuator_biasprm",
            "mujoco:actuator_dynprm",
            "mujoco:actuator_ctrlrange",
            "mujoco:actuator_forcerange",
            "mujoco:actuator_actrange",
            "mujoco:actuator_gear",
            "mujoco:actuator_cranklength",
        ),
    }
    return {name: flag for flag, names in categories.items() for name in names}


_MODEL_FLAGS_BY_ATTRIBUTE = _model_flags_by_attribute()


class StateFlags(IntEnum):
    """Flags indicating which state attributes were updated or should be reset.

    These flags are used with :meth:`~newton.solvers.SolverBase.reset` to
    control which parts of the simulation state are reset, and with
    :meth:`~newton.solvers.experimental.coupled.CouplingInterface.coupling_notify_input_state_update`
    to describe which public state inputs a coupler updated.

    .. experimental::

        The interpretation of these flags by
        :class:`~newton.solvers.experimental.coupled.CouplingInterface` may
        change without prior notice.
    """

    NONE = 0
    """Indicates no state attributes were updated."""

    JOINT_Q = 1 << 0
    """Indicates reduced joint position coordinates: ``State.joint_q``."""

    JOINT_QD = 1 << 1
    """Indicates reduced joint velocity coordinates: ``State.joint_qd``."""

    BODY_Q = 1 << 2
    """Indicates maximal body position coordinates: ``State.body_q``."""

    BODY_QD = 1 << 3
    """Indicates maximal body velocity coordinates: ``State.body_qd``."""

    PARTICLE_Q = 1 << 4
    """Indicates particle positions: ``State.particle_q``."""

    PARTICLE_QD = 1 << 5
    """Indicates particle velocities: ``State.particle_qd``."""

    BODY_F = 1 << 6
    """Indicates rigid-body force inputs: ``State.body_f``."""

    PARTICLE_F = 1 << 7
    """Indicates particle force inputs: ``State.particle_f``."""

    JOINT_F = 1 << 8
    """Indicates joint force inputs: ``Control.joint_f`` or solver-local equivalents."""

    BODY = BODY_Q | BODY_QD
    """Indicates rigid-body pose and velocity inputs."""

    PARTICLE = PARTICLE_Q | PARTICLE_QD
    """Indicates particle position and velocity inputs."""

    JOINT = JOINT_Q | JOINT_QD
    """Indicates joint position and velocity inputs."""

    FORCE = BODY_F | PARTICLE_F | JOINT_F
    """Indicates force-input arrays."""

    ALL = BODY | PARTICLE | JOINT | FORCE
    """Indicates all public state and force-input attributes."""


# Body flags
class BodyFlags(IntEnum):
    """
    Per-body dynamic state flags.

    Each finalized model body must store exactly one runtime state flag:
    :attr:`DYNAMIC` or :attr:`KINEMATIC`. Coupled solver views may OR in
    :attr:`PROXY` on view-local ``body_flags`` overrides. :attr:`ALL` is a
    convenience filter mask for APIs such as :func:`newton.eval_fk` and is not
    a valid stored body state.

    .. experimental::

        :attr:`PROXY` and its inclusion in :attr:`ALL` are part of the
        experimental coupled-solver contract and may change without prior
        notice.
    """

    DYNAMIC = 1 << 0
    """Dynamic body that participates in simulation dynamics."""

    KINEMATIC = 1 << 1
    """User-prescribed body that does not respond to applied forces."""

    PROXY = 1 << 2
    """View-local proxy body marker for coupled simulations."""

    ALL = DYNAMIC | KINEMATIC | PROXY
    """Filter bitmask selecting all body types."""


def _warn_joint_type_cable_deprecated() -> None:
    warnings.warn(
        "newton.JointType.CABLE is deprecated in Newton 1.6; use newton.JointType.ROD instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class _DeprecatedJointTypeMeta(EnumMeta):
    def __getattribute__(cls, name: str):
        # Defined members resolve before EnumMeta.__getattr__, so intercept deprecated access here.
        value = super().__getattribute__(name)
        if name == "CABLE":
            _warn_joint_type_cable_deprecated()
        return value

    def __getitem__(cls, name: str):
        value = super().__getitem__(name)
        if name == "CABLE":
            _warn_joint_type_cable_deprecated()
        return value

    def __dir__(cls):
        # EnumMeta.__dir__ omits aliases; keep the preferred ROD name discoverable.
        names = super().__dir__()
        return names if "ROD" in names else [*names, "ROD"]


# Types of joints linking rigid bodies
class JointType(IntEnum, metaclass=_DeprecatedJointTypeMeta):
    """
    Enumeration of joint types supported in Newton.
    """

    PRISMATIC = 0
    """Prismatic joint: allows translation along a single axis (1 DoF)."""

    REVOLUTE = 1
    """Revolute joint: allows rotation about a single axis (1 DoF)."""

    BALL = 2
    """Ball joint: allows rotation about all three axes (3 DoF, quaternion parameterization)."""

    FIXED = 3
    """Fixed joint: locks all relative motion (0 DoF)."""

    FREE = 4
    """Free joint: allows full 6-DoF motion (translation and rotation, 7 coordinates)."""

    DISTANCE = 5
    """Distance joint: keeps two bodies at a distance within its joint limits (6 DoF, 7 coordinates)."""

    D6 = 6
    """6-DoF joint: Generic joint with up to 3 translational and 3 rotational degrees of freedom."""

    # Keep CABLE as the canonical enum name throughout its 1.6 deprecation.
    CABLE = 7
    """Deprecated name for :attr:`ROD`.

    .. deprecated:: 1.6
        Use :attr:`ROD` instead.
    """

    ROD = CABLE
    """Rod joint: four VBD material slots for stretch, shear, bend, and twist."""

    def dof_count(self, num_axes: int) -> tuple[int, int]:
        """
        Returns the number of degrees of freedom (DoF) in velocity and the number of coordinates
        in position for this joint type.

        Args:
            num_axes: The number of axes for the joint.

        Returns:
            tuple[int, int]: A tuple (dof_count, coord_count) where:
                - dof_count: Number of velocity degrees of freedom for the joint.
                - coord_count: Number of position coordinates for the joint.

        Notes:
            - For PRISMATIC and REVOLUTE joints, both values are 1 (single axis).
            - For BALL joints, dof_count is 3 (angular velocity), coord_count is 4 (quaternion).
            - For FREE and DISTANCE joints, dof_count is 6 (3 translation + 3 rotation), coord_count is 7 (3 position + 4 quaternion).
            - For FIXED joints, both values are 0.
        """
        dof_count = num_axes
        coord_count = num_axes
        if self == JointType.BALL:
            dof_count = 3
            coord_count = 4
        elif self == JointType.FREE or self == JointType.DISTANCE:
            dof_count = 6
            coord_count = 7
        elif self == JointType.FIXED:
            dof_count = 0
            coord_count = 0
        return dof_count, coord_count

    def constraint_count(self, num_axes: int) -> int:
        """
        Returns the number of velocity-level bilateral kinematic constraints for this joint type.

        Args:
            num_axes: The number of DoF axes for the joint.

        Returns:
            int: The number of bilateral kinematic constraints for the joint.

        Notes:
            - For PRISMATIC and REVOLUTE joints, this equals 5 (single DoF axis).
            - For FREE and DISTANCE joints, `cts_count = 0` since it yields no constraints.
            - For FIXED joints, `cts_count = 6` since it fully constrains the associated bodies.
        """
        cts_count = 6 - num_axes
        if self == JointType.BALL:
            cts_count = 3
        elif self == JointType.FREE or self == JointType.DISTANCE:
            cts_count = 0
        elif self == JointType.FIXED:
            cts_count = 6
        return cts_count


class JointTargetMode(IntEnum):
    """
    Enumeration of actuator modes for joint degrees of freedom.

    This enum manages UsdPhysics compliance by specifying whether joint_target_q/qd
    inputs are active for a given DOF. It determines which actuators are installed when
    using solvers that require explicit actuator definitions (e.g., MuJoCo solver).

    Note:
        MuJoCo general actuators (motor, general, etc.) are handled separately via
        custom attributes with "mujoco:actuator" frequency and control.mujoco.ctrl,
        not through this enum.
    """

    NONE = 0
    """No actuators are installed for this DOF. The joint is passive/unactuated."""

    POSITION = 1
    """Only a position actuator is installed for this DOF. Tracks joint_target_q."""

    VELOCITY = 2
    """Only a velocity actuator is installed for this DOF. Tracks joint_target_qd."""

    POSITION_VELOCITY = 3
    """Both position and velocity actuators are installed. Tracks both joint_target_q and joint_target_qd."""

    EFFORT = 4
    """A drive is applied but no gains are configured. No MuJoCo actuator is created for this DOF.
    The user is expected to supply force via joint_f."""

    @staticmethod
    def from_gains(
        target_ke: float,
        target_kd: float,
        force_position_velocity: bool = False,
        has_drive: bool = False,
    ) -> "JointTargetMode":
        """Infer actuator mode from position and velocity gains.

        Args:
            target_ke: Position gain (stiffness).
            target_kd: Velocity gain (damping).
            force_position_velocity: If True and both gains are non-zero,
                forces POSITION_VELOCITY mode instead of just POSITION.
            has_drive: If True, a drive/actuator is applied to the joint.
                When True but both gains are 0, returns EFFORT mode.
                When False, returns NONE regardless of gains.

        Returns:
            The inferred JointTargetMode based on which gains are non-zero:
            - NONE: No drive applied
            - EFFORT: Drive applied but both gains are 0 (direct torque control)
            - POSITION: Only position gain is non-zero
            - VELOCITY: Only velocity gain is non-zero
            - POSITION_VELOCITY: Both gains non-zero (or forced)
        """
        if not has_drive:
            return JointTargetMode.NONE

        if force_position_velocity and (target_ke != 0.0 and target_kd != 0.0):
            return JointTargetMode.POSITION_VELOCITY
        elif target_ke != 0.0:
            return JointTargetMode.POSITION
        elif target_kd != 0.0:
            return JointTargetMode.VELOCITY
        else:
            return JointTargetMode.EFFORT


__all__ = [
    "BodyFlags",
    "JointTargetMode",
    "JointType",
    "ModelFlags",
    "StateFlags",
]
