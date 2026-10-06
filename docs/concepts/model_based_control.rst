.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _model-based-control:

Model-Based Control with Newton
===============================

A model-based controller evaluates kinematics and dynamics at configurations
other than the simulated one. This page lists the Newton functions that do
this, how to call them without changing the simulation, and how Newton's joint
coordinates relate to MuJoCo's ``qpos`` and ``qvel``, for use of MuJoCo's own
functions on the model that :class:`~newton.solvers.SolverMuJoCo` compiles.

.. testsetup:: model-based-control

    import mujoco

    MJCF = """<mujoco>
      <worldbody>
        <body name="torso" pos="0 0 1">
          <freejoint/>
          <inertial pos="0.05 -0.02 0.1" mass="4" diaginertia="0.2 0.3 0.25"/>
          <body name="arm" pos="0.2 0.1 0">
            <joint name="shoulder" type="ball"/>
            <inertial pos="0 0.15 0" mass="1" diaginertia="0.02 0.01 0.02"/>
            <body name="forearm" pos="0 0.3 0">
              <joint name="elbow" type="hinge" axis="1 0 0" armature="0.01"/>
              <inertial pos="0 0.1 0.02" mass="0.6" diaginertia="0.01 0.005 0.01"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>"""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(MJCF)
    model = builder.finalize(device="cpu")
    rng = np.random.default_rng(0)
    q = model.joint_q.numpy().astype(np.float64)
    q[3:7] = np.array([0.1, 0.2, 0.3, 0.9]) / np.linalg.norm([0.1, 0.2, 0.3, 0.9])
    q[7:11] = np.array([0.3, -0.1, 0.2, 0.9]) / np.linalg.norm([0.3, -0.1, 0.2, 0.9])
    q[11] = 0.4
    qd = rng.normal(size=model.joint_dof_count)


Kinematics and dynamics functions
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - API
     - Computes
   * - :func:`newton.eval_fk`
     - Body poses :attr:`State.body_q` and velocities :attr:`State.body_qd`
       (COM linear velocity [m/s], angular velocity [rad/s], world frame)
       from ``joint_q`` and ``joint_qd``.
   * - :func:`newton.eval_ik`
     - ``joint_q`` and ``joint_qd`` from body poses and velocities.
   * - :func:`newton.eval_jacobian`
     - Per articulation, shape ``(articulation_count, 6 * max_joints_per_articulation,
       max_dofs_per_articulation)``. Rows ``6 * i`` to ``6 * i + 5`` map the
       articulation's ``joint_qd`` to the :attr:`State.body_qd` of the child body
       of its ``i``-th joint.
   * - :func:`newton.eval_mass_matrix`
     - Joint-space mass matrix per articulation, shape ``(articulation_count,
       max_dofs_per_articulation, max_dofs_per_articulation)``.
   * - :func:`newton.eval_inverse_dynamics_passive`
     - Any of the mass matrix ``M(q)``, the gravity force ``g(q)``, and the
       Coriolis and centrifugal force ``C(q, qd) qd`` [N or N·m].
   * - :func:`newton.eval_inverse_dynamics_force`
     - ``tau = M(q) qdd + C(q, qd) qd + g(q)`` [N or N·m] in the layout of
       :attr:`Control.joint_f`.
   * - :class:`newton.ik.IKSolver`
     - Joint coordinates that meet position, rotation, and joint-limit
       objectives, for many problems in one call.
   * - :class:`newton.selection.ArticulationView`
     - Articulations selected by label pattern, with per-world views of their
       joint, link, and root arrays and masked forms of the functions above.
   * - :mod:`newton.controllers`
     - Joint-impedance, differential-IK, and operational-space control laws
       (experimental).

:doc:`articulations` defines the ``joint_q`` and ``joint_qd`` layouts and the
inverse-dynamics terms.

Evaluating a configuration without changing the simulation
-----------------------------------------------------------

These functions read coordinates from the arrays and the :class:`~newton.State`
passed to them and write only their outputs. :meth:`Model.state` returns copies
of the model's arrays, so evaluating such a scratch state leaves
:attr:`Model.joint_q`, the simulated states, and the solver unchanged:

.. testcode:: model-based-control

    scratch = model.state()
    scratch.joint_q.assign(q)  # any configuration [m or rad]
    scratch.joint_qd.assign(qd)
    newton.eval_fk(model, scratch.joint_q, scratch.joint_qd, scratch)

    J = newton.eval_jacobian(model, scratch)
    M = newton.eval_mass_matrix(model, scratch)
    gravity_force = wp.empty_like(scratch.joint_qd)
    newton.eval_inverse_dynamics_passive(model, scratch, gravity_force=gravity_force)

- :func:`~newton.eval_jacobian`, :func:`~newton.eval_mass_matrix`, and the
  inverse-dynamics functions read ``state.joint_q`` and ``state.body_q`` (the
  Coriolis force also ``state.joint_qd``); ``body_q`` must come from
  :func:`~newton.eval_fk` of the same ``joint_q``.
- :func:`~newton.eval_fk` called with the model as its target writes
  :attr:`Model.body_q` and :attr:`Model.body_qd`.
- The functions run on ``model.device``. A model finalized with
  ``device="cpu"`` evaluates in host memory.
- Each articulation is evaluated independently: a model holding ``N`` copies
  of a robot (:meth:`ModelBuilder.replicate`) evaluates ``N`` configurations
  per call.
- The mass matrix and inverse dynamics do not include
  :attr:`Model.joint_armature`. MuJoCo includes armature (``dof_armature``) in
  its mass matrix.
- The inverse-dynamics functions are experimental and consider only the
  kinematic tree; loop closures do not contribute.

.. _mujoco-joint-coordinates:

Newton and MuJoCo coordinates
-----------------------------

:class:`~newton.solvers.SolverMuJoCo` compiles the model into a MuJoCo model of
one world, ``solver.mj_model``. Its ``qpos`` and ``qvel`` differ from
:attr:`State.joint_q` and :attr:`State.joint_qd` per joint type:

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - Joint
     - Newton ``joint_q`` / ``joint_qd``
     - MuJoCo ``qpos`` / ``qvel``
   * - FREE
     - Position [m] and ``(x, y, z, w)`` quaternion of the transform from
       the joint's parent anchor frame (:attr:`Model.joint_X_p`) to its child
       anchor frame (:attr:`Model.joint_X_c`); the body's world pose when both
       are identity.
       Linear velocity of the body's COM [m/s] and angular velocity [rad/s],
       both in the joint's parent frame (the world frame when
       :attr:`Model.joint_X_p` is identity).
     - World position [m] of the body origin and ``(w, x, y, z)`` quaternion.
       Linear velocity of the body origin [m/s] in the world frame, then the
       angular velocity [rad/s] in the body frame.
   * - BALL
     - ``(x, y, z, w)`` quaternion; angular velocity [rad/s] in the joint's
       parent anchor frame.
     - ``(w, x, y, z)`` quaternion; angular velocity [rad/s] in the child body
       frame.
   * - REVOLUTE, PRISMATIC, D6 axes
     - One coordinate per axis [m or rad], relative to the authored pose;
       a D6 joint lists its linear axes before its angular axes.
     - One MuJoCo joint per axis. ``qpos = joint_q + ref``, where ``ref`` is
       the ``mujoco:dof_ref`` custom attribute (see
       :ref:`MuJoCo joint reference values <mujoco-joint-ref>`);
       ``qvel = joint_qd``.
   * - FIXED
     - No coordinates.
     - No coordinates; a fixed root becomes a mocap body.

With ``R`` the body's world rotation and ``r`` its COM offset
(:attr:`Model.body_com`, MuJoCo ``body_ipos``), the FREE-joint velocities
relate by ``v_origin = v_com - omega × (R r)`` and
``omega_body = R^T omega_world`` (see :ref:`MuJoCo conversion <MuJoCo conversion>`).

Coordinate order:

- ``joint_q`` follows the model's joint order. MuJoCo orders ``qpos`` by its
  depth-first body tree. :class:`~newton.solvers.SolverMuJoCo` warns at
  construction when the joint order is not depth-first; the two orders then
  differ.
- ``solver.mj_model`` is compiled from the Newton model, not from the source
  file, and its names derive from Newton labels. An MJCF body with several
  joints imports as one D6 joint, so a body with ``<joint type="hinge"/>``
  followed by ``<joint type="slide"/>`` has ``qpos`` order (hinge, slide) in a
  model MuJoCo compiles from the file and (slide, hinge) in
  ``solver.mj_model``.
- Index maps per MuJoCo world: ``solver.mjc_body_to_newton``,
  ``solver.mjc_jnt_to_newton_jnt``, ``solver.mjc_dof_to_newton_dof``, and,
  per actuator, ``solver.mjc_actuator_to_newton_idx``.

Converting states
~~~~~~~~~~~~~~~~~

:meth:`~newton.solvers.SolverMuJoCo.convert_joint_coords_to_mujoco` and
:meth:`~newton.solvers.SolverMuJoCo.convert_joint_coords_from_mujoco` apply the
solver's conversion in float64 on the host, to any number of states, without
reading or writing the solver's MuJoCo data:

.. testcode:: model-based-control

    planner = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
    data = mujoco.MjData(planner.mj_model)
    data.qpos[:], data.qvel[:] = planner.convert_joint_coords_to_mujoco(q, qd)
    mujoco.mj_forward(planner.mj_model, data)

    q_back, qd_back = planner.convert_joint_coords_from_mujoco(data.qpos, data.qvel)
    assert np.allclose(q_back, q) and np.allclose(qd_back, qd)

- Leading array dimensions are batch dimensions; a trajectory of shape
  ``[T, joint_coord_count]`` converts in one call.
- For a model with several worlds, the MuJoCo side has the worlds
  concatenated, shape ``[..., world_count * nq]``;
  :meth:`~newton.solvers.SolverMuJoCo.convert_joint_coords_from_mujoco` also
  accepts ``[..., world_count, nq]``, the layout of ``solver.mjw_data.qpos``.
- Loop-closure joints have no MuJoCo coordinates; converting from MuJoCo
  copies theirs from :attr:`Model.joint_q` and :attr:`Model.joint_qd`.

At a fixed ``joint_q``, the velocity conversion is linear,
``qvel = T joint_qd``; converting the identity matrix gives ``T``. On scalar
joints ``T`` only reorders coordinates. The Newton quantities relate to
MuJoCo's on ``solver.mj_model`` through ``T``: a body's COM Jacobian by
``J = [jacp; jacr] T`` (``mujoco.mj_jacBodyCom``), the mass matrix by
``M = T^T (M_mujoco - diag(dof_armature)) T``, and the gravity force by
``g = T^T qfrc_bias`` at ``qvel = 0``.

.. testcode:: model-based-control

    T = planner.convert_joint_coords_to_mujoco(q, np.eye(model.joint_dof_count))[1].T
    nv = planner.mj_model.nv

    data.qvel[:] = 0.0
    mujoco.mj_forward(planner.mj_model, data)
    M_mujoco = np.zeros((nv, nv))
    mujoco.mj_fullM(planner.mj_model, data, M_mujoco)
    M_mujoco -= np.diag(planner.mj_model.dof_armature)
    assert np.allclose(M.numpy()[0], T.T @ M_mujoco @ T, atol=1e-4)
    assert np.allclose(gravity_force.numpy(), T.T @ data.qfrc_bias, atol=1e-3)

    # Newton body 2 (the forearm) is the child of the articulation's joint 2: Jacobian rows 12 to 17.
    forearm = int(np.flatnonzero(planner.mjc_body_to_newton.numpy()[0] == 2)[0])
    jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
    mujoco.mj_jacBodyCom(planner.mj_model, data, jacp, jacr, forearm)
    assert np.allclose(J.numpy()[0, 12:18], np.vstack([jacp, jacr]) @ T, atol=1e-4)

The MuJoCo model of the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``solver.mj_model`` exists for both backends. With ``use_mujoco_cpu=True``,
  :meth:`~newton.solvers.SolverMuJoCo.step` integrates it with
  ``solver.mj_data``, and :meth:`~newton.solvers.SolverMuJoCo.notify_model_changed`
  updates it. With the MuJoCo Warp backend,
  :meth:`~newton.solvers.SolverMuJoCo.step` integrates ``solver.mjw_model``,
  :meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` updates only
  ``solver.mjw_model``, and ``solver.mj_model`` keeps its values from
  construction.
- ``mj_model.opt.timestep`` is MuJoCo's default of 0.002 s after construction;
  :meth:`~newton.solvers.SolverMuJoCo.step` with ``use_mujoco_cpu=True`` sets
  it to ``dt``.
- Joint-target actuators take ``gainprm`` and ``biasprm`` from
  :attr:`Model.joint_target_ke` and :attr:`Model.joint_target_kd` at
  construction and on
  :meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` with
  :attr:`ModelFlags.JOINT_DOF_FORCE_PROPERTIES` (see
  :ref:`MuJoCo actuators <mujoco-actuators>`). Entry ``i`` of
  ``solver.mjc_actuator_to_newton_idx`` is the Newton DOF (of world 0) of
  actuator ``i``: ``d >= 0`` for a position actuator, ``-(d + 2)`` for a
  velocity actuator.
  The ``ctrl`` of a position actuator is its target in MuJoCo's ``qpos``
  convention, ``joint_target_q + ref``.
- :attr:`Model.joint_effort_limit` of a scalar joint becomes its
  ``jnt_actfrcrange`` and follows
  :meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` with
  :attr:`ModelFlags.JOINT_DOF_FORCE_PROPERTIES`; an authored
  ``mujoco:actuator_forcerange`` sets the actuator's ``forcerange`` at
  construction.
- With ``use_mujoco_cpu=True`` and no applied forces, ``mujoco.mj_step`` on
  an ``MjData`` of ``solver.mj_model`` whose ``qpos``, ``qvel``, and ``ctrl``
  are converted as above gives the next state of
  :meth:`~newton.solvers.SolverMuJoCo.step`.
