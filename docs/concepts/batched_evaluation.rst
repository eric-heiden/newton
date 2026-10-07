.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton.utils

.. _batched-evaluation:

Batched Evaluation
==================

:class:`BatchRollout` runs many variants of one scene as the worlds of one
model: candidates (parameter sets or controllers), scenarios (perturbed starts,
control schedules, negative controls), replicates, branches from a saved state,
and the sampled rollouts of a planner. It builds the model, the solver, and the
collision pipeline once, steps every world together, and records named probes
into arrays of shape ``[T, world, ...]``. On CUDA devices every frame replays a
CUDA graph, and schedules and probes live in device arrays, so no step
synchronizes with the host.

:func:`compare_trajectories` compares named time series, such as a log and a
replay, per signal.

The examples on this page use one scene: a puck on a floor, pushed along x.

.. testcode:: batched-evaluation

    import numpy as np
    import warp as wp

    import newton
    from newton.utils import BatchRollout, compare_trajectories


    def build_puck():
        builder = newton.ModelBuilder()
        builder.add_ground_plane(label="floor")
        puck = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), label="puck")
        builder.add_shape_box(puck, hx=0.05, hy=0.05, hz=0.05, label="puck_geom")
        return builder


    rollout = BatchRollout(
        build_puck,
        8,
        solver=newton.solvers.SolverXPBD,
        pipeline=newton.CollisionPipeline,
        dt=1.0 / 240.0,
        substeps=4,
        device="cpu",
    )

The batch
---------

``build`` describes one world: a :class:`~newton.ModelBuilder`, or a function
that returns one or returns the :class:`~newton.Model` it finalized. A
script's ``build_model()`` can be passed as it is; values it sets on the Model
after :meth:`~newton.ModelBuilder.finalize` are copied to every world and
listed in :attr:`BatchRollout.copied_attributes`. ``solver`` and ``pipeline``
create the solver and the collision pipeline for the batched model, for example
a solver class or a script's ``make_solver``. A frame is ``substeps`` physics
steps of ``dt``; before every step the rollout clears the state's forces,
applies the control, collides (with a pipeline), and steps the solver.

:attr:`BatchRollout.view` is a :class:`~newton.selection.WorldView` of the
batch. It checks model edits against the solver
(:meth:`~newton.solvers.SolverBase.check_world_values`), so it raises for
attributes the solver shares across worlds or reads only when it is
constructed, and it notifies the solver after a write.
:meth:`BatchRollout.set_state` writes per-world start values and keeps the
state consistent for every solver: body poses follow edited joint coordinates
(:func:`~newton.eval_fk`), and joint coordinates follow edited body poses and
velocities (:func:`~newton.eval_ik`), which solvers that integrate joint
coordinates, such as :class:`~newton.solvers.SolverMuJoCo`, start from.

.. testcode:: batched-evaluation

    rollout.view.set_attribute("shape_material_mu", rollout.model, np.linspace(0.1, 0.8, 8), labels=["puck_geom", "floor"])
    rollout.set_state("joint_qd", [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 8, labels="puck*")
    records = rollout.run(60, record={"puck": ("body_q", "puck")}, every=10)

    print(records["puck"].shape)  # [T, world, rows, 7]
    slide = records["puck"][-1, :, 0, 0]
    assert np.all(np.diff(slide) < 0.0)  # more friction, shorter slide

.. testoutput:: batched-evaluation

    (7, 8, 1, 7)

Running and recording
---------------------

:meth:`BatchRollout.run` steps every world from the current state:

- ``record`` maps names to probes: an attribute of the state or the control,
  optionally with labels (``("body_q", "puck")``), or a function that returns
  a device array with one row per world. Row 0 holds the values at the start
  of the run and row ``i`` the values after ``i * every`` frames;
  :attr:`BatchRollout.record_time` holds their times [s].
- ``control`` holds schedules and control functions. A schedule maps a control
  attribute, optionally with labels, to values of shape ``[F, world, k, ...]``
  (or ``[F, k, ...]`` for the same values in every world); frame ``f`` since
  the last reset applies row ``min(f, F - 1)``. A control function
  ``fn(rollout)`` runs before every physics step; on CUDA devices it is part of
  the captured graph and reads the frame from :attr:`BatchRollout.frame_index`.

Starting states and branches
----------------------------

:meth:`BatchRollout.reset` starts every world from the model's initial state or
from a given state, copied as it is:

- ``reset(saved)`` continues a saved state of the batch world by world;
  ``reset(saved, world=3)`` copies its world 3 into every world;
- ``reset(plant_state, model=plant_model)`` copies a world of another model
  with the same per-world layout into every world, for example the state of a
  running single-world simulation into a planning batch.

The solver's internal buffers, such as warm starts, are cleared, so a
continued state matches an uninterrupted run up to the solver's tolerance.

.. testcode:: batched-evaluation

    saved = rollout.model.state()
    saved.assign(rollout.state)
    rollout.reset(saved, world=0)  # every world continues from world 0
    branched = rollout.run(20, record={"puck": ("body_q", "puck")}, every=20)

Candidates and scenarios
------------------------

:meth:`BatchRollout.evaluate` runs every candidate in every scenario, one case
per world, and returns a table. A ``setup(world, candidate, scenario)``
function sets the case's values through a :class:`BatchRollout.WorldSetup`
(model values, start-state values, constant controls, and per-frame
schedules), and ``score(records, cases)`` returns metrics with one value per
case of a batch. ``passed(metrics, candidate, scenario)`` decides whether a
case passes.

.. testcode:: batched-evaluation

    def setup(world, mu, scenario):
        world.set_model("shape_material_mu", scenario.get("mu", mu), labels=["puck_geom", "floor"])
        world.set_state("joint_qd", [scenario["push"], 0.0, 0.0, 0.0, 0.0, 0.0], labels="puck*")


    def score(records, cases):
        x = records["puck"][:, :, 0, 0]
        return {"slide": x[-1] - x[0]}


    def passed(metrics, mu, scenario):
        if scenario.get("control"):
            return metrics["slide"] > 0.4  # without friction the puck must slide on
        return metrics["slide"] < 0.5


    scenarios = {
        "slow": {"push": 1.0},
        "fast": {"push": 2.5},
        "frictionless": {"push": 1.0, "mu": 0.0, "control": True},  # a negative control
    }
    result = rollout.evaluate(
        {"mu=0.1": 0.1, "mu=0.8": 0.8},
        scenarios,
        frames=30,
        setup=setup,
        record={"puck": ("body_q", "puck")},
        every=10,
        score=score,
        passed=passed,
        worst={"slide": "max"},
    )
    print(result.summary[1]["pass_fraction"], result.summary[0]["failed"])

.. testoutput:: batched-evaluation

    1.0 ['fast']

``print(result)`` shows the batches, one row per case (up to 40 cases), and a
summary per candidate: passed cases, failed scenarios, and per metric its
range or, for metrics named in ``worst``, its worst value and scenario.
:attr:`~BatchRollout.Evaluation.rows`, :attr:`~BatchRollout.Evaluation.summary`,
and :meth:`~BatchRollout.Evaluation.best` give the same data as Python
objects. ``result.compare(previous)`` lists the cases that pass now and failed
before, those that fail now and passed before, and the largest metric changes.

Cases are grouped into batches, which :attr:`~BatchRollout.Evaluation.batches`
lists with the reason for each separate one:

- The rollout runs ``world_count`` cases at a time; unused worlds of the last
  batch run unchanged and are not scored. Every batch starts from the same
  model values and control, so a case's values and schedules do not carry
  over to later batches.
- A setup that sets a model attribute the solver does not take per world (a
  value it reads only when it is constructed, or shares across worlds, for
  example ``mujoco:condim`` or a solver option of
  :class:`~newton.solvers.SolverMuJoCo`) runs in a separate model with the
  values of world 0 of the rollout's model and that value in every world.
  Values equal to the model's are not a change.
- With ``build=``, each candidate's world is ``build(candidate)``, and
  candidates whose builds differ (for example in geometry) run in separate
  models.

A case's writes apply in the order its setup made them, so a later write of an
overlapping selection wins. Model values that setups changed and the control
are restored afterwards. ``initial_state`` and ``initial_control`` (with
``initial_model`` and ``initial_world``) start every case, also in separate
models, from a saved state and control instead of the model's initial state
and the rollout's control.

Comparing trajectories
----------------------

:func:`compare_trajectories` compares named signals of a candidate against a
reference, each with time as its first axis. A candidate may hold one
trajectory per world (``[T, world, ...]``, the layout of the records), and
``candidate_times`` resamples it to the reference times. Per signal, the
result reports the RMSE, the largest error and its time, the first time an
error exceeds the tolerance, and the components with the largest RMSE.
:meth:`TrajectoryComparison.objective` sums the RMSE, one value per world, so
a ``score`` function can return it as a metric of a fit.

.. testcode:: batched-evaluation

    log_time = np.linspace(0.0, 0.5, 11)
    log = {"x": 1.5 * log_time[:, None]}  # a measured puck position [m]

    rollout.reset()
    rollout.view.set_attribute("shape_material_mu", rollout.model, np.linspace(0.0, 0.7, 8), labels=["puck_geom", "floor"])
    rollout.set_state("joint_qd", [[1.5, 0.0, 0.0, 0.0, 0.0, 0.0]] * 8, labels="puck*")
    records = rollout.run(30, record={"x": ("body_q", "puck")}, every=3)
    simulated = {"x": records["x"][:, :, :, 0]}  # [T, world, 1]

    comparison = compare_trajectories(log, simulated, times=log_time, candidate_times=rollout.record_time, tolerance=0.02)
    print(int(np.argmin(comparison.objective())))  # the frictionless world matches the log

.. testoutput:: batched-evaluation

    0
