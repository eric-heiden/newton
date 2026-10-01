# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched Franka cup transport with live XPBD water and device-resident task data.

Run ``python -m newton.examples fluid_cup_transfer --world-count 4``. Worlds
share physical coordinates and are separated only for display. ``actions`` has
shape ``(world_count, 4)``: normalized Cartesian target velocities followed by
gripper closing velocity (maximum 0.25 m/s per Cartesian axis and 0.08 m/s
for the fingers).
``set_actions()`` disables the reference controller;
``step()`` updates persistent observations, rewards, terminated/truncated flags,
and episode counters. ``reset(mask)`` accepts Newton's bool(world_count + 1)
world mask. No training framework is required.

This example deliberately uses a kinematic grasp abstraction: a closed gripper
must reach the cup rim before attachment, and the upright cup follows the
*achieved* FK position. It is not a force-controlled grasp. Robot shapes are
visual-only; water collides with the cup and ground. The action workspace and
speed are bounded for this discrete-contact demonstration; release is allowed
only within 1.5 cm of the floor. Optional automatic
reset happens at the start of the next step, leaving terminal observations
available to a training loop. Disable it with ``--no-auto-reset`` when the
training loop owns reset timing.
"""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.examples.fluid.utils import (
    FluidParticleRenderer,
    build_cup_mesh,
    parse_particle_count,
    resolve_particle_grid,
)


@wp.kernel
def _reference_actions(
    enabled: wp.array[bool],
    steps: wp.array[int],
    targets: wp.array[wp.vec3],
    durations: wp.array[float],
    waypoints: wp.array[wp.vec3],
    dt: float,
    speed: float,
    phases: wp.array[int],
    actions: wp.array2d[float],
):
    world = wp.tid()
    if not enabled[0]:
        return
    time = float(steps[world] + 1) * dt
    phase = int(0)
    for i in range(7):
        if time > durations[i]:
            time -= durations[i]
            phase = i + 1
        else:
            break
    phase = wp.min(phase, 7)
    fraction = wp.clamp(time / durations[phase], 0.0, 1.0)
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    target = wp.lerp(waypoints[phase], waypoints[phase + 1], fraction)
    velocity = (target - targets[world]) / (dt * speed)
    for axis in range(3):
        actions[world, axis] = wp.clamp(velocity[axis], -1.0, 1.0)
    actions[world, 3] = -1.0
    if phase >= 2 and phase <= 5:
        actions[world, 3] = 1.0
    phases[world] = phase


@wp.kernel
def _apply_actions(
    actions: wp.array2d[float],
    dt: float,
    speed: float,
    targets: wp.array[wp.vec3],
    aperture: wp.array[float],
):
    world = wp.tid()
    target = targets[world]
    for axis in range(3):
        value = actions[world, axis]
        if not wp.isfinite(value):
            value = 0.0
        target[axis] += dt * speed * wp.clamp(value, -1.0, 1.0)
    targets[world] = wp.vec3(
        wp.clamp(target[0], 0.25, 0.70), wp.clamp(target[1], -0.40, 0.40), wp.clamp(target[2], 0.12, 0.65)
    )
    command = actions[world, 3]
    if not wp.isfinite(command):
        command = 0.0
    aperture[world] = wp.clamp(aperture[world] - dt * 0.08 * wp.clamp(command, -1.0, 1.0), 0.005, 0.04)


@wp.kernel
def _apply_joints(
    joint_q_ik: wp.array2d[float], aperture: wp.array[float], coords_per_world: int, joint_q: wp.array[float]
):
    world, coord = wp.tid()
    value = joint_q_ik[world, coord]
    if coord >= 7:
        value = aperture[world]
    joint_q[world * coords_per_world + coord] = value


@wp.kernel
def _update_attachment(
    body_q: wp.array[wp.transform],
    bodies_per_world: int,
    ee_index: int,
    grasp_offset: wp.vec3,
    aperture: wp.array[float],
    dt: float,
    attached: wp.array[bool],
    lifted: wp.array[bool],
    old_pos: wp.array[wp.vec3],
    cup_pos: wp.array[wp.vec3],
    cup_velocity: wp.array[wp.vec3],
):
    world = wp.tid()
    position = cup_pos[world]
    old_pos[world] = position
    ee = wp.transform_get_translation(body_q[world * bodies_per_world + ee_index])
    if not attached[world] and aperture[world] < 0.012 and wp.length(ee - position - grasp_offset) < 0.015:
        attached[world] = True
    # Release is allowed only close to the floor: this abstraction has no
    # dynamic cup solver to model a dropped container.
    if attached[world] and aperture[world] > 0.025 and position[2] < 0.015:
        attached[world] = False
    if attached[world]:
        position = ee - grasp_offset
        position[2] = wp.max(position[2], 0.0)
    elif position[2] < 0.015:
        position[2] = 0.0
    if position[2] > 0.10:
        lifted[world] = True
    cup_pos[world] = position
    cup_velocity[world] = (position - old_pos[world]) / dt


@wp.kernel
def _pose_cups(
    bodies_per_world: int,
    cup_body: int,
    old_pos: wp.array[wp.vec3],
    new_pos: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    alpha: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    world = wp.tid()
    body = world * bodies_per_world + cup_body
    body_q[body] = wp.transform(wp.lerp(old_pos[world], new_pos[world], alpha), wp.quat_identity())
    body_qd[body] = wp.spatial_vector(velocity[world], wp.vec3(0.0))


@wp.kernel
def _count_retained(
    positions: wp.array[wp.vec3],
    particle_world: wp.array[int],
    cup_pos: wp.array[wp.vec3],
    radius: float,
    height: float,
    floor: float,
    particle_radius: float,
    retained: wp.array[int],
):
    particle = wp.tid()
    world = particle_world[particle]
    local = positions[particle] - cup_pos[world]
    if (
        wp.length(wp.vec2(local[0], local[1])) < radius + particle_radius
        and local[2] >= floor - particle_radius - 1.0e-4
        and local[2] <= height + particle_radius
    ):
        wp.atomic_add(retained, world, 1)


@wp.kernel
def _observe(
    body_q: wp.array[wp.transform],
    bodies_per_world: int,
    ee_index: int,
    cup_pos: wp.array[wp.vec3],
    cup_velocity: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    goal: wp.vec3,
    aperture: wp.array[float],
    attached: wp.array[bool],
    lifted: wp.array[bool],
    retained: wp.array[int],
    particles_per_world: int,
    steps: wp.array[int],
    max_steps: int,
    dt: float,
    observations: wp.array2d[float],
    rewards: wp.array[float],
    terminated: wp.array[bool],
    truncated: wp.array[bool],
    successes: wp.array[bool],
):
    world = wp.tid()
    ee = wp.transform_get_translation(body_q[world * bodies_per_world + ee_index])
    position = cup_pos[world]
    goal_delta = goal - position
    retention = float(retained[world]) / float(particles_per_world)
    for axis in range(3):
        observations[world, axis] = ee[axis]
        observations[world, axis + 3] = position[axis]
        observations[world, axis + 6] = target[world][axis] - ee[axis]
        observations[world, axis + 9] = goal_delta[axis]
        observations[world, axis + 12] = cup_velocity[world][axis]
    observations[world, 15] = aperture[world] / 0.04
    observations[world, 16] = float(attached[world])
    observations[world, 17] = retention
    observations[world, 18] = float(lifted[world])
    observations[world, 19] = float(steps[world]) / float(max_steps)
    success = lifted[world] and not attached[world] and wp.length(goal_delta) < 0.02 and retention >= 0.95
    successes[world] = success
    failure = steps[world] > 60 and retention < 0.70
    terminated[world] = success or failure
    truncated[world] = steps[world] >= max_steps
    rewards[world] = dt * (retention - 1.0 - 2.0 * wp.length(goal_delta)) + 5.0 * float(success) - 5.0 * float(failure)


@wp.kernel
def _advance_steps(steps: wp.array[int]):
    world = wp.tid()
    steps[world] += 1


@wp.kernel
def _done_mask(terminated: wp.array[bool], truncated: wp.array[bool], mask: wp.array[bool]):
    world = wp.tid()
    mask[world] = terminated[world] or truncated[world]


@wp.kernel
def _reset_particles(
    mask: wp.array[bool],
    particle_world: wp.array[int],
    initial_q: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    index = wp.tid()
    if mask[particle_world[index]]:
        particle_q[index] = initial_q[index]
        particle_qd[index] = wp.vec3(0.0)
        particle_f[index] = wp.vec3(0.0)


@wp.kernel
def _reset_bodies(
    mask: wp.array[bool],
    body_world: wp.array[int],
    initial_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    if mask[body_world[index]]:
        body_q[index] = initial_q[index]
        body_qd[index] = wp.spatial_vector()
        body_f[index] = wp.spatial_vector()


@wp.kernel
def _reset_tasks(
    mask: wp.array[bool],
    clear_actions: bool,
    home: wp.vec3,
    start: wp.vec3,
    initial_joints: wp.array[float],
    coords_per_world: int,
    joint_q: wp.array[float],
    ik_q: wp.array2d[float],
    targets: wp.array[wp.vec3],
    aperture: wp.array[float],
    attached: wp.array[bool],
    lifted: wp.array[bool],
    old_pos: wp.array[wp.vec3],
    cup_pos: wp.array[wp.vec3],
    cup_velocity: wp.array[wp.vec3],
    steps: wp.array[int],
    episode_count: wp.array[int],
    phase: wp.array[int],
    actions: wp.array2d[float],
):
    world = wp.tid()
    if mask[world]:
        for coord in range(coords_per_world):
            joint_q[world * coords_per_world + coord] = initial_joints[coord]
        for coord in range(ik_q.shape[1]):
            ik_q[world, coord] = initial_joints[coord]
        targets[world] = home
        aperture[world] = 0.04
        attached[world] = False
        lifted[world] = False
        old_pos[world] = start
        cup_pos[world] = start
        cup_velocity[world] = wp.vec3(0.0)
        steps[world] = 0
        episode_count[world] += 1
        phase[world] = 0
        if clear_actions:
            for component in range(4):
                actions[world, component] = 0.0


class Example:
    """Keep the training interface local to this experimental example.

    ``observations`` has 20 columns: EE position [m], cup position [m], target
    minus EE [m], goal minus cup [m], cup velocity [m/s], normalized aperture,
    attachment flag, retained-water fraction, completed-lift flag, and episode
    progress. Running reward is ``frame_dt * (retention - 1 - 2 * goal_distance)``,
    using goal distance [m], with a five-point placement bonus or spill penalty.
    Nonterminal rewards cannot be positive, so postponing release cannot
    accumulate a holding reward. Success requires a lift, placement within
    2 cm, release, and 95% retention; losing 30% terminates after settling.
    All arrays stay on ``model.device`` and retain their addresses across resets.
    Achieved kinematic joints are in ``model.joint_q``; ``state_0.joint_q`` and
    robot-link velocities do not track these prescribed IK updates.
    """

    def __init__(self, viewer, args):
        if args.world_count < 1 or args.substeps < 1 or args.iterations < 1 or args.episode_steps < 1:
            raise ValueError("world count, substeps, iterations, and episode steps must be positive")
        self.viewer = viewer
        self.world_count = args.world_count
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.max_steps = args.episode_steps
        self.auto_reset = args.auto_reset
        self.reference_controller = True
        self.action_speed = 0.25
        self.cup_radius = 0.05
        self.cup_height = 0.13
        self.cup_floor = 0.012
        self.start = wp.vec3(0.5, -0.23, 0.0)
        self.goal = wp.vec3(0.5, 0.23, 0.0)
        self.grasp_offset = wp.vec3(0.054, 0.0, 0.125)
        grid = resolve_particle_grid(args.particle_count, (0.064, 0.064, 0.060), 0.005, minimum=(2, 2, 2))
        self.spacing = grid.spacing

        robot = newton.ModelBuilder()
        robot.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
        )
        # Start within the arm limits, away from the straight-arm singularity.
        robot.joint_q[:9] = [0.0, 0.02, 0.0, -2.37, 0.0, 2.39, 0.785, 0.04, 0.04]
        # Grasping is abstracted; detailed arm triangles must not consume fluid
        # collision work or catch spilled water above the cup.
        robot.shape_flags = [int(newton.ShapeFlags.VISIBLE)] * robot.shape_count
        robot.body_flags = [int(newton.BodyFlags.KINEMATIC)] * robot.body_count
        self.ik_model = robot.finalize()
        self.robot_coords = self.ik_model.joint_coord_count
        self.ee_index = next(i for i, name in enumerate(robot.body_label) if name.endswith("fr3_hand_tcp"))
        ik_state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, ik_state)
        ee = ik_state.body_q.numpy()[self.ee_index]
        self.home = wp.vec3(ee[:3])
        self.ee_rotation = wp.vec4(ee[3:])

        world = newton.ModelBuilder()
        world.add_builder(robot)
        self.cup_body = world.add_body(
            xform=wp.transform(self.start, wp.quat_identity()), is_kinematic=True, label="cup"
        )
        cup_mesh = build_cup_mesh(self.cup_radius, self.cup_floor, self.cup_height, segments=32)
        world.add_shape_mesh(
            self.cup_body,
            mesh=cup_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, mu=0.3),
            color=(0.65, 0.82, 0.95),
            opacity=0.25,
        )
        world.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, mu=0.3))
        for position, color in ((self.start, (0.20, 0.45, 0.25)), (self.goal, (0.25, 0.35, 0.70))):
            world.add_shape_box(
                -1,
                xform=wp.transform(position + wp.vec3(0.0, 0.0, 0.001), wp.quat_identity()),
                hx=0.075,
                hy=0.075,
                hz=0.001,
                cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False, has_shape_collision=False),
                color=color,
            )
        nx, ny, nz = grid.dimensions
        world.add_particle_grid(
            pos=self.start
            + wp.vec3(
                -0.5 * (nx - 1) * grid.spacing, -0.5 * (ny - 1) * grid.spacing, self.cup_floor + grid.radius * 1.1
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=nx,
            dim_y=ny,
            dim_z=nz,
            cell_x=grid.spacing,
            cell_y=grid.spacing,
            cell_z=grid.spacing,
            mass=1000.0 * grid.spacing**3,
            radius_mean=grid.radius,
            jitter=grid.spacing * 0.05,
            flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
        )
        self.particles_per_world = world.particle_count
        self.bodies_per_world = world.body_count
        self.coords_per_world = world.joint_coord_count
        builder = newton.ModelBuilder()
        builder.replicate(world, self.world_count)
        self.model = builder.finalize()
        self.model.particle_max_velocity = 0.85 * self.cup_floor / (2.0 * self.sim_dt)
        self.model.soft_contact_mu = 0.3
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        for state in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        self.initial_state = self.model.state()
        self.initial_state.assign(self.state_0)
        self.initial_joints = wp.array(world.joint_q, dtype=float, device=self.model.device)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=grid.spacing,
            fluid_cohesion=0.6,
            fluid_relaxation=0.6,
            fluid_viscosity=0.0,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self._allocate_task()
        self._setup_ik()
        self.graph = None
        self.reset()
        if self.model.device.is_cuda:
            # Warm kernels and scratch before capture, then restore initial state.
            self._step_device()
            self.reset()
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._step_device()
            self.graph = capture.graph
        self.episode_count.fill_(1)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.particle_renderer = FluidParticleRenderer(self.model)
        self.viewer.set_world_offsets((1.6, 1.5, 0.0))
        camera_scale = max(1.0, np.sqrt(self.world_count) / 2.0)
        self.viewer.set_camera(
            pos=wp.vec3(3.6 * camera_scale, -3.7 * camera_scale, 2.9 * camera_scale), pitch=-28.0, yaw=130.0
        )

    def _allocate_task(self):
        count, device = self.world_count, self.model.device
        self._reference_enabled = wp.array([True], dtype=bool, device=device)
        self.actions = wp.zeros((count, 4), dtype=float, device=device)
        self.observations = wp.zeros((count, 20), dtype=float, device=device)
        self.rewards = wp.zeros(count, dtype=float, device=device)
        self.terminated = wp.zeros(count, dtype=bool, device=device)
        self.truncated = wp.zeros(count, dtype=bool, device=device)
        self.successes = wp.zeros(count, dtype=bool, device=device)
        self.episode_steps = wp.zeros(count, dtype=int, device=device)
        self.episode_count = wp.zeros(count, dtype=int, device=device)
        self.phase = wp.zeros(count, dtype=int, device=device)
        self.attached = wp.zeros(count, dtype=bool, device=device)
        self.lifted = wp.zeros(count, dtype=bool, device=device)
        self.aperture = wp.zeros(count, dtype=float, device=device)
        self.targets = wp.zeros(count, dtype=wp.vec3, device=device)
        self.cup_pos = wp.zeros(count, dtype=wp.vec3, device=device)
        self.cup_old = wp.zeros(count, dtype=wp.vec3, device=device)
        self.cup_velocity = wp.zeros(count, dtype=wp.vec3, device=device)
        self.retained = wp.zeros(count, dtype=int, device=device)
        self._reset_mask = wp.zeros(count + 1, dtype=bool, device=device)
        self._all_worlds = wp.array([True] * count + [False], dtype=bool, device=device)
        pick_low, place_low = self.start + self.grasp_offset, self.goal + self.grasp_offset
        pick_high, place_high = pick_low + wp.vec3(0.0, 0.0, 0.23), place_low + wp.vec3(0.0, 0.0, 0.23)
        self._waypoints = wp.array(
            [self.home, pick_high, pick_low, pick_low, pick_high, place_high, place_low, place_low, place_high],
            dtype=wp.vec3,
            device=device,
        )
        self._durations = wp.array([2.0, 1.6, 0.6, 1.6, 3.2, 1.6, 0.6, 1.6], dtype=float, device=device)

    def _setup_ik(self):
        position = ik.IKObjectivePosition(self.ee_index, wp.vec3(0.0), self.targets)
        rotation = ik.IKObjectiveRotation(
            self.ee_index,
            wp.quat_identity(),
            wp.array([self.ee_rotation] * self.world_count, dtype=wp.vec4, device=self.model.device),
        )
        limits = ik.IKObjectiveJointLimit(self.ik_model.joint_limit_lower, self.ik_model.joint_limit_upper)
        self.joint_q_ik = wp.array(
            np.tile(self.ik_model.joint_q.numpy(), (self.world_count, 1)), dtype=float, device=self.model.device
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=self.world_count,
            objectives=[position, rotation, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def set_actions(self, actions: wp.array):
        """Copy actions and disable the script; clamp inputs and neutralize NaNs.

        Inputs are float32 on the model device, shape ``(world_count, 4)``.
        Components are clipped to [-1, 1]; non-finite values become zero.
        """
        if actions.shape != self.actions.shape or actions.dtype != wp.float32 or actions.device != self.model.device:
            raise ValueError("actions must be float32 (world_count, 4) on the model device")
        wp.copy(self.actions, actions)
        if self.reference_controller:
            self._reference_enabled.fill_(False)
        self.reference_controller = False

    def reset(self, world_mask: wp.array | None = None):
        """Restore selected worlds in place, preserving unselected trajectories."""
        mask = self._all_worlds if world_mask is None else world_mask
        self._reset(mask)
        self._observe()

    def _reset(self, mask, *, clear_actions=True):
        # The solver validates the canonical mask shape, dtype, and device.
        self.solver.reset(self.state_0, world_mask=mask)
        for state in (self.state_0, self.state_1):
            wp.launch(
                _reset_particles,
                self.model.particle_count,
                [
                    mask,
                    self.model.particle_world,
                    self.initial_state.particle_q,
                    state.particle_q,
                    state.particle_qd,
                    state.particle_f,
                ],
                device=self.model.device,
            )
            wp.launch(
                _reset_bodies,
                self.model.body_count,
                [mask, self.model.body_world, self.initial_state.body_q, state.body_q, state.body_qd, state.body_f],
                device=self.model.device,
            )
        wp.launch(
            _reset_tasks,
            self.world_count,
            [
                mask,
                clear_actions,
                self.home,
                self.start,
                self.initial_joints,
                self.coords_per_world,
                self.model.joint_q,
                self.joint_q_ik,
                self.targets,
                self.aperture,
                self.attached,
                self.lifted,
                self.cup_old,
                self.cup_pos,
                self.cup_velocity,
                self.episode_steps,
                self.episode_count,
                self.phase,
                self.actions,
            ],
            device=self.model.device,
        )

    def _step_device(self):
        if self.auto_reset:
            wp.launch(
                _done_mask,
                self.world_count,
                [self.terminated, self.truncated, self._reset_mask],
                device=self.model.device,
            )
            self._reset(self._reset_mask, clear_actions=False)
        wp.launch(
            _reference_actions,
            self.world_count,
            [
                self._reference_enabled,
                self.episode_steps,
                self.targets,
                self._durations,
                self._waypoints,
                self.frame_dt,
                self.action_speed,
                self.phase,
                self.actions,
            ],
            device=self.model.device,
        )
        wp.launch(
            _apply_actions,
            self.world_count,
            [self.actions, self.frame_dt, self.action_speed, self.targets, self.aperture],
            device=self.model.device,
        )
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=6)
        wp.launch(
            _apply_joints,
            (self.world_count, self.robot_coords),
            [self.joint_q_ik, self.aperture, self.coords_per_world, self.model.joint_q],
            device=self.model.device,
        )
        for state in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        wp.launch(
            _update_attachment,
            self.world_count,
            [
                self.state_0.body_q,
                self.bodies_per_world,
                self.ee_index,
                self.grasp_offset,
                self.aperture,
                self.frame_dt,
                self.attached,
                self.lifted,
                self.cup_old,
                self.cup_pos,
                self.cup_velocity,
            ],
            device=self.model.device,
        )
        for substep in range(self.sim_substeps):
            state_in = self.state_0 if substep % 2 == 0 else self.state_1
            state_out = self.state_1 if substep % 2 == 0 else self.state_0
            wp.launch(
                _pose_cups,
                self.world_count,
                [
                    self.bodies_per_world,
                    self.cup_body,
                    self.cup_old,
                    self.cup_pos,
                    self.cup_velocity,
                    float(substep + 1) / self.sim_substeps,
                    state_in.body_q,
                    state_in.body_qd,
                ],
                device=self.model.device,
            )
            state_in.clear_forces()
            self.collision_pipeline.collide(state_in, self.contacts)
            self.solver.step(state_in, state_out, None, self.contacts, self.sim_dt)
        if self.sim_substeps % 2:
            self.state_0.assign(self.state_1)
        wp.launch(_advance_steps, self.world_count, [self.episode_steps], device=self.model.device)
        self._observe()

    def _observe(self):
        self.retained.zero_()
        wp.launch(
            _count_retained,
            self.model.particle_count,
            [
                self.state_0.particle_q,
                self.model.particle_world,
                self.cup_pos,
                self.cup_radius,
                self.cup_height,
                self.cup_floor,
                self.spacing * 0.5,
                self.retained,
            ],
            device=self.model.device,
        )
        wp.launch(
            _observe,
            self.world_count,
            [
                self.state_0.body_q,
                self.bodies_per_world,
                self.ee_index,
                self.cup_pos,
                self.cup_velocity,
                self.targets,
                self.goal,
                self.aperture,
                self.attached,
                self.lifted,
                self.retained,
                self.particles_per_world,
                self.episode_steps,
                self.max_steps,
                self.frame_dt,
                self.observations,
                self.rewards,
                self.terminated,
                self.truncated,
                self.successes,
            ],
            device=self.model.device,
        )

    def step(self):
        if self.graph is None:
            self._step_device()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.particle_renderer.log_state(self.viewer, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if (
            not np.isfinite(self.state_0.particle_q.numpy()).all()
            or not np.isfinite(self.state_0.particle_qd.numpy()).all()
        ):
            raise ValueError("Fluid state is not finite")
        if not np.isfinite(self.observations.numpy()).all():
            raise ValueError("Task observations are not finite")
        if np.any(self.retained.numpy() < 0.95 * self.particles_per_world):
            raise ValueError("More than five percent of the water left a cup")
        if self.reference_controller and not self.auto_reset and np.min(self.episode_steps.numpy()) >= 800:
            if not self.successes.numpy().all():
                raise ValueError("Reference controller did not lift, carry, place, and release every cup")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        parser.add_argument("--solver", choices=("xpbd",), default="xpbd", help="Fluid solver supported by this task.")
        parser.add_argument(
            "--particle-count", type=parse_particle_count, default=1000, help="Target particles per world."
        )
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--iterations", type=int, default=3)
        parser.add_argument("--episode-steps", type=int, default=1000)
        parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
