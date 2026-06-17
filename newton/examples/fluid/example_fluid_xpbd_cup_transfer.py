# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A Franka arm endlessly carries a transparent cup of XPBD water between two
spots on the floor (A -> B -> A ...). The arm is driven kinematically by IK
waypoints; the cup is attached to the gripper while carried, and the water is
a live XPBD position-based fluid colliding with the cup mesh through the
standard particle-shape contact pipeline. A GUI slider scales the robot
speed: crank it up and the water's inertia makes it slosh over the rim."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# Cache the cooked cup SDF on disk so repeated runs skip the voxelization.
_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_cup_transfer_sdf"


@wp.kernel
def set_cup_state(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_index: int,
    xform: wp.transform,
    linear_velocity: wp.vec3,
):
    body_q[body_index] = xform
    body_qd[body_index] = wp.spatial_vector(linear_velocity, wp.vec3(0.0))


@wp.kernel
def interp_cup_state(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_index: int,
    old_pos: wp.array[wp.vec3],
    new_pos: wp.array[wp.vec3],
    linear_velocity: wp.array[wp.vec3],
    alpha: float,
):
    # Device-side per-substep cup pose so the whole substep loop is CUDA-graph
    # capturable: the host writes old/new/velocity once per frame and the graph
    # interpolates with a constant ``alpha`` baked into each unrolled substep.
    p = old_pos[0] + alpha * (new_pos[0] - old_pos[0])
    body_q[body_index] = wp.transform(p, wp.quat_identity())
    body_qd[body_index] = wp.spatial_vector(linear_velocity[0], wp.vec3(0.0))


class Example:
    # phase schedule: (name, duration [s] at speed 1)
    PHASES = (
        ("move_above_pick", 1.6),
        ("descend_pick", 1.0),
        ("grasp", 0.6),
        ("lift", 1.0),
        ("traverse", 2.0),
        ("descend_place", 1.0),
        ("release", 0.6),
        ("ascend", 1.0),
    )

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.wall_thickness = 0.010
        # Adaptive substeps: the cup's peak speed grows with the robot speed, so
        # refine the timestep as the user speeds up to keep the moving wall from
        # sweeping past the water in one step (see _substeps_for_speed).
        self.base_substeps = args.substeps
        self.speed = args.speed
        self.sim_substeps = self._substeps_for_speed(self.speed)
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.spot_a = np.array(args.spot_a, dtype=np.float32)
        self.spot_b = np.array(args.spot_b, dtype=np.float32)
        self.lift_height = args.lift_height
        # IK link 11 is effectively the grasp frame between the fingertips
        self.fingertip_offset = 0.015
        self.rim_depth = 0.02
        self.grasp_radius = args.cup_inner_radius + 0.004

        self.cup_inner_radius = args.cup_inner_radius
        self.cup_height = args.cup_height
        wall_thickness = self.wall_thickness

        # IK runs on a robot-only model
        urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        ik_builder = newton.ModelBuilder()
        ik_builder.add_urdf(urdf, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), enable_self_collisions=False)
        self.ik_model = ik_builder.finalize()

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.default_particle_radius = args.radius
        builder.add_urdf(urdf, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), enable_self_collisions=False)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.4), color=(0.55, 0.56, 0.54))

        self.arm_dofs = 7
        self.robot_coords = self.ik_model.joint_coord_count

        self.cup_body = builder.add_body(
            xform=wp.transform(wp.vec3(self.spot_a[0], self.spot_a[1], 0.0), wp.quat_identity()),
            label="cup",
        )
        cup_mesh = self._build_cup_mesh(self.cup_inner_radius, wall_thickness, self.cup_height)
        # Build an SDF on the cup so the ~100k water particles collide with it via
        # one cheap SDF sample each instead of a per-triangle mesh query.
        cup_mesh.build_sdf(
            max_resolution=args.sdf_resolution,
            narrow_band_range=(-0.03, 0.03),
            margin=0.02,
            cache_dir=_SDF_CACHE_DIR,
        )
        builder.add_shape_mesh(
            self.cup_body,
            mesh=cup_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.3),
            color=(0.75, 0.85, 0.95),
            opacity=args.cup_opacity,
        )

        # the entire robot and the cup are posed kinematically (IK + phase
        # machine); XPBD then only simulates the fluid against their shapes
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)

        # water column inside the cup
        fill = self.cup_inner_radius * 1.35
        dim_xy = max(int(fill / args.spacing), 2)
        builder.add_particle_grid(
            pos=wp.vec3(
                float(self.spot_a[0] - 0.5 * (dim_xy - 1) * args.spacing),
                float(self.spot_a[1] - 0.5 * (dim_xy - 1) * args.spacing),
                1.2 * wall_thickness,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dim_xy,
            dim_y=dim_xy,
            dim_z=args.fill_layers,
            cell_x=args.spacing,
            cell_y=args.spacing,
            cell_z=args.spacing,
            mass=args.rest_density * args.spacing**3,
            jitter=0.0005,
            radius_mean=args.radius,
            flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
        )

        self.model = builder.finalize()
        self._apply_water_velocity_cap()
        # grippy fluid-shape friction so spilled water crawls to a stop instead
        # of sliding far across the floor and aliasing the neighbor grid
        self.model.soft_contact_mu = 0.3
        # roomier hash grid: when the arm spills water at high --speed it spreads
        # well past the cup; the default 128^3 grid would alias far cells onto
        # the cup region and stall the neighbor queries
        with wp.ScopedDevice(self.model.device):
            self.model.particle_grid = wp.HashGrid(256, 256, 256)
            self.model.particle_grid.reserve(self.model.particle_count)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            fluid_rest_distance=args.spacing,
            fluid_cohesion=args.cohesion,
            fluid_viscosity=args.viscosity,
            fluid_relaxation=args.relaxation,
            max_diffuse_particles=args.foam_max_particles,
            diffuse_lifetime=1.5,
            diffuse_threshold=1.0,
            diffuse_spawn_probability=0.5,
            # carrying the cup fast slams water into the wall in over-compressed
            # clumps; capping above the bulk neighbor count bounds the per-warp
            # cost so the frame rate holds up (see the wave-pool example)
            fluid_max_neighbors=args.max_neighbors,
        )

        self._setup_ik()

        # phase machine
        self.phase_index = 0
        self.phase_time = 0.0
        self.pick_spot = self.spot_a.copy()
        self.place_spot = self.spot_b.copy()
        self.cup_attached = False
        self.cup_pos = np.array([self.spot_a[0], self.spot_a[1], 0.0], dtype=np.float32)
        self.ee_pos = self.home_pos.copy()
        self._update_cup_state(self.cup_pos, np.zeros(3, dtype=np.float32))

        self.viewer.set_model(self.model)
        use_fluid_surface = getattr(self.viewer, "fluids", None) is not None
        self.viewer.show_particles = not use_fluid_surface
        if hasattr(self.viewer, "show_fluid"):
            self.viewer.show_fluid = use_fluid_surface
        self.fluid_color = tuple(args.fluid_color)
        self.viewer.set_camera(pos=wp.vec3(args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            sun = np.array((0.5, -0.35, 0.75), dtype=np.float32)
            renderer._sun_direction = sun / np.linalg.norm(sun)
            renderer.sky_upper = (0.55, 0.75, 0.95)
            renderer.sky_lower = (0.78, 0.82, 0.80)
            renderer.ambient_sky = (0.92, 0.96, 1.0)
            renderer.ambient_ground = (0.52, 0.54, 0.56)
            if hasattr(renderer, "exposure"):
                renderer.exposure = 1.1

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self._update_cup_state(self.cup_pos, np.zeros(3, dtype=np.float32))

        # Per-frame cup keyframe (start/end pose + velocity) that the captured
        # substep loop interpolates on device.
        self._cup_old = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self._cup_new = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self._cup_vel = wp.zeros(1, dtype=wp.vec3, device=self.model.device)

        # Replay the whole substep loop (cup pose, collide, fluid solve) from a
        # CUDA graph. Only the IK/eval_fk pose update stays on the host each
        # frame; it writes the cup keyframe arrays the graph reads. Prime the
        # reorder scratch now -- it allocates on first use, which cannot happen
        # inside the capture below.
        self.solver.reorder_particles(self.state_0)
        self.graph = None
        self.use_cuda_graph = wp.get_device(self.model.device).is_cuda

    def _substeps_for_speed(self, speed):
        # The wall-crossing speed (0.5*wall/sim_dt, the most a particle or the
        # wall may move per substep without skipping the contact) scales with the
        # substep count, while the cup's peak speed grows ~linearly with the
        # robot speed. Add substeps as the carry speeds up so the wall stays
        # collision-tight -- a CCD-style timestep refinement -- capped so an
        # extreme slider value cannot stall the frame. 4 substeps already cover
        # the default and 2x carries; faster carries refine further.
        return int(min(16, max(self.base_substeps, np.ceil(2.0 * speed))))

    def _apply_water_velocity_cap(self):
        # Cap the water just under the per-substep wall-crossing speed so a fast
        # carry pushes it up over the rim instead of letting it tunnel the wall.
        self.model.particle_max_velocity = 0.85 * 0.5 * self.wall_thickness / self.sim_dt

    def _set_speed(self, speed):
        """Update the robot speed and, if it changes the substep count, refine the
        timestep and re-capture the graph (the captured loop is unrolled per
        substep, so its count is baked in)."""
        self.speed = speed
        substeps = self._substeps_for_speed(speed)
        if substeps != self.sim_substeps:
            self.sim_substeps = substeps
            self.sim_dt = self.frame_dt / substeps
            self._apply_water_velocity_cap()
            self.graph = None

    @staticmethod
    def _build_cup_mesh(inner_radius, wall_thickness, height, segments=40):
        """Closed solid of revolution: cylindrical cup with an open cavity."""
        ri = inner_radius
        ro = inner_radius + wall_thickness
        t = wall_thickness
        profile = [(ro, 0.0), (ro, height), (ri, height), (ri, t)]
        vertices = []
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            c, sn = np.cos(angle), np.sin(angle)
            for r, z in profile:
                vertices.append((r * c, r * sn, z))
        bottom_center = len(vertices)
        vertices.append((0.0, 0.0, 0.0))
        cavity_center = len(vertices)
        vertices.append((0.0, 0.0, t))

        rows = len(profile)
        indices = []
        for i in range(segments):
            j = (i + 1) % segments
            for k in range(rows - 1):
                a = i * rows + k
                b = i * rows + k + 1
                c0 = j * rows + k
                d = j * rows + k + 1
                indices += [a, c0, b, b, c0, d]
            indices += [i * rows + 0, bottom_center, j * rows + 0]
            indices += [i * rows + rows - 1, j * rows + rows - 1, cavity_center]

        return newton.Mesh(
            np.asarray(vertices, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
        )

    def _setup_ik(self):
        self.ee_index = 11  # fr3 hand link
        ik_state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, ik_state)
        ee_tf = wp.transform(*ik_state.body_q.numpy()[self.ee_index])
        self.ee_rotation = np.array(wp.transform_get_rotation(ee_tf), dtype=np.float32)
        self.home_pos = np.array(wp.transform_get_translation(ee_tf), dtype=np.float32)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*self.home_pos)], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*self.ee_rotation)], dtype=wp.vec4),
        )
        limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
        )
        self.joint_q_ik = wp.clone(self.ik_model.joint_q.reshape((1, -1)))
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    # ------------------------------------------------------------------
    def _phase_targets(self, name):
        """Returns (ee_start, ee_end, gripper_start, gripper_end) for a phase."""
        grasp_z = self.cup_height - self.rim_depth + self.fingertip_offset
        high_z = self.cup_height + self.lift_height
        pick_high = np.array([self.pick_spot[0] + self.grasp_radius, self.pick_spot[1], high_z], np.float32)
        pick_low = np.array([self.pick_spot[0] + self.grasp_radius, self.pick_spot[1], grasp_z], np.float32)
        place_high = np.array([self.place_spot[0] + self.grasp_radius, self.place_spot[1], high_z], np.float32)
        place_low = np.array([self.place_spot[0] + self.grasp_radius, self.place_spot[1], grasp_z], np.float32)
        g_open, g_closed = 0.04, 0.005
        if name == "move_above_pick":
            return self.ee_pos.copy(), pick_high, g_open, g_open
        if name == "descend_pick":
            return pick_high, pick_low, g_open, g_open
        if name == "grasp":
            return pick_low, pick_low, g_open, g_closed
        if name == "lift":
            return pick_low, pick_high, g_closed, g_closed
        if name == "traverse":
            return pick_high, place_high, g_closed, g_closed
        if name == "descend_place":
            return place_high, place_low, g_closed, g_closed
        if name == "release":
            return place_low, place_low, g_closed, g_open
        return place_low, place_high, g_open, g_open  # ascend

    def _update_cup_state(self, position, velocity):
        # eval_fk reposes the cup from its (stale) free-joint coordinates, so
        # both simulation states must be overwritten or the fluid collides
        # against a ghost cup left at the previous location.
        for state in (self.state_0, self.state_1):
            wp.launch(
                set_cup_state,
                dim=1,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    self.cup_body,
                    wp.transform(wp.vec3(*[float(v) for v in position]), wp.quat_identity()),
                    wp.vec3(*[float(v) for v in velocity]),
                ],
                device=self.model.device,
            )

    def _advance_robot(self):
        """Host-side per-frame update: solve IK, pose the arm, and stage the cup
        keyframe arrays the captured substep loop reads. Returns the phase
        progress ``t`` so :meth:`step` can advance the phase machine."""
        name, duration = self.PHASES[self.phase_index]
        self.phase_time += self.frame_dt * self.speed
        t = min(self.phase_time / duration, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)

        ee_start, ee_end, g_start, g_end = self._phase_targets(name)
        ee_target = ee_start * (1.0 - smooth_t) + ee_end * smooth_t
        gripper = g_start * (1.0 - smooth_t) + g_end * smooth_t

        # IK toward the interpolated waypoint
        self.pos_obj.set_target_positions(wp.array([wp.vec3(*[float(v) for v in ee_target])], dtype=wp.vec3))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=12)

        # apply the IK solution kinematically: arm joints + cosmetic fingers
        joint_q = self.model.joint_q.numpy()
        ik_q = self.joint_q_ik.numpy()[0]
        joint_q[: self.arm_dofs] = ik_q[: self.arm_dofs]
        joint_q[self.arm_dofs : self.robot_coords] = gripper
        self.model.joint_q.assign(joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        # carry or rest the cup
        if name == "grasp" and t >= 1.0:
            self.cup_attached = True
        if name == "release" and t >= 0.5:
            self.cup_attached = False
        if self.cup_attached:
            new_pos = np.array(
                [
                    ee_target[0] - self.grasp_radius,
                    ee_target[1],
                    ee_target[2] - self.fingertip_offset - self.cup_height + self.rim_depth,
                ],
                np.float32,
            )
        else:
            new_pos = self.cup_pos.copy()
            new_pos[2] = 0.0
        old_pos = self.cup_pos.copy()
        cup_vel = (new_pos - old_pos) / self.frame_dt
        self.cup_pos = new_pos
        self.ee_pos = ee_target

        # stage the cup keyframe for the device-side per-substep interpolation
        self._cup_old.assign(old_pos.reshape(1, 3))
        self._cup_new.assign(new_pos.reshape(1, 3))
        self._cup_vel.assign(cup_vel.reshape(1, 3))
        return t

    def simulate(self):
        # Re-sort the water into spatial order once per frame so the density
        # solve's neighbor reads stay cache-coherent after the arm churns or
        # spills it (a pure relabel; see SolverXPBD.reorder_particles).
        self.solver.reorder_particles(self.state_0)

        # fluid substeps against the kinematically posed shapes; the cup pose is
        # interpolated per substep -- teleporting it a full frame at once creates
        # penetrations the position solver converts into large velocities,
        # slingshotting particles out of the cup during the lift
        for k in range(self.sim_substeps):
            alpha = float(k + 1) / float(self.sim_substeps)
            wp.launch(
                interp_cup_state,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.cup_body,
                    self._cup_old,
                    self._cup_new,
                    self._cup_vel,
                    alpha,
                ],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        t = self._advance_robot()

        if self.use_cuda_graph:
            if self.graph is None:
                try:
                    with wp.ScopedCapture() as capture:
                        self.simulate()
                    self.graph = capture.graph
                except Exception as exc:
                    warnings.warn(f"CUDA graph capture failed; running uncaptured: {exc}", stacklevel=2)
                    self.use_cuda_graph = False
                    self.graph = None
                    self.simulate()
            else:
                wp.capture_launch(self.graph)
        else:
            self.simulate()

        if t >= 1.0:
            self.phase_index = (self.phase_index + 1) % len(self.PHASES)
            self.phase_time = 0.0
            if self.phase_index == 0:
                self.pick_spot, self.place_spot = self.place_spot, self.pick_spot

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        show_fluid = getattr(self.viewer, "show_fluid", False)
        if show_fluid:
            self.viewer.show_fluid = False
        try:
            self.viewer.log_state(self.state_0)
        finally:
            if show_fluid:
                self.viewer.show_fluid = show_fluid
        # Hide the fluid surface while debugging with raw particles, so toggling
        # "Show Particles" in the GUI leaves only the particles visible.
        if show_fluid and not self.viewer.show_particles:
            self._log_fluid()
        elif self.solver.diffuse_positions is not None:
            # the foam is emitted by _log_fluid; when that is skipped, clear the
            # last foam batch so it doesn't stay frozen on screen
            self.viewer.log_fluid_diffuse("/model/fluid/diffuse", None)
        self.viewer.end_frame()

    def _log_fluid(self):
        self.solver.update_render_particles(self.state_0, smoothing=0.5, anisotropy_scale=0.8)
        self.viewer.log_fluid(
            "/model/fluid",
            self.solver.render_positions,
            radii=self.model.particle_radius,
            radius_scale=1.8,
            color=self.fluid_color,
            blur_radius_world=0.02,
            anisotropy=self.solver.render_anisotropy,
            anisotropy_secondary=self.solver.render_anisotropy_secondary,
            anisotropy_tertiary=self.solver.render_anisotropy_tertiary,
            hidden=False,
        )
        if getattr(self.viewer, "show_fluid_diffuse", False) and self.solver.diffuse_positions is not None:
            self.viewer.log_fluid_diffuse(
                "/model/fluid/diffuse",
                self.solver.diffuse_positions,
                self.solver.diffuse_velocities,
                radius=0.0033,
                color=(0.9, 0.95, 1.0, 1.1),
                motion_blur_scale=3.0,
                lifetime=self.solver.diffuse_lifetime,
                surface_bias=0.025,
                hidden=False,
            )

    def gui(self, ui):
        changed_speed, speed = ui.slider_float("Robot Speed", self.speed, 0.25, 5.0, "%.2f")
        if changed_speed:
            self._set_speed(speed)
        ui.text(f"Crank up the speed to spill the water.  (substeps: {self.sim_substeps})")

        ui.separator()
        ui.text("Fluid properties")
        # These feed the solver as plain Python floats that the captured graph
        # bakes in, so any change must refresh the solver's derived constants and
        # invalidate the graph to force a re-capture on the next step.
        changed = False
        c, self.solver.fluid_relaxation = ui.slider_float("Relaxation", self.solver.fluid_relaxation, 0.05, 1.0, "%.2f")
        changed |= c
        c, self.solver.fluid_cohesion = ui.slider_float("Cohesion", self.solver.fluid_cohesion, 0.0, 1.0, "%.2f")
        changed |= c
        c, self.solver.fluid_viscosity = ui.slider_float("Viscosity", self.solver.fluid_viscosity, 0.0, 1.0, "%.2f")
        changed |= c
        # Keep the cap below the wall-crossing speed or fast water tunnels the cup.
        c, self.model.particle_max_velocity = ui.slider_float(
            "Max speed", self.model.particle_max_velocity, 0.25, 3.0, "%.2f"
        )
        changed |= c
        if changed:
            self.solver._update_fluid_settings()
            self.graph = None

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("XPBD fluid particles contain non-finite positions")
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("Bodies contain non-finite transforms")
        # at default speed the water should still be carried with the cup
        active = (self.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        heights = q[active][:, 2]
        if heights.max() < 0.005:
            raise ValueError("All water ended on the floor; the cup carry failed")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        # ~100k water particles. The cup carries a texture SDF so the water
        # collides via one cheap SDF sample per particle. This is the *base*
        # substep count, used at the default and 2x carries; faster carries scale
        # it up so the moving wall stays collision-tight (see _substeps_for_speed).
        parser.add_argument("--substeps", type=int, default=4)
        # The under-relaxed density correction (see --relaxation) needs several
        # more iterations to converge: with these fine 100k particles too few
        # leaves the water buzzing instead of settling, while too low a relaxation
        # would over-compress it. 8 settles it and keeps the column filled.
        parser.add_argument("--iterations", type=int, default=8)
        # cap fluid neighbors above the bulk (~80) so a hard slosh can't stall a warp
        parser.add_argument("--max-neighbors", type=int, default=128)
        parser.add_argument("--speed", type=float, default=1.0)
        parser.add_argument("--gravity", type=float, default=-9.81)

        parser.add_argument("--spot-a", type=float, nargs=2, default=(0.5, -0.3))
        parser.add_argument("--spot-b", type=float, nargs=2, default=(0.5, 0.3))
        parser.add_argument("--lift-height", type=float, default=0.32)
        parser.add_argument("--cup-inner-radius", type=float, default=0.05)
        parser.add_argument("--cup-height", type=float, default=0.13)
        parser.add_argument("--cup-opacity", type=float, default=0.35)
        parser.add_argument("--sdf-resolution", type=int, default=256, help="Cup SDF grid resolution.")

        # ~100k particles filling the cup. The fill leaves headroom so a slosh
        # has somewhere to go.
        parser.add_argument("--spacing", type=float, default=0.0016)
        parser.add_argument("--radius", type=float, default=0.0008)
        parser.add_argument("--fill-layers", type=int, default=57)
        parser.add_argument("--rest-density", type=float, default=1000.0)
        parser.add_argument("--cohesion", type=float, default=0.6)
        # The summed (standard-PBF) density correction overshoots at full
        # strength while the cup is carried -- the water buzzes at its velocity
        # cap instead of settling. Under-relaxing the density push lets it come
        # to rest; --iterations compensates for the gentler per-iteration push.
        parser.add_argument("--relaxation", type=float, default=0.3)
        # The XSPH viscosity pass is a full extra neighbor sweep per substep
        # (~40% of the frame at 100k); the density + cohesion solve already reads
        # smooth here, so it is off by default. Raise it for thicker, syrupy flow.
        parser.add_argument("--viscosity", type=float, default=0.0)
        # Foam/spray is barely visible for a carried cup but its spawn pass scans
        # every fluid particle each step; off by default. Raise to add spray.
        parser.add_argument("--foam-max-particles", type=int, default=0)
        parser.add_argument("--fluid-color", type=float, nargs=4, default=(0.113, 0.425, 0.55, 0.8))

        parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.45, -0.85, 0.75))
        parser.add_argument("--camera-pitch", type=float, default=-20.0)
        parser.add_argument("--camera-yaw", type=float, default=150.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
