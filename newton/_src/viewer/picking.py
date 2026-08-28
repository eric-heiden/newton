# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
import math

import numpy as np
import warp as wp

import newton

from ..geometry import raycast
from .kernels import (
    PickingState,
    apply_particle_picking_force_kernel,
    apply_picking_force_kernel,
    compute_particle_pick_anchor_kernel,
    compute_pick_state_kernel,
    raycast_particle_triangles_kernel,
    update_pick_target_kernel,
)


class Picking:
    """
    Picking system.

    Allows to pick a body in the viewer by right clicking on it and dragging the mouse.
    This can be used to move objects around in the viewer, a typical use case is to check for solver resilience or
    see how well a RL policy is coping with disturbances.
    """

    def __init__(
        self,
        model: newton.Model,
        pick_stiffness: float = 50.0,
        pick_damping: float = 5.0,
        pick_max_acceleration: float = 5.0,
        particle_pick_radius: float = 0.15,
        particle_pick_stiffness: float = 600.0,
        particle_pick_damping: float = 50.0,
        particle_pick_max_acceleration: float = 40.0,
        world_offsets: wp.array[wp.vec3] | None = None,
    ) -> None:
        """
        Initializes the picking system.

        Args:
            model: The model to pick from.
            pick_stiffness: The stiffness that will be used to compute the force applied to the picked body.
            pick_damping: The damping that will be used to compute the force applied to the picked body.
            pick_max_acceleration: Maximum picking acceleration in multiples of g [9.81 m/s^2].
                Clamps both linear and equivalent rotational acceleration to prevent
                runaway divergence on light or low-inertia objects.
            particle_pick_radius: Rest-surface geodesic radius of a picked particle patch [m].
            particle_pick_stiffness: Particle-patch position-error gain [1/s²].
            particle_pick_damping: Particle-patch velocity damping gain [1/s].
            particle_pick_max_acceleration: Maximum particle-patch acceleration in multiples of g [9.81 m/s²].
            world_offsets: Optional warp array of world offsets (dtype=wp.vec3) for multi-world picking support.

        Raises:
            ValueError: If a picking parameter is invalid.
        """
        self._validate_parameters(
            pick_stiffness,
            pick_damping,
            pick_max_acceleration,
            particle_pick_radius,
            particle_pick_stiffness,
            particle_pick_damping,
            particle_pick_max_acceleration,
        )

        self.model = model
        self.pick_stiffness = float(pick_stiffness)
        self.pick_damping = float(pick_damping)
        self.pick_max_acceleration = float(pick_max_acceleration)
        self.particle_pick_radius = float(particle_pick_radius)
        self.particle_pick_stiffness = float(particle_pick_stiffness)
        self.particle_pick_damping = float(particle_pick_damping)
        self.particle_pick_max_acceleration = float(particle_pick_max_acceleration)
        self.world_offsets = world_offsets
        self.visible_worlds_mask: wp.array[int] | None = None

        self.min_dist = None
        self.min_index = None
        self.min_body_index = None
        self.lock = None
        self._contact_points0 = None
        self._contact_points1 = None
        self._debug = False

        self._particle_min_dist = wp.array([1.0e10], dtype=float, device=model.device if model else "cpu")
        self._particle_min_triangle = wp.array([-1], dtype=int, device=model.device if model else "cpu")
        self._particle_lock = wp.array([0], dtype=wp.int32, device=model.device if model else "cpu")

        particle_count = model.particle_count if model else 0
        particle_device = model.device if model else "cpu"
        if model and model.device.is_cuda:
            self.pick_triangle = wp.array([-1], dtype=int, pinned=True, device=model.device)
        else:
            self.pick_triangle = wp.array([-1], dtype=int, device="cpu")
        self.particle_pick_weights = wp.zeros(particle_count, dtype=float, device=particle_device)
        self._particle_pick_weight_square_sum = wp.zeros(1, dtype=float, device=particle_device)
        self._particle_pick_anchor = wp.zeros(1, dtype=wp.vec3, device=particle_device)
        self._particle_pick_velocity = wp.zeros(1, dtype=wp.vec3, device=particle_device)
        self._particle_triangles = None
        self._particle_rest_positions = None
        self._particle_adjacency = None

        # picking state
        if model and model.device.is_cuda:
            self.pick_body = wp.array([-1], dtype=int, pinned=True, device=model.device)
        else:
            self.pick_body = wp.array([-1], dtype=int, device="cpu")

        pick_state_np = np.empty(1, dtype=PickingState.numpy_dtype())
        pick_state_np[0]["pick_stiffness"] = self.pick_stiffness
        pick_state_np[0]["pick_damping"] = self.pick_damping
        pick_state_np[0]["pick_max_acceleration"] = self.pick_max_acceleration
        pick_state_np[0]["particle_pick_stiffness"] = self.particle_pick_stiffness
        pick_state_np[0]["particle_pick_damping"] = self.particle_pick_damping
        pick_state_np[0]["particle_pick_max_acceleration"] = self.particle_pick_max_acceleration
        self.pick_state = wp.array(pick_state_np, dtype=PickingState, device=model.device if model else "cpu", ndim=1)

        self.pick_dist = 0.0
        self.picking_active = False

        self._default_on_mouse_drag = None

        # Pre-compute effective mass per body for picking force clamping.
        # For articulated bodies, use the total articulation mass so that
        # picking a light link (e.g. fingertip) still allows enough force
        # to move the whole chain. Free bodies use their own mass.
        self._pick_effective_mass = self._compute_effective_mass(model)

    @staticmethod
    def _validate_parameters(
        pick_stiffness: float,
        pick_damping: float,
        pick_max_acceleration: float,
        particle_pick_radius: float,
        particle_pick_stiffness: float,
        particle_pick_damping: float,
        particle_pick_max_acceleration: float,
    ) -> None:
        """Validate picking parameters."""
        for name, value in (
            ("stiffness", pick_stiffness),
            ("damping", pick_damping),
            ("maximum acceleration", pick_max_acceleration),
            ("particle stiffness", particle_pick_stiffness),
            ("particle damping", particle_pick_damping),
            ("particle maximum acceleration", particle_pick_max_acceleration),
        ):
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"Picking {name} must be finite and nonnegative.")
        if not math.isfinite(float(particle_pick_radius)) or particle_pick_radius <= 0.0:
            raise ValueError("Particle picking radius must be finite and positive.")

    @staticmethod
    def _build_particle_topology(model: newton.Model | None):
        """Build the rest-surface graph used by smooth particle picking."""
        if model is None or model.tri_count == 0 or model.particle_count == 0:
            return None, None, None

        triangles = np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape((-1, 3))
        rest_positions = np.asarray(model.particle_q.numpy(), dtype=np.float64)
        neighbor_maps: list[dict[int, float]] = [{} for _ in range(model.particle_count)]

        for triangle in triangles:
            for raw_i, raw_j in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                i = int(raw_i)
                j = int(raw_j)
                length = float(np.linalg.norm(rest_positions[i] - rest_positions[j]))
                previous = neighbor_maps[i].get(j)
                if previous is None or length < previous:
                    neighbor_maps[i][j] = length
                    neighbor_maps[j][i] = length

        adjacency = tuple(tuple(neighbors.items()) for neighbors in neighbor_maps)
        return triangles, rest_positions, adjacency

    def _ensure_particle_topology(self) -> None:
        """Build particle topology on the first triangle-surface pick."""
        if self._particle_triangles is None:
            self._particle_triangles, self._particle_rest_positions, self._particle_adjacency = (
                self._build_particle_topology(self.model)
            )

    def configure(
        self,
        *,
        pick_stiffness: float | None = None,
        pick_damping: float | None = None,
        pick_max_acceleration: float | None = None,
        particle_pick_radius: float | None = None,
        particle_pick_stiffness: float | None = None,
        particle_pick_damping: float | None = None,
        particle_pick_max_acceleration: float | None = None,
    ) -> None:
        """Configure rigid-body and smooth particle-patch picking.

        Args:
            pick_stiffness: Position-error gain [1/s²].
            pick_damping: Velocity damping gain [1/s].
            pick_max_acceleration: Maximum acceleration in multiples of g [9.81 m/s²].
            particle_pick_radius: Rest-surface geodesic radius of the particle patch [m].
            particle_pick_stiffness: Particle-patch position-error gain [1/s²].
            particle_pick_damping: Particle-patch velocity damping gain [1/s].
            particle_pick_max_acceleration: Maximum particle-patch acceleration in multiples of g [9.81 m/s²].
        """
        stiffness = self.pick_stiffness if pick_stiffness is None else float(pick_stiffness)
        damping = self.pick_damping if pick_damping is None else float(pick_damping)
        max_acceleration = self.pick_max_acceleration if pick_max_acceleration is None else float(pick_max_acceleration)
        radius = self.particle_pick_radius if particle_pick_radius is None else float(particle_pick_radius)
        particle_stiffness = (
            self.particle_pick_stiffness if particle_pick_stiffness is None else float(particle_pick_stiffness)
        )
        particle_damping = self.particle_pick_damping if particle_pick_damping is None else float(particle_pick_damping)
        particle_max_acceleration = (
            self.particle_pick_max_acceleration
            if particle_pick_max_acceleration is None
            else float(particle_pick_max_acceleration)
        )
        self._validate_parameters(
            stiffness,
            damping,
            max_acceleration,
            radius,
            particle_stiffness,
            particle_damping,
            particle_max_acceleration,
        )

        self.pick_stiffness = stiffness
        self.pick_damping = damping
        self.pick_max_acceleration = max_acceleration
        self.particle_pick_radius = radius
        self.particle_pick_stiffness = particle_stiffness
        self.particle_pick_damping = particle_damping
        self.particle_pick_max_acceleration = particle_max_acceleration
        pick_state = self.pick_state.numpy()
        pick_state[0]["pick_stiffness"] = stiffness
        pick_state[0]["pick_damping"] = damping
        pick_state[0]["pick_max_acceleration"] = max_acceleration
        pick_state[0]["particle_pick_stiffness"] = particle_stiffness
        pick_state[0]["particle_pick_damping"] = particle_damping
        pick_state[0]["particle_pick_max_acceleration"] = particle_max_acceleration
        self.pick_state.assign(pick_state)

    def _apply_picking_force(self, state: newton.State) -> None:
        """
        Applies a force to the picked body.

        Args:
            state: The simulation state.
        """
        if self.model is None:
            return

        if (
            self.model.body_count > 0
            and state.body_q is not None
            and state.body_qd is not None
            and state.body_f is not None
        ):
            # Launch even when inactive so captured graphs retain picking.
            wp.launch(
                kernel=apply_picking_force_kernel,
                dim=1,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    state.body_f,
                    self.pick_body,
                    self.pick_state,
                    self.model.body_flags,
                    self.model.body_com,
                    self.model.body_mass,
                    self.model.body_inv_inertia,
                    self._pick_effective_mass,
                ],
                device=self.model.device,
            )

        if (
            self.model.particle_count > 0
            and state.particle_q is not None
            and state.particle_qd is not None
            and state.particle_f is not None
        ):
            self._particle_pick_anchor.zero_()
            self._particle_pick_velocity.zero_()
            wp.launch(
                kernel=compute_particle_pick_anchor_kernel,
                dim=self.model.particle_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    self.particle_pick_weights,
                    self.pick_triangle,
                ],
                outputs=[self._particle_pick_anchor, self._particle_pick_velocity],
                device=self.model.device,
            )
            wp.launch(
                kernel=apply_particle_picking_force_kernel,
                dim=self.model.particle_count,
                inputs=[
                    state.particle_q,
                    state.particle_f,
                    self.model.particle_mass,
                    self.model.particle_flags,
                    self.particle_pick_weights,
                    self._particle_pick_weight_square_sum,
                    self.pick_triangle,
                    self.pick_state,
                    self._particle_pick_anchor,
                    self._particle_pick_velocity,
                ],
                device=self.model.device,
            )

    @staticmethod
    def _compute_effective_mass(model: newton.Model) -> wp.array[float]:
        """Compute per-body effective mass for picking force clamping.

        For bodies in an articulation, returns the total mass of that
        articulation so that picking a light link still allows enough
        force to move the whole chain.  Free bodies get their own mass.
        """
        if model is None:
            return wp.zeros(1, dtype=float)

        body_mass_np = model.body_mass.numpy()
        effective = body_mass_np.copy()

        if model.joint_count > 0:
            joint_child_np = model.joint_child.numpy()
            joint_art_np = model.joint_articulation.numpy()

            # Map each body to its articulation index (-1 if free)
            body_art = np.full(model.body_count, -1, dtype=np.int32)
            for j in range(model.joint_count):
                child = joint_child_np[j]
                if child >= 0:
                    body_art[child] = joint_art_np[j]

            # Sum mass per articulation
            art_mass = {}
            for b in range(model.body_count):
                a = body_art[b]
                if a >= 0:
                    art_mass[a] = art_mass.get(a, 0.0) + body_mass_np[b]

            # Assign total articulation mass to each body in that articulation
            for b in range(model.body_count):
                a = body_art[b]
                if a >= 0:
                    effective[b] = art_mass[a]

        return wp.array(effective, dtype=float, device=model.device)

    def is_picking(self) -> bool:
        """Checks if picking is active.

        Returns:
            bool: True if picking is active, False otherwise.
        """
        return self.picking_active

    def release(self) -> None:
        """Releases the picking."""
        self.pick_body.fill_(-1)
        self.pick_triangle.fill_(-1)
        self.picking_active = False

    def get_picked_world_index(self) -> int:
        """Return the world containing the current pick, or -1 for the global world."""
        if self.model is None:
            return -1

        picked_body = int(self.pick_body.numpy()[0])
        if picked_body >= 0 and self.model.body_world is not None:
            return int(self.model.body_world.numpy()[picked_body])

        picked_triangle = int(self.pick_triangle.numpy()[0])
        if picked_triangle >= 0 and self._particle_triangles is not None and self.model.particle_world is not None:
            particle = int(self._particle_triangles[picked_triangle, 0])
            return int(self.model.particle_world.numpy()[particle])
        return -1

    def update(self, ray_start: wp.vec3f, ray_dir: wp.vec3f) -> None:
        """
        Updates the picking target.

        This function is used to track the force that needs to be applied to the picked body as the mouse is dragged.

        Args:
            ray_start: The start point of the ray.
            ray_dir: The direction of the ray.
        """
        if not self.is_picking():
            return

        world_offset = wp.vec3(0.0, 0.0, 0.0)
        if self.world_offsets is not None and self.world_offsets.shape[0] > 0:
            world = self.get_picked_world_index()
            if 0 <= world < self.world_offsets.shape[0]:
                offset_np = self.world_offsets.numpy()[world]
                world_offset = wp.vec3(float(offset_np[0]), float(offset_np[1]), float(offset_np[2]))

        wp.launch(
            kernel=update_pick_target_kernel,
            dim=1,
            inputs=[
                ray_start,
                ray_dir,
                world_offset,
                self.pick_state,
            ],
            device=self.model.device,
        )

    @staticmethod
    def _triangle_barycentric(point: np.ndarray, triangle_positions: np.ndarray) -> np.ndarray:
        """Compute robust barycentric coordinates for a point on a triangle."""
        a, b, c = triangle_positions
        edge_ab = b - a
        edge_ac = c - a
        point_a = point - a
        dot_ab_ab = float(np.dot(edge_ab, edge_ab))
        dot_ab_ac = float(np.dot(edge_ab, edge_ac))
        dot_ac_ac = float(np.dot(edge_ac, edge_ac))
        dot_point_ab = float(np.dot(point_a, edge_ab))
        dot_point_ac = float(np.dot(point_a, edge_ac))
        denominator = dot_ab_ab * dot_ac_ac - dot_ab_ac * dot_ab_ac
        if abs(denominator) <= 1.0e-16:
            nearest = int(np.argmin(np.linalg.norm(triangle_positions - point, axis=1)))
            barycentric = np.zeros(3, dtype=np.float64)
            barycentric[nearest] = 1.0
            return barycentric

        bary_b = (dot_ac_ac * dot_point_ab - dot_ab_ac * dot_point_ac) / denominator
        bary_c = (dot_ab_ab * dot_point_ac - dot_ab_ac * dot_point_ab) / denominator
        barycentric = np.array([1.0 - bary_b - bary_c, bary_b, bary_c], dtype=np.float64)
        barycentric = np.clip(barycentric, 0.0, 1.0)
        total = float(np.sum(barycentric))
        return barycentric / total if total > 0.0 else np.array([1.0, 0.0, 0.0])

    def _compute_particle_patch_weights(
        self,
        state: newton.State,
        triangle_index: int,
        hit_point_world: np.ndarray,
    ) -> np.ndarray | None:
        """Compute compact rest-surface geodesic weights around a triangle hit."""
        self._ensure_particle_topology()
        if (
            self._particle_triangles is None
            or self._particle_rest_positions is None
            or self._particle_adjacency is None
            or state.particle_q is None
        ):
            return None

        triangle = self._particle_triangles[triangle_index]
        current_positions = np.asarray(state.particle_q.numpy(), dtype=np.float64)
        barycentric = self._triangle_barycentric(hit_point_world, current_positions[triangle])
        rest_hit = np.sum(self._particle_rest_positions[triangle] * barycentric[:, None], axis=0)

        distances = np.full(self.model.particle_count, np.inf, dtype=np.float64)
        queue: list[tuple[float, int]] = []
        for raw_particle in triangle:
            particle = int(raw_particle)
            distance = float(np.linalg.norm(self._particle_rest_positions[particle] - rest_hit))
            if distance < distances[particle]:
                distances[particle] = distance
                heapq.heappush(queue, (distance, particle))

        radius = self.particle_pick_radius
        while queue:
            distance, particle = heapq.heappop(queue)
            if distance != distances[particle]:
                continue
            if distance > radius:
                break
            for neighbor, edge_length in self._particle_adjacency[particle]:
                neighbor_distance = distance + edge_length
                if neighbor_distance < distances[neighbor] and neighbor_distance <= radius:
                    distances[neighbor] = neighbor_distance
                    heapq.heappush(queue, (neighbor_distance, neighbor))

        inverse_mass = self.model.particle_inv_mass.numpy()
        flags = self.model.particle_flags.numpy()
        active = (inverse_mass > 0.0) & ((flags & int(newton.ParticleFlags.ACTIVE)) != 0)
        support = active & (distances < radius)

        weights = np.zeros(self.model.particle_count, dtype=np.float64)
        if np.any(support):
            normalized_distance = distances[support] / radius
            one_minus_distance = 1.0 - normalized_distance
            # Wendland C2: compact, monotone, and zero-slope at the support edge.
            weights[support] = one_minus_distance**4 * (4.0 * normalized_distance + 1.0)
        else:
            reachable_dynamic = np.flatnonzero(active & np.isfinite(distances))
            if len(reachable_dynamic) == 0:
                return None
            nearest = int(reachable_dynamic[np.argmin(distances[reachable_dynamic])])
            weights[nearest] = 1.0

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return None
        return (weights / weight_sum).astype(np.float32)

    def _pick_particle_triangle(
        self,
        state: newton.State,
        triangle_index: int,
        hit_point_world: wp.vec3f,
    ) -> bool:
        """Initialize a smooth particle-patch pick from a triangle hit."""
        hit_point = np.array([hit_point_world[0], hit_point_world[1], hit_point_world[2]], dtype=np.float64)
        weights = self._compute_particle_patch_weights(state, triangle_index, hit_point)
        if weights is None:
            return False

        current_positions = np.asarray(state.particle_q.numpy(), dtype=np.float64)
        anchor = weights.astype(np.float64) @ current_positions
        weight_square_sum = float(np.dot(weights, weights))
        self.particle_pick_weights.assign(weights)
        self._particle_pick_weight_square_sum.assign([weight_square_sum])
        self.pick_triangle.assign([triangle_index])
        self.pick_body.fill_(-1)

        pick_state = self.pick_state.numpy()
        pick_state[0]["picked_point_local"] = anchor
        pick_state[0]["picked_point_world"] = anchor
        pick_state[0]["picking_target_world"] = anchor
        self.pick_state.assign(pick_state)
        return True

    def pick(self, state: newton.State, ray_start: wp.vec3f, ray_dir: wp.vec3f) -> None:
        """
        Picks the selected geometry and computes the initial state of the picking. I.e. the force that
        will be applied to the picked body.

        Args:
            state: The simulation state.
            ray_start: The start point of the ray.
            ray_dir: The direction of the ray.
        """

        if self.model is None:
            return

        p, d = ray_start, ray_dir

        self.pick_body.fill_(-1)
        self.pick_triangle.fill_(-1)
        self.picking_active = False

        num_geoms = self.model.shape_count

        if self.min_dist is None:
            self.min_dist = wp.array([1.0e10], dtype=float, device=self.model.device)
            self.min_index = wp.array([-1], dtype=int, device=self.model.device)
            self.min_body_index = wp.array([-1], dtype=int, device=self.model.device)
            self.lock = wp.array([0], dtype=wp.int32, device=self.model.device)
        else:
            self.min_dist.fill_(1.0e10)
            self.min_index.fill_(-1)
            self.min_body_index.fill_(-1)
            self.lock.zero_()

        # Get world offsets if available
        shape_world = (
            self.model.shape_world
            if self.model.shape_world is not None
            else wp.array([], dtype=int, device=self.model.device)
        )
        if self.world_offsets is not None:
            world_offsets = self.world_offsets
        else:
            world_offsets = wp.array([], dtype=wp.vec3, device=self.model.device)

        if num_geoms > 0:
            wp.launch(
                kernel=raycast.raycast_kernel,
                dim=num_geoms,
                inputs=[
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_type,
                    self.model.shape_scale,
                    self.model.shape_source_ptr,
                    p,
                    d,
                    self.lock,
                ],
                outputs=[
                    self.min_dist,
                    self.min_index,
                    self.min_body_index,
                    shape_world,
                    world_offsets,
                    self.visible_worlds_mask,
                ],
                device=self.model.device,
            )

        self._particle_min_dist.fill_(1.0e10)
        self._particle_min_triangle.fill_(-1)
        self._particle_lock.zero_()
        if self.model.tri_count > 0 and state.particle_q is not None:
            wp.launch(
                kernel=raycast_particle_triangles_kernel,
                dim=self.model.tri_count,
                inputs=[
                    state.particle_q,
                    self.model.tri_indices,
                    self.model.particle_world,
                    world_offsets,
                    self.visible_worlds_mask,
                    p,
                    d,
                    self._particle_lock,
                ],
                outputs=[self._particle_min_dist, self._particle_min_triangle],
                device=self.model.device,
            )
        wp.synchronize()

        dist = self.min_dist.numpy()[0] if num_geoms > 0 else 1.0e10
        index = self.min_index.numpy()[0]
        body_index = self.min_body_index.numpy()[0]
        particle_dist = self._particle_min_dist.numpy()[0]
        triangle_index = int(self._particle_min_triangle.numpy()[0])

        if particle_dist < dist and triangle_index >= 0:
            self._ensure_particle_topology()
            self.pick_dist = float(particle_dist)
            d = wp.vec3f(d[0], d[1], d[2])
            p = wp.vec3f(p[0], p[1], p[2])
            hit_point_world = p + d * float(particle_dist)

            if world_offsets.shape[0] > 0 and self.model.particle_world is not None:
                triangle = self._particle_triangles[triangle_index]
                world = int(self.model.particle_world.numpy()[int(triangle[0])])
                if 0 <= world < world_offsets.shape[0]:
                    offset = world_offsets.numpy()[world]
                    hit_point_world = wp.vec3f(
                        hit_point_world[0] - offset[0],
                        hit_point_world[1] - offset[1],
                        hit_point_world[2] - offset[2],
                    )

            self.picking_active = self._pick_particle_triangle(state, triangle_index, hit_point_world)
            return

        if dist < 1.0e10 and body_index >= 0:
            self.pick_dist = dist

            # Ensures that the ray direction and start point are vec3f objects
            d = wp.vec3f(d[0], d[1], d[2])
            p = wp.vec3f(p[0], p[1], p[2])
            # world space hit point (in offset coordinate system from raycast)
            hit_point_world = p + d * float(dist)

            # Convert hit point from offset space to physics space
            # The raycast was done with world offsets applied, so we need to remove them
            if world_offsets.shape[0] > 0 and shape_world.shape[0] > 0 and index >= 0:
                world_idx_np = shape_world.numpy()[index] if hasattr(shape_world, "numpy") else shape_world[index]
                if world_idx_np >= 0 and world_idx_np < world_offsets.shape[0]:
                    offset_np = world_offsets.numpy()[world_idx_np]
                    hit_point_world = wp.vec3f(
                        hit_point_world[0] - offset_np[0],
                        hit_point_world[1] - offset_np[1],
                        hit_point_world[2] - offset_np[2],
                    )

            wp.launch(
                kernel=compute_pick_state_kernel,
                dim=1,
                inputs=[state.body_q, self.model.body_flags, body_index, hit_point_world],
                outputs=[self.pick_body, self.pick_state],
                device=self.model.device,
            )
            wp.synchronize()

        self.picking_active = self.pick_body.numpy()[0] >= 0

        if self._debug:
            if dist < 1.0e10:
                print("#" * 80)
                print(f"Hit geom {index} of body {body_index} at distance {dist}")
                print("#" * 80)
