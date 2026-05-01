# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Sequence

import numpy as np
import warp as wp

from ..sim import Contacts, Model, State


def _to_numpy_prefix(world_start: wp.array(dtype=wp.int32) | np.ndarray) -> np.ndarray:
    """Return a host int32 prefix-sum array."""
    if isinstance(world_start, np.ndarray):
        return world_start.astype(np.int32, copy=False)
    return world_start.numpy().astype(np.int32, copy=False)


def _resolve_contact_world(
    shape_world: np.ndarray,
    world_count: int,
    shape0: np.ndarray,
    shape1: np.ndarray,
) -> np.ndarray:
    """Resolve a world id per rigid contact row."""
    world0 = shape_world[shape0]
    world1 = shape_world[shape1]

    resolved = np.where(world0 >= 0, world0, world1)
    all_global = np.logical_and(world0 < 0, world1 < 0)
    if np.any(all_global):
        if world_count != 1:
            raise ValueError("Global-only rigid contacts are only supported for single-world models.")
        resolved = resolved.copy()
        resolved[all_global] = 0

    if np.any(np.logical_and(world0 >= 0, world1 >= 0) & (world0 != world1)):
        raise ValueError("Cross-world rigid contacts are not supported in SensorStepPacket.")

    return resolved.astype(np.int32, copy=False)


def _world_major_order(world_ids: np.ndarray, world_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable world-major permutation and the matching prefix sum."""
    if world_ids.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(world_count + 1, dtype=np.int32)

    order = np.argsort(world_ids, kind="stable")
    counts = np.bincount(world_ids[order], minlength=world_count).astype(np.int32, copy=False)
    world_start = np.zeros(world_count + 1, dtype=np.int32)
    world_start[1:] = np.cumsum(counts, dtype=np.int32)
    return order.astype(np.int32, copy=False), world_start


def _spatial_top_np(force: np.ndarray) -> np.ndarray:
    """Extract linear force from a spatial-vector array returned by Warp."""
    force = np.asarray(force)
    if force.ndim == 1:
        return force[:3]
    return force[:, :3]


@dataclass(frozen=True)
class SensorSceneTables:
    """Static solver-to-sensor lookup tables derived from a model."""

    world_count: int
    shape_body: wp.array(dtype=wp.int32)
    shape_world: wp.array(dtype=wp.int32)
    shape_transform_local: wp.array(dtype=wp.transform)
    shape_type: wp.array(dtype=wp.int32)
    shape_scale: wp.array(dtype=wp.vec3)
    shape_source_ptr: wp.array(dtype=wp.uint64)
    shape_material_ke: wp.array(dtype=wp.float32) | None
    shape_material_kd: wp.array(dtype=wp.float32) | None
    shape_material_kf: wp.array(dtype=wp.float32) | None
    shape_material_ka: wp.array(dtype=wp.float32) | None
    shape_material_mu: wp.array(dtype=wp.float32) | None
    shape_material_mu_torsional: wp.array(dtype=wp.float32) | None
    shape_material_mu_rolling: wp.array(dtype=wp.float32) | None
    body_world: wp.array(dtype=wp.int32) | None


@dataclass(frozen=True)
class SensorCameraQueryBatch:
    """Flat world-major camera query storage for tiled sensing."""

    class Flags(IntFlag):
        NONE = 0
        ACTIVE = 1 << 0

    world_start: wp.array(dtype=wp.int32)
    transform: wp.array(dtype=wp.transformf)
    flags: wp.array(dtype=wp.uint32)
    intrinsics: wp.array(dtype=wp.vec4f) | None = None
    image_size: wp.array(dtype=wp.vec2i) | None = None

    @property
    def query_count(self) -> int:
        return int(self.transform.shape[0])


@dataclass(frozen=True)
class SensorStepPacket:
    """Per-step solver state prepared for downstream sensor runtimes."""

    class ContactFlags(IntFlag):
        NONE = 0
        VALID_BODY_IDS = 1 << 0
        VALID_LOCAL_POINTS = 1 << 1
        VALID_NORMAL = 1 << 2
        VALID_MARGINS = 1 << 3
        VALID_FORCE_WORLD = 1 << 4
        BEST_EFFORT_HANDLE = 1 << 5
        BACKEND_NATIVE = 1 << 8
        BACKEND_MUJOCO = 1 << 9

    scene: SensorSceneTables
    body_q: wp.array(dtype=wp.transform) | None
    body_qd: wp.array(dtype=wp.spatial_vector) | None
    body_qdd: wp.array(dtype=wp.spatial_vector) | None
    particle_q: wp.array(dtype=wp.vec3) | None
    contact_world_start: wp.array(dtype=wp.int32)
    contact_shape0: wp.array(dtype=wp.int32)
    contact_shape1: wp.array(dtype=wp.int32)
    contact_body0: wp.array(dtype=wp.int32)
    contact_body1: wp.array(dtype=wp.int32)
    contact_point0_local: wp.array(dtype=wp.vec3)
    contact_point1_local: wp.array(dtype=wp.vec3)
    contact_normal: wp.array(dtype=wp.vec3)
    contact_margin0: wp.array(dtype=wp.float32)
    contact_margin1: wp.array(dtype=wp.float32)
    contact_force_world: wp.array(dtype=wp.vec3) | None
    contact_handle: wp.array(dtype=wp.uint64)
    contact_flags: wp.array(dtype=wp.uint32)
    camera_queries: SensorCameraQueryBatch | None = None

    @property
    def world_count(self) -> int:
        return self.scene.world_count

    @property
    def contact_count(self) -> int:
        return int(self.contact_shape0.shape[0])


def build_sensor_scene_tables(model: Model) -> SensorSceneTables:
    """Build immutable lookup tables shared across sensor packets."""
    return SensorSceneTables(
        world_count=model.world_count,
        shape_body=model.shape_body,
        shape_world=model.shape_world,
        shape_transform_local=model.shape_transform,
        shape_type=model.shape_type,
        shape_scale=model.shape_scale,
        shape_source_ptr=model.shape_source_ptr,
        shape_material_ke=model.shape_material_ke,
        shape_material_kd=model.shape_material_kd,
        shape_material_kf=model.shape_material_kf,
        shape_material_ka=model.shape_material_ka,
        shape_material_mu=model.shape_material_mu,
        shape_material_mu_torsional=model.shape_material_mu_torsional,
        shape_material_mu_rolling=model.shape_material_mu_rolling,
        body_world=model.body_world,
    )


def build_sensor_camera_query_batch(
    camera_transforms_by_world: Sequence[Sequence[wp.transformf]],
    *,
    device: str | wp.context.Device | None = None,
) -> SensorCameraQueryBatch:
    """Build a flat world-major camera query batch."""
    world_count = len(camera_transforms_by_world)
    world_start = np.zeros(world_count + 1, dtype=np.int32)
    flat_transforms: list[wp.transformf] = []
    flat_flags: list[int] = []

    running = 0
    for world_id, transforms in enumerate(camera_transforms_by_world):
        running += len(transforms)
        world_start[world_id + 1] = running
        for transform in transforms:
            flat_transforms.append(transform)
            flat_flags.append(int(SensorCameraQueryBatch.Flags.ACTIVE))

    return SensorCameraQueryBatch(
        world_start=wp.array(world_start, dtype=wp.int32, device=device),
        transform=wp.array(flat_transforms, dtype=wp.transformf, device=device),
        flags=wp.array(np.asarray(flat_flags, dtype=np.uint32), dtype=wp.uint32, device=device),
    )


def build_sensor_step_packet(
    scene: SensorSceneTables,
    model: Model,
    state: State,
    *,
    contacts: Contacts | None = None,
    camera_queries: SensorCameraQueryBatch | None = None,
    backend: str = "native",
) -> SensorStepPacket:
    """Build the per-step solver packet for downstream sensors."""
    device = model.device
    contact_count = 0
    if contacts is not None:
        contact_count = int(contacts.rigid_contact_count.numpy()[0])

    shape_world_np = model.shape_world.numpy() if model.shape_world is not None else np.zeros(0, dtype=np.int32)
    shape_body_np = model.shape_body.numpy() if model.shape_body is not None else np.zeros(0, dtype=np.int32)

    contact_world_start_np = np.zeros(scene.world_count + 1, dtype=np.int32)
    shape0_np = np.zeros(contact_count, dtype=np.int32)
    shape1_np = np.zeros(contact_count, dtype=np.int32)
    body0_np = np.full(contact_count, -1, dtype=np.int32)
    body1_np = np.full(contact_count, -1, dtype=np.int32)
    point0_np = np.zeros((contact_count, 3), dtype=np.float32)
    point1_np = np.zeros((contact_count, 3), dtype=np.float32)
    normal_np = np.zeros((contact_count, 3), dtype=np.float32)
    margin0_np = np.zeros(contact_count, dtype=np.float32)
    margin1_np = np.zeros(contact_count, dtype=np.float32)
    force_np: np.ndarray | None = None
    handle_np = np.zeros(contact_count, dtype=np.uint64)
    flags_np = np.zeros(contact_count, dtype=np.uint32)

    if contact_count > 0 and contacts is not None:
        shape0_raw = contacts.rigid_contact_shape0.numpy()[:contact_count].astype(np.int32, copy=False)
        shape1_raw = contacts.rigid_contact_shape1.numpy()[:contact_count].astype(np.int32, copy=False)
        world_ids = _resolve_contact_world(shape_world_np, scene.world_count, shape0_raw, shape1_raw)
        order, contact_world_start_np = _world_major_order(world_ids, scene.world_count)

        shape0_np = shape0_raw[order]
        shape1_np = shape1_raw[order]
        body0_np = shape_body_np[shape0_np]
        body1_np = shape_body_np[shape1_np]
        point0_np = contacts.rigid_contact_point0.numpy()[:contact_count][order].astype(np.float32, copy=False)
        point1_np = contacts.rigid_contact_point1.numpy()[:contact_count][order].astype(np.float32, copy=False)
        normal_np = contacts.rigid_contact_normal.numpy()[:contact_count][order].astype(np.float32, copy=False)
        margin0_np = contacts.rigid_contact_margin0.numpy()[:contact_count][order].astype(np.float32, copy=False)
        margin1_np = contacts.rigid_contact_margin1.numpy()[:contact_count][order].astype(np.float32, copy=False)

        base_flags = (
            SensorStepPacket.ContactFlags.VALID_BODY_IDS
            | SensorStepPacket.ContactFlags.VALID_LOCAL_POINTS
            | SensorStepPacket.ContactFlags.VALID_NORMAL
            | SensorStepPacket.ContactFlags.VALID_MARGINS
        )
        if backend == "native":
            base_flags |= SensorStepPacket.ContactFlags.BACKEND_NATIVE
        elif backend == "mujoco":
            base_flags |= SensorStepPacket.ContactFlags.BACKEND_MUJOCO
        flags_np.fill(int(base_flags))

        tids_np = contacts.rigid_contact_tids.numpy()[:contact_count][order].astype(np.int64, copy=False)
        valid_handle = tids_np >= 0
        if np.any(valid_handle):
            handle_np[valid_handle] = tids_np[valid_handle].astype(np.uint64, copy=False)
            flags_np[valid_handle] |= int(SensorStepPacket.ContactFlags.BEST_EFFORT_HANDLE)

        if contacts.force is not None:
            force_np = _spatial_top_np(contacts.force.numpy()[:contact_count][order]).astype(np.float32, copy=False)
            flags_np |= int(SensorStepPacket.ContactFlags.VALID_FORCE_WORLD)
        else:
            rigid_force_np = contacts.rigid_contact_force.numpy()[:contact_count][order].astype(np.float32, copy=False)
            if np.any(rigid_force_np):
                force_np = rigid_force_np
                flags_np |= int(SensorStepPacket.ContactFlags.VALID_FORCE_WORLD)

    return SensorStepPacket(
        scene=scene,
        body_q=state.body_q,
        body_qd=state.body_qd,
        body_qdd=state.body_qdd,
        particle_q=state.particle_q,
        contact_world_start=wp.array(contact_world_start_np, dtype=wp.int32, device=device),
        contact_shape0=wp.array(shape0_np, dtype=wp.int32, device=device),
        contact_shape1=wp.array(shape1_np, dtype=wp.int32, device=device),
        contact_body0=wp.array(body0_np, dtype=wp.int32, device=device),
        contact_body1=wp.array(body1_np, dtype=wp.int32, device=device),
        contact_point0_local=wp.array(point0_np, dtype=wp.vec3, device=device),
        contact_point1_local=wp.array(point1_np, dtype=wp.vec3, device=device),
        contact_normal=wp.array(normal_np, dtype=wp.vec3, device=device),
        contact_margin0=wp.array(margin0_np, dtype=wp.float32, device=device),
        contact_margin1=wp.array(margin1_np, dtype=wp.float32, device=device),
        contact_force_world=None
        if force_np is None
        else wp.array(force_np, dtype=wp.vec3, device=device),
        contact_handle=wp.array(handle_np, dtype=wp.uint64, device=device),
        contact_flags=wp.array(flags_np, dtype=wp.uint32, device=device),
        camera_queries=camera_queries,
    )


def render_tiled_camera_from_packet(
    sensor,
    packet: SensorStepPacket,
    camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
    *,
    color_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
    depth_image: wp.array(dtype=wp.float32, ndim=4) | None = None,
    shape_index_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
    normal_image: wp.array(dtype=wp.vec3f, ndim=4) | None = None,
    albedo_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
    refit_bvh: bool = True,
    clear_data=None,
) -> None:
    """Adapter from ``SensorStepPacket`` to the current tiled-camera runtime."""
    if packet.camera_queries is None:
        raise ValueError("SensorStepPacket is missing camera queries.")

    world_start = _to_numpy_prefix(packet.camera_queries.world_start)
    counts = np.diff(world_start)
    if counts.size != packet.world_count:
        raise ValueError("Camera query world_start does not match the packet world count.")
    if counts.size == 0:
        raise ValueError("SensorStepPacket must contain at least one world.")
    if np.any(counts != counts[0]):
        raise ValueError("Current tiled-camera runtime requires the same camera count in every world.")

    camera_count = int(counts[0])
    flat_transforms = np.asarray(packet.camera_queries.transform.numpy())
    dense: list[list[wp.transformf]] = []
    for camera_id in range(camera_count):
        camera_row: list[wp.transformf] = []
        for world_id in range(packet.world_count):
            index = int(world_start[world_id]) + camera_id
            values = flat_transforms[index]
            camera_row.append(
                wp.transformf(
                    wp.vec3f(*values[:3]),
                    wp.quatf(*values[3:7]),
                )
            )
        dense.append(camera_row)

    camera_transforms = wp.array(dense, dtype=wp.transformf, device=packet.camera_queries.transform.device)

    state = State()
    state.body_q = packet.body_q
    state.particle_q = packet.particle_q

    sensor.update(
        state,
        camera_transforms,
        camera_rays,
        color_image=color_image,
        depth_image=depth_image,
        shape_index_image=shape_index_image,
        normal_image=normal_image,
        albedo_image=albedo_image,
        refit_bvh=refit_bvh,
        clear_data=clear_data,
    )
