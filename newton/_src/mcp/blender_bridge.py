# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional Blender render worker for MCP observations.

Newton stays the source of truth: :func:`export_scene` writes one world's visible
shapes once (``scene.json`` + ``geometry.npz`` + texture PNGs), a long-running
Blender process (``blender_server.py``) builds a scene from it, and every render
streams only shape and camera poses over a Unix socket. ``exec`` runs ``bpy``
code in that process for look development (materials, lights, world, exposure).
Blender is found through the ``NEWTON_BLENDER`` environment variable or ``PATH``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import newton

SERVER = Path(__file__).with_name("blender_server.py")


def find_blender() -> str | None:
    """Path of the Blender executable, from ``NEWTON_BLENDER`` or ``PATH``."""
    path = os.environ.get("NEWTON_BLENDER")
    if path and os.access(path, os.X_OK):
        return path
    return shutil.which("blender")


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _primitive_mesh(geo_type: int, scale: np.ndarray) -> newton.Mesh | None:
    z = newton.Axis.Z
    if geo_type == newton.GeoType.SPHERE:
        return newton.Mesh.create_sphere(float(scale[0]), num_latitudes=48, num_longitudes=64, compute_inertia=False)
    if geo_type == newton.GeoType.BOX:
        return newton.Mesh.create_box(*(float(v) for v in scale[:3]), compute_inertia=False)
    if geo_type == newton.GeoType.CAPSULE:
        return newton.Mesh.create_capsule(
            float(scale[0]), float(scale[1]), up_axis=z, segments=48, compute_inertia=False
        )
    if geo_type == newton.GeoType.CYLINDER:
        return newton.Mesh.create_cylinder(
            float(scale[0]), float(scale[1]), up_axis=z, segments=64, compute_inertia=False
        )
    if geo_type == newton.GeoType.CONE:
        return newton.Mesh.create_cone(float(scale[0]), float(scale[1]), up_axis=z, compute_inertia=False)
    if geo_type == newton.GeoType.ELLIPSOID:
        return newton.Mesh.create_ellipsoid(*(float(v) for v in scale[:3]), compute_inertia=False)
    if geo_type == newton.GeoType.PLANE:
        # Infinite planes (non-positive extents) become a 40 m square.
        width = float(scale[0]) if scale[0] > 0 else 20.0
        length = float(scale[1]) if scale[1] > 0 else 20.0
        return newton.Mesh.create_plane(2.0 * width, 2.0 * length, compute_inertia=False)
    return None


def _save_texture(texture, path: Path) -> str | None:
    if texture is None:
        return None
    if isinstance(texture, str):
        return os.path.abspath(texture)
    from PIL import Image

    image = np.asarray(texture)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)
    return path.name


def export_scene(model: newton.Model, directory: str | Path, *, world_id: int = 0) -> dict:
    """Write the visible shapes of one world as a Blender bridge package.

    Returns the scene description (also written as ``scene.json``); its ``shapes`` list
    fixes the order of the per-shape transforms that renders stream.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    types = model.shape_type.numpy()
    flags = model.shape_flags.numpy()
    worlds = model.shape_world.numpy()
    bodies = model.shape_body.numpy()
    transforms = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    colors = model.shape_color.numpy() if model.shape_color is not None else None
    opacities = model.shape_opacity.numpy() if model.shape_opacity is not None else None
    labels = model.shape_label or [f"shape_{i}" for i in range(model.shape_count)]
    visible = int(newton.ShapeFlags.VISIBLE)

    arrays, geometry_keys, shapes, skipped = {}, {}, [], []
    for i in range(model.shape_count):
        if not flags[i] & visible or worlds[i] not in (world_id, -1):
            continue
        if opacities is not None and opacities[i] <= 0.0:
            continue
        geo_type = int(types[i])
        source = model.shape_source[i]
        is_mesh = geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH)
        if is_mesh:
            cache_id = ("mesh", id(source))
            scale = scales[i].tolist()
        else:
            cache_id = (geo_type, tuple(np.round(scales[i], 6).tolist()))
            scale = [1.0, 1.0, 1.0]
        key = geometry_keys.get(cache_id)
        if key is None:
            mesh = source if is_mesh else _primitive_mesh(geo_type, scales[i])
            if mesh is None:
                skipped.append((i, newton.GeoType(geo_type).name))
                continue
            key = f"g{len(geometry_keys)}_{newton.GeoType(geo_type).name.lower()}"
            geometry_keys[cache_id] = key
            arrays[f"{key}/vertices"] = np.asarray(mesh.vertices, np.float32)
            arrays[f"{key}/indices"] = np.asarray(mesh.indices, np.int32).reshape(-1)
            uvs = getattr(mesh, "_uvs", None)
            if uvs is not None and len(uvs) == len(mesh.vertices):
                uvs = np.asarray(uvs, np.float32)
                transform = getattr(mesh, "texture_transform", None)
                if transform is not None and getattr(mesh, "texture", None) is not None:
                    t = np.asarray(transform, np.float32)
                    uvs = uvs @ t[:, :2].T + t[:, 2]
                arrays[f"{key}/uvs"] = uvs
        color = colors[i].tolist() if colors is not None else [0.5, 0.5, 0.5]
        material = {"color": [round(float(c), 5) for c in color], "roughness": 0.5, "metallic": 0.0}
        if is_mesh:
            if getattr(source, "roughness", None) is not None:
                material["roughness"] = float(source.roughness)
            if getattr(source, "metallic", None) is not None:
                material["metallic"] = float(source.metallic)
            texture = _save_texture(getattr(source, "texture", None), directory / f"{key}_texture.png")
            if texture:
                material["texture"] = texture
        if geo_type == newton.GeoType.PLANE:
            material.update(roughness=0.8, checker=True)
        if opacities is not None and opacities[i] < 1.0:
            material["opacity"] = float(opacities[i])
        shapes.append(
            {
                "index": i,
                "name": str(labels[i]).replace("/", "_")[-60:],
                "body": int(bodies[i]),
                "local_transform": transforms[i].tolist(),
                "geometry": key,
                "scale": [float(v) for v in scale],
                "material": material,
            }
        )
    world = shape_world_transforms(model.body_q, shapes)
    for shape, pose in zip(shapes, world, strict=True):
        shape["transform"] = pose.tolist()
    finite = [k for k, shape in enumerate(shapes) if "plane" not in shape["geometry"]]
    center = np.median(world[finite, :3], axis=0) if finite else np.zeros(3)
    meta = {"up_axis": int(model.up_axis), "shapes": shapes, "skipped": skipped, "scene_center": center.tolist()}
    np.savez(directory / "geometry.npz", **arrays)
    (directory / "scene.json").write_text(json.dumps(meta))
    return meta


def shape_world_transforms(body_q, shapes: list[dict]) -> np.ndarray:
    """World transforms ``[p, q_xyzw]`` of exported shapes, ``body_q[body] * shape_transform``, shape [N, 7]."""
    local = np.asarray([s["local_transform"] for s in shapes], dtype=np.float64).reshape(-1, 7)
    body = np.asarray([s["body"] for s in shapes], dtype=np.int64)
    moving = body >= 0
    if body_q is None or not moving.any():
        return local
    poses = body_q.numpy().astype(np.float64)[body[moving]]
    out = local.copy()
    out[moving, :3] = poses[:, :3] + _quat_rotate(poses[:, 3:], local[moving, :3])
    out[moving, 3:] = _quat_multiply(poses[:, 3:], local[moving, 3:])
    return out


class BlenderRenderer:
    """One persistent Blender process rendering one world of a Newton model on request.

    Args:
        model: Model whose visible shapes in ``world_id`` are mirrored in Blender.
        blender: Blender executable (see :func:`find_blender`).
        world_id: World to export.
        device: Cycles device, ``"OPTIX"``, ``"CUDA"``, or ``"CPU"``.
        workdir: Directory for the scene package, socket, and log; a temporary directory by default.
    """

    def __init__(self, model, *, blender: str, world_id: int = 0, device: str = "OPTIX", workdir=None):
        started = time.perf_counter()
        self.model = model
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="newton_blender_"))
        self.meta = export_scene(model, self.workdir / "scene", world_id=world_id)
        self.socket_path = str(self.workdir / "blender.sock")
        env = dict(os.environ)
        # With an X display (for example Xvfb) EEVEE picks Mesa's CPU renderer; headless EGL uses the GPU.
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        self.log = open(self.workdir / "blender.log", "w")
        self.proc = subprocess.Popen(
            [
                blender,
                "-b",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(SERVER),
                "--",
                "--scene",
                str(self.workdir / "scene"),
                "--socket",
                self.socket_path,
                "--device",
                device,
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.sock = socket.socket(socket.AF_UNIX)
        while True:
            if self.proc.poll() is not None:
                self.log.close()
                log = (self.workdir / "blender.log").read_text()[-1500:]
                raise RuntimeError(f"Blender exited with code {self.proc.returncode}:\n{log}")
            try:
                self.sock.connect(self.socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)
        self.stream = self.sock.makefile("rwb")
        json.loads(self.stream.readline())
        self.startup_s = time.perf_counter() - started

    def request(self, payload: dict) -> dict:
        self.stream.write((json.dumps(payload) + "\n").encode())
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise RuntimeError(f"Blender closed the connection; see {self.workdir / 'blender.log'}")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Blender error") + "\n" + reply.get("traceback", ""))
        return reply

    def render(
        self,
        state,
        *,
        pose,
        fov_y: float,
        width: int,
        height: int,
        samples: int = 16,
        engine: str = "EEVEE",
        intrinsics: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Render ``state`` from a Newton camera pose; returns an sRGB image [H, W, 3] uint8 and timings."""
        from .imaging import load_image  # noqa: PLC0415

        out = self.workdir / "frame.png"
        camera = {"pose": [float(v) for v in pose], "fov_y": float(fov_y)}
        if intrinsics is not None:
            camera.update({k: float(v) for k, v in intrinsics.items() if k in ("fx", "fy", "cx", "cy")})
            camera["image_width"] = float(intrinsics.get("image_width", width))
            camera["image_height"] = float(intrinsics.get("image_height", height))
        reply = self.request(
            {
                "cmd": "render",
                "transforms": shape_world_transforms(state.body_q, self.meta["shapes"]).tolist(),
                "camera": camera,
                "width": int(width),
                "height": int(height),
                "samples": int(samples),
                "engine": engine.upper(),
                "out": str(out),
            }
        )
        return load_image(out), {"render_s": reply.get("render_s")}

    def exec(self, code: str) -> str:
        """Run ``bpy`` code in the worker; ``result`` set by the code is returned as its repr."""
        return self.request({"cmd": "exec", "code": code}).get("result")

    def save_blend(self, path) -> str:
        return self.request({"cmd": "save_blend", "path": str(path)})["path"]

    def close(self):
        try:
            self.request({"cmd": "quit"})
        except Exception:
            pass
        for item in (self.stream, self.sock):
            try:
                item.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
