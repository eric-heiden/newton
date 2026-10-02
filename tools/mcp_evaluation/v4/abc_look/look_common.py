# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the ABC look-matching task (the verifier uses its own copy; do not edit).

``build_model`` assembles the fixed station twin from ``scene.json``. ``LookRenderer``
renders it in Blender EEVEE through the real top camera after running the look code
(``look.py``), and ``look_integrity`` checks that the look code only changed appearance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import twin_render
import warp as wp
from blender_bridge import BlenderRenderer, find_blender

import newton

HERE = Path(__file__).resolve().parent
NEUTRAL = (0.6, 0.6, 0.6)
ENGINE, SAMPLES = "EEVEE", 16


def build_model(directory: Path = HERE) -> newton.Model:
    """The station twin of ``scene.json``: edited station shapes plus added objects in a neutral color."""
    scene = json.loads((Path(directory) / "scene.json").read_text())
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(Path(directory) / "station" / "yam_bimanual_empty.xml"))
    index = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(builder.shape_label)}
    for shape in scene["shapes"]:
        if "edit" in shape:
            i = index[shape["edit"]]
            if "xform" in shape:
                builder.shape_transform[i] = wp.transform(*shape["xform"][:3], *shape["xform"][3:])
            if "scale" in shape:
                builder.shape_scale[i] = wp.vec3(*shape["scale"])
            if shape.get("visible") is False:
                builder.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
            continue
        xform = wp.transform(*shape["xform"][:3], *shape["xform"][3:])
        sx, sy, sz = shape["scale"]
        if shape["add"] == newton.GeoType.BOX:
            builder.add_shape_box(-1, xform=xform, hx=sx, hy=sy, hz=sz, color=NEUTRAL, label=shape["label"])
        elif shape["add"] == newton.GeoType.CYLINDER:
            builder.add_shape_cylinder(-1, xform=xform, radius=sx, half_height=sy, color=NEUTRAL, label=shape["label"])
        else:
            raise ValueError(f"unsupported shape type {shape['add']}")
    return builder.finalize()


def camera(directory: Path = HERE) -> tuple[list[float], dict]:
    """Top-camera pose ``[x, y, z, qx, qy, qz, qw]`` and pinhole intrinsics for the renderer."""
    scene = json.loads((Path(directory) / "scene.json").read_text())
    calibration = json.loads((Path(directory) / "camera.json").read_text())
    k = calibration["K"]
    intrinsics = {
        "fx": k[0],
        "fy": k[4],
        "cx": k[2],
        "cy": k[5],
        "image_width": calibration["width"],
        "image_height": calibration["height"],
    }
    return [*scene["camera"]["position"], *scene["camera"]["rotation"]], intrinsics


class LookRenderer:
    """Blender EEVEE renders of the twin through the real top camera, with a look applied.

    Args:
        model: Twin model from :func:`build_model`.
        look_code: ``bpy`` code (``look.py``) run once in the Blender worker before rendering.
        directory: Task directory with ``scene.json`` and ``camera.json``.
        workdir: Blender worker directory (scene export, socket, log).
    """

    def __init__(self, model: newton.Model, look_code: str = "", directory: Path = HERE, workdir=None):
        blender = find_blender()
        if blender is None:
            raise RuntimeError("Blender not found; set NEWTON_BLENDER")
        self.model = model
        self.pose, self.intrinsics = camera(directory)
        self.worker = BlenderRenderer(model, blender=blender, workdir=workdir)
        self.before = self.worker.exec(_INVENTORY)
        if look_code.strip():
            self.worker.exec(look_code)
        self.worker.exec(LOCK)
        self.after = self.worker.exec(_INVENTORY)

    def render(self, q: np.ndarray) -> np.ndarray:
        """Image with the robot posed from a joint-log row, shape [480, 640, 3], dtype uint8."""
        state = self.model.state()
        twin_render.pose_robot(self.model, state, q)
        image, _ = self.worker.render(
            state,
            pose=self.pose,
            fov_y=58.0,
            width=int(self.intrinsics["image_width"]),
            height=int(self.intrinsics["image_height"]),
            samples=SAMPLES,
            engine=ENGINE,
            intrinsics=self.intrinsics,
        )
        return image

    def close(self):
        self.worker.close()


# Mesh objects, their geometry, and loaded images in the Blender scene (the look may change none of them).
_INVENTORY = """
import numpy as _np
def _checksum(mesh):
    co = _np.empty(len(mesh.vertices) * 3, dtype=_np.float64)
    mesh.vertices.foreach_get("co", co)
    return round(float(_np.abs(co).sum()), 4)
result = {
    "meshes": __import__("hashlib").sha256(
        repr(sorted((o.name, o.data.name, _checksum(o.data)) for o in bpy.data.objects if o.type == "MESH")).encode()
    ).hexdigest(),
    "images": sorted(i.name for i in bpy.data.images if i.source in ("FILE", "SEQUENCE", "MOVIE", "GENERATED")),
}
"""

# Applied after the look: no compositing or sequencer post-processing, default pixel filter.
LOCK = """
scene.render.use_compositing = False
scene.render.use_sequencer = False
scene.render.filter_size = 1.5
scene.render.resolution_percentage = 100
"""


def look_integrity(renderer: LookRenderer) -> dict:
    """The look changed appearance only: same mesh objects and geometry, and no new images (no backplates)."""
    import ast  # noqa: PLC0415

    before, after = ast.literal_eval(renderer.before), ast.literal_eval(renderer.after)
    return {
        "meshes_unchanged": before["meshes"] == after["meshes"],
        "no_new_images": before["images"] == after["images"],
    }


def shape_masks(model: newton.Model, q: np.ndarray, directory: Path = HERE) -> np.ndarray:
    """Shape index per pixel of the top camera (sensor render, aligned with Blender's), shape [480, 640]."""
    import tempfile  # noqa: PLC0415

    pose, _ = camera(directory)
    # Blender renders a distortion-free pinhole; match it.
    calibration = json.loads((Path(directory) / "camera.json").read_text())
    calibration["D"] = [0.0] * 5
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as file:
        json.dump(calibration, file)
    sensor = twin_render.TopCamera(model, pose[:3], pose[3:], camera_file=Path(file.name), supersample=1)
    Path(file.name).unlink()
    output = sensor.sensor.utils.create_shape_index_image_output(sensor.width, sensor.height, camera_count=1)
    state = model.state()
    twin_render.pose_robot(model, state, q)
    model.bvh_refit_shapes(state)
    sensor.sensor.update(state, sensor.transforms, sensor.rays, shape_index_image=output)
    return output.numpy()[0, 0]


def region_color_error(
    masks: list[np.ndarray], renders: list[np.ndarray], recordings: list[np.ndarray]
) -> tuple[float, dict]:
    """Mean over shape regions of the distance between mean rendered and recorded sRGB colors [0-255].

    Regions are the visible shapes, eroded by 2 px to skip edges, with at least 400 pixels over all frames.
    """
    sums = {}
    for mask, render, recording in zip(masks, renders, recordings, strict=True):
        interior = mask.copy()
        for axis in (0, 1):
            for shift in (-2, -1, 1, 2):
                interior[np.roll(mask, shift, axis=axis) != mask] = 0xFFFFFFFF
        for shape in np.unique(interior):
            if shape == 0xFFFFFFFF:
                continue
            pixels = interior == shape
            entry = sums.setdefault(int(shape), [np.zeros(3), np.zeros(3), 0])
            entry[0] += render[pixels].astype(np.float64).sum(0)
            entry[1] += recording[pixels].astype(np.float64).sum(0)
            entry[2] += int(pixels.sum())
    errors = {shape: float(np.linalg.norm(a / n - b / n)) for shape, (a, b, n) in sums.items() if n >= 400}
    return float(np.mean(list(errors.values()))), errors
