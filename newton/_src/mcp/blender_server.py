# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Blender side of the MCP render worker (runs inside Blender and uses ``bpy``).

    blender -b --factory-startup --python blender_server.py -- --scene DIR --socket PATH

Builds a Blender scene from the package written by ``blender_bridge.export_scene``
and serves newline-delimited JSON requests on a Unix socket::

    {
        "cmd": "render",
        "transforms": [[x, y, z, qx, qy, qz, qw], ...],
        "camera": {"pose": [...], "fov_y": 45.0},
        "width": 640,
        "height": 480,
        "samples": 16,
        "engine": "EEVEE" | "CYCLES",
        "out": "/path/frame.png",
    }
    {"cmd": "exec", "code": "<bpy code; names: bpy, scene, shape_objects, materials, camera, world, sun, key>"}
    {"cmd": "save_blend", "path": "/path/scene.blend"}
    {"cmd": "quit"}

Each reply is one JSON line, ``{"ok": true, ...}`` or ``{"ok": false, "error": ...}``.
Newton's camera convention (local -Z forward, +Y up) is Blender's, so poses map
directly. Shape colors are sRGB and become linear base colors, as in Newton's
ray tracer. The view transform is Standard without exposure offset, so pixel
values are comparable with other renderers and real frames.
"""

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--socket", required=True)
parser.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
args = parser.parse_args(argv)

started = time.perf_counter()
meta = json.load(open(os.path.join(args.scene, "scene.json")))
geometry = np.load(os.path.join(args.scene, "geometry.npz"))
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Blender's world lighting (sun elevation, HDRI) assumes Z-up; rotate Y-up and X-up scenes.
UP = {
    0: Matrix(((0, 1, 0), (0, 0, 1), (1, 0, 0))),
    1: Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0))),
    2: Matrix.Identity(3),
}[int(meta.get("up_axis", 2))].to_4x4()


def srgb_to_linear(c):
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def pose_matrix(p):
    x, y, z, qx, qy, qz, qw = (float(v) for v in p)
    return UP @ Matrix.Translation((x, y, z)) @ Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()


materials = {}


def principled(key, spec):
    if key in materials:
        return materials[key]
    material = bpy.data.materials.new(f"mat_{len(materials)}")
    material.use_nodes = True
    nodes = material.node_tree
    bsdf = nodes.nodes["Principled BSDF"]
    rgb = [srgb_to_linear(c) for c in spec["color"]]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = spec.get("roughness", 0.5)
    bsdf.inputs["Metallic"].default_value = spec.get("metallic", 0.0)
    if spec.get("opacity", 1.0) < 1.0:
        bsdf.inputs["Alpha"].default_value = spec["opacity"]
    if spec.get("texture"):
        texture = nodes.nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(os.path.join(args.scene, spec["texture"]))
        mix = nodes.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        sockets = {s.identifier: s for s in mix.inputs}
        sockets["Factor_Float"].default_value = 1.0
        sockets["A_Color"].default_value = (*rgb, 1.0)
        nodes.links.new(texture.outputs["Color"], sockets["B_Color"])
        nodes.links.new(next(o for o in mix.outputs if o.identifier == "Result_Color"), bsdf.inputs["Base Color"])
    if spec.get("checker"):
        # A subtle 0.5 m checker on ground planes keeps scale cues, like Newton's viewers.
        coordinates = nodes.nodes.new("ShaderNodeTexCoord")
        checker = nodes.nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = 1.0
        checker.inputs["Color1"].default_value = (*rgb, 1.0)
        checker.inputs["Color2"].default_value = (*[0.8 * c for c in rgb], 1.0)
        mapping = nodes.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (2.0, 2.0, 2.0)
        nodes.links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])
        nodes.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
        nodes.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    materials[key] = material
    return material


meshes = {}


def mesh_for(key):
    if key in meshes:
        return meshes[key]
    vertices = geometry[f"{key}/vertices"].astype(np.float32)
    triangles = geometry[f"{key}/indices"].astype(np.int32).reshape(-1, 3)
    mesh = bpy.data.meshes.new(key)
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.ravel())
    mesh.loops.add(triangles.size)
    mesh.loops.foreach_set("vertex_index", triangles.ravel())
    mesh.polygons.add(len(triangles))
    mesh.polygons.foreach_set("loop_start", np.arange(0, triangles.size, 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(triangles), dtype=bool))
    if f"{key}/uvs" in geometry.files:
        uvs = geometry[f"{key}/uvs"].astype(np.float32)
        mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", uvs[triangles.ravel()].ravel())
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)
    # Keep box edges crisp and curved CAD surfaces smooth.
    mesh.set_sharp_from_angle(angle=math.radians(35.0))
    mesh.materials.append(None)
    meshes[key] = mesh
    return mesh


shape_objects = []
for shape in meta["shapes"]:
    obj = bpy.data.objects.new(shape["name"], mesh_for(shape["geometry"]))
    # Object-linked material slots let shapes that share a mesh differ in color.
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = principled(json.dumps(shape["material"], sort_keys=True), shape["material"])
    obj.matrix_world = pose_matrix(shape["transform"]) @ Matrix.Diagonal((*shape["scale"], 1.0))
    obj["newton_shape"] = shape["index"]
    scene.collection.objects.link(obj)
    shape_objects.append(obj)

# Cycles bakes the transform of single-user meshes into their vertices, so moving them rebuilds the
# BVH. A second, parked user keeps every mesh instanced and pose updates refit the top level only.
parking = bpy.data.collections.new("instancing_users")
scene.collection.children.link(parking)
for key, mesh in meshes.items():
    if mesh.users < 2:
        user = bpy.data.objects.new(f"{key}_user", mesh)
        user.location = (0.0, 0.0, -1000.0)
        user.scale = (1e-6, 1e-6, 1e-6)
        parking.objects.link(user)

world = bpy.data.worlds.new("world")
world.use_nodes = True
scene.world = world
if hasattr(world, "sun_threshold"):
    world.sun_threshold = 0.0  # EEVEE would otherwise extract a second shadow-casting sun from the HDRI
world_nodes = world.node_tree
background = world_nodes.nodes["Background"]
studio = os.path.join(
    os.path.dirname(bpy.app.binary_path),
    bpy.app.version_string.split()[0].rsplit(".", 1)[0],
    "datafiles",
    "studiolights",
    "world",
    "studio.exr",
)
if os.path.exists(studio):
    environment = world_nodes.nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(studio)
    world_nodes.links.new(environment.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 0.35
    # Camera rays see a neutral backdrop instead of the low-resolution HDRI.
    light_path = world_nodes.nodes.new("ShaderNodeLightPath")
    backdrop = world_nodes.nodes.new("ShaderNodeBackground")
    backdrop.inputs["Color"].default_value = (0.62, 0.66, 0.72, 1.0)
    mix = world_nodes.nodes.new("ShaderNodeMixShader")
    world_nodes.links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    world_nodes.links.new(background.outputs["Background"], mix.inputs[1])
    world_nodes.links.new(backdrop.outputs["Background"], mix.inputs[2])
    world_nodes.links.new(mix.outputs["Shader"], world_nodes.nodes["World Output"].inputs["Surface"])
else:
    background.inputs["Color"].default_value = (0.5, 0.55, 0.62, 1.0)
    background.inputs["Strength"].default_value = 0.5

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.0
sun.data.angle = math.radians(4.0)
sun.rotation_euler = (UP.to_3x3() @ Vector((-0.577, 0.577, -0.577))).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(sun)
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
key.data.energy, key.data.size = 30.0, 1.5
center = UP.to_3x3() @ Vector(meta.get("scene_center", (0.0, 0.0, 0.0)))
key.location = center + Vector((1.5, -1.5, 2.0))
key.rotation_euler = (center - key.location).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(key)

camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
camera.data.sensor_fit = "VERTICAL"
camera.data.sensor_height = 24.0
camera.data.clip_start, camera.data.clip_end = 0.005, 1000.0
scene.collection.objects.link(camera)
scene.camera = camera

render = scene.render
render.use_persistent_data = True
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGB"
render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
configured = {}


def configure(engine, samples):
    if configured.get("key") == (engine, samples):
        return
    if engine == "CYCLES":
        render.engine = "CYCLES"
        cycles = scene.cycles
        if args.device != "CPU":
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = args.device
            preferences.refresh_devices()
            for device in preferences.devices:
                device.use = device.type == args.device
            cycles.device = "GPU"
        else:
            cycles.device = "CPU"
        cycles.samples = samples
        cycles.use_adaptive_sampling = True
        cycles.adaptive_threshold = 0.02
        cycles.use_denoising = True
        cycles.denoiser = "OPENIMAGEDENOISE"
        cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        cycles.denoising_use_gpu = args.device != "CPU"
        cycles.max_bounces, cycles.diffuse_bounces, cycles.glossy_bounces = 6, 3, 3
        cycles.caustics_reflective = cycles.caustics_refractive = False
    elif engine == "EEVEE":
        # Blender 4.2-4.5 call the engine BLENDER_EEVEE_NEXT.
        engines = {item.identifier for item in render.bl_rna.properties["engine"].enum_items}
        render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in engines else "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = samples
        scene.eevee.use_shadows = True
        scene.eevee.use_raytracing = True
    else:
        raise ValueError(f"unknown engine {engine!r}; use EEVEE or CYCLES")
    configured["key"] = (engine, samples)


def set_camera(request):
    camera.matrix_world = pose_matrix(request["pose"])
    data = camera.data
    if "fy" in request:
        # Calibrated pinhole: focal length from fy and the principal point as lens shift, which
        # Blender measures in units of the fitted (vertical) sensor dimension; +x moves the view
        # right, +y moves it up.
        image_width, image_height = request["image_width"], request["image_height"]
        data.lens = request["fy"] / image_height * data.sensor_height
        data.shift_x = (0.5 * image_width - request["cx"]) / image_height
        data.shift_y = (request["cy"] - 0.5 * image_height) / image_height
    else:
        data.lens = 0.5 * data.sensor_height / math.tan(0.5 * math.radians(float(request.get("fov_y", 45.0))))
        data.shift_x = data.shift_y = 0.0


def handle(request):
    command = request.get("cmd", "render")
    if command == "render":
        t0 = time.perf_counter()
        for obj, shape, pose in zip(shape_objects, meta["shapes"], request["transforms"], strict=True):
            obj.matrix_world = pose_matrix(pose) @ Matrix.Diagonal((*shape["scale"], 1.0))
        width, height = int(request.get("width", 640)), int(request.get("height", 480))
        set_camera(request["camera"])
        configure(request.get("engine", "EEVEE"), int(request.get("samples", 16)))
        render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
        render.filepath = request["out"]
        t1 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        return {"ok": True, "update_s": round(t1 - t0, 4), "render_s": round(time.perf_counter() - t1, 4)}
    if command == "exec":
        names = {
            "bpy": bpy,
            "scene": scene,
            "shape_objects": shape_objects,
            "materials": materials,
            "camera": camera,
            "world": world,
            "sun": sun,
            "key": key,
            "np": np,
            "Vector": Vector,
            "Matrix": Matrix,
        }
        exec(request["code"], names)
        return {"ok": True, "result": repr(names.get("result"))[:4000]}
    if command == "save_blend":
        bpy.ops.wm.save_as_mainfile(filepath=request["path"])
        return {"ok": True, "path": request["path"]}
    if command == "quit":
        return {"ok": True, "quit": True}
    raise ValueError(f"unknown command {command!r}")


server = socket.socket(socket.AF_UNIX)
if os.path.exists(args.socket):
    os.unlink(args.socket)
server.bind(args.socket)
server.listen(1)
connection = server.accept()[0]
stream = connection.makefile("rwb")
stream.write((json.dumps({"ok": True, "setup_s": round(time.perf_counter() - started, 3)}) + "\n").encode())
stream.flush()
while line := stream.readline():
    try:
        reply = handle(json.loads(line))
    except Exception as error:  # keep serving after a bad request
        reply = {"ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-2000:]}
    stream.write((json.dumps(reply) + "\n").encode())
    stream.flush()
    if reply.get("quit"):
        break
connection.close()
server.close()
os.unlink(args.socket)
