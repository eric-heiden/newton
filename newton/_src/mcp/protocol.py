# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small stdio MCP adapter following the 2025-11-25 tools specification.

The loopback connection is a private attachment protocol, not an HTTP MCP
endpoint. Device operations occur exclusively in the embedding process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from typing import ClassVar

from .transport import _MAX_REQUEST, _MAX_RESPONSE, SimulationClient, _encode

_INDEX = {
    "oneOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 256}]
}
_ENTRY = {
    "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}, "maxItems": 8}],
    "description": "Coupled solver entry name or nested slash-separated path; indices are local to that entry.",
}
_PAGE = {
    "offset": {"type": "integer", "minimum": 0, "default": 0},
    "limit": {"type": "integer", "minimum": 1, "maximum": 256, "default": 100},
}
_FILTER = {
    name: {
        **_INDEX,
        "description": f"Select {name} indices" + ("; -1 selects global entities." if name == "world" else "."),
    }
    for name in ("world", "body", "shape", "joint")
}
_CAMERA_NOTE = "Omit eye/target/pose to auto-frame the scene from the view preset."
_VIEW = {"type": "string", "enum": ["iso", "front", "back", "left", "right", "top"], "description": _CAMERA_NOTE}
_VIEWS = {
    "type": "array",
    "minItems": 1,
    "maxItems": 16,
    "items": {
        "oneOf": [_VIEW, {"type": "object", "description": "Per-view camera settings, e.g. eye/target/fov_y/label."}]
    },
    "description": "Render several cameras into one labeled grid (presets or camera objects).",
}
_OBSERVE = {
    "view": _VIEW,
    "backend": {"type": "string", "enum": ["sensor", "viewer"], "default": "sensor"},
    "eye": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Camera position in world coordinates [m].",
    },
    "target": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Camera look-at point in world coordinates [m].",
    },
    "up": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
    "pose": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 7,
        "maxItems": 7,
        "description": "Position xyz [m] and quaternion xyzw, camera-local -Z forward/+Y up.",
    },
    "world_id": {"type": "integer", "minimum": 0, "default": 0},
    "width": {"type": "integer", "minimum": 1, "maximum": 2048, "default": 640},
    "height": {"type": "integer", "minimum": 1, "maximum": 2048, "default": 480},
    "fov_y": {
        "type": "number",
        "minimum": 1,
        "maximum": 175,
        "description": "Vertical field of view [degrees].",
    },
    "channel": {"type": "string", "enum": ["color", "albedo", "depth", "forward_depth", "normal", "shape_index"]},
    "shadows": {"type": "boolean"},
    "textures": {"type": ["boolean", "null"]},
    "wireframe": {"type": "boolean"},
    "contacts": {"type": "boolean"},
    "raw": {"type": "boolean"},
    "contact_depth": {"type": "string", "enum": ["visible", "always"]},
    "depth_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    "pick": {
        "type": "array",
        "maxItems": 32,
        "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
    },
}
_LABEL = {"type": ["boolean", "string"], "description": "Caption tiles (default on for grids)."}


def _tool(name: str, description: str, properties: dict | None = None, required: tuple = (), *, read_only=False):
    return {
        "name": "newton_" + name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties or {},
            "required": list(required),
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": read_only, "openWorldHint": False},
    }


TOOLS = [
    _tool(
        "describe",
        "Inspect live scene time/revision, solver entries, editable fields, capabilities and budgets.",
        read_only=True,
    ),
    _tool(
        "query",
        "Read bounded public fields. Omit field to list fields. Filters use semantic frequency metadata; joint DOF/coordinate filters expand joint indices. Global rows require world=-1. Query output contains indices, pagination and page statistics.",
        {
            "root": {
                "type": "string",
                "enum": ["model", "state", "control", "solver", "collision"],
                "default": "state",
            },
            "field": {"type": "string"},
            "entry": _ENTRY,
            **_PAGE,
            **_FILTER,
        },
        read_only=True,
    ),
    _tool(
        "edit",
        "Validate all numeric patches, update existing arrays in place, and notify the top-level solver once. Model fields are explicitly allowed by describe. Positive body_mass edits scale inertia and update inverses. Values must exactly match selected row shape; no broadcasting. Topology changes require rebuild.",
        {
            "patches": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {
                    "type": "object",
                    "properties": {
                        "root": {"type": "string", "enum": ["model", "control"], "default": "model"},
                        "field": {"type": "string"},
                        "indices": _INDEX,
                        "values": {"type": "array"},
                    },
                    "required": ["field", "values"],
                    "additionalProperties": False,
                },
            },
            "flags": {
                "oneOf": [{"type": "integer"}, {"type": "array", "items": {"type": "string"}}],
                "description": "Optional ModelFlags bitmask or names; must include inferred categories.",
            },
            "expected_revision": {"type": "integer"},
        },
        ("patches",),
    ),
    _tool(
        "contacts",
        "Inspect rigid or soft collision-pipeline contacts with capacity, count, overflow hints and world-space positions [m]. Rigid normals point A-to-B and distance is the signed surface gap [m]. Refresh recomputes top-level contacts; native solver contact ownership is reported separately.",
        {
            "refresh": {"type": "boolean", "default": False},
            "kind": {"type": "string", "enum": ["rigid", "soft"]},
            "entry": _ENTRY,
            "include_global": {"type": "boolean", "default": False},
            **_PAGE,
            **{k: v for k, v in _FILTER.items() if k != "joint"},
        },
    ),
    _tool("collide", "Recompute collision-pipeline contacts at the current state without advancing time."),
    _tool(
        "step",
        "Advance a bounded number of physics timesteps. The embedding callback may update control each step; inspect application behavior before editing its control arrays.",
        {
            "count": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 1},
            "dt": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "description": "Physics timestep [s]."},
        },
    ),
    _tool("play", "Resume playback in a session run loop. Custom application loops must honor session.paused."),
    _tool("pause", "Pause playback while continuing to service queued requests."),
    _tool(
        "reset",
        "Restore initial state/control/time, reset solver caches and application callback, and clear contact buffers. Use collide or contacts(refresh=True) to regenerate diagnostic contacts. Model parameter edits persist. An invalid model mutation requires rebuilding rather than state reset.",
    ),
    _tool(
        "checkpoint",
        "Save named public state/control arrays and session time. Hidden solver state is not captured; restore resets caches and does not promise bitwise replay.",
        {"name": {"type": "string", "minLength": 1, "maxLength": 64, "default": "default"}},
    ),
    _tool(
        "restore",
        "Restore a checkpoint's public arrays/time and reset hidden solver caches. Clear diagnostic contacts until collide or contacts(refresh=True). Model edits persist. Application reset callback also runs.",
        {"name": {"type": "string", "default": "default"}},
    ),
    _tool(
        "observe",
        "Render the current state as an inline PNG. Omit the camera to auto-frame the scene (view preset, default iso). "
        "views=[...] renders several cameras into one labeled grid. reference='photo.png' (or a list aligned with views) "
        "renders with the same size and returns simulated | reference | mismatch panels plus pixel statistics. "
        "Sensor backend needs no GL; optional raw depth/IDs and pixel picking preserve numeric values.",
        {
            **_OBSERVE,
            "views": _VIEWS,
            "reference": {
                "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": ["string", "null"]}}],
                "description": "Reference image path(s) taken with the same camera.",
            },
            "label": _LABEL,
        },
    ),
    _tool(
        "filmstrip",
        "Advance the simulation and return ONE labeled image grid (columns = times, rows = views): the fastest way to "
        "see a motion. Give absolute times [s] (use reset=true to start from t=0, or restore='checkpoint'), or count "
        "frames every_steps apart. references=[[paths per time] per view] adds reference and mismatch rows. "
        "Default tiles 320x240; state stays at the last time.",
        {
            "times": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 32},
            "count": {"type": "integer", "minimum": 1, "maximum": 32},
            "every_steps": {"type": "integer", "minimum": 1},
            "reset": {"type": "boolean", "default": False},
            "restore": {"type": "string"},
            "views": {**_VIEWS, "maxItems": 4},
            "references": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            **{
                k: v
                for k, v in _OBSERVE.items()
                if k in ("view", "eye", "target", "up", "fov_y", "width", "height", "world_id", "channel", "shadows")
            },
        },
    ),
    _tool(
        "record",
        "Start/stop/query a bounded PNG sequence recording with simulation timestamps and manifest. No ffmpeg required. Start captures the initial frame; step advances recording.",
        {
            "action": {"type": "string", "enum": ["start", "stop", "status"]},
            "every_steps": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
            "max_frames": {"type": "integer", "minimum": 1, "maximum": 1000},
            **_OBSERVE,
        },
        ("action",),
    ),
    _tool(
        "execute",
        "Run trusted Python in the live application process. Variables, imports and functions persist across calls "
        "(and across reset/rebuild). Globals: session, model, state, control, solver, contacts, np, wp, show, plus "
        "application objects listed in the server instructions. The last expression is returned (large/opaque values "
        "are summarized; _ keeps the value). show(img, label) returns images inline: numpy arrays, matplotlib figures, "
        "PNG paths, or observe/filmstrip results. Batch many evaluations per call and print compact numbers. "
        "session.dispatch(op, args) runs structured operations (observe, filmstrip, step, reset, checkpoint, restore, "
        "query, edit). A runtime error keeps variables and states whether the scene stayed valid; if invalid, fix "
        "with recovery='inspect' then 'acknowledge', or rebuild. Not a sandbox.",
        {
            "code": {"type": "string", "maxLength": 65536},
            "reset_namespace": {
                "type": "boolean",
                "default": False,
                "description": "Clear variables, imports, functions and source history before executing. Does not reset or repair physics.",
            },
            "recovery": {
                "type": "string",
                "enum": ["none", "inspect", "acknowledge"],
                "default": "none",
                "description": "Explicit trusted recovery while invalid. Inspect does not automatically resume; acknowledge accepts responsibility for model/solver coherence after successful code and leaves playback paused. No rollback.",
            },
        },
        ("code",),
    ),
    _tool(
        "rebuild",
        "Invoke the application rebuild callback for topology/solver changes within the same process. Callback arguments are application-specific and new bindings replace the scene, contacts, render caches and checkpoints. Python variables survive unless reset_namespace=true.",
        {"arguments": {"type": "object"}, "reset_namespace": {"type": "boolean", "default": False}},
    ),
]


_INSTRUCTIONS = """Live Newton simulation running in another process; its Python state persists between calls.
Efficient workflow:
- newton_observe() returns an inline image; omit the camera to auto-frame (view='iso'|'front'|'left'|'right'|'top'). views=[...] gives a multi-view grid in one image. reference='photo.png' renders at the photo's size with the same camera and adds reference and mismatch panels plus pixel statistics.
- newton_filmstrip(times=[...], reset=true) runs forward and returns one labeled grid of frames; references=[[...]] compares each frame with reference images.
- newton_execute runs Python in the app: batch several parameter candidates in one call, compute numeric comparisons, and call show(image_or_figure, label) to see custom plots or composites inline. Prefer one larger call over many small ones.
- session.dispatch('checkpoint', {'name': ...}) / ('restore', ...) branches from a saved state instead of re-simulating.
- Built-in helpers (no import needed; newton, np, wp are preloaded): rollout(frames or seconds=..., record={'name': 'expr' or fn}, start=True|'checkpoint', until='expr', every=k, plot=True) steps and returns NumPy series in one call; solver_contacts() lists active contacts per shape pair with the solver's effective parameters (MuJoCo solref/solimp/friction after priority and mixing) next to the authored materials; health() flags NaNs, runaway velocities, deep penetration, and solver buffer overflow.
If a cell raises, the error states whether the scene stayed valid. If invalid, use execute(recovery='inspect') to diagnose and 'acknowledge' after repair, or rebuild."""


_RTX_NOTE = (
    "; backend='rtx' path-traces a photographic image (about 1 s, first call 5-10 s) for judging appearance, "
    "the default sensor backend takes about 20 ms."
)


_BLENDER_NOTE = (
    "\n- backend='blender' renders with Blender EEVEE (about 1 s, first call 5-30 s; 'blender_cycles' path-traces) "
    "for matching real photos; blender(code) runs bpy in that worker to edit materials, lights, world, and "
    "exposure (shape_objects, sun, key, camera, world are predefined)."
)


def blender_available() -> bool:
    """Whether a Blender executable for ``observe(backend='blender')`` is configured or on ``PATH``."""
    # Same lookup as blender_bridge.find_blender, without importing Newton in the MCP server process.
    path = os.environ.get("NEWTON_BLENDER")
    return bool(path and os.access(path, os.X_OK)) or shutil.which("blender") is not None


def rtx_available() -> bool:
    """Whether the optional OVRTX renderer behind ``observe(backend='rtx')`` is installed."""
    return importlib.util.find_spec("ovrtx") is not None


_INSTRUCTIONS_LEAN = """Live Newton simulation running in another process; its Python state persists between calls.
newton_execute runs Python in it (preloaded: session, model, state, control, solver, newton, np, wp, show, rollout, health, solver_contacts, render, compare_images, contacts_between). Batch many evaluations per call and print compact numbers.
- render(**observe_options) returns an RGB numpy image directly (fast path for fitting loops); compare_images(sim, ref, mask=None, panel='edges'|'blend'|'mismatch') returns PSNR, SSIM and edge NCC (geometric alignment) and shows a comparison panel.
- rollout(frames or seconds=..., record={'name': 'expr' or fn}, start=True|'checkpoint', until='expr', plot=True) steps and returns NumPy series.
- solver_contacts(): active contacts per shape pair with the parameters the solver integrates and which material decided them. contacts_between(a, b=None): contact count, solver normal/friction force, slip speed, and penetration between two shape sets (label substrings), recordable over time in rollout(record=...). health(): NaNs, runaway velocities, penetration, full solver buffers.
- Images: show(session.dispatch('observe', {'view': 'iso'})) or show(session.dispatch('filmstrip', {'times': [0.5, 1.0], 'reset': True})); show() also takes arrays and matplotlib figures. observe options: views=[...], width/height, eye/target or pose, fov_y or intrinsics={'fx','fy','cx','cy', distortion..., 'distortion_model': 'opencv'|'inverse_brown_conrady' (RealSense)} for calibrated cameras, world_id, reference='photo.png'<<RTX>>
- session.dispatch('checkpoint' | 'restore' | 'reset' | 'describe', {...}) manage and inspect the scene.<<BLENDER>>
newton_rebuild reloads the application (for hosted scripts: re-imports the edited file) in the same process."""


def _compact(data: dict, *, full: bool = False) -> dict:
    """Drop default-valued status fields so responses stay short for language models."""
    if full:
        return data
    result = {}
    for key, value in data.items():
        if key == "workspace":
            continue
        if (key, value) in (("closed", False), ("valid", True), ("requires_rebuild", False), ("truncated", False)):
            continue
        if key in ("last_error", "result_repr", "stdout", "result") and value in (None, ""):
            continue
        if key in ("revision", "paused"):
            continue
        result[key] = value
    return result


class _Protocol:
    _PROFILES: ClassVar[dict[str, set[str] | None]] = {
        "full": None,
        "code": {"newton_describe", "newton_execute", "newton_observe", "newton_filmstrip", "newton_rebuild"},
        "lean": {"newton_execute", "newton_rebuild"},
    }

    def __init__(self, client: SimulationClient, *, profile: str = "full", app_guide: bool = True):
        if profile not in self._PROFILES:
            raise ValueError(f"profile must be one of {sorted(self._PROFILES)}")
        self.client = client
        self.initialized = False
        names = self._PROFILES[profile]
        self.tools = TOOLS if names is None else [tool for tool in TOOLS if tool["name"] in names]
        self.profile = profile
        self.app_guide = app_guide

    def handle(self, message: dict) -> dict | None:
        request_id = message.get("id")
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return self._error(request_id, -32600, "Invalid JSON-RPC request")
        method, params = message["method"], message.get("params", {})
        if "id" not in message:
            return None
        if not isinstance(request_id, str | int) or isinstance(request_id, bool):
            return self._error(None, -32600, "Invalid request id")
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "params must be an object")
        if method == "initialize":
            version = params.get("protocolVersion")
            versions = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
            self.initialized = True
            result = {
                "protocolVersion": version if version in versions else versions[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "newton-live", "version": "0.1.0"},
                "instructions": self._instructions(),
            }
        elif method == "ping":
            result = {}
        elif not self.initialized:
            return self._error(request_id, -32000, "Initialize before using tools")
        elif method == "tools/list":
            if params.get("cursor"):
                return self._error(request_id, -32602, "This server returns all tools in one page")
            result = {"tools": self.tools}
        elif method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str) or name not in {tool["name"] for tool in self.tools}:
                return self._error(request_id, -32602, "Unknown tool")
            if not isinstance(arguments, dict):
                return self._error(request_id, -32602, "Tool arguments must be an object")
            try:
                if name == "newton_rebuild":
                    reset_namespace = arguments.get("reset_namespace", False)
                    arguments = arguments.get("arguments", {})
                    if not isinstance(arguments, dict):
                        raise ValueError("Rebuild arguments must be an object")
                    arguments = {**arguments, "reset_namespace": reset_namespace}
                data = dict(self.client.request(name.removeprefix("newton_"), **arguments))
                content = []
                if "image_base64" in data:
                    content.append(
                        {
                            "type": "image",
                            "data": data.pop("image_base64"),
                            "mimeType": data.pop("mime_type", "image/png"),
                        }
                    )
                for image in data.pop("images", None) or []:
                    content.append({"type": "image", "data": image["image_base64"], "mimeType": image["mime_type"]})
                text = json.dumps(
                    _compact(data, full=name == "newton_describe"), allow_nan=False, separators=(",", ":")
                )
                content.insert(0, {"type": "text", "text": text})
                result = {"content": content, "isError": False}
            except Exception as error:
                result = {"content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}], "isError": True}
        else:
            return self._error(request_id, -32601, "Method not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _instructions(self) -> str:
        text = _INSTRUCTIONS_LEAN if self.profile == "lean" else _INSTRUCTIONS
        text = text.replace("<<RTX>>", _RTX_NOTE if rtx_available() else ".")
        text = text.replace("<<BLENDER>>", _BLENDER_NOTE if blender_available() else "")
        if not self.app_guide:
            return text
        try:
            guide = self.client.request("guide").get("guide")
        except Exception:
            guide = None
        if guide:
            text += "\n\nApplication guide:\n" + str(guide)[:8192]
        return text

    @staticmethod
    def _error(request_id, code, message):
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a stdio MCP server to an embedded Newton session")
    parser.add_argument("--connect", required=True, help="Private JSON connection descriptor")
    parser.add_argument("--timeout", type=float, default=30, help="Maximum waiting time before execution begins [s]")
    parser.add_argument(
        "--profile",
        choices=("full", "code", "lean"),
        default="full",
        help="Advertise all tools, five code-oriented tools, or only execute and rebuild; this does not change permissions",
    )
    parser.add_argument(
        "--no-app-guide",
        action="store_true",
        help="Omit the application guide from the server instructions (e.g. when the client prompt already has it)",
    )
    args = parser.parse_args()
    protocol = _Protocol(
        SimulationClient(args.connect, timeout=args.timeout), profile=args.profile, app_guide=not args.no_app_guide
    )
    source, output = sys.stdin.buffer, sys.stdout.buffer
    while True:
        line = source.readline(_MAX_REQUEST + 1)
        if not line:
            break
        if len(line) > _MAX_REQUEST:
            output.write(_encode(protocol._error(None, -32700, "Message too large"), _MAX_RESPONSE))
            output.flush()
            break
        try:
            message = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(message, dict):
                raise ValueError("Expected a JSON object")
            response = protocol.handle(message)
        except (ValueError, UnicodeError):
            response = protocol._error(None, -32700, "Invalid JSON")
        if response is not None:
            output.write(_encode(response, _MAX_RESPONSE))
            output.flush()
