# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small stdio MCP adapter following the 2025-11-25 tools specification.

The loopback connection is a private attachment protocol, not an HTTP MCP
endpoint. Device operations occur exclusively in the embedding process.
"""

from __future__ import annotations

import argparse
import json
import sys

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
_OBSERVE = {
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
        "Render a bounded camera observation and return an MCP PNG image with metadata. Sensor is default and requires no GL. Viewer backend requires an attached ViewerGL. Backend settings are explicit; unsupported settings fail. Optional raw depth/IDs and pixel picking preserve numeric values.",
        _OBSERVE,
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
        "Run trusted unrestricted Python in a persistent workspace. Imports, functions and variables survive calls and physical resets; rebuilding clears them. Live globals session/model/solver/state/state_next/control/contacts/viewer/wp/np refresh after managed state changes. The last expression is returned (opaque or oversized values use bounded summaries without user repr); explicit result= takes precedence, and _ retains the last value. Use session.dispatch(operation, arguments) for structured operations. Runtime errors preserve partial work but pause/invalidate the scene: recovery='inspect' permits diagnosis while invalid; recovery='acknowledge' explicitly accepts caller-verified/repaired coherence after successful code, always paused. There is no rollback or automatic proof of safety. Compile errors do not execute. Source history and output are bounded; full Python is not a sandbox and cannot be preempted.",
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
        "Invoke the application rebuild callback for topology/solver changes within the same process. Callback arguments are application-specific and new bindings replace the scene, contacts, render caches and checkpoints.",
        {"arguments": {"type": "object"}},
    ),
]


class _Protocol:
    def __init__(self, client: SimulationClient, *, profile: str = "full"):
        if profile not in {"full", "code"}:
            raise ValueError("profile must be full or code")
        self.client = client
        self.initialized = False
        names = {"newton_describe", "newton_execute", "newton_observe", "newton_rebuild"}
        self.tools = TOOLS if profile == "full" else [tool for tool in TOOLS if tool["name"] in names]
        self.profile = profile

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
                "instructions": "Experimental live Newton session. Use newton_describe to discover bindings and capabilities when needed. Physics and rendering run on the simulation owner thread. In the code profile, use newton_execute with session.dispatch(operation, arguments) for structured operations; describe lists operation names. Python variables persist across calls. Runtime failures require explicit inspect/acknowledge recovery or rebuilding; successful code alone is not proof of solver coherence. Profiles change tool presentation, not permissions.",
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
                    arguments = arguments.get("arguments", {})
                    if not isinstance(arguments, dict):
                        raise ValueError("Rebuild arguments must be an object")
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
                content.append({"type": "text", "text": json.dumps(data, allow_nan=False)})
                result = {"content": content, "structuredContent": data, "isError": False}
            except Exception as error:
                result = {"content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}], "isError": True}
        else:
            return self._error(request_id, -32601, "Method not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id, code, message):
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a stdio MCP server to an embedded Newton session")
    parser.add_argument("--connect", required=True, help="Private JSON connection descriptor")
    parser.add_argument("--timeout", type=float, default=30, help="Maximum waiting time before execution begins [s]")
    parser.add_argument(
        "--profile",
        choices=("full", "code"),
        default="full",
        help="Advertise all tools or four code-oriented tools; this does not change permissions",
    )
    args = parser.parse_args()
    protocol = _Protocol(SimulationClient(args.connect, timeout=args.timeout), profile=args.profile)
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
