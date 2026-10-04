# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host an unmodified Newton example script as a live MCP session.

Newton examples share one convention: an ``Example(viewer, args)`` class that
owns ``model``, ``solver``, ``state_0``/``state_1``, ``control``, and advances
one frame per ``step()``. :class:`ExampleHost` wraps such a script without
changes, so an agent can inspect, checkpoint, and edit a running simulation,
and reload the edited script in the same process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import linecache
import os
import sys
import threading
import time
from pathlib import Path
from types import FunctionType, MethodType, ModuleType
from typing import Any

import numpy as np
import warp as wp

import newton

from .fresh import FreshRunner
from .protocol import rtx_available

_RTX_GUIDE = (
    " backend='rtx' path-traces a photographic image (about 1 s; the first call and calls after shape_color "
    "edits rebuild the renderer, 5-10 s);"
)


def _load_module(path: Path, generation: int, source: bytes):
    # A fresh module name per load keeps Warp kernels of the old and new script apart.
    name = f"_newton_hosted_{path.stem}_{generation}"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    # Compile from source rather than __pycache__: an edit within the same second
    # that keeps the file size would otherwise load stale bytecode.
    linecache.checkcache(str(path))
    code = compile(source, str(path), "exec", dont_inherit=True)
    sys.path.insert(0, str(path.parent))
    try:
        exec(code, module.__dict__)
    finally:
        sys.path.remove(str(path.parent))
    return module


_SCALARS = (int, float, bool, str, type(None))


def _overridden(current: Any, value: Any) -> Any:
    """``value`` as the new value of a module global that is ``current``: dicts merge, other values replace."""
    if isinstance(current, dict) and isinstance(value, dict):
        merged = copy.copy(current)
        for key, item in value.items():
            merged[key] = _overridden(current[key], item) if key in current else item
        return merged
    # JSON has no tuples or integer floats; keep the script's own types where that is unambiguous.
    if type(current) is float and type(value) is int:
        return float(value)
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current, np.ndarray) and isinstance(value, list | int | float):
        return np.asarray(value, dtype=current.dtype)
    return value


def _apply_overrides(module: ModuleType, overrides: dict) -> None:
    namespace = vars(module)
    unknown = sorted(name for name in overrides if name not in namespace)
    if unknown:
        data = sorted(
            name
            for name, value in namespace.items()
            if not name.startswith("_") and isinstance(value, (*_SCALARS, list, tuple, dict))
        )
        raise ValueError(
            f"The script defines no module global {', '.join(unknown)}; its data globals are {', '.join(data[:80])}"
        )
    for name, value in overrides.items():
        namespace[name] = _overridden(namespace[name], value)


def _checked_overrides(overrides: Any) -> dict:
    if not isinstance(overrides, dict) or not all(isinstance(name, str) and name.isidentifier() for name in overrides):
        raise ValueError("overrides must map module global names to values, e.g. {'SUBSTEPS': 32}")
    return copy.deepcopy(overrides)


_UNFROZEN = object()


def _frozen(value: Any, budget: list[int]) -> Any:
    """Hashable copy of a small structure of scalars, lists, tuples, and dicts, or ``_UNFROZEN``."""
    kind = type(value)
    budget[0] -= 1
    if budget[0] < 0:
        return _UNFROZEN
    if kind in _SCALARS:
        return value if value == value else "nan"
    if isinstance(value, np.generic):
        return value.item()
    if kind is np.ndarray and value.size <= 64:
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if kind in (list, tuple, dict):
        frozen = []
        for item in value.items() if kind is dict else value:
            entry = _frozen(item, budget)
            if entry is _UNFROZEN:
                return _UNFROZEN
            frozen.append(entry)
        return (kind.__name__, tuple(frozen))
    return _UNFROZEN


def _setting(value: Any) -> tuple:
    """Fingerprint entry: scalars and small plain-data containers by value, everything else by identity."""
    if type(value) in _SCALARS:
        return ("value", value if value == value else "nan")
    frozen = _frozen(value, [64])
    return ("object", id(value)) if frozen is _UNFROZEN else ("value", frozen)


def _settings_object(value: Any) -> bool:
    """Whether the fingerprint walks into ``value``'s attributes.

    These are objects whose settings and buffers a CUDA graph may bake in: Newton and MuJoCo-Warp objects
    (solvers, models, options) and instances of classes from the hosted script, its local modules, or cells.
    """
    if isinstance(value, (*_SCALARS, wp.array, np.ndarray, wp.Graph, ModuleType, type, FunctionType, MethodType)):
        return False
    if not hasattr(value, "__dict__") or isinstance(value, newton.viewer.ViewerBase):
        return False
    name = getattr(type(value), "__module__", "") or ""
    if name.startswith(("newton", "mujoco_warp", "_newton_hosted_", "_newton_mcp_")):
        return True
    from .rollback import user_frame  # noqa: PLC0415

    path = getattr(sys.modules.get(name), "__file__", None)
    return bool(path) and user_frame(path)


class _RecordingViewer:
    """Null-viewer mixin that keeps the meshes an example logs in ``render()`` for MCP observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_meshes = {}

    def _log_triangles(self, state):
        # The camera sensor renders the model's own triangle mesh (cloth) directly, with its colors;
        # recording it as an overlay would rebuild a second scene on every observation.
        pass

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        *args,
        **kwargs,
    ):
        if hidden or points is None or indices is None:
            self.logged_meshes.pop(name, None)
        else:
            self.logged_meshes[name] = (points, indices, color)


class _NoSolver:
    """Stand-in for examples that advance without a dynamics solver (e.g. pure collision queries)."""

    def reset(self, *args, **kwargs) -> None:
        pass

    def notify_model_changed(self, flags) -> None:
        pass


def _frame_dt(example) -> str:
    frame_dt = getattr(example, "frame_dt", None)
    return f"{frame_dt:g} s" if isinstance(frame_dt, (int, float)) else "example.frame_dt"


def _worker_counts(workers) -> tuple[int, int]:
    if workers is None:
        return 0, 0
    if isinstance(workers, list | tuple):
        return len(workers), len(workers)
    return workers.count, workers.max_count


class ExampleHost:
    """Construct and serve a Newton ``Example`` script.

    Args:
        script: Path to a Python file defining ``Example`` (or ``example_class``).
        argv: Command-line arguments for the example's own parser.
        example_class: Class name inside the script.
        overrides: Module globals to set on every build (see :meth:`build`).
    """

    def __init__(
        self,
        script: str | Path,
        argv: list[str] | None = None,
        *,
        example_class: str = "Example",
        overrides: dict | None = None,
    ):
        self.script = Path(script).resolve()
        self.argv = list(argv or [])
        self.example_class = example_class
        self.overrides = _checked_overrides(overrides or {})
        """Active module-global overrides, applied by every build until replaced."""
        self.generation = 0
        self.module = None
        self.example = None
        self.build_seconds = 0.0
        self.recaptures = 0
        self._fingerprint = {}
        self._batch_start = None
        self._notes = []
        self._dynamic_scalars: set[str] = set()
        self._dynamic_keys: set[str] = set()
        self._no_solver = _NoSolver()
        self.restart_requested = False
        self.source_sha256: str | None = None
        """SHA-256 of the script source the current example was built from."""
        self.fresh = self._fresh_runner()
        """Runs the script from disk in clean subprocesses (``fresh`` in trusted execution)."""

    def _fresh_runner(self) -> FreshRunner:
        return FreshRunner(
            self.script,
            argv=lambda: self.argv,
            example_class=self.example_class,
            build_digest=lambda: self.source_sha256,
            overrides=lambda: self.overrides,
        )

    def build(self, argv: list[str] | None = None, overrides: dict | None = None) -> Any:
        """(Re)load the script from disk and construct its example with a null viewer.

        Nothing changes if loading or construction raises.

        Args:
            argv: Example arguments; ``None`` keeps the current ones.
            overrides: Module globals to set after the script is loaded and before ``Example()`` is
                constructed, e.g. ``{"SUBSTEPS": 32, "PARAMS": {"dt": 0.001}}``. A dictionary merges into a
                dictionary global (recursively); other values replace the global. The mapping becomes the
                active set used by later builds; ``None`` keeps the active set and ``{}`` clears it. Module
                code that already ran while loading (constants computed from the original value, default
                arguments) keeps the original values.
        """
        argv = self.argv if argv is None else list(argv)
        overrides = self.overrides if overrides is None else _checked_overrides(overrides)
        started = time.perf_counter()
        self.generation += 1
        source = self.script.read_bytes()
        module = _load_module(self.script, self.generation, source)
        _apply_overrides(module, copy.deepcopy(overrides))
        cls = getattr(module, self.example_class)
        import newton.examples  # noqa: PLC0415

        parser = cls.create_parser() if hasattr(cls, "create_parser") else newton.examples.create_parser()
        args, _ = parser.parse_known_args(argv)
        args.viewer = "null"
        viewer_class = type("RecordingViewerNull", (_RecordingViewer, newton.viewer.ViewerNull), {})
        viewer = viewer_class(num_frames=1 << 62)
        example = cls(viewer, args)
        self.argv, self.overrides = argv, overrides
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        self.module, self.example, self.args = module, example, args
        self._dynamic_scalars = set()
        self._dynamic_keys = set()
        self.build_seconds = time.perf_counter() - started
        self._fingerprint = self.fingerprint()
        return example

    def fingerprint(self, *, deep: bool = True) -> dict:
        """Settings a CUDA graph may have baked in, to detect edits that require re-recording it.

        Covers the example's attributes and, with ``deep``, the script's module globals and the
        attributes of the objects the example holds (solver, model, collision pipeline, the script's own
        controller objects) and of their option objects, three levels deep. Scalars and small plain-data
        containers are compared by value, other objects by identity.
        """
        example = self.example
        entries = {
            f"example.{name}": _setting(value)
            for name, value in vars(example).items()
            if not isinstance(value, wp.Graph)
        }
        if not deep:
            return entries
        for name, value in vars(self.module).items():
            if not name.startswith("__") and not isinstance(value, ModuleType):
                entries[f"module.{name}"] = _setting(value)
        visited = {id(example)}
        budget = [50_000]

        def walk(prefix: str, obj: Any, depth: int) -> None:
            for name, value in vars(obj).items():
                if budget[0] <= 0:
                    return
                budget[0] -= 1
                kind = type(value)
                # Fast paths for the common leaves (the rest of the walk runs once per step batch).
                if kind is wp.array:
                    entries[f"{prefix}.{name}"] = ("object", id(value))
                    continue
                if kind in _SCALARS:
                    entries[f"{prefix}.{name}"] = ("value", value if value == value else "nan")
                    continue
                if kind is wp.Graph:
                    continue
                key = f"{prefix}.{name}"
                entries[key] = _setting(value)
                if depth < 3 and id(value) not in visited and _settings_object(value):
                    visited.add(id(value))
                    walk(key, value, depth + 1)

        for name, value in vars(example).items():
            if id(value) not in visited and _settings_object(value):
                visited.add(id(value))
                walk(f"example.{name}", value, 1)
        return entries

    @staticmethod
    def _shallow(key: str) -> bool:
        return key.startswith("example.") and key.count(".") == 1

    def recapture(self) -> bool:
        """Re-record the example's CUDA graphs so they use its current solver, arrays, and scalars.

        Returns:
            Whether a graph was recaptured.
        """
        example = self.example
        graphs = [value for value in vars(example).values() if isinstance(value, wp.Graph)]
        if not graphs:
            return False
        capture_method = next(
            (
                getattr(example, name)
                for name in ("capture", "capture_graph", "capture_graphs", "_capture_graph", "_capture_graphs")
                if callable(getattr(example, name, None))
            ),
            None,
        )
        if capture_method is not None:
            capture_method()
        elif isinstance(getattr(example, "graph", None), wp.Graph) and callable(getattr(example, "simulate", None)):
            # The graph replays from the buffers bound at capture time.
            bound = {name: getattr(example, name) for name in ("state_0", "state_1") if hasattr(example, name)}
            # Record on the old graph's device, which need not be Warp's current default device.
            with wp.ScopedCapture(device=getattr(example.graph, "device", None)) as capture:
                example.simulate()
            for name, value in bound.items():
                setattr(example, name, value)
            example.graph = capture.graph
        else:
            return False
        self.recaptures += 1
        self._fingerprint = self.fingerprint()
        return True

    def sync(self, session, *, deep: bool = True) -> None:
        """Rebind the session and re-record CUDA graphs if settings changed since the last step.

        Args:
            session: The hosted session.
            deep: Compare all settings, not only the example's own attributes.
        """
        current = self.fingerprint(deep=deep)
        baseline = self._fingerprint
        keys = current.keys() | (baseline.keys() if deep else {key for key in baseline if self._shallow(key)})
        changed = [key for key in keys if current.get(key) != baseline.get(key)]
        if not changed:
            return
        if deep:
            self._fingerprint = current
        else:
            for key in changed:
                if key in current:
                    baseline[key] = current[key]
                else:
                    baseline.pop(key, None)
        state = getattr(self.example, "state_0", None) or getattr(self.example, "state", None)
        solver = getattr(self.example, "solver", None) or self._no_solver
        if session.solver is not solver or session.state is not state:
            session.solver, session.state = solver, state
            session.state_next = getattr(self.example, "state_1", None) or session.state_next
            session.control = getattr(self.example, "control", session.control)
        if not any(isinstance(value, wp.Graph) for value in vars(self.example).values()):
            return
        # Timers and phase counters that stepping itself advances do not require a new graph.
        edited = [key for key in changed if key not in self._dynamic_keys or current.get(key, ("",))[0] == "object"]
        if edited and self.recapture():
            # A replaced object implies new values for everything below it; name only the object.
            replaced = [key for key in edited if current.get(key, ("",))[0] == "object"]
            edited = [key for key in edited if not any(key.startswith(f"{parent}.") for parent in replaced)]
            public = sorted((key for key in edited if "._" not in key), key=lambda key: (key.count("."), key))
            names = public[:6] + ([f"{len(public) - 6} more"] if len(public) > 6 else [])
            # Private solver bookkeeping (e.g. snapshots refreshed by notify_model_changed) is summarized.
            owners = sorted({key.split("._", 1)[0] for key in edited if "._" in key})
            names += [f"private attributes of {owner}" for owner in owners[:3]]
            note = f"CUDA graphs recaptured after changes to {', '.join(names)}"
            if note not in self._notes:
                self._notes.append(note)

    def end_batch(self, session) -> None:
        """Refresh the settings baseline after consecutive steps; settings that stepping changed are dynamic."""
        current = self.fingerprint()
        start = self._batch_start or {}
        self._dynamic_keys.update(
            key for key, value in current.items() if value[0] == "value" and key in start and start[key] != value
        )
        self._fingerprint = current
        self._batch_start = None

    def overlay_meshes(self, session) -> list:
        """Meshes the example draws itself in ``render()`` (e.g. extracted surfaces), as NumPy arrays."""
        render = getattr(self.example, "render", None)
        viewer = getattr(self.example, "viewer", None)
        if not callable(render) or not hasattr(viewer, "logged_meshes"):
            return []
        viewer.logged_meshes.clear()
        render()
        meshes = []
        for name, (points, indices, color) in viewer.logged_meshes.items():
            vertices = points.numpy() if hasattr(points, "numpy") else np.asarray(points)
            triangles = indices.numpy() if hasattr(indices, "numpy") else np.asarray(indices)
            if len(vertices) and len(triangles) >= 3:
                meshes.append((name, np.asarray(vertices, np.float32).reshape(-1, 3), triangles.reshape(-1), color))
        return meshes

    def after_execute(self, session) -> str | None:
        self.sync(session)
        notes, self._notes = self._notes, []
        return "; ".join(notes) or None

    def undo_point(self, session, copies) -> Any:
        """Remember the example's and module's attributes and copy the example's arrays, for a rollback."""
        from .rollback import restore_attributes  # noqa: PLC0415

        example, module = self.example, self.module
        attributes, module_globals = dict(vars(example)), dict(vars(module))
        # Small plain-data lists and dicts (e.g. a PARAMS dict) are also restored when edited in place.
        contents = {
            (label, name): (value, copy.deepcopy(value), frozen)
            for label, namespace in (("example", attributes), ("module", module_globals))
            for name, value in namespace.items()
            if type(value) in (list, dict) and (frozen := _frozen(value, [256])) is not _UNFROZEN
        }
        copies.capture("example", example)
        saved = (dict(self._fingerprint), set(self._dynamic_scalars), set(self._dynamic_keys), list(self._notes))

        def undo() -> list[str]:
            if self.example is not example:
                return []
            restored = restore_attributes(vars(example), attributes, "example")
            restored += restore_attributes(vars(module), module_globals, "module")
            for (label, name), (value, original, frozen) in contents.items():
                if _frozen(value, [256]) != frozen:
                    if isinstance(value, dict):
                        value.clear()
                        value.update(original)
                    else:
                        value[:] = original
                    if f"{label}.{name}" not in restored:
                        restored.append(f"{label}.{name}")
            self._fingerprint, self._dynamic_scalars, self._dynamic_keys, self._notes = saved
            return restored

        return undo

    def install_solver(self, session, solver) -> None:
        """Make ``solver`` the example's solver and re-record its CUDA graphs (for ``swap_solver``)."""
        self.example.solver = solver
        self.sync(session)

    def echo(self, session) -> None:
        """Report the active overrides in every response of ``session``."""
        if self.overrides:
            session.status_fields["overrides"] = copy.deepcopy(self.overrides)
        else:
            session.status_fields.pop("overrides", None)

    def bindings(self) -> dict:
        example = self.example
        state = getattr(example, "state_0", None) or getattr(example, "state", None)
        return {
            "model": example.model,
            "solver": getattr(example, "solver", None) or self._no_solver,
            "state": state,
            "state_next": getattr(example, "state_1", None),
            "control": getattr(example, "control", None),
            "collision_pipeline": getattr(example, "collision_pipeline", None),
            "contacts": getattr(example, "contacts", None),
        }

    def _scalars(self) -> dict:
        return {name: value for name, value in vars(self.example).items() if type(value) in (int, float, bool)}

    def snapshot(self, session) -> dict:
        """Capture the example's own Warp arrays and scalar attributes (controller phases, timers)."""
        arrays = {
            name: value.numpy().copy()
            for name, value in vars(self.example).items()
            if isinstance(value, wp.array) and value.ndim >= 1
        }
        return {"arrays": arrays, "scalars": self._scalars()}

    def restore(self, session, data: dict) -> None:
        for name, value in data["arrays"].items():
            target = getattr(self.example, name, None)
            if isinstance(target, wp.array) and target.shape == value.shape:
                target.assign(value)
        # Only scalars that step() advances (timers, phase counters) rewind; settings the
        # agent assigned (gains, amplitudes) are not state and survive reset/restore.
        for name, value in data["scalars"].items():
            if name in self._dynamic_scalars:
                setattr(self.example, name, value)

    def guide(self, workers: int = 0, max_workers: int = 0) -> str:
        rtx = _RTX_GUIDE if rtx_available() else ""
        text = f"""Hosted Newton example: {self.script} (class {self.example_class}, args {self.argv}).
- `example` is the live Example instance and `module` its script module; one step is one example frame ({_frame_dt(self.example)}). Use rollout(...) or session.dispatch('step', {{'count': n}}) rather than example.step() so time, recordings, and bindings stay in sync.
- Checkpoints (session.dispatch('checkpoint'/'restore', {{'name': ...}})) and reset rewind the physics state, the example's own Warp arrays, and the scalar attributes that step() changes (timers, phase counters). Scalar settings you assign (gains, amplitudes, look-ahead) are kept across reset/restore; model arrays you edit are kept too. Other objects (meshes, SDFs, textures, Python containers) are not rewound; newton_rebuild gives a fresh scene. Branch candidates from one checkpoint instead of re-simulating the approach each time.
- Live edits: change model arrays and call example.solver.notify_model_changed(newton.ModelFlags....) (arrays are read at run time, so this works with CUDA graphs); assign example attributes (gains, amplitudes, substeps), module globals, or a new example.solver. Before the next step, the host compares the example's attributes, the script's module globals, and the scalar settings and buffers of the solver, model, their option objects, and the script's own objects (e.g. controllers) with their values when the CUDA graphs were recorded, and re-records the graphs if any changed (reported as `note`).
- Helpers (preloaded with newton, np, wp): rollout(frames or seconds=..., record={{'name': 'expr' or fn}}, start=True|'checkpoint', until='expr', every=k, plot=True) steps and returns NumPy series in one call; solver_contacts() lists active contacts per shape pair with the parameters the solver actually integrates and which shape's material decided them; contacts_between(a, b=None, detail=False) summarizes contacts between two shape sets (count, solver normal and friction force, slip speed, penetration) and works as a rollout(record=...) probe; health() flags NaNs, runaway velocities, deep penetration, and full solver buffers; swap_solver(factory, frames=2) replaces example.solver with factory(model): it re-records the CUDA graphs, steps a copy of the current state for two frames, runs health(), and restores the state, keeping the previous solver if any of this fails or health() reports a new warning.
- Observations (session.dispatch('observe'/'filmstrip', ...), shown with show()) draw the model's visible shapes plus meshes the example logs in its own render() (e.g. extracted surfaces), auto-framed, with a sky and a ground checker of reported cell size (environment=False for plain renders).{rtx} intrinsics={{fx, fy, cx, cy, ...}} matches a calibrated real camera (distortion_model='inverse_brown_conrady' for RealSense). render(**observe_options) returns a numpy image for fitting loops; compare_images(sim, ref, mask=None, panel='edges') scores PSNR, SSIM, and edge NCC against a real frame; observe(reference='frame.png', comparison='edges' or 'blend') shows the comparison panel. filmstrip(times=..., reset=True, references=video_frames, stride=k, comparison='edges', pose=..., intrinsics=...) steps the simulation to each recorded time and scores every frame against the video. camera_body='label' (with an optional camera_offset pose in the body frame) mounts the camera on a body such as a wrist camera link instead of a fixed pose, and overlay={{'name': 'expr' or fn or {{'body': label, 'point': [x, y, z]}}}} draws labeled rings at simulated points on the simulated and reference frames and returns their pixel coordinates.
- In models with several worlds, world_id selects the world in observations, and body labels resolve within it.
- If a cell raises (also inside rollout or a step), the simulation returns to its state before the cell: time, state, control, model arrays (the solver is notified of restored fields), the example's attributes and Warp arrays, and the script's module globals; the error lists what was restored. Python variables assigned before the error are kept, and objects changed in place (dicts, lists, solver internals) are not restored.
- fresh(argv_list=None, call=None, frames=None, timeout=300, parallel=2, wait=True) runs the script file as saved on disk (not this session's live edits) in new Python processes through `python -m newton.examples.headless`, one per argument list (default: this session's arguments), at most `parallel` at a time. Each process constructs Example with a null viewer, steps `frames` frames (default: the script's --num-frames), then evaluates the code string `call` with `example`, `module`, and `args` in scope. It returns one dict per argument list: status, exit_code, wall_seconds, frames, value (the JSON-converted value of `call`), exception with traceback, stdout/stderr tails, and after a timeout the Python stack the process was in; a timeout kills the process and everything it started. wait=False returns a handle with done(), result(), and cancel().
- newton_rebuild(arguments={{"restart": true}}) restarts the whole host process (fresh CUDA context, same script and arguments; Python variables are lost, and the next call waits for the new process). Use it only if the process is broken, e.g. after a CUDA error.
- After editing the script on disk, newton_rebuild reloads and reconstructs it in this process (Python variables survive; pass arguments={{"argv": [...]}} to change example arguments). If loading or constructing fails, the previous scene keeps running and the error shows the traceback with file:line.
- newton_rebuild(arguments={{"overrides": {{"NAME": value, ...}}}}) sets module globals of the script after it loads and before Example() is constructed: a dict merges into a dict global, other values replace the global. Module code that already ran while loading keeps values computed from the originals. The overrides stay active for later rebuilds (and restarts) and are echoed as `overrides` in every response until replaced; "overrides": {{}} clears them.
- persist('NAME', value) rewrites the script's module-level literal assignment NAME = ... (dict, list, tuple, number, string, bool, None; NumPy and Warp values become plain literals; without value it writes module.NAME), changing only the differing entries; persist_source(fn_or_class, target='Example.step') replaces the same-named top-level def/class (or the target method) with the definition from a cell. Both refuse when the target is missing, bound more than once, or not a literal/def, print a unified diff, save the previous file under the artifact directory, and then rebuild (rebuild=False skips); check='expr' reports the expression's live and rebuilt values and whether they agree. diff_model(since='build'|'last') lists model arrays changed since the build by entity label ([old, new]) with the ModelFlags inferred for them."""
        if max_workers:
            text += f"""
- `workers`: {workers} sibling copies of this example (same script and arguments; own processes, scenes, and globals); workers.resize(n) sets the count (0 to {max_workers}), workers.status() lists them. workers.map(fn, items) (or map(fn, xs, ys)) calls fn per item on free workers and returns results in input order ({{'error': ...}} for a call that raised); workers.submit(fn, *args, **kwargs) returns a Future; workers.broadcast(fn_or_code, *args) runs on every worker. Cell functions (lambdas and closures too) are sent by source with the cell functions and classes they use and the session globals they read; arguments, globals, and results are pickled. On a worker, example, model, state, solver, module, rollout and the other preloaded names are that worker's own (without this session's live edits). A code string runs as a cell with the item as `args`. workers.sync(name=value) or workers.sync('name') copies values to every worker; functions do not resend a synced name while it is bound to the same object. Printed text is echoed as `[worker i] ...`. newton_rebuild also rebuilds the workers. A worker whose process exits or whose CUDA context fails is restarted and replays earlier broadcast/sync calls; restarts are listed under `workers` in the next response.
- jobs.start(fn_or_code, *args, where='worker', **kwargs) queues a call on a free worker and returns a job id at once; jobs.wait(timeout=None, any=True) returns finished jobs (result or error, printed lines) and new lines of running jobs; jobs.result(id), jobs.cancel(id) (queued only), jobs.status(). Each execute response lists jobs finished since the previous response under `jobs`, with the ids still running or queued."""
        return text

    def session(self, *, artifact_directory=None, workers=None, allow_execute: bool = True):
        """Create a :class:`SimulationSession` bound to the example on the calling thread."""
        from .session import SimulationSession  # noqa: PLC0415

        host = self

        def step(session, dt):
            first = session._batch_step == 0
            # Settings beyond the example's own attributes can only change between batches of steps.
            host.sync(session, deep=first)
            if first:
                host._batch_start = dict(host._fingerprint)
            before = host._scalars()
            host.example.step()
            advanced = {k for k, v in host._scalars().items() if before.get(k, v) != v}
            host._dynamic_scalars.update(advanced)
            host._dynamic_keys.update(f"example.{name}" for name in advanced)
            session.state = getattr(host.example, "state_0", None) or getattr(host.example, "state", None)
            session.state_next = getattr(host.example, "state_1", None) or session.state_next
            shallow = host.fingerprint(deep=False)
            for key in [key for key in host._fingerprint if host._shallow(key) and key not in shallow]:
                del host._fingerprint[key]
            host._fingerprint.update(shallow)

        def rebuild(session, argv=None, restart=False, overrides=None):
            if restart:
                if overrides is not None:
                    host.overrides = _checked_overrides(overrides)
                    host.echo(session)
                # The host process re-executes itself once this response has been sent.
                host.restart_requested = True
                return host.bindings()
            host.build(argv, overrides)
            session.namespace.update(example=host.example, module=host.module)
            session.dt = getattr(host.example, "frame_dt", session.dt)
            host.echo(session)
            return host.bindings()

        if self.example is None:
            self.build()
        if self.fresh.closed:
            self.fresh = self._fresh_runner()
        session = SimulationSession(
            **self.bindings(),
            dt=getattr(self.example, "frame_dt", 1.0 / 60.0),
            step_callback=step,
            rebuild_callback=rebuild,
            snapshot_callback=self.snapshot,
            restore_callback=self.restore,
            allow_execute=allow_execute,
            artifact_directory=artifact_directory,
            namespace={
                "example": self.example,
                "module": self.module,
                "recapture": self.recapture,
                "fresh": self.fresh,
            },
            guide=self.guide(*_worker_counts(workers)),
            workers=workers,
            execute_callback=self.after_execute,
            overlay_callback=self.overlay_meshes,
            undo_callback=self.undo_point,
            solver_callback=self.install_solver,
            batch_callback=self.end_batch,
            close_callback=lambda _session: self.fresh.close(),
        )
        self.echo(session)
        # The session may have adjusted the model (e.g. contact capacity for its collision pipeline).
        self._fingerprint = self.fingerprint()
        session.host = self
        session.source_path = self.script
        return session


def _exit_with_parent(parent: int) -> None:
    """End this worker process when the host that launched it is gone, even if it was killed."""

    def watch():
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=watch, name="newton-mcp-parent-watch", daemon=True).start()


def main(argv: list[str] | None = None) -> None:
    """``python -m newton.mcp host SCRIPT --connection-file FILE [-- example args]``."""
    parser = argparse.ArgumentParser(prog="python -m newton.mcp host", description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--connection-file", type=Path, required=True)
    parser.add_argument("--class", dest="example_class", default="Example")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--workers", type=int, default=0, help="Also host this many sibling copies as a worker pool")
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Largest worker count workers.resize() may set (default: max(--workers, 4) with workers, else 0)",
    )
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument(
        "--overrides",
        type=json.loads,
        help="JSON object of module globals to set before Example() is constructed (as newton_rebuild overrides)",
    )
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after "--" belongs to the example's own argument parser.
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    example_args = argv[split + 1 :]
    started = time.perf_counter()
    if args.parent_pid is not None:
        _exit_with_parent(args.parent_pid)
    # Kernel-load messages would otherwise flood every execute result.
    if hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING
    else:
        wp.config.quiet = True
    max_workers = args.max_workers if args.max_workers is not None else (max(args.workers, 4) if args.workers else 0)
    if args.workers < 0 or max_workers < args.workers:
        parser.error("--workers must be in [0, --max-workers]")
    pool = None
    if max_workers:
        import signal  # noqa: PLC0415

        from .workers import WorkerPool  # noqa: PLC0415

        # SIGTERM unwinds through the cleanup below, which stops the worker processes by PID.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

        # Workers start (and compile kernels) while this process builds its own example.
        pool = WorkerPool.launch(
            args.script,
            example_args,
            count=args.workers,
            max_count=max_workers,
            example_class=args.example_class,
            directory=args.connection_file.parent,
            name=args.connection_file.stem,
            overrides=args.overrides,
        )
    server = None
    try:
        host = ExampleHost(args.script, example_args, example_class=args.example_class, overrides=args.overrides)
        host.build()
        if pool is not None:
            pool.wait_ready()
        session = host.session(artifact_directory=args.artifacts, workers=pool)
        from .transport import SimulationServer  # noqa: PLC0415

        server = SimulationServer(session, connection_file=args.connection_file)
        server.start()
        marker = args.ready_file or args.connection_file.with_suffix(".ready")
        marker.write_text(json.dumps({"pid": os.getpid(), "startup_seconds": time.perf_counter() - started}))
        print(f"READY: {args.connection_file}", flush=True)
        session.run(until=lambda: host.restart_requested)
    finally:
        if server is not None:
            if host.restart_requested:
                time.sleep(0.5)  # let the transport thread deliver the rebuild response
            server.close()
        if pool is not None:
            pool.close()
    if host.restart_requested:
        marker.unlink(missing_ok=True)
        print("RESTART: re-executing the host process", flush=True)
        command = _restart_command(argv, host.overrides, host.argv)
        os.execv(sys.executable, [sys.executable, "-m", "newton.mcp", "host", *command])


def _restart_command(argv: list[str], overrides: dict, example_args: list[str]) -> list[str]:
    """Host command line ``argv`` with the active ``--overrides`` (dropped if empty) and example arguments."""
    split = argv.index("--") if "--" in argv else len(argv)
    head, tail = argv[:split], ["--", *example_args]
    kept, skip = [], False
    for item in head:
        if skip:
            skip = False
        elif item == "--overrides":
            skip = True
        elif not item.startswith("--overrides="):
            kept.append(item)
    if overrides:
        kept += ["--overrides", json.dumps(overrides)]
    return kept + tail
