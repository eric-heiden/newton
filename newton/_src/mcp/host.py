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
import json
import linecache
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import warp as wp


def _load_module(path: Path, generation: int):
    # A fresh module name per load keeps Warp kernels of the old and new script apart.
    name = f"_newton_hosted_{path.stem}_{generation}"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    # Compile from source rather than __pycache__: an edit within the same second
    # that keeps the file size would otherwise load stale bytecode.
    linecache.checkcache(str(path))
    code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    sys.path.insert(0, str(path.parent))
    try:
        exec(code, module.__dict__)
    finally:
        sys.path.remove(str(path.parent))
    return module


class _RecordingViewer:
    """Null-viewer mixin that keeps the meshes an example logs in ``render()`` for MCP observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_meshes = {}

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


class ExampleHost:
    """Construct and serve a Newton ``Example`` script.

    Args:
        script: Path to a Python file defining ``Example`` (or ``example_class``).
        argv: Command-line arguments for the example's own parser.
        example_class: Class name inside the script.
    """

    def __init__(self, script: str | Path, argv: list[str] | None = None, *, example_class: str = "Example"):
        self.script = Path(script).resolve()
        self.argv = list(argv or [])
        self.example_class = example_class
        self.generation = 0
        self.module = None
        self.example = None
        self.build_seconds = 0.0
        self.recaptures = 0
        self._fingerprint = None
        self._notes = []
        self._dynamic_scalars: set[str] = set()
        self._retired_graphs: list = []
        self._captured_solver = None
        self._no_solver = _NoSolver()

    def build(self, argv: list[str] | None = None) -> Any:
        """(Re)load the script from disk and construct its example with a null viewer."""
        if argv is not None:
            self.argv = list(argv)
        started = time.perf_counter()
        self.generation += 1
        module = _load_module(self.script, self.generation)
        cls = getattr(module, self.example_class)
        import newton.examples  # noqa: PLC0415

        parser = cls.create_parser() if hasattr(cls, "create_parser") else newton.examples.create_parser()
        args, _ = parser.parse_known_args(self.argv)
        args.viewer = "null"
        viewer_class = type("RecordingViewerNull", (_RecordingViewer, newton.viewer.ViewerNull), {})
        viewer = viewer_class(num_frames=1 << 62)
        example = cls(viewer, args)
        self.module, self.example, self.args = module, example, args
        self._dynamic_scalars = set()
        self._retired_graphs = []
        self._captured_solver = id(getattr(example, "solver", None))
        self.build_seconds = time.perf_counter() - started
        self._fingerprint = self.fingerprint()
        return example

    def fingerprint(self) -> dict:
        """Identity of the example's objects and values of its scalars, to detect edits a CUDA graph missed."""
        return {
            name: ("value", value) if type(value) in (int, float, bool, str) else ("object", id(value))
            for name, value in vars(self.example).items()
            if not isinstance(value, wp.Graph)
        }

    def recapture(self) -> bool:
        """Re-record the example's CUDA graphs after its solver was replaced.

        Solvers allocate some buffers lazily during their first step. Examples usually
        capture that first step, so those buffers belong to the first graph, and
        recording a second graph around the same solver instance corrupts memory.
        Recapture is therefore only allowed once ``example.solver`` is a new object.

        Returns:
            Whether a graph was recaptured.
        """
        example = self.example
        graphs = [value for value in vars(example).values() if isinstance(value, wp.Graph)]
        if not graphs:
            return False
        if id(getattr(example, "solver", None)) == self._captured_solver:
            raise RuntimeError(
                "Recapturing CUDA graphs around the same solver instance is unsafe; assign a new solver to "
                "example.solver first, or write the change into the script and newton_rebuild"
            )
        # Keep replaced graphs alive: buffers that solvers allocate lazily during their first
        # captured step belong to that graph, and freeing it would leave dangling pointers.
        self._retired_graphs.extend(graphs)
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
            with wp.ScopedCapture() as capture:
                example.simulate()
            for name, value in bound.items():
                setattr(example, name, value)
            example.graph = capture.graph
        else:
            return False
        self.recaptures += 1
        self._fingerprint = self.fingerprint()
        self._captured_solver = id(getattr(example, "solver", None))
        return True

    def sync(self, session) -> None:
        """Rebind the session and recapture CUDA graphs if example attributes changed since the last step."""
        current = self.fingerprint()
        if current == self._fingerprint:
            return
        changed = sorted(
            k for k in current.keys() | self._fingerprint.keys() if current.get(k) != self._fingerprint.get(k)
        )
        previous, self._fingerprint = self._fingerprint, current
        state = getattr(self.example, "state_0", None) or getattr(self.example, "state", None)
        solver = getattr(self.example, "solver", None) or self._no_solver
        if session.solver is not solver or session.state is not state:
            session.solver, session.state = solver, state
            session.state_next = getattr(self.example, "state_1", None) or session.state_next
            session.control = getattr(self.example, "control", session.control)
        if not any(isinstance(value, wp.Graph) for value in vars(self.example).values()):
            return
        objects = [k for k in changed if "object" in (current.get(k, ("",))[0], previous.get(k, ("",))[0])]
        values = [k for k in changed if k not in objects and k not in self._dynamic_scalars]
        note = None
        if id(getattr(self.example, "solver", None)) != self._captured_solver:
            self.recapture()
            note = f"CUDA graphs recaptured for the new example.solver (changed: {', '.join(changed[:6])})"
        elif objects:
            note = (
                f"example.{', example.'.join(objects[:4])} replaced, but this example replays CUDA graphs captured "
                "with the old objects; replace example.solver as well to recapture, or newton_rebuild"
            )
        elif values:
            note = (
                f"example.{', example.'.join(values[:4])} changed: Python code in step() sees it now, but values "
                "captured inside the example's CUDA graphs keep their captured values until newton_rebuild"
            )
        if note and note not in self._notes:
            self._notes.append(note)

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

    def guide(self, workers: int = 0) -> str:
        text = f"""Hosted Newton example: {self.script} (class {self.example_class}, args {self.argv}).
- `example` is the live Example instance and `module` its script module; one step is one example frame of {getattr(self.example, "frame_dt", "?")} s. Use rollout(...) or session.dispatch('step', {{'count': n}}) rather than example.step() so time, recordings, and bindings stay in sync.
- Checkpoints (session.dispatch('checkpoint'/'restore', {{'name': ...}})) and reset rewind the physics state, the example's own Warp arrays, and the scalar attributes that step() changes (timers, phase counters). Scalar settings you assign (gains, amplitudes, look-ahead) are kept across reset/restore; model arrays you edit are kept too. Other objects (meshes, SDFs, textures, Python containers) are not rewound; newton_rebuild gives a fresh scene. Branch candidates from one checkpoint instead of re-simulating the approach each time.
- Live edits: change model arrays and call example.solver.notify_model_changed(newton.ModelFlags....) (arrays are read at run time, so this works with CUDA graphs); assign example attributes read by step() in Python (gains, amplitudes); or assign a new solver to example.solver, after which the host re-records the example's CUDA graphs. A `note` in the result warns when a change cannot reach code inside the captured graphs; then write it into the script and newton_rebuild.
- Helpers (preloaded with newton, np, wp): rollout(frames or seconds=..., record={{'name': 'expr' or fn}}, start=True|'checkpoint', until='expr', every=k, plot=True) steps and returns NumPy series in one call; solver_contacts() lists active contacts per shape pair with the parameters the solver actually integrates and which shape's material decided them; health() flags NaNs, runaway velocities, deep penetration, and full solver buffers.
- Observations (session.dispatch('observe'/'filmstrip', ...), shown with show()) draw the model's visible shapes plus meshes the example logs in its own render() (e.g. extracted surfaces), auto-framed.
- Python errors in a cell are reported but keep the scene valid; statements before the failing line keep their effects.
- After editing the script on disk, newton_rebuild reloads and reconstructs it in this process (Python variables survive; pass arguments={{"argv": [...]}} to change example arguments). Rebuild once to confirm the edited script reproduces your live result."""
        if workers:
            text += f"""
- `workers` holds {workers} sibling live copies of this example (same script and arguments, separate processes and scenes). workers.map(code, [args, ...]) runs a code string once per item in parallel (the item is `args` inside; `example`, `rollout`, ... exist there too) and returns each result; workers.broadcast(code) defines helpers on all of them; workers.submit(code, args) returns a Future at once, so a sweep can run while you keep working here and collect .result() later. Workers do not see this session's Python variables or live edits: send the settings to test in `args`, and rebuild them (workers.broadcast("session.dispatch('rebuild', {{}})")) after editing the script."""
        return text

    def session(self, *, artifact_directory=None, workers=None, allow_execute: bool = True):
        """Create a :class:`SimulationSession` bound to the example on the calling thread."""
        from .session import SimulationSession  # noqa: PLC0415

        host = self

        def step(session, dt):
            host.sync(session)
            before = host._scalars()
            host.example.step()
            host._dynamic_scalars.update(k for k, v in host._scalars().items() if before.get(k, v) != v)
            session.state = getattr(host.example, "state_0", None) or getattr(host.example, "state", None)
            session.state_next = getattr(host.example, "state_1", None) or session.state_next
            host._fingerprint = host.fingerprint()

        def rebuild(session, argv=None, **_):
            host.build(argv)
            session.namespace.update(example=host.example, module=host.module)
            session.dt = getattr(host.example, "frame_dt", session.dt)
            return host.bindings()

        if self.example is None:
            self.build()
        session = SimulationSession(
            **self.bindings(),
            dt=getattr(self.example, "frame_dt", 1.0 / 60.0),
            step_callback=step,
            rebuild_callback=rebuild,
            snapshot_callback=self.snapshot,
            restore_callback=self.restore,
            allow_execute=allow_execute,
            artifact_directory=artifact_directory,
            namespace={"example": self.example, "module": self.module, "recapture": self.recapture},
            guide=self.guide(len(workers or [])),
            workers=workers,
            execute_callback=self.after_execute,
            overlay_callback=self.overlay_meshes,
            invalidate_on_error=False,
        )
        session.host = self
        return session


def main(argv: list[str] | None = None) -> None:
    """``python -m newton.mcp host SCRIPT --connection-file FILE [-- example args]``."""
    parser = argparse.ArgumentParser(prog="python -m newton.mcp host", description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--connection-file", type=Path, required=True)
    parser.add_argument("--class", dest="example_class", default="Example")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--workers", type=int, default=0, help="Also host this many sibling copies as a worker pool")
    parser.add_argument("--ready-file", type=Path)
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after "--" belongs to the example's own argument parser.
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    example_args = argv[split + 1 :]
    started = time.perf_counter()
    # Kernel-load messages would otherwise flood every execute result.
    if hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING
    else:
        wp.config.quiet = True
    children, worker_files = [], []
    import subprocess  # noqa: PLC0415

    for index in range(args.workers):
        connection = args.connection_file.with_name(f"{args.connection_file.stem}.worker-{index}.json")
        ready = args.connection_file.with_name(f"{args.connection_file.stem}.worker-{index}.ready")
        worker_files.append(connection)
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "newton.mcp",
                    "host",
                    str(args.script),
                    "--connection-file",
                    str(connection),
                    "--class",
                    args.example_class,
                    "--ready-file",
                    str(ready),
                    "--",
                    *example_args,
                ]
            )
        )
    host = ExampleHost(args.script, example_args, example_class=args.example_class)
    host.build()
    for index, child in enumerate(children):
        ready = args.connection_file.with_name(f"{args.connection_file.stem}.worker-{index}.ready")
        while not ready.exists():
            if child.poll() is not None:
                raise RuntimeError(f"Worker {index} exited during startup")
            time.sleep(0.05)
    session = host.session(artifact_directory=args.artifacts, workers=worker_files or None)
    from .transport import SimulationServer  # noqa: PLC0415

    server = SimulationServer(session, connection_file=args.connection_file)
    server.start()
    marker = args.ready_file or args.connection_file.with_suffix(".ready")
    marker.write_text(json.dumps({"pid": os.getpid(), "startup_seconds": time.perf_counter() - started}))
    print(f"READY: {args.connection_file}", flush=True)
    try:
        session.run()
    finally:
        server.close()
        for child in children:
            child.terminate()
