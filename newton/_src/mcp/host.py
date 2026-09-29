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
        viewer = newton.viewer.ViewerNull(num_frames=1 << 62)
        example = cls(viewer, args)
        self.module, self.example, self.args = module, example, args
        self.build_seconds = time.perf_counter() - started
        self._fingerprint = self.fingerprint()
        return example

    def fingerprint(self) -> dict:
        """Identity of the example's objects and values of its scalars, to detect edits a CUDA graph missed."""
        return {
            name: value if type(value) in (int, float, bool, str) else id(value)
            for name, value in vars(self.example).items()
            if not isinstance(value, wp.Graph)
        }

    def recapture(self) -> bool:
        """Re-record the example's CUDA graph so it uses the current solver, arrays, and scalar settings.

        Returns:
            Whether a graph was recaptured.
        """
        example = self.example
        if not any(isinstance(value, wp.Graph) for value in vars(example).values()):
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
            with wp.ScopedCapture() as capture:
                example.simulate()
            for name, value in bound.items():
                setattr(example, name, value)
            example.graph = capture.graph
        else:
            return False
        self.recaptures += 1
        self._fingerprint = self.fingerprint()
        return True

    def sync(self, session) -> None:
        """Rebind the session and recapture CUDA graphs if example attributes changed since the last step."""
        current = self.fingerprint()
        if current == self._fingerprint:
            return
        changed = sorted(
            k for k in current.keys() | self._fingerprint.keys() if current.get(k) != self._fingerprint.get(k)
        )
        self._fingerprint = current
        state = getattr(self.example, "state_0", None) or getattr(self.example, "state", None)
        if session.solver is not self.example.solver or session.state is not state:
            session.solver, session.state = self.example.solver, state
            session.state_next = getattr(self.example, "state_1", None)
            session.control = getattr(self.example, "control", session.control)
        if self.recapture():
            note = f"CUDA graph recaptured after changes to example.{', example.'.join(changed[:6])}"
            if note not in self._notes:
                self._notes.append(note)

    def after_execute(self, session) -> str | None:
        self.sync(session)
        notes, self._notes = self._notes, []
        return "; ".join(notes) or None

    def bindings(self) -> dict:
        example = self.example
        state = getattr(example, "state_0", None) or getattr(example, "state", None)
        return {
            "model": example.model,
            "solver": example.solver,
            "state": state,
            "state_next": getattr(example, "state_1", None),
            "control": getattr(example, "control", None),
            "collision_pipeline": getattr(example, "collision_pipeline", None),
            "contacts": getattr(example, "contacts", None),
        }

    def snapshot(self, session) -> dict:
        """Capture the example's own Warp arrays and scalar attributes (controller phases, timers)."""
        arrays, scalars = {}, {}
        for name, value in vars(self.example).items():
            if isinstance(value, wp.array) and value.ndim >= 1:
                arrays[name] = value.numpy().copy()
            elif type(value) in (int, float, bool):
                scalars[name] = value
        return {"arrays": arrays, "scalars": scalars}

    def restore(self, session, data: dict) -> None:
        for name, value in data["arrays"].items():
            target = getattr(self.example, name, None)
            if isinstance(target, wp.array) and target.shape == value.shape:
                target.assign(value)
        for name, value in data["scalars"].items():
            setattr(self.example, name, value)

    def guide(self, workers: int = 0) -> str:
        text = f"""Hosted Newton example: {self.script} (class {self.example_class}, args {self.argv}).
- `example` is the live Example instance and `module` its script module; one step is one example frame of {getattr(self.example, "frame_dt", "?")} s. Use rollout(...) or session.dispatch('step', {{'count': n}}) rather than example.step() so time, recordings, and bindings stay in sync.
- Checkpoints (session.dispatch('checkpoint'/'restore', {{'name': ...}})) and reset include the example's own Warp arrays and scalar attributes, so controller phases and timers rewind with the physics state. Branch candidates from one checkpoint instead of re-simulating the approach each time.
- Live edits: change model arrays and call example.solver.notify_model_changed(newton.ModelFlags....); assign example attributes (gains, amplitudes) or replace example.solver with a new solver. After each cell the host recaptures the example's CUDA graph if any example attribute changed (reported as `note`); call recapture() after in-place changes it cannot see, such as solver option arrays.
- Helpers (preloaded with newton, np, wp): rollout(frames or seconds=..., record={{'name': 'expr' or fn}}, start=True|'checkpoint', until='expr', every=k, plot=True) steps and returns NumPy series in one call; solver_contacts() lists active contacts per shape pair with the parameters the solver actually integrates and which shape's material decided them; health() flags NaNs, runaway velocities, deep penetration, and full solver buffers.
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
            host.example.step()
            session.state = getattr(host.example, "state_0", None) or getattr(host.example, "state", None)
            session.state_next = getattr(host.example, "state_1", None)
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
