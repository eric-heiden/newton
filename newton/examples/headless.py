# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run an example script in a clean headless process and report the outcome as JSON.

.. code-block:: console

    python -m newton.examples.headless SCRIPT [script args] [--frames N] [--call CODE]
        [--set NAME=VALUE ...] [--repeat K] [--json OUT] [--timeout S] [--progress S]
        [--render] [--class NAME] [--tail CHARS] [-- script args]

``SCRIPT`` is a Python file that defines an ``Example(viewer, args)`` class, or
the short name of a bundled example such as ``basic_pendulum``. A new Python
process loads the script without running its ``__main__`` block, assigns the
``--set`` overrides, parses the script arguments with the script's own parser,
constructs the example with a null viewer, and steps ``N`` frames (default: the
script's ``--num-frames``).
It then evaluates ``CODE`` (an expression, or statements ending in one) with
``example``, ``module``, ``args``, ``newton``, ``np``, and ``wp`` in scope.
With ``--test`` among the script arguments, ``test_post_step()``,
``test_final()``, and the NaN checks run as in ``python -m newton.examples``
(after ``CODE``). ``--render`` also calls ``example.render()`` after every step.

``--set NAME=VALUE`` (repeatable) assigns a module global of the script after
it is loaded and before its parser and example are created. ``NAME`` must exist;
a dotted name such as ``Example.horizon`` or ``PARAMS.gain`` assigns an
attribute or an existing dictionary key. ``VALUE`` is a Python literal
(``0.002``, ``True``, ``[1, 2]``, ``'fast'``); any other text is a string. A
NumPy array global receives an array of its dtype. Values the script computed
from a global while loading keep the loaded value.

``--repeat K`` runs the script ``K`` times, each in a new process, one after
another; ``{run}`` in a ``--set`` value becomes the 0-based run index, e.g.
``--set SEED={run}``. ``--progress S`` prints a progress line to stderr every
``S`` seconds while a run is going (default 10; 0 turns them off).

Runner options are recognized anywhere after ``SCRIPT``; arguments after ``--``
always go to the script, for scripts that define options with the same names.

The JSON report (written to ``OUT``, or printed when ``--json`` is omitted)
contains ``status`` (``ok``, ``error``, ``timeout``, ``crashed``, or
``cancelled``), the process ``exit_code`` and terminating ``signal``,
``wall_seconds`` of the process, the last ``phase`` reached (``start``,
``load``, ``build``, ``step``, ``call``, ``test``, or ``done``), the ``frames``
stepped, per-phase host ``seconds``, the example's ``sim_time``, the
``realtime_factor`` (``sim_time`` over the seconds of the step phase), the
``overrides`` assigned, the JSON-converted ``value`` of ``CODE``, the
``exception`` with its traceback, the Python ``stack`` of every thread when a
timeout stopped the process, and the last characters of ``stdout`` and
``stderr``. With ``--repeat``, the report holds one such report per run in
``runs``, the ``status`` of the first run that was not ``ok`` (or ``ok``), the
``status_counts``, each run's ``values``, and the minimum, median, maximum, and
total ``wall_seconds``. In ``value``, non-finite floats become ``"nan"``,
``"inf"``, or ``"-inf"``, and objects without a JSON form become their
``repr()``. A timeout kills the process and everything it started. The exit
status is 0 for ``ok``, 124 for ``timeout``, 130 for ``cancelled``, and
otherwise nonzero; with ``--repeat``, that of the first run that was not ``ok``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import enum
import faulthandler
import importlib
import json
import linecache
import math
import mmap
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["main", "run_headless"]

# Options consumed by the runner, and whether each takes a value.
_RUNNER_OPTIONS = {
    "--frames": True,
    "--call": True,
    "--json": True,
    "--timeout": True,
    "--class": True,
    "--tail": True,
    "--set": True,
    "--repeat": True,
    "--progress": True,
    "--render": False,
}
_PROGRESS_INTERVAL = 1.0
# The child also arms its own deadline so it ends even if the supervising process dies.
_DEADLINE_GRACE = 10.0
_STACK_CHARS = 8000
_TRACEBACK_CHARS = 8000


def run_headless(
    script: str | os.PathLike,
    argv: list[str] | None = None,
    *,
    frames: int | None = None,
    call: str | None = None,
    timeout: float | None = None,
    example_class: str = "Example",
    render: bool = False,
    tail: int = 4000,
    echo: bool = False,
    cancel: threading.Event | None = None,
    overrides: Mapping[str, Any] | None = None,
    progress: float | None = None,
) -> dict[str, Any]:
    """Run an example script in a new Python process with a null viewer and report the outcome.

    The process reuses :data:`sys.executable` and the current environment and
    working directory. It loads the script as it is on disk, constructs its
    example from the script's own parser and ``argv``, steps ``frames``
    frames, and evaluates ``call``. See :mod:`newton.examples.headless` for
    the report fields.

    Args:
        script: Python file that defines the example class, or the short name
            of a bundled example.
        argv: Arguments for the script's own parser.
        frames: Number of frames to step; ``None`` steps the script's
            ``--num-frames``.
        call: Python expression, or statements ending in an expression,
            evaluated after stepping with ``example``, ``module``, ``args``,
            ``newton``, ``np``, and ``wp`` in scope. Its value (or the variable
            ``result`` if the code does not end in an expression) is reported
            as ``value``.
        timeout: Wall-clock limit for the process [s]. On expiry the runner
            records the Python stack of every thread, then kills the process
            group.
        example_class: Name of the example class in the script.
        render: Also call ``example.render()`` after every step.
        tail: Number of trailing characters of stdout and stderr to report.
        echo: Forward the process's stdout and stderr to this process while it runs.
        cancel: Event that stops the run and kills the process group when set.
        overrides: Module globals of the script to assign after it is loaded
            and before its parser and example are created, by name; dotted
            names assign attributes or dictionary keys. Every name must exist,
            and values must be Python literals (numbers, strings, booleans,
            ``None``, and lists, tuples, sets, and dictionaries of them).
        progress: Seconds between progress lines (phase, frames, ``sim_time``,
            wall time) printed to stderr while the run is going; ``None`` prints none.

    Returns:
        The JSON-compatible report.

    Raises:
        FileNotFoundError: If ``script`` is neither a file nor a bundled example name.
        ValueError: If an option is out of range.
    """
    target = _resolve(script)
    argv = [str(item) for item in (argv or [])]
    if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 0):
        raise ValueError("frames must be a non-negative integer or None")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, int | float) or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError("timeout must be a positive number of seconds or None")
    if call is not None and not isinstance(call, str):
        raise ValueError("call must be a string of Python code")
    if isinstance(tail, bool) or not isinstance(tail, int) or tail < 0:
        raise ValueError("tail must be a non-negative integer")
    if progress is not None and (
        isinstance(progress, bool)
        or not isinstance(progress, int | float)
        or not math.isfinite(progress)
        or progress <= 0
    ):
        raise ValueError("progress must be a positive number of seconds or None")
    assignments = _override_sources(overrides)
    with tempfile.TemporaryDirectory(prefix="newton-headless-") as temporary:
        directory = Path(temporary)
        spec = {
            **target,
            "argv": argv,
            "frames": frames,
            "call": call,
            "example_class": example_class,
            "render": bool(render),
            "result": str(directory / "result.json"),
            "stack": str(directory / "stack.txt"),
            "progress": str(directory / "frames.bin"),
            "deadline": None if timeout is None else float(timeout) + _DEADLINE_GRACE,
            "parent": os.getpid(),
            "overrides": assignments,
        }
        (directory / "spec.json").write_text(json.dumps(spec))
        (directory / "frames.bin").write_bytes(bytes(8))
        # -u: output written before a kill or crash still reaches the report.
        command = [sys.executable, "-u", "-m", "newton.examples.headless", "--child", str(directory / "spec.json")]
        outcome = _supervise(
            command, directory, timeout=timeout, tail=tail, echo=echo, cancel=cancel, progress=progress
        )
    return {
        "script": target.get("script") or target["module"],
        "argv": argv,
        "call": call,
        "timeout": timeout,
        "overrides": {name: _to_json(ast.literal_eval(source)) for name, source in assignments},
        **outcome,
    }


def _override_sources(overrides: Mapping[str, Any] | None) -> list[list[str]]:
    """``[name, source]`` pairs whose sources the child evaluates with :func:`ast.literal_eval`."""
    if overrides is None:
        return []
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be a mapping of global names to values")
    pairs = []
    for name, value in overrides.items():
        if not isinstance(name, str) or not all(part.isidentifier() for part in name.split(".")):
            raise ValueError(f"override name {name!r} must be a global name, optionally dotted (Example.horizon)")
        source = repr(value)
        try:
            same = ast.literal_eval(source) == value
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            same = False
        if not same:
            raise ValueError(
                f"override {name}={source[:200]} must be a Python literal (numbers, strings, booleans, None, "
                "and lists, tuples, sets, or dictionaries of them)"
            )
        pairs.append([name, source])
    return pairs


def _parse_override(text: str, run: int = 0) -> tuple[str, Any]:
    """``NAME`` and value of a ``--set NAME=VALUE`` option; VALUE is a Python literal, other text a string."""
    name, separator, value = text.partition("=")
    name = name.strip()
    if not separator or not name:
        raise ValueError(f"--set takes NAME=VALUE, got {text!r}")
    value = value.replace("{run}", str(run))
    try:
        return name, ast.literal_eval(value.strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return name, value


def _resolve(script: str | os.PathLike) -> dict[str, str]:
    path = Path(script)
    if path.is_file():
        return {"script": str(path.resolve())}
    import newton.examples  # noqa: PLC0415

    module = newton.examples.get_examples().get(str(script))
    if module is None:
        raise FileNotFoundError(f"No script file or bundled example named {str(script)!r}")
    return {"module": module}


# ---------------------------------------------------------------------------
# Supervising process
# ---------------------------------------------------------------------------


class _Tail:
    """Drain one output pipe on a thread, keeping its last characters."""

    def __init__(self, pipe, chars: int, echo):
        self.chars = chars
        self.data = bytearray()
        self.size = 0
        self._keep = max(4 * chars, 1024)  # UTF-8 uses up to four bytes per character
        self.thread = threading.Thread(target=self._read, args=(pipe, echo), daemon=True)
        self.thread.start()

    def _read(self, pipe, echo) -> None:
        target = getattr(echo, "buffer", None)
        with pipe:
            while chunk := pipe.read1(65536):
                self.size += len(chunk)
                self.data += chunk
                if len(self.data) > 2 * self._keep:
                    del self.data[: -self._keep]
                if target is not None:
                    try:
                        target.write(chunk)
                        target.flush()
                    except (OSError, ValueError):
                        target = None

    def text(self) -> str:
        self.thread.join(timeout=5.0)
        if self.chars == 0:
            return ""
        text = bytes(self.data[-self._keep :]).decode(errors="replace")
        if len(text) > self.chars or self.size > self._keep:
            return "..." + text[-self.chars :]
        return text


def _exited(process: subprocess.Popen) -> bool:
    """Whether the process has exited, without reaping it on POSIX (its group id stays reserved)."""
    if os.name == "posix" and hasattr(os, "waitid"):
        try:
            return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return True
    return process.poll() is not None


def _kill_group(process: subprocess.Popen) -> None:
    """Kill the process and everything it started (its own process group, created at launch)."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif process.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)


def _dump_stack(process: subprocess.Popen, path: Path) -> str | None:
    """Ask the child's faulthandler for the Python stack of every thread."""
    if not hasattr(signal, "SIGUSR1"):
        return None
    try:
        os.kill(process.pid, signal.SIGUSR1)
    except ProcessLookupError:
        return None
    size, deadline = -1, time.monotonic() + 1.0
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = path.stat().st_size if path.exists() else 0
        if current and current == size:
            break
        size = current
    text = path.read_text(errors="replace") if path.exists() else ""
    # Keep the script's frames; the runner's own frames are the same in every dump.
    lines = [line for line in text.splitlines() if __file__ not in line and "<frozen runpy>" not in line]
    return "\n".join(lines)[:_STACK_CHARS] or None


def _read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _progress_line(directory: Path, elapsed: float) -> str:
    record = _read_record(directory / "result.json")
    frames = int.from_bytes((directory / "frames.bin").read_bytes()[:8], "little")
    text = f"headless: {elapsed:.0f} s, phase {record.get('phase', 'start')}"
    if record.get("frames_requested") is not None:
        text += f", frame {max(frames, record.get('frames', 0))}/{record['frames_requested']}"
    if isinstance(record.get("sim_time"), int | float):
        text += f", sim_time {record['sim_time']:.4g} s"
    return text


def _supervise(command, directory: Path, *, timeout, tail, echo, cancel, progress=None) -> dict:
    if os.name == "posix":
        group = {"start_new_session": True}
    else:
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    started = time.perf_counter()
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **group
    )
    streams = [_Tail(process.stdout, tail, sys.stdout if echo else None)]
    streams.append(_Tail(process.stderr, tail, sys.stderr if echo else None))
    stopped, stack = None, None
    reported = started
    try:
        while not _exited(process):
            if progress is not None and time.perf_counter() - reported >= progress:
                reported = time.perf_counter()
                print(_progress_line(directory, reported - started), file=sys.stderr, flush=True)
            if timeout is not None and time.perf_counter() - started >= timeout:
                stopped = "timeout"
                stack = _dump_stack(process, directory / "stack.txt")
                break
            if cancel is not None and cancel.is_set():
                stopped = "cancelled"
                break
            time.sleep(0.05)
        wall = time.perf_counter() - started
    finally:
        # Also removes processes a finished script left running.
        _kill_group(process)
        try:
            returncode = process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            returncode = None  # not killable yet, e.g. blocked in a driver call
    stdout, stderr = (stream.text() for stream in streams)
    record = _read_record(directory / "result.json")
    # The shared counter is current even when the process was killed between progress records.
    frames = max(record.get("frames", 0), int.from_bytes((directory / "frames.bin").read_bytes()[:8], "little"))
    if stopped is not None:
        status = stopped
    elif record.get("complete") and record.get("status") == "error":
        status = "error"
    elif record.get("complete") and record.get("status") == "ok" and returncode == 0:
        status = "ok"
    else:
        status = "crashed"
    name = None
    if returncode is not None and returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = str(-returncode)
    seconds = record.get("seconds", {})
    sim_time, step_seconds = record.get("sim_time"), seconds.get("step")
    realtime = None
    if isinstance(sim_time, int | float) and isinstance(step_seconds, int | float) and step_seconds > 0.0:
        realtime = round(sim_time / step_seconds, 4)
    return {
        "status": status,
        "exit_code": returncode,
        "signal": name,
        "wall_seconds": round(wall, 3),
        "phase": record.get("phase", "start"),
        "frames": frames,
        "frames_requested": record.get("frames_requested"),
        "sim_time": sim_time,
        "realtime_factor": realtime,
        "device": record.get("device"),
        "seconds": seconds,
        "value": record.get("value"),
        "exception": record.get("exception"),
        "stack": stack,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


def _to_json(value: Any, _depth: int = 0) -> Any:
    """Convert a Python value to JSON-compatible data for a report.

    Non-finite floats become ``"nan"``, ``"inf"``, or ``"-inf"``; NumPy and Warp
    arrays become nested lists; objects without a JSON form become their ``repr()``.
    """
    if _depth > 64:
        return _repr(value)
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, enum.Enum):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return "nan" if value != value else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {str(key): _to_json(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_to_json(item, _depth + 1) for item in value]
    if isinstance(value, enum.Enum):
        return _to_json(value.value, _depth + 1)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_json({field.name: getattr(value, field.name) for field in dataclasses.fields(value)}, _depth + 1)
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, numpy.generic | numpy.ndarray):
        return _to_json(value.tolist(), _depth + 1)
    if type(value).__module__.split(".")[0] == "warp":
        try:
            converted = value.numpy() if hasattr(value, "numpy") else numpy.asarray(value)
            return _to_json(converted.tolist(), _depth + 1)
        except Exception:
            return _repr(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_json(tolist(), _depth + 1)
        except Exception:
            pass
    return _repr(value)


def _repr(value: Any) -> str:
    try:
        return repr(value)[:2000]
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


def _exception(error: BaseException) -> dict:
    # Drop the runner's own frames so the traceback starts in the script.
    tb = error.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
        tb = tb.tb_next
    text = "".join(traceback.format_exception(type(error), error, tb))
    return {
        "type": type(error).__name__,
        "message": str(error)[:4000],
        "traceback": text if len(text) <= _TRACEBACK_CHARS else "..." + text[-_TRACEBACK_CHARS:],
    }


class _Record:
    """Progress and result of the child, rewritten atomically so a killed run still reports how far it got."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {"phase": "start", "frames": 0, "seconds": {}}
        self.written = 0.0

    def write(self, **fields) -> None:
        self.data.update(fields)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_to_json(self.data)))
        os.replace(temporary, self.path)
        self.written = time.monotonic()


def _exit_with_parent(parent: int) -> None:
    if sys.platform.startswith("linux"):
        try:
            import ctypes  # noqa: PLC0415

            ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL))  # PR_SET_PDEATHSIG
        except Exception:
            pass
    if os.getppid() != parent:
        os._exit(1)  # the supervisor died before the death signal was armed


def _load(spec: dict) -> ModuleType:
    if spec.get("module"):
        sys.argv = [spec["module"], *spec["argv"]]
        return importlib.import_module(spec["module"])
    path = Path(spec["script"])
    sys.argv = [str(path), *spec["argv"]]
    # A name other than __main__ keeps the script's own main block from running.
    name = "_newton_headless_" + re.sub(r"\W", "_", path.stem)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    # Compile from source: bytecode caches can be stale after a same-size edit within a second.
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def _resolve_override(target: Any, part: str, name: str) -> Any:
    if isinstance(target, dict):
        if part in target:
            return target[part]
        options = [str(key) for key in target]
    else:
        try:
            return getattr(target, part)
        except AttributeError:
            options = [key for key in dir(target) if not key.startswith("__")]
    close = difflib.get_close_matches(part, options, n=3)
    hint = f"; did you mean {', '.join(repr(option) for option in close)}?" if close else ""
    owner = "the script" if isinstance(target, ModuleType) else repr(name.rsplit(".", 1)[0] if "." in name else name)
    raise AttributeError(f"--set {name}: {owner} has no {'key' if isinstance(target, dict) else 'name'} {part!r}{hint}")


def _apply_overrides(module: ModuleType, overrides: list[list[str]]) -> None:
    """Assign ``--set`` values to the script's globals, attributes, or dictionary keys; each must exist."""
    for name, source in overrides:
        value = ast.literal_eval(source)
        *path, last = name.split(".")
        target = module
        for index, part in enumerate(path):
            target = _resolve_override(target, part, ".".join(path[: index + 1]))
        current = _resolve_override(target, last, name)
        numpy = sys.modules.get("numpy")
        if numpy is not None and isinstance(current, numpy.ndarray):
            value = numpy.asarray(value, dtype=current.dtype)
        if isinstance(target, dict):
            target[last] = value
        else:
            setattr(target, last, value)


def _evaluate(code: str, scope: dict) -> Any:
    filename = "<call>"
    lines = code.splitlines(keepends=True)
    linecache.cache[filename] = (len(code), None, lines, filename)
    tree = ast.parse(code, filename=filename, mode="exec")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, filename, "exec"), scope)
    if last is None:
        return scope.get("result")
    return eval(compile(ast.Expression(last.value), filename, "eval"), scope)


def _sim_time(example) -> float | None:
    value = getattr(example, "sim_time", None)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _child_run(spec: dict, record: _Record) -> None:
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415
    import newton.examples  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415

    newton.examples._enable_example_deprecation_warnings()
    seconds = {}

    def phase(name: str, started: float, **fields) -> float:
        now = time.perf_counter()
        seconds[record.data["phase"]] = round(now - started, 4)
        record.write(phase=name, seconds=seconds, **fields)
        return now

    started = time.perf_counter()
    record.write(phase="load")
    module = _load(spec)
    _apply_overrides(module, spec.get("overrides", []))
    started = phase("build", started)
    cls = getattr(module, spec["example_class"], None)
    if cls is None:
        raise AttributeError(f"{spec.get('script') or spec['module']} defines no class {spec['example_class']!r}")
    create_parser = getattr(cls, "create_parser", None) or getattr(module, "create_parser", None)
    parser = create_parser() if callable(create_parser) else newton.examples.create_parser()
    args = parser.parse_args(spec["argv"])
    if hasattr(args, "warp_config"):
        newton.examples._apply_warp_config(parser, args)
    if getattr(args, "quiet", False):
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
    if getattr(args, "device", None):
        wp.set_device(args.device)
    args.viewer = "null"
    frames = spec["frames"] if spec["frames"] is not None else getattr(args, "num_frames", None)
    if frames is None:
        raise ValueError("The script's parser defines no --num-frames; pass --frames")
    viewer = newton.viewer.ViewerNull(num_frames=getattr(args, "num_frames", None) or max(frames, 1))
    example = cls(viewer, args)
    model = getattr(example, "model", None)
    device = getattr(model, "device", None)
    started = phase("step", started, frames_requested=frames, device=None if device is None else str(device))

    progress = open(spec["progress"], "r+b")
    counter = mmap.mmap(progress.fileno(), 8)
    test = bool(getattr(args, "test", False))
    post_step = test and hasattr(example, "test_post_step")
    for frame in range(frames):
        example.step()
        if post_step:
            example.test_post_step()
        if spec["render"]:
            example.render()
        record.data["frames"] = frame + 1
        counter[:8] = (frame + 1).to_bytes(8, "little")
        if time.monotonic() - record.written >= _PROGRESS_INTERVAL:
            record.write(sim_time=_sim_time(example))
    if device is not None and getattr(device, "is_cuda", False):
        # Count queued GPU work toward the step time rather than the call.
        wp.synchronize_device(device)
    started = phase("call", started, sim_time=_sim_time(example))

    if spec["call"] is not None:
        scope = {"example": example, "module": module, "args": args, "newton": newton, "np": np, "wp": wp}
        record.data["value"] = _to_json(_evaluate(spec["call"], {"__name__": "__headless_call__", **scope}))
    started = phase("test", started, sim_time=_sim_time(example))

    if test:
        if hasattr(example, "test_final"):
            example.test_final()
        elif not post_step:
            raise NotImplementedError("Example does not have a test_final or test_post_step method")
        newton.examples._test_finite(example)
    phase("done", started, sim_time=_sim_time(example))


def _child(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    _exit_with_parent(spec["parent"])
    record = _Record(Path(spec["result"]))
    # Stays open for the lifetime of the process: faulthandler writes to it from a signal handler.
    stack_file = open(spec["stack"], "w")
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True)
    if spec.get("deadline"):
        faulthandler.dump_traceback_later(spec["deadline"], exit=True)
    code = 0
    try:
        _child_run(spec, record)
        record.write(status="ok", complete=True)
    except BaseException as error:
        code = error.code if isinstance(error, SystemExit) and isinstance(error.code, int) else 1
        record.write(status="error", complete=True, exception=_exception(error))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(code)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _split(argv: list[str]) -> tuple[list[str], str | None, list[str]]:
    """Separate runner options, the script, and the script's own arguments."""
    runner, script, script_args = [], None, []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            script_args.extend(argv[index + 1 :])
            break
        name = token.split("=", 1)[0]
        if name in _RUNNER_OPTIONS:
            count = 1 if "=" in token or not _RUNNER_OPTIONS[name] else 2
            runner.extend(argv[index : index + count])
            index += count
            continue
        if script is None:
            # Before SCRIPT only runner options (and --help) are accepted; argparse reports the rest.
            if token.startswith("-"):
                runner.append(token)
            else:
                script = token
        else:
            script_args.append(token)
        index += 1
    return runner, script, script_args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m newton.examples.headless",
        usage="%(prog)s SCRIPT [script args] [--frames N] [--call CODE] [--set NAME=VALUE ...] [--repeat K] "
        "[--json OUT] [--timeout S] [--progress S] [-- script args]",
        description="Run an example script in a clean headless process and report the outcome as JSON.",
        allow_abbrev=False,
    )
    parser.add_argument("script", help="Python file defining Example(viewer, args), or a bundled example name.")
    parser.add_argument("--frames", type=int, default=None, help="Frames to step (default: the script's --num-frames).")
    parser.add_argument(
        "--call",
        default=None,
        help="Python code evaluated after stepping, with example, module, args, newton, np, and wp in scope; "
        "the value of its final expression is reported.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Assign an existing module global (or a dotted attribute or dictionary key) of the script before the "
        "example is created; VALUE is a Python literal, other text a string, and {run} the run index. Repeatable.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Run the script this many times, each in a new process (default 1)."
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the report to this file instead of stdout.")
    parser.add_argument("--timeout", type=float, default=None, help="Kill each run after this many seconds.")
    parser.add_argument(
        "--progress",
        type=float,
        default=10.0,
        help="Seconds between progress lines on stderr while a run is going (default 10; 0 for none).",
    )
    parser.add_argument("--class", dest="example_class", default="Example", help="Example class name in the script.")
    parser.add_argument("--render", action="store_true", help="Also call example.render() after every step.")
    parser.add_argument("--tail", type=int, default=4000, help="Characters of stdout and stderr to report.")
    return parser


def _summary(report: dict, prefix: str = "headless: ") -> str:
    text = (
        f"{prefix}{report['status']} after {report['wall_seconds']:.1f} s, phase {report['phase']}, "
        f"{report['frames']} frames, exit code {report['exit_code']}"
    )
    if report.get("exception"):
        text += f"; {report['exception']['type']}: {report['exception']['message'][:300]}"
    return text


def _repeat_report(runs: list[dict], repeat: int) -> dict:
    """One report for runs of ``--repeat``: per-status counts, values, wall-time statistics, and every run."""
    counts: dict[str, int] = {}
    for run in runs:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    failed = next((run for run in runs if run["status"] != "ok"), None)
    if failed is None and len(runs) < repeat:
        failed = {"status": "cancelled", "exit_code": None}
    walls = [run["wall_seconds"] for run in runs] or [0.0]
    return {
        "status": "ok" if failed is None else failed["status"],
        "exit_code": 0 if failed is None else failed["exit_code"],
        "repeat": repeat,
        "status_counts": counts,
        "wall_seconds": {
            "min": min(walls),
            "median": round(statistics.median(walls), 3),
            "max": max(walls),
            "total": round(sum(walls), 3),
        },
        "values": [run["value"] for run in runs],
        "runs": runs,
    }


def _exit_code(report: dict) -> int:
    status, code = report["status"], report["exit_code"]
    if status == "ok":
        return 0
    if status == "timeout":
        return 124
    if status == "cancelled":
        return 130
    if code is None or code == 0:
        return 1
    return code if code > 0 else 128 - code


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see :mod:`newton.examples.headless`.

    Returns:
        The process exit status.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--child"]:
        _child(argv[1])
    runner, script, script_args = _split(argv)
    parser = _parser()
    options = parser.parse_args([*runner, *([] if script is None else [script])])
    if options.frames is not None and options.frames < 0:
        parser.error("--frames must be non-negative")
    if options.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not math.isfinite(options.progress) or options.progress < 0:
        parser.error("--progress must be a non-negative number of seconds")
    for text in options.overrides:
        if "=" not in text or not text.partition("=")[0].strip():
            parser.error(f"--set takes NAME=VALUE, got {text!r}")
    cancel = threading.Event()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, lambda *_: cancel.set())
    runs = []
    try:
        for run in range(options.repeat):
            if cancel.is_set():
                break
            prefix = f"headless: run {run + 1}/{options.repeat}: " if options.repeat > 1 else "headless: "
            if options.repeat > 1:
                print(f"{prefix}starting", file=sys.stderr, flush=True)
            report = run_headless(
                options.script,
                script_args,
                frames=options.frames,
                call=options.call,
                timeout=options.timeout,
                example_class=options.example_class,
                render=options.render,
                tail=options.tail,
                echo=True,
                cancel=cancel,
                overrides=dict(_parse_override(text, run) for text in options.overrides),
                progress=options.progress or None,
            )
            runs.append({"run": run, **report} if options.repeat > 1 else report)
            if options.repeat > 1:
                print(_summary(report, prefix), file=sys.stderr, flush=True)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    report = runs[0] if options.repeat == 1 else _repeat_report(runs, options.repeat)
    text = json.dumps(report, indent=2)
    if options.json is None:
        print(text, flush=True)
    else:
        temporary = options.json.with_name(options.json.name + ".tmp")
        temporary.write_text(text + "\n")
        os.replace(temporary, options.json)
    if options.repeat == 1:
        print(_summary(report), file=sys.stderr, flush=True)
    else:
        walls = report["wall_seconds"]
        counts = ", ".join(f"{count} {status}" for status, count in report["status_counts"].items())
        print(
            f"headless: {len(runs)} of {options.repeat} runs: {counts}; wall {walls['min']:.1f}-{walls['max']:.1f} s "
            f"(median {walls['median']:.1f} s)",
            file=sys.stderr,
            flush=True,
        )
    return _exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
