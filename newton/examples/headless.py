# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run an example script in a clean headless process and report the outcome as JSON.

.. code-block:: console

    python -m newton.examples.headless SCRIPT [script args] [--frames N] [--call CODE]
        [--json OUT] [--timeout S] [--render] [--class NAME] [--tail CHARS] [-- script args]

``SCRIPT`` is a Python file that defines an ``Example(viewer, args)`` class, or
the short name of a bundled example such as ``basic_pendulum``. A new Python
process loads the script without running its ``__main__`` block, parses the
script arguments with the script's own parser, constructs the example with a
null viewer, and steps ``N`` frames (default: the script's ``--num-frames``).
It then evaluates ``CODE`` (an expression, or statements ending in one) with
``example``, ``module``, ``args``, ``newton``, ``np``, and ``wp`` in scope.
With ``--test`` among the script arguments, ``test_post_step()``,
``test_final()``, and the NaN checks run as in ``python -m newton.examples``
(after ``CODE``). ``--render`` also calls ``example.render()`` after every step.

Runner options are recognized anywhere after ``SCRIPT``; arguments after ``--``
always go to the script, for scripts that define options with the same names.

The JSON report (written to ``OUT``, or printed when ``--json`` is omitted)
contains ``status`` (``ok``, ``error``, ``timeout``, ``crashed``, or
``cancelled``), the process ``exit_code`` and terminating ``signal``,
``wall_seconds``, the last ``phase`` reached (``start``, ``load``, ``build``,
``step``, ``call``, ``test``, or ``done``), the ``frames`` stepped, per-phase
host ``seconds``, the example's ``sim_time``, the JSON-converted ``value`` of
``CODE``, the ``exception`` with its traceback, the Python ``stack`` of every
thread when a timeout stopped the process, and the last characters of
``stdout`` and ``stderr``. In ``value``, non-finite floats become ``"nan"``,
``"inf"``, or ``"-inf"``, and objects without a JSON form become their
``repr()``. A timeout kills the process and everything it started. The exit
status is 0 for ``ok``, 124 for ``timeout``, 130 for ``cancelled``, and
otherwise nonzero.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
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
import subprocess
import sys
import tempfile
import threading
import time
import traceback
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
        }
        (directory / "spec.json").write_text(json.dumps(spec))
        (directory / "frames.bin").write_bytes(bytes(8))
        # -u: output written before a kill or crash still reaches the report.
        command = [sys.executable, "-u", "-m", "newton.examples.headless", "--child", str(directory / "spec.json")]
        outcome = _supervise(command, directory, timeout=timeout, tail=tail, echo=echo, cancel=cancel)
    return {
        "script": target.get("script") or target["module"],
        "argv": argv,
        "call": call,
        "timeout": timeout,
        **outcome,
    }


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


def _supervise(command, directory: Path, *, timeout, tail, echo, cancel) -> dict:
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
    try:
        while not _exited(process):
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
    return {
        "status": status,
        "exit_code": returncode,
        "signal": name,
        "wall_seconds": round(wall, 3),
        "phase": record.get("phase", "start"),
        "frames": frames,
        "frames_requested": record.get("frames_requested"),
        "sim_time": record.get("sim_time"),
        "device": record.get("device"),
        "seconds": record.get("seconds", {}),
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
        usage="%(prog)s SCRIPT [script args] [--frames N] [--call CODE] [--json OUT] [--timeout S] [-- script args]",
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
    parser.add_argument("--json", type=Path, default=None, help="Write the report to this file instead of stdout.")
    parser.add_argument("--timeout", type=float, default=None, help="Kill the run after this many seconds.")
    parser.add_argument("--class", dest="example_class", default="Example", help="Example class name in the script.")
    parser.add_argument("--render", action="store_true", help="Also call example.render() after every step.")
    parser.add_argument("--tail", type=int, default=4000, help="Characters of stdout and stderr to report.")
    return parser


def _summary(report: dict) -> str:
    text = (
        f"headless: {report['status']} after {report['wall_seconds']:.1f} s, phase {report['phase']}, "
        f"{report['frames']} frames, exit code {report['exit_code']}"
    )
    if report.get("exception"):
        text += f"; {report['exception']['type']}: {report['exception']['message'][:300]}"
    return text


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
    cancel = threading.Event()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, lambda *_: cancel.set())
    try:
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
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    text = json.dumps(report, indent=2)
    if options.json is None:
        print(text, flush=True)
    else:
        temporary = options.json.with_name(options.json.name + ".tmp")
        temporary.write_text(text + "\n")
        os.replace(temporary, options.json)
    print(_summary(report), file=sys.stderr, flush=True)
    return _exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
