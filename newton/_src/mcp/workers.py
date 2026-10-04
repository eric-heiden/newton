# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Concurrent trusted execution on sibling application processes."""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from . import shipping
from .transport import SimulationClient

_MISSING = object()
_RESTART = object()
_SERVE = "result = __import__('newton._src.mcp.shipping', fromlist=['serve']).serve(globals(), {request!r})"
_SERVE_FILE = (
    "result = __import__('newton._src.mcp.shipping', fromlist=['serve']).serve("
    "globals(), __import__('pickle').loads(__import__('pathlib').Path({path!r}).read_bytes()))"
)
_DEVICE_CHECK = (
    "_newton_error = __import__('newton._src.mcp.shipping', fromlist=['device_error']).device_error()\n"
    "if _newton_error is not None:\n"
    "    raise RuntimeError(_newton_error)"
)
_DEFAULT_BOUND = frozenset(
    (
        "session",
        "model",
        "solver",
        "state",
        "state_next",
        "control",
        "contacts",
        "viewer",
        "collision_pipeline",
        "np",
        "wp",
        "newton",
        "show",
        "rollout",
        "health",
        "solver_contacts",
        "solver_params",
        "render",
        "contacts_between",
        "swap_solver",
        "persist",
        "persist_source",
        "diff_model",
        "example",
        "module",
        "recapture",
        "fresh",
        "workers",
        "jobs",
    )
)


class WorkerError(RuntimeError):
    """Python code raised an exception on a worker session; the message includes its traceback lines."""

    def __init__(self, message: str, *, worker: int | None = None, report: dict | None = None):
        super().__init__(message)
        self.worker = worker
        self.report = report or {}


class _Lost(Exception):
    """The worker process exited or its connection broke."""


class _Outcome:
    """Response of one worker request; decoded on the thread that collects it."""

    __slots__ = ("data", "device", "error", "slot", "stdout")

    def __init__(self, slot: int, result: dict, stdout: str):
        self.slot = slot
        self.stdout = stdout
        self.error = result.get("error")
        self.data = shipping.get(result["value"], remove=True) if "value" in result else None
        self.device = result.get("device")
        if self.device is not None:
            # The call ran on a failing CUDA context, so its value cannot be trusted.
            message = f"the worker's CUDA context failed during this call ({self.device})"
            if self.error is not None:
                message += f"; the call raised {self.error.get('type')}: {self.error.get('message')}"
            self.error = {"type": "RuntimeError", "message": message, "frames": (self.error or {}).get("frames", [])}


class _Setup:
    """A broadcast or sync request that a restarted worker replays."""

    def __init__(self, seq: int, label: str, request: dict, files: list[dict], name: str | None = None):
        self.seq = seq
        self.label = label
        self.request = request
        self.files = files
        self.name = name


class _Task:
    def __init__(self, label: str, run: Callable, future: Future, setup: _Setup | None = None):
        self.label = label
        self.run = run
        self.future = future
        self.setup = setup


class _Worker:
    def __init__(self, slot: int, connection: Path | None = None):
        self.slot = slot
        self.connection = connection
        self.process: subprocess.Popen | None = None
        self.client: SimulationClient | None = None
        self.private: collections.deque[_Task] = collections.deque()
        self.applied = 0
        self.replayed: dict[int, Any] = {}
        self.state = "starting"
        self.reason: str | None = None
        self.retire = False
        self.task: _Task | None = None
        self.thread: threading.Thread | None = None
        self.restarts = 0
        self.can_rebuild = False
        self.build = 0
        self.ready = threading.Event()


class WorkerFuture(Future):
    """Future of a worker call whose result is unpickled on the thread that collects it."""

    def __init__(self, pool: WorkerPool, shipment: shipping.Shipment | None, *, echo: bool = True):
        super().__init__()
        self._pool = pool
        self._shipment = shipment
        self._echo = echo
        self._decoded: Any = _MISSING
        self._failure: BaseException | None = None
        self.worker: int | None = None
        """Worker that ran (or runs) the call."""
        self.build: int | None = None
        """Pool build (see :attr:`WorkerPool.build`) of the worker's scene when the call started."""

    def result(self, timeout: float | None = None) -> Any:
        outcome = super().result(timeout)
        if self._failure is None and self._decoded is _MISSING:
            try:
                self._decoded = self._pool._decode(outcome, self._shipment, echo=self._echo)
            except BaseException as error:
                self._failure = error
        if self._failure is not None:
            raise self._failure
        return self._decoded

    def exception(self, timeout: float | None = None) -> BaseException | None:
        error = super().exception(timeout)
        if error is not None:
            return error
        try:
            self.result()
        except BaseException as failure:
            return failure
        return None


class WorkerPool:
    """Run Python functions and cells concurrently on sibling live sessions.

    Each worker is another instance of the same application with its own scene
    and persistent Python workspace. Functions defined in trusted-execution
    cells (including lambdas and closures) are sent by source, together with
    the cell functions and classes they use and the session globals they read;
    arguments, those global values, and results are pickled (Warp arrays travel
    as NumPy data). Globals that name a session's own live objects
    (``example``, ``model``, ``state``, ...) are not sent: on a worker they
    refer to the worker's scene. Code strings run as cells on the worker, with
    the call's argument bound to ``args`` and ``result`` (or the last
    expression) returned. Text a call prints is echoed with a ``[worker i]``
    prefix when its result is collected. A call that raises rolls its
    worker's simulation back like any failed cell, so later calls on that
    worker are unaffected.

    A pool created by :meth:`launch` owns its worker processes: they follow
    :meth:`rebuild`, a worker whose process exits or whose CUDA context fails
    is restarted with earlier :meth:`broadcast` and :meth:`sync` calls
    replayed in order, and :meth:`resize` changes the worker count. Such events
    are collected by :meth:`drain_events`. A pool attached to existing
    sessions through connection files has a fixed size and cannot restart them.

    Args:
        connection_files: Connection descriptors of existing worker sessions.
        timeout: Maximum queue waiting time per request [s].
    """

    startup_timeout = 900.0
    """Longest time a launched worker may take to start (kernel compilation included) [s]."""

    history_limit = 64
    """Broadcast and sync calls kept for replay on restarted workers (oldest dropped first)."""

    def __init__(self, connection_files: list[str | Path] | None = None, *, timeout: float = 300.0):
        self.timeout = timeout
        self.max_count = len(connection_files or [])
        """Upper bound for :meth:`resize`."""
        self.argv: list[str] = []
        self.overrides: dict = {}
        """Module-global overrides that launched workers are started with (see ``newton_rebuild``)."""
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._workers: list[_Worker] = []
        self._shared: collections.deque[_Task] = collections.deque()
        self._history: list[_Setup] = []
        self._seq = 0
        self._events: list[str] = []
        self._synced: dict[str, Any] = {}
        self._closed = False
        self._launch: dict | None = None
        self._rebuild_arguments: dict = {}
        self.build = 0
        """Number of :meth:`rebuild` calls; a worker's scene is at this build once it has followed the last one."""
        self._namespace: Callable[[], dict | None] = lambda: None
        self._bound: Callable[[], Iterable[str]] = lambda: _DEFAULT_BOUND
        self._exchange = Path(tempfile.mkdtemp(prefix="newton-mcp-workers-"))
        for path in connection_files or []:
            self._add(_Worker(len(self._workers), Path(path)))

    @classmethod
    def launch(
        cls,
        script: str | Path,
        argv: list[str] | None = None,
        *,
        count: int,
        max_count: int | None = None,
        example_class: str = "Example",
        directory: str | Path | None = None,
        name: str = "session",
        timeout: float = 300.0,
        overrides: dict | None = None,
    ) -> WorkerPool:
        """Start ``count`` worker processes that host ``script`` (``python -m newton.mcp host``).

        Workers start in the background; :meth:`wait_ready` waits for them. Processes reuse
        :data:`sys.executable` and inherit the environment and working directory.

        Args:
            script: Example script the workers host.
            argv: Example arguments.
            count: Initial number of workers.
            max_count: Upper bound for :meth:`resize` (default ``count``).
            example_class: Example class name inside the script.
            directory: Directory for connection, log, and data-exchange files (default: a new temporary one).
            name: Prefix of those files.
            timeout: Maximum queue waiting time per request [s].
            overrides: Module globals the workers set before constructing the example.
        """
        pool = cls(timeout=timeout)
        directory = Path(directory) if directory is not None else pool._exchange
        if directory != pool._exchange:
            shutil.rmtree(pool._exchange, ignore_errors=True)
            pool._exchange = directory / f"{name}.workers"
            pool._exchange.mkdir(parents=True, exist_ok=True)
        pool._launch = {"script": Path(script).resolve(), "class": example_class, "directory": directory, "name": name}
        pool.argv = list(argv or [])
        pool.overrides = dict(overrides or {})
        pool.max_count = max(count, max_count if max_count is not None else count)
        for _ in range(count):
            pool._add(_Worker(pool._free_slot()))
        return pool

    # ------------------------------------------------------------------
    # Public API

    def __len__(self) -> int:
        return self.count

    @property
    def count(self) -> int:
        """Number of workers that are running or starting."""
        with self._lock:
            return len(self._active())

    def status(self) -> list[dict]:
        """State of every worker: ``worker``, ``state`` (starting/ready/failed), ``pid``, ``restarts``, ``running``,
        and ``build`` (the last :meth:`rebuild` its scene followed; see :attr:`build`)."""
        with self._lock:
            return [
                {
                    "worker": worker.slot,
                    "state": worker.state,
                    "pid": worker.process.pid if worker.process is not None else None,
                    "restarts": worker.restarts,
                    "running": worker.task.label if worker.task is not None else None,
                    "build": worker.build,
                }
                for worker in self._workers
                if not worker.retire
            ]

    def submit(self, function: Callable | str, *args: Any, **kwargs: Any) -> WorkerFuture:
        """Queue one call on the next idle worker and return a future for its result.

        Args:
            function: Function (sent by source) or code string; for a code string the first
                positional argument (or ``arguments=``) is bound to ``args``.
            *args: Positional arguments of the call.
            **kwargs: Keyword arguments of the call.
        """
        if isinstance(function, str):
            value = args[0] if args else kwargs.get("arguments", None)
            return self._submit(function, [value])[0]
        return self._submit(function, [(args, kwargs)])[0]

    def map(self, function: Callable | str, *iterables: Iterable, timeout: float | None = None) -> list[Any]:
        """Call ``function`` once per item (``function(a, b)`` for ``map(function, as_, bs)``), spread over idle workers.

        A code string runs once per item of the single iterable, with the item bound to ``args``.

        Returns:
            Results in input order. A failed call yields ``{"error": message}`` instead of raising,
            so one bad candidate does not discard the others.
        """
        if isinstance(function, str):
            if len(iterables) != 1:
                raise ValueError("A code string maps over exactly one iterable (bound to args)")
            calls = list(iterables[0])
        else:
            calls = [(args, {}) for args in zip(*iterables, strict=False)]
        futures = self._submit(function, calls)
        results = []
        for future in futures:
            try:
                results.append(future.result(timeout))
            except WorkerError as error:
                results.append({"error": str(error)[:2000]})
            except Exception as error:
                results.append({"error": f"{type(error).__name__}: {str(error)[:2000]}"})
        return results

    def broadcast(self, function: Callable | str, *args: Any, **kwargs: Any) -> list[Any]:
        """Run a function or code string once on every worker concurrently, e.g. to define helpers or load data.

        Restarted and added workers replay earlier broadcasts in order.

        Returns:
            One result per worker, in worker order. Errors are raised.
        """
        if isinstance(function, str):
            call = args[0] if args else kwargs.get("arguments", _MISSING)
        else:
            call = (args, kwargs)
        request, files, shipment = self._request(function, call, persistent=True)
        label = "broadcast code" if isinstance(function, str) else f"broadcast {shipment.label}"
        setup = self._record(label, request, files)
        outcomes = self._run_setup(setup)
        results, errors = [], []
        for outcome in outcomes:
            try:
                results.append(self._decode(outcome, shipment))
            except BaseException as error:
                errors.append(error)
                results.append(None)
        if errors:
            if len(errors) == len(outcomes):
                self._forget(setup)
            raise errors[0]
        return results

    def sync(self, *names: str, **values: Any) -> dict:
        """Copy session values (NumPy arrays, dicts, ...) into every worker's globals.

        Positional arguments name globals of the calling session; keyword arguments give values.
        Functions sent later do not resend a synced global while the session still binds the same
        object. Restarted and added workers replay syncs.

        Returns:
            ``names``, pickled ``bytes``, and the number of ``workers`` updated.
        """
        namespace = self._namespace() or {}
        items = {}
        for name in names:
            if name not in namespace:
                raise NameError(f"{name!r} is not defined in this session")
            items[name] = namespace[name]
        items.update(values)
        bound = set(self._bound())
        for name in items:
            if not name.isidentifier():
                raise ValueError(f"{name!r} is not a valid Python name")
            if name in bound:
                raise ValueError(f"{name!r} names each worker's own live object and is rebound before every call")
        total, setups = 0, []
        for name, value in items.items():
            found: dict[int, Any] = {}
            data = shipping.dumps(value, found)
            total += len(data)
            path = self._exchange / f"sync-{name}-{len(data)}-{time.monotonic_ns()}.pkl"
            path.write_bytes(data)
            files = [{"file": str(path)}]
            ship = None
            if found:
                shipment = shipping.prepare(None, self._skip, found.values())
                ship = {"key": shipment.key, **self._put(shipment.data, "ship", persistent=True)}
                files.append(ship)
            request = {"exchange": str(self._exchange), "sync": {name: {"file": str(path)}}, "ship": ship}
            setups.append(self._record(f"sync {name}", request, files, name=name))
        updated = None
        for setup in setups:
            outcomes = self._run_setup(setup)
            for outcome in outcomes:
                self._decode(outcome, None)
            updated = len(outcomes) if updated is None else min(updated, len(outcomes))
        self._synced.update(items)
        return {"names": sorted(items), "bytes": total, "workers": updated or 0}

    def resize(self, count: int, *, wait: bool = True) -> dict:
        """Change the number of workers at run time, within ``max_count``.

        New workers replay earlier broadcast and sync calls before they take work. Removed workers
        finish their running call first.

        Args:
            count: New number of workers.
            wait: Wait until new workers are ready (or failed to start).

        Returns:
            ``count``, ``max_count``, and :meth:`status`.
        """
        if self._launch is None:
            raise ValueError("This pool attaches to existing sessions; its size is fixed")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.max_count:
            raise ValueError(f"count must be an integer in [0, {self.max_count}]")
        added = []
        with self._condition:
            active = self._active()
            for worker in active[count:]:
                worker.retire = True
            for _ in range(count - len(active)):
                worker = _Worker(self._free_slot())
                self._add(worker)
                added.append(worker)
            self._condition.notify_all()
        if count < len(active):
            self._fail_if_starved()
        if wait:
            for worker in added:
                worker.ready.wait(self.startup_timeout)
        return {"count": self.count, "max_count": self.max_count, "workers": self.status()}

    def restart(self, worker: int | None = None, *, wait: bool = True) -> list[dict]:
        """Restart one worker (or all) in a fresh process; the running call on it fails.

        Returns:
            :meth:`status` after the restart.
        """
        stale = []
        with self._condition:
            targets = [w for w in self._workers if not w.retire and (worker is None or w.slot == worker)]
            if worker is not None and not targets:
                raise ValueError(f"No worker {worker}")
            for target in targets:
                if target.state == "starting":
                    continue
                target.ready.clear()
                restart_thread = target.state == "failed"
                # An empty reason restarts without an event: the caller asked for it.
                target.state, target.reason = "starting", ""
                if restart_thread:
                    self._start_thread(target)
                elif target.process is not None:
                    stale.append(target.process)
            self._condition.notify_all()
        # Only the processes seen above: the worker threads spawn their replacements concurrently.
        for process in stale:
            _terminate(process)
        if wait:
            for target in targets:
                target.ready.wait(self.startup_timeout)
        return self.status()

    def wait_ready(self, timeout: float | None = None) -> list[dict]:
        """Wait until every worker has started (or failed to); returns :meth:`status`."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for worker in list(self._workers):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            worker.ready.wait(remaining)
        return self.status()

    def rebuild(self, arguments: dict | None = None, *, timeout: float = 60.0) -> dict:
        """Rebuild every worker's scene with ``arguments`` (as ``newton_rebuild``), waiting up to ``timeout``.

        Idle workers are waited for. A worker that is running a call or still starting rebuilds after it,
        before any queued call, and is listed as ``pending`` at once; pending rebuilds and rebuilds that
        finish after ``timeout`` are reported by :meth:`drain_events`. Later restarts use the same arguments.
        Each rebuild increments :attr:`build`; calls report the build their worker had (``jobs``).

        Returns:
            ``build``, ``rebuilt`` count, ``seconds``, ``pending`` workers, and ``errors``.
        """
        arguments = {k: v for k, v in (arguments or {}).items() if k != "restart"}
        started = time.perf_counter()
        futures, busy = [], []
        with self._condition:
            self.build += 1
            build = self.build
            if arguments.get("argv") is not None:
                self.argv = list(arguments["argv"])
            if arguments.get("overrides") is not None:
                self.overrides = dict(arguments["overrides"])
            # Restarted workers get the arguments and overrides on their command line.
            self._rebuild_arguments = {
                k: v for k, v in arguments.items() if k not in ("argv", "overrides", "reset_namespace")
            }
            if arguments.get("reset_namespace"):
                for setup in list(self._history):
                    self._forget(setup)
                self._synced.clear()
            for worker in self._active():
                if self._launch is None and not worker.can_rebuild:
                    continue
                future: Future = Future()
                worker.private.append(
                    _Task("rebuild", lambda w, a=arguments, b=build: self._rebuild_worker(w, a, b), future)
                )
                # Waiting for a running call (a background job, say) would stall this response.
                if worker.task is not None or worker.state == "starting":
                    busy.append((worker.slot, future))
                else:
                    futures.append((worker.slot, future))
            self._condition.notify_all()
        rebuilt, errors, pending = 0, [], []
        deadline = time.monotonic() + timeout
        for slot, future in futures:
            try:
                future.result(max(0.0, deadline - time.monotonic()))
                rebuilt += 1
            except TimeoutError:
                busy.append((slot, future))
            except Exception as error:
                errors.append(f"worker {slot}: {type(error).__name__}: {str(error)[:500]}")
        for slot, future in sorted(busy):
            pending.append(slot)
            future.add_done_callback(lambda f, s=slot: self._late_rebuild(s, f, started))
        return {
            "build": build,
            "rebuilt": rebuilt,
            "seconds": round(time.perf_counter() - started, 2),
            **({"pending": pending} if pending else {}),
            **({"errors": errors} if errors else {}),
        }

    def _rebuild_worker(self, worker: _Worker, arguments: dict, build: int) -> dict:
        result = self._operation(worker, "rebuild", **arguments)
        worker.build = max(worker.build, build)
        return result

    def drain_events(self) -> list[str]:
        """Worker restarts, start failures, and late rebuilds since the last call."""
        with self._lock:
            events, self._events = self._events, []
        return events

    def close(self) -> None:
        """Stop all workers (terminating launched processes by PID) and fail calls that have not run."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            tasks = list(self._shared)
            self._shared.clear()
            for worker in self._workers:
                tasks.extend(worker.private)
                worker.private.clear()
            self._condition.notify_all()
        for task in tasks:
            if task.future.set_running_or_notify_cancel():
                task.future.set_exception(RuntimeError("Worker pool closed"))
        for worker in list(self._workers):
            self._kill(worker)
        for worker in list(self._workers):
            if worker.thread is not None and worker.thread is not threading.current_thread():
                worker.thread.join(timeout=10)
        shutil.rmtree(self._exchange, ignore_errors=True)

    def attach(self, namespace: Callable[[], dict | None], bound: Callable[[], Iterable[str]]) -> None:
        """Bind the calling session: its workspace (for :meth:`sync` names and results) and the names not sent."""
        self._namespace = namespace
        self._bound = bound

    # ------------------------------------------------------------------
    # Scheduling

    def _active(self) -> list[_Worker]:
        return [w for w in self._workers if not w.retire and w.state in ("starting", "ready")]

    def _free_slot(self) -> int:
        used = {w.slot for w in self._workers if not w.retire}
        return next(i for i in range(len(used) + 1) if i not in used)

    def _add(self, worker: _Worker) -> None:
        self._workers.append(worker)
        self._workers.sort(key=lambda w: w.slot)
        self._start_thread(worker)

    def _start_thread(self, worker: _Worker) -> None:
        worker.thread = threading.Thread(
            target=self._loop, args=(worker,), name=f"newton-mcp-worker-{worker.slot}", daemon=True
        )
        worker.thread.start()

    def _loop(self, worker: _Worker) -> None:
        while True:
            with self._lock:
                if worker.retire or self._closed:
                    break
                starting = worker.state == "starting"
            if starting:
                self._start(worker)
                if worker.state != "ready":
                    if worker.retire or self._closed:
                        self._stop(worker)
                    return
            task = self._next(worker)
            if task is None:
                break
            if task is _RESTART:
                continue
            self._execute(worker, task)
        self._stop(worker)

    def _next(self, worker: _Worker) -> Any:
        with self._condition:
            while True:
                if worker.retire or self._closed:
                    return None
                if worker.state == "starting":
                    return _RESTART
                # Marked busy under the lock, so rebuild() sees calls that are about to start.
                if worker.private:
                    worker.task = worker.private.popleft()
                    return worker.task
                if self._shared:
                    worker.task = self._shared.popleft()
                    return worker.task
                self._condition.wait()

    def _execute(self, worker: _Worker, task: _Task) -> None:
        try:
            self._run_task(worker, task)
        finally:
            worker.task = None

    def _run_task(self, worker: _Worker, task: _Task) -> None:
        setup = task.setup
        if setup is not None and worker.applied >= setup.seq:
            # Already applied while the worker replayed the history after a restart.
            if task.future.set_running_or_notify_cancel():
                task.future.set_result(worker.replayed.get(setup.seq))
            return
        if not task.future.set_running_or_notify_cancel():
            return
        worker.task = task
        if isinstance(task.future, WorkerFuture):
            task.future.worker = worker.slot
            task.future.build = worker.build
        try:
            result = task.run(worker)
            if setup is not None:
                worker.applied = max(worker.applied, setup.seq)
            task.future.set_result(result)
            if isinstance(result, _Outcome) and result.device is not None:
                self._lost(worker, f"CUDA context failed during {task.label} ({result.device})")
        except _Lost as lost:
            task.future.set_exception(RuntimeError(f"Worker {worker.slot} {lost} while running {task.label}"))
            self._lost(worker, f"{lost} while running {task.label}")
        except BaseException as error:
            task.future.set_exception(error)
            if self._launch is not None:
                self._check_device(worker, task)

    def _check_device(self, worker: _Worker, task: _Task) -> None:
        try:
            self._operation(worker, "execute", code=_DEVICE_CHECK)
        except _Lost as lost:
            self._lost(worker, f"{lost} after {task.label}")
        except Exception as error:
            # Keep the exception line, not the worker's notes about scene validity and frames.
            first = re.split(r"\. (?:The simulation|No simulation|Nothing was|Rolling)", str(error).strip())[0]
            first = first.splitlines()[0][:300] if first else ""
            self._lost(worker, f"failed a device check after {task.label} ({first or type(error).__name__})")

    def _lost(self, worker: _Worker, reason: str) -> None:
        with self._condition:
            if self._closed or worker.retire:
                return
            if worker.state == "starting":
                return
            if self._launch is None:
                worker.state = "failed"
                tasks = list(worker.private)
                worker.private.clear()
                self._events.append(f"worker {worker.slot} {reason}; attached workers cannot be restarted")
            else:
                worker.state, worker.reason, tasks = "starting", reason, []
            worker.ready.clear()
            self._condition.notify_all()
        for task in tasks:
            if task.future.set_running_or_notify_cancel():
                task.future.set_exception(RuntimeError(f"Worker {worker.slot} is no longer available"))
        if self._launch is not None:
            self._kill(worker)
        self._fail_if_starved()

    def _fail_if_starved(self) -> None:
        with self._condition:
            if self._active() or not self._shared:
                return
            tasks = list(self._shared)
            self._shared.clear()
        for task in tasks:
            if task.future.set_running_or_notify_cancel():
                task.future.set_exception(RuntimeError("No worker session is running (see workers.status())"))

    def _stop(self, worker: _Worker) -> None:
        self._kill(worker)
        with self._condition:
            tasks = list(worker.private)
            worker.private.clear()
            if worker.retire and worker in self._workers:
                self._workers.remove(worker)
            worker.ready.set()
        for task in tasks:
            if task.future.set_running_or_notify_cancel():
                task.future.set_exception(RuntimeError(f"Worker {worker.slot} was removed"))

    # ------------------------------------------------------------------
    # Processes

    def _start(self, worker: _Worker) -> None:
        started = time.perf_counter()
        reason, worker.reason = worker.reason, None
        # Read before the process starts with the current arguments; a later rebuild queues its own task.
        build = self.build
        try:
            if self._launch is not None:
                self._spawn(worker)
            else:
                worker.client = SimulationClient(worker.connection, timeout=self.timeout)
            worker.can_rebuild = bool(self._operation(worker, "describe")["capabilities"].get("rebuild"))
            if self._launch is not None and self._rebuild_arguments:
                self._operation(worker, "rebuild", **self._rebuild_arguments)
            if self._launch is not None:
                # Started with the arguments and overrides of the latest rebuild.
                worker.build = max(worker.build, build)
            replayed, failures = self._replay(worker)
            with self._condition:
                if worker.state == "starting":
                    worker.state = "ready"
                self._condition.notify_all()
            if reason is not None:
                worker.restarts += 1
            if reason:
                text = f"worker {worker.slot} {reason}; restarted in {time.perf_counter() - started:.1f} s"
                if replayed:
                    text += f", replayed {replayed} broadcast/sync call{'s' if replayed > 1 else ''}"
                if failures:
                    text += f" ({len(failures)} failed: {'; '.join(failures)[:600]})"
                self._event(text)
        except Exception as error:
            with self._condition:
                worker.state = "failed"
                self._condition.notify_all()
            self._kill(worker)
            if not (self._closed or worker.retire):
                self._event(f"worker {worker.slot} failed to start: {str(error)[:1500]}")
        finally:
            worker.ready.set()
            self._fail_if_starved()

    def _spawn(self, worker: _Worker) -> None:
        launch = self._launch
        connection = self._worker_file(worker, ".json")
        ready = self._worker_file(worker, ".ready")
        log = self._worker_file(worker, ".log")
        self._kill(worker)
        # A crashed worker leaves its files behind; the new server refuses an existing connection file.
        connection.unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "newton.mcp",
            "host",
            str(launch["script"]),
            "--connection-file",
            str(connection),
            "--class",
            launch["class"],
            "--ready-file",
            str(ready),
            "--parent-pid",
            str(os.getpid()),
            *(["--overrides", json.dumps(self.overrides)] if self.overrides else []),
            "--",
            *self.argv,
        ]
        with open(log, "ab") as output:
            output.write(f"\n--- worker {worker.slot} start {time.strftime('%H:%M:%S')}\n".encode())
            output.flush()
            worker.process = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
        worker.connection = connection
        deadline = time.monotonic() + self.startup_timeout
        while not ready.exists():
            code = worker.process.poll()
            if code is not None:
                raise RuntimeError(f"exited with code {code} during startup; log {log}:\n{_tail(log)}")
            if self._closed or worker.retire:
                raise RuntimeError("stopped during startup")
            if time.monotonic() > deadline:
                raise TimeoutError(f"not ready after {self.startup_timeout:.0f} s; log {log}:\n{_tail(log)}")
            time.sleep(0.05)
        worker.client = SimulationClient(connection, timeout=self.timeout)

    def _kill(self, worker: _Worker) -> None:
        process = worker.process
        if process is None:
            return
        _terminate(process)
        # A terminated server leaves its files behind; a reused slot may already hold a newer process's files.
        ready = self._worker_file(worker, ".ready")
        if ready is not None and _written_by(ready, process.pid):
            self._worker_file(worker, ".json").unlink(missing_ok=True)
            ready.unlink(missing_ok=True)

    def _worker_file(self, worker: _Worker, suffix: str) -> Path | None:
        if self._launch is None:
            return None
        return self._launch["directory"] / f"{self._launch['name']}.worker-{worker.slot}{suffix}"

    def _replay(self, worker: _Worker) -> tuple[int, list[str]]:
        with self._lock:
            history = list(self._history)
        worker.applied, worker.replayed, failures = 0, {}, []
        for setup in history:
            try:
                outcome = self._call(worker, setup.request)
                if outcome.device is not None:
                    raise _Lost(f"CUDA context failed while replaying {setup.label} ({outcome.device})")
                if outcome.error is not None:
                    failures.append(f"{setup.label}: {outcome.error.get('type')}: {outcome.error.get('message')}")
            except _Lost:
                raise
            except Exception as error:
                outcome = None
                failures.append(f"{setup.label}: {type(error).__name__}: {error}")
            worker.replayed[setup.seq] = outcome
            worker.applied = setup.seq
        return len(history), failures

    def _operation(self, worker: _Worker, operation: str, **arguments: Any) -> dict:
        client = worker.client
        if client is None:
            raise _Lost("has no connection")
        try:
            if worker.process is None:
                return client.request(operation, **arguments)
            response = client._send(operation, arguments)
        except (OSError, ConnectionError) as error:
            raise _Lost(self._exit(worker) or f"lost its connection ({type(error).__name__})") from None
        if "error" in response:
            message = str(response["error"].get("message", ""))
            exit_reason = self._exit(worker)
            if exit_reason is not None or "Session closed" in message:
                raise _Lost(exit_reason or "closed its session")
            raise RuntimeError(message)
        return response["result"]

    def _exit(self, worker: _Worker) -> str | None:
        process = worker.process
        if process is None:
            return None
        try:
            code = process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            return None
        return f"was killed by signal {-code}" if code < 0 else f"exited with code {code}"

    def _call(self, worker: _Worker, request: dict) -> _Outcome:
        code = _SERVE.format(request=request)
        path = None
        if len(code) > 60000:
            path = self._exchange / f"request-{time.monotonic_ns()}-{worker.slot}.pkl"
            import pickle  # noqa: PLC0415

            path.write_bytes(pickle.dumps(request))
            code = _SERVE_FILE.format(path=str(path))
        try:
            response = self._operation(worker, "execute", code=code)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
        return _Outcome(worker.slot, response.get("result") or {}, response.get("stdout") or "")

    def _event(self, text: str) -> None:
        with self._lock:
            self._events.append(text)

    def _late_rebuild(self, slot: int, future: Future, started: float) -> None:
        error = future.exception()
        seconds = time.perf_counter() - started
        if error is None:
            self._event(f"worker {slot} rebuilt {seconds:.1f} s after the rebuild request (it was busy)")
        else:
            self._event(f"worker {slot} rebuild failed: {type(error).__name__}: {str(error)[:500]}")

    # ------------------------------------------------------------------
    # Requests

    def _skip(self, name: str, value: Any) -> bool:
        if name in set(self._bound()):
            return True
        return self._synced.get(name, _MISSING) is value

    def _put(self, data: bytes, prefix: str, *, persistent: bool = False) -> dict:
        if persistent and len(data) * 4 // 3 > shipping.INLINE_LIMIT:
            path = self._exchange / f"{prefix}-{time.monotonic_ns()}.pkl"
            path.write_bytes(data)
            return {"file": str(path)}
        return shipping.put(data, self._exchange, prefix)

    def _request(self, function: Callable | str, call: Any, *, persistent: bool = False):
        """Request dict, data files it owns, and the shipment for one call (``call`` is pickled)."""
        found: dict[int, Any] = {}
        call_data = None if call is _MISSING else shipping.dumps(call, found)
        code = function if isinstance(function, str) else None
        shipment = None
        if code is None or found:
            shipment = shipping.prepare(None if code is not None else function, self._skip, found.values())
        files = []
        ship = None
        if shipment is not None:
            ship = {"key": shipment.key, **self._put(shipment.data, "ship", persistent=persistent)}
            files.append(ship)
        call_reference = None
        if call_data is not None:
            call_reference = self._put(call_data, "call", persistent=persistent)
            files.append(call_reference)
        request = {"exchange": str(self._exchange), "ship": ship, "code": code, "call": call_reference}
        return request, files, shipment

    def _submit(self, function: Callable | str, calls: list, *, progress: list | None = None, echo=True):
        if self._closed:
            raise RuntimeError("Worker pool closed")
        if not self.count:
            raise RuntimeError("No worker session is running (see workers.status())")
        code = function if isinstance(function, str) else None
        found: dict[int, Any] = {}
        blobs = [shipping.dumps(call, found) for call in calls]
        shipment = None
        if code is None or found:
            shipment = shipping.prepare(None if code is not None else function, self._skip, found.values())
        ship = None
        if shipment is not None:
            ship = {"key": shipment.key, **shipping.put(shipment.data, self._exchange, "ship")}
        remaining = [len(calls)]
        guard = threading.Lock()

        def release(_future):
            with guard:
                remaining[0] -= 1
                if remaining[0] == 0:
                    shipping.discard(ship)

        futures = []
        label = "code" if code is not None else shipment.label
        tasks = []
        for index, blob in enumerate(blobs):
            reference = shipping.put(blob, self._exchange, "call")
            request = {
                "exchange": str(self._exchange),
                "ship": ship,
                "code": code,
                "call": reference,
                "progress": progress[index] if progress else None,
            }
            future = WorkerFuture(self, shipment, echo=echo)

            def run(worker, request=request, reference=reference):
                try:
                    return self._call(worker, request)
                finally:
                    shipping.discard(reference)

            future.add_done_callback(release)
            tasks.append(_Task(label, run, future))
            futures.append(future)
        with self._condition:
            if not self._active():
                raise RuntimeError("No worker session is running (see workers.status())")
            self._shared.extend(tasks)
            self._condition.notify_all()
        return futures

    def _record(self, label: str, request: dict, files: list[dict], name: str | None = None) -> _Setup:
        with self._lock:
            self._seq += 1
            setup = _Setup(self._seq, label, request, files, name)
            if name is not None:
                for old in [s for s in self._history if s.name == name]:
                    self._forget(old)
            self._history.append(setup)
            if len(self._history) > self.history_limit:
                self._event(
                    f"more than {self.history_limit} broadcast/sync calls; restarted and added workers replay "
                    f"only the latest {self.history_limit} (dropped: {self._history[0].label})"
                )
            while len(self._history) > self.history_limit:
                self._forget(self._history[0])
        return setup

    def _forget(self, setup: _Setup) -> None:
        with self._lock:
            if setup in self._history:
                self._history.remove(setup)
        for reference in setup.files:
            shipping.discard(reference)

    def _run_setup(self, setup: _Setup) -> list[_Outcome]:
        futures = []
        with self._condition:
            workers = self._active()
            if not workers and self._launch is None:
                raise RuntimeError("No worker session is running (see workers.status())")
            # Without workers, a launched pool only records the call for workers it starts later.
            for worker in workers:
                future: Future = Future()
                worker.private.append(_Task(setup.label, lambda w: self._call(w, setup.request), future, setup))
                futures.append(future)
            self._condition.notify_all()
        outcomes = []
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    def _decode(self, outcome: _Outcome, shipment: shipping.Shipment | None, *, echo: bool = True) -> Any:
        if outcome is None:
            return None
        if echo and outcome.stdout:
            for line in outcome.stdout.rstrip("\n").splitlines():
                print(f"[worker {outcome.slot}] {line}")
        if outcome.error is not None:
            raise WorkerError(
                _error_text(outcome.error, outcome.slot, shipment), worker=outcome.slot, report=outcome.error
            )
        return shipping.loads(outcome.data, self._namespace())


def _terminate(process: subprocess.Popen) -> None:
    """Stop one worker process by its PID: SIGTERM, then SIGKILL after 10 s."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _written_by(ready: Path, pid: int) -> bool:
    """Whether the ready marker ``ready`` was written by process ``pid``."""
    try:
        return json.loads(ready.read_text()).get("pid") == pid
    except (OSError, ValueError, AttributeError):
        return False


def _tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    kept = [line for line in text.splitlines() if not line.startswith("Module ")]
    return "\n".join(kept[-lines:])


def _error_text(report: dict, slot: int, shipment: shipping.Shipment | None) -> str:
    text = f"{report.get('type', 'Error')}: {report.get('message', '')}"
    frames = []
    for frame in report.get("frames", [])[-3:]:
        filename = str(frame.get("file", ""))
        if ":cell-" in filename:
            filename = "cell " + filename.rsplit(":cell-", 1)[1].rstrip(">")
        else:
            filename = Path(filename).name if not filename.startswith("<") else filename
        frames.append(f"{filename} line {frame.get('line')} in {frame.get('function')}: {frame.get('source', '')}")
    text += f" [worker {slot}" + ("; " + " | ".join(frames) if frames else "") + "]"
    if report.get("rollback"):
        text += f" Worker {slot}: {report['rollback']}"
    name = report.get("name")
    if shipment is not None and name in shipment.not_sent:
        text += f"; the session's {name!r} was not sent with the function ({shipment.not_sent[name]})"
    return text
