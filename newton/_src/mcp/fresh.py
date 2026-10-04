# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a hosted script in clean subprocesses through :mod:`newton.examples.headless`."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SOURCE = "script file as saved on disk when the run started; live edits in this session are not applied"
# Every report has the same keys (those of newton.examples.headless) whether or not the run launched.
_REPORT = {
    "status": None,
    "exit_code": None,
    "signal": None,
    "wall_seconds": 0.0,
    "queued_seconds": 0.0,
    "phase": "queued",
    "frames": 0,
    "frames_requested": None,
    "sim_time": None,
    "device": None,
    "seconds": {},
    "value": None,
    "exception": None,
    "stack": None,
    "stdout_tail": "",
    "stderr_tail": "",
    "source": _SOURCE,
    "script_changed_since_build": None,
}


class _Job:
    def __init__(self, argv: list[str], parallel: int):
        self.argv = argv
        self.parallel = parallel
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.queued = time.perf_counter()
        self.report: dict | None = None


class FreshRuns:
    """Handle of fresh-process runs started with ``fresh(..., wait=False)``."""

    def __init__(self, jobs: list[_Job], wake: Callable[[], None]):
        self._jobs = jobs
        self._wake = wake

    def done(self) -> bool:
        """Whether every run has finished (including failed, timed-out, and cancelled runs)."""
        return all(job.finished.is_set() for job in self._jobs)

    def result(self, timeout: float | None = None) -> list[dict]:
        """Wait for every run and return one report per argv list, in order.

        Args:
            timeout: Maximum waiting time [s]; ``None`` waits until all runs finish.

        Raises:
            TimeoutError: If runs are still going after ``timeout``.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        for job in self._jobs:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not job.finished.wait(remaining):
                raise TimeoutError(f"{self._finished()} of {len(self._jobs)} runs finished")
        return [job.report for job in self._jobs]

    def cancel(self) -> None:
        """Kill running processes and drop queued runs; their reports have status ``cancelled``."""
        for job in self._jobs:
            job.cancel.set()
        self._wake()

    def _finished(self) -> int:
        return sum(job.finished.is_set() for job in self._jobs)

    def __repr__(self) -> str:
        return f"<FreshRuns {self._finished()} of {len(self._jobs)} finished>"


class FreshRunner:
    """Queue clean-process runs of one script so at most ``parallel`` run at once.

    Each run is a new Python process (``python -m newton.examples.headless``,
    with :data:`sys.executable` and the inherited environment) that loads the
    script from disk. Processes are killed by their process group on timeout,
    on :meth:`FreshRuns.cancel`, and on :meth:`close`.

    Args:
        script: Hosted script path.
        argv: Returns the host's current example arguments (the default run).
        example_class: Example class name in the script.
        build_digest: Returns the SHA-256 of the script source the live session last built.
    """

    def __init__(
        self,
        script: str | Path,
        *,
        argv: Callable[[], list[str]],
        example_class: str = "Example",
        build_digest: Callable[[], str | None] | None = None,
    ):
        self.script = Path(script)
        self._argv = argv
        self._example_class = example_class
        self._build_digest = build_digest or (lambda: None)
        self._condition = threading.Condition()
        self._queue: deque[_Job] = deque()
        self._pending: set[_Job] = set()
        self._running = 0
        self._closed = False

    def __call__(
        self,
        argv_list: list[list[str]] | list[str] | None = None,
        call: str | None = None,
        frames: int | None = None,
        timeout: float | None = 300.0,
        parallel: int = 2,
        wait: bool = True,
        tail: int = 2000,
    ) -> list[dict] | FreshRuns:
        """Run the script from disk in clean headless processes, one per argv list.

        Args:
            argv_list: Example arguments per run, e.g. ``[["--seed", "1"], ["--seed", "2"]]``; a flat list
                of strings is one run, and ``None`` runs once with the session's own arguments.
            call: Python code evaluated in each process after stepping (``example``, ``module``, ``args``,
                ``newton``, ``np``, ``wp`` in scope); the value of its final expression is reported as ``value``.
            frames: Frames to step; ``None`` steps the script's ``--num-frames``.
            timeout: Wall-clock limit per process [s]; the process group is killed after it.
            parallel: Maximum number of this session's fresh processes running at once.
            wait: Return the reports when all runs finish; ``False`` returns a :class:`FreshRuns` handle.
            tail: Characters of stdout and stderr kept per run.

        Returns:
            One report per argv list (see :mod:`newton.examples.headless`), or a handle.
        """
        runs = self._argv_list(argv_list)
        if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
            raise ValueError("parallel must be a positive integer")
        from newton.examples.headless import run_headless  # noqa: PLC0415

        options = {"frames": frames, "call": call, "timeout": timeout, "tail": tail}
        # Validate once here so a bad option raises in the cell instead of in every report.
        if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 0):
            raise ValueError("frames must be a non-negative integer or None")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0):
            raise ValueError("timeout must be a positive number of seconds or None")
        if call is not None and not isinstance(call, str):
            raise ValueError("call must be a string of Python code")
        jobs = [_Job(argv, parallel) for argv in runs]
        with self._condition:
            if self._closed:
                raise RuntimeError("The session is closed")
            self._queue.extend(jobs)
            self._pending.update(jobs)
        for job in jobs:
            threading.Thread(target=self._work, args=(job, run_headless, options), daemon=True).start()
        handle = FreshRuns(jobs, self._wake)
        return handle.result() if wait else handle

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` was called; a closed runner rejects new runs."""
        return self._closed

    def _argv_list(self, argv_list) -> list[list[str]]:
        if argv_list is None:
            return [list(self._argv())]
        if not isinstance(argv_list, list | tuple):
            raise ValueError("argv_list must be a list of argument lists")
        if argv_list and all(not isinstance(item, list | tuple) for item in argv_list):
            argv_list = [argv_list]
        runs = []
        for argv in argv_list:
            if not isinstance(argv, list | tuple):
                raise ValueError("argv_list must be a list of argument lists")
            if any(isinstance(item, bool) or not isinstance(item, str | int | float | Path) for item in argv):
                raise ValueError("Arguments must be strings (numbers and paths are converted)")
            runs.append([str(item) for item in argv])
        return runs

    def _wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _work(self, job: _Job, run_headless, options: dict) -> None:
        with self._condition:
            while not job.cancel.is_set() and (self._queue[0] is not job or self._running >= job.parallel):
                self._condition.wait()
            self._queue.remove(job)
            launch = not job.cancel.is_set()
            self._running += launch
            self._condition.notify_all()
        report: dict[str, Any] = {"argv": job.argv, **_REPORT, "seconds": {}}
        report["queued_seconds"] = round(time.perf_counter() - job.queued, 3)
        try:
            if not launch:
                report["status"] = "cancelled"
            else:
                digest = hashlib.sha256(self.script.read_bytes()).hexdigest()
                build = self._build_digest()
                report["script_changed_since_build"] = None if build is None else digest != build
                outcome = run_headless(
                    self.script, job.argv, example_class=self._example_class, cancel=job.cancel, **options
                )
                report.update((key, value) for key, value in outcome.items() if key in report)
        except Exception as error:
            report.update(status="error", phase="launch")
            report["exception"] = {"type": type(error).__name__, "message": str(error)[:4000], "traceback": None}
        finally:
            with self._condition:
                self._running -= launch
                self._pending.discard(job)
                self._condition.notify_all()
        job.report = report
        job.finished.set()

    def close(self, timeout: float = 10.0) -> None:
        """Kill running processes, drop queued runs, and wait up to ``timeout`` [s] for them to end."""
        with self._condition:
            self._closed = True
            jobs = list(self._pending)
            for job in jobs:
                job.cancel.set()
            self._condition.notify_all()
        deadline = time.monotonic() + timeout
        for job in jobs:
            job.finished.wait(max(0.0, deadline - time.monotonic()))
