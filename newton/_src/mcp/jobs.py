# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Background jobs for trusted execution: start a call now, collect its result later."""

from __future__ import annotations

import concurrent.futures
import reprlib
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workers import WorkerPool

_PREVIEW = reprlib.Repr()
_PREVIEW.maxstring = 120
_PREVIEW.maxother = 120
_PREVIEW.maxlist = _PREVIEW.maxtuple = _PREVIEW.maxdict = _PREVIEW.maxset = 8
_PREVIEW.maxlevel = 3


class _Job:
    def __init__(self, job_id: int, label: str, where: str, progress: Path | None):
        self.id = job_id
        self.label = label
        self.where = where
        self.progress = progress
        self.offset = 0
        self.future: Future | None = None
        self.started = time.perf_counter()
        self.finished: float | None = None
        self.reported = False
        self.collected = False

    def done(self) -> bool:
        return self.future is not None and self.future.done()

    def outcome(self) -> tuple[str, Any]:
        """``("done", value)``, ``("failed", exception)``, or ``("cancelled", None)``; call only when done."""
        if self.future.cancelled():
            return "cancelled", None
        try:
            return "done", self.future.result()
        except BaseException as error:
            return "failed", error

    def seconds(self) -> float:
        end = self.finished if self.finished is not None else time.perf_counter()
        return round(end - self.started, 2)

    def worker(self) -> int | None:
        return getattr(self.future, "worker", None)

    def where_ran(self) -> dict:
        """``worker`` and, for worker jobs, the pool ``build`` the worker's scene had when the job started."""
        build = getattr(self.future, "build", None)
        return {"worker": self.worker(), **({"build": build} if build is not None else {})}

    def new_lines(self, limit: int = 20) -> list[str]:
        """Lines the job printed since the previous call (at most ``limit``, the latest ones)."""
        if self.progress is None:
            return []
        try:
            with open(self.progress, "rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
        except OSError:
            return []
        # Keep an unfinished last line for the next call.
        end = data.rfind(b"\n") + 1 if not self.done() else len(data)
        self.offset += end
        return data[:end].decode("utf-8", errors="replace").splitlines()[-limit:]


class JobQueue:
    """Start calls in the background and collect their results in a later cell.

    ``start`` returns a job id at once; the call runs on the next free worker
    session of a :class:`WorkerPool` (``where="worker"``) while trusted
    execution continues. Nothing here runs on, or touches, the calling
    session's simulation. Jobs that finished since the previous report are
    listed by :meth:`report`, which a session adds to its next execution
    response. Other places to run a job are added by the embedding
    application with :meth:`register_backend`.

    Args:
        workers: Worker pool for ``where="worker"``, or ``None``.
        seconds_left: Optional callable returning the seconds left until the running call must
            reply to its caller (``None`` without a limit); waits without a ``timeout`` end a
            little earlier.
    """

    max_jobs = 512
    """Job records kept; the oldest collected jobs are dropped beyond this."""

    reply_margin = 5.0
    """Seconds before the reply limit at which waits without a ``timeout`` return [s]."""

    def __init__(self, workers: WorkerPool | None = None, seconds_left: Callable[[], float | None] | None = None):
        self.workers = workers
        self._seconds_left = seconds_left
        self._backends: dict[str, Callable] = {}
        self._jobs: dict[int, _Job] = {}
        self._next = 1
        self._lock = threading.Lock()
        self._directory: Path | None = None

    def register_backend(self, where: str, launch: Callable) -> None:
        """Add a place to run jobs, selected by ``start(..., where=where)``.

        Args:
            where: Backend name, e.g. ``"cluster"``.
            launch: ``launch(function, args, kwargs, progress)`` starts the call and returns a
                :class:`~concurrent.futures.Future` whose result is the call's value (or raises).
                ``function`` is a callable or a code string; ``progress`` is a file that should
                receive the text the job prints.
        """
        if not isinstance(where, str) or not where:
            raise ValueError("where must be a non-empty string")
        self._backends[where] = launch

    def start(self, function: Callable | str, *args: Any, where: str = "worker", **kwargs: Any) -> int:
        """Start ``function(*args, **kwargs)`` (or a code string with ``args`` bound to the first argument).

        Args:
            function: Function (sent by source, as :meth:`WorkerPool.submit`) or code string.
            *args: Positional arguments of the call.
            where: ``"worker"`` or a registered backend name.
            **kwargs: Keyword arguments of the call.

        Returns:
            The job id.
        """
        label = "code" if isinstance(function, str) else getattr(function, "__qualname__", type(function).__name__)
        with self._lock:
            job_id = self._next
            self._next += 1
        if where == "worker":
            if self.workers is None:
                raise ValueError("where='worker' needs worker sessions; this session has none")
            progress = self.workers._exchange / f"job-{job_id}.out"
            job = _Job(job_id, str(label), where, progress)
            call = (args[0] if args else None) if isinstance(function, str) else (args, kwargs)
            job.future = self.workers._submit(function, [call], progress=[str(progress)], echo=False)[0]
        elif where in self._backends:
            if self._directory is None:
                self._directory = Path(tempfile.mkdtemp(prefix="newton-mcp-jobs-"))
            job = _Job(job_id, str(label), where, self._directory / f"job-{job_id}.out")
            job.future = self._backends[where](function, args, kwargs, job.progress)
        else:
            known = ", ".join(repr(name) for name in ["worker", *self._backends])
            raise ValueError(f"where={where!r} is not available in this session (available: {known})")

        def finished(_future, job=job):
            job.finished = time.perf_counter()

        job.future.add_done_callback(finished)
        with self._lock:
            self._jobs[job_id] = job
            self._trim()
        return job_id

    def _default_timeout(self) -> float | None:
        left = self._seconds_left() if self._seconds_left is not None else None
        return None if left is None else max(0.0, left - self.reply_margin)

    def wait(self, timeout: float | None = None, any: bool = True, ids: list[int] | None = None) -> dict:
        """Wait for jobs that have not been collected yet and collect the finished ones.

        Args:
            timeout: Longest wait [s]; ``None`` waits until the condition holds or, inside a call with a
                reply limit, until shortly before that limit.
            any: Return when at least one job has finished (``False``: when all have).
            ids: Jobs to consider (default: all uncollected jobs).

        Returns:
            ``finished``: ``id``, ``status`` (done/failed/cancelled), ``seconds``, ``worker``, ``build``
            (the worker pool's rebuild count the worker's scene had when the job started),
            ``result`` or ``error``, and the job's remaining printed ``lines``;
            ``running``: ``id``, ``status`` (running/queued), ``seconds``, ``worker``, ``build``, and the
            ``lines`` printed since the previous wait.
        """
        with self._lock:
            jobs = [job for job in self._jobs.values() if not job.collected and (ids is None or job.id in ids)]
        pending = [job.future for job in jobs if not job.done()]
        if timeout is None:
            timeout = self._default_timeout()
        if pending and not (any and len(pending) < len(jobs)):
            concurrent.futures.wait(
                pending,
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED if any else concurrent.futures.ALL_COMPLETED,
            )
        finished, running = [], []
        for job in jobs:
            if job.done():
                status, value = job.outcome()
                entry = {"id": job.id, "status": status, "seconds": job.seconds(), **job.where_ran()}
                if status == "done":
                    entry["result"] = value
                elif status == "failed":
                    entry["error"] = _error(value)
                entry["lines"] = job.new_lines(limit=50)
                job.collected = job.reported = True
                self._discard_progress(job)
                finished.append(entry)
            else:
                status = "running" if job.future.running() else "queued"
                entry = {"id": job.id, "status": status, "seconds": job.seconds(), **job.where_ran()}
                running.append({**entry, "lines": job.new_lines()})
        return {"finished": finished, "running": running}

    def result(self, job_id: int, timeout: float | None = None) -> Any:
        """Wait for one job and return its value (raising its error).

        Args:
            job_id: Job id from :meth:`start`.
            timeout: Longest wait [s]; ``None`` waits until the job finishes or, inside a call with a
                reply limit, until shortly before that limit.

        Raises:
            TimeoutError: The job did not finish in time; it keeps running.
        """
        job = self._job(job_id)
        limit = self._default_timeout() if timeout is None else timeout
        try:
            return job.future.result(limit)
        except concurrent.futures.TimeoutError:
            if job.done():
                raise  # the job's own error
            raise TimeoutError(
                f"Job {job_id} is still running after {job.seconds():.0f} s (waited {limit:.0f} s); "
                f"jobs.result({job_id}) or jobs.wait() in a later call collects it"
            ) from None
        finally:
            if job.done():
                job.collected = job.reported = True

    def cancel(self, job_id: int) -> bool:
        """Cancel a job that has not started yet; returns whether it was cancelled."""
        return self._job(job_id).future.cancel()

    def status(self) -> list[dict]:
        """``id``, ``label``, ``where``, ``status`` (queued/running/done/failed/cancelled), ``seconds``, ``worker``,
        ``build``."""
        with self._lock:
            jobs = list(self._jobs.values())
        rows = []
        for job in jobs:
            if job.done():
                status = job.outcome()[0]
            else:
                status = "running" if job.future.running() else "queued"
            rows.append(
                {
                    "id": job.id,
                    "label": job.label,
                    "where": job.where,
                    "status": status,
                    "seconds": job.seconds(),
                    **job.where_ran(),
                }
            )
        return rows

    def report(self) -> dict:
        """Jobs that finished since the previous report or wait (with a short result preview), and unfinished ids."""
        with self._lock:
            jobs = list(self._jobs.values())
        finished, running, queued = [], [], []
        for job in jobs:
            if not job.done():
                (running if job.future.running() else queued).append(job.id)
                continue
            if job.reported:
                continue
            job.reported = True
            status, value = job.outcome()
            entry = {"id": job.id, "status": status, "seconds": job.seconds(), **job.where_ran()}
            if status == "done":
                entry["result"] = _PREVIEW.repr(value)
            elif status == "failed":
                entry["error"] = _error(value)[:600]
            finished.append(entry)
        report = {"finished": finished, "running": running, "queued": queued}
        return {key: value for key, value in report.items() if value}

    def _job(self, job_id: int) -> _Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No job {job_id}")
            return self._jobs[job_id]

    def _discard_progress(self, job: _Job) -> None:
        if job.progress is not None:
            job.progress.unlink(missing_ok=True)

    def _trim(self) -> None:
        excess = len(self._jobs) - self.max_jobs
        for job_id in [i for i, job in self._jobs.items() if job.collected][: max(0, excess)]:
            del self._jobs[job_id]


def _error(error: BaseException) -> str:
    from .workers import WorkerError  # noqa: PLC0415

    # A worker error's message already starts with the type of the exception raised on the worker.
    return str(error) if isinstance(error, WorkerError) else f"{type(error).__name__}: {error}"
