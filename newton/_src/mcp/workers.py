# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Concurrent trusted execution on sibling application processes."""

from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from .transport import SimulationClient


class WorkerPool:
    """Run Python cells concurrently on sibling live sessions.

    Each worker is another instance of the same application with its own scene
    and persistent Python workspace, reached through its connection file. A
    cell's ``result`` (or last expression) is returned; ``args`` holds the
    JSON arguments of a :meth:`map` job. Jobs run with
    ``recovery="acknowledge"`` so a failed job does not block later jobs on
    that worker: an exception is returned for that item, and code that may
    have left a worker's scene inconsistent should rebuild or reset it.

    Args:
        connection_files: Connection descriptors of the worker sessions.
        timeout: Maximum queue waiting time per request [s].
    """

    def __init__(self, connection_files: list[str | Path], *, timeout: float = 300.0):
        self._clients = [SimulationClient(path, timeout=timeout) for path in connection_files]
        self._idle: queue.Queue[int] = queue.Queue()
        for index in range(len(self._clients)):
            self._idle.put(index)

    def __len__(self) -> int:
        return len(self._clients)

    @property
    def count(self) -> int:
        """Number of worker sessions."""
        return len(self._clients)

    def _run(self, index: int, code: str, arguments: Any) -> Any:
        prefix = "" if arguments is None else f"args = __import__('json').loads({json.dumps(json.dumps(arguments))})\n"
        response = self._clients[index].request("execute", code=prefix + code, recovery="acknowledge")
        return response.get("result") if response.get("result") is not None else response.get("result_repr")

    def broadcast(self, code: str) -> list[Any]:
        """Execute ``code`` once on every worker concurrently, e.g. to define helpers or load data.

        Returns:
            One result per worker, in worker order. Errors are raised.
        """
        results: list[Any] = [None] * len(self._clients)
        errors: list[BaseException] = []

        def run(index):
            try:
                results[index] = self._run(index, code, None)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(self._clients))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return results

    def submit(self, code: str, arguments: Any = None) -> Future:
        """Queue one job on the next idle worker and return a future for its result."""
        future: Future = Future()

        def run():
            index = self._idle.get()
            try:
                future.set_result(self._run(index, code, arguments))
            except BaseException as error:
                future.set_exception(error)
            finally:
                self._idle.put(index)

        threading.Thread(target=run, daemon=True).start()
        return future

    def map(self, code: str, arguments: list[Any]) -> list[Any]:
        """Run ``code`` once per item of ``arguments`` (available as ``args``), spread over idle workers.

        Returns:
            Results in input order. A failed job yields ``{"error": message}`` instead of raising,
            so one bad candidate does not discard the others.
        """
        futures = [self.submit(code, item) for item in arguments]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                results.append({"error": f"{type(error).__name__}: {str(error)[:2000]}"})
        return results
