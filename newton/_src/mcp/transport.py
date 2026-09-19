# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authenticated loopback attachment without device work on network threads."""

from __future__ import annotations

import json
import math
import os
import secrets
import socket
import socketserver
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .session import SimulationSession

_MAX_REQUEST = 1_048_576
_MAX_RESPONSE = 32 * 1_048_576


def _encode(value: Any, maximum: int) -> bytes:
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(data) > maximum:
        raise ValueError(f"Message exceeds {maximum} bytes")
    return data


def _read(stream: Any, maximum: int) -> dict:
    line = stream.readline(maximum + 1)
    if not line:
        raise ConnectionError("Connection closed")
    if len(line) > maximum or not line.endswith(b"\n"):
        raise ValueError("Message is too large or lacks newline framing")

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value {value} is unsupported")

    message = json.loads(line, parse_constant=invalid_constant)
    if not isinstance(message, dict):
        raise ValueError("Messages must be JSON objects")
    return message


class SimulationServer:
    """Expose an embedded session through authenticated loopback TCP.

    .. experimental::

        This entire class may change without a deprecation period. This TCP
        attachment protocol is internal to the stdio MCP bridge. Keep the
        connection file private: its token grants the session's capabilities,
        including unrestricted Python when explicitly enabled. File permissions
        are restricted on POSIX; Windows uses the parent directory's ACL.

    Args:
        session: Session whose owner thread keeps calling ``pump()`` or ``run()``.
        connection_file: New JSON file for host, port, and authentication token.
        port: Loopback port, or zero to choose an available port.
    """

    class _TCPServer(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = False
        request_queue_size = 16

        def process_request(self, request, client_address):
            if not self.slots.acquire(blocking=False):
                request.close()
                return
            try:
                super().process_request(request, client_address)
            except Exception:
                self.slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.slots.release()

    class _Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(10.0)
            server = self.server.owner
            try:
                message = _read(self.rfile, _MAX_REQUEST)
                token = message.get("token")
                if not isinstance(token, str) or not secrets.compare_digest(token, server._token):
                    raise PermissionError("Invalid connection token")
                if server._closed:
                    raise RuntimeError("Server is closed")
                operation, arguments = message.get("operation"), message.get("arguments", {})
                if not isinstance(operation, str) or not isinstance(arguments, dict):
                    raise ValueError("operation must be a string and arguments an object")
                timeout = message.get("timeout", 30.0)
                if isinstance(timeout, bool) or not isinstance(timeout, int | float):
                    raise ValueError("timeout must be a number")
                with server._pending_lock:
                    if server._closed:
                        raise RuntimeError("Server is closed")
                    request = server.session.enqueue(operation, arguments, timeout=timeout)
                    server._pending.add(request)
                try:
                    response = {"result": request.wait(timeout)}
                finally:
                    with server._pending_lock:
                        server._pending.discard(request)
                encoded = _encode(response, _MAX_RESPONSE)
            except Exception as error:
                encoded = _encode(
                    {"error": {"type": type(error).__name__, "message": str(error)[:32768]}}, _MAX_RESPONSE
                )
            try:
                self.wfile.write(encoded)
                self.wfile.flush()
            except OSError:
                pass

    def __init__(self, session: SimulationSession, *, connection_file: str | Path, port: int = 0):
        self.session = session
        self.connection_file = Path(connection_file)
        self._token = secrets.token_hex(32)
        self._closed = False
        self._started = False
        self._pending_lock = threading.Lock()
        self._pending = set()
        self._server = self._TCPServer(("127.0.0.1", port), self._Handler)
        self._server.owner = self
        self._server.slots = threading.BoundedSemaphore(16)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, name="newton-mcp-loopback", daemon=True
        )

    def start(self) -> SimulationServer:
        """Write a private connection file and start transport threads.

        Returns:
            This server, suitable for use as a context manager.
        """
        if self._started or self._closed:
            raise RuntimeError("Server is already started or closed")
        self.connection_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor = {"version": 1, "host": "127.0.0.1", "port": self._server.server_address[1], "token": self._token}
        try:
            fd = os.open(self.connection_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(descriptor, output)
                output.write("\n")
            self._thread.start()
            self._started = True
        except Exception:
            self._server.server_close()
            self._closed = True
            raise
        return self

    def close(self) -> None:
        """Stop attachment and remove this server's connection file.

        Pending requests from this server are cancelled. Close the session
        separately on its owner thread to release its renderer and other work.
        """
        if self._closed:
            return
        with self._pending_lock:
            self._closed = True
            for request in self._pending:
                with request.lock:
                    if not request.started:
                        request.cancelled = True
                        request.error = RuntimeError("Server closed before request execution")
                        request.done.set()
        if self._started:
            self._server.shutdown()
        self._server.server_close()
        try:
            descriptor = json.loads(self.connection_file.read_text(encoding="utf-8"))
            if descriptor.get("token") == self._token:
                self.connection_file.unlink()
        except (OSError, ValueError):
            pass

    def __enter__(self) -> SimulationServer:
        return self if self._started else self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class SimulationClient:
    """Attach to an instrumented session from another process or thread.

    .. experimental::

        This entire class may change without a deprecation period. Requests
        wait at most ``timeout`` seconds in the simulation queue. Once execution
        starts, the client waits for completion: Warp, GL, and arbitrary Python
        cannot be safely preempted. A transport failure after execution starts
        has an unknown outcome and must not trigger automatic mutation retries.

    Args:
        connection_file: Private JSON descriptor written by :class:`SimulationServer`.
        timeout: Maximum queue waiting time [s], in ``(0, 300]``.
    """

    def __init__(self, connection_file: str | Path, *, timeout: float = 30.0):
        self.connection_file = Path(connection_file)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 300
        ):
            raise ValueError("timeout must be in (0, 300] seconds")
        self.timeout = timeout
        if self.connection_file.stat().st_size > 4096:
            raise ValueError("Connection descriptor is too large")
        descriptor = json.loads(self.connection_file.read_text(encoding="utf-8"))
        if descriptor.get("version") != 1 or descriptor.get("host") != "127.0.0.1":
            raise ValueError("Expected version 1 loopback connection descriptor")
        port, token = descriptor.get("port"), descriptor.get("token")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port <= 65535:
            raise ValueError("Invalid port in descriptor")
        if not isinstance(token, str) or len(token) != 64:
            raise ValueError("Invalid token in descriptor")
        self._descriptor = descriptor

    def request(self, operation: str, **arguments: Any) -> dict:
        """Call a structured operation through the simulation thread.

        Args:
            operation: Operation accepted by :meth:`SimulationSession.dispatch`.
            **arguments: JSON-compatible operation arguments.

        Returns:
            JSON-compatible result from the live simulation.
        """
        data = _encode(
            {
                "token": self._descriptor["token"],
                "operation": operation,
                "arguments": arguments,
                "timeout": self.timeout,
            },
            _MAX_REQUEST,
        )
        with socket.create_connection(("127.0.0.1", self._descriptor["port"]), timeout=10) as connection:
            connection.sendall(data)
            connection.settimeout(None)
            with connection.makefile("rb") as stream:
                response = _read(stream, _MAX_RESPONSE)
        if "error" in response:
            error = response["error"]
            kind = {
                "TimeoutError": TimeoutError,
                "PermissionError": PermissionError,
                "ValueError": ValueError,
                "KeyError": KeyError,
            }.get(error["type"], RuntimeError)
            raise kind(error["message"])
        return response["result"]
