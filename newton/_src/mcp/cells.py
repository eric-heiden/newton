# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source text of trusted-execution cells.

Cells are ``exec``'d strings, so :mod:`linecache` is the only place their
source lives. Each cell is registered there under a stable pseudo-filename,
which lets :func:`inspect.getsource`, tracebacks, :mod:`warp` kernels, and
worker processes read definitions made in earlier cells.
"""

from __future__ import annotations

import functools
import inspect
import linecache
import sys
from collections.abc import Callable
from types import FunctionType, ModuleType
from typing import Any

import warp as wp

_PREFIX = "<_newton_mcp_"


def cell_filename(workspace_name: str, index: int) -> str:
    """Pseudo-filename of cell ``index`` in the workspace module ``workspace_name``."""
    return f"<{workspace_name}:cell-{index}>"


def is_cell_filename(filename: Any) -> bool:
    """Whether ``filename`` names a trusted-execution cell of any session."""
    return isinstance(filename, str) and filename.startswith(_PREFIX) and ":cell-" in filename


def register_cell_source(filename: str, code: str) -> None:
    """Make ``code`` the source of ``filename`` for :mod:`linecache`; registering the same text again is a no-op."""
    lines = code.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    # mtime None marks the entry as not backed by a file, so linecache.checkcache() keeps it.
    entry = (len(code), None, lines, filename)
    if linecache.cache.get(filename) != entry:
        linecache.cache[filename] = entry


def forget_cell_source(filename: str) -> None:
    """Drop the source of ``filename`` from :mod:`linecache`."""
    linecache.cache.pop(filename, None)


def cell_source(filename: str) -> str | None:
    """Registered source of ``filename``, or ``None`` if it is not (or no longer) cached."""
    entry = linecache.cache.get(filename)
    if entry is None or len(entry) != 4:
        return None
    return "".join(entry[2])


def class_functions(cls: type) -> list[FunctionType]:
    """Functions a class body defined: methods, static and class methods, and property getters."""
    functions = []
    for member in list(vars(cls).values()):
        function = member.__func__ if isinstance(member, staticmethod | classmethod) else member
        function = function.fget if isinstance(function, property) else function
        if inspect.isfunction(function):
            functions.append(function)
    return functions


def definition_files(value: Any) -> set[str]:
    """Source files of a function, method, Warp kernel/function, or class (via its methods or module file)."""
    if inspect.ismethod(value):
        value = value.__func__
    if isinstance(value, wp.Kernel | wp.Function):
        value = getattr(value, "func", None)
    if inspect.isfunction(value):
        return {value.__code__.co_filename}
    if not isinstance(value, type):
        return set()
    files = set()
    module = sys.modules.get(getattr(value, "__module__", None) or "")
    filename = getattr(module, "__file__", None) if isinstance(module, ModuleType) else None
    if isinstance(filename, str):
        files.add(filename)
    files.update(function.__code__.co_filename for function in class_functions(value))
    return files


def _definitions(value: Any, seen: set[int], depth: int = 2):
    """``value`` and the functions and classes it holds: container items, ``functools.partial`` targets, bound
    methods, and the class of an instance, ``depth`` levels deep; objects in ``seen`` are skipped."""
    if id(value) in seen:
        return
    if inspect.isfunction(value) or isinstance(value, type):
        seen.add(id(value))
        yield value
        return
    yield value
    if depth <= 0 or isinstance(value, str | bytes):
        return
    if isinstance(value, dict):
        items = list(value.values())[:1024]
    elif isinstance(value, list | tuple | set | frozenset):
        items = list(value)[:1024]
    elif isinstance(value, functools.partial):
        items = [value.func, *value.args, *value.keywords.values()]
    elif inspect.ismethod(value):
        items = [value.__func__, value.__self__]
    else:
        items = [type(value)]
    for item in items:
        yield from _definitions(item, seen, depth - 1)


def referenced_cell_files(namespace: dict) -> set[str]:
    """Cells that define the functions and classes reachable from ``namespace`` (see :func:`_definitions`)."""
    files, seen = set(), set()
    for value in list(namespace.values()):
        for item in _definitions(value, seen):
            files.update(name for name in definition_files(item) if is_cell_filename(name))
    return files


def retain_cell_sources(
    filenames: list[str],
    namespace: dict,
    *,
    recent: int = 64,
    maximum: int = 512,
    forget: Callable[[str], None] = forget_cell_source,
) -> list[str]:
    """Keep up to ``maximum`` cells; beyond that, forget the oldest ones no reachable definition comes from.

    Args:
        filenames: Registered cells, oldest first.
        namespace: Workspace whose functions and classes (also inside containers) keep their cells.
        recent: Number of most recent cells that are always kept.
        maximum: Number of cells kept before any is forgotten.
        forget: Called with each dropped cell (default: :func:`forget_cell_source`).

    Returns:
        The cells that remain registered, oldest first.
    """
    if len(filenames) <= maximum:
        return list(filenames)
    older = filenames[:-recent]
    live = referenced_cell_files(namespace)
    excess = len(filenames) - maximum
    dropped = set()
    for name in older:
        if len(dropped) == excess:
            break
        if name not in live:
            dropped.add(name)
    # Still too many: the oldest cells go even if their definitions are still bound.
    for name in older:
        if len(dropped) == excess:
            break
        dropped.add(name)
    for name in dropped:
        forget(name)
    return [name for name in filenames if name not in dropped]
