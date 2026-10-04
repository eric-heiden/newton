# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write live results back into a hosted script, and report model edits since the last build.

Trusted-execution cells are ``exec``'d strings, so their source lives only in
:mod:`linecache`. This module also gives classes defined in a cell a per-cell
module, which is how :func:`inspect.getsource` locates a class's file.
"""

from __future__ import annotations

import ast
import builtins
import collections
import difflib
import hashlib
import inspect
import io
import itertools
import keyword
import linecache
import math
import os
import re
import symtable
import sys
import time
import tokenize
import weakref
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import warp as wp

from ..sim.enums import ModelFlags
from ..sim.model import Model

MISSING = object()
"""Sentinel for an omitted ``persist`` value."""

_WIDTH = 120
_MAX_ELEMENTS = 100_000
_DIFF_LINES = 160
_STRING_TOKENS = {
    getattr(tokenize, name) for name in ("STRING", "FSTRING_MIDDLE", "TSTRING_MIDDLE") if hasattr(tokenize, name)
}


# ---------------------------------------------------------------------------
# Cell source registration


def instrument_cell_classes(tree: ast.Module, hook: str) -> bool:
    """Insert ``hook(Name)`` after each top-level class statement of a cell.

    The hook runs as soon as the class exists, so later statements of the same cell can already
    call :func:`inspect.getsource` on it.

    Returns:
        Whether the cell defines a top-level class.
    """
    body = []
    for node in tree.body:
        body.append(node)
        if isinstance(node, ast.ClassDef):
            call = ast.Call(ast.Name(hook, ast.Load()), [ast.Name(node.name, ast.Load())], [])
            body.append(ast.copy_location(ast.Expr(call), node))
    found = len(body) > len(tree.body)
    tree.body = body
    ast.fix_missing_locations(tree)
    return found


class _CellModule(ModuleType):
    """Module of the classes one cell defined: its ``__file__`` is the cell, its names are the workspace's.

    Name lookups (pickling by reference, ``typing.get_type_hints``, ``vars()``) resolve through the
    shared workspace globals, exactly as they did before the class was moved here.
    """

    def __init__(self, name: str, filename: str, workspace: dict):
        super().__init__(name, f"Classes defined in {filename}")
        self.__file__ = filename
        object.__setattr__(self, "_workspace", workspace)

    @property
    def __dict__(self) -> dict:
        return object.__getattribute__(self, "_workspace")

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "_workspace")[name]
        except KeyError:
            raise AttributeError(name) from None


def bind_cell_class(cls: Any, workspace: dict, workspace_name: str, filename: str, module_name: str) -> bool:
    """Move a class a cell defined into a per-cell module whose ``__file__`` is the cell's linecache entry.

    :func:`inspect.getsource` finds a class through ``sys.modules[cls.__module__].__file__``,
    and all cells share one workspace module without a file.

    Returns:
        Whether the class was moved.
    """
    if not isinstance(cls, type) or cls.__module__ != workspace_name:
        return False
    module = sys.modules.get(module_name)
    if not isinstance(module, _CellModule) or module.__file__ != filename:
        sys.modules[module_name] = _CellModule(module_name, filename, workspace)
    _move_class(cls, workspace_name, module_name)
    return True


def _move_class(cls: type, workspace_name: str, module_name: str) -> None:
    for value in list(vars(cls).values()):
        if (
            isinstance(value, type)
            and value.__module__ == workspace_name
            and value.__qualname__.startswith(cls.__qualname__ + ".")
        ):
            _move_class(value, workspace_name, module_name)
    first_line = cls.__dict__.get("__firstlineno__")
    try:
        cls.__module__ = module_name
        # Python 3.13+ drops __firstlineno__ when __module__ changes; inspect needs it.
        if first_line is not None and "__firstlineno__" not in cls.__dict__:
            cls.__firstlineno__ = first_line
    except (AttributeError, TypeError):
        pass


def referenced_cell_files(workspace: dict) -> set[str]:
    """Source files of the functions and classes currently bound in the workspace."""
    files = set()
    for bound in list(workspace.values()):
        value = bound.__func__ if inspect.ismethod(bound) else bound
        if inspect.isfunction(value):
            files.add(value.__code__.co_filename)
        elif isinstance(value, type):
            module = sys.modules.get(value.__module__)
            filename = getattr(module, "__file__", None) if isinstance(module, ModuleType) else None
            if isinstance(filename, str):
                files.add(filename)
    return files


# ---------------------------------------------------------------------------
# Literal conversion and formatting


def _plain(value: Any, budget: list[int] | None = None, depth: int = 0) -> Any:
    """Convert ``value`` to built-in literal types, or raise ``TypeError``."""
    budget = [_MAX_ELEMENTS] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        raise ValueError(f"Value exceeds {_MAX_ELEMENTS} elements or 32 nesting levels")
    if value is None or isinstance(value, bool | str | bytes):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{value!r} has no literal form")
        return float(value)
    if isinstance(value, np.generic):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            # Shortest repr that round-trips the stored precision: float32 0.05 stays 0.05.
            number = float(str(value)) if value.dtype.itemsize < 8 else float(value)
            return _plain(number, budget, depth)
        if isinstance(value, np.str_ | np.bytes_):
            return value.item()
        raise TypeError(f"NumPy {value.dtype} values have no literal form")
    if isinstance(value, wp.array):
        if value.size > _MAX_ELEMENTS:
            raise ValueError(f"Warp array exceeds {_MAX_ELEMENTS} elements")
        value = value.numpy()
    elif hasattr(type(value), "_wp_scalar_type_"):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        if value.size > budget[0]:
            raise ValueError(f"Array exceeds {_MAX_ELEMENTS} elements")
        if value.dtype.kind not in "biufUS":
            raise TypeError(f"NumPy {value.dtype} arrays have no literal form")
        if value.ndim == 0:
            return _plain(value[()], budget, depth)
        return [_plain(item, budget, depth + 1) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            plain_key = _plain(key, budget, depth + 1)
            if isinstance(plain_key, list | dict | set):
                raise TypeError("Dictionary keys must be scalars or tuples")
            result[plain_key] = _plain(item, budget, depth + 1)
        return result
    if isinstance(value, tuple):
        return tuple(_plain(item, budget, depth + 1) for item in value)
    if isinstance(value, list):
        return [_plain(item, budget, depth + 1) for item in value]
    if isinstance(value, set) and value:
        return {_plain(item, budget, depth + 1) for item in value}
    raise TypeError(
        f"{type(value).__name__} values cannot be written as a literal; use dicts, lists, tuples, non-empty sets, "
        "numbers, strings, bools, None, or NumPy/Warp values"
    )


def _same(a: Any, b: Any) -> bool:
    """Type-strict equality, so ``1``, ``1.0`` and ``True`` count as different literals."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return _same_keys(a, b) and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _same_keys(a: dict, b: dict) -> bool:
    """Whether two dictionaries have equal keys of identical types (``{1: x}`` differs from ``{True: x}``)."""
    if a.keys() != b.keys():
        return False
    keys = {key: key for key in b}
    return all(type(key) is type(keys[key]) for key in a)


def _quote(text: str) -> str:
    literal = repr(text)
    # Prefer double quotes (ruff style). A single-quoted repr of text without " has no quote escapes.
    if literal.startswith("'") and '"' not in text:
        literal = '"' + literal[1:-1] + '"'
    return literal


def _inline(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_inline(key)}: {_inline(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_inline(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(_inline(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, set):
        return "{" + ", ".join(sorted(_inline(item) for item in value)) + "}"
    if isinstance(value, str):
        return _quote(value)
    return repr(value)


def _format(value: Any, indent: str, column: int, multiline: bool, newline: str) -> str:
    """Literal source for ``value`` starting at ``column``; continuation lines use ``indent``."""
    text = _inline(value)
    if not isinstance(value, dict | list | tuple | set) or not value:
        return text
    if not multiline and column + len(text) <= _WIDTH:
        return text
    inner = indent + "    "
    brackets = {dict: "{}", list: "[]", tuple: "()", set: "{}"}[type(value)]
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            head = f"{_inline(key)}: "
            items.append(head + _format(item, inner, len(inner) + len(head), False, newline))
    elif all(not isinstance(item, dict | list | tuple | set) for item in value):
        # Pack scalar sequences (arrays) into lines instead of one number per line.
        items, line = [], ""
        for item in sorted(map(_inline, value)) if isinstance(value, set) else map(_inline, value):
            if line and len(inner) + len(line) + len(item) + 3 > _WIDTH:
                items.append(line)
                line = ""
            line = f"{line}, {item}" if line else item
        items.append(line)
    else:
        items = [_format(item, inner, len(inner), False, newline) for item in value]
    body = "".join(f"{inner}{item},{newline}" for item in items)
    return brackets[0] + newline + body + indent + brackets[1]


# ---------------------------------------------------------------------------
# Script text and module-level bindings


def _split_lines(text: str) -> list[str]:
    """Split like the Python tokenizer (``\\n``, ``\\r\\n``, ``\\r``), keeping line ends."""
    return re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z", text)


class _Script:
    """A script's exact text plus helpers to splice AST node spans without touching other bytes."""

    def __init__(self, path: Path):
        self.path = path
        self.raw = path.read_bytes()
        self.encoding, _ = tokenize.detect_encoding(io.BytesIO(self.raw).readline)
        self.text = self.raw.decode(self.encoding)
        try:
            self.tree = ast.parse(self.text, filename=str(path))
        except SyntaxError as error:
            raise ValueError(f"{path} does not parse ({error}); fix it before persisting into it") from error
        self.lines = _split_lines(self.text)
        self.starts = list(itertools.accumulate((len(line) for line in self.lines), initial=0))

    def offset(self, line: int, column: int) -> int:
        """Character offset of an AST position (1-based line, UTF-8 byte column)."""
        text = self.lines[line - 1]
        return self.starts[line - 1] + len(text.encode("utf-8")[:column].decode("utf-8"))

    def indent(self, line: int) -> str:
        text = self.lines[line - 1]
        return text[: len(text) - len(text.lstrip(" \t"))]

    def newline(self, line: int) -> str:
        text = self.lines[line - 1] if line - 1 < len(self.lines) else ""
        ending = text[len(text.rstrip("\r\n")) :]
        return ending or ("\r\n" if "\r\n" in self.text else "\n")

    @staticmethod
    def splice(text: str, edits: list[tuple[int, int, str]]) -> str:
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text


def _scope_statements(body: list[ast.stmt]):
    """Statements executed in this scope, including inside if/for/while/with/try/match blocks."""
    for node in body:
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from _scope_statements(getattr(node, field, None) or [])
        for handler in getattr(node, "handlers", None) or []:
            yield from _scope_statements(handler.body)
        for case in getattr(node, "cases", None) or []:
            yield from _scope_statements(case.body)


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        return [name for element in target.elts for name in _target_names(element)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _bound_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Assign):
        return [name for target in node.targets for name in _target_names(target)]
    if isinstance(node, ast.AnnAssign):
        return _target_names(node.target) if node.value is not None else []
    if isinstance(node, ast.AugAssign | ast.For | ast.AsyncFor):
        return _target_names(node.target)
    if isinstance(node, ast.With | ast.AsyncWith):
        return [name for item in node.items if item.optional_vars for name in _target_names(item.optional_vars)]
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return [node.name]
    if isinstance(node, ast.Import | ast.ImportFrom):
        return [alias.asname or alias.name.split(".")[0] for alias in node.names if alias.name != "*"]
    if isinstance(node, ast.Delete):
        return [name for target in node.targets for name in _target_names(target)]
    return []


def _bindings(body: list[ast.stmt], name: str) -> list[ast.stmt]:
    return [node for node in _scope_statements(body) if name in _bound_names(node)]


def _all_bound(body: list[ast.stmt]) -> set[str]:
    return {name for node in _scope_statements(body) for name in _bound_names(node)}


def _star_import(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
        for node in _scope_statements(body)
    )


def _lines_of(nodes: list[ast.stmt]) -> str:
    return ", ".join(str(node.lineno) for node in nodes)


def _missing(filename: str, body: list[ast.stmt], name: str, what: str, where: str) -> ValueError:
    close = difflib.get_close_matches(name, sorted(_all_bound(body)), n=4, cutoff=0.6)
    hint = f" Similar names: {', '.join(close)}." if close else ""
    return ValueError(f"{filename} has no {what} named {name!r} {where}.{hint}")


# ---------------------------------------------------------------------------
# Writing, rebuilding, and checking


def script_path(session: Any, path: str | Path | None) -> Path:
    """Resolve the file to edit: ``path`` (relative to the hosted script's directory) or the hosted script."""
    source = getattr(session, "source_path", None)
    if path is None:
        if source is None:
            raise ValueError("This session has no hosted script; pass path=")
        return Path(source)
    result = Path(path).expanduser()
    if not result.is_absolute() and source is not None:
        result = Path(source).parent / result
    return result.resolve()


def _evaluate(session: Any, check: str | Callable) -> Any:
    if isinstance(check, str):
        value = eval(compile(check, "<persist:check>", "eval"), session._eval_scope())
    else:
        from .session import _session_callable  # noqa: PLC0415

        value = _session_callable(check)(session)
    return _detach(value)


def _detach(value: Any) -> Any:
    """Copy arrays so later simulation steps cannot change a recorded check value."""
    if isinstance(value, wp.array):
        return value.numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_detach(item) for item in value) if type(value) in (list, tuple) else value
    return value


def _compare(a: Any, b: Any, tolerance: float) -> tuple[bool, float | None]:
    """Whether two check values agree within ``tolerance`` (relative and absolute), and their largest difference."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False, None
        results = [_compare(a[key], b[key], tolerance) for key in a]
        differences = [difference for _, difference in results if difference is not None]
        return all(equal for equal, _ in results), max(differences, default=None)
    try:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        try:
            return bool(a == b), None
        except Exception:
            return False, None
    if x.shape != y.shape:
        return False, None
    if x.size == 0:
        return True, 0.0
    both_nan = np.isnan(x) & np.isnan(y)
    difference = np.where(both_nan, 0.0, np.abs(x - y))
    largest = float(np.max(difference)) if np.all(np.isfinite(difference) | both_nan) else math.inf
    equal = bool(np.allclose(x, y, rtol=tolerance, atol=tolerance, equal_nan=True))
    return equal, largest


def _brief(value: Any) -> Any:
    """A JSON-friendly rendering of a check value, summarized when large."""
    if isinstance(value, dict):
        return {str(key): _brief(item) for key, item in list(value.items())[:16]}
    if isinstance(value, np.ndarray | list | tuple):
        array = np.asarray(value, dtype=object) if not isinstance(value, np.ndarray) else value
        if array.size <= 16:
            try:
                return _plain(value)
            except (TypeError, ValueError):
                pass
        return f"<{type(value).__name__} shape={list(np.shape(array))}>"
    try:
        return _plain(value)
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"


def _backup(session: Any, script: _Script) -> Path:
    directory = Path(session.artifact_directory) / "persist"
    directory.mkdir(parents=True, exist_ok=True)
    for index in itertools.count(1):
        candidate = directory / f"{script.path.stem}.{index:03d}{script.path.suffix}"
        if not candidate.exists():
            candidate.write_bytes(script.raw)
            return candidate
    raise AssertionError("unreachable")


def _diff(script: _Script, text: str) -> str:
    old = [line.rstrip("\r\n") for line in script.lines]
    new = [line.rstrip("\r\n") for line in _split_lines(text)]
    name = script.path.name
    lines = list(difflib.unified_diff(old, new, f"a/{name}", f"b/{name}", n=2, lineterm=""))
    if len(lines) > _DIFF_LINES:
        lines = [*lines[:_DIFF_LINES], f"... ({len(lines) - _DIFF_LINES} more diff lines)"]
    return "\n".join(lines)


def _commit(
    session: Any,
    script: _Script,
    text: str,
    *,
    rebuild: bool,
    check: str | Callable | None,
    tolerance: float,
    result: dict,
) -> dict:
    """Write ``text`` with a backup, print the diff, then optionally rebuild and re-evaluate ``check``."""
    if check is not None and not rebuild:
        raise ValueError("check compares the live and rebuilt scenes; it needs rebuild=True")
    if check is not None and getattr(session, "rebuild_callback", None) is None:
        raise ValueError("check needs a session with a rebuild callback")
    live = _evaluate(session, check) if check is not None else None
    result = {"path": str(script.path), "changed": text != script.text, **result}
    if result["changed"]:
        result["backup"] = str(_backup(session, script))
        script.path.write_bytes(text.encode(script.encoding))
        linecache.checkcache(str(script.path))
        print(_diff(script, text))
    else:
        print(f"{script.path.name}: no change")
    if not rebuild:
        return result
    if getattr(session, "rebuild_callback", None) is None:
        result["rebuilt"] = False
        return result
    started = time.perf_counter()
    try:
        session.dispatch("rebuild", {})
    except Exception as error:
        where = f" (backup {result['backup']})" if "backup" in result else ""
        raise RuntimeError(f"Wrote {script.path}{where}, but rebuilding it failed: {error}") from error
    result["rebuilt"] = True
    result["rebuild_seconds"] = round(time.perf_counter() - started, 3)
    workers = getattr(session, "workers", None)
    if workers is not None and workers.count:
        try:
            workers.broadcast("session.dispatch('rebuild', {})")
            result["workers_rebuilt"] = workers.count
        except Exception as error:
            result["workers_rebuild_error"] = f"{type(error).__name__}: {str(error)[:500]}"
    if check is not None:
        rebuilt = _evaluate(session, check)
        reproduced, difference = _compare(live, rebuilt, tolerance)
        result["check"] = {
            "live": _brief(live),
            "rebuilt": _brief(rebuilt),
            "reproduced": reproduced,
            "max_abs_difference": difference,
            "tolerance": tolerance,
        }
    return result


# ---------------------------------------------------------------------------
# persist: literal assignments


def _patch(script: _Script, node: ast.expr, old: Any, new: Any, edits: list) -> None:
    """Rewrite only the parts of a literal that differ, keeping comments and layout elsewhere."""
    if _same(old, new):
        return
    if isinstance(node, ast.Dict) and type(old) is dict and type(new) is dict and None not in node.keys:
        keys = [ast.literal_eval(key) for key in node.keys]
        if len(set(keys)) == len(keys) and _same_keys(dict.fromkeys(keys), new):
            for key, value in zip(keys, node.values, strict=True):
                _patch(script, value, old[key], new[key], edits)
            return
    if (
        isinstance(node, ast.List | ast.Tuple)
        and type(old) is type(new)
        and len(old) == len(new)
        and len(node.elts) == len(old)
    ):
        for element, x, y in zip(node.elts, old, new, strict=True):
            _patch(script, element, x, y, edits)
        return
    multiline = node.end_lineno > node.lineno
    replacement = _format(new, script.indent(node.lineno), node.col_offset, multiline, script.newline(node.lineno))
    edits.append(
        (
            script.offset(node.lineno, node.col_offset),
            script.offset(node.end_lineno, node.end_col_offset),
            replacement,
        )
    )


def _assignment(script: _Script, name: str) -> ast.Assign | ast.AnnAssign:
    found = _bindings(script.tree.body, name)
    if not found:
        raise _missing(script.path.name, script.tree.body, name, "assignment", "at module level")
    if len(found) > 1:
        raise ValueError(
            f"{name} is bound {len(found)} times at module level in {script.path.name} (lines {_lines_of(found)}); "
            "persist edits a single assignment"
        )
    node = found[0]
    simple = (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)) or (
        isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    )
    if not simple:
        raise ValueError(
            f"{name} is bound on line {node.lineno} of {script.path.name} by a {type(node).__name__} statement "
            "that is not a plain NAME = <literal> assignment"
        )
    try:
        ast.literal_eval(node.value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        segment = ast.get_source_segment(script.text, node.value) or ""
        segment = segment if len(segment) <= 80 else segment[:77] + "..."
        raise ValueError(
            f"{name} = {segment} on line {node.lineno} of {script.path.name} is not a literal; persist only replaces "
            "literal values (dict, list, tuple, set, number, string, bytes, bool, None)"
        ) from None
    return node


def persist(
    session: Any,
    name: str,
    value: Any = MISSING,
    *,
    rebuild: bool = True,
    check: str | Callable | None = None,
    tolerance: float = 1e-6,
    path: str | Path | None = None,
) -> dict:
    """Implementation of :meth:`SimulationSession.persist`."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError("name must be a Python identifier")
    if value is MISSING:
        module = session.namespace.get("module")
        if module is None or not hasattr(module, name):
            raise ValueError(f"Give a value; the hosted script module has no attribute {name!r}")
        value = getattr(module, name)
    new = _plain(value)
    script = _Script(script_path(session, path))
    node = _assignment(script, name)
    old = ast.literal_eval(node.value)
    edits = []
    _patch(script, node.value, old, new, edits)
    text = _Script.splice(script.text, edits)
    # Re-parse to prove the file still compiles and now holds exactly the requested value.
    check_script = ast.parse(text)
    rewritten = _bindings(check_script.body, name)
    if len(rewritten) != 1 or not _same(ast.literal_eval(rewritten[0].value), new):
        raise RuntimeError(f"Internal error: rewriting {name} did not round-trip; {script.path} was not changed")
    mutations = [
        statement.lineno
        for statement in _scope_statements(script.tree.body)
        if isinstance(statement, ast.Assign | ast.AugAssign | ast.AnnAssign)
        for target in (statement.targets if isinstance(statement, ast.Assign) else [statement.target])
        if isinstance(target, ast.Subscript | ast.Attribute) and _root_name(target) == name
    ]
    details = {"name": name, "line": node.lineno, "replaced_spans": len(edits)}
    if mutations:
        details["module_level_item_assignments"] = mutations
    return _commit(session, script, text, rebuild=rebuild, check=check, tolerance=tolerance, result=details)


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Subscript | ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


# ---------------------------------------------------------------------------
# persist_source: definitions


def _string_interior_lines(source: str) -> set[int]:
    """1-based lines that begin inside a multi-line string, whose indentation is string content."""
    interior = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in _STRING_TOKENS and token.end[0] > token.start[0]:
                interior.update(range(token.start[0] + 1, token.end[0] + 1))
    except (tokenize.TokenError, SyntaxError):
        pass
    return interior


def _reindent(source: str, indent: str, newline: str) -> str:
    lines = _split_lines(source)
    interior = _string_interior_lines(source)
    code = [
        line[: len(line) - len(line.lstrip(" \t"))]
        for number, line in enumerate(lines, 1)
        if number not in interior and line.strip()
    ]
    common = os.path.commonprefix(code) if code else ""
    result = []
    for number, line in enumerate(lines, 1):
        body = line.rstrip("\r\n")
        if number in interior:
            result.append(body)
        elif not body.strip():
            result.append("")
        else:
            result.append(indent + body[len(common) :])
    return newline.join(result) + newline


def _resolve(session: Any, name: str) -> Any:
    head, *rest = name.split(".")
    scope = session._eval_scope()
    if head not in scope:
        raise ValueError(f"{head!r} is not defined in the workspace")
    value = scope[head]
    for part in rest:
        value = getattr(value, part)
    return value


def _definition(tree: ast.Module, filename: str, target: str) -> ast.stmt:
    """The single def/class statement named by ``target`` ('Name' or 'Class.method'), or raise."""
    body, where = tree.body, "at the top level"
    parts = target.split(".")
    for part in parts[:-1]:
        found = _bindings(body, part)
        if not any(isinstance(node, ast.ClassDef) for node in found):
            raise _missing(filename, body, part, "class", where)
        if len(found) > 1:
            raise ValueError(f"{part} is bound {len(found)} times in {filename} (lines {_lines_of(found)})")
        body, where = found[0].body, f"in class {part}"
    found = _bindings(body, parts[-1])
    if not found:
        raise _missing(filename, body, parts[-1], "def or class", where)
    if len(found) > 1:
        raise ValueError(
            f"{target} is bound {len(found)} times in {filename} (lines {_lines_of(found)}); "
            "persist_source replaces a single definition"
        )
    node = found[0]
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        raise ValueError(
            f"{target} is bound by a {type(node).__name__} statement on line {node.lineno} of {filename}, "
            "not by a def or class"
        )
    return node


def _undefined_names(source: str, script: _Script) -> list[str]:
    """Global names the new definition reads that the script does not bind (facts for NameErrors at run time)."""
    if _star_import(script.tree.body):
        return []
    referenced = set()

    def visit(table: symtable.SymbolTable, top: bool) -> None:
        for symbol in table.get_symbols():
            bound = symbol.is_assigned() or symbol.is_imported()
            if (top or symbol.is_global()) and symbol.is_referenced() and not bound:
                referenced.add(symbol.get_name())
        for child in table.get_children():
            visit(child, False)

    visit(symtable.symtable(source, "<persist_source>", "exec"), True)
    known = _all_bound(script.tree.body) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    return sorted(referenced - known)


def persist_source(
    session: Any,
    obj: Any,
    *,
    target: str | None = None,
    rebuild: bool = True,
    check: str | Callable | None = None,
    tolerance: float = 1e-6,
    path: str | Path | None = None,
) -> dict:
    """Implementation of :meth:`SimulationSession.persist_source`."""
    if isinstance(obj, str):
        obj = _resolve(session, obj)
    if isinstance(obj, staticmethod | classmethod | wp.Kernel | wp.Function):
        # Warp kernels and functions keep the decorated Python function as ``func``.
        obj = obj.func if isinstance(obj, wp.Kernel | wp.Function) else obj.__func__
    if inspect.ismethod(obj):
        obj = obj.__func__
    if not (inspect.isfunction(obj) or inspect.isclass(obj)) or obj.__name__ == "<lambda>":
        raise TypeError("persist_source takes a function or class defined with def/class, or its workspace name")
    qualname = obj.__qualname__
    if target is None:
        if "<locals>" in qualname:
            raise ValueError(f"{qualname} is defined inside a function; pass target='Name' or 'Class.method'")
        target = qualname
    if not all(part.isidentifier() for part in target.split(".")):
        raise ValueError("target must be 'Name' or a dotted path such as 'Example.step'")
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError) as error:
        raise ValueError(f"The source of {qualname} is unavailable ({error})") from error
    source = _reindent(source, "", "\n")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        raise ValueError(f"The source found for {qualname} is not a single def or class statement")
    new_node = tree.body[0]
    leaf = target.rsplit(".", 1)[-1]
    if new_node.name != leaf:
        lines = _split_lines(source)
        keyword_ = "class" if isinstance(new_node, ast.ClassDef) else r"(?:async\s+)?def"
        pattern = rf"^(\s*{keyword_}\s+){re.escape(new_node.name)}\b"
        lines[new_node.lineno - 1] = re.sub(pattern, rf"\g<1>{leaf}", lines[new_node.lineno - 1], count=1)
        source = "".join(lines)
        tree = ast.parse(source)
        new_node = tree.body[0]
    script = _Script(script_path(session, path))
    old = _definition(script.tree, script.path.name, target)
    first = min([old.lineno, *(decorator.lineno for decorator in old.decorator_list)])
    tail = script.lines[old.end_lineno - 1].encode("utf-8")[old.end_col_offset :].decode("utf-8").strip()
    head = script.lines[first - 1].lstrip(" \t")
    if (tail and not tail.startswith("#")) or not head.startswith(("@", "def", "async", "class")):
        raise ValueError(f"{target} shares a line with other statements in {script.path.name}; split them first")
    block = _reindent(source, script.indent(first), script.newline(first))
    end = script.starts[old.end_lineno]
    if not script.lines[old.end_lineno - 1].endswith(("\n", "\r")):
        block = block.rstrip("\r\n")
    text = script.text[: script.starts[first - 1]] + block + script.text[end:]
    # Re-parse to prove the file still compiles and now holds exactly the new definition.
    try:
        rewritten = _definition(ast.parse(text), script.path.name, target)
    except SyntaxError as error:
        raise RuntimeError(f"Internal error: the edited script does not parse ({error}); it was not changed") from error
    if ast.dump(rewritten) != ast.dump(new_node):
        raise RuntimeError(f"Internal error: replacing {target} did not round-trip; {script.path} was not changed")
    details = {"target": target, "lines": [first, old.end_lineno], "kind": type(new_node).__name__}
    undefined = _undefined_names(source, script)
    if undefined:
        details["names_not_defined_in_script"] = undefined
    if inspect.isclass(obj):
        # Methods assigned to the class from other cells are not part of its class statement.
        cells, own = f"<{session._workspace_name}:", inspect.getsourcefile(obj)
        elsewhere = sorted(
            name
            for name, value in vars(obj).items()
            if inspect.isfunction(value) and value.__code__.co_filename.startswith(cells)
            if value.__code__.co_filename != own
        )
        if elsewhere:
            details["methods_from_other_cells"] = elsewhere
    return _commit(session, script, text, rebuild=rebuild, check=check, tolerance=tolerance, result=details)


# ---------------------------------------------------------------------------
# diff_model


_DOCUMENTED_FLAGS: dict[str, ModelFlags] = {
    # Fields listed in the ModelFlags docstrings, beyond the editable ones.
    **dict.fromkeys(("joint_q", "joint_X_p", "joint_X_c"), ModelFlags.JOINT_PROPERTIES),
    **dict.fromkeys(("body_q", "body_qd", "body_flags"), ModelFlags.BODY_PROPERTIES),
    **dict.fromkeys(
        ("body_com", "body_inertia", "body_inv_inertia", "body_mass", "body_inv_mass"),
        ModelFlags.BODY_INERTIAL_PROPERTIES,
    ),
    **dict.fromkeys(("shape_transform", "shape_scale", "shape_collision_radius"), ModelFlags.SHAPE_PROPERTIES),
    **dict.fromkeys(
        (
            "mujoco.equality_constraint_anchor",
            "mujoco.equality_constraint_relpose",
            "mujoco.equality_constraint_polycoef",
            "mujoco.equality_constraint_torquescale",
            "mujoco.equality_constraint_enabled",
            "mujoco.eq_solref",
            "mujoco.eq_solimp",
            "joint_mimic_coeffs",
            "constraint_mimic_coef0",
            "constraint_mimic_coef1",
            "constraint_mimic_enabled",
        ),
        ModelFlags.CONSTRAINT_PROPERTIES,
    ),
}

_BASELINE_BYTES = 128 * 1024 * 1024


def _model_values(model: Model) -> tuple[dict[str, wp.array], dict[str, Any]]:
    arrays, scalars = {}, {}
    for name, value in vars(model).items():
        if name.startswith("_"):
            continue
        if isinstance(value, wp.array):
            arrays[name] = value
        elif type(value) in (int, float, bool):
            scalars[name] = value
        elif isinstance(value, Model.AttributeNamespace):
            for child, item in vars(value).items():
                if child.startswith("_"):
                    continue
                if isinstance(item, wp.array):
                    arrays[f"{name}.{child}"] = item
                elif type(item) in (int, float, bool):
                    scalars[f"{name}.{child}"] = item
    return arrays, scalars


def _host(array: wp.array) -> np.ndarray:
    data = array.numpy()
    # CPU arrays return a view of live memory.
    return data.copy() if array.device.is_cpu else data


class ModelBaseline:
    """Host copies of a model's public arrays and scalars; arrays beyond a byte budget keep only a digest."""

    def __init__(self, model: Model):
        self.model = weakref.ref(model)
        arrays, self.scalars = _model_values(model)
        self.arrays = {}
        total = 0
        for name, array in sorted(arrays.items(), key=lambda item: item[1].capacity):
            data = _host(array)
            total += data.nbytes
            digest = None
            if total > _BASELINE_BYTES:
                digest, data = hashlib.blake2b(np.ascontiguousarray(data), digest_size=16).digest(), None
            self.arrays[name] = (weakref.ref(array), tuple(array.shape), str(array.dtype), data, digest)


def _frequency(model: Model, name: str, rows: int) -> Any:
    if name == "gravity":
        return Model.AttributeFrequency.WORLD
    try:
        frequency = model.get_attribute_frequency(name.replace(".", ":", 1))
    except KeyError:
        return None
    try:
        count = model._attribute_frequency_count(frequency)
    except (KeyError, ValueError, AttributeError):
        return None
    return frequency if count == rows else None


def _row_keys(model: Model, frequency: Any, rows: np.ndarray) -> list[str]:
    """Entity labels for array rows, suffixed with ``@world`` in multi-world models."""
    name = getattr(frequency, "name", frequency)
    labels = worlds = None
    if name in ("BODY", "SHAPE", "JOINT", "ARTICULATION"):
        labels = getattr(model, f"{name.lower()}_label", None)
        worlds = getattr(model, f"{name.lower()}_world", None)
        names = [None if labels is None or row >= len(labels) else labels[row] for row in rows]
    elif name in ("JOINT_DOF", "JOINT_COORD"):
        starts = (model.joint_qd_start if name == "JOINT_DOF" else model.joint_q_start).numpy()
        joints = np.searchsorted(starts, rows, side="right") - 1
        worlds = model.joint_world.numpy()[joints]
        names = []
        for row, joint in zip(rows, joints, strict=True):
            label = model.joint_label[joint] if joint < len(model.joint_label) else f"joint {joint}"
            width = starts[joint + 1] - starts[joint] if joint + 1 < len(starts) else 1
            names.append(label if width == 1 else f"{label}[{row - starts[joint]}]")
        worlds = list(worlds)
        return _keys(model, rows, names, worlds)
    elif name == "WORLD":
        names = ["global" if row == model.world_count else f"world {row}" for row in rows]
        return _keys(model, rows, names, None)
    elif isinstance(name, str) and ":" in name:
        namespace, kind = name.split(":", 1)
        container = getattr(model, namespace, None)
        labels = getattr(container, f"{kind}_label", None)
        worlds = getattr(container, f"{kind}_world", None)
        names = [None if not isinstance(labels, list) or row >= len(labels) else labels[row] for row in rows]
    else:
        names = [None] * len(rows)
    if isinstance(worlds, wp.array):
        worlds = worlds.numpy()
        worlds = [int(worlds[row]) if row < len(worlds) else None for row in rows]
    else:
        worlds = None
    return _keys(model, rows, names, worlds)


def _keys(model: Model, rows: np.ndarray, names: list, worlds: list | None) -> list[str]:
    keys = []
    for index, row in enumerate(rows):
        key = names[index] if names[index] else f"#{row}"
        if model.world_count > 1 and worlds is not None and worlds[index] is not None:
            key += "@global" if worlds[index] < 0 else f"@{worlds[index]}"
        keys.append(key)
    seen = collections.Counter(keys)
    return [key if seen[key] == 1 else f"{key}#{row}" for key, row in zip(keys, rows, strict=True)]


def _flag(name: str) -> ModelFlags | None:
    from .session import SimulationSession  # noqa: PLC0415

    return SimulationSession._EDIT_FLAGS.get(name, _DOCUMENTED_FLAGS.get(name))


def diff_model(session: Any, since: str = "build", *, limit: int = 16) -> dict:
    """Implementation of :meth:`SimulationSession.diff_model`."""
    if since not in ("build", "last"):
        raise ValueError("since must be 'build' or 'last'")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 4096:
        raise ValueError("limit must be an integer in [0, 4096]")
    baselines = session._model_baselines
    used = "last" if since == "last" and "last" in baselines else "build"
    baseline = baselines.get(used)
    if baseline is None:
        raise ValueError(f"No model baseline was recorded at build time ({baselines.get('error', 'unavailable')})")
    model = session.model
    arrays, scalars = _model_values(model)
    fields, flags, unflagged = {}, 0, []
    for name, array in arrays.items():
        entry = baseline.arrays.get(name)
        if entry is None:
            fields[name] = {"added": True, "shape": list(array.shape)}
            continue
        reference, shape, dtype, old, digest = entry
        report = {}
        if reference() is not array:
            report["replaced"] = True
        if tuple(array.shape) != shape or str(array.dtype) != dtype:
            report.update(shape=[list(shape), list(array.shape)], dtype=[dtype, str(array.dtype)])
        elif old is None:
            current = hashlib.blake2b(np.ascontiguousarray(array.numpy()), digest_size=16).digest()
            if current != digest:
                report["changed"] = "yes (array beyond the baseline budget; rows were not compared)"
        else:
            new = array.numpy()
            different = old != new
            if old.dtype.kind in "fc":
                different &= ~(np.isnan(old) & np.isnan(new))
            rows = np.flatnonzero(different.reshape(len(old), -1).any(axis=1)) if old.size else np.zeros(0, int)
            if rows.size:
                report["changed_rows"] = int(rows.size)
                report["rows"] = len(old)
                shown = rows[:limit]
                frequency = _frequency(model, name, len(old))
                keys = _row_keys(model, frequency, shown) if frequency is not None else [f"#{r}" for r in shown]
                report["values"] = {
                    key: [_plain(old[row]), _plain(new[row])] for key, row in zip(keys, shown, strict=True)
                }
        if not report:
            continue
        flag = _flag(name)
        report["flag"] = flag.name if flag is not None else None
        if flag is not None:
            flags |= int(flag)
        else:
            unflagged.append(name)
        fields[name] = report
    for name in baseline.arrays.keys() - arrays.keys():
        fields[name] = {"removed": True}
    for name, value in scalars.items():
        old = baseline.scalars.get(name, MISSING)
        if old is MISSING or old != value or type(old) is not type(value):
            flag = _flag(name)
            fields[name] = {"values": [None if old is MISSING else old, value], "flag": flag.name if flag else None}
            if flag is not None:
                flags |= int(flag)
            else:
                unflagged.append(name)
    baselines["last"] = ModelBaseline(model)
    result = {
        "since": used,
        "fields": dict(sorted(fields.items())),
        "flags": [flag.name for flag in ModelFlags if flag != ModelFlags.ALL and flags & int(flag)],
        "flags_value": flags,
    }
    if baseline.model() is not model:
        result["model_replaced"] = True
    if unflagged:
        result["fields_without_flag"] = sorted(unflagged)
    return result
