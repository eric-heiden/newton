# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Send Python functions, code strings, and values to worker sessions in other processes.

Functions travel by source. A function defined in a trusted-execution cell is
re-executed on the worker from the cell's source (see :mod:`.cells`), together
with the cell functions and classes it uses and the session globals it reads.
Global values, arguments, closure values, and results travel as pickles. Names
that refer to a session's own live objects (``example``, ``model``, ``state``,
...) are not sent: on a worker they refer to the worker's own scene.

The worker side is :func:`serve`, which a worker runs inside an ordinary
trusted-execution cell.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import importlib
import inspect
import io
import os
import pickle
import symtable
import sys
import textwrap
import traceback
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import numpy as np
import warp as wp

from .cells import cell_source, class_functions, is_cell_filename, register_cell_source

INLINE_LIMIT = 48 * 1024
"""Largest base64 payload sent inside a request or response; larger payloads go through files."""

_WORKSPACE_PREFIX = "_newton_mcp_"
_HOSTED_PREFIX = "_newton_hosted_"
_CODE_PREFIX = "<newton-mcp-code:"
_CACHE = "__newton_shipments__"
_CODE_RESULT = "__newton_code_result__"
_MISSING = object()


# ---------------------------------------------------------------------------
# Pickling


def _warp_array(data: np.ndarray, dtype: Any, device: str) -> wp.array:
    return wp.array(data, dtype=dtype, device=device)


def _workspace_object(value: Any) -> bool:
    """Whether ``value`` is a function or class defined by trusted execution (and so has no importable module)."""
    if isinstance(value, wp.Kernel | wp.Function):
        value = getattr(value, "func", None)
    if not (isinstance(value, FunctionType) or isinstance(value, type)):
        return False
    return str(getattr(value, "__module__", "") or "").startswith(_WORKSPACE_PREFIX)


class _Pickler(pickle.Pickler):
    def __init__(self, file: io.BytesIO, found: dict[int, Any] | None):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.found = found

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, wp.array):
            data = obj.numpy()
            try:
                pickle.dumps(obj.dtype)
            except Exception:
                return np.ndarray.__reduce__(np.ascontiguousarray(data))
            return _warp_array, (data, obj.dtype, str(obj.device))
        if self.found is not None and _workspace_object(obj):
            # Pickled by reference; the receiving side needs the definition first.
            self.found[id(obj)] = obj
        return NotImplemented


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: io.BytesIO, namespace: dict | None):
        super().__init__(file)
        self.namespace = namespace

    def find_class(self, module: str, name: str) -> Any:
        # Workspace modules of other sessions (and hosted script modules) have per-process names;
        # resolve their objects in this process's workspace instead.
        if self.namespace is not None and module.startswith(_WORKSPACE_PREFIX):
            return _resolve(self.namespace, name, f"{name} (defined in a cell)")
        if self.namespace is not None and module.startswith(_HOSTED_PREFIX):
            hosted = self.namespace.get("module")
            if not isinstance(hosted, ModuleType):
                raise AttributeError(f"{name} comes from the hosted script, which this session does not have")
            return _resolve(vars(hosted), name, f"{name} (from the hosted script)")
        return super().find_class(module, name)


def _resolve(namespace: dict, qualname: str, description: str) -> Any:
    head, *rest = qualname.split(".")
    if head not in namespace:
        raise AttributeError(f"{description} is not defined in this process")
    value = namespace[head]
    for part in rest:
        value = getattr(value, part)
    return value


def dumps(value: Any, found: dict[int, Any] | None = None) -> bytes:
    """Pickle ``value``; Warp arrays travel as NumPy data and are rebuilt on their device.

    Args:
        value: Object to pickle.
        found: Collects functions and classes defined in cells that ``value`` refers to.
    """
    buffer = io.BytesIO()
    _Pickler(buffer, found).dump(value)
    return buffer.getvalue()


def loads(data: bytes, namespace: dict | None = None) -> Any:
    """Unpickle ``data``, resolving cell-defined classes and functions in ``namespace``."""
    return _Unpickler(io.BytesIO(data), namespace).load()


# ---------------------------------------------------------------------------
# Payload transfer


def put(data: bytes, directory: Path, prefix: str) -> dict:
    """Describe ``data`` inline when small, otherwise write it to a new file in ``directory``."""
    if len(data) * 4 // 3 <= INLINE_LIMIT:
        return {"inline": base64.b64encode(data).decode("ascii")}
    path = directory / f"{prefix}-{uuid.uuid4().hex}.pkl"
    path.write_bytes(data)
    return {"file": str(path)}


def get(reference: dict, *, remove: bool = False) -> bytes:
    """Read data described by :func:`put`; ``remove`` deletes its file afterwards."""
    if "inline" in reference:
        return base64.b64decode(reference["inline"])
    path = Path(reference["file"])
    data = path.read_bytes()
    if remove:
        path.unlink(missing_ok=True)
    return data


def discard(reference: dict | None) -> None:
    """Delete the file behind a :func:`put` reference, if any."""
    if reference and "file" in reference:
        Path(reference["file"]).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Function shipping (sending side)


class Shipment:
    """A callable prepared for worker sessions: the definitions it needs and the globals it reads.

    Attributes:
        key: Content hash; workers install a shipment once and reuse it while its names stay bound.
        data: Pickled payload for :func:`serve`.
        label: Human-readable name of the target.
        not_sent: Global names the target reads that were not sent, with the reason.
    """

    def __init__(self, data: bytes, label: str, not_sent: dict[str, str]):
        self.data = data
        self.key = hashlib.sha1(data).hexdigest()[:20]
        self.label = label
        self.not_sent = not_sent


def _global_names(source: str, exclude: Iterable[str] = ()) -> set[str]:
    """Global names a definition's source reads (including decorators, defaults, and base classes)."""
    table = symtable.symtable(source, "<definition>", "exec")
    names = set()

    def visit(scope, top):
        for symbol in scope.get_symbols():
            if top:
                if symbol.is_referenced():
                    names.add(symbol.get_name())
            elif symbol.is_global() and (symbol.is_referenced() or symbol.is_declared_global()):
                names.add(symbol.get_name())
        for child in scope.get_children():
            visit(child, False)

    visit(table, True)
    return names - set(exclude)


def _cell_text(filename: str, what: str) -> str:
    text = cell_source(filename)
    if text is None:
        raise ValueError(f"The source of {what} ({filename}) is no longer cached")
    return text


def _class_node(text: str, name: str, line: int | None) -> ast.ClassDef | None:
    """The class statement named ``name`` in a cell: the one spanning ``line``, else the last one."""
    found = None
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            if line is not None and start <= line <= node.end_lineno:
                return node
            if found is None or node.lineno > found.lineno:
                found = node
    return None if line is not None else found


def _class_source(cls: type) -> tuple[str, int, str]:
    """Source, first line, and cell of a class defined in a cell."""
    methods = {name: line for name, line in _method_lines(cls).items() if is_cell_filename(name)}
    module_file = getattr(sys.modules.get(cls.__module__), "__file__", None)
    if methods:
        candidates = list(methods.items())[:1]
    elif is_cell_filename(module_file):
        candidates = [(module_file, None)]
    else:
        # Classes without methods (e.g. plain dataclasses): search the workspace's cells, newest first.
        prefix = f"<{cls.__module__.split('.')[0]}:cell-"
        cells = [name for name in _cells() if name.startswith(prefix) and name[len(prefix) : -1].isdigit()]
        cells.sort(key=lambda name: int(name[len(prefix) : -1]), reverse=True)
        candidates = [(name, None) for name in cells]
    for filename, line in candidates:
        text = cell_source(filename)
        node = None if text is None else _class_node(text, cls.__name__, line)
        if node is not None:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines = text.splitlines(keepends=True)[start - 1 : node.end_lineno]
            return textwrap.dedent("".join(lines)), start, filename
    raise ValueError(f"The source of class {cls.__qualname__} is not among the cached cells")


def _cells() -> list[str]:
    import linecache  # noqa: PLC0415

    return [name for name in linecache.cache if is_cell_filename(name)]


def _method_lines(cls: type) -> dict[str, int]:
    lines = {}
    for function in class_functions(cls):
        lines.setdefault(function.__code__.co_filename, function.__code__.co_firstlineno)
    return lines


def _lambda_node(function: FunctionType, text: str) -> ast.Lambda:
    code = function.__code__
    candidates = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno
    ]
    for node in candidates:
        compiled = compile(ast.Expression(body=node), code.co_filename, "eval")
        inner = next((c for c in compiled.co_consts if inspect.iscode(c)), None)
        if inner is not None and inner.co_code == code.co_code and inner.co_consts == code.co_consts:
            return node
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot locate the source of the lambda on line {code.co_firstlineno} of {code.co_filename}")


class _Builder:
    """Collect the definitions, imports, and values a target needs on a worker."""

    def __init__(self, skip: Callable[[str, Any], bool]):
        self.skip = skip
        self.cells: dict[str, str] = {}
        self.steps: list[tuple] = []
        self.not_sent: dict[str, str] = {}
        self._defined: dict[int, str] = {}
        self._names: set[str] = set()

    def _found(self, found: dict[int, Any]) -> None:
        for value in list(found.values()):
            if id(value) not in self._defined:
                self.define(value)

    def _pickle(self, value: Any) -> bytes:
        found: dict[int, Any] = {}
        data = dumps(value, found)
        self._found(found)
        return data

    def define(self, value: Any, bind: str | None = None) -> str:
        """Add the definition of a cell function, lambda, Warp kernel/function, or class; returns its bound name."""
        if id(value) in self._defined:
            return self._defined[id(value)]
        target = getattr(value, "func", None) if isinstance(value, wp.Kernel | wp.Function) else value
        if isinstance(target, type):
            source, line, filename = _class_source(target)
            name = bind or target.__name__
            record = {"kind": "class", "name": name, "filename": filename, "line": line, "source": source}
            record["binds"] = target.__name__
            exclude: tuple = ()
        elif isinstance(target, FunctionType):
            code = target.__code__
            filename = code.co_filename
            text = _cell_text(filename, target.__qualname__)
            exclude = code.co_freevars
            decorated = False
            if target.__name__ == "<lambda>":
                node = _lambda_node(target, text)
                source = ast.get_source_segment(text, node)
                line = node.lineno
                name = bind or _hidden_name(filename, line, target)
                kind = "lambda"
            else:
                try:
                    lines, line = inspect.getsourcelines(target)
                except (OSError, TypeError) as error:
                    raise ValueError(f"The source of {target.__qualname__} is not available: {error}") from error
                source = textwrap.dedent("".join(lines))
                nested = "<locals>" in target.__qualname__
                # Nested functions are not workspace names in this session either; keep them under a private name.
                name = bind or (_hidden_name(filename, line, target) if nested else target.__name__)
                kind = "def"
                decorated = bool(ast.parse(source).body[0].decorator_list)
            record = {"kind": kind, "name": name, "filename": filename, "line": line, "source": source}
            # The name the source itself binds (the def name; the assignment target for a lambda).
            record["binds"] = name if kind == "lambda" else target.__name__
            # Registered before pickling closure values and defaults, which may refer back to it.
            self._defined[id(value)] = name
            if code.co_freevars:
                try:
                    cells = [cell.cell_contents for cell in target.__closure__]
                except ValueError as error:
                    raise ValueError(f"{target.__qualname__} uses a closure variable that is not set yet") from error
                record["freevars"] = list(code.co_freevars)
                record["closure"] = self._pickle(cells)
            if not decorated and (target.__defaults__ or target.__kwdefaults__):
                # Defaults were evaluated in this session; send their values rather than re-evaluating them.
                record["defaults"] = self._pickle((target.__defaults__, target.__kwdefaults__))
        else:
            raise TypeError(f"Cannot send {type(value).__name__} objects by source")
        self._defined[id(value)] = name
        self.cells[filename] = _cell_text(filename, name)
        namespace = target.__globals__ if isinstance(target, FunctionType) else vars(sys.modules[target.__module__])
        for global_name in sorted(_global_names(source, exclude)):
            self.reference(global_name, namespace)
        self.steps.append(("define", record))
        return name

    def reference(self, name: str, namespace: dict) -> None:
        """Send what a global ``name`` refers to, unless the worker keeps its own object of that name."""
        if name in self._names or name.startswith("__"):
            return
        self._names.add(name)
        value = namespace.get(name, _MISSING)
        if value is _MISSING or self.skip(name, value):
            return
        if isinstance(value, ModuleType):
            self.steps.append(("import", name, value.__name__))
        elif _workspace_object(value):
            bound = self.define(value, bind=name if name != getattr(value, "__name__", name) else None)
            if bound != name:
                self.steps.append(("alias", name, bound))
        elif (isinstance(value, FunctionType) or isinstance(value, type)) and str(
            getattr(value, "__module__", "")
        ).startswith(_HOSTED_PREFIX):
            self.steps.append(("hosted", name, value.__qualname__))
        else:
            try:
                self.steps.append(("value", name, self._pickle(value)))
            except Exception as error:
                self.not_sent[name] = f"not picklable: {type(error).__name__}: {str(error)[:200]}"

    def entry(self, target: Any) -> tuple:
        if _workspace_object(target):
            return ("name", self.define(target))
        if callable(target):
            return ("value", self._pickle(target))
        raise TypeError(f"Expected a function or a code string, got {type(target).__name__}")


def _hidden_name(filename: str, line: int, function: FunctionType) -> str:
    digest = hashlib.sha1(f"{filename}:{line}:{function.__qualname__}".encode()).hexdigest()[:12]
    return f"__newton_shipped_{digest}__"


def prepare(target: Any, skip: Callable[[str, Any], bool], extra: Iterable[Any] = ()) -> Shipment:
    """Build the shipment that makes ``target`` (and the cell objects in ``extra``) callable on a worker.

    Args:
        target: Function, lambda, class, Warp kernel, or other picklable callable; ``None`` ships only ``extra``.
        skip: ``skip(name, value)`` is true for globals the worker keeps its own value of.
        extra: Cell-defined functions and classes the call arguments refer to.
    """
    builder = _Builder(skip)
    for value in extra:
        builder.define(value)
    entry = ("none",) if target is None else builder.entry(target)
    label = "code" if target is None else getattr(target, "__qualname__", None) or type(target).__name__
    payload = {"cells": builder.cells, "steps": builder.steps, "entry": entry}
    return Shipment(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), str(label), builder.not_sent)


# ---------------------------------------------------------------------------
# Worker side


def _compile_definition(record: dict) -> Any:
    filename, line, source = record["filename"], record["line"], record["source"]
    if record["kind"] == "lambda":
        tree = ast.parse(f"{record['name']} = ({source}\n)", filename=filename)
        ast.increment_lineno(tree, line - 1)
    else:
        tree = ast.parse(source, filename=filename)
        ast.increment_lineno(tree, line - 1)
    if "defaults" in record:
        node = tree.body[0].value if record["kind"] == "lambda" else tree.body[0]
        node.args.defaults = []
        node.args.kw_defaults = [None] * len(node.args.kwonlyargs)
    if record.get("freevars"):
        # Rebuild the closure: define the function inside a factory whose parameters are its free variables.
        factory = ast.parse(f"def __newton_closure__({', '.join(record['freevars'])}):\n    pass\n")
        factory.body[0].body = [*tree.body, ast.Return(ast.Name(record["binds"], ast.Load()))]
        ast.fix_missing_locations(factory)
        tree = factory
    return compile(tree, filename, "exec")


def _apply(namespace: dict, step: tuple, bound: dict[str, Any]) -> None:
    kind = step[0]
    if kind == "import":
        _, name, module = step
        namespace[name] = importlib.import_module(module)
        bound[name] = namespace[name]
    elif kind == "hosted":
        _, name, qualname = step
        hosted = namespace.get("module")
        if not isinstance(hosted, ModuleType):
            raise NameError(f"{name} comes from the hosted script, which this worker does not have")
        namespace[name] = _resolve(vars(hosted), qualname, name)
        bound[name] = namespace[name]
    elif kind == "alias":
        _, name, original = step
        if original not in bound:
            raise NameError(f"{original} is not defined yet")
        namespace[name] = bound[name] = bound[original]
    elif kind == "value":
        _, name, data = step
        namespace[name] = loads(data, namespace)
        bound[name] = namespace[name]
    elif kind == "define":
        record = step[1]
        code = _compile_definition(record)
        name = record["name"]
        # Separate locals keep a nested function's own name from replacing a worker global of that name.
        scope: dict = {}
        with _shipped_module(namespace, record):
            exec(code, namespace, scope)
            if record.get("freevars"):
                function = scope["__newton_closure__"](*loads(record["closure"], namespace))
            else:
                function = scope[record["binds"]]
        namespace[name] = function
        if "defaults" in record:
            function.__defaults__, function.__kwdefaults__ = loads(record["defaults"], namespace)
        bound[name] = function
    else:
        raise ValueError(f"Unknown shipment step {kind!r}")


@contextlib.contextmanager
def _shipped_module(namespace: dict, record: dict):
    """Define a shipped object under a module name derived from its source instead of the workspace's.

    Warp names a kernel's cache entry after its module, and workspace names differ per process; the
    same name in every worker lets the processes (and restarted ones) share compiled kernels.
    """
    digest = hashlib.sha256(f"{record['kind']}\0{record['name']}\0{record['source']}".encode()).hexdigest()[:16]
    name = f"{_WORKSPACE_PREFIX}shipped_{digest}"
    workspace = sys.modules.get(str(namespace.get("__name__", "")))
    if isinstance(workspace, ModuleType):
        # Pickling by reference looks the definitions up through sys.modules; they live in the workspace.
        sys.modules[name] = workspace
    original = namespace.get("__name__", _MISSING)
    namespace["__name__"] = name
    try:
        yield
    finally:
        if original is _MISSING:
            namespace.pop("__name__", None)
        else:
            namespace["__name__"] = original


def _install(namespace: dict, payload: dict) -> tuple[Any, dict[str, Any]]:
    for filename, text in payload["cells"].items():
        register_cell_source(filename, text)
    pending, bound = list(payload["steps"]), {}
    # Definitions, values, and defaults can depend on each other in any order; retry until nothing changes.
    while pending:
        failed = []
        for step in pending:
            try:
                _apply(namespace, step, bound)
            except Exception as error:
                failed.append((step, error))
        if len(failed) == len(pending):
            step, error = failed[0]
            label = step[1]["name"] if step[0] == "define" else step[1]
            raise RuntimeError(f"Could not set up {label!r} on the worker: {type(error).__name__}: {error}") from error
        pending = [step for step, _ in failed]
    entry = payload["entry"]
    if entry[0] == "name":
        return namespace[entry[1]], bound
    if entry[0] == "value":
        return loads(entry[1], namespace), bound
    return None, bound


def _shipped(namespace: dict, reference: dict) -> Any:
    cache = namespace.setdefault(_CACHE, {})
    cached = cache.get(reference["key"])
    if cached is not None and all(namespace.get(name, _MISSING) is value for name, value in cached[1].items()):
        return cached[0]
    entry, bound = _install(namespace, pickle.loads(get(reference)))
    cache[reference["key"]] = (entry, bound)
    return entry


def _run_code(namespace: dict, code: str, arguments: Any) -> Any:
    filename = f"{_CODE_PREFIX}{hashlib.sha1(code.encode()).hexdigest()[:12]}>"
    tree = ast.parse(code, filename=filename)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.copy_location(ast.Assign([ast.Name(_CODE_RESULT, ast.Store())], last.value), last)
        ast.fix_missing_locations(tree)
    register_cell_source(filename, code)
    if arguments is not _MISSING:
        namespace["args"] = arguments
    namespace.pop(_CODE_RESULT, None)
    exec(compile(tree, filename, "exec"), namespace)
    # As in a cell, only the last expression is returned; ``result`` is an ordinary variable.
    return namespace.pop(_CODE_RESULT, None)


class _Tee(io.TextIOBase):
    """Copy printed text to a progress file as it is written."""

    def __init__(self, stream: Any, path: str):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        self.file.write(text)
        self.file.flush()
        return self.stream.write(text)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def close(self) -> None:
        self.file.close()


def _error_report(error: BaseException) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    frames = [
        {"file": frame.filename, "line": frame.lineno, "function": frame.name, "source": (frame.line or "")[:200]}
        for frame in traceback.extract_tb(error.__traceback__)
        if not os.path.abspath(frame.filename).startswith(here)
    ][-8:]
    report = {"type": type(error).__name__, "message": str(error)[:4000], "frames": frames}
    if isinstance(error, NameError) and getattr(error, "name", None):
        report["name"] = error.name
    return report


def device_error() -> str | None:
    """Error of the first CUDA device this process has used whose context no longer works, or ``None``.

    A kernel fault (e.g. CUDA error 700) is not raised by the launch; it fails every later
    allocation and copy on that device. Devices without a context are not touched.
    """
    for device in wp.get_cuda_devices() if wp.is_cuda_available() else []:
        if not device.has_context:
            continue
        try:
            wp.zeros(1, dtype=float, device=device).numpy()
        except Exception as error:
            return f"{device.alias}: {type(error).__name__}: {str(error)[:300]}"
    return None


def serve(namespace: dict, request: dict) -> dict:
    """Run one worker request inside a trusted-execution cell of the worker session.

    Args:
        namespace: The worker workspace (the cell's ``globals()``).
        request: ``exchange`` directory; ``ship`` (a shipment reference with ``key``) or ``code``;
            ``call`` (pickled ``(args, kwargs)``, or the ``args`` value of a code string), ``sync``
            (name to pickled value), and an optional ``progress`` file that receives printed text.

    Returns:
        ``{"value": reference}`` with the pickled result, or ``{"error": report}``; ``device`` reports
        a CUDA context that failed during the call. A call that raised rolls the worker's simulation back
        to the start of the call, as a failed cell would; the report's ``rollback`` says what was restored.
    """
    response = _serve(namespace, request)
    failure = device_error()
    if failure is not None:
        # The worker is restarted; its arrays cannot be copied back on a failed context.
        response["device"] = failure
    elif "error" in response:
        roll_back = getattr(namespace.get("session"), "_roll_back_cell", None)
        if roll_back is not None:
            try:
                outcome = roll_back()
            except Exception as error:
                outcome = f"Rolling the worker back failed: {type(error).__name__}: {str(error)[:300]}"
            if outcome:
                response["error"]["rollback"] = outcome
    return response


def _serve(namespace: dict, request: dict) -> dict:
    directory = Path(request["exchange"])
    try:
        for name, reference in (request.get("sync") or {}).items():
            namespace[name] = loads(get(reference), namespace)
        if request.get("sync") is not None:
            return {"value": put(dumps(sorted(request["sync"])), directory, "result")}
        function = _shipped(namespace, request["ship"]) if request.get("ship") else None
        call = loads(get(request["call"]), namespace) if request.get("call") else _MISSING
        stack = contextlib.ExitStack()
        with stack:
            if request.get("progress"):
                tee = stack.enter_context(contextlib.closing(_Tee(sys.stdout, request["progress"])))
                stack.enter_context(contextlib.redirect_stdout(tee))
                stack.enter_context(contextlib.redirect_stderr(tee))
            if request.get("code") is not None:
                value = _run_code(namespace, request["code"], call)
            else:
                args, kwargs = call
                value = function(*args, **kwargs)
        return {"value": put(dumps(value), directory, "result")}
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        return {"error": _error_report(error)}
