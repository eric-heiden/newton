# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Workspace snapshots taken during a trial, and the search for the first one that passes verification.

During a trial, :class:`WorkspaceSnapshots` copies the agent's workspace into the run directory at the budget
start, every ``period`` seconds of the budget (default 300 s), and once after the agent ends (before the trial's
own verification). The copies are made by the harness process into the run directory, which the agent's
sandbox hides; the agent's processes, files, and timing do not change.

Layout of ``RUN_DIR/snapshots/``:

- ``objects/<sha256>``: file contents, each stored once.
- ``t<seconds>.json``: one manifest per snapshot: ``name``, ``seconds`` (since the budget start), ``unix``,
  ``final`` (taken after the agent ended), ``files`` (relative path -> ``{"sha256", "size", "mode"}`` or
  ``{"link": target}``), ``skipped`` (files not copied, with the reason), ``copy_seconds``.
- ``caches/``: the trial's compile caches after its own verification, so snapshot verifications see the
  kernels that verification saw (verifiers time setup and rollouts).

After an iteration, ``python -m tools.mcp_evaluation.v4.run_v4 --verify-snapshots RUN_DIR [...] [--all]``
verifies the snapshots with the task's sandboxed verifier and writes ``RUN_DIR/snapshot_verification.json``.
:func:`search` is the search it runs. Consecutive snapshots whose submitted files are identical (the task's
``snapshot_ignore`` patterns name files its verifier does not read) form one version, verified once.

``snapshot_verification.json`` fields:

- ``first_pass_seconds``: seconds after the budget start of the first snapshot that passed verification;
  ``None`` if none passed. The submission first passed in (``last_fail_seconds``, ``first_pass_seconds``].
- ``last_fail_seconds``: seconds of the snapshot before the first passing one (``None`` if that is the first).
- ``mode``: ``binary`` (default) or ``all``. Binary mode assumes that once a version passes, every later version
  passes (``assumption``), and verifies O(log n) versions; ``all`` verifies every version, so a pass that a
  later edit broke is found too.
- ``snapshots``: every snapshot with its version, submission digest, and result (``success``; ``source`` says
  whether it was verified, equal to a verified version, taken from the trial's own verification, or inferred
  from the monotonicity assumption, in which case ``success`` is ``None`` and ``inferred_success`` is set).
- ``verifications``: every verification run, in order, with its duration and result.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

PERIOD = 300.0
MAX_FILE_BYTES = 200_000_000
"""Larger files are not copied (verifiers skip files of 100-200 MB and more)."""
SKIPPED_DIRS = ("__pycache__",)


class WorkspaceSnapshots:
    """Copy ``workspace`` into ``directory`` now and then, deduplicating file contents.

    Args:
        workspace: The agent's workspace.
        directory: Where snapshots are stored (``RUN_DIR/snapshots``).
        period: Seconds of budget between snapshots.
    """

    def __init__(self, workspace: Path, directory: Path, period: float = PERIOD):
        self.workspace, self.directory, self.period = Path(workspace), Path(directory), period
        self.objects = self.directory / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.errors: list[str] = []
        self._hashes: dict[tuple, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _store(self, path: Path) -> str:
        """Copy one file into the object store while hashing it; returns its SHA-256."""
        temporary = self.objects / f".tmp-{threading.get_ident()}"
        digest = hashlib.sha256()
        with path.open("rb") as source, temporary.open("wb") as target:
            while chunk := source.read(1 << 20):
                digest.update(chunk)
                target.write(chunk)
        name = digest.hexdigest()
        if (self.objects / name).exists():
            temporary.unlink()
        else:
            temporary.replace(self.objects / name)
        return name

    def _entry(self, path: Path, relative: str, skipped: list[dict]) -> dict | None:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            return {"link": os.readlink(path)}
        if not stat.S_ISREG(info.st_mode):
            return None
        if info.st_size > MAX_FILE_BYTES:
            skipped.append({"path": relative, "reason": f"{info.st_size} bytes"})
            return None
        key = (relative, info.st_size, info.st_mtime_ns, info.st_ino)
        name = self._hashes.get(key)
        if name is None:
            # A file the agent rewrites while it is copied is copied again (twice at most).
            for _ in range(3):
                name = self._store(path)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) == (info.st_size, info.st_mtime_ns):
                    break
                info = after
            else:
                skipped.append({"path": relative, "reason": "changed while copied"})
            key = (relative, info.st_size, info.st_mtime_ns, info.st_ino)
            self._hashes[key] = name
        return {"sha256": name, "size": info.st_size, "mode": stat.S_IMODE(info.st_mode)}

    def _scan(self) -> tuple[dict, list[dict]]:
        files, skipped = {}, []
        for root, directories, names in os.walk(self.workspace):
            directories[:] = sorted(d for d in directories if d not in SKIPPED_DIRS)
            # Links to directories are recorded as links (os.walk does not descend into them).
            for name in sorted(names) + [d for d in directories if (Path(root) / d).is_symlink()]:
                path = Path(root) / name
                relative = path.relative_to(self.workspace).as_posix()
                try:
                    entry = self._entry(path, relative, skipped)
                except OSError as error:
                    skipped.append({"path": relative, "reason": repr(error)})
                    continue
                if entry is not None:
                    files[relative] = entry
        return files, skipped

    def prime(self) -> None:
        """Store the workspace's current files without recording a snapshot, so later snapshots copy less."""
        with self._lock:
            self._scan()

    def take(self, seconds: float, *, final: bool = False) -> dict:
        """Snapshot the workspace now, labelled ``seconds`` after the budget start."""
        with self._lock:
            started = time.perf_counter()
            files, skipped = self._scan()
            name = f"t{round(seconds):05d}"
            while (self.directory / f"{name}.json").exists():
                name += "+"
            manifest = {
                "name": name,
                "seconds": seconds,
                "unix": time.time(),
                "final": final,
                "files": files,
                "skipped": skipped,
                "copy_seconds": time.perf_counter() - started,
            }
            temporary = self.directory / f".{name}.json.tmp"
            temporary.write_text(json.dumps(manifest, indent=1) + "\n")
            temporary.replace(self.directory / f"{name}.json")
            return manifest

    def start(self, budget_start: float) -> None:
        """Snapshot at the budget start and every ``period`` seconds after it, in a background thread."""

        def run():
            tick = 0
            while True:
                try:
                    self.take(max(0.0, time.time() - budget_start))
                except Exception as error:
                    self.errors.append(repr(error))
                # Ticks missed while copying are skipped rather than taken late in a burst.
                tick = max(tick + 1, int((time.time() - budget_start) / self.period) + 1)
                if self._stop.wait(max(0.0, budget_start + tick * self.period - time.time())):
                    return

        self._thread = threading.Thread(target=run, name="workspace-snapshots", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def load(directory: Path) -> list[dict]:
    """Snapshot manifests in ``directory``, in time order."""
    manifests = [json.loads(path.read_text()) for path in Path(directory).glob("t*.json")]
    return sorted(manifests, key=lambda manifest: (manifest["seconds"], manifest["name"]))


def materialize(directory: Path, manifest: dict, target: Path) -> None:
    """Recreate the snapshot's files under ``target`` (which must not exist yet)."""
    target.mkdir(parents=True)
    for relative, entry in manifest["files"].items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if "link" in entry:
            os.symlink(entry["link"], path)
        else:
            shutil.copyfile(Path(directory) / "objects" / entry["sha256"], path)
            path.chmod(entry["mode"])


def ignored(relative: str, patterns: Sequence[str]) -> bool:
    """Whether a workspace path matches a pattern.

    ``name/`` matches everything in a top-level directory; other patterns match the whole relative path with
    :func:`fnmatch.fnmatchcase` (``*`` also matches ``/``, so ``*.png`` matches at any depth and ``a.npz`` only at
    the top level).
    """
    for pattern in patterns:
        if relative.startswith(pattern) if pattern.endswith("/") else fnmatch.fnmatchcase(relative, pattern):
            return True
    return False


def submission_digest(manifest: dict, ignore: Sequence[str] = ()) -> str:
    """Digest of the snapshot's files except those matching ``ignore``."""
    digest = hashlib.sha256()
    for relative in sorted(manifest["files"]):
        if not ignored(relative, ignore):
            entry = manifest["files"][relative]
            digest.update(json.dumps([relative, entry.get("sha256"), entry.get("link")]).encode())
    return digest.hexdigest()


MONOTONE = "once a submission version passes verification, every later version passes"


def search(
    snapshots: list[dict],
    digests: list[str],
    check: Callable[[int], dict],
    *,
    exhaustive: bool = False,
    known: dict[str, dict] | None = None,
) -> dict:
    """Find the first snapshot whose submission passes.

    Args:
        snapshots: Manifests in time order.
        digests: Submission digest per snapshot.
        check: Verifies snapshot ``index`` and returns a result with ``success``.
        exhaustive: Verify every version instead of binary search under the assumption :data:`MONOTONE`.
        known: Results by digest from earlier verifications (updated in place).

    Returns:
        ``first_pass_seconds``, ``last_fail_seconds``, and per snapshot its ``version`` and result.
    """
    known = {} if known is None else known
    versions: list[int] = []  # index of each version's first snapshot
    for index, digest in enumerate(digests):
        if index == 0 or digest != digests[index - 1]:
            versions.append(index)

    def passes(version: int) -> bool:
        digest = digests[versions[version]]
        if digest not in known:
            known[digest] = check(versions[version])
        return bool(known[digest].get("success"))

    first = None
    if exhaustive:
        outcomes = [passes(version) for version in range(len(versions))]
        first = outcomes.index(True) if True in outcomes else None
    elif versions and passes(len(versions) - 1):
        low, high = 0, len(versions) - 1
        while low < high:
            middle = (low + high) // 2
            if passes(middle):
                high = middle
            else:
                low = middle + 1
        first = low
    rows = []
    for index, snapshot in enumerate(snapshots):
        version = max(v for v, start in enumerate(versions) if start <= index)
        row = {"name": snapshot["name"], "seconds": snapshot["seconds"], "version": version, "digest": digests[index]}
        result = known.get(digests[index])
        if result is not None:
            row["success"] = bool(result.get("success"))
            here = result.get("snapshot") == snapshot["name"]
            row["source"] = result.get("source", "verifier") if here else f"same as {result.get('snapshot')}"
        else:
            row["success"] = None
            row["inferred_success"] = first is not None and version >= first
            row["source"] = "inferred"
        rows.append(row)
    first_index = versions[first] if first is not None else None
    return {
        "first_pass_seconds": snapshots[first_index]["seconds"] if first_index is not None else None,
        "first_pass_snapshot": snapshots[first_index]["name"] if first_index is not None else None,
        "last_fail_seconds": snapshots[first_index - 1]["seconds"] if first_index else None,
        "versions": len(versions),
        "snapshots": rows,
    }
