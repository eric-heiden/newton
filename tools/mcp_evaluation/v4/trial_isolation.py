# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fair-start helpers for the paired MCP-vs-restart loop trials (proposed for tools/mcp_evaluation/v4).

- ``trial_env``: environment without the operator session's variables, with every compile and shader
  cache redirected into the trial's own directory.
- ``seed_caches``: compile caches after one starter run, built once per task and harness commit and
  copied into both conditions, so neither inherits the other's (or an earlier trial's) compile work.
- ``sandbox``: bwrap wrapper with a private /tmp, read-only source tree and venv, and hidden ground
  truth, other trials, and operator transcripts. /proc cannot be remounted in this container, so
  ``ps`` still shows other processes; trial names and command lines must therefore be opaque.
- ``kill_tagged``: stops processes the agent detached into other sessions (Claude Code runs each
  shell command in its own session, so killing the agent's process group misses them).
- ``pair_barrier``: both conditions launch their agents at the same instant, after both are ready.
- ``ResourceSampler``: load, the trial's CPU seconds, and live trials sampled through the whole trial.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

HOME = Path.home()
SEEDS = Path(os.environ.get("NEWTON_CACHE_SEEDS", HOME / ".cache/newton-trial-seeds"))
# Env var -> cache subdirectory. NVIDIA's GL/Vulkan shader cache is ignored unless the directory exists.
CACHE_DIRS = {
    "WARP_CACHE_PATH": "warp",
    "CUDA_CACHE_PATH": "cuda",
    "__GL_SHADER_DISK_CACHE_PATH": "nvgl",
    "OPTIX_CACHE_PATH": "optix",
    "MESA_SHADER_CACHE_DIR": "mesa",
    "XDG_CACHE_HOME": "xdg",
    "NEWTON_CACHE_PATH": "newton-assets",
}
# Inherited from the operator's Claude Code session or the pod; none of these belong in a trial.
DROP_PREFIXES = (
    "CLAUDE",
    "AWS_",
    "PM_REMOTE_",
    "HUB_",
    "VM_",
    "SSH_",
    "AI_AGENT",
    "AGENT_",
    "SESSION_ID",
    "TASK_TIMEOUT",
    "S3_",
    "DATABASE_",
    "CLOUDCLI_",
    "PYTHONPATH",
)


def cache_env(directory: Path) -> dict[str, str]:
    env = {}
    for var, name in CACHE_DIRS.items():
        (directory / name).mkdir(parents=True, exist_ok=True)
        env[var] = str(directory / name)
    env["__GL_SHADER_DISK_CACHE"] = "1"
    env["__GL_SHADER_DISK_CACHE_SKIP_CLEANUP"] = "1"
    # XDG_CACHE_HOME moves uv's cache too; keep the shared one (uv run --no-sync only reads it).
    env["UV_CACHE_DIR"] = str(HOME / ".cache/uv")
    env["UV_NO_SYNC"] = "1"
    return env


def cli_homes(trial_root: Path) -> dict[str, str]:
    """Per-trial CLI state with the operator's credentials but none of its history.

    Codex gets its own CODEX_HOME holding only auth.json and config.toml (the operator's thread history and
    log databases name study results), and Claude a copy of ~/.claude.json without per-project entries;
    ``sandbox`` binds that copy over the original and hides the rest of both CLIs' operator state.
    """
    codex = trial_root / "codex-home"
    codex.mkdir(parents=True, exist_ok=True)
    for name in ("auth.json", "config.toml"):
        if (HOME / ".codex" / name).exists():
            shutil.copy2(HOME / ".codex" / name, codex / name)
    claude = HOME / ".claude.json"
    if claude.exists():
        state = json.loads(claude.read_text())
        for key in ("projects", "skillUsage", "pluginUsage", "githubRepoPaths"):
            state.pop(key, None)
        (trial_root / "claude.json").write_text(json.dumps(state))
    return {"CODEX_HOME": str(codex)}


def trial_env(root: Path, caches: Path, trial_id: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROP_PREFIXES)}
    env["PYTHONPATH"] = str(root)
    env["NEWTON_TRIAL_ID"] = trial_id
    env.update(cache_env(caches))
    env.update(extra or {})
    return env


def provenance(root: Path, python: Path) -> dict:
    """Code, packages, and driver a trial runs on; also keys the cache seeds."""

    def run(*command: str) -> str:
        try:
            return subprocess.run(command, capture_output=True, text=True, cwd=root, check=False).stdout
        except OSError:
            return ""

    diff = run("git", "diff", "HEAD") + run("git", "status", "--porcelain", "-uall")
    for line in run("git", "ls-files", "--others", "--exclude-standard").splitlines():
        path = root / line
        if path.is_file() and path.stat().st_size < 1_000_000:
            diff += f"\n--- untracked {line}\n" + path.read_text(errors="replace")
    return {
        "commit": run("git", "rev-parse", "HEAD").strip(),
        "dirty_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "venv_sha256": hashlib.sha256(run("uv", "pip", "freeze", "--python", str(python)).encode()).hexdigest(),
        "driver": run("nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader").strip(),
        "claude_cli": run("claude", "--version").strip(),
        "codex_cli": run("codex", "--version").strip(),
        "diff": diff,
    }


def _harness_key(root: Path, python: Path, files: dict[str, Path]) -> str:
    """Identity of the code, packages, driver, and starter files the seed is compiled from."""
    info = provenance(root, python)
    digest = hashlib.sha256("|".join(info[k] for k in ("commit", "dirty_sha256", "venv_sha256", "driver")).encode())
    for target in sorted(files):
        source = Path(files[target])
        for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
            if path.is_file() and path.suffix in (".py", ".xml", ".json", ".usd", ".usda"):
                digest.update(target.encode() + path.read_bytes())
    return digest.hexdigest()[:16]


def copy_files(files: dict[str, Path], workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for target, source in files.items():
        if Path(source).is_dir():
            shutil.copytree(source, workspace / target)
        else:
            shutil.copyfile(source, workspace / target)


def seed_caches(
    name: str, files: dict[str, Path], warmup: list[list[str]], python: Path, root: Path, *, timeout: float = 3600
) -> Path:
    """Caches after the task's warm-up commands ran once on the starter (built once, under a lock)."""
    seed = SEEDS / f"{name}-{_harness_key(root, python, files)}"
    SEEDS.mkdir(parents=True, exist_ok=True)
    with (SEEDS / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (seed / "complete.json").exists():
            shutil.rmtree(seed, ignore_errors=True)
            scratch = seed / "workspace"
            copy_files(files, scratch)
            env = trial_env(root, seed / "caches", f"seed-{name}")
            log = []
            for command in warmup:
                started = time.perf_counter()
                result = subprocess.run(
                    [str(python), *command],
                    cwd=scratch,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                log.append({"command": command, "seconds": time.perf_counter() - started, "rc": result.returncode})
                if result.returncode != 0:
                    raise RuntimeError(f"seed warm-up failed: {command}\n{result.stderr[-2000:]}")
            shutil.rmtree(scratch)
            (seed / "complete.json").write_text(json.dumps(log, indent=2) + "\n")
    return seed / "caches"


def sandbox(
    command: list[str],
    run_dir: Path,
    root: Path,
    private: Path,
    *,
    extra_ro: list[Path] = (),
    extra_hidden: list[Path] = (),
) -> list[str]:
    """Run ``command`` with only ``run_dir`` writable among the study's paths.

    ``run_dir/tmp``, ``run_dir/var-tmp`` and ``run_dir/shm`` become /tmp, /var/tmp and /dev/shm, so the
    MCP host and the agent of one trial share them while other trials and the operator do not.
    """
    for name in ("tmp", "var-tmp", "shm"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    args = ["bwrap", "--dev-bind", "/", "/", "--die-with-parent"]
    # On this pod /tmp and /var/tmp are binds of ~/.horde-tmp and ~/.horde-var-tmp, which `/` exposes again.
    hidden = [
        HOME / "apps",
        HOME / "repos",
        HOME / "artifacts",
        HOME / ".horde-tmp",
        HOME / ".horde-var-tmp",
        HOME / ".claude/projects",
        HOME / ".claude/sessions",
        HOME / ".claude/session-env",
        HOME / ".claude/shell-snapshots",
        HOME / ".claude/backups",
        HOME / ".codex",
        SEEDS,
        private,
        *extra_hidden,
    ]
    for path in hidden:
        if path.is_dir():
            args += ["--tmpfs", str(path)]
    # Empty files, not /dev/null: read-only binds are nodev, so a masked /dev/null would fail to open.
    empty = run_dir / ".empty"
    empty.write_text("")
    for history in (
        HOME / ".codex/history.jsonl",
        HOME / ".claude/history.jsonl",
        HOME / "AGENTS.md",
        HOME / "CLAUDE.md",
    ):
        if history.is_file():
            args += ["--ro-bind", str(empty), str(history)]
    args += ["--ro-bind", str(root), str(root)]
    # The venv's base interpreter, the CLIs, and installed tools must not change between trials.
    shared = (HOME / "opt", HOME / ".local/share/uv/python", HOME / ".local/bin", HOME / ".cache/uv")
    for path in (*shared, *extra_ro):
        if Path(path).exists():
            args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(run_dir), str(run_dir)]
    if (run_dir / "claude.json").exists():
        args += ["--bind", str(run_dir / "claude.json"), str(HOME / ".claude.json")]
    args += ["--bind", str(run_dir / "tmp"), "/tmp", "--bind", str(run_dir / "var-tmp"), "/var/tmp"]
    args += ["--bind", str(run_dir / "shm"), "/dev/shm"]
    return [*args, *command]


def cli_child(pid: int) -> int | None:
    """The process bwrap started (the CLI), so signals reach it rather than bwrap."""
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        return None
    for child in children:
        try:
            name = Path(f"/proc/{child}/comm").read_text().strip()
        except OSError:
            continue
        if name == "bwrap":
            return cli_child(int(child))
        return int(child)
    return None


def mount_namespace(pid: int | None) -> str | None:
    """The mount namespace of a process (``mnt:[inode]``), e.g. of a trial's sandboxed child."""
    try:
        return os.readlink(f"/proc/{pid}/ns/mnt") if pid else None
    except OSError:
        return None


def kill_tagged(trial_id: str, grace: float = 5.0, namespaces: set[str] = frozenset()) -> list[int]:
    """SIGTERM, then SIGKILL, every process of this trial (exact PIDs only).

    A process belongs to the trial if it carries the trial's NEWTON_TRIAL_ID or lives in one of the
    trial's sandbox mount ``namespaces`` (detached jobs can clear their environment, not their namespace).
    """
    needle = f"NEWTON_TRIAL_ID={trial_id}".encode()
    own = mount_namespace(os.getpid())
    namespaces = {ns for ns in namespaces if ns and ns != own}

    def tagged() -> list[int]:
        pids = []
        for proc in Path("/proc").iterdir():
            if proc.name.isdigit() and int(proc.name) != os.getpid():
                try:
                    if needle in (proc / "environ").read_bytes().split(b"\0"):
                        pids.append(int(proc.name))
                        continue
                except OSError:
                    pass
                if namespaces and mount_namespace(int(proc.name)) in namespaces:
                    pids.append(int(proc.name))
        return pids

    found = tagged()
    for pid in found:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline and tagged():
        time.sleep(0.2)
    for pid in tagged():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return found


def pair_barrier(directory: Path, party: str, parties: int, timeout: float = 900) -> float:
    """Block until every condition of the pair is ready; returns the common release time [unix s]."""
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "release").exists() or (directory / party).exists():
        raise RuntimeError(f"pair barrier {directory} was already used; launch with a fresh barrier directory")
    (directory / party).write_text(str(time.time()))
    deadline = time.time() + timeout
    while len([p for p in directory.iterdir() if p.name != "release"]) < parties:
        if time.time() > deadline:
            raise RuntimeError(f"pair barrier {directory} timed out")
        time.sleep(0.1)
    release = directory / "release"
    try:
        with release.open("x") as handle:
            handle.write(str(time.time()))
    except FileExistsError:
        pass
    while not release.read_text().strip():
        time.sleep(0.01)
    return float(release.read_text())


def _tagged_cpu() -> dict[str, dict[int, float]]:
    """CPU seconds (utime + stime) of live processes per NEWTON_TRIAL_ID, by PID."""
    ticks = os.sysconf("SC_CLK_TCK")
    usage: dict[str, dict[int, float]] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            environ = (proc / "environ").read_bytes().split(b"\0")
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        for entry in environ:
            if entry.startswith(b"NEWTON_TRIAL_ID="):
                trial = entry.split(b"=", 1)[1].decode()
                usage.setdefault(trial, {})[int(proc.name)] = (int(stat[11]) + int(stat[12])) / ticks
                break
    return usage


class ResourceSampler(threading.Thread):
    """Append load, the trial's accumulated CPU seconds, and the number of live trials, every ``period`` s.

    CPU is accumulated per PID across samples, so processes that exit between samples still count up to
    their last sample. Samples are taken at start and at stop too.
    """

    def __init__(self, path: Path, trial_id: str, period: float = 10.0):
        super().__init__(daemon=True)
        self.path, self.trial_id, self.period, self.stopped = path, trial_id, period, threading.Event()
        self.cpu: dict[int, float] = {}
        self.finished = threading.Event()

    def _sample(self, out) -> None:
        usage = _tagged_cpu()
        self.cpu.update(usage.get(self.trial_id, {}))
        sample = {
            "t": time.time(),
            "load": os.getloadavg(),
            "trial_cpu_s": round(sum(self.cpu.values()), 2),
            "live_trials": len(usage),
        }
        out.write(json.dumps(sample) + "\n")
        out.flush()

    def run(self) -> None:
        with self.path.open("a") as out:
            self._sample(out)
            while not self.stopped.wait(self.period):
                self._sample(out)
            self._sample(out)
        self.finished.set()

    def stop(self) -> None:
        self.stopped.set()
        self.finished.wait(timeout=30)


def warm(script: str, args: list[str]) -> None:
    """Build and step the starter under the module names a trial uses, so its kernels land in the seed.

    Warp caches kernels per module name: the MCP host loads the script as ``_newton_hosted_<stem>_1`` and
    helper scripts import it as ``<stem>``, while the plain warm-up run caches it as ``__main__``.
    """
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import newton.examples  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415
    from newton.mcp import ExampleHost  # noqa: PLC0415

    host = ExampleHost(script, args)
    host.build()
    host.example.step()
    sys.path.insert(0, str(Path(script).resolve().parent))
    module = importlib.import_module(Path(script).stem)
    cls = module.Example
    parser = cls.create_parser() if hasattr(cls, "create_parser") else newton.examples.create_parser()
    parsed, _ = parser.parse_known_args([*args, "--viewer", "null"])
    cls(newton.viewer.ViewerNull(), parsed).step()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "warm":
        warm(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit("usage: python -m tools.mcp_evaluation.v4.trial_isolation warm SCRIPT [ARGS...]")
