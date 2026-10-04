# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a submitted G1 motion-tracking controller (``g1_mpc``).

Only the submission's ``Controller`` (and its ``MotionClip``, which builds the controller's view of the
reference) is used. The verifier simulates its own plant, the unchanged starter ``g1_mpc.py`` next to this
file: the G1 model, SolverMuJoCo with 2 ms steps, and the actuator law of :class:`g1_mpc.Command`. Every
10 ms it hands the controller copies of the state, validates the returned command (finite, gains within
range; torques are clipped to the MJCF limits by the actuators), and steps the physics. The controller gets
its own copy of the model for planning and never sees the plant.

Each clip runs in a fresh process: the three given clips (walk, dance, jumpjack), two unseen ones (wave,
high5), and an unseen slower playback of walk at a speed drawn per verification (:data:`SCALED`), each under a
neutral file name. Per clip the robot must stay on its feet for the whole clip (root above 0.55 m, up-axis
cosine above 0.7, nothing but the feet on the floor), and the root position, root orientation, joint-angle,
and sole position RMSE must be within :data:`THRESHOLDS`; wrist errors, lift recall, high-frequency jitter,
and a score per clip are reported as graded metrics (a run cut short scores the part it ran). Setup and
rollout wall times are capped; the caps grow with the machine's load, measured between controller calls by
:class:`LoadProbe`, by at most :data:`LOAD_FACTOR_MAX`.

Integrity: the submission's sources may not use introspection, and an audit hook blocks frame and garbage
collector access, threads, process spawning, and access to hidden data by submitted code. Submitted code runs
only while the Controller is built and inside compute(): after every call the verifier converts the command to
plain arrays and checks that Newton, Warp, MuJoCo, NumPy, and the verifier are unpatched (no import, trace,
or signal hooks), and that the plant's state and parameters are unchanged; the submission's objects stay
alive until the process exits; after the rollout every array of the plant's MuJoCo model is compared with its
start. Each child echoes a nonce it got before importing the submission, and the parent, which never imports
the submission, recomputes every gate from the reported metrics. Hidden clips are read from
``~/.newton-visual-private/g1_mpc/clips``.
"""

from __future__ import annotations

import _thread
import argparse
import builtins
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

import newton

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The verifier's own plant, metrics, and command validation: the unchanged starter, under a private name.
plant = _load_module("g1_mpc_verifier_plant", HERE / "g1_mpc.py")

PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "g1_mpc"
# name: whether the agent has the clip.
CLIPS = {"walk": True, "dance": True, "jumpjack": True, "wave": False, "high5": False, "walk_slow": False}
# Unseen time-scaled playbacks of a given clip: name -> (source clip, speed range); the parent draws the speed
# per verification, so no plan prepared for one speed fits.
SCALED = {"walk_slow": ("walk", (0.85, 0.95))}

# Per-clip gates (every clip), from calibrate.py propose (the private reference's worst clip with headroom); the
# goal text states them.
THRESHOLDS = {
    "root_rmse_m": 0.06,
    "root_rot_rmse_deg": 10.0,
    "joint_rmse_rad": 0.12,
    "sole_rmse_m": 0.07,
}
# Graded metrics reported per clip, with the reference values used to normalize them.
GRADED = {"wrist_rmse_m": 0.10, "jitter_mrad": 20.0}
SETUP_LIMIT_S = 120.0  # wall time to import the submission and construct its Controller, per clip
ROLLOUT_LIMIT_S = 180.0  # wall time of the controlled rollout, per clip, on this machine when otherwise idle
# The machine is shared: the caps scale with the measured slowdown of fixed GPU and CPU workloads (LoadProbe)
# against their idle times here, by at most LOAD_FACTOR_MAX.
LOAD_FACTOR_MAX = 2.5
LOAD_BASELINE_S = {"gpu": 0.0050, "cpu": 0.0041}  # LoadProbe medians on the idle machine (2026-10-04)
LOAD_WORLDS, LOAD_STEPS, LOAD_REPEATS = 256, 10, 5
LOAD_PERIOD = 50  # control periods between load probes during the rollout
CHILD_TIMEOUT_S = (SETUP_LIMIT_S + ROLLOUT_LIMIT_S) * LOAD_FACTOR_MAX + 180.0
TOTAL_BUDGET_S = 3500.0  # run_v4 gives this verifier 3600 s (g1_mpc's verify_seconds)

# Modules whose functions and classes the submission must not replace.
WATCHED_ROOTS = {"newton", "warp", "mujoco", "mujoco_warp", "numpy", "math", "json", "copy", plant.__name__}
# Modules the verifier writes its result and talks to its parent with.
WATCHED_ROOTS |= {"os", "posixpath", "pathlib", "io", "sys", "secrets", "subprocess", "shutil", "tempfile", "signal"}


def clip_path(name: str) -> Path:
    return PRIVATE / "clips" / f"{name}.csv"


def scale_clip(qpos: np.ndarray, speed: float, fps: float = 30.0) -> np.ndarray:
    """A clip played at ``speed`` (below 1: slower), resampled at ``fps``: frame i shows the source at time
    i * speed / fps (linear interpolation, root quaternion normalized)."""
    duration = (len(qpos) - 1) / fps
    count = math.floor(duration / speed * fps + 1e-9) + 1
    x = np.clip(np.arange(count) * speed, 0.0, len(qpos) - 1.0)
    i = np.minimum(x.astype(int), len(qpos) - 2)
    a = (x - i)[:, None]
    out = (1.0 - a) * qpos[i] + a * qpos[i + 1]
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def clip_rows(name: str, speed: float | None = None) -> np.ndarray:
    """The clip's qpos rows (time-scaled clips: their source played at ``speed``)."""
    if name in SCALED:
        source, (low, high) = SCALED[name]
        if speed is None or not low <= speed <= high:
            raise ValueError(f"{name} needs a speed within [{low}, {high}]")
        return scale_clip(np.loadtxt(clip_path(source), delimiter=",", ndmin=2), speed)
    return np.loadtxt(clip_path(name), delimiter=",", ndmin=2)


def draw_speed(name: str) -> float | None:
    """A speed for a time-scaled clip, drawn per verification (``None`` for recorded clips)."""
    if name not in SCALED:
        return None
    low, high = SCALED[name][1]
    return round(low + (high - low) * secrets.randbelow(1001) / 1000.0, 3)


def expected_samples(name: str, speed: float | None = None) -> int:
    """Control periods of a clip, from its row count (the parent computes this without the plant)."""
    return round((len(clip_rows(name, speed)) - 1) / 30.0 / plant.CONTROL_DT)


# ----------------------------------------------------------------------------- loading


def prepare_workspace(script: Path, work: Path) -> Path:
    """Copy the submission's workspace (the clip under test is passed separately, under a neutral name)."""
    work.mkdir(parents=True)
    for item in script.parent.iterdir():
        if item.name in ("__pycache__", ".git"):
            continue
        if item.is_dir():
            shutil.copytree(item, work / item.name, ignore=shutil.ignore_patterns("__pycache__"))
        elif item.is_file() and item.stat().st_size < 200_000_000:
            shutil.copy2(item, work / item.name)
    return work / script.name


def load(path: Path):
    # The submission imports under its own stem, so Warp kernels it compiled during the trial are reused.
    sys.path.insert(0, str(path.parent))
    return _load_module(path.stem, path)


# ----------------------------------------------------------------------------- tamper detection


def _is_code(value) -> bool:
    return callable(value) or isinstance(value, (staticmethod, classmethod, property))


def snapshot() -> dict:
    """Functions and classes (and their methods) of the watched modules, this verifier, and builtins, by
    identity; and copies of the plain-data constants of this verifier and its plant module."""
    out = {}
    verifier = sys.modules[__name__]
    for module in (verifier, plant):
        for key, value in vars(module).items():
            if key.startswith("__"):
                continue
            if isinstance(value, (bool, int, float, str, tuple, list, dict, set)):
                out[("const", module.__name__, key)] = copy.deepcopy(value)
            elif isinstance(value, np.ndarray):
                out[("const", module.__name__, key)] = value.copy()
    for name, module in list(sys.modules.items()):
        if module is None or (name.split(".")[0] not in WATCHED_ROOTS and module is not verifier):
            continue
        try:
            members = list(vars(module).items())
        except TypeError:
            continue
        for key, value in members:
            if not _is_code(value):
                continue
            out[(name, key)] = value
            if isinstance(value, type) and getattr(value, "__module__", None) == name:
                for attr, member in list(vars(value).items()):
                    if _is_code(member):
                        out[(name, key, attr)] = member
    for key, value in vars(builtins).items():
        if _is_code(value):
            out[("builtins", key)] = value
    return out


def patched(before: dict) -> list[str]:
    """Entries of a :func:`snapshot` that were replaced or removed since."""
    changed = []
    for key, value in before.items():
        if key[0] == "const":
            current = vars(sys.modules[key[1]]).get(key[2])
            same = (
                isinstance(current, np.ndarray) and current.shape == value.shape and np.array_equal(current, value)
                if isinstance(value, np.ndarray)
                else current == value
            )
            if not same:
                changed.append(".".join(key[1:]))
            continue
        holder = builtins if key[0] == "builtins" else sys.modules.get(key[0])
        current = vars(holder).get(key[1]) if holder is not None else None
        if len(key) == 3:
            current = vars(current).get(key[2]) if isinstance(current, type) else None
        if current is not value:
            changed.append(".".join(key))
    return changed


# Introspection a controller never needs: frames, the garbage collector, raw memory, code objects. With it, a
# submission could reach the plant or read the run's nonce from the verifier's frames and forge a result.
GUARDED_EVENTS = {
    "sys._getframe",
    "sys._getframemodulename",
    "sys._current_frames",
    "sys._current_exceptions",
    "sys.settrace",
    "sys.setprofile",
    "sys.addaudithook",
    "sys.monitoring.register_callback",
    "gc.get_objects",
    "gc.get_referrers",
    "gc.get_referents",
    "object.__getattr__",
    "object.__setattr__",
    "object.__delattr__",
    "code.__new__",
    "function.__new__",
    "compile",
    "exec",
    "marshal.loads",
    "marshal.load",
}
SPAWN_EVENTS = {
    "subprocess.Popen",
    "os.system",
    "os.posix_spawn",
    "os.spawn",
    "os.exec",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
}
# Signals would hand a submitted handler the verifier's frame.
SPAWN_EVENTS |= {"os.kill", "os.killpg", "signal.pthread_kill", "signal.raise_signal"}
# A submitted thread could run between the verifier's last check and its result: controllers run in one thread.
SPAWN_EVENTS |= {"_thread.start_new_thread", "_thread.start_joinable_thread"}
PATH_EVENTS = {
    "open",
    "os.listdir",
    "os.scandir",
    "glob.glob",
    "os.chdir",
    "shutil.copytree",
    "shutil.copyfile",
    "os.link",
    "os.symlink",
    "os.rename",
    "os.remove",
    "os.rmdir",
    "os.truncate",
}
# Frame-returning standard modules: their introspection counts as the caller's.
FRAME_HELPERS = ("inspect.py", "traceback.py", "pdb.py", "bdb.py", "trace.py", "profile.py", "cProfile.py")
FRAME_HELPERS += ("logging/__init__.py",)
LDCONFIG = {"/sbin/ldconfig", "/usr/sbin/ldconfig", "/sbin/ldconfig.real", "/usr/sbin/ldconfig.real"}


def install_guard(extra_hidden: tuple[Path, ...] = ()):
    """Audit hook that blocks introspection, process spawning, threads, and access to hidden data by untrusted code.

    Code is trusted if it comes from the standard library, the virtual environment, Newton, Warp, or this
    verifier's directory; everything else (the submission's workspace, files it writes elsewhere, code compiled
    from strings) is not. Blocked operations raise in the submission and are recorded; any record fails the
    integrity check. Audit hooks cannot be removed, and the hook's state lives only in this closure. Hidden data
    are the private directory, the study's harness tree, and ``extra_hidden`` (the child's result file). Returns a
    function that lists the records.
    """
    paths = sysconfig.get_paths()
    roots = {os.path.realpath(paths[key]) for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    roots |= {os.path.realpath(Path(module.__file__).parent) for module in (newton, wp, np)}
    roots.add(os.path.realpath(HERE))
    roots = tuple(sorted(roots))
    # Hidden data (the unseen clips also live in the study's harness tree, which agents never see).
    hidden_roots = {str(PRIVATE.parent), os.path.realpath(PRIVATE.parent)}
    for tree in {ROOT, Path(newton.__file__).resolve().parents[1]}:  # this harness and the imported Newton's
        hidden_roots |= {str(tree / "tools" / "mcp_evaluation"), os.path.realpath(tree / "tools" / "mcp_evaluation")}
    hidden_roots |= {str(path) for path in extra_hidden} | {os.path.realpath(path) for path in extra_hidden}
    guarded = tuple(hidden_roots)
    trusted_files, active, records = {}, set(), []

    def trusted(filename: str) -> bool:
        known = trusted_files.get(filename)
        if known is None:
            if filename.startswith("<frozen "):
                known = True
            elif not filename.startswith("/"):
                known = False
            else:
                real = os.path.realpath(filename)
                known = any(real.startswith(root + os.sep) for root in roots)
            trusted_files[filename] = known
        return known

    def hidden(target) -> bool:
        if not isinstance(target, (str, bytes, os.PathLike)):
            return False
        path = os.path.abspath(os.fsdecode(target))
        if path.startswith("/proc/") and path.rsplit("/", 1)[-1] in ("mem", "maps", "pagemap"):
            return True
        return any(path == root or path.startswith(root + os.sep) for root in guarded)

    def hook(event: str, args: tuple) -> None:
        if event not in GUARDED_EVENTS and event not in SPAWN_EVENTS and event not in PATH_EVENTS:
            if not event.startswith("ctypes."):
                return
        thread = _thread.get_ident()
        if thread in active:
            return
        active.add(thread)
        try:
            caller = sys._getframe(1)
            stack, frame = [], caller
            while frame is not None and len(stack) < 200:
                stack.append(frame.f_code.co_filename)
                frame = frame.f_back
            untrusted = any(not trusted(name) for name in stack)
            if event in PATH_EVENTS:
                blocked = untrusted and bool(args) and hidden(args[0])
            elif event in SPAWN_EVENTS:
                command = args[1] if event == "subprocess.Popen" and len(args) > 1 else None
                ldconfig = isinstance(command, (list, tuple)) and list(command[1:]) == ["-p"]
                blocked = untrusted and not (ldconfig and os.path.realpath(os.fsdecode(args[0] or "")) in LDCONFIG)
            else:
                direct = not trusted(stack[0])
                # Only when submitted code calls the helper itself (Warp's decorators use inspect too).
                helper = stack[0].endswith(FRAME_HELPERS) and len(stack) > 1 and not trusted(stack[1])
                blocked = direct or helper
            if blocked:
                if len(records) < 20:
                    records.append(f"{event} from {stack[0]}: {str(args)[:160]}")
                raise RuntimeError(f"{event} is not allowed in g1_mpc submissions")
        finally:
            active.discard(thread)

    sys.addaudithook(hook)
    return lambda: list(records)


# Names a controller never needs; the import-time scan rejects them before any submitted code runs.
SUSPICIOUS_MODULES = {"inspect", "gc", "ctypes", "atexit", "faulthandler", "signal", "sysconfig", "marshal", "_thread"}
SUSPICIOUS_NAMES = {
    "_getframe",
    "_current_frames",
    "currentframe",
    "f_back",
    "f_locals",
    "f_globals",
    "tb_frame",
    "gi_frame",
    "settrace",
    "setprofile",
    "addaudithook",
    "_exit",
    "get_referrers",
    "get_referents",
    "get_objects",
    "__import__",
    "nonce",
    "g1_mpc_verifier_plant",
}
# Builtins flagged only as bare names: attributes such as re.compile or a model's eval() are harmless, and the
# audit hook blocks compile and exec by submitted code at run time.
SUSPICIOUS_BUILTINS = {"exec", "eval", "compile"}


def source_findings(work: Path) -> list[str]:
    """Introspection in the submitted Python sources."""
    import ast  # noqa: PLC0415

    findings = []
    for path in sorted(work.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError as error:
            findings.append(f"{path.name}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                names += [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                bad = [name for name in names if name.split(".")[0] in SUSPICIOUS_MODULES]
            elif isinstance(node, ast.Attribute):
                bad = [node.attr] if node.attr in SUSPICIOUS_NAMES else []
            elif isinstance(node, ast.Name):
                bad = [node.id] if node.id in SUSPICIOUS_NAMES | SUSPICIOUS_BUILTINS else []
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                bad = [name for name in SUSPICIOUS_NAMES if len(name) > 4 and name in node.value]
            else:
                bad = []
            findings += [f"{path.relative_to(work)}:{getattr(node, 'lineno', '?')} {name}" for name in bad]
    return findings[:20]


def _handlers() -> dict:
    out = {}
    for number in signal.valid_signals():
        try:
            out[int(number)] = signal.getsignal(number)
        except (OSError, ValueError):
            continue
    return out


def _hooks() -> list:
    hooks = []
    if sys.gettrace() is not None or threading.gettrace() is not None:
        hooks.append("trace")
    if sys.getprofile() is not None or threading.getprofile() is not None:
        hooks.append("profile")
    # Import hooks would run submitted code inside the verifier's later imports.
    hooks.append(tuple(id(hook) for hook in (*sys.meta_path, *sys.path_hooks)))
    return hooks


PLANT_MODEL_ARRAYS = (
    "body_mass",
    "body_inertia",
    "body_com",
    "body_q",
    "joint_q",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_X_p",
    "joint_X_c",
    "joint_axis",
    "joint_armature",
    "joint_effort_limit",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_damping",
    "joint_friction",
    "shape_body",
    "shape_transform",
    "shape_scale",
    "shape_type",
    "shape_flags",
    "shape_material_mu",
    "gravity",
)
PLANT_MJW_ARRAYS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_pos",
    "body_quat",
    "body_gravcomp",
    "dof_armature",
    "dof_damping",
    "dof_dampingpoly",
    "dof_frictionloss",
    "dof_solref",
    "dof_solimp",
    "jnt_range",
    "jnt_stiffness",
    "jnt_stiffnesspoly",
    "jnt_actgravcomp",
    "jnt_solref",
    "jnt_solimp",
    "qpos_spring",
    "actuator_forcerange",
    "actuator_forcelimited",
    "actuator_gear",
    "actuator_trnid",
    "actuator_ctrllimited",
    "actuator_ctrlrange",
    "actuator_gaintype",
    "actuator_biastype",
    "actuator_dyntype",
    "actuator_dynprm",
    "actuator_actlimited",
    "actuator_delay",
    "geom_friction",
    "geom_size",
    "geom_pos",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_priority",
    "geom_solmix",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_surfacevel",
    "eq_active0",
)
# Written by Robot.apply from each command (outside the controller's calls; part of the state digest).
COMMAND_MJW_ARRAYS = ("actuator_gainprm", "actuator_biasprm")


def plant_fingerprint(robot) -> dict:
    """Digests of the plant's physical parameters (Newton model and the MuJoCo model it simulates)."""
    out = {}
    for source, names in (("model", PLANT_MODEL_ARRAYS), ("mjw", PLANT_MJW_ARRAYS)):
        holder = robot.model if source == "model" else robot.solver.mjw_model
        for name in names:
            value = getattr(holder, name, None)
            if value is None:
                continue
            data = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            out[f"{source}.{name}"] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    opt = robot.solver.mjw_model.opt
    for name, value in sorted(vars(opt).items()):
        if hasattr(value, "numpy"):
            data = value.numpy()
        elif isinstance(value, (bool, int, float, np.ndarray, np.generic)):
            data = np.asarray(value)
        else:
            continue
        out[f"opt.{name}"] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    return out


def plant_state_digest(robot) -> bytes:
    """Digest of the plant's simulation state and inputs, servo gains included (controllers never write them)."""
    digest = hashlib.sha256()
    state, data = robot.state_0, robot.solver.mjw_data
    for array in (state.joint_q, state.joint_qd, state.body_q, state.body_qd, data.qpos, data.qvel):
        digest.update(array.numpy().tobytes())
    for array in (robot.control.joint_target_q, robot.control.joint_f):
        if array is not None:
            digest.update(array.numpy().tobytes())
    for name in COMMAND_MJW_ARRAYS:
        digest.update(getattr(robot.solver.mjw_model, name).numpy().tobytes())
    return digest.digest()


def solver_fingerprint(robot) -> dict:
    """Digests of every array of the plant's MuJoCo Warp model except the servo gains Robot.apply writes."""
    out = {}
    for name, value in sorted(vars(robot.solver.mjw_model).items()):
        if name not in COMMAND_MJW_ARRAYS and hasattr(value, "numpy") and hasattr(value, "shape"):
            out[name] = hashlib.sha256(np.ascontiguousarray(value.numpy()).tobytes()).hexdigest()
    return out


def _cpu_workload() -> None:
    rng = np.random.default_rng(0)
    matrix, vector = rng.standard_normal((12, 12)), rng.standard_normal(12)
    for _ in range(4000):
        vector = np.tanh(matrix @ vector)


class LoadProbe:
    """Fixed workloads the verifier times between controller calls (no submitted code runs meanwhile): a CUDA
    graph of :data:`LOAD_STEPS` physics steps of :data:`LOAD_WORLDS` copies of the plant from its MJCF pose, and
    a NumPy loop. Their slowdown against :data:`LOAD_BASELINE_S` measures how busy the shared GPU and CPUs are."""

    def __init__(self):
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.add_mjcf(plant.asset_path(), collapse_fixed_joints=True)
        shapes = [i for i, body in enumerate(robot.shape_body) if body >= 0]
        for i, shape in enumerate(shapes):
            for other in shapes[i + 1 :]:
                robot.add_shape_collision_filter_pair(shape, other)
        robot.joint_armature[6:] = plant.ARMATURE.tolist()
        robot.joint_target_ke[6:] = [100.0] * plant.JOINT_COUNT
        robot.joint_target_kd[6:] = [2.0] * plant.JOINT_COUNT
        robot.joint_target_mode[6:] = [int(newton.JointTargetMode.POSITION)] * plant.JOINT_COUNT
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(robot, LOAD_WORLDS)
        self.model = model = builder.finalize()
        self.solver = plant.make_solver(model)
        self.state_0, self.state_1, self.control = model.state(), model.state(), model.control()
        self.control.joint_target_q.assign(model.joint_q)
        self.start = model.state()
        self.simulate()  # load the kernels before the capture
        self.graph = None
        if model.device.is_cuda:
            with wp.ScopedCapture(device=model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        self.records = []

    def simulate(self) -> None:
        for _ in range(LOAD_STEPS):
            self.solver.step(self.state_0, self.state_1, self.control, None, plant.SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def measure(self) -> float:
        """Slowdown factor now (at least 1); recorded in :attr:`records`."""
        device = self.model.device
        wp.synchronize_device(device)  # work left queued by the controller is not timed
        gpu = []
        for _ in range(LOAD_REPEATS):
            self.state_0.assign(self.start)
            wp.synchronize_device(device)
            tic = time.perf_counter()
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            wp.synchronize_device(device)
            gpu.append(time.perf_counter() - tic)
        cpu = []
        for _ in range(3):
            tic = time.perf_counter()
            _cpu_workload()
            cpu.append(time.perf_counter() - tic)
        gpu_s, cpu_s = float(np.median(gpu)), float(np.median(cpu))
        factor = max(1.0, gpu_s / LOAD_BASELINE_S["gpu"], cpu_s / LOAD_BASELINE_S["cpu"])
        self.records.append({"gpu_ms": 1000.0 * gpu_s, "cpu_ms": 1000.0 * cpu_s, "factor": factor})
        return factor

    def factor(self) -> float:
        """Mean slowdown over the measurements so far, within [1, LOAD_FACTOR_MAX]."""
        if not self.records:
            return 1.0
        return float(min(LOAD_FACTOR_MAX, max(1.0, np.mean([r["factor"] for r in self.records]))))


class Audit:
    """Named integrity checks with the reasons they failed."""

    def __init__(self):
        self.notes: dict[str, list[str]] = {}

    def check(self, name: str, ok: bool, message: str = "") -> bool:
        notes = self.notes.setdefault(name, [])
        if not ok:
            notes.append(message if len(notes) < 6 else "...")
            del notes[7:]
        return bool(ok)

    @property
    def checks(self) -> dict[str, bool]:
        return {name: not notes for name, notes in self.notes.items()}

    @property
    def failed(self) -> list[str]:
        return [name for name, notes in self.notes.items() if notes]


# ----------------------------------------------------------------------------- one clip


def clip_gates(metrics: dict, t: dict) -> list[str]:
    """Gates a clip's metrics fail."""
    failed = [] if metrics.get("upright") is True else ["upright"]
    for key, limit in t.items():
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
            failed.append(key)
    return failed


def verify_clip(script: Path, name: str, keep: list, hidden: tuple[Path, ...] = (), speed: float | None = None) -> dict:
    """One clip in this process (the submission is imported here).

    The submission's module and controller are appended to ``keep``, which the caller holds until the process
    exits: their finalizers never run after the last check. ``hidden`` paths are closed to submitted code.
    ``speed`` plays a time-scaled clip (:data:`SCALED`) at that speed.
    """
    started, timing = time.perf_counter(), {}
    thresholds = dict(THRESHOLDS)
    result = {"task": "g1_mpc", "clip": name, "speed": speed, "success": False, "thresholds": thresholds}
    result["metrics"] = {}
    audit = Audit()
    newton.solvers.SolverMuJoCo.import_mujoco()
    module, rollout_s, compute_s, compute_max, done = None, 0.0, 0.0, 0.0, 0
    with tempfile.TemporaryDirectory() as tmp:
        # The clip under test, under a neutral name: the verifier's reference and the controller's input.
        motion_path = Path(tmp) / "motion" / "reference.csv"
        motion_path.parent.mkdir()
        if name in SCALED:
            np.savetxt(motion_path, clip_rows(name, speed), delimiter=",", fmt="%.9g")
        else:
            shutil.copyfile(clip_path(name), motion_path)
        # The plant, the scored reference, the controller's model copy, and the load probe, built before any
        # submitted code runs.
        scratch = plant.build_model()
        reference = plant.MotionClip(str(motion_path), scratch)
        start = reference.sample(0.0)
        robot = plant.Robot(start)
        report = plant.TrackingReport(robot, reference)
        planning = plant.build_model(start)
        fingerprint, full_fingerprint = plant_fingerprint(robot), solver_fingerprint(robot)
        steps = round(reference.duration / plant.CONTROL_DT)
        load_probe = LoadProbe()
        setup_cap = SETUP_LIMIT_S * min(LOAD_FACTOR_MAX, load_probe.measure())
        timing["plant_s"] = time.perf_counter() - started
        threads, hooks = set(threading.enumerate()), _hooks()
        path = prepare_workspace(script, Path(tmp) / "work")
        findings = source_findings(path.parent)
        if findings:
            audit.check("source", False, f"introspection in the submission: {findings}")
            result.update(integrity=audit.checks, failed_checks=audit.failed, details={"notes": audit.notes})
            return {**result, "deterministic": True, "seconds": time.perf_counter() - started}
        audit.check("source", True)
        blocked = install_guard(hidden)
        handlers = _handlers()
        before = snapshot()

        def untampered(when: str) -> bool:
            changed = patched(before)
            ok = audit.check("untampered", not changed, f"replaced {when}: {changed[:6]}")
            ok &= audit.check("untampered", _hooks() == hooks, f"trace or profile hooks {when}")
            ok &= audit.check("untampered", _handlers() == handlers, f"signal handlers changed {when}")
            records = blocked()
            return audit.check("sandbox", not records, f"blocked {when}: {records[:4]}") and ok

        tic = time.perf_counter()
        try:
            module = load(path)
            clip_class = getattr(module, "MotionClip", plant.MotionClip)
            controller = module.Controller(planning, clip_class(str(motion_path), planning))
            keep += [module, controller]
            audit.check("build", callable(getattr(controller, "compute", None)), "Controller has no compute()")
        except Exception:
            audit.check("build", False, traceback.format_exc()[-3000:])
        timing["setup_s"] = setup = time.perf_counter() - tic
        timing["setup_cap_s"] = setup_cap
        audit.check("runtime", setup <= setup_cap, f"setup took {setup:.0f} s (limit {setup_cap:.0f} s)")
        untampered("during setup")
        if not audit.failed:
            # Submitted code runs only inside controller.compute from here on: no collector-run finalizers (the
            # process exits without collecting).
            gc.disable()
            for check in ("controller", "command", "plant_state", "plant_model"):
                audit.check(check, True)
            hard_cap = ROLLOUT_LIMIT_S * LOAD_FACTOR_MAX
            probe_s = 0.0
            tic = time.perf_counter()
            for k in range(steps):
                if k and k % LOAD_PERIOD == 0:
                    probe_tic = time.perf_counter()
                    load_probe.measure()
                    probe_s += time.perf_counter() - probe_tic
                elapsed = time.perf_counter() - tic - probe_s
                # Stop runs that cannot finish within the largest cap (projected after a quarter of the clip).
                projected = elapsed * steps / k if k >= steps // 4 and k else elapsed
                if elapsed > hard_cap or projected > 1.25 * hard_cap:
                    at = f"{k * plant.CONTROL_DT:.2f} s"
                    audit.check(
                        "runtime", False, f"rollout stopped at {at} after {elapsed:.0f} s (projected {projected:.0f} s)"
                    )
                    break
                joint_q = robot.state_0.joint_q.numpy().astype(np.float64)
                joint_qd = robot.state_0.joint_qd.numpy().astype(np.float64)
                digest = plant_state_digest(robot)
                call = time.perf_counter()
                try:
                    command = controller.compute(k * plant.CONTROL_DT, joint_q, joint_qd)
                except Exception:
                    audit.check(
                        "controller", False, f"t={k * plant.CONTROL_DT:.2f} s: {traceback.format_exc()[-2000:]}"
                    )
                    break
                elapsed = time.perf_counter() - call
                compute_s, compute_max = compute_s + elapsed, max(compute_max, elapsed)
                try:
                    # Submitted code behind the command (properties, __array__, __del__) runs here, before the check.
                    arrays = plant.command_arrays(command)
                    command = None
                except Exception as error:
                    audit.check("command", False, f"t={k * plant.CONTROL_DT:.2f} s: {error}")
                    break
                # GPU work the controller left on any stream finishes before the plant is checked and stepped.
                wp.synchronize_device(robot.model.device)
                # Submitted code runs only inside compute(): anything it changed is caught before the plant steps.
                when = f"at t={k * plant.CONTROL_DT:.2f} s"
                if not untampered(when):
                    break
                if not audit.check("plant_state", plant_state_digest(robot) == digest, f"plant state written {when}"):
                    break
                changed = [key for key, value in plant_fingerprint(robot).items() if fingerprint.get(key) != value]
                if not audit.check("plant_model", not changed, f"plant parameters changed {when}: {changed[:6]}"):
                    break
                robot.apply(plant.Command(**arrays))
                robot.advance()
                done = k + 1
                if not report.update(done * plant.CONTROL_DT, robot.state_0):
                    break
            wp.synchronize_device(robot.model.device)
            rollout_s = time.perf_counter() - tic - probe_s
            load_probe.measure()
            rollout_cap = ROLLOUT_LIMIT_S * load_probe.factor()
            timing.update(rollout_cap_s=rollout_cap, load_factor=load_probe.factor(), load_probes=load_probe.records)
            message = (
                f"rollout took {rollout_s:.0f} s (limit {rollout_cap:.0f} s at load factor {load_probe.factor():.2f})"
            )
            audit.check("runtime", rollout_s <= rollout_cap, message)
            untampered("during the rollout")
            changed = [k for k, v in plant_fingerprint(robot).items() if fingerprint.get(k) != v]
            changed += [k for k, v in solver_fingerprint(robot).items() if full_fingerprint.get(k) != v]
            audit.check("plant_model", not changed and len(fingerprint) > 0, f"plant parameters changed: {changed[:6]}")
    metrics = report.summary()
    # A run cut short (time cap, controller error) survives only the part it ran.
    metrics["completed"] = done / steps if steps else 0.0
    metrics["survival"] = min(metrics["survival"], metrics["completed"])
    gates = clip_gates(metrics, thresholds) if not audit.failed else ["integrity"]
    timing.update(rollout_s=rollout_s, compute_s=compute_s, compute_max_s=compute_max, steps=done, expected_steps=steps)
    timing.setdefault("load_factor", load_probe.factor())
    timing.setdefault("load_probes", load_probe.records)
    result.update(
        success=not audit.failed and not gates,
        integrity=audit.checks,
        failed_checks=audit.failed + gates,
        metrics=metrics,
        details={
            "notes": {k: v for k, v in audit.notes.items() if v},
            "timing": timing,
            "threads_started": len(set(threading.enumerate()) - threads),
        },
        deterministic=bool(audit.failed and "controller" not in audit.failed and "runtime" not in audit.failed),
        seconds=time.perf_counter() - started,
    )
    return result


# ----------------------------------------------------------------------------- parent


def _scrub(value):
    """JSON-safe copy (non-finite floats as strings, numpy scalars as Python numbers)."""
    if isinstance(value, dict):
        return {str(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else str(float(value))
    return value


def _number(value) -> float:
    """A reported metric as a float (non-finite values arrive as strings)."""
    if isinstance(value, bool) or value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _child(script: Path, name: str, timeout: float, speed: float | None = None) -> dict:
    """One clip in a fresh process; its result must carry the nonce sent to it before the import."""
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / f"{name}.json"
        command = [sys.executable, str(Path(__file__).resolve()), str(script), "--clip", name, "--nonce"]
        command += ["--output", str(output)]
        if speed is not None:
            command += ["--speed", f"{speed:g}"]
        try:
            process = subprocess.run(
                command, input=nonce + "\n", capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"{name} timed out after {timeout:.0f} s", "failed_checks": ["timeout"]}
        try:
            result = json.loads(output.read_text())
        except (OSError, ValueError):
            result = None
        if process.returncode != 0 or result is None or result.get("nonce") != nonce:
            error = (process.stderr or process.stdout)[-3000:] or "no result with the run's nonce"
            return {"success": False, "error": error, "failed_checks": ["crash"]}
        result.pop("nonce", None)
        return result


REQUIRED_CHECKS = {
    "source",
    "build",
    "runtime",
    "sandbox",
    "untampered",
    "controller",
    "command",
    "plant_state",
    "plant_model",
}


def implausible(result: dict, name: str, t: dict, speed: float | None = None) -> list[str]:
    """Why a passing child result does not look like a real run: the parent, which never imports the
    submission, recomputes every gate from the reported metrics and requires every integrity check."""
    problems = []
    metrics, integrity = result.get("metrics") or {}, result.get("integrity") or {}
    timing = (result.get("details") or {}).get("timing") or {}
    missing = REQUIRED_CHECKS - set(integrity)
    if missing or not all(value is True for value in integrity.values()):
        problems.append(f"integrity checks missing or failed: {sorted(missing)[:6]}")
    if result.get("clip") != name or result.get("speed") != speed:
        problems.append(f"clip {result.get('clip')} at {result.get('speed')} instead of {name} at {speed}")
    expected = expected_samples(name, speed)
    if metrics.get("samples") != expected or timing.get("steps") != expected:
        problems.append(f"{metrics.get('samples')} samples instead of {expected}")
    if metrics.get("upright") is not True:
        problems.append("not upright")
    for key, limit in t.items():
        value = _number(metrics.get(key))
        if not math.isfinite(value) or value > limit:
            problems.append(f"{key} fails")
    factor = _number(timing.get("load_factor"))
    if not 1.0 <= factor <= LOAD_FACTOR_MAX:
        problems.append(f"load factor {factor} outside [1, {LOAD_FACTOR_MAX}]")
    rollout = _number(timing.get("rollout_s"))
    if not math.isfinite(rollout) or not 1.0 <= factor <= LOAD_FACTOR_MAX or rollout > ROLLOUT_LIMIT_S * factor + 5.0:
        problems.append("no plausible rollout timing")
    return problems


def clip_score(metrics: dict, t: dict) -> float:
    """Graded score of one clip in [0, 1]: survival times the mean of min(1, threshold / value) over the gates."""
    survival = _number(metrics.get("survival"))
    if not math.isfinite(survival):
        return 0.0
    parts = []
    for key, limit in t.items():
        value = _number(metrics.get(key))
        parts.append(0.0 if not math.isfinite(value) else min(1.0, limit / max(value, 1e-9)))
    return float(max(0.0, min(1.0, survival)) * np.mean(parts))


def summarize(results: dict, t: dict) -> dict:
    """Overall verdict and graded metrics from the per-clip results."""
    passed = {name: bool(r.get("success")) for name, r in results.items()}
    metrics = {}
    for name, r in results.items():
        m = r.get("metrics") or {}
        metrics[f"{name}.passed"] = passed[name]
        metrics[f"{name}.upright"] = m.get("upright") is True
        for key in ("survival", *t, *GRADED, "lift_recall"):
            metrics[f"{name}.{key}"] = m.get(key)
        if r.get("speed") is not None:
            metrics[f"{name}.speed"] = r.get("speed")
        # The time cap and controller errors only cut a run short (its survival counts the part it ran); any
        # other integrity failure zeroes the clip's score.
        integrity = r.get("integrity") or {}
        honest = bool(integrity) and all(
            ok for check, ok in integrity.items() if check not in ("runtime", "controller")
        )
        metrics[f"{name}.score"] = clip_score(m, t) if honest else 0.0
    metrics["clips_passed"] = sum(passed.values())
    metrics["given_clips_passed"] = sum(passed[n] for n in results if CLIPS[n])
    metrics["unseen_clips_passed"] = sum(passed[n] for n in results if not CLIPS[n])
    metrics["score"] = float(np.mean([metrics[f"{n}.score"] for n in results])) if results else 0.0
    normalized = {}
    for key, limit in {**t, **GRADED}.items():
        values = []
        for r in results.values():
            m = r.get("metrics") or {}
            value = _number(m.get(key)) if m.get("upright") is True else math.inf
            values.append(value / limit if math.isfinite(value) else math.inf)
        normalized[key] = max(values) if values else math.inf
    return {"metrics": metrics, "normalized_worst": normalized}


def verify_all(script: Path, clips: list[str], budget: float = TOTAL_BUDGET_S) -> dict:
    """Every clip in its own fresh process; success requires every clip to pass."""
    started, results = time.perf_counter(), {}
    thresholds = dict(THRESHOLDS)
    speeds = {name: draw_speed(name) for name in clips}
    for name in clips:
        remaining = budget - (time.perf_counter() - started)
        if remaining < 60.0:
            results[name] = {"success": False, "error": "verification budget exhausted", "failed_checks": ["budget"]}
            continue
        result = _child(script, name, min(CHILD_TIMEOUT_S, remaining), speeds[name])
        problems = implausible(result, name, thresholds, speeds[name]) if result.get("success") else []
        if problems:
            result = {
                "success": False,
                "error": f"the run's result does not match a real verification: {problems[:6]}",
                "failed_checks": ["result"],
            }
        results[name] = result
        if result.get("deterministic") and any(k in result.get("failed_checks", []) for k in ("source", "build")):
            # Source and build failures do not depend on the clip.
            for rest in clips[clips.index(name) + 1 :]:
                results[rest] = {**result, "clip": rest}
            break
    summary = summarize(results, thresholds)
    integrity = {
        f"{name}.{check}": ok for name, r in results.items() for check, ok in (r.get("integrity") or {}).items()
    }
    return {
        "task": "g1_mpc",
        "success": bool(results) and all(bool(r.get("success")) for r in results.values()),
        "integrity": integrity,
        "failed_checks": [f"{name}.{check}" for name, r in results.items() for check in r.get("failed_checks", [])],
        "metrics": summary["metrics"],
        "normalized_worst": summary["normalized_worst"],
        "thresholds": thresholds,
        "limits": {"setup_s": SETUP_LIMIT_S, "rollout_s": ROLLOUT_LIMIT_S, "load_factor_max": LOAD_FACTOR_MAX},
        "speeds": {name: speed for name, speed in speeds.items() if speed is not None},
        "details": {
            name: {
                k: r.get(k) for k in ("success", "integrity", "failed_checks", "metrics", "details", "error", "seconds")
            }
            for name, r in results.items()
        },
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clip", choices=sorted(CLIPS), help="verify one clip in this process")
    parser.add_argument("--clips", nargs="+", choices=sorted(CLIPS), default=list(CLIPS), help="clips (parent)")
    parser.add_argument("--speed", type=float, help="playback speed of a time-scaled clip (with --clip)")
    parser.add_argument("--nonce", action="store_true", help="read a nonce from stdin and echo it in the result")
    args = parser.parse_args()
    # Read before the submission is imported; kept in this frame, not in a module global.
    nonce = sys.stdin.readline().strip() if args.nonce else None
    wp.config.log_level = wp.LOG_WARNING
    script = args.script.resolve()
    keep = []  # the submission's objects, alive until the process exits
    if args.clip:
        hidden = (args.output.resolve(),) if args.output else ()
        result = verify_clip(script, args.clip, keep, hidden, args.speed)
    else:
        result = verify_all(script, args.clips)
    if nonce is not None:
        result["nonce"] = nonce
    if args.output:
        args.output.write_text(json.dumps(_scrub(result), indent=2) + "\n")
    summary = {key: result.get(key) for key in ("success", "failed_checks")}
    summary["metrics"] = {
        k: v for k, v in (result.get("metrics") or {}).items() if not isinstance(v, (list, dict)) and v is not None
    }
    print(json.dumps(_scrub(summary)), flush=True)
    if args.clip:
        # Skip exit handlers the submission may have registered (they could rewrite the result).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
