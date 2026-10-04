# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a from-scratch physical scene of a real ABC-130k fruit pick-and-place episode (``abc_scratch``).

The agent starts from the bare station (``scene_replay.py``), photos, and the episode's logs, and recreates the
objects, the tray, and the physics. This verifier loads the submitted ``scene_replay.py`` in a clean copy of its
workspace (its own station MJCF, episode, and camera file; no photos or arm logs), builds one batched model with
the submission's ``build_model(num_worlds)``, and identifies in every world the free bodies that stand for the
three real objects by their start positions (best assignment, at most 6 cm from where the top camera saw each
object at the start). It then applies its own perturbations and runs its own replay loop
(:class:`replay_core.Replay`, the starter's command rule) with ``make_solver(model)`` and
``make_pipeline(model)`` over the whole episode:

- the main episode x 8: 2 nominal copies and 6 with every object's start moved by up to 3 mm and turned by up
  to 5 degrees about the vertical. Per real object: held through the real carry window, and at rest in the
  real tray within 5 cm of where the real object rested after its release (its final position, or for the
  pear also where it rested before the later objects pushed it), each in enough copies;
- two negative controls, 2 copies each: gripper commands forced open, and friction 0.02 on every arm and object
  shape. No object may rise more than 1 cm; a diverged control copy fails the control.

The copies' order is drawn per run, and the controls' friction is set only after ``make_solver`` and
``make_pipeline`` returned, so submitted code cannot single out the controls. SolverMuJoCo keeps its own copy of
the model: :func:`resync_solver` re-applies the audited model to it (``notify_model_changed``) and checks what
that does not rewrite (the solver's arrays against the model it compiled, per-world copies, bodies on fixed
joints).

Before the rollout, :func:`audit_submission` checks that the submission is a physical model of the real
station and scene: identical worlds (every per-entity model array); solver and pipeline types, with no submitted
objects inside them; station kinematics, finger geometry, arm bases (the MJCF's), table, and gravity; three
free, dynamic object bodies within the tracked size ranges +-30 %, 20 to 300 g, plausible inertia, resting on
the table within 5 cm of the video's start, without drives, damping, gravity compensation, or applied forces;
other added geometry (the tray, static or one body) inside a generous envelope of the real tray; materials,
actuators (plain servos within the gain and force bounds), equality constraints, contact stiffness, what the
MuJoCo solver actually simulates; and that nothing in Newton, Warp, MuJoCo, NumPy, or this verifier was
patched. The workspace may not link to files outside it. Hidden data (``truth.json``: video-tracker starts,
sizes, events, and final positions; the tray) and thresholds are read from
``~/.newton-visual-private/abc_scratch``.

By default the verification runs twice in fresh processes (a third time if they disagree), and the majority
decides.
"""

from __future__ import annotations

import _thread
import argparse
import builtins
import copy
import gc
import hashlib
import importlib.util
import itertools
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The verifier's replay loop and metrics, under a name of its own (the workspace has no copy).
core = _load_module("abc_scratch_verifier_core", HERE / "replay_core.py")

PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "abc_scratch"
OBJECTS = ("pear", "orange", "dark_fruit")  # the real objects of truth.json (the agent names its bodies freely)

# Gates (main episode: per real object, copies of 8), from the task spec; the private reference passes them in
# every copy. PRIVATE/thresholds.json, if present, overrides them (same keys).
DEFAULT_THRESHOLDS = {
    "main_held_min": 6,  # copies holding the object through its real carry window
    "main_placed_min": 6,  # copies with the object at rest in the tray, within rest_xy_err_m_max of a real rest
    "held_fraction_min": 0.8,  # fraction of the carry window an object counts as held
    # [m] final centre to the nearest centre at which the real object rested after its release: its final
    # position, or one it rested at before a later object pushed it (the pear: three rest positions)
    "rest_xy_err_m_max": 0.05,
    "control_rise_m_max": 0.01,  # [m] highest object rise in either negative control
}

BOUNDS = {
    "dt": (0.00025, 0.002),
    "command_delay": (0.0, 0.2),
    "object_mass_kg": (0.02, 0.3),
    "object_size_scale": (0.7, 1.3),  # extents within these multiples of the low and high tracked sizes
    "object_shapes_max": 8,
    "object_armature": 1e-6,
    "object_com_offset_m": 0.02,  # centre of mass from the centre of the collision geometry
    "match_m": 0.06,  # a free body stands for a real object if it starts this close to the video's start
    # Object start (centre of its collision geometry) from the video's start. With the MJCF's arm bases the
    # simulated grippers close 1.4 / 2.3 / 4.0 cm (pear, orange, dark fruit) from the video's starts, and a dark
    # fruit placed at the video's start is not grasped: the bound admits starts consistent with either.
    "start_xy_m": 0.05,
    "start_clearance_m": (-0.001, 0.003),  # lowest collision point above the table
    "base_m": 0.003,
    "frame_m": 0.0005,  # station joint frames and collision shapes
    "frame_rad": 0.01,
    "table_z_m": 0.002,
    "tray_shapes_max": 40,
    "tray_margin_m": 0.04,  # added geometry within the real tray footprint dilated by this
    "tray_top_m": 0.04,  # ... and at most this high above the table
    "tray_mass_kg": (0.1, 1.0),
    "mu": 1.5,
    "mu_torsional": 0.02,
    "mu_rolling": 0.005,
    "restitution": 0.8,
    "margin": 0.002,
    "gap": 0.1,
    "arm_effort_nm": (28.0, 28.0, 28.0, 10.0, 10.0, 10.0),
    "gravcomp": (0.0, 1.0),
    "finger_kp": (100.0, 3000.0),
    "finger_force_n": (5.0, 60.0),
    "dampratio": (0.5, 2.0),  # MuJoCo contact and finger-mirror solref; time constant at least 2 dt
    "identical_tol": 1e-5,  # worlds must be copies of each other
}
GRIPPER_RANGE = 0.0495  # finger slide command range [m] (MJCF ctrlrange)
MAIN_COPIES, MAIN_NOMINAL, CONTROL_COPIES = 8, 2, 2
JITTER_XY_M, JITTER_YAW_DEG = 0.003, 5.0
CONTROL_MU = 0.02
SEED = 7067
RUNTIME_LIMIT_S = 400.0  # predicted wall time of the batched rollout
RUNS, MAX_RUNS = 2, 3
CHILD_TIMEOUT_S, TOTAL_BUDGET_S = 840.0, 1700.0  # the harness stops the verifier after 1800 s
SIDES = core.SIDES
FINGER_BODIES = {
    f"{side}_{name}"
    for side in SIDES
    for name in ("link_left_finger", "lf_rot", "lf_down", "link_right_finger", "rf_rot", "rf_down")
}
DRIVEN_JOINTS = {f"{side}_joint{k}" for side in SIDES for k in range(1, 7)} | {
    f"{side}_{finger}" for side in SIDES for finger in ("left_finger", "right_finger")
}
FINGER_PAIRS = {(f"{side}_left_finger", f"{side}_right_finger") for side in SIDES}
BASE_JOINTS = {"left_arm_joint": "left", "right_arm_joint": "right"}
ARM_PREFIXES = tuple(f"{side}_" for side in SIDES)
SOLVERS = tuple(
    getattr(newton.solvers, name)
    for name in ("SolverMuJoCo", "SolverXPBD", "SolverFeatherstone", "SolverSemiImplicit", "SolverKamino", "SolverVBD")
    if hasattr(newton.solvers, name)
)
# Modules whose functions and classes the submission must not replace.
WATCHED_ROOTS = {"newton", "warp", "mujoco", "mujoco_warp", "numpy", "math", "json", "copy", core.__name__}
# Modules the verifier writes its result and talks to its parent with.
WATCHED_ROOTS |= {"os", "posixpath", "pathlib", "io", "sys", "secrets", "subprocess", "shutil", "tempfile", "signal"}
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
GEO = newton.GeoType


def load_thresholds() -> dict:
    thresholds = dict(DEFAULT_THRESHOLDS)
    path = PRIVATE / "thresholds.json"
    if path.exists():
        thresholds.update({k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")})
    return thresholds


def thresholds_source() -> dict:
    """Where the gates come from: the private thresholds file and its calibration date, or the defaults."""
    path = PRIVATE / "thresholds.json"
    if not path.exists():
        return {"file": None}
    calibration = json.loads(path.read_text()).get("_calibration", {})
    return {"file": path.name, "calibrated": calibration.get("date"), "runs": calibration.get("runs")}


THRESHOLDS = load_thresholds()


# ----------------------------------------------------------------------------- loading


def load_truth() -> tuple[dict, dict]:
    """Hidden truth of the main episode (``truth.json``) and the video tracker's tracks (``gt.npz``)."""
    truth = json.loads((PRIVATE / "truth.json").read_text())
    with np.load(PRIVATE / "gt.npz") as data:
        gt = {key: np.asarray(data[key]) for key in data.files}
    return truth, gt


WORKSPACE_SKIP = {"photos", "arm_logs", "station", "observations", "__pycache__", ".git", "episode.npz", "camera.json"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".mp4")


def outside_links(workspace: Path) -> list[str]:
    """Symbolic links among the files the clean copy would take that resolve outside the workspace (the copy
    follows links, so they could bring hidden files into it)."""
    root = os.path.realpath(workspace)
    found = []
    for item in workspace.iterdir():
        if item.name in WORKSPACE_SKIP or item.suffix in IMAGE_SUFFIXES:
            continue
        paths = [item]
        if item.is_dir() and not item.is_symlink():
            for folder, dirs, files in os.walk(item):
                paths += [Path(folder) / name for name in (*dirs, *files)]
        for path in paths:
            target = os.path.realpath(path)
            if path.is_symlink() and not (target == root or target.startswith(root + os.sep)):
                found.append(str(path.relative_to(workspace)))
    return found[:10]


def prepare_workspace(script: Path, work: Path) -> Path:
    """Copy the submission's workspace without photos and arm logs, with this verifier's station, episode, and
    camera file."""
    for item in script.parent.iterdir():
        if item.name in WORKSPACE_SKIP or item.suffix in IMAGE_SUFFIXES:
            continue
        if item.is_dir():
            shutil.copytree(item, work / item.name, ignore=shutil.ignore_patterns("__pycache__", "*.jpg", "*.png"))
        elif item.is_file() and item.stat().st_size < 100_000_000:
            shutil.copy2(item, work / item.name)
    shutil.copytree(PRIVATE / "station", work / "station")
    shutil.copy2(PRIVATE / "episode.npz", work / "episode.npz")
    shutil.copy2(PRIVATE / "camera.json", work / "camera.json")
    return work / script.name


def load(path: Path):
    sys.path.insert(0, str(path.parent))
    return _load_module("submitted_scene_replay", path)


def plan_worlds() -> list[dict]:
    """The batched worlds: the main episode's copies (with their start perturbations) and the controls."""
    episode = core.load_episode(PRIVATE / "episode.npz")
    rng = np.random.default_rng([SEED, 0])
    worlds = []
    for index in range(MAIN_COPIES):
        jitter = None
        if index >= MAIN_NOMINAL:
            jitter = {}
            for name in OBJECTS:
                radius, angle = JITTER_XY_M * math.sqrt(rng.uniform()), rng.uniform(0.0, 2.0 * math.pi)
                yaw = math.radians(rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG))
                jitter[name] = (radius * math.cos(angle), radius * math.sin(angle), yaw)
        worlds.append({"group": "main", "copy": index, "data": episode, "control": None, "jitter": jitter})
    opened = dict(episode)
    for side in SIDES:
        opened[f"{side}_grip_cmd"] = np.ones_like(episode[f"{side}_grip_cmd"])
    for control, data in (("grip_open", opened), ("low_mu", episode)):
        for index in range(CONTROL_COPIES):
            worlds.append({"group": "control", "copy": index, "data": data, "control": control, "jitter": None})
    return worlds


# ----------------------------------------------------------------------------- tamper detection


def _is_code(value) -> bool:
    return callable(value) or isinstance(value, (staticmethod, classmethod, property))


def snapshot() -> dict:
    """Functions and classes (and their methods) of the watched modules, this verifier, and builtins, by
    identity; and copies of the plain-data constants of this verifier and its replay_common."""
    out = {}
    verifier = sys.modules[__name__]
    for module in (verifier, core):
        for key, value in vars(module).items():
            if not key.startswith("__") and isinstance(value, (bool, int, float, str, tuple, list, dict, set)):
                out[("const", module.__name__, key)] = copy.deepcopy(value)
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
            if vars(sys.modules[key[1]]).get(key[2]) != value:
                changed.append(".".join(key[1:]))
            continue
        holder = builtins if key[0] == "builtins" else sys.modules.get(key[0])
        current = vars(holder).get(key[1]) if holder is not None else None
        if len(key) == 3:
            current = vars(current).get(key[2]) if isinstance(current, type) else None
        if current is not value:
            changed.append(".".join(key))
    return changed


def overridden(obj) -> list[str]:
    """Instance attributes that shadow methods of the object's class."""
    if obj is None:
        return []
    cls = type(obj)
    return [k for k in vars(obj) if callable(getattr(cls, k, None)) and not isinstance(cls.__dict__.get(k), property)]


# Introspection a replay script never needs: frames, the garbage collector, raw memory, code objects. With it, a
# submission could read the run's nonce from the verifier's frames and forge a result.
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
}
# Frame-returning standard modules: their introspection counts as the caller's.
FRAME_HELPERS = ("inspect.py", "traceback.py", "pdb.py", "bdb.py", "trace.py", "profile.py", "cProfile.py")
FRAME_HELPERS += ("logging/__init__.py",)
LDCONFIG = {"/sbin/ldconfig", "/usr/sbin/ldconfig", "/sbin/ldconfig.real", "/usr/sbin/ldconfig.real"}


def install_guard(workspace: Path):
    """Audit hook that blocks introspection, process spawning, and reads of hidden data by untrusted code.

    Code is trusted if it comes from the standard library, the virtual environment, Newton, Warp, or this
    verifier's directory; everything else (the submission's workspace, files it writes elsewhere, code compiled
    from strings) is not. Blocked operations raise in the submission and are recorded; any record fails the
    integrity check. Audit hooks cannot be removed, and the hook's state lives only in this closure. Returns a
    function that lists the records.
    """
    paths = sysconfig.get_paths()
    roots = {os.path.realpath(paths[key]) for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    roots |= {os.path.realpath(Path(module.__file__).parent) for module in (newton, wp, np)}
    roots.add(os.path.realpath(HERE))
    roots = tuple(sorted(roots))
    guarded = tuple({str(PRIVATE.parent), os.path.realpath(PRIVATE.parent)})
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
        # Resolved, so links into the hidden directory count as the directory.
        path = os.path.realpath(os.fsdecode(target))
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
                raise RuntimeError(f"{event} is not allowed in abc_scratch submissions")
        finally:
            active.discard(thread)

    sys.addaudithook(hook)
    return lambda: list(records)


# Names a replay script never needs; the import-time scan rejects them before any submitted code runs.
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
    core.__name__,
}
# Builtins flagged only as bare names: attributes such as re.compile or a model's eval() are harmless, and the
# audit hook blocks compile and exec by submitted code at run time.
SUSPICIOUS_BUILTINS = {"exec", "eval", "compile"}


def _string_names(value: str) -> set[str]:
    """Names a string constant could look up with getattr or importlib: the string itself if it is a (dotted)
    identifier, and its components. Prose (docstrings, messages) names nothing."""
    text = value.strip()
    if not text or len(text) > 200 or not all(part.isidentifier() for part in text.split(".")):
        return set()
    return {text, *text.split(".")}


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
                bad = sorted(_string_names(node.value) & (SUSPICIOUS_NAMES | SUSPICIOUS_BUILTINS))
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


def _hooks() -> list[str]:
    hooks = []
    if sys.gettrace() is not None or threading.gettrace() is not None:
        hooks.append("trace")
    if sys.getprofile() is not None or threading.getprofile() is not None:
        hooks.append("profile")
    return hooks


# ----------------------------------------------------------------------------- geometry


def _leaf(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _fibonacci(count: int = 200) -> np.ndarray:
    k = np.arange(count) + 0.5
    z = 1.0 - 2.0 * k / count
    r = np.sqrt(1.0 - z * z)
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def _circle(radius: float, z: float, count: int = 32) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(count, z)], axis=-1)


def shape_points(model: newton.Model, shape: int) -> np.ndarray | None:
    """Surface points of a shape in its own frame [m]; ``None`` for unbounded shapes (planes, height fields)."""
    kind = int(model.shape_type.numpy()[shape])
    scale = model.shape_scale.numpy()[shape].astype(np.float64)
    if kind == GEO.SPHERE:
        return _fibonacci() * scale[0]
    if kind == GEO.ELLIPSOID:
        return _fibonacci() * scale
    if kind == GEO.CAPSULE:
        hemisphere, offset = _fibonacci() * scale[0], np.array([0.0, 0.0, scale[1]])
        return np.vstack([hemisphere + offset, hemisphere - offset])
    if kind in (GEO.CYLINDER, GEO.CONE):
        r, h = scale[0], scale[1]
        return np.vstack([_circle(r, h), _circle(r, -h), [[0.0, 0.0, h], [0.0, 0.0, -h]]])
    if kind == GEO.BOX:
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64) * scale
    if kind in (GEO.MESH, GEO.CONVEX_MESH):
        mesh = model.shape_source[shape]
        if mesh is None or getattr(mesh, "vertices", None) is None:
            return None
        return np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3) * scale
    return None


def _apply(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ core.quat_to_matrix(pose[3:7]).T + pose[:3]


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose a * b (7-vectors p, q xyzw)."""
    q = core.quat_multiply(np.asarray(a[3:7]), np.asarray(b[3:7]))
    return np.concatenate([a[:3] + core.quat_to_matrix(a[3:7]) @ b[:3], q])


def _rotation_close(qa: np.ndarray, qb: np.ndarray) -> bool:
    return float(np.abs(core.quat_to_matrix(qa) - core.quat_to_matrix(qb)).max()) <= BOUNDS["frame_rad"]


def _pose_close(a: np.ndarray, b: np.ndarray) -> bool:
    return float(np.abs(a[:3] - b[:3]).max()) <= BOUNDS["frame_m"] and _rotation_close(a[3:7], b[3:7])


def _principal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounding box of points along their principal axes: centre, extents (descending), axes (columns)."""
    mean = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - mean).T))
    local = (points - mean) @ vectors
    low, high = local.min(axis=0), local.max(axis=0)
    order = np.argsort(high - low)[::-1]
    return mean + vectors @ (0.5 * (low + high)), (high - low)[order], vectors[:, order]


# ----------------------------------------------------------------------------- integrity


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


class Index:
    """Lookups of a model's bodies, joints, and shapes by world and label leaf."""

    def __init__(self, model: newton.Model):
        self.model = model
        self.body_world = model.body_world.numpy()
        self.joint_world = model.joint_world.numpy()
        self.shape_world = model.shape_world.numpy()
        self.body_leaf = [_leaf(label) for label in model.body_label]
        self.joint_leaf = [_leaf(label) for label in model.joint_label]
        self.shape_leaf = [_leaf(label) for label in model.shape_label]
        self.shape_body = model.shape_body.numpy()
        self.shape_flags = model.shape_flags.numpy()
        self.joint_parent = model.joint_parent.numpy()
        self.joint_child = model.joint_child.numpy()
        self.joint_type = model.joint_type.numpy()
        self.qd_start = model.joint_qd_start.numpy()

    def bodies(self, world: int) -> dict[str, list[int]]:
        out = {}
        for i in np.flatnonzero(self.body_world == world):
            out.setdefault(self.body_leaf[i], []).append(int(i))
        return out

    def joints_into(self, body: int) -> list[int]:
        return [int(j) for j in np.flatnonzero(self.joint_child == body)]

    def dofs(self, joint: int) -> list[int]:
        return list(range(int(self.qd_start[joint]), int(self.qd_start[joint + 1])))

    def dof_joint(self, dof: int) -> int:
        return int(np.searchsorted(self.qd_start, dof, side="right") - 1)

    def colliders(self, world: int) -> list[int]:
        return [int(s) for s in np.flatnonzero((self.shape_world == world) & ((self.shape_flags & COLLIDE) != 0))]


class Station:
    """The reference station (this verifier's MJCF copy, one world): joints and collision shapes by label leaf."""

    def __init__(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(str(PRIVATE / "station" / "yam_bimanual_empty.xml"))
        top = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(top)
        top.add_world(builder)
        self.model = model = top.finalize()
        self.index = ix = Index(model)
        self.bodies = set(ix.body_leaf)
        x_p, x_c, axis = model.joint_X_p.numpy(), model.joint_X_c.numpy(), model.joint_axis.numpy()
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        self.joints = {}  # by child body leaf
        for j in range(model.joint_count):
            dofs = ix.dofs(j)
            self.joints[ix.body_leaf[ix.joint_child[j]]] = {
                "leaf": ix.joint_leaf[j],
                "type": int(ix.joint_type[j]),
                "parent": ix.body_leaf[ix.joint_parent[j]] if ix.joint_parent[j] >= 0 else None,
                "X_p": x_p[j].astype(np.float64),
                "X_c": x_c[j].astype(np.float64),
                "axis": axis[dofs].astype(np.float64),
                "limits": np.stack([lower[dofs], upper[dofs]], axis=-1).astype(np.float64),
            }
        self.shapes = {}  # by (body leaf or None for static, shape leaf)
        material = {
            key: getattr(model, f"shape_material_{key}").numpy() for key in ("mu", "mu_torsional", "mu_rolling")
        }
        for s in ix.colliders(0):
            body = ix.shape_body[s]
            self.shapes[(ix.body_leaf[body] if body >= 0 else None, ix.shape_leaf[s])] = {
                "type": int(model.shape_type.numpy()[s]),
                "scale": model.shape_scale.numpy()[s].astype(np.float64),
                "xform": model.shape_transform.numpy()[s].astype(np.float64),
                **{key: float(values[s]) for key, values in material.items()},
            }

        # Station bodies welded to the world (table, walls, arm bases): added static geometry may sit on them.
        self.welded = {
            leaf
            for leaf, joint in self.joints.items()
            if joint["parent"] is None and joint["type"] == int(newton.JointType.FIXED)
        }


class Submission:
    """The arrays of the submitted model that the audit reads, fetched once."""

    def __init__(self, model: newton.Model):
        self.model = model
        self.ix = Index(model)
        self.x_p, self.x_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
        self.axis = model.joint_axis.numpy()
        self.limits = np.stack([model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()], axis=-1)
        self.transform = model.shape_transform.numpy().astype(np.float64)
        self.scale = model.shape_scale.numpy()
        self.kind = model.shape_type.numpy()
        self.mass, self.inv_mass = model.body_mass.numpy(), model.body_inv_mass.numpy()
        self.inertia, self.inv_inertia = model.body_inertia.numpy(), model.body_inv_inertia.numpy()
        self.com = model.body_com.numpy()
        self.body_flags = model.body_flags.numpy() if model.body_flags is not None else None
        mujoco = getattr(model, "mujoco", None)
        self.gravcomp = getattr(mujoco, "gravcomp", None)
        self.gravcomp = self.gravcomp.numpy() if self.gravcomp is not None else None
        self.passive = getattr(mujoco, "dof_passive_stiffness", None)
        self.passive = self.passive.numpy() if self.passive is not None else None
        self.target_mode = model.joint_target_mode.numpy()
        self.ke, self.kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        self.damping, self.friction = model.joint_damping.numpy(), model.joint_friction.numpy()
        self.armature, self.enabled = model.joint_armature.numpy(), model.joint_enabled.numpy()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        self.body_q = state.body_q.numpy().astype(np.float64)
        pairs = model.shape_contact_pairs.numpy() if model.shape_contact_pair_count else np.zeros((0, 2), int)
        self.contact_pairs = {(int(min(a, b)), int(max(a, b))) for a, b in pairs}
        self.groups = model.shape_collision_group.numpy()

    def world_points(self, shape: int) -> np.ndarray | None:
        points = shape_points(self.model, shape)
        if points is None:
            return None
        pose = self.transform[shape]
        body = int(self.ix.shape_body[shape])
        return _apply(_compose(self.body_q[body], pose) if body >= 0 else pose, points)

    def blocked(self, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Shape pairs the model never lets touch (flags, collision groups, filter pairs, contact pair list)."""
        if not pairs:
            return []
        filtered = self.model.shape_collision_filter_mask(np.asarray(pairs, dtype=np.int64))
        out = []
        for (a, b), is_filtered in zip(pairs, filtered, strict=True):
            ga, gb = int(self.groups[a]), int(self.groups[b])
            groups_ok = ga != 0 and gb != 0 and ((ga > 0 and (ga == gb or gb < 0)) or (ga < 0 and ga != gb))
            flags_ok = bool(self.ix.shape_flags[a] & COLLIDE) and bool(self.ix.shape_flags[b] & COLLIDE)
            if not (groups_ok and flags_ok and not is_filtered and (min(a, b), max(a, b)) in self.contact_pairs):
                out.append((a, b))
        return out


def read_params(module) -> dict:
    """``PARAMS`` read once into plain floats (``nan`` if missing or not a number)."""
    raw = getattr(module, "PARAMS", {})
    params = {}
    for key, default in (("dt", math.nan), ("command_delay", 0.0)):
        value = raw.get(key, default) if isinstance(raw, dict) else math.nan
        params[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
    return params


FREQ = newton.Model.AttributeFrequency
IDENTITY_FREQUENCIES = (FREQ.SHAPE, FREQ.BODY, FREQ.JOINT, FREQ.JOINT_DOF, FREQ.JOINT_COORD, FREQ.WORLD)
# Arrays per-world copies may differ in: the entity-to-world maps, visual-only attributes, and handles into
# model-wide tables (the geometry sources behind them are compared by content).
IDENTITY_SKIP = {
    "body_world",
    "shape_world",
    "joint_world",
    "shape_color",
    "shape_opacity",
    "shape_source_ptr",
    "shape_heightfield_index",
    "shape_edge_range",
    "mujoco:geom_group",
    "mujoco:site_size_is_display",
    "mujoco:collision_mask_domain",  # which add_mjcf() call a shape came from
}


def model_arrays(model: newton.Model) -> dict[str, tuple]:
    """Copies of every per-entity (shape, body, joint, DOF, coordinate, world) array of the model, Newton's and
    its custom attributes (``model.mujoco``): name -> (frequency, referenced frequency or ``None``, values)."""
    counts = {
        FREQ.SHAPE: model.shape_count,
        FREQ.BODY: model.body_count,
        FREQ.JOINT: model.joint_count,
        FREQ.JOINT_DOF: model.joint_dof_count,
        FREQ.JOINT_COORD: model.joint_coord_count,
        FREQ.WORLD: model.world_count,
    }
    out = {}
    for name, spec in model._iter_attribute_specs():
        if spec.frequency not in IDENTITY_FREQUENCIES or name in IDENTITY_SKIP or name.startswith("_"):
            continue
        holder, attr = (
            (getattr(model, name.split(":")[0], None), name.split(":", 1)[1]) if ":" in name else (model, name)
        )
        value = getattr(holder, attr, None) if holder is not None else None
        if isinstance(value, wp.array) and value.shape and value.shape[0] == counts[spec.frequency]:
            out[name] = (FREQ(spec.frequency), spec.references, value.numpy().copy())
    return out


class Scene:
    """The submitted model as built, before this verifier's perturbations: per world the extra (non-station)
    bodies, the centre of each one's collision geometry, and which of them stand for the real objects.

    The real objects are assigned to free bodies by their starts: the assignment with the smallest summed xy
    distance between body centres and the video's starts; a body further than ``BOUNDS["match_m"]`` matches
    nothing. One remaining extra body may be the tray.
    """

    def __init__(self, model: newton.Model, station: Station, truth: dict):
        self.model = model
        self.ix = ix = Index(model)
        self.joint_q = model.joint_q.numpy().copy()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        self.body_q = state.body_q.numpy().astype(np.float64)
        self.transform = model.shape_transform.numpy().astype(np.float64)
        self.mu = model.shape_material_mu.numpy().copy()  # before the low-friction control
        self.arrays = model_arrays(model)  # every per-entity array, before this verifier's perturbations
        starts = {name: np.asarray(truth["objects"][name]["start_image_xyz"][:2]) for name in OBJECTS}
        self.worlds = []
        for w in range(model.world_count):
            bodies = ix.bodies(w)
            extra = sorted(b for leaf, found in bodies.items() if leaf not in station.bodies for b in found)
            centre_local, centre = {}, {}
            for b in extra:
                local = self.local_points(b)
                if local is not None:
                    centre_local[b] = _principal(local)[0]
                    centre[b] = _apply(self.body_q[b], centre_local[b][None])[0]
            candidates = sorted(centre)
            objects, distance = {}, {}
            if 1 <= len(candidates) <= 8:
                best = None
                for chosen in itertools.permutations(candidates, min(len(OBJECTS), len(candidates))):
                    cost = [
                        float(np.linalg.norm(centre[b][:2] - starts[n])) for n, b in zip(OBJECTS, chosen, strict=False)
                    ]
                    if best is None or sum(cost) < best[0]:
                        best = (sum(cost), chosen, cost)
                for name, b, cost in zip(OBJECTS, best[1], best[2], strict=False):
                    distance[name] = cost
                    if cost <= BOUNDS["match_m"]:
                        objects[name] = b
            rest = [b for b in extra if b not in objects.values()]
            self.worlds.append(
                {
                    "extra": extra,
                    "centre_local": centre_local,
                    "centre": centre,
                    "objects": objects,
                    "match_distance": distance,
                    "tray_body": rest[0] if len(rest) == 1 else None,
                    "unexplained": rest[1:] if len(rest) > 1 else [],
                }
            )

    def local_points(self, body: int) -> np.ndarray | None:
        """Surface points of a body's collision shapes in its frame; ``None`` without (bounded) shapes."""
        shapes = [s for s in np.flatnonzero(self.ix.shape_body == body) if self.ix.shape_flags[s] & COLLIDE]
        points = [shape_points(self.model, int(s)) for s in shapes]
        if not shapes or any(p is None for p in points):
            return None
        return np.vstack([_apply(self.transform[s], p) for s, p in zip(shapes, points, strict=True)])


def audit_submission(
    model, solver, pipeline, params: dict, worlds: list[dict], station: Station, scene: Scene, truth: dict, audit: Audit
) -> None:
    """Integrity checks of the submitted model, solver, and pipeline (see the module docstring)."""
    for key in ("dt", "command_delay"):
        value, (low, high) = params[key], BOUNDS[key]
        audit.check("params", low <= value <= high, f"PARAMS[{key!r}] = {value} outside [{low}, {high}]")
    audit.check("model_type", type(model) is newton.Model, f"model is a {type(model).__name__}")
    audit.check("model_type", not overridden(model), f"model instance overrides {overridden(model)}")
    solver_name = f"{type(solver).__module__}.{type(solver).__name__}"
    audit.check("solver_type", type(solver) in SOLVERS, f"solver {solver_name} is not an allowed Newton solver")
    audit.check("solver_type", not overridden(solver), f"solver instance overrides {overridden(solver)}")
    ok = pipeline is None or type(pipeline) is newton.CollisionPipeline
    audit.check("pipeline_type", ok, f"pipeline is a {type(pipeline).__name__}")
    audit.check("pipeline_type", not overridden(pipeline), f"pipeline instance overrides {overridden(pipeline)}")
    audit.check("solver_type", getattr(solver, "model", None) is model, "the solver simulates another model")
    foreign = foreign_members(solver)
    audit.check("solver_type", not foreign, f"the solver holds objects defined outside Newton: {foreign}")
    foreign = foreign_members(pipeline)
    audit.check("pipeline_type", not foreign, f"the pipeline holds objects defined outside Newton: {foreign}")
    if pipeline is not None:
        audit.check("pipeline_type", getattr(pipeline, "model", None) is model, "the pipeline is for another model")
    if not audit.check("world_count", model.world_count == len(worlds), f"{model.world_count} worlds"):
        return

    sub = Submission(model)
    explicit = getattr(pipeline, "shape_pairs_filtered", None) if pipeline is not None else None
    if isinstance(explicit, wp.array):
        # An explicit broad phase only tests the pairs it was given.
        given = explicit.numpy().reshape(-1, 2)
        sub.contact_pairs &= {(int(min(a, b)), int(max(a, b))) for a, b in given}
    gravity = model.gravity.numpy().reshape(-1, 3)[: len(worlds)]
    audit.check("gravity", np.allclose(gravity, [0.0, 0.0, -9.81], atol=1e-4), f"gravity {gravity[0].tolist()}")
    audit.check("structure", model.particle_count == 0, f"{model.particle_count} particles")
    audit.check("structure", not np.any(sub.ix.body_world < 0), "global bodies")
    audit.check("structure", not np.any(sub.ix.joint_world < 0), "global joints")
    audit.check("structure", not sub.ix.colliders(-1), "global collision shapes")
    audit.check("applied_forces", not np.any(model.joint_f.numpy()), "model.joint_f is not zero")
    if model.joint_act is not None:
        audit.check("applied_forces", not np.any(model.joint_act.numpy()), "model.joint_act is not zero")

    station_shapes = {}
    for w in range(len(worlds)):
        roles = _audit_world(sub, station, scene, w, truth, audit)
        station_shapes.update(roles["station"])
        for name in OBJECTS:
            _audit_object(sub, scene, w, name, roles, truth, audit)
        # Objects collide with the fingers, the table, the tray, and each other.
        required = []
        for i, name in enumerate(OBJECTS):
            for s in roles["objects"].get(name, []):
                required += [(s, other) for other in roles["pads"] + roles["table"] + roles["tray"]]
                for other in OBJECTS[i + 1 :]:
                    required += [(s, o) for o in roles["objects"].get(other, [])]
        blocked = sub.blocked(required)
        names = ", ".join(f"{sub.ix.shape_leaf[a]}/{sub.ix.shape_leaf[b]}" for a, b in blocked[:3])
        audit.check(
            "object_collisions", not blocked, f"world {w}: {len(blocked)} required pairs never collide: {names}"
        )
    _audit_identical(sub, scene, audit)
    _audit_materials(sub, station_shapes, audit)
    _audit_constraints(sub, station, len(worlds), audit)
    if sub.gravcomp is not None:
        low, high = BOUNDS["gravcomp"]
        bad = [
            leaf
            for i, leaf in enumerate(sub.ix.body_leaf)
            if leaf in station.bodies and not low <= sub.gravcomp[i] <= high
        ]
        audit.check("gravcomp", not bad, f"gravity compensation outside [0, 1] on {bad[:4]}")
    _audit_drives(sub, solver, audit)
    if isinstance(solver, newton.solvers.SolverMuJoCo):
        audit_mujoco(sub, solver, scene, params, audit)


def _audit_world(sub: Submission, station: Station, scene: Scene, w: int, truth: dict, audit: Audit) -> dict:
    """Bodies and joints, station kinematics and bases, station collision shapes, table, and the added geometry
    of one world: shapes on the matched object bodies are the objects', every other added collision shape
    (static, on a station body welded to the world, or on the one remaining extra body) is the tray's."""
    ix, table_z, info = sub.ix, float(truth["table_z"]), scene.worlds[w]
    bodies = ix.bodies(w)
    for leaf in station.bodies:
        found = len(bodies.get(leaf, []))
        audit.check("structure", found == 1, f"world {w}: {found} bodies named {leaf}")
    for name in OBJECTS:
        distance = info["match_distance"].get(name)
        where = "no candidate" if distance is None else f"the closest is {100 * distance:.1f} cm away"
        message = f"world {w}: no free body starts within {100 * BOUNDS['match_m']:.0f} cm of the real {name} ({where})"
        audit.check("object_match", name in info["objects"], message)
    extra, tray_body = info["extra"], info["tray_body"]
    labels = [ix.body_leaf[b] for b in info["unexplained"]]
    audit.check(
        "structure", not info["unexplained"], f"world {w}: more than one extra body besides the objects {labels}"
    )
    without = [ix.body_leaf[b] for b in extra if b not in info["centre"]]
    audit.check("structure", not without, f"world {w}: extra bodies without bounded collision shapes {without[:4]}")
    joints, expected = int(np.count_nonzero(ix.joint_world == w)), len(station.joints) + len(extra)
    audit.check("structure", joints == expected, f"world {w}: {joints} joints, expected {expected}")

    bases = truth["mjcf_bases"]
    for leaf, ref in station.joints.items():
        if len(bodies.get(leaf, [])) != 1:
            continue
        into = ix.joints_into(bodies[leaf][0])
        if not audit.check("kinematics", len(into) == 1, f"world {w}: {len(into)} joints into {leaf}"):
            continue
        j = into[0]
        parent = ix.body_leaf[ix.joint_parent[j]] if ix.joint_parent[j] >= 0 else None
        dofs = ix.dofs(j)
        same = int(ix.joint_type[j]) == ref["type"] and parent == ref["parent"] and len(dofs) == len(ref["axis"])
        if not audit.check("kinematics", same, f"world {w}: joint into {leaf} has another type or parent"):
            continue
        side = BASE_JOINTS.get(ref["leaf"])
        if side is not None:
            base = np.asarray(bases[side], dtype=np.float64)
            ok = float(np.abs(sub.x_p[j][:3] - base).max()) <= BOUNDS["base_m"]
            ok = ok and _rotation_close(sub.x_p[j][3:], ref["X_p"][3:])
            audit.check("arm_bases", ok, f"world {w}: {side} base at {np.round(sub.x_p[j][:3], 4).tolist()}")
        else:
            audit.check("kinematics", _pose_close(sub.x_p[j], ref["X_p"]), f"world {w}: parent frame into {leaf}")
        audit.check("kinematics", _pose_close(sub.x_c[j], ref["X_c"]), f"world {w}: child frame into {leaf}")
        if dofs:
            audit.check("kinematics", np.allclose(sub.axis[dofs], ref["axis"], atol=1e-4), f"world {w}: {leaf} axis")
            ok = np.allclose(sub.limits[dofs], ref["limits"], atol=1e-4)
            audit.check("kinematics", ok, f"world {w}: {leaf} joint limits")

    station_of = {bodies[leaf][0]: leaf for leaf in station.bodies if len(bodies.get(leaf, [])) == 1}
    object_of = {b: name for name, b in info["objects"].items()}
    roles = {"station": {}, "objects": {name: [] for name in OBJECTS}, "pads": [], "table": [], "tray": []}
    for s in ix.colliders(w):
        body = int(ix.shape_body[s])
        key = (station_of.get(body), ix.shape_leaf[s])
        if body in object_of:
            roles["objects"][object_of[body]].append(s)
        elif (body < 0 or body in station_of) and key in station.shapes:
            ref = station.shapes[key]
            same = int(sub.kind[s]) == ref["type"] and np.allclose(sub.scale[s], ref["scale"], atol=BOUNDS["frame_m"])
            same = same and _pose_close(sub.transform[s], ref["xform"])
            audit.check("station_shapes", same, f"world {w}: {key[1]} on {key[0]} differs from the MJCF")
            roles["station"][s] = ref
            if key[0] in FINGER_BODIES:
                roles["pads"].append(s)
            if key[1] == "table_plane":
                roles["table"].append(s)
        elif body < 0 or (body >= 0 and body == tray_body) or station_of.get(body) in station.welded:
            roles["tray"].append(s)
        elif body in station_of:
            audit.check("station_shapes", False, f"world {w}: added collision shape {key[1]} on {key[0]}")
    present = {(station_of.get(int(ix.shape_body[s])), ix.shape_leaf[s]) for s in roles["pads"]}
    missing = [key[1] for key in station.shapes if key[0] in FINGER_BODIES and key not in present]
    audit.check("finger_geometry", not missing, f"world {w}: finger collision shapes missing: {missing[:4]}")
    if audit.check("table", len(roles["table"]) == 1, f"world {w}: {len(roles['table'])} table planes"):
        t = roles["table"][0]
        up = core.quat_to_matrix(sub.transform[t][3:7])[2, 2]
        ok = int(sub.kind[t]) == GEO.PLANE and abs(sub.transform[t][2] - table_z) <= BOUNDS["table_z_m"] and up > 0.999
        audit.check("table", ok, f"world {w}: table plane at z {sub.transform[t][2]:.4f}")

    # The tray: every added shape inside the real tray's footprint (at the start or the end of the episode, as for
    # placement) + tray_margin_m, at most tray_top_m high.
    tray = truth["tray"]
    count = len(roles["tray"])
    audit.check("tray", count <= BOUNDS["tray_shapes_max"], f"world {w}: {count} added tray shapes")
    for s in roles["tray"]:
        points = sub.world_points(s)
        if points is None:
            audit.check("tray", False, f"world {w}: unbounded added shape {ix.shape_leaf[s]}")
            continue
        stride = max(1, len(points) // 400)
        outside = max(core.tray_distance(p[:2], tray) for p in points[::stride])
        top = float(points[:, 2].max() - table_z)
        ok = outside <= BOUNDS["tray_margin_m"] and top <= BOUNDS["tray_top_m"]
        message = f"{1000 * outside:.0f} mm outside the tray, top {1000 * top:.0f} mm above the table"
        audit.check("tray", ok, f"world {w}: added shape {ix.shape_leaf[s]} {message}")
    if tray_body is not None:
        low, high = BOUNDS["tray_mass_kg"]
        tag = f"world {w}: extra body {ix.body_leaf[tray_body]} (tray)"
        audit.check("tray", low <= sub.mass[tray_body] <= high, f"{tag} mass {sub.mass[tray_body]:.3f} kg")
        ok = sub.mass[tray_body] > 0 and abs(sub.inv_mass[tray_body] * sub.mass[tray_body] - 1.0) < 1e-3
        audit.check("tray", ok, f"{tag} inverse mass")
        _audit_free_body(sub, tray_body, tag, "tray", audit, fixed=True)
    roles["object_bodies"] = dict(info["objects"])
    return roles


def _audit_free_body(sub: Submission, body: int, tag: str, prefix: str, audit: Audit, fixed: bool = False) -> bool:
    """A free root body (or, with ``fixed``, one welded to the world) without drives, springs, damping, joint
    friction, armature, or gravity compensation. Returns whether its joint is free."""
    ix = sub.ix
    into = ix.joints_into(body)
    root = len(into) == 1 and ix.joint_parent[into[0]] < 0
    kind = int(ix.joint_type[into[0]]) if len(into) == 1 else None
    free = root and kind == newton.JointType.FREE
    ok = free or (fixed and root and kind == newton.JointType.FIXED)
    audit.check(f"{prefix}_joint", ok, f"{tag} is not a free root body")
    audit.check(f"{prefix}_joint", not np.any(ix.joint_parent == body), f"{tag} has bodies attached")
    if sub.gravcomp is not None:
        audit.check(f"{prefix}_gravcomp", sub.gravcomp[body] == 0.0, f"{tag} gravity compensation {sub.gravcomp[body]}")
    if free:
        j = into[0]
        audit.check(f"{prefix}_joint", bool(sub.enabled[j]), f"{tag} joint disabled")
        for d in ix.dofs(j):
            drive = int(sub.target_mode[d]) == int(newton.JointTargetMode.NONE) and sub.ke[d] == 0 and sub.kd[d] == 0
            if sub.passive is not None:
                drive = drive and sub.passive[d] == 0.0
            audit.check(f"{prefix}_drive", drive, f"{tag} dof {d - ix.qd_start[j]} has a drive or spring")
            ok = sub.damping[d] == 0 and sub.friction[d] == 0 and sub.armature[d] <= BOUNDS["object_armature"]
            audit.check(f"{prefix}_damping", ok, f"{tag} dof {d - ix.qd_start[j]} has damping, friction, or armature")
    return free


def _audit_object(sub: Submission, scene: Scene, w: int, name: str, roles: dict, truth: dict, audit: Audit) -> None:
    """A real object's body: free, dynamic, without drives; within its size range, mass, and inertia bounds;
    resting on the table near the video's start."""
    body = roles["object_bodies"].get(name)
    if body is None:
        return
    real = truth["objects"][name]
    tag = f"world {w}: body {sub.ix.body_leaf[body]} ({name})"
    _audit_free_body(sub, body, tag, "object", audit)
    if sub.body_flags is not None:
        ok = int(sub.body_flags[body]) == int(newton.BodyFlags.DYNAMIC)
        audit.check("object_dynamic", ok, f"{tag} body flags {sub.body_flags[body]}")
    mass = float(sub.mass[body])
    low, high = BOUNDS["object_mass_kg"]
    audit.check("object_mass", low <= mass <= high, f"{tag} mass {mass:.4f} kg")
    audit.check("object_mass", mass > 0 and abs(sub.inv_mass[body] * mass - 1.0) < 1e-3, f"{tag} inverse mass")

    shapes = roles["objects"][name]
    if not audit.check(
        "object_shapes", 1 <= len(shapes) <= BOUNDS["object_shapes_max"], f"{tag}: {len(shapes)} shapes"
    ):
        return
    local = [shape_points(sub.model, s) for s in shapes]
    if not audit.check("object_shapes", all(p is not None for p in local), f"{tag} has an unbounded shape"):
        return
    local = np.vstack([_apply(sub.transform[s], p) for s, p in zip(shapes, local, strict=True)])
    centre, extents, _ = _principal(local)
    low_scale, high_scale = BOUNDS["object_size_scale"]
    ranges = np.asarray(real["extent_ranges_m"], dtype=np.float64)
    fits = bool(np.all(extents >= low_scale * ranges[:, 0]) and np.all(extents <= high_scale * ranges[:, 1]))
    audit.check("object_size", fits, f"{tag} extents {np.round(1000 * extents, 1).tolist()} mm")

    # Principal moments between 0.8 x a solid and 1.2 x a thin-shell ellipsoid of the object's extents.
    a, b, c = extents / 2
    solid = np.sort(mass * np.array([b * b + c * c, a * a + c * c, a * a + b * b]) / 5.0)
    matrix = sub.inertia[body].astype(np.float64).reshape(3, 3)
    moments = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    ok = bool(np.all(moments >= 0.8 * solid - 1e-12) and np.all(moments <= 1.2 * solid * 5.0 / 3.0 + 1e-12))
    audit.check("object_inertia", ok, f"{tag} principal moments {np.round(moments, 8).tolist()}")
    ok = np.allclose(sub.inv_inertia[body].reshape(3, 3).astype(np.float64) @ matrix, np.eye(3), atol=1e-2)
    audit.check("object_inertia", ok, f"{tag} inverse inertia")
    offset = float(np.linalg.norm(sub.com[body] - centre))
    audit.check("object_inertia", offset <= BOUNDS["object_com_offset_m"], f"{tag} centre of mass {offset:.3f} m off")

    # The start, before this verifier's perturbations.
    pose = scene.body_q[body]
    world_centre = _apply(pose, centre[None])[0]
    offset = float(np.linalg.norm(world_centre[:2] - np.asarray(real["start_image_xyz"][:2])))
    message = f"{tag} starts {100 * offset:.1f} cm from the video's start"
    audit.check("object_start", offset <= BOUNDS["start_xy_m"], message)
    clearance = float(_apply(pose, local)[:, 2].min() - truth["table_z"])
    low, high = BOUNDS["start_clearance_m"]
    message = f"{tag} lowest point {1000 * clearance:.1f} mm above the table"
    audit.check("object_start", low <= clearance <= high, message)


def _source_digest(source) -> str:
    """Fingerprint of a shape's geometry source (mesh vertices and indices, height field data)."""
    if source is None:
        return ""
    digest = hashlib.sha1(type(source).__name__.encode())
    for attr in ("vertices", "indices", "data", "nrow", "ncol", "hx", "hy", "min_z", "max_z"):
        value = getattr(source, attr, None)
        if value is None or callable(value):
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(f"{attr}{array.shape}{array.dtype}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _entity_worlds(model: newton.Model, ix: Index) -> dict:
    """World of every entity, per frequency."""
    dof_world = np.repeat(ix.joint_world, np.diff(ix.qd_start))
    coord_world = np.repeat(ix.joint_world, np.diff(model.joint_q_start.numpy()))
    worlds = {
        FREQ.SHAPE: ix.shape_world,
        FREQ.BODY: ix.body_world,
        FREQ.JOINT: ix.joint_world,
        FREQ.JOINT_DOF: dof_world,
        FREQ.JOINT_COORD: coord_world,
        FREQ.WORLD: np.arange(model.world_count),
    }
    articulation_world = getattr(model, "articulation_world", None)
    if model.articulation_count and articulation_world is not None:
        worlds[FREQ.ARTICULATION] = articulation_world.numpy()
    return worlds


def _audit_identical(sub: Submission, scene: Scene, audit: Audit) -> None:
    """Every world is a copy of world 0: every per-entity array of the model as built (Newton's and its
    ``model.mujoco`` attributes, before this verifier's perturbations and controls; entities compared in their
    order within the world, indices taken relative to the world) and the shapes' mesh or height-field sources.
    SolverMuJoCo with its own contacts collides every world against world 0's mesh, cone, and height-field
    assets, so differing geometry would not be simulated as built."""
    model, ix, tol = sub.model, sub.ix, BOUNDS["identical_tol"]
    worlds = _entity_worlds(model, ix)
    starts = {
        freq: np.array([int(np.argmax(of == w)) if np.any(of == w) else 0 for w in range(model.world_count)])
        for freq, of in worlds.items()
    }
    rows_of = {}
    for name, (freq, referenced, values) in scene.arrays.items():
        of = worlds[freq]
        try:
            references = None if referenced is None else FREQ(referenced)
        except ValueError:
            references = "custom"
        relative = references in starts and references != FREQ.WORLD
        if references is not None and not relative:
            continue  # indices into tables this check cannot map per world
        per_world = []
        for w in range(model.world_count):
            rows = values[of == w]
            if relative:
                rows = rows.astype(np.int64)
                rows = np.where(rows >= 0, rows - starts[references][w], rows)
            per_world.append(rows)
        rows_of[name] = per_world
    digests = {}
    reference_sources = None
    for w in range(model.world_count):
        sources = []
        for shape in np.flatnonzero(ix.shape_world == w):
            source = model.shape_source[int(shape)]
            if id(source) not in digests:
                digests[id(source)] = _source_digest(source)
            sources.append(digests[id(source)])
        if reference_sources is None:
            reference_sources = sources
            continue
        if sources != reference_sources:
            audit.check("identical_worlds", False, f"world {w} differs from world 0 in shape sources (meshes)")
        differs = []
        for name, per_world in rows_of.items():
            a, b = per_world[w], per_world[0]
            if a.shape != b.shape:
                differs.append(name)
            elif a.dtype.kind in "fc":
                if not np.allclose(a.astype(np.float64), b.astype(np.float64), atol=tol, rtol=tol, equal_nan=True):
                    differs.append(name)
            elif not np.array_equal(a, b):
                differs.append(name)
        audit.check("identical_worlds", not differs, f"world {w} differs from world 0 in {differs[:6]}")


def _audit_materials(sub: Submission, station_shapes: dict, audit: Audit) -> None:
    """Material bounds of every collision shape; station shapes may keep their MJCF values."""
    model = sub.model
    values = {
        "mu": model.shape_material_mu.numpy(),
        "mu_torsional": model.shape_material_mu_torsional.numpy(),
        "mu_rolling": model.shape_material_mu_rolling.numpy(),
        "restitution": model.shape_material_restitution.numpy(),
        "margin": model.shape_margin.numpy(),
        "gap": model.shape_gap.numpy(),
    }
    adhesion = model.shape_material_ka.numpy()
    for s in np.flatnonzero((sub.ix.shape_flags & COLLIDE) != 0):
        ref = station_shapes.get(int(s), {})
        for key, array in values.items():
            limit = max(BOUNDS[key], ref.get(key, 0.0))
            value = float(array[s])
            audit.check("materials", 0.0 <= value <= limit + 1e-6, f"{sub.ix.shape_leaf[s]}: {key} {value:g}")
        audit.check("materials", float(adhesion[s]) == 0.0, f"{sub.ix.shape_leaf[s]}: adhesion {float(adhesion[s]):g}")


def _multiset(rows) -> dict:
    out = {}
    for row in rows:
        out[row] = out.get(row, 0) + 1
    return out


def _equalities(model: newton.Model, ix: Index, world: int) -> dict:
    mj = getattr(model, "mujoco", None)
    count = int(getattr(mj, "equality_constraint_count", 0) or 0)
    if not count:
        return {}
    in_world = mj.equality_constraint_world.numpy() == world
    kind = mj.equality_constraint_type.numpy()
    joints = mj.equality_constraint_joint1.numpy(), mj.equality_constraint_joint2.numpy()
    bodies = mj.equality_constraint_body1.numpy(), mj.equality_constraint_body2.numpy()
    poly, on = mj.equality_constraint_polycoef.numpy(), mj.equality_constraint_enabled.numpy()

    def leaf(table, i):
        return table[int(i)] if i >= 0 else None

    return _multiset(
        (
            int(kind[e]),
            *(leaf(ix.joint_leaf, j[e]) for j in joints),
            *(leaf(ix.body_leaf, b[e]) for b in bodies),
            tuple(np.round(poly[e], 4).tolist()),
            bool(on[e]),
        )
        for e in np.flatnonzero(in_world)
    )


def _mimics(model: newton.Model, ix: Index, world: int) -> dict:
    rows = []
    if model.constraint_mimic_count:
        in_world = model.constraint_mimic_world.numpy() == world
        a, b = model.constraint_mimic_joint0.numpy(), model.constraint_mimic_joint1.numpy()
        c0, c1 = model.constraint_mimic_coef0.numpy(), model.constraint_mimic_coef1.numpy()
        on = model.constraint_mimic_enabled.numpy()
        for i in np.flatnonzero(in_world):
            rows.append(
                (ix.joint_leaf[a[i]], ix.joint_leaf[b[i]], round(float(c0[i]), 4), round(float(c1[i]), 4), bool(on[i]))
            )
    if model.joint_mimic_joint is not None:
        leader = model.joint_mimic_joint.numpy()
        for j in np.flatnonzero((ix.joint_world == world) & (leader >= 0)):
            rows.append((ix.joint_leaf[j], ix.joint_leaf[int(leader[j])]))
    return _multiset(rows)


def _custom_actuators(model: newton.Model, ix: Index, world: int) -> dict:
    mj = getattr(model, "mujoco", None)
    count = model.custom_frequency_counts.get("mujoco:actuator", 0)
    if not count or mj is None:
        return {}
    in_world = mj.actuator_world.numpy() == world
    source, kind, target = mj.ctrl_source.numpy(), mj.actuator_trntype.numpy(), mj.actuator_trnid.numpy()[:, 0]
    rows = []
    for a in np.flatnonzero(in_world):
        dof = int(target[a])
        joint = ix.joint_leaf[ix.dof_joint(dof)] if int(kind[a]) == 0 and 0 <= dof < model.joint_dof_count else dof
        rows.append((int(source[a]), int(kind[a]), joint))
    return _multiset(rows)


def _audit_constraints(sub: Submission, station: Station, worlds: int, audit: Audit) -> None:
    """Equality constraints, mimics, and MuJoCo actuators exactly as in the station; no tendons or contact pairs."""
    ref_ix = station.index
    ref = (
        _equalities(station.model, ref_ix, 0),
        _mimics(station.model, ref_ix, 0),
        _custom_actuators(station.model, ref_ix, 0),
    )
    for w in range(worlds):
        audit.check("equality_constraints", _equalities(sub.model, sub.ix, w) == ref[0], f"world {w}: equalities")
        audit.check("equality_constraints", _mimics(sub.model, sub.ix, w) == ref[1], f"world {w}: mimic constraints")
        audit.check("actuators", _custom_actuators(sub.model, sub.ix, w) == ref[2], f"world {w}: MuJoCo actuators")
    for frequency in ("mujoco:tendon", "mujoco:tendon_joint", "mujoco:tendon_wrap", "mujoco:pair"):
        count = sub.model.custom_frequency_counts.get(frequency, 0)
        audit.check("actuators", count == 0, f"{count} {frequency} entries")
    mujoco = getattr(sub.model, "mujoco", None)
    for key in ("wind", "density", "viscosity"):
        if mujoco is not None and hasattr(mujoco, key):
            audit.check("environment", not np.any(getattr(mujoco, key).numpy()), f"mujoco:{key} is not zero")


def _np(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _per_world(array, worlds: int, item_ndim: int) -> np.ndarray:
    """MuJoCo Warp model arrays are batched over worlds unless shared; broadcast the shared ones."""
    array = _np(array)
    return np.broadcast_to(array, (worlds, *array.shape)) if array.ndim == item_ndim else array


def effective_drives(sub: Submission, solver) -> dict:
    """Per (world, driven station joint): position gain [N/m or N m/rad] and effective force limit [N or N m]."""
    model, ix, out = sub.model, sub.ix, {}
    if isinstance(solver, newton.solvers.SolverMuJoCo):
        mj, mw, worlds = solver.mj_model, solver.mjw_model, model.world_count
        to_newton = _per_world(solver.mjc_jnt_to_newton_jnt, worlds, 1)
        gain, bias = _per_world(mw.actuator_gainprm, worlds, 2), _per_world(mw.actuator_biasprm, worlds, 2)
        forcerange = _per_world(mw.actuator_forcerange, worlds, 2)
        joint_range = _per_world(mw.jnt_actfrcrange, worlds, 2) if hasattr(mw, "jnt_actfrcrange") else None
        for w in range(worlds):
            for a in range(mj.nu):
                if int(mj.actuator_trntype[a]) != 0:
                    continue
                jid = int(mj.actuator_trnid[a, 0])
                entry = out.setdefault((w, ix.joint_leaf[int(to_newton[w, jid])]), {"kp": 0.0, "limit": math.inf})
                if float(bias[w, a, 1]) != 0.0:  # position servo (velocity servos have no position bias)
                    entry["kp"] = max(entry["kp"], float(gain[w, a, 0]))
                if mj.actuator_forcelimited[a]:
                    entry["limit"] = min(entry["limit"], float(np.abs(forcerange[w, a]).max()))
                if joint_range is not None and mj.jnt_actfrclimited[jid]:
                    entry["limit"] = min(entry["limit"], float(np.abs(joint_range[w, jid]).max()))
    else:
        effort = model.joint_effort_limit.numpy()
        for j in range(model.joint_count):
            for d in ix.dofs(j):
                if int(sub.target_mode[d]) != int(newton.JointTargetMode.NONE):
                    out[(int(ix.joint_world[j]), ix.joint_leaf[j])] = {
                        "kp": float(sub.ke[d]),
                        "limit": float(effort[d]),
                    }
    return {key: value for key, value in out.items() if key[1] in DRIVEN_JOINTS}


def _audit_drives(sub: Submission, solver, audit: Audit) -> None:
    """Arm effort limits and the gripper's position gain and squeeze force, as the solver applies them."""
    for (w, leaf), drive in effective_drives(sub, solver).items():
        if leaf.endswith("_finger"):
            if drive["kp"] == 0.0:
                continue
            low, high = BOUNDS["finger_kp"]
            audit.check("actuators", low <= drive["kp"] <= high, f"world {w}: {leaf} kp {drive['kp']:g}")
            force = min(drive["limit"], drive["kp"] * GRIPPER_RANGE)
            low, high = BOUNDS["finger_force_n"]
            audit.check("actuators", low <= force <= high, f"world {w}: {leaf} squeezes with up to {force:.1f} N")
        else:
            cap = BOUNDS["arm_effort_nm"][int(leaf[-1]) - 1]
            audit.check("actuators", drive["limit"] <= cap + 1e-3, f"world {w}: {leaf} effort limit {drive['limit']:g}")


def solref_problems(solref: np.ndarray, dt: float, refsafe: bool, what: str) -> list[str]:
    """MuJoCo solref rows [n, 2] outside the stability bounds: standard (positive) form, time constant at least
    2 dt as MuJoCo applies it (its refsafe raises shorter ones), damping ratio within ``BOUNDS["dampratio"]``."""
    solref = np.asarray(solref, dtype=np.float64).reshape(-1, 2)
    if not len(solref):
        return []
    low, high = BOUNDS["dampratio"]
    timeconst = np.maximum(solref[:, 0], 2.0 * dt) if refsafe else solref[:, 0]
    bad = (solref[:, 0] <= 0.0) | (solref[:, 1] <= 0.0) | (timeconst < 2.0 * dt * (1.0 - 1e-6))
    bad |= (solref[:, 1] < low - 1e-6) | (solref[:, 1] > high + 1e-6)
    if not bad.any():
        return []
    return [f"{int(bad.sum())} {what} solref outside the bounds, e.g. {np.round(solref[bad][0], 5).tolist()}"]


def audit_mujoco(sub: Submission, solver, scene: Scene, params: dict, audit: Audit) -> None:
    """What SolverMuJoCo actually simulates: actuators, equalities, options, the object and tray bodies and
    dofs, geom friction, and the contact and finger-mirror stiffness."""
    import mujoco

    model, ix = sub.model, sub.ix
    mj, mw, worlds = solver.mj_model, solver.mjw_model, model.world_count
    to_joint = _per_world(solver.mjc_jnt_to_newton_jnt, worlds, 1)
    to_body = _per_world(solver.mjc_body_to_newton, worlds, 1)
    to_shape = _per_world(solver.mjc_geom_to_newton_shape, worlds, 1)
    for a in range(mj.nu):
        kind, jid = int(mj.actuator_trntype[a]), int(mj.actuator_trnid[a, 0])
        leaf = ix.joint_leaf[int(to_joint[0, jid])] if kind == 0 and jid >= 0 else None
        audit.check("mujoco", leaf in DRIVEN_JOINTS, f"actuator {a} drives {leaf or f'transmission type {kind}'}")
    # Every actuator is a plain servo: force = kp (target - q) - kv qd (or kv (target_qd - qd)), unit gear, no
    # activation dynamics and no constant bias.
    gain, bias = _per_world(mw.actuator_gainprm, worlds, 2), _per_world(mw.actuator_biasprm, worlds, 2)
    gear = _per_world(mw.actuator_gear, worlds, 2)
    for a in range(mj.nu):
        law = (
            int(mj.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED)
            and int(mj.actuator_biastype[a]) in (int(mujoco.mjtBias.mjBIAS_NONE), int(mujoco.mjtBias.mjBIAS_AFFINE))
            and int(mj.actuator_dyntype[a]) == int(mujoco.mjtDyn.mjDYN_NONE)
        )
        audit.check("actuators", law, f"actuator {a}: gain, bias, or dynamics type of a non-servo")
        g, b = gain[:, a].astype(np.float64), bias[:, a].astype(np.float64)
        position = np.isclose(b[:, 1], -g[:, 0], rtol=1e-5, atol=1e-6)
        velocity = (b[:, 1] == 0.0) & np.isclose(b[:, 2], -g[:, 0], rtol=1e-5, atol=1e-6)
        ok = bool(np.all(g[:, 0] >= 0.0) and np.all(b[:, 0] == 0.0) and np.all(b[:, 2] <= 0.0))
        ok = ok and bool(np.all(position | velocity))
        audit.check("actuators", ok, f"actuator {a} is not a plain servo: gain {g[0, :3]}, bias {b[0, :3]}")
        unit = np.allclose(gear[:, a].astype(np.float64), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        audit.check("actuators", unit, f"actuator {a} gear {gear[0, a].tolist()}")
    for e in range(mj.neq):
        kind, a, b = int(mj.eq_type[e]), int(mj.eq_obj1id[e]), int(mj.eq_obj2id[e])
        names = (ix.joint_leaf[int(to_joint[0, a])], ix.joint_leaf[int(to_joint[0, b])]) if kind == 2 else None
        audit.check("mujoco", names in FINGER_PAIRS, f"equality constraint {e} of type {kind}")
    audit.check("mujoco", mj.neq == len(FINGER_PAIRS), f"{mj.neq} equality constraints")
    for count in ("ntendon", "npair", "nflex"):
        audit.check("mujoco", getattr(mj, count) == 0, f"{count} = {getattr(mj, count)}")
    disable, enable = int(_np(mw.opt.disableflags).reshape(-1)[0]), int(_np(mw.opt.enableflags).reshape(-1)[0])
    audit.check("mujoco", not disable & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY), "gravity disabled")
    audit.check("mujoco", not enable & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE), "contact override enabled")
    gravity = _per_world(mw.opt.gravity, worlds, 1)
    audit.check("mujoco", np.allclose(gravity, [0.0, 0.0, -9.81], atol=1e-4), "solver gravity differs")
    for key in ("wind", "density", "viscosity"):
        audit.check("mujoco", not np.any(_np(getattr(mw.opt, key))), f"solver {key} is not zero")

    added = set()
    for info in scene.worlds:
        added |= set(info["objects"].values())
        if info["tray_body"] is not None:
            added.add(info["tray_body"])
    mass, inertia = _per_world(mw.body_mass, worlds, 1), _per_world(mw.body_inertia, worlds, 2)
    gravcomp = _per_world(mw.body_gravcomp, worlds, 1)
    armature, damping = _per_world(mw.dof_armature, worlds, 1), _per_world(mw.dof_damping, worlds, 1)
    frictionloss, stiffness = _per_world(mw.dof_frictionloss, worlds, 1), _per_world(mw.jnt_stiffness, worlds, 1)
    for w in range(worlds):
        for b in range(mj.nbody):
            body = int(to_body[w, b])
            if body not in added:
                continue
            tag = f"world {w}: solver's {ix.body_leaf[body]}"
            audit.check("mujoco", gravcomp[w, b] == 0.0, f"{tag} has gravity compensation")
            moments = np.sort(np.linalg.eigvalsh(sub.inertia[body].reshape(3, 3).astype(np.float64)))
            same = abs(mass[w, b] - sub.mass[body]) <= 1e-4 * sub.mass[body]
            same = same and np.allclose(np.sort(inertia[w, b]), moments, rtol=1e-2, atol=1e-10)
            audit.check("mujoco", same, f"{tag} mass or inertia differs from the model")
            for d in np.flatnonzero(np.asarray(mj.dof_bodyid) == b):
                ok = damping[w, d] == 0 and frictionloss[w, d] == 0 and armature[w, d] <= BOUNDS["object_armature"]
                audit.check("mujoco", ok, f"{tag} dof {d} has damping, friction, or armature")
            for j in np.flatnonzero(np.asarray(mj.jnt_bodyid) == b):
                audit.check("mujoco", stiffness[w, j] == 0.0, f"{tag} joint has stiffness")
    friction = _per_world(mw.geom_friction, worlds, 2)
    material = np.stack(
        [
            model.shape_material_mu.numpy(),
            model.shape_material_mu_torsional.numpy(),
            model.shape_material_mu_rolling.numpy(),
        ],
        axis=-1,
    )
    for w in range(worlds):
        for g in np.flatnonzero(to_shape[w] >= 0):
            shape = int(to_shape[w, g])
            ok = np.allclose(friction[w, g], material[shape], rtol=1e-4, atol=1e-7)
            audit.check("mujoco", ok, f"world {w}: solver friction of {ix.shape_leaf[shape]} differs from the model")

    # Contact and finger-mirror stiffness within MuJoCo's stability limits (contacts are also sampled during the
    # rollout, see contact_problems).
    refsafe = not disable & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    audit.check("contact_stiffness", refsafe, "MuJoCo's refsafe (time constant at least 2 dt) is disabled")
    dt = params["dt"] if math.isfinite(params["dt"]) else BOUNDS["dt"][1]
    solref = _per_world(mw.geom_solref, worlds, 2)
    rows = np.concatenate([solref[w][to_shape[w] >= 0] for w in range(worlds)])
    for problem in solref_problems(rows, dt, refsafe, "geom"):
        audit.check("contact_stiffness", False, problem)
    if mj.neq:
        for problem in solref_problems(_per_world(mw.eq_solref, worlds, 2).reshape(-1, 2), dt, refsafe, "equality"):
            audit.check("contact_stiffness", False, problem)


# Solver arrays SolverMuJoCo rewrites from the Newton model (notify_model_changed) after compiling it, and
# constants it derives from them, so they may differ from the compiled CPU model; every other array must still
# equal it.
TWIN_EXEMPT = {
    "body_ipos",
    "body_iquat",
    "body_inertia",
    "jnt_range",
    "jnt_actfrcrange",
    "geom_size",
    "body_subtreemass",
    "body_invweight0",
    "dof_invweight0",
    "tendon_invweight0",
    "actuator_acc0",
    "actuator_lengthrange",
}


def solver_arrays(solver) -> dict[str, np.ndarray]:
    """Host copies of the MuJoCo Warp model's arrays and options."""
    out = {}
    for prefix, holder in (("", solver.mjw_model), ("opt.", solver.mjw_model.opt)):
        for key, value in vars(holder).items():
            if isinstance(value, wp.array):
                out[prefix + key] = value.numpy().copy()
    return out


def resync_solver(model: newton.Model, solver, audit: Audit, details: dict) -> None:
    """SolverMuJoCo keeps its own copy of the model: re-apply the audited model to it (what the submission did to
    the copy in between is overwritten), then check what the update does not rewrite. The solver's arrays must
    equal the model it compiled (SolverMuJoCo builds both from the Newton model), every world's rows must equal
    world 0's, bodies on fixed joints must sit where their Newton joints put them, and the solver may add no
    polynomial springs or dampers and no surface velocities. Runs before the controls change the friction."""
    if not isinstance(solver, newton.solvers.SolverMuJoCo):
        return
    before = solver_arrays(solver)
    solver.notify_model_changed(newton.ModelFlags.ALL)
    after = solver_arrays(solver)
    details["resync_changed"] = sorted(
        k
        for k in before
        if before[k].shape != after[k].shape
        or not np.allclose(before[k].astype(np.float64), after[k].astype(np.float64), atol=1e-6, rtol=1e-5)
    )[:20]
    mj, mw, worlds = solver.mj_model, solver.mjw_model, model.world_count
    twin, per_world = [], []
    for key, gpu in after.items():
        option = key.startswith("opt.")
        cpu = getattr(mj.opt, key[4:], None) if option else getattr(mj, key, None)
        if cpu is None or isinstance(cpu, (str, bytes)) or callable(cpu):
            continue  # derived index tables have no CPU twin and are not batched over worlds
        cpu = np.asarray(cpu)
        if gpu.ndim == cpu.ndim + 1 and gpu.shape[1:] == cpu.shape and gpu.shape[0] == worlds > 1:
            first = gpu[0].astype(np.float64)
            if any(not np.allclose(gpu[w].astype(np.float64), first, atol=1e-6, rtol=1e-5) for w in range(1, worlds)):
                per_world.append(key)
        if option or key in TWIN_EXEMPT or cpu.dtype.kind not in "biuf":
            continue
        cpu = cpu.astype(np.float64)
        if gpu.ndim == cpu.ndim + 1 and gpu.shape[1:] == cpu.shape:
            same = all(np.allclose(row.astype(np.float64), cpu, atol=1e-6, rtol=1e-5) for row in gpu)
        elif gpu.size == cpu.size:
            same = np.allclose(gpu.reshape(-1).astype(np.float64), cpu.reshape(-1), atol=1e-6, rtol=1e-5)
        else:
            same = False
        if not same:
            twin.append(key)
    message = "the solver's MuJoCo arrays differ from the model it compiled (edited, or the model changed after"
    audit.check("solver_model", not twin, f"{message} make_solver built the solver): {twin[:8]}")
    audit.check("identical_worlds", not per_world, f"the solver's worlds differ in {per_world[:8]}")
    for key in ("dof_dampingpoly", "jnt_stiffnesspoly", "geom_surfacevel", "actuator_delay"):
        value = after.get(key)
        audit.check("solver_model", value is None or not np.any(value), f"the solver's {key} is not zero")

    # Bodies on fixed joints (arm bases, finger pads) keep their solver pose through the update.
    ix = Index(model)
    to_body = _per_world(solver.mjc_body_to_newton, worlds, 1)
    x_p, x_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    pos, quat = _per_world(mw.body_pos, worlds, 2), _per_world(mw.body_quat, worlds, 2)
    moved = []
    for w in range(worlds):
        for b in range(1, mj.nbody):
            n = int(to_body[w, b])
            into = ix.joints_into(n) if n >= 0 else []
            if len(into) != 1 or int(ix.joint_type[into[0]]) != int(newton.JointType.FIXED):
                continue
            j = into[0]
            r_p, r_c = core.quat_to_matrix(x_p[j, 3:]), core.quat_to_matrix(x_c[j, 3:])
            expected_pos = x_p[j, :3] - r_p @ r_c.T @ x_c[j, :3]
            expected_rot = r_p @ r_c.T
            q = quat[w, b].astype(np.float64)  # wxyz
            rot = core.quat_to_matrix(np.array([q[1], q[2], q[3], q[0]]))
            if np.abs(pos[w, b] - expected_pos).max() > 1e-5 or np.abs(rot - expected_rot).max() > 1e-5:
                moved.append(f"world {w}: {ix.body_leaf[n]}")
    audit.check("solver_model", not moved, f"the solver moved bodies on fixed joints: {moved[:4]}")


# Packages whose objects a solver or collision pipeline may hold; anything else is submitted code (subclassed
# components, callbacks, kernels) that would run inside the verifier's rollout.
TRUSTED_PACKAGES = {"newton", "warp", "mujoco", "mujoco_warp", "numpy", "builtins", "enum", "collections", "typing"}
TRUSTED_PACKAGES |= {"functools", "types", "dataclasses", "_thread", "threading", "weakref", "ctypes"}


def _origin(value) -> str:
    """Top-level package that defines a value (for functions and Warp kernels: of their Python code)."""
    func = getattr(value, "func", None) if isinstance(value, (wp.Kernel, wp.Function)) else None
    if func is not None:
        value = func
    if callable(value) and not isinstance(value, type) and getattr(value, "__module__", None):
        module = value.__module__
    else:
        module = getattr(value if isinstance(value, type) else type(value), "__module__", None) or "builtins"
    return module.split(".")[0]


def foreign_members(obj) -> list[str]:
    """Attributes of a solver or pipeline (and the items of attribute containers) defined outside
    :data:`TRUSTED_PACKAGES`."""
    if obj is None:
        return []
    found = []
    for key, value in vars(obj).items():
        items = [value]
        if isinstance(value, (list, tuple, set)):
            items += list(value)[:1000]
        elif isinstance(value, dict):
            items += list(value.values())[:1000]
        for item in items:
            # Components (narrow phase, broad phase): their own members too.
            nested = vars(item).values() if _origin(item) == "newton" and hasattr(item, "__dict__") else ()
            for member in (item, *[m for m in nested if not isinstance(m, (dict, list, tuple, set))]):
                if _origin(member) not in TRUSTED_PACKAGES:
                    found.append(f"{key}: {type(member).__name__} from {_origin(member)}")
    return sorted(set(found))[:6]


def contact_problems(solver, dt: float) -> tuple[list[str], int]:
    """Contacts the MuJoCo solver currently simulates whose solref leaves the bounds; and their count."""
    import mujoco

    data = solver.mjw_data
    count = min(int(_np(data.nacon).reshape(-1)[0]), int(data.naconmax))
    if count <= 0:
        return [], 0
    disable = int(_np(solver.mjw_model.opt.disableflags).reshape(-1)[0])
    refsafe = not disable & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    contact = data.contact
    problems = solref_problems(_np(contact.solref)[:count], dt, refsafe, "contact")
    friction_ref = _np(contact.solreffriction)[:count].reshape(-1, 2)
    friction_ref = friction_ref[np.any(friction_ref != 0.0, axis=1)]  # (0, 0): the normal solref applies
    problems += solref_problems(friction_ref, dt, refsafe, "contact friction")
    return problems, count


# ----------------------------------------------------------------------------- perturbations, rollout, scoring


def apply_jitter(model: newton.Model, scene: Scene, worlds: list[dict]) -> np.ndarray:
    """Move and turn the matched objects of the perturbed worlds (about the vertical through the centre of their
    collision geometry) in ``model.joint_q``; returns the new ``joint_q``."""
    q = scene.joint_q.copy()
    q_start = model.joint_q_start.numpy()
    for w, spec in enumerate(worlds):
        if not spec["jitter"]:
            continue
        for name, body in scene.worlds[w]["objects"].items():
            into = scene.ix.joints_into(body)
            if len(into) != 1 or int(scene.ix.joint_type[into[0]]) != newton.JointType.FREE:
                continue
            dx, dy, yaw = spec["jitter"][name]
            start = int(q_start[into[0]])
            turn = core.quat_about_z(yaw)
            pose = scene.body_q[body]
            centre = _apply(pose, scene.worlds[w]["centre_local"][body][None])[0]
            position = centre + np.array([dx, dy, 0.0]) + core.quat_to_matrix(turn) @ (pose[:3] - centre)
            q[start : start + 3] = position
            q[start + 3 : start + 7] = core.quat_multiply(turn, pose[3:7])
    model.joint_q.assign(q.astype(np.float32))
    return q


def apply_controls(model: newton.Model, scene: Scene, worlds: list[dict]) -> list[int]:
    """Set the friction of all arm and object shapes in the low-friction control worlds; returns those shapes."""
    ix = scene.ix
    shapes = []
    for w, spec in enumerate(worlds):
        if spec["control"] != "low_mu":
            continue
        objects = set(scene.worlds[w]["objects"].values())
        for s in ix.colliders(w):
            body = int(ix.shape_body[s])
            if body >= 0 and (ix.body_leaf[body].startswith(ARM_PREFIXES) or body in objects):
                shapes.append(s)
    mu = model.shape_material_mu.numpy()
    mu[shapes] = CONTROL_MU
    model.shape_material_mu.assign(mu)
    return shapes


def _median(values: list, absolute: bool = False, missing: float = math.inf) -> float:
    """Median over copies; ``None``/non-finite values count as ``missing`` (a failure for upper bounds)."""
    values = [missing if v is None or not np.isfinite(v) else (abs(v) if absolute else v) for v in values]
    return float(np.median(values)) if values else missing


def _sample_steps(truth: dict, episode: dict, dt: float, total: int) -> list[int]:
    """Steps at which the solver's contacts are inspected: the middle of each real carry."""
    t = np.asarray(episode["left_t"], dtype=np.float64)
    rows = lambda frame: min(max(frame - 1, 0), len(t) - 1)  # noqa: E731
    steps = []
    for real in truth["objects"].values():
        events = real["events"]
        middle = 0.5 * (t[rows(events["liftoff"])] + t[rows(events["cmd_open"])])
        steps.append(int(np.clip(round((middle - t[0]) / dt), 1, total - 1)))
    return sorted(set(steps))


def rollout(
    model, solver, pipeline, params: dict, worlds: list[dict], scene: Scene, truth: dict, gt: dict, audit, timing
) -> list[dict]:
    """The verifier's own replay of every world; per world the metrics of each real object's body."""
    episodes = [spec["data"] for spec in worlds]
    replay = core.Replay(model, solver, pipeline, episodes, dt=params["dt"], command_delay=params["command_delay"])
    audit.check("applied_forces", not np.any(replay.control.joint_f.numpy()), "control.joint_f is not zero")
    audit.check("applied_forces", not np.any(replay.state_0.body_f.numpy()), "state.body_f is not zero")
    total = replay.total_steps
    probe = max(1, total // 50)
    tic = time.perf_counter()
    replay.step(probe)
    wp.synchronize_device(model.device)
    timing["predicted_run_s"] = predicted = (time.perf_counter() - tic) * total / probe
    if not audit.check("runtime", predicted <= RUNTIME_LIMIT_S, f"rollout would take {predicted:.0f} s"):
        return []
    done, inspected = probe, []
    if isinstance(solver, newton.solvers.SolverMuJoCo):
        for step in _sample_steps(truth, worlds[0]["data"], params["dt"], total):
            if step > done:
                replay.step(step - done)
                done = step
            problems, count = contact_problems(solver, params["dt"])
            inspected.append({"step": step, "contacts": count})
            for problem in problems:
                audit.check("contact_stiffness", False, f"step {step}: {problem}")
    timing["contact_samples"] = inspected
    replay.step(total - done)
    wp.synchronize_device(model.device)
    timing["run_s"] = time.perf_counter() - tic
    results = []
    for w, rec in enumerate(replay.recordings()):
        info, result = scene.worlds[w], {"objects": {}, "arm_rmse_rad": {}}
        for side in SIDES:
            error = np.asarray(rec[f"{side}_q"]) - np.asarray(rec[f"{side}_q_real"])
            result["arm_rmse_rad"][side] = float(np.sqrt(np.mean(error**2)))
        for name in OBJECTS:
            body = info["objects"].get(name)
            if body is None:
                continue
            position = core.object_track(rec, body, info["centre_local"][body])
            track = gt.get(f"{name}_pos")
            result["objects"][name] = core.score_object(
                rec, position, truth["objects"][name], truth["tray"], truth["table_z"], track
            )
        results.append(result)
    return results


def evaluate(worlds: list[dict], results: list[dict], t: dict) -> tuple[dict, dict, list[str], dict]:
    """Ensemble gates: metrics, normalized worst values (> 1 fails), failed gates, and details."""
    metrics, normalized, failed, details = {}, {}, [], {}

    def gate(key: str, value: float, limit: float, kind: str) -> None:
        metrics[key] = value
        if kind == "min":
            normalized[key] = limit / value if value > 0 else math.inf
            ok = value >= limit
        else:
            normalized[key] = value / limit if limit > 0 else (0.0 if value <= 0 else math.inf)
            ok = value <= limit
        if not ok:
            failed.append(key)

    main = [r for spec, r in zip(worlds, results, strict=True) if spec["group"] == "main"]
    objects = {}
    for name in OBJECTS:
        rows = [r["objects"].get(name) for r in main]
        rows = [m for m in rows if m is not None]
        held = [m["held_fraction"] >= t["held_fraction_min"] for m in rows]
        placed = [bool(m["placed"] and m["rest_xy_err_m"] <= t["rest_xy_err_m_max"]) for m in rows]
        objects[name] = {
            "held": int(sum(held)),
            "placed": int(sum(placed)),
            "in_tray": int(sum(bool(m["placed"]) for m in rows)),
            "held_fraction": _median([m["held_fraction"] for m in rows], missing=0.0),
            "final_xy_err_m": _median([m["final_xy_err_m"] for m in rows]),
            "rest_xy_err_m": _median([m["rest_xy_err_m"] for m in rows]),
            "carry_track_err_m": _median([m["carry_track_err_m"] for m in rows]),
            "liftoff_err_rows": _median([m["liftoff_err_rows"] for m in rows], absolute=True),
            "release_err_s": _median([m["release_err_s"] for m in rows], missing=math.nan),
            "grip_gap_err_mm": _median([m["grip_gap_err_mm"] for m in rows], absolute=True),
            "slip_m": _median([m["slip_m"] for m in rows]),
            "moved_before_grasp_m": _median([m["moved_before_grasp_m"] for m in rows]),
            "final_xyz": [m["final_xyz"] for m in rows],
        }
    arm = {side: _median([r["arm_rmse_rad"][side] for r in main]) for side in SIDES}
    details["main"] = {"objects": objects, "arm_rmse_rad": arm, "copies": len(main)}
    gate("main_held_min", min(o["held"] for o in objects.values()), t["main_held_min"], "min")
    gate("main_placed_min", min(o["placed"] for o in objects.values()), t["main_placed_min"], "min")
    for name, o in objects.items():
        for key in ("held", "placed", "in_tray", "rest_xy_err_m", "final_xy_err_m", "carry_track_err_m"):
            metrics[f"main_{name}_{key}"] = o[key]
        for key in ("liftoff_err_rows", "release_err_s", "grip_gap_err_mm", "held_fraction"):
            metrics[f"main_{name}_{key}"] = o[key]
    metrics["main_rest_xy_err_m_max"] = max(o["rest_xy_err_m"] for o in objects.values())
    metrics["main_final_xy_err_m_max"] = max(o["final_xy_err_m"] for o in objects.values())
    metrics["main_arm_rmse_rad_max"] = max(arm.values())

    # A diverged control world (non-finite state) counts as a failed control.
    controls = {}
    for control in ("grip_open", "low_mu"):
        rows = [r for spec, r in zip(worlds, results, strict=True) if spec["control"] == control]
        rises = {name: [r["objects"][name]["max_rise_m"] for r in rows if name in r["objects"]] for name in OBJECTS}
        controls[control] = {"max_rise_m": rises}
        rise = max((v if np.isfinite(v) else math.inf for values in rises.values() for v in values), default=math.inf)
        gate(f"control_{control}_rise_m", rise, t["control_rise_m_max"], "max")
    details["controls"] = controls
    metrics["diverged_worlds"] = sum(
        not all(np.isfinite(m["max_rise_m"]) for m in r["objects"].values()) for r in results
    )
    return metrics, normalized, failed, details


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


def _changed(before: dict, model: newton.Model) -> list[str]:
    """Model arrays that differ from ``before`` (the verifier's values before make_solver and make_pipeline)."""
    return [key for key, value in before.items() if not np.array_equal(getattr(model, key).numpy(), value)]


WATCHED_ARRAYS = (
    "joint_q",
    "body_mass",
    "body_inertia",
    "body_com",
    "shape_scale",
    "shape_transform",
    "shape_material_mu",
    "shape_material_mu_torsional",
    "shape_material_mu_rolling",
)


def verify(script: Path, full: bool = False) -> dict:
    """One verification run in this process (the submission is imported here)."""
    started, timing = time.perf_counter(), {}
    thresholds = load_thresholds()
    result = {
        "task": "abc_scratch",
        "success": False,
        "thresholds": thresholds,
        "thresholds_source": thresholds_source(),
        "metrics": {},
        "normalized_worst": {},
    }
    newton.solvers.SolverMuJoCo.import_mujoco()
    station = Station()
    truth, gt = load_truth()
    # The copies' order is drawn per run, so the submitted code cannot tell the controls or the perturbed copies
    # by their index (the controls' friction is set only after make_solver and make_pipeline).
    worlds = plan_worlds()
    worlds = [worlds[i] for i in secrets.SystemRandom().sample(range(len(worlds)), len(worlds))]
    timing["world_order"] = [spec["control"] or f"main{spec['copy']}" for spec in worlds]
    threads, hooks = set(threading.enumerate()), _hooks()
    audit = Audit()
    with tempfile.TemporaryDirectory() as tmp:
        links = outside_links(script.parent)
        if links:
            audit.check("source", False, f"links to files outside the workspace: {links}")
            result.update(integrity=audit.checks, failed_checks=audit.failed, details={"integrity_notes": audit.notes})
            return {**result, "deterministic": True, "seconds": time.perf_counter() - started}
        path = prepare_workspace(script, Path(tmp))
        findings = source_findings(Path(tmp))
        if findings:
            audit.check("source", False, f"introspection in the submission: {findings}")
            result.update(integrity=audit.checks, failed_checks=audit.failed, details={"integrity_notes": audit.notes})
            return {**result, "deterministic": True, "seconds": time.perf_counter() - started}
        audit.check("source", True)
        blocked = install_guard(Path(tmp))
        handlers = _handlers()
        before = snapshot()
        scene, control_shapes, solver_info = None, [], {}
        try:
            module = load(path)
            tic = time.perf_counter()
            model = module.build_model(len(worlds))
            timing["build_s"] = time.perf_counter() - tic
            if isinstance(model, newton.Model) and model.world_count == len(worlds):
                scene = Scene(model, station, truth)
                apply_jitter(model, scene, worlds)
                expected = {key: getattr(model, key).numpy().copy() for key in WATCHED_ARRAYS}
            tic = time.perf_counter()
            solver = module.make_solver(model)
            pipeline = module.make_pipeline(model)
            timing["solver_s"] = time.perf_counter() - tic
            params = read_params(module)
        except Exception:
            result.update(error=traceback.format_exc()[-4000:], integrity={"build": False}, failed_checks=["build"])
            return {**result, "deterministic": True, "seconds": time.perf_counter() - started}

        # Submitted code runs only inside the calls above: no collector-run finalizers from here on.
        gc_enabled = gc.isenabled()
        gc.disable()

        def untampered(when: str) -> None:
            changed = patched(before)
            audit.check("untampered", not changed, f"replaced {when}: {changed[:6]}")
            audit.check("untampered", set(threading.enumerate()) <= threads, f"threads running {when}")
            audit.check("untampered", _hooks() == hooks, f"trace or profile hooks {when}")
            audit.check("untampered", _handlers() == handlers, f"signal handlers changed {when}")
            records = blocked()
            audit.check("sandbox", not records, f"blocked {when}: {records[:4]}")

        tic = time.perf_counter()
        try:
            if audit.check("world_count", scene is not None, "build_model(num_worlds) returned another model"):
                changed = _changed(expected, model)
                audit.check("perturbations", not changed, f"make_solver or make_pipeline changed model.{changed[:4]}")
                # Only a Newton solver's own code runs here (a substitute fails solver_type below).
                if type(solver) in SOLVERS and not overridden(solver):
                    resync_solver(model, solver, audit, solver_info)
                    control_shapes = apply_controls(model, scene, worlds)
                    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                audit_submission(model, solver, pipeline, params, worlds, station, scene, truth, audit)
                mu = model.shape_material_mu.numpy()[control_shapes]
                audit.check("controls", bool(np.all(mu == CONTROL_MU)), "the control worlds' friction was changed")
        except Exception:
            audit.check("audit", False, traceback.format_exc()[-2000:])
        untampered("after build")
        timing["audit_s"] = time.perf_counter() - tic
        structural = audit.failed
        metrics, normalized, gates, details, results = {}, {}, [], {}, []
        if scene is not None and (not structural or full):
            try:
                results = rollout(model, solver, pipeline, params, worlds, scene, truth, gt, audit, timing)
                if results:
                    metrics, normalized, gates, details = evaluate(worlds, results, thresholds)
            except Exception:
                audit.check("rollout", False, traceback.format_exc()[-3000:])
            untampered("during the rollout")
        untampered("before the result")
        if gc_enabled:
            gc.enable()
    objects = {}
    if scene is not None:
        for name in OBJECTS:
            body = scene.worlds[0]["objects"].get(name)
            distance = scene.worlds[0]["match_distance"].get(name)
            objects[name] = {
                "body": None if body is None else scene.ix.body_leaf[body],
                "start_offset_m": distance,
            }
    result.update(
        success=not audit.failed and not gates and bool(results),
        integrity=audit.checks,
        failed_checks=audit.failed + gates,
        metrics=metrics,
        normalized_worst=normalized,
        details={
            **details,
            "integrity_notes": {name: notes for name, notes in audit.notes.items() if notes},
            "matched": objects,
            "worlds": len(worlds),
            "solver": f"{type(solver).__module__}.{type(solver).__name__}",
            "pipeline": None if pipeline is None else type(pipeline).__name__,
            "params": read_params(module),
            "solver_resync_changed": solver_info.get("resync_changed"),
            "timing": timing,
        },
        deterministic=bool(structural),
        seconds=time.perf_counter() - started,
    )
    return result


# ----------------------------------------------------------------------------- repeats


def _child(script: Path, index: int, full: bool, timeout: float) -> dict:
    """One :func:`verify` run in a fresh process; its result must carry the nonce sent to it before the import."""
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / f"run{index}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(script),
            "--single",
            "--nonce",
            "--output",
            str(output),
        ]
        if full:
            command.append("--full")
        try:
            process = subprocess.run(
                command, input=nonce + "\n", capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"run {index} timed out after {timeout:.0f} s",
                "failed_checks": ["timeout"],
            }
        try:
            result = json.loads(output.read_text())
        except (OSError, ValueError):
            result = None
        if process.returncode != 0 or result is None or result.get("nonce") != nonce:
            error = (process.stderr or process.stdout)[-3000:] or "no result with the run's nonce"
            return {"success": False, "error": error, "failed_checks": ["crash"]}
        return result


GATES = {
    "main_held_min": ("main_held_min", "min"),
    "main_placed_min": ("main_placed_min", "min"),
    "control_grip_open_rise_m": ("control_rise_m_max", "max"),
    "control_low_mu_rise_m": ("control_rise_m_max", "max"),
}


def implausible(result: dict, worlds: int, t: dict) -> list[str]:
    """Why a passing child result does not look like a real run: the parent, which never imports the
    submission, recomputes every gate from the reported metrics and requires every integrity check."""
    problems = []
    details, metrics, integrity = (
        result.get("details") or {},
        result.get("metrics") or {},
        result.get("integrity") or {},
    )
    if details.get("worlds") != worlds:
        problems.append(f"world count {details.get('worlds')} instead of {worlds}")
    missing = REQUIRED_CHECKS - set(integrity)
    if missing or not all(value is True for value in integrity.values()):
        problems.append(f"integrity checks missing or failed: {sorted(missing)[:6]}")
    for key, (limit, kind) in GATES.items():
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"{key} missing")
        elif (value < t[limit]) if kind == "min" else (value > t[limit]):
            problems.append(f"{key} fails")
    for name in OBJECTS:
        held, placed = metrics.get(f"main_{name}_held"), metrics.get(f"main_{name}_placed")
        if not isinstance(held, int) or not isinstance(placed, int):
            problems.append(f"{name} counts missing")
        elif held < t["main_held_min"] or placed < t["main_placed_min"]:
            problems.append(f"{name} counts fail")
    if not isinstance((details.get("timing") or {}).get("run_s"), (int, float)):
        problems.append("no rollout timing")
    return problems


# Checks every real run reports (the MuJoCo, tray-body, and environment checks depend on the submission).
REQUIRED_CHECKS = {
    "actuators",
    "applied_forces",
    "arm_bases",
    "controls",
    "equality_constraints",
    "finger_geometry",
    "gravcomp",
    "gravity",
    "identical_worlds",
    "kinematics",
    "materials",
    "model_type",
    "object_collisions",
    "object_damping",
    "object_drive",
    "object_dynamic",
    "object_gravcomp",
    "object_inertia",
    "object_joint",
    "object_mass",
    "object_match",
    "object_shapes",
    "object_size",
    "object_start",
    "params",
    "perturbations",
    "pipeline_type",
    "runtime",
    "sandbox",
    "solver_type",
    "source",
    "station_shapes",
    "structure",
    "table",
    "tray",
    "untampered",
    "world_count",
}


def verify_repeated(script: Path, runs: int = RUNS, full: bool = False, budget: float = TOTAL_BUDGET_S) -> dict:
    """Run :func:`verify` in fresh processes, once more if they disagree; the majority decides."""
    started, results = time.perf_counter(), []
    # Planned here, in a process that never imports the submission.
    worlds, thresholds = len(plan_worlds()), load_thresholds()
    while len(results) < max(1, runs) or (len({r["success"] for r in results}) > 1 and len(results) < MAX_RUNS):
        remaining = budget - (time.perf_counter() - started)
        if remaining < 60.0:
            break
        results.append(_child(script, len(results), full, min(CHILD_TIMEOUT_S, remaining)))
        problems = implausible(results[-1], worlds, thresholds) if results[-1].get("success") else []
        if problems:
            results[-1] = {
                "success": False,
                "error": f"the run's result does not match a real verification: {problems[:6]}",
                "failed_checks": ["result"],
            }
        if results[-1].get("deterministic"):
            break  # integrity failures do not depend on the run
    success = 2 * sum(bool(r["success"]) for r in results) > len(results)
    chosen = next((r for r in results if bool(r["success"]) == success), {"success": False, "error": "no runs"})
    runs_summary = [
        {
            "success": bool(r["success"]),
            "failed_checks": r.get("failed_checks", []),
            "seconds": r.get("seconds"),
            "error": (r.get("error") or "")[-500:] or None,
        }
        for r in results
    ]
    return {
        **chosen,
        "success": success,
        "integrity": chosen.get("integrity", {}),
        "failed_checks": chosen.get("failed_checks", []),
        "metrics": chosen.get("metrics", {}),
        "normalized_worst": chosen.get("normalized_worst", {}),
        "runs": runs_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single", action="store_true", help="one run in this process, no repeats")
    parser.add_argument("--runs", type=int, default=RUNS, help="fresh-process runs before a tie-break run")
    parser.add_argument("--full", action="store_true", help="roll out even if integrity checks fail")
    parser.add_argument("--nonce", action="store_true", help="read a nonce from stdin and echo it in the result")
    args = parser.parse_args()
    # Read before the submission is imported; kept in this frame, not in a module global.
    nonce = sys.stdin.readline().strip() if args.nonce else None
    wp.config.log_level = wp.LOG_WARNING
    script = args.script.resolve()
    result = verify(script, full=args.full) if args.single else verify_repeated(script, args.runs, args.full)
    if nonce is not None:
        result["nonce"] = nonce
    if args.output:
        args.output.write_text(json.dumps(_scrub(result), indent=2) + "\n")
    summary = {key: result.get(key) for key in ("success", "failed_checks", "metrics")}
    summary["runs"] = [run["success"] for run in result.get("runs", [])]
    print(json.dumps(_scrub(summary)), flush=True)
    if args.single:
        # Skip exit handlers the submission may have registered (they could rewrite the result).
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
