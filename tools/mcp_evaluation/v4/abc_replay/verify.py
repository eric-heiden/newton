# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify a physical replay of real ABC-130k fruit pick-and-place episodes (``abc_replay``).

Loads the submitted ``fruit_replay.py`` in a clean copy of its workspace (this verifier's own
``replay_common.py`` and station MJCF, no recorded video), builds one batched model with the
submission's ``build_model(scenes)``, ``make_solver(model)`` and ``make_pipeline(model)``, and runs
its own replay loop (:class:`replay_common.Replay`) over the whole timeline of every episode:

- the main episode x 8 (2 nominal copies, 6 with jittered fruit starts), gated per fruit,
- six held-out episodes x 4 (1 nominal, 3 jittered), gated on how many fruits are held and placed,
- the four agent-facing sibling episodes x 2 (report only),
- two negative controls on the main episode, 2 copies each: gripper commands forced open, and arm
  and fruit friction forced to 0.02. No fruit may rise more than 1 cm in either.

Before the rollout, :func:`audit_submission` checks that the submission is still a physical model of
the real station: solver and pipeline types, station kinematics and finger geometry, arm bases,
table, gravity, the fruits (free, dynamic, bounded size, mass and inertia, at the scene's start poses,
no drives, damping, gravity compensation or applied forces), the tray envelope, materials, actuators
and equality constraints, what the MuJoCo solver actually simulates, and that nothing in Newton,
Warp, MuJoCo, NumPy or ``replay_common`` was patched. Hidden data and thresholds are read from
``~/.newton-visual-private/abc_replay``.

By default the verification runs twice in fresh processes (a third time if they disagree), and the
majority decides.
"""

from __future__ import annotations

import _thread
import argparse
import builtins
import copy
import gc
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The verifier's own replay_common, under a name of its own: the submission imports a separate copy.
rc = _load_module("abc_replay_verifier_common", HERE / "replay_common.py")

PRIVATE = Path(os.environ.get("NEWTON_VISUAL_PRIVATE", Path.home() / ".newton-visual-private")) / "abc_replay"

# Gates, calibrated by calibrate.py (rule in its docstring) from fresh-process runs of the reference and the
# starter. Main-episode values are per fruit: copies (of 8) or the median over copies. PRIVATE/thresholds.json
# (written by calibrate.py --write) overrides them (same keys).
DEFAULT_THRESHOLDS = {
    "main_held_min": 7,  # copies holding the fruit through the carry
    "main_placed_min": 6,  # copies with the fruit resting in the tray at the end
    "lifted_fraction_min": 0.9,
    "carry_track_err_m_max": 0.0343,  # [m] xy, against the FK-attached carry track
    "liftoff_err_rows_max": 3,  # [30 Hz state samples]
    "release_err_s_min": -0.10,  # [s]
    "release_err_s_max": 0.10,  # [s]
    "moved_before_grasp_m_max": 0.02,  # [m]
    "arm_rmse_rad_max": 0.0261,  # [rad] per arm, whole episode
    "grip_gap_err_mm_max": 5.0,  # [mm]
    "final_xy_err_m_max": 0.142,  # [m]
    # Held-out: fraction of (episode, fruit, copy) held and placed over the gated episodes and fruits.
    "heldout_fruit_rate_min": 0.892,
    "heldout_exclude_verdicts": ["unreliable"],  # scene or fruit verdicts reported but not gated
    "heldout_arm_rmse_rad_max": 0.0308,  # [rad] per arm, mean over the held-out episodes
    "heldout_copies_min": 3,  # report: an episode passes with its gated fruits held and placed in this many copies
    "control_rise_m_max": 0.01,  # [m] highest fruit rise in either negative control
}

# Bounds of the physical model (fruits/task_spec_draft.json, corrected: station shapes may keep their MJCF
# materials, e.g. finger pads mu 4; Newton's default contact gap of 0.1 m is allowed).
BOUNDS = {
    "dt": (0.00025, 0.002),
    "command_delay": (0.0, 0.2),
    "fruit_mass_kg": (0.02, 0.15),
    "fruit_size_m": {
        "pear": {"length": (0.065, 0.092), "width": (0.044, 0.058)},
        "orange": {"diameter": (0.052, 0.066)},
        "dark_fruit": {"diameter": (0.060, 0.074)},
    },
    "fruit_shapes_max": 8,
    "fruit_armature": 1e-6,
    "fruit_com_offset_m": 0.02,
    "start_xy_m": 0.002,
    "start_clearance_m": (-0.001, 0.003),  # lowest collision point above the table
    "pear_axis_deg": 5.0,
    "base_m": 0.003,
    "frame_m": 0.0005,  # station joint frames and collision shapes
    "frame_rad": 0.01,
    "table_z_m": 0.002,
    "tray_shapes_max": 40,
    "tray_margin_m": 0.015,
    "tray_top_m": 0.035,
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
}
GRIPPER_RANGE = 0.0495  # finger slide command range [m] (MJCF ctrlrange)
MAIN_COPIES, MAIN_NOMINAL = 8, 2
HELDOUT_COPIES, SIBLING_COPIES, CONTROL_COPIES = 4, 2, 2
CONTROL_MU = 0.02
SEED = 7067
RUNTIME_LIMIT_S = 400.0  # predicted wall time of the batched rollout
RUNS, MAX_RUNS = 2, 3
CHILD_TIMEOUT_S, TOTAL_BUDGET_S = 840.0, 1700.0  # the harness stops the verifier after 1800 s
FINGER_BODIES = {
    f"{side}_{name}"
    for side in rc.SIDES
    for name in ("link_left_finger", "lf_rot", "lf_down", "link_right_finger", "rf_rot", "rf_down")
}
DRIVEN_JOINTS = {f"{side}_joint{k}" for side in rc.SIDES for k in range(1, 7)} | {
    f"{side}_{finger}" for side in rc.SIDES for finger in ("left_finger", "right_finger")
}
FINGER_PAIRS = {(f"{side}_left_finger", f"{side}_right_finger") for side in rc.SIDES}
BASE_JOINTS = {"left_arm_joint": "left", "right_arm_joint": "right"}
ARM_PREFIXES = tuple(f"{side}_" for side in rc.SIDES)
SOLVERS = tuple(
    getattr(newton.solvers, name)
    for name in ("SolverMuJoCo", "SolverXPBD", "SolverFeatherstone", "SolverSemiImplicit", "SolverKamino", "SolverVBD")
    if hasattr(newton.solvers, name)
)
# Modules whose functions and classes the submission must not replace.
WATCHED_ROOTS = {"newton", "warp", "mujoco", "mujoco_warp", "numpy", "math", "json", "copy", rc.__name__}
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


def prepare_workspace(script: Path, work: Path) -> Path:
    """Copy the submission's workspace without recorded data, adding this verifier's fixed files."""
    skip = {"frames", "arm_logs", "station", "episodes", "scenes", "gt", "__pycache__", ".git"}
    skip |= {"replay_common.py", "video_reference.npz", "camera.json"}
    for item in script.parent.iterdir():
        if item.name in skip or item.suffix in (".jpg", ".png", ".mp4"):
            continue
        if item.is_dir():
            shutil.copytree(item, work / item.name, ignore=shutil.ignore_patterns("__pycache__", "*.jpg", "*.png"))
        elif item.is_file() and item.stat().st_size < 100_000_000:
            shutil.copy2(item, work / item.name)
    shutil.copy2(HERE / "replay_common.py", work / "replay_common.py")
    for folder in ("station", "episodes", "scenes"):
        shutil.copytree(PRIVATE / folder, work / folder)
    shutil.copy2(PRIVATE / "camera.json", work / "camera.json")
    return work / script.name


PRIVATE_SCENE_KEYS = ("episode", "uuid", "source", "notes", "verdict")


def public_scene(spec: dict) -> dict:
    """The copy of a world's scene that build_model gets: geometry, starts, sizes, and events, without data paths,
    episode ids, or reliability verdicts (held-out episodes are named by group)."""
    scene = copy.deepcopy(spec["scene"])
    for key in PRIVATE_SCENE_KEYS:
        scene.pop(key, None)
    for fruit in scene.get("fruits", {}).values():
        for key in PRIVATE_SCENE_KEYS:
            fruit.pop(key, None)
    if spec["group"] == "heldout":
        scene["name"] = "heldout"
    return scene


def load(path: Path):
    # The submission imports its own copy of replay_common (the verifier's file, copied into its workspace).
    sys.modules.pop("replay_common", None)
    sys.path.insert(0, str(path.parent))
    return _load_module("submitted_fruit_replay", path)


def plan_worlds() -> list[dict]:
    """The batched worlds, each with its group, episode, copy, (jittered) scene, episode data, and ground truth."""
    worlds = []

    def add(group, name, scene, data, gt, copies, nominal, rng, control=None):
        for index in range(copies):
            jittered = scene if index < nominal else rc.jitter_scene(scene, rng)
            worlds.append(
                {
                    "group": group,
                    "episode": name,
                    "copy": index,
                    "scene": jittered,
                    "data": data,
                    "gt": gt,
                    "control": control,
                    "verdict": scene.get("verdict", "usable"),
                }
            )

    scene = rc.load_scene(PRIVATE / "scenes" / "main.json")
    episode = rc.load_episode(PRIVATE / "episodes" / "main.npz")
    gt = rc.load_gt(PRIVATE / "gt" / "main.npz")
    add("main", "main", scene, episode, gt, MAIN_COPIES, MAIN_NOMINAL, np.random.default_rng([SEED, 0]))
    heldout = PRIVATE / "heldout"
    for index, path in enumerate(sorted((heldout / "scenes").glob("*.json"))):
        data = rc.load_episode(heldout / "episodes" / f"{path.stem}.npz")
        truth = rc.load_gt(heldout / "gt" / f"{path.stem}.npz")
        rng = np.random.default_rng([SEED, 1, index])
        add("heldout", path.stem, rc.load_scene(path), data, truth, HELDOUT_COPIES, 1, rng)
    for index, path in enumerate(sorted((PRIVATE / "scenes").glob("sib_*.json"))):
        data = rc.load_episode(PRIVATE / "episodes" / f"{path.stem}.npz")
        truth = rc.load_gt(PRIVATE / "gt" / f"{path.stem}.npz")
        rng = np.random.default_rng([SEED, 2, index])
        add("sibling", path.stem, rc.load_scene(path), data, truth, SIBLING_COPIES, 1, rng)
    opened = dict(episode)
    for side in rc.SIDES:
        opened[f"{side}_grip_cmd"] = np.ones_like(episode[f"{side}_grip_cmd"])
    add("control", "grip_open", scene, opened, gt, CONTROL_COPIES, CONTROL_COPIES, None, control="grip_open")
    add("control", "low_mu", scene, episode, gt, CONTROL_COPIES, CONTROL_COPIES, None, control="low_mu")
    return worlds


# ----------------------------------------------------------------------------- tamper detection


def _is_code(value) -> bool:
    return callable(value) or isinstance(value, (staticmethod, classmethod, property))


def snapshot() -> dict:
    """Functions and classes (and their methods) of the watched modules, this verifier, and builtins, by
    identity; and copies of the plain-data constants of this verifier and its replay_common."""
    out = {}
    verifier = sys.modules[__name__]
    for module in (verifier, rc):
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
                raise RuntimeError(f"{event} is not allowed in abc_replay submissions")
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
    "exec",
    "eval",
    "compile",
    "__import__",
    "nonce",
}


def source_findings(work: Path) -> list[str]:
    """Introspection in the submitted Python sources (everything but this verifier's replay_common copy)."""
    import ast  # noqa: PLC0415

    findings = []
    for path in sorted(work.rglob("*.py")):
        if path == work / "replay_common.py":
            continue
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
                bad = [node.id] if node.id in SUSPICIOUS_NAMES else []
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                bad = [name for name in SUSPICIOUS_NAMES if len(name) > 4 and name in node.value]
            else:
                bad = []
            findings += [f"{path.name}:{getattr(node, 'lineno', '?')} {name}" for name in bad]
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
    return points @ rc.quat_to_matrix(pose[3:7]).T + pose[:3]


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose a * b (7-vectors p, q xyzw)."""
    q = rc._quat_multiply(np.asarray(a[3:7]), np.asarray(b[3:7]))
    return np.concatenate([a[:3] + rc.quat_to_matrix(a[3:7]) @ b[:3], q])


def _rotation_close(qa: np.ndarray, qb: np.ndarray) -> bool:
    return float(np.abs(rc.quat_to_matrix(qa) - rc.quat_to_matrix(qb)).max()) <= BOUNDS["frame_rad"]


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


def audit_submission(model, solver, pipeline, params: dict, worlds: list[dict], station: Station, audit: Audit) -> None:
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
    if pipeline is not None:
        audit.check("pipeline_type", getattr(pipeline, "model", None) is model, "the pipeline is for another model")
    if not audit.check("world_count", model.world_count == len(worlds), f"{model.world_count} worlds"):
        return

    sub = Submission(model)
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
    for w, spec in enumerate(worlds):
        roles = _audit_world(sub, station, w, spec["scene"], audit)
        station_shapes.update(roles["station"])
        for name in rc.FRUITS:
            _audit_fruit(sub, w, spec["scene"], name, roles, audit)
        # Fruits collide with the fingers, the table, the tray, and each other.
        required = []
        for i, name in enumerate(rc.FRUITS):
            for s in roles["fruits"][name]:
                required += [(s, other) for other in roles["pads"] + roles["table"] + roles["tray"]]
                for other in rc.FRUITS[i + 1 :]:
                    required += [(s, o) for o in roles["fruits"][other]]
        blocked = sub.blocked(required)
        names = ", ".join(f"{sub.ix.shape_leaf[a]}/{sub.ix.shape_leaf[b]}" for a, b in blocked[:3])
        audit.check("fruit_collisions", not blocked, f"world {w}: {len(blocked)} required pairs never collide: {names}")
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
        audit_mujoco(sub, solver, audit)


def _audit_world(sub: Submission, station: Station, w: int, scene: dict, audit: Audit) -> dict:
    """Bodies and joints, station kinematics and bases, station collision shapes, table, and tray of one world."""
    ix, table_z = sub.ix, float(scene["table_z"])
    bodies = ix.bodies(w)
    for leaf in [*station.bodies, *rc.FRUITS]:
        found = len(bodies.get(leaf, []))
        audit.check("structure", found == 1, f"world {w}: {found} bodies named {leaf}")
    extra = [leaf for leaf in bodies if leaf not in station.bodies and leaf not in rc.FRUITS]
    trays = [b for leaf in extra if leaf.startswith("tray") for b in bodies[leaf]]
    others = [leaf for leaf in extra if not leaf.startswith("tray")]
    audit.check("structure", not others and len(trays) <= 1, f"world {w}: extra bodies {extra[:4]}")
    joints, expected = int(np.count_nonzero(ix.joint_world == w)), len(station.joints) + len(rc.FRUITS) + len(trays)
    audit.check("structure", joints == expected, f"world {w}: {joints} joints, expected {expected}")

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
            base = np.asarray(scene["bases"][side], dtype=np.float64)
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

    # Collision shapes: the station's as in the MJCF (all finger shapes present), the rest fruit or tray.
    station_of = {bodies[leaf][0]: leaf for leaf in station.bodies if len(bodies.get(leaf, [])) == 1}
    fruit_of = {bodies[name][0]: name for name in rc.FRUITS if len(bodies.get(name, [])) == 1}
    roles = {"station": {}, "fruits": {name: [] for name in rc.FRUITS}, "pads": [], "table": [], "tray": []}
    for s in ix.colliders(w):
        body = int(ix.shape_body[s])
        key = (station_of.get(body), ix.shape_leaf[s])
        if body in fruit_of:
            roles["fruits"][fruit_of[body]].append(s)
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
        elif body in station_of:
            audit.check("station_shapes", False, f"world {w}: added collision shape {key[1]} on {key[0]}")
        else:
            roles["tray"].append(s)
    present = {(station_of.get(int(ix.shape_body[s])), ix.shape_leaf[s]) for s in roles["pads"]}
    missing = [key[1] for key in station.shapes if key[0] in FINGER_BODIES and key not in present]
    audit.check("finger_geometry", not missing, f"world {w}: finger collision shapes missing: {missing[:4]}")
    if audit.check("table", len(roles["table"]) == 1, f"world {w}: {len(roles['table'])} table planes"):
        t = roles["table"][0]
        up = rc.quat_to_matrix(sub.transform[t][3:7])[2, 2]
        ok = int(sub.kind[t]) == GEO.PLANE and abs(sub.transform[t][2] - table_z) <= BOUNDS["table_z_m"] and up > 0.999
        audit.check("table", ok, f"world {w}: table plane at z {sub.transform[t][2]:.4f}")

    # Tray: every added shape inside the scene's tray envelope.
    tray_count = len(roles["tray"])
    audit.check("tray", tray_count <= BOUNDS["tray_shapes_max"], f"world {w}: {tray_count} added shapes")
    for s in roles["tray"]:
        points = sub.world_points(s)
        if points is None:
            audit.check("tray", False, f"world {w}: unbounded added shape {ix.shape_leaf[s]}")
            continue
        stride = max(1, len(points) // 400)
        outside = max(rc.sector_distance(p[:2], scene["tray"]) for p in points[::stride])
        top = float(points[:, 2].max() - table_z)
        ok = outside <= BOUNDS["tray_margin_m"] and top <= BOUNDS["tray_top_m"]
        message = f"{1000 * outside:.0f} mm outside the tray, top {1000 * top:.0f} mm above the table"
        audit.check("tray", ok, f"world {w}: {ix.shape_leaf[s]} {message}")
    for body in trays:
        low, high = BOUNDS["tray_mass_kg"]
        audit.check("tray", low <= sub.mass[body] <= high, f"world {w}: tray body mass {sub.mass[body]:.3f} kg")
        if sub.gravcomp is not None:
            audit.check("tray", sub.gravcomp[body] == 0.0, f"world {w}: tray gravity compensation")
    roles["bodies"] = fruit_of
    return roles


def _audit_fruit(sub: Submission, w: int, scene: dict, name: str, roles: dict, audit: Audit) -> None:
    """A fruit: a free, dynamic rigid body without drives, within its size, mass, and inertia bounds, at its start."""
    body = next((b for b, n in roles["bodies"].items() if n == name), None)
    if body is None:
        return
    ix, start = sub.ix, scene["fruits"][name]["start"]
    tag = f"world {w}: {name}"
    into = ix.joints_into(body)
    free = len(into) == 1 and int(ix.joint_type[into[0]]) == newton.JointType.FREE and ix.joint_parent[into[0]] < 0
    audit.check("fruit_joint", free, f"{tag} is not a free root body")
    audit.check("fruit_joint", not np.any(ix.joint_parent == body), f"{tag} has bodies attached")
    if sub.body_flags is not None:
        ok = int(sub.body_flags[body]) == int(newton.BodyFlags.DYNAMIC)
        audit.check("fruit_dynamic", ok, f"{tag} body flags {sub.body_flags[body]}")
    mass = float(sub.mass[body])
    low, high = BOUNDS["fruit_mass_kg"]
    audit.check("fruit_mass", low <= mass <= high, f"{tag} mass {mass:.4f} kg")
    audit.check("fruit_mass", mass > 0 and abs(sub.inv_mass[body] * mass - 1.0) < 1e-3, f"{tag} inverse mass")
    if sub.gravcomp is not None:
        audit.check("fruit_gravcomp", sub.gravcomp[body] == 0.0, f"{tag} gravity compensation {sub.gravcomp[body]}")
    if free:
        j = into[0]
        audit.check("fruit_joint", bool(sub.enabled[j]), f"{tag} joint disabled")
        for d in ix.dofs(j):
            drive = int(sub.target_mode[d]) == int(newton.JointTargetMode.NONE) and sub.ke[d] == 0 and sub.kd[d] == 0
            if sub.passive is not None:
                drive = drive and sub.passive[d] == 0.0
            audit.check("fruit_drive", drive, f"{tag} dof {d - ix.qd_start[j]} has a drive or spring")
            ok = sub.damping[d] == 0 and sub.friction[d] == 0 and sub.armature[d] <= BOUNDS["fruit_armature"]
            audit.check("fruit_damping", ok, f"{tag} dof {d - ix.qd_start[j]} has damping, friction, or armature")

    shapes = roles["fruits"][name]
    if not audit.check("fruit_shapes", 1 <= len(shapes) <= BOUNDS["fruit_shapes_max"], f"{tag}: {len(shapes)} shapes"):
        return
    local = [shape_points(sub.model, s) for s in shapes]
    if not audit.check("fruit_shapes", all(p is not None for p in local), f"{tag} has an unbounded shape"):
        return
    local = np.vstack([_apply(sub.transform[s], p) for s, p in zip(shapes, local, strict=True)])
    centre, extents, axes = _principal(local)
    sizes = BOUNDS["fruit_size_m"][name]
    if "diameter" in sizes:
        fits = all(sizes["diameter"][0] <= e <= sizes["diameter"][1] for e in extents)
    else:
        fits = sizes["length"][0] <= extents[0] <= sizes["length"][1]
        fits = fits and all(sizes["width"][0] <= e <= sizes["width"][1] for e in extents[1:])
    audit.check("fruit_size", fits, f"{tag} extents {np.round(1000 * extents, 1).tolist()} mm")

    # Principal moments between 0.8 x a solid and 1.2 x a thin-shell ellipsoid of the fruit's extents.
    a, b, c = extents / 2
    solid = np.sort(mass * np.array([b * b + c * c, a * a + c * c, a * a + b * b]) / 5.0)
    matrix = sub.inertia[body].astype(np.float64).reshape(3, 3)
    moments = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    ok = bool(np.all(moments >= 0.8 * solid - 1e-12) and np.all(moments <= 1.2 * solid * 5.0 / 3.0 + 1e-12))
    audit.check("fruit_inertia", ok, f"{tag} principal moments {np.round(moments, 8).tolist()}")
    ok = np.allclose(sub.inv_inertia[body].reshape(3, 3).astype(np.float64) @ matrix, np.eye(3), atol=1e-2)
    audit.check("fruit_inertia", ok, f"{tag} inverse inertia")
    offset = float(np.linalg.norm(sub.com[body] - centre))
    audit.check("fruit_inertia", offset <= BOUNDS["fruit_com_offset_m"], f"{tag} centre of mass {offset:.3f} m off")

    pose = sub.body_q[body]
    world_centre = _apply(pose, centre[None])[0]
    offset = float(np.linalg.norm(world_centre[:2] - np.asarray(start["pos"][:2])))
    audit.check("fruit_start", offset <= BOUNDS["start_xy_m"], f"{tag} starts {1000 * offset:.1f} mm from the scene")
    clearance = float(_apply(pose, local)[:, 2].min() - scene["table_z"])
    low, high = BOUNDS["start_clearance_m"]
    message = f"{tag} lowest point {1000 * clearance:.1f} mm above the table"
    audit.check("fruit_start", low <= clearance <= high, message)
    if "length" in sizes:
        # Only the heading of the long axis is fixed; a pear may lie with its neck lower than its bulb.
        long_axis = (rc.quat_to_matrix(pose[3:7]) @ axes[:, 0])[:2]
        wanted = rc.quat_to_matrix(np.asarray(start["quat_xyzw"], dtype=np.float64))[:2, 0]
        cosine = abs(float(long_axis @ wanted)) / max(float(np.linalg.norm(long_axis) * np.linalg.norm(wanted)), 1e-12)
        angle = math.degrees(math.acos(min(1.0, cosine)))
        message = f"{tag} long axis heading {angle:.1f} deg off the scene"
        audit.check("fruit_start", angle <= BOUNDS["pear_axis_deg"], message)


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


def audit_mujoco(sub: Submission, solver, audit: Audit) -> None:
    """What SolverMuJoCo actually simulates: actuators, equalities, options, fruit bodies and dofs, geom friction."""
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

    fruits = {b for b, leaf in enumerate(ix.body_leaf) if leaf in rc.FRUITS}
    mass, inertia = _per_world(mw.body_mass, worlds, 1), _per_world(mw.body_inertia, worlds, 2)
    gravcomp = _per_world(mw.body_gravcomp, worlds, 1)
    armature, damping = _per_world(mw.dof_armature, worlds, 1), _per_world(mw.dof_damping, worlds, 1)
    frictionloss, stiffness = _per_world(mw.dof_frictionloss, worlds, 1), _per_world(mw.jnt_stiffness, worlds, 1)
    for w in range(worlds):
        for b in range(mj.nbody):
            body = int(to_body[w, b])
            if body not in fruits:
                continue
            tag = f"world {w}: solver's {ix.body_leaf[body]}"
            audit.check("mujoco", gravcomp[w, b] == 0.0, f"{tag} has gravity compensation")
            moments = np.sort(np.linalg.eigvalsh(sub.inertia[body].reshape(3, 3).astype(np.float64)))
            same = abs(mass[w, b] - sub.mass[body]) <= 1e-4 * sub.mass[body]
            same = same and np.allclose(np.sort(inertia[w, b]), moments, rtol=1e-2, atol=1e-10)
            audit.check("mujoco", same, f"{tag} mass or inertia differs from the model")
            for d in np.flatnonzero(np.asarray(mj.dof_bodyid) == b):
                ok = damping[w, d] == 0 and frictionloss[w, d] == 0 and armature[w, d] <= BOUNDS["fruit_armature"]
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


# ----------------------------------------------------------------------------- rollout and scoring


def apply_controls(model: newton.Model, worlds: list[dict]) -> list[int]:
    """Set the friction of all arm and fruit shapes in the low-friction control worlds; returns those shapes."""
    ix = Index(model)
    shapes = [
        s
        for w, spec in enumerate(worlds)
        if spec["control"] == "low_mu"
        for s in ix.colliders(w)
        if ix.shape_body[s] >= 0
        and (ix.body_leaf[ix.shape_body[s]].startswith(ARM_PREFIXES) or ix.body_leaf[ix.shape_body[s]] in rc.FRUITS)
    ]
    mu = model.shape_material_mu.numpy()
    mu[shapes] = CONTROL_MU
    model.shape_material_mu.assign(mu)
    return shapes


def _median(values: list, absolute: bool = False, missing: float = math.inf) -> float:
    """Median over copies; ``None``/non-finite values count as ``missing`` (a failure for upper bounds)."""
    values = [missing if v is None or not np.isfinite(v) else (abs(v) if absolute else v) for v in values]
    return float(np.median(values)) if values else missing


def _counts(rows: list[dict]) -> dict:
    return {
        "copies": len(rows),
        "ok": sum(bool(r["all_held"] and r["all_placed"]) for r in rows),
        "fruits": {
            name: {key: sum(bool(r["fruits"][name][key]) for r in rows) for key in ("held", "placed")}
            for name in rc.FRUITS
        },
    }


def evaluate(worlds: list[dict], results: list[dict], t: dict) -> tuple[dict, dict, list[str], dict]:
    """Ensemble gates: metrics, normalized worst values (> 1 fails), failed gates, and details."""
    metrics, normalized, failed, details = {}, {}, [], {}

    def gate(key: str, value: float, limit: float, kind: str) -> None:
        metrics[key] = value
        if kind == "min":
            normalized[key] = limit / value if value > 0 else math.inf
            ok = value >= limit
        else:
            normalized[key] = value / limit if limit > 0 else math.inf
            ok = value <= limit
        if not ok:
            failed.append(key)

    groups = {}
    for spec, result in zip(worlds, results, strict=True):
        groups.setdefault(spec["group"], {}).setdefault(spec["episode"], []).append(result)

    main = groups["main"]["main"]
    fruits = {}
    for name in rc.FRUITS:
        rows = [r["fruits"][name] for r in main]
        fruits[name] = {
            "held": sum(m["held"] for m in rows),
            "placed": sum(m["placed"] for m in rows),
            "lifted_fraction": _median([m["lifted_fraction"] for m in rows], missing=0.0),
            "carry_track_err_m": _median([m["carry_track_err_m"] for m in rows]),
            "liftoff_err_rows": _median([m["liftoff_err_rows"] for m in rows], absolute=True),
            "release_err_s": _median([m["release_err_s"] for m in rows]),
            "moved_before_grasp_m": _median([m["moved_before_grasp_m"] for m in rows]),
            "grip_gap_err_mm": _median([m["grip_gap_err_mm"] for m in rows], absolute=True),
            "final_xy_err_m": _median([m["final_xy_err_m"] for m in rows]),
            "slip_m": _median([m["slip_m"] for m in rows]),
            "carry_track_err_3d_m": _median([m["carry_track_err_3d_m"] for m in rows]),
            "image_err_px_post": _median([m["image_err_px_post"] for m in rows], missing=math.nan),
        }
    arm = {side: _median([r["arm_rmse_rad"][side] for r in main]) for side in rc.SIDES}
    details["main"] = {"fruits": fruits, "arm_rmse_rad": arm, **_counts(main)}
    values = fruits.values()
    gate("main_held_min", min(f["held"] for f in values), t["main_held_min"], "min")
    gate("main_placed_min", min(f["placed"] for f in values), t["main_placed_min"], "min")
    gate("main_lifted_fraction_min", min(f["lifted_fraction"] for f in values), t["lifted_fraction_min"], "min")
    gate("main_carry_track_err_m_max", max(f["carry_track_err_m"] for f in values), t["carry_track_err_m_max"], "max")
    gate("main_liftoff_err_rows_max", max(f["liftoff_err_rows"] for f in values), t["liftoff_err_rows_max"], "max")
    release = [f["release_err_s"] for f in values]
    metrics["main_release_err_s_min"], metrics["main_release_err_s_max"] = min(release), max(release)
    low, high = t["release_err_s_min"], t["release_err_s_max"]
    normalized["main_release_err_s"] = max(v / high if v >= 0 else v / low for v in release)
    if not all(low <= v <= high for v in release):
        failed.append("main_release_err_s")
    moved = max(f["moved_before_grasp_m"] for f in values)
    gate("main_moved_before_grasp_m_max", moved, t["moved_before_grasp_m_max"], "max")
    gate("main_arm_rmse_rad_max", max(arm.values()), t["arm_rmse_rad_max"], "max")
    gate("main_grip_gap_err_mm_max", max(f["grip_gap_err_mm"] for f in values), t["grip_gap_err_mm_max"], "max")
    gate("main_final_xy_err_m_max", max(f["final_xy_err_m"] for f in values), t["final_xy_err_m_max"], "max")
    for name, f in fruits.items():
        for key in ("held", "placed", "carry_track_err_m", "final_xy_err_m", "image_err_px_post"):
            metrics[f"main_{name}_{key}"] = f[key]

    # Held-out: fruits held and placed, pooled over the gated (episode, fruit) pairs and their copies. A scene
    # verdict excludes a whole episode, a fruit verdict one fruit (its real grasp is not reproducible from the
    # data, see heldout_spec.json); arm tracking is gated on every held-out episode.
    heldout, excluded = {}, set(t["heldout_exclude_verdicts"])
    for episode, rows in groups.get("heldout", {}).items():
        scene = next(s["scene"] for s in worlds if s["episode"] == episode)
        verdict = scene.get("verdict", "usable")
        names = [
            name
            for name in rc.FRUITS
            if verdict not in excluded and scene["fruits"][name].get("verdict", "usable") not in excluded
        ]
        entry = {"verdict": verdict, **_counts(rows), "gated_fruits": names}
        ok = sum(all(r["fruits"][n]["held"] and r["fruits"][n]["placed"] for n in names) for r in rows)
        entry["ok_gated"] = ok if names else None
        entry["pass"] = bool(names) and ok >= t["heldout_copies_min"]
        entry["arm_rmse_rad"] = {side: _median([r["arm_rmse_rad"][side] for r in rows]) for side in rc.SIDES}
        entry["gated"] = bool(names)
        heldout[episode] = entry
    details["heldout"] = heldout
    gated = [e for e in heldout.values() if e["gated"]]
    if gated:
        both = sum(
            sum(
                bool(r["fruits"][name]["held"] and r["fruits"][name]["placed"])
                for name in heldout[episode]["gated_fruits"]
            )
            for episode, rows in groups["heldout"].items()
            for r in rows
        )
        total = sum(e["copies"] * len(e["gated_fruits"]) for e in gated)
        gate("heldout_fruit_rate", both / total, t["heldout_fruit_rate_min"], "min")
        metrics["heldout_fruit_count"] = both
        metrics["heldout_fruit_total"] = total
        metrics["heldout_episodes_passed"] = sum(e["pass"] for e in gated)
        metrics["heldout_episodes_gated"] = len(gated)
    if heldout:
        rmse = max(float(np.mean([e["arm_rmse_rad"][side] for e in heldout.values()])) for side in rc.SIDES)
        gate("heldout_arm_rmse_rad_max", rmse, t["heldout_arm_rmse_rad_max"], "max")

    siblings = {episode: _counts(rows) for episode, rows in groups.get("sibling", {}).items()}
    details["siblings"] = siblings
    metrics["siblings_ok_copies"] = sum(e["ok"] for e in siblings.values())
    metrics["siblings_copies"] = sum(e["copies"] for e in siblings.values())

    # A diverged control world (non-finite state) counts as a failed control.
    controls = {}
    for control, rows in groups.get("control", {}).items():
        rises = {name: [r["fruits"][name]["max_rise_m"] for r in rows] for name in rc.FRUITS}
        diverged = sum(not all(np.isfinite(rises[name][k]) for name in rc.FRUITS) for k in range(len(rows)))
        controls[control] = {"max_rise_m": rises, "diverged_copies": diverged}
        rise = max(v if np.isfinite(v) else math.inf for values in rises.values() for v in values)
        gate(f"control_{control}_rise_m", rise, t["control_rise_m_max"], "max")
    details["controls"] = controls
    metrics["diverged_worlds"] = sum(
        not all(np.isfinite(r["fruits"][name]["max_rise_m"]) for name in rc.FRUITS) for r in results
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


def rollout(model, solver, pipeline, params: dict, worlds: list[dict], audit: Audit, timing: dict) -> list[dict]:
    """The verifier's own replay of every world, scored with :func:`replay_common.score`."""
    episodes = [spec["data"] for spec in worlds]
    replay = rc.Replay(model, solver, pipeline, episodes, dt=params["dt"], command_delay=params["command_delay"])
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
    replay.step(total - probe)
    wp.synchronize_device(model.device)
    timing["run_s"] = time.perf_counter() - tic
    camera = rc.load_camera(PRIVATE / "camera.json")
    return [
        rc.score(recording, spec["scene"], spec["gt"], camera if spec["group"] == "main" else None)
        for recording, spec in zip(replay.recordings(), worlds, strict=True)
    ]


def verify(script: Path, full: bool = False) -> dict:
    """One verification run in this process (the submission is imported here)."""
    started, timing = time.perf_counter(), {}
    thresholds = load_thresholds()
    result = {
        "task": "abc_replay",
        "success": False,
        "thresholds": thresholds,
        "thresholds_source": thresholds_source(),
        "metrics": {},
        "normalized_worst": {},
    }
    newton.solvers.SolverMuJoCo.import_mujoco()
    station = Station()
    worlds = plan_worlds()
    threads, hooks = set(threading.enumerate()), _hooks()
    audit = Audit()
    with tempfile.TemporaryDirectory() as tmp:
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
        try:
            module = load(path)
            tic = time.perf_counter()
            # Copies: the audit and the scores use the verifier's scenes, whatever build_model does to its own.
            model = module.build_model([public_scene(spec) for spec in worlds])
            timing["build_s"] = time.perf_counter() - tic
            control_shapes = apply_controls(model, worlds) if isinstance(model, newton.Model) else []
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
            audit_submission(model, solver, pipeline, params, worlds, station, audit)
            mu = model.shape_material_mu.numpy()[control_shapes]
            audit.check("controls", bool(np.all(mu == CONTROL_MU)), "the control worlds' friction was changed")
        except Exception:
            audit.check("audit", False, traceback.format_exc()[-2000:])
        untampered("after build")
        timing["audit_s"] = time.perf_counter() - tic
        structural = audit.failed
        metrics, normalized, gates, details, results = {}, {}, [], {}, []
        if not structural or full:
            try:
                results = rollout(model, solver, pipeline, params, worlds, audit, timing)
                if results:
                    metrics, normalized, gates, details = evaluate(worlds, results, thresholds)
            except Exception:
                audit.check("rollout", False, traceback.format_exc()[-3000:])
            untampered("during the rollout")
        untampered("before the result")
        if gc_enabled:
            gc.enable()
    result.update(
        success=not audit.failed and not gates and bool(results),
        integrity=audit.checks,
        failed_checks=audit.failed + gates,
        metrics=metrics,
        normalized_worst=normalized,
        details={
            **details,
            "integrity_notes": {name: notes for name, notes in audit.notes.items() if notes},
            "worlds": len(worlds),
            "solver": f"{type(solver).__module__}.{type(solver).__name__}",
            "pipeline": None if pipeline is None else type(pipeline).__name__,
            "params": read_params(module),
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
    "main_lifted_fraction_min": ("lifted_fraction_min", "min"),
    "main_carry_track_err_m_max": ("carry_track_err_m_max", "max"),
    "main_liftoff_err_rows_max": ("liftoff_err_rows_max", "max"),
    "main_moved_before_grasp_m_max": ("moved_before_grasp_m_max", "max"),
    "main_arm_rmse_rad_max": ("arm_rmse_rad_max", "max"),
    "main_grip_gap_err_mm_max": ("grip_gap_err_mm_max", "max"),
    "main_final_xy_err_m_max": ("final_xy_err_m_max", "max"),
    "heldout_fruit_rate": ("heldout_fruit_rate_min", "min"),
    "heldout_arm_rmse_rad_max": ("heldout_arm_rmse_rad_max", "max"),
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
    low, high = metrics.get("main_release_err_s_min"), metrics.get("main_release_err_s_max")
    if not all(isinstance(v, (int, float)) for v in (low, high)) or not (
        t["release_err_s_min"] <= low <= high <= t["release_err_s_max"]
    ):
        problems.append("main_release_err_s fails")
    if not isinstance((details.get("timing") or {}).get("run_s"), (int, float)):
        problems.append("no rollout timing")
    return problems


# Checks every real run reports (the MuJoCo, tray, and environment checks depend on the submission's choices).
REQUIRED_CHECKS = {
    "actuators",
    "applied_forces",
    "arm_bases",
    "controls",
    "equality_constraints",
    "finger_geometry",
    "fruit_collisions",
    "fruit_damping",
    "fruit_drive",
    "fruit_dynamic",
    "fruit_gravcomp",
    "fruit_inertia",
    "fruit_joint",
    "fruit_mass",
    "fruit_shapes",
    "fruit_size",
    "fruit_start",
    "gravcomp",
    "gravity",
    "kinematics",
    "materials",
    "model_type",
    "params",
    "pipeline_type",
    "runtime",
    "sandbox",
    "solver_type",
    "source",
    "station_shapes",
    "structure",
    "table",
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
