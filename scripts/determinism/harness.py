"""Determinism test harness for Newton.

Core abstractions used by every scenario:

- :class:`SolverSpec`: lookup table from CLI solver name to a factory that
  produces a concrete solver bound to a Newton :class:`~newton.Model`.
- :class:`ScenarioArgs`: parsed CLI state that scenarios may read (world
  count, num steps, seed, viewer, solver spec).
- :class:`Scenario`: abstract base class. Subclasses implement
  :meth:`Scenario.build` (model + solver) and, optionally,
  :meth:`Scenario.per_step` (custom control input, contact logging, etc.)
  and :meth:`Scenario.extra_snapshot` (scenario-specific telemetry).
- :class:`ScenarioSnapshot`: result of a run. ``core`` is the canonical
  determinism fingerprint; rigid-body scenarios populate it with body and
  joint state, while micro and diffsim scenarios populate it with their
  workload outputs and gradients. ``extras`` is free-form diagnostics.
- :func:`compare_runs`: launches N independent subprocess invocations of
  ``run_determinism.py`` with identical args, loads their snapshots, and
  asserts byte-level ``core`` agreement.

All scenarios must:

- Support ``--world-count`` by replicating the sub-model via
  :meth:`ModelBuilder.replicate`.
- Be fully reproducible from a single ``seed`` (no hidden global state).
- Work on both the GL viewer (visualization) and the null viewer
  (headless determinism runs).
"""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.solvers as newton_solvers

# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------


SolverFactory = Callable[..., "newton.solvers.SolverBase"]


@dataclasses.dataclass(frozen=True)
class SolverSpec:
    """Describes how to construct a specific Newton solver for the harness."""

    name: str
    """CLI-visible solver identifier (e.g. ``"xpbd"``)."""
    factory: SolverFactory
    """Callable that takes a :class:`~newton.Model` and returns a solver."""
    needs_articulated: bool = False
    """When True, the harness skips scenarios that don't add articulated bodies."""
    notes: str = ""
    """Free-form note shown in ``--list`` output."""


def _mj_factory(
    model: newton.Model,
    deterministic: str | None = None,
    deterministic_max_records: int = 16,
) -> Any:
    # Sensible defaults; scenarios can construct their own solver if they
    # need scenario-specific tuning. Contact count is sized for stacks of
    # small boxes; bumped as needed per scenario.
    kwargs = {"solver": "cg"}
    if model.joint_dof_count <= 60:
        kwargs["jacobian"] = "dense"

    solver = newton_solvers.SolverMuJoCo(
        model,
        iterations=100,
        ls_iterations=50,
        njmax=200,
        nconmax=300,
        use_mujoco_contacts=False,
        deterministic=deterministic,
        deterministic_max_records=deterministic_max_records,
        **kwargs,
    )
    if getattr(solver, "mjw_model", None) is not None:
        solver.mjw_model.opt.graph_conditional = False
    return solver


def _vbd_factory(
    model: newton.Model,
    deterministic: str | None = None,
    deterministic_max_records: int | None = None,
) -> Any:
    return newton_solvers.SolverVBD(
        model,
        iterations=10,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
        deterministic=deterministic,
        deterministic_max_records=deterministic_max_records,
    )


def _xpbd_factory(
    model: newton.Model,
    deterministic: str | None = None,
    deterministic_max_records: int | None = None,
) -> Any:
    return newton_solvers.SolverXPBD(
        model,
        deterministic=deterministic,
        deterministic_max_records=deterministic_max_records,
    )


def _featherstone_factory(
    model: newton.Model,
    deterministic: str | None = None,
    deterministic_max_records: int | None = None,
) -> Any:
    return newton_solvers.SolverFeatherstone(
        model,
        deterministic=deterministic,
        deterministic_max_records=deterministic_max_records,
    )


def _semi_implicit_factory(
    model: newton.Model,
    deterministic: str | None = None,
    deterministic_max_records: int | None = None,
) -> Any:
    return newton_solvers.SolverSemiImplicit(
        model,
        deterministic=deterministic,
        deterministic_max_records=deterministic_max_records,
    )


SOLVER_SPECS: dict[str, SolverSpec] = {
    "xpbd": SolverSpec(
        name="xpbd",
        factory=_xpbd_factory,
        notes="Position-based dynamics; fast rigid contacts.",
    ),
    "featherstone": SolverSpec(
        name="featherstone",
        factory=_featherstone_factory,
        needs_articulated=True,
        notes="Featherstone articulated-body algorithm; requires joints.",
    ),
    "semi_implicit": SolverSpec(
        name="semi_implicit",
        factory=_semi_implicit_factory,
        notes="Explicit symplectic integrator; simple dynamics only.",
    ),
    "vbd": SolverSpec(
        name="vbd",
        factory=_vbd_factory,
        notes="Vertex Block Descent / AVBD solver for particles and simple rigid bodies.",
    ),
    "mujoco": SolverSpec(
        name="mujoco",
        factory=_mj_factory,
        notes="MuJoCo (warp-accelerated); requires cuSolverDx / CUDA >= 12.6.3.",
    ),
}


def list_solvers() -> list[SolverSpec]:
    return list(SOLVER_SPECS.values())


# ---------------------------------------------------------------------------
# Scenario base class
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScenarioArgs:
    """Parsed CLI arguments that every scenario consumes."""

    scenario: str
    solver: SolverSpec
    world_count: int
    num_steps: int
    seed: int
    viewer_name: str
    fps: int
    substeps: int
    collision_pipeline_deterministic: bool = False
    collision_pipeline_warp_deterministic: str | None = None
    solver_deterministic: str | None = None
    solver_deterministic_max_records: int | None = None


@dataclasses.dataclass
class ScenarioSnapshot:
    """Final simulation state produced by a scenario run.

    ``core`` is the determinism fingerprint: every scenario populates it
    with the arrays that define bit-exact success for that workload. Two
    runs are considered deterministic iff their ``core`` bytes match exactly.

    ``extras`` is scenario-specific diagnostics (e.g. contact count history,
    COM trajectory, torque time-series). Extras are reported but *not*
    required to be bit-exact — solvers can differ in telemetry layout.
    """

    core: dict[str, np.ndarray]
    extras: dict[str, Any]
    meta: dict[str, Any]

    def core_bytes(self) -> bytes:
        """Canonical byte layout for hashing the core physics state."""
        parts: list[bytes] = []
        for key in sorted(self.core):
            arr = np.ascontiguousarray(self.core[key])
            parts.append(key.encode() + b"\0" + arr.dtype.str.encode() + b"\0")
            parts.append(np.asarray(arr.shape, dtype=np.int64).tobytes())
            parts.append(arr.tobytes())
        return b"|".join(parts)


class Scenario(ABC):
    """Abstract base for a determinism scenario.

    Implementers override :meth:`build` to create a single-world
    :class:`ModelBuilder`; the harness replicates it across
    ``args.world_count`` worlds, adds the ground plane, and finalizes the
    model. ``build`` must also set :attr:`solver` to an active solver bound
    to ``self.model``.
    """

    id: str = ""
    supported_solvers: tuple[str, ...] = ()

    def __init__(self, args: ScenarioArgs) -> None:
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.model: newton.Model | None = None
        self.collision_pipeline: newton.CollisionPipeline | None = None
        self.solver: Any | None = None
        self.state_0: Any | None = None
        self.state_1: Any | None = None
        self.control: Any | None = None
        self.contacts: Any | None = None
        self.step_index: int = 0
        self.graph: Any | None = None
        self._collision_pipeline_warp_deterministic: str | None = None

    # --- required overrides -------------------------------------------------

    @abstractmethod
    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        """Populate ``builder`` with a *single-world* copy of the scene.

        The harness will call :meth:`ModelBuilder.replicate` afterwards to
        create ``args.world_count`` worlds. Scenarios should not call
        ``finalize()`` or add a ground plane themselves — the harness does
        both after replication.
        """

    # --- optional overrides -------------------------------------------------

    def per_step(self) -> None:
        """Override to apply a scenario-specific control signal each step.

        Called once per physics substep, inside the graph capture when CUDA
        is used. Default is a no-op.
        """
        return None

    def extra_snapshot(self) -> dict[str, Any]:
        """Scenario-specific telemetry merged into the final snapshot."""
        return {}

    # --- harness lifecycle --------------------------------------------------

    def build(self, viewer: Any) -> None:
        """Standard build pipeline: subworld -> replicate -> ground -> solver."""
        sub = newton.ModelBuilder()
        self._configure_sub_builder(sub)
        self.build_subworld(sub)

        builder = newton.ModelBuilder()
        builder.replicate(sub, self.args.world_count)
        if self.use_ground_plane():
            builder.add_ground_plane()
        if self.args.solver.name == "vbd":
            # VBD requires conflict-free particle/body color groups before
            # finalize(); keep it in the shared harness so every VBD scenario
            # gets the same preprocessing path.
            builder.color(include_bending=self.use_vbd_bending_coloring())

        self.model = builder.finalize()
        solver_kwargs = {"deterministic": self.args.solver_deterministic}
        if self.args.solver_deterministic_max_records is not None:
            solver_kwargs["deterministic_max_records"] = self.args.solver_deterministic_max_records
        self.solver = self.args.solver.factory(self.model, **solver_kwargs)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        if self._uses_custom_collision_pipeline():
            self._configure_collision_pipeline()
        else:
            self.contacts = self.model.contacts()
            self._collision_pipeline_warp_deterministic = getattr(wp.config, "deterministic", None)

        viewer.set_model(self.model)
        self.viewer = viewer

        # Give subclasses a chance to set up per-instance state *before*
        # graph capture runs (which calls ``per_step`` as a warm-up).
        self._on_built()
        self._maybe_capture_graph()

    def _configure_sub_builder(self, sub: newton.ModelBuilder) -> None:
        """Hook for scenarios that need solver-specific attribute registration.

        Called on the single-world builder before :meth:`build_subworld`.
        MuJoCo solver requires ``register_custom_attributes`` to wire up
        per-scenario solver attributes; other solvers are a no-op.
        """
        if self.args.solver.name == "mujoco":
            newton_solvers.SolverMuJoCo.register_custom_attributes(sub)

    def _on_built(self) -> None:
        """Scenario hook right after the model + solver are ready."""
        return None

    def use_ground_plane(self) -> bool:
        """Return whether the standard build pipeline should add a ground plane."""
        return True

    def use_deterministic_collision_pipeline(self) -> bool:
        """Return whether the scenario should sort generated contacts deterministically."""
        return self.use_ground_plane()

    def use_vbd_bending_coloring(self) -> bool:
        """Return whether VBD coloring should include bending-edge dependencies."""
        return False

    def _uses_custom_collision_pipeline(self) -> bool:
        """Return whether this run needs a non-default collision pipeline."""
        return bool(
            self.use_deterministic_collision_pipeline()
            or self.args.collision_pipeline_deterministic
            or self.args.collision_pipeline_warp_deterministic is not None
        )

    def _configure_collision_pipeline(self) -> None:
        """Install and warm a custom collision pipeline for this scenario.

        Warp bakes ``wp.config.deterministic`` into compiled kernel variants. To
        compare solver determinism against a different collision-kernel mode, we
        compile and warm the collision pipeline under its requested Warp mode,
        then restore the process-global mode before solver kernels are captured.
        """
        assert self.model is not None
        assert self.state_0 is not None

        original_mode = getattr(wp.config, "deterministic", None)
        compile_mode = self.args.collision_pipeline_warp_deterministic or original_mode

        try:
            if compile_mode is not None:
                wp.config.deterministic = compile_mode

            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="explicit",
                deterministic=self.use_deterministic_collision_pipeline() or self.args.collision_pipeline_deterministic,
            )
            self.model._collision_pipeline = self.collision_pipeline
            self.contacts = self.collision_pipeline.contacts()
            self._collision_pipeline_warp_deterministic = compile_mode

            if compile_mode != original_mode:
                # Warm one collide() call so this pipeline keeps its own compiled
                # kernels even after we restore the process-global solver mode.
                self.collision_pipeline.collide(self.state_0, self.contacts)
        finally:
            if original_mode is not None:
                wp.config.deterministic = original_mode

    def _maybe_capture_graph(self) -> None:
        """Capture a CUDA graph for the substep loop when running on CUDA."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    # --- per-frame loop -----------------------------------------------------

    def _simulate(self) -> None:
        """Inner substep loop; graph-capturable."""
        assert self.model is not None
        assert self.solver is not None
        assert self.state_0 is not None
        assert self.state_1 is not None
        assert self.control is not None
        assert self.contacts is not None
        model = self.model
        solver = self.solver
        state_in = self.state_0
        state_out = self.state_1
        control = self.control
        contacts = self.contacts
        dt = 1.0 / self.args.fps / self.args.substeps
        for _substep in range(self.args.substeps):
            state_in.clear_forces()
            # Expose the current substep buffers to scenario hooks so per_step()
            # can read and modify the live simulation state rather than the
            # frame-start buffers from the previous swap.
            self.state_0, self.state_1 = state_in, state_out
            self.viewer.apply_forces(state_in)
            self.per_step()
            collision_mode = self._collision_pipeline_warp_deterministic
            original_mode = getattr(wp.config, "deterministic", None)
            try:
                if collision_mode is not None and collision_mode != original_mode:
                    wp.config.deterministic = collision_mode
                model.collide(state_in, contacts)
            finally:
                if collision_mode is not None and original_mode is not None:
                    wp.config.deterministic = original_mode
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

        self.state_0, self.state_1 = state_in, state_out

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.step_index += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.step_index * (1.0 / self.args.fps))
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # --- snapshot -----------------------------------------------------------

    def snapshot(self) -> ScenarioSnapshot:
        assert self.state_0 is not None and self.model is not None
        core = {
            "body_q": self.state_0.body_q.numpy().copy(),
            "body_qd": self.state_0.body_qd.numpy().copy(),
            "joint_q": self.state_0.joint_q.numpy().copy(),
            "joint_qd": self.state_0.joint_qd.numpy().copy(),
        }
        extras = self.extra_snapshot()
        meta = {
            "scenario": self.id,
            "solver": self.args.solver.name,
            "world_count": self.args.world_count,
            "num_steps": self.args.num_steps,
            "seed": self.args.seed,
            "fps": self.args.fps,
            "substeps": self.args.substeps,
            "warp_version": wp.__version__,
            "wp_deterministic": wp.config.deterministic,
            "collision_pipeline_deterministic": bool(getattr(self.model._collision_pipeline, "deterministic", False)),
            "collision_pipeline_wp_deterministic": self._collision_pipeline_warp_deterministic,
            "custom_collision_pipeline": self.collision_pipeline is not None,
            "graph_capture_enabled": self.graph is not None,
        }
        return ScenarioSnapshot(core=core, extras=extras, meta=meta)


# ---------------------------------------------------------------------------
# Subprocess-based determinism comparison
# ---------------------------------------------------------------------------


def hash_core(snap: ScenarioSnapshot) -> str:
    return hashlib.sha256(snap.core_bytes()).hexdigest()[:16]


def compare_runs(
    cli_args: list[str],
    runs: int,
    runner_path: Path,
) -> tuple[bool, list[ScenarioSnapshot], list[str]]:
    """Run the CLI ``runs`` times in subprocesses and compare ``core`` hashes.

    Each subrun writes its pickled :class:`ScenarioSnapshot` to a unique
    temp file passed via ``--_snapshot-out``. We avoid stdout piping
    because Warp's init banner and kernel-cache chatter are written to
    stdout and would otherwise corrupt the pickle stream.

    Args:
        cli_args: Arguments to pass to ``run_determinism.py`` *after* the
            internal ``_subrun`` marker. Must include the scenario and
            solver selection. The harness appends ``--_snapshot-out`` itself.
        runs: Number of independent subprocess invocations.
        runner_path: Absolute path to ``run_determinism.py`` so we
            re-execute ourselves with the active ``sys.executable``.

    Returns:
        ``(all_equal, snapshots, hashes)``. ``all_equal`` is True iff every
        snapshot's ``core_bytes`` hash matches the first.
    """
    snapshots: list[ScenarioSnapshot] = []
    hashes: list[str] = []

    with tempfile.TemporaryDirectory(prefix="newton-det-") as tmp:
        for run_idx in range(runs):
            snap_path = Path(tmp) / f"run_{run_idx:03d}.pkl"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(runner_path),
                    "_subrun",
                    "--_snapshot-out",
                    str(snap_path),
                    *cli_args,
                ],
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                tail = proc.stderr.decode(errors="replace").splitlines()[-25:]
                stdout_tail = proc.stdout.decode(errors="replace").splitlines()[-5:]
                raise RuntimeError(
                    f"Subrun {run_idx} failed (exit={proc.returncode}):\n"
                    + "stderr (tail):\n"
                    + "\n".join(tail)
                    + "\nstdout (tail):\n"
                    + "\n".join(stdout_tail)
                )
            if not snap_path.exists():
                raise RuntimeError(
                    f"Subrun {run_idx} exited 0 but did not write the snapshot "
                    f"file {snap_path}. stdout tail:\n"
                    + "\n".join(proc.stdout.decode(errors="replace").splitlines()[-10:])
                )
            snap: ScenarioSnapshot = pickle.loads(snap_path.read_bytes())
            snapshots.append(snap)
            hashes.append(hash_core(snap))

    all_equal = len(set(hashes)) == 1
    return all_equal, snapshots, hashes


def write_snapshot_to_path(snap: ScenarioSnapshot, path: Path) -> None:
    """Pickle ``snap`` to ``path`` atomically (write + rename)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(snap))
    tmp.replace(path)
