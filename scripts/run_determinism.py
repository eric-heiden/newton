"""CLI entry point for the Newton determinism test harness.

Usage
-----

List scenarios and solvers:

    uv run --no-sync python scripts/run_determinism.py --list

Run a single scenario headless and see the final extras (no determinism
check):

    uv run --no-sync python scripts/run_determinism.py \\
        --scenario falling_cube --solver xpbd --world-count 4 \\
        --num-steps 500 --viewer null

Check bit-exact determinism by running the scenario in 3 subprocesses and
comparing final states:

    uv run --no-sync python scripts/run_determinism.py \\
        --scenario falling_cube --solver xpbd --runs 3

Visualize interactively with the GL viewer:

    uv run --no-sync python scripts/run_determinism.py \\
        --scenario domino_chain --solver xpbd --world-count 1 \\
        --num-steps 600 --viewer gl

Toggle Warp's new deterministic atomics:

    uv run --no-sync python scripts/run_determinism.py \\
        --scenario falling_cube --solver xpbd --runs 3 \\
        --warp-deterministic run_to_run

Design
------

- The single-process path (``--runs 1``, default) builds the scenario in
  the current process and either runs headless or opens the chosen viewer.
- The multi-run path (``--runs >= 2``) re-invokes this script with
  ``_subrun`` for each independent replay, hashes the ``core`` bytes of
  each snapshot, and asserts bit-exact agreement.
- ``_subrun`` is an internal flag; users don't call it directly.
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS_FILE.parent

# Allow ``python scripts/run_determinism.py`` to import the sibling package
# without installing anything. We add the repo root (scripts' parent) so
# that ``import scripts.determinism`` resolves.
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_determinism",
        description="Newton determinism test harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and solvers, then exit.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="box_stack",
        help="Scenario id (see --list).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        help="Solver id (see --list).",
    )
    parser.add_argument(
        "--world-count",
        type=int,
        default=1,
        help="Number of replicated worlds in the scenario.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=300,
        help="Total simulation frames to step.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="Physics substeps per frame.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Simulation frame rate used to derive dt = 1/fps/substeps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for scenario-level RNG (pose jitter etc).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent subprocess replays for determinism check. "
        "1 = single-process run, no determinism comparison.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["null", "gl", "usd", "rerun", "viser"],
        help="Viewer backend. Use 'null' for headless runs.",
    )
    parser.add_argument(
        "--warp-deterministic",
        type=str,
        default="run_to_run",
        choices=[None, "not_guaranteed", "run_to_run", "gpu_to_gpu"],
        help="Set wp.config.deterministic before initializing Warp. Propagated to subprocesses via env var.",
    )
    parser.add_argument(
        "--mujoco-deterministic-max-records",
        type=int,
        default=None,
        dest="mujoco_deterministic_max_records",
        help="Deprecated alias for --solver-deterministic-max-records used by MuJoCo scenarios.",
    )
    parser.add_argument(
        "--solver-deterministic",
        type=str,
        default=None,
        choices=[None, "not_guaranteed", "run_to_run", "gpu_to_gpu"],
        help="Per-solver deterministic mode applied via wp.set_module_options on the "
        "solver's kernel modules. Independent of --warp-deterministic; when both are set, "
        "the solver flag pins the solver's modules and the global flag covers everything else.",
    )
    parser.add_argument(
        "--solver-deterministic-max-records",
        type=int,
        default=None,
        help="Per-solver deterministic_max_records module option. If omitted, each solver uses its constructor default.",
    )
    parser.add_argument(
        "--collision-pipeline-deterministic",
        action="store_true",
        help="Use a custom CollisionPipeline with deterministic contact sorting.",
    )
    parser.add_argument(
        "--collision-pipeline-warp-deterministic",
        type=str,
        default=None,
        choices=["not_guaranteed", "run_to_run", "gpu_to_gpu"],
        help="Compile and warm the CollisionPipeline under a specific "
        "wp.config.deterministic mode before restoring the global solver mode.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N frames (applies to gl/usd/viser viewers).",
    )
    parser.add_argument(
        "--print-extras",
        action="store_true",
        help="Print scenario extras after the run.",
    )
    # Internal flag used by the determinism comparison path: the subrun
    # writes its pickled ScenarioSnapshot to this file. We pickle to a
    # file rather than stdout because Warp's init banner and kernel cache
    # chatter are written to stdout and would otherwise prefix the pickle
    # stream with unparsable bytes (UnpicklingError: 'invalid load key').
    parser.add_argument(
        "--_snapshot-out",
        type=str,
        default=None,
        dest="snapshot_out",
        help=argparse.SUPPRESS,
    )
    return parser


def _apply_warp_deterministic(mode: str | None) -> None:
    """Set ``wp.config.deterministic`` *before* any heavy Warp import.

    The new deterministic mode is baked into module compilation hashes, so
    we must set it before any module is compiled for this process. The
    subprocess path also reads this from ``NEWTON_DET_WP_DET`` so settings
    propagate without needing to re-parse CLI args early.
    """
    effective = mode or os.environ.get("NEWTON_DET_WP_DET")
    if effective:
        import warp as wp

        wp.config.deterministic = effective


def _make_viewer(args: argparse.Namespace):
    import newton

    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=args.num_steps)
    if args.viewer == "gl":
        return newton.viewer.ViewerGL()
    if args.viewer == "usd":
        out = _SCRIPTS_DIR / f"_determinism_{args.scenario}_{args.solver}.usd"
        return newton.viewer.ViewerUSD(output_path=str(out), num_frames=args.num_steps)
    if args.viewer == "rerun":
        return newton.viewer.ViewerRerun()
    if args.viewer == "viser":
        return newton.viewer.ViewerViser()
    raise ValueError(f"unknown viewer: {args.viewer}")


def _build_scenario(args: argparse.Namespace):
    from scripts.determinism import SCENARIOS, SOLVER_SPECS, ScenarioArgs

    if args.scenario not in SCENARIOS:
        raise SystemExit(f"Unknown scenario '{args.scenario}'. Known: {sorted(SCENARIOS)}")
    if args.solver not in SOLVER_SPECS:
        raise SystemExit(f"Unknown solver '{args.solver}'. Known: {sorted(SOLVER_SPECS)}")
    scen_cls = SCENARIOS[args.scenario]
    solver = SOLVER_SPECS[args.solver]
    if args.solver not in scen_cls.supported_solvers:
        raise SystemExit(
            f"Scenario '{args.scenario}' does not support solver "
            f"'{args.solver}'. Supported: {scen_cls.supported_solvers}"
        )

    solver_deterministic_max_records = getattr(args, "solver_deterministic_max_records", None)
    if solver_deterministic_max_records is None and args.solver == "mujoco":
        solver_deterministic_max_records = getattr(args, "mujoco_deterministic_max_records", None)

    sargs = ScenarioArgs(
        scenario=args.scenario,
        solver=solver,
        world_count=max(1, args.world_count),
        num_steps=args.num_steps,
        seed=args.seed,
        viewer_name=args.viewer,
        fps=args.fps,
        substeps=args.substeps,
        collision_pipeline_deterministic=args.collision_pipeline_deterministic,
        collision_pipeline_warp_deterministic=args.collision_pipeline_warp_deterministic,
        solver_deterministic=getattr(args, "solver_deterministic", None),
        solver_deterministic_max_records=solver_deterministic_max_records,
    )
    return scen_cls(sargs)


def _run_once(args: argparse.Namespace):
    scen = _build_scenario(args)
    viewer = _make_viewer(args)
    scen.build(viewer)

    for frame in range(args.num_steps):
        if args.viewer != "null" and not viewer.is_running():
            break
        scen.step()
        if args.viewer != "null" and (frame % args.render_every) == 0:
            scen.render()

    return scen.snapshot()


def _format_list() -> str:
    from scripts.determinism import SCENARIOS, SOLVER_SPECS

    lines = ["Scenarios:"]
    for sid, cls in sorted(SCENARIOS.items()):
        doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        lines.append(f"  {sid:<20} supports={cls.supported_solvers}")
        if doc:
            lines.append(f"    {doc}")

    lines.append("")
    lines.append("Solvers:")
    for spec in SOLVER_SPECS.values():
        marker = " (articulated only)" if spec.needs_articulated else ""
        lines.append(f"  {spec.name:<20}{marker}")
        if spec.notes:
            lines.append(f"    {spec.notes}")
    return "\n".join(lines)


def _compare(args: argparse.Namespace) -> int:
    from scripts.determinism import compare_runs

    # Rebuild the subprocess CLI (drop --runs, --viewer; force --viewer null)
    sub_args: list[str] = [
        "--scenario",
        args.scenario,
        "--solver",
        args.solver,
        "--world-count",
        str(args.world_count),
        "--num-steps",
        str(args.num_steps),
        "--substeps",
        str(args.substeps),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
    ]
    if args.warp_deterministic:
        sub_args += ["--warp-deterministic", args.warp_deterministic]
    if args.mujoco_deterministic_max_records is not None:
        sub_args += ["--mujoco-deterministic-max-records", str(args.mujoco_deterministic_max_records)]
    if args.solver_deterministic:
        sub_args += ["--solver-deterministic", args.solver_deterministic]
    if args.solver_deterministic_max_records is not None:
        sub_args += ["--solver-deterministic-max-records", str(args.solver_deterministic_max_records)]
    if args.collision_pipeline_deterministic:
        sub_args += ["--collision-pipeline-deterministic"]
    if args.collision_pipeline_warp_deterministic:
        sub_args += [
            "--collision-pipeline-warp-deterministic",
            args.collision_pipeline_warp_deterministic,
        ]

    all_equal, snapshots, hashes = compare_runs(sub_args, args.runs, _THIS_FILE)

    print(f"\n### {args.scenario} / {args.solver} ### ")
    print(f"  world_count={args.world_count} num_steps={args.num_steps} substeps={args.substeps} fps={args.fps}")
    if args.warp_deterministic:
        print(f"  wp.config.deterministic = {args.warp_deterministic}")
    if args.mujoco_deterministic_max_records is not None:
        print(f"  MJWarp deterministic_max_records = {args.mujoco_deterministic_max_records}")
    if args.solver_deterministic:
        print(f"  solver deterministic = {args.solver_deterministic}")
    if args.solver_deterministic_max_records is not None:
        print(f"  solver deterministic_max_records = {args.solver_deterministic_max_records}")
    if args.collision_pipeline_deterministic or args.collision_pipeline_warp_deterministic:
        print(
            "  collision pipeline:"
            f" deterministic={args.collision_pipeline_deterministic}"
            f" warp={args.collision_pipeline_warp_deterministic or args.warp_deterministic}"
        )

    import numpy as np

    ref = snapshots[0]
    nonfinite: dict[int, list[str]] = {}
    for i, (snap, h) in enumerate(zip(snapshots, hashes, strict=True)):
        if i == 0:
            diff_str = "(reference)"
        else:
            diffs = {k: float(np.abs(snap.core[k] - ref.core[k]).max()) for k in snap.core}
            diff_str = ", ".join(f"{k}:{v:.3e}" for k, v in diffs.items())
        bad_keys = [k for k, v in snap.core.items() if np.asarray(v).dtype.kind in "fc" and not np.isfinite(v).all()]
        if bad_keys:
            nonfinite[i] = bad_keys
        print(f"  run {i}: hash={h}  max core diff: {diff_str}")

    if nonfinite:
        verdict = "NON-FINITE"
    else:
        verdict = "BIT-EXACT" if all_equal else "NON-DETERMINISTIC"
    print(f"  -> {len(set(hashes))} unique hash(es) across {args.runs} runs ({verdict})")
    if nonfinite:
        print("  non-finite core arrays:")
        for run_idx, keys in nonfinite.items():
            print(f"    run {run_idx}: {', '.join(keys)}")

    # Extras diff — not required to be bit-exact, but very useful for
    # understanding *how* the runs diverged.
    extras_keys = sorted(ref.extras.keys())
    if extras_keys:
        print("\nExtras (first-run summary + max abs diff vs run 0):")
        for k in extras_keys:
            v = ref.extras[k]
            if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
                max_diff = 0.0
                for s in snapshots[1:]:
                    other = np.asarray(s.extras.get(k))
                    if other.shape == v.shape and other.dtype == v.dtype:
                        max_diff = max(max_diff, float(np.abs(other - v).max()))
                print(f"  {k}: shape={v.shape} dtype={v.dtype}  max diff={max_diff:.3e}")
            else:
                summary = f"{v!r}"
                if len(summary) > 80:
                    summary = summary[:77] + "..."
                print(f"  {k}: {summary}")

    return 0 if all_equal and not nonfinite else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    subrun = False
    if argv and argv[0] == "_subrun":
        subrun = True
        argv = argv[1:]

    args = _build_parser().parse_args(argv)

    if args.list:
        print(_format_list())
        return 0

    if not args.scenario:
        raise SystemExit("error: --scenario is required (see --list)")

    _apply_warp_deterministic(args.warp_deterministic)

    if subrun:
        # Propagate the mode to match any later subprocess invocations.
        if args.warp_deterministic:
            os.environ["NEWTON_DET_WP_DET"] = args.warp_deterministic
        # Subruns are headless by construction — the comparison path
        # doesn't need (and often can't open) a viewer.
        args.viewer = "null"
        snap = _run_once(args)
        from scripts.determinism import write_snapshot_to_path

        if not args.snapshot_out:
            raise SystemExit(
                "error: _subrun requires --_snapshot-out <path>. This flag is set automatically by compare_runs()."
            )
        write_snapshot_to_path(snap, Path(args.snapshot_out))
        return 0

    if args.runs <= 1:
        snap = _run_once(args)
        from scripts.determinism import hash_core

        print(f"\nSingle-run snapshot: hash={hash_core(snap)}")
        print(f"  scenario={snap.meta['scenario']}  solver={snap.meta['solver']}")
        print("  core shapes: " + ", ".join(f"{k}={v.shape}" for k, v in snap.core.items()))
        if args.print_extras:
            import numpy as np

            print("Extras:")
            for k, v in snap.extras.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"  {k}: {v!r}")
        return 0

    # Propagate wp.config.deterministic to the subprocess env; argparse
    # value is re-parsed inside the subrun as well.
    if args.warp_deterministic:
        os.environ["NEWTON_DET_WP_DET"] = args.warp_deterministic
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
