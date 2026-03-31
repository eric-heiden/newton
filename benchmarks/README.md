# Solver Benchmarks

Run the benchmark runner from the repository root:

```bash
uv run --extra dev python benchmarks/run_solver_benchmarks.py
```

Useful options:

```bash
uv run --extra dev python benchmarks/run_solver_benchmarks.py --list-scenarios
uv run --extra dev python benchmarks/run_solver_benchmarks.py --scenario g1_humanoid --world-count 8
uv run --extra dev python benchmarks/run_solver_benchmarks.py --solver SolverXPBD --solver SolverMuJoCo --solver SolverMABD
uv run --extra dev python benchmarks/run_solver_benchmarks.py --mujoco-contacts
```

Outputs:

- Raw run logs: `benchmarks/results/runs/*.json`
- Aggregated index: `benchmarks/results/index.json`
- Static dashboard: `benchmarks/dashboard/index.html`
