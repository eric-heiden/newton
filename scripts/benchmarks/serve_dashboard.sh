#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly WORKTREE_PATH="${NEWTON_BENCHMARK_WORKTREE_PATH:-${REPO_ROOT}/.runtime/benchmark-dashboard/main}"
readonly RESULTS_DIR="${NEWTON_BENCHMARK_RESULTS_DIR:-${WORKTREE_PATH}/asv/results}"
readonly HTML_DIR="${NEWTON_BENCHMARK_HTML_DIR:-${WORKTREE_PATH}/asv/html}"
readonly BENCHMARK_INDEX="${NEWTON_BENCHMARK_INDEX_PATH:-${REPO_ROOT}/benchmarks/results/index.json}"
readonly HOST="${NEWTON_BENCHMARK_HOST:-0.0.0.0}"
readonly PORT="${NEWTON_BENCHMARK_PORT:-7000}"

if [[ ! -f "${HTML_DIR}/index.html" && ! -f "${BENCHMARK_INDEX}" ]]; then
    printf 'Dashboard HTML not found at %s\n' "${HTML_DIR}" >&2
    printf 'Solver benchmark artifact not found at %s\n' "${BENCHMARK_INDEX}" >&2
    printf 'Run scripts/benchmarks/refresh_dashboard.sh or benchmarks/run_solver_benchmarks.py first.\n' >&2
    exit 1
fi

cd "${REPO_ROOT}"
exec uv run python scripts/benchmark_dashboard.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --results-dir "${RESULTS_DIR}" \
    --html-dir "${HTML_DIR}" \
    --benchmark-index "${BENCHMARK_INDEX}"
