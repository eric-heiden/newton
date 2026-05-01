#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly WORKTREE_ROOT="${NEWTON_BENCHMARK_WORKTREE_ROOT:-${REPO_ROOT}/.runtime/benchmark-dashboard}"
readonly WORKTREE_PATH="${NEWTON_BENCHMARK_WORKTREE_PATH:-${WORKTREE_ROOT}/main}"
readonly DASHBOARD_REF="${NEWTON_BENCHMARK_REF:-origin/main}"
readonly ASV_MACHINE_NAME="${ASV_MACHINE:-$(hostname -s)}"

mkdir -p "${WORKTREE_ROOT}"

git -C "${REPO_ROOT}" fetch --prune origin main

if [[ ! -d "${WORKTREE_PATH}/.git" ]]; then
    git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE_PATH}" "${DASHBOARD_REF}"
else
    git -C "${WORKTREE_PATH}" fetch --prune origin main
    git -C "${WORKTREE_PATH}" checkout --detach "${DASHBOARD_REF}"
fi

cd "${WORKTREE_PATH}"
mkdir -p asv/results asv/html

uvx --with virtualenv asv machine --yes --machine "${ASV_MACHINE_NAME}"
uvx --with virtualenv asv run --launch-method spawn main^!
uvx --with virtualenv asv publish
