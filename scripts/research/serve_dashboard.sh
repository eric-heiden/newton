#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly HOST="${NEWTON_RESEARCH_HOST:-0.0.0.0}"
readonly PORT="${NEWTON_RESEARCH_PORT:-7070}"
readonly ARTIFACT_PATH="${NEWTON_RESEARCH_ARTIFACT_PATH:-${REPO_ROOT}/scripts/research_dashboard.json}"

if [[ ! -f "${ARTIFACT_PATH}" ]]; then
    printf 'Research artifact not found at %s\n' "${ARTIFACT_PATH}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
exec uv run python scripts/research_dashboard.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --artifact-path "${ARTIFACT_PATH}"
