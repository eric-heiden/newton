#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly BASE_URL="${NEWTON_RESEARCH_DASHBOARD_URL:-http://127.0.0.1:7070}"

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <entry-json>\n' "${0##*/}" >&2
    exit 1
fi

readonly PAYLOAD_PATH="$1"

if [[ ! -f "${PAYLOAD_PATH}" ]]; then
    printf 'Entry payload not found at %s\n' "${PAYLOAD_PATH}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
curl --fail --silent --show-error \
    -X POST \
    -H 'Content-Type: application/json' \
    --data "@${PAYLOAD_PATH}" \
    "${BASE_URL}/api/research/entries"
