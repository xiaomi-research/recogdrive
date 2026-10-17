#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CACHE_HIDDEN_STATE=False
exec bash "${SCRIPT_DIR}/run_caching_recogdrive.sh" "$@"
