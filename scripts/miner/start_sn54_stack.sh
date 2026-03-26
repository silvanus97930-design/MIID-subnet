#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

"${SCRIPT_DIR}/start_sn54_miner.sh"

dash_enabled="${DASHBOARD_ENABLED,,}"
if [[ "${dash_enabled}" == "1" || "${dash_enabled}" == "true" || "${dash_enabled}" == "yes" ]]; then
  "${SCRIPT_DIR}/start_sn54_dashboard.sh"
else
  echo "SN54 dashboard start skipped (DASHBOARD_ENABLED=${DASHBOARD_ENABLED})."
fi

"${SCRIPT_DIR}/start_sn54_telegram.sh"

echo "SN54 mining stack is up."
