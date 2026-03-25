#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/start_sn54_miner.sh"
"${SCRIPT_DIR}/start_sn54_dashboard.sh"
"${SCRIPT_DIR}/start_sn54_telegram.sh"

echo "SN54 mining stack is up."
