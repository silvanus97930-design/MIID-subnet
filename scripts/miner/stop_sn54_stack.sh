#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/stop_sn54_telegram.sh"
"${SCRIPT_DIR}/stop_sn54_dashboard.sh"
"${SCRIPT_DIR}/stop_sn54_miner.sh"

echo "SN54 mining stack stopped."
