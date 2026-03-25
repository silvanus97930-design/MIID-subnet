#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 is not installed or not in PATH."
  exit 1
fi

pm2 delete sn54-telegram >/dev/null 2>&1 || true
pm2 delete sn54-dashboard >/dev/null 2>&1 || true
pm2 delete sn54-miner >/dev/null 2>&1 || true
pm2 save >/dev/null 2>&1 || true

rm -f "${MINER_PID_FILE}" "${DASHBOARD_PID_FILE}" "${TELEGRAM_PID_FILE}"

echo "PM2 SN54 stack stopped."
