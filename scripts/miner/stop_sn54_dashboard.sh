#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if [[ ! -f "${DASHBOARD_PID_FILE}" ]]; then
  echo "Dashboard PID file not found: ${DASHBOARD_PID_FILE}"
  exit 0
fi

pid="$(cat "${DASHBOARD_PID_FILE}" 2>/dev/null || true)"
if [[ -z "${pid}" ]]; then
  echo "Dashboard PID file is empty."
  rm -f "${DASHBOARD_PID_FILE}"
  exit 0
fi

if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  sleep 1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}"
  fi
  echo "SN54 dashboard stopped (PID ${pid})."
else
  echo "No running process for PID ${pid}. Cleaning stale PID file."
fi

rm -f "${DASHBOARD_PID_FILE}"
