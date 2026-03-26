#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

dash_enabled="${DASHBOARD_ENABLED,,}"
if [[ "${dash_enabled}" != "1" && "${dash_enabled}" != "true" && "${dash_enabled}" != "yes" ]]; then
  echo "SN54 dashboard status"
  echo "state: DISABLED (DASHBOARD_ENABLED=${DASHBOARD_ENABLED})"
  exit 0
fi

echo "SN54 dashboard status"
echo "url: http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
echo "log: ${DASHBOARD_LOG_FILE}"

if [[ -f "${DASHBOARD_PID_FILE}" ]]; then
  pid="$(cat "${DASHBOARD_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    etime="$(ps -p "${pid}" -o etime= | awk '{print $1}')"
    echo "state: RUNNING (pid ${pid}, uptime ${etime})"
  else
    echo "state: DOWN (stale pid file)"
  fi
else
  echo "state: DOWN"
fi

if [[ -f "${DASHBOARD_LOG_FILE}" ]]; then
  echo
  echo "last 20 log lines"
  tail -n 20 "${DASHBOARD_LOG_FILE}"
fi
