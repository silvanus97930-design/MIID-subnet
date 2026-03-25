#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

echo "SN54 telegram notifier status"
echo "chat_id: ${TELEGRAM_CHAT_ID:-unset}"
echo "log: ${TELEGRAM_LOG_FILE}"

if [[ -f "${TELEGRAM_PID_FILE}" ]]; then
  pid="$(cat "${TELEGRAM_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    etime="$(ps -p "${pid}" -o etime= | awk '{print $1}')"
    echo "state: RUNNING (pid ${pid}, uptime ${etime})"
  else
    echo "state: DOWN (stale pid file)"
  fi
else
  echo "state: DOWN"
fi

if [[ -f "${TELEGRAM_LOG_FILE}" ]]; then
  echo
  echo "last 20 log lines"
  tail -n 20 "${TELEGRAM_LOG_FILE}"
fi
