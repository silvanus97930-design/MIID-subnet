#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if [[ ! -f "${TELEGRAM_PID_FILE}" ]]; then
  echo "Telegram notifier PID file not found: ${TELEGRAM_PID_FILE}"
  exit 0
fi

pid="$(cat "${TELEGRAM_PID_FILE}" 2>/dev/null || true)"
if [[ -z "${pid}" ]]; then
  echo "Telegram notifier PID file is empty."
  rm -f "${TELEGRAM_PID_FILE}"
  exit 0
fi

if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  sleep 1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}"
  fi
  echo "SN54 telegram notifier stopped (PID ${pid})."
else
  echo "No running process for PID ${pid}. Cleaning stale PID file."
fi

rm -f "${TELEGRAM_PID_FILE}"
