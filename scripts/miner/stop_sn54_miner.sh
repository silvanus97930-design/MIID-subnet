#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

find_existing_miner_pid() {
  pgrep -f "neurons/miner.py.*--wallet.name ${WALLET_NAME}.*--wallet.hotkey ${WALLET_HOTKEY}" | head -n 1 || true
}

pid=""
if [[ -f "${MINER_PID_FILE}" ]]; then
  pid="$(cat "${MINER_PID_FILE}" 2>/dev/null || true)"
fi

if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
  pid="$(find_existing_miner_pid)"
fi

if [[ -z "${pid}" ]]; then
  echo "Miner PID file not found and no matching running miner process detected."
  rm -f "${MINER_PID_FILE}"
  exit 0
fi

if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  sleep 2
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}"
  fi
  echo "SN54 miner stopped (PID ${pid})."
else
  echo "No running process for PID ${pid}. Cleaning stale PID file."
fi

rm -f "${MINER_PID_FILE}"
