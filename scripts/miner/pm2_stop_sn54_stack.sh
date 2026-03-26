#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 is not installed or not in PATH."
  exit 1
fi

find_wallet_miner_pids() {
  pgrep -f "neurons/miner.py.*--wallet.name ${WALLET_NAME}.*--wallet.hotkey ${WALLET_HOTKEY}" || true
}

find_wallet_telegram_pids() {
  pgrep -f "telegram_notifier.py.*--state-file ${TELEGRAM_STATE_FILE}" || true
}

stop_wallet_miner_pids() {
  local pids=()
  local still_running=()
  mapfile -t pids < <(find_wallet_miner_pids)
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return
  fi

  echo "Stopping wallet miner process(es): ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 2

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      still_running+=("${pid}")
    fi
  done

  if [[ "${#still_running[@]}" -gt 0 ]]; then
    echo "Force killing stubborn wallet miner process(es): ${still_running[*]}"
    kill -9 "${still_running[@]}" 2>/dev/null || true
  fi
}

stop_wallet_telegram_pids() {
  local pids=()
  local still_running=()
  mapfile -t pids < <(find_wallet_telegram_pids)
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return
  fi

  echo "Stopping wallet telegram notifier process(es): ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 2

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      still_running+=("${pid}")
    fi
  done

  if [[ "${#still_running[@]}" -gt 0 ]]; then
    echo "Force killing stubborn wallet telegram notifier process(es): ${still_running[*]}"
    kill -9 "${still_running[@]}" 2>/dev/null || true
  fi
}

pm2 delete sn54-telegram >/dev/null 2>&1 || true
pm2 delete sn54-dashboard >/dev/null 2>&1 || true
pm2 delete sn54-miner >/dev/null 2>&1 || true
pm2 save >/dev/null 2>&1 || true

stop_wallet_telegram_pids
stop_wallet_miner_pids

rm -f "${MINER_PID_FILE}" "${DASHBOARD_PID_FILE}" "${TELEGRAM_PID_FILE}"

echo "PM2 SN54 stack stopped."
