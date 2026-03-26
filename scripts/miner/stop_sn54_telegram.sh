#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

find_telegram_pids() {
  pgrep -f "telegram_notifier.py.*--state-file ${TELEGRAM_STATE_FILE}" || true
}

pids=()
if [[ -f "${TELEGRAM_PID_FILE}" ]]; then
  pid="$(cat "${TELEGRAM_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]]; then
    pids+=("${pid}")
  fi
fi

mapfile -t detected < <(find_telegram_pids)
if [[ "${#detected[@]}" -gt 0 ]]; then
  pids+=("${detected[@]}")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "Telegram notifier PID file not found and no matching running process detected."
  rm -f "${TELEGRAM_PID_FILE}"
  exit 0
fi

unique_pids=()
for pid in "${pids[@]}"; do
  [[ -z "${pid}" ]] && continue
  already=false
  for seen in "${unique_pids[@]}"; do
    if [[ "${seen}" == "${pid}" ]]; then
      already=true
      break
    fi
  done
  if [[ "${already}" == "false" ]]; then
    unique_pids+=("${pid}")
  fi
done

running=()
for pid in "${unique_pids[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    running+=("${pid}")
  fi
done

if [[ "${#running[@]}" -eq 0 ]]; then
  echo "No running telegram notifier process found. Cleaning stale PID file."
  rm -f "${TELEGRAM_PID_FILE}"
  exit 0
fi

echo "Stopping telegram notifier process(es): ${running[*]}"
kill "${running[@]}" 2>/dev/null || true
sleep 1

still_running=()
for pid in "${running[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    still_running+=("${pid}")
  fi
done

if [[ "${#still_running[@]}" -gt 0 ]]; then
  echo "Force killing stubborn telegram notifier process(es): ${still_running[*]}"
  kill -9 "${still_running[@]}" 2>/dev/null || true
fi

rm -f "${TELEGRAM_PID_FILE}"
echo "SN54 telegram notifier stopped."
