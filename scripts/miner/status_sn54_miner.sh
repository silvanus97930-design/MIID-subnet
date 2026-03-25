#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

find_existing_miner_pid() {
  pgrep -f "neurons/miner.py.*--wallet.name ${WALLET_NAME}.*--wallet.hotkey ${WALLET_HOTKEY}" | head -n 1 || true
}

echo "SN54 miner status"
echo "wallet: ${WALLET_NAME:-unset}/${WALLET_HOTKEY:-unset}"
echo "network: ${SUBTENSOR_NETWORK} netuid ${NETUID}"
echo "model: ${MINER_MODEL_NAME}"
echo "log: ${MINER_LOG_FILE}"

status_reported=false
stale_pid_file=false
if [[ -f "${MINER_PID_FILE}" ]]; then
  pid="$(cat "${MINER_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    etime="$(ps -p "${pid}" -o etime= | awk '{print $1}')"
    echo "state: RUNNING (pid ${pid}, uptime ${etime})"
    status_reported=true
  else
    stale_pid_file=true
  fi
fi

if [[ "${status_reported}" == "false" ]]; then
  existing_pid="$(find_existing_miner_pid)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    etime="$(ps -p "${existing_pid}" -o etime= | awk '{print $1}')"
    echo "${existing_pid}" > "${MINER_PID_FILE}"
    echo "state: RUNNING (detected pid ${existing_pid}, uptime ${etime})"
    status_reported=true
  fi
fi

if [[ "${status_reported}" == "false" ]]; then
  if [[ "${stale_pid_file}" == "true" ]]; then
    echo "state: DOWN (stale pid file)"
  else
    echo "state: DOWN"
  fi
fi

echo
"${MINER_ENV_PATH}/bin/python" - <<PY
import sys
sys.path.insert(0, "${PROJECT_ROOT}/monitoring")
from port_health import PublicPortHealthMonitor, parse_csv_urls, str_to_bool

monitor = PublicPortHealthMonitor(
    enabled=str_to_bool("${PORT_HEALTH_ENABLED}", default=True),
    axon_port=int("${MINER_AXON_PORT}"),
    external_ip="${AXON_EXTERNAL_IP}".strip() or None,
    external_port=(int("${AXON_EXTERNAL_PORT}") if "${AXON_EXTERNAL_PORT}".strip() else None),
    check_interval_seconds=float("${PORT_HEALTH_CHECK_SECONDS}"),
    failure_threshold=int("${PORT_HEALTH_FAILURE_THRESHOLD}"),
    recovery_threshold=int("${PORT_HEALTH_RECOVERY_THRESHOLD}"),
    connect_timeout_seconds=float("${PORT_HEALTH_CONNECT_TIMEOUT_SECONDS}"),
    ip_discovery_urls=parse_csv_urls("${PORT_HEALTH_IP_DISCOVERY_URLS}"),
    port_check_url="${PORT_HEALTH_CHECK_URL}",
)
snapshot = monitor.snapshot_dict(force=True)
print(
    "public_port_health: "
    f"{snapshot.get('state', 'unknown').upper()} "
    f"target={snapshot.get('target_ip', 'n/a')}:{snapshot.get('target_port', 'n/a')} "
    f"detail={snapshot.get('reason', 'n/a')}"
)
PY

if [[ -f "${MINER_LOG_FILE}" ]]; then
  echo
  echo "last 20 log lines"
  tail -n 20 "${MINER_LOG_FILE}"
else
  echo
  echo "No miner log file yet."
fi
