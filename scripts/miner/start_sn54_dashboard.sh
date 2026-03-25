#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if [[ ! -x "${MINER_ENV_PATH}/bin/python" ]]; then
  echo "Missing virtualenv python at ${MINER_ENV_PATH}/bin/python"
  echo "Run: bash scripts/miner/setup.sh"
  exit 1
fi

if [[ -f "${DASHBOARD_PID_FILE}" ]]; then
  old_pid="$(cat "${DASHBOARD_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Dashboard is already running with PID ${old_pid}."
    echo "URL: http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
    exit 0
  fi
fi

dashboard_cmd=(
  "${MINER_ENV_PATH}/bin/python"
  "${PROJECT_ROOT}/monitoring/sn54_dashboard.py"
  --host "${DASHBOARD_HOST}"
  --port "${DASHBOARD_PORT}"
  --log-file "${MINER_LOG_FILE}"
  --pid-file "${MINER_PID_FILE}"
  --axon-port "${MINER_AXON_PORT}"
  --axon-external-ip "${AXON_EXTERNAL_IP:-}"
  --axon-external-port "${AXON_EXTERNAL_PORT:-0}"
  --port-health-enabled "${PORT_HEALTH_ENABLED}"
  --port-health-check-seconds "${PORT_HEALTH_CHECK_SECONDS}"
  --port-health-failure-threshold "${PORT_HEALTH_FAILURE_THRESHOLD}"
  --port-health-recovery-threshold "${PORT_HEALTH_RECOVERY_THRESHOLD}"
  --port-health-timeout-seconds "${PORT_HEALTH_CONNECT_TIMEOUT_SECONDS}"
  --port-health-ip-discovery-urls "${PORT_HEALTH_IP_DISCOVERY_URLS}"
  --port-health-check-url "${PORT_HEALTH_CHECK_URL}"
)

nohup "${dashboard_cmd[@]}" >>"${DASHBOARD_LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" > "${DASHBOARD_PID_FILE}"

echo "SN54 dashboard started."
echo "PID: ${pid}"
echo "URL: http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
echo "Log: ${DASHBOARD_LOG_FILE}"
