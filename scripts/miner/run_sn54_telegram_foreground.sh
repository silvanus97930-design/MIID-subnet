#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

enabled="${TELEGRAM_NOTIFIER_ENABLED,,}"
if [[ "${enabled}" != "1" && "${enabled}" != "true" && "${enabled}" != "yes" ]]; then
  echo "SN54 telegram notifier is disabled (TELEGRAM_NOTIFIER_ENABLED=${TELEGRAM_NOTIFIER_ENABLED})."
  exit 0
fi

if [[ -z "${TELEGRAM_BOT_TOKEN}" || -z "${TELEGRAM_CHAT_ID}" ]]; then
  echo "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in ${ENV_FILE}"
  exit 1
fi

if [[ ! -x "${MINER_ENV_PATH}/bin/python" ]]; then
  echo "Missing virtualenv python at ${MINER_ENV_PATH}/bin/python"
  echo "Run: bash scripts/miner/setup.sh"
  exit 1
fi

notifier_cmd=(
  "${MINER_ENV_PATH}/bin/python"
  "${PROJECT_ROOT}/monitoring/telegram_notifier.py"
  --log-file "${MINER_LOG_FILE}"
  --miner-pid-file "${MINER_PID_FILE}"
  --state-file "${TELEGRAM_STATE_FILE}"
  --axon-port "${MINER_AXON_PORT}"
  --axon-external-ip "${AXON_EXTERNAL_IP:-}"
  --axon-external-port "${AXON_EXTERNAL_PORT:-0}"
  --port-health-enabled "${PORT_HEALTH_ENABLED}"
  --port-health-check-seconds "${PORT_HEALTH_CHECK_SECONDS}"
  --port-health-failure-threshold "${PORT_HEALTH_FAILURE_THRESHOLD}"
  --port-health-recovery-threshold "${PORT_HEALTH_RECOVERY_THRESHOLD}"
  --port-health-timeout-seconds "${PORT_HEALTH_CONNECT_TIMEOUT_SECONDS}"
  --port-health-alert-cooldown-minutes "${PORT_HEALTH_ALERT_COOLDOWN_MINUTES}"
  --port-health-ip-discovery-urls "${PORT_HEALTH_IP_DISCOVERY_URLS}"
  --port-health-check-url "${PORT_HEALTH_CHECK_URL}"
  --bot-token "${TELEGRAM_BOT_TOKEN}"
  --chat-id "${TELEGRAM_CHAT_ID}"
  --poll-seconds "${TELEGRAM_POLL_SECONDS}"
  --flush-seconds "${TELEGRAM_FLUSH_SECONDS}"
  --heartbeat-seconds "${TELEGRAM_HEARTBEAT_SECONDS}"
  --no-traffic-alert-minutes "${TELEGRAM_NO_TRAFFIC_ALERT_MINUTES}"
  --no-traffic-alert-cooldown-minutes "${TELEGRAM_NO_TRAFFIC_ALERT_COOLDOWN_MINUTES}"
  --max-lines-per-message "${TELEGRAM_MAX_LINES_PER_MESSAGE}"
  --max-chars-per-message "${TELEGRAM_MAX_CHARS_PER_MESSAGE}"
)

echo "$$" > "${TELEGRAM_PID_FILE}"
exec "${notifier_cmd[@]}"
