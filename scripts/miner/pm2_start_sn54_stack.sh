#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 is not installed or not in PATH."
  exit 1
fi

if [[ ! -x "${MINER_ENV_PATH}/bin/python" ]]; then
  echo "Missing virtualenv python at ${MINER_ENV_PATH}/bin/python"
  echo "Run: bash scripts/miner/setup.sh"
  exit 1
fi

if [[ -z "${WALLET_NAME:-}" || -z "${WALLET_HOTKEY:-}" ]]; then
  echo "WALLET_NAME and WALLET_HOTKEY must be set in ${ENV_FILE}"
  exit 1
fi

pm2 delete sn54-miner >/dev/null 2>&1 || true
pm2 delete sn54-dashboard >/dev/null 2>&1 || true
pm2 delete sn54-telegram >/dev/null 2>&1 || true

miner_args=(
  neurons/miner.py
  --netuid "${NETUID}"
  --subtensor.network "${SUBTENSOR_NETWORK}"
  --subtensor.chain_endpoint "${CHAIN_ENDPOINT}"
  --wallet.name "${WALLET_NAME}"
  --wallet.hotkey "${WALLET_HOTKEY}"
  --axon.port "${MINER_AXON_PORT}"
  --neuron.model_name "${MINER_MODEL_NAME}"
  --neuron.ollama_url "${OLLAMA_URL}"
)

if [[ -n "${AXON_EXTERNAL_IP:-}" ]]; then
  miner_args+=(--axon.external_ip "${AXON_EXTERNAL_IP}")
fi
if [[ -n "${AXON_EXTERNAL_PORT:-}" ]]; then
  miner_args+=(--axon.external_port "${AXON_EXTERNAL_PORT}")
fi
if [[ "${LOG_LEVEL}" == "debug" ]]; then
  miner_args+=(--logging.debug)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  miner_args+=("${extra[@]}")
fi

pm2 start "${MINER_ENV_PATH}/bin/python" \
  --name sn54-miner \
  --cwd "${PROJECT_ROOT}" \
  --interpreter none \
  --time \
  --output "${MINER_LOG_FILE}" \
  --error "${MINER_LOG_FILE}" \
  -- "${miner_args[@]}"

dashboard_args=(
  monitoring/sn54_dashboard.py
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

pm2 start "${MINER_ENV_PATH}/bin/python" \
  --name sn54-dashboard \
  --cwd "${PROJECT_ROOT}" \
  --interpreter none \
  --time \
  --output "${DASHBOARD_LOG_FILE}" \
  --error "${DASHBOARD_LOG_FILE}" \
  -- "${dashboard_args[@]}"

enabled="${TELEGRAM_NOTIFIER_ENABLED,,}"
if [[ "${enabled}" == "1" || "${enabled}" == "true" || "${enabled}" == "yes" ]]; then
  if [[ -z "${TELEGRAM_BOT_TOKEN}" || -z "${TELEGRAM_CHAT_ID}" ]]; then
    echo "TELEGRAM_NOTIFIER_ENABLED is true but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are missing."
    exit 1
  fi

  telegram_args=(
    monitoring/telegram_notifier.py
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

  pm2 start "${MINER_ENV_PATH}/bin/python" \
    --name sn54-telegram \
    --cwd "${PROJECT_ROOT}" \
    --interpreter none \
    --time \
    --output "${TELEGRAM_LOG_FILE}" \
    --error "${TELEGRAM_LOG_FILE}" \
    -- "${telegram_args[@]}"
fi

sleep 1

miner_pid="$(pm2 pid sn54-miner | tail -n 1 | tr -d '[:space:]' || true)"
dash_pid="$(pm2 pid sn54-dashboard | tail -n 1 | tr -d '[:space:]' || true)"
tele_pid="$(pm2 pid sn54-telegram | tail -n 1 | tr -d '[:space:]' || true)"

if [[ -n "${miner_pid}" && "${miner_pid}" != "0" ]]; then
  echo "${miner_pid}" > "${MINER_PID_FILE}"
fi
if [[ -n "${dash_pid}" && "${dash_pid}" != "0" ]]; then
  echo "${dash_pid}" > "${DASHBOARD_PID_FILE}"
fi
if [[ -n "${tele_pid}" && "${tele_pid}" != "0" ]]; then
  echo "${tele_pid}" > "${TELEGRAM_PID_FILE}"
fi

pm2 save >/dev/null 2>&1 || true

echo "PM2 SN54 stack started."
echo "Miner log: ${MINER_LOG_FILE}"
echo "Dashboard: http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
echo "Telegram log: ${TELEGRAM_LOG_FILE}"
