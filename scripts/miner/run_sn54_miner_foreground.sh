#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${WALLET_NAME:-}" || -z "${WALLET_HOTKEY:-}" ]]; then
  echo "WALLET_NAME and WALLET_HOTKEY must be set in ${ENV_FILE}"
  exit 1
fi

if [[ ! -x "${MINER_ENV_PATH}/bin/python" ]]; then
  echo "Missing virtualenv python at ${MINER_ENV_PATH}/bin/python"
  echo "Run: bash scripts/miner/setup.sh"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama binary was not found in PATH."
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi
if [[ -n "${FLUX_DEVICE:-}" ]]; then
  export FLUX_DEVICE
fi

miner_cmd=(
  "${MINER_ENV_PATH}/bin/python"
  "${PROJECT_ROOT}/neurons/miner.py"
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
  miner_cmd+=(--axon.external_ip "${AXON_EXTERNAL_IP}")
fi
if [[ -n "${AXON_EXTERNAL_PORT:-}" ]]; then
  miner_cmd+=(--axon.external_port "${AXON_EXTERNAL_PORT}")
fi
if [[ "${LOG_LEVEL}" == "debug" ]]; then
  miner_cmd+=(--logging.debug)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  miner_cmd+=("${extra[@]}")
fi

echo "$$" > "${MINER_PID_FILE}"
exec "${miner_cmd[@]}"
