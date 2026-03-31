#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

find_existing_miner_pid() {
  pgrep -f "neurons/miner.py.*--wallet.name ${WALLET_NAME}.*--wallet.hotkey ${WALLET_HOTKEY}" | head -n 1 || true
}

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

if [[ "${SN54_IMAGE_GENERATION_BACKEND:-comfyui}" == "comfyui" ]]; then
  if ! python3 - <<'PY'
import os
import urllib.request

base_url = (os.environ.get("COMFYUI_BASE_URL") or "http://127.0.0.1:20007").strip().rstrip("/")
with urllib.request.urlopen(f"{base_url}/system_stats", timeout=20) as response:
    if response.status != 200:
        raise RuntimeError(f"unexpected HTTP status {response.status}")
print(f"ComfyUI reachable at {base_url}")
PY
  then
    echo "ComfyUI preflight failed. Check COMFYUI_BASE_URL and ensure the server is running."
    exit 1
  fi
fi

if [[ -f "${MINER_PID_FILE}" ]]; then
  old_pid="$(cat "${MINER_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "SN54 miner is already running with PID ${old_pid}."
    exit 0
  fi
fi

existing_pid="$(find_existing_miner_pid)"
if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
  echo "${existing_pid}" > "${MINER_PID_FILE}"
  echo "SN54 miner is already running with PID ${existing_pid} (detected by wallet/hotkey)."
  exit 0
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi
if [[ -n "${FLUX_DEVICE:-}" ]]; then
  export FLUX_DEVICE
fi
if [[ -n "${COMFYUI_BASE_URL:-}" ]]; then
  export COMFYUI_BASE_URL
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

nohup "${miner_cmd[@]}" >>"${MINER_LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" > "${MINER_PID_FILE}"

echo "SN54 miner started."
echo "PID: ${pid}"
echo "Log: ${MINER_LOG_FILE}"
