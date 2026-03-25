#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${WALLET_NAME:-}" || -z "${WALLET_HOTKEY:-}" ]]; then
  echo "WALLET_NAME and WALLET_HOTKEY must be set in ${ENV_FILE}"
  exit 1
fi

if ! command -v btcli >/dev/null 2>&1; then
  echo "btcli was not found in PATH."
  exit 1
fi

echo "Registering wallet ${WALLET_NAME}/${WALLET_HOTKEY} on netuid ${NETUID} (${SUBTENSOR_NETWORK})"
btcli subnets register \
  --netuid "${NETUID}" \
  --wallet-name "${WALLET_NAME}" \
  --hotkey "${WALLET_HOTKEY}" \
  --network "${SUBTENSOR_NETWORK}"
