#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${WALLET_NAME:-}" || -z "${WALLET_HOTKEY:-}" ]]; then
  echo "WALLET_NAME and WALLET_HOTKEY must be set in ${ENV_FILE}"
  exit 1
fi

"${MINER_ENV_PATH}/bin/python" - <<PY
import bittensor as bt

wallet_name = "${WALLET_NAME}"
wallet_hotkey = "${WALLET_HOTKEY}"
netuid = int("${NETUID}")
network = "${SUBTENSOR_NETWORK}"

sub = bt.Subtensor(network=network)
wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
address = wallet.hotkey.ss58_address
registered = sub.is_hotkey_registered(netuid=netuid, hotkey_ss58=address)

print(f"wallet: {wallet_name}/{wallet_hotkey}")
print(f"hotkey_ss58: {address}")
print(f"network: {network}")
print(f"netuid: {netuid}")
print(f"registered: {registered}")
PY
