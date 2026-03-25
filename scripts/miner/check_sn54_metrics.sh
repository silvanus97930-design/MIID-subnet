#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

"${MINER_ENV_PATH}/bin/python" - <<PY
import bittensor as bt

network = "${SUBTENSOR_NETWORK}"
netuid = int("${NETUID}")
wallet_name = "${WALLET_NAME}"
wallet_hotkey = "${WALLET_HOTKEY}"

sub = bt.Subtensor(network=network)
mg = sub.metagraph(netuid)
wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
hotkey = wallet.hotkey.ss58_address

if hotkey not in mg.hotkeys:
    print(f"hotkey {hotkey} not found on netuid {netuid}")
    raise SystemExit(1)

uid = mg.hotkeys.index(hotkey)

vals = [float(x) for x in mg.I]
rank = 1 + sum(v > vals[uid] for v in vals)

print(f"wallet: {wallet_name}/{wallet_hotkey}")
print(f"hotkey: {hotkey}")
print(f"network: {network}")
print(f"netuid: {netuid}")
print(f"uid: {uid}")
print(f"incentive: {float(mg.I[uid])}")
print(f"emission: {float(mg.E[uid])}")
print(f"trust: {float(mg.T[uid])}")
print(f"consensus: {float(mg.C[uid])}")
print(f"dividends: {float(mg.D[uid])}")
print(f"incentive_rank: {rank}/{len(vals)}")
print(f"axon: {mg.axons[uid].ip}:{mg.axons[uid].port} serving={mg.axons[uid].is_serving}")
PY
