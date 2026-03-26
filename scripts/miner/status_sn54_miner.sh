#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/load_env.sh"

find_miner_pids() {
  pgrep -f "neurons/miner.py.*--wallet.name ${WALLET_NAME}.*--wallet.hotkey ${WALLET_HOTKEY}" || true
}

find_existing_miner_pid() {
  find_miner_pids | head -n 1 || true
}

echo "SN54 miner status"
echo "wallet: ${WALLET_NAME:-unset}/${WALLET_HOTKEY:-unset}"
echo "network: ${SUBTENSOR_NETWORK} netuid ${NETUID}"
echo "model: ${MINER_MODEL_NAME}"
echo "log: ${MINER_LOG_FILE}"

runtime="host"
if [[ -f "/.dockerenv" ]]; then
  runtime="docker"
fi
echo "runtime: ${runtime}"
if [[ "${runtime}" == "docker" ]]; then
  container_ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "${container_ip}" ]]; then
    echo "container_ip: ${container_ip}"
  fi
fi

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

mapfile -t wallet_miner_pids < <(find_miner_pids)
if [[ "${#wallet_miner_pids[@]}" -gt 1 ]]; then
  echo "warning: duplicate miner processes detected for this wallet/hotkey: ${wallet_miner_pids[*]}"
fi

echo
"${MINER_ENV_PATH}/bin/python" - <<PY
import socket
import sys

sys.path.insert(0, "${PROJECT_ROOT}/monitoring")
from port_health import PublicPortHealthMonitor, parse_csv_urls, str_to_bool


def probe_target(ip: str | None, port: int | None, timeout_seconds: float) -> bool | None:
    if not ip or not port:
        return None
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)
        sock.connect((ip, int(port)))
        return True
    except Exception:
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


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
state = str(snapshot.get("state", "unknown")).upper()
target_ip = snapshot.get("target_ip", "n/a")
target_port = snapshot.get("target_port", "n/a")
reason = snapshot.get("reason", "n/a")
source = snapshot.get("source", "n/a")
local_listener = snapshot.get("local_listener")

print(
    "public_port_health: "
    f"{state} "
    f"target={target_ip}:{target_port} "
    f"detail={reason}"
)
print(
    "public_port_meta: "
    f"source={source} local_listener={local_listener}"
)

self_probe = probe_target(
    target_ip if isinstance(target_ip, str) else None,
    int(target_port) if isinstance(target_port, int) else None,
    timeout_seconds=min(float("${PORT_HEALTH_CONNECT_TIMEOUT_SECONDS}"), 2.0),
)
if self_probe is not None:
    print(f"self_probe_to_target: {'OPEN' if self_probe else 'CLOSED'}")

if state == "UNREACHABLE" and local_listener is True:
    if self_probe is True:
        hint = (
            "Local listener is UP and self-probe to public target works, "
            "so closure is likely source-dependent filtering (provider firewall/ACL)."
        )
    else:
        hint = (
            "Local listener is UP but external probe is CLOSED. "
            "Check host firewall, cloud security group, and Docker port publish."
        )
    print(f"reachability_hint: {hint}")
PY

if [[ -f "${MINER_LOG_FILE}" ]]; then
  echo
  echo "last 20 log lines"
  tail -n 20 "${MINER_LOG_FILE}"
else
  echo
  echo "No miner log file yet."
fi
