#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR="/etc/systemd/system"
MINER_UNIT="${UNIT_DIR}/sn54-miner.service"
TELEGRAM_UNIT="${UNIT_DIR}/sn54-telegram.service"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root (or with sudo)."
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl is not available on this machine."
  echo "Use non-systemd mode: bash ${PROJECT_ROOT}/scripts/miner/start_sn54_stack.sh"
  exit 1
fi

if ! systemctl show --property=Version >/dev/null 2>&1; then
  echo "This environment is not booted with systemd as PID 1."
  echo "systemd services cannot run here."
  echo
  echo "Use non-systemd mode instead:"
  echo "  bash ${PROJECT_ROOT}/scripts/miner/start_sn54_stack.sh"
  echo "  bash ${PROJECT_ROOT}/scripts/miner/status_sn54_miner.sh"
  echo "  bash ${PROJECT_ROOT}/scripts/miner/status_sn54_telegram.sh"
  echo
  echo "If you want auto-restart in this container, use PM2 mode:"
  echo "  bash ${PROJECT_ROOT}/scripts/miner/pm2_start_sn54_stack.sh"
  exit 2
fi

cat > "${MINER_UNIT}" <<UNIT
[Unit]
Description=SN54 MIID Miner
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${PROJECT_ROOT}
Environment=MINER_ENV_FILE=${PROJECT_ROOT}/scripts/miner/miner.env
Environment=PYTHONUNBUFFERED=1
ExecStartPre=/bin/mkdir -p ${PROJECT_ROOT}/logs ${PROJECT_ROOT}/run
ExecStartPre=/bin/rm -f ${PROJECT_ROOT}/run/sn54-miner.pid
ExecStart=${PROJECT_ROOT}/scripts/miner/run_sn54_miner_foreground.sh
ExecStopPost=/bin/rm -f ${PROJECT_ROOT}/run/sn54-miner.pid
Restart=on-failure
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=45
LimitNOFILE=65535
StandardOutput=append:${PROJECT_ROOT}/logs/sn54-miner.log
StandardError=append:${PROJECT_ROOT}/logs/sn54-miner.log

[Install]
WantedBy=multi-user.target
UNIT

cat > "${TELEGRAM_UNIT}" <<UNIT
[Unit]
Description=SN54 Miner Telegram Notifier
Wants=network-online.target
After=network-online.target sn54-miner.service
Requires=sn54-miner.service

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${PROJECT_ROOT}
Environment=MINER_ENV_FILE=${PROJECT_ROOT}/scripts/miner/miner.env
Environment=PYTHONUNBUFFERED=1
ExecStartPre=/bin/mkdir -p ${PROJECT_ROOT}/logs ${PROJECT_ROOT}/run
ExecStartPre=/bin/rm -f ${PROJECT_ROOT}/run/sn54-telegram.pid
ExecStart=${PROJECT_ROOT}/scripts/miner/run_sn54_telegram_foreground.sh
ExecStopPost=/bin/rm -f ${PROJECT_ROOT}/run/sn54-telegram.pid
Restart=on-failure
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=30
StandardOutput=append:${PROJECT_ROOT}/logs/sn54-telegram.log
StandardError=append:${PROJECT_ROOT}/logs/sn54-telegram.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable sn54-miner.service sn54-telegram.service
systemctl restart sn54-miner.service
systemctl restart sn54-telegram.service

echo
echo "Services installed and restarted."
systemctl --no-pager --full status sn54-miner.service | sed -n '1,40p'
echo
systemctl --no-pager --full status sn54-telegram.service | sed -n '1,40p'
