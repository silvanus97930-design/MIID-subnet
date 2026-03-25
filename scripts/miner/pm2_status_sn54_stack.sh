#!/usr/bin/env bash
set -euo pipefail

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 is not installed or not in PATH."
  exit 1
fi

echo "PM2 SN54 stack status"
pm2 list

echo
echo "PM2 process details:"
pm2 show sn54-miner 2>/dev/null | sed -n '1,120p' || echo "sn54-miner not found"
pm2 show sn54-dashboard 2>/dev/null | sed -n '1,90p' || echo "sn54-dashboard not found"
pm2 show sn54-telegram 2>/dev/null | sed -n '1,90p' || echo "sn54-telegram not found"
