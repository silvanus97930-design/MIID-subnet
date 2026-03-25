# SN54 Miner + Local Dashboard

This project now includes local scripts to run a Subnet 54 miner and monitor it from a browser.

## 1) Configure Wallet and Runtime

Edit:

`scripts/miner/miner.env`

Important fields:
- `WALLET_NAME`
- `WALLET_HOTKEY`
- `MINER_MODEL_NAME` (default: `llama3.1:latest`)
- `HF_TOKEN` / `FLUX_DEVICE` (for Phase 4 image variation tasks)

## 2) Register (if not already registered)

Check registration:

```bash
bash scripts/miner/check_sn54_registration.sh
```

If it returns `registered: False`, register:

```bash
bash scripts/miner/register_sn54_miner.sh
```

## 3) Start Mining

```bash
bash scripts/miner/start_sn54_miner.sh
```

Status and logs:

```bash
bash scripts/miner/status_sn54_miner.sh
```

Stop miner:

```bash
bash scripts/miner/stop_sn54_miner.sh
```

## 4) Start Dashboard

```bash
bash scripts/miner/start_sn54_dashboard.sh
```

Open:

`http://127.0.0.1:8810`

Dashboard status:

```bash
bash scripts/miner/status_sn54_dashboard.sh
```

Stop dashboard:

```bash
bash scripts/miner/stop_sn54_dashboard.sh
```

## Telegram Notifier

Configure in:

`scripts/miner/miner.env`

Required values:
- `TELEGRAM_NOTIFIER_ENABLED=true`
- `TELEGRAM_BOT_TOKEN=...`
- `TELEGRAM_CHAT_ID=...`

Start only Telegram notifier:

```bash
bash scripts/miner/start_sn54_telegram.sh
```

Check status:

```bash
bash scripts/miner/status_sn54_telegram.sh
```

Stop:

```bash
bash scripts/miner/stop_sn54_telegram.sh
```

The notifier sends:
- miner up/down state changes
- batched miner log lines
- immediate alert batches when errors/exceptions appear
- periodic heartbeat messages
- no-traffic alerts when no validator request hits the miner for a configured time window
- public-port unreachable alerts (plus recovery alerts) when external reachability changes

No-traffic alert controls in `scripts/miner/miner.env`:
- `TELEGRAM_NO_TRAFFIC_ALERT_MINUTES=20`
- `TELEGRAM_NO_TRAFFIC_ALERT_COOLDOWN_MINUTES=30`

Public port health controls in `scripts/miner/miner.env`:
- `PORT_HEALTH_ENABLED=true`
- `PORT_HEALTH_CHECK_SECONDS=60`
- `PORT_HEALTH_FAILURE_THRESHOLD=1`
- `PORT_HEALTH_RECOVERY_THRESHOLD=1`
- `PORT_HEALTH_CONNECT_TIMEOUT_SECONDS=8`
- `PORT_HEALTH_ALERT_COOLDOWN_MINUTES=30`

## If Incentive Stays Zero

Check these first:

1. Ensure your hotkey is registered:
```bash
bash scripts/miner/check_sn54_registration.sh
```

2. Check live SN54 incentives/metrics for your UID:
```bash
bash scripts/miner/check_sn54_metrics.sh
```

3. Ensure miner is receiving validator traffic:
```bash
rg -n "Starting run|Verified call from" logs/sn54-miner.log
```

3. If you run behind NAT, forward TCP `8091` on your router/firewall to this miner host and set:
- `AXON_EXTERNAL_IP`
- `AXON_EXTERNAL_PORT`

in `scripts/miner/miner.env`.

Without public reachability to your axon port, validators cannot query your miner and incentive remains zero.

## Optional: Start/Stop Everything Together

```bash
bash scripts/miner/start_sn54_stack.sh
bash scripts/miner/stop_sn54_stack.sh
```

## Recommended Persistent Mode (PM2)

To keep miner + dashboard + telegram notifier alive across shell exits/restarts:

```bash
bash scripts/miner/pm2_start_sn54_stack.sh
```

Status:

```bash
bash scripts/miner/pm2_status_sn54_stack.sh
```

Stop:

```bash
bash scripts/miner/pm2_stop_sn54_stack.sh
```

## What the Dashboard Shows

- Miner process status (running/stopped, PID, uptime)
- Total runs observed in miner logs
- Requested vs processed name count
- Phase 4 S3 submission count
- Verified validator calls
- Public port reachability (reachable/unreachable/unknown)
- Error/warning counts
- Recent event feed and live log tail
