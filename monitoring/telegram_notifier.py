#!/usr/bin/env python3
"""Send miner log updates to Telegram."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from port_health import PublicPortHealthMonitor, parse_csv_urls, str_to_bool


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


@dataclass
class TailState:
    inode: Optional[int] = None
    offset: int = 0


class TelegramNotifier:
    def __init__(self, args: argparse.Namespace):
        self.log_file = Path(args.log_file)
        self.miner_pid_file = Path(args.miner_pid_file)
        self.state_file = Path(args.state_file)

        self.bot_token = args.bot_token
        self.chat_id = str(args.chat_id)

        self.poll_seconds = float(args.poll_seconds)
        self.flush_seconds = float(args.flush_seconds)
        self.heartbeat_seconds = float(args.heartbeat_seconds)
        self.no_traffic_alert_minutes = float(args.no_traffic_alert_minutes)
        self.no_traffic_alert_cooldown_minutes = float(args.no_traffic_alert_cooldown_minutes)
        self.max_lines_per_message = int(args.max_lines_per_message)
        self.max_chars_per_message = int(args.max_chars_per_message)
        self.port_health_alert_cooldown_minutes = float(args.port_health_alert_cooldown_minutes)

        self.hostname = socket.gethostname()
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        self.tail_state = self._load_state()
        self.pending_lines: List[str] = []

        self.last_flush = time.monotonic()
        self.next_heartbeat = time.monotonic() + self.heartbeat_seconds
        self.last_run_seen: Optional[float] = None
        self.no_traffic_reference = time.monotonic()
        self.next_no_traffic_alert = 0.0

        self.last_pid: Optional[int] = None
        self.last_running: Optional[bool] = None
        self.next_port_health_alert = 0.0

        self.port_monitor = PublicPortHealthMonitor(
            enabled=str_to_bool(args.port_health_enabled, default=True),
            axon_port=args.axon_port,
            external_ip=args.axon_external_ip.strip() or None,
            external_port=(args.axon_external_port or None),
            check_interval_seconds=args.port_health_check_seconds,
            failure_threshold=args.port_health_failure_threshold,
            recovery_threshold=args.port_health_recovery_threshold,
            connect_timeout_seconds=args.port_health_timeout_seconds,
            ip_discovery_urls=parse_csv_urls(args.port_health_ip_discovery_urls),
            port_check_url=args.port_health_check_url,
        )

        self.running = True

    def _load_state(self) -> TailState:
        if not self.state_file.exists():
            return TailState()
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            inode = data.get("inode")
            offset = int(data.get("offset", 0))
            return TailState(inode=inode, offset=max(offset, 0))
        except Exception:
            return TailState()

    def _save_state(self) -> None:
        payload = {"inode": self.tail_state.inode, "offset": self.tail_state.offset}
        self.state_file.write_text(json.dumps(payload), encoding="utf-8")

    def _strip_ansi(self, line: str) -> str:
        return ANSI_ESCAPE_RE.sub("", line).rstrip("\n")

    def _send_message(self, text: str) -> bool:
        if not text.strip():
            return True
        try:
            response = requests.post(
                self.api_url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if response.status_code != 200:
                print(
                    f"[telegram] send failed status={response.status_code} body={response.text[:300]}",
                    flush=True,
                )
                return False
            data = response.json()
            ok = bool(data.get("ok"))
            if not ok:
                print(f"[telegram] send failed payload={data}", flush=True)
            return ok
        except Exception as exc:
            print(f"[telegram] send exception: {exc}", flush=True)
            return False

    def _split_message(self, text: str) -> List[str]:
        if len(text) <= self.max_chars_per_message:
            return [text]

        out: List[str] = []
        lines = text.splitlines()
        chunk = ""
        for line in lines:
            candidate = f"{chunk}\n{line}".strip("\n") if chunk else line
            if len(candidate) <= self.max_chars_per_message:
                chunk = candidate
                continue
            if chunk:
                out.append(chunk)
            if len(line) <= self.max_chars_per_message:
                chunk = line
            else:
                start = 0
                while start < len(line):
                    out.append(line[start : start + self.max_chars_per_message])
                    start += self.max_chars_per_message
                chunk = ""
        if chunk:
            out.append(chunk)
        return out

    def _flush_pending(self, force: bool = False) -> None:
        if not self.pending_lines:
            return
        age = time.monotonic() - self.last_flush
        if not force and age < self.flush_seconds and len(self.pending_lines) < self.max_lines_per_message:
            return

        lines_to_send = self.pending_lines[: self.max_lines_per_message]
        self.pending_lines = self.pending_lines[self.max_lines_per_message :]
        self.last_flush = time.monotonic()

        header = f"SN54 Miner Log [{self.hostname}]"
        body = "\n".join(lines_to_send)
        text = f"{header}\n{body}"

        for part in self._split_message(text):
            self._send_message(part)

    @staticmethod
    def _is_alert_line(line: str) -> bool:
        lower = line.lower()
        alert_terms = (
            " error ",
            "error:",
            "traceback",
            "exception",
            " failed ",
            "critical",
            "not registered",
        )
        return any(term in lower for term in alert_terms)

    def _read_new_lines(self) -> List[str]:
        if not self.log_file.exists():
            return []

        stat = self.log_file.stat()
        inode = int(stat.st_ino)
        size = int(stat.st_size)

        if self.tail_state.inode is None:
            self.tail_state.inode = inode
            self.tail_state.offset = 0
        elif self.tail_state.inode != inode or size < self.tail_state.offset:
            self.tail_state.inode = inode
            self.tail_state.offset = 0

        with self.log_file.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self.tail_state.offset)
            data = f.read()
            self.tail_state.offset = f.tell()

        self._save_state()

        if not data:
            return []
        return [self._strip_ansi(line) for line in data.splitlines() if line.strip()]

    def _check_miner_process(self) -> Tuple[bool, Optional[int]]:
        pid: Optional[int] = None
        if self.miner_pid_file.exists():
            try:
                candidate = int(self.miner_pid_file.read_text(encoding="utf-8").strip())
                if Path(f"/proc/{candidate}").exists():
                    pid = candidate
            except Exception:
                pid = None

        if pid is None:
            pid = self._find_miner_pid_fallback()
            if pid is not None:
                try:
                    self.miner_pid_file.write_text(f"{pid}\n", encoding="utf-8")
                except OSError:
                    pass

        if pid is None:
            return False, None
        return True, pid

    def _find_miner_pid_fallback(self) -> Optional[int]:
        """Best-effort miner PID lookup when pid file is missing or stale."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "neurons/miner.py"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None

        pids: List[int] = []
        for token in result.stdout.split():
            try:
                pid = int(token.strip())
            except ValueError:
                continue
            if Path(f"/proc/{pid}").exists():
                pids.append(pid)
        if not pids:
            return None
        return max(pids)

    def _notify_state_change(self) -> None:
        running, pid = self._check_miner_process()
        if running == self.last_running and pid == self.last_pid:
            return

        self.last_running = running
        self.last_pid = pid

        if running:
            text = f"SN54 miner state update [{self.hostname}]\nMiner is RUNNING (pid={pid})"
            self.no_traffic_reference = time.monotonic()
            self.next_no_traffic_alert = 0.0
        else:
            text = f"SN54 miner state update [{self.hostname}]\nMiner is STOPPED"
        self._send_message(text)

    def _check_no_traffic_alert(self) -> None:
        if self.no_traffic_alert_minutes <= 0:
            return

        running, pid = self._check_miner_process()
        if not running:
            return

        now = time.monotonic()
        threshold_s = self.no_traffic_alert_minutes * 60.0
        cooldown_s = max(self.no_traffic_alert_cooldown_minutes * 60.0, threshold_s)
        since_last = now - self.no_traffic_reference

        if since_last < threshold_s:
            return

        if self.next_no_traffic_alert and now < self.next_no_traffic_alert:
            return

        text = (
            f"SN54 no-traffic alert [{self.hostname}]\n"
            f"Miner is RUNNING (pid={pid}) but no validator requests seen for "
            f"{int(since_last)}s.\n"
            f"Threshold: {self.no_traffic_alert_minutes} min\n"
            f"Check port reachability and validator intake."
        )
        if self._send_message(text):
            self.next_no_traffic_alert = now + cooldown_s

    def _format_port_health_line(self, snapshot: dict) -> str:
        state = str(snapshot.get("state", "unknown")).upper()
        target_ip = snapshot.get("target_ip") or "n/a"
        target_port = snapshot.get("target_port") or "n/a"
        reason = str(snapshot.get("reason", "n/a"))
        return f"public port: {state} ({target_ip}:{target_port}) - {reason}"

    def _check_port_health_alert(self) -> None:
        snapshot = self.port_monitor.snapshot_dict(force=False)
        state = str(snapshot.get("state", "unknown")).lower()
        transition = snapshot.get("transition")
        if state == "disabled":
            return

        now = time.monotonic()
        cooldown_seconds = max(60.0, self.port_health_alert_cooldown_minutes * 60.0)

        if transition == "unreachable":
            if now < self.next_port_health_alert:
                return
            local_listener = snapshot.get("local_listener")
            listener_msg = (
                "local miner socket is UP; likely NAT/port-forward/firewall issue."
                if local_listener
                else "local miner socket is DOWN; check miner process or bind port."
            )
            text = (
                f"SN54 public port alert [{self.hostname}]\n"
                f"State: UNREACHABLE\n"
                f"Target: {snapshot.get('target_ip', 'n/a')}:{snapshot.get('target_port', 'n/a')}\n"
                f"Detail: {snapshot.get('reason', 'n/a')}\n"
                f"Hint: {listener_msg}"
            )
            if self._send_message(text):
                self.next_port_health_alert = now + cooldown_seconds
            return

        if transition in {"recovered", "reachable"}:
            self.next_port_health_alert = 0.0
            text = (
                f"SN54 public port recovered [{self.hostname}]\n"
                f"State: REACHABLE\n"
                f"Target: {snapshot.get('target_ip', 'n/a')}:{snapshot.get('target_port', 'n/a')}\n"
                f"Detail: {snapshot.get('reason', 'n/a')}"
            )
            self._send_message(text)

    def _send_heartbeat(self) -> None:
        now = time.monotonic()
        if now < self.next_heartbeat:
            return
        self.next_heartbeat = now + self.heartbeat_seconds
        running, pid = self._check_miner_process()
        status = "RUNNING" if running else "STOPPED"
        port_snapshot = self.port_monitor.snapshot_dict(force=False)
        if self.last_run_seen is None:
            run_age = "never seen"
        else:
            run_age = f"{int(now - self.last_run_seen)}s ago"
        text = (
            f"SN54 notifier heartbeat [{self.hostname}]\n"
            f"Miner status: {status}\n"
            f"pid: {pid if pid else 'n/a'}\n"
            f"last request run: {run_age}\n"
            f"{self._format_port_health_line(port_snapshot)}\n"
            f"log file: {self.log_file}"
        )
        self._send_message(text)

    def start(self) -> None:
        initial_port = self.port_monitor.snapshot_dict(force=True)
        self._send_message(
            f"SN54 Telegram notifier started [{self.hostname}]\n"
            f"Watching: {self.log_file}\n"
            f"Flush every {self.flush_seconds}s\n"
            f"{self._format_port_health_line(initial_port)}"
        )

        while self.running:
            try:
                self._notify_state_change()

                new_lines = self._read_new_lines()
                for line in new_lines:
                    self.pending_lines.append(line)
                    if "Starting run " in line:
                        now = time.monotonic()
                        self.last_run_seen = now
                        self.no_traffic_reference = now
                        self.next_no_traffic_alert = 0.0
                    if self._is_alert_line(line):
                        self._flush_pending(force=True)

                self._flush_pending(force=False)
                self._check_no_traffic_alert()
                self._check_port_health_alert()
                self._send_heartbeat()

                time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                self._send_message(f"SN54 notifier internal error [{self.hostname}]\n{exc}")
                time.sleep(max(self.poll_seconds, 2.0))

        self._flush_pending(force=True)
        self._send_message(f"SN54 Telegram notifier stopped [{self.hostname}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram notifier for SN54 miner logs")
    parser.add_argument("--log-file", required=True, help="Miner log file path")
    parser.add_argument("--miner-pid-file", required=True, help="Miner PID file path")
    parser.add_argument("--state-file", required=True, help="State file path for file offset persistence")
    parser.add_argument("--axon-port", type=int, default=8091, help="Miner local axon port")
    parser.add_argument("--axon-external-ip", default="", help="Advertised external IP")
    parser.add_argument("--axon-external-port", type=int, default=0, help="Advertised external port")
    parser.add_argument(
        "--port-health-enabled",
        default="true",
        help="Enable periodic public port health checks (true/false)",
    )
    parser.add_argument(
        "--port-health-check-seconds",
        type=float,
        default=60.0,
        help="Seconds between external port checks",
    )
    parser.add_argument(
        "--port-health-failure-threshold",
        type=int,
        default=1,
        help="Consecutive failed checks before marking unreachable",
    )
    parser.add_argument(
        "--port-health-recovery-threshold",
        type=int,
        default=1,
        help="Consecutive successful checks before marking reachable",
    )
    parser.add_argument(
        "--port-health-timeout-seconds",
        type=float,
        default=8.0,
        help="Timeout for each external port check request",
    )
    parser.add_argument(
        "--port-health-alert-cooldown-minutes",
        type=float,
        default=30.0,
        help="Minimum minutes between repeated port-unreachable alerts",
    )
    parser.add_argument(
        "--port-health-ip-discovery-urls",
        default="https://api.ipify.org,https://ifconfig.me/ip",
        help="Comma-separated public IP discovery URLs",
    )
    parser.add_argument(
        "--port-health-check-url",
        default="https://ports.yougetsignal.com/check-port.php",
        help="External TCP port checker endpoint",
    )
    parser.add_argument("--bot-token", required=True, help="Telegram bot token")
    parser.add_argument("--chat-id", required=True, help="Telegram chat ID")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling interval")
    parser.add_argument("--flush-seconds", type=float, default=5.0, help="Batch flush interval")
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0, help="Heartbeat message interval")
    parser.add_argument(
        "--no-traffic-alert-minutes",
        type=float,
        default=20.0,
        help="Alert if no 'Starting run' log line is seen for this many minutes while miner is running. Set 0 to disable.",
    )
    parser.add_argument(
        "--no-traffic-alert-cooldown-minutes",
        type=float,
        default=30.0,
        help="Minimum minutes between repeated no-traffic alerts.",
    )
    parser.add_argument("--max-lines-per-message", type=int, default=60, help="Max log lines in one Telegram message")
    parser.add_argument("--max-chars-per-message", type=int, default=3500, help="Max text size in one Telegram message")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notifier = TelegramNotifier(args)

    def _stop(_signum, _frame):
        notifier.running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    notifier.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
