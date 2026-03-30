#!/usr/bin/env python3
"""Send miner log updates to Telegram."""

from __future__ import annotations

import argparse
import json
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from port_health import PublicPortHealthMonitor, parse_csv_urls, str_to_bool


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
WRAPPER_TS_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}:\s*")
BT_STYLE_RE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[\.,]\d+)?)\s*\|\s*([A-Za-z]+)\s*\|\s*(?:[^|]*\|\s*)?(.*)$"
)
STD_LEVEL_RE = re.compile(
    r"^\s*\[?(DEBUG|INFO|WARNING|ERROR|CRITICAL|SUCCESS|TRACE)\]?\s*[:\-]\s*(.*)$",
    re.IGNORECASE,
)
RUN_PATTERN = re.compile(r"Starting run (\d+) for (\d+) names")
PROCESSED_PATTERN = re.compile(r"Processed (\d+)/(\d+) names")
PROCESSED_STRUCTURED_PATTERN = re.compile(
    r"Processed (\d+) structured variations for (.+?) \(DOB non-empty: (\d+), Address non-empty: (\d+)\)"
)
GENERATING_NAME_PATTERN = re.compile(r"Generating variations for name: (.+?), remaining time: ([0-9.]+)s")
REQUEST_COMPLETED_PATTERN = re.compile(
    r"Request completed in ([0-9.]+)s of ([0-9.]+)s allowed\. Processed (\d+)/(\d+) names\."
)
PHASE4_GENERATING_PATTERN = re.compile(r"Phase 4: Generating (\d+) variations \(from validator: (.+)\)")
PHASE4_GENERATED_PATTERN = re.compile(r"Phase 4: Generated (\d+) S3 submissions")
PHASE4_SUB_PATTERN = re.compile(r"Phase 4: Successfully created (\d+) S3 submissions")
RUN_METRICS_PATTERN = re.compile(r"RUN_METRICS\s+(\{.*\})")
TELEGRAM_EVENT_PATTERN = re.compile(r"TELEGRAM_EVENT\s+(\{.*\})")
VERIFIED_CALL_PATTERN = re.compile(r"Verified call from (.+?) \(([^)]+)\)")
KNOWN_PROBE_NOISE_RE = re.compile(r"UnknownSynapseError.*Synapse name '' not found", re.IGNORECASE)
LONG_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/=]{180,}\b")
HUGGINGFACE_REQ_RE = re.compile(
    r"Request [0-9a-fA-F-]+:\s+(GET|HEAD)\s+https?://[^\s]*huggingface\.co",
    re.IGNORECASE,
)


@dataclass
class TailState:
    inode: Optional[int] = None
    offset: int = 0


@dataclass
class RunContext:
    run_id: Optional[int] = None
    requested_names: int = 0
    processed_names: int = 0
    validator_name: str = ""
    validator_hotkey: str = ""


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
        self.last_validator_call_seen: Optional[float] = None
        self.last_submission_summary = "none"
        self.no_traffic_reference = time.monotonic()
        self.next_no_traffic_alert = 0.0

        self.last_pid: Optional[int] = None
        self.last_running: Optional[bool] = None
        self.next_port_health_alert = 0.0

        self.current_run = RunContext()
        self.last_submission_event_run_id: Optional[int] = None

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

    @staticmethod
    def _strip_ansi(line: str) -> str:
        return ANSI_ESCAPE_RE.sub("", line).rstrip("\n")

    @staticmethod
    def _compact_ws(value: str) -> str:
        return " ".join(value.split())

    @staticmethod
    def _truncate(value: str, max_len: int = 260) -> str:
        if len(value) <= max_len:
            return value
        return f"{value[: max_len - 3]}..."

    @staticmethod
    def _is_known_probe_noise(text: str) -> bool:
        return bool(KNOWN_PROBE_NOISE_RE.search(text or ""))

    def _extract_log_message(self, line: str) -> Tuple[str, str]:
        raw = self._strip_ansi(line).strip()
        if not raw:
            return "", ""
        raw = WRAPPER_TS_PREFIX_RE.sub("", raw).strip()

        bt_match = BT_STYLE_RE.match(raw)
        if bt_match:
            level = bt_match.group(2).upper()
            message = bt_match.group(3).strip()
            return level, message

        parts = raw.split(" | ", 3)
        if len(parts) == 4 and re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", parts[0]):
            level = parts[1].strip().upper()
            message = parts[3].strip()
            return level, message

        level_match = STD_LEVEL_RE.match(raw)
        if level_match:
            level = level_match.group(1).upper()
            message = level_match.group(2).strip()
            return level, message

        return "", raw

    def _format_fail_reasons(self, fail_reasons: object) -> str:
        if not isinstance(fail_reasons, dict) or not fail_reasons:
            return "none"
        ordered = sorted(
            ((str(key), int(value)) for key, value in fail_reasons.items() if int(value) > 0),
            key=lambda item: (-item[1], item[0]),
        )
        if not ordered:
            return "none"
        return ", ".join(f"{key}:{value}" for key, value in ordered)

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
                # Telegram rate limiting: back off to avoid hammering the API.
                if response.status_code == 429:
                    retry_after_s: Optional[float] = None
                    try:
                        data = response.json()
                        params = data.get("parameters") if isinstance(data, dict) else None
                        if isinstance(params, dict) and params.get("retry_after") is not None:
                            retry_after_s = float(params.get("retry_after"))
                    except Exception:
                        retry_after_s = None

                    # Fallback: some responses are not reliably parseable as JSON.
                    if retry_after_s is None:
                        m = re.search(r"retry after\s*(\d+)", response.text or "")
                        if m:
                            retry_after_s = float(m.group(1))

                    # Best-effort: respect retry-after if present; otherwise just return.
                    if retry_after_s is not None and retry_after_s > 0:
                        # Cap to keep the notifier responsive.
                        retry_after_s = min(retry_after_s, 300.0)
                        print(
                            f"[telegram] send rate-limited status=429; sleeping {retry_after_s:.0f}s",
                            flush=True,
                        )
                        time.sleep(retry_after_s)

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

    def _send_long_message(self, text: str) -> None:
        for part in self._split_message(text):
            self._send_message(part)

    def _flush_pending(self, force: bool = False) -> None:
        if not self.pending_lines:
            return
        age = time.monotonic() - self.last_flush
        if not force and age < self.flush_seconds and len(self.pending_lines) < self.max_lines_per_message:
            return

        lines_to_send = self.pending_lines[: self.max_lines_per_message]
        self.pending_lines = self.pending_lines[self.max_lines_per_message :]
        self.last_flush = time.monotonic()

        line_word = "line" if len(lines_to_send) == 1 else "lines"
        header = f"SN54 notifier alerts [{self.hostname}] ({len(lines_to_send)} {line_word})"
        body = "\n".join(f"- {line}" for line in lines_to_send)
        self._send_long_message(f"{header}\n{body}")

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
            # On first notifier start, begin at EOF so we do not replay old logs.
            self.tail_state.offset = size
            self._save_state()
            return []
        if self.tail_state.inode != inode or size < self.tail_state.offset:
            self.tail_state.inode = inode
            self.tail_state.offset = 0

        with self.log_file.open("r", encoding="utf-8", errors="replace") as file_obj:
            file_obj.seek(self.tail_state.offset)
            data = file_obj.read()
            self.tail_state.offset = file_obj.tell()

        self._save_state()

        if not data:
            return []
        return [line for line in data.splitlines() if line.strip()]

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
            text = (
                f"SN54 miner state [{self.hostname}]\n"
                f"miner: RUNNING\n"
                f"pid: {pid if pid else 'n/a'}"
            )
            self.no_traffic_reference = time.monotonic()
            self.next_no_traffic_alert = 0.0
        else:
            text = (
                f"SN54 miner state [{self.hostname}]\n"
                "miner: STOPPED\n"
                "pid: n/a"
            )
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
            f"miner: RUNNING (pid={pid})\n"
            f"no validator requests seen for: {int(since_last)}s\n"
            f"threshold: {self.no_traffic_alert_minutes} min\n"
            "check public reachability and validator intake."
        )
        if self._send_message(text):
            self.next_no_traffic_alert = now + cooldown_s

    def _format_port_health_line(self, snapshot: Dict[str, object]) -> str:
        state = str(snapshot.get("state", "unknown")).upper()
        target_ip = snapshot.get("target_ip") or "n/a"
        target_port = snapshot.get("target_port") or "n/a"
        reason = str(snapshot.get("reason", "n/a"))
        return f"{state} ({target_ip}:{target_port}) - {reason}"

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
                "state: UNREACHABLE\n"
                f"target: {snapshot.get('target_ip', 'n/a')}:{snapshot.get('target_port', 'n/a')}\n"
                f"detail: {snapshot.get('reason', 'n/a')}\n"
                f"hint: {listener_msg}"
            )
            if self._send_message(text):
                self.next_port_health_alert = now + cooldown_seconds
            return

        if transition in {"recovered", "reachable"}:
            self.next_port_health_alert = 0.0
            text = (
                f"SN54 public port recovered [{self.hostname}]\n"
                "state: REACHABLE\n"
                f"target: {snapshot.get('target_ip', 'n/a')}:{snapshot.get('target_port', 'n/a')}\n"
                f"detail: {snapshot.get('reason', 'n/a')}"
            )
            self._send_message(text)

    @staticmethod
    def _format_age(last_seen: Optional[float]) -> str:
        if last_seen is None:
            return "never"
        return f"{int(max(0.0, time.monotonic() - last_seen))}s ago"

    def _send_heartbeat(self) -> None:
        now = time.monotonic()
        if now < self.next_heartbeat:
            return

        self.next_heartbeat = now + self.heartbeat_seconds
        running, pid = self._check_miner_process()
        status = "RUNNING" if running else "STOPPED"
        port_snapshot = self.port_monitor.snapshot_dict(force=False)

        text = (
            f"SN54 heartbeat [{self.hostname}]\n"
            f"time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"miner running check (5m): {status} (pid={pid if pid else 'n/a'})\n"
            f"health check (5m): {self._format_port_health_line(port_snapshot)}\n"
            f"last validator call: {self._format_age(self.last_validator_call_seen)}\n"
            f"last run seen: {self._format_age(self.last_run_seen)}\n"
            f"last submission: {self.last_submission_summary}\n"
            f"log file: {self.log_file}"
        )
        self._send_message(text)

    def _handle_telegram_event(self, message: str) -> bool:
        match = TELEGRAM_EVENT_PATTERN.search(message)
        if not match:
            return False

        try:
            envelope = json.loads(match.group(1))
        except json.JSONDecodeError:
            self.pending_lines.append("[ERROR] TELEGRAM_EVENT parse error")
            return True

        event = str(envelope.get("event", "") or "")
        payload = envelope.get("payload")
        payload_dict = payload if isinstance(payload, dict) else {}

        if event == "validator_request":
            run_id = int(payload_dict.get("run_id", 0) or 0)
            requested_names = int(payload_dict.get("requested_names", 0) or 0)

            self.current_run.run_id = run_id
            self.current_run.requested_names = requested_names
            self.current_run.processed_names = 0
            self.current_run.validator_name = str(payload_dict.get("validator_name", "") or "")
            self.current_run.validator_hotkey = str(payload_dict.get("validator_hotkey", "") or "")

            now = time.monotonic()
            self.last_run_seen = now
            self.last_validator_call_seen = now
            self.no_traffic_reference = now
            self.next_no_traffic_alert = 0.0

            self._send_long_message(self._format_validator_request(payload_dict))
            return True

        if event == "validator_submission_status":
            run_id = int(payload_dict.get("run_id", 0) or 0)
            self.last_submission_event_run_id = run_id
            self._send_long_message(self._format_submission_status(payload_dict))
            return True

        return True

    def _format_validator_request(self, payload: Dict[str, object]) -> str:
        run_id = payload.get("run_id", "?")
        validator_name = str(payload.get("validator_name", "unknown") or "unknown")
        validator_hotkey = str(payload.get("validator_hotkey", "unknown") or "unknown")
        timeout_seconds = payload.get("timeout_seconds", "?")
        requested_names = payload.get("requested_names", "?")

        lines: List[str] = [
            f"SN54 validator request [{self.hostname}]",
            f"run id: {run_id}",
            f"validator: {validator_name}",
            f"hotkey: {validator_hotkey}",
            f"timeout: {timeout_seconds}s",
            f"requested names: {requested_names}",
        ]

        identity_payload = payload.get("identity")
        if isinstance(identity_payload, list) and identity_payload:
            lines.append("identity payload:")
            for row in identity_payload:
                if not isinstance(row, dict):
                    continue
                index = row.get("index", "?")
                name = str(row.get("name", "") or "")
                dob = str(row.get("dob", "") or "")
                address = str(row.get("address", "") or "")
                lines.append(f"{index}. name={name} | dob={dob} | address={address}")
        else:
            lines.append("identity payload: none")

        query_template = str(payload.get("query_template", "") or "").strip()
        if query_template:
            lines.append("query template:")
            lines.append(query_template)

        image_request = payload.get("image_request")
        if isinstance(image_request, dict) and image_request:
            lines.append("image request:")
            lines.append(f"filename: {image_request.get('image_filename', '')}")
            lines.append(f"challenge_id: {image_request.get('challenge_id', '')}")
            lines.append(f"target_drand_round: {image_request.get('target_drand_round', 0)}")
            lines.append(f"reveal_timestamp: {image_request.get('reveal_timestamp', 0)}")
            lines.append(f"base_image_base64_chars: {image_request.get('base_image_base64_chars', 0)}")
            lines.append(f"base_image_sha256: {image_request.get('base_image_sha256', '')}")
            saved_seed = image_request.get("saved_seed_image")
            if isinstance(saved_seed, dict):
                lines.append(f"seed saved: {'yes' if saved_seed.get('saved') else 'no'}")
                if saved_seed.get("saved_image_path"):
                    lines.append(f"seed image path: {saved_seed.get('saved_image_path', '')}")
                if saved_seed.get("metadata_path"):
                    lines.append(f"seed metadata path: {saved_seed.get('metadata_path', '')}")
                if saved_seed.get("error"):
                    lines.append(f"seed save error: {saved_seed.get('error', '')}")

            variation_requests = image_request.get("variation_requests")
            if isinstance(variation_requests, list) and variation_requests:
                lines.append("variation requests:")
                for request in variation_requests:
                    if not isinstance(request, dict):
                        continue
                    lines.append(
                        f"{request.get('index', '?')}. type={request.get('type', '')} | "
                        f"intensity={request.get('intensity', '')} | "
                        f"description={request.get('description', '')} | "
                        f"detail={request.get('detail', '')}"
                    )

        return "\n".join(lines)

    def _format_submission_status(self, payload: Dict[str, object]) -> str:
        run_id = payload.get("run_id", "?")
        submitted = bool(payload.get("submitted_to_validator", False))
        requested = int(payload.get("requested_names", 0) or 0)
        returned = int(payload.get("returned_names", 0) or 0)
        s3_submissions = int(payload.get("s3_submissions", 0) or 0)
        note = str(payload.get("note", "") or "")
        fail_summary = self._format_fail_reasons(payload.get("fail_reasons"))

        submitted_label = "YES" if submitted else "NO"
        self.last_submission_summary = (
            f"run_id={run_id}, submitted={submitted_label}, returned={returned}/{requested}, s3={s3_submissions}"
        )

        lines = [
            f"SN54 miner submission [{self.hostname}]",
            f"run id: {run_id}",
            f"submitted to validator: {submitted_label}",
            f"returned names: {returned}/{requested}",
            f"s3 submissions: {s3_submissions}",
            f"fail reasons: {fail_summary}",
        ]
        if note:
            lines.append(f"note: {note}")
        return "\n".join(lines)

    def _handle_lifecycle_message(self, level: str, message: str) -> bool:
        if self._handle_telegram_event(message):
            return True

        compact = self._compact_ws(message)

        verified_match = VERIFIED_CALL_PATTERN.search(compact)
        if verified_match:
            validator_name = verified_match.group(1).strip()
            validator_hotkey = verified_match.group(2).strip()
            now = time.monotonic()
            self.last_validator_call_seen = now
            self.no_traffic_reference = now
            self.next_no_traffic_alert = 0.0
            self._send_message(
                f"SN54 validator call verified [{self.hostname}]\n"
                f"validator: {validator_name}\n"
                f"hotkey: {validator_hotkey}\n"
                "status: VERIFIED"
            )
            return True

        run_match = RUN_PATTERN.search(compact)
        if run_match:
            run_id = int(run_match.group(1))
            requested_names = int(run_match.group(2))
            self.current_run.run_id = run_id
            self.current_run.requested_names = requested_names
            self.current_run.processed_names = 0

            now = time.monotonic()
            self.last_run_seen = now
            self.no_traffic_reference = now
            self.next_no_traffic_alert = 0.0

            self._send_message(
                f"SN54 miner work start [{self.hostname}]\n"
                f"run id: {run_id}\n"
                f"requested names: {requested_names}"
            )
            return True

        generating_match = GENERATING_NAME_PATTERN.search(compact)
        if generating_match:
            name = generating_match.group(1).strip()
            remaining = generating_match.group(2)
            run_id = self.current_run.run_id if self.current_run.run_id is not None else "?"
            self._send_message(
                f"SN54 miner working [{self.hostname}]\n"
                f"run id: {run_id}\n"
                "stage: generating variations\n"
                f"seed name: {name}\n"
                f"remaining timeout: {remaining}s"
            )
            return True

        structured_match = PROCESSED_STRUCTURED_PATTERN.search(compact)
        if structured_match:
            variation_count = int(structured_match.group(1))
            seed_name = structured_match.group(2).strip()
            dob_non_empty = int(structured_match.group(3))
            address_non_empty = int(structured_match.group(4))

            self.current_run.processed_names += 1
            run_id = self.current_run.run_id if self.current_run.run_id is not None else "?"
            requested_names = self.current_run.requested_names or "?"

            self._send_message(
                f"SN54 miner work progress [{self.hostname}]\n"
                f"run id: {run_id}\n"
                f"processed names: {self.current_run.processed_names}/{requested_names}\n"
                f"seed name: {seed_name}\n"
                f"name variations: {variation_count}\n"
                f"dob non-empty: {dob_non_empty}\n"
                f"address non-empty: {address_non_empty}"
            )
            return True

        phase4_generating_match = PHASE4_GENERATING_PATTERN.search(compact)
        if phase4_generating_match:
            count = int(phase4_generating_match.group(1))
            requested = phase4_generating_match.group(2).strip()
            run_id = self.current_run.run_id if self.current_run.run_id is not None else "?"
            self._send_long_message(
                f"SN54 phase4 work [{self.hostname}]\n"
                f"run id: {run_id}\n"
                f"requested image variations: {count}\n"
                f"validator request details: {requested}"
            )
            return True

        phase4_generated_match = PHASE4_GENERATED_PATTERN.search(compact)
        if phase4_generated_match:
            generated = int(phase4_generated_match.group(1))
            run_id = self.current_run.run_id if self.current_run.run_id is not None else "?"
            self._send_message(
                f"SN54 phase4 result [{self.hostname}]\n"
                f"run id: {run_id}\n"
                f"generated s3 submissions: {generated}"
            )
            return True

        phase4_sub_match = PHASE4_SUB_PATTERN.search(compact)
        if phase4_sub_match:
            created = int(phase4_sub_match.group(1))
            run_id = self.current_run.run_id if self.current_run.run_id is not None else "?"
            self._send_message(
                f"SN54 phase4 summary [{self.hostname}]\n"
                f"run id: {run_id}\n"
                f"successfully created submissions: {created}"
            )
            return True

        completed_match = REQUEST_COMPLETED_PATTERN.search(compact)
        if completed_match:
            elapsed = completed_match.group(1)
            timeout = completed_match.group(2)
            processed = completed_match.group(3)
            requested = completed_match.group(4)
            run_id = self.current_run.run_id if self.current_run.run_id is not None else "?"
            self._send_message(
                f"SN54 miner processing summary [{self.hostname}]\n"
                f"run id: {run_id}\n"
                f"elapsed: {elapsed}s / {timeout}s\n"
                f"processed names: {processed}/{requested}"
            )
            return True

        processed_match = PROCESSED_PATTERN.search(compact)
        if processed_match:
            return True

        metrics_match = RUN_METRICS_PATTERN.search(compact)
        if metrics_match:
            try:
                payload = json.loads(metrics_match.group(1))
            except json.JSONDecodeError:
                self.pending_lines.append("[ERROR] RUN_METRICS parse error")
                return True

            run_id = int(payload.get("run_id", 0) or 0)
            if self.last_submission_event_run_id is not None and run_id == self.last_submission_event_run_id:
                return True

            requested = int(payload.get("requested_names", 0) or 0)
            returned = int(payload.get("returned_names", 0) or 0)
            s3_subs = int(payload.get("s3_submissions", 0) or 0)
            fail_summary = self._format_fail_reasons(payload.get("fail_reasons"))

            self.last_submission_summary = (
                f"run_id={run_id}, submitted=YES, returned={returned}/{requested}, s3={s3_subs}"
            )

            self._send_message(
                f"SN54 miner result [{self.hostname}]\n"
                f"run id: {run_id}\n"
                f"requested names: {requested}\n"
                f"returned names: {returned}\n"
                f"s3 submissions: {s3_subs}\n"
                f"fail reasons: {fail_summary}\n"
                "submitted to validator: YES (synapse returned by miner)"
            )
            return True

        return False

    def _is_noise_message(self, text: str) -> bool:
        compact = self._compact_ws(text)

        if self._is_known_probe_noise(compact):
            return True
        if HUGGINGFACE_REQ_RE.search(compact):
            return True
        if compact.startswith(("Loading pipeline components", "Loading weights", "Fetching ")):
            return True
        if compact.startswith(("Downloading '", "Download complete.", "Creating pointer from ")):
            return True
        if compact.startswith(("Instantiating ", "Updating config from ", "Loaded ")):
            return True
        if compact.startswith(("Model config ", "Generate config ", "loading configuration file ")):
            return True
        if compact.startswith("Xet Storage is enabled for this repo."):
            return True
        if LONG_BLOB_RE.search(compact):
            return True
        return False

    def _format_fallback_line(self, level: str, message: str) -> str:
        if not message:
            return ""
        if self._is_noise_message(message):
            return ""

        compact = self._compact_ws(message)
        if not compact:
            return ""

        if level in {"ERROR", "CRITICAL", "WARNING"}:
            return f"[{level}] {self._truncate(compact, 300)}"

        if self._is_alert_line(compact):
            return self._truncate(compact, 300)

        # Suppress non-critical generic info lines to keep Telegram readable.
        return ""

    def start(self) -> None:
        initial_port = self.port_monitor.snapshot_dict(force=True)
        self._send_message(
            f"SN54 Telegram notifier started [{self.hostname}]\n"
            f"watching: {self.log_file}\n"
            f"poll interval: {self.poll_seconds}s\n"
            f"heartbeat interval: {self.heartbeat_seconds}s\n"
            f"public port health: {self._format_port_health_line(initial_port)}"
        )

        while self.running:
            try:
                self._notify_state_change()

                new_lines = self._read_new_lines()
                for line in new_lines:
                    level, message = self._extract_log_message(line)
                    if not message:
                        continue

                    if self._handle_lifecycle_message(level, message):
                        continue

                    formatted_line = self._format_fallback_line(level, message)
                    if formatted_line:
                        self.pending_lines.append(formatted_line)
                    if formatted_line and self._is_alert_line(formatted_line):
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
