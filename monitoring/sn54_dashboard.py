#!/usr/bin/env python3
"""Local dashboard for MIID Subnet 54 miner operations."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from port_health import PublicPortHealthMonitor, parse_csv_urls, str_to_bool


RUN_PATTERN = re.compile(r"Starting run (\d+) for (\d+) names")
PROCESSED_PATTERN = re.compile(r"Processed (\d+)/(\d+) names")
PHASE4_PATTERN = re.compile(r"Phase 4: Successfully created (\d+) S3 submissions")
LINE_TS_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
TELEGRAM_EVENT_PATTERN = re.compile(r"TELEGRAM_EVENT (\{.*\})")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def human_duration(seconds: Optional[int]) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 0:
        return "n/a"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_tail_lines(path: Path, max_bytes: int = 2_000_000) -> List[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start, os.SEEK_SET)
        data = f.read().decode("utf-8", errors="replace")
    return data.splitlines()


def find_miner_pid_fallback() -> Optional[int]:
    """Best-effort miner PID lookup when pid file is missing or stale."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "neurons/miner.py"],
            capture_output=True,
            text=True,
            check=False,
        )
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
    except OSError:
        return None


def get_pid_status(pid_file: Path) -> Dict[str, object]:
    details: Dict[str, object] = {
        "running": False,
        "pid": None,
        "uptime_seconds": None,
        "uptime_human": "n/a",
        "cmdline": "",
        "pid_source": "pid_file",
    }

    pid: Optional[int] = None
    if pid_file.exists():
        try:
            pid_candidate = int(pid_file.read_text(encoding="utf-8").strip())
            if Path(f"/proc/{pid_candidate}").exists():
                pid = pid_candidate
        except (ValueError, OSError):
            pid = None

    if pid is None:
        pid = find_miner_pid_fallback()
        if pid is not None:
            details["pid_source"] = "process_scan"
            try:
                pid_file.write_text(f"{pid}\n", encoding="utf-8")
            except OSError:
                pass
        else:
            return details

    details["pid"] = pid
    proc_dir = Path(f"/proc/{pid}")
    if not proc_dir.exists():
        return details

    details["running"] = True

    try:
        cmdline = (proc_dir / "cmdline").read_text(encoding="utf-8", errors="replace")
        details["cmdline"] = cmdline.replace("\x00", " ").strip()
    except OSError:
        details["cmdline"] = ""

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etimes="],
            capture_output=True,
            text=True,
            check=False,
        )
        uptime = int(result.stdout.strip()) if result.stdout.strip() else None
    except (ValueError, OSError):
        uptime = None

    details["uptime_seconds"] = uptime
    details["uptime_human"] = human_duration(uptime)
    return details


def parse_timestamp_from_line(line: str) -> Optional[str]:
    match = LINE_TS_PATTERN.search(line)
    if not match:
        return None
    raw = match.group(1).replace(" ", "T")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_telegram_event(line: str) -> Optional[Tuple[str, Dict[str, object]]]:
    match = TELEGRAM_EVENT_PATTERN.search(line)
    if not match:
        return None
    try:
        envelope = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    event = str(envelope.get("event", "") or "")
    payload = envelope.get("payload")
    return event, (payload if isinstance(payload, dict) else {})


def _fail_reason_summary(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        return "none"
    parts = [f"{k}={int(v)}" for k, v in sorted(payload.items()) if int(v) > 0]
    return ", ".join(parts) if parts else "none"


def _recent_event_entry(line: str) -> Optional[Dict[str, str]]:
    ts = parse_timestamp_from_line(line) or ""
    low = line.lower()
    te = _parse_telegram_event(line)
    if te is not None:
        event, payload = te
        if event == "validator_request":
            validator_name = str(payload.get("validator_name", "unknown") or "unknown")
            run_id = payload.get("run_id", "n/a")
            image_request = payload.get("image_request")
            challenge = ""
            req_count = 0
            if isinstance(image_request, dict):
                challenge = str(image_request.get("challenge_id", "") or "")
                req_count = int(image_request.get("requested_variations", 0) or 0)
            return {
                "timestamp_utc": ts,
                "level": "info",
                "title": "Validator Request",
                "detail": f"{validator_name} | run {run_id} | challenge {challenge or 'n/a'} | image variations {req_count}",
            }
        if event == "phase4_variation_preview":
            label = str(payload.get("label", "") or "variation")
            attempt = int(payload.get("attempt", 0) or 0)
            primary = payload.get("primary_score")
            final = payload.get("final_score")
            req_index = int(payload.get("request_index", 0) or 0)
            return {
                "timestamp_utc": ts,
                "level": "info",
                "title": "Generated Preview",
                "detail": (
                    f"req #{req_index} | {label} | attempt {attempt} | "
                    f"primary {primary if primary is not None else 'n/a'} | "
                    f"final {final if final is not None else 'n/a'}"
                ),
            }
        if event == "validator_submission_status":
            run_id = payload.get("run_id", "n/a")
            returned = int(payload.get("returned_names", 0) or 0)
            requested = int(payload.get("requested_names", 0) or 0)
            s3_submissions = int(payload.get("s3_submissions", 0) or 0)
            return {
                "timestamp_utc": ts,
                "level": "info",
                "title": "Submission Status",
                "detail": (
                    f"run {run_id} | returned {returned}/{requested} | "
                    f"s3 submissions {s3_submissions} | failures {_fail_reason_summary(payload.get('fail_reasons'))}"
                ),
            }

    run_match = RUN_PATTERN.search(line)
    if run_match:
        return {
            "timestamp_utc": ts,
            "level": "info",
            "title": "Run Started",
            "detail": f"run {run_match.group(1)} for {run_match.group(2)} names",
        }
    if "verified call from" in low:
        return {
            "timestamp_utc": ts,
            "level": "info",
            "title": "Validator Verified",
            "detail": line.strip(),
        }
    if "warning" in low:
        return {
            "timestamp_utc": ts,
            "level": "warn",
            "title": "Warning",
            "detail": line.strip(),
        }
    if "error" in low:
        return {
            "timestamp_utc": ts,
            "level": "error",
            "title": "Error",
            "detail": line.strip(),
        }
    if "phase 4:" in low:
        return {
            "timestamp_utc": ts,
            "level": "info",
            "title": "Phase 4",
            "detail": line.strip(),
        }
    return None


def _fresh_current_request_from_payload(payload: Dict[str, object]) -> Dict[str, object]:
    image_request = payload.get("image_request")
    image_request_dict = image_request if isinstance(image_request, dict) else {}
    saved_seed = image_request_dict.get("saved_seed_image")
    saved_seed_dict = saved_seed if isinstance(saved_seed, dict) else {}
    return {
        "run_id": int(payload.get("run_id", 0) or 0),
        "validator_name": str(payload.get("validator_name", "") or ""),
        "validator_hotkey": str(payload.get("validator_hotkey", "") or ""),
        "challenge_id": str(image_request_dict.get("challenge_id", "") or ""),
        "image_filename": str(image_request_dict.get("image_filename", "") or ""),
        "requested_variations": list(image_request_dict.get("variation_requests", []) or []),
        "seed_image_path": str(saved_seed_dict.get("saved_image_path", "") or ""),
        "seed_metadata_path": str(saved_seed_dict.get("metadata_path", "") or ""),
        "seed_saved": bool(saved_seed_dict.get("saved", False)),
        "generated_previews": [],
        "submission_status": None,
    }


def summarize_logs(lines: List[str]) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "runs_total": 0,
        "requested_names_total": 0,
        "processed_names_total": 0,
        "phase4_submissions_total": 0,
        "validator_calls_verified": 0,
        "errors_total": 0,
        "warnings_total": 0,
        "last_run_id": None,
        "last_activity_line": "",
        "last_activity_ts": None,
        "recent_events": [],
        "current_request": None,
    }

    recent_events: List[Dict[str, str]] = []
    current_request: Optional[Dict[str, object]] = None
    preview_map: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    interesting_terms = (
        "starting run",
        "processed ",
        "phase 4:",
        "verified call from",
        "warning",
        "error",
    )

    for line in lines:
        striped = line.strip()
        if striped:
            metrics["last_activity_line"] = striped
            metrics["last_activity_ts"] = parse_timestamp_from_line(striped) or metrics["last_activity_ts"]

        low = line.lower()

        match = RUN_PATTERN.search(line)
        if match:
            metrics["runs_total"] += 1
            metrics["requested_names_total"] += int(match.group(2))
            metrics["last_run_id"] = int(match.group(1))

        match = PROCESSED_PATTERN.search(line)
        if match:
            metrics["processed_names_total"] += int(match.group(1))

        match = PHASE4_PATTERN.search(line)
        if match:
            metrics["phase4_submissions_total"] += int(match.group(1))

        if "verified call from" in low:
            metrics["validator_calls_verified"] += 1
        if "error" in low:
            metrics["errors_total"] += 1
        if "warning" in low:
            metrics["warnings_total"] += 1

        parsed_event = _recent_event_entry(line)
        if parsed_event is not None:
            recent_events.append(parsed_event)
        elif any(term in low for term in interesting_terms):
            recent_events.append(
                {
                    "timestamp_utc": parse_timestamp_from_line(striped) or "",
                    "level": "info",
                    "title": "Log",
                    "detail": striped,
                }
            )

        te = _parse_telegram_event(line)
        if te is not None:
            event, payload = te
            if event == "validator_request":
                current_request = _fresh_current_request_from_payload(payload)
                preview_map = OrderedDict()
            elif event == "phase4_variation_preview" and current_request is not None:
                run_id = int(payload.get("run_id", 0) or 0)
                challenge_id = str(payload.get("challenge_id", "") or "")
                current_challenge = str(current_request.get("challenge_id", "") or "")
                if challenge_id == current_challenge or run_id == int(current_request.get("run_id", 0) or 0):
                    key = f"{int(payload.get('request_index', 0) or 0)}::{str(payload.get('label', '') or '')}"
                    preview_map[key] = {
                        "request_index": int(payload.get("request_index", 0) or 0),
                        "label": str(payload.get("label", "") or ""),
                        "attempt": int(payload.get("attempt", 0) or 0),
                        "primary_score": payload.get("primary_score"),
                        "final_score": payload.get("final_score"),
                        "candidate_count": int(payload.get("candidate_count", 0) or 0),
                        "selected_candidate_index": int(payload.get("selected_candidate_index", 0) or 0),
                        "image_path": str(payload.get("image_path", "") or ""),
                        "metadata_path": str(payload.get("metadata_path", "") or ""),
                        "image_hash": str(payload.get("image_hash", "") or ""),
                    }
            elif event == "validator_submission_status" and current_request is not None:
                run_id = int(payload.get("run_id", 0) or 0)
                if run_id == int(current_request.get("run_id", 0) or 0):
                    current_request["submission_status"] = {
                        "returned_names": int(payload.get("returned_names", 0) or 0),
                        "requested_names": int(payload.get("requested_names", 0) or 0),
                        "s3_submissions": int(payload.get("s3_submissions", 0) or 0),
                        "note": str(payload.get("note", "") or ""),
                        "fail_reasons": _fail_reason_summary(payload.get("fail_reasons")),
                    }

    metrics["recent_events"] = recent_events[-20:]
    if current_request is not None:
        previews = list(preview_map.values())
        previews.sort(key=lambda item: (int(item.get("request_index", 0)), str(item.get("label", ""))))
        current_request["generated_previews"] = previews
    metrics["current_request"] = current_request
    return metrics


class DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        handler,
        log_file: Path,
        pid_file: Path,
        port_monitor: PublicPortHealthMonitor,
    ):
        super().__init__(server_address, handler)
        self.log_file = log_file
        self.pid_file = pid_file
        self.port_monitor = port_monitor
        self.project_root = Path(__file__).resolve().parent.parent

    def collect_status(self) -> Dict[str, object]:
        pid_status = get_pid_status(self.pid_file)
        lines = read_tail_lines(self.log_file)
        log_summary = summarize_logs(lines)
        log_exists = self.log_file.exists()
        log_size = self.log_file.stat().st_size if log_exists else 0
        log_mtime = self.log_file.stat().st_mtime if log_exists else None
        port_health = self.port_monitor.snapshot_dict(force=False)

        return {
            "now_utc": utc_now_iso(),
            "miner": pid_status,
            "log": {
                "path": str(self.log_file),
                "exists": log_exists,
                "size_bytes": log_size,
                "size_human": human_bytes(log_size),
                "updated_utc": datetime.fromtimestamp(log_mtime, timezone.utc).isoformat() if log_mtime else None,
                "tail_line_count": len(lines),
                "tail_preview": lines[-80:],
            },
            "metrics": log_summary,
            "port_health": port_health,
        }


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def _send_json(self, data: Dict[str, object], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
            resolved.relative_to(self.server.project_root)
            if not resolved.is_file():
                raise FileNotFoundError(str(resolved))
            data = resolved.read_bytes()
        except Exception:
            self._send_json({"error": "Not found"}, status=404)
            return

        suffix = resolved.suffix.lower()
        if suffix in {".png"}:
            mime = "image/png"
        elif suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix in {".webp"}:
            mime = "image/webp"
        elif suffix in {".gif"}:
            mime = "image/gif"
        else:
            mime = "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self._send_json(self.server.collect_status())
            return
        if path == "/media":
            query = parse_qs(parsed.query)
            requested = ""
            if query.get("path"):
                requested = str(query["path"][0] or "")
            if not requested:
                self._send_json({"error": "Missing path"}, status=400)
                return
            self._send_file(Path(requested))
            return
        if path in ("/", "/index.html"):
            self._send_html(self._render_html())
            return
        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, fmt: str, *args) -> None:
        return

    @staticmethod
    def _render_html() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SN54 Miner Dashboard</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: rgba(255, 255, 255, 0.82);
      --ink: #151515;
      --subtle: #596273;
      --accent: #0f7b6c;
      --accent-2: #df6d14;
      --ok: #198754;
      --bad: #c1121f;
      --warn: #b9770e;
      --line: rgba(21, 21, 21, 0.12);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Nunito Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 480px at 10% -10%, rgba(15,123,108,0.22), transparent 60%),
        radial-gradient(1000px 520px at 100% 0%, rgba(223,109,20,0.20), transparent 62%),
        var(--bg);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px 18px 40px;
    }
    .hero {
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }
    .headline {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.45rem, 2.6vw, 2.2rem);
      font-weight: 800;
      letter-spacing: 0.01em;
    }
    .badge {
      border-radius: 999px;
      padding: 6px 12px;
      font-weight: 700;
      font-size: 0.82rem;
      letter-spacing: 0.02em;
      color: #fff;
      background: var(--warn);
    }
    .meta {
      color: var(--subtle);
      font-size: 0.95rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 14px 15px;
      backdrop-filter: blur(3px);
      box-shadow: 0 8px 28px rgba(0,0,0,0.05);
    }
    .card h2 {
      margin: 0 0 8px;
      font-size: 0.92rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--subtle);
    }
    .value {
      font-size: 1.65rem;
      font-weight: 800;
      line-height: 1.1;
    }
    .hint {
      margin-top: 4px;
      color: var(--subtle);
      font-size: 0.9rem;
      word-break: break-word;
    }
    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-6 { grid-column: span 6; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .events, .tail {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }
    .events td {
      border-top: 1px solid var(--line);
      padding: 8px 0;
      vertical-align: top;
      color: #2f3744;
      word-break: break-word;
    }
    .event-row td:first-child {
      width: 128px;
      color: var(--subtle);
      font-size: 0.82rem;
      padding-right: 12px;
      white-space: nowrap;
    }
    .event-title {
      font-weight: 800;
      margin-bottom: 3px;
    }
    .event-detail {
      color: #324050;
      line-height: 1.35;
    }
    .event-warn .event-title { color: var(--warn); }
    .event-error .event-title { color: var(--bad); }
    .tail {
      background: #121821;
      color: #d8e5ff;
      border-radius: 12px;
      max-height: 260px;
      overflow: auto;
      padding: 12px;
      font-family: "JetBrains Mono", "SFMono-Regular", Menlo, monospace;
      line-height: 1.35;
    }
    .tail-line {
      white-space: pre-wrap;
      margin: 0;
    }
    .request-meta {
      display: grid;
      gap: 6px;
      color: #2f3744;
      font-size: 0.95rem;
      margin-bottom: 12px;
    }
    .request-meta strong {
      color: var(--ink);
    }
    .variation-list {
      margin: 0;
      padding-left: 18px;
      color: #324050;
    }
    .visual-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .image-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: rgba(255,255,255,0.72);
    }
    .image-card img {
      width: 100%;
      aspect-ratio: 3 / 4;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: rgba(0,0,0,0.05);
      display: block;
    }
    .image-card h3 {
      margin: 10px 0 6px;
      font-size: 0.98rem;
    }
    .mini {
      color: var(--subtle);
      font-size: 0.84rem;
      line-height: 1.35;
      word-break: break-word;
    }
    .placeholder {
      color: var(--subtle);
      font-style: italic;
    }
    @media (max-width: 980px) {
      .span-3 { grid-column: span 6; }
      .span-4 { grid-column: span 6; }
      .span-6 { grid-column: span 12; }
      .span-8 { grid-column: span 12; }
    }
    @media (max-width: 640px) {
      .span-3, .span-4 { grid-column: span 12; }
      .wrap { padding: 14px 12px 28px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="headline">
        <h1>MIID Subnet 54 Miner Console</h1>
        <span id="minerBadge" class="badge">CHECKING</span>
      </div>
      <div id="lastRefreshed" class="meta">Loading status...</div>
    </section>

    <section class="grid">
      <article class="card span-3">
        <h2>Miner Uptime</h2>
        <div id="uptime" class="value">n/a</div>
        <div id="pid" class="hint">PID: n/a</div>
      </article>
      <article class="card span-3">
        <h2>Total Runs</h2>
        <div id="runsTotal" class="value">0</div>
        <div id="lastRun" class="hint">Last run: n/a</div>
      </article>
      <article class="card span-3">
        <h2>Names Throughput</h2>
        <div id="processedNames" class="value">0</div>
        <div id="requestedNames" class="hint">Requested: 0</div>
      </article>
      <article class="card span-3">
        <h2>Phase 4 Subs</h2>
        <div id="phase4" class="value">0</div>
        <div id="verifiedCalls" class="hint">Verified validator calls: 0</div>
      </article>

      <article class="card span-3">
        <h2>Health</h2>
        <div id="errorCount" class="value">0</div>
        <div id="warningCount" class="hint">Warnings: 0</div>
      </article>
      <article class="card span-3">
        <h2>Public Port</h2>
        <div id="portState" class="value">UNKNOWN</div>
        <div id="portTarget" class="hint">Target: n/a</div>
        <div id="portDetail" class="hint">Last check: n/a</div>
      </article>
      <article class="card span-6">
        <h2>Log File</h2>
        <div id="logPath" class="hint">n/a</div>
        <div id="logMeta" class="hint">n/a</div>
        <div id="lastActivity" class="hint">Last activity: n/a</div>
      </article>

      <article class="card span-12">
        <h2>Current Validator Request</h2>
        <div id="currentRequestMeta" class="request-meta">No validator image request seen yet.</div>
        <div class="visual-grid">
          <section class="image-card">
            <h3>Seed Image</h3>
            <div id="seedPanel" class="placeholder">Waiting for the next validator seed image.</div>
          </section>
          <section class="image-card" style="grid-column: span 2;">
            <h3>Generated Images Before Submission</h3>
            <div id="generatedPanel" class="visual-grid">
              <div class="placeholder">Waiting for generated previews.</div>
            </div>
          </section>
        </div>
      </article>

      <article class="card span-6">
        <h2>Recent Events</h2>
        <table class="events"><tbody id="eventsBody"></tbody></table>
      </article>
      <article class="card span-6">
        <h2>Live Log Tail</h2>
        <div id="tailBody" class="tail"></div>
      </article>
    </section>
  </div>

  <script>
    const REFRESH_MS = 5000;

    function setBadge(running) {
      const badge = document.getElementById("minerBadge");
      badge.textContent = running ? "RUNNING" : "STOPPED";
      badge.style.background = running ? "var(--ok)" : "var(--bad)";
    }

    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }

    function setPortState(state) {
      const el = document.getElementById("portState");
      if (!el) return;
      const low = (state || "unknown").toLowerCase();
      el.textContent = low.toUpperCase();
      if (low === "reachable" || low === "open") {
        el.style.color = "var(--ok)";
      } else if (low === "unreachable" || low === "closed") {
        el.style.color = "var(--bad)";
      } else if (low === "disabled") {
        el.style.color = "var(--subtle)";
      } else {
        el.style.color = "var(--warn)";
      }
    }

    function renderEvents(events) {
      const body = document.getElementById("eventsBody");
      body.innerHTML = "";
      if (!events || events.length === 0) {
        body.innerHTML = "<tr><td>No events yet.</td></tr>";
        return;
      }
      for (const event of events.slice().reverse()) {
        const tr = document.createElement("tr");
        tr.className = "event-row event-" + (event.level || "info");
        const when = document.createElement("td");
        when.textContent = event.timestamp_utc
          ? new Date(event.timestamp_utc).toLocaleTimeString()
          : "n/a";
        const td = document.createElement("td");
        const title = document.createElement("div");
        title.className = "event-title";
        title.textContent = event.title || "Event";
        const detail = document.createElement("div");
        detail.className = "event-detail";
        detail.textContent = event.detail || "";
        td.appendChild(title);
        td.appendChild(detail);
        tr.appendChild(when);
        tr.appendChild(td);
        body.appendChild(tr);
      }
    }

    function mediaUrl(path) {
      return "/media?path=" + encodeURIComponent(path);
    }

    function renderSeedPanel(current) {
      const seedPanel = document.getElementById("seedPanel");
      const existingImg = seedPanel.querySelector("img");
      const nextPath = current && current.seed_saved && current.seed_image_path
        ? mediaUrl(current.seed_image_path)
        : "";

      if (!nextPath) {
        seedPanel.innerHTML = "";
        seedPanel.textContent = "Seed image not saved for this request.";
        seedPanel.className = "placeholder";
        return;
      }

      seedPanel.className = "";
      if (!existingImg || existingImg.getAttribute("src") !== nextPath) {
        seedPanel.innerHTML = "";
        const img = document.createElement("img");
        img.src = nextPath;
        img.alt = "Validator seed image";
        seedPanel.appendChild(img);
      }

      let info = seedPanel.querySelector(".mini");
      const infoText = current.seed_metadata_path || current.seed_image_path || "";
      if (!info) {
        info = document.createElement("div");
        info.className = "mini";
        seedPanel.appendChild(info);
      }
      info.textContent = infoText;
    }

    function renderGeneratedPanel(previews) {
      const generatedPanel = document.getElementById("generatedPanel");
      const normalized = Array.isArray(previews) ? previews : [];
      if (normalized.length === 0) {
        if (generatedPanel.dataset.empty !== "true") {
          generatedPanel.innerHTML = '<div class="placeholder">Waiting for generated previews.</div>';
          generatedPanel.dataset.empty = "true";
        }
        return;
      }
      generatedPanel.dataset.empty = "false";

      const desiredKeys = normalized.map(
        (preview) => `${preview.request_index || 0}::${preview.label || ""}`
      );

      for (const child of Array.from(generatedPanel.children)) {
        if (child.classList && child.classList.contains("placeholder")) {
          child.remove();
          continue;
        }
        const key = child.dataset ? child.dataset.key : "";
        if (key && !desiredKeys.includes(key)) {
          child.remove();
        }
      }

      for (const preview of normalized) {
        const key = `${preview.request_index || 0}::${preview.label || ""}`;
        let card = generatedPanel.querySelector(`.image-card[data-key="${CSS.escape(key)}"]`);
        if (!card) {
          card = document.createElement("div");
          card.className = "image-card";
          card.dataset.key = key;
          generatedPanel.appendChild(card);
        }

        const nextSrc = preview.image_path ? mediaUrl(preview.image_path) : "";
        let img = card.querySelector("img");
        if (nextSrc) {
          if (!img) {
            img = document.createElement("img");
            card.appendChild(img);
          }
          if (img.getAttribute("src") !== nextSrc) {
            img.src = nextSrc;
          }
          img.alt = preview.label || "Generated preview";
        } else if (img) {
          img.remove();
        }

        let title = card.querySelector("h3");
        if (!title) {
          title = document.createElement("h3");
          card.appendChild(title);
        }
        title.textContent = preview.label || "Generated preview";

        let info = card.querySelector(".mini");
        if (!info) {
          info = document.createElement("div");
          info.className = "mini";
          card.appendChild(info);
        }
        const infoHtml =
          `request #${preview.request_index || 0}<br>` +
          `attempt ${preview.attempt || 0}, selected ${preview.selected_candidate_index || 0}/${preview.candidate_count || 0}<br>` +
          `primary score: ${preview.primary_score ?? "n/a"}<br>` +
          `final score: ${preview.final_score ?? "n/a"}`;
        if (info.innerHTML !== infoHtml) {
          info.innerHTML = infoHtml;
        }
      }

      normalized.forEach((preview, index) => {
        const key = `${preview.request_index || 0}::${preview.label || ""}`;
        const card = generatedPanel.querySelector(`.image-card[data-key="${CSS.escape(key)}"]`);
        if (card) {
          const currentChild = generatedPanel.children[index];
          if (currentChild !== card) {
            generatedPanel.insertBefore(card, currentChild || null);
          }
        }
      });

      const signature = JSON.stringify(
        normalized.map((preview) => ({
          request_index: preview.request_index || 0,
          label: preview.label || "",
          attempt: preview.attempt || 0,
          primary_score: preview.primary_score ?? null,
          final_score: preview.final_score ?? null,
          candidate_count: preview.candidate_count || 0,
          selected_candidate_index: preview.selected_candidate_index || 0,
          image_path: preview.image_path || "",
        }))
      );
      generatedPanel.dataset.signature = signature;
      if (generatedPanel.dataset.signature === signature) {
        return;
      }
    }

    function renderCurrentRequest(current) {
      const meta = document.getElementById("currentRequestMeta");
      meta.innerHTML = "";

      if (!current) {
        meta.textContent = "No validator image request seen yet.";
        renderSeedPanel(null);
        renderGeneratedPanel([]);
        return;
      }

      const lines = [
        ["Validator", current.validator_name || "unknown"],
        ["Hotkey", current.validator_hotkey || "n/a"],
        ["Run", String(current.run_id || "n/a")],
        ["Challenge", current.challenge_id || "n/a"],
        ["Seed filename", current.image_filename || "n/a"],
      ];
      if (current.submission_status) {
        const sub = current.submission_status;
        lines.push([
          "Submission",
          `returned ${sub.returned_names || 0}/${sub.requested_names || 0}, s3 ${sub.s3_submissions || 0}, failures ${sub.fail_reasons || "none"}`
        ]);
      }
      for (const [label, value] of lines) {
        const row = document.createElement("div");
        row.innerHTML = `<strong>${label}:</strong> ${value}`;
        meta.appendChild(row);
      }

      const requested = Array.isArray(current.requested_variations) ? current.requested_variations : [];
      if (requested.length > 0) {
        const listTitle = document.createElement("div");
        listTitle.innerHTML = "<strong>Requested variations:</strong>";
        meta.appendChild(listTitle);
        const ul = document.createElement("ul");
        ul.className = "variation-list";
        for (const req of requested) {
          const li = document.createElement("li");
          li.textContent = `${req.type || "unknown"} (${req.intensity || "n/a"}): ${req.detail || req.description || ""}`;
          ul.appendChild(li);
        }
        meta.appendChild(ul);
      }

      renderSeedPanel(current);

      const previews = Array.isArray(current.generated_previews) ? current.generated_previews : [];
      renderGeneratedPanel(previews);
    }

    function renderTail(lines) {
      const node = document.getElementById("tailBody");
      node.innerHTML = "";
      if (!lines || lines.length === 0) {
        node.textContent = "No log lines yet.";
        return;
      }
      for (const line of lines) {
        const p = document.createElement("p");
        p.className = "tail-line";
        p.textContent = line;
        node.appendChild(p);
      }
      node.scrollTop = node.scrollHeight;
    }

    async function refresh() {
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        if (!res.ok) throw new Error("status " + res.status);
        const data = await res.json();

        const miner = data.miner || {};
        const metrics = data.metrics || {};
        const log = data.log || {};
        const port = data.port_health || {};
        const current = metrics.current_request || null;

        setBadge(Boolean(miner.running));
        setText("lastRefreshed", "Refreshed at " + new Date().toLocaleString());

        setText("uptime", miner.uptime_human || "n/a");
        setText("pid", "PID: " + (miner.pid ?? "n/a"));
        setText("runsTotal", String(metrics.runs_total ?? 0));
        setText("lastRun", "Last run: " + (metrics.last_run_id ?? "n/a"));
        setText("processedNames", String(metrics.processed_names_total ?? 0));
        setText("requestedNames", "Requested: " + String(metrics.requested_names_total ?? 0));
        setText("phase4", String(metrics.phase4_submissions_total ?? 0));
        setText("verifiedCalls", "Verified validator calls: " + String(metrics.validator_calls_verified ?? 0));
        setText("errorCount", String(metrics.errors_total ?? 0));
        setText("warningCount", "Warnings: " + String(metrics.warnings_total ?? 0));
        setPortState(port.state || "unknown");
        setText(
          "portTarget",
          "Target: " + (port.target_ip || "n/a") + ":" + String(port.target_port ?? "n/a")
        );
        setText(
          "portDetail",
          "Last check: " + (port.checked_utc || "n/a") +
          " | " + (port.reason || "n/a") +
          " | fail streak: " + String(port.failure_streak ?? 0)
        );
        setText("logPath", log.path ?? "n/a");
        setText(
          "logMeta",
          "Size: " + (log.size_human ?? "n/a") +
          " | Updated (UTC): " + (log.updated_utc ?? "n/a") +
          " | Tail lines: " + String(log.tail_line_count ?? 0)
        );
        setText("lastActivity", "Last activity: " + (metrics.last_activity_line || "n/a"));

        renderEvents(metrics.recent_events || []);
        renderCurrentRequest(current);
        renderTail(log.tail_preview || []);
      } catch (err) {
        setBadge(false);
        setText("lastRefreshed", "Dashboard error: " + err.message);
      }
    }

    refresh();
    setInterval(refresh, REFRESH_MS);
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SN54 miner monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=8810, help="Dashboard bind port")
    parser.add_argument("--log-file", required=True, help="Path to miner log file")
    parser.add_argument("--pid-file", required=True, help="Path to miner pid file")
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
        "--port-health-ip-discovery-urls",
        default="https://api.ipify.org,https://ifconfig.me/ip",
        help="Comma-separated public IP discovery URLs",
    )
    parser.add_argument(
        "--port-health-check-url",
        default="https://ports.yougetsignal.com/check-port.php",
        help="External TCP port checker endpoint",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_file = Path(args.log_file).expanduser().resolve()
    pid_file = Path(args.pid_file).expanduser().resolve()
    port_monitor = PublicPortHealthMonitor(
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

    server = DashboardServer((args.host, args.port), DashboardHandler, log_file, pid_file, port_monitor)
    print(
        f"[{utc_now_iso()}] SN54 dashboard serving http://{args.host}:{args.port} "
        f"(log={log_file}, pid={pid_file})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
