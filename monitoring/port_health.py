#!/usr/bin/env python3
"""Public port health checks for SN54 miner monitoring."""

from __future__ import annotations

import socket
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple


DEFAULT_IP_DISCOVERY_URLS = [
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
]
DEFAULT_PORT_CHECK_URL = "https://ports.yougetsignal.com/check-port.php"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def str_to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_csv_urls(value: str) -> List[str]:
    urls = [part.strip() for part in value.split(",") if part.strip()]
    if urls:
        return urls
    return list(DEFAULT_IP_DISCOVERY_URLS)


def check_local_listener(port: Optional[int], timeout_seconds: float = 1.0) -> Optional[bool]:
    if not port or port <= 0:
        return None
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)
        sock.connect(("127.0.0.1", int(port)))
        return True
    except Exception:
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def discover_public_ip(
    configured_external_ip: Optional[str],
    discovery_urls: List[str],
    timeout_seconds: float,
) -> Tuple[Optional[str], str]:
    if configured_external_ip:
        return configured_external_ip.strip(), "configured_external_ip"

    for url in discovery_urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                candidate = response.read(128).decode("utf-8", errors="replace").strip()
                if candidate:
                    return candidate, f"ip_discovery:{url}"
        except Exception:
            continue
    return None, "ip_discovery_failed"


def check_public_port_with_yougetsignal(
    ip: str,
    port: int,
    timeout_seconds: float,
    check_url: str = DEFAULT_PORT_CHECK_URL,
) -> Tuple[Optional[bool], str]:
    payload = urllib.parse.urlencode(
        {"remoteAddress": ip, "portNumber": str(port)},
    ).encode("utf-8")
    request = urllib.request.Request(
        check_url,
        data=payload,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(12_000).decode("utf-8", errors="replace")
    except Exception as exc:
        return None, f"external_check_failed: {exc}"

    low = body.lower()
    if " is open on " in low or "flag_green" in low:
        return True, "external checker reports OPEN"
    if " is closed on " in low or "flag_red" in low:
        return False, "external checker reports CLOSED"
    return None, "external checker response could not be parsed"


@dataclass
class PortHealthSnapshot:
    enabled: bool
    state: str
    reachable: Optional[bool]
    target_ip: Optional[str]
    target_port: Optional[int]
    source: str
    reason: str
    checked_utc: Optional[str]
    check_interval_seconds: float
    failure_threshold: int
    recovery_threshold: int
    failure_streak: int
    success_streak: int
    local_listener: Optional[bool]
    transition: Optional[str]
    age_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class PublicPortHealthMonitor:
    """Caches periodic external port reachability checks."""

    def __init__(
        self,
        enabled: bool,
        axon_port: Optional[int],
        external_ip: Optional[str],
        external_port: Optional[int],
        check_interval_seconds: float = 60.0,
        failure_threshold: int = 1,
        recovery_threshold: int = 1,
        connect_timeout_seconds: float = 8.0,
        ip_discovery_urls: Optional[List[str]] = None,
        port_check_url: str = DEFAULT_PORT_CHECK_URL,
    ) -> None:
        self.enabled = bool(enabled)
        self.axon_port = int(axon_port) if axon_port else None
        self.external_ip = (external_ip or "").strip() or None
        self.external_port = int(external_port) if external_port else None
        self.target_port = self.external_port or self.axon_port

        self.check_interval_seconds = max(float(check_interval_seconds), 5.0)
        self.failure_threshold = max(int(failure_threshold), 1)
        self.recovery_threshold = max(int(recovery_threshold), 1)
        self.connect_timeout_seconds = max(float(connect_timeout_seconds), 1.0)
        self.ip_discovery_urls = ip_discovery_urls or list(DEFAULT_IP_DISCOVERY_URLS)
        self.port_check_url = port_check_url

        self._failure_streak = 0
        self._success_streak = 0
        self._stable_reachable: Optional[bool] = None
        self._last_checked_monotonic = 0.0

        self._snapshot = PortHealthSnapshot(
            enabled=self.enabled,
            state="disabled" if not self.enabled else "unknown",
            reachable=None,
            target_ip=self.external_ip,
            target_port=self.target_port,
            source="init",
            reason="health check not started",
            checked_utc=None,
            check_interval_seconds=self.check_interval_seconds,
            failure_threshold=self.failure_threshold,
            recovery_threshold=self.recovery_threshold,
            failure_streak=self._failure_streak,
            success_streak=self._success_streak,
            local_listener=None,
            transition=None,
            age_seconds=0.0,
        )

    def _update_age(self) -> None:
        if self._last_checked_monotonic <= 0:
            self._snapshot.age_seconds = 0.0
            return
        self._snapshot.age_seconds = max(0.0, time.monotonic() - self._last_checked_monotonic)

    def check_due(self, force: bool = False) -> PortHealthSnapshot:
        if not self.enabled:
            self._snapshot = PortHealthSnapshot(
                enabled=False,
                state="disabled",
                reachable=None,
                target_ip=self.external_ip,
                target_port=self.target_port,
                source="disabled",
                reason="PORT_HEALTH_ENABLED is false",
                checked_utc=utc_now_iso(),
                check_interval_seconds=self.check_interval_seconds,
                failure_threshold=self.failure_threshold,
                recovery_threshold=self.recovery_threshold,
                failure_streak=0,
                success_streak=0,
                local_listener=check_local_listener(self.axon_port, timeout_seconds=0.5),
                transition=None,
                age_seconds=0.0,
            )
            return self._snapshot

        now = time.monotonic()
        if (
            not force
            and self._last_checked_monotonic > 0
            and now - self._last_checked_monotonic < self.check_interval_seconds
        ):
            self._update_age()
            self._snapshot.transition = None
            return self._snapshot

        local_listener = check_local_listener(
            self.axon_port,
            timeout_seconds=min(self.connect_timeout_seconds, 1.0),
        )

        if not self.target_port:
            self._last_checked_monotonic = now
            self._snapshot = PortHealthSnapshot(
                enabled=True,
                state="unknown",
                reachable=None,
                target_ip=self.external_ip,
                target_port=None,
                source="configuration",
                reason="target port is not configured",
                checked_utc=utc_now_iso(),
                check_interval_seconds=self.check_interval_seconds,
                failure_threshold=self.failure_threshold,
                recovery_threshold=self.recovery_threshold,
                failure_streak=self._failure_streak,
                success_streak=self._success_streak,
                local_listener=local_listener,
                transition=None,
                age_seconds=0.0,
            )
            return self._snapshot

        ip, ip_source = discover_public_ip(
            configured_external_ip=self.external_ip,
            discovery_urls=self.ip_discovery_urls,
            timeout_seconds=self.connect_timeout_seconds,
        )

        if not ip:
            self._last_checked_monotonic = now
            self._snapshot = PortHealthSnapshot(
                enabled=True,
                state="unknown",
                reachable=None,
                target_ip=None,
                target_port=self.target_port,
                source=ip_source,
                reason="could not determine public IP",
                checked_utc=utc_now_iso(),
                check_interval_seconds=self.check_interval_seconds,
                failure_threshold=self.failure_threshold,
                recovery_threshold=self.recovery_threshold,
                failure_streak=self._failure_streak,
                success_streak=self._success_streak,
                local_listener=local_listener,
                transition=None,
                age_seconds=0.0,
            )
            return self._snapshot

        reachable, reason = check_public_port_with_yougetsignal(
            ip=ip,
            port=self.target_port,
            timeout_seconds=self.connect_timeout_seconds,
            check_url=self.port_check_url,
        )

        if reachable is True:
            self._success_streak += 1
            self._failure_streak = 0
        elif reachable is False:
            self._failure_streak += 1
            self._success_streak = 0

        transition: Optional[str] = None
        previous_stable = self._stable_reachable
        if reachable is True and self._success_streak >= self.recovery_threshold and previous_stable is not True:
            self._stable_reachable = True
            transition = "recovered" if previous_stable is False else "reachable"
        elif (
            reachable is False
            and self._failure_streak >= self.failure_threshold
            and previous_stable is not False
        ):
            self._stable_reachable = False
            transition = "unreachable"

        if self._stable_reachable is True:
            state = "reachable"
        elif self._stable_reachable is False:
            state = "unreachable"
        elif reachable is True:
            state = "reachable"
        elif reachable is False:
            state = "unreachable"
        else:
            state = "unknown"

        self._last_checked_monotonic = now
        self._snapshot = PortHealthSnapshot(
            enabled=True,
            state=state,
            reachable=reachable,
            target_ip=ip,
            target_port=self.target_port,
            source=ip_source,
            reason=reason,
            checked_utc=utc_now_iso(),
            check_interval_seconds=self.check_interval_seconds,
            failure_threshold=self.failure_threshold,
            recovery_threshold=self.recovery_threshold,
            failure_streak=self._failure_streak,
            success_streak=self._success_streak,
            local_listener=local_listener,
            transition=transition,
            age_seconds=0.0,
        )
        return self._snapshot

    def snapshot_dict(self, force: bool = False) -> dict:
        snapshot = self.check_due(force=force)
        self._update_age()
        return snapshot.to_dict()
