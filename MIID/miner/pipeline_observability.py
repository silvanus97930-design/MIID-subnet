# Structured JSON-friendly logging for Phase 4 image pipeline observability.

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import bittensor as bt


def log_phase4_json(event: str, payload: Optional[Dict[str, Any]] = None, **fields: Any) -> None:
    """Emit a single structured log line (JSON) for log aggregators."""
    data: Dict[str, Any] = {"event": str(event)}
    if payload:
        data.update(payload)
    for k, v in fields.items():
        if v is not None:
            data[k] = v
    bt.logging.info(json.dumps(data, default=str, ensure_ascii=False))
