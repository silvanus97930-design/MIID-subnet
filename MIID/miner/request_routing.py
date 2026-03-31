from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _str_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return str(value).strip() if value is not None else str(default)


def _req_type(req: Any) -> str:
    return str(
        getattr(req, "type", None) or (req.get("type") if isinstance(req, dict) else None) or ""
    ).strip().lower()


def _route_suffix(req_type: str) -> str:
    return {
        "pose_edit": "POSE_EDIT",
        "expression_edit": "EXPRESSION_EDIT",
        "background_edit": "BACKGROUND_EDIT",
        "lighting_edit": "LIGHTING_EDIT",
        "screen_replay": "SCREEN_REPLAY",
    }.get(req_type, "DEFAULT")


def resolve_backend_name_for_request(req: Any) -> str:
    default_backend = _str_env("SN54_IMAGE_GENERATION_BACKEND", "comfyui").lower()
    if not _bool_env("SN54_ENABLE_BACKEND_ROUTING", True):
        return default_backend
    suffix = _route_suffix(_req_type(req))
    return _str_env(f"SN54_BACKEND_{suffix}", _str_env("SN54_BACKEND_DEFAULT", default_backend)).lower()


def resolve_workflow_name_for_request(req: Any) -> str:
    req_type = _req_type(req)
    defaults = {
        "pose_edit": "pose_edit_sdxl_instantid_or_fallback",
        "expression_edit": "expression_edit_sdxl_ipadapter_faceid",
        "background_edit": "background_edit_sdxl_inpaint_ipadapter",
        "lighting_edit": "lighting_edit_iclight_or_sdxl_fallback",
        "screen_replay": "screen_replay_sdxl_postprocess_chain",
    }
    suffix = _route_suffix(req_type)
    return _str_env(f"COMFYUI_WORKFLOW_{suffix}", defaults.get(req_type, f"{req_type or 'generic'}_sdxl"))


@dataclass(frozen=True)
class RequestRoute:
    backend_name: str
    workflow_name: str
    request_type: str


def resolve_request_route(req: Any) -> RequestRoute:
    return RequestRoute(
        backend_name=resolve_backend_name_for_request(req),
        workflow_name=resolve_workflow_name_for_request(req),
        request_type=_req_type(req),
    )
