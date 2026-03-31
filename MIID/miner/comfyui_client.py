from __future__ import annotations

import io
import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import bittensor as bt
import requests
from PIL import Image


def _int_env(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(str(raw).strip()) if raw is not None else int(default)
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(str(raw).strip()) if raw is not None else float(default)
    except Exception:
        return float(default)


class ComfyUIClient:
    """Tiny synchronous client for the ComfyUI HTTP API."""

    def __init__(self) -> None:
        self.base_url = (os.environ.get("COMFYUI_BASE_URL") or "http://127.0.0.1:20007").strip().rstrip("/")
        self.request_timeout_s = _float_env("COMFYUI_REQUEST_TIMEOUT_SECONDS", 60.0)
        self.execution_timeout_s = _float_env("COMFYUI_EXECUTION_TIMEOUT_SECONDS", 300.0)
        self.poll_interval_s = _float_env("COMFYUI_POLL_INTERVAL_SECONDS", 1.0)
        self.upload_subfolder = (os.environ.get("COMFYUI_UPLOAD_SUBFOLDER") or "miid").strip().strip("/")
        self.client_id = (os.environ.get("COMFYUI_CLIENT_ID") or str(uuid.uuid4())).strip()
        self._session = requests.Session()
        self._object_info: Optional[Dict[str, Any]] = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _check_response(self, response: requests.Response) -> Dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected ComfyUI response payload type: {type(payload).__name__}")
        return payload

    def system_stats(self) -> Dict[str, Any]:
        response = self._session.get(self._url("/system_stats"), timeout=self.request_timeout_s)
        return self._check_response(response)

    def object_info(self) -> Dict[str, Any]:
        if self._object_info is None:
            response = self._session.get(self._url("/object_info"), timeout=self.request_timeout_s)
            self._object_info = self._check_response(response)
        return self._object_info

    def prewarm(self) -> None:
        stats = self.system_stats()
        self.object_info()
        version = ((stats.get("system") or {}).get("comfyui_version") or "unknown")
        bt.logging.info(f"ComfyUI prewarm: connected to {self.base_url} (version={version})")

    def upload_image(self, image: Image.Image, *, filename_prefix: str) -> str:
        safe_prefix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in filename_prefix)[:80] or "miid"
        filename = f"{safe_prefix}_{uuid.uuid4().hex[:10]}.png"
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        files = {"image": (filename, buf.getvalue(), "image/png")}
        data = {
            "type": "input",
            "overwrite": "true",
            "subfolder": self.upload_subfolder,
        }
        response = self._session.post(
            self._url("/upload/image"),
            files=files,
            data=data,
            timeout=self.request_timeout_s,
        )
        payload = self._check_response(response)
        name = str(payload.get("name") or filename).strip()
        subfolder = str(payload.get("subfolder") or self.upload_subfolder).strip().strip("/")
        return f"{subfolder}/{name}" if subfolder else name

    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        payload = {"prompt": prompt, "client_id": self.client_id}
        response = self._session.post(
            self._url("/prompt"),
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.request_timeout_s,
        )
        data = self._check_response(response)
        prompt_id = str(data.get("prompt_id") or "").strip()
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return prompt_id

    def history(self, prompt_id: str) -> Dict[str, Any]:
        response = self._session.get(
            self._url(f"/history/{quote(str(prompt_id), safe='')}"),
            timeout=self.request_timeout_s,
        )
        return self._check_response(response)

    def wait_for_completion(self, prompt_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.execution_timeout_s
        while time.monotonic() < deadline:
            payload = self.history(prompt_id)
            entry = payload.get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    messages = status.get("messages") or []
                    raise RuntimeError(f"ComfyUI prompt failed: {messages}")
                outputs = entry.get("outputs")
                if isinstance(outputs, dict) and outputs:
                    return entry
            time.sleep(max(self.poll_interval_s, 0.2))
        raise TimeoutError(
            f"Timed out after {self.execution_timeout_s:.1f}s waiting for ComfyUI prompt {prompt_id}"
        )

    def fetch_image(self, *, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        params = {
            "filename": filename,
            "subfolder": subfolder or "",
            "type": folder_type,
        }
        response = self._session.get(self._url("/view"), params=params, timeout=self.request_timeout_s)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    def collect_output_images(
        self,
        history_entry: Dict[str, Any],
        *,
        node_ids: Optional[Iterable[str]] = None,
    ) -> List[Image.Image]:
        outputs = history_entry.get("outputs")
        if not isinstance(outputs, dict):
            return []
        ordered_ids = [str(node_id) for node_id in node_ids] if node_ids else sorted(outputs.keys(), key=str)
        images: List[Image.Image] = []
        for node_id in ordered_ids:
            node_output = outputs.get(str(node_id)) or {}
            img_entries = node_output.get("images") or []
            for meta in img_entries:
                if not isinstance(meta, dict):
                    continue
                filename = str(meta.get("filename") or "").strip()
                if not filename:
                    continue
                images.append(
                    self.fetch_image(
                        filename=filename,
                        subfolder=str(meta.get("subfolder") or ""),
                        folder_type=str(meta.get("type") or "output"),
                    )
                )
        return images
