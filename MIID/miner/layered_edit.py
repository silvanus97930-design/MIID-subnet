from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bittensor as bt
import numpy as np
from PIL import Image, ImageFilter

FaceBox = Tuple[float, float, float, float]

_PIPE = None
_PIPE_LOCK = threading.Lock()


def _bool_env(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


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


def layered_helper_enabled() -> bool:
    return _bool_env("SN54_QWEN_LAYERED_ENABLED", True)


def layered_background_enabled() -> bool:
    return _bool_env("SN54_QWEN_LAYERED_BACKGROUND_EDIT", True)


def layered_screen_replay_enabled() -> bool:
    return _bool_env("SN54_QWEN_LAYERED_SCREEN_REPLAY", True)


def layered_allow_cpu_fallback() -> bool:
    return _bool_env("SN54_QWEN_LAYERED_ALLOW_CPU_FALLBACK", False)


def _resolve_device() -> str:
    requested = (os.environ.get("QWEN_LAYERED_DEVICE") or os.environ.get("FLUX_DEVICE") or "").strip().lower()
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_dtype(device: str):
    try:
        import torch
    except Exception:
        return None
    requested = (os.environ.get("QWEN_LAYERED_DTYPE") or "").strip().lower()
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    if requested in {"fp32", "float32"}:
        return torch.float32
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _resolution_bucket(rgb: Image.Image) -> int:
    w, h = rgb.size
    long_side = max(w, h)
    return 1024 if long_side >= 896 else 640


def _release_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def prewarm_qwen_layered_pipeline() -> None:
    """Eagerly load ``QwenImageLayeredPipeline`` when layered helpers are enabled.

    Called from miner startup (with the main image backend prewarm) so the first
    ``background_edit`` / ``screen_replay`` request does not pay full load latency.
    """
    if not layered_helper_enabled():
        return
    if not (layered_background_enabled() or layered_screen_replay_enabled()):
        return
    with _PIPE_LOCK:
        _get_pipeline()


def _get_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from diffusers import QwenImageLayeredPipeline
    import torch

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    device = _resolve_device()
    # HF hub id is Qwen/Qwen-Image-Layered (same weights as qwen/Qwen-Image-Layered redirects).
    model_id = os.environ.get("QWEN_LAYERED_MODEL_ID", "Qwen/Qwen-Image-Layered")

    def _build_pipeline(target_device: str, target_dtype) -> Any:
        kwargs: Dict[str, Any] = {}
        if target_dtype is not None:
            kwargs["torch_dtype"] = target_dtype
        if token:
            kwargs["token"] = token
        pipe = QwenImageLayeredPipeline.from_pretrained(model_id, **kwargs)
        return pipe.to(target_device)

    try:
        _PIPE = _build_pipeline(device, _resolve_dtype(device))
    except torch.OutOfMemoryError:
        if device == "cpu" or not layered_allow_cpu_fallback():
            raise
        bt.logging.warning(f"Qwen-Image-Layered helper ran out of VRAM on {device}; retrying on CPU.")
        _release_cuda_memory()
        _PIPE = _build_pipeline("cpu", torch.float32)

    try:
        _PIPE.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return _PIPE


@dataclass
class LayeredSubjectResult:
    subject_rgba: Image.Image
    alpha_mask: Image.Image
    layer_count: int
    used_layered: bool
    evidence: Dict[str, Any]


def _face_core_mask(size: Tuple[int, int], face_box_xyxy: Optional[FaceBox], expand: float) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if face_box_xyxy is not None:
        x0, y0, x1, y1 = [float(v) for v in face_box_xyxy]
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        rx = max(8.0, 0.5 * (x1 - x0) * expand)
        ry = max(10.0, 0.58 * (y1 - y0) * expand)
    else:
        cx = w * 0.5
        cy = h * 0.42
        rx = w * 0.18 * expand
        ry = h * 0.20 * expand
    ell = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2) <= 1.0
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[ell] = 255
    return Image.fromarray(arr, mode="L").filter(
        ImageFilter.GaussianBlur(radius=max(2.0, min(w, h) * 0.018))
    )


def _fallback_subject_result(base_rgb: Image.Image, face_box_xyxy: Optional[FaceBox]) -> LayeredSubjectResult:
    rgb = base_rgb.convert("RGB")
    core = _face_core_mask(rgb.size, face_box_xyxy, _float_env("SN54_QWEN_LAYERED_FACE_EXPAND", 1.55))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(core)
    return LayeredSubjectResult(
        subject_rgba=rgba,
        alpha_mask=core,
        layer_count=0,
        used_layered=False,
        evidence={"mode": "fallback_face_core"},
    )


def _extract_layers(rgb: Image.Image) -> List[Image.Image]:
    pipe = _get_pipeline()
    resolution = _resolution_bucket(rgb)
    inputs: Dict[str, Any] = {
        "image": rgb.convert("RGBA"),
        "layers": _int_env("SN54_QWEN_LAYERED_LAYERS", 4),
        "num_inference_steps": _int_env("SN54_QWEN_LAYERED_STEPS", 24),
        "resolution": resolution,
        "true_cfg_scale": _float_env("SN54_QWEN_LAYERED_TRUE_CFG_SCALE", 4.0),
        "cfg_normalize": _bool_env("SN54_QWEN_LAYERED_CFG_NORMALIZE", True),
        "use_en_prompt": _bool_env("SN54_QWEN_LAYERED_USE_EN_PROMPT", True),
        "negative_prompt": (os.environ.get("SN54_QWEN_LAYERED_NEGATIVE_PROMPT") or " ").strip() or " ",
        "prompt": (os.environ.get("SN54_QWEN_LAYERED_PROMPT") or "").strip() or None,
        "output_type": "pil",
        "num_images_per_prompt": 1,
    }
    out = pipe(**inputs)
    images = getattr(out, "images", None)
    if not images:
        return []
    first = images[0]
    if isinstance(first, list):
        return [img.convert("RGBA") for img in first if isinstance(img, Image.Image)]
    if isinstance(first, Image.Image):
        return [first.convert("RGBA")]
    return []


def _layer_score(layer: Image.Image, face_box_xyxy: Optional[FaceBox]) -> Tuple[float, np.ndarray]:
    rgba = layer.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    if float(alpha.mean()) <= 1e-4:
        return 0.0, alpha
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if face_box_xyxy is not None:
        x0, y0, x1, y1 = [float(v) for v in face_box_xyxy]
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        rx = max(6.0, 0.62 * (x1 - x0))
        ry = max(8.0, 0.70 * (y1 - y0))
    else:
        cx = w * 0.5
        cy = h * 0.42
        rx = w * 0.18
        ry = h * 0.20
    face_mask = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2) <= 1.0
    center_mask = (((xx - w * 0.5) / max(w * 0.28, 1e-6)) ** 2 + ((yy - h * 0.5) / max(h * 0.34, 1e-6)) ** 2) <= 1.0
    face_cov = float(np.mean(alpha[face_mask])) if np.any(face_mask) else 0.0
    center_cov = float(np.mean(alpha[center_mask])) if np.any(center_mask) else 0.0
    alpha_mass = float(alpha.mean())
    score = face_cov * 0.62 + center_cov * 0.28 + min(alpha_mass, 0.25) * 0.10
    return score, alpha


def extract_subject_layers(
    base_rgb: Image.Image,
    face_box_xyxy: Optional[FaceBox] = None,
) -> LayeredSubjectResult:
    rgb = base_rgb.convert("RGB")
    if not layered_helper_enabled():
        return _fallback_subject_result(rgb, face_box_xyxy)

    with _PIPE_LOCK:
        try:
            layers = _extract_layers(rgb)
        except Exception as e:
            _release_cuda_memory()
            return LayeredSubjectResult(
                **_fallback_subject_result(rgb, face_box_xyxy).__dict__,
                evidence={"mode": "fallback_after_exception", "detail": str(e)},
            )

    if not layers:
        return _fallback_subject_result(rgb, face_box_xyxy)

    scored: List[Tuple[float, Image.Image, np.ndarray]] = []
    for layer in layers:
        score, alpha = _layer_score(layer, face_box_xyxy)
        if score > 0.02:
            scored.append((score, layer.convert("RGBA"), alpha))
    if not scored:
        return _fallback_subject_result(rgb, face_box_xyxy)

    scored.sort(key=lambda row: row[0], reverse=True)
    keep_n = max(1, min(len(scored), _int_env("SN54_QWEN_LAYERED_KEEP_LAYERS", 2)))
    selected = scored[:keep_n]

    comp = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    union_alpha = np.zeros((rgb.size[1], rgb.size[0]), dtype=np.float32)
    for _, layer, alpha in selected:
        comp = Image.alpha_composite(comp, layer)
        union_alpha = np.maximum(union_alpha, alpha)

    core_mask = np.asarray(
        _face_core_mask(rgb.size, face_box_xyxy, _float_env("SN54_QWEN_LAYERED_FACE_EXPAND", 1.55)),
        dtype=np.float32,
    ) / 255.0
    final_alpha = np.clip(np.maximum(union_alpha * core_mask, core_mask * 0.85), 0.0, 1.0)
    alpha_img = Image.fromarray((final_alpha * 255.0).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=max(2.0, min(rgb.size) * 0.012))
    )
    comp.putalpha(alpha_img)

    return LayeredSubjectResult(
        subject_rgba=comp,
        alpha_mask=alpha_img,
        layer_count=len(selected),
        used_layered=True,
        evidence={
            "mode": "layered_subject",
            "selected_layer_scores": [round(float(row[0]), 4) for row in selected],
            "candidate_layer_count": len(layers),
        },
    )


def preserve_subject_over_candidate(
    base_rgb: Image.Image,
    candidate_rgb: Image.Image,
    face_box_xyxy: Optional[FaceBox] = None,
    *,
    subject: Optional[LayeredSubjectResult] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    cand = candidate_rgb.convert("RGB")
    subj = subject or extract_subject_layers(base_rgb, face_box_xyxy)
    overlay = subj.subject_rgba.convert("RGBA")
    out = cand.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB"), {
        "layered_subject_used": subj.used_layered,
        "layered_subject_layer_count": subj.layer_count,
        "layered_subject_evidence": subj.evidence,
    }


def prepare_screen_replay_subject(
    base_rgb: Image.Image,
    face_box_xyxy: Optional[FaceBox] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    subj = extract_subject_layers(base_rgb, face_box_xyxy)
    rgb = base_rgb.convert("RGB")
    bg = rgb.filter(ImageFilter.GaussianBlur(radius=max(4.0, min(rgb.size) * 0.02)))
    canvas = bg.convert("RGBA")
    canvas.alpha_composite(subj.subject_rgba)
    return canvas.convert("RGB"), {
        "layered_subject_used": subj.used_layered,
        "layered_subject_layer_count": subj.layer_count,
        "layered_subject_evidence": subj.evidence,
    }
