from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence

import bittensor as bt
import torch
from diffusers import ZImagePipeline
from PIL import Image

from MIID.miner.generate_variations import _get_prompt_from_request, _get_type_and_intensity, _wire_text_field
from MIID.miner.generator_backends.base import GenerationConfig, ImageGeneratorBackend


def _normalize_device_name(device_name: str) -> str:
    value = (device_name or "").strip().lower()
    if not value:
        return ""
    if value.startswith("cuda") and not torch.cuda.is_available():
        bt.logging.warning(
            f"ZIMAGE_DEVICE={device_name} requested but CUDA is unavailable. Falling back to CPU."
        )
        return "cpu"
    return value


def _resolve_device() -> str:
    requested = _normalize_device_name(os.environ.get("ZIMAGE_DEVICE", ""))
    if requested:
        return requested
    flux_requested = _normalize_device_name(os.environ.get("FLUX_DEVICE", ""))
    if flux_requested:
        return flux_requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_dtype(name: str) -> torch.dtype | None:
    normalized = (name or "").strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(normalized)


def _resolve_dtype(device: str) -> torch.dtype:
    requested = _parse_dtype(os.environ.get("ZIMAGE_DTYPE", os.environ.get("FLUX_DTYPE", "auto")))
    if requested is not None:
        if requested in (torch.float16, torch.bfloat16) and not device.startswith("cuda"):
            bt.logging.warning(
                "ZIMAGE_DTYPE requested GPU-only precision on non-CUDA device. Falling back to float32."
            )
            return torch.float32
        return requested
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


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


def _chunked(values: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _release_cuda_memory(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        pass


class ZImageGeneratorBackend(ImageGeneratorBackend):
    """Tongyi-MAI/Z-Image text-to-image backend for SN54 experimentation."""

    def __init__(self) -> None:
        self._pipe: Optional[ZImagePipeline] = None
        self._pipe_lock = threading.Lock()
        self._device = _resolve_device()
        self._dtype = _resolve_dtype(self._device)
        self._serialize = (
            os.environ.get("ZIMAGE_SERIALIZE_REQUESTS", os.environ.get("FLUX_SERIALIZE_REQUESTS", "true"))
            .strip()
            .lower()
            in {"1", "true", "yes"}
        )
        self._model_id = os.environ.get("ZIMAGE_MODEL_ID", "Tongyi-MAI/Z-Image")
        self._default_steps = _int_env("ZIMAGE_NUM_INFERENCE_STEPS", 40)
        self._default_guidance = _float_env("ZIMAGE_GUIDANCE_SCALE", 4.0)
        self._cfg_normalization = (
            os.environ.get("ZIMAGE_CFG_NORMALIZATION", "false").strip().lower() in {"1", "true", "yes", "on"}
        )
        self._negative_prompt = (
            os.environ.get(
                "ZIMAGE_NEGATIVE_PROMPT",
                "blurry, low quality, deformed face, duplicate face, extra people, bad anatomy, cropped head, disfigured eyes",
            )
            or ""
        ).strip()

    def _generation_lock_context(self):
        return self._pipe_lock if self._serialize else nullcontext()

    def _get_pipeline(self) -> ZImagePipeline:
        if self._pipe is not None:
            return self._pipe

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
        if not token:
            raise RuntimeError(
                "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_TOKEN in your environment, e.g.\n"
                '  export HF_TOKEN="hf_..."'
            )

        pipe_kwargs: Dict[str, Any] = {
            "torch_dtype": self._dtype,
            "token": token,
            "low_cpu_mem_usage": False,
        }
        bt.logging.info(
            f"Initializing Z-Image pipeline model={self._model_id} device={self._device} dtype={self._dtype}"
        )
        self._pipe = ZImagePipeline.from_pretrained(self._model_id, **pipe_kwargs)
        if hasattr(self._pipe, "enable_attention_slicing"):
            self._pipe.enable_attention_slicing()
        if hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()
        if hasattr(self._pipe, "enable_vae_tiling"):
            self._pipe.enable_vae_tiling()
        self._pipe = self._pipe.to(self._device)
        return self._pipe

    @staticmethod
    def _target_resolution(base_image: Image.Image) -> tuple[int, int]:
        w, h = base_image.size
        area_target = _int_env("ZIMAGE_TARGET_PIXEL_AREA", 1024 * 1024)
        min_side = _int_env("ZIMAGE_MIN_SIDE", 768)
        max_side = _int_env("ZIMAGE_MAX_SIDE", 1536)
        if w <= 0 or h <= 0:
            w, h = 768, 1024
        scale = (area_target / max(w * h, 1)) ** 0.5
        tw = max(min_side, int(round(w * scale / 32.0) * 32))
        th = max(min_side, int(round(h * scale / 32.0) * 32))
        tw = max(512, min(max_side, tw))
        th = max(512, min(max_side, th))
        tw = max(32, int(round(tw / 32.0) * 32))
        th = max(32, int(round(th / 32.0) * 32))
        return tw, th

    def load(self) -> None:
        self._get_pipeline()

    def prewarm(self) -> None:
        warm_enabled = os.environ.get("ZIMAGE_PREWARM_INFERENCE", "true").strip().lower() in {"1", "true", "yes", "on"}
        warm_steps = max(1, _int_env("ZIMAGE_PREWARM_STEPS", min(4, self._default_steps)))
        warm_prompt = os.environ.get(
            "ZIMAGE_PREWARM_PROMPT",
            "Professional passport-style portrait of the same person, neutral expression, clean background, photorealistic.",
        )
        with self._generation_lock_context():
            pipe = self._get_pipeline()
            if not warm_enabled:
                bt.logging.info("Z-Image prewarm: pipeline loaded (inference warm-up disabled)")
                return
            bt.logging.info(f"Z-Image prewarm: running warm-up inference (steps={warm_steps})")
            with torch.inference_mode():
                _ = pipe(
                    prompt=warm_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=warm_steps,
                    guidance_scale=self._default_guidance,
                    cfg_normalization=self._cfg_normalization,
                    negative_prompt=self._negative_prompt or None,
                )
            _release_cuda_memory(self._device)

    def generate_candidates(
        self,
        base_image: Any,
        compiled_request: Sequence[Any],
        generation_config: GenerationConfig,
        timings_out: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        if not compiled_request:
            return []

        if not isinstance(base_image, Image.Image):
            raise TypeError("ZImageGeneratorBackend expects a PIL.Image base_image")

        num_candidates = max(1, int(generation_config.candidates_per_request))
        batch_size = max(1, int(generation_config.request_batch_size))
        num_inference_steps = int(generation_config.num_inference_steps_override or self._default_steps)
        guidance_scale = float(generation_config.guidance_scale_override or self._default_guidance)
        target_w, target_h = self._target_resolution(base_image.convert("RGB"))

        with self._generation_lock_context():
            pipe = self._get_pipeline()
            prepared: List[Dict[str, Any]] = []
            t_prep0 = time.perf_counter()
            for req in compiled_request:
                var_type, intensity = _get_type_and_intensity(req)
                prompt = _get_prompt_from_request(req, var_type, intensity)
                prepared.append(
                    {
                        "request": req,
                        "variation_type": var_type,
                        "intensity": intensity,
                        "prompt": prompt,
                        "description": _wire_text_field(req, "description"),
                        "detail": _wire_text_field(req, "detail"),
                    }
                )
            prepare_ms = (time.perf_counter() - t_prep0) * 1000.0

            results: List[Dict[str, Any]] = []
            generation_ms_total = 0.0

            for chunk in _chunked(prepared, batch_size):
                prompts = [entry["prompt"] for entry in chunk]
                _release_cuda_memory(self._device)
                t_gen0 = time.perf_counter()
                with torch.inference_mode():
                    out = pipe(
                        prompt=prompts,
                        height=target_h,
                        width=target_w,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        cfg_normalization=self._cfg_normalization,
                        negative_prompt=self._negative_prompt or None,
                        num_images_per_prompt=num_candidates,
                    )
                generation_ms_total += (time.perf_counter() - t_gen0) * 1000.0
                flat_images = list(getattr(out, "images", []) or [])
                expected = len(chunk) * num_candidates
                if len(flat_images) != expected:
                    raise RuntimeError(
                        f"Z-Image returned {len(flat_images)} images, expected {expected} "
                        f"(chunk={len(chunk)}, candidates={num_candidates})"
                    )
                for idx, entry in enumerate(chunk):
                    start = idx * num_candidates
                    end = start + num_candidates
                    results.append(
                        {
                            "variation_type": entry["variation_type"],
                            "intensity": entry["intensity"],
                            "prompt": entry["prompt"],
                            "description": entry["description"],
                            "detail": entry["detail"],
                            "candidates": flat_images[start:end],
                            "backend": "zimage",
                        }
                    )

            _release_cuda_memory(self._device)

            if timings_out is not None:
                timings_out["variation_request_prepare_ms"] = prepare_ms
                timings_out["flux_generation_ms"] = generation_ms_total

            return results
