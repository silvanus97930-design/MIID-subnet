# MIID/miner/generator_backends/
#
# Pluggable SN54 Phase 4 image generation backends (FLUX default; SDXL stub).

from __future__ import annotations

import os
from typing import Dict, Type

import bittensor as bt

from MIID.miner.generator_backends.base import GenerationConfig, ImageGeneratorBackend
from MIID.miner.generator_backends.comfyui import ComfyUIGeneratorBackend
from MIID.miner.generator_backends.flux import FluxGeneratorBackend
from MIID.miner.generator_backends.sdxl_img2img import SdxlImg2ImgGeneratorBackend
from MIID.miner.generator_backends.zimage import ZImageGeneratorBackend

__all__ = [
    "GenerationConfig",
    "ComfyUIGeneratorBackend",
    "ImageGeneratorBackend",
    "FluxGeneratorBackend",
    "SdxlImg2ImgGeneratorBackend",
    "ZImageGeneratorBackend",
    "get_image_generator_backend",
    "resolve_image_generation_backend_name",
]

_BACKEND_REGISTRY: Dict[str, Type[ImageGeneratorBackend]] = {
    "comfyui": ComfyUIGeneratorBackend,
    "flux": FluxGeneratorBackend,
    "sdxl_img2img": SdxlImg2ImgGeneratorBackend,
    "zimage": ZImageGeneratorBackend,
}

_backend_singletons: Dict[str, ImageGeneratorBackend] = {}


def resolve_image_generation_backend_name() -> str:
    """Read SN54_IMAGE_GENERATION_BACKEND (default: comfyui)."""
    raw = (os.environ.get("SN54_IMAGE_GENERATION_BACKEND") or "comfyui").strip().lower()
    if raw in _BACKEND_REGISTRY:
        return raw
    bt.logging.warning(
        f"Unknown SN54_IMAGE_GENERATION_BACKEND={raw!r}; falling back to 'comfyui'. "
        f"Valid: {', '.join(sorted(_BACKEND_REGISTRY))}"
    )
    return "comfyui"


def get_image_generator_backend(name: str | None = None) -> ImageGeneratorBackend:
    """Return a named backend singleton (lazy instantiation)."""
    resolved_name = (name or resolve_image_generation_backend_name()).strip().lower()
    if resolved_name not in _BACKEND_REGISTRY:
        resolved_name = resolve_image_generation_backend_name()
    if resolved_name not in _backend_singletons:
        cls = _BACKEND_REGISTRY[resolved_name]
        _backend_singletons[resolved_name] = cls()
        bt.logging.info(f"SN54 image generation backend: {resolved_name} ({cls.__name__})")
    return _backend_singletons[resolved_name]
