# MIID/miner/generator_backends/
#
# Pluggable SN54 Phase 4 image generation backends (FLUX default; SDXL stub).

from __future__ import annotations

import os
from typing import Dict, Type

import bittensor as bt

from MIID.miner.generator_backends.base import GenerationConfig, ImageGeneratorBackend
from MIID.miner.generator_backends.flux import FluxGeneratorBackend
from MIID.miner.generator_backends.sdxl_img2img import SdxlImg2ImgGeneratorBackend

__all__ = [
    "GenerationConfig",
    "ImageGeneratorBackend",
    "FluxGeneratorBackend",
    "SdxlImg2ImgGeneratorBackend",
    "get_image_generator_backend",
    "resolve_image_generation_backend_name",
]

_BACKEND_REGISTRY: Dict[str, Type[ImageGeneratorBackend]] = {
    "flux": FluxGeneratorBackend,
    "sdxl_img2img": SdxlImg2ImgGeneratorBackend,
}

_backend_singleton: ImageGeneratorBackend | None = None


def resolve_image_generation_backend_name() -> str:
    """Read SN54_IMAGE_GENERATION_BACKEND (default: flux)."""
    raw = (os.environ.get("SN54_IMAGE_GENERATION_BACKEND") or "flux").strip().lower()
    if raw in _BACKEND_REGISTRY:
        return raw
    bt.logging.warning(
        f"Unknown SN54_IMAGE_GENERATION_BACKEND={raw!r}; falling back to 'flux'. "
        f"Valid: {', '.join(sorted(_BACKEND_REGISTRY))}"
    )
    return "flux"


def get_image_generator_backend() -> ImageGeneratorBackend:
    """Return the configured backend singleton (lazy instantiation)."""
    global _backend_singleton
    if _backend_singleton is None:
        name = resolve_image_generation_backend_name()
        cls = _BACKEND_REGISTRY[name]
        _backend_singleton = cls()
        bt.logging.info(f"SN54 image generation backend: {name} ({cls.__name__})")
    return _backend_singleton
