# MIID/miner/generator_backends/sdxl_img2img.py
#
# Placeholder for SDXL img2img + face conditioning (IP-Adapter / PuLID / similar).
# Wire real pipelines here: load weights in load(), optional dummy step in prewarm(),
# and implement generate_candidates() to return the same dict shape as FluxGeneratorBackend.

from __future__ import annotations

import bittensor as bt
from typing import Any, Dict, List, Optional, Sequence

from MIID.miner.generator_backends.base import GenerationConfig, ImageGeneratorBackend


class SdxlImg2ImgGeneratorBackend(ImageGeneratorBackend):
    """Stub backend — select with SN54_IMAGE_GENERATION_BACKEND=sdxl_img2img."""

    def load(self) -> None:
        bt.logging.info(
            "SDXL img2img backend: load() is a stub; no weights loaded. "
            "Implement MIID.miner.generator_backends.sdxl_img2img.SdxlImg2ImgGeneratorBackend.load"
        )

    def prewarm(self) -> None:
        bt.logging.debug("SDXL img2img backend: prewarm() no-op (stub).")

    def generate_candidates(
        self,
        base_image: Any,
        compiled_request: Sequence[Any],
        generation_config: GenerationConfig,
        timings_out: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "SN54_IMAGE_GENERATION_BACKEND=sdxl_img2img is not implemented yet. "
            "Add an SDXL img2img pipeline with optional face conditioning in "
            "MIID.miner.generator_backends.sdxl_img2img, or set "
            "SN54_IMAGE_GENERATION_BACKEND=flux. "
            "Target contract: same return shape as FluxGeneratorBackend.generate_candidates "
            "(variation_type, intensity, prompt, candidates list per request) and populate "
            "timings_out with variation_request_prepare_ms and flux_generation_ms for miner metrics."
        )
