# MIID/miner/generator_backends/flux.py
#
# FLUX.2-klein (Flux2KleinPipeline) backend — delegates to generate_variations.py.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from MIID.miner.generate_variations import (
    ensure_flux_pipeline_loaded,
    generate_variations as generate_variations_flux,
    prewarm_pipeline as prewarm_pipeline_flux,
)
from MIID.miner.generator_backends.base import GenerationConfig, ImageGeneratorBackend


class FluxGeneratorBackend(ImageGeneratorBackend):
    """Current default miner path: batched FLUX img2img-style variation generation."""

    def load(self) -> None:
        ensure_flux_pipeline_loaded()

    def prewarm(self) -> None:
        prewarm_pipeline_flux()

    def generate_candidates(
        self,
        base_image: Any,
        compiled_request: Sequence[Any],
        generation_config: GenerationConfig,
        timings_out: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        return generate_variations_flux(
            base_image,
            list(compiled_request),
            candidates_per_request=generation_config.candidates_per_request,
            request_batch_size=generation_config.request_batch_size,
            num_inference_steps_override=generation_config.num_inference_steps_override,
            guidance_scale_override=generation_config.guidance_scale_override,
            timings_out=timings_out,
        )
