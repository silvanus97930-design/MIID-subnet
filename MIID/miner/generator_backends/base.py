# MIID/miner/generator_backends/base.py
#
# Abstract image generation backend for SN54 Phase 4 (multiple diffusion backends).

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class GenerationConfig:
    """Per-call knobs for candidate generation (batching and diffusion overrides)."""

    candidates_per_request: int
    request_batch_size: int
    num_inference_steps_override: Optional[int] = None
    guidance_scale_override: Optional[float] = None


class ImageGeneratorBackend(ABC):
    """Load a diffusion stack once, optionally prewarm, then produce raw candidates.

    Implementations should write timing breakdowns into ``timings_out`` using keys
    ``variation_request_prepare_ms`` and ``flux_generation_ms`` so Phase 4
    observability and ``neurons/miner.py`` stay compatible (the latter key names
    the primary image-generation stage regardless of backend).
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights / pipeline (no user inference required)."""

    @abstractmethod
    def prewarm(self) -> None:
        """Optional warm-up after load (e.g. dummy forward) to reduce first-request latency."""

    @abstractmethod
    def generate_candidates(
        self,
        base_image: Any,
        compiled_request: Sequence[Any],
        generation_config: GenerationConfig,
        timings_out: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run diffusion and return raw candidate lists per request.

        Args:
            base_image: PIL ``Image`` (RGB) face crop / seed.
            compiled_request: Sequence of protocol-compatible objects (``type``,
                ``intensity``, ``description``, ``detail``) or dicts with those keys.
                This is the batch for one ``generate_variations`` call from
                ``image_generator`` (often a single compiled variation).
            generation_config: Batching and step/CFG overrides.
            timings_out: Optional dict to populate with backend-specific timings.

        Returns:
            List of dicts, each with ``variation_type``, ``intensity``, ``prompt``,
            ``candidates`` (list of PIL images), matching the existing FLUX contract.
        """
