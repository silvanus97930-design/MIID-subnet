# The MIT License (MIT)
# Copyright © 2025 YANEZ
# MIID miner: FLUX-based image variation generation.

"""
FLUX-based image variation generation for the MIID miner (Phase 4).

This module generates identity-preserving image variations (pose, expression,
lighting, background) from a base face image using the FLUX.2-klein diffusion
model. The miner calls the generate_variations() method from here (via
image_generator) when validators send an image_request in an IdentitySynapse.

================================================================================
SETUP: Steps to use this module
================================================================================

1. Create a Hugging Face account (free):
   - Go to https://huggingface.co/join and sign up.

2. Accept the model license and get a token:
   - Open https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
   - Click "Agree and access repository" if required.
   - Go to https://huggingface.co/settings/tokens and create a token (read access).

3. Set your token in the environment before running the miner:
   - export HF_TOKEN="hf_..."
   or
   - export HUGGINGFACE_TOKEN="hf_..."

4. Download the model (optional but recommended before running the miner):
   - Run: python -m MIID.miner.downloading_model
   - Or use the script in this repo that loads FLUX.2-klein so weights are cached.

5. Install dependencies (see requirements.txt):
   - torch, diffusers, transformers, PIL, etc.

BASE MODEL:
  - Default: black-forest-labs/FLUX.2-klein-4B (Flux2KleinPipeline).
  - This is the base miner model used for image-to-image editing.

OTHER MODELS YOU CAN USE:
  - Other FLUX variants (e.g. FLUX.1-dev, FLUX.1-schnell) with their matching
    pipeline classes (FluxPipeline, FluxImg2ImgPipeline, etc.).
  - Custom fine-tuned checkpoints: change MODEL_ID and use the pipeline class
    that matches the checkpoint (e.g. Flux2KleinPipeline for klein-style models).

To switch models: set MODEL_ID and use the correct pipeline class in
_get_pipeline(); adjust prompts if the model expects different wording.
"""

import os
import threading
from contextlib import nullcontext
from typing import Any, Dict, List

import bittensor as bt
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration (parameters you can change)
# -----------------------------------------------------------------------------


def _normalize_device_name(device_name: str) -> str:
    value = (device_name or "").strip().lower()
    if not value:
        return ""
    if value.startswith("cuda") and not torch.cuda.is_available():
        bt.logging.warning(
            f"FLUX_DEVICE={device_name} requested but CUDA is unavailable. Falling back to CPU."
        )
        return "cpu"
    return value


def _resolve_device() -> str:
    requested = _normalize_device_name(os.environ.get("FLUX_DEVICE", ""))
    if requested:
        return requested
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
    requested = _parse_dtype(os.environ.get("FLUX_DTYPE", "auto"))
    if requested is not None:
        if requested in (torch.float16, torch.bfloat16) and not device.startswith("cuda"):
            bt.logging.warning(
                "FLUX_DTYPE requested GPU-only precision on non-CUDA device. Falling back to float32."
            )
            return torch.float32
        return requested

    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# Device: "cuda" for NVIDIA GPU, "mps" for Apple Silicon, "cpu" for CPU-only.
# GPU is strongly recommended for acceptable speed.
DEVICE = _resolve_device()

# dtype: torch.float32 (safest), torch.float16 or torch.bfloat16 (less memory,
# faster on GPU). Use float32 on CPU; float16/bfloat16 on GPU if supported.
DTYPE = _resolve_dtype(DEVICE)

# Hugging Face model ID. This is the base miner model; change if using another
# FLUX or compatible diffusion model (see docstring for alternatives).
MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")

# Inference steps: more steps = higher quality, slower. Typical range 20–50.
# Lower (e.g. 20) for speed; higher for quality.
NUM_INFERENCE_STEPS = int(os.environ.get("FLUX_NUM_INFERENCE_STEPS", "20"))

# Guidance scale: how closely the output follows the prompt. Typical 3.5–7.5.
# Higher = stronger adherence to prompt; lower = more variation.
GUIDANCE_SCALE = float(os.environ.get("FLUX_GUIDANCE_SCALE", "3.5"))

# Default intensity when the caller does not specify one (light / medium / far).
DEFAULT_INTENSITY = "medium"

# When true, only one image-request batch is processed at a time. This avoids
# memory spikes when multiple validators hit Phase 4 concurrently.
SERIALIZE_VARIATION_REQUESTS = (
    os.environ.get("FLUX_SERIALIZE_REQUESTS", "true").strip().lower() in {"1", "true", "yes"}
)

# -----------------------------------------------------------------------------
# Prompts: use protocol text from validator (single source of truth)
# -----------------------------------------------------------------------------
# Each VariationRequest from the validator includes .description (type) and .detail
# (intensity-specific) from IMAGE_VARIATION_TYPES in validator/image_variations.py.
# We use that as the FLUX prompt so the miner does not duplicate prompt definitions.

# Lazy-loaded pipeline (loaded on first use to avoid requiring HF token at import).
_pipe: Flux2KleinPipeline | None = None
_PIPE_LOCK = threading.Lock()


def _get_pipeline() -> Flux2KleinPipeline:
    """Load the FLUX pipeline once and reuse. Uses HF_TOKEN or HUGGINGFACE_TOKEN."""
    global _pipe
    if _pipe is not None:
        return _pipe

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_TOKEN in your environment, e.g.\n"
            '  export HF_TOKEN="hf_..."'
        )

    pipe_kwargs: Dict[str, Any] = {
        "torch_dtype": DTYPE,
        "token": token,
        "low_cpu_mem_usage": True,
    }
    bt.logging.info(
        f"Initializing FLUX pipeline model={MODEL_ID} device={DEVICE} dtype={DTYPE}"
    )
    _pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, **pipe_kwargs)
    if hasattr(_pipe, "enable_attention_slicing"):
        _pipe.enable_attention_slicing()
    if hasattr(_pipe, "enable_vae_slicing"):
        _pipe.enable_vae_slicing()
    if hasattr(_pipe, "enable_vae_tiling"):
        _pipe.enable_vae_tiling()
    _pipe = _pipe.to(DEVICE)
    return _pipe


def _get_type_and_intensity(req: Any) -> tuple[str, str]:
    """Get .type and .intensity from a VariationRequest-like object or dict."""
    var_type = getattr(req, "type", None) or (req.get("type") if isinstance(req, dict) else None)
    intensity = getattr(req, "intensity", None) or (req.get("intensity") if isinstance(req, dict) else None)
    if not var_type:
        raise ValueError("variation_requests entry missing 'type'")
    if intensity not in ("light", "medium", "far"):
        intensity = DEFAULT_INTENSITY
    return (var_type, intensity)


def _get_prompt_from_request(req: Any, var_type: str, intensity: str) -> str:
    """Build FLUX prompt from protocol fields (description + detail from validator)."""
    description = getattr(req, "description", None) or (req.get("description") if isinstance(req, dict) else None) or ""
    detail = getattr(req, "detail", None) or (req.get("detail") if isinstance(req, dict) else None) or ""
    parts = [p.strip() for p in (description, detail) if p and p.strip()]
    if parts:
        return f"Same person, same identity, {', '.join(parts)}. Preserve face identity."
    return f"Same person, same identity, {var_type} variation ({intensity} intensity). Preserve face identity."


def _generation_lock_context():
    return _PIPE_LOCK if SERIALIZE_VARIATION_REQUESTS else nullcontext()


def prewarm_pipeline() -> None:
    """Optional one-time warm-up to reduce first real request latency."""
    warm_enabled = os.environ.get("FLUX_PREWARM_INFERENCE", "true").strip().lower() in {"1", "true", "yes", "on"}
    warm_steps = max(1, int(os.environ.get("FLUX_PREWARM_STEPS", "2")))
    warm_prompt = os.environ.get(
        "FLUX_PREWARM_PROMPT",
        "Same person, same identity, neutral passport photo with slight expression change. Preserve face identity.",
    )

    with _generation_lock_context():
        pipe = _get_pipeline()
        if not warm_enabled:
            bt.logging.info("FLUX prewarm: pipeline loaded (inference warm-up disabled)")
            return

        warm_image = Image.new("RGB", (384, 512), color=(128, 128, 128))
        bt.logging.info(f"FLUX prewarm: running warm-up inference (steps={warm_steps})")
        with torch.inference_mode():
            _ = pipe(
                prompt=warm_prompt,
                image=[warm_image],
                num_inference_steps=warm_steps,
                guidance_scale=max(1.0, min(GUIDANCE_SCALE, 3.0)),
            )

        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()


def generate_variations(
    base_image: Image.Image,
    variation_requests: List[Any],
) -> List[Dict[str, Any]]:
    """
    Generate image variations from a base face image using the FLUX.2-klein model.

    Called by the miner (via image_generator.generate_variations) when processing
    an image_request. For each validator request we use the protocol's .description
    and .detail (from validator IMAGE_VARIATION_TYPES) as the FLUX prompt—single
    source of truth—then run the pipeline with the base image and that prompt.

    Flow: validator sends variation_requests (type, intensity, description, detail)
    -> miner passes them here -> prompt = protocol description + detail
    -> FLUX takes base image + prompt -> one variation image.

    Parameters you can change (module-level constants above):
      DEVICE, DTYPE, MODEL_ID, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, DEFAULT_INTENSITY.
    Prompt text comes from the protocol (validator's IMAGE_VARIATION_TYPES).

    Args:
        base_image: PIL Image of the base face (from validator's image_request.base_image).
        variation_requests: List of validator requests; each has .type and .intensity
            (e.g. VariationRequest from protocol, or dict with "type"/"intensity").
            Order is preserved: first request -> first output.

    Returns:
        List of dicts, each with "image" (PIL.Image) and "variation_type" (str).
        Caller (image_generator) adds image_bytes and image_hash.
    """
    if not variation_requests:
        return []

    with _generation_lock_context():
        pipe = _get_pipeline()
        results: List[Dict[str, Any]] = []

        for req in variation_requests:
            var_type, intensity = _get_type_and_intensity(req)
            prompt = _get_prompt_from_request(req, var_type, intensity)

            try:
                # Run the diffusion pipeline: base image + prompt -> one variation image.
                with torch.inference_mode():
                    out = pipe(
                        prompt=prompt,
                        image=[base_image],
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                    )
                gen_image = out.images[0]
            except Exception as e:
                raise RuntimeError(f"FLUX variation failed for {var_type}({intensity}): {e}") from e

            results.append({
                "image": gen_image,
                "variation_type": var_type,
            })

        if DEVICE.startswith("cuda"):
            # Helps avoid VRAM fragmentation across long-running miner sessions.
            torch.cuda.empty_cache()

        return results
