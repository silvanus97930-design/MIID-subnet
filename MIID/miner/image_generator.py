# MIID/miner/image_generator.py
#
# Phase 4: Image variation generator for miners.
# Uses FLUX-based generation from generate_variations.py when configured
# (HF token + model). See MIID.miner.generate_variations module docstring for setup.

import base64
import hashlib
import io
import os
import time
from typing import Any, Dict, List, Optional

import bittensor as bt
from PIL import Image
import torch

from MIID.miner.ada_face_compare import (
    extract_face_embedding,
    load_adaface_model,
    score_variation_candidates,
    validate_single_variation,
)
from MIID.miner.generate_variations import (
    generate_variations as generate_variations_flux,
    prewarm_pipeline as prewarm_pipeline_flux,
)
from MIID.miner.pipeline_observability import log_phase4_json


def decode_base_image(base64_image: str) -> Image.Image:
    """Decode a Base64 encoded image to a PIL Image."""
    image_bytes = base64.b64decode(base64_image)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Encode a PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def calculate_image_hash(image_bytes: bytes) -> str:
    """Calculate SHA256 hash of image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def prewarm_flux_pipeline() -> None:
    """Load and optionally warm FLUX once after miner startup."""
    prewarm_pipeline_flux()


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or "").strip() or str(default)))
    except Exception:
        return default


def _select_best_candidate(scores: List[Optional[float]]) -> tuple[int, Optional[float]]:
    best_idx = 0
    best_score: Optional[float] = None
    for idx, score in enumerate(scores):
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_idx = idx
            best_score = float(score)
    return best_idx, best_score


def generate_variations(
    base_image: Image.Image,
    variation_requests: List,
    candidates_per_request_override: Optional[int] = None,
    request_batch_size_override: Optional[int] = None,
    pipeline_timings_out: Optional[Dict[str, float]] = None,
    obs_context: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """Generate best-scoring image variation per request.

    Workflow:
    1) FLUX produces multiple candidates per request using batched GPU inference.
    2) AdaFace scores every candidate via cosine similarity.
    3) The highest-scoring candidate is selected and returned.

    Returns one selected variation per request with metadata including:
      - adaface_similarity
      - candidate_scores
      - candidate_count
      - selected_candidate_index
    """
    if not variation_requests:
        return []

    if candidates_per_request_override is None:
        candidates_per_request = _int_env(
            "PHASE4_CANDIDATES_PER_REQUEST",
            _int_env("FLUX_CANDIDATES_PER_REQUEST", 4),
        )
    else:
        candidates_per_request = max(1, int(candidates_per_request_override))

    if request_batch_size_override is None:
        request_batch_size = _int_env(
            "PHASE4_GENERATION_BATCH_SIZE",
            _int_env("FLUX_GENERATION_BATCH_SIZE", 4),
        )
    else:
        request_batch_size = max(1, int(request_batch_size_override))

    def _float_env(name: str, default: float) -> float:
        try:
            raw = os.environ.get(name)
            return float(str(raw).strip()) if raw is not None else float(default)
        except Exception:
            return float(default)

    # Extra quality for far pose edits: they are near-profile and identity is harder
    # to preserve, so we allow more candidates and slightly different FLUX settings.
    num_inference_steps_override: int | None = None
    guidance_scale_override: float | None = None
    if len(variation_requests) == 1:
        req = variation_requests[0]
        req_type = str(getattr(req, "type", None) or (req.get("type") if isinstance(req, dict) else None) or "").strip()
        req_intensity = str(getattr(req, "intensity", None) or (req.get("intensity") if isinstance(req, dict) else None) or "").strip().lower()
        if req_type == "pose_edit" and req_intensity == "far":
            far_min = _int_env("PHASE4_POSE_FAR_CANDIDATES_PER_REQUEST_MIN", 4)
            far_max = _int_env("PHASE4_POSE_FAR_CANDIDATES_PER_REQUEST_MAX", 8)
            candidates_per_request = min(max(candidates_per_request, far_min), far_max)

            num_inference_steps_override = _int_env(
                "FLUX_NUM_INFERENCE_STEPS_POSE_FAR",
                _int_env("FLUX_NUM_INFERENCE_STEPS", 20) + 5,
            )
            guidance_scale_override = _float_env("FLUX_GUIDANCE_SCALE_POSE_FAR", 3.0)

        if req_type == "lighting_edit" and req_intensity == "medium":
            # Lighting changes can significantly shift embedding similarity; we
            # broaden candidate sampling and spend a few extra inference steps.
            light_min = _int_env("PHASE4_LIGHTING_MEDIUM_CANDIDATES_PER_REQUEST_MIN", 3)
            light_max = _int_env("PHASE4_LIGHTING_MEDIUM_CANDIDATES_PER_REQUEST_MAX", 6)
            candidates_per_request = min(max(candidates_per_request, light_min), light_max)

            num_inference_steps_override = _int_env(
                "FLUX_NUM_INFERENCE_STEPS_LIGHTING_MEDIUM",
                _int_env("FLUX_NUM_INFERENCE_STEPS", 20) + 3,
            )
            guidance_scale_override = _float_env("FLUX_GUIDANCE_SCALE_LIGHTING_MEDIUM", 3.5)

        if req_type == "screen_replay" and req_intensity == "standard":
            # Screen-capture is harder for identity embeddings due to display optics
            # (moire/glare). Sample a few more candidates and give FLUX a bit
            # more budget to keep identity stable.
            screen_min = _int_env("PHASE4_SCREEN_REPLAY_STANDARD_CANDIDATES_PER_REQUEST_MIN", 3)
            screen_max = _int_env("PHASE4_SCREEN_REPLAY_STANDARD_CANDIDATES_PER_REQUEST_MAX", 6)
            candidates_per_request = min(max(candidates_per_request, screen_min), screen_max)

            num_inference_steps_override = _int_env(
                "FLUX_NUM_INFERENCE_STEPS_SCREEN_REPLAY_STANDARD",
                _int_env("FLUX_NUM_INFERENCE_STEPS", 20) + 3,
            )
            guidance_scale_override = _float_env("FLUX_GUIDANCE_SCALE_SCREEN_REPLAY_STANDARD", 3.2)
    adaface_workers = _int_env("ADAFACE_ALIGN_WORKERS", 4)
    default_adaface_device = "cuda" if torch.cuda.is_available() else "cpu"
    adaface_device = (os.environ.get("ADAFACE_DEVICE", default_adaface_device) or "cpu").strip().lower()

    flux_timings: Dict[str, float] = {}
    raw_results = generate_variations_flux(
        base_image,
        variation_requests,
        candidates_per_request=candidates_per_request,
        request_batch_size=request_batch_size,
        num_inference_steps_override=num_inference_steps_override,
        guidance_scale_override=guidance_scale_override,
        timings_out=flux_timings,
    )

    # Keep one model and one base embedding per request to avoid repeated overhead.
    t_ada0 = time.perf_counter()
    adaface_model = load_adaface_model(device=adaface_device)
    base_embedding = extract_face_embedding(adaface_model, base_image, device=adaface_device)
    adaface_setup_ms = (time.perf_counter() - t_ada0) * 1000.0

    if base_embedding is None:
        bt.logging.warning(
            "AdaFace could not extract base embedding. Falling back to first generated candidate per request."
        )

    variations: List[Dict] = []
    adaface_rerank_ms = 0.0
    variation_packaging_ms = 0.0

    for item in raw_results:
        var_type = item.get("variation_type", "unknown")
        intensity = item.get("intensity", "standard")
        prompt = item.get("prompt", "")
        candidates = list(item.get("candidates", []) or [])

        if not candidates:
            bt.logging.warning(f"No candidates generated for {var_type}({intensity})")
            continue

        tr0 = time.perf_counter()
        if base_embedding is None:
            scores = [None] * len(candidates)
        else:
            scores = score_variation_candidates(
                base_image,
                candidates,
                model=adaface_model,
                device=adaface_device,
                parallel_workers=adaface_workers,
                base_embedding=base_embedding,
            )

        best_idx, best_score = _select_best_candidate(scores)
        adaface_rerank_ms += (time.perf_counter() - tr0) * 1000.0

        best_image = candidates[best_idx]

        tp0 = time.perf_counter()
        image_bytes = encode_image_to_bytes(best_image)
        image_hash = calculate_image_hash(image_bytes)

        score_payload: List[Optional[float]] = [float(s) if s is not None else None for s in scores]

        variations.append(
            {
                "image": best_image,
                "variation_type": var_type,
                "image_bytes": image_bytes,
                "image_hash": image_hash,
                "adaface_similarity": float(best_score) if best_score is not None else 0.0,
                "candidate_scores": score_payload,
                "candidate_count": len(candidates),
                "selected_candidate_index": int(best_idx),
                "prompt": prompt,
            }
        )
        variation_packaging_ms += (time.perf_counter() - tp0) * 1000.0

        score_text = f"{best_score:.4f}" if best_score is not None else "None"
        bt.logging.debug(
            f"Generated {var_type}({intensity}) best candidate "
            f"idx={best_idx}/{len(candidates)} score={score_text} hash={image_hash[:16]}..."
        )

    bt.logging.info(
        f"Generated {len(variations)} best variations "
        f"(candidates/request={candidates_per_request}, batch_size={request_batch_size})"
    )

    reranking_total_ms = adaface_setup_ms + adaface_rerank_ms
    total_ms = (
        float(flux_timings.get("variation_request_prepare_ms", 0.0))
        + float(flux_timings.get("flux_generation_ms", 0.0))
        + reranking_total_ms
        + variation_packaging_ms
    )
    stage_timings_ms = {
        "request_parse_ms": round(float(flux_timings.get("variation_request_prepare_ms", 0.0)), 4),
        "generation_ms": round(float(flux_timings.get("flux_generation_ms", 0.0)), 4),
        "reranking_ms": round(reranking_total_ms, 4),
        "final_packaging_ms": round(variation_packaging_ms, 4),
    }
    if pipeline_timings_out is not None:
        pipeline_timings_out.update(
            {
                "variation_request_prepare_ms": float(flux_timings.get("variation_request_prepare_ms", 0.0)),
                "flux_generation_ms": float(flux_timings.get("flux_generation_ms", 0.0)),
                "adaface_setup_ms": adaface_setup_ms,
                "adaface_rerank_ms": adaface_rerank_ms,
                "variation_packaging_ms": variation_packaging_ms,
                "generate_variations_total_ms": total_ms,
            }
        )

    log_payload: Dict[str, Any] = {
        "stage_timings_ms": stage_timings_ms,
        "candidates_per_request": candidates_per_request,
        "request_batch_size": request_batch_size,
        "variations_returned": len(variations),
        "generate_variations_total_ms": round(total_ms, 4),
    }
    if obs_context:
        log_payload.update({k: v for k, v in obs_context.items() if v is not None})
    log_phase4_json("phase4_generate_variations", log_payload)

    return variations


def validate_variation(
    variation: Dict,
    base_image: Image.Image,
    min_similarity: float = 0.7,
) -> bool:
    """Backward-compatible boolean identity check."""
    similarity = variation.get("adaface_similarity")
    if similarity is not None:
        try:
            return float(similarity) >= float(min_similarity)
        except Exception:
            pass

    return validate_single_variation(
        base_image,
        variation["image"],
        min_similarity=min_similarity,
    )
