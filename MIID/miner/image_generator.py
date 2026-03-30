# MIID/miner/image_generator.py
#
# Phase 4: Image variation generator for miners.
# Delegates raw candidate generation to a pluggable backend (default: FLUX via
# MIID.miner.generator_backends). Select with SN54_IMAGE_GENERATION_BACKEND=flux|sdxl_img2img.
# See MIID.miner.generate_variations module docstring for FLUX setup (HF token + model).

import base64
import hashlib
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import bittensor as bt
from PIL import Image
import torch

from MIID.miner.ada_face_compare import validate_single_variation
from MIID.miner.face_preprocess import FacePreprocessResult, preprocess_seed_face
from MIID.miner.adherence import VariationAdherenceContext
from MIID.miner.generator_backends import GenerationConfig, get_image_generator_backend
from MIID.miner.identity_scoring import IdentityScoreResult, IdentityScoringService
from MIID.miner.layered_edit import (
    layered_background_enabled,
    layered_screen_replay_enabled,
    prepare_screen_replay_subject,
    preserve_subject_over_candidate,
)
from MIID.miner.pipeline_observability import log_phase4_json
from MIID.miner.reranker import (
    build_candidate_scores,
    leaderboard_entries,
    load_rerank_config_from_env,
    select_best_candidate_index,
)
from MIID.miner.screen_replay import (
    build_raw_results_screen_replay,
    extract_identity_focus_crop,
    screen_replay_pipeline_enabled,
)


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
    """Load and optionally warm the configured image generation backend after miner startup.

    Name kept for backward compatibility; when SN54_IMAGE_GENERATION_BACKEND=flux this
    matches the previous FLUX-only prewarm behavior.
    """
    get_image_generator_backend().prewarm()


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or "").strip() or str(default)))
    except Exception:
        return default


def _face_preprocess_summary(prep: FacePreprocessResult) -> Dict[str, Any]:
    return {
        "ok": prep.ok,
        "face_count": prep.face_count,
        "dominant_index": prep.dominant_index,
        "message": prep.message,
        "warnings": list(prep.warnings),
    }


def _primary_identity_score(r: IdentityScoreResult) -> Optional[float]:
    if r.primary_similarity is not None:
        return r.primary_similarity
    s = r.adaface.similarity if r.adaface.available else None
    if s is None and r.insightface_arcface.available:
        s = r.insightface_arcface.similarity
    return s


def _req_type(req: Any) -> str:
    return str(
        getattr(req, "type", None) or (req.get("type") if isinstance(req, dict) else None) or ""
    ).strip()


def _identity_scoring_candidates(item: Dict[str, Any], candidates: List[Image.Image]) -> List[Image.Image]:
    if item.get("pipeline") != "screen_replay_dedicated":
        return candidates

    metas = item.get("screen_replay_candidate_metas")
    if not isinstance(metas, list) or len(metas) != len(candidates):
        return candidates

    return [
        extract_identity_focus_crop(candidate, meta)
        for candidate, meta in zip(candidates, metas)
    ]


def _repair_screen_replay_identity_results(
    item: Dict[str, Any],
    candidates: List[Image.Image],
    identity_service: IdentityScoringService,
    results: List[IdentityScoreResult],
) -> List[IdentityScoreResult]:
    if item.get("pipeline") != "screen_replay_dedicated":
        return results

    repaired = list(results)
    recovered = 0
    for idx, res in enumerate(repaired):
        if _primary_identity_score(res) is not None:
            continue
        try:
            alt = identity_service.score_candidate(candidates[idx].convert("RGB"))
        except Exception:
            continue
        if _primary_identity_score(alt) is not None:
            repaired[idx] = alt
            recovered += 1

    if recovered:
        bt.logging.info(
            "Phase4 screen_replay identity fallback recovered %d/%d null crop scores",
            recovered,
            len(candidates),
        )
    return repaired


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
    1) The configured backend (default FLUX) produces multiple candidates per request.
    2) Identity backends score every candidate; an ensemble reranker combines
       ArcFace, AdaFace, adherence (CLIP-free MSE-to-target), realism, and duplicate penalties.
    3) The candidate with the highest ensemble ``final_score`` is selected (deterministic tie-break).

    Returns one selected variation per request with metadata including:
      - adaface_similarity (best candidate; AdaFace-first, else ArcFace)
      - candidate_scores (legacy: primary identity similarity per candidate)
      - ensemble_final_scores, ensemble_leaderboard, candidate_rerank_scores
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

        if req_type == "expression_edit" and req_intensity == "far":
            # Strong expression edits need extra search budget; otherwise reranking
            # often settles for high-adherence / low-identity outliers.
            expr_min = _int_env("PHASE4_EXPRESSION_FAR_CANDIDATES_PER_REQUEST_MIN", 4)
            expr_max = _int_env("PHASE4_EXPRESSION_FAR_MAX_CANDIDATES_PER_REQUEST", 8)
            candidates_per_request = min(max(candidates_per_request, expr_min), expr_max)

            num_inference_steps_override = _int_env(
                "FLUX_NUM_INFERENCE_STEPS_EXPRESSION_FAR",
                _int_env("FLUX_NUM_INFERENCE_STEPS", 20) + 4,
            )
            guidance_scale_override = _float_env("FLUX_GUIDANCE_SCALE_EXPRESSION_FAR", 3.1)

        if req_type == "background_edit" and req_intensity in {"light", "medium"}:
            bg_min = _int_env(
                f"PHASE4_BACKGROUND_{req_intensity.upper()}_CANDIDATES_PER_REQUEST_MIN",
                4,
            )
            bg_max = _int_env(
                f"PHASE4_BACKGROUND_{req_intensity.upper()}_CANDIDATES_PER_REQUEST_MAX",
                6,
            )
            candidates_per_request = min(max(candidates_per_request, bg_min), bg_max)

            num_inference_steps_override = _int_env(
                f"FLUX_NUM_INFERENCE_STEPS_BACKGROUND_{req_intensity.upper()}",
                _int_env("FLUX_NUM_INFERENCE_STEPS", 20) + 2,
            )
            guidance_scale_override = _float_env(
                f"FLUX_GUIDANCE_SCALE_BACKGROUND_{req_intensity.upper()}",
                2.9,
            )

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
            guidance_scale_override = _float_env("FLUX_GUIDANCE_SCALE_LIGHTING_MEDIUM", 3.0)

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
    default_adaface_device = "cuda" if torch.cuda.is_available() else "cpu"
    adaface_device = (os.environ.get("ADAFACE_DEVICE", default_adaface_device) or "cpu").strip().lower()

    generation_timings: Dict[str, float] = {}
    use_sr_dedicated = (
        screen_replay_pipeline_enabled()
        and len(variation_requests) == 1
        and _req_type(variation_requests[0]) == "screen_replay"
    )
    screen_replay_layered_meta: Optional[Dict[str, Any]] = None
    face_prep: FacePreprocessResult
    if use_sr_dedicated:
        t_face0 = time.perf_counter()
        face_prep = preprocess_seed_face(base_image, adaface_device)
        face_preprocess_ms = (time.perf_counter() - t_face0) * 1000.0
        if face_prep.warnings:
            bt.logging.debug(f"Phase 4 face preprocess: {'; '.join(face_prep.warnings)}")
        sr_source = face_prep.aligned_face if face_prep.ok else None
        if layered_screen_replay_enabled():
            try:
                sr_source, screen_replay_layered_meta = prepare_screen_replay_subject(
                    base_image,
                    face_prep.face_box_xyxy if face_prep.ok else None,
                )
            except Exception as e:
                bt.logging.warning(f"Phase 4 layered screen_replay helper failed: {e}")
        raw_results = build_raw_results_screen_replay(
            base_image,
            variation_requests[0],
            candidates_per_request,
            generation_timings,
            aligned_face=sr_source,
        )
        if not raw_results:
            bt.logging.warning(
                "SN54 dedicated screen_replay: missing device/cue metadata; falling back to diffusion backend"
            )
            use_sr_dedicated = False
    if not use_sr_dedicated:
        backend = get_image_generator_backend()
        gen_config = GenerationConfig(
            candidates_per_request=candidates_per_request,
            request_batch_size=request_batch_size,
            num_inference_steps_override=num_inference_steps_override,
            guidance_scale_override=guidance_scale_override,
        )
        raw_results = backend.generate_candidates(
            base_image,
            variation_requests,
            gen_config,
            timings_out=generation_timings,
        )
        t_prep0 = time.perf_counter()
        face_prep = preprocess_seed_face(base_image, adaface_device)
        face_preprocess_ms = (time.perf_counter() - t_prep0) * 1000.0
        if face_prep.warnings:
            bt.logging.debug(f"Phase 4 face preprocess: {'; '.join(face_prep.warnings)}")

    t_id0 = time.perf_counter()
    identity_service = IdentityScoringService(device=adaface_device)
    identity_service.begin_request(
        base_image,
        preprocessed_aligned=face_prep.aligned_face if face_prep.ok else None,
    )
    adaface_setup_ms = (time.perf_counter() - t_id0) * 1000.0

    rerank_cfg = load_rerank_config_from_env()

    variations: List[Dict] = []
    adaface_rerank_ms = 0.0
    variation_packaging_ms = 0.0

    for item in raw_results:
        var_type = item.get("variation_type", "unknown")
        intensity = item.get("intensity", "standard")
        prompt = item.get("prompt", "")
        description = str(item.get("description") or "")
        detail = str(item.get("detail") or "")
        candidates = list(item.get("candidates", []) or [])
        layered_adjustments: List[Dict[str, Any]] = []

        if not candidates:
            bt.logging.warning(f"No candidates generated for {var_type}({intensity})")
            continue

        if var_type == "background_edit" and layered_background_enabled():
            patched_candidates: List[Image.Image] = []
            for cand in candidates:
                try:
                    patched, meta = preserve_subject_over_candidate(
                        base_image,
                        cand,
                        face_prep.face_box_xyxy if face_prep.ok else None,
                    )
                    patched_candidates.append(patched)
                    layered_adjustments.append(meta)
                except Exception as e:
                    patched_candidates.append(cand)
                    layered_adjustments.append({"layered_subject_used": False, "error": str(e)})
            candidates = patched_candidates

        tr0 = time.perf_counter()
        identity_candidates = _identity_scoring_candidates(item, candidates)
        id_results_struct = identity_service.score_candidates(identity_candidates)
        id_results_struct = _repair_screen_replay_identity_results(
            item,
            candidates,
            identity_service,
            id_results_struct,
        )
        scores = [_primary_identity_score(r) for r in id_results_struct]

        sd = item.get("screen_replay_device")
        vk = item.get("visual_cue_keys")
        adherence_ctx = VariationAdherenceContext(
            variation_type=str(var_type),
            intensity=str(intensity),
            description=description,
            detail=detail,
            prompt=str(prompt),
            screen_replay_device=str(sd).strip() if sd else None,
            visual_cue_keys=tuple(vk) if isinstance(vk, (list, tuple)) and vk else None,
        )
        ensemble_rows = build_candidate_scores(
            id_results_struct,
            base_image,
            candidates,
            config=rerank_cfg,
            variation_context=adherence_ctx,
        )
        best_idx = select_best_candidate_index(ensemble_rows)
        adaface_rerank_ms += (time.perf_counter() - tr0) * 1000.0

        best_image = candidates[best_idx]

        tp0 = time.perf_counter()
        image_bytes = encode_image_to_bytes(best_image)
        image_hash = calculate_image_hash(image_bytes)

        score_payload: List[Optional[float]] = [float(s) if s is not None else None for s in scores]
        identity_scores_payload: List[Dict[str, Any]] = [r.to_dict() for r in id_results_struct]
        best_identity = (
            id_results_struct[best_idx].to_dict()
            if id_results_struct and 0 <= best_idx < len(id_results_struct)
            else None
        )
        best_primary = scores[best_idx] if scores and 0 <= best_idx < len(scores) else None
        ensemble_final_scores = [float(s.final_score) for s in ensemble_rows]
        candidate_rerank_scores = [s.to_dict() for s in ensemble_rows]
        variation_label = f"{var_type}({intensity})"
        board = leaderboard_entries(ensemble_rows, variation_label=variation_label)
        bt.logging.info(
            "Phase4 ensemble leaderboard %s: %s",
            variation_label,
            json.dumps(board, default=str),
        )

        variations.append(
            {
                "image": best_image,
                "variation_type": var_type,
                "intensity": intensity,
                "image_bytes": image_bytes,
                "image_hash": image_hash,
                "adaface_similarity": float(best_primary) if best_primary is not None else 0.0,
                "candidate_scores": score_payload,
                "ensemble_final_scores": ensemble_final_scores,
                "ensemble_leaderboard": board,
                "candidate_rerank_scores": candidate_rerank_scores,
                "candidate_count": len(candidates),
                "selected_candidate_index": int(best_idx),
                "prompt": prompt,
                "identity_scores": identity_scores_payload,
                "identity_score_selected": best_identity,
                "face_preprocess": _face_preprocess_summary(face_prep),
                "layered_adjustments": layered_adjustments,
                "screen_replay_layered_helper": screen_replay_layered_meta,
            }
        )
        variation_packaging_ms += (time.perf_counter() - tp0) * 1000.0

        best_final = (
            ensemble_rows[best_idx].final_score
            if ensemble_rows and 0 <= best_idx < len(ensemble_rows)
            else None
        )
        ef = f"{best_final:.4f}" if best_final is not None else "n/a"
        pf = f"{best_primary:.4f}" if best_primary is not None else "n/a"
        score_text = f"ensemble={ef} primary={pf}"
        bt.logging.debug(
            f"Generated {var_type}({intensity}) best candidate "
            f"idx={best_idx}/{len(candidates)} {score_text} hash={image_hash[:16]}..."
        )

    bt.logging.info(
        f"Generated {len(variations)} best variations "
        f"(candidates/request={candidates_per_request}, batch_size={request_batch_size})"
    )

    identity_service.end_request()

    reranking_total_ms = adaface_setup_ms + adaface_rerank_ms
    total_ms = (
        float(generation_timings.get("variation_request_prepare_ms", 0.0))
        + float(generation_timings.get("flux_generation_ms", 0.0))
        + reranking_total_ms
        + variation_packaging_ms
    )
    stage_timings_ms = {
        "request_parse_ms": round(float(generation_timings.get("variation_request_prepare_ms", 0.0)), 4),
        "generation_ms": round(float(generation_timings.get("flux_generation_ms", 0.0)), 4),
        "reranking_ms": round(reranking_total_ms, 4),
        "final_packaging_ms": round(variation_packaging_ms, 4),
    }
    if pipeline_timings_out is not None:
        pipeline_timings_out.update(
            {
                "variation_request_prepare_ms": float(generation_timings.get("variation_request_prepare_ms", 0.0)),
                "flux_generation_ms": float(generation_timings.get("flux_generation_ms", 0.0)),
                "face_preprocess_ms": face_preprocess_ms,
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
