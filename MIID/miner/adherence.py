# MIID/miner/adherence.py
#
# Task-specific adherence scoring for SN54 Phase 4 image variations (modular validators).
#
# Dispatch: ``score_variation_adherence`` -> ``ADHERENCE_VALIDATORS[variation_type]``.
# Extend via ``register_adherence_validator`` or pass ``AdherenceScorerBundle`` for tests / ML backends.
#
# Env: ``PHASE4_ADHERENCE_PASS_THRESHOLD`` (default 0.48), ``PHASE4_ADHERENCE_SCREEN_MIN_CUES`` (default 2).

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

def _float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _pass_threshold() -> float:
    return max(0.0, min(1.0, _float_env("PHASE4_ADHERENCE_PASS_THRESHOLD", 0.48)))


def _triangular_score(value: float, peak: float, half_width: float) -> float:
    """1 at peak, 0 at peak ± half_width (clamped)."""
    if half_width <= 0:
        return 1.0 if abs(value - peak) < 1e-9 else 0.0
    d = abs(float(value) - float(peak)) / half_width
    return float(max(0.0, min(1.0, 1.0 - d)))


def _band_score(value: float, lo: float, hi: float) -> float:
    """1 inside [lo, hi], linear taper outside."""
    v = float(value)
    if lo <= v <= hi:
        return 1.0
    if v < lo:
        return max(0.0, 1.0 - (lo - v) / max(lo, 1e-6))
    return max(0.0, 1.0 - (v - hi) / max(1.0 - hi, 1e-6))


@dataclass
class VariationAdherenceContext:
    """Wire / protocol context for one variation request."""

    variation_type: str
    intensity: str
    description: str = ""
    detail: str = ""
    prompt: str = ""


@dataclass
class AdherenceResult:
    adherence_score: float  # [0, 1]
    evidence: Dict[str, Any]
    pass_recommendation: bool


@dataclass
class AdherenceScorerBundle:
    """Inject mocked or upgraded metrics in tests / future ML backends."""

    pose_metrics: Optional[Callable[[Image.Image, Image.Image], Dict[str, float]]] = None
    expression_metrics: Optional[Callable[[Image.Image, Image.Image], Dict[str, float]]] = None
    lighting_metrics: Optional[Callable[[Image.Image, Image.Image], Dict[str, float]]] = None
    background_metrics: Optional[Callable[[Image.Image, Image.Image], Dict[str, float]]] = None
    screen_metrics: Optional[Callable[[Image.Image, Image.Image, VariationAdherenceContext], Dict[str, float]]] = None


# --- shared image primitives ---


def _resize_gray(img: Image.Image, size: int = 64) -> np.ndarray:
    return np.asarray(img.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _resize_rgb(img: Image.Image, size: int = 64) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _lr_asymmetry(gray: np.ndarray) -> float:
    h, w = gray.shape
    mid = w // 2
    left = float(np.mean(gray[:, :mid]))
    right = float(np.mean(gray[:, mid:]))
    m = float(np.mean(gray)) + 1e-6
    return (left - right) / m


def _tb_asymmetry(gray: np.ndarray) -> float:
    h, w = gray.shape
    mid = h // 2
    top = float(np.mean(gray[:mid, :]))
    bot = float(np.mean(gray[mid:, :]))
    m = float(np.mean(gray)) + 1e-6
    return (top - bot) / m


def _high_pass_correlation(g0: np.ndarray, g1: np.ndarray) -> float:
    """Correlation of Laplacian-like residuals (structure preservation)."""
    k = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    try:
        import cv2

        r0 = cv2.filter2D(g0, -1, k)
        r1 = cv2.filter2D(g1, -1, k)
    except Exception:
        r0 = g0 - np.mean(g0)
        r1 = g1 - np.mean(g1)
    a = r0.flatten() - np.mean(r0)
    b = r1.flatten() - np.mean(r1)
    da = np.std(a) + 1e-6
    db = np.std(b) + 1e-6
    c = float(np.clip(np.dot(a, b) / (a.size * da * db), -1.0, 1.0))
    return (c + 1.0) * 0.5


def _default_pose_metrics(base: Image.Image, cand: Image.Image) -> Dict[str, float]:
    b = _resize_gray(base, 64)
    c = _resize_gray(cand, 64)
    yaw_b, yaw_c = _lr_asymmetry(b), _lr_asymmetry(c)
    pitch_b, pitch_c = _tb_asymmetry(b), _tb_asymmetry(c)
    return {
        "yaw_delta": abs(yaw_c - yaw_b),
        "pitch_delta": abs(pitch_c - pitch_b),
        "global_change": float(np.mean(np.abs(c - b))),
    }


def _pose_intensity_targets(intensity: str) -> Tuple[float, float, float]:
    """Expected (mid yaw_delta, half_width, min global_change)."""
    i = (intensity or "medium").strip().lower()
    if i == "light":
        return 0.055, 0.05, 0.012
    if i == "far":
        return 0.18, 0.12, 0.04
    return 0.10, 0.08, 0.022  # medium


def validate_pose_edit(
    base_rgb: Image.Image,
    cand_rgb: Image.Image,
    ctx: VariationAdherenceContext,
    scorers: Optional[AdherenceScorerBundle] = None,
) -> AdherenceResult:
    fn = scorers.pose_metrics if scorers and scorers.pose_metrics else _default_pose_metrics
    m = fn(base_rgb.convert("RGB"), cand_rgb.convert("RGB"))
    peak, width, gmin = _pose_intensity_targets(ctx.intensity)
    yaw_s = _triangular_score(m.get("yaw_delta", 0.0), peak, width)
    pitch_s = _triangular_score(m.get("pitch_delta", 0.0), peak * 0.85, width * 0.9)
    gchg = float(m.get("global_change", 0.0))
    g_s = _band_score(gchg, gmin, 0.45)
    score = float(np.clip(0.45 * yaw_s + 0.35 * pitch_s + 0.2 * g_s, 0.0, 1.0))
    ev = {
        "family": "pose_edit",
        "intensity": ctx.intensity,
        "validator": "pose_edit_heuristic_v1",
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "subscores": {"yaw_match": yaw_s, "pitch_match": pitch_s, "motion_sanity": g_s},
        "targets": {"yaw_peak": peak, "yaw_half_width": width, "min_global_change": gmin},
    }
    return AdherenceResult(score, ev, score >= _pass_threshold())


def _default_expression_metrics(base: Image.Image, cand: Image.Image) -> Dict[str, float]:
    b = _resize_gray(base, 64)
    c = _resize_gray(cand, 64)
    h = b.shape[0]
    mouth_b, mouth_c = b[int(0.62 * h) :, :], c[int(0.62 * h) :, :]
    eye_b, eye_c = b[: int(0.38 * h), :], c[: int(0.38 * h), :]
    return {
        "mouth_std_delta": abs(float(np.std(mouth_c)) - float(np.std(mouth_b))),
        "eye_std_delta": abs(float(np.std(eye_c)) - float(np.std(eye_b))),
        "mean_abs_delta": float(np.mean(np.abs(c - b))),
    }


def _expression_targets(intensity: str) -> Tuple[float, float, float]:
    i = (intensity or "medium").strip().lower()
    if i == "light":
        return 0.02, 0.018, 0.015
    if i == "far":
        return 0.09, 0.06, 0.045
    return 0.045, 0.035, 0.028


def validate_expression_edit(
    base_rgb: Image.Image,
    cand_rgb: Image.Image,
    ctx: VariationAdherenceContext,
    scorers: Optional[AdherenceScorerBundle] = None,
) -> AdherenceResult:
    fn = scorers.expression_metrics if scorers and scorers.expression_metrics else _default_expression_metrics
    m = fn(base_rgb.convert("RGB"), cand_rgb.convert("RGB"))
    mid_m, w_m, min_mean = _expression_targets(ctx.intensity)
    ms = _triangular_score(m.get("mouth_std_delta", 0.0), mid_m, w_m)
    es = _triangular_score(m.get("eye_std_delta", 0.0), mid_m * 0.7, w_m * 0.8)
    mean_s = _band_score(m.get("mean_abs_delta", 0.0), min_mean, 0.35)
    score = float(np.clip(0.5 * ms + 0.25 * es + 0.25 * mean_s, 0.0, 1.0))
    ev = {
        "family": "expression_edit",
        "intensity": ctx.intensity,
        "validator": "expression_edit_heuristic_v1",
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "subscores": {"mouth_bucket_match": ms, "eye_bucket_match": es, "global_expression_motion": mean_s},
        "targets": {"mouth_peak_delta": mid_m, "mouth_half_width": w_m},
    }
    return AdherenceResult(score, ev, score >= _pass_threshold())


def _default_lighting_metrics(base: Image.Image, cand: Image.Image) -> Dict[str, float]:
    b = _resize_gray(base, 96)
    c = _resize_gray(cand, 96)
    mean_shift = abs(float(np.mean(c) - np.mean(b)))
    cr = (float(np.std(c)) + 1e-6) / (float(np.std(b)) + 1e-6)
    struct = _high_pass_correlation(b, c)
    lr_b = float(np.mean(b[:, :48]) - np.mean(b[:, 48:]))
    lr_c = float(np.mean(c[:, :48]) - np.mean(c[:, 48:]))
    directional_shift = abs(lr_c - lr_b) / (float(np.mean(b)) + 1e-6)
    return {
        "mean_luma_shift": mean_shift,
        "contrast_ratio": cr,
        "structure_preservation": struct,
        "directional_light_shift": directional_shift,
    }


def _lighting_targets(intensity: str) -> Tuple[float, float, float]:
    i = (intensity or "medium").strip().lower()
    if i == "light":
        return 0.035, 0.25, 0.04
    if i == "far":
        return 0.12, 0.65, 0.12
    return 0.07, 0.45, 0.07


def validate_lighting_edit(
    base_rgb: Image.Image,
    cand_rgb: Image.Image,
    ctx: VariationAdherenceContext,
    scorers: Optional[AdherenceScorerBundle] = None,
) -> AdherenceResult:
    fn = scorers.lighting_metrics if scorers and scorers.lighting_metrics else _default_lighting_metrics
    m = fn(base_rgb.convert("RGB"), cand_rgb.convert("RGB"))
    ms_target, cr_dev_max, dir_target = _lighting_targets(ctx.intensity)
    lum_s = _triangular_score(m.get("mean_luma_shift", 0.0), ms_target, ms_target * 0.85)
    cr = m.get("contrast_ratio", 1.0)
    cr_dev = abs(float(cr) - 1.0)
    cr_s = _triangular_score(cr_dev, cr_dev_max * 0.4, cr_dev_max * 0.55)
    struct_s = float(m.get("structure_preservation", 0.5))
    dir_s = _triangular_score(m.get("directional_light_shift", 0.0), dir_target, dir_target * 0.9)
    score = float(np.clip(0.25 * lum_s + 0.2 * cr_s + 0.35 * struct_s + 0.2 * dir_s, 0.0, 1.0))
    ev = {
        "family": "lighting_edit",
        "intensity": ctx.intensity,
        "validator": "lighting_edit_heuristic_v1",
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "subscores": {
            "luma_shift_match": lum_s,
            "contrast_change_match": cr_s,
            "structure_preserved": struct_s,
            "directional_match": dir_s,
        },
        "notes": ["structure_preservation rewards facial high-frequency alignment with seed"],
    }
    return AdherenceResult(score, ev, score >= _pass_threshold())


def _default_background_metrics(base: Image.Image, cand: Image.Image) -> Dict[str, float]:
    b = _resize_gray(base, 64)
    c = _resize_gray(cand, 64)
    h, w = b.shape
    c0, c1 = int(0.25 * h), int(0.75 * h)
    r0, r1 = int(0.25 * w), int(0.75 * w)
    center_mse = float(np.mean((c[c0:c1, r0:r1] - b[c0:c1, r0:r1]) ** 2))
    # border change: candidate border vs base border
    bc = np.concatenate([c[0:4, :].flatten(), c[-4:, :].flatten(), c[:, 0:4].flatten(), c[:, -4:].flatten()])
    bb = np.concatenate([b[0:4, :].flatten(), b[-4:, :].flatten(), b[:, 0:4].flatten(), b[:, -4:].flatten()])
    border_mse = float(np.mean((bc - bb) ** 2))
    return {"center_mse": center_mse, "border_mse": border_mse, "border_to_center_ratio": border_mse / (center_mse + 1e-6)}


def _background_targets(intensity: str) -> Tuple[float, float]:
    i = (intensity or "medium").strip().lower()
    if i == "light":
        return 0.012, 1.8
    if i == "far":
        return 0.06, 4.5
    return 0.028, 3.0


def validate_background_edit(
    base_rgb: Image.Image,
    cand_rgb: Image.Image,
    ctx: VariationAdherenceContext,
    scorers: Optional[AdherenceScorerBundle] = None,
) -> AdherenceResult:
    fn = scorers.background_metrics if scorers and scorers.background_metrics else _default_background_metrics
    m = fn(base_rgb.convert("RGB"), cand_rgb.convert("RGB"))
    max_center, min_ratio = _background_targets(ctx.intensity)
    ce = m.get("center_mse", 0.0)
    ratio = m.get("border_to_center_ratio", 0.0)
    subject_s = 1.0 - min(1.0, float(ce) / max(max_center, 1e-6))
    bg_s = _band_score(float(ratio), min_ratio * 0.65, min_ratio * 2.0)
    score = float(np.clip(0.55 * subject_s + 0.45 * bg_s, 0.0, 1.0))
    ev = {
        "family": "background_edit",
        "intensity": ctx.intensity,
        "validator": "background_edit_heuristic_v1",
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "subscores": {"subject_preserved": subject_s, "background_changed": bg_s},
        "targets": {"max_center_mse": max_center, "min_border_center_ratio": min_ratio},
        "notes": ["Low center MSE implies subject stable; high border delta implies background edit"],
    }
    return AdherenceResult(score, ev, score >= _pass_threshold())


# --- screen replay ---

_SCREEN_CUE_TAGS: Dict[str, Tuple[str, ...]] = {
    "moire_pixel_grid": ("moire", "moiré", "pixel", "grid", "subpixel", "interference"),
    "screen_glare_hotspots": ("glare", "specular", "hotspot", "reflection"),
    "perspective_keystone_distortion": ("perspective", "keystone", "distortion", "off-angle"),
    "gamma_contrast_shift": ("gamma", "contrast", "colour", "color", "temperature"),
    "edge_crop_cues": ("edge", "crop", "bezel", "border", "display capture"),
}

ALL_SCREEN_CUE_KEYS: Tuple[str, ...] = tuple(_SCREEN_CUE_TAGS.keys())


def parse_requested_screen_cues(description: str, detail: str) -> List[str]:
    text = f"{description} {detail}".lower()
    found: List[str] = []
    for key, tags in _SCREEN_CUE_TAGS.items():
        if any(t in text for t in tags):
            found.append(key)
    if not found:
        found.extend(ALL_SCREEN_CUE_KEYS)
    return list(dict.fromkeys(found))


def _fft_moire_score(gray: np.ndarray) -> float:
    """Mid-frequency energy proxy for moiré / pixel grid."""
    g = gray - np.mean(gray)
    fft = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(fft)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    ring = (d > min(h, w) * 0.08) & (d < min(h, w) * 0.35)
    energy = float(np.mean(mag[ring] ** 2))
    return float(max(0.0, min(1.0, 1.0 - math.exp(-energy / 50.0))))


def _glare_score(rgb: np.ndarray) -> float:
    v = np.max(rgb, axis=2)
    return float(np.mean(v > 0.92))


def _border_edge_score(gray: np.ndarray) -> float:
    try:
        import cv2

        g8 = (gray * 255).astype(np.uint8)
        e = cv2.Canny(g8, 60, 120)
        ring = np.zeros_like(e, dtype=bool)
        ring[:3, :] = True
        ring[-3:, :] = True
        ring[:, :3] = True
        ring[:, -3:] = True
        return float(np.mean(e[ring] > 0))
    except Exception:
        return float(np.std(gray[0:4, :]) + np.std(gray[-4:, :]))


def _default_screen_metrics(
    base_rgb: Image.Image, cand_rgb: Image.Image, ctx: VariationAdherenceContext
) -> Dict[str, float]:
    c = _resize_gray(cand_rgb, 128)
    b = _resize_gray(base_rgb, 128)
    rgb = _resize_rgb(cand_rgb, 128)
    return {
        "moire": _fft_moire_score(c),
        "glare": _glare_score(rgb),
        "border_edges": _border_edge_score(c),
        "gamma_shift": abs(float(np.mean(c) - np.mean(b))),
        "keystone_proxy": abs(_lr_asymmetry(c) - _lr_asymmetry(b)),
    }


def _map_cue_to_detector(key: str, m: Dict[str, float]) -> float:
    if key == "moire_pixel_grid":
        return float(m.get("moire", 0.0))
    if key == "screen_glare_hotspots":
        return float(m.get("glare", 0.0))
    if key == "edge_crop_cues":
        return float(m.get("border_edges", 0.0))
    if key == "gamma_contrast_shift":
        return float(max(0.0, min(1.0, m.get("gamma_shift", 0.0) * 8.0)))
    if key == "perspective_keystone_distortion":
        return float(max(0.0, min(1.0, m.get("keystone_proxy", 0.0) * 5.0)))
    return 0.5


def validate_screen_replay(
    base_rgb: Image.Image,
    cand_rgb: Image.Image,
    ctx: VariationAdherenceContext,
    scorers: Optional[AdherenceScorerBundle] = None,
) -> AdherenceResult:
    requested = parse_requested_screen_cues(ctx.description, ctx.detail)
    fn = scorers.screen_metrics if scorers and scorers.screen_metrics else _default_screen_metrics
    m = fn(base_rgb.convert("RGB"), cand_rgb.convert("RGB"), ctx)
    per_cue = {k: round(_map_cue_to_detector(k, m), 4) for k in requested}
    need = max(1, min(len(requested), int(_float_env("PHASE4_ADHERENCE_SCREEN_MIN_CUES", 2))))
    sorted_scores = sorted(per_cue.values(), reverse=True)
    topk = sorted_scores[:need] if sorted_scores else [0.0]
    cue_score = float(np.mean(topk)) if topk else 0.0
    framing = float(max(0.0, min(1.0, m.get("border_edges", 0.0) * 12.0 + m.get("gamma_shift", 0.0) * 4.0)))
    intensity = (ctx.intensity or "standard").strip().lower()
    if intensity == "standard":
        w_cue, w_frame = 0.55, 0.45
    else:
        w_cue, w_frame = 0.6, 0.4
    score = float(np.clip(w_cue * cue_score + w_frame * framing, 0.0, 1.0))
    ev = {
        "family": "screen_replay",
        "intensity": ctx.intensity,
        "validator": "screen_replay_heuristic_v1",
        "requested_cue_keys": requested,
        "cue_detector_scores": per_cue,
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "subscores": {"requested_cue_coverage": cue_score, "device_framing_proxy": framing},
        "notes": [
            "Cue list parsed from description/detail; image detectors are lightweight proxies",
            f"Top-{need} cue scores averaged for coverage",
        ],
    }
    return AdherenceResult(score, ev, score >= _pass_threshold())


def _fallback_generic(
    base_rgb: Image.Image,
    cand_rgb: Image.Image,
    ctx: VariationAdherenceContext,
    scorers: Optional[AdherenceScorerBundle],
) -> AdherenceResult:
    b = _resize_rgb(base_rgb, 64)
    c = _resize_rgb(cand_rgb, 64)
    mse = float(np.mean((c - b) ** 2))
    target = max(1e-8, _float_env("PHASE4_RERANK_ADHERENCE_TARGET_MSE", 0.015))
    dev = abs(mse - target) / target
    score = float(max(0.0, min(1.0, 1.0 - dev)))
    ev = {
        "family": ctx.variation_type or "unknown",
        "intensity": ctx.intensity,
        "validator": "generic_mse_target_v1",
        "metrics": {"seed_candidate_mse": round(mse, 6), "target_mse": target},
        "notes": ["No specific validator registered; MSE-to-target fallback"],
    }
    return AdherenceResult(score, ev, score >= _pass_threshold())


ValidatorFn = Callable[
    [Image.Image, Image.Image, VariationAdherenceContext, Optional[AdherenceScorerBundle]],
    AdherenceResult,
]

ADHERENCE_VALIDATORS: Dict[str, ValidatorFn] = {
    "pose_edit": validate_pose_edit,
    "expression_edit": validate_expression_edit,
    "lighting_edit": validate_lighting_edit,
    "background_edit": validate_background_edit,
    "screen_replay": validate_screen_replay,
}


def score_variation_adherence(
    base_rgb: Image.Image,
    cand_rgb: Image.Image,
    context: VariationAdherenceContext,
    scorers: Optional[AdherenceScorerBundle] = None,
) -> AdherenceResult:
    """Dispatch task-specific adherence validator by ``context.variation_type``."""
    key = (context.variation_type or "").strip().lower()
    fn = ADHERENCE_VALIDATORS.get(key, _fallback_generic)
    return fn(base_rgb.convert("RGB"), cand_rgb.convert("RGB"), context, scorers)


def register_adherence_validator(name: str, fn: ValidatorFn) -> None:
    """Register or replace a validator (e.g. A/B models) without editing dispatch."""
    ADHERENCE_VALIDATORS[name.strip().lower()] = fn


# Illustrative evidence shapes (for logging contracts / dashboards — not live scores).
EXAMPLE_EVIDENCE_PAYLOADS: Dict[str, Dict[str, Any]] = {
    "pose_edit": {
        "family": "pose_edit",
        "intensity": "medium",
        "validator": "pose_edit_heuristic_v1",
        "metrics": {"yaw_delta": 0.102, "pitch_delta": 0.041, "global_change": 0.028},
        "subscores": {"yaw_match": 0.85, "pitch_match": 0.78, "motion_sanity": 0.9},
        "targets": {"yaw_peak": 0.1, "yaw_half_width": 0.08, "min_global_change": 0.022},
    },
    "expression_edit": {
        "family": "expression_edit",
        "intensity": "light",
        "validator": "expression_edit_heuristic_v1",
        "metrics": {"mouth_std_delta": 0.024, "eye_std_delta": 0.011, "mean_abs_delta": 0.019},
        "subscores": {"mouth_bucket_match": 0.82, "eye_bucket_match": 0.74, "global_expression_motion": 0.88},
        "targets": {"mouth_peak_delta": 0.02, "mouth_half_width": 0.018},
    },
    "lighting_edit": {
        "family": "lighting_edit",
        "intensity": "far",
        "validator": "lighting_edit_heuristic_v1",
        "metrics": {
            "mean_luma_shift": 0.11,
            "contrast_ratio": 1.42,
            "structure_preservation": 0.76,
            "directional_light_shift": 0.09,
        },
        "subscores": {
            "luma_shift_match": 0.91,
            "contrast_change_match": 0.72,
            "structure_preserved": 0.76,
            "directional_match": 0.68,
        },
        "notes": ["structure_preservation rewards facial high-frequency alignment with seed"],
    },
    "background_edit": {
        "family": "background_edit",
        "intensity": "medium",
        "validator": "background_edit_heuristic_v1",
        "metrics": {"center_mse": 0.006, "border_mse": 0.045, "border_to_center_ratio": 7.5},
        "subscores": {"subject_preserved": 0.79, "background_changed": 0.88},
        "targets": {"max_center_mse": 0.028, "min_border_center_ratio": 3.0},
        "notes": ["Low center MSE implies subject stable; high border delta implies background edit"],
    },
    "screen_replay": {
        "family": "screen_replay",
        "intensity": "standard",
        "validator": "screen_replay_heuristic_v1",
        "requested_cue_keys": ["moire_pixel_grid", "screen_glare_hotspots"],
        "cue_detector_scores": {"moire_pixel_grid": 0.62, "screen_glare_hotspots": 0.41},
        "metrics": {"moire": 0.55, "glare": 0.12, "border_edges": 0.08, "gamma_shift": 0.06, "keystone_proxy": 0.04},
        "subscores": {"requested_cue_coverage": 0.515, "device_framing_proxy": 0.72},
        "notes": [
            "Cue list parsed from description/detail; image detectors are lightweight proxies",
            "Top-2 cue scores averaged for coverage",
        ],
    },
}
