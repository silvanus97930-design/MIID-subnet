# MIID/miner/screen_replay.py
#
# Dedicated SN54 screen-replay synthesis: identity-preserving face → device composite → cue simulation.
#
# Routing: set ``SN54_SCREEN_REPLAY_PIPELINE=true`` so ``image_generator.generate_variations`` uses this
# path for a single ``screen_replay`` request (see ``build_raw_results_screen_replay``).
#
# Environment knobs:
#
#   SN54_SCREEN_REPLAY_PIPELINE — enable dedicated path (default false).
#   SN54_SCREEN_REPLAY_DEBUG_DIR — optional folder for before/after PNGs.
#   SN54_SCREEN_REPLAY_DEBUG_SAVE — must be true to write PNGs when DEBUG_DIR set.
#
# Per-cue strengths (only cues listed on the request are applied):
#
#   SN54_SR_CUE_MOIRE_STRENGTH, SN54_SR_CUE_GLARE_STRENGTH,
#   SN54_SR_CUE_KEYSTONE_MAX_SHIFT_FRAC, SN54_SR_CUE_GAMMA_SHIFT,
#   SN54_SR_CUE_CONTRAST_SCALE, SN54_SR_CUE_EDGE_VIGNETTE
#
# Camera / dominance verification:
#
#   SN54_SR_CAMERA_BLUR_SIGMA, SN54_SR_CAMERA_NOISE,
#   SN54_SR_DOMINANCE_CENTER_ENERGY_FRAC_MIN

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Config (env knobs per cue / stage)
# ---------------------------------------------------------------------------

KNOWN_CUE_KEYS = frozenset(
    {
        "moire_pixel_grid",
        "screen_glare_hotspots",
        "perspective_keystone_distortion",
        "gamma_contrast_shift",
        "edge_crop_cues",
    }
)


def _float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or str(default))
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


@dataclass
class ScreenReplayCueStrengths:
    moire: float = field(default_factory=lambda: _float_env("SN54_SR_CUE_MOIRE_STRENGTH", 0.32))
    glare: float = field(default_factory=lambda: _float_env("SN54_SR_CUE_GLARE_STRENGTH", 0.38))
    keystone_max_shift_frac: float = field(
        default_factory=lambda: _float_env("SN54_SR_CUE_KEYSTONE_MAX_SHIFT_FRAC", 0.022)
    )
    gamma_shift: float = field(default_factory=lambda: _float_env("SN54_SR_CUE_GAMMA_SHIFT", 0.11))
    contrast_scale: float = field(default_factory=lambda: _float_env("SN54_SR_CUE_CONTRAST_SCALE", 1.12))
    edge_vignette: float = field(default_factory=lambda: _float_env("SN54_SR_CUE_EDGE_VIGNETTE", 0.28))


@dataclass
class ScreenReplaySpec:
    device: str
    cue_keys: Tuple[str, ...]
    description: str
    detail: str


# Output canvas (W, H) per device — face stays large in inner screen rect
_DEVICE_CANVAS: Dict[str, Tuple[int, int]] = {
    "phone": (720, 1280),
    "tablet": (1024, 1280),
    "laptop": (1280, 800),
    "monitor": (1280, 720),
    "tv": (1600, 900),
    "unknown": (1024, 1280),
}

# Inner screen rectangle as fractions (x0, y0, x1, y1) of canvas
_DEVICE_INNER_FRAC: Dict[str, Tuple[float, float, float, float]] = {
    "phone": (0.085, 0.11, 0.915, 0.86),
    "tablet": (0.075, 0.10, 0.925, 0.84),
    "laptop": (0.070, 0.065, 0.930, 0.82),
    "monitor": (0.055, 0.065, 0.945, 0.88),
    "tv": (0.045, 0.055, 0.955, 0.90),
    "unknown": (0.075, 0.10, 0.925, 0.84),
}

# Deterministic cue application order (only requested cues run)
_CUE_PIPELINE_ORDER: Tuple[str, ...] = (
    "gamma_contrast_shift",
    "moire_pixel_grid",
    "perspective_keystone_distortion",
    "screen_glare_hotspots",
    "edge_crop_cues",
)


def _normalize_device(raw: str) -> str:
    s = (raw or "unknown").strip().lower()
    if s in _DEVICE_CANVAS:
        return s
    return "unknown"


def _normalize_cues(keys: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for k in keys:
        kk = str(k).strip()
        if kk in KNOWN_CUE_KEYS and kk not in seen:
            out.append(kk)
            seen.add(kk)
    return tuple(out)


def parse_screen_replay_request(req: Any) -> Optional[ScreenReplaySpec]:
    """Build spec from protocol object or dict (compiled miner request)."""
    t = _get_str(req, "type")
    if t != "screen_replay":
        return None
    dev = _get_str(req, "screen_replay_device") or _get_str(req, "device_type")
    cues = getattr(req, "visual_cue_keys", None)
    if cues is None and isinstance(req, dict):
        cues = req.get("visual_cue_keys")
    if cues is None:
        cues = ()
    cue_t = _normalize_cues(list(cues))
    if not cue_t:
        return None
    return ScreenReplaySpec(
        device=_normalize_device(dev),
        cue_keys=cue_t,
        description=_get_str(req, "description"),
        detail=_get_str(req, "detail"),
    )


def _get_str(obj: Any, key: str) -> str:
    v = getattr(obj, key, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(key)
    if v is None:
        return ""
    return str(v).strip()


def _rgb01(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_pil(arr: np.ndarray) -> Image.Image:
    x = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")


def stage_a_prepare_face(
    base_rgb: Image.Image,
    inner_size: Tuple[int, int],
    aligned_face: Optional[Image.Image] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Scale/crop identity content to fill inner screen without generative drift."""
    src = aligned_face.convert("RGB") if aligned_face is not None else base_rgb.convert("RGB")
    iw, ih = inner_size
    sw, sh = src.size
    scale = max(iw / sw, ih / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - iw) // 2)
    top = max(0, (nh - ih) // 2)
    crop = resized.crop((left, top, left + iw, top + ih))
    meta = {
        "stage": "A",
        "source": "aligned_face" if aligned_face is not None else "base_image",
        "inner_size": [iw, ih],
    }
    return crop, meta


def _apply_gamma_contrast(arr: np.ndarray, strengths: ScreenReplayCueStrengths, rng: np.random.Generator) -> np.ndarray:
    g = float(strengths.gamma_shift) * float(rng.uniform(0.75, 1.25))
    gamma = float(np.clip(1.0 + g * rng.choice([-1.0, 1.0]), 0.65, 1.35))
    x = np.clip(arr, 0.0, 1.0) ** gamma
    c = float(np.clip(strengths.contrast_scale * float(rng.uniform(0.92, 1.08)), 0.85, 1.35))
    m = float(np.mean(x))
    x = np.clip((x - m) * c + m, 0.0, 1.0)
    return x


def _apply_moire(arr: np.ndarray, strengths: ScreenReplayCueStrengths, rng: np.random.Generator) -> np.ndarray:
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f1 = rng.uniform(0.35, 0.55) * strengths.moire
    f2 = rng.uniform(0.40, 0.60) * strengths.moire
    ph = float(rng.uniform(0, 2 * np.pi))
    pat = 0.5 * (
        np.sin(xx * f1 + ph) * np.sin(yy * f2 + ph * 0.7)
        + np.sin((xx + yy) * (f1 + f2) * 0.5 + ph * 1.3)
    )
    amp = 0.045 * strengths.moire
    return np.clip(arr + amp * pat[..., None], 0.0, 1.0)


def _apply_glare(
    arr: np.ndarray,
    inner_xyxy: Tuple[int, int, int, int],
    strengths: ScreenReplayCueStrengths,
    rng: np.random.Generator,
) -> np.ndarray:
    x0, y0, x1, y1 = inner_xyxy
    out = arr.copy()
    h, w, _ = out.shape
    gx, gy = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(1, 4))):
        lo_x, hi_x = x0 + 8, x1 - 8
        lo_y, hi_y = y0 + 8, y1 - 8
        if hi_x <= lo_x:
            lo_x, hi_x = x0, max(x0 + 1, x1 - 1)
        if hi_y <= lo_y:
            lo_y, hi_y = y0, max(y0 + 1, y1 - 1)
        cx = int(rng.integers(lo_x, hi_x + 1))
        cy = int(rng.integers(lo_y, hi_y + 1))
        sigma = float(rng.uniform(max(12.0, (x1 - x0) * 0.06), max(20.0, (x1 - x0) * 0.14)))
        d = np.sqrt((gx - cy) ** 2 + (gy - cx) ** 2)
        blob = np.exp(-(d**2) / (2 * sigma * sigma + 1e-6))[..., None]
        spot = float(strengths.glare) * float(rng.uniform(0.45, 1.0))
        out = np.clip(out + spot * blob * 0.5, 0.0, 1.0)
    return out


def _apply_edge_vignette(arr: np.ndarray, strengths: ScreenReplayCueStrengths, rng: np.random.Generator) -> np.ndarray:
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.5 + 1e-6)
    ny = (yy - h * 0.5) / (h * 0.5 + 1e-6)
    r = np.sqrt(nx**2 + ny**2)
    v = float(strengths.edge_vignette) * float(rng.uniform(0.85, 1.15))
    mask = np.clip(1.0 - v * (r**1.8), 0.35, 1.0)[..., None]
    return np.clip(arr * mask, 0.0, 1.0)


def _apply_keystone_pil(img: Image.Image, strengths: ScreenReplayCueStrengths, rng: np.random.Generator) -> Image.Image:
    try:
        import cv2
    except Exception:
        return img
    w, h = img.size
    m = float(strengths.keystone_max_shift_frac) * float(rng.uniform(0.6, 1.0))
    dx, dy = w * m, h * m
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jx = rng.uniform(-1, 1, size=4) * dx
    jy = rng.uniform(-1, 1, size=4) * dy
    dst = src + np.stack([jx, jy], axis=1)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(np.asarray(img.convert("RGB")), M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped, "RGB")


def _draw_device_bezel(
    canvas: Image.Image,
    inner_xyxy: Tuple[int, int, int, int],
    device: str,
) -> None:
    """Draw bezel around inner screen (mutates canvas)."""
    drw = ImageDraw.Draw(canvas)
    W, H = canvas.size
    x0, y0, x1, y1 = inner_xyxy
    bezel_col = (32, 34, 40)
    # outer frame
    drw.rounded_rectangle([2, 2, W - 3, H - 3], radius=18, fill=bezel_col, outline=(12, 12, 16), width=3)
    # inner black mask for screen hole — screen content pasted separately
    drw.rounded_rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], radius=8, fill=(8, 8, 10), outline=(55, 58, 68), width=2)
    if device == "laptop":
        kb_h = int(H * 0.12)
        drw.rectangle([0, H - kb_h, W, H], fill=(26, 28, 32))
        for i in range(10):
            kx = 40 + i * ((W - 80) // 9)
            drw.rounded_rectangle([kx, H - kb_h + 8, kx + 48, H - 14], radius=3, fill=(40, 42, 48))
    if device == "tv":
        drw.rectangle([W // 2 - 25, H - 6, W // 2 + 25, H - 1], fill=(20, 20, 22))


def _inner_pixels(device: str) -> Tuple[int, int, Tuple[int, int, int, int]]:
    W, H = _DEVICE_CANVAS.get(device, _DEVICE_CANVAS["unknown"])
    fx0, fy0, fx1, fy1 = _DEVICE_INNER_FRAC.get(device, _DEVICE_INNER_FRAC["unknown"])
    x0, y0 = int(W * fx0), int(H * fy0)
    x1, y1 = int(W * fx1), int(H * fy1)
    return W, H, (x0, y0, x1, y1)


def synthesize_screen_replay(
    base_rgb: Image.Image,
    spec: ScreenReplaySpec,
    *,
    seed: int,
    aligned_face: Optional[Image.Image] = None,
    strengths: Optional[ScreenReplayCueStrengths] = None,
    save_debug_prefix: Optional[str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Run stage A + B; only ``spec.cue_keys`` artifacts (subset of pipeline)."""
    st = strengths or ScreenReplayCueStrengths()
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    W, H, inner = _inner_pixels(spec.device)
    x0, y0, x1, y1 = inner
    iw, ih = x1 - x0, y1 - y0

    t0 = time.perf_counter()
    face_img, meta_a = stage_a_prepare_face(base_rgb, (iw, ih), aligned_face=aligned_face)
    if save_debug_prefix:
        face_img.save(f"{save_debug_prefix}_stage_a.png", format="PNG")

    arr = _rgb01(face_img)
    for cue in _CUE_PIPELINE_ORDER:
        if cue not in spec.cue_keys:
            continue
        if cue == "gamma_contrast_shift":
            arr = _apply_gamma_contrast(arr, st, rng)
        elif cue == "moire_pixel_grid":
            arr = _apply_moire(arr, st, rng)

    screen_pil = _to_pil(arr)
    if save_debug_prefix:
        screen_pil.save(f"{save_debug_prefix}_screen_content_pre_composite.png", format="PNG")

    canvas = Image.new("RGB", (W, H), (12, 12, 14))
    _draw_device_bezel(canvas, inner, spec.device)
    canvas.paste(screen_pil, (x0, y0))

    out = canvas
    if "perspective_keystone_distortion" in spec.cue_keys:
        out = _apply_keystone_pil(out, st, rng)

    arr_full = _rgb01(out)
    if "screen_glare_hotspots" in spec.cue_keys:
        arr_full = _apply_glare(arr_full, inner, st, rng)
    if "edge_crop_cues" in spec.cue_keys:
        arr_full = _apply_edge_vignette(arr_full, st, rng)
    out = _to_pil(arr_full)

    sig = _float_env("SN54_SR_CAMERA_BLUR_SIGMA", 0.55)
    out = out.filter(ImageFilter.GaussianBlur(radius=max(0.0, sig)))
    arr_c = _rgb01(out)
    noise = float(_float_env("SN54_SR_CAMERA_NOISE", 0.012))
    arr_c = np.clip(arr_c + rng.normal(0.0, noise, arr_c.shape), 0.0, 1.0)
    out = _to_pil(arr_c)

    if save_debug_prefix:
        out.save(f"{save_debug_prefix}_final.png", format="PNG")

    ms = (time.perf_counter() - t0) * 1000.0
    meta = {
        "pipeline": "screen_replay_dedicated",
        "device": spec.device,
        "cue_keys": list(spec.cue_keys),
        "applied_cue_order": [c for c in _CUE_PIPELINE_ORDER if c in spec.cue_keys],
        "stage_a": meta_a,
        "timing_ms": round(ms, 3),
        "canvas": [W, H],
        "inner_rect": list(inner),
    }
    return out, meta


def generate_screen_replay_candidates(
    base_rgb: Image.Image,
    spec: ScreenReplaySpec,
    n_candidates: int,
    *,
    aligned_face: Optional[Image.Image] = None,
    debug_dir: Optional[str] = None,
    debug_tag: str = "sr",
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    """Multiple candidates = different RNG seeds (cue geometry / camera noise)."""
    n = max(1, int(n_candidates))
    outs: List[Image.Image] = []
    metas: List[Dict[str, Any]] = []
    base_hash = hashlib.sha256(np.asarray(base_rgb.convert("RGB")).tobytes()).hexdigest()[:12]
    for i in range(n):
        seed = int(hashlib.sha256(f"{base_hash}:{spec.device}:{spec.cue_keys}:{i}".encode()).hexdigest()[:8], 16)
        prefix = None
        if debug_dir and _bool_env("SN54_SCREEN_REPLAY_DEBUG_SAVE", False):
            os.makedirs(debug_dir, exist_ok=True)
            prefix = os.path.join(debug_dir, f"{debug_tag}_{base_hash}_cand{i}")
        img, meta = synthesize_screen_replay(
            base_rgb,
            spec,
            seed=seed,
            aligned_face=aligned_face,
            save_debug_prefix=prefix,
        )
        meta["candidate_index"] = i
        outs.append(img)
        metas.append(meta)
    return outs, metas


# --- Adherence / scoring hooks (used by adherence.validate_screen_replay) ---


def score_face_dominance(rgb: Image.Image) -> Tuple[float, Dict[str, Any]]:
    """Center energy fraction — proxy for face dominance (no face detector required)."""
    g = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    h, w = g.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = ((xx - cx) ** 2 / ((w * 0.42) ** 2 + 1e-6) + (yy - cy) ** 2 / ((h * 0.48) ** 2 + 1e-6)) <= 1.0
    try:
        import cv2

        lap = cv2.Laplacian((g * 255).astype(np.uint8), cv2.CV_64F)
        e = np.abs(lap)
    except Exception:
        e = np.abs(g - np.mean(g))
    total = float(np.sum(e) + 1e-6)
    center = float(np.sum(e[mask]))
    frac = center / total
    thr = _float_env("SN54_SR_DOMINANCE_CENTER_ENERGY_FRAC_MIN", 0.38)
    score = float(np.clip((frac - thr) / max(1e-6, 0.5 - thr), 0.0, 1.0))
    return score, {"center_detail_energy_fraction": round(frac, 4), "threshold": thr}


def score_device_plausibility(rgb: Image.Image, device: str) -> Tuple[float, Dict[str, Any]]:
    """Aspect ratio of canvas vs expected device profile."""
    W, H = rgb.size
    ar = (W / max(H, 1)) if W >= H else (H / max(W, 1))
    exp_w, exp_h = _DEVICE_CANVAS.get(device, _DEVICE_CANVAS["unknown"])
    exp_ar = max(exp_w, exp_h) / max(min(exp_w, exp_h), 1)
    err = abs(np.log(ar + 1e-6) - np.log(exp_ar + 1e-6))
    score = float(np.clip(1.0 - err / 0.35, 0.0, 1.0))
    return score, {"image_aspect": round(ar, 4), "expected_aspect": round(exp_ar, 4), "log_err": round(float(err), 4)}


def _sr_fft_moire_score(gray: np.ndarray) -> float:
    """Mid-frequency energy proxy (same idea as adherence._fft_moire_score; no cross-import)."""
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


def score_cue_presence(rgb: Image.Image, cue_keys: Sequence[str]) -> Tuple[float, Dict[str, float]]:
    """Lightweight per-cue detectors (aligned with adherence heuristics)."""
    gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    per: Dict[str, float] = {}
    for k in cue_keys:
        if k == "moire_pixel_grid":
            per[k] = float(_sr_fft_moire_score(gray))
        elif k == "screen_glare_hotspots":
            v = np.max(_rgb01(rgb), axis=2)
            per[k] = float(np.clip(np.mean(v > 0.9) * 12.0, 0.0, 1.0))
        elif k == "perspective_keystone_distortion":
            h, w = gray.shape
            lr = abs(float(np.mean(gray[:, : w // 3]) - np.mean(gray[:, 2 * w // 3 :])))
            per[k] = float(np.clip(lr * 5.0, 0.0, 1.0))
        elif k == "gamma_contrast_shift":
            per[k] = float(np.clip(np.std(gray) * 3.5, 0.0, 1.0))
        elif k == "edge_crop_cues":
            per[k] = float(
                np.clip(
                    (np.std(gray[0:5, :]) + np.std(gray[-5:, :]) + np.std(gray[:, 0:5]) + np.std(gray[:, -5:])) * 2.0,
                    0.0,
                    1.0,
                )
            )
        else:
            per[k] = 0.5
    avg = float(np.mean(list(per.values()))) if per else 0.0
    return avg, per


def verify_screen_replay_artifacts(
    rgb: Image.Image,
    spec: ScreenReplaySpec,
) -> Dict[str, Any]:
    """Structured evidence for reranker / logs."""
    dom_s, dom_ev = score_face_dominance(rgb)
    dev_s, dev_ev = score_device_plausibility(rgb, spec.device)
    cue_avg, cue_per = score_cue_presence(rgb, spec.cue_keys)
    return {
        "face_dominance_score": round(dom_s, 4),
        "face_dominance_evidence": dom_ev,
        "device_plausibility_score": round(dev_s, 4),
        "device_plausibility_evidence": dev_ev,
        "requested_cue_detector_avg": round(cue_avg, 4),
        "requested_cue_detector_scores": {k: round(v, 4) for k, v in cue_per.items()},
    }


def build_raw_results_screen_replay(
    base_rgb: Image.Image,
    protocol_request: Any,
    n_candidates: int,
    timings_out: Optional[Dict[str, float]],
    *,
    aligned_face: Optional[Image.Image] = None,
    prompt_fallback: str = "",
) -> List[Dict[str, Any]]:
    """Shape-compatible with FLUX ``generate_variations`` output list."""
    spec = parse_screen_replay_request(protocol_request)
    if spec is None:
        return []

    t_prep0 = time.perf_counter()
    desc = _get_str(protocol_request, "description")
    detail = _get_str(protocol_request, "detail")
    prompt = _get_str(protocol_request, "prompt") or prompt_fallback
    prep_ms = (time.perf_counter() - t_prep0) * 1000.0

    debug_dir = (os.environ.get("SN54_SCREEN_REPLAY_DEBUG_DIR") or "").strip() or None

    t_gen0 = time.perf_counter()
    candidates, metas = generate_screen_replay_candidates(
        base_rgb,
        spec,
        n_candidates,
        aligned_face=aligned_face,
        debug_dir=debug_dir,
        debug_tag=_get_str(protocol_request, "type") or "screen_replay",
    )
    gen_ms = (time.perf_counter() - t_gen0) * 1000.0

    if timings_out is not None:
        timings_out["variation_request_prepare_ms"] = prep_ms
        timings_out["flux_generation_ms"] = gen_ms
        timings_out["screen_replay_generation_ms"] = gen_ms

    return [
        {
            "variation_type": "screen_replay",
            "intensity": "standard",
            "prompt": prompt,
            "description": desc,
            "detail": detail,
            "screen_replay_device": spec.device,
            "visual_cue_keys": list(spec.cue_keys),
            "candidates": candidates,
            "pipeline": "screen_replay_dedicated",
            "screen_replay_candidate_metas": metas,
        }
    ]


def screen_replay_pipeline_enabled() -> bool:
    return _bool_env("SN54_SCREEN_REPLAY_PIPELINE", False)
