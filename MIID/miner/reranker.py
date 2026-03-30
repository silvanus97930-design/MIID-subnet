# MIID/miner/reranker.py
#
# Ensemble reranking for Phase 4 generated candidates (identity + heuristics + duplicates).
#
# Environment variables (weights default to a convex mix; duplicate term is subtracted):
#
#   PHASE4_RERANK_WEIGHT_ARCFACE, PHASE4_RERANK_WEIGHT_ADAFACE,
#   PHASE4_RERANK_WEIGHT_ADHERENCE, PHASE4_RERANK_WEIGHT_REALISM,
#   PHASE4_RERANK_WEIGHT_DUPLICATE
#
# Missing similarity / heuristic fallbacks (used when a score is unavailable):
#
#   PHASE4_RERANK_FALLBACK_ARCFACE, PHASE4_RERANK_FALLBACK_ADAFACE,
#   PHASE4_RERANK_FALLBACK_ADHERENCE, PHASE4_RERANK_FALLBACK_REALISM
#
# Heuristic and duplicate knobs:
#
#   PHASE4_RERANK_ADHERENCE_TARGET_MSE — peak adherence when seed/candidate MSE matches this
#   PHASE4_RERANK_TASK_ADHERENCE_BLEND — weight for task validators vs MSE proxy (see MIID.miner.adherence)
#   PHASE4_RERANK_REALISM_VAR_SCALE — Laplacian variance scaling into [0,1]
#   PHASE4_RERANK_DUP_SEED_MSE_MAX, PHASE4_RERANK_DUP_SIBLING_MSE_MAX
#   PHASE4_RERANK_DUP_PENALTY_SEED, PHASE4_RERANK_DUP_PENALTY_SIBLING, PHASE4_RERANK_DUP_PENALTY_CAP

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from MIID.miner.adherence import (
    AdherenceScorerBundle,
    VariationAdherenceContext,
    score_variation_adherence,
)
from MIID.miner.identity_scoring import IdentityScoreResult


def _float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip().upper())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _variation_float_env(prefix: str, variation_type: str, intensity: str, default: float) -> float:
    type_tok = _env_token(variation_type)
    intensity_tok = _env_token(intensity)
    legacy_type_tok = type_tok[:-5] if type_tok.endswith("_EDIT") else type_tok
    keys: List[str] = []
    if type_tok and intensity_tok:
        keys.append(f"{prefix}_{type_tok}_{intensity_tok}")
    if legacy_type_tok and legacy_type_tok != type_tok and intensity_tok:
        keys.append(f"{prefix}_{legacy_type_tok}_{intensity_tok}")
    if type_tok:
        keys.append(f"{prefix}_{type_tok}")
    if legacy_type_tok and legacy_type_tok != type_tok:
        keys.append(f"{prefix}_{legacy_type_tok}")
    keys.append(prefix)

    for key in keys:
        raw = os.environ.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(str(raw).strip())
        except Exception:
            continue
    return float(default)


@dataclass(frozen=True)
class RerankWeights:
    """Positive weights for identity/heuristics; duplicate term is subtracted."""

    arcface: float
    adaface: float
    adherence: float
    realism: float
    duplicate: float  # multiplier on duplicate_penalty before subtraction


@dataclass(frozen=True)
class RerankConfig:
    weights: RerankWeights
    fallback_arcface: float
    fallback_adaface: float
    fallback_adherence: float
    fallback_realism: float
    adherence_target_mse: float
    task_adherence_blend: float
    realism_var_scale: float
    dup_seed_mse_max: float
    dup_sibling_mse_max: float
    dup_penalty_seed: float
    dup_penalty_per_sibling: float
    dup_penalty_cap: float
    missing_score_borrow_scale: float
    primary_identity_penalty_scale: float
    primary_identity_floor: float


def load_rerank_config_from_env() -> RerankConfig:
    """Load ensemble rerank knobs from environment (see miner docs / summary below)."""
    w = RerankWeights(
        arcface=_float_env("PHASE4_RERANK_WEIGHT_ARCFACE", 0.35),
        adaface=_float_env("PHASE4_RERANK_WEIGHT_ADAFACE", 0.35),
        adherence=_float_env("PHASE4_RERANK_WEIGHT_ADHERENCE", 0.15),
        realism=_float_env("PHASE4_RERANK_WEIGHT_REALISM", 0.15),
        duplicate=_float_env("PHASE4_RERANK_WEIGHT_DUPLICATE", 1.0),
    )
    return RerankConfig(
        weights=w,
        fallback_arcface=_float_env("PHASE4_RERANK_FALLBACK_ARCFACE", 0.5),
        fallback_adaface=_float_env("PHASE4_RERANK_FALLBACK_ADAFACE", 0.5),
        fallback_adherence=_float_env("PHASE4_RERANK_FALLBACK_ADHERENCE", 0.5),
        fallback_realism=_float_env("PHASE4_RERANK_FALLBACK_REALISM", 0.5),
        adherence_target_mse=max(1e-8, _float_env("PHASE4_RERANK_ADHERENCE_TARGET_MSE", 0.015)),
        task_adherence_blend=max(0.0, min(1.0, _float_env("PHASE4_RERANK_TASK_ADHERENCE_BLEND", 0.88))),
        realism_var_scale=max(1e-6, _float_env("PHASE4_RERANK_REALISM_VAR_SCALE", 400.0)),
        dup_seed_mse_max=max(0.0, _float_env("PHASE4_RERANK_DUP_SEED_MSE_MAX", 1.5e-4)),
        dup_sibling_mse_max=max(0.0, _float_env("PHASE4_RERANK_DUP_SIBLING_MSE_MAX", 2.5e-4)),
        dup_penalty_seed=max(0.0, _float_env("PHASE4_RERANK_DUP_PENALTY_SEED", 0.85)),
        dup_penalty_per_sibling=max(0.0, _float_env("PHASE4_RERANK_DUP_PENALTY_SIBLING", 0.35)),
        dup_penalty_cap=max(0.0, _float_env("PHASE4_RERANK_DUP_PENALTY_CAP", 1.0)),
        missing_score_borrow_scale=max(0.0, min(1.0, _float_env("PHASE4_RERANK_MISSING_SCORE_BORROW_SCALE", 0.0))),
        primary_identity_penalty_scale=max(0.0, _float_env("PHASE4_RERANK_PRIMARY_IDENTITY_PENALTY_SCALE", 0.0)),
        primary_identity_floor=max(0.0, min(1.0, _float_env("PHASE4_RERANK_PRIMARY_IDENTITY_FLOOR", 0.0))),
    )


@dataclass
class CandidateScore:
    arcface_score: Optional[float]
    adaface_score: Optional[float]
    adherence_score: float
    realism_score: float
    duplicate_penalty: float
    final_score: float
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "arcface_score": self.arcface_score,
            "adaface_score": self.adaface_score,
            "adherence_score": round(float(self.adherence_score), 6),
            "realism_score": round(float(self.realism_score), 6),
            "duplicate_penalty": round(float(self.duplicate_penalty), 6),
            "final_score": round(float(self.final_score), 6),
        }
        if self.extras:
            out["extras"] = dict(self.extras)
        return out


def _resize_float01(rgb: Image.Image, size: int = 64) -> np.ndarray:
    img = rgb.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def normalized_mse_seed_candidate(base_rgb: Image.Image, cand_rgb: Image.Image, size: int = 64) -> float:
    """Mean squared error in [0, 1] over normalized RGB crops (structural delta proxy)."""
    a = _resize_float01(base_rgb, size)
    b = _resize_float01(cand_rgb, size)
    return float(np.mean((a - b) ** 2))


def adherence_score_from_mse(mse: float, target_mse: float) -> float:
    """Peak when observed MSE is near ``target_mse`` (CLIP-free request/change proxy)."""
    if target_mse <= 0:
        return 0.0
    dev = abs(float(mse) - float(target_mse)) / target_mse
    return float(max(0.0, min(1.0, 1.0 - dev)))


def realism_score_from_image(rgb: Image.Image, *, var_scale: float) -> float:
    """Sharpness / sanity via Laplacian variance (opencv); falls back to numpy gradient."""
    gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    if gray.size == 0:
        return 0.0
    try:
        import cv2

        g8 = (gray * 255.0).clip(0, 255).astype(np.uint8)
        lap = cv2.Laplacian(g8, cv2.CV_64F)
        var = float(lap.var())
    except Exception:
        gz = np.gradient(gray.astype(np.float64))
        var = float(np.var(gz[0]) + np.var(gz[1]))
    scale = max(1e-6, float(var_scale))
    return float(max(0.0, min(1.0, var / (var + scale))))


def duplicate_penalties_for_candidates(
    base_rgb: Image.Image,
    candidates: Sequence[Image.Image],
    config: Optional[RerankConfig] = None,
) -> List[float]:
    """Penalties in [0, dup_penalty_cap] per candidate: near-duplicate of seed + near-duplicate siblings."""
    cfg = config or load_rerank_config_from_env()
    n = len(candidates)
    if n == 0:
        return []

    mse_seed = [normalized_mse_seed_candidate(base_rgb, c.convert("RGB")) for c in candidates]
    mse_pair = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m = normalized_mse_seed_candidate(candidates[i].convert("RGB"), candidates[j].convert("RGB"))
            mse_pair[i][j] = m
            mse_pair[j][i] = m

    penalties: List[float] = []
    for i in range(n):
        p = 0.0
        if mse_seed[i] <= cfg.dup_seed_mse_max:
            p += cfg.dup_penalty_seed
        sib = 0
        for j in range(n):
            if i == j:
                continue
            if mse_pair[i][j] <= cfg.dup_sibling_mse_max:
                sib += 1
        p += float(sib) * cfg.dup_penalty_per_sibling
        p = min(p, cfg.dup_penalty_cap)
        penalties.append(p)
    return penalties


def _identity_raw_scores(result: IdentityScoreResult) -> Tuple[Optional[float], Optional[float]]:
    ada = result.adaface.similarity if result.adaface.available else None
    arc = result.insightface_arcface.similarity if result.insightface_arcface.available else None
    return arc, ada


def ensemble_final_score(
    arcface_score: Optional[float],
    adaface_score: Optional[float],
    adherence_score: float,
    realism_score: float,
    duplicate_penalty: float,
    config: RerankConfig,
) -> Tuple[float, Dict[str, float]]:
    """Return (final_score, effective_inputs_used)."""
    w = config.weights
    if arcface_score is None:
        arc_e = float(config.fallback_arcface)
        if adaface_score is not None and config.missing_score_borrow_scale > 0.0:
            arc_e = max(arc_e, min(1.0, float(adaface_score) * config.missing_score_borrow_scale))
    else:
        arc_e = float(arcface_score)
    if adaface_score is None:
        ada_e = float(config.fallback_adaface)
        if arcface_score is not None and config.missing_score_borrow_scale > 0.0:
            ada_e = max(ada_e, min(1.0, float(arcface_score) * config.missing_score_borrow_scale))
    else:
        ada_e = float(adaface_score)
    adh_e = float(adherence_score if adherence_score is not None else config.fallback_adherence)
    real_e = float(realism_score if realism_score is not None else config.fallback_realism)

    pos = w.arcface * arc_e + w.adaface * ada_e + w.adherence * adh_e + w.realism * real_e
    final = pos - w.duplicate * float(duplicate_penalty)
    extras = {
        "arcface_effective": arc_e,
        "adaface_effective": ada_e,
        "adherence_effective": adh_e,
        "realism_effective": real_e,
        "weighted_pos": pos,
    }
    return float(final), extras


def build_candidate_scores(
    identity_results: Sequence[IdentityScoreResult],
    base_rgb: Image.Image,
    candidates: Sequence[Image.Image],
    config: Optional[RerankConfig] = None,
    variation_context: Optional[VariationAdherenceContext] = None,
    adherence_scorers: Optional[AdherenceScorerBundle] = None,
) -> List[CandidateScore]:
    """One CandidateScore per candidate index (same order as ``candidates``)."""
    cfg = config or load_rerank_config_from_env()
    base_rgb = base_rgb.convert("RGB")
    dups = duplicate_penalties_for_candidates(base_rgb, candidates, cfg)
    out: List[CandidateScore] = []

    for i, (id_res, cand) in enumerate(zip(identity_results, candidates)):
        arc, ada = _identity_raw_scores(id_res)
        mse = normalized_mse_seed_candidate(base_rgb, cand.convert("RGB"))
        mse_adh = adherence_score_from_mse(mse, cfg.adherence_target_mse)
        if variation_context is not None:
            ar = score_variation_adherence(
                base_rgb,
                cand.convert("RGB"),
                variation_context,
                adherence_scorers,
            )
            task_adh = float(ar.adherence_score)
            blend = float(cfg.task_adherence_blend)
            adh = float(blend * task_adh + (1.0 - blend) * mse_adh)
            adh_evidence = {
                "task": ar.evidence,
                "task_adherence_score": round(task_adh, 6),
                "mse_proxy_score": round(mse_adh, 6),
                "blend": blend,
                "pass_recommendation_task": ar.pass_recommendation,
            }
        else:
            adh = mse_adh
            adh_evidence = {
                "task": None,
                "task_adherence_score": None,
                "mse_proxy_score": round(mse_adh, 6),
                "blend": 0.0,
                "notes": ["no VariationAdherenceContext; MSE-only adherence"],
            }
        real_scale = cfg.realism_var_scale
        var_type = variation_context.variation_type if variation_context is not None else ""
        var_intensity = variation_context.intensity if variation_context is not None else ""
        if var_type:
            real_scale = _variation_float_env(
                "PHASE4_RERANK_REALISM_VAR_SCALE",
                var_type,
                var_intensity,
                cfg.realism_var_scale,
            )
        real = realism_score_from_image(cand.convert("RGB"), var_scale=real_scale)
        dup = dups[i] if i < len(dups) else 0.0
        final, ex = ensemble_final_score(arc, ada, adh, real, dup, cfg)
        primary = id_res.primary_similarity
        if primary is None:
            primary = ada if ada is not None else arc
        identity_floor = cfg.primary_identity_floor
        identity_penalty_scale = cfg.primary_identity_penalty_scale
        if var_type:
            identity_floor = _variation_float_env(
                "PHASE4_RERANK_PRIMARY_IDENTITY_FLOOR",
                var_type,
                var_intensity,
                identity_floor,
            )
            identity_penalty_scale = _variation_float_env(
                "PHASE4_RERANK_PRIMARY_IDENTITY_PENALTY_SCALE",
                var_type,
                var_intensity,
                identity_penalty_scale,
            )
        identity_penalty = 0.0
        if identity_penalty_scale > 0.0 and identity_floor > 0.0:
            if primary is None:
                identity_penalty = float(identity_penalty_scale)
            elif float(primary) < float(identity_floor):
                identity_penalty = float(
                    min(
                        identity_penalty_scale,
                        identity_penalty_scale * ((float(identity_floor) - float(primary)) / max(float(identity_floor), 1e-6)),
                    )
                )
        final -= identity_penalty
        out.append(
            CandidateScore(
                arcface_score=arc,
                adaface_score=ada,
                adherence_score=adh,
                realism_score=real,
                duplicate_penalty=dup,
                final_score=final,
                extras={
                    **ex,
                    "seed_candidate_mse": mse,
                    "realism_var_scale": real_scale,
                    "primary_identity_score": primary,
                    "identity_floor": identity_floor,
                    "identity_penalty_scale": identity_penalty_scale,
                    "identity_penalty": identity_penalty,
                    "adherence_evidence": adh_evidence,
                },
            )
        )
    return out


def select_best_candidate_index(scores: Sequence[CandidateScore]) -> int:
    """Highest ``final_score``; ties broken by lowest index (deterministic)."""
    if not scores:
        return 0
    best_i = 0
    best_f = scores[0].final_score
    for i in range(1, len(scores)):
        f = scores[i].final_score
        if f > best_f:
            best_f = f
            best_i = i
    return best_i


def leaderboard_entries(
    scores: Sequence[CandidateScore],
    *,
    variation_label: str = "",
) -> List[Dict[str, Any]]:
    """Sorted by final_score descending, then candidate_index ascending (stable tie-break)."""
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda t: (-t[1].final_score, t[0]))
    rows: List[Dict[str, Any]] = []
    rank = 0
    for cand_idx, sc in indexed:
        rank += 1
        row = {
            "rank": rank,
            "candidate_index": cand_idx,
            "variation": variation_label,
            **sc.to_dict(),
        }
        rows.append(row)
    return rows
