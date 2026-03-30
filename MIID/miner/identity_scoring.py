# MIID/miner/identity_scoring.py
#
# Pluggable identity scoring: AdaFace + optional InsightFace/ArcFace, with per-request base cache.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import torch
from PIL import Image


@dataclass
class ScorerComponentResult:
    """One backend's similarity output (JSON-serializable via to_dict)."""

    name: str
    available: bool
    similarity: Optional[float]
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "similarity": self.similarity,
            "error": self.error,
            "extra": dict(self.extra),
        }


@dataclass
class IdentityScoreResult:
    """Structured pair score for logging / downstream policy."""

    adaface: ScorerComponentResult
    insightface_arcface: ScorerComponentResult
    primary_similarity: Optional[float]
    fusion: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adaface": self.adaface.to_dict(),
            "insightface_arcface": self.insightface_arcface.to_dict(),
            "primary_similarity": self.primary_similarity,
            "fusion": self.fusion,
        }


@runtime_checkable
class IdentityScorerBackend(Protocol):
    """Pluggable backend: embed a face image, compare two embeddings."""

    name: str

    def embed(self, rgb: Image.Image) -> Tuple[Optional[Any], Optional[str]]:
        """Returns (embedding, error_message)."""

    def cosine_similarity(self, a: Any, b: Any) -> Optional[float]:
        """Similarity in [0, 1] when embeddings are L2-normalized."""


class AdaFaceBackend:
    name = "adaface"

    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device
        from MIID.miner.ada_face_compare import extract_face_embedding, _resolve_device

        self._extract = extract_face_embedding
        self._device_resolved = _resolve_device(device)

    def embed(self, rgb: Image.Image) -> Tuple[Optional[Any], Optional[str]]:
        try:
            emb = self._extract(self._model, rgb, device=self._device_resolved)
            if emb is None:
                return None, "no_face_or_align_failed"
            return emb, None
        except Exception as e:
            return None, str(e)

    def cosine_similarity(self, a: Any, b: Any) -> Optional[float]:
        if a is None or b is None:
            return None
        from MIID.miner.ada_face_compare import compute_cosine_similarity

        s = compute_cosine_similarity(a, b)
        return float(s) if s is not None else None


class InsightFaceArcFaceBackend:
    """Optional InsightFace buffalo_l (ArcFace) embeddings. Install: pip install insightface onnxruntime."""

    name = "insightface_arcface"

    def __init__(self, device_id: int = 0) -> None:
        self._app = None
        self._device_id = device_id
        self._init_error: Optional[str] = None
        try:
            from insightface.app import FaceAnalysis

            name = (os.environ.get("PHASE4_INSIGHTFACE_MODEL", "buffalo_l") or "buffalo_l").strip()
            self._app = FaceAnalysis(name=name, providers=self._providers())
            self._app.prepare(ctx_id=device_id, det_size=(640, 640))
        except Exception as e:
            self._init_error = str(e)

    def _providers(self) -> List[str]:
        try:
            import onnxruntime as ort

            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return ["CPUExecutionProvider"]

    def embed(self, rgb: Image.Image) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if self._app is None:
            return None, self._init_error or "insightface_not_initialized"
        try:
            bgr = np.array(rgb)[:, :, ::-1].copy()
            faces = self._app.get(bgr)
            if not faces:
                return None, "no_face_detected"
            emb = faces[0].normed_embedding.astype(np.float32)
            return emb, None
        except Exception as e:
            return None, str(e)

    def cosine_similarity(self, a: Any, b: Any) -> Optional[float]:
        if a is None or b is None:
            return None
        ea = np.asarray(a, dtype=np.float32).reshape(-1)
        eb = np.asarray(b, dtype=np.float32).reshape(-1)
        if ea.size == 0 or eb.size == 0:
            return None
        sim = float(np.dot(ea, eb))
        return max(0.0, min(1.0, sim))


def _primary_from_components(
    ada: ScorerComponentResult,
    arc: ScorerComponentResult,
) -> Tuple[Optional[float], str]:
    mode = (os.environ.get("PHASE4_IDENTITY_PRIMARY", "adaface") or "adaface").strip().lower()
    if mode == "insightface" or mode == "insightface_arcface":
        if arc.available and arc.similarity is not None:
            return arc.similarity, "insightface_arcface"
        if ada.available and ada.similarity is not None:
            return ada.similarity, "adaface_fallback"
        return None, "no_scores"
    if mode == "mean" and ada.similarity is not None and arc.similarity is not None:
        return (ada.similarity + arc.similarity) * 0.5, "mean"
    if mode == "min" and ada.similarity is not None and arc.similarity is not None:
        return min(ada.similarity, arc.similarity), "min"
    if ada.available and ada.similarity is not None:
        return ada.similarity, "adaface"
    if arc.available and arc.similarity is not None:
        return arc.similarity, "insightface_arcface_fallback"
    return None, "no_scores"


class IdentityScoringService:
    """
    Per-request cache for base embeddings; scores candidates with all active backends.

    Call :meth:`begin_request` once per base image, then :meth:`score_candidate` per image.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        backends: Optional[Sequence[IdentityScorerBackend]] = None,
    ) -> None:
        from MIID.miner.ada_face_compare import _resolve_device

        self._device = _resolve_device(device)
        self._ada_model: Any = None
        self._backends: List[IdentityScorerBackend] = []
        if backends is not None:
            self._backends = list(backends)
        else:
            from MIID.miner.ada_face_compare import load_adaface_model

            self._ada_model = load_adaface_model(device=self._device)
            ctx_id = -1
            if self._device.startswith("cuda") and torch.cuda.is_available():
                ctx_id = 0
            self._backends = [
                AdaFaceBackend(self._ada_model, self._device),
                InsightFaceArcFaceBackend(device_id=ctx_id),
            ]

        self._base_embeds: Dict[str, Any] = {}
        self._started = False

    def has_base_embedding(self, backend_name: str = "adaface") -> bool:
        return self._base_embeds.get(backend_name, {}).get("emb") is not None

    def begin_request(self, base_rgb: Image.Image, preprocessed_aligned: Optional[Image.Image] = None) -> None:
        """Compute and cache base embeddings (aligned face preferred for AdaFace)."""
        self._base_embeds.clear()
        base_rgb = base_rgb.convert("RGB")
        for b in self._backends:
            if b.name == "adaface":
                src = preprocessed_aligned if preprocessed_aligned is not None else base_rgb
                emb, err = b.embed(src)
            else:
                emb, err = b.embed(base_rgb)
            self._base_embeds[b.name] = {"emb": emb, "error": err}
        self._started = True

    def score_candidate(self, candidate_rgb: Image.Image) -> IdentityScoreResult:
        if not self._started:
            raise RuntimeError("IdentityScoringService.begin_request must be called first")

        ada_res = ScorerComponentResult("adaface", False, None, "not_run")
        arc_res = ScorerComponentResult("insightface_arcface", False, None, "not_run")

        for b in self._backends:
            key = b.name
            cached = self._base_embeds.get(key, {})
            base_e = cached.get("emb")
            base_err = cached.get("error")
            cand_e, cand_err = b.embed(candidate_rgb)

            if key == "adaface":
                if base_err:
                    ada_res = ScorerComponentResult("adaface", False, None, base_err)
                elif cand_err:
                    ada_res = ScorerComponentResult("adaface", False, None, cand_err)
                else:
                    sim = b.cosine_similarity(base_e, cand_e)
                    ada_res = ScorerComponentResult(
                        "adaface", True, sim, None if sim is not None else "similarity_undefined"
                    )
            elif key == "insightface_arcface":
                if base_err:
                    arc_res = ScorerComponentResult("insightface_arcface", False, None, base_err)
                elif cand_err:
                    arc_res = ScorerComponentResult("insightface_arcface", False, None, cand_err)
                else:
                    sim = b.cosine_similarity(base_e, cand_e)
                    arc_res = ScorerComponentResult(
                        "insightface_arcface", True, sim, None if sim is not None else "similarity_undefined"
                    )

        primary, fusion = _primary_from_components(ada_res, arc_res)
        return IdentityScoreResult(
            adaface=ada_res,
            insightface_arcface=arc_res,
            primary_similarity=primary,
            fusion=fusion,
        )

    def score_candidates(self, candidates: Sequence[Image.Image]) -> List[IdentityScoreResult]:
        return [self.score_candidate(c.convert("RGB")) for c in candidates]

    def end_request(self) -> None:
        self._base_embeds.clear()
        self._started = False


def score_identity(
    base_image: Image.Image,
    candidate_image: Image.Image,
    *,
    device: str = "cpu",
    service: Optional[IdentityScoringService] = None,
) -> IdentityScoreResult:
    """
    One-shot pair score (no cross-call cache). For tests and simple call sites.

    Uses default backends (AdaFace + InsightFace when installable; otherwise InsightFace errors only).
    """
    svc = service or IdentityScoringService(device=device)
    base_image = base_image.convert("RGB")
    candidate_image = candidate_image.convert("RGB")
    svc.begin_request(base_image, preprocessed_aligned=None)
    try:
        return svc.score_candidate(candidate_image)
    finally:
        svc.end_request()
