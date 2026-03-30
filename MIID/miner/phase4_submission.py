# MIID/miner/phase4_submission.py
#
# Strict pre-upload verification and auditable submission manifests for SN54 Phase 4.

from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from MIID.miner.pipeline_observability import log_phase4_json


def _float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def compute_image_sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def decode_image_strict(image_bytes: bytes) -> Tuple[bool, str, Optional[Image.Image]]:
    """Decode and fully load pixels (stricter than verify-only)."""
    if not image_bytes:
        return False, "empty_payload", None
    try:
        buf = io.BytesIO(image_bytes)
        with Image.open(buf) as im:
            im.load()
            rgb = im.convert("RGB")
        return True, "ok", rgb
    except Exception as e:
        return False, f"decode_error:{e}", None


def extract_submission_final_score(variation: Dict[str, Any]) -> Optional[float]:
    """Prefer ensemble final_score for selected candidate; else AdaFace similarity."""
    scores = variation.get("ensemble_final_scores")
    idx = int(variation.get("selected_candidate_index", 0) or 0)
    if isinstance(scores, list) and scores and 0 <= idx < len(scores):
        try:
            return float(scores[idx])
        except (TypeError, ValueError):
            pass
    sim = variation.get("adaface_similarity")
    if sim is not None:
        try:
            return float(sim)
        except (TypeError, ValueError):
            pass
    return None


def verify_pre_upload(
    *,
    variation: Dict[str, Any],
    image_bytes: bytes,
    declared_hash: str,
    compiled_type: str,
    compiled_intensity: str,
    challenge_id: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Fail-closed gates before encryption/upload.

    Returns:
        (ok, reason_code, evidence_dict)
    """
    ev: Dict[str, Any] = {"challenge_id": challenge_id}

    if not image_bytes:
        return False, "phase4_submission_payload_empty", {**ev, "detail": "zero_byte_image"}

    ok_dec, dec_msg, _pil = decode_image_strict(image_bytes)
    if not ok_dec:
        return False, "phase4_submission_image_decode_failed", {**ev, "detail": dec_msg}

    computed = compute_image_sha256(image_bytes)
    dh = (declared_hash or "").strip().lower()
    if not dh or computed != dh:
        return False, "phase4_submission_hash_mismatch", {
            **ev,
            "declared_hash_prefix": dh[:16] if dh else "",
            "computed_hash_prefix": computed[:16],
        }

    vt = str(variation.get("variation_type") or "").strip()
    it = str(variation.get("intensity") or "").strip()
    if vt != compiled_type or it != compiled_intensity:
        return False, "phase4_submission_metadata_mismatch", {
            **ev,
            "got_type": vt,
            "expected_type": compiled_type,
            "got_intensity": it,
            "expected_intensity": compiled_intensity,
        }

    min_final = _float_env("PHASE4_MIN_FINAL_SCORE", 0.0)
    if min_final > 0.0:
        fs = extract_submission_final_score(variation)
        if fs is None:
            return False, "phase4_submission_final_score_unknown", {**ev, "min_final": min_final}
        if fs < min_final:
            return False, "phase4_submission_final_score_below_minimum", {
                **ev,
                "final_score": round(fs, 6),
                "min_final": min_final,
            }
        ev["final_score"] = round(fs, 6)

    ev["plaintext_size"] = len(image_bytes)
    ev["image_hash_ok"] = True
    return True, "ok", ev


def verify_submission_signature(hotkey: Any, message: str, signature_hex: str) -> Tuple[bool, str]:
    """Verify hotkey signature over UTF-8 message (hex-encoded signature)."""
    try:
        sig = bytes.fromhex(signature_hex.strip())
    except ValueError:
        return False, "invalid_hex_signature"
    try:
        if hotkey.verify(message.encode("utf-8"), sig):
            return True, "ok"
    except Exception as e:
        return False, f"verify_exception:{e}"
    return False, "signature_mismatch"


def build_submission_manifest(
    *,
    challenge_id: str,
    variation_type: str,
    intensity: str,
    image_hash: str,
    s3_key: str,
    signature: str,
    path_signature: str,
    mime: str,
    size: int,
    target_drand_round: int,
    request_index: int,
    final_score: Optional[float],
    verified_ok: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic, JSON-serializable manifest (sorted keys at dump time)."""
    m: Dict[str, Any] = {
        "challenge_id": challenge_id,
        "final_score": final_score,
        "image_hash": image_hash,
        "intensity": intensity,
        "mime": mime,
        "path_signature": path_signature,
        "request_index": int(request_index),
        "s3_key": s3_key,
        "signature": signature,
        "size": int(size),
        "target_drand_round": int(target_drand_round),
        "variation_type": variation_type,
        "verified_pre_upload": bool(verified_ok),
    }
    if extra:
        m["extra"] = extra
    return m


def write_submission_manifest_debug(
    manifest: Dict[str, Any],
    *,
    directory: str,
    basename: str,
) -> Optional[str]:
    """Write manifest JSON for local inspection. Returns path or None."""
    if not directory.strip():
        return None
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{basename}.submission_manifest.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        return path
    except OSError:
        return None


def log_submission_failure(
    reason: str,
    *,
    challenge_id: str,
    label: str,
    request_index: int,
    attempt: int,
    evidence: Optional[Dict[str, Any]] = None,
) -> None:
    fields: Dict[str, Any] = {
        "reason": reason,
        "challenge_id": challenge_id,
        "label": label,
        "request_index": request_index,
        "attempt": attempt,
    }
    if evidence:
        fields["evidence"] = evidence
    log_phase4_json("phase4_submission_failure", **fields)


# Example manifest (documentation / contract tests)
EXAMPLE_SUBMISSION_MANIFEST_JSON = """
{
  "challenge_id": "ch_01HZZ_example",
  "final_score": 0.812345,
  "image_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
  "intensity": "medium",
  "mime": "image/png",
  "path_signature": "deadbeefcafe4242",
  "request_index": 1,
  "s3_key": "submissions/ch_01HZZ_example/5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty/deadbeefcafe4242/seed/pose_edit_1710000000.png.tlock",
  "signature": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "size": 184920,
  "target_drand_round": 4500000,
  "variation_type": "pose_edit",
  "verified_pre_upload": true
}
""".strip()
