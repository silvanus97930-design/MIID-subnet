# MIID/miner/face_preprocess.py
#
# Dominant-face detection, alignment, and passport-style 3:4 working crop for Phase 4.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

FaceBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Landmark5:
    """Five facial landmarks (x, y) in original image coordinates."""

    points: Tuple[Tuple[float, float], ...]

    def as_flat(self) -> Tuple[float, ...]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return tuple(xs + ys)


@dataclass
class FacePreprocessResult:
    """Output of seed-face preprocessing (one base image per Phase 4 request)."""

    ok: bool
    face_count: int
    dominant_index: int
    face_box_xyxy: Optional[FaceBox]
    landmarks: Optional[Landmark5]
    aligned_face: Optional[Image.Image]
    working_canvas: Optional[Image.Image]
    message: str = ""
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def _box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _landmark_from_row(row: np.ndarray) -> Landmark5:
    pts = tuple((float(row[i]), float(row[i + 5])) for i in range(5))
    return Landmark5(points=pts)


def _passport_canvas_3_4(
    rgb: Image.Image,
    box_xyxy: FaceBox,
    *,
    face_expand: float = None,
    target_short_side: int = None,
) -> Image.Image:
    """Crop a portrait 3:4 (width:height) region around the face box, then resize."""
    if face_expand is None:
        face_expand = float(os.environ.get("PHASE4_FACE_CROP_EXPAND", "0.45") or 0.45)
    if target_short_side is None:
        target_short_side = max(
            256, int(os.environ.get("PHASE4_WORKING_CANVAS_SHORT_SIDE", "512") or 512)
        )

    w, h = rgb.size
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    side = max(bw, bh) * (1.0 + 2.0 * face_expand)
    crop_h = side
    crop_w = crop_h * (3.0 / 4.0)

    left = cx - crop_w * 0.5
    top = cy - crop_h * 0.5
    right = left + crop_w
    bottom = top + crop_h

    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > w:
        shift = right - w
        left -= shift
        right = w
    if bottom > h:
        shift = bottom - h
        top -= shift
        bottom = h

    left_i = int(max(0, min(w - 1, round(left))))
    top_i = int(max(0, min(h - 1, round(top))))
    right_i = int(max(left_i + 1, min(w, round(right))))
    bottom_i = int(max(top_i + 1, min(h, round(bottom))))

    cropped = rgb.crop((left_i, top_i, right_i, bottom_i))
    cw, ch = cropped.size
    if cw <= 0 or ch <= 0:
        return rgb.copy()

    target_w = target_short_side
    target_h = int(round(target_short_side * (4.0 / 3.0)))
    return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)


def preprocess_seed_face(
    image: Union[Image.Image, str],
    device: str = "cpu",
) -> FacePreprocessResult:
    """
    Detect the dominant face, align it (112×112), and build a passport-style 3:4 working canvas.

    Uses the miner's bundled AdaFace MTCNN stack (same as ada_face_compare).
    """
    from MIID.miner import ada_face_compare as afc

    resolved = afc._resolve_device(device)
    afc._ensure_mtcnn_model(resolved)
    m = afc.align.mtcnn_model

    if isinstance(image, str):
        rgb = afc._to_rgb_pil_image(image)
    else:
        rgb = image.convert("RGB")

    boxes, landmarks = m.detect_faces(
        rgb, m.min_face_size, m.thresholds, m.nms_thresholds, m.factor
    )

    if boxes is None or len(boxes) == 0:
        return FacePreprocessResult(
            ok=False,
            face_count=0,
            dominant_index=-1,
            face_box_xyxy=None,
            landmarks=None,
            aligned_face=None,
            working_canvas=None,
            message="no_face_detected",
        )

    n = len(boxes)
    areas = [_box_area(boxes[i]) for i in range(n)]
    dom_idx = int(np.argmax(np.array(areas)))

    _, faces = m.align_multi(rgb, limit=None)
    if dom_idx >= len(faces):
        return FacePreprocessResult(
            ok=False,
            face_count=n,
            dominant_index=dom_idx,
            face_box_xyxy=None,
            landmarks=None,
            aligned_face=None,
            working_canvas=None,
            message="alignment_index_mismatch",
        )

    b = boxes[dom_idx]
    face_box: FaceBox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    lm = _landmark_from_row(landmarks[dom_idx])
    aligned = faces[dom_idx]
    canvas = _passport_canvas_3_4(rgb, face_box)

    warns: List[str] = []
    if n > 1:
        warns.append(f"multiple_faces_using_largest_area(n={n})")

    return FacePreprocessResult(
        ok=True,
        face_count=n,
        dominant_index=dom_idx,
        face_box_xyxy=face_box,
        landmarks=lm,
        aligned_face=aligned,
        working_canvas=canvas,
        message="ok",
        warnings=tuple(warns),
    )
