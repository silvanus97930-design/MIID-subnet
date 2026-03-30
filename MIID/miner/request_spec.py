# MIID/miner/request_spec.py
#
# Strict Phase 4 image variation request compiler: wire payloads -> validated internal specs.

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
import logging
from typing import Any, List, Optional, Sequence, Tuple

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums (wire values match validator IMAGE_VARIATION_TYPES / screen_replay)
# ---------------------------------------------------------------------------


class VariationType(str, Enum):
    POSE_EDIT = "pose_edit"
    LIGHTING_EDIT = "lighting_edit"
    EXPRESSION_EDIT = "expression_edit"
    BACKGROUND_EDIT = "background_edit"
    SCREEN_REPLAY = "screen_replay"


class PortraitIntensity(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    FAR = "far"


class ScreenReplayIntensity(str, Enum):
    STANDARD = "standard"


class ParsedIntensity(str, Enum):
    """Normalized intensity on the wire."""

    LIGHT = "light"
    MEDIUM = "medium"
    FAR = "far"
    STANDARD = "standard"


class PoseAngleBin(str, Enum):
    """Pose detail bins (validator YEVS-style copy)."""

    LIGHT_PM15 = "light_pm15"
    MEDIUM_PM30 = "medium_pm30"
    FAR_GT45 = "far_gt45"
    UNKNOWN = "unknown"


class ExpressionTarget(str, Enum):
    NEUTRAL = "neutral"
    SLIGHT_SMILE = "slight_smile"
    SMILE = "smile"
    SERIOUS = "serious"
    SURPRISED = "surprised"
    LAUGHING = "laughing"
    CONCERNED = "concerned"
    ATTENTIVE = "attentive"
    RELAXED = "relaxed"
    MINOR_BROW = "minor_brow"
    PRONOUNCED_EXPRESSION = "pronounced_expression"


class BackgroundModifier(str, Enum):
    COLOR_SHIFT = "color_shift"
    BLUR = "blur"
    TEXTURE = "texture"
    OFFICE = "office"
    OUTDOOR = "outdoor"
    GRADIENT = "gradient"
    SOLID_COLOR = "solid_color"
    DRAMATIC = "dramatic"
    CONTRASTING_ENV = "contrasting_env"
    COMPLEX_SCENE = "complex_scene"
    UNUSUAL_ENV = "unusual_env"


class AccessoryKind(str, Enum):
    RELIGIOUS_HEAD_COVERING = "religious_head_covering"
    BRIM_HAT = "brim_hat"
    KNIT_WINTER_HAT = "knit_winter_hat"
    BANDANA = "bandana"
    BASEBALL_CAP = "baseball_cap"


class ScreenDeviceType(str, Enum):
    PHONE = "phone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    MONITOR = "monitor"
    TV = "tv"
    UNKNOWN = "unknown"


# Validator detail embeds these cue phrases (see MIID.validator.image_variations.SCREEN_REPLAY_VISUAL_CUES).
_SCREEN_CUE_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("edge_crop_cues", "edge / crop"),
    ("gamma_contrast_shift", "gamma / contrast"),
    ("moire_pixel_grid", "moiré / pixel grid"),
    ("perspective_keystone_distortion", "perspective / keystone"),
    ("screen_glare_hotspots", "screen glare hotspots"),
)

# Expression phrases longest-first for deterministic greedy match
_EXPRESSION_PHRASES: Tuple[Tuple[str, ExpressionTarget], ...] = tuple(
    sorted(
        (
            ("neutral to slight smile", ExpressionTarget.SLIGHT_SMILE),
            ("neutral to smile", ExpressionTarget.SMILE),
            ("neutral to attentive", ExpressionTarget.ATTENTIVE),
            ("minor brow movement", ExpressionTarget.MINOR_BROW),
            ("relaxed to attentive", ExpressionTarget.ATTENTIVE),
            ("pronounced expression", ExpressionTarget.PRONOUNCED_EXPRESSION),
            ("mildly surprised", ExpressionTarget.SURPRISED),
            ("slight smile", ExpressionTarget.SLIGHT_SMILE),
            ("serious", ExpressionTarget.SERIOUS),
            ("surprised", ExpressionTarget.SURPRISED),
            ("laughing", ExpressionTarget.LAUGHING),
            ("concerned", ExpressionTarget.CONCERNED),
            ("attentive", ExpressionTarget.ATTENTIVE),
            ("relaxed", ExpressionTarget.RELAXED),
            ("neutral", ExpressionTarget.NEUTRAL),
            ("smile", ExpressionTarget.SMILE),
        ),
        key=lambda x: len(x[0]),
        reverse=True,
    )
)

_BACKGROUND_PHRASES: Tuple[Tuple[str, BackgroundModifier], ...] = tuple(
    sorted(
        (
            ("unusual or contrasting environment", BackgroundModifier.UNUSUAL_ENV),
            ("complex scene", BackgroundModifier.COMPLEX_SCENE),
            ("dramatic setting", BackgroundModifier.DRAMATIC),
            ("solid color to gradient", BackgroundModifier.GRADIENT),
            ("office to outdoor", BackgroundModifier.OUTDOOR),
            ("outdoor", BackgroundModifier.OUTDOOR),
            ("office", BackgroundModifier.OFFICE),
            ("gradient", BackgroundModifier.GRADIENT),
            ("solid color", BackgroundModifier.SOLID_COLOR),
            ("blur adjustment", BackgroundModifier.BLUR),
            ("texture change", BackgroundModifier.TEXTURE),
            ("color shift", BackgroundModifier.COLOR_SHIFT),
            ("contrasting environment", BackgroundModifier.CONTRASTING_ENV),
        ),
        key=lambda x: len(x[0]),
        reverse=True,
    )
)

_ACCESSORY_RULES: Tuple[Tuple[str, AccessoryKind], ...] = (
    ("hijab", AccessoryKind.RELIGIOUS_HEAD_COVERING),
    ("turban", AccessoryKind.RELIGIOUS_HEAD_COVERING),
    ("kippah", AccessoryKind.RELIGIOUS_HEAD_COVERING),
    ("taqiyah", AccessoryKind.RELIGIOUS_HEAD_COVERING),
    ("religious head covering", AccessoryKind.RELIGIOUS_HEAD_COVERING),
    ("baseball cap", AccessoryKind.BASEBALL_CAP),
    ("sports cap", AccessoryKind.BASEBALL_CAP),
    ("fedora", AccessoryKind.BRIM_HAT),
    ("wide-brim", AccessoryKind.BRIM_HAT),
    ("sun hat", AccessoryKind.BRIM_HAT),
    ("brim hat", AccessoryKind.BRIM_HAT),
    ("beanie", AccessoryKind.KNIT_WINTER_HAT),
    ("knit hat", AccessoryKind.KNIT_WINTER_HAT),
    ("winter hat", AccessoryKind.KNIT_WINTER_HAT),
    ("bandana", AccessoryKind.BANDANA),
)


@dataclass(frozen=True)
class PortraitFramingRequirements:
    """Structured portrait constraints parsed from template / description / detail text."""

    passport_style: bool
    aspect_ratio_3_4: bool
    head_and_shoulders: bool
    chest_up: bool


@dataclass(frozen=True)
class ScreenReplayConstraints:
    """Screen replay: device + visual cues (parsed from description/detail; wire has no extra fields)."""

    primary_device: ScreenDeviceType
    visual_cue_keys: Tuple[str, ...]


@dataclass(frozen=True)
class CompiledVariationRequest:
    """Validated, normalized single variation."""

    variation_type: VariationType
    intensity: ParsedIntensity
    description: str
    detail: str
    portrait_framing: PortraitFramingRequirements
    screen_replay: Optional[ScreenReplayConstraints]
    pose_angle_bin: PoseAngleBin
    expression_targets: Tuple[ExpressionTarget, ...]
    background_modifiers: Tuple[BackgroundModifier, ...]
    accessories: Tuple[AccessoryKind, ...]
    extra_modifiers: Tuple[str, ...]

    def as_protocol_request(self) -> SimpleNamespace:
        """Object compatible with FLUX path (getattr .type / .intensity / .description / .detail).

        For ``screen_replay``, also exposes ``screen_replay_device`` and ``visual_cue_keys`` for the
        dedicated screen-replay synthesizer (``MIID.miner.screen_replay``).
        """
        ns = SimpleNamespace(
            type=self.variation_type.value,
            intensity=self.intensity.value,
            description=self.description,
            detail=self.detail,
            pose_angle_bin=self.pose_angle_bin.value,
            expression_targets=tuple(t.value for t in self.expression_targets),
            background_modifiers=tuple(t.value for t in self.background_modifiers),
            accessories=tuple(t.value for t in self.accessories),
            extra_modifiers=tuple(self.extra_modifiers),
            passport_style=bool(self.portrait_framing.passport_style),
            aspect_ratio_3_4=bool(self.portrait_framing.aspect_ratio_3_4),
            head_and_shoulders=bool(self.portrait_framing.head_and_shoulders),
            chest_up=bool(self.portrait_framing.chest_up),
        )
        if self.screen_replay is not None:
            ns.screen_replay_device = self.screen_replay.primary_device.value
            ns.visual_cue_keys = tuple(self.screen_replay.visual_cue_keys)
        return ns

    def label(self) -> str:
        return f"{self.variation_type.value}({self.intensity.value})"


@dataclass(frozen=True)
class CompiledImageRequest:
    """Compiled variation list for one Phase 4 image request."""

    variations: Tuple[CompiledVariationRequest, ...]


def _get_str(obj: Any, key: str) -> str:
    if obj is None:
        return ""
    v = getattr(obj, key, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(key)
    if v is None:
        return ""
    return str(v).strip()


def _normalize_variation_type(raw: str) -> Optional[VariationType]:
    s = raw.strip().lower()
    if not s:
        return None
    for vt in VariationType:
        if vt.value == s:
            return vt
    return None


def _normalize_intensity(raw: str) -> Optional[ParsedIntensity]:
    s = raw.strip().lower()
    if not s:
        return None
    for pi in ParsedIntensity:
        if pi.value == s:
            return pi
    return None


def _validate_type_intensity_pair(vt: VariationType, intensity: ParsedIntensity) -> Optional[str]:
    if vt == VariationType.SCREEN_REPLAY:
        if intensity != ParsedIntensity.STANDARD:
            return f"screen_replay requires intensity 'standard', got '{intensity.value}'"
    else:
        if intensity not in (ParsedIntensity.LIGHT, ParsedIntensity.MEDIUM, ParsedIntensity.FAR):
            return f"{vt.value} requires intensity light|medium|far, got '{intensity.value}'"
    return None


def _parse_portrait_framing(description: str, detail: str) -> PortraitFramingRequirements:
    blob = f"{description}\n{detail}".lower()
    return PortraitFramingRequirements(
        passport_style=("passport-style" in blob or "passport style" in blob),
        aspect_ratio_3_4=("3:4" in blob or "3 : 4" in blob),
        head_and_shoulders=("head-and-shoulders" in blob or "head and shoulders" in blob),
        chest_up=("chest up" in blob or "from chest up" in blob),
    )


def _parse_pose_angle_bin(vt: VariationType, intensity: ParsedIntensity, detail: str) -> PoseAngleBin:
    if vt != VariationType.POSE_EDIT:
        return PoseAngleBin.UNKNOWN
    d = detail.lower()
    if ">±45" in detail or "near-profile" in d or "near profile" in d:
        return PoseAngleBin.FAR_GT45
    if "±30" in detail or "30°" in detail or "30 degree" in d:
        return PoseAngleBin.MEDIUM_PM30
    if "±15" in detail or "15°" in detail or "15 degree" in d:
        return PoseAngleBin.LIGHT_PM15
    if intensity == ParsedIntensity.FAR:
        return PoseAngleBin.FAR_GT45
    if intensity == ParsedIntensity.MEDIUM:
        return PoseAngleBin.MEDIUM_PM30
    if intensity == ParsedIntensity.LIGHT:
        return PoseAngleBin.LIGHT_PM15
    return PoseAngleBin.UNKNOWN


def _parse_expression_targets(vt: VariationType, detail: str) -> Tuple[ExpressionTarget, ...]:
    if vt != VariationType.EXPRESSION_EDIT:
        return ()
    d = detail.lower()
    found: List[ExpressionTarget] = []
    seen = set()
    for phrase, target in _EXPRESSION_PHRASES:
        if phrase in d and target not in seen:
            found.append(target)
            seen.add(target)
    return tuple(found)


def _parse_background_modifiers(vt: VariationType, detail: str, description: str) -> Tuple[BackgroundModifier, ...]:
    if vt != VariationType.BACKGROUND_EDIT:
        return ()
    blob = f"{description}\n{detail}".lower()
    found: List[BackgroundModifier] = []
    seen = set()
    for phrase, mod in _BACKGROUND_PHRASES:
        if phrase in blob and mod not in seen:
            found.append(mod)
            seen.add(mod)
    return tuple(found)


def _split_accessory_segment(detail: str) -> str:
    lower = detail.lower()
    idx = lower.find("additionally, include:")
    if idx < 0:
        return ""
    return detail[idx + len("Additionally, include:") :].strip()


def _parse_accessories(vt: VariationType, detail: str) -> Tuple[AccessoryKind, ...]:
    if vt != VariationType.BACKGROUND_EDIT:
        return ()
    seg = _split_accessory_segment(detail)
    if not seg:
        return ()
    blob = seg.lower()
    found: List[AccessoryKind] = []
    seen = set()
    for phrase, kind in _ACCESSORY_RULES:
        if phrase in blob and kind not in seen:
            found.append(kind)
            seen.add(kind)
    return tuple(found)


def _parse_screen_device(description: str, detail: str) -> ScreenDeviceType:
    blob = f"{description}\n{detail}".lower()
    order = (
        ScreenDeviceType.TABLET,
        ScreenDeviceType.MONITOR,
        ScreenDeviceType.LAPTOP,
        ScreenDeviceType.PHONE,
        ScreenDeviceType.TV,
    )
    for dev in order:
        token = dev.value
        if re.search(rf"\b{re.escape(token)}\b", blob):
            return dev
    return ScreenDeviceType.UNKNOWN


def _parse_screen_cues(detail: str) -> Tuple[str, ...]:
    d = detail.lower()
    matched: List[str] = []
    seen = set()
    for key, needle in _SCREEN_CUE_MARKERS:
        if needle in d and key not in seen:
            matched.append(key)
            seen.add(key)
    if "moire_pixel_grid" not in seen and "moire" in d and "pixel" in d:
        matched.append("moire_pixel_grid")
        seen.add("moire_pixel_grid")
    matched.sort()
    return tuple(matched)


def _parse_screen_replay_constraints(
    vt: VariationType, description: str, detail: str
) -> Optional[ScreenReplayConstraints]:
    if vt != VariationType.SCREEN_REPLAY:
        return None
    device = _parse_screen_device(description, detail)
    cues = _parse_screen_cues(detail)
    return ScreenReplayConstraints(primary_device=device, visual_cue_keys=cues)


def _extra_modifiers(description: str, detail: str) -> Tuple[str, ...]:
    """Residue tokens from bracketed or quoted hints (deterministic, conservative)."""
    text = f"{description} {detail}"
    parts = re.findall(r"\[([^\]]{1,120})\]", text)
    out = tuple(p.strip() for p in parts if p.strip())
    return out


def _compile_one(raw: Any, index: int) -> Tuple[Optional[CompiledVariationRequest], List[str]]:
    errors: List[str] = []
    type_raw = _get_str(raw, "type")
    intensity_raw = _get_str(raw, "intensity")
    description = _get_str(raw, "description")
    detail = _get_str(raw, "detail")

    vt = _normalize_variation_type(type_raw)
    if vt is None:
        errors.append(f"variation[{index}]: invalid or missing type {type_raw!r}")
        return None, errors

    intensity = _normalize_intensity(intensity_raw)
    if intensity is None:
        errors.append(f"variation[{index}]: invalid or missing intensity {intensity_raw!r}")
        return None, errors

    pair_err = _validate_type_intensity_pair(vt, intensity)
    if pair_err:
        errors.append(f"variation[{index}]: {pair_err}")
        return None, errors

    portrait = _parse_portrait_framing(description, detail)
    screen = _parse_screen_replay_constraints(vt, description, detail)
    pose_bin = _parse_pose_angle_bin(vt, intensity, detail)
    expr = _parse_expression_targets(vt, detail)
    bg = _parse_background_modifiers(vt, detail, description)
    acc = _parse_accessories(vt, detail)
    extra = _extra_modifiers(description, detail)

    if vt == VariationType.SCREEN_REPLAY and screen is not None:
        if len(screen.visual_cue_keys) < 1:
            errors.append(
                f"variation[{index}]: screen_replay detail did not match known visual cue phrases "
                f"(expected validator cues such as moiré/glare/keystone); detail may be malformed"
            )
            return None, errors

    compiled = CompiledVariationRequest(
        variation_type=vt,
        intensity=intensity,
        description=description,
        detail=detail,
        portrait_framing=portrait,
        screen_replay=screen,
        pose_angle_bin=pose_bin,
        expression_targets=expr,
        background_modifiers=bg,
        accessories=acc,
        extra_modifiers=extra,
    )
    return compiled, []


def compile_phase4_variation_requests(raw: Sequence[Any]) -> Tuple[Optional[CompiledImageRequest], List[str]]:
    """
    Compile validator variation payloads into strict internal specs.

    Returns:
        (CompiledImageRequest, []) on success
        (None, [error strings]) on failure (caller should log and abort Phase 4).
    """
    if not raw:
        return None, ["no variation_requests"]

    all_errors: List[str] = []
    compiled_list: List[CompiledVariationRequest] = []

    for i, item in enumerate(raw, start=1):
        one, errs = _compile_one(item, i)
        all_errors.extend(errs)
        if one is None:
            continue
        compiled_list.append(one)

    if all_errors:
        return None, all_errors

    return CompiledImageRequest(variations=tuple(compiled_list)), []


def log_request_spec_errors(errors: Sequence[str], *, challenge_id: Optional[str] = None) -> None:
    """Structured rejection logs (JSON-friendly lines)."""
    prefix = f"Phase 4 request spec: challenge_id={challenge_id} " if challenge_id else "Phase 4 request spec: "
    for msg in errors:
        _logger.error("%s%s", prefix, msg)
