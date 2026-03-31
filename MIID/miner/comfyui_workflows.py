from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from MIID.miner.generate_variations import _get_prompt_from_request, _get_type_and_intensity, _wire_seq_field, _wire_text_field
from MIID.miner.generator_backends.base import GenerationConfig


def _int_env(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(str(raw).strip()) if raw is not None else int(default)
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(str(raw).strip()) if raw is not None else float(default)
    except Exception:
        return float(default)


def _str_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return str(value).strip() if value is not None else str(default)


def _target_dimensions(width: int, height: int) -> Tuple[int, int]:
    area_target = _int_env("COMFYUI_TARGET_PIXEL_AREA", 1024 * 1024)
    min_side = _int_env("COMFYUI_MIN_SIDE", 768)
    max_side = _int_env("COMFYUI_MAX_SIDE", 1536)
    if width <= 0 or height <= 0:
        width, height = 768, 1024
    scale = (area_target / max(width * height, 1)) ** 0.5
    tw = max(min_side, int(round(width * scale / 64.0) * 64))
    th = max(min_side, int(round(height * scale / 64.0) * 64))
    tw = max(512, min(max_side, tw))
    th = max(512, min(max_side, th))
    tw = max(64, int(round(tw / 64.0) * 64))
    th = max(64, int(round(th / 64.0) * 64))
    return tw, th


def _seed_base(req: Any, request_index: int) -> int:
    raw = "|".join(
        [
            str(time.time_ns()),
            str(request_index),
            _wire_text_field(req, "type"),
            _wire_text_field(req, "intensity"),
            _wire_text_field(req, "description"),
            _wire_text_field(req, "detail"),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _screen_replay_prompt_suffix(req: Any) -> str:
    device = _wire_text_field(req, "screen_replay_device") or _wire_text_field(req, "device_type")
    cues = _wire_seq_field(req, "visual_cue_keys")
    clauses: List[str] = []
    if device:
        clauses.append(f"display the portrait on a realistic {device} screen")
    if cues:
        clauses.append(f"screen artifacts to include: {', '.join(cues)}")
    clauses.append("preserve the same subject identity while simulating a real photographed display")
    return ". ".join(clauses) + "."


def _task_negative_prompt(var_type: str) -> str:
    default = _str_env(
        "COMFYUI_NEGATIVE_PROMPT",
        "blurry, low quality, deformed face, duplicate face, extra people, bad anatomy, cropped head, unreadable image",
    )
    if var_type == "screen_replay":
        extra = _str_env(
            "COMFYUI_SCREEN_REPLAY_NEGATIVE_PROMPT",
            "broken bezel, duplicated monitor frame, unreadable moire blocks, distorted fingers, second person",
        )
        return f"{default}, {extra}".strip(", ")
    return default


def _intensity_strength(var_type: str, intensity: str) -> float:
    family = var_type.upper()
    level = intensity.upper()
    return _float_env(f"COMFYUI_{family}_{level}_DENOISE", _float_env(f"COMFYUI_{family}_DENOISE", 0.42))


def _intensity_steps(var_type: str, intensity: str, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    family = var_type.upper()
    level = intensity.upper()
    return _int_env(f"COMFYUI_{family}_{level}_STEPS", _int_env(f"COMFYUI_{family}_STEPS", 28))


def _intensity_cfg(var_type: str, intensity: str, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    family = var_type.upper()
    level = intensity.upper()
    return _float_env(f"COMFYUI_{family}_{level}_CFG", _float_env(f"COMFYUI_{family}_CFG", 5.0))


@dataclass(frozen=True)
class WorkflowPlan:
    workflow_name: str
    prompt: Dict[str, Dict[str, Any]]
    save_node_ids: Tuple[str, ...]
    metadata: Dict[str, Any]
    required_nodes: Tuple[str, ...]
    capability_notes: Tuple[str, ...]


class _GraphBuilder:
    def __init__(self) -> None:
        self.prompt: Dict[str, Dict[str, Any]] = {}
        self._counter = 1

    def add(self, class_type: str, inputs: Dict[str, Any]) -> str:
        node_id = str(self._counter)
        self._counter += 1
        self.prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }
        return node_id

    @staticmethod
    def ref(node_id: str, index: int = 0) -> List[Any]:
        return [str(node_id), int(index)]


def _build_img2img_workflow(
    *,
    workflow_name: str,
    req: Any,
    uploaded_image: str,
    source_size: Tuple[int, int],
    generation_config: GenerationConfig,
    available_nodes: Set[str],
    request_index: int,
    use_ipadapter: bool,
    use_outpaint: bool,
    use_controlnet: bool,
    add_screen_prompt: bool,
) -> WorkflowPlan:
    var_type, intensity = _get_type_and_intensity(req)
    positive_prompt = _get_prompt_from_request(req, var_type, intensity)
    if add_screen_prompt:
        positive_prompt = f"{positive_prompt} {_screen_replay_prompt_suffix(req)}"
    negative_prompt = _task_negative_prompt(var_type)
    steps = _intensity_steps(var_type, intensity, generation_config.num_inference_steps_override)
    cfg = _intensity_cfg(var_type, intensity, generation_config.guidance_scale_override)
    denoise = max(0.05, min(1.0, _intensity_strength(var_type, intensity)))
    num_candidates = max(1, int(generation_config.candidates_per_request))
    width, height = _target_dimensions(*source_size)
    sampler_name = _str_env("COMFYUI_SAMPLER_NAME", "euler")
    scheduler = _str_env("COMFYUI_SCHEDULER", "karras")
    checkpoint = _str_env("COMFYUI_SDXL_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    capability_notes: List[str] = []

    gb = _GraphBuilder()
    ckpt = gb.add("CheckpointLoaderSimple", {"ckpt_name": checkpoint})
    load = gb.add("LoadImage", {"image": uploaded_image})
    image_ref = _GraphBuilder.ref(load, 0)

    if use_outpaint:
        pad = gb.add(
            "ImagePadForOutpaint",
            {
                "image": image_ref,
                "left": _int_env("COMFYUI_BACKGROUND_OUTPAINT_LEFT", 128),
                "top": _int_env("COMFYUI_BACKGROUND_OUTPAINT_TOP", 128),
                "right": _int_env("COMFYUI_BACKGROUND_OUTPAINT_RIGHT", 128),
                "bottom": _int_env("COMFYUI_BACKGROUND_OUTPAINT_BOTTOM", 128),
                "feathering": _int_env("COMFYUI_BACKGROUND_OUTPAINT_FEATHER", 48),
            },
        )
        image_ref = _GraphBuilder.ref(pad, 0)

    scale = gb.add(
        "ImageScale",
        {
            "image": image_ref,
            "upscale_method": "lanczos",
            "width": width,
            "height": height,
            "crop": "disabled",
        },
    )
    scaled_image = _GraphBuilder.ref(scale, 0)
    positive = gb.add("CLIPTextEncode", {"text": positive_prompt, "clip": _GraphBuilder.ref(ckpt, 1)})
    negative = gb.add("CLIPTextEncode", {"text": negative_prompt, "clip": _GraphBuilder.ref(ckpt, 1)})
    model_ref: List[Any] = _GraphBuilder.ref(ckpt, 0)
    positive_ref: List[Any] = _GraphBuilder.ref(positive, 0)

    if use_controlnet and "ControlNetLoader" in available_nodes and "ControlNetApply" in available_nodes:
        controlnet_name = _str_env("COMFYUI_POSE_CONTROLNET_MODEL", "diffusion_pytorch_model.bin")
        cnet = gb.add("ControlNetLoader", {"control_net_name": controlnet_name})
        applied = gb.add(
            "ControlNetApply",
            {
                "conditioning": positive_ref,
                "control_net": _GraphBuilder.ref(cnet, 0),
                "image": scaled_image,
                "strength": _float_env("COMFYUI_POSE_CONTROLNET_STRENGTH", 0.65),
            },
        )
        positive_ref = _GraphBuilder.ref(applied, 0)
    elif use_controlnet:
        capability_notes.append("pose_edit fell back without ControlNet because the required ComfyUI nodes were unavailable")

    if use_ipadapter and {"IPAdapterModelLoader", "IPAdapterFaceID"}.issubset(available_nodes):
        ip_model = gb.add(
            "IPAdapterModelLoader",
            {"ipadapter_file": _str_env("COMFYUI_IPADAPTER_MODEL", "ip-adapter-faceid-plusv2_sdxl.bin")},
        )
        face_inputs: Dict[str, Any] = {
            "model": model_ref,
            "ipadapter": _GraphBuilder.ref(ip_model, 0),
            "image": scaled_image,
            "weight": _float_env("COMFYUI_IPADAPTER_WEIGHT", 0.8),
            "weight_faceidv2": _float_env("COMFYUI_IPADAPTER_FACEID_V2_WEIGHT", 1.2),
            "weight_type": _str_env("COMFYUI_IPADAPTER_WEIGHT_TYPE", "linear"),
            "combine_embeds": _str_env("COMFYUI_IPADAPTER_COMBINE_EMBEDS", "concat"),
            "start_at": _float_env("COMFYUI_IPADAPTER_START_AT", 0.0),
            "end_at": _float_env("COMFYUI_IPADAPTER_END_AT", 1.0),
            "embeds_scaling": _str_env("COMFYUI_IPADAPTER_EMBEDS_SCALING", "V only"),
        }
        if "IPAdapterInsightFaceLoader" in available_nodes:
            insight = gb.add(
                "IPAdapterInsightFaceLoader",
                {
                    "provider": _str_env(
                        "COMFYUI_INSIGHTFACE_PROVIDER",
                        "CUDA" if str(os.environ.get("INSIGHTFACE_DEVICE", "cuda")).lower().startswith("cuda") else "CPU",
                    ),
                    "model_name": _str_env("COMFYUI_INSIGHTFACE_MODEL", "buffalo_l"),
                },
            )
            face_inputs["insightface"] = _GraphBuilder.ref(insight, 0)
        face = gb.add("IPAdapterFaceID", face_inputs)
        model_ref = _GraphBuilder.ref(face, 0)
    elif use_ipadapter:
        capability_notes.append(f"{var_type} fell back without IP-Adapter FaceID because the required ComfyUI nodes were unavailable")

    latent = gb.add("VAEEncode", {"pixels": scaled_image, "vae": _GraphBuilder.ref(ckpt, 2)})
    save_nodes: List[str] = []
    seed_base = _seed_base(req, request_index)
    save_prefix = _str_env("COMFYUI_SAVE_PREFIX", "MIID")
    for idx in range(num_candidates):
        sampler = gb.add(
            "KSampler",
            {
                "model": model_ref,
                "seed": int((seed_base + idx) % (2**63 - 1)),
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive_ref,
                "negative": _GraphBuilder.ref(negative, 0),
                "latent_image": _GraphBuilder.ref(latent, 0),
                "denoise": denoise,
            },
        )
        decode = gb.add("VAEDecode", {"samples": _GraphBuilder.ref(sampler, 0), "vae": _GraphBuilder.ref(ckpt, 2)})
        save = gb.add(
            "SaveImage",
            {"images": _GraphBuilder.ref(decode, 0), "filename_prefix": f"{save_prefix}_{workflow_name}_{request_index}_{idx}"},
        )
        save_nodes.append(save)

    metadata: Dict[str, Any] = {
        "variation_type": var_type,
        "intensity": intensity,
        "prompt": positive_prompt,
        "description": _wire_text_field(req, "description"),
        "detail": _wire_text_field(req, "detail"),
        "workflow_name": workflow_name,
        "pipeline": f"comfyui_{var_type}",
        "comfyui_capability_notes": capability_notes,
    }
    if var_type == "screen_replay":
        device = _wire_text_field(req, "screen_replay_device") or _wire_text_field(req, "device_type")
        cues = _wire_seq_field(req, "visual_cue_keys")
        metadata["screen_replay_device"] = device or None
        metadata["visual_cue_keys"] = cues or None

    return WorkflowPlan(
        workflow_name=workflow_name,
        prompt=gb.prompt,
        save_node_ids=tuple(save_nodes),
        metadata=metadata,
        required_nodes=("CheckpointLoaderSimple", "LoadImage", "CLIPTextEncode", "VAEEncode", "KSampler", "VAEDecode", "SaveImage"),
        capability_notes=tuple(capability_notes),
    )


def build_workflow_plan(
    req: Any,
    *,
    uploaded_image: str,
    source_size: Tuple[int, int],
    generation_config: GenerationConfig,
    available_nodes: Set[str],
    request_index: int,
    workflow_name_override: Optional[str] = None,
) -> WorkflowPlan:
    var_type, _ = _get_type_and_intensity(req)
    if var_type == "pose_edit":
        return _build_img2img_workflow(
            workflow_name=workflow_name_override or "pose_edit_sdxl_instantid_or_fallback",
            req=req,
            uploaded_image=uploaded_image,
            source_size=source_size,
            generation_config=generation_config,
            available_nodes=available_nodes,
            request_index=request_index,
            use_ipadapter=True,
            use_outpaint=False,
            use_controlnet=True,
            add_screen_prompt=False,
        )
    if var_type == "expression_edit":
        return _build_img2img_workflow(
            workflow_name=workflow_name_override or "expression_edit_sdxl_ipadapter_faceid",
            req=req,
            uploaded_image=uploaded_image,
            source_size=source_size,
            generation_config=generation_config,
            available_nodes=available_nodes,
            request_index=request_index,
            use_ipadapter=True,
            use_outpaint=False,
            use_controlnet=False,
            add_screen_prompt=False,
        )
    if var_type == "background_edit":
        return _build_img2img_workflow(
            workflow_name=workflow_name_override or "background_edit_sdxl_inpaint_ipadapter",
            req=req,
            uploaded_image=uploaded_image,
            source_size=source_size,
            generation_config=generation_config,
            available_nodes=available_nodes,
            request_index=request_index,
            use_ipadapter=True,
            use_outpaint=True,
            use_controlnet=False,
            add_screen_prompt=False,
        )
    if var_type == "lighting_edit":
        return _build_img2img_workflow(
            workflow_name=workflow_name_override or "lighting_edit_iclight_or_sdxl_fallback",
            req=req,
            uploaded_image=uploaded_image,
            source_size=source_size,
            generation_config=generation_config,
            available_nodes=available_nodes,
            request_index=request_index,
            use_ipadapter=False,
            use_outpaint=False,
            use_controlnet=False,
            add_screen_prompt=False,
        )
    if var_type == "screen_replay":
        return _build_img2img_workflow(
            workflow_name=workflow_name_override or "screen_replay_sdxl_postprocess_chain",
            req=req,
            uploaded_image=uploaded_image,
            source_size=source_size,
            generation_config=generation_config,
            available_nodes=available_nodes,
            request_index=request_index,
            use_ipadapter=True,
            use_outpaint=False,
            use_controlnet=False,
            add_screen_prompt=True,
        )
    return _build_img2img_workflow(
        workflow_name=workflow_name_override or f"{var_type}_generic_sdxl",
        req=req,
        uploaded_image=uploaded_image,
        source_size=source_size,
        generation_config=generation_config,
        available_nodes=available_nodes,
        request_index=request_index,
        use_ipadapter=False,
        use_outpaint=False,
        use_controlnet=False,
        add_screen_prompt=False,
    )
