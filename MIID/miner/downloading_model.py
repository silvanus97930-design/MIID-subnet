"""Download and load the configured Phase 4 image model into the HF cache."""

import os

import torch
from diffusers import Flux2KleinPipeline, QwenImageLayeredPipeline, ZImagePipeline

BACKEND = (os.environ.get("SN54_IMAGE_GENERATION_BACKEND") or "flux").strip().lower()
device = (os.environ.get("ZIMAGE_DEVICE") or os.environ.get("FLUX_DEVICE") or "").strip().lower()
if not device:
    device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
if not token.strip():
    raise RuntimeError(
        "Set HF_TOKEN or HUGGINGFACE_TOKEN (read token from huggingface.co/settings/tokens)."
)


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _verify_loaded(pipe, *, target_device: str):
    loaded_device = target_device
    try:
        pipe = pipe.to(target_device)
    except torch.OutOfMemoryError:
        if target_device != "cpu":
            print(
                f"GPU load verification ran out of memory on {target_device}. "
                "The model weights are cached already; falling back to CPU verification."
            )
            loaded_device = "cpu"
            pipe = pipe.to("cpu")
        else:
            raise
    return pipe, loaded_device

if BACKEND == "zimage":
    model_id = os.environ.get("ZIMAGE_MODEL_ID", "Tongyi-MAI/Z-Image")
    print(f"Downloading Z-Image model: {model_id}")
    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=False,
    )
else:
    model_id = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
    print(f"Downloading FLUX model: {model_id}")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
    )

pipe, loaded_device = _verify_loaded(pipe, target_device=device)

print(f"Model downloaded and loaded successfully for backend={BACKEND} on {loaded_device}.")

if _bool_env("SN54_QWEN_LAYERED_ENABLED", True) and (
    _bool_env("SN54_QWEN_LAYERED_BACKGROUND_EDIT", True)
    or _bool_env("SN54_QWEN_LAYERED_SCREEN_REPLAY", True)
):
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    layered_model_id = os.environ.get("QWEN_LAYERED_MODEL_ID", "Qwen/Qwen-Image-Layered")
    print(f"Downloading Qwen layered helper model: {layered_model_id}")
    layered_pipe = QwenImageLayeredPipeline.from_pretrained(
        layered_model_id,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=False,
    )
    layered_pipe, layered_loaded_device = _verify_loaded(layered_pipe, target_device=device)
    print(
        "Model downloaded and loaded successfully for layered helper="
        f"{layered_model_id} on {layered_loaded_device}."
    )
