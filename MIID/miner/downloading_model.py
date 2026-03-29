"""
FLUX.2-klein model downloader for MIID miner image generation.

This script uses FLUX.2-klein as a starting point—one of many models you can use.
You can swap in other diffusion models (e.g. other FLUX variants, Stable Diffusion,
SDXL, or custom checkpoints) by changing the model ID and the pipeline class below.

What this is:
  A standalone script that downloads and loads the FLUX.2-klein-4B diffusion model
  from Hugging Face. Running it once caches the model on disk so the miner does
  not need to download it during normal operation.

What it does:
  - Fetches the black-forest-labs/FLUX.2-klein-4B weights (if not already cached).
  - Loads the pipeline into memory to verify the download.
  - After a successful run, the cached model can be used by the miner's Phase 4
    image generation (e.g. generate_variations in image_generator.py) to produce
    identity-preserving variations (pose, expression, lighting, background) from
    a base face image when validators send image_request in an IdentitySynapse.

How it fits the miner:
  - Miner Phase 4 (neurons/miner.py) handles synapse.image_request and calls
    process_image_request() -> generate_variations() in MIID.miner.image_generator.
  - That pipeline is intended to use FLUX (or similar) to generate high-quality
    face variations; this script ensures the FLUX.2-klein model is available
    locally before you run the miner with real image generation enabled.

Parameters you can change for your machine:
  - device: "cpu" | "cuda" | "mps"
      Use "cuda" for NVIDIA GPU, "mps" for Apple Silicon, "cpu" for CPU-only.
  - dtype: torch.float32 | torch.float16 | torch.bfloat16
      float32 is safest; float16/bfloat16 reduce memory and can be faster on GPU.
  - Model ID: "black-forest-labs/FLUX.2-klein-4B" is the default; change only if
      you switch to a different FLUX variant and update the pipeline class to match.
"""

import os

import torch
from diffusers import Flux2KleinPipeline

MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
if not token.strip():
    raise RuntimeError(
        "Set HF_TOKEN or HUGGINGFACE_TOKEN (read token from huggingface.co/settings/tokens)."
    )

pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    token=token,
)
pipe = pipe.to(device)

print("Model downloaded and loaded successfully.")
