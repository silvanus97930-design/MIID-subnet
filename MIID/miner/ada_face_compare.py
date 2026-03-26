#!/usr/bin/env python3
"""
Face identity comparison using AdaFace.

This module is used by the miner to validate that generated image variations
preserve the identity of the original face. It loads a pretrained AdaFace model
(ir_50 on MS1MV2), extracts normalized face embeddings via MTCNN alignment,
and compares embeddings using cosine similarity.
"""

import importlib.util
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

# Check for required dependencies
import cv2  # noqa: F401

# Add AdaFace to path (relative to this file)
_this_dir = os.path.dirname(os.path.abspath(__file__))
ADA_FACE_PATH = os.path.join(_this_dir, "AdaFace")
ADA_FACE_ALIGNMENT_PATH = os.path.join(ADA_FACE_PATH, "face_alignment")

# Important: add face_alignment module directory first, otherwise the external
# "face_alignment" pip package can shadow AdaFace's local modules.
if os.path.isdir(ADA_FACE_ALIGNMENT_PATH):
    sys.path.insert(0, ADA_FACE_ALIGNMENT_PATH)
if os.path.isdir(ADA_FACE_PATH):
    sys.path.insert(0, ADA_FACE_PATH)

_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE = {}
_MTCNN_LOCK = threading.Lock()


def _resolve_device(device: str | None = None) -> str:
    requested = (device or os.environ.get("ADAFACE_DEVICE", "cpu")).strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return requested


def _load_local_module(module_name: str, file_path: str):
    """Load a module directly from a local file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    # Force local AdaFace face_alignment modules to avoid collision with the
    # pip package named "face_alignment".
    mtcnn = _load_local_module("adaface_local_mtcnn", os.path.join(ADA_FACE_ALIGNMENT_PATH, "mtcnn.py"))

    # AdaFace align.py instantiates MTCNN with device='cuda:0' at import time.
    # On CPU-only hosts we remap that to 'cpu' before loading align.py.
    original_mtcnn_class = mtcnn.MTCNN

    class SafeMTCNN(original_mtcnn_class):
        def __init__(self, *args, **kwargs):
            req_device = kwargs.get("device")
            if isinstance(req_device, str) and req_device.startswith("cuda") and not torch.cuda.is_available():
                kwargs["device"] = "cpu"
            super().__init__(*args, **kwargs)

    mtcnn.MTCNN = SafeMTCNN

    face_alignment_pkg = types.ModuleType("face_alignment")
    face_alignment_pkg.__path__ = [ADA_FACE_ALIGNMENT_PATH]
    face_alignment_pkg.mtcnn = mtcnn
    sys.modules["face_alignment"] = face_alignment_pkg
    sys.modules["face_alignment.mtcnn"] = mtcnn

    align = _load_local_module("adaface_local_align", os.path.join(ADA_FACE_ALIGNMENT_PATH, "align.py"))
    face_alignment_pkg.align = align
    sys.modules["face_alignment.align"] = align

    from inference import load_pretrained_model, to_input

    # Initialize with a safe default.
    align_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    align.mtcnn_model = mtcnn.MTCNN(device=align_device, crop_size=(112, 112))

except ImportError as e:
    raise ImportError(
        f"AdaFace modules could not be imported: {e}. "
        f"Ensure AdaFace is cloned at {ADA_FACE_PATH} and local modules are available."
    ) from e


def _ensure_mtcnn_model(resolved_device: str) -> None:
    """Ensure align.mtcnn_model matches requested device."""
    target_device = "cuda:0" if resolved_device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    with _MTCNN_LOCK:
        current = getattr(align, "mtcnn_model", None)
        current_device = str(getattr(current, "device", "")) if current is not None else ""
        if current is None or current_device != target_device:
            align.mtcnn_model = mtcnn.MTCNN(device=target_device, crop_size=(112, 112))


def _to_rgb_pil_image(image_input: Any) -> Image.Image:
    """Convert path/PIL input into an RGB PIL image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image path not found: {image_input}")
        return Image.open(image_input).convert("RGB")

    raise TypeError("Image input must be a file path (str) or PIL Image")


def load_adaface_model(architecture='ir_50', model_path=None, device='cpu'):
    """
    Load AdaFace model for face recognition.

    Args:
        architecture: Model architecture ('ir_50', 'ir_101', etc.)
        model_path: Optional path to pretrained model. If None, uses default from inference.py
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded AdaFace model
    """
    import inference

    resolved_device = _resolve_device(device)

    # Set default model path to absolute path if not provided.
    if model_path is None:
        model_path = os.path.join(ADA_FACE_PATH, "pretrained", "adaface_ir50_ms1mv2.ckpt")

    # Convert to absolute path if it's relative.
    if not os.path.isabs(model_path):
        model_path = os.path.join(ADA_FACE_PATH, model_path)

    model_path = os.path.abspath(model_path)
    cache_key = (architecture, model_path, resolved_device)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Update the model path in inference module.
    inference.adaface_models[architecture] = model_path

    # Verify file exists.
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please download it from: https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing"
        )

    print(f"Loading AdaFace model: {architecture}")
    print(f"Model path: {model_path}")

    original_load = torch.load

    def cpu_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = 'cpu'
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False  # Required for PyTorch 2.6+ with checkpoint files
        return original_load(*args, **kwargs)

    # Monkey patch torch.load for this call.
    torch.load = cpu_load

    try:
        model = load_pretrained_model(architecture)
    finally:
        torch.load = original_load

    model.eval()

    # Move model to target device.
    if resolved_device.startswith('cuda') and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    _ensure_mtcnn_model(resolved_device)

    print("✓ Model loaded successfully")

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = model

    return model


def _extract_aligned_face(image_input: Any, device: str) -> Optional[Image.Image]:
    """Align one face from input image using AdaFace MTCNN."""
    try:
        rgb_image = _to_rgb_pil_image(image_input)
        _ensure_mtcnn_model(device)
        aligned_rgb_img = align.get_aligned_face("", rgb_pil_image=rgb_image)
        if aligned_rgb_img is None:
            return None
        return aligned_rgb_img
    except Exception:
        return None


def extract_face_embeddings(
    model,
    images: Iterable[Any],
    device: str = 'cpu',
    parallel_workers: Optional[int] = None,
) -> List[Optional[torch.Tensor]]:
    """
    Extract normalized AdaFace embeddings for a list of images.

    - Alignment runs in-memory (no temporary files).
    - Model inference is batched in a single forward pass when possible.
    """
    resolved_device = _resolve_device(device)
    image_list = list(images)
    if not image_list:
        return []

    if resolved_device.startswith('cuda') and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    workers_env = int(os.environ.get("ADAFACE_ALIGN_WORKERS", "0") or 0)
    workers = int(parallel_workers if parallel_workers is not None else workers_env)
    workers = max(0, workers)

    if workers > 1 and len(image_list) > 1:
        max_workers = min(workers, len(image_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            aligned_faces = list(executor.map(lambda img: _extract_aligned_face(img, resolved_device), image_list))
    else:
        aligned_faces = [_extract_aligned_face(img, resolved_device) for img in image_list]

    tensors: List[torch.Tensor] = []
    indices: List[int] = []

    for idx, aligned in enumerate(aligned_faces):
        if aligned is None:
            continue
        try:
            tensors.append(to_input(aligned))
            indices.append(idx)
        except Exception:
            continue

    outputs: List[Optional[torch.Tensor]] = [None] * len(image_list)
    if not tensors:
        return outputs

    batch = torch.cat(tensors, dim=0)

    if resolved_device.startswith('cuda') and torch.cuda.is_available():
        batch = batch.cuda(non_blocking=True)

    try:
        with torch.no_grad():
            features, _ = model(batch)
        features = features / torch.norm(features, 2, dim=1, keepdim=True)
        features = features.detach().cpu()
    except Exception:
        return outputs

    for row, idx in enumerate(indices):
        outputs[idx] = features[row:row + 1]

    return outputs


def extract_face_embedding(model, image_input, device='cpu'):
    """
    Extract face embedding from one image using AdaFace.

    Args:
        model: AdaFace model
        image_input: Path to image file or PIL image
        device: Device to run inference on ('cpu' or 'cuda')

    Returns:
        Face embedding tensor (normalized) or None if face detection fails
    """
    embeddings = extract_face_embeddings(model, [image_input], device=device, parallel_workers=0)
    return embeddings[0] if embeddings else None


def compute_cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor

    Returns:
        Cosine similarity score (0-1, where 1 is identical)
    """
    if embedding1 is None or embedding2 is None:
        return None

    similarity = torch.mm(embedding1, embedding2.t()).item()
    similarity = max(0.0, similarity)
    return similarity


def score_variation_candidates(
    base_image,
    variation_images: List[Any],
    model=None,
    device='cpu',
    parallel_workers: Optional[int] = None,
    base_embedding: Optional[torch.Tensor] = None,
) -> List[Optional[float]]:
    """Score variation candidates against the base image using AdaFace cosine similarity."""
    resolved_device = _resolve_device(device)

    if model is None:
        model = load_adaface_model(device=resolved_device)

    if base_embedding is None:
        base_embedding = extract_face_embedding(model, base_image, device=resolved_device)

    if base_embedding is None:
        return [None] * len(variation_images)

    var_embeddings = extract_face_embeddings(
        model,
        variation_images,
        device=resolved_device,
        parallel_workers=parallel_workers,
    )

    scores: List[Optional[float]] = []
    for emb in var_embeddings:
        score = compute_cosine_similarity(base_embedding, emb)
        scores.append(float(score) if score is not None else None)

    return scores


def validate_single_variation(base_image, variation_image, model=None, min_similarity=0.7, device='cpu'):
    """
    Validate that a single variation image preserves the identity of the base face.

    Args:
        base_image: Original face image as file path (str) or PIL Image
        variation_image: Variation image as file path (str) or PIL Image
        model: AdaFace model (if None, will be loaded)
        min_similarity: Minimum cosine similarity to consider identity preserved (default 0.7)
        device: Device to run inference on ('cpu' or 'cuda')

    Returns:
        True if similarity >= min_similarity, False otherwise.
    """
    resolved_device = _resolve_device(device)
    scores = score_variation_candidates(
        base_image,
        [variation_image],
        model=model,
        device=resolved_device,
        parallel_workers=0,
    )
    similarity = scores[0] if scores else None
    return bool(similarity is not None and similarity >= min_similarity)


def compare_faces(original_image_path, variation_image_paths, model=None, device='cpu'):
    """
    Compare original face with variation faces.

    Args:
        original_image_path: Path to original face image or PIL image
        variation_image_paths: List of paths/PIL images for variation images
        model: AdaFace model (if None, will be loaded)
        device: Device to run inference on

    Returns:
        Dictionary mapping variation input identifiers to similarity scores
    """
    resolved_device = _resolve_device(device)

    if model is None:
        model = load_adaface_model(device=resolved_device)

    scores = score_variation_candidates(
        original_image_path,
        list(variation_image_paths),
        model=model,
        device=resolved_device,
    )

    results = {}
    for variation_input, score in zip(variation_image_paths, scores):
        key = variation_input if isinstance(variation_input, str) else repr(variation_input)
        results[key] = score

    return results
