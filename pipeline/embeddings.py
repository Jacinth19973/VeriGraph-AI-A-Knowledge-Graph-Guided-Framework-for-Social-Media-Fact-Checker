"""VeriGraph AI — Embedding generation (BGE-base + CLIP)."""
from __future__ import annotations
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import torch

# Singleton model caches
_text_model = None
_clip_model = None
_clip_processor = None
_clip_text_model = None
_clip_text_processor = None
_device = None


def _get_device():
    # Check CUDA every call (handles GPU toggling at runtime)
    return "cuda" if torch.cuda.is_available() else "cpu"


def report_device() -> str:
    dev = _get_device()
    if dev == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)} @ {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM"
    return "CPU (no CUDA available)"


def _load_bge_model():
    """Load BGE text model once (singleton)."""
    global _text_model
    if _text_model is not None:
        return _text_model
    import sentence_transformers
    from config.settings import TEXT_MODEL_NAME, TEXT_MODEL_PATH
    device = _get_device()
    model_source = str(TEXT_MODEL_PATH) if TEXT_MODEL_PATH.exists() else TEXT_MODEL_NAME
    _text_model = sentence_transformers.SentenceTransformer(model_source, device=device)
    return _text_model


def _load_clip_model():
    """Load CLIP image model once (singleton)."""
    global _clip_model, _clip_processor
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor
    from transformers import CLIPModel, CLIPProcessor
    device = _get_device()
    from config.settings import CLIP_MODEL_NAME, CLIP_MODEL_PATH
    model_source = str(CLIP_MODEL_PATH) if CLIP_MODEL_PATH.exists() else CLIP_MODEL_NAME
    _clip_model = CLIPModel.from_pretrained(model_source).to(device)
    _clip_processor = CLIPProcessor.from_pretrained(model_source)
    _clip_model.eval()
    return _clip_model, _clip_processor


def _load_clip_text_model():
    """Load CLIP text model once (singleton)."""
    global _clip_text_model, _clip_text_processor
    if _clip_text_model is not None and _clip_text_processor is not None:
        return _clip_text_model, _clip_text_processor
    _clip_text_model, _clip_text_processor = _load_clip_model()   # same CLIP checkpoint serves text and image
    return _clip_text_model, _clip_text_processor


def embed_text(text: str) -> np.ndarray:
    """Generate BGE text embedding.

    Output: 768-dimensional L2-normalized float32 vector.
    """
    model = _load_bge_model()
    emb = model.encode([text], normalize_embeddings=True,
                       convert_to_numpy=True)
    return emb[0].astype(np.float32)


def embed_image(image) -> Optional[np.ndarray]:
    """Generate CLIP image embedding.

    Input: PIL image / valid image path
    Output: 512-dimensional L2-normalized float32 vector.
    """
    if isinstance(image, str):
        path = Path(image)
        if not path.exists():
            return None
        image = Image.open(str(path)).convert("RGB")
    elif isinstance(image, Image.Image):
        pass
    else:
        return None

    model, processor = _load_clip_model()
    device = _get_device()
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        if hasattr(out, "pooler_output"):
            features = out.pooler_output
        else:
            features = out.last_hidden_state[:, 0]
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.cpu().numpy()[0].astype(np.float32)


def embed_clip_text(text: str) -> np.ndarray:
    """Generate CLIP text embedding.

    Output: 512-dimensional L2-normalized float32 vector.
    This is separate from BGE text embedding (768-D).
    CLIP text embedding is designed for cross-modal similarity
    (claim text <-> image, evidence text <-> image).
    """
    model, processor = _load_clip_text_model()
    device = _get_device()
    inputs = processor(text=[text], return_tensors="pt", padding=True,
                       truncation=True).to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.cpu().numpy()[0].astype(np.float32)


def get_embedding_metadata() -> dict:
    """Return embedding model metadata for health/api endpoints."""
    return {
        "bge": {
            "model": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "normalized": True,
        },
        "clip_image": {
            "model": "openai/clip-vit-base-patch32",
            "dimension": 512,
            "normalized": True,
        },
        "clip_text": {
            "model": "openai/clip-vit-base-patch32",
            "dimension": 512,
            "normalized": True,
        },
    }
