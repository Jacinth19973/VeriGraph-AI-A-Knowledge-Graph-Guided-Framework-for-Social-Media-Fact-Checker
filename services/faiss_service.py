"""VeriGraph AI — FAISS retrieval service with separate multimodal indexes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Separate indexes for different modalities
_text_index = None  # BGE text index (768-D)
_image_index = None  # CLIP image index (512-D)
_text_clip_index = None  # CLIP text index (512-D)
_kb_meta = None
_image_row_ids = None
_index_initialized = False


def _ensure_faiss():
    """Ensure FAISS indexes are initialized from disk."""
    global _text_index, _image_index, _text_clip_index, _kb_meta, _image_row_ids, _index_initialized
    
    if _index_initialized:
        return
    
    try:
        import faiss
    except ImportError:
        _index_initialized = True
        return
    
    from config.settings import KB_DIR
    kb_dir = KB_DIR or Path(".")
    
    # Try to load BGE text index (768-D)
    text_index_path = kb_dir / "text_index_bge.faiss"
    image_index_path = kb_dir / "image_index_clip.faiss"
    # Existing notebook artifacts use the legacy names.  Load them explicitly
    # rather than silently replacing a populated index with an empty one.
    if not text_index_path.exists() and (kb_dir / "text_index.faiss").exists():
        text_index_path = kb_dir / "text_index.faiss"
    if not image_index_path.exists() and (kb_dir / "image_index.faiss").exists():
        image_index_path = kb_dir / "image_index.faiss"
    text_clip_path = kb_dir / "text_index_clip.faiss"
    meta_path = kb_dir / "knowledge_base_metadata.csv"
    
    loaded_any = False
    
    if text_index_path.exists():
        _text_index = faiss.read_index(str(text_index_path))
        loaded_any = True
        print(f"[FAISS] Loaded BGE text index: {text_index_path}, dim={_text_index.d}")
    else:
        print(f"[FAISS] No BGE text index found at {text_index_path}")
    
    if image_index_path.exists():
        _image_index = faiss.read_index(str(image_index_path))
        loaded_any = True
        print(f"[FAISS] Loaded CLIP image index: {image_index_path}, dim={_image_index.d}")
    else:
        print(f"[FAISS] No CLIP image index found at {image_index_path}")
    
    if text_clip_path.exists():
        _text_clip_index = faiss.read_index(str(text_clip_path))
        loaded_any = True
        print(f"[FAISS] Loaded CLIP text index: {text_clip_path}, dim={_text_clip_index.d}")
    else:
        print(f"[FAISS] No CLIP text index found at {text_clip_path}")
    
    if meta_path.exists():
        _kb_meta = pd.read_csv(str(meta_path), low_memory=False)
        loaded_any = True
    else:
        print(f"[FAISS] No metadata found at {meta_path}")
    
    image_row_path = kb_dir / "image_index_row_ids.npy"
    if image_row_path.exists():
        _image_row_ids = np.load(str(image_row_path))
    else:
        print(f"[FAISS] No image row IDs found at {image_row_path}")
    
    _index_initialized = True
    print(f"[FAISS] Initialization complete. Indexes loaded: text={_text_index is not None}, image={_image_index is not None}, clip_text={_text_clip_index is not None}")


def init_faiss_indexes(kb_dir: Path = None):
    """Initialize FAISS indexes. Creates empty indexes if none exist."""
    global _text_index, _image_index, _text_clip_index, _kb_meta, _image_row_ids, _index_initialized
    
    if _index_initialized:
        return
    
    try:
        import faiss
    except ImportError:
        _index_initialized = True
        return
    
    from config.settings import KB_DIR
    kb_dir = kb_dir or KB_DIR
    
    # Check if indexes already exist on disk
    text_index_path = kb_dir / "text_index_bge.faiss"
    image_index_path = kb_dir / "image_index_clip.faiss"
    if not text_index_path.exists() and (kb_dir / "text_index.faiss").exists():
        text_index_path = kb_dir / "text_index.faiss"
    if not image_index_path.exists() and (kb_dir / "image_index.faiss").exists():
        image_index_path = kb_dir / "image_index.faiss"
    text_clip_path = kb_dir / "text_index_clip.faiss"
    meta_path = kb_dir / "knowledge_base_metadata.csv"
    
    loaded_any = False
    
    if text_index_path.exists():
        _text_index = faiss.read_index(str(text_index_path))
        try:
            _kb_meta = pd.read_csv(str(meta_path), low_memory=False) if meta_path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            _kb_meta = pd.DataFrame()
            print(f"[FAISS] Metadata is empty: {meta_path}; historical vectors cannot be mapped to evidence.")
        loaded_any = True
        print(f"[FAISS] Loaded BGE text index: dim={_text_index.d}")
    else:
        # Create empty BGE text index (768-D, cosine similarity/IP)
        _text_index = faiss.IndexFlatIP(768)
        # Ensure metadata dir exists
        kb_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(str(meta_path), index=False) if meta_path.exists() else None
        print(f"[FAISS] Created empty BGE text index: dim=768")
    
    if image_index_path.exists():
        _image_index = faiss.read_index(str(image_index_path))
        loaded_any = True
        print(f"[FAISS] Loaded CLIP image index: dim={_image_index.d}")
    else:
        # Create empty CLIP image index (512-D, cosine similarity/IP)
        _image_index = faiss.IndexFlatIP(512)
        print(f"[FAISS] Created empty CLIP image index: dim=512")
    
    if text_clip_path.exists():
        _text_clip_index = faiss.read_index(str(text_clip_path))
        loaded_any = True
        print(f"[FAISS] Loaded CLIP text index: dim={_text_clip_index.d}")
    else:
        # Create empty CLIP text index (512-D, cosine similarity/IP)
        _text_clip_index = faiss.IndexFlatIP(512)
        print(f"[FAISS] Created empty CLIP text index: dim=512")
    
    _index_initialized = True


def retrieve_historical_text_bge(query_emb: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve using BGE text index (768-D)."""
    _ensure_faiss()
    q = query_emb.astype(np.float32).reshape(1, -1)
    
    if _text_index is None or _text_index.d != 768:
        return []
    
    scores, indices = _text_index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if _kb_meta is None:
            continue
        if idx < 0 or idx >= len(_kb_meta):
            continue
        r = _kb_meta.iloc[idx]
        results.append({
            "retrieval_channel": "persistent_text",
            "retrieval_score": float(score),
            "kb_row_index": int(idx),
            "sample_id": str(r.get("sample_id", "")),
            "dataset": str(r.get("dataset", "")),
            "split": str(r.get("split", "")),
            "claim": str(r.get("caption", r.get("claim", ""))),
            "evidence_text": str(r.get("evidence_text", "")),
            "image_path": str(r.get("image_path", "")),
            "source": str(r.get("source", "")),
            "url": str(r.get("url", "")),
            "source_domain": str(r.get("source_domain", "")),
            "publication_date": str(r.get("publication_date", "")),
        })
    return results


def retrieve_historical_image_clip(query_emb: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve using CLIP image index (512-D)."""
    _ensure_faiss()
    q = query_emb.astype(np.float32).reshape(1, -1)
    
    if _image_index is None or _image_index.d != 512:
        return []
    
    scores, local_indices = _image_index.search(q, top_k)
    results = []
    for score, local_idx in zip(scores[0], local_indices[0]):
        if _image_row_ids is None:
            continue
        if local_idx < 0 or local_idx >= len(_image_row_ids):
            continue
        global_idx = int(_image_row_ids[local_idx])
        if _kb_meta is not None and (global_idx < 0 or global_idx >= len(_kb_meta)):
            continue
        r = _kb_meta.iloc[global_idx] if _kb_meta is not None else pd.Series()
        results.append({
            "retrieval_channel": "persistent_image",
            "retrieval_score": float(score),
            "kb_row_index": global_idx,
            "sample_id": str(r.get("sample_id", "")),
            "dataset": str(r.get("dataset", "")),
            "split": str(r.get("split", "")),
            "claim": str(r.get("caption", r.get("claim", ""))),
            "evidence_text": str(r.get("evidence_text", "")),
            "image_path": str(r.get("image_path", "")),
            "source": str(r.get("source", "")),
            "url": str(r.get("url", "")),
            "source_domain": str(r.get("source_domain", "")),
            "publication_date": str(r.get("publication_date", "")),
        })
    return results


def retrieve_historical_text_clip(query_emb: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve using CLIP text index (512-D)."""
    _ensure_faiss()
    q = query_emb.astype(np.float32).reshape(1, -1)
    
    if _text_clip_index is None or _text_clip_index.d != 512:
        return []
    
    scores, indices = _text_clip_index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if _kb_meta is None:
            continue
        if idx < 0 or idx >= len(_kb_meta):
            continue
        r = _kb_meta.iloc[idx]
        results.append({
            "retrieval_channel": "persistent_clip_text",
            "retrieval_score": float(score),
            "kb_row_index": int(idx),
            "sample_id": str(r.get("sample_id", "")),
            "dataset": str(r.get("dataset", "")),
            "split": str(r.get("split", "")),
            "claim": str(r.get("caption", r.get("claim", ""))),
            "evidence_text": str(r.get("evidence_text", "")),
            "image_path": str(r.get("image_path", "")),
            "source": str(r.get("source", "")),
            "url": str(r.get("url", "")),
            "source_domain": str(r.get("source_domain", "")),
            "publication_date": str(r.get("publication_date", "")),
        })
    return results


def get_faiss_index_info() -> Dict[str, Any]:
    """Return FAISS index metadata for health/api endpoints."""
    info = {
        "bge_text": {"dimension": 768, "compatible": False, "vectors": 0},
        "clip_image": {"dimension": 512, "compatible": False, "vectors": 0},
        "clip_text": {"dimension": 512, "compatible": False, "vectors": 0},
    }
    
    if _text_index is not None:
        info["bge_text"] = {
            "dimension": int(_text_index.d),
            "compatible": True,
            "vectors": int(_text_index.ntotal),
        }
    
    if _image_index is not None:
        info["clip_image"] = {
            "dimension": int(_image_index.d),
            "compatible": True,
            "vectors": int(_image_index.ntotal),
        }
    
    if _text_clip_index is not None:
        info["clip_text"] = {
            "dimension": int(_text_clip_index.d),
            "compatible": True,
            "vectors": int(_text_clip_index.ntotal),
        }
    
    return info
