"""Turn an image reference into a local file the CLIP model can read.

The UI can send an image as a local path, an http(s) URL (social-media posts) or a base64
data URL (uploaded files). embed_image() only understands local paths, so without this step
every uploaded or fetched image was silently ignored.
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image

from config.settings import RUNTIME_DIR

UPLOAD_DIR = RUNTIME_DIR / "uploads"
MAX_BYTES = 8 * 1024 * 1024
_EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp",
        "image/gif": ".gif", "image/bmp": ".bmp"}


def _store(data: bytes, mime: str) -> str:
    if len(data) > MAX_BYTES:
        raise ValueError(f"image is larger than {MAX_BYTES // (1024 * 1024)} MB")
    Image.open(io.BytesIO(data)).verify()          # raises if it is not a real image
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{hashlib.sha1(data).hexdigest()}{_EXT.get(mime, '.img')}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def ensure_local_image(src: str) -> Tuple[str, str]:
    """Return (local_path, note). local_path is '' when the image cannot be used; note says why."""
    src = (src or "").strip()
    if not src:
        return "", "no image supplied"
    try:
        if src.startswith("data:"):
            head, _, payload = src.partition(",")
            mime = head[5:].split(";")[0].lower()
            return _store(base64.b64decode(payload), mime), "decoded uploaded image"
        if re.match(r"^https?://", src, re.I):
            resp = requests.get(src, timeout=15, headers={"User-Agent": "Mozilla/5.0 (VeriGraph AI)"})
            resp.raise_for_status()
            mime = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if mime not in _EXT:
                return "", f"URL did not return an image (content-type {mime or 'unknown'})"
            return _store(resp.content, mime), "downloaded from URL"
        p = Path(src)
        if p.exists():
            return str(p), "local file"
        return "", f"file not found: {src[:80]}"
    except Exception as exc:
        return "", f"{type(exc).__name__}: {str(exc)[:120]}"
