"""Check how well an image matches a claim, to calibrate the image thresholds.

    python scripts/check_image.py <image path or URL> "<claim text>" ["another caption" ...]

Prints the CLIP image-text cosine similarity for each text and which status it gives with the current
IMAGE_MATCH_SIMILARITY / IMAGE_MISMATCH_SIMILARITY. Try one image that belongs to a claim and one that does
not, then set the two thresholds in .env between the values you see.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.image_fetch import ensure_local_image          # noqa: E402
from pipeline.multimodal import MATCH_SIM, MISMATCH_SIM      # noqa: E402

if len(sys.argv) < 3:
    sys.exit(__doc__)
local, note = ensure_local_image(sys.argv[1])
if not local:
    sys.exit(f"Could not use the image: {note}")
from pipeline.embeddings import embed_image, embed_clip_text  # noqa: E402

img = embed_image(local)
print(f"image: {local} ({note})   match >= {MATCH_SIM}   mismatch < {MISMATCH_SIM}")
for text in sys.argv[2:]:
    sim = float(np.dot(img, embed_clip_text(text)))
    status = "match" if sim >= MATCH_SIM else "mismatch" if sim < MISMATCH_SIM else "weak"
    print(f"  {sim:.3f}  {status:8s}  {text[:90]}")
