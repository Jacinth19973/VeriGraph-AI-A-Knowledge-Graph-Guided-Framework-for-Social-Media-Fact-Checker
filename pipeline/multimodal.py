"""Image-vs-claim consistency and visual filtering.

Two independent signals decide whether an attached image belongs to the claim:

1. CLIP cross-modal similarity between the image and the claim text (and the titles of the best
   evidence). Thresholds come from IMAGE_MATCH_SIMILARITY / IMAGE_MISMATCH_SIMILARITY; check yours with
   `python scripts/check_image.py <image> "<claim>"`.
2. Reverse image search: the pages that host the same image. If they are about the claimed event the image
   is in context; if they are about something else (or the same event on another date/place) it is out of context.

A mismatch turns a "True" text verdict into "Misleading" (a true claim with an unrelated image).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

from services.claim_conflicts import find_conflicts

MATCH_SIM = float(os.getenv("IMAGE_MATCH_SIMILARITY", "0.22"))
MISMATCH_SIM = float(os.getenv("IMAGE_MISMATCH_SIMILARITY", "0.17"))
SEMANTIC_RELEVANT = float(os.getenv("IMAGE_CONTEXT_RELEVANCE", "0.62"))


def _clip_check(claim: str, local_image: str, evidence_items: List[Dict]) -> Dict[str, Any]:
    if not local_image:
        return {"status": "unavailable", "reason": "image file not available"}
    try:
        from pipeline.embeddings import embed_image, embed_clip_text
        img = embed_image(local_image)
        if img is None:
            return {"status": "unavailable", "reason": "image could not be embedded"}
        texts = {"claim": claim}
        top = [e for e in evidence_items if e.get("evidence_type") != "image"][:3]
        for e in top:
            texts[e.get("evidence_id", "evidence")] = str(e.get("title", ""))
        sims = {k: round(float(np.dot(img, embed_clip_text(v))), 4) for k, v in texts.items() if v}
        best = max(sims.values())
        status = "match" if best >= MATCH_SIM else "mismatch" if best < MISMATCH_SIM else "weak"
        return {"status": status, "claim_similarity": sims.get("claim"), "best_similarity": best,
                "similarities": sims, "match_threshold": MATCH_SIM, "mismatch_threshold": MISMATCH_SIM}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {str(exc)[:120]}"}


def _is_relevant(claim: str, text: str) -> bool:
    try:
        from pipeline.narrative import _specific_anchor_ok
        if _specific_anchor_ok(claim, text):
            return True
    except Exception:
        pass
    try:
        from pipeline.embeddings import embed_text
        a, b = embed_text(claim), embed_text(text)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) or 1.0)) >= SEMANTIC_RELEVANT
    except Exception:
        return False


def _reverse_check(claim: str, reverse: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    reverse = reverse or {}
    status = reverse.get("status", "skipped")
    if status != "success":
        return {"status": status, "reason": reverse.get("error", ""), "result_count": 0, "rows": []}
    rows = []
    for r in reverse.get("results", [])[:10]:
        text = f"{r.get('title', '')} {r.get('snippet', '')}"
        relevant = _is_relevant(claim, text)
        conflicts = find_conflicts(claim, text) if relevant else []
        rows.append({"rank": r.get("rank"), "title": r.get("title"), "url": r.get("url"), "domain": r.get("domain"),
                     "relevant": relevant, "conflicts": [c["detail"] for c in conflicts],
                     "evidence_id": r.get("evidence_id")})
    ok = [x for x in rows if x["relevant"] and not x["conflicts"]]
    bad = [x for x in rows if x["relevant"] and x["conflicts"]]
    if ok:
        verdict = "context_match"
    elif bad:
        verdict = "context_conflict"
    elif len(rows) >= 3:
        verdict = "no_relevant_context"
    else:
        verdict = "inconclusive"
    years = [int(y) for x in reverse.get("results", []) for y in re.findall(r"\b(?:19|20)\d{2}\b", f"{x.get('title', '')} {x.get('snippet', '')} {x.get('date', '')}")]
    return {"status": "success", "context": verdict, "result_count": len(rows), "relevant_count": len(ok),
            "conflict_count": len(bad), "earliest_year": min(years) if years else None, "rows": rows}


def analyze_image_context(claim: str, local_image: str, evidence_items: List[Dict], all_items: List[Dict],
                          reverse: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide whether the image belongs to the claim, and mark the evidence with visual-filter flags."""
    clip = _clip_check(claim, local_image, evidence_items)
    rev = _reverse_check(claim, reverse)
    reasons: List[str] = []

    clip_s, ctx = clip.get("status"), rev.get("context")
    if clip_s == "match" or ctx == "context_match":
        consistency = "consistent"
        if ctx == "context_match":
            reasons.append(f"{rev['relevant_count']} web page(s) show this image in the context of the claim")
        if clip_s == "match":
            reasons.append(f"CLIP image-text similarity {clip['best_similarity']} is above {MATCH_SIM}")
    elif ctx == "context_conflict":
        consistency = "mismatch"
        reasons.append("the image appears on pages about the same event but with a different date or place: "
                       + "; ".join(sorted({c for x in rev["rows"] for c in x["conflicts"]})[:2]))
    elif clip_s == "mismatch" or (ctx == "no_relevant_context" and clip_s in ("weak", "unavailable", "mismatch")):
        consistency = "mismatch"
        if clip_s == "mismatch":
            reasons.append(f"CLIP image-text similarity {clip['best_similarity']} is below {MISMATCH_SIM} (the image does not depict the claim)")
        if ctx == "no_relevant_context":
            reasons.append(f"reverse image search found {rev['result_count']} pages with this image and none concern the claim")
    elif clip_s in ("unavailable", None) and ctx in ("not_public", "unconfigured", "skipped", "empty", "inconclusive", None):
        consistency = "unavailable"
        reasons.append("the image could not be checked (no CLIP result and no usable reverse-image context)")
    else:
        consistency = "uncertain"
        reasons.append("the image-text similarity is weak and reverse image search was inconclusive")

    # Visual filtering of the evidence pool
    kept = dropped = 0
    row_by_id = {x.get("evidence_id"): x for x in rev.get("rows", [])}
    for e in all_items:
        if e.get("evidence_type") != "image":
            continue
        row = row_by_id.get(e.get("evidence_id"))
        in_context = bool(row and row["relevant"] and not row["conflicts"])
        e["visual_filter_status"] = "VISUAL_MATCH" if in_context else "VISUAL_OUT_OF_CONTEXT"
        e["out_of_context_flag"] = not in_context
        e["multimodal_score"] = round(float(e.get("visual_similarity_score", 0.5)) * (1.0 if in_context else 0.0), 3)
        kept, dropped = kept + int(in_context), dropped + int(not in_context)

    return {"status": "analyzed", "consistency": consistency, "reasons": reasons, "clip": clip,
            "reverse_image": {k: v for k, v in rev.items() if k != "rows"} | {"pages": rev.get("rows", [])},
            "visual_filter": {"in_context": kept, "out_of_context": dropped}}


def apply_image_verdict(case) -> bool:
    """A true text claim with an image that does not belong to it is Misleading. Returns True if changed."""
    mm = getattr(case, "multimodal", None) or {}
    consistency = mm.get("consistency")
    if consistency not in ("mismatch", "uncertain"):
        return False
    j = case.judge
    note = ("The attached image does not match the claim: " if consistency == "mismatch"
            else "The attached image could not be confirmed to match the claim: ") + "; ".join(mm.get("reasons", [])[:2])
    j.dissent_notes = f"{j.dissent_notes} {note}".strip()
    if consistency == "mismatch" and j.final_verdict == "True":
        j.final_verdict = "Misleading"
        j.confidence = round(min(float(j.confidence), 0.8), 2)
        j.reasoning = (f"{j.reasoning} [Image check] {note}. The text claim is supported by the evidence, but the image "
                       f"appears out of context, so the verdict was changed from True to Misleading.")
        j.label_reasoning_alignment = {**(j.label_reasoning_alignment or {}), "adjusted_by": "image_check",
                                       "model_verdict": "True", "aligned": True}
        return True
    return False


def apply_image_to_explanation(case) -> None:
    mm = getattr(case, "multimodal", None) or {}
    if mm.get("status") != "analyzed":
        return
    x = case.explanation
    best = (mm.get("clip") or {}).get("best_similarity")
    x.factor_cards = list(x.factor_cards) + [{
        "factor": "Image-Claim Consistency", "assessment": mm.get("consistency", "unavailable"),
        "confidence": float(best) if isinstance(best, (int, float)) else 0.0, "status": "completed",
        "detail": "; ".join(mm.get("reasons", [])),
    }]
    if mm.get("consistency") == "mismatch" and case.judge.final_verdict == "Misleading":
        x.summary = ("The text of the claim is supported by the evidence, but the attached image does not match it "
                     "(it appears to be out of context), so the post is misleading.")
        x.natural_language_summary = x.summary
        x.verdict_reason = case.judge.reasoning
