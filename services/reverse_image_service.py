"""Reverse image search through SerpApi's `google_reverse_image` engine.

Question answered: where else on the web does this image appear, and in what context?
(The contextual-narratives fact-checking method uses this to catch images reused out of context.)

Only a public http(s) image URL can be searched. Uploaded files and local paths cannot, because
SerpApi/Google must be able to download the image. Every outcome keeps its own status
(success, empty, not_public, unconfigured, timeout, http_error, ssl_error, exception) so a provider
that found nothing is never confused with one that failed.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from config.settings import SERPAPI_KEY
from services.live_search_service import _get
from services.source_credibility import classify_source


def reverse_image_search(image_url: str, limit: int = 10) -> Dict[str, Any]:
    base: Dict[str, Any] = {"provider": "serpapi_reverse_image", "query": image_url if len(image_url or "") < 300 else "",
                            "results": [], "result_count": 0, "status": "exception", "error": ""}
    if not re.match(r"^https?://", (image_url or ""), re.I):
        return {**base, "status": "not_public",
                "error": "Only a public image URL can be reverse-searched; uploads and local files cannot."}
    if not SERPAPI_KEY:
        return {**base, "status": "unconfigured", "error": "SERPAPI_KEY is not set"}

    res = _get("https://serpapi.com/search.json",
               {"engine": "google_reverse_image", "image_url": image_url, "api_key": SERPAPI_KEY})
    if res["status"] != "success":
        return {**base, "status": res["status"], "error": res.get("error", "")}

    data = res["data"] or {}
    err = str(data.get("error") or "")
    if err:
        empty = "hasn't returned any results" in err.lower() or "no results" in err.lower()
        return {**base, "status": "empty" if empty else "exception", "error": err}

    results: List[Dict[str, Any]] = []
    for rank, r in enumerate((data.get("image_results") or [])[:limit], start=1):
        link = r.get("link") or ""
        host = (urlparse(link).hostname or "").lower()
        results.append({
            "rank": rank, "title": (r.get("title") or "")[:200], "url": link,
            "snippet": (r.get("snippet") or "")[:300], "domain": host[4:] if host.startswith("www.") else host,
            "source": r.get("source") or r.get("displayed_link") or "", "date": r.get("date") or "",
        })
    return {**base, "status": "success" if results else "empty", "results": results, "result_count": len(results)}


def reverse_results_to_evidence(results: List[Dict[str, Any]], case_id: str) -> List[Dict[str, Any]]:
    """Turn reverse-search pages into evidence items (they then pass the same relevance gate as any evidence)."""
    items = []
    for r in results:
        cred = classify_source(r.get("domain", ""), r.get("url", ""))
        rank = int(r.get("rank", 1) or 1)
        items.append({
            "evidence_id": f"ev_{case_id}_img{rank:02d}", "evidence_type": "image",
            "retrieval_channel": "reverse_image", "provider": "serpapi_reverse_image",
            "title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", ""),
            "domain": r.get("domain", ""), "source": r.get("source", ""),
            "published_at": r.get("date") or "date not specified",
            "relevance_score": 0.5, "credibility_score": cred["score"], "credibility_reason": cred["reason"],
            "credibility_tier": cred["tier"], "claim_relevance_score": 0.5, "directness_score": 0.5,
            "entity_match_score": 0.5, "event_match_score": 1.0, "temporal_match_score": 0.5,
            "location_match_score": 0.0, "narrative_coherence_score": 0.5, "intent_match_score": 0.5,
            "final_score": round(max(0.3, 0.7 - 0.03 * rank), 2), "fusion_score": 0.3,
            "visual_similarity_score": round(max(0.3, 1.0 - 0.07 * (rank - 1)), 2),
            "visual_filter_status": "PENDING", "out_of_context_flag": False, "multimodal_score": 0.0,
            "provenance": {"url": r.get("url", ""), "domain": r.get("domain", ""), "provenance_type": "REVERSE_IMAGE_MATCH"},
            "provenance_type": "REVERSE_IMAGE_MATCH", "query": "reverse image search",
        })
    return items
