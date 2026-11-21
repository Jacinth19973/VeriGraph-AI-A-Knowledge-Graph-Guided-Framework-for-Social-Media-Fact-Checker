"""Evidence API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from storage.case_store import load_case

router = APIRouter()


class EvidenceItemResponse(BaseModel):
    evidence_id: str
    evidence_type: str
    retrieval_channel: str
    provider: str
    title: str
    url: str
    snippet: str
    domain: str
    source: str
    published_at: str
    relevance_score: float
    credibility_score: float
    fusion_score: float
    cluster_id: int
    provenance_type: str
    url_status: str
    display_url: str
    is_clickable: bool
    display_text: str
    sample_id: str
    dataset: str


class EvidenceListResponse(BaseModel):
    evidence: List[EvidenceItemResponse]
    total: int
    channels: Dict[str, int]
    providers: Dict[str, int]


@router.get("/{case_id}", response_model=EvidenceListResponse)
async def get_evidence(
    case_id: str,
    channel: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    min_relevance: float = Query(0.0, ge=0.0, le=1.0),
    min_credibility: float = Query(0.0, ge=0.0, le=1.0),
    min_fusion: float = Query(0.0, ge=0.0, le=1.0),
    cluster_id: Optional[int] = Query(None),
    provenance_type: Optional[str] = Query(None),
):
    """Get evidence for a case with filtering."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.retrieval:
        raise HTTPException(status_code=404, detail="Case not found or no evidence")
    
    fused = case.retrieval.fused_evidence or []
    
    # Apply filters
    filtered = fused
    if channel:
        filtered = [e for e in filtered if e.get("retrieval_channel") == channel]
    if provider:
        filtered = [e for e in filtered if e.get("provider") == provider]
    if min_relevance > 0:
        filtered = [e for e in filtered if e.get("relevance_score", 0) >= min_relevance]
    if min_credibility > 0:
        filtered = [e for e in filtered if e.get("credibility_score", 0) >= min_credibility]
    if min_fusion > 0:
        filtered = [e for e in filtered if e.get("fusion_score", 0) >= min_fusion]
    if cluster_id is not None:
        filtered = [e for e in filtered if e.get("cluster_id") == cluster_id]
    if provenance_type:
        filtered = [e for e in filtered if e.get("provenance_type") == provenance_type]
    
    # Convert to response format
    evidence_items = []
    for e in filtered:
        provenance = e.get("provenance", {})
        display = e.get("display", {})
        evidence_items.append(EvidenceItemResponse(
            evidence_id=e.get("evidence_id", ""),
            evidence_type=e.get("evidence_type", ""),
            retrieval_channel=e.get("retrieval_channel", ""),
            provider=e.get("provider", ""),
            title=e.get("title", ""),
            url=e.get("url", ""),
            snippet=e.get("snippet", ""),
            domain=e.get("domain", ""),
            source=e.get("source", ""),
            published_at=e.get("published_at", ""),
            relevance_score=e.get("relevance_score", 0.0),
            credibility_score=e.get("credibility_score", 0.0),
            fusion_score=e.get("fusion_score", 0.0),
            cluster_id=e.get("cluster_id", -1),
            provenance_type=e.get("provenance_type", "UNRESOLVED"),
            url_status=e.get("url_status", "not_available"),
            display_url=display.get("display_url", ""),
            is_clickable=display.get("is_clickable", False),
            display_text=display.get("display_text", ""),
            sample_id=e.get("sample_id", ""),
            dataset=e.get("dataset", ""),
        ))
    
    # Channel and provider stats
    channels = {}
    providers = {}
    for e in fused:
        ch = e.get("retrieval_channel", "unknown")
        pr = e.get("provider", "unknown")
        channels[ch] = channels.get(ch, 0) + 1
        providers[pr] = providers.get(pr, 0) + 1
    
    return EvidenceListResponse(
        evidence=evidence_items,
        total=len(evidence_items),
        channels=channels,
        providers=providers,
    )


@router.get("/{case_id}/{evidence_id}")
async def get_evidence_detail(case_id: str, evidence_id: str):
    """Get detailed evidence item."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.retrieval:
        raise HTTPException(status_code=404, detail="Case not found")
    
    for e in case.retrieval.fused_evidence or []:
        if e.get("evidence_id") == evidence_id:
            return e
    
    raise HTTPException(status_code=404, detail="Evidence not found")


@router.get("/{case_id}/stats/summary")
async def get_evidence_summary(case_id: str):
    """Get evidence statistics summary."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.retrieval:
        raise HTTPException(status_code=404, detail="Case not found")
    
    fused = case.retrieval.fused_evidence or []
    
    return {
        "total": len(fused),
        "by_type": _count_by(fused, "evidence_type"),
        "by_channel": _count_by(fused, "retrieval_channel"),
        "by_provider": _count_by(fused, "provider"),
        "by_provenance": _count_by(fused, "provenance_type"),
        "by_domain": _count_by(fused, "domain"),
        "avg_relevance": _avg(fused, "relevance_score"),
        "avg_credibility": _avg(fused, "credibility_score"),
        "avg_fusion": _avg(fused, "fusion_score"),
        "clusters": _count_by(fused, "cluster_id"),
        "clickable_urls": sum(1 for e in fused if e.get("display", {}).get("is_clickable", False)),
    }


def _count_by(items: List[Dict], key: str) -> Dict[str, int]:
    counts = {}
    for item in items:
        val = item.get(key, "unknown")
        counts[str(val)] = counts.get(str(val), 0) + 1
    return counts


def _avg(items: List[Dict], key: str) -> float:
    vals = [item.get(key, 0) for item in items if isinstance(item.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0