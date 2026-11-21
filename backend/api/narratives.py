"""Narratives API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from storage.case_store import load_case

router = APIRouter()


class NarrativeClusterResponse(BaseModel):
    cluster_id: int
    size: int
    theme_keywords: List[str]
    theme: str
    source_count: int
    unique_source_domains: List[str]
    mean_cluster_coherence: float
    mean_credibility: float
    mean_relevance: float
    evidence_sufficiency: str
    support_signal_count: int
    conflict_signal_count: int
    neutral_signal_count: int = 0
    mixed_signal_count: int = 0
    support_ratio: Optional[float] = None
    conflict_ratio: Optional[float] = None
    representative_evidence_ids: List[str] = []
    representative_titles: List[str] = []
    evidence_ids: List[str] = []
    retrieval_channels: List[str] = []
    source_diversity: float = 0.0
    conflict_summary: str = ""
    narrative_summary: str = ""


class NarrativesResponse(BaseModel):
    clusters: List[NarrativeClusterResponse]
    total_clusters: int
    global_coherence: float
    global_credibility: float
    total_evidence: int
    overall_sufficiency: str
    diagnostics: Dict[str, Any]


@router.get("/{case_id}", response_model=NarrativesResponse)
async def get_narratives(case_id: str):
    """Get narrative clusters for a case."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.narratives:
        raise HTTPException(status_code=404, detail="Case not found or no narratives")
    
    clusters = case.narratives.clusters or []
    
    cluster_responses = []
    for cluster in clusters:
        # Clusters are NarrativeCluster models; the response builder below reads
        # them as plain mappings.
        c = cluster.model_dump() if hasattr(cluster, "model_dump") else dict(cluster)

        # Calculate support/conflict ratios
        total_signals = c.get("support_signal_count", 0) + c.get("conflict_signal_count", 0)
        support_ratio = c.get("support_signal_count", 0) / total_signals if total_signals > 0 else None
        conflict_ratio = c.get("conflict_signal_count", 0) / total_signals if total_signals > 0 else None
        
        # Get representative evidence
        rep_evidence = c.get("representative_evidence", [])
        rep_ids = [e.get("evidence_id", "") for e in rep_evidence]
        rep_titles = [e.get("title", "") for e in rep_evidence]
        
# Get all evidence IDs in this cluster
        evidence_ids = c.get("evidence_ids", []) or []
        # Fall back to representative evidence IDs if evidence_ids is empty
        if not evidence_ids:
            evidence_ids = rep_ids
        
        # Get retrieval channels from representative evidence
        channels = []
        for e in c.get("representative_evidence", []):
            ch = e.get("retrieval_channel", "")
            if ch and ch not in channels:
                channels.append(ch)
        
        # Source diversity
        domains = c.get("unique_source_domains", [])
        source_diversity = len(domains) / max(1, c.get("source_count", 1))
        
        # Conflict summary
        support = c.get("support_signal_count", 0)
        conflict = c.get("conflict_signal_count", 0)
        if support > 0 and conflict > 0:
            conflict_summary = f"Mixed signals: {support} supporting, {conflict} conflicting"
        elif support > 0:
            conflict_summary = f"{support} supporting signals"
        elif conflict > 0:
            conflict_summary = f"{conflict} conflicting signals"
        else:
            conflict_summary = "No strong signals"
        
        cluster_responses.append(NarrativeClusterResponse(
            cluster_id=c.get("cluster_id", 0),
            size=c.get("size", 0),
            theme_keywords=c.get("theme_keywords", []),
            theme=c.get("theme", "") or (", ".join(c.get("theme_keywords", [])[:5]) if c.get("theme_keywords") else "No theme"),
            source_count=c.get("source_count", 0),
            unique_source_domains=c.get("unique_source_domains", []),
            mean_cluster_coherence=c.get("mean_cluster_coherence", 0.0),
            mean_credibility=c.get("mean_credibility", 0.0),
            mean_relevance=c.get("mean_relevance", 0.0),
            evidence_sufficiency=c.get("evidence_sufficiency", "unknown"),
            support_signal_count=c.get("support_signal_count", 0),
            conflict_signal_count=c.get("conflict_signal_count", 0),
            support_ratio=c.get("support_signal_count", 0) / max(1, c.get("support_signal_count", 0) + c.get("conflict_signal_count", 0)) if (c.get("support_signal_count", 0) + c.get("conflict_signal_count", 0)) > 0 else None,
            conflict_ratio=c.get("conflict_signal_count", 0) / max(1, c.get("support_signal_count", 0) + c.get("conflict_signal_count", 0)) if (c.get("support_signal_count", 0) + c.get("conflict_signal_count", 0)) > 0 else None,
            representative_evidence_ids=rep_ids,
            representative_titles=rep_titles,
            evidence_ids=evidence_ids,
            retrieval_channels=channels,
            source_diversity=source_diversity,
            conflict_summary=conflict_summary,
            narrative_summary=c.get("narrative_summary", ""),
        ))
    
    return NarrativesResponse(
        clusters=cluster_responses,
        total_clusters=len(clusters),
        global_coherence=case.narratives.global_cluster_coherence,
        global_credibility=case.narratives.global_credibility,
        total_evidence=case.narratives.diagnostics.get("total_evidence", 0) if case.narratives.diagnostics else 0,
        overall_sufficiency=case.narratives.overall_evidence_sufficiency,
        diagnostics=case.narratives.diagnostics or {},
    )


@router.get("/{case_id}/{cluster_id}")
async def get_narrative_detail(case_id: str, cluster_id: int):
    """Get detailed narrative cluster."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.narratives:
        raise HTTPException(status_code=404, detail="Case not found")
    
    for cluster in case.narratives.clusters or []:
        c = cluster.model_dump() if hasattr(cluster, "model_dump") else dict(cluster)
        if c.get("cluster_id") == cluster_id:
            return c
    
    raise HTTPException(status_code=404, detail="Cluster not found")