"""Analytics API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from storage.case_store import list_cases, load_case

router = APIRouter()


def _normalize_verdict(v):
    """Canonicalize verdict labels to avoid Insufficient vs INSUFFICIENT split."""
    if not v:
        return "UNKNOWN"
    s = str(v).strip().upper()
    if s in ("TRUE", "FALSE", "MISLEADING"):
        return s
    if s in ("INSUFFICIENT", "INSUFFICIENT_EVIDENCE"):
        return "INSUFFICIENT"
    return s


class AnalyticsResponse(BaseModel):
    total_cases: int
    verdict_distribution: Dict[str, int]
    average_processing_time: float
    average_debate_rounds: float
    agent_agreement_rate: float
    average_evidence_count: float
    average_narrative_count: float
    source_distribution: Dict[str, int]
    platform_distribution: Dict[str, int]
    verdict_by_platform: Dict[str, Dict[str, int]]


@router.get("/overview", response_model=AnalyticsResponse)
async def get_analytics_overview():
    """Get overall analytics across all cases."""
    case_ids = list_cases()
    
    verdict_dist = {}
    total_processing_time = 0.0
    total_debate_rounds = 0
    agent_agreements = 0
    agent_total = 0
    total_evidence = 0
    total_narratives = 0
    source_dist = {}
    platform_dist = {}
    verdict_by_platform = {}
    
    valid_cases = 0
    
    for case_id in case_ids:
        case = load_case(case_id)
        if not case:
            continue
        
        valid_cases += 1
        
        # Verdict distribution
        verdict = _normalize_verdict(case.judge.final_verdict if case.judge else "UNKNOWN")
        verdict_dist[verdict] = verdict_dist.get(verdict, 0) + 1
        
        # Processing time
        processing_metrics = case.processing_metrics or {}
        total_processing_time += processing_metrics.get("duration_seconds", 0.0)
        
        # Debate rounds
        if case.debate:
            total_debate_rounds += case.debate.actual_rounds
        
        # Agent agreement (claim_verification confidence as proxy)
        if case.specialists and case.specialists.claim_verification:
            agent_total += 1
            if case.specialists.claim_verification.confidence > 0.7:
                agent_agreements += 1
        
        # Evidence count
        if case.retrieval:
            total_evidence += case.retrieval.evidence_count
        
        # Narrative count
        if case.narratives:
            total_narratives += len(case.narratives.clusters)
        
        # Source distribution
        if case.retrieval:
            for e in case.retrieval.fused_evidence or []:
                src = e.get("retrieval_channel", "unknown")
                source_dist[src] = source_dist.get(src, 0) + 1
        
        # Platform distribution
        platform = case.input_metadata.get("source_platform", "local")
        platform_dist[platform] = platform_dist.get(platform, 0) + 1
        
        # Verdict by platform
        if platform not in verdict_by_platform:
            verdict_by_platform[platform] = {}
        verdict_by_platform[platform][verdict] = verdict_by_platform[platform].get(verdict, 0) + 1
    
    return AnalyticsResponse(
        total_cases=valid_cases,
        verdict_distribution=verdict_dist,
        average_processing_time=total_processing_time / max(1, valid_cases),
        average_debate_rounds=total_debate_rounds / max(1, valid_cases),
        agent_agreement_rate=agent_agreements / max(1, agent_total),
        average_evidence_count=total_evidence / max(1, valid_cases),
        average_narrative_count=total_narratives / max(1, valid_cases),
        source_distribution=source_dist,
        platform_distribution=platform_dist,
        verdict_by_platform=verdict_by_platform,
    )


@router.get("/verdict-trends")
async def get_verdict_trends(
    days: int = Query(30, ge=1, le=365),
):
    """Get verdict trends over time."""
    from datetime import datetime, timedelta, timezone
    from services.time_service import utc_now

    case_ids = list_cases()
    cutoff = utc_now() - timedelta(days=days)
    
    trends = {}
    for case_id in case_ids:
        case = load_case(case_id)
        if not case:
            continue
        
        try:
            created = datetime.fromisoformat(case.created_at.replace("Z", "+00:00"))
            # Legacy records may store naive timestamps; assume UTC so the
            # comparison against the tz-aware cutoff is valid.
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created < cutoff:
                continue
        except Exception:
            continue
        
        date_key = created.strftime("%Y-%m-%d")
        verdict = _normalize_verdict(case.judge.final_verdict if case.judge else "UNKNOWN")
        
        if date_key not in trends:
            trends[date_key] = {}
        trends[date_key][verdict] = trends[date_key].get(verdict, 0) + 1
    
    # Fill missing dates with zeros
    all_verdicts = {"TRUE", "FALSE", "MISLEADING", "INSUFFICIENT"}
    sorted_dates = sorted(trends.keys())
    result = []
    for date in sorted_dates:
        day_data = {"date": date}
        for v in all_verdicts:
            day_data[v] = trends[date].get(v, 0)
        result.append(day_data)
    
    return {"trends": result}


@router.get("/source-quality")
async def get_source_quality():
    """Get source quality distribution."""
    case_ids = list_cases()
    
    source_credibility = {}
    source_counts = {}
    
    for case_id in case_ids:
        case = load_case(case_id)
        if not case or not case.retrieval:
            continue
        
        for e in case.retrieval.fused_evidence or []:
            src = e.get("provider", "unknown")
            cred = e.get("credibility_score", 0.0)
            
            if src not in source_credibility:
                source_credibility[src] = []
            source_credibility[src].append(cred)
            source_counts[src] = source_counts.get(src, 0) + 1
    
    result = {}
    for src, creds in source_credibility.items():
        result[src] = {
            "avg_credibility": sum(creds) / len(creds) if creds else 0,
            "count": source_counts.get(src, 0),
            "min_credibility": min(creds) if creds else 0,
            "max_credibility": max(creds) if creds else 0,
        }
    
    return {"sources": result}


@router.get("/evidence-stats")
async def get_evidence_statistics():
    """Get evidence statistics."""
    case_ids = list_cases()
    
    total_evidence = 0
    type_dist = {}
    channel_dist = {}
    prov_dist = {}
    domain_dist = {}
    avg_relevance = 0.0
    avg_credibility = 0.0
    avg_fusion = 0.0
    total_with_scores = 0
    
    for case_id in case_ids:
        case = load_case(case_id)
        if not case or not case.retrieval:
            continue
        
        for e in case.retrieval.fused_evidence or []:
            total_evidence += 1
            total_with_scores += 1
            
            avg_relevance += e.get("relevance_score", 0)
            avg_credibility += e.get("credibility_score", 0)
            avg_fusion += e.get("fusion_score", 0)
            
            type_dist[e.get("evidence_type", "unknown")] = type_dist.get(e.get("evidence_type", "unknown"), 0) + 1
            channel_dist[e.get("retrieval_channel", "unknown")] = channel_dist.get(e.get("retrieval_channel", "unknown"), 0) + 1
            prov_dist[e.get("provenance_type", "UNRESOLVED")] = prov_dist.get(e.get("provenance_type", "UNRESOLVED"), 0) + 1
            domain_dist[e.get("domain", "unknown")] = domain_dist.get(e.get("domain", "unknown"), 0) + 1
    
    return {
        "total_evidence": total_evidence,
        "type_distribution": type_dist,
        "channel_distribution": channel_dist,
        "provenance_distribution": prov_dist,
        "top_domains": dict(sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)[:20]),
        "avg_relevance": avg_relevance / max(1, total_with_scores),
        "avg_credibility": avg_credibility / max(1, total_with_scores),
        "avg_fusion": avg_fusion / max(1, total_with_scores),
    }