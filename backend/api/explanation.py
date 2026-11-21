"""Explanation API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from storage.case_store import load_case

router = APIRouter()


class VerificationFactorResponse(BaseModel):
    factor: str
    assessment: str
    confidence: float
    status: str
    detail: str = ""


class ExplanationResponse(BaseModel):
    summary: str
    verdict_reason: str
    verification_factors: List[VerificationFactorResponse] = []
    uncertainty: Dict[str, Any] = {}
    supporting_evidence: List[str] = []
    contradicting_evidence: List[str] = []
    neutral_evidence: List[str] = []
    temporal_findings: List[str] = []
    location_findings: List[str] = []
    source_findings: List[str] = []
    kg_findings: List[str] = []
    debate_findings: List[str] = []
    uncertainty_factors: List[str] = []
    knowledge_gaps: List[str] = []
    natural_language_summary: str = ""
    confidence_definition: str = ""
    provenance_chain: List[Dict[str, Any]] = []
    agent_summary: List[Dict[str, Any]] = []


@router.get("/{case_id}", response_model=ExplanationResponse)
async def get_explanation(case_id: str):
    """Get explanation for a case."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.explanation:
        raise HTTPException(status_code=404, detail="Case not found or no explanation")
    
    exp = case.explanation
    
    # Convert factor_cards to verification factors
    verification_factors = []
    for f in exp.factor_cards or []:
        verification_factors.append(VerificationFactorResponse(
            factor=f.get("factor", ""),
            assessment=f.get("assessment", ""),
            confidence=f.get("confidence", 0.0),
            status=f.get("status", ""),
            detail=f.get("detail", ""),
        ))
    
    # Build uncertainty object
    uncertainty_obj = {}
    if exp.uncertainty_factors:
        for uf in exp.uncertainty_factors:
            if "=" in uf:
                k, v = uf.split("=", 1)
                try:
                    uncertainty_obj[k.strip().replace(" ", "_")] = float(v.strip())
                except ValueError:
                    uncertainty_obj[k.strip().replace(" ", "_")] = v.strip()
    
    uncertainty_obj["overall_score"] = case.uncertainty.weighted_uncertainty_score if case.uncertainty else 0.0
    uncertainty_obj["level"] = case.uncertainty.debate_priority if case.uncertainty else "low"
    
    return ExplanationResponse(
        summary=exp.summary or "",
        verdict_reason=exp.verdict_reason or "",
        verification_factors=verification_factors,
        uncertainty=uncertainty_obj,
        supporting_evidence=exp.supporting_evidence or [],
        contradicting_evidence=exp.contradicting_evidence or [],
        neutral_evidence=exp.neutral_evidence or [],
        temporal_findings=exp.temporal_findings or [],
        location_findings=exp.location_findings or [],
        source_findings=exp.source_findings or [],
        kg_findings=exp.kg_findings or [],
        debate_findings=exp.debate_findings or [],
        uncertainty_factors=exp.uncertainty_factors or [],
        knowledge_gaps=exp.knowledge_gaps or [],
        natural_language_summary=exp.natural_language_summary or "",
        confidence_definition=exp.confidence_definition or "Verification Confidence",
        provenance_chain=exp.provenance_chain or [],
        agent_summary=exp.agent_summary or [],
    )


@router.get("/{case_id}/natural-language")
async def get_natural_language_summary(case_id: str):
    """Get natural language summary."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.explanation:
        raise HTTPException(status_code=404, detail="Case not found")
    
    return {
        "summary": case.explanation.natural_language_summary or "",
        "verdict": case.judge.final_verdict if case.judge else "UNKNOWN",
        "confidence": case.judge.confidence if case.judge else 0.0,
    }