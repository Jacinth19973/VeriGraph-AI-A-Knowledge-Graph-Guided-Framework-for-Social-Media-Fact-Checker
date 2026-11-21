"""Specialist Agents API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from storage.case_store import load_case

router = APIRouter()


class AgentAssessment(BaseModel):
    evidence_id: str
    relevance: str
    stance: str
    quality: str
    confidence: float
    reason: str
    status: str


class SpecialistResponse(BaseModel):
    agent: str
    status: str
    assessment: str
    confidence: float
    overall_assessment: str = ""
    key_findings: List[str] = []
    uncertainty_flags: List[str] = []
    reasoning: str = ""
    supporting_evidence_ids: List[str] = []
    refuting_evidence_ids: List[str] = []
    evidence_assessments: List[AgentAssessment] = []
    source_assessments: List[Dict[str, Any]] = []
    temporal_findings: List[Dict[str, Any]] = []
    location_findings: List[Dict[str, Any]] = []
    contradiction_evidence_ids: List[str] = []
    evidence_coverage: float = 0.0
    retrieval_support: float = 0.0
    composite_assessment_score: float = 0.0
    score_components: Dict[str, float] = {}


class SpecialistsResponse(BaseModel):
    evidence_verification: SpecialistResponse
    claim_verification: SpecialistResponse
    source_credibility: SpecialistResponse
    temporal_location_consistency: SpecialistResponse
    specialist_summary: Dict[str, Any] = {}
    evidence_assessment_matrix: List[Dict[str, Any]] = []


def _map_specialist(agent_name: str, data: Dict[str, Any]) -> SpecialistResponse:
    """Map specialist data to response model."""
    if not data:
        return SpecialistResponse(
            agent=agent_name, status="not_run", assessment="unknown", confidence=0.0
        )
    
    evidence_assessments = []
    for e in data.get("evidence_assessments", []):
        evidence_assessments.append(AgentAssessment(
            evidence_id=e.get("evidence_id", ""),
            relevance=e.get("relevance", ""),
            stance=e.get("stance", ""),
            quality=e.get("quality", ""),
            confidence=e.get("confidence", 0.0),
            reason=e.get("reason", ""),
            status=e.get("status", ""),
        ))
    
    # Calculate composite assessment score
    agent_conf = data.get("confidence", 0.0)
    evidence_cov = 0.0
    retrieval_sup = 0.0
    
    if agent_name == "evidence_verification":
        valid = [r for r in data.get("evidence_assessments", []) if r.get("status") == "completed"]
        evidence_cov = len(valid) / max(1, len(data.get("evidence_assessments", []))) if data.get("evidence_assessments") else 0.0
        retrieval_sup = sum(1 for r in valid if r.get("stance") == "supports") / max(1, len(valid)) if valid else 0.0
    
    score_components = {
        "agent_confidence": data.get("confidence", 0.0),
        "evidence_coverage": evidence_cov,
        "retrieval_support": retrieval_sup,
    }
    composite = sum(score_components.values()) / len(score_components)
    
    return SpecialistResponse(
        agent=data.get("agent", agent_name),
        status=data.get("status", "completed"),
        assessment=data.get("assessment") or data.get("overall_assessment", "insufficient"),
        confidence=data.get("confidence", 0.0),
        overall_assessment=data.get("overall_assessment", ""),
        key_findings=data.get("key_findings", []),
        uncertainty_flags=data.get("uncertainty_flags", []),
        reasoning=data.get("reasoning", ""),
        supporting_evidence_ids=data.get("supporting_evidence_ids", []),
        refuting_evidence_ids=data.get("refuting_evidence_ids", []),
        evidence_assessments=evidence_assessments,
        source_assessments=data.get("source_assessments", []),
        temporal_findings=data.get("temporal_findings", []),
        location_findings=data.get("location_findings", []),
        contradiction_evidence_ids=data.get("contradiction_evidence_ids", []),
        evidence_coverage=evidence_cov,
        retrieval_support=retrieval_sup,
        composite_assessment_score=composite,
        score_components=score_components,
    )


@router.get("/{case_id}", response_model=SpecialistsResponse)
async def get_specialists(case_id: str):
    """Get specialist agent results for a case."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.specialists:
        raise HTTPException(status_code=404, detail="Case not found or no specialist results")
    
    spec = case.specialists
    
    return SpecialistsResponse(
        evidence_verification=_map_specialist("evidence_verification", spec.evidence_verification.dict() if hasattr(spec.evidence_verification, 'dict') else spec.evidence_verification),
        claim_verification=_map_specialist("claim_verification", spec.claim_verification.dict() if hasattr(spec.claim_verification, 'dict') else spec.claim_verification),
        source_credibility=_map_specialist("source_credibility", spec.source_credibility.dict() if hasattr(spec.source_credibility, 'dict') else spec.source_credibility),
        temporal_location_consistency=_map_specialist("temporal_location_consistency", spec.temporal_location_consistency.dict() if hasattr(spec.temporal_location_consistency, 'dict') else spec.temporal_location_consistency),
        specialist_summary={},
        evidence_assessment_matrix=spec.evidence_matrix if spec.evidence_matrix else [],
    )


@router.get("/{case_id}/{agent_name}")
async def get_agent_detail(case_id: str, agent_name: str):
    """Get detailed specialist agent result."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.specialists:
        raise HTTPException(status_code=404, detail="Case not found")
    
    spec = case.specialists
    agent_map = {
        "evidence_verification": spec.evidence_verification,
        "claim_verification": spec.claim_verification,
        "source_credibility": spec.source_credibility,
        "temporal_location_consistency": spec.temporal_location_consistency,
    }
    
    agent_data = agent_map.get(agent_name)
    if not agent_data:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    return _map_specialist(agent_name, agent_data.dict() if hasattr(agent_data, 'dict') else agent_data)