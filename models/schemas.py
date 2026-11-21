"""VeriGraph AI — Pydantic schemas for the full pipeline."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from services.time_service import utc_iso_now


class CaseInput(BaseModel):
    case_id: str
    claim: str
    image_path: str
    created_at: str = Field(default_factory=utc_iso_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    text_model: str = "BAAI/bge-base-en-v1.5"
    image_model: str = "openai/clip-vit-base-patch32"


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_type: str = ""
    retrieval_channel: str = ""
    provider: str = ""
    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = "unknown"
    source: str = ""
    published_at: str = ""
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    entity_match_score: float = 0.0
    location_match_score: float = 0.0
    temporal_match_score: float = 0.0
    fusion_score: float = 0.0
    cluster_id: int = -1
    nb6_final_evidence_score: float = 0.0


class RetrievalResult(BaseModel):
    historical_text: List[Dict[str, Any]] = Field(default_factory=list)
    historical_image: List[Dict[str, Any]] = Field(default_factory=list)
    live_results: List[Dict[str, Any]] = Field(default_factory=list)
    combined_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    provider_status: Dict[str, Any] = Field(default_factory=dict)
    fused_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_count: int = 0


class NarrativeCluster(BaseModel):
    cluster_id: int
    size: int = 0
    theme: str = ""
    theme_keywords: List[str] = Field(default_factory=list)
    source_count: int = 0
    unique_source_domains: List[str] = Field(default_factory=list)
    mean_cluster_coherence: float = 0.0
    mean_credibility: float = 0.0
    mean_relevance: float = 0.0
    evidence_sufficiency: str = "unknown"
    support_signal_count: int = 0
    conflict_signal_count: int = 0
    evidence_ids: List[str] = Field(default_factory=list)
    representative_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    representative_evidence_id: str = ""
    representative_title: str = ""


class NarrativeResult(BaseModel):
    clusters: List[NarrativeCluster] = Field(default_factory=list)
    fused_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    overall_evidence_sufficiency: str = "unknown"
    global_cluster_coherence: float = 0.0
    global_credibility: float = 0.0
    grounded_narratives: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class SpecialistResult(BaseModel):
    agent: str
    status: str = "pending"
    assessment: str = "insufficient"
    confidence: float = 0.0
    key_findings: List[str] = Field(default_factory=list)
    uncertainty_flags: List[str] = Field(default_factory=list)
    reasoning: str = ""
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    refuting_evidence_ids: List[str] = Field(default_factory=list)
    evidence_assessments: List[Dict[str, Any]] = Field(default_factory=list)
    source_assessments: List[Dict[str, Any]] = Field(default_factory=list)
    temporal_findings: List[Dict[str, Any]] = Field(default_factory=list)
    location_findings: List[Dict[str, Any]] = Field(default_factory=list)
    contradiction_evidence_ids: List[str] = Field(default_factory=list)


class SpecialistResultBundle(BaseModel):
    evidence_verification: SpecialistResult = Field(default_factory=lambda: SpecialistResult(agent="evidence_verification"))
    claim_verification: SpecialistResult = Field(default_factory=lambda: SpecialistResult(agent="claim_verification"))
    source_credibility: SpecialistResult = Field(default_factory=lambda: SpecialistResult(agent="source_credibility"))
    temporal_location_consistency: SpecialistResult = Field(default_factory=lambda: SpecialistResult(agent="temporal_location_consistency"))
    evidence_matrix: List[Dict[str, Any]] = Field(default_factory=list)


class UncertaintyProfile(BaseModel):
    agent_disagreement: float = 0.0
    evidence_conflict: float = 0.0
    low_source_credibility: float = 0.0
    temporal_location_inconsistency: float = 0.0
    evidence_insufficiency: float = 0.0
    weighted_uncertainty_score: float = 0.0
    debate_priority: str = "low"


class DebateTurn(BaseModel):
    round: int
    speaker: str
    position: str = ""
    confidence: float = 0.0
    arguments: List[Dict[str, Any]] = Field(default_factory=list)
    response_to_other: List[Dict[str, Any]] = Field(default_factory=list)
    key_summary: str = ""


class DebateResult(BaseModel):
    max_rounds: int = 2
    actual_rounds: int = 0
    stop_reason: str = ""
    turns: List[DebateTurn] = Field(default_factory=list)


class JudgeResult(BaseModel):
    final_verdict: str = "INSUFFICIENT"
    confidence: float = 0.0
    supporter_confidence: float = 0.0
    skeptic_confidence: float = 0.0
    reasoning: str = ""
    key_evidence_cited: List[str] = Field(default_factory=list)
    key_kg_gaps_cited: List[str] = Field(default_factory=list)
    dissent_notes: str = ""
    label_reasoning_alignment: Dict[str, Any] = Field(default_factory=dict)
    judge_status: str = "pending"
    judge_error: str = ""
    temporal_analysis: Dict[str, Any] = Field(default_factory=dict)
    current_date: str = ""
    current_date_ist: str = ""


class ExplanationResult(BaseModel):
    summary: str = ""
    verdict_reason: str = ""
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    neutral_evidence: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    temporal_findings: List[str] = Field(default_factory=list)
    location_findings: List[str] = Field(default_factory=list)
    source_findings: List[str] = Field(default_factory=list)
    kg_findings: List[str] = Field(default_factory=list)
    debate_findings: List[str] = Field(default_factory=list)
    uncertainty_factors: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    natural_language_summary: str = ""
    factor_cards: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_chain: List[Dict[str, Any]] = Field(default_factory=list)
    agent_summary: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_definition: str = ""


class KGCandidate(BaseModel):
    candidate_id: str
    entity_type: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_evidence_ids: List[str] = Field(default_factory=list)
    source_case_id: str = ""
    confidence: float = 0.0
    validation_status: str = "CANDIDATE"
    timestamp: str = ""


class KGExpansionResult(BaseModel):
    candidates_generated: int = 0
    candidates_validated: int = 0
    candidates_rejected: int = 0
    candidates_promoted: int = 0
    candidates: List[KGCandidate] = Field(default_factory=list)


class AuditEntry(BaseModel):
    stage: str
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    message: str = ""
    error: Optional[str] = None


class VerificationResult(BaseModel):
    case_id: str
    claim: str
    image_path: str = ""
    created_at: str = ""
    input_metadata: Dict[str, Any] = Field(default_factory=dict)
    clip_text_embedding: Optional[List[float]] = None
    retrieval: RetrievalResult = Field(default_factory=RetrievalResult)
    narratives: NarrativeResult = Field(default_factory=NarrativeResult)
    specialists: SpecialistResultBundle = Field(default_factory=SpecialistResultBundle)
    uncertainty: UncertaintyProfile = Field(default_factory=UncertaintyProfile)
    debate: DebateResult = Field(default_factory=DebateResult)
    judge: JudgeResult = Field(default_factory=JudgeResult)
    explanation: ExplanationResult = Field(default_factory=ExplanationResult)
    kg_context: Dict[str, Any] = Field(default_factory=dict)
    multimodal: Dict[str, Any] = Field(default_factory=dict)
    kg_update: KGExpansionResult = Field(default_factory=KGExpansionResult)
    audit: List[AuditEntry] = Field(default_factory=list)
    processing_metrics: Dict[str, Any] = Field(default_factory=dict)
    pipeline_config: Dict[str, bool] = Field(default_factory=dict)
    config_name: str = "D"
