"""Knowledge Graph API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from storage.case_store import load_case
from services.neo4j_service import query_kg_context, get_kg_stats

router = APIRouter()


class KGNodeResponse(BaseModel):
    node_id: str
    name: str
    labels: List[str]
    properties: Dict[str, Any] = Field(default_factory=dict)


class KGRelationshipResponse(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class KGContextResponse(BaseModel):
    status: str
    entities: List[str] = Field(default_factory=list)
    persons: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    source_domains: List[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    related_claims: List[str] = Field(default_factory=list)
    nodes_inspected: int = 0
    nodes_returned: int = 0
    relationships_returned: int = 0
    duration_seconds: float = 0.0
    neo4j_used: bool = False
    reason: str = ""


class KGCandidateResponse(BaseModel):
    candidate_id: str
    entity_type: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_evidence_ids: List[str] = Field(default_factory=list)
    source_case_id: str = ""
    confidence: float = 0.0
    validation_status: str = "CANDIDATE"
    source_quote: str = ""


class KGResponse(BaseModel):
    kg_context: KGContextResponse
    kg_expansion: Dict[str, Any] = {}
    kg_findings: List[str] = []
    kg_gaps: List[str] = []
    expansion_candidates: List[KGCandidateResponse] = []


@router.get("/{case_id}", response_model=KGResponse)
async def get_knowledge_graph(case_id: str):
    """Get knowledge graph context and expansion for a case."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Use the KG context captured during the run; only query Neo4j on demand
    # for older cases that were saved before kg_context was persisted.
    kg_context_data = dict(getattr(case, "kg_context", None) or {})
    if not kg_context_data:
        evidence_items = case.retrieval.fused_evidence if case.retrieval else []
        try:
            kg_context_data = query_kg_context(case_id=case.case_id, claim=case.claim, evidence_items=evidence_items)
        except Exception as exc:
            kg_context_data = {"status": "unavailable", "reason": str(exc)}
    kg_context_data.setdefault("status", "unavailable")
    allowed = set(KGContextResponse.model_fields.keys())
    kg_context_data = {k: v for k, v in kg_context_data.items() if k in allowed}
    
    # Get explanation KG findings and gaps
    exp = case.explanation
    kg_findings = exp.kg_findings if exp else []
    kg_gaps = exp.knowledge_gaps if exp else []
    
    # KG expansion candidates
    candidates = []
    if case.kg_update and case.kg_update.candidates:
        for c in case.kg_update.candidates:
            c_data = c.dict() if hasattr(c, 'dict') else c
            candidates.append(KGCandidateResponse(**c_data))
    
    return KGResponse(
        kg_context=KGContextResponse(**kg_context_data),
        kg_expansion={
            "candidates_generated": len(candidates),
            "candidates_validated": case.kg_update.candidates_validated if case.kg_update else 0,
            "candidates_rejected": case.kg_update.candidates_rejected if case.kg_update else 0,
            "candidates_promoted": case.kg_update.candidates_promoted if case.kg_update else 0,
        },
        kg_findings=kg_findings,
        kg_gaps=kg_gaps,
        expansion_candidates=candidates,
    )


@router.get("/{case_id}/graph")
async def get_kg_graph(case_id: str):
    """Get KG graph data for visualization (nodes and edges)."""
    from storage.case_store import load_case
    from services.neo4j_service import query_kg_context
    
    case = load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get KG context
    evidence_items = case.retrieval.fused_evidence if case.retrieval else []
    kg_context = query_kg_context(case_id=case.case_id, claim=case.claim, evidence_items=evidence_items)
    
    # Build graph structure for React Flow / Cytoscape
    nodes = []
    edges = []
    
    # Add claim node
    nodes.append({
        "id": f"claim_{case_id}",
        "type": "Claim",
        "data": {"label": case.claim[:50] + "...", "full": case.claim, "type": "Claim"},
        "position": {"x": 0, "y": 0},
    })
    
    # Add entities
    for i, entity in enumerate(kg_context.get("entities", [])):
        node_id = f"entity_{i}"
        nodes.append({
            "id": node_id,
            "type": "Entity",
            "data": {"label": entity, "type": "Entity"},
            "position": {"x": 100 * i, "y": 100},
        })
        edges.append({
            "id": f"edge_claim_{node_id}",
            "source": f"claim_{case_id}",
            "target": node_id,
            "label": "MENTIONS",
        })
    
    # Add persons
    for i, person in enumerate(kg_context.get("persons", [])):
        node_id = f"person_{i}"
        nodes.append({
            "id": node_id,
            "type": "Person",
            "data": {"label": person, "type": "Person"},
            "position": {"x": 100 * i, "y": 200},
        })
        edges.append({
            "id": f"edge_claim_{node_id}",
            "source": f"claim_{case_id}",
            "target": node_id,
            "label": "ABOUT",
        })
    
    # Add organizations
    for i, org in enumerate(kg_context.get("organizations", [])):
        node_id = f"org_{i}"
        nodes.append({
            "id": node_id,
            "type": "Organization",
            "data": {"label": org, "type": "Organization"},
            "position": {"x": 100 * i, "y": 300},
        })
        edges.append({
            "id": f"edge_claim_{node_id}",
            "source": f"claim_{case_id}",
            "target": node_id,
            "label": "ABOUT",
        })
    
    # Add locations
    for i, loc in enumerate(kg_context.get("locations", [])):
        node_id = f"location_{i}"
        nodes.append({
            "id": node_id,
            "type": "Location",
            "data": {"label": loc, "type": "Location"},
            "position": {"x": 100 * i, "y": 400},
        })
        edges.append({
            "id": f"edge_claim_{node_id}",
            "source": f"claim_{case_id}",
            "target": node_id,
            "label": "OCCURRED_AT",
        })
    
    # Add events
    for i, event in enumerate(kg_context.get("events", [])):
        node_id = f"event_{i}"
        nodes.append({
            "id": node_id,
            "type": "Event",
            "data": {"label": event, "type": "Event"},
            "position": {"x": 100 * i, "y": 500},
        })
        edges.append({
            "id": f"edge_claim_{node_id}",
            "source": f"claim_{case_id}",
            "target": node_id,
            "label": "ABOUT",
        })
    
    # Add dates
    for i, date in enumerate(kg_context.get("dates", [])):
        node_id = f"date_{i}"
        nodes.append({
            "id": node_id,
            "type": "Date",
            "data": {"label": date, "type": "Date"},
            "position": {"x": 100 * i, "y": 600},
        })
        edges.append({
            "id": f"edge_claim_{node_id}",
            "source": f"claim_{case_id}",
            "target": node_id,
            "label": "HAS_DATE",
        })
    
    # Add KG relationships if available
    for rel in kg_context.get("relationships", []):
        edges.append({
            "id": f"rel_{rel.get('source', '')}_{rel.get('target', '')}",
            "source": rel.get("source", ""),
            "target": rel.get("target", ""),
            "label": rel.get("type", "RELATED"),
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "case_id": case_id,
            "claim": case.claim,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
    }


@router.get("/stats")
async def get_kg_statistics():
    """Get Neo4j database statistics."""
    try:
        stats = get_kg_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")