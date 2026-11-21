"""Cases API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from storage.case_store import list_cases, load_case, save_case
from models.schemas import VerificationResult

router = APIRouter()


class CaseSummary(BaseModel):
    case_id: str
    claim: str
    verdict: str
    confidence: float
    created_at: str
    source_platform: Optional[str] = None
    source_url: Optional[str] = None
    processing_time_seconds: float = 0.0
    total_evidence: int = 0
    narrative_count: int = 0


class CaseListResponse(BaseModel):
    cases: List[CaseSummary]
    total: int
    page: int
    page_size: int


@router.get("/", response_model=CaseListResponse)
async def list_cases_endpoint(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    verdict: Optional[str] = None,
    platform: Optional[str] = None,
    search: Optional[str] = None,
):
    """List all stored verification cases with pagination and filters."""
    case_ids = list_cases()
    
    cases_data = []
    for case_id in case_ids:
        case = load_case(case_id)
        if not case:
            continue
        
        # Apply filters
        if verdict and (case.judge.final_verdict or "").strip().upper() != verdict.strip().upper():
            continue
        if platform:
            source_platform = case.input_metadata.get("source_platform", "")
            if source_platform != platform:
                continue
        if search:
            if search.lower() not in (case.claim or "").lower():
                continue
        
        processing_metrics = case.processing_metrics or {}
        
        cases_data.append(CaseSummary(
            case_id=case.case_id,
            claim=case.claim,
            verdict=case.judge.final_verdict if case.judge else "UNKNOWN",
            confidence=case.judge.confidence if case.judge else 0.0,
            created_at=case.created_at,
            source_platform=case.input_metadata.get("source_platform"),
            source_url=case.input_metadata.get("source_url"),
            processing_time_seconds=processing_metrics.get("duration_seconds", 0.0),
            total_evidence=case.retrieval.evidence_count if case.retrieval else 0,
            narrative_count=len(case.narratives.clusters) if case.narratives else 0,
        ))
    
    # Sort by created_at descending
    cases_data.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    total = len(cases_data)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = cases_data[start:end]
    
    return CaseListResponse(
        cases=paginated,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{case_id}")
async def get_case(case_id: str):
    """Get full case details."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case:
        from storage.case_store import case_progress, describe_missing_case
        progress = case_progress(case_id)
        if progress:                       # still running (or ended without a result): a normal 200, not an HTTP error
            return progress
        code, detail = describe_missing_case(case_id)
        raise HTTPException(status_code=code, detail=detail)
    from services.json_utils import make_json_safe
    return make_json_safe(case.model_dump())


@router.get("/{case_id}/summary")
async def get_case_summary(case_id: str):
    """Get case summary for quick preview."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    processing_metrics = case.processing_metrics or {}
    
    return {
        "case_id": case.case_id,
        "claim": case.claim,
        "verdict": case.judge.final_verdict if case.judge else "UNKNOWN",
        "confidence": case.judge.confidence if case.judge else 0.0,
        "created_at": case.created_at,
        "source_platform": case.input_metadata.get("source_platform"),
        "source_url": case.input_metadata.get("source_url"),
        "processing_time_seconds": processing_metrics.get("duration_seconds", 0.0),
        "total_evidence": case.retrieval.evidence_count if case.retrieval else 0,
        "narrative_count": len(case.narratives.clusters) if case.narratives else 0,
        "image_path": case.image_path,
    }


@router.delete("/{case_id}")
async def delete_case(case_id: str):
    """Delete a case (soft delete - mark as deleted)."""
    import shutil
    from config.settings import CASES_DIR
    case_dir = CASES_DIR / case_id
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Move to deleted folder
    deleted_dir = case_dir.parent / "deleted"
    deleted_dir.mkdir(exist_ok=True)
    shutil.move(str(case_dir), str(deleted_dir / case_id))
    
    return {"message": f"Case {case_id} deleted"}