"""Audit Trail API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from storage.case_store import load_case

router = APIRouter()


class AuditStageResponse(BaseModel):
    stage: str
    status: str
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    message: str = ""
    error: str = ""


class AuditResponse(BaseModel):
    case_id: str
    created_at: str
    pipeline_started_at: str = ""
    pipeline_completed_at: str = ""
    total_duration_seconds: float = 0.0
    stages: List[AuditStageResponse] = []
    processing_metrics: Dict[str, Any] = {}


@router.get("/{case_id}", response_model=AuditResponse)
async def get_audit_trail(case_id: str):
    """Get audit trail for a case."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    stages = []
    for e in case.audit:
        stages.append(AuditStageResponse(
            stage=e.stage,
            status=e.status,
            start_time=e.start_time or "",
            end_time=e.end_time or "",
            duration_seconds=e.duration_seconds or 0.0,
            message=e.message or "",
            error=e.error or "",
        ))
    
    processing_metrics = case.processing_metrics or {}
    
    return AuditResponse(
        case_id=case.case_id,
        created_at=case.created_at,
        pipeline_started_at=stages[0].start_time if stages else "",
        pipeline_completed_at=stages[-1].end_time if stages else "",
        total_duration_seconds=processing_metrics.get("duration_seconds", 0.0),
        stages=stages,
        processing_metrics=processing_metrics,
    )


@router.get("/{case_id}/timeline")
async def get_audit_timeline(case_id: str):
    """Get audit timeline for visualization."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    timeline = []
    for e in case.audit:
        timeline.append({
            "stage": e.stage,
            "status": e.status,
            "start": e.start_time,
            "end": e.end_time,
            "duration": e.duration_seconds,
            "message": e.message,
            "error": e.error,
        })
    
    return {
        "case_id": case_id,
        "timeline": timeline,
        "total_duration": sum(e.duration_seconds or 0 for e in case.audit),
    }