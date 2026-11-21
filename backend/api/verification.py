"""Verification API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uuid

from pipeline.orchestrator import verify_claim
from services.json_utils import make_json_safe
from services.time_service import utc_iso_now, india_now
from storage.case_store import (
    save_case,
    save_case_stage,
    load_case,
    cases_store,
    load_job_status,
)

router = APIRouter()


class VerificationRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=2000)
    image_path: str = ""
    promote_knowledge: bool = False
    case_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResponse(BaseModel):
    case_id: str
    job_id: str
    claim: str
    verdict: str
    confidence: float
    status: str
    message: str


class CaseInput(BaseModel):
    case_id: str
    claim: str
    image_path: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Persistent job store using the case store infrastructure
# Job IDs map to case_ids for durability across workers
verification_jobs: Dict[str, Dict[str, Any]] = {}


def run_verification_job(job_id: str, case_input: Dict[str, Any]):
    """Background verification job with persistent state."""
    # Ensure job exists in persistent store
    if job_id not in verification_jobs:
        verification_jobs[job_id] = {
            "job_id": job_id,
            "case_id": case_input.get("case_id", f"CASE-{uuid.uuid4().hex[:8]}"),
            "status": "queued",
            "created_at": utc_iso_now(),
            "case_input": case_input,
            "updated_at": utc_iso_now(),
        }
    
    job = verification_jobs[job_id]
    job["status"] = "running"
    job["started_at"] = utc_iso_now()
    job["updated_at"] = utc_iso_now()
    
    def progress_cb(stage: str, status: str):
        job["current_stage"] = stage
        job["stage_status"] = status
        job["updated_at"] = utc_iso_now()
        # Persist progress + job_id so state survives backend restarts
        case_id = job.get("case_id", "")
        if case_id:
            save_case_stage(case_id, stage, status, job_id=job_id)
    
    try:
        result = verify_claim(case_input, progress_cb=progress_cb)
        job["status"] = "completed"
        job["result"] = make_json_safe(result.model_dump() if hasattr(result, 'model_dump') else result)
        job["completed_at"] = utc_iso_now()
        job["updated_at"] = utc_iso_now()
        # Persist completed result + job_id so state survives backend restarts
        case_id = job.get("case_id", "")
        if case_id:
            save_case_stage(case_id, "completed", status="completed",
                            result=job["result"], job_id=job_id)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = utc_iso_now()
        job["updated_at"] = utc_iso_now()
        # Persist failed status + job_id
        case_id = job.get("case_id", "")
        if case_id:
            save_case_stage(case_id, "failed", status="failed", error=str(e), job_id=job_id)


@router.post("/start", response_model=VerificationResponse)
async def start_verification(request: VerificationRequest, background_tasks: BackgroundTasks):
    """Start a new verification job."""
    case_id = request.case_id or f"CASE-{uuid.uuid4().hex[:8]}"
    
    case_input = {
        "case_id": case_id,
        "claim": request.claim,
        "image_path": request.image_path,
        "promote_knowledge": request.promote_knowledge,
        "metadata": request.metadata,
    }
    
    job_id = str(uuid.uuid4())
    verification_jobs[job_id] = {
        "job_id": job_id,
        "case_id": case_id,
        "status": "queued",
        "created_at": utc_iso_now(),
        "case_input": case_input,
        "updated_at": utc_iso_now(),
    }
    
    # Also initialize in case store
    save_case_stage(case_id, "queued", status="queued", job_id=job_id)
    
    background_tasks.add_task(run_verification_job, job_id, case_input)
    
    return VerificationResponse(
        case_id=case_id,
        job_id=job_id,
        claim=request.claim,
        verdict="PENDING",
        confidence=0.0,
        status="queued",
        message=f"Verification job started. Job ID: {job_id}",
    )


@router.get("/status/{job_id}")
async def get_verification_status(job_id: str):
    """Get verification job status.

    Priority: in-memory job store (live run), in-memory case_id match,
    persistent case store on disk, and finally direct case record fallback.
    """
    # First check in-memory job store
    if job_id in verification_jobs:
        return verification_jobs[job_id]

    # Check if job_id matches case_id in memory
    for j in verification_jobs.values():
        if j.get("case_id") == job_id:
            return j

    # Then check persistent case store on disk
    persistent = load_job_status(job_id)
    if persistent is not None:
        # Jobs run inside this process, so one that is only on disk and unfinished was orphaned by a restart.
        from storage.case_store import mark_job_interrupted
        return mark_job_interrupted(persistent)

    # Direct case fallback if job_id was actually a case_id
    case = load_case(job_id)
    if case:
        return {
            "job_id": job_id,
            "case_id": job_id,
            "status": "completed",
            "current_stage": "completed",
            "stage_status": "completed",
            "result": make_json_safe(case.model_dump() if hasattr(case, 'model_dump') else case),
            "updated_at": utc_iso_now(),
        }

    raise HTTPException(status_code=404, detail=f"Verification job '{job_id}' not found")


@router.get("/result/{case_id}")
async def get_verification_result(case_id: str):
    """Get verification result by case ID."""
    # Find job with this case_id
    for job in verification_jobs.values():
        if job.get("case_id") == case_id and job.get("status") == "completed":
            return job.get("result")

    # Also check stored cases
    case = load_case(case_id)
    if case:
        return make_json_safe(case.model_dump())

    raise HTTPException(status_code=404, detail="Case not found")


@router.post("/sync", response_model=Dict[str, Any])
async def run_verification_sync(request: VerificationRequest):
    """Run verification synchronously (for testing)."""
    case_id = request.case_id or f"CASE-{india_now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    
    case_input = {
        "case_id": case_id,
        "claim": request.claim,
        "image_path": request.image_path,
        "promote_knowledge": request.promote_knowledge,
        "metadata": request.metadata,
    }
    
    try:
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(verify_claim, case_input)
        return make_json_safe(result.model_dump() if hasattr(result, 'model_dump') else result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
