"""VeriGraph AI — Runtime case store (JSON per case, no hallucination)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any

from config.settings import RUNTIME_DIR, CASES_DIR
from models.schemas import VerificationResult
from services.json_utils import make_json_safe
from services.time_service import utc_iso_now


# Persistent case store dictionary (survives across workers)
cases_store: Dict[str, Dict[str, Any]] = {}

def ensure_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)


def save_case(case: VerificationResult) -> Path:
    ensure_dirs()
    case_dir = CASES_DIR / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = make_json_safe(case.model_dump(mode="python"))
    payload["case_id"] = case.case_id
    out = case_dir / "case_record.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (case_dir / "claim.txt").write_text(case.claim or "", encoding="utf-8", errors="replace")
    metadata = {
        "source_image": case.image_path if not str(case.image_path).startswith("data:") else "uploaded image",
        "created_at": case.created_at,
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def save_case_stage(case_id: str, stage: str, status: str = "",
                     result: Dict[str, Any] = None, error: str = None,
                     job_id: str = None) -> None:
    """Save a verification job stage to the persistent case store.
    
    This allows job state to survive across backend worker restarts
    and HTTP request boundaries.
    """
    ensure_dirs()
    case_dir = CASES_DIR / case_id
    if not case_dir.exists():
        case_dir.mkdir(parents=True, exist_ok=True)
    
    stage_file = case_dir / "stage.json"
    stage_data = {
        "case_id": case_id,
        "stage": stage,
        "status": status,
        "updated_at": utc_iso_now(),
    }
    
    if result is not None:
        stage_data["result"] = result
    if error is not None:
        stage_data["error"] = error
    if job_id is not None:
        stage_data["job_id"] = job_id
    
    stage_file.write_text(json.dumps(make_json_safe(stage_data), indent=2), encoding="utf-8")


def list_cases() -> list:
    ensure_dirs()
    if not CASES_DIR.exists():
        return []
    return sorted(
        [p.name for p in CASES_DIR.iterdir() if p.is_dir()],
        key=lambda s: s, reverse=True,
    )


def load_case(case_id: str) -> Union[VerificationResult, None]:
    ensure_dirs()
    p = CASES_DIR / case_id / "case_record.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return VerificationResult(**data)
    except Exception as exc:
        print(f"[case_store] could not load {case_id}: {type(exc).__name__}: {exc}")
        return None


_INTERRUPTED = "Interrupted: the backend was restarted while this case was running. Please run it again."


def mark_interrupted_cases() -> int:
    '''At start-up no job can be running, so any case without a result that is not finished was interrupted.'''
    ensure_dirs()
    count = 0
    for case_dir in CASES_DIR.iterdir():
        stage_file = case_dir / "stage.json"
        if not case_dir.is_dir() or not stage_file.exists() or (case_dir / "case_record.json").exists():
            continue
        try:
            data = json.loads(stage_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("stage") in ("completed", "failed") or data.get("status") in ("completed", "failed"):
            continue
        save_case_stage(case_dir.name, "failed", status="failed", error=_INTERRUPTED, job_id=data.get("job_id"))
        count += 1
    return count


def mark_job_interrupted(job: Dict[str, Any]) -> Dict[str, Any]:
    '''A job read back from disk (not in this process's memory) that never finished was orphaned by a restart.'''
    if job.get("status") in ("completed", "failed"):
        return job
    if job.get("case_id"):
        save_case_stage(job["case_id"], "failed", status="failed", error=_INTERRUPTED, job_id=job.get("job_id"))
    return {**job, "status": "failed", "current_stage": "failed", "error": _INTERRUPTED}


def case_progress(case_id: str):
    '''Progress record for a case that has no result yet, or None if the case is unknown.'''
    ensure_dirs()
    case_dir = CASES_DIR / case_id
    stage_file = case_dir / "stage.json"
    if (case_dir / "case_record.json").exists() or not stage_file.exists():
        return None
    try:
        data = json.loads(stage_file.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    stage = str(data.get("stage", "")) or "queued"
    if stage == "failed":
        return {"case_id": case_id, "state": "failed", "stage": stage,
                "message": f"Case {case_id} failed: {data.get('error', 'unknown error')}"}
    if stage == "completed":
        return {"case_id": case_id, "state": "failed", "stage": stage,
                "message": f"Case {case_id} finished but its result was not saved; see the backend terminal ('persist failed')."}
    return {"case_id": case_id, "state": "running", "stage": stage,
            "message": f"Case {case_id} is running (stage: {stage})."}


def describe_missing_case(case_id: str):
    """(http_status, message) explaining why a case cannot be opened."""
    ensure_dirs()
    case_dir = CASES_DIR / case_id
    if (case_dir / "case_record.json").exists():
        return 500, f"Case {case_id} exists but could not be read; see the backend terminal for the validation error."
    stage_file = case_dir / "stage.json"
    if stage_file.exists():
        try:
            data = json.loads(stage_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        stage = str(data.get("stage", ""))
        if stage == "failed":
            return 409, f"Case {case_id} failed: {data.get('error', 'unknown error')}"
        if stage == "completed":
            return 500, f"Case {case_id} finished but its result was not saved; see the backend terminal ('persist failed')."
        return 409, f"Case {case_id} is still running (stage: {stage or 'queued'}). It will open when the run finishes."
    return 404, "Case not found"


def load_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Find a persisted job by id across all case directories.

    Each running/completed/failed verification job writes its job_id into
    ``runtime/cases/<case_id>/stage.json``. This lets the status endpoint
    recover jobs that were still running when the backend restarted.
    """
    ensure_dirs()
    if not CASES_DIR.exists():
        return None
    for case_dir in CASES_DIR.iterdir():
        if not case_dir.is_dir():
            continue
        stage_file = case_dir / "stage.json"
        if not stage_file.exists():
            continue
        try:
            data = json.loads(stage_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("job_id") != job_id:
            continue
        case_id = data.get("case_id") or case_dir.name
        record = case_dir / "case_record.json"
        result = None
        if record.exists():
            try:
                result = json.loads(record.read_text(encoding="utf-8"))
            except Exception:
                result = None
        status = data.get("status", "unknown")
        if result is not None:
            status = "completed"
        return {
            "job_id": job_id,
            "case_id": case_id,
            "status": status,
            "current_stage": data.get("stage", ""),
            "stage_status": data.get("status", ""),
            "error": data.get("error"),
            "result": result,
            "updated_at": data.get("updated_at"),
        }
    return None
