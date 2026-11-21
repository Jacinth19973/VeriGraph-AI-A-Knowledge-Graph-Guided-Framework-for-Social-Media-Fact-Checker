"""Debate API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from storage.case_store import load_case

router = APIRouter()


class DebateTurnResponse(BaseModel):
    round: int
    speaker: str
    position: str
    confidence: float
    arguments: List[Dict[str, Any]] = []
    response_to_other: List[Dict[str, Any]] = []
    key_summary: str = ""
    key_evidence_summary: str = ""
    uncertainty_concessions: List[str] = []


class DebateResponse(BaseModel):
    max_rounds: int
    actual_rounds: int
    stop_reason: str
    turns: List[DebateTurnResponse]
    debate_log: List[Dict[str, Any]] = []
    debate_history: List[Dict[str, Any]] = []
    specialist_summary: Dict[str, Any] = {}
    uncertainty_profile: Dict[str, Any] = {}
    judge: Dict[str, Any] = {}
    final_verdict: str = ""
    final_confidence: float = 0.0


@router.get("/{case_id}", response_model=DebateResponse)
async def get_debate(case_id: str):
    """Get debate trace for a case."""
    from storage.case_store import load_case
    case = load_case(case_id)
    if not case or not case.debate:
        raise HTTPException(status_code=404, detail="Case not found or no debate")
    
    debate = case.debate
    
    # Get debate history from case if available
    debate_log = getattr(case, 'debate_history', [])
    
    return DebateResponse(
        max_rounds=debate.max_rounds,
        actual_rounds=debate.actual_rounds,
        stop_reason=debate.stop_reason,
        turns=[
            DebateTurnResponse(
                round=t.round,
                speaker=t.speaker,
                position=t.position,
                confidence=t.confidence,
                arguments=t.arguments,
                response_to_other=t.response_to_other,
                key_summary=t.key_summary,
            ) for t in debate.turns
        ],
        debate_log=debate_log,
        debate_history=debate_log,
        specialist_summary={},
        uncertainty_profile=case.uncertainty.dict() if case.uncertainty else {},
        judge=case.judge.dict() if case.judge else {},
        final_verdict=case.judge.final_verdict if case.judge else "UNKNOWN",
        final_confidence=case.judge.confidence if case.judge else 0.0,
    )


class PromptTraceResponse(BaseModel):
    agent_name: str
    round: int
    prompt_template_name: str
    system_prompt: str
    task_prompt: str
    structured_context: Dict[str, Any]
    response_schema: Dict[str, Any]
    timestamp: str
    latency_ms: float
    model: str
    temperature: float
    response_json: Dict[str, Any]


@router.get("/{case_id}/prompt-trace")
async def get_prompt_trace(case_id: str):
    """Get agent prompt trace for a case (from NB9 LLM trace log)."""
    # Load from NB9 LLM trace log
    from pathlib import Path
    from config.settings import PREPROCESSED_DIR
    trace_path = PREPROCESSED_DIR / "evidence_retrieval" / "debate_judgement" / "debate_llm_trace.jsonl"
    
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="Prompt trace not available for this case")
    
    traces = []
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                    # Redact sensitive information
                    redacted = _redact_trace(trace)
                    traces.append(redacted)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prompt trace: {e}")
    
    return {"traces": traces, "total": len(traces)}


def _redact_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive information from prompt trace."""
    redacted = trace.copy()
    # Redact API keys, tokens, etc.
    if "system_prompt" in redacted:
        redacted["system_prompt"] = _redact_secrets(redacted["system_prompt"])
    if "task_prompt" in redacted:
        redacted["task_prompt"] = _redact_secrets(redacted["task_prompt"])
    if "response" in redacted:
        redacted["response"] = _redact_secrets(redacted["response"])
    return redacted


def _redact_secrets(text: str) -> str:
    """Redact API keys and secrets from text."""
    import re
    # Redact API keys (common patterns)
    text = re.sub(r'(api[_-]?key|token|secret|password|auth)[\s:=]+[\w\-]+', r'\1=***REDACTED***', text, flags=re.IGNORECASE)
    # Redact bearer tokens
    text = re.sub(r'bearer\s+[\w\-]+', 'bearer ***REDACTED***', text, flags=re.IGNORECASE)
    return text


import json