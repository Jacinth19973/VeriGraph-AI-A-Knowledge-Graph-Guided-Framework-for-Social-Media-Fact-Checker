"""VeriGraph AI — NB8 Specialist Artifacts Loader."""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from config.settings import NB8_DIR


def _load_nb8_specialist_results(case_id: str = None) -> Optional[Dict[str, Any]]:
    """Load pre-computed specialist results from NB8 artifacts."""
    p = NB8_DIR / "specialist_agent_results.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # If case_id provided, verify it matches
        if case_id and data.get("input", {}).get("case", {}).get("case_id") != case_id:
            return None
        return data
    except Exception:
        return None


def _load_nb8_evidence_assessments(case_id: str = None) -> Optional[List[Dict[str, Any]]]:
    """Load evidence assessment matrix from NB8 artifacts."""
    p = NB8_DIR / "specialist_evidence_assessments.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        # If case_id provided, we'd filter but the CSV doesn't have case_id
        # In production, we'd have case-specific files
        return df.to_dict(orient="records")
    except Exception:
        return None


def load_nb8_specialists(case_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Load pre-computed NB8 specialist results.
    
    Returns a dict with all four specialist agents populated from artifacts,
    or None if not available.
    """
    results = _load_nb8_specialist_results(case_id)
    if not results:
        return None
    
    specialists = results.get("specialist_agents", {})
    if not specialists:
        return None
    
    # Format to match expected pipeline output
    return {
        "evidence_verification": specialists.get("evidence_verification"),
        "claim_verification": specialists.get("claim_verification"),
        "source_credibility": specialists.get("source_credibility"),
        "temporal_location_consistency": specialists.get("temporal_location_consistency"),
        "specialist_summary": results.get("specialist_summary"),
        "evidence_assessment_matrix": results.get("evidence_assessment_matrix"),
    }


def get_evidence_assessments_from_nb8(case_id: str = None) -> List[Dict[str, Any]]:
    """Get evidence assessment matrix from NB8 artifacts."""
    assessments = _load_nb8_evidence_assessments(case_id)
    if assessments:
        return assessments
    return []