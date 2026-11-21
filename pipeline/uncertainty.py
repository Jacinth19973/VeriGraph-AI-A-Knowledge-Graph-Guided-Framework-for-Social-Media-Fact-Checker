"""VeriGraph AI — Uncertainty Analysis."""
from __future__ import annotations
import numpy as np
from typing import Any, Dict


def compute_uncertainty(specialists: Dict[str, Any],
                        evidence_items: list,
                        narrative_diag: Dict[str, Any] = None) -> Dict[str, Any]:
    claim = specialists.get("claim_verification", {})
    ev = specialists.get("evidence_verification", {})

    claim_assessment = (claim.get("assessment") or "insufficient").lower()
    items = [x for x in ev.get("evidence_assessments", []) if x.get("status") == "completed"]
    support = sum(1 for x in items if x.get("stance") == "supports")
    refute = sum(1 for x in items if x.get("stance") == "refutes")

    if support + refute > 0:
        evidence_stance = (support - refute) / (support + refute)
    else:
        evidence_stance = None

    claim_stance_map = {"supports": 1, "refutes": -1, "mixed": 0}
    claim_stance = claim_stance_map.get(claim_assessment)

    if claim_stance is not None and evidence_stance is not None:
        agent_disagreement = abs(claim_stance - evidence_stance) / 2
    elif claim_assessment == "insufficient":
        agent_disagreement = 0.65
    else:
        agent_disagreement = 0.50

    nb7_conflict = 0.0
    if narrative_diag:
        cc = float(narrative_diag.get("conflict_signal_count", 0))
        sc = float(narrative_diag.get("support_signal_count", 0))
        nb7_conflict = (2 * min(cc, sc) / (cc + sc)) if cc and sc else 0.0

    nb8_conflict = (2 * min(support, refute) / max(1, support + refute)) if support and refute else 0
    evidence_conflict = max(nb7_conflict, nb8_conflict)

    creds = [float(x.get("credibility_score", 0.5)) for x in evidence_items[:6]]
    low_source_cred = 1 - float(np.mean(creds)) if creds else 0.75

    temporal = specialists.get("temporal_location_consistency", {})
    temp_assessment = (temporal.get("overall_assessment") or "insufficient").lower()
    temporal_inconsistency = {"inconsistent": 1, "mixed": 0.5, "consistent": 0, "insufficient": 0.5}.get(temp_assessment, 0.5)

    suff = (narrative_diag or {}).get("overall_evidence_sufficiency", "medium")
    evidence_insuff = {"high": 0, "medium": 0.5, "low": 1}.get(suff, 0.5)
    evidence_insuff = max(evidence_insuff, 1 - len(items) / max(1, len(evidence_items)))

    weights = {"agent_disagreement": 0.3, "evidence_conflict": 0.25,
               "low_source_credibility": 0.15, "temporal_location_inconsistency": 0.15,
               "evidence_insufficiency": 0.15}

    weighted = float(np.clip(
        weights["agent_disagreement"] * agent_disagreement +
        weights["evidence_conflict"] * evidence_conflict +
        weights["low_source_credibility"] * low_source_cred +
        weights["temporal_location_inconsistency"] * temporal_inconsistency +
        weights["evidence_insufficiency"] * evidence_insuff, 0, 1))

    priority = "high" if weighted >= 0.67 else "medium" if weighted >= 0.34 else "low"

    return {"agent_disagreement": round(agent_disagreement, 3),
            "evidence_conflict": round(evidence_conflict, 3),
            "low_source_credibility": round(low_source_cred, 3),
            "temporal_location_inconsistency": round(temporal_inconsistency, 3),
            "evidence_insufficiency": round(evidence_insuff, 3),
            "weighted_uncertainty_score": round(weighted, 3),
            "debate_priority": priority}
