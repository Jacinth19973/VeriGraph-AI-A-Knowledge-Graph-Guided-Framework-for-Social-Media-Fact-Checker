"""VeriGraph AI — Explainability module (factor cards, no chain-of-thought)."""
from __future__ import annotations
from typing import Any, Dict, List


def _safe(v, d=0.0):
    try:
        f = float(v)
        return f if f == f else d
    except Exception:
        return d


_GENERIC_WORDS = {
    "the", "this", "that", "was", "were", "held", "summit", "conference", "meeting", "forum",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _claim_entities(claim: str) -> List[str]:
    """Named entities and dates stated in the claim (acronyms, proper names, month + year)."""
    import re
    found: List[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if term and term.lower() not in _GENERIC_WORDS and term not in found:
            found.append(term)

    for m in re.findall(r"\b[A-Z]{2,}\b", claim or ""):
        add(m)
    for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim or ""):
        add(m)
    for m in re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", claim or ""):
        add(m)
    return found[:10]


def _name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("label") or "")
    return str(item)


def _temporal_lines(findings: List[Dict], judge: Dict[str, Any]) -> List[str]:
    lines = []
    note = (judge.get("temporal_analysis") or {}).get("note")
    if note:
        lines.append(f"Claim timing: {note}")
    for f in findings[:5]:
        years = ", ".join(sorted(set(f.get("evidence_years", [])))) or "no year found"
        status = "anachronism" if f.get("is_anachronism") else "consistent"
        extra = f" CONFLICT: {f['conflict']}" if f.get("conflict") else ""
        lines.append(f"{f.get('evidence_id', '?')}: claim year {f.get('claimed_year', 'unspecified')}, "
                     f"evidence {years} -> {status} (match {f.get('temporal_match_score')}){extra}")
    return lines or ["No temporal evidence was assessed"]


def _location_lines(findings: List[Dict]) -> List[str]:
    lines = [f"{f.get('evidence_id', '?')}: location match {f.get('location_match_score')} ({f.get('status', 'n/a')})"
             + (f" CONFLICT: {f['conflict']}" if f.get("conflict") else "")
             for f in findings[:5]]
    return lines or ["No location evidence was assessed"]


def build_explanations(specialists: Dict[str, Any],
                        uncertainty: Dict[str, Any],
                        judge: Dict[str, Any],
                        evidence_items: List[Dict],
                        kg_context: Dict[str, Any],
                        debate_log: List[Dict],
                        claim: str = "") -> Dict[str, Any]:
    factors = []
    support_ids, refute_ids, neutral_ids = [], [], []

    for e in specialists.get("evidence_verification", {}).get("evidence_assessments", [])[:10]:
        eid = e.get("evidence_id", "")
        stance = e.get("stance", "neutral")
        if stance in ("supports",): support_ids.append(eid)
        elif stance in ("refutes",): refute_ids.append(eid)
        else: neutral_ids.append(eid)

    agent_summary = []
    for agent in ("claim_verification", "evidence_verification", "source_credibility",
                  "temporal_location_consistency"):
        r = specialists.get(agent, {})
        factors.append({
            "factor": agent.replace("_", " ").title(),
            "assessment": r.get("assessment") or r.get("overall_assessment") or "insufficient",
            "confidence": _safe(r.get("confidence")),
            "status": r.get("status", "completed"),
            "detail": "; ".join(r.get("key_findings", [])[:3]),
        })
        if agent == "claim_verification" and r.get("reasoning"):
            agent_summary.append({"agent": agent, "headline": r.get("reasoning", "")[:220]})
        elif r.get("reasoning"):
            agent_summary.append({"agent": agent, "headline": r.get("reasoning", "")[:220]})

    factors.append({
        "factor": "Overall Uncertainty",
        "assessment": uncertainty.get("debate_priority", "low"),
        "confidence": 1 - _safe(uncertainty.get("weighted_uncertainty_score")),
        "status": "computed",
        "detail": f"Weighted uncertainty score {_safe(uncertainty.get('weighted_uncertainty_score'))}",
    })

    provenance = []
    for e in evidence_items[:6]:
        provenance.append({
            "evidence_id": e.get("evidence_id", ""),
            "channel": e.get("retrieval_channel", ""),
            "provider": e.get("provider", ""),
            "domain": e.get("domain", "unknown"),
            "url": e.get("url", ""),
            "relevance_score": _safe(e.get("relevance_score")),
            "cluster_id": e.get("cluster_id", -1),
        })

    kg_findings = []
    if kg_context.get("events"):
        kg_findings.append("KG events mention: " + ", ".join(kg_context["events"][:5]))
    if kg_context.get("locations"):
        kg_findings.append("KG locations: " + ", ".join(kg_context["locations"][:5]))
    if kg_context.get("entities"):
        kg_findings.append("KG entities: " + ", ".join(kg_context["entities"][:8]))
    if not kg_context.get("events") and not kg_context.get("locations") and not kg_context.get("entities"):
        kg_findings.append("Knowledge Graph is sparse for this claim; no corroborating nodes found.")

    debate_findings = []
    for t in debate_log:
        debate_findings.append(
            f"Round {t.get('round')} {t.get('speaker')}: position={t.get('position')}, "
            f"confidence={_safe(t.get('confidence'))}, arguments={t.get('args', 0)}")

    uncertainty_factors = [f"{k.replace('_', ' ')} = {v:.2f}"
                           for k, v in uncertainty.items()
                           if k != "debate_priority" and isinstance(v, (int, float))]

    knowledge_gaps = []
    if not kg_context.get("events"): knowledge_gaps.append("No KG events linked to this claim")
    if not kg_context.get("persons"): knowledge_gaps.append("No KG persons linked to this claim")
    if not kg_context.get("locations"): knowledge_gaps.append("No KG locations linked to this claim")
    if not kg_context.get("sources"): knowledge_gaps.append("No KG sources linked to this claim")
    if len(evidence_items) < 3: knowledge_gaps.append("Fewer than 3 evidence items retrieved")
    if not kg_findings or kg_findings[0].startswith("Knowledge Graph is sparse"):
        knowledge_gaps.append("KG expansion opportunity detected")

    verdict = judge.get("final_verdict", "Insufficient")
    verdict_reason = judge.get("reasoning", "")

    if verdict == "True":
        summary = "The claim is supported by converging evidence from the retrieval, specialist and debate stages."
    elif verdict == "False":
        summary = "The claim is contradicted by the weight of evidence and specialist assessments."
    elif str(verdict).strip().lower() == "misleading":
        summary = "The claim mixes facts with misleading framing; evidence is partial or ambiguous."
    else:
        summary = "There is not enough relevant evidence to support or contradict the claim."

    return {
        "summary": summary,
        "verdict_reason": verdict_reason,
        "supporting_evidence": support_ids,
        "contradicting_evidence": refute_ids,
        "neutral_evidence": neutral_ids,
        "key_entities": list(dict.fromkeys(_claim_entities(claim) + [n for n in (_name(e) for e in kg_context.get("entities", [])) if n])),
        "temporal_findings": _temporal_lines(specialists.get("temporal_location_consistency", {}).get("temporal_findings", []), judge),
        "location_findings": _location_lines(specialists.get("temporal_location_consistency", {}).get("location_findings", [])),
        "source_findings": [f"{s.get('domain', '')}: {s.get('credibility_tier', '')} (score {s.get('credibility_score')})"
                            for s in specialists.get("source_credibility", {}).get("source_assessments", [])[:5]] or ["No sources were assessed"],
        "kg_findings": kg_findings,
        "debate_findings": debate_findings,
        "uncertainty_factors": uncertainty_factors,
        "knowledge_gaps": knowledge_gaps,
        "factor_cards": factors,
        "provenance_chain": provenance,
        "agent_summary": agent_summary,
        "natural_language_summary": f"{summary} Specialist average confidence: "
                                    f"{_safe(specialists.get('claim_verification', {}).get('confidence')):.2f}; "
                                    f"judge confidence {_safe(judge.get('confidence')):.2f}.",
        "confidence_definition": "Verification Confidence",
    }