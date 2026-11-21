
from __future__ import annotations
import json
import re
import numpy as np
from typing import Any, Dict, List, Optional
from services.ollama_service import call_llm, extract_json_object
from config.settings import MAX_EVIDENCE_FOR_AGENTS, CONFIDENCE_DECIMALS
from services.nb8_service import load_nb8_specialists, get_evidence_assessments_from_nb8
from services.claim_conflicts import find_conflicts, unmatched_details, evidence_conflicts

COMMON_SYSTEM = '''You are one independent specialist in a multimodal social-media fact-verification research prototype.
Use ONLY the supplied information. Do not invent facts.
Rules:
1. Do not invent facts, dates, locations, people, URLs, or source properties.
2. Evidence IDs must be copied exactly.
3. Missing information must be reported as uncertainty, not converted into a contradiction.
4. Missing publication dates do not invalidate evidence; assess relevance and stance based on content.
5. The "confidence" field must be a REAL number between 0.0 and 1.0, never a placeholder 0.0.
6. Return only valid JSON matching the requested schema.'''

EVIDENCE_SCHEMA = {
    "evidence_id": "id",
    "relevance": "direct|indirect|irrelevant",
    "stance": "supports|refutes|mixed|neutral",
    "quality": "strong|moderate|weak",
    "confidence": 0.0,
    "reason": ""
}

CLAIM_SCHEMA = {
    "agent": "claim_verification",
    "assessment": "supports|refutes|mixed|insufficient",
    "confidence": 0.0,
    "supporting_evidence_ids": [],
    "refuting_evidence_ids": [],
    "key_findings": [],
    "reasoning": ""
}


def _clamp(x):
    return float(np.clip(float(x) if x == x else 0.0, 0, 1))


def _normalize(x):
    s = str(x or "").strip().lower()
    return re.sub(r"[^a-z0-9_ -]", "", s).replace(" ", "_")


def run_evidence_verification(evidence_items: List[Dict], claim: str) -> Dict[str, Any]:
    """
    Evaluates individual evidence items against the claim.
    Always includes rich scores (claim_relevance_score, entity_match_score,
    temporal_match_score, credibility_score) so downstream UI and judge can inspect them.
    """
    results = []
    for e in evidence_items[:MAX_EVIDENCE_FOR_AGENTS]:
        eid = e.get("evidence_id", "")
        rel = float(e.get("claim_relevance_score", e.get("relevance_score", 0.5)))
        ent = float(e.get("entity_match_score", 0.5))
        temp = float(e.get("temporal_match_score", 0.5))
        cred = float(e.get("credibility_score", 0.5))
        vis_score = float(e.get("visual_similarity_score", 0.5))

        task = (
            f"Claim: {claim}\n"
            f"Evidence {eid} ({e.get('domain', '')}): {str(e.get('title', ''))[:160]} - {str(e.get('snippet', ''))[:320]}\n"
            "Decide the evidence's stance toward the claim. If it describes the SAME event with a different date or "
            "a different place than the claim, the stance is 'refutes'. If it does not address the claim's specific "
            "details, the stance is 'neutral'.\n"
            f"Reply with ONLY this JSON (reason at most 20 words): {json.dumps(EVIDENCE_SCHEMA)}"
        )
        try:
            raw = call_llm(COMMON_SYSTEM, task, num_predict=160)
            if raw.get("error"):
                raise RuntimeError(f"Ollama {raw.get('error')} after {raw.get('generation_duration')}s")
            try:
                obj = extract_json_object(raw.get("content") or "")
            except ValueError:
                raise ValueError(f"reply was not valid JSON; it starts: {(raw.get('content') or '')[:120]!r}")
            if str(obj.get("stance", "")).strip().lower() not in ("supports", "refutes", "neutral"):
                raise ValueError("LLM reply has no valid 'stance' field")
            obj["evidence_id"] = eid
            obj["status"] = "completed"
            obj["method"] = "llm"
            _conflicts = evidence_conflicts(claim, e)
            obj["conflicts"] = _conflicts
            if _conflicts and str(obj.get("stance")).lower() == "supports":
                obj["stance"] = "refutes"
                obj["method"] = "llm+conflict_check"
                obj["reason"] = _conflicts[0]["detail"]
            if (str(obj.get("stance")).lower() == "refutes" and not _conflicts
                    and float(e.get("event_match_score", 1) or 0) == 0):
                obj["stance"] = "neutral"
                obj["method"] = "llm+event_check"
                obj["reason"] = ("Not the claimed event, so it cannot refute the claim: "
                                 + str(obj.get("reason", ""))[:140])
            obj["confidence"] = _clamp(obj.get("confidence", rel))
            # Enrich with numerical score attribution
            obj["claim_relevance_score"] = round(rel, CONFIDENCE_DECIMALS)
            obj["entity_match_score"] = round(ent, CONFIDENCE_DECIMALS)
            obj["temporal_match_score"] = round(temp, CONFIDENCE_DECIMALS)
            obj["credibility_score"] = round(cred, CONFIDENCE_DECIMALS)
            obj["visual_similarity_score"] = round(vis_score, CONFIDENCE_DECIMALS)
            results.append(obj)
        except Exception as exc:
            # Deterministic heuristic fallback based on calibrated evidence scores
            ev_title = str(e.get("title", "")).lower()
            ev_snippet = str(e.get("snippet", "")).lower()
            combined = ev_title + " " + ev_snippet

            is_refute = any(w in combined for w in ("false", "fake", "hoax", "debunk", "misleading", "fabricated", "untrue", "denied"))

            conflict_text = f"{e.get('title', '')} {e.get('snippet', '')}"
            conflicts = evidence_conflicts(claim, e)
            missing = unmatched_details(claim, conflict_text)

            if is_refute or temp < 0.2 or conflicts:
                stance = "refutes"
                quality = "strong" if rel > 0.6 else "moderate"
                conf = round(min(0.95, max(0.65, rel * 0.7 + cred * 0.3)), 2)
                reason = (conflicts[0]["detail"] if conflicts
                          else "Text contains direct refuting signals or incompatible temporal markers")
            elif missing:
                stance = "neutral"
                quality = "weak"
                conf = 0.4
                reason = "Evidence does not mention the claim's " + ", ".join(missing)
            elif rel >= 0.45 and ent >= 0.35:
                stance = "supports"
                quality = "strong" if rel >= 0.65 else "moderate"
                conf = round(min(0.96, rel * 0.6 + cred * 0.4), 2)
                reason = "Content corroborates focal claim entities and predicate actions"
            else:
                stance = "neutral"
                quality = "weak"
                conf = 0.35
                reason = "Peripheral topic without conclusive affirmative or refuting stance"

            results.append({
                "evidence_id": eid,
                "stance": stance,
                "relevance": "direct" if rel >= 0.50 else "indirect" if rel >= 0.30 else "irrelevant",
                "quality": quality,
                "confidence": conf,
                "status": "completed",
                "claim_relevance_score": round(rel, CONFIDENCE_DECIMALS),
                "entity_match_score": round(ent, CONFIDENCE_DECIMALS),
                "temporal_match_score": round(temp, CONFIDENCE_DECIMALS),
                "credibility_score": round(cred, CONFIDENCE_DECIMALS),
                "visual_similarity_score": round(vis_score, CONFIDENCE_DECIMALS),
                "reason": reason,
                "method": "heuristic_fallback",
                "conflicts": conflicts,
                "llm_error": str(exc)[:240],
            })

    supports = sum(1 for r in results if r.get("stance") == "supports")
    refutes = sum(1 for r in results if r.get("stance") == "refutes")
    valid = [r for r in results if r.get("status") == "completed"]

    if supports > 0 and refutes > 0:
        overall = "mixed"
    elif supports > 0 and supports >= refutes:
        overall = "strong"
    elif refutes > 0:
        overall = "refuted"
    elif any(r.get("confidence", 0) > 0.4 for r in valid):
        overall = "moderate"
    else:
        overall = "insufficient"

    avg_conf = float(np.mean([r.get("confidence", 0) for r in valid])) if valid else 0.0
    coverage = len(valid) / max(1, len(evidence_items))

    return {
        "agent": "evidence_verification",
        "status": "completed",
        "overall_assessment": overall,
        "confidence": round(0.65 * avg_conf + 0.35 * coverage, CONFIDENCE_DECIMALS),
        "supporting_evidence_ids": [r["evidence_id"] for r in results if r.get("stance") == "supports" and r.get("evidence_id")],
        "refuting_evidence_ids": [r["evidence_id"] for r in results if r.get("stance") == "refutes" and r.get("evidence_id")],
        "evidence_assessments": results,
        "key_findings": [f"supports={supports}", f"refutes={refutes}", f"evaluated={len(valid)}"],
        "uncertainty_flags": ["neutral_evidence_present"] if any(r.get("stance") == "neutral" for r in results) else []
    }


def run_claim_verification(evidence_result: Dict, evidence_items: List[Dict], claim: str) -> Dict[str, Any]:
    """Synthesizes evidence checks into claim-level verification verdict."""
    evidence_summary = json.dumps(evidence_result.get("evidence_assessments", [])[:6], default=str)[:800]
    task = f"Assess the claim based on evidence assessments.\nClaim: {claim}\nEvidence assessments: {evidence_summary}\nReturn JSON with assessment, confidence, supporting/refuting evidence IDs."
    try:
        raw = call_llm(COMMON_SYSTEM, task, num_predict=420)
        obj = extract_json_object(raw.get("content") or "")
        obj["agent"] = "claim_verification"
        obj["status"] = "completed"
        obj["assessment"] = _normalize(obj.get("assessment", "insufficient"))
        if obj["assessment"] not in {"supports", "refutes", "mixed", "insufficient"}:
            obj["assessment"] = "insufficient"
        obj["confidence"] = _clamp(obj.get("confidence", 0.5))
        _ass = evidence_result.get("evidence_assessments", [])
        obj["supporting_evidence_ids"] = [r["evidence_id"] for r in _ass if r.get("stance") == "supports" and r.get("evidence_id")]
        obj["refuting_evidence_ids"] = [r["evidence_id"] for r in _ass if r.get("stance") == "refutes" and r.get("evidence_id")]
        if obj["assessment"] == "supports" and obj["refuting_evidence_ids"] and not obj["supporting_evidence_ids"]:
            obj["assessment"], obj["reasoning"] = "refutes", "Adjusted: every assessed evidence item refutes the claim."
        elif obj["assessment"] == "refutes" and obj["supporting_evidence_ids"] and not obj["refuting_evidence_ids"]:
            obj["assessment"], obj["reasoning"] = "supports", "Adjusted: every assessed evidence item supports the claim."
        return obj
    except Exception:
        assessments = evidence_result.get("evidence_assessments", [])
        sup_ids = [r.get("evidence_id") for r in assessments if r.get("stance") == "supports" and r.get("evidence_id")]
        ref_ids = [r.get("evidence_id") for r in assessments if r.get("stance") == "refutes" and r.get("evidence_id")]

        if ref_ids and len(ref_ids) >= len(sup_ids):
            assessment = "refutes"
            conf = 0.88
            reasoning = f"Corroborated refutation identified by {len(ref_ids)} specialist evidence assessments."
        elif sup_ids and not ref_ids:
            assessment = "supports"
            conf = 0.86
            reasoning = f"Corroborated affirmative support identified by {len(sup_ids)} specialist evidence assessments."
        elif sup_ids and ref_ids:
            assessment = "mixed"
            conf = 0.65
            reasoning = f"Divergent evidence detected: {len(sup_ids)} supporting vs {len(ref_ids)} refuting items."
        else:
            assessment = "insufficient"
            conf = 0.25
            reasoning = "Insufficient specialist evidence signals to substantiate or debunk."

        return {
            "agent": "claim_verification",
            "status": "completed",
            "assessment": assessment,
            "confidence": conf,
            "supporting_evidence_ids": sup_ids,
            "refuting_evidence_ids": ref_ids,
            "key_findings": [f"Deterministic consensus: {len(sup_ids)} supporting, {len(ref_ids)} refuting"],
            "reasoning": reasoning
        }


def run_source_credibility_specialist(evidence_items: List[Dict], claim: str) -> Dict[str, Any]:
    """
    Evaluates publishing source domains, establishing credibility tiers,
    institutional authority, and editorial reliability.
    """
    assessments = []
    tier_counts = {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0, "unclassified": 0}

    for e in evidence_items[:MAX_EVIDENCE_FOR_AGENTS]:
        eid = e.get("evidence_id", "")
        domain = (e.get("domain") or "").lower()
        source = e.get("source") or domain or "unknown"
        cred_score = float(e.get("credibility_score", 0.5))

        from services.source_credibility import classify_source
        _c = classify_source(domain, e.get("url", ""))
        tier, score = _c["label"], _c["score"]
        tier_counts[f"tier_{_c['tier']}" if _c["tier"] else "unclassified"] += 1

        assessments.append({
            "evidence_id": eid,
            "source_name": source,
            "domain": domain,
            "credibility_tier": tier,
            "credibility_score": round(score, CONFIDENCE_DECIMALS),
            "is_authoritative": score >= 0.70
        })

    avg_score = float(np.mean([a["credibility_score"] for a in assessments])) if assessments else 0.50
    overall = "high" if avg_score >= 0.75 else "moderate" if avg_score >= 0.55 else "low"

    return {
        "agent": "source_credibility",
        "status": "completed",
        "overall_assessment": overall,
        "confidence": round(avg_score, CONFIDENCE_DECIMALS),
        "source_assessments": assessments,
        "key_findings": [
            f"Tier 1 authoritative: {tier_counts['tier_1']}",
            f"Tier 2 mainstream: {tier_counts['tier_2']}",
            f"Tier 3 regional: {tier_counts['tier_3']}",
            f"Tier 4 unverified: {tier_counts['tier_4']}",
            f"Unclassified (no source): {tier_counts['unclassified']}"
        ]
    }


def run_temporal_location_specialist(evidence_items: List[Dict], claim: str) -> Dict[str, Any]:
    """
    Evaluates temporal and geospatial alignment between the claim anchors
    and the retrieved evidence, pinpointing out-of-context temporal reuse.
    """
    claim_years = re.findall(r"\b(?:19|20)\d{2}\b", claim)
    target_year = claim_years[0] if claim_years else None

    temporal_findings = []
    location_findings = []
    contradiction_ids = []
    location_conflict_ids = []

    for e in evidence_items[:MAX_EVIDENCE_FOR_AGENTS]:
        eid = e.get("evidence_id", "")
        ev_text = f"{e.get('title', '')} {e.get('snippet', '')}".lower()
        ev_years = re.findall(r"\b(?:19|20)\d{2}\b", ev_text)
        temp_score = float(e.get("temporal_match_score", 0.5))
        loc_score = float(e.get("location_match_score", 0.5))

        has_year_conflict = False
        if target_year and ev_years:
            if target_year not in ev_years:
                has_year_conflict = True
                contradiction_ids.append(eid)

        _conflicts = evidence_conflicts(claim, e)
        date_conflict = next((c for c in _conflicts if c["type"] == "date"), None)
        place_conflict = next((c for c in _conflicts if c["type"] == "location"), None)
        if date_conflict and eid not in contradiction_ids:
            contradiction_ids.append(eid)
        if place_conflict:
            location_conflict_ids.append(eid)

        temporal_findings.append({
            "evidence_id": eid,
            "claimed_year": target_year or "unspecified",
            "evidence_years": ev_years,
            "temporal_match_score": round(temp_score, CONFIDENCE_DECIMALS),
            "is_anachronism": has_year_conflict or bool(date_conflict),
            "conflict": date_conflict["detail"] if date_conflict else "",
        })

        location_findings.append({
            "evidence_id": eid,
            "location_match_score": round(loc_score, CONFIDENCE_DECIMALS),
            "status": "conflict" if place_conflict else ("aligned" if loc_score >= 0.5 else "unverified"),
            "conflict": place_conflict["detail"] if place_conflict else "",
        })

    anachronism_count = len(contradiction_ids)
    if anachronism_count and location_conflict_ids:
        overall = "temporal_location_mismatch_detected"
    elif anachronism_count:
        overall = "temporal_mismatch_detected"
    elif location_conflict_ids:
        overall = "location_mismatch_detected"
    else:
        overall = "consistent"
    conf = 0.90 if (anachronism_count > 0 or location_conflict_ids) else 0.82

    return {
        "agent": "temporal_location_consistency",
        "status": "completed",
        "overall_assessment": overall,
        "confidence": conf,
        "temporal_findings": temporal_findings,
        "location_findings": location_findings,
        "contradiction_evidence_ids": contradiction_ids,
        "key_findings": [
            f"Claimed year anchor: {target_year or 'None'}",
            f"Temporal mismatch contradictions: {anachronism_count}",
            f"Evaluated evidence items: {len(temporal_findings)}"
        ]
    }


def run_specialists(evidence_items: List[Dict], claim: str, case_id: str = None) -> Dict[str, Any]:
    """Orchestrates all four specialist verification agents."""
    ev_result = run_evidence_verification(evidence_items, claim)
    
    claim_result = run_claim_verification(ev_result, evidence_items, claim)

    source_result = run_source_credibility_specialist(evidence_items, claim)

    temporal_result = run_temporal_location_specialist(evidence_items, claim)

    return {
        "evidence_verification": ev_result,
        "claim_verification": claim_result,
        "source_credibility": source_result,
        "temporal_location_consistency": temporal_result
    }
