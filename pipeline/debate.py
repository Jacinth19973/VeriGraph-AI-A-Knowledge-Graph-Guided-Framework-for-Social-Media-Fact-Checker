"""VeriGraph AI — Adaptive Multi-Agent Debate (NB9 logic)."""
from __future__ import annotations
import json
from typing import Any, Dict, List

import numpy as np
from services.ollama_service import call_llm, extract_json_object
from services.claim_conflicts import find_conflicts, evidence_conflicts
from config.settings import (
    ROUNDS_BY_PRIORITY,
    MAX_DEBATE_HISTORY_CHARS,
    MAX_TURN_CHARS,
    CONFIDENCE_DECIMALS,
    OLLAMA_TIMEOUT,
    OLLAMA_MODEL,
)

# Judge calls are the longest single LLM invocation in the pipeline; the shared
# OLLAMA_TIMEOUT is meant for short specialist prompts, so give the judge its own
# budget instead of cutting it off mid-generation.
JUDGE_TIMEOUT = max(OLLAMA_TIMEOUT, 300)


def _c(x):
    import re
    if x is None: return ""
    return re.sub(r"\s+", " ", str(x).strip())[:500] if str(x).strip() not in ("nan", "none") else ""


def _clamp(x):
    import numpy as np
    return round(float(np.clip(float(x) if x == x else 0, 0, 1)), CONFIDENCE_DECIMALS)


def _truncate_history(history: List[Dict]) -> str:
    lines, total = [], 0
    for turn in history:
        content = str(turn.get("content", ""))
        if len(content) > MAX_TURN_CHARS:
            content = content[:MAX_TURN_CHARS] + "..."
        line = f"[Round {turn.get('round','?')}] {turn.get('speaker','?').upper()}: {content}"
        total += len(line)
        if total > MAX_DEBATE_HISTORY_CHARS:
            lines.append("...[earlier rounds omitted]")
            break
        lines.append(line)
    return "\n".join(lines)


SUPPORTER_SYSTEM = '''You are the SUPPORTER (affirmative) agent in a fact-verification debate.
Argue that the claim is supported by the available evidence.
1. Only cite evidence from the supplied package. Use no outside knowledge; if nothing supports the claim, say so and keep confidence low. Keep each argument under 60 words and give at most 2 arguments.
2. Respond point-by-point to the Skeptic's arguments.
3. confidence = how confident you are in YOUR position (0.0-1.0, never 0.0).
4. Return only valid JSON.'''

SKEPTIC_SYSTEM = '''You are the SKEPTIC (negative) agent in a fact-verification debate.
Argue that the claim is NOT supported, or that the evidence is insufficient.
1. Use only the supplied evidence and KG facts; no outside knowledge. Judge dates only against TODAY as given in the prompt, never against your training cutoff. Use KG structure to identify factual gaps or entity mismatches. Keep each argument under 60 words and give at most 2 arguments.
2. Respond point-by-point to the Supporter's arguments.
3. confidence = how confident you are in YOUR position (0.0-1.0).
4. Return only valid JSON.'''

SUPPORTER_SCHEMA = {"agent": "supporter", "round": 1, "position": "supports|does_not_support",
                    "confidence": 0.0, "arguments": [{"id": "arg_s1", "type": "evidence_citation",
                    "content": "...", "evidence_ids": [], "strength": "strong|moderate|weak"}],
                    "response_to_skeptic": [], "key_evidence_summary": ""}

SKEPTIC_SCHEMA = {"agent": "skeptic", "round": 1, "position": "supports|does_not_support",
                  "confidence": 0.0, "arguments": [{"id": "arg_sk1", "type": "insufficient_evidence",
                  "content": "...", "kg_facts": [], "strength": "strong|moderate|weak"}],
                  "response_to_supporter": [], "key_finding_summary": ""}

JUDGE_SCHEMA = {"agent": "judge", "final_verdict": "True|False|Misleading|Insufficient",
                "confidence": 0.0, "supporter_final_confidence": 0.0,
                "skeptic_final_confidence": 0.0, "reasoning": "",
                "key_evidence_cited": [], "key_kg_gaps_cited": []}
JUDGE_SYSTEM = '''You are the JUDGE in a fact-verification debate. Completely neutral.

1. Read the debate transcript and all evidence. Use ONLY the supplied evidence; no outside knowledge.
2. Every factual conclusion must cite the evidence_ids that support it.
3. Respect event dates. The EVENT date is not the PUBLICATION date. A missing publication date does not
   disqualify evidence.
4. Choose exactly one verdict:
   - "True": reliable evidence confirms the claim's key details (event, place and date).
   - "False": reliable evidence about the SAME event contradicts a key detail of the claim (a different date,
     a different place or a different outcome). Evidence that the event took place elsewhere or at another
     time makes the claim False, not Insufficient.
   - "Misleading": the claim is partly accurate but distorts or omits important context.
   - "Insufficient": there is little or no relevant evidence either way.
5. "confidence" = how confident you are in YOUR chosen verdict, from 0.0 to 1.0. Never answer 1.0.
6. Return only valid JSON.'''


def run_debate(evidence_items: List[Dict], specialist_results: Dict,
               kg_context: Dict, uncertainty: Dict, claim: str) -> Dict[str, Any]:
    priority = uncertainty.get("debate_priority", "low")
    max_rounds = ROUNDS_BY_PRIORITY.get(priority, 2)
    max_rounds = max(1, min(max_rounds, int(__import__("os").getenv("VERIGRAPH_DEBATE_ROUNDS", max_rounds))))

    from services.time_service import (
        analyze_claim_temporality, temporal_summary_text, utc_now, india_now,
    )
    _today = f"{utc_now().date().isoformat()} UTC ({india_now().strftime('%B %d, %Y')} IST)"
    _timing = temporal_summary_text(analyze_claim_temporality(claim))
    time_ctx = f"TODAY (system clock): {_today}\nCLAIM TIMING: {_timing}\n\n"

    supporter_bundle = {"claim": claim, "evidence": evidence_items[:10],
                        "specialist_summary": {k: {"assessment": v.get("assessment") or v.get("overall_assessment"),
                                                   "confidence": v.get("confidence", 0)} for k, v in specialist_results.items()}}
    skeptic_bundle = {"claim": claim, "evidence": evidence_items[:10], "kg_context": kg_context, "uncertainty": uncertainty}

    history = []
    debate_log = []
    stop_reason = "max_rounds_reached"

    for rnd in range(1, max_rounds + 1):
        history_str = _truncate_history(history)
        stage_s = "Round 1 — OPENING." if rnd == 1 else f"Round {rnd} — REBUTTAL. Answer the Skeptic's latest argument."
        user_prompt = f"{time_ctx}STAGE: {stage_s}\n\nSUPPORTER CONTEXT:\n{json.dumps(supporter_bundle, default=str)[:3000]}\n\nDEBATE HISTORY:\n{history_str}\n\nSchema: {json.dumps(SUPPORTER_SCHEMA)}\nRound: {rnd}\nReturn ONLY JSON."
        try:
            raw = call_llm(SUPPORTER_SYSTEM, user_prompt, num_predict=400)
            if raw.get("error"):
                raise RuntimeError(f"Ollama {raw['error']} after {raw.get('generation_duration')}s")
            resp = extract_json_object(raw.get("content") or "")
            _check_debater_reply(resp)
            resp["round"] = rnd
            resp["agent"] = "supporter"
            resp["confidence"] = _clamp(resp.get("confidence", 0))
        except Exception as e:
            resp = {"agent": "supporter", "round": rnd, "position": "unavailable",
                    "confidence": 0.0, "arguments": [], "key_evidence_summary": f"Supporter call failed: {e}"}

        history.append({"round": rnd, "speaker": "supporter", "content": json.dumps(resp, default=str)})
        debate_log.append({"round": rnd, "speaker": "supporter", "position": resp.get("position", ""),
                           "confidence": resp.get("confidence", 0),
                           "arguments": resp.get("arguments", []),
                           "key_evidence_summary": resp.get("key_evidence_summary", ""),
                           "key_summary": resp.get("key_evidence_summary", "") or resp.get("key_summary", ""),
                           "args": len(resp.get("arguments", []))})

        history_str = _truncate_history(history)
        stage_k = "Round 1 — OPENING." if rnd == 1 else f"Round {rnd} — REBUTTAL. Answer the Supporter's latest argument."
        user_prompt = f"{time_ctx}STAGE: {stage_k}\n\nSKEPTIC CONTEXT:\n{json.dumps(skeptic_bundle, default=str)[:4000]}\n\nDEBATE HISTORY:\n{history_str}\n\nSchema: {json.dumps(SKEPTIC_SCHEMA)}\nRound: {rnd}\nReturn ONLY JSON."
        try:
            raw = call_llm(SKEPTIC_SYSTEM, user_prompt, num_predict=400)
            if raw.get("error"):
                raise RuntimeError(f"Ollama {raw['error']} after {raw.get('generation_duration')}s")
            resp2 = extract_json_object(raw.get("content") or "")
            _check_debater_reply(resp2)
            resp2["round"] = rnd
            resp2["agent"] = "skeptic"
            resp2["confidence"] = _clamp(resp2.get("confidence", 0))
        except Exception as e:
            resp2 = {"agent": "skeptic", "round": rnd, "position": "unavailable",
                     "confidence": 0.0, "arguments": [], "key_finding_summary": f"Skeptic call failed: {e}"}

        history.append({"round": rnd, "speaker": "skeptic", "content": json.dumps(resp2, default=str)})
        debate_log.append({"round": rnd, "speaker": "skeptic", "position": resp2.get("position", ""),
                           "confidence": resp2.get("confidence", 0),
                           "arguments": resp2.get("arguments", []),
                           "key_finding_summary": resp2.get("key_finding_summary", ""),
                           "key_summary": resp2.get("key_finding_summary", "") or resp2.get("key_summary", ""),
                           "args": len(resp2.get("arguments", []))})

        if rnd > 1:
            positions = [resp.get("position"), resp2.get("position")]
            if positions[0] == positions[1] and positions[0] in ("supports", "does_not_support"):
                stop_reason = f"early_agreement_round_{rnd}"
                break

    return {"max_rounds": max_rounds, "actual_rounds": len(debate_log) // 2,
            "stop_reason": stop_reason, "turns": debate_log, "history": history}


def run_judge(evidence_items: List, specialist_results: Dict,
              debate_history: List[Dict], kg_context: Dict,
              uncertainty: Dict, claim: str, consensus: bool = True) -> Dict[str, Any]:
    history_str = _truncate_history(debate_history)
    from services.time_service import (
        analyze_claim_temporality, temporal_summary_text, utc_now, utc_date, india_now,
    )

    # Temporal analysis is computed at runtime from the real clock. The current
    # date is never hard-coded.
    current_date = utc_date().strftime("%B %d, %Y")
    current_date_iso = utc_now().date().isoformat()
    current_date_ist = india_now().strftime("%B %d, %Y")
    temporal = analyze_claim_temporality(claim)
    temporal_text = temporal_summary_text(temporal)

    # Select top-ranked evidence only to reduce prompt size (Phase 10)
    top_evidence = evidence_items[:10] if evidence_items else []
    evidence_summary = []
    for e in top_evidence:
        evidence_summary.append({
            "evidence_id": e.get("evidence_id", ""),
            "title": e.get("title", ""),
            "snippet": e.get("snippet", "")[:200] if e.get("snippet") else "",
            "provider": e.get("provider", ""),
            "publication_date": e.get("published_at", ""),
            "fusion_score": float(e.get("fusion_score", 0) or 0.0),
        })
    bundle = {"claim": claim, "evidence_count": len(evidence_items),
              "current_date": current_date,
              "current_date_iso": current_date_iso,
              "current_date_ist": current_date_ist,
              "temporal_analysis": temporal,
              "specialists": {k: {"assessment": v.get("assessment") or v.get("overall_assessment"),
                                  "confidence": v.get("confidence")} for k, v in specialist_results.items()},
              "evidence": evidence_summary,
              "uncertainty": uncertainty, "debate": history_str[:2000]}

    # Get valid evidence IDs for citation validation
    valid_ids = {e.get("evidence_id", "") for e in top_evidence if e.get("evidence_id")}

    user_prompt = (
        f"CLAIM: {claim}\n\n"
        f"TEMPORAL CONTEXT: {temporal_text}\n\n"
        f"BUNDLE:\n{json.dumps(bundle, default=str)[:3000]}\n\n"
        f"Schema: {json.dumps(JUDGE_SCHEMA)}\n"
        f"Return ONLY JSON."
    )
    judge_status = "success"
    judge_error = None
    try:
        raw = call_llm(JUDGE_SYSTEM, user_prompt, num_predict=600, timeout=JUDGE_TIMEOUT)
        import requests as _rq
        _err = raw.get("error")
        _meta = (f"model={OLLAMA_MODEL}, prompt_tokens~{raw.get('prompt_tokens')}, "
                 f"duration={raw.get('generation_duration')}s")
        if _err == "timeout":
            raise _rq.exceptions.Timeout(f"Ollama gave no answer within {JUDGE_TIMEOUT}s ({_meta})")
        if _err in ("ConnectionError", "ConnectTimeout"):
            raise _rq.exceptions.ConnectionError(f"Ollama is not reachable ({_meta})")
        if _err == "empty_response":
            raise ValueError(f"Ollama returned an empty response ({_meta})")
        if _err:
            raise RuntimeError(f"Ollama call failed: {_err} ({_meta})")
        try:
            obj = extract_json_object(raw.get("content") or "")
        except ValueError:
            _head = (raw.get("content") or "")[:200].replace("\n", " ")
            raise ValueError(f"Judge reply was not valid JSON ({_meta}); starts with: {_head!r}")
        
        if "final_verdict" not in obj:
            raise ValueError(f"Judge reply had no final_verdict (possibly cut off); keys: {list(obj)[:6]}")

        # Validate and fix final_verdict
        verdict_raw = obj.get("final_verdict")
        if isinstance(verdict_raw, str):
            verdict = verdict_raw.strip().capitalize()
        elif verdict_raw is None:
            verdict = ""
        else:
            verdict = str(verdict_raw).strip().capitalize()
        valid_verdicts = {"True", "False", "Misleading", "Insufficient"}
        if verdict not in valid_verdicts:
            # Map common variations
            verdict_lower = verdict.lower()
            if verdict_lower in {"true", "false", "misleading", "insufficient", "insufficient"}:
                # Normalize first letter capitalization
                verdict = verdict_lower[0].upper() + verdict_lower[1:]
            else:
                verdict = "Insufficient"
        
        # INSUFFICIENT handling: if confidence is 0 and there's no strong evidence, keep as INSUFFICIENT
        # but if there is evidence, set confidence based on evidence strength
        judge_confidence = _clamp(obj.get("confidence", 0))
        
        # Citation validation: remove invalid evidence IDs
        key_evidence_cited = obj.get("key_evidence_cited", [])
        # Filter to only valid evidence IDs
        filtered_evidence = [eid for eid in key_evidence_cited if eid in valid_ids]
        
        # Do NOT fabricate citations. If all citations are invalid, record
        # that the Judge failed to cite valid evidence instead of substituting
        # a top-ranked item as if it supports the claim.
        if not filtered_evidence and valid_ids:
            filtered_evidence = []
            judge_status = "degraded"
            judge_error = "Judge cited no valid evidence IDs"
        
        # Filter kg gaps cited similarly
        key_kg_gaps_cited = obj.get("key_kg_gaps_cited", [])
        
        # Compute final confidence aggregation
        # Use evidence-strength-based fallback if judge confidence is 0 but we have valid evidence
        if judge_confidence == 0 and filtered_evidence:
            # Compute evidence strength from specialist results and evidence scores
            evidence_strength = 0.0
            spec_assessments = specialist_results.get("evidence_verification", {}).get("evidence_assessments", [])
            if spec_assessments:
                supports = sum(1 for r in spec_assessments if r.get("stance") == "supports")
                refutes = sum(1 for r in spec_assessments if r.get("stance") == "refutes")
                if supports + refutes > 0:
                    evidence_strength = float(supports) / float(supports + refutes)
            
            # Also consider narrative coherence
            narr_clusters = specialist_results.get("temporal_location_consistency", {}).get("key_findings", [])
            if not judge_confidence and narr_clusters:
                judge_confidence = round(min(1.0, evidence_strength + 0.2), 2)
            
            # If still 0 and we have evidence, do NOT fabricate a minimum.
            # Return a transparent "not assessed" status instead.
            if judge_confidence == 0:
                judge_confidence = 0.0
        
        # Compute supporter/skeptic final confidences
        supporter_conf = _clamp(obj.get("supporter_final_confidence", 0))
        skeptic_conf = _clamp(obj.get("skeptic_final_confidence", 0))
        
        # Build reasoning steps for transparency
        rlow = (obj.get("reasoning") or "").lower()
        hints = []
        if any(t in rlow for t in ("is false", "not supported", "unfounded", "incorrect", "contradict", "refute", "no evidence")): hints.append("False")
        if any(t in rlow for t in ("misleading", "out of context", "deceptive")): hints.append("Misleading")
        if any(t in rlow for t in ("is true", "supported by", "confirmed by", "corroborated", "consistent with")): hints.append("True")
        
        reasoning_out = obj.get("reasoning", "") or _generate_judge_reasoning(verdict, filtered_evidence, key_kg_gaps_cited, rlow)
        dissent = ""
        adjusted_from = None
        adjusted_by = None

        # Consistency check: authoritative evidence about the same event that contradicts the claim's
        # date or place means the claim is False, even if the model answered "Insufficient".
        conflicts = _conflicting_evidence(claim, evidence_items) if consensus else []
        if conflicts and verdict == "Insufficient" and len(conflicts) >= 2:
            adjusted_from = verdict
            verdict = "False"
            adjusted_by = "conflict_check"
            judge_confidence = round(min(0.9, 0.5 + 0.1 * len(conflicts)), 2)
            filtered_evidence = list(dict.fromkeys(filtered_evidence + [eid for eid, _ in conflicts]))
            note = "; ".join(sorted({c["detail"] for _, cs in conflicts for c in cs}))
            reasoning_out = (f"{reasoning_out} [Consistency check] {len(conflicts)} authoritative evidence items "
                             f"about the same event contradict the claim: {note}. The verdict was changed from "
                             f"{adjusted_from} to False.")
            if judge_status == "degraded":
                judge_status, judge_error = "success", ""
        elif conflicts and verdict == "True":
            dissent = ("Some authoritative evidence conflicts with the claim: "
                       + "; ".join(sorted({c["detail"] for _, cs in conflicts for c in cs})))
        elif consensus and verdict == "Insufficient":
            corroborating = [] if conflicts else _corroborating_evidence(specialist_results, evidence_items)
            consensus = _stance_consensus(evidence_items, specialist_results) or {}
            refs = consensus.get("refutes", [])
            if len(refs) >= 2 and not consensus.get("supports") and any(w >= 0.7 for _, w in refs):
                adjusted_from, adjusted_by, verdict = verdict, "stance_consensus", "False"
                judge_confidence = round(min(0.9, 0.5 + 0.1 * len(refs)), 2)
                filtered_evidence = list(dict.fromkeys(filtered_evidence + [i for i, _ in refs]))
                reasoning_out = (f"{reasoning_out} [Consistency check] {len(refs)} sources, at least one authoritative, were "
                                 f"assessed as contradicting the claim and none as supporting it. The verdict was changed "
                                 f"from {adjusted_from} to False.")
                if judge_status == "degraded":
                    judge_status, judge_error = "success", ""
            elif len(corroborating) >= 2:
                adjusted_from, adjusted_by, verdict = verdict, "corroboration_check", "True"
                judge_confidence = round(min(0.9, 0.5 + 0.1 * len(corroborating)), 2)
                filtered_evidence = list(dict.fromkeys(filtered_evidence + corroborating))
                reasoning_out = (f"{reasoning_out} [Consistency check] {len(corroborating)} authoritative evidence items "
                                 f"corroborate the claim and none contradict it. The verdict was changed from "
                                 f"{adjusted_from} to True.")
                if judge_status == "degraded":
                    judge_status, judge_error = "success", ""

        judge_confidence = min(judge_confidence, 0.95)
        if verdict == "Insufficient":
            judge_confidence = min(judge_confidence, 0.5)

        turn_conf = _debate_confidences(debate_history)
        supporter_conf = supporter_conf or turn_conf.get("supporter", 0.0)
        skeptic_conf = skeptic_conf or turn_conf.get("skeptic", 0.0)

        label_alignment = {"verdict": verdict, "hints": hints, "aligned": verdict in hints}
        if adjusted_from:
            label_alignment.update({"adjusted_by": adjusted_by, "model_verdict": adjusted_from, "aligned": True})
        
        return {
            "agent": "judge",
            "final_verdict": verdict,
            "confidence": judge_confidence,
            "supporter_final_confidence": supporter_conf,
            "skeptic_final_confidence": skeptic_conf,
            "reasoning": reasoning_out,
            "dissent_notes": dissent,
            "key_evidence_cited": filtered_evidence,
            "key_kg_gaps_cited": key_kg_gaps_cited,
            "label_reasoning_alignment": label_alignment,
            "judge_status": judge_status,
            "judge_error": judge_error or "",
            "temporal_analysis": temporal,
            "current_date": current_date_iso,
            "current_date_ist": current_date_ist,
        }
    except Exception as e:
        # Judge failure must never erase retrieval results. The deterministic
        # fallback still returns a structured verdict plus an explicit status.
        print(f"[judge] Judge error: {e}")
        fallback = _judge_fallback(evidence_items, valid_ids, claim, specialist_results)
        fallback.update({
            "judge_status": _classify_judge_error(e),
            "judge_error": str(e),
            "temporal_analysis": temporal,
            "current_date": current_date_iso,
            "current_date_ist": current_date_ist,
        })
        return fallback


def _conflicting_evidence(claim: str, evidence_items: List) -> List:
    """Authoritative evidence items whose date or place contradicts the claim: [(evidence_id, conflicts)]."""
    found = []
    for e in (evidence_items or [])[:10]:
        if float(e.get("credibility_score", 0) or 0) < 0.7 or not e.get("specific_anchor_match", True):
            continue
        cs = evidence_conflicts(claim, e)
        if cs and e.get("evidence_id"):
            found.append((e["evidence_id"], cs))
    return found


def _check_debater_reply(obj: Dict[str, Any]) -> None:
    if not str(obj.get("position", "")).strip() or not isinstance(obj.get("arguments"), list) or not obj["arguments"]:
        raise ValueError("reply is missing 'position' or 'arguments' (probably cut off by the token limit)")


def _corroborating_evidence(specialist_results: Dict, evidence_items: List) -> List[str]:
    '''Authoritative, directly relevant evidence the specialists found supporting (and nothing refuting).'''
    ev = (specialist_results or {}).get("evidence_verification", {}) or {}
    stances = {a.get("evidence_id"): str(a.get("stance", "")).lower() for a in ev.get("evidence_assessments", [])}
    if any(s == "refutes" for s in stances.values()):
        return []
    out = []
    for e in (evidence_items or [])[:10]:
        if (float(e.get("credibility_score", 0) or 0) >= 0.7 and e.get("specific_anchor_match", True)
                and float(e.get("claim_relevance_score", 0) or 0) >= 0.7
                and stances.get(e.get("evidence_id")) == "supports"):
            out.append(e["evidence_id"])
    return out


def _debate_confidences(debate_history: List[Dict]) -> Dict[str, float]:
    """Last reported confidence of each debater, read from the transcript."""
    out: Dict[str, float] = {}
    for turn in debate_history or []:
        try:
            body = json.loads(turn.get("content") or "{}")
            if str(body.get("position", "")).strip() == "unavailable" or not _clamp(body.get("confidence", 0)):
                continue                      # a failed turn must not overwrite an earlier valid one
            out[str(turn.get("speaker"))] = _clamp(body.get("confidence", 0))
        except Exception:
            continue
    return out


def _classify_judge_error(exc: Exception) -> str:
    """Map a judge exception to an explicit status code."""
    import requests

    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "unavailable"
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "invalid_json"
    return "error"


def _generate_judge_reasoning(verdict: str, supporting_evidence: List[str],
                               contradicting_evidence: List[str], reasoning_text: str) -> str:
    """Generate structured reasoning for the judge verdict."""
    parts = []
    
    if verdict == "Insufficient":
        parts.append("Insufficient evidence to justify a verdict.")
    elif verdict in ("True", "False", "Misleading"):
        parts.append(f"The evidence {verdict.lower()} supports the claim.")
    
    if supporting_evidence:
        parts.append(f"Supporting evidence: {', '.join(supporting_evidence)}.")
    
    if contradicting_evidence:
        parts.append(f"Contradicting evidence: {', '.join(contradicting_evidence)}.")
    
    return " ".join(parts) if parts else ""


def _stance_consensus(evidence_items: List, specialist_results: Dict) -> Dict[str, Any]:
    '''Evidence the specialists judged to support / refute the claim, with each source's credibility.'''
    ev = (specialist_results or {}).get("evidence_verification", {}) or {}
    stances = {a.get("evidence_id"): str(a.get("stance", "")).lower()
               for a in ev.get("evidence_assessments", []) if a.get("evidence_id")}
    if not stances:
        return {}
    sup, ref = [], []
    for it in evidence_items or []:
        cred = float(it.get("credibility_score", 0.5) or 0.5)
        if stances.get(it.get("evidence_id")) == "supports":
            sup.append((it["evidence_id"], cred))
        elif stances.get(it.get("evidence_id")) == "refutes":
            ref.append((it["evidence_id"], cred))
    return {"supports": sup, "refutes": ref, "assessed": len(stances)}


def _verdict_from_stances(c: Dict[str, Any]) -> Dict[str, Any]:
    sup, ref = c["supports"], c["refutes"]
    sup_w, ref_w = sum(w for _, w in sup), sum(w for _, w in ref)
    authoritative = lambda lst: any(w >= 0.7 for _, w in lst)
    if ref and (not sup or ref_w >= 1.5 * sup_w):
        ids = [i for i, _ in ref][:3]
        return {"verdict": "False", "ids": ids, "supporter": 0.15, "skeptic": 0.85,
                "confidence": round(min(0.9, 0.5 + 0.1 * len(ref) + (0.1 if authoritative(ref) else 0.0)), CONFIDENCE_DECIMALS),
                "reasoning": f"The evidence specialists found {len(ref)} source(s) contradicting the claim and {len(sup)} supporting it ({', '.join(ids)})."}
    if sup and (not ref or sup_w >= 1.5 * ref_w) and (len(sup) >= 2 or authoritative(sup)):
        ids = [i for i, _ in sup][:3]
        return {"verdict": "True", "ids": ids, "supporter": 0.85, "skeptic": 0.15,
                "confidence": round(min(0.9, 0.5 + 0.1 * len(sup) + (0.1 if authoritative(sup) else 0.0)), CONFIDENCE_DECIMALS),
                "reasoning": f"The evidence specialists found {len(sup)} source(s) supporting the claim and {len(ref)} contradicting it ({', '.join(ids)})."}
    if sup and ref:
        ids = ([i for i, _ in sup][:2] + [i for i, _ in ref][:2])
        return {"verdict": "Misleading", "ids": ids, "supporter": 0.5, "skeptic": 0.5, "confidence": 0.6,
                "reasoning": f"The evidence is genuinely mixed: {len(sup)} source(s) support and {len(ref)} contradict the claim ({', '.join(ids)})."}
    return {"verdict": "Insufficient", "ids": [], "supporter": 0.3, "skeptic": 0.3, "confidence": 0.35,
            "reasoning": "The assessed evidence is neutral or too weak to support or contradict the claim."}


def _judge_fallback(evidence_items: List, valid_ids, claim: str, specialist_results: Dict = None) -> Dict[str, Any]:
    """Deterministic evidence-grounded consensus arbiter when Judge LLM call is unavailable.
    
    Rather than defaulting blindly to 'Insufficient', this computes consensus
    from calibrated evidence scores, source credibility, entity matching, and
    temporal-event consistency.
    """
    if not evidence_items:
        return {
            "agent": "judge",
            "final_verdict": "Insufficient",
            "confidence": 0.0,
            "supporter_final_confidence": 0.0,
            "skeptic_final_confidence": 0.0,
            "reasoning": "No relevant evidence items retrieved for this claim.",
            "key_evidence_cited": [],
            "key_kg_gaps_cited": [],
            "label_reasoning_alignment": {"verdict": "Insufficient", "hints": [], "aligned": True},
        }

    # Preferred path: use the specialists' per-evidence stances (they already include the date/place checks)
    consensus = _stance_consensus(evidence_items, specialist_results)
    if consensus and (consensus["supports"] or consensus["refutes"] or consensus["assessed"]):
        v = _verdict_from_stances(consensus)
        return {
            "agent": "judge", "final_verdict": v["verdict"], "confidence": v["confidence"],
            "supporter_final_confidence": v["supporter"], "skeptic_final_confidence": v["skeptic"],
            "reasoning": v["reasoning"], "key_evidence_cited": v["ids"], "key_kg_gaps_cited": [],
            "label_reasoning_alignment": {"verdict": v["verdict"], "hints": [v["verdict"]], "aligned": True,
                                          "adjusted_by": "stance_consensus"},
        }

    # No specialist stances available: fall back to the score thresholds below
    top_items = sorted(
        evidence_items,
        key=lambda x: (x.get("final_score", 0.0), x.get("claim_relevance_score", 0.0)),
        reverse=True
    )[:5]

    mean_rel = float(np.mean([x.get("claim_relevance_score", 0.0) for x in top_items])) if top_items else 0.0
    mean_cred = float(np.mean([x.get("credibility_score", 0.5) for x in top_items])) if top_items else 0.5
    mean_entity = float(np.mean([x.get("entity_match_score", 0.0) for x in top_items])) if top_items else 0.0
    mean_temporal = float(np.mean([x.get("temporal_match_score", 0.5) for x in top_items])) if top_items else 0.5
    
    # Check for strong debunk / contradiction signals
    refuting_signals = 0
    supporting_signals = 0
    for it in top_items:
        txt = (str(it.get("title", "")) + " " + str(it.get("snippet", ""))).lower()
        if any(w in txt for w in ("false", "fake", "hoax", "debunk", "misleading", "fabricated", "doctored", "untrue")):
            refuting_signals += 1
        elif it.get("temporal_match_score", 0.5) < 0.2 or evidence_conflicts(claim, it):
            refuting_signals += 1
        elif it.get("claim_relevance_score", 0) > 0.6 and it.get("entity_match_score", 0) > 0.35:
            supporting_signals += 1

    cited_ids = [it.get("evidence_id") for it in top_items if it.get("evidence_id") in valid_ids][:3]
    if not cited_ids and top_items:
        cited_ids = [it.get("evidence_id") for it in top_items[:2] if it.get("evidence_id")]

    # Calibrate verdict from evidence consensus
    if mean_rel < 0.30 or not cited_ids:
        verdict = "Insufficient"
        confidence = 0.25
        supporter_conf = 0.20
        skeptic_conf = 0.70
        reasoning = f"Available evidence lacks sufficient direct factual corroboration (mean relevance {mean_rel:.2f})."
    elif refuting_signals > supporting_signals and refuting_signals >= 1:
        verdict = "False"
        confidence = round(min(0.95, 0.50 + 0.35 * mean_cred), CONFIDENCE_DECIMALS)
        supporter_conf = 0.15
        skeptic_conf = 0.85
        reasoning = f"Evidence directly refutes or contradicts factual elements of the claim ({', '.join(cited_ids)})."
    elif supporting_signals >= 1 and mean_rel >= 0.50 and mean_temporal >= 0.40:
        verdict = "True"
        confidence = round(min(0.96, 0.45 + 0.35 * mean_rel + 0.20 * mean_cred), CONFIDENCE_DECIMALS)
        supporter_conf = 0.85
        skeptic_conf = 0.15
        reasoning = f"Claim is verified and corroborated across authoritative sources with matching entities and temporal markers ({', '.join(cited_ids)})."
    elif mean_rel >= 0.40:
        verdict = "Misleading"
        confidence = 0.65
        supporter_conf = 0.45
        skeptic_conf = 0.55
        reasoning = f"Evidence contains partial truths but lacks full temporal or contextual alignment with the claim ({', '.join(cited_ids)})."
    else:
        verdict = "Insufficient"
        confidence = 0.30
        supporter_conf = 0.30
        skeptic_conf = 0.60
        reasoning = "Evidence volume is inadequate to establish conclusive truth or falsehood."

    return {
        "agent": "judge",
        "final_verdict": verdict,
        "confidence": confidence,
        "supporter_final_confidence": supporter_conf,
        "skeptic_final_confidence": skeptic_conf,
        "reasoning": reasoning,
        "key_evidence_cited": cited_ids,
        "key_kg_gaps_cited": [],
        "label_reasoning_alignment": {"verdict": verdict, "hints": [verdict], "aligned": True},
    }
