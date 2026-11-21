"""VeriGraph AI — Adaptive KG expansion (sparse-graph targeted, evidence-grounded)."""
from __future__ import annotations
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List
import json
from services.ollama_service import call_llm, extract_json_object
from config.settings import (
    KG_EXPANSION_MAX_CANDIDATES,
    KG_EXPANSION_CONFIDENCE_THRESHOLD,
    OLLAMA_TIMEOUT,
)

# KG expansion generates a large candidate list, so it needs a much larger
# generation budget than the short specialist prompts.
KG_EXPANSION_TIMEOUT = max(OLLAMA_TIMEOUT, 300)

EXPANSION_SYSTEM = """KG expansion with evidence-grounded facts.
1. Every candidate must trace to at least one evidence item.
2. Do not invent values. Omit absent fields.
3. Validate dates/locations exactly as written in evidence.
4. Output JSON array: [{entity_type, name, properties, evidence_ids, candidate_confidence, source_quote}].
"""


def _c(x, n=160):
    s = str(x or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:n] if s.lower() != "nan" else ""


def generate_expansion_candidates(evidence_items: List[Dict], claim: str,
                                  kg_context: Dict[str, Any],
                                  max_candidates: int = None) -> Dict[str, Any]:
    max_candidates = max_candidates or KG_EXPANSION_MAX_CANDIDATES
    prompt_bundle = {
        "claim": _c(claim, 400),
        "existing_kg": {
            "entities": kg_context.get("entities", [])[:5],
            "events": kg_context.get("events", [])[:5],
            "locations": kg_context.get("locations", [])[:5],
            "persons": kg_context.get("persons", [])[:5],
        },
        "evidence": [
            {"id": e.get("evidence_id"), "title": _c(e.get("title"), 120),
             "snippet": _c(e.get("snippet", ""), 200),
             "url": _c(e.get("url", ""), 80), "date": _c(e.get("published_at", ""), 20)}
            for e in evidence_items[:5]
        ],
        "instructions": "Extract NEW entities with entity_type, name, properties, evidence_ids, "
                        "candidate_confidence, source evidence text quote. Skip anything already in existing_kg.",
    }

    def _empty(status: str, reason: str) -> Dict[str, Any]:
        return {"candidates": [], "candidates_generated": 0, "candidates_validated": 0,
                "candidates_rejected": 0, "candidates_promoted": 0,
                "llm_status": status, "reason": reason}

    candidates = []
    try:
        raw = call_llm(EXPANSION_SYSTEM, json.dumps(prompt_bundle, default=str)[:4000],
                       num_predict=int(__import__("os").getenv("KG_EXPANSION_NUM_PREDICT", "400")), timeout=KG_EXPANSION_TIMEOUT)
        err = raw.get("error")
        if err:
            status = ("timeout" if err == "timeout"
                      else "unavailable" if err in ("ConnectionError", "ConnectTimeout") else "error")
            return _empty(status, f"Ollama {err} (generation_duration={raw.get('generation_duration')}s)")
        data = extract_json_object(raw.get("content") or "")
        raw_candidates = data if isinstance(data, list) else data.get("candidates", [])
    except Exception as e:
        print(f"[kg_expansion] LLM call failed: {e}")
        return _empty("invalid_json", f"{type(e).__name__}: {e}")

    if not raw_candidates:
        return _empty("no_candidates",
                      f"The model found no new entities; reply started: {(raw.get('content') or '')[:160]!r}")

    # Process candidates
    processed = []
    for c in raw_candidates[:max_candidates]:
        if isinstance(c, dict):
            processed.append({
                "entity_type": c.get("entity_type", ""),
                "name": c.get("name", ""),
                "properties": c.get("properties", {}),
                "evidence_ids": c.get("evidence_ids", []),
                "candidate_confidence": c.get("candidate_confidence", 0.0),
                "source_quote": c.get("source_quote", ""),
            })

    return {
        "candidates": processed,
        "candidates_generated": len(raw_candidates),
        "candidates_validated": len(processed),
        "candidates_rejected": len(raw_candidates) - len(processed),
        "candidates_promoted": len(processed),
        "llm_status": "success",
        "reason": "",
    }


def _normalize_tokens(text: str):
    """Lowercase alphanumeric tokens of length >= 3, for grounding checks."""
    return {
        t for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(t) >= 3
    }


#: Minimum fraction of candidate-name tokens that must appear in the cited
#: evidence for the candidate to count as grounded.
GROUNDING_TOKEN_THRESHOLD = 0.5

#: Minimum BGE cosine similarity between candidate name and cited evidence.
GROUNDING_SIMILARITY_THRESHOLD = 0.35

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from",
    "was", "were", "are", "has", "have", "had", "its",
    "his", "her", "their", "our", "not", "but", "all",
    "new", "one", "two", "first", "last", "more", "some",
}


def _is_grounded(candidate_name: str, cited_evidence: List[Dict], claim: str) -> tuple:
    """Check that a candidate name is actually supported by its cited evidence.

    Validation is intentionally strict: a candidate that cites a real evidence
    id is still rejected when its name does not occur in that evidence and is
    semantically unrelated to it. Otherwise every candidate referencing any
    known evidence id would be marked validated.

    Returns (is_grounded: bool, reason: str, similarity: float).
    """
    name_tokens = _normalize_tokens(candidate_name) - _STOPWORDS
    if not name_tokens:
        return False, "empty_entity_name", 0.0

    evidence_text = " ".join(
        f"{e.get('title', '')} {e.get('snippet', '')}" for e in cited_evidence
    )
    evidence_tokens = _normalize_tokens(evidence_text)

    if not evidence_tokens:
        return False, "no_evidence_content", 0.0

    # Lexical grounding: candidate name tokens must appear in the cited evidence.
    overlap = name_tokens & evidence_tokens
    grounding_ratio = len(overlap) / len(name_tokens)
    if grounding_ratio < GROUNDING_TOKEN_THRESHOLD:
        return False, f"low_lexical_grounding({grounding_ratio:.2f})", 0.0

    # Semantic grounding via BGE, when available.
    similarity = 0.0
    try:
        import numpy as np
        from pipeline.embeddings import embed_text

        cand_emb = np.asarray(embed_text(candidate_name), dtype=np.float32)
        ev_emb = np.asarray(embed_text(evidence_text[:600]), dtype=np.float32)
        cand_norm = float(np.linalg.norm(cand_emb))
        ev_norm = float(np.linalg.norm(ev_emb))
        if cand_norm > 0 and ev_norm > 0:
            similarity = float(np.dot(cand_emb, ev_emb) / (cand_norm * ev_norm))
        if similarity < GROUNDING_SIMILARITY_THRESHOLD:
            return False, f"low_semantic_similarity({similarity:.3f})", similarity
    except Exception:
        # Embeddings unavailable: lexical grounding above is sufficient.
        pass

    return True, "grounded_in_evidence", similarity


def validate_candidates(cand_dicts, ev_dicts, claim: str = ""):
    """Validate generated candidates against evidence items.

    A candidate is validated only when it is structurally complete, cites at
    least one evidence id that exists, and its name is grounded in that cited
    evidence. Everything else is rejected with an explicit reason.

    Args:
        cand_dicts: List of candidate dicts from KG expansion
        ev_dicts: List of evidence item dicts
        claim: Optional claim text used for additional grounding context

    Returns:
        Dict with validated candidates and counts:
        - "candidates": list of ALL candidate dicts with validation_status set
        - "validated": list of grounded candidates
        - "rejected": list of rejected candidates with reasons
        - "validated_count" / "rejected_count"
    """
    validated: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []

    evidence_by_id = {
        e.get("evidence_id"): e
        for e in (ev_dicts or [])
        if isinstance(e, dict) and e.get("evidence_id")
    }

    for c in cand_dicts:
        if not isinstance(c, dict):
            malformed = {
                "candidate_id": str(c) if c else "None",
                "validation_status": "MALFORMED",
                "reason": "candidate is not an object",
            }
            all_candidates.append(malformed)
            rejected.append(malformed)
            continue

        required = ["entity_type", "name", "source_evidence_ids", "candidate_id"]
        missing = [k for k in required if k not in c]

        if missing:
            c["validation_status"] = "MALFORMED"
            c["reason"] = f"missing required fields: {','.join(missing)}"
            rejected.append(c)
            all_candidates.append(c)
            continue

        if not str(c.get("name", "")).strip():
            c["validation_status"] = "MALFORMED"
            c["reason"] = "empty entity name"
            rejected.append(c)
            all_candidates.append(c)
            continue

        cited_ids = [eid for eid in (c.get("source_evidence_ids") or []) if eid]
        cited_evidence = [evidence_by_id[eid] for eid in cited_ids if eid in evidence_by_id]

        if not cited_evidence:
            c["validation_status"] = "REJECTED_NO_EVIDENCE"
            c["reason"] = "no cited evidence id exists in the evidence package"
            rejected.append(c)
            all_candidates.append(c)
            continue

        grounded, reason, similarity = _is_grounded(
            c.get("name", ""), cited_evidence, claim or ""
        )
        c["grounding_similarity"] = round(float(similarity), 4)
        c["cited_evidence_count"] = int(len(cited_evidence))

        if grounded:
            c["validation_status"] = "VALIDATED"
            validated.append(c)
        else:
            c["validation_status"] = "REJECTED_NOT_GROUNDED"
            c["reason"] = reason
            rejected.append(c)

        all_candidates.append(c)

    return {
        "candidates": all_candidates,
        "validated": validated,
        "rejected": rejected,
        "validated_count": len(validated),
        "rejected_count": len(rejected),
    }

def run_kg_expansion(
    claim: str,
    narratives=None,
    evidence_items=None,
    verifier_result=None,
    case_id: str = "",
    kg_context=None,
):
    if __import__("os").getenv("VERIGRAPH_SKIP_KG_EXPANSION", "").strip().lower() in ("1", "true", "yes"):
        return {"status": "skipped", "llm_status": "skipped", "reason": "disabled by VERIGRAPH_SKIP_KG_EXPANSION",
                "candidates": [], "candidates_generated": 0, "candidates_validated": 0,
                "candidates_rejected": 0, "candidates_promoted": 0}
    """Public KG expansion entry point.

    Returns a structured result and never crashes the main
    verification pipeline because KG expansion is optional.
    """
    try:
        generated = generate_expansion_candidates(evidence_items or [], claim, kg_context or {}, KG_EXPANSION_MAX_CANDIDATES)
        candidates = [
            {"candidate_id": f"kg_{case_id or 'candidate'}_{index}",
             "entity_type": str(candidate.get("entity_type", "")),
             "name": str(candidate.get("name", "")),
             "properties": candidate.get("properties", {}),
             "source_evidence_ids": list(candidate.get("evidence_ids", [])),
             "source_case_id": case_id,
             "confidence": float(candidate.get("candidate_confidence", 0.0) or 0.0)}
            for index, candidate in enumerate(generated.get("candidates", []), start=1)
        ]
        validation = validate_candidates(candidates, evidence_items or [], claim=claim or "")

        return {
            'status': 'completed' if generated.get('llm_status', 'success') in ('success', 'no_candidates') else 'degraded',
            'llm_status': generated.get('llm_status', 'success'),
            'reason': generated.get('reason', ''),
            'candidates': validation.get('candidates', []),
            'validated': validation.get('validated', []),
            'rejected': validation.get('rejected', []),
            'candidates_generated': generated.get('candidates_generated', 0),
            'candidates_validated': validation.get('validated_count', 0),
            'candidates_rejected': validation.get('rejected_count', 0),
            'candidates_promoted': 0,
            'validated_count': validation.get('validated_count', 0),
            'rejected_count': validation.get('rejected_count', 0),
        }

    except Exception as exc:
        return {
            'status': 'failed',
            'error': str(exc),
            'candidates': [],
            'validated': [],
            'rejected': [],
            'validated_count': 0,
            'rejected_count': 0,
            'candidates_generated': 0, 'candidates_validated': 0,
            'candidates_rejected': 0, 'candidates_promoted': 0,
        }

